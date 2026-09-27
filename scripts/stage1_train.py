"""Stage 1: train the vision connector (frozen Steerling + frozen CLIP) with the block-wise MDM loss.

Evaluation every --eval_every optimizer steps:
  - held-out LLaVA: MDM loss (t=1, first answer block) with matched vs shuffled images
  - COCO masked-object probe: log p(object word) with the image (IMG) vs no context (NONE);
    IMG > NONE is the M0 prerequisite ("the image actually carries information")
Checkpoints (connector + optimizer + step) are written at every evaluation; --resume continues.

Usage: python stage1_train.py TRAIN_JSON HELDOUT_JSON OUT_DIR --connector mlp|resampler
         --coco_dir /workspace/data/coco --coco_map results/week1/e7/coco2concept_v1_reviewed.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.coco_probe import CocoProbe, summarize_logp  # noqa: E402
from smm.mdm import build_mdm_batch, mdm_loss, mdm_loss_fast  # noqa: E402
from smm.steerling_io import answer_ids, chat_prompt_ids, load_steerling  # noqa: E402
from smm.vlm import VisionEncoder, build_connector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("train")
ap.add_argument("heldout")
ap.add_argument("out")
ap.add_argument("--connector", default="mlp")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--eval_every", type=int, default=200)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--coco_dir", required=True)
ap.add_argument("--coco_map", required=True)
ap.add_argument("--n_probe", type=int, default=100)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--full_forward", action="store_true", help="use the full interpretable forward (slow reference)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)

lm, tok = load_steerling()  # attention mode from STEERLING_USE_FLEX_ATTN
mask_id = tok.convert_tokens_to_ids("<|mask|>")
emb = lm.transformer.tok_emb
vision = VisionEncoder().cuda()
proj = build_connector(args.connector, vision.dim).cuda()
NP = proj.n_tokens
print(f"connector={args.connector} n_prefix={NP} params={sum(p.numel() for p in proj.parameters()):,}", flush=True)


class LlavaDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = Image.open(r["image"]).convert("RGB")
        except Exception:
            return None
        pv = vision.processor(images=[im], return_tensors="pt")["pixel_values"][0]
        return pv, chat_prompt_ids(tok, r["instruction"]), answer_ids(tok, r["caption"])


def collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


train_rows = json.load(open(args.train))
train_ids = {r["id"] for r in train_rows}
held_rows = [r for r in json.load(open(args.heldout)) if r["id"] not in train_ids][:128]
loader = DataLoader(LlavaDS(train_rows), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)


def step_loss(pv, P, A, **kw):
    txt, lab, w = build_mdm_batch(P, A, n_prefix=NP, mask_id=mask_id, **kw)
    with torch.no_grad():
        feats = vision(pv.cuda(non_blocking=True))
    img = proj(feats.float()).to(torch.bfloat16)
    x = torch.cat([img, emb(txt.cuda())], 1)
    if args.full_forward:
        logits, _ = lm(None, input_embeds=x, minimal_output=True)
        return mdm_loss(logits, lab.cuda(), w.cuda())
    return mdm_loss_fast(lm, x, lab.cuda(), w.cuda())


# ---- evaluation assets ----------------------------------------------------------------------------
held_pv = torch.stack([vision.processor(images=[Image.open(r["image"]).convert("RGB")], return_tensors="pt")
                       ["pixel_values"][0] for r in held_rows])
held_P = [chat_prompt_ids(tok, r["instruction"]) for r in held_rows]
held_A = [answer_ids(tok, r["caption"]) for r in held_rows]
probe = CocoProbe(tok, args.coco_dir, json.load(open(args.coco_map)), n=args.n_probe, seed=1)
proj.eval()
none_lp = probe.per_sample_logp(lm, emb, vision, proj, conds=("NONE",), pad_to_block=True)["NONE"]
proj.train()


@torch.no_grad()
def evaluate():
    proj.eval()
    g = torch.Generator().manual_seed(123)
    perm = torch.randperm(len(held_rows), generator=g).tolist()
    res = {}
    for name, order in (("matched", list(range(len(held_rows)))), ("shuffled", perm)):
        ls = []
        for i in range(0, len(held_rows), args.bs):
            idx = list(range(i, min(i + args.bs, len(held_rows))))
            ls.append(step_loss(held_pv[[order[j] for j in idx]], [held_P[j] for j in idx], [held_A[j] for j in idx],
                                t_fixed=1.0, block_idx="first").item() * len(idx))
        res[f"heldout_loss_{name}"] = sum(ls) / len(held_rows)
    res["heldout_gap"] = res["heldout_loss_shuffled"] - res["heldout_loss_matched"]
    img_lp = probe.per_sample_logp(lm, emb, vision, proj, conds=("IMG",), pad_to_block=True)["IMG"]
    res["coco"] = summarize_logp({"IMG": img_lp, "NONE": none_lp})
    proj.train()
    return res


# ---- optimisation -----------------------------------------------------------------------------------
opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.0)
total = len(loader) // args.accum * args.epochs
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
step, start_epoch = 0, 0
ck = os.path.join(args.out, "ckpt_last.pt")
if args.resume and os.path.exists(ck):
    st = torch.load(ck)
    proj.load_state_dict(st["proj"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
    step, start_epoch = st["step"], st["epoch"]
    print(f"resumed at step {step}", flush=True)
log = open(os.path.join(args.out, "train_log.jsonl"), "a")
evals = open(os.path.join(args.out, "eval_log.jsonl"), "a")
if step == 0:
    e = {"step": 0, **evaluate()}
    evals.write(json.dumps(e) + "\n"), evals.flush()
    print(json.dumps(e), flush=True)

t0, micro, run_loss = time.time(), 0, 0.0
skip = step * args.accum  # micro-batches already consumed in the resumed epoch (approximate resume)
for ep in range(start_epoch, args.epochs):
    for bi, (pv, P, A) in enumerate(loader):
        if skip:
            skip -= 1
            continue
        loss = step_loss(pv, P, A) / args.accum
        loss.backward()
        run_loss += loss.item()
        micro += 1
        if micro % args.accum:
            continue
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1
        log.write(json.dumps({"step": step, "loss": run_loss, "lr": sched.get_last_lr()[0]}) + "\n")
        run_loss = 0.0
        if step % 20 == 0:
            log.flush()
            el = time.time() - t0
            print(f"step {step}/{total} {el / 60:.1f}min eta {(total - step) * el / max(micro // args.accum, 1) / 60:.0f}min",
                  flush=True)
        if step % args.eval_every == 0 or step == total:
            e = {"step": step, **evaluate()}
            evals.write(json.dumps(e) + "\n"), evals.flush()
            print(json.dumps(e), flush=True)
            torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "epoch": ep}, ck)
        if step >= total:
            break

torch.save(proj.state_dict(), os.path.join(args.out, f"connector_stage1_{args.connector}.pt"))
print("DONE", flush=True)
