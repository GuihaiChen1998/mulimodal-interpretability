"""E4 (vision path wiring) + E6 (1K overfit) with the E5 block-wise MDM loss.

Frozen steerling-8b-instruct + frozen CLIP ViT-L/14-336 + trainable 2-layer MLP projector (M0).
CLIP features are precomputed once (the encoder is frozen).

Usage: python e4_e6_overfit.py DATA_JSON OUT_DIR [--epochs 10] [--bs 8] [--lr 1e-3]
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.generate import generate_from_embeds  # noqa: E402
from smm.mdm import build_mdm_batch, mdm_loss  # noqa: E402
from smm.steerling_io import answer_ids, chat_prompt_ids, load_steerling  # noqa: E402
from smm.vlm import N_IMG_TOKENS, MLPProjector, SteerlingVLM, VisionEncoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--n_eval_gen", type=int, default=16)
ap.add_argument("--n_samples", type=int, default=None, help="use only the first N samples")
ap.add_argument("--const_lr", action="store_true", help="constant LR after warmup (no cosine decay)")
ap.add_argument("--eval_every", type=int, default=5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)

lm, tok = load_steerling()
mask_id = tok.convert_tokens_to_ids("<|mask|>")
stops = [tok.convert_tokens_to_ids(t) for t in ("<|endofchunk|>", "<|eot_id|>", "<|endoftext|>")]
banned = [mask_id, tok.convert_tokens_to_ids("<|pad|>")]
vision = VisionEncoder().cuda()
proj = MLPProjector(vision.dim).cuda()  # fp32 master weights
vlm = SteerlingVLM(lm, vision, proj)

rows = json.load(open(args.data))[: args.n_samples]
prompts = [chat_prompt_ids(tok, r["instruction"]) for r in rows]
answers = [answer_ids(tok, r["caption"]) for r in rows]

# ---- precompute CLIP features --------------------------------------------------------------
t0 = time.time()
feats = []
for i in range(0, len(rows), 64):
    ims = [Image.open(r["image"]).convert("RGB") for r in rows[i: i + 64]]
    pv = vision.processor(images=ims, return_tensors="pt")["pixel_values"].cuda()
    feats.append(vision(pv))
feats = torch.cat(feats)  # [N, 576, 1024] bf16
clip_s = time.time() - t0

# ---- E4 checks -----------------------------------------------------------------------------
report = {"n_samples": len(rows), "clip_feature_shape": list(feats.shape), "clip_precompute_s": round(clip_s, 1)}
trainable = [n for n, p in vlm.named_parameters() if p.requires_grad]
report["e4_trainable_params"] = sum(p.numel() for p in proj.parameters())
report["e4_only_projector_trainable"] = all(n.startswith("projector.") for n in trainable)
report["e4_steerling_requires_grad_any"] = any(p.requires_grad for p in lm.parameters())


def forward_batch(idx, **kw):
    txt, labels, w = build_mdm_batch([prompts[i] for i in idx], [answers[i] for i in idx],
                                     n_prefix=N_IMG_TOKENS, mask_id=mask_id, **kw)
    img = proj(feats[idx].float()).to(torch.bfloat16)
    x = torch.cat([img, lm.transformer.tok_emb(txt.cuda())], 1)
    logits, _ = lm(None, input_embeds=x, minimal_output=True)
    return mdm_loss(logits, labels.cuda(), w.cuda()), x.shape[1]


torch.cuda.reset_peak_memory_stats()
loss, T = forward_batch(list(range(args.bs)))
loss.backward()
gn = {n: p.grad.norm().item() for n, p in proj.named_parameters()}
report["e4_first_loss"] = loss.item()
report["e4_first_seq_len"] = T
report["e4_projector_grad_norms"] = gn
report["e4_pass"] = (report["e4_only_projector_trainable"] and not report["e4_steerling_requires_grad_any"]
                     and all(v > 0 and math.isfinite(v) for v in gn.values()))
report["peak_mem_fwd_bwd_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
proj.zero_grad(set_to_none=True)
print(json.dumps(report, indent=2), flush=True)

# ---- E6: overfit ---------------------------------------------------------------------------
opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(rows) / args.bs)
total = steps_per_epoch * args.epochs
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (1.0 if args.const_lr else 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
eval_idx = list(range(min(args.n_eval_gen, len(rows))))
log = open(os.path.join(args.out, "train_log.jsonl"), "w")


def eval_loss():
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        n = min(256, len(rows))
        ls = [forward_batch(list(range(i, min(i + args.bs, n))), t_fixed=1.0, block_idx="first", gen=g)[0].item()
              for i in range(0, n, args.bs)]
    return sum(ls) / len(ls)


def token_f1(a, b):
    a, b = a.lower().split(), b.lower().split()
    if not a or not b:
        return 0.0
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def eval_gen():
    outs = []
    for i in eval_idx:
        with torch.no_grad():
            img = proj(feats[i: i + 1].float()).to(torch.bfloat16)
            pre = torch.cat([img, lm.transformer.tok_emb(torch.tensor([prompts[i]], device="cuda"))], 1)
            ids = generate_from_embeds(lm, pre, mask_id=mask_id, stop_ids=stops, banned_ids=banned,
                                       max_new_tokens=64, steps_per_block=16)
        text = tok.decode(ids, skip_special_tokens=True).strip()
        outs.append({"ref": rows[i]["caption"], "gen": text, "f1": token_f1(text, rows[i]["caption"])})
    return outs


history = []
el0 = eval_loss()
g0 = eval_gen()
history.append({"epoch": 0, "eval_loss_t1": el0, "gen_f1": sum(o["f1"] for o in g0) / len(g0)})
print(json.dumps(history[-1]), flush=True)
step, t0 = 0, time.time()
for ep in range(1, args.epochs + 1):
    perm = torch.randperm(len(rows)).tolist()
    for i in range(0, len(rows), args.bs):
        loss, T = forward_batch(perm[i: i + args.bs])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.write(json.dumps({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0], "T": T}) + "\n")
        if step % 25 == 0:
            log.flush()
            print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {(time.time() - t0) / step:.2f}s/step",
                  flush=True)
    gens = eval_gen() if ep in (1, args.epochs) or ep % args.eval_every == 0 else None
    h = {"epoch": ep, "eval_loss_t1": eval_loss()}
    if gens:
        h["gen_f1"] = sum(o["f1"] for o in gens) / len(gens)
        json.dump(gens, open(os.path.join(args.out, f"gens_ep{ep}.json"), "w"), indent=1, ensure_ascii=False)
    history.append(h)
    print(json.dumps(h), flush=True)

report["history"] = history
report["sec_per_step"] = round((time.time() - t0) / step, 3)
report["peak_mem_train_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
json.dump(g0, open(os.path.join(args.out, "gens_ep0.json"), "w"), indent=1, ensure_ascii=False)
torch.save(proj.state_dict(), os.path.join(args.out, "projector_m0.pt"))
json.dump(report, open(os.path.join(args.out, "e4_e6_report.json"), "w"), indent=2)
print("DONE")
