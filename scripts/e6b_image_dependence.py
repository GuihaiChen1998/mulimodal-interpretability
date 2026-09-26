"""E6b: does the trained M0 projector carry *image-specific* information into Steerling?

Eval MDM loss (t=1, first answer block, deterministic) with matched vs shuffled image features,
on training samples and on held-out samples. Matched < shuffled => the visual path is used.

Usage: python e6b_image_dependence.py TRAIN_JSON HELDOUT_JSON PROJECTOR_PT OUT_JSON
"""

import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.mdm import build_mdm_batch, mdm_loss  # noqa: E402
from smm.steerling_io import answer_ids, chat_prompt_ids, load_steerling  # noqa: E402
from smm.vlm import N_IMG_TOKENS, MLPProjector, VisionEncoder  # noqa: E402

train_json, held_json, proj_pt, out_json = sys.argv[1:5]
lm, tok = load_steerling()
mask_id = tok.convert_tokens_to_ids("<|mask|>")
vision = VisionEncoder().cuda()
proj = MLPProjector(vision.dim).cuda()
proj.load_state_dict(torch.load(proj_pt))
proj.eval()


def load(path, n, exclude=()):
    rows = [r for r in json.load(open(path)) if r["id"] not in exclude][:n]
    feats = []
    for i in range(0, len(rows), 64):
        ims = [Image.open(r["image"]).convert("RGB") for r in rows[i: i + 64]]
        feats.append(vision(vision.processor(images=ims, return_tensors="pt")["pixel_values"].cuda()))
    return rows, torch.cat(feats)


@torch.no_grad()
def eval_loss(rows, feats, perm, bs=8):
    P = [chat_prompt_ids(tok, r["instruction"]) for r in rows]
    A = [answer_ids(tok, r["caption"]) for r in rows]
    ls = []
    for i in range(0, len(rows), bs):
        idx = list(range(i, min(i + bs, len(rows))))
        txt, lab, w = build_mdm_batch([P[j] for j in idx], [A[j] for j in idx], n_prefix=N_IMG_TOKENS,
                                      mask_id=mask_id, t_fixed=1.0, block_idx="first")
        img = proj(feats[[perm[j] for j in idx]].float()).to(torch.bfloat16)
        x = torch.cat([img, lm.transformer.tok_emb(txt.cuda())], 1)
        logits, _ = lm(None, input_embeds=x, minimal_output=True)
        ls.append(mdm_loss(logits, lab.cuda(), w.cuda()).item() * len(idx))
    return sum(ls) / len(rows)


res = {}
train_rows, train_feats = load(train_json, 256)
held_rows, held_feats = load(held_json, 256, exclude={r["id"] for r in json.load(open(train_json))})
g = torch.Generator().manual_seed(0)
for name, rows, feats in [("train", train_rows, train_feats), ("heldout", held_rows, held_feats)]:
    ident = list(range(len(rows)))
    shuf = torch.randperm(len(rows), generator=g).tolist()
    m, s = eval_loss(rows, feats, ident), eval_loss(rows, feats, shuf)
    res[name] = {"n": len(rows), "loss_matched": m, "loss_shuffled": s, "gap": s - m}
    print(name, res[name], flush=True)
json.dump(res, open(out_json, "w"), indent=2)
