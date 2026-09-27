"""M0 diagnostic pilot: same masked target word, three information sources.

For COCO val2017 images: caption A contains a word for an object annotated in the image; that word is
masked. Conditions:
  IMG  : [576 image tokens] + prompt + caption A (word masked)
  TXT  : prompt containing another caption B of the same image + caption A (word masked)
  NONE : prompt + caption A (word masked)            -- language prior / leakage from caption A
Measures (see smm/diagnostics.py): exact logit contribution of known / discovered / epsilon, log p(target),
activation of the mapped COCO concept, and SIM symmetry-II violation w.r.t. the context (IMG, TXT).

--target_type (M0 v2) replaces the object word by a color / count / spatial / attribute word taken from the
caption (smm/coco_probe.py LEXICON); the mapped-concept check then does not apply (recorded as null).

Usage: python m0_diagnostics.py PROJECTOR_PT COCO_DIR COCO_MAP_JSON OUT_DIR [--n 200] [--n_sim 24]
                                [--target_type object|color|count|spatial|attribute]
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.coco_probe import CocoProbe  # noqa: E402
from smm.diagnostics import logit_contributions, sim_violation  # noqa: E402
from smm.steerling_io import load_steerling  # noqa: E402
from smm.vlm import VisionEncoder, build_connector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("projector")
ap.add_argument("coco_dir")
ap.add_argument("coco_map")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--n_sim", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--connector", default="mlp", help="mlp | resampler | resampler<N>")
ap.add_argument("--ks", default="8,32,128,256", help="nested K values for the SIM test")
ap.add_argument("--target_type", default="object", help="object | color | count | spatial | attribute")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
KS = tuple(int(k) for k in args.ks.split(","))
random.seed(args.seed)

lm, tok = load_steerling(attn="sdpa")
emb = lm.transformer.tok_emb
vision = VisionEncoder().cuda()
proj = build_connector(args.connector, vision.dim).cuda()
proj.load_state_dict(torch.load(args.projector, map_location="cpu"))
proj.eval()
cmap = json.load(open(args.coco_map))
probe = CocoProbe(tok, args.coco_dir, cmap, n=args.n, seed=args.seed, target_type=args.target_type)
samples = probe.samples
print(f"{len(samples)} samples", flush=True)


def build(s, cond):
    return probe.build(s, cond, emb, vision, proj)


def enc(x):
    return probe.enc(x)


records = []
t0 = time.time()
for si, s in enumerate(samples):
    rec = {k: s[k] for k in ("image_id", "category", "word", "capA", "capB", "b_has_word")}
    rec["target_type"] = args.target_type
    mapped = cmap[s["category"]]["primary"] if args.target_type == "object" else None
    rec["mapped_concept"] = mapped
    for cond in ("IMG", "TXT", "NONE"):
        parts, ctx, pos, tg = build(s, cond)
        with torch.no_grad():
            r = logit_contributions(lm, torch.cat(parts, 1), pos, tg)
        ids0, w0 = r["known_topk_ids"][0], r["known_topk_w"][0]
        r["mapped_in_top32"] = (mapped in ids0) if mapped is not None else None
        r["mapped_weight"] = (w0[ids0.index(mapped)] if mapped in ids0 else 0.0) if mapped is not None else None
        del r["known_topk_ids"], r["known_topk_w"]
        if si < args.n_sim and ctx is not None:
            m = len(enc(s["capB"])) if cond == "IMG" else None  # size-matched to the caption context
            r["sim"] = sim_violation(lm, parts, ctx, pos[0], tg[0], ks=KS, matched_m=m)
        rec[cond] = r
    records.append(rec)
    if si % 10 == 0:
        print(f"{si + 1}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

json.dump(records, open(os.path.join(args.out, "m0_pilot_records.json"), "w"), indent=1)
print("DONE", flush=True)
