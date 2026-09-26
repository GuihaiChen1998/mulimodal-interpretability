"""M0 diagnostic pilot: same masked target word, three information sources.

For COCO val2017 images: caption A contains a word for an object annotated in the image; that word is
masked. Conditions:
  IMG  : [576 image tokens] + prompt + caption A (word masked)
  TXT  : prompt containing another caption B of the same image + caption A (word masked)
  NONE : prompt + caption A (word masked)            -- language prior / leakage from caption A
Measures (see smm/diagnostics.py): exact logit contribution of known / discovered / epsilon, log p(target),
activation of the mapped COCO concept, and SIM symmetry-II violation w.r.t. the context (IMG, TXT).

Usage: python m0_diagnostics.py PROJECTOR_PT COCO_DIR COCO_MAP_JSON OUT_DIR [--n 200] [--n_sim 24]
"""

import argparse
import json
import os
import random
import re
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.diagnostics import logit_contributions, sim_violation  # noqa: E402
from smm.steerling_io import answer_ids, load_steerling  # noqa: E402
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
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
random.seed(args.seed)

INSTR = "Give a brief description of the image."
lm, tok = load_steerling(attn="sdpa")
mask_id = tok.convert_tokens_to_ids("<|mask|>")
emb = lm.transformer.tok_emb
vision = VisionEncoder().cuda()
proj = build_connector(args.connector, vision.dim).cuda()
proj.load_state_dict(torch.load(args.projector))
proj.eval()
cmap = json.load(open(args.coco_map))

ann = json.load(open(os.path.join(args.coco_dir, "annotations", "instances_val2017.json")))
caps = json.load(open(os.path.join(args.coco_dir, "annotations", "captions_val2017.json")))
catname = {c["id"]: c["name"] for c in ann["categories"]}
img_cats: dict[int, set] = {}
for a in ann["annotations"]:
    img_cats.setdefault(a["image_id"], set()).add(catname[a["category_id"]])
img_caps: dict[int, list] = {}
for c in caps["annotations"]:
    img_caps.setdefault(c["image_id"], []).append(c["caption"].strip())
img_file = {i["id"]: i["file_name"] for i in ann["images"]}


def chat_text(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def enc(s):
    return tok.encode(s, add_special_tokens=False)


# ---- build samples -----------------------------------------------------------------------------
samples = []
ids = list(img_caps)
random.shuffle(ids)
for iid in ids:
    cands = [c for c in img_cats.get(iid, ()) if c in cmap]
    random.shuffle(cands)
    done = False
    for cat in cands:
        phrases = cmap[cat].get("phrases", [cat])
        pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r")\b", re.I)
        for ci, capA in enumerate(img_caps[iid]):
            m = pat.search(capA)
            if not m:
                continue
            others = [c for j, c in enumerate(img_caps[iid]) if j != ci]
            capB = others[0]
            ans = answer_ids(tok, capA)
            a, b = len(enc(capA[: m.start()].rstrip())), len(enc(capA[: m.end()]))
            if b <= a or m.group(0).lower() not in tok.decode(ans[a:b]).lower():
                continue
            samples.append({"image_id": iid, "file": img_file[iid], "category": cat, "word": m.group(0),
                            "capA": capA, "capB": capB, "ans": ans, "span": (a, b)})
            done = True
            break
        if done:
            break
    if len(samples) >= args.n:
        break
print(f"{len(samples)} samples", flush=True)


def build(s, cond):
    """Returns (embedding parts, context part index, absolute target positions, target ids)."""
    a, b = s["span"]
    ans = list(s["ans"])
    targets = ans[a:b]
    ans[a:b] = [mask_id] * (b - a)
    ans_e = emb(torch.tensor([ans], device="cuda"))
    if cond == "TXT":
        t = chat_text(INSTR + "\nReference description: " + s["capB"])
        i = t.index(s["capB"])
        pre, cap, post = enc(t[:i]), enc(t[i: i + len(s["capB"])]), enc(t[i + len(s["capB"]):])
        parts = [emb(torch.tensor([pre], device="cuda")), emb(torch.tensor([cap], device="cuda")),
                 emb(torch.tensor([post], device="cuda")), ans_e]
        ctx = 1
    else:
        parts = [emb(torch.tensor([enc(chat_text(INSTR))], device="cuda")), ans_e]
        ctx = None
        if cond == "IMG":
            im = Image.open(os.path.join(args.coco_dir, "val2017", s["file"])).convert("RGB")
            pv = vision.processor(images=[im], return_tensors="pt")["pixel_values"].cuda()
            with torch.no_grad():
                img = proj(vision(pv).float()).to(torch.bfloat16)
            parts = [img] + parts
            ctx = 0
    off = sum(p.shape[1] for p in parts[:-1])
    return parts, ctx, [off + a + j for j in range(b - a)], targets


records = []
t0 = time.time()
for si, s in enumerate(samples):
    rec = {k: s[k] for k in ("image_id", "category", "word", "capA", "capB")}
    mapped = cmap[s["category"]]["primary"]
    rec["mapped_concept"] = mapped
    for cond in ("IMG", "TXT", "NONE"):
        parts, ctx, pos, tg = build(s, cond)
        with torch.no_grad():
            r = logit_contributions(lm, torch.cat(parts, 1), pos, tg)
        ids0, w0 = r["known_topk_ids"][0], r["known_topk_w"][0]
        r["mapped_in_top32"] = mapped in ids0
        r["mapped_weight"] = w0[ids0.index(mapped)] if mapped in ids0 else 0.0
        del r["known_topk_ids"], r["known_topk_w"]
        if si < args.n_sim and ctx is not None:
            r["sim"] = sim_violation(lm, parts, ctx, pos[0], tg[0])
        rec[cond] = r
    records.append(rec)
    if si % 10 == 0:
        print(f"{si + 1}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

json.dump(records, open(os.path.join(args.out, "m0_pilot_records.json"), "w"), indent=1)
print("DONE", flush=True)
