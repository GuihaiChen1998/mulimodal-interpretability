"""E7c: merged COCO category -> Steerling known-concept map (v1, for human review).

Per category, a candidate pool is scored with three independent signals:
  spec  - Steerling masked read-out specificity (E7b protocol, recomputed over all concepts so that
          every candidate has a value): P(active at the masked category word) * idf(doc freq over
          15K captions). Uses COCO synonyms when the category name itself is rare in captions.
  clip  - CLIP text similarity between "a photo of a {category}" and "a photo of {concept name}".
  lex   - the concept's name/group contains the word (name_hit) and its embedding promotes the word
          through the LM head (token_hit).
score = 0.5*spec_n + 0.3*clip_n + 0.1*name_hit + 0.1*token_hit   (spec_n, clip_n min-max within pool)
confidence = "high" if the winner is also the top candidate under spec AND under clip, or its margin
over the runner-up is >= 0.15; otherwise "review".

Usage: python e7c_merge_coco_map.py COCO_CAPTIONS COCO_INSTANCES E7_DIR
"""

import collections
import csv
import io
import json
import os
import random
import re
import sys
import zipfile

import torch
from safetensors import safe_open
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import load_steerling  # noqa: E402

cap_json, inst_json, e7_dir = sys.argv[1:4]
W_MIN, PER_CAT = 0.1, 80
random.seed(0)
SYNONYMS = {  # extra caption phrasings for COCO categories whose label is rare in captions
    "handbag": ["purse", "handbag"], "hair drier": ["hair dryer", "blow dryer", "hair drier"],
    "sports ball": ["soccer ball", "tennis ball", "baseball", "football", "basketball", "ball"],
    "potted plant": ["potted plant", "plant", "house plant"], "baseball glove": ["baseball glove", "mitt", "glove"],
    "dining table": ["dining table", "table"], "tv": ["tv", "television"], "cell phone": ["cell phone", "phone"],
    "couch": ["couch", "sofa"], "motorcycle": ["motorcycle", "motorbike"], "airplane": ["airplane", "plane", "jet"],
    "wine glass": ["wine glass", "wine glasses"], "teddy bear": ["teddy bear", "stuffed animal"],
    "remote": ["remote", "remote control", "controller"], "laptop": ["laptop", "computer"],
    "bicycle": ["bicycle", "bike"], "donut": ["donut", "doughnut"], "fire hydrant": ["fire hydrant", "hydrant"],
}

zpath = os.path.join(os.path.dirname(__file__), "..", "datasets", "known_concepts(1).zip")
with zipfile.ZipFile(zpath) as z:
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1", newline="")))
N = len(rows)
vcap = json.load(open(os.path.join(e7_dir, "v_vis_caption.json")))
df = torch.zeros(N)
for k, v in vcap["doc_freq"].items():
    df[int(k)] = v
N_DOCS = 15000
idf = torch.log(N_DOCS / (df + 1))
ubiq = df / N_DOCS > 0.5
vvis = set(vcap["concept_ids"])
e7a = json.load(open(os.path.join(e7_dir, "coco2concept.json")))

model, tok = load_steerling(attn="sdpa")
bos = tok.convert_tokens_to_ids("<|bos|>")
mask_id = tok.convert_tokens_to_ids("<|mask|>")
enc = lambda t: [bos] + tok.encode(t, add_special_tokens=False)  # noqa: E731

# LM-head top tokens per concept (for token_hit)
E = model.known_head.concept_embedding.weight[:N].float()
E = E / E.norm(dim=-1, keepdim=True)
W = model.transformer.lm_head.weight.float()
top_tok = []
for i in range(0, N, 2048):
    for row in (E[i: i + 2048] @ W.T).topk(20, dim=1).indices.cpu():
        top_tok.append({re.sub(r"[^a-z]", "", tok.decode([int(t)]).lower()) for t in row})
del E, W

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", dtype=torch.float16).cuda().eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


@torch.no_grad()
def temb(texts, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        t = proc.tokenizer(texts[i: i + bs], padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        e = clip.get_text_features(**t).float()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


Tc = temb([f"a photo of {r['concept_name'].lower()}" for r in rows])


@torch.no_grad()
def readout(seqs):
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), mask_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    _, o = model(x.cuda(), minimal_output=False)
    return o.known_topk_indices.cpu(), torch.sigmoid(o.known_topk_logits.float()).cpu()


coco_caps = [a["caption"].strip() for a in json.load(open(cap_json))["annotations"]]
cats = json.load(open(inst_json))["categories"]
result = {}
for c in cats:
    name = c["name"]
    phrases = SYNONYMS.get(name, [name])
    pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r")(s|es)?\b", re.I)
    hits = [s for s in coco_caps if pat.search(s)]
    random.shuffle(hits)
    hits = hits[:PER_CAT]
    present = collections.Counter()
    for k in range(0, len(hits), 32):
        seqs, spans = [], []
        for s in hits[k: k + 32]:
            m = pat.search(s)
            a = len(enc(s[: m.start()].rstrip()))
            b = max(min(len(enc(s[: m.end()])), 128), a + 1)
            e = enc(s)[:128]
            e[a:b] = [mask_id] * (b - a)
            seqs.append(e)
            spans.append((a, b))
        idx, w = readout(seqs)
        for j, (a, b) in enumerate(spans):
            sel = idx[j, a:b][w[j, a:b] >= W_MIN]
            present.update(set(sel[sel < N].tolist()))
    n_h = max(len(hits), 1)
    spec = {cid: present[cid] / n_h * idf[cid].item() for cid in present if not ubiq[cid]}
    clip_sim = (temb([f"a photo of a {name}"]) @ Tc.T)[0].cpu()
    words = {p.lower() for p in phrases} | {p.lower() + "s" for p in phrases}
    words |= {w for p in phrases for w in p.lower().split() if len(w) > 3}
    name_pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")s?\b", re.I)

    pool = set(sorted(spec, key=lambda k: -spec[k])[:10])
    pool |= set(clip_sim.topk(10).indices.tolist())
    pool |= {d["id"] for d in e7a[name]["candidates"][:10]}
    cands = []
    for cid in pool:
        nm = rows[cid]["concept_name"] + " " + rows[cid]["group_name"]
        cands.append({"id": cid, "name": rows[cid]["concept_name"], "spec": spec.get(cid, 0.0),
                      "frac": round(present[cid] / n_h, 3), "clip": clip_sim[cid].item(),
                      "name_hit": bool(name_pat.search(nm)), "token_hit": bool(top_tok[cid] & words),
                      "steerable": rows[cid]["is_steerable"] == "TRUE", "in_vvis": cid in vvis})
    smax = max(d["spec"] for d in cands) or 1.0
    cmin, cmax = min(d["clip"] for d in cands), max(d["clip"] for d in cands)
    for d in cands:
        d["score"] = (0.5 * d["spec"] / smax + 0.3 * (d["clip"] - cmin) / max(cmax - cmin, 1e-6)
                      + 0.1 * d["name_hit"] + 0.1 * d["token_hit"])
        d["spec"], d["clip"], d["score"] = round(d["spec"], 3), round(d["clip"], 4), round(d["score"], 3)
    cands.sort(key=lambda d: -d["score"])
    best = cands[0]
    top_spec = max(cands, key=lambda d: d["spec"])["id"]
    top_clip = max(cands, key=lambda d: d["clip"])["id"]
    margin = best["score"] - cands[1]["score"] if len(cands) > 1 else 1.0
    conf = "high" if (best["id"] == top_spec == top_clip) or margin >= 0.15 else "review"
    result[name] = {"coco_id": c["id"], "supercategory": c["supercategory"], "phrases": phrases,
                    "n_captions": len(hits), "primary": best["id"], "primary_name": best["name"],
                    "confidence": conf, "margin": round(margin, 3),
                    "top_by_spec": top_spec, "top_by_clip": top_clip, "candidates": cands[:6]}
    print(f"{name:15s} -> {best['name'][:45]:45s} [{conf}] n={len(hits)}", flush=True)

json.dump(result, open(os.path.join(e7_dir, "coco2concept_v1.json"), "w"), indent=1)
print("high:", sum(r["confidence"] == "high" for r in result.values()), "/ 80")
