"""E7b: V_vis and COCO->concept mapping from what Steerling itself activates on image captions.

Read-out protocol: Steerling is a masked-diffusion LM, trained to predict only MASKED positions, so
concepts are read at masked positions ("which concepts does the model use to predict this token").
(A first version read concepts at clean positions; those were dominated by code/orthographic concepts,
kept as e7b_clean_readout_summary.json for the record.)

- V_vis(caption): mask 30% of caption tokens (2 random passes per caption); known concepts in the top-32
  (sigmoid weight >= W_MIN) at masked positions, in at least MIN_DOCS captions (COCO val2017 + LLaVA).
  This is the same signal M2's text->visual concept distillation will use as its target.
- COCO mapping: for each category, captions that mention it; mask the category word's tokens and
  aggregate concepts at those positions over captions (context disambiguates polysemy).

Usage: python e7b_caption_concepts.py COCO_CAPTIONS_JSON COCO_INSTANCES_JSON LLAVA_POOL_JSON E7_DIR
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import load_steerling  # noqa: E402

cap_json, inst_json, pool_json, out_dir = sys.argv[1:5]
W_MIN, MIN_DOCS, PER_CAT = 0.1, 5, 60
random.seed(0)

zpath = os.path.join(os.path.dirname(__file__), "..", "datasets", "known_concepts(1).zip")
with zipfile.ZipFile(zpath) as z:
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1", newline="")))
N = len(rows)

model, tok = load_steerling(attn="sdpa")
bos = tok.convert_tokens_to_ids("<|bos|>")
mask_id = tok.convert_tokens_to_ids("<|mask|>")


@torch.no_grad()
def concepts(batch_ids):
    """Per token: top-32 known concept ids and sigmoid weights. batch_ids: list of lists."""
    L = max(map(len, batch_ids))
    x = torch.full((len(batch_ids), L), mask_id, dtype=torch.long)
    for i, s in enumerate(batch_ids):
        x[i, : len(s)] = torch.tensor(s)
    _, o = model(x.cuda(), minimal_output=False)
    return o.known_topk_indices.cpu(), torch.sigmoid(o.known_topk_logits.float()).cpu()


def encode(text):
    return [bos] + tok.encode(text, add_special_tokens=False)


# ---- V_vis from captions ---------------------------------------------------------------------
coco_caps = [a["caption"].strip() for a in json.load(open(cap_json))["annotations"]]
llava_caps = [r["caption"] for r in json.load(open(pool_json))]
caps = random.sample(coco_caps, 10000) + llava_caps
df = torch.zeros(N)  # document frequency
g = torch.Generator().manual_seed(0)
for i in range(0, len(caps), 64):
    base = [encode(c)[:128] for c in caps[i: i + 64]]
    seen = [set() for _ in base]
    for _ in range(2):
        ids, masked = [], []
        for s in base:
            m = torch.rand(len(s), generator=g) < 0.3
            m[0] = False  # keep BOS
            ids.append([mask_id if mm else t for t, mm in zip(s, m.tolist(), strict=True)])
            masked.append(torch.nonzero(m).squeeze(1))
        idx, w = concepts(ids)
        for j, pos in enumerate(masked):
            sel = idx[j, pos][w[j, pos] >= W_MIN]
            seen[j].update(sel[sel < N].tolist())
    for st in seen:
        if st:
            df[list(st)] += 1
n_docs = len(caps)
ubiq = df / n_docs > 0.5  # fires on most captions: generic / stylistic rather than visual content
vvis_cap = [i for i in range(N) if df[i] >= MIN_DOCS and not ubiq[i]]

# ---- COCO category -> concept via token-level activations ------------------------------------
cats = json.load(open(inst_json))["categories"]
mapping = {}
for c in cats:
    name = c["name"]
    pat = re.compile(r"\b" + re.escape(name) + r"(s|es)?\b", re.I)
    hits = [s for s in coco_caps if pat.search(s)]
    random.shuffle(hits)
    hits = hits[:PER_CAT]
    agg, present = collections.Counter(), collections.Counter()
    for k in range(0, len(hits), 32):
        chunk = hits[k: k + 32]
        ids, spans = [], []
        for s in chunk:
            m = pat.search(s)
            a = len(encode(s[: m.start()].rstrip()))  # first token of the word (tiktoken folds the space)
            b = max(min(len(encode(s[: m.end()])), 128), a + 1)
            e = encode(s)[:128]
            e[a:b] = [mask_id] * (b - a)
            ids.append(e)
            spans.append((a, b))
        idx, w = concepts(ids)
        for j, (a, b) in enumerate(spans):
            seen = set()
            for p in range(a, max(b, a + 1)):
                for cid, wt in zip(idx[j, p].tolist(), w[j, p].tolist(), strict=True):
                    if cid < N and wt >= W_MIN:
                        agg[cid] += wt
                        seen.add(cid)
            present.update(seen)
    # rank by specificity: P(active | this category's masked word) * idf over all caption masked positions,
    # so hub concepts active almost everywhere (e.g. 'Animal Species') do not win by frequency alone
    n_h = max(len(hits), 1)
    spec = {cid: (present[cid] / n_h) * float(torch.log(torch.tensor(n_docs / (df[cid].item() + 1))))
            for cid in present if not ubiq[cid]}
    top = sorted(spec.items(), key=lambda kv: -kv[1])[:5]
    hub = agg.most_common(3)
    mapping[name] = {
        "coco_id": c["id"], "supercategory": c["supercategory"], "n_captions": len(hits),
        "primary": top[0][0] if top else None,
        "primary_name": rows[top[0][0]]["concept_name"] if top else None,
        "primary_consistency": present[top[0][0]] / len(hits) if top else 0.0,
        "top5": [{"id": cid, "name": rows[cid]["concept_name"], "specificity": round(v, 3),
                  "frac_captions": round(present[cid] / n_h, 3), "doc_freq": int(df[cid]),
                  "in_vvis_caption": cid in set(vvis_cap)} for cid, v in top],
        "top3_by_raw_weight": [{"id": cid, "name": rows[cid]["concept_name"], "weight_sum": round(v, 2)}
                               for cid, v in hub],
    }

vset = set(vvis_cap)
e7 = json.load(open(os.path.join(out_dir, "v_vis.json")))
clip_set = set(e7["concept_ids"])
summary = {
    "n_captions": n_docs, "w_min": W_MIN, "min_docs": MIN_DOCS,
    "n_concepts_ever_active": int((df > 0).sum()),
    "n_ubiquitous(>50% captions)": int(ubiq.sum()),
    "ubiquitous_examples": [rows[i]["concept_name"] for i in torch.nonzero(ubiq).squeeze(1).tolist()[:20]],
    "n_vvis_caption": len(vvis_cap),
    "n_vvis_caption_steerable": sum(rows[i]["is_steerable"] == "TRUE" for i in vvis_cap),
    "overlap_with_clip_vvis": len(vset & clip_set),
    "most_frequent_vvis": [(rows[i]["concept_name"], int(df[i])) for i in sorted(vvis_cap, key=lambda i: -df[i])[:30]],
    "coco_mapped": sum(m["primary"] is not None for m in mapping.values()),
    "coco_primary_consistency_mean": sum(m["primary_consistency"] for m in mapping.values()) / len(mapping),
    "coco_primary": {k: [m["primary"], m["primary_name"], round(m["primary_consistency"], 2), m["n_captions"]]
                     for k, m in mapping.items()},
}
json.dump({"w_min": W_MIN, "min_docs": MIN_DOCS, "concept_ids": vvis_cap,
           "doc_freq": {str(i): int(df[i]) for i in range(N) if df[i] > 0}},
          open(os.path.join(out_dir, "v_vis_caption.json"), "w"))
json.dump(mapping, open(os.path.join(out_dir, "coco2concept_caption.json"), "w"), indent=1)
json.dump(summary, open(os.path.join(out_dir, "e7b_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
