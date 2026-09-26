"""E7: build the visual concept subset V_vis and the COCO category -> concept-id mapping.

Signals per concept (all 33,732 known concepts):
  1. rule filter  - drop code / LaTeX / markup / math-notation / pure-linguistic concepts by keyword
  2. visualness   - CLIP ViT-L/14-336: text "a photo of {concept_name}" vs a pool of real images
                    (LLaVA-Pretrain subset). score = mean(top-10 sims) - mean(all sims)
  3. model tokens - the vocabulary tokens each concept embedding promotes through Steerling's LM head
                    (used for COCO matching; checks the concept *does* what its name says)

COCO mapping per category: candidates = concepts whose name / group / promoted tokens contain the
category word, ranked by CLIP text-text similarity; plus a pure CLIP-similarity fallback.

Usage: python e7_build_vvis.py IMAGE_POOL_JSON COCO_INSTANCES_JSON OUT_DIR
"""

import csv
import io
import json
import os
import re
import sys
import zipfile

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import MODEL_ID  # noqa: E402

pool_json, coco_json, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
dev = "cuda"

zpath = os.path.join(os.path.dirname(__file__), "..", "datasets", "known_concepts(1).zip")
with zipfile.ZipFile(zpath) as z:
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1", newline="")))
N = len(rows)

# ---- 1. rule filter ----------------------------------------------------------------------
NONVIS = re.compile(
    r"\b(latex|tex|code|coding|programming|python|java|javascript|c\+\+|html|css|xml|json|sql|regex|syntax|"
    r"api|compiler|markdown|scripting|software|debug\w*|"
    r"math(ematical)? notation|equations?|formulas?|punctuation|suffix(es)?|prefix(es)?|affix(es)?|"
    r"grammar|grammatical|verbs?|nouns?|adjectives?|adverbs?|pronouns?|prepositions?|conjunctions?|"
    r"morpholog\w*|lexical|spelling|capitaliz\w*|abbreviations?|acronyms?|subwords?|single letters?|"
    r"letter [a-z]|numerals?|digits?|number formats?|date formats?|unicode|encoding|whitespace|newlines?)\b",
    re.I)


def rule_ok(r):
    txt = r["concept_name"] + " | " + r["group_name"]
    return NONVIS.search(txt) is None


rule_pass = [rule_ok(r) for r in rows]

# ---- 2. CLIP visualness -------------------------------------------------------------------
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16).to(dev).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


@torch.no_grad()
def text_emb(texts, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        t = proc.tokenizer(texts[i: i + bs], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
        e = clip.get_text_features(**t).float()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


@torch.no_grad()
def image_emb(paths, bs=128):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i: i + bs]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, torch.float16)
        e = clip.get_image_features(pixel_values=px).float()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


pool = [r["image"] for r in json.load(open(pool_json))]
I = image_emb(pool)
names = [r["concept_name"] for r in rows]
Tc = text_emb([f"a photo of {n.lower()}" for n in names])
S = Tc @ I.T  # [N, n_img]
vis = (S.topk(10, dim=1).values.mean(1) - S.mean(1)).cpu()

# reference distributions: visual controls (COCO category names + scenes/attributes) vs
# non-visual controls (abstract / procedural phrases). Threshold = score keeping 90% of visual controls.
coco_names = [c["name"] for c in json.load(open(coco_json))["categories"]]
ctrl_vis = coco_names + ["mountain", "beach", "kitchen", "forest", "city street", "sunset", "snow", "river",
                         "red", "wooden table", "crowd", "bridge", "garden", "castle", "airplane cockpit",
                         "waterfall", "desert", "classroom", "stadium", "farm"]
ctrl_non = ["python syntax", "latex equation", "verb conjugation", "tax law", "abstract algebra", "semicolon usage",
            "database schema", "philosophy of mind", "contract negotiation", "inflation rate", "moral obligation",
            "statistical significance", "legal liability", "software licensing", "political ideology", "grammar rules",
            "memory allocation", "probability theory", "customer satisfaction", "time complexity", "insurance policy",
            "voting rights", "logical fallacy", "interest rates", "copyright infringement", "cognitive bias",
            "regular expressions", "API documentation", "quantum entanglement", "social contract", "monetary policy",
            "password hashing", "epistemology", "tense and aspect", "supply chain management", "network protocol",
            "burden of proof", "fiscal year", "user authentication", "algorithmic complexity"]


def ctrl_score(ws):
    s = text_emb([f"a photo of {w}" for w in ws]) @ I.T
    return (s.topk(10, dim=1).values.mean(1) - s.mean(1)).cpu()


cv, cn = ctrl_score(ctrl_vis), ctrl_score(ctrl_non)
thr = float(torch.quantile(cv, 0.10))
auroc = float((cv[:, None] > cn[None, :]).float().mean())
in_vvis = [rule_pass[i] and vis[i].item() >= thr for i in range(N)]
vvis = [i for i in range(N) if in_vvis[i]]

# ---- 3. concept -> promoted tokens through Steerling's LM head -----------------------------
from huggingface_hub import snapshot_download  # noqa: E402

snap = snapshot_download(MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
with safe_open(os.path.join(snap, "model.safetensors"), "pt", device=dev) as f:
    keys = list(f.keys())
    k_emb = [k for k in keys if k.startswith("known_head") and "embedding" in k and "weight" in k]
    k_lm = [k for k in keys if k.endswith("tok_emb.weight") or k.endswith("lm_head.weight")]
    E = f.get_tensor(k_emb[0]).float()[:N]
    W = f.get_tensor(k_lm[0]).float()
E = E / E.norm(dim=-1, keepdim=True)
top_tok = []
for i in range(0, N, 2048):
    idx = (E[i: i + 2048] @ W.T).topk(20, dim=1).indices.cpu()
    for row in idx:
        top_tok.append([re.sub(r"[^a-z]", "", tok.decode([int(t)]).lower()) for t in row])

# ---- 4. COCO mapping ------------------------------------------------------------------------
cats = json.load(open(coco_json))["categories"]
cat_names = [c["name"] for c in cats]
Tcat = text_emb([f"a photo of a {c}" for c in cat_names])
sim_cat = (Tcat @ Tc.T).cpu()  # [80, N]


def word_forms(c):
    w = c.lower()
    forms = {w, w + "s", w + "es"}
    if w.endswith("y"):
        forms.add(w[:-1] + "ies")
    if " " in w:
        forms |= {w.replace(" ", "")}
        forms |= {p for p in w.split() if len(p) > 3}
    return forms


mapping = {}
tiers = {"A_name+token": 0, "B_token_only": 0, "C_name_only": 0, "D_clip_only": 0}
for ci, c in enumerate(cat_names):
    forms = word_forms(c)
    full = re.compile(r"\b(" + "|".join(re.escape(f) for f in forms if " " not in f or f == c) + r")\b", re.I) \
        if " " not in c else re.compile(r"\b" + re.escape(c) + r"s?\b", re.I)
    cand_ids = set(sim_cat[ci].topk(20).indices.tolist())
    for i in range(N):
        if any(t in forms for t in top_tok[i]) or full.search(rows[i]["concept_name"] + " " + rows[i]["group_name"]):
            cand_ids.add(i)
    cands = []
    for i in cand_ids:
        name_hit = bool(full.search(rows[i]["concept_name"] + " " + rows[i]["group_name"]))
        tok_hit = any(t in forms for t in top_tok[i])
        score = sim_cat[ci, i].item() + 0.05 * name_hit + 0.05 * tok_hit
        cands.append({"id": i, "name": rows[i]["concept_name"], "name_hit": name_hit, "token_hit": tok_hit,
                      "clip_sim": round(sim_cat[ci, i].item(), 4), "score": round(score, 4),
                      "in_vvis": in_vvis[i], "steerable": rows[i]["is_steerable"] == "TRUE",
                      "top_tokens": top_tok[i][:8]})
    cands.sort(key=lambda d: d["score"], reverse=True)
    best = cands[0]
    tier = ("A_name+token" if best["name_hit"] and best["token_hit"] else "B_token_only" if best["token_hit"]
            else "C_name_only" if best["name_hit"] else "D_clip_only")
    tiers[tier] += 1
    mapping[c] = {"coco_id": cats[ci]["id"], "supercategory": cats[ci]["supercategory"], "primary": best["id"],
                  "primary_name": best["name"], "tier": tier, "candidates": cands[:10]}

summary = {
    "n_concepts": N,
    "rule_pass": int(sum(rule_pass)),
    "visualness_threshold(10th pct of visual controls)": thr,
    "visualness_auroc_controls": auroc,
    "nonvisual_controls_above_threshold": float((cn >= thr).float().mean()),
    "control_visual_score_mean": float(cv.mean()), "control_nonvisual_score_mean": float(cn.mean()),
    "n_vvis": len(vvis),
    "n_vvis_steerable": sum(rows[i]["is_steerable"] == "TRUE" for i in vvis),
    "image_pool_size": len(pool),
    "coco_categories": len(cat_names),
    "coco_primary_tiers": tiers,
    "coco_primary": {c: [m["primary"], m["primary_name"], m["tier"]] for c, m in mapping.items()},
    "top_visual_examples": [names[i] for i in vis.topk(25).indices.tolist()],
    "rejected_by_clip_examples": [names[i] for i in range(N) if rule_pass[i] and not in_vvis[i]][:25:],
    "rejected_by_rule_examples": [names[i] for i in range(N) if not rule_pass[i]][:25],
}
json.dump({"threshold": thr, "concept_ids": vvis,
           "scores": {str(i): round(vis[i].item(), 4) for i in range(N)}},
          open(os.path.join(out_dir, "v_vis.json"), "w"))
json.dump(mapping, open(os.path.join(out_dir, "coco2concept.json"), "w"), indent=1)
json.dump(summary, open(os.path.join(out_dir, "e7_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
