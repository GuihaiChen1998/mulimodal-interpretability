"""E2: load steerling-8b-instruct, generate, verify the known/unknown/epsilon decomposition,
and check that CSV concept_id == known-head row index.

Usage: python e2_load_decompose.py OUT_DIR   (set STEERLING_USE_FLEX_ATTN=0/1 outside)
"""

import csv
import io
import json
import os
import re
import sys
import time
import zipfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import chat_prompt_ids, load_steerling  # noqa: E402
from steerling import GenerationConfig, SteerlingGenerator  # noqa: E402

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
mode = "flex" if os.environ.get("STEERLING_USE_FLEX_ATTN") == "1" else "sdpa"
res = {"attention": mode}

t0 = time.time()
model, tok = load_steerling()
res["load_s"] = round(time.time() - t0, 1)
res["mem_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
res["chat_template_example"] = tok.apply_chat_template(
    [{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True)

# ---- decomposition on a fixed prompt ------------------------------------------------------
text = ("A red double-decker bus drives past a clock tower on a rainy afternoon in London, "
        "while pedestrians with umbrellas wait at the crossing.")
ids = torch.tensor([chat_prompt_ids(tok, "Describe this scene: " + text)], device="cuda")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    logits, o = model(ids)
res["seq_len"] = ids.shape[1]
res["peak_mem_fwd_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
h = o.hidden.float()
comp = o.composed.float()
kf, uh, eps = o.known_features.float(), o.unk_hat.float(), o.epsilon.float()
hn = h.norm(dim=-1)
res["decomp"] = {
    "max_abs_composed_minus_hidden": (comp - h).abs().max().item(),
    "max_abs_known+unkhat+eps_minus_hidden": (kf + uh + eps - h).abs().max().item(),
    "mean_rel_norm_known": (kf.norm(dim=-1) / hn).mean().item(),
    "mean_rel_norm_unk_hat": (uh.norm(dim=-1) / hn).mean().item(),
    "mean_rel_norm_epsilon": (eps.norm(dim=-1) / hn).mean().item(),
    "known_topk_shape": list(o.known_topk_indices.shape) if o.known_topk_indices is not None else None,
}
torch.save(logits[0].float().cpu(), os.path.join(out_dir, f"e2_logits_{mode}.pt"))

# ---- concept id alignment vs CSV ----------------------------------------------------------
gen = SteerlingGenerator.from_model(model, tok, device="cuda")
zpath = os.path.join(os.path.dirname(__file__), "..", "datasets", "known_concepts(1).zip")
with zipfile.ZipFile(zpath) as z:
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1", newline="")))
res["n_known_rows_model"] = int(model.known_head.n_concepts)
res["n_csv_rows"] = len(rows)
g = torch.Generator().manual_seed(0)
sample = [int(i) for i in torch.randint(0, len(rows), (40,), generator=g)]


def hit_rate(cid, off=0):
    j = (cid + off) % len(rows)
    desc = (rows[j]["concept_name"] + " " + rows[j]["concept_description"]).lower()
    toks = [t.strip().lower() for t, _ in gen.concept_top_tokens(cid, k=15)]
    toks = [re.sub(r"[^a-z0-9]", "", t) for t in toks]
    toks = [t for t in toks if len(t) >= 3]
    return sum(t in desc for t in toks) / max(len(toks), 1)


res["concept_alignment"] = {
    "mean_hit_rate_same_id": sum(hit_rate(c) for c in sample) / len(sample),
    "mean_hit_rate_shifted_id(+1)": sum(hit_rate(c, 1) for c in sample) / len(sample),
    "examples": [{"id": c, "name": rows[c]["concept_name"],
                  "top_tokens": [t for t, _ in gen.concept_top_tokens(c, k=10)]} for c in sample[:5]],
}

# ---- generation sanity --------------------------------------------------------------------
eot = tok.convert_tokens_to_ids("<|eot_id|>")
gens = []
for q in ["What is the capital of France? Answer in one sentence.",
          "List three colors of a rainbow."]:
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    out = gen.generate(prompt, GenerationConfig(max_new_tokens=64, steps=64, seed=0, stop_tokens=[eot]))
    gens.append({"q": q, "a": out, "s": round(time.time() - t0, 1)})
res["generations"] = gens

json.dump(res, open(os.path.join(out_dir, f"e2_{mode}.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
