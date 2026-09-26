"""E2b: relative error of known + unk_hat + eps vs hidden, in bf16 (as the model runs) and recomputed in fp32."""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import chat_prompt_ids, load_steerling  # noqa: E402

model, tok = load_steerling(attn="sdpa")
texts = ["Describe this scene: a red bus passes a clock tower in the rain.",
         "Explain photosynthesis to a child.", "Write a haiku about winter mountains."]
out = []
for t in texts:
    ids = torch.tensor([chat_prompt_ids(tok, t)], device="cuda")
    with torch.no_grad():
        logits, o = model(ids)
    h = o.hidden.float()
    hn = h.norm(dim=-1)
    diag = {"n_pos": h.shape[1], "n_nan_hidden_rows": int(h.isnan().any(-1).sum()), "n_zero_norm_rows": int((hn == 0).sum()),
            "n_nan_composed_rows": int(o.composed.float().isnan().any(-1).sum())}
    ok = (hn > 0) & ~h.isnan().any(-1)
    h = h[ok]
    rel_bf16 = ((o.composed.float()[ok] - h).norm(dim=-1) / h.norm(dim=-1)).max().item()
    kf, uh = o.known_features.float()[ok], o.unk_hat.float()[ok]
    rel_fp32 = ((kf + uh + (h - kf - uh) - h).norm(dim=-1) / h.norm(dim=-1)).max().item()
    ulp = (h.abs().max() * 2**-8).item()
    out.append({**diag, "seq_len": ids.shape[1], "max_rel_err_composed_bf16": rel_bf16, "max_abs_hidden": h.abs().max().item(),
                "bf16_ulp_at_max": ulp, "max_rel_err_fp32_recompute": rel_fp32,
                "logits_from_composed_vs_hidden_argmax_agree":
                    (model.transformer.lm_head(o.hidden).argmax(-1) == logits.argmax(-1)).float().mean().item()})
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
