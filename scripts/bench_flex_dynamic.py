"""flex_attention compiled with automatic dynamic shapes vs static per-length compilation,
on the variable sequence lengths seen in training (fwd+bwd through Steerling, B=8)."""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import load_steerling, use_static_flex  # noqa: E402

mode = sys.argv[1]  # "dynamic" | "static"
lm, tok = load_steerling(attn="flex")
if mode == "static":
    use_static_flex(lm)
emb = lm.transformer.tok_emb
res = {}
for T in [640, 704, 768, 640, 704, 768]:
    x = torch.randn(8, T, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ts = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        lm.transformer(None, input_embeds=x, return_hidden=True).float().pow(2).mean().backward()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    res.setdefault(T, []).append(round(min(ts), 3))
    print(mode, T, [round(t, 2) for t in ts], flush=True)
json.dump(res, open(f"results/week1/bench_flex_{mode}.json", "w"))
