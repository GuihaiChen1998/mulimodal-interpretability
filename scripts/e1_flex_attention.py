"""E1: flex_attention (block-causal, 64-token blocks, GQA) vs dense reference on this GPU.

Checks forward and backward, several sequence lengths (incl. non-multiples of 128),
and times flex vs SDPA-with-dense-mask (Steerling's default fallback).
"""

import json
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

BLOCK = 64
H, KV, D = 32, 4, 128
dev = "cuda"
flex = torch.compile(flex_attention, fullgraph=True)


def mask_mod(b, h, q, kv):
    return q // BLOCK >= kv // BLOCK


def dense_mask(T):
    i = torch.arange(T, device=dev)
    return (i[:, None] // BLOCK) >= (i[None, :] // BLOCK)


def reference(q, k, v):
    # fp32 dense attention with explicit GQA repeat
    k = k.repeat_interleave(H // KV, dim=1).float()
    v = v.repeat_interleave(H // KV, dim=1).float()
    s = (q.float() @ k.transpose(-1, -2)) / D**0.5
    s = s.masked_fill(~dense_mask(q.shape[2]), float("-inf"))
    return (s.softmax(-1) @ v).to(q.dtype)


def sdpa(q, k, v):
    m = torch.zeros(q.shape[2], q.shape[2], device=dev, dtype=q.dtype)
    m.masked_fill_(~dense_mask(q.shape[2]), float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)


def bench(fn, *a, n=20):
    for _ in range(3):
        fn(*a)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


results = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}
ok = True
for B, T in [(2, 640), (2, 704), (1, 1024), (1, 4096)]:
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, KV, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    bm = create_block_mask(mask_mod, None, None, T, T, device=dev)

    y_flex = flex(q, k, v, block_mask=bm, enable_gqa=True)
    y_ref = reference(q, k, v)
    y_sdpa = sdpa(q, k, v)
    g = torch.randn_like(y_ref)
    gf = torch.autograd.grad(y_flex, (q, k, v), g)
    gr = torch.autograd.grad(y_ref, (q, k, v), g)

    case = {
        "B": B, "T": T,
        "fwd_maxabs_flex_vs_ref": (y_flex - y_ref).abs().max().item(),
        "fwd_maxabs_sdpa_vs_ref": (y_sdpa - y_ref).abs().max().item(),
        "bwd_maxrel_flex_vs_ref": max(((a - b).float().norm() / b.float().norm()).item() for a, b in zip(gf, gr)),
        "ms_flex_fwd": bench(lambda: flex(q, k, v, block_mask=bm, enable_gqa=True)),
        "ms_sdpa_fwd": bench(lambda: sdpa(q, k, v)),
    }
    case["pass"] = case["fwd_maxabs_flex_vs_ref"] < 1e-2 and case["bwd_maxrel_flex_vs_ref"] < 1e-2
    ok &= case["pass"]
    results["cases"].append(case)
    print(json.dumps(case), flush=True)

results["pass"] = ok
results["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
out = sys.argv[1] if len(sys.argv) > 1 else "e1_flex_attention.json"
json.dump(results, open(out, "w"), indent=2)
print("E1", "PASS" if ok else "FAIL")
