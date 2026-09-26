import torch, sys
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
mm = lambda b, h, q, k: q // 64 >= k // 64
for T in [int(x) for x in sys.argv[1].split(",")]:
    torch._dynamo.reset()
    f = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    q = torch.randn(1, 32, T, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, T, 128, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
    bm = create_block_mask(mm, None, None, T, T, device="cuda")
    for mode in ["infer", "train"]:
        try:
            if mode == "infer":
                with torch.inference_mode():
                    f(q, k, v, block_mask=bm, enable_gqa=True)
            else:
                qq, kk, vv = [t.clone().requires_grad_() for t in (q, k, v)]
                f(qq, kk, vv, block_mask=bm, enable_gqa=True).sum().backward()
            print(T, mode, "ok", flush=True)
        except Exception as e:
            print(T, mode, "FAIL", str(e)[:90].replace("\n", " "), flush=True)
