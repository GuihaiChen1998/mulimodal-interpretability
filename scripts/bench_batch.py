"""Throughput / memory benchmark for Stage-1 training configurations (random images, real models).

Usage: python bench_batch.py OUT_JSON   (set STEERLING_USE_FLEX_ATTN=1)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.mdm import build_mdm_batch, mdm_loss_fast  # noqa: E402
from smm.steerling_io import answer_ids, chat_prompt_ids, enable_block_checkpointing, load_steerling  # noqa: E402
from smm.vlm import VisionEncoder, build_connector  # noqa: E402

lm, tok = load_steerling()
mask_id = tok.convert_tokens_to_ids("<|mask|>")
vision = VisionEncoder().cuda()
P = chat_prompt_ids(tok, "Render a clear and concise summary of the photo.")
A = answer_ids(tok, "a red double decker bus driving past a clock tower on a rainy afternoon in the city")
# all no-checkpointing configs first: checkpointing is patched in once and cannot be undone in-process
CONFIGS = [("mlp", 8, False), ("mlp", 12, False), ("resampler", 8, False), ("resampler", 32, False),
           ("resampler", 64, False), ("mlp", 16, True), ("mlp", 32, True), ("resampler", 64, True),
           ("resampler", 128, True)]
res, ckpt_on = [], False
for conn, B, ck in CONFIGS:
    if ck and not ckpt_on:
        enable_block_checkpointing(lm)
        ckpt_on = True
    torch.manual_seed(0)
    proj = build_connector(conn, vision.dim).cuda()
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-4)
    pv = torch.randn(B, 3, 336, 336, device="cuda")
    txt, lab, w = build_mdm_batch([P] * B, [A] * B, n_prefix=proj.n_tokens, mask_id=mask_id, t_fixed=0.5)

    def step(proj=proj, opt=opt, pv=pv, txt=txt, lab=lab, w=w):
        with torch.no_grad():
            f = vision(pv)
        x = torch.cat([proj(f.float()).to(torch.bfloat16), lm.transformer.tok_emb(txt.cuda())], 1)
        mdm_loss_fast(lm, x, lab.cuda(), w.cuda()).backward()
        opt.step(), opt.zero_grad(set_to_none=True)

    rec = {"connector": conn, "B": B, "checkpointing": ck, "seq_len": lab.shape[1]}
    try:
        torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(4):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 4
        rec.update(s_per_step=round(dt, 3), ms_per_sample=round(1000 * dt / B, 1),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
    except torch.OutOfMemoryError:
        rec["oom"] = True
    del proj, opt
    torch.cuda.empty_cache()
    res.append(rec)
    print(json.dumps(rec), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=2)
