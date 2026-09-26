"""Greedy confidence-ordered unmasking with an embedding prefix (image + prompt).

Generation proceeds over attention-aligned 64-token blocks (the first one may be partial,
starting right after the prompt), matching the block-wise MDM training in mdm.py.
"""

from __future__ import annotations

import torch

BLOCK = 64


@torch.no_grad()
def generate_from_embeds(model, prefix_embeds: torch.Tensor, *, mask_id: int, stop_ids: list[int],
                         banned_ids: list[int], max_new_tokens: int = 128, steps_per_block: int = 16) -> list[int]:
    emb = model.transformer.tok_emb
    P = prefix_embeds.shape[1]
    out: list[int] = []
    end = P
    while len(out) < max_new_tokens:
        blk_end = (end // BLOCK + 1) * BLOCK
        n = blk_end - end
        cur = torch.full((1, n), mask_id, device=prefix_embeds.device, dtype=torch.long)
        committed = torch.zeros(n, dtype=torch.bool, device=cur.device)
        per_step = [n // steps_per_block + (1 if i < n % steps_per_block else 0) for i in range(steps_per_block)]
        prev = torch.tensor([out], device=cur.device, dtype=torch.long) if out else None
        for k in per_step:
            if k == 0:
                continue
            parts = [prefix_embeds] + ([emb(prev)] if prev is not None else []) + [emb(cur)]
            logits, _ = model(None, input_embeds=torch.cat(parts, 1), minimal_output=True)
            lg = logits[0, -n:].float()
            lg[:, banned_ids] = float("-inf")
            conf, tokv = lg.softmax(-1).max(-1)
            conf[committed] = -1
            pick = conf.topk(min(k, int((~committed).sum()))).indices
            cur[0, pick] = tokv[pick]
            committed[pick] = True
        block_tokens = cur[0].tolist()
        out.extend(block_tokens)
        end = blk_end
        hits = [out.index(t) for t in stop_ids if t in out]
        if hits:
            return out[: min(hits)]
    return out[:max_new_tokens]
