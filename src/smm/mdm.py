"""Masked-diffusion (MDM) training loss for a block-causal diffusion LM with a visual prefix.

Why block-wise and not LLaDA-V's "mask the whole answer at rate t":
Steerling attends block-causally (bidirectional inside a 64-token block, causal across blocks) and
generates block by block, so when a block is being denoised every earlier block is already clean.
We mirror that at training time: per sample, pick one answer block b, keep earlier answer tokens
clean, mask answer tokens inside b with probability t ~ U(t_min, 1], and drop everything after b.
Loss = sum_{masked} CE / t, normalised by the number of answer tokens in b (LLaDA-style estimator).
Positions in block b after the answer ends are filled with <|mask|> and carry no loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BLOCK = 64


def build_mdm_batch(prompts: list[list[int]], answers: list[list[int]], *, n_prefix: int, mask_id: int,
                    t_min: float = 1e-3, t_fixed: float | None = None, block_idx: str = "random",
                    gen: torch.Generator | None = None):
    """Returns text_ids [B, L-n_prefix], labels [B, L] (-100 = no loss), weights [B, L] (1/t / n_b)."""
    assert n_prefix % BLOCK == 0
    rows = []
    for p, a in zip(prompts, answers, strict=True):
        text = p + a
        a0, a1 = n_prefix + len(p), n_prefix + len(text)  # absolute answer span
        blocks = list(range(a0 // BLOCK, (a1 - 1) // BLOCK + 1))
        if block_idx == "first":
            b = blocks[0]
        else:
            b = blocks[int(torch.randint(len(blocks), (1,), generator=gen))]
        L = (b + 1) * BLOCK
        ids = torch.full((L - n_prefix,), mask_id, dtype=torch.long)
        n_keep = min(len(text), L - n_prefix)
        ids[:n_keep] = torch.tensor(text[:n_keep])
        s, e = max(a0, b * BLOCK), min(a1, L)  # answer positions inside block b
        t = t_fixed if t_fixed is not None else float(t_min + (1 - t_min) * torch.rand(1, generator=gen))
        m = torch.rand(e - s, generator=gen) < t
        if not m.any():
            m[int(torch.randint(e - s, (1,), generator=gen))] = True
        labels = torch.full((L,), -100, dtype=torch.long)
        weights = torch.zeros(L)
        pos = torch.arange(s, e)[m]
        labels[pos] = ids[pos - n_prefix]
        weights[pos] = 1.0 / (t * (e - s))
        ids[pos - n_prefix] = mask_id
        rows.append((ids, labels, weights))

    Lmax = max(r[1].shape[0] for r in rows)
    B = len(rows)
    text_ids = torch.full((B, Lmax - n_prefix), mask_id, dtype=torch.long)
    labels = torch.full((B, Lmax), -100, dtype=torch.long)
    weights = torch.zeros(B, Lmax)
    for i, (ids, lab, w) in enumerate(rows):
        text_ids[i, : ids.shape[0]] = ids
        labels[i, : lab.shape[0]] = lab
        weights[i, : w.shape[0]] = w
    return text_ids, labels, weights


def mdm_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sel = labels != -100
    ce = F.cross_entropy(logits[sel].float(), labels[sel], reduction="none")
    return (ce * weights[sel]).sum() / labels.shape[0]
