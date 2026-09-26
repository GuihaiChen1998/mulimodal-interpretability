"""E5 unit tests for the block-wise MDM batch builder and loss (CPU only)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.mdm import BLOCK, build_mdm_batch, mdm_loss  # noqa: E402

MASK = 999
P = 576


def _sample(plen, alen, **kw):
    prompt = list(range(1, plen + 1))
    answer = list(range(1000, 1000 + alen))
    return prompt, answer, build_mdm_batch([prompt], [answer], n_prefix=P, mask_id=MASK, **kw)


def test_lengths_are_block_multiples_and_prefix_untouched():
    for plen, alen in [(20, 10), (50, 30), (40, 200)]:
        _, _, (txt, lab, w) = _sample(plen, alen)
        assert lab.shape[1] % BLOCK == 0
        assert txt.shape[1] == lab.shape[1] - P
        assert (lab[:, :P] == -100).all() and (w[:, :P] == 0).all()


def test_only_answer_positions_in_chosen_block_carry_loss():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        prompt, answer, (txt, lab, w) = _sample(30, 150, gen=g)
        pos = (lab[0] != -100).nonzero().squeeze(1)
        a0 = P + len(prompt)
        assert (pos >= a0).all()
        assert (pos // BLOCK == pos[0] // BLOCK).all()           # single block
        assert pos.max() // BLOCK == (lab.shape[1] - 1) // BLOCK  # it is the last block
        assert (txt[0, pos - P] == MASK).all()                   # masked in input
        assert (lab[0, pos] == torch.tensor(answer)[pos - a0]).all()  # labels are true tokens


def test_earlier_answer_blocks_are_clean():
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        prompt, answer, (txt, lab, _) = _sample(30, 200, gen=g)
        b = (lab[0] != -100).nonzero()[0, 0].item() // BLOCK
        text = torch.tensor(prompt + answer)
        clean_len = b * BLOCK - P
        assert (txt[0, :clean_len] == text[:clean_len]).all()


def test_mask_ratio_tracks_t():
    g = torch.Generator().manual_seed(2)
    for t in (0.2, 0.5, 0.9):
        ratios = []
        for _ in range(300):
            prompt, answer, (_, lab, _) = _sample(0, 64, t_fixed=t, block_idx="first", gen=g)
            ratios.append((lab != -100).sum().item() / 64)
        assert abs(sum(ratios) / len(ratios) - t) < 0.03


def test_at_least_one_masked_and_weights():
    g = torch.Generator().manual_seed(3)
    _, _, (_, lab, w) = _sample(10, 5, t_fixed=1e-6, gen=g)
    assert (lab != -100).sum() == 1
    _, _, (_, lab, w) = _sample(0, 64, t_fixed=0.5, block_idx="first", gen=g)
    sel = lab != -100
    assert torch.allclose(w[sel], torch.full_like(w[sel], 1 / (0.5 * 64)))


def test_loss_finite_and_zero_for_perfect_logits():
    _, _, (txt, lab, w) = _sample(20, 40, t_fixed=1.0)
    V = 2000
    logits = torch.zeros(1, lab.shape[1], V)
    sel = lab != -100
    logits[sel, lab[sel]] = 100.0
    assert mdm_loss(logits, lab, w).item() < 1e-6
    rnd = mdm_loss(torch.randn(1, lab.shape[1], V), lab, w)
    assert torch.isfinite(rnd) and rnd.item() > 0
