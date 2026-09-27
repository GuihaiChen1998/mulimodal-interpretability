"""SIM symmetry-II tool on a toy model with the Steerling interface (CPU)."""

import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.diagnostics import sim_violation  # noqa: E402

D, V, C, T = 16, 50, 40, 12


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, V, bias=False)

    def forward(self, _ids, input_embeds, return_hidden=True):
        x = torch.tanh(self.mix(input_embeds))
        return x.cumsum(1)  # position t sees tokens <= t


def toy(lm_row_in_concepts: bool):
    torch.manual_seed(0)
    m = types.SimpleNamespace()
    m.transformer = Backbone()
    head = types.SimpleNamespace(concept_predictor=nn.Linear(D, C, bias=False), n_concepts=C)
    m.known_head = head
    if lm_row_in_concepts:  # target logit is an exact combination of 3 concept readouts
        with torch.no_grad():
            m.transformer.lm_head.weight[0] = head.concept_predictor.weight[:3].sum(0)
    return m


def run(m, **kw):
    x = [torch.randn(1, T, D), torch.randn(1, 4, D)]
    return sim_violation(m, x, ctx_idx=0, position=T + 3, target=0, ks=(1, 2, 4, 8, 16), chunk=4, **kw)


def test_nested_and_bounded():
    r = run(toy(False), matched_m=5)
    for key in ("top", "random"):
        v = [r[key][k] for k in (1, 2, 4, 8, 16)]
        assert all(0 <= a <= 1 + 1e-6 for a in v)
        assert all(a >= b - 1e-6 for a, b in zip(v, v[1:], strict=False))  # more concepts never hurt
    assert r["ctx_tokens"] == T and r["matched"]["m"] == 5 and 0 < r["matched"]["gf_share"] <= 1


def test_zero_violation_when_prediction_goes_through_concepts():
    torch.manual_seed(1)
    m = toy(True)
    x = [torch.randn(1, T, D), torch.randn(1, 4, D)]
    h = m.transformer(None, torch.cat(x, 1))[0, T + 3]
    # make sure the 3 concepts used by the target are the top-3 active ones at this position
    with torch.no_grad():
        W = m.known_head.concept_predictor.weight
        W[:3] *= torch.sign(W[:3] @ h).unsqueeze(1)  # these three are active (positive logits)...
        W[3:] *= 0.01                                  # ...and dominate every other concept
        m.transformer.lm_head.weight[0] = W[:3].sum(0)
    r = sim_violation(m, x, 0, T + 3, 0, ks=(3, 8), chunk=4)
    assert r["top"][3] < 1e-4, r
    assert h.shape == (D,)


def test_matched_skipped_when_context_small():
    r = run(toy(False), matched_m=T)
    assert "matched" not in r
