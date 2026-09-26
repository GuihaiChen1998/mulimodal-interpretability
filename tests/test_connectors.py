"""Connector shapes, block alignment and gradient flow (CPU)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.vlm import MLPProjector, QueryResampler, build_connector  # noqa: E402


@pytest.mark.parametrize("name,n", [("mlp", 576), ("resampler", 64), ("resampler128", 128)])
def test_shapes_and_alignment(name, n):
    c = build_connector(name, 1024, 256)
    y = c(torch.randn(2, 576, 1024))
    assert y.shape == (2, n, 256) and c.n_tokens == n and n % 64 == 0


def test_resampler_depends_on_image_and_trains():
    torch.manual_seed(0)
    c = QueryResampler(1024, 256, n_queries=64, width=128, depth=2, heads=4)
    a, b = torch.randn(1, 576, 1024), torch.randn(1, 576, 1024)
    assert not torch.allclose(c(a), c(b))
    c(a).pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())


def test_bad_query_count_rejected():
    with pytest.raises(AssertionError):
        build_connector("resampler100", 1024)


def test_mlp_is_per_patch():
    c = MLPProjector(8, 4)
    x = torch.randn(1, 576, 8)
    y = c(x)
    x2 = x.clone()
    x2[0, 5] += 1
    d = (c(x2) - y).abs().sum(-1)[0]
    assert d[5] > 0 and d[torch.arange(576) != 5].max() == 0
