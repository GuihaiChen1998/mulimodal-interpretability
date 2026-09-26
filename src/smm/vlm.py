"""Vision encoder + projector + frozen Steerling (M0 baseline wiring).

Sequence layout: [576 image tokens = 9 diffusion blocks][chat prompt][answer ...].
Image tokens are never masked; text starts at a block boundary (position 576).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel

CLIP_ID = "openai/clip-vit-large-patch14-336"
N_IMG_TOKENS = 576


class VisionEncoder(nn.Module):
    """Frozen CLIP ViT-L/14-336; penultimate layer, CLS dropped (LLaVA-1.5 convention)."""

    def __init__(self, model_id: str = CLIP_ID, select_layer: int = -2):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.select_layer = select_layer
        self.clip.requires_grad_(False).eval()

    @property
    def dim(self) -> int:
        return self.clip.config.hidden_size

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.clip(pixel_values.to(self.clip.dtype), output_hidden_states=True)
        return out.hidden_states[self.select_layer][:, 1:]  # [B, 576, 1024]


class MLPProjector(nn.Module):
    """M0 connector A: 2-layer GELU MLP (LLaVA-1.5 'mlp2x_gelu'); one output token per CLIP patch."""

    def __init__(self, d_in: int, d_out: int = 4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        self.n_tokens = N_IMG_TOKENS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryResampler(nn.Module):
    """M0 connector B: Perceiver/Q-Former-style resampler (Flamingo, BLIP-2 family).

    n_queries learnable queries cross-attend to the CLIP patch features through `depth` blocks of
    (cross-attention -> FFN), then a linear map to the LM width. The default 64 queries fill exactly
    one 64-token diffusion block of Steerling.
    """

    def __init__(self, d_in: int, d_out: int = 4096, n_queries: int = 64, width: int = 1024,
                 depth: int = 2, heads: int = 16):
        super().__init__()
        self.n_tokens = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, width) * 0.02)
        self.in_proj = nn.Linear(d_in, width)
        self.pos = nn.Parameter(torch.zeros(N_IMG_TOKENS, width))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(width), "ln_kv": nn.LayerNorm(width),
                "attn": nn.MultiheadAttention(width, heads, batch_first=True),
                "ln_ff": nn.LayerNorm(width),
                "ff": nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)),
            }))
        self.ln_out = nn.LayerNorm(width)
        self.out = nn.Linear(width, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kv = self.in_proj(x) + self.pos[: x.shape[1]]
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        for b in self.blocks:
            k = b["ln_kv"](kv)
            q = q + b["attn"](b["ln_q"](q), k, k, need_weights=False)[0]
            q = q + b["ff"](b["ln_ff"](q))
        return self.out(self.ln_out(q))


def build_connector(name: str, d_in: int, d_out: int = 4096) -> nn.Module:
    """name: 'mlp' (576 tokens) or 'resampler' / 'resampler<N>' (N query tokens, default 64)."""
    if name == "mlp":
        return MLPProjector(d_in, d_out)
    if name.startswith("resampler"):
        n = int(name[len("resampler"):] or 64)
        assert n % 64 == 0, "keep the image prefix block-aligned (multiple of 64 tokens)"
        return QueryResampler(d_in, d_out, n_queries=n)
    raise ValueError(f"unknown connector {name!r}")


class SteerlingVLM(nn.Module):
    def __init__(self, steerling, vision: VisionEncoder, projector: nn.Module):
        super().__init__()
        self.lm = steerling
        self.vision = vision
        self.projector = projector

    def embed(self, pixel_values: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        img = self.projector(self.vision(pixel_values).to(self.projector_dtype))
        txt = self.lm.transformer.tok_emb(text_ids)
        return torch.cat([img.to(txt.dtype), txt], dim=1)

    @property
    def projector_dtype(self):
        return next(self.projector.parameters()).dtype

    def forward(self, pixel_values: torch.Tensor, text_ids: torch.Tensor, **kw):
        return self.lm(None, input_embeds=self.embed(pixel_values, text_ids), **kw)
