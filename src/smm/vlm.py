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
    """M0: 2-layer GELU MLP (LLaVA-1.5 'mlp2x_gelu')."""

    def __init__(self, d_in: int, d_out: int = 4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
