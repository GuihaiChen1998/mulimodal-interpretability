"""Loading Steerling-8B and building chat-formatted token sequences."""

from __future__ import annotations

import os

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "guidelabs/steerling-8b-instruct"
DIFF_BLOCK = 64


def load_steerling(model_id: str = MODEL_ID, device: str = "cuda", freeze: bool = True, attn: str | None = None):
    """attn: "flex" or "sdpa"; must be decided before the remote modeling module is first imported
    (it reads STEERLING_USE_FLEX_ATTN at import time). None keeps the environment's choice."""
    if attn is not None:
        os.environ["STEERLING_USE_FLEX_ATTN"] = "1" if attn == "flex" else "0"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tok


def special_ids(tok) -> dict[str, int]:
    names = ["<|pad|>", "<|bos|>", "<|endoftext|>", "<|mask|>",
             "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    return {n: tok.convert_tokens_to_ids(n) for n in names}


def chat_prompt_ids(tok, user_text: str, system: str | None = None) -> list[int]:
    """Token ids of a chat prompt ending right where the assistant answer starts."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def answer_ids(tok, answer: str) -> list[int]:
    """Assistant answer closed exactly as the chat template closes a turn (e.g. <|endofchunk|><|eot_id|>)."""
    user = [{"role": "user", "content": "x"}]
    prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": answer}], tokenize=False)
    assert full.startswith(prompt), "chat template: assistant turn does not extend the generation prompt"
    return tok.encode(full[len(prompt):], add_special_tokens=False)
