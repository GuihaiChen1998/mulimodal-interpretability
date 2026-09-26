"""E3: input_embeds path + block alignment of a 576-token image prefix.

1. logits(input_ids) == logits(input_embeds = tok_emb(input_ids))
2. With a 576-token prefix (9 blocks), text starts at position 576 = start of block 9.
3. Block-causal visibility, checked empirically through the model:
   - perturbing the LAST image block must not change outputs of earlier image blocks (causal across blocks)
   - perturbing a token inside a block changes outputs of other tokens in that same block (bidirectional)
   - perturbing any image block changes text outputs (text sees all image blocks)
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import DIFF_BLOCK, chat_prompt_ids, load_steerling  # noqa: E402

out_dir = sys.argv[1]
model, tok = load_steerling()
emb = model.transformer.tok_emb
res = {}

ids = torch.tensor([chat_prompt_ids(tok, "Describe the image in one sentence.")], device="cuda")
with torch.no_grad():
    l_ids, _ = model(ids)
    l_emb, _ = model(None, input_embeds=emb(ids))
res["ids_vs_embeds_max_abs_logit_diff"] = (l_ids.float() - l_emb.float()).abs().max().item()

N_IMG = 576
torch.manual_seed(0)
scale = emb.weight.float().norm(dim=-1).mean().item() / (4096 ** 0.5)
img = torch.randn(1, N_IMG, 4096, device="cuda", dtype=torch.bfloat16) * scale
x = torch.cat([img, emb(ids)], dim=1)
T = x.shape[1]
res["seq_len"] = T
res["text_start_pos"] = N_IMG
res["text_start_block"] = N_IMG // DIFF_BLOCK
res["text_start_is_block_boundary"] = N_IMG % DIFF_BLOCK == 0


def hidden(inp):
    with torch.no_grad():
        return model.transformer(None, input_embeds=inp, return_hidden=True).float()


h0 = hidden(x)


def perturb(pos):
    y = x.clone()
    y[0, pos] = y[0, pos] + 5 * scale * torch.randn_like(y[0, pos])
    return (hidden(y) - h0).abs().amax(dim=-1)[0]  # [T]


d_last = perturb(slice(8 * 64, 9 * 64))          # last image block (block 8)
d_tok = perturb(slice(3 * 64 + 10, 3 * 64 + 11))  # one token in image block 3
d_first = perturb(slice(0, 64))                   # first image block
res["visibility"] = {
    "perturb_block8__max_change_in_blocks0-7": d_last[: 8 * 64].max().item(),
    "perturb_block8__min_change_in_text": d_last[N_IMG:].min().item(),
    "perturb_tok_in_block3__min_change_other_tokens_block3": torch.cat(
        [d_tok[3 * 64: 3 * 64 + 10], d_tok[3 * 64 + 11: 4 * 64]]).min().item(),
    "perturb_tok_in_block3__max_change_blocks0-2": d_tok[: 3 * 64].max().item(),
    "perturb_block0__min_change_in_text": d_first[N_IMG:].min().item(),
}
v = res["visibility"]
res["pass"] = (
    res["ids_vs_embeds_max_abs_logit_diff"] < 1e-3
    and v["perturb_block8__max_change_in_blocks0-7"] == 0.0
    and v["perturb_tok_in_block3__max_change_blocks0-2"] == 0.0
    and v["perturb_tok_in_block3__min_change_other_tokens_block3"] > 0
    and v["perturb_block8__min_change_in_text"] > 0
    and v["perturb_block0__min_change_in_text"] > 0
)
json.dump(res, open(os.path.join(out_dir, "e3_input_embeds.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
print("E3", "PASS" if res["pass"] else "FAIL")
