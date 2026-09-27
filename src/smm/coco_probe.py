"""COCO masked-object probe shared by M0 diagnostics and Stage-1 evaluation.

A sample = (COCO val image, caption A containing a word for an object annotated in the image, another
caption B of the same image). The object word in caption A is masked; three conditions supply the
information: IMG (image tokens), TXT (caption B in the prompt), NONE (language prior only).
"""

from __future__ import annotations

import json
import os
import random
import re

import torch
from PIL import Image

from smm.steerling_io import answer_ids

INSTR = "Give a brief description of the image."


class CocoProbe:
    def __init__(self, tok, coco_dir: str, cmap: dict, n: int, seed: int = 0):
        self.tok, self.coco_dir, self.cmap = tok, coco_dir, cmap
        self.mask_id = tok.convert_tokens_to_ids("<|mask|>")
        ann = json.load(open(os.path.join(coco_dir, "annotations", "instances_val2017.json")))
        caps = json.load(open(os.path.join(coco_dir, "annotations", "captions_val2017.json")))
        catname = {c["id"]: c["name"] for c in ann["categories"]}
        img_cats: dict[int, set] = {}
        for a in ann["annotations"]:
            img_cats.setdefault(a["image_id"], set()).add(catname[a["category_id"]])
        img_caps: dict[int, list] = {}
        for c in caps["annotations"]:
            img_caps.setdefault(c["image_id"], []).append(c["caption"].strip())
        img_file = {i["id"]: i["file_name"] for i in ann["images"]}

        rng = random.Random(seed)
        self.samples = []
        ids = sorted(img_caps)
        rng.shuffle(ids)
        for iid in ids:
            cands = sorted(c for c in img_cats.get(iid, ()) if c in cmap)
            rng.shuffle(cands)
            done = False
            for cat in cands:
                phrases = cmap[cat].get("phrases", [cat])
                pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
                                 + r")\b", re.I)
                for ci, capA in enumerate(img_caps[iid]):
                    m = pat.search(capA)
                    if not m:
                        continue
                    capB = [c for j, c in enumerate(img_caps[iid]) if j != ci][0]
                    ans = answer_ids(tok, capA)
                    a, b = len(self.enc(capA[: m.start()].rstrip())), len(self.enc(capA[: m.end()]))
                    if b <= a or m.group(0).lower() not in tok.decode(ans[a:b]).lower():
                        continue
                    self.samples.append({"image_id": iid, "file": img_file[iid], "category": cat,
                                         "word": m.group(0), "capA": capA, "capB": capB, "ans": ans, "span": (a, b)})
                    done = True
                    break
                if done:
                    break
            if len(self.samples) >= n:
                break

    def enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def chat_text(self, user: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True)

    def image_tokens(self, s: dict, vision, proj) -> torch.Tensor:
        im = Image.open(os.path.join(self.coco_dir, "val2017", s["file"])).convert("RGB")
        pv = vision.processor(images=[im], return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad():
            return proj(vision(pv).float()).to(torch.bfloat16)

    def build(self, s: dict, cond: str, emb, vision=None, proj=None, pad_to_block: bool = False):
        """Returns (embedding parts, context part index or None, absolute target positions, target ids).

        pad_to_block appends <|mask|> embeddings up to a multiple of 64 tokens (the future region is
        masked at generation time anyway); needed with flex attention, whose Triton kernel fails on A100
        for 64 < T < 128 when T is not a multiple of 64."""
        a, b = s["span"]
        ans = list(s["ans"])
        targets = ans[a:b]
        ans[a:b] = [self.mask_id] * (b - a)
        dev = emb.weight.device
        ans_e = emb(torch.tensor([ans], device=dev))
        if cond == "TXT":
            t = self.chat_text(INSTR + "\nReference description: " + s["capB"])
            i = t.index(s["capB"])
            pre, cap, post = self.enc(t[:i]), self.enc(t[i: i + len(s["capB"])]), self.enc(t[i + len(s["capB"]):])
            parts = [emb(torch.tensor([x], device=dev)) for x in (pre, cap, post)] + [ans_e]
            ctx = 1
        else:
            parts = [emb(torch.tensor([self.enc(self.chat_text(INSTR))], device=dev)), ans_e]
            ctx = None
            if cond == "IMG":
                parts = [self.image_tokens(s, vision, proj)] + parts
                ctx = 0
        off = sum(p.shape[1] for p in parts[:-1])
        if pad_to_block:
            T = sum(p.shape[1] for p in parts)
            if T % 64:
                parts = parts + [emb(torch.full((1, 64 - T % 64), self.mask_id, device=dev))]
        return parts, ctx, [off + a + j for j in range(b - a)], targets

    @torch.no_grad()
    def per_sample_logp(self, lm, emb, vision, proj, conds=("IMG", "NONE"), n: int | None = None,
                        pad_to_block: bool = False) -> dict:
        """Per condition: list over samples of mean log p(target tokens)."""
        per = {c: [] for c in conds}
        for s in self.samples[:n]:
            for c in conds:
                parts, _, pos, tg = self.build(s, c, emb, vision, proj, pad_to_block=pad_to_block)
                logits, _ = lm(None, input_embeds=torch.cat(parts, 1), minimal_output=True)
                lp = torch.log_softmax(logits[0, pos].float(), -1)
                per[c].append(lp[torch.arange(len(tg)), torch.tensor(tg, device=lp.device)].mean().item())
        return per


def summarize_logp(per: dict) -> dict:
    out = {c: sum(v) / len(v) for c, v in per.items()}
    if "IMG" in per and "NONE" in per:
        d = [x - y for x, y in zip(per["IMG"], per["NONE"], strict=True)]
        out["IMG-NONE"] = sum(d) / len(d)
        out["frac_IMG_better"] = sum(x > 0 for x in d) / len(d)
    return out
