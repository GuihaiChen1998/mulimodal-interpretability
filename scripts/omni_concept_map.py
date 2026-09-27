"""Audio / video concept sets for Steerling (same protocol as the COCO map, e7b/e7c).

For a modality we build
  1. V_<mod>: the known concepts Steerling actually uses when reading captions of that modality
     (mask 30% of caption tokens, 2 passes; top-32 concepts with sigmoid weight >= 0.1 at masked positions;
     kept if active in >= 5 captions and not in > 50% of them).
  2. label -> concept maps for each label vocabulary, scored by
       spec  (masked read-out specificity at the label word in captions: P(active) * idf),
       clip  (text similarity label vs concept name in a modality-aligned text encoder: CLAP for audio,
            X-CLIP for video; plain CLIP similarity is also stored as clip_img_text for comparison),
       lex   (name hit, LM-head token hit);
     score = 0.5 spec_n + 0.3 clip_n + 0.1 name_hit + 0.1 token_hit, with a high/review confidence flag.

Caption corpora are pooled with sentences from AVoCaDO's audiovisual captions (Qwen2.5-Omni-7B captioner,
ICLR 2026; training-set captions, Apache-2.0 model), which describe picture, sound and speech together.

Vocabularies / captions (under /workspace/data/omni, see tools/pod/download_omni.sh):
  audio: AudioSet ontology (527 labels), ESC-50 (50), VGGSound (~310 sound-source labels of an audio-visual
         dataset); captions: AudioCaps + Clotho
  video: Kinetics-700; captions: VATEX (en) + MSR-VTT

Usage: python omni_concept_map.py {audio|video} OMNI_DIR OUT_DIR [--limit_labels N] [--n_caps 15000]
"""

import argparse
import collections
import csv
import io
import json
import os
import random
import re
import sys
import zipfile

import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.steerling_io import load_steerling  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("modality", choices=["audio", "video"])
ap.add_argument("omni")
ap.add_argument("out")
ap.add_argument("--limit_labels", type=int, default=None)
ap.add_argument("--n_caps", type=int, default=15000)
ap.add_argument("--per_label", type=int, default=60)
ap.add_argument("--max_caps", type=int, default=120000, help="captions searched for label matches")
ap.add_argument("--av_sentences", type=int, default=60000,
                help="sentences taken from AVoCaDO audiovisual captions (0 = off)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = random.Random(0)
W_MIN = 0.1
STOP = {"a", "an", "the", "of", "and", "or", "in", "on", "with", "to", "for", "at", "by", "from", "etc", "other"}


# ---------------------------------------------------------------- vocabularies and captions
def clean(s):
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", s)).strip()


def load_vocab_and_captions(mod, O):
    vocabs, caps = {}, []
    if mod == "audio":
        rows = list(csv.DictReader(open(f"{O}/audio/audioset/class_labels_indices.csv")))
        vocabs["audioset"] = [{"label": r["display_name"],
                               "phrases": [clean(p) for p in r["display_name"].split(",") if clean(p)]} for r in rows]
        esc = sorted({r["category"] for r in csv.DictReader(open(f"{O}/audio/esc50/ESC-50-master/meta/esc50.csv"))})
        vocabs["esc50"] = [{"label": c, "phrases": [c.replace("_", " ")]} for c in esc]
        # VGGSound clips are audio-visual, but its labels are sound-source categories -> audio captions
        vgg = sorted({row[2] for row in csv.reader(open(f"{O}/video/vggsound/vggsound.csv")) if len(row) >= 3})
        vocabs["vggsound"] = [{"label": c, "phrases": [clean(p) for p in c.split(",") if clean(p)]} for c in vgg]
        for s in ("train", "val", "test"):
            caps += [r["caption"] for r in csv.DictReader(open(f"{O}/audio/audiocaps/{s}.csv"))]
        for s in ("development", "validation", "evaluation"):
            for r in csv.DictReader(open(f"{O}/audio/clotho/clotho_captions_{s}.csv")):
                caps += [r[f"caption_{i}"] for i in range(1, 6)]
    else:
        k700 = sorted({r["label"] for r in csv.DictReader(open(f"{O}/video/kinetics700/kinetics700_2020/train.csv"))})
        vocabs["kinetics700"] = [{"label": c, "phrases": [clean(c)]} for c in k700]
        vat = json.load(open(f"{O}/video/vatex/vatex_training_v1.0.json"))
        caps += [c for v in vat for c in v["enCap"]]
        for v in json.load(open(f"{O}/video/msrvtt/msrvtt_train_9k.json")):
            caps += v["caption"]
    caps = [c.strip() for c in caps if isinstance(c, str) and c.strip()]
    return vocabs, caps


def load_av_sentences(O, n):
    """Sentences from AVoCaDO's temporally aligned audiovisual captions (106,959 long paragraphs that
    describe picture, sound and speech). Split into sentences so they fit Steerling's 128-token read-out."""
    path = f"{O}/av/avocado/AVoCaDO_training_set.jsonl"
    if n <= 0 or not os.path.exists(path):
        return []
    sents = set()
    with open(path) as f:
        for line in f:
            for m in json.loads(line)["messages"]:
                if m["role"] != "assistant":
                    continue
                for snt in re.split(r"(?<=[.!?])\s+", m["content"]):
                    snt = snt.strip()
                    if 5 <= len(snt.split()) <= 60:
                        sents.add(snt)
    sents = sorted(sents)
    random.Random(1).shuffle(sents)
    return sents[:n]


vocabs, all_caps = load_vocab_and_captions(args.modality, args.omni)
n_mod_caps = len(all_caps)
av_caps = load_av_sentences(args.omni, args.av_sentences)
all_caps += av_caps
rng.shuffle(all_caps)
n_caps_total = len(all_caps)
all_caps = all_caps[: args.max_caps]
print({k: len(v) for k, v in vocabs.items()}, "captions:", len(all_caps), flush=True)

zpath = os.path.join(os.path.dirname(__file__), "..", "datasets", "known_concepts(1).zip")
with zipfile.ZipFile(zpath) as z:
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1", newline="")))
N = len(rows)

model, tok = load_steerling()
bos, mask_id = tok.convert_tokens_to_ids("<|bos|>"), tok.convert_tokens_to_ids("<|mask|>")


def enc(t, bos_=True):
    return ([bos] if bos_ else []) + tok.encode(t, add_special_tokens=False)


@torch.no_grad()
def readout(seqs):
    L = -(-max(map(len, seqs)) // 64) * 64  # block-aligned length (flex-safe; tail is <|mask|>)
    x = torch.full((len(seqs), L), mask_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    _, o = model(x.cuda(), minimal_output=False)
    return o.known_topk_indices.cpu(), torch.sigmoid(o.known_topk_logits.float()).cpu()


# ---------------------------------------------------------------- 1. V_mod from captions
g = torch.Generator().manual_seed(0)
df = torch.zeros(N)
sample_caps = all_caps[: args.n_caps]
for i in range(0, len(sample_caps), 64):
    base = [enc(c)[:128] for c in sample_caps[i: i + 64]]
    seen = [set() for _ in base]
    for _ in range(2):
        seqs, masked = [], []
        for s in base:
            m = torch.rand(len(s), generator=g) < 0.3
            m[0] = False
            seqs.append([mask_id if mm else t for t, mm in zip(s, m.tolist(), strict=True)])
            masked.append(torch.nonzero(m).squeeze(1))
        idx, w = readout(seqs)
        for j, pos in enumerate(masked):
            sel = idx[j, pos][w[j, pos] >= W_MIN]
            seen[j].update(sel[sel < N].tolist())
    for st in seen:
        if st:
            df[list(st)] += 1
n_docs = len(sample_caps)
ubiq = df / n_docs > 0.5
vmod = [i for i in range(N) if df[i] >= 5 and not ubiq[i]]
idf = torch.log(n_docs / (df + 1))
print(f"V_{args.modality}: {len(vmod)} concepts", flush=True)

# ---------------------------------------------------------------- 2. label -> concept
E = model.known_head.concept_embedding.weight[:N].float()
E = E / E.norm(dim=-1, keepdim=True)
Wl = model.transformer.lm_head.weight.float()
top_tok = []
for i in range(0, N, 2048):
    for row in (E[i: i + 2048] @ Wl.T).topk(20, dim=1).indices.cpu():
        top_tok.append({re.sub(r"[^a-z]", "", tok.decode([int(t)]).lower()) for t in row})
del E, Wl
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", dtype=torch.float16).cuda().eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# modality-specific text encoder: CLAP (audio-text) or X-CLIP (video-text, trained on Kinetics-400)
MOD_ID = {"audio": "laion/larger_clap_general", "video": "microsoft/xclip-base-patch32"}[args.modality]
if args.modality == "audio":
    from transformers import ClapModel, ClapProcessor
    mod_model, mod_proc = ClapModel.from_pretrained(MOD_ID).cuda().eval(), ClapProcessor.from_pretrained(MOD_ID)
else:
    from transformers import XCLIPModel, XCLIPProcessor
    mod_model, mod_proc = XCLIPModel.from_pretrained(MOD_ID).cuda().eval(), XCLIPProcessor.from_pretrained(MOD_ID)


@torch.no_grad()
def temb(texts, bs=512, which="clip"):
    m, tk = (clip, proc.tokenizer) if which == "clip" else (mod_model, mod_proc.tokenizer)
    out = []
    for i in range(0, len(texts), bs):
        t = tk(texts[i: i + bs], padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        e = m.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).float()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


concept_names = [r["concept_name"].lower() for r in rows]
Tc = temb(concept_names)
Tm = temb(concept_names, which="mod")
cap_lower = [c.lower() for c in all_caps]


def find_hits(phrases):
    """Captions matching a phrase exactly (with plural), else all content words (masking the last one)."""
    pat = re.compile(r"\b(" + "|".join(re.escape(p.lower()) for p in sorted(phrases, key=len, reverse=True))
                     + r")(s|es)?\b")
    hits = []
    for c, cl in zip(all_caps, cap_lower, strict=True):
        m = pat.search(cl)
        if m:
            hits.append((c, m.start(), m.end(), "exact"))
            if len(hits) >= args.per_label:
                return hits
    if len(hits) >= 5:
        return hits
    words = [w for w in re.findall(r"[a-z]+", " ".join(phrases[:1]).lower()) if w not in STOP and len(w) > 2]
    if len(words) < 2:
        return hits
    anchor = re.compile(r"\b" + re.escape(words[-1]) + r"(s|es)?\b")
    for c, cl in zip(all_caps, cap_lower, strict=True):
        if all(re.search(r"\b" + re.escape(w[:5]), cl) for w in words[:-1]):
            m = anchor.search(cl)
            if m:
                hits.append((c, m.start(), m.end(), "content-words"))
                if len(hits) >= args.per_label:
                    break
    return hits


def map_label(item):
    hits = find_hits(item["phrases"])
    present = collections.Counter()
    for k in range(0, len(hits), 32):
        seqs, spans = [], []
        for c, s0, s1, _ in hits[k: k + 32]:
            a = len(enc(c[:s0].rstrip()))
            b = max(min(len(enc(c[:s1])), 128), a + 1)
            e = enc(c)[:128]
            e[a:b] = [mask_id] * (b - a)
            seqs.append(e)
            spans.append((a, b))
        idx, w = readout(seqs)
        for j, (a, b) in enumerate(spans):
            sel = idx[j, a:b][w[j, a:b] >= W_MIN]
            present.update(set(sel[sel < N].tolist()))
    n_h = max(len(hits), 1)
    spec = {cid: present[cid] / n_h * idf[cid].item() for cid in present if not ubiq[cid]}
    sim = (temb([item["label"].lower()], which="mod") @ Tm.T)[0].cpu()      # modality text encoder
    sim_clip = (temb([item["label"].lower()]) @ Tc.T)[0].cpu()              # CLIP text encoder (reference)
    words = {p.lower() for p in item["phrases"]} | {w for p in item["phrases"] for w in re.findall(r"[a-z]+", p.lower())
                                                   if w not in STOP and len(w) > 3}
    name_pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in item["phrases"]) + r")s?\b", re.I)
    pool = set(sorted(spec, key=lambda k: -spec[k])[:10]) | set(sim.topk(10).indices.tolist())
    cands = []
    for cid in pool:
        cands.append({"id": cid, "name": rows[cid]["concept_name"], "spec": spec.get(cid, 0.0),
                      "frac": round(present[cid] / n_h, 3), "clip": sim[cid].item(), "clip_img_text": round(sim_clip[cid].item(), 4),
                      "name_hit": bool(name_pat.search(rows[cid]["concept_name"] + " " + rows[cid]["group_name"])),
                      "token_hit": bool(top_tok[cid] & words), "steerable": rows[cid]["is_steerable"] == "TRUE",
                      "in_v_mod": df[cid].item() >= 5 and not ubiq[cid].item()})
    smax = max(d["spec"] for d in cands) or 1.0
    cmin, cmax = min(d["clip"] for d in cands), max(d["clip"] for d in cands)
    for d in cands:
        d["score"] = round(0.5 * d["spec"] / smax + 0.3 * (d["clip"] - cmin) / max(cmax - cmin, 1e-6)
                           + 0.1 * d["name_hit"] + 0.1 * d["token_hit"], 3)
        d["spec"], d["clip"] = round(d["spec"], 3), round(d["clip"], 4)
    cands.sort(key=lambda d: -d["score"])
    best = cands[0]
    top_spec, top_clip = max(cands, key=lambda d: d["spec"])["id"], max(cands, key=lambda d: d["clip"])["id"]
    margin = best["score"] - cands[1]["score"] if len(cands) > 1 else 1.0
    return {"label": item["label"], "phrases": item["phrases"], "n_captions": len(hits),
            "match": hits[0][3] if hits else "none", "primary": best["id"], "primary_name": best["name"],
            "confidence": ("no-captions" if not hits else
                           "high" if (best["id"] == top_spec == top_clip) or margin >= 0.15 else "review"),
            "margin": round(margin, 3), "candidates": cands[:6]}


summary = {"modality": args.modality, "n_caption_docs": n_docs, "n_captions_total": n_caps_total, "n_captions_searched": len(all_caps),
           "caption_sources": {"modality_corpora": n_mod_caps, "avocado_av_sentences": len(av_caps)},
           "n_v_mod": len(vmod), "n_v_mod_steerable": sum(rows[i]["is_steerable"] == "TRUE" for i in vmod),
           "most_frequent_v_mod": [(rows[i]["concept_name"], int(df[i])) for i in sorted(vmod, key=lambda i: -df[i])[:25]],
           "vocabs": {}}
for vname, items in vocabs.items():
    res = []
    for it in items[: args.limit_labels]:
        res.append(map_label(it))
        if len(res) % 50 == 0:
            print(f"{vname}: {len(res)}/{len(items)}", flush=True)
    json.dump(res, open(os.path.join(args.out, f"{vname}2concept.json"), "w"), indent=1)
    summary["vocabs"][vname] = {
        "n_labels": len(res), "with_captions": sum(r["n_captions"] > 0 for r in res),
        "high_confidence": sum(r["confidence"] == "high" for r in res),
        "no_captions": sum(r["confidence"] == "no-captions" for r in res),
        "distinct_primary": len({r["primary"] for r in res}),
        "examples": [(r["label"], r["primary_name"], r["confidence"], r["n_captions"]) for r in res[:: max(1, len(res) // 25)]]}
json.dump({"concept_ids": vmod, "doc_freq": {str(i): int(df[i]) for i in range(N) if df[i] > 0}},
          open(os.path.join(args.out, f"v_{args.modality}.json"), "w"))
json.dump(summary, open(os.path.join(args.out, f"{args.modality}_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=1)[:3000], flush=True)
print("DONE", flush=True)
