"""Window-level audiovisual captions with AVoCaDO (Qwen2.5-Omni-7B captioner), for time-localized concepts.

Each clip is cut into overlapping windows (default 6 s, hop 3 s) with ffmpeg (re-encoded so cuts are exact
and the audio track is kept); every window is captioned separately so caption and signal cover the same span.
Also captions the full clip once (clip-level reference). Records per-window generation time.

Inference follows the official AVoCaDO inference.py (Qwen2_5OmniForConditionalGeneration, talker disabled,
use_audio_in_video=True, greedy), except attention uses SDPA (flash-attn is not installed) and the number of
new tokens is capped.

Usage: python avocado_window_captions.py VIDEO_DIR OUT_JSONL [--n_clips 50] [--win 6] [--hop 3]
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

ap = argparse.ArgumentParser()
ap.add_argument("video_dir")
ap.add_argument("out")
ap.add_argument("--n_clips", type=int, default=50)
ap.add_argument("--win", type=float, default=6.0)
ap.add_argument("--hop", type=float, default=3.0)
ap.add_argument("--max_new_tokens", type=int, default=384)
ap.add_argument("--model", default="AVoCaDO-Captioner/AVoCaDO")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

VIDEO_MAX_PIXELS = 401408  # 512*28*28, as in the official script
os.environ["VIDEO_MAX_PIXELS"] = str(401408 * 50)
PROMPT = ("Please provide a thorough description of all the content in the video, including every detail. "
          "As you describe, ensure that you also cover as much information from the audio as possible, "
          "and be mindful of the synchronization between the audio and video as you do so.")
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                            device_map="cuda", attn_implementation="sdpa")
model.disable_talker()
proc = Qwen2_5OmniProcessor.from_pretrained(args.model)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def cut(src, start, length, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", src, "-t", f"{length:.3f}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", dst], check=True)


def load_audio(path, sr=16000):
    """Mono 16 kHz float32 via ffmpeg. qwen_omni_utils hands librosa an audioread object, which the
    installed librosa rejects ("Invalid file"), so the audio track is decoded here instead."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


@torch.no_grad()
def caption(path):
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "video", "video": path, "max_pixels": VIDEO_MAX_PIXELS},
                                         {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    _, images, videos = process_mm_info(conv, use_audio_in_video=False)
    wav = load_audio(path)
    use_audio = wav.size > 1600  # clips without an audio track fall back to video only
    inputs = proc(text=text, audio=[wav] if use_audio else None, images=images, videos=videos, return_tensors="pt",
                  padding=True, use_audio_in_video=use_audio).to(model.device).to(model.dtype)
    t0 = time.time()
    ids = model.generate(**inputs, use_audio_in_video=use_audio, do_sample=False,
                         thinker_max_new_tokens=args.max_new_tokens, return_audio=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    out = proc.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    n_in = inputs["input_ids"].shape[1]
    return out.split("\nassistant\n")[-1].strip(), dt, n_in, ids.shape[1] - n_in


files = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
random.Random(args.seed).shuffle(files)
files = files[: args.n_clips]
done = set()
if os.path.exists(args.out):
    done = {json.loads(line)["clip_id"] for line in open(args.out)}
t_start = time.time()
with open(args.out, "a") as fo, tempfile.TemporaryDirectory() as tmp:
    for ci, f in enumerate(files):
        cid = f[:-4]
        if cid in done:
            continue
        src = os.path.join(args.video_dir, f)
        dur = duration(src)
        rec = {"clip_id": cid, "duration": dur, "windows": []}
        try:
            cap, dt, n_in, n_out = caption(src)
            rec["clip_caption"] = {"caption": cap, "gen_s": round(dt, 2), "in_tokens": n_in, "out_tokens": n_out}
            s = 0.0
            while s < max(dur - 1.0, 0.0):
                length = min(args.win, dur - s)
                w = os.path.join(tmp, f"{cid}_{s:.0f}.mp4")
                cut(src, s, length, w)
                cap, dt, n_in, n_out = caption(w)
                rec["windows"].append({"start": round(s, 2), "end": round(s + length, 2), "caption": cap,
                                       "gen_s": round(dt, 2), "in_tokens": n_in, "out_tokens": n_out})
                os.remove(w)
                if s + args.win >= dur:
                    break
                s += args.hop
        except Exception as e:  # keep going; record the failure
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        nw = len(rec["windows"])
        mean_w = sum(w["gen_s"] for w in rec["windows"]) / max(nw, 1)
        print(f"[{ci + 1}/{len(files)}] {cid} dur={dur:.1f}s windows={nw} mean_gen={mean_w:.1f}s "
              f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
print("DONE", flush=True)
