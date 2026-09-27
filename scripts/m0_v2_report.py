"""M0 v2 report: compare target-word types (object / color / count / spatial / attribute) in one place.

For every results/m0_v2/<type>/m0_pilot_records.json:
  - runs scripts/m0_analyze.py (per-type summary + the two standard figures),
  - confidence-controlled known share (smm/m0_stats.py), also on the subset where caption B does not
    contain the target word (TXT then carries no copyable answer),
and writes results/m0_v2/REPORT.md, summary.json and m0_v2_overview.png. It also bundles the run
information Claude needs to debug a failed run (_run_info.md: env check, job status, log tails).
The A100 object run (results/m0_stage1_mlp) is included as a reference row.

Usage: python scripts/m0_v2_report.py [--dir results/m0_v2] [--logs logs/m0_v2]
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from smm.m0_stats import confidence_controlled  # noqa: E402

TYPES = ["object", "color", "count", "spatial", "attribute"]
REF = ("object (A100 ref)", os.path.join(ROOT, "results", "m0_stage1_mlp"))
INK, GRID = "#0b0b0b", "#e4e3df"
C_MAIN, C_REF = "#2a78d6", "#8a8984"

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(ROOT, "results", "m0_v2"))
ap.add_argument("--logs", default=os.path.join(ROOT, "logs", "m0_v2"))
args = ap.parse_args()

rows = []
for name, d in [REF] + [(t, os.path.join(args.dir, t)) for t in TYPES]:
    rec_path = os.path.join(d, "m0_pilot_records.json")
    if not os.path.exists(rec_path):
        print(f"skip {name}: no records")
        continue
    if d != REF[1]:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "m0_analyze.py"), rec_path, d],
                       check=True, capture_output=True)
    recs = json.load(open(rec_path))
    S = json.load(open(os.path.join(d, "m0_pilot_summary.json")))
    cc = confidence_controlled(recs)
    no_b = [r for r in recs if not r.get("b_has_word", False)] if "b_has_word" in recs[0] else []
    cc_nob = confidence_controlled(no_b) if len(no_b) >= 30 else None
    sim = S.get("sim", {})
    words = {}
    for r in recs:
        words[r["word"].lower()] = words.get(r["word"].lower(), 0) + 1
    rows.append({
        "name": name, "n": len(recs), "n_sim": sim.get("n", 0),
        "b_has_word": (float(np.mean([r["b_has_word"] for r in recs])) if "b_has_word" in recs[0] else None),
        "top_words": sorted(words.items(), key=lambda x: -x[1])[:8],
        "logp": {c: S[c]["logp"] for c in ("IMG", "TXT", "NONE")},
        "logp_IMG_minus_NONE": S["paired"]["logp IMG-NONE"],
        "known_frac": {c: S[c]["known_frac"] for c in ("IMG", "TXT", "NONE")},
        "eps_frac": {c: S[c]["eps_frac"] for c in ("IMG", "TXT", "NONE")},
        "disc_frac": {c: S[c]["disc_frac"] for c in ("IMG", "TXT", "NONE")},
        "known_IMG_minus_TXT_raw": S["paired"]["known_frac IMG-TXT"],
        "known_IMG_vs_TXT_controlled": cc["IMG_vs_TXT"],
        "known_NONE_vs_TXT_controlled": cc["NONE_vs_TXT"],
        "known_IMG_vs_TXT_controlled_noB": cc_nob["IMG_vs_TXT"] if cc_nob else None,
        "n_noB": len(no_b),
        "ebc128": {k: sim.get(k, {}).get("explained_beyond_chance@128") for k in ("IMG", "TXT", "IMG_matched")},
        "ebc128_IMG_minus_TXT": sim.get("paired EBC IMG-TXT", {}).get("128"),
        "ebc128_IMGmatched_minus_TXT": sim.get("paired EBC IMG_matched-TXT", {}).get("128"),
    })

os.makedirs(args.dir, exist_ok=True)
json.dump(rows, open(os.path.join(args.dir, "summary.json"), "w"), indent=1)


def f(v, pct=True, sign=True):
    if v is None:
        return "—"
    s = 100 if pct else 1
    fmt = "{:+.1f}" if sign else "{:.1f}"
    if not pct:
        fmt = "{:+.2f}" if sign else "{:.2f}"
    return f"{fmt.format(v[0] * s)} [{fmt.format(v[1] * s)}, {fmt.format(v[2] * s)}]"


# ---- figure: three effects per target type ------------------------------------------------------
panels = [("logp_IMG_minus_NONE", "Image informative?\nlog p  IMG − NONE (nats)", 1),
          ("known_IMG_vs_TXT_controlled", "Known-concept share, IMG vs TXT\n(controlled for log p; pp)", 100),
          ("ebc128_IMGmatched_minus_TXT", "SIM beyond chance @K=128,\nIMG (size-matched) − TXT (pp)", 100)]
if rows:
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(rows) + 1.8), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, (key, title, sc) in zip(axes, panels, strict=True):
        for y, r in zip(ys, rows, strict=True):
            v = r[key]
            if v is None:
                continue
            col = C_REF if "ref" in r["name"] else C_MAIN
            ax.errorbar(v[0] * sc, y, xerr=[[(v[0] - v[1]) * sc], [(v[2] - v[0]) * sc]], fmt="o", color=col,
                        ecolor=col, capsize=3, ms=6)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r["name"] for r in rows])
    fig.suptitle("M0 v2: does visual information bypass named concepts for non-object words? "
                 "(H1 predicts < 0 in the middle and right panels)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "m0_v2_overview.png"), dpi=150)

# ---- REPORT.md --------------------------------------------------------------------------------
L = ["# M0 v2 report", "", "Generated by `scripts/m0_v2_report.py`. Mean [95% CI]; percentages are points (pp).", "",
     "![overview](m0_v2_overview.png)", "",
     "| type | n | caption B contains word | log p IMG−NONE | known share IMG/TXT/NONE | IMG vs TXT raw | "
     "IMG vs TXT controlled | same, B without word | SIM EBC@128 IMG−TXT | size-matched |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    kf = " / ".join(f"{r['known_frac'][c][0] * 100:.1f}" for c in ("IMG", "TXT", "NONE"))
    bw = "—" if r["b_has_word"] is None else f"{r['b_has_word'] * 100:.0f}%"
    L.append(f"| {r['name']} | {r['n']} | {bw} | {f(r['logp_IMG_minus_NONE'], pct=False)} | {kf} | "
             f"{f(r['known_IMG_minus_TXT_raw'])} | {f(r['known_IMG_vs_TXT_controlled'])} | "
             f"{f(r['known_IMG_vs_TXT_controlled_noB'])} (n={r['n_noB']}) | {f(r['ebc128_IMG_minus_TXT'])} | "
             f"{f(r['ebc128_IMGmatched_minus_TXT'])} |")
L += ["", "ε share IMG/TXT/NONE and most frequent target words:", ""]
for r in rows:
    ef = " / ".join(f"{r['eps_frac'][c][0] * 100:.1f}%" for c in ("IMG", "TXT", "NONE"))
    L.append(f"- **{r['name']}**: ε {ef}; words: " + ", ".join(f"{w} ({c})" for w, c in r["top_words"]))
open(os.path.join(args.dir, "REPORT.md"), "w").write("\n".join(L) + "\n")

# ---- run info for debugging -----------------------------------------------------------------------
info = ["# Run info", ""]
env = os.path.join(ROOT, "logs", "check_env.txt")
if os.path.exists(env):
    info += ["## check_env", "```", open(env).read().strip(), "```", ""]
st = os.path.join(args.logs, "status.json")
if os.path.exists(st):
    info += ["## job status", "```json", open(st).read().strip(), "```", ""]
if os.path.isdir(args.logs):
    for fn in sorted(os.listdir(args.logs)):
        if fn.endswith(".log"):
            tail = open(os.path.join(args.logs, fn), errors="replace").read().splitlines()[-25:]
            info += [f"## {fn} (last 25 lines)", "```", *tail, "```", ""]
open(os.path.join(args.dir, "_run_info.md"), "w").write("\n".join(info) + "\n")
print(f"wrote {args.dir}/REPORT.md, summary.json, m0_v2_overview.png, _run_info.md")
