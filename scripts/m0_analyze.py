"""Summarise and plot the M0 diagnostic pilot (m0_pilot_records.json).

Per sample, multi-token targets are averaged first, so samples are the unit of analysis.
CIs: 95% bootstrap over samples (paired for condition differences).

Usage: python m0_analyze.py RECORDS_JSON OUT_DIR
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

recs = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
CONDS = ["IMG", "TXT", "NONE"]
LABEL = {"IMG": "Image context", "TXT": "Text context\n(another caption)", "NONE": "No context\n(language prior)"}
# reference categorical palette, first three slots (validated all-pairs, light mode)
C_KNOWN, C_DISC, C_EPS = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
rng = np.random.default_rng(0)


def boot(x, n=5000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_sample(cond, key):
    return np.array([np.mean(r[cond][key]) for r in recs])


def fracs(cond):
    k, d, e = (np.array([np.abs(r[cond][c]) for r in recs], dtype=object) for c in ("known_c", "disc_c", "eps_c"))
    fk, fd, fe = [], [], []
    for a, b, c in zip(k, d, e, strict=True):
        s = a + b + c
        fk.append(np.mean(a / s)), fd.append(np.mean(b / s)), fe.append(np.mean(c / s))
    return np.array(fk), np.array(fd), np.array(fe)


S = {"n_samples": len(recs)}
for c in CONDS:
    fk, fd, fe = fracs(c)
    S[c] = {
        "known_frac": boot(fk), "disc_frac": boot(fd), "eps_frac": boot(fe),
        "known_c": boot(per_sample(c, "known_c")), "disc_c": boot(per_sample(c, "disc_c")),
        "eps_c": boot(per_sample(c, "eps_c")),
        "logp": boot(per_sample(c, "logp")),
        "mapped_in_top32": (boot([float(r[c]["mapped_in_top32"]) for r in recs])
                            if recs[0][c].get("mapped_in_top32") is not None else None),
    }
fk = {c: fracs(c)[0] for c in CONDS}
lp = {c: per_sample(c, "logp") for c in CONDS}
S["paired"] = {
    "known_frac IMG-TXT": boot(fk["IMG"] - fk["TXT"]), "known_frac IMG-NONE": boot(fk["IMG"] - fk["NONE"]),
    "logp IMG-NONE": boot(lp["IMG"] - lp["NONE"]), "logp TXT-NONE": boot(lp["TXT"] - lp["NONE"]),
}
sim = [r for r in recs if "sim" in r["IMG"]]
KS = sorted(int(k) for k in sim[0]["IMG"]["sim"]["top"]) if sim else []
# records from the first pilot have a single random-128 value; newer ones have random-K for every K
PER_K = bool(sim) and isinstance(sim[0]["IMG"]["sim"]["random"], dict)


def rnd(sim_rec, k=128):
    v = sim_rec["random"]
    return v[str(k)] if isinstance(v, dict) else v


def ebc(sim_rec, k):  # explained beyond chance at K
    return 1 - sim_rec["top"][str(k)] / rnd(sim_rec, k)


S["sim"] = {"n": len(sim), "random_per_K": PER_K}
for c in ("IMG", "TXT"):
    S["sim"][c] = {f"top{k}": boot([r[c]["sim"]["top"][str(k)] for r in sim]) for k in KS}
    S["sim"][c]["random128"] = boot([rnd(r[c]["sim"]) for r in sim])
    S["sim"][c]["explained_beyond_chance@128"] = boot([ebc(r[c]["sim"], 128) for r in sim])
    if PER_K:
        for k in KS:
            S["sim"][c][f"random{k}"] = boot([rnd(r[c]["sim"], k) for r in sim])
            S["sim"][c][f"explained_beyond_chance@{k}"] = boot([ebc(r[c]["sim"], k) for r in sim])
S["sim"]["paired top128 IMG-TXT"] = boot([r["IMG"]["sim"]["top"]["128"] - r["TXT"]["sim"]["top"]["128"] for r in sim])
matched = [r for r in sim if "matched" in r["IMG"]["sim"]]
if PER_K:
    S["sim"]["paired EBC IMG-TXT"] = {k: boot([ebc(r["IMG"]["sim"], k) - ebc(r["TXT"]["sim"], k) for r in sim])
                                      for k in KS}
if matched:
    S["sim"]["IMG_matched"] = {
        "n": len(matched), "m_mean": float(np.mean([r["IMG"]["sim"]["matched"]["m"] for r in matched])),
        "gf_share": boot([r["IMG"]["sim"]["matched"]["gf_share"] for r in matched])}
    for k in KS:
        S["sim"]["IMG_matched"][f"top{k}"] = boot([r["IMG"]["sim"]["matched"]["top"][str(k)] for r in matched])
        S["sim"]["IMG_matched"][f"random{k}"] = boot([r["IMG"]["sim"]["matched"]["random"][str(k)] for r in matched])
        S["sim"]["IMG_matched"][f"explained_beyond_chance@{k}"] = boot(
            [ebc(r["IMG"]["sim"]["matched"], k) for r in matched])
    S["sim"]["paired EBC IMG_matched-TXT"] = {
        k: boot([ebc(r["IMG"]["sim"]["matched"], k) - ebc(r["TXT"]["sim"], k) for r in matched]) for k in KS}
json.dump(S, open(os.path.join(out, "m0_pilot_summary.json"), "w"), indent=2)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) share of |logit| by component
x = np.arange(len(CONDS))
bottom = np.zeros(len(CONDS))
for key, col, name in (("known_frac", C_KNOWN, "Known (named) concepts"), ("disc_frac", C_DISC, "Discovered concepts"),
                       ("eps_frac", C_EPS, "Residual ε")):
    v = np.array([S[c][key][0] for c in CONDS])
    ax[0].bar(x, v, 0.55, bottom=bottom, color=col, label=name, edgecolor="white", linewidth=2)
    for i in range(len(CONDS)):
        if v[i] > 0.06:
            ax[0].text(x[i], bottom[i] + v[i] / 2, f"{v[i]:.0%}", ha="center", va="center", color="white", fontsize=9)
    bottom += v
ax[0].set_xticks(x, [LABEL[c] for c in CONDS])
ax[0].set_ylim(0, 1)
ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[0].set_ylabel("Share of |target-token logit|")
ax[0].set_title("(a) Logit decomposition at the masked object word", loc="left", color=INK)
ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5)

# (b) SIM symmetry-II violation vs K
for c, col, ls in (("IMG", C_KNOWN, "-"), ("TXT", C_DISC, "-")):
    m = [S["sim"][c][f"top{k}"] for k in KS]
    ax[1].plot(KS, [v[0] for v in m], ls, color=col, lw=2, marker="o", ms=8, label=f"{c}: top-K named concepts")
    ax[1].fill_between(KS, [v[1] for v in m], [v[2] for v in m], color=col, alpha=0.15, lw=0)
    r = S["sim"][c]["random128"]  # original view: chance level at K=128 only
    ax[1].errorbar([128], [r[0]], yerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt="s", ms=8, mfc="white", mec=col, ecolor=col,
                   label=f"{c}: 128 random concepts")
ax[1].set_xscale("log", base=2)
ax[1].set_xticks(KS, [str(k) for k in KS])
ax[1].set_xlabel("K (number of concepts)")
ax[1].set_ylabel("Violation  ||g − P_K g||² / ||g||²")
ax[1].set_ylim(0, None)
ax[1].set_title(f"(b) SIM symmetry-II violation (n={len(sim)})", loc="left", color=INK)
ax[1].legend(frameon=False, fontsize=8.5)

# (c) information check: log p(target)
for i, c in enumerate(CONDS):
    v = lp[c]
    ax[2].scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=INK2, alpha=0.25, lw=0)
    m, lo, hi = S[c]["logp"]
    ax[2].errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=9, color=C_KNOWN, capsize=0, lw=2)
ax[2].set_xticks(x, [LABEL[c] for c in CONDS])
ax[2].set_ylabel("log p(masked object word)")
ax[2].set_title("(c) How much each context helps", loc="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(out, "m0_pilot.png"), dpi=160)

# ---------------------------------------------------------------- normalized SIM view (needs random-K per K)
if PER_K:
    C_MATCH = "#1baf7a"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    series = [("IMG", C_KNOWN, "Image context (all tokens)"), ("TXT", C_DISC, "Text context")]
    # (a) raw violation: top-K (solid) vs random-K chance level (dashed)
    for c, col, name in series:
        for key, ls, mk, lab in (("top", "-", "o", "top-K active concepts"), ("random", "--", "s", "K random concepts")):
            m = [S["sim"][c][f"{key}{k}"] for k in KS]
            ax[0].plot(KS, [v[0] for v in m], ls, color=col, lw=2, marker=mk, ms=7,
                       mfc=col if key == "top" else "white", label=f"{c}: {lab}")
            ax[0].fill_between(KS, [v[1] for v in m], [v[2] for v in m], color=col, alpha=0.12, lw=0)
    ax[0].set_ylabel("Violation  ||g − P_K g||² / ||g||²")
    ax[0].set_title("(a) Violation vs chance level at every K", loc="left", color=INK)

    # (b) explained beyond chance
    ser_b = series + ([("IMG_matched", C_MATCH, "Image context, size-matched")] if matched else [])
    for c, col, name in ser_b:
        m = [S["sim"][c][f"explained_beyond_chance@{k}"] for k in KS]
        ax[1].plot(KS, [v[0] for v in m], "-", color=col, lw=2, marker="o", ms=8, label=name)
        ax[1].fill_between(KS, [v[1] for v in m], [v[2] for v in m], color=col, alpha=0.15, lw=0)
    ax[1].axhline(0, color=INK2, lw=1)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].set_ylabel("Explained beyond chance  1 − V_top / V_random")
    ax[1].set_title("(b) How much better named concepts do than chance", loc="left", color=INK)

    # (c) paired difference IMG − TXT (H1 predicts < 0)
    ser_c = [("paired EBC IMG-TXT", C_KNOWN, "IMG − TXT")]
    if matched:
        ser_c.append(("paired EBC IMG_matched-TXT", C_MATCH, "IMG (size-matched) − TXT"))
    for key, col, name in ser_c:
        m = [S["sim"][key][k] for k in KS]
        ax[2].errorbar(KS, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]],
                       fmt="-o", color=col, lw=2, ms=8, capsize=0, label=name)
    ax[2].axhline(0, color=INK2, lw=1)
    ax[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[2].set_ylabel("Paired difference in explained beyond chance")
    ax[2].set_title("(c) Image vs text (H1 predicts < 0)", loc="left", color=INK)
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(KS, [str(k) for k in KS])
        a.set_xlabel("K (number of concepts)")
        a.legend(frameon=False, fontsize=8.5, loc="best")
    ax[0].set_ylim(0, None)
    ax[1].set_ylim(min(0, ax[1].get_ylim()[0]) - 0.05, ax[1].get_ylim()[1] + 0.05)
    fig.suptitle(f"SIM symmetry-II, normalized (n={len(sim)} samples)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "m0_sim_normalized.png"), dpi=160)
print(json.dumps(S, indent=1))
