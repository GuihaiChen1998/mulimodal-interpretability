"""Statistics shared by the M0 analysis scripts (samples are the unit; 95% bootstrap CIs over samples)."""

from __future__ import annotations

import numpy as np

CONDS = ("IMG", "TXT", "NONE")


def known_frac(d: dict) -> float:
    """Share of |target logit| carried by known concepts, averaged over the target's tokens."""
    a, b, c = (np.abs(np.array(d[k])) for k in ("known_c", "disc_c", "eps_c"))
    return float(np.mean(a / (a + b + c)))


def confidence_controlled(recs: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """known_frac ~ 1 + logp + [IMG] + [NONE] over all (sample, condition) rows, TXT as baseline.
    Returns {name: [estimate, lo, hi]} for the logp slope and the two condition effects (fractions)."""
    X = np.array([[1, np.mean(r[c]["logp"]), c == "IMG", c == "NONE"] for r in recs for c in CONDS], float)
    y = np.array([known_frac(r[c]) for r in recs for c in CONDS])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    n = len(recs)
    boots = np.array([np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
                      for rows in ((3 * rng.integers(0, n, n)[:, None] + np.arange(3)).ravel() for _ in range(n_boot))])
    out = {}
    for name, j in (("logp_slope", 1), ("IMG_vs_TXT", 2), ("NONE_vs_TXT", 3)):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        out[name] = [float(beta[j]), float(lo), float(hi)]
    out["raw_known_frac"] = {c: float(y[k::3].mean()) for k, c in enumerate(CONDS)}
    return out
