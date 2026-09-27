"""Known-concept share controlled for prediction confidence.

The known share rises with log p(target) in every condition, and the conditions differ in how informative
their context is (TXT > IMG > NONE for a well-trained connector). So the raw IMG-TXT gap in known share
mixes "modality" with "confidence". Fit, over all (sample, condition) rows:
    known_frac ~ 1 + logp + [cond == IMG] + [cond == NONE]      (TXT is the baseline)
and report coefficients with 95% bootstrap CIs over samples.

Usage: python m0_confidence_control.py RECORDS_JSON [RECORDS_JSON ...]
"""

import json
import sys

import numpy as np

CONDS = ["IMG", "TXT", "NONE"]


def known_frac(d):
    a, b, c = (np.abs(np.array(d[k])) for k in ("known_c", "disc_c", "eps_c"))
    return float(np.mean(a / (a + b + c)))


for path in sys.argv[1:]:
    recs = json.load(open(path))
    X = np.array([[1, np.mean(r[c]["logp"]), c == "IMG", c == "NONE"] for r in recs for c in CONDS], float)
    y = np.array([known_frac(r[c]) for r in recs for c in CONDS])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0)
    n = len(recs)
    boots = []
    for _ in range(2000):
        rows = (3 * rng.integers(0, n, n)[:, None] + np.arange(3)).ravel()
        boots.append(np.linalg.lstsq(X[rows], y[rows], rcond=None)[0])
    boots = np.array(boots)
    print(path)
    print("  raw known share IMG/TXT/NONE: " + " / ".join(f"{y[k::3].mean() * 100:.1f}%" for k in range(3)))
    for name, j in (("logp slope (per nat)", 1), ("IMG vs TXT", 2), ("NONE vs TXT", 3)):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5]) * 100
        print(f"  {name}: {beta[j] * 100:+.2f}pp [{lo:+.2f}, {hi:+.2f}]")
