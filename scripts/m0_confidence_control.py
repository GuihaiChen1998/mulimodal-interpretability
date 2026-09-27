"""Known-concept share controlled for prediction confidence.

The known share rises with log p(target) in every condition, and the conditions differ in how informative
their context is (TXT > IMG > NONE for a well-trained connector). So the raw IMG-TXT gap in known share
mixes "modality" with "confidence". Fit, over all (sample, condition) rows:
    known_frac ~ 1 + logp + [cond == IMG] + [cond == NONE]      (TXT is the baseline)
and report coefficients with 95% bootstrap CIs over samples (smm/m0_stats.py).

Usage: python m0_confidence_control.py RECORDS_JSON [RECORDS_JSON ...]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from smm.m0_stats import confidence_controlled  # noqa: E402

for path in sys.argv[1:]:
    r = confidence_controlled(json.load(open(path)))
    print(path)
    print("  raw known share IMG/TXT/NONE: " + " / ".join(f"{v * 100:.1f}%" for v in r["raw_known_frac"].values()))
    for name, key in (("logp slope (per nat)", "logp_slope"), ("IMG vs TXT", "IMG_vs_TXT"),
                      ("NONE vs TXT", "NONE_vs_TXT")):
        b, lo, hi = r[key]
        print(f"  {name}: {b * 100:+.2f}pp [{lo * 100:+.2f}, {hi * 100:+.2f}]")
