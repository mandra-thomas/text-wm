"""How the lexsem rules (wf/lexsem_rules.py) read planted scenarios at a given precision (notes/lexsem_design.md
section 9). Each relative lift is drawn independently around its true value with the given 95% half-width (the
contrasts share conditions, so real errors are correlated; this is an approximation), and the single-account
reading is tallied. No score is read.

    python scripts/lexsem/simulate_rules.py --half-widths 0,0.12,0.18,0.24 --n 4000
"""
from __future__ import annotations

import argparse
import collections
import random

from wf import lexsem_rules as R

SCENARIOS = {
    "pure lexical": ({"lexical": 1}, 0.0, True),
    "pure semantic": ({"semantic": 1}, 0.0, True),
    "pure evidence": ({"evidence": 1}, 0.0, True),
    "pure evidence, denial lowers": ({"evidence": 1}, -0.5, True),
    "half lexical, half evidence": ({"lexical": .5, "evidence": .5}, 0.0, False),
    "half semantic, half evidence": ({"semantic": .5, "evidence": .5}, 0.0, False),
    "30% semantic, 70% evidence": ({"semantic": .3, "evidence": .7}, 0.0, False),
    "15% semantic, 85% evidence": ({"semantic": .15, "evidence": .85}, 0.0, False),
    "half lexical, half semantic": ({"lexical": .5, "semantic": .5}, 0.0, False),
    "70% lexical, 30% semantic": ({"lexical": .7, "semantic": .3}, 0.0, False),
}


def short(reading: str) -> str:
    for key in ("only lexical", "only semantic", "only evidence", "lexical not contradicted",
                "semantic not contradicted", "evidence not contradicted", "no single account", "not distinguished",
                "not assessable"):
        if reading.startswith(key):
            return key
    return reading


def run(half: float, n: int, a_true: float = 0.95, seed: int = 20260924) -> dict[str, collections.Counter]:
    rng = random.Random(seed)
    sd = half / 1.96 if half else 0.0
    out = {}
    for name, (w, deny, _pure) in SCENARIOS.items():
        true = R.planted(w, deny)
        c = collections.Counter()
        for _ in range(n if half else 1):
            iv = {}
            for k, v in true.items():
                p = v + (rng.gauss(0, sd) if sd else 0.0)
                iv[k] = (p - half, p + half)
            a = a_true + (rng.gauss(0, sd) if sd else 0.0)
            c[short(R.single_account_reading(iv, (a - half, a + half)))] += 1
        out[name] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-widths", default="0,0.12,0.18,0.24")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    for h in (float(x) for x in args.half_widths.split(",")):
        res = run(h, args.n)
        print(f"\n=== 95% half-width {h:.2f} ===")
        for name, c in res.items():
            tot = sum(c.values())
            print(f"{name:32s} " + "; ".join(f"{k} {100 * v / tot:.0f}%" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
