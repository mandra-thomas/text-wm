"""How the lexsem procedure reads planted truths (notes/lexsem_design.md section 9). No score is read.

Two modes.

- `rules`: each relative lift drawn independently around its planted value with a given 95% half-width, read by
  wf/lexsem_rules.py alone. Quick, but it ignores that contrasts share items and controls.
- `procedure` (the validation the design relies on): item-level scores are generated for a whole stratum (300 later
  items in 108 games, or 150 step-0 items in 96 games) and read by the frozen analysis itself
  (wf/lexsem_analysis.analyse: game bootstrap, gates, reference rule, rules). The generator has
  * a goal effect per item with the plain-text first-token mean and spread (1.31 nats, SD 1.10; Qwen audit sample),
  * an item-level mention response correlated 0.45 with the goal effect (stage 0's item-level correlation) and a game
    component, shared by every mention condition of the item (so contrasts are correlated through items),
  * one control per carrier shared by every condition that uses it (so contrasts are correlated through controls),
  * unequal carrier noise across conditions (0.35 nats for most, 0.45 for the listed word, 0.55 for descriptions,
    0.5 for the alias and homonym),
  * a grid of reference sizes: the name assertion lifting 0.95, 0.5 or 0.3 of the goal effect.

    python scripts/lexsem/simulate_rules.py procedure --reps 60 --B 200
"""
from __future__ import annotations

import argparse
import collections
import math
import random

from wf import lexsem
from wf import lexsem_rules as R

SCENARIOS = {
    "pure lexical": ({"lexical": 1}, {}),
    "pure semantic": ({"semantic": 1}, {}),
    "pure semantic, homonym activates the concept at 0.4": ({"semantic": 1}, {"homonym": 0.4}),
    "pure evidence": ({"evidence": 1}, {}),
    "partial-strength evidence (claims at 0.7)": ({"evidence": 1}, {"desc_assert": 0.7, "alias_assert": 0.7}),
    "85% evidence, 15% semantic": ({"evidence": .85, "semantic": .15}, {}),
    "70% evidence, 30% semantic": ({"evidence": .7, "semantic": .3}, {}),
    "half semantic, half evidence": ({"semantic": .5, "evidence": .5}, {}),
    "half lexical, half semantic": ({"lexical": .5, "semantic": .5}, {}),
    "half lexical, half evidence": ({"lexical": .5, "evidence": .5}, {}),
    "unlisted: every mention lifts alike": (None, {}),
}
NOISE = {"name_use": 0.45, "ctrl_use": 0.45, "desc_assert": 0.55, "cdesc_assert": 0.55, "desc_use": 0.55,
         "cdesc_use": 0.55, "alias_assert": 0.5, "homonym": 0.5, "ctrl_homonym": 0.5}
BASE_NOISE = 0.35
# Condition-specific item heterogeneity, in item-SDs of the mention response. 0.5 gives relative-lift half-widths of
# about 0.06 to 0.10 on the later stratum (optimistic); 1.75 gives about 0.17 to 0.18 (the planning value; section 9).
TAU = 1.75


def short(reading: str) -> str:
    for key in ("only lexical", "only semantic", "only evidence", "lexical not contradicted",
                "semantic not contradicted", "evidence not contradicted", "no single proposed account",
                "not distinguished", "not assessable"):
        if reading.startswith(key):
            return key
    return reading


def true_rho(weights: dict | None, override: dict) -> dict[str, float]:
    if weights is None:
        rho = {k: 1.0 for k in R.PURE["lexical"]}
    else:
        rho = R.planted(weights)
    return {**rho, **override}


def generate(rho: dict[str, float], a_ref: float, n_items: int, n_games: int, rng: random.Random,
             tau: float = TAU) -> list[dict]:
    """Item records in the scorer's format (first-token readout only; mass 0.9 throughout)."""
    conds = lexsem.GROUPS["core"]
    objs = ("mug", "pan", "saltshaker", "apple", "bowl", "cd", "tomato")  # homonym, alias, spelling, plain mixes
    game_r = {g: rng.gauss(0, 0.3) for g in range(n_games)}
    recs = []
    for i in range(n_items):
        gm = i % n_games
        obj = objs[i % len(objs)]
        zg, zr = rng.gauss(0, 1), rng.gauss(0, 1)
        g = 1.31 + 1.10 * zg
        r = a_ref * (1.31 + 1.10 * (0.45 * zg + math.sqrt(1 - 0.45 ** 2) * zr) + game_r[gm])
        b = rng.gauss(-3, 1)
        y = {"swapped": b, "aligned": b + g}
        for c in conds:
            if c.startswith(("ctrl", "cdesc")):
                y[c] = b + 0.2 + rng.gauss(0, NOISE.get(c, BASE_NOISE))
        y["name_assert"] = y["ctrl_assert"] + r + rng.gauss(0, BASE_NOISE)
        for c in conds:
            if c in y or c in lexsem.ANCHORS:
                continue
            form = lexsem.MENTIONS[c][0]
            if (form == "variant" and obj not in lexsem.VARIANTS) or (form == "alias" and obj not in lexsem.ALIASES) \
                    or (form == "homonym" and obj not in lexsem.HOMONYMS):
                continue
            het = rng.gauss(0, tau * a_ref * 1.10)  # this condition's own item-level response
            y[c] = y[lexsem.CONTROL_OF[c]] + rho.get(c, 1.0) * r + het + rng.gauss(0, NOISE.get(c, BASE_NOISE))
        scores = {c: {"obs": {"lp_a": math.log(0.9) + v - math.log1p(math.exp(v)),
                              "lp_b": math.log(0.9) - math.log1p(math.exp(v))}} for c, v in y.items()}
        recs.append({"id": f"s{i}", "game_id": f"g{gm}", "goal_object": obj, "fewshot_object": False,
                     "stratum": "later", "scores": scores})
    return recs


def procedure(reps: int, B: int, sizes: tuple[float, ...], n_items: int, n_games: int, seed: int,
              tau: float = TAU) -> dict:
    from wf import lexsem_analysis as LA

    LA.NUMERICAL_FLOOR = 0.0  # statistical behaviour only; numerical resolution is a separate gate (design section 7)
    out = {}
    for name, (w, override) in SCENARIOS.items():
        rho = true_rho(w, override)
        for a in sizes:
            rng = random.Random(f"{seed}|{name}|{a}")
            c = collections.Counter()
            false_contra = 0
            for k in range(reps):
                recs = generate(rho, a, n_items, n_games, rng, tau)
                boot = LA.Boot([r["game_id"] for r in recs], B, seed + k)
                res = LA.analyse(recs, "first", boot, None, "later")
                c[short(res["single_account_reading"])] += 1
                if w is not None and len(w) == 1:
                    acc = next(iter(w))
                    false_contra += res.get("accounts", {}).get(acc, {}).get("overall") == "contradicted"
            out[(name, a)] = (c, false_contra)
    return out


def rules(half: float, n: int, seed: int = 20260924) -> dict:
    rng = random.Random(seed)
    sd = half / 1.96
    out = {}
    for name, (w, override) in SCENARIOS.items():
        true = true_rho(w, override)
        c = collections.Counter()
        for _ in range(n):
            pts = {k: v + rng.gauss(0, sd) for k, v in true.items()}
            iv = {k: (p - half, p + half) for k, p in pts.items()}
            a = 0.95 + rng.gauss(0, sd)
            c[short(R.single_account_reading(iv, (a - half, a + half)))] += 1
        out[name] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("rules", "procedure"))
    ap.add_argument("--half-widths", default="0.12,0.18,0.24")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--sizes", default="0.95,0.5,0.3")
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--games", type=int, default=108)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()
    if args.mode == "rules":
        for h in (float(x) for x in args.half_widths.split(",")):
            print(f"\n=== 95% half-width {h:.2f} ===")
            for name, c in rules(h, args.n).items():
                tot = sum(c.values())
                print(f"{name:52s} " + "; ".join(f"{k} {100 * v / tot:.0f}%" for k, v in c.most_common()))
        return
    res = procedure(args.reps, args.B, tuple(float(x) for x in args.sizes.split(",")), args.items, args.games,
                    20260924, args.tau)
    for (name, a), (c, fc) in res.items():
        tot = sum(c.values())
        print(f"{name:52s} A={a:.2f} | " + "; ".join(f"{k} {100 * v / tot:.0f}%" for k, v in c.most_common())
              + (f" | true account contradicted {100 * fc / tot:.0f}%" if "pure" in name or "partial" in name else ""),
              flush=True)


if __name__ == "__main__":
    main()
