#!/usr/bin/env python3
"""Stage 1b precision and power (notes/decisionsim_agent_design.md, "Precision", amended after review 1). CPU only, no
model: simulates the primary (aware - blind over all states) and the key secondary (the same on encounter states) on
the real game structure of the 134 RQ3 no-foresight episodes.

Encounter states are put in a fixed number of games, drawn with weights favouring long failing games (games are
weighted by their state count times one plus their failing greedy actions, as the archived encounters are); among
encounter states an aware-blind difference is harm with probability `harm` and help with probability `help`; every
other state changes outcome in either direction with probability `bg` each (symmetric background). Inference as in
the analysis: game-clustered percentile bootstrap and whole-game sign-flip permutation. Holm over the two priority
models: with the other model null, the first threshold is p < 0.025; both p < 0.05 is the second.

  python scripts/decisionsim_agent_precision.py <RQ3 run folder> [dry-run metrics.json] [--reps 300]

The dry-run metrics (decisionsim_agent_dryrun) add the rank-1-only scenario: the archived greedy regime states, game by
game.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

NOTHING = "Nothing happens."


def p_min(g: int) -> float:
    """The smallest two-sided sign-flip p with g games whose net difference is not zero."""
    return 2.0 ** (1 - g) if g else 1.0


def min_games_for(alpha: float) -> int:
    g = 1
    while p_min(g) >= alpha:
        g += 1
    return g


def load(run: Path) -> tuple[list[str], list[int], list[int]]:
    eps = [json.loads(line) for f in sorted(glob.glob(str(run / "shard_*" / "episodes.jsonl"))) for line in open(f)]
    none = sorted((e for e in eps if e["condition"] == "none"), key=lambda e: e["game_id"])
    games = [e["game_id"] for e in none]
    sizes = [len(e["trace"]) for e in none]
    fails = [sum(s["real_obs"].strip() == NOTHING for s in e["trace"]) for e in none]
    return games, sizes, fails


def simulate(sizes, fails, enc_games, enc_mean, harm, help_, bg=0.005, reps=300, B=1000, P=1000, seed=1,
             fixed_k=None):
    import numpy as np

    rng = np.random.default_rng(seed)
    sizes = np.array(sizes)
    G = len(sizes)
    w = sizes * (1 + np.array(fails))
    out = []
    for _ in range(reps):
        if fixed_k is not None:
            k = np.minimum(np.array(fixed_k), sizes)
        else:
            eg = rng.choice(G, size=enc_games, replace=False, p=w / w.sum())
            k = np.zeros(G, int)
            k[eg] = np.minimum(sizes[eg], np.maximum(1, rng.poisson(enc_mean, len(eg))))
        enc = rng.binomial(k, harm) - rng.binomial(k, help_)
        back = rng.binomial(sizes - k, bg) - rng.binomial(sizes - k, bg)
        tot = enc + back
        row = []
        for t, n in ((tot.astype(float), sizes), (enc.astype(float), k)):
            keep = n > 0
            t, n = t[keep], n[keep]
            g = len(t)
            est = t.sum() / n.sum()
            idx = rng.integers(0, g, size=(B, g))
            bs = t[idx].sum(1) / n[idx].sum(1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            perm = np.abs((rng.choice([-1, 1], size=(P, g)) * t).sum(1)) / n.sum()
            p = ((perm >= abs(est) - 1e-15).sum() + 1) / (P + 1)
            row += [est, (hi - lo) / 2, lo > 0 and p < 0.025, lo > 0 and p < 0.05, int((t != 0).sum())]
        row.append(int(k.sum()))
        out.append(row)
    o = np.array(out, float)
    names = ["delta", "halfwidth", "power_holm_first", "power_p05", "games_nonzero"]
    res = {f"overall_{n}": float(o[:, i].mean()) for i, n in enumerate(names)}
    res.update({f"conditional_{n}": float(o[:, 5 + i].mean()) for i, n in enumerate(names)})
    res["encounter_states"] = float(o[:, 10].mean())
    for k in ("overall_delta", "overall_halfwidth", "conditional_delta", "conditional_halfwidth"):
        res[k] *= 100  # points
    return res


SCENARIOS = [  # (encounter games, mean encounter states per encounter game, harm, help)
    (15, 15, 0.10, 0.02), (15, 15, 0.20, 0.03),
    (30, 8, 0.10, 0.02), (30, 20, 0.10, 0.02), (30, 20, 0.20, 0.03),
    (60, 8, 0.10, 0.02), (60, 8, 0.20, 0.03),
]


def main(argv: list[str]) -> int:
    reps = 300
    if "--reps" in argv:
        reps = int(argv[argv.index("--reps") + 1])
        argv = [a for i, a in enumerate(argv) if a != "--reps" and (i == 0 or argv[i - 1] != "--reps")]
    games, sizes, fails = load(Path(argv[0]))
    print(f"{len(games)} games, {sum(sizes)} states; sign-flip floor p_min(g) = 2^(1-g): "
          f"{min_games_for(0.025)} games needed for p < 0.025, {min_games_for(0.05)} for p < 0.05")
    rows = []
    if len(argv) > 1:
        per = json.load(open(argv[1]))["per_game"]
        k = [per[g]["regime_rank1"] for g in games]
        for harm, help_ in ((0.10, 0.02), (0.20, 0.03)):
            r = simulate(sizes, fails, 0, 0, harm, help_, reps=reps, fixed_k=k)
            rows.append({"scenario": f"rank 1 only ({sum(k)} states in {sum(x > 0 for x in k)} games)", "harm": harm,
                         "help": help_, **r})
    for eg, m, harm, help_ in SCENARIOS:
        r = simulate(sizes, fails, eg, m, harm, help_, reps=reps)
        rows.append({"scenario": f"{eg} games, about {m} encounter states each", "harm": harm, "help": help_, **r})
    for r in rows:
        print(f"{r['scenario']:<44} harm {r['harm']:.2f} help {r['help']:.2f} | encounter states {r['encounter_states']:.0f} | "
              f"overall {r['overall_delta']:.2f} pts ±{r['overall_halfwidth']:.2f}, power {r['overall_power_holm_first']:.2f} "
              f"(p<.05: {r['overall_power_p05']:.2f}) | conditional {r['conditional_delta']:.1f} pts "
              f"±{r['conditional_halfwidth']:.1f}, power {r['conditional_power_holm_first']:.2f}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
