"""Synthetic check of the mechanism battery's decision rules on the real item structure (no GPU, no scores).

    RUNS_DIR=<runs with rq12_transitions_s*> python scripts/mech2_simulate.py REPS B [scenario,...]

Plants a pattern of size ratios r (each arm against the swapped task, as a fraction of the aligned effect) in synthetic
scores for the 1,403 real items of the battery, both formats, with each model's spread and within-game correlation from
the earlier runs (two-candidate: rq1x_v2 `named_goal`; yes/no: rq1v_v2 `goal_effect_failures`), and reads them with
`wf/mech2.py`'s primary families, categories and readouts (amendment after review 1). Reports, per scenario and model:

- each contrast's categories (two-candidate format);
- per account, how often it is contradicted on a contrast whose true r lies in a zone the account predicts (a false
  contradiction), in each format alone and under the two-format rule;
- the joint verdicts (which content accounts are not contradicted).

Scenarios: `noise` (every r = 0), `trivial10` and `trivialneg10` (every contrast at r = +0.10 or -0.10, inside the
band), and one planted fingerprint per account (values well inside each predicted zone).
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict

from wf import mech2, rq1
from wf.rq2 import SOURCES, load_items

MODELS = {  # (aligned effect D in nats, per-item SD of the effect, within-game correlation), per format
    "qwen": {"cand": (6.12, 6.0, 0.26), "yn": (2.80, 3.20, 0.10)},
    "llama": {"cand": (1.65, 1.94, 0.12), "yn": (2.21, 1.97, 0.19)},
}
ARMS = ("aligned", "swapped", "neutral", "desire_keep", "desire_check", "mention_ignore", "mention_exists", "described",
        "swapped_described", "done", "cancelled", "negated", "aligned_start", "swapped_start")
_C = dict(desire_keep=0, desire_check=0, mention_ignore=0, described=0, done=0, cancelled=0, negated=0, p6=0)
SCENARIOS = {
    "noise": dict(_C),
    # every primary contrast at +0.10 (P1 = keep - ignore = 0.10, P5 = done - cancelled = 0.10, P6 = +0.10)
    "trivial10": dict(desire_keep=.2, desire_check=.1, mention_ignore=.1, described=.1, done=.1, cancelled=0, negated=.1, p6=.1),
    "trivialneg10": dict(desire_keep=-.2, desire_check=-.1, mention_ignore=-.1, described=-.1, done=-.1, cancelled=0,
                         negated=-.1, p6=-.1),
    "H2_evidence": dict(desire_keep=.6, desire_check=.6, mention_ignore=.6, described=.8, done=0, cancelled=.6, negated=-.5, p6=0),
    "H3_salience": dict(desire_keep=.7, desire_check=.7, mention_ignore=.7, described=.6, done=.7, cancelled=.7, negated=.6, p6=0),
    "H4_wishful": dict(desire_keep=-.5, desire_check=-.5, mention_ignore=0, described=.9, done=0, cancelled=0, negated=0, p6=0),
    "H5_policy_blend": dict(desire_keep=-.4, desire_check=.7, mention_ignore=0, described=.9, done=0, cancelled=0, negated=0, p6=0),
    "H6_lexical": dict(desire_keep=.7, desire_check=.7, mention_ignore=.7, described=0, done=.7, cancelled=.7, negated=.7, p6=0),
    "H6_position": dict(desire_keep=.7, desire_check=.7, mention_ignore=.7, described=.6, done=.7, cancelled=.7, negated=.6, p6=-.5),
}


def true_r(fp: dict) -> dict[str, float]:
    """Each primary contrast's planted r."""
    return {"P1_keep_vs_ignore": fp["desire_keep"] - fp["mention_ignore"], "P2_desire_check": fp["desire_check"],
            "P3_mention_without_need": fp["mention_ignore"], "P4_need_without_name": fp["described"],
            "P5_done_vs_cancelled": fp["done"] - fp["cancelled"], "P6_position_step0": fp["p6"],
            "P7_negated_existence": fp["negated"]}


def synth(union: list[dict], fp: dict, params: dict, rng: random.Random) -> list[dict]:
    out = []
    games = sorted({t["game_id"] for t in union})
    for fmt, (D, sd, icc) in params.items():
        e_sd = sd / 2 ** 0.5  # each arm's own noise, so a difference of two arms spreads as the effect did
        g_sd, i_sd = e_sd * icc ** 0.5, e_sd * (1 - icc) ** 0.5
        ge = {(g, a): rng.gauss(0, g_sd) for g in games for a in ARMS}
        for t in union:
            b = rng.gauss(-3, 3)
            di = D * (1 + rng.gauss(0, 0.3))
            for a in ARMS:
                if mech2.alias_of(t, a) or mech2.task_line(t, a) is None:
                    continue
                r = (1.0 if a == "aligned" else 1.0 + fp["p6"] if a == "aligned_start" else
                     0.0 if a in ("swapped", "swapped_start", "neutral", "swapped_described") else
                     fp["mention_ignore"] if a == "mention_exists" else fp[a])
                if a == "aligned_start" and t.get("history_len", 0) > 0:
                    r = 1.0
                out.append({"transition_id": t["id"], "condition": mech2.condition(a, fmt),
                            "L": b + r * di + ge[(t["game_id"], a)] + rng.gauss(0, i_sd)})
    return out


def main() -> None:
    reps, B = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(SCENARIOS)
    primary = rq1.rq1_sets(load_items(SOURCES))["primary"]
    sets = mech2.battery_items(primary)
    pops = mech2.populations(sets["union"], sets["sample"], sets["step0"])
    for model, params in MODELS.items():
        for name in names:
            fp = SCENARIOS[name]
            truth = {c: mech2.zone(v) for c, v in true_r(fp).items()}
            cats = defaultdict(Counter)
            false_contra = {k: Counter() for k in ("cand", "yn", "joint")}  # account -> replications with one
            verdicts = Counter()
            for rep in range(reps):
                rng = random.Random(f"{model}|{name}|{rep}")
                by = mech2._lookup(synth(sets["union"], fp, params, rng))
                fam = {f: mech2.primary_family(by, pops, f, B=B, perms=B) for f in ("cand", "yn")}
                joint = mech2.joint_readout(fam["cand"]["readout"], fam["yn"]["readout"])
                for c, r in fam["cand"]["contrasts"].items():
                    cats[c][r["category"]] += 1
                for h, preds in mech2.PREDICTIONS.items():
                    true_ok = [c for c in preds if truth[c] in preds[c]]  # contrasts where the account is right
                    for k, ro in (("cand", fam["cand"]["readout"]), ("yn", fam["yn"]["readout"]), ("joint", joint)):
                        if any(ro[h]["per_contrast"][c] == "contradicted" for c in true_ok):
                            false_contra[k][h] += 1
                verdicts[",".join(joint["summary"]["content_accounts_not_contradicted"]) or "none"] += 1
            print(f"== {model} {name}  (true zones: {' '.join(f'{c[:2]}={z}' for c, z in truth.items())})")
            for c in mech2.PRIMARY:
                print(f"   {c:26s} {dict(cats[c])}")
            for h in mech2.PREDICTIONS:
                print(f"   false contradictions {h:16s} cand {false_contra['cand'][h]}/{reps}  yn {false_contra['yn'][h]}/{reps}"
                      f"  both formats {false_contra['joint'][h]}/{reps}")
            print("   content accounts not contradicted (both formats):", dict(verdicts), flush=True)


if __name__ == "__main__":
    main()
