"""Synthetic check of the mechanism battery's decision rules on the real item structure (no GPU, no scores).

    RUNS_DIR=<runs with rq12_transitions_s*> python scripts/mech2_simulate.py REPS B [scenario,...]

Plants a pattern of size ratios r (each arm against the swapped task, as a fraction of the aligned effect) in synthetic
scores for the 1,403 real items of the battery, both formats, with each model's spread and within-game correlation from
the earlier runs (two-candidate: rq1x_v2 `named_goal`; yes/no: rq1v_v2 `goal_effect_failures`), and reads them with
`wf/mech2.py`'s primary families, categories and readouts (as amended after review 2). Reports, per scenario and model:

- each contrast's categories in each format;
- per account, how often it is contradicted on a contrast where the scenario's TRUE fingerprint (the account's intended
  pattern, without the artefacts a scenario adds) lies in a zone the account allows: a false contradiction, in each
  format alone and under the two-format rule (where a contrast counts as truly allowed when either format's true value
  is allowed);
- the joint verdicts: which content accounts are consistent, untested and contradicted, how often the planted account
  is the sole survivor (the only content account not contradicted, and consistent), how many (account, contrast)
  readings are decided, and how many are format-dependent;
- for mixtures, the categories and mean r of P5, P7 and P8 (the readout fixed for mixtures).

A scenario plants one fingerprint in both formats, or one per format (formats that disagree), and may name the account
it is meant to show and a `truth` fingerprint to judge contradictions against (when it adds artefacts such as a gap in
the desire lines' generic lift or word copying in one format).
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
KEYS = ("desire_keep", "desire_keep_sw", "desire_check", "desire_check_can", "mention_ignore", "described", "done",
        "cancelled", "negated", "affirmed", "p6")
ZERO = dict.fromkeys(KEYS, 0.0)


def fp(**kw) -> dict[str, float]:
    out = {**ZERO, **kw}
    assert set(out) == set(KEYS), kw
    return out


def every(r: float) -> dict[str, float]:
    """Every primary contrast at r: P1 = keep_sw - ignore, P2 = check - check_can, P5 = done - cancelled,
    P8 = negated - affirmed; the rest are arms against swapped."""
    return fp(desire_keep=r, desire_keep_sw=2 * r, desire_check=r, desire_check_can=0, mention_ignore=r, described=r,
              done=r, cancelled=0, negated=r, affirmed=0, p6=r)


# Each account's intended fingerprint, every value well inside its predicted zones (section 12 of the design note).
FP = {
    "H2_evidence": fp(desire_keep=.6, desire_keep_sw=.7, desire_check=.5, desire_check_can=.5, mention_ignore=.6,
                      described=.8, done=0, cancelled=.6, negated=-.5, affirmed=.7),
    "H3_salience": fp(desire_keep=.7, desire_keep_sw=.7, desire_check=.7, desire_check_can=.7, mention_ignore=.7,
                      described=.6, done=.7, cancelled=.7, negated=.7, affirmed=.7),
    "H4_wishful": fp(desire_keep=-.5, desire_keep_sw=-.5, desire_check=-.5, desire_check_can=.4, mention_ignore=0,
                     described=.9),
    "H5_policy_blend": fp(desire_keep=-.4, desire_keep_sw=-.4, desire_check=.7, desire_check_can=.7, mention_ignore=0,
                          described=.9),
    "H6_lexical": fp(desire_keep=.7, desire_keep_sw=.7, desire_check=.7, desire_check_can=.7, mention_ignore=.7,
                     described=0, done=.7, cancelled=.7, negated=.7, affirmed=.7),
}
GAP = 0.5  # review 2: the bare desire lines lack the swapped task's generic lift (g/D = 0.42 Qwen, 0.73 Llama)


def gap(f: dict[str, float]) -> dict[str, float]:
    return {**f, **{k: f[k] - GAP for k in ("desire_keep", "desire_check", "desire_check_can")}}


def mix(a: dict, b: dict, w: float = 0.5) -> dict[str, float]:
    return {k: w * a[k] + (1 - w) * b[k] for k in KEYS}


SCENARIOS: dict[str, dict] = {
    "noise": {"cand": ZERO},
    "trivial10": {"cand": every(.10)},
    "trivialneg10": {"cand": every(-.10)},
    # review 2: boundaries. -0.20 sits inside the band near its edge (truth Z); 0.25 is the band's edge (truth Z);
    # 0.30 and 0.45 are small (S); 0.50 is the large threshold (truth L)
    "boundary_neg20": {"cand": every(-.20)},
    "boundary_25": {"cand": every(.25)},
    "boundary_30": {"cand": every(.30)},
    "boundary_45": {"cand": every(.45)},
    "boundary_50": {"cand": every(.50)},
    **{h: {"cand": f, "account": h} for h, f in FP.items()},
    "H6_position": {"cand": {**FP["H3_salience"], "p6": -.5}, "account": "H6_position"},
    # review 2: the desire-arm gap on the new P1 and P2 (desire_keep_sw keeps the swapped task, so P1 has no gap;
    # desire_check and desire_check_can share it, so it cancels in P2)
    "H3_desire_gap": {"cand": gap(FP["H3_salience"]), "truth": FP["H3_salience"], "account": "H3_salience"},
    "H5_desire_gap": {"cand": gap(FP["H5_policy_blend"]), "truth": FP["H5_policy_blend"], "account": "H5_policy_blend"},
    # formats that disagree: (a) different accounts per format; (b) H2 with word copying in the two-candidate format
    # only (the receptacle named in negated and affirmed: +0.6 on both); (c) H3 with a yes/no answer bias toward "No"
    # after "cannot" (desire_check -0.4 in yes/no only)
    "formats_H3_cand_H2_yn": {"cand": FP["H3_salience"], "yn": FP["H2_evidence"]},
    "H2_copying_cand": {"cand": {**FP["H2_evidence"], "negated": FP["H2_evidence"]["negated"] + .6,
                                 "affirmed": FP["H2_evidence"]["affirmed"] + .6},
                        "yn": FP["H2_evidence"], "truth": FP["H2_evidence"], "account": "H2_evidence"},
    "H3_no_bias_yn": {"cand": FP["H3_salience"], "yn": {**FP["H3_salience"], "desire_check": FP["H3_salience"]["desire_check"] - .4},
                      "truth": FP["H3_salience"], "account": "H3_salience"},
    # review 2: the 50/50 mixture of H2 and H3
    "mix_H2_H3": {"cand": mix(FP["H2_evidence"], FP["H3_salience"]), "mixture": True},
}


def true_r(f: dict) -> dict[str, float]:
    """Each primary contrast's r under a fingerprint."""
    return {"P1_keep_vs_ignore": f["desire_keep_sw"] - f["mention_ignore"],
            "P2_hope_polarity": f["desire_check"] - f["desire_check_can"],
            "P3_mention_without_need": f["mention_ignore"], "P4_need_without_name": f["described"],
            "P5_done_vs_cancelled": f["done"] - f["cancelled"], "P6_position_step0": f["p6"],
            "P7_negated_existence": f["negated"], "P8_negated_vs_affirmed": f["negated"] - f["affirmed"]}


def synth(union: list[dict], fps: dict[str, dict], params: dict, rng: random.Random) -> list[dict]:
    out = []
    games = sorted({t["game_id"] for t in union})
    for fmt, (D, sd, icc) in params.items():
        f = fps[fmt]
        e_sd = sd / 2 ** 0.5  # each arm's own noise, so a difference of two arms spreads as the effect did
        g_sd, i_sd = e_sd * icc ** 0.5, e_sd * (1 - icc) ** 0.5
        ge = {(g, a): rng.gauss(0, g_sd) for g in games for a in mech2.ARMS}
        for t in union:
            b = rng.gauss(-3, 3)
            di = D * (1 + rng.gauss(0, 0.3))
            for a in mech2.ARMS:
                if mech2.task_line(t, a) is None:
                    continue
                r = (1.0 if a == "aligned" else 1.0 + f["p6"] if a == "aligned_start" else
                     0.0 if a in ("swapped", "swapped_start", "neutral") else
                     f["mention_ignore"] if a == "mention_exists" else f[a])
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
            sc = SCENARIOS[name]
            fps = {"cand": sc["cand"], "yn": sc.get("yn", sc["cand"])}
            truth = {f: {c: mech2.zone(v) for c, v in true_r(sc.get("truth", fps[f])).items()} for f in ("cand", "yn")}
            cats = {f: defaultdict(Counter) for f in ("cand", "yn")}
            false_contra = {k: Counter() for k in ("cand", "yn", "joint")}  # account -> replications with one
            verdicts, sole, decided, fdep = Counter(), 0, Counter(), Counter()
            mix_r = defaultdict(list)
            for rep in range(reps):
                rng = random.Random(f"{model}|{name}|{rep}")
                by = mech2._lookup(synth(sets["union"], fps, params, rng))
                fam = {f: mech2.primary_family(by, pops, f, B=B, perms=B) for f in ("cand", "yn")}
                joint = mech2.joint_readout(fam["cand"]["readout"], fam["yn"]["readout"])
                for f in ("cand", "yn"):
                    for c, r in fam[f]["contrasts"].items():
                        cats[f][c][r["category"]] += 1
                for c in ("P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed"):
                    mix_r[c].append(fam["cand"]["contrasts"][c]["ratio"]["estimate"])
                for h, preds in mech2.PREDICTIONS.items():
                    for k, ro in (("cand", fam["cand"]["readout"]), ("yn", fam["yn"]["readout"]), ("joint", joint)):
                        fs = ("cand", "yn") if k == "joint" else (k,)
                        ok = [c for c in preds if any(truth[f][c] in preds[c] for f in fs)]
                        if any(ro[h]["per_contrast"][c] == "contradicted" for c in ok):
                            false_contra[k][h] += 1
                    for c in preds:
                        v = joint[h]["per_contrast"][c]
                        decided["decided" if v in ("consistent", "contradicted") else v] += 1
                    fdep[h] += len(joint[h]["format_dependent"])
                s = joint["summary"]
                verdicts[f"consistent={','.join(s['content_accounts_consistent']) or '-'} "
                         f"untested={','.join(s['content_accounts_untested']) or '-'}"] += 1
                acc = sc.get("account")
                if acc == "H6_position":
                    sole += joint["H6_position"]["verdict"] == "consistent"
                elif acc:
                    sole += s["content_accounts_not_contradicted"] == [acc] and joint[acc]["verdict"] == "consistent"
            line = " ".join(f"{c[:2]}={z}" for c, z in truth["cand"].items())
            if truth["yn"] != truth["cand"]:
                line += " | yes/no " + " ".join(f"{c[:2]}={z}" for c, z in truth["yn"].items())
            print(f"== {model} {name}  (true zones: {line})")
            for c in mech2.PRIMARY:
                print(f"   {c:26s} cand {dict(cats['cand'][c])}  yn {dict(cats['yn'][c])}")
            print("   false contradictions (cand / yn / both formats): " + "; ".join(
                f"{h} {false_contra['cand'][h]}/{false_contra['yn'][h]}/{false_contra['joint'][h]}" for h in mech2.PREDICTIONS))
            n_read = sum(decided.values())
            print(f"   joint readings decided {decided['decided']}/{n_read}, undecided {decided['undecided']}, "
                  f"format-dependent {decided['format_dependent']} (per account {dict(fdep)})")
            if sc.get("account"):
                print(f"   sole survivor ({sc['account']} the only content account not contradicted, and consistent): {sole}/{reps}")
            if sc.get("mixture") or name.startswith("mix"):
                print("   mixture readout, mean r (cand): " + ", ".join(
                    f"{c[:2]} {sum(v) / len(v):+.2f}" for c, v in mix_r.items()))
            print("   verdicts (both formats):", dict(verdicts), flush=True)


if __name__ == "__main__":
    main()
