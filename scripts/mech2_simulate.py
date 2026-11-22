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
- for mixtures, the categories and mean r of P5, P7 and P8 (the readout fixed for mixtures);
- after review 3, for every scenario: M, the mention-associated residual after averaging opposing claims (half of
  negated plus half of affirmed, minus swapped, on P7's items), per format (mean r and the zones of its interval), and
  its two halves; the negation placebo's zones and P8 net of it; how often each summary sentence is given; how often
  an account is untested because it was consistent only where it allows zero; and the most common summary text;
- after the adversarial check of M (2026-11-21): the verdicts of task as evidence and salience across replications,
  how often the summary attributes any of the effect to salience (it must never), and how often salience is called
  consistent next to task as evidence. The `asym_*` scenarios plant pure task as evidence with asymmetric responses to
  affirmations and denials (a grid, a denial-to-affirmation ratio family and a logistic saturation family), each with
  the placebo at zero and with a placebo effect (`_pl`: the placebo's "no" line 0.4 lower).

A scenario plants one fingerprint in both formats, or one per format (formats that disagree), and may name the account
it is meant to show and a `truth` fingerprint to judge contradictions against (when it adds artefacts such as a gap in
the desire lines' generic lift or word copying in one format).
"""
from __future__ import annotations

import math
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
        "cancelled", "negated", "affirmed", "placebo_negated", "placebo_affirmed", "p6")
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
    # review 3: salience drives the goal effect, and the model also reads explicit claims as any competent reader would
    # (the denial lowers, the affirmation raises, "done" lowers); and salience with a lean toward failure after negation
    # words ("not", "cannot", "no") in both formats, which the negation placebo shows (its "no" line 0.4 lower)
    "H3_plus_comprehension": {"cand": {**FP["H3_salience"], "negated": .1, "affirmed": .9, "done": .2, "cancelled": .7},
                              "truth": FP["H3_salience"]},
    "H3_negation_priming": {"cand": {**FP["H3_salience"], "desire_keep_sw": .3, "desire_check": .3, "negated": .3,
                                     "placebo_negated": -.4},
                            "truth": FP["H3_salience"]},
    # review 4: policy blending plus a lean toward failure at the word "no" (negated and the placebo's "no" line 0.4
    # lower), which makes P8 look like evidence
    "H5_negation_art": {"cand": {**FP["H5_policy_blend"], "negated": -.4, "placebo_negated": -.4},
                        "truth": FP["H5_policy_blend"], "account": "H5_policy_blend"},
}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# The adversarial check of M (2026-11-21): pure task as evidence (the H2 fingerprint, no salience, word overlap,
# wishful or blending component; the aligned-minus-swapped effect is 1) with the affirmed and negated arms set by an
# asymmetric response to the user's claim. The same pattern in both formats, no copying, no generic negation priming.
ASYM: dict[str, tuple[float, float]] = {}
for aff in (.6, 1.0, 1.4, 1.8):  # (i) a grid of the two responses
    for neg in (-1.0, -.6, -.2, 0.0):
        ASYM[f"asym_grid_a{aff:+.1f}_n{neg:+.1f}"] = (aff, neg)
for g in (.25, .5, 1.0, 2.0):  # (ii) the denial's response a multiple g of the affirmation's (0.8)
    ASYM[f"asym_ratio_g{g:.2f}"] = (.8, -.8 * g)
for x0 in (-2, -1, 0, 1, 2):  # (iii) logistic saturation: belief at log-odds x0, each claim moves it by 2, L follows
    ASYM[f"asym_logit_x{x0:+d}"] = (round(2 * (_sigmoid(x0 + 2) - _sigmoid(x0)), 3), round(2 * (_sigmoid(x0 - 2) - _sigmoid(x0)), 3))
for name, (aff, neg) in ASYM.items():  # each with the placebo at zero, and with a placebo effect (its "no" line 0.4 lower)
    base = {**FP["H2_evidence"], "affirmed": aff, "negated": neg}
    SCENARIOS[name] = {"cand": base, "account": "H2_evidence"}
    SCENARIOS[name + "_pl"] = {"cand": {**base, "placebo_negated": -.4}, "account": "H2_evidence"}
# Phrases that would attribute part of the effect to salience on the strength of M (never allowed)
SALIENCE_ATTRIBUTION = ("salience is likely", "likely part of the effect", "salience is part of the effect",
                        "upper bound on salience", "salience contributes to", "salience plays")


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
            m_r, m_zones, placebo_zones, flags, readings, why_untested = (defaultdict(list), defaultdict(Counter),
                                                                          defaultdict(Counter), Counter(), Counter(), Counter())
            halves, net_r, h_verdicts = defaultdict(list), defaultdict(list), Counter()
            for rep in range(reps):
                rng = random.Random(f"{model}|{name}|{rep}")
                by = mech2._lookup(synth(sets["union"], fps, params, rng))
                fam = {f: mech2.primary_family(by, pops, f, B=B, perms=B) for f in ("cand", "yn")}
                tk = pops["sample_take"]
                mention, placebo, net = ({f: mech2.contrast(by, tk, terms, f, reference=mech2._REF, B=B, perms=B)
                                          for f in ("cand", "yn")}
                                         for terms in (mech2.MENTION_TERMS, mech2.PLACEBO_TERMS, mech2.P8_NET_TERMS))
                joint = mech2.joint_readout(fam["cand"]["readout"], fam["yn"]["readout"], mention, placebo, net)
                for f in ("cand", "yn"):
                    m_r[f].append(mention[f]["ratio"]["estimate"])
                    m_zones[f][str(mech2.secondary_zones(mention[f]))] += 1
                    placebo_zones[f][str(mech2.secondary_zones(placebo[f]))] += 1
                    for half, terms in mech2.M_HALVES.items():
                        halves[(half, f)].append(mech2.contrast(by, tk, terms, f, reference=mech2._REF, B=B, perms=B)
                                                 ["ratio"]["estimate"])
                    net_r[f].append(net[f]["ratio"]["estimate"])
                summ = joint["summary"]
                for k in ("mention_residual_remains", "mention_residual_absent", "claims_read_not_evidence_shown",
                          "negation_placebo_below_band_both_formats"):
                    flags[k] += summ[k]
                flags["salience_attributed"] += any(ph in summ["reading"] for ph in SALIENCE_ATTRIBUTION)
                flags["H3_consistent_next_to_H2"] += (joint["H3_salience"]["verdict"] == "consistent"
                                                      and joint["H2_evidence"]["verdict"] == "consistent")
                h_verdicts[("H2", joint["H2_evidence"]["verdict"])] += 1
                h_verdicts[("H3", joint["H3_salience"]["verdict"])] += 1
                readings[summ["reading"]] += 1
                for h in mech2.CONTENT_ACCOUNTS:
                    if joint[h]["untested_reason"] == "consistent only where it allows zero":
                        why_untested[h] += 1
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
            print("   verdicts (both formats):", dict(verdicts))
            print("   M (mention, claim balanced): mean r " + ", ".join(f"{f} {sum(v) / len(v):+.2f}" for f, v in m_r.items())
                  + "; zones " + "; ".join(f"{f} {dict(z)}" for f, z in m_zones.items()))
            print("   M's halves, mean r: " + "; ".join(f"{h[7:]} {f} {sum(v) / len(v):+.2f}" for (h, f), v in halves.items()))
            print("   negation placebo zones: " + "; ".join(f"{f} {dict(z)}" for f, z in placebo_zones.items())
                  + "; P8 net of the placebo, mean r: " + ", ".join(f"{f} {sum(v) / len(v):+.2f}" for f, v in net_r.items()))
            print("   verdicts of H2 and H3: " + ", ".join(f"{h} {v} {n}" for (h, v), n in sorted(h_verdicts.items()))
                  + f"; salience attributed in the summary {flags['salience_attributed']}/{reps}; H3 consistent next to H2 "
                  f"{flags['H3_consistent_next_to_H2']}/{reps}")
            print(f"   summary sentences: mention residual remains {flags['mention_residual_remains']}/{reps}, none "
                  f"remains {flags['mention_residual_absent']}/{reps}, claims read (evidence not shown) "
                  f"{flags['claims_read_not_evidence_shown']}/{reps}, placebo below the band "
                  f"{flags['negation_placebo_below_band_both_formats']}/{reps}; untested because consistent only where it "
                  f"allows zero: {dict(why_untested)}")
            top, n = readings.most_common(1)[0]
            print(f"   most common summary ({n}/{reps}): {top}", flush=True)


if __name__ == "__main__":
    main()
