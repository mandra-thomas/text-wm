"""Stage 1b, why the null: exposure, verifier inertia and transfer (exploratory, post hoc; notes/decisionsim_agent_exposure.md).

Reads local copies of the label and score runs (no model, no server, nothing scored) and writes one JSON file. Decisions
use the frozen module's own functions (wf.decisionsim_agent at tag decisionsim-stage1b-frozen-2026-11-22), unchanged;
the forced-choice rule and the classifications are new and exploratory. Nothing here changes a frozen estimand or
verdict.

    PYTHONPATH=. python scripts/decisionsim_agent_exposure.py --runs <folder with the decisionsim_agent_* runs> \
        --out notes/decisionsim_agent_exposure_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import decisionsim_agent as D

MODELS = ("llama3.1-8b", "mistral-7b-v0.3", "qwen3-8b")
A, B, S, N = D.AWARE, D.BLIND, D.SWAPPED, D.NEUTRAL
GRID = 1 / 64
BANDS = (("step_0", 0, 0), ("steps_1_11", 1, 11), ("steps_12_29", 12, 10**6))
CLASSES = ("A_both_fall_back", "B_only_aware_falls_back", "C_only_blind_falls_back", "D_both_choose_rank1",
           "E_both_choose_same_alternative", "F_choose_different_candidates")


# ---------------- small helpers ----------------


def band(s: dict) -> str:
    t = int(s["state_idx"])
    return next(n for n, lo, hi in BANDS if lo <= t <= hi)


def cands(s: dict) -> list[dict]:
    return D.scored_candidates(s)


def multi(s: dict) -> bool:
    return len(cands(s)) > 1


def has_working(s: dict) -> bool:
    return any(not c["failed"] for c in cands(s))


def populations(states: list[dict]) -> dict[str, list[dict]]:
    """The populations named as in the frozen design."""
    enc = [s for s in states if D.encounter(s)]
    return {"all_states": states, "encounter_states": enc,
            "encounter_states_with_a_working_candidate": [s for s in enc if has_working(s)]}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def boot(values: list[float], games: list[str]) -> dict[str, Any]:
    return D._boot(values, games)


def spread(states: list[dict], f: Callable[[dict], float]) -> dict[str, Any]:
    """Per-game means of a state-level quantity: min, quartiles, max, and games with a non-zero value."""
    by: dict[str, list[float]] = defaultdict(list)
    for s in states:
        by[s["game_id"]].append(f(s))
    per = sorted(_mean(v) for v in by.values())
    if not per:
        return {"n_games": 0}
    q = statistics.quantiles(per, n=4, method="inclusive") if len(per) > 1 else [per[0]] * 3
    return {"n_games": len(per), "min": per[0], "q25": q[0], "median": q[1], "q75": q[2], "max": per[-1],
            "games_nonzero": sum(abs(x) > 1e-12 for x in per)}


def by_band(states: list[dict], f: Callable[[dict], float]) -> dict[str, Any]:
    out = {}
    for name, lo, hi in BANDS:
        b = [s for s in states if lo <= int(s["state_idx"]) <= hi]
        out[name] = boot([f(s) for s in b], [s["game_id"] for s in b])
    return out


def summary(states: list[dict], f: Callable[[dict], float]) -> dict[str, Any]:
    """A state-level quantity: game-clustered interval overall, by step band, and its per-game spread."""
    return {**boot([f(s) for s in states], [s["game_id"] for s in states]), "by_step_band": by_band(states, f),
            "per_game": spread(states, f)}


# ---------------- forced choice (exploratory rule, pre-registered) ----------------


def forced_choice(arm: str, s: dict, L: dict[str, dict[str, float]]) -> dict:
    """The candidate with the highest rounded log-odds under `arm`; ties go to the lower (better) agent rank. A
    single-candidate state returns its one candidate (every condition picks it)."""
    cs = cands(s)
    if len(cs) == 1:
        return cs[0]
    return min(cs, key=lambda c: (-D.quantize(L[arm][D.cand_id(s, c)]), c["rank"]))


# ---------------- the rule's classes ----------------


def rule_class(s: dict, L: dict[str, dict[str, float]]) -> str | None:
    """Which way the frozen rule let the scores decide at a multi-candidate state (None at a single-candidate state)."""
    if not multi(s):
        return None
    fa, fb = D.fell_back("aware", s, L), D.fell_back("blind", s, L)
    if fa and fb:
        return CLASSES[0]
    if fa:
        return CLASSES[1]
    if fb:
        return CLASSES[2]
    ca, cb = D.choose("aware", s, L), D.choose("blind", s, L)
    if ca["key"] != cb["key"]:
        return CLASSES[5]
    return CLASSES[3] if ca["key"] == cands(s)[0]["key"] else CLASSES[4]


def classes(states: list[dict], L: dict) -> dict[str, Any]:
    n_all = len(states)
    total = sum(float(D.choose("aware", s, L)["failed"]) - float(D.choose("blind", s, L)["failed"]) for s in states)
    out: dict[str, Any] = {"n_states": n_all, "n_multi": sum(multi(s) for s in states), "difference_points_per_decision": total / n_all if n_all else None}
    for k in CLASSES:
        cs = [s for s in states if rule_class(s, L) == k]
        fa = [float(D.choose("aware", s, L)["failed"]) for s in cs]
        fb = [float(D.choose("blind", s, L)["failed"]) for s in cs]
        harm = sum(x and not y for x, y in zip(fa, fb))
        help_ = sum(y and not x for x, y in zip(fa, fb))
        contrib = (harm - help_) / n_all if n_all else float("nan")
        out[k] = {"n": len(cs), "share_of_multi": len(cs) / out["n_multi"] if out["n_multi"] else None,
                  "fail_rate_aware": _mean(fa), "fail_rate_blind": _mean(fb), "harm": harm, "help": help_,
                  "contribution_to_difference": contrib,
                  "share_of_total_difference": (contrib * n_all / total) if abs(total) > 1e-12 else None,
                  "by_step_band": dict(Counter(band(s) for s in cs))}
    out["partition_holds"] = sum(out[k]["n"] for k in CLASSES) == out["n_multi"]
    out["contributions_add_up"] = (not n_all) or abs(sum(out[k]["contribution_to_difference"] for k in CLASSES) - total / n_all) < 1e-12
    return out


# ---------------- the decision chain ----------------


def cand_rows(states: list[dict], L: dict, Lraw: dict, a: str = A, b: str = B) -> list[dict]:
    rows = []
    for s in states:
        if not multi(s):
            continue
        cs = cands(s)
        for i, c in enumerate(cs):
            cid = D.cand_id(s, c)
            rows.append({"game_id": s["game_id"], "band": band(s), "rank1": i == 0, "failed": bool(c["failed"]),
                         "regime": D.regime(c), "d": Lraw[a][cid] - Lraw[b][cid],
                         "acc_a": D.accepts(L[a], cid), "acc_b": D.accepts(L[b], cid)})
    return rows


def group(rows: list[dict], name: str) -> list[dict]:
    pick = {"all": lambda r: True, "working": lambda r: not r["failed"], "failing": lambda r: r["failed"],
            "regime": lambda r: r["regime"], "non_regime": lambda r: not r["regime"],
            "other_failing": lambda r: r["failed"] and not r["regime"],
            "rank1": lambda r: r["rank1"], "alternatives": lambda r: not r["rank1"],
            "rank1_working": lambda r: r["rank1"] and not r["failed"], "rank1_failing": lambda r: r["rank1"] and r["failed"],
            "alternatives_working": lambda r: not r["rank1"] and not r["failed"],
            "alternatives_failing": lambda r: not r["rank1"] and r["failed"]}[name]
    return [r for r in rows if pick(r)]


GROUPS = ("all", "working", "failing", "regime", "non_regime", "other_failing", "rank1", "alternatives", "rank1_working",
          "rank1_failing", "alternatives_working", "alternatives_failing")


def _q(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    pick = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {"p05": pick(0.05), "p25": pick(0.25), "p50": pick(0.5), "p75": pick(0.75), "p95": pick(0.95), "mean": _mean(xs)}


def acceptance(rows: list[dict]) -> dict[str, Any]:
    out = {}
    for g in GROUPS:
        rs = group(rows, g)
        gm = [r["game_id"] for r in rs]
        out[g] = {"n": len(rs), "aware": boot([float(r["acc_a"]) for r in rs], gm),
                  "blind": boot([float(r["acc_b"]) for r in rs], gm),
                  "aware_minus_blind": boot([float(r["acc_a"]) - float(r["acc_b"]) for r in rs], gm)}
    return out


def chain_candidates(rows: list[dict]) -> dict[str, Any]:
    out = {}
    for g in ("all", "regime", "other_failing", "working", "rank1", "alternatives"):
        rs = group(rows, g)
        gm = [r["game_id"] for r in rs]
        out[g] = {"n": len(rs),
                  "score_changes_over_grid": boot([float(abs(r["d"]) > GRID) for r in rs], gm),
                  "score_changes_over_1_nat": boot([float(abs(r["d"]) > 1.0) for r in rs], gm),
                  "aware_minus_blind_L": _q([r["d"] for r in rs]),
                  "flip_to_accept": boot([float(r["acc_a"] and not r["acc_b"]) for r in rs], gm),
                  "flip_to_reject": boot([float(r["acc_b"] and not r["acc_a"]) for r in rs], gm),
                  "any_flip": boot([float(r["acc_a"] != r["acc_b"]) for r in rs], gm)}
    return out


def score_order(arm: str, s: dict, L: dict) -> list[str]:
    return [c["key"] for c in sorted(cands(s), key=lambda c: (-D.quantize(L[arm][D.cand_id(s, c)]), c["rank"]))]


def state_flags(s: dict, L: dict, Lraw: dict, a: str = "aware", b: str = "blind") -> dict[str, bool]:
    """The decision chain at one state, aware against blind (False at every level for a single-candidate state)."""
    if not multi(s):
        return {k: False for k in ("multi", "score_grid", "score_1nat", "flip", "order", "choice", "outcome")}
    aa, bb = D.CONFIG_ARM[a], D.CONFIG_ARM[b]
    ids = [D.cand_id(s, c) for c in cands(s)]
    d = [Lraw[aa][i] - Lraw[bb][i] for i in ids]
    ca, cb = D.choose(a, s, L), D.choose(b, s, L)
    return {"multi": True, "score_grid": any(abs(x) > GRID for x in d), "score_1nat": any(abs(x) > 1.0 for x in d),
            "flip": any(D.accepts(L[aa], i) != D.accepts(L[bb], i) for i in ids),
            "order": score_order(aa, s, L) != score_order(bb, s, L),
            "choice": ca["key"] != cb["key"], "outcome": ca["failed"] != cb["failed"]}


FUNNEL = ("multi", "score_grid", "score_1nat", "flip", "order", "choice", "outcome")


def funnel(states: list[dict], L: dict, Lraw: dict) -> dict[str, Any]:
    flags = [state_flags(s, L, Lraw) for s in states]
    n = len(states)
    out: dict[str, Any] = {"n_states": n}
    for k in FUNNEL:
        c = sum(f[k] for f in flags)
        out[k] = {"n": c, "share_of_states": c / n if n else None}
    harm = sum(D.choose("aware", s, L)["failed"] and not D.choose("blind", s, L)["failed"] for s in states)
    help_ = sum(D.choose("blind", s, L)["failed"] and not D.choose("aware", s, L)["failed"] for s in states)
    out["harm"], out["help"], out["net"] = harm, help_, harm - help_
    # A changed choice needs an acceptance flip (same accepted set, same rule, same choice), and a changed outcome
    # needs a changed choice; both are checked, not assumed.
    out["choice_without_flip"] = sum(f["choice"] and not f["flip"] for f in flags)
    out["outcome_without_choice"] = sum(f["outcome"] and not f["choice"] for f in flags)
    out["nested"] = {"score_grid": sum(f["score_grid"] for f in flags),
                     "and_flip": sum(f["score_grid"] and f["flip"] for f in flags),
                     "and_choice": sum(f["score_grid"] and f["flip"] and f["choice"] for f in flags),
                     "and_outcome": sum(f["score_grid"] and f["flip"] and f["choice"] and f["outcome"] for f in flags)}
    return out


def regime_funnel(states: list[dict], L: dict) -> dict[str, Any]:
    enc = [s for s in states if D.encounter(s)]
    m = [s for s in enc if multi(s)]

    def flipped_up(s: dict) -> list[dict]:
        return [c for c in cands(s) if D.regime(c) and D.accepts(L[A], D.cand_id(s, c)) and not D.accepts(L[B], D.cand_id(s, c))]

    up = [s for s in m if flipped_up(s)]
    chosen = [s for s in up if any(D.choose("aware", s, L)["key"] == c["key"] for c in flipped_up(s))]
    harm = [s for s in chosen if not D.choose("blind", s, L)["failed"]]
    all_regime_harm = [s for s in enc if D.regime(D.choose("aware", s, L)) and D.choose("aware", s, L)["failed"]
                       and not D.choose("blind", s, L)["failed"]]
    return {"encounter_states": len(enc), "at_multi_candidate_states": len(m),
            "with_a_working_candidate": sum(has_working(s) for s in m),
            "regime_candidate_flips_to_accept_under_the_task": len(up),
            "that_candidate_becomes_the_aware_choice": len(chosen),
            "and_the_blind_choice_works": len(harm),
            "all_regime_harm_states": len(all_regime_harm),
            "regime_accepted_by_both_and_chosen_by_both": sum(
                D.regime(D.choose("aware", s, L)) and D.choose("aware", s, L)["key"] == D.choose("blind", s, L)["key"] for s in m)}


# ---------------- failure-rate differences ----------------


def paired(states: list[dict], fa: Callable[[dict], dict], fb: Callable[[dict], dict]) -> dict[str, Any]:
    ca, cb = [fa(s) for s in states], [fb(s) for s in states]
    out = D._paired([float(c["failed"]) for c in ca], [float(c["failed"]) for c in cb], [s["game_id"] for s in states])
    out["harm"] = sum(x["failed"] and not y["failed"] for x, y in zip(ca, cb))
    out["help"] = sum(y["failed"] and not x["failed"] for x, y in zip(ca, cb))
    out["harm_regime"] = sum(x["failed"] and not y["failed"] and D.regime(x) for x, y in zip(ca, cb))
    out["harm_regime_share"] = out["harm_regime"] / out["harm"] if out["harm"] else None
    return out


def frozen_differences(states: list[dict], L: dict) -> dict[str, Any]:
    return {k: paired(v, lambda s: D.choose("aware", s, L), lambda s: D.choose("blind", s, L))
            for k, v in populations(states).items()}


def forced(states: list[dict], L: dict) -> dict[str, Any]:
    """The forced-choice diagnostic (exploratory): multi-candidate states only; per-decision versions over all states."""
    arms = [a for a in (A, B, S, N) if a in L]
    pops = {k: [s for s in v if multi(s)] for k, v in populations(states).items()}
    out: dict[str, Any] = {}
    for pname, ps in pops.items():
        sw = [s for s in ps if s.get("swap")]
        g = [s["game_id"] for s in ps]
        row: dict[str, Any] = {"n_states": len(ps), "n_games": len(set(g))}
        row["fail_rate"] = {a: boot([float(forced_choice(a, s, L)["failed"]) for s in (sw if a == S else ps)],
                                    [s["game_id"] for s in (sw if a == S else ps)]) for a in arms}
        row["fail_rate"]["rank1"] = boot([float(cands(s)[0]["failed"]) for s in ps], g)
        row["fail_rate"]["oracle"] = boot([float(D.choose("oracle", s)["failed"]) for s in ps], g)
        row["fail_rate"]["frozen_aware"] = boot([float(D.choose("aware", s, L)["failed"]) for s in ps], g)
        row["fail_rate"]["frozen_blind"] = boot([float(D.choose("blind", s, L)["failed"]) for s in ps], g)
        row["aware_minus_blind"] = paired(ps, lambda s: forced_choice(A, s, L), lambda s: forced_choice(B, s, L))
        if S in L:
            row["aware_minus_swapped"] = paired(sw, lambda s: forced_choice(A, s, L), lambda s: forced_choice(S, s, L))
            row["swapped_minus_blind"] = paired(sw, lambda s: forced_choice(S, s, L), lambda s: forced_choice(B, s, L))
        if N in L:
            row["aware_minus_neutral"] = paired(ps, lambda s: forced_choice(A, s, L), lambda s: forced_choice(N, s, L))
        # added after the pre-registered list (robustness of the forced-choice difference): leave one game out, and
        # the harm states per game
        fa_, fb_ = [float(forced_choice(A, s, L)["failed"]) for s in ps], [float(forced_choice(B, s, L)["failed"]) for s in ps]
        row["aware_minus_blind_leave_one_game_out"] = D.leave_one_game_out(fa_, fb_, g)
        row["aware_minus_blind_harm_per_game"] = dict(Counter(x for x, a_, b_ in zip(g, fa_, fb_) if a_ > b_).most_common())
        row["aware_minus_rank1"] = paired(ps, lambda s: forced_choice(A, s, L), lambda s: cands(s)[0])
        row["blind_minus_rank1"] = paired(ps, lambda s: forced_choice(B, s, L), lambda s: cands(s)[0])
        row["by_step_band"] = {b: paired([s for s in ps if band(s) == b], lambda s: forced_choice(A, s, L),
                                         lambda s: forced_choice(B, s, L)) for b, _, _ in BANDS}
        out[pname] = row
    # per decision over all states (single-candidate states contribute zero)
    out["aware_minus_blind_per_decision_all_states"] = paired(states, lambda s: forced_choice(A, s, L), lambda s: forced_choice(B, s, L))
    m = pops["all_states"]
    out["per_game_spread_multi"] = spread(m, lambda s: float(forced_choice(A, s, L)["failed"]) - float(forced_choice(B, s, L)["failed"]))
    return out


# ---------------- one model, one list ----------------


def model_block(states: list[dict], L: dict, Lraw: dict, frozen_model: dict | None, full: bool = True) -> dict[str, Any]:
    m = [s for s in states if multi(s)]
    out: dict[str, Any] = {"n_states": len(states), "n_multi": len(m)}
    diffs = frozen_differences(states, L)
    out["frozen_rule_differences"] = diffs
    if frozen_model is not None:
        fp, fc = frozen_model["primary"], frozen_model["conditional"]
        out["matches_frozen"] = {
            "overall": abs(diffs["all_states"]["estimate"] - fp["difference"]["estimate"]) < 1e-12
            and [diffs["all_states"]["harm"], diffs["all_states"]["help"]] == [fp["changed"]["harm"], fp["changed"]["help"]]
            and diffs["all_states"]["n_states"] == fp["n_states"]
            and diffs["all_states"]["ci"] == fp["difference"]["ci"] and diffs["all_states"]["p"] == fp["difference"]["p"],
            "encounter_states": abs(diffs["encounter_states"]["estimate"] - fc["difference"]["estimate"]) < 1e-12
            and [diffs["encounter_states"]["harm"], diffs["encounter_states"]["help"]] == [fc["changed"]["harm"], fc["changed"]["help"]]
            and diffs["encounter_states"]["n_states"] == fc["n_states"]
            and diffs["encounter_states"]["ci"] == fc["difference"]["ci"] and diffs["encounter_states"]["p"] == fc["difference"]["p"]}
    out["rule_classes"] = classes(states, L)
    out["rule_classes_by_population"] = {k: classes(v, L) for k, v in populations(states).items() if k != "all_states"}
    out["forced_choice"] = forced(states, L)
    if not full:
        return out
    rows = cand_rows(states, L, Lraw)
    out["acceptance"] = acceptance(rows)
    out["fallback"] = {cfg: summary(m, lambda s, c=cfg: float(D.fell_back(c, s, L))) for cfg in ("aware", "blind")}
    out["disagreement"] = {"all_states": summary(states, lambda s: float(D.choose("aware", s, L)["key"] != D.choose("blind", s, L)["key"])),
                           "multi_candidate_states": boot([float(D.choose("aware", s, L)["key"] != D.choose("blind", s, L)["key"]) for s in m],
                                                          [s["game_id"] for s in m])}
    out["chain_candidates"] = chain_candidates(rows)
    out["chain_candidates_by_step_band"] = {b: chain_candidates([r for r in rows if r["band"] == b]) for b, _, _ in BANDS}
    out["chain_states"] = {"order_of_accepted_differs": summary(m, lambda s: float(state_flags(s, L, Lraw)["flip"])),
                           "score_order_differs": summary(m, lambda s: float(state_flags(s, L, Lraw)["order"])),
                           "choice_differs": summary(m, lambda s: float(state_flags(s, L, Lraw)["choice"]))}
    out["funnel"] = {k: funnel(v, L, Lraw) for k, v in populations(states).items()}
    out["funnel_by_step_band"] = {b: funnel([s for s in states if band(s) == b], L, Lraw) for b, _, _ in BANDS}
    out["regime_funnel"] = regime_funnel(states, L)
    out["difference_by_step_band"] = {b: paired([s for s in states if band(s) == b], lambda s: D.choose("aware", s, L),
                                                lambda s: D.choose("blind", s, L)) for b, _, _ in BANDS}
    out["difference_per_game"] = spread(states, lambda s: float(D.choose("aware", s, L)["failed"]) - float(D.choose("blind", s, L)["failed"]))
    out["rule_class_shares_by_step_band"] = {b: {k: sum(rule_class(s, L) == k for s in m if band(s) == b) for k in CLASSES} for b, _, _ in BANDS}
    out["rule_class_per_game"] = {k: spread(m, lambda s, k=k: float(rule_class(s, L) == k)) for k in CLASSES}
    out["candidate_flip_per_game"] = spread(m, lambda s: _mean([float(D.accepts(L[A], D.cand_id(s, c)) != D.accepts(L[B], D.cand_id(s, c))) for c in cands(s)]))
    return out


def exposure(states: list[dict]) -> dict[str, Any]:
    return {"encounter_states": summary(states, lambda s: float(D.encounter(s))),
            "encounter_states_with_a_working_candidate": summary(states, lambda s: float(D.encounter(s) and has_working(s))),
            "strict_encounter_states": summary(states, lambda s: float(D.encounter(s, strict=True))),
            "multi_candidate_states": summary(states, lambda s: float(multi(s))),
            "encounter_games": len({s["game_id"] for s in states if D.encounter(s)}),
            "encounter_with_working_games": len({s["game_id"] for s in states if D.encounter(s) and has_working(s)}),
            "strict_games": len({s["game_id"] for s in states if D.encounter(s, strict=True)})}


# ---------------- main ----------------


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _r(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 6) if x == x else None
    if isinstance(x, dict):
        return {str(k): _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    prov: dict[str, str] = {}
    labs = []
    for p in sorted((a.runs / "decisionsim_agent_label").glob("shard_*_of_*")):
        prov[str((p / "labels.jsonl").relative_to(a.runs))] = hashlib.sha256((p / "labels.jsonl").read_bytes()).hexdigest()
        labs += read_jsonl(p / "labels.jsonl")
    labs.sort(key=lambda s: (s["game_id"], s["state_idx"]))
    frozen = json.loads((a.runs / "decisionsim_agent_analysis" / "metrics.json").read_text(encoding="utf-8"))
    good = [s for s in labs if D.usable(s)]
    res: dict[str, Any] = {"meta": {"note": "notes/decisionsim_agent_exposure.md (exploratory)", "n_usable": len(good),
                                    "provenance_sha256": prov}}
    prim, t1 = D.views(good, D.PRIMARY_LIST), D.views(good, D.T1_LIST)
    res["exposure"] = {"primary": exposure(prim), "t1": exposure(t1)}
    res["models"] = {}
    for t in MODELS:
        recs = []
        for p in sorted((a.runs / f"decisionsim_agent_score_{t}").glob("shard_*_of_*")):
            prov[str((p / "scoring.jsonl").relative_to(a.runs))] = hashlib.sha256((p / "scoring.jsonl").read_bytes()).hexdigest()
            recs += read_jsonl(p / "scoring.jsonl")
        L, Lraw = D.scores_by_arm(recs)
        sts, short = D.complete_labs(good, L)
        res["models"][t] = {"n_states_incomplete": short,
                            "primary": model_block(D.views(sts, D.PRIMARY_LIST), L, Lraw, frozen["models"][t]),
                            "t1_sensitivity": model_block(D.views(sts, D.T1_LIST), L, Lraw, None, full=False)}
        print(t, "done", flush=True)
    a.out.write_text(json.dumps(_r(res), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
