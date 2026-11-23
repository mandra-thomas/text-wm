"""Stage 1b deep-dive (exploratory, post hoc): the eight pre-registered analyses of notes/decisionsim_agent_dx.md.

Reads local copies of the run folders (labels and yes/no scores; no model, no server) and writes one JSON file.
Item 1 rebuilds every decision with its own code (rule, rounding, bootstrap, sign-flip permutation) and checks it
against the frozen analysis output. Items 2-8 use the frozen module's own decision functions (wf.decisionsim_agent at
tag decisionsim-stage1b-frozen-2026-11-22), unchanged, and its own bootstrap and permutation for intervals. Nothing is
corrected for multiple comparisons, and nothing here changes a verdict.

    python scripts/decisionsim_agent_dx.py --runs <folder with the decisionsim_agent_* run folders> \
        --stage1a-dx notes/decisionsim_dx_metrics.json --out notes/decisionsim_agent_dx_metrics.json

`--stage1a-dx` is Stage 1a's deep-dive metrics (branch main); it is only quoted next to this stage's step bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from wf import decisionsim_agent as D

MODELS = ("llama3.1-8b", "mistral-7b-v0.3", "qwen3-8b")
PRIORITY = ("llama3.1-8b", "mistral-7b-v0.3")
A, B, S, N = D.AWARE, D.BLIND, D.SWAPPED, D.NEUTRAL
SEED = 20260924
BANDS3 = (("step_0", 0, 0), ("steps_1_11", 1, 11), ("steps_12_29", 12, 10**6))


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def r(x: Any, k: int = 5) -> Any:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return round(x, k) if x == x else None
    if isinstance(x, dict):
        return {kk: r(v, k) for kk, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, k) for v in x]
    return x


# ---------------- loading ----------------


def load(root: Path) -> tuple[list[dict], dict[str, list[dict]], dict[str, str]]:
    prov: dict[str, str] = {}
    labs = []
    for p in sorted((root / "decisionsim_agent_label").glob("shard_*_of_*")):
        f = p / "labels.jsonl"
        prov[str(f.relative_to(root))] = sha256(f)
        labs += read_jsonl(f)
    labs.sort(key=lambda s: (s["game_id"], s["state_idx"]))
    scores = {}
    for t in MODELS:
        recs = []
        for p in sorted((root / f"decisionsim_agent_score_{t}").glob("shard_*_of_*")):
            f = p / "scoring.jsonl"
            prov[str(f.relative_to(root))] = sha256(f)
            recs += read_jsonl(f)
        scores[t] = recs
    f = root / "decisionsim_agent_analysis" / "metrics.json"
    prov[str(f.relative_to(root))] = sha256(f)
    return labs, scores, prov


def task_type(line: str) -> str:
    """Task type from the task line (Stage 1a deep-dive's parser)."""
    s = line.replace("Your task is to: ", "").rstrip(".")
    if s.startswith(("examine the", "look at")):
        return "examine"
    if "two " in s:
        return "pick_two"
    for verb, adj, kind in (("clean", "clean", "clean"), ("heat", "hot", "heat"), ("cool", "cool", "cool")):
        if s.startswith(verb + " ") or f" {adj} " in s:
            return kind
    return "pick"


# ---------------- item 1: independent recompute ----------------


def _norm(a: str) -> str:
    return " ".join(a.split()).lower().rstrip(".").strip()


def _key(a: str) -> str:
    return hashlib.sha256((_norm(a) or a).encode()).hexdigest()[:12]


def _regime(c: dict) -> bool:
    return bool(c["failed"] and c["targets_goal_object"] and c["context_determined"] and not c["shared_location"])


def _states_indep(labs: list[dict], lst: str = "primary") -> list[dict]:
    out = []
    for lab in labs:
        if not (lab["rank1_replay_ok"] and lab.get("history_replay_ok", True)):
            continue
        by = {c["key"]: c for c in lab["candidates"] if c.get("server_error") is None}
        cands = [by[_key(e["action"])] for e in lab["lists"][lst] if _key(e["action"]) in by]
        if cands:
            out.append({"g": lab["game_id"], "t": int(lab["state_idx"]), "cands": cands, "swap": lab.get("swap")})
    return out


def _tid(s: dict, c: dict) -> str:
    return f"{s['g']}#{s['t']:02d}#{c['key']}"


def _scores_indep(recs: list[dict]) -> dict[str, dict[str, float]]:
    q: dict[str, dict[str, float]] = defaultdict(dict)
    for x in recs:
        if x["L"] == x["L"]:
            q[x["condition"]][x["transition_id"]] = round(x["L"] * 64) / 64
    return q


def _pick(s: dict, sc: dict[str, float]) -> dict:
    if len(s["cands"]) == 1:
        return s["cands"][0]
    for c in s["cands"]:
        if sc[_tid(s, c)] > 0:
            return c
    return s["cands"][0]


def _boot_np(diffs: np.ndarray, games: list[str], B: int = 2000, seed: int = SEED) -> list[float]:
    ug = sorted(set(games))
    idx = {g: i for i, g in enumerate(ug)}
    gi = np.array([idx[g] for g in games])
    sums = np.bincount(gi, weights=diffs, minlength=len(ug))
    cnts = np.bincount(gi, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ug), size=(B, len(ug)))
    est = sums[draw].sum(1) / cnts[draw].sum(1)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def _perm_np(diffs: np.ndarray, games: list[str], n: int = 10000, seed: int = SEED) -> float:
    ug = sorted(set(games))
    idx = {g: i for i, g in enumerate(ug)}
    sums = np.bincount(np.array([idx[g] for g in games]), weights=diffs, minlength=len(ug))
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(ug)))
    stat = np.abs((signs * sums).sum(1))
    return float((1 + (stat >= obs - 1e-12).sum()) / (n + 1))


def item1(labs: list[dict], scores: dict[str, list[dict]], frozen: dict) -> dict:
    sts = _states_indep(labs)
    out: dict[str, Any] = {"n_states": len(sts), "n_games": len({s["g"] for s in sts}),
                           "n_encounter": sum(any(_regime(c) for c in s["cands"]) for s in sts),
                           "encounter_games": len({s["g"] for s in sts if any(_regime(c) for c in s["cands"])}),
                           "models": {}}
    for t in MODELS:
        q = _scores_indep(scores[t])
        fa = np.array([float(_pick(s, q[A])["failed"]) for s in sts])
        fb = np.array([float(_pick(s, q[B])["failed"]) for s in sts])
        f1 = np.array([float(s["cands"][0]["failed"]) for s in sts])
        games = [s["g"] for s in sts]
        enc = np.array([any(_regime(c) for c in s["cands"]) for s in sts])
        res = {"F_none": f1.mean(), "F_aware": fa.mean(), "F_blind": fb.mean()}
        for name, m in (("overall", np.ones(len(sts), bool)), ("conditional", enc)):
            d = (fa - fb)[m]
            g = [x for x, k in zip(games, m) if k]
            res[name] = {"estimate": float(d.mean()), "ci": _boot_np(d, g), "p_10000_flips": _perm_np(d, g),
                         "harm": int(((fa == 1) & (fb == 0))[m].sum()), "help": int(((fa == 0) & (fb == 1))[m].sum()),
                         "n_states": int(m.sum())}
        fm = frozen["models"][t]
        res["matches_frozen"] = {
            "F_aware": abs(res["F_aware"] - fm["fail_rate"]["aware"]["estimate"]) < 1e-12,
            "F_blind": abs(res["F_blind"] - fm["fail_rate"]["blind"]["estimate"]) < 1e-12,
            "F_none": abs(res["F_none"] - fm["fail_rate"]["none"]["estimate"]) < 1e-12,
            "overall_estimate": abs(res["overall"]["estimate"] - fm["primary"]["difference"]["estimate"]) < 1e-12,
            "overall_harm_help": [res["overall"]["harm"], res["overall"]["help"]] == [fm["primary"]["changed"]["harm"], fm["primary"]["changed"]["help"]],
            "conditional_estimate": abs(res["conditional"]["estimate"] - fm["conditional"]["difference"]["estimate"]) < 1e-12,
            "conditional_harm_help": [res["conditional"]["harm"], res["conditional"]["help"]] == [fm["conditional"]["changed"]["harm"], fm["conditional"]["changed"]["help"]],
            "conditional_n": res["conditional"]["n_states"] == fm["conditional"]["n_states"]}
        out["models"][t] = res
    return out


# ---------------- frozen helpers for items 2-8 ----------------


def frozen_states(labs: list[dict], recs: list[dict]) -> tuple[list[dict], dict, dict, int]:
    good = [s for s in labs if D.usable(s)]
    L, Lraw = D.scores_by_arm(recs)
    sts, short = D.complete_labs(good, L)
    return D.views(sts, D.PRIMARY_LIST), L, Lraw, short


def fails(cfg: str, s: dict, L: dict, **kw: Any) -> float:
    return float(D.choose(cfg, s, L, **kw)["failed"])


def diff(states: list[dict], L: dict, a: str = "aware", b: str = "blind", **kw: Any) -> dict:
    games = [s["game_id"] for s in states]
    out = D._paired([fails(a, s, L, **kw) for s in states], [fails(b, s, L, **kw) for s in states], games)
    ca = [D.choose(a, s, L, **kw) for s in states]
    cb = [D.choose(b, s, L, **kw) for s in states]
    out["harm"] = sum(x["failed"] and not y["failed"] for x, y in zip(ca, cb))
    out["help"] = sum(y["failed"] and not x["failed"] for x, y in zip(ca, cb))
    return out


def kind(c: dict) -> str:
    if not c["failed"]:
        return "works"
    return "regime" if D.regime(c) else "other_failure"


def cell(ca: dict, cb: dict) -> str:
    fa, fb = ca["failed"], cb["failed"]
    return "harm" if fa and not fb else "help" if fb and not fa else "both_fail" if fa else "both_succeed"


# ---------------- item 2: per-game stability ----------------


def per_game(states: list[dict], L: dict) -> dict:
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for s in states:
        ca, cb = D.choose("aware", s, L), D.choose("blind", s, L)
        by[s["game_id"]][2] += 1
        if ca["failed"] and not cb["failed"]:
            by[s["game_id"]][0] += 1
        elif cb["failed"] and not ca["failed"]:
            by[s["game_id"]][1] += 1
    nets = {g: v[0] - v[1] for g, v in by.items()}
    harm = {g: v[0] for g, v in by.items()}
    total_harm = sum(harm.values())
    top5 = sorted(harm.values(), reverse=True)[:5]
    return {"n_games": len(by), "games_positive": sum(v > 0 for v in nets.values()),
            "games_negative": sum(v < 0 for v in nets.values()), "games_zero": sum(v == 0 for v in nets.values()),
            "games_with_harm": sum(v > 0 for v in harm.values()), "max_harm_in_one_game": max(harm.values() or [0]),
            "harm_total": total_harm, "harm_share_top5_games": (sum(top5) / total_harm) if total_harm else None,
            "largest_nets": sorted(([g, n] for g, n in nets.items() if n), key=lambda x: (-abs(x[1]), x[0]))[:6],
            "net_distribution": dict(sorted(Counter(nets.values()).items()))}


def item2(states: list[dict], L: dict) -> dict:
    enc = [s for s in states if D.encounter(s)]
    tt = {tt: [s for s in states if task_type(s["task_line"]) == tt] for tt in ("pick", "pick_two", "clean", "heat", "cool", "examine")}
    return {"overall": per_game(states, L), "encounter": per_game(enc, L),
            "by_task_type": {k: {"n_games": len({s["game_id"] for s in v}), **diff(v, L)} for k, v in tt.items()},
            "leave_task_type_out": {k: diff([s for s in states if task_type(s["task_line"]) != k], L)["estimate"] for k in tt}}


# ---------------- item 3: leave one game out ----------------


def item3(states: list[dict], L: dict) -> dict:
    enc = [s for s in states if D.encounter(s)]
    out = {}
    for name, sts in (("overall", states), ("conditional", enc)):
        games = [s["game_id"] for s in sts]
        lo = D.leave_one_game_out([fails("aware", s, L) for s in sts], [fails("blind", s, L) for s in sts], games)
        full = _mean([fails("aware", s, L) - fails("blind", s, L) for s in sts])
        out[name] = {"full": full, **lo}
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------- item 4: encounter states ----------------


def item4(states: list[dict], models: dict[str, tuple[list[dict], dict]]) -> dict:
    enc = [s for s in states if D.encounter(s)]
    desc = []
    for s in enc:
        cs = D.scored_candidates(s)
        reg = [c["rank"] for c in cs if D.regime(c)]
        work = [c["rank"] for c in cs if not c["failed"]]
        desc.append({"game_id": s["game_id"], "step": s["state_idx"], "list_length": len(cs), "regime_ranks": reg,
                     "rank1_regime": D.regime(cs[0]), "working_ranks": work, "reachable": D.reachable(s),
                     "strict": D.encounter(s, strict=True)})
    band = Counter(next(n for n, lo, hi in BANDS3 if lo <= d["step"] <= hi) for d in desc)
    out: dict[str, Any] = {
        "n_states": len(enc), "n_games": len({s["game_id"] for s in enc}),
        "states_per_game": sorted(Counter(s["game_id"] for s in enc).values(), reverse=True),
        "by_step_band": dict(band), "median_step": statistics.median(d["step"] for d in desc),
        "list_length": dict(sorted(Counter(d["list_length"] for d in desc).items())),
        "single_candidate": sum(d["list_length"] == 1 for d in desc),
        "rank1_regime": sum(d["rank1_regime"] for d in desc),
        "rank1_regime_single_candidate": sum(d["rank1_regime"] and d["list_length"] == 1 for d in desc),
        "with_working_candidate": sum(bool(d["working_ranks"]) for d in desc),
        "reachable": sum(d["reachable"] for d in desc),
        "working_above_first_regime": sum(bool(d["working_ranks"]) and min(d["working_ranks"]) < min(d["regime_ranks"]) for d in desc),
        "strict_states": [d for d in desc if d["strict"]], "models": {}}
    for t, (sts, L) in models.items():
        es = [s for s in sts if D.encounter(s)]
        cross = Counter()
        changed = []
        for s in es:
            ca, cb = D.choose("aware", s, L), D.choose("blind", s, L)
            cross[f"aware {kind(ca)} / blind {kind(cb)}"] += 1
            if ca["key"] != cb["key"]:
                changed.append({"game_id": s["game_id"], "step": s["state_idx"], "aware": f"{ca['action']} (rank {ca['rank']}, {kind(ca)})",
                                "blind": f"{cb['action']} (rank {cb['rank']}, {kind(cb)})", "cell": cell(ca, cb),
                                "path": D.change_path("aware", "blind", s, L, ca, cb)})
        strict = []
        for s in es:
            if D.encounter(s, strict=True):
                ca, cb = D.choose("aware", s, L), D.choose("blind", s, L)
                strict.append({"game_id": s["game_id"], "step": s["state_idx"], "aware": kind(ca), "blind": kind(cb)})
        out["models"][t] = {"choices": dict(sorted(cross.items())), "changed": changed, "strict": strict}
    return out


# ---------------- item 5: regime candidates through the rule ----------------


def item5(sts: list[dict], L: dict) -> dict:
    rows = []
    for s in sts:
        cs = D.scored_candidates(s)
        for c in cs:
            if not D.regime(c):
                continue
            cid = D.cand_id(s, c)
            single = len(cs) == 1
            ca, cb = D.choose("aware", s, L), D.choose("blind", s, L)
            row = {"rank1": c is cs[0], "single": single,
                   "aware_acc": None if single else D.accepts(L[A], cid),
                   "blind_acc": D.accepts(L[B], cid),
                   "swapped_acc": (D.accepts(L[S], cid) if (S in L and s.get("swap") and not single) else None),
                   "aware_chose": ca["key"] == c["key"], "blind_chose": cb["key"] == c["key"],
                   "aware_choice": kind(ca), "blind_choice": kind(cb),
                   "aware_choice_rank_vs": ("above" if ca["rank"] < c["rank"] else "below" if ca["rank"] > c["rank"] else "same"),
                   "aware_abstained": (not single) and D.fell_back("aware", s, L)}
            rows.append(row)

    def summ(rs: list[dict]) -> dict:
        multi = [x for x in rs if not x["single"]]
        acc = [x for x in multi if x["aware_acc"]]
        not_chosen = [x for x in acc if not x["aware_chose"]]
        chosen = [x for x in acc if x["aware_chose"]]
        sw = [x for x in multi if x["swapped_acc"] is not None]
        return {"n": len(rs), "single_candidate_forced": len(rs) - len(multi), "multi": len(multi),
                "aware_accepts": len(acc), "blind_accepts": sum(x["blind_acc"] for x in multi),
                "both_accept": sum(x["blind_acc"] and x["aware_acc"] for x in multi),
                "aware_only": sum(x["aware_acc"] and not x["blind_acc"] for x in multi),
                "blind_only": sum(x["blind_acc"] and not x["aware_acc"] for x in multi),
                "swapped_scored": len(sw), "swapped_accepts": sum(bool(x["swapped_acc"]) for x in sw),
                "aware_accepts_on_swapped_states": sum(bool(x["aware_acc"]) for x in sw),
                "aware_accepted_and_chose": len(chosen),
                "aware_accepted_not_chosen": len(not_chosen),
                "not_chosen_because_higher_ranked_accepted": Counter(x["aware_choice"] for x in not_chosen if x["aware_choice_rank_vs"] == "above"),
                "not_chosen_other": sum(x["aware_choice_rank_vs"] != "above" for x in not_chosen),
                "aware_chose_regime_blind_choice": Counter(x["blind_choice"] for x in chosen),
                "aware_rejected_but_chosen_by_abstention": sum((not x["aware_acc"]) and x["aware_chose"] for x in multi),
                "blind_chose": sum(x["blind_chose"] for x in multi)}
    return {"all": summ(rows), "rank1": summ([x for x in rows if x["rank1"]]),
            "alternatives": summ([x for x in rows if not x["rank1"]])}


# ---------------- item 6: abstention ----------------


def _band3(t: int) -> str:
    return next(n for n, lo, hi in BANDS3 if lo <= t <= hi)


def item6(sts: list[dict], L: dict, Lraw: dict) -> dict:
    multi = [s for s in sts if len(D.scored_candidates(s)) > 1]
    arms = [c for c in ("aware", "blind", "swapped", "neutral") if D.CONFIG_ARM[c] in L]
    fb: dict[str, Any] = {}
    for cfg in arms:
        ss = [s for s in multi if cfg != "swapped" or s.get("swap")]
        row = {}
        for bn, lo, hi in BANDS3:
            b = [s for s in ss if lo <= s["state_idx"] <= hi]
            w = [s for s in b if any(not c["failed"] for c in D.scored_candidates(s))]
            nw = [s for s in b if all(c["failed"] for c in D.scored_candidates(s))]
            row[bn] = {"n": len(b), "fallback_rate": _mean([float(D.fell_back(cfg, s, L)) for s in b]),
                       "n_with_working": len(w),
                       "fallback_rate_with_working": _mean([float(D.fell_back(cfg, s, L)) for s in w]),
                       "n_without_working": len(nw),
                       "fallback_rate_without_working": _mean([float(D.fell_back(cfg, s, L)) for s in nw]),
                       "fallback_with_working_rank1_fails": sum(D.fell_back(cfg, s, L) and D.scored_candidates(s)[0]["failed"] for s in w)}
        fb[cfg] = row
    # Scores of working candidates (rank 1, alternatives) by arm, step band and verb, at multi-candidate states.
    scores: dict[str, Any] = {}
    for pos in ("rank1", "alternative"):
        for bn, lo, hi in BANDS3:
            vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for s in multi:
                if not (lo <= s["state_idx"] <= hi):
                    continue
                for i, c in enumerate(D.scored_candidates(s)):
                    if c["failed"] or (i == 0) != (pos == "rank1"):
                        continue
                    cid = D.cand_id(s, c)
                    for verb in ("all", c.get("action_type") or "other"):
                        for arm in (A, B, S, N):
                            if arm in Lraw and cid in Lraw[arm]:
                                vals[verb][arm].append(Lraw[arm][cid])
                        if A in Lraw and cid in Lraw[A]:
                            vals[verb]["shift_aware_minus_blind"].append(Lraw[A][cid] - Lraw[B][cid])
                            if S in Lraw and cid in Lraw[S]:
                                vals[verb]["shift_swapped_minus_blind"].append(Lraw[S][cid] - Lraw[B][cid])
                                vals[verb]["shift_aware_minus_swapped"].append(Lraw[A][cid] - Lraw[S][cid])
            scores[f"{pos}/{bn}"] = {verb: {k: {"n": len(v), "median": statistics.median(v), "yes_rate": _mean([float(x > 0) for x in v])}
                                            for k, v in d.items() if v}
                                     for verb, d in sorted(vals.items()) if sum(len(v) for v in d.values()) >= 5}
    # Every harm and help state against blind, by how it came about.
    how: dict[str, Counter] = {"harm": Counter(), "help": Counter()}
    for s in sts:
        ca, cb = D.choose("aware", s, L), D.choose("blind", s, L)
        c = cell(ca, cb)
        if c not in how:
            continue
        r1 = D.scored_candidates(s)[0]
        ab = len(D.scored_candidates(s)) > 1 and D.fell_back("aware", s, L)
        bb = len(D.scored_candidates(s)) > 1 and D.fell_back("blind", s, L)
        if c == "harm":
            if ca["key"] == r1["key"]:
                lab = "aware took failing rank 1 (abstained), blind chose a working alternative" if ab else \
                      "aware accepted failing rank 1, blind chose a working alternative"
            else:
                lab = f"aware chose a failing alternative ({kind(ca)}), blind chose working " + ("rank 1" if cb["key"] == r1["key"] else "alternative") + \
                      (" (blind abstained)" if bb else "")
        else:
            if cb["key"] == r1["key"]:
                lab = "blind took failing rank 1 (abstained), aware chose a working alternative" if bb else \
                      "blind accepted failing rank 1, aware chose a working alternative"
            else:
                lab = f"blind chose a failing alternative ({kind(cb)}), aware chose working " + ("rank 1" if ca["key"] == r1["key"] else "alternative") + \
                      (" (aware abstained)" if ab else "")
        how[c][lab] += 1
    # The cost of abstaining: states where aware abstains, rank 1 fails and a working candidate is listed.
    return {"fallback": fb, "working_candidate_scores": scores,
            "harm_help_by_route": {k: dict(v.most_common()) for k, v in how.items()}}


# ---------------- item 7: matched step bands ----------------


def item7(sts: list[dict], L: dict, Lraw: dict) -> dict:
    bands = {n: [s for s in sts if lo <= s["state_idx"] <= hi] for n, lo, hi in BANDS3}
    bands["even_steps_0_10"] = [s for s in sts if s["state_idx"] <= 10 and s["state_idx"] % 2 == 0]
    bands["even_steps_2_10"] = [s for s in sts if 2 <= s["state_idx"] <= 10 and s["state_idx"] % 2 == 0]
    bands["steps_1_29"] = [s for s in sts if s["state_idx"] >= 1]
    out: dict[str, Any] = {"aware_minus_blind": {}, "aware_minus_blind_fallback_max_L": {}, "swapped_minus_blind": {}, "aware_minus_swapped": {}}
    for n, b in bands.items():
        out["aware_minus_blind"][n] = {"n_states": len(b), **diff(b, L)}
        out["aware_minus_blind_fallback_max_L"][n] = diff(b, L, fallback="max_L")
        if S in L:
            sw = [s for s in b if s.get("swap")]
            out["swapped_minus_blind"][n] = diff(sw, L, "swapped", "blind")
            out["aware_minus_swapped"][n] = diff(sw, L, "aware", "swapped")
    # ingredient (i): Yes on working rank-1 actions, per arm and band (where the arm scored them)
    yes: dict[str, Any] = {}
    for n, lo, hi in BANDS3:
        row = {}
        for arm in (A, B, S, N):
            if arm not in L:
                continue
            v = [float(D.accepts(L[arm], D.cand_id(s, D.scored_candidates(s)[0]))) for s in bands[n]
                 if not D.scored_candidates(s)[0]["failed"] and D.cand_id(s, D.scored_candidates(s)[0]) in L[arm]]
            row[arm] = {"n": len(v), "yes_rate": _mean(v)}
        yes[n] = row
    out["yes_on_working_rank1"] = yes
    # ingredient (ii): what the agent's step-0 lists hold
    s0 = bands["step_0"]
    cands0 = [c for s in s0 for c in D.scored_candidates(s)]
    out["step0_lists"] = {"n_states": len(s0), "multi_candidate": sum(len(D.scored_candidates(s)) > 1 for s in s0),
                          "n_candidates": len(cands0), "failing": sum(c["failed"] for c in cands0),
                          "failing_goal_object": sum(c["failed"] and bool(c["targets_goal_object"]) for c in cands0),
                          "verbs": dict(Counter(c.get("action_type") for c in cands0).most_common()),
                          "rank1_verbs": dict(Counter(D.scored_candidates(s)[0].get("action_type") for s in s0).most_common())}
    return out


# ---------------- item 8: precision ----------------


def item8(frozen: dict, enc_rate: float) -> dict:
    out = {}
    for t in MODELS:
        m = frozen["models"][t]
        o, c = m["primary"]["difference"], m["conditional"]["difference"]
        out[t] = {"overall": {"estimate": o["estimate"], "ci": o["ci"], "n_games_nonzero": o["n_games_nonzero"],
                              "sign_flip_floor": o["p_min_sign_flip"],
                              "upper_bound_as_per_encounter_points": o["ci"][1] / enc_rate},
                  "conditional": {"estimate": c["estimate"], "ci": c["ci"], "n_games_nonzero": c["n_games_nonzero"],
                                  "sign_flip_floor": c["p_min_sign_flip"],
                                  "upper_bound_as_per_decision_points": c["ci"][1] * enc_rate}}
    return out


# ---------------- main ----------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--stage1a-dx", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    labs, scores, prov = load(a.runs)
    frozen = json.loads((a.runs / "decisionsim_agent_analysis" / "metrics.json").read_text(encoding="utf-8"))
    res: dict[str, Any] = {"meta": {"provenance_sha256": prov, "n_labelled_states": len(labs),
                                    "code": "scripts/decisionsim_agent_dx.py; frozen functions from wf/decisionsim_agent.py at 2a05707"}}
    res["item1_recompute"] = item1(labs, scores, frozen)
    models = {}
    for t in MODELS:
        sts, L, Lraw, short = frozen_states(labs, scores[t])
        models[t] = (sts, L, Lraw, short)
    enc_rate = frozen["encounter"]["encounter"]["estimate"]
    res["item2_per_game"] = {t: item2(sts, L) for t, (sts, L, _, _) in models.items()}
    res["item3_leave_one_game_out"] = {t: item3(sts, L) for t, (sts, L, _, _) in models.items()}
    res["item4_encounter_states"] = item4(models[MODELS[0]][0], {t: (sts, L) for t, (sts, L, _, _) in models.items()})
    res["item5_regime_candidates"] = {t: item5(sts, L) for t, (sts, L, _, _) in models.items()}
    res["item6_abstention"] = {t: item6(sts, L, Lraw) for t, (sts, L, Lraw, _) in models.items()}
    res["item7_step_bands"] = {t: item7(sts, L, Lraw) for t, (sts, L, Lraw, _) in models.items()}
    if a.stage1a_dx:
        s1 = json.loads(a.stage1a_dx.read_text(encoding="utf-8"))
        res["item7_stage1a_reference"] = {
            t: {"overall": s1["summary"][t]["dF_vs_blind"],
                "step0": s1["models"][t]["q1_recompute"]["by_step_vs_blind"]["step0"]["dF"],
                "later_even_steps_2_10": s1["models"][t]["q1_recompute"]["by_step_vs_blind"]["later"]["dF"],
                "switch_only_to_accepted_alternative": s1["models"][t]["q3_variants"]["switch_only_to_accepted_alternative"],
                "working_first_move_acceptance_step0": s1["models"][t]["q2_step0_mechanism"]["working_proposal_acceptance_by_step"]["0"]}
            for t in MODELS if t in s1["summary"]}
    res["item8_precision"] = {"encounter_rate": enc_rate, **item8(frozen, enc_rate)}
    a.out.write_text(json.dumps(r(res), indent=1, sort_keys=False, default=lambda o: dict(o)) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
