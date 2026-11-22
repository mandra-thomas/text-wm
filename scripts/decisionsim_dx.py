"""Exploratory, post hoc diagnostics of the decision-simulation Stage 1a results (notes/decisionsim_dx.md).

Nothing here is part of the frozen analysis (wf/decisionsim.py, tag decisionsim-stage1a-frozen-2026-11-21) and nothing
here scores anything. It re-reads the saved yes/no scores of `decisionsim_score_<model>` and the three item files of the
collection, rebuilds the eligible states with its own code, recomputes the gate quantities with its own bootstrap and
permutation, and then computes descriptive variants. The only things shared with the frozen code are the definitions it
must reproduce to be a recompute: eligibility, rounding to 1/64 nat, rule 2, and the seeded sha256 tie key.

Every interval is a 95% percentile interval from 2,000 resamples of whole games (seed 20260924); for a subset of states
the games resampled are the subset's own. Permutation p (only for the recomputed gate contrast) is a game sign-flip with
1,000 flips. Many cuts are examined; nothing is corrected for multiplicity.

Usage (repository root; score copies verified by sha256 against ICE):
    .venv/bin/python scripts/decisionsim_dx.py --transitions <dir with s0.jsonl s1.jsonl s2.jsonl> \
        --scores <dir with one folder per model holding scoring.jsonl> --official notes/decisionsim_metrics.json \
        --out notes/decisionsim_dx_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20260924
B = 2000
NPERM = 1000
NOTHING = "Nothing happens."
A, BL, BA, NE, SW = "yn_aligned", "yn_blind", "yn_blind_all", "yn_absent", "yn_swapped"
NAME = {A: "aligned", BL: "blind", BA: "blind_all", NE: "neutral", SW: "swapped"}
MODELS = ("qwen3-8b", "llama3.1-8b", "mistral-7b-v0.3", "gemma2-9b-it", "llama3.2-3b")
TOOL = {"clean": "sinkbasin", "heat": "microwave", "cool": "fridge"}


# ------------------------------------------------------------------ data


def ok(t: dict) -> bool:
    return (t.get("true_obs") or "").strip() != NOTHING


def rnd(x: float) -> float:
    return round(x * 64) / 64


def task_info(line: str) -> tuple[str, str | None]:
    """(task type, destination receptacle type) from the task line."""
    s = line.replace("Your task is to: ", "").rstrip(".")
    if s.startswith(("examine the", "look at")):
        return "examine", None
    dest = re.search(r" (?:in|on) (\w+)$", s)
    dest = dest.group(1) if dest else None
    if "two " in s:
        return "pick_two", dest
    for verb, adj, kind in (("clean", "clean", "clean"), ("heat", "hot", "heat"), ("cool", "cool", "cool")):
        if s.startswith(verb + " ") or f" {adj} " in s:
            return kind, dest
    return "pick", dest


def load_states(tdir: Path) -> list[dict]:
    ts = [json.loads(line) for s in (0, 1, 2) for line in open(tdir / f"s{s}.jsonl")]
    agent: dict[tuple[str, int], str] = {}
    for t in ts:
        for i, a in enumerate(t.get("prefix_actions") or []):
            if agent.setdefault((t["game_id"], i), a) != a:
                raise ValueError("inconsistent agent actions")
    by: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for t in ts:
        by[(t["game_id"], t["state_idx"])].append(t)
    out = []
    for (g, i), cs in sorted(by.items()):
        prop = agent.get((g, i))
        p = [t for t in cs if t["action"] == prop]
        if not p or not any(ok(t) for t in cs) or not any(not ok(t) and t.get("context_determined") for t in cs):
            continue
        ttype, dest = task_info(cs[0]["task_line"])
        goal = cs[0]["goal_object"]
        locs = cs[0].get("scene_locations") or {}
        goal_recs = {r for k, v in locs.items() if k.rsplit(" ", 1)[0] == goal for r in v}
        lamp_recs = {r for k, v in locs.items() if k.rsplit(" ", 1)[0] in ("desklamp", "floorlamp") for r in v}
        out.append({"game": g, "step": i, "prop": p[0], "cands": cs, "gen": [t for t in cs if t["action"] != prop],
                    "ttype": ttype, "dest": dest, "goal": goal, "goal_recs": goal_recs, "lamp_recs": lamp_recs,
                    "task_line": cs[0]["task_line"]})
    return out


def goto_relevance(s: dict, t: dict) -> dict[str, bool] | None:
    """For a `go to R` action: is R the task's destination type, the task's tool (sinkbasin, microwave, fridge, or
    where a lamp is for examine tasks), or a receptacle holding an instance of the goal object (from the hidden scene,
    which the history may not yet show)?"""
    m = re.fullmatch(r"go to (\w+) (\d+)", t["action"])
    if not m:
        return None
    rec, rtype = f"{m.group(1)} {m.group(2)}", m.group(1)
    tool = (rtype == TOOL.get(s["ttype"])) or (s["ttype"] == "examine" and rec in s["lamp_recs"])
    flags = {"destination": rtype == s["dest"], "tool": tool, "goal_location": rec in s["goal_recs"]}
    flags["any"] = any(flags.values())
    return flags


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    L: dict[str, dict[str, float]] = defaultdict(dict)
    for line in open(path):
        x = json.loads(line)
        if x["L"] == x["L"]:
            L[x["condition"]][x["transition_id"]] = rnd(x["L"])
    return dict(L)


def complete(states: list[dict], L: dict) -> tuple[list[dict], dict]:
    """Keep states with every arm (but swapped) scored for every candidate; mark the swap states. A partly scored swap
    state is an error."""
    keep, partial = [], 0
    for s in states:
        ids = [t["id"] for t in s["cands"]]
        if not all(all(i in L[a] for i in ids) for a in L if a != SW):
            continue
        have = [i in L.get(SW, {}) for i in ids]
        if any(have) and not all(have):
            partial += 1
        keep.append({**s, "swap": all(have) and SW in L})
    return keep, {"n_complete": len(keep), "n_dropped": len(states) - len(keep), "n_partial_swap": partial}


# ------------------------------------------------------------------ choice


def tie(s: dict, t: dict, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}|{s['game']}|{s['step']}|{t['action']}".encode()).hexdigest()


def pick(s: dict, menu: list[dict], L: dict, seed: int = SEED) -> dict:
    return min(menu, key=lambda t: (-L[t["id"]], tie(s, t, seed)))


def rule2(s: dict, Lacc: dict, Lfb: dict, thr: float = 0.0, filt=None, seed: int = SEED) -> dict:
    """Accept the proposal if its score exceeds thr, else the best of the other candidates (optionally filtered; if
    the filter leaves nothing, the proposal is kept)."""
    p = s["prop"]
    if Lacc[p["id"]] > thr:
        return p
    menu = [t for t in s["gen"] if filt is None or filt(s, t)]
    return pick(s, menu, Lfb, seed) if menu else p


def rule2_split(s: dict, Lacc: dict, Lfb: dict, thr: float = 0.0) -> float:
    """Rule 2 with every tied top set of the fallback split evenly (share of failures)."""
    p = s["prop"]
    if Lacc[p["id"]] > thr:
        return float(not ok(p))
    top = max(Lfb[t["id"]] for t in s["gen"])
    tied = [t for t in s["gen"] if Lfb[t["id"]] == top]
    return sum(not ok(t) for t in tied) / len(tied)


# ------------------------------------------------------------------ inference


def _weights(games: list[str], seed: int = SEED, b: int = B):
    keys = sorted(set(games))
    k = {g: i for i, g in enumerate(keys)}
    idx = np.array([k[g] for g in games])
    C = np.random.default_rng(seed).multinomial(len(keys), [1.0 / len(keys)] * len(keys), size=b).astype(float)
    return idx, C, len(keys)


def boot(values, games, seed: int = SEED) -> dict:
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"est": None, "ci": [None, None], "n": 0, "games": 0}
    idx, C, K = _weights(list(games), seed)
    S, N = np.bincount(idx, weights=v, minlength=K), np.bincount(idx, minlength=K).astype(float)
    den = C @ N
    est = (C @ S) / np.where(den > 0, den, np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"est": float(v.mean()), "ci": [float(lo), float(hi)], "n": int(len(v)), "games": K}


def perm_p(diffs, games, seed: int = SEED, n: int = NPERM) -> float:
    keys = sorted(set(games))
    k = {g: i for i, g in enumerate(keys)}
    S = np.bincount([k[g] for g in games], weights=np.asarray(diffs, float), minlength=len(keys))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(keys)))
    obs = abs(S.sum())
    return float((np.sum(np.abs(signs @ S) >= obs - 1e-9) + 1) / (n + 1))


def boot_auc(pos_scores, pos_games, neg_scores, neg_games, seed: int = SEED) -> dict:
    """AUC (ties count one half) of positives over negatives, with a game bootstrap that resamples both sides with the
    same game weights."""
    if not pos_scores or not neg_scores:
        return {"est": None, "ci": [None, None], "n_pos": len(pos_scores), "n_neg": len(neg_scores)}
    games = list(pos_games) + list(neg_games)
    idx, C, K = _weights(games, seed)
    ip, ineg = idx[: len(pos_scores)], idx[len(pos_scores):]
    ps, ns = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    M = (ps[:, None] > ns[None, :]) + 0.5 * (ps[:, None] == ns[None, :])
    Wp, Wn = C[:, ip], C[:, ineg]
    num = np.einsum("bi,ij,bj->b", Wp, M, Wn)
    den = Wp.sum(1) * Wn.sum(1)
    est = num / np.where(den > 0, den, np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"est": float(M.mean()), "ci": [float(lo), float(hi)], "n_pos": len(ps), "n_neg": len(ns)}


def quant(xs) -> dict:
    if not len(xs):
        return {}
    v = np.asarray(xs, float)
    return {k: float(np.percentile(v, q)) for k, q in (("p10", 10), ("q25", 25), ("median", 50), ("q75", 75), ("p90", 90))} | {
        "mean": float(v.mean()), "n": int(len(v))}


BINS = [(-1e9, -10), (-10, -2), (-2, 0), (0, 2), (2, 10), (10, 1e9)]
BIN_NAMES = ["<=-10", "(-10,-2]", "(-2,0]", "(0,2]", "(2,10]", ">10"]


def hist(xs) -> dict:
    n = max(1, len(xs))
    return {nm: sum(lo < x <= hi for x in xs) / n for nm, (lo, hi) in zip(BIN_NAMES, BINS)}


# ------------------------------------------------------------------ contrasts


def compare(states: list[dict], pa, pb, with_p: bool = False) -> dict:
    """Policies pa, pb: state -> failure indicator (float, or a share for tie splits)."""
    fa = [pa(s) for s in states]
    fb = [pb(s) for s in states]
    g = [s["game"] for s in states]
    d = [x - y for x, y in zip(fa, fb)]
    out = {"n": len(states), "F_a": boot(fa, g), "F_b": boot(fb, g), "dF": boot(d, g),
           "n_harm": int(sum(x == 1 and y == 0 for x, y in zip(fa, fb))),
           "n_help": int(sum(x == 0 and y == 1 for x, y in zip(fa, fb)))}
    out["net"] = out["n_harm"] - out["n_help"]
    if with_p:
        out["dF"]["p"] = perm_p(d, g)
    return out


def fail_of(choice_fn):
    return lambda s: float(not ok(choice_fn(s)))


def r2(La, Lb=None, **kw):
    return fail_of(lambda s: rule2(s, La, Lb if Lb is not None else La, **kw))


def logo(states: list[dict], pa, pb, key) -> dict:
    """Leave one group out (games or task types): the range of the aligned-minus-comparison difference."""
    fa = {id(s): pa(s) for s in states}
    fb = {id(s): pb(s) for s in states}
    groups = sorted({key(s) for s in states})
    res = {}
    for gname in groups:
        rest = [s for s in states if key(s) != gname]
        res[gname] = sum(fa[id(s)] - fb[id(s)] for s in rest) / len(rest)
    vals = list(res.values())
    worst = min(res, key=res.get)
    best = max(res, key=res.get)
    return {"n_groups": len(groups), "min": min(vals), "max": max(vals), "min_dropped": worst, "max_dropped": best,
            "values": res if len(groups) <= 10 else None}


# ------------------------------------------------------------------ per model


def analyse(states: list[dict], L: dict, model: str, full: bool) -> dict:
    ev = [s for s in states if s["step"] % 2 == 0]
    sw = [s for s in ev if s["swap"]]
    s0 = [s for s in ev if s["step"] == 0]
    later = [s for s in ev if s["step"] > 0]
    arms = [a for a in (A, BL, BA, NE, SW) if a in L]
    none = lambda s: float(not ok(s["prop"]))
    out: dict = {"n_even": len(ev), "n_even_swap": len(sw), "n_step0": len(s0), "n_later": len(later), "arms": [NAME[a] for a in arms]}

    # ---------------- Q1: recompute
    q1: dict = {"F_no_verifier": boot([none(s) for s in ev], [s["game"] for s in ev])}
    for a in arms:
        sts = sw if a == SW else ev
        q1[f"F_{NAME[a]}"] = boot([r2(L[a])(s) for s in sts], [s["game"] for s in sts])
    q1["contrasts"] = {}
    for b in [x for x in (BL, BA, NE, SW) if x in L]:
        sts = sw if b == SW else ev
        q1["contrasts"][f"aligned_vs_{NAME[b]}"] = compare(sts, r2(L[A]), r2(L[b]), with_p=True)
    if SW in L:
        q1["contrasts"]["swapped_vs_blind"] = compare(sw, r2(L[SW]), r2(L[BL]), with_p=True)
        q1["contrasts"]["aligned_vs_blind_on_swap_states"] = compare(sw, r2(L[A]), r2(L[BL]))
    cells = defaultdict(lambda: {"n": 0, "harm": 0, "help": 0})
    for s in ev:
        p = s["prop"]
        c = ("acc" if L[A][p["id"]] > 0 else "rej") + "_aligned/" + ("acc" if L[BL][p["id"]] > 0 else "rej") + "_blind"
        fa, fb = r2(L[A])(s), r2(L[BL])(s)
        cells[c]["n"] += 1
        cells[c]["harm"] += int(fa == 1 and fb == 0)
        cells[c]["help"] += int(fa == 0 and fb == 1)
    q1["acceptance_cells_vs_blind"] = {k: {**v, "net": v["harm"] - v["help"]} for k, v in sorted(cells.items())}
    q1["by_step_vs_blind"] = {"step0": compare(s0, r2(L[A]), r2(L[BL])), "later": compare(later, r2(L[A]), r2(L[BL]))}
    if SW in L:
        q1["by_step_vs_swapped"] = {"step0": compare([s for s in s0 if s["swap"]], r2(L[A]), r2(L[SW])),
                                    "later": compare([s for s in later if s["swap"]], r2(L[A]), r2(L[SW]))}
    q1["by_step_vs_neutral"] = {"step0": compare(s0, r2(L[A]), r2(L[NE])), "later": compare(later, r2(L[A]), r2(L[NE]))}
    q1["leave_one_game_out_vs_blind"] = logo(ev, r2(L[A]), r2(L[BL]), lambda s: s["game"])
    q1["leave_one_task_type_out_vs_blind"] = logo(ev, r2(L[A]), r2(L[BL]), lambda s: s["ttype"])
    harm_by_game = Counter(s["game"] for s in ev if r2(L[A])(s) == 1 and r2(L[BL])(s) == 0)
    q1["harm_concentration"] = {"games_with_harm": len(harm_by_game), "max_harm_in_one_game": max(harm_by_game.values(), default=0),
                                "top10_games_share_of_harm": sum(v for _, v in harm_by_game.most_common(10)) / max(1, sum(harm_by_game.values()))}
    q1["by_task_type"] = {}
    for tt in ("pick", "pick_two", "clean", "heat", "cool", "examine"):
        sts = [s for s in ev if s["ttype"] == tt]
        q1["by_task_type"][tt] = {"n_games": len({s["game"] for s in sts}),
                                  "all": compare(sts, r2(L[A]), r2(L[BL])),
                                  "step0": compare([s for s in sts if s["step"] == 0], r2(L[A]), r2(L[BL])),
                                  "later": compare([s for s in sts if s["step"] > 0], r2(L[A]), r2(L[BL]))}
    out["q1_recompute"] = q1

    # ---------------- Q2: step 0 mechanism (and the same readouts later)
    q2: dict = {"proposal_types_step0": dict(Counter(s["prop"]["action_type"] for s in s0)),
                "proposal_types_later": dict(Counter(s["prop"]["action_type"] for s in later))}
    acc = {}
    for label, sts in (("step0", s0), ("later", later)):
        acc[label] = {}
        for a in arms:
            use = [s for s in sts if s["swap"]] if a == SW else sts
            wp = [s for s in use if ok(s["prop"])]
            gens_go = [(s, t) for s in use for t in s["gen"] if ok(t) and t["action_type"] == "go to"]
            gens_goal_fail = [(s, t) for s in use for t in s["gen"] if not ok(t) and t.get("targets_goal_object")]
            any_yes = [float(any(L[a][t["id"]] > 0 for t in s["cands"])) for s in use]
            acc[label][NAME[a]] = {
                "n_states": len(use),
                "accept_working_proposal": boot([float(L[a][s["prop"]["id"]] > 0) for s in wp], [s["game"] for s in wp]),
                "L_working_proposal": quant([L[a][s["prop"]["id"]] for s in wp]),
                "yes_working_generated_goto": sum(L[a][t["id"]] > 0 for _, t in gens_go) / max(1, len(gens_go)),
                "L_working_generated_goto": quant([L[a][t["id"]] for _, t in gens_go]),
                "yes_failing_goal_object_generated": sum(L[a][t["id"]] > 0 for _, t in gens_goal_fail) / max(1, len(gens_goal_fail)),
                "L_failing_goal_object_generated": quant([L[a][t["id"]] for _, t in gens_goal_fail]),
                "share_states_no_candidate_yes": sum(1 - x for x in any_yes) / max(1, len(any_yes)),
            }
    q2["acceptance_and_scores"] = acc
    by_step = {}
    for k in sorted({s["step"] for s in ev}):
        wp = [s for s in ev if s["step"] == k and ok(s["prop"])]
        by_step[str(k)] = {"n_working": len(wp)} | {
            NAME[a]: sum(L[a][s["prop"]["id"]] > 0 for s in wp if a != SW or s["swap"]) / max(1, sum(1 for s in wp if a != SW or s["swap"]))
            for a in arms} | {"aligned_rejects_blind_accepts": sum(L[A][s["prop"]["id"]] <= 0 < L[BL][s["prop"]["id"]] for s in wp),
                              "harm_vs_blind": sum(r2(L[A])(s) == 1 and r2(L[BL])(s) == 0 for s in ev if s["step"] == k)}
    q2["working_proposal_acceptance_by_step"] = by_step
    # go-to relevance, every working go-to candidate (proposal or generated)
    rel: dict = {}
    for label, sts in (("step0", s0), ("later", later)):
        rel[label] = {}
        rows = [(s, t, goto_relevance(s, t)) for s in sts for t in s["cands"] if ok(t) and goto_relevance(s, t) is not None]
        for cls in ("any", "destination", "tool", "goal_location", "irrelevant"):
            sel = [(s, t) for s, t, f in rows if (not f["any"] if cls == "irrelevant" else f[cls])]
            entry = {"n": len(sel)}
            for a in arms:
                ss = [(s, t) for s, t in sel if a != SW or s["swap"]]
                entry[NAME[a]] = {"yes": sum(L[a][t["id"]] > 0 for _, t in ss) / max(1, len(ss)),
                                  "median_L": float(np.median([L[a][t["id"]] for _, t in ss])) if ss else None}
            entry["shift_aligned_minus_blind"] = boot([L[A][t["id"]] - L[BL][t["id"]] for _, t in sel], [s["game"] for s, _ in sel]) if sel else None
            if SW in L:
                ss = [(s, t) for s, t in sel if s["swap"]]
                entry["shift_aligned_minus_swapped"] = boot([L[A][t["id"]] - L[SW][t["id"]] for _, t in ss], [s["game"] for s, _ in ss]) if ss else None
            rel[label][cls] = entry
        # within-state: relevant minus irrelevant working go-to, aligned L, at states with both
        diffs, gs = [], []
        for s in sts:
            r = [t for t in s["cands"] if ok(t) and (f := goto_relevance(s, t)) is not None and f["any"]]
            ir = [t for t in s["cands"] if ok(t) and (f := goto_relevance(s, t)) is not None and not f["any"]]
            if r and ir:
                diffs.append(np.mean([L[A][t["id"]] for t in r]) - np.mean([L[A][t["id"]] for t in ir]))
                gs.append(s["game"])
        rel[label]["within_state_relevant_minus_irrelevant_aligned"] = boot(diffs, gs)
        diffs_b = []
        for s in sts:
            r = [t for t in s["cands"] if ok(t) and (f := goto_relevance(s, t)) is not None and f["any"]]
            ir = [t for t in s["cands"] if ok(t) and (f := goto_relevance(s, t)) is not None and not f["any"]]
            if r and ir:
                diffs_b.append(np.mean([L[BL][t["id"]] for t in r]) - np.mean([L[BL][t["id"]] for t in ir]))
        rel[label]["within_state_relevant_minus_irrelevant_blind"] = boot(diffs_b, gs)
    q2["goto_relevance"] = rel
    # what the aligned fallback picks where only the blind verifier accepts a working proposal
    for label, sts in (("step0", s0), ("later", later)):
        cases = [s for s in sts if ok(s["prop"]) and L[A][s["prop"]["id"]] <= 0 and L[BL][s["prop"]["id"]] > 0]
        picks = [(s, rule2(s, L[A], L[A])) for s in cases]
        fails = [(s, c) for s, c in picks if not ok(c)]
        gaps = []
        for s, c in fails:
            best_ok = max((L[A][t["id"]] for t in s["gen"] if ok(t)), default=None)
            if best_ok is not None:
                gaps.append(L[A][c["id"]] - best_ok)
        q2[f"aligned_only_rejections_{label}"] = {
            "n": len(cases), "fallback_fails": len(fails),
            "fallback_action_type": dict(Counter(c["action_type"] for _, c in picks)),
            "failing_fallback_goal_object": sum(bool(c.get("targets_goal_object")) for _, c in fails),
            "failing_fallback_reason": dict(Counter(c.get("determination_reason") for _, c in fails)),
            "failing_fallback_L": quant([L[A][c["id"]] for _, c in fails]),
            "gap_failing_pick_minus_best_working_alternative": quant(gaps),
            "share_gap_at_most_2_nats": sum(g <= 2 for g in gaps) / max(1, len(gaps)),
        }
    # factorial: accept/reject from one arm, fallback ranking from another
    fact: dict = {}
    pairs = [(x, y) for x in arms for y in arms if x in (A, BL, SW, NE) and y in (A, BL, SW, NE)]
    for label, sts in (("all", ev), ("step0", s0), ("later", later)):
        use = [s for s in sts if s["swap"]] if SW in L else sts
        fact[label] = {"n": len(use)}
        for x, y in pairs:
            fact[label][f"acc_{NAME[x]}__fb_{NAME[y]}"] = sum(r2(L[x], L[y])(s) for s in use) / max(1, len(use))
        if SW in L:
            # interval for the two halves of aligned minus blind, and of aligned minus swapped
            g = [s["game"] for s in use]
            fact[label]["acceptance_part_vs_blind"] = boot([r2(L[A], L[BL])(s) - r2(L[BL])(s) for s in use], g)
            fact[label]["fallback_part_vs_blind"] = boot([r2(L[A])(s) - r2(L[A], L[BL])(s) for s in use], g)
            fact[label]["acceptance_part_vs_swapped"] = boot([r2(L[A], L[SW])(s) - r2(L[SW])(s) for s in use], g)
            fact[label]["fallback_part_vs_swapped"] = boot([r2(L[A])(s) - r2(L[A], L[SW])(s) for s in use], g)
    q2["factorial_on_swap_states"] = fact
    out["q2_step0_mechanism"] = q2

    # ---------------- Q3: rule-design and menu variants (descriptive)
    q3: dict = {}

    def vs_blind(sts, fa_fn, fb_fn):
        c = compare(sts, fa_fn, fb_fn)
        return {"F_aligned": c["F_a"]["est"], "F_blind": c["F_b"]["est"], "dF": c["dF"], "net": c["net"]}

    thr = {}
    for k in (-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8):
        thr[str(k)] = {"all": vs_blind(ev, r2(L[A], thr=k), r2(L[BL], thr=k)),
                       "step0": vs_blind(s0, r2(L[A], thr=k), r2(L[BL], thr=k)),
                       "later": vs_blind(later, r2(L[A], thr=k), r2(L[BL], thr=k))}
        if SW in L:
            thr[str(k)]["aligned_minus_swapped"] = compare(sw, r2(L[A], thr=k), r2(L[SW], thr=k))["dF"]["est"]
    q3["threshold_shared"] = thr
    # per-arm threshold matched to the blind verifier's acceptance share of proposals
    share_b = sum(L[BL][s["prop"]["id"]] > 0 for s in ev) / len(ev)
    matched = {}
    for a in arms:
        sc = sorted((L[a][s["prop"]["id"]] for s in (sw if a == SW else ev)), reverse=True)
        k = int(round(share_b * len(sc)))
        matched[NAME[a]] = sc[k] if k < len(sc) else sc[-1] - 1
    q3["matched_acceptance_thresholds"] = matched
    q3["matched_acceptance"] = {
        "all": vs_blind(ev, r2(L[A], thr=matched["aligned"]), r2(L[BL])),
        "step0": vs_blind(s0, r2(L[A], thr=matched["aligned"]), r2(L[BL])),
        "later": vs_blind(later, r2(L[A], thr=matched["aligned"]), r2(L[BL]))}
    # the best single threshold for each arm, chosen in sample (a lower bound on what tuning could achieve)
    best = {}
    for a in arms:
        sts = sw if a == SW else ev
        pf = [float(not ok(s["prop"])) for s in sts]
        fb = [float(not ok(pick(s, s["gen"], L[a]))) for s in sts]
        lp = [L[a][s["prop"]["id"]] for s in sts]
        cand_thr = sorted(set(lp) | {0.0}) + [1e9]
        scores = [(sum(x if l >= t else y for x, y, l in zip(pf, fb, lp)) / len(sts), t) for t in cand_thr]
        f, t = min(scores)
        best[NAME[a]] = {"F_min": f, "threshold_accept_at_or_above": t}
    q3["best_threshold_in_sample"] = best
    # discrimination of the proposal's own outcome (later steps hold every failing proposal)
    auc = {}
    for a in arms:
        sts = sw if a == SW else ev
        pos = [s for s in sts if ok(s["prop"])]
        neg = [s for s in sts if not ok(s["prop"])]
        auc[NAME[a]] = {"all_even": boot_auc([L[a][s["prop"]["id"]] for s in pos], [s["game"] for s in pos],
                                             [L[a][s["prop"]["id"]] for s in neg], [s["game"] for s in neg]),
                        "later_only": boot_auc([L[a][s["prop"]["id"]] for s in pos if s["step"] > 0], [s["game"] for s in pos if s["step"] > 0],
                                               [L[a][s["prop"]["id"]] for s in neg], [s["game"] for s in neg])}
    q3["proposal_auc"] = auc
    # within-state ranking quality of the fallback menu (generated candidates): mean per-state AUC
    rank = {}
    for a in arms:
        rank[NAME[a]] = {}
        for label, sts in (("step0", s0), ("later", later)):
            use = [s for s in sts if s["swap"]] if a == SW else sts
            vals, gs = [], []
            for s in use:
                po = [L[a][t["id"]] for t in s["gen"] if ok(t)]
                ne = [L[a][t["id"]] for t in s["gen"] if not ok(t)]
                if po and ne:
                    vals.append(float(np.mean([(x > y) + 0.5 * (x == y) for x in po for y in ne])))
                    gs.append(s["game"])
            rank[NAME[a]][label] = boot(vals, gs)
    q3["fallback_menu_within_state_auc"] = rank
    filters = {
        "non_goal_object": lambda s, t: not t.get("targets_goal_object"),
        "same_action_type_as_proposal": lambda s, t: t["action_type"] == s["prop"]["action_type"],
        "go_to_or_look_at_step0_else_all": lambda s, t: s["step"] > 0 or t["action_type"] in ("go to", "look"),
        "oracle_fallback_equals_admissible_only": lambda s, t: ok(t),
    }
    fv = {}
    for nm, f in filters.items():
        fv[nm] = {"all": vs_blind(ev, r2(L[A], filt=f), r2(L[BL], filt=f)),
                  "step0": vs_blind(s0, r2(L[A], filt=f), r2(L[BL], filt=f)),
                  "later": vs_blind(later, r2(L[A], filt=f), r2(L[BL], filt=f)),
                  "n_states_filter_empty": sum(not [t for t in s["gen"] if f(s, t)] for s in ev)}
        if SW in L:
            fv[nm]["aligned_minus_swapped"] = compare(sw, r2(L[A], filt=f), r2(L[SW], filt=f))["dF"]["est"]
    q3["fallback_filters"] = fv
    # rule 2 that switches only to an alternative the verifier itself accepts (L > 0), else keeps the proposal
    def r2_cons(La):
        def f(s):
            p = s["prop"]
            if La[p["id"]] > 0:
                return float(not ok(p))
            menu = [t for t in s["gen"] if La[t["id"]] > 0]
            return float(not ok(pick(s, menu, La) if menu else p))
        return f
    q3["switch_only_to_accepted_alternative"] = {"all": vs_blind(ev, r2_cons(L[A]), r2_cons(L[BL])),
                                                 "step0": vs_blind(s0, r2_cons(L[A]), r2_cons(L[BL])),
                                                 "later": vs_blind(later, r2_cons(L[A]), r2_cons(L[BL])),
                                                 "F_no_verifier": sum(none(s) for s in ev) / len(ev)}
    if SW in L:
        q3["switch_only_to_accepted_alternative"]["aligned_minus_swapped"] = compare(sw, r2_cons(L[A]), r2_cons(L[SW]))["dF"]["est"]
    # the frozen gate's conditions applied to each variant (descriptive: the gate is defined for rule 2 as frozen)
    if SW in L:
        def gate(sts, fa, fb, fs, acc_a, acc_b):
            d = [fa(s) - fb(s) for s in sts]
            g = [s["game"] for s in sts]
            b = boot(d, g)
            pp = perm_p(d, g)
            net = int(round(sum(d)))
            head = sum(not ok(s["prop"]) for s in sts)
            rb = int(round(sum(fa(s) - fb(s) for s in sts if not acc_a(s) and not acc_b(s))))
            ss = [s for s in sts if s["swap"]]
            ab = sum(fa(s) - fb(s) for s in ss) / len(ss)
            as_ = sum(fa(s) - fs(s) for s in ss) / len(ss)
            tf = b["ci"][0] > 0 and pp < 0.05
            enough = net >= head / 2
            attr = ab > 1e-12 and as_ > 1e-12 and as_ - ab / 2 >= -1e-12
            via_fb = rb - net / 2 > 1e-12
            cat = ("worth doing" if tf and enough and attr and not via_fb else "through the fallback menu" if tf and enough and attr
                   else "not attributable" if tf and enough else "too rare" if tf else "toward success" if b["ci"][1] < 0 else "no reliable change")
            return {"category": cat, "dF": b, "p": pp, "net": net, "half_headroom": head / 2, "rejected_both_net": rb,
                    "aligned_minus_blind_swap_states": ab, "aligned_minus_swapped": as_}
        acc0 = lambda La: (lambda s: La[s["prop"]["id"]] > 0)
        gv = {"as_frozen": gate(ev, r2(L[A]), r2(L[BL]), r2(L[SW]), acc0(L[A]), acc0(L[BL])),
              "switch_only_to_accepted_alternative": gate(ev, r2_cons(L[A]), r2_cons(L[BL]), r2_cons(L[SW]), acc0(L[A]), acc0(L[BL]))}
        for nm in ("non_goal_object", "go_to_or_look_at_step0_else_all"):
            fl = filters[nm]
            gv[f"fallback_{nm}"] = gate(ev, r2(L[A], filt=fl), r2(L[BL], filt=fl), r2(L[SW], filt=fl), acc0(L[A]), acc0(L[BL]))
        gv["later_steps_only"] = gate(later, r2(L[A]), r2(L[BL]), r2(L[SW]), acc0(L[A]), acc0(L[BL]))
        ta, ts_ = matched["aligned"], matched["swapped"]
        gv["matched_acceptance_threshold"] = gate(ev, r2(L[A], thr=ta), r2(L[BL]), r2(L[SW], thr=ts_),
                                                  lambda s: L[A][s["prop"]["id"]] > ta, acc0(L[BL]))
        q3["gate_under_variants"] = gv
    nl = [s for s in ev if s["prop"]["action_type"] not in ("look", "inventory")]
    q3["excluding_look_inventory_proposals"] = vs_blind(nl, r2(L[A]), r2(L[BL])) | {"n": len(nl)}
    q3["later_steps_only"] = vs_blind(later, r2(L[A]), r2(L[BL]))
    q3["later_steps_excluding_step2"] = vs_blind([s for s in later if s["step"] > 2], r2(L[A]), r2(L[BL]))
    # rules 1 and 3 and the lure rate
    r1 = lambda La, gen=False: fail_of(lambda s: pick(s, s["gen"] if gen else s["cands"], La))
    gr = lambda s: [t for t in s["cands"] if t.get("targets_goal_object")]
    r3s = [s for s in ev if any(ok(t) for t in gr(s))]
    r3 = lambda La: fail_of(lambda s: pick(s, gr(s), La))
    q3["rule1_with_proposal"] = {"all": vs_blind(ev, r1(L[A]), r1(L[BL])), "step0": vs_blind(s0, r1(L[A]), r1(L[BL])),
                                 "later": vs_blind(later, r1(L[A]), r1(L[BL]))}
    q3["rule1_generated_only"] = {"all": vs_blind(ev, r1(L[A], True), r1(L[BL], True)),
                                  "step0": vs_blind(s0, r1(L[A], True), r1(L[BL], True)),
                                  "later": vs_blind(later, r1(L[A], True), r1(L[BL], True))}
    q3["rule3_goal_relevant"] = vs_blind(r3s, r3(L[A]), r3(L[BL])) | {"n": len(r3s)}
    if SW in L:
        q3["rule1_with_proposal"]["aligned_minus_swapped"] = compare(sw, r1(L[A]), r1(L[SW]))["dF"]["est"]
        q3["rule1_generated_only"]["aligned_minus_swapped"] = compare(sw, r1(L[A], True), r1(L[SW], True))["dF"]["est"]
    lure = []
    for s in ev:
        if not ok(s["prop"]):
            continue
        for t in s["gen"]:
            if t.get("targets_goal_object") and not ok(t):
                la = float(pick(s, [s["prop"], t], L[A]) is t)
                lb = float(pick(s, [s["prop"], t], L[BL]) is t)
                lure.append((s["game"], s["step"], la, lb))
    q3["lure_rate"] = {"n_pairs": len(lure),
                       "aligned": boot([x[2] for x in lure], [x[0] for x in lure]),
                       "blind": boot([x[3] for x in lure], [x[0] for x in lure]),
                       "aligned_step0": sum(x[2] for x in lure if x[1] == 0) / max(1, sum(x[1] == 0 for x in lure)),
                       "aligned_later": sum(x[2] for x in lure if x[1] > 0) / max(1, sum(x[1] > 0 for x in lure))}
    # ties: tied top sets split evenly, and the gate difference over 200 other tie seeds
    split = [rule2_split(s, L[A], L[A]) - rule2_split(s, L[BL], L[BL]) for s in ev]
    seeds = []
    for sd in range(200):
        seeds.append(sum(r2(L[A], seed=sd)(s) - r2(L[BL], seed=sd)(s) for s in ev) / len(ev))
    q3["ties"] = {"split_even_dF": float(np.mean(split)), "seed_dF": quant(seeds),
                  "tied_top_fallback_states_aligned": sum(
                      (L[A][s["prop"]["id"]] <= 0) and sum(L[A][t["id"]] == max(L[A][u["id"]] for u in s["gen"]) for t in s["gen"]) > 1 for s in ev)}
    out["q3_variants"] = q3

    # ---------------- Q4: calibration of the proposal's score
    q4: dict = {}
    for label, sts in (("step0", s0), ("later", later)):
        q4[label] = {}
        for a in arms:
            use = [s for s in sts if s["swap"]] if a == SW else sts
            w = [L[a][s["prop"]["id"]] for s in use if ok(s["prop"])]
            f = [L[a][s["prop"]["id"]] for s in use if not ok(s["prop"])]
            q4[label][NAME[a]] = {"working": quant(w), "working_hist": hist(w), "failing": quant(f), "failing_hist": hist(f)}
        wp = [s for s in sts if ok(s["prop"])]
        sh = [L[A][s["prop"]["id"]] - L[BL][s["prop"]["id"]] for s in wp]
        q4[label]["shift_aligned_minus_blind_working"] = quant(sh)
        q4[label]["shift_hist"] = hist(sh)
        flips = [s for s in wp if L[A][s["prop"]["id"]] <= 0 < L[BL][s["prop"]["id"]]]
        q4[label]["aligned_only_rejections"] = {
            "n": len(flips),
            "blind_L": quant([L[BL][s["prop"]["id"]] for s in flips]),
            "aligned_L": quant([L[A][s["prop"]["id"]] for s in flips]),
            "shift": quant([L[A][s["prop"]["id"]] - L[BL][s["prop"]["id"]] for s in flips]),
            "share_blind_L_at_most_2": sum(L[BL][s["prop"]["id"]] <= 2 for s in flips) / max(1, len(flips)),
            "share_aligned_L_above_minus_2": sum(L[A][s["prop"]["id"]] > -2 for s in flips) / max(1, len(flips)),
            "share_shift_below_minus_10": sum(L[A][s["prop"]["id"]] - L[BL][s["prop"]["id"]] < -10 for s in flips) / max(1, len(flips)),
        }
    out["q4_calibration"] = q4

    # ---------------- Q5: per-candidate shifts by group (link to the item-level effects)
    q5: dict = {}
    comps = [(A, BL), (A, SW), (SW, BL), (A, NE), (NE, BL)]
    comps = [(x, y) for x, y in comps if x in L and y in L]
    for label, sts in (("step0", s0), ("later", later)):
        q5[label] = {}
        groups = defaultdict(list)
        for s in sts:
            for t in s["cands"]:
                at = t["action_type"] if t["action_type"] in ("go to", "take", "look") else "other"
                key = f"{'goal' if t.get('targets_goal_object') else 'other_obj'}/{at}/{'success' if ok(t) else 'failure'}"
                groups[key].append((s, t))
        for key, rows in sorted(groups.items()):
            if len(rows) < 10:
                continue
            e = {"n": len(rows)}
            for x, y in comps:
                rr = [(s, t) for s, t in rows if SW not in (x, y) or s["swap"]]
                if rr:
                    e[f"{NAME[x]}_minus_{NAME[y]}"] = boot([L[x][t["id"]] - L[y][t["id"]] for _, t in rr], [s["game"] for s, _ in rr])
            for a in arms:
                rr = [(s, t) for s, t in rows if a != SW or s["swap"]]
                e[f"yes_{NAME[a]}"] = sum(L[a][t["id"]] > 0 for _, t in rr) / max(1, len(rr))
            q5[label][key] = e
    out["q5_shifts_by_group"] = q5

    out["official_crosscheck"] = None  # filled by main
    return out


def summary_line(rep: dict) -> dict:
    q1 = rep["q1_recompute"]
    c = q1["contrasts"]["aligned_vs_blind"]
    return {"F": {k[2:]: v["est"] for k, v in q1.items() if k.startswith("F_")},
            "dF_vs_blind": c["dF"], "net": c["net"], "harm": c["n_harm"], "help": c["n_help"],
            "step0_dF": q1["by_step_vs_blind"]["step0"]["dF"]["est"], "later_dF": q1["by_step_vs_blind"]["later"]["dF"]["est"]}


def crosscheck(rep: dict, off: dict | None) -> dict | None:
    if not off:
        return None
    c = off["yn_aligned_vs_yn_blind"]["rule2_verification"]
    mine = rep["q1_recompute"]["contrasts"]["aligned_vs_blind"]
    return {"official_F_aligned": c["fail_rate_a"]["estimate"], "mine_F_aligned": mine["F_a"]["est"],
            "official_F_blind": c["fail_rate_b"]["estimate"], "mine_F_blind": mine["F_b"]["est"],
            "official_net": c["net_states"], "mine_net": mine["net"],
            "official_ci": c["fail_rate_difference"]["ci"], "mine_ci": mine["dF"]["ci"],
            "official_p": c["fail_rate_difference"]["p"], "mine_p": mine["dF"].get("p"),
            "match": abs(c["fail_rate_a"]["estimate"] - mine["F_a"]["est"]) < 1e-12
            and abs(c["fail_rate_b"]["estimate"] - mine["F_b"]["est"]) < 1e-12 and c["net_states"] == mine["net"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--official", type=Path)
    ap.add_argument("--official-secondary", type=Path, help="an analysis metrics.json holding the secondary models")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    states = load_states(args.transitions)
    ev = [s for s in states if s["step"] % 2 == 0]
    meta = {"n_eligible": len(states), "n_even": len(ev), "n_games": len({s["game"] for s in states}),
            "n_candidates": sum(len(s["cands"]) for s in states),
            "task_types_even": dict(Counter(s["ttype"] for s in ev)),
            "games_by_task_type": {k: len({s["game"] for s in ev if s["ttype"] == k}) for k in sorted({s["ttype"] for s in ev})},
            "failing_proposals_even": sum(not ok(s["prop"]) for s in ev)}
    off = {}
    for p in (args.official, args.official_secondary):
        if p and p.exists():
            off.update(json.load(open(p))["models"])
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    meta["provenance"] = {"transitions": {f.name: sha(f) for f in sorted(args.transitions.glob("s*.jsonl"))},
                          "scores": {m: sha(args.scores / m / "scoring.jsonl") for m in MODELS if (args.scores / m / "scoring.jsonl").exists()}}
    result = {"meta": meta, "models": {}, "summary": {}}
    for m in MODELS:
        f = args.scores / m / "scoring.jsonl"
        if not f.exists():
            result["models"][m] = "pending"
            continue
        L = load_scores(f)
        sts, comp = complete(states, L)
        rep = analyse(sts, L, m, full=True)
        rep["completeness"] = comp
        rep["official_crosscheck"] = crosscheck(rep, off.get(m))
        result["models"][m] = rep
        result["summary"][m] = summary_line(rep)
        print(m, json.dumps(result["summary"][m]))
        for k, v in rep["q3_variants"].get("gate_under_variants", {}).items():
            print("   ", k, v["category"], round(v["dF"]["est"], 4), [round(x, 4) for x in v["dF"]["ci"]], v["net"], v["rejected_both_net"],
                  round(v["aligned_minus_blind_swap_states"], 4), round(v["aligned_minus_swapped"], 4))
    args.out.write_text(json.dumps(result, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)) + "\n")


if __name__ == "__main__":
    main()
