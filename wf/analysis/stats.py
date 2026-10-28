"""Game-clustered stats (PILOT_PROTOCOL §4.9)."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable

from wf.constants import BOOTSTRAP_RESAMPLES, PERMUTATIONS, STEP_CAP
from wf.util import sigmoid


def delta_p(
    scores: list[dict],
    transitions: list[dict] | None = None,
    primary_condition: str = "aligned",
    control: str = "swapped",
) -> dict[str, Any]:
    """Mean sigmoid(L_aligned) - sigmoid(L_swapped) in percentage points.

    Restricted to context_determined & true failure & history_goal_agnostic when
    transition metadata is provided.
    """
    tmeta = {t["id"]: t for t in (transitions or [])}
    # group by transition_id
    by_tid: dict[str, dict[str, dict]] = defaultdict(dict)
    for s in scores:
        by_tid[s["transition_id"]][s["condition"]] = s

    diffs: list[float] = []
    game_vals: dict[str, list[float]] = defaultdict(list)
    for tid, conds in by_tid.items():
        if primary_condition not in conds or control not in conds:
            continue
        meta = tmeta.get(tid)
        if meta is not None:
            if not meta.get("context_determined"):
                continue
            if (meta.get("true_obs") or "").strip() != "Nothing happens.":
                continue
            if not meta.get("history_goal_agnostic", True):
                continue
        d = (sigmoid(conds[primary_condition]["L"]) - sigmoid(conds[control]["L"])) * 100.0
        diffs.append(d)
        gid = (meta or {}).get("game_id") or tid.split("_")[0]
        # Prefer game_id from scores side-channel if present
        gid = conds[primary_condition].get("game_id", gid)
        if meta:
            gid = meta["game_id"]
        game_vals[gid].append(d)

    mean = sum(diffs) / len(diffs) if diffs else float("nan")
    return {
        "delta_p_pp": mean,
        "n_items": len(diffs),
        "n_games": len(game_vals),
        "per_game_means": {g: sum(v) / len(v) for g, v in game_vals.items()},
        "item_diffs": diffs,
        "game_ids": list(game_vals.keys()),
    }


def asymmetry_A(scores: list[dict], transitions: list[dict]) -> dict[str, Any]:
    """A = P(sign(L)>0 | true failure) - P(sign(L)<0 | true success) on context_determined."""
    tmeta = {t["id"]: t for t in transitions}
    # use aligned condition
    fail_pos = fail_n = succ_neg = succ_n = 0
    for s in scores:
        if s.get("condition") != "aligned":
            continue
        meta = tmeta.get(s["transition_id"])
        if not meta or not meta.get("context_determined"):
            continue
        true_fail = (meta.get("true_obs") or "").strip() == "Nothing happens."
        L = s["L"]
        if true_fail:
            fail_n += 1
            if L > 0:
                fail_pos += 1
        else:
            succ_n += 1
            if L < 0:
                succ_neg += 1
    fs_rate = fail_pos / fail_n if fail_n else float("nan")
    ff_rate = succ_neg / succ_n if succ_n else float("nan")
    A = fs_rate - ff_rate if fail_n and succ_n else float("nan")
    return {
        "A": A,
        "false_success_rate": fs_rate,
        "false_failure_rate": ff_rate,
        "n_fail": fail_n,
        "n_succ": succ_n,
    }


def cluster_bootstrap(
    stat_fn: Callable[[list[Any]], float],
    items: list[Any],
    cluster_ids: list[str],
    B: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile CI resampling games."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for it, cid in zip(items, cluster_ids):
        by[cid].append(it)
    games = list(by.keys())
    if not games:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    point = stat_fn(items)
    boots = []
    for _ in range(B):
        sample = []
        for g in rng.choices(games, k=len(games)):
            sample.extend(by[g])
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[min(B - 1, int(0.975 * B))]
    return {"estimate": point, "ci_low": lo, "ci_high": hi}


def pair_permutation(diffs: list[float], n: int = PERMUTATIONS, seed: int = 0) -> float:
    """Sign-flip within pairs; two-sided p-value vs observed mean."""
    if not diffs:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(sum(diffs) / len(diffs))
    count = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(s) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n + 1)


def icc(items: list[float], cluster_ids: list[str]) -> float:
    """One-way ICC of item values clustered by game."""
    by = defaultdict(list)
    for v, c in zip(items, cluster_ids):
        by[c].append(v)
    groups = [v for v in by.values() if v]
    if len(groups) < 2:
        return float("nan")
    n = sum(len(g) for g in groups)
    k = len(groups)
    grand = sum(sum(g) for g in groups) / n
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ssw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    msb = ssb / (k - 1) if k > 1 else float("nan")
    dfw = n - k
    msw = ssw / dfw if dfw > 0 else float("nan")
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    if msb != msb or msw != msw or n0 == 0:
        return float("nan")
    return (msb - msw) / (msb + (n0 - 1) * msw)


def rank_outcome(episodes: list[dict], step_cap: int = STEP_CAP) -> dict[str, Any]:
    """Per episode S = -min(steps, STEP_CAP); failures -> -(STEP_CAP+1)."""
    per_game: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        if ep.get("success"):
            S = -min(int(ep.get("steps", step_cap)), step_cap)
        else:
            S = -(step_cap + 1)
        per_game[ep["game_id"]].append(S)
    game_means = {g: sum(v) / len(v) for g, v in per_game.items()}
    overall = sum(game_means.values()) / len(game_means) if game_means else float("nan")
    return {"mean_S": overall, "per_game_S": game_means, "n_games": len(game_means)}


def H_cell(episodes: list[dict], retention: str) -> float:
    """Mean over games of S_favorable - S_unfavorable for a retention mode."""
    fav = [e for e in episodes if e.get("retention") == retention and e.get("arm") == "favorable"]
    unf = [e for e in episodes if e.get("retention") == retention and e.get("arm") == "unfavorable"]
    sf = rank_outcome(fav)["per_game_S"]
    su = rank_outcome(unf)["per_game_S"]
    games = set(sf) & set(su)
    if not games:
        return float("nan")
    return sum(sf[g] - su[g] for g in games) / len(games)


def interaction_I(episodes: list[dict]) -> float:
    """I = H_retained - H_discarded."""
    return H_cell(episodes, "retained") - H_cell(episodes, "discarded")


def implied_se_delta_p(icc_val: float, n_items: int, n_games: int) -> float:
    """Rough SE for planning sample size (pilot A)."""
    if n_items <= 1 or n_games <= 0 or icc_val != icc_val:
        return float("nan")
    # design effect
    m = n_items / n_games
    deff = 1 + (m - 1) * max(icc_val, 0.0)
    # assume item SD ~ 20 pp as a planning default if unknown
    sd = 20.0
    return sd * math.sqrt(deff / n_items)
