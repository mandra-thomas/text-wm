"""Game-clustered stats (PILOT_PROTOCOL §4.9)."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable

from wf.constants import BOOTSTRAP_RESAMPLES, PERMUTATIONS, STEP_CAP
from wf.util import sigmoid


def paired_conditions(
    scores: list[dict],
    transitions: list[dict] | None = None,
    primary_condition: str = "aligned",
    control: str = "swapped",
    require_agnostic: bool = True,
) -> list[tuple[str, float, float]]:
    """(game_id, L_primary, L_control) per item.

    With transitions given, only those items count, restricted to context_determined & true
    failure (& history_goal_agnostic unless require_agnostic=False).
    """
    tmeta = None if transitions is None else {t["id"]: t for t in transitions}
    by_tid: dict[str, dict[str, dict]] = defaultdict(dict)
    for s in scores:
        by_tid[s["transition_id"]][s["condition"]] = s

    out: list[tuple[str, float, float]] = []
    for tid, conds in by_tid.items():
        if primary_condition not in conds or control not in conds:
            continue
        gid = conds[primary_condition].get("game_id", tid)
        if tmeta is not None:
            meta = tmeta.get(tid)
            if meta is None or not meta.get("context_determined"):
                continue
            if (meta.get("true_obs") or "").strip() != "Nothing happens.":
                continue
            if require_agnostic and not meta.get("history_goal_agnostic", True):
                continue
            gid = meta["game_id"]
        out.append((gid, conds[primary_condition]["L"], conds[control]["L"]))
    return out


def _paired_effect(pairs: list[tuple[str, float, float]], diff: Callable[[float, float], float], key: str) -> dict[str, Any]:
    diffs = [diff(a, c) for _, a, c in pairs]
    games = [g for g, _, _ in pairs]
    game_vals: dict[str, list[float]] = defaultdict(list)
    for g, d in zip(games, diffs):
        game_vals[g].append(d)
    return {
        key: sum(diffs) / len(diffs) if diffs else float("nan"),
        "n_items": len(diffs),
        "n_games": len(game_vals),
        "per_game_means": {g: sum(v) / len(v) for g, v in game_vals.items()},
        "item_diffs": diffs,
        "item_games": games,
    }


def delta_p(
    scores: list[dict],
    transitions: list[dict] | None = None,
    primary_condition: str = "aligned",
    control: str = "swapped",
    require_agnostic: bool = True,
) -> dict[str, Any]:
    """Mean sigmoid(L_aligned) - sigmoid(L_swapped) in percentage points (§4.9)."""
    pairs = paired_conditions(scores, transitions, primary_condition, control, require_agnostic)
    return _paired_effect(pairs, lambda a, c: (sigmoid(a) - sigmoid(c)) * 100.0, "delta_p_pp")


def delta_L(
    scores: list[dict],
    transitions: list[dict] | None = None,
    primary_condition: str = "aligned",
    control: str = "swapped",
    require_agnostic: bool = True,
) -> dict[str, Any]:
    """Prereg headline: mean L_aligned - L_swapped in nats."""
    pairs = paired_conditions(scores, transitions, primary_condition, control, require_agnostic)
    return _paired_effect(pairs, lambda a, c: a - c, "delta_L_nats")


def asymmetry_A(scores: list[dict], transitions: list[dict]) -> dict[str, Any]:
    """A = P(sign(L)>0 | true failure) - P(sign(L)<0 | true success) on context_determined."""
    labelled = asymmetry_items(scores, transitions)
    return _asymmetry([item for _, item in labelled])


def asymmetry_items(scores: list[dict], transitions: list[dict]) -> list[tuple[str, tuple[bool, float]]]:
    """(game_id, (true_failure, L)) for aligned-condition context_determined items."""
    tmeta = {t["id"]: t for t in transitions}
    out = []
    for s in scores:
        meta = tmeta.get(s["transition_id"])
        if s.get("condition") != "aligned" or not meta or not meta.get("context_determined"):
            continue
        out.append((meta["game_id"], ((meta.get("true_obs") or "").strip() == "Nothing happens.", s["L"])))
    return out


def _asymmetry(items: list[tuple[bool, float]]) -> dict[str, Any]:
    fail_n = sum(1 for failed, _ in items if failed)
    succ_n = len(items) - fail_n
    fail_pos = sum(1 for failed, L in items if failed and L > 0)
    succ_neg = sum(1 for failed, L in items if not failed and L < 0)
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


def asymmetry_ci(scores: list[dict], transitions: list[dict], seed: int = 0) -> dict[str, float]:
    labelled = asymmetry_items(scores, transitions)
    return cluster_bootstrap(
        lambda items: _asymmetry(items)["A"],
        [item for _, item in labelled],
        [game for game, _ in labelled],
        seed=seed,
    )


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
    boots = sorted(b for b in boots if b == b)  # a resample can lack a class entirely
    if not boots:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return {"estimate": point, "ci_low": lo, "ci_high": hi}


def pair_permutation(
    diffs: list[float],
    n: int = PERMUTATIONS,
    seed: int = 0,
    cluster_ids: list[str] | None = None,
) -> float:
    """Two-sided sign-flip p-value for the mean.

    With cluster_ids, all items of a game flip together, as the prereg requires so that the
    null respects clustering. Without them, each item flips alone.
    """
    if not diffs:
        return float("nan")
    rng = random.Random(seed)
    totals: dict[Any, float] = defaultdict(float)
    for i, d in enumerate(diffs):
        totals[cluster_ids[i] if cluster_ids else i] += d
    groups = list(totals.values())
    obs = abs(sum(groups)) / len(diffs)
    count = 0
    for _ in range(n):
        s = sum(g if rng.random() < 0.5 else -g for g in groups) / len(diffs)
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
    denominator = msb + (n0 - 1) * msw
    if msb != msb or msw != msw or denominator == 0:  # e.g. every item difference identical
        return float("nan")
    return (msb - msw) / denominator


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


def _game_means(episodes: list[dict], value: Callable[[dict], float]) -> dict[str, float]:
    per_game: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        per_game[ep["game_id"]].append(value(ep))
    return {g: sum(v) / len(v) for g, v in per_game.items()}


def success_gain_ci(better: list[dict], baseline: list[dict], seed: int = 0) -> dict[str, float]:
    """Mean success-rate difference in points over shared games, game-bootstrapped."""
    a = _game_means(better, lambda e: 100.0 * bool(e.get("success")))
    b = _game_means(baseline, lambda e: 100.0 * bool(e.get("success")))
    games = sorted(set(a) & set(b))
    return cluster_bootstrap(
        lambda xs: sum(xs) / len(xs) if xs else float("nan"),
        [a[g] - b[g] for g in games],
        games,
        seed=seed,
    )


def H_success_points(episodes: list[dict], retention: str) -> float:
    """H in success-rate points, the scale of H_FLOOR_PP."""
    cell = [e for e in episodes if e.get("retention") == retention]
    fav = _game_means([e for e in cell if e.get("arm") == "favorable"], lambda e: 100.0 * bool(e.get("success")))
    unf = _game_means([e for e in cell if e.get("arm") == "unfavorable"], lambda e: 100.0 * bool(e.get("success")))
    games = set(fav) & set(unf)
    return sum(fav[g] - unf[g] for g in games) / len(games) if games else float("nan")


def interaction_ci(episodes: list[dict], seed: int = 0) -> dict[str, float]:
    """Game-bootstrapped I (§4.9)."""
    per_cell: dict[tuple[str, str], dict[str, float]] = {}
    for arm in ("favorable", "unfavorable"):
        for ret in ("discarded", "retained"):
            eps = [e for e in episodes if e.get("arm") == arm and e.get("retention") == ret]
            per_cell[(arm, ret)] = rank_outcome(eps)["per_game_S"]
    games = sorted(set.intersection(*(set(c) for c in per_cell.values())))
    per_game = [
        (per_cell[("favorable", "retained")][g] - per_cell[("unfavorable", "retained")][g])
        - (per_cell[("favorable", "discarded")][g] - per_cell[("unfavorable", "discarded")][g])
        for g in games
    ]
    return cluster_bootstrap(lambda xs: sum(xs) / len(xs) if xs else float("nan"), per_game, games, seed=seed)


def corruption_diagnostics(episodes: list[dict], oracle_episodes: list[dict] | None = None) -> dict[str, Any]:
    """Mini-run report items (§6): follow rates, rank lost per error, retry rate."""
    steps = [tr for ep in episodes for tr in ep.get("trace", [])]

    def follow(kind: str) -> float | None:
        picked = [tr for tr in steps if tr.get("prediction_type") == kind]
        return sum(bool(tr.get("followed")) for tr in picked) / len(picked) if picked else None

    retried = rejected = 0
    for ep in episodes:
        trace = ep.get("trace", [])
        for i, tr in enumerate(trace):
            if tr.get("prediction_type") == "corrupted_unfavorable" and tr.get("executed") != tr.get("proposal"):
                rejected += 1
                retried += any(later.get("executed") == tr.get("proposal") for later in trace[i + 1:])

    lost: dict[str, float | None] = {}
    baseline = rank_outcome(oracle_episodes)["per_game_S"] if oracle_episodes else {}
    for arm in ("favorable", "unfavorable"):
        cell = [e for e in episodes if e.get("arm") == arm]
        per_game = rank_outcome(cell)["per_game_S"]
        games = set(per_game) & set(baseline)
        errors = sum(e.get("n_corrupted", 0) for e in cell)
        lost[arm] = (
            sum(baseline[g] - per_game[g] for g in games) / len(games) / (errors / len(cell))
            if games and errors
            else None
        )
    return {
        "follow_rate_corrupted_favorable": follow("corrupted_favorable"),
        "follow_rate_corrupted_unfavorable": follow("corrupted_unfavorable"),
        "follow_rate_correct": follow("oracle"),
        "rank_lost_per_error": lost,
        "retry_rate_after_unfavorable": retried / rejected if rejected else None,
    }


def implied_se_delta_p(icc_val: float, n_items: int, n_games: int, sd: float) -> float:
    """SE of a game-clustered mean with n_items spread over n_games (sample-size planning)."""
    if n_items <= 1 or n_games <= 0 or icc_val != icc_val or sd != sd:
        return float("nan")
    m = n_items / n_games
    deff = 1 + (m - 1) * max(icc_val, 0.0)
    return sd * math.sqrt(deff / n_items)
