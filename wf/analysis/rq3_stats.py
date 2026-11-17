"""Confirmatory RQ3 analysis (notes/deviations.md, "RQ3 primary analysis set", 2026-11-17).

A unit is one (game, schedule): its FS and FF episodes under both retention modes. Per unit,
H(ret) = S_FS(ret) - S_FF(ret) and I = H(retained) - H(discarded). Values are averaged over a
game's units first, then over games, and CIs resample games, so a game's schedules are one cluster.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

from wf.analysis.stats import cluster_bootstrap, rank_gain_ci, rank_outcome, success_gain_ci
from wf.constants import STEP_CAP

ARMS = ("favorable", "unfavorable")
RETENTIONS = ("discarded", "retained")
Unit = tuple[str, int]


def episode_S(ep: dict, step_cap: int = STEP_CAP) -> float:
    return rank_outcome([ep], step_cap)["mean_S"]


def units(episodes: list[dict]) -> dict[Unit, dict[tuple[str, str], dict]]:
    out: dict[Unit, dict[tuple[str, str], dict]] = defaultdict(dict)
    for ep in episodes:
        if ep.get("condition") == "corrupted":
            out[(ep["game_id"], ep["schedule_seed"])][(ep["arm"], ep["retention"])] = ep
    return dict(out)


def complete_units(by_unit: dict[Unit, dict]) -> list[Unit]:
    return sorted(u for u, cells in by_unit.items() if all((a, r) in cells for a in ARMS for r in RETENTIONS))


def primary_units(by_unit: dict[Unit, dict]) -> list[Unit]:
    """Units whose FS episode received at least one error in both retention modes.

    With none delivered an episode is perfect foresight, so membership depends only on behaviour
    before any injection: a pre-treatment selection.
    """
    return [u for u in complete_units(by_unit)
            if all(by_unit[u][("favorable", r)]["delivered_errors"] >= 1 for r in RETENTIONS)]


def _per_game(keys: list[Unit], value: Callable[[Unit], float]) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for u in keys:
        vals[u[0]].append(value(u))
    return {g: sum(v) / len(v) for g, v in vals.items()}


def _mean_ci(per_game: dict[str, float], seed: int) -> dict[str, float]:
    games = sorted(per_game)
    return cluster_bootstrap(lambda xs: sum(xs) / len(xs) if xs else float("nan"),
                             [per_game[g] for g in games], games, seed=seed)


def interaction(by_unit: dict[Unit, dict], keys: list[Unit], step_cap: int = STEP_CAP, seed: int = 0) -> dict[str, Any]:
    def H(u: Unit, ret: str) -> float:
        return episode_S(by_unit[u][("favorable", ret)], step_cap) - episode_S(by_unit[u][("unfavorable", ret)], step_cap)

    per_game_I = _per_game(keys, lambda u: H(u, "retained") - H(u, "discarded"))
    return {
        "I": _mean_ci(per_game_I, seed),
        "H_discarded": _mean_ci(_per_game(keys, lambda u: H(u, "discarded")), seed),
        "H_retained": _mean_ci(_per_game(keys, lambda u: H(u, "retained")), seed),
        "n_units": len(keys),
        "n_games": len(per_game_I),
        "games_by_schedules_in_set": dict(sorted(Counter(Counter(g for g, _ in keys).values()).items())),
        "per_game_I": per_game_I,
    }


def cell_summary(episodes: list[dict], step_cap: int = STEP_CAP) -> dict[str, dict]:
    """Per arm and retention: outcome, delivered errors, and how often the plan fell short.

    For FS episodes, planned is the cap and falling short means the chances ran out; for FF
    episodes, planned is the partner's delivered count and falling short means the yoke failed.
    """
    out = {}
    for arm in ARMS:
        for ret in RETENTIONS:
            eps = [e for e in episodes if e.get("condition") == "corrupted" and e["arm"] == arm and e["retention"] == ret]
            steps = [st for e in eps for st in e["trace"]]
            out[f"{arm}/{ret}"] = {
                "episodes": len(eps),
                "success_pct": 100.0 * sum(bool(e["success"]) for e in eps) / len(eps) if eps else float("nan"),
                "mean_S": rank_outcome(eps, step_cap)["mean_S"],
                "mean_delivered": sum(e["delivered_errors"] for e in eps) / len(eps) if eps else float("nan"),
                "delivered_histogram": dict(sorted(Counter(e["delivered_errors"] for e in eps).items())),
                "planned_errors": sum(e["planned_errors"] for e in eps),
                "delivered_errors": sum(e["delivered_errors"] for e in eps),
                "shortfall_episodes": sum(e["shortfall"] > 0 for e in eps),
                "shortfall_errors": sum(e["shortfall"] for e in eps),
                "fallbacks": sum(e["fallbacks"] for e in eps),
                "fallback_episodes": sum(e["fallbacks"] > 0 for e in eps),
                "restarted_episodes": sum(bool(e["restarts"]) for e in eps),
                "format_error_rate": (sum(bool(st.get("proposal_format_error") or st.get("executed_format_error"))
                                          for st in steps) / len(steps)) if steps else float("nan"),
            }
    return out


def cost_per_error(by_unit: dict[Unit, dict], baselines: list[dict], keys: list[Unit],
                   step_cap: int = STEP_CAP, seed: int = 0) -> dict[str, dict]:
    """Steps lost against perfect foresight in the same retention mode, per delivered error.

    Over units whose episode received at least one error: mean loss over mean errors, per game
    first, game-bootstrapped.
    """
    out = {}
    for ret in RETENTIONS:
        oracle = rank_outcome([e for e in baselines if e["condition"] == "oracle" and e["retention"] == ret],
                              step_cap)["per_game_S"]
        for arm in ARMS:
            pairs: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for u in keys:
                ep = by_unit[u][(arm, ret)]
                if ep["delivered_errors"] >= 1 and u[0] in oracle:
                    pairs[u[0]].append((oracle[u[0]] - episode_S(ep, step_cap), ep["delivered_errors"]))
            games = sorted(pairs)
            items = [(sum(x for x, _ in pairs[g]) / len(pairs[g]), sum(n for _, n in pairs[g]) / len(pairs[g]))
                     for g in games]
            ratio = cluster_bootstrap(
                lambda xs: sum(a for a, _ in xs) / sum(b for _, b in xs) if xs else float("nan"),
                items, games, seed=seed)
            out[f"{arm}/{ret}"] = {**ratio, "n_games": len(games)}
    return out


def delivered_covariate(by_unit: dict[Unit, dict], keys: list[Unit], seed: int = 0) -> dict[str, Any]:
    """Errors the FS arm received by retention mode; kept FS predictions invite retries, so more chances."""
    def d(u: Unit, ret: str) -> int:
        return by_unit[u][("favorable", ret)]["delivered_errors"]

    return {
        "mean_fs_delivered": {r: _mean_ci(_per_game(keys, lambda u, r=r: d(u, r)), seed) for r in RETENTIONS},
        "retained_minus_discarded": _mean_ci(_per_game(keys, lambda u: d(u, "retained") - d(u, "discarded")), seed),
    }


def baseline_summary(baselines: list[dict], step_cap: int = STEP_CAP, seed: int = 0) -> dict[str, Any]:
    def cell(cond: str, ret: str) -> list[dict]:
        return [e for e in baselines if e["condition"] == cond and e["retention"] == ret]

    per_condition = {
        f"{cond}/{ret}": {"episodes": len(cell(cond, ret)),
                          "success_pct": 100.0 * sum(bool(e["success"]) for e in cell(cond, ret)) / max(len(cell(cond, ret)), 1),
                          "mean_S": rank_outcome(cell(cond, ret), step_cap)["mean_S"]}
        for cond, ret in (("none", "discarded"), ("oracle", "discarded"), ("oracle", "retained"), ("validity_only", "discarded"))
    }
    none, oracle_d = cell("none", "discarded"), cell("oracle", "discarded")
    return {
        "per_condition": per_condition,
        "oracle_steps_gain": rank_gain_ci(oracle_d, none, seed=seed),
        "oracle_success_gain_pp": success_gain_ci(oracle_d, none, seed=seed),
        "retained_minus_discarded_oracle_steps": rank_gain_ci(cell("oracle", "retained"), oracle_d, seed=seed),
    }


def reading(I: dict[str, float]) -> str:
    est, lo, hi = I["estimate"], I["ci_low"], I["ci_high"]
    if est != est or lo != lo or hi != hi:
        return "I could not be estimated"
    where = ("lies below zero, the sign H3 predicts" if hi < 0 else "lies above zero, opposite to H3" if lo > 0
             else "includes zero")
    return f"I = {est:.2f} steps (95% game-bootstrap CI {lo:.2f} to {hi:.2f}): the CI {where}"


def rq3_summary(episodes: list[dict], step_cap: int = STEP_CAP, seed: int = 0) -> dict[str, Any]:
    by_unit = units(episodes)
    everything = complete_units(by_unit)
    primary = primary_units(by_unit)
    baselines = [e for e in episodes if e.get("condition") != "corrupted"]
    cells = cell_summary(episodes, step_cap)
    prim = interaction(by_unit, primary, step_cap, seed)
    return {
        "primary": {k: v for k, v in prim.items() if k != "per_game_I"},
        "all_units": {k: v for k, v in interaction(by_unit, everything, step_cap, seed).items() if k != "per_game_I"},
        "reading": reading(prim["I"]),
        "cells": cells,
        "shortfalls": {c: v["shortfall_episodes"] for c, v in cells.items()},
        "fallbacks": {c: v["fallbacks"] for c, v in cells.items() if c.startswith("unfavorable")},
        "cost_per_error_primary": cost_per_error(by_unit, baselines, primary, step_cap, seed),
        "cost_per_error_all_units": cost_per_error(by_unit, baselines, everything, step_cap, seed),
        "delivered_covariate_primary": delivered_covariate(by_unit, primary, seed),
        "baselines": baseline_summary(baselines, step_cap, seed),
        "primary_units": [list(u) for u in primary],
    }
