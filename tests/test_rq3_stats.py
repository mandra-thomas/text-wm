"""Confirmatory RQ3 analysis on planted episodes with known answers ("RQ3 primary analysis set", 2026-11-17)."""
from __future__ import annotations

import pytest

from wf.analysis.rq3_stats import (
    cell_summary,
    complete_units,
    cost_per_error,
    delivered_covariate,
    interaction,
    primary_units,
    reading,
    rq3_summary,
    units,
)

FAIL = None  # an episode that never succeeds ranks S = -31


def ep(game: str, seed: int, arm: str, ret: str, steps: int | None, d: int = 1, planned: int | None = None,
       fallbacks: int = 0) -> dict:
    return {"game_id": game, "schedule_seed": seed, "condition": "corrupted", "arm": arm, "retention": ret,
            "steps": 30 if steps is None else steps, "success": steps is not None, "delivered_errors": d,
            "planned_errors": d if planned is None else planned, "shortfall": (d if planned is None else planned) - d,
            "fallbacks": fallbacks, "restarts": [], "trace": []}


def unit(game: str, seed: int, S: dict[tuple[str, str], int | None], d: tuple[int, int] = (1, 1)) -> list[dict]:
    """S maps (arm, retention) to steps to success; d is the FS errors (discarded, retained)."""
    return [ep(game, seed, arm, ret, S[(arm, ret)], d=d[ret == "retained"]) for arm, ret in S]


def cells(fs_d: int | None, ff_d: int | None, fs_r: int | None, ff_r: int | None) -> dict:
    return {("favorable", "discarded"): fs_d, ("unfavorable", "discarded"): ff_d,
            ("favorable", "retained"): fs_r, ("unfavorable", "retained"): ff_r}


# Game A, schedule 1: H_d = -10 - -14 = 4, H_r = -31 - -12 = -19, I = -23
# Game A, schedule 2: H_d = 0, H_r = -10 - -9 = -1, I = -1          -> game A: I = -12
# Game B, schedule 1: H_d = -20 - -10 = -10, H_r = -5 - -10 = 5, I = 15
# Game B, schedule 2: no error in the discarded FS episode, so outside the primary set; I = 52
PLANTED = (unit("A", 1, cells(10, 14, FAIL, 12)) + unit("A", 2, cells(8, 8, 10, 9))
           + unit("B", 1, cells(20, 10, 5, 10)) + unit("B", 2, cells(FAIL, 5, 5, FAIL), d=(0, 2)))


def test_primary_set_needs_an_error_in_both_retention_modes():
    eps = (unit("g", 1, cells(5, 5, 5, 5), d=(1, 1)) + unit("g", 2, cells(5, 5, 5, 5), d=(0, 2))
           + unit("g", 3, cells(5, 5, 5, 5), d=(2, 0)) + unit("g", 4, cells(5, 5, 5, 5), d=(0, 0))
           + unit("h", 1, cells(5, 5, 5, 5), d=(3, 3))[:3])  # incomplete
    by_unit = units(eps)
    assert primary_units(by_unit) == [("g", 1)]
    assert complete_units(by_unit) == [("g", 1), ("g", 2), ("g", 3), ("g", 4)]


def test_interaction_on_a_planted_example():
    by_unit = units(PLANTED)
    prim = interaction(by_unit, primary_units(by_unit))
    assert prim["I"]["estimate"] == pytest.approx(1.5)  # mean of games A (-12) and B (15), not of units
    assert prim["H_discarded"]["estimate"] == pytest.approx(-4.0)
    assert prim["H_retained"]["estimate"] == pytest.approx(-2.5)
    assert (prim["n_units"], prim["n_games"]) == (3, 2)
    assert prim["games_by_schedules_in_set"] == {1: 1, 2: 1}
    assert -12 <= prim["I"]["ci_low"] <= prim["I"]["ci_high"] <= 15
    every = interaction(by_unit, complete_units(by_unit))
    assert every["I"]["estimate"] == pytest.approx((-12 + (15 + 52) / 2) / 2)


def test_a_games_schedules_resample_together():
    # One game: every resample is that game, so the interval collapses on its mean.
    eps = unit("A", 1, cells(10, 14, FAIL, 12)) + unit("A", 2, cells(8, 8, 10, 9))
    by_unit = units(eps)
    I = interaction(by_unit, primary_units(by_unit), seed=3)["I"]
    assert I["estimate"] == I["ci_low"] == I["ci_high"] == pytest.approx(-12)


def test_reading_states_where_the_interval_lies():
    assert "below zero" in reading({"estimate": -2.0, "ci_low": -3.0, "ci_high": -0.5})
    assert "includes zero" in reading({"estimate": -2.0, "ci_low": -3.0, "ci_high": 0.5})
    assert "opposite to H3" in reading({"estimate": 2.0, "ci_low": 0.5, "ci_high": 3.0})


def test_cells_report_shortfalls_and_fallbacks():
    eps = [ep("A", 1, "favorable", "discarded", 10, d=2, planned=3),
           ep("A", 1, "unfavorable", "discarded", 10, d=1, planned=2, fallbacks=1),
           ep("B", 1, "unfavorable", "discarded", FAIL, d=2, planned=2, fallbacks=2)]
    out = cell_summary(eps)["unfavorable/discarded"]
    assert (out["shortfall_episodes"], out["shortfall_errors"], out["fallbacks"], out["fallback_episodes"]) == (1, 1, 3, 2)
    assert out["delivered_histogram"] == {1: 1, 2: 1} and out["success_pct"] == 50.0
    assert cell_summary(eps)["favorable/discarded"]["shortfall_episodes"] == 1  # below the cap


def test_cost_per_error_counts_steps_lost_against_perfect_foresight_in_the_same_mode():
    eps = unit("A", 1, cells(12, 10, 20, 10), d=(2, 2)) + unit("B", 1, cells(14, 8, 8, 8), d=(1, 1))
    baselines = [{"game_id": g, "condition": "oracle", "retention": r, "steps": s, "success": True}
                 for g, r, s in (("A", "discarded", 8), ("B", "discarded", 8), ("A", "retained", 10), ("B", "retained", 8))]
    by_unit = units(eps)
    cost = cost_per_error(by_unit, baselines, primary_units(by_unit))
    # favorable/discarded: losses 4 (A, 2 errors) and 6 (B, 1 error) -> 10 / 3
    assert cost["favorable/discarded"]["estimate"] == pytest.approx(10 / 3)
    # favorable/retained: losses 10 (A, 2 errors) and 0 (B, 1 error) -> 10 / 3
    assert cost["favorable/retained"]["estimate"] == pytest.approx(10 / 3)
    assert cost["unfavorable/discarded"]["estimate"] == pytest.approx(2 / 3)


def test_delivered_errors_by_retention_mode():
    eps = unit("A", 1, cells(5, 5, 5, 5), d=(1, 3)) + unit("B", 1, cells(5, 5, 5, 5), d=(2, 2))
    by_unit = units(eps)
    cov = delivered_covariate(by_unit, primary_units(by_unit))
    assert cov["mean_fs_delivered"]["discarded"]["estimate"] == pytest.approx(1.5)
    assert cov["retained_minus_discarded"]["estimate"] == pytest.approx(1.0)


def test_summary_keeps_baselines_out_of_the_units():
    base = [{"game_id": g, "condition": c, "arm": None, "retention": r, "schedule_seed": None, "steps": 9,
             "success": True, "trace": []}
            for g in ("A", "B") for c, r in (("none", "discarded"), ("oracle", "discarded"), ("oracle", "retained"),
                                             ("validity_only", "discarded"))]
    out = rq3_summary(PLANTED + base)
    assert out["primary"]["I"]["estimate"] == pytest.approx(1.5)
    assert out["primary_units"] == [["A", 1], ["A", 2], ["B", 1]]
    assert out["baselines"]["per_condition"]["oracle/retained"]["episodes"] == 2


def test_determinism_check_flags_an_unerrored_episode_that_differs_from_its_baseline():
    from wf.analysis.rq3_stats import determinism_check
    tr = lambda *acts: [{"executed": a} for a in acts]
    eps = [
        {"game_id": "g", "condition": "oracle", "arm": None, "retention": "discarded", "success": True, "trace": tr("a", "b")},
        {"game_id": "g", "condition": "corrupted", "arm": "favorable", "retention": "discarded", "schedule_seed": 1,
         "delivered_errors": 0, "success": True, "trace": tr("a", "b")},
        {"game_id": "g", "condition": "corrupted", "arm": "favorable", "retention": "discarded", "schedule_seed": 2,
         "delivered_errors": 0, "success": True, "trace": tr("a", "c")},
        {"game_id": "g", "condition": "corrupted", "arm": "favorable", "retention": "discarded", "schedule_seed": 3,
         "delivered_errors": 2, "success": False, "trace": tr("x")},
    ]
    out = determinism_check(eps)
    assert out["checked"] == 2 and out["identical"] == 1 and out["differing"] == [["g", "discarded", 2]]
