"""Confirmatory RQ3 error plans: capped false successes and yoked false failures (decided 2026-11-17)."""
from __future__ import annotations

import pytest

from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.env.server import AlfworldClient
from wf.rq3 import FalseFailurePlan, FalseSuccessPlan, play_episode, run_false_failure, run_false_success

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
NOOP = "Nothing happens."
TWIN = "You pick up the mug 1 from the countertop 1."
SHARED = ("t", "proposal", "proposal_valid", "proposal_format_error", "prediction_shown", "prediction_type",
          "action_under_true_prediction", "executed", "real_obs", "decision_critical", "followed")


@pytest.fixture
def mock_env(monkeypatch):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    client = AlfworldClient()
    client.create()
    return client


def chance(plan: FalseSuccessPlan, t: int, proposal: str = "take mug 1 from countertop 1") -> bool:
    return plan.false_text(t, proposal, False, False, lambda: TWIN) is not None


# --- false successes ---------------------------------------------------------------------------


def test_false_success_injects_at_the_chosen_rate():
    plan = FalseSuccessPlan("valid_unseen/3", 7, 0.75, cap=10**9)
    hits = sum(chance(plan, t) for t in range(4000))
    assert plan.chances == 4000
    assert 0.72 < hits / 4000 < 0.78


def test_false_success_stops_at_the_cap_and_stops_looking():
    plan = FalseSuccessPlan("valid_unseen/3", 7, 1.0, cap=3)
    assert [chance(plan, t) for t in range(3)] == [True, True, True]
    looked = []
    assert plan.false_text(3, "take mug 1 from countertop 1", False, False, lambda: looked.append(1) or TWIN) is None
    assert not looked and plan.chances == 3 and len(plan.injections) == 3


def test_only_a_failing_proposal_with_a_twin_is_a_chance():
    plan = FalseSuccessPlan("valid_unseen/3", 7, 1.0, cap=3)
    assert plan.false_text(0, "go to drawer 1", True, False, lambda: TWIN) is None  # it works
    assert plan.false_text(1, "take mug 1 from countertop 1", False, False, lambda: None) is None  # no twin
    assert plan.chances == 0 and plan.injections == []
    assert plan.false_text(2, "take mug 1 from countertop 1", False, False, lambda: TWIN) == TWIN
    assert plan.injections == [{"i": 0, "step": 2, "verb": "take", "proposal": "take mug 1 from countertop 1", "chance": 0}]


def test_a_schedule_repeats_for_its_seed_and_differs_between_seeds():
    def decisions(game: str, seed: int) -> list[bool]:
        plan = FalseSuccessPlan(game, seed, 0.75, cap=10**9)
        return [chance(plan, t) for t in range(40)]

    assert decisions("valid_unseen/3", 1) == decisions("valid_unseen/3", 1)
    assert len({tuple(decisions("valid_unseen/3", s)) for s in (1, 2, 3)}) == 3
    assert decisions("valid_unseen/3", 1) != decisions("valid_unseen/4", 1)


def test_draws_follow_the_chance_not_the_step():
    # Retention modes part ways, so the k-th chance comes at different steps; it gets the same draw.
    early, late = FalseSuccessPlan("g", 5, 0.5, 10**9), FalseSuccessPlan("g", 5, 0.5, 10**9)
    assert [chance(early, t) for t in range(20)] == [chance(late, 3 * t + 7) for t in range(20)]


# --- false failures ----------------------------------------------------------------------------


def place(plan: FalseFailurePlan, steps: list[tuple[str, bool]]) -> list[int]:
    return [t for t, (a, ok) in enumerate(steps) if plan.false_text(t, a, ok, False) is not None]


def test_false_failure_goes_at_the_first_working_same_verb_proposal_at_or_after_its_partner():
    plan = FalseFailurePlan([{"step": 2, "verb": "take"}, {"step": 3, "verb": "open"}])
    steps = [("take mug 1 from drawer 1", True),  # before the partner's step
             ("open drawer 1", True),
             ("go to drawer 1", True),  # other verb
             ("take mug 1 from drawer 2", False),  # same verb, but it would fail anyway
             ("take mug 1 from drawer 1", True),  # error 1
             ("open cabinet 1", True)]  # error 2: after error 1 and at or after step 3
    assert place(plan, steps) == [4, 5]
    assert [(j["fs_step"], j["fs_verb"], j["verb"], j["fallback"]) for j in plan.injections] == [
        (2, "take", "take", False), (3, "open", "open", False)]


def test_one_step_takes_one_error():
    plan = FalseFailurePlan([{"step": 0, "verb": "take"}, {"step": 0, "verb": "take"}])
    assert place(plan, [("take a 1 from b 1", True), ("take a 1 from b 1", True)]) == [0, 1]


def test_a_format_error_is_never_a_working_proposal():
    plan = FalseFailurePlan([{"step": 0, "verb": "look"}])
    assert plan.false_text(0, "look", True, True) is None
    assert plan.false_text(1, "look", True, False) == NOOP


def test_a_fallback_index_takes_any_verb_and_is_logged():
    plan = FalseFailurePlan([{"step": 1, "verb": "heat"}], any_verb=frozenset({0}))
    assert place(plan, [("go to drawer 1", True), ("take mug 1 from drawer 1", False), ("go to fridge 1", True)]) == [2]
    assert plan.injections[0]["fallback"] is True and plan.injections[0]["verb"] == "go to"


def test_no_partner_errors_means_no_false_failures():
    plan = FalseFailurePlan([])
    assert place(plan, [("go to drawer 1", True)] * 10) == []


# --- whole episodes (mock environment) ----------------------------------------------------------


def test_a_false_success_shows_the_environments_success_text_at_chances_only(mock_env):
    ep = run_false_success(mock_env, AgentClient(), "train/0", "train", "retained", 3, 0.75, 3)
    hit = [st for st in ep["trace"] if st["injected"]]
    assert hit and len(hit) == ep["delivered_errors"] <= 3
    for st in hit:
        assert not st["proposal_valid"] and st["prediction_shown"] != NOOP
        assert st["prediction_type"] == "corrupted_favorable"
    plan = FalseSuccessPlan("train/0", 3, 0.75, 3)
    drawn = [k for k in range(ep["chances"]) if plan.draw(k)]
    assert [j["chance"] for j in ep["injections"]] == drawn[:3]
    assert [j["step"] for j in ep["injections"]] == [st["t"] for st in hit]


def test_perfect_foresight_path_matches_the_pilot_episode(mock_env):
    for retention in ("discarded", "retained"):
        mine = play_episode(mock_env, AgentClient(), "train/5", "train", retention)
        pilot = run_episode(mock_env, AgentClient(), "train/5", "train", "x", ForesightConfig("oracle", retention=retention))
        assert [{k: st[k] for k in SHARED} for st in mine["trace"]] == [{k: st[k] for k in SHARED} for st in pilot["trace"]]
        assert (mine["steps"], mine["success"]) == (pilot["steps"], pilot["success"])


def partner(*targets: tuple[int, str]) -> dict:
    return {"injections": [{"step": s, "verb": v} for s, v in targets], "schedule_seed": 1, "cap": 3}


def test_zero_partner_errors_give_the_perfect_foresight_episode(mock_env):
    ff = run_false_failure(mock_env, AgentClient(), "train/5", "train", "retained", partner())
    oracle = play_episode(mock_env, AgentClient(), "train/5", "train", "retained")
    assert ff["trace"] == oracle["trace"]
    assert (ff["planned_errors"], ff["delivered_errors"], ff["shortfall"], ff["restarts"]) == (0, 0, 0, [])


def test_a_missing_verb_falls_back_to_the_first_working_proposal_on_the_same_path(mock_env):
    oracle = play_episode(mock_env, AgentClient(), "train/5", "train", "discarded")
    first = next(st["t"] for st in oracle["trace"] if st["proposal_valid"] and not st["proposal_format_error"])
    ff = run_false_failure(mock_env, AgentClient(), "train/5", "train", "discarded", partner((0, "examine")))
    assert ff["restarts"] == [first]
    assert ff["trace"][:first] == oracle["trace"][:first]  # replayed, not re-asked
    assert ff["injections"][0]["step"] == first and ff["injections"][0]["fallback"] is True
    assert ff["fallbacks"] == 1 and ff["shortfall"] == 0
    assert ff["trace"][first]["prediction_shown"] == NOOP


def test_a_later_fallback_keeps_the_earlier_error(mock_env):
    oracle = play_episode(mock_env, AgentClient(), "train/5", "train", "discarded")
    working = [st for st in oracle["trace"] if st["proposal_valid"] and not st["proposal_format_error"]]
    first = working[0]
    ff = run_false_failure(mock_env, AgentClient(), "train/5", "train", "discarded",
                           partner((first["t"], first["proposal_verb"]), (first["t"], "examine")))
    assert [j["fallback"] for j in ff["injections"]] == [False, True]
    assert [st["t"] for st in ff["trace"] if st["injected"]] == [j["step"] for j in ff["injections"]]
    assert ff["injections"][0]["step"] == first["t"] < ff["injections"][1]["step"]


def test_an_error_with_nowhere_to_go_is_a_logged_shortfall(mock_env):
    ff = run_false_failure(mock_env, AgentClient(), "train/5", "train", "discarded", partner((40, "take")))
    assert (ff["planned_errors"], ff["delivered_errors"], ff["shortfall"], ff["restarts"]) == (1, 0, 1, [])
