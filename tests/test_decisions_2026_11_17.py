"""Decisions 4, 5, 6, 13 and 14 of 2026-11-17: what each changed, pinned so an edit cannot undo it."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import wf.runs as runs_mod
from wf import constants as C
from wf.agent.foresight import ForesightConfig
from wf.agent.react import run_episode
from wf.analysis import stats
from wf.scoring.logodds import LogoddsScorer, ScoreResult
from wf.util import read_jsonl, sigmoid

from test_audit_known_bugs import FixedAgent, isolated_foresight, real_client  # noqa: F401  (fixtures)
from test_probes import MOCK_KEYS, PlantedScorer

ROOT = Path(__file__).resolve().parents[1]


def _metrics(tmp_path):
    return json.loads((tmp_path / "pilotA_score" / "metrics.json").read_text().replace("NaN", "null"))


def _run(monkeypatch, tmp_path, scorer_cls, base=-2.0, effect=1.0, cfg=None, expect=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    if not (tmp_path / "pilotA_transitions" / "transitions.jsonl").exists():
        runs_mod.run_pilotA_transitions({"run_id": "pilotA_transitions", "n_games": 40})
    monkeypatch.setattr(scorer_cls, "base", base)
    monkeypatch.setattr(scorer_cls, "effect", effect)
    monkeypatch.setattr(runs_mod, "LogoddsScorer", scorer_cls)
    assert runs_mod.run_pilotA_score(cfg or {"run_id": "pilotA_score"}) == expect
    return _metrics(tmp_path)


class SaysSuccess(PlantedScorer):
    """Writes a success for every item, whatever L says."""

    def free_generate(self, messages, max_new_tokens=64):
        return "You pick up the mug 1 from the countertop 1."


class SaysFailure(PlantedScorer):
    def free_generate(self, messages, max_new_tokens=64):
        return "Nothing happens."


class FirstTokenPlanted(PlantedScorer):
    """Also plants the effect on the first token, at half the size."""

    def score_pair(self, transition_id, condition, messages, o_plus, o_minus):
        r = super().score_pair(transition_id, condition, messages, o_plus, o_minus)
        first = -1.0 + (self.effect / 2 if condition == "aligned" else 0.0)
        return ScoreResult(r.transition_id, r.model, r.condition, r.logp_plus, r.logp_minus, r.L, r.p_success,
                           1, 1, logp_first_plus=first, logp_first_minus=0.0)


# --- decision 5: RQ1 reads the model's own text --------------------------------------------------


def test_A_follows_the_written_prediction_not_the_sign_of_L(monkeypatch, tmp_path):
    # L is far below 0 on every item, yet the model writes "success" every time.
    A = _run(monkeypatch, tmp_path, SaysSuccess, base=-7.0, effect=0.0)["reports"]["qwen3-8b-fewshot"]["A"]
    assert A["false_success_rate"] == 1.0 and A["false_failure_rate"] == 0.0 and A["A"] == 1.0


def test_unparseable_text_is_left_out_of_A_and_counted():
    items = [{"id": "f", "game_id": "g", "context_determined": True, "true_obs": "Nothing happens."},
             {"id": "s", "game_id": "g", "context_determined": True, "true_obs": "You open it."}]
    gens = [{"transition_id": "f", "condition": "aligned", "generated_class": "unparseable"},
            {"transition_id": "s", "condition": "aligned", "generated_class": "noop"}]
    out = stats.asymmetry_A(gens, items)
    assert out["n_fail"] == 0 and out["n_succ"] == 1 and out["n_unparseable"] == 1


def test_first_token_effect_is_reported_beside_delta_L(monkeypatch, tmp_path):
    core = _run(monkeypatch, tmp_path, FirstTokenPlanted)["reports"]["qwen3-8b-fewshot"]["goal_agnostic"]
    assert core["delta_L_nats"] == pytest.approx(1.0, abs=1e-9)
    assert core["delta_L_first_nats"] == pytest.approx(0.5, abs=1e-9)
    assert core["delta_L_first_ci"][0] <= 0.5 <= core["delta_L_first_ci"][1]


def test_penalty_and_first_token_mass_follow_their_definitions():
    r = ScoreResult("t", "m", "aligned", -9.0, -1.0, -8.0, sigmoid(-8.0), 5, 3,
                    logp_first_plus=-0.5, logp_first_minus=-2.0, first_token_plus="You", first_token_minus="Nothing")
    assert r.L_first == pytest.approx(1.5)
    assert r.c_hat == pytest.approx(1.5 - (-8.0))  # logit(q_first) - L
    assert r.first_mass == pytest.approx(math.exp(-0.5) + math.exp(-2.0))
    d = r.to_dict()
    assert d["first_token_plus"] == "You" and d["first_token_minus"] == "Nothing"


def test_real_scorer_saves_first_token_terms(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    scorer = LogoddsScorer("m", model_tag="m")
    r = scorer.score_pair("t", "aligned", [{"role": "user", "content": "x"}], "You open the drawer 1.", "Nothing happens.")
    d = r.to_dict()
    assert d["L_first"] == pytest.approx(d["logp_first_plus"] - d["logp_first_minus"])
    assert (d["L_first"] > 0) == (d["L"] > 0)


def test_decision_flip_counts_items_whose_first_token_choice_changes():
    items = [{"id": i, "game_id": "g", "context_determined": True, "true_obs": "Nothing happens."} for i in "abc"]
    scores = [{"transition_id": "a", "condition": "aligned", "L_first": 1.0},
              {"transition_id": "a", "condition": "swapped", "L_first": -1.0},  # flips toward success
              {"transition_id": "b", "condition": "aligned", "L_first": -2.0},
              {"transition_id": "b", "condition": "swapped", "L_first": -3.0},  # moves, no flip
              {"transition_id": "c", "condition": "aligned", "L_first": -1.0},
              {"transition_id": "c", "condition": "swapped", "L_first": 2.0}]  # flips away
    out = stats.decision_flip_share(scores, items)
    assert out["n_items"] == 3 and out["share"] == pytest.approx(2 / 3)
    assert out["toward_success"] == 1 and out["away_from_success"] == 1


def test_gen_success_shift_is_aligned_minus_swapped():
    items = [{"id": i, "game_id": g, "context_determined": True, "true_obs": "Nothing happens."}
             for i, g in (("a", "g1"), ("b", "g2"))]
    gens = [{"transition_id": "a", "condition": "aligned", "generated_class": "state_change"},
            {"transition_id": "a", "condition": "swapped", "generated_class": "noop"},
            {"transition_id": "b", "condition": "aligned", "generated_class": "noop"},
            {"transition_id": "b", "condition": "swapped", "generated_class": "noop"}]
    assert stats.gen_success_shift(gens, items)["shift_pp"] == pytest.approx(50.0)


# --- decision 13: the stop rule on Pilot A's own failures ------------------------------------------


def test_stop_rule_fires_when_text_and_sign_disagree_on_failures(monkeypatch, tmp_path):
    metrics = _run(monkeypatch, tmp_path, SaysSuccess, base=-7.0, effect=0.0)
    assert any(s.startswith("qwen3-8b-fewshot") for s in metrics["scoring_stop"])
    assert not any(s.startswith("w2w") for s in metrics["scoring_stop"])  # the full-state control is not gated
    assert (tmp_path / "pilotA_score" / "STATUS.md").read_text().count("STOP:") == 1


def test_stop_rule_is_quiet_when_text_and_sign_agree(monkeypatch, tmp_path):
    metrics = _run(monkeypatch, tmp_path, SaysFailure, base=-7.0, effect=0.0)
    rep = metrics["reports"]["qwen3-8b-fewshot"]["scoring_agreement_failures"]
    assert metrics["scoring_stop"] == [] and rep["agreement"] == 1.0 and rep["n"] > 0


# --- decision 4: the goal-channel rule is advisory and read in nats --------------------------------


def test_goal_channel_rule_in_nats(monkeypatch, tmp_path):
    null = _run(monkeypatch, tmp_path / "null", PlantedScorer, base=-2.0, effect=0.0)
    real = _run(monkeypatch, tmp_path / "real", PlantedScorer, base=-2.0, effect=1.0)
    assert null["PILOT_RULE_GOAL_CHANNEL_nats"] is True and real["PILOT_RULE_GOAL_CHANNEL_nats"] is False
    assert null["goal_channel_rule_is_advisory"] is True


def test_power_planning_is_reported_in_nats(monkeypatch, tmp_path):
    rep = _run(monkeypatch, tmp_path, PlantedScorer)["reports"]["qwen3-8b-fewshot"]
    n = rep["n_for_80pct_power"]
    assert set(n) == {"0.5", "0.75", "1.0"}
    assert n["1.0"] is None or n["0.5"] is None or n["1.0"] <= n["0.5"]


# --- co-primary Llama: a gated model is recorded, not fatal ---------------------------------------


class GatedLlama(PlantedScorer):
    def __init__(self, model_name, model_tag=None, **kw):
        if model_name == C.WM_FEWSHOT_LLAMA:
            raise OSError("You are trying to access a gated repo.\nMake sure to have access.")
        super().__init__(model_name, model_tag=model_tag, **kw)


def test_gated_model_leaves_the_run_open_and_resume_scores_only_it(monkeypatch, tmp_path):
    metrics = _run(monkeypatch, tmp_path, GatedLlama, expect=1)
    assert "llama3.1-8b-fewshot" in metrics["models_not_scored"]
    assert "INCOMPLETE" in (tmp_path / "pilotA_score" / "STATUS.md").read_text()
    before = read_jsonl(tmp_path / "pilotA_score" / "scoring.jsonl")
    assert before and all(s["model"] != "llama3.1-8b-fewshot" for s in before)

    metrics = _run(monkeypatch, tmp_path, PlantedScorer, cfg={"run_id": "pilotA_score", "resume": True})
    after = read_jsonl(tmp_path / "pilotA_score" / "scoring.jsonl")
    assert metrics["models_not_scored"] == {} and "llama3.1-8b-fewshot" in metrics["reports"]
    assert after[: len(before)] == before  # nothing rescored
    assert {s["model"] for s in after[len(before):]} == {"llama3.1-8b-fewshot"}


# --- decision 6: results by the fact that decides each failure ------------------------------------


def test_effects_are_split_by_determination_reason(monkeypatch, tmp_path):
    rep = _run(monkeypatch, tmp_path, PlantedScorer)["reports"]["qwen3-8b-fewshot"]
    by_reason = rep["by_reason"]
    assert by_reason and "not_determined" not in by_reason
    assert sum(r["n_items"] for r in by_reason.values()) == rep["goal_agnostic"]["n_items"]
    assert set(rep["false_success_by_reason"]) >= set(by_reason)


# --- retained-format check: the agent's raw text is kept -------------------------------------------


def test_trace_keeps_the_agents_raw_text(isolated_foresight):  # noqa: F811
    ep = run_episode(isolated_foresight, FixedAgent("go to drawer 1"), "train/0", "train", "t",
                     ForesightConfig("oracle", retention="retained"), step_cap=2)
    for step in ep["trace"]:
        assert step["proposal_raw"] == "Action: go to drawer 1"
        assert step["executed_raw"] == "Action: go to drawer 1"


class Rambler(FixedAgent):
    """Copies the retained-history pattern and never writes an Action line."""

    def complete(self, messages, max_tokens=256):
        return "Thought: x\nProposed action: go to drawer 1\nExpected outcome: You arrive at drawer 1."


def test_reply_without_action_line_is_a_format_error_not_an_action(isolated_foresight):  # noqa: F811
    ep = run_episode(isolated_foresight, Rambler("unused"), "train/0", "train", "t",
                     ForesightConfig("oracle", retention="retained"), step_cap=1)
    step = ep["trace"][0]
    assert step["proposal_format_error"] and step["executed_format_error"] and step["under_true_format_error"]
    assert step["executed"] == "look" and step["proposal_valid"]  # not counted as an invalid proposal
    assert step["decision_critical"] is False


# --- decision 14 and the home quota: how vLLM is started -------------------------------------------


def test_vllm_runs_without_prefix_caching_and_caches_stay_off_home():
    common = (ROOT / "pace" / "_common.sh").read_text()
    assert "--no-enable-prefix-caching" in common
    assert "--enable-prefix-caching" not in common.replace("--no-enable-prefix-caching", "")
    for var in ("XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"):
        assert f'export {var}="' in common
    assert 'export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"' in common


def test_sigmoid_still_maps_L_for_the_filed_delta_p():
    assert sigmoid(0.0) == 0.5


class SaysCannot(PlantedScorer):
    """Writes failures in its own words, as Llama does."""

    def free_generate(self, messages, max_new_tokens=64):
        return "You can't find the mug 1 here."


def test_pilot_a_reads_written_failures_with_the_prediction_parser(monkeypatch, tmp_path):
    from wf.util import read_jsonl
    _run(monkeypatch, tmp_path / "cannot", SaysCannot)
    gens = read_jsonl(tmp_path / "cannot" / "pilotA_score" / "generations.jsonl")
    assert gens and {g["generated_class"] for g in gens} == {"noop_paraphrase"}  # the old parser said "unparseable"
