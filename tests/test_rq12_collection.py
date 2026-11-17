"""Confirmatory RQ1/RQ2 item collection (decided 2026-11-17, before any unseen-game data)."""
import json

import pytest
import yaml

import wf.runs as runs_mod
from wf.data.transitions import HistoryFacts, contents_holding_alternatives, determination_reason
from wf.env.scene import Scene
from wf.util import read_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")


@pytest.fixture
def mock_root(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    return tmp_path


SCENE = Scene(receptacles=["cabinet 1", "countertop 1", "coffeemachine 1", "microwave 1", "sinkbasin 1"],
              locations={"mug 1": {"cabinet 1"}, "apple 1": {"countertop 1"}})


def test_contents_failure_at_the_current_receptacle():
    facts = HistoryFacts(knows_position=True, location="countertop 1", contents={"countertop 1": {"apple 1"}})
    acts = [a for a, _, _ in contents_holding_alternatives(facts, "mug 1", "coffeemachine", SCENE)]
    assert "take mug 1 from countertop 1" in acts


def test_closed_receptacle_counts_as_open_state():
    facts = HistoryFacts(knows_position=True, location="cabinet 1", closed={"cabinet 1"})
    acts = [a for a, _, _ in contents_holding_alternatives(facts, "mug 1", "coffeemachine", SCENE)]
    assert "take mug 1 from cabinet 1" in acts


def test_hands_full_and_not_holding_failures():
    facts = HistoryFacts(knows_position=True, location="countertop 1", holding="spoon 1",
                         contents={"countertop 1": {"apple 1"}})
    acts = [a for a, _, _ in contents_holding_alternatives(facts, "mug 1", "coffeemachine", SCENE)]
    assert "take apple 1 from countertop 1" in acts  # hands full
    assert {"move mug 1 to countertop 1", "move mug 1 to coffeemachine 1", "heat mug 1 with microwave 1",
            "clean mug 1 with sinkbasin 1"} <= set(acts)  # not holding the goal object


def test_nothing_is_generated_while_holding_the_goal_or_with_unknown_position():
    holding_goal = HistoryFacts(knows_position=True, location="countertop 1", holding="mug 1",
                                contents={"countertop 1": set()})
    acts = [a for a, _, _ in contents_holding_alternatives(holding_goal, "mug 1", "coffeemachine", SCENE)]
    assert not any(a.startswith(("move mug", "heat mug", "clean mug")) for a in acts)
    assert contents_holding_alternatives(HistoryFacts(), "mug 1", "coffeemachine", SCENE) == []


def test_collection_saves_what_the_confirmatory_scoring_needs(mock_root):
    assert runs_mod.run_pilotA_transitions({"run_id": "rq12_transitions", "split": "valid_unseen", "n_games": 3,
                                            "step_cap": 12, "min_items": 1, "min_states": 1}) in (0, 1)
    recs = read_jsonl(mock_root / "rq12_transitions" / "transitions.jsonl")
    assert recs and {r["split"] for r in recs} == {"valid_unseen"}
    for r in recs:
        assert r["hidden_state"] and len(r["hidden_state_sha256"]) == 64
        assert r["scene_receptacles"] and isinstance(r["scene_locations"], dict)
        failed = r["true_obs"].strip() == "Nothing happens."
        assert (r["determination_reason"] == determination_reason(r)) if failed else r["determination_reason"] is None
    # contents/holding failures are generated at steps the sampled alternatives skip
    odd = {r["state_idx"] for r in recs if r["state_idx"] % 2 == 1}
    assert odd


def test_collection_stops_after_the_per_job_limit_and_resumes(mock_root):
    cfg = {"run_id": "rq12_transitions", "split": "valid_unseen", "n_games": 3, "step_cap": 12,
           "min_items": 1, "min_states": 1, "max_games_per_job": 2}
    assert runs_mod.run_pilotA_transitions(dict(cfg)) == 1
    status = (mock_root / "rq12_transitions" / "STATUS.md").read_text()
    assert "PARTIAL — 2/3 games saved" in status
    runs_mod.run_pilotA_transitions({**cfg, "resume": True})
    games = {r["game_id"] for r in read_jsonl(mock_root / "rq12_transitions" / "transitions.jsonl")}
    assert len(games) == 3
    assert "PARTIAL" not in (mock_root / "rq12_transitions" / "STATUS.md").read_text()


def test_confirmatory_config_matches_the_prereg():
    from pathlib import Path
    cfg = yaml.safe_load((Path(runs_mod.__file__).parent / "config" / "rq12_transitions.yaml").read_text())
    assert cfg["split"] == "valid_unseen" and cfg["n_games"] == 134
    assert cfg["min_items"] >= 1000 and cfg["min_states"] >= 200
    assert "rq12_transitions" in runs_mod.RUNNERS
