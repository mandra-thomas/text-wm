"""Confirmatory RQ2 scoring and analysis (wf/rq2.py)."""
import json

import pytest

import wf.runs as runs_mod
from wf import rq2
from wf.util import read_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")


def _t(action, task, swapped, recs=("countertop 1", "cabinet 1", "shelf 1")):
    return {"id": "t1", "action": action, "task_line": f"Your task is to: {task}",
            "swapped_task_line": f"Your task is to: {swapped}", "scene_receptacles": list(recs)}


@pytest.mark.parametrize("action, task, swapped, mention, fourth", [
    ("heat cup 1 with microwave 1", "heat some cup and put it in countertop.", "heat some potato and put it in countertop.",
     "clean some cup and put it in countertop.", "clean some potato and put it in countertop."),
    ("cool mug 1 with fridge 1", "put a cool mug in countertop.", "put a cool apple in countertop.",
     "put a clean mug in countertop.", "put a clean apple in countertop."),
    ("clean knife 1 with sinkbasin 1", "clean some knife and put it in drawer.", "clean some fork and put it in drawer.",
     "heat some knife and put it in drawer.", "heat some fork and put it in drawer."),
])
def test_mention_lines_swap_the_transformation_for_a_merely_irrelevant_one(action, task, swapped, mention, fourth):
    m, f = rq2.mention_lines(_t(action, task, swapped))
    assert m == f"Your task is to: {mention}" and f == f"Your task is to: {fourth}"


def test_mention_lines_for_move_change_the_destination_to_another_type_in_the_room():
    t = _t("move mug 1 to countertop 1", "put a mug in countertop.", "put a spraybottle in countertop.")
    m, f = rq2.mention_lines(t)
    other = m.rsplit(" ", 1)[1].rstrip(".")
    assert other in ("cabinet", "shelf") and m.startswith("Your task is to: put a mug in ")
    assert f == f"Your task is to: put a spraybottle in {other}."
    assert rq2.mention_lines(t) == (m, f)  # seeded per item


@pytest.mark.parametrize("action, task", [
    ("take mug 1 from cabinet 1", "put a mug in countertop."),           # take: any task naming it needs it picked up
    ("heat cup 1 with microwave 1", "put a cup in countertop."),         # aligned task is not a heat task
    ("move mug 1 to cabinet 1", "put a mug in countertop."),             # destination is not the goal's receptacle
])
def test_mention_lines_are_not_built_where_the_design_does_not_apply(action, task):
    assert rq2.mention_lines(_t(action, task, task.replace("mug", "apple").replace("cup", "apple"))) is None


def test_shards_split_games_disjointly_and_completely():
    items = [{"game_id": f"g{i % 7}", "id": str(i)} for i in range(40)]
    parts = [rq2.shard_items(items, i, 3) for i in range(3)]
    games = [{t["game_id"] for t in p} for p in parts]
    assert sum(len(p) for p in parts) == 40 and not (games[0] & games[1] or games[0] & games[2] or games[1] & games[2])


@pytest.fixture
def mock_root(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    assert runs_mod.run_pilotA_transitions({"run_id": "rq12_transitions_smoke", "split": "train", "n_games": 6,
                                            "step_cap": 12, "min_items": 1, "min_states": 1}) == 0
    from wf.tools.verify_items import verify_run
    assert verify_run("rq12_transitions_smoke", server=True)["passed"]
    return tmp_path


def _score_cfg(**kw):
    return {"run_id": "rq2_smoke", "sources": ["rq12_transitions_smoke"], "models": list(rq2.CO_PRIMARY),
            "mention": True, "generations": True, "n_shards": 2, **kw}


def test_end_to_end_two_shards_then_analysis(mock_root):
    for i in range(2):
        assert rq2.run_rq2_score(_score_cfg(shard=i)) == 0
    acfg = {"run_id": "rq2_smoke_analysis", "source": "rq2_smoke", "sources": ["rq12_transitions_smoke"],
            "min_items": 1, "min_states": 1}
    assert rq2.run_rq2_analysis(acfg) == 0
    m = json.loads((mock_root / "rq2_smoke_analysis" / "metrics.json").read_text())
    assert m["items"]["missing_primary_scores"] == 0 and m["items"]["prereg_minimum_met"]
    for model in rq2.CO_PRIMARY:
        assert "delta_L_nats" in m["per_model"][model] and m["verdicts"]["per_model"][model]["verdict"]
        assert "goal_directed" in m["per_model"][model]["object_mention_2x2"]
    status = (mock_root / "rq2_smoke_analysis" / "STATUS.md").read_text()
    assert "H2 overall" in status


def test_analysis_refuses_until_every_shard_is_complete(mock_root):
    assert rq2.run_rq2_score(_score_cfg(shard=0)) == 0
    acfg = {"run_id": "rq2_smoke_analysis", "source": "rq2_smoke", "sources": ["rq12_transitions_smoke"],
            "min_items": 1, "min_states": 1}
    assert rq2.run_rq2_analysis(acfg) == 1
    assert "waits for every scoring shard" in (mock_root / "rq2_smoke_analysis" / "STATUS.md").read_text()


def test_the_prereg_minimum_is_enforced(mock_root):
    for i in range(2):
        rq2.run_rq2_score(_score_cfg(shard=i))
    acfg = {"run_id": "rq2_smoke_analysis", "source": "rq2_smoke", "sources": ["rq12_transitions_smoke"],
            "min_items": 1000, "min_states": 200}
    assert rq2.run_rq2_analysis(acfg) == 1


def test_resume_does_not_duplicate_scores(mock_root):
    cfg = _score_cfg(shard=0, n_shards=1)
    rq2.run_rq2_score(cfg)
    path = mock_root / "rq2_smoke" / "shard_00_of_01" / "scoring.jsonl"
    before = len(read_jsonl(path))
    rq2.run_rq2_score({**cfg, "resume": True})
    assert len(read_jsonl(path)) == before


def test_the_scoring_job_needs_a_named_run():
    from pathlib import Path
    text = (Path(runs_mod.__file__).parents[1] / "pace" / "rq2_score.sbatch").read_text()
    assert "${RQ2_RUN:?" in text and "start_vllm" not in text
