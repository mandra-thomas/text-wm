"""Exploratory validation runs (wf/rq1v.py): counterfactual aligned task and yes/no feasibility."""
import json

from wf import rq1, rq1v
from tests.test_rq1 import _item, mock_root, mock_score  # noqa: F401

RECS = ["cabinet 1", "cabinet 2", "countertop 1", "drawer 1", "shelf 1"]


def test_counterfactual_lines_change_only_the_destination():
    t = _item(scene_receptacles=RECS)
    al, sw = rq1v.cf_lines(t)
    assert al != t["task_line"] and sw != t["swapped_task_line"]
    assert al.startswith("Your task is to: put some saltshaker on ") and al.endswith(".")
    assert sw == al.replace("saltshaker", "mug")
    dest = rq1v.cf_destination(t)
    assert dest in rq1v.DESTINATIONS and dest != "cabinet"
    assert rq1v.cf_destination(t) == dest  # deterministic


def test_counterfactual_skips_moves_and_rooms_without_another_destination():
    assert rq1v.cf_lines(_item(action="move saltshaker 1 to cabinet 1", scene_receptacles=RECS)) is None
    assert rq1v.cf_lines(_item(scene_receptacles=["cabinet 1", "cabinet 2"])) is None


def test_yes_no_prompt_ends_with_the_question_and_keeps_the_task_line(mock_score):
    t = _item(scene_receptacles=RECS)
    for arm, base in (("yn_aligned", "aligned"), ("yn_swapped", "swapped"), ("yn_absent", "absent")):
        m, b = rq1v.arm_messages(t, arm), rq1.arm_messages(t, base)
        assert m[:-1] == b[:-1] and m[-1]["content"].endswith(rq1v.YN_QUESTION)
        assert m[-1]["content"][: -len(rq1v.YN_QUESTION)] == b[-1]["content"][: -len("Next observation:")]


def test_counterfactual_pairs_differ_only_in_the_goal(mock_score):
    t = _item(scene_receptacles=RECS)
    a, b = rq1v.arm_messages(t, "cf_aligned"), rq1v.arm_messages(t, "cf_swapped")
    assert a[:-1] == b[:-1]
    al, sw = rq1v.cf_lines(t)
    assert a[-1]["content"].replace(al, sw) == b[-1]["content"]


def test_end_to_end(mock_root, monkeypatch):
    from wf import rq1x
    real = rq1x.rq1x_sets
    monkeypatch.setattr(rq1x, "rq1x_sets", lambda ts: {**real(ts), "primary": [
        {**t, "scene_receptacles": RECS} for t in real(ts)["primary"]]})
    cfg = {"run_id": "rq1v_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 2}
    for i in range(2):
        assert rq1v.run_rq1v_score({**cfg, "shard": i}) == 0
    assert rq1v.run_rq1v_analysis({"run_id": "rq1v_smoke_analysis", "source": "rq1v_smoke",
                                    "sources": ["rq12_transitions_smoke"]}) == 0
    m = json.loads((mock_root / "rq1v_smoke_analysis" / "metrics.json").read_text())
    for tag in rq1.FEWSHOT:
        assert {"counterfactual_task", "yes_no"} <= set(m[tag])
