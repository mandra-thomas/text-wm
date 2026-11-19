"""Exploratory dissociation test (wf/rq1d.py): a shared sentence about the action's object in both prompts."""
import json

from wf import rq1, rq1d
from tests.test_rq1 import _item, _score_cfg, mock_root, mock_score  # noqa: F401


def _located(**kw):
    return _item(scene_locations={"saltshaker 1": ["cabinet 1"], "mug 1": ["shelf 2"]}, **kw)


def test_sentences_name_the_action_object_and_its_true_location():
    t = _located()
    assert rq1d.instance(t) == "saltshaker 1"
    assert rq1d.sentence(t, "exist") == "There is a saltshaker somewhere in the room."
    assert rq1d.sentence(t, "located") == "The saltshaker 1 is at cabinet 1."
    assert rq1d.sentence(t, "absent") == "There is no saltshaker in this room."
    assert rq1d.sentence(_item(), "located") is None  # no scene, no located arm
    assert rq1d.location(_item(scene_locations={"saltshaker 1": ["cabinet 1", "shelf 1"]})) is None


def test_pairs_differ_only_in_the_goal_and_plain_arms_match_rq1(mock_score):
    t = _located()
    assert rq1d.arm_messages(t, "aligned") == rq1.arm_messages(t, "aligned")
    assert rq1d.arm_messages(t, "swapped") == rq1.arm_messages(t, "swapped")
    for k in rq1d.SENTENCES:
        a, b = rq1d.arm_messages(t, f"aligned_{k}"), rq1d.arm_messages(t, f"swapped_{k}")
        assert a[:-1] == b[:-1]
        s = rq1d.sentence(t, k)
        assert s in a[-1]["content"] and s in b[-1]["content"]
        assert a[-1]["content"].replace(t["task_line"], t["swapped_task_line"]) == b[-1]["content"]


def test_end_to_end(mock_root, monkeypatch):
    real = rq1.rq1_sets
    monkeypatch.setattr(rq1, "rq1_sets", lambda ts: {**real(ts), "primary": [
        {**t, "scene_locations": {f"{t['goal_object']} 1": ["cabinet 1"]}} for t in real(ts)["primary"]]})
    cfg = {"run_id": "rq1d_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 2}
    for i in range(2):
        assert rq1d.run_rq1d_score({**cfg, "shard": i}) == 0
    acfg = {"run_id": "rq1d_smoke_analysis", "source": "rq1d_smoke", "sources": ["rq12_transitions_smoke"]}
    assert rq1d.run_rq1d_analysis(acfg) == 0
    m = json.loads((mock_root / "rq1d_smoke_analysis" / "metrics.json").read_text())
    for tag in rq1.FEWSHOT:
        assert {"plain", "exist", "located", "absent"} <= set(m[tag]["all"])
