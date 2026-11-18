"""Full verification of collected items (wf/tools/verify_items.py) and the RQ2 gate on it."""
import json

import pytest

import wf.runs as runs_mod
from wf import rq2
from wf.tools import verify_items as V
from wf.util import read_jsonl, write_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
HIDDEN = """# Environment Information (Only visible to Assistant)

=== Objects on Receptacles ===
 - On the countertop 1, you see a apple 1, a mug 1
 - cabinet 1 is closed, if opened, in it, you see nothing.
 - drawer 1 is closed, if opened, on the drawer 1, you see a pen 1, a pen 2
"""


def test_hidden_locations_reads_open_closed_and_surface_lines():
    assert V.hidden_locations(HIDDEN) == {"apple 1": {"countertop 1"}, "mug 1": {"countertop 1"},
                                         "pen 1": {"drawer 1"}, "pen 2": {"drawer 1"}}


def _item(**kw):
    base = {"id": "t1", "game_id": "g", "state_idx": 0, "action": "take mug 1 from cabinet 1",
            "true_obs": "Nothing happens.", "history_text": "You are in the middle of a room.\\nYour task is to: x.",
            "hidden_state": HIDDEN, "scene_locations": {"apple 1": ["countertop 1"], "mug 1": ["countertop 1"],
                                                       "pen 1": ["drawer 1"], "pen 2": ["drawer 1"]}}
    return {**base, **kw}


def test_offline_checks_flag_a_contents_label_the_hidden_state_contradicts():
    hist = "You are in the middle of a room.\nYour task is to: x.\n> go to countertop 1\nYou arrive at countertop 1. On the countertop 1, you see a apple 1."
    t = _item(action="take mug 1 from countertop 1", history_text=hist, determination_reason="contents_or_open_state")
    assert V.offline_checks([t])["problems"]["contents_label_disagrees_with_hidden_state"] == [["t1", "take mug 1 from countertop 1"]]


def test_offline_checks_flag_a_scan_that_disagrees_with_the_hidden_state():
    t = _item(scene_locations={"apple 1": ["countertop 1"], "mug 1": ["cabinet 1"], "pen 1": ["drawer 1"], "pen 2": ["drawer 1"]})
    assert V.offline_checks([t])["problems"]["scan_disagrees_with_hidden_state"] == [["g", "mug 1", ["cabinet 1"], ["countertop 1"]]]


def test_offline_checks_flag_a_success_the_rules_call_a_sure_failure():
    hist = "You are in the middle of a room.\nYour task is to: x.\n> go to drawer 1\nYou arrive at drawer 1. The drawer 1 is closed."
    t = _item(action="go to drawer 1", history_text=hist, true_obs="You arrive at drawer 1. The drawer 1 is closed.")
    assert V.offline_checks([t])["problems"]["rule_says_fail_but_succeeded"]


@pytest.fixture
def mock_root(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    assert runs_mod.run_pilotA_transitions({"run_id": "rq12_transitions_smoke", "split": "train", "n_games": 3,
                                            "step_cap": 12, "min_items": 1, "min_states": 1}) == 0
    return tmp_path


def test_real_collected_items_pass_a_full_server_verification(mock_root):
    r = V.verify_run("rq12_transitions_smoke", server=True)
    assert r["passed"] and r["full"] and r["server"]["replayed"] > 0 and r["server"]["twins_checked"] > 0
    assert V.verified("rq12_transitions_smoke") == (True, "ok")


def test_a_corrupted_outcome_fails_verification_and_blocks_rq2(mock_root):
    path = mock_root / "rq12_transitions_smoke" / "transitions.jsonl"
    items = read_jsonl(path)
    items[3]["true_obs"] = "You pick up the mug 9 from the moon 1." if items[3]["true_obs"].strip() == "Nothing happens." else "Nothing happens."
    write_jsonl(path, items)
    r = V.verify_run("rq12_transitions_smoke", server=True)
    assert not r["passed"] and r["server"]["replay_mismatches"]
    cfg = {"run_id": "rq2_smoke", "sources": ["rq12_transitions_smoke"], "models": list(rq2.CO_PRIMARY), "n_shards": 1}
    assert rq2.run_rq2_score(cfg) == 1
    assert "not verified" in (mock_root / "rq2_smoke" / "STATUS.md").read_text()


def test_verification_goes_stale_when_the_items_change(mock_root):
    V.verify_run("rq12_transitions_smoke", server=True)
    path = mock_root / "rq12_transitions_smoke" / "transitions.jsonl"
    path.write_text(path.read_text() + "")
    with path.open("a") as f:
        f.write(json.dumps(read_jsonl(path)[0]) + "\n")
    assert V.verified("rq12_transitions_smoke")[0] is False


def test_a_sample_is_not_a_full_verification(mock_root):
    V.verify_run("rq12_transitions_smoke", server=True, sample=5)
    assert V.verified("rq12_transitions_smoke")[0] is False
