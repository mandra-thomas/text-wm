"""Receptacles sharing a spot, as measured on the real server (notes/deviations.md)."""
from __future__ import annotations

import json

import pytest

import wf.runs as runs_mod
from wf.data.transitions import (
    _receptacle,
    at_shared_spot,
    exclusion_counts,
    filter_pilot_a_items,
    history_facts,
    main_test_items,
)
from wf.env.scene import scan_scene
from wf.env.server import AlfworldClient
from wf.env.sim import World
from wf.util import read_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WF_MOCK_ENV", "1")
    c = AlfworldClient()
    c.create()
    return c


def games(client, shared: bool) -> list[str]:
    """Odd sim games have drawer 1 and countertop 1 at one spot."""
    return [g for g in client.list_games("train") if bool(int(g.split("/")[1]) % 2) == shared]


def test_sim_behaves_like_the_real_server_at_a_shared_spot():
    world = World(shared=("drawer 1", "cabinet 1"))
    assert "go to cabinet 1" in world.admissible()
    world.step("go to drawer 1")
    assert "go to cabinet 1" not in world.admissible()
    assert world.step("go to cabinet 1") == "Nothing happens."
    assert world.step("open cabinet 1").startswith("You open the cabinet 1.")
    assert world.step("take plate 1 from cabinet 1") == "You pick up the plate 1 from the cabinet 1."


def test_scan_sees_inside_both_receptacles_of_a_shared_spot(client):
    scene = scan_scene(client, games(client, True)[0], "train")
    assert scene.shares_spot("drawer 1", "countertop 1") and scene.shares_spot("countertop 1", "drawer 1")
    # The scan reaches countertop 1 straight from drawer 1, where `go to` does nothing.
    assert scene.receptacles.index("countertop 1") == scene.receptacles.index("drawer 1") + 1
    on_counter = [o for o, r in scene.locations.items() if r == "countertop 1"]
    assert on_counter  # the real scan missed exactly these objects


def test_scan_reports_no_shared_spot_where_there_is_none(client):
    scene = scan_scene(client, games(client, False)[0], "train")
    assert not any(scene.shared.values())
    assert "plate 1" in scene.locations


@pytest.mark.parametrize("action, receptacle", [
    ("take potato 1 from cabinet 2", "cabinet 2"),  # "to " inside the object name must not match
    ("move tomato 1 to shelf 3", "shelf 3"),
    ("heat egg 1 with microwave 1", "microwave 1"),
    ("open drawer 1", "drawer 1"),
    ("close fridge 1", "fridge 1"),
    ("look", None),
])
def test_receptacle_named_by_an_action(action, receptacle):
    assert _receptacle(action) == receptacle


def test_flag_needs_a_known_position_and_a_shared_target(client):
    scene = scan_scene(client, games(client, True)[0], "train")
    at_counter = history_facts("You arrive at countertop 1. On the countertop 1, you see a apple 1.")
    assert at_shared_spot(at_counter, "take pen 1 from drawer 1", scene) is True
    assert at_shared_spot(at_counter, "take plate 1 from cabinet 1", scene) is False
    assert at_shared_spot(history_facts(""), "take plate 1 from cabinet 1", scene) is False


def test_flagged_items_are_kept_but_left_out_of_the_main_test():
    items = [{"id": 1, "shared_location": True}, {"id": 2, "shared_location": False}, {"id": 3}]
    assert [t["id"] for t in main_test_items(items)] == [2, 3]
    assert len(items) == 3


def _run_pilot_a(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    runs_mod.run_pilotA_transitions({"run_id": "pilotA_transitions", "n_games": 12, "step_cap": 12})
    transitions = read_jsonl(tmp_path / "pilotA_transitions" / "transitions.jsonl")
    runs_mod.run_pilotA_score({"run_id": "pilotA_score"})
    return transitions, json.loads((tmp_path / "pilotA_score" / "metrics.json").read_text())


def test_main_result_leaves_flagged_items_out_and_sensitivity_puts_them_back(monkeypatch, tmp_path):
    transitions, metrics = _run_pilot_a(monkeypatch, tmp_path)
    eligible = filter_pilot_a_items(transitions)
    usable = [t for t in eligible if t["targets_goal_object"] and t["history_goal_agnostic"]]
    flagged = [t for t in usable if t["shared_location"]]
    assert flagged, "the simulator must produce shared-location items or this test proves nothing"
    assert exclusion_counts(transitions).get("shared_location", 0) > 0

    report = metrics["reports"]["qwen3-8b-fewshot"]
    assert report["goal_agnostic"]["n_items"] == len(usable) - len(flagged)
    assert report["with_shared_location"]["n_flagged"] == len(flagged)
    assert report["with_shared_location"]["goal_agnostic"]["n_items"] == len(usable)

    # Flagged items are still scored, so the sensitivity check never needs a rerun.
    scored = {s["transition_id"] for s in read_jsonl(tmp_path / "pilotA_score" / "scoring.jsonl")}
    assert {t["id"] for t in flagged} <= scored


def test_sample_size_gate_counts_only_main_test_items(monkeypatch, tmp_path):
    transitions, _ = _run_pilot_a(monkeypatch, tmp_path)
    status = (tmp_path / "pilotA_transitions" / "STATUS.md").read_text()
    eligible = filter_pilot_a_items(transitions)
    assert f"main_test={len(main_test_items(eligible))}" in status
    assert f"shared_location={len(eligible) - len(main_test_items(eligible))}" in status


def test_flagged_items_cannot_carry_the_sample_size_gate(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    config = {"run_id": "pilotA_transitions", "n_games": 60, "step_cap": 12}
    assert runs_mod.run_pilotA_transitions(dict(config)) == 0
    # One item short of the gate, with plenty of states, so only the item count can fail it.
    monkeypatch.setattr(runs_mod, "main_test_items", lambda items: list(items)[:499])
    assert runs_mod.run_pilotA_transitions(dict(config)) == 1
    assert "PASS" not in (tmp_path / "pilotA_transitions" / "STATUS.md").read_text().splitlines()[-1]
    assert "states=" in (tmp_path / "pilotA_transitions" / "STATUS.md").read_text()
    status = (tmp_path / "pilotA_transitions" / "STATUS.md").read_text()
    assert int(status.split(" states=")[1].split()[0]) >= 80


def test_A_is_computed_without_and_then_with_flagged_items(monkeypatch, tmp_path):
    sizes: list[int] = []
    real = runs_mod.asymmetry_A

    def spy(scores, items):
        sizes.append(len(items))
        return real(scores, items)

    monkeypatch.setattr(runs_mod, "asymmetry_A", spy)
    transitions, metrics = _run_pilot_a(monkeypatch, tmp_path)
    eligible = filter_pilot_a_items(transitions)
    flagged = len(eligible) - len(main_test_items(eligible))
    assert flagged > 0
    main, with_flagged = sizes[0], sizes[1]
    assert with_flagged - main == flagged
