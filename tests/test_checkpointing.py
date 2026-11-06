"""A job killed at any moment keeps what it finished, and --resume completes it without rework."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import wf.runs as runs_mod
import wf.tools.assemble_pilot_report as assemble_pilot_report
import wf.tools.verify_models as verify_models
import wf.tools.verify_real_env as verify_env
from wf.env.server import AlfworldClient
from wf.util import append_jsonl, read_jsonl, repair_jsonl, write_json

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")

    def use(name: str) -> Path:
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        (root / "FILED.txt").write_text("filed_date: x\n")
        monkeypatch.setenv("RUNS_DIR", str(root))
        return root

    return use


def lines(path: Path) -> list[str]:
    return sorted(json.dumps(r, sort_keys=True) for r in read_jsonl(path))


def dies_on_call(monkeypatch, name: str, n: int) -> callable:
    """Stands in for a wall-clock kill on the n-th call; returns the undo."""
    real = getattr(runs_mod, name)
    calls = [0]

    def wrapped(*args, **kwargs):
        calls[0] += 1
        if calls[0] == n:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(runs_mod, name, wrapped)
    return lambda: monkeypatch.setattr(runs_mod, name, real)


def forbid(monkeypatch, name: str) -> None:
    def boom(*args, **kwargs):
        raise AssertionError(f"{name} ran on finished work")

    monkeypatch.setattr(runs_mod, name, boom)


# --- files -------------------------------------------------------------------------------------


def test_repair_cuts_only_a_half_written_last_line(tmp_path, monkeypatch):
    for key in MOCK_KEYS:
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "x.jsonl"
    append_jsonl(path, [{"a": 1}, {"a": 2}])
    assert repair_jsonl(path) is False
    with path.open("a") as f:
        f.write('{"a": 3, "b')
    assert repair_jsonl(path) is True
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
    append_jsonl(path, [{"a": 4}])
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 4}]


def test_repair_handles_a_missing_file_and_a_file_with_no_complete_line(tmp_path):
    assert repair_jsonl(tmp_path / "none.jsonl") is False
    path = tmp_path / "x.jsonl"
    path.write_text('{"a"')
    assert repair_jsonl(path) is True
    assert read_jsonl(path) == []


def test_a_failed_write_keeps_the_old_json(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    write_json(path, {"good": 1})

    def fails(*args):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", fails)
    with pytest.raises(OSError):
        write_json(path, {"bad": 1})
    assert json.loads(path.read_text()) == {"good": 1}


def test_resume_survives_a_torn_last_line(mock_runs, monkeypatch):
    root = mock_runs("run")
    config = {"run_id": "pilotB_headroom", "n_games": 2}
    runs_mod.run_pilotB_headroom(dict(config))
    path = root / "pilotB_headroom" / "episodes.jsonl"
    whole = lines(path)
    raw = path.read_bytes()
    path.write_bytes(raw[:-200])  # the kill landed inside the last episode
    runs_mod.run_pilotB_headroom({**config, "resume": True})
    assert lines(path) == whole


# --- pilots ------------------------------------------------------------------------------------


def test_transitions_resume_matches_a_clean_run(mock_runs, monkeypatch):
    config = {"run_id": "pilotA_transitions", "n_games": 4, "step_cap": 12}
    clean = mock_runs("clean")
    runs_mod.run_pilotA_transitions(dict(config))

    crash = mock_runs("crash")
    undo = dies_on_call(monkeypatch, "build_transitions_from_rollout", 3)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_pilotA_transitions(dict(config))
    undo()
    path = crash / "pilotA_transitions" / "transitions.jsonl"
    assert len({t["game_id"] for t in read_jsonl(path)}) == 2  # two games survived the kill
    runs_mod.run_pilotA_transitions({**config, "resume": True})
    assert lines(path) == lines(clean / "pilotA_transitions" / "transitions.jsonl")


def test_a_game_saved_without_its_done_mark_is_redone_not_trusted(mock_runs):
    config = {"run_id": "pilotA_transitions", "n_games": 3, "step_cap": 12}
    root = mock_runs("run")
    runs_mod.run_pilotA_transitions(dict(config))
    d = root / "pilotA_transitions"
    whole = lines(d / "transitions.jsonl")
    records = read_jsonl(d / "transitions.jsonl")
    last = records[-1]["game_id"]
    keep = [r for r in records if r["game_id"] != last] + [r for r in records if r["game_id"] == last][:2]
    (d / "transitions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    (d / "games_done.jsonl").write_text("".join(json.dumps({"game_id": g}) + "\n"
                                                for g in dict.fromkeys(r["game_id"] for r in records) if g != last))
    runs_mod.run_pilotA_transitions({**config, "resume": True})
    assert lines(d / "transitions.jsonl") == whole


def test_pilot_a_scoring_saves_each_score_and_resumes_to_the_same_answer(mock_runs, monkeypatch):
    config = {"run_id": "pilotA_transitions", "n_games": 6, "step_cap": 12}
    clean = mock_runs("clean")
    runs_mod.run_pilotA_transitions(dict(config))
    runs_mod.run_pilotA_score({"run_id": "pilotA_score"})

    crash = mock_runs("crash")
    runs_mod.run_pilotA_transitions(dict(config))
    undo = dies_on_call(monkeypatch, "score_transition", 41)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_pilotA_score({"run_id": "pilotA_score"})
    undo()
    path = crash / "pilotA_score" / "scoring.jsonl"
    assert len(read_jsonl(path)) == 40  # nothing scored before the kill is lost

    runs_mod.run_pilotA_score({"run_id": "pilotA_score", "resume": True})
    resumed = read_jsonl(path)
    keys = [(s["model"], s["transition_id"], s["condition"]) for s in resumed]
    assert len(keys) == len(set(keys))  # no item scored twice
    assert lines(path) == lines(clean / "pilotA_score" / "scoring.jsonl")
    assert (crash / "pilotA_score" / "metrics.json").read_text() == (clean / "pilotA_score" / "metrics.json").read_text()


def test_a_finished_scoring_run_rescores_nothing_on_resume(mock_runs, monkeypatch):
    mock_runs("run")
    runs_mod.run_pilotA_transitions({"run_id": "pilotA_transitions", "n_games": 6, "step_cap": 12})
    runs_mod.run_pilotA_score({"run_id": "pilotA_score"})
    forbid(monkeypatch, "score_transition")
    runs_mod.run_pilotA_score({"run_id": "pilotA_score", "resume": True})


# --- Tier 0 ------------------------------------------------------------------------------------


def test_smoke_resume_matches_a_clean_run(mock_runs, monkeypatch):
    config = {"run_id": "t0_smoke", "n_games": 12, "step_cap": 12}
    clean = mock_runs("clean")
    runs_mod.run_t0_smoke(dict(config))
    crash = mock_runs("crash")
    undo = dies_on_call(monkeypatch, "run_episode", 3)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_smoke(dict(config))
    undo()
    assert len(read_jsonl(crash / "t0_smoke" / "episodes.jsonl")) == 2
    runs_mod.run_t0_smoke({**config, "resume": True})
    for name in ("episodes.jsonl", "observations.jsonl"):
        assert lines(crash / "t0_smoke" / name) == lines(clean / "t0_smoke" / name)
    assert (crash / "t0_smoke" / "STATUS.md").read_text() == (clean / "t0_smoke" / "STATUS.md").read_text()


def test_determinism_resume_matches_a_clean_run(mock_runs, monkeypatch):
    config = {"run_id": "t0_agent_determinism", "n_games": 4}
    clean = mock_runs("clean")
    runs_mod.run_t0_agent_determinism(dict(config))
    crash = mock_runs("crash")
    undo = dies_on_call(monkeypatch, "run_episode", 6)  # second episode of the third game
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_agent_determinism(dict(config))
    undo()
    assert len(read_jsonl(crash / "t0_agent_determinism" / "episodes.jsonl")) == 4
    runs_mod.run_t0_agent_determinism({**config, "resume": True})
    assert lines(crash / "t0_agent_determinism" / "episodes.jsonl") == lines(clean / "t0_agent_determinism" / "episodes.jsonl")
    assert (crash / "t0_agent_determinism" / "STATUS.md").read_text() == (clean / "t0_agent_determinism" / "STATUS.md").read_text()


def scorer_dies_after(monkeypatch, method: str, n: int) -> callable:
    real = getattr(runs_mod.LogoddsScorer, method)
    calls = [0]

    def wrapped(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == n:
            raise KeyboardInterrupt
        return real(self, *args, **kwargs)

    monkeypatch.setattr(runs_mod.LogoddsScorer, method, wrapped)
    return lambda: monkeypatch.setattr(runs_mod.LogoddsScorer, method, real)


def test_ckpt_accuracy_keeps_built_games_and_generations(mock_runs, monkeypatch):
    config = {"run_id": "t0_ckpt_accuracy", "n_items": 40}
    clean = mock_runs("clean")
    runs_mod.run_t0_ckpt_accuracy(dict(config))

    crash = mock_runs("crash")
    d = crash / "t0_ckpt_accuracy"
    undo = dies_on_call(monkeypatch, "_walk_transitions", 3)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_ckpt_accuracy(dict(config))
    undo()
    assert len(read_jsonl(d / "games_done.jsonl")) == 2

    undo = scorer_dies_after(monkeypatch, "free_generate", 11)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_ckpt_accuracy({**config, "resume": True})
    undo()
    assert len(read_jsonl(d / "generations.jsonl")) == 10

    runs_mod.run_t0_ckpt_accuracy({**config, "resume": True})
    assert lines(d / "generations.jsonl") == lines(clean / "t0_ckpt_accuracy" / "generations.jsonl")
    assert (d / "STATUS.md").read_text() == (clean / "t0_ckpt_accuracy" / "STATUS.md").read_text()


def test_scoring_agreement_keeps_built_games_and_scores(mock_runs, monkeypatch):
    config = {"run_id": "t0_scoring_agreement"}
    clean = mock_runs("clean")
    runs_mod.run_t0_scoring_agreement(dict(config))

    crash = mock_runs("crash")
    d = crash / "t0_scoring_agreement"
    undo = dies_on_call(monkeypatch, "_walk_transitions", 4)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_scoring_agreement(dict(config))
    undo()
    assert len(read_jsonl(d / "games_done.jsonl")) == 3

    undo = scorer_dies_after(monkeypatch, "free_generate", 16)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_scoring_agreement({**config, "resume": True})
    undo()
    assert len(read_jsonl(d / "scoring.jsonl")) == 15

    runs_mod.run_t0_scoring_agreement({**config, "resume": True})
    assert lines(d / "scoring.jsonl") == lines(clean / "t0_scoring_agreement" / "scoring.jsonl")
    assert (d / "STATUS.md").read_text() == (clean / "t0_scoring_agreement" / "STATUS.md").read_text()


# --- what resume may and may not trust -----------------------------------------------------------


def test_a_crash_leaves_no_stale_pass(mock_runs, monkeypatch):
    root = mock_runs("run")
    config = {"run_id": "t0_smoke", "n_games": 12, "step_cap": 12}
    runs_mod.run_t0_smoke(dict(config))
    assert (root / "t0_smoke" / "STATUS.md").exists()
    undo = dies_on_call(monkeypatch, "run_episode", 2)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_t0_smoke(dict(config))
    undo()
    assert not (root / "t0_smoke" / "STATUS.md").exists()
    assert runs_mod._already_passed("t0_smoke") is False


def test_resume_skips_only_a_pass_at_this_commit(mock_runs, monkeypatch):
    root = mock_runs("run")
    runs_mod.run_t0_smoke({"run_id": "t0_smoke", "n_games": 12, "step_cap": 12})
    assert runs_mod._already_passed("t0_smoke") is True
    monkeypatch.setattr(runs_mod, "git_sha", lambda: "another-commit")
    assert runs_mod._already_passed("t0_smoke") is False
    status = root / "t0_smoke" / "STATUS.md"
    status.write_text(status.read_text().replace("PASS", "FAIL"))
    monkeypatch.setattr(runs_mod, "git_sha", lambda: (root / "t0_smoke" / "git_sha").read_text().strip())
    assert runs_mod._already_passed("t0_smoke") is False


def test_main_skips_a_passed_run_on_resume(mock_runs, monkeypatch):
    mock_runs("run")
    runs_mod.run_t0_smoke({"run_id": "t0_smoke", "n_games": 12, "step_cap": 12})
    monkeypatch.setitem(runs_mod.RUNNERS, "t0_smoke", lambda cfg: pytest.fail("a passed run ran again"))
    with pytest.raises(SystemExit) as stop:
        runs_mod.main(["t0_smoke", "--resume"])
    assert stop.value.code == 0


def test_resume_refuses_to_mix_commits(mock_runs, monkeypatch):
    mock_runs("run")
    runs_mod.run_t0_smoke({"run_id": "t0_smoke", "n_games": 12, "step_cap": 12})
    monkeypatch.setattr(runs_mod, "git_sha", lambda: "another-commit")
    with pytest.raises(SystemExit, match="must not mix"):
        runs_mod.run_t0_smoke({"run_id": "t0_smoke", "n_games": 12, "step_cap": 12, "resume": True})
    runs_mod.run_t0_smoke({"run_id": "t0_smoke", "n_games": 12, "step_cap": 12})  # a fresh run is fine


def test_reported_runs_never_say_pass_and_resume_still_skips_them(mock_runs, monkeypatch):
    root = mock_runs("run")
    runs_mod.run_t0_ckpt_accuracy({"run_id": "t0_ckpt_accuracy", "n_items": 10})
    first = (root / "t0_ckpt_accuracy" / "STATUS.md").read_text().splitlines()[0]
    assert "REPORTED (no target)" in first and "PASS" not in first
    assert runs_mod._already_passed("t0_ckpt_accuracy") is True


def test_low_determinism_says_below_target_on_the_first_line(mock_runs, monkeypatch):
    root = mock_runs("run")
    real = runs_mod.run_episode
    calls = [0]

    def drifts(*args, **kwargs):
        calls[0] += 1
        ep = real(*args, **kwargs)
        if calls[0] % 2 == 0:  # the second run of each game disagrees everywhere
            for step in ep["trace"]:
                step["executed"] = "look"
        return ep

    monkeypatch.setattr(runs_mod, "run_episode", drifts)
    runs_mod.run_t0_agent_determinism({"run_id": "t0_agent_determinism", "n_games": 2})
    first = (root / "t0_agent_determinism" / "STATUS.md").read_text().splitlines()[0]
    assert "REPORTED (BELOW TARGET" in first and "PASS" not in first
    # The report's status column comes from this line, whatever the rest of the file says.
    first, _ = assemble_pilot_report._status(root, "t0_agent_determinism")
    assert "BELOW TARGET" in first


# --- verification tools --------------------------------------------------------------------------


def test_verifiers_report_after_every_game(mock_runs):
    client = AlfworldClient()
    client.create()
    games = list(client.list_games("train"))
    seen: list[dict] = []
    verify_env.check_games(client, games[:3], None, checkpoint=seen.append)
    assert [p["games"] for p in seen] == [1, 2, 3]
    assert set(seen[0]["timing_s"]) == {"scan_s", "reset_s", "step_s", "soundness_s", "build_s"}
    seen.clear()
    verify_env.check_winning(client, games[:6], 2, checkpoint=seen.append)
    assert [p["games_with_plan"] for p in seen] == [1, 2]


def test_winning_check_stops_scanning_at_the_cap(mock_runs, monkeypatch):
    client = AlfworldClient()
    client.create()
    scans = []
    real = verify_env.scan_scene

    def counted(*args, **kwargs):
        scans.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(verify_env, "scan_scene", counted)
    verify_env.check_winning(client, list(client.list_games("train"))[:30], 1)
    assert len(scans) < 30


def test_agent_checks_save_after_every_episode(mock_runs, monkeypatch):
    class Args:
        games = 2

    monkeypatch.setattr(verify_models, "_games", lambda client, split, n, cfg: list(client.list_games("train"))[:n])
    out: dict = {}
    snapshots: list[dict] = []
    verify_models.agent_checks(Args(), out, lambda: snapshots.append(json.loads(json.dumps(out, default=str))))
    assert len(snapshots) == 12  # 6 conditions x 2 games
    assert snapshots[0]["none"]["games"] == 1
