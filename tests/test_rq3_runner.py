"""Confirmatory RQ3 runner: yoking on whole runs, shards, per-job limits and resume, all-shard analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import wf.rq3 as rq3_mod
import wf.runs as runs_mod
from wf import constants as C
from wf.data.transitions import action_type
from wf.util import read_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
REPO = Path(__file__).resolve().parents[1]
CFG = {"run_id": "rq3", "split": "valid_unseen", "n_games": 4, "step_cap": 30, "schedule_seeds": [1, 2, 3],
       "fs_probability": 0.75, "fs_cap": 3, "n_shards": 1}


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    for key in ("WF_SHARD", "WF_N_SHARDS"):
        monkeypatch.delenv(key, raising=False)

    def use(name: str) -> Path:
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        (root / "FILED.txt").write_text("filed_date: x\n")
        monkeypatch.setenv("RUNS_DIR", str(root))
        return root

    return use


def shard_dir(root: Path, i: int = 0, n: int = 1, base: str = "rq3") -> Path:
    return root / base / f"shard_{i:02d}_of_{n:02d}"


def lines(path: Path, drop: tuple[str, ...] = ()) -> list[str]:
    return sorted(json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True) for r in read_jsonl(path))


def resume_until_done(cfg: dict) -> int:
    """Resume jobs until the shard is complete; returns how many it took."""
    jobs = 1
    while rq3_mod.run_rq3({**cfg, "resume": True}) != 0:
        jobs += 1
        assert jobs < 100
    return jobs


# --- yoking on a whole run -----------------------------------------------------------------------


def test_every_false_failure_episode_follows_the_yoke(mock_runs):
    root = mock_runs("run")
    assert rq3_mod.run_rq3(dict(CFG)) == 0
    eps = read_jsonl(shard_dir(root) / "episodes.jsonl")
    assert len(eps) == 4 * 16
    fs = {(e["game_id"], e["retention"], e["schedule_seed"]): e for e in eps if e.get("arm") == "favorable"}
    ff = [e for e in eps if e.get("arm") == "unfavorable"]
    assert sum(e["fallbacks"] for e in ff) and sum(e["shortfall"] > 0 for e in ff)  # the mock exercises both

    def working(st: dict, verb: str | None = None) -> bool:
        return st["proposal_valid"] and not st["proposal_format_error"] and verb in (None, action_type(st["proposal"]))

    for e in ff:
        mate = fs[(e["game_id"], e["retention"], e["schedule_seed"])]
        assert e["yoked_to"] == [{"step": j["step"], "verb": j["verb"]} for j in mate["injections"]]
        assert e["planned_errors"] == mate["delivered_errors"]
        assert e["delivered_errors"] + e["shortfall"] == e["planned_errors"]
        assert [st["t"] for st in e["trace"] if st["injected"]] == [j["step"] for j in e["injections"]]
        trace, last = e["trace"], -1
        for j, target in zip(e["injections"], e["yoked_to"]):
            start = max(target["step"], last + 1)
            assert j["step"] >= start and working(trace[j["step"]])
            assert j["fallback"] == (j["verb"] != target["verb"])
            skipped = trace[start:j["step"]]
            assert not any(working(st, None if j["fallback"] else target["verb"]) for st in skipped)
            last = j["step"]
        if e["shortfall"]:
            start = max(e["yoked_to"][e["delivered_errors"]]["step"], last + 1)
            assert not any(working(st) for st in trace[start:])


def test_a_false_success_episode_with_no_errors_yokes_no_false_failures(mock_runs):
    root = mock_runs("run")
    rq3_mod.run_rq3({**CFG, "fs_probability": 0.0, "n_games": 2})
    eps = read_jsonl(shard_dir(root) / "episodes.jsonl")
    corrupted = [e for e in eps if e["condition"] == "corrupted"]
    assert corrupted and all(e["delivered_errors"] == 0 for e in corrupted)
    oracle = {(e["game_id"], e["retention"]): e["trace"] for e in eps if e["condition"] == "oracle"}
    for e in corrupted:
        assert [st["executed"] for st in e["trace"]] == [st["executed"] for st in oracle[(e["game_id"], e["retention"])]]


# --- shards --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8])
def test_shards_split_the_games_into_disjoint_complete_parts(n):
    games = [f"valid_unseen/{g}" for g in range(134)]
    parts = [rq3_mod.shard_games(games, i, n) for i in range(n)]
    flat = [g for p in parts for g in p]
    assert sorted(flat) == sorted(games) and len(flat) == len(set(flat))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_the_shard_comes_from_the_environment_first(monkeypatch):
    assert rq3_mod.shard_spec({"run_id": "rq3", "n_shards": 4, "shard": 1}) == (1, 4)
    monkeypatch.setenv("WF_SHARD", "3")
    monkeypatch.setenv("WF_N_SHARDS", "8")
    assert rq3_mod.shard_spec({"run_id": "rq3", "n_shards": 4}) == (3, 8)
    monkeypatch.setenv("WF_SHARD", "8")
    with pytest.raises(SystemExit):
        rq3_mod.shard_spec({"run_id": "rq3", "n_shards": 4})


def test_sharded_run_gives_the_same_episodes_as_one_job(mock_runs, monkeypatch):
    one = mock_runs("one")
    rq3_mod.run_rq3(dict(CFG))
    two = mock_runs("two")
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        monkeypatch.setenv("WF_N_SHARDS", "2")
        assert rq3_mod.run_rq3(dict(CFG)) == 0
    games = [set(json.loads((shard_dir(two, i, 2) / "shard.json").read_text())["games"]) for i in (0, 1)]
    assert games[0] and games[1] and not games[0] & games[1]
    split = sorted(x for i in (0, 1) for x in lines(shard_dir(two, i, 2) / "episodes.jsonl", drop=("shard",)))
    assert split == lines(shard_dir(one) / "episodes.jsonl", drop=("shard",))

    assert rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"}) == 0
    metrics = json.loads((two / "rq3_analysis" / "metrics.json").read_text())
    mock_runs("one")
    rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"})
    alone = json.loads((one / "rq3_analysis" / "metrics.json").read_text())
    for part in ("primary", "all_units", "cells", "baselines", "cost_per_error_primary"):
        assert metrics[part] == alone[part]


def test_analysis_waits_for_every_shard(mock_runs, monkeypatch):
    root = mock_runs("run")
    cfg = {**CFG, "max_episodes_per_job": 3}
    monkeypatch.setenv("WF_N_SHARDS", "2")
    monkeypatch.setenv("WF_SHARD", "0")
    rq3_mod.run_rq3(dict(cfg))
    resume_until_done(cfg)
    assert rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"}) == 1
    status = (root / "rq3_analysis" / "STATUS.md").read_text()
    assert "FAIL" in status.splitlines()[0] and "expected 0..1" in status
    assert not (root / "rq3_analysis" / "metrics.json").exists()

    monkeypatch.setenv("WF_SHARD", "1")
    assert rq3_mod.run_rq3(dict(cfg)) == 1  # stopped part-way
    assert rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"}) == 1
    assert "shard_01_of_02 is not complete" in (root / "rq3_analysis" / "STATUS.md").read_text()

    resume_until_done(cfg)
    assert rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"}) == 0


def test_analysis_refuses_shards_from_two_different_splits(mock_runs, monkeypatch):
    root = mock_runs("run")
    rq3_mod.run_rq3(dict(CFG))
    monkeypatch.setenv("WF_N_SHARDS", "2")
    monkeypatch.setenv("WF_SHARD", "0")
    rq3_mod.run_rq3(dict(CFG))
    assert rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"}) == 1
    assert "different splits" in (root / "rq3_analysis" / "STATUS.md").read_text()


def test_analysis_refuses_shards_from_different_commits(mock_runs, monkeypatch):
    root = mock_runs("run")
    monkeypatch.setenv("WF_N_SHARDS", "2")
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        rq3_mod.run_rq3(dict(CFG))
    (shard_dir(root, 1, 2) / "git_sha").write_text("another-commit\n")
    assert rq3_mod.run_rq3_analysis({"run_id": "rq3_analysis", "source": "rq3"}) == 1
    assert "different commits" in (root / "rq3_analysis" / "STATUS.md").read_text()


# --- per-job limits, kills and resume ------------------------------------------------------------


@pytest.mark.parametrize("limit", [{"max_episodes_per_job": 5}, {"max_games_per_job": 1}])
def test_a_job_stops_after_its_limit_and_resume_matches_one_clean_job(mock_runs, limit):
    clean = mock_runs("clean")
    rq3_mod.run_rq3(dict(CFG))
    stop = mock_runs("stop")
    cfg = {**CFG, **limit}
    assert rq3_mod.run_rq3(dict(cfg)) == 1
    status = (shard_dir(stop) / "STATUS.md").read_text()
    assert "PARTIAL" in status and "WF_RESUME=1" in status
    first = read_jsonl(shard_dir(stop) / "episodes.jsonl")
    assert len(first) == (5 if "max_episodes_per_job" in limit else 16)
    jobs = 1 + resume_until_done(cfg)
    assert jobs == (13 if "max_episodes_per_job" in limit else 4)  # 64 episodes, or 4 games
    for name in ("episodes.jsonl", "games_done.jsonl"):
        assert lines(shard_dir(stop) / name) == lines(shard_dir(clean) / name)
    assert (shard_dir(stop) / "STATUS.md").read_text() == (shard_dir(clean) / "STATUS.md").read_text()


def test_a_false_failure_resumed_in_a_later_job_uses_its_saved_partner(mock_runs):
    root = mock_runs("run")
    cfg = {**CFG, "n_games": 1, "max_episodes_per_job": 5}  # 4 baselines and the first FS episode
    rq3_mod.run_rq3(dict(cfg))
    saved = read_jsonl(shard_dir(root) / "episodes.jsonl")
    assert saved[-1]["arm"] == "favorable"
    rq3_mod.run_rq3({**cfg, "resume": True})
    ff = read_jsonl(shard_dir(root) / "episodes.jsonl")[5]
    assert ff["arm"] == "unfavorable" and ff["planned_errors"] == saved[-1]["delivered_errors"]


def test_a_kill_keeps_finished_episodes_and_resume_matches_a_clean_run(mock_runs, monkeypatch):
    clean = mock_runs("clean")
    rq3_mod.run_rq3(dict(CFG))
    crash = mock_runs("crash")
    real = rq3_mod.run_false_failure
    calls = [0]

    def dies(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 9:  # mid-way through the second game
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(rq3_mod, "run_false_failure", dies)
    with pytest.raises(KeyboardInterrupt):
        rq3_mod.run_rq3(dict(CFG))
    monkeypatch.setattr(rq3_mod, "run_false_failure", real)
    # game 1, then game 2's baselines, two FS/FF pairs and the partner of the killed FF episode
    assert len(read_jsonl(shard_dir(crash) / "episodes.jsonl")) == 16 + 4 + 2 * 2 + 1
    assert len(read_jsonl(shard_dir(crash) / "games_done.jsonl")) == 1
    rq3_mod.run_rq3({**CFG, "resume": True})
    for name in ("episodes.jsonl", "games_done.jsonl"):
        assert lines(shard_dir(crash) / name) == lines(shard_dir(clean) / name)


def test_a_finished_shard_runs_nothing_on_resume(mock_runs, monkeypatch):
    mock_runs("run")
    rq3_mod.run_rq3(dict(CFG))

    def boom(*args, **kwargs):
        raise AssertionError("an episode ran again")

    for name in ("run_episode", "run_false_success", "run_false_failure"):
        monkeypatch.setattr(rq3_mod, name, boom)
    assert rq3_mod.run_rq3({**CFG, "resume": True}) == 0


def test_resume_refuses_changed_settings(mock_runs):
    mock_runs("run")
    rq3_mod.run_rq3({**CFG, "max_episodes_per_job": 5})
    with pytest.raises(SystemExit, match="settings changed"):
        rq3_mod.run_rq3({**CFG, "max_episodes_per_job": 5, "fs_probability": 1.0, "resume": True})


# --- configs, CLI and job script -----------------------------------------------------------------


def test_confirmatory_config_matches_the_decision():
    cfg = yaml.safe_load((REPO / "wf" / "config" / "rq3.yaml").read_text())
    assert cfg["split"] == "valid_unseen" and cfg["n_games"] == 134 and cfg["step_cap"] == C.STEP_CAP
    assert len(set(cfg["schedule_seeds"])) == len(cfg["schedule_seeds"]) == C.N_SCHEDULES
    assert cfg["fs_probability"] == 0.75 and cfg["fs_cap"] == C.CORRUPTIONS_PER_EPISODE_TARGET
    assert cfg["max_episodes_per_job"] > 0
    smoke = yaml.safe_load((REPO / "wf" / "config" / "rq3_smoke.yaml").read_text())
    assert smoke["split"] == "train" and smoke["n_games"] == 2
    assert {k: v for k, v in smoke.items() if k not in ("run_id", "split", "n_games", "n_shards")} == {
        k: v for k, v in cfg.items() if k not in ("run_id", "split", "n_games", "n_shards", "max_episodes_per_job")}
    for name, source in (("rq3_analysis", "rq3"), ("rq3_smoke_analysis", "rq3_smoke")):
        assert yaml.safe_load((REPO / "wf" / "config" / f"{name}.yaml").read_text())["source"] == source


def test_the_cli_runs_the_smoke_config_and_its_analysis(mock_runs):
    root = mock_runs("run")
    for run_id in ("rq3_smoke", "rq3_smoke_analysis"):
        with pytest.raises(SystemExit) as done:
            runs_mod.main([run_id])
        assert done.value.code == 0
    assert len(read_jsonl(shard_dir(root, base="rq3_smoke") / "episodes.jsonl")) == 2 * 16
    assert "REPORTED (RQ3 interaction)" in (root / "rq3_smoke_analysis" / "STATUS.md").read_text()


def test_the_job_script_takes_its_shard_and_resumes():
    script = (REPO / "pace" / "rq3.sbatch").read_text()
    assert "#SBATCH --no-requeue" in script and "start_alfworld_server" in script and "start_vllm" in script
    assert 'RUN="${RQ3_RUN:-rq3}"' in script and 'run_wf "${RUN}"' in script and "WF_SHARD" in script
    # the game server restarts inside the job whenever the run stops with PARTIAL
    loop = script.split("while true")[1]
    assert "start_alfworld_server" in loop and "PARTIAL" in loop and "WF_RESUME=1" in loop
    assert "FILED.txt" in script


def test_a_false_failure_with_nothing_to_yoke_reuses_the_baseline_and_says_so():
    from wf import rq3
    baseline = {"game_id": "g", "condition": "oracle", "arm": None, "retention": "retained", "schedule_seed": None,
                "steps": 3, "success": True, "trace": [{"executed": "a"}]}
    partner = {"game_id": "g", "condition": "corrupted", "arm": "favorable", "retention": "retained",
               "schedule_seed": 7, "delivered_errors": 0, "cap": 3, "injections": []}
    saved = {("g", "oracle", None, "retained", None): baseline, ("g", "corrupted", "favorable", "retained", 7): partner}
    ep = rq3._run_job(("g", "corrupted", "unfavorable", "retained", 7), None, None, "train",
                      {"run_id": "x", "copy_unyoked_ff": True}, saved, {})
    assert ep["copied_from_baseline"] and ep["trace"] == baseline["trace"] and ep["delivered_errors"] == 0
    assert rq3.episode_key(ep) == ("g", "corrupted", "unfavorable", "retained", 7)
