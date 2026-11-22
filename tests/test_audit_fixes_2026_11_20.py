"""Fixes and new runs from the 2026-11-20 documentation and rigor audit (notes/deviations.md).

B1 outcome selection in the paired effects; B2 strict task-line replacement; B3 the thinking mock flag; B4 NaN
in the size flips and slope; B5 submit.sh and pace.env.example; B6 provenance and the golden prompt; B8 the
collector's exact hidden-state check; A5 pinned model revisions; A2 the shared-spot sensitivity; C the audit
analyses.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import wf.runs as runs_mod
from wf import audit, constants as C, rq1, rq1v, rq1x, rq2, rq2_shared, sizes, thinking
from wf.analysis.stats import paired_conditions
from wf.provenance import provenance, run_record
from wf.scoring import logodds
from wf.scoring.logodds import LogoddsScorer, ScoreResult, build_messages
from wf.scoring.prompts import _replace_task_line
from wf.util import file_sha, read_jsonl
from tests.test_rq1 import _score_cfg, mock_root, mock_score  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOOP = "Nothing happens."


def _t(i, failed=True, **kw):
    t = {"id": f"t{i}", "game_id": f"g{i % 4}", "context_determined": True, "history_goal_agnostic": True,
         "true_obs": NOOP if failed else "You pick up the mug 1 from the shelf 1."}
    return {**t, **kw}


def _s(tid, cond, L, model="m"):
    return {"transition_id": tid, "condition": cond, "L": L, "L_first": L / 2, "c_hat": -L / 2, "model": model,
            "first_mass": 0.9, "p_success": 1 / (1 + math.exp(-L))}


# ---------------- B1 ----------------


def test_paired_conditions_selects_by_true_outcome():
    items = [_t(0), _t(1), _t(2, failed=False), _t(3, failed=False)]
    scores = [_s(t["id"], c, v) for t in items for c, v in (("aligned", 2.0), ("swapped", 1.0))]
    ids = lambda pairs: len(pairs)
    assert ids(paired_conditions(scores, items)) == 2  # the default is unchanged: failures only
    assert ids(paired_conditions(scores, items, outcome="success")) == 2
    assert ids(paired_conditions(scores, items, outcome="any")) == 4
    with pytest.raises(ValueError):
        paired_conditions(scores, items, outcome="successes")


def test_goal_effect_on_successes_is_computed_not_empty():
    """The logged readout that came out NaN with n = 0 (rq1x working_actions, rq1v yes_no)."""
    succ = [_t(i, failed=False) for i in range(8)]
    scores = [_s(t["id"], c, v) for t in succ for c, v in (("aligned", 3.0), ("swapped", 1.0))]
    old = runs_mod._effect(scores, succ, require_agnostic=False)
    assert old["n_items"] == 0  # what the analyses silently reported before the fix
    new = runs_mod._effect(scores, succ, require_agnostic=False, outcome="success")
    assert new["n_items"] == 8 and new["delta_L_nats"] == pytest.approx(2.0)


def test_rq1x_and_rq1v_report_the_goal_effect_on_successes(mock_root, monkeypatch):
    assert rq1.run_rq1_score(_score_cfg(shard=0, n_shards=1)) == 0
    cfg = {"run_id": "rq1x_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 1}
    assert rq1x.run_rq1x_score({**cfg, "shard": 0}) == 0
    acfg = {"run_id": "rq1x_smoke_analysis", "source": "rq1x_smoke", "base": "rq1_smoke", "sources": ["rq12_transitions_smoke"]}
    assert rq1x.run_rq1x_analysis(acfg) == 0
    m = json.loads((mock_root / "rq1x_smoke_analysis" / "metrics.json").read_text())
    n_succ = m["items"]["n_successes"]
    assert n_succ > 0
    for tag in rq1.FEWSHOT:
        assert m[tag]["working_actions"]["goal_effect_on_successes"]["n_items"] == n_succ
    assert m["provenance"]["runs_read"]["rq1_smoke"]["shards"]
    vcfg = {"run_id": "rq1v_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 1}
    assert rq1v.run_rq1v_score({**vcfg, "shard": 0}) == 0
    assert rq1v.run_rq1v_analysis({"run_id": "rq1v_smoke_analysis", "source": "rq1v_smoke",
                                    "sources": ["rq12_transitions_smoke"]}) == 0
    v = json.loads((mock_root / "rq1v_smoke_analysis" / "metrics.json").read_text())
    for tag in rq1.FEWSHOT:
        assert v[tag]["yes_no"]["goal_effect_successes"]["n_items"] == v["items"]["n_successes"] > 0


# ---------------- B2 ----------------


def test_task_line_replacement_needs_exactly_one_line():
    hist = "You are in the middle of a room.\nYour task is to: put a mug in shelf.\n> go to shelf 1\nYou arrive."
    out = _replace_task_line(hist, "Your task is to: put a cup in shelf.")
    assert out == hist.replace("put a mug", "put a cup")
    with pytest.raises(ValueError):
        _replace_task_line("You are in the middle of a room.", "Your task is to: x.")  # used to prepend silently
    with pytest.raises(ValueError):
        _replace_task_line(hist + "\nYour task is to: put a mug in shelf.", "Your task is to: x.")


def test_every_prompt_path_still_builds_on_a_real_item(mock_score):
    item = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())["item"]
    item = {**item, "scene_receptacles": ["cabinet 1", "countertop 1", "garbagecan 1", "toilet 1"],
            "scene_locations": {"soapbar 1": ["countertop 1"]}, "determination_reason": "contents_or_open_state",
            "hidden_state": None, "state_idx": 1, "prefix_actions": ["go to cabinet 1"]}
    for arm in rq1.ARMS:
        if "full" not in arm:
            rq1.arm_messages(item, arm)
    for arm in rq1v.YN_ARMS + ("aligned", "swapped"):
        rq1v.arm_messages(item, arm)
    from wf import mech
    for arm in mech.PRED_ARMS + mech.POLICY_ARMS:
        mech.arm_messages(item, arm)


# ---------------- B3 ----------------


def test_thinking_mock_flag_zero_means_a_real_run(mock_root, monkeypatch):
    class Real(Exception):
        pass

    def real_engine(*a, **k):
        raise Real()

    monkeypatch.setattr(thinking, "_Engine", real_engine)
    monkeypatch.setenv("WF_MOCK_SCORE", "0")  # "0" used to count as mock (bool of a non-empty string)
    with pytest.raises(Real):
        thinking.run_thinking({"run_id": "thinking_smoke", "sources": ["rq12_transitions_smoke"], "n_items": 3})


def test_thinking_analysis_can_reread_answers_with_the_current_parser(mock_root):
    cfg = {"run_id": "thinking_smoke", "sources": ["rq12_transitions_smoke"], "n_items": 10, "chunk": 5}
    assert thinking.run_thinking(cfg) == 0
    acfg = {"run_id": "thinking_smoke_analysis", "source": "thinking_smoke", "sources": ["rq12_transitions_smoke"],
            "n_items": 10, "reparse": True}
    assert thinking.run_thinking_analysis(acfg) == 0
    m = json.loads((mock_root / "thinking_smoke_analysis" / "metrics.json").read_text())
    from wf.tools.parser_audit import parser_version
    assert m["answers_read_with"]["parser_version"] == parser_version()
    assert m["provenance"]["parser_version"] == parser_version()


# ---------------- B4 ----------------


def test_flips_leave_nan_out_of_the_denominator():
    items = [{"id": f"t{i}", "game_id": "g"} for i in range(4)]
    s = [{"transition_id": "t0", "condition": "aligned", "L": 1.0}, {"transition_id": "t0", "condition": "swapped", "L": -1.0},
         {"transition_id": "t1", "condition": "aligned", "L": float("nan")}, {"transition_id": "t1", "condition": "swapped", "L": -1.0}]
    assert sizes._flips(s, items) == 1.0  # one valid pair, and it flips


def test_size_slope_drops_resamples_with_a_missing_size():
    tags = ["qwen3-0.6b", "qwen3-4b", "qwen3-32b"]
    per = {m: {f"g{k}": [math.log(sizes.SIZES[m][1]) + 0.1 * k] for k in range(6)} for m in tags}
    per["qwen3-32b"] = {"g0": [3.0]}  # only one game, so many resamples lack it
    out = sizes.size_slope(per, B=200)
    assert out["n_resamples_dropped_nan"] > 0
    assert all(x == x for x in out["ci"]) and out["ci"][0] <= out["ci"][1]


# ---------------- B5 ----------------


def _submit(tmp_path: Path, script: str, env: dict) -> list[str]:
    pace = tmp_path / "pace"
    pace.mkdir()
    shutil.copy(ROOT / "pace" / "submit.sh", pace / "submit.sh")
    (pace / "_common.sh").write_text(f'export PROJECT_ROOT="{tmp_path}"\n')  # the real one would read pace.env
    (pace / "job.sbatch").write_text(script)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "sbatch").write_text('#!/bin/sh\nfor a in "$@"; do echo "$a"; done\n')
    (bin_ / "sbatch").chmod(0o755)
    run_env = {"PATH": f"{bin_}:{os.environ['PATH']}", **env}
    out = subprocess.run(["bash", str(pace / "submit.sh"), "job.sbatch", "--export=ALL,X=1"], capture_output=True,
                         text=True, env=run_env, check=True)
    return out.stdout.split()


def test_submit_keeps_a_scripts_own_gpu_request_and_constraint(tmp_path):
    gpu = "#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --constraint=gpu-a100-80gb\n"
    args = _submit(tmp_path, gpu, {"PACE_GRES": "gpu:A100:1", "PACE_PARTITION": "coc-gpu", "PACE_ACCOUNT": "a"})
    assert not any(a.startswith("--gres") or a.startswith("--constraint") for a in args)
    assert "--partition=coc-gpu" in args and "--account=a" in args and "--export=ALL,X=1" in args


def test_submit_gives_cpu_scripts_no_gpu(tmp_path):
    args = _submit(tmp_path, "#!/bin/bash\n#SBATCH --partition=coc-cpu\n", {"PACE_GRES": "gpu:1", "PACE_PARTITION": "coc-gpu"})
    assert not any(a.startswith("--gres") or a.startswith("--partition") for a in args)


def test_env_example_asks_for_any_gpu():
    text = (ROOT / "pace" / "pace.env.example").read_text()
    assert 'PACE_GRES="gpu:1"' in text and "A100:1\"" not in text.split("PACE_GRES=")[1].splitlines()[0]


@pytest.mark.parametrize("path", sorted(p.name for p in (ROOT / "pace").iterdir() if p.suffix in (".sh", ".sbatch")))
def test_pace_scripts_parse(path):
    assert subprocess.run(["bash", "-n", str(ROOT / "pace" / path)], capture_output=True).returncode == 0


# ---------------- B6 ----------------


def test_provenance_records_commits_hashes_and_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    shard = tmp_path / "run" / "shard_00_of_01"
    shard.mkdir(parents=True)
    (shard / "git_sha").write_text("abc123\n")
    (shard / "STATUS.md").write_text("# STATUS — run/shard_00_of_01 — REPORTED (shard complete)\n")
    (shard / "scoring.jsonl").write_text('{"L": 1}\n')
    (shard / "jobs.jsonl").write_text('{"gpu": ["NVIDIA A40"], "slurm_job_id": "7"}\n')
    block = provenance(runs={"run": tmp_path / "run"}, inputs=[shard / "scoring.jsonl"], parser=True)
    rec = block["runs_read"]["run"]["shards"]["shard_00_of_01"]
    assert rec["commit"] == "abc123" and rec["gpu"] == ["NVIDIA A40"] and "shard complete" in rec["status"]
    assert block["input_sha256"][str(shard / "scoring.jsonl")] == file_sha(shard / "scoring.jsonl")
    assert block["analysis_commit"] and "parser_version" in block
    assert run_record(tmp_path)["gpu"] == "not recorded"


def test_rq2_analysis_metrics_carry_provenance_and_shards_record_their_job(mock_root):
    cfg = {"run_id": "rq2_smoke", "sources": ["rq12_transitions_smoke"], "models": list(rq2.CO_PRIMARY),
           "mention": False, "generations": False, "n_shards": 1, "shard": 0}
    assert rq2.run_rq2_score(cfg) == 0
    shard = mock_root / "rq2_smoke" / "shard_00_of_01"
    assert read_jsonl(shard / "jobs.jsonl")[0]["host"]
    assert rq2.run_rq2_analysis({"run_id": "rq2_smoke_analysis", "source": "rq2_smoke",
                                 "sources": ["rq12_transitions_smoke"], "min_items": 1, "min_states": 1}) in (0, 1)
    m = json.loads((mock_root / "rq2_smoke_analysis" / "metrics.json").read_text())
    prov = m["provenance"]
    assert prov["runs_read"]["rq2_smoke"]["shards"]["shard_00_of_01"]["commit"]
    assert prov["input_sha256"][str(shard / "scoring.jsonl")] == file_sha(shard / "scoring.jsonl")
    assert str(mock_root / "rq12_transitions_smoke" / "transitions.jsonl") in prov["input_sha256"]


def test_golden_default_prompt_message_list(mock_score):
    """The exact default prompt (system, the three set-1 examples, the user turn) for one real item, both goals.

    A change to any prompt text, example or the task-line swap shows up here as a diff. Regenerate the golden file
    only on purpose, and record why in notes/deviations.md.
    """
    golden = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())
    item = golden["item"]
    assert build_messages(item, "aligned") == golden["aligned"]
    assert build_messages(item, "swapped") == golden["swapped"]
    a, s = golden["aligned"], golden["swapped"]
    assert len(a) == 8 and a[0]["role"] == "system" and [m["role"] for m in a[1:]] == ["user", "assistant"] * 3 + ["user"]
    diff = [(x["content"], y["content"]) for x, y in zip(a, s) if x != y]
    assert diff == [(a[-1]["content"], a[-1]["content"].replace(item["task_line"], item["swapped_task_line"]))]


def _expected_render(messages: list[dict], family: str) -> str:
    """The chat template's output as documented for each family, built by hand from the message list."""
    if family == "qwen3":
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    head = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
            f"Today Date: 26 Jul 2024\n\n{messages[0]['content'].strip()}<|eot_id|>")
    body = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>" for m in messages[1:])
    return head + body + "<|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.mark.parametrize("model, family", [(C.WM_FEWSHOT_MODEL, "qwen3"), (C.WM_FEWSHOT_LLAMA, "llama3")])
def test_golden_prompt_as_rendered_by_the_pinned_chat_template(model, family):
    """Runs where transformers and the pinned tokenizer are available offline (ICE); skipped elsewhere."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(model, revision=C.REVISIONS[model], local_files_only=True)
    except Exception as e:  # not cached here
        pytest.skip(f"{model} tokenizer not available offline: {e}")
    golden = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())
    for cond in ("aligned", "swapped"):
        text = tok.apply_chat_template(golden[cond], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert text == _expected_render(golden[cond], family)


# ---------------- B8 ----------------


def test_collector_hidden_state_check_matches_instances_exactly(monkeypatch):
    for k in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.delenv(k, raising=False)
    hidden = "=== Objects on Receptacles ===\n - On the shelf 1, you see a mug 10, and a cup 1.\n"
    bad = {"game_id": "g1", "hidden_state": hidden, "scene_locations": {"mug 1": ["shelf 1"]}}
    good = {"game_id": "g2", "hidden_state": hidden, "scene_locations": {"mug 10": ["shelf 1"], "cup 1": ["shelf 1"]}}
    assert runs_mod._hidden_state_mismatches([bad, good]) == ["g1"]  # "mug 1" is not "mug 10"


# ---------------- A5 ----------------


def test_every_model_the_code_loads_is_pinned():
    ids = {m for m, _, _ in rq1.MODELS} | {m for m, _ in sizes.SIZES.values()} | {C.AGENT_MODEL}
    assert ids <= set(C.REVISIONS) and all(len(v) == 40 for v in C.REVISIONS.values())
    env_model = [ln for ln in (ROOT / "pace" / "pace.env.example").read_text().splitlines() if "VLLM_MODEL=" in ln][0]
    assert env_model.split('"')[1] in C.REVISIONS


def test_unpinned_model_is_refused_unless_allowed(monkeypatch):
    monkeypatch.delenv("WF_ALLOW_UNPINNED", raising=False)
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    with pytest.raises(KeyError):
        C.revision_for("someone/unknown-model")
    with pytest.raises(KeyError):  # refused before any weights would load
        LogoddsScorer("someone/unknown-model")
    monkeypatch.setenv("WF_ALLOW_UNPINNED", "1")
    assert C.revision_for("someone/unknown-model") is None
    assert C.revision_for("Qwen/Qwen3-8B") == "b968826d9c46dd6066d109eabc6255188de91218"


def test_scores_record_the_model_revision(mock_score):
    sc = LogoddsScorer(C.WM_FEWSHOT_MODEL, model_tag="q").score_pair("t", "aligned", [{"role": "user", "content": "x"}],
                                                                     "You pick up.", NOOP)
    assert sc.to_dict()["model_revision"] == C.REVISIONS[C.WM_FEWSHOT_MODEL]
    assert ScoreResult("t", "m", "c", 0, 0, 0, 0.5, 1, 1).to_dict()["model_revision"] is None


# ---------------- A2 ----------------


def _shared_fixture(monkeypatch, tmp_path, mark: int):
    """A mock collection with `mark` primary items flagged as at a shared spot, then verified."""
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    logodds._fewshot_hidden.cache_clear()
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    assert runs_mod.run_pilotA_transitions({"run_id": "rq12_transitions_smoke", "split": "train", "n_games": 6,
                                            "step_cap": 12, "min_items": 1, "min_states": 1}) == 0
    path = tmp_path / "rq12_transitions_smoke" / "transitions.jsonl"
    items = read_jsonl(path)
    # the simulator has shared spots of its own; only the flags chosen here remain on otherwise-primary items
    candidates = {t["id"] for t in runs_mod._pilotA_item_sets(items)["agnostic_all"]}
    flagged = sorted(candidates)[:mark]
    path.write_text("".join(json.dumps({**t, "shared_location": t["id"] in flagged} if t["id"] in candidates else t)
                            + "\n" for t in items))
    from wf.tools.verify_items import verify_run
    assert verify_run("rq12_transitions_smoke", server=True)["passed"]
    return flagged


def test_shared_spot_items_are_those_excluded_only_by_that_rule():
    base = {"context_determined": True, "true_obs": NOOP, "candidates": {"o_plus": "x", "o_minus": NOOP},
            "swapped_goal_object": "cup", "targets_goal_object": True, "history_goal_agnostic": True}
    items = [{**base, "id": "a", "shared_location": True},
             {**base, "id": "b", "shared_location": True, "history_goal_agnostic": False},  # excluded twice
             {**base, "id": "c", "shared_location": False}]
    assert [t["id"] for t in rq2_shared.shared_spot_items(items)] == ["a"]
    assert rq2_shared.shared_flag_flow(items)["goal_agnostic_history"] == 1


def test_shared_spot_sensitivity_end_to_end(monkeypatch, tmp_path):
    flagged = _shared_fixture(monkeypatch, tmp_path, mark=2)
    src = ["rq12_transitions_smoke"]
    assert rq2.run_rq2_score({"run_id": "rq2_score", "sources": src, "models": list(rq2.CO_PRIMARY), "mention": False,
                              "generations": False, "n_shards": 1, "shard": 0}) == 0
    assert rq2_shared.run_rq2_shared_score({"run_id": "rq2_shared_score", "sources": src, "n_shards": 1}) == 0
    scored = read_jsonl(tmp_path / "rq2_shared_score" / "shard_00_of_01" / "scoring.jsonl")
    assert {s["transition_id"] for s in scored} == set(flagged) and len(scored) == 2 * 2 * len(flagged)
    assert rq2_shared.run_rq2_shared_analysis({"run_id": "rq2_shared_analysis", "source": "rq2_shared_score",
                                               "base": "rq2_score", "sources": src}) == 0
    m = json.loads((tmp_path / "rq2_shared_analysis" / "metrics.json").read_text())
    for model in rq2.CO_PRIMARY:
        r = m["per_model"][model]
        assert r["shared_spot_only"]["n_items"] == len(flagged)
        assert r["confirmatory_with_shared_spot"]["n_items"] == r["confirmatory_as_scored"]["n_items"] + len(flagged)
        assert {"delta_L_first_nats", "delta_p_pp", "delta_L_ci"} <= set(r["shared_spot_only"])


def test_shared_spot_analysis_with_no_such_item_needs_no_shared_scores(monkeypatch, tmp_path):
    _shared_fixture(monkeypatch, tmp_path, mark=0)
    src = ["rq12_transitions_smoke"]
    assert rq2_shared.run_rq2_shared_score({"run_id": "rq2_shared_score", "sources": src}) == 0  # loads no model
    assert not (tmp_path / "rq2_shared_score" / "shard_00_of_01" / "scoring.jsonl").exists()
    assert rq2.run_rq2_score({"run_id": "rq2_score", "sources": src, "models": list(rq2.CO_PRIMARY), "mention": False,
                              "generations": False, "n_shards": 1, "shard": 0}) == 0
    assert rq2_shared.run_rq2_shared_analysis({"run_id": "rq2_shared_analysis", "source": "nowhere",
                                               "base": "rq2_score", "sources": src}) == 0
    m = json.loads((tmp_path / "rq2_shared_analysis" / "metrics.json").read_text())
    assert m["items"]["n_shared_spot_only"] == 0
    for r in m["per_model"].values():
        assert r["confirmatory_with_shared_spot"]["delta_L_nats"] == r["confirmatory_as_scored"]["delta_L_nats"]


# ---------------- C ----------------


def test_exclusion_flow_follows_each_stage_to_the_primary_set():
    ok = {"game_id": "g", "true_obs": NOOP, "context_determined": True, "candidates": {"o_plus": "x"},
          "swapped_goal_object": "cup", "shared_location": False, "targets_goal_object": True, "history_goal_agnostic": True}
    items = [{**ok, "id": "p"}, {**ok, "id": "s", "true_obs": "You arrive."}, {**ok, "id": "n", "candidates": None,
             "exclusion_reason": "no_success_twin"}, {**ok, "id": "h", "history_goal_agnostic": False}]
    out = audit.exclusion_flow(items)
    assert [s["remaining"] for s in out["flow"]] == [4, 3, 3, 3, 2, 2, 2, 2, 1]
    assert out["flow"][4]["dropped_by_recorded_reason"] == {"no_success_twin": 1}
    assert out["flow_matches_primary_set"]


def test_instance_number_items_are_split_out():
    hist = ("You are in the middle of a room.\nYour task is to: put a mug in desk.\n> go to shelf 1\n"
            "You arrive at shelf 1. On the shelf 1, you see a mug 2, and a pen 1.")
    t = {"action": "take mug 1 from shelf 1", "determination_reason": "contents_or_open_state", "history_text": hist}
    assert audit.seen_without(t) == (True, ["mug 2"])
    assert audit.seen_without({**t, "action": "take cup 1 from shelf 1"}) == (True, [])
    assert audit.seen_without({**t, "action": "take mug 2 from shelf 1"}) == (False, [])


def test_twin_plan_wins_only_for_a_take_from_the_put_tasks_target():
    t = {"true_obs": NOOP, "exclusion_reason": "no_success_twin", "task_line": "Your task is to: put a mug in desk.",
         "action": "take mug 1 from desk 1", "scene_locations": {"mug 1": ["shelf 1"]}, "scene_receptacles": ["desk 1", "shelf 1"]}
    assert audit.twin_plan_wins(t)
    assert not audit.twin_plan_wins({**t, "task_line": "Your task is to: heat some mug and put it in desk."})
    assert not audit.twin_plan_wins({**t, "action": "take mug 1 from shelf 2"})
    assert not audit.twin_plan_wins({**t, "scene_locations": {"mug 1": ["desk 1"]}})


def test_top_candidate_ties_and_near_duplicates():
    cands = [{"id": "a", "game_id": "g", "state_idx": 0, "true_obs": NOOP},
             {"id": "b", "game_id": "g", "state_idx": 0, "true_obs": "You pick up the mug 1."}]
    assert audit.top_candidate_ties({("a", "x"): 1.0, ("b", "x"): 1.0}, cands, "x") == {
        "n_states": 1, "top_score_tied": 1, "tie_between_working_and_failing": 1}
    hist = "You are in the middle of a room.\nYour task is to: put a mug in desk.\n> go to shelf 1\nYou arrive at shelf 1."
    loop = hist + "\n> go to shelf 1\nNothing happens."
    items = [{"id": "1", "game_id": "g", "action": "take mug 1 from desk 1", "history_text": hist},
             {"id": "2", "game_id": "g", "action": "take mug 1 from desk 1", "history_text": loop}]
    nd = audit.items_per_game(items)["near_duplicates"]
    assert nd["groups"] == 1 and nd["extra_items"] == 1


def test_audit_analysis_end_to_end(mock_root):
    for i in range(2):
        assert rq1.run_rq1_score(_score_cfg(shard=i)) == 0
    cfg = {"run_id": "audit_analysis", "sources": ["rq12_transitions_smoke"],
           "inputs": {"rq1_score": "rq1_smoke", "rq2_score": "missing_run",
                      "size_scores": {"qwen3-0.6b": str(mock_root / "size_score_qwen3-0.6b")}}}
    sizes.run_size_score({"run_id": "size_score_qwen3-0.6b", "model": "qwen3-0.6b", "sources": ["rq12_transitions_smoke"],
                          "n_shards": 1})
    assert audit.run_audit_analysis(cfg) == 0
    m = json.loads((mock_root / "audit_analysis" / "metrics.json").read_text())
    assert m["exclusion_flow"]["flow_matches_primary_set"]
    assert {"ratios", "size_yes_share", "first_token", "items_per_game", "twin_plan_wins"} <= set(m)
    assert "rq2_score" in m["not_computed"] and "seen_without_object" not in m
    assert "share_yes" in m["size_yes_share"]["qwen3-0.6b"]["failures"]["yn_aligned"]
    assert m["provenance"]["input_sha256"]


def test_mech_share_of_effect_removed_is_one_minus_the_ratio():
    from wf import mech
    items = [{"id": f"t{i}", "game_id": f"g{i % 3}", "context_determined": True, "true_obs": NOOP,
              "history_goal_agnostic": True, "determination_reason": "away_from_receptacle"} for i in range(6)]
    arms = {"pred_plain_aligned": 5.0, "pred_plain_swapped": 1.0, "pred_facts_aligned": 2.0, "pred_facts_swapped": 1.0}
    scores = [_s(t["id"], a, v) for t in items for a, v in arms.items()]
    out = mech._model_block(scores, items)
    assert out["interventions"]["facts"]["share_of_effect_removed"]["estimate"] == pytest.approx(0.75)
    assert out["interventions"]["plain"]["goal_effect"]["delta_L_nats"] == pytest.approx(4.0)


def test_analysis_refuses_shards_from_different_commits(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for i, sha in enumerate(("aaa", "bbb")):
        p = tmp_path / "src" / f"shard_{i:02d}_of_02"
        p.mkdir(parents=True)
        (p / "STATUS.md").write_text(f"# STATUS — src/shard_{i:02d}_of_02 — REPORTED (shard complete)\n")
        (p / "git_sha").write_text(sha + "\n")
    _, problems = rq2.complete_shards("src")
    assert any("different commits" in p for p in problems)


def test_new_runs_are_registered_with_configs():
    for name in ("rq2_shared_score", "rq2_shared_analysis", "audit_analysis", "rq1_analysis_v2", "rq1x_analysis_v2",
                 "rq1v_analysis_v2", "thinking_analysis_v2", "size_analysis_v2"):
        assert name in runs_mod.RUNNERS and (ROOT / "wf" / "config" / f"{name}.yaml").exists()
    assert runs_mod.load_config("thinking_analysis_v2")["reparse"] is True
