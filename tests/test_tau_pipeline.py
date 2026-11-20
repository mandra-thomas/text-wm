"""τ-bench runs (wf/tau/pipeline.py) end to end on mock scoring: build -> verify -> score -> analysis, the gates
that refuse unverified or changed items, resume, the two-sided decision rule and the contamination helpers."""
from __future__ import annotations

import json
import math
import re

import pytest

from wf.runs import RUNNERS, load_config
from wf.tau import pipeline as P
from wf.util import read_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A built and verified small item set in a temp runs folder (mock mode)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("tau_runs")
    for k in MOCK_KEYS:
        mp.setenv(k, "1")
    mp.setenv("RUNS_DIR", str(root))
    mp.setenv("WF_MOCK_CONTINUE", "1")
    (root / "FILED.txt").write_text("filed_date: x\n")
    assert P.run_tau_build({"run_id": "tau_build", "natural_required": False, "max_users": 15}) == 0
    assert P.run_tau_verify({"run_id": "tau_verify", "fresh_sample": 5}) == 0
    yield root
    mp.undo()


def _small_pins():
    d = P.runs_root() / "tau_build"
    return {n: P.file_sha(d / n) for n in P.OUTPUT_FILES}


def _score_cfg(**kw):
    """The real scoring config, pinned to this small build (the config's own pins are the full build's). Its
    exploratory arms (500 items each) take every primary item of this small build."""
    return {**load_config("tau_score"), "n_shards": 2, "expected_outputs_sha256": _small_pins(), **kw}


def test_scoring_configs_pin_every_item_file_alike():
    pins = [load_config(n).get("expected_outputs_sha256") for n in ("tau_score", "tau_score_smoke", "tau_contamination")]
    assert all(p == pins[0] for p in pins)
    assert set(pins[0]) == set(P.OUTPUT_FILES) and all(len(v) == 64 for v in pins[0].values())
    full = load_config("tau_score")
    assert full["witness"] and full["mention"] and full["target_check"] and full["rules_arm_items"] == 500
    assert full["matched_arm_items"] == 500
    smoke = load_config("tau_score_smoke")
    assert smoke["add_longest"] and smoke["limit_items"] == 20 and smoke["matched_arm_items"] == smoke["rules_arm_items"] == 20


def test_runs_are_registered():
    for name in ("tau_build", "tau_verify", "tau_score", "tau_score_smoke", "tau_analysis", "tau_smoke_analysis", "tau_contamination"):
        assert name in RUNNERS


def test_build_writes_items_manifest_and_strata(runs):
    d = runs / "tau_build"
    gen = read_jsonl(d / "items.jsonl")
    man = json.loads((d / "manifest.json").read_text())
    assert gen and man["outputs"]["items.jsonl"] == P.file_sha(d / "items.jsonl")
    assert man["strata_generated"]["all"]["items"] == len(gen)
    assert man["vendor"]["commit"].startswith("c06b07d")
    assert "REPORTED (items built)" in (d / "STATUS.md").read_text()
    v = json.loads((runs / "tau_verify" / "verify.json").read_text())
    assert v["passed"] and v["n_failed_items"] == 0 and v["fresh_process"]["generated"]["passed"]


def test_score_then_analysis(runs, monkeypatch):
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        assert P.run_tau_score(_score_cfg()) == 0
    monkeypatch.delenv("WF_SHARD")
    main = [t for t in read_jsonl(runs / "tau_build" / "items.jsonl") if t["main_pool"]]
    recs = [r for i in (0, 1) for r in read_jsonl(runs / "tau_score" / f"shard_{i:02d}_of_02" / "scoring.jsonl")]
    # every main item x model x job, once; shards split by user. Per item: failing call 3 conditions x (first, yn,
    # full) = 9, mention pair x (first, yn) = 4, manipulation check 2, witness 9; the rules-only and the matched arm
    # 6 each on every primary item here (fewer than their 500), and the matched arm's witness 3 (first token, fix round 3).
    assert all(t["task_swapped"] and t["aligned_witness"] for t in main)
    n_primary = sum(t["status_matched"] for t in main)
    assert 0 < n_primary < 500
    assert len(recs) == 2 * (len(main) * (9 + 4 + 2 + 9) + n_primary * 15)
    keys = [P.score_key(r) for r in recs]
    assert len(keys) == len(set(keys)) and all(r["seconds"] >= 0 for r in recs)
    assert {(r["arm"], r["action"], r["readout"]) for r in recs} == {
        ("main", "fail", "first"), ("main", "fail", "yn"), ("main", "fail", "full"), ("main", "fail", "target"),
        ("main", "witness", "first"), ("main", "witness", "yn"), ("main", "witness", "full"),
        ("rules_only", "fail", "first"), ("rules_only", "fail", "yn"), ("matched", "fail", "first"), ("matched", "fail", "yn"),
        ("matched", "witness", "first")}
    users = [{r["user_id"] for r in read_jsonl(runs / "tau_score" / f"shard_{i:02d}_of_02" / "scoring.jsonl")} for i in (0, 1)]
    assert not users[0] & users[1]
    assert P.run_tau_analysis(load_config("tau_analysis")) == 0
    m = json.loads((runs / "tau_analysis" / "metrics.json").read_text())
    assert m["missing_scores"] == 0 and m["stale_scores"] == 0 and m["items"]["main"] == len(main)
    primary = [t for t in main if t["status_matched"]]
    assert m["items"]["primary"] == len(primary) > 0
    q = m["models"]["qwen3-8b-fewshot"]
    blk = q["primary"]["first"]["aligned_minus_swapped"]
    assert blk["n_items"] == len(primary) and blk["n_users"] == len({t["user_id"] for t in primary})
    assert q["main_pool"]["first"]["aligned_minus_swapped"]["n_items"] == len(main)
    assert q["primary_controls"]["manipulation"]["n_items"] == len(primary)
    assert q["primary_controls"]["witness"]["first"]["aligned_minus_swapped"]["n_items"] == len(primary)
    assert q["primary_controls"]["discrimination"]["first"]["n_items"] == sum(t["witness_checker"] == "PASS" for t in primary)
    assert q["primary_controls"]["mention"]["first"]["mention_aligned_minus_mention_swapped"]["n_items"] == len(primary)
    assert q["primary"]["first_product_clustered"]["cluster"] == "product"
    assert q["primary_controls"]["fail_minus_witness_effect"]["first"]["n_items"] == len(primary)
    assert q["primary_controls"]["write_minus_mention_effect"]["yn"]["n_items"] == len(primary)
    assert q["rules_only_arm"]["first"]["rules_only"]["aligned_minus_swapped"]["n_items"] == len(primary)
    assert q["matched_arm"]["first"]["matched"]["aligned_minus_swapped"]["n_items"] == len(primary)
    assert q["matched_arm"]["yn"]["main_minus_matched"]["n_items"] == len(primary)
    mw = q["matched_arm"]["witness_first"]
    assert mw["descriptive"] and mw["matched"]["aligned_minus_swapped"]["n_items"] == len(primary)
    assert mw["fail_minus_witness_matched"]["n_items"] == mw["fail_minus_witness_main_same_items"]["n_items"] == len(primary)
    assert "witness_first" not in q["rules_only_arm"]
    assert "witness" in m["verdicts"]["per_model"]["qwen3-8b-fewshot"]["outcome_table"]
    assert set(m["matched_arm_verdicts"]) == set(P.CO_PRIMARY) and all("reading" in v for v in m["matched_arm_verdicts"].values())
    assert q["primary"]["first_without_fewshot_tool_classes"]["aligned_minus_swapped"]["n_items"] == sum(
        t["cls"] not in ("wrong_return_pm", "gc_balance") for t in primary)
    by_cls = q["primary_controls"]["discrimination_by_class"]
    if any(t["cls"] == "status" for t in primary):
        assert by_cls["status"]["evidence"] is False and by_cls["status"]["holds"] is None
    if any(t["cls"] == "unavailable" for t in primary):
        assert by_cls["unavailable"]["evidence"] is True and by_cls["unavailable"]["n_items"] == sum(
            t["cls"] == "unavailable" and t["witness_checker"] == "PASS" for t in primary)
    assert "outcome_table" in m["verdicts"]["per_model"]["qwen3-8b-fewshot"]
    sens = m["sensitivity_without_fewshot_tool_classes"]
    assert set(sens) == set(P.CO_PRIMARY) and all("same_category" in x for x in sens.values())
    assert set(q["primary"]["by_class"]) <= {"status", "unavailable"}
    assert q["format"]["yn_mass_median"] > 0 and q["format"]["target_mass_median"] > 0
    assert set(m["verdicts"]["per_model"]) == set(P.CO_PRIMARY)


def test_analysis_waits_for_every_shard(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    assert P.run_tau_score(_score_cfg(run_id="tau_score_smoke", n_shards=3)) == 0
    assert P.run_tau_analysis({**load_config("tau_smoke_analysis")}) == 1
    assert "waits for every scoring shard" in (runs / "tau_smoke_analysis" / "STATUS.md").read_text()


def test_resume_finishes_without_duplicates(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    cfg = _score_cfg(run_id="tau_score_resume")
    assert P.run_tau_score(cfg) == 0
    path = runs / "tau_score_resume" / "shard_00_of_02" / "scoring.jsonl"
    full = read_jsonl(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in full[: len(full) // 2]))
    assert P.run_tau_score({**cfg, "resume": True}) == 0
    again = read_jsonl(path)
    key = lambda r: (r["model"], r["transition_id"], r["readout"], r["condition"])
    assert sorted(map(key, again)) == sorted(map(key, full))


def _pins(runs):
    return {n: P.file_sha(runs / "tau_build" / n) for n in P.OUTPUT_FILES}


@pytest.mark.parametrize("name", P.OUTPUT_FILES)
def test_scoring_refuses_any_unpinned_or_changed_item_file(runs, monkeypatch, name):
    monkeypatch.setenv("WF_SHARD", "0")
    assert P.verified(_score_cfg(expected_outputs_sha256=_pins(runs)))[0]
    assert P.run_tau_score(_score_cfg(run_id="tau_score_pinned", expected_outputs_sha256={**_pins(runs), name: "0" * 64})) == 1
    assert "differs from the pinned" in (runs / "tau_score_pinned" / "STATUS.md").read_text()
    ok, why = P.verified(_score_cfg(expected_outputs_sha256={k: v for k, v in _pins(runs).items() if k != name}))
    assert not ok and "no pinned sha256" in why


def test_scoring_refuses_a_failed_verification_even_with_matching_files(runs):
    path = runs / "tau_verify" / "verify.json"
    original = path.read_text()
    try:
        path.write_text(json.dumps({**json.loads(original), "passed": False}))
        ok, why = P.verified(_score_cfg())
        assert not ok and "passed" in why
    finally:
        path.write_text(original)


def test_scoring_stops_when_the_token_check_fails(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    monkeypatch.setattr(P, "token_report", lambda scorer, sample: {"first_token_ok": False})
    assert P.run_tau_score(_score_cfg(run_id="tau_score_badtok")) == 1
    assert "token check fails" in (runs / "tau_score_badtok" / "shard_00_of_02" / "STATUS.md").read_text()
    assert not (runs / "tau_score_badtok" / "shard_00_of_02" / "scoring.jsonl").exists()


def _gap_run(runs, monkeypatch, run_id, edit):
    """Score both shards into run_id, edit the saved records, and analyse them."""
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        assert P.run_tau_score(_score_cfg(run_id=run_id)) == 0
    monkeypatch.delenv("WF_SHARD")
    path = runs / run_id / "shard_00_of_02" / "scoring.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in edit(read_jsonl(path))))
    real = P.load_config
    monkeypatch.setattr(P, "load_config", lambda name: _score_cfg(run_id=run_id) if name == run_id else real(name))
    rc = P.run_tau_analysis({"run_id": f"{run_id}_analysis", "source": run_id, "build_run": "tau_build", "models": list(P.CO_PRIMARY)})
    return rc, json.loads((runs / f"{run_id}_analysis" / "metrics.json").read_text())


def test_analysis_refuses_when_a_swapped_score_is_missing(runs, monkeypatch):
    def drop(recs):
        gone = next(r for r in recs if r["readout"] == "first" and r["condition"] == "swapped" and r["action"] == "fail")
        return [r for r in recs if r is not gone]
    rc, m = _gap_run(runs, monkeypatch, "tau_score_gap", drop)
    assert rc == 1 and m["missing_scores"] == 1


def test_analysis_refuses_stale_scores(runs, monkeypatch):
    rc, m = _gap_run(runs, monkeypatch, "tau_score_stale", lambda recs: [{**recs[0], "item_hash": "0" * 16}] + recs[1:])
    assert rc == 1 and m["stale_scores"] == 1


def test_scoring_refuses_changed_items(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    items = runs / "tau_build" / "items.jsonl"
    original = items.read_bytes()
    try:
        items.write_bytes(original.replace(b'"main_pool": false', b'"main_pool": true', 1))
        assert P.run_tau_score(_score_cfg(run_id="tau_score_changed")) == 1
        assert "not verified" in (runs / "tau_score_changed" / "STATUS.md").read_text()
    finally:
        items.write_bytes(original)


def test_scoring_refuses_a_verification_of_other_files(runs):
    """verify.json must name exactly the item files scoring reads (a second guard beside the run's input hashes)."""
    path = runs / "tau_verify" / "verify.json"
    original = path.read_text()
    try:
        rep = json.loads(original)
        rep["files"]["items.jsonl"] = "0" * 64
        path.write_text(json.dumps(rep))
        ok, why = P.verified(_score_cfg())
        assert not ok and "does not cover" in why
    finally:
        path.write_text(original)
    assert P.verified(_score_cfg())[0]


def test_contamination_runs_on_mock(runs):
    assert P.run_tau_contamination({**load_config("tau_contamination"), "per_kind": 3, "expected_outputs_sha256": _small_pins()}) == 0
    m = json.loads((runs / "tau_contamination" / "metrics.json").read_text())
    assert m["n_records"] == 9 and m["models"]["qwen3-8b-fewshot"]["all"]["n"] == 9
    assert m["minus_llama"]["qwen3-8b-fewshot"]["all"]["n"] == 9 and "llama3.1-8b-fewshot" not in m["minus_llama"]


def test_contamination_compares_with_llama_record_by_record():
    recs = [{"id": f"r{k}"} for k in range(4)]
    scores = [{"model": "llama3.1-8b-fewshot", "transition_id": f"r{k}", "kind": "user", "L": 1.0} for k in range(4)]
    scores += [{"model": "qwen3-8b-fewshot", "transition_id": f"r{k}", "kind": "user", "L": 1.0 + k} for k in range(4)]
    m = P.contamination_report(scores, recs, P.CO_PRIMARY)
    assert m["minus_llama"]["qwen3-8b-fewshot"]["all"]["mean"] == pytest.approx(1.5)
    assert m["models"]["llama3.1-8b-fewshot"]["user"]["mean_logp_verbatim_minus_resampled"] == 1.0


# ---------------- statistics and helpers ----------------


@pytest.mark.parametrize("est,ci,p,want", [
    (1.2, [0.6, 1.8], 0.002, "replicates (ALFWorld direction)"),
    (-0.9, [-1.4, -0.5], 0.002, "reversed (naming sharpens failure prediction)"),
    (0.2, [0.1, 0.3], 0.002, "small effect, direction positive"),
    (-0.2, [-0.3, -0.1], 0.002, "small effect, direction negative"),
    (0.05, [-0.2, 0.3], 0.5, "no effect larger than 0.5 nats"),
    (0.8, [-0.1, 1.7], 0.2, "inconclusive"),
    (1.2, [0.6, 1.8], 0.02, "inconclusive"),
    (float("nan"), [0, 1], 0.1, "not computable"),
])
def test_two_sided_decision_rule(est, ci, p, want):
    assert P.tau_verdict(est, ci, p) == want


OK = {"manipulation_check_passed": True, "discrimination_holds": True}


def _checks(**kw):
    return {m: {**OK, **kw} for m in P.CO_PRIMARY}


def test_holm_runs_over_both_models_and_one_model_is_never_a_replication():
    good = {"estimate": 1.0, "ci": [0.6, 1.4], "perm_p": 0.004}
    v = P.verdicts({"qwen3-8b-fewshot": good}, _checks())
    assert v["per_model"]["qwen3-8b-fewshot"]["holm_p"] == pytest.approx(0.008)
    assert v["per_model"]["llama3.1-8b-fewshot"]["verdict"] == "not scored"
    assert v["overall"] == "one model only (not a replication)"
    assert P.verdicts({m: good for m in P.CO_PRIMARY}, _checks())["overall"] == "replicates (both models)"
    rev = {"estimate": -1.0, "ci": [-1.4, -0.6], "perm_p": 0.004}
    assert P.verdicts({"qwen3-8b-fewshot": good, "llama3.1-8b-fewshot": rev}, _checks())["overall"].startswith("mixed")
    assert P.verdicts({m: rev for m in P.CO_PRIMARY}, _checks())["overall"] == "reversed (both models)"
    small = {"estimate": 0.2, "ci": [0.1, 0.3], "perm_p": 0.001}
    assert P.verdicts({"qwen3-8b-fewshot": good, "llama3.1-8b-fewshot": small}, _checks())["overall"] == \
        "ALFWorld direction in both models, size met by one (not a replication)"


def test_holm_decides_when_both_raw_p_values_are_below_the_bar():
    """Raw p 0.006 for both models: Holm makes both 0.012, above 0.01, so neither replicates (review 2)."""
    both = {m: {"estimate": 1.0, "ci": [0.6, 1.4], "perm_p": 0.006} for m in P.CO_PRIMARY}
    v = P.verdicts(both, _checks())
    for m in P.CO_PRIMARY:
        assert v["per_model"][m]["holm_p"] == pytest.approx(0.012)
        assert v["per_model"][m]["verdict"] == "inconclusive"
    assert v["overall"] == "inconclusive"
    assert P.tau_verdict(1.0, [0.6, 1.4], 0.006) == "replicates (ALFWorld direction)"  # unadjusted, it would


@pytest.mark.parametrize("v,want", [
    ({"a": "replicates (ALFWorld direction)", "b": "replicates (ALFWorld direction)"}, "replicates (both models)"),
    ({"a": "replicates (ALFWorld direction)", "b": "inconclusive"}, "one model only (not a replication)"),
    ({"a": "reversed (naming sharpens failure prediction)", "b": "no effect larger than 0.5 nats"}, "one model only (not a replication)"),
    ({"a": "no effect larger than 0.5 nats", "b": "no effect larger than 0.5 nats"}, "no effect larger than 0.5 nats (both models)"),
    ({"a": "no effect larger than 0.5 nats", "b": "inconclusive"}, "inconclusive"),
    ({"a": "small effect, direction positive", "b": "small effect, direction positive"}, "inconclusive"),
    ({"a": "replicates (ALFWorld direction)", "b": "small effect, direction positive"},
     "ALFWorld direction in both models, size met by one (not a replication)"),
    ({"a": "reversed (naming sharpens failure prediction)", "b": "small effect, direction negative"},
     "reversed direction in both models, size met by one"),
    ({"a": "replicates (ALFWorld direction)", "b": "small effect, direction negative"}, "mixed (the models disagree in direction)"),
    ({"a": "reversed (naming sharpens failure prediction)", "b": "small effect, direction positive"}, "mixed (the models disagree in direction)"),
    ({"a": "replicates (ALFWorld direction)", "b": "reversed (naming sharpens failure prediction)"}, "mixed (the models disagree in direction)"),
    ({"a": "replicates (ALFWorld direction)", "b": "inconclusive (reversed, but the model is not shown to discriminate failing from working calls)"},
     "one model only (not a replication)"),
])
def test_joint_verdict_rule(v, want):
    assert P.joint_verdict(v) == want


NULL = {"estimate": 0.05, "ci": [-0.2, 0.3], "perm_p": 0.6}
REV = {"estimate": -1.0, "ci": [-1.4, -0.6], "perm_p": 0.004}
GOOD = {"estimate": 1.0, "ci": [0.6, 1.4], "perm_p": 0.004}


@pytest.mark.parametrize("effect,manip,disc,want", [
    (NULL, True, True, "no effect larger than 0.5 nats"),
    (NULL, False, True, "inconclusive (null, but the task line is not shown to be read)"),
    (NULL, True, False, "inconclusive (null, but the model is not shown to discriminate failing from working calls)"),
    (NULL, False, False, "inconclusive (null, but the task line is not shown to be read)"),
    (REV, True, True, "reversed (naming sharpens failure prediction)"),
    (REV, True, False, "inconclusive (reversed, but the model is not shown to discriminate failing from working calls)"),
    (REV, False, True, "reversed (naming sharpens failure prediction)"),
    (GOOD, False, False, "replicates (ALFWorld direction)"),
])
def test_gates_read_the_right_check_for_each_verdict(effect, manip, disc, want):
    """A null needs the manipulation check and then discrimination; a reversal needs discrimination; the two flags
    are set differently so that a gate reading the wrong check is caught (review 2)."""
    v = P.verdicts({m: effect for m in P.CO_PRIMARY}, _checks(manipulation_check_passed=manip, discrimination_holds=disc))
    assert v["per_model"]["qwen3-8b-fewshot"]["verdict"] == want


def _blk(lo, hi, n=10):
    return {"estimate": (lo + hi) / 2, "ci": [lo, hi], "n_items": n}


@pytest.mark.parametrize("verdict,disc,fmw,wmm,reading,mention", [
    ("replicates (ALFWorld direction)", True, _blk(0.4, 1.2), _blk(0.3, 0.9), "overrides a visible fact", "adds to naming"),
    ("replicates (ALFWorld direction)", True, _blk(-0.2, 0.2), _blk(-0.2, 0.2), "general task-consistency bias", "naming the order is enough"),
    ("replicates (ALFWorld direction)", True, _blk(-1.2, -0.4), None, "general task-consistency bias", None),
    ("replicates (ALFWorld direction)", True, _blk(-0.3, 0.9), _blk(-0.3, 0.9), "general task-consistency bias", "uncertain"),
    ("replicates (ALFWorld direction)", False, _blk(0.4, 1.2), None, "not read as the task overriding visible evidence", None),
    ("small effect, direction positive", True, _blk(0.1, 0.3), None, "below the 0.5-nat size", None),
    ("reversed (naming sharpens failure prediction)", True, None, None, "sharpens failure prediction", None),
    ("no effect larger than 0.5 nats", True, None, None, "No leak", None),
    ("inconclusive", True, None, None, "Inconclusive", None),
])
def test_outcome_table_reading(verdict, disc, fmw, wmm, reading, mention):
    r = P.outcome_reading(verdict, disc, fmw, wmm)
    assert reading in r["reading"]
    if mention:
        assert mention in r["mention"]
    if fmw is not None:
        assert r["specificity"].startswith("specific" if fmw["ci"][0] > 0 else "not specific")


def test_verdict_reading_uses_the_controls_passed_in():
    c = _checks(fail_minus_witness=_blk(-0.2, 0.2), write_minus_mention=_blk(0.3, 0.9))
    v = P.verdicts({m: GOOD for m in P.CO_PRIMARY}, c)["per_model"]["qwen3-8b-fewshot"]
    assert "general task-consistency bias" in v["reading"] and "adds to naming" in v["outcome_table"]["mention"]
    bad = P.verdicts({m: GOOD for m in P.CO_PRIMARY}, _checks(discrimination_holds=False))["per_model"]["qwen3-8b-fewshot"]
    assert "not read as the task overriding" in bad["reading"]


@pytest.mark.parametrize("main,matched,want", [
    ("replicates (ALFWorld direction)", "replicates (ALFWorld direction)", "effect under both prompts"),
    ("no effect larger than 0.5 nats", "small effect, direction positive", "cannot be separated from a belief that the backend checks"),
    ("no effect larger than 0.5 nats", "no effect larger than 0.5 nats", "under either prompt"),
    ("replicates (ALFWorld direction)", "no effect larger than 0.5 nats", "main prompt only"),
    ("inconclusive (null, but the task line is not shown to be read)", "replicates (ALFWorld direction)", "inconclusive"),
])
def test_matched_arm_reading(main, matched, want):
    assert want in P.matched_reading(main, matched)
    mv = P.matched_verdicts({m: GOOD for m in P.CO_PRIMARY}, {m: {"verdict": main} for m in P.CO_PRIMARY})
    assert mv["qwen3-8b-fewshot"]["verdict"] == "replicates (ALFWorld direction)" and mv["qwen3-8b-fewshot"]["main_verdict"] == main


@pytest.mark.parametrize("est,ci,want", [(1.2, [0.3, 2.0], True), (0.9, [0.3, 1.5], False), (1.5, [-0.1, 3.0], False),
                                         (float("nan"), [0, 1], False)])
def test_check_threshold(est, ci, want):
    assert P.check_passes({"estimate": est, "ci": ci, "n_items": 10}) is want
    assert P.check_passes({"estimate": 2.0, "ci": [1, 3], "n_items": 0}) is False


def test_flip_shares_count_both_directions_and_product_clusters():
    items = [{"id": f"i{k}", "user_id": "u", "target_product_id": f"p{k % 2}"} for k in range(4)]
    L = {"i0": (1.0, -1.0), "i1": (-1.0, 1.0), "i2": (1.0, -2.0), "i3": (-2.0, -1.0)}  # two toward success, one toward failure
    by = {}
    for t in items:
        a, s = L[t["id"]]
        by[(t["id"], "main", "fail", "first", "aligned")] = {"L": a}
        by[(t["id"], "main", "fail", "first", "swapped")] = {"L": s}
    f = P.flip_shares(by, items)
    assert (f["toward_success"], f["toward_failure"], f["either"]) == (0.5, 0.25, 0.75)
    blk = P.paired(by, items, "first", "aligned", "swapped", cluster="product")
    assert blk["n_clusters"] == 2 and blk["n_users"] == 1
    assert P.cluster_of({"id": "x", "user_id": "u", "target_product_id": None}, "product") == "no-product:x"
    shift = P.paired(by, items, "first", "aligned", "swapped", scale="prob")
    sig = lambda x: 1 / (1 + math.exp(-x))
    assert shift["estimate"] == pytest.approx(sum(sig(a) - sig(s) for a, s in L.values()) / 4)


def test_paired_effects_cluster_by_user():
    items = [{"id": f"i{k}", "user_id": "u1"} for k in range(30)]
    by = {}
    for k, t in enumerate(items):
        by[(t["id"], "main", "fail", "first", "aligned")] = {"L": 1.0 + 0.01 * k}
        by[(t["id"], "main", "fail", "first", "swapped")] = {"L": 0.0}
    one_user = P.paired(by, items, "first", "aligned", "swapped")
    assert one_user["n_users"] == 1 and one_user["perm_p"] == 1.0  # one cluster: the sign flip has nothing to say
    spread = P.paired(by, [{**t, "user_id": t["id"]} for t in items], "first", "aligned", "swapped")
    assert spread["perm_p"] < 0.01 and spread["estimate"] == pytest.approx(one_user["estimate"])


def test_strata_allow_items_without_a_swapped_line():
    """Natural items with no other order shown have swapped_achievable None (regression, 2026-11-20)."""
    xs = [{"user_id": "u", "main_pool": True, "status_matched": False, "cls": "status", "swapped_achievable": v,
           "action": {"name": "x"}} for v in (True, None, False)]
    assert P.strata_counts(xs)["main_by_swapped_achievable"] == {k: {"items": 1, "users": 1} for k in ("False", "None", "True")}


def test_full_norm_is_per_token():
    rec = {"L": -40.0, "logp_plus": -60.0, "n_tokens_plus": 300, "logp_minus": -20.0, "n_tokens_minus": 10}
    assert P.value(rec, "full") == -40.0 and P.value(rec, "full_norm") == pytest.approx(-0.2 + 2.0)


def test_resampled_ids_are_consistent_and_keep_addresses_and_numbers():
    text = json.dumps({"order_id": "#W4140680", "x": "#W4140680", "address": {"address1": "61523 Laurel Lane", "zip": "19036"},
                       "items": [{"item_id": "0273929280", "price": 244.95}], "payment_methods": {"gift_card_4374071": {"id": "gift_card_4374071"}},
                       "user": "anya_3271", "short": "#A123456"})
    out = P.resample_ids(text, "k")
    assert out == P.resample_ids(text, "k") and out != text and len(out) == len(text)
    new = json.loads(out)
    assert new["order_id"] == new["x"] != "#W4140680"
    assert new["address"] == {"address1": "61523 Laurel Lane", "zip": "19036"}  # geography stays
    assert new["items"][0]["price"] == 244.95 and new["user"] == "anya_3271" and new["short"] != "#A123456"
    assert new["items"][0]["item_id"] != "0273929280" and new["items"][0]["item_id"][0] in "123456789"
    (key, rec), = new["payment_methods"].items()
    assert key == rec["id"] != "gift_card_4374071" and key.startswith("gift_card_")
    with pytest.raises(ValueError):
        P.resample_ids('{"a":  1}', "k")


def test_resampling_keeps_every_zip_of_real_records(runs):
    gen = read_jsonl(runs / "tau_build" / "items.jsonl")
    recs = P.contamination_records(gen, 5)
    assert recs
    for r in recs:
        a, b = json.loads(r["verbatim"]), json.loads(r["resampled"])
        if "address" in a:
            assert a["address"] == b["address"]
        assert r["verbatim"] != r["resampled"]


def test_contrast_of_effects_is_a_difference_of_paired_differences():
    items = [{"id": f"i{k}", "user_id": f"u{k}"} for k in range(6)]
    by = {}
    for t in items:
        for act, (a, s) in (("fail", (3.0, 1.0)), ("witness", (2.5, 2.0))):
            by[(t["id"], "main", act, "first", "aligned")] = {"L": a}
            by[(t["id"], "main", act, "first", "swapped")] = {"L": s}
    c = P.contrast_of_effects(by, items, "first", ("fail", "aligned", "swapped"), ("witness", "aligned", "swapped"))
    assert c["estimate"] == pytest.approx(2.0 - 0.5) and c["n_items"] == 6 and c["n_users"] == 6
    one = P.contrast_of_effects(by, [{**t, "user_id": "u"} for t in items], "first", ("fail", "aligned", "swapped"),
                                ("witness", "aligned", "swapped"))
    assert one["n_users"] == 1 and one["perm_p"] == 1.0


class _Recorder:
    """A stand-in scorer that records what each job asks for."""
    mock = True

    def __init__(self):
        self.calls = []

    def score_one_token_pair(self, tid, cond, messages, plus, minus):
        self.calls.append(("one", cond, messages, plus, minus))
        return None

    def score_pair(self, tid, cond, messages, plus, minus):
        self.calls.append(("pair", cond, messages, plus, minus))
        return None


def test_each_job_scores_the_right_call_question_and_candidates(runs):
    from wf.tau.prompts import OUTPUT_CUE, TARGET_QUESTION, YN_QUESTION, system_message
    gen, _, few = P.load_built({"build_run": "tau_build"})
    t = next(x for x in gen if x["main_pool"])
    cases = {("main", "fail", "first", "aligned"): ("one", '{"', "Error", OUTPUT_CUE, t["action"]),
             ("main", "fail", "yn", "swapped"): ("one", "Yes", "No", YN_QUESTION, t["action"]),
             ("main", "fail", "target", "swapped"): ("one", "Yes", "No", TARGET_QUESTION, t["action"]),
             ("main", "fail", "full", "neutral"): ("pair", t["success_twin"], t["failure_reply"], OUTPUT_CUE, t["action"]),
             ("main", "witness", "full", "aligned"): ("pair", t["witness_output"], t["witness_failure"], OUTPUT_CUE, t["aligned_witness"]),
             ("main", "witness", "first", "swapped"): ("one", '{"', "Error", OUTPUT_CUE, t["aligned_witness"]),
             ("rules_only", "fail", "first", "aligned"): ("one", '{"', "Error", OUTPUT_CUE, t["action"]),
             ("main", "fail", "first", "mention_swapped"): ("one", '{"', "Error", OUTPUT_CUE, t["action"])}
    for job, (kind, plus, minus, cue, call) in cases.items():
        rec = _Recorder()
        P._score_job(rec, t, few, job)
        (k, cond, ms, a, b), = rec.calls
        assert (k, cond, a, b) == (kind, job[3], plus, minus), job
        assert ms[-1]["content"].endswith("\n" + cue) and ms[-1]["content"].startswith(f"Task: {t['task_' + job[3]]}\n")
        assert f"Next call: {json.dumps({'name': call['name'], 'arguments': call['arguments']})}\n" in ms[-1]["content"]
        assert ms[0]["content"] == system_message(job[0])


# ---------------- review 2: direction and condition of every control, arms, hashes, projection, smoke ----------------


def _control_scores(n=12):
    """Constructed scores where every condition, action and readout has its own value, so that a control reading the
    wrong condition, the wrong action or the wrong direction gives a different number."""
    items = [{"id": f"i{k}", "user_id": f"u{k}", "cls": "unavailable", "witness_checker": "PASS"} for k in range(n)]
    L = {("fail", "aligned"): 3.0, ("fail", "swapped"): 1.0, ("fail", "neutral"): -2.0,
         ("fail", "mention_aligned"): 0.5, ("fail", "mention_swapped"): 0.25,
         ("witness", "aligned"): 4.0, ("witness", "swapped"): 3.5, ("witness", "neutral"): 2.5}
    by = {}
    for k, t in enumerate(items):
        jitter = 0.01 * k
        for (act, c), v in L.items():
            for r in ("first", "yn"):
                by[(t["id"], "main", act, r, c)] = {"L": v + jitter + (0.1 if r == "yn" else 0.0), "logp_plus": 0.0, "logp_minus": 0.0}
        by[(t["id"], "main", "fail", "target", "aligned")] = {"L": 2.0 + jitter}
        by[(t["id"], "main", "fail", "target", "swapped")] = {"L": -1.0 + jitter}
    return items, by


def test_controls_read_the_designed_condition_action_and_direction():
    items, by = _control_scores()
    c = P.controls_block(by, items)
    assert c["manipulation"]["estimate"] == pytest.approx(3.0)  # target: aligned minus swapped, not the reverse
    assert c["discrimination"]["first"]["estimate"] == pytest.approx(2.5 - (-2.0))  # neutral task, witness minus fail
    assert c["discrimination"]["yn"]["estimate"] == pytest.approx(4.5)
    assert c["discrimination_by_class"]["unavailable"]["first"]["estimate"] == pytest.approx(4.5)
    assert c["fail_minus_witness_effect"]["first"]["estimate"] == pytest.approx((3.0 - 1.0) - (4.0 - 3.5))
    assert c["write_minus_mention_effect"]["first"]["estimate"] == pytest.approx((3.0 - 1.0) - (0.5 - 0.25))
    m = c["mention"]["first"]
    assert m["mention_aligned_minus_mention_swapped"]["estimate"] == pytest.approx(0.25)
    assert m["mention_aligned_minus_neutral"]["estimate"] == pytest.approx(2.5)
    assert m["aligned_minus_mention_aligned"]["estimate"] == pytest.approx(2.5)
    assert c["witness"]["first"]["aligned_minus_swapped"]["estimate"] == pytest.approx(0.5)
    assert P.check_passes(c["manipulation"]) and P.check_passes(c["discrimination"]["first"])


def test_arm_contrast_subtracts_the_named_arm():
    items, by = _control_scores()
    for t in items:
        for c, v in (("aligned", 1.0), ("swapped", 0.75)):
            by[(t["id"], "matched", "fail", "first", c)] = {"L": v}
    out = P.contrast_of_effects(by, items, "first", ("fail", "aligned", "swapped"), ("fail", "aligned", "swapped"), arms=("main", "matched"))
    assert out["estimate"] == pytest.approx(2.0 - 0.25)


def test_status_class_is_flagged_without_discrimination_evidence():
    items, by = _control_scores()
    status = [{**t, "id": f"s{k}", "cls": "status", "witness_checker": "UNKNOWN"} for k, t in enumerate(items)]
    for k, t in enumerate(items):
        for key in [x for x in by if x[0] == t["id"]]:
            by[(f"s{k}",) + key[1:]] = by[key]
    c = P.controls_block(by, items + status)
    assert c["discrimination_by_class"]["status"]["evidence"] is False and "no discrimination evidence" in c["discrimination_by_class"]["status"]["note"]
    assert c["discrimination_by_class"]["unavailable"]["holds"] is True
    assert c["discrimination"]["first"]["n_items"] == len(items)  # status witnesses never enter the check


def test_exploratory_arms_are_drawn_from_the_whole_set_not_per_shard(runs, monkeypatch):
    """With 3 items per arm over 2 shards, exactly 3 items get each arm (drawn per shard it would be up to 6)."""
    cfg = _score_cfg(run_id="tau_score_arms", rules_arm_items=3, matched_arm_items=3)
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        assert P.run_tau_score(cfg) == 0
    recs = [r for i in (0, 1) for r in read_jsonl(runs / "tau_score_arms" / f"shard_{i:02d}_of_02" / "scoring.jsonl")]
    every = P.scoring_sets(cfg)
    for arm in ("rules_only", "matched"):
        got = {r["transition_id"] for r in recs if r["arm"] == arm}
        assert got == P.arm_ids(every, 3) and len(got) == 3
    assert P.extra_arm_ids(every, cfg)["rules_only"] == P.extra_arm_ids(every, cfg)["matched"]  # the same items


@pytest.mark.parametrize("field", ["history", "action", "failure_reply", "success_twin", "task_aligned", "task_swapped",
                                   "task_neutral", "task_mention_aligned", "task_mention_swapped", "aligned_witness",
                                   "witness_output", "witness_failure"])
def test_item_hash_covers_every_scored_field(runs, field):
    t = next(x for x in read_jsonl(runs / "tau_build" / "items.jsonl") if x["main_pool"])
    changed = {**t, field: (t[field] + "x") if isinstance(t[field], str) else {"changed": t[field]}}
    assert P.item_hash(changed) != P.item_hash(t)


def _fake_smoke_report(sec_a, sec_b, sd=1.0):
    blk = lambda s: {"format": {"seconds_mean_by_readout": {r: s for r in ("first", "yn", "full", "target")}},
                     "sd_of_differences": {"first": {"n": 20, "sd": sd}}}
    return {"models": {"qwen3-8b-fewshot": blk(sec_a), "llama3.1-8b-fewshot": blk(sec_b)}}


def test_projection_adds_both_models_hours_and_checks_the_limit(runs, monkeypatch):
    real = P.load_config
    cfg = _score_cfg()
    monkeypatch.setattr(P, "load_config", lambda name: cfg if name == "tau_score" else real(name))
    pr = P.projection("tau_score", _fake_smoke_report(1.0, 3.0), {"build_run": "tau_build"},
                      {"time_limit_hours": 8, "alfworld_first_token_delta_l": {"llama3.1-8b-fewshot": 0.68}})
    q, l = pr["models"]["qwen3-8b-fewshot"], pr["models"]["llama3.1-8b-fewshot"]
    assert l["hours_max_shard"] == pytest.approx(3 * q["hours_max_shard"])
    assert pr["hours_max_shard_both_models"] == pytest.approx(q["hours_max_shard"] + l["hours_max_shard"])
    assert pr["fits_time_limit"] is (pr["hours_max_shard_both_models"] <= 8)
    every = P.scoring_sets(cfg)
    jobs = sum(len(P.jobs_for(t, cfg, P.extra_arm_ids(every, cfg))) for t in every)
    assert q["hours_all_shards"] == pytest.approx(jobs / 3600)
    prim = [t for t in every if P.is_primary(t)]
    assert l["precision"]["first"]["primary"]["n"] == len(prim)
    assert l["power_if_alfworld_sized"]["alfworld_first_token_delta_l"] == 0.68 and q["power_if_alfworld_sized"] is None


def test_precision_and_power_formulas():
    items = [{"user_id": u} for u in ["a"] * 3 + ["b"]]  # n = 4, m_eff = (9 + 1) / 4 = 2.5
    p = P.precision(2.0, items)
    assert p["m_eff"] == pytest.approx(2.5) and p["se"] == pytest.approx([1.0, 2.0 * math.sqrt(2.5 / 4)])
    assert p["ci_half_width"][0] == pytest.approx(1.96)
    w = P.power_at(0.68, 0.1)
    from statistics import NormalDist
    assert w["p_estimate_at_least_0.5"] == pytest.approx(NormalDist().cdf(1.8))
    assert w["p_ci_above_0"] == pytest.approx(NormalDist().cdf(6.8 - 1.96))
    assert P.power_at(0.0, 0.1)["p_null_verdict_if_true_effect_0"] == pytest.approx(2 * NormalDist().cdf(5 - 1.96) - 1)
    assert P.power_at(0.0, 1.0)["p_null_verdict_if_true_effect_0"] == 0.0


def test_smoke_analysis_reports_format_timing_and_precision_but_no_estimate(runs, monkeypatch):
    run_id = "tau_score_smoke_r2"
    cfg = {**load_config("tau_score_smoke"), "run_id": run_id, "expected_outputs_sha256": _small_pins()}
    assert P.run_tau_score(cfg) == 0
    real = P.load_config
    monkeypatch.setattr(P, "load_config", lambda name: cfg if name == run_id else _score_cfg() if name == "tau_score" else real(name))
    acfg = {**load_config("tau_smoke_analysis"), "run_id": f"{run_id}_analysis", "source": run_id}
    assert P.run_tau_analysis(acfg) == 0
    text = (runs / f"{run_id}_analysis" / "metrics.json").read_text()
    m = json.loads(text)
    assert "verdicts" not in m and "matched_arm_verdicts" not in m
    assert '"estimate"' not in text and '"perm_p"' not in text and '"ci"' not in text
    for blk in m["models"].values():
        assert set(blk) == {"format", "sd_of_differences"}
    pr = m["projection"]
    assert pr["time_limit_hours"] == 8 and "fits_time_limit" in pr
    assert pr["models"]["llama3.1-8b-fewshot"]["power_if_alfworld_sized"]["alfworld_first_token_delta_l"] == 0.68
    status = (runs / f"{run_id}_analysis" / "STATUS.md").read_text()
    assert "ΔL" not in status and "overall" not in status and "projected hours" in status


def test_smoke_set_adds_the_longest_items_and_the_limit_matches_the_sbatch(runs):
    cfg = {**load_config("tau_score_smoke"), "expected_outputs_sha256": _small_pins()}
    got = P.scoring_sets(cfg)
    main = [t for t in read_jsonl(runs / "tau_build" / "items.jsonl") if t["main_pool"]]
    longest = max(main, key=lambda t: (P.prompt_chars(t), t["id"]))
    assert longest["id"] in {t["id"] for t in got} and 20 <= len(got) <= 22
    assert len({t["id"] for t in got}) == len(got)
    sbatch = (P.Path(__file__).resolve().parents[1] / "pace" / "tau_score.sbatch").read_text()
    hours = int(re.search(r"#SBATCH --time=(\d+):00:00", sbatch).group(1))
    assert load_config("tau_smoke_analysis")["time_limit_hours"] == hours == 8


def test_a_scoring_config_without_pins_is_refused(runs):
    for pins in (None, {}):
        ok, why = P.verified(_score_cfg(expected_outputs_sha256=pins))
        assert not ok and "pins no sha256" in why
    assert P.verified(_score_cfg())[0]


def test_sensitivity_verdict_compares_categories_under_holm():
    prim = {m: GOOD for m in P.CO_PRIMARY}
    sens = {m: {"estimate": 0.3, "ci": [0.1, 0.5], "perm_p": 0.001} for m in P.CO_PRIMARY}
    out = P.sensitivity_verdicts(sens, prim)
    assert out["qwen3-8b-fewshot"] == {"verdict": "small effect, direction positive",
                                       "primary_ungated_verdict": "replicates (ALFWorld direction)", "same_category": False}
    both = P.sensitivity_verdicts({m: {**GOOD, "perm_p": 0.006} for m in P.CO_PRIMARY}, prim)
    assert both["llama3.1-8b-fewshot"]["verdict"] == "inconclusive"  # Holm within the sensitivity analysis
    assert P.sensitivity_verdicts(prim, prim)["llama3.1-8b-fewshot"]["same_category"] is True


def test_sensitivity_population_leaves_out_exactly_the_fewshot_tool_classes():
    """Constructed scores: the classes that share a few-shot example's tool carry a large opposite effect, so the
    sensitivity estimate differs from the primary one only if they are left out (review 2's B6)."""
    items, by = _control_scores(12)
    cls = ["status", "unavailable", "wrong_return_pm", "gc_balance"]
    items = [{**t, "cls": cls[k % 4], "target_product_id": None} for k, t in enumerate(items)]
    for t in items:
        if t["cls"] in ("wrong_return_pm", "gc_balance"):
            by[(t["id"], "main", "fail", "first", "aligned")] = {**by[(t["id"], "main", "fail", "first", "aligned")], "L": -10.0}
    blk = P.population_block(by, items)
    sens = blk["first_without_fewshot_tool_classes"]["aligned_minus_swapped"]
    assert sens["n_items"] == 6 and blk["first"]["aligned_minus_swapped"]["n_items"] == 12
    assert sens["estimate"] == pytest.approx(2.0) and blk["first"]["aligned_minus_swapped"]["estimate"] < 0


# ---------------- fix round 3 (review 3 and the task-invariance gate) ----------------


@pytest.mark.parametrize("witness,want", [
    (_blk(0.2, 1.4), P.WITNESS_SHIFT_NULL),
    (_blk(-0.3, 0.6), "No leak of more than 0.5 nats, with the task line read and the visible record used"),
    (_blk(-1.4, -0.2), "No leak of more than 0.5 nats, with the task line read and the visible record used"),
    (None, "No leak of more than 0.5 nats, with the task line read and the visible record used"),
])
def test_a_null_reads_the_witness(witness, want):
    """A null on the failing call is "no leak" only if the working call does not move with the task; a witness
    aligned - swapped with a CI above 0 makes it an authorization or task-consistency reading (review 3)."""
    r = P.outcome_reading("no effect larger than 0.5 nats", True, _blk(-0.2, 0.2), None, witness)
    assert r["reading"] == want
    assert P.WITNESS_SHIFT_NULL == ("No leak for failing calls with a visible failure, but the task shifts predictions for "
                                    "working calls (an authorization or task-consistency reading)")
    # the witness never changes a non-null reading's category
    assert "overrides a visible fact" in P.outcome_reading("replicates (ALFWorld direction)", True, _blk(0.4, 1.2), None, witness)["reading"]


def test_null_verdicts_read_the_witness_and_the_joint_line_says_so():
    shift = P.verdicts({m: NULL for m in P.CO_PRIMARY}, _checks(witness_effect=_blk(0.2, 1.4)))
    assert shift["overall"] == "no effect larger than 0.5 nats (both models)"  # the category is about failing calls
    assert all(x["verdict"] == "no effect larger than 0.5 nats" and x["reading"] == P.WITNESS_SHIFT_NULL
               for x in shift["per_model"].values())
    assert "for failing calls with a visible failure only" in shift["overall_note"]
    assert all(m in shift["overall_note"] for m in P.CO_PRIMARY)
    flat = P.verdicts({m: NULL for m in P.CO_PRIMARY}, _checks(witness_effect=_blk(-0.3, 0.6)))
    assert "overall_note" not in flat and all(x["reading"].startswith("No leak of more") for x in flat["per_model"].values())
    one = P.verdicts({m: NULL for m in P.CO_PRIMARY}, {"qwen3-8b-fewshot": {**OK, "witness_effect": _blk(0.2, 1.4)},
                                                       "llama3.1-8b-fewshot": {**OK, "witness_effect": _blk(-0.3, 0.6)}})
    assert one["overall_note"].endswith("in qwen3-8b-fewshot (an authorization or task-consistency reading)")
    # a gated null (inconclusive) keeps its inconclusive reading whatever the witness does
    gated = P.verdicts({m: NULL for m in P.CO_PRIMARY}, _checks(manipulation_check_passed=False, witness_effect=_blk(0.2, 1.4)))
    assert "overall_note" not in gated and gated["per_model"]["qwen3-8b-fewshot"]["reading"].startswith("Inconclusive")


def test_task_consistency_row_and_mention_reading_name_the_authorization_reading():
    r = P.outcome_reading("replicates (ALFWorld direction)", True, _blk(-0.2, 0.2), _blk(0.3, 0.9))
    assert "or a belief that the backend checks calls against the request" in r["reading"]
    assert "an authorization reading predicts this too" in r["mention"]
    assert "authorization" not in P.outcome_reading("replicates (ALFWorld direction)", True, _blk(-0.2, 0.2), _blk(-0.2, 0.2))["mention"]


def test_matched_reading_for_a_main_null_is_the_prestated_text():
    want = ("null under the main prompt but an effect under the matched prompt: the effect appears when the prompt does not "
            "state that the backend ignores the task; it cannot be separated from a belief that the backend checks calls "
            "against the user's request, so it is not read as goal leakage into the backend's state.")
    assert P.matched_reading("no effect larger than 0.5 nats", "replicates (ALFWorld direction)") == want
    assert P.matched_reading("no effect larger than 0.5 nats", "small effect, direction positive") == want
    assert P.matched_reading("replicates (ALFWorld direction)", "replicates (ALFWorld direction)") != want


def test_matched_arm_scores_the_witness_first_token_under_three_task_lines(runs):
    cfg = _score_cfg()
    gen, _, _ = P.load_built({"build_run": "tau_build"})
    every = P.scoring_sets(cfg)
    arms = P.extra_arm_ids(every, cfg)
    t = next(x for x in every if x["id"] in arms["matched"])
    jobs = P.jobs_for(t, cfg, arms)
    assert sorted(j for j in jobs if j[:2] == ("matched", "witness")) == sorted(
        ("matched", "witness", "first", c) for c in ("aligned", "swapped", "neutral"))
    assert not [j for j in jobs if j[:2] == ("rules_only", "witness")]
    assert not [j for j in P.jobs_for(t, {**cfg, "matched_witness": False}, arms) if j[:2] == ("matched", "witness")]
    assert not [j for j in P.jobs_for({**t, "aligned_witness": None}, cfg, arms) if j[1] == "witness"]
    outside = next(x for x in every if x["id"] not in arms["matched"])
    assert not [j for j in P.jobs_for(outside, cfg, arms) if j[0] == "matched"]
    rec = _Recorder()
    P._score_job(rec, t, P.load_built({"build_run": "tau_build"})[2], ("matched", "witness", "first", "swapped"))
    (k, cond, ms, a, b), = rec.calls
    from wf.tau.prompts import system_message
    assert (k, cond, a, b) == ("one", "swapped", '{"', "Error") and ms[0]["content"] == system_message("matched")
    assert f"Next call: {json.dumps({'name': t['aligned_witness']['name'], 'arguments': t['aligned_witness']['arguments']})}\n" in ms[-1]["content"]


def test_matched_arm_witness_readout_reads_the_matched_arm():
    items, by = _control_scores()
    for t in items:
        for (act, c), v in {("fail", "aligned"): 1.0, ("fail", "swapped"): 0.75, ("fail", "neutral"): 0.0,
                            ("witness", "aligned"): 2.0, ("witness", "swapped"): 1.25, ("witness", "neutral"): 0.5}.items():
            by[(t["id"], "matched", act, "first", c)] = {"L": v}
            if act == "fail":
                by[(t["id"], "matched", act, "yn", c)] = {"L": v}
    blk = P.arm_block(by, items, "matched")["witness_first"]
    assert blk["descriptive"] is True
    assert blk["matched"]["aligned_minus_swapped"]["estimate"] == pytest.approx(0.75)
    assert blk["matched"]["swapped_minus_neutral"]["estimate"] == pytest.approx(0.75)
    assert blk["main_same_items"]["aligned_minus_swapped"]["estimate"] == pytest.approx(0.5)  # main: 4.0 - 3.5
    assert blk["fail_minus_witness_matched"]["estimate"] == pytest.approx(0.25 - 0.75)
    assert blk["fail_minus_witness_main_same_items"]["estimate"] == pytest.approx(2.0 - 0.5)
    assert "witness_first" not in P.arm_block(by, items, "rules_only")


def test_checks_pass_the_witness_aligned_minus_swapped_to_the_reading():
    """The analysis hands the outcome table the witness's first-token aligned - swapped (4.0 - 3.5 here), not its
    aligned - neutral (1.5) nor the failing call's (2.0)."""
    items, by = _control_scores()
    c = P.checks_of(P.controls_block(by, items))
    assert c["witness_effect"]["estimate"] == pytest.approx(0.5)
    assert c["fail_minus_witness"]["estimate"] == pytest.approx(1.5) and c["write_minus_mention"]["estimate"] == pytest.approx(1.75)
    assert c["manipulation_check_passed"] and c["discrimination_holds"]
