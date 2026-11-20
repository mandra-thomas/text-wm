"""τ-bench runs (wf/tau/pipeline.py) end to end on mock scoring: build -> verify -> score -> analysis, the gates
that refuse unverified or changed items, resume, the two-sided decision rule and the contamination helpers."""
from __future__ import annotations

import json

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


def _score_cfg(**kw):
    return {**load_config("tau_score"), "n_shards": 2, **kw}


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
    # every main item x model x readout x condition, once; shards split by user
    assert len(recs) == len(main) * 2 * 3 * 3
    users = [{r["user_id"] for r in read_jsonl(runs / "tau_score" / f"shard_{i:02d}_of_02" / "scoring.jsonl")} for i in (0, 1)]
    assert not users[0] & users[1]
    assert P.run_tau_analysis(load_config("tau_analysis")) == 0
    m = json.loads((runs / "tau_analysis" / "metrics.json").read_text())
    assert m["missing_primary_scores"] == 0 and m["items"]["main"] == len(main)
    blk = m["models"]["qwen3-8b-fewshot"]["first"]["main"]["aligned_minus_swapped"]
    assert blk["n_items"] == len(main) and blk["n_users"] == len({t["user_id"] for t in main})
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


def test_scoring_refuses_changed_or_unpinned_items(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    assert P.run_tau_score(_score_cfg(run_id="tau_score_pinned", expected_items_sha256="0" * 64)) == 1
    assert "differs from the pinned" in (runs / "tau_score_pinned" / "STATUS.md").read_text()
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
    assert P.run_tau_contamination({**load_config("tau_contamination"), "per_kind": 3}) == 0
    m = json.loads((runs / "tau_contamination" / "metrics.json").read_text())
    assert m["n_records"] == 9 and m["models"]["qwen3-8b-fewshot"]["all"]["n"] == 9


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


def test_holm_runs_over_both_models_and_a_missing_model_is_never_a_free_pass():
    good = {"estimate": 1.0, "ci": [0.6, 1.4], "perm_p": 0.004}
    v = P.verdicts({"qwen3-8b-fewshot": good})
    assert v["per_model"]["qwen3-8b-fewshot"]["holm_p"] == pytest.approx(0.008)
    assert v["per_model"]["llama3.1-8b-fewshot"]["verdict"] == "not scored" and v["overall"] == "replicates"
    rev = {"estimate": -1.0, "ci": [-1.4, -0.6], "perm_p": 0.004}
    assert P.verdicts({"qwen3-8b-fewshot": good, "llama3.1-8b-fewshot": rev})["overall"] == "mixed"


def test_paired_effects_cluster_by_user():
    items = [{"id": f"i{k}", "user_id": "u1"} for k in range(30)]
    by = {}
    for k, t in enumerate(items):
        by[(t["id"], "first", "aligned")] = {"L": 1.0 + 0.01 * k}
        by[(t["id"], "first", "swapped")] = {"L": 0.0}
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


def test_resampled_ids_are_consistent_and_keep_everything_else():
    text = '{"order_id": "#W4140680", "x": "#W4140680", "zip": "19036", "price": 244.95, "user": "anya_3271"}'
    out = P.resample_ids(text, "k")
    assert out == P.resample_ids(text, "k") and out != text and len(out) == len(text)
    new = json.loads(out)
    assert new["order_id"] == new["x"] != "#W4140680" and new["zip"] != "19036"
    assert new["price"] == 244.95 and new["user"] == "anya_3271"
