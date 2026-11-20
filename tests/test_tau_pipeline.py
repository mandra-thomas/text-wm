"""τ-bench runs (wf/tau/pipeline.py) end to end on mock scoring: build -> verify -> score -> analysis, the gates
that refuse unverified or changed items, resume, the two-sided decision rule and the contamination helpers."""
from __future__ import annotations

import json
import math

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
    """The real scoring config, unpinned: the pinned sha256 values are the full build's, not this small one's. Its
    rules-only arm (500 items) takes every primary item of this small build."""
    return {**load_config("tau_score"), "n_shards": 2, "expected_outputs_sha256": None, **kw}


def test_scoring_configs_pin_every_item_file_alike():
    pins = [load_config(n).get("expected_outputs_sha256") for n in ("tau_score", "tau_score_smoke", "tau_contamination")]
    assert all(p == pins[0] for p in pins)
    assert set(pins[0]) == set(P.OUTPUT_FILES) and all(len(v) == 64 for v in pins[0].values())
    full = load_config("tau_score")
    assert full["witness"] and full["mention"] and full["target_check"] and full["rules_arm_items"] == 500


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
    # full) = 9, mention pair x (first, yn) = 4, manipulation check 2, witness 9; the rules-only arm 6 on every
    # primary item here (fewer than its 500).
    assert all(t["task_swapped"] and t["aligned_witness"] for t in main)
    n_primary = sum(t["status_matched"] for t in main)
    assert 0 < n_primary < 500
    assert len(recs) == 2 * (len(main) * (9 + 4 + 2 + 9) + n_primary * 6)
    keys = [P.score_key(r) for r in recs]
    assert len(keys) == len(set(keys)) and all(r["seconds"] >= 0 for r in recs)
    assert {(r["arm"], r["action"], r["readout"]) for r in recs} == {
        ("main", "fail", "first"), ("main", "fail", "yn"), ("main", "fail", "full"), ("main", "fail", "target"),
        ("main", "witness", "first"), ("main", "witness", "yn"), ("main", "witness", "full"),
        ("rules_only", "fail", "first"), ("rules_only", "fail", "yn")}
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
    assert P.run_tau_contamination({**load_config("tau_contamination"), "per_kind": 3, "expected_outputs_sha256": None}) == 0
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


def test_holm_runs_over_both_models_and_one_model_is_never_a_replication():
    good = {"estimate": 1.0, "ci": [0.6, 1.4], "perm_p": 0.004}
    v = P.verdicts({"qwen3-8b-fewshot": good})
    assert v["per_model"]["qwen3-8b-fewshot"]["holm_p"] == pytest.approx(0.008)
    assert v["per_model"]["llama3.1-8b-fewshot"]["verdict"] == "not scored"
    assert v["overall"] == "one model only (not a replication)"
    assert P.verdicts({m: good for m in P.CO_PRIMARY})["overall"] == "replicates (both models)"
    rev = {"estimate": -1.0, "ci": [-1.4, -0.6], "perm_p": 0.004}
    assert P.verdicts({"qwen3-8b-fewshot": good, "llama3.1-8b-fewshot": rev})["overall"].startswith("mixed")
    assert P.verdicts({m: rev for m in P.CO_PRIMARY})["overall"] == "reversed (both models)"
    small = {"estimate": 0.2, "ci": [0.1, 0.3], "perm_p": 0.001}
    assert P.verdicts({"qwen3-8b-fewshot": good, "llama3.1-8b-fewshot": small})["overall"] == "one model only (not a replication)"


@pytest.mark.parametrize("v,want", [
    ({"a": "replicates (ALFWorld direction)", "b": "replicates (ALFWorld direction)"}, "replicates (both models)"),
    ({"a": "replicates (ALFWorld direction)", "b": "inconclusive"}, "one model only (not a replication)"),
    ({"a": "no effect larger than 0.5 nats", "b": "no effect larger than 0.5 nats"}, "no effect larger than 0.5 nats (both models)"),
    ({"a": "no effect larger than 0.5 nats", "b": "inconclusive"}, "inconclusive"),
    ({"a": "small effect, direction positive", "b": "small effect, direction positive"}, "inconclusive"),
])
def test_joint_verdict_rule(v, want):
    assert P.joint_verdict(v) == want


def test_a_null_needs_the_manipulation_check_and_an_effect_is_read_through_discrimination():
    null = {"estimate": 0.05, "ci": [-0.2, 0.3], "perm_p": 0.6}
    ok = {m: {"manipulation_check_passed": True, "discrimination_holds": True} for m in P.CO_PRIMARY}
    bad = {m: {"manipulation_check_passed": False, "discrimination_holds": False} for m in P.CO_PRIMARY}
    assert P.verdicts({m: null for m in P.CO_PRIMARY}, ok)["overall"] == "no effect larger than 0.5 nats (both models)"
    v = P.verdicts({m: null for m in P.CO_PRIMARY}, bad)
    assert v["per_model"]["qwen3-8b-fewshot"]["verdict"].startswith("inconclusive (null") and v["overall"] == "inconclusive"
    good = {"estimate": 1.0, "ci": [0.6, 1.4], "perm_p": 0.004}
    assert "not read as the task overriding" in P.verdicts({m: good for m in P.CO_PRIMARY}, bad)["per_model"]["qwen3-8b-fewshot"]["reading"]
    assert "can see" in P.verdicts({m: good for m in P.CO_PRIMARY}, ok)["per_model"]["qwen3-8b-fewshot"]["reading"]


@pytest.mark.parametrize("est,ci,want", [(1.2, [0.3, 2.0], True), (0.9, [0.3, 1.5], False), (1.5, [-0.1, 3.0], False),
                                         (float("nan"), [0, 1], False)])
def test_check_threshold(est, ci, want):
    assert P.check_passes({"estimate": est, "ci": ci, "n_items": 10}) is want
    assert P.check_passes({"estimate": 2.0, "ci": [1, 3], "n_items": 0}) is False


def test_flip_shares_count_both_directions_and_product_clusters():
    items = [{"id": f"i{k}", "user_id": "u", "target_product_id": f"p{k % 2}"} for k in range(4)]
    L = {"i0": (1.0, -1.0), "i1": (-1.0, 1.0), "i2": (1.0, 2.0), "i3": (-2.0, -1.0)}
    by = {}
    for t in items:
        a, s = L[t["id"]]
        by[(t["id"], "main", "fail", "first", "aligned")] = {"L": a}
        by[(t["id"], "main", "fail", "first", "swapped")] = {"L": s}
    f = P.flip_shares(by, items)
    assert (f["toward_success"], f["toward_failure"], f["either"]) == (0.25, 0.25, 0.5)
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
