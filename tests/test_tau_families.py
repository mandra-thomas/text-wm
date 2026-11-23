"""τ-bench retail, wave-A model families (wf/tau/families.py; notes/tau_families_design.md): the protocol's jobs,
prompts with each model's template handling, the token check, the gates, mock scoring end to end, the per-model
verdict with Holm over the scored models and the competence gate, the descriptive comparisons, the placement check
and the smoke projection."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from tests.test_families import FakeTok
from wf import constants as C
from wf import families as F
from wf.runs import RUNNERS, load_config
from wf.tau import families as T
from wf.tau import pipeline as P
from wf.tau.prompts import OUTPUT_CUE, TARGET_QUESTION, YN_QUESTION, render_call
from wf.util import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
GOLDEN = ROOT / "tests" / "golden" / "tau_families_rendered.json"


class TauTok(FakeTok):
    """The families stand-in tokenizer, with '{"' and 'Error' split off the front of a word, as the real tokenizers
    split a JSON reply and an error reply."""

    def _words(self, text: str) -> list[str]:
        out = []
        for w in super()._words(text):
            if w.startswith('{"') and w != '{"':
                out += ['{"', w[2:]]
            elif w.startswith("Error") and w != "Error":
                out += ["Error", w[5:]]
            else:
                out.append(w)
        return out


# ---------------- fixtures ----------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A built and verified small τ item set in a temp runs folder (mock mode)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("tau_family_runs")
    for k in MOCK_KEYS:
        mp.setenv(k, "1")
    mp.setenv("RUNS_DIR", str(root))
    mp.setenv("WF_MOCK_CONTINUE", "1")
    (root / "FILED.txt").write_text("filed_date: x\n")
    assert P.run_tau_build({"run_id": "tau_build", "natural_required": False, "max_users": 15}) == 0
    assert P.run_tau_verify({"run_id": "tau_verify", "fresh_sample": 5}) == 0
    yield root
    mp.undo()


def _pins():
    d = P.runs_root() / "tau_build"
    return {n: P.file_sha(d / n) for n in P.OUTPUT_FILES}


def _cfg(name: str, **kw) -> dict:
    """A real config of this extension, pinned to the small build."""
    base = load_config(name)
    if "expected_outputs_sha256" in base:
        base["expected_outputs_sha256"] = _pins()
    return {**base, **kw}


def _primary() -> list[dict]:
    return [t for t in read_jsonl(P.runs_root() / "tau_build" / "items.jsonl") if P.is_primary(t)]


def _fewshot() -> list[dict]:
    return json.loads((P.runs_root() / "tau_build" / "fewshot.json").read_text())


@pytest.fixture(scope="module")
def token_checked(runs):
    """The token check run in mock mode with the stand-in tokenizer (one process)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(T, "_load_tokenizer", lambda spec: TauTok())
    assert T.run_tau_family_token_check(_cfg("tau_family_token_check", tokenize_in_mock=True, workers=1, placement_items=4)) == 0
    yield runs
    mp.undo()


# ---------------- registry and configs ----------------


def test_runs_and_configs_are_registered_and_pinned_alike():
    for name in ("tau_family_token_check", "tau_family_score_smoke", "tau_family_score_mistral-7b-v0.3",
                 "tau_family_score_llama3.2-3b", "tau_family_placement_mistral-7b-v0.3", "tau_family_smoke_analysis",
                 "tau_family_analysis"):
        assert name in RUNNERS
    pins = load_config("tau_score")["expected_outputs_sha256"]
    for name in ("tau_family_token_check", "tau_family_score_smoke", "tau_family_score_mistral-7b-v0.3",
                 "tau_family_score_llama3.2-3b", "tau_family_placement_mistral-7b-v0.3"):
        cfg = load_config(name)
        assert cfg["expected_outputs_sha256"] == pins and cfg["verify_run"] == "tau_verify", name
    assert load_config("tau_family_token_check")["models"] == list(T.SCORED)
    assert load_config("tau_family_token_check")["length_only_models"] == ["gemma2-9b-it"]
    for tag in T.SCORED:
        cfg = load_config(f"tau_family_score_{tag}")
        assert cfg["models"] == [tag] and cfg["n_shards"] == 4 and not cfg.get("limit_items")
    place = load_config("tau_family_placement_mistral-7b-v0.3")
    assert place["placement"] is True and place["n_items"] == 300 and place["models"] == ["mistral-7b-v0.3"]
    smoke = load_config("tau_family_score_smoke")
    assert smoke["limit_items"] == 20 and smoke["add_longest"] and smoke["models"] == list(T.SCORED)
    ana = load_config("tau_family_analysis")
    assert ana["reference_commits"] == ["c737cd6"] and ana["published_metrics"] == "notes/tau_metrics.json"
    assert ana["placement_items"] == 300 and ana["models"] == list(T.SCORED)


def test_models_contexts_and_the_excluded_gemma():
    assert T.SCORED == ("mistral-7b-v0.3", "llama3.2-3b")
    assert set(T.EXCLUDED) == {"gemma2-9b-it"} and "8,192" in T.EXCLUDED["gemma2-9b-it"]
    assert T.CONTEXT == {"mistral-7b-v0.3": 32768, "llama3.2-3b": 131072, "gemma2-9b-it": 8192}
    with pytest.raises(ValueError, match="excluded"):
        T.spec_for("gemma2-9b-it")
    assert T.spec_for("gemma2-9b-it", allow_excluded=True).system == "fold"
    assert T.spec_for("mistral-7b-v0.3").system == "native"
    assert T.spec_for("mistral-7b-v0.3" + T.FIRST_TURN).system == "fold"
    assert T.spec_for("llama3.2-3b").template_kwargs == {"date_string": "26 Jul 2024"}
    for tag in T.SCORED:
        assert re.fullmatch(r"[0-9a-f]{40}", T.spec_for(tag).revision)


def test_the_smoke_limit_equals_the_job_script_limit():
    sbatch = (ROOT / "pace" / "tau_family_score.sbatch").read_text()
    hours = int(re.search(r"#SBATCH --time=(\d+):00:00", sbatch).group(1))
    assert load_config("tau_family_smoke_analysis")["time_limit_hours"] == hours
    assert "gpu-a40" not in sbatch and "(gpu-l40s|gpu-a100|gpu-h100)" in sbatch


# ---------------- jobs and prompts ----------------


def test_protocol_is_the_designs_sixteen_jobs():
    assert len(T.PROTOCOL_JOBS) == 16 and len(set(T.PROTOCOL_JOBS)) == 16
    want = ({("fail", "first", c) for c in ("aligned", "swapped", "neutral", "mention_aligned", "mention_swapped")}
            | {("fail", "yn", c) for c in ("aligned", "swapped", "neutral")}
            | {("fail", "target", c) for c in ("aligned", "swapped")}
            | {("witness", r, c) for r in ("first", "yn") for c in ("aligned", "swapped", "neutral")})
    assert {j[1:] for j in T.PROTOCOL_JOBS} == want and {j[0] for j in T.PROTOCOL_JOBS} == {"main"}
    assert T.PLACEMENT_JOBS == (("main", "fail", "first", "aligned"), ("main", "fail", "first", "swapped"))
    assert T.READOUT_WORDS == {"first": ('{"', "Error"), "yn": ("Yes", "No"), "target": ("Yes", "No")}


@pytest.mark.parametrize("field", list(T.REQUIRED_FIELDS))
def test_an_item_missing_a_scored_line_or_its_witness_is_refused(runs, field):
    t = {**_primary()[0], field: None}
    with pytest.raises(ValueError, match="missing"):
        T.jobs_for(t)


def test_each_job_has_its_cue_call_and_template_handling(runs):
    t, few = _primary()[0], _fewshot()
    native, fold = T.spec_for("mistral-7b-v0.3"), T.spec_for("mistral-7b-v0.3" + T.FIRST_TURN)
    for _, action, readout, cond in T.PROTOCOL_JOBS:
        m = T.messages(native, t, cond, few, readout, action)
        assert m[0]["role"] == "system" and m == P.build_messages(t, cond, few, cue=T.CUE[readout], action=action)
        end = {"first": OUTPUT_CUE, "yn": YN_QUESTION, "target": TARGET_QUESTION}[readout]
        assert m[-1]["content"].endswith(end)
        call = t["action"] if action == "fail" else t["aligned_witness"]
        assert f"Next call: {render_call(call['name'], call['arguments'])}\n" in m[-1]["content"]
        f = T.messages(fold, t, cond, few, readout, action)
        assert f == F.fold_system(m) and f[0]["role"] == "user"
    assert T.messages(T.spec_for("llama3.2-3b"), t, "aligned", few) == P.build_messages(t, "aligned", few)


def test_smoke_items_and_placement_items(runs):
    prim = _primary()
    cfg = _cfg("tau_family_score_smoke", limit_items=3)
    smoke = T.primary_items(cfg)
    longest = max(prim, key=lambda t: (P.prompt_chars(t), t["id"]))
    assert longest in smoke and 3 <= len(smoke) <= 4 and all(P.is_primary(t) for t in smoke)
    assert len(T.primary_items(_cfg("tau_family_score_mistral-7b-v0.3"))) == len(prim)
    a = T.placement_items(prim, 5)
    assert a == T.placement_items(list(reversed(prim)), 5) and len(a) == 5
    assert [t["id"] for t in a] == sorted(t["id"] for t in a)
    import random
    pool = sorted(prim, key=lambda t: t["id"])
    assert {t["id"] for t in a} == {t["id"] for t in random.Random(f"{C.RNG_SEED}|tau_family_placement").sample(pool, 5)}


# ---------------- token check ----------------


def _check(tok=None, **kw):
    return T.check_model("mistral-7b-v0.3", tok or TauTok(), _primary()[:3], _fewshot(), **kw)


def test_token_check_passes_with_template_tokens(runs):
    rec = _check(placement=_primary()[:2])
    assert rec["passed"], rec["examples"]
    assert rec["n_prompts"] == 3 * 16 and rec["n_replies_checked"] == 12 and rec["n_yes_no_prompts_checked"] == 3 * 8
    assert rec["placement"]["n_prompts"] == 4 and rec["placement"]["n_problems"] == 0
    assert set(rec["prompt_tokens"]) == {"fail/first", "fail/yn", "fail/target", "witness/first", "witness/yn"}


def test_token_check_fails_when_a_reply_opens_with_another_token(runs):
    rec = _check(TauTok(odd_reply_after=(("Tool output:", '{"'),)))
    assert not rec["passed"] and rec["problems"]["first_token"] > 0


def test_token_check_fails_when_the_error_reply_opens_with_another_token(runs):
    rec = _check(TauTok(odd_reply_after=(("Tool output:", "Error"),)))
    assert not rec["passed"] and rec["problems"]["first_token"] > 0


def test_token_check_fails_when_yes_is_not_the_templates_token_after_a_question(runs):
    rec = _check(TauTok(odd_reply_after=(("is this call on the order", "Yes"),)))
    assert not rec["passed"] and rec["problems"]["yes_no"] > 0
    rec = _check(TauTok(yes_pieces=2))
    assert not rec["passed"] and rec["problems"]["single_tokens"] > 0


def test_token_check_fails_at_the_context_limit(runs, monkeypatch):
    rec = _check()
    n = rec["longest_prompt"][0]
    monkeypatch.setitem(T.CONTEXT, "mistral-7b-v0.3", n)
    rec = _check()
    assert not rec["passed"] and rec["problems"]["context"] >= 1
    monkeypatch.setitem(T.CONTEXT, "mistral-7b-v0.3", n + 1)
    assert _check()["passed"]


def test_token_check_fails_when_prompts_differ_outside_the_task_line(runs, monkeypatch):
    real = T.build_messages

    def leaky(t, cond, few, **kw):
        m = real(t, cond, few, **kw)
        if cond == "swapped" and kw.get("cue") == "yn":
            m[1] = {**m[1], "content": m[1]["content"] + " leak"}
        return m
    monkeypatch.setattr(T, "build_messages", leaky)
    rec = _check()
    assert not rec["passed"] and rec["problems"]["only_task_line"] == 6  # failing call and witness, 3 items


def test_token_check_fails_on_a_boundary_merge(runs):
    class Merging(TauTok):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
            if messages[-1]["role"] == "assistant" and messages[-1]["content"].startswith("Error"):
                messages = list(messages[:-1]) + [{"role": "assistant", "content": messages[-1]["content"].replace(": ", ":", 1)}]
            return super().apply_chat_template(messages, tokenize, add_generation_prompt, **kw)
    rec = _check(Merging())
    assert not rec["passed"] and rec["problems"]["boundary"] > 0 and rec["problems"]["first_token"] == 0


def test_gemma_is_checked_for_lengths_only(runs):
    rec = T.check_model("gemma2-9b-it", TauTok(), _primary()[:2], _fewshot(), length_only=True)
    assert rec["length_only"] and rec["passed"] is None and rec["n_prompts"] == 32 and "excluded" in rec["excluded"]


def test_token_check_run_writes_a_passing_report_for_the_scored_models(token_checked):
    rep = json.loads((token_checked / "tau_family_token_check" / "token_check.json").read_text())
    assert rep["passed"] and rep["files"] == _pins() and rep["n_items"] == len(_primary())
    assert rep["models"]["mistral-7b-v0.3"]["passed"] and rep["models"]["llama3.2-3b"]["passed"]
    assert rep["models"]["mistral-7b-v0.3"]["placement"]["n_items"] == 4 and rep["models"]["llama3.2-3b"]["placement"] is None
    assert rep["models"]["gemma2-9b-it"]["length_only"]
    for tag in T.SCORED:
        assert T.token_check_ok(_cfg(f"tau_family_score_{tag}"), tag) == (True, "")
    assert not T.token_check_ok(_cfg("tau_family_score_smoke"), "gemma2-9b-it")[0]


def test_token_check_gate_refuses_failures_and_other_files(token_checked, monkeypatch, tmp_path):
    path = token_checked / "tau_family_token_check" / "token_check.json"
    saved = path.read_text()
    try:
        rep = json.loads(saved)
        rep["models"]["llama3.2-3b"]["passed"] = False
        path.write_text(json.dumps(rep))
        assert not T.token_check_ok(_cfg("tau_family_score_llama3.2-3b"), "llama3.2-3b")[0]
        assert T.token_check_ok(_cfg("tau_family_score_mistral-7b-v0.3"), "mistral-7b-v0.3")[0]
        rep = json.loads(saved)
        rep["files"]["items.jsonl"] = "0" * 64
        path.write_text(json.dumps(rep))
        assert "other item files" in T.token_check_ok(_cfg("tau_family_score_llama3.2-3b"), "llama3.2-3b")[1]
        rep = json.loads(saved)
        rep["models"]["llama3.2-3b"] = {"skipped": "mock (no tokenizer)", "passed": None}
        path.write_text(json.dumps(rep))
        assert T.token_check_ok(_cfg("tau_family_score_llama3.2-3b"), "llama3.2-3b")[0]  # mock: a skip is accepted
        monkeypatch.setattr(T, "is_mock", lambda: False)
        assert not T.token_check_ok(_cfg("tau_family_score_llama3.2-3b"), "llama3.2-3b")[0]  # never in a real run
    finally:
        path.write_text(saved)


def test_scoring_refuses_without_a_passed_token_check(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(T, "token_check_ok", lambda cfg, tag: (False, "no"))
    assert T.run_tau_family_score(_cfg("tau_family_score_llama3.2-3b", run_id="tf_refused")) == 1
    assert "token check" in (runs / "tf_refused" / "STATUS.md").read_text()


def test_scoring_stops_when_the_in_job_readout_report_fails(token_checked, monkeypatch):
    monkeypatch.setattr(T, "readout_report", lambda scorer, spec, t, few: {"ok": False})
    assert T.run_tau_family_score(_cfg("tau_family_score_llama3.2-3b", run_id="tf_readout", n_shards=1)) == 1
    d = token_checked / "tf_readout" / "shard_00_of_01"
    assert "not the template's" in (d / "STATUS.md").read_text() and not read_jsonl(d / "scoring.jsonl")


def test_token_check_run_fails_when_a_model_fails(runs, monkeypatch):
    monkeypatch.setattr(T, "_load_tokenizer", lambda spec: TauTok(yes_pieces=2))
    cfg = _cfg("tau_family_token_check", run_id="tf_tc_fail", tokenize_in_mock=True, workers=1, placement_items=2)
    assert T.run_tau_family_token_check(cfg) == 1
    rep = json.loads((runs / "tf_tc_fail" / "token_check.json").read_text())
    assert rep["passed"] is False and rep["models"]["llama3.2-3b"]["passed"] is False


def test_scoring_refuses_gemma_and_unknown_models(token_checked):
    for tag in ("gemma2-9b-it", "qwen3-8b"):
        assert T.run_tau_family_score(_cfg("tau_family_score_smoke", run_id="tf_bad", models=[tag])) == 1
        assert "not a scored model" in (token_checked / "tf_bad" / "STATUS.md").read_text()


@pytest.mark.parametrize("name", P.OUTPUT_FILES)
def test_scoring_refuses_unpinned_or_changed_items(token_checked, name):
    cfg = _cfg("tau_family_score_llama3.2-3b", run_id="tf_pins")
    cfg["expected_outputs_sha256"] = {**cfg["expected_outputs_sha256"], name: "0" * 64}
    assert T.run_tau_family_score(cfg) == 1
    assert "not verified" in (token_checked / "tf_pins" / "STATUS.md").read_text()


def test_scoring_refuses_a_prompt_at_the_context_limit(token_checked, monkeypatch):
    monkeypatch.setattr(T, "prompt_tokens", lambda scorer, msgs: 10 ** 6)
    with pytest.raises(SystemExit, match="context"):
        T.run_tau_family_score(_cfg("tau_family_score_llama3.2-3b", run_id="tf_long", n_shards=1))
    T.check_prompt_length(131071, "llama3.2-3b")
    with pytest.raises(SystemExit):
        T.check_prompt_length(131072, "llama3.2-3b")
    with pytest.raises(SystemExit):
        T.check_prompt_length(32768, "mistral-7b-v0.3" + T.FIRST_TURN)


def test_in_job_readout_report_checks_the_template(runs):
    t, few = _primary()[0], _fewshot()
    spec = T.spec_for("mistral-7b-v0.3")

    def scorer(tok):
        s = T._scorer_for(spec, tok)
        s.mock = False
        return s
    assert T.readout_report(scorer(TauTok()), spec, t, few)["ok"]
    assert not T.readout_report(scorer(TauTok(odd_reply_after=(("Tool output:", '{"'),))), spec, t, few)["ok"]
    assert not T.readout_report(scorer(TauTok(odd_reply_after=(("Question:", "No"),))), spec, t, few)["ok"]
    assert not T.readout_report(scorer(TauTok(odd_reply_after=(("Tool output:", "Error"),))), spec, t, few)["ok"]
    assert not T.readout_report(scorer(TauTok(yes_pieces=2)), spec, t, few)["ok"]


# ---------------- mock scoring end to end ----------------


@pytest.fixture(scope="module")
def scored(token_checked):
    """Both scored models in 2 shards, the placement check (4 items), a mock reference τ run with its own analysis."""
    mp = pytest.MonkeyPatch()
    for tag in T.SCORED:
        for i in (0, 1):
            mp.setenv("WF_SHARD", str(i))
            assert T.run_tau_family_score(_cfg(f"tau_family_score_{tag}", n_shards=2)) == 0
    mp.delenv("WF_SHARD")
    assert T.run_tau_family_score(_cfg("tau_family_placement_mistral-7b-v0.3", n_items=4)) == 0
    ref = {**load_config("tau_score"), "run_id": "tau_score_reference", "n_shards": 2, "expected_outputs_sha256": _pins()}
    for i in (0, 1):
        mp.setenv("WF_SHARD", str(i))
        assert P.run_tau_score(ref) == 0
    mp.delenv("WF_SHARD")
    assert P.run_tau_analysis({**load_config("tau_analysis"), "source": "tau_score_reference"}) == 0
    yield token_checked
    mp.undo()


def _analysis_cfg(root, **kw):
    return {**load_config("tau_family_analysis"), "reference_commits": [], "placement_items": 4,
            "published_metrics": str(root / "tau_analysis" / "metrics.json"), **kw}


def test_scoring_writes_every_job_once_per_model(scored):
    prim = _primary()
    for tag in T.SCORED:
        recs = [r for i in (0, 1) for r in read_jsonl(scored / f"tau_family_score_{tag}" / f"shard_{i:02d}_of_02" / "scoring.jsonl")]
        assert len(recs) == 16 * len(prim) and {r["model"] for r in recs} == {tag}
        keys = [T.score_key(r) for r in recs]
        assert len(set(keys)) == len(keys)
        assert {k[2:] for k in keys} == set(T.PROTOCOL_JOBS)
        assert all(r["item_hash"] == P.item_hash(next(t for t in prim if t["id"] == r["transition_id"])) for r in recs)
        assert all("prompt_tokens" in r and r["seconds"] >= 0 for r in recs)
        users = [{r["user_id"] for r in read_jsonl(scored / f"tau_family_score_{tag}" / f"shard_{i:02d}_of_02" / "scoring.jsonl")} for i in (0, 1)]
        assert not users[0] & users[1]
    place = read_jsonl(scored / "tau_family_placement_mistral-7b-v0.3" / "shard_00_of_01" / "scoring.jsonl")
    assert len(place) == 8 and {r["model"] for r in place} == {"mistral-7b-v0.3-firstturn"}
    assert {(r["readout"], r["condition"]) for r in place} == {("first", "aligned"), ("first", "swapped")}
    assert {r["transition_id"] for r in place} == {t["id"] for t in T.placement_items(prim, 4)}


def test_resume_finishes_without_duplicates(scored, monkeypatch):
    d = scored / "tau_family_score_llama3.2-3b" / "shard_00_of_02"
    before = read_jsonl(d / "scoring.jsonl")
    monkeypatch.setenv("WF_SHARD", "0")
    assert T.run_tau_family_score(_cfg("tau_family_score_llama3.2-3b", n_shards=2, resume=True)) == 0
    assert read_jsonl(d / "scoring.jsonl") == before


def test_analysis_end_to_end(scored):
    assert T.run_tau_family_analysis(_analysis_cfg(scored)) == 0
    m = json.loads((scored / "tau_family_analysis" / "metrics.json").read_text())
    prim = _primary()
    assert m["missing_scores"] == {t: 0 for t in T.SCORED} and m["stale_scores"] == {t: 0 for t in T.SCORED}
    assert m["items"]["primary"] == len(prim) and set(m["excluded"]) == {"gemma2-9b-it"}
    assert set(m["models"]) == set(T.SCORED) and set(m["reference"]) == set(P.CO_PRIMARY)
    assert m["reference_run"]["missing"] == 0 and m["reference_run"]["stale"] == 0
    assert m["reference_check"]["all_equal"] is True
    blk = m["models"]["mistral-7b-v0.3"]
    assert blk["primary"]["first"]["aligned_minus_swapped"]["n_items"] == len(prim)
    assert blk["primary_controls"]["discrimination"]["first"]["n_items"] == sum(t["witness_checker"] == "PASS" for t in prim)
    assert blk["competence"]["first"]["n_failures"] == sum(t["witness_checker"] == "PASS" for t in prim)
    assert blk["competence"]["first_all_items"]["n_failures"] == len(prim)
    assert set(m["verdicts"]["per_model"]) == set(T.SCORED) and "overall" not in m["verdicts"]
    assert m["verdicts"]["family"] == list(T.SCORED)
    assert set(m["comparisons"]) == {"mistral-7b-v0.3 vs qwen3-8b-fewshot", "mistral-7b-v0.3 vs llama3.1-8b-fewshot",
                                     "llama3.2-3b vs qwen3-8b-fewshot", "llama3.2-3b vs llama3.1-8b-fewshot",
                                     "mistral-7b-v0.3 vs llama3.2-3b"}
    assert m["comparisons"]["mistral-7b-v0.3 vs llama3.2-3b"]["first"]["n_items"] == len(prim)
    # every comparison entry says whether it is read (R1-S6): the analysis marks them
    assert all("read" in c and "not_read_because" in c for v in m["comparisons"].values() for c in v.values())
    assert m["placement"]["n_items"] == 4 and "reading" in m["placement"]
    assert set(m["sensitivity_without_fewshot_tool_classes"]) == set(T.SCORED)
    assert "REPORTED (τ families verdicts)" in (scored / "tau_family_analysis" / "STATUS.md").read_text()


def test_analysis_refuses_a_reference_run_from_another_commit(scored):
    assert T.run_tau_family_analysis(_analysis_cfg(scored, run_id="tf_ana_commit", reference_commits=["deadbee"])) == 1
    m = json.loads((scored / "tf_ana_commit" / "metrics.json").read_text())
    assert any("not ['deadbee']" in p for p in m["reference_run"]["problems"]) and m["reference"] == {}


def test_analysis_flags_a_reference_result_that_differs_from_the_published_one(scored, tmp_path):
    pub = json.loads((scored / "tau_analysis" / "metrics.json").read_text())
    pub["models"]["qwen3-8b-fewshot"]["primary"]["first"]["aligned_minus_swapped"]["estimate"] += 0.001
    (tmp_path / "pub.json").write_text(json.dumps(pub))
    assert T.run_tau_family_analysis(_analysis_cfg(scored, run_id="tf_ana_pub", published_metrics=str(tmp_path / "pub.json"))) == 1
    chk = json.loads((scored / "tf_ana_pub" / "metrics.json").read_text())["reference_check"]
    assert chk["models"]["qwen3-8b-fewshot"]["equal"] is False and chk["models"]["llama3.1-8b-fewshot"]["equal"] is True


def test_analysis_counts_stale_reference_records(scored):
    d = scored / "tau_score_reference" / "shard_00_of_02"
    saved = (d / "scoring.jsonl").read_text()
    try:
        recs = read_jsonl(d / "scoring.jsonl")
        prim = {t["id"] for t in _primary()}
        k = next(i for i, r in enumerate(recs) if r["transition_id"] in prim and T.score_key(r)[2:] in set(T.PROTOCOL_JOBS))
        recs[k]["item_hash"] = "x"
        (d / "scoring.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        assert T.run_tau_family_analysis(_analysis_cfg(scored, run_id="tf_ana_stale")) == 1
        m = json.loads((scored / "tf_ana_stale" / "metrics.json").read_text())
        assert m["reference_run"]["stale"] == 1
    finally:
        (d / "scoring.jsonl").write_text(saved)


def test_analysis_waits_for_a_missing_shard(scored):
    d = scored / "tau_family_score_llama3.2-3b"
    (d / "shard_01_of_02").rename(d / "hidden_shard")
    try:
        assert T.run_tau_family_analysis(_analysis_cfg(scored, run_id="tf_ana_wait")) == 1
        assert "waits for every scoring shard" in (scored / "tf_ana_wait" / "STATUS.md").read_text()
    finally:
        (d / "hidden_shard").rename(d / "shard_01_of_02")


def test_smoke_run_and_projection_report_no_estimate(token_checked):
    assert T.run_tau_family_score(_cfg("tau_family_score_smoke", limit_items=3)) == 0
    assert T.run_tau_family_analysis(_cfg("tau_family_smoke_analysis")) == 0
    m = json.loads((token_checked / "tau_family_smoke_analysis" / "metrics.json").read_text())

    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    found = set(keys(m))
    assert not found & {"estimate", "perm_p", "verdicts", "verdict", "aligned_minus_swapped", "comparisons"}
    pr = m["projection"]
    assert set(pr["models"]) == set(T.SCORED) and pr["time_limit_hours"] == 3 and pr["budget_gpu_hours"] == 6.5
    assert pr["models"]["mistral-7b-v0.3"]["hours_placement"] > 0 and pr["models"]["llama3.2-3b"]["hours_placement"] == 0
    assert len(pr["models"]["llama3.2-3b"]["hours_per_shard"]) == 4
    assert "ΔL" not in (token_checked / "tau_family_smoke_analysis" / "STATUS.md").read_text()


def test_projection_arithmetic(runs, monkeypatch):
    prim = _primary()
    sec = {"first": 2.0, "yn": 3.0, "target": 5.0}
    report = {"models": {"mistral-7b-v0.3": {"format": {"seconds_mean_by_readout": sec, "seconds_total": 3600.0},
                                             "sd_of_differences": {"first": {"sd": 1.0, "n": 5}}}}}
    pr = T.projection({"time_limit_hours": 3}, report, prim)["models"]["mistral-7b-v0.3"]
    per_item = (3 * 2 + 3 * 3 + 2 * 2 + 2 * 5 + 3 * 2 + 3 * 3) / 3600  # first 8 jobs, yn 6, target 2
    shards = [P.shard_by_user(prim, i, 4) for i in range(4)]
    assert pr["hours_per_shard"] == pytest.approx([len(s) * per_item for s in shards])
    assert pr["hours_max_shard"] == pytest.approx(max(len(s) for s in shards) * per_item)
    assert pr["hours_placement"] == pytest.approx(len(T.placement_items(prim)) * 2 * 2.0 / 3600)
    assert pr["gpu_hours_total"] == pytest.approx(len(prim) * per_item + pr["hours_placement"] + 1.0)
    assert pr["within_budget"] == (pr["gpu_hours_total"] <= 6.5)
    report["models"]["mistral-7b-v0.3"]["format"]["seconds_mean_by_readout"] = {"first": 4000.0, "yn": 1.0, "target": 1.0}
    pr = T.projection({"time_limit_hours": 3}, report, prim)
    assert not pr["fits_time_limit"] and not pr["within_budget"]


# ---------------- verdicts, competence, shares, comparisons (constructed scores) ----------------


def _rec(L: float) -> dict:
    return {"L": L, "logp_plus": -math.log1p(math.exp(-L)), "logp_minus": -math.log1p(math.exp(L))}


def _items(n: int, users: int = 10, pass_every: int = 1) -> list[dict]:
    return [{"id": f"i{k}", "user_id": f"u{k % users}", "cls": "unavailable", "target_product_id": f"p{k % 3}",
             "witness_checker": "PASS" if k % pass_every == 0 else "UNKNOWN"} for k in range(n)]


def _by(items, f):
    """`f(item index, action, readout, condition)` -> L, for every protocol job."""
    return {(t["id"], "main", action, readout, cond): _rec(f(k, action, readout, cond))
            for k, t in enumerate(items) for _, action, readout, cond in T.PROTOCOL_JOBS}


def test_competence_reads_the_neutral_witness_against_the_neutral_failing_call_on_pass_items():
    items = _items(40, pass_every=2)

    def f(k, action, readout, cond):
        if cond != "neutral":
            return 50.0 if action == "witness" else -50.0  # other conditions must not matter
        if items[k]["witness_checker"] != "PASS":
            return -9.0 if action == "witness" else 9.0  # the non-PASS items would reverse it
        return (1.0 + k * 0.01) if action == "witness" else (-1.0 - k * 0.01)
    by = _by(items, f)
    c = T.competence(by, items, "first")
    assert c["n_successes"] == c["n_failures"] == 20
    assert c["checks"]["auc_neutral"]["value"] == 1.0 and c["interpretable"]
    assert c["checks"]["yes_share_neutral_failures"]["value"] == 0.0
    assert c["checks"]["yes_share_neutral_successes"]["value"] == 1.0
    allc = T.competence(by, items, "first", population="all")
    assert allc["n_failures"] == 40 and allc["checks"]["auc_neutral"]["value"] < 1.0
    yn = T.competence(by, items, "yn")
    assert yn["readout"] == "yn" and yn["interpretable"]


@pytest.mark.parametrize("w,fl,check", [
    (lambda k: 0.1 * (k % 2), lambda k: 0.1 * ((k + 1) % 2), "auc_neutral"),  # AUC 0.5
    (lambda k: 3.0, lambda k: 2.0, "yes_share_neutral_failures"),  # Yes to every failure
    (lambda k: -2.0, lambda k: -3.0, "yes_share_neutral_successes"),  # No to every success
])
def test_competence_rule_fails_each_condition(w, fl, check):
    items = _items(30)
    by = _by(items, lambda k, a, r, c: w(k) if a == "witness" else fl(k))
    comp = T.competence(by, items, "first")
    assert not comp["interpretable"] and not comp["checks"][check]["pass"]


def test_competence_boundaries_and_mass():
    assert T._auc([1, 2], [0, 0]) == 1.0 and T._auc([1], [1]) == 0.5 and T._auc([0], [1]) == 0.0
    rule = F.COMPETENCE
    assert rule == {"auc_neutral_min": 0.70, "yes_share_neutral_failures_max": 0.95,
                    "yes_share_neutral_successes_min": 0.05, "median_yes_no_mass_neutral_min": 0.5}
    items = _items(20)
    by = _by(items, lambda k, a, r, c: 2.0 if a == "witness" else -2.0)
    for v in by.values():  # mass 0.4 < 0.5: fails on mass alone
        v["logp_plus"], v["logp_minus"] = math.log(0.2 if v["L"] > 0 else 0.2 * math.exp(-2)), math.log(0.2 * math.exp(-2) if v["L"] > 0 else 0.2)
    comp = T.competence(by, items, "first")
    assert not comp["checks"]["median_two_way_mass_neutral"]["pass"] and comp["checks"]["auc_neutral"]["pass"]


def test_item_share_counts_ties_as_not_moved_and_is_directional():
    items = _items(10)
    by = _by(items, lambda k, a, r, c: {"aligned": 1.0 if k < 6 else 0.0, "swapped": 0.0}.get(c, 0.0))
    s = T.item_share(by, items, "first")
    assert s["estimate"] == 0.6 and s["null"] == 0.5 and s["n_items"] == 10
    by = _by(items, lambda k, a, r, c: {"aligned": 0.0, "swapped": 1.0}.get(c, 0.0))
    assert T.item_share(by, items, "first")["estimate"] == 0.0


def _block(est, ci, p, *, manip=True, disc=True, fmw=(0.5, [0.2, 0.8]), wit=(0.0, [-0.1, 0.1]), comp=True, comp_yn=True):
    mk = lambda e, c: {"estimate": e, "ci": c, "n_items": 50, "perm_p": 0.001}
    return {"primary": {"first": {"aligned_minus_swapped": {"estimate": est, "ci": ci, "perm_p": p, "n_items": 50}}},
            "checks": {"manipulation_check_passed": manip, "discrimination_holds": disc, "fail_minus_witness": mk(*fmw),
                       "write_minus_mention": mk(0.3, [0.1, 0.5]), "witness_effect": mk(*wit)},
            "competence": {"first": {"interpretable": comp}, "yn": {"interpretable": comp_yn}}}


def test_holm_runs_over_the_two_scored_models_at_a_fixed_size():
    blocks = {"mistral-7b-v0.3": _block(1.0, [0.5, 1.5], 0.006), "llama3.2-3b": _block(1.0, [0.5, 1.5], 0.006)}
    v = T.family_verdicts(blocks)["per_model"]
    assert all(x["holm_p"] == pytest.approx(0.012) and not x["verdict"].startswith("replicates") for x in v.values())
    blocks = {"mistral-7b-v0.3": _block(1.0, [0.5, 1.5], 0.004), "llama3.2-3b": _block(1.0, [0.5, 1.5], 0.004)}
    assert all(x["verdict"].startswith("replicates") for x in T.family_verdicts(blocks)["per_model"].values())
    only = T.family_verdicts({"mistral-7b-v0.3": _block(1.0, [0.5, 1.5], 0.006)})["per_model"]
    assert only["llama3.2-3b"]["verdict"] == "not scored" and only["mistral-7b-v0.3"]["holm_p"] == pytest.approx(0.012)
    assert T.family_verdicts({"mistral-7b-v0.3": _block(1.0, [0.5, 1.5], 0.004)})["per_model"]["mistral-7b-v0.3"]["verdict"].startswith("replicates")


def test_competence_gate_overrides_any_verdict_and_keeps_the_estimate():
    blocks = {"mistral-7b-v0.3": _block(1.0, [0.5, 1.5], 0.001, comp=False), "llama3.2-3b": _block(1.0, [0.5, 1.5], 0.001)}
    v = T.family_verdicts(blocks)["per_model"]
    assert v["mistral-7b-v0.3"]["verdict"] == T.NOT_INTERPRETABLE
    assert v["mistral-7b-v0.3"]["verdict_before_competence"].startswith("replicates")
    assert v["mistral-7b-v0.3"]["reading"].startswith("Not interpretable")
    assert v["llama3.2-3b"]["verdict"].startswith("replicates") and v["llama3.2-3b"]["competence_first_token_interpretable"]
    blocks["llama3.2-3b"] = _block(1.0, [0.5, 1.5], 0.001, comp_yn=False)  # the yes/no rule does not gate the verdict
    v = T.family_verdicts(blocks)["per_model"]
    assert v["llama3.2-3b"]["verdict"].startswith("replicates") and v["llama3.2-3b"]["competence_yes_no_interpretable"] is False


def test_tau_gates_and_reading_table_are_applied():
    blocks = {"mistral-7b-v0.3": _block(0.1, [-0.2, 0.3], 0.5, manip=False), "llama3.2-3b": _block(0.1, [-0.2, 0.3], 0.5, disc=False)}
    v = T.family_verdicts(blocks)["per_model"]
    assert v["mistral-7b-v0.3"]["verdict"] == "inconclusive (null, but the task line is not shown to be read)"
    assert v["llama3.2-3b"]["verdict"].startswith("inconclusive (null, but the model is not shown to discriminate")
    blocks = {"mistral-7b-v0.3": _block(1.0, [0.5, 1.5], 0.001, fmw=(0.0, [-0.1, 0.1]), wit=(1.0, [0.5, 1.5])),
              "llama3.2-3b": _block(1.0, [0.5, 1.5], 0.001)}
    v = T.family_verdicts(blocks)["per_model"]
    assert v["mistral-7b-v0.3"]["reading"].startswith("A general task-consistency bias")
    assert v["llama3.2-3b"]["reading"].startswith("The task overrides a visible fact")  # the τ table, unchanged
    fs, shared = {"failure_specific": True}, {"failure_specific": False}
    assert T.specificity(v["llama3.2-3b"], fs) == "failure-specific" and T.specificity(v["llama3.2-3b"], shared) == "shared"
    assert T.specificity(v["llama3.2-3b"], None) is None  # the cross-model label never comes from the τ table's row


def test_compare_is_a_minus_b_on_shared_items_with_paired_draws():
    items = _items(60, users=20)
    a = _by(items, lambda k, x, r, c: {"aligned": 1.0, "swapped": 0.0}.get(c, 0.0))  # A moves every item, by 1 nat
    b = _by(items, lambda k, x, r, c: {"aligned": 1.0 if k % 2 else 0.0, "swapped": 0.0}.get(c, 0.0))  # B half, by 0.5
    out = T.compare(a, b, items)
    s = out["item_share_difference"]
    assert s["estimate"] == pytest.approx(0.5) and s["differs"] and not s["matches"] and s["reading"] == "differs"
    assert out["delta_L_difference_nats"]["estimate"] == pytest.approx(0.5)
    assert out["delta_L_difference_nats"]["reading"] == "differs"
    assert T.compare(b, a, items)["item_share_difference"]["estimate"] == pytest.approx(-0.5)
    same = T.compare(a, a, items)
    assert same["item_share_difference"]["reading"] == "matches" and same["delta_L_difference_nats"]["reading"] == "no difference detected"
    del b[(items[0]["id"], "main", "fail", "first", "swapped")]
    assert T.compare(a, b, items)["n_items"] == 59


def test_compare_matches_on_the_90_percent_interval(monkeypatch):
    items = _items(20)
    a = _by(items, lambda k, x, r, c: {"aligned": 1.0, "swapped": 0.0}.get(c, 0.0))
    draws = [(-0.054 + 0.108 * i / 999, 0.0) for i in range(1000)]  # 5th/95th percentiles inside ±0.05, 2.5th/97.5th not
    monkeypatch.setattr(T, "_paired_draws", lambda rows, users, stat: draws)
    s = T.compare(a, a, items)["item_share_difference"]
    assert abs(s["ci90"][0]) < 0.05 and abs(s["ci95"][0]) > 0.05 and s["matches"] and not s["differs"]


def test_compare_noise_floor_and_paired_resampling():
    items = _items(60, users=20)
    a = _by(items, lambda k, x, r, c: {"aligned": 0.2 + 0.01 * (k % 3), "swapped": 0.0}.get(c, 0.0))
    b = _by(items, lambda k, x, r, c: {"aligned": 0.01 * (k % 3), "swapped": 0.0}.get(c, 0.0))
    out = T.compare(a, b, items)["delta_L_difference_nats"]
    assert out["estimate"] == pytest.approx(0.2) and out["ci95"][0] > 0  # excludes 0 ...
    assert out["reading"] == "no difference detected"  # ... but under the 0.3-nat floor
    # paired: A and B move exactly the same items (users differ in how many: user u has items u, u + 20 and u + 40,
    # and items below 30 move, so users 0-9 move two and users 10-19 one): resampling users once for both models gives
    # a share difference with no spread at all; resampling each model separately would not
    c = _by(items, lambda k, x, r, cond: {"aligned": 1.0 if k < 30 else -1.0, "swapped": 0.0}.get(cond, 0.0))
    d = _by(items, lambda k, x, r, cond: {"aligned": 2.0 if k < 30 else -1.0, "swapped": 0.0}.get(cond, 0.0))
    s = T.compare(c, d, items)["item_share_difference"]
    assert s["ci95"] == [0.0, 0.0] and s["matches"]


def test_placement_rule_needs_an_interval_off_zero_and_a_fifth_of_the_native_effect():
    items = _items(40, users=20)
    native = _by(items, lambda k, x, r, c: {"aligned": 2.0, "swapped": 0.0}.get(c, 0.0))
    first = lambda shift: {k: _rec(v["L"] + (shift(int(k[0][1:])) if k[4] == "aligned" else 0.0)) for k, v in native.items()}
    big = T.placement_block(native, first(lambda i: 0.5 + 0.01 * (i % 5)), items)
    assert big["first_turn_minus_native"]["estimate"] == pytest.approx(0.52) and big["format_dependent"]
    small = T.placement_block(native, first(lambda i: 0.3 + 0.01 * (i % 5)), items)  # off zero but under 2.0 / 5
    assert small["first_turn_minus_native"]["ci"][0] > 0 and not small["format_dependent"]
    neg = T.placement_block(native, first(lambda i: -0.5 - 0.01 * (i % 5)), items)
    assert neg["first_turn_minus_native"]["estimate"] < 0 and neg["format_dependent"]
    noisy = T.placement_block(native, first(lambda i: 1.0 if i % 2 else -1.0), items)
    assert not noisy["format_dependent"]
    assert big["native"]["estimate"] == pytest.approx(2.0) and big["first_turn"]["estimate"] == pytest.approx(2.52)


def _desc(fs=False, mix=False, minority=False):
    return {"failure_specific": fs, "class_mix": mix, "minority": minority, "label": "x"}


def test_pattern_readings():
    v = lambda verdict, ci=True, disc=True: {"verdict": verdict, "discrimination_holds": disc, "ci_above_0": ci}
    rep_, small = "replicates (ALFWorld direction)", "small effect, direction positive"
    shared = T.pattern_readings({"mistral-7b-v0.3": v(rep_), "llama3.2-3b": v(small)}, {},
                                {"mistral-7b-v0.3": _desc(), "llama3.2-3b": _desc()})
    assert any(r.startswith("The direction of the τ goal effect is not specific") and "qualified" not in r for r in shared["readings"])
    assert any(r.startswith("In both scored models the task shifts working and failing calls alike") for r in shared["readings"])
    both_fs = T.pattern_readings({"mistral-7b-v0.3": v(rep_), "llama3.2-3b": v(rep_)},
                                 {"llama3.2-3b vs llama3.1-8b-fewshot": {"item_share_difference": {"reading": "matches"}}},
                                 {"mistral-7b-v0.3": _desc(fs=True), "llama3.2-3b": _desc(fs=True)},
                                 {"llama3.1-8b-fewshot": {**_desc(), "label": "a shift shared by working and failing calls"}})
    assert any(r.startswith("Both scored models show a shift specific to failing calls") for r in both_fs["readings"])
    assert not any("Llama 3.2 3B has Llama-3.1-8B's label" in r for r in both_fs["readings"])  # labels differ
    split = T.pattern_readings({"mistral-7b-v0.3": v(rep_), "llama3.2-3b": v(rep_)}, {},
                               {"mistral-7b-v0.3": _desc(fs=True), "llama3.2-3b": _desc()})
    assert any(r.startswith("The pattern varies by model") for r in split["readings"])
    gated = T.pattern_readings({"mistral-7b-v0.3": v(T.NOT_INTERPRETABLE), "llama3.2-3b": v(rep_)}, {},
                               {"mistral-7b-v0.3": _desc(), "llama3.2-3b": _desc()})
    assert gated["readings"] == [] and gated["left_out"] == {"mistral-7b-v0.3": T.NOT_INTERPRETABLE}
    nul = T.pattern_readings({"mistral-7b-v0.3": v("no effect larger than 0.5 nats", ci=False), "llama3.2-3b": v(rep_)}, {},
                             {"mistral-7b-v0.3": _desc(), "llama3.2-3b": _desc()})
    assert nul["readings"] == [] and nul["labels"]["mistral-7b-v0.3"] is None


def test_pattern_direction_sentence_is_qualified_for_a_minority_or_a_failed_discrimination():
    v = lambda disc=True: {"verdict": "small effect, direction positive", "discrimination_holds": disc, "ci_above_0": True}
    out = T.pattern_readings({"mistral-7b-v0.3": v(), "llama3.2-3b": v(disc=False)}, {},
                             {"mistral-7b-v0.3": _desc(minority=True), "llama3.2-3b": _desc()})
    r = next(x for x in out["readings"] if x.startswith("The direction"))
    assert "mistral-7b-v0.3: carried by a minority of items" in r and "llama3.2-3b: not shown to use the visible record" in r
    assert out["labels"]["llama3.2-3b"] is None  # no cross-model label without discrimination (reviewer's R3)


def test_class_mix_and_llama_line_readings():
    v = {"verdict": "small effect, direction positive", "discrimination_holds": True, "ci_above_0": True}
    ref = {"llama3.1-8b-fewshot": _desc()}
    per = {"mistral-7b-v0.3": v, "llama3.2-3b": v}
    d = {"mistral-7b-v0.3": _desc(), "llama3.2-3b": _desc(mix=True)}
    for reading, want in (("matches", True), ("differs, within the margin", False), ("no difference detected", False)):
        out = T.pattern_readings(per, {"llama3.2-3b vs llama3.1-8b-fewshot": {"item_share_difference": {"reading": reading}}}, d, ref)
        assert any(r.startswith("Llama 3.2 3B has Llama-3.1-8B's label") for r in out["readings"]) == want, reading
        assert "llama3.2-3b: the pooled effect depends on the class mix; reported per class" in out["readings"]


def test_reference_check_compares_estimate_interval_and_n(tmp_path):
    pub = {"models": {m: {"primary": {"first": {"aligned_minus_swapped": {"estimate": 1.0, "ci": [0.5, 1.5], "n_items": 9}}}}
                      for m in P.CO_PRIMARY}}
    (tmp_path / "p.json").write_text(json.dumps(pub))
    blocks = {m: {"primary": {"first": {"aligned_minus_swapped": {"estimate": 1.0, "ci": [0.5, 1.5], "n_items": 9}}}} for m in P.CO_PRIMARY}
    assert T.reference_check(blocks, tmp_path / "p.json")["all_equal"]
    blocks["llama3.1-8b-fewshot"]["primary"]["first"]["aligned_minus_swapped"]["ci"] = [0.5, 1.6]
    assert not T.reference_check(blocks, tmp_path / "p.json")["all_equal"]
    blocks["llama3.1-8b-fewshot"]["primary"]["first"]["aligned_minus_swapped"]["ci"] = [0.5, 1.5]
    blocks["llama3.1-8b-fewshot"]["primary"]["first"]["aligned_minus_swapped"]["n_items"] = 8
    assert not T.reference_check(blocks, tmp_path / "p.json")["all_equal"]
    assert T.reference_check(blocks, None)["checked"] is False


def test_the_published_reference_metrics_are_committed():
    pub = json.loads((ROOT / "notes" / "tau_metrics.json").read_text())
    for m in P.CO_PRIMARY:
        blk = pub["models"][m]["primary"]["first"]["aligned_minus_swapped"]
        assert blk["n_items"] == 856 and len(blk["ci"]) == 2


# ---------------- golden prompts (one real primary item) ----------------


def test_golden_messages_are_unchanged():
    """Tokenizer-free: the messages of every job of one real primary item, per model handling, hash as recorded."""
    import hashlib

    g = json.loads(GOLDEN.read_text())
    for tag, variants in g["messages_sha256"].items():
        spec = T.spec_for(tag)
        for key, want in variants.items():
            action, readout, cond = key.split("/")
            msgs = T.messages(spec, g["item"], cond, g["fewshot"], readout, action)
            assert hashlib.sha256(json.dumps(msgs).encode()).hexdigest() == want, (tag, key)


@pytest.mark.parametrize("tag", list(T.SCORED) + ["gemma2-9b-it"])
def test_golden_rendered_prompts_with_the_pinned_tokenizers(tag):
    """Runs where transformers and the pinned tokenizers are cached (ICE); skipped elsewhere."""
    import hashlib

    transformers = pytest.importorskip("transformers")
    g = json.loads(GOLDEN.read_text())
    spec = T.spec_for(tag, allow_excluded=True)
    try:
        tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, local_files_only=True)
    except Exception as e:  # not cached here
        pytest.skip(f"{spec.hf_id} not available offline: {str(e)[:100]}")
    kw = {"enable_thinking": False, **spec.template_kwargs}
    for key, want in g["rendered"][tag].items():
        action, readout, cond = key.split("/")
        text = tok.apply_chat_template(T.messages(spec, g["item"], cond, g["fewshot"], readout, action), tokenize=False,
                                       add_generation_prompt=True, **kw)
        assert hashlib.sha256(text.encode()).hexdigest() == want["sha256"] and len(tok(text, add_special_tokens=False).input_ids) == want["tokens"]


# ---------------- amendments after review 1 ----------------


def _eff(est, lo, hi, n=50):
    return {"estimate": est, "ci": [lo, hi], "n_items": n}


def _blk(fail, wit, fmw, cls=None, share=(0.7, 0.62, 0.78)):
    """A model block with the fields `describe` reads: effects as (estimate, low, high)."""
    cls = cls or {"status": ((1.0, 0.8, 1.2), (0.3, 0.1, 0.5)), "unavailable": ((1.0, 0.8, 1.2), (0.3, 0.1, 0.5))}
    return {"primary": {"first": {"aligned_minus_swapped": _eff(*fail)},
                        "item_share": {"first": {"estimate": share[0], "ci": [share[1], share[2]], "n_items": 50,
                                                 "median_item_effect": 0.2}}},
            "primary_controls": {"witness": {"first": {"aligned_minus_swapped": _eff(*wit)}},
                                 "fail_minus_witness_effect": {"first": _eff(*fmw)},
                                 "by_class": {k: {"failing": _eff(*f), "fail_minus_witness": _eff(*d)} for k, (f, d) in cls.items()}}}


def test_llama31_like_profile_is_not_failure_specific():
    """tau_dx's Llama-3.1-8B: +0.47, witness +0.38 (80%), fail minus witness +0.09 [0.04, 0.14], classes +1.23 and -0.33."""
    d = T.describe(_blk((0.47, 0.40, 0.54), (0.38, 0.31, 0.45), (0.09, 0.04, 0.14),
                        {"status": ((1.23, 1.10, 1.34), (0.40, 0.25, 0.55)), "unavailable": ((-0.33, -0.39, -0.27), (-0.27, -0.33, -0.20))}))
    assert d["row"] == "fail minus witness above 0" and not d["failure_specific"]
    assert d["label"] == "a shift shared by working and failing calls"
    assert d["witness_share_of_failing_effect"] == pytest.approx(0.38 / 0.47)
    assert d["classes_with_fail_minus_witness_below_0"] == ["unavailable"] and d["class_mix"]
    assert d["class_mix_reading"] == "the pooled effect depends on the class mix; reported per class"


@pytest.mark.parametrize("wit,cls_unav_fmw,want", [
    ((0.49, 0.4, 0.6), (0.3, 0.1, 0.5), True),     # all three conditions hold
    ((0.50, 0.4, 0.6), (0.3, 0.1, 0.5), False),    # the witness moves exactly half as much: not less than half
    ((0.30, 0.2, 0.4), (-0.2, -0.3, -0.1), False),  # one class's fail minus witness below 0
    ((0.30, 0.2, 0.4), (-0.2, -0.4, 0.1), True),   # a negative class estimate whose interval includes 0 does not block
])
def test_failure_specific_needs_all_three_conditions(wit, cls_unav_fmw, want):
    d = T.describe(_blk((1.0, 0.8, 1.2), wit, (0.6, 0.3, 0.9),
                        {"status": ((1.0, 0.8, 1.2), (0.3, 0.1, 0.5)), "unavailable": ((1.0, 0.8, 1.2), cls_unav_fmw)}))
    assert d["failure_specific"] is want and (d["label"] == "failure-specific") is want


def test_failure_specific_needs_the_pooled_interval_above_zero():
    d = T.describe(_blk((1.0, 0.8, 1.2), (0.3, 0.2, 0.4), (0.7, -0.1, 1.5)))
    assert not d["failure_specific"] and d["row"] == "fail minus witness not above 0"


def test_qwen_like_profile_says_working_calls_move_more():
    d = T.describe(_blk((3.98, 3.61, 4.35), (9.45, 8.94, 9.94), (-5.47, -5.87, -5.05)))
    assert d["label"] == "a shift shared by working and failing calls (working calls move more)"
    assert d["witness_share_of_failing_effect"] == pytest.approx(9.45 / 3.98) and not d["class_mix"]


def test_class_mix_needs_intervals_on_opposite_sides_not_opposite_estimates():
    straddle = {"status": ((1.0, 0.8, 1.2), (0.3, 0.1, 0.5)), "unavailable": ((-0.1, -0.3, 0.1), (0.0, -0.2, 0.2))}
    assert not T.describe(_blk((0.5, 0.4, 0.6), (0.2, 0.1, 0.3), (0.3, 0.1, 0.5), straddle))["class_mix"]
    opposite = {"status": ((1.0, 0.8, 1.2), (0.3, 0.1, 0.5)), "unavailable": ((-0.3, -0.4, -0.2), (0.0, -0.2, 0.2))}
    assert T.describe(_blk((0.5, 0.4, 0.6), (0.2, 0.1, 0.3), (0.3, 0.1, 0.5), opposite))["class_mix"]


@pytest.mark.parametrize("share,want", [((0.55, 0.48, 0.62), True), ((0.45, 0.38, 0.52), True), ((0.56, 0.51, 0.61), False)])
def test_minority_reading_when_the_item_share_interval_is_not_above_one_half(share, want):
    d = T.describe(_blk((1.0, 0.8, 1.2), (0.3, 0.2, 0.4), (0.7, 0.4, 1.0), share=share))
    assert d["minority"] is want and (d["minority_reading"] is not None) is want


def test_minority_and_class_mix_become_verdict_qualifiers_without_changing_the_verdict():
    blocks = {m: {**_block(1.0, [0.5, 1.5], 0.001)} for m in T.SCORED}
    blocks["mistral-7b-v0.3"]["description"] = {"minority": True, "minority_reading": "MIN", "class_mix": True,
                                                "class_mix_reading": "MIX"}
    blocks["llama3.2-3b"]["description"] = {"minority": False, "minority_reading": None, "class_mix": False,
                                            "class_mix_reading": None}
    v = T.family_verdicts(blocks)["per_model"]
    assert v["mistral-7b-v0.3"]["verdict"].startswith("replicates") and v["mistral-7b-v0.3"]["qualifiers"] == ["MIN", "MIX"]
    assert v["llama3.2-3b"]["qualifiers"] == []


def _constructed(t, action, readout, cond, scheme):
    """A score for the reviewer's scenario (review 1, M1): aligned minus swapped by class, failing call and witness."""
    import hashlib
    noise = (int(hashlib.sha256(f"{t['id']}|{action}|{readout}|{cond}".encode()).hexdigest()[:6], 16) / 0xFFFFFF - 0.5) * 0.04
    if readout == "target":
        return {"aligned": 2.0, "swapped": -2.0}[cond] + noise
    base = 2.0 if action == "witness" else -2.0  # neutral: witnesses succeed, failing calls fail (competent, discriminating)
    eff = scheme[(t["cls"] if t["cls"] in ("status", "unavailable") else "other", action)]
    shift = {"aligned": eff / 2, "swapped": -eff / 2, "neutral": 0.0, "mention_aligned": 0.1, "mention_swapped": 0.0}[cond]
    return base + shift + noise


def test_reviewers_scenario_end_to_end_is_not_called_failure_specific(scored):
    """Review 1, M1: a Llama 3.2 3B result like +0.40 with the witness at about 80% of it, fail minus witness a little
    above 0, and class effects of opposite sign. The τ table's reading may print its stronger-form text (the τ rule is
    unchanged), but the extension's label, class-mix reading and pattern sentences must not call it failure-specific."""
    scheme = {("status", "fail"): 1.2, ("status", "witness"): 0.9, ("unavailable", "fail"): -0.4,
              ("unavailable", "witness"): -0.3, ("other", "fail"): 0.4, ("other", "witness"): 0.32}
    prim = {t["id"]: t for t in _primary()}
    assert {t["cls"] for t in prim.values()} >= {"status", "unavailable"}
    d = scored / "tau_family_score_llama3.2-3b"
    saved = {p: (p / "scoring.jsonl").read_text() for p in d.glob("shard_*")}
    try:
        for p in saved:
            recs = read_jsonl(p / "scoring.jsonl")
            for r in recs:
                L = _constructed(prim[r["transition_id"]], r["action"], r["readout"], r["condition"], scheme)
                r.update({"L": L, **{k: v for k, v in _rec(L).items() if k != "L"}})
            (p / "scoring.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        T.run_tau_family_analysis(_analysis_cfg(scored, run_id="tf_ana_scenario"))
        m = json.loads((scored / "tf_ana_scenario" / "metrics.json").read_text())
    finally:
        for p, text in saved.items():
            (p / "scoring.jsonl").write_text(text)
    dsc = m["models"]["llama3.2-3b"]["description"]
    assert dsc["row"] == "fail minus witness above 0" and not dsc["failure_specific"]
    assert dsc["label"].startswith("a shift shared by working and failing calls")
    assert dsc["class_mix"] and dsc["classes_with_fail_minus_witness_below_0"] == ["unavailable"]
    assert 0.5 < dsc["witness_share_of_failing_effect"] < 1.0
    assert m["models"]["llama3.2-3b"]["competence"]["first"]["interpretable"]
    v = m["verdicts"]["per_model"]["llama3.2-3b"]
    assert "the pooled effect depends on the class mix; reported per class" in v["qualifiers"]
    assert m["patterns"]["labels"].get("llama3.2-3b") in (None, "shared")
    text = " ".join(m["patterns"]["readings"])
    assert "specific to failing calls by the pre-stated rule, which" not in text
    assert "task-consistency pattern is specific to Qwen3-8B" not in text
    assert "llama3.2-3b: the pooled effect depends on the class mix; reported per class" in m["patterns"]["readings"]


def test_holm_family_is_both_scored_models_on_the_run_path(scored):
    """Review 1, S2: with one model in the analysis config, the other enters the Holm family with p = 1."""
    assert T.run_tau_family_analysis(_analysis_cfg(scored, run_id="tf_ana_one", models=["mistral-7b-v0.3"])) == 0
    m = json.loads((scored / "tf_ana_one" / "metrics.json").read_text())
    v = m["verdicts"]
    assert v["family"] == list(T.SCORED) and v["per_model"]["llama3.2-3b"]["verdict"] == "not scored"
    p = m["models"]["mistral-7b-v0.3"]["primary"]["first"]["aligned_minus_swapped"]["perm_p"]
    assert v["per_model"]["mistral-7b-v0.3"]["holm_p"] == pytest.approx(min(1.0, 2 * p))
    assert set(m["sensitivity_without_fewshot_tool_classes"]) == {"mistral-7b-v0.3"}


def test_comparisons_of_gated_models_are_marked_not_read():
    cmp = lambda: {"item_share_difference": {"reading": "differs"}, "delta_L_difference_nats": {"reading": "differs"}}
    report = {"verdicts": {"per_model": {"mistral-7b-v0.3": {"verdict": T.NOT_INTERPRETABLE},
                                         "llama3.2-3b": {"verdict": "replicates (ALFWorld direction)"}}},
              "models": {"llama3.2-3b": {"competence": {"first": {"interpretable": True}, "yn": {"interpretable": False}}}},
              "reference": {"qwen3-8b-fewshot": {"competence": {"first": {"interpretable": True}, "yn": {"interpretable": True}}}},
              "comparisons": {"mistral-7b-v0.3 vs qwen3-8b-fewshot": {"first": cmp(), "yn": cmp()},
                              "llama3.2-3b vs qwen3-8b-fewshot": {"first": cmp(), "yn": cmp()}}}
    T.mark_comparisons(report)
    c = report["comparisons"]
    for r in ("first", "yn"):
        assert c["mistral-7b-v0.3 vs qwen3-8b-fewshot"][r]["read"] is False
        assert c["mistral-7b-v0.3 vs qwen3-8b-fewshot"][r]["item_share_difference"]["reading"].startswith("not read: mistral-7b-v0.3")
        assert c["mistral-7b-v0.3 vs qwen3-8b-fewshot"][r]["delta_L_difference_nats"]["reading_unread"] == "differs"
    assert c["llama3.2-3b vs qwen3-8b-fewshot"]["first"]["read"] is True
    assert c["llama3.2-3b vs qwen3-8b-fewshot"]["first"]["item_share_difference"]["reading"] == "differs"
    assert c["llama3.2-3b vs qwen3-8b-fewshot"]["yn"]["read"] is False
    assert "competence rule on the yn readout" in c["llama3.2-3b vs qwen3-8b-fewshot"]["yn"]["not_read_because"][0]


def test_neutral_anchored_fail_minus_witness_and_class_controls():
    items = [{**t, "cls": "status" if int(t["id"][1:]) % 2 else "unavailable"} for t in _items(40, users=10)]
    shift = {("fail", "aligned"): 1.0, ("fail", "swapped"): -0.5, ("witness", "aligned"): 0.4, ("witness", "swapped"): -1.5}
    by = _by(items, lambda k, a, r, c: shift.get((a, c), 0.0) + (0.3 if a == "fail" and k % 2 else 0.0) * (c == "aligned"))
    ctl = T.controls_block(by, items)
    an = ctl["fail_minus_witness_vs_neutral"]["first"]
    assert an["aligned_minus_neutral"]["estimate"] == pytest.approx(0.6 + 0.15)
    assert an["swapped_minus_neutral"]["estimate"] == pytest.approx(1.0)
    assert ctl["witness"]["yn"]["aligned_minus_swapped"]["estimate"] == pytest.approx(1.9)
    assert ctl["fail_minus_witness_effect"]["yn"]["estimate"] == pytest.approx(1.5 - 1.9 + 0.15)
    assert ctl["by_class"]["status"]["fail_minus_witness"]["estimate"] == pytest.approx(1.5 + 0.3 - 1.9)
    assert ctl["by_class"]["unavailable"]["failing"]["estimate"] == pytest.approx(1.5)
    assert ctl["by_class"]["unavailable"]["witness"]["estimate"] == pytest.approx(1.9)


def test_task_judgement_share_no_and_line_intercepts():
    items = _items(60, users=20)
    icpt = {"aligned": 1.0, "neutral": 0.0, "swapped": -2.0}

    def f(k, action, readout, cond):
        x = 4.0 + (k % 5)  # first token: confidently JSON for every call
        if readout == "first":
            return x
        if readout == "yn":
            return 0.5 * x + icpt.get(cond, 0.0) - (5.0 if cond == "swapped" and k % 10 else 0.0) * (action == "fail")
        return 0.0
    tj = T.task_judgement(_by(items, f), items)
    assert tj["confident_json_threshold"] == 3.0
    w = tj["witness"]
    assert w["fit"]["slope"] == pytest.approx(0.5)
    assert w["intercept_differences"]["aligned_minus_neutral"]["estimate"] == pytest.approx(1.0)
    assert w["intercept_differences"]["swapped_minus_neutral"]["estimate"] == pytest.approx(-2.0)
    assert w["share_no_when_first_token_confident_json"]["aligned"] == {"n_confident_json": 60, "share_no": 0.0}
    sw = tj["fail"]["share_no_when_first_token_confident_json"]["swapped"]
    assert sw["n_confident_json"] == 60 and sw["share_no"] == pytest.approx(54 / 60)
    assert w["fit"]["intercept_neutral"] == pytest.approx(0.0) and w["fit"]["intercept_aligned"] == pytest.approx(1.0)
    below = _by(items, lambda k, a, r, c: 2.9 if r == "first" else -1.0)
    assert T.task_judgement(below, items)["fail"]["share_no_when_first_token_confident_json"]["neutral"]["n_confident_json"] == 0
    at = _by(items, lambda k, a, r, c: 3.0 if r == "first" else -1.0)  # exactly +3 counts as confidently JSON
    assert T.task_judgement(at, items)["fail"]["share_no_when_first_token_confident_json"]["neutral"] == {"n_confident_json": 60, "share_no": 1.0}


# ---------------- review 1, S5: the reviewer's surviving mutants ----------------


@pytest.mark.parametrize("shift,want", [(0.42, True), (0.38, False)])
def test_placement_threshold_is_one_fifth_of_the_native_effect(shift, want):
    items = _items(40, users=20)
    native = _by(items, lambda k, x, r, c: {"aligned": 2.0, "swapped": 0.0}.get(c, 0.0))
    first = {k: _rec(v["L"] + (shift + 0.001 * (int(k[0][1:]) % 3) if k[4] == "aligned" else 0.0)) for k, v in native.items()}
    out = T.placement_block(native, first, items)
    assert out["first_turn_minus_native"]["ci"][0] > 0 and out["format_dependent"] is want


def test_competence_boundaries_are_inclusive():
    items = _items(10)
    by = _by(items, lambda k, a, r, c: 0.5 if a == "witness" else (-1.0 if k < 7 else 1.0))  # AUC exactly 0.70
    c = T.competence(by, items, "first")
    assert c["checks"]["auc_neutral"]["value"] == 0.7 and c["checks"]["auc_neutral"]["pass"]
    items = _items(20)
    by = _by(items, lambda k, a, r, c: 5.0 if a == "witness" else (1.0 if k < 19 else -1.0))  # Yes on exactly 95% of failures
    c = T.competence(by, items, "first")
    assert c["checks"]["yes_share_neutral_failures"]["value"] == 0.95 and c["checks"]["yes_share_neutral_failures"]["pass"]


def test_competence_mass_is_a_median_not_a_mean():
    items = _items(20)
    by = _by(items, lambda k, a, r, c: 2.0 if a == "witness" else -2.0)
    for (iid, _, action, readout, cond), v in by.items():
        k = int(iid[1:])
        low = (action == "fail" and k < 16)  # 16 of the 40 neutral records carry almost no two-way mass
        v["logp_plus"], v["logp_minus"] = (math.log(0.005), math.log(0.005)) if low else (math.log(0.3), math.log(0.3))
    c = T.competence(by, items, "first")
    assert c["checks"]["median_two_way_mass_neutral"]["value"] == pytest.approx(0.6) and c["checks"]["median_two_way_mass_neutral"]["pass"]


def test_paired_draws_resample_every_user_like_the_cluster_bootstrap():
    from wf.analysis.stats import cluster_bootstrap
    import random
    rng = random.Random(1)
    rows, users = [], []
    for u in range(40):
        level = rng.gauss(0, 1)
        for _ in range(5):
            rows.append((level + rng.gauss(0, 0.3), 0.0, 0.0, 0.0))
            users.append(f"u{u}")
    draws = T._paired_draws(rows, users, lambda v: T._mean([r[0] for r in v]))
    width = T._pct(draws, 0.975) - T._pct(draws, 0.025)
    ref = cluster_bootstrap(lambda v: T._mean([r[0] for r in v]), rows, users, seed=C.RNG_SEED)
    assert 0.8 < width / (ref["ci_high"] - ref["ci_low"]) < 1.25


def test_item_share_interval_is_clustered_by_user():
    from wf.analysis.stats import cluster_bootstrap
    items = _items(100, users=20)  # user u holds items u, u+20, ..., u+80
    by = _by(items, lambda k, a, r, c: {"aligned": 1.0 if (k % 20) % 2 == 0 else -1.0, "swapped": 0.0}.get(c, 0.0))
    s = T.item_share(by, items, "first")
    flat = cluster_bootstrap(T._mean, [1.0 if (k % 20) % 2 == 0 else 0.0 for k in range(100)], [f"i{k}" for k in range(100)], seed=C.RNG_SEED)
    assert (s["ci"][1] - s["ci"][0]) > 1.5 * (flat["ci_high"] - flat["ci_low"])
    assert s["estimate"] == 0.5 and s["tie_share"] == 0.0 and s["median_item_effect"] == 0.0


def test_in_job_readout_report_checks_the_witness_reply(runs):
    t, few = _primary()[0], _fewshot()
    spec = T.spec_for("mistral-7b-v0.3")
    marker = render_call(t["aligned_witness"]["name"], t["aligned_witness"]["arguments"])
    s = T._scorer_for(spec, TauTok(odd_reply_after=((marker, '{"'),)))
    s.mock = False
    rep = T.readout_report(s, spec, t, few)
    assert not rep["ok"] and rep["reply_starts"]["twin"] == rep["alone"]['{"'] and rep["reply_starts"]["witness"] != rep["alone"]['{"']
