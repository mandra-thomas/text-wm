"""TextCraft round 3 (notes/textcraft_r3_design.md): models, gates, scoring runs, the thinking arm, and every decision
rule of the analysis, without a model or a real tokenizer."""
from __future__ import annotations

import copy
import json
import math
import random
import re
import statistics
import types
from dataclasses import replace

import pytest

from wf import constants as C
from wf import families as FAM
from wf.runs import RUNNERS
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r2_analysis as A
from wf.textcraft import r2_items as I
from wf.textcraft import r2_score as S2
from wf.textcraft import r3 as R3
from wf.textcraft import r3_gates as G
from wf.textcraft import r3_analysis as Z
from wf.textcraft import r3_score as S
from wf.textcraft import r3_think as K
from wf.textcraft import r3_tokcheck as T
from wf.textcraft.verify import sha256

SHOTS = P.fewshot_examples()


@pytest.fixture(scope="module")
def records():
    return I.read_records()


@pytest.fixture(scope="module")
def small(records):
    """Every record of eight pairs with parent records and of eight with padded goal and ingredient records."""
    parent_pairs = sorted({r["pair_id"] for r in records if r["kind"] == "parent"})[:8]
    roles: dict[str, set] = {}
    for r in records:
        if r["kind"] == "main" and "pad8_aligned" in r["arms"]:
            roles.setdefault(r["pair_id"], set()).add(r["role"])
    both_roles = [p for p in sorted(roles) if roles[p] == {"goal", "ingredient"}][:8]
    keep = set(parent_pairs) | set(both_roles)
    return [r for r in records if r["pair_id"] in keep]


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(A, "B_DRAWS", 300)
    monkeypatch.setattr(A, "N_FLIPS", 300)
    monkeypatch.setattr(A, "DESC_DRAWS", 50)
    monkeypatch.setattr(A, "AUC_DRAWS", 50)
    monkeypatch.setattr(Z, "B_DRAWS", 300)
    monkeypatch.setattr(Z, "DESC_DRAWS", 50)


# ---------------- models ----------------


def test_core_extension_fallback_and_thinking_arm():
    assert R3.CORE == ("qwen3.5-9b", "ministral3-8b-it", "gemma4-12b-it")
    assert R3.EXTENSION == ("qwen3.8-27b", "gemma4-31b-it", "olmo3-7b-it")
    assert R3.core_tags("gemma4-e4b-it") == ("qwen3.5-9b", "ministral3-8b-it", "gemma4-e4b-it")
    with pytest.raises(ValueError):
        R3.core_tags("gemma3-12b-it")
    assert R3.pairs("gemma4-e4b-it")[2] == ("gemma2-9b-it", "gemma4-e4b-it")
    assert R3.PAIRS == (("qwen3-8b", "qwen3.5-9b"), ("mistral-7b-v0.3", "ministral3-8b-it"),
                        ("gemma2-9b-it", "gemma4-12b-it"))
    for t in ("qwen3.5-9b", "ministral3-8b-it", "qwen3.8-27b", "gemma4-31b-it", "gemma4-e4b-it"):
        assert R3.MODELS[t] == FAM.WAVE_B[t]  # wave B's specs, unchanged
    assert R3.FAMILY_SIZES == {"A3": 30, "A3x": 30, "B3": 30}


def test_every_model_is_pinned_at_the_design_revision():
    want = {"Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "mistralai/Ministral-3-8B-Instruct-2512-BF16": "f6fae9795746f63c9be8344932f01275f3c63734",
            "google/gemma-4-12B-it": "43c9d014b231474fa7212dbb48a762566c2b255d",
            "google/gemma-4-E4B-it": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            "Qwen/Qwen3.8-27B": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
            "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "allenai/Olmo-3-7B-Instruct": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"}
    for spec in R3.MODELS.values():
        assert C.REVISIONS[spec.hf_id] == want[spec.hf_id] == spec.revision
    assert R3.THINK_SPEC.hf_id == "Qwen/Qwen3.5-9B" and R3.THINK_SPEC.template_kwargs == {"enable_thinking": True}


def test_new_specs_take_the_31b_template_handling_and_strict_loading():
    g12, g31 = R3.MODELS["gemma4-12b-it"], R3.MODELS["gemma4-31b-it"]
    for f in ("generation_suffix", "thinking_off_ending", "thinking_on_markers", "thinking_switch", "system",
              "model_class", "strict_load", "attn_agreement_check"):
        assert getattr(g12, f) == getattr(g31, f), f
    assert g12.unexpected_keys_allowed == "" and g12.gpu_min_gb >= 24 and g12.env == "serve2"
    olmo = R3.MODELS["olmo3-7b-it"]
    assert olmo.model_class is None and olmo.strict_load and olmo.unexpected_keys_allowed == ""
    assert olmo.thinking_off_ending == "<|im_start|>assistant\n" and "function-calling" in olmo.forbidden_in_prompt
    assert not olmo.thinking_switch and olmo.env == "serve2"


def test_spec_for_variants_and_base_tags():
    assert R3.spec_for("qwen3-8b-serve2").env == "serve2" and R3.spec_for("qwen3-8b-serve2").tag == "qwen3-8b-serve2"
    assert R3.spec_for("gemma4-12b-it-firstturn").system == "fold"
    assert R3.spec_for("qwen3.8-27b-nopreserve").template_kwargs["preserve_thinking"] is False
    assert R3.spec_for(R3.THINK_TAG) is R3.THINK_SPEC
    for t, base in (("qwen3-8b-serve2", "qwen3-8b"), ("gemma4-12b-it-firstturn", "gemma4-12b-it"),
                    ("qwen3.8-27b-nopreserve", "qwen3.8-27b"), ("olmo3-7b-it", "olmo3-7b-it")):
        assert R3.base_tag(t) == base
    with pytest.raises(KeyError):
        R3.spec_for("llama4")


def test_gpu_classes_and_pins():
    assert R3.gpu_class(["NVIDIA L40S"]) == "NVIDIA L40S"
    assert R3.gpu_class(["NVIDIA L40S", "NVIDIA L40S"]) == "2x NVIDIA L40S"
    assert R3.gpu_class(["NVIDIA A100-SXM4-80GB"]) == R3.A100_80
    assert R3.gpu_class(["NVIDIA A100-PCIE-40GB"]) is None and R3.gpu_class(["NVIDIA A40"]) is None
    assert R3.gpu_class(None) is None
    assert R3.gpu_type_ok("qwen3.5-9b", ["NVIDIA L40S"]) and not R3.gpu_type_ok("qwen3.5-9b", ["NVIDIA A40"])
    assert not R3.gpu_type_ok("qwen3.5-9b", ["NVIDIA L40S", "NVIDIA L40S"])
    assert R3.gpu_type_ok("qwen3.8-27b", ["NVIDIA A100-SXM4-80GB"])
    assert R3.gpu_type_ok("gemma4-31b-it", ["NVIDIA L40S", "NVIDIA L40S"])
    assert not R3.gpu_type_ok("gemma4-31b-it", ["NVIDIA L40S"]) and not R3.gpu_type_ok("qwen3.8-27b", ["NVIDIA L40S"])
    assert R3.GPU_PIN["qwen3.8-27b"] == R3.GPU_PIN["gemma4-31b-it"] == (R3.A100_80, "2x NVIDIA L40S")
    assert all(R3.GPU_PIN[t] == (R3.L40S,) for t in ("qwen3.5-9b", "ministral3-8b-it", "gemma4-12b-it", "gemma4-e4b-it",
                                                      "olmo3-7b-it", R3.THINK_TAG))
    with pytest.raises(SystemExit):
        S.gpu_check("qwen3.5-9b", ["NVIDIA A40"])
    assert S.gpu_check("qwen3.5-9b", ["NVIDIA L40S"])["checked"] is True
    assert S.gpu_check("gemma4-12b-it-firstturn", ["NVIDIA L40S"])["checked"] is True
    assert "recorded only" in S.gpu_check("qwen3-8b-serve2", ["NVIDIA A40"])["checked"]


def test_runners_and_configs_registered():
    import yaml

    from wf.util import REPO_ROOT

    names = ([f"textcraft_r3_{k}_{t}" for k in ("score", "smoke") for t in R3.MODELS]
             + [f"textcraft_r3_envcheck_{t}" for t in R3.ENVCHECK_MODELS]
             + [f"textcraft_r3_rescore_{t}" for t in R3.RESCORE_MODELS]
             + ["textcraft_r3_placement_gemma4-12b-it", "textcraft_r3_nopreserve_qwen3.8-27b",
                "textcraft_r3_think_qwen3.5-9b", "textcraft_r3_think_smoke_qwen3.5-9b", "textcraft_r3_analysis"])
    for n in names:
        assert n in RUNNERS, n
        cfg = yaml.safe_load((REPO_ROOT / "wf" / "config" / f"{n}.yaml").read_text())
        assert cfg["run_id"] == n and cfg["manifest"] == "wf/textcraft/data/r2_manifest.json"
    for n in R3.RUN_ORDER:
        assert n in RUNNERS


def test_seeded_subsets_are_fixed(records):
    smoke, env, fmt = S.smoke_subset(records), S.envcheck_subset(records), S.format_subset(records)
    assert len(smoke) == 20 and len(env) == 300 and len(fmt) == 300
    assert all(r["kind"] == "main" and r["population"] == "main" for r in fmt)
    shuffled = list(records)
    random.Random(1).shuffle(shuffled)
    assert {r["id"] for r in S.envcheck_subset(shuffled)} == {r["id"] for r in env}
    assert {r["id"] for r in smoke} != {r["id"] for r in S2.smoke_subset(records, 20)}  # its own seed, not round 2's


# ---------------- the gate ----------------


class FakeTok:
    """A word-level chat tokenizer stand-in with a thinking switch: the generation prompt ends '<a> THINKOFF' with
    thinking off (a suffix a rendered reply does not contain) and '<a> THINKON' with it on."""

    def __init__(self, merge_goal=False, forbid=None):
        self.vocab = {"<s>": 0, "<e>": 1, "<u>": 2, "<a>": 3, "THINKOFF": 4, "THINKON": 5}
        self.all_special_ids = [0, 1, 2, 3, 4, 5]
        self.added_tokens_decoder = {}
        self.merge_goal, self.forbid = merge_goal, forbid

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def _words(self, text):
        words = re.findall(r"<[seua]>|[^\s<]+", text)
        if self.merge_goal:  # a tokenizer that merges the goal line with the next word (an isolation failure)
            out, i, goal_at = [], 0, -99
            while i < len(words):
                if words[i] == "Goal:":
                    goal_at = i
                if words[i].endswith(".") and i + 1 < len(words) and 0 < i - goal_at <= 8:
                    out.append(words[i] + "+" + words[i + 1])
                    i += 2
                else:
                    out.append(words[i])
                    i += 1
            return out
        return words

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return types.SimpleNamespace(input_ids=[self._id(w) for w in self._words(text)])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, **kw):
        body = " ".join(f"<{'a' if m['role'] == 'assistant' else 'u'}> " + m["content"] + " <e>" for m in messages
                        if m["role"] != "system")
        if self.forbid:
            body = self.forbid + " " + body
        text = "<s> " + body + ((" <a> THINKON" if enable_thinking else " <a> THINKOFF") if add_generation_prompt else "")
        return self(text).input_ids if tokenize else text

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


FAKE_SPEC = replace(R3.MODELS["qwen3.5-9b"], generation_suffix="THINKOFF", thinking_off_ending="<a> THINKOFF",
                    thinking_on_ending="<a> THINKON", thinking_on_markers=("THINKON",), mistral_common_check=False)


def _scorer(tok, spec=FAKE_SPEC):
    from wf.scoring.logodds import LogoddsScorer

    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs)


def test_gate_passes_and_counts_every_record_and_arm(small):
    recs = small[:25]
    rep = T.gate(_scorer(FakeTok()), FAKE_SPEC, recs, SHOTS)
    assert rep["ok"], rep["problems"]
    n_two = sum(1 for r in recs for a in r["arms"] if not R.parse_arm(a)[0])
    assert rep["n_candidates_checked"] == 2 * n_two
    assert rep["n_yes_no_prompts_checked"] == sum(1 for r in recs for a in r["arms"] if R.parse_arm(a)[0])
    assert rep["n_prompts_checked_ending"] == sum(len(r["arms"]) for r in recs)
    assert rep["n_arm_pairs_checked"] > 0 and rep["n_arm_pairs_not_isolated"] == 0 and rep["n_switch_renders"] > 0


def test_gate_needs_the_declared_suffix(small):
    rep = T.gate(_scorer(FakeTok()), replace(FAKE_SPEC, generation_suffix=""), small[:3], SHOTS)
    assert not rep["ok"] and any("does not begin with the prompt" in p for p in rep["problems"])


def test_gate_catches_a_wrong_ending_a_forbidden_text_and_an_unread_switch(small):
    rep = T.gate(_scorer(FakeTok()), replace(FAKE_SPEC, thinking_off_ending="<a> NOPE"), small[:2], SHOTS)
    assert any("does not end with" in p for p in rep["problems"])
    rep = T.gate(_scorer(FakeTok(forbid="Le Chat")), replace(FAKE_SPEC, forbidden_in_prompt=("Le",)), small[:2], SHOTS)
    assert any("contains 'Le'" in p for p in rep["problems"])
    rep = T.gate(_scorer(FakeTok()), replace(FAKE_SPEC, thinking_on_markers=("MISSING",)), small[:2], SHOTS)
    assert any("lacks 'MISSING'" in p for p in rep["problems"])
    rep = T.gate(_scorer(FakeTok()), replace(FAKE_SPEC, thinking_on_ending="<a> X"), small[:2], SHOTS)
    assert any("thinking on does not end" in p for p in rep["problems"])


class NoSwitch(FakeTok):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, **kw):
        return super().apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking=False, **kw)


def test_gate_catches_a_template_that_ignores_the_thinking_switch(small):
    rep = T.gate(_scorer(NoSwitch()), replace(FAKE_SPEC, thinking_on_ending="", thinking_on_markers=()), small[:2], SHOTS)
    assert not rep["ok"] and any("renders like thinking off" in p for p in rep["problems"])


class Prefixed(FakeTok):
    """A template that writes a marker word before every reply, so a candidate tokenized alone is not the reply."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, **kw):
        messages = [{**m, "content": "~ " + m["content"]} if m["role"] == "assistant" else m for m in messages]
        return super().apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking, **kw)


def test_gate_catches_a_candidate_that_is_not_the_templates_reply(small):
    rep = T.gate(_scorer(Prefixed()), FAKE_SPEC, small[:2], SHOTS)
    assert not rep["ok"] and any("vs template" in p and ("o_plus" in p or "o_minus" in p) for p in rep["problems"])


def test_gate_requires_mistral_common_when_declared(small):
    rep = T.gate(_scorer(FakeTok()), replace(FAKE_SPEC, mistral_common_check=True), small[:2], SHOTS)
    assert not rep["ok"] and any("mistral-common" in p for p in rep["problems"])
    rep = T.gate(_scorer(FakeTok()), replace(FAKE_SPEC, mistral_common_check=True), small[:2], SHOTS,
                 reference_encoder=lambda m: [99])
    assert any("differ from mistral-common" in p for p in rep["problems"])


def test_gate_catches_tokens_changed_outside_the_goal_line(small):
    rep = T.gate(_scorer(FakeTok(merge_goal=True)), FAKE_SPEC, small[:4], SHOTS)
    assert rep["n_arm_pairs_not_isolated"] > 0 and not rep["ok"]


def test_isolation_check():
    tok = FakeTok()
    a = tok("<s> x Goal: craft stick. y").input_ids
    b = tok("<s> x Goal: craft torch. y").input_ids
    assert T.isolated(tok, a, b, "Goal: craft stick.", "Goal: craft torch.")
    c = tok("<s> z Goal: craft torch. y").input_ids
    assert not T.isolated(tok, a, c, "Goal: craft stick.", "Goal: craft torch.")
    d = tok("<s> x Goal: craft torch. Q y").input_ids  # only the second prompt changes a token after its line
    assert not T.isolated(tok, a, d, "Goal: craft stick.", "Goal: craft torch.")
    assert not T.isolated(tok, d, a, "Goal: craft torch.", "Goal: craft stick.")


def test_gate_catches_a_multi_token_yes_and_shared_first_tokens(small, monkeypatch):
    class Split(FakeTok):
        def _words(self, text):
            out = []
            for w in super()._words(text):
                out += ["Y", "es"] if w == "Yes" else ["C", w] if w in ("Crafted", "Could") else [w]
            return out
    rep = T.gate(_scorer(Split()), FAKE_SPEC, small[:2], SHOTS)
    assert any("is not one token" in p for p in rep["problems"])
    assert any("share their first token" in p for p in rep["problems"])


def test_golden_problems(small):
    sc = _scorer(FakeTok())
    rid = small[0]["id"]
    one = T.render_one(sc, FAKE_SPEC, small[0], "aligned", SHOTS)
    golden = {"cases": [{"id": rid, "arm": "aligned"}], "models": {"qwen3.5-9b": {f"{rid}|aligned": one}}}
    assert T.golden_problems(sc, "qwen3.5-9b", small, SHOTS, golden) == []
    golden["models"]["qwen3.5-9b"][f"{rid}|aligned"] = {**one, "n_ids": 1}
    assert T.golden_problems(sc, "qwen3.5-9b", small, SHOTS, golden)
    assert T.golden_problems(sc, "olmo3-7b-it", small, SHOTS, golden) == ["no golden rendering stored for olmo3-7b-it"]


# ---------------- scoring runs, mock ----------------


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "FILED.txt").write_text("filed_date: x\n")
    return tmp_path


def _manifest(tmp_path, recs):
    path = tmp_path / "r2_items.jsonl"
    path.write_text(I.dumps(recs))
    m = {"records_file": str(path), "records_sha256": sha256(path), "fewshot_file": str(P.FEWSHOT),
         "fewshot_sha256": sha256(P.FEWSHOT), "verified": True, "why_not": []}
    (tmp_path / "r2_manifest.json").write_text(json.dumps(m))
    return str(tmp_path / "r2_manifest.json")


def test_score_run_scores_every_record_and_arm_then_resumes(mock_runs, small):
    man = _manifest(mock_runs, small)
    for i in range(2):
        cfg = {"run_id": "textcraft_r3_score_qwen3.5-9b", "model": "qwen3.5-9b", "manifest": man, "n_shards": 2, "shard": i}
        assert S.run_textcraft_r3_score(cfg) == 0
    scores = Z.scores_of("textcraft_r3_score_qwen3.5-9b", "qwen3.5-9b", mock_runs / "runs")
    assert len(scores) == sum(len(r["arms"]) for r in small)
    by, missing = A.index_scores(scores, small, SHOTS, "qwen3.5-9b")
    assert missing == []
    d = mock_runs / "runs" / "textcraft_r3_score_qwen3.5-9b" / "shard_00_of_02"
    n0 = len((d / "scoring.jsonl").read_text().splitlines())
    cfg = {"run_id": "textcraft_r3_score_qwen3.5-9b", "model": "qwen3.5-9b", "manifest": man, "n_shards": 2, "shard": 0,
           "resume": True}
    assert S.run_textcraft_r3_score(cfg) == 0
    assert len((d / "scoring.jsonl").read_text().splitlines()) == n0
    lines = (d / "scoring.jsonl").read_text().splitlines()
    bad = json.loads(lines[0])
    bad["item_hash"] = "0" * 16
    (d / "scoring.jsonl").write_text("\n".join([json.dumps(bad)] + lines[1:-2]) + "\n")
    with pytest.raises(SystemExit):
        S.run_textcraft_r3_score(cfg)


def test_runs_refuse_unknown_models_and_unverified_records(mock_runs, small):
    man = _manifest(mock_runs, small)
    with pytest.raises(SystemExit):
        S.run_textcraft_r3_score({"run_id": "x", "model": "qwen3-8b", "manifest": man})
    with pytest.raises(SystemExit):
        S.run_textcraft_r3_envcheck({"run_id": "x", "model": "llama3.1-8b", "manifest": man, "plan": {"models": {
            "llama3.1-8b": {"runnable": True, "n_r2_shards": 1, "jobs": [{"job": 0, "gpu": "mock", "r2_shards": [0]}]}}}})
    with pytest.raises(SystemExit):
        S.run_textcraft_r3_format({"run_id": "x", "model": "qwen3.5-9b", "check": "placement", "manifest": man})
    m = json.loads((mock_runs / "r2_manifest.json").read_text())
    (mock_runs / "r2_manifest.json").write_text(json.dumps({**m, "verified": False}))
    assert S.run_textcraft_r3_score({"run_id": "textcraft_r3_score_qwen3.5-9b", "model": "qwen3.5-9b", "manifest": man}) == 1


def test_smoke_envcheck_and_format_runs_score_their_subsets(mock_runs, records):
    man = _manifest(mock_runs, records)
    assert S.run_textcraft_r3_score({"run_id": "textcraft_r3_smoke_olmo3-7b-it", "model": "olmo3-7b-it", "smoke": True,
                                     "manifest": man}) == 0
    sm = Z.scores_of("textcraft_r3_smoke_olmo3-7b-it", "olmo3-7b-it", mock_runs / "runs")
    assert {s["transition_id"] for s in sm} == {r["id"] for r in S.smoke_subset(records)}
    assert S.run_textcraft_r3_format({"run_id": "textcraft_r3_placement_gemma4-12b-it", "model": "gemma4-12b-it",
                                      "check": "placement", "manifest": man}) == 0
    fm = Z.scores_of("textcraft_r3_placement_gemma4-12b-it", "gemma4-12b-it-firstturn", mock_runs / "runs")
    assert len(fm) == 300 * 4 and {s["condition"] for s in fm} == set(R3.FORMAT_ARMS)


def test_gate_failure_stops_before_any_score(mock_runs, small, monkeypatch):
    man = _manifest(mock_runs, small)
    monkeypatch.setattr(S, "run_gate", lambda *a, **k: False)
    cfg = {"run_id": "textcraft_r3_score_ministral3-8b-it", "model": "ministral3-8b-it", "manifest": man}
    assert S.run_textcraft_r3_score(cfg) == 1
    assert not (mock_runs / "runs" / "textcraft_r3_score_ministral3-8b-it" / "shard_00_of_01" / "scoring.jsonl").exists()


def _gate_scorer(tok, spec, attn="sdpa"):
    sc = _scorer(tok, spec)
    sc.mock = False
    sc._model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation=attn), dtype="bf16")
    return sc


def test_run_gate_requires_the_golden_and_records_the_loading(small, tmp_path, monkeypatch):
    sc = _gate_scorer(FakeTok(), FAKE_SPEC)
    assert not S.run_gate(tmp_path, sc, FAKE_SPEC, "qwen3.5-9b", small[:3], small, SHOTS, {})
    rec = json.loads((tmp_path / "tokenization_checks.json").read_text())
    assert rec["golden_problems"] and rec["ok"] is False and "loading" in rec
    monkeypatch.setattr(T, "golden_problems", lambda *a, **k: [])
    assert S.run_gate(tmp_path, sc, FAKE_SPEC, "qwen3.5-9b", small[:3], small, SHOTS, {})
    monkeypatch.setattr(S, "attention_agreement", lambda *a, **k: {"ok": False, "problems": ["differ"]})
    assert not S.run_gate(tmp_path, sc, replace(FAKE_SPEC, attn_agreement_check=True), "qwen3.5-9b", small[:3], small,
                          SHOTS, {})


def test_attention_agreement_rule(small, monkeypatch):
    state = {"impl": "sdpa", "shift": 0.0}
    model = types.SimpleNamespace(config=types.SimpleNamespace())
    monkeypatch.setattr(FAM, "_attn_loaded", lambda m: [state["impl"]])
    monkeypatch.setattr(FAM, "_set_text_attn", lambda m, impl: state.__setitem__("impl", impl))
    calls = []

    def fake_score(scorer, spec, r, arm, shots):
        calls.append(arm)
        return types.SimpleNamespace(L=1.0 + (state["shift"] if state["impl"] == "eager" else 0.0))

    monkeypatch.setattr(S, "score_one", fake_score)
    sc = types.SimpleNamespace(_model=model, _cache={})
    out = S.attention_agreement(sc, FAKE_SPEC, small, SHOTS)
    assert out["ok"] and state["impl"] == "sdpa" and out["n_items"] == 5
    state["shift"] = 0.3  # above the yes/no mean tolerance (0.10)
    out = S.attention_agreement(sc, FAKE_SPEC, small, SHOTS)
    assert not out["ok"] and any("yes_no" in p for p in out["problems"])
    state["shift"] = 0.05
    assert S.attention_agreement(sc, FAKE_SPEC, small, SHOTS)["ok"]
    state["shift"], state["impl"] = 0.0, "eager"
    assert not S.attention_agreement(sc, FAKE_SPEC, small, SHOTS)["ok"]  # loaded attention must be SDPA


# ---------------- the thinking arm ----------------


def test_thinking_jobs_and_procedure(records):
    jobs = K.think_jobs(records)
    assert len(jobs) == 4530
    arms = {a for _r, a in jobs}
    assert arms == {"ext_aligned", "ext_parent", "aligned", "swapped", "yn_neutral"}
    assert sum(a == "yn_neutral" and r["population"] == "main" for r, a in jobs) == 600
    assert sum(a == "yn_neutral" and r["population"] == "success" for r, a in jobs) == 600
    assert K.MAX_REASONING == 2048 and K.SAMPLING == {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert K.batch_seed("a|b") == K.batch_seed("a|b") != K.batch_seed("a|c")


def test_thinking_run_mock_generates_scores_and_resumes(mock_runs, records):
    man = _manifest(mock_runs, records)
    cfg = {"run_id": "textcraft_r3_think_smoke_qwen3.5-9b", "manifest": man, "max_jobs": 20}
    assert K.run_textcraft_r3_think(cfg) == 0
    d = mock_runs / "runs" / "textcraft_r3_think_smoke_qwen3.5-9b" / "shard_00_of_01"
    gens = [json.loads(x) for x in (d / "generations.jsonl").read_text().splitlines()]
    sc = [json.loads(x) for x in (d / "scoring.jsonl").read_text().splitlines()]
    assert len(gens) == len(sc) == 20 and all(s["model"] == R3.THINK_TAG for s in sc)
    assert any(g["forced"] for g in gens) and all(g["reasoning"].endswith("</think>") for g in gens)
    assert all(s["L"] == pytest.approx(s["logp_plus"] - s["logp_minus"]) for s in sc)
    assert K.run_textcraft_r3_think({**cfg, "resume": True}) == 0
    assert len((d / "generations.jsonl").read_text().splitlines()) == 20
    lines = (d / "generations.jsonl").read_text().splitlines()
    bad = json.loads(lines[0])
    bad["item_hash"] = "0" * 16
    (d / "generations.jsonl").write_text("\n".join([json.dumps(bad)] + lines[1:]) + "\n")
    with pytest.raises(SystemExit):
        K.run_textcraft_r3_think({**cfg, "resume": True})


# ---------------- analysis: building blocks ----------------


def test_stats_match_the_definitions():
    xs = [0.5, 1.5, -0.2, 2.0, 0.7]
    raw, z = Z._stats((len(xs), sum(xs), sum(x * x for x in xs), 0, 0, 0), False)
    assert raw == pytest.approx(statistics.mean(xs)) and z == pytest.approx(statistics.mean(xs) / statistics.stdev(xs))
    g, i = [1.0, 2.0, 4.0], [0.0, 1.0, 0.5, 0.2]
    raw, z = Z._stats((3, sum(g), sum(x * x for x in g), 4, sum(i), sum(x * x for x in i)), True)
    pooled = math.sqrt((2 * statistics.variance(g) + 3 * statistics.variance(i)) / 5)
    assert raw == pytest.approx(statistics.mean(g) - statistics.mean(i))
    assert z == pytest.approx((statistics.mean(g) - statistics.mean(i)) / pooled)
    assert math.isnan(Z._stats((1, 1.0, 1.0, 0, 0, 0), False)[1])


def _values(n_pairs=30, per=6, gap=False, shift=0.4, scale=1.0, seed=0):
    rng = random.Random(seed)
    out = {}
    for p in range(n_pairs):
        for k in range(per):
            role = ("goal" if k % 2 == 0 else "ingredient") if gap else None
            mu = shift if (not gap or role == "goal") else 0.0
            out[f"r{p}-{k}"] = (role, scale * (mu + rng.gauss(0, 1)), f"p{p}")
    return out


def test_paired_difference_is_scale_free_and_zero_for_the_same_model():
    old = _values()
    same = Z.paired_difference(old, old, False, B=200)
    assert same["standardized"]["ci95"] == [0.0, 0.0]  # one resample serves both models, so every draw is zero
    assert same["standardized"]["estimate"] == pytest.approx(0) and same["raw_nats_descriptive"]["estimate"] == pytest.approx(0)
    triple = {k: (r, 3 * x, p) for k, (r, x, p) in old.items()}
    d = Z.paired_difference(old, triple, False, B=200)
    assert d["standardized"]["estimate"] == pytest.approx(0, abs=1e-12)  # a pure change of scale moves nothing
    assert d["raw_nats_descriptive"]["estimate"] == pytest.approx(2 * statistics.mean(x for _r, x, _p in old.values()))
    assert d["standardized"]["ci95"][0] <= 0 <= d["standardized"]["ci95"][1]
    gold, gnew = _values(gap=True, seed=1), _values(gap=True, seed=1, scale=5.0)
    assert Z.paired_difference(gold, gnew, True, B=100)["standardized"]["estimate"] == pytest.approx(0, abs=1e-12)


def test_paired_difference_detects_a_real_difference_and_uses_common_records_only():
    old, new = _values(shift=0.0, seed=2), _values(shift=1.0, seed=3)
    d = Z.paired_difference(old, new, False, B=500)
    assert d["standardized"]["estimate"] > 0.5 and d["standardized"]["ci95"][0] > 0
    assert d["standardized"]["p_differs"] < 0.01 and d["standardized"]["p_equivalent"] > 0.5
    new_less = dict(list(new.items())[:-12])
    assert Z.paired_difference(old, new_less, False, B=50)["n_items"] == len(new_less)
    extreme = {**old, **{f"x{k}": (None, 50.0, "p0") for k in range(10)}}  # records only the older model scored
    a = Z.paired_difference(extreme, new, False, B=50)
    b = Z.paired_difference(old, new, False, B=50)
    assert a["standardized"]["estimate"] == pytest.approx(b["standardized"]["estimate"])


def test_paired_difference_resamples_whole_goal_pairs():
    old = _values(seed=4)
    a = Z.paired_difference(old, _values(seed=5), False, B=300, seed=7)
    b = Z.paired_difference(old, _values(seed=5), False, B=300, seed=7)
    assert a == b and a["n_pairs"] == 30


def test_item_share_and_its_difference():
    old = {"a": (None, 1.0, "p"), "b": (None, -1.0, "p"), "c": (None, 2.0, "q"), "d": (None, -0.5, "q")}
    new = {k: (r, abs(x), p) for k, (r, x, p) in old.items()}
    assert Z.item_share(old) == 0.5 and Z.item_share(new) == 1.0
    assert Z.share_difference(old, new, B=50)["difference"] == pytest.approx(0.5)


def _planted(recs, tag, scale=1.0, penalty=-1.0, lift=0.5):
    """Mock scores with known effects: ext_aligned below ext_parent by `penalty`, aligned above swapped by `lift`."""
    out = []
    rng = random.Random(tag)
    for r in recs:
        for a in r["arms"]:
            base = rng.gauss(0, 1)
            L = base
            if a in ("ext_aligned",):
                L += penalty
            if a in ("aligned", "pad1_aligned", "pad8_aligned", "yn_aligned") and r["role"] == "ingredient":
                L += lift
            if a.startswith("yn_") or a == "neutral":
                L += 3.0 if r["population"] == "success" else -3.0
            L *= scale
            first = L / 2
            out.append({"transition_id": r["id"], "condition": a, "model": tag, "L": L, "logp_plus": -1.0,
                        "logp_minus": -1.0 - L, "logp_first_plus": -0.5, "logp_first_minus": -0.5 - first,
                        "item_hash": R.item_hash(r, SHOTS), "model_revision": "x"})
    return out


def test_record_values_agree_with_round_twos_contrast_inputs(small):
    by = {(s["transition_id"], s["condition"]): s for s in _planted(small, "m")}
    pops = A.populations(small)
    inputs = A.contrast_inputs(by, pops)
    for c in A.CONTRASTS:
        kind, rows, cl = inputs[c]
        v = Z.record_values(by, pops, c)
        if kind == "within":
            assert statistics.mean(rows) == pytest.approx(statistics.mean(x for _r, x, _p in v.values()))
        else:
            assert A._gap(rows) == pytest.approx(Z._stats(Z._total(Z._cluster_sums(v, None, True),
                                                                   {p for _r, _x, p in v.values()}), True)[0])
        assert sorted(cl) == sorted(p for _r, _x, p in v.values())


def test_first_token_tail_split_adds_up(small, fast):
    by = {(s["transition_id"], s["condition"]): s for s in _planted(small, "m")}
    pops = A.populations(small)
    split = Z.first_tail(by, pops)
    full = A.contrasts(by, pops, 0.4)
    for c in A.CONTRASTS:
        assert split["first_token"][c]["estimate"] + split["tail"][c]["estimate"] == pytest.approx(full[c]["estimate"])


# ---------------- analysis: decision rules ----------------

V = lambda **kw: {c: kw.get(c, "inconclusive") for c in A.CONTRASTS}  # noqa: E731


def test_r2_for_round_three():
    v = {"a": V(parent_penalty="confirmed below zero"), "b": V(parent_penalty="equivalent to zero")}
    assert Z.r2_contemporary(v, {"a": True, "b": True})["value"] is False
    assert Z.r2_contemporary(v, {"a": False, "b": True})["value"] is True
    assert Z.r2_contemporary({"a": V(parent_penalty="confirmed above zero")}, {"a": True})["fires"] is True
    assert Z.r2_contemporary({"a": V()}, {"a": True})["value"] == "undecided"
    assert Z.r2_contemporary(v, {"a": False, "b": False})["value"] == "undecided"


def test_round2_finding():
    b, a, e, i = "confirmed below zero", "confirmed above zero", "equivalent to zero", "inconclusive"
    comp = {"x": True, "y": True, "z": False}
    assert Z.round2_finding({"x": b, "y": b, "z": a}, comp) == {"kind": "general", "sign": -1,
                                                              "models_competent": ["x", "y"],
                                                              "verdicts": {"x": b, "y": b}}
    assert Z.round2_finding({"x": a, "y": a, "z": b}, comp)["sign"] == 1
    assert Z.round2_finding({"x": e, "y": e, "z": b}, comp)["kind"] == "general null"
    assert Z.round2_finding({"x": b, "y": e, "z": b}, comp)["kind"] == "model-specific"
    assert Z.round2_finding({"x": b, "y": a, "z": b}, comp)["kind"] == "model-specific"
    assert Z.round2_finding({"x": b, "y": i, "z": b}, comp)["kind"] == "not established"
    assert Z.round2_finding({"x": b, "y": b, "z": b}, {"x": True, "y": False, "z": False})["kind"] == "not established"


def test_replication_rule():
    b, a, e, i = "confirmed below zero", "confirmed above zero", "equivalent to zero", "inconclusive"
    gen = {"kind": "general", "sign": -1}
    null = {"kind": "general null", "sign": 0}
    comp = {"p": True, "q": True, "r": True}
    rd = lambda f, v, c=comp: Z.replication(f, v, c)["reading"]  # noqa: E731
    assert rd(gen, {"p": b, "q": b, "r": b}) == "replicated"
    assert rd(gen, {"p": b, "q": i, "r": b}) == "partial"
    assert rd(gen, {"p": b, "q": e, "r": b}) == "partial"
    assert rd(gen, {"p": e, "q": i, "r": i}) == "not replicated"
    assert rd(gen, {"p": a, "q": i, "r": i}) == "not replicated"
    assert rd(gen, {"p": i, "q": i, "r": i}) == "inconclusive"
    assert rd(gen, {"p": b, "q": b, "r": e}, {"p": True, "q": False, "r": False}) == "inconclusive"
    assert rd(gen, {"p": b, "q": b, "r": e}, {"p": True, "q": True, "r": False}) == "replicated"
    assert rd({"kind": "general", "sign": 1}, {"p": a, "q": a, "r": a}) == "replicated"
    assert rd({"kind": "general", "sign": 1}, {"p": b, "q": i, "r": i}) == "not replicated"
    assert rd(null, {"p": e, "q": e, "r": e}) == "replicated"
    assert rd(null, {"p": e, "q": b, "r": e}) == "not replicated"
    assert rd(null, {"p": e, "q": i, "r": i}) == "partial"
    assert rd(null, {"p": i, "q": i, "r": i}) == "inconclusive"
    assert rd({"kind": "model-specific", "sign": None}, {"p": b, "q": b, "r": b}) == "no replication target"
    out = Z.replication(gen, {"p": b, "q": i, "r": b}, comp)
    assert out["counted_as_replication"] is False
    assert Z.replication(gen, {"p": b, "q": b, "r": b}, comp)["counted_as_replication"] is True


def test_replication_block_waits_for_round_two():
    out = Z.replication_block(None, {}, {})
    assert all(v["reading"].startswith("waiting") for v in out.values())


def test_read_round2(tmp_path):
    assert Z.read_round2(tmp_path / "missing.json") is None
    (tmp_path / "m.json").write_text(json.dumps({"verdicts": "withheld: x"}))
    assert Z.read_round2(tmp_path / "m.json") is None
    (tmp_path / "m.json").write_text(json.dumps({"verdicts": {"t": {"contrasts": V(), "competent": {c: True for c in A.CONTRASTS},
                                                                    "other": 1}}}))
    assert Z.read_round2(tmp_path / "m.json") == {"t": {"contrasts": V(), "competent": {c: True for c in A.CONTRASTS}}}


def test_b3_family_and_pair_readings():
    rec = lambda est, lo, hi, pd, pe: {"standardized": {"estimate": est, "ci95": [lo, hi], "p_differs": pd, "p_equivalent": pe}}  # noqa: E731
    pairs = list(R3.PAIRS)
    diffs = {"qwen3-8b->qwen3.5-9b": {"lift": rec(0.5, 0.3, 0.7, 1e-5, 0.9), "parent_penalty": rec(-0.5, -0.7, -0.3, 1e-5, 0.9),
                                      "naming": rec(0.01, -0.05, 0.07, 0.8, 1e-5), "completion": rec(0.1, -0.2, 0.4, 0.4, 0.4),
                                      "timing": rec(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))}}
    fam = Z.b3_family(diffs, pairs)
    assert len(fam) == 30 and fam["mistral-7b-v0.3->ministral3-8b-it:lift:differs"]["p_in_family"] == 1.0
    k = "qwen3-8b->qwen3.5-9b"
    assert Z.pair_reading(diffs[k]["lift"], fam, f"{k}:lift") == "differs, newer higher"
    assert Z.pair_reading(diffs[k]["parent_penalty"], fam, f"{k}:parent_penalty") == "differs, newer lower"
    assert Z.pair_reading(diffs[k]["naming"], fam, f"{k}:naming") == "similar within the margin"
    assert Z.pair_reading(diffs[k]["completion"], fam, f"{k}:completion") == "not distinguishable"
    assert Z.pair_reading(None, fam, f"{k}:timing") == "waiting"
    unadj = {f"{k}:{c}:{t}": {"p_in_family": 0.01, "p_holm": 0.3} for c in A.CONTRASTS for t in A.TESTS}
    assert Z.pair_reading(diffs[k]["naming"], unadj, f"{k}:naming") == "not distinguishable"  # Holm, not the raw p
    off = rec(0.5, -0.1, 0.9, 1e-5, 0.9)  # p small but the interval crosses zero
    fam2 = Z.b3_family({k: {"lift": off}}, pairs)
    assert Z.pair_reading(off, fam2, f"{k}:lift") == "not distinguishable"
    assert Z.B3_MARGIN == 0.20


def test_family_a_has_thirty_tests_and_refuses_another_size(small, fast):
    scores = {t: _planted(small, t) for t in R3.CORE}
    ma = Z.model_analysis(small, SHOTS, scores, list(R3.CORE))
    out = Z.family_a("A3", list(R3.CORE), ma, None)
    assert out["family_size"] == 30 and out["waiting"] == []
    assert out["replication"]["lift"]["reading"].startswith("waiting")
    with pytest.raises(ValueError):
        Z.family_a("A3", list(R3.CORE)[:2], ma, None)
    part = Z.family_a("A3", list(R3.CORE), {**ma, "blocks": {k: v for k, v in ma["blocks"].items() if k != "gemma4-12b-it"}}, None)
    assert part["waiting"] == ["gemma4-12b-it"] and "H3_P" not in part


def test_model_analysis_withholds_a_model_with_missing_scores(small, fast):
    s = _planted(small, "qwen3.5-9b")[:-5]
    ma = Z.model_analysis(small, SHOTS, {"qwen3.5-9b": s}, ["qwen3.5-9b"])
    assert ma["missing"]["qwen3.5-9b"] == 5 and "qwen3.5-9b" not in ma["blocks"]


def test_round3_extras_are_descriptive_and_include_the_strict_floor(small, fast):
    by = {(s["transition_id"], s["condition"]): s for s in _planted(small, "m")}
    pops = A.populations(small)
    block = A.model_block(by, pops)
    ex = Z.r3_extras(by, pops, block)
    assert set(ex["standardized"]) == set(A.CONTRASTS) and set(ex["item_share"]) == {"lift", "parent_penalty"}
    assert ex["strict_floor"] == 0.80
    for c in A.CONTRASTS:  # the stricter floor can only remove competence
        assert not ex["competent_at_strict_floor"][c] or ex["competent"][c]
    pc = block["competence"]["local"]["populations"]
    for name in pc:
        pc[name]["yes_no_auc"] = 0.75  # between the two floors
    ex2 = Z.r3_extras(by, pops, block)
    assert not any(ex2["competent_at_strict_floor"].values())


def test_golden_tokens_check():
    case = {"n_ids": 10, "ids_sha256": "a"}
    r2g = {"models": {"qwen3-8b": {"revision": "r", "x|aligned": case}}}
    r3g = {"models": {"qwen3-8b-serve2": {"revision": "r", "x|aligned": case}}}
    assert Z.golden_tokens_check(r3g, r2g, "qwen3-8b")["ok"]
    r3g["models"]["qwen3-8b-serve2"]["x|aligned"] = {"n_ids": 11, "ids_sha256": "a"}
    assert not Z.golden_tokens_check(r3g, r2g, "qwen3-8b")["ok"]
    assert not Z.golden_tokens_check({"models": {}}, r2g, "qwen3-8b")["ok"]


def test_format_check_one_fifth_rule(records):
    recs = S.format_subset(records)
    native = {}
    for r in recs:
        native[(r["id"], "aligned")] = {"L": 1.0 + random.Random(r["id"]).random()}
        native[(r["id"], "swapped")] = {"L": 0.0}
        native[(r["id"], "yn_aligned")] = {"L": 1.0}
        native[(r["id"], "yn_swapped")] = {"L": 0.0}
    same = Z.format_block(native, dict(native), records)
    assert not same["two_candidate"]["format_dependent"]
    moved = {k: {"L": v["L"] + (0.6 if k[1] == "aligned" else 0.0)} for k, v in native.items()}
    out = Z.format_block(native, moved, records)
    assert out["two_candidate"]["format_dependent"] and not out["yes_no"]["format_dependent"]
    small_move = {k: {"L": v["L"] + (0.05 if k[1] == "aligned" else 0.0)} for k, v in native.items()}
    assert not Z.format_block(native, small_move, records)["two_candidate"]["above_one_fifth"]
    mean_native = Z.format_block(native, dict(native), records)["two_candidate"]["native_effect"]
    sixth = {k: {"L": v["L"] + (mean_native * 0.15 if k[1] == "aligned" else 0.0)} for k, v in native.items()}
    out = Z.format_block(native, sixth, records)["two_candidate"]
    assert out["ci_excludes_zero"] and not out["above_one_fifth"] and not out["format_dependent"]


# ---------------- end to end, mock ----------------


def test_golden_file_covers_every_tag_and_matches_round_two_for_the_earlier_models():
    """The golden renderings made on ICE in serve2 (job 7246224): every round-3 tag and check variant, round 2's five
    cases each; the earlier models' renderings equal round 2's (made in transformers 4.51.3), text and ids: part 1 of
    the environment check."""
    g3 = json.loads(T.GOLDEN.read_text())
    g2 = json.loads((T.GOLDEN.parent / "r2_chat_golden.json").read_text())
    want = set(R3.MODELS) | {"gemma4-12b-it-firstturn", "qwen3.8-27b-nopreserve"} | {f"{t}-serve2" for t in R3.RESCORE_MODELS}
    assert set(g3["models"]) == want
    assert [(c["id"], c["arm"]) for c in g3["cases"]] == [(c["id"], c["arm"]) for c in g2["cases"]]
    for tag, rend in g3["models"].items():
        assert rend["revision"] == R3.spec_for(tag).revision
        assert len([k for k in rend if k != "revision"]) == 5
    for t in R3.RESCORE_MODELS:
        assert Z.golden_tokens_check(g3, g2, t)["ok"]
        assert all(g3["models"][t + "-serve2"][k]["text"] == g2["models"][t][k]["text"] for k in g2["models"][t] if k != "revision")
    q = next(v for k, v in g3["models"]["qwen3.5-9b"].items() if k.endswith("|aligned"))
    assert q["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and R.SYSTEM in q["text"]
    g = next(v for k, v in g3["models"]["gemma4-12b-it"].items() if k.endswith("|aligned"))
    assert g["text"].endswith("<|turn>model\n<|channel>thought\n<channel|>")


# ---------------- review 1 fixes (design section 14) ----------------


def test_budget_forcing_and_the_readout_point():
    assert K.close_reasoning("think a bit</think>\n\nCrafted 1") == ("think a bit</think>", None)
    assert K.close_reasoning("still thinking", K.MAX_REASONING) == ("still thinking\n</think>", K.CAP)
    assert K.close_reasoning("short and stopped", 12) == ("short and stopped\n</think>", K.ENDED)
    assert K.close_reasoning("cut<|im_end|>junk", K.MAX_REASONING) == ("cut\n</think>", K.ENDED)
    assert K.AFTER_CLOSE == "\n\n"
    rec = {"object": "oak planks"}
    assert K.mentions("The goal needs oak planks", rec) == {"goal_word": True, "crafted_item": True}
    assert K.mentions("nothing", rec) == {"goal_word": False, "crafted_item": False}


def test_h3p_reading():
    below, eq = "confirmed below zero", "equivalent to zero"
    v = {"a": V(parent_penalty=below), "b": V(parent_penalty=below), "c": V(parent_penalty=below)}
    comp = {"a": True, "b": True, "c": True}
    r1 = {t: False for t in v}
    r2 = {"fires": False}
    assert Z.h3p_reading(v, comp, r1, r2)["overall"] == "supported"
    out = Z.h3p_reading({**v, "c": V()}, comp, r1, r2)
    assert out["overall"] == "partial" and out["per_model"]["c"] == "inconclusive"
    out = Z.h3p_reading({**v, "c": V(parent_penalty=eq)}, comp, {**r1, "c": True}, r2)
    assert out["overall"] == "partial" and out["per_model"]["c"] == "rejected"
    only = {"a": V(parent_penalty=eq), "b": V(), "c": V()}
    assert Z.h3p_reading(only, comp, {"a": True, "b": False, "c": False}, r2)["overall"] == "rejected"
    assert Z.h3p_reading(v, comp, r1, {"fires": True})["overall"] == "rejected"
    # S1: one competent model, supported, is "inconclusive (fewer than two competent)", not "partial"
    one = Z.h3p_reading(v, {"a": True, "b": False, "c": False}, r1, r2)
    assert one["overall"] == "inconclusive" and "fewer than 2" in one["reason"]
    assert one["per_model"]["b"].startswith("measurable only") and one["per_model"]["a"] == "supported"
    none = Z.h3p_reading({t: V() for t in v}, {t: False for t in v}, r1, r2)
    assert none["overall"] == "inconclusive" and "fewer than 2" in none["reason"]
    assert Z.h3p_reading({t: V() for t in v}, comp, r1, r2)["overall"] == "inconclusive"


# M1 and S12: provenance per job, per family, at the freeze commit


def _shard(tmp_path, name, jobs):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (d / "git_sha").write_text(jobs[0].get("git_sha", "c1") + "\n")
    return d


def _git(tag_commit="c1", diffs=None):
    diffs = diffs or {}

    def run(args):
        if args[:2] == ["tag", "-l"]:
            return "textcraft-r3-design-frozen-2026-11-30" if tag_commit else ""
        if args[0] == "rev-parse":
            return tag_commit if "frozen" in args[1] else "head"
        if args[0] == "diff":
            return diffs.get(args[3], "wf/x.py")
        raise AssertionError(args)
    return run


def test_provenance_accepts_two_l40s_per_job_and_checks_each_job(tmp_path):
    two = ["NVIDIA L40S", "NVIDIA L40S"]
    pin = C.REVISIONS["Qwen/Qwen3.8-27B"]
    ok = _shard(tmp_path, "a/shard_00_of_04", [{"gpu": two, "git_sha": "c1"}, {"gpu": two, "git_sha": "c1"}])
    probs = Z.provenance_problems({"qwen3.8-27b": [ok]}, {"qwen3.8-27b": [{"model_revision": pin}]}, "c1", run_git=_git())
    assert probs == {}
    one = _shard(tmp_path, "b/shard_01_of_04", [{"gpu": ["NVIDIA L40S"], "git_sha": "c1"}])
    probs = Z.provenance_problems({"qwen3.8-27b": [ok, one]}, {}, "c1", run_git=_git())
    assert any("different" in p for p in probs["qwen3.8-27b"])
    a100 = ["NVIDIA A100-SXM4-80GB"]
    mixed = _shard(tmp_path, "c/shard_00_of_01", [{"gpu": a100, "git_sha": "c1"}, {"gpu": two, "git_sha": "c1"}])
    assert Z.provenance_problems({"gemma4-31b-it": [mixed]}, {}, "c1", run_git=_git())["gemma4-31b-it"]
    single = _shard(tmp_path, "d/shard_00_of_01", [{"gpu": ["NVIDIA L40S"], "git_sha": "c1"}])
    assert any("not the pinned" in p for p in
               Z.provenance_problems({"gemma4-31b-it": [single]}, {}, "c1", run_git=_git())["gemma4-31b-it"])


def test_provenance_requires_the_freeze_commit_and_one_revision(tmp_path):
    l40 = ["NVIDIA L40S"]
    pin = C.REVISIONS["Qwen/Qwen3.5-9B"]
    late = _shard(tmp_path, "a/shard_00_of_01", [{"gpu": l40, "git_sha": "c1"}, {"gpu": l40, "git_sha": "c2"}])
    probs = Z.provenance_problems({"qwen3.5-9b": [late]}, {}, "c1", run_git=_git())
    assert any("freeze tag" in p for p in probs["qwen3.5-9b"])
    # a later commit that changes nothing under wf/, scripts/ or pace/ is code-identical
    assert Z.provenance_problems({"qwen3.5-9b": [late]}, {}, "c1", run_git=_git(diffs={"c2": "notes/x.md\nREADME.md"})) == {}
    assert Z.provenance_problems({"qwen3.5-9b": [late]}, {}, None, run_git=_git())["qwen3.5-9b"] == \
        ["no round-3 freeze tag in this checkout"]
    mixed = Z.provenance_problems({}, {"qwen3.5-9b": [{"model_revision": pin}, {"model_revision": "other"}]}, "c1")
    assert any("revisions" in p for p in mixed["qwen3.5-9b"])  # the pinned one among others is not enough
    assert Z.provenance_problems({}, {"qwen3.5-9b": [{"model_revision": pin}]}, "c1") == {}


def test_a_family_with_a_provenance_problem_is_refused_alone(small, fast):
    scores = {t: _planted(small, t) for t in list(R3.CORE) + list(R3.EXTENSION)}
    out = Z.analyse(small, SHOTS, scores, {}, None, None, {"status": "waiting"}, refused={"A3x": ["x"]})
    assert out["A3x"]["refused"] == ["x"] and "H3_P" in out["A3"]


# M2: "not scored" from the filed gate record


def _gate_rec(tmp_path, run, rec, rows=None):
    d = tmp_path / run / "shard_00_of_01"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenization_checks.json").write_text(json.dumps(rec))
    if rows is not None:
        (d / "scoring.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_model_state_from_filed_gate_records(tmp_path):
    assert G.model_state(tmp_path, "ministral3-8b-it")["state"] == "waiting"
    _gate_rec(tmp_path, "textcraft_r3_smoke_ministral3-8b-it", {"ok": True})
    assert G.model_state(tmp_path, "ministral3-8b-it")["state"] == "waiting"
    _gate_rec(tmp_path, "textcraft_r3_score_ministral3-8b-it", {"ok": True})
    assert G.model_state(tmp_path, "ministral3-8b-it")["state"] == "scored"
    _gate_rec(tmp_path, "textcraft_r3_smoke_qwen3.5-9b", {"ok": False, "loading_failed": "missing keys"})
    st = G.model_state(tmp_path, "qwen3.5-9b")
    assert st["state"] == "not scored" and "loading check failed" in st["reasons"][0]
    _gate_rec(tmp_path, "textcraft_r3_score_olmo3-7b-it", {"ok": False, "attention_agreement": {"ok": False}})
    assert "attention" in G.model_state(tmp_path, "olmo3-7b-it")["reasons"][0]
    _gate_rec(tmp_path, "textcraft_r3_score_gemma4-31b-it", {"ok": False, "problems": ["x"]})
    assert "tokenization gate" in G.model_state(tmp_path, "gemma4-31b-it")["reasons"][0]
    _gate_rec(tmp_path, "textcraft_r3_smoke_qwen3.8-27b", {"ok": None, "skipped": "mock"})
    assert G.model_state(tmp_path, "qwen3.8-27b")["state"] == "waiting"


def test_a_core_model_not_scored_enters_at_p_one_and_the_readings_proceed(small, fast):
    scores = {t: _planted(small, t) for t in ("qwen3.5-9b", "gemma4-12b-it") + R3.EXTENSION}
    states = {"ministral3-8b-it": {"state": "not scored", "reasons": ["loading check failed"]}}
    out = Z.analyse(small, SHOTS, scores, {}, None, None, {"status": "waiting"}, states=states)
    a3 = out["A3"]
    assert a3["not_scored"] == ["ministral3-8b-it"] and a3["waiting"] == []
    assert all(a3["tests"][f"ministral3-8b-it:{c}:{t}"]["p_in_family"] == 1.0 for c in A.CONTRASTS for t in A.TESTS)
    assert a3["verdicts"]["ministral3-8b-it"]["contrasts"]["parent_penalty"] == "not scored"
    assert "H3_P" in a3 and "replication" in a3 and "R2_contemporary" in a3
    assert "ministral3-8b-it" not in a3["H3_P"]["competent"]
    assert all(v == "not scored" for c, v in out["B3"]["readings"]["mistral-7b-v0.3->ministral3-8b-it"].items()
               if c in A.CONTRASTS)
    waiting = Z.analyse(small, SHOTS, {t: s for t, s in scores.items() if t != "qwen3.5-9b"}, {}, None, None,
                        {"status": "waiting"}, states=states)
    assert waiting["A3"]["waiting"] == ["qwen3.5-9b"] and "H3_P" not in waiting["A3"]


def test_a_loading_failure_is_filed_and_stops_the_model(mock_runs, small, monkeypatch):
    man = _manifest(mock_runs, small)

    def broken(spec, tag=None):
        raise RuntimeError("google/gemma-4-12B-it: loading check failed: ['missing: x']")

    monkeypatch.setattr(FAM, "open_scorer", broken)
    cfg = {"run_id": "textcraft_r3_smoke_gemma4-12b-it", "model": "gemma4-12b-it", "smoke": True, "manifest": man}
    assert S.run_textcraft_r3_score(cfg) == 1
    rec = json.loads((mock_runs / "runs" / "textcraft_r3_smoke_gemma4-12b-it" / "shard_00_of_01" /
                      "tokenization_checks.json").read_text())
    assert rec["ok"] is False and "loading check failed" in rec["loading_failed"]
    assert G.model_state(mock_runs / "runs", "gemma4-12b-it")["state"] == "not scored"
    monkeypatch.setattr(FAM, "open_scorer", lambda spec, tag=None: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    with pytest.raises(RuntimeError):  # any other failure is not filed as a gate failure
        S.run_textcraft_r3_score({**cfg, "run_id": "textcraft_r3_smoke_olmo3-7b-it", "model": "olmo3-7b-it"})


# M3 and S9: the Gemma fallback as code


def _yn(mass, top="Yes", special=False):
    lp = math.log(mass / 2)
    return {"condition": "yn_aligned", "logp_plus": lp, "logp_minus": lp, "top_token": top, "top_token_special": special}


def test_gemma_slot_rule(tmp_path):
    run = f"textcraft_r3_smoke_{R3.GEMMA_CORE}"
    assert G.gemma_slot(tmp_path)["slot"] == "waiting"
    _gate_rec(tmp_path, run, {"ok": True}, [_yn(0.9)] * 4)
    assert G.gemma_slot(tmp_path)["slot"] == R3.GEMMA_CORE
    for rec, why in (({"ok": False, "loading_failed": "x"}, "loading"), ({"ok": False, "problems": ["x"]}, "gate"),
                     ({"ok": False, "attention_agreement": {"ok": False}}, "attention")):
        _gate_rec(tmp_path, run, rec, [_yn(0.9)] * 4)
        out = G.gemma_slot(tmp_path)
        assert out["slot"] == R3.GEMMA_FALLBACK and why in out["triggers"][0]
    # S9: low mass alone does not fire (it goes to the competence gate); with structural next tokens it does
    _gate_rec(tmp_path, run, {"ok": True}, [_yn(0.1, "No")] * 4)
    assert G.gemma_slot(tmp_path)["slot"] == R3.GEMMA_CORE
    _gate_rec(tmp_path, run, {"ok": True}, [_yn(0.1, "<|channel>", True)] * 3 + [_yn(0.1, "Yes")])
    out = G.gemma_slot(tmp_path)
    assert out["slot"] == R3.GEMMA_FALLBACK and "format mismatch" in out["triggers"][0]
    _gate_rec(tmp_path, run, {"ok": True}, [_yn(0.9, "**")] * 4)  # structural but enough mass: no
    assert G.gemma_slot(tmp_path)["slot"] == R3.GEMMA_CORE
    _gate_rec(tmp_path, run, {"ok": True}, [_yn(0.1, "**")] * 2 + [_yn(0.1, "Yes")] * 2)  # half is not most
    assert G.gemma_slot(tmp_path)["slot"] == R3.GEMMA_CORE


def test_structural_tokens():
    for t, sp in (("<|channel>", False), ("  ", False), ("**", False), ("#", False), ("`", False), ("x", True)):
        assert G.structural(t, sp)
    for t in ("Yes", " No", "yes", "The", "C"):
        assert not G.structural(t, False)


def test_the_fallback_model_runs_only_when_the_rule_fired(mock_runs, small):
    man = _manifest(mock_runs, small)
    cfg = {"run_id": "textcraft_r3_smoke_gemma4-e4b-it", "model": "gemma4-e4b-it", "smoke": True, "manifest": man}
    with pytest.raises(SystemExit):
        S.run_textcraft_r3_score(cfg)
    _gate_rec(mock_runs / "runs", f"textcraft_r3_smoke_{R3.GEMMA_CORE}", {"ok": False, "problems": ["x"]}, [])
    assert S.run_textcraft_r3_score(cfg) == 0


def test_smoke_runs_record_the_format_diagnostic(mock_runs, small):
    man = _manifest(mock_runs, small)
    assert S.run_textcraft_r3_score({"run_id": "textcraft_r3_smoke_qwen3.5-9b", "model": "qwen3.5-9b", "smoke": True,
                                     "manifest": man}) == 0
    rows = Z.scores_of("textcraft_r3_smoke_qwen3.5-9b", "qwen3.5-9b", mock_runs / "runs")
    yn = [r for r in rows if r["condition"].startswith("yn_")]
    assert yn and all("top_token" in r for r in yn) and not any("top_token" in r for r in rows if r not in yn)
    assert G.smoke_diagnostic(rows)["n_top_tokens"] == len(yn)


def test_the_analysis_refuses_a_gemma_core_the_rule_does_not_give(mock_runs, small):
    man = _manifest(mock_runs, small)
    assert Z.run_textcraft_r3_analysis({"run_id": "textcraft_r3_analysis", "manifest": man,
                                        "gemma_core": "gemma4-e4b-it"}) == 1
    status = (mock_runs / "runs" / "textcraft_r3_analysis" / "STATUS.md").read_text()
    assert "fallback rule gives 'gemma4-12b-it'" in status


def test_the_fallback_slot_moves_every_core_role(small, fast):
    scores = {t: _planted(small, t) for t in ("qwen3.5-9b", "ministral3-8b-it", "gemma4-e4b-it") + R3.EXTENSION}
    out = Z.analyse(small, SHOTS, scores, {}, None, None, {"status": "waiting"}, gemma="gemma4-e4b-it")
    assert out["A3"]["tags"] == ["qwen3.5-9b", "ministral3-8b-it", "gemma4-e4b-it"] and out["fallback_used"]
    assert "gemma2-9b-it->gemma4-e4b-it" in out["B3"]["readings"]


# M4: the environment check mirrors round 2's shards


def _r2_shards(tmp_path, tag, gpus):
    for i, g in enumerate(gpus):
        jobs = g if isinstance(g, list) and g and isinstance(g[0], list) else [g]
        _shard(tmp_path, f"textcraft_r2_score_{tag}/shard_{i:02d}_of_{len(gpus):02d}", [{"gpu": j} for j in jobs])


def test_envcheck_plan_from_round_two_gpu_names(tmp_path, records):
    from wf.textcraft.r2_score import shard_records

    L, A40 = ["NVIDIA L40S"], ["NVIDIA A40"]
    _r2_shards(tmp_path, "qwen3-8b", [L, L, L, L])
    p = G.envcheck_plan(tmp_path, "qwen3-8b", records)
    assert p["single_type"] and p["runnable"] and p["n_comparable_planned"] == 300
    assert p["jobs"] == [{"job": 0, "gpu": "NVIDIA L40S", "r2_shards": [0, 1, 2, 3]}]
    _r2_shards(tmp_path, "mistral-7b-v0.3", [L, A40, [L, A40], None])
    p = G.envcheck_plan(tmp_path, "mistral-7b-v0.3", records)
    assert [s["typed"] for s in p["shards"]] == [True, True, False, False]
    assert p["jobs"] == [{"job": 0, "gpu": "NVIDIA L40S", "r2_shards": [0]}, {"job": 1, "gpu": "NVIDIA A40", "r2_shards": [1]}]
    seeded = {r["id"] for r in S.envcheck_subset(records)}
    assert p["shards"][0]["n_records"] == sum(r["id"] in seeded for r in shard_records(records, 0, 4))
    assert p["n_comparable_planned"] == p["shards"][0]["n_records"] + p["shards"][1]["n_records"]
    assert not p["runnable"]  # 75 + 55 seeded records
    assert G.envcheck_plan(tmp_path, "gemma2-9b-it", records)["status"] == "waiting"


def test_envcheck_jobs_score_round_two_shards_on_the_exact_gpu(mock_runs, records, monkeypatch):
    from wf.textcraft.r2_score import shard_records

    plan = {"models": {"qwen3-8b": {"runnable": True, "n_r2_shards": 4, "jobs": [
        {"job": 0, "gpu": "NVIDIA L40S", "r2_shards": [0]}, {"job": 1, "gpu": "NVIDIA A40", "r2_shards": [1, 3]}]}}}
    job = G.plan_job("qwen3-8b", 1, plan)
    recs = S.envcheck_job_records(records, job)
    ids = {r["id"] for i in (1, 3) for r in shard_records(records, i, 4)}
    assert recs and all(r["id"] in ids for r in recs) and len(recs) == len([r for r in S.envcheck_subset(records) if r["id"] in ids])
    with pytest.raises(SystemExit):
        G.plan_job("qwen3-8b", 2, plan)
    with pytest.raises(SystemExit):
        G.plan_job("qwen3-8b", 0, {"models": {"qwen3-8b": {"runnable": False, "jobs": []}}})
    monkeypatch.setenv("WF_MOCK_SCORE", "0")
    with pytest.raises(SystemExit):
        S.gpu_check("qwen3-8b-serve2", ["NVIDIA L40S"], exact=["NVIDIA A40"])
    assert S.gpu_check("qwen3-8b-serve2", ["NVIDIA A40"], exact=["NVIDIA A40"])["checked"] is True
    with pytest.raises(SystemExit):
        S.gpu_check("qwen3-8b-serve2", ["NVIDIA A40", "NVIDIA A40"], exact=["NVIDIA A40"])
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    man = _manifest(mock_runs, records)
    assert S.run_textcraft_r3_envcheck({"run_id": "textcraft_r3_envcheck_qwen3-8b", "model": "qwen3-8b",
                                        "manifest": man, "job": 1, "plan": plan}) == 0
    d = mock_runs / "runs" / "textcraft_r3_envcheck_qwen3-8b" / "shard_01_of_02"
    got = {json.loads(x)["transition_id"] for x in (d / "scoring.jsonl").read_text().splitlines()}
    assert got == {r["id"] for r in recs}


def _env_scores(recs, tag, shift=0.0, model=None, only_non_main=False):
    out = []
    for r in recs:
        for a in r["arms"]:
            base = (hash((r["id"], a)) % 100) / 10
            bump = shift if a in ("aligned", "yn_aligned") else 0.0
            if only_non_main and r["kind"] == "main" and r["population"] == "main":
                bump = 0.0
            out.append({"transition_id": r["id"], "condition": a, "model": model or tag, "L": base + bump})
    return out


def test_environment_check_matches_gpus_record_by_record(records, tmp_path):
    from wf.textcraft.r2_score import shard_records

    seeded = S.envcheck_subset(records)
    parts = {i: [r for r in shard_records(records, i, 4) if r in seeded] for i in range(4)}
    gpus = {}
    old, new = [], []
    for i in range(4):
        po, pn = tmp_path / f"old{i}", tmp_path / f"new{i}"
        gpus[po] = gpus[pn] = ["NVIDIA L40S"] if i % 2 == 0 else ["NVIDIA A40"]
        old.append((po, _env_scores(parts[i], "qwen3-8b")))
        new.append((pn, _env_scores(parts[i], "qwen3-8b", model="qwen3-8b-serve2")))
    out = Z.envcheck_block(old, new, "qwen3-8b", records, gpus.get)
    assert out["status"] == "pass" and out["n_comparable_records"] == 300
    gpus[tmp_path / "new3"] = ["NVIDIA L40S"]  # shard 3 rescored on another type: its records are not comparable
    out = Z.envcheck_block(old, new, "qwen3-8b", records, gpus.get)
    assert out["n_comparable_records"] == 300 - len(parts[3])
    gpus[tmp_path / "new3"] = None  # a rescoring shard on mixed GPUs (untyped): not comparable either
    assert Z.envcheck_block(old, new, "qwen3-8b", records, gpus.get)["n_comparable_records"] == 300 - len(parts[3])
    gpus[tmp_path / "new0"] = gpus[tmp_path / "new3"] = ["NVIDIA H100"]
    few = Z.envcheck_block(old, new, "qwen3-8b", records, gpus.get)
    assert few["status"] == "waiting" and few["pass"] is None
    assert Z.envcheck_block(old, new, "qwen3-8b", records, gpus.get, unobtainable=True)["status"] == "not run (GPU types)"


def test_environment_check_criteria(records, tmp_path):
    seeded = S.envcheck_subset(records)
    g = {tmp_path / "o": ["NVIDIA L40S"], tmp_path / "n": ["NVIDIA L40S"]}
    old = [(tmp_path / "o", _env_scores(seeded, "qwen3-8b"))]
    run = lambda **kw: Z.envcheck_block(old, [(tmp_path / "n", _env_scores(seeded, "qwen3-8b", model="qwen3-8b-serve2", **kw))],  # noqa: E731
                                        "qwen3-8b", records, g.get)
    assert run()["status"] == "pass"
    bad = run(shift=0.2)
    assert bad["status"] == "fail" and not bad["checks"]["two_candidate_goal_effect"]
    # the goal effect is read on the main failures only: a shift on other records moves no effect criterion
    other = run(shift=0.2, only_non_main=True)
    assert other["checks"]["two_candidate_goal_effect"] and other["checks"]["yes_no_goal_effect"]
    assert other["two_candidate"]["n_main_failures"] == sum(r["kind"] == "main" and r["population"] == "main" for r in seeded)
    half = [(tmp_path / "n", _env_scores(seeded[:150], "qwen3-8b", model="qwen3-8b-serve2"))]
    assert Z.envcheck_block(old, half, "qwen3-8b", records, g.get)["status"] == "waiting"
    drop = {r["id"] for r in seeded[:10]}
    gap = [(tmp_path / "n", [x for x in _env_scores(seeded, "qwen3-8b", model="qwen3-8b-serve2")
                            if not (x["transition_id"] in drop and x["condition"] == "yn_swapped")])]
    out = Z.envcheck_block(old, gap, "qwen3-8b", records, g.get)
    assert not out["checks"]["complete"] and out["status"] == "fail"


def test_environment_check_share_criterion(records, tmp_path):
    seeded = S.envcheck_subset(records)
    mf = [r for r in seeded if r["kind"] == "main" and r["population"] == "main"]
    old_rows, new_rows = [], []
    for k, r in enumerate(seeded):
        for a in r["arms"]:
            L = 0.0
            if a == "yn_aligned" and r in mf:
                L = 0.01 if k % 2 else -0.01  # aligned minus swapped straddles zero by a hundredth
            old_rows.append({"transition_id": r["id"], "condition": a, "model": "qwen3-8b", "L": L})
            new_rows.append({"transition_id": r["id"], "condition": a, "model": "qwen3-8b-serve2",
                             "L": abs(L) if a == "yn_aligned" else L})  # every item now toward yes
    g = {tmp_path / "o": ["NVIDIA L40S"], tmp_path / "n": ["NVIDIA L40S"]}
    out = Z.envcheck_block([(tmp_path / "o", old_rows)], [(tmp_path / "n", new_rows)], "qwen3-8b", records, g.get)
    assert out["checks"]["yes_no_goal_effect"] and out["checks"]["yes_no_per_score"]
    assert not out["checks"]["yes_no_share"] and out["status"] == "fail"


def test_environment_status_and_b3_source():
    ok_tok = {t: {"ok": True} for t in R3.ENVCHECK_MODELS}
    st = lambda *s: {t: {"status": x} for t, x in zip(R3.ENVCHECK_MODELS, s)}  # noqa: E731
    assert Z.environment_status(st("pass", "pass", "pass"), ok_tok) == "pass"
    assert Z.environment_status(st("pass", "fail", "not run (GPU types)"), ok_tok) == "fail"
    assert Z.environment_status(st("pass", "not run (GPU types)", "pass"), ok_tok) == "not run (GPU types)"
    assert Z.environment_status(st("pass", "not run (GPU types)", "waiting"), ok_tok) == "waiting"
    assert Z.environment_status(st("pass", "pass"), ok_tok) == "waiting"  # Gemma 2 required too (S8)
    assert Z.environment_status(st("pass", "pass", "pass"), {**ok_tok, "gemma2-9b-it": {"ok": False}}) == "fail"
    assert Z.b3_source("pass", {}) == "round2" and Z.b3_source("fail", {}) == "rescore"
    assert Z.b3_source("not run (GPU types)", {}) == "rescore" and Z.b3_source("waiting", {}) == "waiting"
    with pytest.raises(ValueError):
        Z.b3_source("pass", {"b3_old_scores": "rescore"})
    assert R3.ENVCHECK_MODELS == ("qwen3-8b", "mistral-7b-v0.3", "gemma2-9b-it")


def test_b3_follows_the_environment_check(small, fast, monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    scores = {t: _planted(small, t) for t in list(R3.CORE) + list(R3.EXTENSION)}
    old = {t: _planted(small, t) for t, _n in R3.PAIRS}
    wait = Z.analyse(small, SHOTS, scores, {}, None, old, {"status": "waiting"})
    assert wait["B3"]["differences"] == {} and wait["B3"]["old_scores"] == "waiting"
    assert all(v == "waiting for the environment check" for r in wait["B3"]["readings"].values() for c, v in r.items()
               if c in A.CONTRASTS)
    failed = Z.analyse(small, SHOTS, scores, {}, None, old, {"status": "fail"})
    assert failed["B3"]["old_scores"] == "rescore" and set(failed["B3"]["differences"]) == {f"{o}->{n}" for o, n in R3.PAIRS}
    with pytest.raises(ValueError):
        Z.analyse(small, SHOTS, scores, {"b3_old_scores": "round2"}, None, old, {"status": "fail"})
    refused = Z.analyse(small, SHOTS, scores, {}, None, old, {"status": "pass"}, refused={"B3": ["round 2: x"]})
    assert refused["B3"]["differences"] == {}
    assert all(v == "refused (provenance)" for r in refused["B3"]["readings"].values() for c, v in r.items() if c in A.CONTRASTS)


def test_b3_uses_twenty_thousand_resamples_outside_mock(small, fast, monkeypatch):
    scores = {t: _planted(small, t) for t in list(R3.CORE) + list(R3.EXTENSION)}
    old = {t: _planted(small, t) for t, _n in R3.PAIRS}
    monkeypatch.setenv("WF_MOCK_SCORE", "0")
    with pytest.raises(ValueError):
        Z.analyse(small, SHOTS, scores, {"b_draws": 200}, None, old, {"status": "pass"})
    out = Z.analyse(small, SHOTS, scores, {}, None, old, {"status": "pass"})
    d = out["B3"]["differences"]["qwen3-8b->qwen3.5-9b"]["lift"]["standardized"]
    assert d["n_draws"] == Z.B_DRAWS == 300  # the family's draw count (the fast fixture's stand-in for 20,000)


def test_planted_pairs_are_read_on_the_standardized_scale(small, fast, monkeypatch):
    """A newer model that is the older one at three times the logit scale differs in nats but not standardized."""
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    scores = {t: _planted(small, t, scale=1.0) for t in list(R3.CORE) + list(R3.EXTENSION)}
    old = {}
    for o, n in R3.PAIRS:
        base = _planted(small, n, scale=1.0)
        old[o] = [{**s, "model": o, "L": s["L"] / 3, "logp_minus": -1.0 - s["L"] / 3,
                   "logp_first_minus": -0.5 - s["L"] / 6} for s in base]
    out = Z.analyse(small, SHOTS, scores, {"b_draws": 200}, None, old, {"status": "pass"})
    d = out["B3"]["differences"]["qwen3-8b->qwen3.5-9b"]["lift"]
    assert d["standardized"]["estimate"] == pytest.approx(0, abs=1e-9)
    assert d["raw_nats_descriptive"]["estimate"] > 0
    assert "fixed_denominator_descriptive" in d and "equivalence_power_at_zero_holm_first_step" in d["standardized"]
    assert not out["B3"]["readings"]["qwen3-8b->qwen3.5-9b"]["lift"].startswith("differs, newer higher")


# S2, S3, S4, S11


def test_round_three_power_uses_holm_first_step_of_thirty(small, fast):
    by = {(s["transition_id"], s["condition"]): s for s in _planted(small, "m")}
    block = A.model_block(by, A.populations(small))
    rec = block["contrasts"]["lift"]
    Z.holm30_power(block)
    assert rec["holm_first_step_z"] == R3.Z_FIRST_30
    assert rec["equivalence_power_at_zero_holm_first_step"] == pytest.approx(
        A.equivalence_power(rec["bootstrap_se"], rec["margin"], 2.935199))
    assert rec["equivalence_underpowered"] == (not rec["equivalence_power_at_zero_holm_first_step"] >= 0.8)
    assert "equivalence_power_at_zero_holm_first_step_round2_family" in rec
    assert abs(statistics.NormalDist().inv_cdf(1 - 0.05 / 30) - R3.Z_FIRST_30) < 1e-5


def test_b3_equivalence_power_and_the_underpowered_flag():
    old, new = _values(seed=11), _values(seed=12)
    d = Z.paired_difference(old, new, False, B=300)["standardized"]
    se = d["bootstrap_se"]
    assert d["equivalence_power_at_zero_holm_first_step"] == pytest.approx(A.equivalence_power(se, 0.20, R3.Z_FIRST_30))
    assert d["equivalence_power_at_zero_alpha_05"] == pytest.approx(A.equivalence_power(se, 0.20, R3.Z_ALPHA))
    k = "qwen3-8b->qwen3.5-9b"
    fam = {f"{k}:{c}:{t}": {"p_holm": 0.5} for c in A.CONTRASTS for t in A.TESTS}
    rec = {"standardized": {"estimate": 0.0, "ci95": [-0.3, 0.3], "old": -0.5, "new": -0.5,
                            "equivalence_underpowered": True}}
    assert Z.pair_reading(rec, fam, f"{k}:timing") == "not distinguishable (equivalence underpowered)"
    rec["standardized"]["equivalence_underpowered"] = False
    assert Z.pair_reading(rec, fam, f"{k}:timing") == "not distinguishable"


def test_b3_fixed_denominator_column():
    old, new = _values(seed=21), _values(seed=22)
    d = Z.paired_difference(old, new, False, B=100, fixed_sd=(2.0, 4.0))["fixed_denominator_descriptive"]
    ro = statistics.mean(x for _r, x, _p in old.values())
    rn = statistics.mean(x for _r, x, _p in new.values())
    assert d["estimate"] == pytest.approx(rn / 4.0 - ro / 2.0) and d["ci95"][0] <= d["ci95"][1]


def test_pair_readings_say_the_direction_in_plain_words():
    k = "qwen3-8b->qwen3.5-9b"
    fam = {f"{k}:{c}:{t}": {"p_holm": 0.001 if t == "differs" else 0.9} for c in A.CONTRASTS for t in A.TESTS}
    rec = lambda o, n: {"standardized": {"estimate": n - o, "ci95": [n - o - 0.05, n - o + 0.05], "old": o, "new": n}}  # noqa: E731
    assert Z.pair_reading(rec(-0.8, -0.3), fam, f"{k}:parent_penalty") == \
        "differs, newer higher (smaller in size (a smaller penalty))"
    assert Z.pair_reading(rec(-0.3, -0.8), fam, f"{k}:parent_penalty") == \
        "differs, newer lower (larger in size (a larger penalty))"
    assert Z.pair_reading(rec(0.3, 0.8), fam, f"{k}:lift") == "differs, newer higher (larger in size)"
    assert Z.pair_reading(rec(-0.3, 0.4), fam, f"{k}:lift") == "differs, newer higher (opposite signs)"


def test_exploratory_blocks_carry_no_p_values(small, fast, monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    scores = {t: _planted(small, t) for t in list(R3.CORE) + list(R3.EXTENSION)}
    think = [{**s, "model": R3.THINK_TAG, "forced": False, "n_new_tokens": 5} for s in _planted(small, "qwen3.5-9b")]
    out = Z.analyse(small, SHOTS, scores, {}, None, None, {"status": "waiting"}, think=think)
    text = json.dumps(out["exploratory"])
    assert '"p_differs"' not in text and '"p_equivalent"' not in text
    assert "qwen3.5-9b->qwen3.8-27b" in out["exploratory"] and "thinking_arm" in out["exploratory"]


# S10 and the reviewer's thinking-arm mutants


def test_thinking_unforced_subset_and_forced_reasons(small):
    base = [{**s, "model": R3.THINK_TAG, "n_new_tokens": 5} for s in _planted(small, "qwen3.5-9b")]
    for k, s in enumerate(base):
        s["forced"] = k % 3 == 0
        s["forced_reason"] = (K.CAP if k % 6 == 0 else K.ENDED) if s["forced"] else None
        if s["forced"]:
            s["L"] = 1e6  # a forced trace must not reach the unforced readouts
    off = {(s["transition_id"], s["condition"]): s for s in _planted(small, "qwen3.5-9b")}
    out = Z.think_block(base, off, small)
    assert out["unforced"]["n_scores"] == sum(not s["forced"] for s in base) < out["all"]["n_scores"]
    assert abs(out["unforced"]["parent_penalty"]["thinking_on"]["estimate"]) < 1e5
    assert out["cap_share"] + out["ended_unclosed_share"] == pytest.approx(out["forced_share"])
    assert out["cap_share"] == pytest.approx(sum(s["forced_reason"] == K.CAP for s in base) / len(base))


def test_thinking_batches_have_their_own_seeds_in_record_and_arm_order(mock_runs, records):
    man = _manifest(mock_runs, records)
    assert K.run_textcraft_r3_think({"run_id": "textcraft_r3_think_smoke_qwen3.5-9b", "manifest": man, "max_jobs": 24}) == 0
    d = mock_runs / "runs" / "textcraft_r3_think_smoke_qwen3.5-9b" / "shard_00_of_01"
    gens = [json.loads(x) for x in (d / "generations.jsonl").read_text().splitlines()]
    keys = [g["batch_key"] for g in gens]
    assert len(set(keys)) == 3 and all(g["seed"] == K.batch_seed(g["batch_key"]) for g in gens)
    assert len({g["seed"] for g in gens}) == 3
    jobs = K.think_jobs(records)[:24]
    assert [(g["transition_id"], g["condition"]) for g in gens] == [(r["id"], a) for r, a in jobs]
    assert jobs == sorted(jobs, key=lambda x: (x[0]["id"], x[1]))
    assert keys[0] == f"{jobs[0][0]['id']}|{jobs[0][1]}" and keys[8] == f"{jobs[8][0]['id']}|{jobs[8][1]}"
    assert {g["forced_reason"] for g in gens} <= {None, K.CAP, K.ENDED}


def test_thinking_start_checks():
    spec = R3.THINK_SPEC
    good = "x " + R.SYSTEM + " y" + spec.thinking_on_ending
    assert K.prompt_problems([good]) == []
    assert any("does not end" in p for p in K.prompt_problems([good + "z"]))
    assert any("system text" in p for p in K.prompt_problems(["x" + spec.thinking_on_ending]))
    assert any("contains" in p for p in K.prompt_problems([good], replace(spec, forbidden_in_prompt=("y",))))
    assert K.padding_problems([-1.0, -2.0], [-1.002, -2.001]) == []  # D-T1: float32 tolerances, 0.01 and 0.05
    assert K.padding_problems([-1.0, -2.0], [-1.5, -2.0])  # mean 0.25 above 0.01
    assert K.padding_problems([-1.0] * 20, [-1.0] * 19 + [-2.5])  # one token 1.5 apart
    assert K.padding_problems([], [])


def test_mock_end_to_end(mock_runs, small, monkeypatch, fast):
    """Round 2 scored and analysed (mock), every round-3 run scored (mock) with the environment check run from a
    plan, the thinking arm and both format checks, then the round-3 analysis: three families of 30, H3-P,
    replication against round 2's filed verdicts, and B3 on round 2's scores after the environment check."""
    man = _manifest(mock_runs, small)
    for tag in R.TAGS:
        assert S2.run_textcraft_r2_score({"run_id": f"textcraft_r2_score_{tag}", "model": tag, "manifest": man}) == 0
    assert A.run_textcraft_r2_analysis({"run_id": "textcraft_r2_analysis", "manifest": man, "models": list(R.TAGS)}) in (0, 1)
    for tag in list(R3.CORE) + list(R3.EXTENSION):
        assert S.run_textcraft_r3_score({"run_id": f"textcraft_r3_score_{tag}", "model": tag, "manifest": man}) == 0
    for tag in R3.ENVCHECK_MODELS:
        plan = {"models": {tag: {"runnable": True, "n_r2_shards": 1, "jobs": [{"job": 0, "gpu": "mock", "r2_shards": [0]}]}}}
        assert S.run_textcraft_r3_envcheck({"run_id": f"textcraft_r3_envcheck_{tag}", "model": tag, "manifest": man,
                                            "plan": plan}) == 0
    assert S.run_textcraft_r3_format({"run_id": "textcraft_r3_placement_gemma4-12b-it", "model": "gemma4-12b-it",
                                      "check": "placement", "manifest": man}) == 0
    assert S.run_textcraft_r3_format({"run_id": "textcraft_r3_nopreserve_qwen3.8-27b", "model": "qwen3.8-27b",
                                      "check": "nopreserve", "manifest": man}) == 0
    assert K.run_textcraft_r3_think({"run_id": "textcraft_r3_think_qwen3.5-9b", "manifest": man}) == 0
    monkeypatch.setattr(Z, "golden_tokens_check", lambda *a: {"ok": True})
    assert Z.run_textcraft_r3_analysis({"run_id": "textcraft_r3_analysis", "manifest": man}) == 0
    m = json.loads((mock_runs / "runs" / "textcraft_r3_analysis" / "metrics.json").read_text())
    assert m["A3"]["family_size"] == m["A3x"]["family_size"] == m["B3"]["family_size"] == 30
    assert m["A3"]["waiting"] == [] and m["A3x"]["waiting"] == []
    assert m["A3"]["H3_P"]["overall"] in ("supported", "rejected", "partial", "inconclusive")
    assert all(r["reading"] != "waiting (round 2's analysis not filed)" for r in m["A3"]["replication"].values())
    assert m["environment_check"]["status"] == "pass" and m["B3"]["old_scores"] == "round2"
    assert set(m["environment_check"]["models"]) == set(R3.ENVCHECK_MODELS)
    assert set(m["B3"]["differences"]) == {f"{o}->{n}" for o, n in R3.PAIRS}
    assert m["round2_analysis_read"]["metrics_sha256"] and m["gemma_slot"]["slot"] == "waiting"
    ex = m["exploratory"]
    assert set(ex["format_checks"]) == {"placement", "nopreserve"} and "thinking_arm" in ex
    assert {"qwen3.5-9b->qwen3.8-27b", "gemma4-12b-it->gemma4-31b-it"} <= set(ex)
    # S6: a round-2 score with another revision fails round 2's provenance check, and B3 is refused
    p = next((mock_runs / "runs" / "textcraft_r2_score_qwen3-8b").glob("shard_*/scoring.jsonl"))
    rows = p.read_text().splitlines()
    first = json.loads(rows[0])
    first["model_revision"] = "another"
    p.write_text("\n".join([json.dumps(first)] + rows[1:]) + "\n")
    assert Z.run_textcraft_r3_analysis({"run_id": "textcraft_r3_analysis", "manifest": man}) == 0
    m = json.loads((mock_runs / "runs" / "textcraft_r3_analysis" / "metrics.json").read_text())
    assert m["B3"]["differences"] == {}
    assert all(v == "refused (provenance)" for r in m["B3"]["readings"].values() for c, v in r.items() if c in A.CONTRASTS)


# ---------------- review 2 (design section 14.R2) ----------------


def test_freeze_commit_and_job_start_check(monkeypatch):
    clean = lambda: False  # noqa: E731
    assert G.freeze_commit(_git("c1")) == "c1" and G.freeze_commit(_git("")) is None
    with pytest.raises(SystemExit):
        G.freeze_commit(lambda a: "textcraft-r3-design-frozen-a\ntextcraft-r3-design-frozen-b" if a[0] == "tag" else "")
    monkeypatch.setenv("WF_MOCK_SCORE", "0")
    with pytest.raises(SystemExit):
        G.require_freeze_commit(_git(""), clean)
    with pytest.raises(SystemExit):  # HEAD differs in code
        G.require_freeze_commit(_git("c1"), clean)
    assert G.require_freeze_commit(_git("c1", diffs={"head": "notes/a.md"}), clean)["checked"] is True
    # R2-S7: a change to a job script is a code change, as under wf/ or scripts/
    for path in ("pace/textcraft_r3_score.sbatch", "scripts/mutation_check.py", "wf/textcraft/r3.py"):
        with pytest.raises(SystemExit):
            G.require_freeze_commit(_git("c1", diffs={"head": f"notes/a.md\n{path}"}), clean)
        assert not G.code_identical("head", "c1", _git("c1", diffs={"head": path}))
    # R2-S2: uncommitted changes to tracked files refuse the job
    with pytest.raises(SystemExit):
        G.require_freeze_commit(_git("c1", diffs={"head": "notes/a.md"}), lambda: True)
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    assert G.require_freeze_commit(_git(""))["checked"] == "mock"


def test_dirty_shards_are_refused(tmp_path):
    l40 = ["NVIDIA L40S"]
    d = _shard(tmp_path, "a/shard_00_of_01", [{"gpu": l40, "git_sha": "c1"}])
    (d / "inputs.json").write_text(json.dumps({"inputs": {}, "git_dirty": True}))
    assert G.shard_dirty(d)
    probs = Z.provenance_problems({"qwen3.5-9b": [d]}, {}, "c1", run_git=_git())
    assert any("uncommitted" in x for x in probs["qwen3.5-9b"])
    (d / "inputs.json").write_text(json.dumps({"inputs": {}, "git_dirty": False}))
    assert not G.shard_dirty(d) and Z.provenance_problems({"qwen3.5-9b": [d]}, {}, "c1", run_git=_git()) == {}


def test_b3_is_refused_through_a_newer_models_provenance():
    probs = {"ministral3-8b-it": ["ministral3-8b-it: jobs ran on different or unrecorded GPUs"], "olmo3-7b-it": ["x"]}
    out = Z.newer_model_refusals(probs, R3.pairs())
    assert out == ["a pair's newer model: ministral3-8b-it: jobs ran on different or unrecorded GPUs"]
    assert Z.newer_model_refusals(probs, [("gemma2-9b-it", "gemma4-e4b-it")]) == []
    assert Z.newer_model_refusals({"gemma4-e4b-it": ["y"]}, R3.pairs("gemma4-e4b-it"))


def test_h3p_labels_a_model_not_scored():
    v = {"a": V(parent_penalty="confirmed below zero"), "b": V(parent_penalty="confirmed below zero"),
         "c": {c: "not scored" for c in A.CONTRASTS}}
    out = Z.h3p_reading(v, {"a": True, "b": True, "c": False}, {t: False for t in v}, {"fires": False})
    assert out["per_model"]["c"] == "not scored" and out["overall"] == "supported"


def test_gpu_unobtainable_needs_the_first_submission_and_seven_days(tmp_path):
    import datetime as dt

    today = dt.date(2026, 12, 10)
    assert G.unobtainable_check(tmp_path, today) == {"filed": False, "valid": False}
    (tmp_path / G.UNOBTAINABLE).write_text(json.dumps({"reason": "no A40"}))
    assert not G.unobtainable_check(tmp_path, today)["valid"]
    (tmp_path / G.UNOBTAINABLE).write_text(json.dumps({"first_submission_job_id": "7250001",
                                                       "first_submission_date": "2026-12-05"}))
    out = G.unobtainable_check(tmp_path, today)
    assert not out["valid"] and out["days"] == 5
    (tmp_path / G.UNOBTAINABLE).write_text(json.dumps({"first_submission_job_id": "7250001",
                                                       "first_submission_date": "2026-12-03"}))
    assert G.unobtainable_check(tmp_path, today)["valid"]
    (tmp_path / G.UNOBTAINABLE).write_text(json.dumps({"first_submission_job_id": "",
                                                       "first_submission_date": "2026-12-01"}))
    assert not G.unobtainable_check(tmp_path, today)["valid"]


def test_a_filed_gate_failure_is_final(mock_runs, small):
    man = _manifest(mock_runs, small)
    _gate_rec(mock_runs / "runs", "textcraft_r3_score_qwen3.5-9b", {"ok": False, "problems": ["x"]})
    with pytest.raises(SystemExit):
        S.run_textcraft_r3_score({"run_id": "textcraft_r3_score_qwen3.5-9b", "model": "qwen3.5-9b", "manifest": man})
    rec = json.loads((mock_runs / "runs" / "textcraft_r3_score_qwen3.5-9b" / "shard_00_of_01" /
                      "tokenization_checks.json").read_text())
    assert rec["ok"] is False  # the record is kept
    _gate_rec(mock_runs / "runs", "textcraft_r3_score_olmo3-7b-it", {"ok": True})
    assert S.run_textcraft_r3_score({"run_id": "textcraft_r3_score_olmo3-7b-it", "model": "olmo3-7b-it", "manifest": man}) == 0


def test_not_run_declarations(tmp_path):
    def declare(tag, when):
        d = tmp_path / f"textcraft_r3_score_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "not_run.json").write_text(json.dumps({"date": when, "reason": "GPU availability"}))

    assert G.not_run_declaration(tmp_path, "qwen3.8-27b", None) == {"declared": False}
    declare("qwen3.8-27b", "2026-12-01T09:00:00")
    assert G.not_run_declaration(tmp_path, "qwen3.8-27b", None)["valid"]
    assert G.not_run_declaration(tmp_path, "qwen3.8-27b", "2026-12-02T10:00:00")["valid"]
    late = G.not_run_declaration(tmp_path, "qwen3.8-27b", "2026-11-30T10:00:00")
    assert not late["valid"] and "first core reading" in late["problems"][0]
    declare("qwen3.5-9b", "2026-12-01")
    assert not G.not_run_declaration(tmp_path, "qwen3.5-9b", None)["valid"]  # a core model cannot be declared
    declare("olmo3-7b-it", "yesterday")
    assert not G.not_run_declaration(tmp_path, "olmo3-7b-it", None)["valid"]


def test_a_declared_extension_model_lets_a3x_proceed(mock_runs, small, fast, monkeypatch):
    man = _manifest(mock_runs, small)
    for tag in list(R3.CORE) + ["qwen3.8-27b", "olmo3-7b-it"]:
        assert S.run_textcraft_r3_score({"run_id": f"textcraft_r3_score_{tag}", "model": tag, "manifest": man}) == 0
    d = mock_runs / "runs" / "textcraft_r3_score_gemma4-31b-it"
    d.mkdir(parents=True)
    (d / "not_run.json").write_text(json.dumps({"date": "2026-12-01T09:00:00", "reason": "no A100 80 GB"}))
    assert Z.run_textcraft_r3_analysis({"run_id": "textcraft_r3_analysis", "manifest": man}) == 0
    m = json.loads((mock_runs / "runs" / "textcraft_r3_analysis" / "metrics.json").read_text())
    assert m["A3x"]["not_scored"] == ["gemma4-31b-it"] and "H3_P" in m["A3x"]
    assert (mock_runs / "runs" / G.FIRST_CORE).exists()
    # a declaration dated after the first core reading is refused, and refuses A3x
    first = json.loads((mock_runs / "runs" / G.FIRST_CORE).read_text())["time"]
    (d / "not_run.json").write_text(json.dumps({"date": "2999-01-01T00:00:00", "reason": "late"}))
    assert Z.run_textcraft_r3_analysis({"run_id": "textcraft_r3_analysis", "manifest": man}) == 0
    m = json.loads((mock_runs / "runs" / "textcraft_r3_analysis" / "metrics.json").read_text())
    assert "refused" in m["A3x"] and json.loads((mock_runs / "runs" / G.FIRST_CORE).read_text())["time"] == first


def test_the_environment_rule_is_round_threes_own_and_unamended(monkeypatch):
    assert R3.ENVCHECK_RULE == {"yes_no_mean_abs_max": 0.07, "yes_no_goal_effect_diff_max": 0.10,
                                "yes_no_share_diff_max": 0.02, "two_candidate_mean_abs_max": 0.35,
                                "two_candidate_goal_effect_diff_max": 0.10}
    assert Z.per_score_check("yes_no", [0.05, -0.05])["ok"] and not Z.per_score_check("yes_no", [0.08, -0.08])["ok"]
    assert Z.per_score_check("two_candidate", [0.3])["ok"] and not Z.per_score_check("two_candidate", [0.4])["ok"]
    # a later change in wave B does not change round 3
    monkeypatch.setitem(FAM.ENVCHECK_RULE, "yes_no_mean_abs_max", 5.0)
    monkeypatch.setattr(FAM, "envcheck_per_score_check", lambda *a, **k: True, raising=False)
    assert not Z.per_score_check("yes_no", [0.08, -0.08])["ok"] and Z.env_rule() == R3.ENVCHECK_RULE


def test_the_environment_rule_is_pinned(tmp_path, monkeypatch):
    pin = tmp_path / "pin.json"
    assert G.rule_problems(pin)  # no pin: refused
    pin.write_text(json.dumps(G.rule_fingerprint()))
    assert G.rule_problems(pin) == []
    assert G.rule_problems() == []  # the committed pin is round 3's constant
    pinned = json.loads(G.RULE_PIN.read_text())
    assert pinned["rule"] == R3.ENVCHECK_RULE and pinned["sha256"] == G.rule_fingerprint()["sha256"]
    monkeypatch.setitem(R3.ENVCHECK_RULE, "yes_no_mean_abs_max", 0.25)
    assert any("rule" in x for x in G.rule_problems(pin)) and G.rule_problems()


def test_the_analysis_refuses_a_changed_environment_rule(mock_runs, small, monkeypatch):
    man = _manifest(mock_runs, small)
    monkeypatch.setitem(R3.ENVCHECK_RULE, "two_candidate_mean_abs_max", 1.0)
    assert Z.run_textcraft_r3_analysis({"run_id": "textcraft_r3_analysis", "manifest": man}) == 1
    assert "environment-check rule" in (mock_runs / "runs" / "textcraft_r3_analysis" / "STATUS.md").read_text()


# ---------------- deviation D-T1 (2026-11-23, notes/textcraft_r3_deviation_thinking_padding.md) ----------------


def _git_dev(head="dev", dev_diff="wf/textcraft/r3_think.py\nwf/textcraft/r3_gates.py\nnotes/x.md", n_dev=1):
    """A checkout with the freeze tag at c1 and D-T1's deviation tag(s) at dev; HEAD at `head`."""
    def run(args):
        if args[:2] == ["tag", "-l"]:
            if args[2].startswith(G.DEVIATION_TAG_PREFIX):
                return "\n".join(f"{G.DEVIATION_TAG_PREFIX}2026-11-2{k}" for k in range(3, 3 + n_dev))
            return "textcraft-r3-design-frozen-2026-11-23"
        if args[0] == "rev-parse":
            return "c1" if "frozen" in args[1] else "dev" if "deviation" in args[1] else head
        if args[0] == "diff":
            x, y = args[2], args[3]
            if x == y:
                return ""
            return dev_diff if {x, y} == {"c1", "dev"} else "wf/x.py"
        raise AssertionError(args)
    return run


def test_dt1_padding_rule_and_the_float32_start_check(small, monkeypatch):
    assert K.PAD_RULE == {"top_k": 20, "mean_abs_max": 0.01, "any_abs_max": 0.05, "dtype": "float32"}
    assert K.padding_problems([-1.0] * 20, [-1.009] * 20) == []
    assert K.padding_problems([-1.0] * 20, [-1.011] * 20)
    assert K.padding_problems([-1.0] * 20, [-1.0] * 19 + [-1.04]) == []
    assert K.padding_problems([-1.0] * 20, [-1.0] * 19 + [-1.06])  # mean 0.003, one token 0.06
    events = []

    class Fake(K._MockEngine):
        def __init__(self, dtype="bfloat16"):
            events.append(("open", dtype))

        def prompt_text(self, msgs):
            return repr(msgs) + R.SYSTEM + R3.THINK_SPEC.thinking_on_ending

        def padded_check(self, texts):
            events.append(("padded", len(texts), texts == sorted(texts, key=len)))
            return {"alone": [-1.0], "padded": [-1.0], "problems": []}

        def close(self):
            events.append(("close",))

    jobs = K.think_jobs(small)[:12]
    shots = P.fewshot_examples(P.FEWSHOT)
    rec = K.start_checks(jobs, R3.THINK_SPEC, shots, engine=Fake)
    assert events == [("open", "float32"), ("padded", K.BATCH, True), ("close",)]
    assert rec["padding"]["dtype"] == "float32" and rec["padding"]["rule"] == K.PAD_RULE and rec["n_prompt_problems"] == 0

    class Boom(Fake):
        def padded_check(self, texts):
            raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        K.start_checks(jobs, R3.THINK_SPEC, shots, engine=Boom)
    assert events[-1] == ("close",)  # closed even when the check fails
    # the two engines: bf16 exactly as before (families.open_scorer); the check's copy from the same snapshot in float32
    calls = {}
    stub = types.SimpleNamespace(_tok="tok", _model="model", revision="rev", close=lambda: calls.setdefault("closed", 1))
    monkeypatch.setattr(FAM, "open_scorer", lambda spec, tag: calls.setdefault("bf16", (spec, tag)) and stub)
    from wf.scoring import logodds as LO

    monkeypatch.setattr(LO, "LogoddsScorer", lambda name, **kw: calls.setdefault("f32", (name, kw)) and stub)
    e = K._Engine()
    assert calls["bf16"] == (R3.THINK_SPEC, R3.THINK_TAG) and "f32" not in calls and e.dtype == "bfloat16"
    e = K._Engine(dtype="float32")
    name, kw = calls["f32"]
    spec = R3.THINK_SPEC
    assert name == spec.hf_id and kw["dtype"] == "float32" and kw["model_tag"] == R3.THINK_TAG
    assert (kw["template_kwargs"], kw["model_class"], kw["tokenizer_kwargs"], kw["strict_load"], kw["attn_implementation"],
            kw["prompt_format"], kw["unexpected_keys_allowed"]) \
        == (spec.template_kwargs, spec.model_class, spec.tokenizer_kwargs, spec.strict_load, spec.attn_implementation,
            spec.prompt_format, spec.unexpected_keys_allowed)
    e.close()
    assert calls["closed"]


def test_dt1_the_run_checks_on_float32_before_loading_bf16(mock_runs, small, monkeypatch):
    man = _manifest(mock_runs, small)
    events = []

    class Fake(K._MockEngine):
        problems: list = []

        def __init__(self, dtype="bfloat16"):
            events.append(dtype)

        def prompt_text(self, msgs):
            return repr(msgs) + R.SYSTEM + R3.THINK_SPEC.thinking_on_ending

        def padded_check(self, texts):
            events.append("padded")
            return {"alone": [-1.0], "padded": [-1.0], "problems": list(Fake.problems)}

        def close(self):
            events.append("close")

    monkeypatch.setenv("WF_MOCK_SCORE", "0")
    monkeypatch.setattr(K, "_Engine", Fake)
    monkeypatch.setattr(FAM, "require_environment", lambda spec, **kw: {})
    monkeypatch.setattr(FAM, "require_resources", lambda spec: {})
    monkeypatch.setattr(S, "gpu_check", lambda tag, **kw: {"ok": True})
    monkeypatch.setattr(G, "require_freeze_commit", lambda **kw: {"checked": "test"})
    cfg = {"run_id": "textcraft_r3_think_smoke_qwen3.5-9b", "manifest": man, "max_jobs": 8}
    Fake.problems = ["padded and alone differ"]
    assert K.run_textcraft_r3_think(cfg) == 1
    assert events == ["float32", "padded", "close"]  # stopped before the bf16 model loaded
    d = mock_runs / "runs" / "textcraft_r3_think_smoke_qwen3.5-9b" / "shard_00_of_01"
    assert json.loads((d / "start_checks.json").read_text())["padding"]["dtype"] == "float32"
    events.clear()
    Fake.problems = []
    assert K.run_textcraft_r3_think({**cfg, "run_id": "textcraft_r3_think_qwen3.5-9b"}) == 0
    assert events == ["float32", "padded", "close", "bfloat16"]


def test_dt1_only_thinking_runs_start_at_the_deviation_commit(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "0")
    clean = lambda: False  # noqa: E731
    think = "textcraft_r3_think_smoke_qwen3.5-9b"
    out = G.require_freeze_commit(_git_dev(), clean, run_id=think)
    assert (out["deviation"], out["deviation_commit"], out["tag_commit"], out["head"]) == ("D-T1", "dev", "c1", "dev")
    assert G.require_freeze_commit(_git_dev(), clean, run_id="textcraft_r3_think_qwen3.5-9b")["deviation"] == "D-T1"
    for rid in ("", "textcraft_r3_score_qwen3.5-9b", "textcraft_r3_smoke_qwen3.5-9b", "textcraft_r3_envcheck_qwen3-8b",
                "textcraft_r3_rescore_qwen3-8b", "textcraft_r3_placement_gemma4-12b-it", "textcraft_r3_analysis"):
        with pytest.raises(SystemExit, match="not the freeze tag"):
            G.require_freeze_commit(_git_dev(), clean, run_id=rid)
    assert "deviation" not in G.require_freeze_commit(_git_dev(head="c1"), clean, run_id=think)
    # refused for the thinking arm too: a deviation commit changing other code, two deviation tags, HEAD elsewhere, dirty
    for g in (_git_dev(dev_diff="wf/textcraft/r3_think.py\nwf/textcraft/r3_score.py"), _git_dev(n_dev=2),
              _git_dev(head="other"), _git_dev(dev_diff="pace/textcraft_r3_score.sbatch")):
        with pytest.raises(SystemExit):
            G.require_freeze_commit(g, clean, run_id=think)
    with pytest.raises(SystemExit):
        G.require_freeze_commit(_git_dev(), lambda: True, run_id=think)
    assert G.deviation_commit(_git_dev())["commit"] == "dev"
    assert G.deviation_commit(_git_dev(dev_diff="notes/a.md\ntests/t.py\nscripts/mutation_check.py"))["commit"] == "dev"
    assert G.deviation_commit(_git_dev(dev_diff="wf/textcraft/r3.py"))["commit"] is None
    assert G.deviation_commit(_git_dev(n_dev=0)) == {"commit": None, "problems": []}


def test_dt1_a_core_run_at_the_deviation_commit_is_refused(mock_runs, small, monkeypatch):
    """At job start: the scoring, smoke and environment-check runs refuse a checkout at the deviation commit; the
    thinking arm starts there and records it."""
    man = _manifest(mock_runs, small)
    real = G.require_freeze_commit

    def at_deviation(run_git=None, dirty=None, run_id=""):
        with monkeypatch.context() as m:
            m.setattr(G, "mock", lambda: False)
            return real(_git_dev(), lambda: False, run_id=run_id)

    monkeypatch.setattr(G, "require_freeze_commit", at_deviation)
    for run in ("score", "smoke"):
        with pytest.raises(SystemExit, match="not the freeze tag"):
            S.run_textcraft_r3_score({"run_id": f"textcraft_r3_{run}_qwen3.5-9b", "model": "qwen3.5-9b",
                                      "manifest": man, "smoke": run == "smoke"})
    plan = {"models": {"qwen3-8b": {"runnable": True, "n_r2_shards": 1,
                                    "jobs": [{"job": 0, "gpu": "mock", "r2_shards": [0]}]}}}
    with pytest.raises(SystemExit, match="not the freeze tag"):
        S.run_textcraft_r3_envcheck({"run_id": "textcraft_r3_envcheck_qwen3-8b", "model": "qwen3-8b", "manifest": man,
                                     "plan": plan})
    assert K.run_textcraft_r3_think({"run_id": "textcraft_r3_think_smoke_qwen3.5-9b", "manifest": man, "max_jobs": 8}) == 0
    d = mock_runs / "runs" / "textcraft_r3_think_smoke_qwen3.5-9b" / "shard_00_of_01"
    env = json.loads((d / "config.json").read_text())["environment"]
    assert env["freeze_commit"]["deviation_commit"] == "dev"


def test_dt1_the_analysis_accepts_thinking_shards_at_the_deviation_commit_only(tmp_path):
    l40 = ["NVIDIA L40S"]
    g = _git_dev()
    t_dev = _shard(tmp_path, "think/shard_00_of_02", [{"gpu": l40, "git_sha": "dev"}])
    t_frz = _shard(tmp_path, "think/shard_01_of_02", [{"gpu": l40, "git_sha": "c1"}])
    e_dev = _shard(tmp_path, "env/shard_00_of_01", [{"gpu": l40, "git_sha": "dev"}])
    r_dev = _shard(tmp_path, "rescore/shard_00_of_01", [{"gpu": l40, "git_sha": "dev"}])
    dev = G.deviation_commit(g)
    cp = Z.other_commit_problems({"think": [t_dev, t_frz], "envcheck:qwen3-8b": [e_dev], "rescore:qwen3-8b": [r_dev]},
                                 "c1", dev, run_git=g)
    assert "think" not in cp and set(cp) == {"envcheck:qwen3-8b", "rescore:qwen3-8b"}
    # with no valid deviation tag the thinking shard at the deviation commit is refused as well
    bad = {"commit": None, "problems": ["deviation tag changes code outside D-T1's files"]}
    cp = Z.other_commit_problems({"think": [t_dev]}, "c1", bad, run_git=g)
    assert len(cp["think"]) == 2
    assert Z.other_commit_problems({"think": [t_frz]}, "c1", {"commit": None, "problems": []}, run_git=g) == {}
    # a core shard at the deviation commit is refused by the families' provenance check
    core = _shard(tmp_path, "core/shard_00_of_02", [{"gpu": l40, "git_sha": "dev"}])
    rev = R3.spec_for("qwen3.5-9b").revision
    probs = Z.provenance_problems({"qwen3.5-9b": [core]}, {"qwen3.5-9b": [{"model_revision": rev}]}, "c1", run_git=g,
                                  dirty_of=lambda p: False)
    assert any("not the freeze tag" in x for x in probs["qwen3.5-9b"])
    assert Z.think_shard_commits([t_dev, t_frz]) == {"think/shard_00_of_02": ["dev"], "think/shard_01_of_02": ["c1"]}

