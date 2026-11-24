"""Model families, wave B (wf/families.py, wf/scoring/logodds.py, wf/scoring/prompts.py, wf/tools/family_gate_cpu.py,
wf/tools/envcheck_tokens.py, wf/tools/prefetch_models.py; notes/families_waveb_design.md)."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import types
from pathlib import Path

import pytest

from wf import constants as C, families, sizes
from wf.scoring import prompts
from wf.scoring.logodds import LogoddsScorer, check_loading
from tests.test_rq1 import mock_root, mock_score  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())
RENDERED_B = ROOT / "tests" / "golden" / "families_waveb_rendered.json"
B = families.WAVE_B


# ---------------- registry and fixed design ----------------


def test_every_wave_b_model_is_pinned_registered_configured_and_loaded_in_serve2():
    from wf.runs import RUNNERS, load_config

    assert list(B) == ["ministral3-8b-it", "qwen3.5-9b", "qwen3.5-9b-plain", "qwen3.5-9b-base-plain", "gemma4-e4b-it",
                       "qwen3.8-27b", "gemma4-31b-it"]
    for tag, spec in B.items():
        assert spec.tag == tag and tag not in families.MODELS
        assert re.fullmatch(r"[0-9a-f]{40}", C.REVISIONS[spec.hf_id]) and spec.revision == C.REVISIONS[spec.hf_id]
        assert f"family_score_{tag}" in RUNNERS
        cfg = load_config(f"family_score_{tag}")
        assert cfg["model"] == tag and cfg["n_shards"] == 1
        assert spec.wave == "B" and spec.env == "serve2" and spec.strict_load
        assert spec.model_class == "image_text_to_text" and spec.tokenizer_kwargs == {"mistral_format": False}
        assert spec.system == "native" and spec.attn_implementation is None
        assert re.fullmatch(r"\d{4}-\d{2}", spec.release) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec.hub_created)
    assert B["ministral3-8b-it"].hf_id.endswith("-BF16")  # the default repository's language model is FP8
    assert B["qwen3.5-9b-plain"].hf_id == B["qwen3.5-9b"].hf_id and B["qwen3.5-9b-base-plain"].hf_id == "Qwen/Qwen3.5-9B-Base"
    assert {t for t, s in B.items() if s.prompt_format == "plain"} == {"qwen3.5-9b-plain", "qwen3.5-9b-base-plain"}
    assert {t for t, s in B.items() if s.gpu_min_gb >= 80} == {"qwen3.8-27b", "gemma4-31b-it"}


def test_wave_a_specs_record_exactly_as_before():
    """Wave A's saved run configs hold model_spec; adding wave B's fields must not change what a wave A spec records."""
    keys = {"tag", "hf_id", "family", "generation", "release", "params_b", "system", "attn_implementation",
            "template_kwargs", "template_notes", "revision"}
    for spec in list(families.MODELS.values()) + list(families.REFERENCE.values()):
        assert set(spec.record()) == keys, spec.tag
    assert {"wave", "env", "strict_load", "model_class", "tokenizer_kwargs"} <= set(B["gemma4-e4b-it"].record())
    assert B["gemma4-31b-it"].record()["generation_suffix"] == "<|channel>thought\n<channel|>"


def test_primary_family_and_planned_comparisons_are_the_designs():
    assert families.WAVE_B_SHARE_TESTS == ("ministral3-8b-it", "qwen3.5-9b", "gemma4-e4b-it", "qwen3.8-27b",
                                           "gemma4-31b-it", "qwen3.5-9b-base-plain")
    assert families.WAVE_B_PLANNED == (("gemma4-e4b-it", "gemma2-9b-it"), ("ministral3-8b-it", "mistral-7b-v0.3"),
                                       ("qwen3.5-9b", "qwen3-8b"), ("qwen3.8-27b", "qwen3-32b"),
                                       ("qwen3.5-9b-plain", "qwen3.5-9b-base-plain"))
    assert families.WAVE_B_FAMILY_SIZE == 16


def test_wave_b_analysis_config_reads_every_model_and_every_planned_partner():
    from wf.runs import RUNNERS, load_config

    cfg = load_config("families_waveb_analysis")
    assert "families_waveb_analysis" in RUNNERS and cfg["wave"] == "B" and cfg["envcheck"] == "families_envcheck"
    assert cfg["models"] == list(B)
    have = set(cfg["models"]) | set(cfg["reference"])
    assert all(a in have and b in have for a, b in families.WAVE_B_PLANNED)
    # section 8 amendment (2026-11-23): every reference is its serve2 rescoring; the originals are descriptive only
    assert set(cfg["reference"]) == set(families.RESCORE_MODELS) == set(cfg["rescore"]) == set(cfg["reference_original"])
    for tag, ref in cfg["reference"].items():
        spec = families.spec_for(tag)
        assert spec.hf_id in C.REVISIONS and ref["model"] == tag + families.RESCORE_SUFFIX
        assert ref["runs"] == {f"family_rescore_{tag}": list(families.ENVCHECK_ARMS)}
        assert cfg["rescore"][tag]["run"] == f"family_rescore_{tag}" and f"family_rescore_{tag}" in RUNNERS
        assert cfg["rescore"][tag]["tokens"] == f"rescore_tokens/compare_{tag}.json"
        assert cfg["reference_original"][tag]["model"] in (tag, "llama3.1-8b-fewshot")
    assert cfg["placement"] == {"gemma4-e4b-it": "family_placement_gemma4-e4b-it"}
    assert "family_placement_gemma4-e4b-it" in RUNNERS


def test_thinking_and_template_settings_per_model():
    q_off, q_on = "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n<think>\n"
    for tag in ("qwen3.5-9b", "qwen3.8-27b"):
        s = B[tag]
        assert s.thinking_switch and s.thinking_off_ending == q_off and s.thinking_on_ending == q_on
        assert s.template_kwargs == {} and s.generation_suffix == ""
    assert B["qwen3.8-27b"].thinking_on_markers == ("Reasoning effort",)
    for tag in ("gemma4-e4b-it", "gemma4-31b-it"):
        assert B[tag].thinking_switch and B[tag].thinking_on_markers == ("<|think|>",)
    assert B["gemma4-e4b-it"].thinking_off_ending == "<|turn>model\n" and B["gemma4-e4b-it"].generation_suffix == ""
    assert B["gemma4-31b-it"].thinking_off_ending == "<|turn>model\n<|channel>thought\n<channel|>"
    m = B["ministral3-8b-it"]
    assert not m.thinking_switch and m.thinking_off_ending == "[/INST]" and m.mistral_common_check
    assert m.forbidden_in_prompt == ("Le Chat", "The current date")
    for tag in ("qwen3.5-9b-plain", "qwen3.5-9b-base-plain"):
        assert not B[tag].thinking_switch and B[tag].thinking_off_ending == ""
    # every scorer call passes thinking off, whatever the spec
    assert LogoddsScorer.for_tokenizer(B["qwen3.5-9b"].hf_id, object()).template_settings()["enable_thinking"] is False


def test_unexpected_tensor_patterns_allow_exactly_the_unused_checkpoint_tensors():
    qwen = re.compile(B["qwen3.5-9b"].unexpected_keys_allowed)
    assert qwen.search("mtp.fc.weight") and qwen.search("mtp.layers.0.mlp.up_proj.weight")
    assert not qwen.search("model.language_model.layers.0.mlp.up_proj.weight")
    gemma = re.compile(B["gemma4-e4b-it"].unexpected_keys_allowed)
    for layer in range(24, 42):
        for name in ("k_proj", "v_proj", "k_norm"):
            assert gemma.search(f"model.language_model.layers.{layer}.self_attn.{name}.weight")
    assert not gemma.search("model.language_model.layers.23.self_attn.k_proj.weight")
    assert not gemma.search("model.language_model.layers.42.self_attn.k_proj.weight")
    assert not gemma.search("model.language_model.layers.30.self_attn.q_proj.weight")
    for tag in ("ministral3-8b-it", "gemma4-31b-it"):
        assert B[tag].unexpected_keys_allowed == ""
    for tag in ("qwen3.5-9b", "qwen3.5-9b-plain", "qwen3.5-9b-base-plain", "qwen3.8-27b"):
        assert B[tag].unexpected_keys_allowed == r"^mtp\."  # the 15 multi-token-prediction tensors, nothing else


# ---------------- plain format ----------------


def test_plain_text_prompt_joins_examples_with_one_space_and_blank_lines():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "H1\nNext observation:"},
            {"role": "assistant", "content": "O1"}, {"role": "user", "content": "H2\nNext observation:"}]
    assert prompts.plain_text_prompt(msgs) == "SYS\n\nH1\nNext observation: O1\n\nH2\nNext observation:"
    assert prompts.plain_text_prompt(msgs[1:]) == "H1\nNext observation: O1\n\nH2\nNext observation:"
    for bad in (msgs[:3], [msgs[0], msgs[2], msgs[1]], [msgs[0]], msgs[:2] + [msgs[1]]):
        with pytest.raises(ValueError):
            prompts.plain_text_prompt(bad)


def test_plain_text_prompt_of_the_real_messages(mock_score):
    item = GOLDEN["item"]
    msgs = sizes.arm_messages(item, "yn_aligned")
    text = prompts.plain_text_prompt(msgs)
    assert text.startswith(msgs[0]["content"] + "\n\n") and text.endswith("\nAnswer:")
    assert text.count("Next observation: ") == 3 and "<|" not in text and "[INST]" not in text


class WordTok:
    """Word-level stand-in tokenizer for the plain format: words (and a leading space folded into the word) are ids;
    `merge` = (left, right): the two words written with one space between them become one token."""

    def __init__(self, merge: tuple[str, str] | None = None, split_yes: bool = False):
        self.vocab: dict[str, int] = {}
        self.merge, self.split_yes = merge, split_yes
        self.all_special_ids, self.added_tokens_decoder = [], {}

    def _id(self, w: str) -> int:
        return self.vocab.setdefault(w, len(self.vocab) + 10)

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors=None):
        words = text.split()
        if self.merge:
            out, i = [], 0
            while i < len(words):
                if i + 1 < len(words) and (words[i], words[i + 1]) == self.merge:
                    out.append(words[i] + "_" + words[i + 1])
                    i += 2
                else:
                    out.append(words[i])
                    i += 1
            words = out
        if self.split_yes:
            words = [p for w in words for p in (["Ye", "s"] if w == "Yes" else [w])]
        return types.SimpleNamespace(input_ids=[[self._id(w) for w in words]] if return_tensors else [self._id(w) for w in words])

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def _plain_scorer(tok, tag="qwen3.5-9b-base-plain"):
    spec = B[tag]
    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=tag, prompt_format=spec.prompt_format), spec


def test_plain_scorer_scores_each_candidate_after_one_space():
    sc, _ = _plain_scorer(WordTok())
    assert sc.candidate_text("Yes") == " Yes" and LogoddsScorer.for_tokenizer(B["qwen3.5-9b"].hf_id, WordTok()).candidate_text("Yes") == "Yes"
    with pytest.raises(ValueError):
        sc._encode([{"role": "user", "content": "x"}], add_generation_prompt=False)


def _items(n_primary, n_successes, **changes):
    base = GOLDEN["item"]
    out = []
    for i in range(n_primary + n_successes):
        t = {**base, "id": f"t{i}", "action": f"take soapbar {i + 1} from cabinet 1", **changes.get(f"t{i}", {})}
        out.append(t)
    return out[:n_primary], out[n_primary:]


def test_plain_gate_passes_a_tokenizer_that_never_merges_across_the_boundary(mock_score):
    sc, spec = _plain_scorer(WordTok())
    primary, successes = _items(3, 2)
    rec = families.tokenization_checks(sc, spec, primary, successes)
    assert rec["ok"], rec["problems"]
    assert rec["prompt_format"] == "plain" and rec["n_candidates_checked"] == 10 and rec["candidate_prefix"] == " "
    assert rec["n_yes_no_prompts_checked"] == 3 * 3 + 2 * 2
    assert {w: v["n_checked"] for w, v in rec["yes_no"].items()} == {"Yes": 13, "No": 13}


@pytest.mark.parametrize("tok, words", [
    (WordTok(merge=("observation:", "You")), "merge across the boundary"),
    (WordTok(merge=("Answer:", "No")), "' No' merge across the boundary"),
    (WordTok(split_yes=True), "' Yes' is not one token"),
])
def test_plain_gate_fails_on_a_merge_or_a_split_answer(mock_score, tok, words):
    sc, spec = _plain_scorer(tok)
    rec = families.tokenization_checks(sc, spec, *_items(2, 1))
    assert not rec["ok"] and any(words in p for p in rec["problems"]), rec["problems"]


# ---------------- chat gate additions ----------------


class ChatTok:
    """A word-level chat template stand-in with wave B's features: a thinking switch (`think_on` written at the end
    of the generation prompt when on, `off` when off), an optional suffix after the generation prompt that a
    rendered reply lacks, optional forbidden text, and options to break each property."""

    def __init__(self, reads_switch=True, marker_when_off=False, suffix="", off="OFF", system_text="",
                 off_on_yes_no=True, on_text="THINK", examples_off=False):
        self.vocab: dict[str, int] = {"<s>": 0, "<e>": 1, "<u>": 2, "<a>": 3}
        self.all_special_ids, self.added_tokens_decoder = [0, 1, 2, 3], {}
        self.reads_switch, self.marker_when_off, self.suffix, self.off = reads_switch, marker_when_off, suffix, off
        self.system_text, self.off_on_yes_no, self.on_text = system_text, off_on_yes_no, on_text
        self.examples_off = examples_off  # every earlier reply carries the thinking-off text too (Qwen3.8's default)

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return types.SimpleNamespace(input_ids=[self._id(w) for w in re.findall(r"<[seua]>|[^\s<]+", text)])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=None, **kw):
        on = bool(enable_thinking) if self.reads_switch else False
        turns = [m for m in messages if m["role"] != "system"]
        yes_no = any("Answer:" in m["content"] for m in turns if m["role"] == "user")
        off = self.off if (self.off_on_yes_no or not yes_no) else ""
        # a final assistant reply is written after the thinking-off text, as Qwen's template writes its empty block
        body = " ".join(f"<a> {off + ' ' if (k == len(turns) - 1 or self.examples_off) else ''}{m['content']} <e>" if m["role"] == "assistant"
                        else f"<u> {m['content']} <e>" for k, m in enumerate(turns))
        text = "<s> " + ("MARKER " if self.marker_when_off else "") + self.system_text + body
        if add_generation_prompt:
            text += " <a> " + (self.on_text if on else off) + (" " + self.suffix if self.suffix and not on else "")
        return self(text).input_ids if tokenize else text

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def _chat_spec(**kw):
    base = dict(thinking_switch=True, thinking_on_markers=("THINK",), thinking_on_ending="THINK",
                thinking_off_ending="<a> OFF")
    return families.replace(B["gemma4-e4b-it"], **{**base, **kw})


def _chat_scorer(tok, spec):
    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs)


def test_wave_b_gate_passes_a_template_that_honours_thinking_off(mock_score):
    spec = _chat_spec()
    primary, successes = _items(2, 1)
    rec = families.tokenization_checks(_chat_scorer(ChatTok(), spec), spec, primary, successes)
    assert rec["ok"], rec["problems"]
    w = rec["waveb_prompt_checks"]
    assert w["n_first_item_renders"] == 6 and w["n_prompts_checked_ending"] == len(sizes.size_jobs(primary, successes))


@pytest.mark.parametrize("tok, words", [
    (ChatTok(reads_switch=False), "switch not read"),
    (ChatTok(marker_when_off=True, off="OFF"), None),  # MARKER is not the spec's marker: passes unless marker matches
    (ChatTok(off_on_yes_no=False), "yn_aligned: the prompt does not end with"),
    (ChatTok(off="DIFFERENT"), "does not end with"),
    (ChatTok(on_text="THINK EXTRA"), "thinking on does not end with 'THINK'"),  # the switch is read, but not as pinned
])
def test_wave_b_gate_fails_when_thinking_is_not_pinned_off(mock_score, tok, words):
    spec = _chat_spec()
    rec = families.tokenization_checks(_chat_scorer(tok, spec), spec, *_items(2, 1))
    if words is None:
        assert rec["ok"], rec["problems"]
        spec2 = _chat_spec(thinking_on_markers=("THINK", "MARKER"))
        rec = families.tokenization_checks(_chat_scorer(tok, spec2), spec2, *_items(2, 1))
        words = "thinking off contains 'MARKER'"
    assert not rec["ok"] and any(words in p for p in rec["problems"]), rec["problems"]


def test_wave_b_gate_finds_forbidden_text(mock_score):
    spec = _chat_spec(forbidden_in_prompt=("Le Chat",))
    rec = families.tokenization_checks(_chat_scorer(ChatTok(system_text="Le Chat "), spec), spec, *_items(1, 0))
    assert not rec["ok"] and any("contains 'Le Chat'" in p for p in rec["problems"])


def test_generation_suffix_is_stripped_before_the_reply_check(mock_score):
    """A generation prompt that ends in tokens a rendered reply lacks (Gemma 4 31B's empty thought channel) fails the
    reply check unless that suffix is declared; a wrong declared suffix fails too."""
    tok = ChatTok(suffix="CHANNEL")
    primary, successes = _items(2, 1)
    without = _chat_spec(thinking_off_ending="<a> OFF CHANNEL")
    rec = families.tokenization_checks(_chat_scorer(tok, without), without, primary, successes)
    assert not rec["ok"] and any("does not begin with the prompt" in p for p in rec["problems"])
    declared = families.replace(without, generation_suffix="CHANNEL")
    rec = families.tokenization_checks(_chat_scorer(tok, declared), declared, primary, successes)
    assert rec["ok"], rec["problems"]
    wrong = families.replace(without, generation_suffix="OTHER")
    rec = families.tokenization_checks(_chat_scorer(tok, wrong), wrong, primary, successes)
    assert not rec["ok"] and any("declared suffix" in p for p in rec["problems"])


def test_mistral_common_agreement_on_every_scored_prompt(mock_score):
    spec = _chat_spec(mistral_common_check=True)
    tok = ChatTok()
    sc = _chat_scorer(tok, spec)
    primary, successes = _items(2, 1)
    same = lambda msgs: families._ids(sc._encode(msgs))
    rec = families.tokenization_checks(sc, spec, primary, successes, same)
    assert rec["ok"] and rec["waveb_prompt_checks"]["n_prompts_checked_mistral_common"] == len(sizes.size_jobs(primary, successes))
    odd = lambda msgs: same(msgs) + ([99] if "t1" in json.dumps(msgs) or "soapbar 2" in json.dumps(msgs) else [])
    rec = families.tokenization_checks(sc, spec, primary, successes, odd)
    assert not rec["ok"] and any("differ from mistral-common" in p for p in rec["problems"])
    rec = families.tokenization_checks(sc, spec, primary, successes, None)
    assert not rec["ok"] and any("no reference encoder" in p for p in rec["problems"])


def test_wave_a_gate_does_not_run_the_wave_b_checks(mock_score):
    from tests.test_families import FakeTok, _scorer

    spec = families.MODELS["mistral-7b-v0.3"]
    rec = families.tokenization_checks(_scorer(FakeTok(), spec), spec, [GOLDEN["item"]], [])
    assert rec["ok"] and "waveb_prompt_checks" not in rec


# ---------------- scorer loading ----------------


def _fake_hf(monkeypatch, version="5.23.0", info=None, device_map=None):
    calls = {}

    class Model:
        def __init__(self, attn="sdpa"):
            self.config = types.SimpleNamespace(_attn_implementation=attn)
            self.hf_device_map = device_map

        def eval(self):
            return self

    def auto(name):
        class Auto:
            @staticmethod
            def from_pretrained(model, **kw):
                calls["class"], calls["model"] = name, kw
                m = Model(kw.get("attn_implementation", "sdpa"))
                return (m, info or {}) if kw.get("output_loading_info") else m
        return Auto

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["tokenizer"] = kw
            return types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        __version__=version, AutoModelForCausalLM=auto("AutoModelForCausalLM"),
        AutoModelForImageTextToText=auto("AutoModelForImageTextToText"), AutoTokenizer=AutoTokenizer))
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    return calls


def test_wave_b_scorer_loads_the_vision_language_class_in_bf16_with_the_loading_check(monkeypatch):
    calls = _fake_hf(monkeypatch, info={"missing_keys": [], "unexpected_keys": ["mtp.fc.weight"]},
                     device_map={"": 0})
    s = B["qwen3.5-9b"]
    sc = families.open_scorer(s)
    assert calls["class"] == "AutoModelForImageTextToText" and calls["model"]["dtype"] == "bf16"
    assert "torch_dtype" not in calls["model"] and calls["model"]["output_loading_info"] is True
    assert calls["model"]["revision"] == C.REVISIONS[s.hf_id] and calls["tokenizer"]["mistral_format"] is False
    assert calls["tokenizer"]["revision"] == C.REVISIONS[s.hf_id]
    assert sc.load_info["ok"] and sc.load_info["n_unexpected_allowed"] == 1 and sc.prompt_format == "chat"


def test_wave_a_scorer_loading_is_unchanged(monkeypatch):
    calls = _fake_hf(monkeypatch, version="4.51.3")
    g = families.MODELS["gemma2-9b-it"]
    families.open_scorer(g)
    assert calls["class"] == "AutoModelForCausalLM" and calls["model"]["torch_dtype"] == "bf16"
    assert "output_loading_info" not in calls["model"] and "dtype" not in calls["model"]
    assert calls["tokenizer"] == {"revision": C.REVISIONS[g.hf_id], "trust_remote_code": True}


@pytest.mark.parametrize("info, device_map, words", [
    ({"missing_keys": ["model.language_model.layers.3.mlp.up_proj.weight"]}, {"": 0}, "missing"),
    ({"mismatched_keys": [("lm_head.weight", (1, 2), (2, 1))]}, {"": 0}, "mismatched"),
    ({"unexpected_keys": ["model.vision_tower.extra.weight"]}, {"": 0}, "unexpected"),
    ({}, {"model.language_model.layers.40": "cpu"}, "off the GPU"),
    ({}, {"model.language_model.layers.40": "disk"}, "off the GPU"),
    ({"error_msgs": ["size mismatch"]}, {"": 0}, "error"),
])
def test_loading_check_refuses_a_model_that_did_not_load_whole(monkeypatch, info, device_map, words):
    rec = check_loading(info, device_map, r"^mtp\.")
    assert not rec["ok"] and any(p.startswith(words) for p in rec["problems"]), rec
    _fake_hf(monkeypatch, info=info, device_map=device_map)
    with pytest.raises(RuntimeError, match="loading check failed"):
        families.open_scorer(B["qwen3.5-9b"])


def test_loading_check_without_a_pattern_allows_no_unexpected_tensor():
    assert not check_loading({"unexpected_keys": ["mtp.fc.weight"]}, {"": 0}, "")["ok"]
    assert check_loading({"unexpected_keys": ["mtp.fc.weight"]}, {"": 0}, r"^mtp\.")["ok"]
    assert check_loading({}, {"a": 0, "b": 1}, "")["devices"] == ["0", "1"]  # sharded over two GPUs is fine


def test_open_scorer_carries_every_setting_of_the_spec(mock_score):
    for tag, spec in B.items():
        sc = families.open_scorer(spec)
        assert (sc.model_tag, sc.prompt_format, sc.model_class) == (tag, spec.prompt_format, spec.model_class)
        assert sc.tokenizer_kwargs == spec.tokenizer_kwargs and sc.strict_load and sc.revision == C.REVISIONS[spec.hf_id]
        assert sc.unexpected_keys_allowed == spec.unexpected_keys_allowed and sc.template_kwargs == spec.template_kwargs
    assert families.open_scorer(B["qwen3.5-9b-base-plain"]).candidate_text("Yes") == " Yes"


def test_scorer_refuses_unknown_settings():
    with pytest.raises(ValueError):
        LogoddsScorer.for_tokenizer(B["qwen3.5-9b"].hf_id, object(), model_class="vision")
    with pytest.raises(ValueError):
        LogoddsScorer.for_tokenizer(B["qwen3.5-9b"].hf_id, object(), prompt_format="raw")


# ---------------- environment ----------------


ENV5 = {"transformers": "5.23.0", "kernels_importable": False}
ENV4 = {"transformers": "4.51.3", "kernels_importable": False}


def test_environment_guard(monkeypatch):
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    assert families.require_environment(B["gemma4-e4b-it"], ENV5)["checked"] is True
    with pytest.raises(SystemExit, match="serve2"):
        families.require_environment(B["gemma4-e4b-it"], ENV4)
    with pytest.raises(SystemExit, match="kernels"):
        families.require_environment(B["gemma4-e4b-it"], {**ENV5, "kernels_importable": True})
    a = families.MODELS["mistral-7b-v0.3"]
    assert families.require_environment(a, ENV4)["checked"] is True
    with pytest.raises(SystemExit, match="family_envcheck"):
        families.require_environment(a, ENV5)
    assert families.require_environment(families.envcheck_spec("mistral-7b-v0.3"), ENV5, envcheck=True)["checked"]
    with pytest.raises(SystemExit):
        families.require_environment(families.envcheck_spec("qwen3-8b"), ENV4, envcheck=True)
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    assert families.require_environment(B["gemma4-e4b-it"], ENV4)["checked"] == "mock"


def test_environment_record_lists_versions_and_kernel_packages():
    rec = families.environment_record()
    assert {"python", "transformers", "torch", "tokenizers", "kernels_importable", "fla_importable",
            "causal_conv1d_importable"} <= set(rec)


def test_envcheck_spec_rescores_an_earlier_model_under_its_own_tag():
    for tag in families.ENVCHECK_MODELS:
        s = families.envcheck_spec(tag)
        base = families.spec_for(tag)
        assert s.tag == tag + "-serve2" and s.hf_id == base.hf_id and s.template_kwargs == base.template_kwargs
        assert s.system == base.system and s.model_class is None and s.strict_load
        assert s.tokenizer_kwargs == {"mistral_format": False} and s.env == "serve2"
    with pytest.raises(ValueError):
        families.envcheck_spec("gemma4-e4b-it")
    assert families.ENVCHECK_MODELS == ("mistral-7b-v0.3", "qwen3-8b")


# ---------------- environment check: comparison and rule ----------------


def _pitems(n=12):
    return [{"id": f"t{i}", "game_id": f"g{i % 4}", "swapped_task_line": "x", "task_line": "y",
             "history_text": "Your task is to: y.\n> go", "action": "a", "candidates": {"o_plus": "You", "o_minus": "No"}}
            for i in range(n)]


def _scores(items, shift=None, model="m"):
    shift = shift or {}
    out = []
    for k, t in enumerate(items):
        for c in families.ENVCHECK_ARMS:
            base = (1.0 if "aligned" in c else -1.0 if "swapped" in c else 0.0) + 0.1 * k
            out.append({"transition_id": t["id"], "condition": c, "L": base + shift.get(c, 0.0), "model": model})
    return out


def _block(old, new, items, monkeypatch):
    monkeypatch.setattr(families.sizes, "arm_messages", lambda t, a: [{"role": "user", "content": "x"}])
    return families.envcheck_block(old, new, items)


def test_envcheck_passes_identical_scores(monkeypatch):
    items = _pitems()
    b = _block(_scores(items), _scores(items), items, monkeypatch)
    assert b["pass"] and b["n_paired"] == b["n_expected"] == 12 * 6
    assert b["yes_no"]["share_identical"] == 1 and b["two_candidate"]["goal_effect_diff"] == 0


@pytest.mark.parametrize("shift, failed", [
    ({"yn_aligned": 0.08, "yn_swapped": 0.08, "yn_absent": 0.08}, "yes_no_mean_abs"),
    ({"yn_aligned": 0.11}, "yes_no_goal_effect"),
    ({"aligned": 0.36, "swapped": 0.36, "absent": 0.36}, "two_candidate_mean_abs"),
    ({"aligned": 0.11}, "two_candidate_goal_effect"),
    ({"aligned": -0.11}, "two_candidate_goal_effect"),
    ({"yn_swapped": 0.11}, "yes_no_goal_effect"),
])
def test_envcheck_rule_thresholds(monkeypatch, shift, failed):
    items = _pitems()
    b = _block(_scores(items), _scores(items, shift), items, monkeypatch)
    assert not b["pass"] and not b["checks"][failed]
    assert [k for k, v in b["checks"].items() if not v] == [failed]


def test_envcheck_rule_just_inside_the_thresholds_passes(monkeypatch):
    items = _pitems()
    shift = {"yn_aligned": 0.06, "yn_swapped": 0.06, "yn_absent": 0.06, "aligned": 0.34, "swapped": 0.34, "absent": 0.34}
    assert _block(_scores(items), _scores(items, shift), items, monkeypatch)["pass"]
    assert _block(_scores(items), _scores(items, {"yn_aligned": 0.09}), items, monkeypatch)["pass"]


def test_envcheck_share_change_and_missing_scores_fail(monkeypatch):
    items = _pitems(50)
    old = _scores(items)
    # two items in fifty turn from 'moved toward Yes' to not: the share changes by 0.04, above the 0.02 allowed
    new = [dict(x) for x in old]
    for x in new:
        if x["transition_id"] in ("t0", "t1") and x["condition"] == "yn_swapped":
            x["L"] = 5.0
    b = _block(old, new, items, monkeypatch)
    assert not b["checks"]["yes_no_share"] and abs(b["yes_no"]["share_diff"] + 0.04) < 1e-9
    b = _block(old, [x for x in old if not (x["transition_id"] == "t3" and x["condition"] == "absent")], items, monkeypatch)
    assert not b["pass"] and not b["checks"]["complete"] and b["missing"] == ["t3/absent"]


def test_envcheck_nan_fails(monkeypatch):
    items = _pitems()
    new = _scores(items)
    new[0]["L"] = float("nan")
    assert not _block(_scores(items), new, items, monkeypatch)["pass"]


# ---------------- analysis: environment gate and predictions ----------------


def _family(verdicts, readings):
    return {"tests": {k: {"verdict": v, "estimate": e} for k, (v, e) in verdicts.items()}, "comparison_readings": dict(readings)}


def test_environment_gate_blocks_every_read_verdict_unless_passed():
    fam = _family({"a:item_share_toward_yes": ("supported", 0.8), "b:item_share_toward_yes": ("not interpretable", 0.5),
                   "a vs b:share_differs": ("not supported", 0.0)}, {"a vs b": "inconclusive (no difference detected)",
                                                                     "c vs d": "waiting"})
    fam["tests"]["c:item_share_toward_yes"] = {"verdict": "waiting", "estimate": 0.5}
    out = families.apply_environment_gate(json.loads(json.dumps(fam)), False)
    assert out["tests"]["a:item_share_toward_yes"]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert out["tests"]["a vs b:share_differs"]["verdict"] == families.ENVIRONMENT_BLOCKED
    # review 1 (S5): a competence verdict is a reading of wave B scores too, and no estimate stays in the record
    assert out["tests"]["b:item_share_toward_yes"]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert out["tests"]["c:item_share_toward_yes"]["verdict"] == "waiting"
    assert all(t["estimate"] is None for t in out["tests"].values())
    assert out["comparison_readings"] == {"a vs b": families.ENVIRONMENT_BLOCKED, "c vs d": "waiting"}
    for missing in (None, False):
        assert families.apply_environment_gate(json.loads(json.dumps(fam)), missing)["tests"][
            "a:item_share_toward_yes"]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert families.apply_environment_gate(json.loads(json.dumps(fam)), True)["tests"] == fam["tests"]


def _report(readings, estimates, share_verdicts, both=True, passed=True, yn_ci=(0.5, 1.5)):
    tests = {f"{t}:item_share_toward_yes": {"verdict": v, "estimate": 0.8} for t, v in share_verdicts.items()}
    tests.update({f"{k}:share_differs": {"verdict": "supported", "estimate": e} for k, e in estimates.items()})
    return {"models": {t: {} for t in B}, "prediction_check": {t: {"holds": both, "yes_no_ci": list(yn_ci)} for t in B},
            "primary_family": {"tests": tests, "comparison_readings": readings},
            # section 8 amendment: the environment check stays failed and no longer gates; the serve2 rescorings do
            "environment_check": {"pass": False, "gates": False}, "serve2_rescoring": {"pass": passed}}


def test_wave_b_predictions_are_read_as_logged():
    pairs = [f"{a} vs {b}" for a, b in families.WAVE_B_PLANNED]
    readings = {pairs[0]: "differs", pairs[1]: "matches", pairs[2]: "inconclusive (no difference detected)",
                pairs[3]: "differs, within the margin", pairs[4]: "differs"}
    est = {pairs[0]: -0.1, pairs[1]: 0.01, pairs[2]: -0.02, pairs[3]: 0.03, pairs[4]: 0.12}
    verdicts = {t: "supported" for t in families.WAVE_B_SHARE_TESTS}
    verdicts["gemma4-31b-it"] = "not interpretable"
    verdicts["qwen3.8-27b"] = "not supported"
    p = families.waveb_predictions(_report(readings, est, verdicts))
    assert p["B2"][pairs[0]]["holds"] is False  # differs with the newer lower
    assert all(p["B2"][k]["holds"] is True for k in pairs[1:4])
    assert p["B3"][pairs[4]]["holds"] is True
    assert p["B1"]["ministral3-8b-it"]["holds"] is True and p["B1"]["qwen3.8-27b"]["holds"] is False
    assert p["B1"]["gemma4-31b-it"]["holds"] is True and p["B1"]["qwen3.5-9b-plain"]["holds"] is True
    readings[pairs[4]] = "matches"
    assert families.waveb_predictions(_report(readings, est, verdicts))["B3"][pairs[4]]["holds"] is False
    readings[pairs[4]] = "differs"
    est[pairs[4]] = -0.05
    assert families.waveb_predictions(_report(readings, est, verdicts))["B3"][pairs[4]]["holds"] is False
    readings[pairs[4]] = families.ENVIRONMENT_BLOCKED
    assert families.waveb_predictions(_report(readings, est, verdicts))["B3"][pairs[4]]["holds"] is None
    verdicts["ministral3-8b-it"] = "waiting"
    assert families.waveb_predictions(_report(readings, est, verdicts))["B1"]["ministral3-8b-it"]["holds"] is None
    assert families.waveb_predictions(_report(readings, est, verdicts, both=False))["B1"]["gemma4-31b-it"]["holds"] is False


def test_b2_generation_pairs_carry_the_size_caveat():
    pairs = [f"{a} vs {b}" for a, b in families.WAVE_B_PLANNED]
    readings = {k: "inconclusive (no difference detected)" for k in pairs}
    p = families.waveb_predictions(_report(readings, {k: 0.0 for k in pairs}, {t: "supported" for t in families.WAVE_B_SHARE_TESTS}))
    for k in pairs[:4]:
        assert p["B2"][k]["caveat"] == families.PAIR_CAVEAT and p["B2"][k]["holds"] is True
    assert "0.49 / 0.89 / 0.70 / 0.71" in families.PAIR_CAVEAT and "not a generation effect" in families.PAIR_CAVEAT
    assert "caveat" not in p["B3"][pairs[4]]
    assert families.QWEN3_SIZE_SERIES["qwen3-4b"] == {"share": 0.491, "yes_no_delta_L": 2.27}


@pytest.mark.parametrize("share, ci, reading", [
    ("supported", (0.5, 1.0), "effect present (share supported)"),
    ("not supported", (0.4, 3.0), "effect carried by a minority of items (as Qwen3-4B)"),
    ("not supported", (-0.2, 0.3), "no detectable effect"),
    ("not supported", (0.0, 0.3), "no detectable effect"),  # a CI touching zero is not above it
    ("not supported", (-0.9, -0.1), "yes/no effect reversed (aligned below swapped)"),
    ("not interpretable", (0.5, 1.0), None),
    ("waiting", (0.5, 1.0), None),
    (None, (0.5, 1.0), None),
])
def test_b1_reading_separates_a_minority_effect_from_no_effect(share, ci, reading):
    """Review 1, M2: 'no detectable effect' only when the share and the yes/no effect are both null; a share that is
    not supported with the yes/no CI above zero is the Qwen3-4B pattern."""
    assert families.b1_reading(share, list(ci)) == reading


def test_b1_reading_in_the_predictions():
    verdicts = {t: "supported" for t in families.WAVE_B_SHARE_TESTS}
    verdicts["qwen3.8-27b"] = "not supported"
    p = families.waveb_predictions(_report({}, {}, verdicts, yn_ci=(0.3, 2.0)))
    assert p["B1"]["qwen3.8-27b"]["reading"] == families.B1_READINGS["minority"] and p["B1"]["qwen3.8-27b"]["holds"] is False
    assert p["B1"]["ministral3-8b-it"]["reading"] == families.B1_READINGS["supported"]
    assert p["B1"]["qwen3.5-9b-plain"]["reading"] is None  # no share test for the plain-format instruct run
    p = families.waveb_predictions(_report({}, {}, verdicts, yn_ci=(-0.3, 0.2)))
    assert p["B1"]["qwen3.8-27b"]["reading"] == families.B1_READINGS["null"]
    assert "Qwen3-4B" in families.PREDICTION_B and "0.49" in families.PREDICTION_B


def test_predictions_read_nothing_unless_the_environment_check_passed():
    """Review 1, S5(a): B1 held for runs without a share test (qwen3.5-9b-plain) although the check had failed."""
    pairs = [f"{a} vs {b}" for a, b in families.WAVE_B_PLANNED]
    readings = {k: "differs" for k in pairs}
    verdicts = {t: "supported" for t in families.WAVE_B_SHARE_TESTS}
    for passed in (False, None):
        p = families.waveb_predictions(_report(readings, {k: 0.1 for k in pairs}, verdicts, passed=passed))
        assert all(v["holds"] is None and v["both_formats_positive"] is None and v["reading"] is None
                   for v in p["B1"].values()), p["B1"]
        assert all(v["holds"] is None and v["estimate"] is None for v in list(p["B2"].values()) + list(p["B3"].values()))
        assert all(v["reading"] == families.ENVIRONMENT_BLOCKED for v in p["B2"].values())
    p = families.waveb_predictions(_report(readings, {k: 0.1 for k in pairs}, verdicts, passed=True))
    assert p["B1"]["qwen3.5-9b-plain"]["holds"] is True and p["B3"][pairs[4]]["holds"] is True
    # an explicit argument overrides the report
    assert families.waveb_predictions(_report(readings, {k: 0.1 for k in pairs}, verdicts, passed=True),
                                      False)["B1"]["qwen3.5-9b-plain"]["holds"] is None


COMPETENCE_CHECKS = ("auc_neutral", "yes_share_neutral_failures", "yes_share_neutral_successes", "median_yes_no_mass_neutral")


def _b3_blocks(failed=("median_yes_no_mass_neutral",), auc2=0.8, inst_ok=True):
    checks = {k: {"pass": k not in failed} for k in COMPETENCE_CHECKS}
    inst, base = families.WAVE_B_PLANNED[4]
    return {inst: {"competence": {"interpretable": inst_ok, "checks": {k: {"pass": True} for k in COMPETENCE_CHECKS}},
                   "auc": {"absent": {"auc": 0.9}}},
            base: {"competence": {"interpretable": not failed, "checks": checks}, "auc": {"absent": {"auc": auc2}}}}


def _b3_scores(shift_inst=1.0, shift_base=0.0, n=40):
    primary = [{"id": f"f{i}", "game_id": f"g{i % 8}"} for i in range(n)]
    successes = [{"id": f"s{i}", "game_id": f"g{i % 8}"} for i in range(8)]

    def scores(shift, model):
        out = []
        for k, t in enumerate(primary):
            noise = 0.1 * ((k * 7919) % 11 - 5)
            for c, v in (("aligned", shift + noise), ("swapped", 0.0), ("absent", -1.0)):
                out.append({"transition_id": t["id"], "condition": c, "L": v, "model": model})
        out += [{"transition_id": t["id"], "condition": "absent", "L": 2.0, "model": model} for t in successes]
        return out
    inst, base = families.WAVE_B_PLANNED[4]
    return {inst: scores(shift_inst, inst), base: scores(shift_base, base)}, primary, successes


def test_b3_falls_back_to_the_two_candidate_share_only_when_the_base_fails_only_coverage():
    scores, primary, successes = _b3_scores()
    fb = families.b3_fallback(_b3_blocks(), scores, primary, successes)
    assert fb["applies"] is True and fb["comparison"]["differs"] and fb["comparison"]["estimate"] > 0.3
    for blocks in (_b3_blocks(failed=()), _b3_blocks(failed=("auc_neutral", "median_yes_no_mass_neutral")),
                   _b3_blocks(auc2=0.69), _b3_blocks(inst_ok=False)):
        assert families.b3_fallback(blocks, scores, primary, successes)["applies"] is False
    assert families.b3_fallback({}, scores, primary, successes)["applies"] is None
    # read into B3: holds on a positive difference, not on none
    inst, base = families.WAVE_B_PLANNED[4]
    key = f"{inst} vs {base}"
    rep = _report({key: "not interpretable"}, {}, {}, passed=True)
    rep["b3_fallback"] = fb
    b3 = families.waveb_predictions(rep)["B3"][key]
    assert b3["holds"] is True and b3["basis"] == "two-candidate fallback"
    same, _, _ = _b3_scores(shift_inst=0.0)
    rep["b3_fallback"] = families.b3_fallback(_b3_blocks(), same, primary, successes)
    assert families.waveb_predictions(rep)["B3"][key]["holds"] is False
    neg, _, _ = _b3_scores(shift_inst=-1.0)  # the instruct model moves fewer failures: a clear negative difference
    rep["b3_fallback"] = families.b3_fallback(_b3_blocks(), neg, primary, successes)
    assert rep["b3_fallback"]["comparison"]["differs"] and families.waveb_predictions(rep)["B3"][key]["holds"] is False
    rep["serve2_rescoring"]["pass"] = False  # section 8 amendment: wave B's gate
    assert families.waveb_predictions(rep)["B3"][key]["holds"] is None
    rep["b3_fallback"] = {"applies": False}
    rep["serve2_rescoring"]["pass"] = True
    assert families.waveb_predictions(rep)["B3"][key]["holds"] is None  # not interpretable, no fallback


# ---------------- tools ----------------


def test_token_compare_needs_every_key_identical():
    from wf.tools import envcheck_tokens as et

    left = {"model": "m", "revision": "r", "records": {"a": [2, "x"], "b": [3, "y"]}}
    assert et.compare(left, json.loads(json.dumps(left)))["ok"]
    for right in ({**left, "records": {"a": [2, "x"], "b": [3, "z"]}}, {**left, "records": {"a": [2, "x"]}},
                  {**left, "records": {**left["records"], "c": [1, "w"]}}, {**left, "revision": "s"},
                  {**left, "model": "n"}):
        r = et.compare(left, right)
        assert not r["ok"], right
    assert not et.compare({**left, "records": {}}, {**left, "records": {}})["ok"]


def test_token_dump_covers_every_prompt_candidate_and_answer(mock_score):
    from tests.test_families import FakeTok
    from wf.tools import envcheck_tokens as et

    primary, successes = _items(3, 2)
    rec = et.dump("mistral-7b-v0.3", primary, successes, FakeTok(), "x")
    n_prompts = len(sizes.size_jobs(primary, successes))
    assert rec["n"] == n_prompts + 2 * 5 + 2 and rec["revision"] == C.REVISIONS["mistralai/Mistral-7B-Instruct-v0.3"]
    assert rec["records"]["answer|Yes"][0] == 1


def test_prefetch_uses_the_cache_transformers_5_reads(monkeypatch):
    from wf.tools import prefetch_models as pm

    hub_v5 = types.SimpleNamespace()  # transformers 5: no TRANSFORMERS_CACHE
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(utils=types.SimpleNamespace(hub=hub_v5)))
    monkeypatch.setitem(sys.modules, "transformers.utils", types.SimpleNamespace(hub=hub_v5))
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub_v5)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(constants=types.SimpleNamespace(HF_HUB_CACHE="/hub5")))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", types.SimpleNamespace(HF_HUB_CACHE="/hub5"))
    assert pm.transformers_cache() == "/hub5"
    hub_v4 = types.SimpleNamespace(TRANSFORMERS_CACHE="/tc4")
    monkeypatch.setitem(sys.modules, "transformers.utils", types.SimpleNamespace(hub=hub_v4))
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub_v4)
    assert pm.transformers_cache() == "/tc4"


def test_prefetch_fetches_the_template_files_wave_b_needs():
    import fnmatch

    from wf.tools import prefetch_models as pm

    def kept(name):
        return any(fnmatch.fnmatch(name, a) for a in pm.ALLOW) and not any(fnmatch.fnmatch(name, i) for i in pm.IGNORE)

    for name in ("chat_template.jinja", "merges.txt", "tokenizer.json", "tokenizer_config.json", "tekken.json",
                 "config.json", "model-00001-of-00018.safetensors", "model.safetensors.index.json"):
        assert kept(name), name
    for name in ("consolidated.safetensors", "SYSTEM_PROMPT.txt", "README.md", "original/params.json"):
        assert not kept(name), name


def test_prefetch_knows_wave_b_tags_and_reports_unknown_ones(capsys):
    from wf.tools import prefetch_models as pm

    assert pm.main(["no-such-model"]) == 2
    assert "gemma4-31b-it" in capsys.readouterr().err


def test_serve2_job_script_pins_the_environment_and_cache():
    text = (ROOT / "pace" / "family_score_serve2.sbatch").read_text()
    assert 'export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"' in text and "export HF_HUB_OFFLINE=1" in text
    assert 'export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"' in text
    assert 'find_spec("kernels")' in text and "< 5" in text
    assert text.index("export HF_HUB_OFFLINE=1") < text.index('run_wf "${RUN}"')
    assert text.index("\nactivate_serve") < text.index('if [[ "${WF_WEIGHTS:-shared}" == "local" ]]')
    assert 'rm -rf "${LOCAL_CACHE:?}"' in text and "trap cleanup EXIT" in text
    gate = (ROOT / "pace" / "families_waveb_gate.sbatch").read_text()
    assert "--partition=coc-cpu" in gate and "HF_HUB_OFFLINE=1" in gate and "family_gate_cpu" in gate
    tok = (ROOT / "pace" / "families_envcheck_tokens.sbatch").read_text()
    assert '"transformers==4.51.3" "tokenizers==0.21.4"' in tok and '"torch==2.6.0"' in tok
    assert 'WORK="${TMPDIR:-/tmp}/wf_envtok_' in tok and "compare_${TAG}.json" in tok


# ---------------- golden renders (made on ICE with transformers 5.23.0 and the pinned tokenizers) ----------------


def _rendered():
    if not RENDERED_B.exists():
        pytest.skip("golden renders not recorded yet")
    return json.loads(RENDERED_B.read_text())


def _expected_render(spec, msgs):
    """The documented format of each wave B template (notes/families_waveb_design.md section 5), built by hand."""
    sys_text, turns = msgs[0]["content"], msgs[1:]
    if spec.prompt_format == "plain":
        return prompts.plain_text_prompt(msgs)
    if spec.family == "qwen":
        assistant = "<think>\n\n</think>\n\n" if spec.tag == "qwen3.8-27b" else ""
        body = "".join(f"<|im_start|>user\n{m['content'].strip()}<|im_end|>\n" if m["role"] == "user" else
                       f"<|im_start|>assistant\n{assistant}{m['content'].strip()}<|im_end|>\n" for m in turns)
        return f"<|im_start|>system\n{sys_text.strip()}<|im_end|>\n{body}{spec.thinking_off_ending}"
    if spec.family == "gemma":
        body = "".join(f"<|turn>{'user' if m['role'] == 'user' else 'model'}\n{m['content'].strip()}<turn|>\n" for m in turns)
        return f"<bos><|turn>system\n{sys_text.strip()}<turn|>\n{body}{spec.thinking_off_ending}"
    if spec.family == "mistral":
        body = "".join(f"[INST]{m['content']}[/INST]" if m["role"] == "user" else f"{m['content']}</s>" for m in turns)
        return f"<s>[SYSTEM_PROMPT]{sys_text}[/SYSTEM_PROMPT]{body}"
    raise AssertionError(spec.family)


@pytest.mark.parametrize("tag", list(B))
def test_golden_render_is_the_documented_format(mock_score, tag):
    spec, item = B[tag], GOLDEN["item"]
    rec = _rendered()["models"][tag]
    assert rec["revision"] == C.REVISIONS[spec.hf_id] and rec["prompt_format"] == spec.prompt_format
    for arm in ("aligned", "swapped", "yn_aligned"):
        assert rec[arm] == _expected_render(spec, families.family_messages(spec, item, arm)), arm
    assert rec["aligned"].replace(item["task_line"], item["swapped_task_line"]) == rec["swapped"]
    assert "enable_thinking" not in rec["aligned"] and "<|think|>" not in rec["aligned"]


def _cached_tokenizer(spec):
    transformers = pytest.importorskip("transformers")
    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("wave B templates are read with transformers 5 (serve2)")
    try:
        return transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=C.REVISIONS[spec.hf_id], local_files_only=True,
                                                          cache_dir=os.environ.get("WF_TEST_HF_CACHE"), **spec.tokenizer_kwargs)
    except Exception as e:  # not cached here
        pytest.skip(f"{spec.hf_id} tokenizer not available offline at the pinned revision: {str(e)[:120]}")


@pytest.mark.parametrize("tag", list(B))
def test_pinned_template_renders_the_golden_item_and_passes_the_gate(mock_score, tag):
    """Runs in serve2 where the pinned tokenizer files are cached (ICE; WF_TEST_HF_CACHE names the cache)."""
    from wf.tools import family_gate_cpu as g

    spec, item = B[tag], GOLDEN["item"]
    tok = _cached_tokenizer(spec)
    pytest.importorskip("torch")
    scorer = g.scorer_for(spec, tok)
    renders = g.golden_renders(spec, scorer)
    golden = _rendered()["models"][tag]
    for arm in ("aligned", "swapped", "yn_aligned"):
        assert renders[arm] == golden[arm]
    ref = families.mistral_common_encoder(spec, cache_dir=os.environ.get("WF_TEST_HF_CACHE"), local_files_only=True) \
        if spec.mistral_common_check else None
    rec = families.tokenization_checks(scorer, spec, [item], [], ref)
    assert rec["ok"], rec["problems"]
    assert rec["first_token_pairs"] in ({"'You' vs 'Nothing'": 1}, {"' You' vs ' Nothing'": 1})


# ---------------- dry run: wave B scoring, the environment check and the analysis, all mocked ----------------


def _cfg(run_id, model, **kw):
    return {"run_id": run_id, "model": model, "sources": ["rq12_transitions_smoke"], "n_shards": 1, **kw}


def _envcheck_runs(monkeypatch, n, gpus=None):
    """Both environment-check runs, each on the GPU its job records (review 1, S4): the pinned type by default."""
    import wf.provenance as prov

    for tag in families.ENVCHECK_MODELS:
        gpu = (gpus or families.ENVCHECK_GPU)[tag]
        monkeypatch.setattr(prov, "gpu_names", lambda gpu=gpu: [gpu] if gpu else None)
        assert families.run_family_envcheck(_cfg(f"family_envcheck_{tag}", tag, n_items=n)) == 0
    monkeypatch.setattr(prov, "gpu_names", lambda: None)


def _rescore_runs(monkeypatch, gpus=None, tokens_ok=None, mock_root=None):
    """Every older comparison model rescored in serve2 (section 8 amendment), each on the GPU its job records (the
    pinned type by default), and the token comparisons (all identical by default) in rescore_tokens/."""
    import wf.provenance as prov

    for tag in families.RESCORE_MODELS:
        gpu = (gpus or families.RESCORE_GPU)[tag]
        monkeypatch.setattr(prov, "gpu_names", lambda gpu=gpu: [gpu] if gpu else None)
        assert families.run_family_rescore(_cfg(f"family_rescore_{tag}", tag)) == 0
    monkeypatch.setattr(prov, "gpu_names", lambda: None)
    if mock_root is not None:
        (mock_root / "rescore_tokens").mkdir(exist_ok=True)
        for tag in families.RESCORE_MODELS:
            ok = (tokens_ok or {}).get(tag, True)
            (mock_root / "rescore_tokens" / f"compare_{tag}.json").write_text(json.dumps(_token_compare(mock_root, tag, ok)))


def _token_compare(mock_root, tag, ok=True, **side_changes):
    """A token comparison as wf.tools.envcheck_tokens writes it, made at the rescoring's commit with the study stack
    on the left and transformers 5 on the right (review 2, S4); `side_changes`: {"left": {...}, "right": {...}}."""
    commit = (mock_root / f"family_rescore_{tag}" / "shard_00_of_01" / "git_sha").read_text().strip()
    pin = C.REVISIONS[families.spec_for(tag).hf_id]
    left = {"transformers": "4.51.3", "tokenizers": "0.21.4", "revision": pin, "git_sha": commit}
    right = {"transformers": "5.23.0", "tokenizers": "0.23.3", "revision": pin, "git_sha": commit}
    left.update(side_changes.get("left") or {})
    right.update(side_changes.get("right") or {})
    return {"ok": ok, "model": tag, "n_compared": 9, "left": left, "right": right}


def _smoke_items():
    from wf import rq1x
    from wf.rq2 import load_items

    sets = rq1x.rq1x_sets(load_items(("rq12_transitions_smoke",)))
    return sets["primary"], sets["successes"]


def _check(tag, tokens="default", run=None):
    """rescore_check of `tag`'s mock rescoring on the smoke items."""
    tokens = f"rescore_tokens/compare_{tag}.json" if tokens == "default" else tokens
    return families.rescore_check(tag, run or f"family_rescore_{tag}", tokens, *_smoke_items())


def test_dry_run_scores_every_wave_b_model_and_reads_the_family(mock_root, monkeypatch):
    for tag in B:
        assert families.run_family_score(_cfg(f"family_score_{tag}", tag)) == 0
        d = mock_root / f"family_score_{tag}" / "shard_00_of_01"
        saved = json.loads((d / "config.json").read_text())
        assert saved["model_spec"]["revision"] == C.REVISIONS[B[tag].hf_id] and saved["model_spec"]["wave"] == "B"
        gate = json.loads((d / "tokenization_checks.json").read_text())
        assert gate["skipped"] == "mock" and gate["environment"]["checked"] == "mock"
        assert gate["environment"]["resources"] == {"checked": "mock"}  # review 1, S6: checked at every job start
        assert {x["model"] for x in map(json.loads, (d / "scoring.jsonl").read_text().splitlines())} == {tag}
    for tag in ("gemma2-9b-it", "mistral-7b-v0.3"):
        assert families.run_family_score(_cfg(f"family_score_{tag}", tag)) == 0
    for tag in ("qwen3-8b", "qwen3-32b"):
        assert sizes.run_size_score(_cfg(f"size_score_{tag}", tag)) == 0
    assert families.run_family_placement(_cfg("family_placement_gemma4-e4b-it", "gemma4-e4b-it", n_items=5)) == 0
    assert families.run_family_nopreserve(_cfg("family_nopreserve_qwen3.8-27b", "qwen3.8-27b", n_items=5)) == 0
    _envcheck_runs(monkeypatch, 5)
    (mock_root / "envcheck_tokens").mkdir()
    for tag in families.ENVCHECK_MODELS:
        (mock_root / "envcheck_tokens" / f"compare_{tag}.json").write_text(json.dumps({"ok": True, "n_compared": 9}))
    from wf.runs import load_config

    env_cfg = {**load_config("families_envcheck"), "sources": ["rq12_transitions_smoke"], "n_items": 5}
    assert families.run_families_envcheck(env_cfg) == 0
    env = json.loads((mock_root / "families_envcheck" / "metrics.json").read_text())
    assert env["pass"] is True and set(env["models"]) == set(families.ENVCHECK_MODELS)
    assert all(m["yes_no"]["share_identical"] == 1 for m in env["models"].values())  # the mock scorer is deterministic
    # section 8 amendment: the environment check stays failed (as on ICE) and no longer gates; the rescorings do
    (mock_root / "families_envcheck" / "metrics.json").write_text(json.dumps({**env, "pass": False}))
    _rescore_runs(monkeypatch, mock_root=mock_root)
    # review 2, S9: the mock scores the same in both environments, so the original-environment Qwen3-8B scores are
    # perturbed (aligned task +0.5 nats in both formats) to tell the two apart
    for f in (mock_root / "size_score_qwen3-8b").glob("shard_*/scoring.jsonl"):
        recs = [json.loads(x) for x in f.read_text().splitlines()]
        for r in recs:
            if r["condition"] in ("aligned", "yn_aligned"):
                r["L"] += 0.5
        f.write_text("".join(json.dumps(r) + "\n" for r in recs))

    cfg = {**load_config("families_waveb_analysis"), "sources": ["rq12_transitions_smoke"]}
    # Llama's original-environment RQ1 runs are not mocked here (its serve2 rescoring is)
    cfg["reference_original"] = {k: v for k, v in cfg["reference_original"].items() if k != "llama3.1-8b"}
    assert families.run_families_analysis(cfg) == 0
    rep = json.loads((mock_root / "families_waveb_analysis" / "metrics.json").read_text())
    fam = rep["primary_family"]
    assert fam["n_tests_planned"] == 16 == len(fam["tests"]) and rep["environment_check"]["pass"] is False
    assert rep["environment_check"]["gates"] is False and rep["serve2_rescoring"]["pass"] is True
    assert set(rep["serve2_rescoring"]["models"]) == set(families.RESCORE_MODELS) and not rep["waiting"]
    # every reference is read from its serve2 rescoring; the originals only in the environment-sensitivity column
    for tag in families.RESCORE_MODELS:
        assert list(rep["reference"][tag]["read"]["problems"]) == []
    assert "qwen3-8b:family_rescore_qwen3-8b" in rep["provenance"]["runs_read"]
    assert "qwen3-8b (original environment):size_score_qwen3-8b" in rep["provenance"]["runs_read"]
    assert set(rep["environment_sensitivity"]) == set(families.RESCORE_MODELS) - {"llama3.1-8b"}
    sens = rep["environment_sensitivity"]
    assert {"serve2", "original", "serve2_minus_original", "serve2_detail", "original_detail", "environment_dependent",
            "rule"} == set(sens["qwen3-8b"])
    # the perturbed original scores show in the descriptive column, and only there
    assert sens["qwen3-8b"]["serve2_minus_original"]["yes_no_delta_L"] == pytest.approx(-0.5)
    assert sens["qwen3-8b"]["serve2_minus_original"]["two_candidate_delta_L"] == pytest.approx(-0.5)
    assert sens["qwen3-8b"]["environment_dependent"] is True
    assert all(sens[t]["serve2_minus_original"]["yes_no_delta_L"] == 0 and not sens[t]["environment_dependent"]
               for t in sens if t != "qwen3-8b")
    key = "qwen3.5-9b vs qwen3-8b"
    serve2_d = rep["comparisons"][key]["delta_L_difference_nats"]["estimate"]
    orig_d = rep["comparisons_original_environment"][key]["delta_L_difference_nats"]["estimate"]
    assert orig_d == pytest.approx(serve2_d - 0.5)
    assert rep["reference"]["qwen3-8b"]["yes_no"]["goal_effect"]["delta_L_nats"] == pytest.approx(
        sens["qwen3-8b"]["serve2"]["yes_no_delta_L"])
    dep = rep["environment_dependent_pairs"]
    assert key in dep and dep[key]["environment_dependent_references"] == ["qwen3-8b"]
    assert dep[key]["reading_serve2"] == rep["primary_family"]["comparison_readings"][key]
    assert all("qwen3-8b" in k for k in dep) and "gemma4-e4b-it vs gemma2-9b-it" not in dep
    assert "qwen3.5-9b vs qwen3-8b" in rep["comparisons_original_environment"]
    assert "qwen3.5-9b vs qwen3.5-9b-plain" not in rep["comparisons_original_environment"]
    assert rep["amendments"]["serve2_rescoring"] == families.RESCORE_AMENDMENT
    # the predictions are read (the rescorings passed) although the environment check failed
    assert all(v["both_formats_positive"] is not None for v in rep["predictions_b"]["B1"].values())
    assert set(rep["models"]) == set(B) and set(rep["predictions_b"]["B1"]) == set(B)
    assert set(rep["comparisons"]) >= {f"{a} vs {b}" for a, b in families.WAVE_B_PLANNED}
    assert "qwen3.5-9b vs qwen3.5-9b-plain" in rep["comparisons"] and "gemma4-e4b-it" in rep["placement"]
    assert not any(v["verdict"] == families.ENVIRONMENT_BLOCKED for v in fam["tests"].values())
    # review 1: the preserve_thinking check and the placement check are read by the one-fifth rule; B3's fallback
    tc = rep["template_checks"]["qwen3.8-27b"]
    assert tc["n_items"] == 5 and set(tc["one_fifth_rule"]) == {"yes_no", "two_candidate"}
    assert set(rep["placement"]["gemma4-e4b-it"]["one_fifth_rule"]) == {"yes_no", "two_candidate"}
    assert rep["b3_fallback"]["applies"] in (True, False) and rep["amendments"]["pair_caveat"] == families.PAIR_CAVEAT
    status = (mock_root / "families_waveb_analysis" / "STATUS.md").read_text()
    assert "share toward Yes" in status and "preserve_thinking=False minus native" in status

    # the same analysis with one rescoring's tokens differing reads no test
    (mock_root / "rescore_tokens" / "compare_gemma2-9b-it.json").write_text(
        json.dumps({"ok": False, "model": "gemma2-9b-it"}))
    assert families.run_families_analysis({**cfg, "run_id": "families_waveb_analysis_blocked"}) == 0
    rep = json.loads((mock_root / "families_waveb_analysis_blocked" / "metrics.json").read_text())
    assert rep["serve2_rescoring"]["pass"] is False and rep["primary_family"]["environment_check_passed"] is False
    assert rep["serve2_rescoring"]["models"]["gemma2-9b-it"]["checks"]["tokens_identical"] is False
    assert not any(v["verdict"] in ("supported", "not supported") for v in rep["primary_family"]["tests"].values())
    # review 1 (S5): nothing read from a wave B score is written anywhere: no estimate, block, comparison or check
    assert all(v["verdict"] in (families.ENVIRONMENT_BLOCKED, "waiting") and v["estimate"] is None and v["p"] is None
               for v in rep["primary_family"]["tests"].values())
    assert all(set(m) == {"withheld", "spec", "tokenization_gate", "records_outside_population"}
               for m in rep["models"].values())
    assert all(c == {"withheld": families.ENVIRONMENT_BLOCKED} for c in rep["comparisons"].values())
    assert rep["comparisons_original_environment"] and all(
        c == {"withheld": families.ENVIRONMENT_BLOCKED} for c in rep["comparisons_original_environment"].values())
    assert "gemma2-9b-it" in rep["environment_sensitivity"]  # reference models only: stays
    assert rep["environment_dependent_pairs"] and all(
        c == {"withheld": families.ENVIRONMENT_BLOCKED} for c in rep["environment_dependent_pairs"].values())
    assert rep["placement"]["gemma4-e4b-it"]["withheld"] == families.ENVIRONMENT_BLOCKED
    assert rep["template_checks"]["qwen3.8-27b"]["withheld"] == families.ENVIRONMENT_BLOCKED
    assert rep["b3_fallback"]["withheld"] == families.ENVIRONMENT_BLOCKED
    assert all(v["holds"] is None and v["both_formats_positive"] is None for v in rep["predictions_b"]["B1"].values())
    assert all(p == {"withheld": families.ENVIRONMENT_BLOCKED} for p in rep["prediction_check"].values())
    assert "qwen3-8b" in rep["reference"] and "primary_metrics" in rep["reference"]["qwen3-8b"]  # earlier models stay
    status = (mock_root / "families_waveb_analysis_blocked" / "STATUS.md").read_text()
    for tag in B:
        assert f"{tag}: {families.ENVIRONMENT_BLOCKED}; scores complete, nothing read" in status
    assert "share toward Yes" not in status.split("(reference)")[0] and "Holm p" not in status
    assert "item share" not in status and "placement" not in status and "B1 " not in status
    assert "serve2 rescoring of the older models not complete or not passed: ['gemma2-9b-it']" in status
    # the mock models may fail the competence gate; whatever would have been read is now blocked
    unblocked = json.loads((mock_root / "families_waveb_analysis" / "metrics.json").read_text())["primary_family"]
    for k, v in unblocked["tests"].items():
        if v["verdict"] in ("supported", "not supported"):
            assert rep["primary_family"]["tests"][k]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert unblocked["environment_check_passed"] is True


# ---------------- section 8 amendment (2026-11-23): the older comparison models rescored in serve2 ----------------


def test_rescore_list_is_every_older_model_a_wave_b_comparison_uses():
    from wf.runs import load_config

    older = {b for _, b in families.WAVE_B_PLANNED if b not in B} | set(families.REFERENCE_TAGS)
    assert set(families.RESCORE_MODELS) == older == {"qwen3-8b", "llama3.1-8b", "qwen3-32b", "gemma2-9b-it",
                                                     "mistral-7b-v0.3"}
    cfg = load_config("families_waveb_analysis")
    # descriptive comparisons, the placement check and the template check compare wave B models only
    assert all(a in B and b in B for a, b in cfg["descriptive_comparisons"])
    assert set(cfg["placement"]) <= set(B) and set(cfg["template_checks"]) <= set(B)
    assert families.RUN_ORDER_B[3:8] == tuple(f"family_rescore_{t}" for t in families.RESCORE_MODELS)
    assert families.RUN_ORDER_B.index("family_rescore_qwen3-32b") < families.RUN_ORDER_B.index("family_score_ministral3-8b-it")


def test_rescore_spec_keeps_the_earlier_prompts_and_loads_as_wave_b():
    for tag in families.RESCORE_MODELS:
        old, new = families.spec_for(tag), families.rescore_spec(tag)
        assert new.tag == tag + "-serve2" and new.env == "serve2" and new.wave == "rescore" and new.strict_load
        assert (new.hf_id, new.revision, new.system, new.attn_implementation, new.template_kwargs) == \
               (old.hf_id, old.revision, old.system, old.attn_implementation, old.template_kwargs)
        assert new.tokenizer_kwargs == ({"mistral_format": False} if tag == "mistral-7b-v0.3" else {})
        assert (new.gpu_min_gb, new.host_mem_min_gb) == ((80, 128) if tag == "qwen3-32b" else (24, 0))
        cfg = load_config_safe(f"family_rescore_{tag}")
        assert cfg["model"] == tag and cfg["n_shards"] == (2 if tag == "qwen3-32b" else 1)
    assert families.rescore_spec("gemma2-9b-it").attn_implementation == "eager"
    for tag in ("gemma4-e4b-it", "llama3.2-1b", "nope"):
        with pytest.raises((ValueError, KeyError)):
            families.rescore_spec(tag)


def load_config_safe(run_id):
    from wf.runs import RUNNERS, load_config

    assert run_id in RUNNERS
    return load_config(run_id)


def test_rescore_gpu_types_and_node_local_weights_are_pinned():
    from wf.tools import weights_manifest as wm

    assert families.RESCORE_GPU == {"mistral-7b-v0.3": "NVIDIA A40", "qwen3-8b": "NVIDIA L40S",
                                    "gemma2-9b-it": "NVIDIA L40S", "llama3.1-8b": "NVIDIA L40S",
                                    "qwen3-32b": "NVIDIA A100 80GB PCIe"}
    assert families.RESCORE_NODE_LOCAL == ("qwen3-32b",)
    m = wm.load_manifest("Qwen/Qwen3-32B", C.REVISIONS["Qwen/Qwen3-32B"])
    assert m["n_files"] == 24 and m["total_bytes"] == 65540268929 and m["dry_run_cross_check"]["same_files_and_sizes"]


def test_family_rescore_refuses_other_models(mock_root):
    with pytest.raises(SystemExit, match="rescoring covers"):
        families.run_family_rescore(_cfg("family_rescore_gemma4-e4b-it", "gemma4-e4b-it"))


def _rewrite(path, **changes):
    rec = json.loads(path.read_text())
    for k, v in changes.items():
        rec[k] = v
    path.write_text(json.dumps(rec))


REAL_GATE = {"ok": True, "skipped": None, "environment": {"checked": True, "transformers": "5.23.0"},
             "strict_load": True, "loading": {"ok": True}}


def test_rescore_check_passes_only_with_every_check(mock_root, monkeypatch):
    _rescore_runs(monkeypatch, mock_root=mock_root)
    for tag in families.RESCORE_MODELS:
        r = _check(tag)
        assert r["pass"] is True, (tag, r["checks"])
        assert set(r["checks"]) == ({"complete", "coverage", "tokens_identical", "tokens_provenance", "gpu_type",
                                     "gate_passed_in_serve2", "pinned_revision"}
                                    | ({"weights_hashed"} if tag == "qwen3-32b" else set()))
    q = "qwen3-8b"

    def failed(tokens="default", tag=q):
        return sorted(k for k, v in _check(tag, tokens)["checks"].items() if not v)

    assert failed(tokens=None) == ["tokens_identical", "tokens_provenance"]
    assert failed(tokens="rescore_tokens/missing.json") == ["tokens_identical", "tokens_provenance"]
    # another model's comparison: its revision and commit differ too
    assert "tokens_identical" in failed(tokens="rescore_tokens/compare_mistral-7b-v0.3.json")
    shard = mock_root / f"family_rescore_{q}" / "shard_00_of_01"
    gate = shard / "tokenization_checks.json"
    good = gate.read_text()
    _rewrite(gate, **{**REAL_GATE, "ok": False})
    assert failed() == ["gate_passed_in_serve2"]
    _rewrite(gate, **{**REAL_GATE, "environment": {"checked": True, "transformers": "4.51.3"}})
    assert failed() == ["gate_passed_in_serve2"]
    # review 2, M3: the loading check must have run and passed
    _rewrite(gate, **{**REAL_GATE, "strict_load": False})
    assert failed() == ["gate_passed_in_serve2"]
    _rewrite(gate, **{**REAL_GATE, "loading": {"ok": False}})
    assert failed() == ["gate_passed_in_serve2"]
    _rewrite(gate, **{**REAL_GATE, "loading": None})
    assert failed() == ["gate_passed_in_serve2"]
    _rewrite(gate, **REAL_GATE)
    assert failed() == []
    gate.write_text(good)
    conf = shard / "config.json"
    spec = json.loads(conf.read_text())["model_spec"]
    _rewrite(conf, model_spec={**spec, "revision": "0" * 40})
    assert failed() == ["pinned_revision"]
    _rewrite(conf, model_spec=spec)
    (shard / "STATUS.md").write_text("# STATUS — running\n")
    assert "complete" in failed()
    # the node-local model needs its download's hash check in every shard
    big = mock_root / "family_rescore_qwen3-32b" / "shard_00_of_01" / "tokenization_checks.json"
    _rewrite(big, **REAL_GATE)
    assert failed(tag="qwen3-32b") == ["weights_hashed"]
    _rewrite(big, weights_check={"ok": True, "revision": "0" * 40})
    assert failed(tag="qwen3-32b") == ["weights_hashed"]
    _rewrite(big, weights_check={"ok": True, "revision": C.REVISIONS["Qwen/Qwen3-32B"]})
    assert failed(tag="qwen3-32b") == []


def test_rescore_check_minor_paths_fail_closed(mock_root, monkeypatch):
    """Review 2, S10: a comparison without a verdict, a gate record without a checked environment, and a mock gate
    record without a mock environment all fail."""
    _rescore_runs(monkeypatch, mock_root=mock_root)
    q = "qwen3-8b"
    tokf = mock_root / "rescore_tokens" / f"compare_{q}.json"
    rec = json.loads(tokf.read_text())
    tokf.write_text(json.dumps({k: v for k, v in rec.items() if k != "ok"}))
    assert _check(q)["checks"]["tokens_identical"] is False
    tokf.write_text(json.dumps(rec))
    gate = mock_root / f"family_rescore_{q}" / "shard_00_of_01" / "tokenization_checks.json"
    _rewrite(gate, **{**REAL_GATE, "environment": {"transformers": "5.23.0"}})
    assert _check(q)["checks"]["gate_passed_in_serve2"] is False
    _rewrite(gate, ok=None, skipped="mock", environment={"checked": True, "transformers": "5.23.0"})
    assert _check(q)["checks"]["gate_passed_in_serve2"] is False


def test_rescore_check_token_provenance(mock_root, monkeypatch):
    """Review 2, S4: the comparison must be study stack against transformers 5, at the pin, at the rescoring's commit."""
    _rescore_runs(monkeypatch, mock_root=mock_root)
    q = "qwen3-8b"
    tokf = mock_root / "rescore_tokens" / f"compare_{q}.json"
    bad = [{"left": {"transformers": "5.23.0"}}, {"left": {"tokenizers": "0.23.3"}}, {"right": {"transformers": "4.51.3"}},
           {"left": {"revision": "0" * 40}}, {"right": {"revision": "0" * 40}}, {"left": {"git_sha": "0" * 40}},
           {"right": {"git_sha": "0" * 40}}, {"right": {"transformers": None}}]
    for change in bad:
        tokf.write_text(json.dumps(_token_compare(mock_root, q, True, **change)))
        r = _check(q)
        assert sorted(k for k, v in r["checks"].items() if not v) == ["tokens_provenance"], change
    tokf.write_text(json.dumps(_token_compare(mock_root, q)))
    assert _check(q)["pass"] is True


def test_rescore_check_coverage_and_item_hashes(mock_root, monkeypatch):
    """Review 2, S5: exactly one score per (item, condition) the analysis's items need, each with its item's hash."""
    _rescore_runs(monkeypatch, mock_root=mock_root)
    q = "qwen3-8b"
    f = mock_root / f"family_rescore_{q}" / "shard_00_of_01" / "scoring.jsonl"
    lines = f.read_text().splitlines()

    def cov(new_lines):
        f.write_text("\n".join(new_lines) + "\n")
        r = _check(q)
        return r["checks"]["coverage"], r["coverage"]

    ok, c = cov(lines)
    primary, successes = _smoke_items()
    assert ok is True and c["n_records"] == c["n_expected"] == len(sizes.size_jobs(primary, successes))
    assert cov(lines[1:])[0] is False  # one score missing
    assert cov(lines + lines[:1])[0] is False  # one score twice
    changed = json.loads(lines[0])
    changed["item_hash"] = "0" * 16
    ok, c = cov([json.dumps(changed)] + lines[1:])
    assert ok is False and c["n_item_hash_mismatch"] == 1
    other = json.loads(lines[0])
    other["condition"] = "not_an_arm"
    assert cov(lines + [json.dumps(other)])[0] is False  # a score the analysis does not expect
    assert cov(lines)[0] is True
    # the analysis's items, not the run's own: one item fewer on the analysis side is a mismatch
    assert families.rescore_check(q, f"family_rescore_{q}", f"rescore_tokens/compare_{q}.json", primary[1:],
                                  successes)["checks"]["coverage"] is False


def test_rescore_check_every_shard_counts(mock_root, monkeypatch):
    """Review 2, S8: Qwen3-32B is rescored in two shards; a failure in either shard fails the check."""
    import wf.provenance as prov

    tag, want = "qwen3-32b", families.RESCORE_GPU["qwen3-32b"]
    monkeypatch.setattr(prov, "gpu_names", lambda: [want])
    for i in (0, 1):
        assert families.run_family_rescore(_cfg(f"family_rescore_{tag}", tag, n_shards=2, shard=i)) == 0
    (mock_root / "rescore_tokens").mkdir(exist_ok=True)
    commit = (mock_root / f"family_rescore_{tag}" / "shard_00_of_02" / "git_sha").read_text().strip()
    pin = C.REVISIONS["Qwen/Qwen3-32B"]
    side = {"revision": pin, "git_sha": commit}
    (mock_root / "rescore_tokens" / f"compare_{tag}.json").write_text(json.dumps(
        {"ok": True, "model": tag, "left": {"transformers": "4.51.3", "tokenizers": "0.21.4", **side},
         "right": {"transformers": "5.23.0", **side}}))
    run = f"family_rescore_{tag}"

    def failed():
        r = families.rescore_check(tag, run, f"rescore_tokens/compare_{tag}.json", *_smoke_items())
        return sorted(k for k, v in r["checks"].items() if not v)

    assert failed() == []
    s1 = mock_root / run / "shard_01_of_02"
    jobs = (s1 / "jobs.jsonl").read_text()
    job = json.loads(jobs.splitlines()[0])
    (s1 / "jobs.jsonl").write_text(json.dumps({**job, "gpu": ["NVIDIA A100-PCIE-40GB"]}) + "\n")
    assert failed() == ["gpu_type"]  # one shard on another type
    (s1 / "jobs.jsonl").write_text(jobs + json.dumps({**job, "gpu": ["NVIDIA L40S"]}) + "\n")
    assert failed() == ["gpu_type"]  # one shard on two types
    (s1 / "jobs.jsonl").write_text(jobs)
    gate = s1 / "tokenization_checks.json"
    good = gate.read_text()
    _rewrite(gate, **{**REAL_GATE, "ok": False, "weights_check": {"ok": True, "revision": pin}})
    assert failed() == ["gate_passed_in_serve2"]  # one shard's gate failed
    gate.write_text(json.dumps(REAL_GATE))
    assert failed() == ["weights_hashed"]  # one shard without its hash check
    gate.write_text(good)
    conf = s1 / "config.json"
    spec = json.loads(conf.read_text())["model_spec"]
    _rewrite(conf, model_spec={**spec, "revision": "0" * 40})
    assert failed() == ["pinned_revision"]  # one shard at another revision
    _rewrite(conf, model_spec={**spec, "tag": "qwen3-8b-serve2"})
    assert "pinned_revision" in failed()  # one shard of another model
    _rewrite(conf, model_spec=spec)
    assert failed() == []


def test_rescore_check_fails_on_another_gpu_type(mock_root, monkeypatch):
    gpus = {**families.RESCORE_GPU, "qwen3-32b": "NVIDIA A100-PCIE-40GB", "mistral-7b-v0.3": None}
    _rescore_runs(monkeypatch, gpus=gpus, mock_root=mock_root)
    for tag in families.RESCORE_MODELS:
        assert _check(tag)["checks"]["gpu_type"] is (tag not in ("qwen3-32b", "mistral-7b-v0.3")), tag


def _gate_cfg(**over):
    given = {t: {"run": f"family_rescore_{t}", "tokens": f"rescore_tokens/compare_{t}.json"}
             for t in families.RESCORE_MODELS}
    refs = {t: families.rescore_reference(t, f"family_rescore_{t}") for t in families.RESCORE_MODELS}
    return {"rescore": given, "reference": refs, **over}


def test_rescore_gate_needs_every_model(mock_root, monkeypatch):
    _rescore_runs(monkeypatch, mock_root=mock_root)
    items = _smoke_items()
    cfg = _gate_cfg()
    assert families.rescore_gate(cfg, *items)["pass"] is True
    g = families.rescore_gate({**cfg, "rescore": {k: v for k, v in cfg["rescore"].items() if k != "llama3.1-8b"}}, *items)
    assert g["pass"] is False and g["missing_from_config"] == ["llama3.1-8b"]
    assert families.rescore_gate({}, *items)["pass"] is False
    g = families.rescore_gate({**cfg, "rescore": {**cfg["rescore"], "qwen3-32b": {"run": "family_rescore_nope",
                                                                                  "tokens": None}}}, *items)
    assert g["pass"] is False and g["models"]["qwen3-32b"]["checks"]["complete"] is False


def test_rescore_gate_requires_the_references_to_be_the_rescorings(mock_root, monkeypatch):
    """Review 2, S3: an analysis whose references read anything but the rescorings is blocked."""
    from wf.runs import load_config

    _rescore_runs(monkeypatch, mock_root=mock_root)
    items = _smoke_items()
    cfg = _gate_cfg()
    assert cfg["reference"] == load_config("families_waveb_analysis")["reference"]
    original = {"model": "qwen3-8b", "runs": {"size_score_qwen3-8b": list(families.ENVCHECK_ARMS)}}
    for bad in (original, {**cfg["reference"]["qwen3-8b"], "model": "qwen3-8b"},
                {**cfg["reference"]["qwen3-8b"], "runs": {"family_rescore_qwen3-8b": ["aligned"]}}):
        g = families.rescore_gate({**cfg, "reference": {**cfg["reference"], "qwen3-8b": bad}}, *items)
        assert g["pass"] is False and g["models"]["qwen3-8b"]["checks"]["reference_reads_this_rescoring"] is False
        assert all(m["pass"] for t, m in g["models"].items() if t != "qwen3-8b")
    g = families.rescore_gate({**cfg, "reference": {k: v for k, v in cfg["reference"].items() if k != "gemma2-9b-it"}},
                              *items)
    assert g["pass"] is False and g["models"]["gemma2-9b-it"]["checks"]["reference_reads_this_rescoring"] is False
    g = families.rescore_gate({**cfg, "reference": {**cfg["reference"], "llama3.2-1b": original}}, *items)
    assert g["pass"] is False and g["unexpected_references"] == ["llama3.2-1b"]


def test_rescoring_scores_the_earlier_prompts_with_the_rescoring_settings(mock_root, monkeypatch):
    """Review 2, M3: what run_family_rescore scores (each model's own prompts, Gemma 2's system text folded) and how
    it loads (rescore_spec: the loading check on, Mistral's tokenizer from tokenizer.json, the earlier attention)."""
    seen: dict = {}

    def spy_shard(cfg, tag, open_scorer, messages, gate=None, select=None, arms=None):
        seen[cfg["model"]] = {"tag": tag, "open": open_scorer, "messages": messages, "gate": gate, "select": select,
                              "arms": arms}
        return 0

    monkeypatch.setattr(families.sizes, "score_shard", spy_shard)
    opened: list = []

    class SpyScorer:
        def __init__(self, model_name, **kw):
            opened.append({"model_name": model_name, **kw})

    monkeypatch.setattr(families, "LogoddsScorer", SpyScorer)
    primary, successes = _smoke_items()
    jobs = sizes.size_jobs(primary, successes)
    for tag in families.RESCORE_MODELS:
        assert families.run_family_rescore(_cfg(f"family_rescore_{tag}", tag)) == 0
        s, old = seen[tag], families.spec_for(tag)
        assert s["tag"] == tag + "-serve2" and callable(s["gate"]) and s["select"] is None and s["arms"] is None
        for t, arm in jobs:
            assert s["messages"](t, arm) == families.family_messages(old, t, arm), (tag, arm)
        opened.clear()
        s["open"]()
        kw = opened[0]
        assert kw["model_name"] == old.hf_id and kw["model_tag"] == tag + "-serve2"
        assert kw["strict_load"] is True and kw["attn_implementation"] == old.attn_implementation
        assert kw["template_kwargs"] == old.template_kwargs and kw["model_class"] is None
        assert kw["prompt_format"] == "chat" and kw["unexpected_keys_allowed"] == ""
        assert kw["tokenizer_kwargs"] == ({"mistral_format": False} if tag == "mistral-7b-v0.3" else {})
    # Gemma 2: no system turn reaches the template; the system text opens the first user turn
    t, arm = jobs[0]
    g = seen["gemma2-9b-it"]["messages"](t, arm)
    assert all(m["role"] != "system" for m in g) and g[0]["content"].startswith(sizes.arm_messages(t, arm)[0]["content"])
    assert seen["gemma2-9b-it"]["messages"](t, arm) != sizes.arm_messages(t, arm)


def test_gate_records_the_loading_check(monkeypatch, tmp_path):
    """Review 2, M3: the shard's gate record says whether the loading check ran (strict_load) and its outcome."""
    monkeypatch.setattr(families, "tokenization_checks", lambda *a, **k: {"ok": True, "problems": [], "n_problems": 0})
    monkeypatch.setattr(families, "weights_check", lambda spec: None)
    spec = families.rescore_spec("qwen3-8b")
    model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"), dtype="torch.bfloat16",
                                  hf_device_map={})
    for strict in (True, False):
        scorer = types.SimpleNamespace(mock=False, _model=model, load_info={"ok": True}, strict_load=strict)
        families._gate(spec, {"checked": True, "transformers": "5.23.0"})(tmp_path, scorer, [], [])
        rec = json.loads((tmp_path / "tokenization_checks.json").read_text())
        assert rec["strict_load"] is strict and rec["loading"] == {"ok": True}


def test_token_dump_encodes_the_rescoring_prompts(mock_root):
    """Review 2, M3: the token comparison encodes exactly the prompts the rescoring scores (Gemma 2 folded), with each
    model's template settings, and records its commit and tokenizer stack (S4)."""
    from wf.tools import envcheck_tokens as et

    class StubTok:
        def __init__(self):
            self.prompts: list = []
            self.settings: list = []

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors=None, **kw):
            self.prompts.append(messages)
            self.settings.append(kw)
            return [[1, 2, 3]]

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            return types.SimpleNamespace(input_ids=[[7]])

    primary, successes = _smoke_items()
    jobs = sizes.size_jobs(primary, successes)
    for tag in families.RESCORE_MODELS:
        tok = StubTok()
        rec = et.dump(tag, primary, successes, tok, "5.23.0")
        spec = families.spec_for(tag)
        assert tok.prompts == [families.family_messages(spec, t, a) for t, a in jobs], tag
        assert all(s == {"enable_thinking": False, **spec.template_kwargs} for s in tok.settings)
        assert rec["git_sha"] and rec["git_sha"] != "unknown" and "tokenizers" in rec and rec["transformers"] == "5.23.0"
    left = {"model": "m", "revision": "r", "records": {"a": [1, "x"]}, "transformers": "4.51.3", "tokenizers": "0.21.4",
            "git_sha": "abc"}
    c = et.compare(left, {**left, "transformers": "5.23.0", "tokenizers": "0.23.3"})
    assert c["ok"] is True and c["left"]["git_sha"] == "abc" and c["left"]["tokenizers"] == "0.21.4"
    assert c["right"]["transformers"] == "5.23.0" and c["right"]["tokenizers"] == "0.23.3"


def test_analysis_waits_for_incomplete_rescorings(mock_root, monkeypatch):
    from wf.runs import load_config

    import wf.provenance as prov

    gpu = families.RESCORE_GPU["qwen3-8b"]
    monkeypatch.setattr(prov, "gpu_names", lambda: [gpu])
    assert families.run_family_rescore(_cfg("family_rescore_qwen3-8b", "qwen3-8b")) == 0
    cfg = {**load_config("families_waveb_analysis"), "sources": ["rq12_transitions_smoke"], "models": [],
           "reference_original": {}, "placement": {}, "template_checks": {}}
    assert families.run_families_analysis(cfg) == 0
    rep = json.loads((mock_root / "families_waveb_analysis" / "metrics.json").read_text())
    want = {f"serve2 rescoring {t}" for t in families.RESCORE_MODELS if t != "qwen3-8b"}
    assert want <= set(rep["waiting"]) and "serve2 rescoring qwen3-8b" not in rep["waiting"]
    assert rep["serve2_rescoring"]["pass"] is False and rep["withheld"] == families.ENVIRONMENT_BLOCKED


def _sens_block(share, r, yn, two, interpretable=True, auc=0.95, ci=(0.8, 0.9)):
    return {"primary_metrics": {"item_share_toward_yes": {"estimate": share, "ci": list(ci)},
                                "goal_to_evidence_ratio": {"estimate": r}},
            "yes_no": {"goal_effect": {"delta_L_nats": yn}}, "goal_effect": {"delta_L_nats": two},
            "competence": {"interpretable": interpretable, "checks": {"auc_neutral": {"value": auc}}}}


def test_environment_sensitivity_is_serve2_minus_original():
    out = families.environment_sensitivity(_sens_block(0.88, 1.2, 2.81, 6.02),
                                           _sens_block(0.89, 1.1, 2.78, 6.01, False, 0.69, (0.7, 0.95)))
    d = out["serve2_minus_original"]
    assert out["serve2"]["item_share_toward_yes"] == 0.88 and out["original"]["yes_no_delta_L"] == 2.78
    assert d == pytest.approx({"item_share_toward_yes": -0.01, "goal_to_evidence_ratio": 0.1,
                               "yes_no_delta_L": 0.03, "two_candidate_delta_L": 0.01})
    assert out["serve2_detail"] == {"interpretable": True, "auc_neutral_yes_no": 0.95, "item_share_ci": [0.8, 0.9]}
    assert out["original_detail"] == {"interpretable": False, "auc_neutral_yes_no": 0.69, "item_share_ci": [0.7, 0.95]}
    assert out["environment_dependent"] is False and out["rule"] == families.ENV_DEPENDENT_RULE


@pytest.mark.parametrize("d_share, d_dl, dependent", [
    (0.0201, 0.0, True), (-0.0201, 0.0, True), (0.0199, 0.0999, False), (-0.0199, -0.0999, False),
    (0.0, 0.1001, True), (0.0, -0.1001, True), (float("nan"), 0.0, True), (0.0, float("nan"), True),
])
def test_environment_dependent_rule(d_share, d_dl, dependent):
    """Review 2, S2, fixed before any score: item share moved by more than 0.02 or yes/no ΔL by more than 0.10."""
    assert families.ENV_DEPENDENT_RULE == {"item_share_max": 0.02, "yes_no_delta_L_max": 0.10}
    out = families.environment_sensitivity(_sens_block(0.5 + d_share, 1.0, 2.0 + d_dl, 6.0), _sens_block(0.5, 1.0, 2.0, 6.0))
    assert out["environment_dependent"] is dependent


def test_environment_dependent_pairs_carry_both_readings():
    def comp(differs, matches):
        return {"item_share_difference": {"differs": differs, "matches": matches}}
    report = {"environment_sensitivity": {"qwen3-8b": {"environment_dependent": True},
                                          "gemma2-9b-it": {"environment_dependent": False}},
              "primary_family": {"comparison_readings": {"qwen3.5-9b vs qwen3-8b": "matches"}},
              "comparisons": {"qwen3.5-9b vs qwen3-8b": comp(False, True), "gemma4-e4b-it vs gemma2-9b-it": comp(True, False),
                              "ministral3-8b-it vs qwen3-8b": comp(True, False), "qwen3.5-9b vs qwen3.5-9b-plain": comp(True, False)},
              "comparisons_original_environment": {"qwen3.5-9b vs qwen3-8b": comp(True, False),
                                                   "gemma4-e4b-it vs gemma2-9b-it": comp(True, False),
                                                   "ministral3-8b-it vs qwen3-8b": comp(False, False)}}
    out = families.environment_dependent_pairs(report)
    assert set(out) == {"qwen3.5-9b vs qwen3-8b", "ministral3-8b-it vs qwen3-8b"}
    a = out["qwen3.5-9b vs qwen3-8b"]
    assert a["reading_serve2"] == "matches" and a["reading_original_environment"] == "differs"
    assert a["environment_dependent_references"] == ["qwen3-8b"]
    b = out["ministral3-8b-it vs qwen3-8b"]  # not planned: the unadjusted serve2 reading
    assert b["reading_serve2"] == "differs" and b["reading_original_environment"] == "inconclusive (no difference detected)"


def test_serve2_job_runs_a_rescoring_only_on_its_pinned_gpu_and_qwen3_32b_node_local():
    text = (ROOT / "pace" / "family_score_serve2.sbatch").read_text()
    assert 'family_rescore_*) TAG=""; RESCORETAG="${RUN#family_rescore_}" ;;' in text
    assert "want, have = RESCORE_GPU[tag], gpu_names()" in text
    assert 'if have != [want]:\n    sys.exit(f"the serve2 rescoring of {tag} runs on one {want} only' in text
    assert 'if tag in RESCORE_NODE_LOCAL and weights != "local":' in text
    assert 'TAG="${TAG:-${RESCORETAG}}"' in text
    # the rescoring's GPU check runs before anything is downloaded or loaded
    assert text.index("RESCORE_GPU[tag]") < text.index("prefetch_models") < text.index('run_wf "${RUN}"')
    tok = (ROOT / "pace" / "families_envcheck_tokens.sbatch").read_text()
    assert 'OUT="${RUNS_DIR}/rescore_tokens"' in tok and 'RESCORE=(--rescore)' in tok
    assert 'TAGS="mistral-7b-v0.3 qwen3-8b gemma2-9b-it llama3.1-8b qwen3-32b"' in tok
    assert 'CACHE="${WORK}/hf"' in tok  # Qwen3-32B's tokenizer files node-local, never the shared cache


def test_token_tool_rescore_mode_covers_every_older_model_with_its_settings(monkeypatch, tmp_path, capsys):
    from wf.tools import envcheck_tokens as et

    class Stop(Exception):
        pass

    seen: dict = {}

    def from_pretrained(hf_id, **kw):
        seen.update(hf_id=hf_id, **kw)
        raise Stop

    stub = types.SimpleNamespace(__version__="5.23.0", AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "transformers", stub)
    out = str(tmp_path / "x.json")
    assert et.main(["dump", "gemma2-9b-it", "--out", out]) == 2  # not an environment-check model
    assert et.main(["dump", "gemma4-e4b-it", "--rescore", "--out", out]) == 2  # not an older comparison model
    for tag in families.RESCORE_MODELS:
        seen.clear()
        with pytest.raises(Stop):
            et.main(["dump", tag, "--rescore", "--out", out])
        assert seen["hf_id"] == families.spec_for(tag).hf_id and seen["local_files_only"] is True
        assert seen.get("mistral_format", "unset") == (False if tag == "mistral-7b-v0.3" else "unset")
    seen.clear()
    with pytest.raises(Stop):  # without --rescore, the environment check's settings, as before
        et.main(["dump", "qwen3-8b", "--out", out])
    assert seen["mistral_format"] is False


def test_prefetch_accepts_the_reference_models(capsys):
    from wf.tools import prefetch_models as pm

    assert pm.main(["qwen3-32b", "--node-local"]) == 2  # stops for the missing --cache, not for an unknown tag
    err = capsys.readouterr().err
    assert "--node-local needs --cache" in err and "unknown tags" not in err
    assert pm.main(["nope"]) == 2 and "unknown tags" in capsys.readouterr().err


def test_envcheck_fails_without_the_token_comparison(mock_root, monkeypatch):
    _envcheck_runs(monkeypatch, 4)
    assert families.run_family_score(_cfg("family_score_mistral-7b-v0.3", "mistral-7b-v0.3")) == 0
    assert sizes.run_size_score(_cfg("size_score_qwen3-8b", "qwen3-8b")) == 0
    from wf.runs import load_config

    cfg = {**load_config("families_envcheck"), "sources": ["rq12_transitions_smoke"], "n_items": 4}
    assert families.run_families_envcheck(cfg) == 0
    rep = json.loads((mock_root / "families_envcheck" / "metrics.json").read_text())
    assert rep["pass"] is False and all(m["pass"] for m in rep["models"].values())
    (mock_root / "envcheck_tokens").mkdir()
    for tag, ok in zip(families.ENVCHECK_MODELS, (True, False)):
        (mock_root / "envcheck_tokens" / f"compare_{tag}.json").write_text(json.dumps({"ok": ok}))
    assert families.run_families_envcheck({**cfg, "run_id": "families_envcheck_2"}) == 0
    assert json.loads((mock_root / "families_envcheck_2" / "metrics.json").read_text())["pass"] is False


def test_unknown_or_wrong_runs_are_refused(mock_root):
    with pytest.raises(SystemExit):
        families.run_family_score(_cfg("family_score_x", "qwen3-8b"))  # a reference model is not rescored this way
    with pytest.raises(SystemExit):
        families.run_family_envcheck(_cfg("family_envcheck_x", "gemma4-e4b-it"))
