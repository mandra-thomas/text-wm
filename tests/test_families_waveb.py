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
    for tag, ref in cfg["reference"].items():
        spec = families.spec_for(tag)
        assert spec.hf_id in C.REVISIONS and ref["model"] in (tag, "llama3.1-8b-fewshot")
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
                 off_on_yes_no=True, on_text="THINK"):
        self.vocab: dict[str, int] = {"<s>": 0, "<e>": 1, "<u>": 2, "<a>": 3}
        self.all_special_ids, self.added_tokens_decoder = [0, 1, 2, 3], {}
        self.reads_switch, self.marker_when_off, self.suffix, self.off = reads_switch, marker_when_off, suffix, off
        self.system_text, self.off_on_yes_no, self.on_text = system_text, off_on_yes_no, on_text

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
        body = " ".join(f"<a> {off + ' ' if (k == len(turns) - 1) else ''}{m['content']} <e>" if m["role"] == "assistant"
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
    out = families.apply_environment_gate(json.loads(json.dumps(fam)), False)
    assert out["tests"]["a:item_share_toward_yes"]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert out["tests"]["a vs b:share_differs"]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert out["tests"]["b:item_share_toward_yes"]["verdict"] == "not interpretable"
    assert out["comparison_readings"] == {"a vs b": families.ENVIRONMENT_BLOCKED, "c vs d": "waiting"}
    for missing in (None, False):
        assert families.apply_environment_gate(json.loads(json.dumps(fam)), missing)["tests"][
            "a:item_share_toward_yes"]["verdict"] == families.ENVIRONMENT_BLOCKED
    assert families.apply_environment_gate(json.loads(json.dumps(fam)), True)["tests"] == fam["tests"]


def _report(readings, estimates, share_verdicts, both=True):
    tests = {f"{t}:item_share_toward_yes": {"verdict": v, "estimate": 0.8} for t, v in share_verdicts.items()}
    tests.update({f"{k}:share_differs": {"verdict": "supported", "estimate": e} for k, e in estimates.items()})
    return {"models": {t: {} for t in B}, "prediction_check": {t: {"holds": both} for t in B},
            "primary_family": {"tests": tests, "comparison_readings": readings}}


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


def test_dry_run_scores_every_wave_b_model_and_reads_the_family(mock_root):
    for tag in B:
        assert families.run_family_score(_cfg(f"family_score_{tag}", tag)) == 0
        d = mock_root / f"family_score_{tag}" / "shard_00_of_01"
        saved = json.loads((d / "config.json").read_text())
        assert saved["model_spec"]["revision"] == C.REVISIONS[B[tag].hf_id] and saved["model_spec"]["wave"] == "B"
        gate = json.loads((d / "tokenization_checks.json").read_text())
        assert gate["skipped"] == "mock" and gate["environment"]["checked"] == "mock"
        assert {x["model"] for x in map(json.loads, (d / "scoring.jsonl").read_text().splitlines())} == {tag}
    for tag in ("gemma2-9b-it", "mistral-7b-v0.3"):
        assert families.run_family_score(_cfg(f"family_score_{tag}", tag)) == 0
    for tag in ("qwen3-8b", "qwen3-32b"):
        assert sizes.run_size_score(_cfg(f"size_score_{tag}", tag)) == 0
    assert families.run_family_placement(_cfg("family_placement_gemma4-e4b-it", "gemma4-e4b-it", n_items=5)) == 0
    for tag in families.ENVCHECK_MODELS:
        assert families.run_family_envcheck(_cfg(f"family_envcheck_{tag}", tag, n_items=5)) == 0
    (mock_root / "envcheck_tokens").mkdir()
    for tag in families.ENVCHECK_MODELS:
        (mock_root / "envcheck_tokens" / f"compare_{tag}.json").write_text(json.dumps({"ok": True, "n_compared": 9}))
    from wf.runs import load_config

    env_cfg = {**load_config("families_envcheck"), "sources": ["rq12_transitions_smoke"], "n_items": 5}
    assert families.run_families_envcheck(env_cfg) == 0
    env = json.loads((mock_root / "families_envcheck" / "metrics.json").read_text())
    assert env["pass"] is True and set(env["models"]) == set(families.ENVCHECK_MODELS)
    assert all(m["yes_no"]["share_identical"] == 1 for m in env["models"].values())  # the mock scorer is deterministic

    cfg = {**load_config("families_waveb_analysis"), "sources": ["rq12_transitions_smoke"]}
    cfg["reference"] = {k: v for k, v in cfg["reference"].items() if k != "llama3.1-8b"}  # its RQ1 runs are not mocked here
    assert families.run_families_analysis(cfg) == 0
    rep = json.loads((mock_root / "families_waveb_analysis" / "metrics.json").read_text())
    fam = rep["primary_family"]
    assert fam["n_tests_planned"] == 16 == len(fam["tests"]) and rep["environment_check"]["pass"] is True
    assert set(rep["models"]) == set(B) and set(rep["predictions_b"]["B1"]) == set(B)
    assert set(rep["comparisons"]) >= {f"{a} vs {b}" for a, b in families.WAVE_B_PLANNED}
    assert "qwen3.5-9b vs qwen3.5-9b-plain" in rep["comparisons"] and "gemma4-e4b-it" in rep["placement"]
    assert not any(v["verdict"] == families.ENVIRONMENT_BLOCKED for v in fam["tests"].values())

    # the same analysis with a failed environment check reads no test
    m = json.loads((mock_root / "families_envcheck" / "metrics.json").read_text())
    (mock_root / "families_envcheck" / "metrics.json").write_text(json.dumps({**m, "pass": False}))
    assert families.run_families_analysis({**cfg, "run_id": "families_waveb_analysis_blocked"}) == 0
    rep = json.loads((mock_root / "families_waveb_analysis_blocked" / "metrics.json").read_text())
    assert rep["environment_check"]["pass"] is False
    assert not any(v["verdict"] in ("supported", "not supported") for v in rep["primary_family"]["tests"].values())


def test_envcheck_fails_without_the_token_comparison(mock_root):
    for tag in families.ENVCHECK_MODELS:
        assert families.run_family_envcheck(_cfg(f"family_envcheck_{tag}", tag, n_items=4)) == 0
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
