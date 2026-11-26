"""Wave B forward passes end to end in serve2 (review 1, M1; 2026-11-22, before any wave B score).

transformers 5's apply_chat_template(tokenize=True) returns a BatchEncoding, which broke every chat-format score; the
gate did not notice because it unwraps ids itself. These tests run `score_pair` and `score_one_token_pair` through the
scorer's own encoding with each pinned wave B tokenizer and a tiny random-weight model built from that checkpoint's
pinned config (two decoder layers, small widths, the real vocabulary), so the text-only forward passes of
Gemma4ForConditionalGeneration, Qwen3_5ForConditionalGeneration and Mistral3ForConditionalGeneration run on a CPU.
For Gemma 4 they also run the eager-against-SDPA agreement check (review 1, S9) on the real class.

They run only where transformers >= 5, torch and the pinned tokenizer and config files are available offline: on
ICE in serve2, with WF_TEST_HF_CACHE naming the shared model cache. Elsewhere every test skips.

    WF_TEST_HF_CACHE=/storage/ice1/2/1/gmat3/hfcache HF_HUB_OFFLINE=1 \\
        /storage/ice1/2/1/gmat3/envs/serve2/bin/python -m pytest -q -p no:cacheprovider tests/test_families_waveb_serve2.py
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest

from wf import constants as C, families
from wf.scoring.logodds import LogoddsScorer
from tests.test_rq1 import mock_score  # noqa: F401  (builds the prompts without the Word2World data)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())
B = families.WAVE_B
ARCHITECTURES = {"Gemma4ForConditionalGeneration", "Qwen3_5ForConditionalGeneration", "Mistral3ForConditionalGeneration"}


def _serve2():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("serve2 only (transformers >= 5)")
    return transformers


def _pinned(transformers, spec, what: str):
    load = dict(revision=C.REVISIONS[spec.hf_id], local_files_only=True, cache_dir=os.environ.get("WF_TEST_HF_CACHE"))
    try:
        if what == "tokenizer":
            return transformers.AutoTokenizer.from_pretrained(spec.hf_id, **load, **spec.tokenizer_kwargs)
        return transformers.AutoConfig.from_pretrained(spec.hf_id, **load)
    except Exception as e:  # not cached here
        pytest.skip(f"{spec.hf_id} {what} not available offline at the pinned revision: {str(e)[:120]}")


def _two_types(types: list[str], n: int) -> list[str]:
    """`n` layer types that keep every kind the model has (linear and full attention; sliding and full)."""
    kinds = list(dict.fromkeys(types))
    return [kinds[i % len(kinds)] for i in range(n)]


def tiny_config(config):
    """The pinned config with every width shrunk and two decoder layers (four for Gemma 4 E4B, whose last layers
    share keys and values with earlier ones of the same type: two shared layers keep that path). The vocabulary,
    rope settings, soft-capping and layer kinds stay as pinned."""
    d = copy.deepcopy(config.to_dict())
    t = d["text_config"]
    n = 4 if t.get("num_kv_shared_layers") else 2
    t.update(num_hidden_layers=n, hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
             head_dim=32)
    if t.get("layer_types"):
        t["layer_types"] = _two_types(t["layer_types"], n)
    if t.get("num_kv_shared_layers"):
        t["num_kv_shared_layers"] = 2
    if "per_layer_config" in t:  # Gemma 4: rebuilt from layer_types for the full-attention layers' wider heads
        del t["per_layer_config"]
        t["global_head_dim"] = 64
        if t.get("attention_k_eq_v"):
            t["num_global_key_value_heads"] = 1
    for key, value in (("global_head_dim", 32), ("hidden_size_per_layer_input", 16), ("num_global_key_value_heads", 1),
                       ("linear_key_head_dim", 16), ("linear_value_head_dim", 16), ("linear_num_key_heads", 2)):
        if t.get(key):
            t[key] = value
    if t.get("linear_num_value_heads"):  # keep the ratio of value to key heads (2 for 9B, 3 for 27B)
        t["linear_num_value_heads"] = 2 * (t["linear_num_value_heads"] // 16)
    rope = t.get("rope_parameters") or {}
    if "mrope_section" in rope:  # sections cover half the rotary dimensions: 32 * partial_rotary_factor / 2
        prf = rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        half = int(32 * prf / 2)
        rope["mrope_section"] = [half - 2 * (half // 3), half // 3, half // 3]
    if t.get("mlp_only_layers"):
        t["mlp_only_layers"] = []
    v = d.get("vision_config") or {}
    for key, value in (("num_hidden_layers", 1), ("depth", 1), ("hidden_size", 32), ("intermediate_size", 64),
                       ("num_attention_heads", 2), ("num_heads", 2), ("num_key_value_heads", 2), ("head_dim", 16),
                       ("global_head_dim", 16)):
        if key in v and v[key] is not None:
            v[key] = value
    if "out_hidden_size" in v:
        v["out_hidden_size"] = t["hidden_size"]
    a = d.get("audio_config") or {}
    if a.get("num_hidden_layers"):
        a["num_hidden_layers"] = 1
    return type(config).from_dict(d)


def _scorer(spec, tok, model):
    s = LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs,
                                    prompt_format=spec.prompt_format)
    s.mock = False  # the prompts are built with WF_MOCK_SCORE=1 (no Word2World data); the scores are real
    s._model = model
    return s


@pytest.mark.parametrize("tag", list(B))
def test_tiny_random_wave_b_model_scores_end_to_end(mock_score, tag):
    transformers = _serve2()
    import torch

    spec, item = B[tag], GOLDEN["item"]
    tok = _pinned(transformers, spec, "tokenizer")
    config = _pinned(transformers, spec, "config")
    torch.manual_seed(0)
    model = transformers.AutoModelForImageTextToText.from_config(tiny_config(config), dtype=torch.float32).eval()
    assert type(model).__name__ == config.architectures[0] and type(model).__name__ in ARCHITECTURES
    sc = _scorer(spec, tok, model)

    # the prompt's ids are a tensor, whatever the template call returns (BatchEncoding in transformers 5)
    msgs = families.family_messages(spec, item, "aligned")
    ids = sc._encode(msgs)
    assert isinstance(ids, torch.Tensor) and ids.dim() == 2 and ids.shape[0] == 1
    if spec.prompt_format == "chat":
        raw = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **sc.template_settings())
        assert ids[0].tolist() == families._ids(raw)

    # two-candidate score, then the yes/no score both ways: one forward pass, and the full teacher-forced pass
    r = sc.score_pair(item["id"], "aligned", msgs, item["candidates"]["o_plus"], item["candidates"]["o_minus"])
    assert math.isfinite(r.L) and r.n_tokens_plus == sc._encode_candidate(item["candidates"]["o_plus"]).shape[1]
    assert r.first_token_plus.strip() == "You" and r.first_token_minus.strip() == "Nothing"
    yn = families.family_messages(spec, item, "yn_aligned")
    one = sc.score_one_token_pair(item["id"], "yn_aligned", yn, "Yes", "No")
    full = sc.score_pair(item["id"] + "-full", "yn_aligned", yn, "Yes", "No")
    assert math.isfinite(one.L) and one.first_token_plus.strip() == "Yes" and one.first_token_minus.strip() == "No"
    assert abs(one.L - full.L) < 1e-3 and abs(one.logp_plus - full.logp_plus) < 1e-3


@pytest.mark.parametrize("tag", ["gemma4-e4b-it", "gemma4-31b-it"])
def test_eager_and_sdpa_agree_on_the_real_gemma4_class(mock_score, tag):
    """Review 1, S9, on Gemma4ForConditionalGeneration: the model switches to eager and back, and the two agree."""
    transformers = _serve2()
    import torch

    spec, item = B[tag], GOLDEN["item"]
    tok = _pinned(transformers, spec, "tokenizer")
    config = _pinned(transformers, spec, "config")
    torch.manual_seed(0)
    # loaded as scoring loads it, with transformers' default attention (SDPA for the text model; E4B's audio encoder
    # has no SDPA and gets eager)
    model = transformers.AutoModelForImageTextToText.from_config(tiny_config(config), dtype=torch.float32).eval()
    items = [{**item, "id": f"{item['id']}-{k}"} for k in range(2)]
    rec = families.attention_agreement(_scorer(spec, tok, model), spec, items)
    assert rec["ok"], rec["problems"]
    assert set(rec["switched"]) == {"eager"} and set(rec["restored"]) == {"sdpa"} and len(rec["scores"]) == 4


def test_float32_attention_check_leaves_the_bf16_model_bit_for_bit(mock_score):
    """D-B1 (2026-11-25): the attention check casts every floating tensor to float32 and back; on the real Gemma 4 E4B
    class loaded in bf16, every parameter and buffer afterwards equals its value before, bit for bit, in its own
    dtype, and the deciding comparison ran in float32."""
    transformers = _serve2()
    import torch

    spec, item = B["gemma4-e4b-it"], GOLDEN["item"]
    tok = _pinned(transformers, spec, "tokenizer")
    config = _pinned(transformers, spec, "config")
    torch.manual_seed(0)
    model = transformers.AutoModelForImageTextToText.from_config(tiny_config(config), dtype=torch.bfloat16).eval()
    before = {k: (v.dtype, v.detach().clone()) for k, v in model.state_dict().items()}
    assert any(d == torch.bfloat16 for d, _ in before.values())
    items = [{**item, "id": f"{item['id']}-{k}"} for k in range(2)]
    rec = families.attention_agreement(_scorer(spec, tok, model), spec, items)
    assert rec["ok"], rec["problems"]
    assert rec["precision"] == "float32" and rec["n_tensors_cast"] > 0 and "bf16_frozen_rule" in rec
    after = model.state_dict()
    assert set(after) == set(before)
    for k, (dtype, v) in before.items():
        assert after[k].dtype == dtype and torch.equal(after[k], v), k


@pytest.mark.parametrize("tag", ["gemma4-e4b-it-firstturn", "qwen3.8-27b-nopreserve"])
def test_check_variants_pass_the_gate_with_the_pinned_template(mock_score, tag):
    """Review 1, S7 and S8: the placement check's first-turn variant and Qwen3.8's preserve_thinking=False variant pass
    the gate on the golden item with the real template; the latter writes no think block before its ending."""
    transformers = _serve2()
    from wf.tools import family_gate_cpu as g

    spec, item = families.variant_spec(tag), GOLDEN["item"]
    tok = _pinned(transformers, spec, "tokenizer")
    rec = families.tokenization_checks(g.scorer_for(spec, tok), spec, [item], [])
    assert rec["ok"], rec["problems"]
    text = g.render(g.scorer_for(spec, tok), spec, families.family_messages(spec, item, "aligned"))
    if tag.endswith(families.NO_PRESERVE):
        assert rec["waveb_prompt_checks"]["forbidden_before_ending"] == ["<think>"]
        assert text.count("<think>") == 1 and text.endswith(spec.thinking_off_ending)
    else:
        assert "<|turn>system" not in text and text.startswith("<bos><|turn>user\n")
