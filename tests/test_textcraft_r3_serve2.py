"""TextCraft round 3 forward passes end to end in serve2 (transformers 5), for the two architectures new to the project:
Gemma 4 12B (`Gemma4UnifiedForConditionalGeneration`) and OLMo 3 7B Instruct (`Olmo3ForCausalLM`).

Each test builds a tiny random-weight model from the checkpoint's pinned config (wave B's `tiny_config`: two decoder
layers, small widths, the real vocabulary and layer kinds), takes the pinned real tokenizer, and runs `score_pair` and
`score_one_token_pair` through the scorer's own encoding on a round-2 record, after round 3's gate on that record. For
Gemma 4 12B it also checks the loading path without the real weights: the tiny model is saved with its vision-embedder
tensors under the names the real checkpoint stores them (read from the checkpoint's safetensors header), loaded back
with the scorer's strict loading check, and every parameter must load (transformers renames the old names). It also
runs the eager-against-SDPA agreement check on the real class.

They run only where transformers >= 5, torch and the pinned tokenizer and config files are available offline (ICE in
serve2, WF_TEST_HF_CACHE naming the cache). Elsewhere every test skips.

    WF_TEST_HF_CACHE=<cache> HF_HUB_OFFLINE=1 <serve2>/bin/python -m pytest -q -p no:cacheprovider tests/test_textcraft_r3_serve2.py
"""
from __future__ import annotations

import math
import os

import pytest

from wf import constants as C
from wf import families as FAM
from wf.scoring.logodds import LogoddsScorer, check_loading
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r3 as R3
from wf.textcraft import r3_score as S
from wf.textcraft import r3_tokcheck as T
from wf.textcraft.r2_items import read_records

SHOTS = P.fewshot_examples()
NEW = {"gemma4-12b-it": "Gemma4UnifiedForConditionalGeneration", "olmo3-7b-it": "Olmo3ForCausalLM"}
# The real checkpoint's names for the vision embedder (safetensors header at the pinned revision) and the class's own
# names (the meta-device architecture in transformers 5.23.0); every other tensor matches (proposal section 2.4).
GEMMA12_STORED = {
    "model.embed_vision.multimodal_embedder.embedding_projection.weight": "model.embed_vision.embedding_projection.weight",
    **{f"model.embed_vision.{k}": f"model.vision_embedder.{k}" for k in
       ("patch_dense.weight", "patch_dense.bias", "patch_ln1.weight", "patch_ln1.bias", "patch_ln2.weight",
        "patch_ln2.bias", "pos_embedding", "pos_norm.weight", "pos_norm.bias")},
}


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


def _tiny(transformers, spec, config):
    import torch

    from tests.test_families_waveb_serve2 import tiny_config

    if getattr(config, "text_config", None) is not None:
        cfg = tiny_config(config)
    else:  # a plain causal LM (OLMo 3)
        d = config.to_dict()
        d.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                 num_key_value_heads=2, head_dim=32)
        if d.get("layer_types"):
            kinds = list(dict.fromkeys(d["layer_types"]))
            d["layer_types"] = [kinds[i % len(kinds)] for i in range(2)]
        cfg = type(config).from_dict(d)
    torch.manual_seed(0)
    cls = transformers.AutoModelForImageTextToText if spec.model_class == "image_text_to_text" else transformers.AutoModelForCausalLM
    return cfg, cls, cls.from_config(cfg, dtype=torch.float32).eval()


def _record():
    return next(r for r in read_records() if r["kind"] == "main" and r["population"] == "main" and "both" in r["arms"])


@pytest.mark.parametrize("tag", list(NEW))
def test_tiny_random_new_model_scores_end_to_end(tag):
    transformers = _serve2()
    import torch

    spec = R3.MODELS[tag]
    tok = _pinned(transformers, spec, "tokenizer")
    config = _pinned(transformers, spec, "config")
    _cfg, _cls, model = _tiny(transformers, spec, config)
    assert type(model).__name__ == NEW[tag] == config.architectures[0]
    sc = LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=tag, template_kwargs=spec.template_kwargs,
                                     tokenizer_kwargs=spec.tokenizer_kwargs)
    sc._model = model
    rec = _record()
    gate = T.gate(sc, spec, [rec], SHOTS)
    assert gate["ok"], gate["problems"]
    msgs = R3.model_messages(spec, rec, "aligned", SHOTS)
    ids = sc._encode(msgs)
    assert isinstance(ids, torch.Tensor) and ids.dim() == 2 and ids.shape[0] == 1
    assert ids[0].tolist() == FAM._ids(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                               **sc.template_settings()))
    plus, minus = R.candidates(rec, "aligned")
    r = sc.score_pair(rec["id"], "aligned", msgs, plus, minus)
    assert math.isfinite(r.L) and r.n_tokens_plus == sc._encode_candidate(plus).shape[1]
    assert r.first_token_plus.strip() == "Craft" and r.first_token_minus.strip() == "Could"
    yn = R3.model_messages(spec, rec, "yn_aligned", SHOTS)
    one = sc.score_one_token_pair(rec["id"], "yn_aligned", yn, "Yes", "No")
    full = sc.score_pair(rec["id"] + "-full", "yn_aligned", yn, "Yes", "No")
    assert one.L == pytest.approx(full.L, abs=1e-4)


def test_gemma4_12b_checkpoint_names_load_strictly(tmp_path):
    """The tiny model saved with the real checkpoint's vision-embedder names loads with no missing, mismatched or
    unexpected parameter under the strict loading check (transformers renames the stored names)."""
    transformers = _serve2()
    import torch
    from safetensors.torch import save_file

    spec = R3.MODELS["gemma4-12b-it"]
    config = _pinned(transformers, spec, "config")
    cfg, cls, model = _tiny(transformers, spec, config)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}  # tied, as stored
    present = [k for k in GEMMA12_STORED if k in state]
    assert present, "the class has none of the expected vision-embedder parameters"
    for new_name in present:
        state[GEMMA12_STORED[new_name]] = state.pop(new_name)
    cfg.save_pretrained(tmp_path)
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    loaded, info = cls.from_pretrained(str(tmp_path), dtype=torch.float32, output_loading_info=True)
    rec = check_loading(info, None, spec.unexpected_keys_allowed)
    assert rec["ok"], rec["problems"]
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k


def test_gemma4_12b_attention_agreement_on_the_real_class():
    transformers = _serve2()

    spec = R3.MODELS["gemma4-12b-it"]
    tok = _pinned(transformers, spec, "tokenizer")
    config = _pinned(transformers, spec, "config")
    _cfg, _cls, model = _tiny(transformers, spec, config)
    sc = LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs)
    sc._model = model
    recs = [r for r in read_records() if r["kind"] == "main" and r["population"] == "main"][:5]
    out = S.attention_agreement(sc, spec, recs, SHOTS)
    assert out["ok"], out["problems"]
    assert out["switched"] == ["eager"] and out["restored"] == ["sdpa"]
