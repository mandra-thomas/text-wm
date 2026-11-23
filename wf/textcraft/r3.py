"""TextCraft round 3: models, families, pairs and run order (notes/textcraft_r3_design.md).

Round 3 scores round 2's records and prompts (wf/textcraft/r2.py, imported, never changed) with contemporary
open-weight instruction models in serve2 (transformers 5). Five of its models are wave B's specs unchanged
(wf.families.WAVE_B); two are new here (Gemma 4 12B and OLMo 3 7B Instruct); the thinking arm is Qwen3.5-9B with its
thinking switch on, scored after its own reasoning (wf/textcraft/r3_think.py), never with the instruction runs.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from wf import constants as C
from wf import families as FAM
from wf.families import ModelSpec
from wf.textcraft import r2 as R

# ---------------- models ----------------

_GEMMA_SUFFIX = "<|channel>thought\n<channel|>"
_WAVE_B = dict(wave="B", env="serve2", model_class="image_text_to_text", tokenizer_kwargs={"mistral_format": False},
               strict_load=True)

NEW_SPECS: dict[str, ModelSpec] = {
    # Same chat template as Gemma 4 31B, byte for byte (proposal section 2.4), so the 31B handling: the empty thought
    # channel as the generation suffix and in the thinking-off ending. A new class (Gemma4UnifiedForConditionalGeneration,
    # encoder-free): every language-model tensor matches it; its ten vision-embedder tensors are stored under older
    # names that transformers renames on load, so no unexpected tensor is allowed (a failed renaming leaves missing
    # parameters and the strict loading check stops the job).
    "gemma4-12b-it": ModelSpec(
        "gemma4-12b-it", "google/gemma-4-12B-it", "gemma", "Gemma 4", "2026-06-03", 11.96, "native", None, {},
        "Gemma 4 12B Unified (encoder-free). Template identical to Gemma 4 31B-it's: system turn; with thinking off the "
        "generation prompt ends with an empty thought channel that a rendered reply does not contain; the score is "
        "read after it. Sliding window 1,024 on 40 of 48 layers.",
        **_WAVE_B, generation_suffix=_GEMMA_SUFFIX, thinking_switch=True, thinking_on_markers=("<|think|>",),
        thinking_off_ending="<|turn>model\n" + _GEMMA_SUFFIX, gpu_min_gb=40, hub_created="2026-05-23",
        attn_agreement_check=True),
    # A plain causal LM with ChatML and a native system role; its default function-calling system prompt appears only
    # without a system message, and is forbidden in every prompt. No thinking mode.
    "olmo3-7b-it": ModelSpec(
        "olmo3-7b-it", "allenai/Olmo-3-7B-Instruct", "olmo", "OLMo 3", "2025-11-20", 7.30, "native", None, {},
        "ChatML; native system role; no thinking mode; the reply follows '<|im_start|>assistant\\n'. Sliding window "
        "4,096 on 24 of 32 layers (no TextCraft input reaches it). Fully open training data.",
        wave="B", env="serve2", model_class=None, strict_load=True, thinking_off_ending="<|im_start|>assistant\n",
        forbidden_in_prompt=("function-calling",), gpu_min_gb=24, hub_created="2025-11-19"),
}

CORE = ("qwen3.5-9b", "ministral3-8b-it", "gemma4-12b-it")
EXTENSION = ("qwen3.8-27b", "gemma4-31b-it", "olmo3-7b-it")
GEMMA_CORE, GEMMA_FALLBACK = "gemma4-12b-it", "gemma4-e4b-it"
THINK_TAG = "qwen3.5-9b-think"  # the thinking arm (exploratory): Qwen3.5-9B's weights, thinking on

MODELS: dict[str, ModelSpec] = {
    **{t: FAM.WAVE_B[t] for t in ("qwen3.5-9b", "ministral3-8b-it", "qwen3.8-27b", "gemma4-31b-it", "gemma4-e4b-it")},
    **NEW_SPECS,
}
TAGS = tuple(MODELS)

# The thinking arm's spec: the same weights and loading as qwen3.5-9b, the template with thinking on.
THINK_SPEC = replace(FAM.WAVE_B["qwen3.5-9b"], tag=THINK_TAG, template_kwargs={"enable_thinking": True},
                     template_notes="Thinking on: the generation prompt ends '<think>\\n'; the model reasons, then the "
                                    "replies are scored after its reasoning (r3_think).")

# ---------------- GPU types (design section 10) ----------------

L40S = "NVIDIA L40S"
A100_80 = "A100 80GB"  # any NVIDIA A100 whose name or memory says 80 GB (checked by gpu_type_ok)
GPU_PIN: dict[str, tuple[str, ...]] = {  # pinned type; then the declared alternative (two L40S) where there is one
    "qwen3.5-9b": (L40S,), "ministral3-8b-it": (L40S,), "gemma4-12b-it": (L40S,), "gemma4-e4b-it": (L40S,),
    "olmo3-7b-it": (L40S,), THINK_TAG: (L40S,),
    "qwen3.8-27b": (A100_80, "2x " + L40S), "gemma4-31b-it": (A100_80, "2x " + L40S),
}


def gpu_class(gpus: list[str] | None) -> str | None:
    """The recorded GPU list as a pinned class: "NVIDIA L40S", "2x NVIDIA L40S", "A100 80GB", or None (other)."""
    if not gpus:
        return None
    if all(g == L40S for g in gpus):
        return L40S if len(gpus) == 1 else f"{len(gpus)}x {L40S}"
    if len(gpus) == 1 and "A100" in gpus[0] and "80" in gpus[0]:
        return A100_80
    return None


def gpu_type_ok(tag: str, gpus: list[str] | None) -> bool:
    """Whether a shard's recorded GPUs are the model's pinned type or its declared alternative."""
    return gpu_class(gpus) in GPU_PIN[tag]


# ---------------- families, pairs, seeds (design sections 7 and 8) ----------------

FAMILY_SIZES = {"A3": 30, "A3x": 30, "B3": 30}
# Older (round 2) -> newer (round 3), fixed before any score.
PAIRS: tuple[tuple[str, str], ...] = (("qwen3-8b", "qwen3.5-9b"), ("mistral-7b-v0.3", "ministral3-8b-it"),
                                      ("gemma2-9b-it", "gemma4-12b-it"))
ENVCHECK_MODELS = ("qwen3-8b", "mistral-7b-v0.3")
ENVCHECK_SUFFIX = FAM.ENVCHECK_SUFFIX  # "-serve2"
ENVCHECK_N = 300
ENVCHECK_MIN_COMPARABLE = 200  # records whose round-2 shard ran on the rescoring's GPU type
RESCORE_MODELS = ("qwen3-8b", "mistral-7b-v0.3", "gemma2-9b-it")  # fallback if the environment check fails
FORMAT_N = 300
FORMAT_ARMS = ("aligned", "swapped", "yn_aligned", "yn_swapped")
SMOKE_N = 20


def seed(name: str) -> str:
    return f"{C.RNG_SEED}|textcraft|r3|{name}"


RUN_ORDER = ("textcraft_r3_envcheck_qwen3-8b", "textcraft_r3_envcheck_mistral-7b-v0.3",
             "textcraft_r3_smoke_qwen3.5-9b", "textcraft_r3_smoke_ministral3-8b-it", "textcraft_r3_smoke_gemma4-12b-it",
             "textcraft_r3_score_qwen3.5-9b", "textcraft_r3_score_ministral3-8b-it", "textcraft_r3_score_gemma4-12b-it",
             # extension, the format checks and the thinking arm, as GPUs allow
             "textcraft_r3_smoke_olmo3-7b-it", "textcraft_r3_score_olmo3-7b-it",
             "textcraft_r3_smoke_qwen3.8-27b", "textcraft_r3_score_qwen3.8-27b", "textcraft_r3_nopreserve_qwen3.8-27b",
             "textcraft_r3_smoke_gemma4-31b-it", "textcraft_r3_score_gemma4-31b-it",
             "textcraft_r3_placement_gemma4-12b-it",
             "textcraft_r3_think_smoke_qwen3.5-9b", "textcraft_r3_think_qwen3.5-9b",
             "textcraft_r3_analysis")


def core_tags(gemma: str = GEMMA_CORE) -> tuple[str, ...]:
    """The three core tags, with the Gemma slot as decided by the fallback rule."""
    if gemma not in (GEMMA_CORE, GEMMA_FALLBACK):
        raise ValueError(f"the Gemma core slot is {GEMMA_CORE} or its fallback {GEMMA_FALLBACK}, not {gemma!r}")
    return tuple(gemma if t == GEMMA_CORE else t for t in CORE)


def pairs(gemma: str = GEMMA_CORE) -> tuple[tuple[str, str], ...]:
    return tuple((old, gemma if new == GEMMA_CORE else new) for old, new in PAIRS)


# ---------------- prompts ----------------


def model_messages(spec: ModelSpec, rec: dict, arm: str, shots) -> list[dict[str, str]]:
    """r2.messages as the model reads it (folded into the first user turn for a 'fold' spec: the Gemma placement
    check and the rescoring of Gemma 2)."""
    return R.model_messages(spec, rec, arm, shots)


def base_tag(tag: str) -> str:
    """The model a scored tag belongs to: a check variant's model (`-firstturn`, `-nopreserve`), an earlier model's
    serve2 rescoring (`-serve2`), or the tag itself."""
    for suffix in (FAM.FIRST_TURN, FAM.NO_PRESERVE, ENVCHECK_SUFFIX):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def spec_for(tag: str) -> ModelSpec:
    """A round-3 model, the thinking arm, or an earlier model as serve2 rescores it (`<tag>-serve2`)."""
    if tag in MODELS:
        return MODELS[tag]
    if tag == THINK_TAG:
        return THINK_SPEC
    if tag.endswith(ENVCHECK_SUFFIX) and tag[: -len(ENVCHECK_SUFFIX)] in R.MODELS:
        return FAM.envcheck_spec(tag[: -len(ENVCHECK_SUFFIX)])
    if tag.endswith(FAM.FIRST_TURN) and tag[: -len(FAM.FIRST_TURN)] in MODELS:
        return FAM.first_turn_spec(MODELS[tag[: -len(FAM.FIRST_TURN)]])
    if tag.endswith(FAM.NO_PRESERVE) and tag[: -len(FAM.NO_PRESERVE)] in MODELS:
        return FAM.nopreserve_spec(MODELS[tag[: -len(FAM.NO_PRESERVE)]])
    raise KeyError(f"unknown round-3 model {tag!r}")


def spec_record(tag: str) -> dict[str, Any]:
    return spec_for(tag).record()
