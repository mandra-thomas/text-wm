"""Model families, wave A (decided 2026-11-20, before any score; notes/families_design.md).

The size series' scoring (wf/sizes.py: RQ2's prompt, example set 1, history only; the aligned, swapped and neutral
tasks, each as the two-candidate score and as the yes/no question; the determined successes for AUC) applied to
models from other families and generations. One scoring run per model (`family_score_<tag>`), with three additions
the Qwen3 series did not need:

1. Chat templates. Gemma 2's template refuses a system message, so the system text is folded into the first user
   turn (`fold_system`), the same way in every condition. Llama 3.2's template writes today's date into the system
   header unless given one, so it gets the fixed date its own template falls back to (and Llama 3.1's template
   uses): the prompt cannot change with the day a shard runs.
2. A tokenization gate (audit C9). Before any score (on a resume too), through the scorer's own encoding: every
   candidate text in the shard (primary items and successes) must tokenize alone exactly as the chat template
   tokenizes it as the assistant's reply; "Yes" and "No" must each be one token, equal to the template's rendering
   of that word as the reply after every yes/no prompt of every item; and the two candidates' first tokens must
   differ. The results go to tokenization_checks.json in the shard's folder, and a failure stops the job before it
   scores anything.
3. Attention. Gemma 2 is loaded with eager attention: transformers 4.51.3 defaults it to SDPA, which silently drops
   Gemma 2's attention-logit soft-capping, so the logits would not be the trained model's. The scorer refuses to
   continue if the loaded model reports another implementation.

`families_analysis` reads every family run and, for comparison, Qwen3-8B (size_score_qwen3-8b) and Llama-3.1-8B
(rq1_score, rq1x_score and rq1v_score, which hold the same prompts), and gives each the size series' block plus a
spread-standardised effect, both flip definitions by name, and first-token coverage, every model on the same
(item, condition) population. `family_placement_<tag>` (added 2026-11-20, before any wave A score was read) rescores
300 seeded primary items with Mistral's system text in the first user turn instead of its template's last-turn
placement; the analysis sets the goal effect under both placements side by side on those items.

Amendment (2026-11-20, before any wave A score was read): the primary cross-model metrics are the item share moved
toward Yes and the goal-to-evidence ratio R, both on the yes/no format (`primary_metrics`), in one Holm-corrected
family over the wave A models (`primary_family`); a model must pass a competence gate fixed before data
(`competence_gate`); models are compared on the same items with a paired game bootstrap, "matching" when R is within
a factor of 1.5 by TOST (`compare_models`); yes/no answers and flips are read as argmax choices (`argmax_answers`).

Third amendment (2026-11-20, still before any wave A score was read; found on the archived reference runs only): the
item share is the single primary cross-model metric. R is secondary and descriptive: across families it mostly
reflects how far each model pushes its log-odds on failures (its denominator), not goal bias. The primary family is
the item share per wave A model plus the planned paired comparisons of item share (difference and TOST equivalence
within ±0.05), Holm-corrected at a fixed size (`primary_family`). An R comparison falls back to the paired
difference R_A - R_B when the ratio is not estimable (`compare_models`).
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from collections import Counter
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf import rq1x, sizes
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation, paired_conditions
from wf.rq2 import SOURCES, complete_shards, load_items
from wf.runs import _effect, _init_run
from wf.scoring.logodds import LogoddsScorer
from wf.scoring.prompts import plain_text_prompt
from wf.util import read_jsonl, require_filed, runs_root, write_json, write_status


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    hf_id: str
    family: str
    generation: str
    release: str  # public release (announcement) date
    params_b: float  # parameters in billions, from the checkpoint's safetensors metadata on the Hub
    system: str  # "native": the template takes the system message; "fold": folded into the first user turn
    attn_implementation: str | None = None  # None: transformers' default
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    template_notes: str = ""
    # Wave B (2026-11-22, notes/families_waveb_design.md). The defaults are wave A's settings, and a spec that keeps
    # them records exactly as before (record()), so wave A's saved run configs are unchanged.
    wave: str = "A"
    env: str = "serve"  # "serve2": transformers 5 (envs/serve2); wave A and the references run in "serve"
    model_class: str | None = None  # "image_text_to_text": the checkpoint's vision-language class
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_format: str = "chat"  # "plain": no template (base against instruct)
    generation_suffix: str = ""  # text the generation prompt ends with that a rendered reply does not contain
    thinking_switch: bool = False  # the template reads enable_thinking
    thinking_on_markers: tuple[str, ...] = ()  # written with thinking on, never with it off
    thinking_on_ending: str = ""  # the thinking-on generation prompt ends with this
    thinking_off_ending: str = ""  # every scored prompt (thinking off) ends with these tokens
    forbidden_in_prompt: tuple[str, ...] = ()  # text no scored prompt may contain (a template's default system prompt)
    mistral_common_check: bool = False  # every prompt's ids must equal mistral-common's encoding
    strict_load: bool = False  # the loading check (LogoddsScorer.check_loading)
    unexpected_keys_allowed: str = ""  # checkpoint tensors the class may leave unused (a regular expression)
    gpu_min_gb: int = 0  # GPU memory the bf16 weights need, with margin (0: any card the job asks for)
    hub_created: str = ""  # the Hub repository's creation date (wave B's release field is a month)
    # Review 1 (2026-11-22, before any wave B score; notes/families_waveb_review1.md):
    host_mem_min_gb: int = 0  # job memory (SLURM_MEM_PER_NODE) loading needs; 0: whatever the job script asks for
    attn_agreement_check: bool = False  # S9: a few items scored under eager and SDPA attention must agree first
    forbidden_before_ending: tuple[str, ...] = ()  # text no prompt may hold before its thinking-off ending (S8)

    @property
    def revision(self) -> str | None:
        return C.REVISIONS.get(self.hf_id)

    def record(self) -> dict[str, Any]:
        """The spec as saved in a run's config: the wave B fields only where they differ from wave A's defaults."""
        rec = asdict(self)
        for f in fields(self)[WAVE_A_FIELDS:]:
            default = f.default_factory() if f.default_factory is not MISSING else f.default  # type: ignore[misc]
            if rec[f.name] == default:
                del rec[f.name]
        return {**rec, "revision": self.revision}


WAVE_A_FIELDS = 10  # ModelSpec's fields up to template_notes; the rest were added for wave B


# Wave A: transformers 4.51.3 loads all four, so they run in the study's environment.
MODELS: dict[str, ModelSpec] = {
    "gemma2-9b-it": ModelSpec(
        "gemma2-9b-it", "google/gemma-2-9b-it", "gemma", "Gemma 2", "2024-06-27", 9.24, "fold", "eager", {},
        "No system role: the template raises on one, so WM_SYSTEM opens the first user turn, followed by a blank "
        "line. Roles must alternate user/model; contents are trimmed; '<bos>' comes from the template. Eager "
        "attention for the attention-logit soft-capping (50); the final logits are tanh-capped at 30."),
    "mistral-7b-v0.3": ModelSpec(
        "mistral-7b-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "mistral", "Mistral 7B v0.3", "2024-05-22", 7.25,
        "native", None, {},
        "The template takes a system message but places it in the last user turn ('[INST] system\\n\\nquery"
        "[/INST]'), not the first. Roles must alternate. A reply follows '[/INST]' after a space, so a candidate is "
        "scored as the template writes a reply (' You ...'). No generation prompt is added."),
    "llama3.2-1b": ModelSpec(
        "llama3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "llama", "Llama 3.2", "2024-09-25", 1.24, "native", None,
        {"date_string": "26 Jul 2024"},
        "Llama 3 header format, as Llama 3.1. Without date_string the template writes today's date (strftime_now) "
        "into the system header, so the date is fixed at the template's own fallback, which Llama 3.1's template "
        "always uses. Pruned and distilled from Llama 3.1 8B and 70B."),
    "llama3.2-3b": ModelSpec(
        "llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "llama", "Llama 3.2", "2024-09-25", 3.21, "native", None,
        {"date_string": "26 Jul 2024"},
        "As Llama 3.2 1B (the same tokenizer files)."),
}

# Already scored, read for comparison; their prompts are the ones above (the size series and RQ1/RQ2 runs).
REFERENCE: dict[str, ModelSpec] = {
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen", "Qwen3", "2025-04-29", 8.19, "native", None,
                          {"enable_thinking": False}, "Thinking off; the template writes an empty think block."),
    "llama3.1-8b": ModelSpec("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama", "Llama 3.1", "2024-07-23",
                             8.03, "native", None, {}, "Fixed date 26 Jul 2024 in the system header."),
    # Wave B's large Qwen pair (2026-11-22): scored in the size series (size_score_qwen3-32b), same prompts.
    "qwen3-32b": ModelSpec("qwen3-32b", "Qwen/Qwen3-32B", "qwen", "Qwen3", "2025-04-29", 32.76, "native", None,
                           {"enable_thinking": False}, "Thinking off; the template writes an empty think block."),
}

# ---------------- wave B (2026-11-22, notes/families_waveb_design.md; before any wave B score) ----------------

_QWEN_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_ON = "<|im_start|>assistant\n<think>\n"
_GEMMA4_E4B_SHARED_KV = r"language_model\.layers\.(2[4-9]|3[0-9]|4[01])\.self_attn\.(k_proj|v_proj|k_norm)\."
_WAVE_B = dict(wave="B", env="serve2", model_class="image_text_to_text", tokenizer_kwargs={"mistral_format": False},
               strict_load=True)


def _qwen_b(tag: str, hf_id: str, generation: str, release: str, params: float, notes: str, *, plain: bool = False,
            markers: tuple[str, ...] = (), gpu: int = 24, created: str = "", host_mem: int = 0) -> ModelSpec:
    if plain:
        return ModelSpec(tag, hf_id, "qwen", generation, release, params, "native", None, {}, notes, **_WAVE_B,
                         prompt_format="plain", unexpected_keys_allowed=r"^mtp\.", gpu_min_gb=gpu, hub_created=created,
                         host_mem_min_gb=host_mem)
    return ModelSpec(tag, hf_id, "qwen", generation, release, params, "native", None, {}, notes, **_WAVE_B,
                     thinking_switch=True, thinking_on_markers=markers, thinking_on_ending=_QWEN_ON,
                     thinking_off_ending=_QWEN_OFF, unexpected_keys_allowed=r"^mtp\.", gpu_min_gb=gpu,
                     hub_created=created, host_mem_min_gb=host_mem)


# Every wave B model is loaded in serve2 (transformers 5) with its checkpoint's vision-language class, the fast
# tokenizer and the pinned chat template (mistral_format=False), the loading check, and thinking off. Parameters:
# every tensor of the architecture (vision and audio encoders included), counted on the meta device.
WAVE_B: dict[str, ModelSpec] = {
    "ministral3-8b-it": ModelSpec(
        "ministral3-8b-it", "mistralai/Ministral-3-8B-Instruct-2512-BF16", "mistral", "Ministral 3", "2025-12", 8.92,
        "native", None, {},
        "System text first ('[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]'); without a system message the template inserts "
        "Mistral's default prompt with today's date, so the gate checks that it never appears. A reply follows "
        "'[/INST]' with no space. No thinking mode. The bf16 release; the default repository's language model is FP8.",
        **_WAVE_B, thinking_off_ending="[/INST]", forbidden_in_prompt=("Le Chat", "The current date"),
        mistral_common_check=True, gpu_min_gb=24, hub_created="2025-10-31"),
    "qwen3.5-9b": _qwen_b(
        "qwen3.5-9b", "Qwen/Qwen3.5-9B", "Qwen3.5", "2026-02", 9.41,
        "ChatML. Thinking is on by default (the generation prompt ends '<think>\\n'); enable_thinking=False writes "
        "an empty think block, as for Qwen3-8B. Linear-attention layers run in PyTorch (no kernels package).",
        created="2026-02-27"),
    "qwen3.5-9b-plain": _qwen_b(
        "qwen3.5-9b-plain", "Qwen/Qwen3.5-9B", "Qwen3.5", "2026-02", 9.41,
        "The instruct model in the plain-text format (no template), the instruct side of base against instruct.",
        plain=True, created="2026-02-27"),
    "qwen3.5-9b-base-plain": _qwen_b(
        "qwen3.5-9b-base-plain", "Qwen/Qwen3.5-9B-Base", "Qwen3.5 base", "2026-02", 9.41,
        "The base model in the plain-text format (no template), the base side of base against instruct.",
        plain=True, created="2026-02-26"),
    "gemma4-e4b-it": ModelSpec(
        "gemma4-e4b-it", "google/gemma-4-E4B-it", "gemma", "Gemma 4", "2026-03", 7.94, "native", None, {},
        "System turn supported (Gemma 2 had none). Thinking off by default; on, the template writes '<|think|>'. "
        "No attention soft-capping (query-key norms), final logits tanh-capped at 30. The last 18 layers share "
        "keys and values with earlier ones, so their own k/v projections and k norms in the checkpoint are unused.",
        **_WAVE_B, thinking_switch=True, thinking_on_markers=("<|think|>",), thinking_off_ending="<|turn>model\n",
        unexpected_keys_allowed=_GEMMA4_E4B_SHARED_KV, gpu_min_gb=24, hub_created="2026-03-02",
        attn_agreement_check=True),
    "qwen3.8-27b": _qwen_b(
        "qwen3.8-27b", "Qwen/Qwen3.8-27B", "Qwen3.8", "2026-08", 27.36,
        "As Qwen3.5, plus: with thinking on the template writes a reasoning-effort instruction into the system "
        "message (none with thinking off), and by default it writes an empty think block into every earlier "
        "assistant turn (the few-shot replies).", markers=("Reasoning effort",), gpu=80, created="2026-08-05",
        host_mem=128),
    "gemma4-31b-it": ModelSpec(
        "gemma4-31b-it", "google/gemma-4-31B-it", "gemma", "Gemma 4", "2026-03", 31.27, "native", None, {},
        "As Gemma 4 E4B, plus: with thinking off the generation prompt ends with an empty thought channel "
        "('<|channel>thought\\n<channel|>') that a rendered reply does not contain; the score is read after it.",
        **_WAVE_B, generation_suffix="<|channel>thought\n<channel|>", thinking_switch=True,
        thinking_on_markers=("<|think|>",), thinking_off_ending="<|turn>model\n<|channel>thought\n<channel|>",
        gpu_min_gb=80, hub_created="2026-03-11", host_mem_min_gb=128, attn_agreement_check=True),
}
WAVE_B_SHARE_TESTS = ("ministral3-8b-it", "qwen3.5-9b", "gemma4-e4b-it", "qwen3.8-27b", "gemma4-31b-it",
                      "qwen3.5-9b-base-plain")
# The planned comparisons of item share (first minus second), two tests each in wave B's primary family.
WAVE_B_PLANNED = (("gemma4-e4b-it", "gemma2-9b-it"), ("ministral3-8b-it", "mistral-7b-v0.3"),
                  ("qwen3.5-9b", "qwen3-8b"), ("qwen3.8-27b", "qwen3-32b"),
                  ("qwen3.5-9b-plain", "qwen3.5-9b-base-plain"))
WAVE_B_FAMILY_SIZE = len(WAVE_B_SHARE_TESTS) + 2 * len(WAVE_B_PLANNED)  # 16, fixed before data
PREDICTION_B = (
    "Logged before any wave B score (2026-11-22). B1: the goal effect (aligned minus swapped) is positive in both "
    "formats (95% game-clustered CI above zero) for every wave B run, and the item share is supported for every "
    "wave B model that passes the competence gate. B2: no generation pair reads 'differs' with the newer model "
    "lower. B3: in the plain format, instruct minus base reads 'differs' (or 'differs, within the margin') with a "
    "positive difference; untestable if the base model fails the competence gate. "
    "Amended after review 1, before any wave B score (notes/families_waveb_review1.md): B1's basis holds for the "
    "log-odds effect (every interpretable model scored so far has it), not for the item share: Qwen3-4B is competent "
    "yet its share is 0.49 with a yes/no effect of 2.27 nats, so a share that is not supported while the yes/no "
    "effect's CI is above zero reads 'effect carried by a minority of items (as Qwen3-4B)', and 'no detectable "
    "effect' only when both are null. B2 can fail only on a clear negative difference ('inconclusive' and 'matches' "
    "count as holding). B3: if the base model fails only the yes/no coverage criterion, B3 is read on the "
    "two-candidate item share (B3_FALLBACK).")
# M2 (review 1): the Qwen3 size series' item share and yes/no effect (size_score_qwen3-<size>, 4,666 failures each;
# recomputed 2026-11-22 from the archived scores). A competent model can carry a large mean effect on a minority of
# items (4B); the share moves non-monotonically with size within one line, so a generation pair, which also changes
# size, cannot isolate the generation.
QWEN3_SIZE_SERIES = {"qwen3-4b": {"share": 0.491, "yes_no_delta_L": 2.27},
                     "qwen3-8b": {"share": 0.887, "yes_no_delta_L": 2.80},
                     "qwen3-14b": {"share": 0.701, "yes_no_delta_L": 3.18},
                     "qwen3-32b": {"share": 0.715, "yes_no_delta_L": 1.55}}
B1_READINGS = {
    "supported": "effect present (share supported)",
    "minority": "effect carried by a minority of items (as Qwen3-4B)",
    "null": "no detectable effect",
    "reversed": "yes/no effect reversed (aligned below swapped)",
}
PAIR_CAVEAT = ("Generation pairs also change size, and within the Qwen3 line the item share runs 0.49 / 0.89 / 0.70 / "
               "0.71 across 4B / 8B / 14B / 32B, so a 'differs' here is not a generation effect: it describes two "
               "whole models.")
# B3's readout if the base model fails only the yes/no coverage criterion (median P(Yes) + P(No) under the neutral
# task below 0.5), fixed after review 1 and before any score: the paired difference in the two-candidate item share
# (failures with two-candidate L(aligned) > L(swapped)), instruct minus base, game-clustered, read by its 95% CI;
# the base model must reach the competence AUC (0.70) on the two-candidate score under the neutral task.
B3_FALLBACK = {"applies_when": "base model fails only the median yes/no mass criterion",
               "metric": "two-candidate item share, instruct minus base (paired, game-clustered 95% CI)",
               "base_two_candidate_auc_min": 0.70}
# Run order, pre-registered after review 1: the environment check first, then the four small models (Ministral 3,
# Qwen3.5 9B in both formats, Qwen3.5 9B base, Gemma 4 E4B with its placement check), then Qwen3.8 27B with its
# preserve_thinking check, then Gemma 4 31B last.
# Amended 2026-11-23 (section 8, after the environment check failed and its cause was found, before any wave B score):
# the older comparison models' serve2 rescorings run after the check and before the wave B models.
RUN_ORDER_B = ("family_envcheck_mistral-7b-v0.3", "family_envcheck_qwen3-8b", "families_envcheck",
               "family_rescore_mistral-7b-v0.3", "family_rescore_qwen3-8b", "family_rescore_gemma2-9b-it",
               "family_rescore_llama3.1-8b", "family_rescore_qwen3-32b",
               "family_score_ministral3-8b-it", "family_score_qwen3.5-9b", "family_score_qwen3.5-9b-plain",
               "family_score_qwen3.5-9b-base-plain", "family_score_gemma4-e4b-it", "family_placement_gemma4-e4b-it",
               "family_score_qwen3.8-27b", "family_nopreserve_qwen3.8-27b", "family_score_gemma4-31b-it")


def spec_for(tag: str) -> ModelSpec:
    """A scored model's spec: wave A, wave B, or an already-scored reference model."""
    for registry in (MODELS, WAVE_B, REFERENCE):
        if tag in registry:
            return registry[tag]
    raise KeyError(f"unknown model {tag!r}; known: {sorted(MODELS) + sorted(WAVE_B) + sorted(REFERENCE)}")


def environment_record() -> dict[str, Any]:
    """The software a shard runs in: versions, and whether packages that fetch or swap in kernels are importable."""
    import importlib.util
    import platform

    rec: dict[str, Any] = {"python": platform.python_version()}
    for name in ("transformers", "torch", "tokenizers", "huggingface_hub", "accelerate", "jinja2", "mistral_common"):
        try:
            rec[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absent or broken: recorded, not fatal here
            rec[name] = None
    for name in ("kernels", "fla", "causal_conv1d", "flash_attn"):
        rec[f"{name}_importable"] = importlib.util.find_spec(name) is not None
    return rec


def require_environment(spec: ModelSpec, env: dict[str, Any] | None = None, *, envcheck: bool = False) -> dict[str, Any]:
    """Refuse to score a model in the wrong environment (wave B, section 4).

    A serve2 model needs transformers 5 or later and no `kernels` package (it would fetch unpinned compiled kernels
    from the Hub at run time). A wave A or reference model is scored with transformers 4 only, except by the
    environment check, which rescores it in serve2 on purpose. Mock runs load nothing and are not checked.
    """
    env = environment_record() if env is None else env
    if os.environ.get("WF_MOCK_SCORE", "0") == "1":
        return {**env, "checked": "mock"}
    major = int(str(env.get("transformers") or "0").split(".")[0])
    if (spec.env == "serve2" or envcheck) and major < 5:
        raise SystemExit(f"{spec.tag} runs in serve2 (transformers >= 5); this Python has transformers {env.get('transformers')}")
    if spec.env == "serve" and not envcheck and major >= 5:
        raise SystemExit(f"{spec.tag} was scored with transformers 4; in serve2 it is rescored only by family_envcheck")
    if (spec.env == "serve2" or envcheck) and env.get("kernels_importable"):
        raise SystemExit("the kernels package is importable: transformers could fetch unpinned kernels from the Hub")
    return {**env, "checked": True}


def gpu_total_gb() -> float:
    """The memory of every GPU this process can see, summed, in GB (10^9 bytes); 0 without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9


def require_resources(spec: ModelSpec, environ: dict[str, str] | None = None,
                      gpu_gb: Callable[[], float] = gpu_total_gb) -> dict[str, Any]:
    """Refuse to load a model the job cannot hold (review 1, S6), before anything is loaded.

    The visible GPUs together must have at least `gpu_min_gb` GB (an A100-80GB reports about 85 GB, two L40S about
    95 GB, one L40S or A40 about 48 GB), and a model with `host_mem_min_gb` (the 27B and 31B models, which load
    through CPU memory) needs a Slurm job with at least that much memory (SLURM_MEM_PER_NODE, in MB). Mock runs load
    nothing and are not checked."""
    environ = dict(os.environ if environ is None else environ)
    if environ.get("WF_MOCK_SCORE", "0") == "1":
        return {"checked": "mock"}
    rec: dict[str, Any] = {"checked": True, "gpu_min_gb": spec.gpu_min_gb, "host_mem_min_gb": spec.host_mem_min_gb}
    if spec.gpu_min_gb:
        have = gpu_gb()
        rec["gpu_total_gb"] = round(have, 1)
        if have < spec.gpu_min_gb:
            raise SystemExit(f"{spec.tag} needs {spec.gpu_min_gb} GB of GPU memory; this job sees {have:.1f} GB")
    if spec.host_mem_min_gb:
        mem = environ.get("SLURM_MEM_PER_NODE", "")
        rec["slurm_mem_per_node_mb"] = mem
        if not mem.isdigit() or int(mem) < spec.host_mem_min_gb * 1024:
            raise SystemExit(f"{spec.tag} needs a job with --mem={spec.host_mem_min_gb}G or more; "
                             f"SLURM_MEM_PER_NODE is {mem or 'unset'} (MB)")
    return rec


def open_scorer(spec: ModelSpec, tag: str | None = None) -> LogoddsScorer:
    """The scorer with every setting of the spec (the families runs, the placement check, the environment check)."""
    return LogoddsScorer(spec.hf_id, model_tag=tag or spec.tag, attn_implementation=spec.attn_implementation,
                         template_kwargs=spec.template_kwargs, model_class=spec.model_class,
                         tokenizer_kwargs=spec.tokenizer_kwargs, prompt_format=spec.prompt_format,
                         strict_load=spec.strict_load, unexpected_keys_allowed=spec.unexpected_keys_allowed)

FOLD_SEPARATOR = "\n\n"
PREDICTION = ("Logged before any family score (2026-11-20): the goal effect (aligned minus swapped) is positive under "
              "both formats, two-candidate and yes/no, for every wave A model; read as the 95% game-clustered CI "
              "lying above zero.")
FORMATS = ("Amended 2026-11-20, before any wave A score was read: the yes/no format is the primary cross-family "
           "comparison, read through the item share moved toward Yes (third amendment: the single primary metric, "
           "per model and in the planned paired comparisons, one Holm-corrected family; a model must pass the "
           "competence gate); the goal-to-evidence ratio R, ΔL and the standardised d are secondary, and the "
           "two-candidate format is read within model only. Across families R mostly reflects how far each model "
           "pushes its log-odds on failures, not goal bias. The two-candidate score's validity was checked "
           "for Qwen3 only; Llama-3.1-8B's sign of L agreed with its written predictions on 65% of pilot items "
           "(notes/deviations.md). The logged prediction is unchanged.")
ARMS = sizes.PRED_ARMS + sizes.YN_ARMS


# ---------------- prompts ----------------


def fold_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The system message's text, a blank line, then the first user turn's text, as one user turn.

    For a template with no system role (Gemma 2). Deterministic and the same in every condition: only the first
    turn changes, and it holds the first prompt example, never the task line being varied.
    """
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError("expected a system message followed by a user turn")
    first = {"role": "user", "content": messages[0]["content"] + FOLD_SEPARATOR + messages[1]["content"]}
    return [first] + [dict(m) for m in messages[2:]]


def family_messages(spec: ModelSpec, t: dict, arm: str) -> list[dict[str, str]] | None:
    msgs = sizes.arm_messages(t, arm)
    if msgs is None:
        return None
    return fold_system(msgs) if spec.system == "fold" else msgs


# ---------------- tokenization gate (audit C9) ----------------


class TokenizationMismatch(RuntimeError):
    pass


def _ids(x: Any) -> list[int]:
    """Token ids from apply_chat_template or a tokenizer call, whatever container they come in."""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)


def _special_ids(tok: Any) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", []) or [])
    for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items():
        if getattr(t, "special", True):
            ids.add(int(i))
    return ids


def template_reply(scorer: LogoddsScorer, messages: list[dict], reply: str, suffix: str = "") -> list[int]:
    """The ids the chat template gives `reply` as the assistant's turn after `messages`, up to its end-of-turn token.

    Rendered by the scorer's own encoding (`_encode`, with its template settings), so the gate sees the prompt
    exactly as scoring does. The conversation with the reply must begin with exactly the prompt the template gives
    before a reply (generation prompt included); otherwise a candidate scored after that prompt is not the reply as
    the template writes it. A system message is left out of this check: Mistral's template writes it into the last
    user turn, which is no longer last once a reply follows, so the two renders would differ for that reason alone.
    Where the prompt ends and the reply begins does not depend on the system message.

    `suffix` (wave B): text the generation prompt ends with that a rendered reply does not contain (Gemma 4 31B's
    empty thought channel with thinking off). The prompt must end with exactly its tokens; they are stripped before
    the comparison, and the reply is read after the rest.
    """
    convo = [m for m in messages if m["role"] != "system"]
    prompt = _ids(scorer._encode(convo))
    if suffix:
        tail_ids = _ids(scorer._tok(suffix, add_special_tokens=False))
        if not tail_ids or prompt[-len(tail_ids):] != tail_ids:
            raise TokenizationMismatch(f"the generation prompt does not end with the declared suffix {suffix!r}")
        prompt = prompt[:-len(tail_ids)]
    full = _ids(scorer._encode(convo + [{"role": "assistant", "content": reply}], add_generation_prompt=False))
    if full[:len(prompt)] != prompt:
        raise TokenizationMismatch("the conversation with the reply does not begin with the prompt before it")
    tail = full[len(prompt):]
    special = _special_ids(scorer._tok)
    end = next((i for i, x in enumerate(tail) if x in special), len(tail))
    return tail[:end]


def check_candidate(scorer: LogoddsScorer, messages: list[dict], text: str, suffix: str = "") -> dict[str, Any]:
    """Does `text`, tokenized alone as the scorer tokenizes a candidate, equal the template's tokens for it as the
    reply?

    Also the verify_models check: the prompt's text and the candidate joined as plain text, tokenized together.
    That one is informative only: a template that puts a space before a reply (Mistral) fails it by design.
    """
    tok = scorer._tok
    reply = template_reply(scorer, messages, text, suffix)
    alone = _ids(scorer._encode_candidate(text))
    prompt = _ids(scorer._encode(messages))
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **scorer.template_settings())
    joined = _ids(tok(prompt_text + text, add_special_tokens=False))
    return {"alone_equals_template_reply": alone == reply, "joined_plain_text_equal": prompt + alone == joined,
            "alone": alone, "template_reply": reply, "prompt_tokens": len(prompt)}


YES_NO = ("Yes", "No")


def tokenization_checks(scorer: LogoddsScorer, spec: ModelSpec, primary: list[dict], successes: list[dict],
                        reference_encoder: Callable[[list[dict]], list[int]] | None = None) -> dict[str, Any]:
    """The gate, on every item of a shard, through the scorer's own encoding. Returns the record; its "ok" says
    whether scoring may start.

    Wave B (notes/families_waveb_design.md section 7): a plain-format spec gets plain_checks instead of the
    template checks; a spec with thinking or prompt settings also gets waveb_prompt_checks (`reference_encoder`:
    mistral-common's encoding, for the agreement check).

    Candidates: both texts of every item (primary items and successes), as the reply after the aligned prompt.
    Yes/No: the scored token of each word (tokenized alone, as score_one_token_pair does) must be one token and
    equal the template's rendering of that word as a real reply, after every yes/no prompt the shard scores
    (sizes.size_jobs: three per primary item, two per success).
    """
    if spec.prompt_format == "plain":
        return plain_checks(scorer, spec, primary, successes)
    tok = scorer._tok
    problems: list[str] = []
    first_pairs: Counter = Counter()
    joined_mismatch = 0
    max_prompt = 0
    n = 0
    for t in primary + successes:
        msgs = family_messages(spec, t, "aligned")
        ids = {}
        for name in ("o_plus", "o_minus"):
            try:
                r = check_candidate(scorer, msgs, t["candidates"][name], spec.generation_suffix)
            except TokenizationMismatch as e:  # recorded with the rest, so the record says what failed
                problems.append(f"{t['id']} {name}: {e}")
                ids[name] = []
                continue
            n += 1
            max_prompt = max(max_prompt, r["prompt_tokens"])
            joined_mismatch += not r["joined_plain_text_equal"]
            if not r["alone_equals_template_reply"]:
                problems.append(f"{t['id']} {name}: alone {r['alone'][:6]} vs template {r['template_reply'][:6]}")
            ids[name] = r["alone"]
        if not ids["o_plus"] or not ids["o_minus"] or ids["o_plus"][0] == ids["o_minus"][0]:
            problems.append(f"{t['id']}: the candidates share their first token")
        else:
            first_pairs[(tok.decode(ids["o_plus"][:1]), tok.decode(ids["o_minus"][:1]))] += 1
    scored = {word: _ids(scorer._encode_candidate(word)) for word in YES_NO}
    yes_no: dict[str, Any] = {word: {"alone": scored[word], "decoded": tok.decode(scored[word]), "template_reply": None,
                                     "n_checked": 0, "n_mismatched": 0} for word in YES_NO}
    for word in YES_NO:
        if len(scored[word]) != 1:
            problems.append(f"{word!r} is not one token: alone {scored[word]}")
    n_yes_no = 0
    for t, arm in sizes.size_jobs(primary, successes):  # every yes/no prompt a shard of these items scores
        if arm not in sizes.YN_ARMS:
            continue
        msgs = family_messages(spec, t, arm)
        n_yes_no += 1
        for word in YES_NO:
            rec = yes_no[word]
            try:
                reply = template_reply(scorer, msgs, word, spec.generation_suffix)
            except TokenizationMismatch as e:
                problems.append(f"{t['id']} {arm} {word!r}: {e}")
                continue
            rec["n_checked"] += 1
            if rec["template_reply"] is None:
                rec["template_reply"] = reply
            if reply != scored[word]:
                rec["n_mismatched"] += 1
                problems.append(f"{t['id']} {arm}: {word!r} is not the template's single token: alone "
                                f"{scored[word]}, template {reply}")
    if n_yes_no == 0:
        problems.append("no yes/no prompt checked")
    if scored["Yes"] == scored["No"]:
        problems.append("'Yes' and 'No' are the same token")
    rec = {"ok": not problems and n > 0, "n_candidates_checked": n, "problems": problems[:50],
           "n_problems": len(problems), "yes_no": yes_no, "n_yes_no_prompts_checked": n_yes_no,
           "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
           "joined_plain_text_mismatches": joined_mismatch, "max_prompt_tokens": max_prompt,
           "template_kwargs": scorer.template_settings(), "system_handling": spec.system}
    if spec.wave != "A":
        extra = waveb_prompt_checks(scorer, spec, primary, successes, reference_encoder)
        rec["waveb_prompt_checks"] = extra
        rec["problems"] = (rec["problems"] + extra["problems"])[:50]
        rec["n_problems"] += extra["n_problems"]
        rec["ok"] = rec["ok"] and extra["ok"]
    return rec


def _render(scorer: LogoddsScorer, messages: list[dict], **overrides: Any) -> str:
    """The prompt as text, with the scorer's template settings (and any overridden, e.g. thinking on)."""
    return scorer._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **{**scorer.template_settings(), **overrides})


def waveb_prompt_checks(scorer: LogoddsScorer, spec: ModelSpec, primary: list[dict], successes: list[dict],
                        reference_encoder: Callable[[list[dict]], list[int]] | None = None) -> dict[str, Any]:
    """Wave B's additions to the gate for a chat-format model (notes/families_waveb_design.md sections 5 and 7).

    On the first item, every arm, as text: thinking off renders differently from thinking on (the template reads
    the switch), the thinking-on render has the model's markers and ending, the thinking-off render has none of
    the markers, and no prompt holds forbidden text (a template's default system prompt). On every scored prompt of
    the shard (sizes.size_jobs), as token ids: it ends with the model's thinking-off ending, and, for Ministral 3,
    it equals mistral-common's encoding of the same messages.
    """
    tok = scorer._tok
    problems: list[str] = []
    jobs = sizes.size_jobs(primary, successes)
    first = primary[0] if primary else (successes[0] if successes else None)
    n_text = 0
    if first is not None:
        for arm in sizes.PRED_ARMS + sizes.YN_ARMS:
            msgs = family_messages(spec, first, arm)
            if msgs is None:
                continue
            n_text += 1
            off = _render(scorer, msgs)
            for bad in spec.forbidden_in_prompt:
                if bad in off:
                    problems.append(f"{first['id']} {arm}: the prompt contains {bad!r}")
            if spec.forbidden_before_ending:
                if not off.endswith(spec.thinking_off_ending):
                    problems.append(f"{first['id']} {arm}: the prompt does not end with {spec.thinking_off_ending!r}")
                body = off[:len(off) - len(spec.thinking_off_ending)]
                for bad in spec.forbidden_before_ending:
                    if bad in body:
                        problems.append(f"{first['id']} {arm}: {bad!r} before the thinking-off ending")
            if not spec.thinking_switch:
                continue
            on = _render(scorer, msgs, enable_thinking=True)
            if on == off:
                problems.append(f"{first['id']} {arm}: thinking on renders like thinking off (switch not read)")
            if spec.thinking_on_ending and not on.endswith(spec.thinking_on_ending):
                problems.append(f"{first['id']} {arm}: thinking on does not end with {spec.thinking_on_ending!r}")
            for marker in spec.thinking_on_markers:
                if marker not in on:
                    problems.append(f"{first['id']} {arm}: thinking on lacks {marker!r}")
                if marker in off:
                    problems.append(f"{first['id']} {arm}: thinking off contains {marker!r}")
    ending = _ids(tok(spec.thinking_off_ending, add_special_tokens=False)) if spec.thinking_off_ending else []
    n_ending = n_reference = 0
    for t, arm in jobs:
        msgs = family_messages(spec, t, arm)
        ids = _ids(scorer._encode(msgs))
        if ending:
            n_ending += 1
            if ids[-len(ending):] != ending:
                problems.append(f"{t['id']} {arm}: the prompt does not end with {spec.thinking_off_ending!r}")
        if spec.mistral_common_check:
            if reference_encoder is None:
                problems.append("mistral-common agreement required but no reference encoder given")
                break
            n_reference += 1
            if list(reference_encoder(msgs)) != ids:
                problems.append(f"{t['id']} {arm}: the template's ids differ from mistral-common's")
    if spec.thinking_off_ending and not n_ending:
        problems.append("no prompt checked for the thinking-off ending")
    return {"ok": not problems, "problems": problems[:50], "n_problems": len(problems),
            "n_first_item_renders": n_text, "n_prompts_checked_ending": n_ending,
            "forbidden_before_ending": list(spec.forbidden_before_ending),
            "thinking_off_ending": spec.thinking_off_ending, "thinking_off_ending_ids": ending,
            "n_prompts_checked_mistral_common": n_reference, "generation_suffix": spec.generation_suffix}


def plain_checks(scorer: LogoddsScorer, spec: ModelSpec, primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """The gate for the plain format (wave B, section 6): no template, so instead of the template's reply, every
    candidate (after its aligned prompt) and ' Yes' and ' No' (after every yes/no prompt) must tokenize together
    with the prompt exactly as the prompt's tokens followed by the candidate's (no merge across the boundary);
    ' Yes' and ' No' are one token each and differ; the two candidates' first tokens differ."""
    tok = scorer._tok
    problems: list[str] = []
    first_pairs: Counter = Counter()
    max_prompt = n = 0

    def joint(msgs: list[dict], text: str) -> tuple[bool, list[int], int]:
        prompt = _ids(scorer._encode(msgs))
        alone = _ids(scorer._encode_candidate(text))
        together = _ids(tok(plain_text_prompt(msgs) + scorer.candidate_text(text), add_special_tokens=True))
        return together == prompt + alone, alone, len(prompt)

    for t in primary + successes:
        msgs = family_messages(spec, t, "aligned")
        ids = {}
        for name in ("o_plus", "o_minus"):
            ok, alone, n_prompt = joint(msgs, t["candidates"][name])
            n += 1
            max_prompt = max(max_prompt, n_prompt + len(alone))
            if not ok:
                problems.append(f"{t['id']} {name}: prompt and candidate merge across the boundary")
            ids[name] = alone
        if not ids["o_plus"] or not ids["o_minus"] or ids["o_plus"][0] == ids["o_minus"][0]:
            problems.append(f"{t['id']}: the candidates share their first token")
        else:
            first_pairs[(tok.decode(ids["o_plus"][:1]), tok.decode(ids["o_minus"][:1]))] += 1
    scored = {word: _ids(scorer._encode_candidate(word)) for word in YES_NO}
    yes_no: dict[str, Any] = {word: {"alone": scored[word], "decoded": tok.decode(scored[word]), "n_checked": 0,
                                     "n_mismatched": 0} for word in YES_NO}
    for word in YES_NO:
        if len(scored[word]) != 1:
            problems.append(f"{scorer.candidate_text(word)!r} is not one token: {scored[word]}")
    if scored["Yes"] == scored["No"]:
        problems.append("' Yes' and ' No' are the same token")
    n_yes_no = 0
    for t, arm in sizes.size_jobs(primary, successes):
        if arm not in sizes.YN_ARMS:
            continue
        msgs = family_messages(spec, t, arm)
        n_yes_no += 1
        for word in YES_NO:
            ok, _, n_prompt = joint(msgs, word)
            max_prompt = max(max_prompt, n_prompt + 1)
            yes_no[word]["n_checked"] += 1
            if not ok:
                yes_no[word]["n_mismatched"] += 1
                problems.append(f"{t['id']} {arm}: prompt and {scorer.candidate_text(word)!r} merge across the boundary")
    if n_yes_no == 0:
        problems.append("no yes/no prompt checked")
    return {"ok": not problems and n > 0, "n_candidates_checked": n, "problems": problems[:50],
            "n_problems": len(problems), "yes_no": yes_no, "n_yes_no_prompts_checked": n_yes_no,
            "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
            "joined_plain_text_mismatches": 0, "max_prompt_tokens": max_prompt, "prompt_format": "plain",
            "candidate_prefix": scorer.candidate_text(""), "template_kwargs": {}, "system_handling": spec.system}


def mistral_common_encoder(spec: ModelSpec, **load: Any) -> Callable[[list[dict]], list[int]]:
    """mistral-common's own encoding of a request (transformers 5's MistralCommonBackend at the pinned revision), for
    the agreement check on Ministral 3 (wave B, section 7). `load`: cache_dir, local_files_only."""
    from transformers import AutoTokenizer

    ref = AutoTokenizer.from_pretrained(spec.hf_id, revision=C.REVISIONS[spec.hf_id], mistral_format=True, **load)
    return lambda msgs: _ids(ref.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))


def _gate(spec: ModelSpec, env: dict[str, Any] | None = None) -> Callable[[Path, LogoddsScorer, list[dict], list[dict]], None]:
    def gate(d: Path, scorer: LogoddsScorer, primary: list[dict], successes: list[dict]) -> None:
        if scorer.mock:
            write_json(d / "tokenization_checks.json", {"ok": None, "skipped": "mock",
                                                         **({"environment": env} if env else {})})
            return
        ref = mistral_common_encoder(spec) if spec.mistral_common_check else None
        rec = tokenization_checks(scorer, spec, primary, successes, ref)
        rec["attn_implementation_loaded"] = getattr(scorer._model.config, "_attn_implementation", None)
        rec["dtype_loaded"] = str(getattr(scorer._model, "dtype", ""))
        if spec.wave != "A":
            rec["environment"] = env
            rec["loading"] = scorer.load_info
            rec["device_map"] = {k: str(v) for k, v in (getattr(scorer._model, "hf_device_map", None) or {}).items()}
            rec["model_class_loaded"] = type(scorer._model).__name__
            # review 1, M3: node-local weights were hashed against the committed manifest before loading
            weights = weights_check(spec)
            if weights is not None:
                write_json(d / "weights_check.json", weights)
                rec["weights_check"] = {k: weights.get(k) for k in ("ok", "hf_id", "revision", "n_files", "total_bytes",
                                                                    "manifest", "record")}
            # review 1, S9: a few items under eager and SDPA attention, once the encoding has passed
            if rec["ok"] and spec.attn_agreement_check:
                rec["attention_agreement"] = attention_agreement(scorer, spec, primary)
                rec["ok"] = rec["attention_agreement"]["ok"]
                if not rec["ok"]:
                    rec["problems"] = rec["problems"] + rec["attention_agreement"]["problems"]
                    rec["n_problems"] += len(rec["attention_agreement"]["problems"])
        write_json(d / "tokenization_checks.json", rec)
        if not rec["ok"]:
            raise TokenizationMismatch(f"{spec.tag}: tokenization gate failed ({rec['n_problems']} problems, first: "
                                       f"{rec['problems'][:3]}); see {d / 'tokenization_checks.json'}. Nothing scored.")
    return gate


# ---------------- wave B: node-local weights against the manifest (review 1, M3) ----------------

WEIGHTS_RECORD_ENV = "WF_WEIGHTS_RECORD"  # the prefetch tool's verification record (pace/family_score_serve2.sbatch)


class WeightsCheckFailed(RuntimeError):
    pass


def weights_check(spec: ModelSpec, environ: dict[str, str] | None = None) -> dict[str, Any] | None:
    """With node-local weights (WF_WEIGHTS=local), the prefetch tool's record of this model's files hashed against the
    committed manifest (wf/tools/prefetch_models.py, verify_snapshot): it must exist, be for this model's id and
    pinned revision, and have passed; otherwise nothing is scored. None with the shared cache (route B, not used)."""
    environ = os.environ if environ is None else environ
    if environ.get("WF_WEIGHTS", "shared") != "local":
        return None
    path = environ.get(WEIGHTS_RECORD_ENV, "")
    if not path or not Path(path).is_file():
        raise WeightsCheckFailed(f"{spec.tag}: node-local weights but no verification record ({WEIGHTS_RECORD_ENV}={path!r})")
    rec = json.loads(Path(path).read_text(encoding="utf-8"))  # the prefetch tool's --out: {tag: result}
    for entry in rec.values():
        if isinstance(entry, dict) and entry.get("hf_id") == spec.hf_id and entry.get("revision") == spec.revision:
            verify = entry.get("verify") or {}
            if entry.get("status") != "OK" or not verify.get("ok"):
                raise WeightsCheckFailed(f"{spec.tag}: the downloaded files do not match the manifest: "
                                         f"{(verify.get('problems') or [entry.get('status')])[:5]}")
            return {**verify, "hf_id": spec.hf_id, "revision": spec.revision, "record": str(path)}
    raise WeightsCheckFailed(f"{spec.tag}: {path} holds no verified download of {spec.hf_id}@{spec.revision}")


# ---------------- wave B: eager against SDPA attention on a few items (review 1, S9) ----------------

# Fixed after review 1, before any wave B score. A real attention bug (a dropped soft-cap, a wrong sliding-window mask)
# moves scores by nats; bf16 rounding between the two kernels moves them by hundredths to tenths (the environment
# check's thresholds per score are 0.07 nats yes/no and 0.35 two-candidate).
ATTN_AGREEMENT = {"n_items": 5, "arms": ("aligned", "yn_aligned"), "yes_no_mean_abs_max": 0.10,
                  "two_candidate_mean_abs_max": 0.50, "any_abs_max": 1.0}


def _text_config_key(model: Any) -> str | None:
    """The name of the model's text sub-config (None if the config is the text config itself)."""
    cfg = model.config
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    if text is cfg:
        return None
    return next((k for k in getattr(cfg, "sub_configs", {}) or {} if getattr(cfg, k, None) is text), "text_config")


def _attn_loaded(model: Any) -> list[str | None]:
    """The attention implementation the text model runs with (its text sub-config's, or the config's own)."""
    key = _text_config_key(model)
    cfg = model.config if key is None else getattr(model.config, key)
    return [getattr(cfg, "_attn_implementation", None)]


def _set_text_attn(model: Any, impl: str) -> None:
    """Switch the text model's attention only: Gemma 4 E4B's audio encoder has no SDPA, so a switch of the whole model
    back to SDPA would raise (found on the real class in serve2, review 1)."""
    key = _text_config_key(model)
    model.set_attn_implementation(impl if key is None else {key: impl})


def attention_agreement(scorer: LogoddsScorer, spec: ModelSpec, primary: list[dict]) -> dict[str, Any]:
    """Score the shard's first ATTN_AGREEMENT["n_items"] items (aligned task, both formats) with the loaded attention
    (SDPA, transformers' default), again with eager attention, then switch back; the two must agree within
    ATTN_AGREEMENT. Each switch is confirmed on the config (a model that cannot switch would compare SDPA with
    itself). The scorer's cache is cleared around each pass, so every score is computed afresh."""
    model = scorer._model
    items = primary[:ATTN_AGREEMENT["n_items"]]
    loaded = _attn_loaded(model)
    problems: list[str] = []

    def score_all() -> dict[tuple[str, str], float]:
        scorer._cache.clear()
        out = {}
        for t in items:
            for arm in ATTN_AGREEMENT["arms"]:
                msgs = family_messages(spec, t, arm)
                if arm in sizes.YN_ARMS:
                    r = scorer.score_one_token_pair(t["id"], arm, msgs, "Yes", "No")
                else:
                    r = scorer.score_pair(t["id"], arm, msgs, t["candidates"]["o_plus"], t["candidates"]["o_minus"])
                out[(t["id"], arm)] = r.L
        scorer._cache.clear()
        return out

    if set(loaded) != {"sdpa"}:
        problems.append(f"loaded attention is {loaded}, not sdpa")
        return {"ok": False, "problems": problems, "loaded": loaded, "rule": ATTN_AGREEMENT}
    sdpa = score_all()
    _set_text_attn(model, "eager")
    switched = _attn_loaded(model)
    eager = score_all() if set(switched) == {"eager"} else {}
    _set_text_attn(model, "sdpa")
    restored = _attn_loaded(model)
    if set(switched) != {"eager"}:
        problems.append(f"could not switch to eager attention (config says {switched})")
    if set(restored) != {"sdpa"}:
        problems.append(f"could not switch back to sdpa (config says {restored})")
    diffs = {k: eager[k] - sdpa[k] for k in sdpa if k in eager}
    by_fmt = {"yes_no": [abs(v) for (i, a), v in diffs.items() if a in sizes.YN_ARMS],
              "two_candidate": [abs(v) for (i, a), v in diffs.items() if a not in sizes.YN_ARMS]}
    out: dict[str, Any] = {"n_items": len(items), "arms": list(ATTN_AGREEMENT["arms"]), "rule": ATTN_AGREEMENT,
                           "loaded": loaded, "switched": switched, "restored": restored,
                           "scores": [{"transition_id": i, "condition": a, "L_sdpa": sdpa[(i, a)], "L_eager": eager.get((i, a))}
                                      for i, a in sdpa]}
    for fmt, v in by_fmt.items():
        mean = sum(v) / len(v) if v else float("nan")
        out[f"{fmt}_mean_abs_diff"] = mean
        if not mean <= ATTN_AGREEMENT[f"{fmt}_mean_abs_max"]:  # NaN (nothing compared) fails too
            problems.append(f"eager and sdpa differ by {mean:.3f} nats on average ({fmt})")
    worst = max((abs(v) for v in diffs.values()), default=float("nan"))
    out["max_abs_diff"] = worst
    if not worst <= ATTN_AGREEMENT["any_abs_max"]:
        problems.append(f"eager and sdpa differ by {worst:.3f} nats on one score")
    if len(diffs) != len(items) * len(ATTN_AGREEMENT["arms"]) or not items:
        problems.append(f"{len(diffs)} scores compared, {len(items) * len(ATTN_AGREEMENT['arms'])} expected")
    return {**out, "ok": not problems, "problems": problems}


# ---------------- scoring ----------------


def run_family_score(cfg: dict) -> int:
    tag = cfg["model"]
    if tag not in MODELS and tag not in WAVE_B:
        raise SystemExit(f"unknown model {tag!r}; known: {sorted(MODELS) + sorted(WAVE_B)}")
    spec = spec_for(tag)
    env = require_environment(spec)
    env["resources"] = require_resources(spec)
    cfg = {**cfg, "model_spec": spec.record()}
    return sizes.score_shard(cfg, tag, lambda: open_scorer(spec), lambda t, arm: family_messages(spec, t, arm),
                             gate=_gate(spec, env if spec.wave != "A" else None))


# ---------------- Mistral's system-text placement (exploratory check, logged 2026-11-20 before any score was read) ----

PLACEMENT_ARMS = ("aligned", "swapped", "yn_aligned", "yn_swapped")
PLACEMENT_ITEMS = 300
FIRST_TURN = "-firstturn"  # suffix of the model tag the first-turn scores are saved under


def placement_items(primary: list[dict], k: int = PLACEMENT_ITEMS) -> list[dict]:
    """A seeded sample of `k` primary items, in id order; the same for any order the items arrive in."""
    pool = sorted(primary, key=lambda t: t["id"])
    picked = random.Random(f"{C.RNG_SEED}|family_placement").sample(pool, min(k, len(pool)))
    return sorted(picked, key=lambda t: t["id"])


def first_turn_spec(spec: ModelSpec) -> ModelSpec:
    """`spec` with the system text folded into the first user turn (Gemma's handling), under its own tag."""
    if spec.system != "native":
        raise ValueError(f"{spec.tag} already folds its system text")
    # Wave A's note (Mistral 7B, whose template puts the system text in the last user turn) is kept word for word, as
    # its placement run saved it; review 1 (S10): "not the last" is wrong for a template with a system turn (Gemma 4).
    note = ("Placement check: the system text opens the first user turn, not the last." if spec.wave == "A" else
            "Placement check: the system text opens the first user turn instead of the template's own system turn.")
    return replace(spec, tag=spec.tag + FIRST_TURN, system="fold", template_notes=note)


# ---------------- wave B: Qwen3.8's empty think blocks in the examples (review 1, S8; before any wave B score) ------

NO_PRESERVE = "-nopreserve"  # suffix of the model tag the preserve_thinking=False scores are saved under
NO_PRESERVE_MODELS = ("qwen3.8-27b",)


def nopreserve_spec(spec: ModelSpec) -> ModelSpec:
    """`spec` with preserve_thinking=False, so its template writes no empty think block into the few-shot replies (as
    Qwen3.5's and Qwen3's templates write none), under its own tag. The gate then requires that no prompt holds
    '<think>' before its thinking-off ending."""
    if spec.tag not in NO_PRESERVE_MODELS:
        raise ValueError(f"the preserve_thinking check is for {NO_PRESERVE_MODELS}, not {spec.tag}")
    return replace(spec, tag=spec.tag + NO_PRESERVE, template_kwargs={**spec.template_kwargs, "preserve_thinking": False},
                   forbidden_before_ending=("<think>",),
                   template_notes="preserve_thinking=False: no empty think block in the few-shot replies (review 1, S8).")


def run_family_nopreserve(cfg: dict) -> int:
    """`family_nopreserve_<tag>` (review 1, S8): Qwen3.8 27B with preserve_thinking=False on PLACEMENT_ITEMS seeded
    primary items under the aligned and swapped tasks in both formats (about 1,800 passes), saved under
    `<tag>-nopreserve`; read against the model's own run by the one-fifth rule (`one_fifth_rule`)."""
    tag = cfg["model"]
    if tag not in NO_PRESERVE_MODELS:
        raise SystemExit(f"the preserve_thinking check is for {NO_PRESERVE_MODELS}, not {tag!r}")
    spec = nopreserve_spec(spec_for(tag))
    env = require_environment(spec)
    env["resources"] = require_resources(spec)
    k = int(cfg.get("n_items", PLACEMENT_ITEMS))
    cfg = {**cfg, "model_spec": spec.record(), "n_items": k, "arms": list(PLACEMENT_ARMS)}
    return sizes.score_shard(cfg, spec.tag, lambda: open_scorer(spec), lambda t, arm: family_messages(spec, t, arm),
                             gate=_gate(spec, env), select=lambda p, s: (placement_items(p, k), []), arms=PLACEMENT_ARMS)


def variant_spec(tag: str) -> ModelSpec:
    """A scored model's spec, or one of its checks' variants: `<tag>-firstturn` (placement) or `<tag>-nopreserve`."""
    if tag.endswith(FIRST_TURN):
        return first_turn_spec(spec_for(tag[:-len(FIRST_TURN)]))
    if tag.endswith(NO_PRESERVE):
        return nopreserve_spec(spec_for(tag[:-len(NO_PRESERVE)]))
    return spec_for(tag)


def one_fifth_rule(block: dict[str, Any], fmt: str) -> dict[str, Any]:
    """Wave A's reading of a format check (placement_block's result, one format): the check moves the effect when the
    95% CI of the per-item difference (variant minus native) excludes zero and the difference is larger than a fifth
    of the native goal effect; then the comparisons that use the model are reported as format-dependent, with both
    numbers."""
    diff = block[fmt]["first_turn_minus_native"]
    lo, hi = diff["ci"]
    native = block[fmt]["native"]["delta_L_nats"]
    est = diff["delta_L_nats"]
    excludes = bool(lo == lo and hi == hi and (lo > 0 or hi < 0))
    large = bool(est == est and native == native and abs(est) > abs(native) / 5)
    return {"difference_nats": est, "ci": [lo, hi], "native_nats": native, "ci_excludes_zero": excludes,
            "above_one_fifth": large, "format_dependent": excludes and large}


def run_family_placement(cfg: dict) -> int:
    """`family_placement_<tag>`: the native model, its system text folded into the first user turn, on
    PLACEMENT_ITEMS seeded primary items under the aligned and swapped tasks in both formats. Scores are saved
    under the tag `<tag>-firstturn`, so they never mix with the model's own run."""
    tag = cfg["model"]
    if tag not in MODELS and tag not in WAVE_B:
        raise SystemExit(f"unknown model {tag!r}; known: {sorted(MODELS) + sorted(WAVE_B)}")
    spec = first_turn_spec(spec_for(tag))
    env = require_environment(spec)
    env["resources"] = require_resources(spec)
    k = int(cfg.get("n_items", PLACEMENT_ITEMS))
    cfg = {**cfg, "model_spec": spec.record(), "n_items": k, "arms": list(PLACEMENT_ARMS)}
    return sizes.score_shard(cfg, spec.tag, lambda: open_scorer(spec), lambda t, arm: family_messages(spec, t, arm),
                             gate=_gate(spec, env if spec.wave != "A" else None),
                             select=lambda p, s: (placement_items(p, k), []), arms=PLACEMENT_ARMS)


# ---------------- wave B: environment-equivalence check (notes/families_waveb_design.md section 8) ----------------

ENVCHECK_MODELS = ("mistral-7b-v0.3", "qwen3-8b")
ENVCHECK_SUFFIX = "-serve2"  # the tag the serve2 rescores are saved under, so they never mix with the earlier ones
ENVCHECK_ARMS = sizes.PRED_ARMS + sizes.YN_ARMS
# Pass rule, fixed 2026-11-22 before any wave B score: about twice the cross-GPU differences of the archived Qwen3-8B
# runs per score, and a third of the 0.3-nat noise floor for effects.
ENVCHECK_RULE = {"yes_no_mean_abs_max": 0.07, "yes_no_goal_effect_diff_max": 0.10, "yes_no_share_diff_max": 0.02,
                 "two_candidate_mean_abs_max": 0.35, "two_candidate_goal_effect_diff_max": 0.10}
# Review 1 (S4): each rescoring runs on the GPU type of the model's earlier run, so only the software differs; the
# shard's recorded GPU (its jobs.jsonl, wf/provenance.py) must be exactly this. Qwen3-8B's size_score shards ran on
# L40S (jobs 7174094-7174099, per their logs); Mistral 7B's wave A run on an A40 (families design note).
ENVCHECK_GPU = {"mistral-7b-v0.3": "NVIDIA A40", "qwen3-8b": "NVIDIA L40S"}


def envcheck_gpu_check(tag: str, folders: list[Path]) -> dict[str, Any]:
    """Every job of every shard of `tag`'s rescoring ran on ENVCHECK_GPU[tag] alone (a shard with no recorded GPU,
    another type, or two types fails)."""
    from wf.provenance import run_record

    want = ENVCHECK_GPU[tag]
    seen = {p.name: run_record(p)["gpu"] for p in folders}
    ok = bool(folders) and all(g == [want] for g in seen.values())
    return {"expected": want, "recorded": seen, "ok": ok}


def envcheck_spec(tag: str) -> ModelSpec:
    """An earlier model as serve2 loads it: wave B's tokenizer settings and loading check, its own template and
    class, scores saved under `<tag>-serve2`."""
    spec = spec_for(tag)
    if spec.env != "serve":
        raise ValueError(f"{tag} is not an earlier model")
    return replace(spec, tag=tag + ENVCHECK_SUFFIX, wave="envcheck", env="serve2",
                   tokenizer_kwargs={"mistral_format": False}, strict_load=True,
                   template_notes=f"Environment check: {tag} rescored in serve2.")


def run_family_envcheck(cfg: dict) -> int:
    """`family_envcheck_<tag>`: an earlier model rescored in serve2 on the placement check's 300 seeded primary items,
    under all three tasks in both formats, after the same gate."""
    tag = cfg["model"]
    if tag not in ENVCHECK_MODELS:
        raise SystemExit(f"the environment check rescores {ENVCHECK_MODELS}, not {tag!r}")
    spec = envcheck_spec(tag)
    env = require_environment(spec, envcheck=True)
    env["resources"] = require_resources(spec)
    k = int(cfg.get("n_items", PLACEMENT_ITEMS))
    cfg = {**cfg, "model_spec": spec.record(), "n_items": k, "arms": list(ENVCHECK_ARMS)}
    return sizes.score_shard(cfg, spec.tag, lambda: open_scorer(spec), lambda t, arm: family_messages(spec, t, arm),
                             gate=_gate(spec, env), select=lambda p, s: (placement_items(p, k), []), arms=ENVCHECK_ARMS)


def _pairs_by_key(old: list[dict], new: list[dict]) -> dict[tuple[str, str], tuple[float, float]]:
    a = {(x["transition_id"], x["condition"]): x["L"] for x in old}
    b = {(x["transition_id"], x["condition"]): x["L"] for x in new}
    return {k: (a[k], b[k]) for k in sorted(set(a) & set(b))}


def envcheck_block(old: list[dict], new: list[dict], items: list[dict]) -> dict[str, Any]:
    """One model's earlier scores against its serve2 rescores on the same (item, condition) pairs, with the verdict
    of the pass rule (ENVCHECK_RULE)."""
    ids = {t["id"] for t in items}
    old = [x for x in old if x["transition_id"] in ids and x["condition"] in ENVCHECK_ARMS]
    new = [x for x in new if x["transition_id"] in ids and x["condition"] in ENVCHECK_ARMS]
    expected = {(t["id"], a) for t in items for a in ENVCHECK_ARMS if sizes.arm_messages(t, a) is not None}
    pairs = _pairs_by_key(old, new)
    out: dict[str, Any] = {"n_expected": len(expected), "n_paired": len(pairs),
                           "missing": sorted(f"{i}/{c}" for i, c in expected - set(pairs))[:20]}
    checks: dict[str, bool] = {"complete": expected <= set(pairs)}
    for fmt, arms, (al, sw) in (("yes_no", sizes.YN_ARMS, ("yn_aligned", "yn_swapped")),
                                ("two_candidate", sizes.PRED_ARMS, ("aligned", "swapped"))):
        d = [b - a for (i, c), (a, b) in pairs.items() if c in arms]
        by_cond = {c: [b - a for (i, cc), (a, b) in pairs.items() if cc == c] for c in arms}
        goal = {}
        for which, idx in (("old", 0), ("new", 1)):
            g = [pairs[(t["id"], al)][idx] - pairs[(t["id"], sw)][idx] for t in items
                 if (t["id"], al) in pairs and (t["id"], sw) in pairs]
            goal[which] = sum(g) / len(g) if g else float("nan")
            if fmt == "yes_no":
                goal[f"share_{which}"] = sum(x > 0 for x in g) / len(g) if g else float("nan")
        rec = {"n": len(d), "mean_abs_diff": sum(abs(x) for x in d) / len(d) if d else float("nan"),
               "max_abs_diff": max((abs(x) for x in d), default=float("nan")),
               "share_identical": sum(x == 0 for x in d) / len(d) if d else float("nan"),
               "mean_diff_by_condition": {c: (sum(v) / len(v) if v else float("nan")) for c, v in by_cond.items()},
               "goal_effect_old": goal["old"], "goal_effect_new": goal["new"],
               "goal_effect_diff": goal["new"] - goal["old"]}
        checks[f"{fmt}_mean_abs"] = rec["mean_abs_diff"] <= ENVCHECK_RULE[f"{fmt}_mean_abs_max"]
        checks[f"{fmt}_goal_effect"] = abs(rec["goal_effect_diff"]) <= ENVCHECK_RULE[f"{fmt}_goal_effect_diff_max"]
        if fmt == "yes_no":
            rec["share_old"], rec["share_new"] = goal["share_old"], goal["share_new"]
            rec["share_diff"] = goal["share_new"] - goal["share_old"]
            checks["yes_no_share"] = abs(rec["share_diff"]) <= ENVCHECK_RULE["yes_no_share_diff_max"]
        out[fmt] = rec
    out["checks"] = checks
    out["pass"] = all(checks.values())  # a NaN fails every comparison, so missing scores fail the check
    return out


def run_families_envcheck(cfg: dict) -> int:
    """`families_envcheck` (CPU): each environment-check model's serve2 rescores against its earlier scores, plus
    part 1's token comparison (`tokens`: {model: path of the compare JSON from wf.tools.envcheck_tokens}); the
    overall verdict "pass" is what families_waveb_analysis reads."""
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    sources = tuple(str(_resolve(s)) for s in cfg.get("sources", SOURCES))
    sets = rq1x.rq1x_sets(load_items(sources))
    items = placement_items(sets["primary"], int(cfg.get("n_items", PLACEMENT_ITEMS)))
    report: dict[str, Any] = {"rule": ENVCHECK_RULE, "gpu_required": ENVCHECK_GPU, "n_items": len(items), "models": {},
                              "tokens": {}, "waiting": {}}
    read: dict[str, list[Path]] = {}
    for tag, mc in cfg["models"].items():
        folders, problems = complete_shards(mc["new_run"])
        if problems:
            report["waiting"][tag] = problems
            continue
        read[mc["new_run"]] = folders
        new = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag + ENVCHECK_SUFFIX]
        old, got, info = reference_scores(mc["old"])
        if info["problems"]:
            report["waiting"][f"{tag} (earlier)"] = info["problems"]
            continue
        read.update({f"{tag}:{run}": f for run, f in got.items()})
        block = {**envcheck_block(old, new, items), "gate": _gate_records(folders)}
        block["gpu"] = envcheck_gpu_check(tag, folders)
        block["checks"]["gpu_type"] = block["gpu"]["ok"]
        block["pass"] = all(block["checks"].values())
        report["models"][tag] = block
    for tag, path in (cfg.get("tokens") or {}).items():
        f = _resolve(path)
        if not f.exists():
            report["waiting"][f"{tag} tokens"] = [f"{f} missing"]
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        report["tokens"][tag] = {k: rec.get(k) for k in ("ok", "n_compared", "n_different", "n_only_one_side", "examples",
                                                          "left", "right")}
    want = set(cfg["models"])
    report["pass"] = bool(not report["waiting"] and set(report["models"]) == want and set(report["tokens"]) >= want
                          and all(m["pass"] for m in report["models"].values())
                          and all(t.get("ok") for t in report["tokens"].values()))
    from wf.provenance import provenance, shard_inputs

    report["provenance"] = provenance(
        runs={**{s: Path(s) for s in sources}, **{k: (v[0].parent if v else Path(k)) for k, v in read.items()}},
        inputs=[Path(s) / "transitions.jsonl" for s in sources] + shard_inputs([p for fs in read.values() for p in fs]))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Model families, wave B: environment-equivalence check", report)
    lines = [f"{tag}: {'PASS' if m['pass'] else 'FAIL'} {m['checks']}" for tag, m in report["models"].items()]
    lines += [f"{tag} tokens: {'identical' if t.get('ok') else 'DIFFER'}" for tag, t in report["tokens"].items()]
    write_status(run_id, "\n".join(lines) + f"\noverall: {'PASS' if report['pass'] else 'NOT PASSED'}"
                 + (f"\nwaiting: {sorted(report['waiting'])}" if report["waiting"] else ""), ok=False,
                 reported="environment check passed" if report["pass"] else "environment check not passed")
    return 0


# ---------------- wave B: every older comparison model rescored in serve2 (design section 8, amendment of 2026-11-23,
# after the environment check failed and its cause was found, before any wave B score existed) ----------------

# Every older model any wave B comparison uses (the reference models of families_waveb_analysis: the planned pairs'
# older members and the two descriptive references), rescored in serve2 on every item and condition it is compared on.
RESCORE_MODELS = ("mistral-7b-v0.3", "qwen3-8b", "gemma2-9b-it", "llama3.1-8b", "qwen3-32b")
RESCORE_SUFFIX = ENVCHECK_SUFFIX  # scores saved under `<tag>-serve2`, in their own runs (family_rescore_<tag>)
# One GPU type per model, for every shard (checked by the job before loading and by the analysis): the type of the
# model's earlier run where it had one (Mistral 7B A40; Qwen3-8B and Gemma 2 9B L40S; Qwen3-32B the A100-80GB),
# L40S for Llama-3.1-8B (its earlier runs were on mixed types).
RESCORE_GPU = {"mistral-7b-v0.3": "NVIDIA A40", "qwen3-8b": "NVIDIA L40S", "gemma2-9b-it": "NVIDIA L40S",
               "llama3.1-8b": "NVIDIA L40S", "qwen3-32b": "NVIDIA A100 80GB PCIe"}
# Route A (section 13, node-local download checked against the committed manifest): the models not in the shared
# cache. Qwen3-32B's weights are in no ICE cache any more.
RESCORE_NODE_LOCAL = ("qwen3-32b",)
RESCORE_AMENDMENT = (
    "Amended 2026-11-23, after the environment check failed and its cause was found, before any wave B score existed "
    "(design section 8): the check stays FAILED (Qwen3-8B's yes/no per-score difference 0.248 nats; a float64 "
    "diagnostic showed a hidden-state perturbation of about one bf16 epsilon, amplified by Qwen's large readout "
    "logits, not rounding). Every older model used in a wave B comparison is rescored in serve2 on every item and "
    "condition (RESCORE_MODELS), and every comparison, planned or descriptive, uses those rescorings. Wave B is read "
    "only when every rescoring is complete, its tokens match the study environment's, it ran on its pinned GPU type, "
    "its gate passed in transformers 5 at the pinned revision and, for node-local weights, its files matched the "
    "manifest (rescore_check). The original-environment scores are reported beside the serve2 ones, descriptively. "
    "The six item-share tests, the 16-test Holm family, the thresholds and the competence gate are unchanged.")


def rescore_spec(tag: str) -> ModelSpec:
    """An older comparison model as serve2 loads it for the full rescoring: its own template, class, attention and
    prompts (as its earlier run), wave B's loading check and environment record, scores under `<tag>-serve2`. The
    Mistral tokenizer is read from tokenizer.json (mistral_format=False, as wave B and the environment check); the
    32B model needs an 80 GB card and a 128 GB job."""
    spec = spec_for(tag)
    if tag not in RESCORE_MODELS or spec.env != "serve":
        raise ValueError(f"{tag} is not an older comparison model of wave B ({RESCORE_MODELS})")
    big = spec.params_b > 20
    return replace(spec, tag=tag + RESCORE_SUFFIX, wave="rescore", env="serve2",
                   tokenizer_kwargs={"mistral_format": False} if spec.family == "mistral" else {},
                   strict_load=True, gpu_min_gb=80 if big else 24, host_mem_min_gb=128 if big else 0,
                   template_notes=f"Serve2 rescoring of {tag} for every wave B comparison (design section 8 amendment).")


def run_family_rescore(cfg: dict) -> int:
    """`family_rescore_<tag>`: an older comparison model rescored in serve2 on every item and condition (all primary
    items and determined successes, every arm of sizes.size_jobs), after the same gate as a wave B model."""
    tag = cfg["model"]
    if tag not in RESCORE_MODELS:
        raise SystemExit(f"the serve2 rescoring covers {RESCORE_MODELS}, not {tag!r}")
    spec = rescore_spec(tag)
    env = require_environment(spec)
    env["resources"] = require_resources(spec)
    cfg = {**cfg, "model_spec": spec.record()}
    return sizes.score_shard(cfg, spec.tag, lambda: open_scorer(spec), lambda t, arm: family_messages(spec, t, arm),
                             gate=_gate(spec, env))


def rescore_check(tag: str, run: str, tokens: str | None) -> dict[str, Any]:
    """Whether `tag`'s serve2 rescoring may be read (the amendment's gate): every shard complete at one commit; the
    token ids identical between the study environment and serve2 (`tokens`: the compare JSON of
    wf.tools.envcheck_tokens); every shard on RESCORE_GPU[tag] alone; every shard's gate passed in transformers 5 at
    the pinned revision; and, for a node-local model, the downloaded files hashed against the manifest."""
    from wf.provenance import run_record

    folders, problems = complete_shards(run)
    out: dict[str, Any] = {"run": run, "problems": list(problems)}
    checks: dict[str, bool] = {"complete": bool(folders) and not problems}
    tok = _resolve(tokens) if tokens else None
    trec = json.loads(tok.read_text(encoding="utf-8")) if tok is not None and tok.exists() else None
    out["tokens"] = {"path": str(tok) if tok else None, "found": trec is not None,
                     **({k: trec.get(k) for k in ("ok", "model", "n_compared", "n_different", "n_only_one_side")}
                        if trec else {})}
    checks["tokens_identical"] = bool(trec and trec.get("ok") is True and trec.get("model") == tag)
    want = RESCORE_GPU[tag]
    gpus = {p.name: run_record(p)["gpu"] for p in folders}
    out["gpu"] = {"expected": want, "recorded": gpus}
    checks["gpu_type"] = bool(folders) and all(g == [want] for g in gpus.values())
    gates, revisions, weights = {}, set(), {}
    for p in folders:
        g = json.loads((p / "tokenization_checks.json").read_text(encoding="utf-8")) \
            if (p / "tokenization_checks.json").exists() else {}
        env = g.get("environment") or {}
        mock = g.get("skipped") == "mock" and env.get("checked") == "mock"
        major = int(str(env.get("transformers") or "0").split(".")[0] or 0)
        gates[p.name] = mock or (g.get("ok") is True and env.get("checked") is True and major >= 5)
        cfg_rec = json.loads((p / "config.json").read_text(encoding="utf-8")) if (p / "config.json").exists() else {}
        spec_rec = cfg_rec.get("model_spec") or {}
        revisions.add((spec_rec.get("tag"), spec_rec.get("revision")))
        if tag in RESCORE_NODE_LOCAL:
            w = g.get("weights_check") or {}
            weights[p.name] = mock or bool(w.get("ok") is True and w.get("revision") == C.REVISIONS[spec_for(tag).hf_id])
    out["gate"], out["weights"] = gates, weights
    checks["gate_passed_in_serve2"] = bool(folders) and all(gates.values())
    checks["pinned_revision"] = revisions == {(tag + RESCORE_SUFFIX, C.REVISIONS[spec_for(tag).hf_id])}
    if tag in RESCORE_NODE_LOCAL:
        checks["weights_hashed"] = bool(folders) and all(weights.values())
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


def rescore_gate(cfg: dict) -> dict[str, Any]:
    """The analysis's gate after the amendment: every older comparison model's serve2 rescoring passes rescore_check
    (cfg["rescore"]: {tag: {"run": ..., "tokens": ...}}); every model of RESCORE_MODELS must be there."""
    given = cfg.get("rescore") or {}
    models = {t: rescore_check(t, given[t]["run"], given[t].get("tokens")) for t in RESCORE_MODELS if t in given}
    missing = [t for t in RESCORE_MODELS if t not in given]
    return {"models": models, "missing_from_config": missing,
            "pass": not missing and all(m["pass"] for m in models.values())}


def environment_sensitivity(serve2: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """A reference model's serve2 block against its original-environment block (descriptive; the amendment): the
    headline numbers side by side, serve2 minus original."""
    def nums(b: dict[str, Any]) -> dict[str, float]:
        return {"item_share_toward_yes": b["primary_metrics"]["item_share_toward_yes"]["estimate"],
                "goal_to_evidence_ratio": b["primary_metrics"]["goal_to_evidence_ratio"]["estimate"],
                "yes_no_delta_L": b["yes_no"]["goal_effect"]["delta_L_nats"],
                "two_candidate_delta_L": b["goal_effect"]["delta_L_nats"]}
    a, b = nums(serve2), nums(original)
    return {"serve2": a, "original": b, "serve2_minus_original": {k: a[k] - b[k] for k in a}}


# ---------------- analysis ----------------


def standardised_effect(s: list[dict], items: list[dict], a: str, b: str) -> dict[str, Any]:
    """Mean (L under `a` minus L under `b`) over the SD of L under `b`, on the same items; game-clustered CI.

    Divides the goal effect by how widely the model's scores spread across items under the swapped task, so a
    model that is more extreme in log-odds everywhere does not look more goal-sensitive for that alone.
    """
    pairs = paired_conditions(s, items, a, b)

    def stat(rows: list[tuple[float, float]]) -> float:
        if len(rows) < 2:
            return float("nan")
        sd = statistics.stdev(lb for _, lb in rows)
        return sum(la - lb for la, lb in rows) / len(rows) / sd if sd > 0 else float("nan")

    rows = [(la, lb) for _, la, lb in pairs]
    boot = cluster_bootstrap(stat, rows, [g for g, _, _ in pairs], seed=C.RNG_SEED)
    return {"estimate": stat(rows), "ci": [boot["ci_low"], boot["ci_high"]], "n_items": len(rows),
            "sd_L_control": statistics.stdev(lb for _, lb in rows) if len(rows) > 1 else float("nan")}


def first_token_coverage(s: list[dict]) -> dict[str, Any]:
    """Per condition: how much next-token mass the two first tokens hold (P(t+) + P(t-); for yes/no, P(Yes) +
    P(No)), and the commonest first-token pairs."""
    out: dict[str, Any] = {}
    for cond in sorted({x["condition"] for x in s}):
        rows = [x for x in s if x["condition"] == cond]
        mass = sorted(x["first_mass"] for x in rows if x.get("first_mass") is not None and x["first_mass"] == x["first_mass"])
        out[cond] = {"n": len(rows), "median": statistics.median(mass) if mass else float("nan"),
                     "p05": mass[int(0.05 * len(mass))] if mass else float("nan"),
                     "share_below_0.5": sum(m < 0.5 for m in mass) / len(mass) if mass else float("nan"),
                     "first_tokens": dict(Counter(f"{x.get('first_token_plus', '')!r} vs {x.get('first_token_minus', '')!r}"
                                                  for x in rows).most_common(5))}
    return out


FLIP_DEFINITIONS = {
    "flip_full_L_toward_success": "share of items whose full two-candidate L is above 0 under the aligned task and "
                                  "below 0 under the swapped task (one direction; NaN pairs left out)",
    "flip_first_token_sign_change": "share of items whose first-token L changes sign between the aligned and swapped "
                                    "tasks, either direction (the confirmatory decision_flip, as in RQ2's tables)",
}


# ---------------- primary cross-model metrics, competence gate, comparisons (amendment, 2026-11-20, pre-data) ----

# Third amendment (2026-11-20, before any wave A score was read): the item share is the one primary cross-model
# metric; R is secondary (on the reference runs R differs 3.8-fold between Qwen3-8B and Llama-3.1-8B while their
# shares are equal, because R's denominator is each model's confidence on failures; notes/families_design.md).
PRIMARY_METRICS = ("item_share_toward_yes",)
SECONDARY_METRICS = ("goal_to_evidence_ratio",)
# Amended 2026-11-20 (second amendment, still before any wave A score was read): the first version required 5% to
# 95% Yes on failures, and its lower bound excluded Qwen3-8B for answering No to failures (notes/families_design.md).
COMPETENCE = {"auc_neutral_min": 0.70, "yes_share_neutral_failures_max": 0.95, "yes_share_neutral_successes_min": 0.05,
              "median_yes_no_mass_neutral_min": 0.5}
EQUIVALENCE_FACTOR = 1.5  # "matches": R within a factor of 1.5, TOST on log R (secondary)
SHARE_MARGIN = 0.05  # "matches": item shares within ±0.05 (5 points), TOST on the paired difference (primary)
DROPPED_DRAWS_MAX = 0.02  # the R ratio is not estimable when more than 2% of paired draws have R <= 0 or undefined
NOISE_FLOOR_NATS = 0.3  # yes/no ΔL differences between models below this are treated as noise
ALPHA = 0.05
REFERENCE_TAGS = ("qwen3-8b", "llama3.1-8b")
# The comparisons the outcome table reads (notes/families_design.md), fixed before data: each is two tests in the
# primary family, a difference in item share and its equivalence within ±SHARE_MARGIN.
PLANNED_COMPARISONS = (("gemma2-9b-it", "llama3.1-8b"), ("gemma2-9b-it", "qwen3-8b"),
                       ("mistral-7b-v0.3", "llama3.1-8b"), ("mistral-7b-v0.3", "qwen3-8b"),
                       ("gemma2-9b-it", "mistral-7b-v0.3"),
                       ("llama3.2-1b", "llama3.1-8b"), ("llama3.2-3b", "llama3.1-8b"))


def _yes_no_rows(s: list[dict], primary: list[dict], successes: list[dict],
                 arms: tuple[str, str, str] = ("yn_aligned", "yn_swapped", "yn_absent")) -> tuple[list[tuple], list[str], list[str]]:
    """Per item, on the yes/no format: ("f", L aligned - L swapped, L neutral) for a failure with all three tasks
    scored, ("s", nan, L neutral) for a success scored under the neutral task; each item's game; each item's id.
    `arms` (aligned, swapped, neutral) = ("aligned", "swapped", "absent") gives the same rows on the two-candidate
    format (B3's fallback readout, review 1)."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    rows, games, ids = [], [], []
    for t in primary:
        k = [(t["id"], a) for a in arms]
        if all(x in by for x in k):
            rows.append(("f", by[k[0]] - by[k[1]], by[k[2]]))
            games.append(t["game_id"])
            ids.append(t["id"])
    for t in successes:
        if (t["id"], arms[2]) in by:
            rows.append(("s", float("nan"), by[(t["id"], arms[2])]))
            games.append(t["game_id"])
            ids.append(t["id"])
    return rows, games, ids


def _share_toward_yes(rows: list[tuple]) -> float:
    d = [r[1] for r in rows if r[0] == "f"]
    return sum(x > 0 for x in d) / len(d) if d else float("nan")


def _ratio(rows: list[tuple]) -> float:
    """R = mean goal effect on failures / (mean neutral L on successes - mean neutral L on failures)."""
    f = [r for r in rows if r[0] == "f"]
    sc = [r[2] for r in rows if r[0] == "s"]
    if not f or not sc:
        return float("nan")
    evidence = sum(sc) / len(sc) - sum(r[2] for r in f) / len(f)
    return (sum(r[1] for r in f) / len(f)) / evidence if evidence > 0 else float("nan")


def primary_metrics(s: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """The item share (primary) and R (secondary since the third amendment), both on the yes/no format, each with a
    game-clustered 95% CI and a two-sided game sign-flip permutation p (for R, the p of its numerator: R is zero
    exactly when the goal effect is, and its denominator is fixed under the flips).

    item_share_toward_yes: the share of failures with L(aligned) > L(swapped).
    goal_to_evidence_ratio: R, the mean goal effect over the evidence effect (determined successes against
    failures, both under the neutral task). Rescaling every logit by one factor leaves R unchanged, but across
    families R mostly reflects how far each model pushes its log-odds on failures (the denominator), so it is
    read within model only.
    """
    rows, games, _ = _yes_no_rows(s, primary, successes)
    f = [(r, g) for r, g in zip(rows, games) if r[0] == "f"]
    sc = [r for r in rows if r[0] == "s"]
    share = cluster_bootstrap(_share_toward_yes, rows, games, seed=C.RNG_SEED)
    ratio = cluster_bootstrap(_ratio, rows, games, seed=C.RNG_SEED)
    fg = [g for _, g in f]
    p_share = pair_permutation([(r[1] > 0) - (r[1] < 0) for r, _ in f], seed=C.RNG_SEED, cluster_ids=fg) if f else float("nan")
    p_ratio = pair_permutation([r[1] for r, _ in f], seed=C.RNG_SEED, cluster_ids=fg) if f else float("nan")
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return {"item_share_toward_yes": {"estimate": share["estimate"], "ci": [share["ci_low"], share["ci_high"]],
                                      "null": 0.5, "n_items": len(f), "p_two_sided": p_share, "role": "primary"},
            "goal_to_evidence_ratio": {"estimate": ratio["estimate"], "ci": [ratio["ci_low"], ratio["ci_high"]], "null": 0.0,
                                       "role": "secondary (descriptive; across families mostly the evidence effect's size)",
                                       "goal_effect_nats": mean([r[1] for r, _ in f]),
                                       "evidence_effect_nats": mean([r[2] for r in sc]) - mean([r[2] for r, _ in f]),
                                       "n_failures": len(f), "n_successes": len(sc), "p_two_sided": p_ratio}}


def argmax_answers(s: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """Yes/no answers read as the model's choice (Yes when L > 0), not as mean P(Yes): the share answering Yes per
    task, and the flips between the swapped and the aligned task on failures, with game-clustered CIs."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}

    def yes_share(items: list[dict], arm: str) -> float:
        v = [by[(t["id"], arm)] > 0 for t in items if (t["id"], arm) in by]
        return sum(v) / len(v) if v else float("nan")

    pairs = [((by[(t["id"], "yn_swapped")] > 0, by[(t["id"], "yn_aligned")] > 0), t["game_id"]) for t in primary
             if (t["id"], "yn_swapped") in by and (t["id"], "yn_aligned") in by]
    to_yes = lambda v: sum(1 for sw, al in v if al and not sw) / len(v) if v else float("nan")
    to_no = lambda v: sum(1 for sw, al in v if sw and not al) / len(v) if v else float("nan")
    out: dict[str, Any] = {"yes_share_failures": {a: yes_share(primary, a) for a in sizes.YN_ARMS},
                           "yes_share_successes": {a: yes_share(successes, a) for a in ("yn_aligned", "yn_absent")},
                           "n_failure_pairs": len(pairs)}
    for name, fn in (("flip_no_to_yes", to_yes), ("flip_yes_to_no", to_no)):
        b = cluster_bootstrap(fn, [p for p, _ in pairs], [g for _, g in pairs], seed=C.RNG_SEED)
        out[name] = {"share": b["estimate"], "ci": [b["ci_low"], b["ci_high"]]}
    return out


def competence_gate(block: dict[str, Any]) -> dict[str, Any]:
    """Fixed before any wave A score (second amendment, 2026-11-20). On the yes/no format under the neutral task, a
    model is interpretable only if all four hold: AUC (successes against failures) at least 0.70; it answers Yes
    (argmax) to at most 95% of failures, so not Yes to everything; it answers Yes to at least 5% of determined
    successes, so not No to everything; and the median P(Yes) + P(No) is at least 0.5. A model that fails is
    reported as not interpretable, never as a null result."""
    auc = block["yes_no"]["auc"]["yn_absent"]["auc"]
    yes_f = block["argmax_answers"]["yes_share_failures"]["yn_absent"]
    yes_s = block["argmax_answers"]["yes_share_successes"]["yn_absent"]
    mass = block["first_token_coverage"].get("yn_absent", {}).get("median", float("nan"))
    checks = {"auc_neutral": {"value": auc, "pass": auc >= COMPETENCE["auc_neutral_min"]},
              "yes_share_neutral_failures": {"value": yes_f, "pass": yes_f <= COMPETENCE["yes_share_neutral_failures_max"]},
              "yes_share_neutral_successes": {"value": yes_s, "pass": yes_s >= COMPETENCE["yes_share_neutral_successes_min"]},
              "median_yes_no_mass_neutral": {"value": mass, "pass": mass >= COMPETENCE["median_yes_no_mass_neutral_min"]}}
    return {"interpretable": all(c["pass"] for c in checks.values()), "checks": checks, "rule": COMPETENCE}


def _draws(stat: Callable[[list], Any], rows: list, games: list[str], B: int = C.BOOTSTRAP_RESAMPLES) -> list:
    rng = random.Random(C.RNG_SEED)
    by: dict[str, list] = {}
    for r, g in zip(rows, games):
        by.setdefault(g, []).append(r)
    keys = sorted(by)
    return [stat([r for g in rng.choices(keys, k=len(keys)) for r in by[g]]) for _ in range(B)]


def _pct(v: list[float], q: float) -> float:
    v = sorted(x for x in v if x == x)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def _exp(x: float) -> float:
    return math.exp(x) if x == x else float("nan")


def _compare_R(rows: list[tuple], games: list[str]) -> dict[str, Any]:
    """R_A against R_B on paired rows ((row A, row B) per item), secondary.

    The ratio R_A / R_B is used when it is estimable: both point Rs above zero and at most DROPPED_DRAWS_MAX of the
    paired draws with either R at or below zero (or undefined), since a log ratio cannot be taken there and dropping
    those draws silently would bias the interval. Then "matches" is the TOST within a factor of EQUIVALENCE_FACTOR
    (90% interval of log(R_A / R_B)) and "differs" is its 95% interval excluding zero. Otherwise the ratio is not
    estimable (reported as such) and "differs" comes from the 95% interval of the paired difference R_A - R_B;
    there is no equivalence statement then. A model without the bias (R near zero) is the case this is for.
    """
    pair = lambda v: (_ratio([x for x, _ in v]), _ratio([y for _, y in v]))
    ra, rb = pair(rows)
    draws = _draws(pair, rows, games)
    B = len(draws)
    ok_ratio = [math.log(a / b) for a, b in draws if a > 0 and b > 0]
    dropped = B - len(ok_ratio)
    diffs = [a - b for a, b in draws if a == a and b == b]
    diff_est = ra - rb if ra == ra and rb == rb else float("nan")
    diff = {"estimate": diff_est, "ci95": [_pct(diffs, 0.025), _pct(diffs, 0.975)],
            "n_resamples_undefined": B - len(diffs)}
    estimable = bool(ra > 0 and rb > 0 and dropped <= DROPPED_DRAWS_MAX * B)
    out: dict[str, Any] = {"R_a": ra, "R_b": rb, "n_resamples": B, "n_resamples_dropped": dropped,
                           "max_share_dropped": DROPPED_DRAWS_MAX, "ratio_estimable": estimable, "difference": diff}
    if estimable:
        bound = math.log(EQUIVALENCE_FACTOR)
        est = math.log(ra / rb)
        lo90, hi90, lo95, hi95 = (_pct(ok_ratio, q) for q in (0.05, 0.95, 0.025, 0.975))
        matches = bool(-bound < lo90 and hi90 < bound)
        differs = bool(lo95 > 0 or hi95 < 0)
        out.update({"basis": "ratio", "estimate": math.exp(est), "ci90": [_exp(lo90), _exp(hi90)],
                    "ci95": [_exp(lo95), _exp(hi95)], "matches": matches, "differs": differs})
    else:
        lo95, hi95 = diff["ci95"]
        computable = diff_est == diff_est and lo95 == lo95
        differs = bool(computable and (lo95 > 0 or hi95 < 0))
        out.update({"basis": "difference" if computable else "not computable", "estimate": float("nan"),
                    "ci90": [float("nan")] * 2, "ci95": [float("nan")] * 2, "matches": False, "differs": differs,
                    "why_not_estimable": ("R at or below zero (or undefined) in the point estimate" if not (ra > 0 and rb > 0)
                                          else f"{dropped} of {B} paired draws have R at or below zero")})
    out["verdict"] = ("differs, within the margin" if out["differs"] and out["matches"] else "differs" if out["differs"]
                      else "matches" if out["matches"] else "not computable" if out["basis"] == "not computable"
                      else "inconclusive")
    return out


def _compare_share(rows: list[tuple], games: list[str]) -> dict[str, Any]:
    """Item share A minus B on the failures both models have (primary): the paired difference with 90% and 95%
    paired-bootstrap CIs; p_differs, a two-sided game sign-flip permutation p on the per-item differences
    (1 if only A moved toward Yes, -1 if only B did; swapping the models within a game flips its signs); and
    p_equivalent, the TOST p for a difference within ±SHARE_MARGIN, the larger of the two one-sided bootstrap ps."""
    f = [(x, y, g) for (x, y), g in zip(rows, games) if x[0] == "f"]
    d = [(x[1] > 0) - (y[1] > 0) for x, y, _ in f]
    fg = [g for _, _, g in f]
    stat = lambda v: sum(v) / len(v) if v else float("nan")
    est = stat(d)
    draws = [x for x in _draws(stat, d, fg) if x == x]
    lo90, hi90, lo95, hi95 = (_pct(draws, q) for q in (0.05, 0.95, 0.025, 0.975))
    B = len(draws)
    p_low = (sum(x <= -SHARE_MARGIN for x in draws) + 1) / (B + 1) if B else float("nan")
    p_high = (sum(x >= SHARE_MARGIN for x in draws) + 1) / (B + 1) if B else float("nan")
    return {"estimate": est, "ci90": [lo90, hi90], "ci95": [lo95, hi95], "n_items": len(d), "margin": SHARE_MARGIN,
            "p_differs": pair_permutation(d, seed=C.RNG_SEED, cluster_ids=fg) if d else float("nan"),
            "p_equivalent": max(p_low, p_high),
            "differs": bool(est == est and (lo95 > 0 or hi95 < 0)),
            "matches": bool(est == est and -SHARE_MARGIN < lo90 and hi90 < SHARE_MARGIN)}


def compare_models(sa: list[dict], sb: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """Model A against model B on the same items (yes/no format), with a paired game-clustered bootstrap: games are
    resampled once for both models.

    item_share_difference (primary): A minus B, see `_compare_share`; "differs" when its 95% CI excludes zero and
    "matches" when its 90% CI lies within ±SHARE_MARGIN (descriptive; the primary family reads the Holm-adjusted
    p_differs and p_equivalent). R (secondary): see `_compare_R`. ΔL as a difference (A minus B) with a 95% CI; a ΔL
    difference under 0.3 nats is flagged as within the noise floor.
    """
    ra, ga, ia = _yes_no_rows(sa, primary, successes)
    rb, _, ib = _yes_no_rows(sb, primary, successes)
    keyed_b = dict(zip(ib, rb))
    common = [(r, g, i) for r, g, i in zip(ra, ga, ia) if i in keyed_b]  # the items both models have
    rows = [(r, keyed_b[i]) for r, _, i in common]
    games = [g for _, g, _ in common]
    mean_goal = lambda v: (lambda d: sum(d) / len(d) if d else float("nan"))([r[1] for r in v if r[0] == "f"])
    dfn = lambda v: mean_goal([x for x, _ in v]) - mean_goal([y for _, y in v])
    d = _draws(dfn, rows, games)
    out: dict[str, Any] = {"n_failures": sum(r[0] == "f" for r, _ in rows), "n_successes": sum(r[0] == "s" for r, _ in rows),
                           "item_share_difference": _compare_share(rows, games), "R": _compare_R(rows, games),
                           "delta_L_difference_nats": {"estimate": dfn(rows), "ci95": [_pct(d, 0.025), _pct(d, 0.975)]}}
    out["delta_L_difference_nats"]["within_noise_floor"] = abs(out["delta_L_difference_nats"]["estimate"]) < NOISE_FLOOR_NATS
    return out


def planned_comparisons(models: list[str], references: list[str]) -> list[tuple[str, str]]:
    """The planned comparisons whose two models are both in the analysis's configuration (fixed before data)."""
    have = set(models) | set(references)
    return [(a, b) for a, b in PLANNED_COMPARISONS if a in have and b in have]


def primary_family(blocks: dict[str, dict[str, Any]], expected: list[str], comparisons: dict[str, dict] | None = None,
                   planned: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """The one primary family (third amendment), Holm-corrected within it at a fixed size:

    - per wave A model in `expected`, the item share toward Yes against one half;
    - per planned comparison (A, B), two tests on the paired item-share difference: that it differs from zero
      (p_differs) and that it lies within ±SHARE_MARGIN (TOST, p_equivalent).

    `blocks` holds the wave A and reference blocks by tag; `comparisons` holds compare_models results keyed
    "A vs B". The family's size is fixed by `expected` and `planned`: a test whose model is still waiting, or whose p
    is missing or NaN, enters with p = 1 so the others are corrected for the full family, never a smaller one.
    A per-model test is supported when its Holm p < 0.05, the share is above one half and the model passed the
    competence gate. A comparison involving a model that failed the gate is "not interpretable".
    """
    comparisons = comparisons or {}
    planned = list(planned or [])
    p: dict[str, float] = {}
    info: dict[str, dict] = {}

    def interpretable(tag: str) -> bool | None:
        return blocks[tag]["competence"]["interpretable"] if tag in blocks else None

    for tag in expected:
        key = f"{tag}:item_share_toward_yes"
        rec = blocks[tag]["primary_metrics"]["item_share_toward_yes"] if tag in blocks else None
        pv = rec["p_two_sided"] if rec else float("nan")
        p[key] = pv if pv == pv else 1.0
        info[key] = {"kind": "share", "tag": tag, "rec": rec, "padded": pv != pv}
    for a, b in planned:
        c = comparisons.get(f"{a} vs {b}", {}).get("item_share_difference")
        for test, field in (("differs", "p_differs"), ("equivalent", "p_equivalent")):
            key = f"{a} vs {b}:share_{test}"
            pv = c[field] if c else float("nan")
            p[key] = pv if pv == pv else 1.0
            info[key] = {"kind": test, "pair": (a, b), "rec": c, "padded": pv != pv}
    adj = holm(p)
    tests: dict[str, dict] = {}
    for key, padj in adj.items():
        i = info[key]
        rec = i["rec"]
        if i["kind"] == "share":
            ok = interpretable(i["tag"])
            if rec is None:
                verdict = "waiting"
            elif not ok:
                verdict = "not interpretable"
            elif i["padded"]:
                verdict = "not computable"
            else:
                side = rec["estimate"] > rec["null"]
                verdict = "supported" if padj < ALPHA and side else "not supported"
            tests[key] = {"estimate": rec["estimate"] if rec else float("nan"), "ci": rec["ci"] if rec else [float("nan")] * 2,
                          "p": rec["p_two_sided"] if rec else float("nan"), "p_holm": padj, "verdict": verdict}
        else:
            oks = [interpretable(t) for t in i["pair"]]
            if rec is None or None in oks:
                verdict = "waiting"
            elif not all(oks):
                verdict = "not interpretable"
            elif i["padded"]:
                verdict = "not computable"
            else:
                verdict = "supported" if padj < ALPHA else "not supported"
            field = "p_differs" if i["kind"] == "differs" else "p_equivalent"
            tests[key] = {"estimate": rec["estimate"] if rec else float("nan"),
                          "ci": (rec["ci95"] if i["kind"] == "differs" else rec["ci90"]) if rec else [float("nan")] * 2,
                          "p": rec[field] if rec else float("nan"), "p_holm": padj, "verdict": verdict}
    readings = {}
    for a, b in planned:
        dv, ev = tests[f"{a} vs {b}:share_differs"]["verdict"], tests[f"{a} vs {b}:share_equivalent"]["verdict"]
        if dv not in ("supported", "not supported"):
            readings[f"{a} vs {b}"] = dv  # waiting, not interpretable or not computable
        elif dv == "supported":
            readings[f"{a} vs {b}"] = "differs, within the margin" if ev == "supported" else "differs"
        else:
            readings[f"{a} vs {b}"] = "matches" if ev == "supported" else "inconclusive (no difference detected)"
    return {"tests": tests, "comparison_readings": readings, "n_tests_planned": len(expected) + 2 * len(planned),
            "n_tests_padded": sum(i["padded"] for i in info.values()),
            "complete": all(t in blocks for t in expected) and all(f"{a} vs {b}" in comparisons for a, b in planned),
            "share_margin": SHARE_MARGIN, "alpha": ALPHA,
            "correction": "Holm over the full planned family (a missing or NaN p enters as 1); everything else is "
                          "descriptive"}


def family_block(s: list[dict], primary: list[dict], successes: list[dict], spec: ModelSpec) -> dict[str, Any]:
    block = sizes.model_block(s, primary, successes)
    out = {"spec": spec.record(), "primary_metrics": primary_metrics(s, primary, successes),
           "argmax_answers": argmax_answers(s, primary, successes), **block,
           "flips": {"flip_full_L_toward_success": block["flips_to_success"],
                     "flip_first_token_sign_change": block["decision_flip"], "definitions": FLIP_DEFINITIONS},
           "standardised_effect": {"two_candidate": standardised_effect(s, primary, "aligned", "swapped"),
                                   "yes_no": standardised_effect(s, primary, "yn_aligned", "yn_swapped")},
           "first_token_coverage": first_token_coverage(s)}
    out["competence"] = competence_gate(out)
    return out


def prediction_check(block: dict[str, Any]) -> dict[str, Any]:
    two, yn = block["goal_effect"]["delta_L_ci"], block["yes_no"]["goal_effect"]["delta_L_ci"]
    out = {"two_candidate_ci_above_0": two[0] > 0, "yes_no_ci_above_0": yn[0] > 0}
    return {**out, "holds": all(out.values()), "yes_no_ci": list(yn), "two_candidate_ci": list(two)}


def same_population(s: list[dict], primary: list[dict], successes: list[dict]) -> tuple[list[dict], int]:
    """The records whose (item, condition) a family run scores (sizes.size_jobs), and how many others were dropped.

    Every model's block is read on this one population. The reference runs hold more: rq1_score scored the
    successes under the swapped task too, which the family runs do not, so Llama-3.1-8B's first-token coverage
    under the swapped task mixed successes into what is failures only for every other model (review, 2026-11-20).
    """
    wanted = {(t["id"], a) for t, a in sizes.size_jobs(primary, successes)}
    kept = [x for x in s if (x["transition_id"], x["condition"]) in wanted]
    return kept, len(s) - len(kept)


def placement_block(native: list[dict], first_turn: list[dict], primary: list[dict]) -> dict[str, Any]:
    """The goal effect (aligned minus swapped) under the native and the first-turn placement, on the items both
    scored, in both formats, and the per-item difference between the two (first turn minus native), with
    game-clustered CIs."""
    ids = {x["transition_id"] for x in first_turn} & {x["transition_id"] for x in native}
    items = [t for t in primary if t["id"] in ids]
    out: dict[str, Any] = {"n_items": len(items)}
    for fmt, (a, b) in (("two_candidate", ("aligned", "swapped")), ("yes_no", ("yn_aligned", "yn_swapped"))):
        nat = _item_effects(native, items, a, b)
        alt = _item_effects(first_turn, items, a, b)
        both = sorted(set(nat) & set(alt))
        rows = [alt[k] - nat[k] for k in both]
        boot = cluster_bootstrap(lambda v: sum(v) / len(v) if v else float("nan"), rows, [g for g, _ in both],
                                 seed=C.RNG_SEED)
        out[fmt] = {"native": _effect(native, items, primary=a, control=b),
                    "first_turn": _effect(first_turn, items, primary=a, control=b),
                    "first_turn_minus_native": {"delta_L_nats": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]],
                                                "n_items": len(rows)}}
    return out


def _item_effects(s: list[dict], items: list[dict], a: str, b: str) -> dict[tuple[str, str], float]:
    """{(game, item): L under `a` minus L under `b`} on the items given (NaN scores left out)."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    return {(t["game_id"], t["id"]): by[(t["id"], a)] - by[(t["id"], b)] for t in items
            if (t["id"], a) in by and (t["id"], b) in by}


def _resolve(ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else runs_root() / ref


def reference_scores(ref: dict) -> tuple[list[dict], dict[str, list[Path]], dict[str, Any]]:
    """A comparison model's scores from finished runs: {run: [conditions]} under `runs`, records of `model` only.

    A (item, condition) seen twice (two runs holding the same prompt) is kept once, the first in the order given;
    how many repeats there were, and the largest difference in L among them, are reported.
    """
    read: dict[str, list[Path]] = {}
    problems: dict[str, Any] = {}
    out: dict[tuple[str, str], dict] = {}
    repeats, worst = 0, 0.0
    for run, conds in ref["runs"].items():
        folders, why = complete_shards(str(_resolve(run)))
        if why:
            problems[run] = why
            continue
        read[run] = folders
        for p in folders:
            for x in read_jsonl(p / "scoring.jsonl"):
                if x["model"] != ref["model"] or x["condition"] not in conds:
                    continue
                key = (x["transition_id"], x["condition"])
                if key in out:
                    repeats += 1
                    worst = max(worst, abs(out[key]["L"] - x["L"]))
                    continue
                out[key] = x
    return list(out.values()), read, {"problems": problems, "repeats": repeats, "max_abs_L_difference": worst}


def _gate_records(folders: list[Path]) -> dict[str, Any]:
    out = {}
    for p in folders:
        f = p / "tokenization_checks.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            out[p.name] = {k: r.get(k) for k in ("ok", "skipped", "n_candidates_checked", "n_yes_no_prompts_checked", "first_token_pairs",
                                                  "joined_plain_text_mismatches", "max_prompt_tokens",
                                                  "attn_implementation_loaded", "dtype_loaded")}
    return out


ENVIRONMENT_BLOCKED = "not interpretable (environment check)"


def apply_environment_gate(family: dict[str, Any], passed: bool | None) -> dict[str, Any]:
    """Wave B: unless the environment check passed, no test in the family is read (section 8). Every test of the family
    involves a wave B model, so every verdict but "waiting" (a competence verdict too: it is a reading of wave B
    scores) and every comparison reading becomes ENVIRONMENT_BLOCKED, and the estimates, intervals and p values are
    removed (review 1, S5)."""
    family["environment_check_passed"] = passed
    if passed:
        return family
    for t in family["tests"].values():
        if t["verdict"] != "waiting":
            t["verdict"] = ENVIRONMENT_BLOCKED
        for k in ("estimate", "ci", "p", "p_holm"):
            if k in t:
                t[k] = None
    family["comparison_readings"] = {k: (v if v == "waiting" else ENVIRONMENT_BLOCKED)
                                     for k, v in family["comparison_readings"].items()}
    return family


def b1_reading(share_verdict: str | None, yes_no_ci: list[float] | None) -> str | None:
    """What a wave B model's share test says, with its yes/no effect (review 1, M2). A share that is not supported is
    "no detectable effect" only when the yes/no effect's 95% CI also covers zero; with that CI above zero the effect
    is carried by a minority of items, as in Qwen3-4B (share 0.49, yes/no effect 2.27 nats). None: nothing to read
    (no share test, waiting, not interpretable, not computable, or blocked by the environment check)."""
    if share_verdict == "supported":
        return B1_READINGS["supported"]
    if share_verdict != "not supported" or not yes_no_ci:
        return None
    lo, hi = yes_no_ci
    if lo > 0:
        return B1_READINGS["minority"]
    if hi < 0:
        return B1_READINGS["reversed"]
    return B1_READINGS["null"]


def waveb_predictions(report: dict[str, Any], environment_passed: bool | None = None) -> dict[str, Any]:
    """The wave B predictions read off the report (section 10). None: not yet decidable. Nothing is read unless wave
    B's gate passed (`environment_passed`; since the section 8 amendment, default: the serve2 rescorings' gate,
    report["serve2_rescoring"]["pass"]); B1's per-model effect and reading are then left out as well (review 1, S5)."""
    if environment_passed is None:
        environment_passed = (report.get("serve2_rescoring") or {}).get("pass")
    fam = report["primary_family"]
    tests, readings = fam["tests"], fam["comparison_readings"]
    b1 = {}
    for tag in report["models"]:
        share = tests.get(f"{tag}:item_share_toward_yes", {}).get("verdict") if tag in WAVE_B_SHARE_TESTS else None
        if not environment_passed:
            b1[tag] = {"both_formats_positive": None, "share_verdict": share, "reading": None, "holds": None}
            continue
        check = report["prediction_check"][tag]
        both = bool(check["holds"])
        if share in ("waiting", ENVIRONMENT_BLOCKED):
            holds = None
        elif share in (None, "supported", "not interpretable"):  # no share test, or one that is read
            holds = both
        else:  # "not supported" or "not computable"
            holds = False
        b1[tag] = {"both_formats_positive": both, "share_verdict": share,
                   "reading": b1_reading(share, check.get("yes_no_ci")), "holds": holds}

    def pair(a: str, b: str, want_positive: bool, caveat: str = "") -> dict[str, Any]:
        r = readings.get(f"{a} vs {b}")
        est = tests.get(f"{a} vs {b}:share_differs", {}).get("estimate", float("nan"))
        extra = {"caveat": caveat} if caveat else {}
        if not environment_passed:
            return {"reading": ENVIRONMENT_BLOCKED if r not in (None, "waiting") else r, "estimate": None,
                    "holds": None, **extra}
        if r in (None, "waiting", ENVIRONMENT_BLOCKED, "not interpretable", "not computable"):
            return {"reading": r, "estimate": est, "holds": None, **extra}
        differs = r in ("differs", "differs, within the margin")
        # B2 fails only on a clear negative difference; "matches" and "inconclusive" count as holding (review 1)
        holds = (differs and est > 0) if want_positive else not (differs and est < 0)
        return {"reading": r, "estimate": est, "holds": bool(holds), **extra}

    b2 = {f"{a} vs {b}": pair(a, b, False, PAIR_CAVEAT) for a, b in WAVE_B_PLANNED[:4]}
    a, b = WAVE_B_PLANNED[4]
    b3 = pair(a, b, True)
    fallback = report.get("b3_fallback") or {}
    if environment_passed and fallback.get("applies"):
        # the base model fails only the yes/no coverage criterion: B3 is read on the two-candidate share instead
        c = fallback.get("comparison") or {}
        est = c.get("estimate", float("nan"))
        differs = bool(c.get("differs"))
        b3 = {"reading": "differs" if differs else "no difference detected", "estimate": est,
              "ci95": c.get("ci95"), "holds": bool(differs and est > 0), "basis": "two-candidate fallback"}
    return {"text": PREDICTION_B, "B1": b1, "B2": b2, "B3": {f"{a} vs {b}": b3}}


def b3_fallback(blocks: dict[str, dict[str, Any]], scores: dict[str, list[dict]], primary: list[dict],
                successes: list[dict]) -> dict[str, Any]:
    """B3's pre-registered fallback (review 1): when the base model (plain format) fails only the yes/no coverage
    criterion and reaches the competence AUC on the two-candidate score under the neutral task, and the instruct model
    (plain format) is interpretable, B3 is read on the two-candidate item share, instruct minus base, paired by item
    with a game-clustered bootstrap (B3_FALLBACK)."""
    inst, base = WAVE_B_PLANNED[4]
    if inst not in blocks or base not in blocks:
        return {"applies": None, "why": "waiting", "rule": B3_FALLBACK}
    failed = sorted(k for k, c in blocks[base]["competence"]["checks"].items() if not c["pass"])
    auc2 = blocks[base]["auc"]["absent"]["auc"]
    out: dict[str, Any] = {"rule": B3_FALLBACK, "base_failed_checks": failed, "base_two_candidate_auc_neutral": auc2,
                           "instruct_interpretable": blocks[inst]["competence"]["interpretable"]}
    if failed != ["median_yes_no_mass_neutral"]:
        return {**out, "applies": False, "why": "the base model does not fail exactly the coverage criterion"}
    if not (auc2 >= B3_FALLBACK["base_two_candidate_auc_min"] and out["instruct_interpretable"]):
        return {**out, "applies": False, "why": "base two-candidate AUC below 0.70 or instruct model not interpretable"}
    two = ("aligned", "swapped", "absent")
    ra, ga, ia = _yes_no_rows(scores[inst], primary, successes, two)
    rb, _, ib = _yes_no_rows(scores[base], primary, successes, two)
    keyed = dict(zip(ib, rb))
    common = [(r, g, i) for r, g, i in zip(ra, ga, ia) if i in keyed]
    return {**out, "applies": True, "why": "base model fails only the yes/no coverage criterion",
            "comparison": _compare_share([(r, keyed[i]) for r, _, i in common], [g for _, g, _ in common])}


def withhold_waveb_readings(report: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    """Review 1 (S5): unless the environment check passed, the wave B analysis writes no reading of a wave B score.
    Each wave B model's block keeps only its gate records; its effect checks, every comparison involving it, the
    placement and template checks, and B3's fallback are replaced by ENVIRONMENT_BLOCKED (the family itself is
    blocked by apply_environment_gate and the predictions by waveb_predictions)."""
    tags = set(tags)
    for tag in list(report["models"]):
        if tag in tags:
            m = report["models"][tag]
            report["models"][tag] = {"withheld": ENVIRONMENT_BLOCKED, "spec": m.get("spec"),
                                     "tokenization_gate": m.get("tokenization_gate"),
                                     "records_outside_population": m.get("records_outside_population")}
            report["prediction_check"][tag] = {"withheld": ENVIRONMENT_BLOCKED}
    for key in ("comparisons", "comparisons_original_environment"):
        for k in list(report.get(key) or {}):
            a, b = k.split(" vs ")
            if a in tags or b in tags:
                report[key][k] = {"withheld": ENVIRONMENT_BLOCKED}
    for key in ("placement", "template_checks"):
        report[key] = {t: {"withheld": ENVIRONMENT_BLOCKED, "run": v.get("run")} for t, v in (report.get(key) or {}).items()}
    if report.get("b3_fallback"):
        report["b3_fallback"] = {"withheld": ENVIRONMENT_BLOCKED, "rule": B3_FALLBACK}
    report["withheld"] = ENVIRONMENT_BLOCKED
    return report


def run_families_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    wave = cfg.get("wave", "A")
    sources = tuple(str(_resolve(s)) for s in cfg.get("sources", SOURCES))
    sets = rq1x.rq1x_sets(load_items(sources))
    primary, successes = sets["primary"], sets["successes"]
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_successes": len(successes)}, "wave": wave,
                              "prediction": PREDICTION if wave == "A" else PREDICTION_B, "formats": FORMATS,
                              "models": {}, "reference": {},
                              "prediction_check": {}, "placement": {}, "waiting": {},
                              "rules": {"competence": COMPETENCE, "equivalence_factor_R": EQUIVALENCE_FACTOR,
                                        "share_margin": SHARE_MARGIN, "R_ratio_max_share_dropped": DROPPED_DRAWS_MAX,
                                        "noise_floor_delta_L_nats": NOISE_FLOOR_NATS, "alpha": ALPHA,
                                        "primary_metrics": list(PRIMARY_METRICS),
                                        "secondary_metrics": list(SECONDARY_METRICS)}}
    read: dict[str, list[Path]] = {}
    scores: dict[str, list[dict]] = {}
    ref_scores: dict[str, list[dict]] = {}
    for tag in cfg["models"]:
        folders, problems = complete_shards(f"family_score_{tag}")
        if problems:
            report["waiting"][tag] = problems
            continue
        read[f"family_score_{tag}"] = folders
        s, dropped = same_population([x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag],
                                     primary, successes)
        scores[tag] = s
        block = family_block(s, primary, successes, spec_for(tag))
        block["tokenization_gate"] = _gate_records(folders)
        block["records_outside_population"] = dropped
        report["models"][tag] = block
        report["prediction_check"][tag] = prediction_check(block)
    for tag, ref in (cfg.get("reference") or {}).items():
        s, got, info = reference_scores(ref)
        if info["problems"]:
            report["waiting"][f"reference {tag}"] = info["problems"]
            continue
        s, info["records_outside_population"] = same_population(s, primary, successes)
        read.update({f"{tag}:{run}": f for run, f in got.items()})
        ref_scores[tag] = s
        report["reference"][tag] = {**family_block(s, primary, successes, spec_for(tag)), "read": info}
    # wave B after the section 8 amendment: the references above are the serve2 rescorings; their original-environment
    # scores are read too, for the descriptive environment-sensitivity column only (never a test or a reading)
    orig_scores: dict[str, list[dict]] = {}
    if wave != "A":
        report["environment_sensitivity"] = {}
        for tag, ref in (cfg.get("reference_original") or {}).items():
            s, got, info = reference_scores(ref)
            if info["problems"]:
                report["waiting"][f"original-environment reference {tag}"] = info["problems"]
                continue
            s, _ = same_population(s, primary, successes)
            read.update({f"{tag} (original environment):{run}": f for run, f in got.items()})
            orig_scores[tag] = s
            if tag in report["reference"]:
                report["environment_sensitivity"][tag] = environment_sensitivity(
                    report["reference"][tag], family_block(s, primary, successes, spec_for(tag)))
    # every family model against each reference model, and the planned comparisons, paired on the same items; the
    # planned ones' item-share tests are in the primary family, the rest is descriptive. Wave B's family is fixed
    # (WAVE_B_SHARE_TESTS and WAVE_B_PLANNED, 16 tests) whatever the configuration holds.
    if wave == "A":
        planned, expected = planned_comparisons(list(cfg["models"]), list(cfg.get("reference") or {})), list(cfg["models"])
        primary_refs = list(ref_scores)
    else:
        planned, expected = list(WAVE_B_PLANNED), list(WAVE_B_SHARE_TESTS)
        primary_refs = [r for r in ref_scores if r in REFERENCE_TAGS]
    both = {**scores, **ref_scores}
    pairs = [(tag, ref) for tag in scores for ref in primary_refs]
    pairs += [(a, b) for a, b in planned if (a, b) not in pairs and a in both and b in both]
    pairs += [(a, b) for a, b in (cfg.get("descriptive_comparisons") or []) if (a, b) not in pairs and a in both and b in both]
    report["comparisons"] = {f"{a} vs {b}": compare_models(both[a], both[b], primary, successes) for a, b in pairs}
    report["primary_family"] = primary_family({**report["models"], **report["reference"]}, expected,
                                              report["comparisons"], planned)
    if wave != "A":
        # the environment check's verdict is reported (it failed, section 8); after the amendment it no longer gates.
        # Wave B is read only when every older comparison model's serve2 rescoring passes rescore_check.
        env_run = cfg.get("envcheck", "families_envcheck")
        f = _resolve(env_run) / "metrics.json"
        env = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        report["environment_check"] = {"run": env_run, "found": env is not None,
                                       "pass": None if env is None else bool(env.get("pass")), "gates": False}
        report["serve2_rescoring"] = rescore_gate(cfg)
        for t, m in report["serve2_rescoring"]["models"].items():
            if not m["checks"]["complete"]:
                report["waiting"][f"serve2 rescoring {t}"] = m["problems"] or ["incomplete"]
        # descriptive: the comparisons involving a reference, with its original-environment scores instead
        report["comparisons_original_environment"] = {
            f"{a} vs {b}": compare_models(orig_scores.get(a, both.get(a)), orig_scores.get(b, both.get(b)), primary,
                                          successes)
            for a, b in pairs if (a in orig_scores or b in orig_scores)}
        apply_environment_gate(report["primary_family"], report["serve2_rescoring"]["pass"])
    for tag, run in (cfg.get("placement") or {}).items():
        folders, problems = complete_shards(run)
        if problems or tag not in scores:
            report["waiting"][f"placement {tag}"] = problems or [f"family_score_{tag} not read"]
            continue
        read[run] = folders
        alt = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag + FIRST_TURN]
        report["placement"][tag] = {**placement_block(scores[tag], alt, primary), "run": run,
                                    "tokenization_gate": _gate_records(folders)}
        if wave != "A":
            report["placement"][tag]["one_fifth_rule"] = {fmt: one_fifth_rule(report["placement"][tag], fmt)
                                                          for fmt in ("yes_no", "two_candidate")}
    if wave != "A":
        # review 1, S8: Qwen3.8 27B with preserve_thinking=False, read against its own run by the one-fifth rule
        report["template_checks"] = {}
        for tag, run in (cfg.get("template_checks") or {}).items():
            folders, problems = complete_shards(run)
            if problems or tag not in scores:
                report["waiting"][f"template check {tag}"] = problems or [f"family_score_{tag} not read"]
                continue
            read[run] = folders
            alt = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag + NO_PRESERVE]
            block = placement_block(scores[tag], alt, primary)
            report["template_checks"][tag] = {**block, "run": run, "variant": "preserve_thinking=False",
                                              "tokenization_gate": _gate_records(folders),
                                              "one_fifth_rule": {fmt: one_fifth_rule(block, fmt)
                                                                 for fmt in ("yes_no", "two_candidate")}}
        report["b3_fallback"] = b3_fallback(report["models"], scores, primary, successes)
    from wf.provenance import provenance, shard_inputs

    report["provenance"] = provenance(
        runs={**{s: Path(s) for s in sources}, **{k: (v[0].parent if v else Path(k)) for k, v in read.items()}},
        inputs=[Path(s) / "transitions.jsonl" for s in sources] + shard_inputs([p for fs in read.values() for p in fs]))
    # wave B is read only when the serve2 rescorings pass (section 8 amendment), whatever the environment check said
    blocked = wave != "A" and not report["serve2_rescoring"]["pass"]
    if wave != "A":
        report["amendments"] = {"review": "notes/families_waveb_review1.md", "pair_caveat": PAIR_CAVEAT,
                                "qwen3_size_series": QWEN3_SIZE_SERIES, "b1_readings": B1_READINGS,
                                "b3_fallback": B3_FALLBACK, "run_order": list(RUN_ORDER_B),
                                "serve2_rescoring": RESCORE_AMENDMENT}
        report["predictions_b"] = waveb_predictions(report, bool(report["serve2_rescoring"]["pass"]))
        if blocked:  # review 1, S5: no wave B score is read, in the metrics, the report or the status lines
            withhold_waveb_readings(report, list(scores))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, f"Model families, wave {wave} (exploratory)", report)
    def head(m: dict) -> str:
        pm = m["primary_metrics"]
        return (f"{'interpretable' if m['competence']['interpretable'] else 'NOT INTERPRETABLE'}; share toward Yes "
                f"{pm['item_share_toward_yes']['estimate']:.3f} (primary), R {pm['goal_to_evidence_ratio']['estimate']:.3f} "
                f"(secondary); "
                f"yn dL {m['yes_no']['goal_effect']['delta_L_nats']:.2f}, two-candidate dL {m['goal_effect']['delta_L_nats']:.2f}")

    if blocked:
        lines = [f"{tag}: {ENVIRONMENT_BLOCKED}; scores complete, nothing read" for tag in report["models"]]
        lines += [f"{tag} (reference): {head(m)}" for tag, m in report["reference"].items()]
        lines += [f"primary family: every test {ENVIRONMENT_BLOCKED} (serve2 rescoring of the older models not "
                  f"complete or not passed: {sorted(t for t, m in report['serve2_rescoring']['models'].items() if not m['pass']) + report['serve2_rescoring']['missing_from_config']})"]
    else:
        lines = [f"{tag}: {head(m)}; prediction holds: {report['prediction_check'][tag]['holds']}" for tag, m in report["models"].items()]
        lines += [f"{tag} (reference): {head(m)}" for tag, m in report["reference"].items()]
        lines += [f"primary family {k}: {v['verdict']} (Holm p {v['p_holm']:.4f})" for k, v in report["primary_family"]["tests"].items()]
        lines += [f"{k}: item share {v}" for k, v in report["primary_family"]["comparison_readings"].items()]
        lines += [f"{tag} placement, first turn minus native: yn {p['yes_no']['first_turn_minus_native']['delta_L_nats']:.2f}, "
                  f"two-candidate {p['two_candidate']['first_turn_minus_native']['delta_L_nats']:.2f} ({p['n_items']} items)"
                  for tag, p in report["placement"].items()]
        if wave != "A":
            lines += [f"{tag} preserve_thinking=False minus native: yn {p['yes_no']['first_turn_minus_native']['delta_L_nats']:.2f}, "
                      f"two-candidate {p['two_candidate']['first_turn_minus_native']['delta_L_nats']:.2f}; format-dependent "
                      f"{ {f: r['format_dependent'] for f, r in p['one_fifth_rule'].items()} }"
                      for tag, p in report["template_checks"].items()]
            lines += [f"B1 {tag}: {v['reading']}" for tag, v in report["predictions_b"]["B1"].items() if v["reading"]]
            lines += [f"(generation pairs) {PAIR_CAVEAT}"]
    ok = not report["waiting"]
    write_status(run_id, "\n".join(lines) + (f"\nwaiting: {sorted(report['waiting'])}" if not ok else ""), ok=False,
                 reported="families report" if ok else "partial families report")
    return 0
