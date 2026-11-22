# Model families, wave B: design (logged 2026-11-22, before any wave B score)

Exploratory, not preregistered as a confirmatory test. Written and committed before any wave B code and before any wave B model was loaded or scored. No wave B weights have been downloaded; only tokenizer and configuration files (and safetensors headers, read over HTTP) were fetched, to check templates and loading. The design is frozen at an annotated git tag after an independent review; no scoring job (the environment check included) is submitted before that tag exists.

Wave A's design ([families_design.md](families_design.md)) is reused wherever it applies: the same items, prompts, conditions, scorer, tokenization gate, metrics, competence gate, comparison tests and Holm correction. This note states only what wave B adds or changes, and why.

## 1. Status at the start of this note

What existed on 2026-11-22, before this note:

- **Wave A** is scored and analysed (Gemma-2-9B-it, Mistral-7B-Instruct-v0.3, Llama-3.2-1B/3B-Instruct; [families_results.md](families_results.md)). Its runs are on ICE in `text-wm-fix23/runs/family_score_<tag>` (identical copies in `text-wm-fix21`). Wave A's note ended with a short "Wave B (planned; not built)" paragraph.
- **A second software environment** exists on ICE: `/storage/ice1/2/1/gmat3/envs/serve2`, built 2026-11-20 by `envs/build_serve2.sbatch` with uv. Python 3.11.18, torch 2.14.1+cu126, transformers 5.23.0, tokenizers 0.23.3, huggingface-hub 2.4.0, accelerate 1.16.0, jinja2 3.1.6, mistral-common 1.13.1, sentencepiece 0.2.3, protobuf 7.36.2, safetensors 0.8.0, numpy 2.3.5, scipy 1.17.1; the full list (80 packages) is `envs/serve2_freeze.txt`. It passed a GPU smoke test (Qwen3-0.6B at its pinned revision loaded and scored one sentence on an L40S, job 7195366). The freeze file has exact versions but no hashes, so it is not the hash-locked requirements file wave A's note promised; that is recorded as a gap (section 4), not fixed by rebuilding, since rebuilding could change the environment.
- **Nothing else for wave B existed**: no model specs, pinned revisions, configs, job scripts, tests or code anywhere in the repository (every branch was searched), and no wave B weights in the model cache.

## 2. Question

Wave A found the goal effect, the share of determined failures that a true task line pushes toward "Yes", in every interpretable 2024 instruct model tested, from four developers. The newest model scored so far was released in April 2025. Wave B asks two questions:

1. **Persistence.** Is the effect present in current-generation (December 2025 to August 2026) open models, small and large?
2. **Post-training.** Does post-training explain it? The direct test available is one base model against its instruct model, scored in one shared plain-text format.

A third, descriptive reading pairs each new model with the older model of its line (generation pairs). All of these are descriptive comparisons of whole models: developer, data, size, tokenizer, template and training all change together, so no comparison names a cause.

## 3. Models and pinned revisions

| Tag | Hugging Face id | Pinned revision | Release (Hub repo created) | Parameters: total / language model | Format | Why |
|---|---|---|---|---|---|---|
| `ministral3-8b-it` | mistralai/Ministral-3-8B-Instruct-2512-BF16 | f6fae9795746 | Dec 2025 (2025-10-31) | 8.92B / 8.49B | chat | newer half of the Mistral pair (wave A: Mistral 7B v0.3) |
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | c20223623576 | Feb 2026 (2026-02-27) | 9.41B / 8.95B | chat | newer half of the Qwen pair (Qwen3-8B) |
| `qwen3.5-9b-plain` | Qwen/Qwen3.5-9B (same weights) | c20223623576 | as above | as above | plain | the instruct side of base against instruct |
| `qwen3.5-9b-base-plain` | Qwen/Qwen3.5-9B-Base | 68c46c4b3498 | Feb 2026 (2026-02-26) | 9.41B / 8.95B | plain | the base side of base against instruct |
| `gemma4-e4b-it` | google/gemma-4-E4B-it | 4ecc2ba63bca | Mar 2026 (2026-03-02) | 7.94B / 7.46B (3.49B of it per-layer embeddings; about 4B "effective") | chat | newer half of the Gemma pair (Gemma 2 9B) |
| `qwen3.8-27b` | Qwen/Qwen3.8-27B | 37561816e66b | Aug 2026 (2026-08-05) | 27.36B / 26.90B | chat | newest model available; newer half of a large Qwen pair (Qwen3-32B, scored in the size series) |
| `gemma4-31b-it` | google/gemma-4-31B-it | 842da3794eaa | Mar 2026 (2026-03-11) | 31.27B / 30.70B | chat | current-generation large model from a second developer |

How these were recorded:

- **Revisions:** the Hub's current commit (`sha` from `https://huggingface.co/api/models/<id>`), read on 2026-11-22, before any download. The full 40-character hashes go into `wf.constants.REVISIONS`, and every job loads exactly those.
- **Release:** the month of the public release; the date in brackets is the Hub repository's creation date, which can precede the release.
- **Parameters:** the total counts every tensor of the checkpoint's architecture (the vision and, for Gemma 4 E4B, audio encoders included); the language-model count is the part that reads text. Counted by instantiating each architecture on PyTorch's meta device from its pinned config (no weights), transformers 5.23.0.
- **Access:** none of the seven repositories is gated, so no token is needed.

**Why the BF16 Ministral repository.** `mistralai/Ministral-3-8B-Instruct-2512` ships its language model in FP8 (7.42B of its parameters are `F8_E4M3`). Every other model in the study is scored in bf16. Mistral publishes the same instruct model in bf16 as `Ministral-3-8B-Instruct-2512-BF16` (same chat template, byte for byte). Scoring the FP8 weights would add a quantization difference to the comparison with Mistral 7B, so the BF16 repository is used.

**Models considered and left out.** Ministral-3-8B-Base and Ministral-3-8B-Reasoning, Gemma-4-E4B (base), Gemma-4-12B-it, the Qwen3.8 FP8 and mixture-of-experts releases, and the OLMo 3 post-training ladder are not part of this wave. One base-against-instruct pair is scored first; whether to add Ministral 3 or Gemma 4 base pairs, or the OLMo ladder, is decided after wave B's results, as a new design. A frontier API check is deferred and uses no paid API here.

**Reference models** (already scored with exactly the same prompts; read, not rescored): Qwen3-8B (`size_score_qwen3-8b`), Llama-3.1-8B (`rq1_score`, `rq1x_score`, `rq1v_score`), Qwen3-32B (`size_score_qwen3-32b`), and the wave A runs of Gemma 2 9B and Mistral 7B v0.3 (`family_score_<tag>`).

## 4. Software environment

Every wave B model needs transformers 5 (Gemma 4, Qwen3.5 and Ministral 3 are not in 4.51.3), so wave B runs in `serve2`. Wave A, the confirmatory runs and every earlier run keep their environment (`SERVE_ENV`, transformers 4.51.3, torch 2.6.0). A scoring job selects serve2 explicitly (`pace/family_score_serve2.sbatch`) and refuses to start if the Python it runs has transformers below 5; a wave A or reference model is scored there only by the environment check (section 8).

Six properties of transformers 5.23 change how the scorer must be set up. Each was found by reading the serve2 source or by running the pinned tokenizers on ICE (CPU, 2026-11-22):

1. **The cache folder.** transformers 4.51.3 reads models from `TRANSFORMERS_CACHE`, which `pace/_common.sh` sets to `hfcache/`. transformers 5.23 no longer reads that variable; it reads `HF_HUB_CACHE`, which `_common.sh` sets to `hfcache/hub/`. Unfixed, a serve2 job would not find the wave A models (they are in `hfcache/`) and would download them again into `hfcache/hub/` on GPU time (14 GB for Mistral 7B), and the prefetch tool, which asks for `TRANSFORMERS_CACHE`, would put wave B weights where the scoring job does not look. So serve2 jobs set `HF_HUB_CACHE` to the folder transformers 4.51.3 reads (one cache for both environments), the prefetch tool asks the installed library which folder it reads, and scoring runs with `HF_HUB_OFFLINE=1`, so a missing file stops the job instead of downloading.
2. **Mistral tokenizers.** With mistral-common installed, transformers 5 loads Ministral 3's tokenizer as `MistralCommonBackend`, which ignores the repository's chat template and encodes requests with mistral-common. It refuses a conversation that ends with the assistant's turn, which the tokenization gate renders to find the reply's tokens. So every serve2 tokenizer is loaded with `mistral_format=False`: the fast tokenizer from `tokenizer.json` and the pinned chat template, as in wave A. For Ministral 3, the gate also encodes every scored prompt with mistral-common and requires identical token ids (section 7). On the golden item all six prompts (three tasks, both formats) are identical.
3. **Model classes.** The wave B checkpoints are saved as vision-language (and, for Gemma 4 E4B, audio) models: `Mistral3ForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`, `Gemma4ForConditionalGeneration`. `AutoModelForCausalLM` refuses Mistral 3's configuration and maps Qwen3.5 to a text-only class. Each wave B model is loaded with `AutoModelForImageTextToText`, which resolves to exactly the architecture named in the checkpoint's `config.json` (checked on the meta device for all six). Only text is passed in; the image and audio encoders are loaded and unused.
4. **A loading check.** The checkpoints' tensor names do not all match the classes' parameter names: Ministral 3's checkpoint uses the older `language_model.model.*` layout, renamed on load by transformers' conversion mapping; Qwen3.5 and 3.8 carry 15 multi-token-prediction tensors (`mtp.*`) that the class does not use; Gemma 4 E4B carries the key and value projections and key norms of its last 18 layers (54 tensors), which its key-value sharing (`num_kv_shared_layers` = 18) does not use; both Gemma 4 models tie the output layer to the embeddings. A renaming that silently failed would leave layers at their random initialization. So wave B models load with `output_loading_info=True`, and the scorer refuses to continue if any parameter is missing or has a mismatched shape, if an unexpected checkpoint tensor falls outside that model's pre-registered pattern (`mtp\.` for the Qwen models; the 18 shared layers' `k_proj`, `v_proj`, `k_norm` for Gemma 4 E4B; none for the others), or if any module was placed on the CPU or disk rather than a GPU. The loading record is saved with the shard.
5. **Downloaded kernels.** Qwen3.5's linear-attention layers (gated DeltaNet) call `use_kernel_func_from_hub_with_fallback`: if the `kernels` package is installed, transformers may fetch compiled kernels from the Hub at run time; otherwise it runs its own PyTorch implementation. serve2 has neither `kernels`, `flash-linear-attention` nor `causal-conv1d`, so the PyTorch path runs. A wave B job refuses to start if `kernels` is importable (unpinned code fetched at run time), and records whether the other two are present.
6. **`torch_dtype` is deprecated** in favour of `dtype`; serve2 loads with `dtype=torch.bfloat16`. Nothing else in the precision changes: bf16 weights, log-softmax in float32.

**Attention.** All wave B text models load with transformers' default, SDPA, recorded per shard. Gemma 4 has no attention-logit soft-capping (it uses query and key normalisation), so wave A's reason for eager attention on Gemma 2 does not apply; Gemma 4's final logits are tanh-capped at 30, like Gemma 2's (wave A note, "Precision and attention"), so its per-token log-odds are bounded at 60 nats. Gemma 4's sliding windows (512 tokens for E4B, 1,024 for 31B) are shorter than the longest prompts (about 1,200 tokens), so they are in play, as they are in the deployed model. Qwen3.5 and 3.8 use multimodal rotary positions; with text only, the three position sections are the plain token positions.

**Recorded per shard:** transformers, torch, tokenizers, huggingface-hub and Python versions, whether `kernels`, `fla` and `causal_conv1d` are importable, the attention implementation and dtype loaded, the device map, and the loading record.

## 5. Prompts and chat templates

**Prompt and conditions:** exactly wave A's (the size series' RQ2 few-shot prompt with example set 1, history only; aligned, swapped and neutral task lines; the two-candidate score and the yes/no score; the 4,666 primary items and 292 determined successes; the same arms per item). Every wave B model gets the system text (`WM_SYSTEM`) as a system message, since every wave B template takes one natively, and every template is used at its pinned revision.

Checked on ICE with transformers 5.23.0 and each pinned tokenizer, on the golden item (`tests/golden/default_prompt.json`), all six prompts per model:

**Ministral 3 8B Instruct.** `<s>[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]reply</s>`. The system text opens the prompt, unlike Mistral 7B v0.3's template, which put it in the last user turn. When a conversation has no system message the template inserts Mistral's own default system prompt (with today's date); ours always has one, so neither the default prompt nor a date ever appears (the gate checks that "Le Chat" and "The current date" are absent). A reply follows `[/INST]` directly, with no space, so a candidate is scored as `You …` (tokens `You`, ` put`, …). There is no thinking mode; `enable_thinking` is ignored (the render is identical either way, checked).

**Qwen3.5 9B (instruct) and Qwen3.8 27B.** ChatML (`<|im_start|>role\n…<|im_end|>`). **Thinking is on by default**: with no setting, the generation prompt ends `<think>\n` and the model would reason first. With `enable_thinking=False` it ends with an empty think block, `<think>\n\n</think>\n\n`, as Qwen3-8B's did in every earlier run. Qwen3.8's template, when thinking is on, also writes a reasoning-effort instruction ("Reasoning effort is set to xhigh…") into the system message; with thinking off it writes none. Qwen3.8 also writes an empty think block into every earlier assistant turn by default (`preserve_thinking` unset counts as true), so its few-shot example replies read `<think>\n\n</think>\n\n<observation>`, where Qwen3.5's read `<observation>`; this is the developer's default rendering and is left as shipped, and noted as a format difference within the Qwen line.

**Gemma 4 E4B-it and 31B-it.** `<bos><|turn>system\n…<turn|>\n<|turn>user\n…<turn|>\n<|turn>model\n`. Gemma 4 takes a system role (Gemma 2 did not; wave A folded its system text into the first user turn). **Thinking is off by default** (`enable_thinking` defaults to false); with it on, the template writes `<|think|>` at the top of the system turn. With thinking off, the 31B template ends the generation prompt with an empty thought channel, `<|turn>model\n<|channel>thought\n<channel|>`; the E4B template does not. The empty channel is what the 31B model sees at inference with thinking off, so the score is read after it. The template does not write that channel into a rendered reply, so the gate's reply check (wave A: "the conversation with the reply begins with the prompt before it") fails for 31B as written; the gate is extended to take a pre-registered generation suffix per model (section 7). Both templates trim each turn's text.

**Thinking off, and how it is enforced.** Thinking is off for every model, as for Qwen3-8B in the earlier runs; a reasoning trace before the answer would change what is being scored (the thinking-mode result showed reasoning cuts Qwen3-8B's goal effect by about a third). It is enforced three ways:

1. The scorer passes `enable_thinking=False` to every template call, as it always has (`LogoddsScorer.template_settings`).
2. The gate checks, for every model with a thinking switch, that the template reads it: the render with thinking on differs from the render with thinking off and contains the model's thinking marker (`<think>\n` at the end for the Qwen models, plus "Reasoning effort" for Qwen3.8; `<|think|>` for Gemma 4), and the render with thinking off does not.
3. The gate checks that every scored prompt's token ids end with the model's pre-registered thinking-off ending: `<think>\n\n</think>\n\n` (Qwen), `<|turn>model\n` (Gemma 4 E4B), `<|turn>model\n<|channel>thought\n<channel|>` (Gemma 4 31B), `[/INST]` (Ministral 3).

Scoring is teacher-forced: the model never generates, so it cannot start a reasoning trace of its own; the only way thinking could enter is through the prompt, which these checks pin.

## 6. Base against instruct: one plain-text format

A base model has no chat training; scoring it through a chat template would compare "template against no template" as much as "base against instruct". So the comparison uses one plain-text format for both models, and the instruct model is scored twice: in its chat template (for every other comparison) and in the plain format (for this one). Both runs use the same weights and revision; only the prompt format differs.

**The plain format.** The same messages, joined as text: the system text, a blank line, then each example as its user text, one space and its reply, followed by a blank line, and finally the query's user text:

```
{system}\n\n{user 1} {reply 1}\n\n{user 2} {reply 2}\n\n{user 3} {reply 3}\n\n{query}
```

Each user text ends with `Next observation:` (two-candidate prompts) or `Answer:` (yes/no prompts), so the examples read `Next observation: You arrive at …`. The candidate is scored with one leading space, as the examples write a reply: ` You put …` against ` Nothing happens.`, and ` Yes` against ` No`. Special tokens are added only if the tokenizer adds them by itself (Qwen3.5's adds none: it has no beginning-of-sequence token). Thinking does not arise: no template is applied.

**Its gate.** In the plain format the gate requires, for every candidate after its aligned prompt and for ` Yes` and ` No` after every yes/no prompt, that tokenizing the prompt and the candidate together gives exactly the prompt's tokens followed by the candidate's tokens (no merge across the boundary), that ` Yes` and ` No` are one token each and different, and that the two candidates' first tokens differ.

**Caveats.** The yes/no examples show observations, not Yes/No answers, in both formats; an instruct model follows the question's instruction, a base model may put little probability on ` Yes` and ` No`. The competence gate's coverage condition (median P(Yes) + P(No) at least 0.5 under the neutral task) decides whether the base model is interpretable; a base model that fails is "not interpretable", not a null. Qwen3.5's base model may have seen instruction-style data late in pretraining (not documented either way); the comparison is "the released base checkpoint against the released instruct checkpoint", not "no instruction exposure against instruction tuning".

## 7. Tokenization and template gate

Wave A's gate (its note, "Tokenization gate") runs unchanged on every item of every shard, through the scorer's own encoding, before any score and again on every resume: candidates match the template's reply; "Yes" and "No" are single tokens equal to the template's reply after every yes/no prompt; the two candidates' first tokens differ. Wave B adds:

- **A generation suffix.** For a model whose generation prompt ends in text that a rendered reply does not contain (Gemma 4 31B: `<|channel>thought\n<channel|>`), the reply check strips exactly that suffix from the prompt before comparing, and requires the prompt to end with it. Every other wave B model, and every wave A model, has none.
- **Thinking checks** (section 5): the switch is read, the marker is absent with thinking off, and every scored prompt ends with the model's thinking-off ending.
- **The plain-format check** (section 6) replaces the reply check for plain-format runs.
- **Mistral-common agreement** (Ministral 3 only): every scored prompt's ids from the chat template equal mistral-common's encoding of the same messages.
- **The record** adds the environment versions and the loading record (section 4).

**Run on ICE before the freeze (CPU, tokenizers only).** The gate is run on all 4,958 items for every wave B model (`pace/families_waveb_gate.sbatch`) and its result is added to this note before the freeze. The golden renders (transformers 5.23.0) go into `tests/golden/families_waveb_rendered.json`, as wave A's did.

## 8. Environment-equivalence gate (before any wave B score is read)

**Why.** Wave B is scored with different software from every earlier run (transformers 5.23 and torch 2.14 against 4.51.3 and 2.6.0). If the environment moved the numbers, a difference between a wave B model and an earlier model could be the software, not the model. Before any wave B score is read, two already-scored models are rescored in serve2 and compared with their earlier scores.

**Models.** Mistral 7B v0.3 (wave A; its tokenizer goes through a different path in transformers 5, as a fast tokenizer read from `tokenizer.json` instead of one rebuilt from sentencepiece, so it is the riskiest of the earlier models) and Qwen3-8B (the study's main model, from `size_score_qwen3-8b`).

**Part 1, tokens (CPU, every item).** Every prompt the scoring covers (all 4,958 items, every arm) and every candidate, plus "Yes" and "No", is encoded through the scorer's own encoding in serve2 and in a copy of the study environment's tokenizer stack (transformers 4.51.3, tokenizers 0.21.4, torch 2.6.0 CPU, built in a job's node-local temporary folder by `pace/families_envcheck_tokens.sbatch`, so the study environment itself is not touched). The token ids must be identical for every prompt and candidate (compared as SHA-256 hashes of the id lists, with lengths).

**Part 2, scores (GPU, 300 items).** Each model scores the 300 seeded primary items of wave A's placement check (`placement_items`), under all three tasks in both formats (2,700 forward passes per model, about 20 GPU-minutes), on the GPU type of its earlier run (Mistral 7B: A40; Qwen3-8B: L40S), so only the software differs. The scores are saved under the tag `<model>-serve2`. `families_envcheck` compares each (item, condition) score with the earlier one.

**Pass rule (fixed now).** The thresholds come from the archived Qwen3-8B runs, where the same prompts scored on different GPU types differed by 0.18 nats per item on average in two-candidate L (0.035 in yes/no L) and moved ΔL by about 0.02 nats (wave A note, "Amendment"); they allow about twice that per score, and a fraction of the 0.3-nat noise floor for effects. For each model, all of:

- yes/no: the mean absolute difference per score is at most 0.07 nats; the sample's goal effect (mean aligned minus swapped) differs by at most 0.10 nats; the item share toward Yes differs by at most 0.02 (6 of 300 items);
- two-candidate: the mean absolute difference per score is at most 0.35 nats; the goal effect differs by at most 0.10 nats;
- every one of the 1,800 (item, condition) scores is present in both runs, and part 1 passed.

Also reported: the largest difference, the share of scores identical to the last bit, and the differences by condition.

**If it fails.** No wave B score is read. The cause is found first (an attention or kernel change, a tokenizer difference, a dtype path) and fixed or documented in an amendment made before any wave B score is read; the check is rerun. `families_waveb_analysis` reads the check's verdict and marks every wave B test "not interpretable (environment check)" unless it passed.

## 9. Measures and analysis

Everything in wave A's "Measures" section and its three amendments applies unchanged: the size series' block per model; the item share toward Yes (primary), R, ΔL, the standardised effect d and the two-candidate format (secondary); argmax answers and flips; first-token coverage; one (item, condition) population for every model; game-clustered bootstrap CIs and sign-flip permutation p-values; paired comparisons on shared items with the difference test and the ±0.05 equivalence test (TOST); readings "differs", "matches", "differs, within the margin", "inconclusive (no difference detected)"; the 0.3-nat noise floor for ΔL differences.

**Competence gate:** wave A's rule as amended (under the neutral task, yes/no format: AUC at least 0.70; Yes to at most 95% of failures; Yes to at least 5% of determined successes; median P(Yes) + P(No) at least 0.5). A model that fails is "not interpretable"; a comparison involving one is "not interpretable".

**The primary family (fixed now), Holm-corrected at α = 0.05 over 16 tests:**

- **Item share per model (6 tests):** `ministral3-8b-it`, `qwen3.5-9b`, `gemma4-e4b-it`, `qwen3.8-27b`, `gemma4-31b-it` (chat format) and `qwen3.5-9b-base-plain` (plain format). Supported when the Holm p is below 0.05, the share is above one half and the model passed the competence gate.
- **Planned comparisons of item share (5 comparisons, two tests each):** the first named minus the second.
  - `gemma4-e4b-it` against `gemma2-9b-it` (Gemma generation pair);
  - `ministral3-8b-it` against `mistral-7b-v0.3` (Mistral generation pair);
  - `qwen3.5-9b` against `qwen3-8b` (Qwen generation pair, about 8–9B);
  - `qwen3.8-27b` against `qwen3-32b` (Qwen generation pair, about 30B);
  - `qwen3.5-9b-plain` against `qwen3.5-9b-base-plain` (instruct minus base, one format).

As in wave A, a test whose model is still waiting, or whose p is missing, enters with p = 1, so the family is never smaller than 16. Everything else is descriptive: R, ΔL, d, the two-candidate format, every wave B model against the two reference models (Qwen3-8B, Llama-3.1-8B), the plain-format instruct run's own share, chat against plain for the same instruct model (the format effect), Gemma 4 31B against E4B and Qwen3.8 27B against Qwen3.5 9B (size and generation within a line), and the placement check below.

**Exploratory placement check, Gemma 4 E4B.** The Gemma generation pair also differs in where the system text sits: Gemma 2 had it folded into the first user turn, Gemma 4 has a system turn. As wave A did for Mistral, Gemma 4 E4B rescores the same 300 seeded items under the aligned and swapped tasks in both formats with the system text folded into the first user turn (`family_placement_gemma4-e4b-it`, tag `gemma4-e4b-it-firstturn`; 1,800 passes). Read by wave A's rule: if the difference's CI excludes zero and it is larger than a fifth of the native effect, the Gemma pair is reported as format-dependent, with both numbers.

## 10. Predictions (logged before any wave B score)

- **B1, persistence.** The goal effect (aligned minus swapped) is positive in both formats, read as the 95% game-clustered CI above zero, for every wave B model and format run (all seven tags), and the item share is supported in the primary family for every wave B model that passes the competence gate. Basis (corrected after review 1): every interpretable model scored so far (six, four developers, 1B to 32B) shows a positive log-odds effect, but not every one has a share well above one half. Qwen3-4B is competent (yes/no AUC 0.94, Yes on 2.2% of failures) yet its share is 0.49 with a yes/no effect of 2.27 nats: the effect sits on a minority of items. Within the Qwen3 line the share runs 0.49 / 0.89 / 0.70 / 0.71 at 4B / 8B / 14B / 32B (recomputed from `size_score_qwen3-<size>`, 4,666 failures each). So the share half of B1 has a known failure case among competent models.
- **B2, generation pairs.** For each of the four generation pairs, the newer model does not move fewer failures: the comparison does not read "differs" with a negative difference. Basis: hypothesis 2 of the post-training account (more agentic and reinforcement-learning post-training, larger goal-congruent shift); recent Qwen and Gemma releases describe heavier agentic post-training than their predecessors. Not predicted: matches against differences, or sizes. B2 can fail only on a clear negative difference: "inconclusive" and "matches" count as holding, and equivalence within ±0.05 is nearly unreachable at wave A's interval widths (amendment, section 18).
- **B3, post-training.** In the plain format, the instruct model moves more failures than its base model: the comparison reads "differs" (or "differs, within the margin") with a positive difference. Basis: hypothesis 1 (base models show a smaller goal-congruent shift). Not predicted: whether the base model passes the competence gate. If the base model fails only the yes/no coverage criterion, B3 is read on the two-candidate item share instead (section 18); if it fails otherwise, B3 is untestable, not refuted.

Not predicted: the ordering across families, the size of any effect, or the format effect.

## 11. Outcome table (pre-data)

| Pattern | Reading |
|---|---|
| The environment check fails | No wave B result is read until the cause is fixed and documented, before data. The analysis writes no wave B number (amendment, section 18). |
| A model fails the competence gate | Not interpretable; no statement about its goal sensitivity in either direction. |
| Every interpretable wave B model's share supported | The goal-congruent shift persists in current-generation open models from three developers, at about 8B and about 30B. |
| A wave B model's share not supported, its yes/no effect's CI above zero | Effect carried by a minority of items (as Qwen3-4B): the task line moves the log-odds on average, but fewer than half of the failures, or not detectably more than half, move toward Yes. |
| A wave B model's share not supported and its yes/no effect's CI covering zero | No detectable effect in that model: a current model without a detectable goal-congruent shift, the first found; followed up before any general claim. |
| A generation pair reads "differs" with the newer model higher | The newer release moves more failures. Not a generation effect: the pair also changes size, and within the Qwen3 line alone the share runs 0.49 / 0.89 / 0.70 / 0.71 across 4B / 8B / 14B / 32B. |
| A generation pair reads "differs" with the newer model lower | B2 fails for that line: the newer release moves fewer failures. Not a generation effect, for the same reason (size and everything else change too). |
| A pair "matches" or is "inconclusive" | Similar or not distinguishable; "inconclusive" is never read as "the same". |
| Instruct minus base "differs", positive | Post-training raises the share in this pair, in one plain format: consistent with hypothesis 1, one pair only. |
| Instruct minus base "matches" or negative | The goal-congruent shift is already present in the base model at the instruct model's level: post-training is not needed for it in this pair. |
| Base model fails only the yes/no coverage criterion | B3 is read on the two-candidate item share, instruct minus base (section 18). |
| Base model not interpretable otherwise | B3 untestable in this format; reported as such. |
| Chat against plain (same instruct model) differs widely | Format matters for this model; the chat-format comparisons are read as format-specific. |
| Gemma 4 placement moves the effect (the one-fifth rule) | The Gemma pair is format-dependent; both numbers reported. |
| Qwen3.8 without the examples' think blocks moves the effect (the one-fifth rule) | The large Qwen pair is format-dependent; both numbers reported (section 18). |

Every reading above is of a goal-congruent shift: the true task line moving predictions toward the outcome that would serve the task. The item share counts how many failures move; it does not separate wishful prediction from a judgement that the task makes an action relevant, and no row reads it that way.

## 12. Cost and GPU memory

Wave A's timings (one model per job, 43,746 forward passes): Mistral 7B 1 h 43 min on an A40, Gemma 2 9B with eager attention 1 h 47 min on an L40S; Qwen3-8B about 1 h on L40S (two shards); Qwen3-32B about 3 h 50 min on an A100-80GB (two shards). Estimates for wave B, not timings (Qwen's linear-attention layers run in PyTorch here, without fused kernels, so the Qwen estimates are the least certain):

| Run | Passes | GPU memory (bf16 weights) | GPU | Estimate |
|---|---|---|---|---|
| Environment check, Mistral 7B and Qwen3-8B | 2,700 each | 14.5 and 16.4 GB | A40, L40S (their earlier types) | about 0.3 h each |
| `ministral3-8b-it` | 43,746 | 17.9 GB | L40S, A40 or A100 | 1.5–2 h |
| `qwen3.5-9b` | 43,746 | 19.3 GB | L40S, A40 or A100 | 1.5–3 h |
| `qwen3.5-9b-plain` | 43,746 | 19.3 GB | same | 1.5–3 h |
| `qwen3.5-9b-base-plain` | 43,746 | 19.3 GB | same | 1.5–3 h |
| `gemma4-e4b-it` | 43,746 | 16.0 GB | same | 1–2 h |
| Gemma 4 E4B placement check | 1,800 | 16.0 GB | same | 0.2 h |
| `qwen3.8-27b` | 43,746 | 55.6 GB | A100-80GB (or two L40S) | 4–6 h |
| `gemma4-31b-it` | 43,746 | 62.6 GB | A100-80GB (or two L40S) | 4–6 h |

Total about 16 to 26 GPU-hours, of which about 8 to 12 on 80 GB cards. With node-local weights (section 13), add about 0.2–0.4 h per job for the download.

**The two large models** do not fit a 40 or 48 GB card in bf16. The partition this account uses (coc-gpu) has two A100-80GB nodes with two cards each; H100 and H200 nodes are in another partition. The scorer loads with `device_map="auto"`, so it can also shard a model over two GPUs of one node (two L40S, 96 GB); the loading check refuses a CPU or disk placement. Each large model's job asks for one A100-80GB by default; two L40S are the fallback, chosen at submission and recorded in the shard's GPU record (a resume must use the same type, as in wave A).

## 13. Storage (decided after review 1: route A, node-local weights)

Measured on 2026-11-22: the scratch quota is 300 GB, with 248.9 GB used (`lfs quota`), so **51.1 GB is free**; home has 20.3 GB free of 30 GB and is not used for models. The model cache `hfcache/` holds 190 GB (wave A, the size series, the reference and fine-tuned models, duplicate `hub/` copies used by vLLM). Nothing is deleted by this design.

Wave B's downloads (the files transformers loads, `consolidated.safetensors` excluded): Ministral 3 17.9 GB, Qwen3.5-9B 19.3 GB, Qwen3.5-9B-Base 19.3 GB, Gemma 4 E4B 16.0 GB, Qwen3.8-27B 55.6 GB, Gemma 4 31B 62.6 GB: **190.7 GB in all.** They cannot all be cached at once, and each of the two large models alone is larger than the free space.

Two ways to run without deleting anything:

- **A. Node-local weights (proposed).** The scoring job downloads the pinned snapshot into the GPU node's local disk (1.7 TB on the A100-80GB nodes, 11 TB on the L40S nodes), scores from there, and the copy disappears when the job ends (`WF_WEIGHTS=local` in `pace/family_score_serve2.sbatch`). No quota is used. Costs: the download runs on GPU time (an estimated 10–20 minutes for the large models), and a resumed shard downloads again. The snapshot's revision is checked before loading, as the prefetch tool checks it.
- **B. Shared cache, one model at a time.** Prefetch one model into `hfcache/` (`pace/families_prefetch.sbatch` with serve2), score it, then delete that model's weights before the next. The four small models fit two at a time. The two large ones do not fit at all without first freeing about 5–12 GB plus margin (for example by deleting a scored wave B model's weights, the size series' Qwen3-14B (28 GB), or asking PACE for more quota). Deleting anything needs the user's approval.

The tokenizer and configuration files of all six repositories (about 0.2 GB) are already in `hfcache/` at the pinned revisions; they are what the CPU checks used.

## 14. Running order (after the freeze)

1. CPU: the wave B gate on all items (already run before the freeze, section 7), and part 1 of the environment check.
2. GPU: part 2 of the environment check (Mistral 7B on an A40, Qwen3-8B on an L40S); then `families_envcheck` (CPU). Stop here if it fails.
3. GPU, in this order (pre-registered after review 1, `wf.families.RUN_ORDER_B`), each with node-local weights (route A): the four small models (Ministral 3; Qwen3.5 9B in chat and plain format; Qwen3.5 9B base; Gemma 4 E4B, then its placement check), then Qwen3.8 27B followed by its preserve_thinking check, then Gemma 4 31B last.
4. CPU: `families_waveb_analysis`.

## 15. Build (code written after this note's first commit)

- `wf/constants.py`: the six pinned revisions.
- `wf/families.py`: `ModelSpec`'s wave B fields (recorded only where they differ from wave A's defaults, so wave A's saved configs are unchanged); `WAVE_B`, `WAVE_B_SHARE_TESTS`, `WAVE_B_PLANNED`, `PREDICTION_B`; `spec_for`, `environment_record`, `require_environment`, `open_scorer`; the gate's generation suffix, `waveb_prompt_checks`, `plain_checks` and `mistral_common_encoder`; `run_family_envcheck`, `envcheck_block`, `run_families_envcheck` (`ENVCHECK_RULE`); `apply_environment_gate`, `waveb_predictions`, and `run_families_analysis` with `wave: B`.
- `wf/scoring/logodds.py`: `model_class`, `tokenizer_kwargs`, `prompt_format`, `strict_load` and `check_loading`; `dtype` under transformers 5. Every default leaves earlier runs' loading and prompts unchanged.
- `wf/scoring/prompts.py`: `plain_text_prompt`.
- `wf/tools/family_gate_cpu.py` (the gate on every item, tokenizers only), `wf/tools/envcheck_tokens.py` (part 1 of the environment check), `wf/tools/prefetch_models.py` (wave B tags; the cache folder transformers 5 reads).
- Runs and configs: `family_score_<tag>` for the seven wave B runs, `family_placement_gemma4-e4b-it`, `family_envcheck_mistral-7b-v0.3`, `family_envcheck_qwen3-8b`, `families_envcheck`, `families_waveb_analysis`.
- Jobs: `pace/family_score_serve2.sbatch` (GPU; `WF_WEIGHTS=local` for node-local weights), `pace/families_waveb_gate.sbatch` and `pace/families_envcheck_tokens.sbatch` (CPU).
- Tests: `tests/test_families_waveb.py`; mutants "families wave B …" in `scripts/mutation_check.py`, which now also names the test that killed each mutant.

## 16. CPU checks run before the freeze (2026-11-22; no model loaded, no score made)

All on ICE in the checkout `text-wm-fix34`, with tokenizer and configuration files only.

**The gate on every item, every wave B run** (`pace/families_waveb_gate.sbatch`, job 7240542, commit cbad070; the gate code is unchanged since). For each of the seven runs: 9,916 candidates (both candidates of the 4,666 primary items and 292 determined successes) and "Yes"/"No" after all 14,582 yes/no prompts, with 0 problems. Every prompt ends with its model's thinking-off ending; the thinking switch, markers and endings hold on the first item's six prompts; for Ministral 3, all 29,164 scored prompts' ids equal mistral-common's encoding and its default system prompt never appears; Gemma 4 31B passes with its declared suffix. First-token pairs, identical for every run up to the plain format's leading space: `You` against `Nothing` for 4,902 items, `There` 27, `The` 25, `This` 4, as in wave A. Every one of the 29,164 scored prompts per chat run was checked for its thinking-off ending. The environment recorded by the job: transformers 5.23.0, torch 2.14.1+cu126, tokenizers 0.23.3, with `kernels`, `fla`, `causal_conv1d` and `flash_attn` all absent. Answer tokens: Ministral 3 `Yes` 16860 / `No` 4753; Qwen chat `Yes` 9175 / `No` 2665; Qwen plain ` Yes` 7179 / ` No` 2233; Gemma 4 `Yes` 10784 / `No` 3771. Longest input: 1,092 (Gemma 4 E4B) to 1,149 tokens (Qwen3.8). The full per-model records are in that job's `logs/gateB_7240542.json` on ICE.

**Golden renders** (`tests/golden/families_waveb_rendered.json`): the golden item's aligned, swapped and yes/no prompts rendered by each pinned template (or the plain format) in serve2. Each equals the hand-built format of section 5 (`test_golden_render_is_the_documented_format`).

**Environment check, part 1: tokens** (`pace/families_envcheck_tokens.sbatch`, job 7240543). Every prompt, candidate and answer token list (39,082 per model) is identical between the study's tokenizer stack (transformers 4.51.3, tokenizers 0.21.4, torch 2.6.0, rebuilt in the job's node-local folder) and serve2, for Mistral 7B v0.3 (`LlamaTokenizerFast` against `TokenizersBackend`) and Qwen3-8B (`Qwen2TokenizerFast` against `Qwen2Tokenizer`). Part 1 passes; part 2 (scores on a GPU) waits for the freeze.

**Tests in serve2.** `tests/test_families_waveb.py` and `tests/test_families.py` pass in serve2 on ICE with the pinned tokenizers cached (150 passed, none skipped): the wave B golden renders and gates with the real tokenizers, the mocked dry run (every wave B run scored, the placement check, both environment-check runs, `families_envcheck` and `families_waveb_analysis` with the 16-test family, and the same analysis blocked by a failed environment check), and wave A's own template and gate tests under transformers 5.

**Mutation check.** 72 wave B mutants, all killed, each by a named test in `tests/test_families_waveb.py` (40 different tests); the list with each killing test is `notes/families_waveb_mutants.txt`. Full local suite: 858 passed, 66 skipped (the skips need transformers or cached tokenizers).

**Not yet checked, because it needs the weights:** the loading check (missing or unexpected tensors, device placement) and the speed of each wave B model. Both are checked at the start of each scoring job, before its first score; a failure stops the job.

**Open before any scoring:** an independent review of this design and its code (and fixes until a round finds no must-fix issue), the annotated freeze tag, and the storage decision (section 13). Then the environment check's part 2. Review 1 and its fixes are in section 18; the storage decision is route A (node-local weights).

## 17. Caveats

- **Descriptive comparisons.** Every comparison changes developer or release, data, tokenizer, template and training at once; the base-against-instruct pair is one pair in one format.
- **Formats differ within pairs.** Ministral 3 puts the system text first where Mistral 7B put it last; Gemma 4 has a system turn where Gemma 2 had the text folded into the first user turn (placement check); Qwen3.8 writes empty think blocks into the example replies where Qwen3.5 does not.
- **Contamination.** ALFWorld is public and newer models may have seen more of it; that would show more in accuracy than in the aligned-minus-swapped difference, but it confounds any generation trend.
- **Two-candidate validity** is unchecked for every wave B model, so the yes/no format leads, as in wave A.
- **Single runs.** One scoring run per model; the environment check bounds the software's contribution on two earlier models, not on the new architectures.
- **Unused encoders.** The vision and audio encoders are loaded but never receive input; they do not affect text logits.

## 18. Amendments after review 1, before any score (2026-11-22)

Review 1 ([families_waveb_review1.md](families_waveb_review1.md)) read the design and its code at dbfbdad. Every change below was made before any wave B weight was downloaded and before any wave B model was scored; nothing in the predictions' direction changed. The code changes are in `wf/families.py`, `wf/scoring/logodds.py`, `wf/tools/prefetch_models.py`, the new `wf/tools/weights_manifest.py`, `wf/tools/family_gate_cpu.py` and `pace/family_score_serve2.sbatch`; the tests in `tests/test_families_waveb_review1.py`, `tests/test_families_waveb.py` and the new serve2-only `tests/test_families_waveb_serve2.py`.

**M1, every chat-format score would have crashed in serve2.** transformers 5 returns a BatchEncoding (a mapping holding `input_ids` and `attention_mask`) from `apply_chat_template(tokenize=True, return_tensors="pt")`, where transformers 4 returned the tensor; confirmed on ICE with the pinned Qwen3.8 and Qwen3.5 tokenizers. The scorer passed it to `torch.cat` and to the model, which refuse it. The gate did not see this because it reads ids through its own unwrapping. `LogoddsScorer._encode` now takes the ids out of a mapping (`template_input_ids`) and returns anything else unchanged, so transformers 4 runs exactly as before. A serve2-only test builds a tiny random-weight model from each pinned wave B config (two decoder layers, four for Gemma 4 E4B so that two key-value-sharing layers stay in; small widths; the real vocabulary, rope settings, soft-capping and layer kinds) and runs `score_pair` and `score_one_token_pair` through the scorer with each pinned tokenizer, so the text-only forward passes of `Mistral3ForConditionalGeneration`, `Qwen3_5ForConditionalGeneration` and `Gemma4ForConditionalGeneration` run on a CPU. It also requires the one-pass yes/no score to equal the full teacher-forced score within 0.001 nats. It skips anywhere but serve2 with the pinned files cached.

**M2, B1's basis and the reading of a share that is not supported.** B1's basis now says what is true: every interpretable model scored so far has a positive log-odds effect, but not a share well above one half; Qwen3-4B is competent and its share is 0.49 with a yes/no effect of 2.27 nats (section 10). A wave B model whose share is not supported now reads "effect carried by a minority of items (as Qwen3-4B)" when its yes/no effect's 95% CI is above zero, "no detectable effect" only when that CI covers zero too, and "yes/no effect reversed" when the CI is below zero (`b1_reading`, reported per model with B1). B1 itself, and whether it holds, are unchanged.

**M3, a file-hash check for node-local weights.** A manifest per model is committed under `wf/config/weights_manifests/`: every file the prefetch tool's patterns select at the pinned revision, with its size and, for a Git LFS file (the weights, the large tokenizer files), the LFS SHA-256, or for a small file kept in Git the Git blob SHA-1 (also a hash of the content). The manifests were built from Hub metadata alone (`HfApi().model_info(..., files_metadata=True)`, no token; no model is gated) and cross-checked against `snapshot_download(dry_run=True)` under the same patterns: the same files and sizes for all six repositories (17.87, 19.33, 19.33, 16.02, 55.59 and 62.58 GB). The Git blob ids were checked against two files fetched from the Hub. In a route-A job the prefetch tool hashes every downloaded file after the download and stops the job on any mismatch, missing file or extra file; the scoring job then refuses to score unless that record exists, is for the model's id and pinned revision, and passed, and saves it in the shard (`weights_check.json`, and a summary in `tokenization_checks.json`). Hashing the largest model's 62.6 GB adds a few minutes to its job.

**Should-fix items, all applied.**

- *S1, the download goes to node-local disk.* The job passes the folder to the prefetch tool by name (`--cache "${LOCAL_CACHE}" --node-local`), and the tool refuses any folder that resolves to the shared file system (`/storage`). Tests cover the refusal (also behind a link), the folder passed to the download, and the job script's exports.
- *S2, disk space.* Before a download the free space must be at least the manifest's total plus 10%.
- *S3.* In route A the job also sets `HF_XET_CACHE` to `${LOCAL_CACHE}/xet`, so hf-xet's chunk cache stays on the node.
- *S4, the environment check's GPU type is enforced.* Pinned per model (`ENVCHECK_GPU`): NVIDIA A40 for Mistral 7B, NVIDIA L40S for Qwen3-8B (its earlier jobs 7174094–7174099 ran on L40S). The serve2 job stops an environment-check run on any other GPU before loading, and `families_envcheck` fails a model whose shard recorded another type, no type, or two types.
- *S5, no wave B score is read unless the environment check passed.* B1 no longer holds for a run without a share test when the check failed; every B1, B2 and B3 entry is "not decidable" (None). The analysis withholds every wave B block, comparison, placement and template check, B3's fallback, the competence verdicts and every estimate and p value in the family, in the metrics, the report and the status lines; the reference models' blocks stay.
- *S6, resources at the start of a job.* Before loading, the visible GPUs together must hold the model's `gpu_min_gb` (24 GB for the small models, 80 GB for the 27B and 31B), and the 27B and 31B runs need a job with at least 128 GB of memory (`SLURM_MEM_PER_NODE`).
- *S7.* The CPU gate tool accepts the check variants (`gemma4-e4b-it-firstturn`, `qwen3.8-27b-nopreserve`) and gates the 300 placement items their shards score. The local tests run the variants on stand-in tokenizers; the pinned templates' gate on the golden item runs in the serve2 test file. On ICE (serve2, login node, read-only) the tool gated both variants on their 300 items: 600 candidates and 900 yes/no prompts each, 0 problems; `You` against `Nothing` on all 300 items; longest prompt 1,088 tokens (Gemma 4 E4B, first turn) and 1,137 (Qwen3.8, no think blocks); 0 problems also for "`<think>` before the thinking-off ending".
- *S8, Qwen3.8's empty think blocks in the examples.* Qwen3.8's template writes `<think>\n\n</think>\n\n` into every earlier assistant turn unless `preserve_thinking=False`; with it, the examples render exactly as Qwen3.5's (checked with the pinned tokenizers on ICE). Pre-registered check: `family_nopreserve_qwen3.8-27b`, Qwen3.8 27B with `preserve_thinking=False` on the 300 placement items under the aligned and swapped tasks in both formats (about 1,800 passes, about 0.3 A100-hours), saved under `qwen3.8-27b-nopreserve`. Its gate also requires that no prompt holds `<think>` before its thinking-off ending. It is read against the model's own run by the one-fifth rule, as the placement checks are (`one_fifth_rule`: the 95% CI of the per-item difference excludes zero and the difference exceeds a fifth of the native effect); if it moves the effect, the large Qwen pair is reported as format-dependent, with both numbers.
- *S9, eager against SDPA for Gemma 4.* After the gate passes and before the first score, each Gemma 4 job (both models and the placement check) scores its first 5 items (aligned task, both formats) with SDPA, switches the text model to eager attention, scores them again, and switches back. Each switch is confirmed on the config. Tolerance, fixed now: mean absolute difference at most 0.10 nats on the yes/no scores and 0.50 nats on the two-candidate scores, and no single score more than 1.0 nat apart. A real attention fault moves scores by nats; kernel rounding moves them by hundredths. The record is saved with the gate's, and a disagreement stops the job. The switch targets the text model only: Gemma 4 E4B's audio encoder has no SDPA, so a whole-model switch back to SDPA raises (found by the serve2 test on the real class).
- *S10.* The first-turn variant's note no longer says "not the last" for Gemma 4 (wave A's Mistral note is kept word for word, as saved). New tests: the CPU gate tool exits 1 when any gate fails, and the plain-format gate fails when the two candidates share a first token.

**Science wording (the author's decisions).**

- *Generation pairs are confounded by size.* Within the Qwen3 line the share runs 0.49 / 0.89 / 0.70 / 0.71 across 4B / 8B / 14B / 32B. Every generation-pair reading carries this spread (`PAIR_CAVEAT`, in B2's entries and in the analysis's status), and a "differs" is not read as a generation effect.
- *The Gemma pair: E4B is kept.* The rule was to swap Gemma 4 E4B for the size-matched Gemma-4-12B-it only if it exists at a pinnable revision, uses the same chat template family and first-turn handling as E4B, and fits the same GPU tier. It exists (`google/gemma-4-12B-it` at 43c9d014b231, created 2026-05-23, not gated) and fits the tier (23.9 GB of bf16 weights), but its chat template is byte for byte the 31B template, whose thinking-off generation prompt ends with the empty thought channel; E4B's does not. It is also a different architecture (`Gemma4UnifiedForConditionalGeneration`, model type `gemma4_unified`), so the loading check and the forward test would not carry over. Two conditions fail, so E4B stays, and the size confound is stated: Gemma 4 E4B (about 4B effective, 7.5B language-model parameters with its per-layer embeddings) against Gemma 2 9B changes size as well as generation, so the Gemma pair is the weakest of the four.
- *B2 can fail only on a clear negative difference.* "Inconclusive" and "matches" count as holding, and equivalence within ±0.05 is nearly unreachable at wave A's interval widths, so B2 is a weak prediction: it is refuted only when a newer model moves detectably fewer failures.
- *B3's fallback.* If the base model fails only the yes/no coverage criterion (median P(Yes) + P(No) under the neutral task below 0.5), B3 is read on the two-candidate format: the paired difference in the two-candidate item share (failures whose two-candidate L under the aligned task exceeds that under the swapped task), instruct minus base, with a game-clustered bootstrap; B3 holds if its 95% CI is above zero. The fallback needs the base model's two-candidate AUC under the neutral task to be at least 0.70 and the instruct model (plain format) to pass the competence gate (`b3_fallback`). It is outside the Holm family, which keeps its 16 tests.
- *Wording.* The outcome table says "goal-congruent shift", not "bias", and states that the item share does not separate wishful prediction from a judgement of task relevance.
- *Run order.* The environment check first; then the four small models (Ministral 3; Qwen3.5 9B in both formats; Qwen3.5 9B base; Gemma 4 E4B and its placement check); then Qwen3.8 27B and its preserve_thinking check; Gemma 4 31B last (section 14, `RUN_ORDER_B`).

**Run on ICE during the fixes (read-only use: serve2's Python on the login node, offline, the shared cache only read, code in a temporary folder removed afterwards).** The serve2 test file passed in full (11 tests): every wave B tag scored end to end through a tiny model of its real class, the eager-SDPA check on both Gemma 4 classes, and the two check variants' gate on the golden item with the pinned templates. With the M1 fix reverted, the forward test fails for all five chat-format tags (the scorer's prompt ids are a BatchEncoding, not a tensor) and passes for the two plain-format tags, which never call the template.

## 19. Still to run on ICE before the freeze

- `tests/test_families_waveb_serve2.py` in serve2 by the author, from a checkout at the fix commit (the command is in its docstring), as the record of that commit.
- The CPU gate for the two check variants as a job, so its record sits with job 7240542's: `pace/families_waveb_gate.sbatch` with `GATE_TAGS="gemma4-e4b-it-firstturn qwen3.8-27b-nopreserve"` (300 items each; the same run on the login node passed, above).
