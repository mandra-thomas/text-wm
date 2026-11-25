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

*Amended 2026-11-23 (section 8, "Amendment after the environment check failed (cause found)"):* the five reference models are now rescored in serve2 on every item and condition (`family_rescore_<tag>`, scores under `<tag>-serve2`), and every comparison reads those rescorings. The runs above are read only for the descriptive original-environment column.

## 4. Software environment

Every wave B model needs transformers 5 (Gemma 4, Qwen3.5 and Ministral 3 are not in 4.51.3), so wave B runs in `serve2`. Wave A, the confirmatory runs and every earlier run keep their environment (`SERVE_ENV`, transformers 4.51.3, torch 2.6.0). A scoring job selects serve2 explicitly (`pace/family_score_serve2.sbatch`) and refuses to start if the Python it runs has transformers below 5; a wave A or reference model is scored there only by the environment check (section 8). *Amended 2026-11-23 (section 8, "Amendment after the environment check failed (cause found)"):* the five reference models are also rescored there in full by `family_rescore_<tag>`, with wave B's loading check, gate and environment record.

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

### Amendment after the environment check failed (cause found), before any wave B score was read (2026-11-23)

Written on 2026-11-23, when no wave B model had been scored (no wave B weight downloaded, no wave B score in any run folder) and nothing from a wave B run had been read. The decision (the user's) is to rescore the older comparison models in serve2, so that every comparison sits in one software environment. Corrected on 2026-11-24 after review 2 ([families_waveb_review2.md](families_waveb_review2.md)), still before any wave B score: the description of the diagnostic, the pointers in sections 3, 4, 11 and 18, and the hardened gate below. A short re-review (2026-11-25) found the fixes complete and the design ready to re-freeze. Its four non-blocking points were applied before the re-freeze, each marked "re-review N1" to "N4" below:

- N1: the size of the logit change, stated at the logits' own magnitude;
- N2: the section 17 caveat, and "environment-check runs";
- N3: like-for-like readings in `environment_dependent_pairs`;
- N4: one commit for all rescorings and the token job, and a clean tree.

**This amendment replaces the gate; it does not satisfy it.** Under the frozen rule, wave B would stay unread until the check was rerun and passed. The check has not been rerun and has not passed. Wave B is instead read against older models rescored in serve2, under the new gate below. The rule it replaces, as frozen at `families-waveb-design-frozen-2026-11-22`:

> **If it fails.** No wave B score is read. The cause is found first (an attention or kernel change, a tokenizer difference, a dtype path) and fixed or documented in an amendment made before any wave B score is read; the check is rerun. `families_waveb_analysis` reads the check's verdict and marks every wave B test "not interpretable (environment check)" unless it passed.

**The check's verdict stays FAILED.** `families_envcheck` (CPU job 7247973, checkout `text-wm-fix39`): Mistral 7B v0.3 passed every criterion, its yes/no criterion narrowly (yes/no mean absolute difference per score 0.0688 nats against the limit of 0.07, goal effect moved 0.003, item share unchanged; two-candidate 0.089 and 0.006). Qwen3-8B failed one criterion, the yes/no mean absolute difference per score: 0.248 nats against the limit of 0.07 (largest 1.0; 30% of scores identical). That is about seven times the archived cross-GPU difference the limit was built on: there, scoring the same Qwen3-8B yes/no prompts on different GPU types (`rq1v_score` against `size_score_qwen3-8b`, 14,582 scores) left 91% of scores identical and gave a mean absolute difference of 0.035. In two-candidate format the stack change (0.341) was about twice the cross-GPU difference (0.18). So the stack change is not the same tolerated kind of difference as the GPU differences already present in every earlier comparison: the check was built to test exactly that, and it failed. Every effect-level criterion passed for Qwen3-8B too: the yes/no goal effect moved by 0.026 nats (limit 0.10), the item share toward Yes by 0.010 (limit 0.02), the two-candidate goal effect by 0.011 (limit 0.10), and the two-candidate mean absolute difference was 0.341 (limit 0.35). Token ids were identical in both environments for both models (part 1), and both environment-check runs ran on their pinned GPU types. The verdict is not re-scored, re-thresholded or re-read: it stays failed.

**The cause, found with a precision diagnostic** (`scripts/families_envcheck_precision.py`, GPU job 7248092 on an L40S, checkout `text-wm-fix45`; 50 of the 300 check items, every sixth, Qwen3-8B in both environments). In each environment the diagnostic reproduced the archived scores exactly (300 of 300 scores equal to the last bit: `size_score_qwen3-8b` in the study environment, `family_envcheck_qwen3-8b` in serve2), and the token ids were identical (450 of 450 prompts). The load-time settings were the same in both (SDPA attention, bf16, the same model class and pinned revision, TF32 matmuls off in both, cuDNN's TF32 switch on in both, the same SDPA back-end switches); only the library versions differed (torch 2.6.0 against 2.14.1, CUDA 12.4 against 12.6, cuDNN 9.1 against 9.10, transformers 4.51.3 against 5.23.0). For each score the diagnostic also kept the same final hidden state multiplied by the output layer in float64, so that no logit is rounded to bf16.

- Qwen3-8B's Yes and No logits at the readout position are large: 30 to 65.5 in magnitude (median 52; the larger of the two has median 62). Where a logit lies between 32 and 64, bf16 values are 0.25 apart, which is why every yes/no difference was a multiple of 0.25.
- The bf16 differences are exactly the rounding of the float64 logits: 205 of the 207 logits that changed equal their float64 value rounded to bf16 in both environments, and rounding the float64 logits predicts 100 of the 101 changed scores exactly.
- But the float64 logits themselves moved. Before any bf16 rounding, the yes/no score differs between the environments by 0.217 nats per score on average (median 0.16, largest 0.96), three times the limit. Each logit moved by 0.25 nats on average, a median relative change of 0.44%: about 0.4% of the logit, close to the bf16 unit roundoff (2⁻⁸ = 0.39%; bf16's machine epsilon is 2⁻⁷ = 0.78%). At the logits' own magnitude (32 to 64) that is about one full bf16 step (0.25 nats), not a change within rounding (re-review N1, 2026-11-25). The float64 differences by condition were −0.059 (aligned; standard error 0.037), +0.054 (swapped; 0.035) and −0.070 (neutral; 0.043), pooled −0.025 (0.023). Recomputing the final norm in float64 as well leaves the mean difference the same to within 0.01 nats (0.218).
- What was measured is the change in the float64 logits. The hidden-state change is inferred from it, not measured directly: the same weights in float64 give different logits, so the final hidden states differ between the environments.
- So the hypothesis that rounding alone explains the failure was refuted. The software stack changes the final hidden state by a few tenths of a percent, and Qwen3-8B's large readout logits turn that into about a quarter of a nat in absolute terms. A possible reason why Mistral 7B's scores moved less is that its readout logits are smaller. That is untested: the diagnostic's Mistral mode was never run, and Mistral passed only narrowly (0.0688 against 0.07).
- The 50 items do not settle whether the change depends on the condition. On them, the yes/no goal effect (aligned minus swapped) moved by −0.11 nats in float64 (standard error 0.053) and by −0.125 in bf16 (0.064). That is above the check's 0.10 limit, at about two standard errors. On the full 300 items, however, it moved by +0.026, within the limit. The 300-item figure is the better estimate, so the subset's shift is probably noise. The effect-level criteria, which are computed on the 300 items, passed.
- The two-candidate scores show the same mechanism. Qwen3-8B's two-candidate L moved by −0.10 (aligned), −0.11 (swapped) and −0.06 (neutral) on the 300 items. On the diagnostic's 50 items, the float64 change was −0.05, −0.04 and −0.05. It comes from the longer candidate ("You …"), whose log-probability fell by 0.04 to 0.07 nats while "Nothing happens." moved by 0.02 at most: each token lost about 0.004 nats because the log-normaliser rose slightly more than the target logit. The shift is the same in the aligned and swapped conditions, so it cancels in the goal effect (moved 0.011).

A rule that only tolerated rounding would not have fixed this: the change is in the numbers before rounding. Comparing a wave B model scored in serve2 with an older model scored in the study environment would therefore carry an environment difference of about this size per score.

**The fix: every older comparison model is rescored in serve2.** Every older model used in any wave B comparison is rescored in serve2 on the full item set and conditions it is compared on: all 4,666 primary items under the aligned, swapped and neutral tasks in both formats, and the 292 determined successes under the aligned and neutral tasks in both formats (every arm of `sizes.size_jobs`, about 43,700 forward passes per model), with its own chat template, prompts and attention setting as in its earlier run, and wave B's loading check, gate and environment record. The list, read off the analysis's design:

| Model | Why it is compared | Rescoring run (scores under `<tag>-serve2`) | GPU type (pinned) |
|---|---|---|---|
| Qwen3-8B | planned pair with Qwen3.5 9B; descriptive reference for every wave B model | `family_rescore_qwen3-8b` | NVIDIA L40S (its earlier type) |
| Llama-3.1-8B | descriptive reference for every wave B model | `family_rescore_llama3.1-8b` | NVIDIA L40S (its earlier runs were on mixed types) |
| Qwen3-32B | planned pair with Qwen3.8 27B | `family_rescore_qwen3-32b` (two shards) | NVIDIA A100 80GB PCIe (its earlier type) |
| Gemma 2 9B | planned pair with Gemma 4 E4B | `family_rescore_gemma2-9b-it` | NVIDIA L40S (its earlier type) |
| Mistral 7B v0.3 | planned pair with Ministral 3 8B | `family_rescore_mistral-7b-v0.3` | NVIDIA A40 (its earlier type) |

No other comparison involves an older model. The fifth planned pair (Qwen3.5 9B instruct against base, plain format), the three descriptive comparisons in the configuration (format, and size within the Gemma 4 and Qwen lines), the Gemma 4 E4B placement check (its first-turn variant against its own run) and the Qwen3.8 preserve_thinking check (against its own run) compare wave B runs only, all already in serve2. `families_waveb_analysis` reads every reference model from its serve2 rescoring, so every comparison, planned or descriptive, sits in serve2: a difference between models can no longer be a difference between environments. (A difference of GPU type between two models remains, as in every earlier comparison; the GPU types are pinned and recorded.)

**Unchanged:** the six per-model item-share tests (wave B models only, already scored in serve2), the 16-test Holm family and its planned pairs, every threshold and margin, the competence gate, the predictions B1 to B3 and their readings, B3's fallback, the placement and template checks and their one-fifth rule, the 0.3-nat noise floor, and the wave B models' runs and their order among themselves.

**How the analysis gates now.** It no longer withholds wave B because the environment check failed. That verdict is still read and reported, as failed, with `gates: false`. Wave B is withheld (every wave B test, comparison, check and prediction "not interpretable (environment check)", with no estimate written, as before) unless every serve2 rescoring passes `rescore_check` (`rescore_gate`):

- every shard is complete, at one commit;
- coverage (review 2, S5): exactly one score for every (item, condition) that `sizes.size_jobs` gives on the analysis's own items, each carrying that item's hash, so an item set that drifted between checkouts cannot shrink a comparison silently;
- the token ids are identical between the study environment's tokenizer stack and serve2 for every prompt, candidate and answer token. This is the comparison of part 1, run again for all five models with the rescoring's tokenizer settings (`wf.tools.envcheck_tokens dump --rescore`, `pace/families_envcheck_tokens.sbatch` with `ENVTOK_RESCORE=1`, written to `rescore_tokens/`; Qwen3-32B's tokenizer files are downloaded at the pinned revision into the job's node-local folder);
- the comparison's provenance (review 2, S4): its left side is the study stack (transformers 4.51.3, tokenizers 0.21.4), its right side transformers 5 or later, both at the pinned revision, and both dumps were made at the rescoring's commit (each dump records its commit and versions);
- every shard ran on the model's pinned GPU type alone;
- every shard's gate passed in transformers 5 at the pinned revision, with the loading check on and passed (review 2, M3: the gate record now holds `strict_load`, and `loading.ok` must be true);
- for the node-local model, the downloaded files matched the committed manifest in every shard;
- the analysis's `reference` entries are exactly the five rescorings, model `<tag>-serve2` and every arm, and there is no other reference (review 2, S3). An analysis whose configuration points a reference anywhere else is blocked.

**The label's meaning has changed.** "not interpretable (environment check)" is kept, so that the frozen outputs keep their form. Since this amendment it means that the serve2 rescorings have not passed `rescore_gate`. It no longer means that the environment check failed.

**The original-environment column (descriptive).** The original-environment scores are reported beside the serve2 ones, descriptively only (re-review N3: readings made like for like, below):

- the runs are `size_score_qwen3-8b`, `size_score_qwen3-32b`, `family_score_gemma2-9b-it`, `family_score_mistral-7b-v0.3`, and Llama's `rq1_score`, `rq1x_score` and `rq1v_score`;
- for each reference, `environment_sensitivity` gives the item share, R and goal effects in both environments and their difference, the competence verdict, the neutral-task yes/no AUC and the item share's CI in each environment;
- every comparison involving a reference is repeated with its original-environment scores (`comparisons_original_environment`).

**Rule for an environment-dependent reference** (review 2, S2; fixed on 2026-11-24 before any score). A reference is *environment-dependent* if, between the environments:

- its item share toward Yes moves by more than 0.02, or
- its yes/no goal effect (ΔL) moves by more than 0.10 nats.

These are the environment check's effect-level limits; a difference that is not a number also counts. Every comparison involving an environment-dependent reference is listed in `environment_dependent_pairs` with both readings. For a planned pair the serve2 reading is the primary family's (Holm-adjusted). Beside it are the serve2 reading without the adjustment and the original-environment reading, also unadjusted, so that a discordance between the last two is not an artefact of the adjustment. The unadjusted readings are "differs", "differs, within the margin" (the difference's 95% CI excludes zero and its 90% CI lies within the margin), "matches" and "inconclusive (no difference detected)". The serve2 reading stands. A comparison whose reference has no original-environment scores (its original run missing) is listed too, with that reference under `references_without_original` and no original reading, so it is never silently absent.

None of these is a test, and all comparison-level entries are withheld with the rest whenever wave B is withheld.

**What the original column also contains (review 2, S1).** The column does not isolate the software stack:

- *Llama-3.1-8B.* Its column also contains the scorer path: `rq1v_score` scored Yes against No with two forward passes over the prompt plus the candidate (`score_pair`), where the rescoring reads both from one pass (`score_one_token_pair`); the two are equal in exact arithmetic, but the kernels see different lengths. It also contains mixed GPU types that were never recorded. And the rq1 runs loaded an unpinned snapshot, which equals the pin only because the cache held a single snapshot (checked by the reviewer on ICE).
- *Qwen3-8B and Qwen3-32B.* The size runs also loaded unpinned snapshots that equal the pins for the same reason. Their shards ran on L40S (Qwen3-8B) and A100 80GB (Qwen3-32B), per their logs.
- *Wave A's Gemma 2 and Mistral runs* loaded the pinned revisions.

So nobody should read Llama's column as the size of the stack effect.

**Storage and cost.** Qwen3-32B's weights are in no ICE cache any more, so it uses route A (section 13): the pinned snapshot is downloaded into the GPU node's local disk and every file is hashed against a committed manifest before loading. The manifest is `wf/config/weights_manifests/Qwen__Qwen3-32B.json`, built from Hub metadata with no token: 24 files, 65.54 GB, and a dry run agrees. The job stops before loading if the model is not on node-local weights. It runs on an A100-80GB with a 128 GB job, in two shards, each downloading the snapshot again. The other four models are in the shared cache. Estimates are scaled from wave A's timings (section 12); serve2 may differ, and the Qwen3-32B figure is the least certain.

| Rescoring | GPU | Estimate |
|---|---|---|
| Mistral 7B v0.3 | A40 | 1.7–2.5 GPU-hours |
| Qwen3-8B | L40S | 1.5–2.5 GPU-hours |
| Gemma 2 9B (eager attention) | L40S | 1.8–2.5 GPU-hours |
| Llama-3.1-8B | L40S | 1.5–2.5 GPU-hours |
| Qwen3-32B | A100-80GB, two shards | 4–8 GPU-hours, plus 0.6–1.0 for the two downloads and hash checks |

The total is about 11 to 19 GPU-hours, of which about 5 to 9 are on 80 GB cards, on top of section 12's wave B budget. The token comparison is a short CPU job. The two-model run of part 1 took 5 minutes 32 seconds (job 7240543), so five models take about 15 to 20 minutes, plus building the study stack and downloading Qwen3-32B's tokenizer files: under an hour in all, against the job's 3-hour limit. A longer run should be looked into. (Corrected after review 2, S11; the first version said one to three hours.)

**Order** (`RUN_ORDER_B`, amended): after the environment check, the five rescorings (Mistral 7B, Qwen3-8B, Gemma 2 9B, Llama-3.1-8B, Qwen3-32B) and the CPU token comparison; then the wave B models in their pre-registered order; then `families_waveb_analysis`. **All five rescorings and the token comparison run at one commit, from one clean checkout** (re-review N4): `rescore_check` requires the token dumps to be made at the rescoring's commit, so a rescoring rerun at a later commit needs the token job rerun at that commit too, and the token job's output then has to match every other rescoring's commit as well, so in practice all five rescorings and the token job are rerun together. This fails closed (the gate blocks), so it cannot leak. Each token dump also records whether its working tree was dirty, and the gate requires a clean tree on both sides. No rescoring and no wave B scoring runs before this amendment has had its independent review and the design is frozen again.

**Code:**
- `wf/families.py`:
  - constants: `RESCORE_MODELS`, `RESCORE_GPU`, `RESCORE_NODE_LOCAL`, `RESCORE_AMENDMENT`, `ENV_DEPENDENT_RULE` and `TOKEN_STUDY_STACK`;
  - functions: `rescore_spec`, `run_family_rescore`, `rescore_check`, `rescore_reference`, `rescore_gate`, `environment_sensitivity` and `environment_dependent_pairs`;
  - the analysis's gate, the original-environment column, `RUN_ORDER_B`, and the gate record's `strict_load`.
- The diagnostic: `scripts/families_envcheck_precision.py` and `pace/families_envcheck_precision.sbatch`.
- `wf/config/family_rescore_<tag>.yaml` (five) and `families_waveb_analysis.yaml` (`reference`, `rescore`, `reference_original`).
- `pace/family_score_serve2.sbatch` (`family_rescore_<tag>`, its GPU check before loading, and node-local weights required for Qwen3-32B) and `pace/families_envcheck_tokens.sbatch` (`ENVTOK_RESCORE=1`).
- `wf/tools/envcheck_tokens.py` (`--rescore`; each dump records its commit and tokenizer versions) and `wf/tools/prefetch_models.py` (reference tags).
- Tests are in `tests/test_families_waveb.py`.

Under the strict mutation rule (killed only when pytest exits 1 and names a failing test), all 33 new mutants are killed, by 14 different tests. The 135 earlier wave B mutants, rerun on the amended code, are all killed too; one of them was retargeted first, because the line it mutates now reads the rescorings' gate. The list is in `notes/families_waveb_mutants.txt`. Full local suite: 948 passed, 77 skipped.

**After review 2 (2026-11-24).** Mutants were run one at a time through the same strict runner (`scripts/mutation_batch.py`), each result checkpointed to `notes/families_waveb_review2_mutants.jsonl` as it finished.

**The bytecode fix.** These counts are from a rerun with the runner's bytecode fix, ported from main (cc1db87): tests run with `PYTHONDONTWRITEBYTECODE=1`, so a stale compiled mutant can never be read. Each mutant also runs alone, on a fresh copy without `__pycache__`.

- The first count of this round was made before the fix was in. It credited each of its 73 mutants to the same test as the rerun, so it was unaffected in practice, but it is superseded.
- The counts reported for the amendment's first version (33 new and 135 earlier mutants, above) were made without the fix, several mutants to a copy. Every wave B mutant is therefore rerun with the fix (`notes/families_waveb_mutants_recheck_2026-11-24.jsonl`).

The file holds 74 results:

| Mutant set | Killed |
|---|---|
| The reviewer's 13 survivors (`R2 …`), rebuilt on the fixed code | 13 of 13 |
| Breaking each review 2 fix in turn | 25 of 25 |
| All 33 earlier rescoring mutants, rerun (they include the 4 the reviewer could not finish) | 33 of 33 |
| Two earlier wave B mutants on lines this round touched ("environment gate not applied"; review 1's "job does not stop on the wrong GPU", retargeted to the environment check's own line) | 2 of 2 |
| Review 1's "plain gate allows a shared first token" (picked up by the selection) | 1 of 1 |

The kills come from 26 different tests; no test is credited with more than 16 of them.

The other 132 wave B mutants were rerun with the fix too: 132 of 132 killed, by 74 different tests, and no test is credited with more than 8. So **every wave B mutant, 206 in all, is killed by a named failing test under the strict rule with the bytecode fix**, by 97 different tests. Full local suite at this state: 965 passed, 77 skipped.

**After the re-review's N3 and N4 (2026-11-25).** Five new mutants break N3 and N4, and the 25 review 2 mutants were rerun on the changed lines, all with the bytecode fix (`notes/families_waveb_rereview_mutants.jsonl`): 30 of 30 killed. Three of the review 2 mutants were retargeted first, because N4 changed their lines. One new mutant survived the whole suite at first: "dump without its tree state", because the test fixed the tree as clean, the same value the mutant writes. The test now records a dirty tree, and the mutant is killed. Wave B test files (with the serve2 file): 245 passed, 22 skipped.

Two corrections were made during the run:

- **Spurious kills.** Fifteen kills were first credited to `test_token_dump_encodes_the_rescoring_prompts`, which failed in the runner's copy of the repository for a reason unrelated to the mutants: the copy has no `.git`, so the dump's commit read "unknown". The test now fixes the commit, the fifteen were rerun, and each was killed by its own test.
- **One survivor.** Review 1's "job does not stop on the wrong GPU" survived the whole suite once, because its test also read the new rescoring block. The test now reads only the environment check's own block, and the mutant is killed.

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
| The environment check fails | As frozen: no wave B result is read until the cause is fixed and documented, before data, and the analysis writes no wave B number (amendment, section 18). *Amended 2026-11-23 (section 8, "Amendment after the environment check failed (cause found)"); the check did fail:* every older comparison model is rescored in serve2, and wave B is read only when every rescoring passes `rescore_check` (`rescore_gate`); the amendment replaces this gate, it does not satisfy it. |
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

Amended 2026-11-23 (section 8, "Amendment after the environment check failed (cause found)"): the check failed, and step 2's "stop here" is replaced. After an independent review and a new freeze, the five serve2 rescorings of the older comparison models and their CPU token comparison run between steps 2 and 3, and the analysis reads wave B only if they pass.

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
- **Single runs.** One scoring run per model; the environment check bounds the software's contribution on two earlier models, not on the new architectures. *Amended 2026-11-23 (section 8, "Amendment after the environment check failed (cause found)"; pointer added after the re-review, N2):* the check failed and bounds nothing; it is the serve2 rescoring of every older comparison model that removes the software stack from the comparisons. The stack's effect on the new architectures remains unmeasured.
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
- *S5, no wave B score is read unless the environment check passed.* B1 no longer holds for a run without a share test when the check failed; every B1, B2 and B3 entry is "not decidable" (None). The analysis withholds every wave B block, comparison, placement and template check, B3's fallback, the competence verdicts and every estimate and p value in the family, in the metrics, the report and the status lines; the reference models' blocks stay. *Amended 2026-11-23 (section 8, "Amendment after the environment check failed (cause found)"):* the withholding is unchanged, but it now applies unless the serve2 rescorings pass `rescore_gate`, whatever the environment check said. The label "not interpretable (environment check)" now means "the serve2 rescorings have not passed".
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

**Mutation check (strict rule: killed only when pytest exits 1 and names a failing test).** 63 new mutants (the review's eight findings put back, and every fix broken in turn): 63 killed, by 31 different tests. The 72 earlier wave B mutants, rerun under the strict rule on the fixed code: 72 killed. Lists in `notes/families_waveb_mutants.txt`, table in `notes/testing.md`. Full local suite on the fixed code: 935 passed, 77 skipped (the skips need transformers, torch or cached tokenizers; the serve2 tests among them passed on ICE).

## 19. Still to run on ICE before the freeze

- `tests/test_families_waveb_serve2.py` in serve2 by the author, from a checkout at the fix commit (the command is in its docstring), as the record of that commit.
- The CPU gate for the two check variants as a job, so its record sits with job 7240542's: `pace/families_waveb_gate.sbatch` with `GATE_TAGS="gemma4-e4b-it-firstturn qwen3.8-27b-nopreserve"` (300 items each; the same run on the login node passed, above).
