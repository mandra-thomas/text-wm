# Model families, wave A: design (logged 2026-11-20, before any score)

Exploratory, not preregistered. Code: `wf/families.py` (runs `family_score_<tag>` and `families_analysis`), the shared shard runner in `wf/sizes.py`, `wf/tools/prefetch_models.py`; jobs `pace/families_prefetch.sbatch` and `pace/family_score.sbatch`; tests `tests/test_families.py`.

## Why

The tracker's section 13 asks whether post-training explains the goal effect. The confirmatory comparison of Qwen3-8B (April 2025) and Llama-3.1-8B (July 2024) can't separate family, generation and post-training. The Qwen3 size series showed that the effect stays positive at every size within one family. Wave A adds models from other families and from Llama's next release. All four load in the study's environment (transformers 4.51.3).

This wave serves two of the section 13 hypotheses:

- **Distillation (hypothesis 4).** Llama 3.2 1B and 3B were pruned and distilled from Llama 3.1 8B and 70B. If the bias comes from the teacher, the small Llama 3.2 models should show an effect like Llama 3.1-8B's, not the near-zero effect of Qwen3's independently trained small sizes. Qwen3-0.6B and 1.7B answered Yes to almost everything.
- **Generation matching for Llama 3.1.** Gemma 2 9B (June 2024) and Mistral 7B v0.3 (May 2024) are 2024-generation instruct models of about Llama 3.1-8B's size, from other developers. If they look like Llama 3.1 and unlike Qwen3, generation or post-training style is the better explanation. If they spread widely, family matters more. They are also the "old" halves of the within-family generation pairs that wave B completes (Gemma 2 against Gemma 4, Mistral 7B against Ministral 3).

## Models

| Tag | Hugging Face id | Pinned revision | Generation | Release | Parameters | Why |
|---|---|---|---|---|---|---|
| `gemma2-9b-it` | google/gemma-2-9b-it | 11c9b309abf7 | Gemma 2 | 2024-06-27 | 9.24B | 2024 generation, another developer; older half of the Gemma pair |
| `mistral-7b-v0.3` | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41d | Mistral 7B v0.3 | 2024-05-22 | 7.25B | 2024 generation, another developer; older half of the Mistral pair |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | 9213176726f5 | Llama 3.2 | 2024-09-25 | 1.24B | distilled from Llama 3.1 |
| `llama3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b | Llama 3.2 | 2024-09-25 | 3.21B | distilled from Llama 3.1 |

How these were recorded:

- **Revisions:** the Hub's current commit (`sha` from `https://huggingface.co/api/models/<id>`), read on 2026-11-20. The full hashes are in `wf.constants.REVISIONS`.
- **Parameter counts:** from each checkpoint's safetensors metadata on the Hub.
- **Release dates:** the public announcements. The Hub repositories were created a few days earlier.

For comparison, `families_analysis` reads Qwen3-8B from `size_score_qwen3-8b` (8.19B, released 2025-04-29) and Llama-3.1-8B from `rq1_score`, `rq1x_score` and `rq1v_score` (8.03B, released 2024-07-23). Those runs used exactly the same prompts.

## Prompts and chat templates

**Prompt and conditions.** The prompt is the size series': RQ2's few-shot prompt with example set 1, history only. There are three tasks:

- aligned: the episode's real task;
- swapped: the same task line with an irrelevant object;
- neutral: "explore the room".

Each task is scored two ways. The two-candidate score compares the success twin's text with "Nothing happens.". The yes/no score compares the single tokens "Yes" and "No" after a fixed feasibility question. The determined successes are scored too, under the aligned and neutral tasks in both formats, for AUC.

Each model gets its own chat template at the pinned revision. Every template was checked with transformers 4.51.3. The tests hold a golden rendering of one real item per model (`tests/golden/families_rendered.json`).

**Gemma 2 9B.** The template raises "System role not supported" on a system message. The system text (`WM_SYSTEM`) is folded into the first user turn: the system text, a blank line, then that turn's own text (`fold_system`).

- **Why it can't bias the comparison.** The fold is applied after the condition's prompt is built, the same way in every condition. It touches only the first turn, which holds the first prompt example, never the task line being varied.
- **Why the first turn.** Google's guidance for Gemma is to give instructions at the start of the user's first turn.
- **The rest of the template.** It alternates `user` and `model` turns, trims each turn's text, and writes `<bos>` itself.

**Mistral 7B v0.3.** The template takes a system message but places it in the last user turn: `[INST] system\n\nquery[/INST]`. It is left as Mistral ships it, since mistral_common, Mistral's reference tokenizer library, places it the same way.

- **Position differs from the others.** The system text sits just before the query, after the examples, not at the start.
- **Optional sensitivity check.** Folding it into the first turn, as for Gemma, would test whether the position matters. It is not planned.
- **Alternation.** The template requires user and assistant turns to alternate, and our prompts do.
- **No generation prompt.** The prompt ends at `[/INST]`.
- **Replies.** The template writes a reply after one space, `[/INST] You ...`, so each candidate is scored as `▁You ...`, exactly as the template writes a reply.

**Llama 3.2 1B and 3B.** The header format is Llama 3.1's.

- **The date.** Unlike Llama 3.1's template, Llama 3.2's writes today's date into the system header (`strftime_now`) unless a `date_string` is passed. Unfixed, the prompt would depend on the day a job ran, and a shard running past midnight would mix two prompts.
- **The fix.** `date_string` is fixed at "26 Jul 2024". That is the template's own fallback and the date Llama 3.1's template always writes, so the Llama 3.2 and Llama 3.1 prompts are byte-identical apart from the model.
- **System message and turns.** The system message goes in the system header, and the example turns alternate as for Llama 3.1.

**Checked for every model, on the golden item:**

- the aligned and swapped prompts differ only in the task line, both as messages and after templating;
- the token sequences differ only in the goal object's tokens.

## Tokenization gate (audit C9)

Before a shard scores anything, `tokenization_checks` runs on every item in the shard. It checks three things:

1. **Candidates match the template's reply.** Each candidate text, tokenized alone (as the scorer does), must equal the tokens the chat template produces for that text as the assistant's reply, cut at the template's end-of-turn token. The conversation with the reply must also begin with exactly the prompt the template shows before a reply.
2. **"Yes" and "No" are single tokens.** Each must be one token and equal to the template's token for that reply. The template's variant is the one scored: for Mistral that is `▁Yes` and `▁No`, after `[/INST]`.
3. **The two candidates' first tokens differ.** Otherwise the first-token score would be zero by construction.

The results go to `tokenization_checks.json` in the shard's folder, with the attention implementation and dtype the model loaded with. A failure stops the job before any score.

Also recorded, for information only: the verify_models check, where the prompt text and the candidate are joined as plain text and tokenized together. It agrees with the gate for every Gemma and Llama candidate. It fails for all 9,916 Mistral candidates by design, because the plain join has no space after `[/INST]`. That is why the gate uses the template's own reply.

**Run locally on the real primary items and successes (2026-11-20).** This used transformers 4.51.3 and each pinned tokenizer. Llama 3.2's files were byte-identical to meta-llama's by git blob id.

- **Every candidate passed for all four models.** That is 9,916 candidates per model: two for each of the 4,666 primary items and 292 successes. The gate took 2 to 3 minutes per model on a laptop.
- **First-token pairs.** "You" against "Nothing" for 4,902 items, "There" for 27, "The" for 25 and "This" for 4.
- **Single-token answers:**
  - Gemma: `Yes` 3553 and `No` 1294;
  - Mistral: `▁Yes` 6360 and `▁No` 2538;
  - Llama 3.2: `Yes` 9642 and `No` 2822.
- **Prompt lengths.** The longest prompts are 1,084 tokens (Gemma) to 1,161 (Mistral), so Gemma 2's 4,096-token sliding window never comes into play.

**Mistral needs sentencepiece.** Its tokenizer config sets `add_prefix_space`, so transformers rebuilds the fast tokenizer from `tokenizer.model`, which needs sentencepiece and protobuf in the environment. The prefetch job's `--check` loads every tokenizer on a CPU, so a missing library shows up there and not on a GPU.

## Precision and attention

Weights load in bf16 and log-softmax runs in float32, as elsewhere.

**Gemma 2 needs eager attention.** transformers 4.51.3 defaults Gemma 2 to SDPA. Its SDPA path accepts the soft-capping argument and ignores it: `sdpa_attention_forward` takes `**kwargs`, and only `eager_attention_forward` applies `softcap`. So SDPA silently drops the attention-logit soft-capping (50.0) the model was trained with. Gemma is therefore loaded with `attn_implementation="eager"`. The scorer refuses to continue if the loaded model reports another implementation.

**Gemma's log-odds are bounded.** Its final logits are tanh-capped at 30. Log-odds between two tokens therefore can't exceed 60 nats, and extreme log-odds are compressed. That is one more reason to read the probability-scale and standardised measures beside raw ΔL when comparing across families.

## Measures

Per model, `families_analysis` gives the size series' block (`sizes.model_block`), plus three additions.

**The size series' block:**

- ΔL two-candidate and yes/no, each with a game-clustered 95% CI and a permutation p. Also by history length and for contents-or-open items.
- The probability-scale shift.
- The mean two-candidate success score, and the share of items answered Yes, on failures and successes under each task.
- The AUC under the neutral and aligned tasks, and their difference, in both formats.

**The additions:**

- **Both flip definitions, by name.**
  - `flip_full_L_toward_success`: the full two-candidate L is above 0 under the aligned task and below 0 under the swapped task. This counts one direction only, and leaves NaN pairs out.
  - `flip_first_token_sign_change`: the first-token L changes sign, either direction. This is the confirmatory `decision_flip`.
- **A spread-standardised effect.** The goal effect is divided by the standard deviation of L under the swapped task, on the same items, with a game-clustered CI:

$$
d = \frac{\overline{L_{\text{aligned}} - L_{\text{swapped}}}}{\operatorname{SD}(L_{\text{swapped}})}
$$

  It is computed for both formats. A model that is more extreme in log-odds everywhere does not look more goal-sensitive for that reason alone.
- **First-token coverage per condition.** The median and 5th percentile of the two first tokens' combined probability, the share below 0.5, and the commonest first-token pairs. For yes/no, the combined probability is P(Yes) + P(No).

Each family block also records the shard's tokenization-gate summary and the model's spec: id, revision, release, parameters, template handling and attention.

The reference blocks (Qwen3-8B, Llama-3.1-8B) record how many (item, condition) scores appeared in more than one source run. Only the first is kept, and the largest difference in L among the repeats is reported.

**Checked on the archived runs (2026-11-20).** `families_analysis` was run on a laptop against copies of the archived reference runs (`size_score_qwen3-8b` from fix18; `rq1_score` and `rq1x_score` from fix16; `rq1v_score` from fix17), with no family run yet. It took about 6 minutes for the two reference blocks.

- **Qwen3-8B reproduces the size series exactly:** ΔL 6.12 two-candidate and 2.80 yes/no, a 5.8-point shift and 9.7% first-token flips.
- **The Llama-3.1-8B sources had no repeated (item, condition) scores.**

## Prediction (logged before any family score)

**The goal effect is positive under both formats, two-candidate and yes/no, for every wave A model,** read as the 95% game-clustered CI lying above zero. `families_analysis` reports `prediction_check` per model.

Not predicted: the effect's size, the ordering across families, or whether Llama 3.2 matches Llama 3.1. Those are the exploratory readings this wave is for. If a model can't do the task, as Qwen3-0.6B and 1.7B couldn't, it will show the same way: a yes-share near 1 on failures and an AUC near 0.5. Restricting any reading to competent models would be post hoc.

## Running (ICE)

1. **Prefetch** (CPU job, `families_prefetch.sbatch`). It downloads the pinned revisions into the cache transformers reads, using HF_TOKEN from pace.env, which is never printed. It reports OK or the HTTP status per model, and a refused model doesn't stop the others. All four are gated; access to Gemma 2 and Llama 3.2 was granted by 2026-11-20 (Meta approved both Llama 3.2 models), and Mistral 7B v0.3 is no longer gated. A 403 on ICE would then mean that pace.env's token belongs to another account or lacks read access. Then it checks each tokenizer offline.
2. **Score** each model, one job and one shard (`family_score.sbatch`, any L40S, A40 or A100). A model is about 43,700 forward passes. From the RQ2 sizing (about 0.4 s a pass at 8B), expect roughly:
   - Mistral: 5 h;
   - Gemma: 6 h, since eager attention is slower;
   - Llama 3.2 3B: 2 h;
   - Llama 3.2 1B: 1 h.

   These are estimates. Resume after a timeout with `WF_RESUME=1`.
3. **Analyse** (CPU, `analysis_cpu.sbatch`) once every shard is complete.

## Wave B (planned; not built)

Gemma-4-E4B-it (March 2026) and Ministral-3-8B-Instruct-2512 (December 2025) need architectures that transformers 4.51.3 doesn't have. They will run in a separate, pinned environment, so wave A and the earlier runs keep theirs:

- **Its own environment.** A uv virtual environment with a locked requirements file (exact versions and hashes) for transformers, torch, tokenizers, huggingface_hub, sentencepiece and protobuf. It is selected by its own `SERVE_ENV` in the job script. The environment's `pip freeze` and the transformers version go into each run record.
- **The same code.** Prompts, gate and scorer are unchanged. The models are added to `MODELS` with pinned revisions and their own template handling, checked as above.
- **A bridge check before any wave B score.** Re-score one wave A model (Mistral 7B v0.3) on a few hundred items in the new environment. Its scores should match the wave A scores within float tolerance. A difference would mean the environment, not the model, moves the numbers.
- **Each model's attention and soft-capping defaults are checked in the new transformers source,** as was done for Gemma 2.

## Caveats

- **Pretraining contamination.** ALFWorld text is public, and newer models may have seen more of it. That would affect overall accuracy more than the aligned-minus-swapped difference, but it is a confound for any generation trend.
- **Different prompt formats.** Each model sees its developers' chat format, so family and format can't be separated. Gemma's fold and Mistral's system-message position are the main differences.
- **Single runs.** Wave A is one scoring run per model, so differences between models carry no estimate of run-to-run variation (tracker section 12). With greedy, teacher-forced scoring that variation is GPU numerics only.
