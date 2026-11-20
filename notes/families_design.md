# Model families, wave A: design (logged 2026-11-20, before any score)

Exploratory, not preregistered. Code: `wf/families.py` (runs `family_score_<tag>`, `family_placement_mistral-7b-v0.3` and `families_analysis`), the shared shard runner in `wf/sizes.py`, `wf/tools/prefetch_models.py`; jobs `pace/families_prefetch.sbatch`, `pace/family_score.sbatch` and `pace/family_placement.sbatch`; tests `tests/test_families.py`.

Amended the same day, before any wave A score was read (section "Amendment" below; scoring jobs had been submitted, and some scores may exist on ICE, but nobody has looked at one).

## Why

The tracker's section 13 asks whether post-training explains the goal effect. The confirmatory comparison of Qwen3-8B (April 2025) and Llama-3.1-8B (July 2024) can't separate family, generation and post-training. The Qwen3 size series showed that the effect stays positive at every size within one family. Wave A adds models from other families and from Llama's next release. All four load in the study's environment (transformers 4.51.3).

This wave serves two of the section 13 hypotheses:

- **Distillation (hypothesis 4; reframed as descriptive in the amendment).** Llama 3.2 1B and 3B were pruned and distilled from Llama 3.1 8B and 70B. The first version of this note read a Llama 3.2 effect like Llama 3.1-8B's, against "the near-zero effect of Qwen3's independently trained small sizes", as pointing to the teacher. That contrast does not hold (section "Amendment"): Qwen3's small models were post-trained by distillation from the Qwen3 flagships too, and Llama 3.2 had its own post-training. The comparison stays, as a description. Qwen3-0.6B and 1.7B answered Yes to almost everything.
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
- **Sensitivity check (added in the amendment).** Folding it into the first turn, as for Gemma, tests whether the position matters. It is now planned as a small exploratory run on 300 items (section "Placement check" below).
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

Before a shard scores anything, `tokenization_checks` runs on every item in the shard, and again whenever a shard is resumed with work left. It checks three things:

1. **Candidates match the template's reply.** Both candidate texts of every item in the shard, primary items and successes alike, are checked after the aligned prompt (two per item). Each, tokenized alone as the scorer tokenizes it, must equal the tokens the chat template produces for that text as the assistant's reply, cut at the template's end-of-turn token. The conversation with the reply must also begin with exactly the prompt the template shows before a reply.
2. **"Yes" and "No" are single tokens, the ones the template writes.** The token the scorer reads for each word (the word tokenized alone) must be one token, and must equal the template's rendering of that word as a real reply. This is checked after every yes/no prompt the shard will score: three per primary item (aligned, swapped and neutral tasks) and two per success. The template's variant is the one scored: for Mistral that is `▁Yes` and `▁No`, after `[/INST]`.
3. **The two candidates' first tokens differ.** Otherwise the first-token score would be zero by construction.

**The gate goes through the scorer's own encoding (amendment).** Since the amendment, the gate renders every prompt and reply with the scorer's `_encode` and its template settings (thinking off, plus Llama 3.2's fixed date), and tokenizes candidates with the scorer's `_encode_candidate`, the exact calls scoring makes. Before, it rebuilt the same settings from the model's spec and called the tokenizer itself. The wave A scoring jobs run the earlier gate (commit 65eb591), whose Yes/No part checked only the first item's aligned yes/no prompt. Both gates were run on every real item with the pinned tokenizers (2026-11-20, below): every candidate check is identical, and the wider Yes/No check passes on every yes/no prompt. So the running jobs' gate results stay valid.

The results go to `tokenization_checks.json` in the shard's folder, with the attention implementation and dtype the model loaded with. A failure stops the job before any score.

Also recorded, for information only: the verify_models check, where the prompt text and the candidate are joined as plain text and tokenized together. It agrees with the gate for every Gemma and Llama candidate. It fails for all 9,916 Mistral candidates by design, because the plain join has no space after `[/INST]`. That is why the gate uses the template's own reply.

**Run locally on the real primary items and successes (2026-11-20).** This used transformers 4.51.3 and each pinned tokenizer. Llama 3.2's files were byte-identical to meta-llama's by git blob id.

- **Every candidate passed for all four models.** That is 9,916 candidates per model: two for each of the 4,666 primary items and 292 successes. The gate took 2 to 3 minutes per model on a laptop.
- **First-token pairs.** "You" against "Nothing" for 4,902 items, "There" for 27, "The" for 25 and "This" for 4.
- **Single-token answers:**
  - Gemma: `Yes` 3553 and `No` 1294;
  - Mistral: `▁Yes` 6360 and `▁No` 2538;
  - Llama 3.2: `Yes` 9642 and `No` 2822.
- **Input lengths.** Counting every input the model reads, the two-candidate prompts with their candidate and the yes/no prompts with their answer token, the longest are 1,113 tokens for Gemma, 1,157 for Llama 3.2 and 1,193 for Mistral (a swapped yes/no prompt). Gemma 2's 4,096-token sliding window never comes into play.

**Old and new gate on every real item (2026-11-20).** The gate at 65eb591 (the one the wave A jobs run) and the amended gate (through the scorer's encoding) were both run on all 4,958 items with each pinned tokenizer, transformers 4.51.3 and torch 2.2.2 on a laptop. For all four models every one of the 9,916 candidate checks was identical: the tokens alone, the template's reply, the prompt length and the plain-join check. The records agree on every field (ok, candidates checked, problems, first-token pairs, plain-join mismatches, longest prompt), and so do the Yes and No tokens. The amended Yes/No check, run after all 14,582 yes/no prompts the scoring covers, found no mismatch for any model. The placement check's gate, Mistral with its system text in the first turn, passes on its 300 items (600 candidates, 900 yes/no prompts). The amended gate takes about twice as long, about 20 minutes per model on a loaded laptop.

**Mistral needs sentencepiece.** Its tokenizer config sets `add_prefix_space`, so transformers rebuilds the fast tokenizer from `tokenizer.model`, which needs sentencepiece and protobuf in the environment. The prefetch job's `--check` loads every tokenizer on a CPU, so a missing library shows up there and not on a GPU.

## Precision and attention

Weights load in bf16 and log-softmax runs in float32, as elsewhere.

**Gemma 2 needs eager attention.** transformers 4.51.3 defaults Gemma 2 to SDPA. Its SDPA path accepts the soft-capping argument and ignores it: `sdpa_attention_forward` takes `**kwargs`, and only `eager_attention_forward` applies `softcap`. So SDPA silently drops the attention-logit soft-capping (50.0) the model was trained with. Gemma is therefore loaded with `attn_implementation="eager"`. The scorer refuses to continue if the loaded model reports another implementation.

**Gemma's log-odds are bounded per token.** Its final logits are tanh-capped at 30, so at any one position the log-odds between two tokens can't exceed 60 nats, and extreme log-odds are compressed. The bound is per token: the yes/no score and the first-token score are single positions and obey it, but the full two-candidate L sums over every token of both candidates and can exceed 60. That is one more reason to read the probability-scale and standardised measures beside raw ΔL when comparing across families.

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

**One population for every model (amendment).** Every block, family or reference, is read on the same (item, condition) pairs: the ones a family run scores (`sizes.size_jobs`). The reference runs hold more. `rq1_score` scored the successes under the swapped task too, so before the amendment Llama-3.1-8B's first-token coverage under the swapped task mixed successes into what is failures only for every other model. The other measures were not affected, since they select their items themselves. The records left out are counted in each block (`records_outside_population`). Checked on the archived reference runs (2026-11-20): 584 Llama-3.1-8B records are left out (the 292 successes under the swapped task, in both formats); its swapped-task first-token coverage now counts 4,666 items instead of 4,958 (median 0.80 either way, share below 0.5 7.3% instead of 7.6%), and every other number of both reference blocks is unchanged, Qwen3-8B's included.

**Checked on the archived runs (2026-11-20).** `families_analysis` was run on a laptop against copies of the archived reference runs (`size_score_qwen3-8b` from fix18; `rq1_score` and `rq1x_score` from fix16; `rq1v_score` from fix17), with no family run yet. It took about 6 minutes for the two reference blocks.

- **Qwen3-8B reproduces the size series exactly:** ΔL 6.12 two-candidate and 2.80 yes/no, a 5.8-point shift and 9.7% first-token flips.
- **The Llama-3.1-8B sources had no repeated (item, condition) scores.**

## Amendment (2026-11-20, before any wave A score was read)

Made after a review of the code, the same day. The four scoring jobs had been submitted at commit 65eb591; scores may already exist on ICE, but none has been opened, summarised or looked at, and this amendment was written without them.

A scientific-validity review of the model comparisons followed, also before any score was read; its points are part of this amendment (from "Primary cross-model metrics" on).

**Which format leads the comparison.** The yes/no format is the primary cross-family comparison. The two-candidate format is read within each model only, never across models. The two-candidate score's validity, whether the sign of L agrees with what the model says when it writes its own prediction, was checked for Qwen3-8B only. For Llama-3.1-8B it agreed on only 65% of pilot items (741 of 1,138; `notes/deviations.md`), and none of the wave A models has been checked. The yes/no score reads one token after a direct question, with no wording penalty, so it means the same thing in every family. The logged prediction below is unchanged and still covers both formats; the analysis reports the yes/no result first.

**Run-to-run variation from the GPU.** Scoring is greedy and teacher-forced, so the only run-to-run variation is GPU numerics, and it is not zero. On the archived Qwen3-8B runs, the same prompts scored on different GPU types (`rq1_score` against `size_score_qwen3-8b`) differ by 0.18 nats per item on average in two-candidate L, at most 2.8. Shard pairs either matched exactly on every item or differed on nearly all, as expected if same-type GPUs reproduce each other exactly. The mean differences largely cancel between conditions, so ΔL moves by about 0.02 nats. Differences in yes/no ΔL between models under 0.3 nats are treated as noise (the noise floor below). A resumed shard that lands on another GPU type mixes two sets of numerics in one run; resume on the type the shard started on (its `jobs.jsonl` records the GPU), by passing that type as `--constraint`.

**The tokenization gate** now goes through the scorer's own encoding, and its Yes/No part covers every yes/no prompt (section "Tokenization gate"). **Every model is read on one population** (section "Measures"). **Mistral's system-text placement** gets a small check (next section).

**Primary cross-model metrics.** Two metrics, both on the yes/no format, are primary for comparing models; ΔL in nats and the spread-standardised d are secondary. Write $L_c(i)$ for item $i$'s yes/no log-odds, $\log P(\text{Yes}) - \log P(\text{No})$, under task $c$, with $F$ the primary items (determined failures) and $S$ the determined successes.

The first is the item share moved toward Yes: the share of failures whose log-odds are higher under the aligned task than under the swapped task,

$$
\text{share} = \frac{1}{|F|} \sum_{i \in F} \mathbf{1}\left[L_{\text{aligned}}(i) > L_{\text{swapped}}(i)\right].
$$

Its null value is one half. A tie counts as not moved.

The second is the goal-to-evidence ratio: the goal effect divided by the evidence effect, the separation between determined successes and failures under the neutral task,

$$
R = \frac{\frac{1}{|F|}\sum_{i \in F} \left[L_{\text{aligned}}(i) - L_{\text{swapped}}(i)\right]}{\frac{1}{|S|}\sum_{j \in S} L_{\text{neutral}}(j) - \frac{1}{|F|}\sum_{i \in F} L_{\text{neutral}}(i)}.
$$

Multiplying every logit by one factor, as a different temperature or a softer output layer would, leaves both metrics unchanged, so they compare across families where raw ΔL does not. R says how far the goal moves the answer in units of how far real evidence moves it; it is undefined (reported as missing) when the evidence effect is not positive. Each metric gets a game-clustered 95% bootstrap CI (games resampled, failures and successes together) and a two-sided permutation p in which each game's items flip sign together. For R the p is that of its numerator: the denominator does not change under the flips, so R is zero exactly when the mean goal effect is. Both are computed for every wave A model and for the Qwen3-8B and Llama-3.1-8B reference blocks (`primary_metrics`).

**Competence and coverage gate (fixed now, before data; revised by the second amendment below).** As first written, a model was interpretable only if all three held:

- its neutral-task yes/no AUC (successes against failures) is at least 0.70;
- under the neutral task it answers Yes (log-odds above zero) to between 5% and 95% of failures;
- the median of P(Yes) + P(No) is at least 0.5 under every yes/no task.

The rule in force is the second amendment's.

A model that fails is reported as not interpretable, never as a null result: a model that cannot tell successes from failures, or answers one word to everything, or barely puts its mass on Yes and No, says nothing about goal sensitivity. This replaces the sentence in the prediction section that called such a restriction post hoc; it was post hoc only if made after the data. The analysis computes the gate per model (`competence_gate`), the reference models included.

**Open issue in the gate, to settle before any wave A score is read.** Run on the archived reference scores (no wave A score involved), the gate as written passes Llama-3.1-8B (neutral AUC 0.97, Yes to 5.3% of failures, median Yes/No mass 0.996) but fails Qwen3-8B, the study's main model, on the second condition alone: under the neutral task Qwen3-8B answers Yes to only 0.15% of failures (AUC 0.99, mass 1.00). Answering No to nearly every failure is correct, not degenerate, so the lower bound on failures excludes the most accurate model rather than one that answers one word to everything. A rule that keeps the intent would bound the Yes share on failures above (at most 95%) and the Yes share on determined successes below (at least 5%). Settled by the second amendment (next paragraph).

**Second amendment (2026-11-20, still before any wave A score was read): the competence gate.** The lower bound on the Yes share on failures was meant to exclude a model that answers No to everything. It cannot do that job: a model that answers No to failures is right, so the bound excludes accurate models, as it did Qwen3-8B. The problem was found on the archived reference scores alone; no wave A score had been opened, and none has been since. "No to everything" shows instead as a model that also answers No to the determined successes, so the lower bound moves there. The rule now in force: a model is interpretable only if all four of these hold on the yes/no format under the neutral task:

- the AUC (determined successes against failures) is at least 0.70;
- it answers Yes (log-odds above zero) to at most 95% of failures, which excludes "Yes to everything";
- it answers Yes to at least 5% of determined successes, which excludes "No to everything";
- the median P(Yes) + P(No) is at least 0.5.

All four are read under the neutral task, so the coverage condition is now the neutral task's median rather than every yes/no task's. Rerun on the archived reference scores, both reference models pass. Llama-3.1-8B has AUC 0.97, answers Yes to 5.3% of failures and 85% of successes, and its median mass is 1.00. Qwen3-8B has AUC 0.99, answers Yes to 0.15% of failures and 59% of successes, and its median mass is 1.00. The code (`competence_gate`), its tests and its mutants follow this rule.

**Flips.** Yes/no flips and yes-shares are read as the model's choice, Yes when its log-odds are above zero, not as the mean P(Yes): the share answering Yes per task, and the shares of failures that go from No under the swapped task to Yes under the aligned task, and back (`argmax_answers`, with game-clustered CIs). The two-candidate flip definitions stay as before, within model.

**One primary family.** The primary family is the item share and R for each of the four wave A models: eight tests, Holm-corrected within the family at α = 0.05. A test is supported when its Holm p is below 0.05, its estimate lies on the goal side (share above one half, R above zero) and the model passed the competence gate; a model that failed is "not interpretable" whatever its p. The family is fixed at eight tests, so a model that is not interpretable or still waiting leaves it incomplete, never smaller. Everything else is descriptive: ΔL, d, the two-candidate format, comparisons between models, the reference models, the placement check.

**Comparing models.** Comparisons are made on the same items with a paired game-clustered bootstrap: games are resampled once and both models are read on the same resample. Each wave A model is compared with each reference model (`compare_models`). Two models' R "match" when R is within a factor of 1.5, tested as two one-sided tests (TOST) on the log of the ratio: the 90% paired interval of $\log(R_A / R_B)$ must lie inside $\pm \log 1.5$. They "differ" when the 95% interval excludes zero. An interval that does neither is inconclusive, which is not a match. Differences in yes/no ΔL under about 0.3 nats between models are treated as noise (the GPU variation above, with margin), and flagged as such; the item-share difference is reported with its 95% interval. All comparisons are descriptive.

**Distillation reframed as a descriptive comparison.** The Qwen3 small models were not trained independently of the flagships. The Qwen3 Technical Report (arXiv 2505.09388, section 4.5, "Strong-to-Weak Distillation") describes a distillation pipeline "specifically designed to optimize lightweight models", among them Qwen3-0.6B, 1.7B, 4B, 8B and 14B, with off-policy distillation from teacher outputs and on-policy logit alignment with Qwen3-32B or Qwen3-235B-A22B (checked 2026-11-20). Llama 3.2 1B and 3B took Llama 3.1 8B and 70B logits during pretraining, but were then post-trained in their own rounds of supervised fine-tuning, rejection sampling and DPO (Llama 3.2 model card, checked 2026-11-20). So both small-model lines carry distillation from their flagships and each has its own post-training: a Llama 3.2 effect like or unlike Llama 3.1's cannot separate inheritance from a shared recipe. The comparison is reported as a description of how the effect sizes line up, not as a causal test of hypothesis 4.

**Outcome table (pre-data).** How each pattern on the primary metrics will be read. "Matches" and "differs" are the TOST and 95% rules above; "positive" means supported in the primary family.

| Pattern | Reading |
|---|---|
| A model fails the competence gate | Not interpretable. No statement about its goal sensitivity, in either direction. |
| Every interpretable wave A model positive on both share and R | The goal effect is not specific to Qwen3 and Llama 3.1: it appears in every interpretable 2024 instruct model tested. |
| Positive R but share not supported | The effect is carried by a minority of items with large shifts; reported as such, not as a general shift. |
| Share supported but R not | Many items move a little, too little against the evidence effect to count as a strong bias; reported as a weak, widespread shift. |
| Neither supported, model interpretable | No detectable goal effect in that model: a family or generation without the bias, to be followed up in wave B. |
| Gemma 2 and Mistral match Llama 3.1 and differ from Qwen3-8B | Consistent with a 2024-generation or post-training-style explanation; family, generation and recipe still cannot be separated. |
| Gemma 2 and Mistral differ from each other | Family matters more than generation. |
| Llama 3.2 matches Llama 3.1 | Descriptive similarity within the Llama line; not evidence that the bias is inherited by distillation (see above). |
| Llama 3.2 differs from Llama 3.1 | Descriptive difference within the line; size, distillation and post-training all differ, so no cause is named. |
| A comparison inconclusive, or a ΔL difference under 0.3 nats | "No difference detected", never "the same". |
| Mistral's placement moves its effect (placement check rule) | Mistral's position in every comparison is reported as format-dependent, with both numbers. |

## Placement check: Mistral's system text (exploratory, logged before data)

**Question.** Mistral's template writes the system text into the last user turn, just before the query, where Gemma gets it at the start of the first turn and Llama and Qwen in a system header. Does Mistral's goal effect depend on that placement?

**Run.** `family_placement_mistral-7b-v0.3` scores Mistral 7B v0.3, same weights and revision, with the system text folded into the first user turn exactly as for Gemma (the system text, a blank line, the first turn's text). It scores 300 primary items, drawn with a fixed seed from all primary items (`placement_items`), under the aligned and swapped tasks, in both formats: 1,800 forward passes, about 20 to 30 GPU-minutes (`pace/family_placement.sbatch`). The scores are saved under the tag `mistral-7b-v0.3-firstturn`, so they can never mix with the native run. The same tokenization gate runs first, on those 300 items.

**Readout.** `families_analysis` reports, on the same 300 items, the goal effect (aligned minus swapped) under the native placement (from `family_score_mistral-7b-v0.3`) and under the first-turn placement, in both formats, each with a game-clustered 95% CI, and the per-item difference between the two placements with its own CI.

**How it will be read.** No prediction. If the difference's CI excludes zero and the difference is larger than a fifth of the native effect, Mistral's position in the cross-family comparison is reported as depending on prompt format, and both numbers are given. Otherwise placement is reported as not moving the effect on this sample. Either way it is one model, one alternative placement and 300 items; it does not make the formats of the other families comparable.

## Prediction (logged before any family score)

**The goal effect is positive under both formats, two-candidate and yes/no, for every wave A model,** read as the 95% game-clustered CI lying above zero. `families_analysis` reports `prediction_check` per model.

Not predicted: the effect's size, the ordering across families, or whether Llama 3.2 matches Llama 3.1. Those are the exploratory readings this wave is for. If a model can't do the task, as Qwen3-0.6B and 1.7B couldn't, it will show the same way: a yes-share near 1 on failures and an AUC near 0.5. Such a model is now reported as not interpretable under the competence gate fixed in the amendment (before data); the first version of this note called that restriction post hoc, which it would have been only if decided after the data.

## Running (ICE)

1. **Prefetch** (CPU job, `families_prefetch.sbatch`). It downloads the pinned revisions into the cache transformers reads, using HF_TOKEN from pace.env, which is never printed. It reports OK or the HTTP status per model, and a refused model doesn't stop the others. All four are gated; access to Gemma 2 and Llama 3.2 was granted by 2026-11-20 (Meta approved both Llama 3.2 models), and Mistral 7B v0.3 is no longer gated. A 403 on ICE would then mean that pace.env's token belongs to another account or lacks read access. Then it checks each tokenizer offline.
2. **Score** each model, one job and one shard (`family_score.sbatch`, any L40S, A40 or A100). A model is about 43,700 forward passes. From the RQ2 sizing (about 0.4 s a pass at 8B), expect roughly:
   - Mistral: 5 h;
   - Gemma: 6 h, since eager attention is slower;
   - Llama 3.2 3B: 2 h;
   - Llama 3.2 1B: 1 h.

   These are estimates. Resume after a timeout with `WF_RESUME=1`.
3. **Placement check** (`family_placement.sbatch`, about 30 GPU-minutes), at the amended commit.
4. **Analyse** (CPU, `analysis_cpu.sbatch`) once every shard is complete, at the amended commit. The scores made at 65eb591 are read as they are: an analysis requires one commit per run, not the analysis's own.

## Wave B (planned; not built)

Gemma-4-E4B-it (March 2026) and Ministral-3-8B-Instruct-2512 (December 2025) need architectures that transformers 4.51.3 doesn't have. They will run in a separate, pinned environment, so wave A and the earlier runs keep theirs:

- **Its own environment.** A uv virtual environment with a locked requirements file (exact versions and hashes) for transformers, torch, tokenizers, huggingface_hub, sentencepiece and protobuf. It is selected by its own `SERVE_ENV` in the job script. The environment's `pip freeze` and the transformers version go into each run record.
- **The same code.** Prompts, gate and scorer are unchanged. The models are added to `MODELS` with pinned revisions and their own template handling, checked as above.
- **A bridge check before any wave B score.** Re-score one wave A model (Mistral 7B v0.3) on a few hundred items in the new environment. Its scores should match the wave A scores within float tolerance. A difference would mean the environment, not the model, moves the numbers.
- **Each model's attention and soft-capping defaults are checked in the new transformers source,** as was done for Gemma 2.

## Caveats

- **Pretraining contamination.** ALFWorld text is public, and newer models may have seen more of it. That would affect overall accuracy more than the aligned-minus-swapped difference, but it is a confound for any generation trend.
- **Different prompt formats.** Each model sees its developers' chat format, so family and format can't be separated. Gemma's fold and Mistral's system-message position are the main differences.
- **Single runs.** Wave A is one scoring run per model, so differences between models carry no estimate of run-to-run variation (tracker section 12). With greedy, teacher-forced scoring that variation is GPU numerics only: about 0.02 nats in ΔL across GPU types on the archived Qwen3-8B runs (section "Amendment").
- **Two-candidate validity unchecked outside Qwen3.** Hence the yes/no format leads (section "Amendment").
