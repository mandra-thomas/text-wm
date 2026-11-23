# Qwen3-8B goal effect: pipeline audit (exploratory, 2026-11-22)

**Status: exploratory.** This note checks whether the scoring pipeline, or a detail of the prompt format, could produce Qwen3-8B's large goal effect in ALFWorld. It does not re-verdict the confirmatory RQ2 result (`rq2_results.md`), which stands as reported. A separate audit reads the stored scores themselves (scale, token positions, heavy tails); this one re-scores a sample from scratch.

## 1. The question and what was done

RQ2 scores 4,666 failing ALFWorld actions whose failure the history already settles. For each item, the world model sees a few-shot chat prompt that ends with the agent's history and the action, and two candidate replies are scored by teacher forcing: the action's success text, written $o^{+}$ (for example "You pick up the pencil 1 from the desk 1."), and the failure text $o^{-}$, "Nothing happens.". The log-probability of a candidate is the sum over its tokens,

$$\log p(o \mid x) = \sum_{i=1}^{n} \log p(o_i \mid x, o_{<i}),$$

where $x$ is the prompt and $o_1, \dots, o_n$ are the candidate's tokens. The two-candidate log-odds is

$$L = \log p(o^{+} \mid x) - \log p(o^{-} \mid x),$$

and the goal effect for an item is

$$\Delta L = L_{\text{aligned}} - L_{\text{swapped}},$$

where the aligned prompt carries the agent's real task line and the swapped prompt the same line with the goal object replaced by another object in the room. Everything else in the two prompts is byte-identical. The confirmatory mean of $\Delta L$ is 6.12 nats for Qwen3-8B (thinking off) and 1.65 for Llama-3.1-8B; other model families give 0.8 to 1.1.

The audit assumed a bug until shown otherwise and did five things:

1. A line-by-line review of the scoring path (`wf/scoring/logodds.py`, `wf/scoring/prompts.py`) at the commit that produced the stored scores (`ddeabc2`) and at the current main, plus the pinned Qwen3-8B chat template itself.
2. A scorer written from scratch (`scripts/qwen_audit/independent_score.py`), run on a seeded sample of 200 primary items, compared item by item with the stored `rq2_score` values.
3. Two further code paths on the same items: vLLM's `prompt_logprobs` (a different engine, attention kernel and batching), and a plain transformers forward pass with float32 weights and eager attention.
4. Six prompt formats for Qwen, aligned against swapped on the same 200 items.
5. Llama-3.1-8B through the same independent scorer and vLLM, as a control for a bug that would affect both models.

### The sample

`scripts/qwen_audit/sample_items.py` reads the stored scores from the checkout that produced them (fix11, commit `ddeabc2`, read-only), takes every item that has aligned and swapped scores for both co-primary models (exactly 4,666, as reported), and draws 200 with `random.Random("20260924|qwen_audit_pipeline").sample` over the sorted item ids. The 200 items come from 94 games and all four stored shards (49, 57, 47 and 47 items from shards 0 to 3). On this sample the stored Qwen effect is 7.34 nats and the stored Llama effect 1.73, against 6.12 and 1.65 on all 4,666 items: the sample happens to draw somewhat more of Qwen's heavy upper tail. Every comparison below is therefore made within the sample, format against format, rather than against the population's 6.12. The "first 50 items" requested for the Llama control are the first 50 of the sample's draw order; Llama was in fact scored on all 200, and both are reported.

### Compute

Three one-GPU jobs from a fresh checkout (`/storage/ice1/2/1/gmat3/text-wm-qwenaudit2`, branch `qwen-audit-pipeline`): Qwen in bf16 with every format, then the study's own scorer, on an A40 (job 7245843, 18 min); Llama in every format, then Qwen with float32 weights, on an L40S (7245844, 19 min); vLLM for both models on an L40S (7245845, 8 min). About 45 GPU-minutes in all. Job script: `pace/qwen_audit.sbatch`.

## 2. Code review of the scoring path

Each point below names what could go wrong, what the code does, and how it was checked.

**Chat template and the thinking-off block.** The scorer renders every prompt with the model's own `apply_chat_template`, generation prompt on, with `enable_thinking=False` (`LogoddsScorer.template_settings`; at `ddeabc2` the same keyword was passed directly in `_encode`). The pinned Qwen3-8B template (`b968826`) writes `<|im_start|>assistant\n` and then, only when thinking is off, the empty block `<think>\n\n</think>\n\n`. This is exactly where Qwen's template puts it: after the final generation prompt and nowhere else. The few-shot answers are earlier assistant turns, and the template renders those without any think block (it only renders reasoning for assistant turns after the last user message). So in the scored prompt the three example answers follow a bare `assistant\n` header, while the scored answer follows the empty think block. That mismatch is Qwen's own design, it is identical in the aligned and swapped prompts, and section 4 tests what happens when it is changed. The system message is rendered as Qwen's template renders any system message; Qwen3's template adds no default system text.

**Aligned and swapped differ only in the task line.** The independent scorer builds its messages from the item text with its own task-line swap and its own swapped line (first whole-word occurrence of the goal object replaced). For all 200 items and both conditions, its messages equal `build_messages` exactly, and its swapped line equals the stored `swapped_task_line`. Every sampled history has exactly one task line, so the difference between the builder at `ddeabc2` (which would have prepended a task line if none existed) and the stricter current one does not arise here.

**BOS and EOS.** Qwen3 has no BOS token (`bos_token` is null and `add_bos_token` false), and the rendered prompts contain none. Llama's template text begins with `<|begin_of_text|>`; the token ids contain exactly one BOS in all 400 Llama prompts, so `apply_chat_template(tokenize=True)` does not add a second. The ids from `apply_chat_template(tokenize=True, return_dict=True)` equal the tokenization of the rendered text for every prompt and both models. No end-of-turn token is appended after either candidate, so neither candidate is charged for ending the reply. That is a design choice rather than an error, and it applies identically to both candidates and both conditions.

**Where the assistant turn begins, and the candidate boundary.** The study's scorer tokenizes the candidate on its own and concatenates it after the prompt ids. If the tokenizer would merge characters across that boundary when the text is tokenized as one string, the scored tokens would differ from what the model generates. The independent scorer instead appends the candidate to the rendered prompt text and tokenizes the whole string once. For every prompt, format and candidate (4,800 candidate scorings for Qwen, 3,200 for Llama), the prompt's tokens were an exact prefix of the joint tokenization, and the candidate's in-context tokens equalled its tokenization alone. The first scored tokens are "You" and "Nothing" in every chat format (" You" and " Nothing" in the plain-text format, where a space precedes them).

**Offsets.** The study's scorer reads the log-probability of candidate token $i$ (counting from 0) from the logits at position $p + i - 1$, where $p$ is the prompt length. That is the right position: the logits at a position predict the next token. The independent scorer slices positions $p-1$ to $p+n-2$ and indexes them by the candidate ids in one vectorised step. An off-by-one would show as a large disagreement; there is none (section 3).

**Batching, padding, attention mask, position ids.** The study's scorer runs one unpadded sequence per forward pass, with no attention mask passed (transformers then attends to every token) and default position ids. There is no batching, so there is no padding side to get wrong. The independent scorer passes an explicit all-ones mask. vLLM batches 1,600 sequences with its own scheduler and kernels.

**Precision.** Weights are bfloat16; the model's logits come out in bfloat16 and are converted to float32 before `log_softmax` (since `db3eb56`, which precedes `ddeabc2`). The conversion happens after the logits have already been rounded, so differences between two logits are quantised. In the stored scores, 96.5% of Qwen's first-token log-odds lie on a 0.25-nat grid and 47.5% on a 0.5-nat grid, against 80.5% of Llama's on a 0.125-nat grid: Qwen's logits are roughly twice as large in magnitude, so bfloat16 resolves them more coarsely. This rounding is symmetric noise of a few tenths of a nat per logit, not a bias, and the float32 run below confirms it moves the mean effect by only 0.05 nats.

**State between prompts.** The scorer's result cache is keyed by model tag, condition, item id and a hash of the messages and both candidates, so two different prompts cannot share an entry. Each forward pass starts fresh; the key-value cache it returns is discarded. Nothing carries over from one prompt to the next.

**What differs between Qwen and Llama in the code.** Only the chat template and the `enable_thinking=False` keyword, which Llama's template ignores. Scoring, indexing and precision are shared.

**Conclusion of the review.** No bug was found. The one model-specific format detail is the empty think block before the scored answer, which is Qwen's own documented non-thinking format; section 4 measures how much it matters.

## 3. Independent re-implementation and second code paths: reproduction of the stored scores

The table compares each re-scoring with the stored `rq2_score` values on the same 200 items, in the study's prompt format. "Mean abs. diff." is over the 400 values of $L$ (200 items in two conditions). The stored shards 0 to 2 ran on A40s and shard 3 on an L40S, so the GPU type of each re-scoring matters.

| model and path | GPU | correlation of L | correlation of the item effect | mean abs. diff. of L (nats) | on stored shards of the same GPU type | effect, stored vs here (nats) |
|---|---|---|---|---|---|---|
| Qwen, independent scorer, bf16 | A40 | 0.9997 | 0.9989 | 0.082 | 0.000 (shards 0–2, 153 items) | 7.341 vs 7.341 |
| Qwen, study's scorer rerun, bf16 | A40 | 0.9997 | 0.9989 | 0.082 | 0.000 | 7.341 vs 7.341 |
| Qwen, float32 weights, eager attention | L40S | 0.9992 | 0.9975 | 0.339 | 0.372 (shard 3) | 7.341 vs 7.287 |
| Qwen, vLLM prompt_logprobs | L40S | 0.9989 | 0.9962 | 0.357 | 0.290 (shard 3) | 7.341 vs 7.328 |
| Llama, independent scorer, bf16 | L40S | 0.9991 | 0.9952 | 0.087 | 0.000 (shard 3, 47 items) | 1.726 vs 1.718 |
| Llama, vLLM prompt_logprobs | L40S | 0.9988 | 0.9943 | 0.123 | 0.125 (shard 3) | 1.726 vs 1.731 |

What this shows:

- **The independent scorer reproduces the stored scores exactly on the same GPU type.** On the A40 it matches every stored Qwen score from the A40 shards to the last digit, and on the L40S it matches every stored Llama score from the L40S shard to the last digit. The differences that remain come only from items scored on the other GPU type (Qwen 0.35 nats on shard 3; Llama 0.10 on shards 0 to 2), which is the known size of cross-GPU bfloat16 differences (`reproducing.md`, "Known limits"). The study's own scorer, rerun in the same job, agrees with the independent one bit for bit on all 400 Qwen scores.
- **The effect does not depend on the engine or on precision.** vLLM, with its own model code, FlashAttention kernels and batching, gives 7.33 against 7.34; float32 weights with eager attention give 7.29. Both differ from the stored scores by about a third of a nat per score, in no consistent direction.
- **The Llama control behaves the same way,** so no bug is affecting both models together either: the first 50 items give 2.01 stored against 2.01 and 2.03 here, and all 200 give 1.73 against 1.72 and 1.73.

## 4. Prompt-format sensitivity

Each format below was scored on the same 200 items, aligned against swapped, with the independent scorer in bf16 (Qwen on the A40, Llama on the L40S). The formats are:

- **Native**: the study's prompt (system message, three examples, query; Qwen's empty think block before the scored answer).
- **No think block**: Qwen's template rendered with thinking on, so the scored answer follows a bare assistant header, like the example answers. This is how Qwen2.5-style templates and most other chat models format the answer.
- **Think block everywhere**: the empty think block also opens each of the three example answers, so the examples show the same format as the scored answer.
- **System folded**: no system message; the system text opens the first user turn.
- **Zero-shot**: system message and the query only, without the three examples.
- **Plain text**: no chat template; system text, examples and query as plain text separated by blank lines, the candidate after "Next observation:" and a space.

The interval is a 95% percentile bootstrap over games (2,000 resamples). "Step 0" is the 22 sampled items whose history has no actions yet.

| format | Qwen effect (nats) | 95% CI | Qwen mean L, aligned / swapped | Qwen effect at step 0 | Llama effect (nats) | Qwen ÷ Llama |
|---|---|---|---|---|---|---|
| native (study) | 7.34 | 6.27 to 8.42 | −14.2 / −21.5 | 12.1 | 1.72 | 4.3 |
| no think block | 3.87 | 3.29 to 4.45 | −3.6 / −7.5 | 2.5 | (no analogue) | 2.3 against native Llama |
| think block everywhere | 6.90 | 5.93 to 7.83 | −11.0 / −17.9 | 7.7 | (no analogue) | |
| system folded | 7.77 | 6.67 to 8.91 | −12.7 / −20.5 | 13.3 | 1.58 | 4.9 |
| zero-shot | 4.81 | 3.94 to 5.64 | +7.2 / +2.4 | 3.8 | 2.19 | 2.2 |
| plain text | 3.63 | 3.11 to 4.10 | −4.1 / −7.7 | 2.5 | 0.85 | 4.3 |

vLLM gives the same picture for the no-think-block format (3.84 against 3.87 from transformers). In every format, 83% to 94% of Qwen's item effects are positive.

How to read this:

- **The effect survives every format, but its size hinges on one detail.** Nowhere does it fall below 3.6 nats on this sample, and the interval is far from zero in every format. But removing the empty think block roughly halves it (7.34 to 3.87, ratio 0.53), and a plain-text prompt, which has no think block either, gives nearly the same 3.63. Putting the think block in front of the example answers as well barely changes the native value (6.90). So what doubles the effect is the presence of the empty think block right before the scored answer. Its mismatch with the examples does not matter.
- **The other format details barely matter.** Folding the system text into the first user turn changes the effect by +0.4 nats. Dropping the examples cuts it to 4.81, but that format is a different measurement altogether: without examples both conditions favour the success text (mean L is positive), so the comparison runs in another range of the log-odds.
- **The step-0 concentration is largely a think-block effect.** The confirmatory note reports 11.0 nats at history length 0 against 5 to 6 nats later. On this sample's 22 step-0 items, the native format gives 12.1. Without the think block, or in plain text, those same items give 2.5, below the later-step average. With only 22 items this is indicative, not settled.
- **Qwen stays larger than Llama in every format,** by a factor of 2.2 to 4.9. In the formats the two models share, the ratio is 4.3 in native and in plain text, 4.9 with the system folded, and 2.2 zero-shot. Against Llama's native value, Qwen without the think block is still 2.3 times larger. So Qwen's outsized effect relative to other models does not come from the think block alone. About half of its absolute size does.

### Where in the candidate the effect sits (float32 and bf16, native and other formats)

The independent scorer keeps the log-probability of every token of the success text, so the item effect can be split into five parts that add up to it: the first token "You", the second token (the verb: " move", " pick", " clean", " cool"), the later tokens that spell the object, all other later tokens, and the "Nothing happens." text, which enters with a minus sign (`scripts/qwen_audit/token_split.py`).

| model and format | effect | "You" | verb | object tokens | other later tokens | "Nothing happens." |
|---|---|---|---|---|---|---|
| Qwen native | 7.34 | 2.44 | 3.85 | 0.19 | 0.38 | 0.49 |
| Qwen, no think block | 3.87 | 0.32 | 2.36 | 0.16 | 0.24 | 0.79 |
| Qwen plain text | 3.63 | 1.07 | 2.01 | 0.06 | −0.01 | 0.49 |
| Llama native | 1.72 | 0.23 | 0.90 | −0.02 | 0.06 | 0.55 |

In both models, the largest single share is the verb right after "You". Once the model has written "You", the swapped goal makes the action's own verb much less likely, and it presumably favours some other continuation (the next tokens were not inspected). The object's name and the rest of the sentence contribute almost nothing, because once the verb is fixed the rest is copied from the action. What the think block adds is mostly on "You" itself: 2.44 nats with it and 0.32 without. Right after the empty think block, the goal moves Qwen's choice between starting "You…" and starting "Nothing…" far more than it does after a bare assistant header. The float32 run splits the same way (2.41, 3.83, 0.18, 0.36, 0.51), so the split does not come from bfloat16 rounding.

## 5. Verdict

**The 6 nats is not a pipeline artefact.** An independent scorer, written without the study's scoring code, reproduces the stored scores bit for bit on the same GPU type, and a second engine (vLLM) and a float32 pass agree to within a few hundredths of a nat on the mean effect. The template, BOS handling, candidate boundary, offsets, absence of padding and absence of state between prompts were each checked directly and are correct. The Llama control reproduces in the same way.

**But about half of its size is a property of the prompt format,** namely Qwen3's own thinking-off block before the scored answer. That block is the documented way to run Qwen3 with thinking off, so the confirmatory measurement used the model as intended, and nothing about it is a bug. Still, an equally reasonable prompt without the block (Qwen's template with thinking on but no reasoning written, or plain text) gives about half the effect on the same items: 3.6 to 3.9 nats against 7.3 here. If that ratio held on all 4,666 items, it would correspond to roughly 3 nats rather than 6.12. That figure is an extrapolation and was not measured. In every format, Qwen's effect remains well above zero and two to five times Llama's.

What would change these conclusions: a different sample (this one runs high on Qwen's tail, which is why comparisons are made within it), prompt formats not tried here (a different system text or example set), and the 22-item basis for the step-0 statements. A cheap next step, if wanted, is a no-think-block sensitivity run of the study's own scorer on all 4,666 items. It would take about an hour of GPU time and would be reported as exploratory beside the confirmatory number, not instead of it.

## 6. Files

- `scripts/qwen_audit/sample_items.py`: draws the sample from the stored run (CPU, read-only on the source).
- `scripts/qwen_audit/independent_score.py`: the independent scorer and the six formats; `--render-only` runs every check on CPU without a model.
- `scripts/qwen_audit/repo_score.py`: the study's own scorer on the sample, for the same-GPU comparison.
- `scripts/qwen_audit/vllm_score.py`: the vLLM `prompt_logprobs` path.
- `scripts/qwen_audit/compare.py`: the reproduction and format tables; writes `summary.json`.
- `scripts/qwen_audit/token_split.py`: the per-token split.
- `pace/qwen_audit.sbatch`: the job script (`AUDIT_STEP=qwen_hf`, `other_hf` or `vllm`).
- `notes/qwen_effect_audit_pipeline_metrics.json`: a copy of `summary.json`.
- Per-item outputs are in `runs/qwen_audit/` of the ICE checkout `/storage/ice1/2/1/gmat3/text-wm-qwenaudit2` (outside git, like all run data).

Software: transformers 4.51.3, torch 2.6.0 and vLLM 0.8.5 (V1 engine, FlashAttention) from the study's serve environment; models at the pinned snapshots in `wf/constants.py`, loaded offline from the shared cache.
