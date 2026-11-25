# TextCraft round 3, deviation D-T1: the thinking arm's padding check runs in float32

Date: 2026-11-23 (diagnosed and approved by the user the same day), after the freeze tag `textcraft-r3-design-frozen-2026-11-23` (0d6f460) and before any thinking-arm
generation was kept. The thinking arm is exploratory and outside every test family (design 8.2, 14.S10), so this
changes no family, test, threshold or confirmatory reading.

## What happened

The thinking smoke run (`textcraft_r3_think_smoke_qwen3.5-9b`, job 7252013, checkout `text-wm-fix47` at the tag)
stopped at its start check (design 14.S10). The check compares the next-token log-probabilities of the shortest prompt
of the first batch, run alone and run left-padded inside its batch, over the alone run's 20 most probable tokens. The
tolerances were borrowed from wave B's attention-agreement rule: at most 0.10 nats on average and 1.0 on any token.
The run measured 0.111 nats on average (0.328 at most) with 105 padding tokens.

## Diagnosis (job 7252180, one L40S, 11 minutes, separate checkout `text-wm-fix48` at the tag; no study quantity computed)

The job took the first batch's eight prompts (1,116 to 1,221 tokens) and compared each prompt's next-token
log-probabilities run alone with the same prompt under other conditions. Each comparison used the same 20 tokens as
the check.

- **Run alone twice:** the two runs were bitwise identical (difference 0.000).
- **No padding, other batch shapes:** the same prompt in a batch of 2 or 8 copies of itself, with no padding at all,
  differed from the alone run by 0.084 nats on average (median over prompts and batch sizes; up to 0.170). Three of
  the 16 comparisons exceeded the check's 0.10 limit.
- **Left padding, batch of one:** the prompt left-padded with 8, 32, 105 or 400 padding tokens differed by 0.077,
  0.068, 0.075 and 0.088 (medians over the eight prompts). The difference does not grow with the amount of padding.
  Passing position ids derived from the mask changes nothing systematic.
- **The real batch:** the batch as the check builds it reproduces 0.111 for the shortest prompt. The longest prompt of
  that batch carries no padding at all, yet it differs from its alone run by 0.062.
- **In float32:** every one of these comparisons falls to at most 0.0001 nats, including the real batch with 105
  padding tokens and a single prompt with 400. If padding leaked into the computation, the difference would survive in
  float32. It does not, so padding is masked exactly. The reason is visible in the code: Qwen3.5's Gated DeltaNet
  layers zero the hidden states at padded positions before the convolution and the delta rule, and the attention
  layers mask them.
- **Generated text:** greedy continuations of 256 tokens are token-identical to the alone run in every condition:
  padding of 8, 105 or 400 tokens, a batch of 8 copies, and the real batch for both its shortest and its longest
  prompt.

The 0.111 is therefore ordinary bf16 noise from the shape of the computation. The check's 0.10 tolerance sits inside
that noise, so a same-length batch with no padding would fail it about one time in five.

## Change (approved 2026-11-23)

`r3_think.start_checks` runs the prompt check and `padded_check` once at job start on a float32 copy of the model
(`_Engine(dtype="float32")`, a `LogoddsScorer` with every setting of the thinking spec and `dtype="float32"`) loaded
from the same verified node-local snapshot; `padded_check` reads the last position's logits only (`logits_to_keep=1`),
as the diagnosis did, because a float32 copy has no room for a batch's full logits. The tolerance becomes 0.01 nats on average and 0.05 on any token; the measured float32
difference is at most 0.0001. The copy is closed (also if the check raises), and only then is the bf16 model used for generation and scoring is loaded
exactly as before. The model is reloaded rather than cast back to bf16, because casting back would also change
float32 buffers such as RoPE's inverse frequencies. Nothing else changes: the batches, the sampling, the 2,048-token
cap, the readout and the analysis. The 0.10 / 1.0 bf16 rule is withdrawn for this check, because it does not separate
padding from batch noise.

Rejected:

- **Batch size 1.** It costs about 4.3 times the batched time: 363 seconds against 85 for the eight first-batch
  prompts. That puts the thinking arm at about 57 GPU-hours.
- **Length bucketing without padding.** The 4,530 prompts have 560 distinct lengths, so they would make 853 batches
  instead of 567, about 20 GPU-hours. It also leaves the same batch noise, because same-length batches deviate just as
  much as padded ones.

## Cost

- **The check:** about one extra minute per thinking job, for the float32 load and check plus the bf16 reload. Loading
  from node-local disk took about 3 seconds in the diagnosis job.
- **The run as planned:** 567 batches at about 85 seconds each (measured on the first batch, where 3 of 8 traces hit
  the 2,048-token cap) is about 13.4 GPU-hours of generation, plus about 0.5 for scoring. That is within the design's 9
  to 22.

## The freeze rule for this deviation (S12, narrowed)

Every round-3 job still refuses any checkout whose code differs from the freeze tag's commit, with one exemption.
Runs whose id starts `textcraft_r3_think` (the thinking smoke and the thinking arm) may start at the commit of the
annotated tag `textcraft-r3-deviation-t1-2026-11-23`, or at a commit code-identical to it. The coordinator creates that
tag after review. The exemption holds only under three conditions:

- **One tag.** There is exactly one such tag.
- **Narrow code change.** Relative to the freeze tag's commit, that commit changes code only in D-T1's files:
  `wf/textcraft/r3_think.py`, `wf/textcraft/r3_gates.py`, `wf/textcraft/r3_analysis.py` and
  `scripts/mutation_check.py`, plus the files of D-G1 (notes/textcraft_r3_deviation_gemma_e4b.md), which shares the tag. Tests and notes may change. Any other code change voids the exemption
  (`r3_gates.deviation_commit`).
- **Clean checkout.** The checkout is clean.

The scoring, smoke, environment-check, rescoring and format runs pass no run id to the check, so they still refuse
the deviation commit (tests `test_dt1_only_thinking_runs_start_at_the_deviation_commit` and
`test_dt1_a_core_run_at_the_deviation_commit_is_refused`).

The analysis accepts the thinking arm's shards at the freeze commit or at the deviation commit, and every other run's
shards only at the freeze commit (`r3_analysis.other_commit_problems`). It records the deviation commit, any problem
with the deviation tag, and the commits each thinking shard's jobs ran at (`metrics["deviation_t1"]`). Every
confirmatory run (the core and extension scoring, the environment check and the rescoring) continues to run from the
frozen checkout `text-wm-fix47`, at the freeze tag.


## The thinking smoke run at the deviation tag (2026-11-24, recorded before the full arm is submitted)

The smoke ran as job 7299373 (one L40S, 7 minutes) from checkout `text-wm-fix49` at `textcraft-r3-deviation-t1-2026-11-23`
(98dfbbf). It is read for its gates, format and timing only. No L value, arm difference or effect was read.

- **Freeze check.** The job started under the D-T1 exemption and recorded the deviation commit. In the same checkout,
  `deviation_commit()` finds the one deviation tag with no problems, and the core, rescoring and environment-check run
  ids are still refused there.
- **Start checks (float32).** Prompt problems: 0. The shortest prompt of the first batch was left-padded with 105
  tokens. Padded against alone over its 20 most probable tokens, the difference is 0.000018 nats on average and 0.000067
  at most, against the limits of 0.01 and 0.05. **Passed.**
- **Generation.** 16 traces in 2 batches, with the pinned revision. 10 of the 16 were closed at the 2,048-token cap
  (`forced_reason` "cap"), and the median length is 2,048 new tokens.
- **Scoring.** 16 scores, none missing or non-finite, no boundary problems.
- **Timing.** About 88 seconds per batch of 8, including the bf16 load. The full arm's 567 batches therefore need about
  13.9 hours of generation, plus about 1 hour of scoring: about 7.5 hours per shard across two shards, within the
  design's 9 to 22 GPU-hours.

Format note for the results: on this sample, most traces (10 of 16) reach the cap and are budget-forced. The thinking
block reports every readout on all traces and on the unforced ones (design 14.S10). If the full arm shows a similar
share, the unforced subset will be small and selected.
