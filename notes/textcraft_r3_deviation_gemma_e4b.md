# TextCraft round 3, deviation D-G1: an exploratory Gemma 4 E4B run, if eager and SDPA agree in float32

Date: 2026-11-23, after the freeze tag `textcraft-r3-design-frozen-2026-11-23` (0d6f460). Decided by the user.

## The confirmatory reading does not change

Both Gemma 4 models failed the attention-agreement gate at their smoke runs:

- **Gemma 4 12B:** yes/no mean 0.216 nats against the 0.10 limit.
- **Gemma 4 E4B** (job 7252166): yes/no mean 0.125 against 0.10, one item 0.375 apart; two-candidate mean 0.015.

Under the frozen rule (design 14.M2 and 14.M3), the Gemma core slot is therefore "not scored". Its tests enter family
A3 at p = 1, and the B3 pair Gemma 2 against Gemma 4 reads "not scored". This follows from the filed gate records, and
nothing in that path changes.

## The pre-stated criterion (written and committed before the diagnostic job runs)

Diagnostic job: Gemma 4 E4B on an L40S, from the separate checkout `text-wm-fix48` at the freeze tag. It scores the
smoke run's five attention-agreement items (the first five main failures of the smoke records, aligned two-candidate
and yes/no arms) under eager and under SDPA attention, once in bf16 (as the gate does) and once in float32. It reports:

- the mean and the maximum absolute eager-against-SDPA difference on each scale;
- the raw yes/no readout logits and the bf16 spacing at their magnitude.

**Criterion:** eager and SDPA agree in float32 if, on the yes/no scale, the mean absolute difference is at most 0.01
nats and no item differs by more than 0.05 nats.

- **If they agree,** the bf16 disagreement is attributed to rounding, and E4B gets the exploratory run below.
- **If they do not agree,** E4B is not scored at all. The diagnostic is recorded and nothing else is added.

## The exploratory run (only if the criterion is met)

`textcraft_r3x_gemma4-e4b-it` scores the same items, arms, prompts and scorer as a core model: SDPA, bf16, one L40S,
two shards, node-local weights by route A. The analysis reads it in a separate exploratory block that contains:

- each of the five contrasts' estimate and interval;
- the competence readout;
- the standardized effects.

The block is printed beside the confirmatory "not scored". It has no p-values. It never enters A3, A3x or B3, never
counts in H3-P, and never reaches the replication reading or R2.

## The freeze rule for this deviation

The S12 exemption of D-T1 is extended, on the same terms, to runs whose id starts `textcraft_r3x_`. Such a run may
start at the commit of the annotated tag `textcraft-r3-deviation-t1-2026-11-23`, or at a code-identical commit. Every
other run still refuses anything but the freeze tag's commit.

## The diagnostic's result (job 7252801, one L40S, 4 minutes, `text-wm-fix48` at the freeze tag)

Files: `notes/textcraft_r3_diag/e4bdiag_7252801.json` (full output), `e4b_attn_diag.py` and `e4b_attn_diag.sbatch`.
The numbers that decide the run are committed in `wf/config/textcraft_r3x_e4b_attention.json`, with the sha256 of the
full output.

| Precision | yes/no mean | yes/no max | two-candidate mean | two-candidate max |
|---|---|---|---|---|
| bf16 (as the gate runs it) | 0.125 | 0.375 | 0.015 | 0.024 |
| float32 | 0.000011 | 0.000027 | 0.000020 | 0.000061 |

The bf16 row reproduces the smoke gate exactly. Two points from the job explain it:

- **The yes/no differences are whole bf16 steps.** Per item, eager minus SDPA is 0, −0.125, −0.375, +0.125 and 0
  nats. The yes/no readout logits are bf16 numbers between 22.5 and 28.9: "No" at about 28.3 to 28.9 and "Yes" at
  about 22.5 to 25.5. In that range bf16 has a spacing of exactly 0.125. Each yes/no score is the difference of two
  such logits, so it can only move in steps of 0.125, and the 0.125 mean is one step on average. The 0.375 item is
  three steps.
- **The attention paths agree.** In float32 the two paths agree to within 0.00003 nats on yes/no and 0.00006 on the
  two-candidate score. On the same items, the float32 logits sit at most about 0.25 from their bf16 values.

**The criterion is met** (yes/no 0.000011 ≤ 0.01 on average and 0.000027 ≤ 0.05 at most), so the exploratory run is
added. The confirmatory reading stays "not scored", as above.

Gemma 4 12B's smoke failure (yes/no 0.216) is plausibly the same rounding, but it was not diagnosed. The user's
decision covers E4B only, so 12B stays "not scored" with no exploratory run.

## Implementation

- **The run module.** `wf/textcraft/r3x.py` holds `textcraft_r3x_gemma4-e4b-it`, registered in `wf/runs.py` with its
  config `wf/config/textcraft_r3x_gemma4-e4b-it.yaml` (two shards). The run refuses to start unless the committed
  float32 record meets the criterion, recomputed from its numbers (`r3x.criterion_problems`).
- **How it scores.** It uses the core scoring loop `r3_score.score_shard` with E4B's core spec, changing only
  `attn_agreement_check=False`. Everything else is as for a core model: SDPA, bf16, the round-3 tokenization gate, the
  golden renderings, the GPU and weights checks.
- **Its records.** Its gate records sit under its own run folder, so `r3_gates.model_state` never sees them and the
  confirmatory state of E4B cannot change.
- **The one change to the core loop.** `score_shard` now passes its run id to `require_freeze_commit`. No core,
  environment-check or rescoring run id starts with `textcraft_r3_think` or `textcraft_r3x_`, so their check is
  unchanged. They still refuse the deviation commit, which the tests show at job start.
- **The job script.** `pace/textcraft_r3x.sbatch` is a copy of the core job script that accepts only `textcraft_r3x_`
  runs. The core script is unchanged.
- **The analysis block.** `r3_analysis.r3x_block` builds the block apart from every family, from the five contrasts'
  estimates and intervals, the competence readout and the standardized effects. `strip_tests` removes every p-value,
  Holm step, equivalence test and power figure from it. The block is stored under
  `metrics["exploratory"]["r3x_gemma4-e4b-it"]` with the confirmatory state beside it.
- **Provenance.** The exploratory shards are commit-checked like the thinking arm's: accepted at the freeze commit or
  the deviation commit, with each shard's commits recorded. Their model revision is checked against the pin.
- **Test.** `test_dg1_the_exploratory_block_cannot_enter_any_family` shows that A3, A3x and B3 are byte-identical
  with and without the exploratory scores. E4B's A3 verdicts and its B3 pair stay "not scored", and the block holds no
  p-value.

## Cost

Two shards on one L40S each, about 2 GPU-hours. This is an estimate scaled from Gemma 4 12B's 2.5 to 3.5: E4B's smoke
stopped at its gate, so no speed was measured. With D-T1's thinking arm at about 14 GPU-hours, the study stays within
the 66 GPU-hour budget of 14.C2.
