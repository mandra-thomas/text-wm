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
