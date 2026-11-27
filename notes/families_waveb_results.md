# Model families, wave B: results (final, 2026-11-26)

This is the final reading of wave B. It replaces the interim reading of 2026-11-25 (`families_waveb_results_interim.md`), which was made at the user's request while both Gemma 4 models were pending. **No verdict changed between the two readings** (section 7).

## 1. What was run

Wave B asks two questions:

- whether the goal effect found earlier holds in newer model families;
- whether it differs between a base model and its instruction-tuned version.

**An item and its three tasks.** An item is an action that fails in an ALFWorld game. The model sees the game so far and is asked a yes/no question: does the action succeed? It is asked under three versions of the task line:

- the **aligned** task: the item's real task;
- the **swapped** task: the same task with another object in it;
- the **neutral** task: no task line.

**The scores.** For each item, the yes/no score is the model's log-odds of "Yes" against "No". A second format, the two-candidate format, scores the two possible next observations against each other: the observation if the action succeeds against "Nothing happens."

**The primary metric.** The item share toward Yes is the share of the 4,666 failing actions whose yes/no score is higher under the aligned task than under the swapped one.

- A share near 0.5 means the task line makes no systematic difference.
- A share near 1 means that on nearly every item the model leans more toward "the action works" when the action serves its own task.

The goal effect in nats is the mean of the same aligned-minus-swapped difference.

**Models scored in the newer software stack** (serve2: transformers 5.23, torch 2.14):

- Ministral 3 8B Instruct;
- Qwen3.5 9B Instruct, scored twice: in its chat template, and in the plain-text format used for the base-against-instruct comparison;
- Qwen3.5 9B Base, in plain text;
- Gemma 4 E4B Instruct;
- Qwen3.8 27B;
- Gemma 4 31B Instruct.

**Older comparison models,** rescored in the same stack under the section 8 amendment: Qwen3-8B, Llama-3.1-8B, Qwen3-32B, Gemma 2 9B and Mistral 7B v0.3.

**The analysis.** `families_waveb_analysis` was run unchanged:

- job: CPU job 7370332;
- checkout: `text-wm-fix53`, at the deviation tag `families-waveb-deviation-attn-float32-2026-11-25` (commit 0ef4672), clean tree;
- inputs: 24 runs, each recorded in the provenance block;
- metrics: copied to `notes/families_waveb_metrics_2026-11-26.json`.

**The deviation, D-B1.** Both Gemma 4 models were scored under deviation D-B1 (design note section 20, decided before any wave B score was read). It moves the attention-agreement start check, eager against SDPA attention, from bf16 to float32. Scoring itself is unchanged: bf16, SDPA.

## 2. Gates

**The environment check: failed, and does not gate.** It is reported as failed, unchanged. Since the section 8 amendment, wave B is gated by the rescorings instead.

**The serve2 rescorings: passed.** All five older models passed every check:

- completeness;
- coverage: 29,164 of 29,164 scores per model, item hashes matching;
- identical token ids between the study stack and serve2 on all 39,082 prompt, candidate and answer token lists, compared at the rescoring commit from a clean tree;
- the pinned GPU type: Mistral 7B on an A40; Qwen3-8B, Gemma 2 and Llama-3.1-8B on an L40S; Qwen3-32B on an A100 80GB;
- the loading check at the pinned revision;
- Qwen3-32B's weight hashes;
- the analysis's references being exactly the rescorings.

**The wave B models' own gates: passed.** For every wave B run, the tokenization and template gate passed with 0 problems in transformers 5.23.0, the loading check passed, and the node-local weights matched their manifests.

**The Gemma 4 attention check, decided in float32 under D-B1: passed** for all three Gemma 4 runs. The frozen bf16 comparison, recorded beside it, fails for the two scoring runs, as it did before D-B1.

| Run | Job, GPU | Float32 yes/no mean / max | Float32 two-candidate mean / max | Bf16 yes/no mean (frozen rule) |
|---|---|---|---|---|
| Gemma 4 E4B scoring | 7329972, L40S | 0.000059 / 0.000078 | 0.000045 / 0.000064 | 0.275 (fails) |
| Gemma 4 E4B placement check | 7329979, L40S | 0.000012 / 0.000019 | 0.000019 / 0.000069 | 0.050 (passes) |
| Gemma 4 31B scoring | 7350401, two A100 80GB | 0.000022 / 0.000038 | 0.000114 / 0.000285 | 0.159 (fails) |

The float32 limits are 0.01 nats mean and 0.05 nats for any single score. The float32 differences are three to four orders of magnitude below them. This agrees with the bf16 failures being rounding, not a wrong implementation.

Gemma 4 31B was split over two cards of one node, because its float32 copy needs about 125 GB. The other runs above used one card each.

**The competence gate: every model and reference is interpretable.** These are measured on the yes/no format under the neutral task.

| Model | AUC | Yes to failures | Yes to determined successes | Median P(Yes) + P(No) |
|---|---|---|---|---|
| Ministral 3 8B | 0.976 | 0.022 | 0.743 | 0.94 |
| Qwen3.5 9B (chat) | 0.996 | 0.000 | 0.257 | 1.00 |
| Qwen3.5 9B (plain) | 0.994 | 0.001 | 0.586 | 0.52 |
| Qwen3.5 9B Base (plain) | 0.989 | 0.001 | 0.637 | 0.79 |
| Gemma 4 E4B | 0.967 | 0.005 | 0.592 | 1.00 |
| Qwen3.8 27B | 0.999 | 0.001 | 0.846 | 0.69 |
| Gemma 4 31B | 1.000 | 0.001 | 0.880 | 1.00 |

**The environment-dependence rule: no reference is environment-dependent.** Between the study stack and serve2, every reference's item share moved by at most 0.008, and its yes/no goal effect by at most 0.016 nats. So no comparison carries a second reading.

## 3. The primary family: 16 tests, Holm-corrected

"Supported" means the Holm-adjusted p is below 0.05. A share test also needs the share to be above one half and the model to pass the competence gate. Estimates are game-clustered. The interval is the 95% interval for shares and "differs" tests, and the 90% interval for "matches" (equivalence within ±0.05) tests.

| Test | Estimate | Interval | p | Holm p | Verdict |
|---|---|---|---|---|---|
| Ministral 3 8B: item share | 0.791 | [0.757, 0.822] | 0.001 | 0.014 | supported |
| Qwen3.5 9B: item share | 0.895 | [0.873, 0.915] | 0.001 | 0.014 | supported |
| Qwen3.5 9B Base (plain): item share | 0.900 | [0.877, 0.919] | 0.001 | 0.014 | supported |
| Gemma 4 E4B: item share | 0.706 | [0.670, 0.740] | 0.001 | 0.014 | supported |
| Qwen3.8 27B: item share | 0.652 | [0.619, 0.685] | 0.001 | 0.014 | supported |
| Gemma 4 31B: item share | 0.730 | [0.697, 0.761] | 0.001 | 0.014 | supported |
| Ministral 3 minus Mistral 7B: differs | −0.088 | [−0.130, −0.047] | 0.001 | 0.014 | supported |
| Ministral 3 minus Mistral 7B: matches | −0.088 | [−0.122, −0.054] | 0.965 | 1 | not supported |
| Qwen3.5 9B minus Qwen3-8B: differs | +0.010 | [−0.010, +0.032] | 0.349 | 1 | not supported |
| Qwen3.5 9B minus Qwen3-8B: matches | +0.010 | [−0.007, +0.029] | 0.0005 | 0.008 | supported |
| Qwen3.8 27B minus Qwen3-32B: differs | −0.055 | [−0.095, −0.015] | 0.011 | 0.077 | not supported |
| Qwen3.8 27B minus Qwen3-32B: matches | −0.055 | [−0.089, −0.022] | 0.606 | 1 | not supported |
| Qwen3.5 9B instruct minus base (plain): differs | +0.006 | [−0.012, +0.025] | 0.551 | 1 | not supported |
| Qwen3.5 9B instruct minus base (plain): matches | +0.006 | [−0.009, +0.022] | 0.0005 | 0.008 | supported |
| Gemma 4 E4B minus Gemma 2 9B: differs | −0.038 | [−0.098, +0.022] | 0.181 | 1 | not supported |
| Gemma 4 E4B minus Gemma 2 9B: matches | −0.038 | [−0.085, +0.013] | 0.337 | 1 | not supported |

**The readings of the planned pairs.**

| Pair | Reading |
|---|---|
| Ministral 3 against Mistral 7B | **differs**: the newer model moves 8.8 points fewer failures toward Yes |
| Qwen3.5 9B against Qwen3-8B | **matches** |
| Qwen3.8 27B against Qwen3-32B | **inconclusive (no difference detected)**: the unadjusted 95% interval excludes zero, but the Holm p is 0.077 |
| Gemma 4 E4B against Gemma 2 9B | **inconclusive (no difference detected)** |
| Qwen3.5 9B instruct against base (plain text) | **matches** |

## 4. The logged predictions

**B1: the goal effect is positive in both formats, and the share is supported, for every model that passes the competence gate.**

- It holds for Ministral 3, Qwen3.5 9B in chat and plain format, Qwen3.5 9B Base, Gemma 4 E4B and Gemma 4 31B.
- **It does not hold for Qwen3.8 27B.** Its share is supported (0.652), and its yes/no goal effect is positive: 0.41 nats [0.34, 0.48]. Its two-candidate goal effect is not: −0.02 nats [−0.08, +0.03].

**B2: no generation pair reads "differs" with the newer model lower.**

- **It fails for Ministral 3 against Mistral 7B**, which differs at −0.088.
- It holds for the other three pairs. The Gemma pair and the large Qwen pair read inconclusive, which counts as holding; Qwen3.5 against Qwen3-8B matches.

Generation pairs also change model size, so a "differs" here describes two whole models, not a generation effect.

**B3: in plain text, the instruct model's share is higher than the base model's.** **It does not hold**: the two shares match, at +0.006, inside the margin. The fallback rule does not apply, because the base model passed every competence criterion.

## 5. Descriptive readings

### Effect sizes in each format

The goal effect is aligned minus swapped. The standardised effect d divides it by the standard deviation of the swapped-task scores. The 95% interval for d is in brackets.

| Model | Yes/no goal effect (nats) | Yes/no d | Two-candidate goal effect (nats) | Two-candidate d |
|---|---|---|---|---|
| Ministral 3 8B | 1.15 | 0.54 [0.46, 0.61] | 1.20 | 0.42 [0.36, 0.48] |
| Qwen3.5 9B (chat) | 1.38 | 0.92 [0.82, 1.02] | 0.87 | 0.50 [0.45, 0.55] |
| Qwen3.5 9B (plain) | 0.64 | 0.98 [0.88, 1.09] | 0.39 | 0.26 [0.24, 0.28] |
| Qwen3.5 9B Base (plain) | 0.36 | 0.95 [0.86, 1.06] | 0.12 | 0.09 [0.07, 0.10] |
| Gemma 4 E4B | 1.16 | 0.59 [0.52, 0.67] | 1.26 | 0.33 [0.28, 0.38] |
| Qwen3.8 27B | 0.41 | 0.25 [0.20, 0.29] | −0.02 | −0.01 [−0.05, +0.02] |
| Gemma 4 31B | 0.32 | 0.20 [0.14, 0.29] | 0.81 | 0.14 [0.09, 0.18] |
| Qwen3-8B (reference) | 2.82 | 1.00 [0.91, 1.11] | 6.11 | 0.78 [0.70, 0.87] |
| Llama-3.1-8B (reference) | 2.21 | 0.93 [0.85, 1.03] | 1.65 | 0.53 [0.47, 0.58] |
| Qwen3-32B (reference) | 1.54 | 0.45 [0.33, 0.56] | 2.37 | 0.59 [0.52, 0.67] |
| Gemma 2 9B (reference) | 1.22 | 0.61 [0.47, 0.77] | 0.81 | 0.20 [0.12, 0.28] |
| Mistral 7B (reference) | 2.67 | 0.96 [0.87, 1.07] | 1.06 | 0.28 [0.24, 0.32] |

Across the scored models, the share stays between 0.65 and 0.91. The standardised effect is smallest in the two largest new models: Qwen3.8 27B, and Gemma 4 31B at 0.20 yes/no. In Gemma 4 31B the effect is small in nats too (0.32 yes/no), but the share is still 0.73. So most items still lean toward Yes under the aligned task, by small amounts.

The ratio R of the goal effect to the evidence effect is the yes/no goal effect relative to how far the model separates determined successes from failures. It is 0.008 for Gemma 4 31B, against 0.09 to 0.43 for the references. The goal effect there is tiny next to the model's discrimination between working and failing actions.

### Base against instruct, in plain text

The two Qwen3.5 9B models are almost the same on this measure:

- shares: 0.906 for instruct, 0.900 for base;
- standardised yes/no effects: 0.98 and 0.95.

Only in nats is the instruct model's effect larger: by 0.28 on yes/no [0.25, 0.31], and by 0.27 on two-candidate. Read through the share, the goal effect is present in the base model as fully as in the instruct model.

### The format change within one instruct model

Qwen3.5 9B in its chat template against the plain format: the share matches (−0.011 [−0.028, +0.008]). The yes/no effect is 0.74 nats larger in the chat template, while the standardised effects are close (0.92 and 0.98).

### Size within the Gemma 4 line

Gemma 4 31B against E4B: the share difference is +0.025 [−0.018, +0.070], so no difference is detected. In nats the yes/no effect is 0.84 smaller in 31B [0.71, 0.99].

### The format checks, read by the one-fifth rule

A check is format-dependent when its difference's interval excludes zero and the difference exceeds a fifth of the native effect.

**Gemma 4 E4B, with the system text in its first user turn** (300 items, the placement check). **It is not format-dependent in either format.**

- Yes/no moves by +0.01 [−0.06, +0.07], against a native effect of 1.09.
- Two-candidate moves by −0.20 [−0.33, −0.07], against 1.12. The interval excludes zero, but the move is below one fifth of the native effect (0.22).

**Qwen3.8 27B with preserve_thinking=False** (300 items). Qwen3.8's template by default writes an empty think block into each example reply; this variant writes none. **It is format-dependent in both formats.**

- Yes/no moves by −0.23 [−0.33, −0.13], against 0.41.
- Two-candidate moves by +0.57 [+0.47, +0.67], against 0.06.

So the large Qwen pair is reported with both numbers. On these items, the native two-candidate effect near zero becomes clearly positive without the empty think blocks.

### Against the two descriptive references

Each entry is the item share of the new model minus that of the reference, with its 95% interval.

| Model | Minus Qwen3-8B | Minus Llama-3.1-8B |
|---|---|---|
| Ministral 3 8B | −0.093 [−0.124, −0.063] | −0.105 [−0.135, −0.077] |
| Qwen3.5 9B | +0.010, matches | −0.002, matches |
| Qwen3.5 9B (plain) | +0.021, matches | +0.009, matches |
| Qwen3.5 9B Base (plain) | +0.015, matches | +0.003, matches |
| Gemma 4 E4B | −0.179 [−0.222, −0.141] | −0.191 [−0.226, −0.157] |
| Qwen3.8 27B | −0.233 [−0.264, −0.201] | −0.245 [−0.275, −0.214] |
| Gemma 4 31B | −0.155 [−0.187, −0.123] | −0.166 [−0.198, −0.136] |

### The original-environment column (descriptive)

Every comparison repeated with the references' original-environment scores gives the same reading. For example:

| Comparison | Serve2 scores | Original-environment scores |
|---|---|---|
| Ministral 3 against Mistral 7B | −0.088 | −0.087 |
| Qwen3.8 27B against Qwen3-32B | −0.055 | −0.062 |
| Gemma 4 E4B against Gemma 2 9B | −0.038 | −0.042 |

## 6. Numerical fragility

These readings are checked against `notes/numerical_sensitivity.md` (on `main`, rule 3). A reading is "numerically fragile" when its interval ends within about twice the numerical standard deviation of its boundary. That standard deviation is about:

- 0.1 nats for means of Qwen-family models;
- 0.03 to 0.05 nats for other families;
- 0.008 for item shares.

**Two readings are fragile.**

- **Qwen3.8 27B's two-candidate goal effect** (−0.02 nats [−0.08, +0.03]). This reading makes B1 fail for that model. The interval lies within the Qwen floor of zero on both sides, and the template check shows the effect also depends strongly on the format.
- **Gemma 4 E4B's placement check, two-candidate** ("not format-dependent"). The move (−0.20) is 0.02 nats inside its one-fifth boundary (0.22), which is within the Gemma family's floor. Another environment could plausibly make it "format-dependent" in that format. Its yes/no reading is not fragile.

**These readings are not fragile:**

- Every share-based verdict in the family, because shares move by at most 0.008. The nearest share boundaries are:
  - 0.022 for the Gemma pair, whose 95% upper end of +0.022 would have to fall below zero to read "differs";
  - 0.021 for Qwen3.5 against Qwen3-8B, whose 90% upper end of 0.029 sits against the 0.05 margin;
  - 0.047 for Ministral's "differs".
- The Holm p of 0.077 for the large Qwen pair, which is far from 0.05.
- Gemma 4 31B's small but positive effects. Their intervals end 0.26 nats (yes/no) and 0.53 nats (two-candidate) above zero.

## 7. Changes from the interim reading

**No verdict, reading or prediction outcome changed** from the interim reading of 2026-11-25. The only change to an earlier number is the Holm-adjusted p of the Qwen3.8 27B against Qwen3-32B "differs" test, which moved from 0.099 to 0.077 once the four Gemma 4 tests had real p-values. It is still not supported.

**What the final run added:**

- Gemma 4 E4B's and 31B's share tests (both supported), competence (both interpretable) and B1 (holds for both);
- the Gemma pair (inconclusive), which completes B2;
- E4B's placement check (not format-dependent);
- the Gemma descriptive comparisons;
- the final Holm adjustment over all 16 tests.
