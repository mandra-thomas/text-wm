# Model families, wave B: INTERIM results (2026-11-25; both Gemma 4 models pending)

**INTERIM.** This reading was made at the user's request before every wave B model was scored. The request was recorded in `notes/deviations.md` before the analysis was run. The two Gemma 4 models (E4B and 31B) are not scored yet: their reruns under deviation D-B1 are queued. The frozen analysis therefore treats them as "waiting", and their four tests enter the 16-test family at p = 1, as the frozen design specifies.

No rule was changed for this reading. The final analysis is run once, after the Gemma 4 reruns finish, and replaces this one. Any verdict that changes between the two will be reported as such.

## 1. What was run

Wave B asks whether the goal effect found earlier holds in newer model families. It also compares one base model with its instruction-tuned version.

The goal effect is measured as follows. An item is an action that fails in an ALFWorld game. The model sees the game so far and is asked a yes/no question: does the action succeed? It is asked under three versions of the task line:

- the **aligned** task, the item's real task;
- the **swapped** task, the same task with another object in it;
- the **neutral** (absent) task, with no task line.

For each item the score is the model's log-odds of "Yes" against "No" (the yes/no format). A second format scores the two possible next observations, success against failure (the two-candidate format).

The **item share toward Yes** is the share of the 4,666 failing actions whose yes/no score is higher under the aligned task than under the swapped task. It is wave B's primary metric. A share near 0.5 means the task line makes no systematic difference; a share near 1 means that, on nearly every item, the model leans more toward "the action works" when the action serves its own task.

**Models scored in the new software stack** (serve2: transformers 5.23, torch 2.14):

- Ministral 3 8B Instruct;
- Qwen3.5 9B Instruct, scored twice: in its chat template, and in one plain-text format used for the base-against-instruct comparison;
- Qwen3.5 9B Base (plain text);
- Qwen3.8 27B.

**Older comparison models**, rescored in the same stack under the section 8 amendment: Qwen3-8B, Llama-3.1-8B, Qwen3-32B, Gemma 2 9B and Mistral 7B v0.3.

The analysis is `families_waveb_analysis`: CPU job 7328891, checkout `text-wm-fix53`, at the deviation tag `families-waveb-deviation-attn-float32-2026-11-25` (commit 0ef4672). Its metrics are copied to `notes/families_waveb_interim_metrics_2026-11-25.json`.

## 2. Gates

Every gate the frozen analysis applies was evaluated before any reading. The outcomes:

**The environment check: FAILED, and it does not gate.** It is reported as failed, and its verdict is unchanged. Since the section 8 amendment, wave B is gated by the rescorings below instead.

**The serve2 rescorings: PASSED.** All five older models passed every check of the rescoring gate:

- the shards are complete;
- coverage: 29,164 of 29,164 scores per model, every item's hash matching;
- token ids are identical between the study stack and serve2 on all 39,082 prompt, candidate and answer token lists, compared at the rescoring commit from a clean tree;
- each model ran on its pinned GPU type (Mistral 7B on an A40; Qwen3-8B, Gemma 2 and Llama-3.1-8B on an L40S; Qwen3-32B on an A100 80GB, both shards);
- each gate passed with the loading check on, at the pinned revision;
- Qwen3-32B's downloaded weights matched the manifest;
- the analysis's references are exactly the five rescorings.

**The wave B models' own gates: PASSED.** For every scored wave B run (the four model runs and Qwen3.8's preserve_thinking check), the tokenization and template gate passed with 0 problems in transformers 5.23.0. The loading check passed and the node-local weights matched their manifests.

**The Gemma 4 models: not scored yet.** Both failed the frozen bf16 attention-agreement check before scoring:

- Gemma 4 E4B: 0.275 nats on average on yes/no.
- Gemma 4 31B: 0.159 nats on average on yes/no.

They are rerun under deviation D-B1, which decides the check in float32.

**The competence gate: all interpretable.** Every scored model and every reference is interpretable. The values, on the yes/no format under the neutral task:

| Model | AUC | Yes to failures | Yes to determined successes | Median P(Yes) + P(No) |
|---|---|---|---|---|
| Ministral 3 8B | 0.976 | 0.022 | 0.743 | 0.94 |
| Qwen3.5 9B (chat) | 0.996 | 0.000 | 0.257 | 1.00 |
| Qwen3.5 9B (plain) | 0.994 | 0.001 | 0.586 | 0.52 |
| Qwen3.5 9B Base (plain) | 0.989 | 0.001 | 0.637 | 0.79 |
| Qwen3.8 27B | 0.999 | 0.001 | 0.846 | 0.69 |

**The environment-sensitivity rule.** Every reference's item share moved by at most 0.008 between the study stack and serve2, and its yes/no goal effect by at most 0.016 nats. No reference is environment-dependent, so no comparison carries a second reading.

## 3. The primary family (16 tests, Holm-corrected)

"Supported" means the Holm-adjusted p is below 0.05. A share test also needs the share to be above one half, and the model to pass the competence gate. The estimates are game-clustered: the CI is 95% for shares and "differs" tests, and 90% for the "matches" (equivalence) tests.

| Test | Estimate | Interval | p | Holm p | Verdict |
|---|---|---|---|---|---|
| Ministral 3 8B: item share | 0.791 | [0.757, 0.822] | 0.001 | 0.014 | supported |
| Qwen3.5 9B: item share | 0.895 | [0.873, 0.915] | 0.001 | 0.014 | supported |
| Qwen3.5 9B Base (plain): item share | 0.900 | [0.877, 0.919] | 0.001 | 0.014 | supported |
| Qwen3.8 27B: item share | 0.652 | [0.619, 0.685] | 0.001 | 0.014 | supported |
| Gemma 4 E4B: item share | — | — | — | 1 | waiting |
| Gemma 4 31B: item share | — | — | — | 1 | waiting |
| Ministral 3 minus Mistral 7B: differs | −0.088 | [−0.130, −0.047] | 0.001 | 0.014 | supported |
| Ministral 3 minus Mistral 7B: matches | −0.088 | [−0.122, −0.054] | 0.965 | 1 | not supported |
| Qwen3.5 9B minus Qwen3-8B: differs | +0.010 | [−0.010, +0.032] | 0.349 | 1 | not supported |
| Qwen3.5 9B minus Qwen3-8B: matches | +0.010 | [−0.007, +0.029] | 0.0005 | 0.008 | supported |
| Qwen3.8 27B minus Qwen3-32B: differs | −0.055 | [−0.095, −0.015] | 0.011 | 0.099 | not supported |
| Qwen3.8 27B minus Qwen3-32B: matches | −0.055 | [−0.089, −0.022] | 0.606 | 1 | not supported |
| Qwen3.5 9B instruct minus base (plain): differs | +0.006 | [−0.012, +0.025] | 0.551 | 1 | not supported |
| Qwen3.5 9B instruct minus base (plain): matches | +0.006 | [−0.009, +0.022] | 0.0005 | 0.008 | supported |
| Gemma 4 E4B minus Gemma 2 9B: differs | — | — | — | 1 | waiting |
| Gemma 4 E4B minus Gemma 2 9B: matches | — | — | — | 1 | waiting |

The readings of the planned pairs:

- **Ministral 3 against Mistral 7B: differs.** The newer model moves 8.8 percentage points fewer failures toward Yes.
- **Qwen3.5 9B against Qwen3-8B: matches**, within ±0.05.
- **Qwen3.8 27B against Qwen3-32B: inconclusive (no difference detected).** The unadjusted 95% interval excludes zero, but the Holm-adjusted p is 0.099.
- **Qwen3.5 9B instruct against base, in plain text: matches**, within ±0.05.
- **Gemma 4 E4B against Gemma 2 9B: waiting.**

The Holm adjustment counts the four waiting tests at p = 1. In the final run they will have real p-values. A small Gemma 4 p-value would rank ahead of other tests and raise their multipliers, and a large one would leave them as they are, so the adjusted p of every other test can move in either direction.

The margins here are wide, though. The supported tests have Holm p of 0.014 or below; with one more test ranked ahead, the smallest raw p (0.0005 to 0.001) would give at most about 0.016, still below 0.05. The not-supported ones are at 0.099 or above. A verdict that changes in the final run will be reported as such.

## 4. The logged predictions

**B1: the goal effect is positive in both formats, and the share is supported for every model that passes the competence gate.**

- It holds for Ministral 3, Qwen3.5 9B (chat and plain) and Qwen3.5 9B Base.
- It **does not hold for Qwen3.8 27B**. Its share is supported (0.652) and its yes/no goal effect is positive: 0.41 nats [0.34, 0.48]. But its two-candidate goal effect is not: −0.02 nats [−0.08, +0.03].

**B2: no generation pair reads "differs" with the newer model lower.**

- **It fails for Ministral 3 against Mistral 7B** (differs, −0.088).
- It holds for the two Qwen pairs.
- The Gemma pair is waiting.

Generation pairs also change model size, so a "differs" here describes two whole models, not a generation effect.

**B3: in plain text, the instruct model's share is higher than the base model's.** It **does not hold**: the instruct and base models match (+0.006, within the margin). The fallback rule does not apply, because the base model passed every competence criterion.

## 5. Descriptive readings

**Effect sizes in each format.** The goal effect (aligned minus swapped) and its standardised form, d (the goal effect over the SD of the swapped-task scores), by model and format:

| Model | Yes/no goal effect (nats) | Yes/no d | Two-candidate goal effect (nats) | Two-candidate d |
|---|---|---|---|---|
| Ministral 3 8B | 1.15 | 0.54 [0.46, 0.61] | 1.20 | 0.42 [0.36, 0.48] |
| Qwen3.5 9B (chat) | 1.38 | 0.92 [0.82, 1.02] | 0.87 | 0.50 [0.45, 0.55] |
| Qwen3.5 9B (plain) | 0.64 | 0.98 [0.88, 1.09] | 0.39 | 0.26 [0.24, 0.28] |
| Qwen3.5 9B Base (plain) | 0.36 | 0.95 [0.86, 1.06] | 0.12 | 0.09 [0.07, 0.10] |
| Qwen3.8 27B | 0.41 | 0.25 [0.20, 0.29] | −0.02 | −0.01 [−0.05, +0.02] |
| Qwen3-8B (reference, serve2) | 2.82 | 1.00 [0.91, 1.11] | 6.11 | 0.78 [0.70, 0.87] |
| Llama-3.1-8B (reference, serve2) | 2.21 | 0.93 [0.85, 1.03] | 1.65 | 0.53 [0.47, 0.58] |
| Qwen3-32B (reference, serve2) | 1.54 | 0.45 [0.33, 0.56] | 2.37 | 0.59 [0.52, 0.67] |
| Gemma 2 9B (reference, serve2) | 1.22 | 0.61 [0.47, 0.77] | 0.81 | 0.20 [0.12, 0.28] |
| Mistral 7B (reference, serve2) | 2.67 | 0.96 [0.87, 1.07] | 1.06 | 0.28 [0.24, 0.32] |

**Base against instruct, in plain text.** The two Qwen3.5 9B models move almost the same share of failures toward Yes (0.906 instruct, 0.900 base) and have nearly the same standardised yes/no effect (0.98 against 0.95). In nats the instruct model's effect is larger: +0.28 nats yes/no [0.25, 0.31] and +0.27 nats two-candidate. Read through the share, the goal effect is present in the base model as fully as in the instruct model.

**The format change within one instruct model.** Qwen3.5 9B in its chat template against the plain format: the share matches (−0.011 [−0.028, +0.008]). The yes/no goal effect is 0.74 nats larger in the chat template, while the standardised effects are close (0.92 against 0.98).

**Qwen3.8 27B with preserve_thinking=False** (300 items; the one-fifth rule). Qwen3.8's default template writes an empty think block into each example reply; this variant writes none. It moves the yes/no goal effect by −0.23 nats [−0.33, −0.13], against 0.41 in the native format, and the two-candidate one by +0.57 nats [+0.47, +0.67], against 0.06. **Both formats are format-dependent.** The Qwen3.8 27B against Qwen3-32B pair is therefore reported with both numbers. On these 300 items, the native two-candidate effect near zero becomes positive when the empty think blocks are removed.

**Against the two descriptive references** (item share, new model minus reference, 95% interval):

| Model | Minus Qwen3-8B | Minus Llama-3.1-8B |
|---|---|---|
| Ministral 3 | −0.093 [−0.124, −0.063] | −0.105 [−0.135, −0.077] |
| Qwen3.5 9B | +0.010, matches | −0.002, matches |
| Qwen3.5 9B (plain) | +0.021, matches | +0.009, matches |
| Qwen3.5 9B Base (plain) | +0.015, matches | +0.003, matches |
| Qwen3.8 27B | −0.233 [−0.264, −0.201] | −0.245 [−0.275, −0.214] |

**The original-environment column (descriptive).** Every comparison repeated with the references' original-environment scores gives the same reading. For example, the Ministral pair is −0.087 there against −0.088 in serve2, and the large Qwen pair −0.062 against −0.055.

## 6. Numerical fragility

These readings are checked against the rules of `notes/numerical_sensitivity.md` (on `main`). A reading is "numerically fragile" if its interval ends within about twice the numerical SD of its boundary. That SD is about 0.1 nats for Qwen-family means, 0.03 to 0.05 for the other families, and 0.008 for item shares.

- **Qwen3.8 27B's two-candidate goal effect** (−0.02 nats, [−0.08, +0.03]) is **numerically fragile.** Its interval lies within the Qwen-family floor of zero on both sides. Its reading, "not positive", is the one that makes B1 fail for this model. A rescore in another environment could move the lower end above zero, though probably not the estimate far from zero. The template check shows it also depends strongly on the format.
- **Not fragile:** every share-based verdict in the family, because item shares move by at most 0.008 across environments.
  - The nearest share boundaries are 0.047 (Ministral's "differs", upper end −0.047 against 0) and 0.021 (Qwen3.5 against Qwen3-8B's "matches", upper end 0.029 against 0.05). Both are several times 0.008.
  - The Qwen3.8 27B against Qwen3-32B Holm p of 0.099 is far from 0.05.
- **Not fragile:** the template check's verdicts. They exceed one fifth of the native effect by 0.15 nats (yes/no) and 0.56 nats (two-candidate).

## 7. What the final run adds

When the Gemma 4 reruns (deviation D-B1) finish, the frozen analysis is run once more, unchanged. It adds:

- Gemma 4 E4B's and Gemma 4 31B's item-share tests, competence and B1 readings;
- the Gemma 4 E4B against Gemma 2 9B pair (two tests, and B2 for the Gemma pair);
- E4B's placement check (one-fifth rule);
- the two descriptive comparisons with the references, and Gemma 4 31B against E4B (size within the Gemma 4 line);
- the final Holm adjustment over all 16 tests.

The final note replaces this one and lists any verdict that changed.
