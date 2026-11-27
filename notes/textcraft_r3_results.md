# TextCraft round 3: results (final, 2026-11-27)

This is the final reading of round 3. It replaces the INTERIM reading of 2026-11-25, and section 8 lists every
verdict that changed. The analysis run `textcraft_r3_analysis_dg2` ran once, after every arm was filed. It ran from a
clean checkout (`text-wm-fix56`) at the final tag `textcraft-r3-deviation-g2a-2026-11-26` (commit `f0ac203`) as ICE
CPU job 7374476. Its run folder records that commit as `git_sha`, and `git_dirty` is false.

- **Design:** `notes/textcraft_r3_design.md`, frozen at tag `textcraft-r3-design-frozen-2026-11-23`.
- **Deviations:** D-T1, D-G1, D-T2, D-G2 and D-G2a, each with a dated note. They are listed in section 15 of the
  design and in `notes/deviations.md`.
- **Where the notes live:** the design, the deviation notes and the Ministral wording note are on branch
  `textcraft-r3-dev-g2`.
- **Metrics:** `notes/textcraft_r3_metrics.json`. The interim metrics are kept as `notes/textcraft_r3_interim_metrics.json`.

## 1. The question and the measures

Round 3 asks whether round 2's measured quantities behave the same way in recent open-weight instruction models when
everything else is held fixed: the same 2,982 records, the same goal interventions, the same clean world-model
prompt, the same scoring and the same estimands. It has three parts.

- **Direct replication.** Two families of 30 tests each: three models, times five contrasts, times two tests per
  contrast ("does it differ from zero" and "is it equivalent to zero"). Each family is corrected with Holm's method.
  - The core family A3 holds Qwen3.5-9B, Ministral-3-8B and the Gemma slot.
  - The extension family A3x holds Qwen3.8-27B, Gemma 4 31B and OLMo 3 7B.
- **Older against newer releases (family B3).** Three pairs, each a round-2 model and a contemporary model of the same
  developer line, compared on a standardized scale (a mean divided by the item standard deviation) with an
  equivalence margin of 0.20.
- **Exploratory readouts.** These have no test: the thinking arm, the exploratory Gemma 4 E4B run, the format checks
  and the size and recency comparisons.

**The score.** The model reads a crafting history and an action and is asked what happens next. Its score $L$ is the
log-probability of the success reply minus that of the failure reply. Both replies are read after the same prompt.
The intervention is the goal line of the prompt. Every contrast is a mean over failures, with goal pairs as the
resampling clusters.

**The intermediate lift Λ** (990 failed ingredient crafts) measures how much more optimistic the model is when the
goal line names the goal the ingredient serves (aligned) than when it names an unrelated goal (swapped):

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}.$$

**The within-item penalty P** (675 failed crafts) compares the same craft with the item named as the goal against the
item named as an ingredient of a parent goal:

$$P = \overline{L(\text{aligned}) - L(\text{parent})}.$$

A negative P means the model is less optimistic when the craft would complete the stated goal. Round 2 found P below
zero in all three of its competent models.

**Naming (Dn) and completion (Dc)** split P using a two-item goal line ("both") that names the item without completing
the goal. Each compares goal-role failures with class-matched ingredient-role failures:

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}},$$

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}}.$$

**Added empty steps (T)** measures how the goal effect changes when eight empty inventory look-ups precede the action
instead of one, for goal-role against ingredient-role failures.

**Verdicts.** Each contrast gets one of these:

- **confirmed above or below zero:** the Holm-adjusted p of the "differs" test is below 0.05, and the 95% interval lies
  on that side;
- **equivalent to zero:** the Holm-adjusted p of the equivalence test is below 0.05. The margin is the smaller of 0.3
  item SDs and 0.4 nats;
- **inconclusive:** neither of the above;
- **not applicable:** used for T only, when the one-look-up role gap has no interval below zero.

**Competence.** A verdict is read as a distortion only if the model is competent. That means it separates successes
from failures on the yes/no question (AUC at least 0.70), it does not answer Yes to nearly everything, and it puts at
least half its probability on Yes and No. Otherwise an effect is "measurable only".

**The prediction and its rules.** H3-P, the prediction carried from round 2, is that each core model's P is confirmed
below zero. The rules applied to it:

- **R1** fires for a competent model whose P is equivalent to zero or confirmed above zero.
- **R2**, restated for round 3, fires if no competent model has P confirmed below zero while at least one has P
  equivalent to zero or above zero. It then rejects the claim that the penalty is shared among these models.
- **The replication rule** compares each family's verdicts with round 2's finding for each contrast.

## 2. Gates and which models enter

**The environment check.** It rescored 300 round-2 records of the three older models in round 3's software, and it
failed for all three: Qwen3-8B, Mistral-7B and Gemma-2-9B each failed the yes/no per-score criterion, and Qwen3-8B also
failed the two-candidate one. Every goal-effect check passed. So B3, as frozen, compares each older model's rescoring
of every record in round 3's software (jobs 7252995–7253000) with the newer model.

**The attention-agreement check (D-G2, D-G2a).** This check scores five items with two equivalent attention
implementations, eager and SDPA, and stops a Gemma 4 job if they disagree. The frozen design ran it in bf16. There,
rounding alone made the implementations disagree beyond the limits for Gemma 4 12B, E4B, and every scoring shard of
31B: up to 0.81 nats on average and 1.73 on one score. D-G2 evaluates the same criterion in float32.

| Model | Yes/no mean | Two-candidate mean | Largest single difference |
|---|---|---|---|
| 12B | 0.00009 | 0.00013 | 0.0003 |
| E4B | 0.00002 | 0.00002 | 0.00004 |
| 31B | 0.00004 | 0.00005 | 0.00009 |

All three pass. The frozen fallback rule then gave the Gemma slot to **Gemma 4 12B**. Its smoke rerun passed every
gate and showed no format mismatch: median yes/no mass 1.000, and no structural top tokens. Gemma 4 31B was scored at
D-G2a's commit. Its four failed bf16 records from the freeze-tag checkout stay filed, marked superseded, and are not
read. Every model then passed every gate: loading, the tokenization gate, the golden renderings, the weights manifest,
the pinned GPU type and the freeze-commit rule. The analysis found no provenance problem.

| Model | Family | Competent overall | Yes/no AUC | Yes to failures / successes | Equivalence margin |
|---|---|---|---|---|---|
| Qwen3.5-9B | A3 | yes | 0.905 | 4.4% / 56.5% | 0.175 |
| Ministral-3-8B | A3 | yes | 0.864 | 10.2% / 65.7% | 0.40 |
| Gemma 4 12B | A3 (the Gemma slot) | yes | 0.943 | 5.4% / 83.5% | 0.40 |
| Qwen3.8-27B | A3x | yes | 0.999 | 0.7% / 96.2% | 0.115 |
| Gemma 4 31B | A3x | yes | 0.997 | 1.9% / 97.5% | 0.40 |
| OLMo 3 7B | A3x | **no** (Yes to 99.8% of failures) | 0.560 | 99.8% / 100% | 0.40 |

## 3. Direct replication: per-model verdicts

Estimates are in nats with 95% goal-pair bootstrap intervals. "Holm p" is the adjusted p of the test that decides the
verdict. The minimum reachable with 20,000 resamples is 0.0015.

### Core family (A3)

| Model | Contrast | Estimate [95% CI] | Verdict | Holm p |
|---|---|---|---|---|
| Qwen3.5-9B | Λ | +0.074 [+0.034, +0.114] | confirmed above zero | 0.016 |
| | P | −0.020 [−0.085, +0.050] | equivalent to zero | 0.0015 (equiv.) |
| | Dn | +0.053 [−0.039, +0.142] | inconclusive | 0.086 (equiv.) |
| | Dc | +0.125 [+0.012, +0.242] | inconclusive | 0.44 |
| | T | −0.086 [−0.161, −0.011] | not applicable | — |
| Ministral-3-8B | Λ | +0.294 [+0.206, +0.385] | confirmed above zero | 0.0015 |
| | P | −1.231 [−1.441, −1.027] | confirmed below zero | 0.0015 |
| | Dn | −0.307 [−0.529, −0.087] | inconclusive | 0.11 |
| | Dc | −0.014 [−0.261, +0.248] | equivalent to zero | 0.040 (equiv.) |
| | T | −0.075 [−0.270, +0.117] | equivalent to zero | 0.013 (equiv.) |
| Gemma 4 12B | Λ | +0.148 [−0.049, +0.344] | inconclusive | 0.13 (equiv.) |
| | P | +0.162 [−0.019, +0.367] | inconclusive | 0.21 (equiv.) |
| | Dn | −0.195 [−0.480, +0.074] | inconclusive | 1.0 |
| | Dc | +0.598 [+0.363, +0.852] | confirmed above zero | 0.0015 |
| | T | −0.430 [−0.849, −0.096] | not applicable | — |

The per-model readings are:

- **Qwen3.5-9B:** "lift present; no penalty larger than the margin". R1 fires.
- **Ministral-3-8B:** "lift present; penalty present, mechanism unresolved". Its penalty is carried mostly by the later
  tokens of the replies (first token −0.11, the rest −1.12 nats). Ministral also puts only about half its probability
  on the replies' first tokens (median 0.49, against 0.97 for Mistral-7B; `textcraft_r3_ministral_wording_note.md`).
- **Gemma 4 12B:** "inconclusive on the penalty". Completing a named goal raises its predicted success relative to
  naming it (Dc above zero).

**The core-family readings.**

- **H3-P: partial.** Ministral-3-8B supports it, Qwen3.5-9B rejects it (R1), and Gemma 4 12B is inconclusive.
  "Partial" is reported per model and never counted as support.
- **R2 does not fire**, because one competent core model, Ministral, has P confirmed below zero.
- **Replication of round 2's general penalty: partial.** Ministral confirms it below zero, Qwen3.5 is equivalent to
  zero, and Gemma 4 12B is inconclusive.
- **The other contrasts:** round 2 established no general finding for them, so they have no replication target. The
  per-model verdicts above are new evidence.

### Extension family (A3x)

| Model | Contrast | Estimate [95% CI] | Verdict | Holm p |
|---|---|---|---|---|
| Qwen3.8-27B | Λ | +0.032 [+0.007, +0.058] | equivalent to zero | 0.0015 (equiv.) |
| | P | +0.106 [+0.072, +0.139] | confirmed above zero | 0.0015 |
| | Dn | +0.033 [−0.022, +0.091] | inconclusive | 0.052 (equiv.) |
| | Dc | +0.092 [+0.045, +0.139] | confirmed above zero | 0.0015 |
| | T | −0.024 [−0.068, +0.021] | not applicable | — |
| Gemma 4 31B | Λ | −0.433 [−0.677, −0.197] | confirmed below zero | 0.011 |
| | P | +0.652 [+0.468, +0.830] | confirmed above zero | 0.0015 |
| | Dn | −0.358 [−0.742, −0.019] | inconclusive | 0.63 |
| | Dc | +0.939 [+0.587, +1.345] | confirmed above zero | 0.0015 |
| | T | −0.073 [−0.547, +0.429] | not applicable | — |
| OLMo 3 7B (not competent) | Λ | +0.456 [+0.307, +0.613] | confirmed above zero (measurable only) | 0.0015 |
| | P | +0.003 [−0.227, +0.246] | equivalent to zero (measurable only) | 0.025 (equiv.) |
| | Dn, Dc, T | — | inconclusive, inconclusive, not applicable | — |

The per-model readings are:

- **Qwen3.8-27B** and **Gemma 4 31B:** both read "completing a named goal raises predicted success (no penalty)", and
  R1 fires for both. For 31B the lift is negative.
- **OLMo 3 7B:** measurable effects only, because it is not competent.

**The extension-family readings.**

- **H3-P: rejected.** Both competent extension models reject it.
- **R2 fires.** The penalty is not shared among the extension models.
- **Replication of the penalty: not replicated.** Both competent models are confirmed above zero, the opposite sign to
  round 2.

## 4. Older against newer releases (B3)

Readings use each model's standardized contrast, with Holm-adjusted tests in B3's family of 30. Raw nats are
descriptive.

| Pair | Contrast | Standardized difference, newer minus older [95% CI] | Reading | Holm p |
|---|---|---|---|---|
| Qwen3-8B → Qwen3.5-9B | Λ | −0.016 [−0.137, +0.119] | similar within the margin | 0.020 (equiv.) |
| | P | +0.172 [+0.014, +0.319] | not distinguishable (equivalence underpowered) | 0.67 |
| | Dn, Dc | +0.075, +0.048 | not distinguishable | 1.0 |
| | T | −0.487 [−0.727, −0.258] | differs, newer lower (opposite signs) | 0.0015 |
| Mistral-7B → Ministral-3-8B | Λ | −0.017 [−0.120, +0.081] | similar within the margin | 0.008 (equiv.) |
| | P | −1.022 [−1.225, −0.834] | differs, newer lower (opposite signs) | 0.0015 |
| | Dn | −0.792 [−1.021, −0.569] | differs, newer lower (opposite signs) | 0.0015 |
| | Dc | −0.055 [−0.292, +0.171] | not distinguishable | 1.0 |
| | T | +0.353 [+0.144, +0.568] | differs, newer higher (smaller in size) | 0.017 |
| Gemma-2-9B → Gemma 4 12B | Λ | +0.305 [+0.200, +0.413] | differs, newer higher (opposite signs) | 0.0015 |
| | P | +0.930 [+0.755, +1.111] | differs, newer higher (opposite signs) | 0.0015 |
| | Dn | +0.369 [+0.160, +0.579] | differs, newer higher (smaller in size) | 0.020 |
| | Dc | +0.350 [+0.172, +0.542] | differs, newer higher (opposite signs) | 0.006 |
| | T | −0.076 [−0.279, +0.142] | not distinguishable | 1.0 |

**Qwen pair.** In raw nats the penalty shrinks from −0.99 to −0.02. On the standardized scale the change is from −0.21
to −0.04 SD, because Qwen3-8B's item values are about eight times as spread.

**Mistral pair.** Mistral-7B is not competent, as in round 2.

**Gemma pair.** Both models are competent. Gemma-2-9B's penalty of −0.84 SD (−1.29 nats) becomes +0.09 SD (+0.16 nats)
in Gemma 4 12B, and the lift changes sign from negative to positive.

## 5. Exploratory readouts (no test)

**Thinking arm (Qwen3.5-9B with reasoning on, D-T1).**

- **The traces.** 4,530 traces; 36.7% were closed at the 2,048-token cap. The median reasoning is 671 new tokens. 67.5%
  of traces name the goal and 99.96% name the crafted item.
- **Penalty after reasoning.** −0.42 nats [−0.74, −0.11], or −0.10 SD [−0.18, −0.03], against −0.02 without
  reasoning. The change with reasoning is −0.40 nats [−0.71, −0.10], but −0.07 SD [−0.20, +0.07] on the standardized
  scale.
- **Unforced traces only** (a selected subset): −0.18 nats [−0.57, +0.22].
- **Discrimination after reasoning:** yes/no AUC 0.965.

The conclusion depends on the scale.

**Exploratory Gemma 4 E4B (D-G1, outside every family).** It is competent (AUC 0.811). Its estimates in nats:

| Contrast | Estimate [95% CI] |
|---|---|
| Λ | +0.100 [+0.031, +0.164] |
| P | −0.296 [−0.450, −0.149] |
| Dn | −0.195 [−0.374, −0.029] |
| Dc | +0.299 [+0.159, +0.447] |
| T | −0.010 |

Under the slot rule it is not used: Gemma 4 12B holds the slot. Its block's "confirmatory state" field reads
"waiting" only because E4B is not among the family tags.

**Format checks** (300 seeded main failures, the aligned-minus-swapped effect, one-fifth rule):

- **Qwen3.8-27B with `preserve_thinking=False`.** On the two-candidate scale the variant moves the effect by −0.073
  nats [−0.121, −0.027], against a native effect of +0.067, so the effect is format-dependent there. On yes/no it is
  not format-dependent.
- **Gemma 4 12B's first-turn placement** is not format-dependent on either scale.

**Size and recency** (standardized differences, larger minus smaller):

- **Gemma 4 12B → 31B:** Λ −0.19 [−0.29, −0.09] and P +0.25 [+0.09, +0.43].
- **Qwen3.5-9B → Qwen3.8-27B:** P +0.37 [+0.22, +0.53].

Within both lines the larger model's penalty is more positive.

## 6. Numerically fragile readings

Under `notes/numerical_sensitivity.md` (Rule 3), a verdict is labelled numerically fragile when its deciding interval
ends within the model's allowance of its boundary. The boundary is zero for "confirmed" and ± the margin for
"equivalent".

The allowances are:

- about 0.1 nats for Qwen-family logits;
- 0.05 for Ministral and OLMo;
- for the Gemma 4 models, the values D-G2a fixed before any 31B score existed, measured from their bf16
  implementation disagreement: 31B allows 0.065 nats for Λ, 0.078 for P, 0.129 for Dn and Dc, and 0.22 for T; 12B
  allows about half of each.

The fragile verdicts are:

- **Qwen3.5-9B, Λ confirmed above zero.** The interval ends 0.034 above zero.
- **Qwen3.5-9B, P equivalent to zero.** The 90% interval ends 0.100 inside the 0.175 margin, and R1 rests on this.
- **Qwen3.8-27B, Λ equivalent to zero.** The 90% interval ends 0.061 inside the 0.115 margin.
- **Qwen3.8-27B, P confirmed above zero.** The interval ends 0.072 above zero. This feeds R1, H3-P and R2 in A3x.
- **Qwen3.8-27B, Dc confirmed above zero.** The interval ends 0.045 above zero.

No Ministral, OLMo, Gemma 4 or B3 verdict is within its allowance of its boundary. In particular, Gemma 4 31B's
penalty above zero is not fragile: its interval ends 0.47 above zero. So A3x's H3-P rejection does not depend on the
fragile Qwen3.8-27B readings alone.

## 7. What the results say

**The penalty no longer replicates.** Round 2's general finding, a within-item penalty in every competent model, does
not hold across these contemporary releases.

- One core model (Ministral-3-8B) shows it, with a penalty as large as round 2's.
- One (Qwen3.5-9B) shows none larger than its margin. This reading is numerically fragile.
- One (Gemma 4 12B) is inconclusive.
- Both competent larger models (Qwen3.8-27B and Gemma 4 31B) show the opposite: completing a named goal raises their
  predicted success.

**B3 separates the release lines.** Within the Gemma line, the penalty disappears from Gemma-2-9B to Gemma 4 12B. That
reading is outside the equivalence margin and not fragile. Within the Qwen line, the drop in nats is mostly a change of
scale. Within the Mistral line, a non-discriminating older model is compared with a discriminating newer one.

**What the results do not say.** As the design states, these are a small purposive sample, one model per developer
line. They license no claim about "model generations". A difference between two releases changes data, size,
tokenizer, template, post-training, architecture and software at once.

## 8. Changes from the interim reading (2026-11-25)

**No verdict reported in the interim reading changed.** The per-model verdicts of Qwen3.5-9B, Ministral-3-8B and OLMo
3, and the B3 readings of the Qwen and Mistral pairs, are the same. Their Holm-adjusted p-values fell slightly as the
padded tests were replaced, as Holm's monotonicity guarantees. For example, Ministral's naming went from 0.12 to
0.11, and Qwen3.5's completion from 0.56 to 0.44. None crossed 0.05.

Newly available:

- the verdicts of Gemma 4 12B, Qwen3.8-27B and Gemma 4 31B;
- B3's Gemma pair;
- both families' H3-P, R2 and replication readings (A3: partial, R2 does not fire, P partial; A3x: rejected, R2
  fires, P not replicated);
- the two format checks and the two size comparisons.

Since the interim reading, the Gemma slot has resolved to Gemma 4 12B, under the frozen rule with the float32 reading
of D-G2.
