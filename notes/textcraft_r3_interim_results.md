# TextCraft round 3: INTERIM results (2026-11-25)

**Superseded (2026-11-27) by the final reading, [textcraft_r3_results.md](textcraft_r3_results.md).** Kept unchanged below as the record of the interim reading. No verdict reported here changed in the final run; section 8 of the final note lists what was added.

**INTERIM.** This reading was made early, at the user's request, before every arm of round 3 was filed. It follows
the five conditions the user approved on 2026-11-24, recorded in `notes/deviations.md` ("TextCraft round 3, interim
reading", 2026-11-25):

- the inputs are complete for what is read;
- the protocol allows the reading;
- no rule is changed;
- the reading cannot influence the unfinished arms;
- the final run preserves the verdicts.

Models whose scores are not filed enter as the frozen design specifies. The final analysis is run once, when every arm
is filed. It replaces this note, and any verdict it changes is reported as changed.

- **Design:** `notes/textcraft_r3_design.md` (on branch `textcraft-r3-dev-g2`, as are the deviation notes and the Ministral wording note), frozen at tag `textcraft-r3-design-frozen-2026-11-23`, with the dated
  deviations D-T1, D-G1, D-T2 and D-G2 (section 15 of the design and `notes/deviations.md`).
- **Code and analysis run:** branch `textcraft-r3-dev-g2`, tag `textcraft-r3-deviation-g2-2026-11-25` (commit
  `fae00b3`). The analysis run `textcraft_r3_analysis_dg2` ran once, from a clean checkout at that tag (ICE CPU job
  7330172; run folder `git_sha` equal to the tag's commit, `git_dirty` false).
- **Metrics:** `notes/textcraft_r3_interim_metrics.json`.

## 1. What round 3 asks, and what was run

Round 3 asks whether round 2's measured quantities behave the same way in recent open-weight instruction models when
everything else is held fixed: the same 2,982 records, the same goal interventions, the same clean world-model
prompt, the same scoring and the same estimands. Its frozen comparisons fall into three groups:

- **Direct replication (families A3 and A3x).** Each new model goes through round 2's five confirmatory contrasts. Its
  tests are corrected with Holm's method within a family of 30 tests: three models × five contrasts × two tests ("does
  it differ from zero" and "is it equivalent to zero").
- **Older against newer releases (family B3).** Three pairs, each a round-2 model and a contemporary model of the same
  developer line. They are compared contrast by contrast on a standardized scale (a mean divided by the item standard
  deviation), with an equivalence margin of 0.20 standardized units.
- **Exploratory readouts.** These carry estimates and intervals but no test: the thinking arm, the exploratory
  Gemma 4 E4B run, and the capability table.

Every score is the two-candidate log-odds $L$. The model reads a crafting history and an action and is asked what
happens next. $L$ is the log-probability it gives the success reply minus the log-probability it gives the failure
reply, each read after the prompt. The goal line of the prompt is the intervention. Each contrast is a mean over
failures, with goal pairs as the resampling clusters. In plain words, the five contrasts are as follows.

**The intermediate lift (Λ).** On 990 failed crafts of an item that is an ingredient of the stated goal: how much
more likely the model judges the craft to succeed when the goal line names the goal it serves (aligned) than when it
names an unrelated goal (swapped),

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}.$$

A positive value means a goal that needs the item makes the model more optimistic about crafting it.

**The within-item penalty (P).** On 675 failed crafts: the same craft is judged with the item named as the goal
(aligned) and with the item named only as an ingredient of a parent goal (parent),

$$P = \overline{L(\text{aligned}) - L(\text{parent})}.$$

A negative value means the model is less optimistic about crafting the item when doing so would complete the stated
goal.

**Naming (Dn) and completion (Dc).** These split the penalty using a two-item goal line that names the item without
completing the goal ("both"). Each is a difference between goal-role failures and class-matched ingredient-role
failures:

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}},$$

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}}.$$

**The sensitivity to added empty steps (T).** How the goal effect changes when eight empty inventory look-ups are
added to the history instead of one, for goal-role against ingredient-role failures. Under the frozen rule it is "not
applicable" when the one-look-up role gap has no interval below zero.

**Verdicts.**
- *Confirmed above or below zero:* the Holm-adjusted p of the "differs" test is below 0.05 and the 95% interval lies
  on that side of zero.
- *Equivalent to zero:* the Holm-adjusted p of the equivalence test against the model's margin is below 0.05. The
  margin is the smaller of 0.3 item standard deviations and 0.4 nats.
- *Inconclusive:* neither of the above. It is flagged when the equivalence test has under 80% power at a true zero.

**Competence.** A model is read as distorting a prediction it can make only if it passes round 2's competence gate:
it must separate successes from failures on the yes/no question (an AUC of at least 0.70), must not answer Yes to
everything, and must put at least half its probability on Yes and No. Otherwise an effect is "measurable only".

## 2. The gates, and which models enter

The environment check rescored 300 round-2 records of the three older models in round 3's software (serve2). It
**failed** for all three: Qwen3-8B, Mistral-7B and Gemma-2-9B each failed the yes/no per-score criterion, and Qwen3-8B
also failed the two-candidate one. Every check of the goal effect itself passed. So, as frozen, B3 compares each older
model's serve2 rescoring of every record (jobs 7252995–7253000) with the newer model, never round 2's original scores.
The analysis found no provenance problem in any shard it read: commit, GPU type, model revision and clean checkout
were all as required.

| Model | Role | State in this reading | Why |
|---|---|---|---|
| Qwen3.5-9B | core (A3) | scored | every gate passed |
| Ministral-3-8B | core (A3) | scored | every gate passed |
| Gemma 4 12B | core (A3), the Gemma slot | **waiting** | under D-G2 the attention check is read in float32, and 12B's float32 check and smoke rerun are not run yet |
| OLMo 3 7B | extension (A3x) | scored | every gate passed |
| Qwen3.8-27B | extension (A3x) | waiting | scoring queued for 80 GB A100s |
| Gemma 4 31B | extension (A3x) | waiting | scoring queued; its float32 attention check is not run yet |

A model that is "waiting" enters its family with its tests padded at p = 1, so the families keep their frozen size of
30 tests. Readings that need every model of a family are withheld until all are filed: the account's prediction
H3-P, the contemporary rejection rule R2 and the replication rule. These readings are therefore **not available in
this interim reading** for A3 or A3x.

## 3. Direct replication: per-model verdicts

Estimates are in nats with 95% goal-pair bootstrap intervals. "Holm p" is the Holm-adjusted p of the test that
decides the verdict, within the model's family of 30. The minimum reachable with 20,000 resamples is 0.0015.

### Qwen3.5-9B (core)

Competent overall: yes/no AUC 0.905, Yes to 4.4% of failures and 56.5% of successes, median yes/no mass 0.994. Its
equivalence margin is 0.175 nats.

| Contrast | Estimate [95% CI] | Verdict | Holm p |
|---|---|---|---|
| Intermediate lift Λ | +0.074 [+0.034, +0.114] | confirmed above zero | 0.016 (differs) |
| Within-item penalty P | −0.020 [−0.085, +0.050] | equivalent to zero | 0.0015 (equivalence) |
| Naming Dn | +0.053 [−0.039, +0.142] | inconclusive (equivalence underpowered) | 0.090 (equivalence) |
| Completion Dc | +0.125 [+0.012, +0.242] | inconclusive (equivalence underpowered) | 0.56 (differs) |
| Added empty steps T | −0.086 [−0.161, −0.011] | not applicable | — |

Rejection rule R1 ("penalty absent") fires for Qwen3.5-9B: its within-item penalty is equivalent to zero, whereas
round 2's Qwen3-8B showed −1.02 nats. Its reading is "lift present; no penalty larger than the margin under the clean
prompt".

### Ministral-3-8B (core)

Competent overall: AUC 0.864, Yes to 10.2% of failures and 65.7% of successes, median yes/no mass 0.925. Its margin
is 0.40 nats.

| Contrast | Estimate [95% CI] | Verdict | Holm p |
|---|---|---|---|
| Intermediate lift Λ | +0.294 [+0.206, +0.385] | confirmed above zero | 0.0015 (differs) |
| Within-item penalty P | −1.231 [−1.441, −1.027] | confirmed below zero | 0.0015 (differs) |
| Naming Dn | −0.307 [−0.529, −0.087] | inconclusive (equivalence underpowered) | 0.12 (differs) |
| Completion Dc | −0.014 [−0.261, +0.248] | equivalent to zero | 0.042 (equivalence) |
| Added empty steps T | −0.075 [−0.270, +0.117] | equivalent to zero | 0.013 (equivalence) |

No rejection rule fires. The reading is "lift present; penalty present, mechanism unresolved", and the central
question's answer is "penalty with shortcut-consistent tracking".

Format note: Ministral puts only about half its next-token probability on the two replies' first tokens (median
0.49; round 2's Mistral-7B: 0.97). Its penalty is carried mostly by the later tokens of the replies: first token
−0.11, the rest −1.12 nats (`notes/textcraft_r3_ministral_wording_note.md`).

### OLMo 3 7B (extension)

**Not competent:** AUC 0.560, and Yes to 99.8% of failures. Its effects are therefore "measurable only" and are not
read as a distortion of a discriminating prediction. Its margin is 0.40 nats.

| Contrast | Estimate [95% CI] | Verdict | Holm p |
|---|---|---|---|
| Intermediate lift Λ | +0.456 [+0.307, +0.613] | confirmed above zero (measurable only) | 0.0015 (differs) |
| Within-item penalty P | +0.003 [−0.227, +0.246] | equivalent to zero (measurable only) | 0.034 (equivalence) |
| Naming Dn | −0.270 [−0.646, +0.062] | inconclusive | 1.0 |
| Completion Dc | +0.381 [+0.084, +0.684] | inconclusive | 0.25 (differs) |
| Added empty steps T | +0.105 [−0.042, +0.259] | not applicable | — |

### The Gemma slot, Qwen3.8-27B and Gemma 4 31B

These have no verdict yet. Their tests are padded at p = 1.

## 4. Older against newer releases (B3)

The comparison uses each model's standardized contrast. "Differs" and "similar within the margin" are decided by
Holm-adjusted tests in B3's family of 30; the Gemma pair's 10 tests are padded at p = 1. Raw nats are descriptive.

**Qwen3-8B (serve2 rescoring) against Qwen3.5-9B.** Both are competent.

| Contrast | Standardized difference, newer minus older [95% CI] | Reading | Holm p |
|---|---|---|---|
| Λ | −0.016 [−0.137, +0.119] | similar within the margin | 0.023 (equivalence) |
| P | +0.172 [+0.014, +0.319] | not distinguishable (equivalence underpowered) | 0.80 (differs) |
| Dn | +0.075 [−0.135, +0.285] | not distinguishable | 1.0 |
| Dc | +0.048 [−0.159, +0.248] | not distinguishable | 1.0 |
| T | −0.487 [−0.727, −0.258] | differs, newer lower (opposite signs) | 0.0015 (differs) |

In raw nats the penalty shrinks from −0.99 to −0.02, a change of +0.97 [+0.41, +1.58]. On the standardized scale the
two are −0.21 and −0.04 SD, because Qwen3-8B's item values are about eight times as spread. So most of the raw drop
is a change of scale, and the standardized difference is not distinguishable from zero.

**Mistral-7B (serve2 rescoring) against Ministral-3-8B.** Mistral-7B is not competent, as in round 2 (it answers Yes
to almost every failure), so the pair compares a non-discriminating model with a discriminating one.

| Contrast | Standardized difference [95% CI] | Reading | Holm p |
|---|---|---|---|
| Λ | −0.017 [−0.120, +0.081] | similar within the margin | 0.009 (equivalence) |
| P | −1.022 [−1.225, −0.834] | differs, newer lower (opposite signs) | 0.0015 (differs) |
| Dn | −0.792 [−1.021, −0.569] | differs, newer lower (opposite signs) | 0.0015 (differs) |
| Dc | −0.055 [−0.292, +0.171] | not distinguishable (equivalence underpowered) | 1.0 |
| T | +0.353 [+0.144, +0.568] | differs, newer higher (smaller in size) | 0.020 (differs) |

The pair also differs in reply format: the first-token mass is 0.97 for Mistral-7B against 0.49 for Ministral. Even a
standardized comparison therefore sets a model that follows the reply wording against one that mostly does not.

**Gemma-2-9B against the Gemma slot.** Waiting, with the slot.

## 5. Exploratory readouts (no test)

**Thinking arm (Qwen3.5-9B with reasoning on, deviation D-T1).** There are 4,530 sampled reasoning traces, scored
after the reasoning. 36.7% of the traces reached the 2,048-token cap and were closed by force. The median reasoning
is 671 new tokens. 67.5% of the traces name the goal ("goal", "task" or "objective"), and nearly all (99.96%) name the
crafted item. After reasoning:

- **Penalty.** The within-item penalty is −0.42 nats [−0.74, −0.11], or −0.10 SD [−0.18, −0.03], against −0.02 nats
  without reasoning. The difference with reasoning minus without is −0.40 nats [−0.71, −0.10] in raw nats and −0.07 SD
  [−0.20, +0.07] standardized.
- **Lift.** The intermediate lift is +0.17 nats [−0.21, +0.56]. Standardized, reasoning lowers it by −0.12 SD [−0.23,
  −0.02].
- **The unforced traces only** (2,869 scores; a selected subset): penalty −0.18 nats [−0.57, +0.22] and lift −0.14
  [−0.72, +0.44].
- **Discrimination** on the yes/no question after reasoning: AUC 0.965.

So reasoning moves the penalty in raw nats but not clearly on the standardized scale, where the traces add spread.
The result depends on the scale, as for round 1's thinking probe.

**Exploratory Gemma 4 E4B (deviation D-G1, outside every family).** It is competent: AUC 0.811, Yes to 28.7% of
failures, median mass 0.998. Its estimates:

| Contrast | Nats [95% CI] | Standardized [95% CI] |
|---|---|---|
| Λ | +0.100 [+0.031, +0.164] | +0.111 [+0.029, +0.195] |
| P | −0.296 [−0.450, −0.149] | −0.247 [−0.349, −0.132] |
| Dn | −0.195 [−0.374, −0.029] | −0.191 [−0.364, −0.033] |
| Dc | +0.299 [+0.159, +0.447] | +0.325 [+0.170, +0.482] |
| T | −0.010 [−0.130, +0.104] | −0.014 [−0.170, +0.148] |

Its confirmatory state is "waiting" (the Gemma slot). This block never enters a family.

## 6. Numerical fragility

Following `notes/numerical_sensitivity.md` (Rule 3), a reading is "numerically fragile" when its deciding interval
ends within about twice the numerical standard deviation of its boundary. That is about 0.1 nats for Qwen-sized
logits, 0.03 to 0.05 nats for other models, and about 0.05 on B3's standardized scale. The fragile readings are:

- **Qwen3.5-9B, intermediate lift, "confirmed above zero".** The 95% interval ends 0.034 nats above zero.
- **Qwen3.5-9B, within-item penalty, "equivalent to zero".** The 90% interval ends 0.100 nats inside the 0.175 margin,
  at the edge of the Qwen allowance. This reading also triggers R1.

The Qwen3.5-9B readings for naming, completion and added empty steps are inconclusive or not applicable, and their
intervals lie near their boundaries too, so a change of environment could move them. No Ministral, OLMo or B3
verdict is within the allowance of its boundary.

Qwen3.5-9B's logit size was not measured here; the Qwen-family allowance is used as the conservative choice.

## 7. What the queued models can still change

- **The Gemma slot** (Gemma 4 12B, or E4B if the fallback rule fires under D-G2) fills A3's last 10 tests and B3's
  Gemma pair. It needs new scoring of about 4.5 to 5.5 GPU-hours if 12B holds the slot, not yet submitted.
- **Qwen3.8-27B and Gemma 4 31B** fill A3x.
- **Holm adjustment.** When a padded test (p = 1) is replaced by a real p-value, the Holm-adjusted p of each existing
  test can only stay the same or fall, because the family size stays 30 and a smaller p ahead of it only lowers its
  multiplier. So a verdict reported here cannot be lost when those models are filed. Some inconclusive tests could
  move: for example, Ministral's naming test (raw p 0.0054) would need about 21 smaller p-values ahead of it, which is
  unlikely.
- **The family readings.** H3-P, R2 (contemporary) and the replication readings for A3 and A3x are computed only when
  their families are complete. They depend on the slot model and on the extension models.
