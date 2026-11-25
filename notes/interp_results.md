# Interpretability pilot ("The Task Is Not the World"), minimal pilot: results (2026-11-24)

**Status.** Exploratory, and a calibration run by design. The design is [interp_design.md](interp_design.md) with amendments 1 to 5, frozen at the annotated tag `interp-design-frozen-2026-11-22` (commit `7ade3b9`, branch `interp`). The three GPU stages (stage 0, patching, ablation) ran at that commit in one job (7245822, one NVIDIA L40S). The CPU analysis crashed at the frozen commit; the fix ([interp_deviation_analysis_fix.md](interp_deviation_analysis_fix.md), tags `interp-deviation-analysis-fix-2026-11-23` at `06e3871` and `interp-deviation-analysis-fix-2026-11-24` at `c73147e`) was made before any analysis output existed, and the analysis that this note reports ran at `c73147e` (job 7295359) in the clean ICE checkout `text-wm-fix41`. Every number below is quoted from that run's `metrics.json` (1,386,852 bytes, SHA-256 `32c15ec3409f5cab…`), of which [interp_metrics.json](interp_metrics.json) is a compact extract with the values copied unchanged. Nothing is recomputed. Where this note does arithmetic on reported numbers (a mean of the two folds' values, a difference, a count of layers), it says so. Paragraphs headed "Observation" are things this note noticed in the outputs; they are not readouts and decide nothing.

The design says plainly what this run is for (section M, amendments C1 and D6). The minimal pilot calibrates the full pilot: it checks that the tooling reproduces the scored effect on the real model, it gives a first localization, and its standard errors set the full pilot's sample sizes. **It gives no verdict about any hypothesis.** Its output names no hypothesis; each pre-specified row is reported as a *reading* against bands the design calls provisional ("low", "after arrival", "not assessable", and so on). The mapping from readings to hypotheses is applied only in the full pilot, on the other fold's games. This note keeps to that.

## Summary

The pilot asks where, inside Qwen3-8B, the agent's task gets into the model's yes/no prediction of whether an ALFWorld action will succeed, for actions whose failure the history already decides. The yes/no log-odds rise by 2.78 nats on average when the task names the action's object instead of another object in the room. The tooling reproduced this item by item on all 4,958 items. Both reproduction gates passed, with mean effects within 0.005 nats of the logged scores.

**Patching** (100 items with a clear effect, from one half of the games) traces the difference along one path. It starts at the object's name in the task line (the only tokens that differ), moves by about layers 11 to 18 into the *action line*, and mostly into the action's own object word. From about layer 19 it reaches the final template tokens and the answer position, which carry essentially all of it from layer 22 on. The history after the task line carries almost none of it. Its largest noising share is 0.031 [0.010, 0.066] over all items, read as **"low (below 0.15)"**. That reading describes the typical item. On the 25 sampled items whose history mentions the action's object, the mention tokens carry 0.34 [0.15, 0.54] under noising and 0.75 [0.47, 1.10] under denoising, against 0.02 for a position-matched control. Those items are about 3% of the population.

**Ablation** removes one direction, the mean task-induced change at the answer position, at one layer. It removes nothing at layers 4 to 16, about a quarter at layer 20, and most of the effect at layers 24 and 28. After step 0, the earliest layer from which the 95% lower bound of the removed share stays at least 0.6 is layer 28. The answer-position arrival from patching is layer 24, so the frozen reading is **"after arrival"**. That reading rests on layer 24's lower bound of 0.566, 0.034 short of the threshold. At step 0 and pooled over strata, no layer qualifies, and the intervals are too wide to say "no single direction". Both read **"inconclusive (no onset, interval too wide)"**. At the chosen layer the ablation keeps the success-versus-failure gap on neutral prompts (ratio 0.938 [0.909, 0.963], **"kept"**). It also moves neutral-task prompts up by 1.2 nats.

**The rows that separate a generic success feature (H4) from a goal-relevance signal (H3)** give one value and two blanks. Before the arrival layers, the task direction and the model's neutral success direction have a cosine of 0.44 once the readout direction is projected out (0.44 with a second projection too; per-layer intervals above 0.25 from layer 12 on). That is above the 0.3 the design writes for H4 and far above the 0.1 it writes for H3. The two intervention rows that H4 also needs are **"not assessable"**: steering along the success direction relative to steering along the task direction, and removal by the task direction without its success component. At every layer before the arrival, edits at the answer position move the readout by almost nothing, so the gates that guard these ratios never open.

The calibration output says the full pilot needs these sizes per stratum: about 19 patching items and 112 ablation failures at step 0; 81 patching items and 1,134 ablation failures after step 0, capped at 400. At the cap, the removal at layers 24 to 32 reaches a half-width of only 0.11 to 0.13, against the 0.075 target. The object-mention share needs 184 mention items, about three times what the other fold holds.

| Pre-specified row (minimal pilot) | Step 0 | Later steps | All (weighted) |
|---|---|---|---|
| Reproduction gates (yes/no; first token) | pass; pass (all items) | | |
| History share, noising (max over layers) | undefined (no history after the task line) | 0.031 [0.008, 0.066], low | 0.031 [0.010, 0.066], low |
| Arrival, answer position (onset rule, grid) | layer 24 | layer 24 | layer 24 |
| Onset of one-direction removal (lower bound at least 0.6) | none (deepest upper bound 0.570) | layer 28 | none (deepest upper bound 0.976) |
| Onset against arrival | inconclusive (no onset, interval too wide) | after arrival | inconclusive (no onset, interval too wide) |
| Neutral success-failure gap, ablated over clean, layer 28 | undefined (no success at step 0) | 0.931 [0.900, 0.958], kept | 0.938 [0.909, 0.963], kept |
| H4 cosine, readout projected out (median, layers 4 to 20) | 0.442 (SE 0.062) | same | same |
| H4 cosine, readout and answer direction projected out | 0.436 | same | same |
| H4 steering ratio | not assessable | not assessable | not assessable |
| H4 removal without success component, over task removal | not assessable | not assessable | not assessable |

## 1. The question

### 1.1 The finding being explained

A **world model** here is a language model asked to predict what an environment replies to an agent's action. In ALFWorld, a text household game, a failing action always gets the reply "Nothing happens." The project's RQ2 finding concerns 4,666 failing actions whose failure the history already decides: the agent is not where the action needs it, the receptacle was seen without the object, and so on. For each one the prompt is built twice, and only the one-line task changes. The **aligned** task names the action's object, and the **swapped** task names another object in the room. The real environment ignores the task, so any difference in the prediction is an error. The model nevertheless predicts success more readily under the aligned task. This is the **goal effect**.

The pilot uses the **yes/no format** (the `rq1v` prompts, byte for byte). The prompt ends with a question about whether the action will succeed, and the readout is the difference of the two answer logits at the **answer position**. For Qwen3-8B with thinking off, the answer position is the last token of the empty `<think></think>` block that the chat template adds. The readout is

$$
D = z_{\text{Yes}} - z_{\text{No}},
$$

which equals the log-odds of "Yes" against "No", in **nats** (natural-log units; one nat multiplies the odds by about 2.7). The goal effect on an item is

$$
\Delta D = D(\text{aligned}) - D(\text{swapped}).
$$

Its logged mean for Qwen3-8B is 2.80 nats. Its sign and size are what this pilot tries to localize inside the network. Qwen answers "No" on nearly all of these items under both tasks (amendment C8). The pilot therefore localizes a shift in log-odds inside an answer that stays "No", not a change of answer. It also does not explain the 6.12-nat headline effect of the two-candidate format, which is a different readout of the same manipulation.

### 1.2 The competing accounts

The design (section 1) lists six accounts that the behavioural studies could not separate:

1. **H1, no state tracking.** The model does not represent the facts that decide the failure, so it falls back on the task.
2. **H2, belief change.** The model represents those facts, but the task changes them; for example, it comes to believe that the object is at the receptacle.
3. **H3, relevance at answer time.** The represented state is unchanged, and a "this action is about the goal" signal is added late, near the answer.
4. **H4, a generic success feature.** As restated in amendment A5, the object match triggers the same feature that separates working from failing actions under a neutral task.
5. **H5, policy blending.** The prediction reuses the agent's "should I take this action" representation.
6. **H6, prompt artefact.** Surface machinery, such as lexical repetition of the object's name, carries the effect.

The central contrast is "late entry, state unchanged" (H3, H4) against "state changed in the middle layers" (H1, H2). The minimal pilot cannot test H1 or H2 directly, because the history-fact probes need stage 1, which it does not run. It also cannot test H5 or transfer to described goals. It measures where the effect sits and whether one direction carries it, and it calibrates.

## 2. What was measured

### 2.1 Items, folds and strata

The population is RQ2's 4,666 determined failures and the 292 determined successes of `rq1x`, from 125 `valid_unseen` games. The games are split into two **folds** by a seeded shuffle (split seed 0). Fold 0 is the **calibration fold**. In this pilot only its items are patched or edited; fold 1's items are never intervened on, so the full pilot can read its verdicts there. Stage 0, which only records, runs on both folds.

Every readout is reported in three **strata**: **step 0** (the agent has not acted yet, so no history follows the task line), **later steps**, and **all**. The pooled "all" numbers are weighted back to the population. Patching drew 30 step-0 items, 25 later items whose history names the action's object, and 45 other later items. Their weights are 0.443, 0.123 and 1.859, each stratum's share of the eligible items over its share of the sample. The ablation drew 20 step-0 failures (weight 0.396), 55 later failures (weight 1.220) and 50 determined successes (weight 1.0). All of them are calibration-fold items.

### 2.2 Activation patching

The aligned and swapped prompts differ only in the object's name in the task line. **Noising** runs the aligned prompt and overwrites the activations of one group of token positions, at one layer, with those from the swapped run. It asks whether those activations are *necessary* for the effect. **Denoising** runs the swapped prompt and writes in the aligned activations. It asks whether they are *sufficient*. The **fraction restored** by a noising cell is

$$
R = \frac{D(\text{aligned}) - D(\text{aligned, cell patched from swapped})}{D(\text{aligned}) - D(\text{swapped})},
$$

so 1 means the cell carries the whole effect and 0 that it carries none. The denoising fraction is defined symmetrically. Fractions are pooled over items as a ratio of sums, weighted, with game-clustered bootstrap intervals (2,000 resamples). Only the residual stream was patched, at all 36 layers, for 12 token groups. In words, the groups are:

- **span**: the object's name in the task line, on items where the two names have the same number of tokens;
- **task rest**: the rest of the task line;
- **history, all**: every history token after the task line;
- **history, action's object**: history tokens naming the action's object;
- **history, matched control**: as many history tokens naming neither object;
- **history, swap object**: history tokens naming the swapped object;
- **action**: the "Action: …" line, split into its **object** word with instance number and the **rest**;
- **query**: the yes/no question;
- **final**: the template tokens after the user turn, including the answer position;
- **last**: the answer position alone.

The pre-specified **summary measures** are these:

- the **history share**, the largest noising fraction of "history, all" over layers, with a bootstrap interval in which the best layer is re-chosen in every resample. It is *low* if the upper bound is below 0.15 and *high* if the lower bound is at least 0.3 (A6). It is undefined at step 0, where no history follows the task line (A2).
- the **object-mention share**, the same for "history, action's object", next to its matched control (B4).
- the **action share**, the same for the action line, split into object and rest (B7).
- the **arrival layer of the final group**, the first layer from which noising the final group restores at least half of the effect for good, as a share of depth (section 5). Since C1 it is reported descriptively.
- the **answer-position arrival**, the earliest layer of the grid 4, 8, …, 32 from which the 95% lower bound of the "last" group's noising fraction stays at least 0.6 (C1). This is what the ablation onset is compared with.
- the **early-span share**, the span's noising fraction at the end of the first third of layers (layer 11) minus that at the end of the second third (layer 23), on equal-length items.
- a **placebo floor**: 50 placebo pairs (the swapped object against a second non-goal object) give, per cell, the mean absolute change in nats. A cell counts as carrying the effect only if the Bonferroni lower bound of its removed nats exceeds the Bonferroni upper bound of the placebo's change (A7). This is descriptive.

### 2.3 Direction ablation

The **task direction** at layer $\ell$ is the unit mean, over the other fold's failures, of the answer-position residual under the aligned prompt minus that under the swapped prompt:

$$
u_\ell = \frac{\overline{h_\ell(\text{aligned}) - h_\ell(\text{swapped})}}{\big\lVert \overline{h_\ell(\text{aligned}) - h_\ell(\text{swapped})} \big\rVert}.
$$

A **mean ablation** sets the residual's component along $u_\ell$, at the answer position and at one layer, to the midpoint $c_\ell$ of the other fold's mean projections under the two tasks:

$$
h \leftarrow h - \left(u_\ell^\top h - c_\ell\right) u_\ell .
$$

The **removed share** is one minus the edited goal effect over the unedited one, summed over items with weights, with a game bootstrap. A direction estimated on one fold is applied to the other fold's items. This **cross-fitting** keeps the estimate and the test apart.

At the last layer, this test removes about everything whatever the mechanism, because the readout is nearly linear there (B1). The pre-registered summary is therefore a **depth profile** over layers 4, 8, …, 28 and its **onset layer**: the earliest profile layer from which the 95% lower bound of the removed share stays at least 0.6 at every deeper profile layer. Layer 32 is ablated as a caveat layer that enters nothing. Layer 35 is a sanity edit, expected to remove about everything.

A second tautology (C2) comes from the part of the effect already lying along the readout direction at a layer, the **direct-path share**. With $r$ the readout direction (the final-norm weight times the difference of the two unembedding rows), $\rho_i$ the root mean square of a prompt's last-layer residual, and $D_i$ the final readout,

$$
\lambda_\ell = \frac{\sum_i \left( r^\top h^A_{\ell,i} / \rho^A_i - r^\top h^S_{\ell,i} / \rho^S_i \right)}{\sum_i \left( D^A_i - D^S_i \right)} ,
$$

computed on every failure of the calibration fold. A profile layer is excluded from the onset only if the upper bound of this share is at least 0.3 *and* its removal passes (E1).

The **onset against arrival** reading compares the onset with the answer-position arrival, rounded up to the grid (C1, D1). The possible readings are "at or before arrival", "after arrival", "no single direction" (no onset, and the deepest kept layer's removal has an upper bound below 0.3), "inconclusive (no onset, interval too wide)", and three kinds of "not assessable". For the full pilot, B1 maps "after arrival" (or no direction) to the accounts in which the state changes (H1, H2), and "at or before arrival" to H3 to H6. D9 adds that "at or before arrival" is not H3 on its own. The minimal pilot does not apply this mapping.

The other ablation readouts are taken at one **chosen layer**, the profile layer with the largest removal. With one fold edited, it is chosen on the same fold's items, flagged "in-sample" (B3); here it is layer 28. They are:

- the removal by the task direction **without its success component**, by the **mention direction** (swapped-plus-mention prompt minus swapped: merely naming the object, not as the goal), and by the task direction **from that layer on**;
- the **neutral gap**, the readout's success-versus-failure difference under the neutral task ("Your task is to: explore the room."), with and without the ablation, and the ratio of the two (defined below);
- the **collateral shift**, the mean change of the readout on neutral prompts;
- the removal on **determined successes**;
- **steering**, the gain on swapped prompts from adding, along the success direction, the mean aligned-minus-swapped projection onto it, as a share of the goal effect.

The neutral gap is the mean readout of determined successes minus that of determined failures, both under the neutral task:

$$
G = \overline{D \mid \text{success}} - \overline{D \mid \text{failure}} .
$$

It is **kept** if the ablated gap is at least 90% of the clean one (A3). The neutral AUC is at its ceiling, so it is reported but not read.

The **success direction** $s_\ell$ is a within-verb weighted mean of the neutral-prompt residual of determined successes minus that of determined failures after step 0 (B2).

### 2.4 The H4 rows

H4 says the task triggers the model's own success feature. Both the task direction and the success direction raise "Yes", so both lean on the readout direction whatever the mechanism. The cosine is therefore read after the readout direction $\hat r$ is projected out of both (B2):

$$
\cos(u_\perp, s_\perp), \qquad v_\perp = \frac{v - (v^\top \hat r)\hat r}{\lVert v - (v^\top \hat r)\hat r \rVert}.
$$

It is also read after a layer-specific **answer direction** is projected out, a ridge fit of the final readout on the layer's neutral residual within outcome-and-verb cells (C3). That second projection is read only where the answer direction's absolute cosine with the readout-free success direction is below 0.5 in both folds (D7, E4). The rows are medians over the **H4 layers**: profile layers up to two thirds of depth (4 to 24), strictly before the earlier of the two arrival layers (D4).

The two intervention rows are gated. The **steering ratio** is the gain from steering along the readout-free success direction over the gain from steering along the task direction itself. It is read only at layers where the task steering's share has a lower bound of at least 0.3 (D3). The **perp-over-task ratio** is the removal by the task direction without its success component over the removal by the task direction. It is read only where the task removal has a lower bound of at least 0.3 (C3).

The design's bands, stated in C3 for the full pilot, are these. For H4: projected cosine at least 0.3 under both projections, with a bootstrap lower bound above 0; steering ratio at least 0.5; and, where assessable, perp-over-task below 0.5. For H3: cosine below 0.1, steering below 0.2, and perp-over-task at least 0.8. A **goal-role row** compares the mention direction's projected cosine with the task direction's.

### 2.5 The hidden-state probe (descriptive)

A logistic probe is trained on neutral prompts of one fold and tested on the other (A4). Its label is "the action's object was at the action's receptacle at reset", from the scene, not the history parser. It is read at the action's receptacle token and at the answer position, where the readout and success directions are first projected out. The probe reports its held-out AUC and the **standardised shift**, the aligned-minus-swapped change in the probe's logit in units of its own separation on neutral prompts, signed so that positive means "the object is there". At the answer position it also reports the shift left after subtracting what the readout's own change predicts (A1). In the minimal pilot it is descriptive (section M). Its κ bounds, which would turn a shift into a share of the effect, are not computed here.

### 2.6 The calibration rule

For each summary measure and stratum, the full pilot's sample size is the pilot's n times $(1.96\,\mathrm{SE}/0.075)^2$ times 1.2 (B3, C6). That is the n at which the 95% interval's half-width reaches 0.075. Per stratum, patching takes the largest need of the history share, action share and arrival, and ablation takes the largest need of the removals, gap ratio, steering ratio and perp-over-task ratio. Each is capped at 400 (D10). The object-mention share is planned in its own items (E6), and the pooled stratum has no plan (E7).

## 3. Stage 0: the reproduction gates

Stage 0 recomputed the yes/no readout of every item under the aligned, swapped and neutral tasks (and, for failures, the swapped task plus a sentence naming the action's object), 28,872 records in all. It also recomputed the first-token readout ("You" against "Nothing") of the failures' two-candidate prompts. Each was gated against the logged scores. The first-token gate checks the format that carries the headline effect, which the minimal pilot otherwise does not study.

| Check | Yes/no | First token | Threshold |
|---|---|---|---|
| Items (prompts) | 4,958 (14,874) | 4,666 (9,332) | |
| Mean goal effect, stage 0 | 2.782 nats | 2.518 nats | |
| Mean goal effect, logged, same items | 2.783 nats | 2.513 nats | difference at most 0.15 |
| Published effect | 2.80 | 2.52 | |
| Median absolute difference per prompt | 0.084 nats | 0.088 nats | at most 0.35 |
| Correlation with logged scores, prompts | 0.9998 | 0.9995 | at least 0.95 |
| Correlation with logged effects, items | 0.998 | 0.998 | at least 0.90 |
| Coverage of logged items | 100% | 100% | at least 99% |
| Item-hash mismatches | 0 | 0 | 0 |
| Gate | **passed** | **passed** | |

The logged scores came from a different GPU run. The per-prompt differences (median about 0.09 nats, 0.15% of yes/no prompts off by more than 1 nat) are what a GPU change produces. They are far below the gate's bounds.

The stage also measured **batch noise**: one prompt's readout across batch rows and batch sizes 1, 2 and 16 varies by 0.132 nats. By rule (A10), this sets the identical-prefix tolerance of the patching and editing jobs to five times that, 0.66 nats. 3,687 of the 4,958 items have an effect of at least 1 nat, the patching eligibility filter.

## 4. Patching: where the effect sits

### 4.1 The pre-specified shares

The table gives each summary measure's maximum over layers (the layer in parentheses) with its 95% bootstrap interval. The layer is re-chosen in every resample, so the interval includes the choice of layer. Noising means aligned prompt, swapped activations written in; denoising is the reverse.

| Measure | Direction | Step 0 (30 items) | Later (70) | All (100, weighted) |
|---|---|---|---|---|
| History share | noising | undefined | 0.031 [0.008, 0.066] (layer 9) | 0.031 [0.010, 0.066] (9) |
| History share | denoising | undefined | 0.040 [0.021, 0.102] (13) | 0.040 [0.022, 0.099] (13) |
| History share reading (noising, A6) | | undefined | **low** (below 0.15) | **low** (below 0.15) |
| Object-mention share (25 mention items) | noising | undefined | 0.340 [0.149, 0.541] (16) | same items |
| Its matched control | noising | | 0.022 [0.016, 0.057] | |
| Object-mention share | denoising | undefined | 0.752 [0.471, 1.080] (15) | 0.752 [0.473, 1.096] |
| Its matched control | denoising | | 0.018 [0.014, 0.062] | |
| Action share | noising | 0.830 [0.770, 0.882] (15) | 0.869 [0.793, 0.942] (15) | 0.859 [0.798, 0.918] (15) |
| of which the action's object word | noising | 0.819 [0.757, 0.882] | 0.806 [0.707, 0.896] | 0.809 [0.731, 0.884] |
| of which the rest of the action line | noising | 0.170 [0.110, 0.221] | 0.254 [0.175, 0.342] | 0.233 [0.168, 0.297] |
| Action share | denoising | 0.870 [0.770, 0.955] | 0.866 [0.776, 0.962] | 0.867 [0.798, 0.943] |
| of which the object word | denoising | 0.857 [0.741, 0.968] | 0.842 [0.717, 0.977] | 0.846 [0.748, 0.951] |
| of which the rest | denoising | 0.027 [0.013, 0.053] | 0.117 [0.077, 0.166] | 0.093 [0.064, 0.126] |
| Final group's arrival at one half (share of depth) | noising | 0.600 (layer 21) | 0.600 (21) | 0.600 (21) |
| Final group's arrival at one half | denoising | 0.629 (layer 22) | 0.629 (22) | 0.629 (22) |
| Answer-position arrival (onset rule, grid) | noising | layer 24 | layer 24 | layer 24 |
| Early-span share (equal-length items) | noising | 0.350 (22 items) | 0.297 (48) | 0.311 (70) |
| Early-span share | denoising | 0.507 | 0.581 | 0.560 |

The frozen rules give one band reading here: the history share is **"low (below 0.15)"** in the later and pooled strata. At step 0 it is undefined by construction. The object-mention share, action share, early-span share and final-group arrival have no numeric band in the minimal pilot. They are reported as values.

### 4.2 The path through the layers (descriptive)

The per-layer noising fractions on all 100 items (weighted) show one path. They are copied from the metrics; the layers named are where each curve crosses the values given.

- **The object's name in the task line** carries everything at layers 0 to 2 (span fraction 1.0, the only positions that differ). It falls to 0.73 by layer 8, 0.31 at layer 11 and 0.18 at layer 12, and stays below 0.07 from layer 15.
- **The action line**, almost all of it **the action's object word**, takes over. The object word's fraction is 0.27 at layer 11, 0.65 at layer 13 and 0.81 at layer 15, then falls to 0.16 at layer 20 and near 0 by layer 22. The rest of the action line peaks at 0.23 (layer 18).
- **The yes/no question** rises from layer 15 to a peak of 0.54 at layers 19 to 21. Then it drops to 0.27 at layer 22 and 0.03 at layer 24.
- **The final template tokens and the answer position** take it last. The answer position's fraction is 0.10 at layer 19, 0.25 at layer 20, 0.47 at layer 21, 0.91 at layer 22 and at least 0.94 from layer 23 on.
- **The history** carries almost nothing at any layer ("history, all": largest 0.031). The rest of the task line also carries almost nothing (largest 0.019).

Against the placebo floor (Bonferroni over 36 layers, noising, all items), the cells that carry the effect are as follows. The span carries it at layers 0 to 8. The action line, through its object word, carries it at layers 13 to 20. The question carries it at layer 17 and layers 20 to 22. The final group carries it at layers 19 to 35, and the answer position at layers 20 to 35. No history group and no task-rest cell does, in any stratum. At step 0 the action line exceeds the floor from layer 11 to 21. The placebo floor itself is 0.15 to 0.2 nats in early layers, the size of the batch noise, and about 0.9 nats at the answer position in late layers, where the placebo pair's own difference passes through.

Denoising tells the same story with the roles reversed. Writing the aligned action-object activations into the swapped run at layer 15 restores 0.85 of the effect, so the object word's state at that depth is sufficient as well as necessary. In the later stratum, the 18 items whose history names the swapped object give that group a noising share of 0.114 [0.039, 0.298] and a denoising share of 0.152 [0.091, 0.242].

## 5. Ablation: does one direction carry the effect?

### 5.1 The depth profile and the onset

Each cell is the removed share of the goal effect by the task direction at that one layer, with its 95% interval. The **direct-path share** at the same layer comes from stage 0, on every calibration-fold failure (246 at step 0, 2,083 later, 2,329 in all). The goal effect before the edit, as an unweighted mean of the sampled failures, is 9.28 nats at step 0, 3.06 later and 4.72 overall.

| Layer | Step 0 (20 failures) | Later (55) | All (75, weighted) | Direct-path share, all |
|---|---|---|---|---|
| 4 | 0.011 [0.000, 0.024] | −0.004 [−0.032, 0.025] | −0.000 [−0.019, 0.019] | −0.001 |
| 8 | 0.007 [0.000, 0.015] | 0.003 [−0.019, 0.030] | 0.004 [−0.012, 0.022] | −0.001 |
| 12 | −0.002 [−0.014, 0.011] | −0.003 [−0.024, 0.023] | −0.003 [−0.018, 0.013] | −0.003 |
| 16 | 0.002 [−0.007, 0.011] | −0.007 [−0.029, 0.015] | −0.005 [−0.021, 0.011] | −0.009 |
| 20 | 0.127 [0.109, 0.147] | 0.301 [0.242, 0.390] | 0.255 [0.214, 0.303] | −0.001 [−0.003, 0.002] |
| 24 | 0.358 [0.209, 0.523] | 0.808 [0.566, 1.118] | 0.689 [0.521, 0.873] | −0.046 [−0.071, −0.020] |
| 28 | 0.395 [0.251, 0.570] | 0.899 [0.638, 1.246] | 0.766 [0.579, 0.976] | 0.187 [0.169, 0.206] |
| 32 (caveat, not read) | 0.445 [0.299, 0.612] | 0.888 [0.637, 1.222] | 0.771 [0.594, 0.972] | |
| 35 (sanity, not read) | 0.499 [0.356, 0.660] | 0.890 [0.660, 1.202] | 0.787 [0.619, 0.975] | |

The direct-path share at layer 28 is 0.163 [0.151, 0.174] at step 0 and 0.197 [0.172, 0.221] later. No profile layer in any stratum has an upper bound of 0.3 or more, so **the direct-path rule excluded nothing**. The onset is read on all seven profile layers. The readings:

- **Step 0.** The lower bound never reaches 0.6 (layer 28: 0.251). The deepest layer's upper bound is 0.570, not below 0.3. Reading: **"inconclusive (no onset, interval too wide)"**.
- **Later steps.** Layer 28's lower bound is 0.638, so the onset is **layer 28**. Layer 24's lower bound is 0.566, short of 0.6. The answer-position arrival is layer 24, so the onset comes after it, and no layer in between was excluded. Reading: **"after arrival"**.
- **All.** Layer 28's lower bound is 0.579, so there is no onset. The deepest upper bound is 0.976. Reading: **"inconclusive (no onset, interval too wide)"**.

### 5.2 Readouts at the chosen layer (28, in-sample)

Layer 28 was chosen as the layer of largest removal on the calibration fold's own failures. This is flagged "in-sample", because only one fold was edited.

| Readout at layer 28 | Step 0 | Later | All (weighted) |
|---|---|---|---|
| Removal by the task direction | 0.395 [0.251, 0.570] | 0.899 [0.638, 1.246] | 0.766 [0.579, 0.976] |
| Removal by the task direction without its success component | 0.000 [−0.013, 0.017] | −0.017 [−0.043, 0.010] | −0.012 [−0.033, 0.005] |
| Removal by the mention direction | 0.219 [0.100, 0.369] | 0.581 [0.394, 0.835] | 0.485 [0.350, 0.634] |
| Removal by the task direction from layer 28 on | 0.395 | 0.899 | 0.766 |
| Neutral gap, clean (nats; AUC) | undefined | 29.44 [23.51, 36.01]; 0.991 | 29.32 [23.67, 35.76]; 0.993 |
| Neutral gap, ablated (nats; AUC) | undefined | 27.42 [21.31, 34.14]; 0.969 | 27.51 [21.72, 34.18]; 0.977 |
| Gap ratio (kept if at least 0.9) | undefined | 0.931 [0.900, 0.958], **kept** | 0.938 [0.909, 0.963], **kept** |
| Collateral shift on neutral prompts (nats) | +0.152 [−0.251, +0.541] | +1.282 [+0.923, +1.649] | +1.210 [+0.880, +1.565] |
| Removal on determined successes (50) | undefined | 0.055 [−0.934, 1.136] | 0.055 [−0.934, 1.136] |
| Steering along the success direction, share of the goal effect | 1.049 [0.861, 1.297] | 0.579 [0.463, 0.779] | 0.703 [0.571, 0.879] |
| Steering along it without its readout component | 0.894 [0.733, 1.106] | 0.470 [0.376, 0.634] | 0.582 [0.470, 0.730] |
| Mean steering gain on swapped prompts (nats) | +9.74 | +1.77 | +2.62 [+2.27, +3.03] |

Task direction "from layer 28 on" is the same edit as at layer 28 alone. Layers 32 and 35 are never part of it (B1), so the two rows agree exactly. Ablated from layer 4 on, the task direction removes 0.792 [0.638, 0.967] (all), about what layer 28 alone removes.

The design's A3 reads the neutral-gap condition at the onset layer. In the later stratum, the onset (28) and the chosen layer coincide. In the pooled stratum there is no onset, and the reported gap is at the chosen layer.

### 5.3 Sanity checks of the editing code

- **Random directions** (two per layer at layers 16 and 28) removed between −0.006 and +0.014 of the effect in every stratum, as expected (A7).
- **Rounding of edits** (B9): on the first edited prompt, the largest realized error of an edit was 0.122 (random direction, layer 28), in residual units. The task direction's norm there is about 38, so this is the bfloat16 rounding the design expected.
- **The last-layer sanity edit** is expected by B1 to remove about everything. It removed 0.787 pooled, 0.890 later and 0.499 at step 0. See Observation 2.

### 5.4 Linearity diagnostic (D9; reported, not read)

The diagnostic compares the task steering's share with the task removal at the same layer. On all items the ratio is 0.852 at layer 20, 0.845 at layer 24 and 0.862 at layer 28. At layers 4 to 16 both quantities are near zero and the ratio is meaningless (from −113 to +0.22).

## 6. The H4 rows and the goal-role row

The H4 layers are **4, 8, 12, 16 and 20**: profile layers up to 24 that lie strictly before the earlier arrival (the final group at layer 21). They are the same in every stratum, because both arrivals coincide across strata. The cosines are stage-0 quantities over every item of each fold, not split by stratum. So the cosine rows are identical in all three strata.

| Row | Value | Null baselines and context |
|---|---|---|
| Cosine of task and success directions, readout projected out (median over H4 layers, mean of folds) | **0.442** (SE 0.062) | random pair SD 0.016; between-fold reliability of the projected task direction 0.970, of the projected success direction 0.765 |
| Same, per layer (mean of folds, computed for this note), with fold-averaged 95% bounds | 4: 0.111 [−0.095, 0.240]; 8: 0.245 [0.066, 0.334]; 12: 0.442 [0.258, 0.499]; 16: 0.476 [0.332, 0.511]; 20: 0.588 [0.531, 0.628] | |
| Same, readout and answer direction projected out | **0.436** (all five layers pass the D7 gate; the answer direction's cosine with the readout-free success direction is at most 0.098) | |
| Disattenuated cosine | 0.520 | |
| Cross-fold cosine (one fold's task direction with the other's success direction) | 0.422 | |
| Raw cosine (reported, not read) | 0.442 | |
| Steering ratio (readout-free success over task) | **not assessable**: no H4 layer has a task-steering share with lower bound of at least 0.3 | task steering share at H4 layers: median 0.005 (all); largest lower bound 0.186 (layer 20) |
| Removal without the success component over task removal | **not assessable**: no H4 layer has a task removal with lower bound of at least 0.3 | largest lower bound 0.214 (layer 20) |
| Goal-role row: mention direction's projected cosine with the success direction | 0.335 (disattenuated 0.397) | task minus mention: 0.087 |

**The reading the rows give.** The cosine row is above the 0.3 the design writes for H4 under both projections. Its per-layer bootstrap lower bounds are above 0 at layers 8 to 20, and it is well above H3's 0.1. The two intervention rows are not assessable, so the H4 rows as a set cannot be read. The minimal pilot attaches no label in any case (C1).

The cosine rises steadily with depth across the H4 window, from 0.11 at layer 4 to 0.59 at layer 20. Layer 20 is the last layer before the final group's arrival at layer 21, where the design warns (D4) that directions tied to the answer align whatever the mechanism. The mention direction, which names the object without making it the goal, is almost as aligned with the success direction (0.335). By the C3 rule, the mention is not a null for H4, so this is reported as the goal-role row only. Descriptively, the task direction's alignment exceeds the mention direction's by 0.087.

**Why the intervention rows are empty.** At layers 4 to 16, editing the answer position at one layer changes the goal effect by at most 0.021 of its size, whatever the direction (sections 5.1 and 5.4). Patching shows why. At those depths the answer position holds none of the effect (noising fractions of at most 0.022 in size up to layer 18), and later layers re-read the context tokens, so an edit there is overwritten. The effect reaches the answer position from layer 19 to 22. The H4 window ends before layer 21. So in this model and format, the layers where answer-position interventions can move anything and the layers where the H4 rows may be read barely overlap: only layer 20 is in both, and its gates do not open (lower bounds 0.186 and 0.214).

## 7. The hidden-state probe (descriptive)

4,796 items carry the label "the action's object was at the action's receptacle at reset" (17.7% true). For each layer, the table gives the held-out AUC on neutral prompts and the standardised shift (aligned minus swapped, failures, all strata; positive means toward "the object is there"). At the answer position it also gives the residual shift, after subtracting what the readout's own change predicts. No intervals are computed for these quantities.

| Layer | Receptacle token: AUC | Receptacle token: shift | Answer position: AUC | Answer position: shift | Answer position: residual shift |
|---|---|---|---|---|---|
| 0 | 0.816 | −0.001 | 0.786 | 0.014 | −0.010 |
| 4 | 0.856 | 0.154 | 0.751 | −0.005 | −0.032 |
| 8 | 0.876 | 0.154 | 0.758 | −0.033 | −0.051 |
| 12 | 0.866 | 0.138 | 0.746 | −0.074 | −0.090 |
| 16 | 0.876 | 0.149 | 0.804 | −0.106 | −0.122 |
| 20 | 0.900 | 0.229 | 0.900 | 0.276 | 0.245 |
| 24 | 0.900 | 0.145 | 0.887 | 0.197 | 0.178 |
| 28 | 0.902 | 0.110 | 0.886 | 0.166 | 0.150 |
| 32 | 0.898 | 0.122 | 0.880 | 0.145 | 0.130 |
| 35 | 0.900 | 0.125 | 0.876 | 0.115 | 0.101 |

The fact is linearly decodable at both positions (held-out AUC 0.746 to 0.902). Under the aligned task, the probe at the receptacle token moves toward "the object is there" by about 0.14 to 0.23 of its own separation from layer 4 on. The answer-position probe moves slightly the other way up to layer 16 and toward "there" from layer 20, where the effect arrives at the answer position. These numbers are descriptive. They have no intervals, and they are not converted into the share of the effect they could carry (κ, A6). The history-fact probes that the belief rows need come from stage 1, which the minimal pilot does not run. As the deviation note records, balancing within true outcome keeps no labelled success, so this probe is in effect trained on failures only.

## 8. Calibration output: sample sizes for the full pilot

The rule multiplies the pilot's n by $(1.96\,\mathrm{SE}/0.075)^2 \times 1.2$. Per stratum:

| Stratum | Patching items (largest need) | Ablation failures (largest need) | Object-mention items |
|---|---|---|---|
| Step 0 | 19 (action share 19, arrival 2; history share undefined) | 112 (removal at layer 28: 112; layer 24: 108) | not applicable |
| Later steps | 81 (action share 81, history share 13, arrival 4) | 1,134 wanted (removal at layer 28), **capped at 400** | 184 |

When the cap binds in the later stratum, these are the half-widths each measure reaches at 400 items. The removal reaches 0.113 at layer 24, 0.126 at layer 28 and 0.120 at layer 32, against the target 0.075. It reaches 0.031 at layer 20 and about 0.01 at layers 4 to 16, and the gap ratio reaches 0.016. The steering and perp-over-task ratios have no SE (not assessable), so they enter no plan. For the pooled stratum (no plan), the needs are 76 for the action share, 18 for the history share, 19 for the gap ratio and 642 for the removal at layer 28.

The stage-0 quantities report SEs only: the H4 cosine (0.062) and the direct-path shares (at most 0.016).

## 9. Deviations, stated plainly

1. **The analysis crashed at the frozen commit, and one function was fixed before any output was read** (`interp_deviation_analysis_fix.md`, commit `06e3871`). The hidden-state probe at the answer position projects out "the neutral success direction of A3". The frozen code estimated it from the probe's own balanced training rows. Those rows contain no success, because every labelled success is a "take" whose object was there, and balancing by outcome drops them all. So the frozen analysis (job 7252128) raised "no verb has both a success and a failure". The fix takes the direction from the rows the design names: the training fold's determined successes and its failures after step 0, the same rows the ablation and H4 code use. Only the answer-position columns of the hidden probe (section 7) depend on this. They are descriptive and feed no other number. The crashed run had written no metrics. A test reproduces the crash, and four mutants of the fix were killed.
2. **The analysis trusts the GPU stages' commit** (commit `c73147e`, addendum to the same note). The dependency check requires every input run to have finished at the analysis's own commit. The GPU stages ran at `7ade3b9`, and the fix touches only analysis code. The analysis config therefore lists `7ade3b9` under `accept_commits` for the three stages, which is the project's existing mechanism for this. The first rerun (job 7253221) had stopped at that check before computing anything. No stage was rerun.
3. **Operational: scikit-learn from outside the study environment.** The study environment has no scikit-learn, and the first analysis attempt (job 7251609) stopped on the import. Jobs 7252128 and 7295359 ran with `PYTHONPATH` pointing to a separate `pip install --target` folder holding scikit-learn 1.9.1 (the job's error log shows its path). The study environment itself was not changed. The log of job 7295359 shows only FutureWarnings from scikit-learn (40 of each of two kinds), no errors.
4. **No 10-item timing run.** Section M asks for a 10-item timing run of the patching stage before the full one, with the estimate redone from it. The job script calls it "advisable". This note found no record of such a run in the run folders or the design note. Patching ran at full size in the same job as stage 0. The whole GPU job took 3 hours 33 minutes on one L40S, against the design's estimate of 8.2 L40S-hours and the job's 12-hour limit.

The three failed analysis runs are kept as `runs/_failed_interp_pilot_analysis_7251609`, `_7252128` and `_7253221` on the cluster. None produced a result.

## 10. Observations

These are things this note noticed in the outputs. They are not readouts and decide nothing. Where one bears on a frozen reading, the reading stands as reported.

**Observation 1: the later stratum's "after arrival" hinges on one bound.** The reading follows from layer 24's lower bound of 0.566, against a threshold of 0.6. Had it been 0.034 higher, the onset would be layer 24, equal to the arrival, and the reading "at or before arrival". The point estimates at layers 24 and 28 (0.81 and 0.90) differ by less than either interval's half-width, which is about 0.28 to 0.30. The calibration output confirms this is a precision problem, not a measured lag. Even 400 later-step failures, the cap, leave the removal at layers 24 and 28 with half-widths of 0.11 and 0.13.

**Observation 2: the last-layer sanity edit is far from "about everything" at step 0.** B1 argued that a mean ablation at the last layer removes about the whole effect whatever the mechanism. It removed 0.890 later and 0.787 pooled, but 0.499 [0.356, 0.660] at step 0. B1's argument holds for items whose mean difference defines the direction. Here the direction is estimated, unweighted, on all of the other fold's failures, about 89% of which are later items. One reading consistent with the numbers is that step-0 items' aligned-minus-swapped difference lies partly off that pooled direction even at the last layer. If so, a single pooled direction can remove at most about half of the step-0 effect at any depth, and the step-0 "inconclusive" reflects that mismatch as much as anything about where the effect sits. A step-0-specific direction would be a design change. The midpoints are pooled by design (C4).

**Observation 3: at layer 28 the task direction without its success component removes nothing.** It removes −0.012 [−0.033, +0.005] while the task direction removes 0.766. At layer 24 it removes 0.449, or 0.65 of the task direction's removal. This note checked the ablation run's `directions.npz` (read only) to rule out a mislabelled direction. At layer 28 the success-free direction has cosine 0.914 (fold 0) and 0.921 (fold 1) with the task direction, and it is orthogonal to the success direction. Steering along the readout-free success direction at layer 28 also reproduces 0.88 [0.78, 1.00] of the gain from steering along the task direction (all items; 0.62 later, 2.36 at step 0). Together these are consistent with the part of the task-induced change that later layers turn into the answer lying, at layer 28, along the model's own success direction. Layer 28 is after both arrivals, though. The design keeps the H4 rows out of exactly this region, because there "two directions tied to the answer align whatever the mechanism" (B2, D4). So it is not evidence for H4 under the frozen rules. Both ratios are "assessable" at layers 24 and 28 and are listed in the extract. The design gives them no reading there.

**Observation 4: the denoising placebo flags can never be true.** All denoising "above the placebo floor" flags are false in every group, including the answer position at layers where denoising restores the whole effect. For denoising, the removed effect in nats (clean minus patched, where the clean run is the swapped prompt) is negative by construction. The placebo floor is a mean absolute change, and the comparison asks the removed effect's lower bound to exceed the floor's upper bound, which a negative number cannot do. The noising flags are meaningful and are the ones section 4.2 uses. The flags are descriptive and no rule reads them. The frozen code is not changed here.

**Observation 5: the history share and the object-mention share describe different items.** The whole history carries 0.031 at most, while its object-mention tokens alone carry 0.340. This is not a contradiction, but it is easy to misread. The history share is pooled over the 70 later items with population weights. The 25 sampled mention items get weight 0.123 there and the other 45 later items 1.859, because mention items are about 3% of the population. The object-mention share is computed on those 25 items alone. So "low" describes the typical item. On the rare items whose history names the action's object, the history's mention tokens are both necessary (noising 0.34) and close to sufficient (denoising 0.75) for a large part of the effect, against a matched control near 0.02. Whether this marks a belief change about that object or a second lexical route (the name repeated in the history) is beyond what patching shows.

**Observation 6: the final group's arrival is "late" by one definition and "middle" by another.** It arrives at 0.6 of depth. Section 9's outcome table writes H3's cell as "Late (at least 0.5 of depth)", which 0.6 meets. The same section defines "late" and "middle" as thirds of depth, and by that definition 0.6 is in the middle third. Since amendment C1 this arrival is descriptive, so no reading depends on it.

**Observation 7: the action share is carried by the object's name.** Up to 0.81 of the effect passes through the action's object word, and up to 0.23 through the rest of the action line. B7 notes that an action share carried by the object's name points to lexical matching (H6) rather than to the verb and receptacle. The minimal pilot has no band for this row. The object-free (described) goals of `rq1x` keep 61% of Qwen's effect behaviourally, so the name is not the whole story. Transfer to described goals is a full-pilot row.

**Observation 8: the removal on determined successes is uninformative as computed.** Its bootstrap SE is 5.06, against a percentile interval of [−0.93, 1.14]. The successes' own aligned-minus-swapped effect is small (1.86 nats unweighted), so some resampled denominators come near zero and the ratio is heavy-tailed. The H4 row "moves successes as much as failures" cannot be read from it.

**Observation 9: the collateral shift is large, and the success-versus-failure gap still passes.** Ablating the task direction at layer 28 raises the neutral-task readout by 1.21 nats on average. That is about a quarter of the pooled goal effect in the sample, so the ablation is not a neutral edit. The success-versus-failure gap on those prompts still keeps 0.94 of its size, because the gap is about 29 nats. The gap ratio's later-stratum lower bound, 0.900, sits exactly at the "kept" threshold. The rule reads the point estimate.

**Observation 10: provenance records.** The analysis run's `inputs.json` records no inputs (`"inputs": {}`). The link to the GPU stages is in its `config.json` (`accept_commits`) and in those stages' own `git_sha` and `jobs.jsonl`, which all show commit `7ade3b9`, job 7245822 and an NVIDIA L40S. The `effect_before` and `effect_after` fields of the ablation output are unweighted means of the sampled items, while the shares are weighted. The 4.72 nats quoted for "all" is the sample's mean, not the population's.

## 11. What the results show, what they do not, and what would change them

### 11.1 What they show

The tooling works on the real model. It reproduces the scored yes/no and first-token effects item by item. Its prefix, batch and rounding checks held, random directions do nothing, and the cross-fitted estimates agree between folds (between-fold cosines of the task direction 0.95 to 0.997 from layer 8 on).

In Qwen3-8B's yes/no format, the task's influence on the answer travels from the object's name in the task line into the action line, almost entirely into the action's own object word, by about layer 13 to 15. Through the question it reaches the answer position between layers 19 and 22. From layer 22 on, the answer position alone carries over 90% of it. For the typical item, the history after the task line plays almost no part (history share 0.031, "low"). One direction at the answer position carries most of the effect after step 0 once the effect has arrived there: 0.81 at layer 24 and 0.90 at layer 28. Removing that direction keeps 93% of the model's ability to tell working from failing actions on neutral prompts, although it shifts all neutral readouts up.

Before the effect arrives, the task direction at the answer position already shares a substantial component with the model's neutral success direction (projected cosine 0.44). The direction of merely mentioning the object is aligned with the success direction almost as much (0.335).

### 11.2 What they do not show

- **No verdict on any hypothesis.** The design forbids one here (C1, D6). The pooled and step-0 onset readings are inconclusive. The later-stratum "after arrival" is one interval bound from the opposite reading (Observation 1), and it comes from the calibration fold, which the full pilot does not use for verdicts.
- **Nothing about whether the represented state changes** (H1 against H2). The history-fact probes are a stage-1 product. The hidden-state probe is descriptive and without intervals. A low history share means that the history tokens' activations do not differ in a way that matters for the answer. It does not mean the state representation is untouched: the effect may change the state where it is read, at the action line, which this pilot does not probe with history-fact labels.
- **Nothing about H4 against H3 from the intervention rows.** They are not assessable, and Observation 3's pattern lies where the design says such alignment is expected anyway.
- **Nothing about H5 (policy blending), transfer to described goals or across formats, heads, attention and MLP blocks, the first-token or full-reply patching maps, Llama, or the base model.** The analysis lists these as not filled by the minimal pilot.
- **Not the headline effect.** The localization is of a 2.8-nat log-odds shift inside an answer that stays "No", in the yes/no format. The 6.12-nat two-candidate effect is a different readout; only its first-token term was scored here, for the reproduction gate.
- **One model, one format, one GPU type, one split, 100 patching items and 125 ablation items from one fold.** All of it is localization, not mechanism in the sense of A8.

### 11.3 What would change the picture, and what the next steps imply

The full pilot reads its verdicts on fold 1 with the planned sizes. The step-0 sizes (19 patching items, 112 ablation failures) are within what fold 1 holds (268 step-0 failures). The later-stratum ablation cannot reach the target precision even at the cap. So the onset-against-arrival row will likely stay near the boundary that Observation 1 describes, unless the true removal at layer 24 is well above 0.6 or the threshold logic changes in a new dated amendment before fold 1 is touched.

The object-mention plan wants 184 items, against about 60 available in fold 1 (E6), so that row cannot reach its target either. Three changes would alter the readings here:

- a lower bound at layer 24 at or above 0.6 on fold 1, which would make the later stratum "at or before arrival";
- a step-0 removal that stays near 0.5 even with a direction estimated on step-0 items, which would point to a different representation at step 0 rather than an estimation mismatch (Observation 2);
- assessable H4 intervention rows. Given section 6, these need either layers where answer-position edits move the readout before the arrival (none exist in this model and format) or edits at more positions than the answer position, which would be a design change.

What would *not* change them is more bootstrap resamples or rerunning the same items. The intervals are dominated by item-to-item and game-to-game variation.

## 12. Provenance

| What | Value |
|---|---|
| Design | `notes/interp_design.md`, amendments 1 to 5; frozen at tag `interp-design-frozen-2026-11-22`, commit `7ade3b98d2b26f2c7b48e5dea04589f5eac7e7fe` (branch `interp`) |
| Deviation | `notes/interp_deviation_analysis_fix.md`; tags `interp-deviation-analysis-fix-2026-11-23` (`06e3871`) and `interp-deviation-analysis-fix-2026-11-24` (`c73147e`), branch `interp-analysis-fix` |
| Model | Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`, bfloat16, thinking off; readout tokens Yes 9454, No 2753 |
| GPU stages | job 7245822 on node `atl1-1-03-004-25-0`, one NVIDIA L40S, commit `7ade3b9`, 3 h 33 min in all: `interp_pilot_stage0` started 19:53, `interp_pilot_patch` 20:34, `interp_pilot_ablate` 22:52, end 23:25 (2026-11-22, EST); study environment Python 3.11.18, torch 2.6.0+cu124, transformers 4.51.3, numpy 2.2.6 |
| Stage 0 | 28,872 records; gates passed (yes/no and first token); batch noise 0.132 nats |
| Patching | 100 items (30 step 0, 25 later with an object mention, 45 other later), 50 placebo pairs, 432 cells per direction, 300 records |
| Ablation | 125 items of fold 0 (75 failures, 50 successes), 375 records; largest realized edit error 0.122 |
| Analysis | `interp_pilot_analysis`, job 7295359 (CPU, 2 cores), node `atl1-1-01-005-3-1`, 2026-11-24 17:59 to 18:33 EST, commit `c73147e3b05c95e3eae3c33825e3e655bd9dea5e`, clean checkout `text-wm-fix41`, 2,000 game-bootstrap resamples, scikit-learn 1.9.1 from `/storage/ice1/2/1/gmat3/pylib/interp_sklearn_1.9.1` |
| Metrics | `runs/interp_pilot_analysis/metrics.json`, 1,386,852 bytes, SHA-256 `32c15ec3409f5cab6e48bbd74babc927ff741a9971ec796dbfe68c7a143ab185`; extract in `notes/interp_metrics.json` |
| Failed analysis attempts | jobs 7251609 (scikit-learn import), 7252128 (the crash of deviation 1), 7253221 (dependency commit check); no outputs |
| Tests | `pytest tests -k interp` at this branch: 105 passed (local CPU environment with scikit-learn 1.9.1, torch 2.2.2, transformers 4.51.3, numpy 1.26.4) |
