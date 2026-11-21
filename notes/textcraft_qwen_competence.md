# TextCraft: how competent is Qwen3-8B as a world model, what does it rely on, and how does its competence relate to its goal effect? (exploratory, post hoc)

Everything in this note is exploratory and post hoc. It was written on 2026-11-21, after the confirmatory TextCraft results (`notes/textcraft_results.md` on `main`), the Llama diagnostics ([textcraft_llama_dx.md](textcraft_llama_dx.md)), the Llama competence note ([textcraft_llama_competence.md](textcraft_llama_competence.md)) and the cross-environment comparison (`notes/textcraft_vs_alfworld.md` on branch `xenv-compare`, sections 4 and 9) had been read. Those notes already report Qwen3-8B's headline competence (neutral AUC 0.86), its fit to the ten heuristics, its use of the repetition cue and the across-strata correlation of −0.64. This note takes Qwen as the subject, with Llama-3.1-8B alongside. It uses only scores that already exist; nothing was scored for it. It changes none of the confirmatory verdicts: the TextCraft goal effect is confirmed for Qwen3-8B and not for Llama-3.1-8B, so the replication holds for one model only.

## Summary

All of this is exploratory and post hoc. It reads only existing scores: the confirmatory TextCraft run and the exploratory prompt-variant run. It changes no confirmatory verdict. Intervals are 95% intervals from resampling whole goal pairs.

- **Qwen3-8B is a working but imperfect TextCraft world model.** Under the neutral goal line it separates determined successes from the 1,500 main failures with AUC 0.86 [0.84, 0.88] on the two-candidate scale and 0.85 [0.82, 0.87] on yes/no. It discriminates well in every role and failure class; its weakest cell is "too few" ingredient crafts (two-candidate 0.79, yes/no 0.72), falling to 0.65 and 0.58 in histories of 9 to 12 steps. It passes the families competence rule overall and on every population a round-2 contrast reads (lowest yes/no AUC 0.76, on "too few" failures). Its ranking is good, but its probabilities are not. It calls 72% of successes and 16% of failures successes. 32% of the items it gives a success probability above 0.9 actually fail, and its middle probability bins carry no information. Its Brier score, the mean squared error of its probabilities, is only 12% better than always predicting the base rate on the two-candidate scale, and no better than the base rate on ingredient crafts or on yes/no.
- **What it relies on.** Qwen's scores follow the true inventory rule more closely than any pre-defined heuristic except "every input's name appears in the history" (tracking AUC 0.89 against 0.86 for the truth). Unlike Llama, once a heuristic is held fixed the truth keeps a large weight in Qwen's scores (+12.6 nats with the names rule, +13.9 with the gross-amount rule). Qwen also leans hard on a repetition cue: it predicts failure for 56% of successful crafts that repeat an earlier craft, against 24% of other successes. A rule that never subtracts consumption, "enough units came in and this craft has not been made before" (H2 ∧ H11, fixed before fitting), fits Qwen's scores as well as the truth overall. Among used-up failures and ingredient successes it fits better (tracking AUC 0.94 against 0.91, paired difference +0.03 [+0.01, +0.05]). By the reading fixed in advance, Qwen's high used-up AUC is therefore not evidence that it subtracts consumption. Qwen still separates used-up failures from successful repeats (AUC 0.80), so it is not using the repeat cue alone. Its errors fall in three places: "too few" crafts where it reads a count close to the need as enough (85% wrong on "craft 4 stick using 2 planks" with one plank held); successful goal crafts, 34% of which it calls failures; and successful repeats.
- **Quantities.** Qwen's score falls steadily with the number of units missing (Spearman −0.55); with one unit short it leans toward success (+1.9 nats). On used-up failures it falls with time since the consumption on the two-candidate scale only (−0.23), and it is lower when an inventory reply has shown the reduced count (−10.1 against −5.9). It ignores how many units are spare on successes (+0.03).
- **Competence and the goal effect.** Across 14 role × class × length strata, Qwen's goal effect ΔL (aligned minus swapped) is larger where its discrimination is weaker. The Spearman correlation is −0.64 [−0.77, −0.34] on the two-candidate scale and −0.30 [−0.62, −0.10] on yes/no. Within the same role × class cell, the effect also grows as Qwen's own neutral prediction becomes less confident: clearly on yes/no (from +0.97 in the most confident fifth to +5.0 in the least, −0.14 nats of effect per nat of confidence) and weakly on the two-candidate scale (−0.05 per nat). The two roles differ. On goal crafts the two-candidate effect is zero where Qwen is surest (−0.04 among 328 items below −15 nats) and +4.5 to +6.0 where it is unsure. On ingredient crafts it is a roughly constant +0.8 nats at every confidence level.
- **Measurable effect, not a broad distortion.** The goal effect is statistically clear at every level of confidence on yes/no and on ingredients. Measured by decisions, though, it barely touches confident, correct predictions. Among the 70% of failures Qwen confidently and correctly calls failures (neutral probability below 1%), the aligned line produces a success prediction for 3.4% and the swapped line for 0.9%; the mean probability shift is under half a point among the surest. Relative to the neutral line, the aligned line leaves Qwen's discrimination unchanged on both scales (AUC difference +0.00 and −0.00). The swapped line lowers it (−0.02, and −0.11 on yes/no, where it pushes successful goal crafts toward "No" by 18 nats). So the confirmed ΔL is a real, measurable sensitivity to the task line that reaches even settled predictions in log-odds. Its practical weight, though, sits where Qwen is unsure or wrong. That fits "the task line is used as evidence where the history leaves the model unsure" better than "the goal overrides a working state judgement". It is not a clean fit to either, because a small uniform shift remains on settled predictions.
- **Inventory shown.** With the inventory printed before the action, Qwen's AUC rises from 0.84 to 0.94 on the 500 diagnostic items and its mean effect does not change (+1.12 against +1.15; paired −0.04 [−0.45, +0.38]). The unchanged mean hides a redistribution. 81% of failures become confident and correct (from 71%), and on them the effect is +0.39 [−0.07, +0.86]; on the 19% still unsure or wrong it is +4.8 and +3.9. The goal changes decisions on 1.7% of confident, correct failures (swapped 0.4%). Under the clean system text, with or without the inventory, the confident, correct items do keep a shift of +0.85 to +0.92 nats with intervals above zero. So the goal still moves settled predictions a little in log-odds, but almost never across the decision boundary.
- **Implications.** (a) Qwen's confirmed TextCraft effect can be described as a goal sensitivity in a competent predictor, and it is not an artefact of incompetence. "Distortion of a working world model" overstates it unless qualified: its consequences concentrate where the model is unsure, and the model's competence itself rests partly on cues (names, repetition) rather than on inventory arithmetic. (b) The cross-model statement "an effect requires competence" survives as a statement about models. Within Qwen, the effect is largest where competence is lowest, and Llama's uncertain items show no positive effect at all. So uncertainty is not sufficient either. (c) Qwen passes every contrast-level gate with margin on round-1 numbers, so contrast-specific gates would not change its status. Llama would pass only on goal-role populations. Every round-2 contrast except the parent penalty compares goal with ingredient items, so Llama would still be gated out of Λ, Dn, Dc and T. A confidence-stratified readout, fixed before round-2 scores, would carry this note's main finding into round 2. (d) The paper should report Qwen's effect as confirmed and real, say that it comes from a competent but imperfect and overconfident predictor, and say that its decision-level consequences are concentrated on actions the model was unsure about.

## What this note asks

The confirmatory TextCraft test found that naming the action's goal in the task line (aligned) rather than the other displayed goal (swapped) raises Qwen3-8B's predicted success for crafts whose failure the history already settles. The estimand is

$$\Delta L = \overline{L_{\text{aligned}} - L_{\text{swapped}}},$$

the mean over the 1,500 main failures of the difference between the two task lines in L, the log-odds of the success reply against the failure reply. It was +0.94 nats [+0.63, +1.24], confirmed. A world model that reads the history correctly gives ΔL = 0. The question here is what kind of predictor produced that number. Is Qwen a competent TextCraft world model whose working state judgement the goal line distorts? Or is it an uncertain one that uses the task line as one more piece of evidence where the history leaves it unsure? Six questions, as in the brief:

1. Qwen's full competence profile, on both reply scales, by failure class, role and history length; its Yes rates, calibration and Brier score; and whether it passes the families competence rule overall, per role and on the population each round-2 contrast reads.
2. What Qwen relies on: the pre-defined heuristics, the repetition cue, and where its errors fall.
3. Whether its scores respond to the quantities that decide the outcome.
4. How its goal effect relates to its competence, stratum by stratum and item by item, with a measurable goal effect kept apart from a distortion of a competent predictor.
5. The same with the inventory shown, where its AUC reaches 0.94.
6. What all this implies for reading the confirmed effect, for the cross-model pattern, for round 2's gates and for the paper.

Llama-3.1-8B appears alongside as the contrast case. Its own profile is in the Llama note.

## Heuristics and analysis plan (fixed 2026-11-21, before any of the new fits below was computed)

### The heuristics reused

H1 to H10 and the combination H2 ∧ H9 are used exactly as defined in `notes/textcraft_llama_competence.md` (definitions committed in `9a83738`, before any heuristic was fitted to either model) and as implemented in `scripts/textcraft_llama_competence.py` (`features`, committed in `6522784`). Nothing about them is redefined. Their fits to Qwen were first computed and reported in the Llama note; this note recomputes them with the same code and reads them with Qwen as the subject.

### Three heuristics added for this note

What was known about Qwen when these were written: the Llama note's post hoc repetition check found that 96.6% of used-up failures repeat a craft that succeeded earlier in the same history, against 14% of successes; that Qwen scores repeated ingredient successes far lower than other ingredient successes (mean L +0.14 against +14.9); and that among repeats only, Qwen still separates used-up failures from successes (AUC 0.80). That check was itself added after the H1–H10 fits had been read. So the three heuristics below are motivated by a cue already seen to matter for Qwen; what has not been computed is any fit of them, alone or combined, to either model. They are fixed here, before that.

- **H11, a repeated craft fails.** Failure when the action is identical, character for character, to an earlier history action whose reply was `Crafted ...` (the script's `posthoc_repeat`); success otherwise. It reads no counts and no names.
- **H2 ∧ H11.** Success when H2 predicts success (every input obtained in at least the needed amount in gross, ignoring consumption) and the action is not a repeat. It is a shortcut that could reproduce good used-up discrimination without any subtraction: "enough came in, and I have not already made this".
- **H6 ∧ H11.** Success when the true accounting predicts success and the action is not a repeat. It describes a model that tracks the state and in addition treats any repeat as exhausted. Its validity is below 1 because 14% of successes are repeats.

These are fitted with the same measures as H1–H10 (validity, tracking AUC, kappa, within-truth tracking, joint regression). The reading fixed now: if Qwen's tracking AUC for H2 ∧ H11 is at or above its tracking AUC for the truth (H6) on used-up failures plus successes, Qwen's used-up competence is no evidence of subtraction, because a rule without subtraction fits as well. If the truth fits clearly better (paired difference with an interval below zero for H2 ∧ H11 minus H6), Qwen tracks consumption beyond the repeat cue. A three-term regression of L on the truth, H2 and H11 apportions the two.

### The new analyses (planned now; the questions are those of the brief)

1. **Competence profile.** As in the Llama note (role × class × history-length AUC on both scales, Yes rates, balanced accuracy, binned calibration, Brier), with three additions: Yes rates by role × class; the Brier score against the base-rate predictor within each role; and the families competence rule (yes/no scale, neutral line, AUC at least 0.70, Yes on at most 95% of failures and at least 5% of successes, median P(Yes) + P(No) at least 0.5) applied per role, per class and on the population each round-2 contrast reads (ingredient failures for the intermediate lift; goal and ingredient failures for the naming and completion contrasts; failures with 1–8 history steps, by role, for the timing contrast). The parent records have no round-1 analogue and are not gated here.
2. **What Qwen relies on.** All heuristic fits; where Qwen's errors fall (by role, class, history length, repeat status, surplus, generic inputs, output item), and for each error group which heuristics agree with Qwen's wrong decision; the repetition cue within ingredient crafts, with surplus held fixed.
3. **Quantities.** The Llama note's analyses (too-few shortfall, used-up recency and visibility, success surplus), with Qwen as the subject, by role where the cells allow.
4. **Competence and the goal effect.** ΔL (aligned minus swapped) by stratum with a pair-bootstrap interval for the across-strata Spearman correlation; ΔL and the probability shift by bins of Qwen's neutral log-odds, overall and within role × class cells; the goal's effect on decisions (share of failures predicted success under aligned, swapped and neutral; flips of correct, confident neutral predictions; balanced accuracy and AUC under each line). The distinction drawn throughout: a *statistically measurable goal effect* is a mean shift in log-odds whose interval excludes zero; a *distortion of a competent predictor* is a shift that turns correct, confident predictions into wrong ones, or lowers discrimination. The reading fixed now: if the log-odds effect is concentrated where Qwen's neutral prediction is uncertain and near zero where it is confident and correct, the data fit "the task line is used as evidence where the history leaves the model unsure"; if it is as large in log-odds among confident, correct predictions, the goal moves settled judgements, and whether that matters depends on how many decisions it changes.
5. **Inventory shown.** The same competence-by-effect analysis on the 500 items of the inventory-shown prompt (and, alongside, the clean and clean + inventory prompts), against the verified prompt on the same items.
6. **Implications** for reading Qwen's effect as a distortion of a working world model, for the cross-model pattern, for round 2's gates, and for the paper.

**What was added while computing (recorded, not hidden).** Three things were added after the plan above was committed, all before the corresponding number was read:

- The paired tracking-AUC difference between H2 ∧ H11 and the truth on the used-up subsets. The reading above needs it, and the reused code computes paired differences only over all items.
- The goal effect by the deciding quantities (shortfall, time since the input was used up, surplus).
- The overall aligned-minus-neutral and swapped-minus-neutral shifts.

Two cuts in the plan turned out empty or uninformative:

- *Generic inputs.* No action in the item set names a generic input such as "planks"; every action names a specific kind. That cut is therefore empty and is not reported.
- *A same-kind item held.* One check was added after the error tables had been read: whether never-obtained failures that Qwen calls successes have a same-kind item in the inventory (for example oak planks held when crimson planks are needed). Such items are not more often wrong: 3.6% against 8.5% on the two-candidate scale, and 10.9% against 10.6% on yes/no. That check is not reported further.

## Data and method

**Scores.** The confirmatory scoring run `textcraft_score` (51,200 records, ICE jobs 7204417–7204420 at `5a3702a`) and the exploratory prompt-variant run `textcraft_dx_prompts` (ICE job 7207018 at `f74c262`, checkout `text-wm-fix27`; 9,000 two-candidate records on 500 items under the clean, inventory and clean + inventory prompts). Both were read from local copies. On 2026-11-21 the SHA-256 hashes of the local copies were checked against the originals on ICE (`text-wm-fix24/runs/textcraft_score/shard_0{0..3}_of_04/scoring.jsonl` and `text-wm-fix27/runs/textcraft_dx_prompts/scoring.jsonl`), and all five match. The items file (`wf/textcraft/data/items.jsonl`) and the frozen recipe database match the manifest's hashes. The recipe file is git-ignored, was copied in for completeness, and is not read by the analysis. The analysis asserts that the text-only checker's accounting reproduces every item's stored outcome. Rerunning the Llama note's analyses with the same code reproduces every one of its Qwen and Llama figures to four decimals.

**Populations, scales, intervals.** As in the Llama note:

- *Items.* The 1,500 main failures (500 used up, 500 too few, 500 never obtained) and the 600 determined successes (300 per role), from 247 goal pairs.
- *Two-candidate scale.* L is the log-odds of the success reply against the failure reply.
- *Yes/no scale.* L is the log-odds of "Yes" against "No" to "Will this action succeed?".
- *Decision.* A model "says success" when its L is above zero.
- *Intervals.* Every interval is a 95% percentile interval from 2,000 resamples of whole goal pairs. One set of resampling weights is shared by every statistic, so paired differences use the same resamples. There are no p-values.
- *Many cuts.* Many cuts were computed, so a single interval that excludes zero is weak evidence on its own. The readings rest on patterns that recur across scales, roles and prompts.

**Terms.**

- *AUC* (area under the ROC curve): the chance that a randomly chosen success scores above a randomly chosen failure. 0.5 is chance and 1 is perfect.
- *Role.* "Goal" crafts make one of the pair's two final items. "Ingredient" crafts make an intermediate item.
- *Confident.* A prediction is "confident" when the neutral L is beyond ±4.6 nats, which is a probability below 1% or above 99%.
- *Aligned − neutral and swapped − neutral.* The shift of each task line against the neutral line ("Goal: explore the crafting commands."), which names no item.

## 1. The competence profile

### Discrimination

Each failure class is compared with the successes of the same role; "all" rows compare every failure with every success.

| Neutral line, AUC | Qwen two-candidate | Qwen yes/no | Llama two-candidate | Llama yes/no |
|---|---|---|---|---|
| All main failures vs all successes | 0.86 [0.84, 0.88] | 0.85 [0.82, 0.87] | 0.61 [0.58, 0.64] | 0.61 [0.58, 0.65] |
| Goal role | 0.90 [0.87, 0.92] | 0.87 [0.84, 0.90] | 0.84 [0.81, 0.87] | 0.85 [0.82, 0.88] |
| Goal × never obtained (252) | 0.97 [0.95, 0.99] | 0.95 [0.93, 0.97] | 0.90 | 0.91 |
| Goal × too few (252) | 0.82 [0.78, 0.86] | 0.80 [0.76, 0.84] | 0.78 | 0.80 |
| Ingredient role | 0.88 [0.86, 0.91] | 0.85 [0.82, 0.87] | 0.58 [0.54, 0.62] | 0.55 [0.51, 0.59] |
| Ingredient × never obtained (248) | 0.93 [0.90, 0.95] | 0.88 [0.85, 0.91] | 0.86 | 0.81 |
| Ingredient × too few (248) | 0.79 [0.74, 0.83] | 0.72 [0.67, 0.77] | 0.60 | 0.54 |
| Ingredient × used up (494), role-matched | 0.91 [0.88, 0.93] | 0.89 [0.86, 0.92] | 0.43 | 0.42 |
| History 1–3 steps | 0.92 [0.89, 0.95] | 0.90 [0.87, 0.93] | 0.79 | 0.74 |
| History 4–8 steps | 0.84 [0.81, 0.88] | 0.83 [0.80, 0.86] | 0.54 | 0.56 |
| History 9–12 steps | 0.83 [0.79, 0.86] | 0.82 [0.78, 0.85] | 0.55 | 0.56 |

Qwen discriminates well everywhere. Its weakness is counting near the threshold in long histories. Within "too few" ingredient crafts its two-candidate AUC falls from 0.92 at 1 to 3 steps to 0.76 at 4 to 8 and 0.65 at 9 to 12 (yes/no 0.85, 0.69, 0.58). Goal "too few" falls from 0.90 to 0.81 and 0.77. Never-obtained and used-up crafts stay at 0.84 to 0.98 in every length bin. Unlike Llama, Qwen ranks used-up failures well below ingredient successes (0.91), and it is about as good on ingredients as on goal crafts.

### How often Qwen says success

| Neutral line, share answered success | Qwen two-candidate | Qwen yes/no | Llama two-candidate | Llama yes/no |
|---|---|---|---|---|
| All failures | 16% [14, 19] | 19% [16, 21] | 39% | 9.9% |
| Goal × never obtained | 1.6% | 2.8% | — | — |
| Goal × too few | 20% | 32% | — | — |
| Ingredient × never obtained | 15% | 19% | — | — |
| Ingredient × too few | 43% | 46% | — | — |
| Ingredient × used up | 9.9% | 6.5% | 82% (all used up) | 17% |
| All successes | 71% [67, 76] | 72% [69, 77] | 46% | 11% |
| Goal-role successes | 66% | 73% | 24% | 6% |
| Ingredient successes, 1–3 / 4–8 / 9–12 steps | 90% / 75% / 66% | 91% / 63% / 62% | — | — |

Qwen's balanced accuracy (the mean of its accuracy on failures and on successes) is 0.78 on the two-candidate scale and 0.77 on yes/no, against 0.53 and 0.51 for Llama. Qwen's mean neutral L is −9.8 on failures and +7.3 on successes (yes/no −12.4 and +6.3). It errs in both directions. It misses a third of goal-craft successes and, in long histories, a third of ingredient successes. It calls nearly half of "too few" ingredient failures successes.

### Calibration and Brier score

Calibration asks whether, among items given a predicted success probability near p, a share near p actually succeeds. The success share among the 2,100 items is 0.29, fixed by the design. The Brier score is the mean squared difference between the predicted probability and the outcome (0 or 1):

$$B = \frac{1}{n}\sum_{i=1}^{n} \left(\sigma(L_i) - y_i\right)^2,$$

where σ is the logistic function and $y_i$ is 1 for a success. A predictor that always gives the base rate $\bar y$ scores $\bar y(1-\bar y)$. The skill score is one minus the model's Brier divided by that base-rate Brier: 0 means no better than the base rate, and 1 is perfect.

| Predicted P(success), neutral | Qwen two-candidate: observed share (items) | Qwen yes/no | Llama two-candidate |
|---|---|---|---|
| below 0.1 | 0.11 (1,322) | 0.11 (1,312) | 0.19 (905) |
| 0.1 to 0.5 | 0.28 (102) | 0.29 (68) | 0.46 (333) |
| 0.5 to 0.9 | 0.31 (87) | 0.32 (65) | 0.31 (389) |
| 0.9 and above | 0.68 (589) | 0.64 (655) | 0.32 (473) |

Qwen's observed share rises from the bottom group to the top group, which Llama's does not. But Qwen puts 91% of items in the two extreme groups, and both extremes are overconfident. 11% of the items it gives less than 10% succeed. 32% of those it gives more than 90% fail (36% on yes/no). Its 189 middle items are flat at about 0.3, the base rate: when Qwen is unsure, its probability says nothing. The expected calibration error, the item-weighted mean gap between predicted and observed shares, is 0.17 (yes/no 0.19).

| Brier score (skill against the base rate) | Qwen two-candidate | Qwen yes/no | Llama two-candidate | Llama yes/no |
|---|---|---|---|---|
| All items (base-rate Brier 0.20) | 0.18; skill +0.12 [+0.01, +0.22] | 0.20; skill +0.03 [−0.09, +0.13] | 0.34; skill −0.69 | 0.26; skill −0.26 |
| Goal role (base-rate Brier 0.23) | 0.18; skill +0.23 [+0.11, +0.35] | 0.20; skill +0.15 [+0.01, +0.29] | 0.24; skill −0.03 | 0.30; skill −0.29 |
| Ingredient role (base-rate Brier 0.18) | 0.18; skill −0.00 [−0.20, +0.17] | 0.20; skill −0.12 [−0.31, +0.05] | 0.41; skill −1.30 | 0.23; skill −0.29 |

So Qwen's ranking is good (AUC 0.86) and its probabilities are poor. On ingredient crafts, and overall on yes/no, its probabilities are no better in squared error than the base rate, because the confident errors cost as much as the confident hits gain. This matters for reading effect sizes in nats. A 1-nat shift on an item at −20 nats changes nothing, while a model this overconfident has many items where it is confidently wrong.

### The families competence rule, overall, per role and per round-2 contrast population

The rule (`notes/families_metrics.json`, `rules.competence`; round 2's gate, `notes/textcraft_r2_design.md` section 3 on branch `textcraft-r2`) reads the yes/no scale under the neutral line. It requires all four of:

- an AUC of at least 0.70;
- Yes on at most 95% of failures;
- Yes on at least 5% of successes;
- a median P(Yes) + P(No) of at least 0.5.

As committed (`f8dd14e`), round 2 applies it once, to all 2,100 main items, and reports role-level competence only descriptively. The table below applies the same rule to each population a contrast reads. It shows what contrast-specific gates would give on round-1 data under the verified prompt.

| Population (failures, successes) | Qwen yes/no AUC | Qwen Yes on failures / successes | Qwen | Llama yes/no AUC | Llama |
|---|---|---|---|---|---|
| All (the round-2 gate) (1,500, 600) | 0.85 [0.82, 0.87] | 19% / 73% | passes | 0.61 [0.58, 0.65] | fails |
| Λ, intermediate lift: ingredient (990, 300) | 0.85 [0.82, 0.87] | 19% / 72% | passes | 0.55 [0.51, 0.59] | fails |
| Dn and Dc, goal part: goal without used-up (504, 300) | 0.87 [0.84, 0.90] | 17% / 73% | passes | 0.85 [0.83, 0.88] | passes |
| T, timing, goal part: 1–8 steps (338, 200) | 0.89 [0.86, 0.92] | 12% / 68% | passes | 0.85 [0.81, 0.88] | passes |
| T, timing, ingredient part: 1–8 steps (582, 200) | 0.88 [0.84, 0.91] | 19% / 77% | passes | 0.57 [0.52, 0.62] | fails |
| Used up vs ingredient successes (494, 300) | 0.89 [0.86, 0.92] | 6.5% / 72% | passes | 0.42 [0.38, 0.47] | fails |
| Too few vs all successes (500, 600) | 0.76 [0.72, 0.79] | 39% / 73% | passes | 0.66 [0.60, 0.71] | fails |
| Never obtained vs all successes (500, 600) | 0.91 [0.89, 0.93] | 11% / 73% | passes | 0.84 [0.80, 0.87] | passes |
| History 9–12 steps (580, 200) | 0.82 [0.78, 0.85] | 22% / 73% | passes | 0.56 [0.52, 0.61] | fails |

The median yes/no mass is 1.00 for Qwen in every population. Qwen passes every population with a margin; its lowest AUC is 0.76 on "too few" (lower interval end 0.72). The parent records of contrast P have no round-1 analogue and cannot be gated here. In ALFWorld, by comparison, Qwen's yes/no AUC is 0.99 (Llama note, section 5), so TextCraft costs it 0.14 of AUC but leaves it well inside the rule.

## 2. What Qwen relies on

### The pre-defined heuristics

For a binary heuristic h, the tracking AUC is the AUC of the model's neutral L between the items h calls a success and those it calls a failure. For the truth (H6) it equals the competence AUC. Kappa is Cohen's agreement between the model's decision and the heuristic, corrected for chance. "Minus truth" is the paired difference in tracking AUC against H6. H11 and the two combinations with it were fixed in this note before any fit.

| Heuristic | Validity (AUC against truth) | Qwen tracking, two-candidate | minus truth | Qwen kappa | Qwen tracking, yes/no | Llama tracking, two-candidate |
|---|---|---|---|---|---|---|
| H1 name seen | 0.70 | 0.89 [0.87, 0.91] | +0.03 [+0.00, +0.06] | 0.26 | 0.81 | 0.87 |
| H2 gross obtained (ignores consumption) | 0.84 | 0.79 [0.77, 0.82] | −0.07 [−0.09, −0.05] | 0.24 | 0.73 | 0.82 |
| H3 obtained in last 3 steps | 0.74 | 0.73 | −0.13 | 0.33 | 0.69 | 0.59 |
| H4 mentioned in last 3 steps | 0.69 | 0.71 | −0.15 | 0.20 | 0.64 | 0.66 |
| H5 history at least 4 steps | 0.45 | 0.54 | −0.32 | −0.04 | 0.55 | 0.59 |
| H6 true accounting | 1.00 | 0.86 [0.84, 0.88] | (the truth) | 0.53 | 0.85 | 0.61 |
| H7 held at all (ignores counts) | 0.84 | 0.83 | −0.03 [−0.06, −0.01] | 0.48 | 0.80 | 0.60 |
| H8 last event is a gain | 0.81 | 0.81 | −0.05 [−0.08, −0.02] | 0.51 | 0.81 | 0.56 |
| H9 not a top item (goal crafts fail) | 0.42 | 0.60 | −0.26 | 0.01 | 0.56 | 0.80 |
| H2 ∧ H9 | 0.59 | 0.69 | −0.18 | 0.05 | 0.60 | 0.85 |
| H11 a repeated craft fails (new) | 0.59 | 0.47 | −0.39 | 0.16 | 0.56 | 0.18 |
| H2 ∧ H11 (new) | 0.93 | 0.86 [0.84, 0.89] | +0.00 [−0.01, +0.01] | 0.52 | 0.87 (+0.02 [+0.01, +0.03]) | 0.58 |
| H6 ∧ H11 (new) | 0.93 | 0.86 [0.84, 0.89] | +0.00 [−0.01, +0.01] | 0.53 | 0.87 (+0.02 [+0.01, +0.03]) | 0.58 |

Read by the criterion fixed in the Llama note, the highest tracking AUC over all items, the best fit for Qwen is again H1, "every input's name appears in the history" (0.89). Its margin over the truth is small (+0.03 [+0.00, +0.06]), and on yes/no the truth fits better (H1 minus truth −0.04 [−0.07, −0.00]). By kappa, the truth and the two new combinations agree best with Qwen's decisions (0.52 to 0.53). H8 is close (0.51; 0.57 on yes/no, where it is the best of all). H1 tracks Qwen well because a third of the failures are never-obtained items, whose input name never appears, and Qwen gets those right. It is not evidence that Qwen ignores counts. On the H1 "success" side, which holds every success, every used-up failure and most too-few failures, Qwen still separates real successes from failures. The joint regression below measures how much.

**The truth keeps its weight once a heuristic is known.** The joint regression fits

$$L_i = \beta_0 + \beta_{\text{truth}}\, y_i + \beta_h\, h_i + \varepsilon_i,$$

where $y_i$ is 1 for a real success and $h_i$ is 1 when the heuristic predicts success. For Qwen the truth slope is +12.6 [+11.3, +13.8] nats with H1 (heuristic +11.0), +13.9 [+12.6, +15.2] with H2 (heuristic +4.8) and +11.9 [+10.4, +13.3] with H8. For Llama it is −0.55, −2.18 and +1.65. Qwen uses both the names and the state, and the state dominates once counts enter. Llama uses the names and gross amounts and not the state.

### The repetition cue and whether Qwen subtracts consumption

96.6% of used-up failures exactly repeat a craft that succeeded earlier in the same history. So do 14% of successes (86 ingredient crafts, no goal crafts). No too-few or never-obtained failure does. A model could therefore score used-up failures low without any arithmetic, by treating "I already made this" as "the inputs are gone".

Qwen uses that cue heavily. Its mean neutral L on ingredient successes is +14.9 [+13.6, +16.2] for non-repeats and +0.14 [−1.21, +1.54] for repeats. It says failure to 56% of repeated successes and 24% of other successes on the two-candidate scale, and to 72% against 20% on yes/no. Repeats are 14% of successes but 28% of the successes Qwen gets wrong (38% on yes/no). Llama shows no such cue: +1.04 against +0.93, and it scores repeated used-up failures *higher* than other failures (H11 tracking 0.18).

The reading fixed before the fit compares H2 ∧ H11 ("enough came in, and this has not been made before") with the truth on used-up failures and successes. If the rule without subtraction fits at least as well, Qwen's used-up competence is no evidence of subtraction.

| Used-up failures and successes, tracking AUC | Two-candidate | Yes/no |
|---|---|---|
| Truth (H6), all successes (1,100 items) | 0.85 [0.82, 0.88] | 0.87 [0.84, 0.90] |
| H2 ∧ H11 minus truth, all successes | −0.01 [−0.03, +0.00] | +0.01 [−0.00, +0.03] |
| Truth (H6), ingredient successes only (794) | 0.91 [0.88, 0.93] | 0.89 [0.86, 0.92] |
| H2 ∧ H11 minus truth, ingredient successes | +0.03 [+0.01, +0.05] | +0.05 [+0.03, +0.07] |
| H11 alone minus truth, ingredient successes | +0.01 [−0.02, +0.04] | +0.04 [+0.01, +0.06] |

The rule without subtraction ties the truth with all successes and beats it in the role-matched comparison, on both scales. By the reading fixed in advance, **Qwen's high used-up AUC is not evidence that it subtracts what crafts consumed.** The two rules disagree on about 90 items, mostly successful repeats (truth says success, the rule says failure). Qwen's scores side with the rule on them more often than with the truth.

That does not make Qwen a pure repeat detector. Three observations show more than the cue:

- *It separates within repeats.* Comparing only used-up failures and successes that are both repeats, its AUC is 0.80 [0.74, 0.85], against 0.35 for Llama.
- *It reads the surplus among repeats.* Successful repeats with spare units score +2.1, against −2.7 for those with exactly enough. Used-up repeats score −6.7.
- *The truth keeps a slope given both cues.* In a three-term regression of L on the truth, H2 and H11, the truth's slope is +7.2 [+5.7, +8.6], next to H2 +12.6 and H11 +8.0. On yes/no these are +7.2, +13.8 and +12.7. On ingredients alone the cue's weight (+14.2) is about twice the truth's (+7.3).

What the data cannot settle is whether the within-repeat separation comes from subtraction or from another cue that needs none. One candidate is H8, "the input's last event was a gain", which a replenished repeat satisfies. H8 tracks Qwen on used-up failures and successes as well as the truth does (0.84 against 0.85 two-candidate; 0.87 and 0.87 yes/no). No combination of H8 with H11 was fixed in advance, so none is fitted here.

### Where Qwen's errors fall

On the two-candidate scale, Qwen is wrong on 418 of 2,100 items (19.9%): 247 failures called successes and 171 successes called failures. Yes/no gives 280 and 165.

| Group | Items | Qwen wrong (two-candidate) | Of all errors |
|---|---|---|---|
| Ingredient × too few | 248 | 43% (107) | 26% |
| Goal × too few | 252 | 20% (50) | 12% |
| Ingredient × used up | 494 | 9.9% (49) | 12% |
| Ingredient × never obtained | 248 | 15% (36) | 9% |
| Goal × never obtained | 252 | 1.6% (4) | 1% |
| Goal-role successes | 300 | 34% (102) | 24% |
| Ingredient successes, repeats | 86 | 56% (48) | 11% |
| Ingredient successes, not repeats | 214 | 10% (21) | 5% |

Three error sources stand out:

1. *Counting near the need, worse in long histories.* "Too few" errors rise with length: ingredient 33%, 45% and 51% at 1–3, 4–8 and 9–12 steps; goal 5%, 13% and 42%. On these errors, H1 (names seen) and H7 (held at all) make the same wrong call far more often than on the failures Qwen gets right: H1 83% against 54%, H7 69% against 25%. The errors are items where every input is held but in too small a number. The single largest item-level cluster is "craft 4 stick using 2 X planks" with one plank held: 41 of 48 such items are called successes. Overall, stick crafts hold 16% of the errors and 8.5% of the items.
2. *Successful goal crafts.* Qwen calls a third of working goal crafts failures, at every history length (32%, 41%, 29%). H9 ("goal crafts fail") agrees with 60% of these errors against 46% of the goal crafts Qwen gets right. So a goal-craft prior is part of the picture, much weaker than Llama's.
3. *Successful repeats*, from the repetition cue above. H11 agrees with 28% of Qwen's false failures against 9% of its correct successes.

Errors do not cluster by goal pair beyond what chance predicts: the variance of per-pair error rates is 0.98 times the binomial expectation. They are mildly concentrated by output item: the most error-prone tenth of the 399 output items holds 65% of errors and 52% of items.

## 3. Does Qwen's score respond to the quantities that decide the outcome?

**Too few** (500 failures). Qwen's L falls steadily with the number of units missing (Spearman −0.55 [−0.62, −0.47]; yes/no −0.53). The fall is steep on ingredients (−0.69) and weaker on goal crafts (−0.33). By shortfall, the mean L is +1.9 [−0.5, +4.0] at one unit short, −6.9 at two or three, −12.8 at four to seven and −16.4 at eight or more (yes/no +2.2, −7.1, −16.2, −22.6). Qwen compares the amount held with the amount needed, but at one unit short it leans toward success. That is the "stick" error above.

**Used up** (500 failures). On the two-candidate scale Qwen's L becomes more negative the longer ago the input fell below the need: −5.3 when it fell at the last step, −6.8 at one or two steps, −7.5 at three to five and −9.8 at six or more (Spearman −0.23 [−0.31, −0.15]). On yes/no it is flat (−11.4, −12.1, −11.0, −11.9; Spearman +0.01). An inventory reply after the consumption, showing the reduced count, makes it more confident: −10.1 against −5.9 (yes/no −14.4 against −10.9). Its L does not follow how large the shortfall is (Spearman +0.06). Qwen recognizes used-up failures as failures at every distance, and recency and visible inventory make it surer. Its scores do not track the arithmetic of how much is left.

**Successes** (600). Qwen's L does not depend on the surplus in units (Spearman +0.03 [−0.06, +0.12]): +6.6 when exactly enough is held, +7.6, +7.0 and +8.7 with more spare. It rises a little with the surplus as a ratio of the need (+0.22). What lowers it sharply is an earlier consumption of an input: +0.2 against +8.4 (yes/no −4.7 against +8.2). That is the repetition cue again: the 87 successes with an earlier consumption are almost exactly the 86 repeats. Llama's L, by contrast, *fell* with surplus.

## 4. Competence and the goal effect

### By stratum

| Cell (failures) | Qwen AUC | Qwen ΔL two-candidate | aligned − neutral | swapped − neutral | Qwen ΔL yes/no | Llama ΔL two-candidate |
|---|---|---|---|---|---|---|
| Goal × never obtained (252) | 0.97 | −0.45 [−0.89, −0.07] | −1.21 | −0.76 | +1.73 [+1.17, +2.24] | −0.75 |
| Goal × too few (252) | 0.82 | +2.52 [+1.60, +3.50] | +0.26 | −2.26 | +8.29 [+7.11, +9.45] | −0.52 |
| Ingredient × never obtained (248) | 0.93 | +1.64 [+1.05, +2.25] | +2.13 | +0.49 | +3.17 [+2.35, +4.04] | −0.11 |
| Ingredient × too few (248) | 0.79 | +2.28 [+1.25, +3.41] | +2.58 | +0.30 | +3.23 [+2.28, +4.14] | +0.27 |
| Ingredient × used up (494) | 0.91 | −0.20 [−0.56, +0.17] | −0.20 | +0.00 | +0.53 [+0.29, +0.78] | −0.02 |
| All failures (1,500) | 0.86 | +0.94 [+0.63, +1.24] | +0.55 [+0.23, +0.87] | −0.39 [−0.67, −0.09] | +2.92 [+2.55, +3.29] | −0.19 |
| All successes (600) | — | +2.61 [+1.82, +3.42] | +1.55 | −1.06 | +9.60 [+8.16, +11.01] | +0.08 |

Over the 14 role × class × length strata with at least 20 failures, the Spearman correlation between Qwen's stratum AUC and its stratum ΔL is −0.64 [−0.77, −0.34] on the two-candidate scale and −0.30 [−0.62, −0.10] on yes/no. Here the stratum AUCs and ΔLs are recomputed in every resample. The effect is larger where Qwen discriminates less well. On the two-candidate scale the correlation runs through the aligned line (AUC against aligned − neutral −0.43); on yes/no it runs through the swapped line (against swapped − neutral +0.53). The strata differ in role and class as well as in competence, so the correlation alone cannot tell competence apart from the type of failure. The item-level analyses below hold the cell fixed.

### By Qwen's own confidence, item by item

The question is whether the goal acts where Qwen's own prediction is unsure, or equally on predictions it holds firmly. Failures were grouped by their neutral L, which was scored separately from the two task lines. The table gives, per group, ΔL, the mean shift in predicted probability between the two lines (percentage points), and the share each line calls a success.

| Failures, neutral two-candidate L | Items | ΔL | Probability shift, points | Says success: aligned / swapped |
|---|---|---|---|---|
| below −15 (sure of failure) | 581 | +0.34 [+0.01, +0.70] | +0.4 | 0.5% / 0.0% |
| −15 to −10 | 207 | +0.73 [−0.02, +1.53] | +2.1 | 4.3% / 1.4% |
| −10 to −4.6 | 264 | +1.38 [+0.73, +2.02] | +5.7 | 9.1% / 2.3% |
| −4.6 to 0 (unsure, correct) | 201 | +0.88 [+0.22, +1.52] | +5.2 | 22% / 16% |
| above 0 (wrong under neutral) | 247 | +2.10 [+1.30, +2.88] | +8.7 | 83% / 75% |

| Failures, neutral yes/no L | Items | ΔL yes/no | Probability shift, points | Says Yes: aligned / swapped |
|---|---|---|---|---|
| below −15 | 744 | +1.86 [+1.51, +2.24] | +0.6 | 0.5% / 0.0% |
| −15 to −10 | 202 | +1.67 [+1.06, +2.37] | +1.3 | 1.5% / 0.0% |
| −10 to −4.6 | 160 | +3.26 [+2.25, +4.32] | +3.4 | 3.8% / 0.0% |
| −4.6 to 0 | 112 | +3.20 [+1.81, +4.62] | +9.4 | 13% / 3.6% |
| above 0 (wrong) | 282 | +6.32 [+5.12, +7.57] | +27.6 | 72% / 43% |

Groups defined across all failures mix cells: the surest group is mostly goal × never obtained, where the effect is negative. To hold the cell fixed, failures were split into fifths of neutral L *within* each role × class cell, and the fifths were pooled across cells.

| Fifth within cell (1 = surest of failure) | Mean neutral L, two-candidate | ΔL two-candidate | Mean neutral L, yes/no | ΔL yes/no |
|---|---|---|---|---|
| 1 | −19.2 | +0.17 [−0.27, +0.70] | −25.7 | +0.97 [+0.66, +1.34] |
| 2 | −14.9 | +0.86 [+0.33, +1.44] | −20.0 | +2.09 [+1.55, +2.66] |
| 3 | −11.0 | +0.68 [+0.18, +1.20] | −13.8 | +2.99 [+2.27, +3.75] |
| 4 | −6.2 | +2.04 [+1.30, +2.75] | −6.1 | +3.62 [+2.76, +4.49] |
| 5 | +2.3 (64% wrong) | +0.93 [+0.28, +1.60] | +3.6 (63% wrong) | +5.02 [+3.99, +5.97] |
| Fifth 5 minus fifth 1 | | +0.76 [−0.11, +1.64] | | +4.05 [+3.01, +5.00] |
| Slope of ΔL per nat of correct-side margin, cell fixed effects | | −0.05 [−0.09, −0.00] | | −0.14 [−0.17, −0.10] |

The two roles behave differently on the two-candidate scale:

| Failures, neutral two-candidate L | Goal role: items, ΔL | Ingredient role: items, ΔL |
|---|---|---|
| below −15 | 328, −0.04 [−0.36, +0.26] | 253, +0.82 [+0.27, +1.47] |
| −15 to −10 | 58, +0.27 [−1.34, +1.90] | 149, +0.91 [+0.20, +1.77] |
| −10 to −4.6 | 42, +4.49 [+1.97, +6.98] | 222, +0.80 [+0.12, +1.45] |
| −4.6 to 0 | 27, +5.98 [+2.71, +8.78] | 174, +0.09 [−0.47, +0.64] |
| above 0 (wrong) | 55, +3.31 [+0.98, +5.54] | 192, +1.76 [+0.89, +2.57] |

On goal crafts, the goal effect is absent where Qwen is sure and large where it is not. The Spearman correlation of the neutral margin with ΔL is −0.23 [−0.34, −0.12]. On ingredient crafts it is a roughly constant shift of about +0.8 nats at every confidence level, including the surest (margin correlation +0.01 [−0.07, +0.09]). On yes/no the gradient holds within both roles: goal from +2.5 to +11.9, ingredient from +1.3 to +3.8.

**A caveat on the gradient.** A smaller shift at −20 nats than at −5 could partly reflect the scale saturating far from the decision boundary rather than Qwen resisting the goal. Nothing here separates the two. The two-candidate ingredient rows, where the shift is as large at −20 as at −5, show that saturation does not force the shift to zero.

Llama, for comparison, shows no positive effect at any confidence level. Its within-cell fifths give −0.02 to −0.38, *more negative* where it is less sure (fifth 5 minus fifth 1 −0.35 [−0.65, −0.05]). Its goal-role penalty is −0.58 to −0.87 at every confidence level.

### The goal's effect on decisions and on discrimination

| Neutral-line comparison | Qwen two-candidate: neutral / aligned / swapped | Qwen yes/no: neutral / aligned / swapped |
|---|---|---|
| Failures answered success | 16.5% / 19.0% / 15.1% | 18.7% / 15.3% / 8.4% |
| Successes answered success | 71.5% / 73.8% / 66.2% | 72.5% / 63.0% / 33.5% |
| Balanced accuracy | 0.775 / 0.774 / 0.755 | 0.769 / 0.738 / 0.626 |
| AUC | 0.862 / 0.863 / 0.843 | 0.846 / 0.845 / 0.738 |
| AUC, aligned − neutral | +0.00 [−0.01, +0.01] | −0.00 [−0.01, +0.01] |
| AUC, swapped − neutral | −0.02 [−0.03, −0.01] | −0.11 [−0.13, −0.08] |

| Failures, by neutral prediction | Share of failures | ΔL two-candidate | Says success: aligned / swapped | Aligned minus swapped |
|---|---|---|---|---|
| Correct and confident (neutral L < −4.6) | 70% | +0.68 [+0.36, +1.02] | 3.4% / 0.9% | +2.6 points [+1.7, +3.4] |
| Correct but unsure (−4.6 to 0) | 13% | +0.88 [+0.22, +1.52] | 22% / 16% | +5.5 points [−0.5, +11.5] |
| Wrong (neutral L ≥ 0) | 16% | +2.10 [+1.30, +2.88] | 83% / 75% | +8.1 points [+1.5, +13.5] |
| Yes/no: correct and confident | 74% | +2.03 [+1.70, +2.37] | 1.2% / 0.0% | +1.2 points [+0.6, +1.8] |
| Yes/no: correct but unsure | 7% | +3.20 [+1.81, +4.62] | 12% / 3.6% | +8.9 points [+2.6, +15.0] |
| Yes/no: wrong | 19% | +6.32 [+5.12, +7.57] | 72% / 43% | +29 points [+22, +36] |

### A measurable goal effect, kept apart from a distortion of a competent predictor

The brief asks for these to be kept apart. Here the two are defined, and the data are read against each.

A **statistically measurable goal effect** is a mean difference in log-odds between the aligned and the swapped line whose interval excludes zero. Qwen has one: +0.94 nats on failures, confirmed, and positive in every cell except two. In the goal × never-obtained cell it is negative (the goal-craft penalty), and in ingredient × used up it is null. On yes/no it is positive even among the failures Qwen is surest of (+0.97 in the surest within-cell fifth, mean neutral −25.7 nats). Its existence is not in doubt.

A **distortion of a competent predictor** would mean the goal line makes a model that reads the state correctly give wrong answers it would not otherwise give, or rank items worse. Four observations bear on that:

1. *Confident, correct predictions rarely change.* Among the 70% of failures Qwen confidently and correctly calls failures, the aligned line produces a success call for 3.4% and the swapped line for 0.9% (yes/no 1.2% and 0%). Those predictions do move in log-odds (+0.68 nats two-candidate, +2.0 yes/no). But at −10 to −25 nats, a shift of that size changes the predicted probability by well under one point in the surest group.
2. *The changed decisions come mostly from uncertain or wrong items.* 5.6% of all failures (84 items) are called failures under the swapped line and successes under the aligned line (1.7% the other way). Of those 84, 28 are confident-correct items and 56 are items where Qwen's neutral prediction was already unsure (22) or wrong (34).
3. *Discrimination.* Against the neutral line, which names no item, the aligned line leaves Qwen's AUC and balanced accuracy unchanged on both scales. The swapped line lowers them (AUC −0.02 two-candidate, −0.11 yes/no). On yes/no the swapped line pushes successful goal crafts toward "No" by 18 nats (swapped − neutral on goal-role successes −18.0 [−19.8, −16.2]): it answers Yes to 73% of them under the neutral line and to 14% under the swapped line. So most of the damage the task lines do to Qwen's competence comes from naming the *other* goal, not from naming the action's own goal. The confirmatory ΔL, aligned minus swapped, is therefore partly "the off-task line makes Qwen pessimistic". On failures the two-candidate split is +0.55 aligned − neutral and −0.39 swapped − neutral. On yes/no both lines lower L against neutral: −2.1 and −5.0.
4. *Successes move more than failures.* ΔL is +2.61 on successes against +0.94 on failures (yes/no +9.6 against +2.9). A goal line that moved every prediction by the same amount would leave the AUC unchanged. One that moves successes more slightly improves aligned-versus-swapped discrimination, which is what the AUCs show (aligned − swapped +0.02 and +0.11).

**Reading.** The data fit "the task line is used as evidence where the history leaves the model unsure" better than "the goal overrides a working state judgement":

- The decision-level consequences are concentrated on items Qwen was unsure or wrong about.
- The log-odds effect grows as Qwen's confidence falls, within cells, on yes/no, on the goal role, and (section 5) with the inventory shown.
- The aligned line does not degrade Qwen's discrimination.

It is not a clean fit, for three reasons:

- A uniform shift of about +0.8 nats on two-candidate ingredient crafts, and about +1 to +2 nats on yes/no, reaches even the predictions Qwen holds most firmly.
- The swapped line *does* degrade discrimination on yes/no.
- On the two-candidate scale the within-cell gradient is weak and not monotone.

The defensible statement is two-part. The goal line shifts Qwen's predictions everywhere by a small amount, which at its confidence levels is almost never consequential. It also shifts them by a lot where Qwen is unsure, which is where answers change. Whether the second part is "rational use of task context as evidence" cannot be decided here. In TextCraft the task line carries no information about whether a craft will succeed, so any use of it as evidence is a mistake for a world model. The data only show *where* the mistake bites.

## 5. With the inventory shown

The diagnostic run scored 500 items (300 main failures in 150 same-history couples, 200 successes) under four prompts. The verified prompt is the confirmatory one. "Clean" removes the agent instructions from the system text. "Inventory" adds one history step, `> inventory`, with the environment's reply at that point, so the state that decides the outcome is on screen. "Clean + inventory" does both. Two-candidate scale only.

| Prompt (same 500 items) | Neutral AUC | ΔL, failures | ΔL, goal / ingredient | Confident-correct failures: share, ΔL | Unsure-correct: ΔL | Wrong: ΔL | Slope of ΔL per nat of margin |
|---|---|---|---|---|---|---|---|
| Verified | 0.84 [0.80, 0.87] | +1.15 [+0.60, +1.70] | +1.37 / +1.05 | 71%, +0.79 [+0.19, +1.38] | +1.54 | +2.45 | −0.09 [−0.16, −0.02] |
| Clean | 0.84 [0.80, 0.88] | +1.52 [+0.99, +2.06] | +3.03 / +0.77 | 72%, +0.92 [+0.38, +1.43] | +2.90 | +3.19 | −0.12 [−0.19, −0.05] |
| Inventory shown | 0.94 [0.91, 0.96] | +1.12 [+0.56, +1.68] | +1.86 / +0.74 | 81%, +0.39 [−0.07, +0.86] | +4.77 [+1.84, +8.06] | +3.91 [+0.82, +6.95] | −0.15 [−0.25, −0.05] |
| Clean + inventory | 0.93 [0.91, 0.96] | +1.42 [+0.90, +1.95] | +3.10 / +0.58 | 81%, +0.85 [+0.37, +1.31] | +2.27 | +5.19 | −0.16 [−0.24, −0.07] |

| Prompt | Confident-correct failures called success: aligned / swapped | Failures flipped to success by the aligned line (vs swapped) | AUC aligned − neutral | Balanced accuracy neutral / aligned / swapped |
|---|---|---|---|---|
| Verified | 4.2% / 0.5% | 6.3% | −0.01 [−0.03, +0.02] | 0.756 / 0.740 / 0.740 |
| Clean | 1.8% / 0.5% | 6.0% | +0.01 | 0.760 / 0.741 / 0.727 |
| Inventory shown | 1.7% / 0.4% | 5.3% | −0.01 [−0.02, +0.00] | 0.848 / 0.844 / 0.813 |
| Clean + inventory | 1.7% / 0.4% | 5.3% | +0.00 | 0.843 / 0.856 / 0.817 |

**The unchanged mean hides a redistribution.** With the inventory shown, Qwen's competence rises by 0.10 of AUC and its mean ΔL does not change (paired difference −0.04 [−0.45, +0.38]). The share of failures it gets confidently right rises from 71% to 81%. On those, the effect is smaller than under the verified prompt (+0.39 against +0.79, intervals overlapping). On the 38 items that *become* confident and correct only when the inventory is shown, ΔL is +0.61 [−0.65, +2.20], against +1.16 [+0.05, +2.50] for the same items under the verified prompt. On the 19% Qwen is still unsure or wrong about, the effect is +4.8 and +3.9, larger than under the verified prompt. The same mean is carried by fewer, more strongly affected items. The slope of ΔL on Qwen's confidence is steepest with the inventory shown (−0.15 and −0.16 per nat).

**Does the goal still move confident, correct predictions?** In log-odds, a little. The interval for confident-correct failures excludes zero under the clean and clean + inventory prompts (+0.92 and +0.85) and not under the inventory prompt alone (+0.39 [−0.07, +0.86]). In the surest group (neutral below −15), the shift is +0.76 [+0.26, +1.27] under clean + inventory and +0.18 [−0.39, +0.73] under inventory alone. In decisions, almost never: 1.7% of confident-correct failures are called successes under the aligned line and 0.4% under the swapped line, with the state on screen. The aligned line does not lower discrimination under any prompt. So with the state fully readable, the goal effect persists at the same mean size. It still reaches settled predictions as a small log-odds shift, and it changes answers almost only where Qwen remains unsure. That is the section 4 picture with sharper contrast. Showing the state shrinks the set of uncertain items but not the effect on them.

Llama, alongside: the inventory raises its AUC from 0.54 to 0.62. Its effect stays negative on goal crafts (−0.87) and null on ingredients (+0.05). On its confident-correct failures it is negative under every prompt (−0.38 to −0.70), as in section 4.

## 6. Implications

None of this changes a confirmatory verdict. The TextCraft goal effect is confirmed for Qwen3-8B and not for Llama-3.1-8B, so the replication holds for one model only.

**(a) Reading Qwen's confirmed effect as a distortion of a working world model.** Qwen is a working TextCraft world model in the sense that matters for the design. It ranks successes above failures in every role and failure class (AUC 0.72 to 0.97 across role × class cells on both scales, though as low as 0.58 for "too few" ingredient crafts in 9- to 12-step histories), and it passes the stricter families rule everywhere. So its effect is not an artefact of incompetence, unlike Llama's null. Three qualifications belong with any "distortion" reading:

- *Its competence is partly cue-based.* A rule with no subtraction (names or gross amounts, plus "not made before") fits its used-up judgements as well as the truth. Its errors cluster where counts are close to the need.
- *It is overconfident.* Its probabilities beat the base rate by only 12% in Brier score, and not at all on ingredients or on yes/no.
- *The effect's consequences sit on its uncertain items.* The goal line moves settled predictions by a small log-odds amount that almost never changes an answer, and it moves uncertain ones a lot.

"A goal-sensitive bias in a competent but imperfect, overconfident predictor, concentrated on the items it is unsure of" is accurate. "The goal overrides a correct world model" is not supported. "The goal distorts confident, correct predictions" is true only in log-odds and not in answers.

**(b) The cross-model pattern ("the effect requires competence").** Across ALFWorld models, every competent model shows a positive effect and the near-null effects come from models that cannot do the task (`notes/textcraft_vs_alfworld.md`, section 9). Qwen and Llama in TextCraft fit that pattern at the level of models. Within Qwen the relation runs the other way: the effect is largest in the strata and on the items where it is least competent or least confident. These are not in conflict, but they narrow the claim. A model needs some working state judgement for a consistent positive effect to appear: Llama, uncertain nearly everywhere, shows none even on its uncertain items. Within a model that has one, the effect lives mostly where that judgement is weak. "Requires competence" should be stated as "requires a working state judgement, and acts mainly at its edges", not as "grows with competence". The cross-environment note's finding that the effect grows as the baseline becomes less certain is the same observation seen from the baseline side.

**(c) Round 2's gates for Qwen.** On round-1 numbers under the verified prompt, Qwen passes the families rule on every population a round-2 contrast reads, with its lowest AUC at 0.85 for the contrast populations and 0.76 for the "too few" class. The clean prompt, round 2's default, left its AUC unchanged on the diagnostic items (0.84 and 0.84). Contrast-specific gates would therefore not change Qwen's status. They would matter for Llama, which passes on goal-role populations (0.85) and fails on ingredient ones (0.55). But Λ is an ingredient contrast, and Dn, Dc and T are differences between goal and ingredient parts, so Llama would still be gated out of all four. Only P, on parent records with no round-1 analogue, could differ. Two descriptive additions, fixed before any round-2 score, would carry this note's findings into round 2. Whether to add them is the owner's decision.

- *A confidence-stratified readout.* Each contrast's effect by within-cell fifths of the neutral L, and the share of confident-correct failures the aligned line turns into success predictions. The mean alone would again hide where the effect acts.
- *The repetition fits.* H11 and H2 ∧ H11 next to H6 on used-up failures and ingredient successes, so that "competent on used up" is not read as "subtracts consumption" for any model.

**(d) How the paper should describe Qwen's TextCraft result.** A suggested wording, to be adapted: "Qwen3-8B is a competent but imperfect TextCraft world model. Under a neutral task line it ranks working above failing crafts with AUC 0.86 (0.72 to 0.97 across roles and failure types). Its probabilities are overconfident: a third of the crafts it gives more than 90% fail. Its accuracy on consumed inputs is matched by a rule that treats a repeated craft as exhausted, so we do not claim it performs inventory arithmetic. Naming the action's goal in the task line, rather than the other displayed goal, raises its log-odds of success by 0.94 nats (yes/no 2.9). The shift is present even on predictions it holds with over 99% confidence, but there it is small and rarely changes an answer (3% of such failures). Its practical consequences concentrate on crafts the model is unsure of, and the effect persists at the same size when the inventory is shown, raising competence to 0.94. Against a task line that names no item, the aligned line does not reduce its discrimination. Naming the other goal does, chiefly by making the model predict failure for working crafts of the unnamed item." The paper should not call Qwen's TextCraft result a distortion of a correct world model without the qualifications in (a). It should report the effect's size alongside the share of answers it changes, as the cross-environment note does for ALFWorld.

## Limits

- *Post hoc.* Everything here is exploratory and was computed after the confirmatory results and the earlier exploratory notes had been read. The heuristics H1–H10 were fixed before any fit and H11 and its combinations before their fits. The cuts by confidence and by the deciding quantities were chosen knowing the earlier stratum correlation.
- *Confidence is self-reported.* "Confident" means Qwen's own neutral L is far from zero. Items differ in many ways besides confidence; within-cell fifths remove the role and class mix but not everything.
- *Scale saturation.* Smaller shifts far from the boundary may partly reflect the log-odds scale saturating rather than resistance to the goal.
- *The yes/no question may be read as task success.* On yes/no, the swapped line lowers Qwen's L on working goal crafts by 18 nats. One reading is that "Will this action succeed?" is being read as "will it advance the task". That is a measurement question for the robustness audit on branch `textcraft-qwen-robust` and is not examined here.
- *Single runs.* There is one scoring run per prompt and one example set, and the prompt-variant results rest on 500 items, so their intervals are wide.

## Reproducing

From the repository root, with the confirmatory scores copied from ICE (`text-wm-fix24/runs/textcraft_score`) and the prompt-variant scores (`text-wm-fix27/runs/textcraft_dx_prompts`):

```
python scripts/textcraft_qwen_competence.py --scores <dir with the four textcraft_score shards> \
    --prompts <textcraft_dx_prompts dir> --out notes/textcraft_qwen_competence_metrics.json
```

It runs on CPU in about 10 minutes and needs numpy and scipy. It imports every function of `scripts/textcraft_llama_competence.py` unchanged, adds H11, H2 ∧ H11 and H6 ∧ H11 and the analyses of this note, and asserts that the checker's accounting reproduces every item's outcome. The metrics file holds every number quoted here, per model and per scale:

- `competence`, `decisions`, `calibration` and `families_rule`;
- `q1_extra`: Yes rates by role × class, calibration and Brier by role, and the rule per population;
- `heuristics` (all fits, including the new ones), `errors`, `repeat` and `quantities`;
- `competence_vs_effect` (the Llama note's strata);
- `q4`: confidence bins, within-cell fifths, decisions and discrimination by line, and the strata correlations with intervals;
- `q4_quantities` and `prompt_variants`.
