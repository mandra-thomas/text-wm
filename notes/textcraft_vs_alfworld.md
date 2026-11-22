# Why is the goal effect smaller in TextCraft than in ALFWorld? (exploratory, post hoc, 2026-11-20)

**Status.** Everything in this note is exploratory and post hoc. None of it was logged before the data; it was computed after both the ALFWorld RQ2 result (`rq2_results.md`) and the TextCraft result (`textcraft_results.md`, on branch `textcraft`) were known. No model was run: the note only re-reads saved scores. Intervals are 95% percentile intervals from 2,000 cluster-bootstrap resamples (seed 20260924), clustered by game for ALFWorld and by goal pair for TextCraft. No p-values are reported and nothing is corrected for multiple comparisons, so with this many readouts some intervals will exclude zero by chance. Every number is in [`textcraft_vs_alfworld_metrics.json`](textcraft_vs_alfworld_metrics.json), produced by `scripts/xenv/compare.py` (loaders in `scripts/xenv/data.py`). The Llama-specific diagnosis of the TextCraft reversal is a separate investigation (branch `textcraft-llama-dx`); here Llama appears only where the cross-environment comparison needs it.

## 1. The question and the measures

Both studies ask whether naming the action's goal in the prompt's task line changes a prompted world model's prediction for an action whose failure the history already settles. Each item is scored three times, with three task lines that differ in nothing else:

- the **aligned** task names the goal the action serves (in ALFWorld, the task with the action's object; in TextCraft, the goal whose recipe tree contains the crafted item);
- the **swapped** task names another object or goal (in ALFWorld, another object type in the room; in TextCraft, the other goal of the pair, whose recipe tree is also on screen);
- the **neutral** task names no object ("Your task is to: explore the room." in ALFWorld, "Goal: explore the crafting commands." in TextCraft).

The models are few-shot Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, with the same scoring code in both environments.

The two-candidate score compares the environment's success reply with its failure reply, teacher-forced after the prompt:

$$L = \log P(o^{+} \mid \text{prompt}) - \log P(o^{-} \mid \text{prompt}).$$

Here $o^{+}$ is the success text and $o^{-}$ the failure text. In ALFWorld the failure text is always "Nothing happens." and the success text is the action-specific sentence the real server gives when the action works ("You pick up the saltshaker 1 from the drawer 1."). In TextCraft the failure text is "Could not find enough items to craft minecraft:x" and the success text is "Crafted N minecraft:x"; both name the item. The goal effect is the mean over items of the aligned-minus-swapped difference:

$$\Delta L = \frac{1}{N}\sum_{i=1}^{N}\left[L_i(\text{aligned}) - L_i(\text{swapped})\right].$$

Because a reply is scored token by token, $L$ splits exactly into a first-token term and the rest of each reply:

$$L = \underbrace{\left[\log P(t^{+}_1) - \log P(t^{-}_1)\right]}_{\ell_{\text{first}}} + \underbrace{\log P(o^{+}_{2:} \mid t^{+}_1)}_{\text{success tail}} - \underbrace{\log P(o^{-}_{2:} \mid t^{-}_1)}_{\text{failure tail}}.$$

The first token is the reply's first word piece: "You" against "Nothing" in ALFWorld, "Craft" against "Could" in TextCraft. It is the closest thing either format has to a pure success-or-failure decision. The tails are the probability of the rest of each sentence given its first token; they carry the object names, receptacle names and counts.

The yes/no score replaces "Next observation:" with a question and compares the single tokens "Yes" and "No", so its replies are the same in both environments. The question wording is not: ALFWorld asks "will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens." and TextCraft asks "Will this action succeed? Answer Yes or No."

The other scales used below are these. The standardized effect is the mean of the item differences divided by their standard deviation:

$$d = \frac{\overline{\Delta L_i}}{\mathrm{SD}(\Delta L_i)}.$$

The probability shift $\Delta p$ is the mean change, in percentage points, of $\sigma(L)$, the logistic of $L$:

$$\Delta p = 100\cdot\frac{1}{N}\sum_i\left[\sigma\!\left(L_i(\text{aligned})\right) - \sigma\!\left(L_i(\text{swapped})\right)\right].$$

For the two-candidate score $\sigma(L)$ is not the model's probability of success (it compares one success wording with one failure wording); for the yes/no score it is the model's probability of "Yes" against "No". The **one-direction flip rate** is the share of items with $L<0$ under the swapped task and $L>0$ under the aligned one. A **two-way sign change** counts a change of sign in either direction, on the full $L$ or on the first-token term. The **share answered Yes** is the share of items whose yes/no $L$ is above zero.

The item populations are ALFWorld's 4,666 RQ2 failures from 125 games (histories of 0 to 11 steps, 514 at step 0), its 292 determined successes, and TextCraft's 1,500 main failures (750 same-history couples, 229 goal pairs, histories of 1 to 12 steps), 600 successes, 600 usefulness failures and 100 empty-history failures. ALFWorld's aligned and swapped two-candidate scores come from the confirmatory run (`rq2_score`); its neutral two-candidate scores from `rq1_score`, its successes from `rq1x_score` and its yes/no and counterfactual-task scores from `rq1v_score`. Contrasts that pair `rq2_score` with `rq1_score` cross scoring jobs, which adds a little noise (for example aligned minus neutral is 8.74 here against 8.66 within `rq1_score`). The TextCraft scores are `textcraft_score` (ICE, `text-wm-fix24`), all arms from one job per shard. The script reproduces the published headline numbers exactly (ALFWorld 6.12, 1.65, yes/no 2.80, 2.21; TextCraft +0.94, −0.19, yes/no +2.92, −0.15).

## 2. Which scales shrink (question 1)

The table compares the two environments' main failure populations on every scale they share. The last column is TextCraft divided by ALFWorld, with an interval from independent bootstrap draws of the two environments.

**Qwen3-8B**

| Scale | ALFWorld (4,666) | TextCraft main (1,500) | TextCraft / ALFWorld |
|---|---|---|---|
| Two-candidate ΔL, nats | 6.12 [5.51, 6.75] | 0.94 [0.64, 1.25] | 0.15 [0.10, 0.21] |
| First-token ΔL, nats | 2.52 [2.26, 2.77] | 0.97 [0.68, 1.26] | 0.39 [0.27, 0.51] |
| Remainder (tails), nats | 3.60 [3.16, 4.05] | −0.03 [−0.07, 0.00] | about 0 |
| Yes/no ΔL, nats | 2.80 [2.60, 3.01] | 2.92 [2.56, 3.30] | 1.04 [0.89, 1.21] |
| Standardized, two-candidate | 1.02 [0.92, 1.14] | 0.20 [0.14, 0.25] | 0.19 [0.13, 0.25] |
| Standardized, first token | 0.75 [0.68, 0.81] | 0.21 [0.15, 0.26] | |
| Standardized, yes/no | 0.88 [0.81, 0.95] | 0.49 [0.45, 0.53] | 0.55 [0.50, 0.62] |
| Two-candidate Δp, points | 5.88 [5.05, 6.74] | 3.59 [2.25, 4.85] | 0.61 [0.37, 0.86] |
| Yes/no Δp, points | 0.17 [0.08, 0.29] | 6.71 [5.30, 8.13] | about 40 |
| One-direction flips, two-candidate | 5.9% [5.1, 6.8] | 5.6% [4.2, 6.9] | |
| Reverse flips (success to failure) | 0.1% | 1.7% | |
| Sign changes, full L | 6.0% | 7.3% | |
| Sign changes, first token | 10.5% [9.1, 12.0] | 7.7% [6.3, 9.2] | |
| One-direction flips, yes/no | 0.2% | 7.5% [6.0, 9.0] | |
| Answered Yes: aligned / swapped / neutral | 0.2% / 0.0% / 0.2% | 15.3% / 8.4% / 18.7% | |

**Llama-3.1-8B**

| Scale | ALFWorld | TextCraft main | TextCraft / ALFWorld |
|---|---|---|---|
| Two-candidate ΔL, nats | 1.65 [1.50, 1.78] | −0.19 [−0.30, −0.09] | −0.12 [−0.18, −0.06] |
| First-token ΔL, nats | 0.68 [0.62, 0.76] | −0.16 [−0.25, −0.08] | −0.24 [−0.37, −0.11] |
| Remainder (tails), nats | 0.96 [0.84, 1.09] | −0.03 [−0.06, 0.00] | about 0 |
| Yes/no ΔL, nats | 2.21 [2.04, 2.39] | −0.15 [−0.21, −0.09] | −0.07 [−0.10, −0.04] |
| Standardized, two-candidate | 0.85 [0.78, 0.93] | −0.14 [−0.22, −0.07] | |
| Standardized, yes/no | 1.12 [1.04, 1.21] | −0.17 [−0.24, −0.10] | |
| Two-candidate Δp, points | 16.4 [14.5, 18.3] | 0.4 [−0.6, 1.3] | 0.02 [−0.04, 0.08] |
| Yes/no Δp, points | 16.2 [14.6, 17.8] | −0.3 [−0.8, 0.2] | |
| One-direction flips, two-candidate | 19.7% | 5.9% | |
| Reverse flips | 2.3% | 4.2% | |
| One-direction flips, yes/no | 16.0% | 0.7% | |
| Answered Yes: aligned / swapped / neutral | 21.0% / 5.6% / 5.3% | 3.5% / 4.0% / 9.9% | |

**What the tables show for Qwen.** The shrinkage depends almost entirely on the scale. On the two-candidate log-odds the TextCraft effect is about a sixth of ALFWorld's, but more than half of ALFWorld's 6.12 nats (3.60 of them) sits in the success reply's tail, the probability of the words after "You" ("pick up the saltshaker 1 from the drawer 1."). The failure tail contributes nothing in either environment (−0.00 in ALFWorld; "Nothing" is always followed by " happens."). In TextCraft the tail term is zero by construction: both replies name the same item, so whatever the goal does to the probability of the item's name cancels, and the prefix-only arm (replies cut before the item name) gives the same effect, 0.95. On the first token, the shared decision word, the ratio rises from 0.15 to 0.39. On the yes/no question it is 1.04: the effect does not shrink at all. On the probability scales TextCraft is comparable or larger: the two-candidate shift is 3.6 points against 5.9, and the yes/no shift is 6.7 points against 0.2, because ALFWorld's yes/no answers on failures sit at the floor (the mean yes/no log-odds under the neutral task is −26.5 nats, and 0.2% of failures are answered Yes), so a 2.8-nat shift there moves almost no probability. Flip rates are about the same in both environments (5.9% and 5.6% one-direction), though TextCraft has more flips in the reverse direction (1.7% against 0.1%).

The standardized effect shrinks by a factor of five on the two-candidate scale and by about half on yes/no. The standardized two-candidate shrinkage is not caused by noisier TextCraft items (the item SD of the difference is 5.98 nats in ALFWorld and 4.74 in TextCraft); it is the mean that shrinks.

**What the tables show for Llama.** The effect is gone on every scale, not just on the reply-dependent ones: two-candidate, first-token, yes/no and both probability scales are all at or just below zero. Llama's shrinkage is therefore not a measurement artefact. The remainder term again shows that 0.96 of its 1.65 ALFWorld nats were in the success tail.

**Is the two-candidate shrinkage explained by the replies?** For the larger part, yes. Three differences in the replies matter:

- **The success tail.** It carries 59% of Qwen's and 58% of Llama's ALFWorld two-candidate effect. TextCraft's design (the object name in both replies) removes it. This alone takes Qwen from 6.12 to a comparable 2.52.
- **How extreme L is.** Under the swapped task Qwen's mean ALFWorld L is −20.7 nats, of which −12.7 is the success tail (a 13-token sentence the model cannot fully predict); under the neutral task it is −23.4. In TextCraft the mean L is about −10 under every task, and almost all of it is the first token (the tails are −0.1 to −0.3 nats). The ALFWorld first-token term itself is not more extreme than TextCraft's (neutral means −9.0 and −9.6 nats).
- **Reply length.** ALFWorld's success replies average 13.3 tokens against a 3-token failure; TextCraft's are 9.1 against 12.1. Length matters only through the tail term above.

After removing the tail, a gap remains on the first token (2.52 against 0.97). The yes/no scale shows no gap, so the remaining first-token gap is itself scale-dependent. Sections 3 and 4 examine it further.

## 3. Matching the populations (question 2)

**ALFWorld's effect is concentrated at step 0, which TextCraft's main set lacks.** The table splits ALFWorld by history length (number of actions before the scored action) and sets TextCraft beside it.

| Population | Qwen two-candidate | Qwen first token | Qwen yes/no | Qwen Δp | Llama two-candidate | Llama first token | Llama yes/no | Llama Δp |
|---|---|---|---|---|---|---|---|---|
| ALFWorld step 0 (514) | 10.99 [10.12, 11.84] | 7.50 [6.96, 8.05] | 7.71 [7.24, 8.21] | 42.0 | 1.52 [1.39, 1.65] | 1.25 [1.15, 1.35] | 3.34 [3.04, 3.65] | 13.2 |
| TextCraft empty history (100) | 5.74 [4.40, 7.04] | 5.76 [4.43, 7.07] | 14.13 [12.21, 16.07] | 14.6 | 0.06 [−0.26, 0.39] | 0.05 [−0.24, 0.35] | 1.09 [0.78, 1.42] | −0.1 |
| ALFWorld steps 1–11 (4,152) | 5.52 [4.91, 6.16] | 1.90 [1.66, 2.13] | 2.20 [1.99, 2.39] | 1.4 | 1.66 [1.50, 1.81] | 0.61 [0.54, 0.70] | 2.07 [1.90, 2.25] | 16.8 |
| ALFWorld steps 1–11, weighted to TextCraft's lengths | 5.48 [4.82, 6.17] | 1.81 [1.52, 2.09] | 2.05 [1.83, 2.27] | 1.5 | 1.69 [1.51, 1.88] | 0.61 [0.53, 0.69] | 1.93 [1.77, 2.10] | 17.4 |
| TextCraft main, 1–12 steps (1,500) | 0.94 [0.64, 1.25] | 0.97 [0.68, 1.26] | 2.92 [2.56, 3.30] | 3.6 | −0.19 [−0.30, −0.09] | −0.16 [−0.25, −0.08] | −0.15 [−0.21, −0.09] | 0.4 |

The weighted row reweights ALFWorld's per-length means (lengths 1 to 11) to the share of TextCraft main items at each length, with TextCraft's 12-step items counted as 11. ALFWorld's effect is flat across lengths 1 to 11 for both models (Qwen two-candidate 5.2 to 6.0, first token 1.8 to 2.0), so length within that range is not the issue; the difference is step 0 against everything else.

For Qwen, matching the populations closes most of the first-token gap and reverses the yes/no comparison:

- **Step 0 against the empty history.** On the first token TextCraft keeps 77% of ALFWorld's effect (ratio 0.77 [0.58, 0.96]); on yes/no it is almost twice as large (ratio 1.83 [1.56, 2.11]); only the two-candidate scale (tail included) is about halved.
- **Later steps against the main set.** At matched lengths the first-token ratio is about 0.54 (0.97 against 1.81), the yes/no effect is larger in TextCraft (2.92 against 2.05), and so is the probability shift (3.6 points against 1.5).

So of Qwen's pooled first-token ratio of 0.39, part comes from ALFWorld's 11% of step-0 items (which carry 7.50 nats each on the first token), and a residual factor of about two remains at later steps on the first-token scale only.

For Llama matching changes nothing: at step 0 against the empty history the two-candidate and first-token effects are about 4% of ALFWorld's, and at later steps they are negative. The one exception is the yes/no question on the 100 empty-history items, where Llama shows a positive effect of 1.09 nats (a third of its ALFWorld step-0 effect).

**The action's object in the aligned task.** In ALFWorld every RQ2 item acts on the goal object, so the aligned task always names the action's object. In TextCraft only the 510 goal-role items craft the goal item itself; the other 990 craft an ingredient of it, and the aligned task names the goal, not the ingredient. The goal-role items are therefore the closest match to ALFWorld. For Qwen they give 1.05 [0.53, 1.60] two-candidate, 1.19 [0.68, 1.73] first token and 4.97 [4.20, 5.75] yes/no (Δp on yes/no 11.5 points), against 0.88, 0.86 and 1.87 for ingredient items. Restricting to goal-role items raises the yes/no effect well above ALFWorld's but leaves the first-token effect at about half of ALFWorld's later-step value. For Llama the goal-role items are where its negative effect sits (−0.63 two-candidate, −0.30 yes/no).

**Failure types that map across.** In ALFWorld's RQ2 population no history acts on the goal object, so for every move, heat, cool and clean item (2,217 items) the object was never obtained: that is the ALFWorld analogue of TextCraft's "never obtained" class. Take actions fail for reasons with no TextCraft counterpart (the agent is away from the receptacle, the receptacle was seen without the object, it is closed, the hands are full). TextCraft's "too few" and "used up" classes, which depend on counting, have no ALFWorld counterpart.

| Cell | Qwen two-candidate | Qwen first token | Qwen yes/no | Llama two-candidate | Llama yes/no |
|---|---|---|---|---|---|
| ALFWorld object never obtained (move, heat, cool, clean; 2,217) | 7.14 [6.41, 7.88] | 2.45 [2.17, 2.73] | 2.74 [2.49, 2.96] | 0.90 [0.77, 1.03] | 1.75 [1.56, 1.95] |
| TextCraft never obtained (500) | 0.59 [0.19, 0.97] | 0.70 [0.33, 1.06] | 2.44 [1.92, 2.96] | −0.43 [−0.57, −0.29] | −0.30 [−0.38, −0.21] |
| TextCraft too few (500) | 2.40 [1.71, 3.14] | 2.39 [1.72, 3.11] | 5.78 [4.91, 6.74] | −0.13 [−0.35, 0.07] | −0.10 [−0.22, 0.02] |
| TextCraft used up (500) | −0.17 [−0.56, 0.21] | −0.18 [−0.57, 0.21] | 0.54 [0.29, 0.77] | −0.02 [−0.11, 0.08] | −0.06 [−0.12, 0.01] |

On the mapped cell Qwen's yes/no effect is the same in both environments (2.74 against 2.44), and the first-token effect is a third or less in TextCraft. Within TextCraft the pooled effect is pulled down by two cells with no effect: used-up items (a third of the main set; first token −0.18, yes/no 0.54) and goal-role never-obtained items (Section 6). The "too few" class, where the failure turns only on a count, carries the largest TextCraft effect: its first-token effect (2.39) is as large as ALFWorld's pooled first-token effect, and its yes/no effect (5.78) is twice ALFWorld's.

## 4. Baselines and competence (question 3)

A goal shift of a given size in nats means different things depending on where the model's baseline sits. The table gives each model's mean L under the neutral task on failures and on successes, and the AUC, the probability that a random success scores above a random failure. The TextCraft AUC uses its main and usefulness failures against its successes, as its design does. ALFWorld's 292 successes and TextCraft's 600 successes are built differently from the failures in each environment, so the AUCs are rough guides to competence, not matched measurements.

| Neutral task | Qwen ALFWorld | Qwen TextCraft | Llama ALFWorld | Llama TextCraft |
|---|---|---|---|---|
| Mean two-candidate L, failures / successes | −23.4 / 15.3 | −10.5 / 7.3 | −3.7 / 2.8 | −3.0 / −0.3 |
| Mean first-token L, failures / successes | −9.0 / 19.0 | −10.3 / 7.3 | 1.6 / 4.3 | −2.5 / −0.2 |
| Mean yes/no L, failures / successes | −26.5 / 3.1 | −12.8 / 6.3 | −4.4 / 1.8 | −4.1 / −2.4 |
| AUC, two-candidate | 0.93 [0.89, 0.96] | 0.87 [0.85, 0.89] | 0.87 [0.83, 0.91] | 0.68 [0.65, 0.71] |
| AUC, first token | 0.99 [0.99, 1.00] | 0.87 [0.85, 0.89] | 0.88 [0.84, 0.92] | 0.69 [0.66, 0.72] |
| AUC, yes/no | 0.99 [0.98, 0.99] | 0.85 [0.83, 0.87] | 0.97 [0.96, 0.98] | 0.67 [0.64, 0.70] |

Qwen is far more certain of ALFWorld failures on the yes/no question (−26.5 nats against −12.8) and on the two-candidate score (because of the success tail), but not on the first token (−9.0 against −10.3). Both models discriminate less well in TextCraft, Llama much less (0.67 to 0.69, against 0.87 to 0.97 in ALFWorld). Dividing ΔL by the neutral success-minus-failure gap (TextCraft's design standardizer) gives Qwen 0.16 (ALFWorld) against 0.05 (TextCraft) on the two-candidate score, 0.09 against 0.06 on the first token and 0.09 against 0.15 on yes/no; for Llama 0.25 against −0.07 and 0.36 against −0.09.

**Within each environment the effect grows as the baseline becomes less certain.** Splitting failures into fifths by the neutral first-token L, Qwen's ALFWorld first-token effect rises from 1.92 (most certain fifth, mean −13.5) to 3.23 (least certain, mean −2.9), and its TextCraft effect from 0.28 (mean −20.8) to 1.85 (mean +5.5). By the neutral yes/no L, TextCraft's yes/no effect rises from 1.41 to 6.32; ALFWorld's is flat (3.15 to 2.97), because nearly all its items sit between −30 and −22. Llama runs the other way in ALFWorld: its effect is largest where it is most certain of failure (two-candidate 2.28 in the fifth with the lowest neutral first-token L, falling to 1.07 in the highest; 2.74 falling to 0.64 by the neutral yes/no L). Its TextCraft effect is zero or negative in every fifth.

**At a matched baseline the first-token gap remains.** In fixed bins of the neutral first-token L, Qwen's first-token effect is 2.07 against 0.71 between −15 and −10, 2.89 against 1.47 between −10 and −5, and 3.34 against 0.87 between −5 and 0. Reweighting ALFWorld to TextCraft's distribution over these bins gives 2.09 [1.70, 2.52] against TextCraft's 0.97, and on the yes/no scale 2.83 against 2.92. Reweighting on the neutral yes/no L is limited by thin overlap: only half of TextCraft's items fall in bins with at least 20 ALFWorld items, and on that half TextCraft's yes/no effect is 1.86 against ALFWorld's 2.75. For Llama, reweighting ALFWorld to TextCraft's first-token baseline gives 2.25 two-candidate and 2.36 yes/no, against −0.16 and −0.14. So baseline confidence does not explain the cross-environment gap for either model, even though it moves the effect within each environment.

**The direction relative to the neutral task differs.** This was not asked but it is the clearest structural difference in the data. In ALFWorld the aligned task lifts the prediction above the neutral task, and the swapped task lifts it a little. In TextCraft the aligned task barely moves it, and the swapped task pushes it toward failure.

| Contrast (Qwen) | ALFWorld all | ALFWorld step 0 | TextCraft main | TextCraft empty history |
|---|---|---|---|---|
| First token, aligned minus neutral | +3.52 [3.23, 3.81] | +9.69 [9.01, 10.34] | +0.43 [0.12, 0.73] | −0.97 [−2.42, 0.55] |
| First token, swapped minus neutral | +1.00 [0.84, 1.16] | +2.19 [1.54, 2.84] | −0.54 [−0.81, −0.27] | −6.74 [−8.19, −5.26] |
| Yes/no, aligned minus neutral | +2.52 [2.26, 2.76] | +6.14 [5.45, 6.80] | −2.07 [−2.49, −1.67] | −2.70 [−4.31, −1.08] |
| Yes/no, swapped minus neutral | −0.28 [−0.44, −0.13] | −1.57 [−2.06, −1.10] | −5.00 [−5.48, −4.52] | −16.83 [−19.16, −14.38] |

The aligned-minus-swapped difference is about the same on the yes/no scale in both environments, but in ALFWorld it is mostly the aligned task raising "Yes", while in TextCraft it is mostly the swapped task lowering it. Llama shows the same split in weaker form: in ALFWorld its aligned task raises yes/no L by 2.44 over neutral; in TextCraft both tasks lower it (−0.60 and −0.45).

## 5. History–goal agreement and other design differences (question 4)

TextCraft's main items come in couples: the same history scored once for each goal of a pair, so on average the history favours neither goal. ALFWorld histories were produced by an agent pursuing the real (aligned) task, so a history might fit the aligned goal. If agreement carried the effect, ALFWorld would look larger for that reason alone. Three pieces of evidence bear on this, none decisive:

- **TextCraft's own agreement estimate is negative.** TextCraft records which goal's recipe tree the history's gets and crafts belong to (`history_fit`). Qwen's effect is 0.62 [0.26, 0.99] where the history fits the aligned goal and 1.24 [0.77, 1.71] where it fits the swapped goal, with 0.99 on ties; half the difference, the design's estimate of what agreement adds, is about −0.3 nats. For Llama it is about −0.05. If anything, agreement makes the TextCraft effect smaller.
- **ALFWorld's largest effect has no history at all.** At step 0 there is nothing for the task to agree with, yet that is where Qwen's effect is largest (10.99 nats).
- **ALFWorld's counterfactual aligned task leaves the effect unchanged.** On the 2,974 take, heat, cool and clean items, replacing the real task's destination with another receptacle (so the aligned task is no longer the one the agent pursued) gives 6.52 [5.85, 7.19] against 6.33 [5.67, 7.02] with the real task in the same run (difference +0.19 [0.02, 0.36]); Llama 2.09 against 1.96. This rules out that the effect needs the exact real task, but the counterfactual task still names the same object, so any agreement between the history and the object survives it.

A rough ALFWorld proxy for agreement is whether the history's observations mention the goal or the swap object (the agent saw it somewhere, without acting on it). Among later-step items, Qwen's effect is 5.50 where neither is named (3,065 items), 6.82 where only the swap object is named, and 2.03 [1.16, 2.84] where only the goal object is named (213 items); Llama's 1.59, 2.17 and 0.97. Seeing the goal object elsewhere shrinks the effect, which fits the dissociation result that location information reduces it (`exploratory_results.md`, section 3), and does not look like agreement inflating it.

**What can and cannot be compared.** The yes/no answers, the first token and the probability shifts can be compared across environments with the caveats above (different question wording, different first-token words, ALFWorld's yes/no floor for Qwen). The full two-candidate log-odds cannot: its tail term exists only in ALFWorld. Four design differences, listed in TextCraft's design (section 13, D7), cannot be separated by any analysis of these scores:

- what the failure turns on (hidden object location against visible inventory counts);
- the swapped goal's status (in TextCraft it is a displayed task with its recipe tree on screen; in ALFWorld it is just another object type in the room);
- the item construction (couples against agent histories), for which ALFWorld has no tie-couple equivalent;
- the reply formats and the yes/no question wording.

## 6. The shared negative cell (question 5)

In TextCraft, crafting the goal item itself when one of its ingredients was never obtained gives a negative two-candidate effect for both models: Qwen −0.45 [−0.89, −0.05] and Llama −0.75 [−0.92, −0.58] (252 items). For Qwen the sign is specific to the two-candidate format: on the first token it is −0.22 [−0.62, 0.15] and on yes/no it is positive, +1.73 [1.20, 2.23]. For Llama it is negative on every scale (yes/no −0.46). This is also the most certain TextCraft cell: Qwen's neutral first-token L there is −18.6 and its yes/no L −22.4, so the cell sits where Section 4 found the effect smallest. On the 57 empty-history goal-role items (all never obtained) Qwen's effect is positive (2.71 two-candidate, 12.86 yes/no) and Llama's is −0.27 two-candidate and +1.58 yes/no.

The ALFWorld analogue is moving the goal object to a receptacle of the task's target type when the agent never picked it up. That is the act that completes a plain placement task ("put some saltshaker on cabinet"), or the last step of a heat, cool or clean task, with a precondition (holding the object) visibly unmet, since RQ2 histories never act on the goal object. Its sign is positive for both models:

| ALFWorld cell | Items | Qwen two-candidate | Qwen first token | Qwen yes/no | Llama two-candidate | Llama yes/no |
|---|---|---|---|---|---|---|
| Move goal object to target type, all | 668 | 7.11 [6.08, 8.10] | 3.45 [3.08, 3.86] | 3.07 [2.68, 3.44] | 0.69 [0.50, 0.89] | 2.10 [1.84, 2.37] |
| … plain placement tasks (completes the task) | 163 | 8.68 [6.88, 10.52] | 3.73 [2.93, 4.70] | 3.52 [2.70, 4.34] | 0.68 [0.44, 0.98] | 2.19 [1.73, 2.78] |
| … at step 0 | 107 | 13.65 [12.38, 14.83] | 9.53 [8.63, 10.37] | 6.49 [5.87, 7.03] | 2.23 [1.94, 2.51] | 3.55 [3.12, 3.95] |
| … at later steps | 561 | 5.86 [4.72, 6.93] | 2.29 [1.95, 2.64] | 2.42 [2.01, 2.78] | 0.40 [0.20, 0.62] | 1.83 [1.58, 2.10] |
| Move goal object elsewhere | 656 | 4.88 [4.16, 5.66] | 1.45 [1.13, 1.77] | 1.84 [1.56, 2.13] | 0.85 [0.69, 1.02] | 1.43 [1.20, 1.69] |
| Heat, cool or clean the goal object | 893 | 8.81 [7.79, 9.75] | 2.44 [2.02, 2.88] | 3.15 [2.82, 3.47] | 1.10 [0.86, 1.32] | 1.72 [1.47, 1.95] |

For Qwen the completing move is among the largest ALFWorld cells, larger than a move elsewhere. For Llama it is smaller than other cells on the two-candidate scale but clearly positive, especially on yes/no. ALFWorld has no analogue of the negative sign. The analogy is imperfect in one way that may matter: in ALFWorld the missing precondition is about the goal object itself (not holding it), which the aligned task names; in TextCraft it is about an ingredient that neither task line names, while the aligned task names the product.

## 7. Interpretation

**For Qwen3-8B, "much smaller in TextCraft" is mostly a property of the scale and the population, with a real but scale-dependent residual.** In order of size:

1. **Reply format.** More than half of the ALFWorld two-candidate effect (3.60 of 6.12 nats) is the goal raising the probability of the success sentence's wording after its first word. TextCraft's design cancels that term. This alone accounts for most of the 6.5-fold shrinkage on the two-candidate scale, and it is why the two-candidate scale should not be compared across environments.
2. **Step 0.** ALFWorld's step-0 items carry 7.5 nats each on the first token, four times the later-step value, and TextCraft's main set has none. Against TextCraft's empty-history items they are comparable (77% on the first token, 183% on yes/no).
3. **A residual of about two on the first token at later steps,** at matched history length (1.81 against 0.97) and matched baseline (2.09 against 0.97). It does not appear on the yes/no question (2.05 against 2.92 at matched lengths) or on the probability scales. Whether the model's decision moves less in TextCraft therefore depends on which readout counts as the decision.

**The sign structure suggests the goal acts differently in the two environments.** In ALFWorld the aligned task lifts success above the neutral task; in TextCraft the swapped task pushes it toward failure and the aligned task does little. One reading, offered as a hypothesis and not tested here, extends the task-as-evidence framing: in ALFWorld, failures turn on whether the object is at hand, and a task naming the object reads as evidence that it is; in TextCraft every item is already listed in the recipes, and a task gives no evidence about inventory counts, so the main thing a task line can do is mark actions outside its tree as off-task, and the model treats off-task crafts as likely to fail. That reading fits the large TextCraft "too few" effect only partly, so it should be tested rather than assumed.

**For Llama-3.1-8B the effect is absent on every scale, at step 0, at matched lengths and at a matched baseline,** with one exception: a small positive yes/no effect on the empty-history items. Scale and population do not explain it. What separates the environments for Llama is competence: it barely tells TextCraft successes from failures (AUC 0.67 to 0.69, and 0.61 on the main failures alone). A model whose predictions do not depend on the inventory state may have little for the goal line to shift. The separate Llama investigation should settle this.

**History–goal agreement is unlikely to explain the ALFWorld excess.** TextCraft's own agreement contrast is negative, ALFWorld's largest effect has no history, and the counterfactual-task check leaves the effect unchanged. It cannot be ruled out completely, because no ALFWorld measure captures how well an agent history fits the goal object.

**The yes/no question does not rescue Llama.** ALFWorld's yes/no effect for Llama (2.21) is larger than its two-candidate first-token effect (0.68), and both are clearly positive; in TextCraft all its scales are near zero. So for Llama, unlike Qwen, the yes/no question does not rescue the cross-environment comparison.

## 8. What remains open, and what would settle it

- **The first-token residual for Qwen.** Is the factor of two at later steps on the first token a difference in how the goal acts, or an artefact of the first-token words ("You"/"Nothing" against "Craft"/"Could") and of ALFWorld's question mentioning "Nothing happens."? A CPU-only check is not possible; it needs scoring.
- **Lift against suppression.** Whether the goal acts as evidence in ALFWorld and as an off-task marker in TextCraft is a hypothesis from the sign structure, not a tested account.
- **The negative goal-role never-obtained cell.** It is negative on the two-candidate scale for Qwen only and on all scales for Llama; whether it reflects the cell's extreme baseline or something about naming the product while its ingredient is missing is open.
- **Llama.** Left to the separate investigation.

**The follow-up that would settle most of this** is one scoring run (GPU, not run here) that puts the two environments on identical readouts: score both with the same yes/no question wording and with a reply format matched across environments (in ALFWorld, a failure reply that names the object, or both replies cut to their first word; in TextCraft, the current prefix arm), with the neutral task, on ALFWorld later-step items and TextCraft main items, plus TextCraft's empty-history items with the `empty_start` system line. If Qwen's first-token gap closes under a matched format and question, the cross-environment difference for Qwen is measurement plus step 0. If it remains, and the TextCraft effect stays "swapped suppresses" while ALFWorld's stays "aligned lifts", the two environments engage different goal mechanisms, which an ALFWorld variant whose failures turn on visible state (for example, actions whose failure the history shows directly, as in the "receptacle seen without the object" items) could test further. Adding an ALFWorld-style object-naming failure reply would also give an exact estimate of how much of ALFWorld's two-candidate effect is the tail term under a matched format.

## 9. "Why did Llama do better when Qwen3 is supposed to be the better model?" (added 2026-11-21)

This section answers that question with the data. It is exploratory and post hoc, like the rest of the note. The numbers come from `scripts/xenv/competence.py` and are stored under `competence_vs_effect` in the metrics file.

The short answer is that Llama does not do better on any ruler where "better" means a better world model. It looks better on two rulers that mostly measure something else:

- **In ALFWorld,** Llama's two-candidate effect (1.65 nats) is smaller than Qwen's (6.12). Section 2 showed that most of Qwen's two-candidate effect is the success sentence's tail. On the shared readouts the two models are close: yes/no 2.21 against 2.80 nats, and the share of failing items pushed toward Yes 0.896 against 0.887. On the practical rulers Llama is far *worse*: its answers change on 16% of failing items, against 0.2% for Qwen.
- **In TextCraft,** Llama shows no positive effect while Qwen does. But Llama also barely separates working from failing actions there, so there is little prediction for the goal to bias.

### (a) Competence next to every effect scale, both models, both environments

Competence here is how well the model's score under the neutral task ("explore the room" or "explore the crafting commands") separates the determined successes from the failures. It is reported as an AUC: the chance that a random success scores above a random failure, where 0.5 is chance and 1 is perfect. Next to it are the mean scores under the neutral task, which show where the model's baseline sits. The effect scales are those of Section 2. The **share toward Yes** is the share of failing items whose yes/no log-odds is higher under the aligned task than under the swapped task; 0.5 means no effect. **Flipped No → Yes** counts failing items answered No under the swapped task and Yes under the aligned task. TextCraft's numbers use its 1,500 main failures and 600 successes; with the usefulness failures added, as its design does, the AUCs are 0.87 and 0.85 for Qwen and 0.68 and 0.67 for Llama.

| | Qwen ALFWorld | Llama ALFWorld | Qwen TextCraft | Llama TextCraft |
|---|---|---|---|---|
| **Competence** | | | | |
| AUC under the neutral task, two-candidate | 0.93 | 0.87 | 0.86 | 0.61 |
| AUC under the neutral task, yes/no | 0.99 | 0.97 | 0.85 | 0.61 |
| Mean two-candidate L, failures / successes | −23.4 / +15.3 | −3.7 / +2.8 | −9.8 / +7.3 | −2.0 / −0.3 |
| Mean yes/no L, failures / successes | −26.5 / +3.1 | −4.4 / +1.8 | −12.4 / +6.3 | −3.6 / −2.4 |
| Answered Yes under the neutral task, failures / successes | 0.2% / 58.9% | 5.3% / 85.3% | 18.7% / 72.5% | 9.9% / 11.3% |
| **Effect** | | | | |
| Two-candidate ΔL, nats | 6.12 [5.51, 6.75] | 1.65 [1.50, 1.78] | 0.94 [0.64, 1.25] | −0.19 [−0.30, −0.09] |
| First-token ΔL, nats | 2.52 | 0.68 | 0.97 | −0.16 |
| Yes/no ΔL, nats | 2.80 [2.60, 3.01] | 2.21 [2.04, 2.39] | 2.92 [2.56, 3.31] | −0.15 [−0.21, −0.09] |
| Standardized, two-candidate / yes/no | 1.02 / 0.88 | 0.85 / 1.12 | 0.20 / 0.49 | −0.14 / −0.17 |
| Two-candidate Δp, points | 5.9 | 16.4 | 3.6 | 0.4 |
| Yes/no Δp, points | 0.2 | 16.2 | 6.7 | −0.3 |
| Share toward Yes | 0.887 [0.859, 0.912] | 0.896 [0.871, 0.920] | 0.661 [0.634, 0.688] | 0.407 [0.375, 0.439] |
| Flipped No → Yes | 0.2% [0.1, 0.3] | 15.9% [14.0, 18.1] | 7.5% [6.0, 9.0] | 0.7% [0.3, 1.2] |
| Answered Yes on failures, swapped → aligned | 0.0% → 0.2% | 5.6% → 21.0% | 8.4% → 15.3% | 4.0% → 3.5% |

**In ALFWorld both models are competent, and Qwen is the more competent and far more confident.** Its yes/no score on failures averages −26.5 nats under the neutral task, against Llama's −4.4. The two models move their scores by similar amounts on the shared readouts: about 2 to 3 nats on yes/no, toward Yes on about 89% of items. What differs is where those moves start. Qwen starts so deep in "No" that a 2.8-nat push changes almost no answers. Llama starts close enough to the boundary that a 2.2-nat push turns 16% of failing items into Yes. So in ALFWorld Llama's goal bias is about as strong as Qwen's in log-odds and much larger in consequences. Its smaller two-candidate number comes from the reply tail and from its lower confidence, not from a sounder judgement.

**In TextCraft only Qwen remains competent.** Llama answers Yes to 9.9% of failures and 11.3% of successes under the neutral task, and its mean yes/no L on successes is −2.4. It predicts failure for nearly everything, whether the action works or not. Its share toward Yes is below one half (0.41): the goal line moves slightly more items toward No than toward Yes, by small amounts (mean −0.15 nats).

### (b) Across models in ALFWorld: does the effect track competence or baseline confidence?

This part uses ALFWorld only, with 11 distinct models:

- the two reference models (Qwen3-8B and Llama-3.1-8B, from their confirmatory scores);
- the four wave-A family models (Gemma-2-9B, Mistral-7B, Llama-3.2-3B, Llama-3.2-1B; scores from `family_score_*`, ICE `text-wm-fix23`);
- the Qwen3 size series at 0.6B, 1.7B, 4B, 14B and 32B (`size_score_*`, archive `text-wm-fix18`).

The size series' own 8B run is a rescore of the reference model; it matches it to two decimals and is left out of the pooled set so that the model is not counted twice. All 11 were scored on the same 4,666 failures and 292 successes.

| Model | AUC two-cand. | AUC yes/no | Yes/no L on failures, neutral | Answered Yes on failures, neutral | ΔL two-cand. | ΔL yes/no | Share toward Yes | Yes/no Δp | Flipped No → Yes |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 0.93 | 0.99 | −26.5 | 0.2% | 6.12 | 2.80 | 0.89 | 0.2 | 0.2% |
| Llama-3.1-8B | 0.87 | 0.97 | −4.4 | 5.3% | 1.65 | 2.21 | 0.90 | 16.2 | 15.9% |
| Gemma-2-9B | 0.90 | 0.99 | −8.4 | 1.0% | 0.80 | 1.23 | 0.75 | 2.5 | 3.0% |
| Mistral-7B | 0.77 | 0.94 | +0.5 | 55.9% | 1.06 | 2.67 | 0.88 | 30.2 | 35.8% |
| Llama-3.2-3B | 0.80 | 0.88 | −0.9 | 16.3% | 0.98 | 0.70 | 0.84 | 14.0 | 21.2% |
| Llama-3.2-1B | 0.60 | 0.68 | −0.3 | 13.1% | 0.31 | 0.09 | 0.60 | 2.0 | 1.4% |
| Qwen3-0.6B | 0.72 | 0.55 | +7.0 | 100% | 0.89 | 0.12 | 0.62 | 0.0 | 0.0% |
| Qwen3-1.7B | 0.81 | 0.90 | +14.1 | 99.9% | 2.49 | 2.08 | 0.80 | 0.1 | 0.1% |
| Qwen3-4B | 0.93 | 0.94 | −21.2 | 2.2% | 7.49 | 2.27 | 0.49 | 5.5 | 6.4% |
| Qwen3-14B | 0.90 | 0.99 | −23.6 | 0.6% | 4.33 | 3.18 | 0.70 | 4.6 | 4.5% |
| Qwen3-32B | 0.91 | 0.97 | −6.8 | 2.4% | 2.35 | 1.55 | 0.71 | 9.2 | 8.9% |

Per-model game-clustered intervals for ΔL, the share and the flips are in the metrics file.

Across models I report Spearman rank correlations: how well the models' order on one measure matches their order on another, from −1 to +1. They are given for all 11 models and for the 8 that pass the families competence rule. That rule requires a neutral yes/no AUC of at least 0.70, Yes on at most 95% of failures and on at least 5% of successes; its fourth criterion, the median yes/no probability mass, is not recomputed here. The 8 are the reference models, Gemma, Mistral, Llama-3.2-3B, and Qwen3 at 4B, 14B and 32B. The permutation p treats the models as exchangeable draws, which they are not; it is a descriptive guide only. The table shows the correlations that answer the question; the full grid is in the metrics file.

| Effect measure | Against yes/no AUC, all 11 | Against two-candidate AUC, all 11 | Against yes/no AUC, competent 8 | Against neutral yes/no L on failures, competent 8 | Against answered Yes on failures (neutral), competent 8 |
|---|---|---|---|---|---|
| Two-candidate ΔL | +0.53 | +0.79 (p 0.006) | +0.31 | −0.67 | −0.50 |
| Yes/no ΔL | +0.75 (p 0.008) | +0.65 | +0.55 | −0.52 | −0.45 |
| Share toward Yes | +0.32 | +0.10 | −0.02 | +0.33 | +0.26 |
| Yes/no Δp | +0.20 | +0.06 | −0.71 | +0.90 (p 0.005) | +0.95 (p 0.002) |
| Flipped No → Yes | +0.15 | +0.05 | −0.81 (p 0.02) | +0.93 (p 0.002) | +0.98 (p 0.001) |

Three descriptive patterns, each with its limits:

1. **Whether there is an effect tracks competence.** The models with the smallest yes/no effects are the two that cannot do the task: Llama-3.2-1B (AUC 0.68; 0.09 nats) and Qwen3-0.6B (answers Yes to everything; 0.12 nats). Across all 11 models, the log-odds effects rise with AUC (yes/no ΔL against yes/no AUC +0.75; two-candidate ΔL against two-candidate AUC +0.79). Among the 8 competent models this relation weakens (+0.55 and +0.79, neither with a clear p). In short, the nat-scale relation is carried mostly by the near-incompetent models. Every competent model shows a positive effect on both scales. No model and no environment combines high competence with no effect.
2. **How many answers the goal changes tracks baseline confidence, not competence.** Among the competent models, the yes/no probability shift and the No → Yes flips follow how close the model's failures sit to the Yes/No boundary under the neutral task (+0.90 to +0.98). Mistral, which already answers Yes to 56% of failures, flips 36%. Qwen3-8B, at −26.5 nats, flips 0.2%. These measures fall with competence (−0.71 and −0.81), because the most competent models are also the most confident. This is the ALFWorld version of the Qwen-against-Llama contrast in (a).
3. **The share toward Yes tracks neither.** It is 0.75 to 0.90 for most competent models, regardless of AUC or baseline (all correlations within ±0.36). The exceptions are within the Qwen3 size series: 4B at 0.49, 14B at 0.70, 32B at 0.71. Qwen3-4B has a mean yes/no effect of 2.27 nats while moving only half the items toward Yes, so large moves on a minority of items carry its mean. Within the six Qwen3 sizes, no relation is reliable with six points.

**What this cannot show.** About a dozen models differ at once in developer, training data, size, tokenizer, chat template and post-training, so a correlation across them cannot isolate competence as a cause. Competence and confidence are themselves correlated (the competent models are the confident ones), so this design cannot separate their roles. The competent subset is defined from the same yes/no scores that the effects use. Each model has one scoring run with one example set. The AUC rests on 292 successes that are built differently from the failures. The analysis covers ALFWorld only: TextCraft has two models, which is too few for any cross-model statement.

### (c) Llama's absent TextCraft effect: robustness, or nothing settled to bias?

Read it as the absence of a settled prediction, not as robustness. The evidence:

- **No settled prediction.** Under the neutral task Llama separates TextCraft's working and failing actions barely above chance (AUC 0.61 on the main failures, 0.67 to 0.68 with the usefulness failures). It answers Yes to 11% of actions that work and 10% of those that fail. Earlier TextCraft analyses found that it ranks used-up failures above real successes (AUC 0.31) and that it cannot do the counting items (AUC 0.42).
- **The same pattern appears elsewhere.** Across ALFWorld models, the only near-zero yes/no effects come from the two models without working state judgements (Llama-3.2-1B, Qwen3-0.6B). No competent model in either environment shows a null effect.
- **The goal line is not ignored.** Both goal lines lower Llama's TextCraft yes/no score relative to the neutral task (aligned −0.60, swapped −0.45 nats). Its main-set effect is small and negative rather than zero. On the 100 empty-history items its yes/no effect is +1.09. So the goal line still reaches its output. It just has no consistent state judgement to push toward success.

Robustness would mean a model that tracks the state (high AUC) and is not moved by the goal. That combination does not occur in any model or environment in this project's data. Calling Llama "robust" in TextCraft would mistake an unformed prediction for a resistant one.

## 10. Naming the object against completing the task, in ALFWorld (added 2026-11-21)

This section is exploratory and post hoc; the numbers come from `scripts/xenv/completion.py`, stored under `completion_vs_intermediate` in the metrics file. In TextCraft the aligned goal names one item, and crafting that item completes the task, so "the action names the goal object" and "the action completes the task" cannot be separated (Section 6). In ALFWorld they can be partly separated, because several kinds of action act on the goal object but stand at different places in the task's plan. Every RQ2 history leaves the goal object untouched, so every move, heat, cool and clean item fails because the object was never picked up. Grouping the 4,666 failures by the action's place in the plan:

- **Completing** (668 items): move the goal object to a receptacle of the task's target type. This is the last step of every placement, pick-two and heat/cool/clean task, and on a plain placement task it would finish the task.
- **Intermediate, advancing** (408): heat, cool or clean the goal object when the task asks for that transformation.
- **Take** (2,449): take the goal object from a receptacle, the first step of every task.
- **Off-plan move** (656): move the goal object to a receptacle that is not of the target type.
- **Off-plan transformation** (485): heat, cool or clean when the task asks for another transformation or none.

The ALFWorld data have no "go to" action carrying the object, because RQ2 actions all name the goal object.

The cleanest contrast puts the completing move against the transformation step **within heat, cool and clean tasks** (393 against 408 items). Matching reweights the transformation items to the completing items' mix of failure reason (agent elsewhere, not holding the object, never moved), step (0 or later) and task kind (heat, cool or clean); every completing item has a match. The other contrasts are matched on failure reason and step. Intervals come from the same game resamples for both groups, so the differences are paired by game.

**Matched difference, completing minus transformation step, heat/cool/clean tasks** (negative means a smaller effect on the completing action, a "completion penalty"):

| Measure | Qwen: completing / transformation / difference | Llama: completing / transformation / difference |
|---|---|---|
| Two-candidate ΔL | 7.57 / 8.14 / −0.57 [−1.77, 0.61] | 0.74 / 1.18 / −0.43 [−0.76, −0.07] |
| First-token ΔL | 3.67 / 2.41 / +1.26 [0.75, 1.83] | 0.73 / 0.74 / −0.02 [−0.17, 0.15] |
| Yes/no ΔL | 3.26 / 3.62 / −0.36 [−0.92, 0.18] | 2.36 / 1.72 / +0.64 [0.42, 0.88] |
| Yes/no Δp, points | 0.1 / 0.4 / −0.3 [−0.9, 0.1] | 13.4 / 20.6 / −7.2 [−10.5, −3.7] |
| 2×2 goal-directed part | +0.56 / 0.00 / +0.56 [−0.59, 1.76] | −0.81 / +0.32 / −1.13 [−1.38, −0.85] |
| 2×2 mention part | 7.01 / 8.14 / −1.14 [−2.21, 0.02] | 1.56 / 0.86 / +0.70 [0.38, 0.99] |
| Object-free (described) task ΔL | 5.00 / 5.22 / −0.22 [−1.38, 0.95] | 0.93 / 1.06 / −0.14 [−0.46, 0.17] |
| Yes/no, aligned minus neutral | 3.18 / 5.42 / −2.23 [−2.89, −1.64] | 2.32 / 2.17 / +0.15 [−0.08, 0.38] |

The 2×2 (RQ2's object-mention design, scored only on these two groups, 1,076 items) splits the aligned-minus-swapped effect into two parts:

- **The mention part** is the effect of naming the action's object in a task the action does not advance.
- **The goal-directed part** is what the action advancing the named task adds on top.

For a completing move, the mention task changes the destination ("put the mug in the cabinet" when the action moves it to the fridge). For a transformation step, it changes the transformation (heat becomes clean).

**Other matched contrasts** (completing minus the other group; two-candidate / first token / yes/no, nats):

- **Against take actions**, matched on reason and step (76% of completing items have a match): Qwen +1.33 / +0.12 / −0.09; Llama −1.42 / −0.09 / −0.34 [−0.67, −0.01].
- **Against off-plan moves** (same verb, same reasons and steps): Qwen +1.85 / +1.00 / +1.39 [0.81, 2.09]; Llama −0.01 / +0.56 / +1.26 [0.91, 1.57].
- **Against off-plan transformations:** Qwen −1.60 / +1.39 / +0.49; Llama −0.35 / +0.38 / +0.49.
- **The object-free task keeps a similar share of the named effect on completing actions as on others:** Qwen 66% on completing moves, 66% on transformation steps, 61% on takes. For Llama the described task gives *more* than the named one on completing moves (139% [109, 182]), against 96% on transformation steps and 85% on takes.

**Is there a completion penalty?**

- **For Qwen3-8B, no.** No decision scale shows a reliable penalty. The two-candidate and yes/no differences against the transformation step are small, with intervals across zero. The first token shows a completion *bonus* (+1.26). Against an off-plan move, the completing move has the larger effect on every scale. Its 2×2 goal-directed part is positive (+0.56 [0.17, 0.95]). One asymmetry exists, but it is not a penalty on the effect: the aligned task lifts the transformation step much further above the neutral task than it lifts the completing move (yes/no 5.42 against 3.18). The swapped task does the same, so the aligned-minus-swapped difference is unchanged.
- **For Llama-3.1-8B, a small one, on some scales only.** The goal-directed part of the 2×2 is negative on completing moves (−0.81 [−1.02, −0.60]) and positive on transformation steps (+0.32), a difference of −1.13. Llama's overall 2×2 goal-directed part, −0.18 in `rq2_results.md`, is the average of these two opposite pieces. Once the object's mention is held fixed, a task that the move would complete makes Llama slightly *less* willing to predict success than a task it would not advance. The two-candidate scale shows a small matched penalty (−0.43). The yes/no question shows the opposite (+0.64), and the first token shows nothing. The penalty is outweighed by a larger mention part on completing moves (1.56 against 0.86), so the total effect on completing moves stays positive.

So ALFWorld has no general completion penalty. Llama's small, scale-dependent penalty on the goal-directed part has the same sign as its negative TextCraft goal-role cell (Section 6), which is suggestive but not a demonstration. In TextCraft the aligned-minus-swapped contrast mixes mention and completion. In ALFWorld the penalty appears only after mention is removed, and only on two of the four scales.

**What matching cannot remove.** The completing move and the transformation step differ in verb, so their success replies differ ("You move the mug 1 to the fridge 1." against "You heat the mug 1 using the microwave 1."). That matters for the two-candidate and tail-sensitive scales, though not for yes/no. The 2×2's mention task changes a different thing for each group (the destination against the transformation), so the goal-directed parts compare two manipulations that are not identical. The take contrast also crosses failure semantics: a take fails because the agent is elsewhere or the object was not seen there, while a move fails because the agent does not hold the object. Matching on the recorded failure reason does not equalize those, since every completing move is also a never-obtained failure whatever reason is recorded. The step-0 items differ by group in which receptacles the agent could plausibly reach. Finally, these groups come from the generators of `rq2_results.md` ("How alternative actions were generated"), so their mix is set by design choices, not by what an agent would propose.

## Reproducing

From the repository root, with the ALFWorld code importable and the archive and TextCraft data available (the TextCraft items are `wf/textcraft/data/items.jsonl` on branch `textcraft`; the scores are `text-wm-fix24/runs/textcraft_score/shard_0{0..3}_of_04/scoring.jsonl` on ICE):

```
python scripts/xenv/compare.py --archive <text-wm-runs-archive> \
    --tc-items <items.jsonl> --tc-scores <the four shard scoring.jsonl files> \
    --out notes/textcraft_vs_alfworld_metrics.json
```

It runs on CPU in about 15 minutes. Sections 9 and 10 are added to the same metrics file by `scripts/xenv/competence.py` (which also needs `--families-dir`, a folder holding the four `family_score_<model>` scoring files from ICE `text-wm-fix23`, named `<model>.jsonl`) and `scripts/xenv/completion.py`; run them after `compare.py`, one at a time. The ALFWorld runs read are `text-wm-fix11` (`rq2_score`), `text-wm-fix12` (`rq1_score`), `text-wm-fix13` (`rq1x_score`) and `text-wm-fix17` (`rq1v_score`, and the collected items through `wf.rq2.load_items`).
