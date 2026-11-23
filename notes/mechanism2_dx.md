# Mechanism battery: how robust are the key contrasts? (exploratory, post hoc, 2026-11-22)

**Status.** Everything in this note is exploratory and post hoc. It was planned after the battery's frozen status line had been read (tag `mechanism2-design-frozen-2026-11-22`, commit `6eab8af`; `mech2_analysis`, ICE job 7244859) and before any of the numbers below were computed. The list of analyses in the next section was written and committed before the computation. Nothing here changes a frozen verdict: the verdicts are the output of the frozen rules in `wf/mech2.py`, reported in [mechanism2_results.md](mechanism2_results.md). No model was run; the note re-reads the saved scores.

## Analyses, decided before computing

Models: Qwen3-8B and Llama-3.1-8B (the primary models), and Mistral-7B-Instruct-v0.3 as a descriptive secondary model wherever its arms exist (it was not scored on the neutral, desire_keep, mention_exists or placebo arms, and only on the 1,000 sample items apart from P6's arms). Key contrasts: P3 (mention_ignore − swapped), P4 (described − swapped), P7 (negated − swapped), P8 (negated − affirmed) and M (the mean of negated and affirmed, minus swapped), with the aligned effect (aligned − swapped) as the reference, in both formats (two-candidate and yes/no). Each estimate is a mean over items with a 95% percentile interval from 2,000 bootstrap resamples of whole games; r is the mean contrast over the mean aligned effect on the same items and in the same format. Nothing is corrected for multiple comparisons.

0. **Independent recompute.** From the raw scoring files (SHA-256 checked against ICE), with this note's own contrast and bootstrap code (the items are rebuilt with the frozen item functions of `wf/mech2.py`), recompute the estimates of P1 to P8, M and its halves, the negation placebo and P8 net of the placebo for every model and format, and report the largest absolute difference from `metrics.json`.
1. **Per game.** For each key contrast: the range of the estimate and of r when any one game is left out; the share of games (with at least three items) whose own mean is above zero; the share of the contrast's total carried by its largest single game and by its five largest.
2. **Per task type.** Estimate, interval and r (over the aligned effect on the same type's items) for each of the six task types; any type whose sign differs from the pooled estimate is named.
3. **Heavy tails.** Mean, median and 10% trimmed mean; the share of items above zero, above +1 nat and below −1 nat; the share of the summed positive contributions carried by the top 10% of items. The same for the aligned effect.
4. **Step 0 against later steps.** On the sample's 111 step-0 items, its 889 later items, and all 514 step-0 items: estimate and r (over the aligned effect on the same subset); and the share of the sample's total contrast carried by its 111 step-0 items, against their 11.1% share of items.
5. **Agreement between formats.** Per item, the Spearman correlation between a contrast's two-candidate and yes/no values, and the share of items whose two values have the same sign; per game, the Pearson correlation of the two formats' game means; r in both formats side by side.
6. **The 293 items whose history does not show the receptacle,** against the 210 whose history shows it without the object: P7, P8, M, M's two halves, the placebo and P8 net of the placebo, both formats, with the difference between the groups; and, within the one failure reason with both groups populated (contents or open state), P8 and M.
7. **The 87 items whose swapped task targets the action's receptacle type:** the placebo, P8 and P8 net of the placebo on the 87 and on the other 416, both formats (Qwen and Llama; Mistral has no placebo).
8. **P3 and word overlap.** P3's two lines both contain the swapped task line, and "Ignore the {obj}." adds the object's name, which every action contains; so any overlap that varies across items comes from the swapped line. For each sample item: (i) the swapped line names the action's receptacle type; (ii) the swapped line contains the action's verb as a word; (iii) the number of word types the swapped line shares with the action string, apart from articles, prepositions and numbers (0, 1, 2 or more); (iv) the same count for the real task line, read against the aligned effect; (v) the object's name already appears in the history. P3 and r (over the aligned effect on the same group) by each, for Qwen (the question asked), with Llama and Mistral alongside; and P3 by the action's verb.
9. **Practical size in the yes/no format.** The share of sample items answered Yes (yes/no log-odds above zero) under each arm, and the share of items that move from No under the swapped task to Yes under the arm.

**Departures from the list, made while computing, all labelled where they apply.** (a) In analysis 3, "the share of the summed positive contributions" is computed in the direction of the mean: for P8, whose mean is negative, it is the share of the summed negative contributions carried by the 10% most negative items. (b) In analysis 8, feature (v) was first computed on the history including its task line, which names the object on every item; it is computed on the history without the task line. (c) Feature (iii) takes only the values 0 and 1 (no swapped line shares two words with the action), and feature (iv) splits the items exactly as (iii) does, because the real and swapped lines differ only in the object's name, which every action contains. So (iv) adds nothing and is reported only in the metrics file.

## Results

Every number below is exploratory and post hoc, from [mechanism2_dx_metrics.json](mechanism2_dx_metrics.json) (written by `scripts/mech2_dx.py`). Intervals here come from this note's own bootstrap (seed 20261122), so they can differ from the frozen analysis's intervals in the second decimal; point estimates are identical. For the frozen verdicts and the confirmatory-style reading, see [mechanism2_results.md](mechanism2_results.md).

### Summary

1. **The frozen numbers reproduce exactly.** An independent recompute of P1 to P8, M, M's halves, the placebo and P8 net of it gives the same estimates as `metrics.json` for all three models and both formats (largest difference 1.1e-14 nats).
2. **The key contrasts are not carried by a few games or task types.** For Qwen, P3, P4, P8 and M keep their sign in all six task types, in at least 96% of games, and within 0.2 nats when any one game is dropped; no game carries more than 4% of any of them. Llama's P4 and P8 are as stable, and its P7 is negative in 86% and 76% of games; its P3 is smaller and less uniform (positive in 72% and 77% of games; about zero for pick two and clean in the two-candidate format). The one fragile key number is Qwen's two-candidate P7, a near-zero mean made of opposite parts.
3. **No heavy tails that change a reading.** Means and medians agree closely for P3 and M (Qwen P3: mean 6.43, median 5.87; 92% of items above zero). Two contrasts have longer tails, both mainly from step-0 items: Qwen's P4 in the yes/no format (mean 2.86, median 1.50; the top 10% of items carry half the total) and Qwen's P8 (mean −11.83, median −9.45).
4. **Step 0 against later steps.** The goal effect is larger per item at step 0 (Qwen 10.75 against 5.67 nats), but in this sample, whose items are 11% step-0, at least half of every contrast's total comes from later steps. Qwen's P3 is about the whole goal effect at step 0 and later (r 0.89 and 1.06, two-candidate). Qwen's format-dependent P7 is explained by step: strongly negative at step 0 in the two-candidate format (−7.50), positive later in both formats (r 0.23 and 0.97).
5. **The formats agree on sign, not on every size.** Item-level rank correlations between formats are moderate (0.40 to 0.73); about 90% of items have the same sign in both formats for Qwen's P3, P8 and M.
6. **The receptacle split.** Where the history does not show the receptacle, the user's report moves Qwen's reading about twice as much (P8 −14.85 against −7.61), as reading the report as evidence predicts. But within one failure reason the two-candidate difference vanishes, so this is partly the failure-reason confound.
7. **The placebo holds on the 87 swapped-target items.** Qwen's placebo is about zero there (+0.00 and −0.50), slightly below its value on the other 416 items, and far from the band's edge. P8 does not differ between the two groups.
8. **Qwen's P3 does not grow with word overlap between the line and the action.** It does not depend on whether the swapped line names the action's receptacle type or shares a word with the action. It is smaller (r 0.34) on the 81 items whose swapped line contains the action's verb, all clean, cool or heat actions on tasks of the same transformation. This analysis cannot separate salience from name overlap; only P4 can, and P4 is large.
9. **Practical size.** Qwen answers "Yes" to essentially none of these failing actions under any line (0.0% aligned, 6.4% affirmed). Llama's answers move: 22.4% Yes under the aligned task against 6.1% swapped, 48.3% under the affirmed report.

### 0. Independent recompute

The scores were copied from ICE (`text-wm-fix36/runs/mech2_score`, eight `scoring.jsonl` files) and the items from `text-wm-fix17/runs/rq12_transitions_s0..s2`; all eleven SHA-256 hashes matched the files on ICE, and the item rebuild gives the battery's populations exactly (4,666 primary items, 1,000 in the sample, 514 at step 0, 1,403 in the union, 746 in P5, 503 in P7, of which 293 not shown and 87 with the swapped target at the receptacle). From the raw scores, with this note's own contrast code, the estimates of P1 to P8 in both formats, M and its two halves, and (Qwen and Llama) the placebo and P8 net of it, 26 estimates per primary model and 22 for Mistral, match `metrics.json`: the largest absolute difference is 1.1e-14 nats (Qwen), 1.8e-15 (Llama) and 6.2e-15 (Mistral).

### 1. Per game

For each key contrast: the range of the estimate and of r when one game is left out; the share of games with at least three items whose own mean is above zero (111 such games for P3 and P4, 79 for P7, P8 and M); and the share of the contrast's total carried by its largest game and by its five largest (in the direction of the total).

| Model, contrast | Leave-one-game-out estimate, two-cand.; yes/no | Games above zero, two-cand.; yes/no | Largest game; five largest, two-cand. |
| --- | --- | --- | --- |
| Qwen P3 | 6.35 to 6.53; 3.17 to 3.24 | 96%; 99% | 2.6%; 11.6% |
| Qwen P4 | 5.18 to 5.33; 2.81 to 2.90 | 98%; 100% | 2.9%; 12.7% |
| Qwen P7 | 0.19 to 0.41; 1.91 to 2.00 | 61%; 96% | 28.9%; 126% |
| Qwen P8 | −12.01 to −11.67; −7.04 to −6.72 | 0%; 1% | 2.7%; 11.8% |
| Qwen M | 6.07 to 6.32; 5.35 to 5.47 | 97%; 100% | 3.2%; 13.4% |
| Llama P3 | 0.46 to 0.53; 0.63 to 0.67 | 72%; 77% | 7.4%; 28.2% |
| Llama P4 | 1.78 to 1.82; 2.54 to 2.60 | 96%; 99% | 2.8%; 12.8% |
| Llama P7 | −1.76 to −1.61; −0.57 to −0.48 | 14%; 24% | 7.5%; 28.3% |
| Llama P8 | −5.08 to −4.87; −5.31 to −5.16 | 0%; 0% | 4.4%; 15.5% |
| Llama M | 0.76 to 0.83; 2.04 to 2.10 | 77%; 99% | 5.4%; 22.4% |
| Mistral P3 | 0.17 to 0.22; −0.21 to −0.15 | 53%; 50% | 15.6%; 70.1% |
| Mistral P4 | 1.12 to 1.17; 2.83 to 2.91 | 80%; 96% | 3.4%; 14.8% |
| Mistral P8 | −6.69 to −6.52; −5.74 to −5.61 | 0%; 0% | 3.7%; 12.6% |

No single game moves any key estimate by more than 0.2 nats, and the r ranges are as narrow (Qwen P3 1.02 to 1.05). The only contrast whose total depends on a handful of games is Qwen's two-candidate P7. Its mean (+0.26) is a small net of large opposite contributions, so one game carries 29% of it and the five largest more than all of it. This is the same contrast the frozen rules called format-dependent, and section 4 shows where its two parts come from. Mistral's P3 is near zero, and its game shares are unstable for the same reason.

### 2. Per task type

r is over the aligned effect on the same task type's items, two-candidate; the yes/no signs are noted where they differ.

| Model, contrast | Pick | Pick two | Clean | Heat | Cool | Examine |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen P3 | +7.73 (r 1.02) | +4.86 (1.63) | +4.07 (0.75) | +6.87 (0.96) | +6.82 (0.75) | +11.02 (2.30) |
| Qwen P4 | +6.16 (0.81) | +4.85 (1.63) | +4.92 (0.90) | +5.84 (0.81) | +5.75 (0.63) | +3.76 (0.78) |
| Qwen P7 | +0.31 (0.04) | −0.68 (−0.21) | −0.92 (−0.21) | +1.34 (0.27) | +0.77 (0.10) | +1.87 (0.38) |
| Qwen P8 | −12.43 (−1.64) | −11.26 (−3.47) | −10.49 (−2.39) | −14.18 (−2.85) | −15.17 (−2.06) | −8.75 (−1.76) |
| Qwen M | +6.52 (0.86) | +4.95 (1.52) | +4.32 (0.98) | +8.43 (1.69) | +8.36 (1.13) | +6.24 (1.26) |
| Llama P3 | +0.80 (0.47) | −0.03 (−0.02) | +0.14 (0.11) | +0.76 (0.42) | +0.77 (0.40) | +0.77 (0.41) |
| Llama P4 | +2.20 (1.30) | +1.84 (1.44) | +1.32 (1.07) | +2.10 (1.15) | +1.93 (1.00) | +1.74 (0.93) |
| Llama P7 | −2.71 (−1.11) | −3.18 (−1.85) | −1.21 (−0.67) | −1.25 (−0.44) | −0.70 (−0.22) | −0.86 (−0.37) |
| Llama P8 | −5.92 (−2.43) | −5.83 (−3.38) | −4.05 (−2.24) | −4.73 (−1.68) | −4.64 (−1.50) | −4.90 (−2.12) |
| Llama M | +0.25 (0.10) | −0.26 (−0.15) | +0.82 (0.45) | +1.12 (0.40) | +1.62 (0.52) | +1.59 (0.69) |

Qwen's P3, P4, P8 and M have the pooled sign in every task type and both formats; in the yes/no format P3's r runs from 0.84 (pick) to 1.94 (examine). Qwen's two-candidate P7 is negative for pick two and clean and positive elsewhere; in the yes/no format it is positive in all six types (r 0.46 to 1.10). For Llama, P4 and P8 keep their sign everywhere. P3 is about zero for pick two and clean in the two-candidate format (positive in all six in yes/no). P7 is negative in all six in the two-candidate format, with examine at +0.20 in yes/no. M is negative for pick two in the two-candidate format only. Mistral (metrics file) is the least uniform: its P3 changes sign across types in both formats, while P4 and P8 keep their sign everywhere.

### 3. Heavy tails

| Model, contrast, format | Mean | Median | 10% trimmed mean | Items above 0 | Above +1 nat | Below −1 nat | Top 10% of items' share, in the direction of the mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen aligned effect, two-cand. | +6.24 | +5.36 | +5.85 | 87.9% | 83.9% | 7.5% | 27.9% |
| Qwen P3, two-cand. | +6.43 | +5.87 | +6.15 | 92.1% | 87.0% | 5.2% | 25.9% |
| Qwen P3, yes/no | +3.21 | +3.25 | +3.15 | 93.6% | 83.3% | 1.6% | 22.8% |
| Qwen P4, two-cand. | +5.25 | +3.92 | +4.52 | 88.3% | 81.0% | 7.1% | 33.7% |
| Qwen P4, yes/no | +2.86 | +1.50 | +1.77 | 78.2% | 56.2% | 6.1% | 51.5% |
| Qwen P7, two-cand. | +0.26 | +0.92 | +0.68 | 61.4% | 48.3% | 32.4% | 36.8% |
| Qwen P7, yes/no | +1.97 | +2.00 | +2.03 | 82.3% | 68.0% | 6.4% | 26.0% |
| Qwen P8, two-cand. | −11.83 | −9.45 | −10.97 | 3.0% | 1.6% | 93.4% | 25.5% |
| Qwen P8, yes/no | −6.90 | −3.75 | −5.16 | 5.8% | 2.4% | 82.5% | 39.6% |
| Qwen M, two-cand. | +6.17 | +5.66 | +5.87 | 89.9% | 84.7% | 7.8% | 27.3% |
| Llama aligned effect, two-cand. | +1.60 | +1.40 | +1.47 | 82.0% | 57.9% | 6.0% | 31.7% |
| Llama P3, two-cand. | +0.49 | +0.45 | +0.48 | 63.6% | 34.2% | 14.6% | 38.4% |
| Llama P4, two-cand. | +1.80 | +1.60 | +1.70 | 85.7% | 64.9% | 2.6% | 28.0% |
| Llama P7, two-cand. | −1.70 | −1.41 | −1.58 | 28.0% | 16.3% | 56.3% | 32.8% |
| Llama P8, two-cand. | −4.98 | −4.95 | −4.96 | 3.4% | 1.6% | 92.8% | 19.8% |
| Llama M, two-cand. | +0.79 | +0.74 | +0.76 | 63.0% | 46.7% | 22.9% | 35.4% |

The key effects are broad shifts of most items, not averages driven by a tail. Qwen's P3 moves 92% of items upward, and its top tenth of items carries a quarter of the total, as the goal effect's own top tenth does (27.9%). Two contrasts are skewed. Qwen's yes/no P4 (median 1.50 against mean 2.86) is concentrated on step-0 items (section 4). Qwen's P8 (median −9.45 against mean −11.83) has a long tail of very large drops, also at step 0. Llama's means and medians agree within 0.3 nats everywhere.

### 4. Step 0 against later steps

Two-candidate unless marked; r over the aligned effect on the same subset. "Sample step 0" is the 111 step-0 items of the sample (50 of them in P7's population); "all step 0" is all 514 step-0 items (242 for P7, P8 and M); "share of total" is the share of the sample's summed contrast contributed by its step-0 items, against their 11.1% share of items (9.9% for P7's population).

| Model, contrast | Sample step 0 | Sample later | All step 0 | Step-0 share of total |
| --- | --- | --- | --- | --- |
| Qwen aligned effect | +10.75 | +5.67 | +11.04 | 19.1% (yes/no 30.4%) |
| Qwen P3 | +9.58 (r 0.89) | +6.04 (1.06) | +10.49 (0.95) | 16.5% |
| Qwen P3, yes/no | +3.75 (0.49) | +3.15 (1.44) | +3.81 (0.49) | 13.0% |
| Qwen P4 | +13.88 (1.29) | +4.17 (0.74) | +14.34 (1.30) | 29.4% (yes/no 49.7%) |
| Qwen P7 | −7.50 (−0.78) | +1.11 (0.23) | −6.75 (−0.67) | opposite sign |
| Qwen P7, yes/no | +0.09 (0.01) | +2.18 (0.97) | +0.64 (0.08) | 0.5% |
| Qwen P8 | −23.40 (−2.42) | −10.55 (−2.22) | −22.49 (−2.23) | 19.7% |
| Qwen M | +4.20 (0.43) | +6.39 (1.35) | +4.49 (0.45) | 6.8% |
| Qwen M, yes/no | +8.01 (0.90) | +5.14 (2.28) | +7.41 (0.90) | 14.7% |
| Llama aligned effect | +1.55 | +1.61 | +1.51 | 10.8% |
| Llama P3 | −0.65 (−0.42) | +0.63 (0.40) | −0.64 (−0.42) | opposite sign |
| Llama P4 | +1.72 (1.11) | +1.81 (1.13) | +1.73 (1.14) | 10.6% |
| Llama P7 | −3.50 (−2.31) | −1.50 (−0.63) | −3.32 (−2.27) | 20.5% |
| Llama P8 | −4.98 (−3.29) | −4.98 (−2.11) | −4.80 (−3.28) | 10.0% |
| Llama M | −1.01 (−0.66) | +0.99 (0.42) | −0.92 (−0.63) | opposite sign |

What this shows:

- *Qwen's goal effect is twice as large per item at step 0, but the battery's totals come mostly from later steps.* The sample holds step-0 items roughly in proportion to their share of the population. P3, M and P8 stand in about the same proportion to the goal effect at step 0 as later in the two-candidate format, so the "mention is enough" reading does not depend on step 0. In the yes/no format, P3 is half the goal effect at step 0 (r 0.49) and more than all of it later (1.44).
- *Qwen's P7 splits by step.* At step 0 (the agent has not moved, the receptacle has never been seen), the denial pulls the two-candidate reading well below the swapped task (−7.50). Later it leaves the reading above the swapped task in both formats (r 0.23 and 0.97). The pooled two-candidate P7 of +0.26 is the net of the two, which is why it is about zero, fragile across games, and format-dependent. The same step-0 pattern appears for P5 in the frozen secondary readouts (done below cancelled at step 0, above later; results note section 6).
- *Llama's mention effect is negative at step 0 and positive later* (P3 −0.65 against +0.63; M likewise), while its goal effect is the same size at both. So for Llama, naming the object without need helps only once the agent has a history.
- *Mistral* has content arms on the sample only, so its "all step 0" column equals its sample step-0 column (metrics file).

### 5. Agreement between formats

| Model | Contrast | Item-level Spearman, two-cand. against yes/no | Items with the same sign in both | Game-mean Pearson |
| --- | --- | --- | --- | --- |
| Qwen | aligned effect | 0.46 | 85.6% | 0.57 |
| Qwen | P3 | 0.53 | 90.1% | 0.72 |
| Qwen | P4 | 0.50 | 76.5% | 0.59 |
| Qwen | P7 | 0.40 | 67.2% | 0.53 |
| Qwen | P8 | 0.70 | 91.8% | 0.72 |
| Qwen | M | 0.49 | 89.7% | 0.65 |
| Llama | aligned effect | 0.63 | 85.9% | 0.57 |
| Llama | P3 | 0.63 | 76.1% | 0.69 |
| Llama | P4 | 0.64 | 86.5% | 0.57 |
| Llama | P7 | 0.46 | 69.4% | 0.39 |
| Llama | P8 | 0.72 | 96.4% | 0.74 |
| Llama | M | 0.42 | 66.4% | 0.27 |
| Mistral | P3 | 0.70 | 75.9% | 0.74 |
| Mistral | P4 | 0.73 | 79.3% | 0.85 |
| Mistral | P8 | 0.69 | 99.6% | 0.76 |
| Mistral | M | 0.71 | 60.6% | 0.73 |

The two formats rank items similarly but not identically. Agreement is highest for P8 (the claim's direction), where 92% to nearly 100% of items move the same way in both formats. It is lowest for the contrasts the design expected to differ by format: P7, where the two-candidate format can copy the receptacle's name from the success text, and M, whose halves weigh differently in the two formats. Mistral's M even has opposite pooled signs (−1.57 two-candidate, +0.88 yes/no) despite a rank correlation of 0.71, because in the two-candidate format its denial half is much larger than its affirmation half (+4.88 against +1.74). The two-format rule was built for this situation, and nothing here suggests it was too strict or too loose for the key contrasts: where the frozen rules decided a contrast, its sign agrees on most items.

### 6. The 293 items whose history does not show the receptacle

On P7's 503 items: 293 whose history never shows the receptacle's contents (the user's report is then the only information about it) and 210 whose history shows the receptacle without the object (the report repeats what the history already says). Difference is not shown minus shown, with a game-bootstrap interval.

| Model, contrast, format | Not shown (293) | Shown without it (210) | Difference |
| --- | --- | --- | --- |
| Qwen P7, two-cand. | +0.22 (r 0.04) | +0.31 (r 0.06) | −0.09 [−0.78, 0.58] |
| Qwen P7, yes/no | +2.15 (r 0.62) | +1.73 (r 0.81) | +0.42 [0.02, 0.82] |
| Qwen P8, two-cand. | −14.85 (r −2.80) | −7.61 (r −1.48) | −7.24 [−8.72, −5.72] |
| Qwen P8, yes/no | −10.21 (r −2.94) | −2.28 (r −1.07) | −7.93 [−9.34, −6.53] |
| Qwen M, two-cand. | +7.64 (r 1.44) | +4.12 (r 0.80) | +3.53 [2.69, 4.46] |
| Qwen affirmed − swapped, two-cand. | +15.07 | +7.92 | +7.15 [5.70, 8.68] |
| Qwen swapped − negated, two-cand. | −0.22 | −0.31 | +0.09 [−0.58, 0.78] |
| Qwen placebo, two-cand. | +0.55 | +1.25 | −0.70 [−1.36, −0.03] |
| Qwen P8 net of the placebo, two-cand. | −15.40 | −8.86 | −6.53 [−8.20, −4.89] |
| Llama P7, two-cand. | −2.14 (r −1.05) | −1.08 (r −0.41) | −1.06 [−1.55, −0.53] |
| Llama P8, two-cand. | −5.42 (r −2.67) | −4.36 (r −1.66) | −1.07 [−1.64, −0.45] |
| Llama P8, yes/no | −5.76 | −4.49 | −1.27 [−1.72, −0.83] |
| Llama M, two-cand. | +0.57 | +1.10 | −0.52 [−0.94, −0.10] |
| Llama affirmed − swapped, two-cand. | +3.29 | +3.28 | +0.01 [−0.50, 0.52] |
| Llama swapped − negated, two-cand. | +2.14 | +1.08 | +1.06 [0.53, 1.55] |

Within the one failure reason where both groups are populated (contents or open state: 57 not shown, 162 shown):

| Model, contrast | Two-candidate difference | Yes/no difference |
| --- | --- | --- |
| Qwen P8 | −0.79 [−2.77, 0.97] | −4.54 [−7.35, −2.31] |
| Qwen M | −0.35 [−1.49, 0.78] | +2.12 [1.10, 3.46] |
| Llama P8 | −1.61 [−2.61, −0.65] | −2.68 [−3.74, −1.66] |
| Llama M | −1.34 [−1.98, −0.74] | +0.09 [−0.48, 0.63] |

For both models the claim moves the reading more where the history is silent about the receptacle, which is what a reader treating the user's report as evidence would do. For Qwen the whole difference sits in the affirmation: "there is a bowl in the cabinet 1" lifts the reading about twice as much where the cabinet was never seen, while the denial does nothing in either group. For Llama the difference sits in the denial. The split is confounded with the failure reason: 230 of the 293 not-shown items fail because the agent is away from the receptacle, 162 of the 210 shown items on contents or open state. Within contents or open state, Llama's larger P8 where the history is silent survives in both formats. Qwen's survives only in the yes/no format, so part of Qwen's two-candidate difference is the failure-reason confound.

### 7. The 87 items whose swapped task targets the action's receptacle type, for the placebo

On these 87 items the placebo's claim ("there is no pan in the cabinet 1", with the swapped task "put a clean pan in cabinet.") bears on the swapped task itself, so it is not a pure placebo (design §14.5).

| Model, readout | The 87, two-cand. | The other 416, two-cand. | Difference | The 87, yes/no | The 416, yes/no | Difference |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen placebo | +0.00 [−0.92, 0.83] | +1.02 [0.66, 1.41] | −1.01 [−1.96, −0.10] | −0.50 [−1.19, 0.03] | +0.02 [−0.15, 0.20] | −0.53 [−1.18, 0.05] |
| Qwen P8 | −11.53 | −11.89 | +0.36 [−2.49, 3.11] | −6.39 | −7.01 | +0.62 [−1.79, 2.93] |
| Qwen P8 net of the placebo | −11.53 | −12.91 | +1.37 [−1.84, 4.58] | −5.88 | −7.03 | +1.15 [−1.24, 3.42] |
| Llama placebo | +0.27 [−0.25, 0.78] | +0.57 [0.35, 0.79] | −0.31 [−0.84, 0.22] | +0.16 [−0.56, 0.77] | +0.40 [0.15, 0.65] | −0.24 [−0.89, 0.40] |
| Llama P8 | −4.48 | −5.08 | +0.61 [−0.05, 1.23] | −4.34 | −5.42 | +1.07 [0.54, 1.59] |
| Llama P8 net of the placebo | −4.74 | −5.66 | +0.91 [0.06, 1.70] | −4.50 | −5.81 | +1.31 [0.46, 2.18] |

On the 87 items the placebo is slightly lower than on the others for Qwen, as expected if "there is no pan in the cabinet 1" reads as bad news for a pan task. Its r stays inside the band (+0.00, r 0.00; yes/no −0.50, r −0.20), and P8 is the same on both groups. For Llama the 87 items differ from the rest by less than half a nat on the placebo, and its P8 is somewhat smaller there. Leaving them out, as the frozen secondary readout does, changes no reading: the placebo is about zero, and P8 is not a "no"-word artefact.

### 8. Qwen's P3 and word overlap

P3 compares the swapped line plus "Ignore the {obj}." with the swapped line alone. The sentence always adds exactly one word that the action contains, the object's name. So this analysis can only ask whether P3 depends on the overlap both lines already share with the action. It cannot separate salience from name overlap; P4 does that.

| Feature (items with it; without) | Qwen P3 with | Qwen P3 without | Difference, two-cand. | Difference, yes/no | Qwen aligned effect, with; without |
| --- | --- | --- | --- | --- | --- |
| Swapped line names the action's receptacle type (260; 740) | +6.84 (r 1.19) | +6.29 (r 0.98) | +0.55 [−0.45, 1.57] | +0.45 [0.03, 0.90] | 5.76; 6.41 |
| Swapped line shares a word with the action (341; 659) | +5.95 (r 0.90) | +6.68 (r 1.10) | −0.73 [−1.76, 0.24] | +0.17 [−0.27, 0.63] | 6.58; 6.06 |
| Swapped line contains the action's verb (81; 919) | +3.10 (r 0.34) | +6.73 (r 1.13) | −3.63 [−5.49, −1.69] | −0.65 [−1.36, 0.14] | 9.23; 5.98 |
| Object already named in the history (59; 941) | +2.91 (r 1.05) | +6.65 (r 1.03) | −3.75 [−5.20, −2.30] | −2.24 [−2.84, −1.57] | 2.77; 6.46 |

By the action's verb, Qwen's two-candidate P3 is +8.25 (r 1.32) for move, +6.22 (1.21) for take, +5.24 (0.47) for clean, +5.08 (0.97) for heat and +2.77 (0.34) for cool.

- *P3 does not grow with overlap.* Whether the swapped line names the action's receptacle type, or shares any word with the action, changes P3 by less than a nat, with intervals that include zero in the two-candidate format.
- *The verb feature is a task-structure feature.* All 81 items whose swapped line contains the action's verb are clean, cool or heat actions on a task of the same transformation (45 clean, 23 cool, 13 heat; for example `clean bowl 1 with sinkbasin 1` under "put a clean pan in cabinet."). On these, the aligned effect is larger (9.23) and P3 smaller (r 0.34). The mention adds less where the swapped task already calls for the action's transformation, and the real task (which also names the object) adds more. This is the one place where Qwen's P3 falls well short of the goal effect.
- *Where the object is already named in the history,* P3 and the aligned effect are both much smaller, by the same proportion (r 1.05 against 1.03). Naming the object in the task line adds less when the history has already named it, for the real task as much as for "Ignore the {obj}.".
- *Llama* (metrics file) shows the same two task-structure dependences (P3 −0.15 against +0.55 with the verb, −0.05 against +0.53 with the object in the history), and no dependence on receptacle overlap.

### 9. Practical size in the yes/no format

The share of items answered Yes (yes/no log-odds above zero) on the 1,000 sample items, or on P7's 503 items for the claim and placebo arms; in brackets, the share of items that move from No under the swapped task to Yes under the arm.

| Arm | Qwen | Llama | Mistral |
| --- | --- | --- | --- |
| swapped | 0.0% | 6.1% | 44.1% |
| aligned | 0.0% (0.0%) | 22.4% (17.0%) | 80.9% (38.2%) |
| neutral | 0.2% | 5.4% | not scored |
| mention_ignore | 0.0% | 5.9% (2.3%) | 40.5% (8.9%) |
| mention_exists | 0.1% | 18.2% (12.6%) | not scored |
| described | 1.0% (1.0%) | 28.5% (22.7%) | 81.9% (39.3%) |
| desire_keep_sw | 0.0% | 4.5% | 55.2% |
| desire_check | 0.3% | 1.9% | 2.9% |
| desire_check_can | 0.7% | 5.4% | 44.6% |
| done | 0.1% | 14.8% | 56.9% |
| cancelled | 0.1% | 14.8% | 56.5% |
| swapped, on P7's items | 0.0% | 2.4% | 46.9% |
| negated | 0.0% | 2.0% | 18.3% |
| affirmed | 6.4% (6.4%) | 48.3% (46.1%) | 92.8% (46.1%) |
| placebo_negated | 0.0% | 10.3% | not scored |
| placebo_affirmed | 0.0% | 2.6% | not scored |

Qwen's yes/no answers on these determined failures are "No" under every line but the affirmed report, so its large yes/no effects in nats change no answers. Llama's answers do change: the real task turns 17% of items from No to Yes, the described task 23%, the existence sentence 13%, and the user's affirmation 46%. "Ignore the bowl" moves only 2.3%, and the denial leaves 2.0% Yes. For Llama the practical effect follows what the line says about the object, not merely its mention. One more Llama detail: the placebo denial ("no pan in the cabinet 1") gets more Yes answers than the placebo affirmation (10.3% against 2.6%), the same small positive placebo seen in the log-odds.

## What the deep-dive changes

Nothing in the frozen verdicts. The robustness checks support reading the battery's headline numbers at face value. Qwen's mention effect (P3), the described task (P4), the claim effect (P8) and M are broad, item-level shifts present in nearly every game and every task type, in both formats, and not produced by step 0 alone. The caveats they add are specific. Qwen's two-candidate P7 is a net of a step-0 drop and a later rise, so it should not be read as "a denial has no effect". The receptacle split is partly a failure-reason confound. Qwen's P3 is about a third of the goal effect, not all of it, on transformation actions whose swapped task already names the transformation. In practical terms, Qwen's answers to the yes/no question do not change at all.

## Method notes

Intervals are 95% percentile intervals from 2,000 resamples of whole games (multinomial game weights, seed 20261122). r's interval drops resamples whose aligned-effect sum is at or below zero (none for the primary models' key contrasts). Group differences resample games once for both groups together. Nothing is corrected for multiple comparisons; with this many cuts some intervals will exclude zero by chance, so the conclusions above rest on patterns that recur across cuts, formats and models.
