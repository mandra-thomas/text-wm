# TextCraft: how robust is Qwen3-8B's positive goal effect? (exploratory, post hoc audit, 2026-11-21)

**Status.** Everything in this note is exploratory and post hoc. It was written after the confirmatory TextCraft result ([textcraft_results.md](textcraft_results.md), [textcraft_metrics.json](textcraft_metrics.json)) and after the Llama diagnostics ([textcraft_llama_dx.md](textcraft_llama_dx.md), [textcraft_llama_competence.md](textcraft_llama_competence.md)) had been read. It applies to Qwen's positive result the same scrutiny those notes applied to Llama's negative one. Nothing here changes the confirmatory verdict, which stands as frozen: confirmed for Qwen3-8B only. No model was run for this note; it re-reads saved scores. Many cuts were examined, no p-values are reported and nothing is corrected for multiple comparisons, so a single interval that excludes zero is weak evidence on its own. The conclusions below rest on patterns that recur across cuts and readouts.

## Summary

**What was asked.** Qwen3-8B's confirmed TextCraft goal effect is a mean log-odds shift of +0.94 nats [+0.63, +1.24] on the two-candidate scale and +2.93 on the yes/no scale. The question here is whether that effect is robust, and whether it means what the results note says it means. The note asks whether the result rests on an artefact, a handful of items or pairs, a lexical confound, one readout, or one prompt.

**The short answer.** The sign of the effect is robust. What it consists of is narrower than the headline suggests, and in two respects it differs from how the results note describes it.

1. **It is real, positive and reproducible.** An independent recompute gives +0.94 [+0.60, +1.27]. The interval excludes zero under clustering by goal pair and by recipe family. It also excludes zero under trimmed and winsorized means, under every reasonable prompt variant and on the yes/no scale. No prompt tried reverses it. The clean system text makes it larger.

2. **It is a heavy right tail, not a shift of the typical item.** The median item's effect is +0.25 [−0.00, +0.48], and 53% of items move upward. The top 5% of items (75 items in 46 pairs) carry 80% of the total, and the 10% trimmed mean is half the mean (+0.48 [+0.24, +0.73]). The interval reaches zero only once the 29 most contributing pairs of 229 are dropped. That is no fragile few-pair result, but the mean overstates what happens to a typical action. On the yes/no scale the effect is more evenly spread (median +1.50 [+1.25, +1.75], 66% of items up).

3. **Two subsets carry it.** The first is goal-completing crafts that fail for too few units. These 252 items have a mean effect of +2.52 and contribute +0.42 to the overall +0.94. The second is intermediate crafts that appear for the first time in the history and whose name shares a word with the goal's name. These 300 items have a mean effect of +3.05 and contribute +0.61. Together these 552 items (37%) contribute about +1.03. The other 948 items net about −0.10.

4. **The "intermediate lift" is mostly carried by word overlap and absent on repeated crafts.**
   - Ingredient crafts whose name shares no word with the goal show +0.18 [−0.40, +0.74], against +1.20 [+0.60, +1.79] for those that do. After standardizing for failure class the gap is +1.26 [+0.46, +2.06].
   - The cleanest comparison has the same history and class within a couple. It gives a gap of the same size, +1.05, but its interval crosses zero [−0.28, +2.48].
   - Separately, ingredient crafts that repeat a craft already shown in the history show no effect (−0.33 [−0.62, −0.03], 535 items). First-time ingredient crafts show +2.30 [+1.66, +3.02].
   - A smaller lift without overlap survives on first-time never-obtained crafts (+1.61). The intermediate lift is therefore partly a naming effect, but not only one.

5. **On the yes/no scale, and on goal-completing crafts, the effect is the swapped line pushing predictions down, not the aligned line pushing them up.**
   - *Against the neutral line ("explore the crafting commands"), on yes/no.* The aligned line lowers the log-odds of "Yes" by 2.07 and the swapped line by 5.00. On failing actions Qwen answers "Yes" to 15.3% under the aligned line, 18.7% under the neutral line and 8.4% under the swapped line. Naming the right goal does not make Qwen say "Yes" to more failing actions than the neutral line does.
   - *On goal crafts that work.* Qwen says "Yes" to 65% under the aligned line, 73% under the neutral line and 14% under the swapped line. For 156 of these 300 working actions the answer flips from Yes to No when the other goal is named.
   - *What that means.* Qwen treats "crafting the item the other goal line does not ask for" as an action that will fail. That is a large decision-level error on working actions.

6. **Practical size on failing actions is modest.** The two-candidate success prediction on failures is 19.0% (aligned), 16.5% (neutral) and 15.1% (swapped). Aligned against swapped is +3.9 points [+2.2, +5.5]. Balanced accuracy is the same under the aligned and neutral lines (0.774 and 0.775) and lower under the swapped line (0.755; yes/no 0.626).

**What this means for the paper.**

- *Keep.* "Goal effect confirmed for Qwen" stands, and it does not rest on a few pairs or on one prompt.
- *Change "naming the goal raises predicted success for settled failures".* The data support "the two task lines separate Qwen's predictions". On failures, part of that separation is a modest rise under the aligned line on the two-candidate scale. On the yes/no scale and on goal-completing crafts, the main part is a fall under the swapped line.
- *Report the effect's shape.* It is concentrated on first-time crafts and, among intermediate crafts, mostly on names that share a word with the goal. The median is far below the mean. The cross-benchmark yes/no comparison (2.92 against ALFWorld's 2.80) should be read with the yes/no decomposition next to it.

## 1. Data, provenance and method

**Scores.** The note re-reads two kinds of run, all as local copies; every copy's SHA-256 matched the original on ICE on 2026-11-21.

- *The confirmatory run.* `textcraft_score`: 51,200 records, ICE jobs 7204417–7204420 at `5a3702a`, `text-wm-fix24/runs/textcraft_score/shard_0{0..3}_of_04/scoring.jsonl`.
- *Every exploratory prompt-variant run that includes Qwen.*
  - `textcraft_score_examples_on_goal`: 1,600 records.
  - `textcraft_score_empty_start`: 2,400 records.
  - `textcraft_dx_zeroshot`: 3,000 records.
  - `textcraft_dx_objectgoal`: 2,000 records.
  - `textcraft_dx_prompts`, the clean, inventory and clean + inventory variants: 9,000 records.
  - The first two are in `text-wm-fix24`, the zero-shot and object-goal runs in `text-wm-fix26`, and the prompt variants in `text-wm-fix27`.

The hashes are listed in the metrics file under `provenance`. Every record's item hash was recomputed for the prompt it claims: the verified prompt for the confirmatory run, and each variant's own system text, examples, goal line or inventory step for the others. No record is stale in any run, and the items file matches the verified manifest.

**Code.** `scripts/textcraft_qwen_robust.py` writes [textcraft_qwen_robustness_metrics.json](textcraft_qwen_robustness_metrics.json). It shares nothing with the frozen analysis (`wf/textcraft/analysis.py`). It uses its own bootstrap, its own weighted robust statistics and its own readers. From the repository it imports only the score reader of `scripts/textcraft_llama_dx.py` and the prompt, hash and recipe-parsing helpers that define the items. The recipe families follow the definitions fixed in the dx note (plan 7) before any effect was computed on them. Llama-3.1-8B is computed alongside on every cut.

**Measures.** For an item $i$ and a goal line $g$, the two-candidate score is the log-odds of the environment's success reply against its failure reply, teacher-forced after the prompt:

$$L_i(g) = \log P(o^{+}_i \mid \text{prompt}_g) - \log P(o^{-}_i \mid \text{prompt}_g).$$

The goal effect of an item, and the estimand, are

$$\Delta L_i = L_i(\text{aligned}) - L_i(\text{swapped}), \qquad \Delta L = \frac{1}{N}\sum_{i=1}^{N} \Delta L_i .$$

The aligned line names the goal whose recipe tree contains the crafted item. The swapped line names the other goal of the pair, whose tree is also on screen. The neutral line is "Goal: explore the crafting commands." The same difference is taken on four other readouts:

- *First token:* the log-odds of the two replies' first tokens, "Craft" against "Could".
- *Prefix:* the replies with the item's name removed.
- *Name remainder:* full minus prefix.
- *Yes/no:* the log-odds of "Yes" against "No" after the question "Will this action succeed? Answer Yes or No."

A *goal-role* item crafts the goal itself; an *ingredient* item crafts a direct ingredient of it. The failure classes are those of the design: *never obtained* (an input never appeared), *too few* (fewer units than the recipe needs) and *used up* (the input was held and then consumed).

**Statistics.**

- *Intervals.* Every interval is a 95% percentile interval from 2,000 resamples of whole goal pairs (seed 20260924), or of whole recipe families where stated.
- *Robust statistics.* Medians, trimmed means and winsorized means are recomputed inside every resample with that resample's weights. Their intervals are therefore honest about the clustering.
- *Shares.* The "share answered Yes" or "share predicted success" is the share of items whose L is above zero. It is never a mean probability; mean probabilities are labelled as such.
- *Probability shift.* $\Delta p$ is the mean of the logistic of L under aligned minus under swapped, in percentage points.

## 2. Check 1: independent recompute, distribution and concentration

**The recompute reproduces the confirmatory numbers.** The independent code gives the numbers below. All match the frozen metrics to two decimals; the intervals differ only by bootstrap noise.

- *Two-candidate:* +0.94 [+0.60, +1.27] (frozen +0.94 [+0.63, +1.24]).
- *Yes/no:* +2.92 [+2.53, +3.32].
- *First token:* +0.97.
- *Prefix:* +0.95.
- *Name remainder:* −0.01.
- *Aligned minus neutral:* +0.55.
- *Swapped minus neutral:* −0.39.
- *Probability shift:* +3.59 points.

All 750 couples are complete. Qwen's probability mass on the two candidates' first tokens is 1.000 on average under every line (minimum 0.82; 99.4% or more of items at 0.99 or above). The two-candidate score is therefore not distorted by mass leaking to other replies, unlike Llama's (mean mass 0.85).

**The distribution is heavy-tailed to the right.** The item SD is 4.74 nats, against 1.38 for Llama. The quantiles of the item effect are −4.75 (5th percentile), −1.50 (25th), +0.25 (median), +2.50 (75th) and +9.25 (95th); the 99th is +18.8. The table compares the mean with statistics that resist the tails.

| Statistic of the item effect | Qwen two-candidate | Qwen yes/no | Llama two-candidate |
|---|---|---|---|
| Mean | +0.94 [+0.60, +1.27] | +2.92 [+2.53, +3.32] | −0.19 [−0.29, −0.09] |
| Median | +0.25 [−0.00, +0.48] | +1.50 [+1.25, +1.75] | −0.15 [−0.26, −0.06] |
| 10% trimmed mean | +0.48 [+0.24, +0.73] | +2.03 [+1.75, +2.36] | −0.18 [−0.27, −0.08] |
| 20% trimmed mean | +0.33 [+0.10, +0.55] | +1.76 [+1.51, +2.04] | −0.15 [−0.25, −0.06] |
| 5% winsorized mean | +0.79 [+0.49, +1.09] | +2.68 [+2.31, +3.11] | −0.19 [−0.29, −0.10] |
| Share of items above zero | 0.53 [0.50, 0.56] | 0.66 [0.63, 0.69] | 0.45 [0.41, 0.48] |

Llama's effect is homogeneous: its median, trimmed means and mean agree. Qwen's mean is about four times its median on the two-candidate scale. Every trimmed and winsorized version still excludes zero, so the positive sign does not depend on the tail. The size does: a reader who takes +0.94 nats as the shift of a typical item overstates it severalfold. The yes/no effect is more evenly spread (median about half the mean, two thirds of items up).

**How many items and pairs carry it.**

- *Items.* The top 1% of items by effect carry 25% of the total sum, the top 5% carry 80% and the top 10% carry 119%; the bottom 10% carry −60%. Removing the 109 largest items (7%) brings the mean to zero. That figure means little on its own, because removing the top of any distribution with a positive mean will eventually do so; the trimmed means above are the symmetric version.
- *Pairs.* 57% of the 229 pairs have a positive mean effect (quartiles −0.52, +0.45, +1.71), and 56% of the 750 couples do.
- *Leaving out pairs.* Leaving out any one pair moves the mean only between +0.85 and +0.96. The table drops the pairs with the largest positive contribution to the sum.

| Pairs dropped (largest contribution first) | Items left | Effect |
|---|---|---|
| 0 | 1,500 | +0.94 [+0.60, +1.27] |
| 1 (p154) | 1,476 | +0.85 [+0.58, +1.13] |
| 5 | 1,398 | +0.71 [+0.47, +0.97] |
| 10 | 1,334 | +0.56 [+0.33, +0.80] |
| 20 | 1,212 | +0.35 [+0.14, +0.55] |
| 30 | 1,126 | +0.18 [−0.02, +0.38] |

The lower end of the interval reaches zero once 29 pairs are dropped, and the point estimate once 42 are dropped. No single pair or small group of pairs carries the result. The largest, p154 (24 items, mean +6.15), contributes about a tenth of it.

**Where on the log-odds scale the shift happens.** In 70% of items both lines put L at or below −3 (a confident failure prediction under both), and these items contribute only +0.15 of the +0.94. In 10% both lines are at or above +3, and these contribute +0.09. The remaining 20% of items, where at least one line is not confident, contribute +0.70. Clipping L to ±3 before differencing gives +0.22 [+0.14, +0.29], and to ±5 gives +0.34 [+0.22, +0.45]. The effect is therefore not an artefact of large movements deep inside predictions that are confidently failure under both lines. It is concentrated where the prediction is open.

**Competent subset.** On the 83.5% of failures that Qwen calls failures under the neutral line (L below zero), the effect is +0.71 [+0.40, +1.03]. On those it gets wrong under the neutral line it is +2.10 [+1.28, +2.93]. The effect is not confined to items the model already misjudges. It is larger where the model is wrong or unsure.

**Artefact checks.** The four scoring shards (pairs split by sorted id) give +1.57, +0.93, +0.41 and +0.86, which is the spread expected from pair heterogeneity. The name remainder is −0.01, so the effect is decided before the item's name is scored.

## 3. Check 2: recipe families

The two family definitions were fixed in the dx note before any effect was computed on them. The first, *shared-craftable components*, links pairs whose displayed lists share any craftable item. It gives 22 components, the largest holding 208 of 229 pairs, so by the rule fixed in advance it yields no usable interval. The second, the *primary material*, assigns each pair to the non-craftable input on most of its lines. It gives 50 families; the largest is acacia logs, with 80 pairs and 528 items.

| Readout, resampling whole families | Mean | Median | 10% trimmed mean |
|---|---|---|---|
| All items, two-candidate | +0.94 [+0.73, +1.31] | +0.25 [+0.00, +0.38] | +0.48 [+0.30, +0.75] |
| All items, yes/no | +2.92 [+2.50, +3.79] | +1.50 [+1.25, +2.00] | +2.03 [+1.73, +2.75] |
| Goal role, two-candidate | +1.05 [+0.58, +1.85] | +0.32 [−0.01, +0.88] | +0.60 [+0.25, +1.29] |
| Ingredient role, two-candidate | +0.88 [+0.53, +1.31] | +0.11 [−0.22, +0.25] | +0.42 [+0.09, +0.67] |

The family-clustered mean reproduces the dx note's +0.94 [+0.73, +1.31]. Leaving out any one family moves the mean between +0.89 (without birch logs) and +1.06 (without acacia logs). 74% of families have a positive mean effect. The acacia-log family itself is positive (+0.72), so the largest cluster is not what makes the effect positive. Family clustering does not threaten the sign; as with pairs, the medians are much smaller than the means.

## 4. Check 3: word overlap between the action and the goal

**The concern.** In 683 of the 990 ingredient failures the crafted item shares a word with the goal's name, as with "stone bricks" for the goal "stone brick wall". In 180 of them the ingredient's full name sits inside the goal's name, as with "sandstone" for "sandstone slab". (These counts reproduce review round 2.) If the goal line raises predicted success mainly when it echoes the action's words, Qwen's "intermediate lift" is a naming effect rather than a judgement that the action serves the task.

**The categories.** Ingredient items were split into three groups.

- *Contained* (180 items): the full name is a run of words of the goal's name.
- *Word overlap only* (503 items): they share a word but the name is not contained.
- *No shared word* (307 items).

Goal-role items name the goal itself verbatim (510 items). A stemmed variant, which drops a plural s, moves 85 items from overlap-only to contained and changes nothing below.

| Group | Items | Two-candidate effect | Yes/no effect | Aligned minus neutral | Swapped minus neutral | Contribution to the 0.94 |
|---|---|---|---|---|---|---|
| Goal role (identical name) | 510 | +1.05 [+0.50, +1.60] | +4.97 [+4.22, +5.75] | −0.48 | −1.53 | +0.36 |
| Ingredient, contained | 180 | +1.98 [+0.68, +3.38] | +3.52 [+2.17, +4.95] | +1.64 | −0.34 | +0.24 |
| Ingredient, overlap only | 503 | +0.92 [+0.32, +1.50] | +1.87 [+1.33, +2.43] | +0.81 | −0.10 | +0.31 |
| Ingredient, no shared word | 307 | +0.18 [−0.40, +0.74] | +0.90 [+0.29, +1.57] | +1.19 | +1.01 | +0.04 |

**The pattern.** Without a shared word, the aligned and swapped lines raise L by the same amount over the neutral line, so there is no goal effect. This is exactly the pattern Llama shows on all its ingredients. With a shared word, the aligned line raises L and the swapped line does not.

**Correcting for failure class.** The overlap groups differ in failure class: 59% of overlap-only items are used up, a class with no effect for any group. Standardizing each group to the class mix of all ingredient items gives three levels: contained +1.39 [+0.49, +2.30], overlap only +1.27 [+0.59, +1.97] and no shared word +0.10 [−0.41, +0.59]. Any overlap minus no shared word, class-standardized, is +1.26 [+0.46, +2.06].

**A within-couple test.** The category subsets break the couple design: a couple's two members can fall in different groups, and the history then favours one goal more within a group. 205 couples have one overlapping and one non-overlapping ingredient member, with the same pair, history and class. Within these couples, the overlapping member's effect is +1.05 [+0.12, +1.98] and the non-overlapping member's is −0.01 [−0.83, +0.79]. The per-couple difference is +1.05 [−0.28, +2.48] on two-candidate and +1.30 [−0.29, +2.96] on yes/no. The same gap appears in the design that controls history, pair and class exactly, but that design alone cannot exclude zero. Split by which goal the history favours, the gap is +0.36 (aligned-favouring), +1.22 (swapped-favouring) and +1.45 (tie). History fit, if anything, works against it.

**Couples of one kind.** Restricted to couples whose two members fall in the same group, the effect is +1.26 [+0.55, +1.98] for 239 couples that both overlap (the confirmatory named readout) and +0.56 [+0.09, +1.09] for 51 couples that share no word.

**The overlap effect lives on first-time crafts.** A second factor turns out to matter as much as overlap: whether the history already shows a successful craft of the same item. 483 of the 494 used-up items repeat a craft the history shows, because the craft that used up the input was usually a craft of the same item. Crossing the two factors on ingredient failures:

| Ingredient failures | Items | Two-candidate effect |
|---|---|---|
| First-time craft, shares a word with the goal | 300 | +3.05 [+2.10, +4.05] |
| First-time craft, no shared word | 155 | +0.86 [−0.09, +1.70] |
| Repeat craft, shares a word | 383 | −0.25 [−0.63, +0.12] |
| Repeat craft, no shared word | 152 | −0.51 [−0.90, −0.08] |

Within first-time crafts the overlap gap holds in each class, but its size differs:

- *Never obtained:* overlap +2.28, no shared word +1.61.
- *Too few:* overlap +3.74, no shared word +0.24.

On first-time crafts with no shared word, the yes/no effect is +2.45 [+1.40, +3.43].

**Does an exact name lift more? The object-goal run.** A pure word-priming account predicts that naming the ingredient itself as the goal ("Goal: craft Y.") would lift its L at least as much as a goal sharing one word. On 300 ingredient failures it does not. The object-goal line is −1.23 [−2.01, −0.47] below the aligned line and −0.19 [−0.98, +0.61] from the swapped line. Its share predicted success, 21.7%, equals the aligned line's 22.0%. On yes/no it does raise Yes over swapped (+4.35; 20.0% Yes against 15.7% aligned and 11.0% swapped). The dx note found the two-candidate drop concentrated where the ingredient's own inputs were never obtained (−4.00), which suggests that "craft Y" prompts Qwen to check Y's inputs.

**Verdict on check 3.**

- *Most of the lift needs a shared word.* Qwen's intermediate lift is mostly found where the goal's name shares a word with the action. Ingredients without a shared word contribute 0.04 of the 0.94, and the class-standardized gap is about +1.3 nats.
- *It is not a pure lexical echo.* The exact-name goal does not lift more than a partial overlap. A residual lift without overlap survives on first-time never-obtained crafts: +1.61 here, +0.74 [+0.33, +1.16] in the usefulness population (whose goal identity is not balanced, design D1), and +0.56 in the 51 same-history usefulness-only couples.
- *A better description.* The goal line moves Qwen where the action's link to the goal is easy to see. The cleanest same-history test leaves the size of the overlap gap uncertain.
- *Llama.* The same split shows a +0.70 [+0.36, +1.04] effect on contained ingredients, its only positive cell. Overlap-only is −0.20 and no shared word +0.02. So Llama has a small, ALFWorld-like effect of naming an ingredient's full name, which its negative goal-role term hides in the overall mean.

## 5. Check 4: scale and format

| Readout (main failures) | Mean | Median | 10% trimmed mean | Share above zero | Standardized |
|---|---|---|---|---|---|
| Two-candidate | +0.94 [+0.60, +1.27] | +0.25 | +0.48 | 0.53 | 0.20 |
| First token | +0.97 [+0.65, +1.28] | +0.25 | +0.51 | 0.53 | 0.21 |
| Prefix | +0.95 [+0.61, +1.27] | +0.25 | +0.49 | 0.53 | 0.20 |
| Name remainder | −0.01 [−0.02, −0.00] | 0.00 | 0.00 | 0.42 | −0.05 |
| Yes/no | +2.92 [+2.53, +3.32] | +1.50 | +2.03 | 0.66 | 0.49 |

**The conclusion holds on every outcome readout.** The two-candidate, first-token and prefix readouts are nearly identical, because Qwen's first-token mass is 1 and the decision is made at the first token. The remainder carries nothing. The yes/no effect is positive overall and in the goal and ingredient roles. Per role and class, though, the two scales disagree in two cells:

- *Goal × never obtained:* two-candidate −0.45 [−0.87, −0.05], yes/no +1.73 [+1.17, +2.27].
- *Ingredient × used up:* two-candidate −0.20 [−0.59, +0.19], yes/no +0.53 [+0.31, +0.76].

In both cells the yes/no readout is the more positive. Across items the two readouts correlate (Spearman 0.57) and agree in sign for 65% of items. In 23% the two-candidate effect is negative while the yes/no effect is positive; the reverse holds for 9%.

**The yes/no anomaly, characterized.** On yes/no, both named lines push Qwen toward "No" relative to the neutral line, the swapped line far more.

| Items | Aligned minus neutral | Swapped minus neutral | Share answered Yes: aligned / swapped / neutral |
|---|---|---|---|
| All main failures (1,500) | −2.07 [−2.46, −1.67] | −5.00 [−5.46, −4.53] | 15.3% / 8.4% / 18.7% |
| Goal role, failures (510) | −0.52 [−1.03, +0.02] | −5.49 [−6.31, −4.71] | 15.3% / 3.5% / 17.4% |
| Ingredient, used up (494) | −5.29 [−5.84, −4.75] | −5.82 [−6.40, −5.24] | 0.2% / 0.4% / 6.5% |
| Ingredient, too few (248) | −1.39 | −4.62 | 44.4% / 32.7% / 45.6% |
| Goal role, successes (300) | −1.58 [−2.89, −0.22] | −18.02 [−19.80, −16.21] | 65.0% / 13.7% / 73.0% |
| Ingredient, successes (300) | −4.42 [−5.11, −3.73] | −7.18 [−8.48, −5.98] | 61.0% / 53.3% / 72.0% |

The anomaly has two parts.

1. **The neutral line makes Qwen more willing to say "Yes" to everything.** This shows most on used-up ingredients: 6.5% Yes under the neutral line against 0.2% under the aligned line. "Explore the crafting commands" reads as a license to try anything, and any named goal makes Qwen stricter. Discrimination does not suffer: the yes/no AUC is 0.85 under the neutral line and 0.84 under the aligned line. It is a level shift, and it is why aligned minus neutral is negative overall.
2. **The swapped line makes Qwen say "No" to crafting the other goal's final item, whether or not that craft works.** On goal-role successes, swapped minus neutral is −18 nats, and the Yes share falls from 73% to 14%. The yes/no AUC under the swapped line drops to 0.74 (goal role 0.76), from 0.84–0.87 under the other two. The swapped line damages discrimination, and the aligned line does not.

So the yes/no effect of +2.92 is mainly the swapped line suppressing "Yes", not the aligned line raising it. The likeliest reading is that Qwen answers "Will this action succeed?" partly as "is this a step toward my goal?". Crafting the top item of the tree the task does not ask for is then "not going to succeed". The two-candidate readout shows the same direction on goal crafts, weaker: on goal-role successes the share predicted success is 72.7% aligned, 66.0% neutral and 55.7% swapped. It cannot be called a pure format effect, because the two-candidate reply ("Crafted 1 minecraft:…" against "Could not find enough items…") asks no question about goals.

## 6. Check 5: prompt robustness

Each variant was scored on a seeded subsample, and its effect is compared with the verified prompt's effect on the same items. All are two-candidate effects, with no yes/no arm in these runs.

| Variant (items, pairs) | Variant effect | Verified, same items | Variant minus verified | Variant median | Goal role, variant | Ingredient, variant |
|---|---|---|---|---|---|---|
| Examples on goal (400, 130) | +0.72 [+0.22, +1.24] | +0.75 [+0.25, +1.28] | −0.03 [−0.19, +0.13] | −0.25 [−0.50, +0.23] | +1.38 | +0.37 |
| Empty start stated (600, 168) | +1.00 [+0.57, +1.41] | +1.04 [+0.62, +1.43] | −0.04 [−0.13, +0.05] | +0.25 [−0.25, +0.50] | +1.16 | +0.90 |
| Clean system text (300, 108) | +1.52 [+1.00, +2.06] | +1.15 [+0.60, +1.73] | +0.36 [+0.08, +0.65] | +0.75 [+0.25, +1.13] | +3.03 | +0.77 |
| Inventory shown (300, 108) | +1.12 [+0.58, +1.72] | +1.15 [+0.60, +1.73] | −0.04 [−0.45, +0.37] | +0.25 [−0.02, +0.62] | +1.86 | +0.74 |
| Clean + inventory (300, 108) | +1.42 [+0.89, +1.97] | +1.15 [+0.60, +1.73] | +0.27 [−0.21, +0.72] | +0.50 [+0.25, +0.88] | +3.10 | +0.58 |
| Zero-shot (300, 102) | +2.11 [+1.61, +2.63] | +0.62 [+0.08, +1.21] | +1.49 [+0.86, +2.08] | +1.43 | +3.46 | +1.33 |

**No reasonable prompt makes the effect vanish or reverse.** Every variant's mean is positive with an interval excluding zero. The variants that remove the agent instructions (clean, clean + inventory) make it larger, chiefly on the goal role (+1.66 and +1.73 against the verified prompt). Under them the median also becomes clearly positive: +0.75 [+0.25, +1.13] and +0.50 [+0.25, +0.88].

**Which prompt changes shift which cell.**

- *Goal × never obtained.* Under the clean prompt this cell turns from −0.31 to +1.61 [+0.78, +2.44], and under clean + inventory to +0.82. The only negative cell in the confirmatory data is thus specific to the agent-instruction system text. Under the clean prompt the goal-role decomposition is aligned minus neutral +0.53 and swapped minus neutral −2.49. The goal-role effect remains mostly a swapped-line penalty.
- *Ingredients.* Showing the inventory lowers the ingredient effect somewhat (−0.30), and so does the clean text (−0.28). This fits the account in check 3: the ingredient lift is largest where the state is least legible, and partly lexical.

**Medians on subsamples.** On the examples-on-goal subsample the median is −0.25 under both the variant and the verified prompt, and the trimmed mean does not exclude zero under either (+0.24 and +0.29). On a 400-item random subsample the robust statistics are too noisy to separate from zero even for the confirmatory prompt. That is a power limit of the subsample, not a change caused by the variant.

**Zero-shot is not interpretable, and the dx note's reading of it needs a correction.** Without examples Qwen's two-candidate L is about +63 for every item, success or failure. Its first-token mass is 0.64 and its AUC 0.43. The "larger effect zero-shot" (+2.11) sits in the reply tails: the zero-shot first-token effect is +0.31 [−0.09, +0.75], smaller than the verified first-token effect, not larger. Neither is a world-model quantity. The dx note's sentence that "for Qwen the examples … hold the goal effect down" should be read with this in mind.

## 7. Check 6: practical size and decisions

The table gives, for each line, the share of items with L above zero ("predicted success"; on yes/no, "answered Yes").

| Items | Readout | Aligned | Neutral | Swapped | Aligned minus swapped (points) | Flips up / down |
|---|---|---|---|---|---|---|
| Failures, all (1,500) | Two-candidate | 19.0% | 16.5% | 15.1% | +3.9 [+2.2, +5.5] | 84 / 26 |
| Failures, all | Yes/no | 15.3% | 18.7% | 8.4% | +6.9 [+5.2, +8.6] | 112 / 8 |
| Failures, goal role (510) | Two-candidate | 13.3% | 10.8% | 7.2% | +6.1 [+3.3, +9.0] | 35 / 4 |
| Failures, goal × too few (252) | Two-candidate | 25.4% | 19.8% | 12.7% | +12.7 [+7.3, +18.8] | 34 / 2 |
| Failures, ingredient (990) | Two-candidate | 21.9% | 19.4% | 19.2% | +2.7 [+0.6, +4.6] | 49 / 22 |
| Successes, goal role (300) | Two-candidate | 72.7% | 66.0% | 55.7% | +17.0 [+10.2, +24.0] | 66 / 15 |
| Successes, goal role | Yes/no | 65.0% | 73.0% | 13.7% | +51.3 [+45.0, +57.7] | 156 / 2 |
| Successes, ingredient (300) | Two-candidate | 75.0% | 77.0% | 76.7% | −1.7 [−5.4, +2.0] | 12 / 17 |
| Successes, ingredient | Yes/no | 61.0% | 72.0% | 53.3% | +7.7 [+3.3, +11.9] | 32 / 9 |

"Flips up" counts items predicted success under the aligned line but not the swapped line; "down" the reverse. The mean probability shift on failures is +3.59 points [+2.32, +4.88] on two-candidate and +6.71 [+5.27, +8.17] on yes/no.

**On failing actions the effect is modest.** Naming the right goal rather than the other one turns about 4 in 100 settled failures into predicted successes on the two-candidate readout. Most of the change happens in the goal × too-few cell, where it is 13 in 100. Against the neutral line the aligned line adds 2.5 points on two-candidate and removes 3.4 points on yes/no.

**On working actions the effect is large.** On goal-completing crafts that would succeed, the swapped line makes Qwen answer "No" much more often.

- *How many.* 156 of the 300 goal-role successes are answered "Yes" under the aligned line and "No" under the swapped line, and 2 the other way. They spread over 75 pairs.
- *Where.* They occur in every length bin (51, 50 and 55 of 100 per bin), at depth 1 (120 of 250) and depth 2 (36 of 50), and whichever goal the history favours (75 of 123 swapped-favouring, 50 of 123 aligned-favouring, 31 of 54 ties).
- *What the other lines say.* 90% of these 156 are answered "Yes" under the neutral line. On the two-candidate readout 91% of them are predicted success under the aligned line and 62% under the swapped line.
- *Over all successes.* The Yes share is 63.0% aligned, 72.5% neutral and 33.5% swapped.

This false-failure effect is the largest decision-level consequence of the goal line in TextCraft, and it is not the direction the headline describes. A world model told the wrong task rejects working crafts of the other task's item.

**Net effect on decisions.** Balanced accuracy over failures and successes is the mean of the share of successes predicted success and the share of failures predicted failure.

- *Two-candidate:* 0.774 under the aligned line, 0.775 under the neutral line and 0.755 under the swapped line.
- *Yes/no:* 0.738, 0.769 and 0.626.
- *Goal role on yes/no:* the swapped line gives 0.551, near chance.

Against neutral, naming the right goal neither helps nor hurts Qwen's decisions on the two-candidate readout, and hurts slightly on yes/no. Naming the wrong goal hurts. Llama's decisions barely move under any line (two-candidate 0.554, 0.534, 0.554).

## 8. Check 7: heterogeneity by class and length

| Cell | Items | Effect | Neutral-line AUC against same-role successes | Share confidently failed under neutral (L at most −5) | Mean L: aligned / swapped / neutral |
|---|---|---|---|---|---|
| Goal, never obtained | 252 | −0.45 [−0.89, −0.05] | 0.97 | 97% | −20.5 / −20.1 / −19.3 |
| Goal, too few | 252 | +2.52 [+1.53, +3.56] | 0.82 | 69% | −9.2 / −11.8 / −9.5 |
| Ingredient, never obtained | 248 | +1.64 [+1.05, +2.27] | 0.93 | 80% | −11.6 / −13.3 / −13.8 |
| Ingredient, too few | 248 | +2.28 [+1.27, +3.49] | 0.79 | 45% | −0.3 / −2.6 / −2.9 |
| Ingredient, used up | 494 | −0.20 [−0.56, +0.18] | 0.91 | 59% | −6.9 / −6.7 / −6.7 |

**Why too few is large.** Too few is the class Qwen judges least well (AUC 0.79–0.82): only half to two thirds of these failures are predicted failure with confidence. Within goal × too few the effect depends on how open the neutral prediction is.

- *By the neutral line's L.* The effect is +0.66 [−0.01, +1.36] where the neutral L is below −10 (141 items). It is +4.08 between −10 and −5, +6.90 between −5 and 0, and +4.26 at or above 0.
- *By history length.* It grows from +0.56 at 1–3 steps to +2.86 at 4–8 and +4.15 at 9–12. Longer histories mean more counting.
- *Over all failures.* The same gradient holds: +0.34 where the neutral L is below −15, rising to +1.36 between −10 and −5 and to +2.35 at or above +2.

The goal line moves Qwen where its own state judgement is uncertain, as on ALFWorld, where it acted most where the history left the location open.

**Why used up is near zero.** It is not that Qwen cannot judge it: its AUC is 0.91, and the three lines give the same mean L (−6.9, −6.7, −6.7) at every level of the neutral L. What sets used-up items apart is that 483 of 494 repeat a craft the history already shows succeeding. In every ingredient class, repeated crafts show no positive effect, while first-time crafts show a large one.

| Ingredient failures | First-time craft | Repeat craft |
|---|---|---|
| Never obtained | +2.06 [+1.40, +2.75] (215 items) | −1.08 [−2.03, −0.31] (33) |
| Too few | +2.44 [+1.37, +3.72] (229) | +0.40 [−1.15, +2.23] (19) |
| Used up | +4.26 [−0.62, +11.51] (11) | −0.30 [−0.62, +0.01] (483) |
| All ingredients | +2.30 [+1.66, +3.02] (455) | −0.33 [−0.62, −0.03] (535) |

The same holds on ingredient successes: +1.18 [+0.25, +2.15] on 206 first-time crafts against −0.01 on 94 repeats.

Class and repetition cannot be fully separated, because only 11 used-up items are first-time crafts. Still, the null on used up is what the repetition pattern predicts, and the rare repeats in the other two classes show it too. The likeliest reading is that a history holding an earlier outcome of the same action anchors Qwen's prediction, and the goal line then has nothing to add. Where the history holds no earlier outcome, the goal line fills the gap.

**Reconciling goal × never obtained (−0.45 on two-candidate, +1.73 on yes/no).** This cell is where Qwen is most competent: AUC 0.97, median L about −21 under every line. Only 1.2–1.6% of items are predicted success under any line on two-candidate, and 0.8–2.8% answered Yes. Both scales therefore describe small movements deep inside a confident failure prediction, with no consequence for decisions. The sign differs because the cell holds two pulls of opposite effect on the difference, weighted differently by the two readouts.

1. *The aligned line lowers L.* On two-candidate, aligned minus neutral is −1.21 and swapped minus neutral −0.76. This fits the object-goal run's reading that "craft G" makes Qwen check G's inputs, which here were never obtained. It is strongest where half or more of the goal's inputs are already held (two-candidate −1.58 [−2.80, −0.51] on 55 items), the case where the missing input stands out.
2. *The swapped line lowers yes/no much more.* On yes/no, aligned minus neutral is −0.67 and swapped minus neutral −2.40. This is the swapped line's "No" to crafting the other goal's item from check 4, and it dominates the yes/no difference.

29% of the cell's items are negative on two-candidate and positive on yes/no. Under the clean system text the two-candidate cell turns clearly positive (+1.61), so the negative sign is also specific to the agent-instruction prompt.

**Length by role.** The goal-role effect grows with history length: −0.02 at 1–3 steps, +1.27 at 4–8 and +1.89 at 9–12. The ingredient effect falls: +2.15, +0.78 and +0.45. The ingredient fall is largely composition: used-up and repeat crafts, which show no effect, are nearly absent at 1–3 steps.

## 9. Check 8: free generation (not run, and why)

The plan allowed a small generation run for Qwen if the scores left open what Qwen actually predicts. They do not, so no GPU job was started and no ICE checkout was made.

- *Two-candidate.* Qwen puts essentially all first-token probability on the two candidates' first tokens: mean 1.000 under every line, minimum 0.82, at least 99.4% of items at 0.99 or above. On yes/no the mass on "Yes" and "No" is exactly 1. Greedy generation's first token is therefore fixed by the sign of the first-token score. That sign agrees with the sign of the full two-candidate score on 99.6% of items.
- *The check on Llama.* For Llama, whose first-token mass is far lower (mean 0.85), the existing generation run (`textcraft_dx_generate`) was checked. Its generated observation followed the first-token sign on 99.4% of the 867 generations that were a success or failure reply. Its free yes/no answer followed the yes/no sign on 100% of 900.

A Qwen generation run would reproduce the decision tables of section 7 to within a fraction of a percent. It would cost GPU time and checks and add no information.

## 10. Overall verdict

**How robust.** Qwen3-8B's positive TextCraft goal effect is robust in sign:

- across pairs, recipe families, shards, trimmed and winsorized means;
- on the first-token, prefix and yes/no readouts;
- under every reasonable prompt tried, including one that shows the full state and one without the agent instructions.

None of the artefacts examined carries it. The name remainder is zero, the first-token mass is one, and every record is current. Nor is it carried by a few items or pairs in the fragile sense: about 30 of 229 pairs must go before the interval reaches zero.

**What it is.** The effect is not a uniform shift of predicted success for settled failures.

- *A tail.* It is a heavy-tailed effect, about +0.25 nats for the median item against +0.94 for the mean.
- *Two subsets.* It lives where Qwen's state judgement is open: crafts the history has not shown before, and above all too-few failures. On intermediate crafts it mostly needs a word shared with the goal's name.
- *What it is not.* It is absent on repeated crafts and inside confident predictions.
- *A contrast, not a lift.* Relative to the neutral line it is less a lift under the right goal than a penalty under the wrong one. That holds especially on the yes/no question and on goal-completing crafts, where naming the other goal makes Qwen reject crafts that would work.

**What it means.** Read as a statement about a world model's use of the task line, the evidence supports this: Qwen lets the task line decide outcomes where the history leaves them open. Where the link between action and task is obvious, it predicts the action works under the matching task. Where the action does not serve the named task, it predicts the action fails. This is a task-relevance bias, the same family as the ALFWorld finding. It is not specifically "naming the goal makes the model optimistic about failing actions". Its costs fall more on working actions under a mismatched task than on failing actions under the right task.

**Where the result could be wrong, and what remains open.**

- The overlap account rests on between-item comparisons. The one design that holds history and class fixed gives the same gap with an interval that crosses zero.
- The repetition account cannot be separated from the used-up class with these items.
- The neutral line is not a neutral zero: "explore" makes Qwen permissive on yes/no. Statements relative to it are descriptive.
- A run padding histories or adding first-time used-up items, and one varying overlap within a pair, would settle the first two. Round 2's arms that separate naming from completion bear on the goal-role part.

## 11. What should change in the paper

1. **Keep the confirmatory statement** ("confirmed for Qwen3-8B only"), and add that its sign survives pair and family clustering, robust means and all prompt variants.
2. **Replace "naming the goal raises predicted success for actions the history has settled"** with a two-line contrast. "Qwen's predictions differ between the matching and the mismatched task line." On failures the matching line adds a few points of predicted success relative to a neutral line (two-candidate). On yes/no and on goal-completing crafts the difference is mainly the mismatched line lowering predicted success, including for working actions.
3. **Report the median and a trimmed mean next to the mean** (+0.25 and +0.48 against +0.94), and the decision-level numbers (+3.9 points on failures; a fall in "Yes" from 65% to 14% on goal-completing successes).
4. **Qualify the "intermediate lift"** as concentrated in first-time crafts whose names share a word with the goal. The results note's sentence "the goal line moves the predicted outcome, not just the copying of a name" is right about the reply side (the remainder is zero). It does not address the prompt-side overlap, which carries most of the ingredient lift.
5. **Qualify the yes/no cross-benchmark comparison.** In TextCraft the +2.92 is mostly the swapped line suppressing "Yes", and the neutral line is permissive. Before equating it with ALFWorld's 2.80, the same decomposition against neutral should be checked on ALFWorld.
6. **Correct the zero-shot remark** in the dx note: Qwen's zero-shot effect is larger only in the reply tails, and its first-token effect is smaller.

## 12. Reproducing

From the repository root, with the score copies above (verified by SHA-256) laid out as `<dir>/main/shard_0{0..3}/scoring.jsonl`, `<dir>/examples_on_goal/`, `<dir>/empty_start/`, `<dir>/zeroshot/`, `<dir>/objectgoal/`, `<dir>/prompts/` and `<dir>/generate/` (Llama's generations, for section 9), and with `wf/textcraft/data/recipes_frozen.json` in place (git-ignored), run:

    python scripts/textcraft_qwen_robust.py --scores <dir>/main --variants <dir> \
        --out notes/textcraft_qwen_robustness_metrics.json

It takes about three minutes on a laptop. Where to find each part of this note in the output:

| Section of this note | Key in the metrics file (under `per_model.qwen` unless stated) |
|---|---|
| Check 1 | `recompute`, `concentration` |
| Check 2 | `families` |
| Check 3 | `overlap` (including `mixed_couples`), and `heterogeneity.repeat_craft.cross_with_overlap` |
| Check 4 | `scales`, `yes_no` |
| Check 5 | top-level `prompts` |
| Check 6 | `decisions` |
| Check 7 | `heterogeneity` |
| Check 8 | top-level `format` |
