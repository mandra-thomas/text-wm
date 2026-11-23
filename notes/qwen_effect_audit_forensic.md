# Is Qwen3-8B's large ALFWorld goal effect real? A forensic audit of the stored scores (2026-11-22)

**Exploratory.** Nothing here re-tests or re-verdicts a confirmatory result. RQ2's preregistered verdict (`rq2_results.md`) stands as filed; this note asks only how its Qwen3-8B number should be read.

## 0. Analysis list, fixed before any number was computed

Written and committed before the analysis script was run. The only scores seen beforehand were the two example records printed while checking the file format (one item of `rq2_score`, shard 0). The stance is adversarial: the working assumption is that Qwen's 6.12 nats are an artefact of measurement or scale, and each check below is a way that assumption could be shown wrong or right.

Data: stored scores only, no new scoring. `rq2_score` (two-candidate, aligned and swapped, 4 shards, fix11), `rq1_score` (neutral task and the extra-mention arm, fix12), `rq1x_score` (described goals, fix13), `rq1v_score` (yes/no, fix17), `size_score_qwen3-*` (fix18), `family_score_*` (fix21), `mech_score` (fix19), `mech2_score` (fix36) and the collection (`rq12_transitions_s0..2`, fix9). Items are RQ2's 4,666 determined failures unless stated. Intervals are game-clustered percentile bootstraps (2,000 resamples, seed 20261122).

**A. Sanity checks on the stored scores (question 6).**
1. Per run, model and condition: record counts, NaN or infinite values, duplicate records (same item, model and condition) and whether duplicates agree, `item_hash` against the hash of the item in the collection (stale scores), and each shard's commit against `reproducing.md`.
2. Re-render the aligned and swapped prompts of every item with the repository's prompt code and confirm on the strings that they differ only in the task line; tokenize with the pinned Qwen3-8B and Llama-3.1-8B tokenizers (CPU) and confirm that the token difference is confined to the task-line span.
3. Candidates: the stored candidate token counts and first tokens must be equal across an item's conditions.
4. GPU and shard: effect by shard (shards 0–2 on A40, shard 3 on L40S); per-item agreement between `rq2_score` and the rescored Qwen3-8B in `size_score_qwen3-8b` (and `rq1_score`), and whether disagreement depends on the shard.

**B. Where on the log-odds scale the shift happens (question 1).** For Qwen3-8B and Llama-3.1-8B:
1. Distribution (mean, SD, percentiles 1, 10, 25, 50, 75, 90, 99) of L under the aligned, swapped and neutral tasks, and of each candidate's log-probability (the success reply, "Nothing happens.") separately.
2. On the probability scale: the full success reply's probability, "Nothing happens."'s probability, the first-token probability of "You", and σ(L). Shares of items with σ(L) above 1%, 10% and 50% under the aligned task (and the swapped task); the same for the success reply's own probability.
3. ΔL split into the change in the success reply's log-probability and the change in "Nothing happens."'s log-probability.
4. ΔL by decile of the swapped-task L.

**C. Token decomposition (question 2).**
1. ΔL split into the first token, the success reply's tail (tokens 2 onward) and the failure reply's tail, for every model with stored scores (Qwen3 sizes, Llama-3.1-8B, Gemma-2-9B, Mistral-7B, Llama-3.2-3B and 1B); the tail share is the mean tail shift over the mean ΔL.
2. Per-token log-probabilities within the tail are not stored (the scorer saves the sum and the first token only). Proxies: the tail shift against the success reply's length and against the number of tokens in the object's name, within verb.
3. Copying or priming test: the tail shift under the described (object-free) goal (`rq1x_score`), under "Ignore the {obj}." added to the swapped task, "there is / there is no {obj} here" (`mech2_score`) and "There is a {obj} somewhere in the room." added to the swapped task (`rq1_score`). If any mention of the object's name raises the tail about as much as the aligned task, the tail behaves like name priming; if the object-free goal also raises it, literal copying of the name is not needed.
4. The tail relative to the neutral task: (aligned − neutral) and (swapped − neutral), to see whether the aligned task raises the tail or the swapped task lowers it.
5. What the tail competes with: Qwen's stored greedy predictions (`rq2_score` generations) that begin with "You", by condition.

**D. Heavy tails and concentration (question 3).**
1. ΔL: mean, median, 10% and 20% trimmed means, share of items with ΔL above 10 and 20 nats and below 0.
2. The share of the summed ΔL carried by the top 1%, 5% and 10% of items and by the top 5 and 10 games; the largest change in the mean from leaving out one game.
3. ΔL by step 0 against later steps, task type, failure reason and verb, with each group's contribution to the pooled mean.
4. The same for Llama-3.1-8B and the families.

**E. Scale comparability (question 4).** Per model, in both formats:
1. SD across items of L under each task, mean |L|, SD of ΔL.
2. Standardized effects: ΔL over the SD of L under the swapped task and under the neutral task (where stored), and mean ΔL over the SD of ΔL.
3. Rank scale: the share of items with ΔL > 0, and the probability that a random aligned score exceeds a random swapped score (pooled across items).
4. Probability scale: the shift in σ(L) and in the first-token probability, and sign changes of L and of the first-token L.
5. Distance to the decision boundary in the yes/no format (median swapped-task L) against ΔL, to explain the 0.2% answer flips.

**F. The yes/no puzzle (question 5).** How much of the gap between Qwen's two-candidate (6.12) and yes/no (2.80) effects, and of the gap between Qwen's and the other models' two-candidate effects, is explained by: the success tail (C), step 0 (D), the scale of each model's L (E), and anything else. Also the item-level rank correlation of the two formats' ΔL per model.

The interpretation rules are fixed too. The 6.12 nats are called **mostly measurement** if, after removing the success tail and standardizing by each model's spread, Qwen falls inside the range of the other competent models; **partly real** if it stays above them on the first-token and rank scales but not on the probability scales; **real and large** if it stays clearly above them on the rank scales as well.

The list above is kept exactly as committed before computing (commit 62be254). Two things were added while computing and are labelled where they appear: the share of the effect that happens where both prompts' scores lie far below any decision (section 3.4), and the battery's own item filter for the negation contrasts (section 4.3).

## 1. The question and the short answer

**What was asked.** On RQ2's 4,666 determined failures, Qwen3-8B's goal effect on the two-candidate log-odds is 6.12 nats, against 1.65 for Llama-3.1-8B and 0.8 to 1.1 for Gemma-2-9B, Mistral-7B and Llama-3.2-3B. On the yes/no question it is ordinary (2.80, against 2.21, 2.67 and 1.23), and it changes Qwen's answer on only 0.2% of failures. The audit starts from the assumption that the 6.12 is an artefact and asks which part of it survives.

The two-candidate log-odds of an item under one task line is

$$L = \log P(o^{+} \mid \text{prompt}) - \log P(o^{-} \mid \text{prompt}),$$

where $o^{+}$ is the action's real success reply (for example "You pick up the mug 1 from the cabinet 1.") and $o^{-}$ is "Nothing happens.", each teacher-forced after the prompt. The goal effect is the mean over items of $\Delta L = L_{\text{aligned}} - L_{\text{swapped}}$, where the aligned prompt carries the real task line and the swapped prompt the same line with another object. Splitting each reply into its first token and the rest gives

$$L = \underbrace{\left[\log P(t^{+}_1) - \log P(t^{-}_1)\right]}_{\text{first token}} + \underbrace{\log P(o^{+}_{2:} \mid t^{+}_1)}_{\text{success tail}} - \underbrace{\log P(o^{-}_{2:} \mid t^{-}_1)}_{\text{failure tail}},$$

where the first tokens are "You" and "Nothing" on every item. The stored scores hold each reply's total and its first token, so this split is exact; finer per-token splits are not stored.

**Short answer.** The effect is real in direction and breadth, but its size in nats is not comparable across models, and almost none of the gap between Qwen and the other models survives a common scale.

- It is broad, not carried by a few items. 88% of items move toward success. The median item moves 5.35 nats and the 20% trimmed mean is 5.56. The top 1% of items carry 4% of the total, and dropping any one game changes the mean by at most 0.10. Scores reproduce exactly on a second run on the same GPU type, and the prompts differ only in the task line.
- 59% of it (3.60 nats) is the success tail: the probability of the rest of the success sentence after "You". The first token carries 2.52. Llama has the same 59% tail share, but in Mistral the tail share is negative and in Qwen3-14B it is 10%, so the tail inflates some models' nats and not others'.
- Qwen's scores are on a much wider scale than the other models'. Across items its swapped-task L has an SD of 7.8 nats (Llama 3.1), and its first-token L an SD of 4.9 (Llama 1.0). Divided by the SD of each model's item-level shifts, Qwen's effect is 1.02 against Llama's 0.85 on the two-candidate scale. On the first token it is 0.75, inside the other models' range of 0.31 to 0.81, and on yes/no it is 0.88, below Llama's 1.12.
- Most of it happens where success is unthinkable under both tasks. For the median item, the success sentence's probability goes from 3e-10 to 8e-8. 59% of the summed shift comes from items whose L is below −10 nats under both tasks. Cutting L off at ±10 nats leaves Qwen with 1.25 nats, less than Llama's 1.64.
- The yes/no puzzle has a simple answer. Qwen's first-token effect (2.52) is about the same as its yes/no effect (2.80), so the whole gap between Qwen's two formats is the success tail. The 0.2% flip rate follows from where Qwen's yes/no scores sit: the median failure is at −26.9 nats, and only 0.1% of items lie within 5 nats of the boundary.

By the rule fixed in section 0, the outsized size of Qwen's two-candidate effect is **mostly measurement**: once the tail is removed and each model's spread is accounted for, Qwen falls inside the range of the other competent models. The existence and direction of the effect are not in question.

## 2. Sanity checks on the stored scores (question 6)

Every run read here was checked the same way (`A` in the metrics file).

- **No missing, non-finite, duplicate or stale scores.** All 16 runs have every primary item under every condition the audit uses. No score is NaN or infinite, and no item is scored twice under the same model and condition. Every record's `item_hash`, a hash of the action, history, both task lines and both replies, matches the item rebuilt from the collection.
- **Commits match `reproducing.md`.** The commits are `rq2_score` ddeabc2, `rq1_score` de40795, `rq1x_score` 39722d6, `rq1v_score` 2155b8d, `mech_score` 10a61a7, `size_score_*` 8bfd84c and `family_score_*` 65eb591, one commit per run. `mech2_score` is at 6eab8af, the battery's frozen commit, which `reproducing.md` does not list yet.
- **The prompts differ only in the task line.** This was checked in a CPU job on ICE, job 7245875 in the checkout `text-wm-qwenaudit` (`scripts/qwen_audit/prompt_check.py`).
  - The prompts were re-rendered with the code at the scoring commit (ddeabc2) and with the current code. All 4,666 aligned and 4,666 swapped prompts are byte-identical across the two versions.
  - In every item exactly one message differs between the aligned and swapped prompts. That message contains the real task line exactly once, and replacing it with the swapped line gives the swapped message exactly.
  - Tokenized with the pinned Qwen3-8B and Llama-3.1-8B tokenizers (local snapshots b968826 and 0e9e39f), the two prompts share all tokens except one contiguous span of 1 to 3 tokens. That span always lies inside the task line. In 108 items it does not contain a whole object name, because the two names share a leading or trailing token (soapbar and soapbottle, saltshaker and peppershaker).
- **The candidates tokenize the same way in every condition.** Every stored record of an item has the same token counts and first tokens in all conditions. The success reply starts with "You" and "Nothing happens." is 3 tokens for both tokenizers on all 4,666 items. Re-tokenizing the prompt and reply as one string gives the same reply tokens as tokenizing the reply alone, as the scorer does, so there is no boundary effect.
- **GPU type adds noise, not bias.** The confirmatory shards 0–2 ran on A40s and shard 3 on an L40S; the GPU names were read from each job's nvidia-smi output on ICE (`scripts/qwen_audit/gpu_by_shard.sh`). The size series' rescoring of Qwen3-8B ran on L40S for both of its shards.
  - On the 1,060 items of shard 3 (L40S in both runs), every aligned and swapped score is identical to the last bit.
  - On the 3,606 items scored on an A40 in the confirmatory run, the per-item difference in the effect between the two runs has a mean of −0.01 to +0.01 nats per shard and an SD of about 0.58 nats.
  - The four shards' means (6.00, 5.54, 6.55, 6.44) differ because the shards hold different games: the rescoring reproduces each shard's mean to within 0.01.
  - The secondary scoring of the same items (`rq1_score`) correlates 0.997 with the confirmatory effect item by item, for both models.

Nothing in the stored scores is broken. Whatever is unusual about Qwen's number is in the model's outputs, not in the pipeline.

## 3. Where on the log-odds scale the shift happens (question 1)

### 3.1 The distributions

The table gives the item distribution of L, and of each reply's log-probability, for the two reference models under the three task lines (neutral is "explore the room", from `rq1_score`). Values are nats; the percentiles are across the 4,666 items.

| Model, quantity | Task | Mean | 10th pct | Median | 90th pct |
|---|---|---|---|---|---|
| Qwen3-8B, L | aligned | −14.6 | −23.8 | −16.1 | −2.1 |
| Qwen3-8B, L | swapped | −20.7 | −28.3 | −21.7 | −11.1 |
| Qwen3-8B, L | neutral | −23.4 | −30.4 | −23.5 | −16.1 |
| Qwen3-8B, log P(success reply) | aligned | −15.5 | −23.9 | −16.3 | −5.3 |
| Qwen3-8B, log P(success reply) | swapped | −21.1 | −28.4 | −21.8 | −11.9 |
| Qwen3-8B, log P("Nothing happens.") | aligned | −0.92 | −2.81 | −0.00 | −0.00 |
| Qwen3-8B, log P("Nothing happens.") | swapped | −0.36 | −0.20 | −0.00 | −0.00 |
| Llama-3.1-8B, L | aligned | −0.9 | −4.9 | −0.9 | 3.4 |
| Llama-3.1-8B, L | swapped | −2.6 | −6.6 | −2.6 | 1.6 |
| Llama-3.1-8B, L | neutral | −3.7 | −7.4 | −3.8 | −0.1 |
| Llama-3.1-8B, log P(success reply) | aligned | −3.6 | −7.5 | −3.0 | −0.4 |
| Llama-3.1-8B, log P(success reply) | swapped | −4.8 | −8.7 | −4.5 | −1.0 |
| Llama-3.1-8B, log P("Nothing happens.") | aligned | −2.70 | −4.30 | −2.53 | −1.40 |
| Llama-3.1-8B, log P("Nothing happens.") | swapped | −2.22 | −3.44 | −2.08 | −1.20 |

Qwen's scores live in a different region from Llama's. Under the swapped task the median Qwen item gives the success sentence a log-probability of −21.8, a probability of about 3e-10, and gives "Nothing happens." essentially all the mass. Llama's median item gives the success sentence 1% and "Nothing happens." 12%; the rest goes to other replies (it often writes "You can't find…" or "You arrive at…").

### 3.2 On the probability scale

| Readout | Qwen3-8B, swapped → aligned | Llama-3.1-8B, swapped → aligned |
|---|---|---|
| Median probability of the full success sentence | 3.4e-10 → 8.2e-8 | 0.011 → 0.048 |
| Median probability that the reply starts with "You" | 7.5e-5 → 7.1e-4 | 0.60 → 0.72 |
| Mean two-candidate score (the logistic of L) | 1.7% → 7.6% | 24.3% → 40.7% |
| Items with two-candidate score above 1% | 3.7% → 13.5% | 74.4% → 87.8% |
| Items with two-candidate score above 10% | 2.6% → 10.1% | 45.4% → 65.0% |
| Items with two-candidate score above 50% | 1.6% → 7.4% | 21.5% → 38.8% |
| Items whose success sentence itself has probability above 1% | 2.1% → 9.0% | 50.6% → 68.3% |
| Items whose success sentence itself has probability above 50% | 1.0% → 4.8% | 7.3% → 16.6% |
| Items whose L crosses zero upward | 5.9% | 19.7% |

So, yes: **for the typical item, Qwen's 6 nats are a move between two vanishingly small probabilities**, from about 3e-10 to 8e-8 for the success sentence and from about 1e-4 to 7e-4 for the first word. There is also a minority of items, roughly 6% of all, where the aligned task takes Qwen from predicting failure to giving success a real chance (the share above 50% rises from 1.6% to 7.4%). On every probability readout Llama moves more items by more.

### 3.3 Which reply moves

The effect can come from the success reply becoming more likely, from "Nothing happens." becoming less likely, or both. For Qwen the success sentence's log-probability rises by 5.55 nats [4.98, 6.12] and "Nothing happens." falls by 0.57 [0.49, 0.66], so 91% of the effect is the success reply rising. That is forced by the scale: "Nothing happens." already has probability near 1 under the swapped task, so its log-probability cannot fall much without the success reply gaining a lot. The success reply rises on 88% of items and "Nothing happens." falls on 81%. For Llama the split is 1.16 and 0.49 (70% and 30%).

### 3.4 Where the shift sits (partly added during the analysis)

Sorting items by their swapped-task L shows where the nats are made. In Qwen's lowest tenth (swapped L between −50 and −28) the effect is 10.7 nats, 8.5 of them in the success tail. In its highest tenth (swapped L above −11) the effect is 5.2 nats, 4.4 of them in the first token. The deeper an item already is in "certain failure", the more of its shift is tail.

Added while computing, to ask the question in 3.2 directly: how much of the summed effect comes from items where both prompts' L are below a level at which no decision could change? Below −10 nats both two-candidate scores are under 0.005%. For Qwen, 59% of the summed shift comes from items below −10 under both tasks (76% of items), and 74% from items below −5 under both (86% of items). For Llama the shares are 0.1% and 5%. A related readout cuts each L off at ±c before taking the difference, which keeps every move inside the band and none of the movement beyond it. Written out,

$$\Delta L^{(c)} = \frac{1}{n}\sum_{i}\left[\operatorname{clip}(L_{i,\text{aligned}}, -c, c) - \operatorname{clip}(L_{i,\text{swapped}}, -c, c)\right].$$

| Model | Effect, nats | Cut off at ±10 | Cut off at ±5 | First token, cut off at ±5 | Yes/no, cut off at ±5 |
|---|---|---|---|---|---|
| Qwen3-8B | 6.12 | 1.25 | 0.60 | 0.94 | 0.02 |
| Llama-3.1-8B | 1.65 | 1.64 | 1.41 | 0.68 | 1.72 |
| Mistral-7B | 1.06 | 1.04 | 0.89 | 1.15 | 2.37 |
| Llama-3.2-3B | 0.98 | 0.97 | 0.77 | 0.69 | 0.70 |
| Gemma-2-9B | 0.80 | 0.77 | 0.37 | 0.40 | 0.27 |

Within ±10 nats of the boundary Qwen's two-candidate effect is smaller than Llama's. The cut-off is a choice, not a law, and it throws away real movement in the tails. But it shows that the gap between the models lives almost entirely in a region where neither prompt's score could support a success prediction.

## 4. Token decomposition (question 2)

### 4.1 First token against the tail, every model

| Model | Effect, nats | First token | Success tail | Failure tail | Tail share of the effect | Tail share, later steps only |
|---|---|---|---|---|---|---|
| Qwen3-8B | 6.12 [5.52, 6.74] | 2.52 [2.27, 2.78] | 3.60 [3.14, 4.05] | 0.00 | 59% [56, 62] | 66% |
| Llama-3.1-8B | 1.65 [1.51, 1.78] | 0.68 [0.61, 0.75] | 0.97 [0.85, 1.09] | 0.01 | 59% [55, 63] | 63% |
| Gemma-2-9B | 0.80 [0.48, 1.10] | 0.44 [0.28, 0.60] | 0.36 [0.17, 0.54] | 0.00 | 45% [32, 53] | 50% |
| Mistral-7B | 1.06 [0.90, 1.23] | 1.25 [1.09, 1.41] | −0.13 [−0.21, −0.05] | 0.05 | −12% | −15% |
| Llama-3.2-3B | 0.98 [0.82, 1.14] | 0.69 [0.60, 0.78] | 0.29 [0.20, 0.38] | −0.01 | 29% [23, 35] | 35% |
| Llama-3.2-1B (fails the competence rule) | 0.31 | 0.24 | 0.06 | −0.01 | 20% | 13% |
| Qwen3-0.6B (cannot do the task) | 0.89 | 0.71 | 0.19 | 0.00 | 21% | 22% |
| Qwen3-1.7B (cannot do the task) | 2.49 | 2.68 | −0.19 | 0.00 | −8% | −11% |
| Qwen3-4B | 7.49 [6.46, 8.49] | 6.20 [5.37, 6.97] | 1.29 | 0.00 | 17% | 21% |
| Qwen3-8B (rescored) | 6.12 | 2.51 | 3.61 | 0.00 | 59% | 66% |
| Qwen3-14B | 4.33 [3.84, 4.81] | 3.89 [3.51, 4.24] | 0.44 | 0.00 | 10% | 9% |
| Qwen3-32B | 2.35 [2.07, 2.63] | 1.20 [1.08, 1.32] | 1.16 | 0.00 | 49% | 56% |

The 59% tail share in `textcraft_vs_alfworld.md` reproduces exactly for Qwen3-8B, and it is the same for Llama. **So the tail share is not what makes Qwen large relative to Llama**: Qwen's first token alone is 3.7 times Llama's (2.52 against 0.68), the same ratio as the full effect. Against the other three families it is different. Their tail shares are lower (45%, 29%) or negative (Mistral), so the tail adds 3.2 nats to Qwen's gap over them and the first token 1.8 (section 7). Within the Qwen3 family the tail share jumps around (17%, 59%, 10%, 49% for 4B to 32B), and Qwen3-4B and 14B have larger first-token effects than 8B. The tail term is a property of each model's way of continuing "You", not a stable part of the goal effect.

"Failure tail" is zero for every model, as expected: after "Nothing" every model puts essentially all its mass on " happens.".

### 4.2 Is the tail shift copying of the object's name?

The worry is that the aligned task names the object ("…some mug…"), the success sentence repeats it ("You pick up the mug 1…"), and the model simply finds a sentence easier to predict when its words were primed, whatever it believes about success. Per-token log-probabilities would settle this directly, by showing whether the tail shift sits on the object-name tokens or on the token right after "You" (" pick" against " try", " cannot"). They are not stored, and computing them needs a GPU, which this audit does not use. Four indirect tests point away from literal copying:

1. **Object-free goals keep the same tail share.** When every object name in the task line is replaced by a description that names no ALFWorld object (`rq1x_score`), Qwen's effect is 3.71 nats, of which 2.11 (57%) is tail. That is the same share as with the named goal, with nothing to copy.
2. **The tail follows what a sentence says, not only whether it names the object.** In the mechanism battery (`mech2_score`, its own 503-item population for the negation contrasts), two task lines both name the object and the receptacle and differ only in "there is no" against "there is a". The negated line lowers Qwen's L by 11.83 nats [10.87, 12.87], of which 6.63 is tail. With the names held fixed, the tail moves by more than the whole aligned effect. In the negated-against-swapped contrast the first token rises (+1.73) while the tail falls (−1.48), so the two parts can even move in opposite directions.
3. **No dose–response in the object name's length.** If copying drove the tail, objects spelled with more tokens should gain more. Within each verb the tail shift does not rise with the number of tokens in the object's name; if anything it falls (Spearman −0.48 for clean, −0.26 for move, −0.17 for take), a pattern that comes from which objects have long names rather than from length. Nor does the tail shift grow with the reply's length within a verb.
4. **What Qwen writes after "You".** Qwen's greedy predictions (stored with the confirmatory run) mostly say "Nothing happens." (3,919 aligned, 4,362 swapped). When they start with "You", many are failure explanations: "You try to…", "You have no…", "You cannot cool…", "You are carrying…", "You do not…". Once "You" is forced, Qwen still chooses between continuing the success sentence and explaining a failure. The tail is therefore largely a second reading of the same success judgement, taken after the first word, not a measure of wording alone.

The same tests also show something the copying worry gets right in another form: Qwen's L rises whenever the object is made salient, whatever role it plays. On the same items, these all raise Qwen's L by about the full goal effect, with the tail rising about as much as the first token in each:

- adding "There is a {obj} somewhere in the room." to the swapped task (6.00 nats; 2.82 first token, 3.18 tail);
- adding "Ignore the {obj}." (6.43; 3.21 and 3.22);
- the battery's mention arm (6.13).

Llama behaves differently: "Ignore the {obj}." raises its tail (+0.63) but lowers its first token (−0.13), which is closer to the copying pattern. So for Qwen the tail is not a separate copying channel layered on top of the judgement. It goes up and down with the first token in every contrast with a sizeable effect (tail shares 0.43 to 0.68 across the mention, description, negation, facts and rules contrasts), except where the content of the sentence pulls the two apart. Item by item the two parts are only weakly correlated (Spearman 0.17), so they are noisy readings of a shared shift, not copies of each other.

### 4.3 How these contrasts were computed

Every contrast in 4.2 is a difference of means between two stored conditions of the same run and model, over the items scored in both, with the same game-clustered bootstrap. The battery's negation contrasts were first computed on all 522 take items of its sample and gave −11.37. Re-filtered to the battery's own population (added during the analysis), which drops items whose history shows the receptacle holding the object, they give the published −11.83 on 503 items exactly. The full table, with Llama, is under `C3_contrasts` in the metrics file.

## 5. Heavy tails and concentration (question 3)

| Model | Mean | Median | 20% trimmed mean | Items above 10 nats | Items above 20 nats | Items below 0 | Top 1% of items' share of the total | Top 5 games' share of the total (and of items) | Largest change from dropping one game |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 6.12 | 5.35 [4.73, 5.90] | 5.56 [4.97, 6.19] | 22.2% | 2.4% | 12.1% | 4.3% | 11.1% (8.7%) | 0.10 |
| Llama-3.1-8B | 1.65 | 1.38 | 1.45 | 0% | 0% | 17.4% | 5.0% | 11.0% (8.7%) | 0.03 |
| Mistral-7B | 1.06 | 0.95 | 1.01 | 0% | 0% | 27.4% | 6.5% | 14.4% (8.7%) | 0.02 |
| Gemma-2-9B | 0.80 | 0.71 | 0.74 | 0.4% | 0% | 34.4% | 11.9% | 24.3% (8.7%) | 0.05 |
| Llama-3.2-3B | 0.98 | 0.81 | 0.88 | 0% | 0% | 21.9% | 5.4% | 17.0% (8.7%) | 0.03 |

Qwen's effect is right-skewed, as every model's is, but **it is not carried by a handful of items or games**. Without the top 1% of items its mean is 5.92. Its game means are positive in 95% of games (median game 6.5 nats, 10th percentile 1.9). Its median and trimmed mean are 3.9 and 3.8 times Llama's, almost exactly the ratio of the means.

The groups that matter are the ones already known:

- **Step 0.** The 514 items with no history (11%) average 11.0 nats and contribute 1.21 of the 6.12. Later steps average 5.52. All of the step-0 excess is on the first token (7.50 against 1.90 later). The tail is the same at step 0 and later (3.49 against 3.62).
- **Verb.** Clean actions average 10.8 nats, 8.2 of them tail ("You clean the mug 1 using the sinkbasin 1."). Cool averages 8.0, heat 6.1, move 6.0 and take 5.2. Their first-token effects are similar (1.7 to 2.7), so the verb differences are almost entirely tail.
- **Failure reason and task type.** "Not at the receptacle, never moved" is the step-0 group (10.9). The other reasons lie between 4.2 (contents seen) and 6.0. By task type the effect runs from 3.2 (pick two) to 8.7 (cool).

## 6. Scale comparability (question 4)

The same nat means different things in different models, because models spread their scores over very different ranges. Three scale-free readouts were computed per model and format.

- **The standardized effect.** It is the mean shift divided by the SD of the item-level shifts (defined below).
- **The share of items moved toward success.**
- **The pooled AUC.** It is the probability that a random item's aligned score beats a random item's swapped score.

In symbols, the standardized effect is

$$d_z = \frac{\overline{\Delta L}}{\operatorname{SD}(\Delta L)}.$$

| Model | SD of swapped L, two-candidate / first token / yes-no | Standardized effect, two-candidate | Standardized effect, first token | Standardized effect, yes/no | Items moved toward success, two-candidate / first token / yes-no | Pooled AUC, two-candidate / first token / yes-no |
|---|---|---|---|---|---|---|
| Qwen3-8B | 7.80 / 4.91 / 2.78 | 1.02 [0.91, 1.14] | 0.75 [0.68, 0.82] | 0.88 [0.81, 0.94] | 0.88 / 0.81 / 0.89 | 0.72 / 0.64 / 0.76 |
| Llama-3.1-8B | 3.13 / 1.03 / 2.37 | 0.85 [0.78, 0.92] | 0.68 [0.61, 0.74] | 1.12 [1.03, 1.21] | 0.83 / 0.75 / 0.90 | 0.64 / 0.66 / 0.75 |
| Mistral-7B | 3.79 / 2.04 / 2.77 | 0.52 [0.44, 0.61] | 0.74 [0.64, 0.84] | 0.96 [0.88, 1.05] | 0.73 / 0.79 / 0.88 | 0.59 / 0.67 / 0.75 |
| Llama-3.2-3B | 2.55 / 1.09 / 0.86 | 0.75 [0.64, 0.86] | 0.81 [0.71, 0.92] | 1.00 [0.88, 1.12] | 0.78 / 0.79 / 0.84 | 0.61 / 0.68 / 0.76 |
| Gemma-2-9B | 4.12 / 2.10 / 2.01 | 0.31 [0.18, 0.44] | 0.31 [0.18, 0.44] | 0.49 [0.39, 0.59] | 0.66 / 0.69 / 0.75 | 0.56 / 0.55 / 0.67 |
| Qwen3-4B | 10.87 / 9.52 / 7.57 | 0.90 | 0.82 | 0.27 | 0.83 / 0.82 / 0.49 | 0.68 / 0.68 / 0.51 |
| Qwen3-14B | 7.79 / 5.75 / 5.16 | 0.96 | 1.09 | 0.46 | 0.86 / 0.90 / 0.70 | 0.66 / 0.66 / 0.59 |
| Qwen3-32B | 3.98 / 3.05 / 3.47 | 0.86 | 0.60 | 0.46 | 0.82 / 0.74 / 0.72 | 0.67 / 0.60 / 0.62 |

Qwen's two-candidate scores spread 2.5 times as widely as Llama's across items, and its first-token scores 4.8 times as widely. Its item-level shifts spread wider in the same proportion: their SD is 5.98 nats against Llama's 1.93. On the scale-free readouts:

- **Two-candidate.** Qwen is still the highest of the five, but by 1.2 times Llama (1.02 against 0.85), not 3.7 times. Its share of items moved (0.88 against 0.83) and its AUC (0.72 against 0.64) are the highest too. Part of this edge is the tail, which Qwen has and Mistral and Llama-3.2-3B largely lack.
- **First token.** Qwen is inside the range: 0.75, against 0.68 (Llama-3.1-8B), 0.74 (Mistral), 0.81 (Llama-3.2-3B) and 0.31 (Gemma). Its share moved (0.81) and AUC (0.64) are typical.
- **Yes/no.** Qwen is below Llama, Mistral and Llama-3.2-3B (0.88 against 0.96 to 1.12), and its share moved (0.89) and AUC (0.76) are the same as theirs.

Expressed in Llama's units (Qwen's standardized effect times the SD of Llama's shifts), Qwen's two-candidate effect is about 2.0 nats against Llama's 1.65, and its first-token effect about 0.76 against 0.68.

The size series tells the same story. In nats, the competent Qwen3 sizes fall from 7.49 to 2.35 as size grows. Standardized, they are flat: 0.90, 1.02, 0.96 and 0.86. What changes with size is how far each model spreads its scores (the SD of the swapped L falls from 10.9 to 4.0), not how strongly the goal moves them relative to that spread.

**The 0.2% answer flips.** A flip needs an item near the boundary. In the yes/no format Qwen's median failure sits at −26.9 nats under the swapped task. 0.0% of items lie within one average shift (2.8 nats) of the boundary and 0.1% within 5 nats. 92% of Qwen's summed yes/no shift happens on items below −10 nats under both tasks. Llama's median failure sits at −4.4 nats, with 21% of items within one average shift and 59% within 5 nats; it flips 16.5%. Mistral's sits at −0.25 and flips 37.5%. The flip rate is set by where each model's scores start, not by how far the goal moves them. On the two-candidate scale Qwen does flip more, 5.9% upward sign changes of L and 9.2% of the first token, because a minority of its items do sit near the boundary there (section 3.2).

## 7. The yes/no puzzle (question 5)

There are two gaps to explain. One is within Qwen: its two-candidate effect is 6.12 and its yes/no effect 2.80, a difference of 3.32 nats [2.85, 3.78]. The other is across models: Qwen's two-candidate effect exceeds the mean of the four other competent families (Llama-3.1-8B, Mistral, Llama-3.2-3B and Gemma; 1.12 nats) by 5.00 nats, while on yes/no it does not stand out. The pieces below overlap, so each is reported as what removing it alone would do, not as an additive split.

**Within Qwen, between formats.**

| Candidate explanation | What it explains of the 3.32-nat gap |
|---|---|
| 1. The success tail | All of it. The tail is 3.60, and the first token minus yes/no is −0.29 [−0.51, −0.08]: Qwen's first-token effect and its yes/no effect agree. |
| 2. Step 0 | None. The gap is 3.28 at step 0 and 3.32 at later steps; step 0 raises both formats about equally (11.0 and 7.7 at step 0). |
| 3. Heavy tails | About 3%. With 20% trimming the gap is 3.22 (5.56 against 2.34). |
| 4. Scale | Most of it. Standardized, the two formats are 1.02 and 0.88. The tail adds spread as well as mean, so the standardized gap is small. |

Stated plainly, the two formats ask the same question and get the same answer. "Will this succeed?" is moved by 2.8 nats, and "is the next word You or Nothing?" by 2.5. The two-candidate number then adds 3.6 nats for the rest of the success sentence.

**Across models, two-candidate format.**

| Candidate explanation | What it explains of Qwen's 5.00-nat excess over the other four families |
|---|---|
| 1. The success tail | 3.23 nats (65%): Qwen's tail is 3.60 against the others' mean of 0.37. |
| What remains on the first token | 1.75 nats (35%): 2.52 against a mean of 0.77. |
| 2. Step 0 | About 0.5 nats (11%): later steps only, the excess is 4.47 (5.52 against a mean of 1.05). |
| 3. Heavy tails | About 0.5 nats (9%): with 20% trimming, the excess is 4.54 (5.56 against a mean of 1.02). |
| 4. Scale | The first-token remainder in full. Standardized, Qwen's first-token effect (0.75) is inside the others' range (0.31 to 0.81), and its two-candidate effect is 1.2 times Llama's. |
| Where the shift happens | Cutting L off at ±10 nats leaves Qwen 1.25 against the others' 0.77 to 1.64. |

So the gap between Qwen and the other models on the two-candidate scale is about two thirds success tail and one third first-token nats. The first-token third disappears when each model's spread is taken into account. Step 0 and heavy tails each account for about a tenth, and they overlap with the other pieces. The item-level rank correlation between Qwen's two-candidate and yes/no shifts is 0.48 (first token against yes/no 0.42), against 0.62 to 0.74 for Llama, Mistral and Gemma. Qwen's two formats agree on the mean but less closely item by item, which again fits a tail term that adds its own variation.

## 8. Interpretation

**What the 6.12 nats are made of.** About 2.5 nats are a shift in the decision word "You" against "Nothing". That shift is real: it is broad across items and games, it is matched by the yes/no question, and it is about as large as other models' shifts once each model's scale is accounted for. About 3.6 nats are the rest of the success sentence. This part is mostly a second reading of the same judgement after "You" has been forced, not literal copying of the object's name: it survives object-free goals and responds to negation with the names held fixed. But how large it is depends on each model's habits after "You", so it does not travel across models. Under both tasks the shift happens mostly where Qwen gives success no realistic chance: 59% of it comes from items whose L is below −10 nats under both. Step 0 adds about a fifth of the mean (1.2 nats) and does not change any of these readings.

**How much is "real belief shift"?** That depends on the scale, so three answers are given:

- **Direction and breadth.** All of it is real: 88% of items move toward success, as in other models.
- **Size relative to other models.** On a common scale Qwen's goal effect is about 1.2 times Llama's on the two-candidate score, about 1.1 times on the first token, and 0.8 times on yes/no. The claim that Qwen is 3 to 7 times more goal-sensitive than other models is an artefact of comparing nats across models with different spreads, plus the success tail.
- **Practical consequence.** It is small for Qwen. The mean two-candidate score moves 5.9 points against Llama's 16.4. Answers change on 0.2% of yes/no failures against 16.5%. Cut off at ±10 nats, the effect is 1.25 against 1.64. The exception is the roughly 6% of items where the aligned task pushes Qwen's two-candidate score above one half.

**What this implies for reporting.** None of this touches the preregistered RQ2 verdict, which tests each model against zero on its own scale; 6.12 [5.48, 6.75] stays the confirmatory number. For the paper, as exploratory guidance:

- Do not compare two-candidate nats across models, or across Qwen sizes, as if they were one scale. Lead cross-model comparisons with a scale-free readout (the standardized effect, the share of items moved, the AUC) and the yes/no format, and give nats per model.
- Report the first-token and tail split next to every two-candidate number, and say that the tail is a model-specific continuation term.
- Say that, for Qwen3-8B, most of the shift happens far below any decision threshold, and pair it with a probability or flip readout.
- The size series should be described as flat in standardized terms, not as falling with size.
- The same caution applies to any other Qwen contrast quoted in nats, including the mechanism battery's +6.43 and −11.83. The battery's own ratio to the aligned effect, r, is the scale-free version within a model. The τ-bench first-token figures were not examined here.

**What would change this reading.**

- **Per-token log-probabilities of the success sentence** (GPU) would show where in the sentence the tail shift sits. If it sits on the object-name tokens, the tail is closer to name copying than this audit's indirect tests suggest. If it sits on the token after "You", it is the second judgement described here. Either way it stays model-specific, so the comparability conclusion would not change.
- **A different standardizer** could reorder the models; none was found that does here. The SD of the shifts, the SD of the swapped scores, the share of items moved and the AUC all put Qwen within the other models' range on the first token and on yes/no.
- **A competent model with a scale as wide as Qwen's but a clearly smaller standardized effect** would weaken the scale argument. Qwen3-4B and 14B have similarly wide scales and similar standardized effects.

## 9. Reproducing

The stored scores were copied read-only from ICE with `scripts/qwen_audit/fetch.sh` into the git-ignored `runs/_qwen_audit/`. The checkouts are text-wm-fix11 (`rq2_score`, with its stored greedy predictions), fix12 (`rq1_score`), fix13 (`rq1x_score`), fix17 (`rq1v_score`), fix18 (`size_score_*`), fix19 (`mech_score`), fix21 (`family_score_*`), fix36 (`mech2_score`) and fix9 (the collection). Then, from the repository root:

```bash
bash scripts/qwen_audit/fetch.sh
ssh ice 'bash -s' < scripts/qwen_audit/gpu_by_shard.sh > runs/_qwen_audit/gpu_by_shard.tsv
# On ICE, in a fresh checkout of this branch at /storage/ice1/2/1/gmat3/text-wm-qwenaudit, with the scoring
# commit's code extracted by: git archive ddeabc2 wf | tar -x -C old_ddeabc2
sbatch scripts/qwen_audit/prompt_check.sbatch    # CPU, under 2 minutes; copy runs/_qwen_audit/prompt_check.json back
RUNS_DIR=runs/_qwen_audit/fix9 PYTHONPATH=. python scripts/qwen_audit/analyze.py \
  --out notes/qwen_effect_audit_forensic_metrics.json
python scripts/qwen_audit/report.py notes/qwen_effect_audit_forensic_metrics.json   # prints the tables above
```

The analysis takes about two minutes on a laptop CPU, with 2,000 game-clustered bootstrap resamples (seed 20261122; 500 for medians, trimmed means and standardized effects). Every number in this note is in `notes/qwen_effect_audit_forensic_metrics.json`, except the means of the four other families quoted in section 7, which are averages of that file's per-model values. No model was run, and no ICE checkout or run was modified; the only ICE writes are in the new checkout `text-wm-qwenaudit`.


## Follow-up: the pipeline audit (2026-11-23)

The second, independent audit (`notes/qwen_effect_audit_pipeline.md`, 2026-11-23) re-implemented the Qwen scoring on GPU: a scorer written from scratch reproduces the stored scores exactly on the same GPU type, and vLLM and float32 weights agree, so no bug produced the number. About half of the effect's size depends on one prompt-format detail, the empty thinking-off block that Qwen3's chat template places before the answer: on 200 sampled items the effect is 7.34 nats with it and 3.87 without it. In every format tried it stays positive and 2 to 5 times Llama-3.1-8B's. The confirmatory result is not re-verdicted.
