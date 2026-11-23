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
