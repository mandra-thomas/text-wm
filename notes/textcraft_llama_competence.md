# TextCraft: how competent is Llama-3.1-8B as a world model, and what does it do instead? (exploratory, post hoc)

Everything in this note is exploratory and post hoc. It was written on 2026-11-21, after the confirmatory TextCraft results ([textcraft_results.md](textcraft_results.md)), the Llama diagnostics ([textcraft_llama_dx.md](textcraft_llama_dx.md)) and the cross-environment comparison (`notes/textcraft_vs_alfworld.md` on branch `xenv-compare`) had been read. It uses only scores that already exist; nothing was scored for it. It changes none of the confirmatory verdicts.

## Summary

All of this is exploratory and post hoc. It reads only existing scores: the confirmatory TextCraft run, one exploratory prompt variant, and the archived ALFWorld runs. It changes no confirmatory verdict.

- **Llama-3.1-8B is not a working TextCraft world model.** Under the neutral goal line its AUC between determined successes and the 1,500 main failures is 0.61 [0.58, 0.64] on the two-candidate scale and 0.61 [0.58, 0.65] on yes/no, against 0.86 and 0.85 for Qwen3-8B. Its predicted probabilities carry almost no information above the lowest bin: among items it gives a probability of success above 0.9, 32% succeed, against 30 to 46% in the middle bins. It fails the model-families competence rule on the AUC criterion (0.61 against the required 0.70). It passes the other three criteria, and it passes the TextCraft design's own, looser rule. Qwen passes both rules. In ALFWorld the same Llama model passes the families rule easily (yes/no AUC 0.97), so the incompetence is specific to TextCraft.
- **What Llama does instead.** Ten heuristics were fixed before any was fitted. Llama's scores follow "every input's name appears in the history" (tracking AUC 0.87) and "every input was obtained in the needed amount at some point, ignoring what later crafts used up" (0.82). Both fit far better than the true inventory rule (0.61). Holding either heuristic fixed, the true outcome adds nothing to Llama's scores: its slope is −0.55 and −2.18 nats. Llama also treats any craft of a finished goal as likely to fail (0.80). The pre-registered combination of the two, "ignores consumption, and goal crafts fail", agrees best with Llama's decisions (kappa 0.57, against 0.06 for the truth). It reproduces the used-up result: by its own rule it ranks used-up failures above successes, with AUC 0.26 against Llama's 0.31. Most of that 0.31 is role composition. Within ingredient crafts alone, Llama's used-up AUC is 0.43, and a residual below one half remains that no listed heuristic explains. Qwen's scores follow the true rule better than any heuristic except the names one.
- **Quantities.** Llama does respond to *how many* units were obtained: within "too few" failures its L falls with the shortfall (Spearman −0.48, about as strong as Qwen's −0.55). It does not respond to consumption. Within used-up failures its L stays above zero however long ago the input was used up (ρ +0.13, the wrong sign). Its L is lower when an inventory reply after the consumption showed the reduced count (+0.59 against +1.98). Within successes, its L *falls* as the surplus grows.
- **Competence and the goal effect.** Across 14 role × class × length strata, Llama's ΔL is more negative where its neutral discrimination is better (Spearman −0.49 two-candidate, −0.76 yes/no). The negative goal-role effect sits where Llama is competent (AUC 0.84 to 0.95). On ingredient crafts, the goal effect is null both where Llama is incompetent (used up, AUC 0.43; ΔL −0.02) and where it is competent (never obtained, AUC 0.86; ΔL −0.11 [−0.29, +0.06]). In that last cell Qwen shows +1.64.
- **Implications.** Incompetence explains why Llama's TextCraft result cannot be read as a statement about a working world model. It does not explain the goal-role negative, which sits where Llama discriminates. It also does not by itself explain the absent positive effect, which is also missing in the one ingredient cell Llama gets right. The confirmatory verdict stands as reported ("confirmed for one model only"). Under the families rule Llama's run would have been classed as not interpretable, a disclosed observation and not a re-verdict. The paper should describe Llama's TextCraft result as that of a model that does not track the inventory. It should not call it a robust or a reversed goal effect of a world model. Round 2 should expect Llama-3.1-8B, and very likely Llama-3.2-3B, to fail its competence gate.

## What this note asks

The confirmatory TextCraft run found that Llama-3.1-8B-Instruct separates working from failing crafts only weakly under the neutral goal line (AUC 0.68 with the usefulness failures, 0.61 on the main failures alone), and that it ranks "used up" failures above real successes (AUC 0.31). This note asks five things, with Qwen3-8B (thinking off) alongside as the reference throughout:

1. The full competence profile: how well each model separates successes from failures, on both reply scales, by failure class, role and history length; how often it predicts success; how calibrated its probabilities are; and whether it passes the competence rule the model-families study fixed in advance.
2. What Llama does instead of tracking the inventory: which simple rule, computable from the prompt text, its scores follow.
3. Whether its scores respond to the quantities that decide the outcome (how many units are missing, how long ago an input was used up, how many units are spare).
4. How competence relates to the goal effect, item by item and stratum by stratum.
5. Whether the incompetence is specific to TextCraft, by comparison with Llama in ALFWorld and with Qwen in TextCraft.

It ends with what the incompetence does and does not explain about the TextCraft results, and how the paper should describe the Llama result.

## Heuristics and fit measures (fixed 2026-11-21, before any fit to either model was computed)

This section was committed on its own before any heuristic was compared with any model's scores, so that the set of candidate explanations is not chosen after seeing which one fits. At the time of writing, what was known about the models' scores was what the earlier notes report (overall and per-class AUCs, the role pattern, the used-up AUC of 0.31). No heuristic had been computed on any item.

### Items and notation

The items are the 1,500 main failures and the 600 determined successes of the confirmatory run (`wf/textcraft/data/items.jsonl`, populations `main` and `success`; 2,100 items from 247 goal pairs; the successes come from 167 of them). Every test action is a craft, `craft K Y using c1 n1, c2 n2, ...`, with inputs $(c_j, n_j)$: $c_j$ units of item $n_j$. The outcome is success exactly when every input is held in at least its count just before the action.

Every heuristic reads only what the prompt shows: the displayed recipe lines, the history's (action, reply) pairs and the action. In the history:

- an *obtaining step* for an item X is a reply `Got N X` (N units obtained) or `Crafted K minecraft:X` (K units obtained, the underscores read as spaces);
- a *consuming step* for X is a history craft whose action lists X as an input (the history's crafts always succeed, so the listed units were removed);
- an item is *mentioned* at a step when its name appears there as an item: the object of a `get`, an input or the output of a `craft`, the object of a `Got` or `Crafted` reply, or an entry of an `Inventory:` reply.

The held amount $h_j$ of an input is the text-only checker's accounting (`wf/textcraft/checker.account`), the same accounting that defines the outcome.

### The heuristics

Each heuristic predicts success or failure for an item. Several also have a graded score, used for rank correlations; higher means "more likely to succeed".

- **H1, name seen.** Success when every input's name is mentioned somewhere in the history. It ignores counts and consumption.
- **H2, gross obtained (ignores consumption).** Success when, for every input, the total number of units obtained over the history (the sum of N and K over its obtaining steps) is at least $c_j$. It counts what came in and ignores what went out. Graded score: the smallest, over the inputs, of the obtained total divided by $c_j$, capped at 1.
- **H3, obtained recently.** Success when every input was obtained in one of the last $k$ history steps, with $k = 3$ as the main setting and $k = 1$ and $k = 6$ reported as sensitivity. It ignores counts and consumption.
- **H4, mentioned recently.** Success when every input's name is mentioned in one of the last 3 history steps, a mention as the input of a craft (its consumption) included. Graded score: minus the number of steps since the least recently mentioned input was last mentioned (an input never mentioned counts as the history length plus one).
- **H5, long history.** Success when the history has at least 4 steps (the lower edge of the middle length bin). Graded score: the history length.
- **H6, true accounting (the environment's rule).** Success when $h_j \ge c_j$ for every input. This is the outcome itself, so its accuracy is 1 by construction. Graded score: the smallest, over the inputs, of $h_j / c_j$, capped at 1.
- **H7, held at all (tracks consumption, ignores counts).** Success when $h_j \ge 1$ for every input.
- **H8, last event is a gain.** Success when, for every input, its most recent obtaining or consuming step is an obtaining step (an input with neither counts as failing). It notices a consumption only if it is the input's last event, and ignores counts.
- **H9, not a top item.** Failure when the action's output is a top item of the displayed list, that is, an item that no displayed line takes as an input (a `planks` or `wool` input counts as taking each of its members); success otherwise. In this design the top items are exactly the two goals of the pair, so under the neutral line, which names neither, this is "crafting a finished goal fails".
- **H10, always fail.** Predicts failure for every item. It has no graded score.
- **H2 ∧ H9 (one combination, fixed now).** Success when H2 predicts success and the output is not a top item.

### How a heuristic is compared with the truth and with a model

All comparisons use the neutral goal line ("Goal: explore the crafting commands."), the two-candidate log-odds L as the main scale and the yes/no log-odds as the second. Intervals are 95% percentile intervals from 2,000 resamples of whole goal pairs, as in the frozen analysis. There are no p-values.

1. **Validity.** The AUC of the heuristic's prediction (or graded score) against the true outcome, with ties counted one half, over all 2,100 items and for each failure class against all successes. This says how good a world model the heuristic would be.
2. **Tracking AUC (the main fit measure).** For a binary heuristic, the AUC of the model's L between the items the heuristic calls a success and those it calls a failure: how well the model's ranking follows the heuristic. For H10 it is undefined. For the true rule H6 it equals the model's competence AUC, so each heuristic's tracking AUC can be set against the truth's.
3. **Rank correlation.** For heuristics with a graded score, the Spearman correlation between that score and L.
4. **Decision agreement.** The share of items on which the model's decision (success when L > 0) agrees with the heuristic, and Cohen's kappa, which corrects that share for agreement expected by chance from the two base rates. Because Llama predicts failure for most items, raw agreement favours H10; kappa is the comparable figure.
5. **Within the truth.** The tracking AUC computed separately among the true failures and among the true successes. A model that follows the truth and not the heuristic has a tracking AUC near 0.5 within each; a model that follows the heuristic has one above 0.5 within the failures, where the heuristic and the truth disagree.
6. **Joint regression.** A least-squares fit of L on an intercept, the true outcome (0 or 1) and the heuristic's prediction (0 or 1), with pair-bootstrap intervals for the two slopes. It says how much of the model's separation goes with the truth once the heuristic is held fixed, and the reverse.

**Reading, fixed now.** The best-fitting heuristic is the one with the highest tracking AUC for Llama over all 2,100 items (for graded heuristics, the highest Spearman correlation is reported beside it). A heuristic *explains* Llama's used-up result if (a) it does not call used-up failures less likely to succeed than real successes (its own AUC, used up against successes, is at or below 0.5), and (b) among used-up failures and successes together, Llama's L follows it more closely than it follows the truth (higher tracking AUC than H6 on that subset). The same fits are computed for Qwen, as a model that does track the state. As a secondary check, H2 and H6 are also fitted on the exploratory inventory-shown prompt (`textcraft_dx_prompts`, 500 items), where the inventory before the action is on screen.

The results follow. One check in section 2 was added after these fits had been read; it is labelled post hoc where it appears.

## Data and method

**Scores.** The confirmatory scoring run `textcraft_score` (51,200 records, ICE jobs 7204417–7204420 at `5a3702a`). The four shard files were copied off the cluster for the earlier diagnostics. Their SHA-256 hashes were checked on 2026-11-21 against the originals in `text-wm-fix24/runs/textcraft_score/shard_0{0..3}_of_04/scoring.jsonl`, and all four match. The items file's hash matches the verified manifest. For the inventory-shown check, the exploratory run `textcraft_dx_prompts` (ICE job 7207018, 9,000 records on 500 items). For ALFWorld, the archived runs `rq1_score` (neutral "explore the room" two-candidate scores on the 4,666 failures; `text-wm-fix12`), `rq1x_score` (the same on the 292 determined successes; `text-wm-fix13`) and `rq1v_score` (yes/no on both; `text-wm-fix17`), read from the local copy of the run archive.

**Populations.** Unless stated otherwise, TextCraft competence uses the 1,500 main failures and the 600 determined successes. The confirmatory competence figure (0.68) also included the 600 usefulness failures, which are never-obtained ingredient crafts and therefore easy. With them, Llama's neutral two-candidate AUC is 0.68 [0.65, 0.71] and its yes/no AUC 0.67 [0.64, 0.70]; without them, both are 0.61. ~~The design's own competence gate used the aligned line, on which Llama also gives 0.681, the same to three decimals as the neutral line by coincidence. (The round-2 design note on branch `textcraft-r2` describes the 0.68 as the aligned-line figure. That is correct, but the neutral line gives the same number, so the 0.68 is not specific to the aligned line.)~~ *(Corrected 2026-11-23.)* The design's competence gate uses the neutral line (design section 13, D7: "'competence' is the neutral-line AUC over all failures"; in `wf/textcraft/analysis.py` the reading passes `competence["neutral"]`, and the aligned-line AUC is computed only as a descriptive companion). Llama's two values coincide by chance: 0.6812 on the neutral line and 0.6813 on the aligned line (`textcraft_metrics.json`, `per_model.llama3.1-8b-fewshot.competence`). The round-2 design note on branch `textcraft-r2` describes the 0.68 as the aligned-line figure; the number is the same, but the gate's line is the neutral one.

**Scales.** L is the two-candidate log-odds of the success reply against the failure reply. The yes/no scale is the log-odds of "Yes" against "No" to "Will this action succeed?". A model "predicts success" when its log-odds is above zero, as in the families rule. A predicted probability is the logistic transform of L.

**Intervals.** Every interval is a 95% percentile interval from 2,000 resamples of whole goal pairs (ALFWorld: whole games). One set of resampling weights is shared by every TextCraft statistic, so paired differences between statistics are computed on the same resamples. There are no p-values. Many cuts were computed, so a single interval that excludes zero is weak evidence on its own. The readings below rest on patterns that recur across scales and cuts. Spearman intervals reuse the full-sample ranks within each resample, which is a slight approximation.

## 1. The competence profile

### Discrimination by class, role and history length

The AUC is the chance that a randomly chosen determined success scores above a randomly chosen failure: 0.5 is chance, 1 is perfect, and below 0.5 means the model ranks that kind of failure *above* real successes. In the table, each class is compared with the successes of the same role. "All classes" compares every failure of the role with the role's successes.

| Neutral line, AUC | Llama two-candidate | Llama yes/no | Qwen two-candidate | Qwen yes/no |
|---|---|---|---|---|
| All main failures vs all successes | 0.61 [0.58, 0.64] | 0.61 [0.58, 0.65] | 0.86 [0.84, 0.88] | 0.85 [0.82, 0.87] |
| Goal role, all classes | 0.84 [0.81, 0.87] | 0.85 [0.82, 0.88] | 0.90 [0.87, 0.92] | 0.87 [0.84, 0.90] |
| Goal × never obtained (252) | 0.90 [0.87, 0.93] | 0.91 [0.88, 0.94] | 0.97 [0.95, 0.99] | 0.95 [0.93, 0.97] |
| Goal × too few (252) | 0.78 [0.74, 0.83] | 0.80 [0.76, 0.84] | 0.82 [0.78, 0.86] | 0.80 [0.76, 0.84] |
| Ingredient role, all classes | 0.58 [0.54, 0.62] | 0.55 [0.51, 0.59] | 0.88 [0.86, 0.91] | 0.85 [0.82, 0.87] |
| Ingredient × never obtained (248) | 0.86 [0.82, 0.89] | 0.81 [0.76, 0.85] | 0.93 [0.90, 0.95] | 0.88 [0.85, 0.91] |
| Ingredient × too few (248) | 0.60 [0.53, 0.67] | 0.55 [0.47, 0.62] | 0.79 [0.74, 0.83] | 0.72 [0.67, 0.77] |
| Ingredient × used up (494) | 0.43 [0.38, 0.49] | 0.42 [0.38, 0.47] | 0.91 [0.88, 0.93] | 0.89 [0.86, 0.92] |
| Used up vs *all* successes (500) | 0.31 [0.26, 0.35] | 0.34 [0.30, 0.38] | 0.85 [0.82, 0.88] | 0.87 [0.84, 0.90] |

The used-up figure quoted in the confirmatory results (0.31) compares used-up failures, which are almost all ingredient crafts, with *all* successes, half of which are goal crafts. Llama scores goal crafts low whether they work or not (section 2). Against ingredient successes alone, Llama's used-up AUC is 0.43 [0.38, 0.49]. That is still below one half, but most of the distance to 0.31 is role composition.

By history length, Llama separates well only in the shortest histories: AUC 0.79 [0.73, 0.85] at 1 to 3 steps, against 0.54 [0.49, 0.60] at 4 to 8 and 0.55 [0.51, 0.59] at 9 to 12. Qwen's AUC falls from 0.92 to 0.85 and 0.83. The drop is largest for Llama because used-up failures, which need longer histories, appear only from 4 steps on. Within a role × class cell the length bins differ little for Llama. For example, goal × too few gives 0.80, 0.77 and 0.77, and ingredient × used up gives 0.37 at 4 to 8 steps and 0.40 at 9 to 12. Every cell, both scales and the first-token readout are in the metrics file (`textcraft.<model>.competence`). The first-token AUCs are within 0.03 of the two-candidate ones.

### How often each model predicts success

| Neutral line, share predicted success | Llama two-candidate | Llama yes/no | Qwen two-candidate | Qwen yes/no |
|---|---|---|---|---|
| All failures | 39% [35, 43] | 9.9% [7.8, 12.0] | 17% [14, 19] | 19% [16, 21] |
| Used-up failures | 82% [76, 87] | 17% [13, 21] | 10% [7, 13] | 7% [5, 9] |
| Too-few failures | 28% [21, 35] | 11% [7, 15] | 31% [26, 37] | 39% [33, 44] |
| Never-obtained failures | 8% [5, 11] | 1.4% [0.4, 2.8] | 8% [5, 11] | 11% [7, 14] |
| Goal-role failures | 2.5% [1.3, 3.9] | 0.8% [0.2, 1.6] | 11% [8, 14] | 17% [14, 22] |
| Ingredient failures | 58% [53, 63] | 15% [12, 18] | 19% [16, 23] | 19% [16, 23] |
| All successes | 46% [41, 51] | 11% [9, 14] | 72% [68, 76] | 72% [69, 77] |
| Goal-role successes | 24% [18, 31] | 6% [2, 10] | 66% [59, 73] | 73% [67, 79] |
| Ingredient successes | 68% [62, 73] | 17% [13, 21] | 77% [72, 82] | 72% [67, 77] |

At the threshold of zero, Llama's balanced accuracy, the mean of its accuracy on failures and on successes, is 0.53 on the two-candidate scale and 0.51 on yes/no. Qwen's is 0.78 and 0.77. On the two-candidate scale Llama calls 82% of used-up failures successes, more than the 68% of ingredient successes it calls successes. On yes/no it says "No" to nearly everything: 90% of failures and 89% of successes. Its mean neutral L is −2.05 [−2.40, −1.73] on failures and −0.33 [−0.64, +0.01] on successes (two-candidate), and −3.62 against −2.43 on yes/no. Qwen's are −9.8 against +7.3, and −12.4 against +6.3.

### Calibration

Calibration asks whether, among items given a predicted probability of success near p, a share near p actually succeed. The success share among the 2,100 items is 0.29. It is fixed by the design, not by the game, so calibration here is only a guide. The more telling property is whether the observed share *rises* with the predicted probability. The table groups ten equal-width bins into four.

| Predicted P(success), neutral two-candidate | Llama TextCraft: observed success share (items) | Qwen TextCraft | Llama ALFWorld | Qwen ALFWorld |
|---|---|---|---|---|
| below 0.1 | 0.19 (905) | 0.11 (1,322) | 0.01 (3,336) | 0.01 (4,673) |
| 0.1 to 0.5 | 0.46 (333) | 0.28 (102) | 0.02 (944) | 0.54 (13) |
| 0.5 to 0.9 | 0.31 (389) | 0.31 (87) | 0.06 (359) | 0.50 (6) |
| 0.9 and above | 0.32 (473) | 0.68 (589) | 0.65 (319) | 0.90 (266) |

Llama's TextCraft probabilities are informative only at the bottom: items it is sure will fail do fail more often (19% succeed, against 29% overall). Above 0.1, the observed share is flat or falls: 46%, then 31%, then 32% among items it gives more than 0.9. Its expected calibration error, the item-weighted mean gap between predicted and observed shares, is 0.33, and its Brier score 0.34. A model that always predicted the base rate 0.29 would score a Brier of 0.20. On yes/no the picture is the same (observed 0.25, 0.34, 0.31 and 0.50 in the four groups, the last from 8 items). Qwen's observed share rises from 0.11 to 0.68 across the groups (Brier 0.18). Its middle bins are flat too, but they hold few items. In ALFWorld, Llama's observed share rises monotonically, from 0.01 to 0.65 on the two-candidate scale and from 0.00 to 0.83 on yes/no (Brier 0.08 and 0.04 at a base rate of 0.06). It is overconfident in the middle bins but ordered. The ALFWorld base rate (0.06) differs from TextCraft's, so the error figures are not comparable across environments. The ordering is.

### The families competence rule

The model-families study fixed a competence rule before any of its scores were read (`notes/families_metrics.json`, `rules.competence`). It reads the yes/no scale under the neutral line and has four criteria. Applied to the confirmatory TextCraft records (1,500 main failures, 600 successes):

| Criterion | Threshold | Llama TextCraft | Qwen TextCraft | Llama ALFWorld | Qwen ALFWorld |
|---|---|---|---|---|---|
| Yes/no AUC | at least 0.70 | 0.61 [0.58, 0.65], **fails** | 0.85 [0.82, 0.87], passes | 0.97 [0.96, 0.98], passes | 0.99 [0.98, 0.99], passes |
| Yes on failures | at most 95% | 9.9%, passes | 18.7%, passes | 5.3%, passes | 0.15%, passes |
| Yes on successes | at least 5% | 11.3%, passes | 72.5%, passes | 85.3%, passes | 58.9%, passes |
| Median P(Yes) + P(No) | at least 0.5 | 0.999, passes | 1.000, passes | 0.997, passes | 1.000, passes |
| Verdict | all four | **not interpretable** | interpretable | interpretable | interpretable |

With the usefulness failures added, Llama's yes/no AUC is 0.67 [0.64, 0.70], still below 0.70. The ALFWorld values reproduce the families note's reference figures (0.97, 5.3%, 85%, 1.00 for Llama; 0.99, 0.15%, 59%, 1.00 for Qwen). The TextCraft design's own rule asked for a two-candidate AUC of at least 0.6 with the interval's lower end above 0.5. Llama passes it: 0.68 [0.65, 0.71] on the design's population, 0.61 [0.58, 0.64] on the main failures alone. So the verdict depends on which rule is applied. Llama passes the frozen TextCraft rule narrowly and fails the families rule clearly. Qwen passes both.

### Per-item accuracy at L = 0

Taking "L above zero" as Llama's prediction, it is right on 61% of main failures and 46% of successes (two-candidate), and on 90% and 11% on yes/no. Qwen is right on 84% and 72%, and on 81% and 73%. Llama's accuracy is high only on the goal role's failures (97.5% two-candidate) and on never-obtained failures (92%), and close to or below chance everywhere else.

## 2. What Llama does instead: the heuristics

The heuristics and fit measures are those fixed in the section above (committed in `9a83738` before any fit). For a binary heuristic h, the *tracking AUC* is the AUC of the model's neutral L between the items h calls a success and those it calls a failure. For the truth (H6) it equals the competence AUC.

| Heuristic | Validity: AUC against the truth | Llama tracking AUC | Llama minus truth (paired) | Llama kappa | Qwen tracking AUC | Qwen minus truth (paired) | Qwen kappa |
|---|---|---|---|---|---|---|---|
| H1 name seen | 0.71 | 0.87 [0.84, 0.89] | +0.26 [+0.22, +0.29] | 0.37 | 0.89 [0.87, 0.91] | +0.03 [+0.01, +0.06] | 0.26 |
| H2 gross obtained (ignores consumption) | 0.84 | 0.82 [0.79, 0.85] | +0.21 [+0.18, +0.24] | 0.45 | 0.79 [0.77, 0.82] | −0.07 [−0.09, −0.05] | 0.24 |
| H3 obtained in last 3 steps | 0.74 | 0.59 [0.56, 0.62] | −0.02 [−0.05, +0.01] | 0.06 | 0.73 [0.71, 0.76] | −0.13 | 0.33 |
| H4 mentioned in last 3 steps | 0.69 | 0.66 [0.63, 0.69] | +0.05 [+0.01, +0.08] | 0.19 | 0.71 [0.69, 0.74] | −0.15 | 0.20 |
| H5 history at least 4 steps | 0.45 | 0.59 [0.55, 0.62] | −0.03 [−0.08, +0.03] | 0.09 | 0.54 [0.50, 0.58] | −0.32 | −0.04 |
| H6 true accounting | 1.00 | 0.61 [0.58, 0.64] | (the truth) | 0.06 | 0.86 [0.84, 0.88] | (the truth) | 0.53 |
| H7 held at all (ignores counts) | 0.84 | 0.60 [0.56, 0.64] | −0.01 [−0.04, +0.03] | 0.06 | 0.83 [0.81, 0.86] | −0.03 | 0.48 |
| H8 last event is a gain | 0.81 | 0.56 [0.52, 0.59] | −0.06 [−0.09, −0.02] | −0.03 | 0.81 [0.79, 0.84] | −0.05 | 0.51 |
| H9 not a top item (goal crafts fail) | 0.42 | 0.80 [0.77, 0.83] | +0.19 [+0.13, +0.24] | 0.45 | 0.61 [0.57, 0.64] | −0.26 | 0.01 |
| H2 ∧ H9 | 0.59 | 0.85 [0.83, 0.88] | +0.24 [+0.21, +0.28] | **0.57** | 0.69 [0.66, 0.72] | −0.18 | 0.05 |
| H10 always fail | 0.50 | — | — | 0.00 | — | — | 0.00 |

The H3 settings k = 1 and k = 6 fit Llama no better (0.54 and 0.69). The yes/no scale gives the same ordering for Llama: H1 0.85, H2 ∧ H9 0.81, H2 0.79, H9 0.78, truth 0.61. Every figure is in `textcraft.<model>.heuristics`. In the design, H9's top items are exactly the goal-role crafts; the script checks this on every item.

**The best-fitting heuristic.** By the criterion fixed in advance, the highest tracking AUC over all 2,100 items, it is **H1, "every input's name appears in the history"** (0.87). Paired on the same resamples, H1 is not separable from the combination H2 ∧ H9 (difference +0.01 [−0.02, +0.05]). The combination fits better than H2 alone (+0.03 [+0.01, +0.05]). By decision agreement corrected for chance, which the note fixed as the comparable figure, H2 ∧ H9 fits best: kappa 0.57 [0.52, 0.63], with Llama's decision agreeing on 80% of items. The truth gets kappa 0.06. Taken together, Llama's predictions are described well by two rules: a craft succeeds if the history shows its inputs coming in, counted but never subtracted; and a craft of a finished goal fails.

**Where H2 and the truth disagree.** In this item set no too-few failure had any of its inputs consumed earlier: every too-few failure is one whose gross amount obtained is below the need. So H2 agrees with the truth on every success, every too-few and every never-obtained failure, and disagrees only on the 490 used-up failures without a never-obtained input, which it calls successes. Llama's preference for H2 over the truth (+0.21 in tracking AUC) is therefore entirely a statement about used-up failures. Llama scores them like successes. H1 differs from H2 in also calling 396 too-few failures successes, so its higher tracking AUC means Llama only partly reads the counts on too-few items (section 3).

**Does Llama use the truth at all, once the heuristic is known?** The joint regression fits

$$L_i = \beta_0 + \beta_{\text{truth}}\, y_i + \beta_h\, h_i + \varepsilon_i,$$

where $y_i$ is 1 for a real success and $h_i$ is 1 when the heuristic predicts success. For Llama with H1, $\beta_{\text{truth}}$ is −0.55 [−0.97, −0.11] and $\beta_h$ is +5.55 [+5.06, +6.05]. With H2, they are −2.18 [−2.61, −1.74] and +5.79 [+5.30, +6.28]. With H2 ∧ H9, they are +0.88 [+0.45, +1.32] and +4.83 [+4.44, +5.20]. Once the names or the gross amounts obtained are known, whether the inputs are still held adds nothing to Llama's score, or even lowers it. For Qwen with H1 the slopes are +12.6 for the truth and +11.0 for the heuristic, and with H2 they are +13.9 and +4.8. Qwen uses both the names and the true state, and the state carries most of the weight once counts enter.

**The names heuristic is not a Llama-only signature.** H1 tracks Qwen's scores as well (0.89, slightly above its truth AUC). H1 is a strong cue in this design: every never-obtained failure, a third of the failures, is one whose input name never appears. What separates the models is everything H1 leaves open. Among the items H1 calls a success (successes, used-up and most too-few failures), Qwen still separates truth from failure and Llama does not.

**Does a heuristic that ignores consumption explain the used-up AUC of 0.31?** The criterion fixed in advance has two parts: (a) the heuristic itself ranks used-up failures no lower than successes, and (b) Llama follows it more closely than the truth on used-up failures plus successes.

- H1 and H2 meet (a) with a tie, own AUC 0.51. They predict "success" for both used-up failures and real successes, so on their own they explain an AUC near 0.5, not below it.
- H1 and H2 meet (b) only on the letter. On that subset they vary only through 10 mixed items, used-up failures that also have a never-obtained input. Their tracking AUC there (0.92) rests on those 10.
- **H2 ∧ H9 meets both parts in substance.** By its own rule it ranks used-up failures *above* successes (own AUC 0.26), because it fails every goal craft, successes included. Llama follows it on that subset with tracking AUC 0.81 [0.77, 0.85], against 0.31 for the truth.

So "ignores consumption, and goal crafts fail" reproduces most of the 0.31. "Ignores consumption" alone accounts for a figure near 0.5; the goal-craft prior moves it to about 0.3.

A residual remains that no listed heuristic explains. Within the ingredient role, where H9 plays no part, Llama ranks used-up failures slightly above successes (AUC 0.43 [0.38, 0.49]).

**A post hoc check, added after the fits above had been read (not one of the heuristics fixed in advance).** 96.6% of used-up failures are exact repeats of a craft that succeeded earlier in the same history. Only 14% of successes are, and no too-few or never-obtained failure is. Llama does not use "repeat" as a cue: its mean L on ingredient successes is +0.93 [+0.58, +1.27] for repeats and +1.04 [+0.57, +1.52] for others. Comparing like with like, used-up failures against ingredient successes that are also repeats, Llama's AUC is 0.35 [0.29, 0.42]. Qwen's is 0.80 [0.74, 0.85]. Qwen *does* use the cue: it scores successful repeats far lower than other ingredient successes (+0.14 against +14.9). That is a heuristic of its own ("a repeated craft has probably run out"), on top of a state judgement that still separates within repeats. The residual for Llama stays unexplained.

**With the inventory shown.** On the 500 items of the inventory-shown prompt, where the inventory just before the action is printed in the history, Llama's competence rises from 0.54 to 0.62. It still tracks H1 (0.89) and H2 (0.81) far better than the truth (0.62). Its used-up AUC rises only from 0.24 to 0.35. Qwen's competence rises from 0.84 to 0.94, and its agreement with the truth (kappa 0.70) becomes the highest of any rule. Showing the state does not move Llama off the name and gross-amount cues. Even with the answer on screen, it mostly ignores what was used up.

## 3. Does Llama's score respond to the quantities that decide the outcome?

**Too few** (500 failures: every short input was obtained, but never in the needed amount at once). Llama's L falls with the total shortfall, the number of units missing (Spearman −0.48 [−0.57, −0.39]). Qwen's falls too (−0.55 [−0.62, −0.47]). By shortfall, Llama's mean L is +0.26 [−0.47, +0.92] when one unit is missing, −3.23 when two or three are, −4.27 at four to seven and −4.51 at eight or more. Qwen's runs from +1.92 to −16.4. Both models compare the amount obtained with the amount needed. Both are least sure when one unit is short. This is consistent with H2, which also counts units obtained.

**Used up** (500 failures: the input was once held in the needed amount and a later craft used it). Llama's L stays positive, a success prediction, however long ago the input fell below the needed amount: +1.61 when it fell at the very last step, +1.50 at one or two steps back, +2.15 at three to five and +1.65 at six or more (Spearman +0.13 [+0.03, +0.24], the wrong sign). Qwen's L is negative throughout and grows more negative with time since the consumption (−5.3 to −9.8; Spearman −0.23). The consuming craft is always visible in the history, because the only way an input is used is a history craft that names it. What changes Llama's score is an *inventory display* after the consumption. When an "Inventory:" reply shows the reduced count, Llama's mean L is +0.59 [−0.34, +1.58] (94 items), against +1.98 [+1.71, +2.24] when none does (406 items). Qwen moves the same way (−10.1 against −5.9). Llama's L is also lower when some of the input is left (+0.98) than when none is (+1.87). On the yes/no scale Llama's L on used-up failures rises with time since the consumption (Spearman +0.30). Llama does not compute consumption from the craft actions. When the history states the reduced amount, it reads it partly.

**Successes** (600: every input held). Llama's L *falls* as the surplus grows. The surplus is the smallest number of units held beyond what is needed. Llama's mean L is +0.55 when the inputs are exactly enough, −0.51 at one or two spare, −0.75 at three to seven and −1.46 at eight or more (Spearman −0.19 [−0.28, −0.11]). Qwen shows no relation (+0.03). A larger surplus means the inputs were fetched over more steps, and Llama's L on successes also falls with history length (−0.13). So the surplus pattern may be a length or recency effect rather than a reading of the amount. Either way, Llama is not reading spare inventory as making success more certain.

## 4. Competence and the goal effect

For each role × class cell, the table gives the neutral-line AUC (the cell's failures against the role's successes), the goal effect ΔL (aligned minus swapped) and the two shifts against the neutral line, on the two-candidate scale.

| Cell (failures) | Llama AUC | Llama ΔL | Llama aligned − neutral | Llama swapped − neutral | Qwen AUC | Qwen ΔL |
|---|---|---|---|---|---|---|
| Goal × never obtained (252) | 0.90 | −0.75 [−0.93, −0.57] | +0.17 | +0.92 | 0.97 | −0.45 [−0.89, −0.07] |
| Goal × too few (252) | 0.78 | −0.52 [−0.79, −0.26] | +0.16 | +0.68 | 0.82 | +2.52 [+1.60, +3.50] |
| Ingredient × never obtained (248) | 0.86 | −0.11 [−0.29, +0.06] | +1.18 | +1.30 | 0.93 | +1.64 [+1.05, +2.25] |
| Ingredient × too few (248) | 0.60 | +0.27 [−0.01, +0.52] | +1.32 | +1.05 | 0.79 | +2.28 [+1.25, +3.41] |
| Ingredient × used up (494) | 0.43 | −0.02 [−0.12, +0.07] | −0.83 | −0.81 | 0.91 | −0.20 [−0.56, +0.17] |

**Across strata.** Over the 14 role × class × length-bin strata with at least 20 failures, the Spearman correlation between Llama's stratum AUC and its stratum ΔL is −0.49 (two-candidate) and −0.76 (yes/no). Better discrimination goes with a *more negative* goal effect, because the goal role is both where Llama discriminates best and where the effect is negative. The correlation of AUC with "swapped minus neutral" is +0.45 and +0.56, and with "aligned minus neutral" +0.16 and −0.02. Where Llama discriminates, the swapped line lifts its scores more than the aligned line does. For Qwen the stratum correlation is −0.64 (two-candidate) and −0.30 (yes/no). Its effect is largest where it discriminates least (too-few, longer histories). That matches the cross-environment note's finding that the effect grows as the baseline becomes less certain. Fourteen strata are few, and the strata differ in more than competence; these correlations are descriptive.

**Item by item.** Within a role, whether Llama's neutral prediction is right does not predict its goal effect.

- *Goal role.* The 497 items Llama gets right give ΔL −0.64 [−0.80, −0.46]. Only 13 items are wrong. The correlation between the neutral margin (how far L sits on the correct side) and ΔL is +0.07 [−0.03, +0.17].
- *Ingredients, right and wrong alike.* The 416 ingredient failures Llama gets right give +0.07 [−0.12, +0.25], and the 574 it gets wrong give +0.00 [−0.12, +0.12].
- *Ingredients, against the neutral line.* Both goal lines pull Llama's ingredient scores toward a common level, by the same amount. On items it gets right, the aligned line lifts L by +1.33 over neutral and the swapped line by +1.26. On items it gets wrong, the two lines lower it by −0.59 and −0.60. The goal line acts on Llama's ingredient scores, but not in a goal-specific way.

**Where Llama is competent and where it is not.** The goal role is where Llama is competent (goal × never obtained, AUC 0.90). There its effect is negative and robust (−0.75), a change of confidence inside a failure judgement it already makes (section 1: it predicts success for 2.5% of goal-role failures). Ingredient × used up is where it is least competent (AUC 0.43). There the effect is null (−0.02), and both lines lower L by the same amount. The informative cell is **ingredient × never obtained**. Llama is competent there (AUC 0.86, comparable to the goal role) and the action is not a goal craft. If "no positive effect" were only a matter of competence, a positive effect should appear in this cell. It does not: −0.11 [−0.29, +0.06]. Qwen shows +1.64 [+1.05, +2.25] in the same cell. On successes, Llama's ΔL is −0.06 [−0.38, +0.27] on the goal role and +0.22 [+0.08, +0.36] on ingredients.

## 5. Is it specific to TextCraft?

| Neutral line | Llama ALFWorld | Llama TextCraft | Qwen ALFWorld | Qwen TextCraft |
|---|---|---|---|---|
| AUC, two-candidate | 0.87 [0.83, 0.91] | 0.61 [0.58, 0.64] | 0.93 [0.89, 0.96] | 0.86 [0.84, 0.88] |
| AUC, yes/no | 0.97 [0.96, 0.98] | 0.61 [0.58, 0.65] | 0.99 [0.98, 0.99] | 0.85 [0.82, 0.87] |
| Predicted success on failures / successes, two-candidate | 9.6% / 79% | 39% / 46% | 0.6% / 83% | 17% / 72% |
| Predicted success on failures / successes, yes/no | 5.3% / 85% | 9.9% / 11% | 0.15% / 59% | 19% / 72% |
| Balanced accuracy, two-candidate / yes/no | 0.85 / 0.90 | 0.53 / 0.51 | 0.91 / 0.79 | 0.78 / 0.77 |
| Observed share rises with predicted probability | yes, monotone | no (flat above 0.1) | yes | yes, at the ends |
| Families competence rule | passes | fails (AUC) | passes | passes |

ALFWorld items: 4,666 failures and 292 successes from 125 games. TextCraft items: 1,500 main failures and 600 successes from 247 goal pairs. The two success sets are built differently, so the AUCs are rough guides, as the cross-environment note says. The contrast is still large. In ALFWorld Llama is a competent, if under-confident, world model. Its yes/no answers are right for 95% of failures and 85% of successes, and its probabilities are ordered. In TextCraft it is close to chance outside the never-obtained and goal-role cells. Qwen loses some competence moving to TextCraft (yes/no AUC 0.99 to 0.85) but stays well inside the rule. What Llama lacks in TextCraft is inventory arithmetic over a history: counting what came in and subtracting what crafts used. ALFWorld's failures do not need that skill; they turn on where an object is and whether it was seen. The heuristics of section 2 are what such a model falls back on.

## 6. Implications

These points are written for the owner and for the paper. None of them changes a confirmatory verdict. The TextCraft result stays as reported: the goal effect is confirmed for Qwen3-8B and not for Llama-3.1-8B, so the replication holds for one model only.

**(a) The absent positive effect.** Incompetence is a sufficient reason not to read Llama's TextCraft ΔL as a measurement of a world model's goal sensitivity: there is little settled state judgement for a goal line to bias. It is not shown to be the *cause* of the missing positive effect. In ingredient × never obtained, the one non-goal cell where Llama discriminates as well as on the goal role (AUC 0.86), there is still no positive effect (−0.11), while Qwen shows +1.64. The alternative account from the diagnostics remains open. In ALFWorld, Llama's effect was almost entirely "mention", the task line naming the action's object, and TextCraft's goal line never names an ingredient. Incompetence and the absence of mention may both contribute. These data cannot apportion them.

**(b) The goal-role negative.** Incompetence does not explain it. The negative effect sits where Llama is competent (goal role, AUC 0.84 to 0.90). It is unchanged on the items Llama gets right (−0.64) and does not scale with how confident Llama is (margin correlation +0.07). Two parts of this note qualify it. First, part of Llama's goal-role "competence" is the prior that goal crafts fail (H9): Llama predicts success for only 24% of goal-role *successes*. So the goal role is where Llama's prior and the truth agree on failures, not a place where it tracks the inventory well. Second, the effect is a change of confidence inside a failure judgement (2.5% of goal-role failures are predicted successes under the neutral line). The diagnostics' reading stands: a terminal-craft penalty, mechanism open.

**(c) The replication verdict.** The verdict was read under the TextCraft design's frozen competence rule, which Llama passes (0.68 with the usefulness failures; 0.61 on the main failures, lower interval end 0.58). Under the families rule, fixed earlier for a different study and stricter, Llama's confirmatory run would be classed as not competent: yes/no AUC 0.61 (0.67 with the usefulness failures) against the required 0.70. This is a disclosed observation, not a re-verdict. Had the families rule been the frozen rule, the reading would have been "confirmed for Qwen3-8B; Llama-3.1-8B not interpretable". That is also not a replication across both models, so the headline does not change. What would change is the label on Llama's result: "not interpretable" rather than "negative".

**(d) "The effect requires competence".** In ALFWorld across 11 models (cross-environment note, section 9), every competent model shows a positive effect, and the only near-null effects come from models that cannot do the task. Llama in TextCraft fits that pattern overall: it is not competent by the families rule, and it shows no positive effect. It does not support the stronger reading that competence is *sufficient*. Within Llama, the competent strata show either a negative effect (goal role) or none (ingredient × never obtained), and across strata discrimination goes with a more negative ΔL. Within Qwen, the effect is largest where discrimination is weakest. The defensible statement is narrower: a model without a working state judgement shows no consistent positive effect, and among competent models the effect's size depends on more than competence (role, baseline certainty, what the goal line names).

**(e) Round 2.** Round 2 applies the families rule to every model, fixed in advance (`notes/textcraft_r2_design.md`, branch `textcraft-r2`, section 3). On this note's numbers Llama-3.1-8B will very likely fail it under the clean prompt as well: its verified-prompt yes/no AUC is 0.61, and the clean prompt lowered its two-candidate AUC by a further 0.016 on the diagnostic sample. The round-2 note already discloses this.

For Llama-3.2-3B there are no TextCraft scores. In ALFWorld it was the weakest model that passed (yes/no AUC 0.88, against 0.97 for Llama-3.1-8B). If TextCraft costs it as much as it cost Llama-3.1-8B, it will fail the AUC criterion by a wide margin. That is an expectation, not data.

If both Llama models fail, round 2's account will be tested on Qwen3-8B, Gemma-2-9B and Mistral-7B only. The account is the terminal-craft penalty plus the intermediate lift, and was built largely on Llama-3.1's diagnostics. Two descriptive additions, fixed before any round-2 score is read, would make a failed gate informative rather than silent: the heuristic fits of section 2 (at least H1, H2, H6 and H2 ∧ H9, with the tracking AUC, kappa and joint regression), and the used-up AUC against ingredient successes. Whether to add them is the owner's decision. They would show whether a failing model falls back on the same cues as Llama-3.1.

**(f) How the paper should describe the Llama TextCraft result.** A suggested wording, to be adapted: "Llama-3.1-8B-Instruct does not track the inventory in TextCraft. Under the neutral goal line it separates working from failing crafts barely above chance (AUC 0.61 on the main failures). Its predictions follow whether the inputs' names appear in the history and how many units were fetched, not what later crafts used up. It ranks used-up failures above real successes, and it treats any craft of a finished goal as likely to fail. Its goal effect was small and negative (−0.19 nats), confined to goal crafts, and a change of confidence within failure predictions it already made. It passed the TextCraft design's competence gate but would fail the stricter rule used for the model-families study. We therefore do not read it as evidence about how a goal line biases a working world model, in either direction. The replication holds for Qwen3-8B only." The paper should not call Llama's TextCraft result "robust to the goal" or "a reversed goal effect" without this qualification, and it should not present it as a failure of the ALFWorld finding to generalize: in ALFWorld the same model is competent.

## Reproducing

From the repository root, with the confirmatory scores copied from ICE (`text-wm-fix24/runs/textcraft_score`), the prompt-variant scores (`text-wm-fix27/runs/textcraft_dx_prompts`) and the run archive:

```
python scripts/textcraft_llama_competence.py --scores <dir with the four textcraft_score shards> \
    --prompts <textcraft_dx_prompts dir> --archive <text-wm-runs-archive> \
    --out notes/textcraft_llama_competence_metrics.json
```

It runs on CPU in about 15 minutes, needs numpy and scipy, and imports only the text-only checker (`wf/textcraft/checker.py`) from the project. It asserts that the checker's accounting reproduces every item's stored outcome, and records that the top items of the displayed list are exactly the goal-role crafts. The metrics file holds every number quoted here, and more: every role × class × bin cell on three scales, all heuristic fits on both scales, the paired differences, the quantity analyses on both scales, the competence-versus-effect strata on both scales, and the ALFWorld calibration bins.
