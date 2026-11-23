# Decision simulation, Stage 1a: what produces Qwen's harm, and how robust is it? (exploratory, post hoc, 2026-11-22)

**Status.** Everything in this note is exploratory and post hoc. It was written after the Stage 1a results ([decisionsim_results.md](decisionsim_results.md), [decisionsim_metrics.json](decisionsim_metrics.json)) had been read. The frozen gate verdict (tag `decisionsim-stage1a-frozen-2026-11-21` on `eeeb7dd`: "worth doing", decided by Qwen3-8B) is not changed by anything here. No model was run; the note re-reads saved yes/no scores. Many cuts were examined and nothing is corrected for multiple comparisons, so the conclusions rest on patterns that recur across cuts, readouts and models rather than on any single interval. Numbers are in [decisionsim_dx_metrics.json](decisionsim_dx_metrics.json); the code is `scripts/decisionsim_dx.py` (independent of `wf/decisionsim.py`) and `scripts/decisionsim_dx_frozen.py` (runs the frozen functions, unchanged, on the secondary models).

## Summary

**The setting, in one paragraph.** At 728 real ALFWorld states (the even steps of 134 Qwen3-8B agent episodes), a verifier asks a language model, for the agent's own proposed next action, "will this action succeed, or will the environment reply 'Nothing happens.'?". If the model's log-odds of Yes over No is at or below zero, the verifier rejects the proposal and takes the highest-scored action from a fixed candidate menu instead (rule 2). The menu was built by the item collector: three to five working actions (mostly random `go to` actions) and four to twenty failing ones, nearly all acting on the task's object (for example `take mug 1 from cabinet 6` before the agent has gone anywhere). The verifier sees the real task (aligned), no task (task-blind), the task with its object swapped for another, or "explore the room" (neutral). The frozen result is that a Qwen verifier that sees the real task makes the chosen action fail at 23.9% of states, against 4.4% for the task-blind verifier and 5.8% for no verifier at all.

**The short answer.** The number is real and reproducible, but what produces it is narrower and different from how the results note describes it, and most of it depends on the menu.

1. **It reproduces exactly and is not fragile in the usual ways.** An independent recompute gives the same failure rates and net count (aligned 23.9%, task-blind 4.4%, net 142 states, p = 0.001); the game-bootstrap interval of the difference is +16.9 to +22.3 points. Dropping any one game leaves +19.0 to +19.7, dropping any task type +18.2 to +20.5, every task type is positive, 110 of 134 games contribute harm (none more than 5 states), and 200 different tie seeds move the difference by at most half a point.

2. **The mechanism has two ingredients, and only one of them is about naming the task's object.**
   - *A blanket "No" under any concrete task at step 0.* At the first step, with the task sentence the last line before the action, Qwen answers No to nearly every candidate: it accepts 7% of the agent's working first moves under the real task and 6% under the swapped task, against 90% task-blind and 90% neutral. On working `go to` actions the score falls by about 39 nats (from about +19 to about −20). The swapped task produces the same fall (aligned minus swapped +0.9 nats), so this ingredient is not about the real object. It is also not a penalty on irrelevant receptacles: `go to` the task's destination receptacle is accepted 0% of the time at step 0.
   - *Object naming lifts the goal-object failures in the fallback menu.* Among actions that are all scored far below zero, the real task lifts the failing goal-object actions by about 5 nats over task-blind and 8 over swapped, enough to rank them above the working `go to` actions (within-state ranking accuracy of successes over failures in the fallback menu is 0.34 under the real task at step 0, against 1.00 task-blind and 0.89 swapped). So the fallback picks an action that Qwen itself predicts will fail (median score −10.9) over working ones it scores lower still.
   - *Rejection alone does no harm.* Recombining the halves shows this: the real task's accept/reject decision with the task-blind verifier's fallback ranking fails at 0.0% of step-0 states. All of the step-0 difference (+71 points on the swap states) and 18.5 of the 19.5 points overall arise in the fallback ranking. The aligned-minus-swapped part (+14.2 points) is entirely in the fallback ranking (0.0 from the accept/reject call).

3. **This is substantively the "through the fallback menu" case the gate meant to exclude, though by the frozen accounting it is not.** The gate counted harm by acceptance cell, and at step 0 the task-blind verifier accepts the proposal, so these states fall in "only the task-blind verifier accepts" and count as the verifier's judgement of the proposal. The results note's sentence "mostly through its judgement of the proposal" is therefore true of the cell accounting but not of the mechanism: the judgement is necessary, but what makes the choice fail, and what makes it attributable to the real task's object, is the ranking of a menu full of goal-object failures.

4. **It is not a threshold artefact; it is a level and ranking collapse.** The rejected working proposals sit about 39 nats below their task-blind scores (median), none has a task-blind score of 2 or less, and moving the acceptance threshold anywhere from −8 to +8 nats leaves the difference between +16.9 and +21.0 points. No single threshold lets the real-task Qwen verifier beat accepting everything (its in-sample best is to accept every proposal, 5.8%), whereas the task-blind one can reach 3.2%.

5. **Under natural variants of the rule or the menu, the effect shrinks to 3 to 6 points and is no longer attributable to the real object.** Applying the frozen gate's own conditions descriptively: a verifier that switches only to an alternative it itself accepts gives +4.1 points [2.2, 6.1], net 30, with aligned minus swapped +1.0 ("not attributable"); a fallback without goal-object actions gives +3.2 [1.6, 4.8], aligned minus swapped −0.1 ("not attributable"); a fallback restricted at step 0 to `go to` and `look` (all the agent ever proposes there) removes the step-0 harm entirely. Later steps alone give +7.6 [5.0, 10.5], also "not attributable" (aligned minus swapped +3.1). A per-arm threshold matched to the task-blind acceptance rate keeps "worth doing" (+9.8).

6. **Llama-3.1-8B shows no change because it lacks the first ingredient, not the second.** Its object naming is strong (at step 0 it says Yes to 54% of failing goal-object actions under the real task, 5% task-blind), but its scores are compressed (working proposals at about +3 nats) and the task moves working `go to` actions by under 1 nat, so it rejects only 10% of first moves and the fallback is rarely reached.

7. **The secondary models (now scored) reach "worth doing" too, by the same mechanism, with caveats.** Running the frozen functions unchanged: Gemma-2-9B +8.9 points (net 65), Llama-3.2-3B +15.4 (net 112), Mistral-7B +3.3 (net 24, three above the required 21). Gemma and Mistral show the same step-0 pattern as Qwen (any concrete task lowers `go to` by 13 and 8 nats; harm entirely in the fallback). *(Corrected 2026-11-23: for Mistral, 21 of the 24 harmed states are at step 0 and come through the fallback; the other 3 are at later steps, where the real-task verifier accepts a failing proposal the task-blind verifier rejects. See section 7.)* Llama-3.2-3B is different: all its scores lie within about half a nat of zero, so tiny shifts cross the threshold, and its task-blind verifier does not pay for itself (8.0% against 5.8% with no verifier), which the frozen rule flags. With a fallback free of goal-object actions, all three fall to "no reliable change" or "too rare".

8. **External validity is low for step 0 and moderate later.** A real ReAct agent of this kind never proposes a goal-object action at step 0 (RQ3: first fresh goal-object failure at a median step of 20). So in a closed loop with agent-generated alternatives, the step-0 component (about two thirds of the net) would turn into a verifier that overrides 93% of first moves with another working `go to`: arbitrary, but rarely a failure. What would remain is a later-step effect of a few points, driven by any concrete task rather than by naming the object.

**What this means.**

- *Keep:* on these fixed menus, a task-aware Qwen verifier with this yes/no question chooses failing actions far more often than a task-blind one; the result reproduces exactly and is not carried by a few games, a task type, ties or the threshold.
- *Change:* the mechanism. It is not "the verifier falsely rejects working proposals because they are irrelevant to the task". It is (a) a blanket No under any concrete task at step 0, most likely the yes/no question being read as "will this action accomplish the task?", and (b) object naming reordering a menu of actions the model itself predicts to fail. Naming the object acts only through (b), so the attribution depends on the menu containing failing goal-object actions.
- *For Option C:* the gate's licence ("a closed-loop study is worth running") stands as frozen, but the closed loop should not be expected to reproduce a 19-point effect, nor an effect attributable to naming the object, unless its alternatives include goal-object actions. Section 8 lists design changes.

*(Updated 2026-11-23, exploratory follow-up: [qwen_effect_audit_pipeline.md](qwen_effect_audit_pipeline.md) §7.2 C; details in [decisionsim_results.md](decisionsim_results.md), "Sensitivity of the step-0 harm to the prompt format".)* The blanket No at step 0 is partly a property of Qwen3's thinking-off format, which inserts an empty `<think></think>` block before the answer. Rescoring the 134 step-0 states in plain text (no chat template), the real-task verifier accepts 26% of the agent's first moves instead of 7.5%, against 98% task-blind, and the step-0 harm is 38.8 points [30.6, 47.0] instead of 71.6. The block roughly doubles the step-0 harm but does not create it, and the real-versus-swapped difference in chosen failures persists in plain text (39 against 4 points). Option C's verifier uses the block, so its step-0 component is format-sensitive. Nothing in this note's recompute changes.

## 1. Data, provenance and method

**Scores.** `decisionsim_score_<model>/shard_00_of_01/scoring.jsonl` in `text-wm-fix30` on ICE for all five models, copied on 2026-11-22: Qwen3-8B and Llama-3.1-8B (48,916 records each, five arms), Mistral-7B-v0.3, Gemma-2-9B-it and Llama-3.2-3B (38,986 each, four arms; all three shards reported complete). Every copy's SHA-256 matched the file on ICE; the hashes are in the metrics file under `meta.provenance`.

**States.** Rebuilt from `rq12_transitions_s0..s2` (archived copy of `text-wm-fix17`, SHA-256 identical to the files `text-wm-fix30` reads) with the note's own code: 761 eligible states, 728 at even steps, 134 games, 9,930 candidates, 42 failing proposals at even steps. Every model has every arm scored at every state (no incomplete states; the swapped arm all-or-none per state).

**Code.** `scripts/decisionsim_dx.py` shares nothing with `wf/decisionsim.py`. It re-implements what a recompute must reproduce (eligibility, rounding to 1/64 nat, rule 2, the seeded sha256 tie key) and has its own bootstrap and permutation. Intervals are 95% percentile intervals from 2,000 resamples of whole games (seed 20260924); for a subset of states the subset's own games are resampled. The frozen code's interval for the gate difference ([17.2, 22.3]) differs slightly from this note's ([16.9, 22.3]) because the resampling implementations differ; the point estimates, counts and permutation p are identical. *(Corrected 2026-11-23: the permutation p is identical for Qwen3-8B (0.001) but not for Llama-3.1-8B. The frozen analysis gives p = 0.160 (`decisionsim_metrics.json`, `option_c.per_model.llama3.1-8b.p`, 0.15984), which is the frozen value and the one quoted in `decisionsim_results.md`; this note's recompute gives 0.170 (`decisionsim_dx_metrics.json`, `models.llama3.1-8b.official_crosscheck.mine_p`, 0.16983). Both are game-level sign-flip tests with 1,000 random flips, so p is a count over 1,001: 160 against 170. The frozen code draws its flips with Python's `random` (`wf/analysis/stats.py`, `pair_permutation`) and this note's code with NumPy's generator (`scripts/decisionsim_dx.py`, `perm_p`), so the two use different random flips of the same data; at p near 0.165 the Monte Carlo standard error of a 1,000-flip p is about 0.012, so a difference of 0.010 is within that noise. The reading is the same: no reliable change.)* `scripts/decisionsim_dx_frozen.py` runs the frozen analysis functions, unchanged, on local copies; it writes nothing into `runs/`, and `runs/decisionsim_analysis` was not touched.

**Task types** are parsed from the task line: pick (24 games, 118 even-step states), pick two (17, 102), clean (31, 176), heat (23, 125), cool (21, 123), examine (18, 84).

**Notation.** Let $L$ be the rounded yes/no score of a candidate, the log-odds of Yes over No,

$$
L = \log P(\text{Yes}) - \log P(\text{No}),
$$

so the verifier accepts when $L > 0$. Let $F_c$ be the share of even-step states whose chosen action fails when the verifier sees condition $c$, and write the gate difference as

$$
\Delta F = F_{\text{aligned}} - F_{\text{blind}} .
$$

## 2. Question 1: independent recompute

The recomputed failure rates under rule 2 are below. They match the frozen analysis to every digit.

| Verifier | Qwen3-8B | Llama-3.1-8B | Mistral-7B | Gemma-2-9B | Llama-3.2-3B |
|---|---|---|---|---|---|
| None (proposal as is) | 5.8 | 5.8 | 5.8 | 5.8 | 5.8 |
| Task-blind | 4.4 | 4.4 | 5.2 | 4.7 | 8.0 |
| Task-blind, examples stripped | 4.1 | 4.8 | not scored | not scored | not scored |
| Neutral ("explore the room") | 3.4 | 4.1 | 5.2 | 3.8 | 6.9 |
| Swapped task (676 states) | 9.8 | 5.3 | 6.8 | 6.8 | 15.1 |
| Real task | 23.9 | 5.5 | 8.5 | 13.6 | 23.4 |

Percent of the 728 even-step states whose chosen action fails.

For Qwen the gate difference is +19.5 points [16.9, 22.3], with 147 states moved to failure and 5 moved to success (net 142), permutation p = 0.001. Against the stripped task-blind prompt it is +19.8 [17.2, 22.6]; against the neutral task +20.5 [17.8, 23.4]; against the swapped task +14.2 [11.7, 16.9]; and the swapped task against task-blind is +5.3 [3.1, 7.8].

**Acceptance cells (Qwen, against task-blind).** Both accept: 383 states, no harm. Only the real-task verifier accepts: 12 states, 3 helped. Only the task-blind verifier accepts: 202 states, 111 harmed. Both reject: 131 states, 36 harmed and 2 helped (net 34). These are the frozen numbers.

**Step 0 against later steps.** At the 134 step-0 states the real-task verifier's choice fails at 72.4% [64.9, 79.9] against 0.0% task-blind (net 97 states). At the 594 later states it is 13.0% against 5.4%, a difference of +7.6 [5.0, 10.5] (net 45). Against the swapped task the split is +63.2 points at step 0 and +3.1 [0.9, 5.5] later: naming the real object adds little after step 0.

**Leave one out.** Dropping any one game leaves a difference between +19.0 and +19.7 points. Dropping any one task type leaves +18.2 (without heat) to +20.5 (without clean). Harm is spread: 110 of 134 games contribute at least one harmed state, no game more than 5, and the 10 heaviest games carry 21% of the harm.

**By task type** (Qwen, real task against task-blind, game intervals):

| Task type | Games | States | Real task | Task-blind | Difference | Step 0 | Later |
|---|---|---|---|---|---|---|---|
| pick | 24 | 118 | 26.3 | 4.2 | +22.0 [17.5, 27.4] | +75.0 | +8.5 [3.3, 14.8] |
| pick two | 17 | 102 | 18.6 | 4.9 | +13.7 [8.8, 19.6] | +64.7 | +3.5 [0.0, 9.4] |
| clean | 31 | 176 | 22.7 | 6.2 | +16.5 [13.0, 20.6] | +83.9 | +2.1 [−1.4, 6.3] |
| heat | 23 | 125 | 29.6 | 4.0 | +25.6 [19.1, 33.3] | +87.0 | +11.8 [3.9, 21.2] |
| cool | 21 | 123 | 29.3 | 4.9 | +24.4 [18.0, 32.5] | +76.2 | +13.7 [6.7, 22.9] |
| examine | 18 | 84 | 13.1 | 0.0 | +13.1 [5.3, 21.6] | +33.3 | +7.6 [0.0, 16.2] |

Every task type is positive with an interval above zero. Examine tasks have the smallest step-0 effect, plausibly because their goal-object actions in the menu are fewer and their goal objects (books, alarm clocks) sit on the same desks the agent goes to first. The step-0 component is large everywhere; the later component is uneven (heat and cool largest, clean near zero).

**Verdict on question 1.** The gate quantities reproduce exactly, and the result is not driven by a few games, a task type or the clustering.

## 3. Question 2: the step-0 mechanism

**What the agent proposes at step 0.** 120 `go to` actions and 14 `look`. All 134 work.

**What each verifier says** (Qwen, step 0; acceptance of the 134 working proposals and the median score):

| Verifier | Working proposals accepted | Median L, working proposal | Median L, working generated `go to` | Median L, failing goal-object action | States where no candidate gets Yes |
|---|---|---|---|---|---|
| Task-blind | 90% | +19.3 | +19.0 | −24.8 | 0% |
| Task-blind, examples stripped | 90% | +20.0 | +19.8 | −25.3 | 0% |
| Neutral | 90% | +18.4 | +18.3 | −25.8 | 0% |
| Swapped task (125 states) | 6% | −20.8 | −24.0 | −27.8 | 94% |
| Real task | 7% | −18.8 | −23.8 | −21.4 | 90% |

The 10% of proposals the task-blind verifier rejects are the 14 `look` actions, which every verifier scores below zero (Qwen seems not to expect `look` to produce an observation). Under the real and the swapped task, Qwen answers No to essentially everything at step 0, including every working `go to`.

**Is it a task-relevance penalty?** Each working `go to` was classed by its target receptacle: the task's destination type (for example a cabinet for "put some saltshaker on cabinet"), the task's tool (sinkbasin for clean, microwave for heat, fridge for cool, the receptacle holding the lamp for examine), a receptacle that holds an instance of the goal object in the hidden scene, or none of these (irrelevant). At step 0 under the real task:

| `go to` target | n | Yes, real task | Median L, real task | Yes, task-blind | Shift, real − blind (nats) | Shift, real − swapped (nats) |
|---|---|---|---|---|---|---|
| Destination | 55 | 0% | −19.8 | 98% | −39.1 | +0.4 |
| Tool | 58 | 7% | −18.8 | 100% | −31.7 | +2.4 |
| Holds the goal object | 61 | 5% | −21.3 | 100% | −38.8 | +0.5 |
| Irrelevant | 362 | 1% | −23.8 | 100% | −40.3 | +0.8 |

Relevance matters a little: within a state, relevant `go to` actions score 4.5 nats [3.1, 6.0] higher than irrelevant ones under the real task, against −1.2 [−2.0, −0.3] task-blind. But every class is pushed about 32 to 40 nats below its task-blind score, and the swapped task pushes it just as far. So the rejection is not a penalty on irrelevant actions and not an object-naming effect; it is a reaction to any concrete task.

**Is it the empty history?** Not the emptiness itself: the task-blind and neutral prompts have the same empty action history and accept 90%. What is special about step 0 is that the task sentence is the last line of the history, immediately before "Action:". The real-task acceptance of working proposals rises steeply once actions separate the task line from the query: 7% at step 0, 60% at step 2, 65% at step 4, 80% at step 6, 66 to 69% at steps 8 and 10 (task-blind 80 to 86% throughout). The swapped task follows the same curve (6%, 49%, 60%, 71%, 69%, 64%).

**The most likely reading** is that Qwen answers the yes/no question as "will this action accomplish the task?" when the task is salient. Four observations fit: any concrete task produces it while "explore the room" does not (and `go to` arguably does accomplish exploring); it fades as the task line recedes; relevant receptacles get somewhat higher scores; and the TextCraft audit found the same reading of an unanchored yes/no question for Qwen (`notes/textcraft_qwen_robustness.md` on branch `textcraft-qwen-robust`, section 9). ALFWorld's question names the "Nothing happens." reply, so it is better anchored than TextCraft's, but at step 0 the anchor evidently does not hold. This is an interpretation; the scores cannot prove it. A direct test would rescore step 0 with the task sentence moved before the room description, or with the observation-prediction readout instead of yes/no.

**What the real-task fallback picks.** At the 110 step-0 states where only the task-blind verifier accepts the working proposal, the real-task fallback chooses `take` 50 times, `heat`/`cool`/`clean` 36 times and `go to` 24 times. The 86 failing picks are all goal-object actions, and all fail for the same reason: the agent has not moved yet, so it is away from every receptacle (`away_from_receptacle_never_moved`). Qwen scores the failing pick at a median of −10.9 nats, that is, it predicts the failure itself; the pick wins because the working alternatives score lower still (median gap 8.6 nats; only 12% of gaps are within 2 nats).

**How much is rejection and how much is fallback?** Rule 2 can be run with the accept/reject decision from one verifier and the fallback ranking from another. On the 676 even-step states with a swap (Qwen):

| Accept/reject from | Fallback ranking from | All states | Step 0 (125) | Later (551) |
|---|---|---|---|---|
| Task-blind | Task-blind | 4.4 | 0.0 | 5.4 |
| Real task | Task-blind | 5.5 | 0.0 | 6.7 |
| Task-blind | Real task | 9.2 | 6.4 | 9.8 |
| Swapped task | Swapped task | 9.8 | 8.0 | 10.2 |
| Real task | Swapped task | 9.8 | 8.0 | 10.2 |
| Swapped task | Real task | 24.3 | 73.6 | 13.1 |
| Real task | Real task | 24.0 | 71.2 | 13.2 |

Failure rates in percent. Splitting the real-minus-task-blind difference in this order, the accept/reject part is +1.0 points [0.1, 2.1] and the fallback part +18.5 [16.2, 21.0]; at step 0 they are 0.0 and +71.2. Splitting the real-minus-swapped difference the same way, the accept/reject part is 0.0 [−1.2, 1.2] and the fallback part +14.2 [11.9, 16.6]. Both halves are needed for harm, but they play different roles: the any-task rejection opens the door, and naming the object decides what walks through it.

## 4. Question 3: rule design and menu

Each variant is computed for the real-task and the task-blind verifier alike; the gate category is the frozen gate's conditions applied descriptively to that variant (interval above zero and p below 0.05; net at least half of the 42-state headroom; the both-reject cell at most half of the net; aligned minus swapped at least half of aligned minus task-blind on the swap states). None of this changes the frozen verdict.

| Variant (Qwen) | Real task | Task-blind | Difference | Net | Real − swapped | Category |
|---|---|---|---|---|---|---|
| As frozen | 23.9 | 4.4 | +19.5 [16.9, 22.3] | 142 | +14.2 | worth doing |
| Switch only to an alternative the verifier accepts | 8.5 | 4.4 | +4.1 [2.2, 6.1] | 30 | +1.0 | not attributable |
| Fallback without goal-object actions | 6.6 | 3.4 | +3.2 [1.6, 4.8] | 23 | −0.1 | not attributable |
| Fallback `go to`/`look` only at step 0 | 10.6 | 4.4 | +6.2 [4.0, 8.5] | 45 | +2.5 | not attributable |
| Fallback of the proposal's own action type | 5.4 | 4.8 | +0.5 [−0.3, 1.4] | 4 | −0.9 | no reliable change |
| Fallback to admissible actions only (= the oracle fallback, since every admissible candidate here succeeds) | 1.5 | 1.5 | 0.0 | 0 | 0.0 | no reliable change |
| Later steps only (594 states) | 13.0 | 5.4 | +7.6 [5.0, 10.5] | 45 | +3.1 | not attributable |
| Excluding `look`/`inventory` proposals (694) | 22.9 | 4.6 | +18.3 [15.7, 21.2] | 127 | | |
| Per-arm threshold matched to the task-blind acceptance rate | 14.1 | 4.4 | +9.8 [7.6, 12.0] | 71 | +7.8 | worth doing |

**Threshold.** With a shared threshold $L > k$ for $k$ from −8 to +8 nats the difference stays between +16.9 and +21.0 points, and the step-0 failure rate between 66% and 77%. Qwen's real-task scores of working first moves are not near zero (median −18.8), so no small change of the cutoff matters. The best in-sample threshold for each verifier (an optimistic bound) gives 5.8% for the real task (accept everything, which is no verifier), 3.2% task-blind, 2.7% neutral and 5.8% swapped. A task-aware Qwen verifier cannot be tuned to beat executing the proposal on these menus.

**Discrimination of the proposal's own outcome.** The area under the ROC curve for separating the 686 working from the 42 failing proposals is 0.74 [0.66, 0.80] under the real task, 0.84 [0.76, 0.91] task-blind, 0.88 neutral and 0.71 swapped. Restricted to later steps, where every failing proposal is, it is 0.81 against 0.83: most of the loss is the step-0 collapse. The ranking of the fallback menu (mean per-state probability that a working generated candidate outranks a failing one) is 0.34 at step 0 under the real task, against 1.00 task-blind and 0.89 swapped; later it is 0.90 against 0.97.

**Rules 1 and 3, lure rate, ties.** Rule 1 on the full menu: +15.9 points [13.6, 18.4] (58.2 at step 0); on the generated-only menu +26.8 [23.6, 30.0]. Rule 3 (the best goal-relevant candidate, 87 states where one works) goes the other way, −4.6 [−10.8, 1.2]: among goal-object actions, the task helps a little. Lure rate (proposal against one failing goal-object action at a time, 5,475 pairs): 7.7% [6.1, 9.4] real task against 1.3% task-blind; 35% of step-0 pairs and 3% of later pairs. Ties: splitting each tied top set evenly gives +19.4; 200 other tie seeds give +19.2 to +19.6 (10th to 90th percentile); only 5 real-task fallbacks have a tied top.

**Verdict on question 3.** The size and the frozen category hold for rule 2 as frozen and for any threshold. They do not survive the two most natural changes a builder would make, a verifier that only switches to an action it believes will work, or a menu like the agent's own: the difference falls to 3 to 6 points, still above zero, and the part attributable to the real object vanishes.

## 5. Question 4: calibration

Qwen's yes/no scores are extreme: task-blind, the median working proposal is at +16.8 nats later and +19.3 at step 0, and the median failing proposal at −16.4. The harm is not small shifts crossing a boundary. Of the 110 step-0 proposals that only the task-blind verifier accepts, none has a task-blind score at or below 2 nats; the median shift from task-blind to real task is −38.9 nats and 95% of the shifts exceed 10 nats. At later steps the 92 such proposals have a median shift of −25.5 nats (6.5% with a task-blind score of 2 or less).

At later steps the whole distribution moves: the median shift of a working proposal is −4.0 nats (quartiles −9.5 to −1.5), 24% move by more than 10 nats, and the real-task scores spread out (20% of working proposals below −10, against 13% task-blind).

For the other models the picture differs in scale. Llama-3.1-8B's working proposals sit at about +3 nats under every verifier and its failing proposals at about +0.8 (it barely separates them, AUC 0.78 under both). Its 14 step-0 flips are near-boundary (median real-task score −0.4). Llama-3.2-3B's scores all lie within about ±0.6 nats of zero (working proposals median +0.25, failing 0.0), so every one of its flips is a sub-nat shift across the threshold. Gemma (working proposals about +7 nats) and Mistral (about +8) sit between Qwen and the Llamas.

## 6. Question 5: link to the item-level effects

Per-candidate mean shifts (nats, Qwen, game intervals in the metrics file):

| Candidates | Step | n | Real − blind | Real − swapped | Swapped − blind | Real − neutral |
|---|---|---|---|---|---|---|
| Working `go to` | 0 | 522 | −39.1 | +0.9 | −40.0 | −39.0 |
| Failing goal-object `take` | 0 | 402 | +4.8 | +8.2 | −3.3 | +5.0 |
| Working `go to` | later | 1,929 | −18.3 | −1.4 | −17.3 | −18.1 |
| Working `take` of another object | later | 98 | −15.0 | −0.1 | −15.1 | −13.1 |
| Failing goal-object `take` | later | 2,812 | +1.9 | +1.9 | −0.1 | +1.5 |

The two item-level findings map onto the two ingredients:

- *The "relevance penalty"* (exploratory_results.md section 7: Qwen's real task lowers unrelated `go to` by 2.99 and unrelated `take` by 13.3 nats against the neutral line, but only by 0.76 and 0.43 against the swapped task). Here the same signature appears, larger for `go to`: about −18 nats against neutral later and −39 at step 0, and about −1 to +1 against swapped. The unrelated `take` figure matches closely (−13.1 here, −13.3 there). Because the swapped task produces it equally, and because relevant receptacles are hit nearly as hard, "relevance penalty" is a misleading name: it is a penalty that any concrete task puts on actions that do not complete it. The section 7 groups were 24 and 94 determined successes; the much larger `go to` shift here comes from step 0 and from random admissible `go to` actions, which section 7 did not cover.
- *The object-naming effect* (RQ2; on yes/no, Qwen's step-0 effect on determined failures was 7.71 nats and the later one 2.20 against the swapped task). Here it is +8.2 at step 0 and +1.9 later on failing goal-object `take` actions: the same size.

So Stage 1a's harm does decompose into the two known item-level effects, but not additively. The any-task penalty creates the opportunity (a rejected working proposal and a menu in which nothing is accepted); the object-naming effect decides the fallback. Neither alone produces much: the any-task penalty with a ranking free of naming (the swapped verifier) fails at 9.8%, and naming with task-blind acceptance fails at 9.2%.

## 7. Question 6: Llama-3.1-8B and the secondary models

**Llama-3.1-8B.** Its object naming is as strong as Qwen's in Yes-rate terms (failing goal-object actions at step 0 accepted 54% under the real task, 5% task-blind; +5.3 nats on goal-object `take`), but the any-task penalty on working actions is under 1 nat (−0.8 on `go to` at step 0, −0.3 later). It therefore accepts 90% of working first moves under the real task, and the fallback is rarely invoked. Where it is, the real-task fallback is worse (the fallback half of the difference is +1.3 points [0.3, 2.5]); and rule 1 on the full menu, which always uses the ranking, does show a reliable change (+4.4 [1.1, 7.6]). Llama's null under rule 2 is thus a property of its accept/reject call, not an absence of task influence. Its verifier is also nearly uninformative on these proposals (AUC 0.78), so it behaves much like no verifier.

**Secondary models** (frozen functions, unchanged, on the now complete scores; they never decide the gate):

| Model | Real − blind | Net | Both-reject net | Real − swapped | Task-blind pays for itself | Frozen category |
|---|---|---|---|---|---|---|
| Mistral-7B | +3.3 [2.1, 4.6] | 24 | 0 | +2.1 | yes (5.2 vs 5.8) | worth doing |
| Gemma-2-9B | +8.9 [6.4, 11.6] | 65 | 2 | +7.0 | yes (4.7 vs 5.8) | worth doing |
| Llama-3.2-3B | +15.4 [12.3, 18.4] | 112 | 25 | +8.4 | no (8.0 vs 5.8), flagged | worth doing |

Intervals here are the frozen code's. The same verdict holds against the neutral task for all three.

- *Mistral and Gemma* repeat Qwen's mechanism at smaller scale. At step 0 any concrete task lowers working `go to` by 8 (Mistral) and 13 nats (Gemma), with real minus swapped near zero, and naming lifts goal-object failures by 6.5 and 5.4 nats over swapped. All of the difference is in the fallback ranking. Mistral's harm is all at step 0 (21 states, every one a goal-object fallback), and its net of 24 clears the threshold of 21 by three states. *(Corrected 2026-11-23: Mistral-7B's harm was described as all at step 0. By `decisionsim_dx_metrics.json` (`models.mistral-7b-v0.3.q1_recompute`), its 24 net states are 24 harmed and 0 helped; 21 are at step 0, where the real-task verifier rejects a working first move and the fallback picks a failing action, and 3 are at later steps, where the real-task verifier accepts a failing proposal that the task-blind verifier rejects. The category and the +3.3 points are unchanged.)*
- *Llama-3.2-3B* is different: half of its difference is in the accept/reject call (+5.6 points), it persists at later steps (+12.8), and every flip is a sub-nat move across zero. Its task-blind verifier already does worse than no verifier, so by the frozen rule this "worth doing" is "harm inside a verifier that does not pay for itself".
- *With a fallback free of goal-object actions* Mistral (+0.4) and Gemma (+0.1) show no reliable change and Llama-3.2-3B falls to "too rare" (+2.7, net 20). With a verifier that only switches to accepted alternatives, Mistral, Gemma and Llama-3.2-3B stay "worth doing" (+3.3, +5.9, +12.4), unlike Qwen, because their failing goal-object fallbacks often score above zero.

## 8. Question 7: external validity and implications

**How often would a real agent meet this?** The step-0 component needs a menu that contains goal-object actions at step 0. The RQ3 agent (exploratory_results.md section 7) never proposes one there; its first fresh goal-object failure comes at a median step of 20, in 20 of 134 episodes, and in this collection every step-0 proposal is a `go to` or `look`. With a fallback limited to those action types at step 0, the step-0 harm is zero for every model. The real-task Qwen verifier would still override the agent's first move in 93 of 100 episodes, replacing a working `go to` with another working `go to` (or, at most states, with whatever scores least badly). That costs no step, but it means the verifier's step-0 judgement is noise. After step 0 the effect is a few points (+7.6 on the collector's menus, +3.9 without goal-object fallbacks) and is driven by stating any concrete task rather than naming the real object.

**Implications for Stage 1b (agent-generated candidates).** Stage 1b is the right next test, and this note predicts its outcome for Qwen: a small later-step effect, mostly not attributable to the object, unless the agent's own alternatives include goal-object actions. It should report, as Stage 1a did not:

- the accept/reject and fallback halves (the recombination of section 3), since the cell accounting can label a fallback effect as a judgement effect;
- the gate under a rule that switches only to an alternative the verifier accepts, alongside rule 2 as frozen;
- the real-minus-swapped split by step;
- Yes-rates at step 0 of working actions under each verifier, as a direct check of the any-task collapse.

**Implications for Option C (closed loop).** The frozen gate licenses running it. The closed-loop design should expect a much smaller effect than 19.5 points and should be able to tell the two ingredients apart:

- *Include a swapped-task verifier arm.* Otherwise the any-task collapse, which dominates the accept/reject call, will be read as an object-naming effect.
- *Pre-register the switching rule.* A verifier that switches to an action it itself predicts to fail is not a design a builder would ship. Either make "switch only to an accepted alternative, else keep the proposal" the primary rule or pre-register it as co-primary.
- *Use agent-generated alternatives,* not the collector's menus.
- *Log step-0 overrides and episode-level loops.* A verifier that rejects nearly everything may cycle between rejected proposals.
- *Consider an anchored readout.* The step-0 collapse looks like a task-success reading of the yes/no question. Either test it before Option C (rescore step 0 with the task sentence moved, or use the observation-prediction readout, which is anchored to the reply), or treat yes/no as one of two readouts.

**For the paper.** The defensible decision-level claim is: on menus rich in goal-object failures, a task-aware yes/no verifier built on Qwen3-8B (and, less strongly, Gemma-2-9B and Mistral-7B) chooses failing actions far more often than a task-blind one. Two effects combine. Stating any concrete task makes the model reject almost every working first move. Naming the task's object then ranks failing goal-object actions above working ones. Claims that the harm comes from "rejecting task-irrelevant actions", or that naming the object hurts verification "through its judgement of the proposal", are not supported.

## 9. What should change in decisionsim_results.md

These are suggestions; the frozen verdict and its table stay as they are.

1. The bullet "By false rejection of working proposals" and the licence sentence "naming the task's object accounts for well over half of the increase, mostly through its judgement of the proposal" should be qualified. False rejection is necessary, but with the task-blind fallback it causes no failures. Both the failures and the aligned-minus-swapped attribution arise in the fallback ranking; the frozen fallback safeguard (the both-reject cell) does not capture this, because at step 0 the task-blind verifier accepts.
2. The sentence linking to the item-level findings ("lowers its predicted success for working actions that do not act on the task's object") should say that the step-0 rejection is caused by any concrete task: the swapped task does it equally, the neutral task does not, and relevant receptacles are rejected too.
3. The step-0 bullet should add that Qwen answers No to nearly every candidate at step 0 under a concrete task (90% of states have no candidate with Yes), so the fallback picks actions the model itself predicts to fail.
4. The secondary models should be added: under the frozen functions Mistral-7B, Gemma-2-9B and Llama-3.2-3B all reach "worth doing" (Llama-3.2-3B flagged: its task-blind verifier does not pay for itself; Mistral clears the net threshold by three states). They never decide the gate.
5. "Model dependence" should say why Llama-3.1-8B is barely moved: its object naming is strong, but the task hardly lowers working actions, so its verifier rarely rejects and the fallback is rarely used.
6. "What it does not license" could add that under a verifier that switches only to an alternative it accepts, or a fallback without goal-object actions, Qwen's difference is 3 to 4 points and not attributable to the real object (post hoc, section 4).
