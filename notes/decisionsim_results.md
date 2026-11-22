# Decision simulation, Stage 1a (fixed candidate menus): results for the primary models (2026-11-22)

Design frozen before scoring at tag `decisionsim-stage1a-frozen-2026-11-21` (commit `eeeb7dd`; [decisionsim_design.md](decisionsim_design.md), gate at line 143). Scoring: Qwen3-8B (ICE job 7230581) and Llama-3.1-8B (job 7230582) at `eeeb7dd`. Analysis: `decisionsim_analysis` (job 7239661). The three secondary models (Mistral-7B, Gemma-2-9B, Llama-3.2-3B) are being scored and will be added; the gate needs only the primary models. Numbers are from [decisionsim_metrics.json](decisionsim_metrics.json).

## Question

At 728 real ALFWorld states (even steps; 134 games), a verifier asks the world model, for the agent's own proposed action, "will this action succeed, or will the environment reply 'Nothing happens.'?". If the answer is No, it picks the best-scored alternative from a fixed candidate menu instead (rule 2). Does showing the verifier the agent's real task make the chosen action fail more often than a task-blind verifier (the same prompt with the task sentence deleted)?

## Pre-registered gate: "worth doing"

The verdict is the same under the deciding comparison (task-blind) and both sensitivity comparisons (neutral task; task-blind with the examples' task lines also removed). It is decided by Qwen3-8B.

| | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Category | worth doing | no reliable change in failures |
| Extra failure rate, aligned − task-blind | +19.5 points [17.2, 22.3], p = 0.001 | +1.1 points [−0.1, 2.6], p = 0.16 |
| Net states moved to failure (required ≥ 21) | 142 (harm 147, help 5) | 8 |
| Of which through the fallback menu (cap: half) | 34 | 6 |
| Aligned − swapped (attribution: ≥ half of aligned − blind) | +14.2 points | +0.4 points |

## How often the chosen action fails, by verifier (rule 2)

| Verifier | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| None (execute the agent's proposal) | 5.8% | 5.8% |
| Task-blind | 4.4% [3.0, 5.9] | 4.4% [2.9, 6.1] |
| Neutral task ("explore the room") | 3.4% [2.2, 4.8] | 4.1% [2.6, 5.6] |
| Swapped task | 9.8% [7.4, 12.3] | 5.3% [3.6, 7.1] |
| **Real task** | **23.9% [21.1, 27.3]** | 5.5% [3.8, 7.5] |

A task-blind Qwen verifier improves on executing the agent's own proposal. A task-aware Qwen verifier makes the chosen action fail about four times as often as no verifier at all.

## Where Qwen's harm arises

*(Revised 2026-11-22 from the exploratory analysis in `decisionsim_dx.md`, branch `decisionsim-dx`. That analysis recomputes every number above independently. The frozen verdict is unchanged.)*

**Robustness.** Dropping any one game leaves +19.0 to +19.7 points; dropping any task type leaves +18.2 to +20.5. All six task types are positive. 110 of 134 games contribute harm, none more than 5 states. Any yes/no cutoff from −8 to +8 nats gives +16.9 to +21.0. The flips are large shifts, not small ones crossing zero: the median shift on a rejected first move is −38.9 nats.

**Two parts, and harm needs both.**

- *Any concrete task makes Qwen's verifier reject working first moves.* At step 0 the agent proposes `go to` or `look`, and every one of those works. Under the real task Qwen accepts 7% of them, under the swapped task 6%, against 90% task-blind and under the neutral line. The swapped task does this as much as the real one, and "explore the room" does not, so it is not a relevance judgement: Qwen rejects `go to` the task's own destination too. At 90% of step-0 states, no candidate gets a Yes. A likely reading, not yet tested directly, is the one found in TextCraft: the yes/no question is answered as "will this action accomplish the task?".
- *Naming the object reorders the fallback.* Among candidates that Qwen scores far below zero, the real task lifts failing actions on the task's object by 4.8 nats over task-blind and 8.2 over swapped. So the fallback picks a goal-object action that Qwen itself predicts will fail (median −10.9 nats). At the 110 step-0 states where only the task-blind verifier accepts the proposal, the real-task fallback fails 86 times. All 86 are goal-object actions taken before the agent has moved.
- *Rejection alone does no harm.* Combining the real-task accept/reject with the task-blind fallback ranking fails at 0.0% of step-0 states. Overall, the accept/reject half contributes +1.0 point and the fallback-ranking half +18.5. The real-minus-swapped attribution (+14.2) lies entirely in the fallback ranking.
- *The fallback cap does not catch this.* The frozen cap on harm through the fallback menu counts only states where both verifiers reject the proposal. At step 0 the task-blind verifier accepts, so this route is not capped. The licence's phrase "mostly through its judgement of the proposal" overstates the role of the accept/reject call.

**Under other rules or menus (post hoc, descriptive).**

| Variant | Real task − task-blind |
|---|---|
| Switch only to an alternative the verifier itself accepts | +4.1 points [2.2, 6.1]; real − swapped +1.0; not attributable to naming |
| Fallback without goal-object actions | +3.2 [1.6, 4.8]; not attributable |
| Later steps only | +7.6 [5.0, 10.5] |
| Step-0 fallback limited to `go to` / `look` | step-0 harm drops to zero; overall +6.2 |

The direction holds under every variant. The size (3–6 points under natural changes) and the attribution to naming the object depend on a fallback menu full of failing goal-object actions, which a real agent would not produce at step 0.

**Other models** (frozen functions, applied post hoc; they never decide the gate):

| Model | Real − task-blind | Net states | Note |
|---|---|---|---|
| Mistral-7B | +3.3 points | 24 (21 needed) | all at step 0 |
| Gemma-2-9B | +8.9 | 65 | same two-part mechanism as Qwen |
| Llama-3.2-3B | +15.4 | 112 | its task-blind verifier fails 8.0%, worse than no verifier (5.8%); scores within ±0.6 nats of zero, so flips are tiny threshold crossings |

All three reach "worth doing" under both the task-blind and neutral comparisons. With a fallback free of goal-object actions, all three drop to "no reliable change" or "too rare".

**Llama-3.1-8B** names objects strongly: at step 0 under the real task it says Yes to 54% of failing goal-object actions, against 5% task-blind. But it shows no any-task rejection: the task moves its `go to` scores by under 1 nat, so it accepts 90% of first moves and its fallback is rarely used. Under rule 1, which always uses the ranking, its effect is +4.4 points [1.1, 7.6].

## Reading and limits

- **What this licenses** (design note, lines 160–167; qualified by the analysis above): in an offline simulation over fixed candidate menus, a verifier that consults the world model with the agent's task chooses failing actions much more often than one without the task. Per the gate, a closed-loop agent study is worth running. The increase needs two things together: stating any concrete task makes the yes/no verifier reject working first moves, and naming the object ranks failing goal-object actions first in the fallback. The pre-registered attribution to naming the object holds only through that fallback ranking.
- **What it does not license:** harm to task success, effects that compound over an episode, or rates for other agents or menus. The menus are the collector's, rich in failing goal-object actions; a real agent's own alternatives are studied in Stage 1b.
- **Model dependence:** the result is carried by Qwen3-8B. Llama-3.1-8B's verifier is barely moved by the task (no reliable change).
- **Step 0:** most of the harm is at the first decision of an episode, where the agent's proposal is always a "go to" or "look".
