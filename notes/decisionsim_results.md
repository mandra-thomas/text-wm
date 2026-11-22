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

- **Mostly at step 0.** At the 134 step-0 states, where the agent's proposal always works, the real-task verifier rejects the proposal at 110 states where the task-blind verifier accepts it. Its fallback then fails at 86 of them. The extra failure rate is +72 points at step 0 and +7.6 points [4.8, 10.5] at later steps.
- **By false rejection of working proposals.** Harm comes from states where only the task-blind verifier accepts the proposal (aligned rejects a working action and falls back to a failing one) and from states where both reject it. Almost none comes from accepting a failing proposal: the agent's own proposals rarely fail.
- This matches the item-level findings: under the real task, Qwen lowers its predicted success for working actions that do not act on the task's object (`exploratory_results.md`, section 7), and raises it for failing actions on the task's object, which fill the fallback menu.

## Reading and limits

- **What this licenses** (design note, lines 160–167): in an offline simulation over fixed candidate menus, a verifier that consults the world model with the agent's task chooses failing actions much more often than one without the task, and naming the task's object accounts for well over half of the increase, mostly through its judgement of the proposal. Per the gate, a closed-loop agent study is worth running.
- **What it does not license:** harm to task success, effects that compound over an episode, or rates for other agents or menus. The menus are the collector's, rich in failing goal-object actions; a real agent's own alternatives are studied in Stage 1b.
- **Model dependence:** the result is carried by Qwen3-8B. Llama-3.1-8B's verifier is barely moved by the task (no reliable change).
- **Step 0:** most of the harm is at the first decision of an episode, where the agent's proposal is always a "go to" or "look".
