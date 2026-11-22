# τ-bench retail replication: results (2026-11-22)

Design and reading rules: [tau_design.md](tau_design.md), fixed at commit `c737cd6` and pushed before full scoring (tag `taubench-design-frozen-2026-11-21`, created after scoring and before this analysis, adding no rule). Scoring: `tau_score`, 16 shards, all at `c737cd6` (items rebuilt and verified at that commit). Analysis: `tau_analysis` (ICE job 7239660). No scores missing or stale. Every number is from [tau_metrics.json](tau_metrics.json).

## Question and measure

A customer-service agent calls a simulated retail backend. A call's outcome depends only on the call and the database, never on the user's request. The test asks whether the world model predicts a call that the visible record shows will fail (for example, cancelling an order whose status is already "delivered") as more likely to succeed when the user's task names that order (aligned) than when it names another of the user's orders (swapped). The primary readout is the first token of the predicted tool output (`{"` for a JSON success against `Error` for a failure), on 856 status-matched items from 175 users, with 95% intervals that resample whole users.

## Pre-registered result

| Model | Effect on the first token, aligned − swapped (nats) | Verdict |
|---|---|---|
| Qwen3-8B | +3.98 [3.61, 4.35] | replicates (ALFWorld direction) |
| Llama-3.1-8B | +0.47 [0.41, 0.54] | small effect, direction positive (below the 0.5-nat size) |

Overall verdict, by the joint rule fixed in advance: **ALFWorld direction in both models, size met by one; not a replication.** Both models pass the manipulation check (the task line is read) and the discrimination check (the visible record is used).

## What each model's effect is, by the pre-registered reading table

- **Qwen3-8B: a general task-consistency bias.** The working "witness" call on the same order also moves toward success under the aligned task, and the failing call moves no more than the witness (fail minus witness about 0). So naming the order raises Qwen's predicted success for calls on it whether they fail or not. By the table this is read as goal leakage, not as an override specific to failures. It is also compatible with a belief that the backend checks calls against the request.
- **Llama-3.1-8B: the ALFWorld effect in its stronger form, but small.** The failing call moves more than the witness (fail minus witness above 0), so the task overrides a visible fact about the named object more for failing calls than for working ones. The size is below the 0.5-nat margin.
- **Both:** the request to act adds to merely naming the order (write minus mention above 0; an authorization reading predicts this too).

## Other readouts (descriptive)

| Readout | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| First token, aligned − neutral | +1.42 [1.10, 1.74] | +0.92 [0.85, 0.99] |
| First token, swapped − neutral | −2.56 [−2.82, −2.28] | +0.45 [0.36, 0.53] |
| Yes/no (reply-anchored question), aligned − swapped | +7.68 [7.21, 8.10] | −0.02 [−0.04, 0.00] |
| Full reply, aligned − swapped | +1.88 [1.21, 2.60] | +2.26 [2.04, 2.49] |

- **Prompt without "the backend does not see the task" (ALFWorld-matched arm):** the effect remains under both prompts for both models, and is larger without the statement (main minus matched: Qwen −0.45, Llama −0.35 on the first token).
- **Sensitivity without the few-shot tool classes:** same verdict category for both models.
- **Natural (agent-produced) failures:** 28 items in 23 users, too few to read.

## Reading and limits

- τ-bench is the third environment. Qwen3-8B shows a large positive effect here, as in ALFWorld and TextCraft, but in τ-bench it is not specific to failing calls: naming the order raises its predicted success for working calls too. Llama-3.1-8B shows a small effect that is specific to failing calls.
- Against the neutral line, Qwen's effect is largely the swapped task lowering predicted success (−2.56), the same pattern as in TextCraft.
- The yes/no question here is anchored to the backend's reply (fixed before scoring), so it is not subject to the task-success reading found in TextCraft. Llama's yes/no effect is null while its first-token effect is positive; Qwen's is large on both.
- One scoring run per model; the items are generated calls on real τ-bench users and orders, not agent proposals.
