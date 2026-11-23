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

- **Qwen3-8B: a general task-consistency bias.** The pre-registered table's label here is "not specific to failing calls (fail minus witness about 0, negative or uncertain)". The value is in fact strongly negative: fail minus witness is −5.47 [−5.88, −5.05] ([tau_dx.md](tau_dx.md)). So the working witness call on the same order moves more than twice as much as the failing call. Naming the order raises Qwen's predicted success for calls on it, working calls more than failing ones. By the table this is read as goal leakage, not as an override specific to failures. A smaller part, about 1 nat, is consistent with a belief that the backend refuses writes the user did not request.
- **Llama-3.1-8B: small, and not robustly failure-specific.** By the pre-registered table, the failing call moves more than the witness, which reads as the ALFWorld effect in its stronger form, below the 0.5-nat size. The exploratory analysis (`tau_dx.md`) qualifies this:
  - the failure-specific part is +0.09 [0.04, 0.14] and comes from two cells (exchanges on pending orders, and returns);
  - the pooled +0.47 averages +1.23 on order-status failures with −0.33 on unavailable-variant failures;
  - the median item moves +0.25;
  - the failure-specificity reverses in the arm without the "backend does not see the task" statement.
  Its largest component is that any write request raises predicted success.

  *(Added 2026-11-23.)* `tau_metrics.json` stores Llama's reading as "The task overrides a visible fact about the named object, more for failing calls than for working ones: the ALFWorld effect in its stronger form (below the 0.5-nat size)". That string is the frozen outcome table's label for the observed pattern, written by the frozen analysis code; the metrics file is a frozen output and is not edited. `tau_dx.md` (summary and "Reading") qualifies it and advises against calling Llama's result "the ALFWorld effect in its stronger form", for the reasons in the list above.
- **Both:** the request to act adds to merely naming the order (write minus mention above 0; an authorization reading predicts this too).

## Other readouts (descriptive)

| Readout | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| First token, aligned − neutral | +1.42 [1.10, 1.74] | +0.92 [0.85, 0.99] |
| First token, swapped − neutral | −2.56 [−2.82, −2.28] | +0.45 [0.36, 0.53] |
| Yes/no (reply-anchored question), aligned − swapped | +7.68 [7.21, 8.10] | −0.02 [−0.043, −0.002] |
| Full reply, aligned − swapped | +1.88 [1.21, 2.60] | +2.26 [2.04, 2.49] |

*(Corrected 2026-11-23: Llama-3.1-8B's yes/no interval was printed as [−0.04, 0.00]. `tau_metrics.json` (`models.llama3.1-8b-fewshot.primary.yn.aligned_minus_swapped`) gives −0.022 [−0.043, −0.002], and the independent recompute in `tau_dx_metrics.json` gives the same interval, so it excludes zero: a very small negative shift, not an exact null. The reading that Llama's yes/no is an insensitive readout is unchanged.)*

- **Prompt without "the backend does not see the task" (ALFWorld-matched arm):** the effect remains under both prompts for both models, and is larger without the statement (main minus matched: Qwen −0.45, Llama −0.35 on the first token).
- **Sensitivity without the few-shot tool classes:** same verdict category for both models.
- **Natural (agent-produced) failures:** 28 items in 23 users, too few to read.

## Reading and limits

*(Corrected 2026-11-22 from the exploratory analysis in [tau_dx.md](tau_dx.md); the pre-registered verdicts are unchanged.)*

- τ-bench is the third environment. Qwen3-8B shows a large positive effect here, as in ALFWorld and TextCraft, but in τ-bench it is not specific to failing calls: naming the order raises its predicted success for working calls too. Llama-3.1-8B shows a small effect; by the pre-registered table it is specific to failing calls, but the failure-specific part is only +0.09 and comes from two cells (see above).
- Against the neutral line, Qwen's effect is largely the swapped task lowering predicted success (−2.56), the same pattern as in TextCraft.
- **The yes/no question still carries a task judgement for Qwen.** It is anchored to the backend's reply (fixed before scoring), yet about +3.9 of Qwen's +7.68 nats lies beyond its own predicted output. Among calls whose first token confidently predicts a JSON success, Qwen answers "No" to 90–93% under the swapped line, 51–67% under neutral and 17–32% under aligned. Anchoring to the output format did not remove the task reading found in TextCraft.
- **Llama's yes/no null is an insensitive readout, not robustness.** Its yes/no moves only 0.29 nats per nat of its first-token prediction.
- **Full-reply effects are mostly copying.** The success JSON repeats the order number and product named in the aligned line.
- **Practical size.** Under the aligned line, Qwen answers "Yes" to 57% of unavailable-variant failing calls, against 10% under neutral. Llama's decisions barely move (about 1–2% flips).
- One scoring run per model; the items are generated calls on real τ-bench users and orders, not agent proposals.
