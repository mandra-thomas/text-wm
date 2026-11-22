# τ-bench retail: what the goal effect is, where it lives, and how competent each model is (exploratory, post hoc)

Everything in this note is exploratory and post hoc. It was written after the confirmatory τ-bench result had been read ([tau_results.md](tau_results.md), [tau_metrics.json](tau_metrics.json); design frozen at `c737cd6`, tag `taubench-design-frozen-2026-11-21`). It changes no confirmatory verdict. The method follows the TextCraft diagnostics (`notes/textcraft_llama_dx.md`, `notes/textcraft_llama_competence.md`, `notes/textcraft_qwen_robustness.md`, `notes/textcraft_qwen_competence.md` on branches `textcraft-llama-dx`, `textcraft-qwen-robust` and `textcraft-qwen-competence`).

## Plan and heuristics (fixed 2026-11-22, before any item-level score was read)

**What had been read when this was written.** The aggregate numbers in `tau_metrics.json` (per-model effects on the primary population, per class for `status` and `unavailable`, the witness, the mention pair, the matched and rules-only arms, the natural set). No item-level score had been opened. The item and build files had been read for their structure only (fields, the class × tool × status composition of the primary population). The heuristics below are therefore defined without having seen how either model scores any individual call.

**Data.** The raw records of `tau_score` (16 shards, all at `c737cd6`) and the build `tau_build` (sha256 of `items.jsonl`, `natural_items.jsonl` and `fewshot.json` checked against the pins `2c8ffeac…`, `9bd6c3f7…`, `231607f7…`). The scripts are `scripts/tau_dx.py` (everything) and nothing in it imports `wf/tau/pipeline.py`; the per-item scores are re-read from the records and the bootstrap is a separate implementation (numpy, multinomial cluster weights).

**Inference.** 95% percentile intervals from 2,000 resamples of whole users (seed 20260924) unless a different cluster is named. No p-values. Many cuts are examined, so a single interval excluding zero is weak evidence on its own; findings rest on patterns that recur across cuts and readouts.

### Questions and analyses

1. **Independent recompute and distribution.** Primary first-token, yes/no and full-reply effects; the per-user mean distribution (median, quartiles), item median, 10% and 20% trimmed means, share of items and of users positive, leave-one-user-out range, the effect after dropping the top-k most influential users (k = 1, 5, 10, 20), and the share of the total carried by the top 10% of users.
2. **Heterogeneity.** By class; by tool × order status (the four cells of the `status` and `unavailable` classes); by history length (number of history calls and characters, in thirds); by whether the target order's record comes first or second in the history; by whether the swapped line's product also occurs in the target order (and the 7 items where both lines name the same product, so that they differ only in the order number); by few-shot order. The position of the failing call in the agent's trajectory exists only for natural items. Whether the target order was mentioned in the user's own messages exists only for natural items (generated items have no user text other than the task line); it is read from the released trajectories on ICE, as a boolean per item, without copying their text.
3. **Specificity and the witness.** Fail minus witness, per class and per tool × status cell, and its per-item distribution. The four readouts that bear on the three readings: (a) a task-consistency bias, (b) a belief that the backend authorizes calls against the request, (c) an override of a visible failure. What each predicts is set out in section 3 before the numbers.
4. **Competence and heuristics.** Under the neutral line: the AUC between the witness (working) calls and the failing calls on the first-token and yes/no readouts, overall, per class and per tool × status cell; Yes rates; calibration (binned) and Brier score against the base rate; the families competence rule (yes/no, neutral line, AUC at least 0.70, Yes on at most 95% of failures and at least 5% of successes, median P(Yes) + P(No) at least 0.5); and the heuristics below.
5. **Readout agreement.** First token against full reply against yes/no, item by item (Spearman, sign agreement); the excess of the yes/no effect over the first-token effect; the shift at a matched first-token score (the regression of the TextCraft robustness note, section 9); the share answered No among calls whose first token confidently predicts success, per line.
6. **Practical size.** Flips of the predicted outcome in both directions, on the first token and on yes/no; Yes rates by line on failing and working calls; the probability shift.
7. **Prompt arms.** Main against matched against rules-only on the same 500 items, per class, fail and witness, and against neutral.
8. **Cross-environment.** τ-bench beside ALFWorld and TextCraft on the first token and the yes/no question, with the neutral-line decomposition.
9. **Natural items.** What 28 items can and cannot say.

### Heuristics (fixed now, before any fit)

Each heuristic is a rule that reads only the visible text and predicts success (1) or failure (0) for a call. They are evaluated on the primary population's calls: the 856 failing calls and their 856 witnesses (the first single write that achieves the aligned task on the same state; it succeeds by construction). The truth T is 0 for every failing call and 1 for every witness.

- **H1, status rule only.** Success when the tool's status rule passes for the target order's status as shown in its `get_order_details` output (`modify_pending_order_items`: exactly `pending`; `exchange_delivered_order_items` and `return_delivered_order_items`: exactly `delivered`; `modify_pending_order_payment`: contains `pending`), whatever else the call does. A model that tracks order status and nothing else.
- **H2, availability only.** Failure when any new item id of the call is shown in a `get_product_details` output with `available: false`; success otherwise, including when no product record is shown. A model that checks variant availability and nothing else.
- **H1 ∧ H2.** Success when both pass. On the `status` and `unavailable` classes this equals the truth except for the `status` witnesses, whose availability is not shown (H2 then says success by default, which is the truth).
- **H3, the word "delivered".** Failure when the target order's shown status is `delivered`, whatever the tool. The surface cue named in the brief.
- **H4, the tool name.** Failure when the tool is `exchange_delivered_order_items`, success for `modify_pending_order_items` (the two tools of 95% of the calls); other tools are left out of this heuristic's fit. A tool prior that ignores the record.
- **H5, gift-card payment.** Failure when the call's payment method is a gift card. Irrelevant to the truth on these items (every twin and witness passed its balance check); a distractor that a model reading "gift card" as risky would follow.
- **H6, crowded variants** (continuous, `unavailable` class only). The share of the target product's variants shown with `available: false`. It is the same for an item's failing call and its witness, so it cannot separate them; it asks whether the model's level of predicted success follows the product's general unavailability instead of the named variant's.

**How they are fitted** (the TextCraft competence notes' measures): each heuristic's validity (agreement with T); its tracking AUC, the AUC of the model's neutral-line score between the calls the heuristic calls successes and those it calls failures; Cohen's kappa between the heuristic and the model's sign; and a joint least-squares regression of the neutral score on T and the heuristic indicators, with user-bootstrap intervals.

**The status 2 × 2, fixed now.** On the `status` and `unavailable` classes every call is `modify_pending_order_items` or `exchange_delivered_order_items` on a `pending` or a `delivered` order. The neutral-line score is regressed on three indicators: the order is `delivered` (the surface cue, H3), the tool is exchange (the tool prior, H4), and the tool's status rule passes (H1, the interaction that only a model reading the status against the tool can produce), plus an indicator for an unavailable new variant (H2's failure). Reading fixed now: if the status-rule coefficient is large and positive and the `delivered` and tool coefficients are small, the model tracks status against the tool. If the `delivered` coefficient is large and negative while the rule coefficient is small, it relies on the word. If the tool coefficient dominates, it relies on the tool name.

**What each reading predicts (question 3), fixed now.**

- *(a) Task-consistency bias:* naming the order raises predicted success for any call on it. Predicts a positive aligned − swapped effect on the witness at least as large as on the failing call (fail minus witness at most about 0), and a positive mention-aligned − mention-swapped effect (a question about the order is enough).
- *(b) Authorization belief:* the model thinks the backend refuses writes the user did not request. Predicts swapped below neutral for both the failing call and the witness; the mention lines, which request no write, should not raise success over neutral; the effect should grow in the matched arm (no statement that the backend does not see the task) and shrink, if anything, in no arm that keeps the statement. It also predicts write minus mention above 0 (a question authorizes no write).
- *(c) Override of a visible failure:* the task outweighs the record that shows the failure. Predicts fail minus witness above 0 and aligned above neutral on the failing call, and an effect that is larger where the failure is decided by a record the model reads (the `unavailable` class, where discrimination is shown) than where it is not.

These predictions overlap; section 3 states which contrasts separate them and which do not.

### GPU (question 10)

None is planned. If a question above cannot be answered from the scores, a dated plan will be added here and committed before any run.
