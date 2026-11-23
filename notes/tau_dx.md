# τ-bench retail: what the goal effect is, where it lives, and how competent each model is (exploratory, post hoc)

Everything in this note is exploratory and post hoc. It was written after the confirmatory τ-bench result had been read ([tau_results.md](tau_results.md), [tau_metrics.json](tau_metrics.json); design frozen at `c737cd6`, tag `taubench-design-frozen-2026-11-21`). It changes no confirmatory verdict. The method follows the TextCraft diagnostics (`notes/textcraft_llama_dx.md`, `notes/textcraft_llama_competence.md`, `notes/textcraft_qwen_robustness.md`, `notes/textcraft_qwen_competence.md` on branches `textcraft-llama-dx`, `textcraft-qwen-robust` and `textcraft-qwen-competence`).

## Summary

All of this is exploratory and post hoc, on CPU, from the confirmatory scores; nothing was scored for it. Intervals are 95% intervals from resampling whole users (175 users, 856 primary items). "Aligned" is the task line that names the call's own order and product, "swapped" names another order of the same user with the same status, and "neutral" is "The user wants help with their account." The "witness" is a working call on the same order that would achieve the aligned task; it is scored under the same three lines.

- **The confirmatory numbers reproduce exactly.** An independent recompute gives Qwen3-8B +3.98 [3.63, 4.34] and Llama-3.1-8B +0.47 [0.40, 0.54] on the first token, and every other primary readout to three decimals. Qwen's effect is broad: positive in 78% of items and 94% of users, median +3.25, 20% trimmed mean +3.49, and +3.41 after dropping the 20 most influential users. Llama's is not robust in size: its median item effect is +0.25, its trimmed means +0.34 to +0.40 and its mean after dropping 10 users +0.41, all below the 0.5-nat size, and 59% of items are positive.
- **Llama's pooled effect is an average of two opposite class effects.** On the order-status failures (409 items) it is +1.23 [1.10, 1.34]; on the variant-unavailable failures (405) it is −0.33 [−0.39, −0.27], a small reversal. The pooled +0.47 and its "small positive" verdict depend on the roughly even mix of the two classes. Qwen is positive in every class but the six gift-card items (status +2.12, unavailable +5.34, the 36 return items +10.6), but zero in one of the four tool-by-status cells (a modify call on a delivered order: +0.00 [−0.42, +0.42], 165 items).
- **Qwen's effect is not "the same on working calls": working calls move more than twice as much.** The witness moves +9.45 [8.94, 9.94] against +3.98 for the failing call, so failing minus witness is −5.47 [−5.87, −5.05], negative in 85% of items and in every class except the 36 returns. This holds at matched baseline scores (the witness moves about 3.2 nats more at the same neutral score), and the effect grows with Qwen's own predicted success: about +2.8 nats on failing calls it confidently predicts to fail, about +9.8 on those it confidently (and wrongly) predicts to succeed. The results note's description "fail minus witness about 0" should be corrected. The reading "a task-consistency bias, not an override specific to failures" survives, in a stronger form: a visible failure that Qwen sees damps the task's pull; it does not amplify it.
- **Llama's "specific to failing calls" rests on one cell.** Failing minus witness is +0.09 [0.04, 0.14] pooled, positive in only 49% of items, +0.68 in the exchange-on-pending status cell and on the 36 returns, −0.10 in the other status cell, −0.27 in the unavailable class, and −0.18 in the arm without "the backend does not see the task". The pre-registered reading "the ALFWorld effect in its stronger form" does not hold up item by item, by class or across prompts. A better description: Llama raises predicted success whenever the user asks for any write (aligned +0.92 and swapped +0.45 over neutral, while naming the order without a request lowers it by 0.5), with a small extra lift for the named order that working and failing calls share.
- **Competence.** Both models discriminate under the neutral line (first-token AUC between working and failing calls 0.80 for both; 0.92 for Qwen and 0.84 for Llama on calls whose outcome the history determines), and both pass the families competence rule on yes/no. Both read order status against the tool, not the word "delivered": within the status class, a call whose tool matches the order's status scores +4.9 nats higher for Qwen and +3.0 for Llama, while "delivered" itself adds under 1 nat. So the design's caveat that the status class gives "no discrimination evidence" is too strong; it lacks a determined witness, but both models visibly use the status record. Their probabilities are poor. Qwen's first token predicts success for 38% of failing calls (60% of unavailable-variant calls, all 36 wrong-refund-method returns), and only 76% of calls it gives above 0.99 succeed; Llama predicts failure for 79% of working calls and is worse than the base rate in Brier score.
- **Yes/no.** Qwen's yes/no effect (+7.68) is its first-token shift plus about +3.9 nats that its own predicted output does not account for. The anchored question did not remove the task reading that TextCraft showed: among calls whose first token confidently predicts a JSON success, Qwen answers "No" (an error) for 90–93% under the swapped line, 51–67% under neutral and 17–32% under aligned. Llama's yes/no null comes from an insensitive readout, not robustness: its yes/no moves 0.29 nats per first-token nat, its answers sit between P(Yes) 0.1 and 0.9 for every call, and a small negative task term cancels the compressed first-token shift.
- **Practical size.** Qwen changes decisions. Under aligned it answers Yes for 57% of unavailable-variant failing calls, against 10% under neutral and 1% under swapped. Under swapped it predicts failure on 13% of the working calls it confidently got right under neutral (first token), and answers No on 91% of such calls (yes/no). Llama's decisions barely move (flips 1–2%, probability shift +0.65 points).
- **Prompt arms.** Dropping "the backend does not see the task" lowers mostly the swapped line (Qwen −1.14, Llama −0.35 nats; aligned −0.69 and 0.00). So the statement partly protects calls on an order the user did not ask about. That fits a belief that the backend checks requests (reading b), but it is only part of the swapped penalty, which stays at −2.56 for Qwen with the statement present. Dropping the agent policy raises Qwen's aligned line by 1.60 nats and all of Llama's lines by about 0.5.
- **Which readings the data separate.** An override of visible failures (c) is ruled out for Qwen and is at most a cell-level effect for Llama. A task-consistency or relevance bias (a) is Qwen's main component: a mere question about the other order, with no write request, lowers the call's predicted success by 1.98 nats. An authorization belief (b) accounts for about a nat of the swapped penalty, shown by the matched arm. The data cannot split (a) from (b) for the write request itself.
- **Cross-environment.** On the first token τ-bench has the largest Qwen effect of the three environments (3.98 against ALFWorld 2.52 and TextCraft 0.97), and Llama's is between ALFWorld (0.68) and TextCraft (−0.16). Qwen's "wrong task penalizes" signature recurs on the first token (64% of the effect is swapped below neutral, against 56% in TextCraft and none in ALFWorld). On yes/no the aligned lift dominates, as in ALFWorld. In τ-bench and TextCraft Qwen moves working calls more than failing ones; ALFWorld's comparison went the other way on the readouts available. The environments differ in readout words, yes/no wording, neutral line and where the deciding fact sits, so only the direction and the sign structure are comparable, not the sizes.
- **Natural failures (28 items, 23 users).** They are mostly status failures on unusual statuses ("exchange requested", "return requested") left by the agent's own earlier writes. Qwen's first token predicts success for 64% of them under neutral, so as a world model it would miss most real agent errors whatever the task says. The effect intervals are ±2.3 nats wide for Qwen and include both zero and the generated-set value.
- **Implications.** The paper can say that Qwen's τ-bench effect is large, broad and robust, and that it is a task-relevance bias that moves working calls more than failing ones and is damped, not overridden, where Qwen sees the failure. It should not call Llama's result "the ALFWorld effect in its stronger form": Llama shows a small request-driven lift whose sign depends on the failure class. The full-reply readout should not be compared across models: Llama's +2.26 is mostly the success JSON's copy of the order number and product name that the aligned line contains.

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

### What was added while computing (recorded, not hidden)

Four analyses were added after the plan above was committed, each before its numbers were read: (1) the fail-versus-witness comparison at a matched neutral baseline (section 3), because the witness's neutral score sits far above the failing call's; (2) the main pool's unmatched status items, split by whether the tool's status rule passes for the order the swapped line names (section 3), to see whether the swapped line acts through that order's status; (3) a version of the status 2 × 2 regression with an indicator for the status class's shorter history and one for gift-card payment, beside the planned one (section 4); (4) the split of the full-reply tail into its success part and its failure part (section 5). The cut "both lines name the same product" turned out to hold for one item only once payment changes (which name no product) were excluded, so it is not read.

## Data and method

The raw records were copied from ICE (`runs/tau_score/shard_*_of_16/scoring.jsonl`, 82,488 records, all 16 shards at `c737cd6`, no duplicate keys) with the build files, whose sha256 values match the pins. The script `scripts/tau_dx.py` reads them with its own code and writes [tau_dx_metrics.json](tau_dx_metrics.json). The natural items' mention flags were computed on ICE from the released trajectories by a small script that prints booleans only (no trajectory text was copied).

For a call and a task line, the first-token score is the log-odds of the two possible openings of the backend's reply, the JSON brace against the word "Error":

$$L_1 = \log P(\texttt{\{"}) - \log P(\texttt{Error}).$$

The yes/no score is the log-odds of "Yes" against "No" after the question "will the backend execute this call without returning an error? Answer Yes if its output is JSON and No if its output is a line starting with "Error: "". The full-reply score is the log-probability of the whole success reply minus that of the whole failure reply. The goal effect of item $i$ is

$$\Delta L_i = L_i(\text{aligned}) - L_i(\text{swapped}),$$

and "aligned minus neutral" and "swapped minus neutral" are defined the same way. "Failing minus witness" is the per-item difference between the failing call's $\Delta L_i$ and the witness's. A model "predicts success" when its score is above zero. The AUC is the probability that a random working call scores above a random failing call (0.5 is chance). Every interval resamples whole users, 2,000 times. The first-token scores come from bf16 logits and take values on a grid of about 0.125 nats, so a few item differences are exactly zero (0.4% for Qwen, 0.5% for Llama); this does not matter for any mean, and it is why medians move in steps of 0.125.

## 1. Independent recompute and distribution

Every primary estimate in `tau_metrics.json` reproduces to three decimals: first token +3.983 (Qwen) and +0.473 (Llama), yes/no +7.678 and −0.022, full reply +1.878 and +2.265, per-token full reply −0.059 and +0.015, and the witness's effects. The bootstrap is a different implementation and its intervals agree to about 0.02: Qwen [3.63, 4.34] against [3.61, 4.35], Llama [0.40, 0.54] against [0.41, 0.54].

| Distribution of the item effect (first token, primary) | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Mean | +3.98 [3.63, 4.34] | +0.47 [0.40, 0.54] |
| Median item | +3.25 [2.75, 3.75] | +0.25 [0.13, 0.25] |
| 10% trimmed mean | +3.64 [3.28, 4.01] | +0.40 [0.33, 0.46] |
| 20% trimmed mean | +3.49 [3.11, 3.89] | +0.34 [0.27, 0.41] |
| Item quartiles | +0.25, +3.25, +6.75 | −0.38, +0.25, +1.25 |
| 5th and 95th percentile | −3.0, +13.0 | −1.1, +2.8 |
| Share of items positive | 78% [75, 81] | 59% [56, 62] |
| Share of users with a positive mean | 94% | 84% |
| Leave one user out, range | 3.94 to 4.02 | 0.47 to 0.48 |
| Drop the 5 / 10 / 20 most influential users | 3.79 / 3.65 / 3.41 | 0.44 / 0.41 / 0.36 |
| Share of the total from the top 10% of users | 26% | 34% |

Qwen's effect is broad. It is positive for nearly every user, its robust location measures sit within 0.5 nats of the mean, and no handful of users carries it. Its spread is large (item SD 4.96 nats), which is why the smoke run's projected interval was wide; the full run's interval is not.

Llama's direction is robust (84% of users positive, and every measure above zero), but its size is not. The confirmatory rule used the mean, and the mean is pulled up by a right tail: the median item is +0.25, both trimmed means are below 0.4, and dropping ten users gives 0.41. The "small effect" verdict is the right category, and any statement that Llama's effect is "close to" the 0.5-nat size overstates it.

## 2. Where each model's effect lives

### By failure class and by tool and status

The primary population has six cells. The status class has two: an exchange called on a pending order (244 items, whose witness is a modify call on the same order) and a modify called on a delivered order (165, witness an exchange). The unavailable class has a modify on a pending order (243) and an exchange on a delivered order (162), each asking for a variant shown as unavailable, with the witness asking for an available variant of the same product. Then 36 returns to a payment method that is neither the original nor a gift card, and 6 payment changes to a gift card with too little balance.

| Cell (items) | Qwen fail | Qwen witness | Qwen fail − witness | Llama fail | Llama witness | Llama fail − witness |
|---|---|---|---|---|---|---|
| All (856) | +3.98 [3.63, 4.34] | +9.45 [8.94, 9.94] | −5.47 [−5.87, −5.05] | +0.47 [0.40, 0.54] | +0.38 [0.32, 0.44] | +0.09 [0.04, 0.14] |
| Status class (409) | +2.12 [1.71, 2.55] | +7.63 [7.21, 8.07] | −5.51 [−5.99, −5.06] | +1.23 [1.10, 1.34] | +0.86 [0.77, 0.95] | +0.36 [0.28, 0.45] |
| … exchange on a pending order (244) | +3.55 [3.11, 4.00] | +8.24 [7.73, 8.79] | −4.69 [−5.27, −4.17] | +1.55 [1.42, 1.68] | +0.88 [0.75, 0.99] | +0.68 [0.59, 0.77] |
| … modify on a delivered order (165) | +0.00 [−0.42, 0.42] | +6.74 [6.03, 7.45] | −6.74 [−7.44, −6.03] | +0.74 [0.60, 0.88] | +0.84 [0.69, 0.99] | −0.10 [−0.21, 0.01] |
| Unavailable class (405) | +5.34 [4.86, 5.82] | +11.09 [10.34, 11.82] | −5.75 [−6.36, −5.13] | −0.33 [−0.39, −0.27] | −0.06 [−0.13, 0.00] | −0.27 [−0.32, −0.22] |
| … modify on a pending order (243) | +6.35 [5.74, 6.96] | +13.33 [12.47, 14.24] | −6.98 [−7.81, −6.20] | −0.21 [−0.28, −0.14] | −0.07 [−0.16, 0.01] | −0.14 [−0.19, −0.09] |
| … exchange on a delivered order (162) | +3.82 [3.12, 4.56] | +7.72 [6.80, 8.67] | −3.90 [−4.75, −3.11] | −0.51 [−0.60, −0.41] | −0.04 [−0.15, 0.05] | −0.46 [−0.54, −0.38] |
| Wrong refund method (36) | +10.58 [9.06, 11.98] | +10.51 [8.79, 12.19] | +0.07 [−1.75, 1.77] | +0.89 [0.64, 1.14] | −0.04 [−0.46, 0.39] | +0.93 [0.51, 1.35] |
| Gift-card balance (6) | +0.08 | +16.31 | −16.2 | +0.71 | +0.21 | +0.50 |

**Qwen** is positive in every class except the six gift-card items, and largest where it is least sure of the failure (the 36 returns, which its first token calls successes 100% of the time under neutral, section 4). The one null cell is the modify call on a delivered order: there both order-naming lines lower its predicted success by the same 2.77 nats below neutral, so naming which delivered order makes no difference. In the other status cell, the exchange on a pending order, the aligned line lifts the call 0.90 nats above neutral and the swapped line drops it 2.65 below. Both lines name a pending order, so the tool fits the task equally badly under both, and the 3.55-nat gap comes purely from which order is named.

**Llama**'s effect lives in the status class. In the unavailable class, the one where the design shows discrimination, it is a small reversal: the aligned line lowers the failing call (relative to swapped) and leaves the witness alone. By the design's own reading table that pattern in that class would read "naming the object sharpens failure prediction", at a size (−0.33) well below any threshold. The pooled +0.47 is therefore close to the weighted average of +1.23 and −0.33 over a near-even class mix; with the unavailable class alone Llama's verdict would have been a small reversal, with the status class alone a replication.

### By history length, presentation order and few-shot order

History length is confounded with class (status histories have three calls and about 2,700 characters of output; unavailable histories five calls and about 7,300), so it is read within class. Within class, length does little for either model. For Qwen, the status class gives +2.67 below its median length and +1.56 above; the unavailable class +5.83 and +4.84. For Llama the status class gives +1.47 and +0.98 and the unavailable class −0.37 and −0.29. The across-thirds gradient in Llama's pooled effect (+1.29, +0.40, −0.29) is the class mix.

Two presentation factors move Llama's effect a lot. The target order's record comes before the comparison order's in about half the items (counterbalanced by hash). When it comes first, and so further from the call, Llama's status-class effect is +1.84; when it comes second, nearest the call, +0.64. Few-shot order (also counterbalanced) moves Llama's pooled effect from +0.65 (design order) to +0.30 (reversed). Qwen moves less: +3.58 against +4.37 by record order and +4.05 against +3.92 by few-shot order. The averages are unbiased, because both factors are balanced, but Llama's effect depends on surface order by more than its own size, which fits a fragile effect better than a stable belief.

Neither the swapped line's product also occurring in the target order (36 items: Qwen +3.23, Llama +0.34) nor the target product occurring in the comparison order (47 items: +4.45 and +0.39) changes the picture. Whether the witness pays with the same payment method as the failing call changes nothing either.

### Whether the order was mentioned in the user's own messages, and the call's position

Generated items have no user messages besides the task line, so this exists only for the 28 natural items (section 9). In the released trajectories the user named the target order before the failing call in 7 of the 28, and the product in 20. The split effects are 7 items against 21 and are not interpretable. The failing call's position in the trajectory (step quartiles 6, 6, 7) correlates with Qwen's aligned-minus-neutral at a Spearman 0.49 over 28 items, which is also too few to read.

## 3. Specificity, the witness, and the three readings

### Fail against witness, at a matched baseline

A larger shift on the witness could be an artefact of where its score sits: the witness's neutral first-token score averages +6.2 nats for Qwen, the failing call's −0.4. Binning calls by their neutral score shows the opposite of an artefact. At every level the witness moves more, and both move more the more Qwen believes the call succeeds:

| Qwen, neutral first-token score of the call | Failing calls, aligned − swapped | Witnesses, aligned − swapped |
|---|---|---|
| below −5 | +2.76 [2.21, 3.31] | +6.45 [3.58, 9.58] |
| −5 to −2 | +2.29 [1.80, 2.77] | +5.32 [4.01, 6.68] |
| −2 to 0 | +2.99 [2.32, 3.69] | +6.79 [5.86, 7.82] |
| 0 to 2 | +3.77 [2.81, 4.85] | +6.96 [6.20, 7.73] |
| 2 to 5 | +5.50 [4.50, 6.51] | +9.02 [8.25, 9.83] |
| 5 to 10 | +7.72 [6.47, 8.93] | +10.42 [9.57, 11.27] |
| above 10 | +9.82 [7.90, 11.59] | +11.93 [10.96, 12.91] |

Reweighting the witnesses to the failing calls' distribution over these bins leaves the witness 3.2 nats ahead on the first token and 4.3 on yes/no. So the gap is not where the scores sit. The gradient is the reverse of TextCraft's, where Qwen's effect was largest where it was least sure: here it is largest where Qwen is surest the call works, whether or not it does. The simplest description is that the task line scales up whatever success belief the record leaves. Where Qwen reads the failure (a strongly negative neutral score), the task moves the call least. For Llama the matched-baseline difference is +0.11, about nothing.

### Does the swapped line act through the status of the order it names?

The primary population keeps the swapped order's status equal to the target's. The main pool's unmatched status items (secondary, and different in other ways) show what happens when it is not. For Qwen's exchange-on-pending cell the effect turns from +3.55 (matched) to −5.41 [−6.75, −4.05] when the swapped line names an order whose status the exchange rule accepts (a delivered order), and to −3.12 [−5.40, −0.81] when it names one that the rule also rejects (for example a processed order). In both cases the swapped line raises the call above neutral by 4.6 to 4.9 nats. So when the task asks to swap an item in an order that is not pending, Qwen becomes more confident that an exchange call on a *different*, pending order will work. Qwen therefore judges the tool partly against the order the task names, not only against the order in the call's arguments. That is a task-consistency effect at the level of "is this the right tool for the request". It does not appear in the other status cell (modify on a delivered order: −0.10 and −1.74), and Llama shows only a weakening (+1.55 to +0.69). The design's decision to make the primary population status-matched (A3) was right: in unmatched items this binding would have dominated.

### The three readings

The design named three readings of a positive effect. (a) A task-consistency or relevance bias: naming the order raises the success belief of any call on it. (b) An authorization belief: the model thinks the backend refuses writes that the user did not request. (c) An override of a visible failure: the task outweighs the record that shows the failure. The predictions were written before the numbers. Four contrasts bear on them: failing calls against witnesses, the mention pair (task lines that only say the user has a question about the order and product, with no request), the matched arm (the prompt without the statement that the backend does not see the task), and the neutral line.

| First token, primary | Qwen | Llama |
|---|---|---|
| Write request, aligned − neutral | +1.42 [1.09, 1.73] | +0.92 [0.85, 0.99] |
| Write request, swapped − neutral | −2.56 [−2.84, −2.30] | +0.45 [0.36, 0.53] |
| Question only, aligned − neutral | −0.17 [−0.36, 0.01] | −0.54 [−0.58, −0.51] |
| Question only, swapped − neutral | −1.98 [−2.17, −1.80] | −0.52 [−0.57, −0.47] |
| Question only, aligned − swapped | +1.81 [1.60, 2.02] | −0.03 [−0.07, 0.02] |
| Write minus question (the two aligned − swapped effects) | +2.17 [1.92, 2.43] | +0.50 [0.45, 0.54] |
| Failing call minus witness | −5.47 [−5.87, −5.05] | +0.09 [0.04, 0.14] |
| Matched arm minus main, swapped line (500 items) | −1.14 [−1.26, −1.01] | −0.35 [−0.41, −0.29] |
| Matched arm minus main, aligned line | −0.69 [−0.81, −0.56] | +0.00 [−0.04, 0.04] |

**Qwen.** Reading (c) predicts failing minus witness above zero. It is −5.47, and below zero in every class except the 36 returns (+0.07), so (c) is ruled out for Qwen as a description of the effect. Reading (a) predicts that naming an order is enough. It is: a question about the other order, with nothing requested, lowers the call's predicted success by about 2 nats, and the question pair alone gives +1.81, 45% of the write effect. Reading (b) does not predict a question-only effect, because a question authorizes nothing on either order. It does predict the matched-arm result: without the statement that the backend does not see the task, the swapped line falls by a further 1.14 nats (the witness's swapped line by 1.6). So Qwen's effect is mainly (a), with a (b) component of about a nat on the swapped side that the statement suppresses but does not remove. What the data cannot do is split the write request's extra +2.17: both readings predict it.

**Llama.** Its largest term is the presence of a write request: both write lines raise the call above neutral, and both question lines lower it by the same half nat, so naming the order without a request does nothing (−0.03). That is neither (a), which predicts a question-only effect, nor (b) in its plain form, which predicts the swapped write line at or below neutral (it is +0.45 above). The order-specific part, +0.47, is about the same for failing calls and witnesses overall (+0.38), and item by item failing minus witness is positive in only 49% of items. Its pooled +0.09 comes from the exchange-on-pending cell (+0.68) and the returns (+0.93); it is −0.27 in the unavailable class and −0.18 [−0.29, −0.07] in the matched arm. So (c) holds for Llama only in particular cells and prompts. The matched arm also nearly doubles Llama's order-specific effect, to +0.82 on failing calls and +1.00 on witnesses, by lowering the swapped line, which points to a (b) component for the order-specific part. The most defensible description of Llama: a request-driven lift in predicted success, plus a small order-specific lift that is partly suppressed by stating that the backend does not see the task, with no consistent preference for failing calls.

## 4. Competence and heuristics

### Discrimination, Yes rates and calibration (neutral line)

| Neutral line | Qwen first token | Qwen yes/no | Llama first token | Llama yes/no |
|---|---|---|---|---|
| AUC, all 1,712 calls | 0.80 [0.79, 0.82] | 0.73 [0.72, 0.74] | 0.80 [0.78, 0.81] | 0.73 [0.72, 0.75] |
| AUC, calls with a determined outcome | 0.92 [0.91, 0.94] | 0.92 [0.90, 0.93] | 0.84 [0.83, 0.86] | 0.86 [0.85, 0.87] |
| AUC, status class | 0.88 [0.86, 0.91] | 0.99 [0.98, 0.99] | 0.94 [0.92, 0.96] | 0.93 [0.91, 0.95] |
| AUC, unavailable class | 0.89 [0.87, 0.91] | 0.86 [0.84, 0.88] | 0.74 [0.72, 0.76] | 0.75 [0.74, 0.77] |
| Predicts success, failing calls: status / unavailable / returns | 11% / 60% / 100% | 0% / 10% / 47% | 0% / 3% / 56% | 0% / 34% / 69% |
| Predicts success, working calls: status / unavailable / returns | 67% / 99% / 100% | 0% / 59% / 78% | 18% / 17% / 100% | 15% / 70% / 86% |
| Brier score (base rate 0.25) | 0.228 | 0.341 | 0.331 | 0.217 |
| Families rule (yes/no) | | passes (AUC 0.73; Yes 7% of failures, 32% of successes) | | passes (AUC 0.73; Yes 20% / 44%) |

The status witnesses' success depends on a variant's availability, which their histories do not show, so "determined" leaves them out. The AUCs here compare an item's failing call with working calls, half of which share its order; they are not the design's discrimination contrast, which is a paired mean.

**Qwen** ranks well and is badly calibrated in both directions. Its first token says success for 60% of calls asking for an unavailable variant and for every return to a wrong refund method, and of the 618 calls it puts above 0.99, 24% fail. Its yes/no is the opposite: it says No to every status-class call, working or not, and 41% of the 1,003 calls it puts below 0.01 actually work. Its yes/no Brier score is worse than always saying 50%. On the status class it fails the families rule's "at least 5% Yes on successes" criterion, because it never says Yes there.

**Llama** ranks moderately and leans hard toward failure on the first token: under neutral it predicts an error for 79% of working calls, and 44% of the 853 calls it puts between 1% and 10% succeed. It ranks status well (0.94) and availability poorly (0.74). Its yes/no answers are compressed: every call gets P(Yes) between 0.1 and 0.9.

Both models pass the families competence rule on the pooled yes/no readout, so neither result is gated out by that rule, as Llama's TextCraft result would have been.

### What the models rely on (heuristics H1 to H6)

| Neutral first token: tracking AUC (and validity) | Qwen | Llama |
|---|---|---|
| Truth (working against failing) | 0.80 | 0.80 |
| H1, status rule only (validity 0.74) | 0.91 [0.89, 0.92] | 0.95 [0.94, 0.97] |
| H2, availability only (0.74) | 0.55 [0.53, 0.58] | 0.50 [0.48, 0.52] |
| H1 ∧ H2 (0.98) | 0.83 [0.82, 0.85] | 0.83 [0.82, 0.85] |
| H3, "delivered" means failure (0.50) | 0.47 [0.45, 0.50] | 0.27 [0.26, 0.29] |
| H4, the tool name (0.55) | 0.64 [0.59, 0.68] | 0.30 [0.25, 0.36] |
| H5, a gift card means failure (0.50) | 0.49 [0.46, 0.52] | 0.46 [0.42, 0.50] |

A tracking AUC is the AUC of the model's score between the calls a heuristic calls successes and those it calls failures; it says how closely the model's ranking follows the rule. Both models follow the status rule (H1) more closely than the truth: they separate status-violating calls from everything else almost perfectly, and separate unavailable-variant calls from available ones less well. On yes/no, Qwen's tracking AUC for H1 is 1.00: its yes/no answer is close to "does the tool's status rule pass", with everything else secondary. Neither model uses the word "delivered" as a failure cue. Llama in fact scores calls on delivered orders higher, through a preference for the exchange tool, and Qwen has a mild preference against exchange. Gift-card payment is ignored.

The status 2 × 2 regression fixed in the plan, on the status and unavailable classes, puts these together (neutral first token, in nats, with the added terms for the status class's shorter history and gift-card payment):

| Term | Qwen | Llama |
|---|---|---|
| The tool's status rule passes for the order's shown status | +4.95 [4.59, 5.34] | +2.92 [2.81, 3.04] |
| The order is delivered | +0.76 [0.31, 1.22] | +0.50 [0.37, 0.63] |
| The tool is exchange | −2.11 [−2.45, −1.77] | +1.50 [1.40, 1.59] |
| The new variant is shown unavailable | −7.90 [−8.44, −7.39] | −1.02 [−1.11, −0.93] |
| Status-class history (no product records) | −7.81 [−8.32, −7.31] | −0.43 [−0.56, −0.31] |

Within the status class alone the status-rule term is +4.92 [4.55, 5.30] for Qwen and +2.97 [2.86, 3.09] for Llama, against a delivered term of +0.56 and +0.76. By the reading fixed in advance, both models track order status against the tool, not the word. That matters for the design's caveat B4, which said the status class carries "no discrimination evidence" because its witness is not determined. The witness is not determined, but both models score the call that breaks the visible status rule far below the call on the same order that keeps it. So the status-class effects are effects in a class where the models do use the status record.

Qwen's large "status-class history" penalty means that, without product records on screen, it doubts even status-consistent calls (its first token gives 67% of status witnesses success, against 99% of unavailable-class witnesses). H6 went the unexpected way: among unavailable-variant failing calls, the larger the share of the product's variants shown unavailable, the *higher* both models score the call (Spearman +0.33 for Qwen and +0.19 for Llama). This is not the "crowded variants" heuristic, which predicts the reverse, and it is left unexplained.

## 5. Readout agreement

| Primary, aligned − swapped | Qwen fail | Qwen witness | Llama fail | Llama witness |
|---|---|---|---|---|
| First token | +3.98 | +9.45 | +0.47 | +0.38 |
| Yes/no | +7.68 [7.25, 8.13] | +12.65 | −0.02 [−0.043, −0.002] | −0.04 |
| Yes/no minus first token | +3.70 [3.43, 3.97] | +3.20 | −0.49 [−0.55, −0.43] | −0.43 |
| Full reply | +1.88 [1.18, 2.55] | +8.25 | +2.27 [2.02, 2.49] | +3.27 |
| … of which the success reply's tail | +1.13 [0.51, 1.76] | +0.39 | +1.71 [1.48, 1.93] | +1.47 |
| … of which the failure reply's tail (enters negatively) | +3.23 [2.90, 3.57] | +1.58 | −0.08 [−0.14, −0.02] | −1.42 |
| Per-token full reply | −0.06 [−0.09, −0.03] | +0.68 | +0.015 [0.009, 0.020] | +0.16 |
| Spearman, item effects first token vs yes/no | 0.73 | 0.71 | 0.61 | 0.37 |

*(Corrected 2026-11-23: Llama's yes/no interval in this table was printed as [−0.04, 0.00]; this note's own metrics (`tau_dx_metrics.json`, `q1_recompute.llama3.1-8b-fewshot.yn_L.aligned_minus_swapped`) and `tau_metrics.json` both give [−0.043, −0.002], which excludes zero. The reading of Llama's yes/no as an insensitive readout is unchanged.)*

The full reply's score is its first token plus the rest of the success reply minus the rest of the failure reply. **Llama's** full-reply effect (+2.27, larger than Qwen's) is mostly the success reply's tail, +1.71: the success JSON repeats the order number and product name that the aligned line contains, so it is a copying effect, not a belief about the outcome. Its per-token form is +0.015. **Qwen's** full reply is smaller than its first token because, under the aligned line, the specific error text also becomes more probable (+3.23 nats), most of all in the unavailable class (+5.16), where the error names an item of the product the aligned line names. The design's choice to interpret only the first token, with the per-token full reply as the secondary, is borne out. The results note's table puts the raw full-reply numbers side by side, which invites the wrong comparison (Llama larger than Qwen).

**Why Qwen's yes/no effect is so large.** Following the TextCraft robustness note (section 9), each call's yes/no score under each line was regressed on its first-token score, with one intercept per line. The intercept differences are yes/no shifts that the change in Qwen's own predicted output does not explain. Within class (pooling classes mixes their levels):

| Qwen, failing calls, at a matched first-token score | Status class | Unavailable class |
|---|---|---|
| Slope (yes/no nats per first-token nat) | 0.46 [0.42, 0.51] | 0.56 [0.53, 0.60] |
| Aligned − neutral | +5.75 [5.36, 6.15] | +3.18 [2.90, 3.46] |
| Swapped − neutral | +2.48 [2.17, 2.80] | −4.53 [−4.99, −4.10] |
| Aligned − swapped | +3.27 [2.98, 3.57] | +7.71 [7.17, 8.30] |

In the unavailable class the yes/no answer adds a lift under the aligned line and a penalty under the swapped line that the predicted output does not account for. The status class shows only a lift, from Qwen's extremely negative neutral answers there (it says No to every status call). The dissociation is the plainest sign. Among failing calls whose first token confidently predicts a JSON success (a score of at least +3), Qwen answers No for 17% under aligned, 67% under neutral and 93% under swapped. Among such working calls the shares are 32%, 51% and 91%. So with the question anchored to the output's shape, as amendment C1 intended, Qwen still answers "will the backend execute this call" partly as "is this call what the user wants". C1 removed the word "succeed" but not the reading. Qwen's yes/no is also pessimistic overall (51% No for working calls it predicts JSON for, under neutral), so the readout mixes a backend prediction, a general reluctance and a task judgement.

**Why Llama's yes/no is null.** Not because its answers resist the task, but because they barely respond to anything. The item effects' SD is 0.31 nats on yes/no against 1.16 on the first token, the neutral-line scores' SD 0.60 against 1.67, and its discrimination on yes/no is 0.30 nats against 1.10. Its yes/no moves 0.29 nats per first-token nat, so the first-token effect of 0.47 would carry over as about +0.14. At a matched first-token score, though, both task lines lower Yes relative to neutral (aligned −0.37, swapped −0.21), leaving aligned minus swapped at −0.16 [−0.18, −0.14], which cancels the carried-over +0.14. Llama's yes/no null is a compressed readout plus a small opposite task term. It is no evidence that Llama's belief is unaffected.

## 6. Practical size

| Share predicted to succeed, primary (aligned / swapped / neutral) | Qwen first token | Qwen yes/no | Llama first token | Llama yes/no |
|---|---|---|---|---|
| Failing calls, all | 45% / 25% / 38% | 31% / 1% / 7% | 5% / 6% / 4% | 11% / 13% / 20% |
| Failing calls, unavailable class | 74% / 38% / 60% | 57% / 1% / 10% | 2% / 4% / 3% | 16% / 22% / 34% |
| Working calls, all | 98% / 68% / 83% | 60% / 4% / 32% | 21% / 15% / 21% | 31% / 35% / 44% |
| Working calls, status class | 96% / 46% / 67% | 25% / 0% / 0% | 14% / 5% / 18% | 3% / 4% / 15% |

| Flips between swapped and aligned | Qwen first token | Qwen yes/no | Llama first token | Llama yes/no |
|---|---|---|---|---|
| Failing calls, toward success | 21% [18, 24] | 31% [28, 33] | 1% | 1% |
| Failing calls, toward failure | 2% | 0% | 2% | 3% |
| Working calls, toward success | 28% | 56% | 7% | 1% |
| Probability shift, failing calls (points) | +19.6 [17.2, 22.1] | +29.9 | +0.7 [0.1, 1.2] | −0.8 |

For Qwen the task line changes decisions on a large scale. Relative to neutral, the aligned line turns 10% of failing calls into predicted successes on the first token and 24% on yes/no. In the unavailable class, 28% of the failing calls that Qwen's yes/no confidently and correctly rejected under neutral are answered Yes under aligned. The swapped line does the converse to working calls: 13% of the working calls whose first token was confidently right under neutral become predicted failures, and on yes/no 91% of the unavailable-class working calls it confidently accepted under neutral become No (only 22 such calls, because Qwen's neutral yes/no rarely accepts with confidence). Among the 171 failing calls Qwen's first token rejects with at least 99% confidence under neutral, only 3.5% become predicted successes under aligned, so, as in TextCraft, the decision-level damage concentrates where Qwen was unsure. Unlike TextCraft, Qwen is unsure (or wrong) on most of these items to begin with.

For Llama the decisions barely move: under 2% flips in either direction on either readout, and a probability shift under one point. Its first token predicts failure for most calls whatever the line, so there is little to flip.

## 7. Prompt arms

Both exploratory arms use the same 500 primary items (status 240, unavailable 235). The "matched" arm drops the statement that the backend sees only the call and the database and does not see the user's task, and makes the few-shot examples own-order. The "rules-only" arm drops the agent-facing policy and keeps the rules list.

| First token, 500 items | Qwen main | Qwen matched | Qwen rules-only | Llama main | Llama matched | Llama rules-only |
|---|---|---|---|---|---|---|
| Failing call, aligned − swapped | +4.01 | +4.46 | +4.66 | +0.47 | +0.82 | +0.38 |
| Failing call, aligned − neutral | +1.45 | +1.37 | +2.76 | +0.92 | +1.06 | +1.08 |
| Failing call, swapped − neutral | −2.56 | −3.09 | −1.91 | +0.45 | +0.24 | +0.71 |
| Witness, aligned − swapped | +9.50 | +11.00 | (not scored) | +0.36 | +1.00 | (not scored) |
| Failing call, status / unavailable class | +2.16 / +5.42 | +2.07 / +6.37 | +2.68 / +6.08 | +1.25 / −0.36 | +1.72 / −0.23 | +1.00 / −0.31 |
| Level change against main: aligned / swapped / neutral | | −0.69 / −1.14 / −0.61 | +1.60 / +0.94 / +0.29 | | +0.00 / −0.35 / −0.14 | +0.55 / +0.65 / +0.39 |

**The statement that the backend does not see the task** works mainly on the swapped line. Without it both models predict more errors for calls on an order the user did not ask about (Qwen −1.14, Llama −0.35 nats against the main prompt), and the witness drops even more under the swapped line (Qwen's witness swapped − neutral goes from −3.02 to −4.64). That is the authorization belief the statement was written to suppress, and it is suppressed only partly, since Qwen's swapped penalty is −2.56 with the statement present. The class pattern is unchanged (Llama's unavailable class stays reversed), and the discrimination contrast hardly moves (Qwen 7.73 against 7.01, Llama 1.10 against 1.21). For Llama the matched arm turns failing minus witness negative (−0.18), so its small failure-specific edge in the main arm depends on the statement.

**The agent policy** makes both models more pessimistic overall. Without it Qwen's aligned line rises 1.6 nats and Llama's lines all rise about half a nat (and its yes/no lines about half a nat too, the only change in Llama's yes/no anywhere). For Qwen the policy suppresses the aligned lift specifically, perhaps through its rules about confirming with the user before acting. The effect itself is somewhat larger without it (4.66 against 4.01).

What the arms imply: the main prompt is a conservative test. Both debiasing elements reduce the effect (the statement on the swapped side, the policy on the aligned side), and neither removes it for Qwen. Qwen's effect is not an artefact of the prompt. Llama's size and its failure-specificity are prompt-dependent (0.38 to 0.82 across the three arms).

## 8. Cross-environment

ALFWorld and TextCraft values come from `notes/textcraft_vs_alfworld_metrics.json` (branch `xenv-compare`). Comparable here means the same model, the same kind of scoring (the first token of the environment's reply, two candidates; the yes/no log-odds) and the same three-line design. It does not mean equal size scales: the first-token words differ ("You" against "Nothing" in ALFWorld, "Craft" against "Could" in TextCraft, `{"` against "Error" here), the yes/no questions are worded differently in each environment (only τ-bench's is anchored to the output's shape), the neutral lines differ ("explore the room", "explore the crafting commands", "help with their account"), and the deciding fact sits in hidden state (ALFWorld), in inventory counts (TextCraft), or in a visible record of the named object (here).

| First token, failures | Qwen ALFWorld | Qwen TextCraft | Qwen τ-bench | Llama ALFWorld | Llama TextCraft | Llama τ-bench |
|---|---|---|---|---|---|---|
| Aligned − swapped | +2.52 | +0.97 | +3.98 | +0.68 | −0.16 | +0.47 |
| Aligned − neutral | +3.52 | +0.43 | +1.42 | +0.59 | −0.06 | +0.92 |
| Swapped − neutral | +1.00 | −0.54 | −2.56 | −0.09 | +0.10 | +0.45 |

| Yes/no, failures | Qwen ALFWorld | Qwen TextCraft | Qwen τ-bench | Llama ALFWorld | Llama TextCraft | Llama τ-bench |
|---|---|---|---|---|---|---|
| Aligned − swapped | +2.80 | +2.92 | +7.68 | +2.21 | −0.15 | −0.02 |
| Aligned − neutral | +2.52 | −2.07 | +5.22 | +2.44 | −0.60 | −0.10 |
| Swapped − neutral | −0.28 | −5.00 | −2.46 | +0.23 | −0.45 | −0.08 |

| Working calls against failures, aligned − swapped | Qwen ALFWorld | Qwen TextCraft | Qwen τ-bench | Llama ALFWorld | Llama TextCraft | Llama τ-bench |
|---|---|---|---|---|---|---|
| Successes (two-candidate in ALFWorld and TextCraft; first token here) | +1.94 | +2.61 | +9.45 | +0.16 | +0.08 | +0.38 |
| Failures, same scale | +6.12 | +0.94 | +3.98 | +1.65 | −0.19 | +0.47 |
| Successes, yes/no | +2.44 | +9.60 | +12.65 | +0.98 | +0.22 | −0.04 |
| Failures, yes/no | +2.80 | +2.92 | +7.68 | +2.21 | −0.15 | −0.02 |

**The direction recurs for Qwen in all three.** On the first token, τ-bench's Qwen effect is the largest of the three environments. **The "wrong task penalizes" signature recurs on the first token**: 64% of Qwen's τ-bench effect is the swapped line falling below neutral, as in TextCraft (56%), while in ALFWorld the swapped line rises above neutral. On yes/no τ-bench looks like ALFWorld instead: the aligned lift dominates. So the decomposition depends on the readout, and "lift against penalty" is not a stable property of an environment. **Successes move more than failures** for Qwen in TextCraft and τ-bench on every readout. In ALFWorld the available comparison went the other way on the two-candidate scale and was about even on yes/no, but ALFWorld's successes are a separately built set and its two-candidate scale carries the success sentence's tail, so that contrast is the weakest of the three. A general task-consistency bias that also moves working actions is therefore Qwen's pattern in at least two environments, and the ALFWorld headline's "override of a failure" reading is the one most in need of the same control.

**Llama** shows a positive first-token effect in ALFWorld and τ-bench and none in TextCraft. Its yes/no tracks only ALFWorld, where (unlike here) its yes/no readout was sensitive (ALFWorld neutral yes/no AUC 0.97). Llama's τ-bench effect is request-driven (both write lines lift), a pattern ALFWorld's design could not show, since every ALFWorld task line is a request.

## 9. Natural (agent-produced) failures

The 28 natural items come from 23 users. Twenty-two are status failures, often on statuses that the agent's own earlier writes created ("exchange requested", "return requested") or on "processed" orders; four are refund-method returns, one an unavailable variant and one a gift-card exchange. Nineteen have a swapped line and only four are status-matched, so they are not a sample of the primary population.

What they can say is about competence. Under the neutral line Qwen's first token predicts success for 64% of these agent-made failing calls (mean score +5.0), and Llama's for 36%. On items such as an exchange of an order whose status is already "exchange requested", Qwen predicts success with near certainty. The agent that made these calls also expected them to work, so they are the hard cases, and Qwen would not catch most of them whatever the task line said. As a world model for checking an agent's proposals, Qwen is weakest exactly on natural errors.

What they cannot say is anything about the goal effect. Qwen's aligned − swapped is +1.78 [−0.53, 4.13] over 19 items and its aligned − neutral −1.26 [−2.67, 0.09], and Llama's are −0.11 [−0.57, 0.43] and +0.47 [0.19, 0.80]. These intervals include both zero and the generated set's values. The user had named the target order before the failing call in 7 of the 28 trajectories; splits at this size are noise. Qwen's yes/no aligned − swapped (+5.88 [3.84, 8.13]) is the only interval clear of zero, and it is the readout shown above to carry a task judgement.

## 10. GPU

No GPU run was needed. Every question above was answered from the existing scores, at the depth 28 natural items allow. The one open question the scores cannot settle is what Qwen would actually generate under each line, rather than the two-token readout. That would matter if the first token's `{"` sometimes opened something other than a success record. Its two-way mass is 0.99996 and the first-token and full-reply passes agree on every item, so this risk is small and was not judged worth a run.

## 11. What should change in tau_results.md

- **Qwen's specificity.** "the failing call moves no more than the witness (fail minus witness about 0)" is wrong as a description. Failing minus witness is −5.47 [−5.88, −5.05] in the metrics: the working call moves more than twice as much. The verdict category (task-consistency, not failure-specific) stands; the sentence should state the negative value and that a visible failure damps the effect rather than leaving it unchanged.
- **Llama's reading.** "The failing call moves more than the witness ... the ALFWorld effect in its stronger form" should be qualified. The failure-specific part is +0.09 nats (a fifth of the effect), positive in about half the items, confined to one status cell and the 36 returns, reversed in the unavailable class (−0.27) and in the matched arm (−0.18). Llama's effect is +1.23 on status failures and −0.33 on unavailable-variant failures, so its pooled size depends on the class mix. Its median item effect is +0.25.
- **The status class's discrimination caveat** can be softened in a post hoc remark: both models clearly score status-violating calls below status-consistent calls on the same order (+4.9 and +3.0 nats), so the status class is not a class without evidence of record use.
- **Yes/no.** "anchored to the backend's reply ... so it is not subject to the task-success reading found in TextCraft" is contradicted by the data. Qwen answers No for 93% of failing calls (91% of working calls) whose first token confidently predicts JSON when the task names another order, against 67% (51%) under neutral. About 3.9 nats of Qwen's 7.68 is beyond its own predicted output. Llama's yes/no null should be described as an insensitive readout (0.29 nats per first-token nat), not as a null effect.
- **Full reply.** The table should either drop the raw full-reply row or note that Llama's +2.26 is mostly the success JSON copying the order number and product from the aligned line (+1.71), and that the per-token form is +0.015 (Llama) and −0.06 (Qwen).
- **Practical size** deserves a line. Qwen's first token predicts success for 38% of failing calls already under neutral (60% of the unavailable class), and the aligned line raises its yes/no Yes rate on unavailable-variant failures from 10% to 57%.

## 12. Implications for the paper

τ-bench supports a statement about Qwen3-8B: a task line that names an object raises the predicted success of calls on that object, by about 4 nats for calls the visible record shows will fail and about 9 for calls that work, and lowers it for calls on objects the task does not name. The effect is broad, not driven by a few users, survives both prompt variants, and changes Qwen's yes/no decisions on a large share of items. It is best described as a relevance or task-consistency bias that the model's own reading of a visible failure damps but does not cancel. It is not an override that singles out failures. With ALFWorld and TextCraft, that makes three environments where Qwen's effect is positive, and two where it is demonstrably larger on working actions. The paper's framing of the ALFWorld effect as "the goal overrides evidence of failure" should be weighed against this, since ALFWorld lacks a matched working-action control.

For Llama-3.1-8B the paper should report a small, request-driven effect whose sign depends on the failure class. It should not report it as a failure-specific override. Its yes/no readout is too compressed in τ-bench to carry a goal effect either way.

On method, two lessons generalize. First, a pooled effect over failure classes can hide opposite class effects; a class-level table belongs beside every pooled verdict. Second, anchoring the yes/no question to the output format did not stop Qwen from answering it partly as a task judgement. Any yes/no readout should be checked against the model's own predicted output (the dissociation table of section 5) before it is read as a belief about the environment.

## Limits

Everything here was examined after the confirmatory result was known, with many cuts and no correction for multiple comparisons. The class and cell results are large relative to their intervals and recur across readouts, but single cut-level intervals should be read as descriptive. The witness differs from the failing call in its arguments (a different tool in the status class, a different variant in the unavailable class), so failing minus witness compares two calls, not one call in two states. The unmatched-pool binding analysis uses the secondary pool, whose items differ from the primary ones in more than the comparison order's status. The neutral line is a vague request, not an absence of task, so "relative to neutral" decompositions depend on that line. The natural set is too small for any effect estimate. The cross-environment comparison sets side by side numbers from populations built differently in each environment.

## Reproducing

```
python scripts/tau_dx.py --scores <copy of runs/tau_score> --build <copy of runs/tau_build> \
    --natural-mentions <natural_mentions.json> --reference notes/tau_metrics.json \
    --xenv <notes/textcraft_vs_alfworld_metrics.json from origin/xenv-compare> --out notes/tau_dx_metrics.json
```

The natural-mention flags come from running the small script described in "Data and method" on ICE against the released trajectories (`/storage/ice1/2/1/gmat3/tau2_results`), which prints one boolean record per natural item. Runtime is about 15 minutes on one CPU, most of it in the competence bootstrap. The script needs numpy and scipy.
