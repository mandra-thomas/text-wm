# τ-bench retail, wave-A model families: results (2026-11-23)

The confirmatory part of this extension is two per-model tests: one for Mistral-7B-Instruct-v0.3 and one for Llama-3.2-3B-Instruct, Holm-adjusted together. Everything else is descriptive. The design and the analysis were frozen before any score of these models at the annotated tag `tau-families-design-frozen-2026-11-23` (commit `a078a8d`, branch `tau-families`). The design note, [tau_families_design.md](tau_families_design.md), comes onto `main` unchanged with this note, together with its review ([tau_families_review1.md](tau_families_review1.md)), its tokenization report ([tau_families_token_check.json](tau_families_token_check.json)), its mutation-testing record (`testing.md`) and its deviations entry. The code (`wf/tau/families.py`), the configs and the tests stay at the tag and on the branch. The smoke run, the eight scoring shards, the placement check and the analysis all ran at that commit, in a clean checkout.

Every number below is quoted from [tau_families_metrics.json](tau_families_metrics.json), a byte-identical copy of the analysis run's `metrics.json` (sha256 `7c9af2d59d2bc278…`). Nothing is recomputed and no verdict is re-derived. The only arithmetic added here is a few ratios of quoted numbers, each given in words (for example "four times"), and the ratio to the ALFWorld effect in section 8, which is labelled there. Paragraphs marked **exploratory reading** interpret the prespecified numbers and decide nothing.

## Summary

The extension asks whether the τ-bench goal effect, which is that a stated task raises a model's predicted success for a backend call the visible record shows will fail, appears in two more instruct models. One is from another developer (Mistral 7B v0.3) and one is from Meta's next release (Llama 3.2 3B). It also asks whether the shift is specific to failing calls or shared with working calls, as it is in both reference models (Qwen3-8B and Llama-3.1-8B).

**Llama 3.2 3B: "replicates (ALFWorld direction)".** Its first-token effect is +0.99 nats [0.90, 1.08] (Holm p 0.002), and every gate passes: it tells working from failing calls, it reads the task line, and it passes the families competence rule on the first-token readout. The working call on the same order (the witness) moves slightly more, +1.07 [0.98, 1.16], so failing minus witness is −0.08 [−0.12, −0.03]. By the extension's pre-stated rule its label is "a shift shared by working and failing calls (working calls move more)", which is Qwen3-8B's label. *(revised 2026-11-25)* The suffix is numerically marginal: the interval's upper end (−0.03) is within the largest environment shifts measured for Llama-family models (`numerical_sensitivity.md` §5.7). The effect is broad: 82% of items move toward success [79%, 84%]. It is positive in both main failure classes, so the class-mix reading does not apply. Its yes/no readout fails the competence rule, because it answers No to nearly every call, working or failing, so no yes/no result is read for it.

**Mistral 7B v0.3: "not interpretable (fails the families competence rule on the first-token readout)".** Under the neutral task its first token barely separates calls that will work from calls that will fail. The AUC (the chance that a random working call outranks a random failing call) is 0.58 against the rule's 0.70. The τ-bench discrimination check also fails: 0.36 nats against 1 nat. Its estimate, +0.23 nats [0.18, 0.27], is reported but not read. Without the competence gate it would have been "small effect, direction positive". The exploratory placement check found Mistral's effect sensitive to where its chat template puts the system text: moving the system text to the first turn more than doubles it on 300 items (+0.53 against +0.23; difference +0.30 [0.22, 0.38]). By the pre-stated rule, Mistral's τ-bench result is therefore "format-dependent". That changes no verdict.

**Gemma 2 9B was not run.** The decision was made before any data: the fixed part of every prompt is 9,575 Gemma tokens, more than Gemma 2's 8,192-token trained context. It is reported as not run, never as a null.

| | Mistral 7B v0.3 | Llama 3.2 3B |
|---|---|---|
| First-token effect, aligned − swapped (nats) | +0.229 [0.184, 0.273] | +0.991 [0.900, 1.084] |
| Holm-adjusted p (family of two) | 0.002 | 0.002 |
| Competence, first token (AUC on the 447 determined items; rule ≥ 0.70) | fails (0.58) | passes (0.78) |
| Discrimination, first token (nats; rule ≥ 1, interval above 0) | fails, 0.36 [0.32, 0.41] | holds, 1.55 [1.43, 1.67] |
| Manipulation check (nats; rule ≥ 1, interval above 0) | passes, 3.76 [3.66, 3.86] | passes, 1.17 [1.11, 1.23] |
| Verdict | not interpretable (fails the families competence rule on the first-token readout) | replicates (ALFWorld direction) |
| Witness effect (working call, same order) | +0.251 [0.203, 0.299] | +1.070 [0.980, 1.159] |
| Failing minus witness | −0.023 [−0.044, −0.002] | −0.079 [−0.122, −0.034] |
| Cross-model label | none (left out: not interpretable) | a shift shared by working and failing calls (working calls move more) |
| Item share, first token (null 0.5) | 0.655 [0.619, 0.691] (not read) | 0.817 [0.788, 0.843] |
| Competence, yes/no | fails (says Yes to everything) | fails (says No to almost everything) |
| Placement check (exploratory) | format-dependent: +0.53 first turn against +0.23 native | not applicable |

## 1. The question

A **world model** here is a language model asked to predict what τ-bench's retail backend replies to an agent's tool call. The backend is a customer-service database of users, orders and products. The agent calls tools such as `cancel_pending_order` or `exchange_delivered_order_items`. The prompt contains the backend's description, its rules and tool schemas, four worked examples (**few-shot examples**), the history of earlier calls with their outputs, a one-line description of what the user wants (the **task line**), and the next call.

Every test item is a **failing call**: a write that the backend rejects because a precondition fails. The deciding fact is printed in an earlier tool output in the history. It may be the order's status (the `status` class), a requested variant shown as unavailable (`unavailable`), the original payment method (`wrong_return_pm`) or a gift card's balance (`gc_balance`). The real backend never reads the task, so any difference between task lines in a model's prediction is an error. That error is the **goal effect**.

The confirmatory τ-bench study (`notes/tau_results.md`) compared three task lines. The **aligned** line asks for exactly the change the failing call makes, to the call's own order and product. The **swapped** line asks for the same kind of change to another order of the same user, and its product. The **neutral** line says only "The user wants help with their account." The study found that the aligned line raises predicted success relative to the swapped line in both reference models: Qwen3-8B by +3.98 nats ("replicates") and Llama-3.1-8B by +0.47 ("small effect, positive").

The exploratory follow-up (`notes/tau_dx.md`) then found that in both models a working call on the same order moves too. In Qwen3-8B it moves more than twice as much. In Llama-3.1-8B it moves about 80% as much, and the pooled effect averages a positive `status` class and a negative `unavailable` class. So on closer reading neither reference model shows a shift specific to failing calls.

This extension asks three descriptive questions of two more instruct models from the families study's wave A:

1. Does the goal effect appear?
2. Is it shared by working and failing calls, or specific to failing calls by a rule fixed before data?
3. Does it hold in both main failure classes?

One model from another developer and one from Meta's next release can describe this. They cannot separate developer, size, tokenizer, chat format and post-training, so every cross-model statement is descriptive.

## 2. What was run

### 2.1 Models and their chat formats

**Mistral-7B-Instruct-v0.3** (revision `c170c708c41d`) is used with its template's own placement. The template writes the system text (the instruction, the backend's rules, the retail policy and the tool schemas, about 5,750 tokens) into the *last* user turn. So the system text sits after the four few-shot examples and immediately before the query's task line, where every other model has it at the start of the prompt. The template writes a reply after a space, so the scored tokens are `▁{"`, `▁Error`, `▁Yes` and `▁No` (each a single token).

**Llama-3.2-3B-Instruct** (revision `0cb88a4f764b`) uses the Llama 3 header format with the date fixed. Apart from the model, its prompts are byte-identical to Llama-3.1-8B's.

**Gemma-2-9B-it** was excluded before any data (design §2.3). The fixed part of every prompt, the system message plus the four few-shot examples, is 9,575 Gemma tokens, and Gemma 2 was trained on 8,192 positions without position scaling. Every one of the 13,696 scored prompts is at or above that length (median 11,339). Three alternatives were considered and rejected before data: truncating the history (it holds the deciding record), a compact prompt (a different task, not comparable with the four models' main prompt) and ignoring the limit. Gemma 2 is reported as "not run: the τ-bench prompt does not fit its 8,192-token context", never as a null result.

### 2.2 Items, task lines and calls

The items are the 856 **primary items** of the τ-bench build at its pins (175 users; 409 `status`, 405 `unavailable`, 36 `wrong_return_pm`, 6 `gc_balance`). The prompts are the τ-bench main arm, unchanged apart from what each chat template imposes. Each item is scored for two calls:

- **The failing call**, which the backend rejects.
- **The witness**: a single write on the same order that the backend accepts and that would achieve the aligned task (for example, the same exchange to an available variant).

Two further task lines form the **mention pair**. Each asks a question about a product in an order and requests no change: **mention-aligned** names the target order and **mention-swapped** names the comparison order.

### 2.3 The readouts

Every backend reply begins either with `{"` (a JSON success) or with `Error`. The **first-token readout** is therefore the log-odds of these two openings at the first position of the predicted reply,

$$L_1 = \log P(\texttt{\{"}) - \log P(\texttt{Error}),$$

measured in **nats**, or natural-log units: +1 nat multiplies the odds of success by about 2.7. The goal effect on an item is $L_1$ under the aligned task minus $L_1$ under the swapped task. Its mean over the 856 primary items is the primary quantity,

$$\overline{\Delta L_1} = \frac{1}{856} \sum_{i} \left[ L_1(i, \text{aligned}) - L_1(i, \text{swapped}) \right].$$

A positive value means the model predicts success more readily when the task asks for the call it is judging.

For a size on a common probability scale, the design also reports the **practical size**: the mean over items of the success probability renormalized over the two openings,

$$\sigma(L_1) = \frac{P(\texttt{\{"})}{P(\texttt{\{"}) + P(\texttt{Error})},$$

under the aligned task minus under the swapped task. The **flip shares** count the items whose predicted opening (the sign of $L_1$) changes between the swapped and the aligned task, toward success or toward failure.

The **yes/no readout** appends the reply-anchored question "will the backend execute this call without returning an error? Answer Yes if its output is JSON and No if its output is a line starting with "Error: "" and scores the log-odds of Yes against No. It is the families study's format and is secondary here. In Qwen3-8B this question was found to carry a judgement about the task beyond the predicted reply (`notes/tau_dx.md`).

The **manipulation question** (the "target" readout in the metrics file) asks whether the call is on the order the user asked about, and scores Yes against No. Its aligned-minus-swapped difference shows whether the model reads the task line at all.

### 2.4 The protocol

Per item and model there are 16 single forward passes, 13,696 per model:

- The failing call on the first token and on yes/no under aligned, swapped and neutral.
- The failing call on the first token under mention-aligned and mention-swapped.
- The failing call on the manipulation question under aligned and swapped.
- The witness on the first token and on yes/no under aligned, swapped and neutral.

The full-reply readout, the exploratory prompt arms, the items outside the primary population and the natural set were dropped to save GPU time (design §4.2). Every input of the per-model verdict and of its reading table was kept.

## 3. The rules, fixed before data

### 3.1 The per-model verdict

This is the τ-bench design's rule, unchanged. The 95% interval is a **user-clustered percentile bootstrap**: users, not items, are resampled 2,000 times, because items from one user share a history. The p is a two-sided **sign-flip permutation** test at the user level (1,000 permutations), which flips the sign of each user's differences at random to build the null distribution. With $p_{\text{Holm}}$ the p after **Holm's correction** over the family of the two scored models (a step-down adjustment that keeps the chance of any false positive in the family at or below 0.01), the verdict is "replicates (ALFWorld direction)" when

$$\overline{\Delta L_1} \ge 0.5 \text{ nats}, \quad \text{the 95\% interval excludes } 0, \quad p_{\text{Holm}} < 0.01.$$

Under the same conditions it is "reversed" at −0.5 or below. It is "small effect" (with its direction) when only the last two conditions hold, "no effect larger than 0.5 nats" when the interval lies inside −0.5 to 0.5, and "inconclusive" otherwise.

Two τ-bench gates apply. A null verdict needs the **manipulation check** (aligned minus swapped on the manipulation question at least 1 nat with its interval above 0) and the **discrimination check**. A reversal needs discrimination. The discrimination check uses the 447 items whose witness the text-only checker determines from the history. On those items, under the neutral task, the witness's $L_1$ minus the failing call's must be at least 1 nat with its interval above 0. Every verdict is then read through the τ-bench outcome table, which uses three controls on the first token: the witness's own aligned-minus-swapped effect, failing minus witness (the failing call's effect minus the witness's, item by item), and write minus mention. The last compares the goal effect under the write requests with the same contrast under the two bare mentions,

$$\text{write minus mention} = \left[ L_1(\text{aligned}) - L_1(\text{swapped}) \right] - \left[ L_1(\text{mention-aligned}) - L_1(\text{mention-swapped}) \right],$$

averaged over items, so it measures how much of the order-specific shift needs the request to act rather than the naming of the order alone.

### 3.2 The third gate: the families competence rule

A model that fails the families competence rule on the first-token readout gets the verdict "not interpretable (fails the families competence rule)", whatever its estimate. The rule is computed on the **discrimination population** $D$: the 447 items whose witness is a determined success (405 `unavailable`, 36 `wrong_return_pm`, 6 `gc_balance`; no `status` witness is determined by its history). On $D$, under the neutral task, write $s_i$ for item $i$'s witness score and $f_i$ for its failing call's score. The rule needs four conditions. The first is a rank separation,

$$\mathrm{AUC} = \frac{1}{|D|^2} \sum_{i \in D} \sum_{j \in D} \left( \mathbf{1}[s_i > f_j] + \tfrac{1}{2}\,\mathbf{1}[s_i = f_j] \right) \ge 0.70,$$

which is the probability that a randomly chosen working call scores above a randomly chosen failing call: 0.5 is chance, 1 is perfect.

The other three conditions are coverage bounds:

- The share of failing calls the model calls successes (score above 0) is at most 0.95.
- The share of working calls it calls successes is at least 0.05.
- The median two-way mass, which is the probability the model puts on the two scored tokens together, is at least 0.5.

The rule is also computed on the yes/no readout, where it gates only the yes/no results. It is reported for the reference models descriptively.

### 3.3 Cross-model description

For each model with a positive verdict, holding discrimination and passing competence, the extension names the row of the outcome table by its mechanical criterion ("fail minus witness above 0" or "not above 0"). The label "failure-specific" is used only if all three of these hold:

1. Failing minus witness has its 95% interval above 0.
2. The witness's effect is less than half the failing call's.
3. Failing minus witness is not below 0 in either main class.

Otherwise the model is "a shift shared by working and failing calls", with "(working calls move more)" added when failing minus witness has its interval below 0.

A **class-mix** qualifier applies when the `status` and `unavailable` effects have intervals on opposite sides of 0. A **minority** qualifier applies when the item share's interval includes or lies below one half.

The **item share** is the families study's scale-free measure. It is the share of the primary items $F$ whose first-token score is higher under the aligned than under the swapped task,

$$\text{share} = \frac{1}{|F|} \sum_{i \in F} \mathbf{1}\left[L_1(i, \text{aligned}) > L_1(i, \text{swapped})\right],$$

with null value one half and ties counted as not moved.

Comparisons between models use a paired user-clustered bootstrap on the items both models have. Two quantities are compared:

- **The item-share difference** "differs" when its 95% interval excludes 0. It "matches" when its 90% interval lies inside ±0.05; the design stated before data that this is very probably unreachable at this sample size.
- **The difference in mean first-token effect, in nats,** "differs" when its interval excludes 0 and it is at least 0.3 nats in size.

A model that is not interpretable is left out of every cross-model reading, and its comparison entries in the metrics file read "not read: <why>".

## 4. Mistral 7B v0.3

### 4.1 Why it fails the competence rule

On the discrimination population under the neutral task, Mistral's first token hardly separates working from failing calls. Its AUC is 0.582, against the rule's 0.70: a random working call outranks a random failing call only 58% of the time, against 50% by chance. The coverage conditions pass:

- It predicts JSON for 53.2% of the determined failures (at most 95% allowed).
- It predicts JSON for 62.9% of the determined successes (at least 5% required).
- Its median two-way mass is 0.96 (at least 0.5 required).

So the failure is not a format problem. The model puts almost all its probability on the two openings. It simply predicts success for about half of all calls, and the visible record barely changes which half.

The τ-bench discrimination check agrees. The witness's neutral score exceeds the failing call's by only 0.36 nats [0.32, 0.41] on the 447 items, and by 0.29 [0.26, 0.33] in the `unavailable` class alone, far below the 1-nat threshold. As a descriptive extra, the metrics file computes the same competence numbers on all 856 items, treating every witness as a success. There Mistral's AUC is 0.699, still below 0.70. The verdict is therefore "not interpretable (fails the families competence rule on the first-token readout)". The pre-written meaning is: "Not interpretable: on these prompts the model is not shown to tell working calls from failing ones, or answers one way to everything; no statement about goal sensitivity."

On the yes/no readout Mistral also fails competence, in a different way: it answers one way to everything. It answers Yes to 100% of the determined failures and 100% of the determined successes under the neutral task (AUC 0.634). This repeats, in a stronger form, its ALFWorld tendency to answer Yes to failures (56% there).

### 4.2 Its estimates, reported and not read

These numbers carry no verdict and enter no cross-model reading. They are given because the design requires every estimate to be reported, and because they describe what the model's predictions do.

The first-token effect is +0.229 nats [0.184, 0.273]. The raw permutation p is at its floor, the Holm p is 0.002, and 65.5% of items [61.9%, 69.1%] move toward success, with a median item effect of +0.25 nats. Before the competence gate the τ-bench rule gives "small effect, direction positive" (stored as `verdict_before_competence`). Clustering by product (56 products) gives a similar interval, [0.179, 0.285]. Without the items whose tool class appears among the few-shot examples (814 items) the effect is +0.223 [0.179, 0.267].

In probability terms the effect is tiny. The mean shift in $\sigma(L_1)$, the success probability renormalized over the two openings, is 0.026 [0.020, 0.032]. Only 1.5% of items change their predicted opening from Error to JSON between the swapped and the aligned line, and none the other way.

The neutral-anchored contrasts show where Mistral's movement is. Against the neutral line, both write requests raise the failing call's score by about 2 nats: aligned minus neutral is +2.11 [2.07, 2.16] and swapped minus neutral is +1.89 [1.83, 1.95]. Which order the request names adds only the 0.23 above. The mention pair goes the other way from the request: mention-aligned minus neutral is −0.98 [−1.04, −0.92]. Mention-aligned minus mention-swapped is +0.30 [0.22, 0.38], so write minus mention is −0.07 [−0.13, −0.01]. The witness moves as much as the failing call: +0.25 [0.20, 0.30] aligned minus swapped, failing minus witness −0.023 [−0.044, −0.002]. By class, the failing call moves +0.17 [0.12, 0.22] in `status` and +0.28 [0.22, 0.33] in `unavailable`.

**Exploratory reading.** Mistral's first token responds strongly to whether the task line requests a write, by about 2 nats, and weakly to which order it names. It treats working and failing calls almost alike. This fits a model that is not reading the deciding record on these prompts. The competence gate exists to keep such a model's goal effect from being read as a statement about goal sensitivity.

Mistral's yes/no readout moves +0.65 [0.62, 0.68] aligned minus swapped, with 93% of items moving, but at a level where every answer is Yes, so the movement is a change of confidence inside "Yes". The manipulation check passes comfortably (+3.76 [3.66, 3.86]): Mistral does read the task line and knows which order it names. What fails is the separation of working from failing calls, not the reading of the task line.

### 4.3 The placement check (exploratory)

Mistral's template moves the system text, about half the prompt, next to the task line. The design therefore fixed a small check before data. On 300 primary items (141 users, drawn with a fixed seed), the failing call's first token was scored under the aligned and swapped lines with the system text folded into the *first* user turn instead, which is Gemma's handling in wave A. The rule is: if the per-item difference (first turn minus native) has an interval excluding 0 and exceeds one fifth of the native effect on the same items in absolute size, Mistral's result is reported as depending on prompt format.

On the 300 items the native effect is +0.231 [0.166, 0.295], with 62% of items moving toward success. The first-turn effect is +0.529 [0.460, 0.596], with 84% moving. The difference is +0.298 [0.219, 0.379]. That interval excludes 0, and the difference is about 1.3 times the native effect, well above one fifth (0.046). The stored reading is "Mistral's τ result depends on prompt format: both numbers are reported in every comparison". It changes no verdict.

What this does not show: the placement run scored only the failing call under the aligned and swapped lines. It has no neutral line and no witness, so it cannot say whether Mistral would pass the competence rule with the system text in the first turn. Whether the native template's placement is what keeps Mistral from reading the record is an open question, not a finding.

## 5. Llama 3.2 3B

### 5.1 Verdict and gates

The first-token effect is +0.991 nats [0.900, 1.084], with Holm p 0.002. It meets all three conditions of "replicates (ALFWorld direction)".

- **Competence (first token), passed.** AUC 0.775. The model predicts JSON for 21.7% of determined failures and 59.1% of determined successes, and its median two-way mass is 0.97.
- **Discrimination, holds.** The witness's neutral score exceeds the failing call's by 1.55 nats [1.43, 1.67], and by 1.70 [1.59, 1.80] in the `unavailable` class.
- **Manipulation check, passed.** +1.17 [1.11, 1.23] on the manipulation question.

The descriptive all-items competence block puts Llama 3.2 3B's AUC at 0.649 when every witness, including the 409 `status` witnesses whose success the history does not determine, is treated as a success. That block is not the gate, since the rule's population is $D$. It is reported because it shows that the competence pass rests on the items whose witness is determined, 405 of 447 of them in the `unavailable` class, as the design noted it would (§7.3). On the `status` items the witness's success is not visible in the history, so a lower AUC when they are included is not by itself evidence of incompetence.

The τ-bench outcome table reads the witness as shifting toward success (CI above 0). Its specificity reading is "not specific to failing calls (fail minus witness about 0, negative or uncertain)", and its mention reading is "the request to act adds to naming the order (write minus mention above 0; an authorization reading predicts this too)". The stored reading is: "A general task-consistency bias, or a belief that the backend checks calls against the request: naming the order raises the success belief of calls on it whether they fail or not; reported as goal leakage, not as an override specific to failures."

### 5.2 Working and failing calls, and the two classes

The witness moves +1.070 [0.980, 1.159] aligned minus swapped, 108% of the failing call's effect. Failing minus witness is −0.079 [−0.122, −0.034]. By the rule of section 3.3 the row is "fail minus witness not above 0" and the label is "a shift shared by working and failing calls (working calls move more)".

| Class | Items | Failing call | Witness | Failing minus witness |
|---|---|---|---|---|
| `status` | 409 | +1.329 [1.213, 1.446] | +1.325 [1.207, 1.446] | +0.005 [−0.055, 0.067] |
| `unavailable` | 405 | +0.556 [0.469, 0.645] | +0.718 [0.646, 0.793] | −0.162 [−0.229, −0.096] |
| `wrong_return_pm` (descriptive) | 36 | +1.99 | not reported | not reported |
| `gc_balance` (descriptive) | 6 | +1.21 | not reported | not reported |

Both main classes are positive with intervals above 0, so the class-mix qualifier does not apply. Unlike Llama-3.1-8B, whose `unavailable` effect was negative (−0.33), Llama 3.2 3B's pooled effect does not depend on the mix of classes. The effect is larger in the `status` class, where the witness moves exactly as much as the failing call. In the `unavailable` class the witness moves more. The two small classes have estimate and n only, as fixed in the design.

### 5.3 Against the neutral line, and the mention pair

Against the neutral line, the aligned task raises the failing call's score by +2.01 [1.94, 2.08] and the swapped task by +1.02 [0.93, 1.11]. So a write request naming *another* order of the user already raises predicted success by about a nat, and naming the call's own order adds about another nat.

Anchored on neutral, failing minus witness is −0.10 [−0.14, −0.05] for aligned minus neutral and −0.02 [−0.07, 0.03] for swapped minus neutral. In other words, the swapped request lifts working and failing calls alike. The design notes that an authorization belief would instead lower swapped below neutral for both calls; that is not what happens here.

The mention pair shows that the request, not the naming, carries most of this. Mention-aligned minus neutral is −0.30 [−0.36, −0.25], so a bare question about the target order lowers predicted success slightly. Aligned minus mention-aligned is +2.31 [2.25, 2.37], which is the write request's effect over the bare question about the same order. Mention-aligned minus mention-swapped is +0.52 [0.45, 0.58], so write minus mention is +0.47 [0.41, 0.53].

**Exploratory reading.** Llama 3.2 3B's first token rises with any write request in the task line. It rises further when the request names the call's own order, and it rises for working calls as much as for failing ones. That is the "task-consistency" pattern of both reference models. Nothing here separates relevance from an authorization belief; the authorization experiment (`notes/tau_auth_results.md`) addressed that question for the reference models only.

### 5.4 Size in probability, item share and flips

The mean shift in the renormalized success probability $\sigma(L_1)$ is 0.183 [0.167, 0.201], about 18 percentage points. The item share is 0.817 [0.788, 0.843], with a median item effect of +0.75 nats and 1.5% exact ties, so the minority qualifier does not apply.

Between the swapped and the aligned line, 28.4% of items change their predicted opening from Error to JSON and 1.6% change the other way. This is a model whose scores sit near the decision boundary, so a 1-nat shift changes many answers. That fits the families finding that flip rates are set by where a model's scores start as much as by how far the task moves them.

**Sensitivity.** Clustering by product (56 products) gives +0.991 [0.901, 1.085]. Dropping the items whose tool class appears among the few-shot examples (814 items) gives +0.945 [0.862, 1.028]. Both keep the verdict category ("replicates (ALFWorld direction)", `same_category: true`).

### 5.5 The yes/no readout: not interpretable

Llama 3.2 3B fails the competence rule on yes/no. Under the neutral task it answers Yes to only 2.0% of the determined successes (at least 5% required) and 1.1% of the determined failures, and its AUC is 0.676 (at least 0.70 required). It answers No to nearly every call. Its yes/no results are therefore labelled "not interpretable on the yes/no readout", and every yes/no comparison involving it is marked "not read" in the metrics file.

For completeness, the descriptive numbers are as follows. Aligned minus swapped on yes/no is +0.33 [0.29, 0.37], with 68.6% of items moving. Aligned minus neutral is +0.42 [0.38, 0.46] and swapped minus neutral is +0.09 [0.04, 0.13]. Yes/no failing minus witness is −0.01 [−0.03, 0.01]. The task-judgement analysis is uninformative for this model. Among calls whose first token confidently predicts JSON (a first-token score of at least +3), the share answered No is 86% of 21 failing calls and 99% of 111 witnesses under the aligned line. That is what a model that answers No to almost everything would show, whatever it believes about the task.

## 6. The reference models, recomputed

The analysis recomputed Qwen3-8B and Llama-3.1-8B from the reference scoring run (`tau_score`, scored at `c737cd6`), using exactly the jobs of this protocol. Their recomputed primary estimates and intervals equal the published ones in `notes/tau_metrics.json`: Qwen3-8B +3.983 [3.607, 4.347] and Llama-3.1-8B +0.473 [0.406, 0.539], `all_equal: true`. Every reference record's item hash equals the current item's, with 0 missing, 0 stale and no problems.

Their published verdicts stand and are not re-tested. Both pass the families competence rule on both readouts, so nothing needs disclosing beside them:

- **Qwen3-8B, first token:** AUC 0.860, JSON for 63.3% of failures and 98.7% of successes.
- **Qwen3-8B, yes/no:** AUC 0.837.
- **Llama-3.1-8B, first token:** AUC 0.717, JSON for 7.2% of failures and 23.0% of successes.
- **Llama-3.1-8B, yes/no:** AUC 0.736.

Under the extension's cross-model label rule, Qwen3-8B is "a shift shared by working and failing calls (working calls move more)": its witness moves 2.37 times its failing call's effect. Llama-3.1-8B is "a shift shared by working and failing calls", with its witness at 81% and its row "fail minus witness above 0" (+0.089 [0.036, 0.138]). It also carries the class-mix qualifier, "the pooled effect depends on the class mix; reported per class", because its two class effects have intervals on opposite sides of 0.

## 7. Comparisons across models (descriptive)

**Nats are not one scale across models.** As `notes/cross_model_summary.md` §5 documents, models spread their scores over very different ranges, so a nat in one model is not a nat in another. In τ-bench, Qwen3-8B's first-token effect is 8 times Llama-3.1-8B's in nats. The design therefore reads every $\Delta L_1$ difference beside the item share, never alone. The metrics file's scale-free measures are the item share and, on a common probability scale, the mean shift in $\sigma(L_1)$. It holds no standardized effect (mean shift divided by the SD of item shifts) for τ-bench, so none is quoted here.

| Model | First-token effect (nats) | Item share, first token | Median item effect (nats) | Shift in success probability | Opening flips toward success | Label |
|---|---|---|---|---|---|---|
| Qwen3-8B (reference) | +3.98 [3.61, 4.35] | 0.778 [0.745, 0.808] | +3.25 | 0.196 [0.171, 0.221] | 20.9% | shared (working calls move more) |
| Llama-3.1-8B (reference) | +0.47 [0.41, 0.54] | 0.590 [0.563, 0.617] | +0.25 | 0.007 [0.001, 0.012] | 1.1% | shared; class mix |
| Llama 3.2 3B | +0.99 [0.90, 1.08] | 0.817 [0.788, 0.843] | +0.75 | 0.183 [0.167, 0.201] | 28.4% | shared (working calls move more) |
| Mistral 7B v0.3 (not interpretable; not read) | +0.23 [0.18, 0.27] | 0.655 [0.619, 0.691] | +0.25 | 0.026 [0.020, 0.032] | 1.5% | none |

In the label column, "shared" abbreviates "a shift shared by working and failing calls". Only Llama 3.2 3B's first-token comparisons are read. Mistral is left out of every cross-model reading because it fails competence, and every yes/no comparison is unread because neither scored model is interpretable on yes/no. The metrics file keeps the computed values of the unread comparisons as `reading_unread`. They are not quoted here as readings.

**Llama 3.2 3B against Llama-3.1-8B.** The item-share difference is +0.227 (95% [0.192, 0.262]): "differs". The $\Delta L_1$ difference is +0.518 nats [0.430, 0.603]: "differs". The 3B model moves more items, by more, than its larger predecessor. The pre-written Llama-line row ("Llama 3.2 3B has Llama-3.1-8B's label by the rule and their item shares match") is not met on either count. The labels differ: Llama-3.1-8B's is "a shift shared by working and failing calls" with the class-mix qualifier, while Llama 3.2 3B's adds "(working calls move more)" and has no class mix. And the item shares differ rather than match.

**Llama 3.2 3B against Qwen3-8B.** The item-share difference is +0.039 (95% [−0.002, 0.076]; 90% [0.006, 0.070]): "no difference detected". The 90% interval is not inside ±0.05, so it is not "matches", and this is never read as "the same". The $\Delta L_1$ difference is −2.99 nats [−3.36, −2.63]: "differs".

This pair shows the scale problem directly. In nats Qwen3-8B's effect is four times Llama 3.2 3B's. Yet the two models move about the same share of items (78% and 82%) and shift success probability by about the same amount (0.20 and 0.18). **Exploratory reading:** this suggests that Llama 3.2 3B's smaller effect in nats reflects a narrower spread of its scores rather than a smaller share of items moved. The metrics file holds no score SD, so this is not checked here.

**Pattern sentences.** None of the design's pre-written cross-model pattern rows applies (`patterns.readings` is empty):

- The rows about both scored models need both to be interpretable, and Mistral is left out.
- The Llama-line row needs matching item shares, and the shares differ.
- No interpretable scored model has a null, a reversal or class effects of opposite sign.

So the extension's cross-model statement is per model. Llama 3.2 3B, the one interpretable new model, shows the goal effect in the ALFWorld direction with Qwen3-8B's label, and no interpretable model tested on τ-bench (now three) shows a shift specific to failing calls by the pre-stated rule.

## 8. Predictions logged before data, against outcomes

The design logged predictions from the ALFWorld wave-A first-token effects and the reference models' ratios of τ-bench to ALFWorld effect.

- **Mistral 7B v0.3.** Predicted: a positive effect with the interval above 0 and the verdict "replicates", scaled to 0.86–2.0 nats. Competence was named as the main risk, with a subjective 0.7 probability of passing competence and both τ gates. Outcome: the risk materialized. The estimate, unread, is 0.23 nats, below the predicted range.
- **Llama 3.2 3B.** Predicted: about 0.48 nats, a positive effect with the interval above 0 and "small effect, positive" (probability of "replicates" 0.35). Also predicted: labelled "a shift shared by working and failing calls" (probability of "failure-specific" 0.15) and plausibly the class-mix reading (0.5). Outcome:
  - The direction and the label were as predicted.
  - The size was larger: 0.99 nats, "replicates".
  - The class-mix reading did not occur; both classes are positive.
  - On arithmetic added here (not in the metrics file), the τ-bench effect is about 1.4 times Llama 3.2 3B's ALFWorld first-token effect of 0.69 nats. The design's two reference ratios were 0.69 and 1.59.

## 9. What these results show and do not show

**They show** that the τ-bench goal effect appears in Llama 3.2 3B in the ALFWorld direction, at about 1 nat, over a broad majority of items, in both main failure classes, with every gate passed. They also show that the effect is again shared with working calls: naming the order in a write request raises the predicted success of a call on that order whether or not the record says it will fail. With Llama 3.2 3B, all three interpretable models scored on τ-bench (Qwen3-8B, Llama-3.1-8B, Llama 3.2 3B) shift in the same direction, and none meets the pre-stated rule for a failure-specific shift.

**They do not show** anything about Mistral's goal sensitivity. On these prompts Mistral does not tell working from failing calls, so its small positive estimate cannot be read as a goal effect or as an absence of one. Nor do they extend the τ-bench finding beyond Qwen and Meta. With Gemma 2 excluded and Mistral not interpretable, every interpretable model is from Qwen or Meta. The question the extension was meant to answer for another developer remains open.

They cannot say why Llama 3.2 3B moves a larger share of items than Llama-3.1-8B. Size, distillation and post-training all differ between them. Because the effect is shared with working calls, they also cannot separate a task-relevance bias from a belief that the backend checks requests; the authorization experiment tested that for the reference models only. Finally, no yes/no result is read for either new model, because both fail the yes/no competence rule in opposite directions: Mistral says Yes and Llama 3.2 3B says No to nearly everything.

**What would change these conclusions.** A Mistral run that passes competence, for example with the first-turn placement that more than doubled its effect, would make its result readable. That would be a new run under its own amendment, not a re-reading of this one. A model from a third developer with a long enough context (the families study's planned Gemma 4 wave) would test whether the direction holds outside Qwen and Meta.

## 10. Provenance

- **Design and analysis.** Frozen at the annotated tag `tau-families-design-frozen-2026-11-23`, commit `a078a8d122367a193d95da2baed7e7b2f9f51a06`. Every run below records that commit with a clean tree (`git_dirty: false`), in the ICE checkout `text-wm-fix44`.
- **Item rebuild, verification and token check** (CPU, job 7246048, 10 minutes). Status PASS. The three item pins were reproduced (items `2c8ffeac…`, natural items `9bd6c3f7…`, few-shot `231607f7…`), and `tau_verify` passed on 3,363 generated and 28 natural items. For both scored models: 0 problems of any kind on every scored prompt, and the 600 placement prompts pass. All 13,696 Gemma 2 prompts are at or above 8,192 tokens.
- **Smoke run** (`tau_family_score_smoke`, job 7246049, NVIDIA L40S, 21 items, both models) and **smoke analysis** (job 7247971). Read for format and timing only. Median two-way masses were 0.96–0.99 on every readout. The projected largest shard was 1.48 h for Mistral and 0.63 h for Llama 3.2 3B (limit 3 h), and the projected totals were 6.17 and 2.50 GPU-hours (flag at 6.5). The standard deviations of the per-item differences were 0.54 (Mistral) and 1.02 (Llama 3.2 3B) on the first token. Nothing was changed after the smoke run.
- **Scoring.** Every shard ran on an NVIDIA L40S, with a single job record per shard (no resume), 0 missing and 0 stale scores, and 13,696 records per model. The in-job readout reports confirm the scored token ids equal the template's reply tokens, with SDPA attention loaded. Mean seconds per pass were 1.43 (Mistral) and 0.61 (Llama 3.2 3B), for 5.4 and 2.3 GPU-hours of scoring. Median prompt lengths were 11,872 and 9,771 tokens (maximum 14,642 and 11,907).

| Run | Job | Items | Records | GPU | Wall time | sha256 of `scoring.jsonl` |
|---|---|---|---|---|---|---|
| Mistral, shard 0 of 4 | 7247978 | 220 | 3,520 | L40S | 1:27:45 | `f69975a9550a889a…` |
| Mistral, shard 1 of 4 | 7247979 | 210 | 3,360 | L40S | 1:22:25 | `70e9eec14571c44b…` |
| Mistral, shard 2 of 4 | 7247980 | 217 | 3,472 | L40S | 1:23:56 | `52c8cd08926ecd61…` |
| Mistral, shard 3 of 4 | 7247981 | 209 | 3,344 | L40S | 1:20:53 | `9722e09cfba3cc77…` |
| Llama 3.2 3B, shard 0 of 4 | 7247982 | 220 | 3,520 | L40S | 0:37:25 | `9342c805c8abe25b…` |
| Llama 3.2 3B, shard 1 of 4 | 7247983 | 210 | 3,360 | L40S | 0:34:21 | `5e3572a5318f2424…` |
| Llama 3.2 3B, shard 2 of 4 | 7247984 | 217 | 3,472 | L40S | 0:36:13 | `7232f3158d43720d…` |
| Llama 3.2 3B, shard 3 of 4 | 7247985 | 209 | 3,344 | L40S | 0:34:06 | `fcf499a39cd5314b…` |
| Mistral placement check | 7247986 | 300 | 600 | L40S | 0:15:40 | `bbc627ab35366c4f…` |
| Smoke (both models) | 7246049 | 21 | 672 | L40S | 0:14:30 | `26eb273f807b2813…` |

Full sha256 values of the eight scoring shards (as recorded in the analysis run's `inputs.json`):

- Mistral shard 0: `f69975a9550a889abed64aef14fe574c58b57c81d77ccf730de2b7a5003562e2`
- Mistral shard 1: `70e9eec14571c44be562a2753206a583f974ef809830f04a6c76d55630906641`
- Mistral shard 2: `52c8cd08926ecd616492ac3b3455958a4a61dbd8aaa78174087a40c4576c9193`
- Mistral shard 3: `9722e09cfba3cc7703485675ddb795e1e9f416006435f7246ba1ca639c44d2fd`
- Llama 3.2 3B shard 0: `9342c805c8abe25bdf7233e0d9b446817b499421f81ed14f15d93604b0090c23`
- Llama 3.2 3B shard 1: `5e3572a5318f24246597fe399a5bde7b6d35be1d639b6a5ed978d09e996c679f`
- Llama 3.2 3B shard 2: `7232f3158d43720db2aeffbec98eb78689883a8d2c391fce6b353bc1c12c60d4`
- Llama 3.2 3B shard 3: `fcf499a39cd5314bf9ca8df06fff66063feaeab05c764f753fa93f1fc3732f1f`
- Placement check: `bbc627ab35366c4f6c9430dd62ba2e93b73759a6203b59e13bfcb22512644ee2`
- Smoke run: `26eb273f807b2813e0824b45a6ee11a86dfbbe7ce2976ffd29321d082fe93409`

- **Analysis** (`tau_family_analysis`, CPU, job 7251608, 1.5 minutes). Status REPORTED. It used the reference run `tau_score_reference`, a symlink in the checkout's `runs/` to `text-wm-fix25/runs/tau_score`. That run is the reference models' 16 shards, all at `c737cd6`, with 0 missing, 0 stale and no problems. The reference results check is equal for both models against `notes/tau_metrics.json`, whose copy in the checkout is byte-identical to the file on `main` (sha256 `1c80cbc2…`). The metrics file is copied unchanged as `notes/tau_families_metrics.json` (sha256 `7c9af2d59d2bc278…`).

## 11. Operational events and deviations

- **The first analysis run failed.** Job 7247987 (04:09, exit code 1 after 50 seconds) ran before the reference scoring run was linked into the checkout's `runs/` folder as `tau_score_reference`. The analysis needs that run for the reference recompute and results check. Its log holds only the job's start line and no traceback, and its status file was overwritten by the second run. The link was then created (05:49), pointing at `text-wm-fix25/runs/tau_score`, and the analysis was rerun unchanged at the same commit (job 7251608). No code, config, score or rule changed between the two runs.
- **Gemma 2 9B was excluded before any data** (design §2.3, decided 2026-11-22). The token check confirmed that every prompt exceeds its context. It is not in the Holm family and has no result.
- **The smoke record was not copied into the design note before full scoring.** The design says the smoke projection "is copied into this note and committed before full scoring" (§6.4, §10). The smoke analysis ran (job 7247971) and was read before scoring, but no commit with its record was made on the branch; the tag is the last commit there. The record is given in section 10 above. It changed nothing: the projection was within the time limit and the budget, and no setting was adjusted.
- **The sensitivity verdict for Mistral is stored without the competence gate.** `sensitivity_without_fewshot_tool_classes` gives Mistral "small effect, direction positive", because the τ-bench pipeline's sensitivity verdict does not know the families competence gate. It is the ungated category of the sensitivity population, matching the main analysis's ungated category (`same_category: true`). It is not Mistral's verdict, which is "not interpretable".
- **Label shorthand.** `patterns.labels` stores Llama 3.2 3B's label as the code "shared". The full label, in `models.llama3.2-3b.description.label`, is "a shift shared by working and failing calls (working calls move more)".
