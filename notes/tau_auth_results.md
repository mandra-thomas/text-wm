# τ-bench authorization experiment: results (2026-11-23)

Exploratory study with a frozen procedure. The design and the analysis were frozen before any score at the annotated tag `tau-auth-design-frozen-2026-11-22` (commit `f70b285`, branch `tau-auth`). The design note, [tau_auth_design.md](tau_auth_design.md), is brought onto `main` unchanged with this note, together with the tokenization report it links ([tau_auth_tokcheck.json](tau_auth_tokcheck.json)); the code (`wf/tau/auth.py`), the tests and the configs stay on the branch. The smoke run, the four scoring shards and the analysis all ran at that commit, in a clean checkout. Every number below is quoted from [tau_auth_metrics.json](tau_auth_metrics.json), a byte-identical copy of the analysis run's `metrics.json` (sha256 `73ad25b9a14c28d0…`). Nothing here is recomputed, no category is re-derived and no test is added. Paragraphs marked "exploratory reading" are interpretation of the prespecified numbers and decide nothing.

## Summary

The experiment asks whether the τ-bench task effect on a failing call is a wrong belief about what the backend checks (that it refuses writes the user did not request, which a clear description of the backend should remove) or a task-relevance bias that persists when the backend is stated, and shown by examples, to ignore the task.

For **Qwen3-8B** both gates passed: it answered all four comprehension questions about the task-blind backend correctly on all 24 items, and its predictions followed a backend described as enforcing a request rule (positive control +3.38 nats [+3.01, +3.77], against a threshold of 1 nat). Its request component (what a write request adds over a bare question about the same order) was +2.30 nats [+1.87, +2.74] under the confirmatory prompt. The task-blind description removed −0.09 [−0.42, +0.26] of it, that is nothing, and +2.39 [+1.94, +2.87] persisted. The naming component behaved the same way (+1.87 under the main prompt, −0.08 removed, +1.95 persisting). Both frozen categories are "relevance-like (reading (a))", and the frozen reading is "(a) throughout". A removal of the size the margin treats as meaningful is excluded: the interval's upper end, +0.26 nats, is about a tenth of the component and well inside the margin of 0.72 nats.

For **Llama-3.1-8B** both gates failed, so both categories are "inconclusive (gate)". In the task-blind arm it said, on all 24 items, that a requested write that breaks a listed rule would be executed without an error (question K2), although it answered the same question correctly under the other two descriptions; and its positive control was +0.30 nats [+0.19, +0.40], above zero but below the prestated 1 nat. Descriptively, its request component did not shrink under the task-blind description; it grew, from +0.54 [+0.43, +0.66] to +0.74 [+0.62, +0.85].

| | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Comprehension of the task-blind arm (gate 1) | passes: K1–K4 each 24/24 | fails: K2 0/24 (K1, K3, K4 24/24) |
| Positive control, nats (gate 2: at least 1, interval above 0) | +3.38 [+3.01, +3.77], passes | +0.30 [+0.19, +0.40], fails |
| Request component, main prompt | +2.30 [+1.87, +2.74] | +0.54 [+0.43, +0.66] |
| Request component, removed by the task-blind arm | −0.09 [−0.42, +0.26] | −0.20 [−0.29, −0.11] |
| Request component, persisting under the task-blind arm | +2.39 [+1.94, +2.87] | +0.74 [+0.62, +0.85] |
| Naming component, main prompt | +1.87 [+1.39, +2.36] | −0.08 [−0.21, +0.05] |
| Naming component, removed by the task-blind arm | −0.08 [−0.33, +0.18] | −0.01 [−0.10, +0.07] |
| Naming component, persisting under the task-blind arm | +1.95 [+1.58, +2.33] | −0.07 [−0.17, +0.04] |
| Category, request component | relevance-like (reading (a)) | inconclusive (gate) |
| Category, naming component | relevance-like (reading (a)) | inconclusive (gate) |

*(Extension, 2026-11-25)* The same frozen design was then run on a third model, **Llama-3.2-3B-Instruct**, under an extension committed and tagged before any of its scores existed (`tau-auth-ext-llama32-2026-11-25`). It also fails both gates: in the task-blind arm it is indifferent on question K3 (correct on 10 of 24 items, mean log-odds of the correct answer +0.01), and its positive control is +0.41 nats [+0.34, +0.48], below 1 nat. Both of its categories are therefore "inconclusive (gate)". Section 12 gives the full extension result.

## 1. The question

A **world model** here is a language model asked to predict what τ-bench's retail backend replies to an agent's tool call. The backend is a customer-service database of users, orders and products; the agent calls tools such as `modify_pending_order_items` or `exchange_delivered_order_items`. The prompt contains the backend's description, a few worked examples (**few-shot examples**), the history of earlier calls and their outputs, a one-line description of what the user wants (the **task line**), and the next call. Every scored item is a **failing call**: a write that the backend rejects because one of its listed rules is broken, and the broken rule is visible in the history (the order's status forbids the write, the `status` class, or the requested replacement variant shows `"available": false`, the `unavailable` class). The real backend never reads the task: its reply is a function of the call and the database only, and a task-invariance check in the confirmatory build confirmed that every scored outcome is fixed by the state whatever the task line says. Any difference between task lines in a model's prediction is therefore an error.

The confirmatory τ-bench study (`notes/tau_results.md`) found such an error. When the task line asks for exactly the change the failing call makes (the **aligned** line), models predict success more readily than when it asks for the same kind of change to another of the user's orders (the **swapped** line). The follow-up (`notes/tau_dx.md`) split this goal effect into two parts using two extra task lines that only ask a question about an order and request nothing ("The user has a question about the {product} in order {order}."): a **naming component**, which is what naming the right order does without any request, and a **request component**, which is what the write request adds on top. The request component was +2.17 nats for Qwen3-8B and +0.50 for Llama-3.1-8B on the 856 primary items, and the follow-up could not say what produced it. Two explanations predict it equally well:

1. **(a) Task relevance or task consistency.** The model predicts success more readily for a call that fits what the user wants. On this reading the bias is not a belief about the backend, so stating and showing that the backend ignores the task should not remove it.
2. **(b) A belief in request-matching authorization.** The model believes the backend checks each write against the user's request and refuses writes the user did not ask for, the way a real customer-service system might. This is a wrong belief about the system being modelled, and a clear description with demonstrations should correct it.

The experiment changes only the description and demonstration of the backend, holding fixed the items, task lines, histories, calls and real outcomes. The part of an effect that **disappears** when the prompt states and demonstrates a task-blind backend is called **authorization-like**; the part that **persists** is called **relevance-like**. This is an operational definition with one known limit, which the design addresses: an explicit instruction might suppress a bias that is not a belief at all. The naming component serves as the control for that, because a bare question requests no write, so an authorization belief has nothing to say about it, while a general suppression of task effects would shrink it too.

## 2. What was run

### 2.1 The three backend descriptions (arms)

Each **arm** is one version of the system message. Everything else (items, task lines, histories, calls, readouts, the order of the few-shot examples for each item) is identical across arms, and the scoring run refuses to score if the prompts differ anywhere else.

1. **Arm (i), main.** The confirmatory prompt, byte for byte. It already says that the backend "sees only each call's name and arguments and the current database: it does not see the user's messages or task". It has four few-shot examples arranged as a balanced two-by-two: the task names the call's own order or another order, and the call succeeds or fails.
2. **Arm (iii), task-blind.** Arm (i) unchanged plus a closing paragraph: "The user's task never changes a call's output: the backend does not compare a call with the user's request. A write call that changes an order the user's task does not ask to change, or that is made when the task asks for no change (only a question about an order, or general help with the account), is executed exactly as it would be if the user had asked for it. A write call that breaks one of the rules above returns that rule's error whatever the task." It adds two few-shot examples, each a **minimal pair** with an existing one: the same user, history and call with the task line changed from another order to this very order, and the same output. So the prompt both states and shows that the task does not change the output.
3. **Arm (ii), authorization.** A counterfactual backend that does check the request: "a write call is executed only if the user's task asks for that change to that order", otherwise it returns "Error: The user did not request this change". It has the same six examples, with the one other-order success replaced by that error. This arm is never read for accuracy. It is the **positive control**: it shows whether the model's predictions follow a stated backend rule at all.

Arms (ii) and (iii) are about 1,670 and 2,060 tokens longer than arm (i). That and the different number of error examples are level shifts that affect every task line in an arm alike; every contrast below is a difference between task lines within an arm, which a level shift does not change.

### 2.2 Items, task lines, actions and readouts

The 192 items are the first 96 of each of the two classes in a fixed hash order of the primary population (116 users). Each item has five task lines: **aligned** (the target order and its product, with the write request), **swapped** (another order of the same user with the same status, and its product, with the same request), **neutral** ("The user wants help with their account."), **mention-aligned** (a question about the target order's product, no request) and **mention-swapped** (the same question about the comparison order). Each item is scored for two **actions**: the failing call, and the **witness**, a single write on the same order that the backend accepts and that would achieve the aligned task (in the example of `GUIDE.md` §2.3, the same exchange to an available variant). The real outcome is failure for the failing call and success for the witness, under every task line and in every arm.

The primary **readout** is the first token of the backend's reply. Every reply begins either with `{"` (a JSON success) or with `Error`, so the score of one prompt is the log-odds of these two openings,

$$L = \log P(\texttt{\{"}) - \log P(\texttt{Error}),$$

measured in **nats** (natural-log units; +1 nat multiplies the odds of success by about 2.7). A secondary yes/no readout asks the model whether the backend will execute the call without an error and scores the log-odds of Yes against No; the confirmatory follow-up showed that Qwen's yes/no answer carries a judgement about the task beyond its predicted output, so this readout never decides anything here.

### 2.3 The comprehension check (K1–K4)

On 24 items (12 per class), in every arm, the prompt with the neutral task line and no scored call ends with one of four general questions about the backend, scored as the log-odds of Yes against No:

| Question | What it asks, in words | Correct in arms (i) and (iii) | Correct in arm (ii) |
|---|---|---|---|
| K1 | A write that breaks no listed rule but changes an order the task does not ask to change: executed without an error? | Yes | No |
| K2 | A write that does what the task asks but breaks a listed rule: executed without an error? | No | No |
| K3 | A write that breaks no rule, made when the task is only a question about that order: does it return an error? | No | Yes |
| K4 | A write that does what the task asks and breaks no rule: executed without an error? | Yes | Yes |

K1 and K3 are the distinction the arms are about. K2 and K4 have the same answer in every arm and catch a general lean toward Yes or No. Each arm has two Yes and two No answers (K3 was reworded after review 1, amendment R1-S3, so that a model leaning toward one answer cannot pass). **An arm passes** when, on each of the four questions, the correct answer has the higher probability on at least 80% of the 24 items. Only the task-blind arm's pass gates anything; the other two arms are reported. In arm (i), K1 and K3 are also a direct question about the authorization belief under the confirmatory prompt.

## 3. The quantities

For one item, one arm and one action, write $L(c)$ for the first-token score under task line $c$, with al, sw, ne, ma and ms for the aligned, swapped, neutral, mention-aligned and mention-swapped lines. The goal effect, the naming component and the request component are

$$G = L(\text{al}) - L(\text{sw}), \qquad M = L(\text{ma}) - L(\text{ms}), \qquad W = G - M.$$

Rearranged, the request component is the difference of two halves, each a write request minus the bare question about the same order:

$$W = \big[L(\text{al}) - L(\text{ma})\big] - \big[L(\text{sw}) - L(\text{ms})\big].$$

The first bracket is the **aligned half** and the second the **swapped half**. A positive request component means that requesting a change to the right order raises predicted success more, relative to just asking about it, than requesting the same change to the wrong order does.

For a component $X$ (either $W$ or $M$), the decomposition across arms is

$$Z_X = X^{\text{main}} - X^{\text{blind}}, \qquad R_X = X^{\text{blind}}, \qquad X^{\text{main}} = Z_X + R_X.$$

$Z$ is the **removed part**: how much of the arm (i) component the task-blind description takes away (authorization-like). $R$ is the **persisting part**: what remains under the task-blind description (relevance-like). A negative $Z$ means the component grew in the task-blind arm. The **positive control** is

$$PC = G^{\text{auth, witness}} - G^{\text{blind, witness}},$$

the extra aligned − swapped gap on the witness when the backend is described as enforcing the request rule rather than ignoring the task. Under arm (ii)'s rule the witness should succeed only under the aligned line, so a model that follows the stated backend should show a large positive value.

Every quantity is a mean over the 192 items of a per-item difference. Its 95% **confidence interval** (CI) is a percentile bootstrap over users (2,000 resamples, resampling whole users because one user's items are correlated). Its p-value is a two-sided **sign-flip permutation test** in which all of a user's items flip sign together (seed 20260924). The eight primary tests (the removed and persisting parts of both components for both models) use 10,000 permutations and form the **Holm family**: Holm's step-down correction is applied to exactly these eight, at α = 0.01. Every other p-value uses 1,000 permutations, is unadjusted and decides nothing (the smallest possible values are about 0.0001 and 0.001).

The **margin** $\delta$ is a third of each model's confirmatory request component: 0.72 nats for Qwen and 0.17 for Llama. A part is **material** if its estimate is at least $\delta$, its CI is above 0 and its Holm-adjusted p is below 0.01; **negligible** if its CI lies inside $(-\delta, \delta)$; otherwise uncertain. A component whose arm (i) CI lies inside $(-\delta, \delta)$ is **absent**. Otherwise its category follows from its two parts (design §6.4, as amended): removed part negligible and persisting part material is **relevance-like (reading (a))**; the reverse is **authorization-like (reading (b))**; both material is "both"; a removed part of at least $\delta$ in the wrong direction (the component grows) is "grows under the task-blind arm", reported and not interpreted; and so on through the table of §6.4. Two **gates** come first: if a model fails either, all its categories become "inconclusive (gate)".

## 4. The gates

### 4.1 Comprehension, per arm and question

Each cell gives the correct answer, the number of the 24 items on which it had the higher probability, and the mean log-odds of the correct answer in nats (positive means the model leaned toward the correct answer on average). Only the task-blind arm's row gates.

| Model, arm | K1 | K2 | K3 | K4 | Arm passes |
|---|---|---|---|---|---|
| Qwen3-8B, arm (i) main | Yes: 0/24, -2.96 | No: 24/24, +22.74 | No: 24/24, +18.76 | Yes: 24/24, +14.51 | no |
| Qwen3-8B, arm (iii) task-blind | Yes: 24/24, +7.51 | No: 24/24, +17.94 | No: 24/24, +16.16 | Yes: 24/24, +15.70 | yes |
| Qwen3-8B, arm (ii) authorization | No: 24/24, +11.04 | No: 24/24, +21.74 | Yes: 0/24, -11.11 | Yes: 24/24, +14.53 | no |
| Llama-3.1-8B, arm (i) main | Yes: 0/24, -0.69 | No: 24/24, +2.14 | No: 24/24, +1.16 | Yes: 24/24, +0.88 | no |
| Llama-3.1-8B, arm (iii) task-blind | Yes: 24/24, +1.62 | No: 0/24, -0.83 | No: 24/24, +0.90 | Yes: 24/24, +2.40 | no |
| Llama-3.1-8B, arm (ii) authorization | No: 23/24, +0.52 | No: 24/24, +2.00 | Yes: 24/24, +2.05 | Yes: 24/24, +0.40 | yes |

Qwen3-8B answered every question correctly on every item in the task-blind arm, with large margins (+7.51 to +17.94 nats), so it passes gate 1. Under the confirmatory prompt it answered K1 wrong on all 24 items (−2.96): asked whether a valid write on an order the task does not ask to change would go through, it said No, which is the authorization belief stated in words. Yet it answered K3 correctly under the same prompt (a valid write when the task is only a question would not return an error), which an authorization belief would get wrong. So its arm (i) answers do not express a consistent request rule: it refuses the other-order write but not the question-only write. In the authorization arm it again failed K3 (−11.11): it said that the request-enforcing backend would not return an error for a write made when the task is only a question, although the arm's text says it would. This failure does not gate.

Llama-3.1-8B failed gate 1 on K2 alone. In the task-blind arm it said Yes on all 24 items (mean log-odds of the correct No −0.83) to whether a requested write that breaks a listed rule would be executed without an error, whereas it answered the same question correctly on all 24 items in arm (i) (+2.14) and arm (ii) (+2.00). Its K1, K3 and K4 answers in the task-blind arm were all correct. Under the confirmatory prompt it, like Qwen, answered K1 wrong on all 24 items (−0.69) and K3 right. Its arm (ii) answers pass on all four questions (K1 on 23 of 24), with small margins (+0.40 to +2.05). All of Llama's yes/no margins are a few nats at most, an order of magnitude smaller than Qwen's.

### 4.2 Positive control

| | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Positive control (authorization − task-blind goal effect on the witness), nats | +3.38 [+3.01, +3.77] | +0.30 [+0.19, +0.40] |
| Items with a positive value | 93.8% | 65.1% |
| Item-level SD, nats | 2.64 | 0.69 |
| Unadjusted p (1,000 permutations) | 0.001 | 0.001 |
| Gate (at least 1 nat, CI above 0) | passes | fails |

Qwen's predictions moved strongly with the stated rule. Llama's moved in the right direction, reliably (the CI excludes 0), but by less than a third of the threshold.

### 4.3 The smoke run's gates

The design allowed the smoke run's gates to be read before the full run (amendment R1-S3), with any change made after reading them logged as a deviation. No change was made: the full run used the same commit and configs. On the 7 smoke items (job 7246045), the smoke analysis (job 7247921) showed the same picture as the full run. Llama failed the task-blind comprehension on K2 (0 of 7, −0.63) and the positive control (−0.09 [−0.64, +0.39]); Qwen passed both (positive control +4.39 [+2.20, +6.41]) and failed K3 in the authorization arm (0 of 7, −9.32), which does not gate. Under arm (i), K1 was wrong on 7 of 7 items for Llama and 6 of 7 for Qwen. So Llama's gate failure was known after the smoke run, and the full run went ahead under the frozen rules, which make a gate failure a reported outcome, not a reason to change the design.

## 5. The primary decomposition and the frozen categories

### 5.1 Estimates

| Quantity | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Request component, main prompt | +2.30 [+1.87, +2.74], p 0.001 | +0.54 [+0.43, +0.66], p 0.001 |
| Request component, removed by the task-blind arm | −0.09 [−0.42, +0.26], p 0.614 | −0.20 [−0.29, −0.11], p 0.0001 |
| Request component, persisting under the task-blind arm | +2.39 [+1.94, +2.87], p 0.0001 | +0.74 [+0.62, +0.85], p 0.0001 |
| Removed part of the aligned half | −0.99 [−1.28, −0.68], p 0.001 | −0.38 [−0.48, −0.27], p 0.001 |
| Contribution of the swapped half to the removed part | +0.90 [+0.67, +1.14], p 0.001 | +0.18 [+0.09, +0.26], p 0.001 |
| Removed part of aligned − neutral | −1.37 [−1.67, −1.05], p 0.001 | −0.39 [−0.53, −0.26], p 0.001 |
| Removed part of swapped − neutral | −1.21 [−1.49, −0.92], p 0.001 | −0.18 [−0.33, −0.02], p 0.023 |
| Naming component, main prompt | +1.87 [+1.39, +2.36], p 0.001 | −0.08 [−0.21, +0.05], p 0.246 |
| Naming component, removed by the task-blind arm | −0.08 [−0.33, +0.18], p 0.545 | −0.01 [−0.10, +0.07], p 0.766 |
| Naming component, persisting under the task-blind arm | +1.95 [+1.58, +2.33], p 0.0001 | −0.07 [−0.17, +0.04], p 0.242 |

The last four rows of the request component are the split of amendment R1-M2: the removed part of the request component equals the removed part of the aligned half plus the contribution of the swapped half (Qwen: −0.99 + 0.90 = −0.09 up to rounding), and the removed parts of aligned − neutral and swapped − neutral show where each moved relative to the neutral line. A negative removed part means the quantity was larger in the task-blind arm.

### 5.2 The Holm family

| Test | Estimate [95% CI] | Unadjusted p (10,000 permutations) | Holm-adjusted p |
|---|---|---|---|
| Qwen3-8B, request component, removed | −0.09 [−0.42, +0.26] | 0.614 | 1.000 |
| Qwen3-8B, request component, persisting | +2.39 [+1.94, +2.87] | 0.0001 | 0.0008 |
| Qwen3-8B, naming component, removed | −0.08 [−0.33, +0.18] | 0.545 | 1.000 |
| Qwen3-8B, naming component, persisting | +1.95 [+1.58, +2.33] | 0.0001 | 0.0008 |
| Llama-3.1-8B, request component, removed | −0.20 [−0.29, −0.11] | 0.0001 | 0.0008 |
| Llama-3.1-8B, request component, persisting | +0.74 [+0.62, +0.85] | 0.0001 | 0.0008 |
| Llama-3.1-8B, naming component, removed | −0.01 [−0.10, +0.07] | 0.766 | 1.000 |
| Llama-3.1-8B, naming component, persisting | −0.07 [−0.17, +0.04] | 0.242 | 0.967 |

A p of 0.0001 is the smallest possible with 10,000 permutations; Holm multiplies the smallest by 8, which gives 0.0008, still below 0.01, as the design intended.

### 5.3 Categories and readings, as output

**Qwen3-8B.** Request component: "relevance-like (reading (a))". Its arm (i) value, +2.30, is far outside the margin, so it is not absent; the removed part's CI, [−0.42, +0.26], lies inside (−0.72, +0.72), so the removed part is negligible; the persisting part, +2.39, exceeds 0.72 with its CI above 0 and Holm p 0.0008, so it is material. Naming component: "relevance-like (reading (a))", by the same steps (removed [−0.33, +0.18] inside the margin; persisting +1.95, Holm p 0.0008). The analysis's reading, verbatim: "(a) throughout: neither the request nor the naming effect behaves like a belief about request-matching authorization; both persist when the backend is stated and shown to ignore the task. Share removed by the task-blind arm (descriptive): request -0.04 [-0.20, +0.10], naming -0.04 [-0.23, +0.08]; a uniform suppression predicts equal shares." The aligned-half condition of R1-M2 (`aligned_half_shown`) is false, but it only matters when a removal is shown, and none is.

The **removed shares** of amendment R1-M1 divide each component's removed part by its arm (i) value:

$$s_W = \frac{\overline{Z_W}}{\overline{W^{\text{main}}}}, \qquad s_M = \frac{\overline{Z_M}}{\overline{M^{\text{main}}}}.$$

For Qwen they are −0.04 [−0.20, +0.10] for the request component and −0.04 [−0.23, +0.08] for the naming component, with a difference of +0.00 [−0.22, +0.26]. At most about a tenth of either component is removed, and the two are removed equally (that is, not at all).

**Llama-3.1-8B.** Both categories: "inconclusive (gate)". Reading, verbatim: "Inconclusive: the model is not shown to understand that the task-blind backend ignores the task (comprehension check of the task-blind arm); the model's predictions are not shown to follow the stated backend (positive control)." Section 6 explains what this means.

The analysis also stores, under `ungated_categories`, what the category rules give when the gates are ignored. This field is an output of the frozen code, not a category of the design, and it decides nothing; it is listed for completeness. For Llama it is "grows under the task-blind arm (predicted by neither; reported, not interpreted)" for the request component (its removed part, −0.20, is beyond −0.17 with its CI below 0 and Holm p 0.0008) and "inconclusive" for the naming component (its arm (i) CI, [−0.21, +0.05], reaches just outside the margin, so it is not "absent"; both of its parts are negligible, a combination the table of §6.4 does not name, so it falls to "inconclusive"). Llama's removed shares are −0.37 [−0.63, −0.19] for the request component; the naming share, +0.18 [−3.50, +4.83], divides by a near-zero component and carries no information. For Qwen the ungated categories equal the gated ones.

## 6. Why Llama reads "inconclusive (gate)", and what that does and does not imply

The design makes two things preconditions for reading a model's decomposition (§6.1 and §6.6). First, the model must be shown to understand the task-blind description; otherwise a persisting effect could mean only that the model did not take in the statement. Second, the model's predictions must be shown to follow a stated backend at all; otherwise an unchanged effect in the task-blind arm would be uninformative, because a model that ignores the system message would show no change whatever its beliefs. Llama fails both, so the rule replaces both of its categories with "inconclusive (gate)".

**The comprehension failure is narrow and odd.** Llama misread one question in one arm: in the task-blind arm it said that a requested write breaking a listed rule would go through. The arm's text says the opposite in its last sentence ("A write call that breaks one of the rules above returns that rule's error whatever the task"), and Llama got the same question right under the other two descriptions. One plausible cause (an exploratory reading, not tested) is that the paragraph's central sentence, that writes are "executed exactly as it would be if the user had asked for it", pulls Llama's answer to any question about a requested write toward Yes. Its K1 and K3 answers in this arm, the two questions about the distinction the arm is about, were correct on every item. So the failure does not show that Llama misunderstood the task-independence the arm states; it shows that Llama does not meet the prestated standard, which requires all four answers.

**The positive-control failure is a matter of size.** Llama's witness predictions did move toward the stated authorization rule: the aligned − swapped gap was +0.30 nats [+0.19, +0.40] larger under the authorization description, on 65% of items. But the threshold is an absolute 1 nat, and Llama's scores are compressed: its whole request component is half a nat, and its comprehension margins are a few nats where Qwen's are mostly above ten. The level changes of §7.3 show how it responded to arm (ii): every line on both actions moved toward `Error`, by 0.5 to 1.1 nats, the aligned line by somewhat less than most of the others. Llama took in that arm (ii) has more errors, but barely applied the rule's distinction between requested and unrequested writes, even though it answered the rule questions about arm (ii) correctly.

**What the inconclusive result does not imply.** It does not mean that Llama has an authorization belief, nor that it lacks one, nor that its request effect is absent: the request component is there under every description (+0.54, +0.74 and +0.78 nats in arms (i), (iii) and (ii)). It does not license reading the ungated "grows" label either; the design says that pattern is "predicted by neither" explanation and is reported, not interpreted. What can be said descriptively is narrower: nothing in Llama's numbers looks like the removal an authorization belief predicts, because the task-blind description made the request component larger, not smaller, and made the aligned half grow (§7.3). Whether that is a relevance bias the description amplified, a response to the two added examples, or noise in a model that follows the system text weakly, the gates are there precisely so that this question is not answered from these data.

**What would change it.** Only a new, separately frozen design could give Llama a category: for example, a task-blind paragraph whose wording Llama passes on all four questions, checked on a smoke run before freezing, and a positive-control threshold scaled to the model's own spread of scores rather than fixed at 1 nat. Any such change made now would be a post hoc change of a frozen rule after seeing the result, and would have to be labelled as such.

## 7. Prespecified secondary analyses (design §7)

These are estimates with CIs and unadjusted p-values; none is a verdict.

### 7.1 Witness decomposition

The same decomposition on the witness, the working call on the same order. Design §7.1 asked for the §6.4 categories as labels here too; the frozen analysis code reports the numbers but does not compute those labels, so none is given (see §10).

| Quantity | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Request component, main prompt | +6.64 [+5.98, +7.37], p 0.001 | +0.42 [+0.32, +0.52], p 0.001 |
| Request component, removed by the task-blind arm | +0.73 [+0.23, +1.26], p 0.009 | −0.51 [−0.59, −0.43], p 0.001 |
| Request component, persisting under the task-blind arm | +5.91 [+5.30, +6.55], p 0.001 | +0.93 [+0.82, +1.04], p 0.001 |
| Removed part of the aligned half | −0.75 [−1.21, −0.27], p 0.003 | −0.68 [−0.80, −0.57], p 0.001 |
| Contribution of the swapped half to the removed part | +1.48 [+1.11, +1.85], p 0.001 | +0.17 [+0.06, +0.29], p 0.003 |
| Removed part of aligned − neutral | −0.88 [−1.34, −0.40], p 0.003 | −0.59 [−0.73, −0.45], p 0.001 |
| Removed part of swapped − neutral | −1.43 [−1.91, −0.97], p 0.001 | +0.05 [−0.11, +0.18], p 0.530 |
| Naming component, main prompt | +3.12 [+2.51, +3.76], p 0.001 | −0.02 [−0.15, +0.10], p 0.722 |
| Naming component, removed by the task-blind arm | −0.18 [−0.45, +0.11], p 0.226 | −0.12 [−0.21, −0.03], p 0.011 |
| Naming component, persisting under the task-blind arm | +3.30 [+2.73, +3.88], p 0.001 | +0.10 [−0.02, +0.21], p 0.136 |

For Qwen, the witness's request component (+6.64 nats) is almost three times the failing call's, as in the confirmatory study, where the witness moved more than the failing call. The task-blind description removed +0.73 [+0.23, +1.26] of it, about a ninth (unadjusted p 0.009). That removal is not in the aligned half, which grew (−0.75); it lies entirely in the swapped half, which rose by 1.48 nats. This is the pattern amendment R1-M2 attributes to copying of the demonstrated minimal pairs rather than to an authorization belief (in the pairs, an other-order write request is shown to give the same output as an own-order one). For Llama, the witness's request component grew from +0.42 to +0.93 under the task-blind description, again mostly through the aligned half.

### 7.2 Per-arm contrasts, both actions

The goal effect, its two components and halves, and every line against the neutral line, per arm. For orientation, the confirmatory values on all 856 primary items were, for the failing call, a goal effect of +3.98 (Qwen) and +0.47 (Llama), a naming component of +1.81 and −0.03, a request component of +2.17 and +0.50, and swapped − neutral of −2.56 and +0.45; the witness's goal effect was +9.45 and +0.38. This run's arm (i) values on the 192-item subset are close to all of them.

**Qwen3-8B, failing call, first token (nats; 95% CI).**

| Quantity | Arm (i), main | Arm (iii), task-blind | Arm (ii), authorization |
|---|---|---|---|
| Goal effect (aligned − swapped) | +4.17 [+3.49, +4.91] | +4.33 [+3.72, +5.00] | +6.27 [+5.54, +7.10] |
| Naming component (mention-aligned − mention-swapped) | +1.87 [+1.39, +2.36] | +1.95 [+1.58, +2.33] | +2.90 [+2.46, +3.38] |
| Request component (goal effect − naming component) | +2.30 [+1.87, +2.74] | +2.39 [+1.94, +2.87] | +3.37 [+2.87, +3.91] |
| Aligned half (aligned − mention-aligned) | +1.61 [+1.14, +2.10] | +2.60 [+2.08, +3.14] | +3.25 [+2.64, +3.87] |
| Swapped half (swapped − mention-swapped) | −0.69 [−1.01, −0.36] | +0.22 [−0.06, +0.52] | −0.12 [−0.45, +0.22] |
| Aligned − neutral | +1.47 [+0.85, +2.13] | +2.84 [+2.21, +3.48] | +3.73 [+3.02, +4.45] |
| Swapped − neutral | −2.70 [−3.12, −2.28] | −1.49 [−1.90, −1.10] | −2.54 [−3.02, −2.09] |
| Mention-aligned − neutral | −0.14 [−0.50, +0.21] | +0.24 [−0.06, +0.52] | +0.47 [+0.14, +0.79] |
| Mention-swapped − neutral | −2.01 [−2.41, −1.65] | −1.71 [−2.04, −1.39] | −2.43 [−2.91, −1.98] |

**Qwen3-8B, witness, first token (nats; 95% CI).**

| Quantity | Arm (i), main | Arm (iii), task-blind | Arm (ii), authorization |
|---|---|---|---|
| Goal effect (aligned − swapped) | +9.76 [+8.86, +10.74] | +9.21 [+8.34, +10.08] | +12.59 [+11.56, +13.62] |
| Naming component (mention-aligned − mention-swapped) | +3.12 [+2.51, +3.76] | +3.30 [+2.73, +3.88] | +5.03 [+4.28, +5.78] |
| Request component (goal effect − naming component) | +6.64 [+5.98, +7.37] | +5.91 [+5.30, +6.55] | +7.56 [+6.85, +8.28] |
| Aligned half (aligned − mention-aligned) | +5.79 [+5.28, +6.32] | +6.54 [+5.92, +7.14] | +7.34 [+6.69, +8.02] |
| Swapped half (swapped − mention-swapped) | −0.85 [−1.38, −0.32] | +0.63 [+0.20, +1.07] | −0.22 [−0.70, +0.30] |
| Aligned − neutral | +6.07 [+5.39, +6.78] | +6.95 [+6.22, +7.68] | +7.91 [+7.14, +8.69] |
| Swapped − neutral | −3.69 [−4.43, −2.97] | −2.26 [−2.85, −1.69] | −4.68 [−5.37, −4.02] |
| Mention-aligned − neutral | +0.28 [−0.20, +0.75] | +0.41 [−0.02, +0.84] | +0.57 [+0.05, +1.09] |
| Mention-swapped − neutral | −2.83 [−3.29, −2.40] | −2.89 [−3.27, −2.54] | −4.46 [−5.06, −3.91] |

**Llama-3.1-8B, failing call, first token (nats; 95% CI).**

| Quantity | Arm (i), main | Arm (iii), task-blind | Arm (ii), authorization |
|---|---|---|---|
| Goal effect (aligned − swapped) | +0.46 [+0.28, +0.65] | +0.67 [+0.51, +0.84] | +1.02 [+0.85, +1.19] |
| Naming component (mention-aligned − mention-swapped) | −0.08 [−0.21, +0.05] | −0.07 [−0.17, +0.04] | +0.23 [+0.12, +0.35] |
| Request component (goal effect − naming component) | +0.54 [+0.43, +0.66] | +0.74 [+0.62, +0.85] | +0.78 [+0.65, +0.91] |
| Aligned half (aligned − mention-aligned) | +1.47 [+1.32, +1.62] | +1.85 [+1.69, +2.00] | +1.95 [+1.78, +2.12] |
| Swapped half (swapped − mention-swapped) | +0.93 [+0.81, +1.05] | +1.11 [+1.01, +1.21] | +1.17 [+1.04, +1.30] |
| Aligned − neutral | +0.91 [+0.75, +1.08] | +1.30 [+1.14, +1.46] | +1.78 [+1.56, +2.01] |
| Swapped − neutral | +0.45 [+0.28, +0.62] | +0.63 [+0.52, +0.73] | +0.76 [+0.60, +0.93] |
| Mention-aligned − neutral | −0.56 [−0.65, −0.47] | −0.54 [−0.62, −0.47] | −0.17 [−0.27, −0.08] |
| Mention-swapped − neutral | −0.48 [−0.60, −0.36] | −0.48 [−0.58, −0.39] | −0.41 [−0.51, −0.31] |

**Llama-3.1-8B, witness, first token (nats; 95% CI).**

| Quantity | Arm (i), main | Arm (iii), task-blind | Arm (ii), authorization |
|---|---|---|---|
| Goal effect (aligned − swapped) | +0.39 [+0.26, +0.53] | +1.02 [+0.88, +1.17] | +1.32 [+1.16, +1.48] |
| Naming component (mention-aligned − mention-swapped) | −0.02 [−0.15, +0.10] | +0.10 [−0.02, +0.21] | +0.47 [+0.36, +0.59] |
| Request component (goal effect − naming component) | +0.42 [+0.32, +0.52] | +0.93 [+0.82, +1.04] | +0.85 [+0.72, +0.97] |
| Aligned half (aligned − mention-aligned) | +1.23 [+1.04, +1.42] | +1.91 [+1.76, +2.06] | +1.70 [+1.54, +1.84] |
| Swapped half (swapped − mention-swapped) | +0.81 [+0.65, +0.97] | +0.99 [+0.87, +1.11] | +0.85 [+0.72, +0.97] |
| Aligned − neutral | +0.86 [+0.71, +1.01] | +1.45 [+1.32, +1.57] | +1.82 [+1.67, +1.98] |
| Swapped − neutral | +0.47 [+0.32, +0.61] | +0.43 [+0.29, +0.54] | +0.50 [+0.34, +0.65] |
| Mention-aligned − neutral | −0.37 [−0.47, −0.26] | −0.46 [−0.58, −0.35] | +0.13 [+0.03, +0.22] |
| Mention-swapped − neutral | −0.34 [−0.47, −0.22] | −0.56 [−0.68, −0.44] | −0.34 [−0.45, −0.24] |

Llama's "request lift", which the design asked to be reported without a decision, is visible in every arm on the failing call: both write lines sit above the neutral line (aligned − neutral +0.91, +1.30 and +1.78 in arms (i), (iii) and (ii); swapped − neutral +0.45, +0.63 and +0.76), while both question lines sit below it (about −0.5 in arms (i) and (iii)). The lift grows in the task-blind arm.

### 7.3 Per-line level changes between arms

Each cell is the paired per-item change of one line's score between two arms. A positive value means the line moved toward predicting success; for the failing call that is toward the wrong answer, for the witness toward the right one.

**Qwen3-8B, first token (nats; 95% CI).**

| Task line | Failing call, task-blind − main | Failing call, authorization − task-blind | Witness, task-blind − main | Witness, authorization − task-blind |
|---|---|---|---|---|
| aligned | +1.71 [+1.30, +2.10] | +1.24 [+0.85, +1.66] | −0.22 [−0.69, +0.20] | +3.02 [+2.65, +3.40] |
| swapped | +1.54 [+1.22, +1.86] | −0.70 [−0.95, −0.42] | +0.33 [−0.01, +0.68] | −0.36 [−0.65, −0.04] |
| neutral | +0.34 [+0.02, +0.65] | +0.35 [+0.03, +0.70] | −1.10 [−1.43, −0.78] | +2.06 [+1.71, +2.43] |
| mention-aligned | +0.72 [+0.34, +1.08] | +0.59 [+0.27, +0.92] | −0.97 [−1.30, −0.65] | +2.22 [+1.88, +2.58] |
| mention-swapped | +0.64 [+0.33, +0.94] | −0.36 [−0.64, −0.08] | −1.15 [−1.42, −0.87] | +0.49 [+0.17, +0.79] |

**Llama-3.1-8B, first token (nats; 95% CI).**

| Task line | Failing call, task-blind − main | Failing call, authorization − task-blind | Witness, task-blind − main | Witness, authorization − task-blind |
|---|---|---|---|---|
| aligned | +0.50 [+0.34, +0.67] | −0.52 [−0.65, −0.38] | +0.76 [+0.62, +0.89] | −0.77 [−0.91, −0.64] |
| swapped | +0.29 [+0.15, +0.44] | −0.86 [−0.98, −0.74] | +0.12 [−0.01, +0.27] | −1.07 [−1.21, −0.93] |
| neutral | +0.11 [−0.05, +0.28] | −0.99 [−1.14, −0.84] | +0.17 [+0.05, +0.29] | −1.14 [−1.26, −1.02] |
| mention-aligned | +0.12 [−0.00, +0.26] | −0.62 [−0.74, −0.50] | +0.07 [−0.04, +0.18] | −0.55 [−0.68, −0.43] |
| mention-swapped | +0.11 [−0.02, +0.25] | −0.92 [−1.04, −0.80] | −0.05 [−0.18, +0.07] | −0.93 [−1.07, −0.79] |

The design's prediction for arm (ii) against arm (iii) on the witness was that "the aligned line changes little, the four other lines fall". For Llama, all five lines fell, the aligned line slightly less than most. For Qwen the pattern is different: the aligned line rose by 3.02 nats, the mention-aligned and neutral lines by about 2, the mention-swapped line by 0.49, and only the swapped line fell (−0.36). So Qwen's positive control passes mainly because the authorization description raised its aligned witness prediction, not because it lowered the unrequested ones; and it raised the neutral line by 2.06 nats, although under the stated rule that call should now fail.

Between the task-blind arm and the main arm, both models raised the two write-request lines on the failing call more than the neutral and question lines (Qwen: aligned +1.71, swapped +1.54, others +0.34 to +0.72; Llama: aligned +0.50, swapped +0.29, others about +0.11). On the witness, Qwen's neutral and question lines fell by about 1 nat while the write lines stayed roughly level; Llama's aligned line rose by 0.76 and the rest barely moved.

### 7.4 The arm (ii) signature

The design predicted that under the authorization description the witness's naming component would be about 0, since neither question requests a write and the stated rule treats both alike. It was not: +5.03 [+4.28, +5.78] for Qwen (89% of items positive) and +0.47 [+0.36, +0.59] for Llama (73%), both larger than in the task-blind arm (+3.30 and +0.10). In both models the gap comes mostly from the question about the other order: for Qwen, mention-swapped − neutral is −4.46 while mention-aligned − neutral is +0.57; for Llama the two are −0.34 and +0.13. Under the authorization description, then, a question about another order lowers the witness's predicted success far more than a question about the target order does. The aligned write stays far above both (aligned − neutral +7.91 for Qwen, +1.82 for Llama). The witness's goal effect in arm (ii) was +12.59 [+11.56, +13.62] for Qwen and +1.32 [+1.16, +1.48] for Llama.

### 7.5 Yes/no

| Model, action | Quantity | Arm (i), main | Arm (iii), task-blind | Arm (ii), authorization |
|---|---|---|---|---|
| Qwen3-8B, failing call | Goal effect (aligned − swapped) | +7.75 [+6.95, +8.53] | +9.06 [+8.30, +9.81] | +9.42 [+8.69, +10.17] |
| Qwen3-8B, failing call | Swapped − neutral | −2.46 [−3.16, −1.77] | −2.81 [−3.53, −2.09] | −3.34 [−4.11, −2.54] |
| Qwen3-8B, failing call | Aligned − neutral | +5.29 [+4.75, +5.82] | +6.25 [+5.68, +6.84] | +6.09 [+5.57, +6.61] |
| Qwen3-8B, witness | Goal effect (aligned − swapped) | +13.26 [+12.39, +14.15] | +13.88 [+13.09, +14.71] | +14.91 [+14.08, +15.73] |
| Qwen3-8B, witness | Swapped − neutral | −6.72 [−7.48, −5.96] | −7.02 [−7.75, −6.25] | −7.86 [−8.55, −7.14] |
| Qwen3-8B, witness | Aligned − neutral | +6.54 [+6.07, +7.02] | +6.86 [+6.46, +7.28] | +7.05 [+6.63, +7.50] |
| Llama-3.1-8B, failing call | Goal effect (aligned − swapped) | −0.05 [−0.10, −0.01] | +0.02 [−0.02, +0.06] | +0.13 [+0.08, +0.18] |
| Llama-3.1-8B, failing call | Swapped − neutral | −0.07 [−0.13, −0.02] | −0.24 [−0.28, −0.20] | +0.11 [+0.06, +0.16] |
| Llama-3.1-8B, failing call | Aligned − neutral | −0.13 [−0.17, −0.08] | −0.22 [−0.26, −0.18] | +0.24 [+0.18, +0.29] |
| Llama-3.1-8B, witness | Goal effect (aligned − swapped) | −0.06 [−0.09, −0.03] | +0.04 [+0.02, +0.07] | +0.15 [+0.12, +0.18] |
| Llama-3.1-8B, witness | Swapped − neutral | −0.04 [−0.09, +0.00] | −0.22 [−0.26, −0.18] | +0.19 [+0.13, +0.25] |
| Llama-3.1-8B, witness | Aligned − neutral | −0.10 [−0.15, −0.06] | −0.17 [−0.21, −0.14] | +0.34 [+0.28, +0.39] |

The removed and persisting parts of the yes/no goal effect (no question lines are scored on yes/no, so it cannot be split into naming and request):

| Model, action | Removed part of the yes/no goal effect | Persisting part |
|---|---|---|
| Qwen3-8B, failing call | −1.31 [−1.64, −0.94], p 0.001 | +9.06 [+8.30, +9.81], p 0.001 |
| Qwen3-8B, witness | −0.62 [−1.03, −0.21], p 0.005 | +13.88 [+13.09, +14.71], p 0.001 |
| Llama-3.1-8B, failing call | −0.07 [−0.11, −0.04], p 0.001 | +0.02 [−0.02, +0.06], p 0.366 |
| Llama-3.1-8B, witness | −0.10 [−0.14, −0.07], p 0.001 | +0.04 [+0.02, +0.07], p 0.003 |

Qwen's yes/no goal effect, which the follow-up had shown to carry a judgement about the task, grew in the task-blind arm (from +7.75 to +9.06 nats on the failing call; removed part −1.31 [−1.64, −0.94]). Llama's yes/no effects are a tenth of a nat or less in every arm.

### 7.6 Per class

| Model, class | Request component, main | Removed | Persisting | Naming component, main | Removed | Persisting |
|---|---|---|---|---|---|---|
| Qwen3-8B, status | +1.60 [+1.03, +2.19] | +0.32 [−0.14, +0.79] | +1.29 [+0.76, +1.81] | +0.89 [+0.30, +1.44] | −0.42 [−0.77, −0.07] | +1.30 [+0.90, +1.73] |
| Qwen3-8B, unavailable | +2.99 [+2.33, +3.63] | −0.49 [−1.05, +0.02] | +3.48 [+2.78, +4.18] | +2.85 [+2.16, +3.55] | +0.26 [−0.12, +0.61] | +2.59 [+2.00, +3.18] |
| Llama-3.1-8B, status | +0.98 [+0.82, +1.15] | −0.22 [−0.36, −0.07] | +1.19 [+1.05, +1.33] | +0.30 [+0.10, +0.49] | +0.23 [+0.11, +0.35] | +0.07 [−0.08, +0.22] |
| Llama-3.1-8B, unavailable | +0.10 [+0.00, +0.20] | −0.18 [−0.29, −0.07] | +0.28 [+0.16, +0.40] | −0.46 [−0.56, −0.36] | −0.26 [−0.36, −0.16] | −0.20 [−0.31, −0.09] |

| Model, class | Removed part of the aligned half | Contribution of the swapped half | Removed part of aligned − neutral | Removed part of swapped − neutral |
|---|---|---|---|---|
| Qwen3-8B, status | −0.36 [−0.77, +0.07] | +0.68 [+0.34, +1.02] | −1.35 [−1.80, −0.91] | −1.25 [−1.61, −0.91] |
| Qwen3-8B, unavailable | −1.62 [−2.06, −1.20] | +1.13 [+0.76, +1.48] | −1.39 [−1.88, −0.93] | −1.16 [−1.65, −0.64] |
| Llama-3.1-8B, status | −0.49 [−0.65, −0.32] | +0.28 [+0.13, +0.42] | −0.62 [−0.84, −0.39] | −0.63 [−0.86, −0.38] |
| Llama-3.1-8B, unavailable | −0.26 [−0.38, −0.15] | +0.08 [−0.03, +0.19] | −0.17 [−0.30, −0.03] | +0.27 [+0.13, +0.43] |

For Qwen, both classes show a large persisting request and naming component, and no class shows a removal at the margin: the removed part of the request component is +0.32 [−0.14, +0.79] in the `status` class and −0.49 [−1.05, +0.02] in the `unavailable` class. The naming component grew a little in the `status` class (removed part −0.42 [−0.77, −0.07]). For Llama, the request component is mostly a `status`-class effect (+0.98 against +0.10), and its naming component has opposite signs in the two classes, +0.30 [+0.10, +0.49] for `status` and −0.46 [−0.56, −0.36] for `unavailable`, both CIs excluding 0. The pooled −0.08 is the average of these opposite effects. Amendment R1-S2 adds a sentence saying so to a model's reading; because Llama's reading is replaced by the gate text, the sentence does not appear in the output, and it is stated here instead.

### 7.7 Comprehension

Reported in full in §4.1.

### 7.8 Format

The median two-way mass (the probability on the two readout tokens together) is at least 0.977 for every arm and readout of both models (Qwen at least 0.9997), and no prompt in any group falls below 0.5, so the readouts measure a choice the models actually make. The analysis raises no format red flag.

### 7.9 Reproducibility of arm (i)

Not computed. The design asked for it "where the confirmatory scoring records are available to the analysis"; the analysis config names no reference run, so the block is absent from the metrics. The arm (i) values in §7.2 are close to the confirmatory ones, but they are on a 192-item subset, so that is orientation, not a reproduction check.

## 8. The predictions of design §5, checked

The design wrote down, before any data, what arm (iii) would show for Qwen's failing call under three hypotheses: relevance throughout (A), naming relevance-like and request authorization-like (AB), and authorization throughout (B). "≈" meant within the 0.72-nat margin.

| Quantity, Qwen3-8B failing call | Arm (i), this run | Predicted under A | Predicted under AB | Predicted under B | Observed, arm (iii) |
|---|---|---|---|---|---|
| Request component | +2.30 | unchanged | about 0 | about 0 | +2.39 |
| Naming component | +1.87 | unchanged | unchanged | about 0 | +1.95 |
| Goal effect | +4.17 | unchanged | about +1.81 | about 0 | +4.33 |
| Swapped − neutral | −2.70 | unchanged | about −1.98 | about 0 | −1.49 |
| Mention-swapped − neutral | −2.01 | unchanged | unchanged | about 0 | −1.71 |

Four of the five rows match A. The swapped − neutral row does not: the swapped line rose by 1.21 nats relative to the neutral line, beyond the margin, which A did not predict. It does not match AB either, because AB attributes that rise to the swapped line catching up with the aligned one, and the goal effect did not shrink: the aligned line rose by as much (aligned − neutral from +1.47 to +2.84). For the witness, A predicted that the goal effect persists (observed +9.76 to +9.21), which it does. For Llama, A predicted a request component of about +0.50 and a goal effect of about +0.47 in arm (iii); the observed values are +0.74 and +0.67, above the arm (i) values by more than Llama's margin, and AB or B predicted about 0. For arm (ii), the predicted positive control of at least 1 nat held for Qwen and not for Llama, and the predicted witness naming component of about 0 held for neither (§7.4).

## 9. Interpretation (exploratory reading)

**For Qwen3-8B, the request effect is not a belief that a clear description corrects.** This is what the frozen category says, and the surrounding numbers support it from several sides. The task-blind description was understood: Qwen answered every comprehension question correctly, and in particular switched its answer to K1 from a clear No under the confirmatory prompt (−2.96) to a confident Yes (+7.51). It can follow a stated backend, as the positive control shows. And yet the extra effect of a write request on the failing call did not change at all (removed −0.09, interval [−0.42, +0.26]); neither did the naming effect. This is the case the design singled out in §6.5: the model states the rule correctly and still shows the full persisting effect, so the effect is not carried by what the model says it believes about the backend. The earlier estimate that about a nat of Qwen's effect looked like an authorization belief came from the swapped line dropping when the sentence "the backend does not see the task" was removed. Here a similar sensitivity of the swapped line to the backend text appears (swapped − neutral rises by 1.21 nats in the task-blind arm), but the aligned line rises by as much, so the gap between requested and unrequested writes, which is what an authorization belief would create, is untouched.

**What the task-blind description did do.** On the failing call it raised Qwen's predicted success under both write-request lines by about 1.5 to 1.7 nats, more than under the neutral and question lines, which made its predictions less accurate (the call fails). On the witness it lowered the neutral and question lines by about 1 nat, again less accurate (the call succeeds). One reading consistent with this, untested here, is that the added demonstrations, all with write-request task lines and two of the three new outputs successes, taught "write requests tend to succeed" more than "the task does not matter". The design listed this risk in §11 (the task-blind arm is a package of a statement, two examples and about 2,000 tokens, and a change is not attributed to any one of them), and the R1-M2 split was built to catch copying of the demonstrated pairs. On the failing call the copying would have shown as a removal carried by the swapped half; instead both halves moved together and cancelled. On the witness the request component did shrink a little, and only through the swapped half, the copying pattern.

**What this shows and does not show.** It shows that, on these 192 items of two classes, Qwen's request effect on the failing call survives an explicit statement that the backend ignores the task, two minimal-pair demonstrations and the model's own correct answers to rule questions: it is not a correctable belief about the backend in the sense this design can test. It does not show what produces the effect (attention to the named order, a prior that requested actions succeed, consistency with the task's content); the design says only what it is not. It does not show that Qwen holds no authorization-like belief at all: under the confirmatory prompt it answered K1 as if it did (though not K3), and in arm (ii) its predictions separated a question about the target order from a question about another order, which the stated rule treats alike, while it answered K3 against that arm's text. Its beliefs as stated in yes/no answers and as expressed in predictions are not the same thing. Its stated answers moved with the description; its predictions' task effect did not.

**For Llama-3.1-8B, nothing is concluded.** See §6. Descriptively, its request effect grew rather than shrank under the task-blind description, which is the opposite of what an authorization belief predicts, but the design gives that pattern no interpretation, and the gates were set so that a model that does not demonstrably read the backend text gets no category.

**What would change these conclusions.** For Qwen, the category would become "authorization-like part shown" if a removal of at least 0.72 nats appeared with its interval above 0 and a Holm p below 0.01; the observed 95% interval reaches only +0.26 nats, so this would require a different population, a different prompt format or a different form of the task-blind description, not more items of the same kind. The categories are about the first token of the failing call on two classes in one domain, under the confirmatory prompt format. That format includes Qwen3's empty thinking-off block before the answer, which the pipeline audit (`notes/qwen_effect_audit_pipeline.md`) found to carry about half of the size of Qwen's ALFWorld effect; the decomposition compares task lines within that format and is not expected to depend on it, but it has not been checked in another format. For Llama, only a new frozen design could give a category (§6).

## 10. Limits and anomalies in the outputs

1. **No witness categories.** Design §7.1 asked for the witness decomposition "with the categories of §6.4 as labels". The frozen code (`model_block` in `wf/tau/auth.py`) computes the witness numbers with the primary block's function but never passes them to the category function, so no labels exist. This note does not compute them, because that would be a category the frozen analysis did not produce. The numbers are in §7.1.
2. **No reproducibility check of arm (i).** §7.9's check is conditional on a reference run, and the analysis config sets none, so it was not computed.
3. **The smoke record was not written into the design note before the full run.** Design §10 lists "the smoke run and its record in this note" before the full run, and §9.2 asks that the smoke run's standard deviations and projected half-widths be "recorded in this note before the full run". The `tau-auth` branch has no commit after the freeze, so they were not. The smoke analysis's outputs are reported here instead, and they changed no rule: the per-item SDs of the request component, its removed part and its persisting part were 4.10, 2.24 and 5.80 nats for Qwen and 0.46, 0.94 and 1.19 for Llama, on 7 items; the projection was 2.23 hours per shard and 7.90 GPU-hours in all.
4. **Runtime close to the limit.** The smoke run, on an L40S, projected 2.23 hours per shard for a 4-hour limit. Three of the four shards were placed on A40 nodes, which were slower per pass: about 1.5 times the smoke run's time on average over all four shards (mean seconds per pass over all shards 1.71 to 2.72, against 1.18 to 1.77 in the smoke run), and shard 0 ran 3 hours 47 minutes. Every shard finished in one job without resuming, and no score is missing or stale.
5. **A first smoke submission refused to score.** Job 7245671 stopped after 10 seconds with "items not verified: tau_verify has not passed" because the checkout lacked the build verification record. It produced no score, and its folder is kept as `runs/_failed_tau_auth_score_smoke_7245671`. The smoke run was then rerun unchanged as job 7246045.
6. **Mixed GPU types.** Shards 0–2 ran on A40s and shard 3 on an L40S. Shards are split by user and each shard scores every arm, line and action of its items with both models, so every within-item contrast compares scores from the same GPU and job.
7. **The task-blind arm is a package.** It adds a statement, two examples and about 2,000 tokens at once (design §11). A change between arms (i) and (iii) is not attributed to any one of them.
8. **Comprehension is measured on yes/no.** For Qwen, yes/no carries a task judgement; the 24 items repeat the same four questions in different contexts, so they test robustness across contexts, not 24 independent understandings.
9. **Scope.** 192 items of two classes, one domain, one prompt format, two models.

## 11. Provenance

| What | Value |
|---|---|
| Design and analysis | Frozen at tag `tau-auth-design-frozen-2026-11-22`, commit `f70b285ccefe50250522c0584aee1743d7166708` (branch `tau-auth`) |
| Cluster checkout | ICE `text-wm-fix38` at `f70b285`, clean (`git_dirty: false` in every run's inputs record) |
| Build | `tau_build`, pinned files `items.jsonl` `2c8ffeac…`, `natural_items.jsonl` `9bd6c3f7…`, `fewshot.json` `231607f7…`; scoring gated on `tau_verify` |
| Models | Qwen3-8B (revision `b968826…`, thinking off) and Llama-3.1-8B-Instruct (revision `0e9e39f…`), bf16, the confirmatory scorer |
| Smoke scoring | `tau_auth_score_smoke`, job 7246045, NVIDIA L40S, 25 minutes, 7 items, 840 records, `scoring.jsonl` sha256 `bf6d5a69314d6566…` |
| Smoke analysis | `tau_auth_smoke_analysis`, job 7247921 (CPU), `metrics.json` sha256 `90a31425e4aa63f9…` |
| Scoring shard 0 | job 7247964, NVIDIA A40, 3 h 47 min, 55 items, 29 users, 5,376 records, sha256 `938be384448fa59af2e93b0d8aaade9fb1287d8c52e202f9ce7b322a272a24cd` |
| Scoring shard 1 | job 7247965, NVIDIA A40, 3 h 04 min, 42 items, 29 users, 4,224 records, sha256 `1ac8db695b6424cb4a79ca387e056456aebb03a546256221a05da6eda21a3988` |
| Scoring shard 2 | job 7247966, NVIDIA A40, 3 h 18 min, 48 items, 29 users, 4,680 records, sha256 `9187e38cd8dfcab3da96016504a18a7a6dd2aaaf7e1f0417d5a8affe57aa4948` |
| Scoring shard 3 | job 7247967, NVIDIA L40S, 1 h 58 min, 47 items, 29 users, 4,728 records, sha256 `719c58cd118ef41f8859c13ac27ea22f430e816bf7d0838112ace2ed4234b75f` |
| Analysis | `tau_auth_analysis`, job 7247968 (CPU, 19 s), 2026-11-23 05:00; `metrics.json` sha256 `73ad25b9a14c28d0afbf15291c6caa8026eadbcee0fd09909f8783529d9e380b` (copied here as `tau_auth_metrics.json`) |
| Completeness | 192 items (96 `status`, 96 `unavailable`), 116 users, 24 comprehension items; 19,008 records, which is 9,504 per model as designed; 0 missing, 0 stale |
| Token check in the analysis | first-token check passes for both models; `{"` and `Error` are single tokens, and the authorization error opens with the same `Error` token as the real failures |

## 12. Extension (2026-11-25): Llama-3.2-3B-Instruct

### 12.1 What was run, and what was fixed beforehand

The frozen design was run on one more model, **Llama-3.2-3B-Instruct** (revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`). It was chosen because it is the only model besides the two reference models that the τ families extension (`notes/tau_families_results.md`) found interpretable on the first-token readout, and because, unlike Llama-3.1-8B, it has a naming component on the confirmatory prompt (+0.52 nats on the 856 primary items), so the control for a general suppression of task effects would have been available.

The extension is the section "Extension 2026-11-25" of [tau_auth_design.md](tau_auth_design.md) (E1 to E10), committed, reviewed once and tagged `tau-auth-ext-llama32-2026-11-25` (commit `a70e652`, branch `tau-auth`) before any score of this model in this experiment existed. Everything in sections 2 to 4 above applies unchanged: the same 192 items and 24 comprehension items, the same three backend descriptions (the confirmatory prompt, the task-blind description and the authorization description), the same five task lines, the same 48 passes per item, the same two gates, the same categories and readings, and the same inference (2,000 user-level bootstrap resamples, user-level sign-flip permutation tests with 10,000 permutations for the primary tests). Only three things were added for the new model, each fixed before its data:

1. **Its chat template**, handled as in the τ families runs: the Llama 3 header format with the date fixed at "26 Jul 2024", so that its prompts are the same text as Llama-3.1-8B's.
2. **Its margin**, by the frozen rule (a third of the model's confirmatory request component). On the 856 primary items under the confirmatory prompt its request component is +0.474 nats (τ families metrics, the same quantity that gave +2.17 and +0.50 for the reference models), so

$$\delta = 0.474 / 3 \approx 0.16 \text{ nats}.$$

3. **Its own Holm family** of its four primary tests (the removed and the persisting parts of the request and the naming component), at α = 0.01. The reference models' categories are not reopened.

The design note also fixed, before data, labels for readings that sit numerically close to a boundary (E6, from `numerical_sensitivity.md` rule 3), a stopping rule for the budget (E7) and the expectation that the gates would more likely fail than pass (E9).

All scoring ran in one environment: NVIDIA L40S GPUs only, the same software stack as the original runs and the τ families runs, bf16, one sequence per pass. A check written into the extension (design §7.9, which the original analysis skipped for lack of a reference) compared this run's arm (i) scores with the τ families scores of the same model on the same prompts: all 2,688 matching scores are identical to the bit (largest difference 0.0). So the extension renders and scores the model exactly as the τ families runs did.

### 12.2 The gates

**Comprehension (gate 1).** Each cell gives the correct answer, the number of the 24 items on which it was the more probable answer, and the mean log-odds of the correct answer in nats. Only the task-blind row gates.

| Llama-3.2-3B, arm | K1 | K2 | K3 | K4 | Arm passes |
|---|---|---|---|---|---|
| (i) main | Yes: 4/24, −0.27 | No: 24/24, +1.98 | No: 24/24, +1.02 | Yes: 24/24, +1.17 | no |
| (iii) task-blind | Yes: 24/24, +0.85 | No: 24/24, +2.42 | No: 10/24, +0.01 | Yes: 24/24, +1.56 | no |
| (ii) authorization | No: 24/24, +0.55 | No: 24/24, +2.96 | Yes: 24/24, +1.59 | Yes: 24/24, +1.42 | yes |

The model fails gate 1 on K3 alone. K3 asks whether, if the user's task were only a question about an order, a write on that order that breaks no rule would return an error; under the task-blind description the answer is No. The model's answer is a coin toss: correct on 10 of 24 items, with a mean log-odds of the correct answer of +0.01 nats, essentially zero. Under the confirmatory prompt it answers the same question correctly on all 24 items (+1.02), and under the authorization description, where the answer is Yes, also correctly on all 24 (+1.59). Its answer to K1 (a valid write on an order the task does not ask to change) flips from mostly wrong under the confirmatory prompt (4 of 24) to right on all 24 under the task-blind description, which is the change the description was written to make. The count of 10 is far from the cutoff of 20 items, so this gate outcome is not numerically fragile by the rule fixed in E6.

**Positive control (gate 2).**

| | Llama-3.2-3B |
|---|---|
| Positive control (extra aligned − swapped gap on the witness under the authorization description, compared with the task-blind one), nats | +0.41 [+0.34, +0.48] |
| Items with a positive value | 83.3% |
| Item-level SD, nats | 0.51 |
| Gate (at least 1 nat, CI above 0) | fails |

The model's witness predictions moved in the direction the authorization rule implies, on five items in six, but by about two fifths of the required 1 nat. The estimate is 0.59 nats below the threshold, far outside the fragility band.

**Result.** Both gates fail, so by the frozen rule both categories are **"inconclusive (gate)"**, with the frozen reading: "Inconclusive: the model is not shown to understand that the task-blind backend ignores the task (comprehension check of the task-blind arm); the model's predictions are not shown to follow the stated backend (positive control)." The smoke run on 7 items had shown the same picture (K3 3 of 7; positive control +0.38), and the full run went ahead as fixed in E8.

### 12.3 The decomposition, reported and not read

These numbers carry no category. They are reported because the design requires every estimate to be reported.

| Quantity, failing call, first token | Llama-3.2-3B |
|---|---|
| Request component, main prompt | +0.46 [+0.35, +0.57] |
| Request component, removed by the task-blind arm | +0.20 [+0.12, +0.29], raw p 0.0001, Holm p 0.0004 |
| Request component, persisting under the task-blind arm | +0.26 [+0.14, +0.38], raw p 0.0002, Holm p 0.0004 |
| Removed part of the aligned half | +0.24 [+0.15, +0.34] |
| Contribution of the swapped half to the removed part | −0.04 [−0.12, +0.05] |
| Removed part of aligned − neutral | +0.04 [−0.08, +0.16] |
| Removed part of swapped − neutral | −0.04 [−0.13, +0.06] |
| Naming component, main prompt | +0.54 [+0.42, +0.67] |
| Naming component, removed by the task-blind arm | −0.12 [−0.21, −0.04], raw p 0.0077, Holm p 0.0077 |
| Naming component, persisting under the task-blind arm | +0.67 [+0.50, +0.84], raw p 0.0001, Holm p 0.0004 |
| Removed share of the request component | +0.44 [+0.25, +0.64] |
| Removed share of the naming component | −0.23 [−0.38, −0.07] |
| Difference of the two shares | +0.66 [+0.37, +0.97] |

The arm (i) values on these 192 items are close to the τ families values on 856 items (request component +0.46 against +0.47, naming component +0.54 against +0.52). Neither class has arm (i) values of opposite sign: the request component is +0.63 [+0.47, +0.82] in the `status` class and +0.28 [+0.17, +0.40] in the `unavailable` class, the naming component +0.75 and +0.33.

The analysis also stores what the category rules would give if the gates were ignored (`ungated_categories`). For the request component that is "both: part authorization-like, part relevance-like" (the removed part, +0.20, is at least the margin with its interval above 0 and Holm p 0.0004; the persisting part, +0.26, likewise), and the aligned-half condition of amendment R1-M2 holds. For the naming component it is "relevance-like part shown; whether any of the component is removed is not determined". These are outputs of the frozen code, not categories of the design, and they decide nothing. Two of them are also numerically fragile by the rule fixed in E6: the request component's removed part (+0.20) is 0.04 nats above the margin of 0.16, inside the 0.08-nat band, and the naming component's removed part has an interval end (−0.21) 0.05 beyond the margin and a Holm p of 0.0077, inside the band of 0.005 to 0.02. The gate outcomes, on which the categories rest, are not near their boundaries, so the cross-environment rescore that E6 names for fragile decisive numbers is not needed.

**Exploratory reading of where the ungated removal comes from.** The goal effect itself (aligned − swapped) hardly changed between the confirmatory and the task-blind description: +1.00 [+0.83, +1.18] and +0.93 [+0.73, +1.13]. Since the request component is the goal effect minus the naming component, its apparent removal (+0.20) mirrors the naming component's growth (from +0.54 to +0.67). Both come mainly from one line: the question about the target order (mention-aligned) rose relative to the neutral line, from −0.28 [−0.38, −0.18] to −0.08 [−0.20, +0.05]. The aligned write line did not move relative to the neutral line (removed part of aligned − neutral +0.04 [−0.08, +0.16]). An authorization belief predicts the opposite pattern: its removal should show as a drop of aligned − neutral, the aligned write losing its advantage over lines that request nothing (design amendment R1-M2). So even ignoring the gates, this model's numbers do not look like the removal of an authorization belief. They look like a change in how the model treats a bare question about the target order. This reading is post hoc and decides nothing.

### 12.4 Secondary analyses (design §7)

**Per arm, failing call, first token (nats; 95% CI).**

| Quantity | Arm (i), main | Arm (iii), task-blind | Arm (ii), authorization |
|---|---|---|---|
| Goal effect (aligned − swapped) | +1.00 [+0.83, +1.18] | +0.93 [+0.73, +1.13] | +1.30 [+1.10, +1.51] |
| Naming component | +0.54 [+0.42, +0.67] | +0.67 [+0.50, +0.84] | +0.81 [+0.69, +0.94] |
| Request component | +0.46 [+0.35, +0.57] | +0.26 [+0.14, +0.38] | +0.49 [+0.35, +0.63] |
| Aligned half (aligned − mention-aligned) | +2.38 [+2.27, +2.48] | +2.14 [+2.02, +2.25] | +2.42 [+2.30, +2.56] |
| Swapped half (swapped − mention-swapped) | +1.92 [+1.80, +2.04] | +1.88 [+1.76, +1.99] | +1.93 [+1.80, +2.08] |
| Aligned − neutral | +2.10 [+1.97, +2.23] | +2.06 [+1.91, +2.22] | +2.64 [+2.47, +2.81] |
| Swapped − neutral | +1.10 [+0.93, +1.26] | +1.14 [+1.00, +1.27] | +1.34 [+1.19, +1.49] |
| Mention-aligned − neutral | −0.28 [−0.38, −0.18] | −0.08 [−0.20, +0.05] | +0.22 [+0.12, +0.32] |
| Mention-swapped − neutral | −0.82 [−0.93, −0.71] | −0.74 [−0.87, −0.62] | −0.59 [−0.68, −0.50] |

Both write requests lift predicted success well above the neutral line in every arm (aligned by about 2 nats, swapped by about 1), and the bare questions sit below it, as in Llama-3.1-8B and in the τ families run of this model. On a call that fails whatever the task, all of this is error.

**Witness (a working call on the same order), first token.** Goal effect +1.04 [+0.89, +1.21] (arm (i)), +0.95 [+0.77, +1.14] (arm (iii)), +1.36 [+1.19, +1.56] (arm (ii)). Its request component is +0.27 [+0.14, +0.40] under the confirmatory prompt; the task-blind description removes +0.16 [+0.06, +0.26] of it, through the aligned half (+0.38 [+0.28, +0.47]) with a drop of aligned − neutral (+0.22 [+0.11, +0.34]), while the swapped half moves the other way (−0.22 [−0.32, −0.11]). Its naming component is +0.77 and persists (+0.84). As in §7.1, no category is computed for the witness.

**Per-line level changes between arms, first token (nats).** Moving from the confirmatory to the task-blind description raised every line on both calls by about 0.7 to 1.05 nats (failing call: aligned +0.81, swapped +0.88, neutral +0.84, mention-aligned +1.04, mention-swapped +0.92), a level shift toward success that the within-arm contrasts do not see. Moving from the task-blind to the authorization description lowered every line on the witness, the aligned line least (aligned −0.70, swapped −1.11, neutral −1.52, mention-aligned −1.08, mention-swapped −1.27, all intervals below 0). The design predicted "the aligned line changes little, the four other lines fall"; the model moved in that direction but shifted the aligned line too, which is why the positive control is positive but small.

**The arm (ii) signature.** Under the authorization description the witness's naming component was predicted to be about 0; it is +1.03 [+0.91, +1.15], as it was clearly non-zero for both reference models.

**Yes/no (not read for this model).** Llama-3.2-3B fails the τ families competence rule on yes/no (it answers No to nearly every call), so these numbers describe only how a near-constant No answer shifts. The yes/no goal effect on the failing call is +0.30 [+0.22, +0.39], +0.46 [+0.36, +0.58] and +0.41 [+0.32, +0.51] in arms (i), (iii) and (ii).

**Per class.** Under the task-blind description the request component's removed part is +0.26 [+0.11, +0.40] in the `status` class and +0.15 [+0.05, +0.24] in the `unavailable` class; the naming component grows in the `status` class (removed part −0.39 [−0.51, −0.27]) and shrinks a little in the `unavailable` class (+0.15 [+0.04, +0.25]).

**Format.** Median two-way mass 0.969 to 1.000 for every arm and readout; no prompt below 0.5; no red flag. The in-run token report passes and records the fixed date.

**Holm over twelve tests (descriptive, fixed in E4).** If this model's four primary tests and the reference models' eight published raw p-values were adjusted together, every p-value that is 0.0002 or smaller becomes 0.0012 and this model's naming removal becomes 0.0385 (from 0.0077). No category of any of the three models changes: the reference models' material parts stay below 0.01, and this model's categories are gated.

### 12.5 Predictions checked (design E9)

The extension predicted, before data, that gate 1 would more likely fail than pass (probability of passing about 0.25), reasoning that a model leaning toward No would fail the questions whose correct answer is Yes (K1 and K4). The outcome matched the prediction, but not its reason: K1 and K4 were answered correctly on all 24 items, and the failure was on K3, whose correct answer is No, where the model was indifferent. The positive control was given about 0.4 to pass and failed at +0.41 nats. The predicted most likely outcome, "inconclusive (gate)" for both components, is what occurred.

### 12.6 What this shows and does not show

**It shows** that the frozen design cannot certify, for Llama-3.2-3B, either of its two preconditions: that the model takes in the task-blind description (it is indifferent on one of the four rule questions) and that its predictions follow a stated backend (they move by about 0.4 nats where the design asks for 1). That is the same outcome as Llama-3.1-8B, through a different comprehension question (K3 here, K2 there) and a similar positive control (+0.41 here, +0.30 there). Of the three models now run, only Qwen3-8B passes both gates, and its result ("relevance-like" for both components) remains the experiment's only read result.

**It does not show** that Llama-3.2-3B holds an authorization belief, or that it lacks one. The ungated numbers, read with every caveat above, do not resemble the removal an authorization belief predicts: the goal effect did not shrink, and the apparent removal of the request component is the naming component growing through the bare question about the target order, not the requested write losing its advantage. Like Llama-3.1-8B's ungated labels, they are reported and not interpreted.

**What would change it.** As for Llama-3.1-8B (§6), only a new, separately frozen design could give these models a category, for example comprehension questions on which the Llama models are not indifferent, checked on a smoke run before freezing, and a positive-control threshold scaled to each model's spread of scores. Any such change made now would be a post hoc change of a frozen rule after seeing the result.

### 12.7 Limits and operational events

1. **The numerical floor of this model was not measured.** Llama-3.1-8B's floor stands in for it (E6), and for the removed parts the margin is only about four times that floor. No cross-environment rescore was run; E6 required one only if a decisive number were fragile, and none is.
2. **192 items of two classes, one domain, one prompt format.** One more Meta model does not separate size, distillation and post-training, and does not extend the result to another developer.
3. **The first smoke submission scored nothing.** Job 7328769 waited on "BadConstraints" (it was eligible for a partition without L40S nodes) and then refused to score because the build's verification record had been copied from the original checkout, where it was made at another commit. The items were rebuilt and verified at the tag (CPU job 7330380); the rebuild reproduced all three pinned files. The smoke run was then repeated unchanged (job 7330545). Every scoring job was submitted with the partitions that hold L40S nodes and a shorter time limit than the script's 4 hours (1 hour for the smoke, 1.5 hours per shard), so that it could start sooner; no shard came near its limit. None of this changes a prompt, item, job or rule (design E10).
4. **GPU time.** Smoke 0.14 hours, four shards 1.87 hours (32, 26, 27 and 27 minutes), the failed submission 22 seconds: about 2.0 GPU-hours in all, as projected.

### 12.8 Provenance

| What | Value |
|---|---|
| Design | Frozen design at `tau-auth-design-frozen-2026-11-22` (`f70b285`) plus the extension section, tagged `tau-auth-ext-llama32-2026-11-25` (commit `a70e652e93762b6a331768eb56083bb5ce17f160`, branch `tau-auth`); one independent review at `8cced24` (no must-fix item); smoke record committed at `e55064e` before the full run |
| Cluster checkout | ICE `text-wm-tauauth-l32` at `a70e652`, clean; build rebuilt and verified there (CPU job 7330380): `items.jsonl` `2c8ffeac…`, `natural_items.jsonl` `9bd6c3f7…`, `fewshot.json` `231607f7…` |
| Model | Llama-3.2-3B-Instruct, revision `0cb88a4…`, bf16, date fixed at "26 Jul 2024", default (SDPA) attention, the confirmatory scorer |
| Token check | CPU job 7328347: PASS on all 9,504 prompts, 0 problems; system messages `7cc04248…`, `11ede0a6…`, `2c621c0e…` unchanged |
| Smoke scoring | `tau_auth_l32_score_smoke`, job 7330545, NVIDIA L40S, 8 min 34 s, 7 items, 420 records, `scoring.jsonl` sha256 `1915093d5b8e03cc…` |
| Smoke analysis | `tau_auth_l32_smoke_analysis`, job 7330785 (CPU), `metrics.json` sha256 `e935c2747a7cb03a…` |
| Scoring shard 0 | job 7330880, NVIDIA L40S, 32 min, 55 items, 29 users, 2,688 records, sha256 `3b1c2eab165708b936c8f408a016f5298b2b22c46b0f3c22cf4237447b06f0ab` |
| Scoring shard 1 | job 7330881, NVIDIA L40S, 26 min, 42 items, 29 users, 2,112 records, sha256 `ed2610ba00f0d2843f50173388d90229fa3f5afc75569b84455511fdc932dce4` |
| Scoring shard 2 | job 7330882, NVIDIA L40S, 27 min, 48 items, 29 users, 2,340 records, sha256 `babc75a073507e4cc2488fe9a138383c5cfa8df094fe6abe5b52e84a16387d81` |
| Scoring shard 3 | job 7330883, NVIDIA L40S, 27 min, 47 items, 29 users, 2,364 records, sha256 `a8fd375c3dab253b87a99bb6728e28eb81eb2027ff6a980b9665f8368d4ddd8c` |
| Analysis | `tau_auth_l32_analysis`, job 7331800 (CPU, 26 s), 2026-11-25 21:32; `metrics.json` sha256 `3713c757608f309c6b27fa5e44243e0e627b52e61374fca9097c27847be43744`, copied unchanged as [tau_auth_l32_metrics.json](tau_auth_l32_metrics.json) |
| Completeness | 192 items (96 `status`, 96 `unavailable`), 116 users, 24 comprehension items; 9,504 records; 0 missing, 0 stale |
| Reproducibility of arm (i) | against the τ families scoring of this model (`text-wm-fix44/runs/tau_family_score_llama3.2-3b`): 2,688 matching scores, all identical |
