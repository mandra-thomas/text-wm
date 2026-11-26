# τ-bench retail: relevance bias or a belief in request-matching authorization? Design of a discriminating experiment (written 2026-11-22, before any code or score)

This note fixes the design of an experiment on the τ-bench retail items of the replication (`notes/tau_design.md`).
It is committed before any code for it is written and before any model is scored on it. Nothing in it is scored
until an independent review has passed and the design is frozen at an annotated git tag. Anything changed after
this commit goes in a dated "Changes" section at the end, with its reason.

## 0. What was known when this was written

The confirmatory result (`notes/tau_results.md`, corrected) and the exploratory analysis (`notes/tau_dx.md`, branch
`tau-dx`) had been read, in aggregate. The numbers below come from them. No item-level score of the items chosen
in §3 was opened for this design; the item subset is fixed by a hash of the item id (§3.1), not by any score.

On the 856 primary items (status-matched main population, 175 users), first token, failing call:

| Quantity | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Aligned − swapped | +3.98 [3.61, 4.35] | +0.47 [0.41, 0.54] |
| Aligned − neutral | +1.42 | +0.92 |
| Swapped − neutral | −2.56 | +0.45 |
| Question about the order, aligned − swapped (the "mention pair") | +1.81 [1.60, 2.02] | −0.03 [−0.07, 0.02] |
| Question about the other order − neutral | −1.98 | −0.52 |
| Write request minus question (the two aligned − swapped effects) | +2.17 [1.92, 2.43] | +0.50 [0.45, 0.54] |
| Witness (a working call on the same order), aligned − swapped | +9.45 | +0.38 |
| Failing call minus witness | −5.47 [−5.88, −5.05] | +0.09 [0.04, 0.14] |
| Prompt without "the backend does not see the task", swapped line, change | −1.14 [−1.26, −1.01] | −0.35 [−0.41, −0.29] |

The exploratory analysis concluded that Qwen's effect is mainly a task-relevance or task-consistency bias (a bare
question about another order, with nothing requested, already lowers predicted success by about 2 nats), with about
a nat consistent with a belief that the backend refuses writes the user did not request (the drop of the swapped
line when the statement that the backend does not see the task is removed). It could not split the two for the
extra +2.17 nats that the write request adds over the bare question: both explanations predict it. Llama's effect
is small, request-driven and class-dependent; its order-naming part without a request is zero.

## 1. The question

### 1.1 Two explanations

A world model here predicts the retail backend's reply to an agent's tool call. The real backend (tau2-bench retail,
pinned commit) never consults the user's task: a call's output is a function of the call and the database only.
Review 3's task-invariance gate replayed items in a fresh environment and confirmed that every scored label is fixed
by the state whatever the task line says. **So both explanations below are prediction errors with respect to the
real backend.** Any difference between task lines in a model's predicted outcome is an error. The experiment asks
which kind of error it is.

- **(a) Task-relevance or task-consistency bias.** The model predicts success more readily for a call that fits the
  user's task (names the requested order, does the requested kind of thing) and failure for one that does not. On
  this reading the bias is not a belief about what the backend checks. So it should not go away when the prompt
  states, and shows by examples, that the backend ignores the task.
- **(b) A belief in request-matching authorization.** The model believes that the backend checks each write against
  the user's request and refuses writes the user did not ask for. This is a wrong belief about the system being
  modelled. The main prompt already says the backend does not see the task, so the belief, if present, survives
  that statement. A stronger statement together with demonstrations should correct it.

### 1.2 Operational definitions

The experiment varies only how the backend is described and demonstrated, and keeps the items, the task lines, the
calls, the histories and the real outcomes fixed.

- The part of a task effect that **disappears** when the prompt states and demonstrates a task-blind backend is
  called **authorization-like**, reading (b).
- The part that **persists** under that prompt is called **relevance-like**, reading (a).

This is an operational split, and it has one known limit. A bias that is not a belief about the backend, but that an
explicit instruction and a demonstration can suppress anyway, would be classed as authorization-like. The design
controls for that with the naming component (§4): a general suppression of task effects would remove the naming
effect too, whereas an authorization belief concerns writes the user requested, and a bare question requests none.

### 1.3 What is unresolved, and what this experiment targets

Write the failing call's aligned − swapped effect as the sum of two components, a naming component (the
aligned − swapped effect of the bare-question lines, which name the same order and product but request nothing) and
a request component (what the write request adds). The request component is the quantity the earlier data could not
attribute: +2.17 nats for Qwen and +0.50 for Llama. **The primary target is the request component on the failing
call.** The naming component is decomposed the same way and serves as the control for a general suppression.

## 2. Arms

Three system-text variants. Everything else (items, task lines, histories, calls, cues, readouts, the scoring code,
the per-item few-shot order) is identical, and the differences are checked by prompt diffs (§8).

| Arm | Name in code | What the backend section says | Few-shot examples |
|---|---|---|---|
| (i) | `main` | The confirmatory main prompt, byte for byte: the backend sees only the call and the database, not the user's messages or task, and checks the listed rules and nothing else | The confirmatory 4 (a balanced 2 × 2: the task names the call's own order or another order, and the call succeeds or fails) |
| (ii) | `auth` | The backend sees the user's task too, checks the listed rules and then a request rule: a write that passes the rules is executed only if the task asks for that change to that order; otherwise it returns a fixed authorization error | 6: the 4 of arm (i), with the other-order success replaced by the authorization error, plus two own-order copies (below) |
| (iii) | `blind` | Arm (i)'s text unchanged, plus a paragraph stating that the task never changes a call's output, that a write on an order the task does not mention, or under a task that asks for no change, is executed exactly as if the user had asked for it | 6: the 4 of arm (i), plus the same two own-order copies |

### 2.1 Exact texts

The system message of every arm is: the instruction (unchanged), then "# How the backend checks write calls", the
arm's lead, the backend's rules (unchanged, `BACKEND_RULES`), the arm's closing paragraph (none in arm (i)), then the
agent policy and the tool schemas (unchanged).

**Lead, arms (i) and (iii)** (the confirmatory text, unchanged): "The policy below is written for the agent. The
backend sees only each call's name and arguments and the current database: it does not see the user's messages or
task, and it does not check the agent's conduct (authentication, confirmation with the user). It does check the
following and nothing else: a write call that breaks any of them returns an error and changes nothing."

**Lead, arm (ii):** "The policy below is written for the agent. The backend sees each call's name and arguments, the
current database and the user's task, and it does not check the agent's conduct (authentication, confirmation with
the user). It does check the following, then the request rule below, and nothing else: a write call that breaks any
of them returns an error and changes nothing."

**Closing paragraph, arm (iii):** "The user's task never changes a call's output: the backend does not compare a call
with the user's request. A write call that changes an order the user's task does not ask to change, or that is made
when the task asks for no change (only a question about an order, or general help with the account), is executed
exactly as it would be if the user had asked for it. A write call that breaks one of the rules above returns that
rule's error whatever the task."

**Closing paragraph, arm (ii):** "Request rule, checked after the rules above: a write call is executed only if the
user's task asks for that change to that order. A write call that changes an order the user's task does not ask to
change, or that is made when the task asks for no change (only a question about an order, or general help with the
account), returns "Error: The user did not request this change" and changes nothing. A write call that breaks one of
the rules above returns that rule's error whatever the task."

The two closing paragraphs are parallel sentence for sentence and differ only where the backends differ. The
authorization error, "Error: The user did not request this change", is not an error the real backend can produce;
it starts with "Error", so the first-token readout is the same in every arm.

### 2.2 Few-shot examples

The confirmatory build's four examples (`runs/tau_build/fewshot.json`, sha256 `231607f7…`) come from four users
excluded from every test set. Each example stores `task` (the line used in arm (i)) and `task_own` (the line naming
the example's own order; built for the ALFWorld-matched arm). Call them E1 to E4 in the design order of arm (i):

- E1, own order, success: change the shipping address of a pending order;
- E2, other order, failure: cancel a pending order with the reason "found a better price" ("Error: Invalid reason"),
  while the task asks to cancel another of the user's orders;
- E3, other order, success: return an item of a delivered order to its original payment method, while the task
  asks to return an item of another order;
- E4, own order, failure: change the payment method to the one already used.

Arms (ii) and (iii) add two own-order copies, each a **minimal pair** with an other-order example: the same user,
history and call, with the task line replaced by `task_own`:

- E2' = E2 with the task asking to cancel this very order; output "Error: Invalid reason" (in both arms: the reason
  rule is broken whatever the task);
- E3' = E3 with the task asking to return this very item; output the real success JSON (in both arms: requested and
  valid).

E3's output is the real success JSON in arm (iii) and the authorization error in arm (ii). Nothing else differs
between the two arms' examples. Design order in arms (ii) and (iii): E1, E2, E2', E3, E3', E4. Each pair is adjacent,
other-order first. Every arm reverses its order for the items whose h(item id, "fewshot-order") is odd, as in the
confirmatory design, so one item has the same order in every job and every arm.

**Balance.** In arm (iii) the task's shape predicts nothing about the outcome: own-order examples succeed in 2 of 4
(E1, E3'), other-order examples in 1 of 2 (E3). Arm (i) has 1 of 2 and 1 of 2. In arm (ii) the examples show each of
the three kinds the design requires: a matching call that succeeds (E1, E3'), a non-matching call rejected for
authorization (E3), and calls that fail a database rule whatever the task (E2 and E2' as a pair, and E4).

**Why arm (iii) adds to arm (i) instead of replacing it.** Arm (i) already shows one other-order success (E3) and
already states that the backend does not see the task; Qwen's swapped penalty survived both. Arm (iii) therefore
keeps everything in arm (i) and adds two things: an explicit statement that the task never changes the output,
including for tasks that request no change, and two minimal pairs that show the same call giving the same output
under a task that requests it and a task that does not. So arm (iii) − arm (i) measures what a stronger statement and
a direct demonstration of task-independence remove. Arm (ii) − arm (iii) changes only the stated rule and the one
example output that the rule decides.

**What the arms do not hold fixed.** Arms (ii) and (iii) are about 1,600 tokens longer than arm (i) (two more
examples and the closing paragraph). Arm (ii) shows three errors in six examples and arm (iii) two, so arm (ii) may
shift every line toward "Error". These are level shifts; every decision contrast is a difference between task lines
within an arm (§4), which a level shift does not change.

### 2.3 What the described backend implies in each arm

The real outcome of every scored call is fixed: the failing call fails, the witness succeeds, under every task line
and in every arm. Arm (ii) describes a counterfactual backend. Under its description the failing call still fails
under every line (the database rules are checked first), but the witness succeeds only under the aligned line, and
returns the authorization error under the swapped, neutral and both question lines. Arm (ii) is therefore never read
as a test of accuracy. It is a **positive control**: it shows whether the model's predictions follow the stated
backend at all, and it shows what an explicit authorization belief looks like in this model's scores.

## 3. Items, task lines, readouts and jobs

### 3.1 Items

**Subset.** 192 primary items: the first 96 `status` items and the first 96 `unavailable` items of the primary
population in h(item id, "auth") order. The hash is the design's (`wf.tau.items.h`). On the pinned build these are
192 items from 116 users, effective cluster size

$$m_{\text{eff}} = \frac{\sum_u n_u^2}{n} = 2.03 .$$

**Why these two classes.** They are 95% of the primary population and the two prespecified classes. Their failing
tools (`modify_pending_order_items`, `exchange_delivered_order_items`) are used by no few-shot example, whereas the
other two classes' tools are (returns: E3 and E3'; payment changes: E4), which would let a model match an example's
cell by tool (confirmatory design B6). Both classes have a witness, so the positive control can be read on every item.

**Why a subset.** The budget (§9) allows about 190 items for three arms with both actions, five task lines and two
readouts. The power analysis (§9.2) says what this size can and cannot decide.

**Comprehension items.** The first 12 items of each class among the 192 in h(item id, "auth-comp") order: 24 items
from 24 users.

**Smoke items.** The first 3 items of each class among the 192 in h(item id, "auth") order (6 items), scored in a
separate run for format and timing only (§10).

**Build.** The confirmatory build, unchanged: `items.jsonl` `2c8ffeac…`, `natural_items.jsonl` `9bd6c3f7…`,
`fewshot.json` `231607f7…`, copied into the scoring checkout and checked against these pins. No item or example is
rebuilt; E2' and E3' are rendered from the stored examples' `task_own` lines.

### 3.2 Task lines

As built and as in the confirmatory design: aligned (the target order and product, with the write request),
swapped (another order of the same user with the same status, and its product, with the same request), neutral
("The user wants help with their account."), mention-aligned ("The user has a question about the {product} in order
{target}.") and mention-swapped (the same with the comparison order and its product).

### 3.3 Readouts

- **First token (primary):** the log-odds of the two possible openings of the backend's reply,

$$L_1 = \log P(\texttt{\{"}) - \log P(\texttt{Error}),$$

  at the first answer position after the chat template's generation prompt (Qwen3 with thinking off).
- **Yes/no (secondary):** log P(Yes) − log P(No) after the confirmatory reply-anchored question, unchanged: "Question:
  will the backend execute this call without returning an error? Answer Yes if its output is JSON and No if its
  output is a line starting with "Error: "." The exploratory analysis showed that Qwen's yes/no carries a task
  judgement beyond its predicted output, so it is never decisive here.

### 3.4 Jobs per item and arm

| Action | First token | Yes/no |
|---|---|---|
| Failing call | aligned, swapped, neutral, mention-aligned, mention-swapped (5) | aligned, swapped, neutral (3) |
| Witness | aligned, swapped, neutral, mention-aligned, mention-swapped (5) | aligned, swapped, neutral (3) |

That is 16 forward passes per item and arm, 48 per item. The witness's two question lines are new: the confirmatory
run scored them on the failing call only. The failing call's yes/no question lines and the full replies of the
confirmatory design are not scored: the first is secondary and carries Qwen's task judgement, the second is
dominated by copying (`tau_dx.md` §5), and both cost passes that the primary contrast needs.

### 3.5 Comprehension check

On the 24 comprehension items, in every arm, the failing call's prompt under the neutral task line with the final cue
replaced by each of four questions, scored as log P(Yes) − log P(No):

- **K1 (a valid write on another order):** "Question: about this backend in general: if a write call broke none of
  the rules listed for its tool but changed an order that the user's task does not ask to change, would the backend
  execute it without returning an error? Answer Yes or No.\nAnswer:"
- **K2 (a broken rule):** "Question: about this backend in general: if a write call did what the user's task asks
  but broke one of the rules listed for its tool, would the backend execute it without returning an error? Answer
  Yes or No.\nAnswer:"
- **K3 (a question only):** "Question: about this backend in general: if the user's task were only a question about
  an order, and a write call changed that order and broke none of the rules listed for its tool, would the backend
  execute it without returning an error? Answer Yes or No.\nAnswer:"
- **K4 (a valid requested write):** "Question: about this backend in general: if a write call did what the user's
  task asks and broke none of the rules listed for its tool, would the backend execute it without returning an
  error? Answer Yes or No.\nAnswer:"

| Question | Correct in arms (i) and (iii) | Correct in arm (ii) |
|---|---|---|
| K1 | Yes | No |
| K2 | No | No |
| K3 | Yes | No |
| K4 | Yes | Yes |

K1 and K3 are the distinction the arms are about; K2 and K4 have the same answer in every arm and catch a general
lean toward Yes or No. **An arm passes** for a model when, on each of the four questions, the correct answer has the
higher probability on at least 80% of the 24 items. In arm (i), K1 and K3 are also a direct question about the belief
in authorization under the confirmatory prompt; their answers there are reported and never gate anything.

4 questions × 3 arms × 24 items = 288 passes per model.

### 3.6 Models

Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, at the pinned revisions of `wf.constants.REVISIONS`
(`b968826…` and `0e9e39f…`), bf16, the confirmatory `LogoddsScorer`.

## 4. Quantities

For item $i$, arm $A$, action $a$ (the failing call or the witness) and task line $c$, write $L_{i}^{A,a}(c)$ for the
first-token score. The per-item quantities, in each arm and for each action, are

$$G = L(\text{al}) - L(\text{sw}), \qquad M = L(\text{ma}) - L(\text{ms}), \qquad W = G - M = \big[L(\text{al}) - L(\text{ma})\big] - \big[L(\text{sw}) - L(\text{ms})\big],$$

$$S = L(\text{sw}) - L(\text{ne}),$$

where al, sw, ne, ma and ms are the aligned, swapped, neutral, mention-aligned and mention-swapped lines. $G$ is the
goal effect, $M$ the naming component, $W$ the request component and $S$ the swapped penalty. For a component $X$
($W$ or $M$) the two parts compared across arms are, per item,

$$Z_X = X^{\text{main}} - X^{\text{blind}}, \qquad R_X = X^{\text{blind}},$$

so that $X^{\text{main}} = Z_X + R_X$ exactly: $Z_X$ is the part removed by the task-blind arm (authorization-like)
and $R_X$ the part that persists (relevance-like). The positive control is

$$PC = G^{\text{auth,wit}} - G^{\text{blind,wit}},$$

the extra aligned − swapped effect on the witness when the backend is described as enforcing the request rule
instead of ignoring the task. Every quantity is a mean over items of a per-item difference, so it is unaffected by a
shift of every line's level within an arm.

## 5. Predictions, written before any data

Three hypotheses, each about where the arm (i) effect comes from:

- **A**, relevance throughout: both the naming and the request component are reading (a);
- **AB**, naming is (a), the request is (b): the hypothesis the exploratory analysis could not exclude;
- **B**, authorization throughout: even the naming component is a soft authorization belief (naming an order in a
  question makes a write on it look more authorized).

The arm (i) column gives the confirmatory values (856 items) for orientation; the decision uses this run's arm (i)
on the 192 items. "≈" means within the margin of §6.2.

**Qwen3-8B, first token, failing call:**

| Quantity | Arm (i), confirmatory | Arm (iii) under A | Arm (iii) under AB | Arm (iii) under B |
|---|---|---|---|---|
| Request component $W$ | +2.17 | ≈ +2.17 (Z ≈ 0) | ≈ 0 (Z ≈ +2.17) | ≈ 0 |
| Naming component $M$ | +1.81 | ≈ +1.81 | ≈ +1.81 | ≈ 0 |
| Goal effect $G$ | +3.98 | ≈ +3.98 | ≈ +1.81 | ≈ 0 |
| Swapped − neutral $S$ | −2.56 | ≈ −2.56 | ≈ −1.98 (the question-about-the-other-order level) | ≈ 0 |
| Question about the other order − neutral | −1.98 | ≈ −1.98 | ≈ −1.98 | ≈ 0 |

**Qwen3-8B, witness (new lines, so the arm (i) values of $M$ and $W$ are first measured here):** under A the
witness's $G$ (+9.45 in arm (i)) persists in arm (iii); under AB it falls to about its own naming component; under B
to about 0.

**Llama-3.1-8B, first token, failing call:**

| Quantity | Arm (i), confirmatory | Arm (iii) under A | Arm (iii) under AB or B |
|---|---|---|---|
| Request component $W$ | +0.50 | ≈ +0.50 | ≈ 0 |
| Naming component $M$ | −0.03 (absent) | absent | absent |
| Goal effect $G$ | +0.47 | ≈ +0.47 | ≈ 0 |

Llama has no naming component, so AB and B make the same prediction for it, and it offers no control for a general
suppression (§6.4). Llama's "request lift" (both write lines above neutral, +0.92 and +0.45, while both question
lines are below it) fits neither explanation in its plain form; under A it persists in arm (iii). It is reported
(aligned − neutral and swapped − neutral per arm) without a decision.

**Arm (ii), any hypothesis, for a model whose predictions follow the stated backend:** on the witness, the swapped,
neutral and both question lines fall toward "Error" relative to arm (iii) while the aligned line does not, so
$PC \ge 1$ nat; the witness's naming component in arm (ii) is about 0 (neither question authorizes a write). On the
failing call every line should predict an error. Arm (ii) makes no prediction that separates A from AB or B: a model
that follows explicit rules amplifies the swapped penalty under either. It is the positive control and the template
of an explicit authorization belief, nothing more.

**Yes/no:** the same predictions in sign. Qwen's yes/no task judgement (about +3.9 nats beyond its predicted output)
is relevance-like if it persists in arm (iii).

## 6. Decision rules

### 6.1 Gates (per model; each must pass, or every primary category of that model becomes "inconclusive" with the
reason)

1. **Comprehension of the task-blind arm:** arm (iii) passes the comprehension check (§3.5). Otherwise: "inconclusive:
   the model is not shown to understand that the task-blind backend ignores the task".
2. **Positive control:** PC is at least 1 nat with its 95% CI above 0. Otherwise: "inconclusive: the model's
   predictions are not shown to follow the stated backend".

The comprehension of arms (i) and (ii) is reported and enters the reading, but does not gate.

### 6.2 Margins

The margin $\delta$ is one third of the model's confirmatory request component, rounded to two decimals:

$$\delta_{\text{Qwen}} = 2.17/3 \approx 0.72 \text{ nats}, \qquad \delta_{\text{Llama}} = 0.50/3 \approx 0.17 \text{ nats}.$$

**Why one third.** The question is which explanation carries the request component, so the margin is set relative
to its size: a part smaller than a third of the component is not taken to change which explanation carries it. For
Qwen the margin also lies below the 1.14-nat drop of the swapped line that the ALFWorld-matched arm attributed to an
authorization belief, so an authorization part of that size would not be called negligible; it is larger than the
project's usual 0.5-nat threshold because at 192 items a 0.5-nat equivalence claim is out of reach (§9.2). For Llama
the project's 0.5-nat threshold equals its whole request component and could not split it; 0.17 nats is half the
0.35-nat drop of its swapped line in the matched arm. The same margin is used for both components of a model.

### 6.3 Inference and the Holm family

Every quantity is a mean of per-item differences on the 192 items, with a 95% percentile bootstrap over users
(2,000 resamples) and a two-sided sign-flip permutation test in which all of a user's items flip together, seed
20260924. **The Holm family** is the eight primary tests: $Z_W$, $R_W$, $Z_M$ and $R_M$ on the failing call's first
token, for each of the two models. Their permutation tests use 10,000 permutations, so that the smallest possible
p (about 0.0001) survives Holm's step-down over eight; Holm at α = 0.01, the confirmatory design's threshold. No other
test is adjusted or decides anything.

### 6.4 Per-component categories (failing call, first token; per model, for $X = W$ and $X = M$)

Let $s$ be the sign of $X^{\text{main}}$'s mean, so the arm (i) component is positive after multiplying by $s$. With
$Z$ and $R$ multiplied by $s$:

- **material:** estimate at least $\delta$, 95% CI above 0, Holm-adjusted p below 0.01;
- **negligible:** 95% CI inside $(-\delta, \delta)$;
- **uncertain:** neither.

If $X^{\text{main}}$'s own CI lies inside $(-\delta, \delta)$, the category is **absent** (there is nothing to
decompose). Otherwise:

| Removed part $Z$ | Persisting part $R$ | Category |
|---|---|---|
| estimate at most $-\delta$, CI below 0, Holm p below 0.01 (checked first) | any | grows under the task-blind arm (predicted by neither; reported, not interpreted) |
| negligible | material | relevance-like (reading (a)) |
| material | negligible | authorization-like (reading (b)) |
| material | material | both: part authorization-like, part relevance-like |
| material | uncertain | authorization-like part shown; whether any of the component persists is not determined |
| uncertain | material | relevance-like part shown; whether any of the component is removed is not determined |
| any other combination | | inconclusive |

### 6.5 Per-model reading (combining the two components)

| Request component $W$ | Naming component $M$ | Reading |
|---|---|---|
| relevance-like | relevance-like or absent | (a) throughout: neither the request nor the naming effect behaves like a belief about request-matching authorization; both persist when the backend is stated and shown to ignore the task |
| authorization-like | relevance-like | (a) for naming, (b) for the request: the write request's extra effect disappears under the task-blind backend while the naming effect persists, so the extra effect is read as a belief that the backend refuses writes the user did not request |
| authorization-like | authorization-like | Both disappear: consistent with an authorization belief throughout, or with the task-blind statement and demonstration suppressing task effects in general; the two are not separated |
| authorization-like | absent | The request component disappears; with no naming component, a general suppression of task effects by the arm is not excluded |
| both | any | Part of the request component (its removed part, with CI) behaves like an authorization belief and part (its persisting part) like a relevance bias; the naming component is read as in its own category |
| either "part shown" category | any | The shown part is stated with its CI; the rest is undetermined |
| any | grows, or either is inconclusive | Reported per component; no combined reading |

Whenever $M$ is "authorization-like" or "both", a removed part of $W$ is read with the general-suppression caveat
above. Two further secondary numbers are written beside the reading: the arm (iii) comprehension of K1 and K3 (if the
model answers them correctly and still shows a persisting request component, the bias persists although the model
states the rule) and the arm (i) answers to K1 and K3 (a direct report of the belief under the confirmatory prompt).

### 6.6 What each pattern would and would not show

- **Relevance-like request component.** Shows that the request's extra effect is not a correctable belief about
  the backend: it survives an explicit statement, a demonstration with minimal pairs and (with the gate) the model's
  own correct answer to the rule question. Does not show what produces it (attention to the named order, a prior that
  requested actions succeed, copying of the task's content); it shows only that it is not a belief the prompt can fix.
- **Authorization-like request component, naming relevance-like.** Shows that the request's extra effect behaves like
  a belief about what the backend checks, which a clear description removes, and that the arm did not suppress task
  effects in general (the naming effect survived). Does not show that the model represents an authorization rule
  internally; only that the effect responds to the stated rule as such a belief would. It also does not say whether
  the belief comes from the agent policy text (rules-only arm, confirmatory A7), customer-service priors or the task
  line's wording.
- **Both.** Gives the split in nats, with CIs, for the request component.
- **A gate fails.** If the model does not pass arm (iii)'s comprehension check, a persisting effect could reflect a
  failure to take in the statement rather than a bias; nothing is concluded. If the positive control fails, the
  model's predictions do not follow the stated backend at all, so a null change in arm (iii) would be uninformative.
- **For both models and every pattern:** the categories are about the failing call's first token on 192 items of
  two classes in one domain. They do not say how the effect behaves for other tools, natural agent errors or other
  prompt formats.

## 7. Secondary analyses (prespecified; estimates with CIs and unadjusted p, no verdicts)

1. **Witness decomposition:** $W$, $M$, $Z$ and $R$ on the witness, with the categories of §6.4 as labels.
2. **Per arm, both actions:** $G$, $S$, aligned − neutral, mention-swapped − neutral, mention-aligned − neutral.
3. **Per-line level changes between arms** (paired per item): blind − main and auth − blind for every line and both
   actions. Prediction for arm (ii) against arm (iii) on the witness: the aligned line changes little, the four
   other lines fall.
4. **The arm (ii) signature:** the witness's $M$ in arm (ii) (predicted about 0) and $G$ in arm (ii).
5. **Yes/no:** $G$, $S$ and aligned − neutral per arm and action, and $Z$ and $R$ for $G$ (no question lines are
   scored on yes/no).
6. **Per class** (`status`, `unavailable`): $W$, $M$, $Z_W$, $R_W$, $Z_M$, $R_M$ on the failing call.
7. **Comprehension:** per arm and question, the share of items answered correctly and the mean log-odds of the
   correct answer.
8. **Format:** the median two-way masses, $P(\texttt{\{"}) + P(\texttt{Error})$ and $P(\text{Yes}) + P(\text{No})$,
   per arm and readout (below 0.5 is a red flag).
9. **Reproducibility of arm (i):** where the confirmatory scoring records are available to the analysis, arm (i)'s
   scores here against the confirmatory scores of the same jobs (the prompts are byte-identical): the share within
   0.13 nats (one bf16 step on this scale) and the largest difference. Descriptive.

## 8. Safeguards

- **Prompt diffs** (tests, and a check at the start of every scoring run that refuses to score on any failure):
  arm (i)'s system message is byte-identical to the confirmatory main message (pinned sha256 `7cc04248…`); arm
  (iii)'s is arm (i)'s with the closing paragraph inserted after the backend rules and nothing else; arm (ii)'s is arm
  (iii)'s with the lead and the closing paragraph replaced and nothing else; arm (iii)'s few-shot turns are arm (i)'s
  with E2' and E3' inserted after their partners; arm (ii)'s are arm (iii)'s with E3's assistant turn replaced by the
  authorization error and nothing else; the final user turn of a job is identical in every arm; within an arm, the
  prompts of a job's task lines differ only in the task line. The rendered system messages of the two new arms are
  pinned by sha256 in a test.
- **Example balance** (tests): the cells of §2.2 per arm, E2'/E3' share everything with E2/E3 but the task line,
  and the real outputs are the stored ones except E3 in arm (ii).
- **Comprehension check** (tests): the exact question texts, the expected answers per arm, the 80% rule and the gate.
- **Analysis** (tests on synthetic scores): one constructed data set per predicted pattern (A, AB, B, both, a failed
  gate, "grows", absent naming component), each giving its prestated category and reading; Holm over exactly the
  eight primary tests; the decomposition identity; the gates.
- **Mutation testing:** deliberate bugs in the new prompts, jobs, item selection, comprehension rule, decomposition,
  categories, Holm family and gates, added to `scripts/mutation_check.py`; each must be killed by a named test.
- **Dry run** without models (mock scores): the whole pipeline on the real build, the job counts and the cost.
- **Token check on ICE** (CPU, tokenizers only, both models), on every prompt the full run scores (every arm, action,
  task line, cue and comprehension question): `{"`, `Error`, `Yes`, `No` single tokens; the authorization error opens
  with `Error`; no merge at the prompt/answer boundary; task-line-only differences within an arm; prompt lengths.
- **Scoring gate:** the confirmatory `tau_verify` must have passed on the exact build files, whose sha256 values are
  pinned in the scoring config, as in the confirmatory design (A11, B10).

## 9. Power and cost

### 9.1 Cost

Mean seconds per pass from the confirmatory smoke run at about 10.7k (Qwen) and 9.7k (Llama) tokens: first token 1.50
and 1.20, yes/no 1.48 and 1.20. Arms (ii) and (iii) add about 1,600 tokens (two examples and a paragraph); their passes
are costed 15% slower, 1.73 and 1.38 s.

| Per model | Arm (i) | Arm (ii) | Arm (iii) | Total |
|---|---|---|---|---|
| Passes (192 items × 16, plus 24 × 4 comprehension) | 3,168 | 3,168 | 3,168 | 9,504 |
| Qwen3-8B, GPU-hours | 1.32 | 1.52 | 1.52 | 4.36 |
| Llama-3.1-8B, GPU-hours | 1.06 | 1.21 | 1.21 | 3.49 |

The full run is about 7.9 GPU-hours, plus about 0.3 for the smoke run (6 items, every job). Four shards by user,
both models one after the other in each shard, about 2 hours per shard; the job asks for 4 hours, and every score is
saved as it finishes so a cut-off shard resumes.

### 9.2 Power

**Standard errors.** From the confirmatory primary population, the request component's 95% CI half-width was 0.27
nats for Qwen and 0.043 for Llama on 856 items with effective cluster size 5.62. Projected to 192 items with effective
cluster size 2.03, its standard error lies between two bounds: about 0.17 (Qwen) and 0.028 (Llama) if items of a user
are as correlated as the clustering allows, and about 0.29 and 0.046 if they are independent. The item-level standard
deviation of Qwen's goal effect (4.96 nats) is close to what independence implies, so the upper bounds are the
realistic ones. For the removed part $Z$, which compares two arms on the same items, the standard error is a ratio r
times that of $W$: in the ALFWorld-matched arm, a smaller change of the backend text, r was about 0.35; r = 0.7 and
r = 1.41 (arms as different as independent samples) are the pessimistic cases.

**Chance of reaching the prestated category** (normal approximation; Holm at α = 0.01 over eight tests, worst
rank; margins of §6.2). Scenarios: A, nothing removed; AB, the whole request component removed; mixed, about half
removed (Qwen 1.0 of 2.17, Llama 0.25 of 0.50).

| Model | SE of W | r | Relevance-like under A | Authorization-like under AB | Removed part shown under AB | Both under mixed |
|---|---|---|---|---|---|---|
| Qwen | 0.17 | 0.35 | 1.00 | 0.97 | 1.00 | 1.00 |
| Qwen | 0.17 | 0.7 | 1.00 | 0.97 | 1.00 | 0.99 |
| Qwen | 0.17 | 1.41 | 0.68 | 0.97 | 1.00 | 0.80 |
| Qwen | 0.29 | 0.35 | 1.00 | 0.42 | 1.00 | 0.80 |
| Qwen | 0.29 | 0.7 | 0.90 | 0.42 | 1.00 | 0.74 |
| Qwen | 0.29 | 1.41 | 0.00 | 0.41 | 0.98 | 0.18 |
| Llama | 0.028 | 0.35 | 1.00 | 1.00 | 1.00 | 1.00 |
| Llama | 0.028 | 0.7 | 1.00 | 1.00 | 1.00 | 1.00 |
| Llama | 0.028 | 1.41 | 0.98 | 1.00 | 1.00 | 0.98 |
| Llama | 0.046 | 0.35 | 1.00 | 0.92 | 1.00 | 0.96 |
| Llama | 0.046 | 0.7 | 1.00 | 0.92 | 1.00 | 0.95 |
| Llama | 0.046 | 1.41 | 0.49 | 0.92 | 1.00 | 0.70 |

**What this means.** A removal of the size AB predicts would be detected almost surely for both models (the "removed
part shown" column). For Qwen at the realistic standard error, showing that nothing persists (the full
"authorization-like" category) has a chance of about 0.4, so the likely outcome under AB is "authorization-like part
shown; whether any of it persists is not determined". Showing that nothing is removed (relevance-like) is likely
unless the arms perturb items about as much as independent samples would. The smoke run (§10) reports the standard
deviations of the per-item $W$, $Z_W$ and $R_W$ (no means), and the projected half-widths are recorded in this note
before the full run; they change no rule.

## 10. Runs and order of work

| Run | Device | What it does |
|---|---|---|
| `tau_auth_score_smoke` | GPU | 6 items, every job, both models; format and timing |
| `tau_auth_smoke_analysis` | CPU | masses, seconds, projected hours per shard, the standard deviations of §9.2; no estimate, CI, p or category |
| `tau_auth_score` | GPU | the 192 items and 24 comprehension items, 4 shards by user, both models |
| `tau_auth_analysis` | CPU | everything in §6 and §7 |

Order: this note (committed first); the code, tests, mutants, the dry run and the ICE token check (second commit); an
independent review, repeated until a round finds no must-fix issue; the design frozen at an annotated tag; the
smoke run and its record in this note; the full run; the analysis.

## 11. Risks stated in advance

- **The task-blind arm is a package.** It adds a statement, two minimal pairs and about 1,600 tokens at once; a change
  between arms (i) and (iii) is not attributed to any one of them.
- **General suppression.** An explicit "the task never changes the output" could suppress a bias that is not a
  belief; the naming control addresses this for Qwen and is unavailable for Llama.
- **Comprehension is read on yes/no**, which for Qwen carries a task judgement and a lean toward No. K2 and K4 detect
  a lean; the questions are general, not about the scored call. *(Review 1, S3:)* the questions now follow the history
  without the scored call, and the correct answers are two Yes and two No in every arm, but the residual risk stands:
  Qwen said No to every status-class call in the confirmatory run, so a lean toward No can still fail the gate in arm
  (iii) (where K1 and K4 are Yes) for a reason other than misreading the backend, and every Qwen category would then be
  inconclusive. The 24 items repeat the same four questions in different contexts; they test robustness across
  contexts, not 24 independent understandings.
- **The demonstrations cover write-request lines only** *(review 1, M2)*. Every few-shot example in every arm has a
  write-request task line, and the two minimal pairs of arms (ii) and (iii) contrast a request on this order with a
  request on another order. A model that copies the pairs raises the swapped write line and leaves the question lines
  alone, which mimics "(a) for naming, (b) for the request". The amended reading (R1-M2) separates the two by where the
  removal lies; a question-line minimal pair was considered and not added (it costs examples and tokens, and the split
  already discriminates).
- **192 items of two classes.** Per-class results are secondary; one domain, one prompt format.
- **Arm (i) is rescored, not reused,** so that all three arms are compared within one run on the same hardware and
  code. The confirmatory scores serve only as the reproducibility check of §7.9.

## Implementation record (2026-11-22, after the design commit `6dd0026`; no design change)

Built as designed (`6bc8a41`, `069ff00`). Nothing has been scored. What the build established:

- **Code.** `wf/tau/auth.py` (prompts, item and job selection, the prompt checks that refuse to score, scoring with
  resume, the analysis and decision rules), `wf/tau/auth_token_check.py`, the configs `tau_auth_score`,
  `tau_auth_score_smoke`, `tau_auth_analysis` and `tau_auth_smoke_analysis`, `pace/tau_auth_score.sbatch` (4 shards,
  4 hours) and `pace/tau_auth_tokcheck.sbatch` (CPU). Arm (i) is rendered by the confirmatory `wf/tau/prompts.py` and is
  checked to be identical to the confirmatory prompt for every scored job. The confirmatory runs and their code are
  unchanged.
- **System messages** (sha256): arm (i) `7cc04248…` (the confirmatory one), arm (ii) `11ede0a6…`, arm (iii)
  `2c621c0e…`, pinned in `tests/test_tau_auth.py`.
- **Items.** On the pinned build (`2c8ffeac…`, `9bd6c3f7…`, `231607f7…`, copied from the confirmatory build and checked
  against the pins): 192 items, 96 per class, from 116 users; 24 comprehension items from 24 users; 9,504 jobs per
  model, as designed.
- **Dry run without models** (mock scores, the real build, the verification gate bypassed; its refusal is tested):
  four shards of 29 users each (55, 42, 48 and 47 items), 19,008 records, no missing or stale score, analysis complete
  with every block. Mock scores are constant, so every contrast is 0 and both gates fail, as they should on such data.
- **Token check on ICE** (`pace/tau_auth_tokcheck.sbatch`, job 7241247, new checkout at `069ff00`, tokenizers at the
  pinned revisions, report in [tau_auth_tokcheck.json](tau_auth_tokcheck.json)): PASS for both models on all 9,504
  scored prompts per model. `{"`, `Error`, `Yes` and `No` are single tokens; the authorization error opens with the
  `Error` token; every twin and witness output opens with `{"`, every failure reply and witness failure with `Error`;
  no token merges at the prompt/answer boundary (first-token prompts with both replies and the authorization error,
  yes/no and comprehension prompts with Yes and No); within an arm the prompts differ only in the task line; the
  scoring run's own prompt checks report no problem. Median prompt length of the first-token prompt, Qwen3 / Llama-3.1:
  arm (i) 10,995 / 9,929, arm (ii) 12,667 / 11,348, arm (iii) 13,059 / 11,660 tokens; the longest prompt is 15,535 /
  13,638 (arm (iii), yes/no), far below both context limits. Arms (ii) and (iii) are thus 1,670 and 2,060 Qwen tokens
  longer than arm (i), about 15% and 19% (§9.1 assumed 15% for both); costed at 19% for arm (iii), the full run is about
  8.0 GPU-hours instead of 7.9, and a shard still about 2 hours. The first token check (job 7240675) stopped before
  any check because the Llama tokenizer is only in the top level of the shared cache; the job now takes the cache
  folder.
- **Tests.** `tests/test_tau_auth.py`, 66 tests: golden system messages and texts, the arms' diffs at the level of the
  system message, the examples and every rendered turn, the example cells and minimal pairs, the per-item few-shot
  order, item, comprehension and job selection, the comprehension rule (80%, ties incorrect, per-arm answers), the
  positive control's arms, action and threshold, the decomposition identity, every row of the category and reading
  tables, Holm over exactly eight tests, one synthetic data set per predicted pattern (A, AB, B, both, grows, a
  Llama-sized request component without a naming component, each failed gate), and the runs end to end on mock scores
  (gates, resume, missing and stale scores, the smoke analysis without estimates, the projection).
- **Mutation testing.** 58 mutants (`tau-auth …` in `scripts/mutation_check.py`), run through
  `scripts/tau_mutation_check.py` against `tests/test_tau_auth.py` after the unmutated copy passed: 57 killed on the
  first run, each with its killing test logged. The survivor (the system-message check no longer comparing arm (iii)
  with arm (i)) lived because a changed closing paragraph of arm (iii) also broke the arm (ii) comparison; the test now
  requires the specific arm's check to fire, and the mutant is killed.

- **Full test suite** (once, single process, at `069ff00`): 966 passed, 7 skipped.

Before scoring: an independent review, then the freeze tag, then the smoke run (§10).

## Amendments after review 1, before any score (2026-11-22)

Review 1 (`notes/tau_auth_review1.md`, read at `d4cc451`) found the design not ready to freeze: two must-fix items in
how a removal is read, and six should-fix items. Everything below was decided and committed before any score of any
model in this experiment exists. No job, item, task line, arm or first-token or yes/no prompt changed; the comprehension
prompts changed (S3). Where an amendment changes an earlier section, the earlier text stands as a record and this
section governs.

### R1-M1. The general-suppression caveat follows the naming control, and the removed shares are reported

**Problem.** §6.5 attached the caveat only when the naming component's removal was itself shown material. A uniform
damping of every task effect by about 35% (Qwen: request removed 0.76, naming removed 0.63) was read as "part of the
request component behaves like an authorization belief" with no caveat; and for Llama, which has no naming component,
partial removals carried no caveat either.

**Now** (`model_reading`, `wf/tau/auth.py`). Whenever a removal of the request component is shown (category
authorization-like, both, or authorization-like part shown), the reading carries the general-suppression caveat unless
the naming component is relevance-like (its removal shown negligible). When the naming component is absent, the reading
always carries the sentence "With no naming component in this model, a general suppression of task effects by the arm
is not excluded." Beside every reading, as a descriptive line that decides nothing, the analysis reports the share of
each component that arm (iii) removes,

$$s_W = \frac{\overline{Z_W}}{\overline{W^{\text{main}}}}, \qquad s_M = \frac{\overline{Z_M}}{\overline{M^{\text{main}}}},$$

and their difference, with 95% intervals from the same 2,000 user resamples (`removed_shares`). A uniform suppression
predicts equal shares; a request-specific belief predicts the request share larger. A test runs the reviewer's uniform
35% scenario end to end and requires the caveat and equal shares.

### R1-M2. Where the removal lies: the aligned and the swapped half of the request component

**Problem.** Copying of the request-line minimal pairs (see §11) raises the swapped write line and leaves the question
lines alone, which the naming control cannot see; complete copying even makes the persisting part about −1.8 nats for
Qwen, which §6.4 labelled "authorization-like part shown".

**The split.** The request component is the difference of two halves,

$$W = \big[L(\text{al}) - L(\text{ma})\big] - \big[L(\text{sw}) - L(\text{ms})\big],$$

so its removed part is the aligned half's removed part minus the swapped half's,

$$Z_W = Z_{\text{al}-\text{ma}} - Z_{\text{sw}-\text{ms}}.$$

An authorization belief refuses every unrequested write, so under arm (i) it lowers the swapped, neutral and both
question lines relative to aligned; its removal lies in the aligned half ($Z_{\text{al}-\text{ma}}$ near $Z_W$) together
with a drop of aligned − neutral, and leaves swapped − mention-swapped and swapped − neutral in place. Copying of the
demonstrated pairs does the opposite: the swapped half carries the removal ($-Z_{\text{sw}-\text{ms}}$ near $Z_W$),
swapped − neutral rises in arm (iii), and the aligned half and aligned − neutral stay. In the confirmatory Qwen numbers the
aligned half is +1.59 and the swapped half +0.58 nats of the 2.17.

**Now.** Prespecified and reported in the primary block: the removed part of the aligned half, the swapped half's
contribution (the two add up to $Z_W$), and the removed parts of aligned − neutral and swapped − neutral. **The
authorization reading is stated only if the aligned half shows a removal**: its removed part is at least half of $Z_W$
and its 95% CI excludes 0, both in the direction of $W$ in arm (i) (`aligned_half_shown`). Otherwise a shown removal
reads: "The request component's removed part is not carried by its aligned half (aligned minus mention-aligned), so it
lies on the demonstrated request lines (swapped minus mention-swapped): it is not separated from copying of the
demonstrated minimal pairs and is not read as an authorization belief." A new category, checked after "grows" and
before every other, is **overshoot**: the persisting part materially of the opposite sign (estimate at most $-\delta$,
CI on the far side of 0, Holm $p < 0.01$), labelled "the task-blind arm removes more than the component ... consistent
with copying of the demonstrated minimal pairs, not with an authorization belief; reported, not interpreted". Tests:
partial copying (category authorization-like, copying reading, no authorization sentence), full copying (overshoot), and
the authorization pattern (aligned half shown, the AB reading).

### R1-S1. The Holm family always has eight members

A primary test whose p is not a number (for example, a model whose blind-arm records for a whole action were lost)
enters the family as p = 1, as a missing model already did; the shared `wf.analysis.stats.holm`, which drops NaN for
other studies, is unchanged. Test added.

### R1-S2. Per-class values in every reading

Each model's decision entry quotes, for both components and both classes, the arm (i) value and the removed and
persisting parts. When the two classes' arm (i) values have opposite signs with both CIs excluding 0, the reading adds
"The pooled request (or naming) component averages opposite class effects (status …, unavailable …); see per class."
(Llama's confirmatory goal effect was +1.23 in the status class and −0.33 in the unavailable class.) No per-class
decision.

### R1-S3. The comprehension check

The four questions now follow the task line and the history with no "Next call" line, so they are not read as
questions about the scored call. K3 is reworded to ask whether the backend would **return an error** (instead of
whether it would execute the call), so the correct answers are two Yes and two No in every arm:

| Question | Arms (i) and (iii) | Arm (ii) |
|---|---|---|
| K1, a valid write on an order the task does not ask to change: executed without an error? | Yes | No |
| K2, a requested write that breaks a rule: executed without an error? | No | No |
| K3, a valid write when the task is only a question: does it return an error? | No | Yes |
| K4, a valid requested write: executed without an error? | Yes | Yes |

The scored first-token and yes/no prompts are unchanged. **Decision on the smoke run:** its comprehension and
positive-control results may be read (`gates_readable` in the smoke metrics), because they are gates, not outcomes.
Any change made after reading them is logged as a deviation, and the contrasts themselves stay unread (the smoke
metrics still contain no estimate, CI, p or category of any contrast). The residual risk is stated in §11.

### R1-S4. The reading table's fallback, and the naming-only removal

Every combination not listed in §6.5 reads "Reported per component; no combined reading (request: …; naming: …)", with
the caveat of R1-M1 when a request removal is shown. One more row is written out: request relevance-like with the naming
component's removal shown (authorization-like, both, or part shown) reads "The naming component is removed while the
request component persists: the control moves and the target does not. Neither explanation of the request component
predicts this; the request component is read as relevance-like and the naming change is reported, not interpreted."

### R1-S5. The longest prompt in the smoke run; no 40 GB A100s

The smoke set adds `tau-g-77cc59152d73`, the item with the longest scored prompt (15,535 Qwen tokens, arm (iii), yes/no),
so the smoke run has 7 items (comprehension on all 7). `pace/tau_auth_score.sbatch` accepts L40S, A40, 80 GB A100 and
H100 nodes (`A100-80GB` replaces `gpu-a100`), because at 15.5k tokens the full-vocabulary logits put a 40 GB A100 near
its limit.

### R1-S6. Tests for the properties the reviewer's surviving mutants exposed

User clustering of both the permutation test and the bootstrap in `lin` (four users with strong user-level effects:
p above 0.1 and an interval spanning the user means); the positive control failing with an estimate above 1 nat and a
CI crossing 0; the scorer refusing resumed records with a wrong item hash; the prompt checks' "final turn differs from
main's" and "differs from the confirmatory prompt" branches; a non-finite per-item value left out; a CI touching 0 not
counting as above 0.

### R1 checks

All at `1eedecb` (the code of these amendments; this note and the deviations entry follow it).

- **Tests:** `tests/test_tau_auth.py`, 100 tests (66 before), all passing. Full suite, once, single process: 1000
  passed, 7 skipped.
- **Mutants:** 87 `tau-auth` mutants in `scripts/mutation_check.py`: the reviewer's 12 (R1-1 to R1-12), 17 for the
  amendments (A-1 to A-17) and the earlier 58 (three adapted to the changed code: the authorization arm's K1, the
  removed part against arm (ii), the Holm family's missing model). Run through `scripts/tau_mutation_check.py` against
  `tests/test_tau_auth.py` after the unmutated copy passed: **87 of 87 killed, each with its killing test logged**,
  none timed out, none failed to apply. The reviewer's six real gaps are killed by the S6 tests, and both of its
  near-equivalent survivors by new tests (a CI touching 0; a non-finite per-item value).
- **Token check on ICE** (job 7244868, checkout `text-wm-fix35` at `1eedecb`, both tokenizers at the pinned revisions,
  report in [tau_auth_tokcheck.json](tau_auth_tokcheck.json), replacing the report of `069ff00`): PASS for both models
  on all 9,504 scored prompts each, including the reworded comprehension prompts; no first-token, boundary or
  task-line problem, and the scoring run's prompt checks report none. System messages unchanged (`7cc04248…`,
  `11ede0a6…`, `2c621c0e…`). The comprehension prompts are now about 85 tokens shorter (median, Qwen: arm (i) 11,081,
  arm (ii) 12,753, arm (iii) 13,145); the longest prompt is unchanged (15,535 Qwen and 13,638 Llama tokens,
  `tau-g-77cc59152d73`).

Before scoring: review 2 of these amendments, then the freeze tag, then the smoke run.

## Extension 2026-11-25: Llama-3.2-3B-Instruct (written before any score of this model in this experiment)

The design was frozen at `tau-auth-design-frozen-2026-11-22` (`f70b285`) and run on Qwen3-8B and Llama-3.1-8B
(`notes/tau_auth_results.md`, on `main`). This section adds one model under the frozen design. Nothing above changes.
It is committed, reviewed once and tagged before any Llama-3.2-3B score of this experiment exists; the smoke record and
the full run follow (E8). Scores of this model on the confirmatory τ-bench prompt (arm (i)'s prompt) already exist
from the τ families extension (`notes/tau_families_results.md`) and were read before this section was written; they
are the source of the margin (E3) and of the predictions (E9).

### E1. The model and why

**Llama-3.2-3B-Instruct**, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` (`wf.constants.REVISIONS`), bf16, with
the τ families extension's handling of its chat template (`wf.families.MODELS["llama3.2-3b"]`): the Llama 3 header
format with `date_string` fixed at "26 Jul 2024", which Llama-3.1-8B's template always writes, so that the rendered
prompts are the same text as Llama-3.1-8B's; default attention (SDPA); the confirmatory `LogoddsScorer`, one sequence
per pass, batch size 1. Tag in records: `llama3.2-3b`.

It is the only model besides the two reference models that the τ families extension found interpretable on the
first-token readout: it passed the families competence rule (AUC 0.775), the discrimination check (+1.55 nats) and the
manipulation check (+1.17), with a first-token goal effect of +0.99 nats. Unlike Llama-3.1-8B it has a naming
component (+0.52 nats on the 856 primary items), so the general-suppression control of §6.5 and R1-M1 is available for
it. Its yes/no readout failed the families competence rule (it answers No to about 98% of calls); see E9.

### E2. What is unchanged

Everything in §2 to §8 and in the amendments after review 1, as frozen: the 192 items, the 24 comprehension items and
the 7 smoke items; the three arms, their texts and system messages (sha256 `7cc04248…`, `11ede0a6…`, `2c621c0e…`); the
few-shot examples and their per-item order; the five task lines; the 48 passes per item and 9,504 per model; both
readouts and the comprehension questions; the gates (arm (iii) passes the comprehension check, the correct answer more
probable on at least 80% of the 24 items on each of K1 to K4; the positive control at least 1 nat with its 95% CI above
0); the categories of §6.4 with R1-M2's overshoot category and aligned-half condition; the readings of §6.5 with R1-M1,
R1-S2 and R1-S4; the secondary analyses of §7; 2,000 user-level bootstrap resamples, sign-flip permutation tests at the
user level with seed 20260924, 10,000 permutations for the primary tests and 1,000 for the others, α = 0.01. The
analysis is the frozen code, with only the changes of E4.

### E3. The margin

§6.2 sets the margin to one third of the model's confirmatory request component. For the reference models that was
the write-minus-question contrast on the 856 primary items under the confirmatory prompt (`tau_dx.md`: +2.17 and
+0.50). The same contrast for Llama-3.2-3B on the same 856 items and prompt is in the τ families metrics
(`write_minus_mention`, +0.474; the same field gives +2.17 and +0.50 for the reference models, so it is the same
quantity), so

$$\delta_{\text{Llama-3.2-3B}} = 0.474/3 \approx 0.16 \text{ nats},$$

used for both components, as for the other models.

### E4. The Holm family, and the code changes

**Holm family.** The frozen family is the eight primary tests of the two reference models; their categories are
published and are not reopened. The new model is decided in its own family of four tests (the removed and the persisting
part of the request and of the naming component on the failing call's first token), Holm at α = 0.01, a missing or
non-numeric p entering as 1 (R1-S1). With 10,000 permutations the smallest raw p is about 0.0001, which survives a
multiplication by 4. As a descriptive check that decides nothing, the results note also gives the Holm-adjusted p each of
the four tests would have in a family of twelve (these four and the eight published raw p-values of the reference
models in `tau_auth_metrics.json`, adjusted by Holm's step-down over all twelve) and names any category that would
differ.

**Code** (branch `tau-auth`, after `f70b285`; no change to any frozen run's prompts, jobs, rules or outputs):

- `wf/tau/auth.py`: `EXTENSION_MARGIN = {"llama3.2-3b": 0.16}` beside the unchanged `MARGIN`; `make_scorer` loads a
  frozen model exactly as before and an extension model with its wave-A spec (revision, attention, template
  arguments); `holm_family` and `decide` take the family as a parameter, whose default is the frozen family of eight;
  the analysis reads it from the config key `holm_family` and refuses a model outside it.
- `wf/tau/auth_token_check.py`: renders an extension model's prompts with its template arguments (the fixed date).
- Configs `tau_auth_l32_score`, `tau_auth_l32_score_smoke`, `tau_auth_l32_smoke_analysis` and `tau_auth_l32_analysis`,
  equal to the frozen four except the run ids, the model and, for the analysis, `holm_family: [llama3.2-3b]` (a test
  checks this key by key); registered in `wf/runs.py` with the frozen runners.
- `tests/test_tau_auth.py`: 6 tests for the above (configs, margin, scorer spec, token-check rendering, the family of
  four, a mock run end to end with the refusal); 106 tests in all, passing.

### E4a. Changes after the review of this section (2026-11-25, before the tag and before any score)

One independent review read the section and code at `8cced24`, found no must-fix item and eight should-fix items.
Accepted and made:

- **The arm (i) reproducibility check of §7.9 runs** (review S5). `tau_auth_l32_analysis` sets `reference_run` to the
  τ families scoring of this model (`text-wm-fix44/runs/tau_family_score_llama3.2-3b`, read only): same stack, GPU type,
  scorer and template handling, and the same arm (i) prompts for the jobs it has (the failing call's five first-token
  lines and three yes/no lines, the witness's three and three). Same-environment rescoring has been bit-identical in
  every case measured (`numerical_sensitivity.md` §4.1), so this checks end to end that the extension renders and
  scores the model as the τ families runs did. It is descriptive, as in §7.9.
- **The run records its template settings** (review N3): `tokens_<model>.json` of each shard now holds
  `template_settings` (for this model `{"enable_thinking": false, "date_string": "26 Jul 2024"}`).
- **`make_scorer` refuses a spec that folds the system message** (review N4).
- **Tests** (review S8, N1, N2): the extension's smoke analysis end to end (it must run without a Holm family); the
  token check passes the fixed date to every render (a stand-in tokenizer records the arguments) and none to the frozen
  models; the scorer's attention setting is the default; the mock run's §7.9 check against itself is exact. 108 tests,
  passing.
- **E5 to E9** are revised below for the stopping rule (S1), the numerical floors of the decisive statistics (S2), two
  more fragility boundaries (S3), the rescore (S4), the "absent" row (S6), the token-check invocation (S7) and wording
  (N7 to N9).

### E5. Environment (numerical_sensitivity.md, rule 1)

- **Software stack:** the environment of the original authorization runs and of the τ families scoring runs (the
  project's `serve` Python environment, the "old stack" of `numerical_sensitivity.md` §2: torch 2.6.0, transformers
  4.51.3), with the `pace.env` of the τ families checkout copied into a new checkout with only its paths rewritten.
- **GPU type:** NVIDIA L40S for the smoke run and all four shards, pinned at submission with
  `--constraint='(gpu-l40s)'` (brackets because ICE appends `&cuda13`); a resumed shard is resubmitted with the same
  constraint. No run record stores the GPU, so the results note cites each job's `nvidia-smi` log line. The original authorization runs used an L40S for the smoke run and
  shard 3 and A40s for shards 0 to 2; the τ families extension scored Llama-3.2-3B on L40S only. One GPU type keeps every
  test of this model's family in one environment (rule 1); every within-item contrast is within one job in any case.
- **Checkout:** a new ICE checkout at the extension's tag, with the pinned build (`tau_build`) and its verification
  record (`tau_verify`) copied from the original authorization checkout (`text-wm-fix38`); the scoring run refuses any
  other build files.
- **Token check** (CPU, before the smoke run): `wf.tau.auth_token_check --models meta-llama/Llama-3.2-3B-Instruct
  --config tau_auth_l32_score` on every prompt the full run scores, with the frozen pass conditions.
  `pace/tau_auth_tokcheck.sbatch` passes no arguments, so it is submitted as a CPU job (`--partition=coc-cpu`, 2 CPUs,
  16 GB) whose command sources `pace/_common.sh` (the checkout's Python and Hugging Face cache, read only) and runs
  `python -m wf.tau.auth_token_check runs/tau_build/items.jsonl runs/tau_build/fewshot.json --models
  meta-llama/Llama-3.2-3B-Instruct --config tau_auth_l32_score --out runs/tau_auth_l32_tokcheck/report.json` with
  `HF_HUB_OFFLINE=1`.

### E6. Numerical sensitivity (numerical_sensitivity.md, rules 3 and 4)

Llama-3.2-3B's numerical floor has not been measured (`numerical_sensitivity.md` §8 says each new model family needs
its own), so the Llama-3.1-8B values stand in for it; this is an assumption. For a two-condition mean contrast, one
standard deviation across environments is about 0.026 nats at 100 items and 0.018 at 300 (§4.8), about 0.02 at 192
items. The decisive statistics combine more scores: the request component $W$ and its persisting part $R_W$ combine
four scores per item, as do $M$ and $R_M$, and the removed parts $Z_W$ and $Z_M$ combine eight, from two arms. With
per-score noise roughly independent (§4.3: the arms of a contrast move almost independently), their floors are about
0.03 nats for $W$, $M$ and $R$ and about 0.04 for $Z$. The margin of 0.16 nats is about five times $R$'s floor and four
times $Z$'s, so rule 3's requirement of at least five times is met only roughly for the removed parts; this is stated as
a limit.

**Fragility labels (rule 3), fixed now.** In the results note, a reading is labelled "numerically fragile" when the
number that decides it lies near its boundary, within a band of about twice the floor of its statistic: 0.08 nats for a
removed part $Z$ (and for R1-M2's aligned-half removal), 0.06 for $W$, $M$ and $R$ and for the positive control. That
is: a CI end within the band of 0 or of ±δ (0.16) where the category needs it on one side; an estimate within the band
of ±δ where the category needs at least δ in size; the positive control's estimate within 0.06 of 1 nat or its CI's
lower end within 0.06 of 0; the arm (i) component's CI end within 0.06 of ±δ (absent or not); R1-M2's aligned-half
condition when the aligned half's removed part is within 0.08 nats of half of $Z_W$ or its CI end within 0.08 of 0; and
a Holm-adjusted p between 0.005 and 0.02 where the category needs p below 0.01 (permutation p-values moved by up to a
factor of about 2 when only the scores changed, §4.8). A comprehension question whose count is 19 or 20 of 24 (one item
from the 80% cutoff of 19.2 items) is labelled fragile too. The labels sit beside the category and change none.

**No cross-environment rescore (rule 4).** Rule 4 asks for a rescore on another GPU type, of at least 300 items, for
small effects and equivalence-type decisions, and a "negligible" removed part is one. This experiment has only 192
items, so the 300-item minimum cannot be met. Only the failing call's 15 first-token passes per item (five lines, three
arms) decide anything; rescoring those for all 192 items would take about 0.5 GPU-hours on an L40S-class card and about
0.8 on an A40, which together with the run exceeds this extension's budget of about 2 GPU-hours. It is therefore not run
within this extension. If any decisive number is labelled fragile, the results note says so and names that descriptive
A40 rescore of the failing call's first-token jobs as the check that would settle it; it is run only with a new
approval of the GPU time. The fragility labels are the substitute, and the results note states the omission as a
limit.

### E7. Budget and the stopping rule

The τ families runs scored Llama-3.2-3B on L40S at 0.61 seconds per pass (median prompt 9,771 tokens). Arms (ii) and
(iii) are about 15% and 19% longer than arm (i) (§9, implementation record), so the full run's 9,504 passes are expected
to take about 1.8 GPU-hours, and the smoke run about 0.1. Four shards of about 0.45 hours each, under the frozen
4-hour limit of `pace/tau_auth_score.sbatch`. **Stopping rule:** the total is the smoke job's elapsed wall time (from
Slurm), plus the smoke analysis's `gpu_hours_full_run` (summed mean seconds per pass by arm and readout, over every
job), plus 0.05 hours for loading the model in four shards. If it exceeds 2.25 GPU-hours, the full run is not submitted
and the projection is reported; no rule, item or job is changed to fit. The limit is kept at 2.25, close to the
approved budget of about 2 GPU-hours. The projection probably errs high: in the τ families runs the smoke projection
for this model was 2.50 GPU-hours and the scoring took 2.32 (about 8% less), and this smoke set includes the item with
the longest prompt.

### E8. Order of work

1. This section and the code (committed together, before any score). 2. One independent review; fixes committed.
3. The annotated tag `tau-auth-ext-llama32-2026-11-25`, pushed. 4. The CPU token check (it scores nothing and may run
before the tag). A failed check (a problem in the report) stops the extension; a job that fails before checking (for
example a missing file) is fixed operationally and rerun, and logged. 5. The smoke run and smoke analysis at the tag;
the smoke record (format, timing, projection, the standard deviations of §9.2 with projected half-widths, and both
gates, which R1-S3 allows to be read) committed to this note before the full run. That commit touches only this note,
so the full run's code is the tag's. 6. The full run, whatever the smoke gates show, as in the original
run: a gate failure is a reported outcome, not a reason to change the design (the only stop is E7's). 7. The analysis
at the tag. 8. A dated extension section in `notes/tau_auth_results.md` with the metrics copied unchanged as
`notes/tau_auth_l32_metrics.json`. Any change after the tag is logged as a deviation.

### E9. Predictions, and what each outcome would mean

**What is known.** On the 856 primary items under the confirmatory prompt, Llama-3.2-3B's failing-call first token
gave a goal effect of +0.99, a naming component of +0.52 and a request component of +0.47 nats; both write requests lift
predicted success above the neutral line (aligned − neutral +2.01, swapped − neutral +1.02), and a bare question about
the target order lowers it (mention-aligned − neutral −0.30); the witness's goal effect was +1.07. Its yes/no answers
were No for about 98% of calls, working or failing, under the neutral line.

**Expected gate outcomes (stated before any data of this experiment).** The comprehension check is read on yes/no. A
model that answers No to almost everything fails K1 and K4 in arm (iii), whose correct answer is Yes, so gate 1 is more
likely to fail than to pass (subjective probability of passing about 0.25). A lean alone does not decide it:
Llama-3.1-8B also leaned No on τ-bench calls (it predicted JSON for 7% of failures and 23% of successes) and still
answered K1, K3 and K4 correctly on 24 of 24 items in arm (iii). The positive control needs the witness's
aligned − swapped gap to grow by 1 nat under the authorization description. Llama-3.1-8B, whose witness goal effect was
+0.38, reached +0.30; Llama-3.2-3B's is +1.07, so passing is possible but not likely (about 0.4). Both gates passing:
about 0.1 if the two were independent, perhaps 0.15 since a model that reads the stated backend well is likely to do
both. The most likely outcome is therefore "inconclusive (gate)" for both components, which is a result: this
design cannot certify, for this model, that it read the task-blind description or follows a stated backend.

**If both gates pass**, the frozen categories read as follows.

| Request component | Naming component | What it would mean |
|---|---|---|
| relevance-like | relevance-like | As Qwen3-8B: the request effect survives a stated, demonstrated and understood task-blind backend; it is not a belief the description corrects. A second model family for reading (a) |
| relevance-like part shown (removed part uncertain) | any | The persisting part is shown; whether a part of up to about the margin is removed is not determined. With a projected half-width of about 0.1 to 0.2 nats for the removed part at 192 items, this is the likely form of reading (a) for this model |
| authorization-like, both, or a part shown, with the aligned half shown | relevance-like | The first evidence in this experiment of an authorization-like part: the request's extra effect shrinks under the task-blind backend, through the aligned half, while the naming effect persists |
| any removal of the request component not carried by the aligned half | any | Copying of the demonstrated minimal pairs (R1-M2), not an authorization belief |
| any removal of the request component | removed too, or absent | A general suppression of task effects by the arm is not excluded (R1-M1) |
| grows or overshoot | any | Reported, not interpreted (as Llama-3.1-8B's ungated request component) |
| absent (arm (i) CI inside ±0.16) | any | Nothing to decompose on these 192 items. Very unlikely: with an expected half-width of about 0.12 to 0.16 nats, it would need the 192-item estimate within about 0.05 of 0, against +0.47 on the 856 items |

**What no outcome shows.** Any category concerns the failing call's first token on 192 items of two classes, in one
domain and one prompt format. One more Meta model does not separate size, distillation and post-training, and does not
extend the result to another developer.
