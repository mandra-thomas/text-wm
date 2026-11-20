# τ-bench retail replication: design (written 2026-11-20, before any item was generated)

This note fixes the design of the second-environment replication of the RQ2 goal effect on the τ-bench
(tau2-bench) retail domain. It is written and committed before the item set is built and before any model is
scored. The feasibility probe of 2026-11-20 (report/TRACKER.md §4.4, "τ-bench trial build result") built a
throwaway version of the items; its counts informed the expected sizes below, and no model was scored on it.
Anything changed after this commit goes in a dated "Changes" section at the end, with its reason.

## 1. Question and prediction

**ALFWorld finding being replicated.** A world model is an LLM that predicts the environment's reply to an
action, given the history and the agent's task. In ALFWorld, when the history determines that an action fails,
naming the action's object in the task line (aligned) instead of another object (swapped) moves the model's
prediction toward success.

**Here.** The environment is the retail customer-service backend. An agent calls tools, and the backend replies
with JSON on success or a line starting with `Error: ` on failure. Each test item is a write call (cancel, modify,
return, exchange) on a target order that fails a precondition. The fact that decides the failure is printed
verbatim in an earlier tool output in the history. The task line names either the target order and product
(aligned), another order of the same user and its product (swapped), or nothing specific (neutral).

**Readout.** L is the log-odds of a success reply against a failure reply. The goal effect is the mean, over
items, of L under the aligned task minus L under the swapped task (ΔL, in nats).

**Prediction (ALFWorld direction).** ΔL > 0: naming the target order and product makes the model more
confident that the failing call succeeds.

**Why the test is two-sided.** There is one important interpretive difference from ALFWorld:

- In ALFWorld the deciding fact was usually about the room (where the agent stands, what a receptacle holds).
- Here the deciding fact is an attribute of the named object itself, and the history shows it: the target
  order's status, the availability of the target product's variant, the order's original payment method, or the
  gift card's balance.

Naming the object could therefore pull the other way. It may draw the model's attention to exactly the record
that blocks the call, which would make the failure more salient under the aligned task. So the test is
two-sided:

- **ΔL > 0** would mean that a task naming an object overrides visible contrary evidence about that same object:
  a stronger form of the ALFWorld effect.
- **ΔL < 0** would mean that naming the object makes the model read that object's record more carefully, so the
  goal improves failure prediction here. The ALFWorld effect would then depend on the deciding fact lying
  outside the named object.
- **ΔL ≈ 0** with a narrow interval would mean the goal does not leak into these predictions. Software rules
  printed in JSON records would then be immune where household physics was not.

## 2. Environment

**Source.** tau2-bench, https://github.com/sierra-research/tau2-bench, commit
`c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4`, retail domain. MIT licence (Copyright (c) 2025 Sierra Research); the
notice is in `wf/tau/vendor/LICENSE`.

**What is vendored, in `wf/tau/vendor/`, with sha256 values recorded in `wf/tau/vendor/MANIFEST.json`:**

- `db.json.gz`: the retail database, gzip of `data/tau2/domains/retail/db.json`; the uncompressed sha256 is
  `413a6516…7766765` (full value in the manifest);
- `policy.md`: the retail agent policy, verbatim;
- `tools_openai.json`: the 16 agent tools' OpenAI schemas, exported from the upstream environment at the pinned
  commit;
- `upstream/`: the upstream `tools.py`, `data_model.py`, `text_match.py` and the response-formatting excerpt of
  `environment.py`, kept as reference text and not imported.

**Port.** `wf/tau/retail.py` is a dependency-free port of the retail toolkit (no pydantic). It keeps the upstream
method names, their precondition checks in the same order, the same error strings, and the same JSON
serialisation. The serialisation is `json.dumps` of the record, with fields in model order and unset optional
order fields shown as `null`. The environment wrapper returns `"Error: <message>"` when a tool raises, as
upstream's `Environment.get_response` does.

**Why a port.** The project's environments do not carry pydantic. A port that can be checked byte for byte
against upstream is easier to audit than a vendored copy with shims.

**Equivalence check.** `scripts/tau_upstream_equivalence.py` runs where upstream tau2 at the pinned commit is
importable. It passes only if every item's history outputs, failure reply and success twin are byte-identical
under upstream, and a seeded random sample of calls (every tool, valid and invalid arguments) gives identical
outputs and an identical database afterwards. Upstream builds the twin by patching the database copy, so this
also checks the port's bypass by a second route (§3.4).

**Determinism.** The tools are pure functions of the database and the arguments. A failed write raises before
it mutates anything. The probe found 3,718 of 3,718 byte-identical replays of released tau2 trajectories.

## 3. Generated items (the confirmatory set)

### 3.1 Users and orders

**Who is used.** Every user with at least two orders, except the three few-shot users (§5.3). Each user is one
cluster.

**Hashing.** "Hash" below means the first 16 hex digits of sha256 of `"20260924|<parts joined by |>"`, read as an
integer.

### 3.2 History

The history holds tool calls and their outputs only; no user or assistant text. For target order W of user u
and comparison order W2, the calls are, in this order:

1. `get_user_details(user_id=u)`;
2. `get_order_details` for W and for W2, in a counterbalanced order: W first when hash(W, W2, "order") is even;
3. only for the variant-unavailable class, `get_product_details` for the target item's product and for the
   comparison item's product, in the same counterbalanced order. If both items are of the same product it is
   shown once.

Each output is produced by running the call on a fresh copy of the pinned database, so every byte is real.

### 3.3 Failing actions, by class

Each action is a write on W that fails exactly one precondition. That precondition is visible verbatim in an
earlier output in the history. Where a call needs a payment method it uses W's original payment method,
`payment_history[0].payment_method_id` (called PM0 below), unless the class itself is about the payment method.
The item chosen inside W is picked by hash among W's items that satisfy the class's requirement.

| Class | Action | When generated | Deciding visible field(s) |
| --- | --- | --- | --- |
| `status` | any of the six order writes, below | the order's status fails that tool's status rule | `status` in W's `get_order_details` output |
| `unavailable` | `modify_pending_order_items` if status is exactly `pending`; `exchange_delivered_order_items` if `delivered`; one item, new item = a variant of the same product with `available: false` that is not the old item | the item's product has such a variant | `variants[<new id>].available` in `get_product_details` of the item's `product_id` |
| `wrong_return_pm` | `return_delivered_order_items` on a `delivered` order with the target item and a non-gift-card payment method of u that is not PM0 | u has such a method | `payment_methods[<pm>].source` in `get_user_details`, and `payment_history[0].payment_method_id` in W's output |
| `gc_balance` (secondary) | `modify_pending_order_payment` on an order whose status contains `pending`, whose `payment_history` has exactly one entry of type `payment`, to a gift card of u that is not PM0 and whose balance is below that payment's amount | u has such a card | that card's `balance` in `get_user_details`, against `payment_history[0].amount` |

The six writes generated for the `status` class, each with its arguments:

- `cancel_pending_order(W, reason="no longer needed")` when the status is not exactly `pending`;
- `modify_pending_order_address(W, <u's default address>)` when the status does not contain `pending`;
- `modify_pending_order_payment(W, <the first of u's payment methods, in profile order, that is not PM0>)` when
  the status does not contain `pending`;
- `modify_pending_order_items(W, [item], [an available other variant], PM0)` when the status is not exactly
  `pending` and such a variant exists;
- `return_delivered_order_items(W, [item], PM0)` when the status is not `delivered`;
- `exchange_delivered_order_items(W, [item], [an available other variant], PM0)` when the status is not
  `delivered` and such a variant exists.

**Accepting an item.** The action is run on the history's state. The item is kept only if:

- the call fails;
- its error text is the one the class predicts (§3.6);
- the database is unchanged afterwards (sha256 of the full state before and after);
- a success twin exists (§3.4).

Every rejection is counted by reason in the build log.

### 3.4 Failure reply and success twin

**Failure reply.** The tool's real output, `Error: …`.

**Success twin.** The output of the same call on a copy of the same state with only the violated check bypassed.
The port has one switch per class (`status`, `unavailable`, `wrong_return_pm`, `gc_balance`). Each switch skips
only that `if`, and nothing else changes. If any later check then fails, the item is rejected ("second check
fails"). For example, the status is bypassed but PM0 is a gift card that cannot cover the price difference.

**Second route, for the equivalence check.** The upstream script makes the same twin by patching the database
copy instead of skipping a check:

| Class | Patch on the database copy |
| --- | --- |
| `status` | set W's status to the one the tool requires |
| `unavailable` | set the variant's `available` to true |
| `wrong_return_pm` | turn the payment method into a gift card with balance 0 |
| `gc_balance` | set the card's balance to 10^6 |

The tools overwrite the status on success, and the order output does not contain availability, balances or a
method's type. So both routes must give identical bytes.

### 3.5 Task lines

The templates follow the task instructions' style:

| Action | Template |
| --- | --- |
| `cancel_pending_order` | The user wants to cancel order {o}. |
| `modify_pending_order_address` | The user wants to change the shipping address of order {o}. |
| `modify_pending_order_payment` | The user wants to change the payment method of order {o}. |
| `modify_pending_order_items`, `exchange_delivered_order_items` | The user wants to swap the {p} in order {o} for a different option of the same product. |
| `return_delivered_order_items` | The user wants to return the {p} from order {o}. |

The three conditions:

- **Aligned:** the action's template with o = W and p = the target item's product name.
- **Swapped:** the same template with o = W2 and p = the comparison item's product name.
- **Neutral:** "The user wants help with their account."

**No false claims.** The builder checks, against the database, that no task line asserts a false fact. An item
fails this check if:

- the named order is not one of u's orders;
- the named product is not among that order's items;
- the template is the swap template and the product has fewer than two variants.

An item whose swapped line fails is rejected.

**Choosing the comparison order W2.**

- W2 is another order of u whose status equals W's status (**status-matched**) when u has one, otherwise any
  other order of u.
- Ties are broken by hash(W, W2).
- For product templates, W2 must have an item whose product has at least two variants (for the swap template)
  or any item (for the return template). That item is the comparison item, chosen by hash, preferring a product
  name different from the target's.
- If no candidate W2 qualifies, the item is rejected.

### 3.6 Achievability

**When a task is achievable.** A task line (template, order o, product p) is achievable from the item's state if
some single write call achieves it, run on a copy of that state, succeeds. The witnesses tried are:

| Template | Witness calls tried |
| --- | --- |
| swap | `modify_pending_order_items` if o's status is exactly `pending`, else `exchange_delivered_order_items` if `delivered`, with the named item, every available variant of the product other than the item, and every payment method of u |
| return | `return_delivered_order_items` with the named item and every payment method of u |
| change payment | `modify_pending_order_payment` with every payment method of u |
| cancel | `cancel_pending_order` with each valid reason |
| change address | `modify_pending_order_address` with u's default address |

The first witness that succeeds is recorded. "Not achievable" means every witness failed.

**Main population.** The aligned task is achievable. For the `status` class this keeps only two cases: the swap
task on a `pending` order where the action was `exchange_delivered_order_items` (achievable by
`modify_pending_order_items`), and the swap task on a `delivered` order where the action was
`modify_pending_order_items` (achievable by exchange). In the main population the agent therefore chose a wrong
tool or a wrong argument for a task that a right one would achieve.

The probe found about 1,376 such items from 297 users. The confirmatory set is **all main-population items**,
with no cap.

**Prespecified strata, each reported with its own n:**

- **Class:** `status`, `unavailable`, `wrong_return_pm`, `gc_balance`.
- **Swapped achievability:** whether the swapped task is achievable.
- **Status matching:** whether W2's status equals W's. The status-matched pool and the full pool are both
  reported; the probe found about 842 status-matched items from 175 users.

**Non-main items.** Items whose aligned task is not achievable are built and verified too, and form the
`unachievable` set. Examples are cancelling a delivered order, or returning items from a pending order. That
set is scored only in the exploratory arm (§6.4).

### 3.7 Uniqueness

No two items may share the same (visible history, action), compared on the canonical JSON of the history and
the action. A duplicate fails verification.

## 4. Natural items (external-validity set, not part of the confirmatory test)

**Source.** The released tau2 retail trajectories at the pinned commit,
`data/tau2/results/final/*_retail_*_4trials.json`. These are four agent models, four trials each. The files are
read from a local path given in the config, and their sha256 values are checked. They are not committed: they
contain text written by commercial models.

**Rule.** For each failed order write in a trajectory:

1. **History.** The history is the earlier tool calls and outputs only, with all user and assistant text
   dropped.
2. **Replay.** Each historical call is replayed on a fresh copy of the database. Every output must match the
   recorded one byte for byte, and the failing call must reproduce its recorded error text. Otherwise the item
   is dropped as a replay mismatch.
3. **Class.** It must be one of the four classes of §3.3, and the text-only checker (§7) must predict FAIL with
   that class from the history alone (determined failures only).
4. **No echoes.** The item is dropped if an earlier call in the same trajectory was a failed write on the same
   order.
5. **Length cap.** At most 30 history calls and 24,000 characters of rendered history (about 7k tokens).
6. **Deduplication.**
   - Exact duplicates of (visible history, action) are dropped.
   - Then at most one item is kept per (order, tool, error text), the one with the shortest history (ties broken
     by hash).
7. **Twin.** As in §3.4, on the replayed state. A second-check failure drops the item.
8. **Task lines.**
   - Aligned names the target order and, for item tools, the product of the first item in `item_ids`.
   - Swapped names another order of the same user whose `get_order_details` output appears in the history,
     chosen by hash.
   - Items with no such order have no swapped line and are scored only aligned against neutral.
9. **Few-shot users.** Items from the few-shot users are dropped.

The probe found about 66 distinct items before the class and twin filters, so this set is small. It is
reported separately, by class.

## 5. Prompt

### 5.1 System message

The system message (`wf/tau/prompts.py`), fixed text, in this order:

- **Instruction:** "You are a world model of the backend of a retail customer-service system. An agent is
  helping a user and calls the backend's tools. Given the user's task, the tool calls made so far with their
  outputs, and the agent's next tool call, predict the exact output of the tool call. Successful calls return
  JSON; failed calls return a single line starting with "Error: ". Output only the tool output."
- **The policy:** "# Retail agent policy", then `policy.md` verbatim.
- **The tool descriptions:** "# Tools", then `tools_openai.json` as compact JSON, one tool per line.

### 5.2 Each user turn

Each user turn (examples and query) reads:

```
Task: <task line>

Tool calls so far:
Call: {"name": "<tool>", "arguments": {...}}
Output: <output>

Call: …
Output: …

Next call: {"name": "<tool>", "arguments": {...}}
Tool output:
```

Calls are `json.dumps({"name": n, "arguments": a})`. Outputs are verbatim. In an example, the assistant turn
holds the call's exact output.

### 5.3 Few-shot examples

There are three examples, from three users who are excluded from every test set. They are chosen as the first
users, in hash(user) order, who satisfy each example's need (one user per example, all distinct):

1. **Success.** `get_user_details`, `get_order_details` of a `pending` order, then
   `modify_pending_order_address` to the user's default address. Task: "change the shipping address".
2. **Failure, not a test class.** `get_user_details`, `get_order_details` of a `pending` order with one payment,
   then `modify_pending_order_payment` to that same method, giving "Error: The new payment method should be
   different from the current one". Task: "change the payment method".
3. **Success.** `get_user_details`, `get_order_details` of a `delivered` order, then
   `return_delivered_order_items` of its first item to PM0. Task: "return the {p}".

Every example output is real tool output from the port.

### 5.4 Conditions

The aligned, swapped and neutral prompts differ only in the task line of the final user turn; a test checks
this. The yes/no variant replaces the final "Tool output:" with:

"Question: will this tool call succeed, or will it return an error? Answer Yes if it succeeds and No if it
returns an error.
Answer:"

## 6. Readouts and analysis

### 6.1 Readouts

**Primary: first-token L1.** L1 = log P(`{"`) − log P(`Error`) at the first answer position after the chat
template's generation prompt. Every success reply starts with `{"` and every failure reply with `Error`.

**Token check.** `wf/tau/token_check.py` must show, for Qwen3-8B and Llama-3.1-8B-Instruct, that:

- `{"` and `Error` are single tokens without a leading space;
- they are the first tokens of the tokenised twin and failure reply;
- the chat template's generation prompt ends in a newline, or in `</think>\n\n` for Qwen3 with thinking off, so
  the no-space variant is the one the model would emit.

If a model fails this check, its primary readout falls back to the first token of each full reply, read off the
full-reply pass. The fallback is recorded.

**Yes/no.** log P(`Yes`) − log P(`No`) after the yes/no question.

**Secondary: full replies.** Two numbers from teacher-forcing the success twin and the failure reply:

- L_full = log P(twin) − log P(failure);
- L_norm = log P(twin)/n_twin − log P(failure)/n_failure, the per-token form. Twins are long, a median of
  about 486 tokens, so L_norm is the one interpreted.

The full-reply pass also gives each reply's first-token log-probability. These must agree with the primary pass
within 0.05 nats on at least 99% of items; any disagreement is reported as a tokenisation-boundary problem.

### 6.2 Effects and inference

**Effects.** For each readout and model, three paired differences:

- ΔL = mean of L(aligned) − L(swapped), on items that have a swapped line;
- aligned − neutral;
- swapped − neutral.

**Inference.**

- 95% percentile cluster bootstrap over users, with 2,000 resamples.
- Two-sided sign-flip permutation test with all of a user's items flipping together, 1,000 permutations.
- Seed 20260924, as in `wf/analysis/stats.py`.

### 6.3 Confirmatory decision (primary readout, main population, per model)

The permutation p is Holm-adjusted over the two models (Qwen3-8B, thinking off; Llama-3.1-8B-Instruct). The
verdict:

- **Replicates (ALFWorld direction):** ΔL ≥ 0.5 nats, the CI excludes 0, and Holm p < 0.01.
- **Reversed (naming sharpens failure prediction):** ΔL ≤ −0.5, the CI excludes 0, and Holm p < 0.01.
- **Small effect, direction stated:** the CI excludes 0 and Holm p < 0.01, but |ΔL| < 0.5.
- **No effect larger than 0.5 nats:** the CI lies inside (−0.5, 0.5).
- **Inconclusive:** anything else.

The replication holds overall if at least one model replicates and none reverses. The thresholds are those of
ALFWorld's preregistered rule, applied to the first-token scale.

### 6.4 Secondary and exploratory analyses (reported, no verdict)

- yes/no and full-reply ΔL;
- aligned and swapped each against neutral;
- every stratum of §3.6;
- the natural set;
- the `unachievable` set (exploratory; scored in the same run only if `score_unachievable: true`);
- the share of items whose sign of L1 flips between swapped and aligned;
- the median two-way first-token mass, P(`{"`) + P(`Error`), as a format check. Below 0.5 is a red flag.

### 6.5 Sample size

All main-population items (expected about 1,376 from about 297 users). There is no cap and no stopping rule. The
analysis refuses to run if any primary score is missing.

## 7. Text-only checker (`wf/tau/checker.py`)

**What it reads.** The checker reads only the history's tool outputs and the action's name and arguments. It
never touches the database or the port; a test fails if the module imports either.

**What it predicts.** `FAIL` with the class and the exact error text, `PASS`, or `UNKNOWN` (a fact it needs is
not visible). Verification requires `FAIL`, the same class and the identical error text for every item.

### 7.1 What the checker reads from the history

**Records.**

- A tool output is a record if it parses as a JSON object.
- An object with `order_id`, `items` and `status` is an order.
- An object with `user_id` and `payment_methods` is a user.
- An object with `product_id` and `variants` is a product.
- For each id, the latest record shown counts. Successful writes return the updated order, so they update it.

**Stale gift-card balances.** A gift card's balance is stale if a later successful write lists that card in its
arguments or output, unless the user is shown again afterwards. Stale means unknown.

**Product records.** No tool changes products, so product records never go stale.

### 7.2 Precondition order of each write tool (upstream code, pinned commit)

The first failing check decides the reply. Checks marked "(n/a)" cannot fail on a consistent database for a
shown order, and the checker assumes they pass. "Count" means that some id appears more often in the call's
`item_ids` than in the order's `items[*].item_id`.

`cancel_pending_order(order_id, reason)`:

1. The order exists. Otherwise "Error: Order not found".
2. `status` is exactly `pending`. Otherwise "Error: Non-pending order cannot be cancelled".
3. `reason` is `no longer needed` or `ordered by mistake`. Otherwise "Error: Invalid reason".
4. (n/a) Each payment's method exists.

`modify_pending_order_address(order_id, address1, address2, city, state, country, zip)`:

1. The order exists.
2. `"pending" in status`, so `pending (item modified)` passes. Otherwise "Error: Non-pending order cannot be
   modified".

`modify_pending_order_payment(order_id, payment_method_id)`:

1. The order exists.
2. `"pending" in status`. Otherwise "Error: Non-pending order cannot be modified".
3. The method is in the user's `payment_methods`. Otherwise "Error: Payment method not found".
4. `payment_history` has exactly one entry and its `transaction_type` is `payment`. Otherwise "Error: There should
   be exactly one payment for a pending order".
5. The method differs from `payment_history[0].payment_method_id`. Otherwise "Error: The new payment method should
   be different from the current one".
6. If the method's `source` is `gift_card`, its `balance` ≥ `payment_history[0].amount`. Otherwise
   "Error: Insufficient gift card balance to pay for the order".

`modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)`:

1. The order exists.
2. `status` is exactly `pending`. Otherwise "Error: Non-pending order cannot be modified".
3. Count. Otherwise "Error: <item_id> not found".
4. `len(item_ids) == len(new_item_ids)`. Otherwise "Error: The number of items to be exchanged should match"
   (no final period).
5. For each pair, in order:
   - old ≠ new. Otherwise "Error: The new item id should be different from the old item id".
   - The new id is a key of the old item's product's `variants`. Otherwise "Error: Variant not found".
   - That variant's `available` is true. Otherwise "Error: New item <new_id> not found or available".
6. The method is in the user's `payment_methods`. Otherwise "Error: Payment method not found".
7. If the method is a gift card, `balance` ≥ the summed price difference (new − old, not rounded). Otherwise
   "Error: Insufficient gift card balance to pay for the new item".

`exchange_delivered_order_items(order_id, item_ids, new_item_ids, payment_method_id)`:

1. The order exists.
2. `status` is exactly `delivered`. Otherwise "Error: Non-delivered order cannot be exchanged".
3. Count. Otherwise "Error: Number of <item_id> not found." (with a final period).
4. Lengths equal. Otherwise "Error: The number of items to be exchanged should match." (with a final period).
5. For each pair, in order. There is no old ≠ new check.
   - The new id is a variant of the old item's product. Otherwise "Error: Variant not found".
   - That variant's `available` is true. Otherwise "Error: New item <new_id> not found or available".
6. The method exists. Otherwise "Error: Payment method not found".
7. If the method is a gift card, `balance` ≥ round(price difference, 2). Otherwise "Error: Insufficient gift
   card balance to pay for the price difference".

`return_delivered_order_items(order_id, item_ids, payment_method_id)`:

1. The order exists.
2. `status` is exactly `delivered`. Otherwise "Error: Non-delivered order cannot be returned".
3. The method is in the user's `payment_methods`. Otherwise "Error: Payment method not found".
4. The method's `source` is `gift_card`, or the method equals `payment_history[0].payment_method_id`. Otherwise
   "Error: Payment method should be the original payment method".
5. Count. Otherwise "Error: Some item not found".

### 7.3 Visible fields the checker needs

| Field | Where it is shown |
| --- | --- |
| order `status`, `items[*].item_id`, `items[*].product_id`, `items[*].price`, `payment_history[*]`, `user_id` | `get_order_details` output, or a successful write's output |
| `payment_methods[<id>].source` and `.balance` | `get_user_details` output |
| `variants[<id>].available` and `.price` | `get_product_details` output |

If a needed field is missing, the checker answers UNKNOWN, never a guess.

### 7.4 Classes

`status`, `unavailable`, `wrong_return_pm` and `gc_balance` are the test classes. The checker also names the
others: `order_not_found`, `invalid_reason`, `pm_not_found`, `one_payment`, `same_pm`, `item_not_in_order`,
`length_mismatch`, `same_item` and `variant_missing`.

**A second, blind checker** will be written by another person from this section only. Verification will require
the two to agree on every item.

## 8. Safeguards (mirroring notes/testing.md)

1. **Pinned inputs.**
   - `wf/tau/vendor/MANIFEST.json` lists each vendored file's sha256.
   - The build refuses to run on a mismatch.
   - The build writes `runs/tau_build/manifest.json` with the sha256 of every output and every input.
2. **Replay of every item in a fresh environment** (`tau_verify`, CPU):
   - the history, failure reply and twin regenerate byte for byte;
   - the failed write leaves the full database hash unchanged;
   - the recorded achievability witnesses succeed, and the "not achievable" flags are re-derived;
   - 100 seeded items are replayed in a new process that reloads the database from disk.
3. **Text-only checker.** FAIL, the same class and the identical error text on every item; refusal on any
   disagreement.
4. **Known-answer tests.** Hand-built histories for every check in §7.2, against both the port and the checker.
5. **Prompts differ only in the task line** (test), and **no task line asserts a false fact** (verify).
6. **Mutation tests.** At least 15 deliberate bugs in the port, the builder and the checker
   (`scripts/tau_mutation_check.py`); each must be caught by `tests/test_tau_*.py`.
7. **Upstream equivalence** (§2), recorded in `runs/tau_build/upstream_equivalence.json`.
8. **Gate.** Scoring refuses items that `tau_verify` has not passed on the exact file (sha256), or whose sha256
   differs from the one pinned in the scoring config after the local build.

## 9. Runs

| Run | Device | What it does |
| --- | --- | --- |
| `tau_build` | CPU | builds the generated and natural items, the few-shot examples and the manifest |
| `tau_verify` | CPU | replays every item and runs the checker (§8) |
| `tau_score` | GPU | scores the main population (and the natural set) with both models: aligned, swapped and neutral; first token, yes/no and full replies. Sharded by user (`WF_SHARD`, `WF_N_SHARDS`) and saved per score |
| `tau_score_smoke` | GPU | 20 items, for format and timing |
| `tau_analysis`, `tau_smoke_analysis` | CPU | the analysis |
| `tau_contamination` | GPU | exploratory pretraining-contamination check (§10) |

## 10. Contamination check (exploratory)

The database and trajectories are public, so the models may have seen them.

**The check.** On 150 records (50 users, 50 orders and 50 products, drawn by hash from the main population's
histories), compare the model's log-probability of the record's JSON verbatim against the same JSON with every
digit of every identifier resampled. The resampling is consistent within the record: the same id gets the same
new digits everywhere. It covers order ids, item ids, product ids, payment-method ids, zips and tracking ids.

**What it measures.** Without memorisation the random digits are unpredictable either way, so the mean
difference should be about 0. A clearly positive difference means the specific records were memorised. That
would argue for reading the effect as possibly retrieval-driven. It does not invalidate the test, because the
history shows the deciding facts anyway.

## 11. Risks stated in advance

- **Users are reused.** About 4.6 items per user, so inference clusters by user.
- **Chosen errors.** Generated actions are errors chosen by us, not by an agent; the natural set addresses this
  partly.
- **Low diversity.** One domain, five templates, 50 product types.
- **Long prompts.** About 6–10k tokens.
- **Swapped achievability varies.** It is a stratum, not a control.
- **A single author built the checker.** Hence the blind second checker.

## Changes and implementation record (after the design commit)

**2026-11-20, no design change.** Built and checked as designed. What the build established:

- **Generated items.** 3,371 items from 304 users.
  - **Main population** (aligned task achievable): 1,389 items from 296 users. By class:
    - `status`: 666;
    - `unavailable`: 655;
    - `wrong_return_pm`: 61;
    - `gc_balance`: 7.
  - **Status-matched:** 856 items from 175 users.
  - **Swapped task achievable:** 1,201 items, against 188 where it is not.
  - **Rejections:** 118 candidates, all because a second check failed when the twin was made (gift-card balance 97, exactly-one-payment 21).
- **Natural items.** Only 28 from 23 users, fewer than the probe's 66. The drops:
  - the class filter (one of the four test classes): 102 dropped;
  - exact duplicates: 60;
  - same order, tool and error as a kept item: 58;
  - not determined by the history: 19;
  - echoes: 9;
  - second-check failures: 3;
  - history too long: 1.

  No trajectory failed to replay. Of the 28, 17 have an achievable aligned task and 5 have no swapped line.
- **Token checks.** Run on ICE with `wf/tau/token_check.py`, on 40 main items, for both models:
  - `{"`, `Error`, `Yes` and `No` are single tokens;
  - the twins and failure replies start with them;
  - the generation prompt ends in a newline;
  - no token merges across the prompt/answer boundary.

  Prompts are about 9.5k tokens for Qwen3 and 8.6k for Llama.
- **Bug fixed before the full build.** Natural items without a swapped line broke the build's stratum counts (a None sort). There is a regression test.
- **Correction to §3.4, found by the upstream equivalence check.** §3.4 says the tools overwrite the status on
  success, so the two twin routes agree. That is wrong for two tools: `modify_pending_order_address` and
  `modify_pending_order_payment` leave the status alone.
  - **What happens.** For `status` items on those two tools, the patch route shows the patched status
    (`pending`), while the bypass route shows the order's real status.
  - **Which twin stands.** The bypass twin is the designed one: only the violated check is removed, and nothing
    in the state is edited.
  - **How the check changed.** The equivalence script now puts the true status back on the patch route for
    those two tools before comparing, and counts how often it does so.
  - **Whom it affects.** 561 items, all `status` items whose aligned task is not achievable, so none is in the
    main population or the natural set.
  - **Before the fix,** every other generated and natural item agreed byte for byte.

## Amendments after review 1, before any score (2026-11-20)

Three reviews read the design and the built items before any model was scored: the engineering review (review 1:
mutation testing, an edge-case equivalence check against upstream, a token check on every item), the blind second
checker (its notes on ambiguities in §7), and the scientific-validity review. Everything below was decided and
committed before any GPU run. Where an amendment replaces an earlier section, the earlier text stands only as a
record. No score of any model on these items exists.

### A1. What this benchmark tests

τ-bench does not replicate the ALFWorld headline, where the fact that decides the failure usually lies outside the
named object (the room, another receptacle). Here the history always shows the named object's own deciding record:
the target order's status, its item's variant availability, its original payment method, or a gift card's balance
against its payment. So the question is narrower and stated as such: **does the task override a visible fact about
the object it names?** The closest ALFWorld analogues are the 788 items where the target object was seen without
the deciding fact being hidden, and the mechanism diagnostic's facts-stated arm. Results are compared with those,
not with the hidden-state headline.

### A2. The prompt makes the failure determined for the model, not only for our checker

**Problem.** The policy text is written for the agent. It mixes rules the backend enforces (order status, variant
availability, the refund method, gift-card balances) with rules it does not (authentication, confirmation, one user
per conversation, "deny requests against this policy"). It also misstates the backend twice: it says modifying is
allowed only for status `pending`, but address and payment changes are accepted on `pending (item modified)`; and
it names the post-modification status `pending (items modifed)`. A model reading only the policy cannot know which
rules make a call fail, so "determined by the visible text" held for our checker but not for the model. Only one
few-shot example showed an enforced error.

**Decision.** The system message gains a fixed section, "How the backend checks write calls", placed after the
instruction and before the policy (`wf/tau/prompts.py`, `BACKEND_RULES`). It says that the policy is written for the
agent; that the backend sees only each call's name and arguments and the current database, not the user's messages
or task, and does not check the agent's conduct; and then, per write tool, every check the tool makes, in words that
match §7.2 exactly (status rules, items in the order, a different and available variant of the same product, the
user's payment methods, the one-payment and different-method rules, the refund-method rule, gift-card balances), and
the status each successful write leaves. Exact error strings are not given; the primary readout needs only success
against failure.

**Why this is not a leak.** The text is identical for every item, task line, action and readout. It states rules,
never an item's facts. The statement that the backend does not see the task is a fact about the system being
modelled, the counterpart of household physics not depending on the agent's goal in ALFWorld. Without it, a model
could reasonably read the swapped condition (a write on an order the user did not ask about) as a policy violation
the backend refuses, which would push ΔL positive for a reason other than the task overriding a visible fact.

**Few-shot examples, replacing §5.3.** Four examples, from four users excluded from every test set, chosen as the
first users in hash(user) order meeting each need. They form a balanced 2 × 2: the task names the call's own order
or another order of the same user, and the call succeeds or fails. No failure is of a test class. In prompt order:

1. own order, success: change the shipping address of a `pending` order;
2. other order, failure: `cancel_pending_order` on a `pending` order with the reason "found a better price",
   giving "Error: Invalid reason", while the task asks to cancel another of the user's orders;
3. other order, success: return the first item of a `delivered` order to its original payment method, while the
   task asks to return an item of another order;
4. own order, failure: change the payment method to the one already used ("Error: The new payment method should be
   different from the current one").

Each task-shape carries one success and one failure, so the shape of the task line predicts nothing about the
outcome; the first and last examples are own-order ones, one success and one failure. The verifier checks the 2 × 2
and its order.

**Why not one enforced-error example per test class.** Considered and rejected. Four more examples would add about
3,800 tokens (an example is 600 to 1,200 Qwen3 tokens: a user record and an order record) to prompts that are already
about 10,500, roughly 40% more GPU time; they would remove four more users; and an example of the same kind as a
test item (an exchange refused for status, say) would let the model answer by matching the example's shape rather
than reading the item's record. The rules section gives the same knowledge without either cost.

**Prompt-length impact** (Qwen3 / Llama-3.1 tokenizers, all scored items, output-cue prompt): median about 10,550 /
9,560 tokens, up from 9,400 / 8,520 (the rules section adds about 420 tokens and the fourth example about 600);
see A15 for the measured maxima.

### A3. Primary population: status-matched items

The confirmatory population is now the **status-matched** main-population items (the comparison order has the
target's status): 856 items from 175 users (by class, `status` 409, `unavailable` 405, `wrong_return_pm` 36,
`gc_balance` 6). The full main pool (1,385 items) is secondary.

**Why.** In unmatched items the swapped task names an order in a different status, and the artefact this creates
has opposite signs by class. For `status` items, 174 of the 255 unmatched ones have a comparison order whose status
the called tool accepts, so a model that binds the call to the order named in the task sees a call that would
succeed under swapped, pushing ΔL negative. For `unavailable` items, none of the 248 unmatched ones does (the
comparison order fails the tool's status rule), pushing ΔL positive. Pooling them mixes two artefacts of opposite
sign. In matched items the comparison order has the same status, so neither artefact arises. Swapped-task
achievability is confounded with matching: every matched item has an achievable swapped task, and all 188 items
whose swapped task is not achievable are unmatched. That stratum is reported in the full pool only and is not
interpreted separately from matching.

### A4. Success controls: the witness

Each main-population item records its aligned witness: the first single write that achieves the aligned task on the
same state (§3.6). It is scored as a **success control** under the aligned, swapped and neutral task lines, in every
readout (first token, yes/no, full replies). Its full-reply pair is its real output against the error the item's
class would give for it (for `status`, the witness tool's status error; for `unavailable`, "New item <id> not found
or available" for the witness's new item; for `wrong_return_pm` and `gc_balance`, the class's text for the witness
tool). The builder stores `witness_output`, `witness_failure` and `witness_checker` (the text-only checker's verdict
on the witness); the verifier re-derives them, and the upstream equivalence check runs every witness on upstream.

For a `status` item the witness is the other tool (`modify_pending_order_items` on a `pending` order where the agent
called exchange, and the reverse). Its success needs the new variant's availability, which the `status` items'
histories do not show, so the checker says UNKNOWN for all of them; for every other class the witness's success is
determined by the history (checker PASS).

Three prespecified uses, each on the primary population:

- **(a) Discrimination check.** Under the neutral task, the witness's first-token L minus the failing call's, paired
  by item, on the items whose witness the checker can also see (PASS). It **holds** for a model if the mean is at
  least 1 nat and its 95% cluster-bootstrap CI lies above 0. The same contrast on all items, and on the yes/no
  readout, is reported.
- **(b) The goal effect on working actions.** The witness's aligned − swapped, aligned − neutral and swapped −
  neutral differences, every readout; and per item the failing call's aligned − swapped effect minus the
  witness's, with a cluster CI ("fail minus witness").
- **(c) Reading rule.** If discrimination does not hold for a model, a non-null primary effect for that model is
  **not** read as the task overriding visible evidence: the model is not shown to use the evidence in the first
  place. The analysis writes this reading next to the verdict.

### A5. Mention-only task pair

Two more task lines: "The user has a question about order {W}." (mention-aligned) and the same with W2
(mention-swapped). They name the order without asking for any write. They are scored on the failing call with the
first-token and yes/no readouts. Reported: mention-aligned − mention-swapped, mention-aligned − neutral, aligned −
mention-aligned, and per item the write-request effect (aligned − swapped) minus the mention effect, with a cluster
CI ("write minus mention"). Readings are in the outcome table (A9).

### A6. Manipulation check

Does a ~10k-token prompt make the model read the task line at all? The failing call's prompt, with the final cue
replaced by "Question: is this call on the order the user asked about? Answer Yes if it is and No if it is not.
Answer:", is scored under the aligned and swapped task lines as log P(Yes) − log P(No). The check **passes** for a
model if, on the primary population, the aligned − swapped mean is at least 1 nat with the CI above 0. A null
verdict for a model requires its check to pass; otherwise the verdict becomes "inconclusive (null, but the task
line is not shown to be read)".

### A7. Exploratory rules-only arm

On 500 primary items (the first 500 in hash(id, "rules") order), the failing call is also scored with a system
message that drops the agent-facing policy and keeps the instruction, the backend's rules (with a lead sentence
that does not mention a policy) and the tools; the few-shot examples are the same. First-token and yes/no
readouts, aligned, swapped and neutral. This mirrors ALFWorld's rules-stated arm and asks whether the
agent-facing policy (confirmation, "deny requests against this policy") is what carries any effect. Reported
beside the main arm on the same items, with no verdict.

**Decision on the scientific review's item 7.** The recommended few-shot example where the task names another order
and the call succeeds is adopted in the main prompt as part of the balanced 2 × 2 (A2), with a matching failure so
that the task's shape stays uninformative. The rules-only arm is adopted as exploratory, on 500 items, because the
rules section (A2) is already in the main prompt and the arm then isolates only the policy's presence; it costs
about 6% more passes.

### A8. Estimates, verdicts and practical size

**Primary.** Per model, ΔL1 = mean first-token L(aligned) − L(swapped) of the failing call on the primary
population, 95% user-clustered percentile bootstrap (2,000 resamples) and user-level sign-flip permutation (1,000),
Holm over the two models, seed 20260924. The per-model rule of §6.3 is unchanged, with the gate of A6 on a null and
the reading of A4(c). Written out, with Holm-adjusted p and the CI as before, the per-model verdict is "replicates"
when

$$\overline{\Delta L_1} \ge 0.5 \text{ nats}, \quad \text{CI excludes } 0, \quad p_{\text{Holm}} < 0.01,$$

"reversed" at −0.5 or below with the same conditions, "small effect" when only the last two hold, "no effect larger
than 0.5 nats" when the CI lies inside (−0.5, 0.5) and the manipulation check passes, and "inconclusive" otherwise.

**Joint verdict, replacing the last paragraph of §6.3.** One model alone never counts as a replication. The overall
result is:

- "replicates (both models)" only if both models' verdicts are "replicates";
- "reversed (both models)" only if both are "reversed";
- "no effect larger than 0.5 nats (both models)" only if both are that null (each with its manipulation check);
- "mixed" if one replicates and the other reverses;
- "one model only (not a replication)" if exactly one model replicates or reverses and the other does not go the
  other way;
- "inconclusive" otherwise.

**Prespecified secondary estimates** (each with CI and p, no verdict), on the primary population unless named:

- the yes/no, full-reply and per-token full-reply ΔL, and aligned − neutral and swapped − neutral for each;
- per class, for `status` and `unavailable` only (first token);
- the product-clustered sensitivity analysis: ΔL1 with the bootstrap and the sign flip clustered by the target
  item's product instead of the user (51 products in the primary population; items with no target item, the gift-card
  payment changes, are their own clusters);
- practical size: the probability shift, the mean over items of the two-way probability of success under aligned
  minus under swapped, where that probability is

$$q = \frac{P(\texttt{\{"})}{P(\texttt{\{"}) + P(\texttt{Error})} = \sigma(L_1);$$

- the flip shares, named by direction: "toward success" (L1 below 0 under swapped, above 0 under aligned), "toward
  failure" (the reverse), and either. Before this amendment the code reported only the first, under a name that did
  not say so (`_flip_share`); §6.4 asked for the share of sign changes, which is "either";
- the controls of A4, A5 and A6;
- the full main pool, with its class, status-matching and swapped-achievability strata.

**Descriptive only** (estimate and n, no CI, no p): `wrong_return_pm` (36 primary items) and `gc_balance` (6), and
the natural set by class.

### A9. Outcome table, written before any data

Each row is a pattern on the primary population for one model, and what it would be taken to mean. "Effect" means
the CI excludes 0; "≈ 0" means the CI lies inside (−0.5, 0.5).

| Failing call, aligned − swapped | Discrimination (A4a) | Witness effect, and fail minus witness | Mention pair | Reading |
| --- | --- | --- | --- | --- |
| at least 0.5 nats | holds | witness ≈ 0, or fail minus witness positive | any | The task overrides a visible fact about the named object: the ALFWorld effect in its stronger form |
| at least 0.5 nats | holds | witness effect of similar size, fail minus witness ≈ 0 | any | A general task-consistency bias: naming the order raises the success belief of any call on it, failing or not; reported as goal leakage, not as an override specific to failures |
| at least 0.5 nats | fails | any | any | The task shifts predictions, but the model is not shown to use the visible record; not read as overriding evidence |
| any non-null effect | any | any | mention effect similar, write minus mention ≈ 0 | Naming the order is enough; the request to act on it adds nothing |
| any non-null effect | any | any | mention ≈ 0, write minus mention positive | The request to act drives the effect, not the name |
| −0.5 or below | holds | any | any | Naming the object sharpens failure prediction (attention to its record); the ALFWorld effect then needs the deciding fact to lie outside the named object |
| CI inside (−0.5, 0.5) | any | any | any | No leak of more than 0.5 nats, if the manipulation check passes; inconclusive if it does not |

Across models: the joint rule of A8. Across classes: a `status` effect without an `unavailable` effect (or the
reverse) is reported as class-specific; `status` items involve a wrong tool choice whose name encodes the status
rule, `unavailable` items a record of variants that is much longer (median history 7.3k characters of tool output against 2.7k).

### A10. The §6.1 fallback is dropped; the token check is a gate

§6.1 said that a model failing the token check would fall back to the first token of each full reply. That was
never implemented. The fallback is dropped rather than implemented: the token check has been run on every scored
item and every prompt variant for both models (A15) and passes, so the fallback would be untested code for a case
that does not occur. Instead `run_tau_score` writes each model's token report and **refuses to score** a model whose
report fails (non-mock runs), saying that the design has no fallback. Any other model would need a new amendment
before it is scored.

### A11. Every scoring input is pinned

The scoring configs (`tau_score`, `tau_score_smoke`, `tau_contamination`) pin the sha256 of all three build
outputs, `items.jsonl`, `natural_items.jsonl` and `fewshot.json` (`expected_outputs_sha256`). Scoring refuses a
file whose hash differs or has no pin, in addition to refusing files that `tau_verify` did not pass.

### A12. Contamination check (replacing the resampling rule of §10)

Only identifiers are resampled: every run of five or more digits in a key or string value, outside the address,
with a first digit from 1 to 9, consistently within the record. Addresses (zip codes, street and suite numbers)
stay verbatim: a zip goes with a city and state, so resampling it tests geography rather than memorisation. Prices
and balances are numbers and stay. The resampled record is serialised as the tools serialise records, so the two
texts differ only in identifier digits.

**A free control.** Llama-3.1's training data ends in December 2023, before τ-bench's release in June 2024, so its
verbatim-minus-resampled difference is the baseline for records of this shape without memorisation. The report
adds, per record kind, Qwen3's difference minus Llama's, paired by record, with a bootstrap CI over records.
tau2's retail database extends τ-bench's, so the control is approximate. Still exploratory.

### A13. Records

- **The natural `gc_balance` item is an exchange.** §3.3 defines `gc_balance` for generated items as a payment
  change. The one natural `gc_balance` item is `exchange_delivered_order_items` failing with "Insufficient gift card
  balance to pay for the price difference". Natural items are classed by their error text (§4 step 3), so it stays
  `gc_balance`; natural classes are descriptive only (A8). Its twin bypasses the exchange tool's balance check.
- **Incoherent twins in the unachievable set** (561 before the rebuild, 560 after). For `status` items on `modify_pending_order_address` and
  `modify_pending_order_payment` (all in the unachievable set), the bypass twin shows the order's real, non-pending
  status after a successful change, which the backend could never print (correction of 2026-11-20 above). Their
  full-reply readout compares the failure with an impossible success text. The unachievable set is exploratory and
  not scored by default; if it is scored, its analysis is reported with and without these items
  (`twin_status_incoherent`).
- **Flip share,** renamed and completed (A8).
- **The blind checker's ambiguities.** It agreed with ours on all 3,399 items. Its resolutions, all consistent with
  upstream and with our checker on these items, are adopted as the reading of §7: a successful write is an order
  write (or `modify_user_address`) whose output parses as a JSON object; a card is named in a write if its id equals
  a key or string value of the arguments or output (exact match, no substring); a later user record clears
  staleness; "the user's payment methods" means the latest record of the order's `user_id`; exchange's "the method
  exists" is membership in that user's methods; `order_not_found` is predicted only after a shown "Error: Order not
  found" and no later record; the count error names the first offending id in call order; the old item is the
  first order line with that id; both other gift-card errors map to `gc_balance`; balances fail only when strictly
  less; an unknown earlier check with a certain later failure gives UNKNOWN; an empty `payment_history` on a return
  gives UNKNOWN. The only semantic difference from our checker is substring against exact matching for staleness
  (ours uses substring), which changes no item.

### A14. GPU time and the smoke run

Per item and model the full run scores about 30 forward passes of a ~10.5k-token prompt: the failing call under
three task lines in the first-token, yes/no and two full-reply passes (12), the mention pair (4), the manipulation
check (2), and the witness (12); the 500 rules-only items add 6 each. That is about 45,000 passes per model and
90,000 in all, or about 5,700 per shard with 16 shards. At about 1.5 s a pass (8B in bf16 at ~10k tokens on an A100
or L40S) a shard takes about 2.5 hours, about 5 on an A40. `pace/tau_score.sbatch` asks for 8 hours, and a shard cut
off resumes where it stopped (every score is saved as it finishes; a resume refuses mixed commits, settings, item
files or item hashes).

Every score record now carries its seconds. The smoke run (20 items, both models, every job type) is read for
format and timing only: the medians of the two-way masses P(`{"`) + P(`Error`), P(Yes) + P(No) for the yes/no
question, and P(Yes) + P(No) for the manipulation question (each below 0.5 is a red flag), the share of jobs below
0.5, the token reports, the seconds per job by readout, and the projected hours per shard of the full run
(`tau_smoke_analysis`, "projection"). The full run is submitted only after the projection fits the limit.

### A15. Rebuild and checks after these amendments

All run at commit f669ac6, which holds the amendments above and their code, before any GPU run.

- **Rebuild.** 3,363 generated items from 303 users (8 fewer than before, because the fourth few-shot user's
  items leave the pool); main population 1,385 items from 295 users (`status` 664, `unavailable` 653,
  `wrong_return_pm` 61, `gc_balance` 7); primary (status-matched) 856 items from 175 users; natural 28 items, 17 with
  an achievable aligned task. Rejections: 117 candidates, all at the twin's second check. The rebuild at the commit
  is byte-identical to the build made before it; the three sha256 values are pinned in the scoring configs
  (items `97208493…`, natural `6748606e…`, few-shot `53feb550…`).
- **Verification** (`tau_verify`): PASS, 0 of 3,391 items with a problem, with the new checks (witness fields,
  mention lines, product-read order, comparison-item names, every prompt variant differing only in the task line,
  the few-shot 2 × 2) and the 100-item fresh-process sample.
- **Upstream equivalence** (`scripts/tau_upstream_equivalence.py`, tau2 at c06b07d): PASS. 10,000 seeded random calls
  over 400 episodes give identical outputs and identical databases; every item's history outputs, failure reply and
  twin are identical under upstream (3,363 generated and 28 natural; the status put back on the patch route for
  560 items, the incoherent-twin items of A13); and all 1,402 witnesses (1,385 generated, 17 natural) give
  exactly the stored witness output.
- **Upstream golden outputs** (`scripts/tau_upstream_golden.py` → `tests/data/tau_upstream_golden.json`): 15 call
  sequences covering every write tool, generated by upstream and replayed byte for byte by the port in the tests.
- **Token check, every scored item** (1,413: 1,385 main and 28 natural), both models, tokenizer files copied from
  the ICE cache (sha256 identical): PASS for both. `{"`, `Error`, `Yes` and `No` are single tokens; every twin and
  witness output starts with `{"`, every failure reply and witness failure with `Error`; no token merges at the
  prompt/answer boundary; every rendered prompt variant differs across task lines only in the task line. Prompt
  length of the main-arm output-cue prompt: Qwen3 median 10,696 tokens, maximum 15,250; Llama-3.1 median 9,675,
  maximum 13,284. The rules-only arm is about 1,430 tokens shorter.
- **Blind checker** (regenerated stripped file: id, source, history and action only): FAIL with the identical class
  and error text on 3,391 of 3,391 items. Run also on the 1,402 witness calls, it gives the same verdict as ours on
  every one (PASS on 736, UNKNOWN on the 666 `status` witnesses).
- **Mutation testing** (`scripts/tau_mutation_check.py`; this claim is corrected in B11: one of these mutants survives, and no kill was traced to a test): 86 mutants, 86 killed: the original 29, review 1's 27
  (all killed, including the 17 that survived review 1 and the three it marked as possibly equivalent), and 30 for
  the amended code. One of the new ones (flip shares counted the wrong way) first survived, because the test's
  data had as many flips each way; the test was made asymmetric and it is killed.

### A16. Caveats the paper must state

- The deciding fact is always on the named object and always visible; this is not the hidden-state setting of the
  ALFWorld headline (A1).
- 95% of main items use one template ("swap the {p} in order {o} …"); `status` items are wrong-tool choices whose
  tool name encodes the violated rule; `unavailable` histories are much longer than `status` ones.
- Failures are chosen by us, not made by an agent; the natural set (28 items) is too small for inference.
- Users are reused (about five items each) and products far more (up to 58 items per product); inference clusters
  by user, with a product-clustered sensitivity analysis.
- The prompt states the backend's rules and that the backend does not see the task (A2). That makes the failures
  determined for the model and removes a policy-violation reading of the swapped task; it may also make any leak
  smaller than it would be with the policy alone. The rules-only arm and the witness controls bound this, but do
  not remove it.
- The status witness's success is not determined by its history (variant availability is not shown), so the
  discrimination check rests on the other classes.
- The 0.5-nat first-token threshold is demanding for Llama: its ALFWorld first-token ΔL was 0.68 against Qwen's
  2.52.
- Both the database and the trajectories are public; contamination is checked only exploratorily (A12).
- One domain, one environment family; a customer-service backend whose rules are software, not physics.

## Amendments after review 2 and the scientific re-review, before any score (2026-11-20)

Review 2 (engineering: 24 new mutants, an independent token-length check on every prompt, the blind checker rerun,
a blind read of 50 rendered prompts) and the scientific re-review read the code and items at fb95207. Everything
below was decided and committed before any GPU run; no score of any model on these items exists. Where an
amendment changes an earlier one, the earlier text stands only as a record.

### B1. ALFWorld-matched arm (exploratory, prestated reading)

**Problem.** The main prompt carries two debiasing cues that ALFWorld never had: the statement that the backend does
not see the user's messages or task (A2), and few-shot examples whose task names *another* order of the user (A2's
2 × 2), which demonstrate the swapped condition itself. ALFWorld's few-shot sets had off-goal actions but never a task
naming another instance, and TextCraft's examples are own-goal only. The scientific re-review also notes that
ALFWorld never ran the combination of a rules statement with an ignore-note, which is what the main arm amounts to.
Review 2's finding 5 is the same point from the other side: no condition measures the goal effect without the
debiasing cues, so a null in the main arm could be produced by the prompt. A16's sentence "the rules-only arm and the
witness controls bound this" overstated: the rules-only arm keeps the statement that the backend does not see the
task, so it bounds the policy's contribution, not the statement's.

**Decision.** A third arm, `matched`, on the same 500 primary items as the rules-only arm (both take the first 500
primary items in hash(id, "rules") order, so the three arms are compared on identical items), failing call only,
first-token and yes/no readouts, aligned, swapped and neutral task lines (6 passes per item and model, 3,000 per
model). It differs from the main arm in exactly two things:

1. The backend section's lead drops the statement of what the backend sees. The main lead reads "The backend sees
   only each call's name and arguments and the current database: it does not see the user's messages or task, and
   it does not check the agent's conduct (authentication, confirmation with the user)." The matched lead reads "The
   backend does not check the agent's conduct (authentication, confirmation with the user)." Everything else in the
   system message (instruction, the backend's rules, the policy, the tools) is byte-identical (a test checks that
   the matched message is the main message with only the lead replaced). The scientific re-review asked to drop
   only the "does not see the user's messages or task" clause and keep the conduct clause; the "sees only each
   call's name and arguments and the current database" clause is dropped too, because it states the same fact in
   positive form, and keeping it would leave the very cue the arm is meant to remove.
2. Own-order few-shot examples. Only the task lines of examples 2 and 3 change, to name their own orders: example 2
   becomes "The user wants to cancel order {its order}." and example 3 "The user wants to return the {returned
   item's product} from order {its order}." Calls, histories and outputs are unchanged (example 2 still fails with
   "Invalid reason", example 3 still succeeds). The builder stores this line as `task_own` on every example (for
   examples 1 and 4 it equals the task), and the verifier re-derives it.

**Prestated reading.** The matched arm gets its own per-model verdict under the §6.3 rule (Holm over the two models
within the arm). Its reading combines that verdict with the confirmatory main-arm verdict of each model:

- an effect in the ALFWorld direction under both prompts: the statement that the backend does not see the task
  does not remove it;
- a null under the main prompt (with its gates passed) but an effect under the matched prompt: an explicit
  statement that the backend does not see the task, together with the other-order examples, suppresses the effect
  here; τ-bench is then read as showing the leak is removable by stating the system's blindness, not as showing no
  leak;
- a null under both: the null does not rest on the debiasing statement;
- an effect under the main prompt only: not predicted; reported, not interpreted;
- anything else: inconclusive.

Also reported per model and readout: the main arm on the same 500 items, and per item the main arm's aligned −
swapped minus the matched arm's, with a user-clustered CI. The matched arm has no manipulation check or witness of
its own; its null is therefore read only beside the main arm's gated verdict. The arm never changes the
confirmatory verdict of A8.

**Few-shot tool matching in this arm.** In the matched arm example 3 is an own-order return that succeeds, so a
`wrong_return_pm` item (a return) under the aligned task matches it in tool and task shape, a positive artefact
(the main arm's is negative; B6). The matched arm's 500 items contain about 25 such items.

### B2. Few-shot order counterbalanced by item

The scientific re-review found an ordering artefact: in the design order the most recent other-order example is a
success and the most recent own-order example a failure, so recency or shape matching pushes swapped toward success
and aligned toward failure, a negative artefact on ΔL. The order is now reversed for the items whose
h(item id, "fewshot-order") is odd (`fewshot_reversed`, about half: the verifier requires 45–55% of all items and of
the primary items). Within an item every job (arm, action, readout, task line) uses the same order, so every paired
contrast still differs only in the task line. The stored `fewshot.json` keeps the design order; the reversal happens
when the prompt is rendered.

### B3. Outcome table and joint verdict, revised (replacing A9's table and A8's joint rule)

Review 2 and the re-review found three gaps: a null or a reversal with discrimination failing was unclassified, or
read as "no leak"; "fail minus witness" negative (the witness moved more than the failing call) was unclassified;
and the joint rule called "one replicates, the other a small positive effect" one model only, and "one replicates,
the other a small negative effect" also one model only instead of mixed. Per model, on the primary population
(`verdicts`, `outcome_reading`):

| Verdict on ΔL1 | Discrimination | Fail minus witness | Reading |
| --- | --- | --- | --- |
| replicates or small positive | holds | CI above 0 | The task overrides a visible fact about the named object, more for failing calls than working ones (the ALFWorld effect in its stronger form) |
| replicates or small positive | holds | about 0, negative, or a CI that includes 0 | A general task-consistency bias: naming the order raises the success belief of calls on it whether they fail or not; goal leakage, not an override specific to failures |
| replicates or small positive | fails | any | The task shifts predictions, but the model is not shown to use the visible record; not read as overriding evidence |
| reversed | holds | any | Naming the object sharpens failure prediction |
| reversed | fails | any | Verdict becomes "inconclusive (reversed, but the model is not shown to discriminate ...)" |
| small negative | fails | any | A shift toward failure without shown use of the record: inconclusive |
| CI inside ±0.5 | holds (and the manipulation check passes) | any | No leak of more than 0.5 nats |
| CI inside ±0.5 | fails | any | Verdict becomes "inconclusive (null, but the model is not shown to discriminate ...)" |
| CI inside ±0.5, manipulation check fails | any | any | Verdict becomes "inconclusive (null, but the task line is not shown to be read)" |

"Small" means the rule's CI and Holm conditions hold but the estimate is under 0.5 nats in absolute value; its
reading carries "(below the 0.5-nat size)". The mention pair is read beside every non-null verdict: write minus
mention with a CI above 0 means the request to act adds to naming the order; a CI inside ±0.5 means naming the
order is enough; otherwise uncertain.

**Joint verdict.** Mixed if one model replicates and the other is reversed or a small negative effect, or one is
reversed and the other a small positive effect; "replicates (both models)" and "reversed (both models)" as before;
"no effect larger than 0.5 nats (both models)" only if both are that null after their gates; "ALFWorld direction in
both models, size met by one (not a replication)" if one replicates and the other is a small positive effect;
"reversed direction in both models, size met by one" for the mirror case; "one model only (not a replication)" if
exactly one replicates or reverses and the other is null or inconclusive; "inconclusive" otherwise.

### B4. Discrimination per class, and the gates

Discrimination (A4a) is read on items whose witness the text-only checker can determine (PASS): 447 of the 856
primary items, 405 of them `unavailable`. `status` items (409) have no discrimination evidence at all: their witness
needs a variant's availability that their histories do not show. So the pooled check is in effect the `unavailable`
class's. The analysis now also reports discrimination per prespecified class (`discrimination_by_class`): for
`unavailable` the check on its PASS items with its own "holds"; for `status` the flag "no discrimination evidence",
with the note that a null or a reversal in that class is not read as use of the record. The per-model gates use the
pooled check, as before, and now apply it to nulls and reversals (B3). A class-level result for `status` is
reported as a goal effect only, never as an override of evidence the model was shown to use.

### B5. The smoke run: format, timing and precision only

- **No estimates in the smoke analysis.** With `project_to` set, `tau_smoke_analysis` computes and writes no
  estimate, CI, p or verdict (review 2 found it printed the per-model ΔL and verdicts). It reports the format
  checks, the token reports, the mean seconds per job by readout, the projection, and the standard deviation of the
  failing call's per-item aligned − swapped difference per readout (no mean). A test checks that its metrics have no
  `estimate`, `ci`, `perm_p` or verdict and its status line no ΔL.
- **The two longest items.** The smoke set adds the longest generated main item and the longest natural item (by
  rendered history and call length in characters) to its 20 hash-chosen items, so the longest prompts are scored
  on the GPU before the full run.
- **Time.** The projection uses the mean seconds per job by readout (not the median) and both models' hours per
  shard added (they run one after the other in one job), and compares them with `time_limit_hours` (8, equal to the
  sbatch `--time`; a test keeps them equal). `fits_time_limit` must be true before the full run is submitted.
- **Precision (scientific re-review item 5).** From each readout's standard deviation, the projection gives the
  standard error of the primary contrast over the full run's primary items, between two bounds: no within-user
  correlation, and perfect within-user correlation, where the design effect is the effective cluster size (5.62 for
  the primary population):

$$\mathrm{SE}_{\text{low}} = \frac{s}{\sqrt{n}}, \qquad \mathrm{SE}_{\text{high}} = s\,\sqrt{\frac{m_{\text{eff}}}{n}}, \qquad m_{\text{eff}} = \frac{\sum_u n_u^2}{n},$$

  with the CI half-width 1.96 times each, for the primary population and the matched arm's 500 items. For each model
  it gives, at both bounds, the chance (normal approximation) that the point estimate reaches 0.5 nats, that the CI
  lies above 0, and both, if the model's τ effect equalled its ALFWorld first-token ΔL (Qwen3 2.52, Llama-3.1
  0.68), and the chance of a null verdict if the true effect were 0. For Llama, with ALFWorld's 0.68 only 0.18 nats
  above the threshold, the chance of reaching 0.5 is close to a coin flip unless the standard error is well below
  0.18; the projection says how close. These numbers are copied into this note and committed before the full run
  is submitted; they change no rule.

### B6. Few-shot tool matching: a prelogged sensitivity analysis

Two classes have the tool of a few-shot example: `wrong_return_pm` items are returns, like example 3 (other order,
success), and `gc_balance` items are payment changes, like example 4 (own order, failure). Under the swapped task a
`wrong_return_pm` item matches example 3's cell (other order, the same tool, success), and under the aligned task a
`gc_balance` item matches example 4's (own order, the same tool, failure); both push ΔL negative, against the
prediction. B2's counterbalancing does not remove this, because it changes positions, not cells. The two classes
are 42 of the 856 primary items (4.9%).

**Decision.** Keep them in the primary population (changing the confirmatory population again for 4.9% of items
would cost more than it saves) and prelog a sensitivity analysis: ΔL1 on the primary population without those two
classes (814 items), with the full inference, its own Holm adjustment over the two models, and its §6.3 category
beside the primary's ungated category (`sensitivity_without_fewshot_tool_classes`, `same_category`). If the
categories differ for a model, the paper reports that model's primary result as sensitive to few-shot tool
matching. The limit is also stated (A16 addition below). The `status` and `unavailable` classes (95% of the primary
items) use exchange and modify-items, which no example uses.

### B7. Wording of the backend rules, tied to the port; golden system messages

Review 2 found three loose phrasings. Now: the payment rule says "the order's payment_history has exactly one entry,
and it is a payment" (the tool checks both); modify-items and exchange say "there are as many new items as old
ones"; a first line says "Every order write needs the order to exist, and a payment method must be one of the
order's user's"; modify-items says each new item "differs from the old item it replaces, is a variant of the same
product, and is available"; exchange says each new item "is a variant of the same product and is available (the
backend accepts the old item itself as its new item, although the policy asks for a different one)", because the
exchange tool, unlike modify-items, has no different-item check. A test per tool pairs every phrase of its rule
line with constructed calls on the port that show the phrase is what the tool enforces (both sides of each status
rule, the reason list, the one-payment rule in both its parts, the different-method rule, gift-card boundaries at
the amount and one cent below, equal lengths, availability, same-product variants, the user's methods, the return
method rule, the same-item exchange). The sha256 of the rendered system message of each arm is pinned in a test,
and the yes/no, manipulation and output cues are pinned as exact text.

### B8. Mention lines name the product

The mention-only lines named only the order, so write − mention measured the request and the product's name
together. For items whose task line names a product, the mention line is now "The user has a question about the
{product} in order {order}." (the swapped one names the comparison item's product), so the two lines differ only in
the request. Items without a product (cancel, address and payment changes) keep "The user has a question about
order {order}."

### B9. Contamination contrast: tokenizer confound

The Qwen3-minus-Llama contrast of A12 is confounded by tokenization: Llama-3's tokenizer splits digit runs into
chunks of up to three digits and Qwen3's into single digits, so resampled identifiers cost the two models different
numbers of tokens and different per-token surprisal. The contrast stays exploratory and is reported with this
caveat; it is low-stakes, because the deciding facts are in the context and the failing calls are ours.

### B10. Scoring gates

A scoring config without `expected_outputs_sha256` (missing or empty) is now refused ("pins no sha256 values"),
where before it was accepted unchecked; the tests pin the small test build's own hashes.

### B11. Tests and mutation testing; correction to A15

**Correction.** A15 said "86 mutants, 86 killed". That claim was not verifiable and, for one mutant, wrong. The
runner ran pytest with `--tb=no` and recorded only killed or survived, so no kill could be traced to a test, and it
never checked that the unmutated copy passed. Review 2 reran review 1's Holm mutant (verdicts using the raw p instead
of the Holm-adjusted p) and it survived: no test had p values where Holm changes the verdict. Review 2's own 24
mutants left 12 alive: the exchange rule's "exactly", the modify-items availability clause, the "does not see the
task" clause and the gift-card return exception in the rules text (M1, M2, M3, M5); the manipulation contrast
reversed (M9); the null gate reading discrimination instead of the manipulation check (M14); discrimination under
the aligned instead of the neutral task (M15); the rules-only arm drawn per shard (M17); the item hash ignoring the
witness fields (M18); the projection taking one model's hours (M19); the manipulation question's wording (M22); and
write − mention computed against neutral (M23).

**Now.** Tests pin the direction and condition of every control on constructed scores where each condition and
action has its own value (manipulation aligned − swapped, discrimination neutral witness − neutral failing call,
fail − witness, write − mention against the mention-swapped line, arm contrasts); a verdict test with raw p = 0.006
for both models (Holm 0.012, so no replication); gate tests that set the manipulation and discrimination flags
differently; the exploratory arms drawn from the whole set (3 items over 2 shards gives exactly 3); the item hash
changing with every scored field; the projection adding both models' hours; the golden system messages and the rule
tests of B7. The mutation runner now first requires the unmutated copy to pass, runs mutants serially within a
deterministic shard (MUTANTS[i::n]), and logs for each kill the first failing test's id and error line. All of
review 2's mutants are in the script (R2-M1 to R2-M24; its adapted copies of review 1's A3, A5, A8 and G1 are the R1-
entries), with 23 more for this round's code (N31-N53).

### B12. GPU time (replacing A14's count)

Per model: about 30 passes per main item, 18 per natural item, plus 6 for each of the 500 rules-only items and 6
for each of the 500 matched-arm items (the same items): about 48,000 passes per model and 96,000 in all, about
6,000 per shard over 16 shards. At about 1.5 s a pass a shard takes about 2.5 hours (about 5 on an A40), under the
8-hour limit. The smoke projection (B5) decides.

### A16, additions

- The main prompt states that the backend does not see the task and shows other-order few-shot examples. The
  ALFWorld-matched arm (B1) measures the effect without both cues on 500 items; the rules-only arm and the witness
  controls do not bound the statement's contribution (correcting the sentence above).
- 4.9% of primary items share a tool with a few-shot example whose cell favours a negative ΔL (B6); a sensitivity
  analysis without them is prelogged.
- `status` items have no discrimination evidence (B4); any `status` effect is a goal effect, not shown to override
  evidence the model uses.
- The contamination contrast is confounded by tokenization (B9).

### B13. Rebuild and checks after these amendments

All run at commit f389f97, which holds the amendments B1–B12 and their code, before any GPU run.

- **Rebuild.** The same items as at f669ac6 (3,363 generated from 303 users; main population 1,385 from 295 users;
  primary 856 from 175 users; natural 28). The only differences from the previous build are the two mention lines
  of each item (2,276 generated and all 28 natural items now name a product in them, B8) and the `task_own` field of
  each few-shot example (B1); every history, action, failure reply, twin and witness is byte-identical. A second
  rebuild at the commit is byte-identical. New pins in `tau_score`, `tau_score_smoke` and `tau_contamination`: items
  `2c8ffeac…`, natural `9bd6c3f7…`, few-shot `231607f7…`; the gate (B10) passes for all three configs.
- **Verification** (`tau_verify`): PASS, 0 of 3,391 items with a problem, with the new checks (the mention lines
  with the product, the few-shot `task_own` lines, every prompt variant of the three arms differing only in the task
  line, the few-shot order reversed for 50.6% of all 3,391 items, 52.0% of the 1,413 scored items and 51.3% of the 856
  primary items) and the 100-item fresh-process sample (all generated).
- **Upstream equivalence** (tau2 at c06b07d): PASS. 10,000 seeded random calls with identical outputs and databases;
  every history output, failure reply and twin of the 3,363 generated and 28 natural items identical (the status put
  back on 560 items, as before); all 1,402 witnesses identical.
- **Blind checker** (rerun on a regenerated stripped file with id, source, history and action only, and on a
  stripped file of the 1,402 witnesses as actions): FAIL with the identical class and error text on 3,391 of 3,391
  items; the same verdict as ours on 1,402 of 1,402 witnesses (PASS 736, UNKNOWN 666). Compared by us item by item.
- **Tokenizers.** The local tokenizer files are sha256-identical to the ICE cache's (Qwen3-8B snapshot b968826,
  Llama-3.1-8B-Instruct snapshot 0e9e39f).
- **Token check (not completed in this round).** The full check of every scored prompt in the three arms, for both
  tokenizers, was started and stopped before it finished. It renders and tokenises about 127,000 prompts per
  tokenizer, about 5 hours on one core. Only a 20-item Qwen3 sample ran to the end: PASS, with no first-token,
  boundary or task-line problem, and a longest prompt of 13,084 tokens. The full check (`python -m
  wf.tau.token_check`, which now also reports the longest prompt over every variant) must pass for both tokenizers
  before any GPU run.
- **Smoke set.** The 20 hash-chosen main items plus the longest generated main item (tau-g-4f7fe83e22da) and the
  longest natural item (tau-n-bc2f0e115848, also the longest prompt in review 2's token count).
- **Size of the full run.** 39,744 scored jobs and 48,175 forward passes per model; the exploratory arms' 500 items
  (status 240, unavailable 235, wrong_return_pm 21, gc_balance 4) have an effective cluster size of 3.67, the primary
  population 5.62.
- **Mutation testing** (`scripts/tau_mutation_check.py`, 4 deterministic shards, mutants serial within a shard, the
  unmutated copy passing first in every shard): 133 mutants. 132 were killed on the first run, each with its killing
  test logged; N44 (the sensitivity analysis leaving nothing out) survived, because the small test build has no
  primary item of the two classes, and was killed by a new test on constructed scores
  (`test_sensitivity_population_leaves_out_exactly_the_fewshot_tool_classes`) when rerun alone. All 12 of review 2's
  survivors and review 1's Holm mutant are now killed, each by a test aimed at it (listed in the fix-round report).
- **Tests.** The full suite passes (752 passed, 1 skipped, before the N44 test was added; the four τ test files
  pass with it, 184 tests).
