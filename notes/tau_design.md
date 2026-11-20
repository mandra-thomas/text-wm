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
