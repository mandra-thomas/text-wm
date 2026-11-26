# τ-bench retail walkthrough: the benchmark, the test items, the scoring and real sample decisions

This walkthrough explains how the project uses the retail domain of τ-bench to ask whether a language model acting as a *world model* of a customer-service backend lets the user's task leak into its predictions. It describes the benchmark, shows how test items are built from it, works through how one prediction becomes a number, and walks through real decisions the models made, with the prompts they saw and the scores they gave. The last part takes each τ-bench study in turn: the confirmatory replication, its exploratory follow-up, the model-families extension, and the authorization experiment with its Llama-3.2-3B extension.

Nothing here is a new analysis. Every prompt was rendered with the project's own prompt builder at the frozen commit of its study, and every score was read from the stored score files; for every sample the item hash stored with each score was recomputed from the item and matched (section 7). Headline numbers are quoted from the committed notes and metrics files, which are cited each time.

## Contents

1. The benchmark itself
2. How the project turns τ-bench into test items
3. How a prediction is scored
4. Sample decisions
5. The studies, one by one
6. Glossary
7. Provenance, licences and how to regenerate this document

---

## 1. The benchmark itself

### 1.1 Origin

τ-bench (Yao et al., 2024, Sierra Research) is a benchmark for language-model agents that talk to a simulated user and call the tools of a simulated business backend. Its retail domain is an online shop: an agent helps a customer cancel, change, return or exchange orders, following a written policy. τ²-bench (`sierra-research/tau2-bench`) is its successor, which keeps the retail domain. The project uses τ²-bench's retail domain at commit `c06b07d`, vendored in `wf/tau/vendor/`: the database, the agent policy and the schemas of the 16 agent tools. The project does not run agents or simulated users. It uses the backend only as a source of exactly determined tool outputs, which a model is asked to predict.

### 1.2 What the backend consists of

The **database** holds fictional customers and their orders: 500 users, 50 products (each with several variants, such as a jacket's sizes and colours) and 1,000 orders. Each order has a status: 423 are `pending`, 373 `delivered`, 102 `processed` and 102 `cancelled`. A user has an address, an email, saved **payment methods** (credit cards, PayPal accounts and gift cards with a balance) and a list of orders. An order has items (each a product variant with an `item_id` and a price), a status and a payment history.

The **tools** are functions the agent calls. Read tools (`get_user_details`, `get_order_details`, `get_product_details` and others) return a database record as JSON. Write tools change an order: `cancel_pending_order`, `modify_pending_order_address`, `modify_pending_order_payment`, `modify_pending_order_items`, `exchange_delivered_order_items` and `return_delivered_order_items`. A call is written as JSON, for example

```text
{"name": "get_order_details", "arguments": {"order_id": "#W1890669"}}
```

and its output is either the JSON record (for a successful write, the updated order) or a single line starting `Error: `.

The **policy** is a text for the agent (about 1,160 words). It mixes rules the backend enforces, such as "only pending orders can be modified", with rules it does not, such as authenticating the user or obtaining explicit confirmation before a write.

### 1.3 What makes a write succeed or fail

Each write tool checks a fixed list of preconditions, in a fixed order, against the current database and the call's arguments only, and raises an error at the first one that fails, without changing anything. The checks that matter for this project are:

- the order's status: cancelling and modifying items need the status to be exactly `pending`; address and payment changes accept any status containing "pending"; exchanges and returns need `delivered`;
- the availability of a requested variant: a modify or exchange to a variant whose `available` field is false fails with "New item ... not found or available";
- the refund method of a return, which must be the order's original payment method or a gift card;
- a gift card's balance, which must cover the amount it is asked to pay.

The project ports these tools to `wf/tau/retail.py` without dependencies, keeping upstream's check order, error strings and JSON serialisation, and an equivalence script checks the port against upstream byte for byte. The design note states the key fact: "The tools are pure functions of the database and the arguments. A failed write raises before it mutates anything." ([tau_design.md](../tau_design.md) §2).

### 1.4 Nothing is hidden

Each test item's history shows, verbatim, the record that decides the call: the order's status, the variant's `available` flag, the payment history, or the gift card's balance. The backend never reads the user's task, so the task cannot change the reply. A model that reads the history and applies the rules has everything it needs, and its prediction should not change when only the task line changes. Any change is the **goal effect** (here also called the task effect) that this project measures.

---

## 2. How the project turns τ-bench into test items

The full rules are in [tau_design.md](../tau_design.md); [GUIDE.md](../GUIDE.md) §2.3 gives a shorter account.

### 2.1 Users, orders and the history

Every test item belongs to one user with at least two orders; the user is the clustering unit for all statistics. For a **target order** W and a **comparison order** W2 of the same user, the history consists of tool calls and their outputs only, with no conversation text: `get_user_details` for the user, `get_order_details` for W and for W2 in a counterbalanced order, and, for the variant class, `get_product_details` for the two products. Each output is produced by running the call on a fresh copy of the pinned database, so every byte is real.

### 2.2 The failing call and its class

The scored action is a write on W that fails exactly one precondition, the one visible in the history. Items come in four **classes**:

- **status**: the order's status fails the tool's status rule, for example an exchange (which needs `delivered`) on a `pending` order;
- **unavailable**: a modify or exchange asks for a variant whose record shows `"available": false`;
- **wrong refund method** (`wrong_return_pm`): a return refunds to a method that is neither the original payment method nor a gift card;
- **gift-card balance** (`gc_balance`): a payment change to a gift card whose balance is below the order's payment.

The failure reply is the backend's real output. The **success twin** is the output of the same call with only the violated check bypassed, so the two candidate replies differ only in whether that one check passed.

### 2.3 The witness: a working call on the same order

For each item the builder also finds a **witness**: the first single write that would achieve the aligned task from the same state, tried in a fixed order on a copy of the database. In a status item where the agent called the exchange tool on a pending order, the witness is the modify tool on the same order; in an unavailable item, it is the same call with an available variant. The witness is scored exactly like the failing call. It answers a question the failing call alone cannot: whether the task line moves predictions for calls that fail more than for calls that work. The **main population** is restricted to items whose aligned task is achievable by some witness, so each item depicts an agent that chose a wrong tool or a wrong argument for a task a right one would achieve.

### 2.4 The task lines

Only the task line of the final turn changes between conditions. Each action type has a template; for a modify or exchange it is "The user wants to swap the {product} in order {order} for a different option of the same product."

- **aligned** fills the template with the target order and its product;
- **swapped** fills it with the comparison order W2 and one of its products;
- **neutral** reads "The user wants help with their account.";
- **mention-aligned** and **mention-swapped** name the target or the comparison order and product without asking for any change: "The user has a question about the {product} in order {order}." (for payment changes, which name no product, "The user has a question about order {order}.").

The mention lines separate two things a write request does: it names an order, and it asks for a change. The **naming component** is what naming the right order does without a request, the mention-aligned minus mention-swapped difference; the **request component** is what the write request adds on top, the goal effect minus the naming component (section 5.4).

### 2.5 The primary population: status-matched items

In the confirmatory set, the comparison order W2 has the same status as W. Without that match, a swapped line could name an order on which the called tool would in fact succeed, which creates artefacts of opposite sign in different classes. The **primary population** is the 856 status-matched main items from 175 users: 409 status, 405 unavailable, 36 wrong refund method and 6 gift-card balance items ([tau_design.md](../tau_design.md), amendment A3).

### 2.6 Verification and checks

A text-only checker that never touches the database must decide from the history alone that each call fails, with the right class and the exact error text; a blind second checker agreed on all 3,391 built items. Verification regenerates every history, failure reply, twin and witness output byte for byte, and checks that prompt variants differ only in the task line ([GUIDE.md](../GUIDE.md) §2.3). For witnesses of the status class the checker returns "unknown", because their success depends on a variant's availability, which the status items' histories do not show; the 447 primary items whose witness is determined by the history (all but the status class) form the **discrimination population** used by the competence checks.

Two further checks guard each model's reading. The **manipulation check** asks, under the aligned and swapped lines, "is this call on the order the user asked about?", to show that the model reads the task line at all in a prompt of about 10,000 tokens. The **discrimination check** asks whether, under the neutral line, the model scores the witness above the failing call by at least 1 nat on the discrimination population, to show that it uses the record in the first place.

---

## 3. How a prediction is scored

### 3.1 The prompt

The prompt is a chat: a long system message, four worked examples as user and assistant turns, and the item's turn. It is about 10,500 Qwen3 tokens long. The system message has four parts (`wf/tau/prompts.py` at `c737cd6`): an instruction, a section on how the backend checks write calls, the retail policy verbatim, and the 16 tool schemas, one per line. The instruction and the backend section are the project's own text:

```text
You are a world model of the backend of a retail customer-service system. An agent is helping a user and calls the backend's tools. Each turn shows the user's task, the tool calls made so far with their outputs, and the agent's next tool call. Predict the exact output the backend returns for that call. Successful calls return JSON; failed calls return a single line starting with "Error: ". Output only the tool output.
```

```text
# How the backend checks write calls

The policy below is written for the agent. The backend sees only each call's name and arguments and the current database: it does not see the user's messages or task, and it does not check the agent's conduct (authentication, confirmation with the user). It does check the following and nothing else: a write call that breaks any of them returns an error and changes nothing.

Every order write needs the order to exist, and a payment method must be one of the order's user's.
- cancel_pending_order: the order's status is exactly "pending", and the reason is "no longer needed" or "ordered by mistake".
- modify_pending_order_address: the order's status contains "pending" (so "pending (item modified)" is accepted).
- modify_pending_order_payment: the order's status contains "pending"; the payment method is one of the user's; the order's payment_history has exactly one entry, and it is a payment; the new method differs from that payment's method; a gift card's balance is at least that payment's amount.
- modify_pending_order_items: the order's status is exactly "pending"; every old item is in the order; there are as many new items as old ones; each new item differs from the old item it replaces, is a variant of the same product, and is available; the payment method is one of the user's; a gift card's balance is at least the total price difference.
- exchange_delivered_order_items: the order's status is exactly "delivered"; every old item is in the order; there are as many new items as old ones; each new item is a variant of the same product and is available (the backend accepts the old item itself as its new item, although the policy asks for a different one); the payment method is one of the user's; a gift card's balance is at least the price difference.
- return_delivered_order_items: the order's status is exactly "delivered"; the refund method is one of the user's and is either the order's original payment method (the first entry of its payment_history) or a gift card; every item is in the order.

A successful order write returns the updated order as JSON. Afterwards its status is "cancelled" (cancel), "pending (item modified)" (modify items), "exchange requested" (exchange) or "return requested" (return); address and payment changes keep the status.
```

The backend section was added before any score because the policy, written for the agent, does not say which of its rules the backend enforces, and in two places misstates the backend ([tau_design.md](../tau_design.md), amendment A2). It states rules, never an item's facts, and is identical for every item and condition.

**The four examples** come from four users excluded from all test sets and form a balanced two-by-two: the task names the call's own order or another order of the same user, and the call succeeds or fails. In design order they are an address change on the user's own order (success); a cancellation with an invalid reason while the task asks to cancel another order ("Error: Invalid reason"); a return on one order while the task asks to return an item of another order (success); and a payment change to the method already used, on the user's own order ("Error: The new payment method should be different from the current one"). So the shape of the task line predicts nothing about the outcome. About half of the items, chosen by a hash of the item id, see the four examples in reverse order.

**The item's turn** has the form

```text
Task: <task line>

Tool calls so far:
Call: {"name": ..., "arguments": {...}}
Output: <output>

[... one block per history call ...]

Next call: {"name": ..., "arguments": {...}}
Tool output:
```

Sample 1 in section 4 shows one in full. Each model's chat template wraps the messages as in the TextCraft walkthrough (02_textcraft.md, section 3.1): Qwen3-8B with thinking off gets an empty think block at the start of the assistant turn, and the Llama models a system header with the template's fixed dates.

### 3.2 The readouts

The primary readout is the **first token** of the predicted tool output. Every success reply begins with `{"` (JSON) and every failure with `Error`, so one forward pass gives

$$L_1 = \log P(\texttt{\{"} \mid x) - \log P(\texttt{Error} \mid x),$$

in **nats** (natural-log units; one nat multiplies the odds of success by about 2.7). The first token is primary in τ-bench because the success replies are long JSON records (a median of about 486 tokens), whose exact wording the model cannot guess, so a full-reply score would mostly measure how well the model copies a record. The token checks confirmed that `{"` and `Error` are single tokens at that position for each model.

The **yes/no readout** replaces the cue "Tool output:" with a question anchored to the reply format:

```text
Question: will the backend execute this call without returning an error? Answer Yes if its output is JSON and No if its output is a line starting with "Error: ".
Answer:
```

and scores $\log P(\text{Yes}) - \log P(\text{No})$. The **manipulation question** ("Question: is this call on the order the user asked about? Answer Yes if it is and No if it is not.") is scored the same way. The confirmatory study also scored the full replies by teacher forcing (feeding each reply's tokens after the prompt and summing their log-probabilities), as a secondary readout.

### 3.3 A worked example

Sample 1 below is an exchange on a `pending` order, which the backend refuses. Qwen3-8B's stored first-token log-probabilities under the aligned task line are $-9.2501$ for `{"` and $-0.0001$ for `Error`, so

$$L_1(\text{aligned}) = -9.2501 - (-0.0001) = -9.25 .$$

Under the swapped task line they are $-13.5000$ and $-0.0000$, so $L_1(\text{swapped}) = -13.50$. The two tokens hold all of the next-token probability (first-token mass 1.0000). The item's goal effect is

$$\Delta L_1 = L_1(\text{aligned}) - L_1(\text{swapped}) = -9.25 - (-13.50) = +4.25 ,$$

close to the study mean of +3.98. As probabilities between the two openings, $\sigma(L_1) = 1/(1 + e^{-L_1})$ is about $9.6 \times 10^{-5}$ under the aligned line and $1.4 \times 10^{-6}$ under the swapped line. Both are correct, confident predictions of an error; asking for exactly this exchange makes Qwen about 70 times less sure of the error, without making it predict success.

### 3.4 From item scores to an effect

The study-level effect is the mean of the per-item differences over the 856 primary items,

$$\overline{\Delta L_1} = \frac{1}{856} \sum_{i} \left[ L_1(i, \text{aligned}) - L_1(i, \text{swapped}) \right].$$

Items of one user share a history, so intervals resample whole users (a **user-clustered bootstrap**, 2,000 resamples) and permutation tests flip the signs of whole users. The witness's own aligned minus swapped effect, and the item-by-item difference "failing call minus witness", say whether the shift is specific to failing calls.

---

## 4. Sample decisions

The samples are primary items scored under the confirmatory prompt (commit `c737cd6`). Qwen3-8B and Llama-3.1-8B-Instruct were scored in the confirmatory run `tau_score`; Llama-3.2-3B-Instruct and Mistral-7B-Instruct-v0.3 in the families runs, with the same prompts apart from each chat template. They were chosen to cover all four classes, the two status cells, a typical item, extreme and reversed items, items where the effect is absent or negative, and items where the witness behaves differently from the failing call. Shorter histories were preferred.

Each table gives, per model, the first-token log-odds of the failing call under the three task lines, its aligned minus swapped difference, the witness's aligned and swapped scores and difference, and the yes/no log-odds of the failing call under the aligned and swapped lines. All values are in nats. Mistral-7B fails the families competence rule on the first-token readout (section 5.3), so its row is shown for completeness and not interpreted.

![Two scatter plots, one per model, of each of the 856 primary failing calls' first-token log-odds under the swapped task line (horizontal) against the aligned task line (vertical). Qwen's points spread from about minus 15 to plus 27 nats and sit mostly above the diagonal; Llama's lie between about minus 7 and plus 6 nats, close to the diagonal. Eight sample items are marked and labelled.](figures/taubench_items.png)

*Figure 1. Every primary failing call under the aligned and swapped task lines, with this section's samples labelled. Points above the diagonal are calls where asking for exactly that change raised predicted success. Look at Qwen's panel: most points are above the diagonal, and a large group sits right of zero on both axes, meaning failing calls that Qwen predicts to succeed whatever the task says (sample 3). Produced by `scripts/walkthroughs/taubench_figures.py`.*

### Sample 1: a typical item (status class, exchange on a pending order)

Item `tau-g-0f4d9dfa8045`. The user has a pending order with a cycling helmet and another pending order with running shoes and a grill. The agent calls the exchange tool, which requires a delivered order. The item's turn under the aligned task, verbatim:

```text
Task: The user wants to swap the Cycling Helmet in order #W1890669 for a different option of the same product.

Tool calls so far:
Call: {"name": "get_user_details", "arguments": {"user_id": "evelyn_lopez_5487"}}
Output: {"user_id": "evelyn_lopez_5487", "name": {"first_name": "Evelyn", "last_name": "Lopez"}, "address": {"address1": "142 Chestnut Street", "address2": "Suite 757", "city": "San Diego", "country": "USA", "state": "CA", "zip": "92195"}, "email": "evelyn.lopez6910@example.com", "payment_methods": {"credit_card_3566337": {"source": "credit_card", "id": "credit_card_3566337", "brand": "mastercard", "last_four": "8951"}}, "orders": ["#W1355800", "#W1890669", "#W3007862"]}

Call: {"name": "get_order_details", "arguments": {"order_id": "#W1890669"}}
Output: {"order_id": "#W1890669", "user_id": "evelyn_lopez_5487", "address": {"address1": "142 Chestnut Street", "address2": "Suite 757", "city": "San Diego", "country": "USA", "state": "CA", "zip": "92195"}, "items": [{"name": "Cycling Helmet", "product_id": "7765186836", "item_id": "1676105083", "price": 191.56, "options": {"size": "S", "color": "blue", "ventilation": "high"}}], "status": "pending", "fulfillments": [], "payment_history": [{"transaction_type": "payment", "amount": 191.56, "payment_method_id": "credit_card_3566337"}], "cancel_reason": null, "exchange_items": null, "exchange_new_items": null, "exchange_payment_method_id": null, "exchange_price_difference": null, "return_items": null, "return_payment_method_id": null}

Call: {"name": "get_order_details", "arguments": {"order_id": "#W3007862"}}
Output: {"order_id": "#W3007862", "user_id": "evelyn_lopez_5487", "address": {"address1": "142 Chestnut Street", "address2": "Suite 757", "city": "San Diego", "country": "USA", "state": "CA", "zip": "92195"}, "items": [{"name": "Running Shoes", "product_id": "6938111410", "item_id": "1775591963", "price": 154.75, "options": {"size": "10", "color": "white", "material": "leather", "sole": "EVA"}}, {"name": "Grill", "product_id": "6819683148", "item_id": "5666020311", "price": 1058.86, "options": {"type": "electric", "size": "medium", "features": "side burner"}}], "status": "pending", "fulfillments": [], "payment_history": [{"transaction_type": "payment", "amount": 1213.61, "payment_method_id": "credit_card_3566337"}], "cancel_reason": null, "exchange_items": null, "exchange_new_items": null, "exchange_payment_method_id": null, "exchange_price_difference": null, "return_items": null, "return_payment_method_id": null}

Next call: {"name": "exchange_delivered_order_items", "arguments": {"order_id": "#W1890669", "item_ids": ["1676105083"], "new_item_ids": ["3358616356"], "payment_method_id": "credit_card_3566337"}}
Tool output:
```

The swapped task is "The user wants to swap the Running Shoes in order #W3007862 for a different option of the same product."; the neutral task "The user wants help with their account."; the mention lines "The user has a question about the Cycling Helmet in order #W1890669." and the same for the running shoes in #W3007862. The true reply is `Error: Non-delivered order cannot be exchanged`. The witness is `modify_pending_order_items` on the same order and item, to variant `1596993217`; it succeeds, but the text-only checker cannot confirm that from this history, because the variant's record is not shown.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −9.25 | −13.50 | −5.75 | +4.25 | +3.75 | −6.25 | +10.00 | −15.00 | −19.50 |
| Llama-3.1-8B | −3.25 | −5.87 | −4.12 | +2.62 | −0.38 | −2.37 | +2.00 | −1.00 | −1.50 |
| Llama-3.2-3B | +0.87 | −1.37 | −1.25 | +2.25 | +1.25 | −1.62 | +2.87 | −1.13 | −2.00 |
| Mistral-7B | +1.44 | +1.19 | −1.37 | +0.25 | +2.25 | +2.19 | +0.06 | +5.88 | +4.88 |

Qwen and Llama-3.1 correctly predict the error under every line, and the aligned line makes them less sure of it; Llama-3.2 even tips to predicting success under the aligned line. The working witness moves more than the failing call for Qwen (+10.00 against +4.25): under the swapped task Qwen predicts an error for this working modify call, under the aligned task success. The mention lines give Qwen −10.50 (target order) and −14.00 (other order) on the failing call, so naming the right order without any request already moves it by +3.50 nats of its +4.25. On the manipulation question ("is this call on the order the user asked about?") Qwen scores −4.00 under the aligned line and −22.00 under the swapped line: on this item it leans to "No" under both, but the 18-nat difference shows that it reads the task line.

### Sample 2: the witness moves by 29 nats (unavailable class)

Item `tau-g-6a254cc8d278`. The user wants a different fleece jacket; the agent asks for variant `4728397765`, whose product record shows it is not available. Excerpt of the item's turn (square brackets mark what this walkthrough left out; the full turn is 4,904 characters):

```text
Task: The user wants to swap the Fleece Jacket in order #W4802126 for a different option of the same product.

Tool calls so far:
Call: {"name": "get_user_details", "arguments": {"user_id": "noah_hernandez_4232"}}
Output: {"user_id": "noah_hernandez_4232", [name, address and email omitted] "payment_methods": {"gift_card_3410768": {"source": "gift_card", "id": "gift_card_3410768", "balance": 56.0}}, "orders": ["#W3897284", "#W4802126"]}

Call: {"name": "get_order_details", "arguments": {"order_id": "#W3897284"}}
Output: {"order_id": "#W3897284", [...] "items": [{"name": "E-Reader", "product_id": "3801771308", "item_id": "5418781403", "price": 267.58, [...]}], "status": "pending", [...]}

Call: {"name": "get_order_details", "arguments": {"order_id": "#W4802126"}}
Output: {"order_id": "#W4802126", [...] "items": [{"name": "Fleece Jacket", "product_id": "8560156827", "item_id": "8733974883", "price": 153.18, "options": {"size": "L", "color": "red", "zipper": "half"}}, {"name": "Smart Thermostat", [...]}], "status": "pending", [...] "payment_history": [{"transaction_type": "payment", "amount": 415.65, "payment_method_id": "gift_card_3410768"}], [...]}

Call: {"name": "get_product_details", "arguments": {"product_id": "3801771308"}}
Output: {"name": "E-Reader", "product_id": "3801771308", "variants": {[six variants omitted]}}

Call: {"name": "get_product_details", "arguments": {"product_id": "8560156827"}}
Output: {"name": "Fleece Jacket", "product_id": "8560156827", "variants": {[five variants omitted] "8733974883": {"item_id": "8733974883", "options": {"size": "L", "color": "red", "zipper": "half"}, "available": true, "price": 153.18}, "4728397765": {"item_id": "4728397765", "options": {"size": "M", "color": "black", "zipper": "full"}, "available": false, "price": 149.48}}}

Next call: {"name": "modify_pending_order_items", "arguments": {"order_id": "#W4802126", "item_ids": ["8733974883"], "new_item_ids": ["4728397765"], "payment_method_id": "gift_card_3410768"}}
Tool output:
```

The true reply is `Error: New item 4728397765 not found or available`. The witness asks for variant `5992316252` instead, which the history shows as available, so the checker confirms that it succeeds.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +1.25 | −4.00 | +0.75 | +5.25 | +28.13 | −1.25 | +29.38 | −6.25 | −18.00 |
| Llama-3.1-8B | −2.25 | −2.13 | −2.75 | −0.12 | −1.50 | −1.25 | −0.25 | −1.00 | −0.50 |
| Llama-3.2-3B | +1.25 | +1.38 | −1.00 | −0.12 | +2.75 | +2.62 | +0.13 | −2.00 | −1.62 |
| Mistral-7B | +1.63 | +0.87 | +0.12 | +0.75 | +2.13 | +1.81 | +0.31 | +4.75 | +4.75 |

Qwen's first token wrongly favours success for the failing call under the aligned and neutral lines and favours the error under the swapped line: the task line flips its predicted outcome. On the working witness, Qwen is certain of success under the aligned line (+28) and predicts an error under the swapped line (−1.25). That 29-nat swing on a call that works shows the shift is a pull toward "calls the user asked for succeed", not an override specific to failures. Llama-3.1 predicts errors for both calls and is not moved, which is its typical behaviour in this class: its effect on the 405 unavailable items is −0.33 nats.

### Sample 3: a failing call that Qwen predicts will succeed under every line (wrong refund method)

Item `tau-g-4c09d40843dc`. The running shoes in order #W1842597 were paid with a gift card; the return asks to refund them to a PayPal account, which is neither the original method nor a gift card. Excerpt:

```text
Task: The user wants to return the Running Shoes from order #W1842597.

Tool calls so far:
Call: {"name": "get_user_details", "arguments": {"user_id": "fatima_anderson_7445"}}
Output: {"user_id": "fatima_anderson_7445", [name, address and email omitted] "payment_methods": {"gift_card_8070316": {"source": "gift_card", "id": "gift_card_8070316", "balance": 64.0}, "paypal_7697967": {"source": "paypal", "id": "paypal_7697967"}}, "orders": ["#W9183908", "#W1842597", "#W6368178"]}

Call: {"name": "get_order_details", "arguments": {"order_id": "#W9183908"}}
Output: {"order_id": "#W9183908", [...] "items": [{"name": "Espresso Machine", [...]}], "status": "delivered", [...] "payment_history": [{"transaction_type": "payment", "amount": 3081.91, "payment_method_id": "gift_card_8070316"}], [...]}

Call: {"name": "get_order_details", "arguments": {"order_id": "#W1842597"}}
Output: {"order_id": "#W1842597", [...] "items": [{"name": "Running Shoes", "product_id": "6938111410", "item_id": "9791469541", "price": 147.05, [...]}], "status": "delivered", [...] "payment_history": [{"transaction_type": "payment", "amount": 147.05, "payment_method_id": "gift_card_8070316"}], [...]}

Next call: {"name": "return_delivered_order_items", "arguments": {"order_id": "#W1842597", "item_ids": ["9791469541"], "payment_method_id": "paypal_7697967"}}
Tool output:
```

The true reply is `Error: Payment method should be the original payment method`; the witness returns the shoes to the gift card.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +22.00 | +7.00 | +20.25 | +15.00 | +30.75 | +11.25 | +19.50 | +7.25 | −9.75 |
| Llama-3.1-8B | +3.12 | +2.25 | +2.00 | +0.87 | +5.62 | +6.63 | −1.00 | +1.25 | +1.37 |
| Llama-3.2-3B | +4.50 | +2.75 | +3.50 | +1.75 | +3.63 | +3.12 | +0.50 | −0.75 | −0.75 |
| Mistral-7B | +2.50 | +2.31 | +0.75 | +0.19 | +4.06 | +3.50 | +0.56 | +5.75 | +4.50 |

Every model's first token predicts success for this failing call under every task line: none applies the refund rule. The follow-up found that Qwen's first token predicts success for all 36 wrong-refund-method returns ([tau_dx.md](../tau_dx.md)). The task line still moves Qwen by 15 nats, on the success side throughout. Only Qwen's yes/no answer under the swapped line says "No", which, as sample 6 shows, has more to do with the task than with the refund method.

### Sample 4: a reversed item (gift-card balance)

Item `tau-g-ca918df46a5f`. Order #W7941031 was paid 2,198.09 by credit card; the call moves that payment to a gift card whose balance is 56.00 (both figures are in the history). Its turn has the same shape as sample 3's, with the call `{"name": "modify_pending_order_payment", "arguments": {"order_id": "#W7941031", "payment_method_id": "gift_card_7794233"}}`; the task lines are "The user wants to change the payment method of order #W7941031." (aligned) and the same for #W3657213 (swapped). The true reply is `Error: Insufficient gift card balance to pay for the order`; the witness pays with the user's PayPal account instead.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +4.50 | +6.00 | +1.25 | −1.50 | +23.25 | +6.50 | +16.75 | +7.25 | +0.00 |
| Llama-3.1-8B | −2.00 | −3.37 | −1.25 | +1.37 | −1.63 | −2.37 | +0.75 | +0.00 | −0.50 |
| Llama-3.2-3B | +0.62 | −1.13 | −0.62 | +1.75 | +2.00 | +1.25 | +0.75 | −0.75 | −1.38 |
| Mistral-7B | −0.88 | −1.50 | −1.69 | +0.63 | −0.12 | −0.75 | +0.62 | +4.62 | +3.13 |

Qwen misses the balance rule and predicts success under every line; here the aligned line moves it slightly toward the error (−1.50), while the working witness gets a 16.75-nat lift. The six gift-card items in the primary population average +0.08 nats for Qwen (computed from the stored scores for this walkthrough), the only class without a clear positive effect ([tau_dx.md](../tau_dx.md)). Llama-3.1 gets the failing call right, but also predicts an error for the working witness.

### Sample 5: the same user with the roles of the two orders exchanged (status class)

Item `tau-g-5b42db1f2d81` belongs to the same user as sample 1, with the same history. Here the target is the running shoes in order #W3007862 and the comparison order is the helmet order #W1890669; the call is again an exchange on a pending order (`"order_id": "#W3007862", "item_ids": ["1775591963"], "new_item_ids": ["9635758562"]`), the aligned task names the running shoes and the swapped task the helmet. The prompt is otherwise sample 1's, with the four examples in design order rather than reversed.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −9.25 | −5.25 | −5.00 | −4.00 | +4.75 | +5.00 | −0.25 | −14.50 | −17.25 |
| Llama-3.1-8B | −3.38 | −3.63 | −4.75 | +0.25 | +0.25 | +0.25 | +0.00 | −0.87 | −1.00 |
| Llama-3.2-3B | −0.12 | +0.25 | −2.50 | −0.38 | +0.25 | +0.25 | +0.00 | −1.37 | −1.13 |
| Mistral-7B | +3.00 | +3.12 | −0.62 | −0.12 | +3.63 | +3.88 | −0.25 | +5.62 | +4.75 |

Everything about the backend is the same as in sample 1; only which order the call targets has changed. Yet Qwen's effect reverses (−4.00 here against +4.25 there), and the witness does not move at all. This mirror pair is a reminder that single items are noisy and that a per-item effect depends on details (here, plausibly, which product name appears in which line) that the study mean averages over: about 22% of Qwen's primary items do not move toward success.

### Sample 6: the yes/no answer contradicts the predicted output (status class, witness)

Item `tau-g-77169873036d` is an exchange on a pending smart-watch order, paid with a gift card; the comparison order holds a skateboard. Its turn has sample 1's shape. The interesting call here is the working witness, `modify_pending_order_items` on the smart watch.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −1.50 | −3.75 | −1.00 | +2.25 | +14.50 | +3.50 | +11.00 | −7.75 | −15.50 |
| Llama-3.1-8B | −1.75 | −5.00 | −4.00 | +3.25 | −0.38 | −2.62 | +2.25 | −0.88 | −0.88 |
| Llama-3.2-3B | −0.75 | −3.37 | −2.75 | +2.62 | +0.25 | −3.00 | +3.25 | −1.13 | −3.25 |
| Mistral-7B | +1.25 | +1.13 | −0.62 | +0.12 | +1.69 | +1.69 | +0.00 | +4.63 | +3.75 |

Qwen's first token predicts JSON for the witness under all three lines (+14.50, +3.50 and +9.50 under the neutral line). Asked the yes/no question about the same witness call, it answers +4.25 under the aligned line, −15.75 under the swapped line and −2.75 under the neutral line: a confident "No, this call will return an error" for a call whose output it itself predicts to be JSON. The question is anchored to the reply format, yet Qwen still answers partly about whether the call serves the user's task. The follow-up counts this: among calls whose first token confidently predicts a JSON success, Qwen answers "No" for 90–93% under the swapped line, 51–67% under neutral and 17–32% under aligned ([tau_dx.md](../tau_dx.md)).

### Sample 7: an effect absent or negative in both reference models (unavailable class, exchange on a delivered order)

Item `tau-g-f3ca5064c885`. A delivered order contains a notebook; the exchange asks for a notebook variant whose record shows `"available": false` (the history has five calls and is 5,370 characters long). The comparison order holds running shoes.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −4.25 | −2.50 | −1.75 | −1.75 | +1.00 | +5.50 | −4.50 | −8.25 | −7.50 |
| Llama-3.1-8B | −3.25 | −2.13 | −3.12 | −1.13 | −0.88 | −0.75 | −0.13 | −0.75 | −0.38 |
| Llama-3.2-3B | +0.62 | +0.00 | −2.25 | +0.62 | +2.50 | +1.00 | +1.50 | −2.38 | −1.88 |
| Mistral-7B | +2.00 | +1.38 | −0.25 | +0.62 | +3.50 | +3.06 | +0.44 | +4.12 | +3.88 |

Both reference models predict the error correctly, and the aligned line makes them slightly *more* sure of it. For Qwen the witness also moves the "wrong" way. Items like this are why the study mean is a mean: Qwen's effect is positive in 78% of primary items, not all.

### Sample 8: four models, four behaviours (unavailable class, modify on a pending order)

Item `tau-g-183ce6de13f7`. The call asks to swap an e-reader for a variant shown as unavailable; the comparison order holds running shoes.

| Model | fail aligned | fail swapped | fail neutral | fail, aligned minus swapped | witness aligned | witness swapped | witness, aligned minus swapped | yes/no fail aligned | yes/no fail swapped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +5.25 | −5.75 | +0.00 | +11.00 | +23.25 | −3.50 | +26.75 | −4.50 | −18.50 |
| Llama-3.1-8B | −3.88 | −3.12 | −4.38 | −0.75 | −2.00 | −0.62 | −1.37 | −1.50 | −1.12 |
| Llama-3.2-3B | +1.37 | −1.50 | −2.50 | +2.88 | +2.63 | −0.38 | +3.00 | −1.50 | −2.25 |
| Mistral-7B | +1.38 | +1.25 | +0.25 | +0.13 | +1.81 | +1.50 | +0.31 | +4.75 | +4.50 |

Qwen's predicted outcome for the failing call flips from success (aligned) to error (swapped), with an 11-nat shift and a 27-nat shift on the witness. Llama-3.1 is right under every line and moves slightly the other way. Llama-3.2-3B flips too, more modestly. Mistral predicts success for everything and answers Yes to everything, so its near-zero effect reflects a model that does not separate working from failing calls.

---

## 5. The studies, one by one

### 5.1 The confirmatory replication (2026-11-22)

**What it asked.** On ALFWorld, naming the action's object in the task makes a world model more confident that an action the history shows will fail will succeed. τ-bench was the third environment, with one difference stated in advance: here the deciding fact is a visible attribute of the named object itself (the order's status, the variant's availability). So the question was narrower: does the task override a visible fact about the object it names? The design was frozen at commit `c737cd6` (tag `taubench-design-frozen-2026-11-21`; [tau_design.md](../tau_design.md)).

**The rule fixed in advance.** Per model, on the 856 primary items, the verdict is "replicates (ALFWorld direction)" when the first-token goal effect is at least 0.5 nats, its interval excludes zero and the Holm-adjusted p is below 0.01; "small effect, direction stated" when it is significant but below 0.5 nats; "reversed", "no effect larger than 0.5 nats" or "inconclusive" otherwise. A null needed the manipulation check (the task line is read) and the discrimination check (the record is used).

**What it found.** From [tau_results.md](../tau_results.md) and [tau_metrics.json](../tau_metrics.json):

| Model | Failing call, aligned minus swapped | Witness, aligned minus swapped | Failing minus witness | Discrimination check | Manipulation check | Verdict |
|---|---|---|---|---|---|---|
| Qwen3-8B | +3.98 [3.61, 4.35] | +9.45 [8.96, 9.93] | −5.47 [−5.88, −5.05] | +7.60, holds | +18.97, passes | replicates (ALFWorld direction) |
| Llama-3.1-8B | +0.47 [0.41, 0.54] | +0.38 [0.32, 0.45] | +0.09 [0.04, 0.14] | +1.10, holds | +4.37, passes | small effect, direction positive |

The joint verdict, by the rule fixed in advance, is "ALFWorld direction in both models, size met by one; not a replication". For Qwen the witness moves more than twice as much as the failing call, so its effect is read as a general task-consistency bias, not an override specific to failures (samples 1, 2, 4 and 8). Against the neutral line, much of Qwen's effect is the swapped task lowering predicted success (−2.56), as in TextCraft. Qwen's yes/no effect is +7.68 nats, about half of it beyond its own predicted output (sample 6); Llama's yes/no readout barely moves at all (−0.02). In practical terms, under the aligned task Qwen answers "Yes" to 57% of unavailable-variant failing calls, against 10% under neutral; Llama's decisions flip on about 1–2% of items.

**The same run's exploratory arms.** On 500 primary items the failing call was also scored with the policy removed ("rules only") and with the statement that the backend does not see the task removed ("ALFWorld-matched"). On sample 3, Qwen's first-token scores under aligned and swapped are +22.00 and +7.00 with the full prompt, +21.50 and +7.00 with rules only, and +16.75 and +6.25 in the matched arm; on sample 5 they are −9.25 and −5.25, −8.00 and −4.50, and −11.25 and −5.25. The effect remains under both prompts for both models and is larger without the statement ([tau_results.md](../tau_results.md)).

### 5.2 The exploratory follow-up (2026-11-22)

**What it asked.** After reading the confirmatory result, a CPU-only analysis of the stored scores asked what the effect is, where it lives and how competent each model is ([tau_dx.md](../tau_dx.md), metrics in [tau_dx_metrics.json](../tau_dx_metrics.json)). It changes no verdict.

**What it found.** Qwen's effect is broad (positive in 78% of items and 94% of users) and robust in size. Llama's pooled +0.47 averages +1.23 on the status class with −0.33 on the unavailable class (sample 2 and sample 7 against sample 1), and its failure-specific part rests on one cell, so the follow-up advises against reading it as "the ALFWorld effect in its stronger form". The mention lines separate naming from requesting: for Qwen a bare question about the target order instead of the other order already moves the failing call by +1.81 nats, and the write request adds +2.17 more (sample 1's +3.50 and +0.75 are one item's version of this split); for Llama naming does nothing (−0.03) and the request adds +0.50. Both models discriminate working from failing calls under the neutral line (first-token AUC about 0.80), but their probabilities are poor: Qwen's first token predicts success for 38% of failing calls (sample 3 is one of all 36 returns it gets wrong). Which of two explanations produces the request component, a task-relevance bias or a belief that the backend refuses writes the user did not request, could not be decided from these scores; the authorization experiment of section 5.4 was built to decide it.

### 5.3 The model-families extension (2026-11-23)

**What it asked.** Whether the τ-bench goal effect appears in two more instruction models, Mistral-7B-Instruct-v0.3 (another developer) and Llama-3.2-3B-Instruct (Meta's next release), and whether it is shared by working and failing calls, as in both reference models. The design was frozen at tag `tau-families-design-frozen-2026-11-23` ([tau_families_design.md](../tau_families_design.md), results in [tau_families_results.md](../tau_families_results.md), numbers in [tau_families_metrics.json](../tau_families_metrics.json)). Gemma-2-9B was excluded before any data: the fixed part of every prompt is 9,575 Gemma tokens, more than Gemma 2's 8,192-token trained context, so it is reported as not run, never as a null. A third gate was added: the families **competence rule**, which needs a first-token AUC of at least 0.70 between witness and failing call on the 447 determined items, plus coverage conditions.

**What it found.**

| Model | Failing call, aligned minus swapped | Witness, aligned minus swapped | Failing minus witness | Competence (first-token AUC) | Verdict |
|---|---|---|---|---|---|
| Llama-3.2-3B | +0.99 [0.90, 1.08] | +1.07 [0.98, 1.16] | −0.08 [−0.12, −0.03] | 0.78, passes | replicates (ALFWorld direction) |
| Mistral-7B-v0.3 | +0.23 [0.18, 0.27] | +0.25 [0.20, 0.30] | −0.02 [−0.04, −0.00] | 0.58, fails | not interpretable |

Llama-3.2-3B's shift is shared by working and failing calls, with working calls moving slightly more, which is Qwen's label; 82% of its items move toward success (samples 1, 6 and 8). Its yes/no readout fails the competence rule because it answers No to nearly every call. Mistral barely separates working from failing calls and answers Yes to everything (every Mistral row in section 4), and an exploratory check found its effect depends on where its chat template places the system text (moving it to the first turn more than doubles it on 300 items, +0.53 against +0.23).

![A dot plot with four model rows, each with two markers and their 95% intervals: the failing call in blue circles and the working witness on the same order in orange squares. Qwen3-8B's failing-call effect is about 4 nats and its witness effect about 9.5 nats; Llama-3.1-8B and Llama-3.2-3B have both effects between about 0.4 and 1.1 nats, close together; Mistral-7B's markers are hollow, near 0.2 nats.](figures/taubench_fail_witness.png)

*Figure 2. The goal effect on the failing call and on the working witness, per model, with 95% user-bootstrap intervals, from [tau_families_metrics.json](../tau_families_metrics.json) (`primary.first.aligned_minus_swapped` and `checks.witness_effect`; the reference models' entries are the confirmatory numbers). Hollow markers: Mistral fails the competence rule. Look at whether the orange square sits left of the blue circle: in no interpretable model does the working call move less than the failing call, so in none is the shift specific to failures.*

### 5.4 The authorization experiment (2026-11-23) and its Llama-3.2-3B extension (2026-11-25)

**What it asked.** Is the request component a wrong belief about the backend, namely that it refuses writes the user did not request (an **authorization** belief, which a clear description of the backend should remove), or a **task-relevance** bias that persists when the backend is stated, and shown, to ignore the task? Exploratory, with a frozen procedure: tag `tau-auth-design-frozen-2026-11-22` ([tau_auth_design.md](../tau_auth_design.md), results in [tau_auth_results.md](../tau_auth_results.md), numbers in [tau_auth_metrics.json](../tau_auth_metrics.json)); the Llama-3.2-3B extension was tagged `tau-auth-ext-llama32-2026-11-25` before any of its scores existed (numbers in [tau_auth_l32_metrics.json](../tau_auth_l32_metrics.json)).

**The three backend descriptions (arms).** Everything else (192 items, 96 per class from the status and unavailable classes, five task lines, histories, calls, readouts and example order) is identical across arms.

The **main** arm is the confirmatory prompt, which already says the backend does not see the task. The **task-blind** arm adds a closing paragraph to the backend section,

```text
The user's task never changes a call's output: the backend does not compare a call with the user's request. A write call that changes an order the user's task does not ask to change, or that is made when the task asks for no change (only a question about an order, or general help with the account), is executed exactly as it would be if the user had asked for it. A write call that breaks one of the rules above returns that rule's error whatever the task.
```

and two more examples, each a minimal pair with an existing one that changes only the task line and keeps the same output, so the prompt both states and shows that the task does not change the output. The **authorization** arm describes a counterfactual backend that does check the request ("The backend sees each call's name and arguments, the current database and the user's task …") with the rule

```text
Request rule, checked after the rules above: a write call is executed only if the user's task asks for that change to that order. A write call that changes an order the user's task does not ask to change, or that is made when the task asks for no change (only a question about an order, or general help with the account), returns "Error: The user did not request this change" and changes nothing. A write call that breaks one of the rules above returns that rule's error whatever the task.
```

and its examples show that error. It is never read for accuracy; it is the **positive control**, showing whether the model's predictions follow a stated backend rule at all.

**The quantities.** For one item, arm and call, write $L(c)$ for the first-token score under task line $c$, with al, sw, ma and ms for aligned, swapped, mention-aligned and mention-swapped. The goal effect, the naming component and the request component are

$$G = L(\text{al}) - L(\text{sw}), \qquad M = L(\text{ma}) - L(\text{ms}), \qquad W = G - M .$$

For a component $X$, the part the task-blind description removes and the part that persists are

$$Z_X = X^{\text{main}} - X^{\text{blind}}, \qquad R_X = X^{\text{blind}} .$$

A removed part is "authorization-like", a persisting part "relevance-like". The naming component is the control: a bare question requests no write, so an authorization belief has nothing to say about it, while a general suppression of task effects would shrink it too. The positive control is the extra goal effect on the witness when the backend is described as enforcing the request rule rather than ignoring the task,

$$PC = G^{\text{auth, witness}} - G^{\text{blind, witness}} ,$$

and must be at least 1 nat with its interval above zero.

**The comprehension gate.** On 24 items, each arm's prompt ends instead with four general questions about the backend (K1 to K4), scored as Yes against No:

- K1: "Question: about this backend in general: if a write call broke none of the rules listed for its tool but changed an order that the user's task does not ask to change, would the backend execute it without returning an error? Answer Yes or No." (correct: Yes in the main and task-blind arms, No under authorization);
- K2: "… if a write call did what the user's task asks but broke one of the rules listed for its tool, would the backend execute it without returning an error? …" (correct: No in every arm);
- K3: "… if the user's task were only a question about an order, and a write call changed that order and broke none of the rules listed for its tool, would the backend return an error? …" (correct: No in the main and task-blind arms, Yes under authorization);
- K4: "… if a write call did what the user's task asks and broke none of the rules listed for its tool, would the backend execute it without returning an error? …" (correct: Yes in every arm).

The task-blind arm passes when the correct answer is the more probable one on at least 80% of the 24 items for each question. On comprehension item `tau-g-02c211baed0b`, the stored scores (log-odds of Yes against No, so positive means Yes) are:

| Model | arm | K1 | K2 | K3 | K4 |
|---|---|---|---|---|---|
| Qwen3-8B | main | −2.75 | −23.75 | −19.00 | +14.50 |
| Qwen3-8B | task-blind | +6.25 | −15.50 | −15.00 | +15.50 |
| Qwen3-8B | authorization | −12.00 | −21.50 | −13.50 | +14.75 |
| Llama-3.1-8B | main | −0.38 | −2.00 | −1.25 | +1.00 |
| Llama-3.1-8B | task-blind | +1.37 | +1.12 | −1.00 | +2.12 |
| Llama-3.1-8B | authorization | −0.38 | −2.13 | +1.75 | +0.50 |
| Llama-3.2-3B | main | +0.00 | −1.50 | −0.87 | +1.12 |
| Llama-3.2-3B | task-blind | +1.37 | −1.63 | +0.00 | +2.00 |
| Llama-3.2-3B | authorization | −0.25 | −2.37 | +2.00 | +1.25 |

Under the task-blind description Qwen answers all four correctly and by wide margins. Under the confirmatory prompt it answers K1 "No", the authorization belief stated in words, yet answers K3 correctly, so its answers there do not express a consistent request rule. Llama-3.1-8B answers K2 "Yes" under the task-blind description (a rule-breaking write would go through), which is the error that fails its gate on all 24 items; Llama-3.2-3B is exactly indifferent on K3 here, the question it fails overall (10 of 24 correct).

**One item under the three arms.** For sample 1 (`tau-g-0f4d9dfa8045`), the failing call's first-token scores and the three quantities are:

| Model | arm | aligned | swapped | mention-aligned | mention-swapped | goal effect | naming component | request component | witness goal effect |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | main | −9.25 | −13.50 | −10.50 | −14.00 | +4.25 | +3.50 | +0.75 | +10.00 |
| Qwen3-8B | task-blind | −6.50 | −11.50 | −5.00 | −10.25 | +5.00 | +5.25 | −0.25 | +7.50 |
| Qwen3-8B | authorization | −5.75 | −12.25 | −5.25 | −11.25 | +6.50 | +6.00 | +0.50 | +10.25 |
| Llama-3.1-8B | main | −3.25 | −5.87 | −4.50 | −6.38 | +2.62 | +1.88 | +0.75 | +2.00 |
| Llama-3.1-8B | task-blind | −1.25 | −4.38 | −4.25 | −5.75 | +3.13 | +1.50 | +1.63 | +1.88 |
| Llama-3.1-8B | authorization | −1.87 | −5.50 | −3.62 | −6.00 | +3.63 | +2.38 | +1.25 | +2.12 |
| Llama-3.2-3B | main | +0.87 | −1.37 | −1.00 | −2.63 | +2.25 | +1.63 | +0.62 | +2.87 |
| Llama-3.2-3B | task-blind | +2.25 | −0.50 | −0.38 | −2.37 | +2.75 | +2.00 | +0.75 | +2.75 |
| Llama-3.2-3B | authorization | +1.38 | −1.75 | −1.00 | −3.12 | +3.13 | +2.12 | +1.00 | +4.00 |

Telling the model, and showing it, that the backend ignores the task does not shrink this item's goal effect for any of the three models; for Qwen it grows slightly. The item's positive control is the witness goal effect under authorization minus under task-blind: +2.75 for Qwen, +0.24 for Llama-3.1 and +1.25 for Llama-3.2. On sample 2 (`tau-g-6a254cc8d278`) the same pattern holds for Qwen, whose naming component is large (+9.50 under the main prompt, +4.75 task-blind) and whose witness goal effect stays near 30 nats in every arm (+29.38, +28.00, +34.50).

**What it found.** From [tau_auth_results.md](../tau_auth_results.md) (summary and §12):

| | Qwen3-8B | Llama-3.1-8B | Llama-3.2-3B |
|---|---|---|---|
| Comprehension of the task-blind arm | passes (K1–K4 each 24/24) | fails (K2 0/24) | fails (K3 10/24) |
| Positive control, nats | +3.38 [+3.01, +3.77], passes | +0.30 [+0.19, +0.40], fails | +0.41 [+0.34, +0.48], fails |
| Request component, main prompt | +2.30 [+1.87, +2.74] | +0.54 [+0.43, +0.66] | +0.46 [+0.35, +0.57] |
| Request component, removed by the task-blind arm | −0.09 [−0.42, +0.26] | −0.20 [−0.29, −0.11] | +0.20 [+0.12, +0.29] |
| Request component, persisting under the task-blind arm | +2.39 [+1.94, +2.87] | +0.74 [+0.62, +0.85] | +0.26 [+0.14, +0.38] |
| Naming component, main prompt | +1.87 [+1.39, +2.36] | −0.08 [−0.21, +0.05] | +0.54 [+0.42, +0.67] |
| Naming component, persisting under the task-blind arm | +1.95 [+1.58, +2.33] | −0.07 [−0.17, +0.04] | +0.67 [+0.50, +0.84] |
| Category, both components | relevance-like (reading (a)) | inconclusive (gate) | inconclusive (gate) |

For Qwen3-8B both gates passed, and the task-blind description removed none of the request component: a removal as large as the margin (0.72 nats) is excluded, the interval's upper end being +0.26. Both of its components are "relevance-like". Both Llama models fail both gates, so their categories are "inconclusive (gate)"; their numbers are reported, not read. For Llama-3.2-3B the apparent removal of +0.20 mirrors the naming component's growth through one line (the question about the target order), not a drop of the write request's advantage, which is the opposite of what an authorization belief predicts (§12.3, exploratory reading).

![Two side-by-side dot plots, one for the request component and one for the naming component, with three model rows each. Each row has a blue circle for the main prompt and an orange square for the task-blind prompt, with 95% intervals. For Qwen3-8B the two markers nearly coincide at about 2.3 nats (request) and 1.9 nats (naming). The two Llama rows are hollow, below 1 nat.](figures/taubench_auth.png)

*Figure 3. The request and naming components under the confirmatory and the task-blind backend descriptions, from [tau_auth_metrics.json](../tau_auth_metrics.json) and [tau_auth_l32_metrics.json](../tau_auth_l32_metrics.json) (`models.<model>.primary.W` and `.M`, entries `main` and `R`). Hollow markers: the model failed a gate, so its category is "inconclusive". Look at Qwen's two markers: describing and demonstrating a task-blind backend leaves both components where they were, which is what a task-relevance bias, and not a belief about authorization, predicts.*

---

## 6. Glossary

**Aligned, swapped, neutral task lines.** A request for exactly the change the scored call makes, on its own order and product; the same kind of request for another order of the same user; and "The user wants help with their account.".

**Authorization belief.** The (wrong) belief that the backend refuses writes the user did not request.

**AUC.** The probability that a randomly chosen working call scores above a randomly chosen failing call; 0.5 is chance.

**Classes.** Status, unavailable variant, wrong refund method, gift-card balance: which precondition the failing call breaks.

**Competence rule (families).** First-token AUC at least 0.70 between witness and failing call on the 447 determined items, plus coverage conditions; a model that fails it is "not interpretable".

**Comprehension questions (K1–K4).** General yes/no questions about how the described backend treats requests, used as the authorization experiment's first gate.

**Discrimination check.** Under the neutral line, the witness must score at least 1 nat above the failing call on the determined items.

**Failing minus witness.** The item-by-item difference between the failing call's goal effect and the witness's; above zero means the shift is specific to failing calls.

**First-token readout.** The log-odds of `{"` (a JSON success) against `Error` at the first position of the predicted output.

**Manipulation check.** The question "is this call on the order the user asked about?", showing that the task line is read.

**Mention lines; naming and request components.** Task lines that only ask a question about an order; the part of the goal effect that naming the order produces, and the part the write request adds.

**Positive control.** The extra goal effect on the witness when the backend is described as enforcing the request rule; shows that the model follows a stated backend rule.

**Primary population.** The 856 status-matched main items from 175 users.

**Removed and persisting parts.** How much of a component the task-blind description takes away, and how much remains under it.

**Success twin.** The output of the same call with only the violated check bypassed.

**User-clustered bootstrap.** Intervals that resample whole users, because one user's items share a history.

**Witness.** A single write on the same order that the backend accepts and that would achieve the aligned task.

**Yes/no readout.** The log-odds of "Yes" against "No" after the question whether the backend will execute the call without an error.

---

## 7. Provenance, licences and how to regenerate this document

**Scores.** Confirmatory run `tau_score` (16 shards, commit `c737cd6`), with its build `tau_build`; families runs `tau_family_score_llama3.2-3b` and `tau_family_score_mistral-7b-v0.3` (commit `a078a8d`); authorization run `tau_auth_score` (commit `f70b285`); its extension `tau_auth_l32_score` (tag `tau-auth-ext-llama32-2026-11-25`). The item files of the four builds are identical, item for item. The score folders are archived outside the repository.

**Verification.** `scripts/walkthroughs/taubench_samples.py` renders every turn with `wf/tau/prompts.py` at `c737cd6` (and the arms with `wf/tau/auth.py` at `f70b285`), recomputes `wf.tau.pipeline.item_hash` of each sample, and stops unless it equals the `item_hash` stored on every score record of that item in every run (80 to 254 records per sample). It matched for every sample. That hash covers the item's history, call, replies, task lines and witness; the system message and examples are pinned separately by each run's own checks. The excerpts of samples 2 and 3 were shortened by hand from the rendered text, and every omission is marked in square brackets. `scripts/walkthroughs/taubench_figures.py` draws the figures. Neither script rescores anything:

```text
python scripts/walkthroughs/taubench_samples.py --runs RUNS_ROOT --trees TREES_ROOT
python scripts/walkthroughs/taubench_figures.py --runs RUNS_ROOT --out notes/walkthroughs/figures
```

**Licences.** τ-bench (`sierra-research/tau-bench`) is under the MIT licence, "Copyright (c) 2024 Sierra"; τ²-bench (`sierra-research/tau2-bench`), whose retail database, policy and tool schemas the project vendors in `wf/tau/vendor/`, is under the MIT licence, "Copyright (c) 2025 Sierra Research" (notice in `wf/tau/vendor/LICENSE` and `NOTICE`). Both licence files were read on 2026-11-26. The quoted database records are fictional customers from that database, quoted only as far as each example needs; the policy text is not reproduced. See [DATA_LICENSES.md](../../DATA_LICENSES.md) for the project's conventions.
