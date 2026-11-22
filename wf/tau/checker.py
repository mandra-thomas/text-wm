"""Text-only checker (notes/tau_design.md §7): from the history's tool outputs alone, will this write fail, why,
and with exactly which error text?

It never reads the database or imports the environment port (a test enforces this). It knows the tools'
preconditions from the documented order of checks (design §7.2) and the facts the history prints. When a fact
it needs is not shown, it answers UNKNOWN rather than guess.

`check(history, action)` -> Verdict(verdict, cls, error, evidence); verdict is "FAIL", "PASS" or "UNKNOWN".
`history` is a list of {"name", "arguments", "output"}; `action` is {"name", "arguments"}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

TEST_CLASSES = ("status", "unavailable", "wrong_return_pm", "gc_balance")
_WRITES = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
           "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
           "return_delivered_order_items"}
VALID_REASONS = ("no longer needed", "ordered by mistake")


@dataclass(frozen=True)
class Verdict:
    verdict: str
    cls: str | None = None
    error: str | None = None
    evidence: str = ""


def _fail(cls: str, message: str, evidence: str) -> Verdict:
    return Verdict("FAIL", cls, f"Error: {message}", evidence)


def _unknown(evidence: str) -> Verdict:
    return Verdict("UNKNOWN", None, None, evidence)


def _record(output: str) -> dict | None:
    if not isinstance(output, str) or output.startswith("Error"):
        return None
    try:
        obj = json.loads(output)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


class View:
    """What the history shows: the latest order, user and product records, and later successful writes."""

    def __init__(self, history: list[dict]):
        self.orders: dict[str, tuple[int, dict]] = {}
        self.users: dict[str, tuple[int, dict]] = {}
        self.products: dict[str, tuple[int, dict]] = {}
        self.writes: list[tuple[int, dict]] = []
        for i, h in enumerate(history):
            rec = _record(h.get("output"))
            if rec is not None:
                if {"order_id", "items", "status"} <= rec.keys():
                    self.orders[rec["order_id"]] = (i, rec)
                elif {"user_id", "payment_methods"} <= rec.keys():
                    self.users[rec["user_id"]] = (i, rec)
                elif {"product_id", "variants"} <= rec.keys():
                    self.products[rec["product_id"]] = (i, rec)
            if h.get("name") in _WRITES and isinstance(h.get("output"), str) and not h["output"].startswith("Error"):
                self.writes.append((i, h))

    def user(self, user_id: str) -> tuple[int, dict] | None:
        return self.users.get(user_id)

    def balance(self, user_id: str, pm: str) -> float | None:
        """A gift card's shown balance, or None if not shown or changed by a later successful write."""
        shown = self.users.get(user_id)
        if shown is None:
            return None
        i, u = shown
        m = u["payment_methods"].get(pm)
        if m is None or "balance" not in m:
            return None
        for j, h in self.writes:
            if j > i and (pm in json.dumps(h.get("arguments")) or pm in h.get("output", "")):
                return None
        return m["balance"]


def _count_problem(item_ids: list, order: dict) -> str | None:
    all_ids = [it["item_id"] for it in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_ids.count(item_id):
            return item_id
    return None


def _pm_facts(view: View, order: dict, pm: str) -> tuple[Verdict | None, dict | None]:
    """(a verdict if the payment-method check fails or cannot be decided, else the method's record)."""
    shown = view.user(order["user_id"])
    if shown is None:
        return _unknown("user not shown"), None
    _, u = shown
    if pm not in u["payment_methods"]:
        return _fail("pm_not_found", "Payment method not found", f"{pm} not in {order['user_id']}'s payment_methods"), None
    return None, u["payment_methods"][pm]


def _items_and_prices(view: View, order: dict, item_ids: list, new_ids: list, modify: bool) -> tuple[Verdict | None, float]:
    """The per-pair checks of modify/exchange in upstream order, and the summed price difference."""
    diff = 0.0
    for old, new in zip(item_ids, new_ids):
        if modify and old == new:
            return _fail("same_item", "The new item id should be different from the old item id", f"{old} = {new}"), 0.0
        item = next(it for it in order["items"] if it["item_id"] == old)
        shown = view.products.get(item["product_id"])
        if shown is None:
            return _unknown(f"product {item['product_id']} not shown"), 0.0
        k, product = shown
        variant = product["variants"].get(new)
        if variant is None:
            return _fail("variant_missing", "Variant not found", f"{new} not a variant of {item['product_id']}"), 0.0
        if not variant["available"]:
            return _fail("unavailable", f"New item {new} not found or available",
                         f"call {k}: product {item['product_id']} variant {new} available=false"), 0.0
        diff += variant["price"] - item["price"]
    return None, diff


def check(history: list[dict], action: dict) -> Verdict:
    name, args = action.get("name"), action.get("arguments") or {}
    if name not in _WRITES or name == "modify_user_address":
        return _unknown("not an order write")
    view = View(history)
    oid = args.get("order_id")
    if oid not in view.orders:
        return _unknown("order not shown")
    i, order = view.orders[oid]
    status = order["status"]
    where = f"call {i}: order {oid} status '{status}'"

    if name == "cancel_pending_order":
        if status != "pending":
            return _fail("status", "Non-pending order cannot be cancelled", where)
        if args.get("reason") not in VALID_REASONS:
            return _fail("invalid_reason", "Invalid reason", repr(args.get("reason")))
        return Verdict("PASS")

    if name == "modify_pending_order_address":
        if "pending" not in status:
            return _fail("status", "Non-pending order cannot be modified", where)
        return Verdict("PASS")

    pm = args.get("payment_method_id")
    if name == "modify_pending_order_payment":
        if "pending" not in status:
            return _fail("status", "Non-pending order cannot be modified", where)
        bad, method = _pm_facts(view, order, pm)
        if bad:
            return bad
        ph = order["payment_history"]
        if len(ph) != 1 or ph[0]["transaction_type"] != "payment":
            return _fail("one_payment", "There should be exactly one payment for a pending order", f"{len(ph)} entries")
        if ph[0]["payment_method_id"] == pm:
            return _fail("same_pm", "The new payment method should be different from the current one", pm)
        if method.get("source") == "gift_card":
            bal = view.balance(order["user_id"], pm)
            if bal is None:
                return _unknown(f"balance of {pm} not shown or stale")
            if bal < ph[0]["amount"]:
                return _fail("gc_balance", "Insufficient gift card balance to pay for the order",
                             f"{pm} balance {bal} < payment {ph[0]['amount']}")
        return Verdict("PASS")

    if name == "return_delivered_order_items":
        if status != "delivered":
            return _fail("status", "Non-delivered order cannot be returned", where)
        bad, method = _pm_facts(view, order, pm)
        if bad:
            return bad
        original = order["payment_history"][0]["payment_method_id"]
        if method.get("source") != "gift_card" and pm != original:
            return _fail("wrong_return_pm", "Payment method should be the original payment method",
                         f"{pm} is {method.get('source')}, original is {original}")
        if _count_problem(args.get("item_ids") or [], order) is not None:
            return _fail("item_not_in_order", "Some item not found", "")
        return Verdict("PASS")

    # modify_pending_order_items / exchange_delivered_order_items
    modify = name == "modify_pending_order_items"
    item_ids, new_ids = args.get("item_ids") or [], args.get("new_item_ids") or []
    if modify and status != "pending":
        return _fail("status", "Non-pending order cannot be modified", where)
    if not modify and status != "delivered":
        return _fail("status", "Non-delivered order cannot be exchanged", where)
    missing = _count_problem(item_ids, order)
    if missing is not None:
        return _fail("item_not_in_order", f"{missing} not found" if modify else f"Number of {missing} not found.", missing)
    if len(item_ids) != len(new_ids):
        return _fail("length_mismatch", "The number of items to be exchanged should match" + ("" if modify else "."), "")
    bad, diff = _items_and_prices(view, order, item_ids, new_ids, modify)
    if bad:
        return bad
    bad, method = _pm_facts(view, order, pm)
    if bad:
        return bad
    if method.get("source") == "gift_card":
        bal = view.balance(order["user_id"], pm)
        if bal is None:
            return _unknown(f"balance of {pm} not shown or stale")
        needed = diff if modify else round(diff, 2)
        if bal < needed:
            what = "the new item" if modify else "the price difference"
            return _fail("gc_balance", f"Insufficient gift card balance to pay for {what}", f"balance {bal} < {needed}")
    return Verdict("PASS")


def error_class(text: str) -> str:
    """The class of a real tool error text (for natural items and the build log); 'other' if unknown."""
    rules: list[tuple[str, Any]] = [
        ("status", lambda m: m.startswith("Error: Non-pending order") or m.startswith("Error: Non-delivered order")),
        ("unavailable", lambda m: m.startswith("Error: New item ") and m.endswith(" not found or available")),
        ("wrong_return_pm", lambda m: m == "Error: Payment method should be the original payment method"),
        ("gc_balance", lambda m: m.startswith("Error: Insufficient gift card balance")),
        ("pm_not_found", lambda m: m == "Error: Payment method not found"),
        ("one_payment", lambda m: m == "Error: There should be exactly one payment for a pending order"),
        ("same_pm", lambda m: m == "Error: The new payment method should be different from the current one"),
        ("invalid_reason", lambda m: m == "Error: Invalid reason"),
        ("same_item", lambda m: m == "Error: The new item id should be different from the old item id"),
        ("length_mismatch", lambda m: m.startswith("Error: The number of items to be exchanged should match")),
        ("variant_missing", lambda m: m == "Error: Variant not found"),
        ("order_not_found", lambda m: m == "Error: Order not found"),
        ("item_not_in_order", lambda m: m == "Error: Some item not found" or (m.startswith("Error: ") and m.endswith(" not found")
                                                                              and m[7:].split(" ")[0].isdigit()) or m.startswith("Error: Number of ")),
    ]
    for cls, rule in rules:
        if rule(text):
            return cls
    return "other"
