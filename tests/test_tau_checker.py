"""The text-only checker (wf/tau/checker.py): it reads only the history's text, predicts exactly the port's errors,
and says UNKNOWN when a fact it needs is not shown."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from wf.tau import checker
from wf.tau.checker import check, error_class
from wf.tau.retail import base_db, fresh_env
from tests.test_tau_retail import STATUS_CASES, _args, _env_with_status, _order

ROOT = Path(__file__).resolve().parents[1]


def _hist(env, calls):
    return [{"name": n, "arguments": a, "output": env.call(n, a)} for n, a in calls]


def _reads(env, o, products=()):
    calls = [("get_user_details", {"user_id": o["user_id"]}), ("get_order_details", {"order_id": o["order_id"]})]
    calls += [("get_product_details", {"product_id": p}) for p in products]
    return _hist(env, calls)


def test_checker_never_loads_the_environment_or_the_database():
    code = "import sys, wf.tau.checker; print([m for m in sys.modules if m.startswith('wf.tau.retail') or m == 'wf.tau.items'])"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
    src = Path(checker.__file__).read_text(encoding="utf-8")
    assert "retail" not in src.split('"""', 2)[2].split("TEST_CLASSES")[0]  # no import of the port
    assert "db.json" not in src and "base_db" not in src


@pytest.mark.parametrize("tool,status,expected", [c for c in STATUS_CASES if c[2]])
def test_status_failures_match_the_port_exactly(tool, status, expected):
    env, o = _env_with_status(status)
    action = {"name": tool, "arguments": _args(tool, o)}
    v = check(_reads(env, o), action)
    assert (v.verdict, v.cls, v.error) == ("FAIL", "status", expected)
    assert env.call(tool, action["arguments"]) == v.error


def test_the_checker_reads_the_text_not_the_database():
    env = fresh_env()
    o = _order(lambda o: o["status"] == "delivered")
    hist = _reads(env, o)
    a = _args("return_delivered_order_items", o)
    assert check(hist, {"name": "return_delivered_order_items", "arguments": a}).verdict == "PASS"
    edited = [dict(h) for h in hist]
    edited[1]["output"] = edited[1]["output"].replace('"status": "delivered"', '"status": "processed"')
    v = check(edited, {"name": "return_delivered_order_items", "arguments": a})
    assert (v.verdict, v.error) == ("FAIL", "Error: Non-delivered order cannot be returned")


def test_unknown_when_a_needed_fact_is_not_shown():
    env = fresh_env()
    db = base_db()
    o = _order(lambda o: o["status"] == "delivered")
    order_only = _hist(env, [("get_order_details", {"order_id": o["order_id"]})])
    assert check([], {"name": "cancel_pending_order", "arguments": {"order_id": o["order_id"], "reason": "x"}}).evidence == "order not shown"
    assert check(order_only, {"name": "return_delivered_order_items", "arguments": _args("return_delivered_order_items", o)}).verdict == "UNKNOWN"
    # the status alone decides a status failure, so the user need not be shown
    assert check(order_only, {"name": "cancel_pending_order", "arguments": {"order_id": o["order_id"], "reason": "no longer needed"}}).verdict == "FAIL"
    it = o["items"][0]
    new = next(k for k in db["products"][it["product_id"]]["variants"] if k != it["item_id"])
    ex = {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": [new], "payment_method_id": o["payment_history"][0]["payment_method_id"]}
    v = check(_reads(env, o), {"name": "exchange_delivered_order_items", "arguments": ex})
    assert v.verdict == "UNKNOWN" and "not shown" in v.evidence
    v = check(_reads(env, o, [it["product_id"]]), {"name": "exchange_delivered_order_items", "arguments": ex})
    assert v.verdict in ("FAIL", "PASS")


def _gc_case():
    o = _order(lambda o: "pending" in o["status"] and len(o["payment_history"]) == 1 and any(
        m["source"] == "gift_card" and p != o["payment_history"][0]["payment_method_id"]
        for p, m in base_db()["users"][o["user_id"]]["payment_methods"].items()))
    gc = next(p for p, m in base_db()["users"][o["user_id"]]["payment_methods"].items()
              if m["source"] == "gift_card" and p != o["payment_history"][0]["payment_method_id"])
    return o, gc


def _set_balance(hist, uid, gc, bal):
    out = [dict(h) for h in hist]
    rec = json.loads(out[0]["output"])
    rec["payment_methods"][gc]["balance"] = bal
    out[0]["output"] = json.dumps(rec)
    return out


def test_gift_card_balance_boundary_and_staleness():
    env = fresh_env()
    o, gc = _gc_case()
    amount = o["payment_history"][0]["amount"]
    hist = _reads(env, o)
    act = {"name": "modify_pending_order_payment", "arguments": {"order_id": o["order_id"], "payment_method_id": gc}}
    assert check(_set_balance(hist, o["user_id"], gc, amount), act).verdict == "PASS"
    v = check(_set_balance(hist, o["user_id"], gc, round(amount - 0.01, 2)), act)
    assert (v.verdict, v.cls, v.error) == ("FAIL", "gc_balance", "Error: Insufficient gift card balance to pay for the order")
    # a later successful write that names the card makes the shown balance stale
    later = {"name": "modify_pending_order_payment", "arguments": {"order_id": "#W1", "payment_method_id": gc},
             "output": json.dumps({"order_id": "#W1", "items": [], "status": "pending", "user_id": o["user_id"]})}
    assert check(_set_balance(hist, o["user_id"], gc, 0.0) + [later], act).verdict == "UNKNOWN"


def test_the_latest_record_of_an_order_counts():
    env = fresh_env()
    o = _order(lambda o: o["status"] == "pending")
    hist = _reads(env, o)
    child = env.fork()
    write = {"name": "cancel_pending_order", "arguments": {"order_id": o["order_id"], "reason": "no longer needed"}}
    write["output"] = child.call(write["name"], write["arguments"])
    act = {"name": "modify_pending_order_address", "arguments": _args("modify_pending_order_address", o)}
    assert check(hist, act).verdict == "PASS"
    v = check(hist + [write], act)
    assert (v.verdict, v.error) == ("FAIL", "Error: Non-pending order cannot be modified")
    assert child.call(act["name"], act["arguments"]) == v.error


def test_random_writes_with_full_history_the_checker_is_always_right():
    """Property: with the user, order and every product shown, the checker never says UNKNOWN, its FAIL text is
    exactly the port's error, and its PASS is a success."""
    rng = random.Random(7)
    db = base_db()
    orders = sorted(db["orders"])
    seen = {"FAIL": 0, "PASS": 0}
    for _ in range(600):
        o = db["orders"][rng.choice(orders)]
        u = db["users"][o["user_id"]]
        it = rng.choice(o["items"])
        variants = sorted(db["products"][it["product_id"]]["variants"])
        pm = rng.choice(sorted(u["payment_methods"]) + [o["payment_history"][0]["payment_method_id"]])
        new = rng.choice(variants)
        tool = rng.choice(["cancel_pending_order", "modify_pending_order_address", "modify_pending_order_payment",
                           "modify_pending_order_items", "exchange_delivered_order_items", "return_delivered_order_items"])
        args = {**_args(tool, o)}
        if "payment_method_id" in args:
            args["payment_method_id"] = pm
        if "new_item_ids" in args:
            args["item_ids"], args["new_item_ids"] = [it["item_id"]], [new]
        if tool == "cancel_pending_order":
            args["reason"] = rng.choice(["no longer needed", "ordered by mistake", "other"])
        env = fresh_env()
        hist = _reads(env, o, sorted({i["product_id"] for i in o["items"]}))
        v = check(hist, {"name": tool, "arguments": args})
        out = env.call(tool, args)
        assert v.verdict != "UNKNOWN", (tool, args, v)
        if v.verdict == "FAIL":
            assert out == v.error, (tool, args, v, out)
        else:
            assert not out.startswith("Error"), (tool, args, out)
        seen[v.verdict] += 1
    assert seen["FAIL"] > 100 and seen["PASS"] > 50


def test_error_class_names_every_known_error():
    cases = {
        "Error: Non-pending order cannot be cancelled": "status", "Error: Non-delivered order cannot be returned": "status",
        "Error: New item 123 not found or available": "unavailable",
        "Error: Payment method should be the original payment method": "wrong_return_pm",
        "Error: Insufficient gift card balance to pay for the order": "gc_balance",
        "Error: Payment method not found": "pm_not_found", "Error: Variant not found": "variant_missing",
        "Error: There should be exactly one payment for a pending order": "one_payment",
        "Error: The new payment method should be different from the current one": "same_pm",
        "Error: Invalid reason": "invalid_reason", "Error: Some item not found": "item_not_in_order",
        "Error: 123 not found": "item_not_in_order", "Error: Number of 123 not found.": "item_not_in_order",
        "Error: The number of items to be exchanged should match.": "length_mismatch",
        "Error: The new item id should be different from the old item id": "same_item",
        "Error: Order not found": "order_not_found", "Error: something else": "other",
    }
    assert {k: error_class(k) for k in cases} == cases
