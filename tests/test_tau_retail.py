"""The τ-bench retail port (wf/tau/retail.py): known answers, upstream golden strings, isolation, bypass."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from wf.tau import retail
from wf.tau.retail import RetailEnv, State, base_db, fresh_env, pristine_sha256

# Golden strings printed by upstream tau2-bench at commit c06b07d (Environment.get_response content).
UP_ORDER_W4140680 = (
    '{"order_id": "#W4140680", "user_id": "anya_garcia_3271", "address": {"address1": "615 Laurel Lane", "address2": '
    '"Suite 552", "city": "Philadelphia", "country": "USA", "state": "PA", "zip": "19036"}, "items": [{"name": '
    '"E-Reader", "product_id": "3801771308", "item_id": "4273929280", "price": 244.95, "options": {"screen size": '
    '"7-inch", "connectivity": "Wi-Fi + Cellular", "storage": "32GB"}}, {"name": "Bookshelf", "product_id": '
    '"8600330539", "item_id": "8895454203", "price": 504.65, "options": {"material": "glass", "color": "white", '
    '"height": "5 ft"}}, {"name": "Bluetooth Speaker", "product_id": "4768869376", "item_id": "9440686670", "price": '
    '298.91, "options": {"color": "green", "battery life": "20 hours", "water resistance": "no"}}], "status": '
    '"cancelled", "fulfillments": [{"tracking_id": ["447833475637"], "item_ids": ["4273929280", "8895454203", '
    '"9440686670"]}], "payment_history": [{"transaction_type": "payment", "amount": 1048.51, "payment_method_id": '
    '"gift_card_4374071"}, {"transaction_type": "refund", "amount": 1048.51, "payment_method_id": "gift_card_4374071"}], '
    '"cancel_reason": null, "exchange_items": null, "exchange_new_items": null, "exchange_payment_method_id": null, '
    '"exchange_price_difference": null, "return_items": null, "return_payment_method_id": null}')
UP_USER_AFTER_CANCEL = (
    '{"user_id": "anya_garcia_3271", "name": {"first_name": "Anya", "last_name": "Garcia"}, "address": {"address1": '
    '"615 Laurel Lane", "address2": "Suite 552", "city": "Philadelphia", "country": "USA", "state": "PA", "zip": '
    '"19036"}, "email": "anya.garcia2061@example.com", "payment_methods": {"gift_card_4374071": {"source": "gift_card", '
    '"id": "gift_card_4374071", "balance": 3056.33}, "credit_card_8955149": {"source": "credit_card", "id": '
    '"credit_card_8955149", "brand": "visa", "last_four": "8674"}}, "orders": ["#W4140680", "#W6310710", "#W6436609"]}')


def _order(pred):
    db = base_db()
    return next(o for _, o in sorted(db["orders"].items()) if pred(o))


def _user(o):
    return base_db()["users"][o["user_id"]]


def test_vendored_files_match_the_manifest_and_upstream_db_hash():
    assert retail.check_vendor() == []
    m = retail.vendor_manifest()
    assert m["commit"] == retail.PINNED_COMMIT
    assert m["db_json_sha256_uncompressed"] == "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"


def test_a_tampered_vendored_file_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "vendor"
    shutil.copytree(retail.VENDOR, copy)
    (copy / "policy.md").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(retail, "VENDOR", copy)
    assert retail.check_vendor() == ["policy.md"]
    with pytest.raises(SystemExit):
        retail.load_db_from_disk()


def test_outputs_match_upstream_golden_strings():
    env = fresh_env()
    assert env.call("get_order_details", {"order_id": "#W4140680"}) == UP_ORDER_W4140680
    out = env.call("cancel_pending_order", {"order_id": "#W6436609", "reason": "ordered by mistake"})
    assert '"status": "cancelled"' in out and '"cancel_reason": "ordered by mistake"' in out
    assert out.endswith('"return_items": null, "return_payment_method_id": null}')
    assert env.call("get_user_details", {"user_id": "anya_garcia_3271"}) == UP_USER_AFTER_CANCEL


def test_reads_and_errors_of_unknown_records():
    env = fresh_env()
    assert env.call("get_order_details", {"order_id": "#W0000000"}) == "Error: Order not found"
    assert env.call("get_user_details", {"user_id": "nobody_1"}) == "Error: User not found"
    assert env.call("get_product_details", {"product_id": "0"}) == "Error: Product not found"
    assert env.call("get_item_details", {"item_id": "0"}) == "Error: Item not found"
    assert env.call("no_such_tool", {}) == "Error: Tool 'no_such_tool' not found."
    assert env.call("calculate", {"expression": "(10 - 3) * 1.5"}) == "10.5"
    assert env.call("find_user_id_by_email", {"email": "ANYA.GARCIA2061@example.com"}) == "anya_garcia_3271"
    assert json.loads(env.call("list_all_product_types", {}))["E-Reader"] == "3801771308"


STATUS_CASES = [
    # (tool, status of the order, expected error or None for "the status check passes")
    ("cancel_pending_order", "processed", "Error: Non-pending order cannot be cancelled"),
    ("cancel_pending_order", "pending (item modified)", "Error: Non-pending order cannot be cancelled"),
    ("modify_pending_order_address", "delivered", "Error: Non-pending order cannot be modified"),
    ("modify_pending_order_address", "pending (item modified)", None),
    ("modify_pending_order_payment", "cancelled", "Error: Non-pending order cannot be modified"),
    ("modify_pending_order_items", "pending (item modified)", "Error: Non-pending order cannot be modified"),
    ("modify_pending_order_items", "delivered", "Error: Non-pending order cannot be modified"),
    ("exchange_delivered_order_items", "pending", "Error: Non-delivered order cannot be exchanged"),
    ("return_delivered_order_items", "pending", "Error: Non-delivered order cannot be returned"),
    ("return_delivered_order_items", "return requested", "Error: Non-delivered order cannot be returned"),
]


def _args(tool, o):
    u = _user(o)
    pm0 = o["payment_history"][0]["payment_method_id"]
    it = o["items"][0]
    alt = next((v for v in base_db()["products"][it["product_id"]]["variants"] if v != it["item_id"]), it["item_id"])
    return {"cancel_pending_order": {"order_id": o["order_id"], "reason": "no longer needed"},
            "modify_pending_order_address": {"order_id": o["order_id"], **{k: u["address"][k] for k in
                                             ("address1", "address2", "city", "state", "country", "zip")}},
            "modify_pending_order_payment": {"order_id": o["order_id"], "payment_method_id": pm0},
            "modify_pending_order_items": {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": [alt],
                                           "payment_method_id": pm0},
            "exchange_delivered_order_items": {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": [alt],
                                               "payment_method_id": pm0},
            "return_delivered_order_items": {"order_id": o["order_id"], "item_ids": [it["item_id"]], "payment_method_id": pm0}}[tool]


def _env_with_status(status):
    """(env, order) with an order in `status`; the two derived statuses are reached by a real successful write."""
    env = fresh_env()
    db = base_db()
    if status == "pending (item modified)":
        o = _order(lambda o: o["status"] == "pending" and o["payment_history"][0]["payment_method_id"].startswith("credit_card")
                   and any(v["available"] and k != o["items"][0]["item_id"]
                           for k, v in db["products"][o["items"][0]["product_id"]]["variants"].items()))
        it = o["items"][0]
        alt = next(k for k, v in db["products"][it["product_id"]]["variants"].items() if v["available"] and k != it["item_id"])
        out = env.call("modify_pending_order_items", {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": [alt],
                                                      "payment_method_id": o["payment_history"][0]["payment_method_id"]})
    elif status == "return requested":
        o = _order(lambda o: o["status"] == "delivered")
        out = env.call("return_delivered_order_items", _args("return_delivered_order_items", o))
    else:
        return env, _order(lambda o: o["status"] == status)
    assert json.loads(out)["status"] == status
    return env, env.state.orders[o["order_id"]]


@pytest.mark.parametrize("tool,status,expected", STATUS_CASES)
def test_status_rules_follow_upstream_exactly(tool, status, expected):
    env, o = _env_with_status(status)
    out = env.call(tool, _args(tool, o))
    if expected:
        assert out == expected
    else:
        assert "cannot be" not in out


def test_later_checks_in_upstream_order():
    env = fresh_env()
    pend = _order(lambda o: o["status"] == "pending" and len(o["payment_history"]) == 1)
    a = _args("modify_pending_order_items", pend)
    assert env.call("modify_pending_order_items", {**a, "item_ids": ["123"]}) == "Error: 123 not found"
    assert env.call("modify_pending_order_items", {**a, "new_item_ids": []}) == "Error: The number of items to be exchanged should match"
    assert env.call("modify_pending_order_items", {**a, "new_item_ids": a["item_ids"]}) == "Error: The new item id should be different from the old item id"
    assert env.call("modify_pending_order_items", {**a, "new_item_ids": ["999"]}) == "Error: Variant not found"
    assert env.call("modify_pending_order_items", {**a, "payment_method_id": "credit_card_0"}) == "Error: Payment method not found"
    assert env.call("modify_pending_order_payment", {"order_id": pend["order_id"], "payment_method_id": "x"}) == "Error: Payment method not found"
    assert env.call("modify_pending_order_payment", {"order_id": pend["order_id"],
                                                     "payment_method_id": pend["payment_history"][0]["payment_method_id"]}) \
        == "Error: The new payment method should be different from the current one"
    assert env.call("cancel_pending_order", {"order_id": pend["order_id"], "reason": "other"}) == "Error: Invalid reason"
    dlv = _order(lambda o: o["status"] == "delivered")
    b = _args("exchange_delivered_order_items", dlv)
    assert env.call("exchange_delivered_order_items", {**b, "item_ids": ["123"]}) == "Error: Number of 123 not found."
    assert env.call("exchange_delivered_order_items", {**b, "new_item_ids": []}) == "Error: The number of items to be exchanged should match."
    r = _args("return_delivered_order_items", dlv)
    assert env.call("return_delivered_order_items", {**r, "item_ids": ["123"]}) == "Error: Some item not found"


def test_unavailable_wrong_return_pm_and_gift_card_errors():
    db = base_db()
    env = fresh_env()
    o = _order(lambda o: o["status"] == "delivered" and any(
        not v["available"] for it in o["items"] for k, v in db["products"][it["product_id"]]["variants"].items() if k != it["item_id"]))
    it = next(it for it in o["items"] if any(not v["available"] for k, v in db["products"][it["product_id"]]["variants"].items() if k != it["item_id"]))
    new = next(k for k, v in db["products"][it["product_id"]]["variants"].items() if k != it["item_id"] and not v["available"])
    a = {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": [new], "payment_method_id": o["payment_history"][0]["payment_method_id"]}
    assert env.call("exchange_delivered_order_items", a) == f"Error: New item {new} not found or available"
    o = _order(lambda o: o["status"] == "delivered" and any(
        m["source"] != "gift_card" and p != o["payment_history"][0]["payment_method_id"] for p, m in _user(o)["payment_methods"].items()))
    pm = next(p for p, m in _user(o)["payment_methods"].items() if m["source"] != "gift_card" and p != o["payment_history"][0]["payment_method_id"])
    assert env.call("return_delivered_order_items", {"order_id": o["order_id"], "item_ids": [o["items"][0]["item_id"]], "payment_method_id": pm}) \
        == "Error: Payment method should be the original payment method"
    o = _order(lambda o: "pending" in o["status"] and len(o["payment_history"]) == 1 and any(
        m["source"] == "gift_card" and p != o["payment_history"][0]["payment_method_id"] and m["balance"] < o["payment_history"][0]["amount"]
        for p, m in _user(o)["payment_methods"].items()))
    gc = next(p for p, m in _user(o)["payment_methods"].items() if m["source"] == "gift_card"
              and p != o["payment_history"][0]["payment_method_id"] and m["balance"] < o["payment_history"][0]["amount"])
    assert env.call("modify_pending_order_payment", {"order_id": o["order_id"], "payment_method_id": gc}) \
        == "Error: Insufficient gift card balance to pay for the order"


def test_failed_writes_leave_the_database_unchanged():
    for tool, status, expected in STATUS_CASES:
        if expected:
            env, o = _env_with_status(status)
            before = env.state.sha256()
            assert env.call(tool, _args(tool, o)) == expected
            assert env.state.sha256() == before


def test_failed_returns_past_the_status_check_leave_the_database_unchanged():
    # A return refused after the status check (another payment method, an unknown one, an item not in the order) must
    # write nothing. Added 2026-11-24: the mutant "failed return writes first" was otherwise caught only by the item
    # builder's fixture erroring in setup, which the strict mutation rule does not count as a kill.
    o = _order(lambda o: o["status"] == "delivered" and any(
        m["source"] != "gift_card" and p != o["payment_history"][0]["payment_method_id"] for p, m in _user(o)["payment_methods"].items()))
    pm = next(p for p, m in _user(o)["payment_methods"].items() if m["source"] != "gift_card" and p != o["payment_history"][0]["payment_method_id"])
    ok = {"order_id": o["order_id"], "item_ids": [o["items"][0]["item_id"]], "payment_method_id": o["payment_history"][0]["payment_method_id"]}
    for change in ({"payment_method_id": pm}, {"payment_method_id": "no_such_payment_method"}, {"item_ids": ["123"]}):
        env = fresh_env()
        before = env.state.sha256()
        assert env.call("return_delivered_order_items", {**ok, **change}).startswith("Error: ")
        assert env.state.sha256() == before, change
    env = fresh_env()  # the unchanged arguments succeed, so the refusals above are the checks under test
    before = env.state.sha256()
    assert not env.call("return_delivered_order_items", ok).startswith("Error: ")
    assert env.state.sha256() != before


def test_forks_never_change_their_parent_or_the_pristine_database():
    before = pristine_sha256()
    parent = fresh_env()
    o = _order(lambda o: o["status"] == "pending")
    child = parent.fork()
    out = child.call("cancel_pending_order", {"order_id": o["order_id"], "reason": "no longer needed"})
    assert json.loads(out)["status"] == "cancelled"
    assert parent.state.orders[o["order_id"]]["status"] == "pending"
    assert json.loads(parent.call("get_order_details", {"order_id": o["order_id"]}))["status"] == "pending"
    assert json.loads(child.call("get_order_details", {"order_id": o["order_id"]}))["status"] == "cancelled"
    # a gift-card refund changes the user's balance only in the child
    o = _order(lambda o: o["status"] == "pending" and o["payment_history"][0]["payment_method_id"].startswith("gift_card"))
    gc = o["payment_history"][0]["payment_method_id"]
    c2 = parent.fork()
    c2.call("cancel_pending_order", {"order_id": o["order_id"], "reason": "no longer needed"})
    assert c2.state.users[o["user_id"]]["payment_methods"][gc]["balance"] != parent.state.users[o["user_id"]]["payment_methods"][gc]["balance"]
    assert pristine_sha256() == before
    with pytest.raises(ValueError):
        RetailEnv(retail._pristine())


def test_bypass_skips_only_the_named_check():
    o = _order(lambda o: o["status"] == "processed")
    a = _args("return_delivered_order_items", o)
    env = fresh_env()
    assert env.call_bypassed("return_delivered_order_items", a, frozenset({"gc_balance"})) == "Error: Non-delivered order cannot be returned"
    out = env.call_bypassed("return_delivered_order_items", a, frozenset({"status"}))
    assert json.loads(out)["status"] == "return requested"
    # the bypassed call ran on this env's own state, never the pristine one
    assert base_db()["orders"][o["order_id"]]["status"] == "processed"
    with pytest.raises(ValueError):
        RetailEnv(State(retail.load_db_from_disk()).fork(), frozenset({"everything"}))


# ---------------- upstream golden outputs of every write tool (scripts/tau_upstream_golden.py) ----------------

GOLDEN = json.loads((Path(__file__).parent / "data" / "tau_upstream_golden.json").read_text(encoding="utf-8"))


def patched_env(patches):
    """A fresh port environment with the case's gift-card balances set, as the golden script set them upstream."""
    env = fresh_env()
    for p in patches:
        env.state.own_user(p["user_id"])
        env.state.users[p["user_id"]]["payment_methods"][p["pm"]]["balance"] = float(p["balance"])
    return env


def test_golden_file_is_from_the_pinned_commit_and_covers_every_write_tool():
    assert GOLDEN["pinned_commit"] == retail.PINNED_COMMIT
    tools = {c["name"] for case in GOLDEN["cases"] for c in case["calls"]}
    assert set(retail.WRITE_TOOLS) | {"calculate"} <= tools
    assert any(c["output"].startswith("Error") for case in GOLDEN["cases"] for c in case["calls"])


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["name"] for c in GOLDEN["cases"]])
def test_writes_match_upstream_byte_for_byte(case):
    env = patched_env(case["patches"])
    for c in case["calls"]:
        assert env.call(c["name"], c["arguments"]) == c["output"], (case["name"], c["name"])
