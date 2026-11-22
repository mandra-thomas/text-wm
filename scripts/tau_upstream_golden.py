#!/usr/bin/env python3
"""Golden outputs of every retail write tool, printed by upstream tau2-bench at the pinned commit (design §2, §8.4).

Needs a Python where `tau2` at commit c06b07d is importable (as scripts/tau_upstream_equivalence.py does):

    TAU2_DATA_DIR=<tau2-bench>/data python scripts/tau_upstream_golden.py --out tests/data/tau_upstream_golden.json

Each case is a short call sequence on a fresh upstream environment, optionally after setting a gift card's balance
on the database copy, and ends with reads of the touched user and order, so the file pins both the outputs and the
state the writes leave. tests/test_tau_retail.py replays every case on the port and requires identical bytes. The
cases are chosen from the database by fixed rules (first match in sorted order) to exercise the arithmetic and
ordering details that random calls rarely reach: unrounded and rounded price differences, a gift card balance
exactly at the limit, sorted exchange and return lists, the last-variant quirk of multi-item modifications,
refund rounding, the payment check order on a modified order, the address field order and `calculate` rounding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wf.tau import retail  # noqa: E402

ADDR = ("address1", "address2", "city", "state", "country", "zip")


def _noisy(x: float) -> bool:
    return x != round(x, 2)


def cases(db: dict) -> list[dict]:
    orders = [db["orders"][k] for k in sorted(db["orders"])]
    users, products = db["users"], db["products"]
    pm0 = lambda o: o["payment_history"][0]["payment_method_id"]
    src = lambda o, pm: users[o["user_id"]]["payment_methods"][pm]["source"]
    alts = lambda it: sorted(v for v, x in products[it["product_id"]]["variants"].items() if v != it["item_id"] and x["available"])
    gcs = lambda o: sorted(p for p, m in users[o["user_id"]]["payment_methods"].items() if m["source"] == "gift_card")
    reads = lambda o: [("get_user_details", {"user_id": o["user_id"]}), ("get_order_details", {"order_id": o["order_id"]})]
    out = []

    def add(name, o, calls, patches=()):
        out.append({"name": name, "patches": list(patches), "calls": [list(c) for c in calls + reads(o)]})

    def one_pending(o):
        return o["status"] == "pending" and len(o["payment_history"]) == 1 and o["payment_history"][0]["transaction_type"] == "payment"

    # modify items, unrounded price difference paid by card (P1)
    o, it, v = next((o, it, v) for o in orders if one_pending(o) and src(o, pm0(o)) != "gift_card"
                    for it in o["items"] for v in alts(it) if _noisy(products[it["product_id"]]["variants"][v]["price"] - it["price"]))
    add("modify_items_unrounded_difference", o, [("modify_pending_order_items", {"order_id": o["order_id"], "item_ids": [it["item_id"]],
        "new_item_ids": [v], "payment_method_id": pm0(o)})])
    # modify items paid by a gift card with an unrounded difference: balance arithmetic and rounding
    o, it, v = next((o, it, v) for o in orders if one_pending(o) and gcs(o)
                    for it in o["items"] for v in alts(it) if _noisy(products[it["product_id"]]["variants"][v]["price"] - it["price"]))
    add("modify_items_gift_card", o, [("modify_pending_order_items", {"order_id": o["order_id"], "item_ids": [it["item_id"]],
        "new_item_ids": [v], "payment_method_id": gcs(o)[0]})], [{"user_id": o["user_id"], "pm": gcs(o)[0], "balance": 5000.0}])
    # modify items, gift card balance exactly the unrounded difference rounded down: fails (unrounded comparison)
    o, it, v, d = next((o, it, v, products[it["product_id"]]["variants"][v]["price"] - it["price"]) for o in orders if one_pending(o) and gcs(o)
                       for it in o["items"] for v in alts(it)
                       if products[it["product_id"]]["variants"][v]["price"] - it["price"] > round(products[it["product_id"]]["variants"][v]["price"] - it["price"], 2) > 0)
    add("modify_items_gift_card_at_rounded_difference", o, [("modify_pending_order_items", {"order_id": o["order_id"],
        "item_ids": [it["item_id"]], "new_item_ids": [v], "payment_method_id": gcs(o)[0]})],
        [{"user_id": o["user_id"], "pm": gcs(o)[0], "balance": round(d, 2)}])
    # multi-item modification, different products: every item takes the last pair's variant (P6)
    o, pairs = next((o, [(a["item_id"], alts(a)[0]), (b["item_id"], alts(b)[0])]) for o in orders if one_pending(o) and src(o, pm0(o)) != "gift_card"
                    for a in o["items"] for b in o["items"] if a["product_id"] < b["product_id"] and alts(a) and alts(b))
    add("modify_items_two_products", o, [("modify_pending_order_items", {"order_id": o["order_id"], "item_ids": [p[0] for p in pairs],
        "new_item_ids": [p[1] for p in pairs], "payment_method_id": pm0(o)})])
    # payment change on a modified order: two payments, so the exactly-one check fires before the same-method check (P3)
    o, it, v = next((o, it, alts(it)[0]) for o in orders if one_pending(o) and src(o, pm0(o)) != "gift_card" for it in o["items"] if alts(it))
    add("payment_after_modification", o, [("modify_pending_order_items", {"order_id": o["order_id"], "item_ids": [it["item_id"]],
        "new_item_ids": [v], "payment_method_id": pm0(o)}), ("modify_pending_order_payment", {"order_id": o["order_id"], "payment_method_id": pm0(o)}),
        ("modify_pending_order_address", {"order_id": o["order_id"], **{k: users[o["user_id"]]["address"][k] for k in ADDR}})])
    # payment change from a gift card to another method: the old card is refunded, with rounding
    o = next(o for o in orders if one_pending(o) and src(o, pm0(o)) == "gift_card"
             and any(m["source"] != "gift_card" for m in users[o["user_id"]]["payment_methods"].values()))
    new = next(p for p, m in sorted(users[o["user_id"]]["payment_methods"].items()) if m["source"] != "gift_card")
    add("payment_gift_card_to_card", o, [("modify_pending_order_payment", {"order_id": o["order_id"], "payment_method_id": new})])
    # payment change from a card to a gift card whose balance exactly covers the payment: the card is charged
    o = next(o for o in orders if one_pending(o) and src(o, pm0(o)) != "gift_card" and gcs(o))
    add("payment_card_to_gift_card_exact", o, [("modify_pending_order_payment", {"order_id": o["order_id"], "payment_method_id": gcs(o)[0]})],
        [{"user_id": o["user_id"], "pm": gcs(o)[0], "balance": o["payment_history"][0]["amount"] + 0.004}])
    # payment change from a card to a gift card whose balance is a cent short
    o = next(o for o in orders if one_pending(o) and src(o, pm0(o)) != "gift_card" and gcs(o))
    add("payment_gift_card_cent_short", o, [("modify_pending_order_payment", {"order_id": o["order_id"], "payment_method_id": gcs(o)[0]})],
        [{"user_id": o["user_id"], "pm": gcs(o)[0], "balance": round(o["payment_history"][0]["amount"] - 0.01, 2)}])
    # exchange paid by a gift card whose balance equals the rounded difference (P2); unrounded difference stored
    o, it, v, d = next((o, it, v, products[it["product_id"]]["variants"][v]["price"] - it["price"]) for o in orders
                       if o["status"] == "delivered" and gcs(o) for it in o["items"] for v in alts(it)
                       if products[it["product_id"]]["variants"][v]["price"] - it["price"] > round(products[it["product_id"]]["variants"][v]["price"] - it["price"], 2) > 0)
    add("exchange_gift_card_at_rounded_difference", o, [("exchange_delivered_order_items", {"order_id": o["order_id"], "item_ids": [it["item_id"]],
        "new_item_ids": [v], "payment_method_id": gcs(o)[0]})], [{"user_id": o["user_id"], "pm": gcs(o)[0], "balance": round(d, 2)}])
    # exchange of two items given in reverse order: the stored lists are sorted (P4)
    o, a, b = next((o, a, b) for o in orders if o["status"] == "delivered" for a in o["items"] for b in o["items"]
                   if a["item_id"] > b["item_id"] and alts(a) and alts(b) and alts(a)[0] > alts(b)[0] and src(o, pm0(o)) != "gift_card")
    add("exchange_two_items_unsorted", o, [("exchange_delivered_order_items", {"order_id": o["order_id"], "item_ids": [a["item_id"], b["item_id"]],
        "new_item_ids": [alts(a)[0], alts(b)[0]], "payment_method_id": pm0(o)})])
    # return of two items given in reverse order to a gift card that is not the original method
    o = next(o for o in orders if o["status"] == "delivered" and len(o["items"]) >= 2 and any(p != pm0(o) for p in gcs(o)))
    ids = sorted(i["item_id"] for i in o["items"])[:2][::-1]
    add("return_two_items_to_other_gift_card", o, [("return_delivered_order_items", {"order_id": o["order_id"], "item_ids": ids,
        "payment_method_id": next(p for p in gcs(o) if p != pm0(o))})])
    # cancel of a gift-card-paid order whose refund sum has float noise (P5)
    o = next(o for o in orders if o["status"] == "pending" and src(o, pm0(o)) == "gift_card"
             and _noisy(users[o["user_id"]]["payment_methods"][pm0(o)]["balance"] + o["payment_history"][0]["amount"]))
    add("cancel_gift_card_refund", o, [("cancel_pending_order", {"order_id": o["order_id"], "reason": "ordered by mistake"})])
    # address change: the stored address's field order (P7), on a pending order
    o = next(o for o in orders if o["status"] == "pending")
    add("modify_order_address", o, [("modify_pending_order_address", {"order_id": o["order_id"], "address1": "1 Main St",
        "address2": "Apt 2", "city": "Springfield", "state": "IL", "country": "USA", "zip": "62701"})])
    # user address change
    o = orders[0]
    add("modify_user_address", o, [("modify_user_address", {"user_id": o["user_id"], "address1": "1 Main St", "address2": "",
        "city": "Springfield", "state": "IL", "country": "USA", "zip": "62701"})])
    # calculate rounds to two places (P8)
    out.append({"name": "calculate", "patches": [], "calls": [["calculate", {"expression": e}] for e in ("1/3", "10 / 3", "2.675 * 3", "(1 + 2) * 0.1")]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from loguru import logger

    logger.remove()
    from tau2.data_model.message import ToolCall
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.environment import get_environment
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    base = RetailDB.load(RETAIL_DB_PATH)
    result = {"pinned_commit": retail.PINNED_COMMIT, "source": "upstream tau2-bench Environment.get_response content", "cases": []}
    for case in cases(retail.base_db()):
        env = get_environment(db=base)
        for p in case["patches"]:
            env.tools.db.users[p["user_id"]].payment_methods[p["pm"]].balance = p["balance"]
        calls = []
        for name, a in case["calls"]:
            out = env.get_response(ToolCall(id="x", name=name, arguments=a, requestor="assistant")).content
            calls.append({"name": name, "arguments": a, "output": out})
        result["cases"].append({"name": case["name"], "patches": case["patches"], "calls": calls})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({c["name"]: [x["output"][:60] for x in c["calls"][:-2]] for c in result["cases"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
