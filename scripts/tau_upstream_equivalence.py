#!/usr/bin/env python3
"""Check wf/tau/retail.py byte for byte against upstream tau2-bench at the pinned commit (design §2).

Needs a Python where `tau2` (tau2-bench at commit c06b07d) is importable, with TAU2_DATA_DIR pointing at its
data. It never imports anything from wf except wf.tau.retail (standard library only).

    TAU2_DATA_DIR=<tau2-bench>/data python scripts/tau_upstream_equivalence.py [items.jsonl ...] --out <json>

Checks:
1. Seeded random episodes: the same sequence of calls (every tool, valid and invalid arguments, writes
   included) gives identical outputs, and the full database is identical afterwards.
2. For every item file given: each item's history outputs and failure reply are identical under upstream,
   and upstream's twin, made by patching the database copy (design §3.4, second route), equals the item's
   twin, which the port made by bypassing the check. modify_pending_order_address and _payment keep the
   order's status on success, so for them the patched status is put back to the true one before comparing.
3. Each item's success witness (the recorded aligned witness), run on the history's state without any patch,
   gives exactly the item's witness output.
Exit status 0 only if every comparison is identical.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wf.tau import retail  # noqa: E402


def _upstream():
    from loguru import logger

    logger.remove()
    from tau2.data_model.message import ToolCall
    from tau2.domains.retail.data_model import GiftCard, RetailDB
    from tau2.domains.retail.environment import get_environment
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    base = RetailDB.load(RETAIL_DB_PATH)

    class Up:
        def __init__(self):
            self.env = get_environment(db=base)

        def call(self, name, args):
            return self.env.get_response(ToolCall(id="x", name=name, arguments=args, requestor="assistant")).content

        def db_json(self):
            return json.dumps(self.env.tools.db.model_dump())

    return Up, GiftCard, RETAIL_DB_PATH


def _random_call(rng: random.Random, db: dict) -> tuple[str, dict]:
    users = sorted(db["users"])
    u = db["users"][rng.choice(users)]
    oid = rng.choice(u["orders"]) if u["orders"] and rng.random() < 0.95 else "#W0000000"
    o = db["orders"].get(oid)
    pms = list(u["payment_methods"]) + ["credit_card_0000000"]
    pm = rng.choice(pms)
    items = [i["item_id"] for i in o["items"]] if o else ["1"]
    item = rng.choice(items)
    pid = o["items"][items.index(item)]["product_id"] if o else rng.choice(sorted(db["products"]))
    variants = list(db["products"][pid]["variants"]) + ["0000000000"]
    new = rng.choice(variants)
    addr = u["address"]
    kind = rng.choice(retail.ALL_TOOLS)
    args = {
        "calculate": {"expression": rng.choice(["2 + 2", "(10 - 3) * 1.5", "1/3", "abc"])},
        "find_user_id_by_name_zip": {"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"],
                                     "zip": rng.choice([addr["zip"], "00000"])},
        "find_user_id_by_email": {"email": rng.choice([u["email"], u["email"].upper(), "nobody@example.com"])},
        "get_order_details": {"order_id": oid},
        "get_product_details": {"product_id": rng.choice([pid, "0"])},
        "get_item_details": {"item_id": rng.choice([new, item, "0"])},
        "get_user_details": {"user_id": rng.choice([u["user_id"], "nobody_1"])},
        "list_all_product_types": {},
        "transfer_to_human_agents": {"summary": "x"},
        "cancel_pending_order": {"order_id": oid, "reason": rng.choice(["no longer needed", "ordered by mistake", "other"])},
        "exchange_delivered_order_items": {"order_id": oid, "item_ids": rng.choice([[item], [item, item], []]),
                                           "new_item_ids": rng.choice([[new], [new, new], []]), "payment_method_id": pm},
        "modify_pending_order_address": {"order_id": oid, **{k: addr[k] for k in ("address1", "address2", "city", "state", "country", "zip")}},
        "modify_pending_order_items": {"order_id": oid, "item_ids": rng.choice([[item], [item, item]]),
                                       "new_item_ids": rng.choice([[new], [item], [new, new]]), "payment_method_id": pm},
        "modify_pending_order_payment": {"order_id": oid, "payment_method_id": pm},
        "modify_user_address": {"user_id": u["user_id"], **{k: addr[k] + rng.choice(["", " x"]) for k in ("address1", "address2", "city", "state", "country", "zip")}},
        "return_delivered_order_items": {"order_id": oid, "item_ids": rng.choice([[item], [item, item], ["0"]]),
                                         "payment_method_id": pm},
    }[kind]
    return kind, args


def fuzz(Up, episodes: int, calls: int, seed: int) -> dict:
    rng = random.Random(seed)
    base = retail.base_db()
    before = retail.pristine_sha256()
    out = {"episodes": episodes, "calls": 0, "output_mismatches": 0, "db_mismatches": 0, "errors": 0, "examples": []}
    for _ in range(episodes):
        up, port = Up(), retail.fresh_env()
        for _ in range(calls):
            name, args = _random_call(rng, base)
            a, b = up.call(name, args), port.call(name, args)
            out["calls"] += 1
            out["errors"] += a.startswith("Error")
            if a != b:
                out["output_mismatches"] += 1
                if len(out["examples"]) < 5:
                    out["examples"].append({"call": [name, args], "upstream": a[:300], "port": b[:300]})
        if up.db_json() != json.dumps(port.state.as_dict()):
            out["db_mismatches"] += 1
    out["pristine_unchanged"] = retail.pristine_sha256() == before
    if not out["pristine_unchanged"]:
        out["db_mismatches"] += 1
    return out


# Tools that do not overwrite the order's status on success (notes/tau_design.md, changes of 2026-11-20).
KEEPS_STATUS = ("modify_pending_order_address", "modify_pending_order_payment")


def _patch(env, GiftCard, item: dict):
    """Upstream twin, second route: patch the database copy so the violated check passes."""
    db = env.env.tools.db
    act = item["action"]
    cls, a = item["cls"], act["arguments"]
    if cls == "status":
        need = "delivered" if act["name"] in ("exchange_delivered_order_items", "return_delivered_order_items") else "pending"
        true_status = db.orders[a["order_id"]].status
        db.orders[a["order_id"]].status = need
        return true_status
    elif cls == "unavailable":
        order = db.orders[a["order_id"]]
        for old, new in zip(a["item_ids"], a["new_item_ids"]):
            pid = next(i.product_id for i in order.items if i.item_id == old)
            db.products[pid].variants[new].available = True
    elif cls == "wrong_return_pm":
        uid = db.orders[a["order_id"]].user_id
        db.users[uid].payment_methods[a["payment_method_id"]] = GiftCard(source="gift_card", id=a["payment_method_id"], balance=0.0)
    elif cls == "gc_balance":
        uid = db.orders[a["order_id"]].user_id
        db.users[uid].payment_methods[a["payment_method_id"]].balance = 10 ** 6
    else:
        raise ValueError(cls)
    return None


def check_items(Up, GiftCard, path: Path) -> dict:
    out = {"file": str(path), "items": 0, "history_mismatches": 0, "failure_mismatches": 0, "twin_mismatches": 0,
           "witnesses": 0, "witness_mismatches": 0, "status_restored": 0, "examples": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        out["items"] += 1
        up = Up()
        got = [up.call(h["name"], h["arguments"]) for h in it["history"]]
        if got != [h["output"] for h in it["history"]]:
            out["history_mismatches"] += 1
        if up.call(it["action"]["name"], it["action"]["arguments"]) != it["failure_reply"]:
            out["failure_mismatches"] += 1
        twin_env = Up()
        for h in it["history"]:
            twin_env.call(h["name"], h["arguments"])
        true_status = _patch(twin_env, GiftCard, it)
        twin = twin_env.call(it["action"]["name"], it["action"]["arguments"])
        if true_status and it["action"]["name"] in KEEPS_STATUS and not twin.startswith("Error"):
            # these two tools leave the status alone on success, so the patched status would show; restore it
            rec = json.loads(twin)
            rec["status"] = true_status
            twin = json.dumps(rec)
            out["status_restored"] += 1
        if it.get("aligned_witness"):
            # the success control: the recorded witness on the same state, unpatched
            w_env = Up()
            for h in it["history"]:
                w_env.call(h["name"], h["arguments"])
            out["witnesses"] += 1
            w = it["aligned_witness"]
            if w_env.call(w["name"], w["arguments"]) != it["witness_output"]:
                out["witness_mismatches"] += 1
        if twin != it["success_twin"]:
            out["twin_mismatches"] += 1
            if len(out["examples"]) < 5:
                out["examples"].append({"id": it["id"], "upstream": twin[:300], "item": it["success_twin"][:300]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", type=Path)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--calls", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    Up, GiftCard, db_path = _upstream()
    import hashlib

    report = {"pinned_commit": retail.PINNED_COMMIT,
              "upstream_db_sha256": hashlib.sha256(Path(db_path).read_bytes()).hexdigest(),
              "port_db_sha256": retail.vendor_manifest()["db_json_sha256_uncompressed"],
              "fuzz": fuzz(Up, args.episodes, args.calls, args.seed),
              "items": [check_items(Up, GiftCard, p) for p in args.items]}
    f = report["fuzz"]
    ok = (report["upstream_db_sha256"] == report["port_db_sha256"] and f["output_mismatches"] == 0
          and f["db_mismatches"] == 0
          and all(r["history_mismatches"] == r["failure_mismatches"] == r["twin_mismatches"] == r["witness_mismatches"] == 0
                  for r in report["items"]))
    report["passed"] = ok
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
