"""Item builder for the τ-bench retail replication (notes/tau_design.md §3–§5.3).

Generated items: a history of reads (user, target order, comparison order, products where needed), then a write
on the target order that fails one precondition visible in that history. Natural items: failed order writes
from the released tau2 retail trajectories, replayed and filtered. Every output in an item is produced by
running the call on the port (wf/tau/retail.py); nothing is hand-written.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from wf.tau.checker import TEST_CLASSES, check, error_class
from wf.tau.prompts import render_history
from wf.tau.retail import ORDER_WRITES, RetailEnv, base_db, fresh_env

SEED = 20260924  # wf.constants.RNG_SEED
TEMPLATES = {
    "cancel_pending_order": "The user wants to cancel order {o}.",
    "modify_pending_order_address": "The user wants to change the shipping address of order {o}.",
    "modify_pending_order_payment": "The user wants to change the payment method of order {o}.",
    "modify_pending_order_items": "The user wants to swap the {p} in order {o} for a different option of the same product.",
    "exchange_delivered_order_items": "The user wants to swap the {p} in order {o} for a different option of the same product.",
    "return_delivered_order_items": "The user wants to return the {p} from order {o}.",
}
KIND = {"cancel_pending_order": "cancel", "modify_pending_order_address": "address",
        "modify_pending_order_payment": "payment", "modify_pending_order_items": "swap",
        "exchange_delivered_order_items": "swap", "return_delivered_order_items": "return"}
NEUTRAL = "The user wants help with their account."
# Mention-only pair (amendment of 2026-11-20 after review 1): names an order without asking for any write.
MENTION = "The user has a question about order {o}."
REASONS = ("no longer needed", "ordered by mistake")
ADDRESS_ARGS = ("address1", "address2", "city", "state", "country", "zip")
NATURAL_MAX_CALLS = 30
NATURAL_MAX_CHARS = 24_000


def h(*parts: Any) -> int:
    """The design's hash: first 16 hex digits of sha256("20260924|<parts joined by |>")."""
    return int(hashlib.sha256("|".join([str(SEED), *map(str, parts)]).encode()).hexdigest()[:16], 16)


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def item_key(history: list[dict], action: dict) -> str:
    """Identity of (visible history, action): two items with the same key are duplicates."""
    return hashlib.sha256(canonical([history, action]).encode()).hexdigest()


def task_line(tool: str, order_id: str, product: str | None) -> str:
    template = TEMPLATES[tool]
    if "{p}" in template and not product:
        raise ValueError(f"{tool} needs a product")
    return template.format(o=order_id, p=product)


def claim_problems(db: dict, user_id: str, tool: str, order_id: str, product: str | None) -> list[str]:
    """Facts the task line asserts that the database contradicts (design §3.5)."""
    problems = []
    if order_id not in db["users"][user_id]["orders"] or order_id not in db["orders"]:
        problems.append("order is not the user's")
        return problems
    if "{p}" in TEMPLATES[tool]:
        items = [it for it in db["orders"][order_id]["items"] if it["name"] == product]
        if not items:
            problems.append("product not in order")
        elif KIND[tool] == "swap" and all(len(db["products"][it["product_id"]]["variants"]) < 2 for it in items):
            problems.append("product has one option")
    return problems


# ---------------- achievability ----------------


def witness_calls(db: dict, env: RetailEnv, kind: str, order_id: str, item_id: str | None) -> list[dict]:
    """Every single write call that would achieve the task, in a fixed order (design §3.6)."""
    order = env.state.orders[order_id]
    user = env.state.users[order["user_id"]]
    pms = list(user["payment_methods"])
    calls = []
    if kind == "swap":
        tool = {"pending": "modify_pending_order_items", "delivered": "exchange_delivered_order_items"}.get(order["status"])
        item = next((it for it in order["items"] if it["item_id"] == item_id), None)
        if tool and item:
            variants = db["products"][item["product_id"]]["variants"]
            for v in sorted(variants):
                if v != item_id and variants[v]["available"]:
                    calls += [{"name": tool, "arguments": {"order_id": order_id, "item_ids": [item_id], "new_item_ids": [v],
                                                           "payment_method_id": pm}} for pm in pms]
    elif kind == "return":
        calls = [{"name": "return_delivered_order_items",
                  "arguments": {"order_id": order_id, "item_ids": [item_id], "payment_method_id": pm}} for pm in pms]
    elif kind == "payment":
        calls = [{"name": "modify_pending_order_payment", "arguments": {"order_id": order_id, "payment_method_id": pm}}
                 for pm in pms]
    elif kind == "cancel":
        calls = [{"name": "cancel_pending_order", "arguments": {"order_id": order_id, "reason": r}} for r in REASONS]
    elif kind == "address":
        calls = [{"name": "modify_pending_order_address",
                  "arguments": {"order_id": order_id, **{k: user["address"][k] for k in ADDRESS_ARGS}}}]
    else:
        raise ValueError(kind)
    return calls


def achievable(db: dict, env: RetailEnv, kind: str, order_id: str, item_id: str | None) -> dict | None:
    """The first witness call that succeeds on a fork of `env`'s state, or None if none does."""
    for call in witness_calls(db, env, kind, order_id, item_id):
        if not env.fork().call(call["name"], call["arguments"]).startswith("Error"):
            return call
    return None


# ---------------- few-shot examples ----------------


def fewshot_examples(db: dict) -> list[dict]:
    """Four examples from four distinct users, the first in hash order meeting each need (design §5.3, as amended
    2026-11-20). Balanced 2x2: the task names the call's own order or another order of the same user, and the call
    succeeds or fails. No failure is of a test class. Prompt order: own-order success, other-order failure,
    other-order success, own-order failure."""
    users = sorted(db["users"], key=lambda u: h(u, "fewshot"))
    used: set[str] = set()

    def first(need):
        for uid in users:
            if uid in used:
                continue
            for oid in db["users"][uid]["orders"]:
                o = db["orders"].get(oid)
                others = [x for x in db["users"][uid]["orders"] if x != oid and x in db["orders"]]
                if o is not None and need(o, others):
                    used.add(uid)
                    return uid, oid, (sorted(others, key=lambda x: h(oid, x, "fewshot"))[0] if others else None)
        raise ValueError("no user for a few-shot example")

    def one_payment(o):
        return len(o["payment_history"]) == 1 and o["payment_history"][0]["transaction_type"] == "payment"

    specs = []
    # 1. own order, success: change the shipping address of a pending order
    uid, oid, _ = first(lambda o, others: o["status"] == "pending")
    addr = db["users"][uid]["address"]
    specs.append(("own", "success", uid, oid, "modify_pending_order_address",
                  {"order_id": oid, **{k: addr[k] for k in ADDRESS_ARGS}}, task_line("modify_pending_order_address", oid, None)))
    # 2. other order, failure: cancel a pending order with a reason the tool does not accept
    uid, oid, w2 = first(lambda o, others: o["status"] == "pending" and bool(others))
    specs.append(("other", "failure", uid, oid, "cancel_pending_order", {"order_id": oid, "reason": "found a better price"},
                  task_line("cancel_pending_order", w2, None)))
    # 3. other order, success: return the first item of a delivered order to its original payment method
    uid, oid, w2 = first(lambda o, others: o["status"] == "delivered" and bool(others))
    o = db["orders"][oid]
    specs.append(("other", "success", uid, oid, "return_delivered_order_items",
                  {"order_id": oid, "item_ids": [o["items"][0]["item_id"]],
                   "payment_method_id": o["payment_history"][0]["payment_method_id"]},
                  task_line("return_delivered_order_items", w2, db["orders"][w2]["items"][0]["name"])))
    # 4. own order, failure: change the payment method of a pending order to the method it already uses
    uid, oid, _ = first(lambda o, others: o["status"] == "pending" and one_payment(o))
    specs.append(("own", "failure", uid, oid, "modify_pending_order_payment",
                  {"order_id": oid, "payment_method_id": db["orders"][oid]["payment_history"][0]["payment_method_id"]},
                  task_line("modify_pending_order_payment", oid, None)))
    examples = []
    for names, outcome, uid, oid, tool, args, task in specs:
        env = fresh_env()
        calls = [{"name": "get_user_details", "arguments": {"user_id": uid}},
                 {"name": "get_order_details", "arguments": {"order_id": oid}}]
        history = [{**c, "output": o} for c, o in zip(calls, env.replay(calls))]
        examples.append({"user_id": uid, "order_id": oid, "task": task, "names": names, "outcome": outcome,
                         "history": history, "action": {"name": tool, "arguments": args}, "output": env.call(tool, args)})
    return examples


# ---------------- success controls (witnesses) ----------------

_STATUS_ERRORS = {"cancel_pending_order": "Non-pending order cannot be cancelled",
                  "modify_pending_order_address": "Non-pending order cannot be modified",
                  "modify_pending_order_payment": "Non-pending order cannot be modified",
                  "modify_pending_order_items": "Non-pending order cannot be modified",
                  "exchange_delivered_order_items": "Non-delivered order cannot be exchanged",
                  "return_delivered_order_items": "Non-delivered order cannot be returned"}
_GC_ERRORS = {"modify_pending_order_payment": "Insufficient gift card balance to pay for the order",
              "modify_pending_order_items": "Insufficient gift card balance to pay for the new item",
              "exchange_delivered_order_items": "Insufficient gift card balance to pay for the price difference"}


def witness_failure_text(cls: str, witness: dict) -> str:
    """The error the item's class would give for the witness call: the failure candidate of the witness's
    full-reply readout, mirroring the item's own (twin, failure) pair."""
    tool, args = witness["name"], witness["arguments"]
    if cls == "status":
        return "Error: " + _STATUS_ERRORS[tool]
    if cls == "unavailable":
        return f"Error: New item {args['new_item_ids'][0]} not found or available"
    if cls == "wrong_return_pm":
        return "Error: Payment method should be the original payment method"
    if cls == "gc_balance":
        return "Error: " + _GC_ERRORS[tool]
    raise ValueError(cls)


def witness_fields(history: list[dict], witness: dict | None, cls: str) -> dict:
    """The witness's real output on the history's state, its class-matched failure text, and what the text-only
    checker can say about it from the history (PASS when the success is determined by what is shown)."""
    if witness is None:
        return {"witness_output": None, "witness_failure": None, "witness_checker": None}
    env = fresh_env()
    env.replay(history)
    out = env.call(witness["name"], witness["arguments"])
    if out.startswith("Error"):
        raise RuntimeError(f"recorded witness fails: {witness}")
    return {"witness_output": out, "witness_failure": witness_failure_text(cls, witness),
            "witness_checker": check(history, witness).verdict}


def target_product_id(history: list[dict], order_id: str, item_id: str | None) -> str | None:
    """The product of the target item, read from the latest shown record of the target order."""
    if item_id is None:
        return None
    pid = None
    for x in history:
        try:
            rec = json.loads(x["output"])
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("order_id") == order_id and "items" in rec:
            pid = next((it["product_id"] for it in rec["items"] if it["item_id"] == item_id), pid)
    return pid


# ---------------- generated items ----------------


def _available_alts(db: dict, item: dict) -> list[str]:
    variants = db["products"][item["product_id"]]["variants"]
    return sorted(v for v in variants if v != item["item_id"] and variants[v]["available"])


def _unavailable_alts(db: dict, item: dict) -> list[str]:
    variants = db["products"][item["product_id"]]["variants"]
    return sorted(v for v in variants if v != item["item_id"] and not variants[v]["available"])


def _by_hash(xs: list, *parts: Any) -> Any:
    return min(xs, key=lambda x: h(*parts, canonical(x))) if xs else None


def candidate_actions(db: dict, uid: str, oid: str) -> list[tuple[str, str, dict, dict | None]]:
    """(class, tool, arguments, target item) for every failing action generated on this order (design §3.3)."""
    o, u = db["orders"][oid], db["users"][uid]
    st, pm0 = o["status"], o["payment_history"][0]["payment_method_id"]
    any_item = _by_hash(o["items"], oid, "item")
    with_alt = [it for it in o["items"] if _available_alts(db, it)]
    alt_item = _by_hash(with_alt, oid, "alt-item")
    alt = _by_hash(_available_alts(db, alt_item), oid, "alt") if alt_item else None
    out: list[tuple[str, str, dict, dict | None]] = []
    if st != "pending":
        out.append(("status", "cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}, None))
        if alt_item:
            out.append(("status", "modify_pending_order_items", {"order_id": oid, "item_ids": [alt_item["item_id"]],
                                                                 "new_item_ids": [alt], "payment_method_id": pm0}, alt_item))
    if "pending" not in st:
        out.append(("status", "modify_pending_order_address",
                    {"order_id": oid, **{k: u["address"][k] for k in ADDRESS_ARGS}}, None))
        other = [p for p in u["payment_methods"] if p != pm0]
        if other:
            out.append(("status", "modify_pending_order_payment", {"order_id": oid, "payment_method_id": other[0]}, None))
    if st != "delivered":
        out.append(("status", "return_delivered_order_items",
                    {"order_id": oid, "item_ids": [any_item["item_id"]], "payment_method_id": pm0}, any_item))
        if alt_item:
            out.append(("status", "exchange_delivered_order_items", {"order_id": oid, "item_ids": [alt_item["item_id"]],
                                                                     "new_item_ids": [alt], "payment_method_id": pm0}, alt_item))
    if st in ("pending", "delivered"):
        with_unavail = [it for it in o["items"] if _unavailable_alts(db, it)]
        it = _by_hash(with_unavail, oid, "unavail-item")
        if it:
            tool = "modify_pending_order_items" if st == "pending" else "exchange_delivered_order_items"
            new = _by_hash(_unavailable_alts(db, it), oid, "unavail")
            out.append(("unavailable", tool, {"order_id": oid, "item_ids": [it["item_id"]], "new_item_ids": [new],
                                              "payment_method_id": pm0}, it))
    if st == "delivered":
        bad = [p for p, m in u["payment_methods"].items() if m["source"] != "gift_card" and p != pm0]
        pm = _by_hash(bad, oid, "wrong-pm")
        if pm:
            out.append(("wrong_return_pm", "return_delivered_order_items",
                        {"order_id": oid, "item_ids": [any_item["item_id"]], "payment_method_id": pm}, any_item))
    ph = o["payment_history"]
    if "pending" in st and len(ph) == 1 and ph[0]["transaction_type"] == "payment":
        low = [p for p, m in u["payment_methods"].items()
               if m["source"] == "gift_card" and p != pm0 and m["balance"] < ph[0]["amount"]]
        pm = _by_hash(low, oid, "gc")
        if pm:
            out.append(("gc_balance", "modify_pending_order_payment", {"order_id": oid, "payment_method_id": pm}, None))
    return out


def comparison(db: dict, uid: str, oid: str, tool: str, target_name: str | None) -> tuple[str, dict | None] | None:
    """(W2, comparison item): another order of the user, status-matched first, then by hash (design §3.5)."""
    st = db["orders"][oid]["status"]
    others = [x for x in db["users"][uid]["orders"] if x != oid and x in db["orders"]]
    for w2 in sorted(others, key=lambda x: (db["orders"][x]["status"] != st, h(oid, x))):
        if "{p}" not in TEMPLATES[tool]:
            return w2, None
        its = [it for it in db["orders"][w2]["items"]
               if KIND[tool] != "swap" or len(db["products"][it["product_id"]]["variants"]) >= 2]
        if its:
            return w2, min(its, key=lambda it: (it["name"] == target_name, h(w2, it["item_id"])))
    return None


def history_calls(uid: str, oid: str, w2: str, products: list[str]) -> tuple[list[dict], bool]:
    target_first = h(oid, w2, "order") % 2 == 0
    pair = [oid, w2] if target_first else [w2, oid]
    calls = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    calls += [{"name": "get_order_details", "arguments": {"order_id": x}} for x in pair]
    prods = products if target_first else list(reversed(products))
    seen: list[str] = []
    for p in prods:
        if p not in seen:
            seen.append(p)
    calls += [{"name": "get_product_details", "arguments": {"product_id": p}} for p in seen]
    return calls, target_first


def make_twin(history: list[dict], action: dict, cls: str) -> str:
    """The action's output on the history's state with only the violated check bypassed. The history is
    replayed without any bypass, so the state is the one the failing call saw."""
    env = fresh_env()
    env.replay(history)
    return env.call_bypassed(action["name"], action["arguments"], frozenset({cls}))


def _touched_sha(env: RetailEnv, oid: str) -> str:
    o = env.state.orders.get(oid)
    u = env.state.users.get(o["user_id"]) if o else None
    return hashlib.sha256(json.dumps([o, u]).encode()).hexdigest()


def build_generated(db: dict, exclude_users: set[str], max_users: int | None = None) -> tuple[list[dict], Counter]:
    """`max_users` (tests only) stops after that many users in sorted order."""
    log: Counter = Counter()
    items: list[dict] = []
    seen: set[str] = set()
    for n_user, uid in enumerate(sorted(db["users"])):
        if max_users is not None and n_user >= max_users:
            break
        if uid in exclude_users:
            continue
        orders = [x for x in db["users"][uid]["orders"] if x in db["orders"]]
        if len(orders) < 2:
            log["user with fewer than two orders"] += 1
            continue
        for oid in orders:
            for cls, tool, args, item in candidate_actions(db, uid, oid):
                log[f"candidate {cls}"] += 1
                product = item["name"] if item else None
                comp = comparison(db, uid, oid, tool, product)
                if comp is None:
                    log[f"rejected {cls}: no comparison order"] += 1
                    continue
                w2, item2 = comp
                products = [item["product_id"], item2["product_id"]] if cls == "unavailable" else []
                calls, target_first = history_calls(uid, oid, w2, products)
                env = fresh_env()
                history = [{**c, "output": out} for c, out in zip(calls, env.replay(calls))]
                action = {"name": tool, "arguments": args}
                before = _touched_sha(env, oid)
                fail = env.call(tool, args)
                if not fail.startswith("Error: ") or error_class(fail) != cls:
                    log[f"rejected {cls}: action gave {error_class(fail) if fail.startswith('Error') else 'success'}"] += 1
                    continue
                if _touched_sha(env, oid) != before:
                    raise RuntimeError(f"failed write changed the database: {tool} {args}")
                twin = make_twin(history, action, cls)
                if twin.startswith("Error"):
                    log[f"rejected {cls}: second check fails ({error_class(twin)})"] += 1
                    continue
                aligned = task_line(tool, oid, product)
                swapped = task_line(tool, w2, item2["name"] if item2 else None)
                if claim_problems(db, uid, tool, oid, product) or claim_problems(db, uid, tool, w2, item2["name"] if item2 else None):
                    log[f"rejected {cls}: task line asserts a false fact"] += 1
                    continue
                key = item_key(history, action)
                if key in seen:
                    log[f"rejected {cls}: duplicate history and action"] += 1
                    continue
                seen.add(key)
                kind = KIND[tool]
                w_al = achievable(db, env, kind, oid, item["item_id"] if item else None)
                w_sw = achievable(db, env, kind, w2, item2["item_id"] if item2 else None)
                items.append({
                    "id": f"tau-g-{key[:12]}", "source": "generated", "cls": cls, "user_id": uid,
                    "order_id": oid, "comparison_order_id": w2, "item_id": item["item_id"] if item else None,
                    "comparison_item_id": item2["item_id"] if item2 else None,
                    "product": product, "comparison_product": item2["name"] if item2 else None,
                    "target_status": db["orders"][oid]["status"], "comparison_status": db["orders"][w2]["status"],
                    "status_matched": db["orders"][oid]["status"] == db["orders"][w2]["status"],
                    "target_first": target_first, "history": history, "action": action,
                    "failure_reply": fail, "success_twin": twin,
                    "task_aligned": aligned, "task_swapped": swapped, "task_neutral": NEUTRAL,
                    "aligned_achievable": w_al is not None, "aligned_witness": w_al,
                    "swapped_achievable": w_sw is not None, "swapped_witness": w_sw,
                    "main_pool": w_al is not None, "key": key,
                    "task_mention_aligned": MENTION.format(o=oid), "task_mention_swapped": MENTION.format(o=w2),
                    "target_product_id": item["product_id"] if item else None,
                    **witness_fields(history, w_al, cls),
                })
                log[f"kept {cls}"] += 1
    return items, log


# ---------------- natural items ----------------


def load_trajectories(paths: list[Path]) -> list[dict]:
    """Each simulation as {file, simulation_id, task_id, trial, agent, events: [{name, arguments, output}]}."""
    out = []
    for path in paths:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = d.get("info", {}).get("agent_info", {}).get("llm", Path(path).name.split("_")[0])
        for s in d["simulations"]:
            pending, events = {}, []
            for m in s["messages"]:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        pending[tc["id"]] = (tc["name"], tc.get("arguments"))
                elif m.get("role") == "tool" and m.get("id") in pending:
                    name, args = pending.pop(m["id"])
                    events.append({"name": name, "arguments": args, "output": m.get("content") or ""})
            out.append({"file": Path(path).name, "simulation_id": s["id"], "task_id": s["task_id"],
                        "trial": s.get("trial"), "agent": agent, "events": events})
    return out


def build_natural(db: dict, trajectories: list[dict], exclude_users: set[str]) -> tuple[list[dict], Counter]:
    log: Counter = Counter()
    found: list[dict] = []
    for tr in trajectories:
        events = tr["events"]
        replay_env = fresh_env()
        replay_ok: list[bool] = []
        for e in events:
            ok = isinstance(e["arguments"], dict) and replay_env.call(e["name"], e["arguments"]) == e["output"]
            replay_ok.append(ok)
        for k, e in enumerate(events):
            if e["name"] not in ORDER_WRITES or not e["output"].startswith("Error"):
                continue
            log["failed order writes"] += 1
            cls = error_class(e["output"])
            oid = (e["arguments"] or {}).get("order_id") if isinstance(e["arguments"], dict) else None
            history = [dict(x) for x in events[:k]]
            if not all(replay_ok[: k + 1]):
                log["dropped: replay mismatch"] += 1
                continue
            if cls not in TEST_CLASSES:
                log[f"dropped: class {cls}"] += 1
                continue
            v = check(history, {"name": e["name"], "arguments": e["arguments"]})
            if v.verdict != "FAIL" or v.cls != cls or v.error != e["output"]:
                log["dropped: not determined by the history"] += 1
                continue
            if any(x["name"] in ORDER_WRITES and x["output"].startswith("Error") and (x["arguments"] or {}).get("order_id") == oid
                   for x in history):
                log["dropped: echo of an earlier failure on the same order"] += 1
                continue
            if len(history) > NATURAL_MAX_CALLS or len(render_history(history)) > NATURAL_MAX_CHARS:
                log["dropped: history too long"] += 1
                continue
            uid = db["orders"][oid]["user_id"]
            if uid in exclude_users:
                log["dropped: few-shot user"] += 1
                continue
            found.append({"tr": tr, "k": k, "cls": cls, "uid": uid, "oid": oid, "history": history,
                          "action": {"name": e["name"], "arguments": e["arguments"]}, "failure_reply": e["output"]})
    # exact duplicates, then one item per (order, tool, error text): the shortest history, ties by hash
    by_key: dict[str, dict] = {}
    for f in found:
        key = item_key(f["history"], f["action"])
        if key in by_key:
            log["dropped: duplicate history and action"] += 1
            continue
        f["key"] = key
        by_key[key] = f
    groups: dict[tuple, list[dict]] = {}
    for f in by_key.values():
        groups.setdefault((f["oid"], f["action"]["name"], f["failure_reply"]), []).append(f)
    items = []
    for _, fs in sorted(groups.items()):
        fs.sort(key=lambda f: (len(f["history"]), h(f["key"])))
        log["dropped: same order, tool and error as a kept item"] += len(fs) - 1
        f = fs[0]
        item = _natural_item(db, f, log)
        if item:
            items.append(item)
    return items, log


def _natural_item(db: dict, f: dict, log: Counter) -> dict | None:
    tool, args, cls, oid, uid = f["action"]["name"], f["action"]["arguments"], f["cls"], f["oid"], f["uid"]
    twin = make_twin(f["history"], f["action"], cls)
    if twin.startswith("Error"):
        log[f"dropped: second check fails ({error_class(twin)})"] += 1
        return None
    env = fresh_env()
    env.replay(f["history"])
    shown = {}
    for x in f["history"]:
        try:
            rec = json.loads(x["output"])
        except ValueError:
            continue
        if isinstance(rec, dict) and {"order_id", "items", "status"} <= rec.keys():
            shown[rec["order_id"]] = rec
    target = shown[oid]
    item_id = args["item_ids"][0] if KIND[tool] in ("swap", "return") and args.get("item_ids") else None
    product = next((it["name"] for it in target["items"] if it["item_id"] == item_id), None) if item_id else None
    if "{p}" in TEMPLATES[tool] and not product:
        log["dropped: no product to name"] += 1
        return None
    others = sorted((w for w, r in shown.items() if w != oid and r.get("user_id") == uid), key=lambda w: h(f["key"], w))
    w2 = item2 = None
    for cand in others:
        if "{p}" not in TEMPLATES[tool]:
            w2 = cand
            break
        its = [it for it in shown[cand]["items"]
               if KIND[tool] != "swap" or len(db["products"][it["product_id"]]["variants"]) >= 2]
        if its:
            w2, item2 = cand, min(its, key=lambda it: (it["name"] == product, h(cand, it["item_id"])))
            break
    swapped = task_line(tool, w2, item2["name"] if item2 else None) if w2 else None
    if claim_problems(db, uid, tool, oid, product) or (w2 and claim_problems(db, uid, tool, w2, item2["name"] if item2 else None)):
        log["dropped: task line asserts a false fact"] += 1
        return None
    kind = KIND[tool]
    w_al = achievable(db, env, kind, oid, item_id)
    w_sw = achievable(db, env, kind, w2, item2["item_id"] if item2 else None) if w2 else None
    tr = f["tr"]
    log[f"kept {cls}"] += 1
    return {
        "id": f"tau-n-{f['key'][:12]}", "source": "natural", "cls": cls, "user_id": uid, "order_id": oid,
        "comparison_order_id": w2, "item_id": item_id, "comparison_item_id": item2["item_id"] if item2 else None,
        "product": product, "comparison_product": item2["name"] if item2 else None,
        "target_status": target["status"], "comparison_status": shown[w2]["status"] if w2 else None,
        "status_matched": bool(w2) and target["status"] == shown[w2]["status"], "target_first": None,
        "history": f["history"], "action": f["action"], "failure_reply": f["failure_reply"], "success_twin": twin,
        "task_aligned": task_line(tool, oid, product), "task_swapped": swapped, "task_neutral": NEUTRAL,
        "aligned_achievable": w_al is not None, "aligned_witness": w_al,
        "swapped_achievable": (w_sw is not None) if w2 else None, "swapped_witness": w_sw,
        "main_pool": w_al is not None, "key": f["key"],
        "task_mention_aligned": MENTION.format(o=oid), "task_mention_swapped": MENTION.format(o=w2) if w2 else None,
        "target_product_id": target_product_id(f["history"], oid, item_id),
        **witness_fields(f["history"], w_al, cls),
        "provenance": {"file": tr["file"], "simulation_id": tr["simulation_id"], "task_id": tr["task_id"],
                       "trial": tr["trial"], "agent": tr["agent"], "step": f["k"]},
    }


def build_all(natural_paths: list[Path] | None = None, max_users: int | None = None) -> dict:
    db = base_db()
    fewshot = fewshot_examples(db)
    exclude = {ex["user_id"] for ex in fewshot}
    generated, glog = build_generated(db, exclude, max_users)
    natural, nlog = (build_natural(db, load_trajectories(natural_paths), exclude) if natural_paths else ([], Counter()))
    return {"fewshot": fewshot, "generated": generated, "natural": natural,
            "log": {"generated": dict(sorted(glog.items())), "natural": dict(sorted(nlog.items()))}}
