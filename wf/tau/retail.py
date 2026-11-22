"""Dependency-free port of tau2-bench's retail toolkit at the pinned commit (notes/tau_design.md §2).

Method names, the order of every precondition check, the error strings and the JSON serialisation follow
upstream `src/tau2/domains/retail/tools.py` (kept as `vendor/upstream/tools.py.txt`). Records are plain dicts
whose keys are in upstream's pydantic field order, so `json.dumps` of a record is byte-identical to
upstream's `json.dumps(model.model_dump())`. scripts/tau_upstream_equivalence.py checks this against upstream.

The only addition is `bypass`: a set of check names that `RetailTools` skips, used to build a success twin
with exactly the violated check removed. With an empty set the tools are upstream's.

States are copy-on-write: `fork()` copies the three top-level tables, and a write call first deep-copies the
order and user records it may touch, so a fork never changes its parent (the pristine database included).
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, List

VENDOR = Path(__file__).resolve().parent / "vendor"
PINNED_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"

# The checks a twin may bypass, one per test class (design §3.4).
BYPASSABLE = ("status", "unavailable", "wrong_return_pm", "gc_balance")

WRITE_TOOLS = ("cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
               "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
               "return_delivered_order_items")
ORDER_WRITES = tuple(t for t in WRITE_TOOLS if t != "modify_user_address")
READ_TOOLS = ("calculate", "find_user_id_by_name_zip", "find_user_id_by_email", "get_order_details",
              "get_product_details", "get_item_details", "get_user_details", "list_all_product_types",
              "transfer_to_human_agents")
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS

# Field order of each upstream pydantic model (data_model.py); model_dump keeps it.
_ADDRESS = ("address1", "address2", "city", "country", "state", "zip")
_PM = {"credit_card": ("source", "id", "brand", "last_four"), "paypal": ("source", "id"),
       "gift_card": ("source", "id", "balance")}
_ORDER_OPTIONAL = ("cancel_reason", "exchange_items", "exchange_new_items", "exchange_payment_method_id",
                   "exchange_price_difference", "return_items", "return_payment_method_id")


def _pick(d: dict, keys: tuple[str, ...]) -> dict:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"record lacks {missing}: {sorted(d)}")
    return {k: d[k] for k in keys}


def _address(a: dict) -> dict:
    return _pick(a, _ADDRESS)


def _canonical(raw: dict) -> dict:
    """The database in upstream's model shape: field order, floats for float fields, optional order fields."""
    products = {}
    for pid, p in raw["products"].items():
        variants = {vid: {"item_id": v["item_id"], "options": dict(v["options"]), "available": bool(v["available"]),
                          "price": float(v["price"])} for vid, v in p["variants"].items()}
        products[pid] = {"name": p["name"], "product_id": p["product_id"], "variants": variants}
    users = {}
    for uid, u in raw["users"].items():
        pms = {}
        for k, m in u["payment_methods"].items():
            rec = _pick(m, _PM[m["source"]])
            if "balance" in rec:
                rec["balance"] = float(rec["balance"])
            pms[k] = rec
        users[uid] = {"user_id": u["user_id"], "name": _pick(u["name"], ("first_name", "last_name")),
                      "address": _address(u["address"]), "email": u["email"], "payment_methods": pms,
                      "orders": list(u["orders"])}
    orders = {}
    for oid, o in raw["orders"].items():
        rec = {"order_id": o["order_id"], "user_id": o["user_id"], "address": _address(o["address"]),
               "items": [{"name": i["name"], "product_id": i["product_id"], "item_id": i["item_id"],
                          "price": float(i["price"]), "options": dict(i["options"])} for i in o["items"]],
               "status": o["status"],
               "fulfillments": [{"tracking_id": list(f["tracking_id"]), "item_ids": list(f["item_ids"])}
                                for f in o["fulfillments"]],
               "payment_history": [{"transaction_type": p["transaction_type"], "amount": float(p["amount"]),
                                    "payment_method_id": p["payment_method_id"]} for p in o["payment_history"]]}
        for k in _ORDER_OPTIONAL:
            rec[k] = o.get(k)
        orders[oid] = rec
    return {"products": products, "users": users, "orders": orders}


def vendor_manifest() -> dict:
    return json.loads((VENDOR / "MANIFEST.json").read_text(encoding="utf-8"))


def check_vendor() -> list[str]:
    """Vendored files whose sha256 differs from MANIFEST.json (empty when all match)."""
    m = vendor_manifest()
    bad = [name for name, sha in m["files"].items()
           if not (VENDOR / name).exists() or hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != sha]
    raw = gzip.decompress((VENDOR / "db.json.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != m["db_json_sha256_uncompressed"]:
        bad.append("db.json (uncompressed)")
    return bad


def load_db_from_disk() -> dict:
    """A brand-new canonical database parsed from the vendored file, after checking its hash."""
    bad = check_vendor()
    if bad:
        raise SystemExit(f"vendored tau2 files differ from wf/tau/vendor/MANIFEST.json: {bad}")
    return _canonical(json.loads(gzip.decompress((VENDOR / "db.json.gz").read_bytes())))


@lru_cache(maxsize=1)
def _pristine() -> "State":
    return State(load_db_from_disk(), owned=None)


def fold_for_match(text: str) -> str:
    """Upstream tau2.utils.text_match.fold_for_match."""
    decomposed = unicodedata.normalize("NFD", text.strip())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


class State:
    """The three tables. `owned` is None for a state that owns every record (only the pristine one, never
    written), else the set of records this fork has copied and may change."""

    def __init__(self, db: dict, owned: set | None = None):
        self.products: dict = db["products"]
        self.users: dict = db["users"]
        self.orders: dict = db["orders"]
        self.owned = owned

    def fork(self) -> "State":
        return State({"products": dict(self.products), "users": dict(self.users), "orders": dict(self.orders)},
                     owned=set())

    def own_order(self, order_id: Any) -> None:
        if isinstance(order_id, str) and order_id in self.orders and ("o", order_id) not in self.owned:
            self.orders[order_id] = copy.deepcopy(self.orders[order_id])
            self.owned.add(("o", order_id))

    def own_user(self, user_id: Any) -> None:
        if isinstance(user_id, str) and user_id in self.users and ("u", user_id) not in self.owned:
            self.users[user_id] = copy.deepcopy(self.users[user_id])
            self.owned.add(("u", user_id))

    def as_dict(self) -> dict:
        return {"products": self.products, "users": self.users, "orders": self.orders}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict()).encode()).hexdigest()


class RetailTools:
    """Upstream RetailTools, method for method. Comments name the upstream check each `if` ports."""

    def __init__(self, db: State, bypass: frozenset[str] = frozenset()):
        unknown = set(bypass) - set(BYPASSABLE)
        if unknown:
            raise ValueError(f"unknown bypass {sorted(unknown)}")
        self.db = db
        self.bypass = frozenset(bypass)

    # ---- lookups ----
    def _get_order(self, order_id: str) -> dict:
        if order_id not in self.db.orders:
            raise ValueError("Order not found")
        return self.db.orders[order_id]

    def _get_user(self, user_id: str) -> dict:
        if user_id not in self.db.users:
            raise ValueError("User not found")
        return self.db.users[user_id]

    def _get_product(self, product_id: str) -> dict:
        if product_id not in self.db.products:
            raise ValueError("Product not found")
        return self.db.products[product_id]

    def _get_item(self, item_id: str) -> dict:
        for _, product in self.db.products.items():
            if item_id in product["variants"]:
                return product["variants"][item_id]
        raise ValueError("Item not found")

    def _get_variant(self, product_id: str, variant_id: str) -> dict:
        product = self._get_product(product_id)
        if variant_id not in product["variants"]:
            raise ValueError("Variant not found")
        return product["variants"][variant_id]

    def _get_payment_method(self, user_id: str, payment_method_id: str) -> dict:
        user = self._get_user(user_id)
        if payment_method_id not in user["payment_methods"]:
            raise ValueError("Payment method not found")
        return user["payment_methods"][payment_method_id]

    @staticmethod
    def _is_gift_card(payment_method: dict) -> bool:
        return payment_method["source"] == "gift_card"

    def _is_pending_order(self, order: dict) -> bool:
        return "pending" in order["status"]

    # ---- generic ----
    def calculate(self, expression: str) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    def transfer_to_human_agents(self, summary: str) -> str:
        return "Transfer successful"

    # ---- reads ----
    def find_user_id_by_name_zip(self, first_name: str, last_name: str, zip: str) -> str:
        wanted_first = fold_for_match(first_name)
        wanted_last = fold_for_match(last_name)
        matches = []
        for user_id, user in self.db.users.items():
            if (fold_for_match(user["name"]["first_name"]) == wanted_first
                    and fold_for_match(user["name"]["last_name"]) == wanted_last
                    and user["address"]["zip"] == zip):
                matches.append(user_id)
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError("User lookup is ambiguous")
        raise ValueError("User not found")

    def find_user_id_by_email(self, email: str) -> str:
        wanted = fold_for_match(email)
        matches = []
        for user_id, user in self.db.users.items():
            if fold_for_match(user["email"]) == wanted:
                matches.append(user_id)
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError("User lookup is ambiguous")
        raise ValueError("User not found")

    def get_order_details(self, order_id: str) -> dict:
        return self._get_order(order_id)

    def get_product_details(self, product_id: str) -> dict:
        return self._get_product(product_id)

    def get_item_details(self, item_id: str) -> dict:
        return self._get_item(item_id)

    def get_user_details(self, user_id: str) -> dict:
        return self._get_user(user_id)

    def list_all_product_types(self) -> str:
        product_dict = {product["name"]: product["product_id"] for product in self.db.products.values()}
        return json.dumps(product_dict, sort_keys=True)

    # ---- writes ----
    def cancel_pending_order(self, order_id: str, reason: str) -> dict:
        order = self._get_order(order_id)
        if "status" not in self.bypass and order["status"] != "pending":
            raise ValueError("Non-pending order cannot be cancelled")
        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("Invalid reason")
        refunds = []
        for payment in order["payment_history"]:
            payment_id = payment["payment_method_id"]
            refunds.append({"transaction_type": "refund", "amount": float(payment["amount"]),
                            "payment_method_id": payment_id})
            user = self._get_user(order["user_id"])
            payment_method = self._get_payment_method(user["user_id"], payment_id)
            if self._is_gift_card(payment_method):
                payment_method["balance"] += payment["amount"]
                payment_method["balance"] = round(payment_method["balance"], 2)
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(refunds)
        return order

    def exchange_delivered_order_items(self, order_id: str, item_ids: List[str], new_item_ids: List[str],
                                       payment_method_id: str) -> dict:
        order = self._get_order(order_id)
        if "status" not in self.bypass and order["status"] != "delivered":
            raise ValueError("Non-delivered order cannot be exchanged")
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                raise ValueError(f"Number of {item_id} not found.")
        if len(item_ids) != len(new_item_ids):
            raise ValueError("The number of items to be exchanged should match.")
        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = next((item for item in order["items"] if item["item_id"] == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            product_id = item["product_id"]
            variant = self._get_variant(product_id, new_item_id)
            if "unavailable" not in self.bypass and not variant["available"]:
                raise ValueError(f"New item {new_item_id} not found or available")
            diff_price += variant["price"] - item["price"]
        diff_price = round(diff_price, 2)
        payment_method = self._get_payment_method(order["user_id"], payment_method_id)
        if ("gc_balance" not in self.bypass and self._is_gift_card(payment_method)
                and payment_method["balance"] < diff_price):
            raise ValueError("Insufficient gift card balance to pay for the price difference")
        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = diff_price
        return order

    def modify_pending_order_address(self, order_id: str, address1: str, address2: str, city: str, state: str,
                                     country: str, zip: str) -> dict:
        order = self._get_order(order_id)
        if "status" not in self.bypass and not self._is_pending_order(order):
            raise ValueError("Non-pending order cannot be modified")
        order["address"] = {"address1": address1, "address2": address2, "city": city, "country": country,
                            "state": state, "zip": zip}
        return order

    def modify_pending_order_items(self, order_id: str, item_ids: List[str], new_item_ids: List[str],
                                   payment_method_id: str) -> dict:
        order = self._get_order(order_id)
        if "status" not in self.bypass and order["status"] != "pending":
            raise ValueError("Non-pending order cannot be modified")
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                raise ValueError(f"{item_id} not found")
        if len(item_ids) != len(new_item_ids):
            raise ValueError("The number of items to be exchanged should match")
        diff_price = 0
        variant = None
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            if item_id == new_item_id:
                raise ValueError("The new item id should be different from the old item id")
            item = next((item for item in order["items"] if item["item_id"] == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            product_id = item["product_id"]
            variant = self._get_variant(product_id, new_item_id)
            if "unavailable" not in self.bypass and not variant["available"]:
                raise ValueError(f"New item {new_item_id} not found or available")
            diff_price += variant["price"] - item["price"]
        payment_method = self._get_payment_method(order["user_id"], payment_method_id)
        if ("gc_balance" not in self.bypass and self._is_gift_card(payment_method)
                and payment_method["balance"] < diff_price):
            raise ValueError("Insufficient gift card balance to pay for the new item")
        # OrderPayment(...) validates amount as a float
        order["payment_history"].append({"transaction_type": "payment" if diff_price > 0 else "refund",
                                         "amount": float(abs(diff_price)), "payment_method_id": payment_method_id})
        if self._is_gift_card(payment_method):
            payment_method["balance"] -= diff_price
            payment_method["balance"] = round(payment_method["balance"], 2)
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = next((item for item in order["items"] if item["item_id"] == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            item["item_id"] = new_item_id
            # upstream uses the last loop's variant for every item; kept as is
            item["price"] = variant["price"]
            item["options"] = variant["options"]
        order["status"] = "pending (item modified)"
        return order

    def modify_pending_order_payment(self, order_id: str, payment_method_id: str) -> dict:
        order = self._get_order(order_id)
        if "status" not in self.bypass and not self._is_pending_order(order):
            raise ValueError("Non-pending order cannot be modified")
        payment_method = self._get_payment_method(order["user_id"], payment_method_id)
        if len(order["payment_history"]) != 1 or order["payment_history"][0]["transaction_type"] != "payment":
            raise ValueError("There should be exactly one payment for a pending order")
        if order["payment_history"][0]["payment_method_id"] == payment_method_id:
            raise ValueError("The new payment method should be different from the current one")
        amount = order["payment_history"][0]["amount"]
        if ("gc_balance" not in self.bypass and self._is_gift_card(payment_method)
                and payment_method["balance"] < amount):
            raise ValueError("Insufficient gift card balance to pay for the order")
        order["payment_history"].extend([
            {"transaction_type": "payment", "amount": float(amount), "payment_method_id": payment_method_id},
            {"transaction_type": "refund", "amount": float(amount),
             "payment_method_id": order["payment_history"][0]["payment_method_id"]},
        ])
        if self._is_gift_card(payment_method):
            payment_method["balance"] -= amount
            payment_method["balance"] = round(payment_method["balance"], 2)
        old_payment_method = self._get_payment_method(order["user_id"], order["payment_history"][0]["payment_method_id"])
        if self._is_gift_card(old_payment_method):
            old_payment_method["balance"] += amount
            old_payment_method["balance"] = round(old_payment_method["balance"], 2)
        return order

    def modify_user_address(self, user_id: str, address1: str, address2: str, city: str, state: str,
                            country: str, zip: str) -> dict:
        user = self._get_user(user_id)
        user["address"] = {"address1": address1, "address2": address2, "city": city, "country": country,
                           "state": state, "zip": zip}
        return user

    def return_delivered_order_items(self, order_id: str, item_ids: List[str], payment_method_id: str) -> dict:
        order = self._get_order(order_id)
        if "status" not in self.bypass and order["status"] != "delivered":
            raise ValueError("Non-delivered order cannot be returned")
        user = self._get_user(order["user_id"])
        payment_method = self._get_payment_method(user["user_id"], payment_method_id)
        if ("wrong_return_pm" not in self.bypass and not self._is_gift_card(payment_method)
                and payment_method_id != order["payment_history"][0]["payment_method_id"]):
            raise ValueError("Payment method should be the original payment method")
        all_item_ids = [item["item_id"] for item in order["items"]]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                raise ValueError("Some item not found")
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        order["return_payment_method_id"] = payment_method_id
        return order


def to_json_str(resp: Any) -> str:
    """Upstream Environment.to_json_str for the response types the retail tools return."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, (dict, list)) or resp is None:
        return json.dumps(resp, default=str)
    if isinstance(resp, (int, float, bool)):
        return json.dumps(str(resp))
    raise ValueError(f"Unsupported type: {type(resp)}")


class RetailEnv:
    """One environment over one state; `call` is upstream Environment.get_response's content."""

    def __init__(self, state: State, bypass: frozenset[str] = frozenset()):
        if state.owned is None:
            raise ValueError("the pristine database is never written; use fresh_env() or State.fork()")
        self.state = state
        self.tools = RetailTools(state, bypass)

    def fork(self, bypass: frozenset[str] = frozenset()) -> "RetailEnv":
        return RetailEnv(self.state.fork(), bypass)

    def _isolate(self, name: str, arguments: dict) -> None:
        """Copy, before a write, every record the write may change, so the parent state is never touched."""
        if name not in WRITE_TOOLS or not isinstance(arguments, dict):
            return
        oid = arguments.get("order_id")
        self.state.own_order(oid)
        if isinstance(oid, str) and oid in self.state.orders:
            self.state.own_user(self.state.orders[oid].get("user_id"))
        self.state.own_user(arguments.get("user_id"))

    def call(self, name: str, arguments: dict) -> str:
        try:
            if name not in ALL_TOOLS:
                raise ValueError(f"Tool '{name}' not found.")
            self._isolate(name, arguments)
            resp = getattr(self.tools, name)(**arguments)
        except Exception as e:  # upstream: resp = f"Error: {e}"
            resp = f"Error: {e}"
        return to_json_str(resp)

    def replay(self, calls: list[dict]) -> list[str]:
        return [self.call(c["name"], c["arguments"]) for c in calls]

    def call_bypassed(self, name: str, arguments: dict, bypass: frozenset[str]) -> str:
        """`call` on this same state with the named checks skipped (the success twin, design §3.4)."""
        return RetailEnv(self.state, bypass).call(name, arguments)


def fresh_env() -> RetailEnv:
    """A new environment on a fork of the pristine database (parsed once per process, never written)."""
    return RetailEnv(_pristine().fork())


def pristine_sha256() -> str:
    return _pristine().sha256()


def base_db() -> dict:
    """Read-only view of the pristine tables, for planning in the builder. Never mutate it."""
    return _pristine().as_dict()
