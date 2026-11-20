#!/usr/bin/env python3
"""Deliberate bugs in the τ-bench port, builder, checker, verifier, prompts and runs (notes/tau_design.md §8.6).

Each mutant is applied alone to a copy of the repo, and tests/test_tau_*.py must fail. A timeout counts as
inconclusive, never as caught. Usage: python scripts/tau_mutation_check.py [shard/shards]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_tau_retail.py", "tests/test_tau_checker.py", "tests/test_tau_items.py", "tests/test_tau_pipeline.py"]

# (bug, file, correct text, buggy text)
MUTANTS = [
    # port
    ("cancel accepts 'pending (item modified)'", "wf/tau/retail.py",
     'if "status" not in self.bypass and order["status"] != "pending":\n            raise ValueError("Non-pending order cannot be cancelled")',
     'if "status" not in self.bypass and not self._is_pending_order(order):\n            raise ValueError("Non-pending order cannot be cancelled")'),
    ("return error says exchanged", "wf/tau/retail.py", 'raise ValueError("Non-delivered order cannot be returned")',
     'raise ValueError("Non-delivered order cannot be exchanged")'),
    ("return refuses non-original gift cards", "wf/tau/retail.py",
     'if ("wrong_return_pm" not in self.bypass and not self._is_gift_card(payment_method)\n',
     'if ("wrong_return_pm" not in self.bypass\n'),
    ("optional order fields not serialised", "wf/tau/retail.py", "        for k in _ORDER_OPTIONAL:\n            rec[k] = o.get(k)\n", ""),
    ("fork shares order records", "wf/tau/retail.py",
     "            self.orders[order_id] = copy.deepcopy(self.orders[order_id])\n", ""),
    ("failed return writes first", "wf/tau/retail.py",
     '        user = self._get_user(order["user_id"])\n        payment_method = self._get_payment_method(user["user_id"], payment_method_id)\n        if ("wrong_return_pm"',
     '        order["return_payment_method_id"] = payment_method_id\n        user = self._get_user(order["user_id"])\n        payment_method = self._get_payment_method(user["user_id"], payment_method_id)\n        if ("wrong_return_pm"'),
    ("bypass ignored for availability", "wf/tau/retail.py",
     'if "unavailable" not in self.bypass and not variant["available"]:\n                raise ValueError(f"New item {new_item_id} not found or available")\n            diff_price += variant["price"] - item["price"]\n        diff_price = round(diff_price, 2)',
     'if not variant["available"]:\n                raise ValueError(f"New item {new_item_id} not found or available")\n            diff_price += variant["price"] - item["price"]\n        diff_price = round(diff_price, 2)'),
    # checker
    ("checker: modify accepts any pending status", "wf/tau/checker.py",
     'if modify and status != "pending":', 'if modify and "pending" not in status:'),
    ("checker: stale balances trusted", "wf/tau/checker.py",
     "        for j, h in self.writes:\n            if j > i", "        for j, h in []:\n            if j > i"),
    ("checker: first record wins", "wf/tau/checker.py",
     '                    self.orders[rec["order_id"]] = (i, rec)', '                    self.orders.setdefault(rec["order_id"], (i, rec))'),
    ("checker: gift card boundary", "wf/tau/checker.py", "            if bal < ph[0][\"amount\"]:", "            if bal <= ph[0][\"amount\"]:"),
    ("checker: no status check on return", "wf/tau/checker.py",
     '        if status != "delivered":\n            return _fail("status", "Non-delivered order cannot be returned", where)\n', ""),
    ("checker: guesses when the product is not shown", "wf/tau/checker.py",
     '            return _unknown(f"product {item[\'product_id\']} not shown"), 0.0', '            continue'),
    # builder
    ("comparison not status-matched first", "wf/tau/items.py",
     'key=lambda x: (db["orders"][x]["status"] != st, h(oid, x))', 'key=lambda x: h(oid, x)'),
    ("target order always shown first", "wf/tau/items.py", 'target_first = h(oid, w2, "order") % 2 == 0', "target_first = True"),
    ("twin bypasses every check", "wf/tau/items.py",
     'return env.call_bypassed(action["name"], action["arguments"], frozenset({cls}))',
     'return env.call_bypassed(action["name"], action["arguments"], frozenset({"status", "unavailable", "wrong_return_pm", "gc_balance"}))'),
    ("twin replays history with the bypass", "wf/tau/items.py",
     '    env = fresh_env()\n    env.replay(history)\n    return env.call_bypassed',
     '    env = fresh_env().fork(frozenset({cls}))\n    env.replay(history)\n    return env.call_bypassed'),
    ("every task achievable", "wf/tau/items.py",
     '        if not env.fork().call(call["name"], call["arguments"]).startswith("Error"):\n            return call',
     '        return call'),
    ("claim check disabled", "wf/tau/items.py",
     '    problems = []\n    if order_id not in db["users"][user_id]["orders"]', '    return []\n    problems = []\n    if order_id not in db["users"][user_id]["orders"]'),
    ("natural echoes kept", "wf/tau/items.py",
     'log["dropped: echo of an earlier failure on the same order"] += 1\n                continue',
     'log["dropped: echo of an earlier failure on the same order"] += 1'),
    ("natural dedupe keeps the longest", "wf/tau/items.py",
     'fs.sort(key=lambda f: (len(f["history"]), h(f["key"])))', 'fs.sort(key=lambda f: (-len(f["history"]), h(f["key"])))'),
    # verifier
    ("verifier skips the twin", "wf/tau/verify.py",
     '    if I.make_twin(it["history"], it["action"], it["cls"]) != it["success_twin"]:', '    if False:'),
    ("verifier skips achievability", "wf/tau/verify.py",
     "    if (w_al is not None) != it[\"aligned_achievable\"]", "    if False and (w_al is not None) != it[\"aligned_achievable\"]"),
    # prompts and runs
    ("query uses the aligned task always", "wf/tau/prompts.py",
     'messages.append({"role": "user", "content": user_turn(task_for(item, condition)', 'messages.append({"role": "user", "content": user_turn(item["task_aligned"]'),
    ("scoring ignores the pinned sha", "wf/tau/pipeline.py",
     '    if pinned and now.get("items.jsonl") != pinned:', '    if False:'),
    ("scoring ignores a changed file", "wf/tau/pipeline.py",
     '    if not rep.get("passed") or rep.get("files") != now:', '    if not rep.get("passed"):'),
    ("one-sided verdict", "wf/tau/pipeline.py",
     '        if est <= -SUPPORT:\n            return "reversed (naming sharpens failure prediction)"\n', ""),
    ("effects clustered by item", "wf/tau/pipeline.py", '            users.append(t["user_id"])', '            users.append(t["id"])'),
    ("shards split items, not users", "wf/tau/pipeline.py",
     '    keep = set(sorted({t["user_id"] for t in items})[i::n])\n    return [t for t in items if t["user_id"] in keep]',
     '    return items[i::n]'),
]


def _run(python: str, work: Path, timeout: int) -> str:
    try:
        r = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no", *TESTS], cwd=work,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    return "killed" if r.returncode != 0 else "survived"


def main() -> int:
    shard, shards = map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1)
    chosen = MUTANTS[shard::shards]
    only = os.environ.get("MUTANTS_MATCHING")  # "a|b": only mutants whose name contains a or b
    if only:
        chosen = [m for m in chosen if any(o in m[0] for o in only.split("|"))]
    timeout = int(os.environ.get("MUTATION_TIMEOUT", "900"))
    results: dict[str, list[str]] = {"killed": [], "survived": [], "timeout": [], "not applied": []}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT, work, ignore=lambda folder, names: [n for n in names if n in (".git", ".venv", "runs", "runs_mock", "__pycache__", ".pytest_cache")])
        for name, rel, good, bad in chosen:
            path = work / rel
            src = path.read_text(encoding="utf-8")
            if src.count(good) != 1:
                results["not applied"].append(name)
                print(f"NOT APPLIED  {name}", flush=True)
                continue
            path.write_text(src.replace(good, bad), encoding="utf-8")
            outcome = _run(sys.executable, work, timeout)
            path.write_text(src, encoding="utf-8")
            results[outcome].append(name)
            print(f"{outcome:<12} {name}", flush=True)
    print(f"\n{len(results['killed'])}/{len(chosen)} killed; survived={results['survived']}; "
          f"timeout={results['timeout']}; not applied={results['not applied']}")
    return 0 if len(results["killed"]) == len(chosen) else 1


if __name__ == "__main__":
    sys.exit(main())
