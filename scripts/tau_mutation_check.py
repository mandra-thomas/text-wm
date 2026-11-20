#!/usr/bin/env python3
"""Deliberate bugs in the τ-bench port, builder, checker, verifier, prompts and runs (notes/tau_design.md §8.6).

Each mutant is applied alone to a copy of the repo, and tests/test_tau_*.py must fail. A timeout counts as
inconclusive, never as caught. Usage: python scripts/tau_mutation_check.py [shard/shards]

Three groups: the original 29; review 1's 27 (prefixed R1-, adapted where the code they hit was rewritten in the
amendments of 2026-11-20: A1 now hits the user cluster in cluster_of, A3 the Holm-adjusted p in verdicts, A5 the
missing-score set, A8 the identifier regex, G1 the pass-flag check); and 30 for the amended code (N1-N30).

Equivalence arguments (review 1 marked B3, B5 and G1 as candidates; each now has a killing test, so none is argued
away): B3 cannot change the item set on this database, which has no product with fewer than two variants, so it is
caught by a unit test of claim_problems on a constructed database; B5 lets a gift card be proposed as the wrong
refund method, which the build then rejects as succeeding, and a test requires that no candidate is rejected for
any reason but a twin's second check; G1 removes a check that _finished already implies for real runs, and a test
sets verify.json's flag to false directly.
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
     '            if now.get(name) != pins[name]:', '            if False:'),
    ("scoring ignores a changed file", "wf/tau/pipeline.py",
     '    if rep.get("files") != now:', '    if False:'),
    ("one-sided verdict", "wf/tau/pipeline.py",
     '        if est <= -SUPPORT:\n            return "reversed (naming sharpens failure prediction)"\n', ""),
    ("contrasts clustered by item", "wf/tau/pipeline.py", '            users.append(t["user_id"])', '            users.append(t["id"])'),
    ("shards split items, not users", "wf/tau/pipeline.py",
     '    keep = set(sorted({t["user_id"] for t in items})[i::n])\n    return [t for t in items if t["user_id"] in keep]',
     '    return items[i::n]'),

    # ---- review 1's 27 mutants (scratchpad review_tau/my_mutants.py), adapted where the code they hit was rewritten ----
    ("R1-P1 modify-items rounds price diff", "wf/tau/retail.py",
     '            diff_price += variant["price"] - item["price"]\n        payment_method = self._get_payment_method(order["user_id"], payment_method_id)',
     '            diff_price += variant["price"] - item["price"]\n        diff_price = round(diff_price, 2)\n        payment_method = self._get_payment_method(order["user_id"], payment_method_id)'),
    ("R1-P2 exchange gift-card boundary <=", "wf/tau/retail.py",
     'and payment_method["balance"] < diff_price):\n            raise ValueError("Insufficient gift card balance to pay for the price difference")',
     'and payment_method["balance"] <= diff_price):\n            raise ValueError("Insufficient gift card balance to pay for the price difference")'),
    ("R1-P3 payment: same-pm check before one-payment", "wf/tau/retail.py",
     '        if len(order["payment_history"]) != 1 or order["payment_history"][0]["transaction_type"] != "payment":\n            raise ValueError("There should be exactly one payment for a pending order")\n        if order["payment_history"][0]["payment_method_id"] == payment_method_id:\n            raise ValueError("The new payment method should be different from the current one")',
     '        if order["payment_history"][0]["payment_method_id"] == payment_method_id:\n            raise ValueError("The new payment method should be different from the current one")\n        if len(order["payment_history"]) != 1 or order["payment_history"][0]["transaction_type"] != "payment":\n            raise ValueError("There should be exactly one payment for a pending order")'),
    ("R1-P4 exchange items not sorted", "wf/tau/retail.py", 'order["exchange_items"] = sorted(item_ids)', 'order["exchange_items"] = list(item_ids)'),
    ("R1-P5 cancel gift-card refund not rounded", "wf/tau/retail.py",
     '                payment_method["balance"] += payment["amount"]\n                payment_method["balance"] = round(payment_method["balance"], 2)',
     '                payment_method["balance"] += payment["amount"]'),
    ("R1-P6 modify-items uses each pair's own variant", "wf/tau/retail.py",
     '            item["price"] = variant["price"]\n            item["options"] = variant["options"]',
     '            v2 = self._get_variant(item["product_id"], new_item_id)\n            item["price"] = v2["price"]\n            item["options"] = v2["options"]'),
    ("R1-P7 modify-address writes address in arg order", "wf/tau/retail.py",
     '        order["address"] = {"address1": address1, "address2": address2, "city": city, "country": country,\n                            "state": state, "zip": zip}\n        return order',
     '        order["address"] = {"address1": address1, "address2": address2, "city": city, "state": state,\n                            "country": country, "zip": zip}\n        return order'),
    ("R1-P8 calculate not rounded", "wf/tau/retail.py", 'return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))',
     'return str(float(eval(expression, {"__builtins__": None}, {})))'),
    ("R1-C1 checker exchange gift card not rounded", "wf/tau/checker.py", 'needed = diff if modify else round(diff, 2)', 'needed = diff'),
    ("R1-C2 checker return: count before pm", "wf/tau/checker.py",
     '        bad, method = _pm_facts(view, order, pm)\n        if bad:\n            return bad\n        original = order["payment_history"][0]["payment_method_id"]',
     '        if _count_problem(args.get("item_ids") or [], order) is not None:\n            return _fail("item_not_in_order", "Some item not found", "")\n        bad, method = _pm_facts(view, order, pm)\n        if bad:\n            return bad\n        original = order["payment_history"][0]["payment_method_id"]'),
    ("R1-B1 product reads not counterbalanced", "wf/tau/items.py", 'prods = products if target_first else list(reversed(products))', 'prods = products'),
    ("R1-B2 comparison item prefers the same product name", "wf/tau/items.py", '(it["name"] == target_name, h(w2, it["item_id"]))',
     '(it["name"] != target_name, h(w2, it["item_id"]))'),
    ("R1-B3 claim check ignores one-variant products", "wf/tau/items.py", 'elif KIND[tool] == "swap" and all(', 'elif False and all('),
    ("R1-B4 achievability witnesses run on the item state (not a fork)", "wf/tau/items.py",
     'if not env.fork().call(call["name"], call["arguments"]).startswith("Error"):', 'if not env.call(call["name"], call["arguments"]).startswith("Error"):'),
    ("R1-B5 wrong_return_pm allows gift cards", "wf/tau/items.py", 'if m["source"] != "gift_card" and p != pm0]', 'if p != pm0]'),
    ("R1-R1 trailing space after Tool output:", "wf/tau/prompts.py", 'OUTPUT_CUE = "Tool output:"', 'OUTPUT_CUE = "Tool output: "'),
    ("R1-R2 few-shot turns carry the query's task", "wf/tau/prompts.py",
     'messages.append({"role": "user", "content": user_turn(ex["task"], ex["history"], ex["action"])})',
     'messages.append({"role": "user", "content": user_turn(task_for(item, condition), ex["history"], ex["action"])})'),
    ("R1-R3 neutral silently falls back", "wf/tau/prompts.py", '        raise ValueError(f"{item[\'id\']}: no {condition} task line")',
     '        line = item["task_aligned"]'),
    ("R1-A1 no clustering (item-level bootstrap/permutation)", "wf/tau/pipeline.py",
     '    if by == "user":\n        return t["user_id"]', '    if by == "user":\n        return t["id"]'),
    ("R1-A2 verdict one-sided CI", "wf/tau/pipeline.py", 'excludes_zero = ci[0] > 0 or ci[1] < 0', 'excludes_zero = ci[0] > 0'),
    ("R1-A3 Holm dropped", "wf/tau/pipeline.py", 'v = tau_verdict(primary[m]["estimate"], primary[m]["ci"], adjusted[m])',
     'v = tau_verdict(primary[m]["estimate"], primary[m]["ci"], primary[m]["perm_p"])'),
    ("R1-A4 full_norm divides by wrong counts", "wf/tau/pipeline.py",
     'return rec["logp_plus"] / rec["n_tokens_plus"] - rec["logp_minus"] / rec["n_tokens_minus"]',
     'return rec["logp_plus"] / rec["n_tokens_minus"] - rec["logp_minus"] / rec["n_tokens_plus"]'),
    ("R1-A5 analysis does not require swapped scores", "wf/tau/pipeline.py",
     'for job in jobs_for(t, scfg, rules_ids)}', 'for job in jobs_for(t, scfg, rules_ids) if job[3] != "swapped"}'),
    ("R1-A6 stale-score check disabled", "wf/tau/pipeline.py",
     'stale = sum(1 for s in scores if hashes.get(s["transition_id"]) != s.get("item_hash"))', 'stale = 0'),
    ("R1-A7 neutral never scored", "wf/tau/pipeline.py", 'conds = [c for c in CONDITIONS if c != "swapped" or t.get("task_swapped")]',
     'conds = [c for c in CONDITIONS if c != "neutral" and (c != "swapped" or t.get("task_swapped"))]'),
    ("R1-A8 contamination: only ids of 7+ digits", "wf/tau/pipeline.py", 'ID_DIGITS = re.compile(r"\\d{5,}")', 'ID_DIGITS = re.compile(r"\\d{7,}")'),
    ("R1-G1 scoring gate ignores verify pass flag", "wf/tau/pipeline.py",
     '    if not rep.get("passed"):\n        return False, "verify.json does not say passed"\n', ''),
    # ---- the amendments of 2026-11-20 (controls, mention pair, manipulation check, rules-only arm, verdicts) ----
    ("N1 few-shot examples out of the design's order", "wf/tau/items.py", '                         "history": history, "action": {"name": tool, "arguments": args}, "output": env.call(tool, args)})\n    return examples',
     '                         "history": history, "action": {"name": tool, "arguments": args}, "output": env.call(tool, args)})\n    return [examples[0], examples[2], examples[1], examples[3]]'),
    ("N2 status witness failure text ignores the witness tool", "wf/tau/items.py", '        return "Error: " + _STATUS_ERRORS[tool]',
     '        return "Error: Non-pending order cannot be modified"'),
    ("N3 witness checker verdict not computed", "wf/tau/items.py", '"witness_checker": check(history, witness).verdict}', '"witness_checker": "PASS"}'),
    ("N4 mention-swapped names the target order", "wf/tau/items.py", '"task_mention_swapped": MENTION.format(o=w2),',
     '"task_mention_swapped": MENTION.format(o=oid),'),
    ("N5 backend rules left out of the main arm", "wf/tau/prompts.py", '    return f"{INSTRUCTION}\\n\\n{backend}\\n\\n{policy}\\n\\n# Tools\\n\\n{tool_lines}"',
     '    return f"{INSTRUCTION}\\n\\n{policy}\\n\\n# Tools\\n\\n{tool_lines}"'),
    ("N6 rules-only arm keeps the policy", "wf/tau/prompts.py", '    if arm == "rules_only":\n        return', '    if False:\n        return'),
    ("N7 manipulation question is the yes/no question", "wf/tau/prompts.py", '"target": TARGET_QUESTION}', '"target": YN_QUESTION}'),
    ("N8 witness action scores the failing call", "wf/tau/prompts.py", '        return item["aligned_witness"]', '        return item["action"]'),
    ("N9 primary population ignores status matching", "wf/tau/pipeline.py",
     'return t["source"] == "generated" and t["main_pool"] and t["status_matched"]', 'return t["source"] == "generated" and t["main_pool"]'),
    ("N10 mention pair never scheduled", "wf/tau/pipeline.py", '    if cfg.get("mention", True):', '    if False:'),
    ("N11 witness full readout uses the item's twin", "wf/tau/pipeline.py",
     'if action == "fail" else (t["witness_output"], t["witness_failure"]))', 'if True else (t["witness_output"], t["witness_failure"]))'),
    ("N12 null verdict without the manipulation check", "wf/tau/pipeline.py", '        if v.startswith("no effect") and not manip:', '        if False:'),
    ("N13 effect always read as override", "wf/tau/pipeline.py", 'if disc else', 'if True else'),
    ("N14 discrimination on every witness", "wf/tau/pipeline.py", '    determined = [t for t in pop if t.get("witness_checker") == "PASS"]', '    determined = pop'),
    ("N15 one model counts as a replication", "wf/tau/pipeline.py", '    if rep == len(v):', '    if rep:'),
    ("N16 token gate disabled", "wf/tau/pipeline.py", '        if not report.get("first_token_ok"):', '        if False:'),
    ("N17 an unpinned file is accepted", "wf/tau/pipeline.py",
     '            if pins.get(name) is None:\n                return False, f"no pinned sha256 for {name}"', '            if pins.get(name) is None:\n                continue'),
    ("N18 addresses resampled", "wf/tau/pipeline.py", 'KEEP_SUBTREES = ("address",)', 'KEEP_SUBTREES = ()'),
    ("N19 flips toward failure counted as toward success", "wf/tau/pipeline.py", '    down = sum(1 for a, s in pairs if s > 0 > a) / len(pairs)',
     '    down = sum(1 for a, s in pairs if a > 0 > s) / len(pairs)'),
    ("N20 product clusters are users", "wf/tau/pipeline.py", '        return t.get("target_product_id") or f"no-product:{t[\'id\']}"', '        return t["user_id"]'),
    ("N21 probability shift on the logit scale", "wf/tau/pipeline.py", '    if scale == "prob":', '    if False:'),
    ("N22 verifier skips the witness", "wf/tau/verify.py", '        if want != got:', '        if False:'),
    ("N23 verifier skips the comparison-item rule", "wf/tau/verify.py", '        if other_names - {it["product"]}:', '        if False:'),
    ("N24 verifier skips the product-read order", "wf/tau/verify.py", '            if shown_products != list(dict.fromkeys(pair)):', '            if False:'),
    ("N25 contamination baseline sign flipped", "wf/tau/pipeline.py", 'diffs = [by_model[tag][i]["L"] - base[i]["L"] for i in ids]', 'diffs = [base[i]["L"] - by_model[tag][i]["L"] for i in ids]'),
    ("N26 contrast of effects adds instead of subtracting", "wf/tau/pipeline.py", '        v = (L[0] - L[1]) - (L[2] - L[3])', '        v = (L[0] - L[1]) + (L[2] - L[3])'),
    ("N27 verifier accepts few-shot examples in any order", "wf/tau/verify.py", '    if cells != [("own", "success"), ("other", "failure"), ("other", "success"), ("own", "failure")]:', '    if False:'),
    ("N28 rules-only arm drawn from the whole main pool", "wf/tau/pipeline.py", 'return set(sorted((t["id"] for t in items if is_primary(t))', 'return set(sorted((t["id"] for t in items if t["main_pool"])'),
    ("N29 seconds not recorded", "wf/tau/pipeline.py", '                                                "seconds": round(time.perf_counter() - t0, 4)}])', '                                                }])'),
    ("N30 discrimination compares the witness with itself", "wf/tau/pipeline.py", 'action="witness", action_b="fail") for r in ("first", "yn")}\n    out["discrimination_all',
     'action="witness", action_b="witness") for r in ("first", "yn")}\n    out["discrimination_all'),
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
