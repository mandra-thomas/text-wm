#!/usr/bin/env python3
"""Deliberate bugs in the τ-bench port, builder, checker, verifier, prompts and runs (notes/tau_design.md §8.6).

Each mutant is applied alone to a copy of the repo, and tests/test_tau_*.py must fail. A timeout counts as
inconclusive, never as caught. Usage:

    python scripts/tau_mutation_check.py [shard/shards] [--out results.json]

Before any mutant, the unmutated copy must pass the tests (otherwise every mutant would look killed). Mutants run
one at a time (serially) in one copy; a shard i/n takes MUTANTS[i::n], a deterministic split. For each killed
mutant the first failing test's id and its error line are recorded (pytest -x --tb=line -rf), so every kill can be
audited.

Four groups: the original 29; review 1's 27 (prefixed R1-, adapted where the code they hit was rewritten in the
amendments of 2026-11-20: A1 now hits the user cluster in cluster_of, A3 the Holm-adjusted p in verdicts, A5 the
missing-score set, A8 the identifier regex, G1 the pass-flag check); 30 for the amended code (N1-N30); and, after
review 2, review 2's 24 (R2-M1 to R2-M24, adapted to the code of this round) and 23 for this round's code (N31-N53);
and 22 for fix round 3 after review 3 (R3-1 to R3-22: the yes/no question, the instruction, "and nothing else", the
witness-aware null reading, the authorization wording, the matched reading and the matched arm's witness).
Review 2's adapted copies of R1-A3, A5, A8 and G1 are the R1- entries themselves.

Equivalence arguments (review 1 marked B3, B5 and G1 as candidates; each now has a killing test, so none is argued
away): B3 cannot change the item set on this database, which has no product with fewer than two variants, so it is
caught by a unit test of claim_problems on a constructed database; B5 lets a gift card be proposed as the wrong
refund method, which the build then rejects as succeeding, and a test requires that no candidate is rejected for
any reason but a twin's second check; G1 removes a check that _finished already implies for real runs, and a test
sets verify.json's flag to false directly.
"""
from __future__ import annotations

import json
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
     '        if now.get(name) != pins[name]:', '        if False:'),
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
     'for job in jobs_for(t, scfg, arms)}', 'for job in jobs_for(t, scfg, arms) if job[3] != "swapped"}'),
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
    ("N4 mention-swapped names the target order", "wf/tau/items.py", '"task_mention_swapped": mention_line(w2, item2["name"] if item2 else None),',
     '"task_mention_swapped": mention_line(oid, product),'),
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
    ("N13 effect always read as override", "wf/tau/pipeline.py", '        if not disc:\n            out["reading"] = ("The task shifts',
     '        if False:\n            out["reading"] = ("The task shifts'),
    ("N14 discrimination on every witness", "wf/tau/pipeline.py", '    determined = [t for t in pop if t.get("witness_checker") == "PASS"]', '    determined = pop'),
    ("N15 one model counts as a replication", "wf/tau/pipeline.py", '    if rep == len(v):', '    if rep:'),
    ("N16 token gate disabled", "wf/tau/pipeline.py", '        if not report.get("first_token_ok"):', '        if False:'),
    ("N17 an unpinned file is accepted", "wf/tau/pipeline.py",
     '        if pins.get(name) is None:\n            return False, f"no pinned sha256 for {name}"', '        if pins.get(name) is None:\n            continue'),
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
    ("N30 discrimination compares the witness with itself", "wf/tau/pipeline.py", 'action="witness", action_b="fail")\n\n\ndef controls_block',
     'action="witness", action_b="witness")\n\n\ndef controls_block'),
    # ---- review 2's mutants (scratchpad review_tau2/my_mutants*.py), adapted to the code after fix round 2 ----
    ("R2-M1 rules: exchange status 'contains' instead of 'exactly'", "wf/tau/prompts.py",
     '- exchange_delivered_order_items: the order\'s status is exactly "delivered"', '- exchange_delivered_order_items: the order\'s status contains "delivered"'),
    ("R2-M2 rules: modify-items availability dropped", "wf/tau/prompts.py",
     'is a variant of the same product, and is available;', 'is a variant of the same product;'),
    ("R2-M3 rules: 'does not see the task' dropped (main lead)", "wf/tau/prompts.py",
     'current database: it does not see the user\'s messages or task, and it does not check', 'current database, and it does not check'),
    ("R2-M4 rules section after the policy", "wf/tau/prompts.py",
     'return f"{INSTRUCTION}\\n\\n{backend}\\n\\n{policy}\\n\\n# Tools\\n\\n{tool_lines}"',
     'return f"{INSTRUCTION}\\n\\n{policy}\\n\\n{backend}\\n\\n# Tools\\n\\n{tool_lines}"'),
    ("R2-M5 rules: return gift-card exception dropped", "wf/tau/prompts.py",
     'payment method (the first entry of its payment_history) or a gift card;', 'payment method (the first entry of its payment_history);'),
    ("R2-M6 few-shot other-success names its own order", "wf/tau/items.py",
     'task_line("return_delivered_order_items", w2, db["orders"][w2]["items"][0]["name"])',
     'task_line("return_delivered_order_items", oid, db["orders"][oid]["items"][0]["name"])'),
    ("R2-M7 witness scored under aligned only", "wf/tau/pipeline.py",
     'jobs += [("main", "witness", r, c) for r in readouts for c in conds]', 'jobs += [("main", "witness", r, "aligned") for r in readouts]'),
    ("R2-M8 mention pair: first token only", "wf/tau/pipeline.py",
     'jobs += [("main", "fail", r, c) for r in single for c in MENTION_CONDITIONS', 'jobs += [("main", "fail", r, c) for r in single[:1] for c in MENTION_CONDITIONS'),
    ("R2-M9 manipulation contrast reversed", "wf/tau/pipeline.py",
     'out["manipulation"] = paired(by, pop, "target", "aligned", "swapped")', 'out["manipulation"] = paired(by, pop, "target", "swapped", "aligned")'),
    ("R2-M10 check passes without CI above 0", "wf/tau/pipeline.py", 'est >= CHECK_MIN_NATS and ci[0] > 0', 'est >= CHECK_MIN_NATS'),
    ("R2-M11 check threshold 0.5 nats", "wf/tau/pipeline.py", 'CHECK_MIN_NATS = 1.0', 'CHECK_MIN_NATS = 0.5'),
    ("R2-M12 mixed direction never reported", "wf/tau/pipeline.py", '    if (rep and (rev or neg)) or (rev and pos):\n', '    if False:\n'),
    ("R2-M13 joint null needs one model only", "wf/tau/pipeline.py", '    if null == len(v):', '    if null:'),
    ("R2-M14 manipulation gate reads discrimination", "wf/tau/pipeline.py",
     '        if v.startswith("no effect") and not manip:', '        if v.startswith("no effect") and not disc:'),
    ("R2-M15 discrimination under the aligned task", "wf/tau/pipeline.py",
     'return paired(by, items, readout, "neutral", "neutral", action="witness"', 'return paired(by, items, readout, "aligned", "aligned", action="witness"'),
    ("R2-M16 analysis ok with missing scores", "wf/tau/pipeline.py", '    ok = missing == 0 and stale == 0', '    ok = stale == 0'),
    ("R2-M17 exploratory arms drawn per shard", "wf/tau/pipeline.py",
     '    arms = extra_arm_ids(every, cfg)\n    items = shard_by_user(every, i, n)', '    items = shard_by_user(every, i, n)\n    arms = extra_arm_ids(items, cfg)'),
    ("R2-M18 item hash ignores witness fields", "wf/tau/pipeline.py", '"aligned_witness", "witness_output", "witness_failure")}', ')}'),
    ("R2-M19 projection: one model's hours only", "wf/tau/pipeline.py", 'out["hours_max_shard_both_models"] = sum(', 'out["hours_max_shard_both_models"] = max('),
    ("R2-M20 witness full pair reversed", "wf/tau/pipeline.py",
     'else (t["witness_output"], t["witness_failure"]))', 'else (t["witness_failure"], t["witness_output"]))'),
    ("R2-M21 replicates needs only CI, not Holm p", "wf/tau/pipeline.py", '    if excludes_zero and holm_p < P_MAX:', '    if excludes_zero:'),
    ("R2-M22 manipulation question changed", "wf/tau/prompts.py",
     'TARGET_QUESTION = ("Question: is this call on the order the user asked about?', 'TARGET_QUESTION = ("Question: is this call on an order of the user?'),
    ("R2-M23 write-minus-mention uses mention_aligned minus neutral", "wf/tau/pipeline.py",
     '("fail", "mention_aligned", "mention_swapped")) for r in ("first", "yn")}', '("fail", "mention_aligned", "neutral")) for r in ("first", "yn")}'),
    ("R2-M24 target check scheduled on neutral instead of swapped", "wf/tau/pipeline.py",
     'jobs += [("main", "fail", "target", c) for c in conds if c != "neutral"]', 'jobs += [("main", "fail", "target", c) for c in conds if c != "swapped"]'),
    # ---- fix round 2 (matched arm, counterbalancing, outcome table, smoke, gates) ----
    ("N31 matched arm keeps the other-order few-shot tasks", "wf/tau/prompts.py",
     'if arm == "matched" else list(fewshot)', 'if False else list(fewshot)'),
    ("N32 few-shot order never reversed", "wf/tau/prompts.py", '    return exs[::-1] if fewshot_reversed(item["id"]) else exs', '    return exs'),
    ("N33 few-shot order hashed with another salt", "wf/tau/prompts.py", 'f"{SEED}|{item_id}|fewshot-order"', 'f"{SEED}|{item_id}|fewshot"'),
    ("N34 matched arm never scheduled", "wf/tau/pipeline.py",
     'EXTRA_ARMS = {"rules_only": "rules_arm_items", "matched": "matched_arm_items"}', 'EXTRA_ARMS = {"rules_only": "rules_arm_items"}'),
    ("N35 a config without pins passes", "wf/tau/pipeline.py",
     '    if not pins:  # a scoring config without pins', '    if False:  # a scoring config without pins'),
    ("N36 smoke analysis computes estimates", "wf/tau/pipeline.py",
     '    if cfg.get("project_to"):\n        # the smoke run', '    if False:\n        # the smoke run'),
    ("N37 projection from medians", "wf/tau/pipeline.py",
     '        sec = blk["format"]["seconds_mean_by_readout"]', '        sec = blk["format"]["seconds_median_by_readout"]'),
    ("N38 smoke set without the longest items", "wf/tau/pipeline.py", '        if cfg.get("add_longest"):', '        if False:'),
    ("N39 a reversal is not gated on discrimination", "wf/tau/pipeline.py",
     '        elif v.startswith(("no effect", "reversed")) and not disc:', '        elif v.startswith("no effect") and not disc:'),
    ("N40 specificity read the wrong way", "wf/tau/pipeline.py",
     '        elif _ci_above(fail_minus_witness):', '        elif not _ci_above(fail_minus_witness):'),
    ("N41 size met by one without the other's direction", "wf/tau/pipeline.py", '    if rep and rep + pos == len(v):', '    if rep:'),
    ("N42 matched reading for a main null inverted", "wf/tau/pipeline.py",
     '    if null(main_verdict) and eff(matched_verdict):', '    if eff(main_verdict) and null(matched_verdict):'),
    ("N43 per-class discrimination on every witness", "wf/tau/pipeline.py",
     '        det = [t for t in v if t.get("witness_checker") == "PASS"]', '        det = v'),
    ("N44 sensitivity analysis excludes nothing", "wf/tau/pipeline.py",
     'FEWSHOT_TOOL_CLASSES = ("wrong_return_pm", "gc_balance")', 'FEWSHOT_TOOL_CLASSES = ()'),
    ("N45 mention line without the product", "wf/tau/items.py",
     '    return MENTION_PRODUCT.format(o=order_id, p=product) if product else MENTION.format(o=order_id)', '    return MENTION.format(o=order_id)'),
    ("N46 verifier skips the own-order task", "wf/tau/verify.py",
     '        if ex.get("task_own") != I.task_line(', '        if False and ex.get("task_own") != I.task_line('),
    ("N47 verifier skips the order balance", "wf/tau/verify.py", '            if not 0.45 <= share <= 0.55:', '            if False:'),
    ("N48 matched lead keeps 'sees only'", "wf/tau/prompts.py",
     '"matched": ("The policy below is written for the agent. The backend does not check',
     '"matched": ("The policy below is written for the agent. The backend sees only each call\'s name and arguments and the current database, and it does not check'),
    ("N49 rules: equal-length check dropped", "wf/tau/prompts.py",
     'every old item is in the order; there are as many new items as old ones; each new item is a variant',
     'every old item is in the order; each new item is a variant'),
    ("N50 rules: payment entry is any transaction", "wf/tau/prompts.py",
     "the order's payment_history has exactly one entry, and it is a payment;", "the order has exactly one payment;"),
    ("N51 precision ignores clustering", "wf/tau/pipeline.py",
     '    lo, hi = sd / math.sqrt(n), sd * math.sqrt(m / n)', '    lo, hi = sd / math.sqrt(n), sd / math.sqrt(n)'),
    ("N53 sensitivity analysis compared without Holm", "wf/tau/pipeline.py",
     '            v = tau_verdict(sens[m]["estimate"], sens[m]["ci"], adj_s[m])', '            v = tau_verdict(sens[m]["estimate"], sens[m]["ci"], sens[m]["perm_p"])'),
    ("N52 arm contrast ignores the arms", "wf/tau/pipeline.py",
     'for arm, (act, a, b) in zip(arms, (first, second))', 'for arm, (act, a, b) in zip(("main", "main"), (first, second))'),
    # ---- fix round 3 (review 3 and the task-invariance gate) ----
    ("R3-1 yes/no question asks whether the call succeeds", "wf/tau/prompts.py",
     'YN_QUESTION = ("Question: will the backend execute this call without returning an error? Answer Yes if its output is "\n               "JSON and No if its output is a line starting with \\"Error: \\".\\nAnswer:")',
     'YN_QUESTION = ("Question: will this tool call succeed, or will it return an error? Answer Yes if it succeeds and No "\n               "if it returns an error.\\nAnswer:")'),
    ("R3-2 yes/no question without the output shapes", "wf/tau/prompts.py",
     'Answer Yes if its output is "\n               "JSON and No if its output is a line starting with \\"Error: \\".\\nAnswer:")',
     'Answer Yes if it does "\n               "and No if it does not.\\nAnswer:")'),
    ("R3-3 instruction gives the task as an input to the prediction", "wf/tau/prompts.py",
     '"calls the backend\'s tools. Each turn shows the user\'s task,', '"calls the backend\'s tools. Given the user\'s task,'),
    ("R3-4 instruction predicts 'the output of the tool call'", "wf/tau/prompts.py",
     'Predict the exact output the backend returns for that call.', 'Predict the exact output of the tool call.'),
    ("R3-5 main lead without 'and nothing else'", "wf/tau/prompts.py",
     '"(authentication, confirmation with the user). It does check the following and nothing else: a write call "',
     '"(authentication, confirmation with the user). It does check the following: a write call "'),
    ("R3-6 matched lead without 'and nothing else' (arms differ in more than the lead)", "wf/tau/prompts.py",
     '"(authentication, confirmation with the user). It does check the following and nothing else: a write "',
     '"(authentication, confirmation with the user). It does check the following: a write "'),
    ("R3-7 a null ignores the witness", "wf/tau/pipeline.py",
     '        out["reading"] = (WITNESS_SHIFT_NULL if _ci_above(witness_effect) else',
     '        out["reading"] = (WITNESS_SHIFT_NULL if False else'),
    ("R3-8 a null reads any witness block as a shift", "wf/tau/pipeline.py",
     '        out["reading"] = (WITNESS_SHIFT_NULL if _ci_above(witness_effect) else',
     '        out["reading"] = (WITNESS_SHIFT_NULL if witness_effect is not None else'),
    ("R3-9 the analysis passes no witness effect", "wf/tau/pipeline.py",
     '            "witness_effect": ctl["witness"]["first"]["aligned_minus_swapped"]}',
     '            "witness_effect": None}'),
    ("R3-10 the analysis passes the witness's aligned minus neutral", "wf/tau/pipeline.py",
     '            "witness_effect": ctl["witness"]["first"]["aligned_minus_swapped"]}',
     '            "witness_effect": ctl["witness"]["first"]["aligned_minus_neutral"]}'),
    ("R3-11 verdicts drop the witness", "wf/tau/pipeline.py",
     'c.get("write_minus_mention"), c.get("witness_effect"))', 'c.get("write_minus_mention"))'),
    ("R3-12 joint null without the witness note", "wf/tau/pipeline.py",
     '    if overall.startswith("no effect") and shifted:', '    if False:'),
    ("R3-13 null witness reading reworded", "wf/tau/pipeline.py",
     'but the task shifts predictions for working "\n                      "calls (an authorization or task-consistency reading)")',
     'but the task shifts predictions for working "\n                      "calls")'),
    ("R3-14 task-consistency row without the authorization belief", "wf/tau/pipeline.py",
     '"A general task-consistency bias, or a belief that the backend checks calls against the "\n                              "request: naming',
     '"A general task-consistency bias: naming'),
    ("R3-15 write minus mention without the authorization note", "wf/tau/pipeline.py",
     '(write minus mention above 0; an authorization "\n                          "reading predicts this too)"',
     '(write minus mention above 0)"'),
    ("R3-16 matched reading for a main null: the old 'suppresses' text", "wf/tau/pipeline.py",
     '        return MATCHED_NULL_MAIN_EFFECT',
     '        return ("null under the main prompt but an effect under the matched prompt: an explicit statement that the "\n                "backend does not see the task suppresses the effect here")'),
    ("R3-17 matched witness never scheduled", "wf/tau/pipeline.py",
     '            if arm == "matched" and cfg.get("matched_witness", True)', '            if False and cfg.get("matched_witness", True)'),
    ("R3-18 matched witness under the aligned task only", "wf/tau/pipeline.py",
     '                jobs += [("matched", "witness", "first", c) for c in conds]', '                jobs += [("matched", "witness", "first", "aligned")]'),
    ("R3-19 matched witness scored in the main arm", "wf/tau/pipeline.py",
     '                jobs += [("matched", "witness", "first", c) for c in conds]', '                jobs += [("main", "witness", "first", c) for c in conds]'),
    ("R3-20 matched witness also in the rules-only arm", "wf/tau/pipeline.py",
     '            if arm == "matched" and cfg.get("matched_witness", True)', '            if cfg.get("matched_witness", True)'),
    ("R3-21 matched witness readout reads the main arm", "wf/tau/pipeline.py",
     '"matched": effects(by, items, "first", action="witness", arm="matched")', '"matched": effects(by, items, "first", action="witness")'),
    ("R3-22 matched fail minus witness computed in the main arm", "wf/tau/pipeline.py",
     'contrast_of_effects(by, items, "first", fail, wit, arms=("matched", "matched"))', 'contrast_of_effects(by, items, "first", fail, wit)'),
]


def _run(python: str, work: Path, timeout: int) -> tuple[str, str]:
    """(outcome, the first failing test's id and error line); never silent about what killed a mutant."""
    try:
        r = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=line", "-rfE", *TESTS],
                           cwd=work, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    if r.returncode == 0:
        return "survived", ""
    lines = r.stdout.splitlines()
    test = next((x.split(" ", 1)[1].split(" - ")[0] for x in lines if x.startswith(("FAILED ", "ERROR "))), "")
    err = next((x.strip() for x in lines if x.startswith("E ") or ": assert" in x or "Error" in x), "")
    if not test:
        tail = (r.stdout + r.stderr).strip().splitlines()
        test = "no test id (collection or import error)"
        err = err or (tail[-1] if tail else "")
    return "killed", f"{test} | {err[:200]}"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--out")]
    out_path = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    shard, shards = map(int, args[0].split("/")) if args else (0, 1)
    chosen = MUTANTS[shard::shards]
    only = os.environ.get("MUTANTS_MATCHING")  # "a|b": only mutants whose name contains a or b
    if only:
        chosen = [m for m in chosen if any(o in m[0] for o in only.split("|"))]
    timeout = int(os.environ.get("MUTATION_TIMEOUT", "900"))
    results: dict[str, list] = {"killed": [], "survived": [], "timeout": [], "not applied": []}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT, work, ignore=lambda folder, names: [n for n in names if n in (".git", ".venv", "runs", "runs_mock", "__pycache__", ".pytest_cache", "pace.env")])
        base, why = _run(sys.executable, work, timeout)
        if base != "survived":
            print(f"BASELINE FAILS ({base}): {why}; no mutant is run", flush=True)
            return 2
        print("baseline: the unmutated copy passes", flush=True)
        for name, rel, good, bad in chosen:
            path = work / rel
            src = path.read_text(encoding="utf-8")
            if src.count(good) != 1:
                results["not applied"].append(name)
                print(f"NOT APPLIED  {name}", flush=True)
                continue
            path.write_text(src.replace(good, bad), encoding="utf-8")
            outcome, by = _run(sys.executable, work, timeout)
            path.write_text(src, encoding="utf-8")
            results[outcome].append([name, by] if outcome == "killed" else name)
            print(f"{outcome:<12} {name}" + (f"  <- {by}" if by else ""), flush=True)
    print(f"\n{len(results['killed'])}/{len(chosen)} killed; survived={results['survived']}; "
          f"timeout={results['timeout']}; not applied={results['not applied']}")
    if out_path:
        Path(out_path).write_text(json.dumps({"shard": f"{shard}/{shards}", "n": len(chosen), **results}, indent=1))
    return 0 if len(results["killed"]) == len(chosen) else 1


if __name__ == "__main__":
    sys.exit(main())
