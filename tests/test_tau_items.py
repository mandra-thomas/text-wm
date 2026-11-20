"""The τ-bench item builder (wf/tau/items.py) and prompts (wf/tau/prompts.py), against notes/tau_design.md."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from wf.tau import items as I
from wf.tau.checker import check, error_class
from wf.tau.prompts import CONDITIONS, YN_QUESTION, build_messages, system_message
from wf.tau.retail import base_db, fresh_env
from wf.tau.verify import item_problems


@pytest.fixture(scope="module")
def built():
    return I.build_all()


def test_hash_is_the_designs():
    assert I.h("a", 1) == int(hashlib.sha256(b"20260924|a|1").hexdigest()[:16], 16)


def test_every_item_fails_as_its_class_says_and_has_a_success_twin(built):
    gen = built["generated"]
    assert len(gen) > 3000
    for it in gen:
        assert it["failure_reply"].startswith("Error: ") and error_class(it["failure_reply"]) == it["cls"]
        assert it["success_twin"].startswith('{"order_id"')
        v = check(it["history"], it["action"])
        assert (v.verdict, v.cls, v.error) == ("FAIL", it["cls"], it["failure_reply"])
    assert set(Counter(t["cls"] for t in gen)) == {"status", "unavailable", "wrong_return_pm", "gc_balance"}


def test_items_whose_twin_fails_a_second_check_are_rejected(built):
    log = built["log"]["generated"]
    assert log.get("rejected status: second check fails (gc_balance)", 0) > 0
    assert log.get("rejected unavailable: second check fails (gc_balance)", 0) > 0
    db = base_db()
    for it in built["generated"]:
        pm = it["action"]["arguments"].get("payment_method_id")
        if it["cls"] == "status" and pm and db["users"][it["user_id"]]["payment_methods"][pm]["source"] == "gift_card":
            assert not I.make_twin(it["history"], it["action"], it["cls"]).startswith("Error")


def test_twin_replays_the_history_without_the_bypass():
    """A failed write in the history must stay failed when the twin is made: only the action is bypassed."""
    o = next(o for _, o in sorted(base_db()["orders"].items()) if o["status"] == "delivered")
    env = fresh_env()
    cancel = {"name": "cancel_pending_order", "arguments": {"order_id": o["order_id"], "reason": "no longer needed"}}
    history = [{**cancel, "output": env.call(cancel["name"], cancel["arguments"])}]
    assert history[0]["output"] == "Error: Non-pending order cannot be cancelled"
    action = {"name": "modify_pending_order_address", "arguments": {"order_id": o["order_id"], "address1": "1 Main St",
              "address2": "", "city": "X", "state": "CA", "country": "USA", "zip": "00000"}}
    twin = json.loads(I.make_twin(history, action, "status"))
    assert twin["status"] == "delivered" and twin["cancel_reason"] is None and twin["address"]["address1"] == "1 Main St"


def test_histories_are_the_designs_reads_counterbalanced(built):
    firsts = Counter()
    for it in built["generated"]:
        names = [x["name"] for x in it["history"]]
        assert names[:3] == ["get_user_details", "get_order_details", "get_order_details"]
        assert all(n == "get_product_details" for n in names[3:]) and (it["cls"] == "unavailable") == bool(names[3:])
        shown = [x["arguments"]["order_id"] for x in it["history"] if x["name"] == "get_order_details"]
        assert shown == ([it["order_id"], it["comparison_order_id"]] if it["target_first"] else [it["comparison_order_id"], it["order_id"]])
        firsts[it["target_first"]] += 1
    assert 0.4 < firsts[True] / sum(firsts.values()) < 0.6


def test_comparison_order_is_status_matched_whenever_the_user_has_one(built):
    db = base_db()
    for it in built["generated"]:
        others = [x for x in db["users"][it["user_id"]]["orders"] if x != it["order_id"]]
        same = [x for x in others if db["orders"][x]["status"] == it["target_status"]]
        if same and "{p}" not in I.TEMPLATES[it["action"]["name"]]:
            assert it["status_matched"]
        assert it["status_matched"] == (it["target_status"] == it["comparison_status"])


def test_task_lines_are_the_templates_and_assert_nothing_false(built):
    db = base_db()
    for it in built["generated"]:
        tool = it["action"]["name"]
        assert it["task_aligned"] == I.task_line(tool, it["order_id"], it["product"])
        assert it["task_swapped"] == I.task_line(tool, it["comparison_order_id"], it["comparison_product"])
        assert it["task_neutral"] == "The user wants help with their account."
        assert not I.claim_problems(db, it["user_id"], tool, it["order_id"], it["product"])
        assert not I.claim_problems(db, it["user_id"], tool, it["comparison_order_id"], it["comparison_product"])
    uid = sorted(db["users"])[0]
    other_users_order = next(o for o in db["orders"] if db["orders"][o]["user_id"] != uid)
    assert I.claim_problems(db, uid, "cancel_pending_order", other_users_order, None) == ["order is not the user's"]
    oid = db["users"][uid]["orders"][0]
    assert "product not in order" in I.claim_problems(db, uid, "return_delivered_order_items", oid, "No Such Product")


def test_main_population_is_exactly_the_items_whose_aligned_task_is_achievable(built):
    gen = built["generated"]
    main = [t for t in gen if t["main_pool"]]
    assert all(t["aligned_achievable"] and t["aligned_witness"] for t in main)
    # cancelling, changing the address of, or returning from an order in the wrong status is never achievable
    for t in gen:
        if t["cls"] == "status" and t["action"]["name"] in ("cancel_pending_order", "modify_pending_order_address",
                                                            "return_delivered_order_items", "modify_pending_order_payment"):
            assert not t["main_pool"]
    status_main = Counter((t["action"]["name"], t["target_status"]) for t in main if t["cls"] == "status")
    assert set(status_main) == {("exchange_delivered_order_items", "pending"), ("modify_pending_order_items", "delivered")}
    assert 1200 < len(main) < 1600
    assert len({t["user_id"] for t in main}) > 250


def test_witnesses_succeed_and_unachievable_means_every_witness_fails(built):
    db = base_db()
    for it in built["generated"][::40]:
        env = fresh_env()
        env.replay(it["history"])
        kind = I.KIND[it["action"]["name"]]
        for order, item, flag, w in ((it["order_id"], it["item_id"], it["aligned_achievable"], it["aligned_witness"]),
                                     (it["comparison_order_id"], it["comparison_item_id"], it["swapped_achievable"], it["swapped_witness"])):
            calls = I.witness_calls(db, env, kind, order, item)
            results = [env.fork().call(c["name"], c["arguments"]) for c in calls]
            assert flag == any(not r.startswith("Error") for r in results)
            if w:
                assert not env.fork().call(w["name"], w["arguments"]).startswith("Error")


def test_no_duplicates_no_fewshot_users_and_ids_are_hashes(built):
    keys = [t["key"] for t in built["generated"]]
    assert len(keys) == len(set(keys))
    users = {ex["user_id"] for ex in built["fewshot"]}
    assert len(users) == 3 and not users & {t["user_id"] for t in built["generated"]}
    for t in built["generated"]:
        assert t["key"] == I.item_key(t["history"], t["action"]) and t["id"] == f"tau-g-{t['key'][:12]}"


def test_build_is_deterministic(built):
    again = I.build_all(max_users=60)
    first = [t for t in built["generated"] if t["user_id"] in {x["user_id"] for x in again["generated"]}]
    assert json.dumps(again["generated"]) == json.dumps(first)
    assert again["fewshot"] == built["fewshot"]


def test_fewshot_examples_are_real_outputs(built):
    few = built["fewshot"]
    assert [ex["output"].startswith("Error") for ex in few] == [False, True, False]
    assert few[1]["output"] == "Error: The new payment method should be different from the current one"
    for ex in few:
        env = fresh_env()
        assert env.replay(ex["history"]) == [h["output"] for h in ex["history"]]
        assert env.call(ex["action"]["name"], ex["action"]["arguments"]) == ex["output"]


def test_verifier_passes_a_sample_and_catches_planted_errors(built):
    db = base_db()
    few = built["fewshot"]
    users = {ex["user_id"] for ex in few}
    for it in built["generated"][::150]:
        assert item_problems(it, db, few, users) == []
    it = built["generated"][0]
    planted = {
        "history output": {**it, "history": [{**it["history"][0], "output": it["history"][0]["output"].replace('"', "'", 1)}] + it["history"][1:]},
        "failure reply": {**it, "failure_reply": it["failure_reply"] + "."},
        "success twin": {**it, "success_twin": it["success_twin"].replace("1", "2", 1)},
        "aligned task": {**it, "task_aligned": it["task_aligned"].replace("order", "the order")},
        "achievability": {**it, "aligned_achievable": not it["aligned_achievable"]},
        "few-shot user": it,
    }
    for name, bad in planted.items():
        found = item_problems(bad, db, few, users | ({it["user_id"]} if name == "few-shot user" else set()))
        assert found, name


# ---------------- prompts ----------------


def test_prompts_differ_only_in_the_task_line(built):
    few = built["fewshot"]
    for it in built["generated"][::200]:
        for yn in (False, True):
            ms = {c: build_messages(it, c, few, yes_no=yn) for c in CONDITIONS}
            for c in ("swapped", "neutral"):
                assert ms[c][:-1] == ms["aligned"][:-1]
                assert ms[c][-1]["content"].replace(it[f"task_{c}"], it["task_aligned"], 1) == ms["aligned"][-1]["content"]
                assert it[f"task_{c}"] in ms[c][-1]["content"] and it["task_aligned"] not in ms[c][-1]["content"]
            assert ms["aligned"][-1]["content"].endswith(YN_QUESTION if yn else "Tool output:")


def test_prompt_layout(built):
    it, few = built["generated"][0], built["fewshot"]
    ms = build_messages(it, "aligned", few)
    assert [m["role"] for m in ms] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    sysm = system_message()
    assert "predict the exact output of the tool call" in sysm and "# Retail agent policy" in sysm
    assert sysm.count('{"type": "function"') == 16
    q = ms[-1]["content"]
    assert q.startswith(f"Task: {it['task_aligned']}\n\nTool calls so far:\nCall: ")
    assert f"Next call: {json.dumps({'name': it['action']['name'], 'arguments': it['action']['arguments']})}\nTool output:" in q
    for h in it["history"]:
        assert f"Output: {h['output']}" in q
    assert ms[2]["content"] == few[0]["output"]
    with pytest.raises(ValueError):
        build_messages({**it, "task_swapped": None}, "swapped", few)
    with pytest.raises(ValueError):
        build_messages(it, "aligned", few[:2])


# ---------------- natural items ----------------


def _traj(events, sid="s1"):
    return {"file": "f.json", "simulation_id": sid, "task_id": "t", "trial": 0, "agent": "a", "events": events}


def _ev(env, name, args):
    return {"name": name, "arguments": args, "output": env.call(name, args)}


def _natural_case():
    db = base_db()
    uid = next(u for u in sorted(db["users"]) if sum(db["orders"][o]["status"] == "pending" for o in db["users"][u]["orders"]) >= 1
               and len(db["users"][u]["orders"]) >= 2)
    o = next(db["orders"][x] for x in db["users"][uid]["orders"] if db["orders"][x]["status"] == "pending")
    w2 = next(x for x in db["users"][uid]["orders"] if x != o["order_id"])
    it = o["items"][0]
    alt = next(k for k in db["products"][it["product_id"]]["variants"] if k != it["item_id"])
    env = fresh_env()
    reads = [_ev(env, "get_user_details", {"user_id": uid}), _ev(env, "get_order_details", {"order_id": o["order_id"]}),
             _ev(env, "get_order_details", {"order_id": w2})]
    bad = _ev(env, "exchange_delivered_order_items", {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": [alt],
                                                      "payment_method_id": o["payment_history"][0]["payment_method_id"]})
    return db, uid, reads, bad


def test_natural_items_keep_determined_failures_and_drop_echoes_mismatches_and_duplicates():
    db, uid, reads, bad = _natural_case()
    assert bad["output"] == "Error: Non-delivered order cannot be exchanged"
    trajs = [_traj(reads + [bad], "kept"), _traj(reads + [bad], "duplicate"),
             _traj(reads + [bad, bad], "echo"),
             _traj([{**reads[0], "output": reads[0]["output"].replace("a", "b", 1)}] + reads[1:] + [bad], "mismatch"),
             _traj(reads[:1] + [bad], "undetermined")]
    items, log = I.build_natural(db, trajs, set())
    assert len(items) == 1 and items[0]["provenance"]["simulation_id"] in ("kept", "duplicate")
    assert log["dropped: echo of an earlier failure on the same order"] == 1
    assert log["dropped: replay mismatch"] == 1
    assert log["dropped: not determined by the history"] == 1
    assert log["dropped: duplicate history and action"] >= 1
    it = items[0]
    assert it["task_swapped"] and it["comparison_order_id"] in [r["arguments"].get("order_id") for r in reads]
    assert it["main_pool"]  # swapping on a pending order is achievable with modify_pending_order_items
    none, log = I.build_natural(db, trajs, {uid})
    assert none == [] and log["dropped: few-shot user"] == 3  # kept, duplicate and the echo's first failure


def test_natural_dedupe_keeps_the_shortest_history():
    db, uid, reads, bad = _natural_case()
    longer = reads + [reads[0]]
    items, log = I.build_natural(db, [_traj(longer + [bad], "long"), _traj(reads + [bad], "short")], set())
    assert [t["provenance"]["simulation_id"] for t in items] == ["short"]
    assert log["dropped: same order, tool and error as a kept item"] == 1
