"""Full verification of the τ-bench items (notes/tau_design.md §8): every item is replayed in a fresh environment
and read by the text-only checker; scoring refuses a file that has not passed.

Per item:
- the history regenerates byte for byte, the failure reply is identical, and the failing write leaves the full
  database hash unchanged;
- the success twin regenerates (history replayed without bypass, then only the class's check bypassed);
- the real error belongs to the item's class, and the text-only checker predicts FAIL, that class and the
  identical error text;
- the achievability flags are re-derived from the witness search, and each recorded witness succeeds;
- the task lines are the templates' (aligned, swapped, neutral), and none asserts a false fact;
- the three prompts differ only in the task line;
- generated histories contain only the design's reads, with both orders shown;
- the item id and key are its (history, action) hash, and no two items share them;
- no item comes from a few-shot user.
At the end the pristine database's hash must be unchanged. `python -m wf.tau.verify --fresh <items> <ids>`
replays the listed items in a new process that reloads the database from disk (run for a seeded sample).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from wf.tau import items as I
from wf.tau.checker import check, error_class
from wf.tau.prompts import CONDITIONS, build_messages
from wf.tau.retail import base_db, fresh_env, pristine_sha256

REPO = Path(__file__).resolve().parents[2]


def replay_problems(it: dict) -> list[str]:
    """History, failure reply, unchanged database and twin, in a fresh environment."""
    out = []
    env = fresh_env()
    got = env.replay(it["history"])
    if got != [x["output"] for x in it["history"]]:
        bad = next(i for i, (a, b) in enumerate(zip(got, [x["output"] for x in it["history"]])) if a != b)
        out.append(f"history output {bad} differs on replay")
    before = env.state.sha256()
    fail = env.call(it["action"]["name"], it["action"]["arguments"])
    if fail != it["failure_reply"]:
        out.append("failure reply differs on replay")
    if env.state.sha256() != before:
        out.append("failing write changed the database")
    if I.make_twin(it["history"], it["action"], it["cls"]) != it["success_twin"]:
        out.append("success twin differs on replay")
    if it["success_twin"].startswith("Error") or not it["success_twin"].startswith('{"'):
        out.append("success twin is not a JSON success")
    if not it["failure_reply"].startswith("Error: "):
        out.append("failure reply is not an error")
    return out


def item_problems(it: dict, db: dict, fewshot: list[dict], fewshot_users: set[str]) -> list[str]:
    out = replay_problems(it)
    if error_class(it["failure_reply"]) != it["cls"]:
        out.append(f"error class {error_class(it['failure_reply'])} != {it['cls']}")
    v = check(it["history"], it["action"])
    if (v.verdict, v.cls, v.error) != ("FAIL", it["cls"], it["failure_reply"]):
        out.append(f"checker says {v.verdict}/{v.cls}/{v.error!r} ({v.evidence})")
    tool = it["action"]["name"]
    # achievability, re-derived on the history's state
    env = fresh_env()
    env.replay(it["history"])
    kind = I.KIND[tool]
    w_al = I.achievable(db, env, kind, it["order_id"], it.get("item_id"))
    if (w_al is not None) != it["aligned_achievable"] or w_al != it.get("aligned_witness") or it["main_pool"] != (w_al is not None):
        out.append("aligned achievability does not re-derive")
    if it.get("comparison_order_id"):
        w_sw = I.achievable(db, env, kind, it["comparison_order_id"], it.get("comparison_item_id"))
        if (w_sw is not None) != it["swapped_achievable"] or w_sw != it.get("swapped_witness"):
            out.append("swapped achievability does not re-derive")
    for w in (it.get("aligned_witness"), it.get("swapped_witness")):
        if w and env.fork().call(w["name"], w["arguments"]).startswith("Error"):
            out.append("a recorded witness fails")
    # task lines
    if it["task_aligned"] != I.task_line(tool, it["order_id"], it.get("product")):
        out.append("aligned task line is not the template")
    if it.get("comparison_order_id") and it["task_swapped"] != I.task_line(tool, it["comparison_order_id"], it.get("comparison_product")):
        out.append("swapped task line is not the template")
    if it["task_neutral"] != I.NEUTRAL:
        out.append("neutral task line is not the fixed one")
    for order, product in ((it["order_id"], it.get("product")), (it.get("comparison_order_id"), it.get("comparison_product"))):
        if order and I.claim_problems(db, it["user_id"], tool, order, product):
            out.append(f"task line for {order} asserts a false fact: {I.claim_problems(db, it['user_id'], tool, order, product)}")
    # prompts differ only in the task line
    conds = [c for c in CONDITIONS if c != "swapped" or it.get("task_swapped")]
    for yn in (False, True):
        ms = {c: build_messages(it, c, fewshot, yes_no=yn) for c in conds}
        base = ms["aligned"]
        for c in conds[1:]:
            other = ms[c]
            line = it[f"task_{c}"]
            if (other[:-1] != base[:-1] or other[-1]["content"].replace(f"Task: {line}\n", f"Task: {it['task_aligned']}\n", 1)
                    != base[-1]["content"]):
                out.append(f"{c} prompt differs from aligned outside the task line")
    # generated histories: exactly the design's reads
    if it["source"] == "generated":
        names = [x["name"] for x in it["history"]]
        orders = [x["arguments"].get("order_id") for x in it["history"] if x["name"] == "get_order_details"]
        want = ["get_user_details", "get_order_details", "get_order_details"]
        if names[:3] != want or any(n != "get_product_details" for n in names[3:]) or sorted(orders) != sorted([it["order_id"], it["comparison_order_id"]]):
            out.append("generated history is not the design's reads")
        if (it["cls"] == "unavailable") != bool(names[3:]):
            out.append("product details shown for the wrong class")
        if (orders[0] == it["order_id"]) != it["target_first"]:
            out.append("target_first does not match the history")
    if it["key"] != I.item_key(it["history"], it["action"]) or not it["id"].endswith(it["key"][:12]):
        out.append("id or key is not the (history, action) hash")
    if it["user_id"] in fewshot_users:
        out.append("item from a few-shot user")
    return out


def fewshot_problems(fewshot: list[dict]) -> list[str]:
    out = []
    if len(fewshot) != 3 or len({ex["user_id"] for ex in fewshot}) != 3:
        out.append("few-shot examples must be three, from three users")
    for ex in fewshot:
        env = fresh_env()
        if env.replay(ex["history"]) != [x["output"] for x in ex["history"]]:
            out.append(f"few-shot history for {ex['user_id']} differs on replay")
        if env.call(ex["action"]["name"], ex["action"]["arguments"]) != ex["output"]:
            out.append(f"few-shot output for {ex['user_id']} differs on replay")
    return out


def verify_sets(generated: list[dict], natural: list[dict], fewshot: list[dict]) -> dict[str, Any]:
    db = base_db()
    before = pristine_sha256()
    fewshot_users = {ex["user_id"] for ex in fewshot}
    failures: dict[str, list[str]] = {}
    problems = fewshot_problems(fewshot)
    keys: dict[str, str] = {}
    for it in generated + natural:
        p = item_problems(it, db, fewshot, fewshot_users)
        if it["key"] in keys:
            p.append(f"duplicate (history, action) of {keys[it['key']]}")
        keys.setdefault(it["key"], it["id"])
        if p:
            failures[it["id"]] = p
    if pristine_sha256() != before:
        problems.append("the pristine database changed during verification")
    return {"n_generated": len(generated), "n_natural": len(natural), "n_failed_items": len(failures),
            "failures": dict(list(failures.items())[:50]), "problems": problems,
            "passed": not failures and not problems and bool(generated)}


def fresh_process_sample(items_path: Path, ids: list[str]) -> dict[str, Any]:
    """Replay `ids` in a new Python process that parses the database from disk."""
    proc = subprocess.run([sys.executable, "-m", "wf.tau.verify", "--fresh", str(items_path), *ids], cwd=REPO,
                          capture_output=True, text=True, timeout=3600)
    try:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"passed": False, "error": proc.stderr[-2000:]}
    return res


def sample_ids(items: list[dict], n: int = 100) -> list[str]:
    ids = sorted(it["id"] for it in items)
    return sorted(random.Random(I.SEED).sample(ids, min(n, len(ids))))


def _fresh_main(items_path: str, ids: list[str]) -> None:
    from wf.tau.retail import RetailEnv, State, load_db_from_disk

    wanted = set(ids)
    items = [json.loads(line) for line in Path(items_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = []
    for it in (x for x in items if x["id"] in wanted):
        env = RetailEnv(State(load_db_from_disk()).fork())
        ok = env.replay(it["history"]) == [x["output"] for x in it["history"]]
        ok &= env.call(it["action"]["name"], it["action"]["arguments"]) == it["failure_reply"]
        twin_env = RetailEnv(State(load_db_from_disk()).fork())
        twin_env.replay(it["history"])
        ok &= twin_env.call_bypassed(it["action"]["name"], it["action"]["arguments"], frozenset({it["cls"]})) == it["success_twin"]
        if not ok:
            bad.append(it["id"])
    found = len([x for x in items if x["id"] in wanted])
    print(json.dumps({"checked": found, "requested": len(wanted), "failed": bad, "passed": not bad and found == len(wanted)}))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--fresh":
        _fresh_main(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit("usage: python -m wf.tau.verify --fresh <items.jsonl> <id> ...")
