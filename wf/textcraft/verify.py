"""Safeguards for the TextCraft items (notes/textcraft_design.md, section 7), and the manifest they produce.

    python -m wf.textcraft.verify            # every check; writes wf/textcraft/data/manifest.json
    python -m wf.textcraft.verify --replay   # (internal) full replay + checker in this process, JSON to stdout

The manifest is marked verified only when every check passes on the exact files it hashes; the scoring run
refuses anything else (wf/textcraft/score.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from wf.textcraft import build as B
from wf.textcraft import checker as K
from wf.textcraft import prompts as P
from wf.textcraft import world as W
from wf.textcraft.freeze import FROZEN, UPSTREAM_COMMIT

DATA = B.DATA
MANIFEST = DATA / "manifest.json"
UPSTREAM_CHECK = DATA / "upstream_check.json"
REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_items(path: Path = B.ITEMS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------- per-item checks ----------------


def replay_problems(item: dict) -> list[str]:
    """Rebuild the game from the stored list and goal; history, failure and twin must reproduce exactly."""
    problems = []
    actions = [a for a, _ in item["history"]]
    env, history = W.replay(item["recipe_lines"], item["env_goal"], actions)
    if history != item["history"]:
        problems.append("history does not replay")
    if item["env_goal"] != W.item_id(item["goal"]):
        problems.append("env goal is not the aligned goal")
    _env, other = W.replay(item["recipe_lines"], W.item_id(item["partner"]), actions)
    if other != item["history"]:
        problems.append("history depends on which goal is real")
    if W.step(W.clone(env), item["action"]) != item["reply"]:
        problems.append("failure reply does not reproduce")
    if item["action_kind"] == "craft":
        twin_env = W.clone(env)
        for name, n in item["topup"].items():
            twin_env.inventory[W.item_id(name)] = twin_env.inventory.get(W.item_id(name), 0) + n
        if W.step(twin_env, item["action"]) != item["twin"]:
            problems.append("twin does not reproduce")
        if item["twin_source"] != "env_topup":
            problems.append("craft twin not from the environment")
    elif item["twin"] != f"Got 1 {item['object']}" or item["twin_source"] != "template":
        problems.append("get twin is not the template")
    goal_ids = {W.item_id(item["goal"]), W.item_id(item["partner"])}
    if any(o.startswith("Crafted ") and o.split(" ", 2)[2] in goal_ids for _a, o in item["history"]):
        problems.append("history crafts a goal")
    return problems


def structure_problems(item: dict) -> list[str]:
    """Population, bin, role and task-line rules that the design fixes."""
    problems = []
    t = item["history_len"]
    if t != len(item["history"]) or item["length_bin"] != B.length_bin(t):
        problems.append("history length or bin wrong")
    want = ("secondary_get" if item["action_kind"] == "get" else "main" if t >= 1 else "empty_history")
    if item["population"] != want:
        problems.append(f"population {item['population']} should be {want}")
    if item["task_line"] != f"Goal: craft {item['goal']}." or item["swapped_task_line"] != f"Goal: craft {item['partner']}.":
        problems.append("task lines wrong")
    if item["neutral_task_line"] != B.NEUTRAL_LINE:
        problems.append("neutral line wrong")
    goal_tree, partner_tree = W.tree_craftables(W.item_id(item["goal"])), W.tree_craftables(W.item_id(item["partner"]))
    if W.item_id(item["object"]) not in goal_tree or W.item_id(item["object"]) in partner_tree:
        problems.append("object not in the aligned goal's tree only")
    if (item["role"] == "goal") != (item["object"] == item["goal"]):
        problems.append("role wrong")
    if item["candidates"] != {"o_plus": item["twin"], "o_minus": item["reply"]}:
        problems.append("candidates are not twin and reply")
    return problems


def prompt_problems(item: dict, shots) -> list[str]:
    """Aligned, swapped and neutral prompts differ only in the goal line; questions only in the ending."""
    problems = []
    base = P.messages(item, "aligned", shots)
    for arm in ("swapped", "neutral"):
        other = P.messages(item, arm, shots)
        if other[:-1] != base[:-1]:
            problems.append(f"{arm}: examples or system differ")
        if other[-1]["content"].replace(P.task_line(item, arm), P.task_line(item, "aligned"), 1) != base[-1]["content"]:
            problems.append(f"{arm}: query differs beyond the goal line")
    for arm in P.CONDITIONS:
        plain, yn = P.messages(item, arm, shots), P.messages(item, f"yn_{arm}", shots)
        if yn[:-1] != plain[:-1] or yn[-1]["content"] != plain[-1]["content"][: -len(P.NEXT)] + P.YN_QUESTION:
            problems.append(f"yn_{arm}: differs beyond the question")
    return problems


def check_items(items: list[dict], shots) -> dict:
    """Every per-item check on every item; returns counts and the first problems found."""
    fails: dict[str, list] = {"replay": [], "checker": [], "structure": [], "prompt": []}
    for it in items:
        for name, probs in (("replay", replay_problems(it)), ("checker", K.check_item(it)),
                            ("structure", structure_problems(it)), ("prompt", prompt_problems(it, shots))):
            if probs:
                fails[name].append((it["id"], probs))
    keys = Counter((it["pair_id"], json.dumps(it["history"]), it["action"]) for it in items)
    ids = Counter(it["id"] for it in items)
    shot_goals = {g for ex in shots for p in B.all_pairs() if p.pair_id == ex["pair_id"] for g in p.goals}
    item_goals = {g for it in items for g in (it["goal"], it["partner"])}
    return {
        "n_items": len(items),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        **{f"{k}_failures": len(v) for k, v in fails.items()},
        "examples": {k: v[:5] for k, v in fails.items() if v},
        "duplicate_history_action": sum(n - 1 for n in keys.values() if n > 1),
        "duplicate_ids": sum(n - 1 for n in ids.values() if n > 1),
        "fewshot_goals_in_items": sorted(shot_goals & item_goals),
    }


def counts(items: list[dict]) -> dict:
    c = lambda f: dict(sorted(Counter("|".join(str(it[k]) for k in f) for it in items).items()))
    return {"population": c(("population",)), "population_class": c(("population", "failure_class")),
            "population_class_bin": c(("population", "failure_class", "length_bin")),
            "population_role": c(("population", "role")),
            "pairs": len({it["pair_id"] for it in items}),
            "goals": len({g for it in items for g in (it["goal"], it["partner"])})}


# ---------------- whole-run checks ----------------


def _fresh(args: list[str], seed: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, check=True)


def determinism(items_sha: str, fewshot_sha: str) -> dict:
    """Two fresh builds under different hash seeds must write the committed files byte for byte."""
    out = {}
    for seed in ("0", "1"):
        with tempfile.TemporaryDirectory() as tmp:
            _fresh(["-m", "wf.textcraft.build", tmp], seed)
            out[seed] = {"items": sha256(Path(tmp) / "items.jsonl") == items_sha,
                         "fewshot": sha256(Path(tmp) / "fewshot.json") == fewshot_sha}
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    items = read_items()
    shots = P.fewshot_examples()
    if argv[:1] == ["--replay"]:
        print(json.dumps(check_items(items, shots)))
        return 0
    items_sha, fewshot_sha = sha256(B.ITEMS), sha256(P.FEWSHOT)
    replays = {seed: json.loads(_fresh(["-m", "wf.textcraft.verify", "--replay"], seed).stdout) for seed in ("0", "1", "2")}
    det = determinism(items_sha, fewshot_sha)
    upstream = json.loads(UPSTREAM_CHECK.read_text()) if UPSTREAM_CHECK.exists() else None
    upstream_ok = bool(upstream and upstream.get("items_sha256") == items_sha and upstream.get("frozen_sha256") == sha256(FROZEN)
                       and upstream.get("mismatches") == 0 and upstream.get("tree_identical") is True)
    replay_ok = all(r["n_items"] == len(items) and not any(r[f"{k}_failures"] for k in ("replay", "checker", "structure", "prompt"))
                    and r["duplicate_history_action"] == 0 and r["duplicate_ids"] == 0 and not r["fewshot_goals_in_items"]
                    for r in replays.values())
    det_ok = all(all(v.values()) for v in det.values())
    manifest = {
        "items_file": str(B.ITEMS.relative_to(REPO)), "items_sha256": items_sha,
        "fewshot_file": str(P.FEWSHOT.relative_to(REPO)), "fewshot_sha256": fewshot_sha,
        "frozen_db_file": str(FROZEN.relative_to(REPO)), "frozen_db_sha256": sha256(FROZEN),
        "upstream_commit": UPSTREAM_COMMIT,
        "counts": counts(items),
        "checks": {"full_replay_and_checker_in_fresh_processes": replays, "build_determinism": det,
                   "upstream_equivalence": upstream},
        "verified": replay_ok and det_ok and upstream_ok,
        "why_not": [w for w, ok in (("replay/checker/structure/prompt/duplicates", replay_ok), ("determinism", det_ok),
                                    ("upstream equivalence (run wf/textcraft/upstream_check.py)", upstream_ok)) if not ok],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified": manifest["verified"], "why_not": manifest["why_not"], "counts": manifest["counts"]["population"]}))
    return 0 if manifest["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
