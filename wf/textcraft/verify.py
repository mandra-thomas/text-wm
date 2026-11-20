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
from wf.textcraft import fit as F
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
    success = item["failure_class"] == "success"
    if W.step(W.clone(env), item["action"]) != item["reply"]:
        problems.append("reply does not reproduce")
    if item["reply"].startswith(("Crafted ", "Got ")) != success:
        problems.append("reply and class disagree on success")
    if item["action_kind"] == "craft":
        twin_env = W.clone(env)
        for name, n in item["topup"].items():
            twin_env.inventory[W.item_id(name)] = twin_env.inventory.get(W.item_id(name), 0) + n
        for name, n in item["removal"].items():
            twin_env.inventory[W.item_id(name)] = twin_env.inventory.get(W.item_id(name), 0) - n
        if W.step(twin_env, item["action"]) != item["twin"]:
            problems.append("twin does not reproduce")
        if item["twin_source"] != ("env_remove" if success else "env_topup") or (success and item["topup"]) \
                or (not success and item["removal"]):
            problems.append("craft twin not from the environment")
    elif item["twin"] != f"Got 1 {item['object']}" or item["twin_source"] != "template":
        problems.append("get twin is not the template")
    held = {W.name_of(k) for k, v in env.inventory.items() if v > 0}
    if any(held & set(B.TAG_READING.get(s["name"], ())) for s in item["short_inputs"]):
        problems.append("a short input is a tag-derived name while a Minecraft member of it is held")
    if any(B._TWIN_OF.get(s["name"]) in held for s in item["short_inputs"]):
        problems.append("a short input's singular or plural twin is held")
    if item["action_kind"] == "craft":
        state = B.State(W.new_env(item["recipe_lines"], None)[0], [], {})
        for a in actions:
            B.apply(state, a)
        inputs = item["short_inputs"] if not success else [
            {"name": n, "peak": state.peaked(n)} for _c, n in K.parse_craft(item["action"])[2]]
        if item["certificate"] != B.certificate(state, inputs):
            problems.append("certificate does not follow from the environment's inventory after each step")
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
    want = ("secondary_get" if item["action_kind"] == "get" else "success" if item["failure_class"] == "success"
            else "main" if t >= 1 else "empty_history")
    if want == "main" and item["population"] == "usefulness" and item["failure_class"] == "never_obtained" \
            and B.usefulness_only(item):
        want = "usefulness"
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
    success = item["population"] == "success"
    plus, minus = (item["reply"], item["twin"]) if success else (item["twin"], item["reply"])
    if item["candidates"] != {"o_plus": plus, "o_minus": minus}:
        problems.append("candidates are not the success and the failure reply")
    if not plus.startswith(("Crafted ", "Got ")) or not minus.startswith("Could not find "):
        problems.append("o_plus is not a success or o_minus not a failure")
    pre = item["prefix_candidates"]
    if pre != B.prefix_candidates(item["action_kind"], plus) or not all(
            full.startswith(pre[k]) and full[len(pre[k]):] == " " + (W.item_id(item["object"]) if item["action_kind"] == "craft"
                                                                       else item["object"])
            for k, full in (("o_plus", plus), ("o_minus", minus))):
        problems.append("prefix candidates are not the replies without the object name")
    if item["object_goal_overlap"] != B.shares_word(item["object"], item["goal"]) or \
            item["object_partner_overlap"] != B.shares_word(item["object"], item["partner"]):
        problems.append("word-overlap flags wrong")
    want_couple = B.couple_id(item["pair_id"], item["history"], item["failure_class"], item["role"]) \
        if item["population"] in ("main", "success") else None
    if item["population"] == "usefulness":
        want_couple = item.get("couple_id") if str(item.get("couple_id")).startswith("mc-") else "an mc- id"
    if item.get("couple_id", "missing") != want_couple:
        problems.append("couple id wrong")
    if [item["history_fit"], item["history_fit_counts"]] != list(B.history_fit(item["history"], item["goal"], item["partner"])):
        problems.append("history fit does not recompute from the environment's trees")
    if item["fit_measures"] != F.measures(item["recipe_lines"], item["history"], item["goal"], item["partner"]):
        problems.append("fit measures do not recompute from the text")
    if item["inventory_shown_after_last_change"] is not B.inventory_shown_after_last_change(item["history"]):
        problems.append("inventory_shown_after_last_change wrong")
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
    for arm, field in (("aligned", "task_line"), ("swapped", "swapped_task_line"), ("neutral", "neutral_task_line")):
        query = P.messages(item, arm, shots)[-1]["content"]
        if [line for line in query.split("\n") if line.startswith("Goal: ")] != [item[field]]:
            problems.append(f"{arm}: the query's goal line is not the item's {field}")
    for arm in P.PREFIX_ARMS:
        if P.messages(item, arm, shots) != P.messages(item, arm.removeprefix("prefix_"), shots):
            problems.append(f"{arm}: prompt differs from its condition's")
        if P.candidates(item, arm) != (item["prefix_candidates"]["o_plus"], item["prefix_candidates"]["o_minus"]):
            problems.append(f"{arm}: candidates are not the prefixes")
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
    fit = Counter((it["failure_class"], it["length_bin"], it["role"], it["history_fit"]) for it in items
                  if it["population"] == "main")
    unbalanced = sorted("|".join(k[:3]) for k in fit if fit[k] != fit[(*k[:3], {"aligned": "swapped", "swapped": "aligned"}.get(k[3], k[3]))])
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
        "main_fit_unbalanced_cells": unbalanced,
        "main_couple_problems": couple_problems(items)[:20],
    }


COUPLED = ("main", "usefulness", "success")


def couple_problems(items: list[dict]) -> list[str]:
    """Every main, usefulness and success item is in a couple of exactly two (design 12, B1): the same population,
    goal pair and history, the same class and role, one item for each of the pair's goals; so every fit measure
    sums to zero over a couple."""
    by: dict[str, list[dict]] = {}
    for it in items:
        if it["population"] in COUPLED:
            by.setdefault(it["couple_id"], []).append(it)
    bad = []
    for cid, its in sorted(by.items(), key=lambda kv: str(kv[0])):
        if its[0]["population"] == "usefulness":  # sign-mirrored couples (12, C2)
            if len(its) != 2 or cid != B.mirrored_couple_id(*its) or its[0]["length_bin"] != its[1]["length_bin"] \
                    or not all(t["population"] == "usefulness" and B.usefulness_only(t) for t in its) \
                    or B.fit_profile(its[0]) != tuple(-v for v in B.fit_profile(its[1])):
                bad.append(f"couple {cid}")
            continue
        if len(its) != 2 or len({json.dumps([t["population"], t["pair_id"], t["history"], t["failure_class"], t["role"]]) for t in its}) != 1 \
                or {its[0]["goal"], its[0]["partner"]} != {its[1]["goal"], its[1]["partner"]} or its[0]["goal"] == its[1]["goal"]:
            bad.append(f"couple {cid}")
        elif any(F.measures(t["recipe_lines"], t["history"], t["goal"], t["partner"])[m][::-1]
                 != F.measures(o["recipe_lines"], o["history"], o["goal"], o["partner"])[m]
                 for t, o in [its] for m in F.MEASURES):
            bad.append(f"couple {cid}: fit measures not mirrored")
    return bad


def counts(items: list[dict]) -> dict:
    c = lambda f, xs=items: dict(sorted(Counter("|".join(str(it[k]) for k in f) for it in xs).items()))
    c_main = lambda f: c(f, [it for it in items if it["population"] == "main"])
    return {"population": c(("population",)), "population_class": c(("population", "failure_class")),
            "population_class_bin": c(("population", "failure_class", "length_bin")),
            "population_role": c(("population", "role")),
            "main_class_role_fit_bin": c_main(("failure_class", "role", "history_fit", "length_bin")),
            "main_class_needs_arithmetic": c_main(("failure_class", "needs_arithmetic")),
            "main_class_pure": c_main(("failure_class", "pure_class")),
            "main_role_overlap_goal_partner": c_main(("role", "object_goal_overlap", "object_partner_overlap")),
            "main_class_role_bin_pairs": {k: len({it["pair_id"] for it in items if it["population"] == "main"
                                                  and "|".join((it["failure_class"], it["role"], it["length_bin"])) == k})
                                          for k in c_main(("failure_class", "role", "length_bin"))},
            "main_class_role_no_fit_for_aligned": dict(sorted(Counter(
                "|".join((it["failure_class"], it["role"])) for it in items
                if it["population"] == "main" and F.none_favours_aligned(it["fit_measures"])).items())),
            "main_class_arithmetic_readout": dict(sorted(Counter(it["failure_class"] for it in items if it["population"] == "main"
                                                                 and it["needs_arithmetic"] and not it["inventory_shown_after_last_change"]).items())),
            "main_couples": len({it["couple_id"] for it in items if it["population"] == "main"}),
            "population_fit": c(("population", "history_fit")),
            "pairs": len({it["pair_id"] for it in items}),
            "goals": len({g for it in items for g in (it["goal"], it["partner"])})}


def replay_passes(r: dict, n_items: int) -> bool:
    """One fresh process's check_items report passes: every item checked, nothing failed, nothing duplicated,
    no prompt-example goal in an item, every main class x bin x role cell balanced on history fit, and every main
    item in a well-formed same-history couple."""
    return (r["n_items"] == n_items and not any(r[f"{k}_failures"] for k in ("replay", "checker", "structure", "prompt"))
            and r["duplicate_history_action"] == 0 and r["duplicate_ids"] == 0 and not r["fewshot_goals_in_items"]
            and not r["main_fit_unbalanced_cells"] and r.get("main_couple_problems", ["missing"]) == [])


# ---------------- whole-run checks ----------------


def _fresh(args: list[str], seed: str, jobs: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONHASHSEED": seed, **({"TEXTCRAFT_BUILD_JOBS": jobs} if jobs else {})}
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, check=True)


def determinism(items_sha: str, fewshot_sha: str) -> dict:
    """Two fresh builds under different hash seeds, with different numbers of worker processes, must write the
    committed files byte for byte."""
    out = {}
    for seed, jobs in (("0", "8"), ("1", "3")):
        with tempfile.TemporaryDirectory() as tmp:
            _fresh(["-m", "wf.textcraft.build", tmp], seed, jobs)
            out[seed] = {"items": sha256(Path(tmp) / "items.jsonl") == items_sha,
                         "fewshot": sha256(Path(tmp) / "fewshot.json") == fewshot_sha, "build_jobs": int(jobs)}
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
    replay_ok = len(replays) == 3 and all(replay_passes(r, len(items)) for r in replays.values())
    det_ok = all(v["items"] and v["fewshot"] for v in det.values())
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
