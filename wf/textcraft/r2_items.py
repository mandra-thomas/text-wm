"""TextCraft round 2 records: build and verify (notes/textcraft_r2_design.md, section 4).

    python -m wf.textcraft.r2_items build        # writes wf/textcraft/data/r2_items.jsonl and r2_build_stats.json
    python -m wf.textcraft.r2_items verify       # every check below in three fresh processes, determinism; writes
                                                 # wf/textcraft/data/r2_manifest.json
    python -m wf.textcraft.r2_items --replay     # the per-record checks in this process (verify runs it)

Main records are the frozen main and success items, copied field for field, plus the two-item line and their arms.
Parent records are goal-role items of the frozen builder's pool (the same pairs, walks, rules and exclusions,
wf/textcraft/build.py) whose goal G has a parent Z: an item of the database with a recipe line that lists G as an
input. Z's tree lines that the list does not already show are added at seeded positions, the list's own lines keeping
their order; a parent qualifies only when its tree shares no craftable item with the partner's tree.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from wf.textcraft import build as B
from wf.textcraft import checker as K
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import world as W

DATA = B.DATA
R2_ITEMS = DATA / "r2_items.jsonl"
R2_STATS = DATA / "r2_build_stats.json"
R2_MANIFEST = DATA / "r2_manifest.json"
REPO = Path(__file__).resolve().parents[2]
SEED = B.SEED
PARENT_FAILURES_PER_BIN = 2  # per qualifying goal and length bin (1-3, 4-8, 9-12)
PARENT_SUCCESSES_PER_BIN = 1
PARENT_BINS = ("1-3", "4-8", "9-12")


# ---------------- parent goals ----------------


@lru_cache(maxsize=1)
def consumers() -> dict[str, frozenset[str]]:
    """name -> the outputs of every recipe line that lists the name itself (not a generic group) as an input."""
    out: dict[str, set[str]] = defaultdict(set)
    for ln in B.line_index().values():
        for _c, n, generic in ln.inputs:
            if not generic:
                out[n].add(ln.out)
    return {k: frozenset(v) for k, v in out.items()}


@lru_cache(maxsize=1)
def fewshot_goals() -> frozenset[str]:
    held = set(B.fewshot_pair_ids())
    return frozenset(g for p in B.all_pairs() if p.pair_id in held for g in p.goals)


def added_lines(lines: list[str] | tuple[str, ...], parent: str) -> list[str]:
    """Z's tree lines (the environment's create_recipe_set for goal Z, without distractors) not already shown."""
    shown = set(lines)
    return sorted(set(W.recipe_tree(W.item_id(parent))) - shown)


def qualifies(goal: str, partner: str, parent: str) -> bool:
    """The rules of design section 4.2, with the environment's trees."""
    if parent in (goal, partner) or parent in B.EXCLUDED_GOALS or parent in fewshot_goals():
        return False
    if parent not in consumers().get(goal, ()):
        return False
    return not (W.tree_craftables(W.item_id(parent)) & W.tree_craftables(W.item_id(partner)))


def choose_parent(pair: B.Pair, goal: str) -> dict | None:
    """The parent of `goal`: among qualifying Z, one sharing no word with the goal if any does; then the fewest added
    lines; then a seeded draw. None when no Z qualifies."""
    partner = next(g for g in pair.goals if g != goal)
    ok = sorted(z for z in consumers().get(goal, ()) if qualifies(goal, partner, z))
    if not ok:
        return None
    strict = [z for z in ok if not B.shares_word(z, goal)]
    pool = strict or ok
    fewest = min(len(added_lines(pair.lines, z)) for z in pool)
    tied = [z for z in pool if len(added_lines(pair.lines, z)) == fewest]
    z = random.Random(f"{SEED}|textcraft|r2|parent|{pair.pair_id}|{goal}").choice(tied)
    return {"parent": z, "rule": "strict" if strict else "word_overlap", "n_qualifying": len(ok)}


def extended_list(lines: list[str] | tuple[str, ...], added: list[str], pair_id: str, goal: str, parent: str) -> list[str]:
    """The list with each added line (sorted) put at a seeded position; the list's own lines keep their order."""
    out = list(lines)
    rng = random.Random(f"{SEED}|textcraft|r2|extlist|{pair_id}|{goal}|{parent}")
    for line in added:
        out.insert(rng.randint(0, len(out)), line)
    return out


def _pool_of(pair_id: str) -> list[dict]:
    excluded: Counter = Counter()
    return B.eligible(next(p for p in B.all_pairs() if p.pair_id == pair_id), excluded)


def parent_records(frozen_ids: set[str], jobs: int = 1, only_pairs: set[str] | None = None) -> tuple[list[dict], dict]:
    """Parent records of every pair (or of `only_pairs`, for tests), in (pair, goal, population, bin, id) order."""
    held_out = set(B.fewshot_pair_ids())
    chosen = {(p.pair_id, g): c for p in B.all_pairs() if p.pair_id not in held_out
              and (only_pairs is None or p.pair_id in only_pairs) for g in p.goals
              if (c := choose_parent(p, g)) is not None}
    pair_ids = sorted({pid for pid, _g in chosen})
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(jobs) as workers:
            pools = dict(zip(pair_ids, workers.map(_pool_of, pair_ids, chunksize=1)))
    else:
        pools = {pid: _pool_of(pid) for pid in pair_ids}
    pairs = {p.pair_id: p for p in B.all_pairs()}
    out, eligible = [], Counter()
    for (pid, goal), choice in sorted(chosen.items()):
        pair, z = pairs[pid], choice["parent"]
        add = added_lines(pair.lines, z)
        ext = extended_list(pair.lines, add, pid, goal, z)
        cands = [it for it in pools[pid] if it["role"] == "goal" and it["object"] == goal and it["id"] not in frozen_ids
                 and it["population"] in ("main", "success")]
        for population, per_bin in (("main", PARENT_FAILURES_PER_BIN), ("success", PARENT_SUCCESSES_PER_BIN)):
            for b in PARENT_BINS:
                cell = [it for it in cands if it["population"] == population and it["length_bin"] == b]
                eligible[f"{population}|{b}"] += len(cell)
                rng = random.Random(f"{SEED}|textcraft|r2|parent|select|{pid}|{goal}|{population}|{b}")
                for it in sorted(rng.sample(cell, min(per_bin, len(cell))), key=lambda x: x["id"]):
                    out.append({**it, "kind": "parent", "parent": z, "parent_rule": choice["rule"],
                                "parent_task_line": B.goal_line(z), "ext_recipe_lines": ext, "added_lines": add,
                                "parent_shares_word_with_goal": B.shares_word(z, goal),
                                "parent_shares_word_with_partner": B.shares_word(z, it["partner"]),
                                "both_task_line": R.two_item_line(goal, it["partner"]), "arms": list(R.PARENT_ARMS)})
    stats = {"qualifying_goals": len(chosen), "qualifying_pairs": len(pair_ids),
             "goals_by_rule": dict(Counter(c["rule"] for c in chosen.values())),
             "eligible_by_population_bin": dict(sorted(eligible.items())),
             "added_lines": dict(sorted(Counter(len(added_lines(pairs[pid].lines, c["parent"]))
                                                for (pid, _g), c in chosen.items()).items())),
             "parents": {f"{pid}|{g}": c["parent"] for (pid, g), c in sorted(chosen.items())}}
    return out, stats


# ---------------- main records ----------------


def main_arms(it: dict) -> list[str]:
    """The eight line arms (without the two-item ones when the pair has no two-item line), and the padding arms for a
    main failure at 1 to 8 steps."""
    both = R.two_item_line(it["goal"], it["partner"]) is not None
    arms = [a for a in R.MAIN_ARMS if both or a not in R.BOTH_ARMS]
    if it["population"] == "main" and it["length_bin"] in R.PAD_BINS:
        arms += list(R.PAD_ARMS)
    return arms


def main_records(frozen: list[dict]) -> list[dict]:
    """The frozen main and success items, unchanged, with the two-item line and their arms."""
    out = []
    for it in frozen:
        if it["population"] not in ("main", "success"):
            continue
        out.append({**it, "kind": "main", "both_task_line": R.two_item_line(it["goal"], it["partner"]),
                    "arms": main_arms(it)})
    return out


def build(jobs: int = 1) -> tuple[list[dict], dict]:
    frozen = [json.loads(line) for line in B.ITEMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    main = main_records(frozen)
    parents, stats = parent_records({t["id"] for t in frozen}, jobs)
    records = main + parents
    stats = {"parent": stats, "records": dict(Counter(r["kind"] for r in records)),
             "scores_per_model": sum(len(r["arms"]) for r in records),
             "arms": dict(sorted(Counter(a for r in records for a in r["arms"]).items())),
             "parent_by_population_class_bin": dict(sorted(Counter(
                 f"{r['population']}|{r['failure_class']}|{r['length_bin']}" for r in parents).items())),
             "parent_pairs": len({r["pair_id"] for r in parents}), "parent_goals": len({r["goal"] for r in parents}),
             "parent_strict_records": sum(r["parent_rule"] == "strict" for r in parents)}
    return records, stats


def dumps(records: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records)


def read_records(path: Path = R2_ITEMS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------- checks ----------------


def goal_line_of(query: str) -> list[str]:
    return [line for line in query.split("\n") if line.startswith("Goal: ")]


def prompt_problems(rec: dict, shots) -> list[str]:
    """Each arm changes only what its name says, against the record's aligned prompt."""
    problems = []
    base = R.messages(rec, "aligned", shots)
    lines, history, _g = R.shown(rec, "aligned")
    for arm in rec["arms"]:
        m = R.messages(rec, arm, shots)
        if m[:-1] != base[:-1]:
            problems.append(f"{arm}: system text or examples differ")
        question, pad, ext, line = R.parse_arm(arm)
        want_lines = rec["ext_recipe_lines"] if ext else rec["recipe_lines"]
        want_history = [list(R.PAD_STEP)] * pad + [list(h) for h in rec["history"]]
        expected = P.turn(want_lines, want_history, rec["action"], rec[R.LINE_FIELDS[line]])
        if question:
            expected = expected[: -len(P.NEXT)] + P.YN_QUESTION
        if m[-1]["content"] != expected:
            problems.append(f"{arm}: query is not the shown list, history, action and goal line")
        if goal_line_of(m[-1]["content"]) != [rec[R.LINE_FIELDS[line]]]:
            problems.append(f"{arm}: the query's goal line is not the record's {R.LINE_FIELDS[line]}")
        plain = m[-1]["content"] if not question else m[-1]["content"][: -len(P.YN_QUESTION)] + P.NEXT
        swapped_back = plain.replace(rec[R.LINE_FIELDS[line]], rec["task_line"], 1)
        if not pad and not ext and swapped_back != base[-1]["content"]:
            problems.append(f"{arm}: query differs from the aligned one beyond the goal line")
        if pad and swapped_back != P.turn(lines, want_history, rec["action"], rec["task_line"]):
            problems.append(f"{arm}: padded query differs beyond the padding and the goal line")
        if pad and history != want_history[pad:]:
            problems.append(f"{arm}: padding is not only at the start")
    if rec["kind"] == "parent":
        ext = rec["ext_recipe_lines"]
        if [x for x in ext if x not in set(rec["added_lines"])] != rec["recipe_lines"]:
            problems.append("extended list does not keep the list's own lines in order")
        if sorted(set(ext) - set(rec["recipe_lines"])) != rec["added_lines"] or len(ext) != len(rec["recipe_lines"]) + len(rec["added_lines"]):
            problems.append("extended list adds other lines than the added lines")
    return problems


def structure_problems(rec: dict, frozen_by_id: dict[str, dict]) -> list[str]:
    problems = []
    if rec["both_task_line"] != R.two_item_line(rec["goal"], rec["partner"]):
        problems.append("two-item line wrong")
    if rec["kind"] == "main":
        f = frozen_by_id.get(rec["id"])
        if f is None:
            return problems + ["main record not in the frozen items"]
        if {k: v for k, v in rec.items() if k not in ("kind", "both_task_line", "arms")} != f:
            problems.append("main record differs from its frozen item")
        if rec["arms"] != main_arms(f):
            problems.append("main arms wrong")
        return problems
    if rec["id"] in frozen_by_id:
        problems.append("parent record reuses a frozen item")
    if rec["role"] != "goal" or rec["object"] != rec["goal"] or rec["population"] not in ("main", "success"):
        problems.append("parent record is not a goal-role failure or success")
    if rec["population"] == "main" and rec["history_len"] < 1:
        problems.append("parent failure with an empty history")
    if rec["arms"] != list(R.PARENT_ARMS):
        problems.append("parent arms wrong")
    z = rec["parent"]
    if not qualifies(rec["goal"], rec["partner"], z):
        problems.append("parent does not qualify")
    if rec["parent_task_line"] != f"Goal: craft {z}.":
        problems.append("parent line wrong")
    if rec["added_lines"] != added_lines(rec["recipe_lines"], z):
        problems.append("added lines are not the parent's tree lines missing from the list")
    if rec["ext_recipe_lines"] != extended_list(rec["recipe_lines"], rec["added_lines"], rec["pair_id"], rec["goal"], z):
        problems.append("extended list is not the seeded insertion")
    pair = next(p for p in B.all_pairs() if p.pair_id == rec["pair_id"])
    choice = choose_parent(pair, rec["goal"])
    if choice is None or choice["parent"] != z or choice["rule"] != rec["parent_rule"]:
        problems.append("parent is not the rule's choice")
    if rec["parent_shares_word_with_goal"] != B.shares_word(z, rec["goal"]) or \
            rec["parent_shares_word_with_partner"] != B.shares_word(z, rec["partner"]):
        problems.append("parent word flags wrong")
    if rec["recipe_lines"] != list(pair.lines):
        problems.append("list is not the pair's list")
    return problems


def text_parent_problems(rec: dict) -> list[str]:
    """Text-only (no environment): on the extended list, Z has a displayed line listing G as an input; the names of
    Z's displayed tree share no craftable (a line's output) with the partner's displayed tree; the parent line does
    not name G; on the extended list the action is still the same determined failure or success."""
    problems = []
    ext = rec["ext_recipe_lines"]
    parsed = [K.parse_craft(x) for x in ext]
    if not any(out == rec["parent"] and any(n == rec["goal"] for _c, n in ins) for _k, out, ins in parsed):
        problems.append("no displayed parent line lists the goal as an input")
    outputs = K.displayed_outputs(ext)
    z_tree = K.tree_names(ext, rec["parent"]) & outputs
    p_tree = K.tree_names(ext, rec["partner"]) & outputs
    if z_tree & p_tree:
        problems.append(f"parent tree shares craftables with the partner's: {sorted(z_tree & p_tree)}")
    if rec["goal"] not in z_tree:
        problems.append("goal is not in the parent's displayed tree")
    if rec["parent_rule"] == "strict" and set(rec["parent"].split()) & set(rec["goal"].split()):
        problems.append("strict parent shares a word with the goal")
    v_ext, v = K.decide(ext, rec["history"], rec["action"]), K.decide(rec["recipe_lines"], rec["history"], rec["action"])
    if (v_ext.determined_failure, v_ext.determined_success, v_ext.reply, v_ext.twin) != \
            (v.determined_failure, v.determined_success, v.reply, v.twin):
        problems.append("the extended list changes the checker's verdict")
    return problems


def replay_problems(rec: dict) -> list[str]:
    """Environment replay. Parent records: the frozen verify replay on the item's list, then on the extended list
    with each of goal, partner, parent and no goal: the same history replies, the same reply to the action, the same
    twin. Padded histories: each `inventory` look-up replies exactly the padding text, and the rest replays as stored."""
    from wf.textcraft import verify as V

    problems = list(V.replay_problems(rec)) if rec["kind"] == "parent" else []
    actions = [a for a, _o in rec["history"]]
    variants = []
    if rec["kind"] == "parent":
        variants += [(rec["ext_recipe_lines"], g, 0) for g in (rec["goal"], rec["partner"], rec["parent"], None)]
    pads = sorted({R.parse_arm(a)[1] for a in rec["arms"]} - {0})
    variants += [(rec["recipe_lines"], g, k) for k in pads for g in (rec["goal"], rec["partner"], None)]
    for lines, goal, pad in variants:
        env, hist = W.replay(lines, W.item_id(goal) if goal else None, ["inventory"] * pad + actions)
        if hist != [list(R.PAD_STEP)] * pad + [list(h) for h in rec["history"]]:
            problems.append(f"history does not replay (goal {goal}, padding {pad}, extended {lines is not rec['recipe_lines']})")
            continue
        if W.step(W.clone(env), rec["action"]) != rec["reply"]:
            problems.append(f"reply does not reproduce (goal {goal}, padding {pad})")
        twin_env = W.clone(env)
        for name, n in rec["topup"].items():
            twin_env.inventory[W.item_id(name)] = twin_env.inventory.get(W.item_id(name), 0) + n
        for name, n in rec["removal"].items():
            twin_env.inventory[W.item_id(name)] = twin_env.inventory.get(W.item_id(name), 0) - n
        if W.step(twin_env, rec["action"]) != rec["twin"]:
            problems.append(f"twin does not reproduce (goal {goal}, padding {pad})")
    return problems


def checker_problems(rec: dict) -> list[str]:
    problems = [f"checker: {p}" for p in K.check_item({k: v for k, v in rec.items()})]
    for k in sorted({R.parse_arm(a)[1] for a in rec["arms"]} - {0}):
        padded = [list(R.PAD_STEP)] * k + [list(h) for h in rec["history"]]
        try:
            K.account(padded)
            v = K.decide(rec["recipe_lines"], padded, rec["action"])
        except K.CheckerError as e:
            problems.append(f"padding {k}: {e}")
            continue
        if not v.determined_failure or v.reply != rec["reply"]:
            problems.append(f"padding {k}: the padded history changes the verdict")
        if v.failure_class != rec["failure_class"] or v.short_inputs != rec["short_inputs"]:
            problems.append(f"padding {k}: the padded history changes the class or short inputs")
    if rec["kind"] == "parent":
        problems += text_parent_problems(rec)
    return problems


def check_records(records: list[dict], shots) -> dict:
    frozen = {t["id"]: t for t in (json.loads(x) for x in B.ITEMS.read_text(encoding="utf-8").splitlines() if x.strip())}
    fails: dict[str, list] = {"replay": [], "checker": [], "structure": [], "prompt": []}
    for r in records:
        for name, probs in (("replay", replay_problems(r)), ("checker", checker_problems(r)),
                            ("structure", structure_problems(r, frozen)), ("prompt", prompt_problems(r, shots))):
            if probs:
                fails[name].append((r["id"], probs))
    ids = Counter(r["id"] for r in records)
    goals = {g for r in records for g in (r["goal"], r["partner"], r.get("parent")) if g}
    return {"n_records": len(records), "hash_seed": os.environ.get("PYTHONHASHSEED"),
            **{f"{k}_failures": len(v) for k, v in fails.items()},
            "examples": {k: v[:5] for k, v in fails.items() if v},
            "duplicate_ids": sum(n - 1 for n in ids.values() if n > 1),
            "fewshot_goals_in_records": sorted(goals & fewshot_goals())}


def passes(r: dict, n: int) -> bool:
    return (r["n_records"] == n and not any(r[f"{k}_failures"] for k in ("replay", "checker", "structure", "prompt"))
            and r["duplicate_ids"] == 0 and not r["fewshot_goals_in_records"])


def _fresh(args: list[str], seed: str, jobs: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONHASHSEED": seed, **({"TEXTCRAFT_BUILD_JOBS": jobs} if jobs else {})}
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, check=True)


def verify() -> int:
    from wf.textcraft.verify import sha256

    records = read_records()
    replays = {s: json.loads(_fresh(["-m", "wf.textcraft.r2_items", "--replay"], s).stdout) for s in ("0", "1", "2")}
    det = {}
    for seed, jobs in (("0", "8"), ("1", "3")):
        with tempfile.TemporaryDirectory() as tmp:
            _fresh(["-m", "wf.textcraft.r2_items", "build", tmp], seed, jobs)
            det[seed] = {"records": sha256(Path(tmp) / "r2_items.jsonl") == sha256(R2_ITEMS), "build_jobs": int(jobs)}
    base = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    base_ok = base.get("verified") is True and sha256(B.ITEMS) == base["items_sha256"] and sha256(P.FEWSHOT) == base["fewshot_sha256"]
    replay_ok = all(passes(r, len(records)) for r in replays.values())
    det_ok = all(v["records"] for v in det.values())
    manifest = {"records_file": str(R2_ITEMS.relative_to(REPO)), "records_sha256": sha256(R2_ITEMS),
                "base_items_sha256": sha256(B.ITEMS), "fewshot_file": str(P.FEWSHOT.relative_to(REPO)),
                "fewshot_sha256": sha256(P.FEWSHOT), "frozen_db_sha256": sha256(DATA / "recipes_frozen.json"),
                "counts": json.loads(R2_STATS.read_text(encoding="utf-8")) if R2_STATS.exists() else None,
                "checks": {"replay_checker_structure_prompt_in_fresh_processes": replays, "build_determinism": det,
                           "base_manifest_verified": base_ok},
                "verified": replay_ok and det_ok and base_ok,
                "why_not": [w for w, ok in (("per-record checks", replay_ok), ("determinism", det_ok),
                                            ("base items manifest", base_ok)) if not ok]}
    R2_MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified": manifest["verified"], "why_not": manifest["why_not"]}))
    return 0 if manifest["verified"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--replay"]:
        print(json.dumps(check_records(read_records(), P.fewshot_examples())))
        return 0
    if argv[:1] == ["verify"]:
        return verify()
    if argv[:1] == ["build"]:
        out = Path(argv[1]) if len(argv) > 1 else DATA
        out.mkdir(parents=True, exist_ok=True)
        records, stats = build(int(os.environ.get("TEXTCRAFT_BUILD_JOBS") or os.cpu_count() or 1))
        (out / "r2_items.jsonl").write_text(dumps(records), encoding="utf-8")
        (out / "r2_build_stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in stats.items() if k not in ("arms",)} | {"parent": {k: v for k, v in stats["parent"].items() if k != "parents"}}))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
