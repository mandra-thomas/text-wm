"""Item builder (wf/textcraft/build.py) and the committed items: design rules, replay, duplicates, manifest."""
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from wf.textcraft import build as B
from wf.textcraft import verify as V
from wf.textcraft import world as W

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def items():
    return V.read_items()


@pytest.fixture(scope="module")
def pairs():
    return {p.pair_id: p for p in B.all_pairs()}


def test_pairs_are_one_to_one_depth_matched_disjoint_and_genuine(pairs):
    pool = dict(W.goal_pool())
    goals = [g for p in pairs.values() for g in p.goals]
    assert len(goals) == len(set(goals)) == 2 * len(pairs) == 538
    for p in pairs.values():
        a, b = (W.item_id(g) for g in p.goals)
        assert pool[a] == pool[b] == p.depth and p.goals == tuple(sorted(p.goals))
        assert not W.tree_craftables(a) & W.tree_craftables(b)
    assert not set(goals) & set(B.EXCLUDED_GOALS)


def test_list_is_both_trees_and_does_not_mark_the_real_goal(pairs):
    for p in list(pairs.values())[:40]:
        a, b = p.goals
        assert set(p.lines) == set(W.recipe_tree(W.item_id(a))) | set(W.recipe_tree(W.item_id(b)))
        assert len(p.lines) == len(set(p.lines))
        assert B.union_list(a, b) == B.union_list(b, a) == p.lines


def test_walk_is_goal_blind_and_only_takes_successful_actions(pairs):
    for p in list(pairs.values())[::25]:
        flipped = B.Pair(p.pair_id, p.goals[::-1], p.depth, p.lines)
        for seed in range(2):
            acts = B.walk(p, seed)
            assert acts == B.walk(flipped, seed) and len(acts) == B.STEPS
            _env, hist = W.replay(list(p.lines), None, acts)
            assert all(o.startswith(("Got ", "Crafted ", "Inventory: ")) for _, o in hist)
            assert not any(o.split(" ", 2)[2] in {W.item_id(g) for g in p.goals} for _, o in hist if o.startswith("Crafted "))


def test_classify_known_answers():
    lines = ("craft 4 stick using 2 planks", "craft 4 oak planks using 1 oak logs")
    state = B.State(W.new_env(lines, None)[0], [], {})
    for a in ["get 1 oak logs", "craft 4 oak planks using 1 oak logs", "craft 4 stick using 2 oak planks",
              "craft 4 stick using 2 oak planks", "get 2 iron ingot"]:
        B.apply(state, a)
    cls, short = B.classify(state, [(2, "oak planks"), (5, "iron ingot"), (1, "gold nugget")])
    assert cls == "used_up"
    assert [(s["name"], s["held"], s["peak"], s["kind"]) for s in short] == [
        ("oak planks", 0, 4, "used_up"), ("iron ingot", 2, 2, "too_few"), ("gold nugget", 0, 0, "never_obtained")]
    assert B.classify(state, [(8, "stick")]) == (None, [])
    assert B.topped_up_reply(state, short[:1], "craft 4 stick using 2 oak planks") == "Crafted 4 minecraft:stick"


def test_generic_ingredient_takes_the_kind_held_most(pairs):
    lines = ("craft 4 stick using 2 planks", "craft 4 oak planks using 1 oak logs", "craft 4 birch planks using 1 birch logs")
    state = B.State(W.new_env(lines, None)[0], [], {})
    shown = {"stick", "oak planks", "birch planks"}
    assert B.fill_generic("planks", state, shown) == "birch planks"  # nothing held: first displayed kind
    for a in ["get 1 oak logs", "craft 4 oak planks using 1 oak logs", "get 1 birch logs", "craft 4 birch planks using 1 birch logs",
              "craft 4 stick using 2 birch planks"]:
        B.apply(state, a)
    assert B.fill_generic("planks", state, shown) == "oak planks"  # 4 oak against 2 birch
    B.apply(state, "craft 4 stick using 2 oak planks")
    B.apply(state, "craft 4 stick using 2 oak planks")
    assert B.fill_generic("planks", state, shown) == "birch planks"  # 2 birch held; oak's peak no longer counts


def test_split_and_round_robin():
    assert B._split(10, {"a": 2, "b": 10, "c": 10}) == {"a": 2, "b": 4, "c": 4}
    assert B._split(5, {"a": 1, "b": 1}) == {"a": 1, "b": 1}
    assert B._split(7, {"a": 9, "b": 9, "c": 9}) == {"a": 3, "b": 2, "c": 2}
    pool = [{"pair_id": f"p{i % 5}", "n": i} for i in range(50)]
    got = B._round_robin(pool, 10, random.Random(0))
    assert Counter(x["pair_id"] for x in got) == Counter({f"p{i}": 2 for i in range(5)})


def test_committed_items_rebuild_exactly(items, pairs):
    """The stored items of a sample of pairs are exactly what the builder makes for them now."""
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], []).append(it)
    for pid in sorted(by_pair)[::30]:
        fresh = {it["id"]: it for it in B.eligible(pairs[pid], Counter())}
        for it in by_pair[pid]:
            assert fresh.get(it["id"]) == it


def test_selection_follows_the_quotas(items):
    c = Counter((it["population"], it["failure_class"], it["length_bin"]) for it in items)
    for cls in B.CLASSES:
        assert sum(v for (pop, k, _b), v in c.items() if pop == "main" and k == cls) == 500
        assert {c[("main", cls, b)] for b in ("1-3", "4-8", "9-12")} <= {166, 167}
    assert all(c[("secondary_get", "get_craftable", b)] == 100 for b in B.BINS)
    assert c[("empty_history", "never_obtained", "0")] == 100 and len(items) == 2000
    assert len({it["pair_id"] for it in items if it["population"] == "main"}) == 266


def test_every_item_replays_and_follows_the_design(items):
    shots = V.P.fewshot_examples()
    for it in items:
        assert V.replay_problems(it) == [], it["id"]
        assert V.structure_problems(it) == [], it["id"]


def test_twin_tops_up_exactly_the_missing_amounts(items):
    crafts = [it for it in items if it["action_kind"] == "craft"]
    for it in crafts:
        assert it["topup"] == {s["name"]: s["need"] - s["held"] for s in it["short_inputs"]}
        assert all(s["held"] < s["need"] for s in it["short_inputs"])
    assert any(s["held"] > 0 for it in crafts for s in it["short_inputs"])  # top-up below the full count happens


def test_no_duplicates(items):
    keys = [(it["pair_id"], json.dumps(it["history"]), it["action"]) for it in items]
    assert len(set(keys)) == len(keys) and len({it["id"] for it in items}) == len(items)


def test_fewshot_pairs_are_held_out(items):
    held = set(B.fewshot_pair_ids())
    shots = V.P.fewshot_examples()
    assert len(held) == 3 and {ex["pair_id"] for ex in shots} == held
    assert not held & {it["pair_id"] for it in items}
    item_goals = {g for it in items for g in (it["goal"], it["partner"])}
    assert not {ex["goal"] for ex in shots} & item_goals


def test_fewshot_examples_replay():
    shots = V.P.fewshot_examples()
    assert [ex["observation"].split(" ")[0] for ex in shots] == ["Crafted", "Could", "Got"]
    for ex in shots:
        env, hist = W.replay(ex["recipe_lines"], W.item_id(ex["goal"]), [a for a, _ in ex["history"]])
        assert hist == ex["history"] and W.step(env, ex["action"]) == ex["observation"]
        assert len(ex["history"]) >= 2


def test_manifest_hashes_the_committed_files():
    m = json.loads(V.MANIFEST.read_text())
    assert m["verified"] is True and m["why_not"] == []
    assert m["items_sha256"] == V.sha256(B.ITEMS)
    assert m["fewshot_sha256"] == V.sha256(V.P.FEWSHOT)
    assert m["frozen_db_sha256"] == V.sha256(V.FROZEN)
    up = m["checks"]["upstream_equivalence"]
    assert up["mismatches"] == 0 and up["tree_identical"] and up["items_sha256"] == m["items_sha256"]
    assert set(m["checks"]["full_replay_and_checker_in_fresh_processes"]) == {"0", "1", "2"}


def test_candidate_pool_drops_the_prompt_example_pairs(pairs):
    fake = {pid: [{"pair_id": pid}] for pid in pairs}
    kept = {it["pair_id"] for it in B.candidate_pool(fake)}
    assert kept == set(pairs) - set(B.fewshot_pair_ids()) and len(kept) == 266
