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
    # in couples (section 11, A3): 250 per class, 84/83/83 per bin; used up has only 2 couples at 1-3 steps
    assert [c[("main", "used_up", b)] for b in ("1-3", "4-8", "9-12")] == [4, 248, 248]
    for cls in ("too_few", "never_obtained"):
        assert [c[("main", cls, b)] for b in ("1-3", "4-8", "9-12")] == [168, 166, 166]
    assert all(c[("secondary_get", "get_craftable", b)] == 100 for b in B.BINS)
    assert c[("empty_history", "never_obtained", "0")] == 100 and len(items) == 2000
    assert len({it["pair_id"] for it in items if it["population"] == "main"}) == 257


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


# ---------------- amendments after review 1 (design section 11) ----------------

def _synthetic_pool():
    """A fixed pool of fake items over 12 pairs, with every population, class, bin, role and history fit."""
    rng = random.Random(7)
    pool = []
    for pop, (_total, classes, bins) in B.QUOTAS.items():
        for c in classes:
            for b in bins:
                for role in B.ROLES:
                    for fit in B.FITS:
                        for k in range(rng.randrange(0, 40)):
                            pid = f"p{rng.randrange(12):03d}"
                            pool.append({"id": f"{pop}-{c}-{b}-{role}-{fit}-{k}", "pair_id": pid, "population": pop,
                                         "failure_class": c, "length_bin": b, "role": role, "history_fit": fit})
    return pool


def test_selection_golden():
    """Selection on a fixed fake pool gives exactly these items: a changed seed, a lost shuffle or a changed quota
    rule fails here, not only in the full rebuild that verify runs."""
    import hashlib
    chosen = B.select(_synthetic_pool())
    digest = hashlib.sha256(json.dumps([it["id"] for it in chosen]).encode()).hexdigest()
    assert digest == GOLDEN_SELECTION, digest


GOLDEN_SELECTION = "5bce8336022a3cd31d7af04ae0106d18e79444e29bb671459a28df29d3918727"


def test_main_selection_balances_history_fit_and_skips_ties():
    chosen = [it for it in B.select(_synthetic_pool()) if it["population"] == "main"]
    c = Counter((it["failure_class"], it["length_bin"], it["role"], it["history_fit"]) for it in chosen)
    assert "tie" not in {k[3] for k in c}
    for (cls, b, role, fit), n in c.items():
        assert c[(cls, b, role, "swapped" if fit == "aligned" else "aligned")] == n


def test_committed_main_is_balanced_on_history_fit(items):
    main = [it for it in items if it["population"] == "main"]
    c = Counter((it["failure_class"], it["length_bin"], it["role"], it["history_fit"]) for it in main)
    assert len(main) == 1500 and "tie" not in {k[3] for k in c}
    for (cls, b, role, fit), n in c.items():
        assert c[(cls, b, role, "swapped" if fit == "aligned" else "aligned")] == n
    assert Counter(it["failure_class"] for it in main) == Counter({k: 500 for k in B.CLASSES})


def test_history_fit_known_answer():
    from wf.textcraft import checker as K
    lines = ["craft 4 stick using 2 planks", "craft 4 oak planks using 1 oak logs", "craft 1 ladder using 7 stick",
             "craft 1 minecart using 5 iron ingot"]
    hist = [["get 1 oak logs", "Got 1 oak logs"], ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"],
            ["get 5 iron ingot", "Got 5 iron ingot"], ["get 2 oak logs", "Got 2 oak logs"]]
    # ladder's tree: stick, planks (and its member oak planks), oak logs; minecart's: iron ingot
    assert K.tree_names(lines, "ladder") == {"ladder", "stick", "planks", "oak planks", "oak logs"}
    assert K.history_fit(lines, hist, "ladder", "minecart") == ("aligned", {"aligned": 3, "swapped": 1})
    assert K.history_fit(lines, hist, "minecart", "ladder") == ("swapped", {"aligned": 1, "swapped": 3})
    assert K.history_fit(lines, hist[2:3] + hist[:1], "ladder", "minecart")[0] == "tie"
    assert K.history_fit(lines, [], "ladder", "minecart") == ("tie", {"aligned": 0, "swapped": 0})


def test_history_fit_from_the_environment_agrees_with_the_text(items):
    from wf.textcraft import checker as K
    for it in items[::7]:
        fit = B.history_fit(it["history"], it["goal"], it["partner"])
        assert fit == K.history_fit(it["recipe_lines"], it["history"], it["goal"], it["partner"])
        assert [it["history_fit"], it["history_fit_counts"]] == list(fit)


def test_tag_reading_ambiguity_is_excluded():
    lines = ("craft 1 campfire using 3 logs, 3 stick, 1 coals", "craft 4 warped planks using 1 warped stems")
    state = B.State(W.new_env(lines, None)[0], [], {})
    for a in ["get 3 warped stems", "get 1 logs"]:
        B.apply(state, a)
    _cls, short = B.classify(state, [(3, "logs"), (3, "stick"), (1, "coals")])
    assert B.tag_reading_ambiguous(state, short)  # 3 warped stems held: a Minecraft reader may count them as logs
    B.apply(state, "craft 4 warped planks using 1 warped stems")
    B.apply(state, "craft 4 warped planks using 1 warped stems")
    B.apply(state, "craft 4 warped planks using 1 warped stems")
    _cls, short = B.classify(state, [(3, "logs"), (3, "stick"), (1, "coals")])
    assert not B.tag_reading_ambiguous(state, short)  # no member held any more


def test_no_committed_item_is_tag_reading_ambiguous(items):
    from wf.textcraft import checker as K
    for it in items:
        if it["action_kind"] != "craft":
            continue
        held = K.account(it["history"]).held
        for s in it["short_inputs"]:
            assert not any(held.get(m, 0) > 0 for m in B.TAG_READING.get(s["name"], ())), it["id"]


def test_arithmetic_and_purity_flags(items):
    for it in items:
        if it["action_kind"] == "get":
            assert it["pure_class"] is None and it["needs_arithmetic"] is None
            continue
        kinds = [s["kind"] for s in it["short_inputs"]]
        assert it["pure_class"] == (len(set(kinds)) == 1)
        assert it["needs_arithmetic"] == ("never_obtained" not in kinds)
    main = [it for it in items if it["population"] == "main"]
    assert not any(it["needs_arithmetic"] for it in main if it["failure_class"] == "never_obtained")
    assert any(not it["pure_class"] for it in main if it["failure_class"] == "too_few")


def test_prefix_candidates_and_overlap_flags(items):
    for it in items:
        pre = it["prefix_candidates"]
        name = W.item_id(it["object"]) if it["action_kind"] == "craft" else it["object"]
        assert it["twin"] == f"{pre['o_plus']} {name}" and it["reply"] == f"{pre['o_minus']} {name}"
        assert it["object_goal_overlap"] == bool(set(it["object"].split()) & set(it["goal"].split()))
        assert it["object_partner_overlap"] == bool(set(it["object"].split()) & set(it["partner"].split()))
    assert {it["prefix_candidates"]["o_minus"] for it in items} == {"Could not find enough items to craft", "Could not find"}


def test_walk_fetch_counts():
    p = next(p for p in B.all_pairs() if p.pair_id not in B.fewshot_pair_ids())
    counts = Counter(int(a.split()[1]) for seed in range(B.WALKS) for a in B.walk(p, seed) if a.startswith("get "))
    assert set(counts) == {1, 2, 3, 4}


def test_verify_reports_broken_replays(items, monkeypatch):
    """Negative tests: each kind of broken item, and a goal-dependent environment, must be reported."""
    it = next(t for t in items if t["action_kind"] == "craft" and len(t["history"]) >= 2)
    assert V.replay_problems(it) == []
    bad_hist = [list(h) for h in it["history"]]
    bad_hist[0][1] = bad_hist[0][1] + " "
    assert "history does not replay" in V.replay_problems({**it, "history": bad_hist})
    assert "failure reply does not reproduce" in V.replay_problems({**it, "reply": it["reply"] + "x"})
    assert "twin does not reproduce" in V.replay_problems({**it, "twin": "Crafted 99 minecraft:x"})
    assert "twin does not reproduce" in V.replay_problems({**it, "topup": {}})
    real = W.replay

    def goal_dependent(lines, goal, actions):
        env, hist = real(lines, goal, actions)
        if goal == W.item_id(it["partner"]) and hist:
            hist = [[hist[0][0], hist[0][1] + "!"]] + hist[1:]
        return env, hist
    monkeypatch.setattr(W, "replay", goal_dependent)
    assert "history depends on which goal is real" in V.replay_problems(it)


def test_verify_reports_broken_structure(items):
    it = next(t for t in items if t["population"] == "main")
    assert V.structure_problems(it) == []
    other = "aligned" if it["history_fit"] == "swapped" else "swapped"
    assert V.structure_problems({**it, "history_fit": other})
    assert V.structure_problems({**it, "history_fit": "tie"})
    assert V.structure_problems({**it, "prefix_candidates": {**it["prefix_candidates"], "o_plus": it["twin"]}})
    assert V.structure_problems({**it, "object_goal_overlap": not it["object_goal_overlap"]})
    assert V.structure_problems({**it, "population": "secondary_get"})
    report = V.check_items([it, {**it, "id": it["id"] + "x", "reply": "Could not find x"}], V.P.fewshot_examples())
    assert report["replay_failures"] == 1 and report["checker_failures"] == 1 and report["duplicate_history_action"] == 1


def test_builder_drops_review_1s_tag_reading_example(pairs):
    """Review 1's example (tc-p168-614ce24f7fb8): 3 warped stems held, the campfire needs 3 logs. Excluded now."""
    p = pairs["p168"]
    assert set(p.goals) == {"campfire", "light blue stained glass"}
    acts = ["get 3 warped stems", "craft 4 warped planks using 1 warped stems", "get 1 logs", "get 4 coals",
            "craft 4 warped planks using 1 warped stems", "get 4 bone", "craft 4 stick using 2 warped planks"]
    state = B.State(W.new_env(p.lines, None)[0], [], {})
    for a in acts:
        B.apply(state, a)
    excluded = Counter()
    made = B.items_at(p, 0, len(acts), state, excluded)
    assert excluded["tag_reading"] >= 1
    assert "craft 1 campfire using 3 logs, 3 stick, 1 coals" not in {it["action"] for it in made}
    assert made  # other actions at this state are still items


def test_fewshot_examples_rebuild_exactly(pairs):
    held = B.fewshot_pair_ids()
    pool = {pid: B.eligible(pairs[pid], Counter()) for pid in held}
    assert B.fewshot_examples(pool) == json.loads(V.P.FEWSHOT.read_text())


def test_replay_gate_refuses_each_failure():
    good = {"n_items": 10, "replay_failures": 0, "checker_failures": 0, "structure_failures": 0, "prompt_failures": 0,
            "duplicate_history_action": 0, "duplicate_ids": 0, "fewshot_goals_in_items": [], "main_fit_unbalanced_cells": []}
    assert V.replay_passes(good, 10)
    assert not V.replay_passes(good, 11)
    for key, bad in (("replay_failures", 1), ("checker_failures", 1), ("structure_failures", 1), ("prompt_failures", 1),
                     ("duplicate_history_action", 1), ("duplicate_ids", 1), ("fewshot_goals_in_items", ["x"]),
                     ("main_fit_unbalanced_cells", ["used_up|1-3|goal"])):
        assert not V.replay_passes({**good, key: bad}, 10), key
    report = V.check_items([it for it in V.read_items() if it["population"] == "main"][:3], V.P.fewshot_examples())
    assert report["main_fit_unbalanced_cells"]  # three items cannot be balanced in couples
