"""TextCraft environment as vendored and frozen (wf/textcraft/world.py, freeze.py, upstream/): known answers.

Every expected reply here was worked out by hand from the rules in notes/textcraft_design.md, section 2.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from wf.textcraft import build as B
from wf.textcraft import world as W
from wf.textcraft.freeze import FROZEN, LOOP_DROPS, load_frozen
from wf.textcraft.upstream.crafting_tree import CraftingTree

REPO = Path(__file__).resolve().parents[1]
STICKS = ["craft 4 stick using 2 planks", "craft 4 oak planks using 1 oak logs"]


def _play(lines, actions, goal=None):
    env, _ = W.new_env(lines, goal)
    return env, [W.step(env, a) for a in actions]


def test_frozen_database_is_sorted_and_drops_exactly_the_three_loop_losers():
    files = load_frozen()
    names = [n for n, _ in files]
    assert names == sorted(names) and len(names) == 857
    meta = json.loads(FROZEN.read_text())
    assert meta["n_upstream_files"] == 860
    assert {d for d, _, _ in LOOP_DROPS} == {"iron_ingot_from_nuggets.json", "gold_nugget.json", "honey_block.json"}
    assert not {d for d, _, _ in LOOP_DROPS} & set(names)
    assert {k for _, _, k in LOOP_DROPS} <= set(names)


def _signature(tree):
    rec = lambda d: {k: sorted(r.recipe_str for r in v) for k, v in d.items()}
    return json.dumps([rec(tree.itemid_recipes), rec(tree.tag_recipes), sorted(tree.tag_set), sorted(tree.itemid_set),
                       {k: tree.item_id_to_tag[k] for k in sorted(tree.item_id_to_tag) if "bed" not in k}], sort_keys=True)


def test_database_does_not_depend_on_file_order():
    """With the loop losers gone, reading the files in reverse builds the same recipes (bed groups are inert).

    Only the order of an item's alternative recipes follows the file order, which the frozen file fixes. craft()
    takes the first alternative that matches the inputs, so the order could matter only if two alternatives took
    the same inputs with different outputs; none do (sugar lists one recipe twice, identically).
    """
    files = load_frozen()
    a, b = W._quiet(CraftingTree, files), W._quiet(CraftingTree, list(reversed(files)))
    assert _signature(a) == _signature(b)
    for recipes in a.itemid_recipes.values():
        outs = {}
        for r in recipes:
            key = tuple(sorted((i.count, i.item_tag.name) for i in r.input_items))
            assert outs.setdefault(key, r.output_item.count) == r.output_item.count


def test_loop_directions_follow_agentgyms_released_data():
    idx = B.line_index()
    assert "craft 9 iron nugget using 1 iron ingot" in idx and "craft 1 iron ingot using 9 iron nugget" not in idx
    assert "craft 1 gold ingot using 9 gold nugget" in idx and "craft 9 gold nugget using 1 gold ingot" not in idx
    assert "craft 4 honey bottle using 1 honey block, 4 glass bottle" in idx
    assert "craft 1 honey block using 4 honey bottle" not in idx
    assert W.gettable("iron ingot") and not W.gettable("iron nugget")
    assert W.gettable("gold nugget") and not W.gettable("gold ingot")
    assert W.gettable("honey block") and not W.gettable("honey bottle")


def test_goal_pool_is_textcrafts_own():
    pool = W.goal_pool()
    assert len(pool) == 544 and all(d >= 1 for _, d in pool)
    assert ("minecraft:crafting_table", 2) in pool


def test_reset_observation_format():
    _env, obs = W.new_env(STICKS, "minecraft:stick")
    assert obs == "Crafting commands:\ncraft 4 stick using 2 planks\ncraft 4 oak planks using 1 oak logs\n\nGoal: craft stick."


def test_known_answer_game():
    actions = ["inventory", "get 1 oak logs", "craft 4 oak planks using 1 oak logs", "get 2 iron ingot", "inventory",
               "craft 4 stick using 2 oak planks", "craft 4 stick using 2 oak planks", "craft 4 stick using 2 oak planks",
               "get 1 oak planks", "get 3 planks", "craft 4 stick using 2 planks", "get 1 oak logs", "inventory"]
    _env, replies = _play(STICKS, actions)
    assert replies == [
        "Inventory: You are not carrying anything.",
        "Got 1 oak logs",
        "Crafted 4 minecraft:oak_planks",
        "Got 2 iron ingot",
        "Inventory: [oak planks] (4) [iron ingot] (2) ",
        "Crafted 4 minecraft:stick",
        "Crafted 4 minecraft:stick",
        "Could not find enough items to craft minecraft:stick",  # the planks were used up
        "Could not find oak planks",  # craftable, so it cannot be fetched
        "Could not find planks",  # a generic group
        "Could not find enough items to craft minecraft:stick",  # a generic name in a craft is never held
        "Got 1 oak logs",
        # oak planks reached zero and left; oak logs came back at the end
        "Inventory: [iron ingot] (2) [stick] (8) [oak logs] (1) ",
    ]


def test_holdings_are_checked_before_the_recipe():
    env, replies = _play(STICKS, ["craft 4 stick using 3 oak planks", "get 1 oak logs", "craft 4 oak planks using 1 oak logs",
                                  "craft 4 stick using 3 oak planks"])
    assert replies[0] == "Could not find enough items to craft minecraft:stick"
    assert replies[3].startswith("Could not find a valid recipe for ")
    assert env.inventory == {"minecraft:oak_planks": 4}


def test_get_success_format_is_the_twin_template():
    _env, replies = _play(STICKS, ["get 1 iron ingot", "get 3 oak logs"])
    assert replies == ["Got 1 iron ingot", "Got 3 oak logs"]


def test_goal_does_not_change_replies():
    acts = ["get 1 oak logs", "craft 4 oak planks using 1 oak logs", "craft 4 stick using 2 oak planks"]
    assert _play(STICKS, acts, "minecraft:stick")[1] == _play(STICKS, acts, "minecraft:oak_planks")[1] == _play(STICKS, acts)[1]


def test_clone_is_independent():
    env, _ = _play(STICKS, ["get 2 oak logs"])
    other = W.clone(env)
    W.step(other, "craft 4 oak planks using 1 oak logs")
    assert env.inventory == {"minecraft:oak_logs": 2} and other.inventory == {"minecraft:oak_logs": 1, "minecraft:oak_planks": 4}


def test_pairs_and_lists_do_not_depend_on_the_hash_seed():
    code = ("import hashlib, json; from wf.textcraft import build as B; "
            "print(hashlib.sha256(json.dumps([[p.pair_id, p.goals, p.depth, p.lines] for p in B.all_pairs()]).encode()).hexdigest())")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": s}).stdout for s in ("0", "7")}
    assert len(outs) == 1


def _json_tags() -> set[str]:
    tags = set()

    def walk(x):
        if isinstance(x, dict):
            if "tag" in x:
                tags.add(x["tag"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for _n, d in load_frozen():
        walk(d.get("key", {}))
        walk(d.get("ingredients", []))
    return tags


def test_only_planks_and_wool_stay_generic_and_fourteen_tags_become_items():
    """Design section 2 and 11 (A1): clean_up_recipes turns 14 JSON tags into plain items; planks and wool stay."""
    from wf.textcraft import checker as K
    tags = _json_tags()
    kept = {t for t in tags if W.tree().is_tag(t)}
    converted = {W.name_of(t) for t in tags - kept}
    assert kept == {"minecraft:planks", "minecraft:wool"}
    assert converted == set(B.TAG_READING) and len(converted) == 14
    assert all(W.gettable(n) for n in converted)  # fetchable, like any base item
    generic_inputs = {n for ln in B.line_index().values() for _c, n, g in ln.inputs if g}
    assert generic_inputs == {"planks", "wool"}
    assert {n: tuple(W.tag_members(n)) for n in generic_inputs} == K.GENERIC_MEMBERS
    assert "white wool" not in K.GENERIC_MEMBERS["wool"] and W.tree().is_valid_item("minecraft:white_wool")


def test_tag_reading_members_are_items_of_the_database():
    for name, members in B.TAG_READING.items():
        assert name not in members
        for m in members:
            assert W.tree().is_valid_item(W.item_id(m)), m
    assert set(B.TAG_READING["logs"]) >= {"warped stems", "jungle logs", "oak log", "stripped crimson hyphae"}
    assert B.TAG_READING["soul fire base blocks"] == () and B.TAG_READING["coals"] == ("coal",)


def test_reply_count_is_the_recipes_and_wrong_recipe_text():
    """K in "Crafted K" is the matched recipe's output count, not the action's; the wrong-recipe reply names the
    recipe object's repr (design section 11, A1)."""
    _env, replies = _play(STICKS, ["get 1 oak logs", "craft 4 oak planks using 1 oak logs", "craft 3 stick using 2 oak planks",
                                   "craft 4 stick using 1 oak planks"])
    assert replies[2] == "Crafted 4 minecraft:stick"
    assert replies[3] == ("Could not find a valid recipe for ItemTagWithCount(item_tag=ItemTag(tag=None, "
                          "item_id='minecraft:stick'), count=4)")


def test_craft_removes_inputs_before_adding_the_output():
    _env, replies = _play(["craft 4 oak planks using 1 oak logs"],
                          ["get 1 oak logs", "get 1 coal", "craft 4 oak planks using 1 oak logs", "get 1 oak logs", "inventory"])
    assert replies[-1] == "Inventory: [coal] (1) [oak planks] (4) [oak logs] (1) "


# ---------------- the frozen file is rebuilt, never committed (design section 11, A6) ----------------

def _fake_checkout(tmp_path, files, extra=()):
    folder = tmp_path / "ag" / "agentenv-textcraft" / "agentenv_textcraft" / "recipes"
    folder.mkdir(parents=True)
    for name, data in files:
        (folder / name).write_text(json.dumps(data))
    for name in extra:
        (folder / name).write_text("{}")
    return tmp_path / "ag"


def test_rebuild_reproduces_the_pinned_bytes(tmp_path):
    from wf.textcraft import freeze as F
    root = _fake_checkout(tmp_path, load_frozen(), [d for d, _, _ in LOOP_DROPS])
    out = tmp_path / "frozen.json"
    assert F.rebuild(root, out) == F.FROZEN_SHA256
    assert out.read_bytes() == FROZEN.read_bytes()
    m = json.loads((REPO / "wf/textcraft/data/manifest.json").read_text())
    assert m["frozen_db_sha256"] == F.FROZEN_SHA256 == F.sha256_text(FROZEN.read_text(encoding="utf-8"))


def test_rebuild_refuses_other_content(tmp_path):
    import pytest
    from wf.textcraft import freeze as F
    files = load_frozen()
    files = files[:1] + [(files[1][0], {**files[1][1], "type": "minecraft:crafting_shapeless_x"})] + files[2:]
    root = _fake_checkout(tmp_path, files, [d for d, _, _ in LOOP_DROPS])
    out = tmp_path / "frozen.json"
    with pytest.raises(SystemExit):
        F.rebuild(root, out)
    assert not out.exists()
    with pytest.raises(FileNotFoundError, match="wf.textcraft.freeze"):
        load_frozen(tmp_path / "absent.json")


def test_frozen_file_is_ignored_by_git():
    rel = str(FROZEN.relative_to(REPO))
    assert rel in (REPO / ".gitignore").read_text().splitlines()
    if (REPO / ".git").exists():  # the mutation check runs on a copy without git
        tracked = subprocess.run(["git", "ls-files", rel], cwd=REPO, capture_output=True, text=True, check=True)
        assert tracked.stdout.strip() == ""
