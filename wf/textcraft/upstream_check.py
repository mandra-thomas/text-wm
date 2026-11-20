"""One-off check that the vendored environment on the frozen database is AgentGym's (design section 7, item 4).

Run with a Python that has AgentGym's TextCraft dependencies (gymnasium, numpy), from the repository root:

    <python> wf/textcraft/upstream_check.py <AgentGym checkout at 3ef9235>

It imports the unmodified upstream package, gives its loader the frozen listing (sorted file names, the three
loop files dropped) in place of os.listdir, and checks (1) that it builds the same database as the vendored copy
and (2) that every item's history, failure reply and twin come out the same. Writes wf/textcraft/data/upstream_check.json.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wf.textcraft import world as W  # noqa: E402  (stdlib-only: the vendored copy and the frozen file)
from wf.textcraft.freeze import FROZEN, LOOP_DROPS, RECIPES_SUBDIR, UPSTREAM_COMMIT  # noqa: E402


def _upstream(root: Path):
    pkg = root / RECIPES_SUBDIR.rsplit("/", 1)[0]
    stub = types.ModuleType("agentenv_textcraft")
    stub.__path__ = [str(pkg)]
    sys.modules["agentenv_textcraft"] = stub
    import agentenv_textcraft.crafting_tree as ct
    from agentenv_textcraft.environment import TextCraftEnv

    dropped = {d for d, _, _ in LOOP_DROPS}
    real = os.listdir
    ct.os.listdir = lambda p: sorted(n for n in real(p) if n not in dropped)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tree = ct.CraftingTree(minecraft_dir=str(pkg) + "/")
    finally:
        ct.os.listdir = real
    return tree, TextCraftEnv


def _signature(tree) -> dict:
    recipes = lambda d: {k: [r.recipe_str for r in v] for k, v in d.items()}
    return {"itemid_recipes": recipes(tree.itemid_recipes), "tag_recipes": recipes(tree.tag_recipes),
            "item_id_to_tag": dict(tree.item_id_to_tag), "tag_set": sorted(tree.tag_set),
            "itemid_set": sorted(tree.itemid_set)}


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    tree, Env = _upstream(root)
    tree_identical = json.dumps(_signature(tree), sort_keys=True) == json.dumps(_signature(W.tree()), sort_keys=True)
    items_path = REPO / "wf" / "textcraft" / "data" / "items.jsonl"
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]

    def step(env, a):
        with contextlib.redirect_stdout(io.StringIO()):
            return env.step(a)[0]

    mismatches, examples = 0, []
    for it in items:
        env = Env(crafting_tree=tree, commands=None, goal=None)
        env.reset(commands="\n".join(it["recipe_lines"]), goal=it["env_goal"])
        history = [[a, step(env, a)] for a, _ in it["history"]]
        saved = dict(env.inventory)
        reply = step(env, it["action"])
        env.inventory = dict(saved)
        for name, n in it["topup"].items():
            key = "minecraft:" + name.replace(" ", "_")
            env.inventory[key] = env.inventory.get(key, 0) + n
        for name, n in it.get("removal", {}).items():  # a success's failure twin (design 12, C3)
            key = "minecraft:" + name.replace(" ", "_")
            env.inventory[key] = env.inventory.get(key, 0) - n
        twin = step(env, it["action"]) if it["action_kind"] == "craft" else it["twin"]
        if history != it["history"] or reply != it["reply"] or twin != it["twin"]:
            mismatches += 1
            examples.append(it["id"])
    out = {"upstream_commit": UPSTREAM_COMMIT, "python": sys.version.split()[0],
           "items_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
           "frozen_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
           "tree_identical": tree_identical, "n_items": len(items), "mismatches": mismatches, "examples": examples[:10]}
    (REPO / "wf" / "textcraft" / "data" / "upstream_check.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out))
    return 0 if tree_identical and not mismatches else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
