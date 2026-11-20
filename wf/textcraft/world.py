"""The TextCraft environment as the item builder uses it: AgentGym's code (vendored) on the frozen recipe database.

Everything here asks the environment. The text-only checker (wf/textcraft/checker.py) must never import this.
"""
from __future__ import annotations

import contextlib
import io
from functools import lru_cache

from wf.textcraft.freeze import load_frozen
from wf.textcraft.upstream.crafting_tree import CraftingTree
from wf.textcraft.upstream.environment import TextCraftEnv
from wf.textcraft.upstream.utils import item_id_to_str


def _quiet(fn, *args):
    """Upstream prints on cycles and on wrong item counts; keep that out of the logs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


@lru_cache(maxsize=1)
def tree() -> CraftingTree:
    return _quiet(CraftingTree, load_frozen())


def item_id(name: str) -> str:
    """"oak planks" -> "minecraft:oak_planks", the environment's own conversion (item_str_to_obj)."""
    return "minecraft:" + name.replace(" ", "_")


def name_of(iid: str) -> str:
    return item_id_to_str(iid)


@lru_cache(maxsize=1)
def goal_pool() -> tuple[tuple[str, int], ...]:
    """TextCraft's own goal set (reset() draws from item_recipes_min_depth(1)): (item id, crafting depth)."""
    return tuple(sorted(_quiet(lambda: list(tree().item_recipes_min_depth(1))), key=lambda x: (x[1], x[0])))


@lru_cache(maxsize=None)
def recipe_tree(goal_iid: str) -> tuple[str, ...]:
    """The displayed lines of the goal's full recipe tree (create_recipe_set's `recipes`, without distractors)."""
    recipes = _quiet(tree().traverse_recipe_tree, goal_iid, set())
    return tuple(sorted({r.recipe_str for r in recipes}))


@lru_cache(maxsize=None)
def tree_craftables(goal_iid: str) -> frozenset[str]:
    """Item ids crafted anywhere in the goal's recipe tree (every alternative recipe, tag members included)."""
    recipes = _quiet(tree().traverse_recipe_tree, goal_iid, set())
    return frozenset(r.output_item.item_tag.item_id for r in recipes)


def is_tag(name: str) -> bool:
    return tree().is_tag(item_id(name))


def gettable(name: str) -> bool:
    """The environment's get succeeds on exactly these names: valid items that are neither craftable nor a tag."""
    i = item_id(name)
    return tree().is_valid_item(i) and not tree().is_craftable(i) and not tree().is_tag(i)


def tag_members(name: str) -> list[str]:
    """Concrete items that may stand for a generic ingredient (environment's item_id_to_tag), sorted names."""
    return sorted(name_of(i) for i in tree().get_items_with_tags(item_id(name)))


def new_env(lines: list[str] | tuple[str, ...], goal_iid: str | None) -> tuple[TextCraftEnv, str]:
    """A fresh game showing exactly `lines`, through the environment's own reset(commands, goal)."""
    env = TextCraftEnv(crafting_tree=tree(), commands=None, goal=None)
    obs, _ = env.reset(commands="\n".join(lines), goal=goal_iid)
    return env, obs


def step(env: TextCraftEnv, action: str) -> str:
    obs, _reward, _done, _trunc, _info = _quiet(env.step, action)
    return obs


def clone(env: TextCraftEnv) -> TextCraftEnv:
    """The same game in the same state; the inventory is the environment's only state."""
    other = TextCraftEnv(crafting_tree=tree(), commands=env.commands, goal=env.goal)
    other.inventory = dict(env.inventory)
    return other


def replay(lines: list[str], goal_iid: str | None, actions: list[str]) -> tuple[TextCraftEnv, list[list[str]]]:
    env, _ = new_env(lines, goal_iid)
    return env, [[a, step(env, a)] for a in actions]
