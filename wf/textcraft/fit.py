"""History-fit measures for TextCraft items, from the prompt text only (notes/textcraft_design.md, section 12, B1).

Each measure asks how far the visible history works toward the aligned goal rather than the partner. It returns two
non-negative numbers (a, p): the amount that speaks for the aligned goal and the amount that speaks for the partner.
Its sign, +1 when a > p, -1 when p > a and 0 otherwise, says which goal the history favours on that measure. Every
measure is antisymmetric: exchanging the goal and the partner exchanges a and p.

Like the checker, this module reads only the recipe lines, the history's (action, reply) pairs and the two goal
names, and imports no environment code (a test enforces this).
"""
from __future__ import annotations

from wf.textcraft.checker import GENERIC_MEMBERS, _inventory, account, from_id, parse_craft, step_object, tree_names

MEASURES = ("steps", "quantity", "crafts", "distinct_names", "first_step", "last_step", "held_now", "completion",
            "goal_words", "last_inventory")


def _exclusive(recipe_lines: list[str], goal: str, partner: str) -> tuple[set[str], set[str]]:
    """Names on the goal's tree lines and not the partner's, and the other way round (design 11, A3)."""
    a, p = tree_names(recipe_lines, goal), tree_names(recipe_lines, partner)
    return a - p, p - a


def _count(reply: str) -> int:
    """N of "Got N X", K of "Crafted K minecraft:Y_"; 0 for any other reply."""
    if reply.startswith(("Got ", "Crafted ")):
        return int(reply.split(" ", 2)[1])
    return 0


def _completion(recipe_lines: list[str], held: dict[str, int], goal: str) -> float:
    """The best displayed line for the goal: mean over its inputs of min(held, need) / need; a generic input
    (planks, wool) counts every member held."""
    best = 0.0
    for line in recipe_lines:
        _k, out, inputs = parse_craft(line)
        if out != goal:
            continue
        have = lambda n: held.get(n, 0) + sum(held.get(m, 0) for m in GENERIC_MEMBERS.get(n, ()))
        best = max(best, sum(min(have(n), c) / c for c, n in inputs) / len(inputs))
    return best


def _words(history: list) -> list[str]:
    """The history's text as words: every action and reply, square brackets removed, "minecraft:x_y" written
    "x y", split on spaces."""
    out = []
    for action, reply in history:
        for text in (action, reply):
            text = text.replace("[", " ").replace("]", " ")
            out += [from_id(w) if w.startswith("minecraft:") else w for w in text.split()]
    return " ".join(out).split()


def measures(recipe_lines: list[str], history: list, goal: str, partner: str) -> dict[str, list]:
    """Every measure as [a, p] (design section 12, B1)."""
    ea, ep = _exclusive(recipe_lines, goal, partner)
    steps = [(step_object(reply), _count(reply), reply.startswith("Crafted ")) for _a, reply in history]
    tagged = [(o, n, c) for o, n, c in steps if o in ea or o in ep]
    side = lambda o: [int(o in ea), int(o in ep)]
    held = account(history).held
    words = _words(history)
    wa, wp = set(goal.split()) - set(partner.split()), set(partner.split()) - set(goal.split())
    last_inv = next((reply for _a, reply in reversed(history) if reply.startswith("Inventory: [")), None)
    listed = list(_inventory(last_inv)) if last_inv else []  # the last inventory reply that lists something
    return {
        "steps": [sum(o in ea for o, _n, _c in steps), sum(o in ep for o, _n, _c in steps)],
        "quantity": [sum(n for o, n, _c in steps if o in ea), sum(n for o, n, _c in steps if o in ep)],
        "crafts": [sum(o in ea for o, _n, c in steps if c), sum(o in ep for o, _n, c in steps if c)],
        "distinct_names": [len({o for o, _n, _c in steps} & ea), len({o for o, _n, _c in steps} & ep)],
        "first_step": side(tagged[0][0]) if tagged else [0, 0],
        "last_step": side(tagged[-1][0]) if tagged else [0, 0],
        "held_now": [sum(v for n, v in held.items() if n in ea), sum(v for n, v in held.items() if n in ep)],
        "completion": [_completion(recipe_lines, held, goal), _completion(recipe_lines, held, partner)],
        "goal_words": [sum(w in wa for w in words), sum(w in wp for w in words)],
        "last_inventory": [sum(n in ea for n in listed), sum(n in ep for n in listed)],
    }


def sign(ap: list) -> int:
    a, p = ap
    return 1 if a > p else -1 if p > a else 0


def signs(fit_measures: dict[str, list]) -> dict[str, int]:
    return {m: sign(fit_measures[m]) for m in MEASURES}


def none_favours_aligned(fit_measures: dict[str, list]) -> bool:
    """Every measure is neutral or favours the partner: the history never speaks for the aligned goal."""
    return all(s <= 0 for s in signs(fit_measures).values())
