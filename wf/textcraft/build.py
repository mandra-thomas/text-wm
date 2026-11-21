"""Build the TextCraft items (notes/textcraft_design.md, sections 3 and 4).

    python -m wf.textcraft.build            # writes wf/textcraft/data/items.jsonl and fewshot.json

Every reply stored here comes from the environment (wf/textcraft/world.py). The failure class is computed from the
environment's own inventory; the text-only checker recomputes it from the visible text in verify.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from wf import constants as C
from wf.textcraft import fit as F
from wf.textcraft import world as W

SEED = C.RNG_SEED
DATA = Path(__file__).resolve().parent / "data"
ITEMS = DATA / "items.jsonl"
FEWSHOT = DATA / "fewshot.json"

EXCLUDED_GOALS = ("blue dye", "dark oak planks", "golden carrot")  # named in the fixed instruction text
WALKS, STEPS = 24, 12  # 6 walks before section 12, B1; the prompt examples still come from the first 6
FEWSHOT_WALKS = 6
P_INVENTORY, P_CRAFT = 0.10, 0.55
GET_COUNTS = (1, 2, 3, 4)
N_FEWSHOT_PAIRS = 3
CLASSES = ("used_up", "too_few", "never_obtained")
BINS = {"0": (0, 0), "1-3": (1, 3), "4-8": (4, 8), "9-12": (9, 12)}
QUOTAS = {  # population -> (total, classes, bins); main, usefulness and success are drawn in same-history couples
    "main": (1500, CLASSES, ("1-3", "4-8", "9-12")),
    "usefulness": (600, ("never_obtained",), ("1-3", "4-8", "9-12")),
    "success": (600, ("success",), ("1-3", "4-8", "9-12")),
    "secondary_get": (400, ("get_craftable",), ("0", "1-3", "4-8", "9-12")),
    "empty_history": (100, ("never_obtained",), ("0",)),
}
NEUTRAL_LINE = "Goal: explore the crafting commands."
FITS = ("aligned", "swapped", "tie")

_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark oak")
_PER_WOOD = {**{f"{w} logs": (f"{w} log", f"{w} wood", f"stripped {w} log", f"stripped {w} wood") for w in _WOODS},
             **{f"{s} stems": (f"{s} stem", f"{s} hyphae", f"stripped {s} stem", f"stripped {s} hyphae")
                for s in ("crimson", "warped")}}
# The 14 JSON tags that upstream's clean_up_recipes turns into plain, fetchable items (design section 2), each with
# the items of the frozen database that belong to the same Minecraft tag. A reader who knows Minecraft may take a
# held member to fill the name, so a test action whose short input is one of these names while a member is held
# is excluded (design section 11, A1). Members absent from the database (charcoal, soul sand, soul soil, cobbled
# deepslate) are left out; a test checks that every listed member is a valid item.
TAG_READING: dict[str, tuple[str, ...]] = {
    **_PER_WOOD,
    "logs": tuple(sorted(set(_PER_WOOD) | {m for ms in _PER_WOOD.values() for m in ms})),
    "coals": ("coal",),
    "stone crafting materials": ("blackstone", "cobblestone"),
    "stone tool materials": ("blackstone", "cobblestone"),
    "wooden slabs": tuple(f"{w} slab" for w in _WOODS + ("crimson", "warped")),
    "soul fire base blocks": (),
}
# Items of the frozen database whose names differ only by a final "s" (design section 12, B5). A reader may take a
# held "nether bricks" for "nether brick", so a test action whose short input has its twin held is excluded; a test
# checks that this is every such pair in the database.
PLURAL_TWINS: tuple[tuple[str, str], ...] = (
    ("acacia log", "acacia logs"), ("birch log", "birch logs"), ("brick", "bricks"), ("coal", "coals"),
    ("crimson stem", "crimson stems"), ("dark oak log", "dark oak logs"), ("jungle log", "jungle logs"),
    ("nether brick", "nether bricks"), ("oak log", "oak logs"), ("spruce log", "spruce logs"),
    ("warped stem", "warped stems"))
_TWIN_OF = {**{a: b for a, b in PLURAL_TWINS}, **{b: a for a, b in PLURAL_TWINS}}


def goal_line(goal: str) -> str:
    return f"Goal: craft {goal}."


def length_bin(n: int) -> str:
    return next(b for b, (lo, hi) in BINS.items() if lo <= n <= hi)


# ---------------- pairs and lists ----------------


@dataclass(frozen=True)
class Pair:
    pair_id: str
    goals: tuple[str, str]  # names, sorted
    depth: int
    lines: tuple[str, ...]  # the displayed list, in display order


def compatible(a_iid: str, b_iid: str) -> bool:
    """Partners share no craftable item (so neither is in the other's tree); depth is matched by the caller."""
    return not (W.tree_craftables(a_iid) & W.tree_craftables(b_iid))


def union_list(a: str, b: str) -> tuple[str, ...]:
    """The two trees' lines, sorted then shuffled by a seed of the unordered pair: it never marks the real goal."""
    a, b = sorted((a, b))
    lines = sorted(set(W.recipe_tree(W.item_id(a))) | set(W.recipe_tree(W.item_id(b))))
    random.Random(f"{SEED}|textcraft|list|{a}|{b}").shuffle(lines)
    return tuple(lines)


@lru_cache(maxsize=1)
def all_pairs() -> tuple[Pair, ...]:
    by_depth: dict[int, list[str]] = defaultdict(list)
    for iid, depth in W.goal_pool():
        if W.name_of(iid) not in EXCLUDED_GOALS:
            by_depth[depth].append(iid)
    pairs = []
    for depth in sorted(by_depth):
        order = sorted(by_depth[depth])
        random.Random(f"{SEED}|textcraft|pairs|{depth}").shuffle(order)
        taken: set[str] = set()
        for i, a in enumerate(order):
            if a in taken:
                continue
            b = next((x for x in order[i + 1:] if x not in taken and compatible(a, x)), None)
            if b is None:
                continue
            taken |= {a, b}
            names = tuple(sorted((W.name_of(a), W.name_of(b))))
            pairs.append(Pair(f"p{len(pairs):03d}", names, depth, union_list(*names)))
    return tuple(pairs)


@lru_cache(maxsize=1)
def fewshot_pair_ids() -> tuple[str, ...]:
    pairs = all_pairs()
    median = statistics.median(len(p.lines) for p in pairs)
    short = [p.pair_id for p in pairs if len(p.lines) <= median]
    return tuple(sorted(random.Random(f"{SEED}|textcraft|fewshot").sample(short, N_FEWSHOT_PAIRS)))


# ---------------- recipe lines ----------------


@dataclass(frozen=True)
class Line:
    text: str
    out_count: int
    out: str
    inputs: tuple[tuple[int, str, bool], ...]  # (count, name, generic?)


@lru_cache(maxsize=None)
def line_index() -> dict[str, Line]:
    """Every displayable line, parsed from the environment's own Recipe objects (no text parsing)."""
    out: dict[str, Line] = {}
    for recipes in list(W.tree().itemid_recipes.values()) + list(W.tree().tag_recipes.values()):
        for r in recipes:
            ins = tuple((i.count, W.name_of(i.item_tag.name), W.is_tag(W.name_of(i.item_tag.name))) for i in r.input_items)
            out[r.recipe_str] = Line(r.recipe_str, r.output_item.count, W.name_of(r.output_item.item_tag.item_id), ins)
    return out


def lines_of(pair: Pair) -> list[Line]:
    idx = line_index()
    return [idx[t] for t in pair.lines]


def render(line: Line, inputs: list[tuple[int, str]]) -> str:
    """The environment's recipe_str format with specific kinds filled in."""
    return f"craft {line.out_count} {line.out} using " + ", ".join(f"{c} {n}" for c, n in inputs)


def bases(pair: Pair) -> list[str]:
    return sorted({n for ln in lines_of(pair) for _c, n, generic in ln.inputs if not generic and W.gettable(n)})


# ---------------- states and walks ----------------


@dataclass
class State:
    env: object
    history: list[list[str]]
    peak: dict[str, int]  # item id -> largest count held so far
    trace: list[dict[str, int]] = field(default_factory=list)  # the inventory after each history step

    def held(self, name: str) -> int:
        return self.env.inventory.get(W.item_id(name), 0)

    def peaked(self, name: str) -> int:
        return self.peak.get(W.item_id(name), 0)


def apply(state: State, action: str) -> str:
    reply = W.step(state.env, action)
    state.history.append([action, reply])
    for k, v in state.env.inventory.items():
        state.peak[k] = max(state.peak.get(k, 0), v)
    state.trace.append(dict(state.env.inventory))
    return reply


def available_crafts(pair: Pair, state: State) -> list[str]:
    """Crafts the walk may take: output not a goal, every input held; a generic input takes the kind held most."""
    out = set()
    for ln in lines_of(pair):
        if ln.out in pair.goals:
            continue
        filled = []
        for c, n, generic in ln.inputs:
            if generic:
                ok = [m for m in W.tag_members(n) if state.held(m) >= c]
                if not ok:
                    break
                filled.append((c, min(ok, key=lambda m: (-state.held(m), m))))
            elif state.held(n) >= c:
                filled.append((c, n))
            else:
                break
        else:
            out.add(render(ln, filled))
    return sorted(out)


def walk(pair: Pair, seed: int) -> list[str]:
    """The goal-blind exploration policy (design section 3). It sees the two goals only as protected outputs."""
    rng = random.Random(f"{SEED}|textcraft|walk|{pair.pair_id}|{seed}")
    state = State(W.new_env(pair.lines, None)[0], [], {})
    base = bases(pair)
    actions = []
    for _ in range(STEPS):
        u = rng.random()
        if u < P_INVENTORY:
            action = "inventory"
        else:
            crafts = available_crafts(pair, state)
            if u < P_CRAFT and crafts:
                action = rng.choice(crafts)
            else:
                action = f"get {rng.choice(GET_COUNTS)} {rng.choice(base)}"
        reply = apply(state, action)
        if not reply.startswith(("Got ", "Crafted ", "Inventory: ")):
            raise RuntimeError(f"{pair.pair_id}/{seed}: walk action failed: {action!r} -> {reply!r}")
        actions.append(action)
    return actions


def states_of(pair: Pair, actions: list[str]):
    """(prefix length, State) for every prefix 0..len(actions), each a fresh game replayed from the start."""
    state = State(W.new_env(pair.lines, None)[0], [], {})
    yield 0, _copy(state)
    for t, a in enumerate(actions, 1):
        apply(state, a)
        yield t, _copy(state)


def _copy(state: State) -> State:
    return State(W.clone(state.env), [list(h) for h in state.history], dict(state.peak), [dict(t) for t in state.trace])


# ---------------- candidate actions ----------------


def fill_generic(name: str, state: State, displayed: set[str]) -> str:
    """A specific kind for a generic ingredient in a test action (design section 3)."""
    members = W.tag_members(name)
    held = [m for m in members if state.held(m) > 0]
    if held:
        return min(held, key=lambda m: (-state.held(m), m))
    peaked = [m for m in members if state.peaked(m) > 0]
    if peaked:
        return min(peaked, key=lambda m: (-state.peaked(m), m))
    shown = [m for m in members if m in displayed]
    return (shown or members)[0]


def filled(line: Line, state: State, displayed: set[str]) -> list[tuple[int, str]]:
    return [(c, fill_generic(n, state, displayed) if generic else n) for c, n, generic in line.inputs]


def targets(pair: Pair, goal: str, state: State) -> list[tuple[str, str]]:
    """(object, role): the goal, then its craftable specific direct ingredients, sorted."""
    displayed = {ln.out for ln in lines_of(pair)}
    ingredients = set()
    for ln in lines_of(pair):
        if ln.out == goal:
            for _c, n in filled(ln, state, displayed):
                if W.tree().is_craftable(W.item_id(n)):
                    ingredients.add(n)
    return [(goal, "goal")] + [(n, "ingredient") for n in sorted(ingredients - {goal})]


def candidates(pair: Pair, goal: str, state: State) -> list[tuple[str, str, str, Line | None]]:
    """(action, object, role, line) for each craft of a displayed line for the object, then `get 1 <object>`."""
    displayed = {ln.out for ln in lines_of(pair)}
    out = []
    for obj, role in targets(pair, goal, state):
        for ln in sorted((ln for ln in lines_of(pair) if ln.out == obj), key=lambda ln: ln.text):
            out.append((render(ln, filled(ln, state, displayed)), obj, role, ln))
        out.append((f"get 1 {obj}", obj, role, None))
    return out


def classify(state: State, inputs: list[tuple[int, str]]) -> tuple[str | None, list[dict]]:
    """Failure class from the environment's inventory and peak holdings (design section 3)."""
    short = []
    for c, n in inputs:
        held, peak = state.held(n), state.peaked(n)
        if held < c:
            kind = "used_up" if peak >= c else "too_few" if peak > 0 else "never_obtained"
            short.append({"name": n, "need": c, "held": held, "peak": peak, "kind": kind})
    kinds = {s["kind"] for s in short}
    return next((k for k in CLASSES if k in kinds), None), short


def tag_reading_ambiguous(state: State, short: list[dict]) -> bool:
    """A short input is one of the 14 tag-derived names while a Minecraft member of it is held (design 11, A1)."""
    return any(state.held(m) > 0 for s in short for m in TAG_READING.get(s["name"], ()))


def plural_ambiguous(state: State, short: list[dict]) -> bool:
    """A short input's singular or plural twin is held (design section 12, B5)."""
    return any(s["name"] in _TWIN_OF and state.held(_TWIN_OF[s["name"]]) > 0 for s in short)


def certificate(state: State, short: list[dict]) -> list[dict]:
    """For each short input, from the environment's inventory after each step: the last history step that changed
    its count (which set the holding now) and the first step at which it reached its peak; "never" when no step
    did (design section 12, B4). Steps count from 1."""
    out = []
    for s in short:
        i = W.item_id(s["name"])
        counts = [0] + [inv.get(i, 0) for inv in state.trace]
        changed = [k for k in range(1, len(counts)) if counts[k] != counts[k - 1]]
        peak_at = next((k for k in range(1, len(counts)) if counts[k] == s["peak"]), None) if s["peak"] > 0 else None
        out.append({"name": s["name"], "held_step": changed[-1] if changed else "never",
                    "peak_step": "never" if peak_at is None else peak_at})
    return out


def inventory_shown_after_last_change(history: list) -> bool:
    """Some history step is an inventory look-up and no later step got or crafted anything: the prompt shows the
    inventory as it is before the action (design section 12, B6)."""
    last_change = max((k for k, (_a, o) in enumerate(history) if o.startswith(("Got ", "Crafted "))), default=-1)
    return any(o.startswith("Inventory: ") for _a, o in history[last_change + 1:])


@lru_cache(maxsize=None)
def tree_names(goal: str) -> frozenset[str]:
    """Every name on the lines of the goal's recipe tree: outputs and inputs (design section 11, A3)."""
    idx = line_index()
    out: set[str] = set()
    for text in W.recipe_tree(W.item_id(goal)):
        ln = idx[text]
        out.add(ln.out)
        out |= {n for _c, n, _g in ln.inputs}
    return frozenset(out)


def step_object(reply: str) -> str | None:
    """The item a successful history step produced: the fetched item, or the crafted output."""
    if reply.startswith("Got "):
        return reply.split(" ", 2)[2]
    if reply.startswith("Crafted "):
        return W.name_of(reply.split(" ", 2)[2])
    return None


def history_fit(history: list, goal: str, partner: str) -> tuple[str, dict[str, int]]:
    """Which goal the history works toward: steps whose object is in only that goal's tree names (design 11, A3)."""
    a, p = tree_names(goal), tree_names(partner)
    objs = [step_object(o) for _a, o in history]
    n = {"aligned": sum(o in a - p for o in objs), "swapped": sum(o in p - a for o in objs)}
    fit = "aligned" if n["aligned"] > n["swapped"] else "swapped" if n["swapped"] > n["aligned"] else "tie"
    return fit, n


def shares_word(a: str, b: str) -> bool:
    return bool(set(a.split(" ")) & set(b.split(" ")))


def prefix_candidates(action_kind: str, twin: str) -> dict[str, str]:
    """The two replies with the object name removed (design section 11, A4)."""
    if action_kind == "craft":
        return {"o_plus": " ".join(twin.split(" ")[:2]), "o_minus": "Could not find enough items to craft"}
    return {"o_plus": " ".join(twin.split(" ")[:2]), "o_minus": "Could not find"}


def removed_reply(state: State, removal: dict[str, int], action: str) -> str:
    """The reply with these amounts taken out of the inventory first: the failure twin of a success (12, C3)."""
    env = W.clone(state.env)
    for name, n in removal.items():
        env.inventory[W.item_id(name)] = env.inventory.get(W.item_id(name), 0) - n
    return W.step(env, action)


def topped_up_reply(state: State, short: list[dict], action: str) -> str:
    env = W.clone(state.env)
    for s in short:
        env.inventory[W.item_id(s["name"])] = env.inventory.get(W.item_id(s["name"]), 0) + s["need"] - s["held"]
    return W.step(env, action)


def item_id_for(pair_id: str, history: list, action: str) -> str:
    h = hashlib.sha256(json.dumps([pair_id, history, action]).encode()).hexdigest()[:12]
    return f"tc-{pair_id}-{h}"


def items_at(pair: Pair, seed: int, t: int, state: State, excluded: Counter) -> list[dict]:
    out = []
    for goal in pair.goals:
        partner = next(g for g in pair.goals if g != goal)
        for action, obj, role, line in candidates(pair, goal, state):
            reply = W.step(W.clone(state.env), action)
            base = {"pair_id": pair.pair_id, "game_id": pair.pair_id, "goal": goal, "partner": partner,
                    "depth": pair.depth, "recipe_lines": list(pair.lines), "env_goal": W.item_id(goal),
                    "walk_seed": seed, "history": [list(h) for h in state.history], "history_len": t,
                    "length_bin": length_bin(t), "action": action, "object": obj, "role": role,
                    "reply": reply, "task_line": goal_line(goal), "swapped_task_line": goal_line(partner),
                    "neutral_task_line": NEUTRAL_LINE}
            if line is None:
                if reply.startswith("Got "):
                    excluded["success_get"] += 1
                    continue
                if reply != f"Could not find {obj}":
                    raise RuntimeError(f"unexpected get reply {reply!r}")
                item = {**base, "action_kind": "get", "failure_class": "get_craftable", "short_inputs": [],
                        "twin": f"Got 1 {obj}", "twin_source": "template", "topup": {},
                        "population": "secondary_get"}
            else:
                inputs = filled(line, state, {ln.out for ln in lines_of(pair)})
                if any(W.is_tag(n) for _c, n in inputs):
                    excluded["category_literal"] += 1
                    continue
                if reply.startswith("Crafted "):  # a determined success (section 12, C3)
                    if t == 0:
                        excluded["success_empty_history"] += 1
                        continue
                    held = [{"name": n, "need": c, "held": state.held(n), "peak": state.peaked(n)} for c, n in inputs]
                    removal = {h["name"]: h["held"] for h in held}
                    twin = removed_reply(state, removal, action)
                    if not twin.startswith("Could not find enough items to craft "):
                        raise RuntimeError(f"emptied inputs still craft: {action!r} -> {twin!r}")
                    item = {**base, "action_kind": "craft", "failure_class": "success", "short_inputs": [],
                            "twin": twin, "twin_source": "env_remove", "topup": {}, "removal": removal,
                            "population": "success", "pure_class": None, "needs_arithmetic": None,
                            "certificate": certificate(state, held)}
                elif not reply.startswith("Could not find enough items to craft "):
                    excluded["wrong_recipe"] += 1
                    continue
                else:
                    item = _failure_item(base, state, inputs, action, t, excluded)
                    if item is None:
                        continue
            if line is None:
                item["pure_class"] = item["needs_arithmetic"] = None
                item["certificate"] = [{"name": obj, "line": 1 + next(
                    k for k, ln in enumerate(lines_of(pair)) if ln.out == obj)}]
            item.setdefault("removal", {})
            success = item["population"] == "success"
            item["candidates"] = {"o_plus": item["reply"], "o_minus": item["twin"]} if success else \
                {"o_plus": item["twin"], "o_minus": item["reply"]}
            item["prefix_candidates"] = prefix_candidates(item["action_kind"], item["candidates"]["o_plus"])
            item["history_fit"], item["history_fit_counts"] = history_fit(state.history, goal, partner)
            item["object_goal_overlap"] = shares_word(obj, goal)
            item["object_partner_overlap"] = shares_word(obj, partner)
            item["id"] = item_id_for(pair.pair_id, item["history"], action)
            item["inventory_shown_after_last_change"] = inventory_shown_after_last_change(state.history)
            item["fit_measures"] = F.measures(list(pair.lines), state.history, goal, partner)
            out.append(item)
    return out


def _failure_item(base: dict, state: State, inputs: list, action: str, t: int, excluded: Counter) -> dict | None:
    """A craft failure item, or None (counted) when it is a wrong recipe or reading-ambiguous."""
    cls, short = classify(state, inputs)
    twin = topped_up_reply(state, short, action)
    if not twin.startswith("Crafted "):
        excluded["wrong_recipe"] += 1
        return None
    if tag_reading_ambiguous(state, short):
        excluded["tag_reading"] += 1
        return None
    if plural_ambiguous(state, short):
        excluded["plural_reading"] += 1
        return None
    return {**base, "action_kind": "craft", "failure_class": cls, "short_inputs": short,
            "twin": twin, "twin_source": "env_topup", "topup": {s["name"]: s["need"] - s["held"] for s in short},
            "population": "main" if t >= 1 else "empty_history",
            "pure_class": len({s["kind"] for s in short}) == 1,
            "needs_arithmetic": all(s["peak"] > 0 for s in short),
            "certificate": certificate(state, short)}


def eligible(pair: Pair, excluded: Counter, walks: int | None = None) -> list[dict]:
    """Every deduplicated item of one pair, in walk order (seed, then prefix length)."""
    seen, out = set(), []
    for seed in range(WALKS if walks is None else walks):
        for t, state in states_of(pair, walk(pair, seed)):
            for it in items_at(pair, seed, t, state, excluded):
                key = (it["pair_id"], json.dumps(it["history"]), it["action"])
                if key in seen:
                    excluded["duplicate"] += 1
                    continue
                seen.add(key)
                out.append(it)
    return out


# ---------------- selection ----------------


def _split(total: int, avail: dict[str, int]) -> dict[str, int]:
    """Equal shares, a short key taking all it has and its remainder shared by the keys with spare, repeated."""
    take = {k: 0 for k in avail}
    left = total
    while left > 0:
        open_ = [k for k in avail if take[k] < avail[k]]
        if not open_:
            break
        share, extra = divmod(left, len(open_))
        for i, k in enumerate(open_):  # a remainder smaller than the number of keys goes to the first keys
            got = min(share + (1 if i < extra else 0), avail[k] - take[k])
            take[k] += got
            left -= got
    return take


def _round_robin(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_pair[it["pair_id"]].append(it)
    order = sorted(by_pair)
    rng.shuffle(order)
    chosen = []
    while len(chosen) < n:
        for p in order:
            if by_pair[p] and len(chosen) < n:
                chosen.append(by_pair[p].pop(rng.randrange(len(by_pair[p]))))
        if not any(by_pair.values()):
            break
    return chosen


ROLES = ("goal", "ingredient")
COUPLE_FITS = ("aligned", "swapped")


def couple_id(pair_id: str, history: list, failure_class: str, role: str) -> str:
    h = hashlib.sha256(json.dumps([pair_id, history, failure_class, role]).encode()).hexdigest()[:12]
    return f"cp-{pair_id}-{h}"


def slot_key(it: dict) -> tuple[str, str, str, str]:
    return it["pair_id"], json.dumps(it["history"]), it["failure_class"], it["role"]


def usefulness_only(it: dict) -> bool:
    """An ingredient whose name shares no word with either goal: the goal line can matter only through what the
    object is for (design section 12, C2)."""
    return it["role"] == "ingredient" and not it["object_goal_overlap"] and not it["object_partner_overlap"]


def couple_slots(pool: list[dict], population: str = "main", keep=None, exclude: frozenset = frozenset()
                 ) -> dict[tuple[str, str, str], dict[str, list[tuple[list[dict], list[dict]]]]]:
    """(class, bin, role) -> pair -> slots. A slot is one visible history of the pair at which both goals have an
    item of this population, class and role: (the first goal's items, the second goal's items), goals in sorted
    order, items sorted by id (design section 12, B1). `keep` filters items; slots in `exclude` are skipped."""
    by: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for it in pool:
        if it["population"] == population and (keep is None or keep(it)) and slot_key(it) not in exclude:
            by[slot_key(it)][it["goal"]].append(it)
    cells: dict = defaultdict(lambda: defaultdict(list))
    for (pid, _h, c, r), goals in sorted(by.items()):
        if len(goals) == 2:
            first, second = (sorted(goals[g], key=lambda it: it["id"]) for g in sorted(goals))
            cells[(c, first[0]["length_bin"], r)][pid].append((first, second))
    return cells


def _draw_couples(slots: dict[str, list], n: int, rng: random.Random, population: str = "main") -> list[dict]:
    """n couples, round-robin over goal pairs (shuffled once), a random slot of each pair per round, then a random
    item of each goal in the slot."""
    by_pair = {p: list(v) for p, v in slots.items()}
    order = sorted(by_pair)
    rng.shuffle(order)
    chosen: list[dict] = []
    while len(chosen) < 2 * n and any(by_pair.values()):
        for p in order:
            if by_pair[p] and len(chosen) < 2 * n:
                first, second = by_pair[p].pop(rng.randrange(len(by_pair[p])))
                for it in (rng.choice(first), rng.choice(second)):
                    chosen.append({**it, "population": population,
                                   "couple_id": couple_id(it["pair_id"], it["history"], it["failure_class"], it["role"])})
    return chosen


def select_main(pool: list[dict], total: int, classes: tuple[str, ...], bins: tuple[str, ...], population: str = "main",
                source: str = "main", keep=None, exclude: frozenset = frozenset(), roles: tuple[str, ...] = ROLES) -> list[dict]:
    """Items in same-history couples (design section 12, B1): both items of a couple come from the same goal pair
    and show the same history, one acting in each goal's tree, with the same class and role. Couples are split over
    classes, then bins, then roles, each equally with the redistribution rule of section 4. Items are drawn from the
    pool's `source` population and labelled `population`."""
    cells = couple_slots(pool, source, keep, exclude)
    cap = {(c, b, r): sum(len(v) for v in cells.get((c, b, r), {}).values()) if r in roles else 0
           for c in classes for b in bins for r in ROLES}
    chosen = []
    per_class = _split(total // 2, {c: sum(cap[(c, b, r)] for b in bins for r in ROLES) for c in classes})
    for c in classes:
        per_bin = _split(per_class[c], {b: sum(cap[(c, b, r)] for r in ROLES) for b in bins})
        for b in bins:
            per_role = _split(per_bin[b], {r: cap[(c, b, r)] for r in ROLES})
            for r in ROLES:
                rng = random.Random(f"{SEED}|textcraft|select|{population}|{c}|{b}|{r}|couples")
                chosen += _draw_couples(cells.get((c, b, r), {}), per_role[r], rng, population)
    return chosen


def fit_profile(it: dict) -> tuple[int, ...]:
    return tuple(F.signs(it["fit_measures"])[m] for m in F.MEASURES)


def mirrored_couple_id(a: dict, b: dict) -> str:
    return "mc-" + hashlib.sha256(json.dumps(sorted([a["id"], b["id"]])).encode()).hexdigest()[:12]


def select_mirrored(pool: list[dict], total: int, bins: tuple[str, ...], exclude: frozenset) -> list[dict]:
    """Usefulness-only never-obtained ingredient items (not at a history a main couple uses), in couples whose ten
    fit-measure signs are exact negatives of each other, possibly from different goal pairs (design section 12, C2).
    Same-history couples of this kind exist in only 23 pairs; mirrored couples spread over many more, and still make
    every listed measure's sign sum to zero. Per bin, total / 3 items: round-robin over goal pairs (shuffled once),
    a random unused item of the pair, matched with a random unused item of the opposite profile."""
    chosen: list[dict] = []
    per_bin = _split(total // 2, {b: 10 ** 9 for b in bins})
    for b in bins:
        cands = sorted((it for it in pool if it["population"] == "main" and it["failure_class"] == "never_obtained"
                        and it["length_bin"] == b and usefulness_only(it) and slot_key(it) not in exclude),
                       key=lambda it: it["id"])
        rng = random.Random(f"{SEED}|textcraft|select|usefulness|{b}|mirrored")
        by_pair: dict[str, list[dict]] = defaultdict(list)
        by_profile: dict[tuple, list[dict]] = defaultdict(list)
        for it in cands:
            by_pair[it["pair_id"]].append(it)
            by_profile[fit_profile(it)].append(it)
        order = sorted(by_pair)
        rng.shuffle(order)
        gone: set[str] = set()
        made = 0
        while made < per_bin[b]:
            progress = False
            for p in order:
                if made >= per_bin[b]:
                    break
                left = [x for x in by_pair[p] if x["id"] not in gone]
                while left:
                    x = left.pop(rng.randrange(len(left)))
                    gone.add(x["id"])
                    want = tuple(-v for v in fit_profile(x))
                    options = [y for y in by_profile[want] if y["id"] not in gone]
                    if options:
                        y = options[rng.randrange(len(options))]
                        gone.add(y["id"])
                        cid = mirrored_couple_id(x, y)
                        chosen += [{**z, "population": "usefulness", "couple_id": cid} for z in (x, y)]
                        made += 1
                        progress = True
                        break
            if not progress:
                break
    return chosen


def select(pool: list[dict]) -> list[dict]:
    chosen = []
    for population, (total, classes, bins) in QUOTAS.items():
        if population == "main":
            chosen += select_main(pool, total, classes, bins)
            continue
        if population == "usefulness":  # extra usefulness-only items in sign-mirrored couples (12, C2)
            used = frozenset(slot_key(it) for it in chosen)
            chosen += select_mirrored(pool, total, bins, used)
            continue
        if population == "success":  # determined successes in couples (12, C3)
            chosen += select_main(pool, total, classes, bins, "success", "success")
            continue
        cells = {(c, b): [it for it in pool if it["population"] == population and it["failure_class"] == c
                          and it["length_bin"] == b] for c in classes for b in bins}
        per_class = _split(total, {c: sum(len(cells[(c, b)]) for b in bins) for c in classes})
        for c in classes:
            per_bin = _split(per_class[c], {b: len(cells[(c, b)]) for b in bins})
            for b in bins:
                rng = random.Random(f"{SEED}|textcraft|select|{population}|{c}|{b}")
                chosen += [{**it, "couple_id": None} for it in _round_robin(cells[(c, b)], per_bin[b], rng)]
    return sorted(chosen, key=lambda it: (it["pair_id"], it["id"]))


# ---------------- prompt examples ----------------


def _walk_example(pair: Pair, kind: str, k: int) -> dict:
    for seed in range(FEWSHOT_WALKS):
        actions = walk(pair, seed)
        for t in range(2, len(actions)):
            if actions[t].startswith(kind):
                env, history = W.replay(list(pair.lines), None, actions[: t + 1])
                action, reply = history[-1]
                if kind == "craft ":
                    crafted = next(ln.out for ln in lines_of(pair) if action.startswith(f"craft {ln.out_count} {ln.out} using"))
                    goal = next(g for g in pair.goals if W.item_id(crafted) in W.tree_craftables(W.item_id(g)))
                else:  # the goal whose tree does not hold the fetched item: an off-goal success (section 12, C8)
                    fetched = action.split(" ", 2)[2]
                    outside = [g for g in pair.goals if fetched not in tree_names(g)]
                    goal = outside[0] if len(outside) == 1 else random.Random(f"{SEED}|textcraft|fewshot|goal|{k}").choice(pair.goals)
                return {"pair_id": pair.pair_id, "goal": goal, "recipe_lines": list(pair.lines), "walk_seed": seed,
                        "history": history[:-1], "action": action, "observation": reply}
    raise RuntimeError(f"{pair.pair_id}: no walk has a {kind.strip()} at step 3 or later")


def fewshot_examples(pool_by_pair: dict[str, list[dict]]) -> list[dict]:
    pairs = {p.pair_id: p for p in all_pairs()}
    p1, p2, p3 = (pairs[i] for i in fewshot_pair_ids())
    failures = sorted((it for it in pool_by_pair[p2.pair_id] if it["population"] == "main" and 2 <= it["history_len"] <= 5
                       and it["walk_seed"] < FEWSHOT_WALKS), key=lambda it: it["id"])
    if not failures:
        raise RuntimeError(f"{p2.pair_id}: no main item with 2-5 history steps")
    f = random.Random(f"{SEED}|textcraft|fewshot|item").choice(failures)
    fail = {"pair_id": f["pair_id"], "goal": f["goal"], "recipe_lines": f["recipe_lines"], "walk_seed": f["walk_seed"],
            "history": f["history"], "action": f["action"], "observation": f["reply"]}
    return [_walk_example(p1, "craft ", 1), fail, _walk_example(p3, "get ", 3)]


# ---------------- main ----------------


def candidate_pool(pool_by_pair: dict[str, list[dict]]) -> list[dict]:
    """Eligible items of every pair except the prompt-example pairs, whose goals must never appear in a test item."""
    held_out = set(fewshot_pair_ids())
    return [it for p in all_pairs() if p.pair_id not in held_out for it in pool_by_pair[p.pair_id]]


def _eligible_of(pair_id: str) -> tuple[list[dict], Counter]:
    excluded: Counter = Counter()
    return eligible(next(p for p in all_pairs() if p.pair_id == pair_id), excluded), excluded


def build(jobs: int = 1) -> tuple[list[dict], list[dict], dict]:
    """Every pair's eligible items (in worker processes when jobs > 1; the result does not depend on jobs), then
    selection and the prompt examples."""
    excluded: Counter = Counter()
    held_out = set(fewshot_pair_ids())
    ids = [p.pair_id for p in all_pairs()]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(jobs) as workers:
            results = workers.map(_eligible_of, ids, chunksize=1)
    else:
        results = [_eligible_of(pid) for pid in ids]
    pool_by_pair = {pid: its for pid, (its, _ex) in zip(ids, results)}
    for _its, ex in results:
        excluded.update(ex)
    pool = candidate_pool(pool_by_pair)
    items = select(pool)
    shots = fewshot_examples(pool_by_pair)
    stats = {
        "pairs": len(all_pairs()), "test_pairs": len(all_pairs()) - len(held_out), "fewshot_pairs": sorted(held_out),
        "excluded": dict(sorted(excluded.items())),
        "eligible": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(Counter(
            (it["population"], it["failure_class"], it["length_bin"]) for it in pool).items())},
        "eligible_main_by_role_and_fit": {"|".join(k): v for k, v in sorted(Counter(
            (it["failure_class"], it["length_bin"], it["role"], it["history_fit"]) for it in pool
            if it["population"] == "main").items())},
        "couple_slots_by_class_bin_role": {f"{pop}|" + "|".join(k): sum(len(x) for x in v.values())
                                           for pop, keep in (("main", None), ("success", None), ("main_usefulness_only", usefulness_only))
                                           for k, v in sorted(couple_slots(pool, pop.split("_")[0], keep).items())},
        "couple_slot_pairs_by_class_bin_role": {f"{pop}|" + "|".join(k): len(v)
                                                for pop, keep in (("main", None), ("success", None), ("main_usefulness_only", usefulness_only))
                                                for k, v in sorted(couple_slots(pool, pop.split("_")[0], keep).items())},
    }
    return items, shots, stats


def dumps_items(items: list[dict]) -> str:
    return "".join(json.dumps(it, sort_keys=True, ensure_ascii=False) + "\n" for it in items)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    jobs = int(os.environ.get("TEXTCRAFT_BUILD_JOBS") or os.cpu_count() or 1)
    out = Path(argv[0]) if argv else DATA
    out.mkdir(parents=True, exist_ok=True)
    items, shots, stats = build(jobs)
    (out / "items.jsonl").write_text(dumps_items(items), encoding="utf-8")
    (out / "fewshot.json").write_text(json.dumps(shots, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(items), **{k: v for k, v in stats.items() if k != "eligible"}}))


if __name__ == "__main__":
    main()
