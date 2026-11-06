"""Counterfactual transition builder (§4.2)."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable

from wf.data.candidates import attach_candidates
from wf.data.goals import (
    history_goal_agnostic,
    interacted_objects,
    make_swapped_task_line,
    names_object,
    parse_task_line,
    swap_feasible,
    swapped_goal,
)
from wf.env.expert import on_expert_path
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.scene import Scene, scan_scene
from wf.env.server import AlfworldClient, GameState
from wf.scoring.parser import parse_observation

_INSTANCE = re.compile(r"\ban? ([a-z]+ \d+)")
_ACTION_TYPE = re.compile(r"^(go to|take|move|open|close|heat|cool|clean|examine|use|slice|inventory|look|help)\b", re.I)


def action_type(action: str) -> str:
    m = _ACTION_TYPE.match(action.strip())
    return m.group(1).lower() if m else "other"


@dataclass
class HistoryFacts:
    """What the history text establishes. Unknown stays unknown, never a default."""

    knows_position: bool = False  # location and hands, from the opening observation on
    location: str | None = None  # None while in the middle of the room
    holding: str | None = None
    room: set[str] = field(default_factory=set)
    opened: set[str] = field(default_factory=set)
    closed: set[str] = field(default_factory=set)
    contents: dict[str, set[str]] = field(default_factory=dict)


def history_facts(history_text: str) -> HistoryFacts:
    facts = HistoryFacts()
    for line in (history_text or "").splitlines():
        if line.startswith(("> ", "Your task is to:")):
            continue
        if line.startswith("You are in the middle of a room"):
            facts.knows_position = True
            facts.room = set(_INSTANCE.findall(line))
            continue
        m = re.match(r"You arrive at (.+?)\.", line)
        if m:
            facts.knows_position, facts.location = True, m[1]
        m = re.match(r"You open the (.+?)\.", line)
        if m:
            facts.opened.add(m[1])
            facts.closed.discard(m[1])
        m = re.match(r"You close the (.+?)\.", line)
        if m:
            facts.closed.add(m[1])
            facts.opened.discard(m[1])
        m = re.search(r"The (.+?) is closed\.", line)
        if m:
            facts.closed.add(m[1])
        is_open = re.search(r"The (.+?) is open\.", line)
        if is_open:
            facts.opened.add(is_open[1])
        m = re.search(r"(?:In it|On the (.+?)), you see (.+)", line)
        if m:
            # With a shared spot, the line can describe a receptacle other than the one the agent is at.
            place = m[1] or (is_open[1] if is_open else facts.location)
            if place:
                facts.contents[place] = set(_INSTANCE.findall(m[2]))
        m = re.match(r"You pick up the (.+?) from the (.+?)\.", line)
        if m:
            facts.holding = m[1]
            facts.contents.get(m[2], set()).discard(m[1])
        m = re.match(r"You move the (.+?) to the (.+?)\.", line)
        if m:
            facts.holding = None
            if m[2] in facts.contents:  # a receptacle never seen stays unknown
                facts.contents[m[2]].add(m[1])
    return facts


def _failure_determined(facts: HistoryFacts, action: str) -> bool:
    away = lambda receptacle: facts.knows_position and facts.location != receptacle
    m = re.fullmatch(r"take (.+) from (.+)", action)
    if m:
        obj, receptacle = m[1], m[2]
        return (
            (facts.knows_position and facts.holding is not None)
            or away(receptacle)
            or receptacle in facts.closed
            or (receptacle in facts.contents and obj not in facts.contents[receptacle])
        )
    m = re.fullmatch(r"move (.+) to (.+)", action) or re.fullmatch(r"(?:heat|cool|clean) (.+) with (.+)", action)
    if m:
        return (facts.knows_position and facts.holding != m[1]) or away(m[2])
    m = re.fullmatch(r"open (.+)", action)
    if m:
        return away(m[1]) or m[1] in facts.opened
    m = re.fullmatch(r"close (.+)", action)
    if m:
        return away(m[1]) or m[1] in facts.closed
    m = re.fullmatch(r"go to (.+)", action)
    if m:
        return facts.location == m[1] or bool(facts.room and m[1] not in facts.room)
    return False


def _receptacle(action: str) -> str | None:
    m = re.search(r"\b(?:from|to|with) (.+)$", action) or re.fullmatch(r"(?:open|close) (.+)", action)
    return m[1] if m else None


def at_shared_spot(facts: HistoryFacts, action: str, scene: Scene) -> bool:
    """Does the action's receptacle share a spot with the one the agent is at?

    Then "you are elsewhere" proves nothing: the action can work from here, and the history
    cannot show that. Such failures stay scored but leave the main test (notes/deviations.md).
    """
    target = _receptacle(action)
    return bool(target and facts.location and scene.shares_spot(facts.location, target))


def context_determined_label(history_text: str, action: str, true_obs: str) -> bool:
    """Is the outcome decidable from history_text alone (§4.2)?

    Failures need the fact that makes the action fail (location, holding state, open/closed,
    or already-seen contents) in the history. Reveals of unseen contents are never determined.
    """
    if true_obs.strip() == "Nothing happens.":
        return _failure_determined(history_facts(history_text), action)
    arrival = re.fullmatch(r"go to (.+)", action)
    if arrival and arrival[1] not in history_facts(history_text).closed:
        return False  # "The drawer 2 is closed." is news unless the history already said so
    return "you see" not in true_obs.lower()


def pick_swap(
    scene: Scene,
    task_line: str,
    goal_object: str,
    action: str,
    history_text: str,
    rng: random.Random,
    also_exclude: set[str] | None = None,
) -> str | None:
    """A swapped goal for this item: feasible for the task, and unrelated to the action and history."""
    named = {o for o in scene.objects if names_object(action, o)}
    return swapped_goal(
        scene.objects,
        goal_object,
        receptacle_in_question=_receptacle(action),
        rng=rng,
        locations=scene.locations,
        exclude=named | interacted_objects(history_text) | (also_exclude or set()),
        feasible=lambda o: swap_feasible(task_line, o, scene.traits, len(scene.instances(o))),
    )


def _object_swaps(action: str, instances: list[str]) -> str | None:
    m = re.fullmatch(r"(take|move) (.+?) (from|to) (.+)", action)
    others = [i for i in instances if m and i != m[2]]
    return f"{m[1]} {others[0]} {m[3]} {m[4]}" if others else None


def generate_alternatives(
    state: GameState,
    proposal: str,
    goal_object: str,
    target: str,
    scene: Scene,
    rng: random.Random,
) -> list[tuple[str, str, bool]]:
    """Return list of (action, error_type, targets_goal_object), built from the real scene."""
    goal = (scene.instances(goal_object) or [f"{goal_object} 1"])[0]
    home = scene.home(goal)
    away = [r for r in scene.receptacles if r not in scene.locations.get(goal, ())]
    rng.shuffle(away)
    targets = [r for r in scene.receptacles if r.rsplit(" ", 1)[0] == target] or away[:1]
    others = sorted(i for i in scene.locations if i.rsplit(" ", 1)[0] != goal_object)

    alts: list[tuple[str, str, bool]] = [(proposal, "proposal", names_object(proposal, goal_object))]
    # targeted goal-object failures
    if home:
        alts.append((f"take {goal} from {home}", "validity", True))
    alts += [(f"take {goal} from {r}", "validity", True) for r in away[:2]]
    alts += [(f"move {goal} to {r}", "precondition", True) for r in targets[:1]]
    for verb, tool in (("heat", "microwave 1"), ("cool", "fridge 1"), ("clean", "sinkbasin 1")):
        if tool in scene.receptacles:
            alts.append((f"{verb} {goal} with {tool}", "precondition", True))
    # open an already-open receptacle
    for act, _ in reversed(state.history):
        if act.startswith("open "):
            alts.append((act, "precondition", False))
            break
    # wrong-object and wrong-receptacle variants of the proposal
    variant = _object_swaps(proposal, others)
    if variant:
        alts.append((variant, "validity", False))
    m = re.fullmatch(r"take (.+) from (.+)", proposal)
    wrong = [r for r in away if m and r != m[2]]
    if wrong:
        alts.append((f"take {m[1]} from {wrong[0]}", "validity", names_object(proposal, goal_object)))
    # successes, so false failures are measurable (balance within state, §4.2)
    picks = [a for a in state.admissible if a != proposal and a not in ("help", "look", "inventory")]
    alts += [(a, "success", names_object(a, goal_object)) for a in rng.sample(picks, min(3, len(picks)))]

    out: dict[str, tuple[str, str, bool]] = {}
    for alt in alts:
        out.setdefault(alt[0], alt)
    return list(out.values())


def build_transitions_at_state(
    client: AlfworldClient,
    game_id: str,
    split: str,
    prefix_actions: list[str],
    proposal: str,
    state_idx: int,
    id_start: int,
    rng_seed: int = 0,
    success_cache: dict | None = None,
) -> list[dict]:
    rng = random.Random(f"{rng_seed}|{game_id}|{state_idx}")
    scene = scan_scene(client, game_id, split)
    state = replay(client, game_id, prefix_actions, split=split).state
    spec = parse_task_line(state.task_line)
    goal_object = spec.goal_object
    hist = history_text_from_state(state)

    records: list[dict] = []
    tid = id_start
    for action, error_type, targets in generate_alternatives(state, proposal, goal_object, spec.receptacle, scene, rng):
        # fresh replay each counterfactual
        h2 = replay(client, game_id, prefix_actions, split=split)
        obs, info = step_counterfactual(h2, action)
        failed = obs.strip() == "Nothing happens."
        swap_obj = pick_swap(scene, state.task_line, goal_object, action, hist, rng)
        rec = {
            "id": f"t_{tid:06d}",
            "game_id": game_id,
            "split": split,
            "state_idx": state_idx,
            "prefix_actions": list(prefix_actions),
            "history_text": hist,
            "history_pairs": [list(pair) for pair in state.history],
            "task_line": state.task_line,
            "goal_object": goal_object,
            "scene_objects": scene.objects,
            "swapped_goal_object": swap_obj,
            "swap_feasible": swap_obj is not None,
            "swapped_task_line": make_swapped_task_line(state.task_line, goal_object, swap_obj) if swap_obj else None,
            "action": action,
            "action_type": action_type(action),
            "true_obs": obs,
            "outcome_class": parse_observation(obs, goal_object),
            "error_type": ("validity" if error_type in ("proposal", "success") else error_type) if failed else "completion",
            "context_determined": context_determined_label(hist, action, obs),
            "shared_location": at_shared_spot(history_facts(hist), action, scene),
            "history_goal_agnostic": history_goal_agnostic(hist, goal_object, swap_obj) if swap_obj else None,
            "history_len": len(prefix_actions),
            "targets_goal_object": targets,
            "source": "counterfactual",
            "on_expert_path": on_expert_path(client, state, action),
            "candidates": None,
        }
        attach_candidates(client, rec, success_cache=success_cache)
        if swap_obj is None:
            rec.setdefault("exclusion_reason", "no_swap_object")
        if rec["shared_location"]:
            rec.setdefault("exclusion_reason", "shared_location")
        records.append(rec)
        tid += 1
    return records


def build_transitions_from_rollout(
    client: AlfworldClient,
    game_id: str,
    split: str,
    actions: list[str],
    id_start: int = 0,
    k: int = 2,
    rng_seed: int = 0,
) -> list[dict]:
    """Sample every k-th step plus invalid proposals from a logged action sequence."""
    success_cache: dict = {}
    all_recs: list[dict] = []
    tid = id_start
    # Also record which proposals were invalid by replaying
    for i, act in enumerate(actions):
        prefix = actions[:i]
        handle = replay(client, game_id, prefix, split=split)
        obs, _ = step_counterfactual(handle, act)
        invalid = obs.strip() == "Nothing happens."
        if (i % k == 0) or invalid:
            recs = build_transitions_at_state(
                client,
                game_id,
                split,
                prefix,
                proposal=act,
                state_idx=i,
                id_start=tid,
                rng_seed=rng_seed,
                success_cache=success_cache,
            )
            all_recs.extend(recs)
            tid += len(recs)
    return all_recs


def filter_pilot_a_items(transitions: Iterable[dict]) -> list[dict]:
    """context_determined & true failure, with candidates and a swapped goal (protocol Pilot A)."""
    out = []
    for t in transitions:
        if not t.get("context_determined"):
            continue
        if (t.get("true_obs") or "").strip() != "Nothing happens.":
            continue
        if t.get("candidates") is None or not t.get("swapped_goal_object"):
            continue
        out.append(t)
    return out


def main_test_items(items: Iterable[dict]) -> list[dict]:
    """Items for the main test. `shared_location` items feed only the sensitivity check."""
    return [t for t in items if not t.get("shared_location")]


def primary_items(items: Iterable[dict]) -> list[dict]:
    """RQ2's primary population: names the goal, goal-agnostic history, not at a shared spot."""
    return main_test_items(t for t in items if t.get("targets_goal_object") and t.get("history_goal_agnostic"))


def exclusion_counts(transitions: Iterable[dict]) -> dict[str, int]:
    """Failures dropped from Pilot A, by reason (§6 report item)."""
    counts: dict[str, int] = {}
    for t in transitions:
        if (t.get("true_obs") or "").strip() != "Nothing happens.":
            continue
        if t.get("outcome_class") == "unparseable":
            reason = "unparseable"
        elif not t.get("context_determined"):
            reason = "not_context_determined"
        else:
            reason = t.get("exclusion_reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def success_exclusion_counts(transitions: Iterable[dict]) -> dict[str, int]:
    """True successes left out of A, by reason."""
    counts: dict[str, int] = {}
    for t in transitions:
        if (t.get("true_obs") or "").strip() == "Nothing happens.":
            continue
        if t.get("outcome_class") == "unparseable":
            reason = "unparseable"
        elif not t.get("context_determined"):
            reason = "not_context_determined"
        elif t.get("shared_location"):
            reason = "shared_location"
        else:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts
