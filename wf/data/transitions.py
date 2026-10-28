"""Counterfactual transition builder (§4.2)."""
from __future__ import annotations

import re
from typing import Any, Iterable

from wf.data.candidates import attach_candidates
from wf.data.goals import (
    extract_goal_object,
    history_goal_agnostic,
    make_swapped_task_line,
    swapped_goal,
)
from wf.env.expert import on_expert_path
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.server import AlfworldClient, GameState
from wf.scoring.parser import parse_observation

_ACTION_TYPE = re.compile(r"^(go to|take|move|open|close|heat|cool|clean|examine|inventory|look|help)\b", re.I)


def action_type(action: str) -> str:
    m = _ACTION_TYPE.match(action.strip())
    return m.group(1).lower() if m else "other"


def context_determined_label(
    history_text: str,
    error_type: str,
    true_obs: str,
) -> bool:
    """Rule: validity/precondition determined when relevant facts appear in history."""
    if error_type in ("validity", "precondition"):
        # Heuristic: if history mentions open/close/holding cues, treat as determined
        cues = ("you open", "you close", "you pick up", "you are carrying", "in it, you see")
        h = history_text.lower()
        return any(c in h for c in cues) or true_obs.strip() == "Nothing happens."
    if "In it, you see" in true_obs or "you see" in true_obs.lower():
        # reveal of unopened receptacle contents is not context-determined
        return False
    return True


def _invalid_action(state: GameState, goal_object: str) -> str:
    # Object not present / not visible
    return f"take {goal_object} 1 from bathtubbasin 1"


def _precondition_action(state: GameState, goal_object: str) -> str:
    # move while not holding
    return f"move {goal_object} 1 to coffeemachine 1"


def _wrong_object_variant(action: str, alt_object: str) -> str:
    return re.sub(r"\b([a-zA-Z_]+)(\s+\d+)", rf"{alt_object}\2", action, count=1)


def generate_alternatives(
    state: GameState,
    proposal: str,
    goal_object: str,
    scene_objects: list[str],
) -> list[tuple[str, str, bool]]:
    """Return list of (action, error_type, targets_goal_object)."""
    alts: list[tuple[str, str, bool]] = []
    alts.append((proposal, "proposal", goal_object.lower() in proposal.lower()))
    alts.append((_invalid_action(state, goal_object), "validity", True))
    alts.append((_precondition_action(state, goal_object), "precondition", True))
    others = [o for o in scene_objects if o.lower() != goal_object.lower()]
    if others:
        alts.append((_wrong_object_variant(proposal, others[0]), "validity", False))
    # targeted goal-object failures
    alts.append((f"take {goal_object} 1 from drawer 99", "validity", True))
    alts.append((f"move {goal_object} 1 to sinkbasin 1", "precondition", True))
    # dedupe
    seen = set()
    out = []
    for a, e, t in alts:
        if a in seen:
            continue
        seen.add(a)
        out.append((a, e, t))
    return out


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
    import random

    rng = random.Random(rng_seed + state_idx)
    handle = replay(client, game_id, prefix_actions, split=split)
    state = handle.state
    goal_object = extract_goal_object(state.task_line)
    scene = state.scene_objects or ["mug", "apple", "spraybottle"]
    swap_obj = swapped_goal(scene, goal_object, rng=rng)
    hist = history_text_from_state(state)

    records: list[dict] = []
    tid = id_start
    for action, error_type, targets in generate_alternatives(state, proposal, goal_object, scene):
        # fresh replay each counterfactual
        h2 = replay(client, game_id, prefix_actions, split=split)
        obs, info = step_counterfactual(h2, action)
        oc = parse_observation(obs, goal_object)
        et = error_type if error_type != "proposal" else (
            "validity" if obs.strip() == "Nothing happens." else "completion"
        )
        rec = {
            "id": f"t_{tid:06d}",
            "game_id": game_id,
            "split": split,
            "state_idx": state_idx,
            "prefix_actions": list(prefix_actions),
            "history_text": hist,
            "task_line": state.task_line,
            "goal_object": goal_object,
            "scene_objects": scene,
            "swapped_goal_object": swap_obj,
            "swapped_task_line": make_swapped_task_line(swap_obj),
            "action": action,
            "action_type": action_type(action),
            "true_obs": obs,
            "outcome_class": oc,
            "error_type": et,
            "context_determined": context_determined_label(hist, et, obs),
            "history_goal_agnostic": history_goal_agnostic(hist, goal_object, swap_obj),
            "history_len": len(prefix_actions),
            "targets_goal_object": targets,
            "source": "counterfactual",
            "on_expert_path": on_expert_path(client, state, action),
            "candidates": None,
        }
        attach_candidates(client, rec, success_cache=success_cache)
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
    """context_determined & true failure (protocol Pilot A)."""
    out = []
    for t in transitions:
        if not t.get("context_determined"):
            continue
        if (t.get("true_obs") or "").strip() != "Nothing happens.":
            continue
        if t.get("candidates") is None:
            continue
        out.append(t)
    return out
