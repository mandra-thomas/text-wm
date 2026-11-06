"""Matched candidate observations o_plus / o_minus (§4.4). Never hand-template."""
from __future__ import annotations

import re

from wf.env.replay import replay, step_counterfactual
from wf.env.scene import Scene, scan_scene
from wf.env.server import AlfworldClient


def attach_candidates(
    client: AlfworldClient,
    transition: dict,
    success_cache: dict[tuple[str, str], str | None] | None = None,
) -> dict:
    """Attach environment-generated o_plus / o_minus, or candidates=null."""
    cache = success_cache if success_cache is not None else {}
    true_obs = transition["true_obs"]
    action = transition["action"]
    game_id = transition["game_id"]
    split = transition.get("split", "train")

    is_fail = true_obs.strip() == "Nothing happens."
    if is_fail:
        o_minus = true_obs
        o_plus = _find_success_obs(client, game_id, split, action, cache)
        if o_plus is None:
            transition["candidates"] = None
            transition["exclusion_reason"] = "no_success_twin"
            return transition
    else:
        o_plus = true_obs
        o_minus = "Nothing happens."

    transition["candidates"] = {"o_plus": o_plus, "o_minus": o_minus}
    return transition


def _setup_plan(action: str, scene: Scene) -> list[str] | None:
    """Actions from reset that reach a state where `action` should work, or None."""

    def arrive(receptacle: str) -> list[str]:
        return [f"go to {receptacle}"] + ([f"open {receptacle}"] if receptacle in scene.closed else [])

    def carry(obj: str, receptacle: str) -> list[str] | None:
        home = scene.locations.get(obj)
        if home is None:
            return None
        steps = arrive(home) + [f"take {obj} from {home}"]
        return steps if home == receptacle else steps + arrive(receptacle)

    m = re.fullmatch(r"take (.+) from (.+)", action)
    if m:
        obj, receptacle = m[1], m[2]
        if receptacle not in scene.receptacles:
            return None
        if scene.locations.get(obj) == receptacle:
            return arrive(receptacle)
        steps = carry(obj, receptacle)
        return None if steps is None else steps + [f"move {obj} to {receptacle}"]
    m = re.fullmatch(r"move (.+) to (.+)", action) or re.fullmatch(r"(?:heat|cool|clean) (.+) with (.+)", action)
    if m:
        return carry(m[1], m[2]) if m[2] in scene.receptacles else None
    m = re.fullmatch(r"go to (.+)", action)
    if m:
        return [] if m[1] in scene.receptacles else None
    m = re.fullmatch(r"(open|close) (.+)", action)
    if m and m[2] in scene.closed:
        return [f"go to {m[2]}"] if m[1] == "open" else arrive(m[2])
    m = re.fullmatch(r"(?:use|examine) (.+)", action)
    if m and m[1] in scene.locations:
        return arrive(scene.locations[m[1]])
    return None


def _find_success_obs(
    client: AlfworldClient,
    game_id: str,
    split: str,
    action: str,
    cache: dict[tuple[str, str], str | None],
) -> str | None:
    """Run the action in a state where it succeeds. No such state means no twin."""
    key = (game_id, action)
    if key in cache:
        return cache[key]
    plan = _setup_plan(action, scan_scene(client, game_id, split))
    obs = None
    if plan is not None:
        try:
            trial, _ = step_counterfactual(replay(client, game_id, plan, split=split), action)
        except RuntimeError:  # plan ended the game, so the env refuses further steps
            trial = None
        if trial and trial.strip() != "Nothing happens.":
            obs = trial
    cache[key] = obs
    return obs
