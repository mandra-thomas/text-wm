"""Matched candidate observations o_plus / o_minus (§4.4). Never hand-template."""
from __future__ import annotations

from typing import Any

from wf.env.replay import EnvHandle, replay, step_counterfactual
from wf.env.server import AlfworldClient


def attach_candidates(
    client: AlfworldClient,
    transition: dict,
    success_cache: dict[tuple[str, str], str] | None = None,
) -> dict:
    """Attach environment-generated o_plus / o_minus, or candidates=null."""
    cache = success_cache if success_cache is not None else {}
    true_obs = transition["true_obs"]
    action = transition["action"]
    game_id = transition["game_id"]
    split = transition.get("split", "train")
    prefix = list(transition.get("prefix_actions") or [])

    is_fail = true_obs.strip() == "Nothing happens."
    if is_fail:
        o_minus = true_obs
        o_plus = _find_success_obs(client, game_id, split, action, prefix, cache)
        if o_plus is None:
            transition["candidates"] = None
            transition["exclusion_reason"] = "no_success_twin"
            return transition
    else:
        o_plus = true_obs
        o_minus = "Nothing happens."

    transition["candidates"] = {"o_plus": o_plus, "o_minus": o_minus}
    return transition


def _find_success_obs(
    client: AlfworldClient,
    game_id: str,
    split: str,
    action: str,
    prefix: list[str],
    cache: dict[tuple[str, str], str],
) -> str | None:
    key = (game_id, action)
    if key in cache:
        return cache[key]

    # Strategy: try progressively shorter prefixes, and a few known setup actions
    # that often make take/move valid in the mock / typical ALFWorld games.
    trial_prefixes: list[list[str]] = []
    for i in range(len(prefix), -1, -1):
        trial_prefixes.append(prefix[:i])
    # Common setup for take-from-drawer style actions
    extras = [
        ["go to drawer 1", "open drawer 1"],
        ["go to countertop 1"],
        ["go to fridge 1", "open fridge 1"],
        ["go to cabinet 1", "open cabinet 1"],
    ]
    trial_prefixes.extend(extras)

    seen: set[tuple[str, ...]] = set()
    for pref in trial_prefixes:
        t = tuple(pref)
        if t in seen:
            continue
        seen.add(t)
        try:
            handle = replay(client, game_id, list(pref), split=split)
            obs, _info = step_counterfactual(handle, action)
            if obs.strip() != "Nothing happens." and obs.strip():
                cache[key] = obs
                return obs
        except Exception:
            continue
    return None
