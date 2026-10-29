"""Deterministic prefix replay + counterfactual step (PILOT_PROTOCOL §4.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wf.env.server import AlfworldClient, GameState


@dataclass
class EnvHandle:
    client: AlfworldClient
    state: GameState

    @property
    def game_id(self) -> str:
        return self.state.game_id

    @property
    def observation(self) -> str:
        return self.state.observation

    @property
    def admissible(self) -> list[str]:
        return self.state.admissible


def replay(client: AlfworldClient, game_id: str, prefix_actions: list[str], split: str = "train") -> EnvHandle:
    """Reset game and re-execute prefix_actions; return handle at resulting state."""
    if client._env_id is None and not client.mock:
        client.create()
    state = client.reset(game_id, split)
    for action in prefix_actions:
        state = client.step(state, action)
        if state.done:
            break
    return EnvHandle(client=client, state=state)


def step_counterfactual(handle: EnvHandle, action: str) -> tuple[str, dict[str, Any]]:
    """Execute one action; return (observation, info). Handle should be discarded after."""
    new_state = handle.client.step(handle.state, action)
    info = {
        "won": new_state.won,
        "done": new_state.done,
        "reward": new_state.reward,
        "admissible_commands": new_state.admissible,
        "inventory": new_state.inventory,
        "scene_objects": new_state.scene_objects,
        "meta": new_state.meta,
    }
    # Discard semantics: do not mutate caller's handle permanently for further use
    # Caller is expected to call replay again for a fresh handle.
    handle.state = new_state
    return new_state.observation, info


def history_text_from_state(state: GameState) -> str:
    """Build interaction history string from reset obs + (action, obs) pairs."""
    parts: list[str] = []
    # Reconstruct: first observation is reset obs before any actions.
    # We only have post-action obs in history; include task from task_line.
    if state.history:
        # Approximate first obs as task-bearing context
        parts.append(state.task_line)
        for act, obs in state.history:
            parts.append(f"> {act}")
            parts.append(obs)
    else:
        parts.append(state.observation)
    return "\n".join(parts)
