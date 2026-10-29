"""Expert-path labels (PILOT_PROTOCOL §4.6).

Prefer ALFWorld handcoded expert via the underlying env. If unavailable, return
None and record in STATUS.md — never approximate with a heuristic.
"""
from __future__ import annotations

from typing import Any

from wf.env.server import AlfworldClient, GameState


def on_expert_path(
    client: AlfworldClient,
    state: GameState,
    action: str,
) -> bool | None:
    """Return True/False if expert label available, else None."""
    # Try info from last step / meta
    meta = state.meta or {}
    if "expert_plan" in meta and isinstance(meta["expert_plan"], list):
        plan = [str(x).strip().lower() for x in meta["expert_plan"]]
        return action.strip().lower() in plan
    if "expert_actions" in meta and isinstance(meta["expert_actions"], list):
        plan = [str(x).strip().lower() for x in meta["expert_actions"]]
        return action.strip().lower() in plan

    # HTTP expert endpoint if server exposes it
    if not client.mock and client._env_id is not None:
        for path in ("/expert", "/env/expert", "/alfworld/expert"):
            try:
                r = client._session.post(
                    client._url(path),
                    json={
                        "id": client._env_id,
                        "env_id": client._env_id,
                        "action": action,
                        "game_id": state.game_id,
                    },
                    timeout=client.timeout,
                )
                if r.ok:
                    data = r.json()
                    if "on_expert_path" in data:
                        return bool(data["on_expert_path"])
                    if "expert_action" in data:
                        return str(data["expert_action"]).strip().lower() == action.strip().lower()
            except Exception:
                continue

    # Mock: treat opening drawer / taking mug / moving to coffeemachine as expert-ish
    if client.mock:
        a = action.strip().lower()
        expertish = {
            "go to drawer 1",
            "open drawer 1",
            "take mug 1 from drawer 1",
            "go to coffeemachine 1",
            "move mug 1 to coffeemachine 1",
        }
        return a in expertish

    return None


def expert_available(client: AlfworldClient) -> bool:
    """Quick probe: whether non-null expert labels can be produced."""
    if client.mock:
        return True
    try:
        games = client.list_games("train")
        if not games:
            return False
        if client._env_id is None:
            client.create()
        st = client.reset(games[0], "train")
        label = on_expert_path(client, st, st.admissible[0] if st.admissible else "look")
        return label is not None
    except Exception:
        return False


def label_action(client: AlfworldClient, state: GameState, action: str) -> dict[str, Any]:
    val = on_expert_path(client, state, action)
    return {"on_expert_path": val, "action": action, "game_id": state.game_id}
