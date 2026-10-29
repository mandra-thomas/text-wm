"""AgentGym ALFWorld HTTP client + optional mock backend for unit tests."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from wf.constants import ALFWORLD_BASE_URL_DEFAULT

# AgentGym loads games in this order into one flat list (env_wrapper.py).
SPLIT_SIZES = {
    "train": 2420,
    "valid_train": 200,  # mappings_test.json
    "valid_seen": 187,
    "valid_unseen": 195,
}
SPLIT_OFFSETS = {
    "train": 0,
    "valid_train": 2420,
    "valid_seen": 2420 + 200,
    "valid_unseen": 2420 + 200 + 187,
}


@dataclass
class GameState:
    game_id: str
    split: str
    task_line: str
    observation: str
    admissible: list[str]
    inventory: list[str] = field(default_factory=list)
    done: bool = False
    reward: float = 0.0
    won: bool = False
    step: int = 0
    history: list[tuple[str, str]] = field(default_factory=list)  # (action, obs)
    scene_objects: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def parse_game_id(game_id: str) -> tuple[str, int]:
    """Parse 'train/12' or bare '12' into (split, global_game_index)."""
    if "/" in game_id:
        split, local = game_id.split("/", 1)
        # local may be an int index within split, or a legacy path-like id
        try:
            local_i = int(local)
            return split, SPLIT_OFFSETS.get(split, 0) + local_i
        except ValueError:
            # fallback: hash into train range for unknown string ids
            return split, SPLIT_OFFSETS.get(split, 0)
    return "train", int(game_id)


class AlfworldClient:
    """Client for Word2World AgentGym ALFWorld FastAPI server.

    API (agentenv_alfworld/server.py):
      POST /create -> {"id": int}
      POST /reset  {"id": int, "game": int, "world_type": "Text"}
      POST /step   {"id": int, "action": str}
      POST /close  {"id": int}
      GET  /available_actions?id=
      GET  /observation?id=

    If WF_MOCK_ENV=1, uses an in-process mock (no server).
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (
            base_url or os.environ.get("ALFWORLD_BASE_URL") or ALFWORLD_BASE_URL_DEFAULT
        ).rstrip("/")
        self.timeout = timeout
        self.mock = os.environ.get("WF_MOCK_ENV", "0") == "1"
        self._session = requests.Session()
        self._env_id: int | None = None
        self._mock_games: dict[str, GameState] = {}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> bool:
        if self.mock:
            return True
        try:
            r = self._session.get(self._url("/"), timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    def create(self) -> str:
        if self.mock:
            self._env_id = 0
            return "0"
        r = self._session.post(self._url("/create"), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"create failed: {data['error']}")
        self._env_id = int(data["id"])
        return str(self._env_id)

    def close(self) -> None:
        if self.mock or self._env_id is None:
            self._env_id = None
            return
        try:
            self._session.post(
                self._url("/close"),
                json={"id": int(self._env_id)},
                timeout=self.timeout,
            )
        except Exception:
            pass
        self._env_id = None

    def list_games(self, split: str) -> list[str]:
        if self.mock:
            return [f"{split}/{i}" for i in range(5)]
        n = SPLIT_SIZES.get(split, 0)
        return [f"{split}/{i}" for i in range(n)]

    def reset(self, game_id: str, split: str) -> GameState:
        if self.mock:
            return self._mock_reset(game_id, split)
        if self._env_id is None:
            self.create()
        split_name, global_idx = parse_game_id(game_id if "/" in game_id else f"{split}/{game_id}")
        payload = {
            "id": int(self._env_id),
            "game": int(global_idx),
            "world_type": "Text",
        }
        r = self._session.post(self._url("/reset"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"reset failed for {game_id}: {data['error']}")
        return self._parse_obs(
            game_id=f"{split_name}/{global_idx - SPLIT_OFFSETS.get(split_name, 0)}",
            split=split_name,
            data=data,
            step=0,
            history=[],
        )

    def step(self, state: GameState, action: str) -> GameState:
        if self.mock:
            return self._mock_step(state, action)
        assert self._env_id is not None, "call create() first"
        r = self._session.post(
            self._url("/step"),
            json={"id": int(self._env_id), "action": action},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"step failed action={action!r}: {data['error']}")
        hist = list(state.history) + [(action, str(data.get("observation", "")))]
        return self._parse_obs(
            game_id=state.game_id,
            split=state.split,
            data=data,
            step=state.step + 1,
            history=hist,
            prev=state,
        )

    def _parse_obs(
        self,
        game_id: str,
        split: str,
        data: dict,
        step: int,
        history: list[tuple[str, str]],
        prev: GameState | None = None,
    ) -> GameState:
        obs = str(data.get("observation") or "")
        admissible = data.get("available_actions") or data.get("admissible_commands") or []
        if prev and not admissible:
            admissible = prev.admissible
        task_line = ""
        if prev and prev.task_line:
            task_line = prev.task_line
        else:
            m = re.search(r"Your task is to:.*", obs)
            task_line = m.group(0).strip() if m else ""
        reward = float(data.get("reward") or 0.0)
        done = bool(data.get("done") or False)
        won = reward >= 1.0 or bool(data.get("won") or False)
        return GameState(
            game_id=game_id,
            split=split,
            task_line=task_line,
            observation=obs,
            admissible=[str(a) for a in admissible],
            inventory=list(prev.inventory) if prev else [],
            done=done,
            reward=reward,
            won=won,
            step=step,
            history=history,
            scene_objects=list(prev.scene_objects) if prev else [],
            meta={"task_type": data.get("task_type"), "raw": {k: data[k] for k in data if k != "observation"}},
        )

    # --- mock backend ---

    def _mock_reset(self, game_id: str, split: str) -> GameState:
        goal = "mug"
        task = f"Your task is to: put a {goal} in coffeemachine."
        obs = (
            f"You are in the middle of a room. Looking quickly around you, you see a drawer 1, "
            f"a countertop 1, and a coffeemachine 1.\n\n{task}"
        )
        adm = [
            "go to drawer 1",
            "go to countertop 1",
            "go to coffeemachine 1",
            "inventory",
            "look",
            "help",
            f"take {goal} 1 from drawer 1",
            "open drawer 1",
            "close drawer 1",
            f"move {goal} 1 to coffeemachine 1",
        ]
        st = GameState(
            game_id=game_id,
            split=split,
            task_line=task,
            observation=obs,
            admissible=adm,
            scene_objects=[goal, "apple", "spraybottle", "mug"],
            meta={"drawer_open": False, "holding": None, "mug_loc": "drawer 1"},
        )
        self._mock_games[game_id] = st
        return st

    def _mock_step(self, state: GameState, action: str) -> GameState:
        meta = dict(state.meta)
        obs = "Nothing happens."
        holding = meta.get("holding")
        a = action.strip().lower()

        if a.startswith("go to"):
            obs = f"You arrive at {action[6:].strip()}."
        elif a == "open drawer 1":
            if meta.get("drawer_open"):
                obs = "Nothing happens."
            else:
                meta["drawer_open"] = True
                obs = "You open the drawer 1. In it, you see a mug 1."
        elif a == "close drawer 1":
            if not meta.get("drawer_open"):
                obs = "Nothing happens."
            else:
                meta["drawer_open"] = False
                obs = "You close the drawer 1."
        elif a.startswith("take "):
            if holding:
                obs = "Nothing happens."
            elif "mug" in a and "drawer 1" in a and meta.get("drawer_open") and meta.get("mug_loc") == "drawer 1":
                meta["holding"] = "mug 1"
                meta["mug_loc"] = "inventory"
                obs = "You pick up the mug 1 from the drawer 1."
            else:
                obs = "Nothing happens."
        elif a.startswith("move ") and "coffeemachine" in a:
            if holding == "mug 1":
                meta["holding"] = None
                meta["mug_loc"] = "coffeemachine 1"
                obs = "You move the mug 1 to the coffeemachine 1."
            else:
                obs = "Nothing happens."
        elif a in ("look", "inventory", "help"):
            obs = state.observation.split("\n\n")[0] if a == "look" else f"You are carrying: {holding}."

        won = meta.get("mug_loc") == "coffeemachine 1"
        hist = list(state.history) + [(action, obs)]
        st = GameState(
            game_id=state.game_id,
            split=state.split,
            task_line=state.task_line,
            observation=obs,
            admissible=list(state.admissible),
            inventory=[holding] if holding else [],
            done=won,
            reward=1.0 if won else 0.0,
            won=won,
            step=state.step + 1,
            history=hist,
            scene_objects=list(state.scene_objects),
            meta=meta,
        )
        self._mock_games[state.game_id] = st
        return st


def server_from_config(cfg: dict) -> AlfworldClient:
    return AlfworldClient(base_url=cfg.get("alfworld_base_url"))
