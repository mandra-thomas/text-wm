"""AgentGym ALFWorld HTTP client + optional mock backend for unit tests."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from wf.constants import ALFWORLD_BASE_URL_DEFAULT


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


class AlfworldClient:
    """Thin client over Word2World/AgentGym ALFWorld server.

    Expected HTTP surface (AgentGym-style):
      POST /create  {"env_name": "alfworld", ...} -> {"id": ...}
      POST /reset   {"id": ..., "game": ..., "split": ...}
      POST /step    {"id": ..., "action": ...}
      POST /close   {"id": ...}

    If WF_MOCK_ENV=1, uses an in-process mock (no server).
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("ALFWORLD_BASE_URL") or ALFWORLD_BASE_URL_DEFAULT).rstrip("/")
        self.timeout = timeout
        self.mock = os.environ.get("WF_MOCK_ENV", "0") == "1"
        self._session = requests.Session()
        self._env_id: str | None = None
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
            try:
                r = self._session.get(self._url("/list_envs"), timeout=5)
                return r.status_code < 500
            except Exception:
                return False

    def create(self) -> str:
        if self.mock:
            self._env_id = "mock-0"
            return self._env_id
        payload = {"env_type": "alfworld", "env_name": "alfworld"}
        for path in ("/create", "/env/create"):
            try:
                r = self._session.post(self._url(path), json=payload, timeout=self.timeout)
                if r.ok:
                    data = r.json()
                    self._env_id = str(data.get("id") or data.get("env_id") or data.get("instance_id"))
                    return self._env_id
            except Exception:
                continue
        raise RuntimeError(f"Could not create ALFWorld env at {self.base_url}")

    def close(self) -> None:
        if self.mock or not self._env_id:
            self._env_id = None
            return
        for path in ("/close", "/env/close"):
            try:
                self._session.post(
                    self._url(path),
                    json={"id": self._env_id, "env_id": self._env_id},
                    timeout=self.timeout,
                )
            except Exception:
                pass
        self._env_id = None

    def list_games(self, split: str) -> list[str]:
        if self.mock:
            return [f"{split}/pick_and_place_simple-MockGame-Mug" for _ in range(5)]
        for path in (f"/games/{split}", "/list_games", "/alfworld/games"):
            try:
                r = self._session.get(
                    self._url(path),
                    params={"split": split},
                    timeout=self.timeout,
                )
                if r.ok:
                    data = r.json()
                    if isinstance(data, list):
                        return [str(x) for x in data]
                    if isinstance(data, dict) and "games" in data:
                        return [str(x) for x in data["games"]]
            except Exception:
                continue
        # Fallback: caller supplies game ids from config / dataset
        return []

    def reset(self, game_id: str, split: str) -> GameState:
        if self.mock:
            return self._mock_reset(game_id, split)
        assert self._env_id, "call create() first"
        payload = {
            "id": self._env_id,
            "env_id": self._env_id,
            "game": game_id,
            "game_id": game_id,
            "split": split,
        }
        last_err: Exception | None = None
        for path in ("/reset", "/env/reset"):
            try:
                r = self._session.post(self._url(path), json=payload, timeout=self.timeout)
                if r.ok:
                    return self._parse_obs(game_id, split, r.json(), step=0, history=[])
            except Exception as e:
                last_err = e
        raise RuntimeError(f"reset failed for {game_id}: {last_err}")

    def step(self, state: GameState, action: str) -> GameState:
        if self.mock:
            return self._mock_step(state, action)
        assert self._env_id, "call create() first"
        payload = {
            "id": self._env_id,
            "env_id": self._env_id,
            "action": action,
        }
        last_err: Exception | None = None
        for path in ("/step", "/env/step"):
            try:
                r = self._session.post(self._url(path), json=payload, timeout=self.timeout)
                if r.ok:
                    data = r.json()
                    hist = list(state.history) + [(action, self._extract_obs_text(data))]
                    return self._parse_obs(
                        state.game_id,
                        state.split,
                        data,
                        step=state.step + 1,
                        history=hist,
                        prev=state,
                    )
            except Exception as e:
                last_err = e
        raise RuntimeError(f"step failed action={action!r}: {last_err}")

    def _extract_obs_text(self, data: dict) -> str:
        if "observation" in data:
            return str(data["observation"])
        if "obs" in data:
            o = data["obs"]
            if isinstance(o, (list, tuple)):
                return str(o[0])
            return str(o)
        if "state" in data and isinstance(data["state"], str):
            return data["state"]
        return str(data)

    def _parse_obs(
        self,
        game_id: str,
        split: str,
        data: dict,
        step: int,
        history: list[tuple[str, str]],
        prev: GameState | None = None,
    ) -> GameState:
        obs = self._extract_obs_text(data)
        info = data.get("info") or data.get("infos") or {}
        if isinstance(info, list):
            info = info[0] if info else {}
        admissible = (
            info.get("admissible_commands")
            or info.get("admissible")
            or data.get("admissible_commands")
            or (prev.admissible if prev else [])
        )
        task_line = ""
        if prev and prev.task_line:
            task_line = prev.task_line
        else:
            m = re.search(r"Your task is to:.*", obs)
            task_line = m.group(0).strip() if m else (prev.task_line if prev else "")
        scene = info.get("scene_objects") or info.get("objects") or (prev.scene_objects if prev else [])
        won = bool(info.get("won") or data.get("won") or False)
        done = bool(data.get("done") or info.get("done") or won)
        reward = float(data.get("reward") or info.get("reward") or (1.0 if won else 0.0))
        return GameState(
            game_id=game_id,
            split=split,
            task_line=task_line,
            observation=obs,
            admissible=[str(a) for a in admissible],
            inventory=list(info.get("inventory") or (prev.inventory if prev else [])),
            done=done,
            reward=reward,
            won=won,
            step=step,
            history=history,
            scene_objects=[str(x) for x in scene],
            meta=dict(info) if isinstance(info, dict) else {},
        )

    # --- mock backend (deterministic, for tests / dry runs) ---

    def _mock_reset(self, game_id: str, split: str) -> GameState:
        goal = "mug"
        if "Apple" in game_id or "apple" in game_id.lower():
            goal = "apple"
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
            f"take {goal} 1 from countertop 1",
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
            # succeed only if taking mug from drawer while open and not holding
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
        adm = list(state.admissible)
        st = GameState(
            game_id=state.game_id,
            split=state.split,
            task_line=state.task_line,
            observation=obs,
            admissible=adm,
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
