"""AgentGym ALFWorld HTTP client + optional mock backend for unit tests."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from wf.constants import ALFWORLD_BASE_URL_DEFAULT
from wf.env import sim

MOCK_GAMES = 140
# AgentGym loads games in this order into one flat list (env_wrapper.py).
# Raw mapping sizes include unsolvable PDDL games that raise KeyError('val1') on reset.
SPLIT_SIZES = {
    "train": 2420,
    "valid_train": 200,  # mappings_test.json
    "valid_seen": 187,
    "valid_unseen": 195,
}
# Solvable-only counts (matches PILOT_PROTOCOL / AlfredTWEnv collect_game_files).
SPLIT_SOLVABLE_SIZES = {
    "train": 2179,
    "valid_train": 197,
    "valid_seen": 140,
    "valid_unseen": 134,
}
SPLIT_OFFSETS = {
    "train": 0,
    "valid_train": 2420,
    "valid_seen": 2420 + 200,
    "valid_unseen": 2420 + 200 + 187,
}
SPLIT_MAPPING_FILES = {
    "train": "mappings_train.json",
    "valid_train": "mappings_test.json",
    "valid_seen": "mappings_valid_seen.json",
    "valid_unseen": "mappings_valid_unseen.json",
}
SPLIT_DATA_DIRS = {
    "train": "train",
    "valid_train": "valid_train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}


@dataclass
class GameState:
    game_id: str
    split: str
    task_line: str
    observation: str
    admissible: list[str]
    first_observation: str = ""
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
        return split, SPLIT_OFFSETS.get(split, 0) + int(local)
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

    The server keeps one mutable game per id, so anything that replays or looks ahead must
    use sandbox(), never the id of a live episode. If WF_MOCK_ENV=1, requests go to the
    in-process simulator in wf/env/sim.py instead of a server.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (
            os.environ.get("ALFWORLD_BASE_URL") or base_url or ALFWORLD_BASE_URL_DEFAULT
        ).rstrip("/")
        self.timeout = timeout
        self.mock = os.environ.get("WF_MOCK_ENV", "0") == "1"
        self._session = sim.Server() if self.mock else requests.Session()
        self._env_id: int | None = None
        self._solvable_local: dict[str, list[int]] = {}
        self._sandbox: AlfworldClient | None = None
        self._is_sandbox = False
        self._scenes: dict[str, Any] = {}
        self._expert_missing = False

    def sandbox(self) -> AlfworldClient:
        """Sibling client on its own env id, for replays that must not touch a live episode."""
        if self._is_sandbox:
            return self
        if self._sandbox is None:
            side = AlfworldClient(base_url=self.base_url, timeout=self.timeout)
            side._session = self._session
            side._is_sandbox = True
            side.create()
            self._sandbox = side
        return self._sandbox

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> bool:
        try:
            r = self._session.get(self._url("/"), timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    def create(self) -> str:
        r = self._session.post(self._url("/create"), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"create failed: {data['error']}")
        self._env_id = int(data["id"])
        return str(self._env_id)

    def close(self) -> None:
        if self._env_id is None:
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

    def _mapping_config_dir(self) -> Path | None:
        root = os.environ.get("WORD2WORLD_ROOT") or os.environ.get("WF_WORD2WORLD_ROOT")
        if not root:
            return None
        p = Path(root) / "AgentGym" / "agentenv-alfworld" / "configs"
        return p if p.is_dir() else None

    def _alfworld_data(self) -> Path | None:
        candidates = []
        if os.environ.get("ALFWORLD_DATA"):
            candidates.append(os.environ["ALFWORLD_DATA"])
        candidates.append(str(Path.home() / ".cache" / "alfworld"))
        root = os.environ.get("WORD2WORLD_ROOT") or os.environ.get("WF_WORD2WORLD_ROOT")
        if root:
            candidates.append(str(Path(root) / "data" / "alfworld"))
        for raw in candidates:
            p = Path(os.path.expandvars(raw)).expanduser()
            if p.is_dir() and (p / "json_2.1.1").is_dir():
                return p
        return None

    def solvable_local_indices(self, split: str) -> list[int]:
        """Local indices within AgentGym's mapping whose game.tw-pddl has solvable=true.

        AgentGym's flat `games` list includes unsolvable files; resetting those raises
        KeyError('val1') (broken PDDL). Protocol split sizes are the solvable subset.
        """
        if split in self._solvable_local:
            return self._solvable_local[split]
        cfg_dir = self._mapping_config_dir()
        data = self._alfworld_data()
        mapping_name = SPLIT_MAPPING_FILES.get(split)
        data_subdir = SPLIT_DATA_DIRS.get(split)
        if not cfg_dir or not data or not mapping_name or not data_subdir:
            # Fall back to all raw indices if paths unavailable (mock/dev).
            idxs = list(range(SPLIT_SIZES.get(split, 0)))
            self._solvable_local[split] = idxs
            return idxs
        mapping_path = cfg_dir / mapping_name
        if not mapping_path.is_file():
            idxs = list(range(SPLIT_SIZES.get(split, 0)))
            self._solvable_local[split] = idxs
            return idxs
        mappings = json.loads(mapping_path.read_text())
        good: list[int] = []
        for i, m in enumerate(mappings):
            game_path = (
                data
                / "json_2.1.1"
                / data_subdir
                / m["task_type"]
                / m["task_id"]
                / "game.tw-pddl"
            )
            try:
                meta = json.loads(game_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("solvable") is True:
                good.append(i)
        self._solvable_local[split] = good
        return good

    def list_games(self, split: str) -> list[str]:
        if self.mock:
            return [f"{split}/{i}" for i in range(MOCK_GAMES)]
        # Prefer solvable-only indices so reset does not hit broken PDDL ('val1').
        return [f"{split}/{i}" for i in self.solvable_local_indices(split)]

    def reset(self, game_id: str, split: str) -> GameState:
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
            first_observation=prev.first_observation if prev else obs,
            inventory=list(prev.inventory) if prev else [],
            done=done,
            reward=reward,
            won=won,
            step=step,
            history=history,
            scene_objects=list(prev.scene_objects) if prev else [],
            meta={"task_type": data.get("task_type"), "raw": {k: data[k] for k in data if k != "observation"}},
        )


def server_from_config(cfg: dict) -> AlfworldClient:
    return AlfworldClient(base_url=cfg.get("alfworld_base_url"))
