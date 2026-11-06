"""In-process stand-in for the AgentGym ALFWorld server (WF_MOCK_ENV=1, unit tests).

Mirrors AgentGym's env_wrapper.py at Word2World e8dc240: one server-side env per id, /reset
rebuilds it, /step mutates it, episodes end at ENV_STEP_LIMIT steps or a win, and after
done=True every /step returns an error. Commands
outside the admissible list return "Nothing happens.". Some receptacles share a spot, as on the
real server: `go to` between them is not offered, and actions on one work from the other.
Observation strings use the templates
of real trajectories. It plugs in as AlfworldClient._session, so mock runs exercise the same
client code as ICE runs.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from wf.constants import ENV_STEP_LIMIT

GOALS = ("mug", "apple", "egg", "pan", "bowl")
TARGETS = ("coffeemachine", "diningtable", "shelf", "sidetable")
TASK_TEMPLATES = (
    "put a {o} in {r}.",
    "put some {o} on {r}.",
    "clean some {o} and put it in {r}.",
    "put a clean {o} in {r}.",
    "heat some {o} and put it in {r}.",
    "put a hot {o} in {r}.",
    "cool some {o} and put it in {r}.",
    "put a cool {o} in {r}.",
    "put two {o} in {r}.",
    "find two {o} and put them in {r}.",
    "look at {o} under the desklamp.",
    "examine the {o} with the desklamp.",
)


class World:
    def __init__(
        self,
        task: str = "Your task is to: put a mug in coffeemachine.",
        goal: str = "mug",
        target: str = "coffeemachine 1",
        shared: tuple[str, ...] = (),
    ):
        self.task, self.goal, self.target, self.shared = task, goal, target, shared
        other = "pear 1" if goal == "apple" else "apple 1"
        self.rec: dict[str, dict[str, Any]] = {
            "drawer 1": {"closed": True, "items": [f"{goal} 1", "pen 1"]},
            "countertop 1": {"closed": False, "items": [other]},
            "cabinet 1": {"closed": True, "items": ["plate 1"]},
            target: {"closed": False, "items": []},
            "sinkbasin 1": {"closed": False, "items": []},
        }
        self.loc: str | None = None
        self.hold: str | None = None
        self.won = False
        self.steps = 0

    def first(self) -> str:
        names = ", ".join("a " + k for k in self.rec)
        return f"You are in the middle of a room. Looking quickly around you, you see {names}.\n{self.task}"

    def _here(self) -> list[str]:
        """The receptacles at the agent's spot."""
        if self.loc is None:
            return []
        return [k for k in self.rec if k == self.loc or (k in self.shared and self.loc in self.shared)]

    def admissible(self) -> list[str]:
        here = self._here()
        acts = [f"go to {k}" for k in self.rec if k not in here] + ["help", "inventory", "look"]
        for name in here:
            rec = self.rec[name]
            if rec["closed"]:
                acts.append(f"open {name}")
            else:
                if name in ("drawer 1", "cabinet 1"):
                    acts.append(f"close {name}")
                if self.hold:
                    acts.append(f"move {self.hold} to {name}")
                else:
                    acts += [f"take {i} from {name}" for i in rec["items"]]
        return acts

    def _describe(self, name: str) -> str:
        rec = self.rec[name]
        if rec["closed"]:
            return f"The {name} is closed."
        if not rec["items"]:
            return f"On the {name}, you see nothing."
        return f"On the {name}, you see " + ", ".join("a " + i for i in rec["items"]) + "."

    def step(self, action: str) -> str:
        if action not in self.admissible():
            return "Nothing happens."
        if action.startswith("go to "):
            self.loc = action[6:]
            return f"You arrive at {self.loc}. {self._describe(self.loc)}"
        if action.startswith("open "):
            name = action[5:]
            rec = self.rec[name]
            rec["closed"] = False
            seen = ", ".join("a " + i for i in rec["items"]) or "nothing"
            return f"You open the {name}. The {name} is open. In it, you see {seen}."
        if action.startswith("close "):
            name = action[6:]
            self.rec[name]["closed"] = True
            return f"You close the {name}."
        m = re.fullmatch(r"take (.+) from (.+)", action)
        if m:
            self.hold = m[1]
            self.rec[m[2]]["items"].remove(self.hold)
            return f"You pick up the {self.hold} from the {m[2]}."
        m = re.fullmatch(r"move (.+) to (.+)", action)
        if m:
            obj, self.hold = self.hold, None
            self.rec[m[2]]["items"].append(obj)
            self.won = m[2] == self.target and obj.startswith(self.goal)
            return f"You move the {obj} to the {m[2]}."
        if action == "inventory":
            return f"You are carrying: a {self.hold}." if self.hold else "You are not carrying anything."
        if action == "look":
            return f"You are facing the {self.loc}. Next to it, you see nothing."
        return "Available commands:\n  look: look around your current location"


def world_for_game(game: int) -> World:
    """Deterministic world per game index, cycling through every real task template."""
    goal = GOALS[game % len(GOALS)]
    template = TASK_TEMPLATES[game % len(TASK_TEMPLATES)]
    target = "desklamp" if "desklamp" in template else TARGETS[game % len(TARGETS)]
    shared = ("drawer 1", "countertop 1") if game % 2 else ()  # holds the goal; visited back to back by a scan
    return World(f"Your task is to: {template.format(o=goal, r=target)}", goal, f"{target} 1", shared)


class _Response:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload, self.status_code = payload, status_code
        self.ok = status_code < 400

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class Server:
    """Duck-types requests.Session."""

    def __init__(self, factory: Callable[[int], World] = world_for_game):
        self.factory = factory
        self.env: dict[int, World | None] = {}
        self.done: dict[int, bool] = {}
        self.next_id = 0
        self.calls = {"reset": 0, "step": 0, "other": 0}

    def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> _Response:
        path = url[url.index("/", url.index("//") + 2):]
        if path == "/create":
            idx = self.next_id
            self.next_id += 1
            self.env[idx], self.done[idx] = None, False
            return _Response({"id": idx})
        if path == "/reset":
            self.calls["reset"] += 1
            idx = json["id"]
            self.env[idx], self.done[idx] = self.factory(json["game"]), False
            world = self.env[idx]
            return _Response({"id": idx, "observation": world.first(),
                              "available_actions": world.admissible(), "task_type": "sim/trial"})
        if path == "/step":
            self.calls["step"] += 1
            idx = json["id"]
            if self.done[idx]:
                return _Response({"error": f"The task with environment {idx} has finished."})
            world = self.env[idx]
            obs = world.step(json["action"])
            world.steps += 1
            self.done[idx] = world.won or world.steps >= ENV_STEP_LIMIT
            return _Response({"observation": obs, "reward": float(world.won),
                              "available_actions": world.admissible(), "done": self.done[idx]})
        if path == "/close":
            return _Response({"id": json["id"]})
        self.calls["other"] += 1
        return _Response({"detail": "Not Found"}, 404)

    def get(self, url: str, timeout: float | None = None) -> _Response:
        return _Response("This is environment AlfWorld.")
