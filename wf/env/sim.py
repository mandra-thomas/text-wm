"""In-process stand-in for the AgentGym ALFWorld server (WF_MOCK_ENV=1, unit tests).

Mirrors AgentGym's env_wrapper.py at Word2World e8dc240: one server-side env per id, /reset
rebuilds it, /step mutates it, episodes end at ENV_STEP_LIMIT steps or a win, and after
done=True every /step returns an error. Commands outside the admissible list return
"Nothing happens.". As on the real server, some receptacles share a spot (`go to` between them
is not offered, and actions on one work from the other), one object can be listed in two
receptacles, and picking it up clears only one of those listings. Observation strings use the
templates of real trajectories. It plugs in as AlfworldClient._session, so mock runs exercise
the same client code as ICE runs.

Still simpler than the real game: fixed rooms, no `slice`, and no canContain limits.
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
# The subsets of real ALFWorld object flags that this room uses.
HEATABLE = {"mug", "apple", "egg", "plate"}
COOLABLE = {"mug", "apple", "egg", "pan", "bowl", "plate"}
CLEANABLE = {"mug", "apple", "egg", "pan", "bowl", "plate"}
OPENABLE = {"drawer 1", "cabinet 1", "fridge 1", "microwave 1"}
_TOOLS = {"microwave 1": ("heat", "heatable"), "fridge 1": ("cool", "coolable"), "sinkbasin 1": ("clean", "cleanable")}


def _kind(name: str) -> str:
    return name.rsplit(" ", 1)[0]


def _listed(items: list[str]) -> str:
    """Real list text: 'a x 1', 'a x 1, and a y 1', 'a x 1, a y 1, and a z 1'."""
    names = ["a " + i for i in items]
    return ", and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + ", and " + names[-1]


class World:
    def __init__(
        self,
        task: str = "Your task is to: put a mug in coffeemachine.",
        goal: str = "mug",
        target: str = "coffeemachine 1",
        shared: tuple[str, ...] = (),
    ):
        self.task, self.goal, self.target, self.shared = task, goal, target, shared
        self.look_task = target.startswith("desklamp")
        self.needed = 2 if " two " in task else 1
        self.prep = next((p for words, p in ((("heat", "hot"), "heat"), (("cool",), "cool"), (("clean",), "clean"))
                          if any(f" {w} " in f" {task} " for w in words)), None)
        other = "pear 1" if goal == "apple" else "apple 1"
        self.rec: dict[str, dict[str, Any]] = {
            "drawer 1": {"closed": True, "items": [f"{goal} 1", "pen 1"]},
            "countertop 1": {"closed": False, "items": [other]},
            "cabinet 1": {"closed": True, "items": ["plate 1", "plate 2"] + ([f"{goal} 2"] if self.needed == 2 else [])},
            "sidetable 1" if self.look_task else target: {"closed": False, "items": []},
            "sinkbasin 1": {"closed": False, "items": ["pen 1"]},  # also listed in drawer 1
            "fridge 1": {"closed": True, "items": []},
            "microwave 1": {"closed": True, "items": []},
        }
        if self.look_task:
            self.rec["sidetable 1"]["items"].append("desklamp 1")
        self.loc: str | None = None
        self.hold: str | None = None
        self.done_prep: set[str] = set()
        self.lamp_on = False
        self.won = False
        self.steps = 0

    def traits(self) -> dict[str, Any]:
        pickupable = {_kind(i) for r in self.rec.values() for i in r["items"]} - {"desklamp"}
        pickupable.add(self.goal)
        return {
            "pickupable": pickupable,
            "heatable": set(HEATABLE),
            "coolable": set(COOLABLE),
            "cleanable": set(CLEANABLE),
            "can_contain": {_kind(r): set(pickupable) for r in self.rec},
        }

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
            acts.append(f"examine {name}")
            if rec["closed"]:
                acts.append(f"open {name}")
            else:
                if name in OPENABLE:
                    acts.append(f"close {name}")
                if self.hold:
                    acts.append(f"move {self.hold} to {name}")
                else:
                    acts += [f"take {i} from {name}" for i in rec["items"] if _kind(i) != "desklamp"]
            if "desklamp 1" in rec["items"]:
                acts.append("use desklamp 1")
            if self.hold and name in _TOOLS and _kind(self.hold) in self.traits()[_TOOLS[name][1]]:
                acts.append(f"{_TOOLS[name][0]} {self.hold} with {name}")
        if self.hold:
            acts.append(f"examine {self.hold}")
        return acts

    def _describe(self, name: str) -> str:
        rec = self.rec[name]
        if rec["closed"]:
            return f"The {name} is closed."
        seen = _listed(rec["items"]) or "nothing"
        return f"The {name} is open. In it, you see {seen}." if name in OPENABLE else f"On the {name}, you see {seen}."

    def _goal_met(self) -> bool:
        if self.look_task:
            return self.lamp_on and bool(self.hold) and _kind(self.hold) == self.goal
        placed = [i for i in self.rec[self.target]["items"] if _kind(i) == self.goal]
        return len([i for i in placed if self.prep is None or i in self.done_prep]) >= self.needed

    def step(self, action: str) -> str:
        if action not in self.admissible():
            return "Nothing happens."
        if action.startswith("go to "):
            self.loc = action[6:]
            return f"You arrive at {self.loc}. {self._describe(self.loc)}"
        if action.startswith("open "):
            name = action[5:]
            self.rec[name]["closed"] = False
            return f"You open the {name}. {self._describe(name)}"
        if action.startswith("close "):
            name = action[6:]
            self.rec[name]["closed"] = True
            return f"You close the {name}."
        m = re.fullmatch(r"take (.+) from (.+)", action)
        if m:
            self.hold = m[1]
            self.rec[m[2]]["items"].remove(self.hold)
            self.won = self._goal_met()
            return f"You pick up the {self.hold} from the {m[2]}."
        m = re.fullmatch(r"move (.+) to (.+)", action)
        if m:
            obj, self.hold = self.hold, None
            self.rec[m[2]]["items"].append(obj)
            self.won = self._goal_met()
            return f"You move the {obj} to the {m[2]}."
        m = re.fullmatch(r"(heat|cool|clean) (.+) with (.+)", action)
        if m:
            if m[1] == self.prep:
                self.done_prep.add(m[2])
            return f"You {m[1]} the {m[2]} using the {m[3]}."
        if action == "use desklamp 1":
            self.lamp_on = True
            self.won = self._goal_met()
            return "You turn on the desklamp 1."
        if action.startswith("examine "):
            name = action[8:]
            return self._describe(name) if name in self.rec else f"This is a normal {name}. In it, you see nothing."
        if action == "inventory":
            return f"You are carrying: a {self.hold}." if self.hold else "You are not carrying anything."
        if action == "look":
            if self.loc is None:
                return "You are in the middle of a room. Looking quickly around you, you see nothing."
            return f"You are facing the {', and '.join(self._here())}. Next to it, you see nothing."
        return "Available commands:\n  look: look around your current location"


def world_for_game(game: int) -> World:
    """Deterministic world per game index, cycling through every real task template."""
    template = TASK_TEMPLATES[game % len(TASK_TEMPLATES)]
    goal = GOALS[game % len(GOALS)]
    if ("heat" in template or "hot" in template) and goal not in HEATABLE:
        goal = "mug"  # a solvable task always has a heatable goal
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
