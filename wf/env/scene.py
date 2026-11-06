"""Hidden scene contents, read from the environment on a private env id.

Observations are environment-generated, so nothing here is hand-written. The result is used
for labels and counterfactual construction only; it is never shown to the agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from wf.constants import ENV_STEP_LIMIT
from wf.env.server import AlfworldClient

_OBJECT = re.compile(r"\ban? ([a-z]+ \d+)")


@dataclass
class Scene:
    receptacles: list[str]
    # "mug 1" -> {"drawer 1"}. ALFWorld can list one object in two receptacles at once.
    locations: dict[str, set[str]] = field(default_factory=dict)
    closed: set[str] = field(default_factory=set)
    shared: dict[str, set[str]] = field(default_factory=dict)  # "cabinet 1" -> {"cabinet 2"}: same spot
    traits: dict[str, Any] = field(default_factory=dict)  # the game's own rules, see object_traits

    def shares_spot(self, here: str, other: str) -> bool:
        return other in self.shared.get(here, set())

    def home(self, instance: str) -> str | None:
        """One receptacle holding this instance; the same one every time."""
        places = self.locations.get(instance)
        return min(places) if places else None

    @property
    def objects(self) -> list[str]:
        return sorted({instance.rsplit(" ", 1)[0] for instance in self.locations})

    def instances(self, object_name: str) -> list[str]:
        return sorted(i for i in self.locations if i.rsplit(" ", 1)[0] == object_name)


def scan_scene(client: AlfworldClient, game_id: str, split: str = "train") -> Scene:
    """Visit (and open) every receptacle once; cached per game."""
    scanner = client.sandbox()
    if game_id in scanner._scenes:
        return scanner._scenes[game_id]
    state = scanner.reset(game_id, split)
    scene = Scene(receptacles=[a[len("go to "):] for a in state.admissible if a.startswith("go to ")],
                  traits=scanner.object_traits(game_id, split))
    for receptacle in scene.receptacles:
        if state.step + 2 >= ENV_STEP_LIMIT:  # large rooms need more steps than one episode allows
            state = scanner.reset(game_id, split)
        elif f"go to {receptacle}" not in state.admissible:  # same spot as here, so the move would do nothing
            state = scanner.reset(game_id, split)
        state = scanner.step(state, f"go to {receptacle}")
        # ALFWorld stops offering `go to` for a receptacle at the agent's own spot; on real games
        # this matched a direct test on all 2,164 receptacle pairs (see notes/deviations.md).
        scene.shared[receptacle] = {r for r in scene.receptacles
                                    if r != receptacle and f"go to {r}" not in state.admissible}
        if f"The {receptacle} is closed" in state.observation:
            scene.closed.add(receptacle)
            state = scanner.step(state, f"open {receptacle}")
        for instance in _OBJECT.findall(state.observation):
            scene.locations.setdefault(instance, set()).add(receptacle)
    scanner._scenes[game_id] = scene
    return scene
