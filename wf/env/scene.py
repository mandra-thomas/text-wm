"""Hidden scene contents, read from the environment on a private env id.

Observations are environment-generated, so nothing here is hand-written. The result is used
for labels and counterfactual construction only; it is never shown to the agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from wf.env.server import AlfworldClient

_OBJECT = re.compile(r"\ban? ([a-z]+ \d+)")


@dataclass
class Scene:
    receptacles: list[str]
    locations: dict[str, str] = field(default_factory=dict)  # "mug 1" -> "drawer 1"
    closed: set[str] = field(default_factory=set)

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
    scene = Scene(receptacles=[a[len("go to "):] for a in state.admissible if a.startswith("go to ")])
    for receptacle in scene.receptacles:
        state = scanner.step(state, f"go to {receptacle}")
        if f"The {receptacle} is closed" in state.observation:
            scene.closed.add(receptacle)
            state = scanner.step(state, f"open {receptacle}")
        for instance in _OBJECT.findall(state.observation):
            scene.locations[instance] = receptacle
    scanner._scenes[game_id] = scene
    return scene
