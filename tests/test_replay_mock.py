import os

os.environ["WF_MOCK_ENV"] = "1"

from wf.env.replay import replay, step_counterfactual
from wf.env.server import AlfworldClient


def test_replay_determinism():
    c = AlfworldClient()
    gid = "train/pick_and_place_simple-Mock-Mug"
    o1, _ = step_counterfactual(replay(c, gid, ["go to drawer 1", "open drawer 1"]), "take mug 1 from drawer 1")
    o2, _ = step_counterfactual(replay(c, gid, ["go to drawer 1", "open drawer 1"]), "take mug 1 from drawer 1")
    assert o1 == o2
    assert "pick up" in o1.lower() or o1 == "Nothing happens."
