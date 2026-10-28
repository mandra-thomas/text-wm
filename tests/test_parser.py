from wf.scoring.parser import parse_observation


def test_noop():
    assert parse_observation("Nothing happens.") == "noop"


def test_state_change():
    assert parse_observation("You pick up the mug 1 from the drawer 1.") == "state_change"


def test_reveal_with_goal():
    obs = "You open the drawer 1. In it, you see a mug 1."
    assert parse_observation(obs, "mug") == "reveal_with_goal"


def test_reveal_without_goal():
    obs = "You open the drawer 1. In it, you see a apple 1."
    assert parse_observation(obs, "mug") == "reveal_without_goal"


def test_unparseable():
    assert parse_observation("???") == "unparseable"
