from wf.data.goals import history_goal_agnostic, swapped_goal


def test_goal_agnostic_true():
    h = "Your task is to: put a mug in coffeemachine.\n> go to drawer 1\nYou arrive at drawer 1.\n> open drawer 1"
    assert history_goal_agnostic(h, "mug", "apple") is True


def test_goal_agnostic_false():
    h = "> take mug 1 from drawer 1\nYou pick up the mug 1."
    assert history_goal_agnostic(h, "mug", "apple") is False


def test_swapped_goal():
    g = swapped_goal(["mug", "apple", "spraybottle"], "mug")
    assert g != "mug"
