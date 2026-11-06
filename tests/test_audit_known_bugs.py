"""Regression tests for the bugs found by the 2026-11-05 audit (see notes/deviations.md).

They run the real client against wf/env/sim.py, which reproduces AgentGym's server semantics:
one mutable env per id, and /step refuses after done. A mock whose step() was a pure function
is what hid most of these originally.
"""
import pytest

import wf.agent.foresight as foresight_mod
import wf.data.candidates as candidates_mod
import wf.runs as runs_mod
from wf.env.sim import Server
from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.analysis.stats import (
    H_cell,
    asymmetry_A,
    delta_L,
    delta_p,
    interaction_ci,
    interaction_I,
    pair_permutation,
    rank_outcome,
)
from wf.corruption.schedule import schedule
from wf.data.candidates import _find_success_obs
from wf.data.goals import extract_goal_object, history_goal_agnostic, swapped_goal
from wf.data.transitions import context_determined_label
from wf.env.replay import history_text_from_state, replay
from wf.env.scene import scan_scene
from wf.env.server import AlfworldClient
from wf.scoring.logodds import _swapped_task, build_messages
from wf.scoring.parser import parse_observation
from wf.scoring.prompts import _replace_task_line, parse_react_action
from wf.util import calibration_dir, filed_ok, runs_root


def admissible_in(messages):
    text = messages[-1]["content"].split("Admissible actions:", 1)[-1]
    return [ln.strip()[2:] for ln in text.splitlines() if ln.strip().startswith("- ")]


class GreedyAgent(AgentClient):
    """Competent, memoryless policy; reads only the admissible list."""

    def __init__(self):
        super().__init__()
        self.mock = False
        self.prompts = []

    def complete(self, messages, max_tokens=256):
        self.prompts.append(messages[-1]["content"])
        adm = admissible_in(messages)
        for act in ("move mug 1 to coffeemachine 1", "take mug 1 from drawer 1", "open drawer 1"):
            if act in adm:
                return f"Action: {act}"
        holding = any(a.startswith("move mug 1 to") for a in adm)
        act = "go to coffeemachine 1" if holding else "go to drawer 1"
        return f"Action: {act if act in adm else 'look'}"


class FixedAgent(AgentClient):
    def __init__(self, action):
        super().__init__()
        self.mock = False
        self.action = action

    def complete(self, messages, max_tokens=256):
        return f"Action: {self.action}"


@pytest.fixture
def real_client(monkeypatch):
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.delenv(key, raising=False)

    def make(server=None):
        client = AlfworldClient()
        client._session = server or Server()
        return client

    return make


@pytest.fixture
def isolated_foresight(real_client, monkeypatch):
    """Control: run replays on their own env id, the way the fix should behave."""
    server = Server()
    live, side = real_client(server), real_client(server)
    side.create()
    original = foresight_mod.replay
    monkeypatch.setattr(foresight_mod, "replay", lambda c, g, p, split="train": original(side, g, p, split))
    monkeypatch.setattr(candidates_mod, "replay", lambda c, g, p, split="train": original(side, g, p, split))
    return live


# --- agent loop -------------------------------------------------------------------------


def test_control_agent_wins_when_foresight_has_its_own_env(isolated_foresight):
    ep = run_episode(isolated_foresight, GreedyAgent(), "train/0", "train", "t", ForesightConfig("none"))
    assert ep["success"] and ep["steps"] == 5


def test_live_observations_are_not_double_executed(real_client):
    ep = run_episode(real_client(), FixedAgent("go to drawer 1"), "train/0", "train", "t",
                     ForesightConfig("none"), step_cap=1)
    assert ep["trace"][0]["real_obs"].startswith("You arrive at drawer 1")


def test_winning_episode_does_not_crash(real_client):
    ep = run_episode(real_client(), GreedyAgent(), "train/0", "train", "t", ForesightConfig("none"))
    assert ep["success"]


@pytest.mark.parametrize("config", [ForesightConfig("none"), ForesightConfig("oracle", retention="discarded")])
def test_agent_keeps_task_and_history_after_step_one(isolated_foresight, config):
    agent = GreedyAgent()
    run_episode(isolated_foresight, agent, "train/0", "train", "t", config, step_cap=3)
    assert "Your task is to:" in agent.prompts[-1]
    assert "go to drawer 1" in agent.prompts[-1].split("Admissible actions:")[0]


def test_agent_can_execute_a_followed_false_success(isolated_foresight):
    action = "take mug 1 from drawer 1"
    config = ForesightConfig("corrupted", arm="favorable",
                             corrupt_fn=schedule("train/0", 1, "favorable", 1.0))
    ep = run_episode(isolated_foresight, FixedAgent(action), "train/0", "train", "t", config, step_cap=1)
    step = ep["trace"][0]
    assert step["prediction_shown"].startswith("You pick up")
    assert step["executed"] == action


def test_no_twin_is_not_counted_as_a_corruption(isolated_foresight):
    config = ForesightConfig("corrupted", arm="favorable",
                             corrupt_fn=schedule("train/0", 1, "favorable", 1.0))
    ep = run_episode(isolated_foresight, FixedAgent("take unicorn 1 from countertop 1"), "train/0",
                     "train", "t", config, step_cap=2)
    shown_false = sum(s["prediction_shown"] != "Nothing happens." for s in ep["trace"])
    assert ep["n_corrupted"] == shown_false


def test_trace_targets_goal_object_means_the_goal_object(isolated_foresight):
    ep = run_episode(isolated_foresight, FixedAgent("go to coffeemachine 1"), "train/0", "train", "t",
                     ForesightConfig("none"), step_cap=1)
    assert ep["trace"][0]["targets_goal_object"] is False


# --- RQ2 items --------------------------------------------------------------------------

REAL_TASK_LINES = [
    ("put a mug in coffeemachine.", "mug"),
    ("put some mug on coffeemachine.", "mug"),
    ("put a clean mug in coffeemachine.", "mug"),
    ("put a hot egg in countertop.", "egg"),
    ("put a cool pan in stoveburner.", "pan"),
    ("clean some apple and put it in diningtable.", "apple"),
    ("heat some apple and put it in countertop.", "apple"),
    ("cool some apple and put it in diningtable.", "apple"),
    ("find two cd and put them in safe.", "cd"),
    ("put two alarmclock in desk.", "alarmclock"),
    ("look at alarmclock under the desklamp.", "alarmclock"),
    ("examine the bowl with the desklamp.", "bowl"),
]


@pytest.mark.parametrize("line,goal", REAL_TASK_LINES)
def test_goal_object_parses_every_real_template(line, goal):
    assert extract_goal_object(f"Your task is to: {line}") == goal


def test_swapped_task_changes_only_the_object():
    item = {"task_line": "Your task is to: heat some apple and put it in countertop.",
            "swapped_goal_object": "egg", "goal_object": "apple"}
    assert _swapped_task(item) == "Your task is to: heat some egg and put it in countertop."


def test_scene_objects_available_on_real_client(real_client):
    client = real_client()
    assert "pen" in scan_scene(client, "train/0").objects


def test_failure_with_no_supporting_history_is_not_context_determined():
    assert context_determined_label("Your task is to: put a mug in coffeemachine.",
                                    "take mug 1 from drawer 1", "Nothing happens.") is False


@pytest.mark.parametrize("line,action,expected", [
    ("You arrive at sinkbasin 1. On the sinkbasin 1, you see nothing.", "take mug 1 from drawer 1", True),
    ("You arrive at drawer 1. The drawer 1 is closed.", "take mug 1 from drawer 1", True),
    ("You arrive at drawer 1. On the drawer 1, you see a pen 1.", "take mug 1 from drawer 1", True),
    ("You arrive at drawer 1. On the drawer 1, you see a pen 1.", "move mug 1 to drawer 1", True),
    ("You arrive at drawer 1. The drawer 1 is closed.", "go to drawer 1", True),
    ("You arrive at drawer 1. On the drawer 1, you see a mug 1.", "examine mug 1", False),
])
def test_failure_is_determined_only_by_a_fact_in_history(line, action, expected):
    opening = "You are in the middle of a room. Looking quickly around you, you see a drawer 1.\n"
    history = f"{opening}Your task is to: put a mug in coffeemachine.\n> step\n{line}"
    assert context_determined_label(history, action, "Nothing happens.") is expected


def test_task_line_adjectives_are_not_interactions():
    history = "You are in the middle of a room.\nYour task is to: put a clean mug in shelf.\n> go to shelf 1"
    assert history_goal_agnostic(history, "mug", "apple") is True


def test_history_keeps_room_description(real_client):
    state = replay(real_client(), "train/0", ["go to drawer 1"]).state
    assert "You are in the middle of a room" in history_text_from_state(state)


# --- parser and tier 0 ------------------------------------------------------------------


@pytest.mark.parametrize("obs", [
    "You are not carrying anything.",
    "You are facing the drawer 1. Next to it, you see nothing.",
    "Available commands:\n  look: look around your current location",
])
def test_parser_handles_real_templates(obs):
    assert parse_observation(obs) != "unparseable"


def test_scoring_agreement_uses_100_distinct_items(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.setenv(key, "1")
    seen = []
    real = runs_mod.score_transition

    def spy(scorer, item, condition, **kw):
        seen.append(item["id"])
        return real(scorer, item, condition, **kw)

    class ManyGames(AlfworldClient):
        def list_games(self, split):
            return [f"{split}/{i}" for i in range(50)]

    monkeypatch.setattr(runs_mod, "score_transition", spy)
    monkeypatch.setattr(runs_mod, "AlfworldClient", ManyGames)
    runs_mod.run_t0_scoring_agreement({"run_id": "t0_scoring_agreement"})
    assert len(set(seen)) == 100


# --- statistics -------------------------------------------------------------------------


def _failure(tid, game, agnostic):
    return {"id": tid, "game_id": game, "context_determined": True,
            "true_obs": "Nothing happens.", "history_goal_agnostic": agnostic}


def test_delta_p_on_agnostic_items_ignores_goal_revealing_scores():
    scores = []
    for tid, aligned, swapped in (("a", 0.0, 0.0), ("r", 5.0, -5.0)):
        scores.append({"transition_id": tid, "condition": "aligned", "L": aligned})
        scores.append({"transition_id": tid, "condition": "swapped", "L": swapped})
    out = delta_p(scores, [_failure("a", "g1", True)])  # as called at runs.py:514
    assert out["n_items"] == 1 and out["delta_p_pp"] == 0.0


def test_permutation_flips_whole_games():
    # 30 games, each 10 identical items, half the games positive: a game-level null is
    # compatible with this, an item-level null is not.
    diffs, games = [], []
    for g in range(30):
        diffs += [1.0 if g < 18 else -1.0] * 10
        games += [g] * 10
    assert pair_permutation(diffs, n=500, seed=0) < 0.01
    assert pair_permutation(diffs, n=500, seed=0, cluster_ids=games) > 0.1


def test_prereg_delta_l_is_mean_log_odds_shift_in_nats():
    scores = []
    for tid, aligned, swapped in (("a", -6.0, -7.0), ("b", -4.0, -5.0)):
        scores.append({"transition_id": tid, "condition": "aligned", "L": aligned})
        scores.append({"transition_id": tid, "condition": "swapped", "L": swapped})
    items = [_failure("a", "g1", True), _failure("b", "g2", True)]
    assert delta_L(scores, items)["delta_L_nats"] == pytest.approx(1.0)
    assert delta_p(scores, items)["delta_p_pp"] < 2.0  # why the Δp futility rule alone can mislead


def test_interaction_sign_matches_preregistered_convention():
    def ep(arm, ret, steps):
        return {"game_id": "g", "arm": arm, "retention": ret, "success": True, "steps": steps}

    eps = [ep("favorable", "discarded", 10), ep("unfavorable", "discarded", 20),
           ep("favorable", "retained", 25), ep("unfavorable", "retained", 12)]
    assert H_cell(eps, "discarded") == 10 and H_cell(eps, "retained") == -13
    assert interaction_I(eps) == -23


def test_failures_rank_below_a_success_at_the_cap():
    fail = rank_outcome([{"game_id": "x", "success": False, "steps": 30}])["mean_S"]
    capped = rank_outcome([{"game_id": "x", "success": True, "steps": 30}])["mean_S"]
    assert fail < capped


# --- regressions for the fixes themselves -------------------------------------------------


def test_sandbox_has_its_own_env_id(real_client):
    client = real_client()
    client.create()
    assert client.sandbox()._env_id != client._env_id
    assert client.sandbox().sandbox() is client.sandbox()


def test_each_prompt_is_only_asked_once_per_step(isolated_foresight):
    class Counting(GreedyAgent):
        calls = 0

        def complete(self, messages, max_tokens=256):
            Counting.calls += 1
            return super().complete(messages, max_tokens)

    ep = run_episode(isolated_foresight, Counting(), "train/0", "train", "t",
                     ForesightConfig("oracle"), step_cap=3)
    assert Counting.calls == 2 * ep["steps"]  # proposal, then one decision with the prediction


def test_retention_changes_only_the_prediction_lines(isolated_foresight, real_client):
    def last_prompt(retention):
        server = Server()
        agent = GreedyAgent()
        run_episode(real_client(server), agent, "train/0", "train", "t",
                    ForesightConfig("oracle", retention=retention), step_cap=3)
        return agent.prompts[-1]

    drop = ("Proposed action:", "Expected outcome:")
    retained = [ln for ln in last_prompt("retained").splitlines() if not ln.startswith(drop)]
    discarded = [ln for ln in last_prompt("discarded").splitlines() if not ln.startswith(drop)]
    assert retained == discarded


def test_prediction_names_the_action_it_describes(isolated_foresight):
    agent = GreedyAgent()
    run_episode(isolated_foresight, agent, "train/0", "train", "t", ForesightConfig("oracle"), step_cap=1)
    assert "Proposed action: go to drawer 1\nExpected outcome: You arrive at drawer 1." in agent.prompts[-1]


@pytest.mark.parametrize("text,action", [
    ("Thought: x\nAction: open drawer 1", "open drawer 1"),
    ("Thought: x\nAction:\nopen drawer 1", "open drawer 1"),
    ("Thought: x\nAction: open drawer 1.", "open drawer 1"),
    ("", "look"),
])
def test_parse_react_action(text, action):
    assert parse_react_action(text) == action


def test_success_twin_needs_a_state_the_environment_actually_reaches(real_client):
    client = real_client()
    cache = {}
    assert _find_success_obs(client, "train/0", "train", "move mug 1 to sinkbasin 1", cache) \
        == "You move the mug 1 to the sinkbasin 1."
    assert _find_success_obs(client, "train/0", "train", "take mug 1 from sinkbasin 1", cache) \
        == "You pick up the mug 1 from the sinkbasin 1."
    assert _find_success_obs(client, "train/0", "train", "take unicorn 1 from sinkbasin 1", cache) is None


def test_swapped_goal_is_in_scene_but_not_in_the_receptacle():
    locations = {"mug 1": "drawer 1", "pen 1": "drawer 1", "apple 1": "countertop 1"}
    assert swapped_goal(["apple", "mug", "pen"], "mug", "drawer 1", locations=locations) == "apple"
    assert swapped_goal(["mug", "pen"], "mug", "drawer 1", locations=locations) is None


def test_finetuned_prompt_uses_trained_chat_format(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    item = {"task_line": "Your task is to: put a mug in coffeemachine.", "goal_object": "mug",
            "swapped_task_line": "Your task is to: put a pen in coffeemachine.",
            "history_text": "", "history_pairs": [["go to drawer 1", "You arrive at drawer 1."]],
            "action": "open drawer 1", "game_id": "train/0", "split": "train"}
    aligned = build_messages(item, "aligned", ft_prompt=True)
    swapped = build_messages(item, "swapped", ft_prompt=True)
    assert [m["role"] for m in aligned] == ["system", "user", "assistant", "user"]
    assert aligned[-1]["content"] == "open drawer 1"
    assert "put a mug" in aligned[0]["content"] and "put a pen" in swapped[0]["content"]
    assert aligned[1:] == swapped[1:]


def test_task_swap_preserves_every_other_byte():
    context = "# Hidden\n\nRoom.\nYour task is to: put a mug in shelf.\nAVAILABLE ACTIONS: look\n"
    swapped = _replace_task_line(context, "Your task is to: put a pen in shelf.")
    assert swapped == context.replace("mug", "pen")


def test_mock_runs_cannot_touch_real_results(monkeypatch):
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.setenv("WF_MOCK_ENV", "1")
    assert runs_root().name == "runs_mock" and calibration_dir() == runs_root()


def test_real_runs_refuse_a_mock_filing(monkeypatch, tmp_path):
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("note: mock filing for pipeline dry-run\n")
    assert filed_ok() is False
    (tmp_path / "FILED.txt").write_text("filed_date: 2026-11-05\n")
    assert filed_ok() is True


def test_headroom_resumes_without_rerunning_saved_episodes(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.setenv(key, "1")
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    config = {"run_id": "pilotB_headroom", "n_games": 3}
    runs_mod.run_pilotB_headroom(config)
    saved = (tmp_path / "pilotB_headroom" / "episodes.jsonl").read_text()

    def boom(*args, **kwargs):
        raise AssertionError("a saved episode was run again")

    monkeypatch.setattr(runs_mod, "run_episode", boom)
    runs_mod.run_pilotB_headroom({**config, "resume": True})
    assert (tmp_path / "pilotB_headroom" / "episodes.jsonl").read_text() == saved


def test_scheduled_draws_are_shared_across_arms():
    fav = schedule("g", 7, "favorable", 0.3)
    unf = schedule("g", 7, "unfavorable", 0.3)
    assert [fav(t, False) for t in range(30)] == [unf(t, True) for t in range(30)]


def test_interaction_ci_brackets_the_point_estimate():
    eps = []
    for g, (fd, ud, fr, ur) in enumerate([(10, 20, 25, 12), (12, 22, 24, 10), (8, 18, 26, 14)]):
        for arm, ret, steps in (("favorable", "discarded", fd), ("unfavorable", "discarded", ud),
                                ("favorable", "retained", fr), ("unfavorable", "retained", ur)):
            eps.append({"game_id": str(g), "arm": arm, "retention": ret, "success": True, "steps": steps})
    ci = interaction_ci(eps, seed=0)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"] < 0
    assert ci["estimate"] == pytest.approx(interaction_I(eps))


def test_asymmetry_is_computable_once_successes_are_scored():
    items = [_failure("f", "g1", True), {**_failure("s", "g1", True), "true_obs": "You open it."}]
    scores = [{"transition_id": "f", "condition": "aligned", "L": 1.0},
              {"transition_id": "s", "condition": "aligned", "L": -1.0}]
    out = asymmetry_A(scores, items)
    assert out["false_success_rate"] == 1.0 and out["false_failure_rate"] == 1.0 and out["A"] == 0.0
