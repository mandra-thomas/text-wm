"""Adversarial probes: properties that must hold for any input, not just the examples.

Run against the real client over wf/env/sim.py. Each test tries to break an invariant the
study depends on: what the agent can see, where text comes from, and whether the statistics
recover an effect planted in the data.
"""
import json
import math
import random

import pytest

import wf.runs as runs_mod
import wf.scoring.logodds as logodds_mod
from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.analysis import stats
from wf.corruption.schedule import schedule
from wf.data.goals import names_object
from wf.data.transitions import (
    build_transitions_at_state,
    build_transitions_from_rollout,
    filter_pilot_a_items,
)
from wf.env import sim
from wf.env.replay import replay, step_counterfactual
from wf.env.scene import scan_scene
from wf.env.server import AlfworldClient
from wf.scoring.logodds import LogoddsScorer, ScoreResult, build_messages
from wf.util import read_jsonl, sigmoid

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
CONDITIONS = [
    ForesightConfig("none"),
    ForesightConfig("oracle", retention="discarded"),
    ForesightConfig("oracle", retention="retained"),
    ForesightConfig("validity_only"),
]


class RecordingServer(sim.Server):
    """Remembers every observation the environment ever produced."""

    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def post(self, url, json=None, timeout=None):
        response = super().post(url, json=json, timeout=timeout)
        payload = response.json()
        if isinstance(payload, dict) and "observation" in payload:
            self.seen.add(payload["observation"])
        return response


@pytest.fixture
def client(monkeypatch):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    return AlfworldClient()


@pytest.fixture
def recording(monkeypatch):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    c = AlfworldClient()
    c._session = RecordingServer()
    return c


class Scripted(AgentClient):
    def __init__(self, replies):
        super().__init__()
        self.mock = False
        self.replies = list(replies)
        self.i = 0
        self.prompts = []

    def complete(self, messages, max_tokens=256):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies[self.i % len(self.replies)]
        self.i += 1
        return reply


# --- the agent can say anything ------------------------------------------------------------

HOSTILE_OUTPUTS = [
    "",
    "   \n\n  ",
    "Action:",
    "Action: ",
    "no action here at all",
    "Thought: only a thought",
    "Action: Action: open drawer 1",
    "ACTION: GO TO DRAWER 1",
    "Action: go to drawer 1\nAction: open drawer 1",
    "Action: 🦄🦄🦄",
    "Action: " + "a" * 5000,
    'Action: {"cmd": "rm -rf /"}',
    "Action: take mug 1 from drawer 1\x00",
    "<think>hmm</think>\nAction: look",
    "Action: go to drawer 99",
    "Action:\n\n\nlook",
]


@pytest.mark.parametrize("config", CONDITIONS, ids=lambda c: f"{c.condition}-{c.retention}")
def test_hostile_model_output_never_crashes_the_episode(client, config):
    ep = run_episode(client, Scripted(HOSTILE_OUTPUTS), "train/0", "train", "t", config, step_cap=30)
    assert ep["steps"] == 30 and all(step["executed"] for step in ep["trace"])


def test_episode_record_survives_a_json_round_trip(client):
    ep = run_episode(client, Scripted(HOSTILE_OUTPUTS), "train/0", "train", "t", CONDITIONS[1], step_cap=16)
    assert json.loads(json.dumps(ep)) == ep


# --- replay and isolation ------------------------------------------------------------------


@pytest.mark.parametrize("game", range(0, 24, 3))
def test_replay_reaches_the_same_state_as_a_live_run(client, game):
    gid, rng = f"train/{game}", random.Random(game)
    state = client.reset(gid, "train")
    actions = []
    for _ in range(14):
        action = rng.choice(state.admissible + ["take mug 1 from nowhere 1"])
        following = client.step(state, action)
        if following.done:
            break
        actions.append(action)
        state = following
    replayed = replay(client.sandbox(), gid, actions).state
    assert (replayed.observation, replayed.admissible) == (state.observation, state.admissible)


def test_a_finished_sandbox_never_affects_the_live_episode(client):
    live = client.reset("train/0", "train")
    side = client.sandbox()
    state = replay(side, "train/0", ["go to drawer 1", "open drawer 1", "take mug 1 from drawer 1",
                                     "go to coffeemachine 1"])
    step_counterfactual(state, "move mug 1 to coffeemachine 1")
    assert state.state.done
    assert not client.step(live, "look").done


@pytest.mark.parametrize("config", CONDITIONS[1:], ids=lambda c: f"{c.condition}-{c.retention}")
def test_every_foresight_mode_survives_a_winning_proposal(client, config):
    plan = ["go to drawer 1", "open drawer 1", "take mug 1 from drawer 1", "go to coffeemachine 1",
            "move mug 1 to coffeemachine 1"]
    replies = [f"Action: {a}" for a in plan for _ in range(2)]
    ep = run_episode(client, Scripted(replies), "train/0", "train", "t", config)
    assert ep["success"] and ep["steps"] == 5


# --- nothing hand-written reaches a model -----------------------------------------------------


def _items(client, games=6):
    items = []
    for g in range(games):
        rng = random.Random(g)
        state = client.reset(f"train/{g}", "train")
        actions = []
        for _ in range(10):
            action = rng.choice(sorted(a for a in state.admissible if a not in ("help", "look", "inventory")))
            following = client.step(state, action)
            if following.done:
                break
            actions.append(action)
            state = following
        items += build_transitions_from_rollout(client, f"train/{g}", "train", actions, id_start=len(items),
                                                 rng_seed=1)
    return items


def test_every_candidate_text_was_produced_by_the_environment(recording):
    items = _items(recording)
    candidates = [t["candidates"] for t in items if t["candidates"]]
    assert candidates
    for pair in candidates:
        assert pair["o_plus"] in recording._session.seen
        assert pair["o_minus"] == "Nothing happens."


def test_every_prediction_shown_was_produced_by_the_environment(recording):
    fav = ForesightConfig("corrupted", arm="favorable", corrupt_fn=schedule("train/0", 1, "favorable", 1.0))
    agent = Scripted(["Action: take mug 1 from drawer 1", "Action: go to drawer 1", "Action: open drawer 1"])
    ep = run_episode(recording, agent, "train/0", "train", "t", fav, step_cap=9)
    assert ep["n_corrupted"] > 0
    assert all(s["prediction_shown"] in recording._session.seen for s in ep["trace"])


@pytest.mark.parametrize("ft", [False, True], ids=["fewshot", "finetuned"])
def test_aligned_and_swapped_differ_only_in_the_task_line(client, ft):
    items = filter_pilot_a_items(_items(client))
    assert items
    for t in items:
        aligned = build_messages(t, "aligned", ft_prompt=ft)
        swapped = build_messages(t, "swapped", ft_prompt=ft)
        a_lines = "\n".join(m["content"] for m in aligned).splitlines()
        s_lines = "\n".join(m["content"] for m in swapped).splitlines()
        assert len(a_lines) == len(s_lines)
        changed = [(x, y) for x, y in zip(a_lines, s_lines) if x != y]
        assert changed == [(t["task_line"], t["swapped_task_line"])]


def test_finetuned_context_comes_from_the_real_file_by_game_index(monkeypatch, tmp_path):
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    monkeypatch.setenv("WF_INIT_CONTEXTS", str(tmp_path))
    rows = [{"messages": [{"role": "system", "content": f"# Hidden {i}\nYour task is to: put a mug in shelf.\n"}]}
            for i in range(3)]
    (tmp_path / "wm_instruct_train.json").write_text(json.dumps(rows))
    logodds_mod._wm_contexts.cache_clear()
    item = {"task_line": "Your task is to: put a mug in shelf.", "swapped_task_line": "Your task is to: put a pen in shelf.",
            "goal_object": "mug", "history_text": "", "history_pairs": [], "action": "look",
            "game_id": "train/2", "split": "train"}
    assert build_messages(item, "aligned", ft_prompt=True)[0]["content"] == rows[2]["messages"][0]["content"]
    assert build_messages(item, "swapped", ft_prompt=True)[0]["content"] == "# Hidden 2\nYour task is to: put a pen in shelf.\n"
    logodds_mod._wm_contexts.cache_clear()


def test_swapped_goal_never_names_the_action_object_or_the_goal(client):
    for t in filter_pilot_a_items(_items(client)):
        assert t["swapped_goal_object"] != t["goal_object"]
        assert not names_object(t["action"], t["swapped_goal_object"])
    # Many draws on an action naming a scene object, so a chance pass cannot hide the rule.
    for seed in range(30):
        built = build_transitions_at_state(client, "train/0", "train", [], "take apple 1 from sinkbasin 1",
                                           0, 0, rng_seed=seed)
        named = [t for t in built if t["action"] == "take apple 1 from sinkbasin 1"]
        assert named and named[0]["swapped_goal_object"] != "apple"


# --- what the agent may and may not see ---------------------------------------------------------


def test_hidden_scene_never_reaches_the_agent_before_it_is_observed(client):
    agent = Scripted(["Action: go to countertop 1", "Action: look", "Action: go to sinkbasin 1"])
    run_episode(client, agent, "train/0", "train", "t", CONDITIONS[2], step_cap=6)
    for prompt in agent.prompts:
        assert "pen 1" not in prompt and "plate 1" not in prompt


def test_no_foresight_condition_shows_no_prediction(client):
    agent = Scripted(["Action: go to drawer 1", "Action: open drawer 1"])
    run_episode(client, agent, "train/0", "train", "t", ForesightConfig("none"), step_cap=4)
    assert not any("Expected outcome" in p or "Proposed action" in p for p in agent.prompts)


@pytest.mark.parametrize("arm", ["favorable", "unfavorable"])
def test_corruption_goes_only_to_eligible_steps_and_is_always_false(client, arm):
    agent = Scripted(["Action: take mug 1 from drawer 1", "Action: go to drawer 1", "Action: open drawer 1",
                      "Action: look", "Action: go to countertop 1"])
    config = ForesightConfig("corrupted", arm=arm, corrupt_fn=schedule("train/1", 3, arm, 1.0))
    ep = run_episode(client, agent, "train/0", "train", "t", config, step_cap=15)
    assert ep["n_corrupted"] == sum(s["prediction_type"].startswith("corrupted") for s in ep["trace"])
    for step in ep["trace"]:
        if step["prediction_type"].startswith("corrupted"):
            assert step["proposal_valid"] is (arm == "unfavorable")
            says_success = step["prediction_shown"] != "Nothing happens."
            assert says_success is (arm == "favorable")


def test_discarded_predictions_leave_the_history_and_retained_ones_stay(client):
    def predictions_in_last_prompt(retention):
        agent = Scripted(["Action: go to drawer 1", "Action: go to drawer 1", "Action: open drawer 1",
                          "Action: open drawer 1", "Action: look", "Action: look", "Action: inventory"])
        run_episode(client, agent, "train/0", "train", "t", ForesightConfig("oracle", retention=retention), step_cap=4)
        return agent.prompts[-1].count("Expected outcome:")

    assert predictions_in_last_prompt("discarded") == 1  # only the one for this decision
    assert predictions_in_last_prompt("retained") == 4  # three past steps plus this decision


def test_schedule_is_stable_across_retention_modes(client):
    picks = {}
    for retention in ("discarded", "retained"):
        fn = schedule("train/0", 5, "unfavorable", 0.4)
        picks[retention] = [t for t in range(30) if fn(t, True)]
    assert picks["discarded"] == picks["retained"]


# --- statistics: symmetries and planted effects -------------------------------------------------


def _pairs(seed=0, games=30, items=8, effect=0.5):
    rng = random.Random(seed)
    transitions, scores = [], []
    for g in range(games):
        game_noise = rng.gauss(0, 0.4)
        for i in range(items):
            tid = f"{g}_{i}"
            transitions.append({"id": tid, "game_id": str(g), "context_determined": True,
                                "true_obs": "Nothing happens.", "history_goal_agnostic": True})
            base = rng.gauss(-2, 1)
            scores.append({"transition_id": tid, "condition": "aligned", "L": base + effect + game_noise})
            scores.append({"transition_id": tid, "condition": "swapped", "L": base})
    return transitions, scores


def test_swapping_condition_labels_negates_the_effect():
    transitions, scores = _pairs()
    flipped = [{**s, "condition": "swapped" if s["condition"] == "aligned" else "aligned"} for s in scores]
    assert stats.delta_L(flipped, transitions)["delta_L_nats"] == pytest.approx(-stats.delta_L(scores, transitions)["delta_L_nats"])
    assert stats.delta_p(flipped, transitions)["delta_p_pp"] == pytest.approx(-stats.delta_p(scores, transitions)["delta_p_pp"])


def test_effect_ignores_order_and_game_names():
    transitions, scores = _pairs()
    expected = stats.delta_L(scores, transitions)["delta_L_nats"]
    shuffled = list(scores)
    random.Random(1).shuffle(shuffled)
    assert stats.delta_L(shuffled, transitions)["delta_L_nats"] == pytest.approx(expected)
    renamed = [{**t, "game_id": f"game-{t['game_id']}"} for t in transitions]
    assert stats.delta_L(scores, renamed)["delta_L_nats"] == pytest.approx(expected)


def test_a_constant_shift_in_both_conditions_changes_nothing_in_nats():
    transitions, scores = _pairs()
    moved = [{**s, "L": s["L"] + 7.0} for s in scores]
    assert stats.delta_L(moved, transitions)["delta_L_nats"] == pytest.approx(stats.delta_L(scores, transitions)["delta_L_nats"])


def test_exact_sigmoid_relation_between_nats_and_points():
    assert stats.delta_p([{"transition_id": "a", "condition": "aligned", "L": 1.0},
                          {"transition_id": "a", "condition": "swapped", "L": 0.0}])["delta_p_pp"] \
        == pytest.approx((sigmoid(1.0) - 0.5) * 100)


@pytest.mark.parametrize("planted", [0.0, 0.5, 1.0])
def test_planted_log_odds_shift_is_recovered_by_the_ci(planted):
    hits = 0
    for seed in range(40):
        transitions, scores = _pairs(seed, effect=planted)
        out = stats.delta_L(scores, transitions)
        ci = stats.cluster_bootstrap(lambda v: sum(v) / len(v), out["item_diffs"], out["item_games"], B=300, seed=seed)
        hits += ci["ci_low"] <= planted <= ci["ci_high"]
    assert hits >= 34  # nominal 95% of 40 is 38; allow Monte Carlo slack


def test_interaction_symmetries():
    eps = []
    rng = random.Random(0)
    for g in range(20):
        for arm in ("favorable", "unfavorable"):
            for retention in ("discarded", "retained"):
                penalty = 4 if (arm, retention) in (("unfavorable", "discarded"), ("favorable", "retained")) else 0
                eps.append({"game_id": str(g), "arm": arm, "retention": retention, "success": True,
                            "steps": 10 + penalty + rng.randint(-1, 1)})
    swap_arm = {"favorable": "unfavorable", "unfavorable": "favorable"}
    swap_ret = {"discarded": "retained", "retained": "discarded"}
    base = stats.interaction_I(eps)
    assert base < -5  # the H3-shaped effect that was planted is recovered with the right sign
    assert stats.interaction_I([{**e, "arm": swap_arm[e["arm"]]} for e in eps]) == pytest.approx(-base)
    assert stats.interaction_I([{**e, "retention": swap_ret[e["retention"]]} for e in eps]) == pytest.approx(-base)


def test_A_flips_sign_when_every_prediction_flips():
    items = [{"id": f"f{i}", "game_id": "g", "context_determined": True, "true_obs": "Nothing happens."} for i in range(3)]
    items += [{"id": f"s{i}", "game_id": "g", "context_determined": True, "true_obs": "You open it."} for i in range(3)]
    values = {"f0": 1, "f1": 1, "f2": -1, "s0": -1, "s1": 1, "s2": 1}
    scores = [{"transition_id": k, "condition": "aligned", "L": float(v)} for k, v in values.items()]
    flipped = [{**s, "L": -s["L"]} for s in scores]
    assert stats.asymmetry_A(flipped, items)["A"] == pytest.approx(-stats.asymmetry_A(scores, items)["A"])


# --- Pilot A end to end with a planted scorer --------------------------------------------------


class PlantedScorer(LogoddsScorer):
    """Aligned goals raise L by a known amount, on top of game-level noise."""

    base = -2.0
    effect = 1.0

    def __init__(self, model_name, model_tag=None, **kw):
        super().__init__(model_name, model_tag=model_tag, **kw)
        self.rng = random.Random(0)
        self.noise = {}

    def score_pair(self, transition_id, condition, messages, o_plus, o_minus):
        item_noise = self.noise.setdefault(transition_id, self.rng.gauss(0, 0.5))
        L = self.base + item_noise + (self.effect if condition == "aligned" else 0.0)
        return ScoreResult(transition_id, self.model_tag, condition, L, 0.0, L, sigmoid(L), 1, 1)


def _pilot_a(monkeypatch, tmp_path, base, effect):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    assert runs_mod.run_pilotA_transitions({"run_id": "pilotA_transitions", "n_games": 40}) == 0
    monkeypatch.setattr(PlantedScorer, "base", base)
    monkeypatch.setattr(PlantedScorer, "effect", effect)
    monkeypatch.setattr(runs_mod, "LogoddsScorer", PlantedScorer)
    assert runs_mod.run_pilotA_score({"run_id": "pilotA_score"}) == 0
    metrics = (tmp_path / "pilotA_score" / "metrics.json").read_text().replace("NaN", "null")
    return json.loads(metrics)


def _pilot_a_A(tmp_path):
    metrics = (tmp_path / "pilotA_score" / "metrics.json").read_text().replace("NaN", "null")
    return json.loads(metrics)["reports"]["qwen3-8b-fewshot"]["A"]


def test_pilot_a_recovers_a_planted_effect_end_to_end(monkeypatch, tmp_path):
    report = _pilot_a(monkeypatch, tmp_path, -2.0, 1.0)["reports"]["qwen3-8b-fewshot"]["goal_agnostic"]
    assert report["delta_L_nats"] == pytest.approx(1.0, abs=1e-9)
    assert report["delta_L_ci"][0] <= 1.0 <= report["delta_L_ci"][1]
    assert report["delta_L_perm_p"] < 0.01
    asymmetry = _pilot_a_A(tmp_path)
    assert asymmetry["n_fail"] > 0 and asymmetry["n_succ"] > 0 and asymmetry["A"] is not None


def test_pilot_a_flags_futility_that_contradicts_the_prereg_metric(monkeypatch, tmp_path):
    metrics = _pilot_a(monkeypatch, tmp_path, -7.0, 1.0)
    assert metrics["PILOT_RULE_GOAL_CHANNEL"] is True
    assert set(metrics["rule_conflicts_with_delta_L"]) == {"qwen3-8b-fewshot", "w2w-qwen2.5-7b"}


def test_pilot_a_with_no_effect_does_not_look_significant(monkeypatch, tmp_path):
    metrics = _pilot_a(monkeypatch, tmp_path, -2.0, 0.0)
    report = metrics["reports"]["qwen3-8b-fewshot"]["goal_agnostic"]
    assert report["delta_L_nats"] == pytest.approx(0.0, abs=1e-9)


# --- failure handling ----------------------------------------------------------------------------


def test_crash_then_resume_matches_an_uninterrupted_run(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    config = {"run_id": "pilotB_headroom", "n_games": 4}

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "FILED.txt").write_text("filed_date: x\n")
    monkeypatch.setenv("RUNS_DIR", str(clean_dir))
    runs_mod.run_pilotB_headroom(dict(config))
    clean = read_jsonl(clean_dir / "pilotB_headroom" / "episodes.jsonl")

    crash_dir = tmp_path / "crash"
    crash_dir.mkdir()
    (crash_dir / "FILED.txt").write_text("filed_date: x\n")
    monkeypatch.setenv("RUNS_DIR", str(crash_dir))
    real_episode = runs_mod.run_episode
    calls = [0]

    def dies_midway(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 7:
            raise KeyboardInterrupt  # stands in for a wall-clock kill
        return real_episode(*args, **kwargs)

    monkeypatch.setattr(runs_mod, "run_episode", dies_midway)
    with pytest.raises(KeyboardInterrupt):
        runs_mod.run_pilotB_headroom(dict(config))
    monkeypatch.setattr(runs_mod, "run_episode", real_episode)
    assert len(read_jsonl(crash_dir / "pilotB_headroom" / "episodes.jsonl")) == 6
    runs_mod.run_pilotB_headroom({**config, "resume": True})
    resumed = read_jsonl(crash_dir / "pilotB_headroom" / "episodes.jsonl")
    assert len(resumed) == len(clean)
    key = lambda e: json.dumps(e, sort_keys=True)
    assert sorted(map(key, resumed)) == sorted(map(key, clean))


def test_a_dead_server_fails_loudly_instead_of_excluding_items(client, monkeypatch):
    def dead(*args, **kwargs):
        raise ConnectionError("server gone")

    monkeypatch.setattr(client._session, "post", dead)
    with pytest.raises(ConnectionError):
        run_episode(client, Scripted(["Action: look"]), "train/0", "train", "t", ForesightConfig("none"))


def test_server_error_during_twin_search_is_not_treated_as_no_twin(client, monkeypatch):
    from wf.data.candidates import _find_success_obs

    scan_scene(client, "train/0")  # the scene is known, so the failure lands in the twin step
    original = client._session.post

    def flaky(url, json=None, timeout=None):
        if url.endswith("/step") and json["action"] == "take mug 1 from drawer 1":
            return sim._Response({"error": "env exploded"})
        return original(url, json=json, timeout=timeout)

    monkeypatch.setattr(client._session, "post", flaky)
    with pytest.raises(RuntimeError, match="env exploded"):
        _find_success_obs(client, "train/0", "train", "take mug 1 from drawer 1", {})


def test_missing_calibration_refuses_to_guess(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    assert runs_mod.run_pilotB_retention_minirun({"run_id": "pilotB_retention_minirun"}) == 1


def test_scoring_agreement_refuses_too_few_items(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    assert runs_mod.run_t0_scoring_agreement({"run_id": "t0_scoring_agreement", "games": ["train/0"]}) == 1


def test_icc_of_identical_differences_is_undefined_not_a_crash():
    assert math.isnan(stats.icc([1.0, 1.0, 1.0, 1.0], ["a", "a", "b", "b"]))


def test_math_guard_sigmoid_is_finite_at_extremes():
    assert sigmoid(-5000) == 0.0 and sigmoid(5000) == 1.0 and math.isfinite(sigmoid(0.0))
