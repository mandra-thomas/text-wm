"""Stage 1b (wf/decisionsim_agent.py, notes/decisionsim_agent_design.md): synthetic checks of every rule, and mock runs
end to end (sample, label, score, analysis), on the simulator and the mock agent and scorer."""
from __future__ import annotations

import json
import random

import pytest

from tests.test_families import FakeTok, _scorer
from tests.test_rq1 import MOCK_KEYS
from wf import decisionsim_agent as A
from wf import families
from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.env.scene import Scene
from wf.env.server import AlfworldClient
from wf.rq1v import YN_QUESTION
from wf.scoring import logodds
from wf.scoring.prompts import agent_react_messages
from wf.util import read_jsonl

TASK = "Your task is to: put some saltshaker on cabinet."
FIRST = "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a shelf 1.\n" + TASK
HIST = FIRST + "\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a mug 1."


# ---------------- synthetic states ----------------


def _cand(rank, failed, goal=False, det=True, shared=False, admissible=True, action=None, progress=False, err=None):
    return {"rank": rank, "action": action or f"act {rank}", "count": 1, "greedy": rank == 1, "true_obs": None if err else
            ("Nothing happens." if failed else "You arrive at x."), "server_error": err, "failed": failed,
            "context_determined": det, "targets_goal_object": goal, "shared_location": shared,
            "admissible": admissible, "action_type": "take" if goal else "go to", "progress": progress}


def _state(cands, game="g1", t=2, swap="mug", agnostic=True):
    return {"game_id": game, "state_idx": t, "task_line": TASK, "goal_object": "saltshaker", "target": "cabinet",
            "history_text": HIST, "history_goal_agnostic": agnostic, "swap": swap,
            "swapped_task_line": TASK.replace("saltshaker", swap) if swap else None, "rank1_replay_ok": True,
            "candidates": cands}


def _L(state, values, arm=A.AWARE):
    return {A.cand_id(state, c): v for c, v in zip(state["candidates"], values)}


def _scores(states, per_arm):
    """per_arm[arm] = list (per state) of lists of L in rank order."""
    return {arm: {k: v for s, vals in zip(states, lists) for k, v in _L(s, vals).items()} for arm, lists in per_arm.items()}


# ---------------- candidate lists ----------------


def test_normalize_collapses_space_case_and_final_stop():
    assert A.normalize_action("  Go  to   Drawer 1. ") == "go to drawer 1"
    assert A.normalize_action(None) is None and A.normalize_action(" . ") is None


def test_rank1_is_the_greedy_action_even_if_never_sampled():
    r = A.rank_candidates("go to shelf 1", ["Action: look"] * 3, "g", 0)
    assert r["candidates"][0] == {"rank": 1, "action": "go to shelf 1", "count": 0, "greedy": True}
    assert [c["action"] for c in r["candidates"]] == ["go to shelf 1", "look"]


def test_greedy_samples_merge_into_rank1_and_keep_the_recorded_string():
    r = A.rank_candidates("Go to shelf 1", ["Action: go to shelf 1", "Action: GO TO SHELF 1.", "Action: look"], "g", 0)
    assert r["candidates"][0]["action"] == "Go to shelf 1" and r["candidates"][0]["count"] == 2
    assert len(r["candidates"]) == 2


def test_ranked_by_count_then_seeded_hash_not_order_of_appearance():
    replies = ["Action: b"] * 3 + ["Action: a"] * 1 + ["Action: c"] * 1 + ["Action: d"] * 2
    r = A.rank_candidates("z", replies, "g", 4)
    acts = [c["action"] for c in r["candidates"]]
    assert acts[:3] == ["z", "b", "d"]
    tied = sorted(["a", "c"], key=lambda a: A.tie_key("g", 4, a))
    assert acts[3:] == tied
    shuffled = replies[:]
    random.Random(1).shuffle(shuffled)
    assert A.rank_candidates("z", shuffled, "g", 4)["candidates"] == r["candidates"]


def test_format_errors_count_and_add_no_candidate():
    r = A.rank_candidates("z", ["Thought: no action", "Action:", "Action: a"], "g", 0)
    assert r["n_format_errors"] == 2 and [c["action"] for c in r["candidates"]] == ["z", "a"]
    assert r["n_samples"] == 3


def test_list_is_cut_at_k_max_and_the_cut_is_recorded():
    replies = [f"Action: a{i}" for i in range(10)]
    r = A.rank_candidates("z", replies, "g", 0, k_max=4)
    assert len(r["candidates"]) == 4 and r["n_cut"] == 7 and r["n_distinct_sampled"] == 10
    assert A.K_MAX == 8 and A.N_SAMPLES == 20 and A.TEMPERATURE == 1.0 and A.TOP_P == 1.0


def test_bookkeeping_flags_identical_replies():
    st = {"n_samples": 3, "n_format_errors": 0, "candidates": [1], "n_cut": 0,
          "replies": [{"text": "Action: look", "finish_reason": "stop"}] * 3}
    other = {**st, "replies": [{"text": "Action: look", "finish_reason": "stop"}, {"text": "Action: x", "finish_reason": "length"}]}
    b = A.sampling_bookkeeping([st, other])
    assert b["states_all_replies_identical"] == 1 and b["truncated_replies"] == 1 and b["format_error_share"] == 0


def test_sample_seed_is_per_state_and_fixed():
    assert A.sample_seed("g", 1) == A.sample_seed("g", 1) != A.sample_seed("g", 2) != A.sample_seed("h", 1)


# ---------------- verifier prompts ----------------


def test_remove_task_line_deletes_only_that_line():
    assert A.remove_task_line(HIST) == HIST.replace("\n" + TASK, "")
    assert A.remove_task_line(TASK + "\nrest") == "rest"
    with pytest.raises(ValueError):
        A.remove_task_line("no task here")
    with pytest.raises(ValueError):
        A.remove_task_line(TASK + "\n" + TASK)


@pytest.fixture
def mock_score(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    logodds._fewshot_hidden.cache_clear()
    yield
    logodds._fewshot_hidden.cache_clear()


@pytest.mark.parametrize("tag", A.MODEL_TAGS)
def test_each_arm_differs_only_in_the_query_task_line(mock_score, tag):
    s = _state([_cand(1, True, goal=True, action="take saltshaker 1 from cabinet 1")])
    c = s["candidates"][0]
    aware, blind, swapped = (A.arm_messages(tag, s, c, a) for a in A.ARMS)
    assert aware[:-1] == blind[:-1] == swapped[:-1]  # the examples and system text never change
    assert all(m[-1]["content"].endswith(YN_QUESTION) for m in (aware, blind, swapped))
    assert TASK in aware[-1]["content"] and "Your task is to:" not in blind[-1]["content"]
    assert "put some mug on cabinet." in swapped[-1]["content"] and "saltshaker on" not in swapped[-1]["content"]
    assert blind[-1]["content"] == aware[-1]["content"].replace("\n" + TASK, "")
    assert "take saltshaker 1 from cabinet 1" in aware[-1]["content"]
    assert any("Your task is to:" in m["content"] for m in blind[:-1])  # the examples keep theirs
    assert A.arm_messages(tag, {**s, "swap": None, "swapped_task_line": None}, c, A.SWAPPED) is None


def test_prompts_are_the_earlier_yes_no_prompts(mock_score):
    from wf.decisionsim import messages_for

    s = _state([_cand(1, True, action="take saltshaker 1 from cabinet 1")])
    item = A.prompt_item(s, s["candidates"][0])
    for tag in A.MODEL_TAGS:
        assert A.arm_messages(tag, s, s["candidates"][0], A.AWARE) == messages_for(tag)(item, "yn_aligned")
        assert A.arm_messages(tag, s, s["candidates"][0], A.SWAPPED) == messages_for(tag)(item, "yn_swapped")


# ---------------- selection ----------------


def test_selection_rules():
    s = _state([_cand(1, True), _cand(2, True, goal=True), _cand(3, False), _cand(4, False)])
    L = _scores([s], {A.AWARE: [[-1, 0, 2, 3]], A.BLIND: [[-1, -1, -1, -1]], A.SWAPPED: [[0.5, 1, 1, 1]]})
    assert A.choose("none", s)["rank"] == 1
    assert A.choose("oracle", s)["rank"] == 3
    assert A.choose("aware", s, L)["rank"] == 3  # rank 2 has L = 0: not accepted
    assert A.choose("blind", s, L)["rank"] == 1 and A.fell_back("blind", s, L)  # none accepted: rank 1
    assert A.choose("swapped", s, L)["rank"] == 1
    assert A.choose("blind", s, L, fallback="max_L")["rank"] == 1  # all tied: the highest rank wins the tie
    assert A.choose("aware", s, L, k=2)["rank"] == 1  # cut to two, neither accepted
    assert A.choose("aware", s, L, fallback="max_L", k=2)["rank"] == 2


def test_oracle_with_no_success_falls_back_to_rank1():
    s = _state([_cand(1, True), _cand(2, True)])
    assert A.choose("oracle", s)["rank"] == 1


def test_server_error_candidates_are_skipped():
    s = _state([_cand(1, True), _cand(2, False, err="boom"), _cand(3, False)])
    L = _scores([s], {A.AWARE: [[-1, 5, 1]]})
    assert A.choose("aware", s, L)["rank"] == 3 and A.choose("oracle", s)["rank"] == 3
    assert [c["rank"] for c in A.scored_candidates(s)] == [1, 3]


def test_scores_are_rounded_to_a_64th_before_the_threshold():
    assert A.quantize(0.007) == 0.0 and A.quantize(0.008) == 1 / 64
    recs = [{"condition": A.AWARE, "transition_id": "x", "L": 0.007}, {"condition": A.AWARE, "transition_id": "y", "L": float("nan")}]
    q, raw = A.scores_by_arm(recs)
    assert q[A.AWARE] == {"x": 0.0} and raw[A.AWARE] == {"x": 0.007}
    assert not A.accepts(q[A.AWARE], "x")


# ---------------- regime, encounter, reach ----------------


def test_regime_needs_failure_goal_object_determination_and_no_shared_spot():
    assert A.regime(_cand(1, True, goal=True))
    assert not A.regime(_cand(1, False, goal=True))
    assert not A.regime(_cand(1, True, goal=False))
    assert not A.regime(_cand(1, True, goal=True, det=False))
    assert not A.regime(_cand(1, True, goal=True, shared=True))


def test_encounter_strict_and_reachable():
    s = _state([_cand(1, False), _cand(2, True, goal=True)])
    assert A.encounter(s) and not A.reachable(s)  # a success is ranked above it
    s2 = _state([_cand(1, True), _cand(2, True, goal=True), _cand(3, False)])
    assert A.reachable(s2)
    assert not A.encounter(_state(s2["candidates"], agnostic=False), strict=True)
    assert A.encounter(s2, strict=True)
    assert not A.encounter(_state([_cand(1, True, goal=True, err="x")]))


# ---------------- contrasts and statistics ----------------


def _four_states():
    s1 = _state([_cand(1, True), _cand(2, True, goal=True), _cand(3, False)], game="g1", t=0)  # harm
    s2 = _state([_cand(1, True), _cand(2, False)], game="g2", t=0)  # help
    s3 = _state([_cand(1, False), _cand(2, False)], game="g3", t=0)  # both succeed, changed
    s4 = _state([_cand(1, False)], game="g4", t=0, swap=None)  # same
    states = [s1, s2, s3, s4]
    L = _scores(states, {A.AWARE: [[-1, 1, 1], [-1, 1], [-1, 1], [1]],
                         A.BLIND: [[-1, -1, 1], [1, 1], [1, 1], [1]],
                         A.SWAPPED: [[-1, -1, 1], [-1, 1], [1, 1], [1]]})
    return states, L


def test_contrast_counts_harm_help_and_the_identity():
    states, L = _four_states()
    c = A.contrast(states, L, "aware", "blind")
    assert c["changed"] == {"n": 3, "harm": 1, "help": 1, "both_fail": 0, "both_succeed": 1}
    assert c["difference"]["estimate"] == 0 and c["identity_holds"]
    assert c["harm_by_kind_of_choice"] == {"regime": 1}
    assert c["fail_rate_a"]["estimate"] == 0.25 and c["fail_rate_b"]["estimate"] == 0.25


def test_decomposition_on_states_with_a_swap_and_its_identity():
    states, L = _four_states()
    dec = A.decomposition(states, L)
    assert dec["n_states"] == 3 and dec["identity_holds"]
    assert dec["aware_minus_swapped"]["estimate"] == pytest.approx(1 / 3)
    assert dec["swapped_minus_blind"]["estimate"] == pytest.approx(-1 / 3)


def test_inference_resamples_and_flips_whole_games(monkeypatch):
    seen = {}
    real_boot, real_perm = A.cluster_bootstrap, A.pair_permutation
    monkeypatch.setattr(A, "cluster_bootstrap", lambda f, v, g, **kw: (seen.__setitem__("boot", list(g)), real_boot(f, v, g, **kw))[1])
    monkeypatch.setattr(A, "pair_permutation", lambda d, **kw: (seen.__setitem__("perm", kw.get("cluster_ids")), real_perm(d, **kw))[1])
    states, L = _four_states()
    A.contrast(states, L, "aware", "blind")
    assert seen["boot"] == ["g1", "g2", "g3", "g4"] and seen["perm"] == ["g1", "g2", "g3", "g4"]


def test_holm_and_verdicts():
    adj = A.holm({"a": 0.01, "b": 0.04})
    assert adj == {"a": 0.02, "b": 0.04}
    assert A.holm({"a": float("nan"), "b": 0.01}) == {"b": 0.02, "a": 1.0}
    assert A.holm({"a": 0.03, "b": 0.031}) == {"a": 0.06, "b": 0.06}  # never below an earlier adjusted p
    assert A.verdict({"ci": [0.001, 0.02]}, 0.03) == "toward failure"
    assert A.verdict({"ci": [-0.02, -0.001]}, 0.03) == "toward success"
    assert A.verdict({"ci": [0.001, 0.02]}, 0.06) == "no reliable difference"
    assert A.verdict({"ci": [-0.001, 0.02]}, 0.01) == "no reliable difference"


def test_auc_and_competence():
    assert A.auc([2, 1], [0, 1]) == pytest.approx(0.875)
    rng = random.Random(3)
    pos, neg = [rng.choice([-1, 0, 0.5, 2]) for _ in range(40)], [rng.choice([-2, -1, 0, 0.5]) for _ in range(30)]
    brute = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    assert A.auc(pos, neg) == pytest.approx(brute)
    assert A.auc([], [1]) != A.auc([], [1])
    acc = {A.BLIND: {"auc": 0.8, "yes_rate_failures": 0.2, "yes_rate_successes": 0.9}}
    assert A.competence(acc)["interpretable"]
    for bad in ({"auc": 0.69}, {"yes_rate_failures": 0.96}, {"yes_rate_successes": 0.04}):
        assert not A.competence({A.BLIND: {**acc[A.BLIND], **bad}})["interpretable"]


def test_model_report_parts_add_up_and_sensitivities_run():
    states, L = _four_states()
    rep = A.model_report(states, L, L)
    assert rep["by_encounter"]["parts_add_up"]
    assert rep["primary"]["identity_holds"]
    assert rep["fallback_rate"] == {"aware": 0.0, "blind": 0.0}
    assert set(rep["sensitivity"]) == {"game_weighted", "first_12_steps", "k_2", "k_4", "fallback_max_L"}
    assert rep["acceptance"][A.AWARE]["by_group"]["regime"]["yes_rate"] == 1.0
    assert rep["acceptance"]["aware_minus_blind"]["regime"]["yes_rate_shift"] == 1.0


def test_complete_states_need_every_arm_where_defined():
    states, L = _four_states()
    keep, short = A.complete_states(states, L)
    assert short == 0
    del L[A.SWAPPED][A.cand_id(states[0], states[0]["candidates"][0])]
    keep, short = A.complete_states(states, L)
    assert short == 1 and states[0] not in keep
    L2 = {a: v for a, v in L.items() if a != A.SWAPPED}
    assert A.complete_states(states, L2)[1] == 0  # swapped not scored at all: not required


def test_disagreements_feed_stage2():
    states, L = _four_states()
    dis = A.disagreements(states, L)
    assert sorted(x["cell"] for x in dis) == ["both_succeed", "harm", "help"]
    feed = A.stage2_feed(dis)
    assert feed["harm"] == 1 and feed["harm_games"] == 1 and feed["disagreement_states"] == 3


# ---------------- labels ----------------


def test_progress_events():
    pe = A.progress_event
    assert pe("take saltshaker 1 from shelf 1", "You pick up the saltshaker 1 from the shelf 1.", "saltshaker", "cabinet", HIST, False)
    assert not pe("take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.", "saltshaker", "cabinet", HIST, False)
    assert pe("move saltshaker 1 to cabinet 1", "You move the saltshaker 1 to the cabinet 1.", "saltshaker", "cabinet", HIST, False)
    assert not pe("move saltshaker 1 to shelf 1", "You move the saltshaker 1 to the shelf 1.", "saltshaker", "cabinet", HIST, False)
    assert pe("go to cabinet 1", "You arrive at cabinet 1. On the cabinet 1, you see a saltshaker 2.", "saltshaker", "cabinet", HIST, False)
    assert not pe("go to cabinet 1", "You arrive at cabinet 1. On the cabinet 1, you see a saltshaker 2.", "saltshaker", "cabinet",
                  HIST + "\nYou see a saltshaker 2.", False)
    assert not pe("take saltshaker 1 from shelf 1", "Nothing happens.", "saltshaker", "cabinet", HIST, True)
    assert pe("look", "You are in the room.", "saltshaker", "cabinet", HIST, True)
    assert pe("use desklamp 1", "You turn on the desklamp 1.", "cd", "desklamp", HIST, False)


def _scene():
    traits = {"pickupable": {"mug", "cup", "plate", "saltshaker", "pen"}, "heatable": set(), "coolable": set(),
              "cleanable": set(), "can_contain": {"cabinet": {"mug", "cup", "plate", "saltshaker", "pen"}}}
    return Scene(receptacles=["cabinet 1", "shelf 1", "drawer 1"],
                 locations={"mug 1": {"shelf 1"}, "cup 1": {"drawer 1"}, "plate 1": {"cabinet 1"},
                            "saltshaker 1": {"drawer 1"}, "pen 1": {"cabinet 1"}},
                 traits=traits)


def test_swap_object_excludes_named_interacted_and_co_located_types():
    sc = _scene()
    acts = ["take saltshaker 1 from drawer 1", "examine plate 1"]
    pen_hist = HIST + "\n> take pen 1 from cabinet 1\nNothing happens."
    swap = A.state_swap_object(sc, TASK, "saltshaker", acts, pen_hist, "g", 3)
    # drawer 1 holds cup and saltshaker; plate is named; pen is acted on in the history: only mug remains
    assert swap == "mug"
    assert A.state_swap_object(sc, TASK, "saltshaker", acts + ["take mug 1 from shelf 1"], pen_hist, "g", 3) is None
    assert A.state_swap_object(sc, TASK, "saltshaker", acts + ["take mug 1 from shelf 1"], HIST, "g", 3) == "pen"
    many = A.state_swap_object(sc, TASK, "saltshaker", [], HIST, "g", 3)
    assert many == A.state_swap_object(sc, TASK, "saltshaker", [], HIST, "g", 3)  # seeded


# ---------------- mock runs end to end ----------------


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    """Mock RQ3 'none' episodes on the simulator (and the prompts the RQ3 agent saw), under a temporary runs root."""
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    logodds._fewshot_hidden.cache_clear()
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    prompts: dict[str, list] = {}
    current = {}
    orig = AgentClient.complete

    def record(self, messages, max_tokens=256):
        prompts.setdefault(current["g"], []).append(messages)
        return orig(self, messages, max_tokens)

    monkeypatch.setattr(AgentClient, "complete", record)
    client, agent = AlfworldClient(), AgentClient()
    client.create()
    eps = []
    for g in ("train/1", "train/2", "train/3", "train/6", "train/7"):
        current["g"] = g
        eps.append({**run_episode(client, agent, g, "train", "rq3_mock", ForesightConfig("none"), step_cap=6), "shard": 0})
    monkeypatch.setattr(AgentClient, "complete", orig)
    d = tmp_path / "rq3_mock" / "shard_00_of_01"
    d.mkdir(parents=True)
    (d / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    yield tmp_path, eps, prompts
    logodds._fewshot_hidden.cache_clear()


def _sample(i=0, n=2, **kw):
    return A.run_decisionsim_agent_sample({"run_id": "dsa_sample", "episodes_source": "rq3_mock", "split": "train",
                                           "n_shards": n, "shard": i, **kw})


def _label(i=0, n=2, **kw):
    return A.run_decisionsim_agent_label({"run_id": "dsa_label", "sample_source": "dsa_sample", "n_shards": n, "shard": i, **kw})


def _score(tag, **kw):
    return A.run_decisionsim_agent_score({"run_id": f"dsa_score_{tag}", "model": tag, "label_source": "dsa_label",
                                          "n_shards": 1, **kw})


def _analysis():
    return A.run_decisionsim_agent_analysis({"run_id": "dsa_analysis", "label_source": "dsa_label",
                                             "models": {t: f"dsa_score_{t}" for t in A.MODEL_TAGS}})


def _pipeline():
    assert _sample(0) == 0 and _sample(1) == 0
    assert _label(0) == 0 and _label(1) == 0
    for tag in A.MODEL_TAGS:
        assert _score(tag) == 0
    assert _analysis() == 0


def _states(root, run="dsa_sample", name="states.jsonl"):
    return sorted((r for p in sorted((root / run).glob("shard_*")) for r in read_jsonl(p / name)),
                  key=lambda r: (r["game_id"], r["state_idx"]))


def test_end_to_end_every_decision_point_once(mock_runs):
    root, eps, _ = mock_runs
    _pipeline()
    states = _states(root)
    assert [(s["game_id"], s["state_idx"]) for s in states] == sorted((e["game_id"], s["t"]) for e in eps for s in e["trace"])
    for s in states:
        ep = next(e for e in eps if e["game_id"] == s["game_id"])
        assert s["candidates"][0]["action"] == ep["trace"][s["state_idx"]]["executed"]
        assert s["prefix_actions"] == [x["executed"] for x in ep["trace"][: s["state_idx"]]]
        assert s["n_samples"] == A.N_SAMPLES and len(s["candidates"]) <= A.K_MAX
    labs = _states(root, "dsa_label", "labels.jsonl")
    assert len(labs) == len(states) and all(x["rank1_replay_ok"] for x in labs)
    assert any(A.regime(c) for x in labs for c in x["candidates"])
    assert any(c["admissible"] is False and c["failed"] for x in labs for c in x["candidates"])
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    assert set(m["models"]) == set(A.MODEL_TAGS) and not m["waiting"]
    for tag, rep in m["models"].items():
        assert rep["primary"]["identity_holds"] and rep["n_states"] == len(states)
        assert rep["decomposition"]["identity_holds"] and rep["by_encounter"]["parts_add_up"]
    assert set(m["primary"]["per_model"]) == set(A.PRIORITY)
    assert all(v["p_holm"] is not None for v in m["primary"]["per_model"].values())
    for v in m["primary"]["per_model"].values():  # the mock scorer is no better than chance: never interpreted
        assert not v["interpretable"] and v["verdict"] == "not interpreted"
    assert m["encounter"]["n_states"] == len(states)


def test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings(mock_runs, monkeypatch):
    root, eps, prompts = mock_runs
    seen = []
    orig = A.AgentSampler.sample

    def spy(self, messages, n, temperature, top_p, max_tokens, seed):
        seen.append((messages, n, temperature, top_p, max_tokens, seed))
        return orig(self, messages, n, temperature, top_p, max_tokens, seed)

    monkeypatch.setattr(A.AgentSampler, "sample", spy)
    assert _sample(0, n=1) == 0
    rq3 = [m for e in sorted(eps, key=lambda e: e["game_id"]) for m in prompts[e["game_id"]]]
    assert [x[0] for x in seen] == rq3
    states = _states(root)
    assert [x[1:] for x in seen] == [(20, 1.0, 1.0, 256, A.sample_seed(s["game_id"], s["state_idx"])) for s in states]


def test_real_sampler_sends_the_designed_payload(monkeypatch):
    monkeypatch.delenv("WF_MOCK_AGENT", raising=False)
    sent = {}

    class R:
        ok = True

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Action: look"}, "finish_reason": "stop"}] * 3}

    s = A.AgentSampler(base_url="http://x/v1")
    monkeypatch.setattr(s._session, "post", lambda url, json, timeout: (sent.update(json), R())[1])
    out = s.sample([{"role": "user", "content": "hi"}], 3, 1.0, 1.0, 256, 7)
    assert len(out) == 3 and sent["n"] == 3 and sent["temperature"] == 1.0 and sent["top_p"] == 1.0
    assert sent["top_k"] == -1 and sent["seed"] == 7 and sent["max_tokens"] == 256
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(RuntimeError):
        s.sample([{"role": "user", "content": "hi"}], 4, 1.0, 1.0, 256, 7)


def test_sample_resume_draws_nothing_new_and_refuses_another_prompt(mock_runs, monkeypatch):
    root, eps, _ = mock_runs
    assert _sample(0, n=1) == 0
    before = (root / "dsa_sample" / "shard_00_of_01" / "states.jsonl").read_text()
    calls = []
    monkeypatch.setattr(A.AgentSampler, "sample", lambda *a, **k: calls.append(1) or [])
    (root / "dsa_sample" / "shard_00_of_01" / "games_done.jsonl").write_text("")  # walk every game again
    assert _sample(0, n=1, resume=True) == 0
    assert not calls and (root / "dsa_sample" / "shard_00_of_01" / "states.jsonl").read_text() == before
    p = root / "dsa_sample" / "shard_00_of_01" / "states.jsonl"
    rows = read_jsonl(p)
    rows[0]["prompt_sha"] = "other"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "dsa_sample" / "shard_00_of_01" / "games_done.jsonl").write_text("")
    with pytest.raises(SystemExit):
        _sample(0, n=1, resume=True)


def test_a_fresh_sample_run_does_not_append_to_old_lists(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    first = _states(root)
    calls = []
    orig = A.AgentSampler.sample
    monkeypatch.setattr(A.AgentSampler, "sample", lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    assert _sample(0, n=1) == 0
    assert _states(root) == first and len(calls) == len(first)  # sampled afresh, not read back from the old file


def test_replay_mismatch_stops_the_game_there(mock_runs):
    root, eps, _ = mock_runs
    p = root / "rq3_mock" / "shard_00_of_01" / "episodes.jsonl"
    rows = read_jsonl(p)
    rows[0]["trace"][2]["real_obs"] = "something else"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert _sample(0, n=1) == 0
    g = rows[0]["game_id"]
    assert [s["state_idx"] for s in _states(root) if s["game_id"] == g] == [0, 1, 2]
    mm = read_jsonl(root / "dsa_sample" / "shard_00_of_01" / "mismatches.jsonl")
    assert [(x["game_id"], x["state_idx"]) for x in mm] == [(g, 2)]


def test_label_waits_for_every_sampling_shard(mock_runs):
    assert _sample(0) == 0
    assert _label(0) == 1


def test_rank1_mismatch_makes_the_state_unusable(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    p = root / "dsa_sample" / "shard_00_of_01" / "states.jsonl"
    rows = read_jsonl(p)
    rows[1]["recorded_obs"] = "not what the server says"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert _label(0, n=1) == 0
    labs = _states(root, "dsa_label", "labels.jsonl")
    bad = [x for x in labs if not x["rank1_replay_ok"]]
    assert [(x["game_id"], x["state_idx"]) for x in bad] == [(rows[1]["game_id"], rows[1]["state_idx"])]
    assert _score("qwen3-8b") == 0
    scored = {(x["game_id"], x["state_idx"]) for x in read_jsonl(root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl")}
    assert (rows[1]["game_id"], rows[1]["state_idx"]) not in scored


def test_label_stops_partial_and_resumes_to_the_same_labels(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    assert _label(0, n=1) == 0
    whole = _states(root, "dsa_label", "labels.jsonl")
    assert _label(0, n=1, max_replays_per_job=10) == 1
    assert "PARTIAL" in (root / "dsa_label" / "shard_00_of_01" / "STATUS.md").read_text()
    while _label(0, n=1, max_replays_per_job=10, resume=True) == 1:
        pass
    assert _states(root, "dsa_label", "labels.jsonl") == whole


def test_labels_match_a_fresh_replay_on_the_simulator(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    from wf.data.transitions import context_determined_label
    from wf.env.replay import history_text_from_state, replay, step_counterfactual

    client = AlfworldClient()
    client.create()
    for x in _states(root, "dsa_label", "labels.jsonl")[:6]:
        for c in x["candidates"]:
            h = replay(client, x["game_id"], x["prefix_actions"], split="train")
            assert history_text_from_state(h.state) == x["history_text"]
            obs, _ = step_counterfactual(h, c["action"])
            assert obs == c["true_obs"] and c["failed"] == (obs == "Nothing happens.")
            assert c["targets_goal_object"] == (x["goal_object"] in c["action"])
            assert c["context_determined"] == context_determined_label(x["history_text"], c["action"], obs)


def test_each_arm_sends_its_own_prompt_once_per_candidate(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    sent = []
    orig = logodds.LogoddsScorer.score_one_token_pair

    def spy(self, transition_id, condition, messages, o_plus, o_minus):
        sent.append((transition_id, condition, messages))
        return orig(self, transition_id, condition, messages, o_plus, o_minus)

    monkeypatch.setattr(logodds.LogoddsScorer, "score_one_token_pair", spy)
    assert _score("llama3.1-8b") == 0
    labs = {f"{x['game_id']}#{x['state_idx']:02d}": x for x in _states(root, "dsa_label", "labels.jsonl")}
    keys = [(t, c) for t, c, _ in sent]
    assert len(keys) == len(set(keys))
    for tid, cond, msgs in sent:
        st = labs[tid.rsplit("#", 1)[0]]
        last = msgs[-1]["content"]
        assert last.endswith(YN_QUESTION)
        if cond == A.AWARE:
            assert st["task_line"] in last
        elif cond == A.BLIND:
            assert "Your task is to:" not in last
        else:
            assert st["swapped_task_line"] in last and st["swap"]
    n_cands = sum(len(A.scored_candidates(x)) for x in labs.values())
    assert sum(c == A.AWARE for _, c, _ in sent) == n_cands == sum(c == A.BLIND for _, c, _ in sent)


def test_score_resume_and_refusal_of_other_prompts(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("qwen3-8b") == 0
    p = root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl"
    before = p.read_text()
    assert _score("qwen3-8b", resume=True) == 0 and p.read_text() == before
    rows = read_jsonl(p)
    rows[0]["score_hash"] = "x"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        _score("qwen3-8b", resume=True)


def test_scores_are_saved_one_at_a_time(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    orig = logodds.LogoddsScorer.score_one_token_pair
    n = {"k": 0}

    def flaky(self, *a, **k):
        n["k"] += 1
        if n["k"] == 4:
            raise RuntimeError("node lost")
        return orig(self, *a, **k)

    monkeypatch.setattr(logodds.LogoddsScorer, "score_one_token_pair", flaky)
    with pytest.raises(RuntimeError):
        _score("qwen3-8b")
    assert len(read_jsonl(root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl")) == 3


def test_a_failed_gate_scores_nothing(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    import wf.decisionsim as D

    spec = families.MODELS["mistral-7b-v0.3"]

    def bad_scorer(tag):
        s = _scorer(FakeTok(yes_pieces=2), spec)
        s.mock = False
        return s

    monkeypatch.setattr(D, "open_scorer", bad_scorer)
    with pytest.raises(families.TokenizationMismatch):
        _score("mistral-7b-v0.3")
    d = root / "dsa_score_mistral-7b-v0.3" / "shard_00_of_01"
    assert not (d / "scoring.jsonl").exists() and json.loads((d / "tokenization_checks.json").read_text())["ok"] is False


def test_arms_must_include_aware_and_blind(mock_runs):
    with pytest.raises(SystemExit):
        _score("qwen3-8b", arms=["yn_aligned", "yn_swapped"])
    with pytest.raises(SystemExit):
        _score("gemma2-9b-it")


def test_analysis_waits_and_refuses_scores_of_another_label_run(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("qwen3-8b") == 0
    cfgp = root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "config.json"
    cfg = json.loads(cfgp.read_text())
    cfgp.write_text(json.dumps({**cfg, "label_source": "other"}))
    assert _analysis() == 0
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    assert "qwen3-8b" in m["waiting"] and "llama3.1-8b" in m["waiting"] and not m["models"]
    assert m["primary"]["per_model"] == {}


def test_holm_waits_for_every_priority_model(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("llama3.1-8b") == 0
    assert _analysis() == 0
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    v = m["primary"]["per_model"]["llama3.1-8b"]
    assert v["p_holm"] is None and v["verdict"] == "waiting for every priority model"


def test_dry_run_on_mock_archives(mock_runs):
    root, eps, _ = mock_runs
    client = AlfworldClient()
    client.create()
    rows = []
    for e in eps:
        st = client.reset(e["game_id"], "train")
        rows.append({"game_id": e["game_id"], "history_len": 0, "history_text": st.first_observation})
    (root / "coll").mkdir()
    (root / "coll" / "transitions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert A.run_decisionsim_agent_dryrun({"run_id": "dsa_dry", "episodes_source": "rq3_mock",
                                           "first_obs_sources": ["coll"]}) == 0
    m = json.loads((root / "dsa_dry" / "metrics.json").read_text())
    assert m["n_states"] == sum(len(e["trace"]) for e in eps) and m["n_history_problems"] == 0
    assert m["plumbing_known_answer"]["is_zero"]
    fails = sum(s["real_obs"] == "Nothing happens." for e in eps for s in e["trace"])
    assert m["rank1"]["candidates"]["failed"] == fails


def test_dry_run_history_is_the_replayed_history(mock_runs):
    root, eps, _ = mock_runs
    from wf.env.replay import history_text_from_state, replay

    client = AlfworldClient()
    client.create()
    e = eps[0]
    first = client.reset(e["game_id"], "train").first_observation
    for t in range(len(e["trace"])):
        h = replay(client, e["game_id"], [x["executed"] for x in e["trace"][:t]], split="train")
        assert A.archived_history(first, e["trace"], t) == history_text_from_state(h.state)


def test_mock_agent_prompt_carries_admissible_actions(mock_runs):
    _, eps, prompts = mock_runs
    msgs = prompts[eps[0]["game_id"]][0]
    assert msgs == agent_react_messages(observation=msgs[-1]["content"].split("Observation: ", 1)[1].split("\nAdmissible", 1)[0],
                                        history=[], admissible=[ln[2:] for ln in msgs[-1]["content"].split("Admissible actions:\n", 1)[1]
                                                                .split("\nRespond", 1)[0].splitlines()])


def test_lists_are_saved_one_state_at_a_time(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    orig = A.AgentSampler.sample
    n = {"k": 0}

    def flaky(self, *a, **k):
        n["k"] += 1
        if n["k"] == 3:
            raise RuntimeError("vLLM went away")
        return orig(self, *a, **k)

    monkeypatch.setattr(A.AgentSampler, "sample", flaky)
    with pytest.raises(RuntimeError):
        _sample(0, n=1)
    assert len(read_jsonl(root / "dsa_sample" / "shard_00_of_01" / "states.jsonl")) == 2


def test_labels_are_saved_one_state_at_a_time(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    orig = A.label_state
    n = {"k": 0}

    def flaky(*a, **k):
        n["k"] += 1
        if n["k"] == 3:
            raise RuntimeError("server went away")
        return orig(*a, **k)

    monkeypatch.setattr(A, "label_state", flaky)
    with pytest.raises(RuntimeError):
        _label(0, n=1)
    assert len(read_jsonl(root / "dsa_label" / "shard_00_of_01" / "labels.jsonl")) == 2


def test_label_state_marks_determination_by_the_rq2_rules(mock_runs):
    from wf.data.transitions import context_determined_label
    from wf.env.scene import scan_scene

    client = AlfworldClient()
    client.create()
    st = client.reset("train/1", "train")
    s = {"game_id": "train/1", "split": "train", "state_idx": 0, "task_line": st.task_line, "prefix_actions": [],
         "recorded_obs": None, "n_format_errors": 0, "n_samples": 0, "n_cut": 0,
         "candidates": [{"rank": 1, "action": "inventory"}, {"rank": 2, "action": "go to drawer 1"},
                        {"rank": 3, "action": "take apple 1 from countertop 1"}]}
    lab, used = A.label_state(client, scan_scene(client, "train/1", "train"), s)
    assert used == 3 and not lab["rank1_replay_ok"]  # recorded reply None: rank 1 cannot match
    inv, go, take = lab["candidates"]
    assert not inv["failed"] and inv["context_determined"]  # no news in the reply: determined success
    assert not go["failed"] and not go["context_determined"]  # reveals contents: not determined
    assert take["failed"] and take["context_determined"] and take["admissible"] is False
    for c in lab["candidates"]:
        assert c["context_determined"] == context_determined_label(lab["history_text"], c["action"], c["true_obs"])
