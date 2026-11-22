"""Stage 1b (wf/decisionsim_agent.py, notes/decisionsim_agent_design.md, amended after review 1): synthetic checks of
every rule, and mock runs end to end (sample, label, score, analysis), on the simulator and the mock agent and scorer."""
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
from wf.util import read_jsonl

TASK = "Your task is to: put some saltshaker on cabinet."
FIRST = "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a shelf 1.\n" + TASK
HIST = FIRST + "\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a mug 1."


# ---------------- synthetic states (list views) ----------------


def _cand(rank, failed, goal=False, det=True, shared=False, admissible=True, action=None, progress=False, err=None,
          count=1):
    a = action or f"act {rank}"
    return {"key": A.akey(a), "rank": rank, "action": a, "count": count, "greedy": rank == 1,
            "true_obs": None if err else ("Nothing happens." if failed else "You arrive at x."), "server_error": err,
            "failed": failed, "context_determined": det, "targets_goal_object": goal, "shared_location": shared,
            "admissible": admissible, "action_type": "take" if goal else "go to", "progress": progress}


def _state(cands, game="g1", t=2, swap="mug", agnostic=True):
    return {"game_id": game, "state_idx": t, "task_line": TASK, "goal_object": "saltshaker", "target": "cabinet",
            "history_text": HIST, "history_goal_agnostic": agnostic, "swap": swap,
            "swapped_task_line": TASK.replace("saltshaker", swap) if swap else None, "rank1_replay_ok": True,
            "history_replay_ok": True, "candidates": cands}


def _lab(cands, t1=None, **kw):
    """A labelled state whose primary list is `cands` (and whose t1 list is `t1`, default the same)."""
    s = _state([{k: v for k, v in c.items() if k not in ("rank", "count", "greedy")} for c in cands + (t1 or [])], **kw)
    uniq = {c["key"]: c for c in s["candidates"]}
    s["candidates"] = list(uniq.values())
    entry = lambda cs: [{"rank": c["rank"], "action": c["action"], "count": c["count"], "greedy": c["greedy"]} for c in cs]
    s["lists"] = {A.PRIMARY_LIST: entry(cands), A.T1_LIST: entry(t1 or cands)}
    return s


def _L(state, values):
    return {A.cand_id(state, c): v for c, v in zip(state["candidates"], values)}


def _scores(states, per_arm):
    """per_arm[arm] = list (per state) of lists of L in candidate order."""
    return {arm: {k: v for s, vals in zip(states, lists) for k, v in _L(s, vals).items()} for arm, lists in per_arm.items()}


# ---------------- candidate lists ----------------


def test_normalize_collapses_space_case_and_final_stop():
    assert A.normalize_action("  Go  to   Drawer 1. ") == "go to drawer 1"
    assert A.normalize_action(None) is None and A.normalize_action(" . ") is None
    assert A.akey("Go to Drawer 1.") == A.akey("go to drawer 1") != A.akey("go to drawer 2")


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
    assert acts[3:] == sorted(["a", "c"], key=lambda a: A.tie_key("g", 4, a))
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
    assert A.K_MAX == 8 and A.N_SAMPLES == 20 and A.MAX_TOKENS == 256


def test_sampling_settings_after_review_1():
    assert A.SAMPLINGS[A.PRIMARY_LIST] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    assert A.SAMPLINGS[A.T1_LIST] == {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
    assert A.GREEDY["temperature"] == 0.0 and A.GREEDY_AGREEMENT_MIN == 0.90
    assert A.FIXED_PARAMS == {"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def test_bookkeeping_flags_identical_replies_drift_and_seed_check():
    lst = lambda texts, fin="stop": {"n_samples": len(texts), "n_format_errors": 0, "n_cut": 0,
                                     "candidates": [{"count": 3}], "replies": [{"text": x, "finish_reason": fin} for x in texts]}
    st = {"lists": {"primary": lst(["Action: look"] * 3)}, "greedy_fresh": {"agrees": True},
          "seed_check": {"distinct_first": 3, "same_set": True}}
    other = {"lists": {"primary": lst(["Action: look", "Action: x"], "length")}, "greedy_fresh": {"agrees": False},
             "seed_check": {"distinct_first": 1, "same_set": True}}
    b = A.sampling_bookkeeping([st, other])
    assert b["primary"]["states_all_replies_identical"] == 1 and b["primary"]["truncated_replies"] == 2
    assert b["greedy_agreement"] == 0.5 and b["drifted_sampler"]
    assert b["seed_check"] == {"n_states": 2, "nondegenerate": 1, "nondegenerate_same_set": 1, "same_seed_reproduces": 0}
    gate = A.smoke_gate({**b, "replay_mismatches": 0})
    assert not gate["passed"]  # a second seed reproduced a non-degenerate set
    ok = A.sampling_bookkeeping([{**st, "seed_check": {"distinct_first": 3, "same_set": False}}])
    assert A.smoke_gate({**ok, "replay_mismatches": 0})["passed"]
    assert not A.smoke_gate({**ok, "replay_mismatches": 1})["passed"]
    bad = A.sampling_bookkeeping([{**st, "lists": {"primary": {**lst(["x"] * 4), "n_format_errors": 1}},
                                   "seed_check": {"distinct_first": 3, "same_set": False}}])
    assert not A.smoke_gate({**bad, "replay_mismatches": 0})["passed"]  # 25% format errors is not under 25%


def test_sample_seed_is_per_state_per_list_and_fixed():
    assert A.sample_seed("g", 1) == A.sample_seed("g", 1) != A.sample_seed("g", 2) != A.sample_seed("h", 1)
    assert A.sample_seed("g", 1, name="primary") != A.sample_seed("g", 1, name="t1")


# ---------------- verifier prompts ----------------


def test_blind_history_deletes_only_the_real_task_line_in_the_first_observation():
    assert A.blind_history(HIST, TASK) == HIST.replace("\n" + TASK, "")
    with pytest.raises(ValueError):
        A.blind_history("no task here", TASK)
    with pytest.raises(ValueError):
        A.blind_history(HIST + "\n" + TASK, TASK)  # two task lines
    with pytest.raises(ValueError):
        A.blind_history(HIST, "Your task is to: put some mug on cabinet.")  # not the real task line
    with pytest.raises(ValueError):
        A.blind_history(TASK + "\nYou are in the middle of a room.", TASK)  # the first line
    moved = "You are in the middle of a room.\n> go to shelf 1\n" + TASK
    with pytest.raises(ValueError):
        A.blind_history(moved, TASK)  # after the first action


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
    aware, blind, swapped, neutral = (A.arm_messages(tag, s, c, a) for a in (A.AWARE, A.BLIND, A.SWAPPED, A.NEUTRAL))
    assert aware[:-1] == blind[:-1] == swapped[:-1] == neutral[:-1]  # the examples and system text never change
    assert all(m[-1]["content"].endswith(YN_QUESTION) for m in (aware, blind, swapped, neutral))
    assert TASK in aware[-1]["content"] and "Your task is to:" not in blind[-1]["content"]
    assert "put some mug on cabinet." in swapped[-1]["content"] and "saltshaker on" not in swapped[-1]["content"]
    assert "Your task is to: explore the room." in neutral[-1]["content"]
    assert blind[-1]["content"] == aware[-1]["content"].replace("\n" + TASK, "")
    assert "take saltshaker 1 from cabinet 1" in aware[-1]["content"]
    assert any("Your task is to:" in m["content"] for m in blind[:-1])  # the examples keep theirs
    assert A.arm_messages(tag, {**s, "swap": None, "swapped_task_line": None}, c, A.SWAPPED) is None


def test_prompts_are_the_earlier_yes_no_prompts(mock_score):
    from wf.decisionsim import messages_for

    s = _state([_cand(1, True, action="take saltshaker 1 from cabinet 1")])
    item = A.prompt_item(s, s["candidates"][0])
    for tag in A.MODEL_TAGS:
        for arm in ("yn_aligned", "yn_swapped", "yn_absent"):
            assert A.arm_messages(tag, s, s["candidates"][0], arm) == messages_for(tag)(item, arm)


def test_arms_at_a_single_candidate_state_is_blind_only():
    one = _state([_cand(1, True)])
    two = _state([_cand(1, True), _cand(2, False)])
    assert A.arms_at(one, A.ARMS) == (A.BLIND,)
    assert A.arms_at(two, A.ARMS) == A.ARMS
    assert A.arms_at({**two, "swap": None}, A.ARMS) == (A.AWARE, A.BLIND, A.NEUTRAL)
    assert A.arms_at(_state([_cand(1, True), _cand(2, False, err="x")]), A.ARMS) == (A.BLIND,)


# ---------------- views ----------------


def test_views_follow_each_list_and_share_labels():
    a, b, c = _cand(1, True), _cand(2, False, count=3), _cand(3, True, goal=True)
    t1c = {**c, "rank": 2}
    lab = _lab([a, b], t1=[{**a}, t1c])
    v = A.view(lab, A.PRIMARY_LIST)
    assert [x["action"] for x in v["candidates"]] == ["act 1", "act 2"] and v["n_union"] == 3
    w = A.view(lab, A.T1_LIST)
    assert [x["action"] for x in w["candidates"]] == ["act 1", "act 3"] and w["candidates"][1]["rank"] == 2
    assert A.cand_id(v, v["candidates"][0]) == A.cand_id(w, w["candidates"][0])  # one cache entry for both lists


def test_sampled_twice_and_first_occurrence_filters():
    a, b, c = _cand(1, False), _cand(2, True, count=1, action="x"), _cand(3, True, count=2, action="y")
    lab = _lab([a, b, c])
    v = A.view(lab, A.PRIMARY_LIST, A.sampled_twice)
    assert [x["action"] for x in v["candidates"]] == ["act 1", "y"]
    s1 = _state([_cand(1, True, action="x"), _cand(2, False)], t=0)
    s2 = _state([_cand(1, True, action="x"), _cand(2, True, action="z"), _cand(3, False)], t=1)
    s3 = _state([_cand(1, True, action="x")], t=2)
    out = A.first_occurrence([s2, s1, s3])
    assert [[c["action"] for c in s["candidates"]] for s in out] == [["x", "act 2"], ["z", "act 3"]]


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


def test_a_single_candidate_is_chosen_without_scores():
    s = _state([_cand(1, True)])
    for cfg in ("none", "oracle", "aware", "blind"):
        assert A.choose(cfg, s, {A.AWARE: {}, A.BLIND: {}})["rank"] == 1


def test_oracle_with_no_success_falls_back_to_rank1():
    assert A.choose("oracle", _state([_cand(1, True), _cand(2, True)]))["rank"] == 1


def test_server_error_candidates_are_skipped_before_the_k_cut():
    s = _state([_cand(1, True), _cand(2, False, err="boom"), _cand(3, False)])
    L = _scores([s], {A.AWARE: [[-1, 5, 1]]})
    assert A.choose("aware", s, L)["rank"] == 3 and A.choose("oracle", s)["rank"] == 3
    assert [c["rank"] for c in A.scored_candidates(s)] == [1, 3]
    assert A.choose("aware", s, L, k=2)["rank"] == 3  # k counts runnable candidates
    assert A.choose("oracle", s, k=2)["rank"] == 3


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


def test_encounter_bands_and_the_futility_rule():
    sts = [_state([_cand(1, True, goal=True), _cand(2, False)], game=f"g{i}", t=t) for i, t in enumerate((0, 11, 12, 29))]
    rep = A.encounter_report(sts)
    b = rep["encounter_by_step_band"]
    assert (b["step_0"]["n_states"], b["steps_1_11"]["n_states"], b["steps_12_29"]["n_states"]) == (1, 1, 2)
    assert rep["reachable_games"] == 4 and rep["futility"]["games"] == 4
    assert rep["futility"]["conditional_cannot_reach_significance"]  # 4 games < 7
    # Encounters in 7 games but no working candidate anywhere: no state can differ, so futile.
    no_success = [_state([_cand(1, True, goal=True)], game=f"g{i}") for i in range(7)]
    rep = A.encounter_report(no_success)
    assert rep["reachable_games"] == 7 and rep["futility"]["games"] == 0
    assert rep["futility"]["conditional_cannot_reach_significance"]
    # Encounters in 7 games, a success always ranked above them: not reachable, yet a verifier that rejects the
    # success can still choose them, so the rule must not call this futile (review 2).
    unreachable = [_state([_cand(1, False), _cand(2, True, goal=True)], game=f"g{i}") for i in range(7)]
    rep = A.encounter_report(unreachable)
    assert rep["encounter_games"] == 7 and rep["reachable_games"] == 0 and rep["futility"]["games"] == 7
    assert not rep["futility"]["conditional_cannot_reach_significance"]
    six = A.encounter_report(unreachable[:6])
    assert six["futility"]["conditional_cannot_reach_significance"]  # the boundary: 6 games are not enough


def test_encounter_bands_count_encounters_not_reachable_ones():
    sts = [_state([_cand(1, False), _cand(2, True, goal=True)], game=f"g{i}", t=t) for i, t in enumerate((0, 5, 20))]
    b = A.encounter_report(sts)["encounter_by_step_band"]
    assert [b[k]["estimate"] for k in ("step_0", "steps_1_11", "steps_12_29")] == [1.0, 1.0, 1.0]


def test_reviewer_counterexample_eight_unreachable_games_can_be_significant():
    # 8 games, one encounter state each: rank 1 works (task-irrelevant), rank 2 a regime candidate, rank 3 works.
    labs = [_lab([_cand(1, False), _cand(2, True, goal=True), _cand(3, False)], game=f"g{i}", t=5) for i in range(8)]
    states = A.views(labs)
    enc = A.encounter_report(states)
    assert enc["reachable_games"] == 0 and enc["futility"]["games"] == 8
    assert not enc["futility"]["conditional_cannot_reach_significance"]
    L = {A.AWARE: {}, A.BLIND: {}, A.SWAPPED: {}}
    for s in labs:  # the aware verifier rejects the working rank 1 and accepts the regime candidate
        L[A.AWARE].update(_L(s, [-1, 1, 1]))
        L[A.BLIND].update(_L(s, [1, -1, 1]))
        L[A.SWAPPED].update(_L(s, [1, -1, 1]))
    c = A.contrast(states, L, "aware", "blind")
    assert c["difference"]["estimate"] == 1.0 and c["difference"]["n_games_nonzero"] == 8
    assert c["difference"]["p"] < 0.025 and c["harm_regime_share"] == 1.0


# ---------------- contrasts and statistics ----------------


def _four_states():
    s1 = _state([_cand(1, True), _cand(2, True, goal=True), _cand(3, False)], game="g1", t=0)  # harm
    s2 = _state([_cand(1, True), _cand(2, False)], game="g2", t=0)  # help
    s3 = _state([_cand(1, False), _cand(2, False)], game="g3", t=0)  # both succeed, changed
    s4 = _state([_cand(1, False), _cand(2, True)], game="g4", t=0, swap=None)  # same
    states = [s1, s2, s3, s4]
    L = _scores(states, {A.AWARE: [[-1, 1, 1], [-1, 1], [-1, 1], [1, 1]],
                         A.BLIND: [[-1, -1, 1], [1, 1], [1, 1], [1, 1]],
                         A.SWAPPED: [[-1, -1, 1], [-1, 1], [1, 1], [1, 1]],
                         A.NEUTRAL: [[-1, -1, 1], [1, 1], [1, 1], [1, 1]]})
    return states, L


def test_contrast_counts_harm_help_and_the_identity():
    states, L = _four_states()
    c = A.contrast(states, L, "aware", "blind")
    assert c["changed"] == {"n": 3, "harm": 1, "help": 1, "both_fail": 0, "both_succeed": 1}
    assert c["difference"]["estimate"] == 0 and c["identity_holds"]
    assert c["harm_by_kind_of_choice"] == {"regime": 1} and c["harm_regime_share"] == 1.0
    assert c["fail_rate_a"]["estimate"] == 0.25 and c["fail_rate_b"]["estimate"] == 0.25
    assert c["difference"]["n_games_nonzero"] == 2 and c["difference"]["p_min_sign_flip"] == 0.5


def test_leave_one_game_out():
    r = A.leave_one_game_out([1, 0, 0, 1], [0, 0, 0, 0], ["a", "b", "c", "c"])
    assert r["min"] == pytest.approx(1 / 3) and r["max"] == pytest.approx(2 / 3)
    assert r["most_influential_game"] == "a" and r["estimate_without_it"] == pytest.approx(1 / 3)


def test_decomposition_on_states_with_a_swap_and_its_identity():
    states, L = _four_states()
    dec = A.decomposition(states, L)
    assert dec["n_states"] == 3 and dec["identity_holds"]
    assert dec["aware_minus_swapped"]["estimate"] == pytest.approx(1 / 3)
    assert dec["swapped_minus_blind"]["estimate"] == pytest.approx(-1 / 3)


def test_attribution_rule():
    mk = lambda ab, as_: {"n_states": 5, "aware_minus_blind": {"estimate": ab}, "aware_minus_swapped": {"estimate": as_}}
    assert A.attributable(mk(0.02, 0.01)) and A.attributable(mk(0.02, 0.03))
    assert not A.attributable(mk(0.02, 0.009)) and not A.attributable(mk(0.02, -0.01))
    assert not A.attributable(mk(-0.02, -0.01)) and not A.attributable(None)
    ci_up = {"ci": [0.001, 0.02]}
    assert A.verdict(ci_up, 0.01, mk(0.02, 0.015)) == "toward failure, attributable to the real task"
    assert A.verdict(ci_up, 0.01, mk(0.02, 0.005)) == "toward failure, not attributable to the real task"


def test_inference_resamples_and_flips_whole_games(monkeypatch):
    seen = {}
    real_boot, real_perm = A.cluster_bootstrap, A.pair_permutation
    monkeypatch.setattr(A, "cluster_bootstrap", lambda f, v, g, **kw: (seen.__setitem__("boot", list(g)), real_boot(f, v, g, **kw))[1])
    monkeypatch.setattr(A, "pair_permutation", lambda d, **kw: (seen.__setitem__("perm", kw.get("cluster_ids")), real_perm(d, **kw))[1])
    states, L = _four_states()
    A.contrast(states, L, "aware", "blind")
    assert seen["boot"] == ["g1", "g2", "g3", "g4"] and seen["perm"] == ["g1", "g2", "g3", "g4"]


def test_holm_and_verdicts():
    assert A.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert A.holm({"a": float("nan"), "b": 0.01}) == {"b": 0.02, "a": 1.0}
    assert A.holm({"a": 0.03, "b": 0.031}) == {"a": 0.06, "b": 0.06}  # never below an earlier adjusted p
    assert A.verdict({"ci": [-0.02, -0.001]}, 0.03, None) == "toward success"
    assert A.verdict({"ci": [0.001, 0.02]}, 0.06, None) == "no reliable difference"
    assert A.verdict({"ci": [-0.001, 0.02]}, 0.01, None) == "no reliable difference"


def _model(p, ci, interpretable=True, harm=3, regime_share=1.0):
    block = {"difference": {"estimate": 0.01, "ci": ci, "p": p, "n_games_nonzero": 9, "p_min_sign_flip": 2 ** -8},
             "n_states": 100, "harm_regime_share": regime_share, "changed": {"harm": harm}}
    dec = {"n_states": 50, "aware_minus_blind": {"estimate": 0.02}, "aware_minus_swapped": {"estimate": 0.015}}
    return {"primary": block, "conditional": block, "competence": {"interpretable": interpretable},
            "decomposition": dec, "conditional_decomposition": dec}


def test_the_verdict_reads_the_holm_p_not_the_raw_p():
    models = {"llama3.1-8b": _model(0.03, [0.001, 0.02]), "mistral-7b-v0.3": _model(0.5, [-0.01, 0.02])}
    out = A.primary_block(models, list(A.PRIORITY), set(), "primary", "decomposition")["per_model"]
    assert out["llama3.1-8b"]["p_holm"] == pytest.approx(0.06)
    assert out["llama3.1-8b"]["verdict"] == "no reliable difference"  # raw p 0.03 with the interval above zero
    models["mistral-7b-v0.3"] = _model(0.02, [0.001, 0.02])
    out = A.primary_block(models, list(A.PRIORITY), set(), "primary", "decomposition")["per_model"]
    assert out["llama3.1-8b"]["verdict"] == "toward failure, attributable to the real task"


def test_a_failed_gate_enters_holm_with_p_one():
    models = {"llama3.1-8b": _model(0.02, [0.001, 0.02])}
    blk = A.primary_block(models, list(A.PRIORITY), {"mistral-7b-v0.3"}, "primary", "decomposition")["per_model"]
    assert blk["mistral-7b-v0.3"]["verdict"] == "not interpreted (Yes/No gate failed)"
    assert blk["llama3.1-8b"]["p_holm"] == pytest.approx(0.04)  # raw 0.02 < 0.025: still significant
    assert blk["llama3.1-8b"]["verdict"].startswith("toward failure")
    models = {"llama3.1-8b": _model(0.03, [0.001, 0.02])}
    blk = A.primary_block(models, list(A.PRIORITY), {"mistral-7b-v0.3"}, "primary", "decomposition")["per_model"]
    assert blk["llama3.1-8b"]["verdict"] == "no reliable difference"  # Llama needs raw p < 0.025
    wait = A.primary_block(models, list(A.PRIORITY), set(), "primary", "decomposition")["per_model"]
    assert wait["llama3.1-8b"]["p_holm"] is None and wait["llama3.1-8b"]["verdict"] == "waiting for every priority model"


def test_incompetent_models_stay_in_holm_and_harm_from_loops_is_flagged():
    models = {"llama3.1-8b": _model(0.01, [0.001, 0.02], interpretable=False),
              "mistral-7b-v0.3": _model(0.02, [0.001, 0.02], regime_share=0.2)}
    out = A.primary_block(models, list(A.PRIORITY), set(), "primary", "decomposition")["per_model"]
    assert out["llama3.1-8b"]["verdict"] == "not interpreted" and out["llama3.1-8b"]["p_holm"] == pytest.approx(0.02)
    assert out["mistral-7b-v0.3"]["harm_mostly_regime"] is False
    assert out["mistral-7b-v0.3"]["verdict"] == ("toward failure, attributable to the real task (not evidence for the "
                                                 "main finding: harm mostly from non-regime actions)")


def test_auc_and_competence():
    assert A.auc([2, 1], [0, 1]) == pytest.approx(0.875)
    rng = random.Random(3)
    pos, neg = [rng.choice([-1, 0, 0.5, 2]) for _ in range(40)], [rng.choice([-2, -1, 0, 0.5]) for _ in range(30)]
    brute = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    assert A.auc(pos, neg) == pytest.approx(brute)
    assert A.auc([], [1]) != A.auc([], [1])
    acc = {A.BLIND: {"auc": 0.70, "yes_rate_failures": 0.95, "yes_rate_successes": 0.05}}
    assert A.competence(acc)["interpretable"]  # the thresholds themselves pass
    for bad in ({"auc": 0.69}, {"yes_rate_failures": 0.96}, {"yes_rate_successes": 0.04}):
        assert not A.competence({A.BLIND: {**acc[A.BLIND], **bad}})["interpretable"]


def test_acceptance_counts_zero_as_no():
    s = _state([_cand(1, True), _cand(2, False)])
    L = _scores([s], {A.AWARE: [[0, 1]], A.BLIND: [[0, 0]]})
    acc = A.acceptance([s], L, L)
    assert acc[A.AWARE]["yes_rate_failures"] == 0 and acc[A.BLIND]["yes_rate_successes"] == 0
    assert acc["aware_minus_blind"]["success/all"]["yes_rate_shift"] == 1.0


def _labs_and_scores():
    states, L = _four_states()
    labs = []
    for s in states:
        labs.append(_lab(s["candidates"], game=s["game_id"], t=s["state_idx"], swap=s["swap"]))
    return labs, L


def test_model_report_parts_add_up_and_readouts():
    labs, L = _labs_and_scores()
    rep = A.model_report(labs, L, L)
    assert rep["by_encounter"]["parts_add_up"] and rep["by_encounter"]["encounter_states"] == 1
    assert rep["primary"]["identity_holds"] and rep["conditional"]["n_states"] == 1
    assert rep["conditional"]["difference"]["estimate"] == 1.0  # the harm state is the encounter state
    assert {k: rep["inert"]["aware"][k] for k in ("fallback_rate", "rank1_share")} == {"fallback_rate": 0.0, "rank1_share": 0.25}
    assert rep["inert"]["blind"]["rank1_share"] == 0.75
    assert set(rep["sensitivity"]) == {"game_weighted", "first_12_steps", "k_2", "k_4", "fallback_max_L", "t1_list",
                                       "t1_list_conditional", "sampled_twice", "first_occurrence_of_failures"}
    assert rep["acceptance"][A.AWARE]["by_group"]["regime"]["yes_rate"] == 1.0
    assert rep["neutral"]["aware_minus_neutral"]["estimate"] == 0.0
    assert rep["fail_rate"]["none"]["estimate"] == 0.5


def test_by_encounter_uses_the_main_not_the_strict_label():
    labs, L = _labs_and_scores()
    labs[0]["history_goal_agnostic"] = False
    rep = A.model_report(labs, L, L)
    assert rep["by_encounter"]["encounter_states"] == 1


def test_first_12_steps_means_steps_0_to_11():
    labs, L = _labs_and_scores()
    ids = {}
    for lab, t in zip(labs, (11, 12, 0, 0)):
        old = {c["key"]: A.cand_id(lab, c) for c in lab["candidates"]}
        lab["state_idx"] = t
        ids.update({old[c["key"]]: A.cand_id(lab, c) for c in lab["candidates"]})
    L = {a: {ids[k]: v for k, v in sc.items()} for a, sc in L.items()}
    assert A.model_report(labs, L, L)["sensitivity"]["first_12_steps"]["n_states"] == 3


def test_verifier_value_sign_and_progress_direction():
    s = _state([_cand(1, True), _cand(2, False, progress=True)], game="g1", t=0)
    labs = [_lab(s["candidates"], game="g1", t=0)]
    L = _scores(labs, {A.AWARE: [[1, 1]], A.BLIND: [[-1, 1]]})
    rep = A.model_report(labs, L, L)
    assert rep["verifier_value"]["blind"]["none_minus_verifier"]["estimate"] == 1.0
    assert rep["verifier_value"]["aware"]["none_minus_verifier"]["estimate"] == 0.0
    assert rep["progress_aware_minus_blind"]["estimate"] == -1.0


def test_game_weighted_delta_weighs_games_equally():
    states = [_state([_cand(1, True), _cand(2, False)], game="big", t=t) for t in range(3)]
    states.append(_state([_cand(1, False), _cand(2, True)], game="small", t=0))
    L = _scores(states, {A.AWARE: [[1, 1]] * 4, A.BLIND: [[-1, 1]] * 4})
    assert A.game_weighted_delta(states, L)["estimate"] == 0.0  # big: +1 on every state; small: -1
    assert A.contrast(states, L, "aware", "blind")["difference"]["estimate"] == 0.5


def test_complete_labs_need_every_arm_where_defined():
    labs, L = _labs_and_scores()
    assert A.complete_labs(labs, L)[1] == 0
    del L[A.SWAPPED][A.cand_id(labs[0], labs[0]["candidates"][0])]
    keep, short = A.complete_labs(labs, L)
    assert short == 1 and labs[0] not in keep
    assert A.complete_labs(labs, {a: v for a, v in L.items() if a not in (A.SWAPPED, A.NEUTRAL)})[1] == 0
    no_neutral = {a: dict(v) for a, v in L.items()}
    del no_neutral[A.NEUTRAL][A.cand_id(labs[1], labs[1]["candidates"][0])]
    keep, short = A.complete_labs(labs, no_neutral)  # labs[0] still lacks its swapped score from above
    assert short == 2 and labs[1] not in keep  # a scored neutral arm is needed where it was scored
    single = _lab([_cand(1, True)], game="g9")
    assert A.complete_labs([single], {A.AWARE: {}, A.BLIND: {A.cand_id(single, single["candidates"][0]): 1.0}})[1] == 0
    assert A.complete_labs([single], {A.AWARE: {}, A.BLIND: {}})[1] == 1


def test_disagreements_feed_stage2():
    states, L = _four_states()
    dis = A.disagreements(states, L)
    assert sorted(x["cell"] for x in dis) == ["both_succeed", "harm", "help"]
    feed = A.stage2_feed(dis)
    assert feed["harm"] == 1 and feed["harm_games"] == 1 and feed["disagreement_states"] == 3


# ---------------- labels ----------------


def test_progress_events():
    pe = A.progress_event
    assert pe("take saltshaker 1 from shelf 1", "You pick up the saltshaker 1 from the shelf 1.", TASK, HIST, False)
    assert not pe("take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.", TASK, HIST, False)
    assert pe("move saltshaker 1 to cabinet 1", "You move the saltshaker 1 to the cabinet 1.", TASK, HIST, False)
    assert not pe("move saltshaker 1 to shelf 1", "You move the saltshaker 1 to the shelf 1.", TASK, HIST, False)
    assert pe("go to cabinet 1", "You arrive at cabinet 1. On the cabinet 1, you see a saltshaker 2.", TASK, HIST, False)
    assert not pe("go to cabinet 1", "You arrive at cabinet 1. On the cabinet 1, you see a saltshaker 2.", TASK,
                  HIST + "\nYou see a saltshaker 2.", False)
    assert not pe("take saltshaker 1 from shelf 1", "Nothing happens.", TASK, HIST, True)
    assert pe("look", "You are in the room.", TASK, HIST, True)


def test_progress_needs_the_task_transformation_no_repeats_and_the_object_in_hand_for_the_lamp():
    pe = A.progress_event
    heat = "Your task is to: heat some egg and put it in countertop."
    assert pe("heat egg 1 with microwave 1", "You heat the egg 1 using the microwave 1.", heat, HIST, False)
    assert not pe("cool egg 1 with fridge 1", "You cool the egg 1 using the fridge 1.", heat, HIST, False)
    assert not pe("heat mug 1 with microwave 1", "You heat the mug 1 using the microwave 1.", heat, HIST, False)
    clean = "Your task is to: put a clean egg in countertop."
    assert not pe("clean mug 1 with sinkbasin 1", "You clean the mug 1 using the sinkbasin 1.", clean, HIST, False)
    again = HIST + "\n> take saltshaker 1 from shelf 1\nYou pick up the saltshaker 1 from the shelf 1."
    assert not pe("take saltshaker 1 from shelf 1", "You pick up the saltshaker 1 from the shelf 1.", TASK, again, False)
    failed_before = HIST + "\n> take saltshaker 1 from shelf 1\nNothing happens."
    assert pe("take saltshaker 1 from shelf 1", "You pick up the saltshaker 1 from the shelf 1.", TASK, failed_before, False)
    lamp = "Your task is to: look at cd under the desklamp."
    held = HIST + "\n> take cd 1 from shelf 1\nYou pick up the cd 1 from the shelf 1."
    assert pe("use desklamp 1", "You turn on the desklamp 1.", lamp, held, False)
    assert not pe("use desklamp 1", "You turn on the desklamp 1.", lamp, HIST, False)


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
    assert A.state_swap_object(sc, TASK, "saltshaker", acts, pen_hist, "g", 3) == "mug"
    assert A.state_swap_object(sc, TASK, "saltshaker", acts + ["take mug 1 from shelf 1"], pen_hist, "g", 3) is None
    assert A.state_swap_object(sc, TASK, "saltshaker", acts + ["take mug 1 from shelf 1"], HIST, "g", 3) == "pen"


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


def _model_free(run_id="dsa_mf"):
    return A.run_decisionsim_agent_analysis({"run_id": run_id, "label_source": "dsa_label", "models": {}})


def _score(tag, model_free=True, **kw):
    if model_free and "model_free_analysis" not in kw:
        assert _model_free() == 0
    arms = list(A.ARMS) if tag in A.PRIORITY else [A.AWARE, A.BLIND, A.SWAPPED]
    return A.run_decisionsim_agent_score({"run_id": f"dsa_score_{tag}", "model": tag, "label_source": "dsa_label",
                                          "n_shards": 1, "arms": arms, "model_free_analysis": "dsa_mf", **kw})


def _analysis():
    return A.run_decisionsim_agent_analysis({"run_id": "dsa_analysis", "label_source": "dsa_label",
                                             "models": {t: f"dsa_score_{t}" for t in A.MODEL_TAGS}})


def _pipeline():
    assert _sample(0) == 0 and _sample(1) == 0
    assert _label(0) == 0 and _label(1) == 0
    for tag in A.MODEL_TAGS:
        assert _score(tag) == 0
    assert _analysis() == 0


def _rows(root, run="dsa_sample", name="states.jsonl"):
    return sorted((r for p in sorted((root / run).glob("shard_*")) for r in read_jsonl(p / name)),
                  key=lambda r: (r["game_id"], r["state_idx"]))


def _rewrite(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _always_swap(monkeypatch):
    """The simulator's rooms are too small for the swap rules (every type sits at some candidate's receptacle), so the
    runs that exercise the swapped arm draw a fixed swap object instead; the rules are tested on their own above."""
    monkeypatch.setattr(A, "state_swap_object", lambda *a, **k: "plate")


def test_end_to_end_every_decision_point_once(mock_runs, monkeypatch):
    root, eps, _ = mock_runs
    _always_swap(monkeypatch)
    _pipeline()
    states = _rows(root)
    assert [(s["game_id"], s["state_idx"]) for s in states] == sorted((e["game_id"], s["t"]) for e in eps for s in e["trace"])
    for s in states:
        ep = next(e for e in eps if e["game_id"] == s["game_id"])
        assert set(s["lists"]) == {A.PRIMARY_LIST, A.T1_LIST}
        for lst in s["lists"].values():
            assert lst["candidates"][0]["action"] == ep["trace"][s["state_idx"]]["executed"]
            assert lst["n_samples"] == A.N_SAMPLES and len(lst["candidates"]) <= A.K_MAX
        assert s["prefix_actions"] == [x["executed"] for x in ep["trace"][: s["state_idx"]]]
        assert "agrees" in s["greedy_fresh"]
    labs = _rows(root, "dsa_label", "labels.jsonl")
    assert len(labs) == len(states) and all(x["rank1_replay_ok"] and x["history_replay_ok"] for x in labs)
    for lab, st in zip(labs, states):  # the union is labelled once: primary order first
        assert [c["action"] for c in lab["candidates"]] == A.union_actions(st)
    assert any(A.regime(c) for x in labs for c in x["candidates"])
    assert any(c["admissible"] is False and c["failed"] for x in labs for c in x["candidates"])
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    assert set(m["models"]) == set(A.MODEL_TAGS) and not m["waiting"] and not m["gate_failed"]
    for tag, rep in m["models"].items():
        assert rep["primary"]["identity_holds"] and rep["n_states"] == len(states)
        assert rep["decomposition"]["identity_holds"] and rep["by_encounter"]["parts_add_up"]
        assert ("neutral" in rep) == (tag in A.PRIORITY)
    for key in ("overall", "conditional_on_encounter"):
        per = m["primary"][key]["per_model"]
        assert set(per) == set(A.PRIORITY) and all(v["p_holm"] is not None for v in per.values())
        for v in per.values():  # the mock scorer is no better than chance: never interpreted
            assert not v["interpretable"] and v["verdict"] == "not interpreted"
    assert m["encounter"]["n_states"] == len(states) and "futility" in m["primary"]
    assert m["sampling"]["greedy_agreement"] == m["sampling"]["greedy_agreement"]


def test_analysis_runs_model_free_before_any_score(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    assert A.run_decisionsim_agent_analysis({"run_id": "dsa_enc", "label_source": "dsa_label", "models": {}}) == 0
    m = json.loads((root / "dsa_enc" / "metrics.json").read_text())
    assert m["encounter"]["n_states"] > 0 and m["primary"]["futility"]["rule"]
    assert m["encounter_t1_list"]["n_states"] == m["encounter"]["n_states"]


def test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings(mock_runs, monkeypatch):
    root, eps, prompts = mock_runs
    seen = []
    orig = A.AgentSampler.sample

    def spy(self, messages, n, params, max_tokens, seed):
        seen.append((messages, n, params, max_tokens, seed))
        return orig(self, messages, n, params, max_tokens, seed)

    monkeypatch.setattr(A.AgentSampler, "sample", spy)
    assert _sample(0, n=1) == 0
    rq3 = [m for e in sorted(eps, key=lambda e: e["game_id"]) for m in prompts[e["game_id"]]]
    assert [x[0] for x in seen[::3]] == rq3 and all(x[0] == y[0] for x, y in zip(seen[::3], seen[1::3]))
    states = _rows(root)
    expect = []
    for s in states:
        g, t = s["game_id"], s["state_idx"]
        expect += [(20, A.SAMPLINGS["primary"], 256, A.sample_seed(g, t, name="primary")),
                   (20, A.SAMPLINGS["t1"], 256, A.sample_seed(g, t, name="t1")),
                   (1, A.GREEDY, 256, A.sample_seed(g, t, name="greedy"))]
    assert [x[1:] for x in seen] == expect
    m = json.loads((root / "dsa_sample" / "shard_00_of_01" / "metrics.json").read_text())
    assert m["vllm"] == {"version": "mock"} and m["request_settings"]["primary"]["top_k"] == 20


def test_sampling_history_text_is_the_world_model_history(mock_runs):
    root, eps, _ = mock_runs
    assert _sample(0, n=1) == 0
    first = AlfworldClient()
    first.create()
    for s in _rows(root):
        ep = next(e for e in eps if e["game_id"] == s["game_id"])
        f = first.reset(s["game_id"], "train").first_observation
        assert s["history_text"] == A.archived_history(f, ep["trace"], s["state_idx"])


def test_seed_check_on_the_smoke_states(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1, seed_check=True, max_games=1) == 0
    m = json.loads((root / "dsa_sample" / "shard_00_of_01" / "metrics.json").read_text())
    assert m["seed_check"]["n_states"] > 0 and m["seed_check"]["nondegenerate_same_set"] == 0
    assert m["smoke_gate"]["passed"]


def test_real_sampler_sends_the_designed_payload(monkeypatch):
    monkeypatch.delenv("WF_MOCK_AGENT", raising=False)
    sent, got = {}, {}

    class R:
        ok = True

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Action: look"}, "finish_reason": "stop"}] * 3, "version": "0.8.5"}

    s = A.AgentSampler(base_url="http://x/v1")
    monkeypatch.setattr(s._session, "post", lambda url, json, timeout: (sent.update(json), R())[1])
    monkeypatch.setattr(s._session, "get", lambda url, timeout: (got.update(url=url), R())[1])
    out = s.sample([{"role": "user", "content": "hi"}], 3, A.SAMPLINGS["primary"], 256, 7)
    assert len(out) == 3 and sent["n"] == 3 and sent["temperature"] == 0.7 and sent["top_p"] == 0.8
    assert sent["top_k"] == 20 and sent["min_p"] == 0.0 and sent["seed"] == 7 and sent["max_tokens"] == 256
    assert sent["repetition_penalty"] == 1.0 and sent["presence_penalty"] == 0 and sent["frequency_penalty"] == 0
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(RuntimeError):
        s.sample([{"role": "user", "content": "hi"}], 4, A.SAMPLINGS["t1"], 256, 7)
    assert s.server_info()["version"] == "0.8.5" and got["url"] == "http://x/version"


def test_sample_resume_draws_nothing_new_and_refuses_another_prompt(mock_runs, monkeypatch):
    root, eps, _ = mock_runs
    assert _sample(0, n=1) == 0
    p = root / "dsa_sample" / "shard_00_of_01" / "states.jsonl"
    before = p.read_text()
    calls = []
    monkeypatch.setattr(A.AgentSampler, "sample", lambda *a, **k: calls.append(1) or [])
    (root / "dsa_sample" / "shard_00_of_01" / "games_done.jsonl").write_text("")  # walk every game again
    assert _sample(0, n=1, resume=True) == 0
    assert not calls and p.read_text() == before
    rows = read_jsonl(p)
    rows[0]["prompt_sha"] = "other"
    _rewrite(p, rows)
    (root / "dsa_sample" / "shard_00_of_01" / "games_done.jsonl").write_text("")
    with pytest.raises(SystemExit):
        _sample(0, n=1, resume=True)


def test_a_fresh_sample_run_does_not_append_to_old_lists(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    first = _rows(root)
    calls = []
    orig = A.AgentSampler.sample
    monkeypatch.setattr(A.AgentSampler, "sample", lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    assert _sample(0, n=1) == 0
    assert _rows(root) == first and len(calls) == 3 * len(first)  # sampled afresh, not read back from the old file


def test_replay_mismatch_stops_the_game_there(mock_runs):
    root, eps, _ = mock_runs
    p = root / "rq3_mock" / "shard_00_of_01" / "episodes.jsonl"
    rows = read_jsonl(p)
    rows[0]["trace"][2]["real_obs"] = "something else"
    _rewrite(p, rows)
    assert _sample(0, n=1) == 0
    g = rows[0]["game_id"]
    assert [s["state_idx"] for s in _rows(root) if s["game_id"] == g] == [0, 1, 2]
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
    _rewrite(p, rows)
    assert _label(0, n=1) == 0
    labs = _rows(root, "dsa_label", "labels.jsonl")
    bad = [x for x in labs if not x["rank1_replay_ok"]]
    assert [(x["game_id"], x["state_idx"]) for x in bad] == [(rows[1]["game_id"], rows[1]["state_idx"])]
    assert _score("qwen3-8b") == 0
    scored = {(x["game_id"], x["state_idx"]) for x in read_jsonl(root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl")}
    assert (rows[1]["game_id"], rows[1]["state_idx"]) not in scored
    assert A.run_decisionsim_agent_analysis({"run_id": "dsa_enc", "label_source": "dsa_label",
                                             "models": {"qwen3-8b": "dsa_score_qwen3-8b"}}) == 0
    m = json.loads((root / "dsa_enc" / "metrics.json").read_text())
    assert m["states"]["n_usable"] == len(labs) - 1 == m["encounter"]["n_states"] == m["models"]["qwen3-8b"]["n_states"]


def test_a_history_that_does_not_replay_makes_the_state_unusable(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    p = root / "dsa_sample" / "shard_00_of_01" / "states.jsonl"
    rows = read_jsonl(p)
    rows[2]["history_text"] += "\n(edited)"
    _rewrite(p, rows)
    assert _label(0, n=1) == 0
    labs = _rows(root, "dsa_label", "labels.jsonl")
    assert [(x["game_id"], x["state_idx"]) for x in labs if not x["history_replay_ok"]] == [(rows[2]["game_id"], rows[2]["state_idx"])]
    assert sum(A.usable(x) for x in labs) == len(labs) - 1


def test_every_replay_of_a_state_must_reproduce_its_history(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    real = A.replay
    n = {"k": 0}

    def drifting(client, game, prefix, split="train"):
        h = real(client, game, prefix, split=split)
        n["k"] += 1
        if n["k"] == 2:  # the second replay of the first state comes back different
            h.state.history = list(h.state.history) + [("look", "drift")]
        return h

    monkeypatch.setattr(A, "replay", drifting)
    assert _label(0, n=1) == 0
    labs = _rows(root, "dsa_label", "labels.jsonl")
    assert not labs[0]["history_replay_ok"] and all(x["history_replay_ok"] for x in labs[1:])


def test_label_stops_partial_and_resumes_to_the_same_labels(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    assert _label(0, n=1) == 0
    whole = _rows(root, "dsa_label", "labels.jsonl")
    assert _label(0, n=1, max_replays_per_job=10) == 1
    assert "PARTIAL" in (root / "dsa_label" / "shard_00_of_01" / "STATUS.md").read_text()
    while _label(0, n=1, max_replays_per_job=10, resume=True) == 1:
        pass
    assert _rows(root, "dsa_label", "labels.jsonl") == whole


def test_labels_match_a_fresh_replay_on_the_simulator(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    from wf.data.transitions import context_determined_label
    from wf.env.replay import history_text_from_state, replay, step_counterfactual

    client = AlfworldClient()
    client.create()
    for x in _rows(root, "dsa_label", "labels.jsonl")[:6]:
        for c in x["candidates"]:
            h = replay(client, x["game_id"], x["prefix_actions"], split="train")
            assert history_text_from_state(h.state) == x["history_text"]
            obs, _ = step_counterfactual(h, c["action"])
            assert obs == c["true_obs"] and c["failed"] == (obs == "Nothing happens.")
            assert c["targets_goal_object"] == (x["goal_object"] in c["action"])
            assert c["context_determined"] == context_determined_label(x["history_text"], c["action"], obs)


def test_label_state_marks_determination_and_goal_agnostic_history(mock_runs):
    from wf.data.transitions import context_determined_label
    from wf.env.scene import scan_scene

    client = AlfworldClient()
    client.create()
    st = client.reset("train/1", "train")
    lists = {"primary": {"candidates": [{"rank": 1, "action": "inventory"}, {"rank": 2, "action": "go to drawer 1"}],
                         "n_samples": 0, "n_format_errors": 0, "n_cut": 0},
             "t1": {"candidates": [{"rank": 1, "action": "inventory"}, {"rank": 2, "action": "take apple 1 from countertop 1"}],
                    "n_samples": 0, "n_format_errors": 0, "n_cut": 0}}
    s = {"game_id": "train/1", "split": "train", "state_idx": 0, "task_line": st.task_line, "prefix_actions": [],
         "recorded_obs": None, "history_text": st.first_observation, "lists": lists}
    lab, used = A.label_state(client, scan_scene(client, "train/1", "train"), s)
    assert used == 3 and not lab["rank1_replay_ok"] and lab["history_replay_ok"]  # recorded reply None
    inv, go, take = lab["candidates"]
    assert not inv["failed"] and inv["context_determined"]  # no news in the reply: determined success
    assert not go["failed"] and not go["context_determined"]  # reveals contents: not determined
    assert take["failed"] and take["context_determined"] and take["admissible"] is False
    for c in lab["candidates"]:
        assert c["context_determined"] == context_determined_label(lab["history_text"], c["action"], c["true_obs"])
    assert lab["history_goal_agnostic"] and lab["lists"]["t1"][1]["action"] == "take apple 1 from countertop 1"
    goal = lab["goal_object"]
    touched = {**s, "prefix_actions": ["go to drawer 1", "open drawer 1", f"take {goal} 1 from drawer 1"]}
    h = client.reset("train/1", "train")
    for a in touched["prefix_actions"]:
        h = client.step(h, a)
    from wf.env.replay import history_text_from_state

    touched["history_text"] = history_text_from_state(h)
    lab2, _ = A.label_state(client, scan_scene(client, "train/1", "train"), touched)
    assert not lab2["history_goal_agnostic"]


def test_each_arm_sends_its_own_prompt_once_per_distinct_candidate(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    _always_swap(monkeypatch)
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    sent = []
    orig = logodds.LogoddsScorer.score_one_token_pair

    def spy(self, transition_id, condition, messages, o_plus, o_minus):
        sent.append((transition_id, condition, messages))
        return orig(self, transition_id, condition, messages, o_plus, o_minus)

    monkeypatch.setattr(logodds.LogoddsScorer, "score_one_token_pair", spy)
    assert _score("llama3.1-8b") == 0
    labs = {f"{x['game_id']}#{x['state_idx']:02d}": x for x in _rows(root, "dsa_label", "labels.jsonl")}
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
        elif cond == A.NEUTRAL:
            assert "Your task is to: explore the room." in last
        else:
            assert st["swapped_task_line"] in last and st["swap"]
    multi = [x for x in labs.values() if len(A.scored_candidates(x)) > 1]
    single = [x for x in labs.values() if len(A.scored_candidates(x)) == 1]
    n_multi = sum(len(A.scored_candidates(x)) for x in multi)
    assert sum(c == A.BLIND for _, c, _ in sent) == n_multi + len(single)
    assert sum(c == A.AWARE for _, c, _ in sent) == n_multi == sum(c == A.NEUTRAL for _, c, _ in sent)
    assert sum(c == A.SWAPPED for _, c, _ in sent) == sum(len(A.scored_candidates(x)) for x in multi if x["swap"]) > 0


def test_score_resume_and_refusal_of_other_prompts(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("qwen3-8b") == 0
    p = root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl"
    before = p.read_text()
    assert _score("qwen3-8b", resume=True) == 0 and p.read_text() == before
    rows = read_jsonl(p)
    rows[0]["score_hash"] = "x"
    _rewrite(p, rows)
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


def test_a_failed_gate_scores_nothing_and_enters_holm_with_p_one(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("llama3.1-8b") == 0
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
    assert _analysis() == 0
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    assert m["gate_failed"] == ["mistral-7b-v0.3"]
    per = m["primary"]["overall"]["per_model"]
    assert per["mistral-7b-v0.3"]["p_holm"] == 1.0 and per["llama3.1-8b"]["p_holm"] is not None


def test_arms_must_include_aware_and_blind(mock_runs):
    with pytest.raises(SystemExit):
        _score("qwen3-8b", model_free=False, arms=["yn_aligned", "yn_swapped"])
    with pytest.raises(SystemExit):
        _score("gemma2-9b-it", model_free=False)


def test_analysis_waits_and_refuses_scores_of_another_label_run(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("qwen3-8b") == 0
    cfgp = root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "config.json"
    cfg = json.loads(cfgp.read_text())
    cfgp.write_text(json.dumps({**cfg, "label_source": "other"}))
    assert _analysis() == 0
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    assert "qwen3-8b" in m["waiting"] and "llama3.1-8b" in m["waiting"] and not m["models"]
    assert m["primary"]["overall"]["per_model"] == {}


def test_holm_waits_for_every_priority_model(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _score("llama3.1-8b") == 0
    assert _analysis() == 0
    m = json.loads((root / "dsa_analysis" / "metrics.json").read_text())
    v = m["primary"]["overall"]["per_model"]["llama3.1-8b"]
    assert v["p_holm"] is None and v["verdict"] == "waiting for every priority model"


def test_lists_are_saved_one_state_at_a_time(mock_runs, monkeypatch):
    root, _, _ = mock_runs
    orig = A.AgentSampler.sample
    n = {"k": 0}

    def flaky(self, *a, **k):
        n["k"] += 1
        if n["k"] == 7:  # the first request of the third state
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


def test_dry_run_on_mock_archives(mock_runs):
    root, eps, _ = mock_runs
    client = AlfworldClient()
    client.create()
    rows = []
    for e in eps:
        st = client.reset(e["game_id"], "train")
        rows.append({"game_id": e["game_id"], "history_len": 0, "history_text": st.first_observation})
    (root / "coll").mkdir()
    _rewrite(root / "coll" / "transitions.jsonl", rows)
    assert A.run_decisionsim_agent_dryrun({"run_id": "dsa_dry", "episodes_source": "rq3_mock",
                                           "first_obs_sources": ["coll"]}) == 0
    m = json.loads((root / "dsa_dry" / "metrics.json").read_text())
    assert m["n_states"] == sum(len(e["trace"]) for e in eps) and m["n_history_problems"] == 0
    assert m["plumbing_known_answer"]["is_zero"]
    fails = sum(s["real_obs"] == "Nothing happens." for e in eps for s in e["trace"])
    assert m["rank1"]["candidates"]["failed"] == fails
    assert sum(v["failing_rank1"] for v in m["per_game"].values()) == fails


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


def test_precision_script_floor_and_holm_threshold():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("prec", Path(__file__).resolve().parents[1] / "scripts" / "decisionsim_agent_precision.py")
    prec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prec)
    assert prec.p_min(7) == 2 ** -6 < 0.025 < prec.p_min(6) == 2 ** -5
    assert prec.min_games_for(0.025) == 7 == A.MIN_GAMES_FOR_SIGNIFICANCE


def test_harm_kinds_count_harm_states_only():
    states, L = _four_states()
    both = _state([_cand(1, True), _cand(2, True)], game="g5", t=0)  # both choices fail at a go-to: not harm
    L2 = {a: {**v, **_L(both, [1, 1])} for a, v in L.items()}
    c = A.contrast(states + [both], L2, "aware", "blind")
    assert c["harm_by_kind_of_choice"] == {"regime": 1} and c["harm_regime_share"] == 1.0


def test_seed_check_flags_a_seed_that_changes_nothing():
    class Stuck:
        def sample(self, messages, n, params, max_tokens, seed):
            return [{"text": f"Action: a{i % 3}", "finish_reason": "stop"} for i in range(n)]

    rec = A.sample_state(Stuck(), [{"role": "user", "content": "x"}], "a0", "g", 0, {"seed_check": True})
    assert rec["seed_check"] == {"distinct_first": 3, "same_set": True, "same_seed_reproduces": True}
    assert set(rec["lists"]) == {A.PRIMARY_LIST, A.T1_LIST} and rec["greedy_fresh"]["agrees"] is True


# ---------------- review 2 ----------------


def test_each_block_reads_its_own_estimand_p_decomposition_and_states():
    # Two non-encounter harm states in g1-g2 and two encounter help states in g3-g4, and one encounter state where a
    # success is ranked above the regime candidate (an encounter, but not reachable).
    harm = [_lab([_cand(1, False), _cand(2, True)], game=f"g{i}", t=0) for i in (1, 2)]
    hlp = [_lab([_cand(1, True, goal=True), _cand(2, False)], game=f"g{i}", t=0) for i in (3, 4)]
    unreach = [_lab([_cand(1, False), _cand(2, True, goal=True)], game="g5", t=0)]
    labs = harm + hlp + unreach
    L = {A.AWARE: {}, A.BLIND: {}, A.SWAPPED: {}}
    for s in harm:  # aware rejects the working rank 1 and lands on the failure; blind keeps rank 1
        L[A.AWARE].update(_L(s, [-1, 1])); L[A.BLIND].update(_L(s, [1, 1])); L[A.SWAPPED].update(_L(s, [1, 1]))
    for s in hlp:  # aware rejects the failing rank 1; blind accepts it; swapped chooses as aware
        L[A.AWARE].update(_L(s, [-1, 1])); L[A.BLIND].update(_L(s, [1, 1])); L[A.SWAPPED].update(_L(s, [-1, 1]))
    for s in unreach:  # aware rejects the working rank 1 and takes the regime failure: harm at an encounter state
        L[A.AWARE].update(_L(s, [-1, 1])); L[A.BLIND].update(_L(s, [1, 1])); L[A.SWAPPED].update(_L(s, [-1, 1]))
    rep = A.model_report(labs, L, L)
    assert rep["primary"]["n_states"] == 5 and rep["conditional"]["n_states"] == 3  # g5 counts though unreachable
    assert rep["primary"]["difference"]["estimate"] == pytest.approx(1 / 5)
    assert rep["conditional"]["difference"]["estimate"] == pytest.approx(-1 / 3)
    assert rep["conditional_decomposition"]["n_states"] == 3 and rep["decomposition"]["n_states"] == 5
    assert rep["decomposition"]["aware_minus_swapped"]["estimate"] == pytest.approx(2 / 5)
    assert rep["conditional_decomposition"]["aware_minus_swapped"]["estimate"] == pytest.approx(0.0)
    # In the Holm blocks each family reads its own p and decomposition.
    m = {"competence": {"interpretable": True}, **{k: rep[k] for k in ("primary", "conditional", "decomposition",
                                                                      "conditional_decomposition")}}
    m["primary"] = {**m["primary"], "difference": {**m["primary"]["difference"], "p": 0.001, "ci": [0.01, 0.3]}}
    m["conditional"] = {**m["conditional"], "difference": {**m["conditional"]["difference"], "p": 0.002, "ci": [0.01, 0.3]},
                        "harm_regime_share": 1.0, "changed": {"harm": 1}}
    models = {t: m for t in A.PRIORITY}
    blocks = A.primary_blocks(models, list(A.PRIORITY), set())
    over, cond = blocks["overall"]["per_model"], blocks["conditional_on_encounter"]["per_model"]
    assert over["llama3.1-8b"]["p_holm"] == pytest.approx(0.002) and cond["llama3.1-8b"]["p_holm"] == pytest.approx(0.004)
    assert cond["llama3.1-8b"]["n_states"] == 3 and over["llama3.1-8b"]["n_states"] == 5
    assert over["llama3.1-8b"]["verdict"].startswith("toward failure, attributable")  # 2/5 against 1/5
    assert cond["llama3.1-8b"]["verdict"] == "toward failure, not attributable to the real task"  # aware-swapped 0


def test_verdict_boundaries():
    assert A.verdict({"ci": [0.0, 0.02]}, 0.001, None) == "no reliable difference"  # the bound must be above zero
    models = {"llama3.1-8b": _model(0.01, [0.001, 0.02], regime_share=0.5),
              "mistral-7b-v0.3": _model(0.02, [0.001, 0.02], regime_share=0.49)}
    out = A.primary_block(models, list(A.PRIORITY), set(), "primary", "decomposition")["per_model"]
    assert out["llama3.1-8b"]["harm_mostly_regime"] is True
    assert out["llama3.1-8b"]["verdict"] == "toward failure, attributable to the real task"  # exactly half: no suffix
    assert out["mistral-7b-v0.3"]["verdict"].endswith("(not evidence for the main finding: harm mostly from non-regime actions)")


def test_t1_list_readouts_read_the_t1_list():
    a, b, c = _cand(1, False), _cand(2, True, goal=True), _cand(3, False, action="z")
    labs = [_lab([a, b], t1=[{**a}, {**c, "rank": 2}], game=f"g{i}", t=0) for i in range(3)]
    L = {arm: {k: 1.0 for s in labs for k in _L(s, [1, 1, 1]).items() for k in [k[0]]} for arm in (A.AWARE, A.BLIND)}
    rep = A.model_report(labs, L, L)
    assert rep["conditional"]["n_states"] == 3  # the primary lists hold the regime candidate
    assert rep["sensitivity"]["t1_list"]["n_states"] == 3
    assert rep["sensitivity"]["t1_list_conditional"]["n_states"] == 0  # the t1 lists hold none


def test_t1_encounter_report_in_the_analysis_reads_the_t1_lists(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _model_free() == 0
    m = json.loads((root / "dsa_mf" / "metrics.json").read_text())
    labs = [x for x in _rows(root, "dsa_label", "labels.jsonl") if A.usable(x)]
    n_t1 = sum(len(A.scored_candidates(v)) for v in A.views(labs, A.T1_LIST))
    n_pr = sum(len(A.scored_candidates(v)) for v in A.views(labs, A.PRIMARY_LIST))
    assert n_t1 != n_pr and m["encounter_t1_list"]["n_candidates"] == n_t1 and m["encounter"]["n_candidates"] == n_pr


def test_shards_with_different_settings_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for i, seed in enumerate((1, 2)):
        d = tmp_path / "x" / f"shard_{i:02d}_of_02"
        d.mkdir(parents=True)
        (d / "STATUS.md").write_text("# STATUS — x — REPORTED (shard complete)\n")
        (d / "git_sha").write_text("abc\n")
        (d / "config.json").write_text(json.dumps({"run_id": f"x/shard_{i:02d}_of_02", "shard": i, "seed": seed}))
    _, problems = A.complete_shards("x")
    assert any("different settings" in p for p in problems)
    (tmp_path / "x" / "shard_01_of_02" / "config.json").write_text(json.dumps({"run_id": "y", "shard": 1, "seed": 1}))
    assert A.complete_shards("x")[1] == []


def test_the_swap_object_excludes_every_candidate_of_both_lists(mock_runs, monkeypatch):
    from wf.env.scene import scan_scene

    seen = {}
    monkeypatch.setattr(A, "state_swap_object", lambda scene, task, goal, actions, *a, **k: seen.update(actions=actions))
    client = AlfworldClient()
    client.create()
    st = client.reset("train/1", "train")
    lists = {"primary": {"candidates": [{"rank": 1, "action": "inventory"}], "n_samples": 0, "n_format_errors": 0, "n_cut": 0},
             "t1": {"candidates": [{"rank": 1, "action": "inventory"}, {"rank": 2, "action": "take apple 1 from countertop 1"}],
                    "n_samples": 0, "n_format_errors": 0, "n_cut": 0}}
    s = {"game_id": "train/1", "split": "train", "state_idx": 0, "task_line": st.task_line, "prefix_actions": [],
         "recorded_obs": None, "history_text": st.first_observation, "lists": lists}
    A.label_state(client, scan_scene(client, "train/1", "train"), s)
    assert seen["actions"] == ["inventory", "take apple 1 from countertop 1"]


def test_sample_seed_value_is_pinned():
    assert A.sample_seed("valid_unseen/1", 3) == 3067989451
    assert A.sample_seed("valid_unseen/1", 3, name="t1") == 912758500


def test_label_budget_counts_the_next_state(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    states = _rows(root)
    sizes = [len(A.union_actions(s)) for s in states]
    budget = sizes[0] + sizes[1] - 1  # the second state would cross it
    assert _label(0, n=1, max_replays_per_job=budget) == 1
    assert len(read_jsonl(root / "dsa_label" / "shard_00_of_01" / "labels.jsonl")) == 1


class _Recorder:
    """A stand-in sampler that records every request and can permute its replies by seed."""

    def __init__(self, permute=False, text="Action: a{i}"):
        self.calls, self.permute, self.text = [], permute, text

    def sample(self, messages, n, params, max_tokens, seed):
        self.calls.append((params, seed))
        out = [{"text": self.text.format(i=i % 4), "finish_reason": "stop"} for i in range(n)]
        return out[::-1] if self.permute and seed % 2 else out


def test_seed_check_uses_the_primary_settings_and_compares_sequences():
    r = _Recorder()
    rec = A.sample_state(r, [{"role": "user", "content": "x"}], "a0", "g", 0, {"seed_check": True})
    assert [p for p, _ in r.calls] == [A.SAMPLINGS["primary"], A.SAMPLINGS["t1"], A.GREEDY,
                                       A.SAMPLINGS["primary"], A.SAMPLINGS["primary"]]
    assert r.calls[4][1] == r.calls[0][1] and r.calls[3][1] != r.calls[0][1]  # the repeat reuses the primary seed
    seeds = {}
    for x in range(40):  # find a state whose seed-check seed and primary seed differ in parity
        a, b = A.sample_seed("g", x, name="primary"), A.sample_seed("g", x, name="seed_check")
        if a % 2 != b % 2:
            seeds = {"t": x}
            break
    rec = A.sample_state(_Recorder(permute=True), [{"role": "user", "content": "x"}], "a0", "g", seeds["t"], {"seed_check": True})
    assert rec["seed_check"]["same_set"] is False  # the same replies in another order are another sequence


def test_smoke_gate_needs_a_non_degenerate_seed_check_state_and_reports_readouts():
    lst = {"n_samples": 4, "n_format_errors": 0, "n_cut": 0, "candidates": [{"count": 1}],
           "replies": [{"text": "Action: look", "finish_reason": "length"}] * 4}
    degenerate = [{"lists": {"primary": lst}, "greedy_fresh": {"agrees": True},
                   "seed_check": {"distinct_first": 1, "same_set": True, "same_seed_reproduces": True}}]
    b = A.sampling_bookkeeping(degenerate)
    gate = A.smoke_gate({**b, "replay_mismatches": 0})
    assert not gate["passed"]  # the seed check never saw a state that could show anything
    assert gate["truncated_replies"]["primary"] == {"n": 4, "share": 1.0}
    assert gate["greedy_agreement"] == 1.0 and gate["drifted_sampler"] is False


def test_fresh_greedy_agreement_compares_normalised_actions_and_the_drift_boundary():
    rec = A.sample_state(_Recorder(text="Action: A{i}."), [{"role": "user", "content": "x"}], "a0", "g", 0, {})
    assert rec["greedy_fresh"]["agrees"] is True  # "A0." and "a0" are one action
    lst = {"n_samples": 1, "n_format_errors": 0, "n_cut": 0, "candidates": [{"count": 1}], "replies": []}
    sts = [{"lists": {"primary": lst}, "greedy_fresh": {"agrees": i < 9}} for i in range(10)]
    b = A.sampling_bookkeeping(sts)
    assert b["greedy_agreement"] == pytest.approx(0.9) and b["drifted_sampler"] is False  # exactly 90% is not drift


def test_request_settings_and_normalisation_counts_are_recorded(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0
    m = json.loads((root / "dsa_sample" / "shard_00_of_01" / "metrics.json").read_text())
    for name, p in A.SAMPLINGS.items():
        assert m["request_settings"][name] == {**p, **A.FIXED_PARAMS, "n": 20, "max_tokens": 256}
    assert m["primary"]["normalised_actions"] > 0  # the mock writes "Go to Drawer 1."
    r = A.rank_candidates("z", ["Action: Go to X.", "Action: look", "Action: look "], "g", 0)
    assert r["n_normalised"] == 1


def test_scoring_waits_for_the_model_free_analysis_unless_overridden(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0
    assert _score("qwen3-8b", model_free=False) == 1  # no model-free analysis yet
    assert "waits for the model-free analysis" in (root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "STATUS.md").read_text()
    assert A.run_decisionsim_agent_analysis({"run_id": "dsa_mf", "label_source": "other", "models": {}}) == 1
    assert _score("qwen3-8b", model_free=False) == 1  # another label run, and unfinished
    assert _score("qwen3-8b", model_free=False, run_order_override="smoke of the scorer") == 0
    order = json.loads((root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "run_order.json").read_text())
    assert order["override"] == "smoke of the scorer" and order["would_have_refused"]
    assert _model_free() == 0 and _score("qwen3-8b", model_free=False) == 0
    order = json.loads((root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "run_order.json").read_text())
    assert "override" not in order and "conditional_cannot_reach_significance" in order["futility"]
    assert A.run_decisionsim_agent_analysis({"run_id": "dsa_mf", "label_source": "dsa_label",
                                             "models": {"qwen3-8b": "dsa_score_qwen3-8b"}}) == 0
    assert _score("qwen3-8b", model_free=False) == 1  # an analysis with models is not the model-free one


def test_views_count_only_runnable_candidates():
    lab = _lab([_cand(1, False), _cand(2, True, err="server error"), _cand(3, True, goal=True)])
    v = A.view(lab)
    assert [c["rank"] for c in v["candidates"]] == [1, 3] and v["n_union"] == 2


def _fake_analysis(root, name, label_source="dsa_label", models=None, status="REPORTED (Stage 1b report)"):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"encounter": {"futility": {"games": 9}}}))
    (d / "config.json").write_text(json.dumps({"run_id": name, "label_source": label_source, "models": models or {}}))
    (d / "STATUS.md").write_text(f"# STATUS — {name} — {status}\n")


def test_run_order_check_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    cfg = {"label_source": "dsa_label", "model_free_analysis": "mf"}
    assert not A.run_order_check(cfg)["ok"]  # nothing written yet
    _fake_analysis(tmp_path, "mf", label_source="other")
    r = A.run_order_check(cfg)
    assert not r["ok"] and "another label run" in r["why"]
    _fake_analysis(tmp_path, "mf", models={"qwen3-8b": "x"})
    assert "lists models" in A.run_order_check(cfg)["why"]
    _fake_analysis(tmp_path, "mf", status="FAIL")
    assert "did not finish" in A.run_order_check(cfg)["why"]
    _fake_analysis(tmp_path, "mf")
    r = A.run_order_check(cfg)
    assert r["ok"] and r["futility"] == {"games": 9} and "override" not in r
    _fake_analysis(tmp_path, "mf", label_source="other")
    r = A.run_order_check({**cfg, "run_order_override": "why not"})
    assert r["ok"] and r["override"] == "why not" and "another label run" in r["would_have_refused"]


# ---------------- review 3 ----------------


def _path_states():
    """One changed state per path (aware against blind), at step 0 or later."""
    ab = _state([_cand(1, False), _cand(2, True)], game="g1", t=0)  # aware abstains (works), blind takes the failure
    bb = _state([_cand(1, False), _cand(2, True)], game="g2", t=4)  # blind abstains, aware takes the failure
    acc = _state([_cand(1, True), _cand(2, True, goal=True), _cand(3, False)], game="g3", t=3)  # aware takes rank 2
    rej = _state([_cand(1, False), _cand(2, True)], game="g4", t=0)  # aware rejects the working rank 1 blind takes
    same = _state([_cand(1, False), _cand(2, True)], game="g5", t=7)  # both take rank 1
    states = [ab, bb, acc, rej, same]
    L = _scores(states, {A.AWARE: [[-1, -1], [-1, 1], [-1, 1, 1], [-1, 1], [1, 1]],
                         A.BLIND: [[-1, 1], [-1, -1], [-1, -1, 1], [1, 1], [1, 1]],
                         A.SWAPPED: [[-1, -1], [-1, -1], [-1, -1, 1], [1, 1], [1, 1]]})
    return states, L


def test_change_paths_by_path_outcome_and_step_band():
    states, L = _path_states()
    paths = A.change_paths(states, L, "aware", "blind")
    zero = {"step_0": 0, "later": 0}
    assert paths["a_abstained"]["help"] == {"step_0": 1, "later": 0}
    assert paths["b_abstained"]["harm"] == {"step_0": 0, "later": 1}
    assert paths["a_accepted_what_b_rejected"]["harm"] == {"step_0": 0, "later": 1}
    assert paths["a_rejected_what_b_chose"]["harm"] == {"step_0": 1, "later": 0}
    total = sum(v for p in paths.values() for cell in p.values() for v in cell.values())
    assert total == 4  # the unchanged state is in no path
    assert paths["a_abstained"]["harm"] == zero and paths["a_rejected_what_b_chose"]["help"] == zero
    c = A.contrast(states, L, "aware", "blind")
    assert c["paths"] == paths and c["changed"]["harm"] == 3 and c["changed"]["help"] == 1
    assert A.change_path("aware", "blind", states[4], L, A.choose("aware", states[4], L), A.choose("blind", states[4], L)) is None


def test_decomposition_parts_get_their_own_paths():
    states, L = _path_states()
    dec = A.decomposition(states, L)
    assert set(dec["paths"]) == {"aware_vs_blind", "aware_vs_swapped", "swapped_vs_blind"}
    assert dec["paths"]["aware_vs_blind"] == A.change_paths(states, L, "aware", "blind")
    # swapped abstains at the first two states as aware does at the first: aware vs swapped differs only where they do
    sw = dec["paths"]["aware_vs_swapped"]
    assert sw["b_abstained"]["harm"] == {"step_0": 0, "later": 1}  # g2: swapped abstains, aware takes the failure
    assert sw["a_rejected_what_b_chose"]["harm"] == {"step_0": 1, "later": 0}  # g4
    sb = dec["paths"]["swapped_vs_blind"]
    assert sb["a_abstained"]["help"] == {"step_0": 1, "later": 0}  # g1: swapped abstains, blind takes the failure
    assert sum(v for p in sb.values() for cell in p.values() for v in cell.values()) == 1  # only g1 differs


def test_hybrid_verifiers():
    s = _state([_cand(1, False), _cand(2, True)], game="g1", t=0)
    L = _scores([s], {A.AWARE: [[-1, 1]], A.BLIND: [[1, 1]]})
    assert A.arm_for("hybrid_aware_rank1", True) == A.AWARE and A.arm_for("hybrid_aware_rank1", False) == A.BLIND
    assert A.arm_for("hybrid_blind_rank1", True) == A.BLIND and A.arm_for("hybrid_blind_rank1", False) == A.AWARE
    assert A.choose("hybrid_aware_rank1", s, L)["rank"] == 2  # aware rejects rank 1, blind accepts rank 2
    assert A.choose("hybrid_blind_rank1", s, L)["rank"] == 1  # blind accepts rank 1
    L2 = _scores([s], {A.AWARE: [[1, -1]], A.BLIND: [[-1, 1]]})
    assert A.fell_back("hybrid_blind_rank1", s, L2) and not A.fell_back("hybrid_aware_rank1", s, L2)
    labs = [_lab(s["candidates"], game="g1", t=0)]
    rep = A.model_report(labs, L, L)
    assert rep["hybrid"]["hybrid_aware_rank1"]["minus_blind"]["estimate"] == 1.0
    assert rep["hybrid"]["hybrid_blind_rank1"]["minus_blind"]["estimate"] == 0.0


def test_acceptance_by_position_and_step_band_and_every_decomposition_shift():
    s0 = _state([_cand(1, False), _cand(2, True)], game="g1", t=0)
    s1 = _state([_cand(1, False), _cand(2, True)], game="g2", t=5)
    L = _scores([s0, s1], {A.AWARE: [[-1, 1], [1, -1]], A.BLIND: [[1, -1], [1, -1]], A.SWAPPED: [[1, -1], [1, 1]]})
    acc = A.acceptance([s0, s1], L, L)
    g = acc[A.AWARE]["by_group"]
    assert g["rank1/success/step_0"] == {"n": 1, "yes_rate": 0.0} and g["rank1/success/later"]["yes_rate"] == 1.0
    assert g["alternative/failure/step_0"]["yes_rate"] == 1.0 and g["alternative/failure/later"]["yes_rate"] == 0.0
    assert acc["headline_yes_rate_working_rank1_step_0"] == {A.AWARE: 0.0, A.BLIND: 1.0, A.SWAPPED: 1.0}
    assert acc["aware_minus_swapped"]["alternative/failure/step_0"]["yes_rate_shift"] == 1.0
    assert acc["aware_minus_swapped"]["rank1/success/step_0"]["yes_rate_shift"] == -1.0
    assert acc["swapped_minus_blind"]["rank1/success/step_0"]["yes_rate_shift"] == 0.0
    assert acc["swapped_minus_blind"]["alternative/failure/later"]["yes_rate_shift"] == 1.0
    assert acc["aware_minus_blind"]["rank1/success/step_0"]["yes_rate_shift"] == -1.0


def test_inert_split_by_step_band():
    s0 = _state([_cand(1, False), _cand(2, True)], game="g1", t=0)
    s1 = _state([_cand(1, False), _cand(2, True)], game="g2", t=1)  # step 1 is a later step
    s2 = _state([_cand(1, False), _cand(2, True)], game="g3", t=6)
    L = _scores([s0, s1, s2], {A.AWARE: [[-1, -1], [-1, 1], [1, 1]], A.BLIND: [[1, 1]] * 3})
    out = A.inert([s0, s1, s2], L)["aware"]
    assert out["step_0"] == {"n_states": 1, "fallback_rate": 1.0, "rank1_share": 1.0}
    assert out["later"] == {"n_states": 2, "fallback_rate": 0.0, "rank1_share": 0.5}
    assert out["fallback_rate"] == pytest.approx(1 / 3)


def test_a_not_attributable_verdict_from_non_regime_harm_gets_the_suffix_too():
    m = _model(0.01, [0.001, 0.02], regime_share=0.2)
    m["decomposition"] = {"n_states": 50, "aware_minus_blind": {"estimate": 0.02}, "aware_minus_swapped": {"estimate": 0.001}}
    models = {"llama3.1-8b": m, "mistral-7b-v0.3": _model(0.5, [-0.01, 0.02])}
    v = A.primary_block(models, list(A.PRIORITY), set(), "primary", "decomposition")["per_model"]["llama3.1-8b"]["verdict"]
    assert v == ("toward failure, not attributable to the real task (not evidence for the main finding: harm mostly "
                 "from non-regime actions)")


def test_run_order_check_compares_the_label_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    lab = tmp_path / "lab.jsonl"
    lab.write_text("one\n")
    _fake_analysis(tmp_path, "mf")
    from wf.util import file_sha

    (tmp_path / "mf" / "inputs.json").write_text(json.dumps({"inputs": {"/elsewhere/labels.jsonl": file_sha(lab)}}))
    cfg = {"label_source": "dsa_label", "model_free_analysis": "mf"}
    assert A.run_order_check(cfg, [lab])["ok"]  # same contents, wherever they were read from
    lab.write_text("two\n")
    r = A.run_order_check(cfg, [lab])
    assert not r["ok"] and "labels changed" in r["why"]
    assert A.run_order_check(cfg)["ok"]  # without label files only the run is checked


def test_scoring_refuses_labels_changed_after_the_model_free_analysis(mock_runs):
    root, _, _ = mock_runs
    assert _sample(0, n=1) == 0 and _label(0, n=1) == 0 and _model_free() == 0
    p = root / "dsa_label" / "shard_00_of_01" / "labels.jsonl"
    rows = read_jsonl(p)
    rows[0]["note"] = "edited after the analysis"
    _rewrite(p, rows)
    assert _score("qwen3-8b", model_free=False) == 1
    assert "labels changed" in (root / "dsa_score_qwen3-8b" / "shard_00_of_01" / "STATUS.md").read_text()
