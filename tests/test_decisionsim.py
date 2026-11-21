"""Option A, the offline decision simulation (wf/decisionsim.py, notes/decisionsim_design.md). Synthetic data."""
from __future__ import annotations

import json
import math

import pytest

from wf import decisionsim as D
from wf import families, sizes
from tests.test_families import FakeTok, _scorer
from tests.test_rq1 import _item, mock_root, mock_score  # noqa: F401

NOTHING = "Nothing happens."
TASK = "Your task is to: put some saltshaker on cabinet."


def _t(tid, action, ok, *, game="g0", step=0, det=True, goal=False, prefix=None, swap="mug", task=TASK):
    return {"id": tid, "game_id": game, "state_idx": step, "action": action,
            "true_obs": "You arrive." if ok else NOTHING, "context_determined": det if not ok else False,
            "targets_goal_object": goal, "task_line": task, "history_text": f"history {game} {step}",
            "goal_object": "saltshaker", "swapped_goal_object": swap,
            "swapped_task_line": task.replace("saltshaker", swap) if swap else None,
            "prefix_actions": list(prefix if prefix is not None else ["x"] * step)}


def _state(cands, proposal, game="g0", step=0, swap="mug"):
    return {"game_id": game, "state_idx": step, "proposal": proposal,
            "candidates": sorted(cands, key=lambda t: t["action"]), "swap": swap}


# ---------------- states ----------------


def _episode(game="g0"):
    """Three saved steps: the agent did "go to shelf 1" then "open drawer 1"; the last step's action is unknown."""
    s0 = [_t(f"{game}a", "go to shelf 1", True, game=game, step=0, prefix=[]),
          _t(f"{game}b", "take saltshaker 1 from cabinet 1", False, game=game, step=0, prefix=[], goal=True)]
    s1 = [_t(f"{game}c", "open drawer 1", True, game=game, step=1, prefix=["go to shelf 1"]),
          _t(f"{game}d", "take saltshaker 1 from drawer 1", False, game=game, step=1, det=False, prefix=["go to shelf 1"])]
    s2 = [_t(f"{game}e", "go to cabinet 1", True, game=game, step=2, prefix=["go to shelf 1", "open drawer 1"]),
          _t(f"{game}f", "take saltshaker 1 from cabinet 1", False, game=game, step=2, prefix=["go to shelf 1", "open drawer 1"])]
    return s0 + s1 + s2


def test_agent_actions_come_from_longer_prefixes():
    acts = D.agent_actions(_episode())
    assert acts == {("g0", 0): "go to shelf 1", ("g0", 1): "open drawer 1"}
    bad = _episode() + [_t("z", "look", True, step=2, prefix=["go to drawer 1", "open drawer 1"])]
    with pytest.raises(ValueError):
        D.agent_actions(bad)


def test_eligible_states_need_the_proposal_a_success_and_a_determined_failure():
    states = D.build_states(_episode())
    # step 0 is eligible; step 1's only failure is not determined; step 2's agent action is unknown
    assert [(s["game_id"], s["state_idx"]) for s in states] == [("g0", 0)]
    assert states[0]["proposal"] == "go to shelf 1"
    no_success = [t for t in _episode() if t["id"] != "g0a"] + [_t("g0x", "look", False, step=0, prefix=[])]
    assert D.build_states(no_success) == []  # the agent's action is no longer among step 0's candidates
    no_ok = [{**t, "true_obs": NOTHING} if t["id"] == "g0a" else t for t in _episode()]
    assert D.build_states(no_ok) == []


def test_candidates_must_share_the_task_and_history():
    ep = _episode()
    ep[1] = {**ep[1], "task_line": "Your task is to: put some mug on cabinet."}
    with pytest.raises(ValueError):
        D.build_states(ep)


def test_duplicate_candidate_actions_are_refused():
    ep = _episode() + [_t("g0dup", "go to shelf 1", True, step=0, prefix=[])]
    with pytest.raises(ValueError, match="duplicate"):
        D.build_states(ep)


def test_sample_is_seeded_sorted_and_all_means_all():
    states = [_state([_t(f"{g}{i}", "look", True, game=f"g{g}", step=i)], "look", game=f"g{g}", step=i)
              for g in range(5) for i in range(4)]
    a = D.select_states(states, 7, seed=1)
    b = D.select_states(list(reversed(states)), 7, seed=1)
    assert [(s["game_id"], s["state_idx"]) for s in a] == sorted((s["game_id"], s["state_idx"]) for s in b)
    assert len(a) == 7 and a == sorted(a, key=lambda s: (s["game_id"], s["state_idx"]))
    assert D.select_states(states, 7, seed=2) != a
    assert D.select_states(states, "all") == states and D.select_states(states, 99) == states
    assert D.sample_digest(a) != D.sample_digest(D.select_states(states, 7, seed=2))


def test_one_swap_object_per_state_never_named_by_a_candidate():
    cands = [_t("a", "take mug 1 from cabinet 1", False, swap="mug"), _t("b", "go to shelf 1", True, swap="plate"),
             _t("c", "look", True, swap="cup")]
    for seed in range(20):
        assert D.state_swap(cands, "g0", 0, seed) in {"plate", "cup"}
    assert D.state_swap(cands[:1], "g0", 0) is None
    s = _state(cands, "look", swap="plate")
    for t in cands:  # every candidate gets the same swapped line, whatever its own draw was
        assert D.prompt_item(t, s, "yn_swapped")["swapped_task_line"] == "Your task is to: put some plate on cabinet."
        assert D.prompt_item(t, s, "yn_aligned") is t
    assert D.prompt_item(cands[0], {**s, "swap": None}, "yn_swapped") is None
    # the draw is seeded by the state, not only the game: different steps of one game get different swaps
    assert len({D.state_swap(cands, "g0", i) for i in range(30)}) == 2


def test_jobs_keep_a_states_arms_together_and_skip_undefined_swaps():
    s1 = _state([_t("a", "look", True), _t("b", "go to x 1", False)], "look", swap=None)
    s2 = _state([_t("c", "look", True, step=1)], "look", step=1)
    jobs = D.score_jobs([s1, s2], ("yn_aligned", "yn_absent", "yn_swapped"))
    assert [(t["id"], a) for _, t, a in jobs] == [("b", "yn_aligned"), ("b", "yn_absent"), ("a", "yn_aligned"),
                                                   ("a", "yn_absent"), ("c", "yn_aligned"), ("c", "yn_absent"),
                                                   ("c", "yn_swapped")]
    # a score's hash changes with the arm and with the swapped line the state gave it
    t = s2["candidates"][0]
    assert D.score_hash(t, "yn_aligned", s2) != D.score_hash(t, "yn_absent", s2)
    assert D.score_hash(t, "yn_swapped", s2) != D.score_hash(t, "yn_swapped", {**s2, "swap": "cup"})


def test_prompts_are_the_yes_no_prompts_of_the_earlier_runs(mock_score):
    t = _item()
    assert D.messages_for("qwen3-8b")(t, "yn_absent") == sizes.arm_messages(t, "yn_absent")
    gemma = D.messages_for("gemma2-9b-it")(t, "yn_aligned")
    assert gemma == families.family_messages(families.MODELS["gemma2-9b-it"], t, "yn_aligned")
    assert gemma[0]["role"] == "user"  # Gemma's system text is folded into the first turn, as in wave A
    assert D.messages_for("llama3.1-8b")(t, "yn_aligned")[-1]["content"].endswith("Answer:")


# ---------------- rules ----------------


def _three(step=0, game="g0", suffix=""):
    """The agent proposed "go to shelf 1" (works); two generated: a failing goal action and a working one."""
    cands = [_t(f"p{suffix}", "go to shelf 1", True, game=game, step=step),
             _t(f"f{suffix}", "take saltshaker 1 from cabinet 1", False, goal=True, game=game, step=step),
             _t(f"w{suffix}", "examine drawer 1", True, game=game, step=step)]
    return _state(cands, "go to shelf 1", game=game, step=step)


def test_rule1_takes_the_largest_score_and_breaks_ties_by_a_seeded_state_hash():
    s = _three()
    assert D.rule_choice("rule1_with_proposal", s, {"p": 1.0, "f": 2.0, "w": 0.5})["id"] == "f"
    assert D.rule_choice("rule1_generated_only", s, {"p": 9.0, "f": 2.0, "w": 0.5})["id"] == "f"
    # a three-way tie goes to the smallest seeded hash of (seed, game, step, action), whatever the scores' level
    want = min(s["candidates"], key=D.tie_key)["id"]
    assert D.rule_choice("rule1_with_proposal", s, {"p": 1.0, "f": 1.0, "w": 1.0})["id"] == want
    assert D.rule_choice("rule1_with_proposal", s, {"p": -3.0, "f": -3.0, "w": -3.0})["id"] == want
    assert D.tie_key(s["candidates"][0]) == D.tie_key(dict(s["candidates"][0]))  # no hidden state
    # the hash depends on the state: the same three actions are not always ordered the same way
    winners = {min(_three(step=2 * k)["candidates"], key=D.tie_key)["action"] for k in range(30)}
    assert len(winners) > 1
    assert D.tie_key(s["candidates"][0], seed=1) != D.tie_key(s["candidates"][0])


def test_rule2_accepts_a_proposal_only_above_zero_and_falls_back_to_the_others():
    s = _three()
    assert D.rule_choice("rule2_verification", s, {"p": 0.1, "f": 5.0, "w": 0.0})["id"] == "p"
    assert D.rule_choice("rule2_verification", s, {"p": 0.0, "f": -5.0, "w": -6.0})["id"] == "f"  # 0 is not Yes
    assert D.rule_choice("rule2_verification", s, {"p": -1.0, "f": -5.0, "w": 3.0})["id"] == "w"
    # the fallback never returns the rejected proposal, even when it scores highest
    assert D.rule_choice("rule2_verification", s, {"p": -0.5, "f": -5.0, "w": -6.0})["id"] == "f"
    assert [t["id"] for t in D.rule_menu("rule2_verification", s, {"p": 1, "f": 0, "w": 0})] == ["p"]
    with pytest.raises(ValueError):
        D.rule_set("rule9", s)


def test_rule3_reads_goal_relevant_candidates_where_one_of_them_works():
    s = _three()
    assert not D.rule_applies("rule3_goal_relevant", s)
    s2 = _state(s["candidates"] + [_t("g", "take saltshaker 1 from shelf 1", True, goal=True)], "go to shelf 1")
    assert D.rule_applies("rule3_goal_relevant", s2)
    assert D.rule_choice("rule3_goal_relevant", s2, {"p": 9, "f": 1, "w": 9, "g": 2})["id"] == "g"


# ---------------- scores: rounding ----------------


def test_scores_are_rounded_to_a_64th_of_a_nat_and_the_raw_ones_kept():
    assert D.quantize(0.004) == 0.0 and D.quantize(0.02) == 1 / 64 and D.quantize(-1.3) == round(-1.3 * 64) / 64
    recs = [{"transition_id": "a", "condition": "yn_aligned", "L": 0.004},
            {"transition_id": "b", "condition": "yn_aligned", "L": 1.001},
            {"transition_id": "a", "condition": "yn_absent", "L": math.nan}]
    q, raw = D.scores_by_arm(recs)
    assert q == {"yn_aligned": {"a": 0.0, "b": 1.0}} and raw == {"yn_aligned": {"a": 0.004, "b": 1.001}}


def test_decisions_ties_and_the_threshold_read_rounded_scores():
    """A proposal at 0.004 nats is a No once rounded; 1.001 and 1.0 are a tie; the mean L stays raw."""
    s = _three()
    recs = [{"transition_id": k, "condition": a, "L": v} for a, Ls in (
        ("yn_aligned", {"p": 0.004, "f": 1.001, "w": 1.0}), ("yn_absent", {"p": 0.02, "f": -1.0, "w": -1.0}))
        for k, v in Ls.items()]
    rep, problems = D.analyse_model([s], recs)
    assert not problems
    c = rep["yn_aligned_vs_yn_absent"]
    assert c["rule2_errors"]["false_rejection"]["rate_a"]["estimate"] == 1.0  # rounded to 0: rejected
    assert c["rule2_errors"]["false_rejection"]["rate_b"]["estimate"] == 0.0  # 0.02 rounds to 1/64: accepted
    assert c["rule1_generated_only"]["ties"]["a"] == 1 and c["rule1_generated_only"]["ties"]["mixed_outcome_a"] == 1
    assert rep["baseline"]["yn_aligned"]["mean_L_successes"] == pytest.approx((0.004 + 1.0) / 2)
    # missing arms are refused, not read
    assert D.analyse_model([s], [r for r in recs if r["condition"] == "yn_aligned"])[0] is None


# ---------------- measures ----------------


def _contrast_data(swap="mug"):
    """Six even-step states in three games. Rule 1 choices, aligned / neutral:
    g0: f / w (harm), f / f (same), g1: w / f (help), f / w (harm), g2: w / w, p / w (changed, both succeed)."""
    L_a, L_n, states = {}, {}, []
    plan = [("g0", "f", "w"), ("g0", "f", "f"), ("g1", "w", "f"), ("g1", "f", "w"), ("g2", "w", "w"), ("g2", "p", "w")]
    for k, (g, ca, cn) in enumerate(plan):
        s = _three()
        cands = [{**t, "id": f"{t['id']}{k}", "game_id": g, "state_idx": 2 * k} for t in s["candidates"]]
        states.append(_state(cands, "go to shelf 1", game=g, step=2 * k, swap=swap))
        for t in cands:
            L_a[t["id"]] = 2.0 if t["id"] == f"{ca}{k}" else -1.0
            L_n[t["id"]] = 2.0 if t["id"] == f"{cn}{k}" else -1.0
    return states, L_a, L_n


def test_contrast_counts_harm_help_and_the_identity():
    states, L_a, L_n = _contrast_data()
    c = D.contrast("rule1_with_proposal", states, L_a, L_n)
    assert c["n_states"] == 6 and c["n_games"] == 3 and c["n_odd_step_states_left_out"] == 0
    assert c["fail_rate_a"]["estimate"] == pytest.approx(3 / 6) and c["fail_rate_b"]["estimate"] == pytest.approx(2 / 6)
    assert c["fail_rate_difference"]["estimate"] == pytest.approx(1 / 6)
    assert c["change_rate"]["estimate"] == pytest.approx(4 / 6)
    ch = c["changed"]
    assert (ch["n"], ch["a_fails_b_succeeds"], ch["a_succeeds_b_fails"], ch["both_fail"], ch["both_succeed"]) == (4, 2, 1, 0, 1)
    assert ch["share_a_fails_b_succeeds"] == pytest.approx(0.5)
    # harm and help as shares of all states; the change rate split by what the change does to the outcome
    assert c["harm_rate"]["estimate"] == pytest.approx(2 / 6) and c["help_rate"]["estimate"] == pytest.approx(1 / 6)
    assert c["change_rate_outcome_changing"] == pytest.approx(3 / 6)
    assert c["change_rate_success_to_success"] == pytest.approx(1 / 6) and c["change_rate_failure_to_failure"] == 0
    assert c["identity_holds"]
    assert c["random_choice_fail_rate"] == pytest.approx(1 / 3)
    assert c["determined_share_of_chosen_failures"] == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}
    assert c["ties"] == {"a": 0, "b": 0, "mixed_outcome_a": 0, "mixed_outcome_b": 0}
    # which choices are the agent's proposal, and how often each source fails
    assert c["proposal_share"] == {"a": pytest.approx(1 / 6), "b": 0.0}
    assert c["fail_rate_by_source"]["a"]["proposal"] == {"n": 1, "fail_rate": 0.0}
    assert c["fail_rate_by_source"]["a"]["generated"] == {"n": 5, "fail_rate": pytest.approx(3 / 5)}
    assert c["fail_rate_by_source"]["b"]["proposal"]["n"] == 0
    lo, hi = c["fail_rate_difference"]["ci"]
    assert lo <= 1 / 6 <= hi and 0 < c["fail_rate_difference"]["p"] <= 1
    assert "by_acceptance" not in c


def test_random_choice_rate_is_over_the_rules_own_set():
    states, L_a, L_n = _contrast_data()
    assert D.contrast("rule1_generated_only", states, L_a, L_n)["random_choice_fail_rate"] == pytest.approx(1 / 2)


def test_odd_step_states_are_left_out_of_every_state_level_rate():
    """At odd steps the collector ran the broad generator only when the proposal failed (k = 2)."""
    assert D.BROAD_EVERY_K == 2
    states, L_a, L_n = _contrast_data()
    odd = _three(step=3, game="g9", suffix="9")
    L_a.update({"p9": -1.0, "f9": 2.0, "w9": -1.0})
    L_n.update({"p9": -1.0, "f9": -1.0, "w9": 2.0})
    base = D.contrast("rule1_with_proposal", states, L_a, L_n)
    c = D.contrast("rule1_with_proposal", states + [odd], L_a, L_n)
    assert c["n_states"] == 6 and c["n_odd_step_states_left_out"] == 1
    assert c["fail_rate_difference"]["estimate"] == base["fail_rate_difference"]["estimate"]
    assert D.calibration(states + [odd], L_a, L_n)["n_states"] == 6
    assert D.lure_rate(states + [odd], L_a, L_n)["n_states"] == 6
    rep = D.model_report(states + [odd], {"yn_aligned": L_a, "yn_absent": L_n})
    assert rep["baseline"]["yn_aligned"]["n_candidates"] == 18 and rep["n_states"]["odd_step"] == 1
    assert rep["baseline_shifts"]["yn_aligned_minus_yn_absent"]["proposal/other/all"]["n"] == 6


def test_verification_error_rates_by_parity():
    ok_p = _three()
    bad_even = _state([_t("p2", "go to shelf 1", False, step=2), _t("w2", "look", True, step=2)], "go to shelf 1", step=2)
    bad_odd = _state([_t("p3", "go to shelf 1", False, step=3, game="g1"), _t("w3", "look", True, step=3, game="g1")],
                     "go to shelf 1", step=3, game="g1")
    ok_zero = _state([_t("p4", "go to shelf 1", True, step=4), _t("w4", "look", True, step=4)], "go to shelf 1", step=4)
    # a succeeding proposal at an odd step (the collector never makes one): left out of false rejection
    ok_odd = _state([_t("p5", "go to shelf 1", True, step=5), _t("w5", "look", True, step=5)], "go to shelf 1", step=5)
    # aligned: rejects the good one, accepts both bad ones; a succeeding proposal at exactly 0 is rejected
    La = {"p": -1.0, "f": 0, "w": 0, "p2": 3.0, "w2": 1.0, "p3": 3.0, "w3": 1.0, "p4": 0.0, "w4": 0.0, "p5": 1.0, "w5": 0}
    Ln = {"p": 1.0, "f": 0, "w": 0, "p2": -3.0, "w2": 1.0, "p3": 3.0, "w3": 1.0, "p4": 1.0, "w4": 0.0, "p5": 1.0, "w5": 0}
    e = D.verification_errors([ok_p, bad_even, bad_odd, ok_zero, ok_odd], La, Ln)
    fa = e["false_acceptance"]
    assert fa["even_steps"]["n_states"] == 1 and fa["even_steps"]["rate_a"]["estimate"] == 1.0
    assert fa["even_steps"]["rate_b"]["estimate"] == 0.0 and fa["even_steps"]["difference"]["estimate"] == 1.0
    assert fa["odd_steps"]["n_states"] == 1 and fa["odd_steps"]["rate_b"]["estimate"] == 1.0
    assert fa["pooled"]["n_states"] == 2 and fa["pooled"]["rate_b"]["estimate"] == 0.5
    fr = e["false_rejection"]
    assert fr["n_states"] == 2 and fr["rate_a"]["estimate"] == 1.0 and fr["rate_b"]["estimate"] == 0.0


def test_rule2_difference_splits_by_acceptance_cell():
    """Accepted under both (no change), rejected under both (fallback harm), accepted only under the aligned task
    (help), accepted only under the neutral task (harm): the cells' contributions add up to the difference."""
    states, La, Ln = [], {}, {}
    plan = [({"p": 1, "f": -1, "w": -1}, {"p": 1, "f": -1, "w": -1}),
            ({"p": -1, "f": 2, "w": -1}, {"p": -1, "f": -1, "w": 2}),
            ({"p": 1, "f": -1, "w": -1}, {"p": -1, "f": 2, "w": -1}),
            ({"p": -1, "f": 2, "w": -1}, {"p": 1, "f": -1, "w": -1})]
    for k, (a, n) in enumerate(plan):
        states.append(_three(step=2 * k, game=f"g{k}", suffix=str(k)))
        La.update({f"{x}{k}": v for x, v in a.items()})
        Ln.update({f"{x}{k}": v for x, v in n.items()})
    c = D.contrast("rule2_verification", states, La, Ln)
    assert c["fail_rate_difference"]["estimate"] == pytest.approx(1 / 4)
    cells = c["by_acceptance"]
    assert {k: (v["n"], v["contribution"]) for k, v in cells.items() if k != "contributions_add_up"} == {
        "accepted_both": (1, 0.0), "rejected_both": (1, 0.25), "accepted_only_a": (1, -0.25), "accepted_only_b": (1, 0.25)}
    assert cells["contributions_add_up"] and cells["accepted_only_b"]["harm"] == 1 and cells["accepted_only_a"]["help"] == 1
    assert c["proposal_share"] == {"a": 0.5, "b": 0.5}


def test_ties_are_counted_and_split_evenly_in_the_sensitivity_readout():
    s = _three()
    La = {"p": 1.0, "f": 1.0, "w": 0.0}  # p (works) and f (fails) tie at the top
    Ln = {"p": 1.0, "f": 1.0, "w": 1.0}  # a three-way tie
    c = D.contrast("rule1_with_proposal", [s], La, Ln)
    assert c["ties"] == {"a": 1, "b": 1, "mixed_outcome_a": 1, "mixed_outcome_b": 1}
    assert c["tie_split"]["fail_rate_a"] == pytest.approx(1 / 2) and c["tie_split"]["fail_rate_b"] == pytest.approx(1 / 3)
    assert c["tie_split"]["difference"] == pytest.approx(1 / 6)
    chosen = min([t for t in s["candidates"] if t["id"] in ("p", "f")], key=D.tie_key)
    assert c["fail_rate_a"]["estimate"] == float(not D.succeeded(chosen))
    same = D.contrast("rule1_with_proposal", [s], {"p": 1.0, "f": 0.0, "w": 1.0}, Ln)
    assert same["ties"]["a"] == 1 and same["ties"]["mixed_outcome_a"] == 0  # p and w both work


def test_condition_order_matters():
    states, L_a, L_n = _contrast_data()
    ab = D.contrast("rule1_with_proposal", states, L_a, L_n)
    ba = D.contrast("rule1_with_proposal", states, L_n, L_a)
    assert ba["fail_rate_difference"]["estimate"] == pytest.approx(-ab["fail_rate_difference"]["estimate"])
    assert ba["changed"]["a_fails_b_succeeds"] == ab["changed"]["a_succeeds_b_fails"]


def _many(n_games, per_game, plan):
    """States in n_games games, per_game each (even steps); plan(g, k) -> (aligned choice, neutral choice)."""
    states, L_a, L_n = [], {}, {}
    for g in range(n_games):
        for k in range(per_game(g)):
            ca, cn = plan(g, k)
            s = _three(step=2 * k, game=f"g{g}", suffix=f"{g}_{k}")
            states.append(s)
            for t in s["candidates"]:
                L_a[t["id"]] = 2.0 if t["id"] == f"{ca}{g}_{k}" else -1.0
                L_n[t["id"]] = 2.0 if t["id"] == f"{cn}{g}_{k}" else -1.0
    return states, L_a, L_n


def test_permutation_flips_whole_games():
    """Twenty games, each with one harmful and one helpful change in a different state order: the per-game sums are
    zero, so a game-level flip gives p = 1; a state-level flip of the same numbers would not."""
    states, L_a, L_n = _many(20, lambda g: 2, lambda g, k: (("f", "w"), ("w", "f"))[k])
    c = D.contrast("rule1_with_proposal", states, L_a, L_n)
    assert c["fail_rate_difference"]["estimate"] == 0 and c["fail_rate_difference"]["p"] == 1.0
    assert c["change_rate"]["estimate"] == 1.0
    # four games of ten harmful changes each: only 2 of the 16 game-level sign patterns reach the observed mean,
    # so p is about 1/8; flipping the 40 states one by one would give p near 1/1001
    states, L_a, L_n = _many(4, lambda g: 10, lambda g, k: ("f", "w"))
    p = D.contrast("rule1_with_proposal", states, L_a, L_n)["fail_rate_difference"]["p"]
    assert 0.05 < p < 0.25


def test_game_bootstrap_resamples_games_not_states():
    """One game holds every failure difference: resampling games gives an interval that reaches zero."""
    states, L_a, L_n = _many(10, lambda g: 10 if g == 0 else 1, lambda g, k: ("f" if g == 0 else "w", "w"))
    c = D.contrast("rule1_with_proposal", states, L_a, L_n)
    assert c["fail_rate_difference"]["estimate"] == pytest.approx(10 / 19)
    assert c["fail_rate_difference"]["ci"][0] == 0.0


def test_baseline_reads_yes_as_a_positive_score_and_splits_by_source_and_goal():
    s = _three()
    s["candidates"].append(_t("u", "take saltshaker 1 from drawer 9", False, det=False))
    b = D.baseline([s], {"p": 1.0, "f": 0.0, "w": -2.0, "u": 4.0}, {"p": 1.01, "f": 0.001, "w": -2.0, "u": 4.0})
    assert b["n_candidates"] == 4
    assert b["accuracy"] == pytest.approx(2 / 4)  # p right, f right (0 is No), w wrong, u wrong
    assert b["yes_rate_failures"] == pytest.approx(0.5) and b["yes_rate_determined_failures"] == 0.0
    assert b["yes_rate_successes"] == pytest.approx(0.5)
    assert b["mean_L_successes"] == pytest.approx((1.01 - 2.0) / 2)  # raw scores
    g = b["by_group"]
    assert g["proposal/other/success"] == {"n": 1, "yes_rate": 1.0, "mean_L": 1.01}
    assert g["generated/goal_object/failure"] == {"n": 1, "yes_rate": 0.0, "mean_L": 0.001}
    assert g["generated/other/all"]["n"] == 2 and g["generated/other/failure"]["yes_rate"] == 1.0
    assert "proposal/goal_object/all" not in g


def test_baseline_shifts_by_source_and_goal():
    s = _three()
    La, Ln = {"p": 1.0, "f": 1.0, "w": -1.0}, {"p": 1.0, "f": -1.0, "w": -1.0}
    sh = D.baseline_shifts([s], La, Ln, {"p": 1.2, "f": 1.0, "w": -1.0}, {"p": 1.0, "f": -1.5, "w": -1.0})
    assert sh["generated/goal_object/failure"] == {"n": 1, "yes_rate_shift": 1.0, "mean_L_shift": 2.5}
    assert sh["proposal/other/success"]["yes_rate_shift"] == 0.0
    assert sh["proposal/other/success"]["mean_L_shift"] == pytest.approx(0.2)


def test_calibration_margin_and_the_aligned_shift_on_the_best_failure():
    s0, s1 = _three(step=0, suffix="0"), _three(step=2, suffix="1")
    Ln = {"p0": 1.0, "f0": -1.0, "w0": 0.5, "p1": 1.0, "f1": 2.0, "w1": 0.0}
    La = {"p0": 1.0, "f0": 2.0, "w0": 0.5, "p1": 1.0, "f1": 2.5, "w1": 0.0}
    c = D.calibration([s0, s1], La, Ln)
    assert c["n_states"] == 2 and c["n_positive_margin"] == 1
    assert c["share_margin_not_positive"] == 0.5 and c["share_shift_closes_margin"] == 1.0
    assert c["mean_aligned_shift_on_best_failure"] == pytest.approx((3.0 + 0.5) / 2)
    assert sorted([c["neutral_margin"]["q25"], c["neutral_margin"]["q75"]]) == [-1.0, 2.0]


def test_lure_rate_pairs_a_working_proposal_with_each_goal_object_failure():
    s = _three()
    s["candidates"].append(_t("g", "take saltshaker 1 from drawer 1", False, goal=True))
    s["candidates"].append(_t("n", "open drawer 2", False))  # a failure off the goal object is no lure
    bad = _state([_t("q", "go to shelf 1", False, step=2), _t("h", "take saltshaker 1 from shelf 1", False, step=2, goal=True),
                  _t("v", "look", True, step=2)], "go to shelf 1", step=2)
    La = {"p": 1.0, "f": 2.0, "w": 0, "g": 0.5, "n": 9, "q": 0, "h": 9, "v": 0}
    Ln = {"p": 1.0, "f": -2.0, "w": 0, "g": -0.5, "n": 9, "q": 0, "h": 9, "v": 0}
    lr = D.lure_rate([s, bad], La, Ln)
    assert lr["n_pairs"] == 2 and lr["n_states"] == 1  # a failing proposal has nothing to be lured from
    assert lr["rate_a"]["estimate"] == 0.5 and lr["rate_b"]["estimate"] == 0.0
    assert lr["difference"]["estimate"] == 0.5


def test_decomposition_adds_up_on_the_states_with_a_swap():
    states, L_a, L_n = _contrast_data()
    L_s = {k: (2.0 if k.startswith("w") else -1.0) for k in L_a}  # swapped: always the working examine action
    states[0]["swap"] = None  # its harm drops out of all three differences
    odd = _three(step=1, game="g9", suffix="9")  # odd steps never enter
    L_a.update({"p9": -1.0, "f9": 2.0, "w9": -1.0})
    L_n.update({"p9": -1.0, "f9": 2.0, "w9": -1.0})
    L_s.update({"p9": -1.0, "f9": -1.0, "w9": 2.0})
    d = D.decomposition(states + [odd], {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_s})["rule1_with_proposal"]
    assert d["n_states"] == 5
    assert d["fail_rates"] == {"yn_aligned": pytest.approx(2 / 5), "yn_absent": pytest.approx(2 / 5), "yn_swapped": 0.0}
    assert d["aligned_minus_neutral"] == pytest.approx(0) and d["aligned_minus_swapped"] == pytest.approx(2 / 5)
    assert d["swapped_minus_neutral"] == pytest.approx(-2 / 5) and d["identity_holds"]


def test_model_report_adds_swapped_contrasts_only_where_scored():
    states, L_a, L_n = _contrast_data()
    rep = D.model_report(states, {"yn_aligned": L_a, "yn_absent": L_n})
    assert {"baseline", "yn_aligned_vs_yn_absent", "calibration", "baseline_shifts"} <= set(rep)
    assert "decomposition" not in rep and "yn_aligned_vs_yn_swapped" not in rep
    assert set(rep["yn_aligned_vs_yn_absent"]) == set(D.RULES) | {"rule2_errors", "lure"}
    states[0]["swap"] = None
    rep = D.model_report(states, {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_n})
    assert rep["yn_aligned_vs_yn_swapped"]["rule1_with_proposal"]["n_states"] == 5
    assert rep["yn_aligned_vs_yn_absent"]["rule1_with_proposal"]["n_states"] == 6
    # the aligned and neutral baselines read every even-step state; the swapped one only those with a swap
    assert rep["baseline"]["yn_aligned"]["n_candidates"] == 18 and rep["baseline"]["yn_absent"]["n_candidates"] == 18
    assert rep["baseline"]["yn_swapped"]["n_candidates"] == 15
    assert set(rep["baseline_shifts"]) == {"yn_aligned_minus_yn_absent", "yn_aligned_minus_yn_swapped",
                                           "yn_swapped_minus_yn_absent"}
    assert rep["baseline_shifts"]["yn_aligned_minus_yn_swapped"]["proposal/other/all"]["n"] == 5
    assert rep["decomposition"]["rule2_verification"]["n_states"] == 5


def test_complete_states_drop_a_state_with_any_unscored_candidate():
    states, L_a, L_n = _contrast_data()
    L_n = dict(L_n)
    del L_n["w3"]
    keep, short = D.complete_states(states, {"yn_aligned": L_a, "yn_absent": L_n})
    assert short == 1 and [s["state_idx"] for s in keep] == [0, 2, 4, 8, 10]


def test_complete_states_keep_a_state_without_a_swap_when_swapped_is_scored():
    states, L_a, L_n = _contrast_data()
    states[0]["swap"] = None
    L_s = {k: v for k, v in L_n.items() if not k.endswith("0")}  # no swapped score where none is defined
    keep, short = D.complete_states(states, {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_s})
    assert short == 0 and len(keep) == 6
    del L_s["p1"]
    assert D.complete_states(states, {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_s})[1] == 1


# ---------------- the Option C rule ----------------


def _rep(harm=0.08, ci=(0.02, 0.12), p=0.01, a_s=0.05, a_n=0.07, rule1=None):
    r2 = {"harm_rate": {"estimate": harm}, "fail_rate_difference": {"estimate": sum(ci) / 2, "ci": list(ci), "p": p},
          "change_rate": {"estimate": 0.2}, "change_rate_outcome_changing": 0.1, "change_rate_success_to_success": 0.1}
    r1 = rule1 or {"harm_rate": {"estimate": 0.0}, "fail_rate_difference": {"estimate": 0.0, "ci": [-0.1, 0.1], "p": 1.0},
                   "change_rate": {"estimate": 0.0}}
    return {"yn_aligned_vs_yn_absent": {"rule2_verification": r2, "rule1_with_proposal": r1},
            "decomposition": {"rule2_verification": {"aligned_minus_swapped": a_s, "aligned_minus_neutral": a_n}}}


NULL = dict(harm=0.01, ci=(-0.02, 0.02), p=0.9, a_s=0.0, a_n=0.0)


@pytest.mark.parametrize("kw, category", [
    (dict(), "worth doing"),
    (dict(harm=0.05), "worth doing"),  # at least 5% of states
    (dict(a_s=0.25, a_n=0.5), "worth doing"),  # exactly half
    (dict(a_s=0.03, a_n=0.07), "toward failure, not attributable to the real task"),
    (dict(a_s=-0.01, a_n=-0.03), "toward failure, not attributable to the real task"),  # must be positive
    (dict(harm=0.049), "toward failure, too rare"),
    (dict(p=0.05), "no reliable change in failures"),  # the interval alone is not enough
    (dict(ci=(0.0, 0.1)), "no reliable change in failures"),  # an interval that reaches zero
    (dict(ci=(-0.12, -0.01), harm=0.0), "changes toward success"),
    (NULL, "no reliable change in failures"),
])
def test_option_c_categories(kw, category):
    assert D.option_c_model(_rep(**kw))["category"] == category
    assert D.option_c({"qwen3-8b": _rep(**kw), "llama3.1-8b": _rep(**NULL)})["verdict"] == category
    assert D.option_c({"qwen3-8b": _rep(**kw), "llama3.1-8b": _rep(**NULL)})["worth_doing"] == (category == "worth doing")


def test_option_c_reads_rule2_only_and_primary_models_only():
    strong_rule1 = {"harm_rate": {"estimate": 0.3}, "fail_rate_difference": {"estimate": 0.3, "ci": [0.2, 0.4], "p": 0.001},
                    "change_rate": {"estimate": 0.5}}
    assert D.option_c({"qwen3-8b": _rep(**NULL, rule1=strong_rule1),
                       "llama3.1-8b": _rep(**NULL)})["verdict"] == "no reliable change in failures"
    assert D.option_c({"qwen3-8b": _rep(**NULL), "llama3.1-8b": _rep(**NULL),
                       "mistral-7b-v0.3": _rep()})["verdict"] == "no reliable change in failures"
    # the first category in order that some primary model reaches
    assert D.option_c({"qwen3-8b": _rep(harm=0.01), "llama3.1-8b": _rep(ci=(-0.1, -0.01))})["verdict"] == "toward failure, too rare"
    assert D.option_c({"qwen3-8b": _rep(**NULL), "llama3.1-8b": _rep(ci=(-0.1, -0.01))})["verdict"] == "changes toward success"


def test_option_c_waits_for_both_primary_models_and_their_swapped_arm():
    assert D.option_c({"qwen3-8b": _rep()})["verdict"] == "waiting for a primary model"
    no_swap = {k: v for k, v in _rep().items() if k != "decomposition"}
    o = D.option_c({"qwen3-8b": _rep(), "llama3.1-8b": no_swap})
    assert o["verdict"] == "waiting for a primary model" and "swapped" in o["per_model"]["llama3.1-8b"]


def test_option_c_on_a_real_report():
    """The rule reads the harm rate, interval, p and decomposition that model_report computes."""
    states, L_a, L_n = _many(12, lambda g: 3, lambda g, k: ("f", "w"))
    for s in states:  # rule 2 falls back: the proposal is rejected under both tasks
        L_a[_p(s)] = L_n[_p(s)] = -5.0
    L_s = dict(L_n)
    rep = D.model_report(states, {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_s})
    m = D.option_c_model(rep)
    assert m["harm_rate"] == 1.0 and m["fail_rate_difference"] == 1.0 and m["p"] < 0.05
    assert m["aligned_minus_swapped"] == 1.0 and m["category"] == "worth doing"
    rep = D.model_report(states, {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_a})
    assert D.option_c_model(rep)["category"] == "toward failure, not attributable to the real task"


def _p(s):
    return D._proposal(s)["id"]


# ---------------- Yes/No gate ----------------


def test_yes_no_gate_passes_a_consistent_tokenizer_and_fails_a_split_yes(mock_score):
    spec = families.MODELS["llama3.2-3b"]
    msgs = [D.messages_for("llama3.2-3b")(_item(), a) for a in ("yn_aligned", "yn_absent")]
    assert D.yes_no_gate(_scorer(FakeTok(), spec), msgs)["ok"]
    bad = D.yes_no_gate(_scorer(FakeTok(yes_pieces=2), spec), msgs)
    assert not bad["ok"] and any("not one token" in p for p in bad["problems"])
    odd = D.yes_no_gate(_scorer(FakeTok(odd_reply_after=(("explore", "Yes"),)), spec), msgs)
    assert not odd["ok"] and odd["n_problems"] == 1  # only the neutral prompt renders Yes differently
    assert not D.yes_no_gate(_scorer(FakeTok(), spec), [])["ok"]


# ---------------- runs (mock) ----------------


def _score_cfg(tag="qwen3-8b", **kw):
    return {"run_id": f"decisionsim_score_{tag}", "model": tag, "sources": ["rq12_transitions_smoke"],
            "n_states": "all", "n_shards": 1, **kw}


def test_every_model_is_registered_and_configured():
    from pathlib import Path
    import yaml
    from wf.runs import RUNNERS
    root = Path(__file__).resolve().parents[1] / "wf" / "config"
    for tag in D.MODEL_TAGS:
        assert f"decisionsim_score_{tag}" in RUNNERS
        cfg = yaml.safe_load((root / f"decisionsim_score_{tag}.yaml").read_text())
        assert cfg["model"] == tag and cfg["n_states"] == "all"
        assert cfg["arms"] == ["yn_aligned", "yn_absent", "yn_swapped"] == list(D.DEFAULT_ARMS)
    ana = yaml.safe_load((root / "decisionsim_analysis.yaml").read_text())
    assert set(ana["models"]) == set(D.MODEL_TAGS) and ana["n_states"] == "all"
    assert "llama3.2-1b" not in D.MODEL_TAGS  # failed wave A's competence gate


def test_end_to_end_score_resume_and_analysis(mock_root):
    assert D.run_decisionsim_score(_score_cfg(arms=["yn_aligned", "yn_absent", "yn_swapped"])) == 0
    d = mock_root / "decisionsim_score_qwen3-8b" / "shard_00_of_01"
    recs = [json.loads(x) for x in (d / "scoring.jsonl").read_text().splitlines()]
    states = json.loads((d / "states.json").read_text())
    assert states["states"] and recs and {r["condition"] for r in recs} == {"yn_aligned", "yn_absent", "yn_swapped"}
    # a resume scores nothing new
    assert D.run_decisionsim_score(_score_cfg(arms=["yn_aligned", "yn_absent", "yn_swapped"], resume=True)) == 0
    assert len((d / "scoring.jsonl").read_text().splitlines()) == len(recs)
    assert D.run_decisionsim_score(_score_cfg("llama3.1-8b", arms=["yn_aligned", "yn_absent"])) == 0
    cfg = {"run_id": "decisionsim_analysis", "sources": ["rq12_transitions_smoke"], "n_states": "all",
           "models": {"qwen3-8b": "decisionsim_score_qwen3-8b", "llama3.1-8b": "decisionsim_score_llama3.1-8b",
                      "mistral-7b-v0.3": "decisionsim_score_mistral-7b-v0.3"}}
    assert D.run_decisionsim_analysis(cfg) == 0
    m = json.loads((mock_root / "decisionsim_analysis" / "metrics.json").read_text())
    assert set(m["models"]) == {"qwen3-8b", "llama3.1-8b"} and "mistral-7b-v0.3" in m["waiting"]
    q = m["models"]["qwen3-8b"]
    assert q["n_states_incomplete"] == 0 and "yn_aligned_vs_yn_swapped" in q
    assert "yn_aligned_vs_yn_swapped" not in m["models"]["llama3.1-8b"]
    assert q["yn_aligned_vs_yn_absent"]["rule1_with_proposal"]["n_states"] == m["states"]["n_even_step"] > 0
    assert m["states"]["n_odd_step"] > 0 and m["states"]["n_selected"] == m["states"]["n_eligible"]
    assert q["yn_aligned_vs_yn_absent"]["rule2_errors"]["false_acceptance"]["odd_steps"]["n_states"] == m["states"]["n_odd_step"]
    # llama was scored without the swapped arm, so the rule waits for it
    assert m["option_c"]["verdict"] == "waiting for a primary model"
    assert m["option_c"]["per_model"]["qwen3-8b"]["category"] in D.OPTION_C_CATEGORIES
    # an analysis over another sample waits instead of reading these scores
    assert D.run_decisionsim_analysis({**cfg, "n_states": 1}) == 0
    m1 = json.loads((mock_root / "decisionsim_analysis" / "metrics.json").read_text())
    assert m1["models"] == {} and any("another sample" in p for p in m1["waiting"]["qwen3-8b"])


def test_resume_refuses_scores_of_another_sample(mock_root):
    assert D.run_decisionsim_score(_score_cfg()) == 0
    d = mock_root / "decisionsim_score_qwen3-8b" / "shard_00_of_01"
    lines = (d / "scoring.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["score_hash"] = "0" * 16
    (d / "scoring.jsonl").write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
    with pytest.raises(SystemExit):
        D.run_decisionsim_score(_score_cfg(resume=True))


def test_scores_are_saved_one_at_a_time(mock_root, monkeypatch):
    """A crash part-way keeps every score already made."""
    calls = {"n": 0}
    real = D.LogoddsScorer.score_one_token_pair

    def crash(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(self, *a, **kw)

    monkeypatch.setattr(D.LogoddsScorer, "score_one_token_pair", crash)
    with pytest.raises(RuntimeError):
        D.run_decisionsim_score(_score_cfg())
    d = mock_root / "decisionsim_score_qwen3-8b" / "shard_00_of_01"
    assert len((d / "scoring.jsonl").read_text().splitlines()) == 2


def test_a_failed_gate_scores_nothing(mock_root, monkeypatch):
    spec = families.MODELS["mistral-7b-v0.3"]

    def bad_scorer(tag):
        s = _scorer(FakeTok(yes_pieces=2), spec)
        s.mock = False
        return s

    monkeypatch.setattr(D, "open_scorer", bad_scorer)
    with pytest.raises(families.TokenizationMismatch):
        D.run_decisionsim_score(_score_cfg("mistral-7b-v0.3"))
    d = mock_root / "decisionsim_score_mistral-7b-v0.3" / "shard_00_of_01"
    assert not (d / "scoring.jsonl").exists() and json.loads((d / "tokenization_checks.json").read_text())["ok"] is False


def test_arms_must_include_aligned_and_neutral(mock_root):
    with pytest.raises(SystemExit):
        D.run_decisionsim_score(_score_cfg(arms=["yn_aligned", "yn_swapped"]))
    with pytest.raises(SystemExit):
        D.run_decisionsim_score(_score_cfg("llama3.2-1b"))


def test_each_condition_gets_its_own_prompt(mock_root, monkeypatch):
    """Aligned: the real task line; neutral: "explore the room"; swapped: the real line with the state's swap object;
    every prompt ends with the yes/no question, and every (candidate, arm) is scored once."""
    from wf import rq1v
    from wf.data.goals import make_swapped_task_line
    from wf.rq2 import load_items
    seen = []
    real = D.LogoddsScorer.score_one_token_pair

    def record(self, *, transition_id, condition, messages, o_plus, o_minus):
        seen.append((transition_id, condition, messages, o_plus, o_minus))
        return real(self, transition_id=transition_id, condition=condition, messages=messages, o_plus=o_plus, o_minus=o_minus)

    monkeypatch.setattr(D.LogoddsScorer, "score_one_token_pair", record)
    assert D.run_decisionsim_score(_score_cfg()) == 0  # the default arms: all three
    states = D.build_states(load_items(("rq12_transitions_smoke",)))
    where = {t["id"]: (t, s) for s in states for t in s["candidates"]}
    assert len(seen) == len({(x[0], x[1]) for x in seen}) == len(D.score_jobs(states, D.DEFAULT_ARMS))
    arms = {}
    for tid, cond, msgs, plus, minus in seen:
        t, s = where[tid]
        last = msgs[-1]["content"]
        assert (plus, minus) == ("Yes", "No") and last.endswith(rq1v.YN_QUESTION)
        assert f"Action: {t['action']}" in last
        line = {"yn_aligned": t["task_line"], "yn_absent": "Your task is to: explore the room.",
                "yn_swapped": make_swapped_task_line(t["task_line"], t["goal_object"], s["swap"] or "")}[cond]
        assert line in last
        if cond != "yn_aligned":
            assert t["task_line"] not in last
        arms.setdefault(tid, set()).add(cond)
    assert all(arms[t] == ({"yn_aligned", "yn_absent", "yn_swapped"} if s["swap"] else {"yn_aligned", "yn_absent"})
               for t, (_, s) in where.items())


def test_shards_split_the_states_by_game(mock_root):
    for i in range(2):
        assert D.run_decisionsim_score(_score_cfg(n_shards=2, shard=i)) == 0
    root = mock_root / "decisionsim_score_qwen3-8b"
    parts = [json.loads((root / f"shard_{i:02d}_of_02" / "states.json").read_text())["states"] for i in range(2)]
    games = [{g for g, _ in p} for p in parts]
    assert parts[0] and parts[1] and not games[0] & games[1]
    assert parts[0] + parts[1] != [] and len(parts[0]) + len(parts[1]) == json.loads(
        (root / "shard_00_of_02" / "states.json").read_text())["n_states_sampled"]


def test_the_gate_on_a_resume_checks_every_prompt_not_only_the_unscored(mock_root, monkeypatch):
    assert D.run_decisionsim_score(_score_cfg("mistral-7b-v0.3")) == 0  # mock scorer: no gate
    d = mock_root / "decisionsim_score_mistral-7b-v0.3" / "shard_00_of_01"
    lines = (d / "scoring.jsonl").read_text().splitlines()
    (d / "scoring.jsonl").write_text("\n".join(lines[:5]) + "\n")
    checked = []

    def fake_open(tag):
        s = _scorer(FakeTok(), families.MODELS["mistral-7b-v0.3"])
        s.mock = False
        return s

    def gate(scorer, prompts):
        checked.append(len(prompts))
        return {"ok": False, "n_problems": 1, "problems": ["stop"], "n_prompts": len(prompts)}

    monkeypatch.setattr(D, "open_scorer", fake_open)
    monkeypatch.setattr(D, "yes_no_gate", gate)
    with pytest.raises(families.TokenizationMismatch):
        D.run_decisionsim_score(_score_cfg("mistral-7b-v0.3", resume=True))
    assert checked == [len(lines)]
