"""Stage 1b exposure analysis (scripts/decisionsim_agent_exposure.py; exploratory): the rule classes, the
forced-choice rule, the decision-chain funnel and a small synthetic data set with a known injected bias."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

from tests.test_decisionsim_agent import _cand, _lab, _scores
from wf import decisionsim_agent as D

_spec = importlib.util.spec_from_file_location(
    "exposure", Path(__file__).resolve().parents[1] / "scripts" / "decisionsim_agent_exposure.py")
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

A, B, S = D.AWARE, D.BLIND, D.SWAPPED


def _one(cands, aware, blind, swapped=None, t=2, game="g1"):
    """A primary-list view and its quantised scores (per arm, in candidate order)."""
    lab = _lab(cands, t=t, game=game)
    per = {A: [aware], B: [blind]}
    if swapped is not None:
        per[S] = [swapped]
    L = _scores([lab], per)
    Lq = {a: {k: D.quantize(v) for k, v in d.items()} for a, d in L.items()}
    return D.views([lab])[0], Lq, L


# ---------------- forced choice ----------------


def test_forced_choice_takes_the_highest_score():
    s, L, _ = _one([_cand(1, False), _cand(2, True, goal=True)], aware=[1.0, 3.0], blind=[2.0, -1.0])
    assert X.forced_choice(A, s, L)["rank"] == 2
    assert X.forced_choice(B, s, L)["rank"] == 1


def test_forced_choice_rounds_to_the_grid_then_breaks_ties_by_rank():
    # 1.004 and 1.006 both round to 1.0 on the 1/64 grid: a tie, so rank 1 wins although its raw score is lower.
    s, L, Lraw = _one([_cand(1, False), _cand(2, True)], aware=[1.004, 1.006], blind=[0, 0])
    assert X.forced_choice(A, s, Lraw)["rank"] == 1
    # an exact tie between ranks 2 and 3 goes to rank 2
    s, L, _ = _one([_cand(1, False), _cand(2, True), _cand(3, False)], aware=[-5, 2, 2], blind=[0, 0, 0])
    assert X.forced_choice(A, s, L)["rank"] == 2


def test_forced_choice_never_falls_back():
    # every candidate rejected: the frozen rule abstains to rank 1, forced choice takes the least-bad score
    s, L, _ = _one([_cand(1, True), _cand(2, False)], aware=[-9, -1], blind=[-9, -1])
    assert D.choose("aware", s, L)["rank"] == 1 and X.forced_choice(A, s, L)["rank"] == 2


def test_forced_choice_single_candidate_is_that_candidate():
    s, L, _ = _one([_cand(1, True)], aware=[-3], blind=[-3])
    assert X.forced_choice(A, s, L)["rank"] == 1 and not X.multi(s)


# ---------------- rule classes ----------------


def test_rule_classes_partition_and_label_each_case():
    w, f = (lambda r: _cand(r, False)), (lambda r: _cand(r, True))
    cases = {
        "A_both_fall_back": ([f(1), w(2)], [-1, -1], [-1, -1]),
        "B_only_aware_falls_back": ([f(1), w(2)], [-1, -1], [-1, 1]),
        "C_only_blind_falls_back": ([f(1), w(2)], [-1, 1], [-1, -1]),
        "D_both_choose_rank1": ([w(1), f(2)], [1, 1], [1, -1]),
        "E_both_choose_same_alternative": ([f(1), w(2)], [-1, 1], [-1, 1]),
        "F_choose_different_candidates": ([w(1), f(2)], [-1, 1], [1, -1]),
    }
    for want, (cs, aw, bl) in cases.items():
        s, L, _ = _one(cs, aware=aw, blind=bl)
        assert X.rule_class(s, L) == want, want
    s, L, _ = _one([_cand(1, False)], aware=[1], blind=[1])
    assert X.rule_class(s, L) is None


def test_class_contributions_add_up_to_the_difference():
    rng = random.Random(3)
    labs, aw, bl = [], [], []
    for i in range(60):
        k = rng.randint(1, 4)
        cs = [_cand(r, rng.random() < 0.4, goal=rng.random() < 0.3, action=f"a{i}_{r}") for r in range(1, k + 1)]
        labs.append(_lab(cs, game=f"g{i % 7}", t=i))
        aw.append([rng.uniform(-3, 3) for _ in cs])
        bl.append([rng.uniform(-3, 3) for _ in cs])
    L = {a: {k: D.quantize(v) for k, v in d.items()} for a, d in _scores(labs, {A: aw, B: bl}).items()}
    states = D.views(labs)
    c = X.classes(states, L)
    assert c["partition_holds"] and c["contributions_add_up"]
    diff = sum(float(D.choose("aware", s, L)["failed"]) - float(D.choose("blind", s, L)["failed"]) for s in states) / len(states)
    assert abs(sum(c[k]["contribution_to_difference"] for k in X.CLASSES) - diff) < 1e-12
    # classes D and E can never contribute: both verifiers chose the same action
    assert c["D_both_choose_rank1"]["harm"] == c["D_both_choose_rank1"]["help"] == 0
    assert c["E_both_choose_same_alternative"]["harm"] == c["E_both_choose_same_alternative"]["help"] == 0


# ---------------- the funnel ----------------


def test_state_flags_follow_the_chain():
    # scores move but no acceptance flips: nothing downstream changes
    s, L, Lraw = _one([_cand(1, False), _cand(2, True)], aware=[3.0, -2.0], blind=[1.0, -4.0])
    f = X.state_flags(s, L, Lraw)
    assert f["score_grid"] and f["score_1nat"] and not f["flip"] and not f["choice"] and not f["outcome"]
    # a flip below rank 1 that the rule pre-empts: accepted set changes, choice does not
    s, L, Lraw = _one([_cand(1, False), _cand(2, True)], aware=[2.0, 1.0], blind=[2.0, -1.0])
    f = X.state_flags(s, L, Lraw)
    assert f["flip"] and f["order"] is False and not f["choice"]
    # a flip at rank 1 that changes the choice and its outcome
    s, L, Lraw = _one([_cand(1, False), _cand(2, True)], aware=[-1.0, 1.0], blind=[1.0, 1.0])
    f = X.state_flags(s, L, Lraw)
    assert f["flip"] and f["order"] and f["choice"] and f["outcome"]
    # a sub-grid change is not a score change
    s, L, Lraw = _one([_cand(1, False), _cand(2, True)], aware=[1.0, 1.01], blind=[1.0, 1.0])
    assert not X.state_flags(s, L, Lraw)["score_grid"]


def test_funnel_counts_and_checks():
    s1, L1, R1 = _one([_cand(1, False), _cand(2, True)], aware=[-1.0, 1.0], blind=[1.0, 1.0], game="g1", t=1)
    s2, L2, R2 = _one([_cand(1, False)], aware=[1.0], blind=[1.0], game="g2", t=1)
    L = {a: {**L1[a], **L2[a]} for a in (A, B)}
    R = {a: {**R1[a], **R2[a]} for a in (A, B)}
    fu = X.funnel([s1, s2], L, R)
    assert fu["multi"]["n"] == 1 and fu["choice"]["n"] == 1 and fu["harm"] == 1 and fu["net"] == 1
    assert fu["choice_without_flip"] == 0 and fu["outcome_without_choice"] == 0


# ---------------- synthetic check with a known bias ----------------


def _synthetic(bias: float, seed: int = 11):
    """Agent lists whose rank 1 works and is clearly accepted; a regime alternative at rank 2. The aware arm lifts the
    regime candidate by `bias` nats. The frozen rule never reaches rank 2 (rank 1 is accepted), so its difference is
    zero whatever the bias; forced choice picks the regime candidate once the bias exceeds rank 1's margin."""
    rng = random.Random(seed)
    labs, aw, bl, sw = [], [], [], []
    for i in range(40):
        cs = [_cand(1, False, action=f"go to x{i}"), _cand(2, True, goal=True, det=True, action=f"take saltshaker {i}")]
        labs.append(_lab(cs, game=f"g{i % 10}", t=12 + i % 10))
        base = rng.uniform(1.0, 2.0)
        bl.append([base + 1.0, base - 0.5])
        aw.append([base + 1.0, base - 0.5 + bias])
        sw.append([base + 1.0, base - 0.5])
    L = {a: {k: D.quantize(v) for k, v in d.items()} for a, d in _scores(labs, {A: aw, B: bl, S: sw}).items()}
    return D.views(labs), L


def test_synthetic_bias_reaches_forced_choice_but_not_the_frozen_rule():
    states, L = _synthetic(bias=3.0)
    assert all(D.encounter(s) for s in states)
    frozen = X.frozen_differences(states, L)["all_states"]
    assert frozen["estimate"] == 0 and frozen["harm"] == 0
    fc = X.forced(states, L)
    ab = fc["all_states"]["aware_minus_blind"]
    assert ab["estimate"] == 1.0 and ab["harm"] == 40 and ab["harm_regime_share"] == 1.0
    assert fc["all_states"]["aware_minus_swapped"]["estimate"] == 1.0
    assert fc["all_states"]["fail_rate"]["oracle"]["estimate"] == 0.0
    c = X.classes(states, L)
    assert c["D_both_choose_rank1"]["n"] == 40
    rf = X.regime_funnel(states, L)
    assert rf["regime_candidate_flips_to_accept_under_the_task"] == 0  # accepted by both already
    # no bias: forced choice agrees too
    states, L = _synthetic(bias=0.0)
    assert X.forced(states, L)["all_states"]["aware_minus_blind"]["estimate"] == 0


def test_synthetic_bias_that_flips_acceptance_is_preempted_by_rank1():
    # the regime candidate is rejected task-blind and accepted under the task, but rank 1 is accepted first
    lab = _lab([_cand(1, False), _cand(2, True, goal=True, det=True)], game="g1", t=15)
    L = {a: {k: D.quantize(v) for k, v in d.items()} for a, d in _scores([lab], {A: [[2.0, 1.0]], B: [[2.0, -1.0]]}).items()}
    s = D.views([lab])[0]
    rf = X.regime_funnel([s], L)
    assert rf["regime_candidate_flips_to_accept_under_the_task"] == 1 and rf["that_candidate_becomes_the_aware_choice"] == 0
