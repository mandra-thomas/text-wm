"""The minimal interpretability pilot (wf/interp/pilot.py, wf/interp/analysis.py; design amendment 1).

Analysis helpers on synthetic numbers, and the four runs end to end on a tiny random model with synthetic items
(no download, no GPU): stage 0 with its reproduction gate passing and failing, patching, cross-fitted ablation,
and the CPU analysis.
"""
import json
import math
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

from tests.test_interp import SPECIALS, SPLIT, TEMPLATE, _item, _qwen  # noqa: E402
from wf.interp import analysis as A  # noqa: E402
from wf.interp import pilot as PI  # noqa: E402
from wf.interp import prompts as PR  # noqa: E402

# ---------------- analysis helpers ----------------


def _gate_records(ids, d):
    return [{"id": i, "fmt": "yn", "goal": g, "d": d[(i, f"yn_{g}")]} for i in ids for g in ("aligned", "swapped", "absent")]


def test_reproduction_gate_checks_items_not_only_the_mean():
    rng = np.random.default_rng(0)
    ids = [f"t{k}" for k in range(200)]
    logged = {}
    for i in ids:
        base = rng.normal(0, 3)
        logged[(i, "yn_aligned")], logged[(i, "yn_swapped")], logged[(i, "yn_absent")] = base + rng.normal(3, 2), base, base + 1
    close = {k: v + rng.normal(0, 0.05) for k, v in logged.items()}
    ok = A.reproduction_gate(_gate_records(ids, close), logged, "qwen3-8b-fewshot")
    assert ok["passed"] and ok["coverage"] == 1.0 and abs(ok["mean_gap"]) < 0.05
    # the same values given to the wrong items: the mean is unchanged, every item is wrong
    perm = dict(zip(ids, ids[1:] + ids[:1]))
    scrambled = {(perm[i], a): v for (i, a), v in close.items()}
    bad = A.reproduction_gate(_gate_records(ids, scrambled), logged, "qwen3-8b-fewshot")
    assert abs(bad["mean_gap"]) < 0.05 and not bad["passed"] and not bad["checks"]["r_effect"]
    shifted = {k: v + (0.5 if k[1] == "yn_aligned" else 0) for k, v in close.items()}
    assert not A.reproduction_gate(_gate_records(ids, shifted), logged, "qwen3-8b-fewshot")["checks"]["mean"]
    few = A.reproduction_gate(_gate_records(ids[:150], close), logged, "qwen3-8b-fewshot")
    assert not few["passed"] and not few["checks"]["coverage"]


def _patch_recs(n, n_layers, frac_by_layer, games=8, step0=False, group="hist_all"):
    recs = []
    for k in range(n):
        cells, cc = {}, {}
        for layer in range(n_layers):
            key = f"resid|{layer}|{group}"
            cells[key] = 4.0 - 3.0 * frac_by_layer[layer] + 0.01 * (k % 3)
            cc[key] = 4.0
        recs.append({"game_id": f"g{k % games}", "d_target": 4.0, "d_source": 1.0, "cells": cells, "cell_clean": cc,
                     "history_len": 0 if step0 else 3})
    return recs


def test_group_curve_and_maximum_over_layers():
    recs = _patch_recs(40, 6, [0.0, 0.1, 0.5, 0.2, 0.0, 0.0])
    c = A.group_curve(recs, "hist_all", 6, B=200, seed=1)
    assert c["fraction"][2] == pytest.approx(0.5, abs=0.02) and c["argmax_layer"] == 2
    assert c["max"] == pytest.approx(max(c["fraction"]))
    assert c["max_ci"][0] <= c["max"] <= c["max_ci"][1]
    lo, hi = c["ci_bonferroni"][2]
    assert lo <= c["ci"][2][0] <= c["ci"][2][1] <= hi  # the Bonferroni interval is the wider one
    assert A.group_curve([], "hist_all", 6)["n"] == 0


def test_patch_summary_history_share_is_undefined_at_step_zero():
    recs = _patch_recs(10, 6, [0.0] * 6, step0=True, group="action")  # no history group at all
    s = A.patch_summary(recs, [], 6, B=50)
    assert s["history_share"]["defined"] is False and s["action_share"]["max"] == pytest.approx(0.0, abs=0.01)
    later = _patch_recs(10, 6, [0.4] * 6)
    plac = _patch_recs(10, 6, [0.01] * 6)
    s2 = A.patch_summary(later, plac, 6, B=50)
    assert s2["history_share"]["defined"] and s2["history_share"]["max"] == pytest.approx(0.4, abs=0.02)
    assert all(s2["above_placebo_floor"]["hist_all"])
    rows = A.by_stratum(recs + later, len)
    assert rows == {"step0": 10, "later": 10, "all": 20}


def test_estimate_directions_on_synthetic_residuals():
    rng = np.random.default_rng(4)
    d = 16
    e = np.eye(d)
    S = rng.standard_normal((300, d))
    Al = S + 2.0 * e[0] + 1.0 * e[1]  # the task moves along e0 and the success direction e1
    succ = rng.standard_normal((200, d)) + 3.0 * e[1]
    fail = rng.standard_normal((200, d))
    plus = S + 0.5 * e[2]
    x = A.estimate_directions({"aligned": Al, "swapped": S, "swapped_plus": plus}, succ, fail)
    assert abs(x["success"] @ e[1]) > 0.95 and abs(x["mention"] @ e[2]) > 0.9
    assert abs(x["task_perp_success"] @ x["success"]) < 1e-9 and x["task_perp_success"] @ e[0] > 0.95
    assert x["steer_success"] == pytest.approx(Al.mean(0) @ x["success"] - S.mean(0) @ x["success"])
    cos = A.direction_cosines(x, e[1])
    assert cos["task_success"] == pytest.approx(x["task"] @ x["success"]) and abs(cos["task_mention"]) < 0.3


def test_removed_share_gap_and_layer_choice():
    games = {f"i{k}": f"g{k % 5}" for k in range(20)}
    by = {"aligned": {}, "swapped": {}, "absent": {}}
    for k in range(20):
        i = f"i{k}"
        by["aligned"][i] = {"d": 3.0, "edits": {"task|4": 1.5, "task|8": 2.5}, "edit_clean": {"task|4": 3.0, "task|8": 3.0}}
        by["swapped"][i] = {"d": 0.0, "edits": {"task|4": 0.0, "task|8": 0.0}, "edit_clean": {"task|4": 0.0, "task|8": 0.0}}
        by["absent"][i] = {"d": 2.0 if k < 10 else -2.0, "edits": {"task|4": 1.0 if k < 10 else -2.0},
                           "edit_clean": {"task|4": 2.0 if k < 10 else -2.0}}
    ids = sorted(games)
    r4 = A.removed_share(by, ids, lambda i: "task|4", games, B=100)
    r8 = A.removed_share(by, ids, lambda i: "task|8", games, B=100)
    assert r4["share"] == pytest.approx(0.5) and r8["share"] == pytest.approx(1 / 6)
    assert A.choose_layer({4: r4["share"], 8: r8["share"], 12: float("nan")}) == 4
    g = A.gap(by["absent"], [f"i{k}" for k in range(10)], [f"i{k}" for k in range(10, 20)], lambda i: "task|4",
              games, B=100)
    assert g["clean"]["gap"] == pytest.approx(4.0) and g["edited"]["gap"] == pytest.approx(3.0)
    assert g["ratio"] == pytest.approx(0.75)
    sh = A.mean_shift(by["absent"], ids, lambda i: "task|4", games, B=100)
    assert sh["mean"] == pytest.approx(-0.5)


# ---------------- amendment 2 (review 2): analysis on synthetic data with known answers ----------------


def _unit(v):
    return v / np.linalg.norm(v)


def _ablate(H, u, c):
    return H - ((H @ u) - c)[:, None] * u


def _world(rng, n, d, shared):
    """Two depths of a toy network: a middle layer M and the last layer, whose readout is linear in the normed
    residual (as at a transformer's last layer). With `shared` False the aligned-minus-swapped differences at the
    middle layer are item-specific (mean zero) and the last layer turns them into a shared shift through a
    nonlinearity (review 2's case); with `shared` True they have a shared direction that the last layer reads."""
    w, e = _unit(rng.standard_normal(d)), _unit(rng.standard_normal(d))
    g = 1 + 0.1 * rng.standard_normal(d)
    P = rng.standard_normal((16, d)) / np.sqrt(d)

    def last(M):
        q = ((M @ P.T) ** 2).sum(1)
        return M + ((3.0 * (M @ e)) if shared else 0.5 * (q - 144.0))[:, None] * w  # E q = 16 * 9 for S

    def D(h):
        return 60 * (h * g / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6)) @ w

    folds = []
    for _f in range(2):
        base = 3 * rng.standard_normal((n, d))
        delta = 0.5 * rng.standard_normal((n, d)) + (1.0 * e if shared else 0.0)
        folds.append((base + delta, base))
    return folds, last, D


def _profile_records(folds, last, D, mid, final):
    """Edit records of fold 1's items with fold 0's directions (cross-fitted), at the middle and last layers."""
    (A0, S0), (A1, S1) = folds
    u_mid = _unit((A0 - S0).mean(0))
    c_mid = ((A0 @ u_mid).mean() + (S0 @ u_mid).mean()) / 2
    L0a, L0s = last(A0), last(S0)
    u_last = _unit((L0a - L0s).mean(0))
    c_last = ((L0a @ u_last).mean() + (L0s @ u_last).mean()) / 2
    by = {"aligned": {}, "swapped": {}}
    for goal, M in (("aligned", A1), ("swapped", S1)):
        clean = D(last(M))
        e_mid = D(last(_ablate(M, u_mid, c_mid)))
        e_last = D(_ablate(last(M), u_last, c_last))
        for k in range(len(M)):
            by[goal][f"i{k}"] = {"d": float(clean[k]), "edits": {f"task|{mid}": float(e_mid[k]),
                                                                 f"task_last|{final}": float(e_last[k])}}
    return by, {f"i{k}": f"g{k % 20}" for k in range(len(A1))}


def test_last_layer_ablation_is_tautological_and_the_depth_profile_is_not():
    """Review 2's simulation: with item-specific differences in the middle of the network, ablating the
    mean-difference direction at the last layer removes about all of the mean effect anyway (so the old "maximum
    over layers" would always have found a carrying direction there), while the middle layer removes almost
    nothing; the depth-profile onset, read on profile layers only, is undefined. When a shared direction does
    carry the effect, the onset finds it at the middle layer."""
    rng = np.random.default_rng(0)
    mid, final = 8, 35
    folds, last, D = _world(rng, 400, 256, shared=False)
    by, games = _profile_records(folds, last, D, mid, final)
    ids = sorted(games)
    r_mid = A.removed_share(by, ids, lambda i: f"task|{mid}", games, B=400)
    r_last = A.removed_share(by, ids, lambda i: f"task_last|{final}", games, B=400)
    assert r_mid["effect_before"] > 10  # a real mean effect, made from item-specific differences
    assert r_last["share"] > 0.9 and r_last["ci"][0] > 0.6  # tautological at the last layer
    assert r_mid["ci"][1] < 0.3
    assert A.onset_layer({mid: r_mid}) is None
    assert A.onset_layer({mid: r_mid, final: r_last}) == final  # what including the last layer would have said
    assert final not in A.profile_layers(final + 1)

    folds, last, D = _world(np.random.default_rng(1), 400, 256, shared=True)
    by, games = _profile_records(folds, last, D, mid, final)
    r_mid = A.removed_share(by, ids, lambda i: f"task|{mid}", games, B=400)
    assert r_mid["ci"][0] >= 0.6 and A.onset_layer({mid: r_mid}) == mid


def test_profile_caveat_and_h4_layers():
    assert A.profile_layers(36) == [4, 8, 12, 16, 20, 24, 28] and A.caveat_layers(36) == [32]
    assert A.profile_layers(32) == [4, 8, 12, 16, 20, 24] and A.caveat_layers(32) == [28]
    assert PI.ablate_layers(36) == A.profile_layers(36) and 35 not in PI.ablate_layers(36)
    assert A.h4_layers(A.profile_layers(36), 36) == [4, 8, 12, 16, 20, 24]
    assert A.h4_layers(A.profile_layers(36), 36, arrival_index=17.5) == [4, 8, 12, 16]
    assert A.h4_layers(A.profile_layers(32), 32) == [4, 8, 12, 16, 20]


def test_onset_layer_needs_the_lower_bound_to_stay_above():
    def prof(lows, highs=None):
        return {m: {"share": lo + 0.1, "ci": (lo, (highs or {}).get(m, lo + 0.2))} for m, lo in lows.items()}
    assert A.onset_layer(prof({4: 0.1, 8: 0.65, 12: 0.7, 16: 0.9})) == 8
    assert A.onset_layer(prof({4: 0.7, 8: 0.5, 12: 0.7, 16: 0.9})) == 12  # a dip resets it: "stays"
    assert A.onset_layer(prof({4: 0.7, 8: 0.7, 12: 0.7, 16: 0.55})) is None  # the deepest layer is below
    # the bound, not the point estimate: a point of 0.65 with a lower bound of 0.55 does not count
    assert A.onset_layer({4: {"share": 0.65, "ci": (0.55, 0.75)}, 8: {"share": 0.9, "ci": (0.8, 1.0)}}) == 8
    assert A.onset_layer({4: {"n": 0}, 8: {"share": 0.9, "ci": (0.8, 1.0)}}) == 8
    assert A.onset_versus_arrival(8, 17.5) == "at or before arrival (H3, H4, H5)"
    assert A.onset_versus_arrival(20, 17.5) == "at or before arrival (H3, H4, H5)"  # 17.5 rounds up to 20
    assert A.onset_versus_arrival(24, 17.5) == "after arrival (H1, H2)"
    assert A.onset_versus_arrival(None, 17.5).startswith("no onset")
    assert A.onset_versus_arrival(8, float("nan")) == "arrival undefined"


def test_success_direction_is_estimated_within_verbs():
    """Successes include verbs no failure has; a verb that shifts the residual must not leak into the success
    direction. Within verbs it cannot; the pooled difference of means picks it up."""
    rng = np.random.default_rng(2)
    d = 32
    e = np.eye(d)
    succ_v = ["take"] * 200 + ["examine"] * 200
    fail_v = ["take"] * 400
    shift = {"take": 0.0 * e[3], "examine": 4.0 * e[3]}
    S = np.array([rng.standard_normal(d) + 3.0 * e[1] + shift[v] for v in succ_v])
    F = np.array([rng.standard_normal(d) + shift[v] for v in fail_v])
    within = A.success_direction(S, F, succ_v, fail_v)
    pooled = A.success_direction(S, F)
    assert within @ e[1] > 0.9 and abs(within @ e[3]) < 0.2
    assert abs(pooled @ e[3]) > 0.4  # the verb leaks into the pooled direction
    with pytest.raises(ValueError):
        A.success_direction(S, F, ["go"] * 400, fail_v)
    # verbs weighted n_s n_f / (n_s + n_f): a verb with one success counts for little
    S2 = np.vstack([S[:200], 10 * e[5][None, :]])
    one = A.success_direction(S2, F, ["take"] * 200 + ["move"], fail_v[:399] + ["move"])
    assert abs(one @ e[5]) < 0.2
    assert A.verb({"action": "take mug 1 from cabinet 3"}) == "take"


def test_h4_cosine_is_read_after_projecting_out_the_readout():
    """Task and success directions that share only the readout component look aligned (raw cosine 0.5) but are
    orthogonal once the readout direction is projected out of both; steering along the success direction without
    its readout component adds the aligned-minus-swapped projection on that direction."""
    rng = np.random.default_rng(3)
    d = 16
    e = np.eye(d)
    r = 2.0 * e[0]  # the readout direction, not unit length
    S0 = rng.standard_normal((400, d))
    Al = S0 + 1.0 * e[0] + 1.0 * e[1]  # the task moves along the readout and e1
    succ = rng.standard_normal((300, d)) + 1.0 * e[0] + 1.0 * e[2]  # success along the readout and e2
    fail = rng.standard_normal((300, d))
    x = A.estimate_directions({"aligned": Al, "swapped": S0}, succ, fail, ["take"] * 300, ["take"] * 300,
                              readout_dir=r)
    cos = A.direction_cosines(x, r)
    assert cos["task_success"] > 0.35 and abs(cos["task_success_perp_readout"]) < 0.15
    assert abs(x["success_perp_readout"] @ e[0]) < 1e-12 and x["success_perp_readout"] @ e[2] > 0.95
    assert x["steer_success_perp_readout"] == pytest.approx(((Al - S0) @ x["success_perp_readout"]).mean())
    plus = S0 + 1.0 * e[2]  # naming the object moves along the success-only part: the null baseline is high
    x2 = A.estimate_directions({"aligned": Al, "swapped": S0, "swapped_plus": plus}, succ, fail, readout_dir=r)
    assert A.direction_cosines(x2, r)["mention_success_perp_readout"] > 0.8
    assert abs(A.project_unit(e[0] + e[1], r) @ e[0]) < 1e-12
    # both sides projected: u = e0 + e1 and s = e0 + e1 + e2 give cos(e1, (e1 + e2)/sqrt 2) = 0.707; projecting
    # only one side would give 0.5 or 0.577
    known = {"task": _unit(e[0] + e[1]), "success": _unit(e[0] + e[1] + e[2])}
    assert A.direction_cosines(known, r)["task_success_perp_readout"] == pytest.approx(1 / np.sqrt(2))


def test_answer_projection_removes_readout_and_success():
    rng = np.random.default_rng(5)
    d = 12
    e = np.eye(d)
    succ = rng.standard_normal((50, d)) + 3 * e[4]
    fail = rng.standard_normal((50, d))
    proj = PI.answer_projection(e[0], succ, fail, ["take"] * 50, ["take"] * 50)
    assert len(proj) == 2
    from wf.interp.probes import project_out

    X = rng.standard_normal((20, d)) + 5 * e[4] + 5 * e[0]
    Xp = project_out(X, proj)
    assert np.abs(Xp @ e[0]).max() < 1e-9 and np.abs(Xp @ proj[1]).max() < 1e-9


def _items_for_selection():
    from tests.test_interp import _item

    items = []
    for k in range(200):
        step0 = k % 10 == 0  # 10% at step 0
        mention = not step0 and k % 10 == 1  # 10% name the action's object after the task line
        hist = ("You are in the middle of a room.\nYour task is to: put some mug on countertop."
                + ("" if step0 else "\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a "
                   + ("mug 2." if mention else "plate 1.")))
        items.append(_item(id=f"t{k:03d}", game_id=f"g{k % 8}", history_len=0 if step0 else 2, history_text=hist,
                           determination_reason="hands_full" if k % 3 else "away_from_receptacle"))
    return items


def test_selection_quotas_and_weights():
    from wf.interp import run as R
    from wf.interp.prompts import mentions_action_object

    items = _items_for_selection()
    effect = {t["id"]: 2.0 for t in items}
    games = {f"g{k}" for k in range(8)}
    pick, w = R.select_stratified(items, effect, 50, 1.0, games, "s", min_step0=15, min_mention=12)
    n0 = sum(t["history_len"] == 0 for t in pick)
    nm = sum(t["history_len"] > 0 and mentions_action_object(t) for t in pick)
    assert len(pick) == 50 and n0 == 15 and nm == 12  # proportional would give 5 and 5
    t0 = next(t for t in pick if t["history_len"] == 0)
    assert w[t0["id"]] == pytest.approx(0.10 / (15 / 50))
    tl = next(t for t in pick if t["history_len"] > 0 and not mentions_action_object(t))
    assert w[tl["id"]] == pytest.approx(0.80 / (23 / 50))
    assert sum(w[t["id"]] for t in pick) == pytest.approx(50)  # the weights describe the eligible population
    # without quotas: proportional, every weight about 1
    pick2, w2 = R.select_stratified(items, effect, 50, 1.0, games, "s")
    assert sum(t["history_len"] == 0 for t in pick2) == 5 and set(np.round(list(w2.values()), 6)) == {1.0}
    # a quota larger than its stratum takes all of it
    pick3, _ = R.select_stratified(items, effect, 50, 1.0, games, "s", min_step0=40)
    assert sum(t["history_len"] == 0 for t in pick3) == 20
    assert R.weighting_stratum(items[1], mention=True) == "later_mention"
    assert R.weighting_stratum(items[1], mention=False) == "later" and R.weighting_stratum(items[0]) == "step0"


def test_weights_enter_pooled_estimates():
    """Two strata with different removal (0.9 and 0.1): oversampling the first pulls the unweighted ratio up;
    the weights bring it back to the population mix."""
    games = {f"i{k}": f"g{k % 6}" for k in range(40)}
    by = {"aligned": {}, "swapped": {}}
    for k in range(40):
        after = 0.2 if k < 30 else 1.8  # removal 0.9 on the first 30 items, 0.1 on the last 10
        by["aligned"][f"i{k}"] = {"d": 2.0, "edits": {"t": after}, "edit_clean": {"t": 2.0}}
        by["swapped"][f"i{k}"] = {"d": 0.0, "edits": {"t": 0.0}, "edit_clean": {"t": 0.0}}
    ids = sorted(games)
    plain = A.removed_share(by, ids, lambda i: "t", games, B=50)
    # population: 25% in the first stratum, 75% in the second; sample: 75% and 25%
    w = {f"i{k}": (0.25 / 0.75 if k < 30 else 0.75 / 0.25) for k in range(40)}
    weighted = A.removed_share(by, ids, lambda i: "t", games, B=50, weights=w)
    assert plain["share"] == pytest.approx(0.75 * 0.9 + 0.25 * 0.1)
    assert weighted["share"] == pytest.approx(0.25 * 0.9 + 0.75 * 0.1)
    recs = []
    for k in range(40):
        f = 0.9 if k < 30 else 0.1
        recs.append({"id": f"i{k}", "game_id": f"g{k % 6}", "d_target": 4.0, "d_source": 1.0, "history_len": 3,
                     "cells": {f"resid|{m}|hist_all": 4.0 - 3.0 * f for m in range(3)},
                     "cell_clean": {f"resid|{m}|hist_all": 4.0 for m in range(3)}})
    c = A.group_curve(recs, "hist_all", 3, B=50, weights=w)
    assert c["fraction"][1] == pytest.approx(0.25 * 0.9 + 0.75 * 0.1)


def test_effects_sign_and_strata():
    recs = [{"id": "a", "fmt": "yn", "goal": "aligned", "d": 3.0}, {"id": "a", "fmt": "yn", "goal": "swapped", "d": 1.0},
            {"id": "a", "fmt": "obs", "goal": "aligned", "d": 9.0}, {"id": "a", "fmt": "obs", "goal": "swapped", "d": 1.0}]
    assert PI.effects(recs) == {"a": 2.0} and PI.effects(recs, "obs") == {"a": 8.0}
    assert A.stratum({"history_len": 0}) == "step0" and A.stratum({"history_len": 1}) == "later"
    assert A.stratum({}) == "step0" and A.in_stratum({"history_len": 1}, "all")


def test_gate_median_check_hashes_and_first_token():
    rng = np.random.default_rng(1)
    ids = [f"t{k}" for k in range(300)]
    logged = {}
    for i in ids:
        base = rng.normal(0, 10)
        logged[(i, "yn_aligned")], logged[(i, "yn_swapped")], logged[(i, "yn_absent")] = base + 3, base, base + 1
    # every prompt off by 0.5 in alternating directions: the mean effect and the correlations survive, the
    # median absolute difference does not
    off = {k: v + (0.5 if (hash(k[0]) + len(k[1])) % 2 else -0.5) for k, v in logged.items()}
    off = {k: v + (0.5 if n % 2 else -0.5) for n, (k, v) in enumerate(sorted(logged.items()))}
    g = A.reproduction_gate(_gate_records(ids, off), logged, "qwen3-8b-fewshot")
    assert g["checks"]["r_prompt"] and g["checks"]["coverage"] and g["median_abs_diff"] == pytest.approx(0.5)
    assert not g["checks"]["median_abs"] and not g["passed"]
    same = {i: "h" for i in ids}
    ok = A.reproduction_gate(_gate_records(ids, logged), logged, "qwen3-8b-fewshot", "yn", same, dict(same))
    assert ok["passed"] and ok["checks"]["item_hash"] and ok["item_hash_mismatches"] == 0
    changed = {**same, "t7": "other"}
    bad = A.reproduction_gate(_gate_records(ids, logged), logged, "qwen3-8b-fewshot", "yn", same, changed)
    assert not bad["passed"] and bad["item_hash_mismatches"] == 1 and not bad["checks"]["item_hash"]
    # first-token: plain arms, L_first; the yes/no log does not stand in for it
    first = {(i, a): logged[(i, f"yn_{a}")] for i in ids for a in ("aligned", "swapped")}
    recs = [{"id": i, "fmt": "obs", "goal": a, "d": first[(i, a)]} for i in ids for a in ("aligned", "swapped")]
    fo = A.reproduction_gate(recs, first, "qwen3-8b-fewshot", "obs")
    assert fo["passed"] and fo["n_items"] == 300 and fo["logged_effect_published"] == 2.52
    assert A.reproduction_gate(recs, logged, "qwen3-8b-fewshot", "obs")["n_items"] == 0
    with pytest.raises(ValueError):
        A.reproduction_gate(recs, first, "qwen3-8b-fewshot", "xyz")


def _noisy_two_layer_recs(n=60, games=12, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for k in range(n):
        g = k % games
        f = [0.0, 0.4 + 0.3 * np.sin(g), 0.4 + 0.3 * np.cos(g), 0.0]  # two layers trade places across games
        cells = {f"resid|{m}|hist_all": 4.0 - 3.0 * f[m] + 0.05 * rng.standard_normal() for m in range(4)}
        recs.append({"id": f"i{k}", "game_id": f"g{g}", "d_target": 4.0, "d_source": 1.0, "history_len": 3,
                     "cells": cells, "cell_clean": {c: 4.0 for c in cells}})
    return recs


def test_maximum_over_layers_rechooses_the_layer_in_every_resample():
    c = A.group_curve(_noisy_two_layer_recs(), "hist_all", 4, B=400, seed=2)
    j = c["argmax_layer"]
    # the maximum is at least every layer's value in every resample, so its interval sits above any one layer's
    assert c["max_ci"][0] >= max(lo for lo, _hi in c["ci"]) - 1e-12
    assert c["max_ci"][0] > c["ci"][j][0] + 0.01


def test_placebo_floor_bonferroni_and_early_span():
    recs = []
    for k in range(30):
        s = 1 if k % 2 else -1
        recs.append({"id": f"p{k}", "game_id": f"g{k % 6}", "d_target": 1.0, "d_source": 1.0, "history_len": 2,
                     "cells": {f"resid|{m}|hist_all": 1.0 - s * 0.4 for m in range(3)},
                     "cell_clean": {f"resid|{m}|hist_all": 1.0 for m in range(3)}})
    fl = A.placebo_floor(recs, "hist_all", 3, B=100)
    assert fl["mean_abs_nats"][0] == pytest.approx(0.4)  # absolute, so +-0.4 does not cancel to 0
    # the goal pairs' removed nats interval straddles the floor: not above it (lower bound against the floor)
    goal = []
    for k in range(30):
        x = 0.4 + (0.3 if k % 2 else -0.3)
        goal.append({"id": f"q{k}", "game_id": f"g{k % 6}", "d_target": 4.0, "d_source": 1.0, "history_len": 2,
                     "cells": {f"resid|{m}|hist_all": 4.0 - x for m in range(3)},
                     "cell_clean": {f"resid|{m}|hist_all": 4.0 for m in range(3)}})
    s = A.patch_summary(goal, recs, 3, B=100)
    lo, hi = s["curves"]["hist_all"]["removed_nats_ci_bonferroni"][0]
    assert lo < fl["upper_bonferroni"][0] < hi and s["above_placebo_floor"]["hist_all"] == [False] * 3
    # Bonferroni intervals are normal intervals from the bootstrap SE, wider than the pointwise ones
    c = A.group_curve(_noisy_two_layer_recs(), "hist_all", 4, B=400, seed=2)
    from statistics import NormalDist

    (blo, bhi), (plo, phi) = c["ci_bonferroni"][1], c["ci"][1]
    assert blo < plo and bhi > phi
    z = NormalDist().inv_cdf(1 - 0.05 / 4 / 2)
    assert (bhi - blo) / (phi - plo) > 1.15  # z = 2.50 against 1.96
    assert (bhi - blo) / 2 == pytest.approx(z * (phi - plo) / (2 * 1.96), rel=0.25)
    lo_, hi_ = A._interval_normal(1.0, np.random.default_rng(0).standard_normal(20000), 0.05 / 36)
    assert (hi_ - lo_) / 2 == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 72), rel=0.02)
    assert c["fraction"][1] == pytest.approx((blo + bhi) / 2)
    # early span: span fraction at the end of the first third minus at the end of the second third
    span = []
    for k in range(10):
        f = [1.0, 1.0, 0.8, 0.5, 0.3, 0.2]
        span.append({"id": f"s{k}", "game_id": f"g{k % 5}", "d_target": 4.0, "d_source": 1.0, "history_len": 2,
                     "cells": {f"resid|{m}|span": 4.0 - 3.0 * f[m] for m in range(6)},
                     "cell_clean": {f"resid|{m}|span": 4.0 for m in range(6)}})
    assert A.patch_summary(span, [], 6, B=20)["early_span_share"] == pytest.approx(1.0 - 0.5)


def _report(hs, arrival=0.5, onset=8, profile=None):
    layers = [4, 8, 12, 16, 20, 24, 28]
    prof = profile or {m: {"share": 0.7, "ci": (0.62, 0.8), "n": 50, "se": 0.05} for m in layers}
    by = {m: {"task_perp_success": {"share": 0.35}, "steer_success_perp_readout": {"ratio": 0.6},
              "steer_success": {"ratio": 0.9}} for m in layers}
    cos = {f"{f}|{m}": {"task_success_perp_readout": 0.4, "mention_success_perp_readout": 0.05,
                        "task_success": 0.7} for f in (0, 1) for m in layers}
    cos.update({f"between_folds|{m}": 0.8 for m in layers})
    noise = {"history_share": hs, "arrival_layer": arrival, "n_items": 60, "action_share": {"se": 0.04}}
    ab = {"onset_layer": onset, "profile": prof, "by_layer": by, "task": {"share": 0.7}, "gap_neutral": {"ratio": 0.95}}
    return {"patching": {"noise": {s: noise for s in ("step0", "later", "all")}},
            "ablation": {"layers": layers, **{s: ab for s in ("step0", "later", "all")}}, "cosines": cos}


def test_verdict_rows_read_interval_bounds_onset_and_h4():
    v = PI.verdict_rows(_report({"defined": True, "max": 0.35, "ci": (0.2, 0.5)}), 36)
    assert v["all"]["history_share"] == "between"  # the point max is above 0.3, the lower bound is not
    v = PI.verdict_rows(_report({"defined": True, "max": 0.1, "ci": (0.05, 0.2)}), 36)
    assert v["all"]["history_share"] == "between"
    v = PI.verdict_rows(_report({"defined": True, "max": 0.4, "ci": (0.31, 0.5)}), 36)
    assert v["all"]["history_share"] == "high (>=0.3)"
    assert v["all"]["arrival_layer_index"] == pytest.approx(17.5)
    assert v["all"]["onset_versus_arrival"] == "at or before arrival (H3, H4, H5)"
    h4 = v["all"]["h4"]
    assert h4["layers"] == [4, 8, 12, 16]  # up to two thirds of depth and before the arrival layer
    assert h4["cos_task_success_perp_readout"] == pytest.approx(0.4)
    assert h4["null_mention_success_perp_readout"] == pytest.approx(0.05)
    assert h4["null_between_folds_task"] == pytest.approx(0.8)
    assert h4["steer_success_perp_readout_share"] == pytest.approx(0.6)
    assert h4["perp_success_over_task_removal"] == pytest.approx(0.5)
    late = PI.verdict_rows(_report({"defined": False}, onset=24), 36)
    assert late["all"]["onset_versus_arrival"] == "after arrival (H1, H2)" and late["all"]["history_share"] is None
    ss = PI.sample_sizes(_report({"defined": True, "max": 0.4, "ci": (0.31, 0.5), "se": 0.06}))
    assert ss["all"]["history_share"]["n_needed"] == A.n_for_half_width(60, 0.06)
    assert ss["step0"]["removal_by_layer"][8]["n_needed"] == A.n_for_half_width(50, 0.05)


def test_sample_size_for_a_half_width():
    # SE 0.06 at 60 items: a half-width of 0.075 needs 60 * (1.96 * 0.06 / 0.075)^2 = 147.5..., so 148
    assert A.n_for_half_width(60, 0.06) == 148
    assert A.n_for_half_width(60, 0.03) == 37 and A.n_for_half_width(0, 0.1) is None
    assert A.n_for_half_width(60, float("nan")) is None


def _ablate_recs(best_by_fold, n=12):
    """Edit records of two folds' failures, where the task direction removes most at layer best_by_fold[f] on
    fold f's items; neutral prompts of successes and failures; steering moves swapped prompts."""
    recs = []
    for f in (0, 1):
        for k in range(n):
            i = f"f{f}_{k}"
            game = f"g{f}_{k % 4}"
            hl = 0 if k < 3 else 2
            edits_a = {f"task|{m}": (0.5 if m == best_by_fold[f] else 2.5) for m in (4, 8)}
            edits_a.update({f"steer_success|{m}": 4.0 for m in (4, 8)})
            edits_s = {f"task|{m}": 0.0 for m in (4, 8)}
            edits_s.update({f"steer_success|{m}": 1.0 + m / 8 for m in (4, 8)})  # +1.5 at 4, +2.0 at 8 on swapped
            edits_n = {f"task|{m}": -2.5 for m in (4, 8)}
            edits_a = {x: v + 1.0 for x, v in edits_a.items()}  # aligned 4, swapped 1: effect 3
            edits_s = {x: v + 1.0 for x, v in edits_s.items()}
            for goal, d, ed in (("aligned", 4.0, edits_a), ("swapped", 1.0, edits_s), ("absent", -2.0, edits_n)):
                recs.append({"id": i, "game_id": game, "goal": goal, "d": d, "edits": ed,
                             "edit_clean": {x: d for x in ed}, "history_len": hl})
        for k in range(4):
            i = f"s{f}_{k}"
            for goal, d in (("aligned", 5.0), ("swapped", 4.0), ("absent", 4.0)):
                ed = {f"task|{m}": d - (0.5 if goal == "aligned" else 0.0) for m in (4, 8)}
                ed.update({f"steer_success|{m}": d for m in (4, 8)})
                if goal == "absent":
                    ed = {f"task|{m}": 3.0 for m in (4, 8)}
                recs.append({"id": i, "game_id": f"g{f}_{k}", "goal": goal, "d": d, "edits": ed,
                             "edit_clean": {x: d for x in ed}, "history_len": 2})
    fold_of = {r["id"]: int(r["id"][1]) for r in recs}
    return recs, fold_of, {r["id"] for r in recs if r["id"].startswith("s")}


def test_ablation_summary_chooses_each_folds_layer_on_the_other_fold():
    recs, fold_of, succ = _ablate_recs({0: 4, 1: 8})
    out = A.ablation_summary(recs, fold_of, succ, [4, 8], B=50)
    # fold 0's items use the layer that removed most on fold 1's items (8), and the reverse
    assert out["chosen_layer_by_fold"] == {0: 8, 1: 4}
    assert out["layer_choice"][0].startswith("cross-fitted")
    # every item is then read at a layer that is not its own best: removal (3 - 2.5) / 3 on all of them
    assert out["all"]["task"]["share"] == pytest.approx(1 - 2.5 / 3)
    # profile per layer: half the items remove 2.5/3 there, half 0.5/3
    assert out["all"]["profile"][4]["share"] == pytest.approx(1 - (12 * 0.5 + 12 * 2.5) / (24 * 3))
    # the neutral gap is read on neutral prompts (successes 4, failures -2: 6 clean; 3 and -2.5 edited: 5.5)
    assert out["all"]["gap_neutral"]["clean"]["gap"] == pytest.approx(6.0)
    assert out["all"]["gap_neutral"]["ratio"] == pytest.approx(5.5 / 6.0)
    assert out["step0"]["gap_neutral"]["defined"] is False and out["step0"]["task_on_successes"]["defined"] is False
    # steering is read on swapped prompts: fold 0 at layer 8 (+2.0), fold 1 at layer 4 (+1.5); effect 3
    assert out["later"]["steer_success_gain_on_swapped"]["mean"] == pytest.approx((2.0 + 1.5) / 2)
    assert out["all"]["steer_success_ratio"]["ratio"] == pytest.approx((2.0 + 1.5) / 2 / 3.0)
    assert out["all"]["by_layer"][8]["steer_success"]["ratio"] == pytest.approx(2.0 / 3.0)
    # only one fold edited: the choice is in-sample and flagged
    one = [r for r in recs if fold_of[r["id"]] == 0]
    o1 = A.ablation_summary(one, {i: f for i, f in fold_of.items() if f == 0}, succ, [4, 8], B=50)
    assert o1["chosen_layer_by_fold"] == {0: 4} and o1["layer_choice"][0].startswith("in-sample")


def test_ablation_summary_reports_caveat_and_last_layers_apart():
    recs, fold_of, succ = _ablate_recs({0: 4, 1: 4})
    for r in recs:
        r["edits"]["task|32"] = 0.0 if r["goal"] != "absent" else r["d"]
        r["edits"]["task_last|35"] = 0.0 if r["goal"] != "absent" else r["d"]
        r["edit_clean"].update({"task|32": r["d"], "task_last|35": r["d"]})
    out = A.ablation_summary(recs, fold_of, succ, [4, 8], caveat=[32], last=35, B=50)
    assert out["all"]["task_last"]["share"] == pytest.approx(1.0) and out["all"]["profile"][32]["share"] == 1.0
    assert out["all"]["onset_layer"] is None  # layers 4 and 8 do not reach it; 32 and 35 are not in the profile
    assert set(out["chosen_layer_by_fold"].values()) <= {4, 8}


def test_ablation_edits_layers_and_targets():
    import torch

    rng = np.random.default_rng(7)
    d = 8

    def dirs_at():
        u = _unit(rng.standard_normal(d))
        return {"task": u, "task_mid": 0.3, "task_perp_success": _unit(rng.standard_normal(d)),
                "task_perp_success_mid": -0.2, "success": _unit(rng.standard_normal(d)), "steer_success": 1.5,
                "success_perp_readout": _unit(rng.standard_normal(d)), "steer_success_perp_readout": 0.7,
                "random": [_unit(rng.standard_normal(d))], "random_mid": np.array([2.5])}
    dirs = {m: dirs_at() for m in (4, 8, 12, 32, 35)}
    checks: dict = {}
    ed = PI.ablation_edits(dirs, [4, 8, 12], caveat=[32], last=35, checks=checks)
    assert [m for m, _f in ed["task_from|8"]] == [8, 12]  # from that layer on, profile layers only
    assert [m for m, _f in ed["task_from|4"]] == [4, 8, 12]
    assert "task|32" in ed and "task_perp_success|32" not in ed and "task_last|35" in ed and "task|35" not in ed
    h = torch.tensor(rng.standard_normal((1, 3, d)), dtype=torch.float32)
    out = ed["random|4|0"][0][1]([2])(h)  # the random direction is set to its midpoint, not to 0
    assert float(out[0, 2] @ torch.tensor(dirs[4]["random"][0], dtype=torch.float32)) == pytest.approx(2.5, abs=1e-5)
    out = ed["steer_success_perp_readout|8"][0][1]([2])(h)
    added = (out - h)[0, 2].numpy()
    assert np.allclose(added, 0.7 * dirs[8]["success_perp_readout"], atol=1e-5)
    assert checks["task|4"][0][2] == 0.3 and checks["steer_success|8"][0][3] == "add"


class _FakeStore:
    def __init__(self, rows):
        self.rows = rows  # {key: (layers, width) array}

    def records(self):
        return [{"key": k, "id": k.split("|")[0], "goal": k.split("|")[2], "fmt": "yn"} for k in self.rows]

    def load(self, name, keys, sel=None):
        return [{"key": k} for k in keys], np.stack([self.rows[k][sel] for k in keys])


def test_fold_directions_use_each_folds_own_successes():
    from tests.test_interp import _item

    rng = np.random.default_rng(9)
    d = 16
    e = np.eye(d)
    fails, succs, rows = [], [], {}
    for f in (0, 1):
        for k in range(30):
            t = _item(id=f"F{f}{k}", game_id=f"G{f}", history_len=2)
            fails.append(t)
            base = rng.standard_normal(d)
            rows[f"{t['id']}|yn|aligned"] = np.stack([base + e[0]] * 16)
            rows[f"{t['id']}|yn|swapped"] = np.stack([base] * 16)
            rows[f"{t['id']}|yn|absent"] = np.stack([base] * 16)
            s = _item(id=f"S{f}{k}", game_id=f"G{f}", history_len=2)
            succs.append(s)
            rows[f"{s['id']}|yn|absent"] = np.stack([rng.standard_normal(d) + 3 * e[1 + f]] * 16)
            z = _item(id=f"Z{f}{k}", game_id=f"G{f}", history_len=0)  # step-0 failures sit apart along e6
            fails.append(z)
            zb = rng.standard_normal(d)
            rows[f"{z['id']}|yn|aligned"] = np.stack([zb + e[0]] * 16)
            rows[f"{z['id']}|yn|swapped"] = np.stack([zb] * 16)
            rows[f"{z['id']}|yn|absent"] = np.stack([zb + 4 * e[6]] * 16)
    out = PI.fold_directions({}, _FakeStore(rows), {"failures": fails, "successes": succs}, {"G0": 0, "G1": 1},
                             [8, 12], lens_dir=e[5])
    assert out[0][8]["success"] @ e[1] > 0.8 and out[1][8]["success"] @ e[2] > 0.8  # each fold's own
    assert abs(out[0][8]["success"] @ e[6]) < 0.2  # no successes at step 0, so no step-0 failures in it
    assert out[0][8]["task"] @ e[0] > 0.99 and abs(out[0][12]["success_perp_readout"] @ e[5]) < 1e-9


def test_gpu_check_refuses_a_different_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "interp_pilot_stage0").mkdir()
    (tmp_path / "interp_pilot_stage0" / "jobs.jsonl").write_text(json.dumps({"gpu": ["NVIDIA A100-SXM4-80GB"]}) + "\n")
    monkeypatch.setattr(PI, "_gpu_names", lambda: ["NVIDIA L40S"])
    assert "not NVIDIA A100" in PI.check_gpu("interp_pilot_patch", {})
    monkeypatch.setattr(PI, "_gpu_names", lambda: None)
    assert PI.check_gpu("interp_pilot_patch", {}) is not None  # cannot read its own GPU
    monkeypatch.setattr(PI, "_gpu_names", lambda: ["NVIDIA A100-SXM4-80GB"])
    assert PI.check_gpu("interp_pilot_patch", {}) is None
    # a resumed stage also matches its own earlier jobs; two GPU types on record are refused outright
    (tmp_path / "interp_pilot_patch").mkdir()
    (tmp_path / "interp_pilot_patch" / "jobs.jsonl").write_text(json.dumps({"gpu": ["NVIDIA L40S"]}) + "\n")
    assert PI.check_gpu("interp_pilot_patch", {"resume": True}) is not None
    assert PI.check_gpu("interp_pilot_patch", {}) is None
    # stage 0 on its own: nothing on record yet, any GPU may start it
    assert PI.check_gpu("interp_pilot_stage0", {}) is None
    monkeypatch.setattr(PI, "_gpu_names", lambda: ["NVIDIA L40S"])
    assert PI.check_gpu("interp_pilot_stage0", {"resume": True}) is not None
    assert PI._refuse_gpu("interp_pilot_stage0", {"resume": True, "run_id": "interp_pilot_stage0"})
    assert "refused" in (tmp_path / "interp_pilot_stage0" / "STATUS.md").read_text()


# ---------------- the runs end to end on a tiny model ----------------

OBJS = ["mug", "plate", "bowl", "apple", "cup", "pen"]
ROOM = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 3, "
        "a shelf 1, and a countertop 1.\n")


def _fail(k):
    gi = k % 8
    g = OBJS[gi % 6]
    sw = OBJS[(gi + 1 + (k // 8)) % 6]
    task = f"Your task is to: put some {g} on countertop."
    step0 = k % 3 == 0
    hist = ROOM + task + ("" if step0 else f"\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a {g} 2, "
                                          f"and a {sw} 1.\n> go to cabinet 1\nYou arrive at cabinet 1. The cabinet 1 is closed.")
    return _item(id=f"f{k:03d}", game_id=f"g{gi}", goal_object=g, swapped_goal_object=sw, task_line=task,
                 swapped_task_line=task.replace(g, sw), history_text=hist, history_len=0 if step0 else 2,
                 action=f"take {g} 1 from cabinet 3", determination_reason="away_from_receptacle",
                 candidates={"o_plus": f"You pick up the {g} 1 from the cabinet 3.", "o_minus": "Nothing happens."},
                 scene_locations={f"{g} 1": ["cabinet 3"] if k % 2 else ["shelf 1"],
                                  **{f"{o} 1": ["shelf 1"] for o in OBJS if o != g}})


def _succ(k):
    t = _fail(k)
    return {**t, "id": f"s{k:03d}", "action": f"take {t['goal_object']} 2 from shelf 1", "determination_reason": None,
            "history_text": ROOM + t["task_line"] + "\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a "
            f"{t['goal_object']} 2, and a {t['swapped_goal_object']} 1.", "history_len": 1}


FAILS = [_fail(k) for k in range(24)]
SUCCS = [_succ(k) for k in range(12)]


@pytest.fixture(scope="module")
def pilot_tok():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    texts = []
    for t in FAILS + SUCCS:
        for goal in ("aligned", "swapped", "absent", "swapped_plus"):
            texts += [m["content"] for m in PR.messages(t, "yn", goal)]
        for goal in ("aligned", "swapped"):
            texts += [m["content"] for m in PR.messages(t, "obs", goal)]
    words = sorted(set(re.findall(SPLIT, "\n".join(texts) + " system user assistant Yes No You Nothing "
                                  + " ".join(OBJS))))
    vocab = {w: i for i, w in enumerate(SPECIALS + words)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(SPLIT), behavior="isolated")
    tk = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>", additional_special_tokens=SPECIALS[:2])
    tk.chat_template = TEMPLATE
    return tk


@pytest.fixture
def pilot_env(pilot_tok, monkeypatch, tmp_path):
    from wf.interp.model import parts

    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    q = parts(_qwen(len(pilot_tok), layers=4))
    monkeypatch.setattr(PI, "load_model", lambda cfg: (q, pilot_tok))
    monkeypatch.setattr(PI, "load_sets", lambda cfg: {"failures": FAILS, "successes": SUCCS})
    monkeypatch.setattr(PI, "all_items", lambda cfg: FAILS + SUCCS)
    logged = _logged(q, pilot_tok)
    monkeypatch.setattr(PI, "logged_scores", lambda cfg: {k: dict(v) for k, v in logged.items()})
    return {"root": tmp_path, "logged": logged, "q": q, "tok": pilot_tok}


_LOGGED: dict = {}


def _logged(q, tok):
    """What rq1v_score would have logged for the tiny model: yes/no L, first-token L_first, item hashes."""
    if _LOGGED:
        return _LOGGED
    from wf.interp.hooks import forward_last
    from wf.runs import _item_hash

    out = {"yn": {}, "obs": {}, "item_hash": {t["id"]: _item_hash(t) for t in FAILS + SUCCS}}
    for fmt, items, goals in (("yn", FAILS + SUCCS, ("aligned", "swapped", "absent")),
                              ("obs", FAILS, ("aligned", "swapped"))):
        plus, minus = PR.readout_ids(tok, fmt)
        for t in items:
            for goal in goals:
                e = PR.encode(tok, PR.messages(t, fmt, goal))
                _, h = forward_last(q, torch.tensor([e.ids]))
                out[fmt][(t["id"], f"yn_{goal}" if fmt == "yn" else goal)] = float(q.readout(h, plus, minus)[0])
    _LOGGED.update(out)
    return _LOGGED


def pilot_tok_of(env):
    return env["tok"]


BASE = {"skip_source_check": True, "min_free_gb": 0, "batch_size": 8}


def test_stage0_gate_fails_on_a_shifted_log(pilot_env, monkeypatch):
    lg = pilot_env["logged"]
    shifted = {**lg, "yn": {k: v + (0.6 if k[1] == "yn_aligned" else 0.0) for k, v in lg["yn"].items()}}
    monkeypatch.setattr(PI, "logged_scores", lambda cfg: shifted)
    assert PI.run_interp_pilot_stage0({**BASE, "run_id": "interp_pilot_stage0"}) == 1
    status = (pilot_env["root"] / "interp_pilot_stage0" / "STATUS.md").read_text()
    assert "FAIL" in status.splitlines()[0] and "yn gate FAILED" in status
    # the first-token gate alone failing also stops stage 0
    first = {**lg, "obs": {k: v + (0.6 if k[1] == "aligned" else 0.0) for k, v in lg["obs"].items()}}
    monkeypatch.setattr(PI, "logged_scores", lambda cfg: first)
    assert PI.run_interp_pilot_stage0({**BASE, "run_id": "interp_pilot_stage0", "resume": True}) == 1
    status = (pilot_env["root"] / "interp_pilot_stage0" / "STATUS.md").read_text()
    assert "yn gate passed" in status and "obs gate FAILED" in status


def test_gpu_runs_refuse_before_any_work(tmp_path, monkeypatch):
    import wf.runs

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"):
        monkeypatch.setenv(key, "1")
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    (tmp_path / "interp_pilot_stage0").mkdir()
    (tmp_path / "interp_pilot_stage0" / "jobs.jsonl").write_text(json.dumps({"gpu": ["NVIDIA A100-SXM4-80GB"]}) + "\n")
    monkeypatch.setattr(wf.runs, "require_passed", lambda *a, **k: None)
    monkeypatch.setattr(PI, "_gpu_names", lambda: ["NVIDIA L40S"])
    monkeypatch.setattr(PI, "load_sets", lambda cfg: pytest.fail("work started on the wrong GPU"))
    monkeypatch.setattr(PI, "logged_scores", lambda cfg: pytest.fail("work started on the wrong GPU"))
    for run, fn in (("interp_pilot_patch", PI.run_interp_pilot_patch), ("interp_pilot_ablate", PI.run_interp_pilot_ablate),
                    ("interp_pilot_stage0", PI.run_interp_pilot_stage0)):
        assert fn({**BASE, "run_id": run, "resume": True}) == 1
        assert "refused" in (tmp_path / run / "STATUS.md").read_text()


def test_minimal_pilot_end_to_end(pilot_env):
    root = pilot_env["root"]
    assert PI.run_interp_pilot_stage0({**BASE, "run_id": "interp_pilot_stage0"}) == 0
    m0 = json.loads((root / "interp_pilot_stage0" / "metrics.json").read_text())
    assert m0["gate"]["passed"] and m0["gate"]["median_abs_diff"] < 1e-4
    # the first-token readout is gated too, against L_first, and every item hash was checked
    assert m0["gate_first_token"]["passed"] and m0["gate_first_token"]["n_items"] == len(FAILS)
    assert m0["gate"]["item_hash_mismatches"] == 0 and m0["gate"]["checks"]["item_hash"]
    man = json.loads((root / "interp_pilot_stage0" / "manifest.json").read_text())
    assert man["n_layers"] == 4 and man["batch_noise_nats"] < 1e-4 and "model_revision" in man["logged_scores_note"]
    recs = PI._stage0_store({}).records()
    assert {r["goal"] for r in recs} == {"aligned", "swapped", "absent", "swapped_plus"}
    assert not any(r["goal"] == "swapped_plus" and r["id"].startswith("s") for r in recs)
    obs = [r for r in recs if r["fmt"] == "obs"]
    assert len(obs) == 2 * len(FAILS) and {r["goal"] for r in obs} == {"aligned", "swapped"}

    cfg_p = {**BASE, "run_id": "interp_pilot_patch", "n_items": 6, "n_placebo": 2, "min_effect": -1e9,
             "batch_size": 16, "min_step0": 3, "min_mention": 0}
    assert PI.run_interp_pilot_patch(cfg_p) == 0
    pm = json.loads((root / "interp_pilot_patch" / "manifest.json").read_text())
    fold = PI.folds(FAILS + SUCCS, 0)
    assert len(pm["items"]) == 6 and all(fold[t["game_id"]] == 0 for t in FAILS if t["id"] in pm["items"])
    chosen = [t for t in FAILS if t["id"] in pm["items"]]
    assert sum(t["history_len"] == 0 for t in chosen) == 3 == pm["strata"]["step0"]  # the step-0 quota
    pop = [t for t in FAILS if fold[t["game_id"]] == 0]
    share0 = sum(t["history_len"] == 0 for t in pop) / len(pop)
    w0 = {pm["weights"][t["id"]] for t in chosen if t["history_len"] == 0}
    assert len(w0) == 1 and w0.pop() == pytest.approx(share0 / 0.5)  # population share over sample share
    assert 1 <= len(pm["placebo"]) <= 2 and pm["n_cells"] == 4 * len(PR.PILOT_GROUPS)
    patch = PI.read_jsonl(root / "interp_pilot_patch" / "patch.jsonl")
    assert len(patch) == 2 * 6 + 2 * len(pm["placebo"])
    for r in patch:
        if r["source_goal"] == "swapped" and r["direction"] == "noise":
            assert r["cells"]["resid|3|last"] == pytest.approx(r["d_source"], abs=1e-4)  # the identity holds

    cfg_a = {**BASE, "run_id": "interp_pilot_ablate", "layers": [1, 2], "n_failures_per_fold": 4,
             "min_step0_per_fold": 2, "n_successes_per_fold": 2, "batch_size": 16}
    assert PI.run_interp_pilot_ablate(cfg_a) == 0
    am = json.loads((root / "interp_pilot_ablate" / "manifest.json").read_text())
    assert list(am["items"]) == ["0"] and am["edit_folds"] == [0]  # the calibration fold only
    for f, ids in am["items"].items():  # every item is edited with directions from the other fold's games
        assert all(fold[t["game_id"]] == int(f) for t in FAILS + SUCCS if t["id"] in ids)
    ab = PI.read_jsonl(root / "interp_pilot_ablate" / "ablate.jsonl")
    assert {r["id"] for r in ab} == set(am["items"]["0"])
    names = set(ab[0]["edits"])
    assert {"task|2", "task_perp_success|2", "mention|2", "steer_success|2", "steer_success_perp_readout|1",
            "task_from|1", "task_last|3"} <= names
    assert not any(n.startswith("task|3") or n.startswith("task_from|3") for n in names)
    assert any(n.startswith("random|") for n in names) and all(r["positions"] == "last" for r in ab)
    dz = np.load(root / "interp_pilot_ablate" / "directions.npz")
    assert dz["0|2|task"].shape == (64,) and abs(float(dz["0|2|task_perp_success"] @ dz["0|2|success"])) < 1e-9
    lens_dir = np.load(root / "interp_pilot_stage0" / "lens_dir.npy")
    assert abs(float(dz["1|2|success_perp_readout"] @ lens_dir)) < 1e-6
    # the between-fold cosine is of the two folds' task directions (a null baseline for every cosine with them)
    assert am["cosines"]["between_folds|2"] == pytest.approx(float(dz["0|2|task"] @ dz["1|2|task"]))
    assert am["cosines"]["between_folds_success|2"] == pytest.approx(float(dz["0|2|success"] @ dz["1|2|success"]))
    assert am["cosines"]["cross_fold_task_success_perp_readout|0|2"] == pytest.approx(
        float(A.project_unit(dz["0|2|task"], lens_dir) @ A.project_unit(dz["1|2|success"], lens_dir)))
    chosen_a = [t for t in FAILS if t["id"] in am["items"]["0"]]
    assert len(chosen_a) == 4 and sum(t["history_len"] == 0 for t in chosen_a) == 2  # the per-fold step-0 quota
    # the realized edit was logged on the first prompt: a mean ablation lands on its target
    chk = json.loads((root / "interp_pilot_ablate" / "edit_checks.json").read_text())
    real = chk["realized"]["0"]
    assert abs(real["task|2"][0]) < 1e-3 and abs(real["steer_success|1"][0]) < 1e-3 and chk["max_abs"] < 1e-3
    # cross-fitting: an item of fold 0 was edited with the direction estimated on fold 1's games
    from wf.interp.hooks import direction_edit, readout_rows

    rec = next(r for r in ab if r["goal"] == "aligned" and r["id"] in am["items"]["0"])
    t = next(x for x in FAILS + SUCCS if x["id"] == rec["id"])
    e = PR.encode(pilot_tok_of(pilot_env), PR.messages(t, "yn", "aligned"))
    plus, minus = PR.readout_ids(pilot_tok_of(pilot_env), "yn")
    for f, want in ((1, True), (0, False)):
        u = torch.tensor(dz[f"{f}|2|task"], dtype=torch.float32)
        got = readout_rows(pilot_env["q"], torch.tensor([e.ids] * 16), plus, minus,
                           edits=[("resid", 2, direction_edit(u, [len(e) - 1], value=float(dz[f"{f}|2|task_mid"])))])[0]
        assert (got == pytest.approx(rec["edits"]["task|2"], abs=1e-4)) is want

    cfg_an = {"run_id": "interp_pilot_analysis", "bootstrap": 50, "hidden_probe": True}
    assert PI.run_interp_pilot_analysis(cfg_an) == 0
    rep = json.loads((root / "interp_pilot_analysis" / "metrics.json").read_text())
    assert set(rep["patching"]["noise"]) - {"weights_by_stratum"} == {"step0", "later", "all"}
    assert rep["patching"]["noise"]["step0"]["history_share"]["defined"] is False
    assert rep["patching"]["noise"]["later"]["history_share"]["defined"] is True
    assert rep["ablation"]["chosen_layer_by_fold"]["0"] in (1, 2)
    assert rep["ablation"]["layer_choice"]["0"].startswith("in-sample")
    assert "task" in rep["ablation"]["all"] and "gap_neutral" in rep["ablation"]["all"]
    assert set(rep["ablation"]["all"]["profile"]) == {"1", "2"} and "task_last" in rep["ablation"]["all"]
    assert rep["ablation"]["step0"]["gap_neutral"]["defined"] is False  # no determined success at step 0
    assert rep["verdicts"]["step0"]["history_share"] is None
    assert "onset_versus_arrival" in rep["verdicts"]["all"] and "h4" in rep["verdicts"]["all"]
    assert "removal_by_layer" in rep["sample_size"]["all"]
    # pooled numbers are weighted back to the population (the step-0 quota oversamples step 0)
    from wf import constants as C

    goal = [r for r in patch if r["direction"] == "noise" and r["source_goal"] == "swapped"]
    plac = [r for r in patch if r["direction"] == "noise" and r["source_goal"] == "placebo"]
    w = A.patch_summary(goal, plac, 4, 50, C.RNG_SEED, weights=pm["weights"])["curves"]["final"]["fraction"]
    u = A.patch_summary(goal, plac, 4, 50, C.RNG_SEED)["curves"]["final"]["fraction"]
    got = rep["patching"]["noise"]["all"]["curves"]["final"]["fraction"]
    assert got == pytest.approx(w) and got != pytest.approx(u, abs=1e-6)
    fold_of = {i: 0 for i in am["items"]["0"]}
    succ_ids = {t["id"] for t in SUCCS}
    sw = A.ablation_summary(ab, fold_of, succ_ids, [1, 2], [], 3, am["weights"], 50, C.RNG_SEED)
    su = A.ablation_summary(ab, fold_of, succ_ids, [1, 2], [], 3, None, 50, C.RNG_SEED)
    assert rep["ablation"]["all"]["profile"]["1"]["share"] == pytest.approx(sw["all"]["profile"][1]["share"])
    assert sw["all"]["profile"][1]["share"] != pytest.approx(su["all"]["profile"][1]["share"], abs=1e-9)
    assert "hidden_probe" in rep
    assert not math.isnan(rep["ablation"]["all"]["task"]["share"])
