"""Controlled Bayesian environment (wf/bayesenv, notes/bayesenv_design.md): tests of what could silently produce a
wrong number. The exact posterior (closed form, forward simulation of the stated story, provenance), the prompts
(only the defining line differs between prompt types; the generator names no object), the design's item sets, and
the analysis on planted models (each planted effect recovered on its own estimand and absent from the others; the
wording random effect's coverage; the specificity rule; the sizing rule), plus the run guards."""
from __future__ import annotations

import difflib
import json
import math
import re
import zlib
from dataclasses import replace

import numpy as np
import pytest

from wf.bayesenv import analysis as A
from wf.bayesenv import design as D
from wf.bayesenv import render as R
from wf.bayesenv import simulate as S
from wf.bayesenv import world as W

# ----------------------------------------------------------------------------- the exact posterior


def test_closed_form_note_shift_matches_enumeration():
    """B1: the Bayes log-odds shift of an existence note is log L + log[(1 + (1-r) o) / (1 + (1-r) L o)]."""
    for pi in (0.2, 0.5, 0.75):
        for looks, miss in ((0, 0.0), (4, 0.3), (4, 0.0), (2, 0.3)):
            ev = W.Evidence(looks=looks, look_miss=miss)
            lam_h = ((W.K - 1 - looks) + 1 + looks * miss) / W.K
            o = pi / (1 - pi) * lam_h
            r = (1 - W.LOCK_P) / ((W.K - looks) + looks * miss)
            for rho in (0.5, 0.8, 0.95):
                lam = W.note_ratio(rho)
                d = W.logit(W.posterior(pi, replace(ev, note=rho))) - W.logit(W.posterior(pi, ev))
                cf = math.log(lam) + math.log((1 + (1 - r) * o) / (1 + (1 - r) * lam * o))
                assert d == pytest.approx(cf, abs=1e-12)


@pytest.mark.parametrize("hist", W.HISTORY_ORDER)
@pytest.mark.parametrize("prov", ["P0", "P2"])
def test_posterior_matches_forward_simulation(hist, prov):
    rng = np.random.default_rng(zlib.crc32(f"{hist}|{prov}".encode()))
    for pi, note in ((0.5, None), (0.907, 0.8), (0.2, 0.0)):
        ev = replace(W.HISTORIES[hist], note=note, provenance=prov)
        ex = W.posterior(pi, ev)
        mc, n = W.simulate(pi, ev, 600_000, rng)
        if ex in (0.0, 1.0):
            assert mc == ex
            continue
        assert abs(mc - ex) / math.sqrt(ex * (1 - ex) / n) < 4.5


def test_history_only_helper_equals_tester_and_is_undefined_where_box3_was_looked_into():
    for h in ("H0", "H2", "H6"):
        for pi in (0.5, 0.9):
            assert W.posterior(pi, replace(W.HISTORIES[h], provenance="P1")) == \
                pytest.approx(W.posterior(pi, W.HISTORIES[h]), abs=1e-15)
    with pytest.raises(ValueError):
        W.posterior(0.5, replace(W.HISTORIES["H7"], provenance="P1"))
    assert all(h not in ("H7", "H9") for _, h in D.PROV_CELLS)


def test_generators_priors_and_settled_cells():
    assert [round(W.likelihood_ratio(g), 2) for g in W.GENERATOR_ORDER] == [1.0, 3.0, 9.0, 39.01]
    assert W.prior("C", True, announced=True) == pytest.approx((0.9 + 2 * 0.5) / 3)
    for g in W.GENERATOR_ORDER:
        assert W.exact(g, "H7", True) == 0.0
        # existence settled: a note has no Bayes effect on any scale
        assert W.exact(g, "H4", False, note=0.95) == pytest.approx(W.exact(g, "H4", False), abs=1e-15)
        # the matched cues share the exact posterior
        for t in ("tag1", "tagN", "coin_tag", "key2"):
            assert R.exact(t, g, "H2") == R.exact("key1", g, "H2")
        assert R.exact("tagO", g, "H2") == R.exact("explore", g, "H2") == R.exact("coin", g, "H2")
        assert R.exact("ann_key", g, "H2") == R.exact("ann_coin", g, "H2")


# ----------------------------------------------------------------------------- prompts


def _diff(a: str, b: str) -> list[str]:
    return [x for x in difflib.ndiff(a.splitlines(), b.splitlines()) if x[:1] in "+-"]


def test_prompt_types_differ_only_in_their_defining_line():
    for g, h, k in (("C", "H6", 3), ("A", "H0", 0), ("D", "H4", 7)):
        base = R.user_text("explore", g, h, k, "wm")
        for t, n_lines in (("key1", 1), ("key5", 1), ("tag3", 1), ("tagO", 1), ("tagN", 1), ("coin", 1),
                           ("coin_tag", 1), ("note80", 1)):
            d = _diff(base, R.user_text(t, g, h, k, "wm"))
            assert len([x for x in d if x.startswith("+")]) == n_lines, (t, d)
        assert len([x for x in _diff(R.user_text("key1", g, h, k, "wm"), R.user_text("tag1", g, h, k, "wm"))
                    if x.startswith("+")]) == 1
        # notes differ from each other only in the stated reliability
        a, b = R.user_text("note0", g, h, k, "si"), R.user_text("note95", g, h, k, "si")
        assert a.replace("probability 0%", "X") == b.replace("probability 95%", "X")


def test_generator_names_no_object_and_framings_share_the_story():
    for g in W.GENERATOR_ORDER:
        txt = R.generator_text(g).lower()
        assert "key" not in txt and "coin" not in txt
    wm = R.user_text("tag2", "R3", "H9", 4, "wm")
    si = R.user_text("tag2", "R3", "H9", 4, "si")
    d = _diff(wm, si)
    assert len(d) == 2 and all("tester" in x for x in d)   # only the provenance sentence differs
    p = R.render("note50", "R3", "H9", 4, "si", "narr")
    assert p.prefix == R.NARR_PREFIX and p.candidates == ("holds", "does")


def test_surfaces_are_deterministic_and_never_look_into_box_3():
    for g, h in D.CELLS:
        for k in range(40):
            s = R.surface(g, h, k)
            assert s == R.surface(g, h, k)
            assert 3 not in s.looked and len(set(s.rooms)) == 6
            assert len(s.looked) == W.HISTORIES[h].looks


def test_alfworld_surface_exact_and_prompt():
    assert R.alf_exact("explore", "A", "AH0") == pytest.approx(0.5 / 6)
    assert R.alf_exact("key1", "C", "AH3") == pytest.approx(0.9 / 6 / (0.1 + 0.9 * 2 / 6))
    assert R.alf_exact("imp_key", "none", "AH0") is None
    p = R.alf_render("tag2", "D", "AH3", 5, "first")
    assert p.prefix.endswith("you see") and p.candidates == ("a", "nothing")
    assert p.messages[-1][1].count("> open drawer") == 4


def test_every_prompt_names_the_key_in_a_line_shared_by_all_prompt_types():
    """The queried object must be named in every prompt, by a line that is the same for every prompt type of an
    item (the action line in the world model, the narrator instruction or question in state inference)."""
    for g, h in (("A", "H0"), ("C", "H2"), ("D", "H6"), ("R3", "H7")):
        for fr, ros in D.FRAMING_READOUTS.items():
            for r in ros:
                shared = None
                for t in R.TYPES:
                    if fr == "si" and t in R.WM_ONLY or (t.startswith("P1") and h == "H7"):
                        continue
                    p = R.render(t, g, h, 0, fr, r)
                    lines = {ln for ln in (p.messages[1][1] + "\n" + p.prefix).splitlines() if "key" in ln}
                    shared = lines if shared is None else shared & lines
                assert shared, (g, h, fr, r)


def test_numbers_in_the_rules_match_the_posterior_constants():
    txt = R.render("note80", "D", "H6", 0, "wm", "first").messages[1][1]
    assert f"locked with probability {round(100 * W.LOCK_P)}%" in txt
    assert f"{round(100 * W.MISS_NOISY)}% of the time" in txt
    assert f"wrong {round(100 * W.LOCK_REPORT_ERROR)}% of the time" in txt
    assert "probability 90.7%" in txt and "probability 20%" in txt and W.GENERATORS["D"] == (0.2, 0.907)
    assert "probability 80%" in R.note_text(0.8)
    g = R.render("explore", "A", "H9", 0, "wm", "first").messages[1][1]
    assert f"{round(100 * W.GLIMPSE_HIT)}% of the time" in g and f"{round(100 * W.GLIMPSE_FALSE)}% of the time" in g
    for wording in R.P2:
        assert f"{round(100 * W.KAPPA)}% of the time" in wording and "twelve" in wording and W.N_TAKE_ACTIONS == 12
    for gen, (lo, hi) in W.GENERATORS.items():
        nums = [float(x) for x in re.findall(r"probability ([0-9.]+)%", R.generator_text(gen))]
        assert nums == [round(100 * hi, 1), round(100 * lo, 1)]
    assert "is not in the house at all" in R.generator_text("A")


def test_implicit_alfworld_prompts_say_who_chose_the_actions():
    txt = R.alf_render("imp_key", "none", "AH0", 0, "first").messages[-1][1]
    assert "picked at random by a tester" in txt and "probability" not in txt


# ----------------------------------------------------------------------------- design


def test_design_counts_subsets_and_disjoint_smoke():
    d = D.Design()
    full = D.full_units(d)
    sub = D.full_units(d, k_max=D.XGPU_K)
    assert len([u for u in sub if u.space == "toy"]) == 28 * D.XGPU_K
    by = {u.uid: u for u in full}
    for u in sub:
        assert by[u.uid].types == u.types and D.unit_hash(by[u.uid]) == D.unit_hash(u)
    assert all(u.k >= D.SMOKE_K0 for u in D.smoke_units())
    assert all(u.k < D.SMOKE_K0 for u in full)
    c = D.count_sequences(full)
    assert c["toy"] == 28 * 30 * 14 * 4 + 8 * 30 * 11 * 4 + 15 * 30 * 2 * 2 + 6 * 30 * 2 * 2


# ----------------------------------------------------------------------------- analysis on planted models


@pytest.fixture(scope="module")
def small_design():
    return D.Design(n=20, n_h4=20, n_ann=20)


@pytest.fixture(scope="module")
def planted(small_design):
    units = [u for u in D.full_units(small_design) if u.space == "toy"]
    out = {}
    for name, pm in (("bayes", S.Planted()), ("goal_role", S.Planted(goal_role=0.6)),
                     ("lexical", S.Planted(lexical_mention=0.6)), ("competing", S.Planted(competing_goal=-0.6)),
                     ("no_redundancy", S.Planted(no_redundancy=True)),
                     ("announcement", S.Planted(announcement_ignored=True)),
                     ("overread", S.Planted(provenance_overread=True))):
        sc = A.load_records(S.planted_records(units, pm, seed=7))
        out[name] = A.estimands(sc, A.PRIMARY, small_design, boot=50)
    return out


EXPECT = {  # estimand -> planted model under which it moves, and by how much (nats, observed scale)
    "goal_minus_tag": {"goal_role": 0.6},
    "tag_naming_zero_information": {"lexical": 0.6},
    "name_minus_description": {"lexical": 0.6},
    "competing_goal": {"competing": -0.6},
    "zero_reliability_note": {"lexical": 0.6},
    "redundancy_residual": {"lexical": 0.6, "no_redundancy": 0.7},
    "redundancy_pattern": {"no_redundancy": 0.7},
    "announced_goal_effect": {"lexical": 0.6, "competing": 0.6, "announcement": 0.5},
    "announced_slope": {"announcement": 0.8},
    "history_only_provenance": {"overread": 2.0},
    "history_only_provenance_slope": {"overread": 0.35},
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_each_planted_effect_moves_only_its_estimand(planted, name):
    base = planted["bayes"][name].est
    for model, E in planted.items():
        shift = E[name].est - base
        want = EXPECT[name].get(model)
        if want is None:
            assert abs(shift) < 1e-9, (model, shift)       # same seed: other planted effects cancel exactly
        else:
            assert shift == pytest.approx(want, rel=0.4), (model, shift)


def test_bayes_planted_residuals_are_zero_within_error():
    """Not relative to another planted model: under a tempered exact reasoner, every family residual is 0."""
    d = D.Design(n=20, n_h4=20, n_ann=20)
    units = [u for u in D.full_units(d) if u.space == "toy"]
    for seed in (11, 12):
        E = A.estimands(A.load_records(S.planted_records(units, S.Planted(), seed=seed)), A.PRIMARY, d, boot=20)
        for k in A.HOLM_FAMILY:
            assert abs(E[k].est) < 4 * E[k].se, (seed, k, E[k].est, E[k].se)


def test_bayes_planted_slopes_equal_the_scale(planted):
    E = planted["bayes"]
    for k in ("evidence_slope", "history_slope", "goal_slope", "tag_slope", "note_slope", "base_rate_slope"):
        assert abs(E[k].est - 0.7) < 3 * E[k].se + 0.05, (k, E[k].est, E[k].se)
    for k in ("goal_calibration_ratio", "tag_calibration_ratio", "note_calibration_ratio"):
        assert abs(E[k].est - 1.0) < 3 * E[k].se + 0.1, (k, E[k].est, E[k].se)
    assert abs(E["seeing_guide_ratio"].est - 0.7) < 0.15
    assert abs(E["named_rate_leakage"].est) < 4 * E["named_rate_leakage"].se


def test_readouts_are_not_mixed_up():
    """A deviation planted on one readout only appears on that readout."""
    d = D.Design(n=10, n_h4=10, n_ann=10)
    units = [u for u in D.full_units(d) if u.space == "toy"]
    sc = A.load_records(S.planted_records(units, S.Planted(goal_role=0.8, wm_only=True), seed=2))
    e_wm = A.estimands(sc, ("wm", "first"), d, boot=10)["goal_minus_tag"].est
    e_si = A.estimands(sc, ("si", "narr"), d, boot=10)["goal_minus_tag"].est
    assert e_wm > 0.5 and abs(e_si) < 0.3


def test_wording_random_effect_keeps_coverage():
    """With wording effects shared across cells, a CI from items alone is anti-conservative; the panel-based random
    effect restores coverage (section 8)."""
    d = D.Design(n=12)
    hs = ("H0", "H2", "H3", "H6", "H9")
    units = [u for u in D.full_units(d) if u.space == "toy" and u.hist in hs]
    cells = [c for c in D.CELLS if c[1] in hs]
    cov, naive = [], []
    for seed in range(60):
        sc = A.load_records(S.planted_records(units, S.Planted(sd_wording=0.25, sd_noise=0.7), seed=seed))
        e = A.mean_contrast(sc, A.PRIMARY, cells, A.goal_side(2), A.tag_side(3))
        er = A.add_wording(e, [(A.wording_components(sc, A.PRIMARY, "goal"), 2),
                               (A.wording_components(sc, A.PRIMARY, "tag"), 3)], len(cells))
        lo, hi = er.ci()
        cov.append(lo < 0 < hi)
        lo, hi = e.ci()
        naive.append(lo < 0 < hi)
    assert np.mean(cov) >= 0.88
    assert np.mean(naive) < np.mean(cov) - 0.15


def test_wording_components_recover_planted_variance():
    d = D.Design(n=40)
    units = [u for u in D.full_units(d) if u.space == "toy" and (u.gen, u.hist) in D.PANEL_CELLS]
    est = []
    for seed in range(12):
        sc = A.load_records(S.planted_records(units, S.Planted(sd_wording=0.3, sd_noise=0.5), seed=seed))
        est.append(A.wording_components(sc, A.PRIMARY, "tag").s2_alpha)
    assert np.mean(est) == pytest.approx(0.09, rel=0.45)


def test_incomplete_cells_stop_the_analysis():
    units = [u for u in D.full_units(D.Design(n=3)) if u.space == "toy" and (u.gen, u.hist) != D.PANEL_CELLS[0]]
    sc = A.load_records(S.planted_records(units, S.Planted(), seed=0))
    with pytest.raises(ValueError, match="panel incomplete"):
        A.wording_components(sc, A.PRIMARY, "goal")
    with pytest.raises(ValueError, match="complete items"):
        A.mean_contrast(sc, A.PRIMARY, list(D.CELLS), {"key1": 1.0}, {"explore": 1.0})


def test_holm():
    assert A.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert A.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


def _spec(design, pm, gate_fail=()):
    units = [u for u in D.full_units(design) if u.space == "toy"]
    sc = A.load_records(S.planted_records(units, pm, seed=3))
    slopes = {ro: A.evidence_slopes(sc, ro, boot=50)["evidence_slope"] for ro in A.READOUTS}
    gates = {ro: {"passed": ro not in gate_fail} for ro in A.READOUTS}
    return A.specificity(sc, design, slopes, gates)["goal_minus_tag"]


def test_specificity_rule_in_both_directions_and_when_an_arm_is_missing(small_design):
    assert _spec(small_design, S.Planted(goal_role=0.8, wm_only=True, sd_noise=0.6))[
        "specific_to_transition_prediction"] is True
    assert _spec(small_design, S.Planted(goal_role=-0.8, wm_only=True, sd_noise=0.6))[
        "specific_to_transition_prediction"] is True
    assert _spec(small_design, S.Planted(goal_role=0.8, sd_noise=0.6))["specific_to_transition_prediction"] is False
    r = _spec(small_design, S.Planted(goal_role=0.8, wm_only=True, sd_noise=0.6), gate_fail=(("si", "yn"),))
    assert r["specific_to_transition_prediction"] is None and r["label"] == "not assessable"


SIZ_OK = {"goal_minus_tag_negative_allowed": True, "patterns_not_for_negative": []}


def _family(est=0.0, se=0.05):
    return {k: A.Est(est, se, n_items=600, sd_item=1.0, n_eff=600) for k in A.HOLM_FAMILY}


def test_readings_pattern_rule_is_two_sided_and_single_residuals_are_constants():
    E = _family()
    r = A.readings(E, "small", True, 0.35, 0.5, SIZ_OK)
    assert not r["residual_pattern_detected"] and r["no_pattern_shown"] and r["goal_minus_tag"]["equivalent"]
    E["zero_reliability_note"] = A.Est(-0.6, 0.05, n_items=600, sd_item=1.0, n_eff=600)
    E["redundancy_pattern"] = A.Est(-0.9, 0.1, n_items=600, sd_item=1.0, n_eff=100)
    E["history_only_provenance"] = A.Est(2.0, 0.1, n_items=180, sd_item=1.0, n_eff=180)
    r = A.readings(E, "small", True, 0.35, 0.5, SIZ_OK)
    assert r["pattern_contrasts_detected"] == {"redundancy_pattern": -1.0}
    assert r["content_blind_constants_detected"] == {"zero_reliability_note": -1.0, "history_only_provenance": 1.0}


def test_equivalence_needs_both_scales_the_low_floor_family_and_the_sizing_permission():
    E = _family(se=0.05)
    assert A.readings(E, "small", True, 0.35, 0.5, SIZ_OK)["goal_minus_tag"]["equivalent"]
    assert not A.readings(E, "large", True, 0.35, 0.5, SIZ_OK)["goal_minus_tag"]["equivalent"]
    lab = A.readings(E, "large", False, 0.35, 0.5, SIZ_OK)["goal_minus_tag"]["label"]
    assert lab.startswith("not detected at an MDE of")
    no = {"goal_minus_tag_negative_allowed": False, "patterns_not_for_negative": ["announced_slope"]}
    r = A.readings(E, "small", True, 0.35, 0.5, no)
    assert not r["goal_minus_tag"]["equivalent"] and "sizing rule" in r["goal_minus_tag"]["label"]
    assert not r["patterns_equivalence"]["announced_slope"]["equivalent"] and not r["no_pattern_shown"]
    E["goal_minus_tag"] = A.Est(0.0, 0.05, n_items=600, sd_item=0.2, n_eff=600)   # nats inside, standardized not
    assert not A.readings(E, "small", True, 0.35, 0.5, SIZ_OK)["goal_minus_tag"]["equivalent"]
    with pytest.raises(ValueError):
        A.readings(E, "small", True, 0.35, 0.5, {})


def _checks(n=60):
    s = {}
    for key in ("toy|wm|first", "toy|si|narr", "alf|alf|first"):
        s[key] = {"a": n, "b": n, "other": 5}
    return {"samples": s, "description": {"passed": True, "yes_minus_no": [1, 1, 1]}}


def test_gates_require_samples_on_both_candidates():
    d = D.Design(n=5, n_h4=5, n_ann=5)
    units = D.full_units(d)
    sc = A.load_records(S.planted_records(units, S.Planted(), seed=1))
    slopes = {ro: A.slope(sc, ro, A.GRADED_CELLS, {"explore": 1.0}, boot=20) for ro in A.READOUTS}
    assert A.gates(sc, slopes, _checks())[("wm", "first")]["passed"]
    few = _checks()
    few["samples"]["toy|wm|first"] = {"a": 60, "b": 3}
    assert not A.gates(sc, slopes, few)[("wm", "first")]["passed"]
    bad = _checks()
    bad["samples"]["toy|si|narr"] = {"a": 30, "a_other": 30, "b": 60}
    assert not A.gates(sc, slopes, bad)[("si", "narr")]["passed"]
    with pytest.raises(ValueError):
        A.gates(sc, slopes, None)


def test_smoke_readout_bands_and_no_effect_leaks():
    units = [u for u in D.smoke_units() if u.space == "toy"]
    timing = {"sequences": 9300, "seconds": 9300 * 0.08}
    quiet = A.smoke_readout(S.planted_records(units, S.Planted(sd_noise=0.4, sd_item=0.5), seed=1), timing, _checks())
    noisy = A.smoke_readout(S.planted_records(units, S.Planted(sd_noise=0.4, sd_item=0.5, sd_wording=0.6), seed=1),
                            timing, _checks())
    assert quiet["decision"]["band"].startswith("at or below target")
    assert quiet["decision"]["sizing"]["goal_minus_tag_negative_allowed"] is True
    assert "not used for negative conclusions" in noisy["decision"]["band"]
    assert noisy["decision"]["sizing"]["goal_minus_tag_negative_allowed"] is False
    # a planted goal role (a constant on every goal prompt) changes nothing the readout reports
    role = A.smoke_readout(S.planted_records(units, S.Planted(sd_noise=0.4, sd_item=0.5, goal_role=0.8), seed=1),
                           timing, _checks())
    for r in (role, quiet):
        r.pop("logit_size")          # raw logit sizes move with any shift of the scores; they carry no contrast
    def flat(x, pre=""):
        if isinstance(x, dict):
            return {k2: v2 for k, v in x.items() for k2, v2 in flat(v, f"{pre}/{k}").items()}
        if isinstance(x, list):
            return {k2: v2 for i, v in enumerate(x) for k2, v2 in flat(v, f"{pre}/{i}").items()}
        return {pre: x}
    fr, fq = flat(role), flat(quiet)
    assert fr.keys() == fq.keys()
    for k in fq:
        if isinstance(fq[k], float):
            assert fr[k] == pytest.approx(fq[k], rel=1e-6, abs=1e-9), k   # float round trip through probabilities
        else:
            assert fr[k] == fq[k], k


def test_cross_gpu_refuses_a_different_subset():
    d = D.Design(n=10, n_h4=10, n_ann=10)
    main = S.planted_records(D.full_units(d), S.Planted(), seed=1)
    other = S.planted_records(D.full_units(D.Design(w_goal=3, w_tag=4, n=10, n_h4=10, n_ann=10), k_max=D.XGPU_K),
                              S.Planted(), seed=2)
    with pytest.raises(ValueError, match="same items and prompts"):
        A.cross_gpu(main, other, d, True, {}, SIZ_OK)


# ----------------------------------------------------------------------------- run guards


def test_score_units_resumes_and_refuses_changed_prompts(tmp_path):
    from wf.bayesenv import score

    units = D.full_units(D.Design(n=1, n_h4=1, n_ann=1, alf=False))[:3]
    fake = lambda u: {f"{t}|{fr}|{r}": {"lp_a": -1.0, "lp_b": -2.0} for t, fr, r in u.sequences()}  # noqa: E731
    out = tmp_path / "scoring.jsonl"
    assert score.score_units(units, out, fake, resume=False) == 3
    assert score.score_units(units, out, fake, resume=True) == 0
    lines = out.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["unit_hash"] = "changed"
    out.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
    with pytest.raises(SystemExit):
        score.score_units(units, out, fake, resume=True)
    assert A.verify_hashes([rec], units) == [rec["unit"]]
    # records are stored with sorted keys; a unit with several prompt types still verifies against its design unit
    fresh = [json.loads(x) for x in lines[1:]]
    multi = [u for u in units if len(u.types) > 1]
    assert multi and A.verify_hashes(fresh, units) == []
    assert all(list(r["scores"]) == sorted(r["scores"]) for r in fresh)


class _WordTok:
    """A stub tokenizer: one id per space-led word (" You" is one token), and markers as single tokens."""
    all_special_ids = [1]

    def __init__(self):
        self.vocab: dict = {}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab) + 2)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(x) for x in re.findall(r"<\|[a-z_]+\|>|\n+| ?[^\s<]+|\s+", text)]}

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, **kw):
        text = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>" for m in msgs) + "<|start|>assistant\n"
        return self(text)["input_ids"] if tokenize else text


def test_encoding_plain_and_chat_with_a_stub_tokenizer():
    from wf.bayesenv import score

    tok = _WordTok()
    for fr, r in (("wm", "first"), ("wm", "yn"), ("si", "narr"), ("si", "yn")):
        p = R.render("tag2", "C", "H2", 0, fr, r)
        for mode in ("plain", "chat"):
            ids, (a, b) = score.encode(tok, p, mode)
            sp = " " if (mode == "plain" or p.prefix) else ""
            assert (a, b) == (tok(sp + p.candidates[0])["input_ids"][0], tok(sp + p.candidates[1])["input_ids"][0])
    with pytest.raises(SystemExit):
        score.encode(tok, R.Prompt(messages=(("user", "<think></think> x"),), prefix="", candidates=("a", "b")),
                     "plain")

    class CharTok(_WordTok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [self._id(c) for c in text]}
    with pytest.raises(SystemExit):
        score.encode(CharTok(), R.render("explore", "A", "H0", 0, "wm", "first"), "plain")


def test_check_run_refuses_mock_dirty_and_mixed_environments(tmp_path, monkeypatch):
    from wf.bayesenv import pipeline

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    env = {"gpu": "NVIDIA L40S", "torch": "2", "transformers": "4", "mode": "plain"}
    for i, (status, dirty, gpu, n) in enumerate((("REPORTED (shard complete)", False, "NVIDIA L40S", 30),
                                                 ("REPORTED (shard complete) (MOCK)", True, "NVIDIA A40", 20))):
        p = tmp_path / "r" / f"shard_{i:02d}_of_02"
        p.mkdir(parents=True)
        (p / "STATUS.md").write_text(f"# STATUS — r — {status}\n")
        (p / "inputs.json").write_text(json.dumps({"inputs": {}, "git_dirty": dirty}))
        (p / "git_sha").write_text("abc\n")
        (p / "env.json").write_text(json.dumps({**env, "gpu": gpu}))
        (p / "config.json").write_text(json.dumps({"design": {"n": n}}))
    _, _, problems = pipeline.check_run("r", "L40S")
    text = " ".join(problems)
    assert "not a complete real run" in text and "dirty" in text and "different environments" in text
    assert "different designs" in text
    checks, cp = pipeline.load_checks("missing", "L40S")
    assert checks is None and cp


def test_classify_reply():
    from wf.bayesenv.score import classify_reply

    assert classify_reply("toy", "wm", "first", " You take the key from box 3.") == "a"
    assert classify_reply("toy", "wm", "first", " You cannot take it.") == "a_other"
    assert classify_reply("toy", "wm", "first", " Nothing happens.") == "b"
    assert classify_reply("toy", "si", "narr", " holds the key and is not locked.") == "a"
    assert classify_reply("toy", "si", "narr", " holds a key and is unlocked.") == "a"
    assert classify_reply("toy", "si", "narr", " holds the key but is locked.") == "a_other"
    assert classify_reply("toy", "si", "narr", " does hold the key and is not locked.") == "b_other"
    assert classify_reply("toy", "si", "narr", " does not hold a key that can be taken.") == "b"
    assert classify_reply("alf", "alf", "first", " a keychain 1.") == "a"
