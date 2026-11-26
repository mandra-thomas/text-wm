"""Controlled Bayesian environment (wf/bayesenv, notes/bayesenv_design.md): tests of what could silently produce a
wrong number. The exact posterior (closed form, forward simulation of the stated story, provenance), the prompts
(only the defining line differs between prompt types; the generator names no object), the design's item sets, and
the analysis on planted models (each planted effect recovered on its own estimand and absent from the others; the
wording random effect's coverage; the specificity rule; the sizing rule), plus the run guards."""
from __future__ import annotations

import difflib
import json
import math
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
    rng = np.random.default_rng(hash((hist, prov)) % 2 ** 32)
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
    assert c["toy"] == 28 * 30 * 14 * 4 + 8 * 30 * 11 * 4 + 18 * 30 * 2 * 2 + 6 * 30 * 2 * 2


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
    "redundancy_pattern": {"no_redundancy": 0.6},
    "announced_slope": {"announcement": 0.8},
    "history_only_provenance": {"overread": 2.0},
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
            assert shift == pytest.approx(want, rel=0.35), (model, shift)


def test_bayes_planted_slopes_equal_the_scale(planted):
    E = planted["bayes"]
    for k in ("evidence_slope", "history_slope", "goal_slope", "tag_slope", "note_slope"):
        assert abs(E[k].est - 0.7) < 3 * E[k].se + 0.05, (k, E[k].est, E[k].se)
    assert abs(E["seeing_guide_ratio"].est - 0.7) < 0.15


def test_wording_random_effect_keeps_coverage():
    """With wording effects shared across cells, a CI from items alone is anti-conservative; the panel-based random
    effect restores coverage (section 8)."""
    d = D.Design(n=12)
    units = [u for u in D.full_units(d) if u.space == "toy" and u.hist in ("H0", "H2", "H3", "H6", "H9")]
    cells = [c for c in D.CELLS if c[1] in ("H0", "H2", "H3", "H6", "H9")]
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


def test_incomplete_wording_panel_stops_the_analysis():
    units = [u for u in D.full_units(D.Design(n=3)) if u.space == "toy" and (u.gen, u.hist) != D.PANEL_CELLS[0]]
    sc = A.load_records(S.planted_records(units, S.Planted(), seed=0))
    with pytest.raises(ValueError, match="panel incomplete"):
        A.wording_components(sc, A.PRIMARY, "goal")


def test_holm():
    assert A.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert A.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


def test_specificity_rule_separates_world_model_only_from_general(small_design):
    units = [u for u in D.full_units(small_design) if u.space == "toy"]
    res = {}
    for name, pm in (("general", S.Planted(goal_role=0.8, sd_noise=0.6)),
                     ("wm_only", S.Planted(goal_role=0.8, wm_only=True, sd_noise=0.6))):
        sc = A.load_records(S.planted_records(units, pm, seed=3))
        slopes = {ro: A.evidence_slopes(sc, ro, boot=50)["evidence_slope"] for ro in A.READOUTS}
        gates = {ro: {"passed": True} for ro in A.READOUTS}
        res[name] = A.specificity(sc, small_design, slopes, gates)["goal_minus_tag"]
    assert res["wm_only"]["specific_to_transition_prediction"] is True
    assert res["general"]["specific_to_transition_prediction"] is False


def test_readings_pattern_rule_and_equivalence():
    E = {k: A.Est(0.0, 0.05, n_items=600, sd_item=1.0) for k in A.HOLM_FAMILY}
    r = A.readings(E, "small", True, target=0.35)
    assert not r["residual_pattern_detected"] and r["no_pattern_shown"]
    assert r["goal_minus_tag"]["equivalent_within_margin"]
    assert not A.readings(E, "small", False, target=0.35)["goal_minus_tag"]["equivalent_within_margin"]
    E["tag_naming_zero_information"] = A.Est(0.5, 0.05, n_items=200)
    r = A.readings(E, "small", True, target=0.35)
    assert r["content_blind_constants_detected"] == ["tag_naming_zero_information"]
    assert not r["residual_pattern_detected"]
    E["announced_slope"] = A.Est(0.8, 0.1, n_items=600)
    assert A.readings(E, "small", True, target=0.35)["pattern_contrasts_detected"] == ["announced_slope"]


def test_smoke_readout_bands_follow_wording_variance():
    units = [u for u in D.smoke_units() if u.space == "toy"]
    timing = {"sequences": 7120, "seconds": 7120 * 0.08}
    quiet = A.smoke_readout(S.planted_records(units, S.Planted(sd_noise=0.4, sd_item=0.5), seed=1), timing, None)
    noisy = A.smoke_readout(S.planted_records(units, S.Planted(sd_noise=0.4, sd_item=0.5, sd_wording=0.6), seed=1),
                            timing, None)
    assert quiet["decision"]["band"].startswith("at or below target")
    assert "not used for negative conclusions" in noisy["decision"]["band"]
    assert "goal_minus_tag" not in json.dumps(quiet["variance"]) or "estimate" not in json.dumps(quiet["variance"])


# ----------------------------------------------------------------------------- run guards


def test_score_units_resumes_and_refuses_changed_prompts(tmp_path):
    from wf.bayesenv import score

    units = D.full_units(D.Design(n=1, n_h4=1, n_ann=1, alf=False))[:3]
    calls = []
    fake = lambda u: calls.append(u.uid) or {"x": {"lp_a": -1.0, "lp_b": -2.0}}  # noqa: E731
    out = tmp_path / "scoring.jsonl"
    assert score.score_units(units, out, fake, resume=False) == 3
    assert score.score_units(units, out, fake, resume=True) == 0
    lines = out.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["unit_hash"] = "changed"
    out.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
    with pytest.raises(SystemExit):
        score.score_units(units, out, fake, resume=True)


def test_plain_encoding_refuses_markup_and_multi_token_candidates():
    from wf.bayesenv import score

    class Tok:
        all_special_ids = [0]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}   # one id per character: " You" is several tokens

    p = R.render("explore", "A", "H0", 0, "wm", "first")
    with pytest.raises(SystemExit):
        score.encode(Tok(), p, "plain")
    bad = R.Prompt(messages=(("user", "<think></think> x"),), prefix="", candidates=("a", "b"))
    with pytest.raises(SystemExit):
        score.encode(Tok(), bad, "plain")


def test_check_run_refuses_mock_dirty_and_mixed_environments(tmp_path, monkeypatch):
    from wf.bayesenv import pipeline

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    env = {"gpu": "NVIDIA L40S", "torch": "2", "transformers": "4", "mode": "plain"}
    for i, (status, dirty, gpu) in enumerate((("REPORTED (shard complete)", False, "NVIDIA L40S"),
                                              ("REPORTED (shard complete) (MOCK)", True, "NVIDIA A40"))):
        p = tmp_path / "r" / f"shard_{i:02d}_of_02"
        p.mkdir(parents=True)
        (p / "STATUS.md").write_text(f"# STATUS — r — {status}\n")
        (p / "inputs.json").write_text(json.dumps({"inputs": {}, "git_dirty": dirty}))
        (p / "git_sha").write_text("abc\n")
        (p / "env.json").write_text(json.dumps({**env, "gpu": gpu}))
    _, _, problems = pipeline.check_run("r", "L40S")
    text = " ".join(problems)
    assert "not a complete real shard" in text and "dirty" in text and "different environments" in text


def test_classify_reply():
    from wf.bayesenv.score import classify_reply

    assert classify_reply("toy", "wm", "first", " You take the key from box 3.") == "a"
    assert classify_reply("toy", "wm", "first", " You cannot take it.") == "a_other"
    assert classify_reply("toy", "si", "narr", " holds the key and is not locked.") == "a"
    assert classify_reply("toy", "si", "narr", " holds the key but is locked.") == "a_other"
    assert classify_reply("alf", "alf", "first", " a keychain 1.") == "a"
