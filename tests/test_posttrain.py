"""Post-training pilot: cells, reuse bookkeeping, readouts and the analysis's reading rules on planted scores."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wf import posttrain as P
from wf import posttrain_analysis as A


def test_cells_and_reuse():
    full = P.arm_cells("tulu-sft_native")
    assert len(full["failure"]) == 12 * 3 and len(full["success"]) == 3 * 3
    assert len(P.arm_cells("tulu-sft_plain")["failure"]) == 12 * 2
    assert P.arm_cells("tulu-sft_plain_floor") == {
        "failure": [(c, f) for c in P.FLOOR_CONDS_PLAIN for f in ("obs", "fail")], "success": []}
    assert P.arm_cells("tulu-sft_native_floor")["failure"] == [(c, f) for c in P.FLOOR_CONDS for f in ("obs", "fail")]
    assert "llama-base_plain7_floor" in P.ARMS
    # lexsem's Qwen base arm scores the anchors, the four mention pairs and the neutral task (obs and failure
    # text, except the failure text of "Ignore the X"), and the neutral task on the successes
    left = P.cells_to_score("qwen-base_plain")
    assert set(left["failure"]) == {("name_ignore", "fail"), ("ctrl_ignore", "fail"), ("aligned_desc", "obs"),
                                    ("aligned_desc", "fail")}
    assert set(left["success"]) == {(c, f) for c in P.SUCC_CONDS for f in ("obs", "fail")}
    assert P.cells_to_score("qwen-base_plain", reuse=False) == P.arm_cells("qwen-base_plain")
    # Llama-3.1-8B-Instruct: lexsem has only yes/no in its chat template
    left = P.cells_to_score("llama-instruct_native")
    assert not any(f == "yn" and c in ("aligned", "swapped", "name_assert") for c, f in left["failure"])
    assert ("aligned", "obs") in left["failure"]


def test_arms_cover_scope():
    assert {a for a, x in P.ARMS.items() if x["kind"] == "main" and not x.get("fewshot")} == \
        {f"{c}_plain" for c in P.PLAIN} | {f"{c}_native" for c in P.NATIVE}
    assert all(P.ARMS[a]["mode"] == "plain" for a in P.ARMS if P.ARMS[a]["ckpt"] in ("llama-base", "qwen-base"))
    assert P.ARMS["w2w-llama_w2w"]["mode"] == "chat"


def _rec(i, game, stratum, vals, item_set="failure", yn=None):
    """A record whose readouts are planted: vals[cond] = (first, full)."""
    sc = {}
    for c, (first, full) in vals.items():
        lp_b = -math.log1p(math.exp(first))  # the two readout tokens hold all the mass
        t0 = first + lp_b
        sc[c] = {"obs": {"tok_lp": [t0, -0.5], "lp_a": t0, "lp_b": lp_b},
                 "fail": {"tok_lp": [lp_b, t0 - 0.5 - full - lp_b], "lp_a": t0, "lp_b": lp_b}}
        if yn is not None:
            sc[c]["yn"] = {"lp_a": yn, "lp_b": -0.1}
    return {"id": i, "game_id": game, "stratum": stratum, "item_set": item_set, "scores": sc}


def test_readout_definitions():
    r = _rec("x", "g", "later", {"aligned": (1.5, 4.0)})
    assert A.readout(r, "aligned", "first") == pytest.approx(1.5)
    assert A.readout(r, "aligned", "full") == pytest.approx(4.0)
    assert A.readout(r, "aligned", "cont") == pytest.approx(2.5)


def _arm(name, goal_first, goal_full, rng, assert_share=1.0, deny_share=0.0, n=300, n_succ=150):
    fail, succ = {}, {}
    for k in range(n):
        noise = rng.normal(0, 1.0)
        base_f, base_F = -4 + rng.normal(0, 1), -8 + rng.normal(0, 2)
        gf, gF = goal_first + noise * 0.8, goal_full + noise * 1.6
        vals = {"swapped": (base_f, base_F), "aligned": (base_f + gf, base_F + gF),
                P.NEUTRAL: (base_f - 0.1, base_F - 0.2), P.DESC: (base_f + 0.8 * gf, base_F + 0.8 * gF)}
        for c, ctl in P.PAIRS.items():
            sh = {"name_assert": assert_share, "name_deny": deny_share}.get(c, 0.5)
            vals[ctl] = (base_f + 0.1, base_F + 0.1)
            vals[c] = (base_f + 0.1 + sh * gf, base_F + 0.1 + sh * gF)
        fail[f"i{k}"] = _rec(f"i{k}", f"g{k // 3}", "later", vals)
    for k in range(n_succ):
        v = (2 + rng.normal(0, 1), 3 + rng.normal(0, 1))
        succ[f"s{k}"] = _rec(f"s{k}", f"h{k // 2}", "success", {c: v for c in P.SUCC_CONDS}, "success")
    return {"arm": name, "fail": fail, "succ": succ, "recognition": {}, "_pair_of": {}, "reuse": {}, "jobs": [],
            "env": {}}


def test_gates_and_stage_readings():
    rng = np.random.default_rng(0)
    arms = {"llama-base_plain": _arm("llama-base_plain", 0.5, 1.0, rng),
            "tulu-sft_plain": _arm("tulu-sft_plain", 1.5, 3.0, rng),
            "tulu-dpo_plain": _arm("tulu-dpo_plain", 1.5, 3.0, rng, deny_share=-0.5)}
    res = A.analyse(arms, {a: x for a, x in arms.items()}, B=300)  # floor rescoring identical: floor 0
    assert res["gates"]["llama-base_plain"]["arm_ok"]
    g = res["arms"]["tulu-sft_plain"]["later"]["full"]["G"]["estimate"]
    assert g == pytest.approx(3.0, abs=0.3)
    c = res["comparisons"]["base -> Tulu SFT"]["readings"]["later"]
    assert c["full"]["reading"] == "higher" and c["first"]["reading"] == "higher"
    c2 = res["comparisons"]["Tulu SFT -> DPO"]["readings"]["later"]
    assert c2["full"]["reading"] in ("the same", "inconclusive")
    assert c2["full"]["shares"]["share|name_deny"] == "lower (descriptive)"  # planted: deny share 0 -> -0.5
    assert res["holm"]["n_readings"] == 35
    nofloor = A.analyse(arms, {}, B=300)["comparisons"]["base -> Tulu SFT"]["readings"]["later"]
    assert nofloor["full"]["reading"] == "not readable (floor not measured)"


def test_degenerate_base_fails_gate():
    rng = np.random.default_rng(1)
    a = _arm("llama-base_plain", 0.5, 1.0, rng)
    for r in a["succ"].values():  # successes scored like failures: no discrimination
        for c in r["scores"].values():
            c["obs"]["tok_lp"][0] = -5.0
    g = A.arm_gates(a)
    assert not g["arm_ok"]


def test_std_reading_scale_dependent():
    d = {"estimate": 0.4, "ci": [0.2, 0.6], "ci90": [0.25, 0.55], "p_two_sided": 0.001}
    neg = {"estimate": -0.1, "ci": [-0.2, 0.0], "ci90": [-0.18, -0.02], "p_two_sided": 0.05}
    r = A.std_g_reading(d, d, d, neg, 0.01, 0.01)
    assert r["reading"] == "scale-dependent"
    r = A.std_g_reading(d, d, d, {**d, "estimate": 0.1, "ci": [0.05, 0.15]}, 0.01, 0.01)
    assert r["reading"] == "higher"
    r = A.std_g_reading(d, d, d, {**d, "estimate": 0.1, "ci": [0.05, 0.15]}, 0.1, 0.01)  # 5 floors = 0.5 > 0.4
    assert r["reading"] != "higher"
    same = {"estimate": 0.02, "ci": [-0.1, 0.12], "ci90": [-0.08, 0.1], "p_two_sided": 0.7}
    assert A.std_g_reading(same, same, same, same, 0.01, 0.9)["reading"] == "the same"


def test_holm_and_auc():
    adj = A.holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.5)
    assert A.auc([1, 2, 3], [0, 0.5]) == 1.0 and math.isclose(A.auc([1], [1]), 0.5)


def test_reference_rule_is_two_sided():
    assert A.ref_ok({"estimate": -2.0, "ci": [-3.0, -1.2]})
    assert not A.ref_ok({"estimate": -2.0, "ci": [-3.0, -0.5]})
    assert A.ref_ok({"estimate": 2.0, "ci": [1.2, 3.0]}) and not A.ref_ok({"estimate": 2.0, "ci": [-0.1, 3.0]})


def test_no_reuse_arm_is_not_merged():
    recs, st = A.merge_lexsem("qwen_plain", [], None, {}, {"reuse": False})
    assert st == {"reused": False, "ok": True}
    recs, st = A.merge_lexsem("qwen_plain", [], None, {}, {"reuse": True})
    assert st["ok"] is False


def test_successes_never_reused():
    assert P.lexsem_cells("qwen8b_base_plain")["success"] == set()
    assert len(P.cells_to_score("qwen-base_plain")["success"]) == 6


def test_accepted_blob_pair_only():
    import subprocess

    have = all(subprocess.run(["git", "cat-file", "-e", c], capture_output=True).returncode == 0
               for c in ("5aabb99", "e0e9c04", "da2e312"))
    if not have:
        pytest.skip("lexsem commits not in this clone")
    L = A.LEXSEM_FILES
    assert A.same_code("5aabb99", "e0e9c04", L) and A.same_code("e0e9c04", "5aabb99", L)  # D-PT1's one-line change
    assert A.same_code("503ba47", "e0e9c04", L)
    assert not A.same_code("da2e312", "e0e9c04", L)  # any other difference still fails
    assert not A.same_code("5aabb99", "e0e9c04")  # the pilot's own files are compared in full
