"""Lexsem analysis (wf/lexsem_analysis.py) on planted scores: readings, gates, flags and optional arms."""
import json
import math
import random

import pytest

from wf import lexsem
from wf import lexsem_analysis as LA
from wf import lexsem_rules as R
from wf.constants import REVISIONS

OBJECTS = ("mug", "pan", "saltshaker", "apple", "soapbar")  # homonym, alias + homonym, spelling, plain, few-shot
A_TRUE = 0.95


def _sig(y):
    return 1 / (1 + math.exp(-y))


def _obs(y, m=0.9, nothing_scale=1.0):
    return {"lp_a": math.log(m * _sig(y)), "lp_b": math.log(m * (1 - _sig(y)) * nothing_scale),
            "tok_lp": [math.log(m * _sig(y)), -0.5, -0.1], "toks": [" You", " pick", " up"], "n_prompt": 600}


def _yn(y):
    return {"lp_a": math.log(_sig(y)), "lp_b": math.log(1 - _sig(y)), "n_prompt": 600}


def plant(tmp_path, weights, groups=("core",), noise=0.02, seed=0, n=(120, 60), games=40, deny_mass=None,
          trap=False, fail=False, desc_fail=False, commit=None, name="qwen", a_true=A_TRUE, model="Qwen/Qwen3-8B",
          gpu="NVIDIA L40S", g_mean=3.0, overrides=None, competition=0.0):
    """Per-item scores from a planted mixture: every mention condition's net lift is its planted relative lift times
    A times the item's goal effect, on top of its control."""
    rng = random.Random(seed)
    rho = {**R.planted(weights), **(overrides or {})}
    conds = tuple(dict.fromkeys(c for g in groups for c in lexsem.GROUPS[g]))
    items, scores = [], []
    for i in range(sum(n)):
        st = "later" if i < n[0] else "step0"
        obj = OBJECTS[i % len(OBJECTS)]
        b, g = rng.gauss(-3, 1), max(0.3, rng.gauss(g_mean, 0.4))
        y = {"swapped": b, "aligned": b + g}
        for c in conds:
            if c in lexsem.ANCHORS or not c.startswith(("ctrl", "cdesc")):
                continue
            y[c] = b + 0.2 + rng.gauss(0, noise)
        y["name_assert"] = y["ctrl_assert"] + a_true * g + rng.gauss(0, noise)
        for c in conds + conds:  # twice, so a condition whose control is itself a mention condition is filled
            if c in y or c in lexsem.ANCHORS:
                continue
            form = lexsem.MENTIONS[c][0]
            if (form == "variant" and obj not in lexsem.VARIANTS) or (form == "alias" and obj not in lexsem.ALIASES) \
                    or (form == "homonym" and obj not in lexsem.HOMONYMS):
                continue
            if c == "aligned_ctrl_assert":  # the carrier's own +0.2 plus any planted competition (in goal effects)
                y[c] = y["aligned"] + 0.2 + competition * g + rng.gauss(0, noise)
                continue
            if c in ("name_reliable", "name_unreliable"):  # planted: the source's stated reliability changes nothing
                y[c] = y[lexsem.CONTROL_OF[c]] + a_true * g + rng.gauss(0, noise)
                continue
            ctrl = lexsem.CONTROL_OF.get(c, "ctrl_assert")
            if ctrl not in y:
                continue
            y[c] = y[ctrl] + rho.get(c, 1.0) * a_true * g + rng.gauss(0, noise)
        rec = {}
        for c, v in y.items():
            m = deny_mass if (deny_mass and c == "name_deny") else 0.9
            rec[c] = {"obs": _obs(v, m), "yn": _yn(v)}
            if trap and c == "name_use":  # the lift comes only from moving mass off " Nothing"
                base = y["ctrl_use"]
                rec[c]["obs"] = _obs(base, 0.9)
                rec[c]["obs"]["lp_b"] = math.log(0.9 * (1 - _sig(base))) - (v - base)
            if fail:
                rec[c]["fail"] = {"lp_a": 0.0, "lp_b": 0.0, "tok_lp": [rec[c]["obs"]["lp_b"], -2.5, -0.2],
                                  "toks": [" Nothing", " happens", "."], "n_prompt": 600}
        gid = f"valid_unseen/{i % games}"
        items.append({"id": f"t{i:04d}", "game_id": gid, "stratum": st, "goal_object": obj,
                      "fewshot_object": obj in lexsem.FEWSHOT_OBJECTS,
                      "references": {"target_prevalence": rng.random(), "ctrl_prevalence": rng.random()}})
        scores.append({"id": f"t{i:04d}", "game_id": gid, "stratum": st, "scores": rec})
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(s) + "\n" for s in scores))
    (tmp_path / f"{name}_items.jsonl").write_text("".join(json.dumps(s) + "\n" for s in items))
    checks = [{"form": "desc", "object": o, "passed": not desc_fail} for o in OBJECTS]
    checks += [{"form": "alias", "object": o, "passed": True} for o in lexsem.ALIASES]
    (tmp_path / f"{name}_check.json").write_text(json.dumps(checks))
    if commit:
        (tmp_path / f"{name}_commit.txt").write_text(commit + "\n")
    env = {"gpu": gpu, "torch": "2.14.1", "transformers": "5.23.0", "model": model, "revision": REVISIONS[model],
           "dtype": "bfloat16", "unbatched": True}
    (tmp_path / f"{name}_env.json").write_text(json.dumps(env))
    return p


def _run(p, **kw):
    return LA.run(str(p), B=200, **kw)


@pytest.mark.parametrize("weights,expected", [
    ({"evidence": 1}, "only evidence fits"),
    ({"lexical": 1}, "only lexical fits"),
    ({"semantic": .5, "evidence": .5}, "no single proposed account"),
    ({"lexical": .5, "semantic": .5}, "no single proposed account"),
])
def test_core_package_reads_planted_accounts_without_optional_arms(tmp_path, weights, expected):
    rep = _run(plant(tmp_path, weights))
    for s in ("later", "step0"):
        x = rep["qwen_plain"]["first"][s]
        assert x["gate_goal_effect"] == "passed"
        assert x["single_account_reading"].startswith(expected), (s, x["single_account_reading"])
        assert x["mention_reading"] == "a non-goal assertion reproduces the goal effect"
        assert abs(x["A"]["estimate"] - A_TRUE) < 0.05
        assert x["ignore"] == x["position"] == x["reliability"] == "not scored"
        assert "single_account_reading" in x["without_fewshot_objects"]  # S7 repeat
        assert x["without_fewshot_objects"]["n_items"] < x["n_items"]
    assert rep["full_two_candidate"] == rep["block_yes_no"] == rep["llama_yes_no"] == "not scored"
    assert rep["qwen_plain"]["first"]["step"]["A"]["reading"] == "no step difference beyond 0.25"


def test_mixture_weights_and_item_agreement(tmp_path):
    x = _run(plant(tmp_path, {"semantic": .5, "evidence": .5}, noise=0.0))["qwen_plain"]["first"]["later"]
    w = x["mixture_weights"]
    assert abs(w["semantic"]["estimate"] - .5) < 0.02 and abs(w["evidence"]["estimate"] - .5) < 0.02
    assert abs(w["lexical"]["estimate"]) < 0.02
    agr = x["item_agreement"]["name_assert"]
    assert agr["correlation"]["estimate"] > 0.99 and agr["reading"] == "tracks the goal effect item by item"
    assert abs(agr["slope"]["estimate"] - A_TRUE) < 0.02


def test_optional_arms_are_read_when_present(tmp_path):
    p = plant(tmp_path, {"evidence": 1}, groups=("core", "ignore", "position", "reliability"), fail=True)
    block = plant(tmp_path, {"evidence": 1}, groups=("names_only",), name="block")
    llama = plant(tmp_path, {"semantic": 1}, groups=("names_only",), name="llama",
                  model="meta-llama/Llama-3.1-8B-Instruct")
    rep = _run(p, block_path=str(block), llama_path=str(llama))
    x = rep["qwen_plain"]["first"]["later"]
    assert x["ignore"]["reading"] == "contradicts relevance as a hint"  # planted: ignore lifts fully
    assert x["position"]["recency"]["reading"] == "no recency effect beyond 0.25" and "confounds" in x["position"]
    assert x["position"]["visibility"]["reading"] == "no visibility effect beyond 0.25"
    assert x["reliability"]["reading"] == "no reliability effect beyond 0.25"
    assert rep["full_two_candidate"] == "scored" and "A" in rep["qwen_plain"]["full2c"]["later"]
    assert rep["block_yes_no"]["yn"]["later"]["single_account_reading"].startswith("only evidence fits")
    assert "name_use" in rep["model_comparison_yes_no"]["later"]  # Qwen evidence-like, Llama semantic-like
    assert "reading" in rep["qwen_plain"]["yn"]["step0"]["bridge"]
    assert rep["bridge_plain_vs_block"]["step0"]["reading"] == "the same in both formats (within 0.25)"
    q = rep["qwen_plain"]
    assert set(q["readings_by_readout"]) == {"first-token observation", "yes/no", "full two-candidate"}
    assert len(q["readout_disagreements"]) == 3
    assert "scope" in x["reliability"] and "homonym_reading" in x


def test_mass_gate_falls_back_to_the_unnormalized_readout(tmp_path):
    rep = _run(plant(tmp_path, {"evidence": 1}, deny_mass=0.6))
    x = rep["qwen_plain"]["first"]["later"]
    assert "name_deny" in x["gate_mass"]["failing"]
    assert x["relative_lift"]["name_deny"]["readout_used"] == "you"
    assert x["relative_lift"]["name_use"]["readout_used"] == "first"


def test_a_lift_made_by_moving_mass_off_nothing_is_caught(tmp_path):
    """The planted listed word lifts the two-word log-odds only by taking probability from " Nothing": the mass gate
    sends it to the unnormalized readout, where it lifts nothing."""
    x = _run(plant(tmp_path, {"lexical": 1}, trap=True))["qwen_plain"]["first"]["later"]
    assert "name_use" in x["gate_mass"]["failing"]
    iv = x["relative_lift"]["name_use"]
    assert iv["readout_used"] == "you" and iv["category"] == ["about zero"]


def test_flags_compare_categories_between_readouts():
    a = {"later": {"A": {"ci": [0.8, 1.0]}, "relative_lift": {"name_use": {"ci": [0.8, 1.1]},
                                                               "name_deny": {"ci": [0.9, 1.0]}}}, "step0": {}}
    b = {"later": {"A": {"ci": [0.9, 1.1]}, "relative_lift": {"name_use": {"ci": [-0.1, 0.1]},
                                                               "name_deny": {"ci": [0.6, 0.8]}}}, "step0": {}}
    assert LA.flags(a, b) == {"later": ["name_use"], "step0": []}


def test_recognition_gate_and_commit_check(tmp_path):
    rep = _run(plant(tmp_path, {"evidence": 1}, desc_fail=True))
    x = rep["qwen_plain"]["first"]["later"]
    assert x["relative_lift"]["desc_assert"]["status"].startswith("not assessable")
    p = plant(tmp_path, {"evidence": 1}, commit="abc123")
    assert _run(p, expect_commit="abc123")["qwen_plain"]
    with pytest.raises(SystemExit, match="frozen commit"):
        _run(p, expect_commit="def456")


def test_ratios_carry_their_numerator_and_denominator_and_need_a_well_estimated_reference(tmp_path):
    x = _run(plant(tmp_path, {"evidence": 1}))["qwen_plain"]["first"]["later"]
    iv = x["relative_lift"]["desc_assert"]
    assert iv["reference_ok"] and "numerator_nats" in iv and "denominator_nats" in iv
    assert iv["num_den_correlation"] is not None
    cov = _run(plant(tmp_path, {"evidence": 1}, name="cov"))["qwen_plain"]["first"]["later"]["prevalence_covariate"]
    assert abs(cov["correlation"]["estimate"]) < 0.3  # planted: the net effect does not depend on the gap
    weak = _run(plant(tmp_path, {"evidence": 1}, a_true=0.03, noise=0.3, name="weak"))["qwen_plain"]["first"]["later"]
    assert all("ratio_not_read" in v for v in weak["relative_lift"].values() if "reference_ok" in v)
    assert weak["single_account_reading"].startswith("not assessable")


def test_environment_check(tmp_path):
    p = plant(tmp_path, {"evidence": 1}, gpu="NVIDIA A40")
    with pytest.raises(SystemExit, match="GPU"):
        _run(p)
    assert _run(p, require_env=False)["qwen_plain"]


def test_goal_effect_below_five_floors_is_not_read(tmp_path):
    """Plain-text Qwen floor 0.08 nats: a goal effect of 1.3 nats gives a quarter-band of 0.33 < 5 x 0.08."""
    x = _run(plant(tmp_path, {"evidence": 1}, g_mean=1.3, name="small"))["qwen_plain"]["first"]["later"]
    assert x["gate_goal_effect"].startswith("failed") and x["numerical_floor_nats"] == 0.08


def test_numerical_floor_makes_small_references_unreadable(tmp_path):
    p = plant(tmp_path, {"evidence": 1})
    ok = _run(p)["qwen_plain"]["first"]["later"]["relative_lift"]["name_use"]
    assert ok["resolvable"] and "ratio_not_read" not in ok
    try:
        low = _run(p, floor=5.0)["qwen_plain"]["first"]["later"]
    finally:
        LA.NUMERICAL_FLOOR = LA.FLOORS["plain"]
    assert "below numerical resolution" in low["relative_lift"]["name_use"]["ratio_not_read"]
    assert low["accounts"]["evidence"]["per_contrast"] == {}


@pytest.mark.parametrize("plain_ignore,expected", [(1.0, "model-level"), (0.3, "block-dependent")])
def test_ignore_across_arms(tmp_path, plain_ignore, expected):
    arms = ("names_only", "ignore", "competition")
    p = plant(tmp_path, {"lexical": 1}, groups=("core", "ignore", "competition"),
              overrides={"name_ignore": plain_ignore})
    block = plant(tmp_path, {"lexical": 1}, groups=("block8", "ignore", "competition"), name="block",
                  overrides={"name_ignore": 1.0})
    llama = plant(tmp_path, {"lexical": 1}, groups=arms, name="llama", model="meta-llama/Llama-3.1-8B-Instruct",
                  overrides={"name_ignore": 0.3})
    rep = _run(p, block_path=str(block), llama_path=str(llama))
    assert rep["ignore_across_arms"]["yn|later"]["reading"].startswith(expected)
    assert "first|later" in rep["ignore_across_arms"]  # plain against block in the first token


def test_competition_reading(tmp_path):
    x = _run(plant(tmp_path, {"evidence": 1}, groups=("core", "competition"), competition=-0.6))
    assert x["qwen_plain"]["first"]["later"]["competition"]["reading"].startswith("competition")
    y = _run(plant(tmp_path, {"evidence": 1}, groups=("core", "competition"), name="nocomp"))
    assert y["qwen_plain"]["first"]["later"]["competition"]["reading"].startswith("no competition")
