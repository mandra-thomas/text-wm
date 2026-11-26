"""Natural ALFWorld bridge (wf/bridge.py, wf/bridge_analysis.py, wf/scoring/prefix.py; notes/bridge_design.md).

Proportionate tests: what could silently produce a wrong number. The sample and its strata; prompts that differ
only where intended; the plain-text render matching the audit's; the cached scoring routine against a plain
forward pass (needs torch and transformers; skipped where they are absent); the analysis on planted data, one data
set per pre-stated reading; the Holm family; the cross-GPU comparison; the run's refusals.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from wf import bridge as B
from wf import bridge_analysis as A
from tests.test_rq1 import _item, mock_root, mock_score  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


# ---------------- sample ----------------

def _pop(n=4666, seed=1):
    rng = random.Random(seed)
    reasons = ["away_from_receptacle", "contents_or_open_state", "hands_full", "not_holding_object"]
    out = []
    for k in range(n):
        hl = rng.choice([0, 0, 1, 2, 5, 6, 9, 11])
        out.append({"id": f"t_{k:06d}", "game_id": f"valid_unseen/{k % 125}", "history_len": hl,
                    "determination_reason": "away_from_receptacle_never_moved" if hl == 0 else rng.choice(reasons)})
    return out


def test_sample_is_proportional_deterministic_and_nested():
    pop = _pop()
    s = B.sample(pop)
    assert len(s) == 1500 and len({t["id"] for t in s}) == 1500
    assert [t["id"] for t in B.sample(list(reversed(pop)))] == [t["id"] for t in s]
    have, full = Counter(B.stratum(t) for t in s), Counter(B.stratum(t) for t in pop)
    for k, v in full.items():
        assert abs(have[k] - 1500 * v / len(pop)) < 1.0
    sets = B.item_sets(pop)
    ids = {t["id"] for t in sets["sample"]}
    assert len(sets["subset"]) == 300 and {t["id"] for t in sets["subset"]} <= ids
    assert [t["id"] for t in sets["smoke"]] == [t["id"] for t in sets["subset"][:20]]


# ---------------- prompts ----------------

def test_statements_sit_between_the_action_and_the_question_and_tasks_differ_only_in_the_task_line(mock_score):
    t = _item()
    for arm in B.ARMS:
        for ending in ("obs", "yn", "c1", "c2", "exist_check"):
            a, b = B.messages(t, "aligned", arm, ending), B.messages(t, "swapped", arm, ending)
            assert a[:-1] == b[:-1]
            la, lb = B.task_line(t, "aligned", arm), B.task_line(t, "swapped", arm)
            assert a[-1]["content"].replace(la, lb) == b[-1]["content"]
            text = a[-1]["content"]
            assert text.count("Action: take saltshaker 1 from cabinet 1") == 1
            if arm in B.STATEMENTS:
                s = B.STATEMENTS[arm]
                assert text.index("Action: take") < text.index(s)
                tail = text[text.index(s) + len(s):]
                assert tail.startswith("\n\n") and "Action:" not in tail
            else:
                assert not any(s in text for s in B.STATEMENTS.values())
            want_end = {"obs": "Next observation:", "yn": B.YN_QUESTION, "c1": B.C1, "c2": B.C2,
                        "exist_check": "Question: is there a saltshaker in the room? Answer Yes or No.\nAnswer:"}[ending]
            assert text.endswith(want_end) and text.count("Next observation:") == (ending == "obs")
    assert B.task_line(t, "swapped", "exist").endswith(" " + B.EXIST_UNNAMED)
    assert B.task_line(t, "aligned", "named") == t["task_line"] + " There is a saltshaker somewhere in the room."
    assert "saltshaker" not in B.EXIST_UNNAMED and "saltshaker" not in "".join(B.STATEMENTS.values())


def test_plain_text_matches_the_audit_render(mock_score):
    spec = importlib.util.spec_from_file_location("fu", ROOT / "scripts" / "qwen_audit" / "followup_score.py")
    fu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fu)
    t = _item()
    for arm in B.ARMS:
        msgs = B.messages(t, "aligned", arm)
        assert B.plain_text(msgs) == fu.plain_text(msgs)


def test_jobs_cover_every_arm_and_readout_and_comprehension_only_on_the_subset():
    t = _item()
    full, base = B.jobs_for(t, True), B.jobs_for(t, False)
    assert len(set(base)) == len(base) == 2 * (len(B.PROV_ARMS) * 2 + 1 + 3 + 2)
    extra = set(full) - set(base)
    assert extra == {("aligned", a, q) for a in B.COMP_ARMS for q in ("c1", "c2")}
    for task in B.TASKS:
        for arm in B.ARMS:
            assert (task, arm, "cand") in base and (task, arm, "yn") in base
    assert {B.group_of(task, arm) for task, arm, _ in base} == {(k, g) for k in B.TASKS for g in ("prov", "exist", "named")}


def test_stated_answers_and_gate_membership():
    for w in ("tester_a", "tester_b", "agent_a", "agent_b"):
        assert B.STATED[w] == {"c1": "No", "c2": "Yes"} and w in B.GATED
    assert B.STATED["positive"] == {"c1": "Yes", "c2": "No"} and "positive" not in B.GATED
    assert set(B.WORDINGS["tester"]) | set(B.WORDINGS["agent"]) == set(B.GATED)


# ---------------- the cached routine against a plain forward pass ----------------

def _tiny(kind):
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config

    torch.manual_seed(0)
    common = dict(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=256)
    cfg = LlamaConfig(**common) if kind == "llama" else Qwen3Config(head_dim=8, **common)
    return AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).eval()


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
def test_prefix_scorer_equals_a_full_forward_pass(kind):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from wf.scoring.prefix import PrefixScorer, full_pass_score

    model = _tiny(kind)
    rng = random.Random(3)
    stem = [rng.randrange(97) for _ in range(40)]
    seqs = []
    for k in range(6):
        p = stem + [rng.randrange(97) for _ in range(rng.randrange(0, 9))]
        c = [rng.randrange(97) for _ in range(rng.randrange(0, 5))]
        seqs.append((p, c, [rng.randrange(97), rng.randrange(97)]))
    seqs.append((list(stem), [5, 6], [1]))  # a prompt that is exactly the shared prefix
    got = PrefixScorer(model).score_group(seqs)
    for (p, c, pr), g in zip(seqs, got):
        ref = full_pass_score(model, p, c, pr)
        assert len(g.cont_logprobs) == len(c) and len(g.probe_logprobs) == len(pr)
        for x, y in zip(g.cont_logprobs + g.probe_logprobs, ref.cont_logprobs + ref.probe_logprobs):
            assert abs(x - y) < 1e-4
    # order does not matter: the cache is cut back to the stem before every sequence
    again = PrefixScorer(model).score_group(list(reversed(seqs)))
    for g, h in zip(got, reversed(again)):
        assert max(abs(x - y) for x, y in zip(g.cont_logprobs + g.probe_logprobs, h.cont_logprobs + h.probe_logprobs)) < 1e-5
    single = PrefixScorer(model).score_group([seqs[0]])[0]
    ref = full_pass_score(model, *seqs[0])
    assert max(abs(x - y) for x, y in zip(single.cont_logprobs + single.probe_logprobs,
                                          ref.cont_logprobs + ref.probe_logprobs)) < 1e-4


# ---------------- analysis on planted data ----------------

def _records(n=600, games=60, model="qwen3-8b", fmt="plain", effect=3.0, under=None, pc=3.0, comp_ok=True,
             exist_ok=True, exist_effect=None, named_effect=0.3, noise=0.3, seed=0, subset_ids=None):
    """Scoring records as the runner writes them. `under` maps an arm to its goal effect (default: the
    no-statement effect); the aligned and swapped levels are -5 + effect/2 and -5 - effect/2 plus noise."""
    rng = random.Random(seed)
    under = {**{a: effect for a in B.ARMS}, "exist": exist_effect if exist_effect is not None else effect,
             "named": named_effect, **(under or {})}
    recs = []
    for k in range(n):
        tid, gid = f"t_{k:06d}", f"g{k % games}"
        common = {"transition_id": tid, "game_id": gid, "model": model, "format": fmt, "item_hash": "x"}
        item_noise = rng.gauss(0, noise)
        for task in B.TASKS:
            for arm in B.ARMS:
                g = under[arm] + item_noise
                level = -5 + (g / 2 if task == "aligned" else -g / 2) + rng.gauss(0, noise)
                if arm == "positive":
                    level += pc
                for ro in ("cand", "yn"):
                    rec = {**common, "job": [task, arm, ro]}
                    if ro == "cand":
                        rec.update({"L": level, "L_first": level / 2, "first_mass": 0.9})
                    else:
                        rec.update({"L": level, "mass": 0.99})
                    recs.append(rec)
            for arm in ("none", "exist"):
                ok = exist_ok or rng.random() < 0.1
                recs.append({**common, "job": [task, arm, "exist_check"], "L": 2.0 if (arm == "exist" and ok) else -1.0})
            if subset_ids is None or tid in subset_ids:
                if task == "aligned":
                    for arm in B.COMP_ARMS:
                        for q in ("c1", "c2"):
                            want = B.STATED.get(arm, {}).get(q, "No")
                            right = comp_ok or rng.random() < 0.5
                            L = (1.5 if want == "Yes" else -1.5) * (1 if right else -1)
                            recs.append({**common, "job": [task, arm, q], "L": L})
    return recs


def _analyse(model_recs: dict):
    ids = sorted({r["transition_id"] for recs in model_recs.values() for r in recs})
    games = {r["transition_id"]: r["game_id"] for recs in model_recs.values() for r in recs}
    items = [{"id": i, "game_id": games[i], "history_len": int(i[-1]) % 9} for i in ids]
    boot = A.Boot([t["game_id"] for t in items])
    tests, blocks = {}, {}
    for model, recs in model_recs.items():
        v = A.load_values(recs)
        comp = A.comprehension(A.Values(v, ids))
        blocks[model] = A.model_block(model, A.Values(v, ids), boot, comp, items, tests)
    A.apply_holm(blocks, tests)
    return blocks, tests


def _both(**kw):
    return {"qwen3-8b": _records(model="qwen3-8b", **kw), "llama3.1-8b": _records(model="llama3.1-8b", seed=1, **kw)}


def test_provenance_carries_the_effect():
    blocks, tests = _analyse(_both(under={w: 0.2 for w in B.GATED}))
    assert len(tests) == 18
    for model in blocks:
        for ro in A.READOUTS:
            assert blocks[model][ro]["tester"]["category"] == "removes most"
            assert blocks[model][ro]["agent"]["category"] == "removes most"


def test_provenance_removes_nothing():
    blocks, _ = _analyse(_both(noise=0.05))
    for model in blocks:
        assert blocks[model]["cand"]["tester"]["category"].startswith("removes nothing")


def test_failed_comprehension_or_positive_control_makes_the_null_unreadable():
    blocks, _ = _analyse(_both(comp_ok=False))
    assert blocks["qwen3-8b"]["cand"]["tester"]["category"].startswith("not readable: comprehension")
    blocks, _ = _analyse(_both(pc=0.0))
    assert blocks["llama3.1-8b"]["yn"]["agent"]["category"].startswith("not readable: positive control")


def test_placebo_is_the_reference():
    # the content-free sentence removes as much as the statements: provenance removes nothing beyond it
    blocks, _ = _analyse(_both(under={**{w: 0.2 for w in B.GATED}, "placebo": 0.2}, noise=0.05))
    assert blocks["llama3.1-8b"]["cand"]["tester"]["category"].startswith("removes nothing")
    assert blocks["llama3.1-8b"]["cand"]["placebo_removal"]["estimate"] > 2.5


def test_existence_readings():
    blocks, _ = _analyse(_both(exist_effect=0.1))
    assert blocks["qwen3-8b"]["first"]["existence"]["category"] == "removes most"
    blocks, _ = _analyse(_both(exist_ok=False))
    assert blocks["qwen3-8b"]["first"]["existence"]["category"].startswith("not assessable")


def test_boot_and_permutation_on_known_data():
    import numpy as np

    games = [f"g{k % 50}" for k in range(500)]
    b = A.Boot(games)
    x = np.ones(500)
    m = b.mean(x)
    assert m["estimate"] == 1 and abs(m["ci_low"] - 1) < 1e-9 and abs(m["ci_high"] - 1) < 1e-9
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 500)
    assert b.perm_p(z + 1.0) < 0.001 and b.perm_p(z - z.mean()) > 0.5
    assert abs(b.ratio(2 * x, x)["estimate"] - 2) < 1e-12
    assert A.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_cross_gpu_comparison_flags_a_shift():
    recs = _records(model="llama3.1-8b", n=300, games=40)
    v = A.load_values(recs)
    ids = sorted({r["transition_id"] for r in recs})
    items = [{"id": i, "game_id": f"g{int(i[2:]) % 40}", "history_len": 3} for i in ids]
    comp = A.comprehension(A.Values(v, ids))
    same = A.rescore_comparison("llama3.1-8b", v, v, items, comp)
    assert not same["any_environment_dependent"]
    shifted = [dict(r, L=r["L"] + (0.5 if r["job"][0] == "aligned" and r["job"][1] == "tester_a" else 0.0)) for r in recs]
    moved = A.rescore_comparison("llama3.1-8b", v, A.load_values(shifted), items, comp)
    assert moved["comparison"]["cand/tester"]["environment_dependent"]


def test_duplicate_records_are_refused():
    recs = _records(n=5, games=5)
    with pytest.raises(ValueError):
        A.load_values(recs + recs[:1])


# ---------------- runs ----------------

def test_configs_pin_gpu_types_and_formats():
    import yaml

    cfg = lambda name: yaml.safe_load((ROOT / "wf" / "config" / f"{name}.yaml").read_text())
    for tag in A.MODELS:
        assert cfg(f"bridge_score_{tag}")["gpu_type"] == "L40S" and cfg(f"bridge_smoke_{tag}")["gpu_type"] == "L40S"
        assert cfg(f"bridge_rescore_{tag}")["gpu_type"] == "A100"
        assert cfg(f"bridge_rescore_{tag}")["formats"] == {A.PRIMARY_FORMAT[tag]: "sample"}
        assert A.PRIMARY_FORMAT[tag] in cfg(f"bridge_score_{tag}")["formats"]
    an = cfg("bridge_analysis")
    assert an["main_gpu"] == "L40S" and an["rescore_gpu"] == "A100"
    assert set(an["score_runs"]) == set(an["rescore_runs"]) == set(A.MODELS)


def test_end_to_end_mock(mock_root):
    for tag in A.MODELS:
        base = {"model": "Qwen/Qwen3-8B" if tag == "qwen3-8b" else "meta-llama/Llama-3.1-8B-Instruct", "tag": tag,
                "sources": ["rq12_transitions_smoke"], "gpu_type": "L40S"}
        fm = {"plain": "sample", "block": "subset"} if tag == "qwen3-8b" else {"chat": "sample"}
        for i in range(2):
            assert B.run_bridge_score({**base, "run_id": f"bridge_score_{tag}", "formats": fm, "items": "sample",
                                       "n_shards": 2, "shard": i}) == 0
        assert B.run_bridge_score({**base, "run_id": f"bridge_rescore_{tag}", "gpu_type": "A100", "n_shards": 1,
                                   "formats": {A.PRIMARY_FORMAT[tag]: "sample"}, "items": "subset"}) == 0
    acfg = {"run_id": "bridge_analysis", "sources": ["rq12_transitions_smoke"], "main_gpu": "L40S", "rescore_gpu": "A100",
            "score_runs": {t: f"bridge_score_{t}" for t in A.MODELS},
            "rescore_runs": {t: f"bridge_rescore_{t}" for t in A.MODELS}}
    assert A.run_bridge_analysis(acfg) == 0
    m = json.loads((mock_root / "bridge_analysis" / "metrics.json").read_text())
    assert len(m["holm_adjusted_p"]) == 18
    assert set(m["models"]) == set(A.MODELS) and "format_link_qwen_subset" in m
    assert set(m["cross_gpu"]) == set(A.MODELS)


def test_analysis_refuses_one_gpu_type_for_both_runs(mock_root):
    acfg = {"run_id": "bridge_analysis", "sources": ["rq12_transitions_smoke"], "main_gpu": "L40S",
            "rescore_gpu": "L40S", "score_runs": {}, "rescore_runs": {}}
    assert A.run_bridge_analysis(acfg) == 1
