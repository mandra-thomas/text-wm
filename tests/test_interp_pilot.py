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
    words = sorted(set(re.findall(SPLIT, "\n".join(texts) + " system user assistant Yes No " + " ".join(OBJS))))
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
    plus, minus = PR.readout_ids(pilot_tok, "yn")
    logged = {}
    from wf.interp.hooks import forward_last

    for t in FAILS + SUCCS:
        for goal in ("aligned", "swapped", "absent"):
            e = PR.encode(pilot_tok, PR.messages(t, "yn", goal))
            _, h = forward_last(q, torch.tensor([e.ids]))
            logged[(t["id"], f"yn_{goal}")] = float(q.readout(h, plus, minus)[0])
    monkeypatch.setattr(PI, "logged_scores", lambda cfg: dict(logged))
    return {"root": tmp_path, "logged": logged, "q": q, "tok": pilot_tok}


def pilot_tok_of(env):
    return env["tok"]


BASE = {"skip_source_check": True, "min_free_gb": 0, "batch_size": 8}


def test_stage0_gate_fails_on_a_shifted_log(pilot_env, monkeypatch):
    shifted = {k: v + (0.6 if k[1] == "yn_aligned" else 0.0) for k, v in pilot_env["logged"].items()}
    monkeypatch.setattr(PI, "logged_scores", lambda cfg: shifted)
    assert PI.run_interp_pilot_stage0({**BASE, "run_id": "interp_pilot_stage0"}) == 1
    status = (pilot_env["root"] / "interp_pilot_stage0" / "STATUS.md").read_text()
    assert "FAIL" in status.splitlines()[0] and "gate FAILED" in status


def test_minimal_pilot_end_to_end(pilot_env):
    root = pilot_env["root"]
    assert PI.run_interp_pilot_stage0({**BASE, "run_id": "interp_pilot_stage0"}) == 0
    m0 = json.loads((root / "interp_pilot_stage0" / "metrics.json").read_text())
    assert m0["gate"]["passed"] and m0["gate"]["median_abs_diff"] < 1e-4
    man = json.loads((root / "interp_pilot_stage0" / "manifest.json").read_text())
    assert man["n_layers"] == 4 and man["batch_noise_nats"] < 1e-4
    recs = PI._stage0_store({}).records()
    assert {r["goal"] for r in recs} == {"aligned", "swapped", "absent", "swapped_plus"}
    assert not any(r["goal"] == "swapped_plus" and r["id"].startswith("s") for r in recs)

    cfg_p = {**BASE, "run_id": "interp_pilot_patch", "n_items": 4, "n_placebo": 2, "min_effect": -1e9,
             "batch_size": 16}
    assert PI.run_interp_pilot_patch(cfg_p) == 0
    pm = json.loads((root / "interp_pilot_patch" / "manifest.json").read_text())
    fold = PI.folds(FAILS + SUCCS, 0)
    assert len(pm["items"]) == 4 and all(fold[t["game_id"]] == 0 for t in FAILS if t["id"] in pm["items"])
    assert 1 <= len(pm["placebo"]) <= 2 and pm["n_cells"] == 4 * len(PR.PILOT_GROUPS)
    patch = PI.read_jsonl(root / "interp_pilot_patch" / "patch.jsonl")
    assert len(patch) == 2 * 4 + 2 * len(pm["placebo"])
    for r in patch:
        if r["source_goal"] == "swapped" and r["direction"] == "noise":
            assert r["cells"]["resid|3|last"] == pytest.approx(r["d_source"], abs=1e-4)  # the identity holds

    cfg_a = {**BASE, "run_id": "interp_pilot_ablate", "layers": [2, 3], "n_failures_per_fold": 3,
             "n_successes_per_fold": 1, "batch_size": 16}
    assert PI.run_interp_pilot_ablate(cfg_a) == 0
    am = json.loads((root / "interp_pilot_ablate" / "manifest.json").read_text())
    for f, ids in am["items"].items():  # every item is edited with directions from the other fold's games
        assert all(fold[t["game_id"]] == int(f) for t in FAILS + SUCCS if t["id"] in ids)
    ab = PI.read_jsonl(root / "interp_pilot_ablate" / "ablate.jsonl")
    names = set(ab[0]["edits"])
    assert {"task|2", "task_perp_success|2", "mention|3", "steer_success|3", "task_from|2"} <= names
    assert any(n.startswith("random|") for n in names) and all(r["positions"] == "last" for r in ab)
    dz = np.load(root / "interp_pilot_ablate" / "directions.npz")
    assert dz["0|2|task"].shape == (64,) and abs(float(dz["0|2|task_perp_success"] @ dz["0|2|success"])) < 1e-9
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
    assert set(rep["patching"]["noise"]) == {"step0", "later", "all"}
    assert rep["patching"]["noise"]["step0"]["history_share"]["defined"] is False
    assert rep["patching"]["noise"]["later"]["history_share"]["defined"] is True
    assert rep["ablation"]["chosen_layer_by_fold"]["0"] in (1, 2, 3)
    assert "task" in rep["ablation"]["all"] and "gap_neutral" in rep["ablation"]["all"]
    assert rep["verdicts"]["step0"]["history_share"] is None
    assert "hidden_probe" in rep
    assert not math.isnan(rep["ablation"]["all"]["task"]["share"])
