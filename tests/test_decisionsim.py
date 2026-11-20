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


def _three():
    """The agent proposed "go to shelf 1" (works); two generated: a failing goal action and a working one."""
    cands = [_t("p", "go to shelf 1", True), _t("f", "take saltshaker 1 from cabinet 1", False, goal=True),
             _t("w", "examine drawer 1", True)]
    return _state(cands, "go to shelf 1")


def test_rule1_takes_the_largest_score_and_breaks_ties_by_action():
    s = _three()
    assert D.rule_choice("rule1_with_proposal", s, {"p": 1.0, "f": 2.0, "w": 0.5})["id"] == "f"
    # tie: "examine drawer 1" sorts before "go to shelf 1" and "take ..."
    assert D.rule_choice("rule1_with_proposal", s, {"p": 1.0, "f": 1.0, "w": 1.0})["id"] == "w"
    assert D.rule_choice("rule1_generated_only", s, {"p": 9.0, "f": 2.0, "w": 0.5})["id"] == "f"


def test_rule2_accepts_a_proposal_only_above_zero_and_falls_back_to_the_others():
    s = _three()
    assert D.rule_choice("rule2_verification", s, {"p": 0.1, "f": 5.0, "w": 0.0})["id"] == "p"
    assert D.rule_choice("rule2_verification", s, {"p": 0.0, "f": -5.0, "w": -6.0})["id"] == "f"  # 0 is not Yes
    assert D.rule_choice("rule2_verification", s, {"p": -1.0, "f": -5.0, "w": 3.0})["id"] == "w"
    # the fallback never returns the rejected proposal, even when it scores highest
    assert D.rule_choice("rule2_verification", s, {"p": -0.5, "f": -5.0, "w": -6.0})["id"] == "f"


def test_rule3_reads_goal_relevant_candidates_where_one_of_them_works():
    s = _three()
    assert not D.rule_applies("rule3_goal_relevant", s)
    s2 = _state(s["candidates"] + [_t("g", "take saltshaker 1 from shelf 1", True, goal=True)], "go to shelf 1")
    assert D.rule_applies("rule3_goal_relevant", s2)
    assert D.rule_choice("rule3_goal_relevant", s2, {"p": 9, "f": 1, "w": 9, "g": 2})["id"] == "g"


# ---------------- measures ----------------


def _contrast_data():
    """Six states in three games. Rule 1 choices, aligned / neutral:
    g0: f / w (harm), f / f (same), g1: w / f (help), f / w (harm), g2: w / w, p / w (changed, both succeed)."""
    L_a, L_n, states = {}, {}, []
    plan = [("g0", "f", "w"), ("g0", "f", "f"), ("g1", "w", "f"), ("g1", "f", "w"), ("g2", "w", "w"), ("g2", "p", "w")]
    for k, (g, ca, cn) in enumerate(plan):
        s = _three()
        cands = [{**t, "id": f"{t['id']}{k}", "game_id": g, "state_idx": k} for t in s["candidates"]]
        states.append(_state(cands, "go to shelf 1", game=g, step=k))
        for t in cands:
            L_a[t["id"]] = 2.0 if t["id"] == f"{ca}{k}" else -1.0
            L_n[t["id"]] = 2.0 if t["id"] == f"{cn}{k}" else -1.0
    return states, L_a, L_n


def test_contrast_counts_harm_help_and_the_identity():
    states, L_a, L_n = _contrast_data()
    c = D.contrast("rule1_with_proposal", states, L_a, L_n)
    assert c["n_states"] == 6 and c["n_games"] == 3
    assert c["fail_rate_a"]["estimate"] == pytest.approx(3 / 6) and c["fail_rate_b"]["estimate"] == pytest.approx(2 / 6)
    assert c["fail_rate_difference"]["estimate"] == pytest.approx(1 / 6)
    assert c["change_rate"]["estimate"] == pytest.approx(4 / 6)
    ch = c["changed"]
    assert (ch["n"], ch["a_fails_b_succeeds"], ch["a_succeeds_b_fails"], ch["both_fail"], ch["both_succeed"]) == (4, 2, 1, 0, 1)
    assert ch["share_a_fails_b_succeeds"] == pytest.approx(0.5)
    assert c["identity_holds"]
    assert c["random_choice_fail_rate"] == pytest.approx(1 / 3)
    assert c["determined_share_of_chosen_failures"]["a"] == pytest.approx(1.0)
    assert c["ties"] == {"a": 0, "b": 0}
    lo, hi = c["fail_rate_difference"]["ci"]
    assert lo <= 1 / 6 <= hi and 0 < c["fail_rate_difference"]["p"] <= 1


def test_condition_order_matters():
    states, L_a, L_n = _contrast_data()
    ab = D.contrast("rule1_with_proposal", states, L_a, L_n)
    ba = D.contrast("rule1_with_proposal", states, L_n, L_a)
    assert ba["fail_rate_difference"]["estimate"] == pytest.approx(-ab["fail_rate_difference"]["estimate"])
    assert ba["changed"]["a_fails_b_succeeds"] == ab["changed"]["a_succeeds_b_fails"]


def test_permutation_flips_whole_games():
    """Twenty games, each with one harmful and one helpful change in a different state order: the per-game sums are
    zero, so a game-level flip gives p = 1; a state-level flip of the same numbers would not."""
    states, L_a, L_n = [], {}, {}
    for g in range(20):
        for k, (ca, cn) in enumerate((("f", "w"), ("w", "f"))):
            s = _three()
            cands = [{**t, "id": f"{t['id']}{g}_{k}", "game_id": f"g{g}", "state_idx": k} for t in s["candidates"]]
            states.append(_state(cands, "go to shelf 1", game=f"g{g}", step=k))
            for t in cands:
                L_a[t["id"]] = 2.0 if t["id"] == f"{ca}{g}_{k}" else -1.0
                L_n[t["id"]] = 2.0 if t["id"] == f"{cn}{g}_{k}" else -1.0
    c = D.contrast("rule1_with_proposal", states, L_a, L_n)
    assert c["fail_rate_difference"]["estimate"] == 0 and c["fail_rate_difference"]["p"] == 1.0
    assert c["change_rate"]["estimate"] == 1.0
    # four games of ten harmful changes each: only 2 of the 16 game-level sign patterns reach the observed mean,
    # so p is about 1/8; flipping the 40 states one by one would give p near 1/1001
    states, L_a, L_n = [], {}, {}
    for g in range(4):
        for k in range(10):
            s = _three()
            cands = [{**t, "id": f"{t['id']}{g}_{k}", "game_id": f"g{g}", "state_idx": k} for t in s["candidates"]]
            states.append(_state(cands, "go to shelf 1", game=f"g{g}", step=k))
            for t in cands:
                L_a[t["id"]] = 2.0 if t["id"] == f"f{g}_{k}" else -1.0
                L_n[t["id"]] = 2.0 if t["id"] == f"w{g}_{k}" else -1.0
    p = D.contrast("rule1_with_proposal", states, L_a, L_n)["fail_rate_difference"]["p"]
    assert 0.05 < p < 0.25


def test_game_bootstrap_resamples_games_not_states():
    """One game holds every failure difference: resampling games gives an interval that reaches zero."""
    states, L_a, L_n = [], {}, {}
    for g in range(10):
        for k in range(10 if g == 0 else 1):
            s = _three()
            cands = [{**t, "id": f"{t['id']}{g}_{k}", "game_id": f"g{g}", "state_idx": k} for t in s["candidates"]]
            states.append(_state(cands, "go to shelf 1", game=f"g{g}", step=k))
            for t in cands:
                L_a[t["id"]] = 2.0 if t["id"] == (f"f{g}_{k}" if g == 0 else f"w{g}_{k}") else -1.0
                L_n[t["id"]] = 2.0 if t["id"] == f"w{g}_{k}" else -1.0
    c = D.contrast("rule1_with_proposal", states, L_a, L_n)
    assert c["fail_rate_difference"]["estimate"] == pytest.approx(10 / 19)
    assert c["fail_rate_difference"]["ci"][0] == 0.0


def test_verification_error_rates():
    ok_p = _three()
    bad = _state([_t("p2", "go to shelf 1", False, step=1), _t("w2", "look", True, step=1)], "go to shelf 1", step=1)
    La = {"p": -1.0, "f": 0, "w": 0, "p2": 3.0, "w2": 1.0}  # aligned: rejects the good one, accepts the bad one
    Ln = {"p": 1.0, "f": 0, "w": 0, "p2": -3.0, "w2": 1.0}
    e = D.verification_errors([ok_p, bad], La, Ln)
    assert e["false_acceptance"]["n_states"] == 1 and e["false_acceptance"]["rate_a"]["estimate"] == 1.0
    assert e["false_acceptance"]["rate_b"]["estimate"] == 0.0 and e["false_acceptance"]["difference"]["estimate"] == 1.0
    assert e["false_rejection"]["rate_a"]["estimate"] == 1.0 and e["false_rejection"]["rate_b"]["estimate"] == 0.0


def test_baseline_reads_yes_as_a_positive_score():
    s = _three()
    s["candidates"].append(_t("u", "take saltshaker 1 from drawer 9", False, det=False))
    b = D.baseline([s], {"p": 1.0, "f": 0.0, "w": -2.0, "u": 4.0})
    assert b["n_candidates"] == 4
    assert b["accuracy"] == pytest.approx(2 / 4)  # p right, f right (0 is No), w wrong, u wrong
    assert b["yes_rate_failures"] == pytest.approx(0.5) and b["yes_rate_determined_failures"] == 0.0
    assert b["yes_rate_successes"] == pytest.approx(0.5)


def test_model_report_adds_swapped_contrasts_only_where_scored():
    states, L_a, L_n = _contrast_data()
    rep = D.model_report(states, {"yn_aligned": L_a, "yn_absent": L_n})
    assert set(rep) == {"baseline", "yn_aligned_vs_yn_absent"}
    assert set(rep["yn_aligned_vs_yn_absent"]) == set(D.RULES) | {"rule2_errors"}
    states[0]["swap"] = None
    rep = D.model_report(states, {"yn_aligned": L_a, "yn_absent": L_n, "yn_swapped": L_n})
    assert rep["yn_aligned_vs_yn_swapped"]["rule1_with_proposal"]["n_states"] == 5
    assert rep["yn_aligned_vs_yn_absent"]["rule1_with_proposal"]["n_states"] == 6


def test_complete_states_drop_a_state_with_any_unscored_candidate():
    states, L_a, L_n = _contrast_data()
    L_n = dict(L_n)
    del L_n["w3"]
    keep, short = D.complete_states(states, {"yn_aligned": L_a, "yn_absent": L_n})
    assert short == 1 and [s["state_idx"] for s in keep] == [0, 1, 2, 4, 5]
    nan = [{"transition_id": "a", "condition": "yn_aligned", "L": math.nan}, {"transition_id": "b", "condition": "yn_aligned", "L": 1.0}]
    assert D.scores_by_arm(nan) == {"yn_aligned": {"b": 1.0}}


def _cell(change, lo):
    return {"change_rate": {"estimate": change}, "fail_rate_difference": {"ci": [lo, 1.0]}}


def _rep(c1, c2):
    return {"yn_aligned_vs_yn_absent": {"rule1_with_proposal": c1, "rule2_verification": c2}}


@pytest.mark.parametrize("qwen, llama, verdict", [
    (_rep(_cell(0.01, -0.1), _cell(0.02, -0.1)), _rep(_cell(0.05, 0.001), _cell(0.0, -0.1)), "worth doing"),
    (_rep(_cell(0.01, -0.1), _cell(0.02, -0.1)), _rep(_cell(0.20, 0.0), _cell(0.0, -0.1)), "weak case"),
    (_rep(_cell(0.049, 0.2), _cell(0.02, 0.1)), _rep(_cell(0.01, 0.1), _cell(0.0, 0.1)), "not worth doing for these models"),
])
def test_option_c_rule(qwen, llama, verdict):
    assert D.option_c({"qwen3-8b": qwen, "llama3.1-8b": llama})["verdict"] == verdict
    assert D.option_c({"qwen3-8b": qwen})["verdict"] == "waiting for a primary model"
    # a secondary model alone never decides it
    assert D.option_c({"qwen3-8b": _rep(_cell(0, 0), _cell(0, 0)), "llama3.1-8b": _rep(_cell(0, 0), _cell(0, 0)),
                       "mistral-7b-v0.3": _rep(_cell(0.5, 0.3), _cell(0.5, 0.3))})["verdict"] == "not worth doing for these models"


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
        assert cfg["model"] == tag and cfg["n_states"] == 300 and cfg["arms"] == ["yn_aligned", "yn_absent"]
    ana = yaml.safe_load((root / "decisionsim_analysis.yaml").read_text())
    assert set(ana["models"]) == set(D.MODEL_TAGS) and ana["n_states"] == 300
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
    assert D.run_decisionsim_score(_score_cfg("llama3.1-8b")) == 0
    cfg = {"run_id": "decisionsim_analysis", "sources": ["rq12_transitions_smoke"], "n_states": "all",
           "models": {"qwen3-8b": "decisionsim_score_qwen3-8b", "llama3.1-8b": "decisionsim_score_llama3.1-8b",
                      "mistral-7b-v0.3": "decisionsim_score_mistral-7b-v0.3"}}
    assert D.run_decisionsim_analysis(cfg) == 0
    m = json.loads((mock_root / "decisionsim_analysis" / "metrics.json").read_text())
    assert set(m["models"]) == {"qwen3-8b", "llama3.1-8b"} and "mistral-7b-v0.3" in m["waiting"]
    q = m["models"]["qwen3-8b"]
    assert q["n_states_incomplete"] == 0 and "yn_aligned_vs_yn_swapped" in q
    assert "yn_aligned_vs_yn_swapped" not in m["models"]["llama3.1-8b"]
    assert q["yn_aligned_vs_yn_absent"]["rule1_with_proposal"]["n_states"] == m["states"]["n_sampled"] > 0
    assert m["option_c"]["verdict"] in ("worth doing", "weak case", "not worth doing for these models")
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
