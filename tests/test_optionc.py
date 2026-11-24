"""Option C (wf/optionc.py, notes/optionc_design.md): the verifier prompts and readouts, the decision rules, the
closed-loop episode on the simulator with stand-in models, the cache, the runs end to end on the mock agent and scorer
(resume, the smoke gate, the observation readout's gate), the step-0 check and the analysis."""
from __future__ import annotations

import json
import math
import re
import types

import pytest

from tests.test_families import FakeTok
from tests.test_rq1 import MOCK_KEYS
from wf import decisionsim_agent as A
from wf import families, sizes
from wf import optionc as O
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.server import AlfworldClient
from wf.rq1v import YN_QUESTION
from wf.scoring import logodds
from wf.scoring.logodds import LogoddsScorer
from wf.util import read_jsonl

TASK = "Your task is to: put some saltshaker on cabinet."
FIRST = "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a shelf 1.\n" + TASK
HIST = FIRST + "\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a mug 1."


@pytest.fixture(autouse=True)
def _fewer_resamples(monkeypatch):
    """Tests only: 200 bootstrap resamples and permutations instead of 2,000 and 1,000, to keep the file fast; the
    analysis itself is unchanged."""
    from wf.analysis import stats

    monkeypatch.setattr(A, "cluster_bootstrap", lambda f, items, ids, seed=0: stats.cluster_bootstrap(f, items, ids, B=200,
                                                                                                     seed=seed))
    monkeypatch.setattr(A, "pair_permutation", lambda d, seed=0, cluster_ids=None: stats.pair_permutation(
        d, n=200, seed=seed, cluster_ids=cluster_ids))


@pytest.fixture
def mock_score(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    logodds._fewshot_hidden.cache_clear()
    yield
    logodds._fewshot_hidden.cache_clear()


def _query(msgs):
    return msgs[-1]["content"]


# ---------------- prompts ----------------


@pytest.mark.parametrize("tag", O.MODEL_TAGS)
@pytest.mark.parametrize("readout", O.READOUTS)
def test_arms_differ_only_in_the_query_task_line(mock_score, tag, readout):
    item = O.verifier_item("g", 1, TASK, HIST, "take saltshaker 1 from shelf 1", "mug")
    p = {a: O.verifier_messages(tag, item, a, readout) for a in O.VERIFIER_ARMS}
    aware = _query(p[O.AWARE])
    assert TASK in aware
    assert _query(p[O.BLIND]) == aware.replace("\n" + TASK, "", 1) and TASK not in _query(p[O.BLIND])
    assert _query(p[O.SWAPPED]) == aware.replace(TASK, TASK.replace("saltshaker", "mug"))
    assert _query(p[O.NEUTRAL]) == aware.replace(TASK, "Your task is to: explore the room.")
    for a in O.VERIFIER_ARMS:
        assert p[a][:-1] == p[O.AWARE][:-1]  # system text and few-shot examples identical in every arm
    end = YN_QUESTION if readout == O.YN else "Next observation:"
    assert all(_query(m).endswith(end) for m in p.values())


def test_prompts_are_the_earlier_prompts(mock_score):
    item = O.verifier_item("g", 1, TASK, HIST, "go to cabinet 1", "mug")
    assert O.verifier_messages("qwen3-8b", item, O.AWARE, O.YN) == sizes.arm_messages(item, "yn_aligned")
    assert O.verifier_messages("qwen3-8b", item, O.SWAPPED, O.YN) == sizes.arm_messages(item, "yn_swapped")
    assert O.verifier_messages("qwen3-8b", item, O.NEUTRAL, O.OBS) == sizes.arm_messages(item, "absent")
    assert O.verifier_messages("llama3.1-8b", item, O.AWARE, O.OBS) == sizes.arm_messages(item, "aligned")
    g = families.MODELS["gemma2-9b-it"]
    assert O.verifier_messages("gemma2-9b-it", item, O.AWARE, O.YN) == families.family_messages(g, item, "yn_aligned")


def test_swapped_arm_needs_a_swap_object_and_unknown_arms_are_refused(mock_score):
    item = O.verifier_item("g", 1, TASK, HIST, "go to cabinet 1", None)
    with pytest.raises(ValueError):
        O.verifier_messages("qwen3-8b", item, O.SWAPPED, O.YN)
    with pytest.raises(ValueError):
        O.verifier_messages("qwen3-8b", item, O.NONE, O.YN)
    with pytest.raises(ValueError):
        O.verifier_messages("qwen3-8b", item, O.AWARE, "free")


def test_swapped_line_replaces_only_the_goal_object():
    item = O.verifier_item("g", 0, TASK, FIRST, "look", "mug")
    assert item["swapped_task_line"] == "Your task is to: put some mug on cabinet."
    assert item["goal_object"] == "saltshaker"


# ---------------- readouts ----------------


class _Fixed:
    mock = False

    def __init__(self, yes=0.0, no=0.0, nothing=-1.0, not_nothing=-0.5, you=-2.0):
        self.v = dict(yes=yes, no=no, nothing=nothing, not_nothing=not_nothing, you=you)
        self.asked = []

    def next_token_logprobs(self, messages, words, complement=None):
        self.asked.append((words, complement))
        return {"Yes": self.v["yes"], "No": self.v["no"], "Nothing": self.v["nothing"], "You": self.v["you"],
                "not:Nothing": self.v["not_nothing"]}


def test_yes_no_readout_is_the_log_odds_rounded_to_a_64th():
    f = _Fixed(yes=-0.3, no=-1.01)
    r = O.read_score(f, [], O.YN)
    assert f.asked == [(("Yes", "No"), None)]
    assert r["L_raw"] == pytest.approx(0.71) and r["L"] == round(0.71 * 64) / 64
    with pytest.raises(ValueError):
        O.read_score(_Fixed(yes=float("-inf")), [], O.YN)


def test_observation_readout_is_the_complement_of_nothing():
    s = _Fixed(nothing=math.log(0.2), not_nothing=math.log(0.8), you=math.log(0.5))
    r = O.read_score(s, [], O.OBS)
    assert s.asked == [(("Nothing", "You"), "Nothing")]
    assert r["L_raw"] == pytest.approx(math.log(4)) and r["L_you"] == pytest.approx(math.log(2.5))
    assert r["L"] == round(math.log(4) * 64) / 64


def test_observation_readout_refuses_a_non_finite_score():
    with pytest.raises(ValueError):
        O.read_score(_Fixed(nothing=0.0, not_nothing=float("-inf")), [], O.OBS)


def test_rounding_decides_acceptance():
    assert not O.accepts(O.quantize(1 / 200)) and O.accepts(O.quantize(1 / 100)) and not O.accepts(0.0)


def test_mock_next_token_logprobs_complement_is_consistent(mock_score):
    sc = LogoddsScorer("Qwen/Qwen3-8B", model_tag="qwen3-8b")
    lp = sc.next_token_logprobs([{"role": "user", "content": "x"}], ("Nothing", "You"), complement="Nothing")
    assert math.exp(lp["Nothing"]) + math.exp(lp["not:Nothing"]) == pytest.approx(1.0)


def test_first_token_gate_passes_and_catches_split_tokens():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "History: x Action: y Next observation:"}]
    good = LogoddsScorer.for_tokenizer("Qwen/Qwen3-8B", FakeTok(), model_tag="qwen3-8b")
    assert O.first_token_gate(good, [msgs])["ok"]
    bad = LogoddsScorer.for_tokenizer("Qwen/Qwen3-8B", FakeTok(first_same=True), model_tag="qwen3-8b")
    g = O.first_token_gate(bad, [msgs])
    assert not g["ok"] and any("not one token" in p for p in g["problems"])
    odd = LogoddsScorer.for_tokenizer("Qwen/Qwen3-8B", FakeTok(odd_reply_after=(("History", "Nothing"),)),
                                      model_tag="qwen3-8b")
    assert not O.first_token_gate(odd, [msgs])["ok"]
    assert not O.first_token_gate(good, [])["ok"]  # nothing gated is not a pass


# ---------------- decision rules ----------------


def _decide(rule, scores, alts, ties=None):
    calls = {"alts": 0, "scored": []}

    def score(a):
        calls["scored"].append(a)
        return scores[a]

    def alternatives():
        calls["alts"] += 1
        return list(alts)

    return O.decide(rule, "p", score, alternatives, (ties or {}).get if ties else (lambda a: a)), calls


@pytest.mark.parametrize("rule", O.RULES)
def test_an_accepted_proposal_is_executed_without_sampling(rule):
    d, calls = _decide(rule, {"p": 0.5, "a": 9.0}, ["a"])
    assert d["chosen"] == "p" and d["reason"] == "accepted" and calls["alts"] == 0 and calls["scored"] == ["p"]


def test_accepted_only_takes_the_first_accepted_in_rank_order_not_the_best():
    d, calls = _decide(O.ACCEPTED_ONLY, {"p": -1.0, "a": -2.0, "b": 0.5, "c": 9.0}, ["a", "b", "c"])
    assert d["chosen"] == "b" and d["reason"] == "accepted_alternative"
    assert calls["scored"] == ["p", "a", "b"]  # stops at the first accepted


def test_accepted_only_keeps_the_proposal_when_nothing_is_accepted():
    d, _ = _decide(O.ACCEPTED_ONLY, {"p": -1.0, "a": -0.5, "b": 0.0}, ["a", "b"])
    assert d["chosen"] == "p" and d["reason"] == "abstained" and d["alt_L"] == {"a": -0.5, "b": 0.0}


@pytest.mark.parametrize("rule", O.RULES)
def test_no_alternative_keeps_the_proposal(rule):
    d, _ = _decide(rule, {"p": -1.0}, [])
    assert d["chosen"] == "p" and d["reason"] == "no_alternative"


def test_argmax_takes_the_highest_alternative_whatever_its_sign():
    d, calls = _decide(O.ARGMAX, {"p": -1.0, "a": -5.0, "b": -0.25, "c": -3.0}, ["a", "b", "c"])
    assert d["chosen"] == "b" and d["reason"] == "argmax" and calls["scored"] == ["p", "a", "b", "c"]


def test_argmax_ties_go_to_the_smallest_seeded_hash():
    d, _ = _decide(O.ARGMAX, {"p": -1.0, "a": -0.5, "b": -0.5}, ["a", "b"], ties={"a": "f", "b": "0"})
    assert d["chosen"] == "b"
    d, _ = _decide(O.ARGMAX, {"p": -1.0, "a": -0.5, "b": -0.5}, ["a", "b"], ties={"a": "0", "b": "f"})
    assert d["chosen"] == "a"


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        _decide("best", {"p": 1.0}, [])


def test_sample_seed_depends_on_seed_game_and_prompt_only():
    s = O.sample_seed(1, "g", "abc")
    assert s == O.sample_seed(1, "g", "abc") != O.sample_seed(2, "g", "abc")
    assert s != O.sample_seed(1, "h", "abc") and s != O.sample_seed(1, "g", "abd")
    assert s == int(__import__("hashlib").sha256(b"20260924|optionc|1|g|abc").hexdigest()[:8], 16)


# ---------------- the cache ----------------


def test_cache_makes_once_saves_and_reloads(tmp_path):
    p = tmp_path / "cache.jsonl"
    c = O.GameCache(p, "g")
    n = {"k": 0}

    def make():
        n["k"] += 1
        return f"v{n['k']}"

    assert c.get("greedy", "x", make) == "v1" and c.get("greedy", "x", make) == "v1" and n["k"] == 1
    assert c.hits["greedy"] == 1 and c.misses["greedy"] == 1
    rows = read_jsonl(p)
    # append_jsonl marks rows {"mock": true} in mock mode, which an earlier test module can leave on
    assert [{k: v for k, v in r.items() if k != "mock"} for r in rows] == [
        {"game_id": "g", "kind": "greedy", "key": "x", "value": "v1"}]
    again = O.GameCache(None, "g", rows + [{"game_id": "h", "kind": "greedy", "key": "y", "value": "z"}])
    assert again.get("greedy", "x", make) == "v1" and "y" not in again.data["greedy"] and n["k"] == 1


# ---------------- episodes on the simulator ----------------


def _action_in(msgs):
    return re.search(r"\nAction: (.*)\n", _query(msgs)).group(1)


class FakeAgent:
    """The mock RQ3 agent, counting calls."""

    def __init__(self):
        from wf.agent.react import AgentClient

        self.inner = AgentClient()
        self.calls = []

    def complete(self, msgs, max_tokens=256):
        self.calls.append(msgs)
        return self.inner.complete(msgs, max_tokens)


class FakeSampler:
    def __init__(self, actions=("look", "inventory", "go to shelf 1")):
        self.actions, self.calls = list(actions), []

    def sample(self, msgs, n, params, max_tokens, seed):
        self.calls.append({"msgs": msgs, "n": n, "params": params, "max_tokens": max_tokens, "seed": seed})
        acts = (self.actions * n)[:n]
        return [{"text": f"Thought: t\nAction: {a}", "finish_reason": "stop"} for a in acts]


class FakeVerifier:
    """yes/no scores from a function of (action, whether the real task line is in the query)."""

    mock = False

    def __init__(self, fn):
        self.fn, self.prompts = fn, []

    def next_token_logprobs(self, messages, words, complement=None):
        self.prompts.append(messages)
        q = _query(messages)
        hist = q.split("History:\n", 1)[1].split("\n\nAction: ", 1)[0]
        L = self.fn(_action_in(messages), hist)
        return {"Yes": L, "No": 0.0}


@pytest.fixture
def sim(monkeypatch):
    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT"):
        monkeypatch.setenv(key, "1")
    logodds._fewshot_hidden.cache_clear()
    client = AlfworldClient()
    client.create()
    yield client
    logodds._fewshot_hidden.cache_clear()


def _models(fn=lambda a, h: 1.0, sampler=None, readout=O.YN):
    return O.Models(FakeAgent(), sampler or FakeSampler(), FakeVerifier(fn), "qwen3-8b", readout)


def _play(client, models, arm, rule=O.ACCEPTED_ONLY, seed=1, cache=None, swap="plate", game="train/1", cap=8):
    return O.play_episode(client, models, game, "train", arm, rule, seed, cache or O.GameCache(None, game), swap, cap)


def test_no_verifier_executes_the_greedy_proposal(sim):
    m = _models()
    ep = _play(sim, m, O.NONE)
    assert all(s["chosen"] == s["proposal"] and s["reason"] == "no_verifier" for s in ep["trace"])
    assert ep["rule"] is None and ep["seed"] is None and len(m.agent.calls) == ep["steps"]
    assert m.scorer.prompts == [] and m.sampler.calls == []
    fails = [s["real_obs"].strip() == O.NOTHING for s in ep["trace"]]
    assert any(fails) and [s["failed"] for s in ep["trace"]] == fails and ep["n_invalid"] == sum(fails)


def test_an_accepting_verifier_reproduces_the_agent_and_never_samples(sim):
    base = _play(sim, _models(), O.NONE)
    m = _models(lambda a, h: 3.0)
    ep = _play(sim, m, O.AWARE)
    assert [s["chosen"] for s in ep["trace"]] == [s["chosen"] for s in base["trace"]]
    assert ep["success"] == base["success"] and m.sampler.calls == []


def test_a_rejecting_verifier_with_nothing_accepted_abstains(sim):
    base = _play(sim, _models(), O.NONE)
    m = _models(lambda a, h: -3.0)
    ep = _play(sim, m, O.BLIND)
    assert [s["chosen"] for s in ep["trace"]] == [s["chosen"] for s in base["trace"]]
    assert all(s["reason"] == "abstained" for s in ep["trace"]) and len(m.sampler.calls) == ep["steps"]
    assert ep["n_rejected"] == ep["steps"] == ep["longest_rejection_run"] and ep["n_abstained"] == ep["steps"]


def test_sampling_uses_the_agent_prompt_and_the_designed_settings(sim):
    m = _models(lambda a, h: -3.0)
    ep = _play(sim, m, O.AWARE, cap=2)
    for call, t, msgs in zip(m.sampler.calls, ep["trace"], m.agent.calls):
        assert call["msgs"] == msgs and call["n"] == 20 and call["max_tokens"] == 256
        assert call["params"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
        assert call["seed"] == O.sample_seed(1, "train/1", t["prompt_sha"])


def test_override_records_the_proposals_true_outcome(sim):
    m = _models(lambda a, h: 2.0 if a == "inventory" else -2.0, FakeSampler(("inventory",)))
    ep = _play(sim, m, O.AWARE, cap=4)
    assert ep["n_override"] == ep["steps"] and all(s["chosen"] == "inventory" for s in ep["trace"])
    prefix = []
    for s in ep["trace"]:
        h = replay(sim.sandbox(), "train/1", list(prefix), split="train")
        obs, _ = step_counterfactual(h, s["proposal"])
        assert s["proposal_obs"] == obs and s["proposal_failed"] == (obs.strip() == O.NOTHING)
        assert s["failed"] == (s["real_obs"].strip() == O.NOTHING)
        h = replay(sim.sandbox(), "train/1", list(prefix), split="train")
        executed, _ = step_counterfactual(h, s["chosen"])
        assert s["real_obs"] == executed  # the live step ran the chosen action, not the proposal
        prefix.append(s["chosen"])
    assert ep["step0_override"] and ep["step0_rejected"]
    assert ep["n_override_harm"] == sum((not s["proposal_failed"]) and s["failed"] for s in ep["trace"])


def test_the_agent_never_sees_the_verifier(sim):
    m = _models(lambda a, h: 2.0 if a == "inventory" else -2.0, FakeSampler(("inventory",)))
    ep = _play(sim, m, O.AWARE, cap=5)
    for t, msgs in enumerate(m.agent.calls):
        body = _query(msgs)
        assert "Expected outcome" not in body and "Proposed action" not in body
        assert [ln[len("Action: "):] for ln in body.splitlines() if ln.startswith("Action: ")] == \
               [s["chosen"] for s in ep["trace"][:t]]


def test_the_verifier_reads_the_world_model_history_of_the_live_state(sim):
    seen = []
    m = _models(lambda a, h: seen.append(h) or 1.0)
    ep = _play(sim, m, O.AWARE, cap=4)
    state = sim.reset("train/1", "train")
    for k, s in enumerate(ep["trace"]):
        assert seen[k] == history_text_from_state(state)
        state = sim.step(state, s["chosen"])


def test_the_arm_decides_the_task_line_the_verifier_sees(sim):
    task = sim.reset("train/1", "train").task_line
    for arm, want in ((O.AWARE, task), (O.BLIND, None), (O.NEUTRAL, "Your task is to: explore the room.")):
        lines = []
        m = _models(lambda a, h: lines.append([x for x in h.splitlines() if x.startswith("Your task")]) or 1.0)
        _play(sim, m, arm, cap=2)
        assert all(x == ([want] if want else []) for x in lines)
    m = _models(lambda a, h: lines.append(h) or 1.0)
    lines = []
    _play(sim, m, O.SWAPPED, cap=1, swap="plate")
    goal = O.parse_task_line(task).goal_object
    assert O.make_swapped_task_line(task, goal, "plate") in lines[0]


def test_episode_argument_checks(sim):
    with pytest.raises(ValueError):
        _play(sim, _models(), O.SWAPPED, swap=None)
    with pytest.raises(ValueError):
        _play(sim, _models(), O.AWARE, rule=None)
    with pytest.raises(ValueError):
        _play(sim, _models(), "oracle")


def test_common_random_numbers_share_the_prefix_through_the_cache(sim):
    cache = O.GameCache(None, "train/1")
    # aware accepts the agent's proposals; blind rejects them and accepts only `inventory`
    m = _models(lambda a, h: 1.0 if ("Your task" in h) == (a != "inventory") else -1.0, FakeSampler(("inventory",)))
    a = _play(sim, m, O.AWARE, cache=cache)
    calls_after_aware = len(m.agent.calls)
    b = _play(sim, m, O.BLIND, cache=cache)
    k = next((i for i, (x, y) in enumerate(zip(a["trace"], b["trace"])) if x["chosen"] != y["chosen"]), None)
    assert k == 0  # blind rejects everything and takes `inventory` at once
    assert cache.hits["greedy"] >= 1  # the first state's greedy reply was reused, not asked again
    assert len(m.agent.calls) - calls_after_aware == b["steps"] - 1
    c = _play(sim, m, O.AWARE, cache=cache)
    assert O._comparable(c) == O._comparable(a)


def test_format_error_becomes_look_and_is_flagged(sim):
    m = _models()
    m.agent.complete = lambda msgs, max_tokens=256: "Thought: I am not sure."
    ep = _play(sim, m, O.NONE, cap=2)
    assert all(s["proposal"] == "look" and s["proposal_format_error"] for s in ep["trace"])
    assert ep["n_format_errors"] == 2


def test_alternatives_are_cut_at_k_max(sim):
    acts = [f"go to drawer {k}" for k in range(1, 13)]
    m = _models(lambda a, h: -1.0, FakeSampler(acts))
    ep = _play(sim, m, O.AWARE, cap=1)
    s = ep["trace"][0]
    assert len(s["candidates"]) == O.K_MAX and len(s["alternatives"]) == O.K_MAX - 1
    assert s["candidates"][0]["action"] == s["proposal"] and set(s["alt_L"]) == set(s["alternatives"])


def test_repeats_and_score(sim):
    m = _models(lambda a, h: 2.0 if a == "inventory" else -2.0, FakeSampler(("inventory",)))
    ep = _play(sim, m, O.AWARE, cap=4)
    assert [s["repeat"] for s in ep["trace"]] == [False, True, True, True]
    assert ep["n_repeats"] == 3 and ep["score_S"] == -5 and not ep["success"]  # cap 4: a loss scores -(4 + 1)


def test_episode_summary_classifies_overrides_and_runs():
    tr = [{"override": True, "proposal_failed": False, "failed": True, "proposal_accepted": False, "reason": "argmax",
           "repeat": False, "proposal_format_error": False},
          {"override": True, "proposal_failed": True, "failed": False, "proposal_accepted": False, "reason": "argmax",
           "repeat": False, "proposal_format_error": True},
          {"override": False, "proposal_failed": True, "failed": True, "proposal_accepted": False, "reason": "abstained",
           "repeat": True, "proposal_format_error": False},
          {"override": False, "proposal_failed": False, "failed": False, "proposal_accepted": True, "reason": "accepted",
           "repeat": False, "proposal_format_error": False},
          {"override": True, "proposal_failed": True, "failed": True, "proposal_accepted": False, "reason": "argmax",
           "repeat": False, "proposal_format_error": False}]  # both fail: neither harm nor help
    s = O.episode_summary({"trace": tr, "steps": 5, "success": True})
    assert (s["n_override"], s["n_override_harm"], s["n_override_help"]) == (3, 1, 1)
    assert (s["n_invalid"], s["n_rejected"], s["n_abstained"], s["n_repeats"]) == (3, 4, 1, 1)
    assert s["longest_rejection_run"] == 3 and s["score_S"] == -5 and s["n_format_errors"] == 1
    assert s["step0_override"] and s["step0_rejected"]
    assert O.episode_summary({"trace": tr, "steps": 5, "success": False}, step_cap=30)["score_S"] == -31


def test_episode_jobs_order():
    assert O.episode_jobs([O.AWARE, O.NONE, O.BLIND], [O.ACCEPTED_ONLY, O.ARGMAX], [1, 2]) == [
        (O.NONE, None, None),
        (O.BLIND, O.ACCEPTED_ONLY, 1), (O.AWARE, O.ACCEPTED_ONLY, 1),
        (O.BLIND, O.ACCEPTED_ONLY, 2), (O.AWARE, O.ACCEPTED_ONLY, 2),
        (O.BLIND, O.ARGMAX, 1), (O.AWARE, O.ARGMAX, 1), (O.BLIND, O.ARGMAX, 2), (O.AWARE, O.ARGMAX, 2)]
    assert O.episode_jobs([O.BLIND], [O.ARGMAX], [3]) == [(O.BLIND, O.ARGMAX, 3)]


# ---------------- runs end to end (mock agent, scorer and environment) ----------------


GAMES = ["train/1", "train/2", "train/3"]


@pytest.fixture
def mock_root(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    logodds._fewshot_hidden.cache_clear()
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    yield tmp_path
    logodds._fewshot_hidden.cache_clear()


def _episodes_cfg(run_id="oc", **kw):
    cfg = {"run_id": run_id, "model": "qwen3-8b", "readout": O.YN, "rules": [O.ACCEPTED_ONLY, O.ARGMAX],
           "arms": [O.NONE, O.BLIND, O.AWARE, O.SWAPPED], "seeds": [1, 2], "split": "train", "games": GAMES,
           "n_games": 3, "step_cap": 6, "n_shards": 1, "smoke": True}  # smoke: no order gate (tested on its own)
    return {**cfg, **kw}


def _eps(root, run="oc"):
    return sorted((e for p in sorted((root / run).glob("shard_*")) for e in read_jsonl(p / "episodes.jsonl")),
                  key=lambda e: (e["game_id"], O.ARMS.index(e["arm"]), e["rule"] or "", e["seed"] or 0))


def test_run_plays_every_episode_once_and_reports(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg()) == 0
    eps = _eps(mock_root)
    want = {(g, a, r, s) for g in GAMES for a, r, s in O.episode_jobs([O.NONE, O.BLIND, O.AWARE, O.SWAPPED],
                                                                         [O.ACCEPTED_ONLY, O.ARGMAX], [1, 2])}
    assert {O.episode_key(e) for e in eps} == want and len(eps) == len(want)
    d = mock_root / "oc" / "shard_00_of_01"
    assert "REPORTED (shard complete)" in (d / "STATUS.md").read_text().splitlines()[0]
    m = json.loads((d / "metrics.json").read_text())
    assert m["n_episodes"] == len(want)
    assert {r["game_id"] for r in read_jsonl(d / "games_done.jsonl")} == set(GAMES)
    assert all(e["swap_object"] for e in eps if e["arm"] == O.SWAPPED)


def test_shards_split_the_games(mock_root):
    for i in (0, 1):
        assert O.run_optionc_episodes(_episodes_cfg(n_shards=2, shard=i, seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    shards = [json.loads((mock_root / "oc" / f"shard_0{i}_of_02" / "shard.json").read_text())["games"] for i in (0, 1)]
    assert sorted(shards[0] + shards[1]) == sorted(GAMES) and not set(shards[0]) & set(shards[1])
    assert shards[0] == sorted(GAMES)[0::2]


def test_partial_job_resumes_to_the_same_episodes(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg(run_id="full")) == 0
    assert O.run_optionc_episodes(_episodes_cfg(run_id="cut", max_episodes_per_job=5)) == 1
    assert "PARTIAL" in (mock_root / "cut" / "shard_00_of_01" / "STATUS.md").read_text()
    assert len(_eps(mock_root, "cut")) == 5
    for _ in range(10):
        if O.run_optionc_episodes(_episodes_cfg(run_id="cut", max_episodes_per_job=5, resume=True)) == 0:
            break
    strip = lambda es: [{k: v for k, v in O._comparable(e).items()} for e in es]
    assert strip(_eps(mock_root, "cut")) == strip(_eps(mock_root, "full"))


def test_a_fresh_run_starts_a_new_cache_and_episodes(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    d = mock_root / "oc" / "shard_00_of_01"
    (d / "cache.jsonl").write_text(json.dumps({"game_id": "train/1", "kind": "greedy", "key": "x", "value": "y"}) + "\n")
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    assert not any(r["key"] == "x" for r in read_jsonl(d / "cache.jsonl"))
    assert len(_eps(mock_root)) == len(GAMES) * 4


def test_smoke_cache_check_and_gate(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg(cache_check=True, seeds=[1])) == 0
    m = json.loads((mock_root / "oc" / "shard_00_of_01" / "metrics.json").read_text())
    assert m["cache_check"]["ok"] and m["cache_check"]["episodes"] == 7 and m["smoke_gate"]["passed"]


def test_cache_check_catches_a_cache_that_does_not_reproduce(mock_root, monkeypatch):
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    d = mock_root / "oc" / "shard_00_of_01"
    rows = [r for r in read_jsonl(d / "cache.jsonl") if not (r["game_id"] == GAMES[0] and r["kind"] == "score")]
    (d / "cache.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    eps = _eps(mock_root)
    res = O.cache_check(AlfworldClient(), eps, GAMES[0], "train", "qwen3-8b", O.YN, d / "cache.jsonl", 6, 20, 8)
    assert not res["ok"] and res["errors"]
    eps[0]["trace"][0]["chosen"] = "flip"
    full = O.cache_check(AlfworldClient(), eps, GAMES[0], "train", "qwen3-8b", O.YN, mock_root / "nothing.jsonl", 6, 20, 8)
    assert not full["ok"]


def test_smoke_gate_reads_format_errors_and_the_cache_check():
    book = {"format_error_share": 0.049, "cache_check": {"ok": True}, "n_episodes": 3}
    assert O.smoke_gate(book)["passed"]
    assert not O.smoke_gate({**book, "format_error_share": 0.05})["passed"]
    assert not O.smoke_gate({**book, "cache_check": {"ok": False}})["passed"]
    assert not O.smoke_gate({**book, "n_episodes": 0})["passed"]


def test_games_without_a_swap_object_skip_only_the_swapped_arm(mock_root, monkeypatch):
    real = O.game_swap_object
    monkeypatch.setattr(O, "game_swap_object", lambda c, g, s, t: None if g == "train/2" else real(c, g, s, t))
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    eps = _eps(mock_root)
    assert not any(e["game_id"] == "train/2" and e["arm"] == O.SWAPPED for e in eps)
    assert sum(e["game_id"] == "train/2" for e in eps) == 3
    eps_loaded, problems = O.load_run("oc")
    assert not problems and len(eps_loaded) == len(eps)


def test_bad_configs_are_refused(mock_root):
    for bad in ({"arms": ["none", "oracle"]}, {"arms": ["aware", "aware"]}, {"model": "mistral-7b-v0.3"},
                {"rules": ["best"]}, {"readout": "free"}, {"rules": []}):
        with pytest.raises(SystemExit):
            O.run_optionc_episodes(_episodes_cfg(**bad))


def test_observation_episodes_wait_for_the_step0_check(mock_root):
    cfg = _episodes_cfg(readout=O.OBS, seeds=[1], rules=[O.ACCEPTED_ONLY], arms=[O.BLIND, O.AWARE])
    assert O.run_optionc_episodes(cfg) == 1
    assert "wait for the step-0 check" in (mock_root / "oc" / "shard_00_of_01" / "STATUS.md").read_text()
    a = mock_root / O.STEP0_ANALYSIS
    a.mkdir()
    (a / "STATUS.md").write_text("# STATUS — x — REPORTED (step-0 check)\n")
    (a / "metrics.json").write_text(json.dumps({"models": {"qwen3-8b": {"obs_competence": {"passed": False}},
                                                           "llama3.1-8b": {"obs_competence": {"passed": True}}}}))
    assert O.run_optionc_episodes(cfg) == 1
    (a / "metrics.json").write_text(json.dumps({"models": {"qwen3-8b": {"obs_competence": {"passed": True}}}}))
    assert O.run_optionc_episodes(cfg) == 0
    gate = json.loads((mock_root / "oc" / "shard_00_of_01" / "obs_gate.json").read_text())
    assert gate["ok"] and gate["why"] is None
    eps = _eps(mock_root)
    assert eps and all(e["readout"] == O.OBS for e in eps)


def test_observation_gate_override_is_logged(mock_root):
    cfg = _episodes_cfg(readout=O.OBS, seeds=[1], rules=[O.ACCEPTED_ONLY], arms=[O.AWARE], obs_gate_override="test")
    assert O.run_optionc_episodes(cfg) == 0
    gate = json.loads((mock_root / "oc" / "shard_00_of_01" / "obs_gate.json").read_text())
    assert gate["override"] == "test" and gate["would_have_refused"]


def test_mock_yes_no_accepts_some_and_rejects_some():
    vals = [O._mock_yn([{"role": "user", "content": f"x{i}"}]) for i in range(60)]
    assert any(v > 0 for v in vals) and any(v < 0 for v in vals) and all(abs(v) >= 2 for v in vals)


# ---------------- analysis ----------------


FAST = ("success", "n_invalid")  # the verdict's outcomes only, to keep the bootstraps few


def _ep(g, arm, rule=O.ACCEPTED_ONLY, seed=1, success=True, n_invalid=0, trace=None, **kw):
    base = {o: 0 for o in O.OUTCOMES}
    base.update({"game_id": g, "arm": arm, "rule": None if arm == O.NONE else rule,
                 "seed": None if arm == O.NONE else seed, "success": success, "n_invalid": n_invalid, "steps": 10,
                 "score_S": -10 if success else -31, "trace": trace or [], "model": "qwen3-8b", "readout": "yn"})
    base.update(kw)
    return base


def _step(accepted=True, failed=False, goal=True, **kw):
    """A verified step whose proposal acts on the real goal object (what the obstruction check reads)."""
    return {"chosen": "x", "proposal": "x", "proposal_accepted": accepted, "proposal_failed": failed, "failed": failed,
            "proposal_targets_goal": goal, "chosen_targets_goal": goal, "reason": "accepted" if accepted else "abstained",
            "override": False, "repeat": False, **kw}


def _design(n=20, aware=None, blind=None, swapped=None, inv_aware=None, swapped_accepts=True, no_swap=()):
    """Per game success for each arm (functions of the game index). Every verifier episode has one working goal-object
    proposal, accepted (the swapped one only if `swapped_accepts`), so the obstruction check reads 'clear' by default."""
    eps = []
    for k in range(n):
        g = f"g{k:02d}"
        eps.append(_ep(g, O.NONE, success=True))
        eps.append(_ep(g, O.BLIND, success=(blind or (lambda k: True))(k), trace=[_step()]))
        eps.append(_ep(g, O.AWARE, success=(aware or (lambda k: True))(k),
                       n_invalid=(inv_aware or (lambda k: 0))(k), trace=[_step()]))
        if k not in no_swap:
            eps.append(_ep(g, O.SWAPPED, success=(swapped or (lambda k: True))(k), trace=[_step(swapped_accepts)]))
    return eps


def test_per_game_means_over_seeds():
    eps = [_ep("g", O.AWARE, seed=1, success=True), _ep("g", O.AWARE, seed=2, success=False),
           _ep("g", O.AWARE, seed=3, success=False), _ep("g", O.AWARE, rule=O.ARGMAX, success=True)]
    assert O.per_game(eps, O.AWARE, O.ACCEPTED_ONLY, "success") == {"g": pytest.approx(1 / 3)}
    assert O.per_game([_ep("g", O.NONE)], O.NONE, O.ARGMAX, "success") == {"g": 1.0}


def test_paired_contrast_by_game():
    eps = _design(aware=lambda k: k % 2 == 0)
    r = O.paired(eps, O.AWARE, O.BLIND, O.ACCEPTED_ONLY, "success")
    assert r["estimate"] == pytest.approx(-0.5) and r["n_games"] == 20 and r["ci"][1] < 0 and r["p"] < 0.05


def test_verdict_harm_attributable_when_the_swapped_verifier_recovers_the_cost():
    eps = _design(aware=lambda k: k % 4 == 0)  # aware loses 15 of 20; swapped never fails: it recovers all of the cost
    r = O.run_report(eps, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]
    assert r["decomposition"]["success"]["identity_holds"] and r["obstruction"]["state"] == "clear"
    assert r["verdict_success"] == "harm, attributable to the real task"


def test_verdict_harm_not_attributable_when_the_swapped_verifier_costs_as_much():
    eps = _design(aware=lambda k: k % 2 == 0, swapped=lambda k: k % 2 == 0)
    r = O.run_report(eps, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]
    assert r["decomposition"]["success"]["aware_minus_swapped"]["estimate"] == 0.0
    assert r["verdict_success"] == "harm, not attributable to the real task"


def test_verdict_not_identifiable_when_the_swapped_verifier_obstructs():
    eps = _design(aware=lambda k: k % 2 == 0, swapped=lambda k: k % 2 == 0, swapped_accepts=False)
    r = O.run_report(eps, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]
    assert r["obstruction"]["state"] == "obstructs"
    assert r["verdict_success"] == "harm, " + O.NOT_IDENTIFIABLE


def test_decomposition_uses_only_the_games_with_a_swap_object():
    eps = _design(aware=lambda k: k % 2 == 0, no_swap=(1,))
    r = O.run_report(eps, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]
    assert r["decomposition"]["success"]["n_games"] == 19
    assert r["contrasts"]["aware_minus_blind"]["success"]["n_games"] == 20


def test_verdict_help_and_no_difference():
    help_ = _design(blind=lambda k: k % 2 == 0)
    assert O.run_report(help_, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]["verdict_success"] == "help"
    flat = _design(aware=lambda k: k != 3)
    assert O.run_report(flat, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]["verdict_success"] == "no reliable difference"


def test_invalid_actions_verdict_has_the_reversed_sign():
    eps = _design(inv_aware=lambda k: 2)
    r = O.run_report(eps, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]
    assert r["contrasts"]["aware_minus_blind"]["n_invalid"]["estimate"] == pytest.approx(2.0)
    assert r["verdict_invalid"] == "harm, attributable to the real task"  # swapped has none, so all of it is the task's
    assert r["verdict_success"] == "no reliable difference"


def test_override_net_harm_is_a_key_secondary_where_more_is_worse():
    eps = _design()
    for e in eps:
        if e["arm"] == O.AWARE:
            e["override_net_harm"] = 2
    r = O.run_report(eps, outcomes=("success", "override_net_harm"))["rules"][O.ACCEPTED_ONLY]
    assert r["verdicts"]["override_net_harm"] == "harm, attributable to the real task"
    assert r["verdicts"]["success"] == "no reliable difference"


def _dec(ab, asw, ci, n=10):
    return {"n_games": n, "aware_minus_blind": {"estimate": ab}, "aware_minus_swapped": {"estimate": asw, "ci": ci}}


CLEAR = {"state": "clear"}


def test_attribution_categories():
    # success: aware costs 0.5 against blind; the threshold is a recovery of 0.25
    assert O.attribution(_dec(-0.5, -0.4, [-0.5, -0.3]), True, CLEAR) == "attributable to the real task"
    assert O.attribution(_dec(-0.5, -0.1, [-0.2, -0.05]), True, CLEAR) == "not attributable to the real task"
    assert O.attribution(_dec(-0.5, -0.2, [-0.4, 0.0]), True, CLEAR) == "attribution unresolved"  # spans 0.25
    assert O.attribution(_dec(-0.5, -0.25, [-0.25, -0.25]), True, CLEAR) == "attribution unresolved"  # exactly half
    assert O.attribution(_dec(-0.5, -0.4, [-0.5, -0.26]), True, CLEAR) == "attributable to the real task"
    assert O.attribution(_dec(-0.5, -0.1, [-0.24, 0.1]), True, CLEAR) == "not attributable to the real task"
    assert O.attribution(_dec(0.0, -0.3, [-0.4, -0.2]), True, CLEAR) == "attribution unresolved"  # no cost on the subset
    assert O.attribution(_dec(-0.5, -0.4, [float("nan")] * 2), True, CLEAR) == "attribution unresolved"
    # invalid actions (more is worse): aware adds 2, swapped recovers 1.6
    assert O.attribution(_dec(2.0, 1.6, [1.2, 2.0]), False, CLEAR) == "attributable to the real task"
    assert O.attribution(_dec(-0.5, -0.4, [-0.5, -0.3]), False, CLEAR) == "attribution unresolved"


def test_attribution_needs_a_clear_obstruction_check_and_a_swapped_arm():
    d = _dec(-0.5, -0.4, [-0.5, -0.3])
    for state in ("obstructs", "not computable", "not tested"):
        assert O.attribution(d, True, {"state": state}) == O.NOT_IDENTIFIABLE
    assert O.attribution(d, True, None) == O.NOT_IDENTIFIABLE
    assert O.attribution(None, True, CLEAR) == "attribution not tested (no swapped arm)"
    assert O.attribution(_dec(-0.5, -0.4, [-0.5, -0.3], n=0), True, CLEAR) == "attribution not tested (no swapped arm)"


def test_verdict_needs_both_interval_and_p():
    assert O.verdict({"ci": [-0.3, -0.1], "p": 0.06}, None) == "no reliable difference"
    assert O.verdict({"ci": [-0.3, 0.0], "p": 0.01}, None) == "no reliable difference"
    assert O.verdict({"ci": [-0.3, -0.01], "p": 0.01}, None) == "harm, attribution not tested (no swapped arm)"
    assert O.verdict({"ci": [0.01, 0.3], "p": 0.01}, None) == "help"
    assert O.verdict({"ci": [0.01, 0.3], "p": float("nan")}, None) == "no reliable difference"
    assert O.verdict({"ci": [0.01, 0.3], "p": 0.01}, None, higher_is_better=False).startswith("harm")


def test_obstruction_check_margin():
    def eps(aware_acc, swapped_acc, n=10):
        out = []
        for k in range(n):
            out.append(_ep(f"g{k}", O.AWARE, trace=[_step(k < aware_acc)]))
            out.append(_ep(f"g{k}", O.SWAPPED, trace=[_step(k < swapped_acc)]))
        return out
    assert O.obstruction_check(eps(10, 9), O.ACCEPTED_ONLY)["state"] == "clear"  # exactly 10 points: not more
    r = O.obstruction_check(eps(10, 8), O.ACCEPTED_ONLY)
    assert r["state"] == "obstructs" and r["working_goal_gap_aware_minus_swapped"] == pytest.approx(0.2)
    assert O.obstruction_check(eps(5, 10), O.ACCEPTED_ONLY)["state"] == "clear"
    only_aware = [e for e in eps(10, 10) if e["arm"] == O.AWARE]
    assert O.obstruction_check(only_aware, O.ACCEPTED_ONLY)["state"] == "not tested"
    no_goal = [_ep("g", O.AWARE, trace=[_step(goal=False)]), _ep("g", O.SWAPPED, trace=[_step(goal=False)])]
    assert O.obstruction_check(no_goal, O.ACCEPTED_ONLY)["state"] == "not computable"
    failing = O.obstruction_check([_ep("g", O.AWARE, trace=[_step(True, failed=True), _step(True)]),
                                   _ep("g", O.SWAPPED, trace=[_step(False, failed=True), _step(True)])], O.ACCEPTED_ONLY)
    assert failing["acceptance"][O.AWARE]["failing_goal"]["estimate"] == 1.0  # the in-loop naming lift
    assert failing["acceptance"][O.SWAPPED]["failing_goal"]["estimate"] == 0.0
    assert failing["state"] == "clear"  # failing proposals do not enter the obstruction gap


def test_the_primary_runs_no_verifier_episodes_serve_other_runs():
    eps = [e for e in _design(blind=lambda k: k % 2 == 0) if e["arm"] != O.NONE]
    none = [_ep(f"g{k:02d}", O.NONE, success=False) for k in range(20)]
    r = O.run_report(eps, none, outcomes=FAST)["rules"][O.ACCEPTED_ONLY]
    assert r["contrasts"]["blind_minus_none"]["success"]["estimate"] == pytest.approx(0.5)


def test_divergence():
    t = lambda acts: [{"chosen": a} for a in acts]
    eps = [_ep("a", O.AWARE, trace=t(["x", "y"])), _ep("a", O.BLIND, trace=t(["x", "z"])),
           _ep("b", O.AWARE, trace=t(["x"])), _ep("b", O.BLIND, trace=t(["x"])),
           _ep("c", O.AWARE, trace=t(["x"])), _ep("c", O.BLIND, trace=t(["x", "w"]))]
    d = O.divergence(eps, O.ACCEPTED_ONLY)
    assert d["n_pairs"] == 3 and d["never_diverge"] == 1 and d["median_first_divergence"] == 1


def test_step_rates_by_the_proposals_true_outcome():
    tr = [{"proposal_accepted": True, "proposal_failed": False, "proposal_targets_goal": False, "reason": "accepted",
           "failed": False},
          {"proposal_accepted": False, "proposal_failed": False, "proposal_targets_goal": False, "reason": "abstained",
           "failed": False},
          {"proposal_accepted": True, "proposal_failed": True, "proposal_targets_goal": True, "reason": "accepted",
           "failed": True}]
    r = O.step_rates([_ep("g", O.AWARE, trace=tr)], O.AWARE, O.ACCEPTED_ONLY)
    assert r["accept_working_proposal"] == 0.5 and r["accept_failing_proposal"] == 1.0
    assert r["accept_failing_goal_proposal"] == 1.0 and r["chosen_fails"] == pytest.approx(1 / 3)


def test_drift_against_the_archived_agent():
    none = [_ep("g1", O.NONE, trace=[{"chosen": "a"}, {"chosen": "b"}], success=True),
            _ep("g2", O.NONE, trace=[{"chosen": "a"}, {"chosen": "d"}], success=True),
            _ep("g3", O.NONE, trace=[{"chosen": "a"}], success=False)]
    old = [{"game_id": "g1", "condition": "none", "success": True, "trace": [{"executed": "a"}, {"executed": "b"}]},
           {"game_id": "g2", "condition": "none", "success": False, "trace": [{"executed": "a"}, {"executed": "c"}]},
           {"game_id": "g3", "condition": "none", "success": False, "trace": [{"executed": "z"}]},
           {"game_id": "g3", "condition": "oracle", "success": True, "trace": [{"executed": "a"}]}]
    d = O.drift(none, old)
    assert d["n_games"] == 3 and d["first_action_agreement"] == pytest.approx(2 / 3)
    assert d["reproduction_rate"] == pytest.approx(1 / 3) and d["n_diverged"] == 2
    assert d["first_divergence_quartiles"] == [0, 1, 1]
    assert d["success_archived"] == pytest.approx(1 / 3) and d["success_now"] == pytest.approx(2 / 3)
    assert d["success_now_minus_archived"]["estimate"] == pytest.approx(1 / 3)


def test_analysis_end_to_end_and_its_refusals(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1])) == 0
    assert O.run_optionc_analysis({"run_id": "oca", "runs": {"oc": "oc"}, "primary_run": "oc"}) == 0
    m = json.loads((mock_root / "oca" / "metrics.json").read_text())
    assert set(m["runs"]["oc"]["rules"]) == {O.ACCEPTED_ONLY, O.ARGMAX}
    assert m["primary"]["verdict_success"] in ("no reliable difference", "help", "harm, not attributable to the real task",
                                               "harm, attributable to the real task")
    p = mock_root / "oc" / "shard_00_of_01" / "episodes.jsonl"
    rows = read_jsonl(p)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    assert O.run_optionc_analysis({"run_id": "oca", "runs": {"oc": "oc"}, "primary_run": "oc"}) == 1
    assert "do not match the design" in (mock_root / "oca" / "STATUS.md").read_text()
    p.write_text("".join(json.dumps(r) + "\n" for r in rows + rows[-1:]))
    assert O.run_optionc_analysis({"run_id": "oca", "runs": {"oc": "oc"}, "primary_run": "oc"}) == 1


def test_analysis_waits_for_an_unfinished_shard(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY], max_episodes_per_job=3)) == 1
    assert O.run_optionc_analysis({"run_id": "oca", "runs": {"oc": "oc"}}) == 1
    assert "wait for every shard" in (mock_root / "oca" / "STATUS.md").read_text()


# ---------------- the step-0 check ----------------


def _t(i, action, ok, goal=False, game="g0"):
    return {"id": f"{game}#00#{i}", "game_id": game, "state_idx": 0, "task_line": TASK, "goal_object": "saltshaker",
            "history_text": FIRST, "action": action, "true_obs": "You arrive." if ok else O.NOTHING,
            "targets_goal_object": goal}


def _s0(game="g0", swap="mug"):
    cands = [_t(0, "go to shelf 1", True, game=game), _t(1, "go to cabinet 1", True, game=game),
             _t(2, "take saltshaker 1 from shelf 1", False, True, game=game), _t(3, "look", True, game=game)]
    return {"game_id": game, "state_idx": 0, "proposal": "go to shelf 1", "candidates": cands, "swap": swap}


def _recs(states, L):
    """L[(arm, readout)] = function of the candidate -> score."""
    return [{"transition_id": t["id"], "arm": a, "readout": r, "L": O.quantize(f(t)), "L_raw": f(t)}
            for s in states for t in s["candidates"] for (a, r), f in L.items()]


def _scenario(yn_aware, obs_aware, blind=lambda t: 5.0 if "Nothing" not in t["true_obs"] else -5.0):
    sw = {(O.SWAPPED, r): yn_aware for r in O.READOUTS}
    L = {(O.AWARE, O.YN): yn_aware, (O.AWARE, O.OBS): obs_aware, (O.BLIND, O.YN): blind, (O.BLIND, O.OBS): blind,
         (O.NEUTRAL, O.YN): blind, (O.NEUTRAL, O.OBS): blind, **sw}
    states = [_s0(f"g{k}") for k in range(6)]
    return states, _recs(states, L)


def test_step0_question_specific():
    states, recs = _scenario(lambda t: -5.0, lambda t: 5.0 if "Nothing" not in t["true_obs"] else -5.0)
    r = O.step0_report(states, recs)
    assert r["classification"] == "question-specific (supports the task-success reading)"
    assert r["gap_states"] == {O.YN: "collapse", O.OBS: "no gap"}
    assert r["proposal_gaps"][O.YN]["estimate"] == -1.0
    assert r["cells"][O.YN][O.AWARE]["working_proposals_accepted"]["estimate"] == 0.0
    assert r["cells"][O.YN][O.AWARE]["states_with_no_accepted_candidate"] == 1.0
    assert r["obs_competence"]["passed"]


def test_step0_question_specific_needs_no_absolute_acceptance():
    """Review 1's second region: the observation readout accepting 70% of working proposals under both arms is no
    gap, so question-specific (it was 'mixed')."""
    blind = lambda t: (5.0 if t["game_id"] not in ("g0", "g1") else -5.0) if "Nothing" not in t["true_obs"] else -5.0
    states, recs = _scenario(lambda t: -5.0, blind, blind=blind)
    for r in recs:  # the yes/no blind arm accepts every working proposal; only the observation arms reject g0, g1
        if r["readout"] == O.YN and r["arm"] == O.BLIND:
            t = next(c for s in states for c in s["candidates"] if c["id"] == r["transition_id"])
            r["L"] = 5.0 if "Nothing" not in t["true_obs"] else -5.0
    rep = O.step0_report(states, recs)
    assert rep["cells"][O.OBS][O.AWARE]["working_proposals_accepted"]["estimate"] == pytest.approx(4 / 6)
    assert rep["classification"] == "question-specific (supports the task-success reading)"


def test_step0_other_classifications():
    states, recs = _scenario(lambda t: -5.0, lambda t: -5.0)
    rep = O.step0_report(states, recs)
    assert rep["classification"] == "not question-specific"
    assert rep["obs_competence"]["passed"]  # competence is read under the blind arm, not the collapsed aware one
    states, recs = _scenario(lambda t: 5.0, lambda t: -5.0)
    assert O.step0_report(states, recs)["classification"] == "no collapse to explain"
    bad_blind = lambda t: -5.0 if "Nothing" not in t["true_obs"] else 5.0  # accepts failures, rejects successes
    states, recs = _scenario(lambda t: -5.0, lambda t: 5.0, blind=bad_blind)
    assert O.step0_report(states, recs)["classification"] == "not classifiable (the task-blind yes/no verifier is not competent)"


def _gap(lo, hi):
    return {"estimate": (lo + hi) / 2, "ci": [lo, hi]}


def test_gap_states_and_classification_at_their_boundaries():
    assert O.gap_state(_gap(-0.9, -0.31)) == "collapse" and O.gap_state(_gap(-0.9, -0.30)) == "unclear"
    assert O.gap_state(_gap(-0.10, 0.10)) == "no gap" and O.gap_state(_gap(-0.11, 0.05)) == "unclear"
    assert O.gap_state(_gap(-0.05, 0.11)) == "unclear" and O.gap_state(_gap(float("nan"), float("nan"))) == "unclear"
    ok = {O.YN: True, O.OBS: True}
    col, none_, mid = _gap(-0.9, -0.6), _gap(-0.05, 0.05), _gap(-0.4, -0.1)
    assert O.classify({O.YN: col, O.OBS: none_}, ok).startswith("question-specific")
    assert O.classify({O.YN: col, O.OBS: col}, ok) == "not question-specific"
    assert O.classify({O.YN: col, O.OBS: mid}, ok).startswith("mixed (the observation")
    assert O.classify({O.YN: mid, O.OBS: none_}, ok).startswith("mixed (the yes/no")
    assert O.classify({O.YN: none_, O.OBS: col}, ok) == "no collapse to explain"
    assert O.classify({O.YN: col, O.OBS: none_}, {O.YN: True, O.OBS: False}).startswith("not classifiable (the observation")
    assert O.classify({O.YN: col, O.OBS: none_}, {O.YN: False, O.OBS: True}).startswith("not classifiable (the task-blind")
    assert O.classify({O.YN: col}, ok).startswith("not classified")


def test_step0_competence_rule():
    good = {"auc": 0.70, "yes_failures": 0.95, "yes_successes": 0.05}
    assert O.competence(good)["passed"]
    for k, v in (("auc", 0.69), ("yes_failures", 0.96), ("yes_successes", 0.04)):
        assert not O.competence({**good, k: v})["passed"]


def test_step0_shifts_naming_and_any_task():
    states, recs = _scenario(lambda t: -5.0, lambda t: 5.0)
    for r in recs:  # naming: aware lifts the failing goal-object action by 3 over swapped
        if r["arm"] == O.AWARE and "take" in next(t["action"] for s in states for t in s["candidates"] if t["id"] == r["transition_id"]):
            r["L_raw"] += 3.0
    rep = O.step0_report(states, recs)
    assert rep["shifts"][O.YN]["naming_lift_aware_minus_swapped"]["estimate"] == pytest.approx(3.0)
    assert rep["shifts"][O.YN]["anytask_drop_aware_minus_blind"]["estimate"] == pytest.approx(-10.0)
    assert rep["shifts"][O.YN]["anytask_drop_swapped_minus_blind"]["estimate"] == pytest.approx(-10.0)


def test_stage1a_agreement():
    recs = [{"transition_id": "a", "arm": O.AWARE, "readout": O.YN, "L": 1.0},
            {"transition_id": "a", "arm": O.SWAPPED, "readout": O.YN, "L": 2.0},
            {"transition_id": "a", "arm": O.BLIND, "readout": O.YN, "L": 9.0},
            {"transition_id": "a", "arm": O.AWARE, "readout": O.OBS, "L": 9.0}]
    old = [{"transition_id": "a", "condition": "yn_aligned", "L": 1.001},
           {"transition_id": "a", "condition": "yn_swapped", "L": 2.5}]
    r = O.stage1a_agreement(recs, old)
    assert r == {"n_compared": 2, "max_abs_diff": 0.5, "n_differing": 1}


def test_step0_run_scores_every_candidate_arm_and_readout(mock_root, monkeypatch):
    import wf.runs
    import wf.tools.verify_items

    states = [_s0("train/1"), _s0("train/2", swap=None)]
    monkeypatch.setattr(O, "step0_states", lambda sources, steps=(0,), seed=O.SEED: states)
    monkeypatch.setattr(wf.runs, "_finished", lambda s, accept=(): True)
    monkeypatch.setattr(wf.tools.verify_items, "verified", lambda s: (True, ""))
    cfg = {"run_id": "s0", "model": "qwen3-8b", "sources": ["x"]}
    assert O.run_optionc_step0(cfg) == 0
    rows = read_jsonl(mock_root / "s0" / "scoring.jsonl")
    assert len(rows) == 4 * (4 + 3) * 2  # the state without a swap object has no swapped arm
    assert {(r["arm"], r["readout"]) for r in rows} == {(a, r) for a in O.VERIFIER_ARMS for r in O.READOUTS}
    assert O.run_optionc_step0({**cfg, "resume": True}) == 0
    assert len(read_jsonl(mock_root / "s0" / "scoring.jsonl")) == len(rows)
    p = mock_root / "s0" / "scoring.jsonl"
    p.write_text("".join(json.dumps({**r, "score_hash": "x"}) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        O.run_optionc_step0({**cfg, "resume": True})
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert O.run_optionc_step0_analysis({"run_id": "s0a", "sources": ["x"], "models": {"qwen3-8b": "s0"}}) == 1
    assert "waits for every scoring run" in (mock_root / "s0a" / "STATUS.md").read_text()  # the refused resume
    assert O.run_optionc_step0({**cfg, "resume": True}) == 0
    assert O.run_optionc_step0_analysis({"run_id": "s0a", "sources": ["x"], "models": {"qwen3-8b": "s0"}}) == 0
    m = json.loads((mock_root / "s0a" / "metrics.json").read_text())
    assert m["n_states"] == 2 and "classification" in m["models"]["qwen3-8b"]
    assert O.run_optionc_step0_analysis({"run_id": "s0b", "sources": ["x"], "models": {"llama3.1-8b": "nope"}}) == 1


def test_step0_waits_for_the_collection(mock_root, monkeypatch):
    import wf.runs
    import wf.tools.verify_items

    monkeypatch.setattr(wf.runs, "_finished", lambda s, accept=(): False)
    monkeypatch.setattr(wf.tools.verify_items, "verified", lambda s: (True, ""))
    assert O.run_optionc_step0({"run_id": "s0", "model": "qwen3-8b", "sources": ["x"]}) == 1


def test_step0_jobs_skip_the_swapped_arm_without_a_swap():
    jobs = O.step0_jobs([_s0(swap=None)], O.VERIFIER_ARMS, (O.YN,))
    assert {a for _, _, a, _ in jobs} == {O.AWARE, O.BLIND, O.NEUTRAL} and len(jobs) == 12


def test_registry_and_configs():
    from wf.runs import RUNNERS, load_config

    for name in ("optionc_smoke", "optionc_qwen3-8b_yn", "optionc_qwen3-8b_yn_argmax", "optionc_llama3.1-8b_yn",
                 "optionc_qwen3-8b_obs", "optionc_gemma2-9b-it_yn", "optionc_step0_qwen3-8b",
                 "optionc_step0_llama3.1-8b", "optionc_step0_analysis", "optionc_analysis", "optionc_smoke_analysis"):
        assert name in RUNNERS
    # Reallocation exercised 2026-11-22 (design note): the observation readout carries Tier 1's seeds and the
    # no-verifier arm and is the primary run; yes/no is the secondary at one seed per arm.
    tier1 = load_config("optionc_qwen3-8b_obs")
    assert (tier1["model"], tier1["readout"], tier1["rules"]) == ("qwen3-8b", "obs", ["accepted_only"])
    assert tier1["arms"] == ["none", "blind", "aware", "swapped"]
    assert tier1["seeds_by_arm"] == {"blind": [1, 2, 3], "aware": [1, 2, 3], "swapped": [1]}
    assert tier1["shadow_arms"] == ["blind", "swapped"] and tier1["cost_upper_gpu_hours"] == 58
    yn = load_config("optionc_qwen3-8b_yn")
    assert (yn["model"], yn["readout"], yn["rules"]) == ("qwen3-8b", "yn", ["accepted_only"])
    assert yn["arms"] == ["blind", "aware", "swapped"] and yn["cost_upper_gpu_hours"] == 28
    assert yn["seeds_by_arm"] == {"blind": [1], "aware": [1], "swapped": [1]} and yn["shadow_arms"] == ["blind", "swapped"]
    assert yn["n_games"] == 134 and yn["step_cap"] == 30 and yn["split"] == "valid_unseen"
    assert load_config("optionc_llama3.1-8b_yn")["tier"] == 3 and load_config("optionc_smoke")["smoke"] is True
    assert "optionc_memsmoke_gemma2-9b-it" in RUNNERS
    assert tier1["n_games"] == 134 and tier1["step_cap"] == 30 and tier1["split"] == "valid_unseen"
    assert load_config("optionc_qwen3-8b_yn_argmax")["rules"] == ["argmax"]
    an = load_config("optionc_analysis")
    assert an["primary_run"] == "optionc_qwen3-8b_obs"
    assert an["runs"] == {"optionc_qwen3-8b_obs": "optionc_qwen3-8b_obs", "optionc_qwen3-8b_yn": "optionc_qwen3-8b_yn",
                         "optionc_qwen3-8b_yn_argmax": "optionc_qwen3-8b_yn_argmax"}
    step0 = {"models": {"qwen3-8b": {"classification": "question-specific (supports the task-success reading)",
                                     "obs_competence": {"passed": True}}}}
    assert O.primary_problems(tier1, step0) == []
    assert O.primary_problems(tier1, {"models": {"qwen3-8b": {"classification": "mixed", "obs_competence": {"passed": True}}}})
    assert O.PRIMARY == {"model": "qwen3-8b", "readout": "yn", "rule": "accepted_only"}


# ---------------- added after review 1 ----------------


def test_shadow_scores_every_aware_step_and_never_acts(sim):
    # aware rejects the proposals and scores `inventory` (accepted); blind would accept them, swapped would not
    fn = lambda a, h: 2.0 if a == "inventory" else (-1.0 if "Your task is to: put" in h else 1.0)
    plain = _play(sim, _models(fn, FakeSampler(("inventory",))), O.AWARE, cap=4)
    m = _models(fn, FakeSampler(("inventory",)))
    ep = O.play_episode(sim, m, "train/1", "train", O.AWARE, O.ACCEPTED_ONLY, 1, O.GameCache(None, "train/1"), "plate",
                        4, shadow=O.SHADOW_ARMS)
    assert [s["chosen"] for s in ep["trace"]] == [s["chosen"] for s in plain["trace"]]
    for s in ep["trace"]:
        assert set(s["shadow"]) == {O.BLIND, O.SWAPPED}
        assert s["shadow"][O.BLIND]["proposal_L"] == 1.0 and s["shadow"][O.SWAPPED]["proposal_L"] == -1.0
        assert s["alt_L"] == {"inventory": 2.0} and s["shadow"][O.BLIND]["alt_L"] == {"inventory": 2.0}
    blind = O.play_episode(sim, m, "train/1", "train", O.BLIND, O.ACCEPTED_ONLY, 1, O.GameCache(None, "train/1"),
                           "plate", 2, shadow=O.SHADOW_ARMS)
    assert all("shadow" not in s for s in blind["trace"])
    noswap = O.play_episode(sim, m, "train/1", "train", O.AWARE, O.ACCEPTED_ONLY, 1, O.GameCache(None, "train/1"), None,
                            2, shadow=O.SHADOW_ARMS)
    assert all(set(s["shadow"]) == {O.BLIND} for s in noswap["trace"])


def test_matched_states_compare_prompts_on_the_same_steps():
    def st(fail, goal, own, blind, swapped, alts=None):
        return {"proposal_failed": fail, "proposal_targets_goal": goal, "proposal_L": own, "alt_L": alts or {},
                "shadow": {O.BLIND: {"proposal_L": blind, "alt_L": {a: -1.0 for a in (alts or {})}},
                           O.SWAPPED: {"proposal_L": swapped, "alt_L": {a: 1.0 for a in (alts or {})}}}}
    task = "Your task is to: put some apple on diningtable."
    eps = [_ep("g1", O.AWARE, task_line=task, trace=[st(True, True, 1.0, -1.0, -1.0), st(False, False, -1.0, 1.0, 1.0)]),
           _ep("g2", O.AWARE, task_line=task, trace=[st(False, True, 1.0, 1.0, -1.0, {"take apple 1 from x 1": 1.0,
                                                                                         "go to x 1": 1.0})]),
           _ep("g3", O.BLIND, task_line=task, trace=[st(True, True, 9.0, 9.0, 9.0)])]
    m = O.matched_states(eps, O.ACCEPTED_ONLY)
    assert m["n_steps"] == 3
    b, w = m[O.BLIND], m[O.SWAPPED]
    assert b["accept_failing_goal_proposal"]["aware"]["estimate"] == 1.0
    assert b["accept_failing_goal_proposal"][O.BLIND]["estimate"] == 0.0
    assert b["reject_working_proposal"]["aware"]["estimate"] == 0.5 and b["reject_working_proposal"][O.BLIND]["estimate"] == 0.0
    assert w["reject_working_goal_proposal"][O.SWAPPED]["estimate"] == 1.0
    assert w["reject_working_goal_proposal"]["aware_minus_swapped"]["estimate"] == -1.0
    assert b["accept_goal_alternative"]["estimate"] == 1.0 and b["accept_goal_alternative"]["n_states"] == 1
    assert w["accept_goal_alternative"]["estimate"] == 0.0


def test_seeds_by_arm_and_the_swapped_arm_at_one_seed(mock_root):
    by = {O.BLIND: [1, 2], O.AWARE: [1, 2], O.SWAPPED: [1]}
    assert O.episode_jobs([O.NONE, O.BLIND, O.AWARE, O.SWAPPED], [O.ACCEPTED_ONLY], by) == [
        (O.NONE, None, None), (O.BLIND, O.ACCEPTED_ONLY, 1), (O.AWARE, O.ACCEPTED_ONLY, 1),
        (O.SWAPPED, O.ACCEPTED_ONLY, 1), (O.BLIND, O.ACCEPTED_ONLY, 2), (O.AWARE, O.ACCEPTED_ONLY, 2)]
    assert O.run_optionc_episodes(_episodes_cfg(rules=[O.ACCEPTED_ONLY], seeds_by_arm=by)) == 0
    eps = _eps(mock_root)
    assert sorted({(e["arm"], e["seed"]) for e in eps if e["arm"] == O.SWAPPED}) == [(O.SWAPPED, 1)]
    assert len(eps) == len(GAMES) * 6
    assert O.load_run("oc")[1] == []
    with pytest.raises(SystemExit):
        O.run_optionc_episodes(_episodes_cfg(seeds_by_arm={O.BLIND: [1], O.AWARE: [1]}))  # swapped has no seed


def test_two_seeds_draw_their_own_alternatives(sim):
    """Reviewer R1: the sample cache key holds the request seed, so seed 2 never reuses seed 1's samples."""
    class SeedSampler(FakeSampler):
        def sample(self, msgs, n, params, max_tokens, seed):
            self.calls.append(seed)
            return [{"text": f"Thought: t\nAction: examine thing {seed % 997}", "finish_reason": "stop"}] * n
    m = _models(lambda a, h: 1.0 if a.startswith("examine thing") else -1.0, SeedSampler())
    cache = O.GameCache(None, "train/1")
    one = _play(sim, m, O.AWARE, seed=1, cache=cache, cap=1)
    two = _play(sim, m, O.AWARE, seed=2, cache=cache, cap=1)
    assert one["trace"][0]["chosen"] != two["trace"][0]["chosen"] and len(m.sampler.calls) == 2


def test_counterfactual_is_classified_at_its_own_prefix(sim):
    """Reviewer R2: the same proposal at two prefixes, failing at the first and working at the second."""
    m = _models(lambda a, h: 1.0 if a == "go to drawer 1" else -1.0, FakeSampler(("go to drawer 1",)))
    m.agent.complete = lambda msgs, max_tokens=256: "Thought: t\nAction: open drawer 1"
    ep = _play(sim, m, O.AWARE, cap=2)
    assert [s["chosen"] for s in ep["trace"]] == ["go to drawer 1", "go to drawer 1"]
    assert ep["trace"][0]["proposal_failed"] and not ep["trace"][1]["proposal_failed"]
    assert ep["trace"][1]["proposal_obs"].startswith("You open the drawer 1")


def test_goal_flags_follow_the_proposal_and_the_chosen_action(sim):
    """Reviewer R7: an override of a goal-object proposal by a non-goal alternative."""
    goal = O.parse_task_line(sim.reset("train/1", "train").task_line).goal_object
    m = _models(lambda a, h: 1.0 if a == "inventory" else -1.0, FakeSampler(("inventory",)))
    m.agent.complete = lambda msgs, max_tokens=256: f"Thought: t\nAction: take {goal} 1 from countertop 1"
    s = _play(sim, m, O.AWARE, cap=1)["trace"][0]
    assert s["override"] and s["proposal_targets_goal"] and not s["chosen_targets_goal"]


def test_no_verifier_episodes_never_count_rejections(sim):
    """Reviewer R8."""
    ep = _play(sim, _models(), O.NONE, cap=3)
    assert not ep["step0_rejected"] and not ep["step0_override"] and ep["n_rejected"] == 0
    assert ep["longest_rejection_run"] == 0 and ep["n_abstained"] == 0


def test_no_alternative_counts_as_an_abstention_and_new_secondaries():
    """Reviewer R6, and the key secondaries of review 1 (S1)."""
    tr = [{"override": False, "proposal_failed": True, "failed": True, "proposal_accepted": False,
           "reason": "no_alternative", "repeat": False, "proposal_format_error": False}]
    tr += [{"override": True, "proposal_failed": False, "failed": True, "proposal_accepted": False, "reason": "argmax",
            "repeat": False, "proposal_format_error": False}] * 3
    tr += [{"override": True, "proposal_failed": True, "failed": False, "proposal_accepted": False, "reason": "argmax",
            "repeat": False, "proposal_format_error": False}]
    tr += [{"override": False, "proposal_failed": False, "failed": True, "proposal_accepted": True, "reason": "accepted",
            "repeat": False, "proposal_format_error": False}] * 7
    s = O.episode_summary({"trace": tr, "steps": len(tr), "success": False})
    assert s["n_abstained"] == 1 and s["override_net_harm"] == 2
    assert s["n_invalid"] == 11 and s["n_invalid_first10"] == 9  # the 11th and 12th steps fail too, after step 10


def test_cache_check_flags_a_replay_that_differs_while_success_does_not(mock_root):
    """Reviewer R10."""
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    d = mock_root / "oc" / "shard_00_of_01"
    eps = _eps(mock_root)
    ok = O.cache_check(AlfworldClient(), eps, GAMES[0], "train", "qwen3-8b", O.YN, d / "cache.jsonl", 6, 20, 8,
                       O.SHADOW_ARMS)
    assert ok["ok"]
    target = next(e for e in eps if e["game_id"] == GAMES[0] and e["arm"] == O.AWARE)
    target["trace"][0]["real_obs"] = "something else"
    bad = O.cache_check(AlfworldClient(), eps, GAMES[0], "train", "qwen3-8b", O.YN, d / "cache.jsonl", 6, 20, 8,
                        O.SHADOW_ARMS)
    assert not bad["ok"] and bad["mismatched"] and not bad["errors"]


class _GateScorer:
    """A non-mock stand-in verifier, so the run takes the real scorer's path (gates, the last-position check)."""

    mock = False

    def __init__(self):
        self.last_only = False
        self.closed = False

    def next_token_logprobs(self, messages, words, complement=None):
        return {"Yes": 1.0, "No": 0.0, "Nothing": -3.0, "You": -0.1, "not:Nothing": -0.05}

    def close(self):
        self.closed = True


def test_family_yes_no_gate_stops_the_run(mock_root, monkeypatch):
    """Reviewer R4: a failed Yes/No gate stops a Gemma run before any episode."""
    import wf.decisionsim

    monkeypatch.setattr(O, "open_verifier", lambda tag: _GateScorer())
    monkeypatch.setattr(wf.decisionsim, "yes_no_gate", lambda sc, prompts: {"ok": False, "n_problems": 1, "problems": ["x"]})
    with pytest.raises(families.TokenizationMismatch):
        O.run_optionc_episodes(_episodes_cfg(model="gemma2-9b-it", seeds=[1], rules=[O.ACCEPTED_ONLY]))
    assert not (mock_root / "oc" / "shard_00_of_01" / "episodes.jsonl").exists() or not _eps(mock_root)
    monkeypatch.setattr(wf.decisionsim, "yes_no_gate", lambda sc, prompts: {"ok": True, "n_problems": 0, "problems": []})
    assert O.run_optionc_episodes(_episodes_cfg(model="gemma2-9b-it", seeds=[1], rules=[O.ACCEPTED_ONLY],
                                                arms=[O.BLIND, O.AWARE])) == 0
    chk = json.loads((mock_root / "oc" / "shard_00_of_01" / "last_only_check.json").read_text())
    assert chk["adopted"] and chk["checked"] == 6  # two arms x three games: fewer than 8 gate prompts


def test_first_token_gate_stops_an_observation_run(mock_root, monkeypatch):
    monkeypatch.setattr(O, "open_verifier", lambda tag: _GateScorer())
    monkeypatch.setattr(O, "first_token_gate", lambda sc, prompts: {"ok": False, "n_problems": 2, "problems": ["y"]})
    cfg = _episodes_cfg(readout=O.OBS, seeds=[1], rules=[O.ACCEPTED_ONLY], arms=[O.AWARE], obs_gate_override="test")
    with pytest.raises(families.TokenizationMismatch):
        O.run_optionc_episodes(cfg)


def test_observation_gate_needs_a_reported_status(mock_root):
    """Reviewer R5: metrics that pass are not enough while the step-0 analysis has not reported."""
    a = mock_root / O.STEP0_ANALYSIS
    a.mkdir()
    (a / "metrics.json").write_text(json.dumps({"models": {"qwen3-8b": {"obs_competence": {"passed": True}}}}))
    (a / "STATUS.md").write_text("# STATUS — x — FAIL\n")
    assert not O.obs_gate_check({"model": "qwen3-8b"})["ok"]
    (a / "STATUS.md").write_text("# STATUS — x — REPORTED (step-0 check)\n")
    assert O.obs_gate_check({"model": "qwen3-8b"})["ok"]


class _Shifty(_GateScorer):
    """Scores that change with last_only by `shift` nats."""

    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def next_token_logprobs(self, messages, words, complement=None):
        y = 1.0 + (self.shift if self.last_only else 0.0)
        return {"Yes": y, "No": 0.0, "Nothing": -3.0 + (self.shift if self.last_only else 0.0), "You": -1.0,
                "not:Nothing": -0.05}


def test_last_only_is_adopted_only_if_no_rounded_score_changes():
    prompts = [[{"role": "user", "content": str(k)}] for k in range(10)]
    same = _Shifty(0.001)  # below half a 64th: every rounded score is the same
    r = O.last_only_check(same, prompts, O.YN)
    assert r["adopted"] and same.last_only and r["checked"] == 8 and r["max_abs_diff_raw"] == pytest.approx(0.001)
    moved = _Shifty(0.05)
    assert not O.last_only_check(moved, prompts, O.YN)["adopted"] and moved.last_only is False
    assert not O.last_only_check(_Shifty(0.05), prompts, O.OBS)["adopted"]
    mock = types.SimpleNamespace(mock=True)
    assert O.last_only_check(mock, prompts, O.YN) == {"checked": 0, "adopted": False}


def test_scorer_asks_for_the_last_position_only_when_told():
    import numpy as np

    calls = []

    class Model:
        def __init__(self, accept):
            self.accept = accept

        def __call__(self, input_ids, **kw):
            calls.append(kw)
            if kw and next(iter(kw)) not in self.accept:
                raise TypeError("unexpected keyword")
            return types.SimpleNamespace(logits=np.arange(12.0).reshape(1, 3, 4))

    sc = LogoddsScorer("Qwen/Qwen3-8B", model_tag="qwen3-8b", load=False)
    sc._model = Model({"logits_to_keep"})
    assert list(sc.last_logits("ids", False)) == [8.0, 9.0, 10.0, 11.0] and calls == [{}]
    calls.clear()
    assert list(sc.last_logits("ids", True)) == [8.0, 9.0, 10.0, 11.0] and calls == [{"logits_to_keep": 1}]
    calls.clear()
    sc._model = Model({"num_logits_to_keep"})
    sc.last_logits("ids", True)
    assert calls == [{"logits_to_keep": 1}, {"num_logits_to_keep": 1}]
    calls.clear()
    sc._model = Model(set())
    sc.last_logits("ids", True)
    assert calls == [{"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}]
    assert LogoddsScorer("Qwen/Qwen3-8B", model_tag="qwen3-8b", load=False).last_only is False


def test_last_position_logits_equal_the_full_pass_on_a_tiny_model():
    """Review 1, S6: on CPU, the last row of the logits is the same whether every position or only the last is kept."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    sc = LogoddsScorer("Qwen/Qwen3-8B", model_tag="qwen3-8b", load=False)
    sc._model = model
    ids = torch.randint(0, 64, (1, 50))
    with torch.no_grad():
        full, last = sc.last_logits(ids, False), sc.last_logits(ids, True)
    assert torch.allclose(full.float(), last.float(), atol=1e-6)


def _smoke_dirs(root, passed=True, seconds=None, status="REPORTED (shard complete)"):
    d = root / O.SMOKE_RUN / "shard_00_of_01"
    d.mkdir(parents=True, exist_ok=True)
    (d / "STATUS.md").write_text(f"# STATUS — x — {status}\n")
    (d / "metrics.json").write_text(json.dumps({"smoke_gate": {"passed": passed},
                                                "seconds_by_arm": seconds or {"none": 60.0, "aware": 120.0},
                                                "by_arm": {"none": 2, "aware": 4}}))
    a = root / O.STEP0_ANALYSIS
    a.mkdir(exist_ok=True)
    (a / "STATUS.md").write_text("# STATUS — x — REPORTED (step-0 check)\n")


def test_projection_from_the_smoke():
    m = [{"seconds_by_arm": {"none": 60.0, "aware": 120.0}, "by_arm": {"none": 2, "aware": 4}}]
    jobs = [(O.NONE, None, None), (O.AWARE, "r", 1), (O.BLIND, "r", 1)]  # blind not in the smoke: the slowest arm's figure
    assert O.project_gpu_hours(m, jobs, 120) == pytest.approx(120 * (30 + 30 + 30) / 3600)
    assert O.project_gpu_hours([], jobs, 120) is None


def test_order_gate_refuses_until_smoke_step0_and_cost_pass(mock_root):
    jobs = [(O.NONE, None, None), (O.AWARE, O.ACCEPTED_ONLY, 1)]
    cfg = {"model": "qwen3-8b", "n_games": 120, "cost_upper_gpu_hours": 1.0}
    assert any("no smoke run" in p for p in O.order_gate_check(cfg, jobs)["problems"])
    _smoke_dirs(mock_root, passed=False)
    assert any("smoke gate not passed" in p for p in O.order_gate_check(cfg, jobs)["problems"])
    _smoke_dirs(mock_root)
    g = O.order_gate_check(cfg, jobs)  # 120 games x (30 + 30) s = 2.0 h, above 1.5 x 1.0
    assert not g["ok"] and g["projected_gpu_hours"] == pytest.approx(2.0)
    assert O.order_gate_check({**cfg, "cost_upper_gpu_hours": 2.0 / 1.5}, jobs)["ok"]  # exactly 1.5 x: allowed
    assert not O.order_gate_check({k: v for k, v in cfg.items() if k != "cost_upper_gpu_hours"}, jobs)["ok"]
    (mock_root / O.STEP0_ANALYSIS / "STATUS.md").write_text("# STATUS — x — FAIL\n")
    assert any("has not reported" in p for p in O.order_gate_check({**cfg, "cost_upper_gpu_hours": 9}, jobs)["problems"])
    over = O.order_gate_check({**cfg, "order_override": "why"}, jobs)
    assert over["ok"] and over["override"] == "why" and over["problems"]


def test_gemma_needs_a_passed_memory_smoke(mock_root):
    _smoke_dirs(mock_root)
    cfg = {"model": "gemma2-9b-it", "n_games": 1, "cost_upper_gpu_hours": 99}
    jobs = [(O.AWARE, O.ACCEPTED_ONLY, 1)]
    assert any("memory smoke" in p for p in O.order_gate_check(cfg, jobs)["problems"])
    ms = mock_root / "optionc_memsmoke_gemma2-9b-it"
    ms.mkdir()
    (ms / "STATUS.md").write_text("# STATUS — x — REPORTED (memory smoke)\n")
    (ms / "metrics.json").write_text(json.dumps({"ok": False}))
    assert not O.order_gate_check(cfg, jobs)["ok"]
    (ms / "metrics.json").write_text(json.dumps({"ok": True}))
    assert O.order_gate_check(cfg, jobs)["ok"]
    assert O.order_gate_check({**cfg, "model": "qwen3-8b"}, jobs)["ok"]


def test_a_run_waits_for_its_gates_and_logs_them(mock_root):
    cfg = _episodes_cfg(seeds=[1], rules=[O.ACCEPTED_ONLY], smoke=False, cost_upper_gpu_hours=99)
    assert O.run_optionc_episodes(cfg) == 1
    assert "waits for its gates" in (mock_root / "oc" / "shard_00_of_01" / "STATUS.md").read_text()
    _smoke_dirs(mock_root)
    assert O.run_optionc_episodes(cfg) == 0
    gate = json.loads((mock_root / "oc" / "shard_00_of_01" / "order_gate.json").read_text())
    assert gate["ok"] and gate["projected_gpu_hours"] is not None


def test_memory_smoke_run(mock_root):
    assert O.run_optionc_memsmoke({"run_id": "ms", "model": "gemma2-9b-it", "split": "train", "games": GAMES,
                                   "n_games": 3, "steps": 8}) == 0
    m = json.loads((mock_root / "ms" / "metrics.json").read_text())
    assert m["ok"] and m["n_prompts"] == 8 and m["history_steps"] > 0
    assert "REPORTED" in (mock_root / "ms" / "STATUS.md").read_text().splitlines()[0]


def test_memory_smoke_records_a_failure(mock_root, monkeypatch):
    class Boom(_GateScorer):
        def next_token_logprobs(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

        def _encode(self, m):
            return [[0, 1, 2]]
    monkeypatch.setattr(O, "open_verifier", lambda tag: types.SimpleNamespace(mock=True, close=lambda: None,
                                                                             next_token_logprobs=Boom().next_token_logprobs))
    assert O.run_optionc_memsmoke({"run_id": "ms", "model": "gemma2-9b-it", "split": "train", "games": GAMES,
                                   "n_games": 3, "steps": 4, "readouts": [O.OBS]}) == 0
    m = json.loads((mock_root / "ms" / "metrics.json").read_text())
    assert not m["ok"] and "out of memory" in m["error"]


def test_primary_run_qualification():
    good = {"model": "qwen3-8b", "rules": [O.ACCEPTED_ONLY], "arms": [O.BLIND, O.AWARE], "readout": O.YN}
    assert O.primary_problems(good, None) == []
    assert O.primary_problems({**good, "model": "llama3.1-8b"}, None)
    assert O.primary_problems({**good, "rules": [O.ARGMAX]}, None)
    assert O.primary_problems({**good, "arms": [O.AWARE]}, None)
    obs = {**good, "readout": O.OBS}
    assert O.primary_problems(obs, None)
    qs = {"models": {"qwen3-8b": {"classification": "question-specific (supports the task-success reading)",
                                  "obs_competence": {"passed": True}}}}
    assert O.primary_problems(obs, qs) == []
    assert O.primary_problems(obs, {"models": {"qwen3-8b": {**qs["models"]["qwen3-8b"], "obs_competence": {"passed": False}}}})
    assert O.primary_problems(obs, {"models": {"qwen3-8b": {"classification": "mixed (x)", "obs_competence": {"passed": True}}}})


def test_analysis_refuses_a_primary_run_that_does_not_qualify_and_records_a_missing_archive(mock_root):
    assert O.run_optionc_episodes(_episodes_cfg(seeds=[1], rules=[O.ARGMAX])) == 0
    assert O.run_optionc_analysis({"run_id": "oca", "runs": {"oc": "oc"}, "primary_run": "oc"}) == 1
    assert "does not qualify" in (mock_root / "oca" / "STATUS.md").read_text()
    assert O.run_optionc_analysis({"run_id": "oca", "runs": {"oc": "oc"}, "primary_run": None,
                                   "rq3_source": "nowhere"}) == 0
    assert O.run_optionc_episodes(_episodes_cfg(run_id="oc2", seeds=[1], rules=[O.ACCEPTED_ONLY])) == 0
    assert O.run_optionc_analysis({"run_id": "ocb", "runs": {"oc2": "oc2"}, "primary_run": "oc2",
                                   "rq3_source": "nowhere"}) == 0
    m = json.loads((mock_root / "ocb" / "metrics.json").read_text())
    assert "missing" in m["drift"] and m["primary"]["obstruction"] in ("clear", "obstructs", "not computable")
    assert set(m["primary"]["key_secondaries"]) == {"override_net_harm", "n_invalid_first10"}


def test_step0_go_to_readouts_count_only_working_go_to():
    """Reviewer R9: a failing `go to` enters neither the go-to acceptance nor the any-task drop."""
    states = [_s0("g0")]
    states[0]["candidates"].append(_t(9, "go to drawer 9", False, game="g0"))
    L = {(O.AWARE, O.YN): lambda t: 5.0 if t["action"] == "go to drawer 9" else -5.0,
         (O.BLIND, O.YN): lambda t: -5.0 if t["action"] == "go to drawer 9" else 5.0,
         (O.SWAPPED, O.YN): lambda t: -5.0}
    rep = O.step0_report(states, _recs(states, L))
    assert rep["cells"][O.YN][O.AWARE]["working_goto_accepted"]["estimate"] == 0.0
    assert rep["shifts"][O.YN]["anytask_drop_aware_minus_blind"]["estimate"] == pytest.approx(-10.0)
