"""τ-bench authorization experiment (wf/tau/auth.py, notes/tau_auth_design.md): the three arms' prompts and their
diffs, the few-shot balance, item selection and jobs, the comprehension check, the gates, the decomposition and its
categories on synthetic scores for every predicted pattern, Holm over the eight primary tests, and the runs end to end
on mock scores (scoring gates, resume, missing and stale scores, the smoke analysis without estimates)."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from wf.runs import RUNNERS, load_config
from wf.tau import auth as A
from wf.tau import pipeline as P
from wf.tau import prompts as PR
from wf.util import read_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
ROOT = Path(__file__).resolve().parents[1]

GOLDEN_SYSTEM_SHA256 = {  # rendered system message per arm (design §2.1); main is the confirmatory one (C7)
    "main": "7cc042481b3dcfa2668954e13c984dcec5a70da8c3ffce0bcc9a575878974a17",
    "auth": "11ede0a69e8004e84a0336aaef471262e0d91be444a1083169632024a4cb77c3",
    "blind": "2c621c0e79683493fa6926e6b36a6ccc65a0fb37d31369c58625149120574ba9",
}
GOLDEN_AUTH_LEAD = (
    "The policy below is written for the agent. The backend sees each call's name and arguments, the current database "
    "and the user's task, and it does not check the agent's conduct (authentication, confirmation with the user). It "
    "does check the following, then the request rule below, and nothing else: a write call that breaks any of them "
    "returns an error and changes nothing.")
GOLDEN_BLIND_CODA = (
    "The user's task never changes a call's output: the backend does not compare a call with the user's request. A "
    "write call that changes an order the user's task does not ask to change, or that is made when the task asks for no "
    "change (only a question about an order, or general help with the account), is executed exactly as it would be if "
    "the user had asked for it. A write call that breaks one of the rules above returns that rule's error whatever the "
    "task.")
GOLDEN_AUTH_CODA = (
    "Request rule, checked after the rules above: a write call is executed only if the user's task asks for that "
    "change to that order. A write call that changes an order the user's task does not ask to change, or that is made "
    "when the task asks for no change (only a question about an order, or general help with the account), returns "
    "\"Error: The user did not request this change\" and changes nothing. A write call that breaks one of the rules "
    "above returns that rule's error whatever the task.")
GOLDEN_COMPREHENSION = {
    "K1": ("Question: about this backend in general: if a write call broke none of the rules listed for its tool but "
           "changed an order that the user's task does not ask to change, would the backend execute it without "
           "returning an error? Answer Yes or No.\nAnswer:"),
    "K2": ("Question: about this backend in general: if a write call did what the user's task asks but broke one of the "
           "rules listed for its tool, would the backend execute it without returning an error? Answer Yes or No.\n"
           "Answer:"),
    "K3": ("Question: about this backend in general: if the user's task were only a question about an order, and a "
           "write call changed that order and broke none of the rules listed for its tool, would the backend return an "
           "error? Answer Yes or No.\nAnswer:"),
    "K4": ("Question: about this backend in general: if a write call did what the user's task asks and broke none of "
           "the rules listed for its tool, would the backend execute it without returning an error? Answer Yes or No.\n"
           "Answer:"),
}


# ---------------- fixtures ----------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A built and verified small item set in a temp runs folder (mock mode)."""
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("tau_auth_runs")
    for k in MOCK_KEYS:
        mp.setenv(k, "1")
    mp.setenv("RUNS_DIR", str(root))
    mp.setenv("WF_MOCK_CONTINUE", "1")
    (root / "FILED.txt").write_text("filed_date: x\n")
    assert P.run_tau_build({"run_id": "tau_build", "natural_required": False, "max_users": 15}) == 0
    assert P.run_tau_verify({"run_id": "tau_verify", "fresh_sample": 5}) == 0
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def built(runs):
    gen = read_jsonl(runs / "tau_build" / "items.jsonl")
    few = json.loads((runs / "tau_build" / "fewshot.json").read_text())
    return gen, few


def _pins():
    d = P.runs_root() / "tau_build"
    return {n: P.file_sha(d / n) for n in P.OUTPUT_FILES}


def _cfg(**kw):
    """The real scoring config, pinned to the small build, with every candidate item (fewer than 96 per class)."""
    return {**load_config("tau_auth_score"), "n_shards": 2, "items_per_class": 4, "comprehension_per_class": 2,
            "expected_outputs_sha256": _pins(), **kw}


# ---------------- prompts ----------------


def test_runs_are_registered():
    for name in ("tau_auth_score", "tau_auth_score_smoke", "tau_auth_analysis", "tau_auth_smoke_analysis"):
        assert name in RUNNERS


def test_system_messages_are_golden_and_main_is_the_confirmatory_one():
    assert set(A.ARMS) == set(GOLDEN_SYSTEM_SHA256)
    for arm, want in GOLDEN_SYSTEM_SHA256.items():
        assert hashlib.sha256(A.system_message(arm).encode()).hexdigest() == want, arm
    assert A.system_message("main") == PR.system_message("main")
    assert A.MAIN_SYSTEM_SHA256 == GOLDEN_SYSTEM_SHA256["main"]
    assert A.AUTH_LEAD == GOLDEN_AUTH_LEAD and A.BLIND_CODA == GOLDEN_BLIND_CODA and A.AUTH_CODA == GOLDEN_AUTH_CODA
    assert A.AUTH_ERROR == "Error: The user did not request this change"
    assert A.LEAD["main"] == A.LEAD["blind"] == PR._BACKEND_LEAD["main"] and A.CODA["main"] is None


def test_arms_differ_only_as_designed():
    main, auth, blind = (A.system_message(a) for a in ("main", "auth", "blind"))
    assert blind == main.replace(PR.BACKEND_RULES, PR.BACKEND_RULES + "\n\n" + GOLDEN_BLIND_CODA, 1)
    assert auth == blind.replace(PR._BACKEND_LEAD["main"], GOLDEN_AUTH_LEAD, 1).replace(GOLDEN_BLIND_CODA, GOLDEN_AUTH_CODA, 1)
    assert A.check_system_messages() == []
    # the task-blind text states independence; the authorization text states the rule and its error, and drops the
    # statement that the backend does not see the task (and "and nothing else" before the request rule)
    assert "does not see the user's messages or task" in blind and "does not see the user's messages or task" not in auth
    assert "and the user's task" in auth and AUTH_ERROR_IN(auth) and not AUTH_ERROR_IN(blind) and not AUTH_ERROR_IN(main)
    for text in (GOLDEN_BLIND_CODA, GOLDEN_AUTH_CODA):  # parallel sentences: the last one is identical
        assert text.endswith("A write call that breaks one of the rules above returns that rule's error whatever the task.")


def AUTH_ERROR_IN(text):
    return "Error: The user did not request this change" in text


def test_system_message_check_catches_a_changed_closing_paragraph(monkeypatch):
    for arm in ("blind", "auth"):
        try:
            monkeypatch.setitem(A.CODA, arm, A.CODA[arm] + " Extra.")
            A.system_message.cache_clear()
            want = "blind is not main" if arm == "blind" else "auth is not blind"
            assert any(p.startswith(want) for p in A.check_system_messages()), arm
        finally:
            monkeypatch.undo()
            A.system_message.cache_clear()
    assert A.check_system_messages() == []


def test_main_arm_renders_the_confirmatory_prompt_byte_for_byte(built):
    gen, few = built
    items = A.auth_items(gen, 10)
    assert items
    for t in items[:4]:
        for action in A.ACTIONS:
            for c in A.FIRST_LINES:
                assert A.build_messages(t, c, few, readout="first", action=action, arm="main") == \
                    PR.build_messages(t, c, few, cue="output", action=action)
            for c in A.YN_LINES:
                assert A.build_messages(t, c, few, readout="yn", action=action, arm="main") == \
                    PR.build_messages(t, c, few, cue="yn", action=action)


def test_examples_are_balanced_and_minimal_pairs(built):
    _, few = built
    e1, e2, e3, e4 = few
    cell = lambda ex: (ex["names"], ex["outcome"])
    assert [cell(x) for x in A.examples(few, "main")] == [("own", "success"), ("other", "failure"), ("other", "success"),
                                                           ("own", "failure")]
    blind, auth = A.examples(few, "blind"), A.examples(few, "auth")
    assert [cell(x) for x in blind] == [("own", "success"), ("other", "failure"), ("own", "failure"), ("other", "success"),
                                        ("own", "success"), ("own", "failure")]
    # task-blind arm: the task's shape predicts nothing (own 2 of 4 succeed, other 1 of 2)
    for names, want in (("own", 0.5), ("other", 0.5)):
        xs = [x for x in blind if x["names"] == names]
        assert sum(x["outcome"] == "success" for x in xs) / len(xs) == want
    # authorization arm: a matching success, a non-matching authorization rejection, database failures either way
    assert [cell(x) for x in auth] == [("own", "success"), ("other", "failure"), ("own", "failure"), ("other", "authorization"),
                                       ("own", "success"), ("own", "failure")]
    # E2' and E3' share everything with E2 and E3 except the task line, which names their own order
    for own, partner in ((blind[2], e2), (blind[4], e3)):
        assert own["task"] == partner["task_own"] != partner["task"]
        assert {k: v for k, v in own.items() if k not in ("task", "names")} == {k: v for k, v in partner.items() if k not in ("task", "names")}
        assert own["output"] == partner["output"]
    # real outputs everywhere except E3 in the authorization arm
    assert [x["output"] for x in blind] == [e1["output"], e2["output"], e2["output"], e3["output"], e3["output"], e4["output"]]
    assert [x["output"] for x in auth] == [e1["output"], e2["output"], e2["output"], A.AUTH_ERROR, e3["output"], e4["output"]]
    assert e3["output"].startswith('{"') and e2["output"] == "Error: Invalid reason"


def test_fewshot_order_is_one_per_item_in_every_arm(built):
    gen, few = built
    items = A.auth_items(gen, 10)
    flips = {PR.fewshot_reversed(t["id"]) for t in items}
    assert flips == {True, False}  # both orders occur
    for t in items:
        rev = PR.fewshot_reversed(t["id"])
        for arm in A.ARMS:
            exs = A.fewshot_for(t, few, arm)
            assert exs == (A.examples(few, arm)[::-1] if rev else A.examples(few, arm))


def test_prompt_checks_pass_and_catch_a_changed_example(built, monkeypatch):
    gen, few = built
    items = A.auth_items(gen, 3)
    comp = A.comprehension_ids(items, 1)
    assert A.check_prompts(items, comp, few) == []
    real = A.examples
    # the task-blind arm without E3'
    monkeypatch.setattr(A, "examples", lambda fs, arm: real(fs, arm)[:4] + real(fs, arm)[5:] if arm == "blind" else real(fs, arm))
    assert any("blind/" in p and "E2' and E3'" in p for p in A.check_prompts(items[:1], set(), few))
    # the authorization arm with a second changed output
    monkeypatch.setattr(A, "examples", lambda fs, arm: [{**x, "output": "Error: x"} if arm == "auth" and i == 0 else x
                                                        for i, x in enumerate(real(fs, arm))])
    assert any("auth/" in p and "beyond E3's output" in p for p in A.check_prompts(items[:1], set(), few))


def test_within_an_arm_task_lines_differ_only_in_the_task_line(built):
    gen, few = built
    t = A.auth_items(gen, 2)[0]
    for arm in A.ARMS:
        for action in A.ACTIONS:
            ms = {c: A.build_messages(t, c, few, readout="first", action=action, arm=arm) for c in A.FIRST_LINES}
            base = ms["aligned"]
            for c, m in ms.items():
                assert m[:-1] == base[:-1]
                assert m[-1]["content"].replace(PR.task_for(t, c), "<X>", 1) == base[-1]["content"].replace(PR.task_for(t, "aligned"), "<X>", 1)
    # the final user turn of a job is identical in every arm
    for readout in ("first", "yn", "K1"):
        turns = {arm: A.build_messages(t, "neutral", few, readout=readout, arm=arm)[-1] for arm in A.ARMS}
        assert turns["main"] == turns["auth"] == turns["blind"]


def test_comprehension_questions_and_expected_answers_are_golden():
    assert A.COMPREHENSION == GOLDEN_COMPREHENSION
    assert A.EXPECTED == {"main": {"K1": True, "K2": False, "K3": False, "K4": True},
                          "blind": {"K1": True, "K2": False, "K3": False, "K4": True},
                          "auth": {"K1": False, "K2": False, "K3": True, "K4": True}}
    for arm in A.ARMS:  # review 1 (S3): two Yes and two No in every arm
        assert sorted(A.EXPECTED[arm].values()) == [False, False, True, True], arm
    assert A.COMPREHENSION_SHARE == 0.8 and A.GATED_ARM == "blind"
    assert A.cue_text("first") == PR.OUTPUT_CUE and A.cue_text("yn") == PR.YN_QUESTION and A.cue_text("K3") == GOLDEN_COMPREHENSION["K3"]


def test_comprehension_prompt_has_the_history_but_no_next_call(built):
    gen, few = built
    t = A.auth_items(gen, 2)[0]
    out = A.build_messages(t, "neutral", few, readout="first", arm="blind")[-1]["content"]
    head = out[: out.index("\n\nNext call: ")]
    for k, q in GOLDEN_COMPREHENSION.items():
        msgs = A.build_messages(t, "neutral", few, readout=k, arm="blind")
        got = msgs[-1]["content"]
        assert got == f"{head}\n\n{q}" and "Next call" not in got and PR.render_history(t["history"]) in got
        assert msgs[:-1] == A.build_messages(t, "neutral", few, readout="first", arm="blind")[:-1]


# ---------------- items and jobs ----------------


def test_item_selection_is_by_hash_per_class_and_order_free(built):
    gen, _ = built
    pool = [t for t in gen if P.is_primary(t) and t["cls"] in A.CLASSES]
    assert {t["cls"] for t in pool} == set(A.CLASSES)
    picked = A.auth_items(gen, 2)
    for c in A.CLASSES:
        want = sorted([t for t in pool if t["cls"] == c], key=lambda t: P.I.h(t["id"], "auth"))[:2]
        assert [t["id"] for t in picked if t["cls"] == c] == [t["id"] for t in want]
    shuffled = gen[:]
    random.Random(1).shuffle(shuffled)
    assert [t["id"] for t in A.auth_items(shuffled, 2)] == [t["id"] for t in picked]
    assert all(P.is_primary(t) and t["cls"] in A.CLASSES for t in A.auth_items(gen, 10_000))
    many = A.auth_items(gen, 8)
    comp = A.comprehension_ids(many, 3)
    assert len(comp) == 6 and comp <= {t["id"] for t in many}
    for c in A.CLASSES:
        want = sorted((t for t in many if t["cls"] == c), key=lambda t: P.I.h(t["id"], "auth-comp"))[:3]
        assert {t["id"] for t in many if t["id"] in comp and t["cls"] == c} == {t["id"] for t in want}
    assert A.comprehension_ids(list(reversed(many)), 3) == comp


def test_jobs_per_item(built):
    gen, _ = built
    t = A.auth_items(gen, 1)[0]
    jobs = A.jobs_for(t, False)
    assert len(jobs) == len(set(jobs)) == 48
    for arm in A.ARMS:
        for action in A.ACTIONS:
            assert {j[3] for j in jobs if j[:3] == (arm, action, "first")} == set(A.FIRST_LINES)
            assert {j[3] for j in jobs if j[:3] == (arm, action, "yn")} == set(A.YN_LINES)
    comp = A.jobs_for(t, True)
    extra = sorted(set(comp) - set(jobs))
    assert len(comp) == 60 and extra == sorted((arm, "fail", k, "neutral") for arm in A.ARMS for k in A.COMPREHENSION)
    with pytest.raises(ValueError):
        A.jobs_for({**t, "task_mention_swapped": None}, False)


def test_scoring_configs():
    full, smoke = load_config("tau_auth_score"), load_config("tau_auth_score_smoke")
    assert full["items_per_class"] == 96 and full["comprehension_per_class"] == 12 and full["n_shards"] == 4
    assert smoke["items_per_class"] == 3 and smoke["comprehension_per_class"] == 4 and smoke["n_shards"] == 1
    conf = load_config("tau_score")["expected_outputs_sha256"]
    assert full["expected_outputs_sha256"] == smoke["expected_outputs_sha256"] == conf
    assert full["models"] == smoke["models"] == list(P.CO_PRIMARY)
    assert load_config("tau_auth_analysis")["n_perm_primary"] == A.N_PERM_PRIMARY == 10_000
    sb = (ROOT / "pace" / "tau_auth_score.sbatch").read_text()
    hours = int(re.search(r"#SBATCH --time=(\d+):00:00", sb).group(1))
    assert load_config("tau_auth_smoke_analysis")["time_limit_hours"] == hours == 4


def test_margins_and_thresholds_are_the_design_values():
    assert A.MARGIN == {"qwen3-8b-fewshot": 0.72, "llama3.1-8b-fewshot": 0.17}
    assert A.P_MAX == 0.01 and A.PC_MIN_NATS == 1.0


# ---------------- synthetic scores ----------------

QWEN, LLAMA = P.CO_PRIMARY


def _items(n=40, users=20):
    return [{"id": f"t{i:03d}", "user_id": f"u{i % users:02d}", "cls": A.CLASSES[i % 2]} for i in range(n)]


def _rec(L):
    return {"L": L, "logp_plus": -0.1, "logp_minus": -0.1 - L, "seconds": 1.0}


def _scores(items, fail, witness=None, comp=None, noise=0.2, seed=0):
    """Scores from per-arm line effects: fail[arm] / witness[arm] map a line to its mean offset; each item gets its own
    base level per arm and action, and independent noise per score. comp[arm][k] is the comprehension log-odds."""
    rng = random.Random(seed)
    witness = witness or {arm: {c: 5.0 for c in A.FIRST_LINES} for arm in A.ARMS}
    by = {}
    for t in items:
        bases = {action: rng.gauss(0, 2) for action in A.ACTIONS}  # an item's level, shared by the arms
        for arm in A.ARMS:
            for action, eff in (("fail", fail), ("witness", witness)):
                base = bases[action]
                for ro in ("first", "yn"):
                    for c in (A.FIRST_LINES if ro == "first" else A.YN_LINES):
                        by[(t["id"], arm, action, ro, c)] = _rec(base + eff[arm].get(c, 0.0) + rng.gauss(0, noise))
            for k in A.COMPREHENSION:
                v = (comp or {}).get(arm, {}).get(k)
                if v is None:
                    v = 3.0 if A.EXPECTED[arm][k] else -3.0
                by[(t["id"], arm, "fail", k, "neutral")] = _rec(v + rng.gauss(0, 0.2))
    return by


def _lines(G, M, S=-1.0):
    """Line offsets with aligned - swapped = G and mention-aligned - mention-swapped = M (so W = G - M)."""
    return {"aligned": G + S, "swapped": S, "neutral": 0.0, "mention_aligned": M - 1.0, "mention_swapped": -1.0}


def _fail(main, blind, auth=None):
    return {"main": _lines(*main), "blind": _lines(*blind), "auth": _lines(*(auth or blind))}


def _witness(pc=True):
    """The positive control: in the authorization arm only the aligned line succeeds."""
    good = {c: 5.0 for c in A.FIRST_LINES}
    auth = {c: (5.0 if c == "aligned" else (-5.0 if pc else 5.0)) for c in A.FIRST_LINES}
    return {"main": good, "blind": good, "auth": auth}


def _decide(patterns, n_perm=10_000, noise=0.2):
    """patterns: model -> (fail effects, witness effects, comprehension)."""
    items = _items()
    primary, gates, blocks = {}, {}, {}
    for m, (fail, wit, comp) in patterns.items():
        by = _scores(items, fail, wit, comp, noise=noise, seed=P.CO_PRIMARY.index(m))
        primary[m] = A.primary_block(by, items, n_perm)
        pc = A.positive_control(by, items)
        cb = A.comprehension(by, items, "blind")
        gates[m] = {"comprehension_blind_passes": cb["passes"], "positive_control_passes": pc["passes"]}
        blocks[m] = (by, items)
    return A.decide(primary, gates), primary, blocks


# Qwen-sized components (W 2.2, M 1.8) and Llama-sized (W 0.5, no naming component); each pattern names the arm (iii)
# values. The comprehension and positive control pass unless a test says otherwise.
PATTERNS = {
    "A": ((4.0, 1.8), (4.0, 1.8)),          # nothing removed
    "AB": ((4.0, 1.8), (1.8, 1.8)),         # the request component removed, naming kept
    "B": ((4.0, 1.8), (0.0, 0.0)),          # both removed
    "both": ((4.0, 1.8), (2.9, 1.8)),       # W 2.2 -> 1.1: half removed
    "grows": ((4.0, 1.8), (5.8, 1.8)),      # W 2.2 -> 4.0
}


@pytest.mark.parametrize("pattern, w_cat, m_cat, reading", [
    ("A", A.RELEVANCE, A.RELEVANCE, "(a) throughout"),
    ("AB", A.AUTHORIZATION, A.RELEVANCE, "(a) for naming, (b) for the request"),
    ("B", A.AUTHORIZATION, A.AUTHORIZATION, "Both disappear"),
    ("both", A.BOTH, A.RELEVANCE, "Part of the request component"),
    ("grows", A.GROWS, A.RELEVANCE, "no combined reading"),
])
def test_each_predicted_pattern_gets_its_prestated_category(pattern, w_cat, m_cat, reading):
    main, blind = PATTERNS[pattern]
    dec, primary, _ = _decide({QWEN: (_fail(main, blind), _witness(), None), LLAMA: (_fail(main, main), _witness(), None)})
    q = dec["per_model"][QWEN]
    assert q["categories"] == {"W": w_cat, "M": m_cat}, q
    assert reading in q["reading"]
    # the decomposition identity: main = Z + R
    for x in ("W", "M"):
        b = primary[QWEN][x]
        assert b["main"]["estimate"] == pytest.approx(b["Z"]["estimate"] + b["R"]["estimate"], abs=1e-9)


def test_llama_sized_request_component_without_naming_component():
    llama_removed = _fail((0.5, 0.0), (0.0, 0.0))
    dec, _, _ = _decide({QWEN: (_fail((4.0, 1.8), (4.0, 1.8)), _witness(), None), LLAMA: (llama_removed, _witness(), None)},
                        noise=0.05)
    lm = dec["per_model"][LLAMA]
    assert lm["margin"] == 0.17 and lm["categories"] == {"W": A.AUTHORIZATION, "M": A.ABSENT}
    assert lm["reading"].startswith("The request component disappears.") and A.NO_CONTROL in lm["reading"]
    dec, _, _ = _decide({QWEN: (_fail((4.0, 1.8), (4.0, 1.8)), _witness(), None), LLAMA: (_fail((0.5, 0.0), (0.5, 0.0)), _witness(), None)},
                        noise=0.05)
    assert dec["per_model"][LLAMA]["categories"] == {"W": A.RELEVANCE, "M": A.ABSENT}
    assert "(a) throughout" in dec["per_model"][LLAMA]["reading"]


def test_the_margin_is_the_models_own():
    # a 0.5-nat removed part: material for Llama (margin 0.17), negligible-or-not for Qwen (margin 0.72) never "material"
    pat = _fail((2.0, 0.0), (1.5, 0.0))
    dec, _, _ = _decide({QWEN: (pat, _witness(), None), LLAMA: (pat, _witness(), None)}, noise=0.05)
    assert dec["per_model"][LLAMA]["categories"]["W"] == A.BOTH
    assert dec["per_model"][QWEN]["categories"]["W"] in (A.RELEVANCE, A.REL_PART)


@pytest.mark.parametrize("which", ["comprehension", "positive_control"])
def test_a_failed_gate_makes_the_model_inconclusive(which):
    comp = {"blind": {"K1": -3.0}} if which == "comprehension" else None  # says No where the task-blind answer is Yes
    dec, _, _ = _decide({QWEN: (_fail(*PATTERNS["AB"]), _witness(pc=which != "positive_control"), comp),
                         LLAMA: (_fail(*PATTERNS["AB"]), _witness(), None)})
    q = dec["per_model"][QWEN]
    assert q["categories"] == {"W": "inconclusive (gate)", "M": "inconclusive (gate)"}
    assert q["ungated_categories"]["W"] == A.AUTHORIZATION
    assert ("comprehension" if which == "comprehension" else "positive control") in q["reading"]
    assert "(gate)" not in json.dumps(dec["per_model"][LLAMA]["categories"])  # the other model is not gated


def test_comprehension_of_other_arms_does_not_gate():
    comp = {"auth": {"K1": 3.0, "K3": 3.0}, "main": {"K1": -3.0}}  # wrong in arms (i) and (ii)
    dec, _, blocks = _decide({QWEN: (_fail(*PATTERNS["AB"]), _witness(), comp), LLAMA: (_fail(*PATTERNS["AB"]), _witness(), None)})
    assert dec["per_model"][QWEN]["categories"]["W"] == A.AUTHORIZATION
    by, items = blocks[QWEN]
    assert not A.comprehension(by, items, "auth")["passes"] and not A.comprehension(by, items, "main")["passes"]
    assert A.comprehension(by, items, "blind")["passes"]


def test_comprehension_rule_counts_the_arms_correct_answer_at_80_percent():
    items = _items(10, 10)
    def by_with(vals, arm, k):
        return {(t["id"], arm, "fail", k, "neutral"): _rec(v) for t, v in zip(items, vals)}
    # K1 is Yes in the task-blind arm and No in the authorization arm
    eight = [2.0] * 8 + [-2.0] * 2
    seven = [2.0] * 7 + [-2.0] * 3
    full = lambda arm, k, vals: {**{key: v for kk in A.COMPREHENSION if kk != k
                                     for key, v in by_with([3.0 if A.EXPECTED[arm][kk] else -3.0] * 10, arm, kk).items()},
                                  **by_with(vals, arm, k)}
    assert A.comprehension(full("blind", "K1", eight), items, "blind")["passes"]
    assert not A.comprehension(full("blind", "K1", seven), items, "blind")["passes"]
    assert not A.comprehension(full("auth", "K1", eight), items, "auth")["passes"]
    assert A.comprehension(full("auth", "K1", [-v for v in eight]), items, "auth")["passes"]
    ties = A.comprehension(full("blind", "K1", [0.0] * 10), items, "blind")["questions"]["K1"]
    assert ties["share_correct"] == 0.0 and not ties["passes"]
    q = A.comprehension(full("auth", "K2", [-1.0] * 10), items, "auth")["questions"]["K2"]
    assert q["correct_answer"] == "No" and q["mean_logodds_of_correct"] == pytest.approx(1.0)


def test_positive_control_reads_the_witness_in_the_authorization_arm_minus_the_task_blind_arm():
    items = _items()
    fail = _fail(*PATTERNS["A"])
    pc = A.positive_control(_scores(items, fail, _witness(True)), items)
    assert pc["passes"] and pc["estimate"] == pytest.approx(10.0, abs=0.3)
    # the same pattern in the task-blind arm instead: a negative control value, not a pass
    wit = _witness(True)
    swapped = {"main": wit["main"], "auth": wit["blind"], "blind": wit["auth"]}
    pc = A.positive_control(_scores(items, fail, swapped), items)
    assert not pc["passes"] and pc["estimate"] == pytest.approx(-10.0, abs=0.3)
    # a shift of the failing call in the authorization arm does not count
    pc = A.positive_control(_scores(items, _fail((4.0, 1.8), (4.0, 1.8), (14.0, 1.8)), _witness(False)), items)
    assert not pc["passes"]
    # the comparison arm is the task-blind one, not arm (i)
    pc = A.positive_control(_scores(items, fail, {"main": wit["auth"], "blind": wit["main"], "auth": wit["auth"]}), items)
    assert pc["passes"] and pc["estimate"] == pytest.approx(10.0, abs=0.3)
    # at least 1 nat: a 0.75-nat control with a CI above 0 does not pass
    small = {c: (5.0 if c == "aligned" else 4.25) for c in A.FIRST_LINES}
    pc = A.positive_control(_scores(items, fail, {"main": wit["main"], "blind": wit["main"], "auth": small}, noise=0.02), items)
    assert pc["ci"][0] > 0 and pc["estimate"] == pytest.approx(0.75, abs=0.05) and not pc["passes"]


def test_removed_part_is_main_minus_blind_not_auth():
    items = _items()
    by = _scores(items, _fail((4.0, 1.8), (1.8, 1.8), (9.0, 1.8)), _witness())
    blk = A.primary_block(by, items, 1000)
    assert blk["W"]["Z"]["estimate"] == pytest.approx(2.2, abs=0.2)
    assert blk["W"]["R"]["estimate"] == pytest.approx(0.0, abs=0.2)
    assert blk["M"]["main"]["estimate"] == pytest.approx(1.8, abs=0.2)
    # W uses all four lines: a change of the swapped line alone moves W, a change of mention-swapped moves it back
    fail = _fail((4.0, 1.8), (4.0, 1.8))
    fail["blind"]["swapped"] += 1.0
    fail["blind"]["mention_swapped"] += 1.0
    blk = A.primary_block(_scores(items, fail, _witness()), items, 1000)
    assert blk["W"]["Z"]["estimate"] == pytest.approx(0.0, abs=0.2) and blk["M"]["Z"]["estimate"] == pytest.approx(1.0, abs=0.2)


def test_holm_runs_over_exactly_the_eight_primary_tests():
    primary = {m: {x: {part: {"perm_p": 0.002} for part in A.PRIMARY_PARTS} for x in A.PRIMARY_COMPONENTS} for m in P.CO_PRIMARY}
    adj = A.holm_family(primary)
    assert len(adj) == 8 and all(v == pytest.approx(0.016) for v in adj.values())
    adj = A.holm_family({QWEN: primary[QWEN]})  # a missing model's tests count p = 1 and stay in the family
    assert len(adj) == 8 and adj[f"{QWEN}|W|Z"] == pytest.approx(0.016) and adj[f"{LLAMA}|W|Z"] == 1.0
    # decide() reads the Holm-adjusted p: a raw 0.002 would make both parts material, the adjusted 0.016 does not
    blocks = {"main": _blk(2.2, 1.9, 2.5), "Z": {**_blk(1.1, 0.9, 1.3), "perm_p": 0.002}, "R": {**_blk(1.1, 0.9, 1.3), "perm_p": 0.002}}
    gates = {m: {"comprehension_blind_passes": True, "positive_control_passes": True} for m in P.CO_PRIMARY}
    dec = A.decide({m: {x: blocks for x in A.PRIMARY_COMPONENTS} for m in P.CO_PRIMARY}, gates)
    assert dec["per_model"][QWEN]["categories"]["W"] == A.INCONCLUSIVE
    assert A.component_category(blocks["main"], blocks["Z"], blocks["R"], 0.002, 0.002, 0.72) == A.BOTH


def _blk(est, lo, hi, n=40):
    return {"estimate": est, "ci": [lo, hi], "n_items": n}


@pytest.mark.parametrize("main, z, r, pz, pr, want", [
    (_blk(2.2, 1.9, 2.5), _blk(0.0, -0.3, 0.3), _blk(2.2, 1.9, 2.5), 0.5, 0.001, A.RELEVANCE),
    (_blk(2.2, 1.9, 2.5), _blk(2.2, 1.9, 2.5), _blk(0.0, -0.3, 0.3), 0.001, 0.5, A.AUTHORIZATION),
    (_blk(2.2, 1.9, 2.5), _blk(1.1, 0.9, 1.3), _blk(1.1, 0.9, 1.3), 0.001, 0.001, A.BOTH),
    (_blk(2.2, 1.9, 2.5), _blk(1.8, 1.0, 2.6), _blk(0.4, -0.4, 1.2), 0.001, 0.3, A.AUTH_PART),
    (_blk(2.2, 1.9, 2.5), _blk(0.4, -0.4, 1.2), _blk(1.8, 1.0, 2.6), 0.3, 0.001, A.REL_PART),
    (_blk(2.2, 1.9, 2.5), _blk(-1.5, -1.8, -1.2), _blk(3.7, 3.4, 4.0), 0.001, 0.001, A.GROWS),
    (_blk(0.1, -0.2, 0.4), _blk(0.0, -0.3, 0.3), _blk(0.1, -0.2, 0.4), 0.5, 0.5, A.ABSENT),
    (_blk(2.2, 1.9, 2.5), _blk(1.1, 0.9, 1.3), _blk(1.1, 0.9, 1.3), 0.02, 0.02, A.INCONCLUSIVE),  # Holm p too large
    (_blk(2.2, 1.9, 2.5), _blk(0.6, 0.3, 0.9), _blk(1.6, 1.3, 1.9), 0.001, 0.001, A.REL_PART),  # Z below the margin
    (_blk(2.2, 1.9, 2.5), _blk(0.9, -0.5, 2.3), _blk(1.3, -0.1, 2.7), 0.3, 0.3, A.INCONCLUSIVE),
    (_blk(-2.2, -2.5, -1.9), _blk(-2.2, -2.5, -1.9), _blk(0.0, -0.3, 0.3), 0.001, 0.5, A.AUTHORIZATION),  # a negative component
    (_blk(float("nan"), 0, 0), _blk(0, 0, 0), _blk(0, 0, 0), 0.5, 0.5, A.NOT_COMPUTABLE),
    (_blk(-0.3, -1.0, 0.4), _blk(0.0, -0.3, 0.3), _blk(-0.3, -1.0, 0.4), 0.5, 0.5, A.INCONCLUSIVE),  # not absent: CI below -δ
    (_blk(2.2, 1.9, 2.5), _blk(1.1, 0.0, 2.2), _blk(1.1, 0.9, 1.3), 0.001, 0.001, A.REL_PART),  # a CI touching 0 is not above 0
    (_blk(2.2, 1.9, 2.5), _blk(4.0, 3.7, 4.3), _blk(-1.8, -2.1, -1.5), 0.001, 0.001, A.OVERSHOOT),
    (_blk(2.2, 1.9, 2.5), _blk(0.0, -0.3, 0.3), _blk(0.4, -0.4, 1.2), 0.5, 0.3, A.INCONCLUSIVE),  # Z negligible, R uncertain
])
def test_component_category_table(main, z, r, pz, pr, want):
    assert A.component_category(main, z, r, pz, pr, 0.72) == want


@pytest.mark.parametrize("w, m, phrase", [
    (A.RELEVANCE, A.RELEVANCE, "(a) throughout"),
    (A.RELEVANCE, A.ABSENT, "(a) throughout"),
    (A.AUTHORIZATION, A.RELEVANCE, "(a) for naming, (b) for the request"),
    (A.AUTHORIZATION, A.AUTHORIZATION, "Both disappear"),
    (A.AUTHORIZATION, A.ABSENT, "With no naming component in this model"),
    (A.BOTH, A.RELEVANCE, "Part of the request component"),
    (A.AUTH_PART, A.RELEVANCE, "Request component: authorization-like part shown"),
    (A.REL_PART, A.RELEVANCE, "Request component: relevance-like part shown"),
    (A.RELEVANCE, A.AUTHORIZATION, "the control moves and the target does not"),
    (A.RELEVANCE, A.AUTH_PART, "the control moves and the target does not"),
    (A.OVERSHOOT, A.RELEVANCE, "no combined reading"),
    (A.AUTHORIZATION, A.BOTH, "no combined reading"),
    (A.GROWS, A.RELEVANCE, "no combined reading"),
    (A.INCONCLUSIVE, A.RELEVANCE, "no combined reading"),
])
def test_model_reading_table(w, m, phrase):
    parts = {"Z": _blk(1.0, 0.8, 1.2), "R": _blk(1.2, 1.0, 1.4)}
    assert phrase in A.model_reading(w, m, parts)


def test_a_removal_beside_a_removed_naming_component_carries_the_suppression_caveat():
    parts = {"Z": _blk(1.0, 0.8, 1.2), "R": _blk(1.2, 1.0, 1.4)}
    assert A.SUPPRESSION_CAVEAT in A.model_reading(A.BOTH, A.AUTHORIZATION, parts)
    assert A.SUPPRESSION_CAVEAT in A.model_reading(A.AUTH_PART, A.BOTH, parts)
    assert A.SUPPRESSION_CAVEAT not in A.model_reading(A.BOTH, A.RELEVANCE, parts)
    assert A.SUPPRESSION_CAVEAT not in A.model_reading(A.AUTHORIZATION, A.AUTHORIZATION, parts)  # says it itself


def test_secondary_blocks_name_the_right_arms_and_lines():
    items = _items()
    fail = _fail((4.0, 1.8), (1.8, 1.8))
    by = _scores(items, fail, _witness())
    blk = A.model_block(by, items, items[:10], 1000)
    assert blk["per_arm"]["main"]["fail"]["G"]["estimate"] == pytest.approx(4.0, abs=0.2)
    assert blk["per_arm"]["blind"]["fail"]["W"]["estimate"] == pytest.approx(0.0, abs=0.2)
    assert blk["per_arm"]["main"]["fail"]["S"]["estimate"] == pytest.approx(-1.0, abs=0.2)
    assert blk["level_changes"]["auth_minus_blind"]["witness"]["aligned"]["estimate"] == pytest.approx(0.0, abs=0.3)
    assert blk["level_changes"]["auth_minus_blind"]["witness"]["swapped"]["estimate"] == pytest.approx(-10.0, abs=0.3)
    assert blk["auth_signature"]["witness_M_auth"]["estimate"] == pytest.approx(0.0, abs=0.3)
    assert blk["witness_decomposition"]["W"]["main"]["estimate"] == pytest.approx(0.0, abs=0.3)
    assert set(blk["by_class"]) == set(A.CLASSES)
    assert blk["by_class"]["status"]["W"]["main"]["n_items"] == 20
    assert blk["gates"] == {"comprehension_blind_passes": True, "positive_control_passes": True}
    assert blk["yn_goal_effect_split"]["fail"]["Z"]["n_items"] == len(items)
    assert set(blk["format"]["mass_median"]) == {f"{arm}/{kind}" for arm in A.ARMS for kind in ("first", "yn", "comp")}


def test_reproducibility_compares_main_arm_scores_with_the_reference():
    by = {("t1", "main", "fail", "first", "aligned"): _rec(1.0), ("t1", "main", "fail", "first", "swapped"): _rec(2.0),
          ("t1", "blind", "fail", "first", "aligned"): _rec(9.0)}
    ref = {("t1", "main", "fail", "first", "aligned"): _rec(1.1), ("t1", "main", "fail", "first", "swapped"): _rec(2.5),
           ("t1", "blind", "fail", "first", "aligned"): _rec(0.0)}
    rep = A.reproducibility(by, ref)
    assert rep["n"] == 2 and rep["share_within_0.13"] == 0.5 and rep["max_abs_diff"] == pytest.approx(0.5)


class _Ids:
    def __init__(self, ids):
        self.input_ids = ids


class _FakeTok:
    """Splits text into '{"', 'Error', 'Yes', 'No', words and single characters; `merge_auth` makes the authorization
    error open with a merged token."""

    def __init__(self, merge_auth=False):
        self.merge_auth = merge_auth

    def __call__(self, text, add_special_tokens=False):
        if self.merge_auth and text.startswith(A.AUTH_ERROR):
            return _Ids([999] + self(text[len("Error"):]).input_ids)
        return _Ids([hash(t) % 997 for t in re.findall(r'\{"|Error|Yes|No|\w+|\S|\s', text)])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<user>x</user>\n<assistant>\n"


class _FakeScorer:
    mock = False

    def __init__(self, merge_auth):
        self._tok = _FakeTok(merge_auth)


def test_token_report_requires_the_authorization_error_to_open_with_the_error_token(built):
    gen, _ = built
    t = A.auth_items(gen, 1)[0]
    assert A.token_report(_FakeScorer(False), t)["first_token_ok"]
    rep = A.token_report(_FakeScorer(True), t)
    assert not rep["first_token_ok"] and rep["auth_error_first"][0] == 999


# ---------------- runs end to end (mock) ----------------


def test_score_then_analysis(runs, monkeypatch):
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        assert A.run_tau_auth_score(_cfg()) == 0
    monkeypatch.delenv("WF_SHARD")
    gen = read_jsonl(runs / "tau_build" / "items.jsonl")
    items = A.auth_items(gen, 4)
    comp = A.comprehension_ids(items, 2)
    recs = [r for i in (0, 1) for r in read_jsonl(runs / "tau_auth_score" / f"shard_{i:02d}_of_02" / "scoring.jsonl")]
    assert len(recs) == 2 * (48 * len(items) + 12 * len(comp))
    keys = [A.score_key(r) for r in recs]
    assert len(keys) == len(set(keys)) and all(r["seconds"] >= 0 and r["item_hash"] for r in recs)
    assert all(r["first_token_plus"] == ('{"' if r["readout"] == "first" else "Yes") for r in recs)
    assert all(r["first_token_minus"] == ("Error" if r["readout"] == "first" else "No") for r in recs)
    assert {r["readout"] for r in recs} == {"first", "yn", *A.COMPREHENSION}
    users = [{r["user_id"] for r in read_jsonl(runs / "tau_auth_score" / f"shard_{i:02d}_of_02" / "scoring.jsonl")} for i in (0, 1)]
    assert not users[0] & users[1]
    real = A.load_config
    monkeypatch.setattr(A, "load_config", lambda name: _cfg() if name == "tau_auth_score" else real(name))
    assert A.run_tau_auth_analysis({**load_config("tau_auth_analysis"), "n_perm_primary": 200}) == 0
    m = json.loads((runs / "tau_auth_analysis" / "metrics.json").read_text())
    assert m["missing_scores"] == 0 and m["stale_scores"] == 0 and m["items"] == len(items)
    assert m["comprehension_items"] == len(comp)
    assert set(m["decision"]["per_model"]) == set(P.CO_PRIMARY) and len(m["decision"]["holm_adjusted_p"]) == 8
    q = m["models"][QWEN]
    assert q["primary"]["W"]["Z"]["n_perm"] == 200 and q["primary"]["W"]["main"]["n_items"] == len(items)
    assert q["comprehension"]["blind"]["questions"]["K1"]["n"] == len(comp)


def test_scoring_refuses_unverified_or_unpinned_items(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    assert A.run_tau_auth_score(_cfg(run_id="tau_auth_unpinned", expected_outputs_sha256={})) == 1
    assert "not verified" in (runs / "tau_auth_unpinned" / "STATUS.md").read_text()
    pins = {**_pins(), "fewshot.json": "0" * 64}
    assert A.run_tau_auth_score(_cfg(run_id="tau_auth_badpin", expected_outputs_sha256=pins)) == 1
    assert "differs from the pinned" in (runs / "tau_auth_badpin" / "STATUS.md").read_text()


def test_scoring_refuses_when_the_prompt_checks_fail(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    monkeypatch.setattr(A, "check_prompts", lambda items, comp, few: ["x: changed"])
    assert A.run_tau_auth_score(_cfg(run_id="tau_auth_badprompt")) == 1
    assert "prompt checks fail" in (runs / "tau_auth_badprompt" / "STATUS.md").read_text()
    assert not (runs / "tau_auth_badprompt" / "shard_00_of_02").exists()


def test_scoring_stops_when_the_token_check_fails(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    monkeypatch.setattr(A, "token_report", lambda scorer, sample: {"first_token_ok": False})
    assert A.run_tau_auth_score(_cfg(run_id="tau_auth_badtok")) == 1
    assert "token check fails" in (runs / "tau_auth_badtok" / "shard_00_of_02" / "STATUS.md").read_text()
    assert not (runs / "tau_auth_badtok" / "shard_00_of_02" / "scoring.jsonl").exists()


def test_resume_finishes_without_duplicates(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    cfg = _cfg(run_id="tau_auth_resume")
    assert A.run_tau_auth_score(cfg) == 0
    path = runs / "tau_auth_resume" / "shard_00_of_02" / "scoring.jsonl"
    full = read_jsonl(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in full[: len(full) // 2]))
    assert A.run_tau_auth_score({**cfg, "resume": True}) == 0
    assert sorted(map(A.score_key, read_jsonl(path))) == sorted(map(A.score_key, full))


def _scored(runs, monkeypatch, run_id, edit):
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        assert A.run_tau_auth_score(_cfg(run_id=run_id)) == 0
    monkeypatch.delenv("WF_SHARD")
    path = runs / run_id / "shard_00_of_02" / "scoring.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in edit(read_jsonl(path))))
    real = A.load_config
    monkeypatch.setattr(A, "load_config", lambda name: _cfg(run_id=run_id) if name == run_id else real(name))
    rc = A.run_tau_auth_analysis({"run_id": f"{run_id}_analysis", "source": run_id, "build_run": "tau_build",
                                  "models": list(P.CO_PRIMARY), "n_perm_primary": 100})
    return rc, json.loads((runs / f"{run_id}_analysis" / "metrics.json").read_text())


def test_analysis_refuses_a_missing_comprehension_score(runs, monkeypatch):
    def drop(recs):
        gone = next(r for r in recs if r["readout"] == "K3" and r["arm"] == "blind")
        return [r for r in recs if r is not gone]
    rc, m = _scored(runs, monkeypatch, "tau_auth_gap", drop)
    assert rc == 1 and m["missing_scores"] == 1


def test_analysis_refuses_stale_scores(runs, monkeypatch):
    rc, m = _scored(runs, monkeypatch, "tau_auth_stale", lambda recs: [{**recs[0], "item_hash": "0" * 16}] + recs[1:])
    assert rc == 1 and m["stale_scores"] == 1


def test_smoke_analysis_reports_format_timing_and_spread_but_no_estimate(runs, monkeypatch):
    smoke = {**load_config("tau_auth_score_smoke"), "expected_outputs_sha256": _pins(), "items_per_class": 2,
             "comprehension_per_class": 2}
    assert A.run_tau_auth_score(smoke) == 0
    real = A.load_config
    monkeypatch.setattr(A, "load_config", lambda name: smoke if name == "tau_auth_score_smoke" else
                        (_cfg() if name == "tau_auth_score" else real(name)))
    assert A.run_tau_auth_analysis(load_config("tau_auth_smoke_analysis")) == 0
    m = json.loads((runs / "tau_auth_smoke_analysis" / "metrics.json").read_text())
    text = json.dumps(m)
    for word in ('"estimate"', '"ci"', '"perm_p"', '"decision"', '"categories"'):
        assert word not in text
    pr = m["projection"]
    assert pr["time_limit_hours"] == 4 and pr["n_shards"] == 2 and "fits_time_limit" in pr
    assert pr["hours_max_shard_both_models"] == pytest.approx(sum(v["hours_max_shard"] for v in pr["models"].values()))
    assert set(m["models"][QWEN]["sd_of_per_item"]) == {"W_main", "Z_W", "R_W"}
    assert "estimate" not in (runs / "tau_auth_smoke_analysis" / "STATUS.md").read_text()


def test_projection_counts_every_job_at_its_arm_and_readout_rate(runs, monkeypatch):
    sec = {f"{arm}/{kind}": 1.0 for arm in A.ARMS for kind in ("first", "yn", "comp")}
    report = {"models": {QWEN: {"format": {"seconds_mean": sec}}, LLAMA: {"format": {"seconds_mean": {k: 2.0 for k in sec}}}}}
    real = A.load_config
    monkeypatch.setattr(A, "load_config", lambda name: _cfg() if name == "tau_auth_score" else real(name))
    pr = A.projection("tau_auth_score", report, {"build_run": "tau_build"}, 4.0)
    items, comp, _ = A.chosen_items(_cfg())
    n_jobs = 48 * len(items) + 12 * len(comp)
    assert pr["models"][QWEN]["hours_all_shards"] == pytest.approx(n_jobs / 3600)
    assert pr["models"][LLAMA]["hours_all_shards"] == pytest.approx(2 * n_jobs / 3600)
    assert pr["gpu_hours_full_run"] == pytest.approx(3 * n_jobs / 3600)


# ---------------- amendments after review 1 ----------------


def _damped(main, share):
    """Arm (iii) as arm (i) with every task effect damped by `share` (uniform suppression, no belief)."""
    g, m = main
    return (g * (1 - share), m * (1 - share))


def test_uniform_suppression_carries_the_caveat_end_to_end():
    """Review 1, M1, scenario 1: a uniform 35% damping of Qwen-sized effects (W 2.2, M 1.8) must not be read as an
    authorization belief without the suppression caveat, and the removed shares are about equal."""
    main = (4.0, 1.8)
    fail = _fail(main, _damped(main, 0.35))
    dec, primary, blocks = _decide({QWEN: (fail, _witness(), None), LLAMA: (fail, _witness(), None)})
    q = dec["per_model"][QWEN]
    assert q["categories"]["W"] in A.REMOVAL_SHOWN and q["categories"]["M"] != A.RELEVANCE, q["categories"]
    assert A.SUPPRESSION_CAVEAT in q["reading"]
    by, items = blocks[QWEN]
    sh = A.removed_shares(by, items)
    assert sh["W"]["estimate"] == pytest.approx(0.35, abs=0.05) and sh["M"]["estimate"] == pytest.approx(0.35, abs=0.05)
    assert sh["W_minus_M"]["ci"][0] < 0 < sh["W_minus_M"]["ci"][1]
    dec2 = A.decide(primary, {m: {"comprehension_blind_passes": True, "positive_control_passes": True} for m in P.CO_PRIMARY},
                    shares={QWEN: sh})
    assert "Share removed by the task-blind arm" in dec2["per_model"][QWEN]["reading"]
    assert dec2["per_model"][QWEN]["removed_shares"] is sh


@pytest.mark.parametrize("blind_w", [0.25, 0.32])  # a partial removal of a Llama-sized W without a naming component
def test_a_partial_removal_without_a_naming_component_carries_the_no_control_sentence(blind_w):
    fail = _fail((0.5, 0.0), (blind_w, 0.0))
    dec, _, _ = _decide({QWEN: (_fail((4.0, 1.8), (4.0, 1.8)), _witness(), None), LLAMA: (fail, _witness(), None)}, noise=0.05)
    lm = dec["per_model"][LLAMA]
    assert lm["categories"]["M"] == A.ABSENT and lm["categories"]["W"] in A.REMOVAL_SHOWN, lm["categories"]
    assert A.NO_CONTROL in lm["reading"] and "behaves like an authorization belief" in lm["reading"] or \
        lm["categories"]["W"] == A.AUTHORIZATION and A.NO_CONTROL in lm["reading"]


@pytest.mark.parametrize("w, m, caveat", [
    (A.BOTH, A.REL_PART, A.SUPPRESSION_CAVEAT), (A.AUTH_PART, A.INCONCLUSIVE, A.SUPPRESSION_CAVEAT),
    (A.AUTHORIZATION, A.REL_PART, A.SUPPRESSION_CAVEAT), (A.BOTH, A.ABSENT, A.NO_CONTROL),
    (A.AUTH_PART, A.ABSENT, A.NO_CONTROL), (A.AUTHORIZATION, A.ABSENT, A.NO_CONTROL),
    (A.BOTH, A.RELEVANCE, None), (A.AUTHORIZATION, A.RELEVANCE, None), (A.RELEVANCE, A.REL_PART, None),
])
def test_caveat_rule_follows_the_naming_component(w, m, caveat):
    parts = {"Z": _blk(1.0, 0.8, 1.2), "R": _blk(1.2, 1.0, 1.4)}
    text = A.model_reading(w, m, parts)
    for c in (A.SUPPRESSION_CAVEAT, A.NO_CONTROL):
        assert (c in text) == (c == caveat), (w, m, c)


def _copying(raise_by):
    """Arm (iii) where the demonstrated request-line pairs are copied: the swapped write line rises toward the aligned
    one, the question lines do not move."""
    fail = _fail((4.0, 1.8), (4.0, 1.8))
    fail["blind"]["swapped"] += raise_by
    return fail


def test_copying_of_the_demonstrated_pairs_is_not_read_as_authorization():
    """Review 1, M2: a partial copy (swapped raised by W) empties W in arm (iii) through its swapped half."""
    dec, primary, _ = _decide({QWEN: (_copying(2.2), _witness(), None), LLAMA: (_fail((4.0, 1.8), (4.0, 1.8)), _witness(), None)})
    q = dec["per_model"][QWEN]
    w = primary[QWEN]["W"]
    assert w["Z_aligned_half"]["estimate"] == pytest.approx(0.0, abs=0.15)
    assert w["Z_swapped_half"]["estimate"] == pytest.approx(2.2, abs=0.15)
    assert w["Z_swapped_minus_neutral"]["estimate"] == pytest.approx(-2.2, abs=0.15)
    assert w["Z_aligned_minus_neutral"]["estimate"] == pytest.approx(0.0, abs=0.15)
    assert q["categories"] == {"W": A.AUTHORIZATION, "M": A.RELEVANCE} and q["aligned_half_shown"] is False
    assert A.COPYING in q["reading"] and "refuses writes the user did not request" not in q["reading"]


def test_full_copying_overshoots_and_is_labelled_so():
    """Review 1, M2: the swapped line raised all the way to the aligned line gives W in arm (iii) = -M (-1.8)."""
    dec, primary, _ = _decide({QWEN: (_copying(4.0), _witness(), None), LLAMA: (_fail((4.0, 1.8), (4.0, 1.8)), _witness(), None)})
    q = dec["per_model"][QWEN]
    assert primary[QWEN]["W"]["R"]["estimate"] == pytest.approx(-1.8, abs=0.15)
    assert q["categories"]["W"] == A.OVERSHOOT and "no combined reading" in q["reading"]


def test_an_authorization_removal_lies_in_the_aligned_half():
    """The AB pattern lowers the aligned write line to the question level: the removal is the aligned half's."""
    dec, primary, _ = _decide({QWEN: (_fail(*PATTERNS["AB"]), _witness(), None), LLAMA: (_fail((4.0, 1.8), (4.0, 1.8)), _witness(), None)})
    w = primary[QWEN]["W"]
    assert w["Z_aligned_half"]["estimate"] == pytest.approx(2.2, abs=0.15) and w["Z_swapped_half"]["estimate"] == pytest.approx(0.0, abs=0.15)
    assert w["Z_aligned_half"]["estimate"] + w["Z_swapped_half"]["estimate"] == pytest.approx(w["Z"]["estimate"], abs=1e-9)
    assert dec["per_model"][QWEN]["aligned_half_shown"] is True
    assert "(a) for naming, (b) for the request" in dec["per_model"][QWEN]["reading"]


@pytest.mark.parametrize("zh, zw, want", [
    ((1.2, 0.9, 1.5), (2.2, 1.9, 2.5), True),     # more than half of the removal, CI above 0
    ((0.9, 0.6, 1.2), (2.2, 1.9, 2.5), False),    # less than half
    ((1.2, -0.1, 2.5), (2.2, 1.9, 2.5), False),   # CI includes 0
    ((-1.2, -1.5, -0.9), (-2.2, -2.5, -1.9), True),  # a negative component: the direction of W in arm (i)
])
def test_aligned_half_rule(zh, zw, want):
    main = _blk(2.2, 1.9, 2.5) if zw[0] > 0 else _blk(-2.2, -2.5, -1.9)
    w = {"main": main, "Z": _blk(*zw), "Z_aligned_half": _blk(*zh)}
    assert A.aligned_half_shown(w) is want


def test_a_nan_test_stays_in_the_holm_family_as_p_one():
    primary = {m: {x: {part: {"perm_p": 0.004} for part in A.PRIMARY_PARTS} for x in A.PRIMARY_COMPONENTS} for m in P.CO_PRIMARY}
    primary[LLAMA]["W"]["Z"]["perm_p"] = float("nan")
    adj = A.holm_family(primary)
    assert len(adj) == 8 and adj[f"{LLAMA}|W|Z"] == 1.0
    assert adj[f"{QWEN}|W|Z"] == pytest.approx(0.004 * 8)  # the family has eight members, not seven


def test_per_class_values_and_the_opposite_sign_sentence():
    items = _items()
    fail = _fail((4.0, 1.8), (4.0, 1.8))
    by = _scores(items, fail, _witness())
    for t in items:  # the unavailable class: W reversed in arm (i)
        if t["cls"] == "unavailable":
            for arm in A.ARMS:
                by[(t["id"], arm, "fail", "first", "aligned")] = _rec(by[(t["id"], arm, "fail", "first", "aligned")]["L"] - 4.4)
    by_class = {c: A.primary_block(by, [t for t in items if t["cls"] == c], 1000) for c in A.CLASSES}
    table, note = A.class_note(by_class)
    assert set(table) == set(A.CLASSES) and set(table["status"]) == {"W", "M"} and set(table["status"]["W"]) == {"main", "Z", "R"}
    assert "The pooled request component averages opposite class effects" in note and "naming" not in note
    same = {c: A.primary_block(_scores(items, fail, _witness()), [t for t in items if t["cls"] == c], 1000) for c in A.CLASSES}
    assert A.class_note(same)[1] == ""
    dec = A.decide({QWEN: A.primary_block(by, items, 1000)}, {QWEN: {"comprehension_blind_passes": True, "positive_control_passes": True}},
                   by_class={QWEN: by_class})
    assert "averages opposite class effects" in dec["per_model"][QWEN]["reading"] and dec["per_model"][QWEN]["per_class"] == table


def test_smoke_set_adds_the_named_longest_item(built, runs):
    gen, _ = built
    cands = [t for t in gen if A.is_candidate(t)]
    picked = {t["id"] for t in A.auth_items(gen, 1)}
    extra = next(t["id"] for t in cands if t["id"] not in picked)
    items, comp, _ = A.chosen_items({**_cfg(), "items_per_class": 1, "extra_items": [extra]})
    assert [t["id"] for t in items][-1] == extra and len(items) == 3
    with pytest.raises(ValueError):
        A.chosen_items({**_cfg(), "extra_items": [next(t["id"] for t in gen if not A.is_candidate(t))]})
    smoke = load_config("tau_auth_score_smoke")
    assert smoke["extra_items"] == ["tau-g-77cc59152d73"] and smoke["comprehension_per_class"] == 4
    sb = (ROOT / "pace" / "tau_auth_score.sbatch").read_text()
    assert "#SBATCH --constraint=(gpu-l40s|gpu-a40|A100-80GB|gpu-h100)" in sb and "gpu-a100" not in sb


def test_inference_in_lin_is_clustered_by_user():
    """Review 1, S6.1: four users with strong user-level effects; item-level inference would be far narrower."""
    items = [{"id": f"t{i:02d}", "user_id": f"u{i // 10}", "cls": "status"} for i in range(40)]
    user_effect = {"u0": 3.0, "u1": 3.0, "u2": 3.0, "u3": -2.0}
    by = {}
    for i, t in enumerate(items):
        by[(t["id"], "main", "fail", "first", "aligned")] = _rec(user_effect[t["user_id"]] + 0.01 * (i % 3))
        by[(t["id"], "main", "fail", "first", "swapped")] = _rec(0.0)
    blk = A.lin(by, items, A.q_terms("G", "main", "fail"), n_perm=2000)
    assert blk["n_users"] == 4
    assert blk["perm_p"] > 0.1           # 4 clusters: no sign-flip p below 1/8; item-level flips would give ~0
    assert blk["ci"][0] < 0.5 and blk["ci"][1] > 2.9  # user resampling spans the user means; item resampling would not


def test_positive_control_needs_its_ci_above_zero():
    items = _items(40, 20)
    by = _scores(items, _fail((4.0, 1.8), (4.0, 1.8)), _witness(False), noise=0.0)
    for t in items:  # the authorization arm's witness swapped line: half the users -8.4, half +6.0 relative to aligned
        k = (t["id"], "auth", "witness", "first", "swapped")
        by[k] = _rec(by[k]["L"] + (-8.4 if int(t["user_id"][1:]) % 2 else 6.0))
    pc = A.positive_control(by, items)
    assert pc["estimate"] == pytest.approx(1.2, abs=0.01) and pc["ci"][0] < 0 and not pc["passes"]


def test_scoring_refuses_saved_records_of_other_items(runs, monkeypatch):
    monkeypatch.setenv("WF_SHARD", "0")
    cfg = _cfg(run_id="tau_auth_stale_resume")
    assert A.run_tau_auth_score(cfg) == 0
    path = runs / "tau_auth_stale_resume" / "shard_00_of_02" / "scoring.jsonl"
    recs = read_jsonl(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in [{**recs[0], "item_hash": "0" * 16}] + recs[1:5]))
    with pytest.raises(SystemExit, match="different items"):
        A.run_tau_auth_score({**cfg, "resume": True})


def test_prompt_check_catches_an_arm_dependent_final_turn_and_a_changed_main_arm(built, monkeypatch):
    gen, few = built
    t = A.auth_items(gen, 1)[0]
    real = A.build_messages

    def blind_final(item, cond, fs, *, readout="first", action="fail", arm="main"):
        ms = real(item, cond, fs, readout=readout, action=action, arm=arm)
        if arm == "blind":
            ms[-1] = {**ms[-1], "content": ms[-1]["content"] + " "}
        return ms
    monkeypatch.setattr(A, "build_messages", blind_final)
    assert any("final turn differs from main's" in p for p in A.check_item_prompts(t, few, ("first",)))
    monkeypatch.setattr(A, "build_messages", real)
    real_conf = PR.build_messages
    monkeypatch.setattr(PR, "build_messages", lambda *a, **k: real_conf(*a, **k)[:-1] + [{"role": "user", "content": "x"}])
    assert any("differs from the confirmatory prompt" in p for p in A.check_item_prompts(t, few, ("first",)))


def test_smoke_analysis_reports_the_gates(runs, monkeypatch):
    smoke = {**load_config("tau_auth_score_smoke"), "run_id": "tau_auth_score_smoke2", "expected_outputs_sha256": _pins(),
             "items_per_class": 2, "comprehension_per_class": 2, "extra_items": []}
    assert A.run_tau_auth_score(smoke) == 0
    real = A.load_config
    monkeypatch.setattr(A, "load_config", lambda name: smoke if name == "tau_auth_score_smoke2" else
                        (_cfg() if name == "tau_auth_score" else real(name)))
    assert A.run_tau_auth_analysis({**load_config("tau_auth_smoke_analysis"), "run_id": "tau_auth_smoke_analysis2",
                                    "source": "tau_auth_score_smoke2"}) == 0
    m = json.loads((runs / "tau_auth_smoke_analysis2" / "metrics.json").read_text())
    g = m["models"][QWEN]["gates_readable"]
    assert set(g["comprehension"]) == set(A.ARMS) and g["comprehension"]["blind"]["questions"]["K1"]["n"] == 4
    assert set(g["positive_control"]) == {"pc_nats", "pc_ci", "passes", "n_items"}


def test_a_non_finite_per_item_value_is_left_out():
    items = _items(6, 3)
    by = {}
    for i, t in enumerate(items):
        by[(t["id"], "main", "fail", "first", "aligned")] = _rec(float("nan") if i == 0 else 1.0)
        by[(t["id"], "main", "fail", "first", "swapped")] = _rec(0.0)
    blk = A.lin(by, items, A.q_terms("G", "main", "fail"), n_perm=100)
    assert blk["n_items"] == 5 and blk["estimate"] == 1.0


# ---------------- extension 2026-11-25: Llama-3.2-3B (design note, "Extension 2026-11-25") ----------------

L32 = "llama3.2-3b"


def test_extension_runs_and_configs_differ_from_the_frozen_ones_only_in_model_and_names():
    for name in ("tau_auth_l32_score", "tau_auth_l32_score_smoke", "tau_auth_l32_analysis", "tau_auth_l32_smoke_analysis"):
        assert name in RUNNERS and RUNNERS[name].__code__.co_consts == RUNNERS[name.replace("_l32", "")].__code__.co_consts
    for name, renamed in (("tau_auth_score", ()), ("tau_auth_score_smoke", ()),
                          ("tau_auth_smoke_analysis", ("source", "project_to")), ("tau_auth_analysis", ("source",))):
        frozen, ext = load_config(name), load_config(name.replace("tau_auth", "tau_auth_l32"))
        assert ext["models"] == [L32] and ext["run_id"] == name.replace("tau_auth", "tau_auth_l32")
        for k in renamed:
            assert ext[k] == frozen[k].replace("tau_auth", "tau_auth_l32")
        same = set(frozen) - {"models", "run_id", *renamed}
        assert {k: ext[k] for k in same} == {k: frozen[k] for k in same}
        assert set(ext) - set(frozen) == ({"holm_family"} if name == "tau_auth_analysis" else set())
    assert load_config("tau_auth_l32_analysis")["holm_family"] == [L32]


def test_extension_margin_is_a_third_of_its_request_component_and_frozen_values_stay():
    assert A.EXTENSION_MARGIN == {L32: round(0.47444510337500617 / 3, 2)} == {L32: 0.16}
    assert A.MARGIN == {"qwen3-8b-fewshot": 0.72, "llama3.1-8b-fewshot": 0.17}


def test_extension_scorer_uses_the_wave_a_spec(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    s = A.make_scorer(L32)
    assert s.model_name == "meta-llama/Llama-3.2-3B-Instruct" and s.model_tag == L32
    assert s.revision == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert s.template_settings() == {"enable_thinking": False, "date_string": "26 Jul 2024"}
    assert A.make_scorer(LLAMA).template_settings() == {"enable_thinking": False}  # frozen models unchanged
    with pytest.raises(KeyError):
        A.make_scorer("mistral-7b-v0.3")


def test_token_check_renders_the_extension_model_with_its_fixed_date():
    from wf.tau import auth_token_check as T

    assert T.template_kwargs("meta-llama/Llama-3.2-3B-Instruct") == {"date_string": "26 Jul 2024"}
    assert T.template_kwargs("meta-llama/Llama-3.1-8B-Instruct") == {} == T.template_kwargs("Qwen/Qwen3-8B")


def test_extension_holm_family_has_the_models_four_tests():
    primary = {L32: {x: {part: {"perm_p": 0.002} for part in A.PRIMARY_PARTS} for x in A.PRIMARY_COMPONENTS}}
    adj = A.holm_family(primary, (L32,))
    assert len(adj) == 4 and all(v == pytest.approx(0.008) for v in adj.values())
    assert len(A.holm_family(primary)) == 8  # the frozen family is unchanged
    # decide(): the extension model is decided in its own family with its own margin; a raw 0.002 adjusts to 0.008
    blocks = {"main": _blk(0.5, 0.4, 0.6), "Z": {**_blk(0.0, -0.1, 0.1), "perm_p": 0.5},
              "R": {**_blk(0.5, 0.4, 0.6), "perm_p": 0.002}}
    gates = {L32: {"comprehension_blind_passes": True, "positive_control_passes": True}}
    dec = A.decide({L32: {x: blocks for x in A.PRIMARY_COMPONENTS}}, gates, family=(L32,))
    assert set(dec["per_model"]) == {L32} and dec["per_model"][L32]["margin"] == 0.16
    assert dec["per_model"][L32]["categories"] == {"W": A.RELEVANCE, "M": A.RELEVANCE}
    # Z's CI [-0.1, 0.1] lies inside (-0.16, 0.16), so it is negligible; at a margin of 0.09 it would not be
    assert A.component_category(blocks["main"], blocks["Z"], blocks["R"], 1.0, 0.008, 0.09) != A.RELEVANCE
    gates[L32]["positive_control_passes"] = False
    dec = A.decide({L32: {x: blocks for x in A.PRIMARY_COMPONENTS}}, gates, family=(L32,))
    assert dec["per_model"][L32]["categories"] == {"W": "inconclusive (gate)", "M": "inconclusive (gate)"}


def test_extension_score_then_analysis(runs, monkeypatch):
    ext = {**_cfg(run_id="tau_auth_l32_score"), "models": [L32]}
    for i in (0, 1):
        monkeypatch.setenv("WF_SHARD", str(i))
        assert A.run_tau_auth_score(ext) == 0
    monkeypatch.delenv("WF_SHARD")
    recs = [r for i in (0, 1) for r in read_jsonl(runs / "tau_auth_l32_score" / f"shard_{i:02d}_of_02" / "scoring.jsonl")]
    assert {r["model"] for r in recs} == {L32}
    real = A.load_config
    monkeypatch.setattr(A, "load_config", lambda name: ext if name == "tau_auth_l32_score" else real(name))
    assert A.run_tau_auth_analysis({**load_config("tau_auth_l32_analysis"), "n_perm_primary": 200}) == 0
    m = json.loads((runs / "tau_auth_l32_analysis" / "metrics.json").read_text())
    assert m["missing_scores"] == 0 and m["stale_scores"] == 0 and set(m["models"]) == {L32}
    assert set(m["decision"]["per_model"]) == {L32} and len(m["decision"]["holm_adjusted_p"]) == 4
    assert m["decision"]["per_model"][L32]["margin"] == 0.16
    # the analysis refuses a model outside its Holm family (the frozen family of eight does not hold it)
    with pytest.raises(SystemExit):
        A.run_tau_auth_analysis({**load_config("tau_auth_l32_analysis"), "run_id": "tau_auth_l32_nofamily",
                                 "holm_family": list(P.CO_PRIMARY), "n_perm_primary": 200})
