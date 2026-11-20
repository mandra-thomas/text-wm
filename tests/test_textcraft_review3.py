"""Tests for the final amendments before any score (notes/textcraft_design.md, section 13): review 3's surviving
mutants (R3-11, R3-12, R3-14, R3-15, R3-17, R3-21), the counting readout on whole couples and its competence gate,
the outcome table as code, the three equivalence scales, the prompt context in the item hash, withheld verdicts,
and the exploratory prompt variants."""
import json
import random

import pytest

from wf.analysis.stats import holm
from wf.textcraft import analysis as A
from wf.textcraft import build as B
from wf.textcraft import prompts as P
from wf.textcraft import score as S
from wf.textcraft import verify as V

from test_textcraft_review2 import _readout_fixture
from test_textcraft_score import _manifest, _small_items, mock_runs  # noqa: F401 (fixture)


@pytest.fixture(scope="module")
def items():
    return V.read_items()


def _dl(t, d):
    return [{"transition_id": t["id"], "condition": "aligned", "L": d}, {"transition_id": t["id"], "condition": "swapped", "L": 0.0}]


# ---------------- competence (R3-11, R3-12) and its gate ----------------

def test_competence_needs_the_intervals_lower_end():
    """AUC 0.68 carried by one goal pair: a resample without that pair gives 0.5, so the lower end is not above 0.5
    and the model does not discriminate, although the AUC is above 0.6 (R3-11)."""
    succ = [{"id": f"s{p}{i}", "pair_id": f"p{p}"} for p in range(5) for i in range(10)]
    fail = [{"id": f"f{p}{i}", "pair_id": f"p{p}", "failure_class": "too_few"} for p in range(5) for i in range(10)]
    sc = [{"transition_id": t["id"], "condition": "neutral", "L": 10.0 if t["pair_id"] == "p0" else 0.0} for t in succ]
    sc += [{"transition_id": t["id"], "condition": "neutral", "L": -10.0 if t["pair_id"] == "p0" else 0.0} for t in fail]
    comp = A.competence(sc, succ, fail)
    assert comp["all"]["auc"] == pytest.approx(0.68) and comp["all"]["ci"][0] <= 0.5
    assert not comp["all"]["discriminates"]


def test_competence_per_class_and_named_sets_read_only_their_failures():
    """Each class's AUC uses that class's failures only (R3-12); a named set in `also` uses exactly that set."""
    succ = [{"id": f"s{i}", "pair_id": f"p{i % 8}"} for i in range(40)]
    low = [{"id": f"a{i}", "pair_id": f"p{i % 8}", "failure_class": "used_up"} for i in range(40)]
    high = [{"id": f"b{i}", "pair_id": f"p{i % 8}", "failure_class": "too_few"} for i in range(40)]
    sc = [{"transition_id": t["id"], "condition": "neutral", "L": 0.0} for t in succ]
    sc += [{"transition_id": t["id"], "condition": "neutral", "L": -5.0} for t in low]
    sc += [{"transition_id": t["id"], "condition": "neutral", "L": 5.0} for t in high]
    comp = A.competence(sc, succ, low + high, also={"counting": low[:20]})
    assert comp["used_up"]["auc"] == pytest.approx(1.0) and comp["used_up"]["discriminates"]
    assert comp["too_few"]["auc"] == pytest.approx(0.0) and not comp["too_few"]["discriminates"]
    assert comp["all"]["auc"] == pytest.approx(0.5)
    assert comp["counting"]["auc"] == pytest.approx(1.0) and comp["counting"]["discriminates"]


def test_counting_reading_is_gated_by_competence_on_its_items():
    named = {"counting": {"estimate": 2.0, "ci": [1.0, 3.0], "holm_p_within_model": 0.001}}
    ok, bad = {"counting": {"discriminates": True}}, {"counting": {"discriminates": False}}
    assert A.counting_reading(named, ok) == "confirmed"
    assert A.counting_reading(named, bad).startswith("withheld")
    assert A.counting_reading({"counting": {**named["counting"], "holm_p_within_model": 0.02}}, ok) == "not confirmed"
    assert A.counting_reading({"counting": {"estimate": -2.0, "ci": [-3.0, -1.0], "holm_p_within_model": 0.001}}, ok) == "negative"


# ---------------- named readouts: Holm family (R3-14), couples (R3-15, R3-21) ----------------

def test_named_readouts_are_holm_adjusted_over_the_listed_family():
    items, scores, add, k = _readout_fixture()
    r = A.named_readouts(scores, items)
    assert A.HOLM_FAMILY == ("goal_role", "ingredient_with_overlap", "usefulness_only", "yes_no", "counting")
    want = holm({name: r[name]["perm_p"] for name in A.HOLM_FAMILY})
    for name in A.HOLM_FAMILY:
        assert r[name]["holm_p_within_model"] == pytest.approx(want[name])
    assert any(r[name]["holm_p_within_model"] > r[name]["perm_p"] for name in A.HOLM_FAMILY)


def test_overlap_and_counting_readouts_take_whole_couples_only():
    """A couple with one overlapping member (R3-15) and a couple with one counting member (R3-21) are in neither
    readout, though one of their items qualifies on its own."""
    items, scores, add, k = _readout_fixture()
    add(k, "never_obtained", "ingredient", True, False, "tie", False, False, 50.0)
    add(k + 1, "never_obtained", "ingredient", False, False, "tie", False, False, 50.0)
    add(k + 2, "too_few", "ingredient", False, True, "tie", False, True, 77.0)  # in no overlap or usefulness readout
    add(k + 3, "too_few", "ingredient", False, True, "tie", False, False, 77.0)
    r = A.named_readouts(scores, items)
    assert r["ingredient_with_overlap"]["estimate"] == pytest.approx(3.0)
    assert r["usefulness_only"]["estimate"] == pytest.approx(1.0)
    assert r["counting"]["estimate"] == pytest.approx(6.0)
    assert r["counting"]["n_items"] == 20


def test_counting_couples_in_the_items(items):
    """In the built items 835 main items qualify for counting on their own, 766 in whole couples (design 13); every
    counting couple shares one history, so the inventory flag never splits a couple."""
    main = [t for t in items if t["population"] == "main"]
    core = [t for t in main if A.not_used_up_goal(t)]
    got = A.arithmetic_items(core)
    assert len(got) == 766 and len({t["couple_id"] for t in got}) == 383
    assert sum(A.counting_item(t) for t in core) == 833


# ---------------- inventory contrast (R3-17, S4) ----------------

def _inv_items():
    rows = []
    def add(i, cls, role, bin_, shown, after, d):
        t = {"id": f"i{i}", "pair_id": f"p{i % 5}", "failure_class": cls, "role": role, "length_bin": bin_,
             "history": [["inventory", "Inventory: [x] (1) "]] if shown else [["get 1 x", "Got 1 x"]],
             "inventory_shown_after_last_change": after}
        rows.append((t, d))
    i = 0
    for _ in range(30):  # a big cell: contrast 1
        add(i, "too_few", "goal", "4-8", True, False, 1.0); add(i + 1, "too_few", "goal", "4-8", False, False, 0.0); i += 2
    for _ in range(5):  # a small cell: contrast 10
        add(i, "used_up", "ingredient", "4-8", True, False, 10.0); add(i + 1, "used_up", "ingredient", "4-8", False, False, 0.0); i += 2
    for _ in range(5):  # same bin, other role, shown only: no contrast in its cell
        add(i, "used_up", "goal", "4-8", True, False, 50.0); i += 1
    for _ in range(5):  # readable directly: left out
        add(i, "used_up", "ingredient", "4-8", True, True, -100.0); i += 1
    return [t for t, _d in rows], [s for t, d in rows for s in _dl(t, d)]


def test_inventory_contrast_weights_cells_and_drops_readable_inventories():
    its, scores = _inv_items()
    r = A.inventory_contrast(scores, its)
    assert r["estimate"] == pytest.approx((60 * 1.0 + 10 * 10.0) / 70)  # weighted by cell size, not (1 + 10) / 2
    assert r["n_items"] == [40, 35]  # the five readable items are out; the shown-only cell counts but has no contrast


# ---------------- outcome table as code (S6) and equivalence scales (S5) ----------------

def _eq(two=False, yn=False, st=False):
    return {"two_candidate": {"equivalent": two}, "yes_no": {"equivalent": yn}, "standardized": {"equivalent": st}}


def test_primary_reading_rows():
    good, comp, nocomp = {"estimate": 0.4, "ci": [0.2, 0.6]}, {"all": {"discriminates": True}}, {"all": {"discriminates": False}}
    robust, weak = {"ci": [0.1, 0.5]}, {"ci": [-0.1, 0.5]}
    assert A.primary_reading(good, 0.001, _eq(), comp, robust) == "confirmed"
    assert A.primary_reading(good, 0.001, _eq(two=True), comp, robust).startswith("confirmed and equivalent")
    assert "competence fails" in A.primary_reading(good, 0.001, _eq(), nocomp, robust)
    assert "one-sided coherence" in A.primary_reading(good, 0.001, _eq(), comp, weak)
    assert A.primary_reading(good, 0.02, _eq(), comp, robust) == "inconclusive"
    neg = {"estimate": -0.4, "ci": [-0.6, -0.2]}
    assert A.primary_reading(neg, 0.001, _eq(), comp, robust) == "negative"
    assert A.primary_reading({"estimate": -0.4, "ci": [-0.6, 0.1]}, 0.001, _eq(), comp, robust) == "inconclusive"
    null = {"estimate": 0.01, "ci": [-0.1, 0.1]}
    assert A.primary_reading(null, 0.8, _eq(True, True, True), comp, robust) == "equivalent to zero on every scale"
    assert A.primary_reading(null, 0.8, _eq(True, False, True), comp, robust).endswith("two_candidate, standardized only")
    assert A.primary_reading(null, 0.8, _eq(True, True, True), nocomp, robust).startswith("uninterpretable")


def test_verdicts_name_the_models_confirmed_and_the_negative_rule():
    good = {"estimate": 2.0, "ci": [1.0, 3.0], "perm_p": 0.001}
    v = A.verdicts({S.CO_PRIMARY[0]: good, S.CO_PRIMARY[1]: {**good, "ci": [-1.0, 3.0]}})
    assert v["confirmed_for"] == f"{S.CO_PRIMARY[0]} only" and not v["replicates"]
    v = A.verdicts({m: {"estimate": -2.0, "ci": [-3.0, -1.0], "perm_p": 0.001} for m in S.CO_PRIMARY})
    assert all(v[m]["negative"] for m in S.CO_PRIMARY) and v["confirmed_for"] == "neither model"


def test_equivalence_margins_on_three_scales():
    assert A.YES_NO_MARGIN == {"qwen3-8b-fewshot": 0.47, "llama3.1-8b-fewshot": 0.37}
    assert A.STANDARDIZED_MARGIN == {"qwen3-8b-fewshot": 0.17, "llama3.1-8b-fewshot": 0.14}
    rng = random.Random(3)
    its = [{"id": f"t{i}", "pair_id": f"p{i % 60}"} for i in range(1200)]
    scores = []
    for t in its:
        d = rng.gauss(0.5, 5.0)  # 0.5 nats, 0.1 SD
        scores += [{"transition_id": t["id"], "condition": "aligned", "L": d}, {"transition_id": t["id"], "condition": "swapped", "L": 0.0},
                   {"transition_id": t["id"], "condition": "yn_aligned", "L": d / 20}, {"transition_id": t["id"], "condition": "yn_swapped", "L": 0.0}]
    e = A.equivalences(scores, its, "llama3.1-8b-fewshot")
    assert not e["two_candidate"]["equivalent"]  # 0.5 nats against a 0.3 margin
    assert e["standardized"]["equivalent"] and e["standardized"]["estimate"] == pytest.approx(0.1, abs=0.03)
    assert e["yes_no"]["equivalent"] and e["yes_no"]["estimate"] == pytest.approx(0.025, abs=0.02)
    assert e["standardized"]["margin"] == 0.14 and e["yes_no"]["margin"] == 0.37 and e["two_candidate"]["margin"] == 0.3


# ---------------- the item hash covers the prompt context (review 3, LOW) ----------------

def test_item_hash_covers_the_system_text_and_prompt_examples(items):
    it, shots = items[0], P.fewshot_examples()
    base = S.item_hash(it)
    assert S.item_hash(it, shots) == base and S.item_hash(it, shots, P.SYSTEM) == base
    assert S.item_hash(it, shots, P.SYSTEM + " ") != base
    other = tuple({**ex, "goal": "birch sign"} if i == 2 else ex for i, ex in enumerate(shots))
    assert S.item_hash(it, other) != base
    scores = [{"transition_id": it["id"], "item_hash": S.item_hash(it, other)}]
    assert A.stale_scores(scores, [it], shots) and not A.stale_scores(scores, [it], other)


# ---------------- withheld verdicts are not written as results ----------------

def test_withheld_verdicts_are_not_results(mock_runs):
    its = _small_items()
    cfg = {"run_id": "textcraft_score", "manifest": _manifest(mock_runs, its), "n_shards": 1, "models": [S.CO_PRIMARY[0]]}
    assert S.run_textcraft_score(cfg) == 0
    path = mock_runs / "runs" / "textcraft_score" / "shard_00_of_01" / "scoring.jsonl"
    keep = [line for line in path.read_text().splitlines() if json.loads(line)["condition"] != "yn_neutral"]
    path.write_text("\n".join(keep) + "\n")
    acfg = {"run_id": "textcraft_analysis", "source": "textcraft_score", "manifest": cfg["manifest"], "models": [S.CO_PRIMARY[0]]}
    assert A.run_textcraft_analysis(acfg) == 1
    m = json.loads((mock_runs / "runs" / "textcraft_analysis" / "metrics.json").read_text())
    assert m["verdicts"].startswith("withheld") and "per_model" not in m and "incomplete_estimates_not_results" in m
    status = (mock_runs / "runs" / "textcraft_analysis" / "STATUS.md").read_text()
    assert "withheld" in status and "dL=" not in status and "confirmed" not in status


# ---------------- exploratory prompt variants (should-fix) ----------------

def test_variant_contexts(items):
    shots = P.fewshot_examples()
    s2, system = S.variant_context("examples_on_goal", shots)
    assert system == P.SYSTEM and s2[:2] == shots[:2] and s2[2] == {**shots[2], "goal": "birch sign"}
    pair = next(p for p in B.all_pairs() if p.pair_id == shots[2]["pair_id"])
    fetched = shots[2]["action"].split(" ", 2)[2]
    assert set(pair.goals) == {shots[2]["goal"], "birch sign"} and fetched in B.tree_names("birch sign")
    assert fetched not in B.tree_names(shots[2]["goal"])  # the verified example is off-goal; the variant is on-goal
    s3, system = S.variant_context("empty_start", shots)
    assert s3 == tuple(shots) and system == P.SYSTEM + P.EMPTY_START
    assert S.variant_context(None, shots) == (tuple(shots), P.SYSTEM)
    msgs = P.messages(items[0], "aligned", s3, system)
    assert msgs[0]["content"].endswith(P.EMPTY_START.strip()) and msgs[1:] == P.messages(items[0], "aligned", shots)[1:]


def test_variant_items_are_whole_seeded_main_couples(items):
    for name, n in (("examples_on_goal", 200), ("empty_start", 300)):
        got = S.variant_items(items, name)
        assert len(got) == 2 * n and all(t["population"] == "main" for t in got)
        assert len({t["couple_id"] for t in got}) == n
        assert sorted(t["id"] for t in got) == sorted(t["id"] for t in S.variant_items(list(reversed(items)), name))
    assert {t["id"] for t in S.variant_items(items, "empty_start")} != {t["id"] for t in S.variant_items(items, "examples_on_goal")}


def test_variant_run_and_its_comparison(mock_runs, items):
    cfg = {"run_id": "textcraft_score_empty_start", "manifest": _manifest(mock_runs, items), "n_shards": 1,
           "variant": "empty_start"}
    assert S.run_textcraft_score(cfg) == 0
    path = mock_runs / "runs" / "textcraft_score_empty_start" / "shard_00_of_01" / "scoring.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    shots, system = S.variant_context("empty_start", P.fewshot_examples())
    by_id = {t["id"]: t for t in items}
    assert len(recs) == 600 * 2 * 2 and {r["condition"] for r in recs} == {"aligned", "swapped"}
    assert all(r["variant"] == "empty_start" and r["item_hash"] == S.item_hash(by_id[r["transition_id"]], shots, system) for r in recs)
    main_scores = [{**s, "L": s["L"] - 1.0} if s["condition"] == "aligned" else s for s in recs]  # verified prompt: ΔL lower by 1
    rep = A.variants_report({"variants": {"empty_start": "textcraft_score_empty_start", "examples_on_goal": "nothing_here"}},
                            items, P.fewshot_examples(), main_scores, list(S.CO_PRIMARY))
    assert "not_read" in rep["examples_on_goal"]
    for m in S.CO_PRIMARY:
        r = rep["empty_start"][m]
        assert r["variant_minus_verified"]["estimate"] == pytest.approx(1.0) and r["variant_minus_verified"]["n_items"] == 600
        assert "perm_p" not in json.dumps(r)
    # scores made under the verified prompt are not read as the variant's
    path.write_text("".join(json.dumps({**r, "item_hash": S.item_hash(by_id[r["transition_id"]])}) + "\n" for r in recs))
    rep = A.variants_report({"variants": {"empty_start": "textcraft_score_empty_start"}}, items, P.fewshot_examples(), main_scores, list(S.CO_PRIMARY))
    assert "not_read" in rep["empty_start"]


VARIANT_GOLDEN = {"examples_on_goal": ["cp-p002-9b828291298c", "cp-p011-66fcff52d837", "cp-p012-895738130015"],
                  "empty_start": ["cp-p001-32f3873da4b9", "cp-p001-ae0af22fd342", "cp-p005-c891da9c303f"]}


def test_variant_samples_golden(items):
    for name, want in VARIANT_GOLDEN.items():
        assert sorted({t["couple_id"] for t in S.variant_items(items, name)})[:3] == want


class _Recorder:
    """A stand-in scorer that records the chat it is asked to score."""
    seen: list = []

    def __init__(self, model_name, model_tag):
        self.tag, self.mock = model_tag, False

    def score_pair(self, transition_id, condition, messages, o_plus, o_minus):
        _Recorder.seen.append(messages)
        d = {"transition_id": transition_id, "model": self.tag, "condition": condition, "L": 0.0, "L_first": 0.0}
        return type("R", (), {"to_dict": lambda self: d})()

    def close(self):
        pass


def test_variant_run_scores_and_checks_the_variant_prompt(mock_runs, items, monkeypatch):
    """The empty-start run sends the variant's system text to the scorer and to the token check, and the golden
    rendering is checked with the verified prompt examples (design 13)."""
    monkeypatch.setattr(S, "LogoddsScorer", _Recorder)
    calls = []
    monkeypatch.setattr(S, "token_report", lambda *a: calls.append(a) or {"ok": True})
    _Recorder.seen = []
    for name in ("empty_start", "examples_on_goal"):
        cfg = {"run_id": f"textcraft_score_{name}", "manifest": _manifest(mock_runs, items), "n_shards": 1, "variant": name,
               "models": [S.CO_PRIMARY[0]], "token_check": True}
        assert S.run_textcraft_score(cfg) == 0
    shots = P.fewshot_examples()
    empty, ongoal = _Recorder.seen[0], _Recorder.seen[-1]
    assert empty[0]["content"] == P.SYSTEM + P.EMPTY_START and empty[1:7] == P.messages(items[0], "aligned", shots)[1:7]
    assert ongoal[0]["content"] == P.SYSTEM and "Goal: craft birch sign." in ongoal[5]["content"]
    (_s, _its, sh, arms, _tag, _all, system, golden), (_s2, _its2, sh2, *_rest, system2, golden2) = calls
    assert system == P.SYSTEM + P.EMPTY_START and tuple(sh) == tuple(shots) and tuple(golden) == tuple(shots)
    assert system2 == P.SYSTEM and sh2[2]["goal"] == "birch sign" and tuple(golden2) == tuple(shots) and arms == S.VARIANT_ARMS


def test_token_report_checks_the_golden_with_the_verified_examples(items, monkeypatch):
    from test_textcraft_review2 import _CharTok, _FakeScorer
    from wf.textcraft import golden_chat
    seen = []
    monkeypatch.setattr(golden_chat, "problems", lambda tok, tag, its, shots: seen.append(shots) or [])
    it = next(t for t in items if t["action_kind"] == "craft")
    shots = P.fewshot_examples()
    variant, system = S.variant_context("examples_on_goal", shots)
    assert S.token_report(_FakeScorer(_CharTok()), [it], variant, ("aligned",), S.CO_PRIMARY[0], items, system, shots)["ok"]
    assert seen == [shots]


def test_tokcheck_renders_the_variant_system_text(items):
    from test_textcraft_review2 import _CharTok
    from wf.textcraft import tokcheck as T
    tok, seen = _CharTok(), []
    encode = lambda m: seen.append(m[0]["content"]) or tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    it = next(t for t in items if t["action_kind"] == "craft")
    assert T.item_problems(tok, it, P.fewshot_examples(), ("aligned",), encode, system="SYSTEM X") == []
    assert seen == ["SYSTEM X"]


def test_analysis_reads_scores_made_with_the_manifests_examples(mock_runs, tmp_path):
    """Scores made with the manifest's prompt examples are current for the analysis even when that file is not the
    default one; the analysis hashes with the manifest's examples (design 13)."""
    its = _small_items()
    shots = json.loads(P.FEWSHOT.read_text())
    shots[0]["goal"] = "a different goal"
    alt = tmp_path / "fewshot_alt.json"
    alt.write_text(json.dumps(shots))
    m = json.loads(open(_manifest(mock_runs, its)).read())
    m["fewshot_file"], m["fewshot_sha256"] = str(alt), V.sha256(alt)
    (mock_runs / "manifest.json").write_text(json.dumps(m))
    cfg = {"run_id": "textcraft_score", "manifest": str(mock_runs / "manifest.json"), "n_shards": 1, "models": [S.CO_PRIMARY[0]]}
    assert S.run_textcraft_score(cfg) == 0
    acfg = {"run_id": "textcraft_analysis", "source": "textcraft_score", "manifest": cfg["manifest"], "models": [S.CO_PRIMARY[0]]}
    assert A.run_textcraft_analysis(acfg) == 0
