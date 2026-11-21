"""The exploratory TextCraft Llama diagnostics (wf/textcraft/dx.py, scripts/textcraft_llama_dx.py) in mock mode."""
import importlib.util
import json
from pathlib import Path

import pytest

from wf.runs import RUNNERS, load_config
from wf.textcraft import dx as X
from wf.textcraft import prompts as P
from wf.textcraft import score as S
from wf.textcraft import verify as V
from wf.util import read_jsonl

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "FILED.txt").write_text("filed_date: x\n")
    return tmp_path


def _manifest(tmp_path, items):
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in items))
    m = {"items_file": str(path), "items_sha256": V.sha256(path), "fewshot_file": str(P.FEWSHOT),
         "fewshot_sha256": V.sha256(P.FEWSHOT), "verified": True, "why_not": []}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return str(tmp_path / "manifest.json")


def _small_items():
    """Every item of a few pairs, chosen so that both roles have main couples (a test on an empty draw is vacuous)."""
    items = V.read_items()
    pairs = set()
    for role in ("goal", "ingredient"):
        pairs |= set(sorted({it["pair_id"] for it in items if it["population"] == "main" and it["role"] == role})[:4])
    return [it for it in items if it["pair_id"] in pairs]


def test_sample_couples_whole_seeded_and_role():
    items = V.read_items()
    a = X.sample_couples(items, "main", 20, "t", "goal")
    b = X.sample_couples(items, "main", 20, "t", "goal")
    assert [t["id"] for t in a] == [t["id"] for t in b] and len(a) == 40
    assert all(t["role"] == "goal" and t["population"] == "main" for t in a)
    by = {}
    for t in a:
        by.setdefault(t["couple_id"], []).append(t)
    assert all(len(v) == 2 for v in by.values())
    assert [t["id"] for t in X.sample_couples(items, "main", 20, "other", "goal")] != [t["id"] for t in a]


def test_zero_shot_messages_drop_only_the_examples():
    t = next(it for it in V.read_items() if it["population"] == "main")
    for c in X.CONDITIONS:
        full, zero = P.messages(t, c), X.zero_shot_messages(t, c)
        assert zero == [full[0], full[-1]]
        assert len(full) == 2 + 2 * len(P.fewshot_examples())


def test_outcome_of():
    assert X.outcome_of("Crafted 1 minecraft:melon") == "success"
    assert X.outcome_of(" Could not find enough items to craft minecraft:melon") == "failure"
    assert X.outcome_of("Could not find a valid recipe for x") == "other_failure"
    assert X.outcome_of("Inventory: [melon] (1)") == "other"


def test_configs_registered_and_selection_sizes():
    items = V.read_items()
    for run, n in (("textcraft_dx_zeroshot", 2 * (150 + 100)), ("textcraft_dx_generate", 2 * (60 + 30 + 30)),
                   ("textcraft_dx_objectgoal", 2 * (150 + 100))):
        assert run in RUNNERS
        cfg = load_config(run)
        assert len(X.selection(cfg, items)) == n


@pytest.mark.parametrize("mode", ["zeroshot", "generate", "objectgoal"])
def test_run_mock_writes_every_record_and_resumes(mock_runs, mode):
    items = _small_items()
    role = "ingredient" if mode == "objectgoal" else None
    cfg = {"run_id": f"dx_{mode}", "manifest": _manifest(mock_runs, items), "mode": mode,
           "models": ["llama3.1-8b-fewshot"], "blocks": [{"population": "main", "couples": 3, "role": role}]}
    assert X.run_textcraft_dx(cfg) == 0
    out = mock_runs / "runs" / f"dx_{mode}" / ("generations.jsonl" if mode == "generate" else "scoring.jsonl")
    recs = read_jsonl(out)
    chosen = X.selection(cfg, items)
    assert len(chosen) == 6
    arms = X.arms_of(mode)
    assert len(recs) == len(chosen) * len(arms) and len(arms) == (2 if mode == "objectgoal" else 3)
    assert {(r["transition_id"], r["condition"]) for r in recs} == {(t["id"], c) for t in chosen for c in arms}
    shots = () if mode == "zeroshot" else P.fewshot_examples()
    prep = X.object_goal_item if mode == "objectgoal" else (lambda t: t)
    assert all(r["item_hash"] == S.item_hash(prep(next(t for t in chosen if t["id"] == r["transition_id"])), shots, P.SYSTEM)
               for r in recs)
    if mode == "objectgoal":
        assert all(r["task_line"] == f"Goal: craft {next(t for t in chosen if t['id'] == r['transition_id'])['object']}."
                   for r in recs)
    if mode == "generate":
        assert all(r["outcome"] in ("success", "failure", "other_failure", "other") and r["top_first"] for r in recs)
    # resume: nothing is scored twice
    assert X.run_textcraft_dx({**cfg, "resume": True}) == 0
    assert len(read_jsonl(out)) == len(recs)


def test_zeroshot_hash_differs_from_verified_prompt():
    t = next(it for it in V.read_items() if it["population"] == "main")
    assert S.item_hash(t, (), P.SYSTEM) != S.item_hash(t)


def _script():
    spec = importlib.util.spec_from_file_location("dx_script", REPO / "scripts" / "textcraft_llama_dx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pair_boot_known_values():
    m = _script()
    r = m.pair_boot([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "b"], b=500)
    assert r["est"] == 4.0 and r["n"] == 4 and r["pairs"] == 2
    assert 2.0 <= r["ci"][0] <= r["ci"][1] <= 6.0  # pair means are 2 and 6
    same = m.pair_boot([2.0, 2.0, 2.0], ["a", "b", "c"], b=200)
    assert same["ci"] == [2.0, 2.0]


def test_auc_and_field_diff_and_loo():
    m = _script()
    assert m.auc([3, 4], [1, 2]) == 1.0 and m.auc([1], [1]) == 0.5 and m.auc([1, 2], [3, 4]) == 0.0
    by = {("x", "aligned"): {"L": 2.0}, ("x", "swapped"): {"L": 5.0}}
    r = m.diff_boot(by, [{"id": "x", "pair_id": "p"}, {"id": "y", "pair_id": "q"}], m.field_diff("aligned", "swapped"))
    assert r["est"] == -3.0 and r["n"] == 1  # item y has no scores and is skipped
    loo = m.leave_one_pair_out([1.0, 1.0, 4.0], ["a", "a", "b"])
    assert loo["min"] == ("b", 1.0) and loo["max"] == ("a", 4.0)


def test_object_goal_prompt_differs_from_aligned_only_in_the_goal_line():
    t = next(it for it in V.read_items() if it["population"] == "main" and it["role"] == "ingredient")
    for arm in ("aligned", "yn_aligned"):
        a, o = P.messages(t, arm), P.messages(X.object_goal_item(t), arm)
        assert a[:-1] == o[:-1]
        la, lo = a[-1]["content"].splitlines(), o[-1]["content"].splitlines()
        diff = [(x, y) for x, y in zip(la, lo) if x != y]
        assert len(la) == len(lo) and diff == [(t["task_line"], f"Goal: craft {t['object']}.")]


def test_object_goal_arms_score_the_right_candidates(mock_runs, monkeypatch):
    from wf.scoring.logodds import LogoddsScorer
    calls = []
    real = LogoddsScorer.score_pair

    def spy(self, transition_id, condition, messages, o_plus, o_minus):
        calls.append((condition, messages[-1]["content"], o_plus, o_minus))
        return real(self, transition_id, condition, messages, o_plus, o_minus)

    monkeypatch.setattr(LogoddsScorer, "score_pair", spy)
    items = _small_items()
    cfg = {"run_id": "dx_og", "manifest": _manifest(mock_runs, items), "mode": "objectgoal",
           "models": ["llama3.1-8b-fewshot"], "blocks": [{"population": "main", "couples": 2, "role": "ingredient"}]}
    assert X.run_textcraft_dx(cfg) == 0
    chosen = {t["id"]: t for t in X.selection(cfg, items)}
    assert len(chosen) == 4 and all(t["role"] == "ingredient" for t in chosen.values())
    assert calls and len(calls) == 2 * len(chosen)
    for cond, query, plus, minus in calls:
        assert f"Goal: craft " in query
        if cond == "yn_object_goal":
            assert (plus, minus) == ("Yes", "No") and query.endswith(P.YN_QUESTION)
        else:
            assert cond == "object_goal" and query.endswith(P.NEXT) and plus.startswith("Crafted ")
            assert minus.startswith("Could not find enough items")
