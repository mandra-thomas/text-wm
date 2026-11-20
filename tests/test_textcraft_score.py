"""TextCraft scoring and analysis runs (wf/textcraft/score.py, analysis.py) in mock mode, and the statistics."""
import json

import pytest

from wf.analysis.stats import cluster_bootstrap
from wf.runs import RUNNERS
from wf.textcraft import analysis as A
from wf.textcraft import prompts as P
from wf.textcraft import score as S
from wf.textcraft import verify as V


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "FILED.txt").write_text("filed_date: x\n")
    return tmp_path


def _manifest(tmp_path, items, verified=True, tamper=False):
    """A manifest over a small items file in tmp_path, as verify.py writes it."""
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in items))
    m = {"items_file": str(path), "items_sha256": V.sha256(path), "fewshot_file": str(P.FEWSHOT),
         "fewshot_sha256": V.sha256(P.FEWSHOT), "verified": verified, "why_not": [] if verified else ["x"]}
    if tamper:
        path.write_text(path.read_text().replace("Could not", "Could  not", 1))
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return str(tmp_path / "manifest.json")


def _small_items():
    items = V.read_items()
    pairs = sorted({it["pair_id"] for it in items if it["population"] == "main"})[:4]
    return [it for it in items if it["pair_id"] in pairs]


def test_runners_registered():
    assert {"textcraft_score", "textcraft_score_smoke", "textcraft_analysis"} <= set(RUNNERS)


def test_refuses_unverified_or_changed_items(mock_runs):
    items = _small_items()
    for kw in ({"verified": False}, {"tamper": True}):
        cfg = {"run_id": "textcraft_score", "manifest": _manifest(mock_runs, items, **kw)}
        assert S.run_textcraft_score(cfg) == 1
    assert isinstance(S.verified_files({"manifest": str(mock_runs / "nothing.json")}), str)


def test_committed_manifest_is_accepted():
    files = S.verified_files({})
    assert not isinstance(files, str) and files[0] == V.B.ITEMS


def test_score_every_arm_both_models_then_resume(mock_runs):
    items = _small_items()
    cfg = {"run_id": "textcraft_score", "manifest": _manifest(mock_runs, items), "n_shards": 2}
    for shard in range(2):
        assert S.run_textcraft_score({**cfg, "shard": shard}) == 0
    recs = [json.loads(line) for shard in range(2)
            for line in (mock_runs / "runs" / "textcraft_score" / f"shard_{shard:02d}_of_02" / "scoring.jsonl").read_text().splitlines()]
    assert len(recs) == len(items) * len(P.ARMS) * 2
    assert {(r["model"], r["condition"]) for r in recs} == {(m, a) for m in S.CO_PRIMARY for a in P.ARMS}
    assert all(r["game_id"] == r["pair_id"] and r["item_hash"] for r in recs)
    shard0 = {it["pair_id"] for it in S.shard_items(items, 0, 2)}
    assert shard0 and not shard0 & {it["pair_id"] for it in S.shard_items(items, 1, 2)}
    # resume: nothing is scored twice
    path = mock_runs / "runs" / "textcraft_score" / "shard_00_of_02" / "scoring.jsonl"
    before = path.read_text()
    assert S.run_textcraft_score({**cfg, "shard": 0, "resume": True}) == 0
    assert path.read_text() == before
    # analysis over both shards
    assert A.run_textcraft_analysis({"run_id": "textcraft_analysis", "source": "textcraft_score", "manifest": cfg["manifest"]}) == 0
    m = json.loads((mock_runs / "runs" / "textcraft_analysis" / "metrics.json").read_text())
    assert m["items"]["missing_primary_scores"] == 0 and m["items"]["main"] == sum(it["population"] == "main" for it in items)
    for tag in S.CO_PRIMARY:
        prim = m["per_model"][tag]["primary_delta_L"]
        assert prim["n_pairs"] == 4 and prim["n_items"] == m["items"]["main"]
        assert set(m["per_model"][tag]["main"]["strata"]) == {"failure_class", "length_bin", "role"}
    assert set(m["verdicts"]) == set(S.CO_PRIMARY) | {"replicates"}


def test_smoke_subset_is_fixed_and_spans_items():
    items = V.read_items()
    a, b = S.smoke_subset(items, 20), S.smoke_subset(items, 20)
    assert a == b and len(a) == 20
    assert S.selected_items({"max_items": 20}, items) == a


def test_effect_clusters_by_pair():
    items = [{"id": f"i{k}", "pair_id": f"p{k // 3}"} for k in range(12)]
    scores = []
    for k, t in enumerate(items):
        scores += [{"transition_id": t["id"], "condition": "aligned", "L": 1.0 + k, "L_first": 0.5},
                   {"transition_id": t["id"], "condition": "swapped", "L": 0.0, "L_first": 0.0}]
    e = A.effect(scores, items, "aligned", "swapped")
    assert e["estimate"] == pytest.approx(6.5) and e["n_items"] == 12 and e["n_pairs"] == 4
    boot = cluster_bootstrap(lambda xs: sum(xs) / len(xs), [1.0 + k for k in range(12)], [t["pair_id"] for t in items], seed=A.C.RNG_SEED)
    assert e["ci"] == [boot["ci_low"], boot["ci_high"]]
    # four clusters, all positive: an exact pair-level sign flip gives p = 2/16 (sampled flips land near it);
    # flipping the twelve items one by one would give about 2/4096
    assert e["perm_p"] > 0.05
    assert A.effect(scores, items, "aligned", "swapped", "L_first")["estimate"] == pytest.approx(0.5)
    dp = A.effect(scores, items, "aligned", "swapped", "p")["estimate"]
    assert 0 < dp < 50


def test_verdicts_use_holm_over_both_models():
    good = {"estimate": 2.0, "ci": [1.0, 3.0], "perm_p": 0.001}
    v = A.verdicts({m: good for m in S.CO_PRIMARY})
    assert v["replicates"] and all(v[m]["holm_p"] == pytest.approx(0.002) for m in S.CO_PRIMARY)
    v = A.verdicts({S.CO_PRIMARY[0]: good})  # an unscored model counts as p = 1 and is not confirmed
    assert not v["replicates"] and v[S.CO_PRIMARY[0]]["confirmed"] and v[S.CO_PRIMARY[0]]["holm_p"] == pytest.approx(0.002)
    v = A.verdicts({m: {**good, "ci": [-0.1, 3.0]} for m in S.CO_PRIMARY})
    assert not v["replicates"]
    v = A.verdicts({m: {**good, "perm_p": 0.006} for m in S.CO_PRIMARY})
    assert not any(v[m]["confirmed"] for m in S.CO_PRIMARY)  # 0.012 after Holm
    assert A.verdicts({})[S.CO_PRIMARY[0]]["holm_p"] == 1.0 and not A.verdicts({})["replicates"]


def test_item_hash_changes_with_the_prompt_inputs():
    it = V.read_items()[0]
    assert S.item_hash(it) != S.item_hash({**it, "swapped_task_line": "Goal: craft x."})
    assert S.item_hash(it) != S.item_hash({**it, "history": it["history"] + [["inventory", "Inventory: x"]]})
