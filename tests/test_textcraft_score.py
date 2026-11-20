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
        assert set(m["per_model"][tag]["descriptive"]["main"]["strata"]) == set(A.STRATA)
        assert m["per_model"][tag]["named"]["counting"]["n_items"] == sum(
            it["population"] == "main" and it["needs_arithmetic"] and not it["inventory_shown_after_last_change"]
            and A.not_used_up_goal(it) for it in items)
        assert m["per_model"][tag]["competence"]["neutral"]["all"]["auc"] == m["per_model"][tag]["competence"]["neutral"]["all"]["auc"]
        assert "perm_p" not in json.dumps(m["per_model"][tag]["descriptive"])  # descriptive readouts carry no p-values
        assert m["per_model"][tag]["descriptive"]["needs_arithmetic_as_logged_in_A2"]["n_items"] == sum(
            it["population"] == "main" and it["needs_arithmetic"] for it in items) < m["items"]["main"]
        assert m["per_model"][tag]["descriptive"]["primary_without_used_up_on_goal"]["n_items"] == sum(
            it["population"] == "main" and A.not_used_up_goal(it) for it in items)
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


# ---------------- gates and readout wiring (review 1, finding 5) ----------------

def test_refuses_a_changed_fewshot_file(mock_runs, tmp_path):
    items = _small_items()
    m = json.loads(open(_manifest(mock_runs, items)).read())
    shots = tmp_path / "fewshot.json"
    shots.write_text(P.FEWSHOT.read_text().replace("Got", "Got ", 1))
    m["fewshot_file"] = str(shots)
    (mock_runs / "manifest.json").write_text(json.dumps(m))
    reason = S.verified_files({"manifest": str(mock_runs / "manifest.json")})
    assert isinstance(reason, str) and "fewshot" in reason


def test_resume_refuses_saved_scores_of_changed_items(mock_runs):
    items = _small_items()
    cfg = {"run_id": "textcraft_score", "manifest": _manifest(mock_runs, items), "n_shards": 1, "arms": ["aligned"],
           "models": [S.CO_PRIMARY[0]]}
    assert S.run_textcraft_score(cfg) == 0
    path = mock_runs / "runs" / "textcraft_score" / "shard_00_of_01" / "scoring.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    recs[0]["item_hash"] = "0" * 16  # a saved score made for an item that has since changed
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit, match="different items"):
        S.run_textcraft_score({**cfg, "resume": True})


def test_item_hash_covers_both_candidate_sets():
    it = V.read_items()[0]
    assert S.item_hash(it) != S.item_hash({**it, "candidates": {**it["candidates"], "o_plus": "x"}})
    assert S.item_hash(it) != S.item_hash({**it, "prefix_candidates": {**it["prefix_candidates"], "o_plus": "x"}})


def test_smoke_subset_golden():
    ids = [t["id"] for t in S.smoke_subset([{"id": f"x{k:03d}"} for k in range(200)], 5)]
    assert ids == SMOKE_GOLDEN, ids


SMOKE_GOLDEN = ["x083", "x110", "x133", "x162", "x166"]


def _wired(items):
    """Scores with a different known value on every arm and field, so each readout can be traced to its inputs."""
    vals = {"aligned": (5.0, 50.0), "swapped": (2.0, 20.0), "neutral": (1.0, 10.0), "yn_aligned": (0.5, 0.5),
            "yn_swapped": (1.5, 1.5), "yn_neutral": (0.0, 0.0), "prefix_aligned": (4.0, 40.0), "prefix_swapped": (3.5, 35.0)}
    return [{"transition_id": t["id"], "condition": arm, "L": L, "L_first": Lf, "model": S.CO_PRIMARY[0],
             "item_hash": S.item_hash(t)} for t in items for arm, (L, Lf) in vals.items()]


def test_readouts_are_wired_to_their_arms():
    items = _small_items()
    r = A.readouts(_wired(items), items)
    assert r["delta_L"]["estimate"] == pytest.approx(3.0)
    assert r["delta_L_first"]["estimate"] == pytest.approx(30.0)  # L_first, not L
    assert r["aligned_minus_neutral"]["estimate"] == pytest.approx(4.0)
    assert r["swapped_minus_neutral"]["estimate"] == pytest.approx(1.0)
    assert r["yes_no_aligned_minus_swapped"]["estimate"] == pytest.approx(-1.0)  # aligned minus swapped, sign kept
    assert r["yes_no_aligned_minus_neutral"]["estimate"] == pytest.approx(0.5)
    assert r["delta_L_prefix"]["estimate"] == pytest.approx(0.5)
    assert r["delta_L_name_remainder"]["estimate"] == pytest.approx((5.0 - 4.0) - (2.0 - 3.5))
    st = A.strata(_wired(items), items)
    assert set(st) == set(A.STRATA)
    for groups in st.values():
        for g in groups.values():
            assert g["delta_L"]["estimate"] == pytest.approx(3.0)  # aligned minus swapped, never neutral
            assert g["aligned_minus_neutral"]["estimate"] == pytest.approx(4.0)
            assert g["yes_no"]["estimate"] == pytest.approx(-1.0) and g["delta_L_prefix"]["estimate"] == pytest.approx(0.5)


def test_analysis_flags_missing_scores_and_refuses_stale_ones(mock_runs):
    items = _small_items()
    cfg = {"run_id": "textcraft_score", "manifest": _manifest(mock_runs, items), "n_shards": 1,
           "models": [S.CO_PRIMARY[0]]}
    assert S.run_textcraft_score(cfg) == 0
    path = mock_runs / "runs" / "textcraft_score" / "shard_00_of_01" / "scoring.jsonl"
    full = path.read_text().splitlines()
    acfg = {"run_id": "textcraft_analysis", "source": "textcraft_score", "manifest": cfg["manifest"], "models": [S.CO_PRIMARY[0]]}
    assert A.run_textcraft_analysis(acfg) == 0
    main_id = next(it["id"] for it in items if it["population"] == "main")
    path.write_text("\n".join(l for l in full if not (json.loads(l)["transition_id"] == main_id
                                                    and json.loads(l)["condition"] == "aligned")) + "\n")
    assert A.run_textcraft_analysis(acfg) == 1
    m = json.loads((mock_runs / "runs" / "textcraft_analysis" / "metrics.json").read_text())
    assert m["items"]["missing_primary_scores"] == 1
    # a missing score on a non-primary arm withholds the verdict too (design 12, B3)
    path.write_text("\n".join(l for l in full if not (json.loads(l)["transition_id"] == main_id
                                                    and json.loads(l)["condition"] == "prefix_swapped")) + "\n")
    assert A.run_textcraft_analysis(acfg) == 1
    m = json.loads((mock_runs / "runs" / "textcraft_analysis" / "metrics.json").read_text())
    assert m["items"]["missing_primary_scores"] == 0 and m["items"]["missing_scores_any_arm"] == 1
    stale = [json.loads(l) for l in full]
    stale[0]["item_hash"] = "0" * 16
    path.write_text("".join(json.dumps(r) + "\n" for r in stale))
    assert A.run_textcraft_analysis(acfg) == 1
    assert "do not match" in (mock_runs / "runs" / "textcraft_analysis" / "STATUS.md").read_text()


def test_primary_and_secondary_read_only_their_items():
    items = V.read_items()
    pick = [it for it in items if it["population"] == "main"][:30] + [it for it in items if it["population"] != "main"][:30]
    scores = []
    for t in pick:
        big = 0.0 if t["population"] == "main" else 100.0
        bump = 1.0 if t["needs_arithmetic"] else 0.0
        scores += [{"transition_id": t["id"], "condition": "aligned", "L": 2.0 + big + bump, "L_first": 0.0},
                   {"transition_id": t["id"], "condition": "swapped", "L": 1.0, "L_first": 0.0}]
    main = [t for t in pick if t["population"] == "main"]
    assert A.effect(scores, main, "aligned", "swapped")["estimate"] < 3.0
    arith = [t for t in main if t["needs_arithmetic"]]
    assert arith and A.effect(scores, arith, "aligned", "swapped")["estimate"] == pytest.approx(2.0)
