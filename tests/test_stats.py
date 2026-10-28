from wf.analysis.stats import cluster_bootstrap, delta_p, icc, pair_permutation, rank_outcome


def test_delta_p_and_icc():
    transitions = [
        {
            "id": "t1",
            "game_id": "g1",
            "context_determined": True,
            "true_obs": "Nothing happens.",
            "history_goal_agnostic": True,
        },
        {
            "id": "t2",
            "game_id": "g1",
            "context_determined": True,
            "true_obs": "Nothing happens.",
            "history_goal_agnostic": True,
        },
        {
            "id": "t3",
            "game_id": "g2",
            "context_determined": True,
            "true_obs": "Nothing happens.",
            "history_goal_agnostic": True,
        },
    ]
    scores = []
    for tid, La, Ls in [("t1", 1.0, 0.0), ("t2", 0.5, -0.5), ("t3", 2.0, 0.0)]:
        scores.append({"transition_id": tid, "condition": "aligned", "L": La, "model": "m"})
        scores.append({"transition_id": tid, "condition": "swapped", "L": Ls, "model": "m"})
    out = delta_p(scores, transitions)
    assert out["n_items"] == 3
    assert out["delta_p_pp"] == out["delta_p_pp"]  # not nan
    diffs = out["item_diffs"]
    # rebuild clusters
    clusters = ["g1", "g1", "g2"]
    assert icc(diffs, clusters) == icc(diffs, clusters)
    p = pair_permutation(diffs, n=200, seed=0)
    assert 0 < p <= 1
    boot = cluster_bootstrap(lambda xs: sum(xs) / len(xs), diffs, clusters, B=50, seed=0)
    assert "ci_low" in boot


def test_rank_outcome():
    eps = [
        {"game_id": "a", "success": True, "steps": 10},
        {"game_id": "a", "success": False, "steps": 30},
        {"game_id": "b", "success": True, "steps": 5},
    ]
    r = rank_outcome(eps)
    assert r["n_games"] == 2
