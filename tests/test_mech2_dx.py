"""Tests of the statistics helpers in scripts/mech2_dx.py (the mechanism battery's exploratory deep-dive)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mech2_dx.py"
_spec = importlib.util.spec_from_file_location("mech2_dx", _PATH)
dx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dx)


def _toy():
    """Six items in three games; aligned − swapped is 2 on every item, mention_ignore − swapped is 1, 2, 3, ..."""
    info = {f"i{k}": {"game": f"g{k // 2}"} for k in range(6)}
    scores = {}
    for k in range(6):
        i = f"i{k}"
        for fmt in ("cand", "yn"):
            scores[("m", i, "swapped", fmt)] = (0.0, 0.0)
            scores[("m", i, "aligned", fmt)] = (2.0, 1.0)
            scores[("m", i, "mention_ignore", fmt)] = (float(k + 1), 0.5)
    return scores, info


def test_estimate_and_ratio_match_plain_means():
    scores, info = _toy()
    st = dx.Stats(scores, info, B=500)
    e = st.est("m", list(info), dx.KEY["P3"][0], "cand")
    assert e["n_items"] == 6 and e["n_games"] == 3
    assert e["estimate"] == pytest.approx(3.5)
    assert e["aligned_effect"] == pytest.approx(2.0)
    assert e["r"] == pytest.approx(1.75)
    assert e["ci"][0] <= 3.5 <= e["ci"][1]
    assert e["r_ci"][0] <= 1.75 <= e["r_ci"][1]


def test_field_one_reads_the_first_token_term():
    scores, info = _toy()
    e = dx.Stats(scores, info, B=200).est("m", list(info), dx.KEY["P3"][0], "cand", field=1)
    assert e["estimate"] == pytest.approx(0.5)


def test_items_without_a_score_are_dropped():
    scores, info = _toy()
    del scores[("m", "i0", "mention_ignore", "cand")]
    e = dx.Stats(scores, info, B=200).est("m", list(info), dx.KEY["P3"][0], "cand")
    assert e["n_items"] == 5 and e["estimate"] == pytest.approx(4.0)


def test_group_difference_is_a_minus_b():
    scores, info = _toy()
    g = dx.Stats(scores, info, B=300).group_diff("m", ["i4", "i5"], ["i0", "i1"], dx.KEY["P3"][0], "cand")
    assert g["estimate"] == pytest.approx(5.5 - 1.5)
    assert g["n_a"] == 2 and g["n_b"] == 2


def test_tails_follow_the_direction_of_the_mean():
    d = np.array([-10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5])
    t = dx.tails(d)
    assert t["direction_of_mean"] == "negative"
    # the top 10% (one item) in the direction of the mean is -10 of the summed negative -18
    assert t["top10pct_share_of_contributions_in_direction_of_mean"] == pytest.approx(10 / 18)
    assert t["median"] == pytest.approx(-1.0)
    assert t["share_above_0"] == pytest.approx(0.1)


def test_leave_one_game_out_and_game_shares():
    d = np.array([-1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
    r = np.full(9, 2.0)
    pg = dx.per_game(d, r, ["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    # leaving out a, b or c gives means 3.5, 2 and 0.5
    assert pg["loo_estimate_range"] == [pytest.approx(0.5), pytest.approx(3.5)]
    assert pg["loo_r_range"] == [pytest.approx(0.25), pytest.approx(1.75)]
    assert pg["n_games_3plus_items"] == 3
    assert pg["share_games_3plus_mean_above_0"] == pytest.approx(2 / 3)
    # total 18; game c carries 15
    assert pg["largest_game_share_of_total"] == pytest.approx(15 / 18)


def test_words_drop_articles_prepositions_and_numbers():
    assert dx.words("take bowl 1 from cabinet 1") == {"take", "bowl", "cabinet"}
    assert dx.words("Your task is to: put a clean pan in cabinet.") == {"put", "clean", "pan", "cabinet"}
