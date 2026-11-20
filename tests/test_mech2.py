"""Mechanism battery (wf/mech2.py): templates per task type, prompt differences, items, the tokenization gate, the
analysis on synthetic data with each account's fingerprint planted, and the runner end to end on mock data."""
import json
import math
import random
import re
from pathlib import Path

import pytest

from wf import constants as C
from wf import families, mech2, rq1, rq1v, rq1x
from wf.scoring.logodds import LogoddsScorer
from wf.scoring.prompts import WM_FEWSHOT_SETS
from wf.util import read_jsonl
from tests.test_rq1 import mock_root, mock_score  # noqa: F401

OPENING = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, "
           "a countertop 2, a desklamp 1, a fridge 1, a microwave 1, and a sinkbasin 1.")
# Every task template in the ALFWorld data, two wordings where the type has two, with an apple as the goal.
TASKS = {
    "pick": ("put an apple in fridge.", "put some apple on countertop."),
    "pick_two": ("put two apple in fridge.", "find two apple and put them in fridge."),
    "clean": ("put a clean apple in fridge.", "clean some apple and put it in fridge."),
    "heat": ("put a hot apple in fridge.", "heat some apple and put it in fridge."),
    "cool": ("put a cool apple in countertop.", "cool some apple and put it in countertop."),
    "examine": ("examine the apple with the desklamp.", "look at apple under the desklamp."),
}
ACTIONS = {"take": "take apple 1 from countertop 2", "move": "move apple 1 to fridge 1",
           "heat": "heat apple 1 with microwave 1", "cool": "cool apple 1 with fridge 1",
           "clean": "clean apple 1 with sinkbasin 1"}
LATER = "\n> go to countertop 1\nYou arrive at countertop 1. On the countertop 1, you see a egg 1."


def _item(task: str = TASKS["clean"][0], action: str = ACTIONS["take"], step0: bool = False, **kw) -> dict:
    line = f"Your task is to: {task}"
    t = {"id": "m1", "game_id": "valid_unseen/1", "action": action, "task_line": line,
         "swapped_task_line": line.replace("apple", "egg", 1), "goal_object": "apple", "swapped_goal_object": "egg",
         "history_text": f"{OPENING}\n{line}" + ("" if step0 else LATER), "history_len": 0 if step0 else 1,
         "history_pairs": [] if step0 else [["go to countertop 1", LATER.split("\n")[2]]],
         "determination_reason": "away_from_receptacle", "scene_receptacles": ["cabinet 1", "countertop 1", "countertop 2",
                                                                             "desklamp 1", "fridge 1", "microwave 1", "sinkbasin 1"],
         "candidates": {"o_plus": "You pick up the apple 1 from the countertop 2.", "o_minus": "Nothing happens."}}
    return {**t, **kw}


def _all_variants():
    for ttype, lines in TASKS.items():
        for task in lines:
            for verb, action in ACTIONS.items():
                for step0 in (True, False):
                    yield ttype, verb, _item(task, action, step0)


# ---------------- templates ----------------


def test_task_type_reads_every_template():
    for ttype, lines in TASKS.items():
        for task in lines:
            assert mech2.task_type(f"Your task is to: {task}") == ttype
    with pytest.raises(ValueError):
        mech2.task_type("Your task is to: dance.")


def _arms(t):
    """The arms an item carries: every arm, except negated on items whose action is not a take."""
    return [a for a in mech2.ARMS if a != "negated" or t["action"].startswith("take ")]


def test_every_arm_builds_for_every_task_type_and_verb():
    for ttype, verb, t in _all_variants():
        assert (mech2.task_line(t, "negated") is None) == (verb != "take")
        for arm in _arms(t):
            line = mech2.task_line(t, arm)
            assert line and line.endswith(".") and "  " not in line and ".." not in line, (ttype, verb, arm, line)
            assert mech2.arm_messages(t, arm) and mech2.arm_messages(t, arm, "yn")


def test_templates_read_as_written():
    t = _item(TASKS["clean"][0], ACTIONS["take"])
    assert mech2.task_line(t, "desire_keep") == "Your task is to: make sure that the apple is not taken; leave it where it is."
    assert mech2.task_line(t, "desire_check") == ("Your task is to: find out whether an apple can be taken from the "
                                                  "countertop 2; the user hopes it cannot.")
    assert mech2.task_line(t, "mention_ignore") == "Your task is to: put a clean egg in fridge. Ignore the apple."
    assert mech2.task_line(t, "mention_exists") == "Your task is to: put a clean egg in fridge. There is an apple somewhere in the room."
    assert mech2.task_line(t, "done") == "Your task was to: put a clean apple in fridge. It is already done; nothing is left to do."
    assert mech2.task_line(t, "cancelled") == ("Your task was to: put a clean apple in fridge. The user has cancelled it; "
                                               "nothing is left to do.")
    assert mech2.task_line(t, "neutral") == rq1.ABSENT_LINE
    assert mech2.task_line(t, "negated") == ("Your task is to: put a clean egg in fridge. The user says there is no apple "
                                             "in the countertop 2.")
    for verb, pp, prep, rec in (("move", "moved", "to", "fridge 1"), ("heat", "heated", "with", "microwave 1"),
                                ("cool", "cooled", "with", "fridge 1"), ("clean", "cleaned", "with", "sinkbasin 1")):
        u = _item(action=ACTIONS[verb])
        assert mech2.task_line(u, "desire_keep") == f"Your task is to: make sure that the apple is not {pp}; leave it where it is."
        assert mech2.task_line(u, "desire_check").endswith(f"an apple can be {pp} {prep} the {rec}; the user hopes it cannot.")
    m = _item(goal_object="mug", task_line="Your task is to: put a mug in fridge.", action="take mug 1 from countertop 2",
              swapped_task_line="Your task is to: put a egg in fridge.")
    assert "whether a mug can be taken" in mech2.task_line(m, "desire_check")


def test_existence_presupposed_only_where_designed():
    for _, _, t in _all_variants():
        keep, check = mech2.task_line(t, "desire_keep"), mech2.task_line(t, "desire_check")
        assert "the apple" in keep and "the apple" not in check and "an apple" in check
        assert not re.search(r"\b[a-z]+ \d+\b", keep)  # no instance, so no location claim
        assert set(re.findall(r"\b[a-z]+ \d+\b", check)) == {mech2.parse_action(t["action"])[2]}
        assert "Ignore the apple." in mech2.task_line(t, "mention_ignore")


def test_an_item_the_action_does_not_name_carries_no_desire_arm():
    t = _item(action="take egg 1 from countertop 2")
    assert mech2.task_line(t, "desire_keep") is None and mech2.task_line(t, "desire_check") is None
    assert mech2.task_line(_item(action="open fridge 1"), "desire_keep") is None


def test_described_arms_name_no_object():
    for _, _, t in _all_variants():
        assert "apple" not in mech2.task_line(t, "described") and "egg" not in mech2.task_line(t, "swapped_described")
        assert rq1x.DESCRIPTIONS["apple"] in mech2.task_line(t, "described")


# ---------------- prompts differ only where intended ----------------


def _query(msgs):
    return msgs[-1]["content"]


def test_content_arms_differ_from_aligned_only_in_the_task_line(mock_score):
    for _, _, t in _all_variants():
        for fmt in mech2.FORMATS:
            base = mech2.arm_messages(t, "aligned", fmt)
            assert _query(base).count(t["task_line"]) == 1
            for arm in [a for a in mech2.CONTENT_ARMS if a in _arms(t)]:
                msgs = mech2.arm_messages(t, arm, fmt)
                assert msgs[:-1] == base[:-1], arm
                assert _query(msgs) == _query(base).replace(t["task_line"], mech2.task_line(t, arm)), arm


def test_only_the_real_task_arms_show_the_real_task(mock_score):
    real = {"aligned", "done", "cancelled", "aligned_start", "aligned_end"}
    for _, _, t in _all_variants():
        for arm in _arms(t):
            shown = t["task_line"][len("Your task is to: "):] in _query(mech2.arm_messages(t, arm))
            assert shown == (arm in real), arm


def test_position_arms_move_only_the_task_line(mock_score):
    for _, _, t in _all_variants():
        for arm in mech2.POSITION_ARMS:
            goal, where = arm.rsplit("_", 1)
            plain, moved = _query(mech2.arm_messages(t, goal)), _query(mech2.arm_messages(t, arm))
            line = mech2.task_line(t, goal)
            assert sorted(plain.split("\n")) == sorted(moved.split("\n"))
            assert [x for x in plain.split("\n") if x != line] == [x for x in moved.split("\n") if x != line]
            hist = moved.split("History:\n", 1)[1].split("\n\nAction:", 1)[0].split("\n")
            assert (hist[0] if where == "start" else hist[-1]) == line
            assert mech2.arm_messages(t, arm)[:-1] == mech2.arm_messages(t, goal)[:-1]


def test_usual_position_and_aliases():
    s0, s1 = _item(step0=True), _item()
    assert mech2.usual_position(s0) == "end" and mech2.usual_position(s1) == "line 2"
    assert mech2.alias_of(s0, "aligned_end") == "aligned" and mech2.alias_of(s0, "swapped_end") == "swapped"
    assert mech2.alias_of(s0, "aligned_start") is None and mech2.alias_of(s1, "aligned_end") is None
    assert mech2.alias_of(s1, "aligned_start") is None  # line 2 to line 1 is still a change
    assert mech2.move_task_line(s1["history_text"], "start").split("\n")[0] == s1["task_line"]
    with pytest.raises(ValueError):
        mech2.move_task_line("no task here", "start")


def test_yes_no_differs_only_in_the_question(mock_score):
    for _, _, t in _all_variants():
        for arm in _arms(t):
            c, y = mech2.arm_messages(t, arm), mech2.arm_messages(t, arm, "yn")
            assert c[:-1] == y[:-1]
            assert _query(y) == _query(c)[: -len("Next observation:")] + rq1v.YN_QUESTION


def test_reused_arms_are_the_earlier_runs_prompts(mock_score):
    t = _item()
    assert mech2.arm_messages(t, "aligned") == rq1.arm_messages(t, "aligned")
    assert mech2.arm_messages(t, "swapped") == rq1.arm_messages(t, "swapped")
    assert mech2.arm_messages(t, "neutral") == rq1.arm_messages(t, "absent")
    assert mech2.arm_messages(t, "mention_exists") == rq1.arm_messages(t, "swapped_plus")
    assert mech2.arm_messages(t, "described") == rq1x.arm_messages(t, "aligned_desc")
    assert mech2.arm_messages(t, "swapped_described") == rq1x.arm_messages(t, "swapped_desc")
    assert mech2.arm_messages(t, "aligned", "yn") == rq1v.arm_messages(t, "yn_aligned")
    assert mech2.arm_messages(t, "neutral", "yn") == rq1v.arm_messages(t, "yn_absent")


def test_conditions_round_trip():
    for arm in mech2.ARMS:
        for fmt in mech2.FORMATS:
            assert mech2.split_condition(mech2.condition(arm, fmt)) == (arm, fmt)
    assert mech2.condition("aligned", "yn") == "yn_aligned"


# ---------------- items ----------------


def _primary(n_games=30, seed=1):
    rng = random.Random(seed)
    reasons = ("away_from_receptacle", "not_holding_object", "contents_or_open_state", "hands_full")
    items = []
    for g in range(n_games):
        for k in range(rng.randint(4, 40)):
            step0 = k < 2
            items.append(_item(id=f"g{g}k{k}", game_id=f"valid_unseen/{g}", step0=step0,
                               determination_reason="away_from_receptacle_never_moved" if step0 else rng.choice(reasons)))
    return items


def test_sample_is_stratified_seeded_and_order_free():
    p = _primary()
    s = mech2.stratified_sample(p, 200)
    assert len(s) == 200 and s == mech2.stratified_sample(list(reversed(p)), 200)
    strata = {}
    for t in p:
        strata.setdefault((t["game_id"], t["determination_reason"]), []).append(t)
    for ts in strata.values():
        got = sum(t["id"] in s for t in ts)
        assert abs(got - len(ts) * 200 / len(p)) < 1  # proportional allocation, largest remainder
    assert mech2.stratified_sample(p, 10_000) == {t["id"] for t in p}


def test_battery_items_add_every_step0_item():
    p = _primary()
    b = mech2.battery_items(p, 100)
    step0 = {t["id"] for t in p if t["history_len"] == 0}
    assert b["step0"] == step0 and {t["id"] for t in b["union"]} == b["sample"] | step0


def test_jobs_cover_every_arm_and_skip_aliases():
    s0, s1 = _item(id="a", step0=True), _item(id="b")
    j = {(t["id"], a, f) for t, a, f in mech2.jobs([s0, s1])}
    for arm in mech2.ARMS:
        for f in mech2.FORMATS:
            assert ("b", arm, f) in j
            assert (("a", arm, f) in j) == (arm not in ("aligned_end", "swapped_end"))


def test_done_contradicted_only_when_every_target_was_seen_without_the_object():
    t = _item(TASKS["pick"][0], history_text=f"{OPENING}\nYour task is to: put an apple in fridge.\n> open fridge 1\n"
              "You open the fridge 1. The fridge 1 is open. In it, you see a egg 1.")
    assert mech2.done_contradicted(t)
    seen_apple = t["history_text"].replace("a egg 1", "a apple 2")
    assert not mech2.done_contradicted({**t, "history_text": seen_apple})
    assert not mech2.done_contradicted(_item(TASKS["pick"][0]))  # fridge never seen
    two = {**t, "scene_receptacles": t["scene_receptacles"] + ["fridge 2"]}
    assert not mech2.done_contradicted(two)  # another fridge was not seen
    assert not mech2.done_contradicted(_item(TASKS["examine"][0]))


# ---------------- analysis on synthetic data ----------------

D = 4.0
# Mean of each content arm minus the swapped arm, in units of the aligned effect, for each content account's
# fingerprint (amended after review 1: every value sits well inside its predicted zone).
FINGERPRINTS = {
    "H2_evidence": {"desire_keep": 0.8, "desire_check": 0.6, "mention_ignore": 0.7, "described": 0.9, "done": -0.3,
                    "cancelled": 0.5, "negated": -0.6},
    "H3_salience": {"desire_keep": 0.8, "desire_check": 0.8, "mention_ignore": 0.8, "described": 0.6, "done": 0.8,
                    "cancelled": 0.8, "negated": 0.7},
    "H4_wishful": {"desire_keep": -0.5, "desire_check": -0.5, "mention_ignore": 0.0, "described": 0.9, "done": 0.0,
                   "cancelled": 0.0, "negated": 0.0},
    "H5_policy_blend": {"desire_keep": -0.5, "desire_check": 0.9, "mention_ignore": 0.0, "described": 0.9, "done": 0.0,
                        "cancelled": 0.0, "negated": 0.0},
    "H6_lexical": {"desire_keep": 0.8, "desire_check": 0.8, "mention_ignore": 0.8, "described": 0.0, "done": 0.8,
                   "cancelled": 0.8, "negated": 0.8},
}
REF = ((1, "aligned"), (-1, "swapped"))


def _syn_items(n_games=40, per_game=12):
    items = []
    for g in range(n_games):
        for k in range(per_game):
            task = TASKS["examine"][0] if k % 6 == 5 else TASKS["clean"][0]
            items.append(_item(task, id=f"g{g}k{k}", game_id=f"valid_unseen/{g}", step0=k < 3))
    return items


def _syn_scores(items, fingerprint, position_shift=0.0, seed=7, noise=1.0, yn_fingerprint=None, yn_scale=1.0):
    """Scores in both formats. The yes/no format follows `yn_fingerprint` (default: the same one) with its aligned
    effect `yn_scale` times the two-candidate one."""
    rng = random.Random(seed)
    game = {}
    out = []
    for t in items:
        g = game.setdefault(t["game_id"], rng.gauss(0, 2))
        base = g + rng.gauss(-8, 2)
        for fmt, fp, scale in (("cand", fingerprint, 1.0), ("yn", yn_fingerprint or fingerprint, yn_scale)):
            d = D * scale
            mu = {"aligned": d, "swapped": 0.0, "neutral": -0.4 * d, "mention_exists": 0.9 * d, "swapped_described": 0.0,
                  **{a: v * d for a, v in fp.items()}}
            mu["aligned_start"] = mu["aligned"] + (position_shift * d if t["history_len"] == 0 else 0.0)
            mu["swapped_start"] = mu["swapped"]
            mu["aligned_end"], mu["swapped_end"] = mu["aligned"], mu["swapped"]
            for arm, m in mu.items():
                if mech2.alias_of(t, arm) or mech2.task_line(t, arm) is None:
                    continue
                L = base + m + rng.gauss(0, noise)
                out.append({"transition_id": t["id"], "condition": mech2.condition(arm, fmt), "L": L, "L_first": L / 2,
                            "game_id": t["game_id"], "model": "m"})
    return out


def _pops(items):
    sample = {t["id"] for t in items}
    step0 = {t["id"] for t in items if t["history_len"] == 0}
    return mech2.populations(items, sample, step0)


def _family(scores, items, B=200, fmt="cand"):
    return mech2.primary_family(mech2._lookup(scores), _pops(items), fmt, B=B, perms=200)


@pytest.mark.parametrize("planted", sorted(FINGERPRINTS))
def test_each_planted_fingerprint_is_read_as_its_own_account(planted):
    items = _syn_items()
    fam = _family(_syn_scores(items, FINGERPRINTS[planted]), items)
    ro = fam["readout"]
    assert ro[planted]["n_contradicted"] == 0 and ro[planted]["n_consistent"] == 6, (planted, fam["contrasts"])
    assert ro[planted]["verdict"] == "consistent"
    for other in FINGERPRINTS:
        if other != planted:
            assert ro[other]["n_contradicted"] >= 1 and ro[other]["verdict"] == "contradicted", (planted, other, ro[other])


def test_position_counts_only_for_the_position_account():
    """P6 is read for H6_position only (H2-H5 predict nothing about position); it is read as an artefact only when
    r's interval lies wholly beyond ±0.25."""
    items = _syn_items()
    for h in ("H2_evidence", "H3_salience", "H4_wishful", "H5_policy_blend", "H6_lexical"):
        assert "P6_position_step0" not in mech2.PREDICTIONS[h]
    fam = _family(_syn_scores(items, FINGERPRINTS["H3_salience"], position_shift=-0.6), items)
    assert fam["contrasts"]["P6_position_step0"]["category"] == "NEG"
    assert fam["readout"]["H6_position"]["verdict"] == "consistent"
    for h in ("H2_evidence", "H4_wishful", "H5_policy_blend"):
        assert "P6_position_step0" not in fam["readout"][h]["per_contrast"]
    assert fam["readout"]["H3_salience"]["verdict"] == "consistent"  # a position effect does not count against it
    none = _family(_syn_scores(items, FINGERPRINTS["H3_salience"]), items)
    assert none["contrasts"]["P6_position_step0"]["category"] in ("NULL", "TRIVIAL")
    assert none["readout"]["H6_position"]["verdict"] == "contradicted"
    small = _family(_syn_scores(items, FINGERPRINTS["H3_salience"], position_shift=-0.12, noise=0.3), items)
    assert small["contrasts"]["P6_position_step0"]["category"] == "TRIVIAL"  # significant, inside the band
    assert small["readout"]["H6_position"]["per_contrast"]["P6_position_step0"] == "contradicted"


def test_p6_reads_the_step0_items_only():
    items = _syn_items()
    fam = _family(_syn_scores(items, FINGERPRINTS["H3_salience"], position_shift=-0.5, noise=0.0), items)
    p6 = fam["contrasts"]["P6_position_step0"]
    assert p6["n_items"] == sum(t["history_len"] == 0 for t in items)
    assert abs(p6["estimate"] + 0.5 * D) < 1e-9 and abs(p6["ratio"]["estimate"] + 0.5) < 1e-9


def test_contrast_signs_sizes_and_ratio():
    items = _syn_items()
    by = mech2._lookup(_syn_scores(items, FINGERPRINTS["H4_wishful"], noise=0.0))
    r = mech2.contrast(by, items, ((1, "desire_keep"), (-1, "swapped")), reference=REF, B=50, perms=50)
    assert abs(r["estimate"] + 0.5 * D) < 1e-9 and abs(r["ratio"]["estimate"] + 0.5) < 1e-9
    assert abs(r["reference_effect"] - D) < 1e-9 and r["n_games"] == 40
    assert r["ratio"]["n_dropped"] == 0 and r["ratio"]["estimable"]
    p5 = mech2.contrast(by, items, ((1, "done"), (-1, "cancelled")), B=50, perms=50)
    assert abs(p5["estimate"]) < 1e-9


def test_yes_no_family_reads_yes_no_scores_with_its_own_denominator():
    items = _syn_items()
    scores = _syn_scores(items, FINGERPRINTS["H4_wishful"], noise=0.0, yn_fingerprint=FINGERPRINTS["H3_salience"],
                         yn_scale=0.25)
    block = mech2.model_block(scores, items, {t["id"] for t in items}, {t["id"] for t in items if t["history_len"] == 0},
                              B=30, perms=30)
    c, y = block["primary"]["contrasts"], block["primary_yes_no"]["contrasts"]
    assert abs(c["P1_keep_vs_ignore"]["ratio"]["estimate"] + 0.5) < 1e-9
    assert abs(y["P1_keep_vs_ignore"]["ratio"]["estimate"]) < 1e-9 and abs(y["P2_desire_check"]["ratio"]["estimate"] - 0.8) < 1e-9
    assert abs(y["P2_desire_check"]["estimate"] - 0.8 * 0.25 * D) < 1e-9
    assert abs(block["arms_on_sample"]["desire_check"]["yn_vs_swapped_L"]["ratio"]["estimate"] - 0.8) < 1e-9


def test_sigma_is_read_through_the_sigmoid():
    t = _item()
    by = {(t["id"], "yn_aligned"): {"L": 0.0}, (t["id"], "yn_swapped"): {"L": 2.0}}
    assert mech2._value(by, t, "aligned", "yn", "s") == 0.5
    assert abs(mech2._value(by, t, "swapped", "yn", "s") - 1 / (1 + math.exp(-2.0))) < 1e-12
    r = mech2.contrast(by, [t], ((1, "aligned"), (-1, "swapped")), "yn", field="s", B=5, perms=5)
    assert abs(r["estimate"] - (0.5 - 1 / (1 + math.exp(-2.0)))) < 1e-12


def test_permutation_flips_whole_games():
    items = [_item(id=f"g{g}k{k}", game_id=f"valid_unseen/{g}") for g in range(4) for k in range(30)]
    by = {}
    for t in items:
        by[(t["id"], "desire_keep")] = {"L": 1.0}
        by[(t["id"], "swapped")] = {"L": 0.0}
    r = mech2.contrast(by, items, ((1, "desire_keep"), (-1, "swapped")), B=50, perms=400)
    assert r["p"] > 0.1  # four games: the smallest two-sided game-level p is 2/16; item flips would give about 0


def test_ratio_needs_a_positive_reference_and_few_dropped_draws():
    assert mech2._ratio([(1.0, -1.0), (1.0, 0.5)]) != mech2._ratio([(1.0, -1.0), (1.0, 0.5)])  # mean -0.25: NaN
    assert mech2._ratio([(1.0, 0.0)]) != mech2._ratio([(1.0, 0.0)])
    assert mech2._ratio([(1.0, 2.0)]) == 0.5
    rng = random.Random(5)
    items, by = [], {}
    for g in range(20):  # a reference effect near zero: many resamples have a mean at or below zero
        for k in range(5):
            t = _item(id=f"g{g}k{k}", game_id=f"valid_unseen/{g}")
            items.append(t)
            by[(t["id"], "aligned")] = {"L": (3.0 if g % 2 else -2.8) + rng.gauss(0, 0.1)}
            by[(t["id"], "swapped")] = {"L": 0.0}
            by[(t["id"], "desire_keep")] = {"L": 0.1}
    r = mech2.contrast(by, items, ((1, "desire_keep"), (-1, "swapped")), reference=REF, B=200, perms=50)
    assert r["reference_effect"] > 0 and r["ratio"]["n_dropped"] > 0.02 * 200 and not r["ratio"]["estimable"]
    assert mech2.categorize(r, 0.001) == ("INCONCLUSIVE", None)


def _res(est, lo, hi, estimable=True):
    return {"estimate": est, "ratio": {"ci": [lo, hi], "estimable": estimable}}


def test_categories_follow_the_zones():
    assert mech2.categorize(_res(-1, -0.9, -0.3), 0.01) == ("NEG", ("N",))
    assert mech2.categorize(_res(-1, -0.4, -0.1), 0.01) == ("NEG_OR_TRIVIAL", ("N", "Z"))
    assert mech2.categorize(_res(1, 0.6, 1.1), 0.01) == ("POS_LARGE", ("L",))
    assert mech2.categorize(_res(1, 0.3, 0.45), 0.01) == ("POS_SMALL", ("S",))
    assert mech2.categorize(_res(1, 0.3, 0.8), 0.01) == ("POS", ("S", "L"))
    assert mech2.categorize(_res(1, 0.1, 0.4), 0.01) == ("TRIVIAL_OR_SMALL", ("Z", "S"))
    assert mech2.categorize(_res(0.1, -0.2, 0.2), 0.3) == ("NULL", ("Z",))
    assert mech2.categorize(_res(0.1, 0.05, 0.2), 0.001) == ("TRIVIAL", ("Z",))  # significant, inside the band: ≈0
    assert mech2.categorize(_res(0.1, -0.3, 0.6), 0.3) == ("WIDE", ("N", "Z", "S", "L"))
    assert mech2.categorize(_res(1, 0.3, 0.8), 0.06) == ("INCONCLUSIVE", None)  # non-trivial size, not significant
    assert mech2.categorize(_res(float("nan"), 0, 0), 0.01) == ("INCONCLUSIVE", None)
    assert mech2.categorize(_res(1, 0.6, 1.1, estimable=False), 0.01) == ("INCONCLUSIVE", None)
    assert mech2.zones_of(-0.25, 0.25) == ("Z",) and mech2.zones_of(0.25001, 0.5) == ("S", "L")
    assert mech2.zone(-0.2501) == "N" and mech2.zone(0.4999) == "S"


def test_a_trivial_effect_never_contradicts_a_null_prediction():
    zs = mech2.categorize(_res(0.15, 0.08, 0.22), 1e-6)[1]
    for h, preds in mech2.PREDICTIONS.items():
        for c, allowed in preds.items():
            if "Z" in allowed:
                assert mech2.reading(allowed, zs) == "consistent", (h, c)
    # straddling the band's edge leaves a ≈0 prediction undecided, never contradicted
    assert mech2.reading(frozenset({"Z"}), ("Z", "S")) == "undecided"
    assert mech2.reading(frozenset({"N"}), ("Z", "S")) == "contradicted"


def test_holm_is_applied_within_the_family():
    items = _syn_items(n_games=12, per_game=6)
    fam = _family(_syn_scores(items, FINGERPRINTS["H4_wishful"], noise=3.0), items, B=50)
    ps = {k: v["p"] for k, v in fam["contrasts"].items()}
    for k, v in fam["contrasts"].items():
        assert v["holm_p"] >= v["p"]
    assert sorted(fam["contrasts"]) == sorted(mech2.PRIMARY) and len(mech2.PRIMARY) == 7
    smallest = min(ps, key=ps.get)
    assert abs(fam["contrasts"][smallest]["holm_p"] - min(1.0, 7 * ps[smallest])) < 1e-12


def test_readout_verdicts_and_the_joint_rule():
    none = {c: None for c in mech2.PRIMARY}
    ro = mech2.readout(none)
    assert all(v["n_undecided"] == len(v["per_contrast"]) and v["verdict"] == "untested" for v in ro.values())
    one = {**none, "P2_desire_check": ("S", "L")}
    ro = mech2.readout(one)
    assert ro["H5_policy_blend"]["per_contrast"]["P2_desire_check"] == "consistent"
    assert ro["H5_policy_blend"]["verdict"] == "consistent"
    assert ro["H4_wishful"]["per_contrast"]["P2_desire_check"] == "contradicted" and ro["H4_wishful"]["verdict"] == "contradicted"
    assert ro["H6_position"]["verdict"] == "untested"
    # the two formats must agree before a contrast is decided
    yn = mech2.readout({**none, "P2_desire_check": ("N",)})
    j = mech2.joint_readout(ro, yn)
    assert j["H4_wishful"]["per_contrast"]["P2_desire_check"] == "format_dependent"
    assert j["H4_wishful"]["verdict"] == "untested" and j["H4_wishful"]["format_dependent"] == ["P2_desire_check"]
    j2 = mech2.joint_readout(ro, mech2.readout(none))
    assert j2["H5_policy_blend"]["per_contrast"]["P2_desire_check"] == "undecided" and j2["H5_policy_blend"]["verdict"] == "untested"
    j3 = mech2.joint_readout(ro, ro)
    assert j3["H4_wishful"]["verdict"] == "contradicted" and j3["H5_policy_blend"]["verdict"] == "consistent"


def test_joint_summary_names_a_mixture_when_every_content_account_is_contradicted():
    # P1 trivial (against H4), P3 below zero (against H2, H3, H6_lexical), P2 below zero (against H5)
    obs = {c: None for c in mech2.PRIMARY}
    obs.update({"P1_keep_vs_ignore": ("Z",), "P3_mention_without_need": ("N",), "P2_desire_check": ("N",)})
    ro = mech2.readout(obs)
    j = mech2.joint_readout(ro, ro)
    assert j["summary"]["content_accounts_not_contradicted"] == []
    assert j["summary"]["reading"].startswith("no single account")


def test_planted_fingerprints_survive_the_joint_rule():
    items = _syn_items()
    by = mech2._lookup(_syn_scores(items, FINGERPRINTS["H2_evidence"]))
    pops = _pops(items)
    c, y = (mech2.primary_family(by, pops, f, B=200, perms=200) for f in ("cand", "yn"))
    j = mech2.joint_readout(c["readout"], y["readout"])
    assert j["summary"]["content_accounts_not_contradicted"] == ["H2_evidence"]
    assert j["summary"]["content_accounts_consistent"] == ["H2_evidence"]


def test_alias_reads_the_plain_score():
    t = _item(step0=True)
    by = {(t["id"], "aligned"): {"L": 2.0}, (t["id"], "yn_swapped"): {"L": -1.0}}
    assert mech2._value(by, t, "aligned_end", "cand") == 2.0 and mech2._value(by, t, "swapped_end", "yn") == -1.0
    assert mech2._value(by, _item(), "aligned_end", "cand") is None


def test_later_holds_no_step0_item():
    items = _syn_items()
    pops = _pops(items)
    assert pops["later"] and all(t["history_len"] > 0 for t in pops["later"])
    assert len(pops["later"]) == sum(t["history_len"] > 0 for t in items)


def test_item_share_counts_ties_against():
    items = [_item(id=f"i{k}", game_id=f"valid_unseen/{k % 3}") for k in range(6)]
    by = {}
    for k, t in enumerate(items):
        by[(t["id"], "yn_desire_keep")] = {"L": 1.0 if k < 2 else 0.0}
        by[(t["id"], "yn_swapped")] = {"L": 0.0}
    r = mech2.item_share(by, items, "desire_keep", "swapped", B=50, perms=50)
    assert abs(r["estimate"] - 2 / 6) < 1e-12 and r["n_items"] == 6


def test_fewshot_overlap_finds_a_planted_overlap_boost():
    line = WM_FEWSHOT_SETS[1][0]["history"].split("\n", 1)[0]
    items = [_item(id=f"o{k}", game_id=f"valid_unseen/{k % 4}", task_line=line if k < 20 else "Your task is to: put a clean apple in fridge.")
             for k in range(80)]
    rng = random.Random(3)
    scores = []
    for t in items:
        for k, (a, s) in {1: ("aligned", "swapped"), 2: ("aligned_set2", "swapped_set2"), 3: ("aligned_set3", "swapped_set3")}.items():
            boost = 2.0 if (k == 1 and t["task_line"] == line) else 0.0
            scores += [{"transition_id": t["id"], "condition": a, "L": 3 + boost + rng.gauss(0, 0.1)},
                       {"transition_id": t["id"], "condition": s, "L": rng.gauss(0, 0.1)}]
    out = mech2.fewshot_overlap(scores, items, B=50)
    assert abs(out["set1"]["aligned_overlap"]["difference_vs_none"] - 2.0) < 0.2
    assert out["set1"]["aligned_overlap"]["n_items"] == 20 and out["set2"]["aligned_overlap"]["n_items"] == 0
    assert abs(out["set1"]["aligned_overlap"]["mean_delta_L"] - 5.0) < 0.2 and abs(out["set1"]["mean_delta_L_none"] - 3.0) < 0.2


def test_knowledge_split_reads_mech_state_questions():
    items = _syn_items(n_games=10, per_game=6)
    scores, mech_scores = [], []
    for k, t in enumerate(items):
        known = k % 2 == 0
        for arm, L in (("aligned", 0.0 if known else 4.0), ("swapped", 0.0)):
            scores.append({"transition_id": t["id"], "condition": arm, "L": L})
        mech_scores += [{"transition_id": t["id"], "condition": "state_absent", "L": -1.0 if known else 1.0},
                        {"transition_id": t["id"], "condition": "statetwin_absent", "L": 1.0}]
    out = mech2.knowledge_split(scores, mech_scores, items, B=50)
    assert out["aligned_effect"]["n_known"] == 30 and abs(out["aligned_effect"]["known_share_of_overall"]) < 1e-12


# ---------------- tokenization gate ----------------


class FakeTok:
    """Whitespace tokens; a template of role markers. Variants put a space token before a reply or split "Yes"."""
    SPECIAL = {"<s>": 0, "<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|end|>": 4}

    def __init__(self, space_before_reply=False, split_yes=False, space_before_answer=False):
        self.vocab = dict(self.SPECIAL)
        self.space_before_reply, self.split_yes = space_before_reply, split_yes
        self.space_before_answer = space_before_answer  # a space token before "Yes" or "No" replies only
        self.all_special_ids = list(self.SPECIAL.values())

    def _tok(self, text):
        out = []
        for w in re.findall(r"\S+|\n", text):
            parts = ["Ye", "s"] if (self.split_yes and w == "Yes") else [w]
            out += [self.vocab.setdefault(p, len(self.vocab)) for p in parts]
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        class R:
            pass
        r = R()
        r.input_ids = [self._tok(text)] if return_tensors else self._tok(text)
        return r

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors=None, **kw):
        ids = [0]
        for i, m in enumerate(messages):
            ids.append(self.SPECIAL[f"<|{m['role']}|>"])
            last_reply = m["role"] == "assistant" and i == len(messages) - 1 and not add_generation_prompt
            if last_reply and (self.space_before_reply or (self.space_before_answer and m["content"] in ("Yes", "No"))):
                ids.append(self.vocab.setdefault("▁", len(self.vocab)))
            ids += self._tok(m["content"]) + [4]
        if add_generation_prompt:
            ids.append(3)
        return [ids] if return_tensors else ids

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


class FakeScorer:
    # the real encoding and template settings, so the gate is checked against what scoring renders
    _encode = LogoddsScorer._encode
    _encode_candidate = LogoddsScorer._encode_candidate
    template_settings = LogoddsScorer.template_settings
    template_kwargs: dict = {}

    def __init__(self, tok, revision=C.REVISIONS["Qwen/Qwen3-8B"]):
        self._tok, self.revision, self.mock, self.model_tag = tok, revision, False, "qwen3-8b-fewshot"
        self.scored = 0

    def score_pair(self, **kw):
        self.scored += 1
        raise AssertionError("scored before the gate")

    score_one_token_pair = score_pair

    def close(self):
        pass


def _todo():
    return mech2.jobs([_item(id="a", step0=True), _item(id="b")])


def test_gate_passes_a_clean_template(mock_score, tmp_path):
    rec = mech2.tokenization_checks(FakeScorer(FakeTok()), _todo())
    assert rec["ok"] and rec["n_candidates_checked"] == 4 and rec["n_yes_no_prompts_checked"] == len(_todo()) // 2
    mech2.gate(tmp_path, FakeScorer(FakeTok()), _todo(), "Qwen/Qwen3-8B")
    assert json.loads((tmp_path / "tokenization_checks_qwen3-8b-fewshot.json").read_text())["ok"] is True


def test_gate_stops_on_a_space_before_the_reply_or_a_split_word(mock_score, tmp_path):
    for tok in (FakeTok(space_before_reply=True), FakeTok(split_yes=True)):
        assert not mech2.tokenization_checks(FakeScorer(tok), _todo())["ok"]
        with pytest.raises(mech2.TokenizationMismatch):
            mech2.gate(tmp_path, FakeScorer(tok), _todo(), "Qwen/Qwen3-8B")


def test_gate_checks_every_yes_no_prompt_it_is_given(mock_score, tmp_path):
    """On a resume that left only yes/no prompts to score, the gate still reads Yes and No after each of them."""
    yn_only = [(t, a, f) for t, a, f in _todo() if f == "yn"]
    rec = mech2.tokenization_checks(FakeScorer(FakeTok()), yn_only)
    assert rec["ok"] and rec["n_yes_no_prompts_checked"] == len(yn_only) == rec["n_yes_no_jobs"]
    bad = mech2.tokenization_checks(FakeScorer(FakeTok(space_before_answer=True)), yn_only)
    assert not bad["ok"] and bad["n_problems"] >= 2 * len(yn_only)
    assert mech2.tokenization_checks(FakeScorer(FakeTok(space_before_answer=True)),
                                     [(t, a, f) for t, a, f in _todo() if f == "cand"])["ok"]  # no yes/no prompt to score


def test_gate_refuses_an_unpinned_revision(mock_score, tmp_path):
    with pytest.raises(SystemExit):
        mech2.gate(tmp_path, FakeScorer(FakeTok(), revision=None), _todo(), "Qwen/Qwen3-8B")
    with pytest.raises(SystemExit):
        mech2.gate(tmp_path, FakeScorer(FakeTok(), revision="main"), _todo(), "Qwen/Qwen3-8B")


# ---------------- runner, end to end on mock data ----------------


def _cfg(**kw):
    return {"run_id": "mech2_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 2, "sample_n": 30, **kw}


def test_end_to_end_two_shards_then_analysis(mock_root):
    for i in range(2):
        assert mech2.run_mech2_score(_cfg(shard=i)) == 0
    recs = [r for p in sorted((mock_root / "mech2_smoke").glob("shard_*")) for r in read_jsonl(p / "scoring.jsonl")]
    primary = rq1.rq1_sets(mech2.load_items(("rq12_transitions_smoke",)))["primary"]
    expected = {(t["id"], mech2.condition(a, f)) for t, a, f in mech2.jobs(mech2.battery_items(primary, 30)["union"])}
    for tag in rq1.FEWSHOT:
        got = [(r["transition_id"], r["condition"]) for r in recs if r["model"] == tag]
        assert sorted(got) == sorted(expected)  # every job once, nothing else
    assert all(r["model_revision"] == C.REVISIONS[{"qwen3-8b-fewshot": "Qwen/Qwen3-8B",
                                                   "llama3.1-8b-fewshot": "meta-llama/Llama-3.1-8B-Instruct"}[r["model"]]]
               for r in recs)
    assert mech2.run_mech2_analysis({"run_id": "mech2_smoke_analysis", "source": "mech2_smoke", "sample_n": 30,
                                     "sources": ["rq12_transitions_smoke"], "bootstrap": 50, "permutations": 50}) == 0
    m = json.loads((mock_root / "mech2_smoke_analysis" / "metrics.json").read_text())
    for tag in rq1.FEWSHOT:
        assert set(m[tag]["primary"]["contrasts"]) == set(mech2.PRIMARY)
        assert set(m[tag]["primary"]["readout"]) == set(mech2.PREDICTIONS)
        assert "position" in m[tag]["secondary"] and "yn_item_share_toward_yes_vs_swapped" in m[tag]["arms_on_sample"]["done"]
        assert set(m[tag]["readout_both_formats"]) == set(mech2.PREDICTIONS) | {"summary"}
    assert m["items"]["n_union"] == len(mech2.battery_items(primary, 30)["union"])


def test_resume_scores_nothing_twice(mock_root):
    assert mech2.run_mech2_score(_cfg(n_shards=1)) == 0
    path = mock_root / "mech2_smoke" / "shard_00_of_01" / "scoring.jsonl"
    before = path.read_text()
    assert mech2.run_mech2_score(_cfg(n_shards=1, resume=True)) == 0
    assert path.read_text() == before


def test_a_failed_gate_stops_the_run_before_any_score(mock_root):
    with pytest.raises(mech2.TokenizationMismatch):
        mech2.run_mech2_score(_cfg(n_shards=1), open_scorer=lambda name, tag: FakeScorer(FakeTok(split_yes=True)))
    folder = mock_root / "mech2_smoke" / "shard_00_of_01"
    assert not (folder / "scoring.jsonl").exists() or not read_jsonl(folder / "scoring.jsonl")
    assert "stopped before scoring" in (folder / "STATUS.md").read_text()


def test_p5_reads_placement_tasks_only():
    items = _syn_items()
    scores = _syn_scores(items, FINGERPRINTS["H4_wishful"])
    examine = {t["id"] for t in items if mech2.task_type(t["task_line"]) == "examine"}
    assert examine
    for s in scores:  # done far below cancelled on examine items only
        if s["transition_id"] in examine and s["condition"] == "done":
            s["L"] -= 2 * D
    fam = _family(scores, items)
    assert fam["contrasts"]["P5_done_vs_cancelled"]["category"] == "NULL"
    assert fam["contrasts"]["P5_done_vs_cancelled"]["n_items"] == len(items) - len(examine)


def test_job_script_runs_only_the_battery_runs():
    import wf.runs as runs_mod

    root = Path(__file__).resolve().parents[1]
    script = (root / "pace" / "mech2_score.sbatch").read_text()
    assert "#SBATCH --no-requeue" in script and "mech2_score|mech2_score_smoke)" in script
    for run in ("mech2_score", "mech2_score_smoke", "mech2_analysis"):
        assert run in runs_mod.RUNNERS and (root / "wf" / "config" / f"{run}.yaml").exists()


# ---------------- amendment after review 1: items, models, resume ----------------


def test_largest_remainders_get_the_extra_seats():
    sizes = {"a": 5, "b": 4, "c": 1}  # n = 3 of 10: exact 1.5, 1.2, 0.3; floors 1, 1, 0; the one seat left goes to a
    p = [_item(id=f"{g}{k}", game_id=f"valid_unseen/{g}") for g, n in sizes.items() for k in range(n)]
    s = mech2.stratified_sample(p, 3)
    got = {g: sum(t["id"] in s for t in p if t["game_id"] == f"valid_unseen/{g}") for g in sizes}
    assert got == {"a": 2, "b": 1, "c": 0}


def test_p5_leaves_out_takes_from_a_target_type_receptacle_and_contradicted_items():
    away = _item(TASKS["pick"][0])  # take apple 1 from countertop 2, target fridge
    assert mech2.p5_eligible(away) and not mech2.takes_from_target_type(away)
    from_target = _item(TASKS["pick"][0], action="take apple 1 from fridge 1")
    assert mech2.takes_from_target_type(from_target) and not mech2.p5_eligible(from_target)
    contradicted = _item(TASKS["pick"][0], history_text=f"{OPENING}\nYour task is to: put an apple in fridge.\n> open fridge 1\n"
                         "You open the fridge 1. The fridge 1 is open. In it, you see a egg 1.")
    assert mech2.done_contradicted(contradicted) and not mech2.p5_eligible(contradicted)
    assert not mech2.p5_eligible(_item(TASKS["examine"][0]))
    assert not mech2.takes_from_target_type(_item(TASKS["pick"][0], action="move apple 1 to fridge 1"))
    pops = _pops([away, {**from_target, "id": "x"}, {**contradicted, "id": "y"}])
    assert [t["id"] for t in pops["p5"]] == ["m1"]


def test_negated_claim_status_and_p7_items():
    t = _item(scene_locations={"apple 1": ["countertop 2"]})
    assert mech2.negated_status(t) == ("false at start", "contents not shown") and mech2.p7_eligible(t)
    seen = _item(history_text=f"{OPENING}\nYour task is to: put a clean apple in fridge.\n> go to countertop 2\n"
                 "You arrive at countertop 2. On the countertop 2, you see a apple 1.", scene_locations={"apple 1": ["countertop 2"]})
    assert mech2.negated_status(seen) == ("false at start", "shown, holds it") and not mech2.p7_eligible(seen)
    empty = {**seen, "history_text": seen["history_text"].replace("a apple 1", "a egg 1"), "scene_locations": {}}
    assert mech2.negated_status(empty) == ("true at start", "shown, without it") and mech2.p7_eligible(empty)
    assert not mech2.p7_eligible(_item(action=ACTIONS["move"]))
    pops = _pops([t, {**seen, "id": "s"}, {**_item(action=ACTIONS["heat"]), "id": "h"}])
    assert [x["id"] for x in pops["sample_take"]] == ["m1"]


def test_mistral_is_scored_with_the_families_handling():
    name, kw, prompt = mech2.model_entry("mistral-7b-v0.3")
    spec = families.MODELS["mistral-7b-v0.3"]
    assert name == spec.hf_id == "mistralai/Mistral-7B-Instruct-v0.3" and C.REVISIONS[name] == spec.revision
    assert kw == {"attn_implementation": spec.attn_implementation, "template_kwargs": spec.template_kwargs}
    t = _item()
    assert spec.system == "native" and mech2.model_messages("mistral-7b-v0.3", t, "desire_keep") == mech2.arm_messages(t, "desire_keep")
    assert mech2.model_entry("qwen3-8b-fewshot")[0] == "Qwen/Qwen3-8B" and mech2.model_entry("qwen3-8b-fewshot")[1] == {}
    with pytest.raises(SystemExit):
        mech2.model_entry("gemma2-9b-it")
    assert "mistral-7b-v0.3" in mech2.SECONDARY_MODELS and "mistral-7b-v0.3" not in mech2.PRIMARY_MODELS


def test_a_folding_model_gets_the_system_text_in_its_first_turn(monkeypatch):
    spec = families.MODELS["mistral-7b-v0.3"]
    monkeypatch.setitem(families.MODELS, "mistral-7b-v0.3", families.replace(spec, system="fold"))
    t = _item()
    msgs = mech2.model_messages("mistral-7b-v0.3", t, "aligned")
    assert msgs == families.fold_system(mech2.arm_messages(t, "aligned")) and msgs[0]["role"] == "user"


def test_resume_refuses_records_of_other_items(mock_root):
    assert mech2.run_mech2_score(_cfg(n_shards=1)) == 0
    path = mock_root / "mech2_smoke" / "shard_00_of_01" / "scoring.jsonl"
    recs = read_jsonl(path)
    recs[0]["item_hash"] = "not this item"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit):
        mech2.run_mech2_score(_cfg(n_shards=1, resume=True))


def test_the_secondary_model_is_scored_and_reported_descriptively(mock_root):
    models = ["qwen3-8b-fewshot", "mistral-7b-v0.3"]
    assert mech2.run_mech2_score(_cfg(n_shards=1, models=models)) == 0
    recs = read_jsonl(mock_root / "mech2_smoke" / "shard_00_of_01" / "scoring.jsonl")
    assert {r["model"] for r in recs} == set(models)
    cfg = json.loads((mock_root / "mech2_smoke" / "shard_00_of_01" / "config.json").read_text())
    assert cfg["revisions"]["mistralai/Mistral-7B-Instruct-v0.3"] == C.REVISIONS["mistralai/Mistral-7B-Instruct-v0.3"]
    assert mech2.run_mech2_analysis({"run_id": "mech2_smoke_analysis", "source": "mech2_smoke", "sample_n": 30, "models": models,
                                     "sources": ["rq12_transitions_smoke"], "bootstrap": 30, "permutations": 30}) == 0
    m = json.loads((mock_root / "mech2_smoke_analysis" / "metrics.json").read_text())
    assert m["qwen3-8b-fewshot"]["role"] == "primary" and m["mistral-7b-v0.3"]["role"].startswith("secondary")
    assert "readout_both_formats" in m["mistral-7b-v0.3"]


def test_the_runner_scores_through_the_model_template_handling(mock_root, monkeypatch):
    """Every prompt the runner scores goes through the model's handling (here a folding model): the system text opens
    the first user turn in every scored prompt."""
    spec = families.MODELS["mistral-7b-v0.3"]
    monkeypatch.setitem(families.MODELS, "mistral-7b-v0.3", families.replace(spec, system="fold"))
    seen = []

    def opener(name, tag):
        real = LogoddsScorer(name, model_tag=tag)  # mock scorer (WF_MOCK_SCORE=1 in mock_root)

        class Recording:
            mock, model_tag, revision = True, tag, real.revision

            def score_pair(self, **kw):
                seen.append(kw["messages"])
                return real.score_pair(**kw)

            def score_one_token_pair(self, **kw):
                seen.append(kw["messages"])
                return real.score_one_token_pair(**kw)

            def close(self):
                pass
        return Recording()

    assert mech2.run_mech2_score(_cfg(n_shards=1, models=["mistral-7b-v0.3"]), open_scorer=opener) == 0
    assert seen and all(m[0]["role"] == "user" and all(x["role"] != "system" for x in m) for m in seen)
