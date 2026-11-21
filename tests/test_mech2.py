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
    """The arms an item carries: every arm, except negated and affirmed on items whose action is not a take, and the
    position arms after step 0 (amendment after review 2)."""
    return [a for a in mech2.ARMS if (a not in ("negated", "affirmed") or t["action"].startswith("take "))
            and (a not in mech2.POSITION_ARMS or t["history_len"] == 0)]


def test_every_arm_builds_for_every_task_type_and_verb():
    for ttype, verb, t in _all_variants():
        assert (mech2.task_line(t, "negated") is None) == (verb != "take")
        assert (mech2.task_line(t, "affirmed") is None) == (verb != "take")
        for arm in mech2.POSITION_ARMS:
            assert (mech2.task_line(t, arm) is None) == (t["history_len"] > 0)
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
    # amendment after review 2: the minimal-pair partners
    assert mech2.task_line(t, "affirmed") == ("Your task is to: put a clean egg in fridge. The user says there is an apple "
                                              "in the countertop 2.")
    assert mech2.task_line(t, "desire_keep_sw") == ("Your task is to: put a clean egg in fridge. Make sure that the apple is "
                                                    "not taken; leave it where it is.")
    assert mech2.task_line(t, "desire_check_can") == ("Your task is to: find out whether an apple can be taken from the "
                                                      "countertop 2; the user hopes it can.")
    for verb, pp, prep, rec in (("move", "moved", "to", "fridge 1"), ("heat", "heated", "with", "microwave 1"),
                                ("cool", "cooled", "with", "fridge 1"), ("clean", "cleaned", "with", "sinkbasin 1")):
        u = _item(action=ACTIONS[verb])
        assert mech2.task_line(u, "desire_keep") == f"Your task is to: make sure that the apple is not {pp}; leave it where it is."
        assert mech2.task_line(u, "desire_check").endswith(f"an apple can be {pp} {prep} the {rec}; the user hopes it cannot.")
    m = _item(goal_object="mug", task_line="Your task is to: put a mug in fridge.", action="take mug 1 from countertop 2",
              swapped_task_line="Your task is to: put a egg in fridge.")
    assert "whether a mug can be taken" in mech2.task_line(m, "desire_check")
    assert "whether a mug can be taken" in mech2.task_line(m, "desire_check_can")
    assert mech2.task_line(m, "affirmed").endswith("The user says there is a mug in the countertop 2.")
    for verb, pp in (("move", "moved"), ("heat", "heated"), ("clean", "cleaned")):
        assert mech2.task_line(_item(action=ACTIONS[verb]), "desire_keep_sw").endswith(
            f"Make sure that the apple is not {pp}; leave it where it is.")


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
        assert "apple" not in mech2.task_line(t, "described")
        assert rq1x.DESCRIPTIONS["apple"] in mech2.task_line(t, "described")
    assert "swapped_described" not in mech2.ARMS  # amendment after review 2: rq1x measured it


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
    real = {"aligned", "done", "cancelled", "aligned_start"}
    for _, _, t in _all_variants():
        for arm in _arms(t):
            shown = t["task_line"][len("Your task is to: "):] in _query(mech2.arm_messages(t, arm))
            assert shown == (arm in real), arm


def test_position_arms_move_only_the_task_line(mock_score):
    for _, _, t in _all_variants():
        if t["history_len"]:
            continue  # amendment after review 2: position arms at step 0 only
        for arm in mech2.POSITION_ARMS:
            goal, where = arm.rsplit("_", 1)
            plain, moved = _query(mech2.arm_messages(t, goal)), _query(mech2.arm_messages(t, arm))
            line = mech2.task_line(t, goal)
            assert sorted(plain.split("\n")) == sorted(moved.split("\n"))
            assert [x for x in plain.split("\n") if x != line] == [x for x in moved.split("\n") if x != line]
            hist = moved.split("History:\n", 1)[1].split("\n\nAction:", 1)[0].split("\n")
            assert (hist[0] if where == "start" else hist[-1]) == line
            assert mech2.arm_messages(t, arm)[:-1] == mech2.arm_messages(t, goal)[:-1]


def test_usual_position_and_position_arms_at_step0_only():
    s0, s1 = _item(step0=True), _item()
    assert mech2.usual_position(s0) == "end" and mech2.usual_position(s1) == "line 2"
    assert mech2.POSITION_ARMS == ("aligned_start", "swapped_start")
    assert mech2.history_for(s0, "aligned_start") != s0["history_text"]  # step 0: the line moves from last to first
    assert mech2.task_line(s0, "aligned_start") == s0["task_line"] and mech2.task_line(s1, "aligned_start") is None
    assert mech2.arm_messages(s1, "swapped_start") is None
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


def test_jobs_cover_every_arm_position_arms_at_step0_only():
    s0, s1 = _item(id="a", step0=True), _item(id="b")
    j = {(t["id"], a, f) for t, a, f in mech2.jobs([s0, s1])}
    for arm in mech2.ARMS:
        for f in mech2.FORMATS:
            assert ("a", arm, f) in j
            assert (("b", arm, f) in j) == (arm not in mech2.POSITION_ARMS)
    assert len(j) == 2 * (2 * len(mech2.ARMS) - len(mech2.POSITION_ARMS))


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
# Amended after review 2: P1 = desire_keep_sw - mention_ignore, P2 = desire_check - desire_check_can,
# P8 = negated - affirmed.
FINGERPRINTS = {
    "H2_evidence": {"desire_keep": 0.8, "desire_keep_sw": 0.8, "desire_check": 0.6, "desire_check_can": 0.6,
                    "mention_ignore": 0.7, "described": 0.9, "done": -0.3, "cancelled": 0.5, "negated": -0.6, "affirmed": 0.8},
    "H3_salience": {"desire_keep": 0.8, "desire_keep_sw": 0.8, "desire_check": 0.8, "desire_check_can": 0.8,
                    "mention_ignore": 0.8, "described": 0.6, "done": 0.8, "cancelled": 0.8, "negated": 0.7, "affirmed": 0.7},
    "H4_wishful": {"desire_keep": -0.5, "desire_keep_sw": -0.5, "desire_check": -0.5, "desire_check_can": 0.4,
                   "mention_ignore": 0.0, "described": 0.9, "done": 0.0, "cancelled": 0.0, "negated": 0.0, "affirmed": 0.0},
    "H5_policy_blend": {"desire_keep": -0.5, "desire_keep_sw": -0.5, "desire_check": 0.9, "desire_check_can": 0.9,
                        "mention_ignore": 0.0, "described": 0.9, "done": 0.0, "cancelled": 0.0, "negated": 0.0, "affirmed": 0.0},
    "H6_lexical": {"desire_keep": 0.8, "desire_keep_sw": 0.8, "desire_check": 0.8, "desire_check_can": 0.8,
                   "mention_ignore": 0.8, "described": 0.0, "done": 0.8, "cancelled": 0.8, "negated": 0.8, "affirmed": 0.8},
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
            mu = {"aligned": d, "swapped": 0.0, "neutral": -0.4 * d, "mention_exists": 0.9 * d,
                  **{a: v * d for a, v in fp.items()}}
            mu["aligned_start"] = mu["aligned"] + position_shift * d
            mu["swapped_start"] = mu["swapped"]
            for arm, m in mu.items():
                if mech2.task_line(t, arm) is None:
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
    assert ro[planted]["n_contradicted"] == 0 and ro[planted]["n_consistent"] == 7, (planted, fam["contrasts"])
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
    assert abs(c["P2_hope_polarity"]["ratio"]["estimate"] + 0.9) < 1e-9
    assert abs(y["P1_keep_vs_ignore"]["ratio"]["estimate"]) < 1e-9 and abs(y["P2_hope_polarity"]["ratio"]["estimate"]) < 1e-9
    assert abs(y["P3_mention_without_need"]["ratio"]["estimate"] - 0.8) < 1e-9
    assert abs(y["P3_mention_without_need"]["estimate"] - 0.8 * 0.25 * D) < 1e-9
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
    assert sorted(fam["contrasts"]) == sorted(mech2.PRIMARY) and len(mech2.PRIMARY) == 8
    smallest = min(ps, key=ps.get)
    assert abs(fam["contrasts"][smallest]["holm_p"] - min(1.0, 8 * ps[smallest])) < 1e-12


def test_readout_verdicts_and_the_joint_rule():
    none = {c: None for c in mech2.PRIMARY}
    ro = mech2.readout(none)
    assert all(ro[h]["n_undecided"] == len(ro[h]["per_contrast"]) and ro[h]["verdict"] == "untested" for h in mech2.PREDICTIONS)
    one = {**none, "P2_hope_polarity": ("N",)}
    ro = mech2.readout(one)
    assert ro["H4_wishful"]["per_contrast"]["P2_hope_polarity"] == "consistent"
    assert ro["H4_wishful"]["verdict"] == "consistent" and ro["H4_wishful"]["consistent_where_a_rival_fails"] == ["P2_hope_polarity"]
    for h in ("H2_evidence", "H3_salience", "H5_policy_blend", "H6_lexical"):
        assert ro[h]["per_contrast"]["P2_hope_polarity"] == "contradicted" and ro[h]["verdict"] == "contradicted"
    assert ro["H6_position"]["verdict"] == "untested"
    # the two formats must agree before a contrast is decided
    yn = mech2.readout({**none, "P2_hope_polarity": ("Z",)})
    j = mech2.joint_readout(ro, yn)
    assert j["H4_wishful"]["per_contrast"]["P2_hope_polarity"] == "format_dependent"
    assert j["H4_wishful"]["verdict"] == "untested" and j["H4_wishful"]["format_dependent"] == ["P2_hope_polarity"]
    j2 = mech2.joint_readout(ro, mech2.readout(none))
    assert j2["H4_wishful"]["per_contrast"]["P2_hope_polarity"] == "undecided" and j2["H4_wishful"]["verdict"] == "untested"
    j3 = mech2.joint_readout(ro, ro)
    assert j3["H4_wishful"]["verdict"] == "consistent" and j3["H5_policy_blend"]["verdict"] == "contradicted"
    assert j3["summary"]["content_accounts_consistent"] == ["H4_wishful"]
    assert j3["summary"]["content_accounts_untested"] == []
    assert set(j3["summary"]["content_accounts_contradicted"]) == set(mech2.CONTENT_ACCOUNTS) - {"H4_wishful"}
    assert j3["summary"]["reading"].startswith("consistent: H4_wishful; untested: none; contradicted (cannot be the sole account): ")


def test_joint_summary_names_a_mixture_when_every_content_account_is_contradicted():
    # P1 trivial (against H4), P3 below zero (against H2, H3, H6_lexical), P2 below zero (against H5)
    obs = {c: None for c in mech2.PRIMARY}
    obs.update({"P1_keep_vs_ignore": ("Z",), "P3_mention_without_need": ("N",), "P2_hope_polarity": ("N",)})
    ro = mech2.readout(obs)
    j = mech2.joint_readout(ro, ro)
    assert j["summary"]["content_accounts_not_contradicted"] == []
    assert j["summary"]["reading"].startswith("no single account")
    assert "P5, P7 and P8" in j["summary"]["reading"] and "sole account" in j["summary"]["reading"]
    assert "reject" not in j["summary"]["reading"]


def test_planted_fingerprints_survive_the_joint_rule():
    items = _syn_items()
    by = mech2._lookup(_syn_scores(items, FINGERPRINTS["H2_evidence"]))
    pops = _pops(items)
    c, y = (mech2.primary_family(by, pops, f, B=200, perms=200) for f in ("cand", "yn"))
    j = mech2.joint_readout(c["readout"], y["readout"])
    assert j["summary"]["content_accounts_not_contradicted"] == ["H2_evidence"]
    assert j["summary"]["content_accounts_consistent"] == ["H2_evidence"]


def test_a_missing_score_reads_as_none():
    t = _item(step0=True)
    by = {(t["id"], "aligned"): {"L": 2.0}, (t["id"], "yn_swapped"): {"L": -1.0}}
    assert mech2._value(by, t, "aligned", "cand") == 2.0 and mech2._value(by, t, "swapped", "yn") == -1.0
    assert mech2._value(by, t, "aligned_start", "cand") is None and mech2._value(by, t, "swapped", "cand") is None


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
        assert set(m[tag]["primary"]["readout"]) == set(mech2.PREDICTIONS) | {"rivals"}
        assert "position" in m[tag]["secondary"] and "yn_item_share_toward_yes_vs_swapped" in m[tag]["arms_on_sample"]["done"]
        assert set(m[tag]["readout_both_formats"]) == set(mech2.PREDICTIONS) | {"summary", "rivals"}
        assert m["integrity"][tag]["status"] == "complete" and m["integrity"][tag]["n_hash_mismatch"] == 0
        assert set(m[tag]["mixture_readout"]) == {"P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed"}
    # the scoring config left Mistral out: reported as not scored, no block
    assert m["models_not_scored"] == ["mistral-7b-v0.3"] and "mistral-7b-v0.3" not in m
    assert m["items"]["n_union"] == len(mech2.battery_items(primary, 30)["union"])
    status = (mock_root / "mech2_smoke_analysis" / "STATUS.md").read_text()
    assert "yes/no" in status and "untested" in status and "cannot be the sole account" in status


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
    # amendment after review 2: Mistral is scored only on the arms the primary contrasts read
    assert {mech2.split_condition(r["condition"])[0] for r in recs if r["model"] == "mistral-7b-v0.3"} == set(mech2.PRIMARY_ARMS)
    assert {mech2.split_condition(r["condition"])[0] for r in recs if r["model"] == "qwen3-8b-fewshot"} == set(mech2.ARMS)


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


# ---------------- amendment after review 2 (2026-11-20, before any score) ----------------


def _words(line):
    return line.split()


def test_minimal_pairs_differ_only_where_designed():
    """P1, P2 and P8 compare lines that differ only in the designed words."""
    for _, verb, t in _all_variants():
        keep_sw, ignore = mech2.task_line(t, "desire_keep_sw"), mech2.task_line(t, "mention_ignore")
        sw = t["swapped_task_line"]
        assert keep_sw.startswith(sw + " ") and ignore.startswith(sw + " ")  # the same swapped task in both
        keep = mech2.task_line(t, "desire_keep")[len("Your task is to: "):]
        assert keep_sw[len(sw) + 1:] == keep[0].upper() + keep[1:]  # desire_keep's own sentence, after the swapped task
        check, can = _words(mech2.task_line(t, "desire_check")), _words(mech2.task_line(t, "desire_check_can"))
        assert len(check) == len(can) and [(a, b) for a, b in zip(check, can) if a != b] == [("cannot.", "can.")]
        if verb == "take":
            neg, aff = _words(mech2.task_line(t, "negated")), _words(mech2.task_line(t, "affirmed"))
            assert len(neg) == len(aff) and [(a, b) for a, b in zip(neg, aff) if a != b] == [("no", "an")]


def _note_table(heading: str) -> list[list[str]]:
    text = (Path(__file__).resolve().parents[1] / "notes" / "mechanism2_design.md").read_text()
    rest = text.split("### 12.3 " + heading, 1)[1]
    table = next(b for b in rest.split("\n\n") if b.startswith("|"))  # the first table after the heading
    rows = [ln for ln in table.splitlines() if ln.startswith("|")]
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in rows]


def _cell(text: str) -> frozenset[str] | None:
    if not text:
        return None
    out: set[str] = set()
    for tok in re.split(r",\s*|\s+or\s+", text):
        out |= {"S", "L"} if tok == "+" else {tok}
    assert out <= set(mech2.ZONES), text
    return frozenset(out)


def test_the_prediction_table_in_the_note_is_the_code_s():
    """The amended table (section 12) read cell by cell, "+" as S or L, equals PREDICTIONS exactly."""
    rows = _note_table("Prediction table (amended after review 2)")
    header, body = rows[0], rows[2:]
    accounts = {"H2 evidence": "H2_evidence", "H3 referent salience": "H3_salience", "H4 wishful": "H4_wishful",
                "H5 policy blending": "H5_policy_blend", "H6_lexical": "H6_lexical", "H6_position": "H6_position"}
    cols = [accounts[h] for h in header[1:]]
    parsed: dict[str, dict[str, frozenset[str]]] = {h: {} for h in cols}
    names = {c.split("_")[0]: c for c in mech2.PRIMARY}
    assert len(body) == len(mech2.PRIMARY)
    for row in body:
        name = names[row[0].split()[0]]
        for h, cell in zip(cols, row[1:]):
            z = _cell(cell)
            if z is not None:
                parsed[h][name] = z
    assert parsed == mech2.PREDICTIONS


def test_p6_counts_every_step0_item_of_the_union_even_outside_the_sample():
    items = _syn_items()
    step0 = {t["id"] for t in items if t["history_len"] == 0}
    sample = {t["id"] for t in items if t["history_len"] > 0} | set(sorted(step0)[:10])  # most step-0 items left out
    pops = mech2.populations(items, sample, step0)
    fam = mech2.primary_family(mech2._lookup(_syn_scores(items, FINGERPRINTS["H3_salience"])), pops, B=30, perms=30)
    assert fam["contrasts"]["P6_position_step0"]["n_items"] == len(step0) > 10
    assert fam["contrasts"]["P1_keep_vs_ignore"]["n_items"] == len(sample)


def test_two_format_rule_contradicted_in_one_format_and_undecided_in_the_other_is_undecided():
    none = {c: None for c in mech2.PRIMARY}
    cand = mech2.readout({**none, "P8_negated_vs_affirmed": ("Z",)})  # contradicts H2 (N) in this format
    yn = mech2.readout({**none, "P8_negated_vs_affirmed": ("N", "Z")})  # undecided for H2
    assert cand["H2_evidence"]["per_contrast"]["P8_negated_vs_affirmed"] == "contradicted"
    assert yn["H2_evidence"]["per_contrast"]["P8_negated_vs_affirmed"] == "undecided"
    j = mech2.joint_readout(cand, yn)
    assert j["H2_evidence"]["per_contrast"]["P8_negated_vs_affirmed"] == "undecided"
    assert j["H2_evidence"]["verdict"] == "untested" and j["H2_evidence"]["n_contradicted"] == 0
    assert "H2_evidence" in j["summary"]["content_accounts_untested"]
    assert "H2_evidence" not in j["summary"]["content_accounts_contradicted"]
    # the same in the other order
    assert mech2.joint_readout(yn, cand)["H2_evidence"]["verdict"] == "untested"


def test_consistent_needs_a_rival_contradicted_on_the_same_contrast():
    none = {c: None for c in mech2.PRIMARY}
    # P5 in N and Z: consistent with H4 (N or Z), undecided for H2 (N) and the Z accounts; no rival contradicted
    ro = mech2.readout({**none, "P5_done_vs_cancelled": ("N", "Z")})
    assert ro["H4_wishful"]["per_contrast"]["P5_done_vs_cancelled"] == "consistent"
    assert ro["H4_wishful"]["verdict"] == "untested" and ro["H4_wishful"]["consistent_where_a_rival_fails"] == []
    # P4 large: consistent with H2-H5, H6_lexical contradicted, so each of H2-H5 is consistent
    ro = mech2.readout({**none, "P4_need_without_name": ("L",)})
    assert all(ro[h]["verdict"] == "consistent" for h in ("H2_evidence", "H3_salience", "H4_wishful", "H5_policy_blend"))
    # P6: H6_position's rival is "no position effect"
    ro = mech2.readout({**none, "P6_position_step0": ("N",)})
    assert ro["H6_position"]["verdict"] == "consistent" and ro["rivals"]["no_position_effect"]["P6_position_step0"] == "contradicted"


def test_no_two_accounts_share_a_row_and_the_thin_pairs_are_known():
    sep = mech2.separations()
    assert not any(v["identical_rows"] for v in sep.values())
    content = [k for k in sep if "H6_position" not in k]
    assert all(sep[k]["separate"] for k in content)
    thin = {k: v["separate"] for k, v in sep.items() if k in content and len(v["separate"]) == 1}
    assert thin == {"H3_salience vs H6_lexical": ["P4_need_without_name"], "H4_wishful vs H5_policy_blend": ["P2_hope_polarity"]}


def test_holm_family_of_eight_counts_a_missing_p_as_one():
    items = _syn_items()
    scores = [s for s in _syn_scores(items, FINGERPRINTS["H4_wishful"], noise=0.5)
              if mech2.split_condition(s["condition"])[0] != "affirmed"]  # P8 cannot be computed
    fam = _family(scores, items)
    assert fam["contrasts"]["P8_negated_vs_affirmed"]["n_items"] == 0
    ps = {k: v["p"] for k, v in fam["contrasts"].items() if v["p"] == v["p"]}
    smallest = min(ps, key=ps.get)
    assert 8 * ps[smallest] < 1 and abs(fam["contrasts"][smallest]["holm_p"] - 8 * ps[smallest]) < 1e-12
    assert fam["contrasts"]["P8_negated_vs_affirmed"]["holm_p"] == 1.0


def test_dropped_draw_rule_counts_resamples_not_items(monkeypatch):
    """r is not estimable when more than 2% of the B resamples are undefined, however many items there are."""
    items = _syn_items()  # 480 items: 2% of the items would allow 9 dropped draws
    by = mech2._lookup(_syn_scores(items, FINGERPRINTS["H4_wishful"], noise=0.0))
    real = mech2.cluster_bootstrap

    def fake(fn, xs, games, B, seed):
        out = real(fn, xs, games, B=B, seed=seed)
        return {**out, "n_dropped": 3} if fn is mech2._ratio else out
    monkeypatch.setattr(mech2, "cluster_bootstrap", fake)
    r = mech2.contrast(by, items, ((1, "desire_keep"), (-1, "swapped")), reference=REF, B=100, perms=20)
    assert r["n_items"] > 3 / 0.02 and r["ratio"]["n_dropped"] == 3 and not r["ratio"]["estimable"]
    r = mech2.contrast(by, items, ((1, "desire_keep"), (-1, "swapped")), reference=REF, B=200, perms=20)
    assert r["ratio"]["estimable"]  # 3 of 200 is 1.5%


def test_secondary_readouts_read_the_arms_they_name():
    items = _syn_items()
    fp = {**FINGERPRINTS["H2_evidence"], "done": -0.3, "cancelled": 0.5, "desire_keep": 0.3}
    block = mech2.model_block(_syn_scores(items, fp, noise=0.0, yn_scale=0.5), items, {t["id"] for t in items},
                              {t["id"] for t in items if t["history_len"] == 0}, B=20, perms=20)
    sec = block["secondary"]
    # done - cancelled on every placement item, as first designed
    assert abs(sec["done_vs_cancelled_all_placement"]["estimate"] + 0.8 * D) < 1e-9
    # P5 against cancelled's own lift over swapped: -0.8 / 0.5
    assert abs(sec["P5_against_cancelled_lift"]["cand"]["ratio"]["estimate"] + 1.6) < 1e-9
    assert abs(sec["cancelled_lift_on_p5_items"]["yn"]["estimate"] - 0.5 * 0.5 * D) < 1e-9
    # the first P1 and P2, now secondary
    assert abs(sec["old_P1_desire_keep_vs_ignore"]["cand"]["ratio"]["estimate"] + 0.4) < 1e-9
    assert abs(sec["old_P2_desire_check_vs_swapped"]["yn"]["ratio"]["estimate"] - 0.6) < 1e-9
    assert abs(sec["desire_keep_alone_vs_after_swapped"]["cand"]["estimate"] + 0.5 * D) < 1e-9
    # P7 and P8 on the items whose receptacle contents the history does not show (here every take item)
    assert abs(sec["P7_P8_contents_not_shown"]["P8_negated_vs_affirmed"]["yn"]["ratio"]["estimate"] + 1.4) < 1e-9
    assert sec["P7_P8_contents_not_shown"]["P7_negated_existence"]["cand"]["n_items"] == len(
        [t for t in items if mech2.p7_eligible(t)])
    # the mixture readout carries P5, P7 and P8 in both formats
    mix = block["mixture_readout"]
    assert abs(mix["P8_negated_vs_affirmed"]["yn"]["estimate"] + 1.4 * 0.5 * D) < 1e-9
    assert abs(mix["P8_negated_vs_affirmed"]["cand"]["estimate"] + 1.4 * D) < 1e-9
    assert mix["P5_done_vs_cancelled"]["cand"]["ratio"]["estimate"] == block["primary"]["contrasts"]["P5_done_vs_cancelled"]["ratio"]["estimate"]
    # the first-token term (L_first = L / 2 in the synthetic scores) has the full L's sign: no split
    assert not block["secondary"]["first_token_P7_P8"]["P7_negated_existence"]["sign_split_with_full_L"]


def test_first_token_sign_split_is_flagged():
    items = _syn_items()
    scores = _syn_scores(items, FINGERPRINTS["H2_evidence"], noise=0.0)
    for s in scores:  # the denial raises the first token while it lowers the full L (rq1d's split for Qwen)
        if s["condition"] == "negated":
            s["L_first"] = s["L"] + 20.0
    block = mech2.model_block(scores, items, {t["id"] for t in items}, {t["id"] for t in items if t["history_len"] == 0},
                              B=20, perms=20)
    assert block["primary"]["contrasts"]["P7_negated_existence"]["estimate"] < 0
    assert block["secondary"]["first_token_P7_P8"]["P7_negated_existence"]["sign_split_with_full_L"]


def test_negated_by_history_reads_takes_of_the_goal_object_only():
    t = _item(id="g", scene_locations={})
    other = _item(id="o", action="take egg 1 from countertop 2")
    by = {}
    for x in (t, other):  # a stray negated score on a take of another object must not enter the readout
        by[(x["id"], "negated")] = {"L": 1.0}
        by[(x["id"], "swapped")] = {"L": 0.0}
        by[(x["id"], "aligned")] = {"L": 2.0}
    block = mech2.model_block(list({"transition_id": k[0], "condition": k[1], **v} for k, v in by.items()),
                              [t, other], {"g", "o"}, set(), B=5, perms=5)
    assert block["secondary"]["negated_by_history"]["contents not shown"]["n_items"] == 1


def test_knowledge_split_reads_the_content_contrasts_only():
    items = _syn_items(n_games=6, per_game=6)
    scores = _syn_scores(items, FINGERPRINTS["H3_salience"], noise=0.0)
    mech_scores = [{"transition_id": t["id"], "condition": c, "L": v} for t in items
                   for c, v in (("state_absent", -1.0), ("statetwin_absent", 1.0))]
    out = mech2.knowledge_split(scores, mech_scores, items, B=10)
    assert set(out) == {"aligned_effect"} | {n for n, s in mech2.PRIMARY.items() if s["population"] != "step0"}
    assert "P6_position_step0" not in out and "P8_negated_vs_affirmed" in out


def test_analysis_reports_the_secondary_model_by_default(mock_root):
    models = ["qwen3-8b-fewshot", "llama3.1-8b-fewshot", "mistral-7b-v0.3"]
    assert mech2.run_mech2_score(_cfg(n_shards=1, models=models)) == 0
    assert mech2.run_mech2_analysis({"run_id": "mech2_smoke_analysis", "source": "mech2_smoke", "sample_n": 30,
                                     "sources": ["rq12_transitions_smoke"], "bootstrap": 20, "permutations": 20}) == 0
    m = json.loads((mock_root / "mech2_smoke_analysis" / "metrics.json").read_text())
    assert m["models_not_scored"] == [] and m["mistral-7b-v0.3"]["role"].startswith("secondary")
    assert m["integrity"]["mistral-7b-v0.3"]["status"] == "complete"


def test_scoring_records_the_item_files_hashes(mock_root):
    assert mech2.run_mech2_score(_cfg(n_shards=1)) == 0
    inputs = json.loads((mock_root / "mech2_smoke" / "shard_00_of_01" / "inputs.json").read_text())["inputs"]
    assert [Path(k).parent.name for k in inputs] == ["rq12_transitions_smoke"] and all(len(v) >= 16 for v in inputs.values())


def _scored_and_analysed(mock_root, edit):
    assert mech2.run_mech2_score(_cfg(n_shards=1)) == 0
    path = mock_root / "mech2_smoke" / "shard_00_of_01" / "scoring.jsonl"
    recs = edit(read_jsonl(path))
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    rc = mech2.run_mech2_analysis({"run_id": "mech2_smoke_analysis", "source": "mech2_smoke", "sample_n": 30,
                                   "models": list(rq1.FEWSHOT), "sources": ["rq12_transitions_smoke"],
                                   "bootstrap": 20, "permutations": 20})
    d = mock_root / "mech2_smoke_analysis"
    return rc, json.loads((d / "integrity.json").read_text()), (d / "STATUS.md").read_text(), (d / "metrics.json").exists()


def test_analysis_refuses_a_score_of_a_changed_item(mock_root):
    def edit(recs):
        recs[0]["item_hash"] = "an older item"
        return recs
    rc, integ, status, metrics = _scored_and_analysed(mock_root, edit)
    assert rc == 1 and integ["refuse"] and not metrics and "nothing analysed" in status
    assert integ[next(iter(rq1.FEWSHOT))]["n_hash_mismatch"] + integ[list(rq1.FEWSHOT)[1]]["n_hash_mismatch"] == 1


def test_analysis_refuses_missing_or_doubled_scores(mock_root):
    rc, integ, _, metrics = _scored_and_analysed(mock_root, lambda recs: recs[1:])
    assert rc == 1 and not metrics and sum(integ[t]["n_missing"] for t in rq1.FEWSHOT) == 1
    cells = [c for t in rq1.FEWSHOT for c in integ[t]["cells_off"].values()]
    assert cells and cells[0]["present"] == cells[0]["expected"] - 1


def test_analysis_refuses_doubled_scores(mock_root):
    rc, integ, _, _ = _scored_and_analysed(mock_root, lambda recs: recs + recs[:1])
    assert rc == 1 and sum(integ[t]["n_duplicates"] for t in rq1.FEWSHOT) == 1


def test_integrity_counts_expected_scores_per_model_arm_and_format():
    items = [_item(id="a", step0=True), _item(id="b", action=ACTIONS["move"])]
    scores = [{"model": "mistral-7b-v0.3", "transition_id": t["id"], "condition": mech2.condition(a, f),
               "item_hash": mech2._item_hash(t), "L": 0.0} for t, a, f in mech2.jobs(items, mech2.PRIMARY_ARMS)]
    out = mech2.score_integrity(scores, items, ["mistral-7b-v0.3", "qwen3-8b-fewshot"])
    assert out["mistral-7b-v0.3"]["status"] == "complete" and out["qwen3-8b-fewshot"]["status"] == "not scored"
    assert out["refuse"]  # a primary model with no score
    assert not mech2.score_integrity(scores, items, ["mistral-7b-v0.3"])["refuse"]
    extra = scores + [{**scores[0], "condition": "neutral"}]  # an arm Mistral is not scored on
    assert mech2.score_integrity(extra, items, ["mistral-7b-v0.3"])["mistral-7b-v0.3"]["n_unexpected"] == 1


def test_mistral_arms_are_those_the_primary_contrasts_read():
    read = {arm for spec in mech2.PRIMARY.values() for _, arm in spec["terms"] + spec["reference"]}
    assert set(mech2.PRIMARY_ARMS) == read and set(mech2.model_arms("mistral-7b-v0.3")) == read
    assert {"neutral", "desire_keep", "mention_exists"}.isdisjoint(read)
    assert mech2.model_arms("qwen3-8b-fewshot") == mech2.ARMS == mech2.model_arms("llama3.1-8b-fewshot")


def test_contents_not_shown_reads_only_the_takes_whose_receptacle_was_not_seen():
    items = _syn_items(n_games=20, per_game=6)
    for k, t in enumerate(items):
        if k % 2 and t["history_len"]:  # the history shows countertop 2 without an apple: P7 item, contents shown
            t["history_text"] = t["history_text"].replace("go to countertop 1", "go to countertop 2").replace(
                "You arrive at countertop 1. On the countertop 1", "You arrive at countertop 2. On the countertop 2")
    shown = [t for t in items if mech2.is_take(t) and mech2.negated_status(t)[1] == "shown, without it"]
    assert shown
    block = mech2.model_block(_syn_scores(items, FINGERPRINTS["H2_evidence"], noise=0.0), items, {t["id"] for t in items},
                              {t["id"] for t in items if t["history_len"] == 0}, B=10, perms=10)
    ns = block["secondary"]["P7_P8_contents_not_shown"]["P7_negated_existence"]["cand"]
    assert ns["n_items"] == sum(mech2.p7_eligible(t) for t in items) - len(shown)
