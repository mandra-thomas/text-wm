"""Mechanism battery (wf/mech2.py): templates per task type, prompt differences, items, the tokenization gate, the
analysis on synthetic data with each account's fingerprint planted, and the runner end to end on mock data."""
import json
import random
import re
from pathlib import Path

import pytest

from wf import constants as C
from wf import mech2, rq1, rq1v, rq1x
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


def test_every_arm_builds_for_every_task_type_and_verb():
    for ttype, verb, t in _all_variants():
        for arm in mech2.ARMS:
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
            for arm in mech2.CONTENT_ARMS:
                msgs = mech2.arm_messages(t, arm, fmt)
                assert msgs[:-1] == base[:-1], arm
                assert _query(msgs) == _query(base).replace(t["task_line"], mech2.task_line(t, arm)), arm


def test_only_the_real_task_arms_show_the_real_task(mock_score):
    real = {"aligned", "done", "cancelled", "aligned_start", "aligned_end"}
    for _, _, t in _all_variants():
        for arm in mech2.ARMS:
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
        for arm in mech2.ARMS:
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
# Mean of each content arm minus the swapped arm, in units of the aligned effect, for each account's fingerprint.
FINGERPRINTS = {
    "H2_evidence": {"desire_keep": 0.8, "desire_check": 0.0, "mention_ignore": 0.0, "described": 0.9, "done": -0.3, "cancelled": 0.6},
    "H3_salience": {"desire_keep": 0.8, "desire_check": 0.9, "mention_ignore": 0.9, "described": 0.0, "done": 0.8, "cancelled": 0.8},
    "H4_wishful": {"desire_keep": -0.5, "desire_check": -0.5, "mention_ignore": 0.0, "described": 0.9, "done": 0.0, "cancelled": 0.0},
    "H5_policy_blend": {"desire_keep": -0.3, "desire_check": 0.9, "mention_ignore": 0.0, "described": 0.9, "done": 0.0, "cancelled": 0.0},
}


def _syn_items(n_games=40, per_game=12):
    items = []
    for g in range(n_games):
        for k in range(per_game):
            task = TASKS["examine"][0] if k % 6 == 5 else TASKS["clean"][0]
            items.append(_item(task, id=f"g{g}k{k}", game_id=f"valid_unseen/{g}", step0=k < 3))
    return items


def _syn_scores(items, fingerprint, position_shift=0.0, seed=7, noise=1.0):
    rng = random.Random(seed)
    game = {}
    out = []
    for t in items:
        g = game.setdefault(t["game_id"], rng.gauss(0, 2))
        base = g + rng.gauss(-8, 2)
        mu = {"aligned": D, "swapped": 0.0, "neutral": -0.4 * D, "mention_exists": 0.9 * D, "swapped_described": 0.0,
              **{a: v * D for a, v in fingerprint.items()}}
        mu["aligned_start"] = mu["aligned"] + (position_shift * D if t["history_len"] == 0 else 0.0)
        mu["swapped_start"] = mu["swapped"]
        mu["aligned_end"], mu["swapped_end"] = mu["aligned"], mu["swapped"]
        for arm, m in mu.items():
            if mech2.alias_of(t, arm):
                continue
            for fmt in mech2.FORMATS:
                L = base + m + rng.gauss(0, noise)
                out.append({"transition_id": t["id"], "condition": mech2.condition(arm, fmt), "L": L, "L_first": L / 2,
                            "game_id": t["game_id"], "model": "m"})
    return out


def _family(scores, items, B=200):
    sample = {t["id"] for t in items}
    step0 = {t["id"] for t in items if t["history_len"] == 0}
    pops = mech2.populations(items, sample, step0)
    return mech2.primary_family(mech2._lookup(scores), pops, "cand", B=B, perms=200)


@pytest.mark.parametrize("planted", sorted(FINGERPRINTS))
def test_each_planted_fingerprint_is_read_as_its_own_account(planted):
    items = _syn_items()
    fam = _family(_syn_scores(items, FINGERPRINTS[planted]), items)
    ro = fam["readout"]
    assert ro[planted]["n_contradicted"] == 0 and ro[planted]["n_consistent"] >= 5, (planted, fam["contrasts"])
    for other in FINGERPRINTS:
        if other != planted:
            assert ro[other]["n_contradicted"] >= 1, (planted, other, ro[other])


def test_a_planted_position_effect_is_read_as_an_artefact():
    items = _syn_items()
    fam = _family(_syn_scores(items, FINGERPRINTS["H3_salience"], position_shift=-0.5), items)
    assert fam["contrasts"]["P6_position_step0"]["category"] == "NEG"
    assert fam["readout"]["H6_artefact"]["n_contradicted"] == 0 and fam["readout"]["H6_artefact"]["n_consistent"] == 1
    for h in ("H2_evidence", "H4_wishful", "H5_policy_blend"):
        assert fam["readout"][h]["per_contrast"]["P6_position_step0"] == "contradicted"
    none = _family(_syn_scores(items, FINGERPRINTS["H3_salience"]), items)
    assert none["contrasts"]["P6_position_step0"]["category"] == "NULL"
    assert none["readout"]["H6_artefact"]["n_contradicted"] == 1


def test_contrast_signs_sizes_and_ratio():
    items = _syn_items()
    by = mech2._lookup(_syn_scores(items, FINGERPRINTS["H4_wishful"], noise=0.0))
    r = mech2.contrast(by, items, ((1, "desire_keep"), (-1, "swapped")), reference=((1, "aligned"), (-1, "swapped")), B=50, perms=50)
    assert abs(r["estimate"] + 0.5 * D) < 1e-9 and abs(r["ratio"]["estimate"] + 0.5) < 1e-9
    assert abs(r["reference_effect"] - D) < 1e-9 and r["n_games"] == 40
    p5 = mech2.contrast(by, items, ((1, "done"), (-1, "cancelled")), B=50, perms=50)
    assert abs(p5["estimate"]) < 1e-9


def test_alias_reads_the_plain_score():
    t = _item(step0=True)
    by = {(t["id"], "aligned"): {"L": 2.0}, (t["id"], "yn_swapped"): {"L": -1.0}}
    assert mech2._value(by, t, "aligned_end", "cand") == 2.0 and mech2._value(by, t, "swapped_end", "yn") == -1.0
    assert mech2._value(by, _item(), "aligned_end", "cand") is None


def test_categories_follow_the_rules():
    res = lambda est, lo, hi: {"estimate": est, "ratio": {"ci": [lo, hi]}}
    assert mech2.categorize(res(-1, -0.5, -0.1), 0.01) == "NEG"
    assert mech2.categorize(res(1, 0.6, 1.1), 0.01) == "POS_LARGE"
    assert mech2.categorize(res(1, 0.1, 0.4), 0.01) == "POS_SMALL"
    assert mech2.categorize(res(1, 0.3, 0.8), 0.01) == "POS"
    assert mech2.categorize(res(0.1, -0.2, 0.2), 0.3) == "NULL"
    assert mech2.categorize(res(0.1, -0.3, 0.2), 0.3) == "INCONCLUSIVE"
    assert mech2.categorize(res(0.1, -0.1, 0.1), 0.04) == "POS_SMALL"  # significant after Holm, so not null
    assert mech2.categorize(res(float("nan"), 0, 0), 0.01) == "INCONCLUSIVE"


def test_holm_is_applied_within_the_family():
    items = _syn_items(n_games=12, per_game=6)
    fam = _family(_syn_scores(items, FINGERPRINTS["H4_wishful"], noise=3.0), items, B=50)
    ps = {k: v["p"] for k, v in fam["contrasts"].items()}
    for k, v in fam["contrasts"].items():
        assert v["holm_p"] >= v["p"]
    assert sorted(fam["contrasts"]) == sorted(mech2.PRIMARY)
    smallest = min(ps, key=ps.get)
    assert abs(fam["contrasts"][smallest]["holm_p"] - min(1.0, 6 * ps[smallest])) < 1e-12


def test_readout_marks_undecided_and_unresolved_size():
    cats = {c: "INCONCLUSIVE" for c in mech2.PRIMARY}
    assert all(v["n_undecided"] == len(v["per_contrast"]) for v in mech2.readout(cats).values())
    cats["P2_desire_check"] = "POS"
    ro = mech2.readout(cats)
    assert ro["H5_policy_blend"]["per_contrast"]["P2_desire_check"] == "consistent"
    assert ro["H4_wishful"]["per_contrast"]["P2_desire_check"] == "contradicted"


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

    def __init__(self, space_before_reply=False, split_yes=False):
        self.vocab = dict(self.SPECIAL)
        self.space_before_reply, self.split_yes = space_before_reply, split_yes
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
            if last_reply and self.space_before_reply:
                ids.append(self.vocab.setdefault("▁", len(self.vocab)))
            ids += self._tok(m["content"]) + [4]
        if add_generation_prompt:
            ids.append(3)
        return [ids] if return_tensors else ids

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


class FakeScorer:
    _encode = LogoddsScorer._encode  # the real encoding, so the gate is checked against what scoring renders

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
