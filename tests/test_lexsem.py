"""Lexsem diagnostic builder (wf/lexsem.py) and the scoring script's pure parts (scripts/lexsem/score.py)."""
import re

import pytest

from scripts.lexsem import score as lscore
from scripts.qwen_audit.followup_score import plain_text
from tests.test_rq1 import _item, mock_score  # noqa: F401
from tests.test_rq1x import VOCAB
from wf import lexsem, rq1, rq1v, rq1x


def _t(**kw):
    return _item(history_len=1, scene_locations={"saltshaker 1": ["shelf 1"], "mug 1": ["cabinet 2"],
                                                  "apple 1": ["fridge 1"]}, **kw)


def _added(line: str, t: dict) -> str:
    assert line.startswith(t["swapped_task_line"] + " ")
    return line[len(t["swapped_task_line"]) + 1:]


def test_every_mention_condition_is_the_swapped_line_plus_one_sentence(mock_score):
    t = _t()
    lines = lexsem.task_lines(t)
    assert lines["aligned"] == t["task_line"] and lines["swapped"] == t["swapped_task_line"]
    for c in lexsem.MENTIONS:
        if c == "alias_assert":
            assert c not in lines  # saltshaker has no alias
            continue
        s = _added(lines[c], t)
        assert s.count(".") >= 1 and s[0].isupper() and s.endswith("."), (c, s)
    assert lines["name_assert"] == rq1.goal_line(t, "swapped_plus")  # stage 0's mention sentence, byte for byte


def test_sentences_carry_the_intended_content():
    assert lexsem.sentence("apple", "assert") == "There is an apple somewhere in the room."
    assert lexsem.sentence("mug", "deny") == "There is no mug in this room."
    assert lexsem.sentence("mug", "use") == 'The word "mug" appears in this note.'
    assert lexsem.sentence("round fruit", "use", phrase=True) == 'The phrase "round fruit" appears in this note.'
    assert lexsem.sentence("egg", "generic") == "An egg is a common household item."
    for content in ("use", "generic", "deny"):  # no sentence but "assert" claims the object is in the room
        s = lexsem.sentence("mug", content)
        assert not re.search(r"there is (a|an) mug", s, re.I)


def test_only_name_conditions_contain_the_exact_name(mock_score):
    t = _t()
    lines = lexsem.task_lines(t)
    for c, (form, _) in lexsem.MENTIONS.items():
        if c not in lines:
            continue
        has = re.search(r"\bsaltshaker\b", _added(lines[c], t)) is not None
        assert has == (form == "name"), c


def test_prompts_differ_between_conditions_only_in_the_task_line(mock_score):
    t = _t()
    lines = lexsem.task_lines(t)
    for fmt in lexsem.FORMATS:
        base = lexsem.messages(t, lines["swapped"], fmt)
        for c, line in lines.items():
            m = lexsem.messages(t, line, fmt)
            assert m[:-1] == base[:-1]
            assert m[-1]["content"].replace(line, lines["swapped"], 1) == base[-1]["content"], (c, fmt)
        assert base[-1]["content"].endswith(rq1v.YN_QUESTION if fmt == "yn" else "Next observation:")
    assert lexsem.messages(t, lines["aligned"], "yn") == rq1v.arm_messages(t, "yn_aligned")
    text = plain_text(lexsem.messages(t, lines["name_use"], "obs"))
    assert text.count('The word "saltshaker" appears in this note.') == 1 and text.endswith("Next observation:")


def test_control_word_matches_token_count_and_never_appears_in_the_item():
    for obj, n in lexsem.OBJECT_TOKENS.items():
        for k in range(5):
            t = _item(id=f"t{k}", goal_object=obj, action=f"take {obj} 1 from cabinet 1")
            w = lexsem.control_word(t)
            assert lexsem.CONTROL_POOL[w][0] == n
            nearest = sorted(lexsem.CONTROL_POOL, key=lambda x: (abs(lexsem.CONTROL_POOL[x][1] - lexsem.OBJECT_ZIPF[obj]), x))
            assert w in [x for x in nearest if lexsem.CONTROL_POOL[x][0] == n][: lexsem.N_NEAREST]
    t = _item(history_text=_item()["history_text"] + "\nYou see a globe 1.")
    assert lexsem.control_word(t) != "globe"
    assert lexsem.control_word(_item()) == lexsem.control_word(_item())  # fixed by the item id


def test_frozen_tables_avoid_alfworld_words_and_the_objects_own_parts():
    for w in lexsem.CONTROL_POOL:
        assert w not in VOCAB and not any(w in o or o in w for o in lexsem.OBJECT_TOKENS), w
    for obj, alias in lexsem.ALIASES.items():
        assert alias not in VOCAB and alias not in obj and obj not in alias
        assert not set(rq1x.OWN_PARTS.get(obj, ())) & set(alias.split())
    for obj, var in lexsem.VARIANTS.items():
        assert var.replace(" ", "") == obj and " " in var
    assert set(lexsem.OBJECT_TOKENS) == set(lexsem.OBJECT_ZIPF) and set(lexsem.OBJECT_TOKENS) <= set(rq1x.DESCRIPTIONS)
    assert set(lexsem.CONTROL_OF) >= set(lexsem.MENTIONS) - {k for k in lexsem.MENTIONS if k.startswith(("ctrl", "cdesc"))}


def test_control_description_is_another_absent_object_of_similar_length():
    t = _t()
    obj = None
    for fake in (None, lambda s: len(s)):
        refs = lexsem.references(t, fake)
        obj = refs["cdesc_object"]
        assert obj not in {"saltshaker", "mug", "apple"} and refs["cdesc"] == rq1x.DESCRIPTIONS[obj]
    words = lambda s: len(s.split())  # noqa: E731
    gaps = {o: abs(words(d) - words(rq1x.DESCRIPTIONS["saltshaker"])) for o, d in rq1x.DESCRIPTIONS.items()
            if o not in {"saltshaker", "mug", "apple"}}
    assert gaps[lexsem.references(t)["cdesc_object"]] == min(gaps.values())


def test_strata_and_seeded_sample_with_a_per_game_cap():
    items = []
    for g in range(40):
        for k in range(6):
            hist = "You are in the middle of a room.\nYour task is to: put some saltshaker on cabinet."
            if k >= 3:
                hist += "\n> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a mug 1."
            if k == 5:
                hist += " And a saltshaker 2."  # history names the action's object: excluded
            items.append(_item(id=f"t{g:02d}{k}", game_id=f"valid_unseen/{g}", history_text=hist,
                               history_len=0 if k < 3 else 1))
    assert lexsem.stratum(items[0]) == "step0" and lexsem.stratum(items[3]) == "later"
    assert lexsem.stratum(items[5]) is None
    s = lexsem.sample(items, n_later=50, n_step0=40)
    assert s == lexsem.sample(items, n_later=50, n_step0=40)
    assert sum(t["lexsem_stratum"] == "later" for t in s) == 50 and sum(t["lexsem_stratum"] == "step0" for t in s) == 40
    for st, cap in (("later", lexsem.CAP_LATER), ("step0", lexsem.CAP_STEP0)):
        per = {}
        for t in s:
            if t["lexsem_stratum"] == st:
                per[t["game_id"]] = per.get(t["game_id"], 0) + 1
        assert max(per.values()) <= cap
    assert not any(t["id"].endswith("5") for t in s)
    with pytest.raises(ValueError):
        lexsem.sample(items, n_later=100, n_step0=10)  # 40 games x 2 eligible later items cannot give 100


def test_build_records_alias_and_variant_only_where_they_exist(mock_score):
    recs = lexsem.build([_t(), _t(id="t2", goal_object="pan", action="take pan 1 from stoveburner 1",
                                  task_line="Your task is to: put some pan on cabinet.",
                                  candidates={"o_plus": "You pick up the pan 1 from the stoveburner 1.",
                                              "o_minus": "Nothing happens."})])
    assert "variant_assert" in recs[0]["lines"] and "alias_assert" not in recs[0]["lines"]
    assert "alias_assert" in recs[1]["lines"] and "variant_assert" not in recs[1]["lines"]
    assert "skillet" in recs[1]["lines"]["alias_assert"]
    assert set(recs[0]["lines"]) <= set(lexsem.CORE)


class _FakeTok:
    """Counts frozen in the tables, one token per readout word."""

    def __init__(self, override=None):
        self.n = {**lexsem.OBJECT_TOKENS, **{w: k for w, (k, _) in lexsem.CONTROL_POOL.items()}, **(override or {})}

    def __call__(self, text, add_special_tokens=False):
        w = text.strip()
        return {"input_ids": list(range(self.n.get(w, 1)))}


def test_scoring_script_refuses_mismatched_token_counts():
    assert lscore.token_checks(_FakeTok()) == []
    assert lscore.token_checks(_FakeTok({"umbrella": 2})) == ["umbrella: 2 != 1"]
    assert lscore.sample_digest(["b", "a"]) == lscore.sample_digest(["a", "b"])
