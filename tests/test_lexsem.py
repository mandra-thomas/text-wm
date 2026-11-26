"""Lexsem diagnostic: the builder (wf/lexsem.py), the rules (wf/lexsem_rules.py) and the scoring script's pure parts
(scripts/lexsem/score.py)."""
import re

import pytest

from scripts.lexsem import score as lscore
from scripts.qwen_audit.followup_score import plain_text
from tests.test_rq1 import _item, mock_score  # noqa: F401
from tests.test_rq1x import VOCAB
from wf import lexsem, rq1, rq1v, rq1x
from wf import lexsem_rules as R
from wf.scoring.prompts import WM_FEWSHOT_SETS

ALL = tuple(dict.fromkeys(c for g in lexsem.GROUPS.values() for c in g))


def _t(**kw):
    return _item(history_len=1, scene_locations={"saltshaker 1": ["shelf 1"], "mug 1": ["cabinet 2"],
                                                  "apple 1": ["fridge 1"]}, **kw)


def _pan():
    return _t(id="t2", goal_object="pan", action="take pan 1 from stoveburner 1",
              task_line="Your task is to: put some pan on cabinet.",
              candidates={"o_plus": "You pick up the pan 1 from the stoveburner 1.", "o_minus": "Nothing happens."})


def _added(spec: dict, t: dict) -> str:
    if spec.get("pre") or spec.get("post"):
        assert spec["task_line"] == t["swapped_task_line"]
        return spec.get("pre") or spec.get("post")
    if spec["task_line"] == lexsem.rq1.ABSENT_LINE:  # the competence condition: the neutral task, no sentence
        return "Neutral."
    if spec["task_line"].startswith(t["task_line"] + " "):  # the competition line is built on the real task
        return spec["task_line"][len(t["task_line"]) + 1:]
    assert spec["task_line"].startswith(t["swapped_task_line"] + " ")
    return spec["task_line"][len(t["swapped_task_line"]) + 1:]


def test_every_mention_condition_adds_one_sentence_and_stage0_is_reproduced(mock_score):
    t = _t()
    sp = lexsem.specs(t, ALL)
    assert sp["aligned"]["task_line"] == t["task_line"] and sp["swapped"]["task_line"] == t["swapped_task_line"]
    for c in sp:
        if c not in lexsem.ANCHORS:
            s = _added(sp[c], t)
            assert s[0].isupper() and s.endswith("."), (c, s)
    assert sp["name_assert"]["task_line"] == rq1.goal_line(t, "swapped_plus")  # stage 0's sentence, byte for byte
    assert "alias_assert" not in sp and "homonym" not in sp and "variant_assert" in sp


def test_name_carriers_keep_the_names_leading_space_and_word_boundary(mock_score):
    """M1: in every carrier the name is preceded by a space and followed by a space or full stop, as in the action
    line, so a pre-tokenizer that splits on spaces and punctuation gives the name the same tokens."""
    for t in (_t(), _pan()):
        sp = lexsem.specs(t, ALL)
        for c, (form, _, _) in lexsem.MENTIONS.items():
            if c in sp and form in ("name", "homonym"):
                s = _added(sp[c], t)
                assert re.search(rf"(?<= ){t['goal_object']}(?=[ .])", s), (c, s)
                assert not re.search(rf"[\"'(]{t['goal_object']}", s), (c, s)


def test_sentences_carry_the_intended_content():
    assert lexsem.sentence("apple", "assert") == "There is an apple somewhere in the room."
    assert lexsem.sentence("mug", "deny") == "There is no mug in this room."
    assert lexsem.sentence("mug", "use") == "The following word appears in this note: mug."
    assert lexsem.sentence("round fruit", "use", phrase=True) == "The following phrase appears in this note: round fruit."
    assert lexsem.sentence("egg", "elsewhere") == "There is an egg somewhere in the house next door."
    for content in ("use", "elsewhere", "deny", "ignore"):  # only "assert" claims the object is in this room
        assert not re.search(r"there is (a|an) mug somewhere in the room", lexsem.sentence("mug", content), re.I)


def test_only_name_and_homonym_conditions_contain_the_exact_name(mock_score):
    for t in (_t(), _pan()):
        sp = lexsem.specs(t, ALL)
        for c, (form, _, _) in lexsem.MENTIONS.items():
            if c in sp:
                has = re.search(rf"\b{t['goal_object']}\b", _added(sp[c], t)) is not None
                assert has == (form in ("name", "homonym")), c


def test_prompts_differ_only_where_the_condition_says(mock_score):
    t = _t()
    sp = lexsem.specs(t, ALL)
    for fmt in lexsem.FORMATS:
        base = lexsem.messages(t, sp["swapped"], fmt)
        assert base[-1]["content"].endswith(rq1v.YN_QUESTION if fmt == "yn" else "Next observation:")
        for c, spec in sp.items():
            m = lexsem.messages(t, spec, fmt)
            assert m[:-1] == base[:-1]
            content = m[-1]["content"]
            if spec.get("pre"):
                assert f"\n{spec['pre']}\n\nAction: " in content
                content = content.replace(f"\n{spec['pre']}", "", 1)
            if spec.get("post"):
                assert re.search(rf"\nAction: [^\n]+\n\n{re.escape(spec['post'])}\n\n", content)
                content = content.replace(f"\n\n{spec['post']}", "", 1)
            assert content.replace(spec["task_line"], sp["swapped"]["task_line"], 1) == base[-1]["content"], (c, fmt)
    assert lexsem.messages(t, sp["aligned"], "yn") == rq1v.arm_messages(t, "yn_aligned")
    text = plain_text(lexsem.messages(t, sp["name_use"], "obs"))
    assert text.count("The following word appears in this note: saltshaker.") == 1


def test_control_object_rules():
    t = _t()
    for k in range(20):
        ti = {**t, "id": f"t{k}"}
        o = lexsem.control_object(ti)
        assert o not in {"saltshaker", "mug", "apple"} | lexsem.FEWSHOT_OBJECTS
        assert lexsem.CLASS_OF[o] != lexsem.CLASS_OF["saltshaker"]
        assert lexsem.NAME_TOKENS[o] == lexsem.NAME_TOKENS["saltshaker"]
    assert lexsem.control_object(t) == lexsem.control_object(t)
    refs = lexsem.references(t)
    assert refs["cdesc"] == rq1x.DESCRIPTIONS[refs["ctrl"]]  # one absent object gives both controls (S3)


def test_frozen_tables():
    assert set(lexsem.NAME_TOKENS) == set(rq1x.DESCRIPTIONS) == set(lexsem.CLASS_OF)
    text = str(WM_FEWSHOT_SETS[1])
    assert {o for o in rq1x.DESCRIPTIONS if re.search(rf"\b{o}\b", text)} == set(lexsem.FEWSHOT_OBJECTS)
    for obj, alias in lexsem.ALIASES.items():
        assert alias not in VOCAB and alias not in obj and obj not in alias
        assert not set(rq1x.OWN_PARTS.get(obj, ())) & set(alias.split())
        assert lexsem.ALIAS_FOIL[obj] != obj
    for obj, var in lexsem.VARIANTS.items():
        assert var.replace(" ", "") == obj and " " in var
    for obj, (s, ctrl) in lexsem.HOMONYMS.items():
        assert s.count(f" {obj} ") == 1 and ctrl not in VOCAB and " " not in ctrl
        assert not re.search(r"\b(room|there is)\b", s, re.I)
    assert set(lexsem.CONTROL_OF) <= set(lexsem.MENTIONS)
    assert all(set(g) <= set(lexsem.MENTIONS) | set(lexsem.ANCHORS) | {lexsem.NEUTRAL} for g in lexsem.GROUPS.values())


def test_lenient_reader():
    assert lexsem.check_reads("soapbar", "Bar of soap") and lexsem.check_reads("keychain", "Key ring")
    assert lexsem.check_reads("saltshaker", "Shaker") and not lexsem.check_reads("mug", "Teapot")
    assert lexsem.check_reads("apple", "An apple.")


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
        lexsem.sample(items, n_later=100, n_step0=10)


def test_build_records_alias_variant_and_homonym_only_where_they_exist(mock_score):
    recs = lexsem.build([_t(), _pan()], ALL)
    assert "variant_assert" in recs[0]["specs"] and "alias_assert" not in recs[0]["specs"]
    assert {"alias_assert", "homonym", "ctrl_homonym"} <= set(recs[1]["specs"])
    assert "skillet" in recs[1]["specs"]["alias_assert"]["task_line"]
    assert "Critics often praise sequels" in recs[1]["specs"]["ctrl_homonym"]["task_line"]


# ---------------- the scoring script's pure parts ----------------

class _Tok:
    """Splits like a byte-level BPE pre-tokenizer: a word with its leading space, or one punctuation mark."""

    def __init__(self, override=None):
        self.vocab: dict[str, int] = {}
        self.override = override or {}

    def __call__(self, text, add_special_tokens=False):
        if text.strip() in self.override:
            return {"input_ids": list(range(self.override[text.strip()]))}
        parts = re.findall(r" ?[A-Za-z0-9]+| ?[^A-Za-z0-9\s]|\s+", text)
        return {"input_ids": [self.vocab.setdefault(p, len(self.vocab)) for p in parts]}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_scoring_script_refuses_mismatched_token_counts():
    tok = _Tok({o: n for o, n in lexsem.NAME_TOKENS.items()})
    assert lscore.token_checks(tok) == []
    tok.override["mug"] = 2
    assert lscore.token_checks(tok) == ["mug: 2 != 1"]
    assert lscore.sample_digest(["b", "a"]) == lscore.sample_digest(["a", "b"])
    assert lscore.count_occurrences([1, 2, 1, 2], [1, 2]) == 2


def test_token_identity_check_passes_the_frozen_carriers_and_refuses_a_quoted_name(mock_score, monkeypatch):
    t = _pan()
    rec = lexsem.build([t], ALL)[0]
    jobs = lscore.item_jobs(t, rec, _Tok(), "plain", ("obs",))
    assert {j["cond"] for j in jobs} == set(rec["specs"])
    monkeypatch.setitem(lexsem.TEMPLATES, "use", 'The {unit} "{ref}" appears in this note.')  # the old carrier
    rec = lexsem.build([t], ("swapped", "name_use"))[0]
    with pytest.raises(SystemExit, match="copies of the name"):
        lscore.item_jobs(t, rec, _Tok(), "plain", ("obs",))


def test_failure_text_only_for_the_chosen_conditions(mock_score):
    t = _pan()
    rec = lexsem.build([t], lexsem.GROUPS["core"] + lexsem.GROUPS["ignore"])[0]
    core = frozenset(lexsem.GROUPS["core"])
    jobs = lscore.item_jobs(t, rec, _Tok(), "plain", ("obs", "yn"), core)
    assert {j["cond"] for j in jobs if j["fmt"] == "fail"} == set(rec["specs"]) & core
    j = next(j for j in jobs if j["fmt"] == "fail")
    assert _Tok().decode([]) == "" and len(j["ids"]) > j["n_prompt"]


# ---------------- the rules ----------------

def _iv(points, half=0.0):
    return {k: (v - half, v + half) for k, v in points.items()}


@pytest.mark.parametrize("weights,expected", [
    ({"lexical": 1}, "only lexical fits"),
    ({"semantic": 1}, "only semantic fits"),
    ({"evidence": 1}, "only evidence fits"),
    ({"lexical": .5, "evidence": .5}, "no single proposed account"),
    ({"semantic": .5, "evidence": .5}, "no single proposed account"),  # read "only evidence" by the first rules (M2)
    ({"lexical": .5, "semantic": .5}, "no single proposed account"),  # read "only lexical" by the first rules (M2)
    ({"semantic": .3, "evidence": .7}, "no single proposed account"),
    ({"lexical": .7, "semantic": .3}, "no single proposed account"),
])
def test_planted_scenarios_without_noise(weights, expected):
    assert R.single_account_reading(_iv(R.planted(weights)), (0.9, 1.0)).startswith(expected)


def test_rule_wording_and_gates():
    pts = R.planted({"evidence": 1})
    assert R.single_account_reading(_iv(pts), (0.1, 0.6)).startswith("not assessable")
    wide = R.single_account_reading(_iv(pts, 0.3), (0.9, 1.0))  # every contrast imprecise
    assert wide.startswith("not distinguished between lexical and semantic and evidence")
    near = R.single_account_reading(_iv(R.planted({"semantic": .15, "evidence": .85}), 0.12), (0.9, 1.0))
    assert near.startswith("evidence not contradicted, not shown consistent")
    alias_only = {**_iv(R.planted({"semantic": 1})), "alias_assert": (-0.1, 0.1)}
    assert R.account_statuses(alias_only)["semantic"]["overall"] == "undecided (alias or homonym only)"


def test_other_readings():
    assert R.mention_reading(0.8, 1.1) == "a non-goal assertion reproduces the goal effect"
    assert R.mention_reading(0.3, 0.7) == "a non-goal assertion reproduces part of the goal effect"
    assert R.relevance_reading(-0.1, 0.2) == {"sole": "consistent", "component": "the goal adds beyond a non-goal assertion"}
    assert R.relevance_reading(0.8, 1.0)["component"].startswith("no goal-role component")
    assert R.difference_reading(-0.6, -0.3, "lower", "same", "higher") == "lower"
    assert R.difference_reading(-0.2, 0.2, "lower", "same", "higher") == "same"
    assert R.difference_reading(-0.3, 0.2, "lower", "same", "higher") == "inconclusive"
    w = R.mixture_weights(R.planted({"semantic": .5, "evidence": .5}))
    assert abs(w["semantic"] - .5) < 1e-9 and abs(w["evidence"] - .5) < 1e-9 and abs(w["lexical"]) < 1e-9


def test_stratum_specific_needs_a_step_difference():
    st_ev = R.account_statuses(_iv(R.planted({"evidence": 1})))
    st_sem = R.account_statuses(_iv(R.planted({"semantic": 1})))
    assert R.stratum_specific(st_ev, st_sem, {"name_use": (-0.2, 0.2)}) == []
    assert "evidence" in R.stratum_specific(st_ev, st_sem, {k: (0.5, 1.2) for k in R.PREDICTIONS})


class _BosTok(_Tok):
    """As _Tok, with a BOS token added in front of a text when asked (Llama-like) or never (Qwen-like)."""

    def __init__(self, bos: bool):
        super().__init__()
        self.adds = bos
        self.bos_token_id = 10**6 if bos else None

    def __call__(self, text, add_special_tokens=False):
        ids = super().__call__(text)["input_ids"]
        return {"input_ids": ([self.bos_token_id] if (self.adds and add_special_tokens) else []) + ids}


@pytest.mark.parametrize("bos", [True, False])
def test_plain_prompts_carry_the_tokenizers_bos_exactly_once(mock_score, bos):
    tok = _BosTok(bos)
    t = _pan()
    rec = lexsem.build([t], ALL)[0]
    for j in lscore.item_jobs(t, rec, tok, "plain", ("obs", "yn"), frozenset(lexsem.GROUPS["core"])):
        n = j["ids"].count(tok.bos_token_id) if bos else 0
        assert (n == 1 and j["ids"][0] == tok.bos_token_id) if bos else n == 0
    if bos:  # a Llama-like prompt without its BOS, or with it twice, is refused
        for ids in ([5, 6], [tok.bos_token_id, tok.bos_token_id, 5]):
            with pytest.raises(SystemExit, match="BOS"):
                lscore.check_bos(tok, ids)
