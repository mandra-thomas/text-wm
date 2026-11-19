"""Exploratory follow-up (wf/rq1x.py): object-free goals, working actions, and the description checks."""
import json
import re

import pytest

from wf import rq1, rq1x
from tests.test_rq1 import _analysis_cfg, _item, _score_cfg, _text, mock_root, mock_score  # noqa: F401
from wf.util import read_jsonl

# Every object and receptacle name in the confirmatory items (scanned 2026-11-19), plus the other ALFWorld
# objects a train game can hold.
VOCAB = set("""alarmclock apple armchair bed book bowl box bread butterknife cabinet candle cd cellphone cloth
coffeemachine countertop creditcard cup desk desklamp dishsponge drawer dresser egg floorlamp fork fridge garbagecan
glassbottle handtowel handtowelholder houseplant keychain knife laptop laundryhamper lettuce microwave mug newspaper
pan papertowelroll pen pencil peppershaker pillow plate pot potato remotecontrol safe saltshaker shelf sidetable
sinkbasin soapbar soapbottle sofa spatula spoon spraybottle statue stoveburner tissuebox toaster toilet toiletpaper
toiletpaperhanger tomato towel towelholder vase watch baseballbat basketball kettle ladle plunger scrubbrush teddybear
tennisracket wateringcan winebottle television ottoman cart diningtable coffeetable bathtubbasin""".split())


def test_descriptions_name_no_object_and_no_part_of_their_own():
    for obj, desc in rq1x.DESCRIPTIONS.items():
        words = set(re.findall(r"[a-z]+", desc.lower()))
        assert not words & VOCAB, (obj, words & VOCAB)
        assert not words & set(rq1x.OWN_PARTS.get(obj, ())), (obj, desc)
        assert obj not in desc.replace(" ", "")


def test_every_confirmatory_goal_object_has_a_description():
    confirmatory = """soapbar cloth mug apple tomato potato bowl plate egg pan bread cd cup keychain cellphone book peppershaker
    vase saltshaker lettuce soapbottle creditcard pencil knife watch pot spatula spraybottle pillow alarmclock toiletpaper
    dishsponge butterknife pen fork statue spoon laptop newspaper handtowel remotecontrol candle box glassbottle""".split()
    assert set(confirmatory) <= set(rq1x.DESCRIPTIONS)


def test_object_free_lines_replace_the_one_mention_and_keep_the_template():
    t = _item()
    line = rq1x.describe_line(t["task_line"], "saltshaker")
    assert line == f"Your task is to: put some {rq1x.DESCRIPTIONS['saltshaker']} on cabinet."
    with pytest.raises(ValueError):
        rq1x.describe_line("Your task is to: put some mug on cabinet.", "saltshaker")


def test_object_free_prompts_differ_from_the_named_ones_only_in_the_goal_line(mock_score):
    t = _item()
    for arm, named, obj in (("aligned_desc", "aligned", "saltshaker"), ("swapped_desc", "swapped", "mug")):
        a, b = rq1x.arm_messages(t, arm), rq1.arm_messages(t, named)
        assert len(a) == len(b) and a[:-1] == b[:-1]
        line = [x for x in a[-1]["content"].splitlines() if x.startswith("Your task is to:")]
        assert line and obj not in line[0] and rq1x.DESCRIPTIONS[obj] in line[0]
        assert a[-1]["content"].replace(rq1x.DESCRIPTIONS[obj], obj) == b[-1]["content"]


def test_manipulation_check_reads_spaced_names():
    assert rq1x.check_says("saltshaker", "A salt shaker.") and not rq1x.check_says("saltshaker", "pepper shaker")


def test_auc_reads_the_right_direction():
    assert rq1x._auc([(True, 2.0), (False, 1.0)]) == 1.0
    assert rq1x._auc([(True, 1.0), (False, 1.0)]) == 0.5


def test_end_to_end_after_rq1(mock_root):
    assert rq1.run_rq1_score(_score_cfg(shard=0, n_shards=1)) == 0
    cfg = {"run_id": "rq1x_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 2}
    for i in range(2):
        assert rq1x.run_rq1x_score({**cfg, "shard": i}) == 0
    acfg = {"run_id": "rq1x_smoke_analysis", "source": "rq1x_smoke", "base": "rq1_smoke", "sources": ["rq12_transitions_smoke"]}
    assert rq1x.run_rq1x_analysis(acfg) == 0
    m = json.loads((mock_root / "rq1x_smoke_analysis" / "metrics.json").read_text())
    for tag in rq1.FEWSHOT:
        assert {"object_free_goal", "working_actions", "object_named_before", "manipulation_check"} <= set(m[tag])
    checks = read_jsonl(mock_root / "rq1x_smoke" / "shard_00_of_02" / "checks.jsonl")
    assert len(checks) == 2 * len(rq1x.DESCRIPTIONS)
    assert not (mock_root / "rq1x_smoke" / "shard_01_of_02" / "checks.jsonl").exists()
    # resuming a finished shard adds nothing
    before = len(read_jsonl(mock_root / "rq1x_smoke" / "shard_00_of_02" / "scoring.jsonl"))
    assert rq1x.run_rq1x_score({**cfg, "shard": 0, "resume": True}) == 0
    assert len(read_jsonl(mock_root / "rq1x_smoke" / "shard_00_of_02" / "scoring.jsonl")) == before
