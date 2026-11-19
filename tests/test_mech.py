"""Mechanism diagnostic (wf/mech.py): facts, rules, state/rule/precondition questions and the agent's preference."""
import json

from wf import mech, rq1
from wf.data.transitions import history_facts
from tests.test_rq1 import _item, mock_root, mock_score  # noqa: F401


def test_facts_come_from_the_history_only():
    t = _item()
    text = mech.facts_text(history_facts(t["history_text"]))
    assert "The agent is at the shelf 1." in text and "The agent is holding nothing." in text
    assert "saltshaker 1" in text  # seen on the shelf
    assert "cabinet 1" not in text  # never visited, so nothing is said about it


def test_state_questions_ask_the_deciding_fact_with_a_yes_twin():
    t = _item(determination_reason="away_from_receptacle")
    no, yes = mech.state_questions(t)
    assert no == "Before this action, is the agent at the cabinet 1?" and yes == "Before this action, is the agent at the shelf 1?"
    t = _item(determination_reason="contents_or_open_state", action="take saltshaker 2 from shelf 1")
    no, yes = mech.state_questions(t)
    assert "was the saltshaker 2 there" in no and "was the saltshaker 1 there" in yes
    assert mech.state_questions(_item(action="go to cabinet 1")) is None


def test_rule_questions_cover_each_reason():
    for reason, act in (("away_from_receptacle", "take saltshaker 1 from cabinet 1"),
                        ("hands_full", "take saltshaker 1 from cabinet 1"),
                        ("not_holding_object", "heat saltshaker 1 with microwave 1"),
                        ("contents_or_open_state", "take saltshaker 2 from shelf 1")):
        no, yes = mech.rule_questions(_item(determination_reason=reason, action=act))
        assert no.startswith("In this environment, can the agent") and yes.startswith("In this environment, can the agent")
        assert no != yes


def test_interventions_change_only_the_query_turn_and_keep_the_pair_matched(mock_score):
    t = _item(determination_reason="away_from_receptacle")
    plain_a, plain_s = mech.arm_messages(t, "pred_plain_aligned"), mech.arm_messages(t, "pred_plain_swapped")
    assert plain_a == rq1.arm_messages(t, "aligned")
    for v in ("facts", "rules", "both"):
        a, s = mech.arm_messages(t, f"pred_{v}_aligned"), mech.arm_messages(t, f"pred_{v}_swapped")
        assert a[:-1] == plain_a[:-1] and a[-1]["content"].endswith("Next observation:")
        assert a[-1]["content"].replace(t["task_line"], t["swapped_task_line"]) == s[-1]["content"]
    assert mech.RULES in mech.arm_messages(t, "pred_both_aligned")[-1]["content"]
    q = mech.arm_messages(t, "state_absent")[-1]["content"]
    assert q.endswith("Answer Yes or No.\nAnswer:") and rq1.ABSENT_LINE in q


def test_policy_prompt_rebuilds_the_agent_turns_with_the_arm_goal():
    t = _item()
    m = mech.policy_messages(t, "swapped")[-1]["content"]
    assert t["swapped_task_line"] in m and t["task_line"] not in m
    assert "Action: go to shelf 1" in m and m.endswith("Respond with the Action line only.")
    assert m.count("Observation:") == 2


def test_rule_sample_is_fixed_and_independent_of_order():
    items = [_item(id=f"t{k}") for k in range(30)]
    assert mech.rule_sample(items) == mech.rule_sample(list(reversed(items)))


def test_spearman():
    assert abs(mech.spearman([(1, 2), (2, 4), (3, 9)]) - 1) < 1e-9
    assert abs(mech.spearman([(1, 3), (2, 2), (3, 1)]) + 1) < 1e-9


def test_end_to_end(mock_root):
    cfg = {"run_id": "mech_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 2}
    for i in range(2):
        assert mech.run_mech_score({**cfg, "shard": i}) == 0
    assert mech.run_mech_analysis({"run_id": "mech_smoke_analysis", "source": "mech_smoke",
                                   "sources": ["rq12_transitions_smoke"]}) == 0
    m = json.loads((mock_root / "mech_smoke_analysis" / "metrics.json").read_text())
    for tag in rq1.FEWSHOT:
        assert {"interventions", "questions", "split_by_knowledge", "policy"} <= set(m[tag])


def test_rules_and_yes_keys_match_alfworld_pddl():
    # Acting works from any receptacle at the same spot; putting into a closed receptacle fails.
    assert "same spot" in mech.RULES and "or put one into, a closed receptacle" in mech.RULES
    _, yes = mech.rule_questions(_item(determination_reason="not_holding_object", action="move saltshaker 1 to cabinet 1"))
    assert "open" in yes and "can hold that object" in yes
    no, yes = mech.rule_questions(_item(determination_reason="away_from_receptacle", action="heat saltshaker 1 with microwave 1"))
    assert "another part of the room" in no and "can be heated" in yes
