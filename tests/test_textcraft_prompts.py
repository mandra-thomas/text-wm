"""TextCraft world-model prompts (wf/textcraft/prompts.py): conditions differ only in the goal line."""
import pytest

from wf.textcraft import prompts as P
from wf.textcraft import verify as V
from wf.textcraft import world as W


@pytest.fixture(scope="module")
def items():
    return V.read_items()


def test_instruction_is_agentgyms_without_the_output_format():
    assert P.AGENT_FORMAT_SENTENCE in P.AGENTGYM_INSTRUCTION
    assert P.INSTRUCTION == P.AGENTGYM_INSTRUCTION.replace(P.AGENT_FORMAT_SENTENCE, "").strip()
    assert "Thought" not in P.INSTRUCTION and P.INSTRUCTION.startswith("You are given few useful crafting recipes")
    assert P.SYSTEM.endswith(P.INSTRUCTION)


def test_first_observation_is_the_environments(items):
    for it in items[::200]:
        _env, obs = W.new_env(it["recipe_lines"], it["env_goal"])
        assert P.first_observation(it["recipe_lines"], it["task_line"]) == obs


def test_history_is_the_transcript():
    text = P.history_text(["craft 4 stick using 2 planks"], [["get 1 oak logs", "Got 1 oak logs"]], "Goal: craft stick.")
    assert text == "Crafting commands:\ncraft 4 stick using 2 planks\n\nGoal: craft stick.\n> get 1 oak logs\nGot 1 oak logs"


def test_query_turn_format(items):
    it = items[0]
    m = P.messages(it, "aligned")
    assert m[0] == {"role": "system", "content": P.SYSTEM}
    assert [x["role"] for x in m] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert m[-1]["content"].startswith("History:\nCrafting commands:\n")
    assert m[-1]["content"].endswith(f"\n\nAction: {it['action']}\n\nNext observation:")
    assert f"\n\n{it['task_line']}" in m[-1]["content"]


def test_conditions_differ_only_in_the_goal_line(items):
    shots = P.fewshot_examples()
    for it in items:
        assert V.prompt_problems(it, shots) == [], it["id"]
    it = items[0]
    a, s, n = (P.messages(it, c)[-1]["content"] for c in P.CONDITIONS)
    assert a.replace(it["task_line"], it["swapped_task_line"]) == s
    assert a.replace(it["task_line"], it["neutral_task_line"]) == n
    assert it["goal"] not in n.split("Crafting commands:")[0]


def test_neutral_line_names_no_item(items):
    assert {it["neutral_task_line"] for it in items} == {"Goal: explore the crafting commands."}


def test_yes_no_arms():
    it = V.read_items()[0]
    for c in P.CONDITIONS:
        q, plain = P.messages(it, f"yn_{c}"), P.messages(it, c)
        assert q[:-1] == plain[:-1] and q[-1]["content"].endswith("\n\nQuestion: Will this action succeed? Answer Yes or No.\nAnswer:")
        assert P.candidates(it, f"yn_{c}") == ("Yes", "No")
        assert P.candidates(it, c) == (it["twin"], it["reply"])
    with pytest.raises(ValueError):
        P.messages(it, "absent")


def test_examples_use_their_own_goal_lines():
    m = P.messages(V.read_items()[0], "aligned")
    for ex, user in zip(P.fewshot_examples(), m[1:-1:2]):
        assert f"\n\nGoal: craft {ex['goal']}.\n" in user["content"]
