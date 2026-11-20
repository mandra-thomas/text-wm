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


def test_each_condition_shows_its_own_goal_line(items):
    """The goal line is read off the rendered query, not through P.task_line (a wrong map would pass a diff test)."""
    for it in items[::50]:
        for arm, field in (("aligned", "task_line"), ("swapped", "swapped_task_line"), ("neutral", "neutral_task_line")):
            query = P.messages(it, arm)[-1]["content"]
            assert [ln for ln in query.split("\n") if ln.startswith("Goal: ")] == [it[field]]
        neutral = P.messages(it, "neutral")[-1]["content"]
        assert "Goal: craft" not in neutral and "Goal: explore the crafting commands." in neutral


def test_prefix_arms_share_the_prompt_and_drop_the_object_name(items):
    it = next(t for t in items if t["action_kind"] == "craft")
    for c in ("aligned", "swapped"):
        assert P.messages(it, f"prefix_{c}") == P.messages(it, c)
    assert P.candidates(it, "prefix_aligned") == (it["prefix_candidates"]["o_plus"], it["prefix_candidates"]["o_minus"])
    plus, minus = P.candidates(it, "prefix_swapped")
    assert minus == "Could not find enough items to craft" and plus.startswith("Crafted ") and len(plus.split()) == 2
    assert it["object"] not in plus + minus and W.item_id(it["object"]) not in plus + minus
    get = next(t for t in items if t["action_kind"] == "get")
    assert P.candidates(get, "prefix_aligned") == ("Got 1", "Could not find")
    assert set(P.ARMS) == {"aligned", "swapped", "neutral", "yn_aligned", "yn_swapped", "yn_neutral",
                           "prefix_aligned", "prefix_swapped"}


def test_verify_reports_a_broken_prompt(items, monkeypatch):
    """Negative test: a prompt that differs beyond the goal line must be reported."""
    shots = P.fewshot_examples()
    it = items[0]
    real = P.messages

    def broken(item, arm, fewshot=None):
        out = real(item, arm, fewshot)
        if arm == "swapped":
            out[-1] = {**out[-1], "content": out[-1]["content"].replace("Action: ", "Action:  ", 1)}
        return out
    monkeypatch.setattr(P, "messages", broken)
    assert any("swapped" in p for p in V.prompt_problems(it, shots))
    # every arm rendering the aligned line (questions and prefixes kept consistent) is caught by the goal-line check
    monkeypatch.setattr(P, "messages", lambda item, arm, fewshot=None: real(
        item, arm.replace("swapped", "aligned").replace("neutral", "aligned"), fewshot))
    assert V.prompt_problems(it, shots) == ["swapped: the query's goal line is not the item's swapped_task_line",
                                            "neutral: the query's goal line is not the item's neutral_task_line"]
