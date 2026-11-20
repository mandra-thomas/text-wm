"""World-model prompts for the τ-bench retail items (notes/tau_design.md §5).

System message: instruction, the retail policy and the tool schemas. Each user turn: the task line, the tool
calls so far with their outputs, the next call and "Tool output:". Three few-shot examples precede the query.
The aligned, swapped and neutral prompts differ only in the query's task line.
"""
from __future__ import annotations

import json
from functools import lru_cache

from wf.tau.retail import VENDOR

INSTRUCTION = (
    "You are a world model of the backend of a retail customer-service system. An agent is helping a user and "
    "calls the backend's tools. Given the user's task, the tool calls made so far with their outputs, and the "
    "agent's next tool call, predict the exact output of the tool call. Successful calls return JSON; failed "
    "calls return a single line starting with \"Error: \". Output only the tool output."
)
YN_QUESTION = ("Question: will this tool call succeed, or will it return an error? Answer Yes if it succeeds and No "
               "if it returns an error.\nAnswer:")
OUTPUT_CUE = "Tool output:"
CONDITIONS = ("aligned", "swapped", "neutral")


@lru_cache(maxsize=1)
def system_message() -> str:
    policy = (VENDOR / "policy.md").read_text(encoding="utf-8").strip()
    tools = json.loads((VENDOR / "tools_openai.json").read_text(encoding="utf-8"))
    tool_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return f"{INSTRUCTION}\n\n{policy}\n\n# Tools\n\n{tool_lines}"


def render_call(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments})


def render_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n\n".join(f"Call: {render_call(h['name'], h['arguments'])}\nOutput: {h['output']}" for h in history)


def user_turn(task: str, history: list[dict], action: dict, yes_no: bool = False) -> str:
    end = YN_QUESTION if yes_no else OUTPUT_CUE
    return (f"Task: {task}\n\nTool calls so far:\n{render_history(history)}\n\n"
            f"Next call: {render_call(action['name'], action['arguments'])}\n{end}")


def task_for(item: dict, condition: str) -> str:
    line = {"aligned": item["task_aligned"], "swapped": item.get("task_swapped"), "neutral": item["task_neutral"]}[condition]
    if not line:
        raise ValueError(f"{item['id']}: no {condition} task line")  # never quietly score another condition
    return line


def build_messages(item: dict, condition: str, fewshot: list[dict], yes_no: bool = False) -> list[dict[str, str]]:
    if len(fewshot) != 3:
        raise ValueError("the design fixes three few-shot examples")
    messages = [{"role": "system", "content": system_message()}]
    for ex in fewshot:
        messages.append({"role": "user", "content": user_turn(ex["task"], ex["history"], ex["action"])})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": user_turn(task_for(item, condition), item["history"], item["action"], yes_no)})
    return messages
