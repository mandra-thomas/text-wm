"""World-model prompts for the τ-bench retail items (notes/tau_design.md §5, as amended 2026-11-20 after reviews 1
and 2).

System message (main arm): instruction, how the backend checks write calls, the retail policy and the tool schemas.
The rules-only arm (exploratory) drops the agent-facing policy and keeps the backend's rules. The ALFWorld-matched
arm (exploratory, amendment after review 2) is the main arm without the statement of what the backend sees, so
nothing says that it does not see the task, and its few-shot examples all name their own order. Each user turn: the
task line, the tool calls so far with their outputs, the next call and a cue. Four few-shot examples precede the
query, in the design's order or reversed by a hash of the item id (one order for every job of an item). Every
condition of one item differs from the others only in the query's task line; the action (the failing call or its
success witness) and the question (output cue, yes/no, or the manipulation check) are fixed per job.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from wf.tau.retail import VENDOR

INSTRUCTION = (
    "You are a world model of the backend of a retail customer-service system. An agent is helping a user and "
    "calls the backend's tools. Given the user's task, the tool calls made so far with their outputs, and the "
    "agent's next tool call, predict the exact output of the tool call. Successful calls return JSON; failed "
    "calls return a single line starting with \"Error: \". Output only the tool output."
)

# The backend's enforced preconditions, exactly as the tools check them (design §7.2). Identical in every condition.
_BACKEND_LEAD = {
    "main": ("The policy below is written for the agent. The backend sees only each call's name and arguments and the "
             "current database: it does not see the user's messages or task, and it does not check the agent's conduct "
             "(authentication, confirmation with the user). It does check the following, and a write call that breaks "
             "any of them returns an error and changes nothing."),
    "rules_only": ("The backend sees only each call's name and arguments and the current database: it does not see the "
                   "user's messages or task. It checks the following, and a write call that breaks any of them returns "
                   "an error and changes nothing."),
    # ALFWorld-matched: the statement of what the backend sees, explicit ("it does not see the user's messages or
    # task") and implied ("sees only each call's name and arguments and the current database"), is dropped; the rest
    # of the main lead is kept word for word, including the conduct clause.
    "matched": ("The policy below is written for the agent. The backend does not check the agent's conduct "
                "(authentication, confirmation with the user). It does check the following, and a write call that "
                "breaks any of them returns an error and changes nothing."),
}
BACKEND_RULES = """Every order write needs the order to exist, and a payment method must be one of the order's user's.
- cancel_pending_order: the order's status is exactly "pending", and the reason is "no longer needed" or "ordered by mistake".
- modify_pending_order_address: the order's status contains "pending" (so "pending (item modified)" is accepted).
- modify_pending_order_payment: the order's status contains "pending"; the payment method is one of the user's; the order's payment_history has exactly one entry, and it is a payment; the new method differs from that payment's method; a gift card's balance is at least that payment's amount.
- modify_pending_order_items: the order's status is exactly "pending"; every old item is in the order; there are as many new items as old ones; each new item differs from the old item it replaces, is a variant of the same product, and is available; the payment method is one of the user's; a gift card's balance is at least the total price difference.
- exchange_delivered_order_items: the order's status is exactly "delivered"; every old item is in the order; there are as many new items as old ones; each new item is a variant of the same product and is available (the backend accepts the old item itself as its new item, although the policy asks for a different one); the payment method is one of the user's; a gift card's balance is at least the price difference.
- return_delivered_order_items: the order's status is exactly "delivered"; the refund method is one of the user's and is either the order's original payment method (the first entry of its payment_history) or a gift card; every item is in the order.

A successful order write returns the updated order as JSON. Afterwards its status is "cancelled" (cancel), "pending (item modified)" (modify items), "exchange requested" (exchange) or "return requested" (return); address and payment changes keep the status."""

YN_QUESTION = ("Question: will this tool call succeed, or will it return an error? Answer Yes if it succeeds and No "
               "if it returns an error.\nAnswer:")
# Manipulation check (amendment of 2026-11-20): is the task line read at all in a ~10k-token prompt?
TARGET_QUESTION = ("Question: is this call on the order the user asked about? Answer Yes if it is and No if it is not."
                   "\nAnswer:")
OUTPUT_CUE = "Tool output:"
CONDITIONS = ("aligned", "swapped", "neutral")
MENTION_CONDITIONS = ("mention_aligned", "mention_swapped")
ALL_CONDITIONS = CONDITIONS + MENTION_CONDITIONS
ARMS = ("main", "rules_only", "matched")
CUES = {"output": OUTPUT_CUE, "yn": YN_QUESTION, "target": TARGET_QUESTION}
N_FEWSHOT = 4
SEED = 20260924  # wf.constants.RNG_SEED; the same hash as wf.tau.items.h (items imports this module, not the reverse)


def fewshot_reversed(item_id: str) -> bool:
    """Few-shot order counterbalanced by item (amendment after review 2): reversed when h(item id, "fewshot-order")
    is odd, about half of the items. Every job of one item (arm, action, readout, task line) uses the same order."""
    return int(hashlib.sha256(f"{SEED}|{item_id}|fewshot-order".encode()).hexdigest()[:16], 16) % 2 == 1


def fewshot_for(item: dict, fewshot: list[dict], arm: str = "main") -> list[dict]:
    """The item's few-shot examples: in the matched arm every example's task names its own order (`task_own`; the
    calls and outputs are unchanged); reversed when fewshot_reversed(item id)."""
    if len(fewshot) != N_FEWSHOT:
        raise ValueError(f"the design fixes {N_FEWSHOT} few-shot examples")
    exs = [{**ex, "task": ex["task_own"]} for ex in fewshot] if arm == "matched" else list(fewshot)
    return exs[::-1] if fewshot_reversed(item["id"]) else exs


@lru_cache(maxsize=4)
def system_message(arm: str = "main") -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    tools = json.loads((VENDOR / "tools_openai.json").read_text(encoding="utf-8"))
    tool_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    backend = f"# How the backend checks write calls\n\n{_BACKEND_LEAD[arm]}\n\n{BACKEND_RULES}"
    if arm == "rules_only":
        return f"{INSTRUCTION}\n\n{backend}\n\n# Tools\n\n{tool_lines}"
    policy = (VENDOR / "policy.md").read_text(encoding="utf-8").strip()
    return f"{INSTRUCTION}\n\n{backend}\n\n{policy}\n\n# Tools\n\n{tool_lines}"


def render_call(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments})


def render_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n\n".join(f"Call: {render_call(h['name'], h['arguments'])}\nOutput: {h['output']}" for h in history)


def user_turn(task: str, history: list[dict], action: dict, yes_no: bool = False, cue: str | None = None) -> str:
    end = CUES[cue] if cue else (YN_QUESTION if yes_no else OUTPUT_CUE)
    return (f"Task: {task}\n\nTool calls so far:\n{render_history(history)}\n\n"
            f"Next call: {render_call(action['name'], action['arguments'])}\n{end}")


def task_for(item: dict, condition: str) -> str:
    lines = {"aligned": item["task_aligned"], "swapped": item.get("task_swapped"), "neutral": item["task_neutral"],
             "mention_aligned": item.get("task_mention_aligned"), "mention_swapped": item.get("task_mention_swapped")}
    line = lines[condition]
    if not line:
        raise ValueError(f"{item['id']}: no {condition} task line")  # never quietly score another condition
    return line


def action_for(item: dict, action: str) -> dict:
    """The scored call: the item's failing write, or its recorded success witness on the same state."""
    if action == "fail":
        return item["action"]
    if action == "witness":
        if not item.get("aligned_witness"):
            raise ValueError(f"{item['id']}: no witness")
        return item["aligned_witness"]
    raise ValueError(action)


def build_messages(item: dict, condition: str, fewshot: list[dict], yes_no: bool = False, *, cue: str | None = None,
                   action: str = "fail", arm: str = "main") -> list[dict[str, str]]:
    examples = fewshot_for(item, fewshot, arm)
    messages = [{"role": "system", "content": system_message(arm)}]
    for ex in examples:
        messages.append({"role": "user", "content": user_turn(ex["task"], ex["history"], ex["action"])})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": user_turn(task_for(item, condition), item["history"],
                                                          action_for(item, action), yes_no, cue)})
    return messages
