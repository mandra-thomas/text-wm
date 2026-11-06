"""ReAct agent loop with foresight hooks (§4.7)."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import requests

from wf.agent.foresight import ForesightConfig, foresight_prediction
from wf.constants import AGENT_MODEL, STEP_CAP, VLLM_BASE_URL_DEFAULT
from wf.data.goals import names_object, parse_task_line
from wf.env.expert import on_expert_path
from wf.env.server import AlfworldClient
from wf.scoring.prompts import agent_react_messages, parse_react_action


class AgentClient:
    """vLLM OpenAI-compatible chat client; mock when WF_MOCK_AGENT=1."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = AGENT_MODEL,
        temperature: float = 0.0,
    ):
        self.base_url = (os.environ.get("VLLM_BASE_URL") or base_url or VLLM_BASE_URL_DEFAULT).rstrip("/")
        self.model = model
        self.temperature = temperature
        self.mock = os.environ.get("WF_MOCK_AGENT", "0") == "1"
        self._session = requests.Session()

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 256) -> str:
        if self.mock:
            return self._mock_complete(messages)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # Also pass via OpenAI extra_body style for some servers
        r = self._session.post(
            f"{self.base_url}/chat/completions",
            json={
                **payload,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            timeout=180,
        )
        # retry without nested extra_body if rejected
        if not r.ok:
            r = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=180,
            )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""

    def _mock_complete(self, messages: list[dict[str, str]]) -> str:
        """Deterministic stand-in: mostly competent, sometimes proposes an invalid action."""
        text = messages[-1]["content"] if messages else ""
        body, _, tail = text.partition("Admissible actions:")
        adm = [ln.strip()[2:] for ln in tail.splitlines() if ln.strip().startswith("- ")]
        task = re.search(r"Your task is to:.*", text)
        spec = parse_task_line(task.group(0) if task else "")
        goal, target = f"{spec.goal_object} 1", f"{spec.receptacle} 1"

        lines = body.strip().splitlines()
        shown = lines[-1].split(":", 1)[1].strip() if lines[-1].startswith("Expected outcome:") else None
        task_text = task.group(0) if task else ""
        if shown is None:
            if int(hashlib.sha256(body.encode()).hexdigest()[:2], 16) % 3 != 0:
                return f"Thought: try\nAction: take {goal} from countertop 1"
            return f"Thought: progress\nAction: {self._mock_best(adm, goal, target, task_text, body)}"

        proposal = lines[-2].split(":", 1)[1].strip()
        if shown in ("Nothing happens.", "This action is not valid."):
            best = self._mock_best(adm, goal, target, task_text, body)
            return f"Thought: avoid it\nAction: {'look' if best == proposal else best}"
        return f"Thought: go ahead\nAction: {proposal}"

    @staticmethod
    def _mock_best(adm: list[str], goal: str, target: str, task: str = "", body: str = "") -> str:
        holding = f"examine {goal}" in adm
        steps = [f"take {goal} from drawer 1", "open drawer 1"]
        go = f"go to {target}"
        if holding and "desklamp" in target:
            steps, go = ["use desklamp 1"], "go to sidetable 1"
        elif holding:
            tools = (("heat", ("heat", "hot"), "microwave 1"), ("cool", ("cool",), "fridge 1"), ("clean", ("clean",), "sinkbasin 1"))
            for verb, words, tool in tools:
                if any(f" {w} " in f" {task} " for w in words) and f"You {verb} the {goal}" not in body:
                    steps, go = [f"{verb} {goal} with {tool}"], f"go to {tool}"
                    break
            else:
                steps = [f"move {goal} to {target}"]
        for act in steps:
            if act in adm:
                return act
        act = go if holding else "go to drawer 1"
        return act if act in adm else "look"


def run_episode(
    env: AlfworldClient,
    agent: AgentClient,
    game_id: str,
    split: str,
    run_id: str,
    foresight: ForesightConfig,
    step_cap: int = STEP_CAP,
    schedule_seed: int | None = None,
    success_cache: dict | None = None,
) -> dict[str, Any]:
    if env._env_id is None:
        env.create()
    state = env.reset(game_id, split)
    goal_object = parse_task_line(state.task_line).goal_object
    prefix: list[str] = []
    history: list[dict[str, str]] = []
    trace: list[dict] = []
    n_corrupted = 0
    n_unavailable = 0
    cache = success_cache if success_cache is not None else {}

    for t in range(step_cap):
        if state.done:
            break

        # Temperature 0: an identical prompt within a step gets the same answer, so ask once.
        replies: dict[str, str] = {}

        def ask(prediction: str | None = None, proposed: str | None = None) -> str:
            msgs = agent_react_messages(
                observation=state.observation,
                history=history,
                foresight_prediction=prediction,
                proposal=proposed,
                admissible=state.admissible,
            )
            key = repr(msgs)
            if key not in replies:
                replies[key] = agent.complete(msgs)
            return parse_react_action(replies[key])

        # 1) proposal without foresight
        proposal = ask()

        # 2) foresight hook
        fs = foresight_prediction(
            env,
            state,
            proposal,
            prefix,
            t,
            foresight,
            success_cache=cache,
        )
        if fs["corrupted"]:
            n_corrupted += 1
        if fs.get("corruption_unavailable"):
            n_unavailable += 1

        # 3) decision-critical: what would the agent do given the true prediction?
        action_under_true = proposal
        if foresight.condition == "validity_only":
            verdict = "This action is valid." if fs["proposal_valid"] else "This action is not valid."
            action_under_true = ask(verdict, proposal)
        elif foresight.condition != "none":
            action_under_true = ask(fs["oracle_obs"], proposal)

        # 4) decide with shown prediction; invalid choices are executed and fail in the env
        shown = fs["prediction_shown"]
        executed = proposal if shown is None else ask(shown, proposal)

        new_state = env.step(state, executed)
        real_obs = new_state.observation

        step_rec = {
            "t": t,
            "proposal": proposal,
            "proposal_valid": fs["proposal_valid"],
            "prediction_shown": shown,
            "prediction_type": fs["prediction_type"],
            "corruption_unavailable": bool(fs.get("corruption_unavailable")),
            "action_under_true_prediction": action_under_true,
            "executed": executed,
            "real_obs": real_obs,
            "decision_critical": action_under_true != proposal,
            "followed": _followed(fs, executed, proposal),
            "on_expert_path": on_expert_path(env, state, executed),
            "proposal_on_expert_path": on_expert_path(env, state, proposal),
            "targets_goal_object": names_object(executed, goal_object),
        }
        trace.append(step_rec)

        # Retention differs only in whether the prediction stays in the history.
        turn = {"observation": state.observation, "action": executed}
        if foresight.retention == "retained" and shown is not None:
            turn.update(proposal=proposal, expected_outcome=shown)
        history.append(turn)

        prefix.append(executed)
        state = new_state

    return {
        "run_id": run_id,
        "game_id": game_id,
        "split": split,
        "condition": foresight.condition,
        "arm": foresight.arm,
        "retention": foresight.retention,
        "schedule_seed": schedule_seed,
        "p_arm": foresight.p_arm,
        "steps": len(trace),
        "success": bool(state.won),
        "n_corrupted": n_corrupted,
        "n_corruption_unavailable": n_unavailable,
        "trace": trace,
    }


def _followed(fs: dict, executed: str, proposal: str) -> bool:
    shown = fs.get("prediction_shown")
    if shown is None:
        return executed == proposal
    if shown in ("This action is valid.",) or (
        shown != "Nothing happens." and shown != "This action is not valid."
    ):
        # prediction said it works -> followed if executed proposal
        return executed == proposal
    # prediction said fail -> followed if did NOT execute proposal
    return executed != proposal
