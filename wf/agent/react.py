"""ReAct agent loop with foresight hooks (§4.7)."""
from __future__ import annotations

import os
import re
from typing import Any

import requests

from wf.agent.foresight import ForesightConfig, foresight_prediction
from wf.constants import AGENT_MODEL, STEP_CAP, VLLM_BASE_URL_DEFAULT
from wf.env.expert import on_expert_path
from wf.env.server import AlfworldClient, GameState
from wf.scoring.prompts import agent_react_messages, parse_react_action


class AgentClient:
    """vLLM OpenAI-compatible chat client; mock when WF_MOCK_AGENT=1."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = AGENT_MODEL,
        temperature: float = 0.0,
    ):
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or VLLM_BASE_URL_DEFAULT).rstrip("/")
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
        text = messages[-1]["content"] if messages else ""
        # Prefer admissible list order for a simple policy
        adm = []
        if "Admissible actions:" in text:
            part = text.split("Admissible actions:", 1)[1]
            for line in part.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    adm.append(line[2:].strip())
        foresight = None
        m = re.search(r"Expected outcome:\s*(.+)", text)
        if m:
            foresight = m.group(1).strip()

        # If foresight says not valid / Nothing happens, pick a different action
        if foresight and (
            foresight == "Nothing happens."
            or foresight == "This action is not valid."
        ):
            for a in adm:
                if a.startswith("go to") or a == "open drawer 1":
                    return f"Thought: foresight says fail\nAction: {a}"

        for preferred in (
            "open drawer 1",
            "take mug 1 from drawer 1",
            "go to drawer 1",
            "go to coffeemachine 1",
            "move mug 1 to coffeemachine 1",
            "look",
        ):
            if preferred in adm:
                return f"Thought: progress\nAction: {preferred}"
        if adm:
            return f"Thought: try\nAction: {adm[0]}"
        return "Thought: fallback\nAction: look"


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
    if env._env_id is None and not env.mock:
        env.create()
    state = env.reset(game_id, split)
    prefix: list[str] = []
    retained_history: list[dict[str, str]] = []
    trace: list[dict] = []
    n_corrupted = 0
    cache = success_cache if success_cache is not None else {}

    for t in range(step_cap):
        if state.done:
            break

        # 1) proposal without foresight
        msgs = agent_react_messages(
            observation=state.observation,
            history=retained_history if foresight.retention == "retained" else None,
            foresight_prediction=None,
            admissible=state.admissible,
        )
        raw = agent.complete(msgs)
        proposal = parse_react_action(raw)

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

        # 3) decision-critical: extra call with oracle prediction at corrupted steps
        action_under_true = proposal
        if fs["corrupted"] or foresight.condition in ("oracle", "validity_only", "corrupted"):
            msgs_true = agent_react_messages(
                observation=state.observation,
                history=retained_history if foresight.retention == "retained" else None,
                foresight_prediction=fs["oracle_obs"]
                if foresight.condition != "validity_only"
                else (
                    "This action is valid."
                    if fs["proposal_valid"]
                    else "This action is not valid."
                ),
                admissible=state.admissible,
            )
            # Only required at corrupted steps per protocol; still useful for oracle
            if fs["corrupted"] or foresight.condition != "none":
                action_under_true = parse_react_action(agent.complete(msgs_true))

        # 4) decide with shown prediction
        msgs_dec = agent_react_messages(
            observation=state.observation,
            history=retained_history if foresight.retention == "retained" else None,
            foresight_prediction=fs["prediction_shown"],
            admissible=state.admissible,
        )
        executed = parse_react_action(agent.complete(msgs_dec))
        # Prefer admissible if model drifts
        if state.admissible and executed not in state.admissible:
            # soft fallback to proposal if admissible
            if proposal in state.admissible:
                executed = proposal
            else:
                executed = state.admissible[0]

        new_state = env.step(state, executed)
        real_obs = new_state.observation

        decision_critical = action_under_true != proposal
        followed = _followed(fs, executed, proposal)

        step_rec = {
            "t": t,
            "proposal": proposal,
            "proposal_valid": fs["proposal_valid"],
            "prediction_shown": fs["prediction_shown"],
            "prediction_type": fs["prediction_type"],
            "action_under_true_prediction": action_under_true,
            "executed": executed,
            "real_obs": real_obs,
            "decision_critical": decision_critical,
            "followed": followed,
            "on_expert_path": on_expert_path(env, state, executed),
            "targets_goal_object": state.task_line.split()[-1].rstrip(".").lower() in executed.lower()
            if state.task_line
            else False,
        }
        trace.append(step_rec)

        # retention: discarded removes prediction from history after execution
        if foresight.retention == "retained" and fs["prediction_shown"] is not None:
            retained_history.append(
                {
                    "observation": state.observation,
                    "expected_outcome": fs["prediction_shown"],
                    "action": executed,
                }
            )
            retained_history.append({"observation": real_obs})
        else:
            # still keep observations for context if discarded? protocol: prediction removed;
            # agent still sees real env obs via state.observation next loop
            pass

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
        "steps": len(trace),
        "success": bool(state.won),
        "n_corrupted": n_corrupted,
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
