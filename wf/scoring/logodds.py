"""Teacher-forced log-prob scoring (PILOT_PROTOCOL §4.5)."""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any

from wf.scoring.prompts import (
    condition_task_line,
    world_model_ft_prompt,
    world_model_prompt,
)
from wf.util import sigmoid


@dataclass
class ScoreResult:
    transition_id: str
    model: str
    condition: str
    logp_plus: float
    logp_minus: float
    L: float
    p_success: float
    n_tokens_plus: int
    n_tokens_minus: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "model": self.model,
            "condition": self.condition,
            "logp_plus": self.logp_plus,
            "logp_minus": self.logp_minus,
            "L": self.L,
            "p_success": self.p_success,
            "n_tokens_plus": self.n_tokens_plus,
            "n_tokens_minus": self.n_tokens_minus,
        }


class LogoddsScorer:
    """HF transformers teacher-forced scorer; mock mode when WF_MOCK_SCORE=1."""

    def __init__(
        self,
        model_name: str,
        model_tag: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        use_chat_template: bool = True,
    ):
        self.model_name = model_name
        self.model_tag = model_tag or model_name.split("/")[-1].lower()
        self.use_chat_template = use_chat_template
        self.mock = os.environ.get("WF_MOCK_SCORE", "0") == "1"
        self._cache: dict[tuple[str, str, str], ScoreResult] = {}
        self._model = None
        self._tok = None
        self.device = device
        self.dtype = dtype
        if not self.mock:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def _mock_prefer_success(self, prompt_messages_or_text: Any) -> bool:
        """Shared mock bias so free_generate and sign(L) agree (Tier 0.6)."""
        h = hashlib.sha256(repr(prompt_messages_or_text).encode()).hexdigest()
        return int(h[:2], 16) % 2 == 0

    def _teacher_force_logprob(self, prompt_messages_or_text: Any, candidate: str) -> tuple[float, int]:
        if self.mock:
            n_tok = max(1, len(candidate.split()))
            prefer_success = self._mock_prefer_success(prompt_messages_or_text)
            is_noop = candidate.strip() == "Nothing happens."
            # Prefer success ⇒ higher logp for non-noop candidates
            if prefer_success:
                logp = -2.0 if not is_noop else -12.0
            else:
                logp = -12.0 if not is_noop else -2.0
            # tiny stable jitter from candidate text
            jitter = int(hashlib.sha256(candidate.encode()).hexdigest()[:4], 16) / 0xFFFF
            return logp - 0.01 * jitter, n_tok

        import torch

        assert self._model is not None and self._tok is not None
        tok = self._tok
        if self.use_chat_template and isinstance(prompt_messages_or_text, list):
            try:
                prompt_ids = tok.apply_chat_template(
                    prompt_messages_or_text,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            except TypeError:
                prompt_ids = tok.apply_chat_template(
                    prompt_messages_or_text,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
        else:
            text = prompt_messages_or_text if isinstance(prompt_messages_or_text, str) else str(prompt_messages_or_text)
            prompt_ids = tok(text, return_tensors="pt").input_ids

        cand_ids = tok(candidate, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = torch.cat([prompt_ids, cand_ids], dim=1)
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids)
            logits = out.logits
        # log p of candidate tokens only
        prompt_len = prompt_ids.shape[1]
        logprobs = []
        for i, tid in enumerate(cand_ids[0].tolist()):
            pos = prompt_len + i - 1
            if pos < 0:
                continue
            lp = torch.log_softmax(logits[0, pos], dim=-1)[tid].item()
            logprobs.append(lp)
        return float(sum(logprobs)), len(logprobs)

    def score_pair(
        self,
        transition_id: str,
        condition: str,
        history_text: str,
        action: str,
        o_plus: str,
        o_minus: str,
        aligned_task: str,
        swapped_task: str | None = None,
        ft_prompt: bool = False,
    ) -> ScoreResult:
        key = (self.model_tag, condition, transition_id)
        if key in self._cache:
            return self._cache[key]

        task_override = condition_task_line(condition, aligned_task, swapped_task)
        if ft_prompt:
            prompt = world_model_ft_prompt(
                _apply_task(history_text, task_override),
                action,
            )
        else:
            prompt = world_model_prompt(
                history_text,
                action,
                task_line_override=task_override,
            )

        # committed / candidate framing: lightly annotate (exploratory in Pilot A)
        if condition == "committed" and isinstance(prompt, list):
            prompt = list(prompt)
            prompt[-1] = {
                **prompt[-1],
                "content": prompt[-1]["content"] + "\n(The agent is committed to this goal.)",
            }
        if condition == "candidate" and isinstance(prompt, list):
            prompt = list(prompt)
            prompt[-1] = {
                **prompt[-1],
                "content": prompt[-1]["content"] + "\n(Treat the goal as one candidate among others.)",
            }

        logp_plus, n_plus = self._teacher_force_logprob(prompt, o_plus)
        logp_minus, n_minus = self._teacher_force_logprob(prompt, o_minus)
        L = logp_plus - logp_minus
        result = ScoreResult(
            transition_id=transition_id,
            model=self.model_tag,
            condition=condition,
            logp_plus=logp_plus,
            logp_minus=logp_minus,
            L=L,
            p_success=sigmoid(L),
            n_tokens_plus=n_plus,
            n_tokens_minus=n_minus,
        )
        self._cache[key] = result
        return result

    def free_generate(self, history_text: str, action: str, max_new_tokens: int = 64) -> str:
        if self.mock:
            prompt = world_model_prompt(history_text, action)
            if self._mock_prefer_success(prompt):
                return "You pick up the mug 1 from the drawer 1."
            return "Nothing happens."

        import torch

        assert self._model is not None and self._tok is not None
        messages = world_model_prompt(history_text, action)
        try:
            input_ids = self._tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            input_ids = self._tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )
        gen = out[0, input_ids.shape[1] :]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


def _apply_task(history_text: str, task_override: str | None) -> str:
    if not task_override:
        return history_text
    from wf.scoring.prompts import _replace_task_line

    return _replace_task_line(history_text, task_override)


def score_transition(
    scorer: LogoddsScorer,
    transition: dict,
    condition: str,
    ft_prompt: bool = False,
) -> ScoreResult | None:
    cands = transition.get("candidates")
    if not cands:
        return None
    return scorer.score_pair(
        transition_id=transition["id"],
        condition=condition,
        history_text=transition["history_text"],
        action=transition["action"],
        o_plus=cands["o_plus"],
        o_minus=cands["o_minus"],
        aligned_task=transition["task_line"],
        swapped_task=_swapped_task(transition),
        ft_prompt=ft_prompt,
    )


def _swapped_task(transition: dict) -> str:
    from wf.scoring.prompts import task_line_for

    obj = transition.get("swapped_goal_object") or "apple"
    return task_line_for(obj)
