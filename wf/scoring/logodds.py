"""Teacher-forced log-prob scoring (PILOT_PROTOCOL §4.5)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from wf.data.goals import make_swapped_task_line
from wf.env.server import parse_game_id
from wf.scoring.prompts import (
    _replace_task_line,
    condition_task_line,
    world_model_ft_messages,
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
    ):
        self.model_name = model_name
        self.model_tag = model_tag or model_name.split("/")[-1].lower()
        self.mock = os.environ.get("WF_MOCK_SCORE", "0") == "1"
        self._cache: dict[tuple[str, str, str, str], ScoreResult] = {}
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

    def _encode(self, messages: list[dict[str, str]]) -> Any:
        return self._tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )

    def _teacher_force_logprob(self, messages: list[dict[str, str]], candidate: str) -> tuple[float, int]:
        if self.mock:
            n_tok = max(1, len(candidate.split()))
            prefer_success = self._mock_prefer_success(messages)
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
        prompt_ids = self._encode(messages)
        cand_ids = self._tok(candidate, return_tensors="pt", add_special_tokens=False).input_ids
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
        messages: list[dict[str, str]],
        o_plus: str,
        o_minus: str,
    ) -> ScoreResult:
        content = hashlib.sha256(json.dumps([messages, o_plus, o_minus], sort_keys=True).encode()).hexdigest()
        key = (self.model_tag, condition, transition_id, content)
        if key in self._cache:
            return self._cache[key]

        logp_plus, n_plus = self._teacher_force_logprob(messages, o_plus)
        logp_minus, n_minus = self._teacher_force_logprob(messages, o_minus)
        L = logp_plus - logp_minus
        if not math.isfinite(L):
            raise ValueError(f"non-finite L for {transition_id}/{condition} on {self.model_tag}: {L}")
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

    def _stop_tokens(self) -> list[int]:
        """Every end-of-turn token. Fine-tuned checkpoints leave the tokenizer's out of their generation config."""
        configured = self._model.generation_config.eos_token_id
        tokens = [self._tok.eos_token_id, self._tok.pad_token_id]
        tokens += configured if isinstance(configured, list) else [configured]
        return sorted({t for t in tokens if t is not None})

    def free_generate(self, messages: list[dict[str, str]], max_new_tokens: int = 64) -> str:
        if self.mock:
            if self._mock_prefer_success(messages):
                return "You pick up the mug 1 from the drawer 1."
            return "Nothing happens."

        import torch

        assert self._model is not None and self._tok is not None
        input_ids = self._encode(messages)
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                eos_token_id=self._stop_tokens(),
            )
        gen = out[0, input_ids.shape[1] :]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


_FRAMING = {
    "committed": "(The agent is committed to this goal.)",
    "candidate": "(Treat the goal as one candidate among others.)",
}


def _init_contexts_dir() -> Path:
    root = os.environ.get("WF_INIT_CONTEXTS")
    if root:
        return Path(root)
    return Path(os.environ.get("WORD2WORLD_ROOT", "third_party/Word2World")) / "data" / "init_contexts" / "alfworld"


@lru_cache(maxsize=None)
def _wm_contexts(split: str) -> dict[int, str]:
    """Keyed by each row's `id`, the game's global index. Rows have gaps, so position means nothing."""
    name = {"valid_train": "test"}.get(split, split)
    path = _init_contexts_dir() / f"wm_instruct_{name}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["id"]): row["messages"][0]["content"] for row in rows}


def world_model_context(transition: dict) -> str:
    """Hidden state + first observation for this game, the checkpoints' trained input."""
    if os.environ.get("WF_MOCK_SCORE", "0") == "1":
        return f"# Environment Information (Only visible to Assistant)\n\n{transition['task_line']}\n"
    split, global_index = parse_game_id(transition["game_id"])
    context = _wm_contexts(split).get(global_index)
    if context is None:
        raise KeyError(f"no world-model context for {transition['game_id']} (global index {global_index})")
    # _replace_task_line would hide another game's hidden state, so prove it is this game's.
    if transition["task_line"].strip() not in context:
        raise ValueError(f"world-model context for {transition['game_id']} is for a different task")
    return context


def build_messages(transition: dict, condition: str, ft_prompt: bool = False) -> list[dict[str, str]]:
    """The prompt for one item under one condition; only the task line and framing vary."""
    aligned, swapped = transition["task_line"], _swapped_task(transition)
    task = condition_task_line(condition, aligned, swapped)
    note = _FRAMING.get(condition)
    if ft_prompt:
        system = _replace_task_line(world_model_context(transition), task)
        if note:
            system = f"{system}{note}\n"
        return world_model_ft_messages(system, transition.get("history_pairs") or [], transition["action"])
    messages = world_model_prompt(transition["history_text"], transition["action"], task_line_override=task)
    if note:
        last = messages[-1]
        messages[-1] = {**last, "content": last["content"].replace("Next observation:", f"{note}\n\nNext observation:")}
    return messages


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
        messages=build_messages(transition, condition, ft_prompt=ft_prompt),
        o_plus=cands["o_plus"],
        o_minus=cands["o_minus"],
    )


def _swapped_task(transition: dict) -> str | None:
    if transition.get("swapped_task_line"):
        return transition["swapped_task_line"]
    swapped = transition.get("swapped_goal_object")
    if not swapped:
        return None
    return make_swapped_task_line(transition["task_line"], transition["goal_object"], swapped)
