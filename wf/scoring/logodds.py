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

from wf.constants import REVISIONS, revision_for
from wf.data.goals import make_swapped_task_line
from wf.env.server import parse_game_id
from wf.scoring.prompts import (
    WM_FEWSHOT_GAMES,
    WM_FEWSHOT_SETS,
    _replace_task_line,
    condition_task_line,
    hidden_block,
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
    # First token of each text only ("You" vs "Nothing"): the belief without the wording penalty.
    logp_first_plus: float = float("nan")
    logp_first_minus: float = float("nan")
    # The first token as the model saw it after the chat template, decoded, to confirm what was scored.
    first_token_plus: str = ""
    first_token_minus: str = ""
    # The Hugging Face snapshot the weights and tokenizer were loaded at (wf.constants.REVISIONS).
    model_revision: str | None = None

    @property
    def L_first(self) -> float:
        """logit(q_first), q_first = P(t+) / (P(t+) + P(t-))."""
        return self.logp_first_plus - self.logp_first_minus

    @property
    def first_mass(self) -> float:
        """P(t+) + P(t-): how much of the next-token mass the two-way q_first actually covers."""
        return math.exp(self.logp_first_plus) + math.exp(self.logp_first_minus)

    @property
    def c_hat(self) -> float:
        """Wording penalty estimate, logit(q_first) - L."""
        return self.L_first - self.L

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
            "logp_first_plus": self.logp_first_plus,
            "logp_first_minus": self.logp_first_minus,
            "L_first": self.L_first,
            "first_token_plus": self.first_token_plus,
            "first_token_minus": self.first_token_minus,
            "first_mass": self.first_mass,
            "c_hat": self.c_hat,
            "model_revision": self.model_revision,
        }


class LogoddsScorer:
    """HF transformers teacher-forced scorer; mock mode when WF_MOCK_SCORE=1."""

    def __init__(
        self,
        model_name: str,
        model_tag: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ):
        """`attn_implementation` is passed to from_pretrained when set (Gemma 2 needs "eager"; wf/families.py);
        `template_kwargs` go to every apply_chat_template call (Llama 3.2's fixed date_string). The defaults leave
        every earlier run's loading and prompts unchanged."""
        self.model_name = model_name
        self.model_tag = model_tag or model_name.split("/")[-1].lower()
        self.mock = os.environ.get("WF_MOCK_SCORE", "0") == "1"
        # Real runs load a pinned snapshot and refuse an unpinned id; mock runs load nothing.
        self.revision = REVISIONS.get(model_name) if self.mock else revision_for(model_name)
        self._cache: dict[tuple[str, str, str, str], ScoreResult] = {}
        self._model = None
        self._tok = None
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.template_kwargs = dict(template_kwargs or {})
        if not self.mock:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # trust_remote_code stays as in every reported run; the pinned revision fixes which code and chat
        # template that is, so a later push to the Hub cannot change them.
        self._tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision, trust_remote_code=True)
        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        extra = {"attn_implementation": self.attn_implementation} if self.attn_implementation else {}
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **extra,
        )
        self._model.eval()
        loaded = getattr(self._model.config, "_attn_implementation", None)
        if self.attn_implementation and loaded != self.attn_implementation:
            raise RuntimeError(f"{self.model_name}: asked for {self.attn_implementation} attention, loaded {loaded}")

    def close(self) -> None:
        """Free the GPU before the next model loads; one A100 holds one 8B model at a time."""
        self._model = None
        self._tok = None
        if not self.mock:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            **self.template_kwargs,
        )

    def _teacher_force_logprob(self, messages: list[dict[str, str]], candidate: str) -> tuple[float, int, float, str]:
        """(log p of the whole candidate, its token count, log p of its first token, that token decoded)."""
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
            return logp - 0.01 * jitter, n_tok, logp / n_tok, candidate.split()[0] if candidate.split() else ""

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
            # float32: a bfloat16 log_softmax rounds each log-probability to about 3 significant digits
            lp = torch.log_softmax(logits[0, pos].float(), dim=-1)[tid].item()
            logprobs.append(lp)
        first = self._tok.decode(cand_ids[0][:1]) if cand_ids.shape[1] else ""
        return float(sum(logprobs)), len(logprobs), (logprobs[0] if logprobs else float("nan")), first

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

        logp_plus, n_plus, first_plus, tok_plus = self._teacher_force_logprob(messages, o_plus)
        logp_minus, n_minus, first_minus, tok_minus = self._teacher_force_logprob(messages, o_minus)
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
            logp_first_plus=first_plus,
            logp_first_minus=first_minus,
            first_token_plus=tok_plus,
            first_token_minus=tok_minus,
            model_revision=self.revision,
        )
        self._cache[key] = result
        return result

    def score_one_token_pair(
        self,
        transition_id: str,
        condition: str,
        messages: list[dict[str, str]],
        o_plus: str,
        o_minus: str,
    ) -> ScoreResult:
        """score_pair for two candidates of one token each, read off a single forward pass.

        The result equals score_pair's: each candidate's log-probability is the next-token log-probability
        after the prompt. Candidates longer than one token are refused rather than truncated.
        """
        if self.mock:
            return self.score_pair(transition_id, condition, messages, o_plus, o_minus)
        import torch

        assert self._model is not None and self._tok is not None
        ids = [self._tok(c, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist() for c in (o_plus, o_minus)]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"not single tokens: {o_plus!r} -> {ids[0]}, {o_minus!r} -> {ids[1]}")
        prompt_ids = self._encode(messages).to(next(self._model.parameters()).device)
        with torch.no_grad():
            logits = self._model(input_ids=prompt_ids).logits
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        plus, minus = lp[ids[0][0]].item(), lp[ids[1][0]].item()
        L = plus - minus
        if not math.isfinite(L):
            raise ValueError(f"non-finite L for {transition_id}/{condition} on {self.model_tag}: {L}")
        return ScoreResult(transition_id=transition_id, model=self.model_tag, condition=condition, logp_plus=plus,
                           logp_minus=minus, L=L, p_success=sigmoid(L), n_tokens_plus=1, n_tokens_minus=1,
                           logp_first_plus=plus, logp_first_minus=minus, first_token_plus=self._tok.decode(ids[0]),
                           first_token_minus=self._tok.decode(ids[1]), model_revision=self.revision)

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


def _layout(context: str) -> str:
    """The starting layout; the mock context has none, so mock runs get a fixed stand-in."""
    if os.environ.get("WF_MOCK_SCORE", "0") == "1" and "=== Objects on Receptacles ===" not in context:
        return "=== Objects on Receptacles ===\n - (mock layout)"
    return hidden_block(context)


@lru_cache(maxsize=None)
def _fewshot_hidden(fewshot_set: int) -> tuple[str, ...]:
    """Each example's starting layout; the lookup proves the game by its task line."""
    shots, games = WM_FEWSHOT_SETS[fewshot_set], WM_FEWSHOT_GAMES[fewshot_set]
    return tuple(_layout(world_model_context({"game_id": g, "task_line": ex["history"].splitlines()[0]}))
                 for ex, g in zip(shots, games))


def build_messages(transition: dict, condition: str, ft_prompt: bool = False, *, task_line: str | None = None,
                   fewshot_set: int = 1, full_state: bool = False, note: str | None = None) -> list[dict[str, str]]:
    """The prompt for one item under one condition; only the task line and framing vary.

    `task_line` overrides the condition's goal line. The others are few-shot options: which example set,
    the full-state condition (the starting layout shown before the history), and a note before
    "Next observation:". The fine-tuned prompt always carries the full state, as trained.
    """
    aligned, swapped = transition["task_line"], _swapped_task(transition)
    task = task_line or condition_task_line(condition, aligned, swapped)
    note = "\n\n".join(x for x in (_FRAMING.get(condition), note) if x) or None
    if ft_prompt:
        system = _replace_task_line(world_model_context(transition), task)
        if note:
            system = f"{system}{note}\n"
        return world_model_ft_messages(system, transition.get("history_pairs") or [], transition["action"])
    hidden = fewshot_hidden = None
    if full_state:
        hidden = _layout(transition.get("hidden_state") or world_model_context(transition))
        fewshot_hidden = list(_fewshot_hidden(fewshot_set))
    messages = world_model_prompt(transition["history_text"], transition["action"], fewshot=WM_FEWSHOT_SETS[fewshot_set],
                                  task_line_override=task, hidden=hidden, fewshot_hidden=fewshot_hidden)
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
