"""Teacher-forced scoring of many sequences that share a long prompt prefix (written for the natural bridge,
notes/bridge_design.md).

A group is a list of sequences, each a prompt (token ids), a continuation to score (token ids, possibly empty) and
"probe" token ids whose next-token log-probabilities are read at the end of the prompt. The prompts of a group share
a prefix (the stem): everything up to the action line of an ALFWorld item. The stem minus its last token is run once;
each sequence then runs from the stem's last token to its own end with the stem's key-value cache, which is cut back
to the same length before every sequence. So every sequence of a group is computed the same way: one cached stem,
one pass over its own remainder, one sequence at a time (batch size 1, no padding).

The scores equal a plain full forward pass up to floating-point rounding: in float32 to about 1e-5 (tested on a
small random model, tests/test_bridge.py); in bf16 the split changes the order of additions, as any change of
input shape does, so they differ from the full pass by rounding noise of the size measured in
notes/numerical_sensitivity.md. Every score of a study must therefore come from this one routine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class SeqScore:
    cont_logprobs: list[float]  # log p of each continuation token
    probe_logprobs: list[float]  # log p of each probe token at the end of the prompt
    stem_len: int  # the group's stem length in tokens (an audit field)


def _cut(cache: Any, length: int) -> None:
    """Cut the key-value cache back to its first `length` tokens. transformers 5 accepts only a negative argument to
    DynamicCache.crop (the number of tokens to remove); transformers 4 reads a negative argument the same way."""
    n = cache.get_seq_length()
    if n < length:
        raise ValueError(f"cache holds {n} tokens, fewer than {length}")
    if n > length:
        cache.crop(length - n)
    if cache.get_seq_length() != length:
        raise RuntimeError(f"cache crop left {cache.get_seq_length()} tokens, expected {length}")


def common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    if not seqs:
        return 0
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(n):
        t = first[i]
        if any(s[i] != t for s in seqs[1:]):
            return i
    return n


class PrefixScorer:
    """Wraps a loaded causal LM (Hugging Face transformers)."""

    def __init__(self, model: Any):
        self.model = model

    def _device(self):
        return next(self.model.parameters()).device

    def score_group(self, seqs: list[tuple[list[int], list[int], list[int]]]) -> list[SeqScore]:
        import torch

        if not seqs:
            return []
        prompts = [p for p, _, _ in seqs]
        if any(len(p) < 1 for p in prompts):
            raise ValueError("empty prompt")
        stem = common_prefix_len(prompts)
        if stem < 1:
            raise ValueError("the group's prompts share no prefix")
        dev = self._device()
        out: list[SeqScore] = []
        with torch.no_grad():
            cache = None
            if stem - 1 > 0:
                ids = torch.tensor([prompts[0][: stem - 1]], device=dev)
                cache = self.model(input_ids=ids, use_cache=True).past_key_values
            for p, c, probes in seqs:
                if cache is not None:
                    _cut(cache, stem - 1)
                full = list(p) + list(c)
                ids = torch.tensor([full[stem - 1:]], device=dev)
                res = self.model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
                if cache is None:
                    pass  # no stem: nothing to reuse (only when the shared prefix is one token)
                # row r of the logits predicts token (stem - 1) + r + 1 of `full`; the last prompt token is row
                # len(p) - stem, and continuation token k is predicted by row len(p) - stem + k
                start = len(p) - stem
                logits = res.logits[0, start: start + len(c) + 1].float()
                lp = torch.log_softmax(logits, dim=-1)
                cont = [lp[k, c[k]].item() for k in range(len(c))]
                pr = [lp[0, t].item() for t in probes]
                out.append(SeqScore(cont_logprobs=cont, probe_logprobs=pr, stem_len=stem))
        return out


def full_pass_score(model: Any, prompt: list[int], cont: list[int], probes: list[int]) -> SeqScore:
    """The same quantities from one plain forward pass over prompt + continuation (the reference the cached
    routine is tested against)."""
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([list(prompt) + list(cont)], device=dev)).logits[0].float()
    lp = torch.log_softmax(logits[len(prompt) - 1: len(prompt) - 1 + len(cont) + 1], dim=-1)
    return SeqScore(cont_logprobs=[lp[k, cont[k]].item() for k in range(len(cont))],
                    probe_logprobs=[lp[0, t].item() for t in probes], stem_len=0)
