"""The real scorer's log-probabilities are float32, not bfloat16 (2026-11-18). Needs torch (runs on ICE).

Checked on ICE: passes with the fix, fails against the bfloat16 scorer."""
from __future__ import annotations

import pytest

from wf.scoring.logodds import LogoddsScorer

torch = pytest.importorskip("torch")


class _Ids:
    def __init__(self, ids):
        self.input_ids = torch.tensor([ids])


class _Tok:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return _Ids([5, 6])

    def apply_chat_template(self, *a, **k):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids):
        return "You"


class _Model:
    """bfloat16 logits, like the 8B models on the GPU."""

    def __init__(self, logits):
        self.logits = logits

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids):
        return type("Out", (), {"logits": self.logits})()


def test_log_probabilities_are_not_rounded_to_bfloat16():
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(1, 5, 1000, generator=gen) * 8).to(torch.bfloat16)
    scorer = LogoddsScorer.__new__(LogoddsScorer)
    scorer.mock, scorer._tok, scorer._model = False, _Tok(), _Model(logits)
    total, n, first, _ = scorer._teacher_force_logprob([{"role": "user", "content": "x"}], "You x")
    exact = torch.log_softmax(logits[0].float(), dim=-1)
    want_first = exact[2, 5].item()
    assert n == 2
    assert first == pytest.approx(want_first, abs=1e-5)
    assert total == pytest.approx(want_first + exact[3, 6].item(), abs=1e-5)
