"""Thinking-mode probe (wf/thinking.py)."""
import json

from wf import thinking
from tests.test_rq1 import _item, mock_root  # noqa: F401
from wf.util import read_jsonl


def test_split_output_reads_reasoning_and_answer():
    think, answer, done = thinking.split_output("<think>\nThe mug is far.\n</think>\n\nNothing happens.")
    assert (think, answer, done) == ("The mug is far.", "Nothing happens.", True)
    assert thinking.split_output("<think>\nstill going")[2] is False
    assert thinking.reasoning_prefix("<think>\nx\n</think>\n\nYou pick") == "<think>\nx\n</think>\n\n"


def test_sample_is_fixed_and_order_free():
    items = [_item(id=f"t{i}", game_id=f"g{i % 7}") for i in range(50)]
    a = thinking.sample_items(items, 10)
    b = thinking.sample_items(list(reversed(items)), 10)
    assert [t["id"] for t in a] == [t["id"] for t in b] and len(a) == 10


def test_end_to_end_mock(mock_root, monkeypatch):
    cfg = {"run_id": "thinking_smoke", "sources": ["rq12_transitions_smoke"], "n_items": 20, "chunk": 7}
    assert thinking.run_thinking(cfg) == 0
    recs = read_jsonl(mock_root / "thinking_smoke" / "thinking.jsonl")
    assert recs and {r["condition"] for r in recs} <= set(thinking.ARMS)
    assert all(r["finished"] and "L" in r for r in recs)
    assert "forced" in recs[0]
    n = len(recs)
    assert thinking.run_thinking({**cfg, "resume": True}) == 0  # resume adds nothing
    assert len(read_jsonl(mock_root / "thinking_smoke" / "thinking.jsonl")) == n
    acfg = {"run_id": "thinking_smoke_analysis", "source": "thinking_smoke", "sources": ["rq12_transitions_smoke"], "n_items": 20}
    assert thinking.run_thinking_analysis(acfg) == 0
    m = json.loads((mock_root / "thinking_smoke_analysis" / "metrics.json").read_text())
    assert {"written_shift_pp", "L_after_reasoning", "reasoning_mentions"} <= set(m)


def test_budget_forcing_closes_unfinished_reasoning(mock_root, monkeypatch):
    calls = []

    def gen(self, prompts, keys, max_tokens):
        calls.append(max_tokens)
        if max_tokens == 64:
            return [("Nothing happens.", 4) for _ in prompts]
        return [("<think>\nloops forever", max_tokens) for _ in prompts]

    monkeypatch.setattr(thinking._MockEngine, "generate", gen)
    cfg = {"run_id": "thinking_force", "sources": ["rq12_transitions_smoke"], "n_items": 4, "chunk": 50, "max_tokens": 100}
    assert thinking.run_thinking(cfg) == 0
    recs = read_jsonl(mock_root / "thinking_force" / "thinking.jsonl")
    assert recs and all(r["forced"] and r["finished"] and r["answer"] == "Nothing happens." for r in recs)
    assert all(r["text"].endswith(thinking.FORCE_CLOSE + "Nothing happens.") for r in recs) and 64 in calls
