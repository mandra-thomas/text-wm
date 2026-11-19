"""Model-size series (wf/sizes.py)."""
import json

from wf import rq1, rq1v, sizes
from tests.test_rq1 import _item, mock_root, mock_score  # noqa: F401


def test_arms_reuse_the_rq2_and_yes_no_prompts(mock_score):
    t = _item()
    assert sizes.arm_messages(t, "aligned") == rq1.arm_messages(t, "aligned")
    assert sizes.arm_messages(t, "yn_swapped") == rq1v.arm_messages(t, "yn_swapped")
    assert sizes.arm_messages({**t, "swapped_task_line": None}, "swapped") is None


def test_every_size_is_registered():
    from wf.runs import RUNNERS as RUNS
    for tag in sizes.SIZES:
        assert f"size_score_{tag}" in RUNS


def test_end_to_end(mock_root):
    cfg = {"run_id": "size_score_qwen3-0.6b", "model": "qwen3-0.6b", "sources": ["rq12_transitions_smoke"], "n_shards": 1}
    assert sizes.run_size_score(cfg) == 0
    assert sizes.run_size_analysis({"run_id": "size_smoke_analysis", "sources": ["rq12_transitions_smoke"],
                                    "models": ["qwen3-0.6b", "qwen3-1.7b"]}) == 0
    m = json.loads((mock_root / "size_smoke_analysis" / "metrics.json").read_text())
    assert "qwen3-0.6b" in m["models"] and "qwen3-1.7b" in m["waiting"]
    assert {"goal_effect", "yes_no", "auc", "flips_to_success"} <= set(m["models"]["qwen3-0.6b"])
