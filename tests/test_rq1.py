"""RQ1, H1 and the secondary RQ2 conditions (wf/rq1.py), and the prompt options they use."""
import json
from pathlib import Path

import pytest

import wf.runs as runs_mod
from wf import rq1, rq2
from wf.scoring import logodds
from wf.scoring.prompts import (
    WM_FEWSHOT_GAMES,
    WM_FEWSHOT_SET0,
    WM_FEWSHOT_SETS,
    hidden_block,
    world_model_prompt,
)
from wf.util import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
CONTEXT = ("# Environment Information (Only visible to Assistant)\n\n=== Objects on Receptacles ===\n"
           " - On the cabinet 1, you see a mug 1\n - On the shelf 1, you see a saltshaker 1\n\n"
           "# User Environment Information (Displayed to User)\n\nYou are in the middle of a room.\n"
           "Your task is to: put some saltshaker on cabinet.\nAVAILABLE ACTIONS: go to cabinet 1,look\n")


def _item(**kw):
    t = {"id": "t1", "game_id": "valid_unseen/1", "action": "take saltshaker 1 from cabinet 1",
         "task_line": "Your task is to: put some saltshaker on cabinet.",
         "swapped_task_line": "Your task is to: put some mug on cabinet.",
         "goal_object": "saltshaker", "swapped_goal_object": "mug", "hidden_state": CONTEXT,
         "history_text": "You are in the middle of a room.\nYour task is to: put some saltshaker on cabinet.\n"
                         "> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a saltshaker 1.",
         "history_pairs": [["go to shelf 1", "You arrive at shelf 1. On the shelf 1, you see a saltshaker 1."]],
         "candidates": {"o_plus": "You pick up the saltshaker 1 from the cabinet 1.", "o_minus": "Nothing happens."}}
    return {**t, **kw}


@pytest.fixture
def mock_score(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    logodds._fewshot_hidden.cache_clear()
    yield
    logodds._fewshot_hidden.cache_clear()


def _text(messages):
    return "\n".join(m["content"] for m in messages)


def test_hidden_block_keeps_the_layout_and_drops_the_task_line():
    block = hidden_block(CONTEXT)
    assert block.startswith("=== Objects on Receptacles ===") and "saltshaker 1" in block
    assert "Your task is to" not in block and "AVAILABLE ACTIONS" not in block
    with pytest.raises(ValueError):
        hidden_block("# Environment Information\n\nYour task is to: x.")


def test_full_state_swapped_prompt_never_shows_the_true_goal(mock_score):
    for arm in ("swapped_full", "swapped", "swapped_set2", "swapped_ignore", "absent"):
        text = _text(rq1.arm_messages(_item(), arm))
        assert "put some saltshaker on cabinet" not in text, arm
    full = _text(rq1.arm_messages(_item(), "swapped_full"))
    assert "On the shelf 1, you see a saltshaker 1" in full  # the item's own layout is shown


def test_full_state_prompt_shows_a_layout_in_every_turn(mock_score):
    users = [m["content"] for m in rq1.arm_messages(_item(), "aligned_full") if m["role"] == "user"]
    assert len(users) == len(WM_FEWSHOT_SET0) + 1 and all(u.startswith("Hidden state") for u in users)
    plain = [m["content"] for m in rq1.arm_messages(_item(), "aligned") if m["role"] == "user"]
    assert not any("Hidden state" in u for u in plain)


def test_the_primary_prompt_is_unchanged_by_the_new_options():
    t = _item()
    expected = world_model_prompt(t["history_text"], t["action"], task_line_override=t["task_line"])
    assert logodds.build_messages(t, "aligned") == expected == rq1.arm_messages(t, "aligned")
    assert [m["content"] for m in expected if m["role"] == "assistant"] == [e["observation"] for e in WM_FEWSHOT_SET0]


def test_example_sets_are_the_frozen_draws_with_their_games():
    frozen = json.loads((ROOT / "notes" / "fewshot_sets_frozen.json").read_text())
    for k in (2, 3):
        assert [(e["history"], e["action"], e["observation"]) for e in frozen[f"set{k}"]] == \
            [(e["history"], e["action"], e["observation"]) for e in WM_FEWSHOT_SETS[k]]
        assert tuple(e["game"] for e in frozen[f"set{k}"]) == WM_FEWSHOT_GAMES[k]
    for k, shots in WM_FEWSHOT_SETS.items():
        assert len(shots) == len(WM_FEWSHOT_GAMES[k]) and all(e["history"].startswith("Your task is to:") for e in shots)
    # set 2's successes include a take, so the verb alone does not predict the outcome
    s2 = WM_FEWSHOT_SETS[2]
    assert any(e["action"].startswith("take ") and e["observation"] != "Nothing happens." for e in s2)


def test_each_set_puts_its_own_examples_in_the_prompt():
    for k in (2, 3):
        got = [m["content"] for m in rq1.arm_messages(_item(), f"aligned_set{k}") if m["role"] == "assistant"]
        assert got == [e["observation"] for e in WM_FEWSHOT_SETS[k]]


def test_plus_lines_name_both_objects_and_differ_only_in_the_goal():
    t = _item()
    a, s = rq1.goal_line(t, "aligned_plus"), rq1.goal_line(t, "swapped_plus")
    assert a == "Your task is to: put some saltshaker on cabinet. There is a mug somewhere in the room."
    assert s == "Your task is to: put some mug on cabinet. There is a saltshaker somewhere in the room."
    assert rq1.goal_line(_item(swapped_goal_object="apple"), "aligned_plus").endswith("There is an apple somewhere in the room.")
    with pytest.raises(ValueError):
        rq1.goal_line(_item(swapped_task_line=None), "swapped_plus")


def test_ignore_note_only_in_the_query_turn():
    msgs = rq1.arm_messages(_item(), "aligned_ignore")
    assert rq1.IGNORE_NOTE in msgs[-1]["content"] and msgs[-1]["content"].endswith("Next observation:")
    assert sum(rq1.IGNORE_NOTE in m["content"] for m in msgs) == 1


def test_fine_tuned_prompt_takes_only_the_two_goals(mock_score):
    msgs = rq1.arm_messages(_item(), "swapped", ft=True)
    assert "put some mug on cabinet" in msgs[0]["content"] and "put some saltshaker on cabinet" not in msgs[0]["content"]
    with pytest.raises(ValueError):
        rq1.arm_messages(_item(), "aligned_full", ft=True)


def test_rq2_mention_lines_reach_the_fine_tuned_prompt(mock_score):
    scorer = logodds.LogoddsScorer("x", model_tag="w2w")
    seen = []
    scorer.score_pair = lambda **kw: seen.append(kw["messages"])
    rq2._score(scorer, _item(), "mention", task_line="Your task is to: put some saltshaker on shelf.", ft=True)
    assert seen[0][0]["role"] == "system" and "put some saltshaker on shelf." in seen[0][0]["content"]


def test_a_difference_is_paired_and_reads_the_right_direction():
    items = [{"id": f"f{g}{i}", "game_id": f"g{g}", "true_obs": "Nothing happens."} for g in range(10) for i in range(4)] + \
            [{"id": f"s{g}{i}", "game_id": f"g{g}", "true_obs": "You pick up the mug 1."} for g in range(10) for i in range(4)]
    # first arm: says success on every failure (A = 1); second: says no effect everywhere (A = -1 + 0 ... = -1)
    first = [{"transition_id": t["id"], "generated_class": "state_change"} for t in items]
    second = [{"transition_id": t["id"], "generated_class": "noop"} for t in items]
    signed = rq1.a_difference(first, second, items, absolute=False)
    assert signed["estimate"] == pytest.approx(2.0) and signed["supported"]
    same = rq1.a_difference(first, first, items, absolute=True)
    assert same["estimate"] == 0 and not same["supported"]


@pytest.fixture
def mock_root(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    logodds._fewshot_hidden.cache_clear()
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    assert runs_mod.run_pilotA_transitions({"run_id": "rq12_transitions_smoke", "split": "train", "n_games": 6,
                                            "step_cap": 12, "min_items": 1, "min_states": 1}) == 0
    from wf.tools.verify_items import verify_run
    assert verify_run("rq12_transitions_smoke", server=True)["passed"]
    yield tmp_path
    logodds._fewshot_hidden.cache_clear()


def _score_cfg(**kw):
    return {"run_id": "rq1_smoke", "sources": ["rq12_transitions_smoke"], "n_shards": 2, **kw}


def _analysis_cfg():
    return {"run_id": "rq1_smoke_analysis", "source": "rq1_smoke", "sources": ["rq12_transitions_smoke"]}


def test_end_to_end_two_shards_then_analysis(mock_root):
    for i in range(2):
        assert rq1.run_rq1_score(_score_cfg(shard=i)) == 0
    assert rq1.run_rq1_analysis(_analysis_cfg()) == 0
    m = json.loads((mock_root / "rq1_smoke_analysis" / "metrics.json").read_text())
    assert m["items"]["missing_predictions"] == 0 and m["items"]["missing_scores"] == 0
    for tag in rq1.FEWSHOT:
        rep = m["per_model"][tag]
        assert {"rq1_history_only", "rq1_full_state", "H1_history_vs_full_state", "A_per_example_set", "rq2_secondary"} <= set(rep)
        assert set(rep["rq2_secondary"]["per_example_set"]) == {"set1", "set2", "set3"}
        assert "goal_role_both_named" in rep["rq2_secondary"]["extra_mention"]["all"]
    for tag in ("w2w-qwen2.5-7b", "w2w-llama3.1-8b"):
        assert "rq1_full_state_as_trained" in m["per_model"][tag] and "delta_L_full_state" in m["per_model"][tag]
    assert "estimate" in m["H1_llama_minus_qwen"]
    # every arm was scored or generated for every item it applies to
    recs = [r for p in sorted((mock_root / "rq1_smoke").glob("shard_*")) for r in read_jsonl(p / "scoring.jsonl")]
    assert {r["condition"] for r in recs if r["model"] == "qwen3-8b-fewshot"} == set(rq1.FEWSHOT_SCORE_ARMS)
    assert {r["condition"] for r in recs if r["model"].startswith("w2w")} == set(rq1.FT_SCORE_ARMS)


def test_analysis_waits_for_every_shard(mock_root):
    assert rq1.run_rq1_score(_score_cfg(shard=0)) == 0
    assert rq1.run_rq1_analysis(_analysis_cfg()) == 1
    assert "waits for every scoring shard" in (mock_root / "rq1_smoke_analysis" / "STATUS.md").read_text()


def test_resume_does_not_duplicate_records(mock_root):
    cfg = _score_cfg(shard=0, n_shards=1)
    rq1.run_rq1_score(cfg)
    d = mock_root / "rq1_smoke" / "shard_00_of_01"
    before = (len(read_jsonl(d / "scoring.jsonl")), len(read_jsonl(d / "generations.jsonl")))
    rq1.run_rq1_score({**cfg, "resume": True})
    assert (len(read_jsonl(d / "scoring.jsonl")), len(read_jsonl(d / "generations.jsonl"))) == before
