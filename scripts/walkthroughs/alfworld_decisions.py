"""Collect the decision-level examples shown in notes/walkthroughs/01_alfworld.md.

  - RQ3: one injected-error episode, with the agent's prompt at one step rebuilt by the study's own builder
    (wf/scoring/prompts.agent_react_messages) from the stored trace;
  - Stage 1a: one state's fixed candidate menu, each candidate's stored yes/no scores under every verifier arm,
    and the action the frozen rule 2 chooses under each (wf/decisionsim.build_states, quantize, rule_choice);
  - Stage 1b: one encounter state from the agent's own candidate list, with the frozen selection rule
    (wf/decisionsim_agent.views, scores_by_arm, choose);
  - Option C's step-0 check: the same Stage 1a state under the yes/no and the observation readouts.

CPU only; nothing is scored. ARCHIVE as in alfworld_data.py; OUT defaults to
notes/walkthroughs/data/alfworld_decisions.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alfworld_data import ARCHIVE, RUNS, load_items, load_rows, read_jsonl  # noqa: E402

from wf import decisionsim as ds  # noqa: E402
from wf import decisionsim_agent as da  # noqa: E402
from wf.scoring.prompts import agent_react_messages  # noqa: E402

OUT = Path(os.environ.get("OUT", "notes/walkthroughs/data/alfworld_decisions.json"))
STAGE1A_STATE = ("valid_unseen/140", 0)
RQ3_EPISODE = {"game_id": "valid_unseen/157", "condition": "corrupted", "arm": "favorable", "retention": "retained",
               "schedule_seed": 20260926}
RQ3_STEP = 4
STAGE1B_GAME = "valid_unseen/150"


def rq3_example(items: dict) -> dict:
    eps = []
    for f in sorted(glob.glob(str(ARCHIVE / str(RUNS["rq3"]) / "shard_*" / "episodes.jsonl"))):
        eps += read_jsonl(f)
    ep = next(e for e in eps if all(e.get(k) == v for k, v in RQ3_EPISODE.items()))
    first = next(t["history_text"] for t in items.values() if t["game_id"] == ep["game_id"] and t["state_idx"] == 0)
    history, obs = [], first
    for st in ep["trace"][:RQ3_STEP]:
        history.append({"observation": obs, "action": st["executed"], "proposal": st["proposal"],
                        "expected_outcome": st["prediction_shown"]})
        obs = st["real_obs"]
    st = ep["trace"][RQ3_STEP]
    msgs = agent_react_messages(observation=obs, history=history, foresight_prediction=st["prediction_shown"],
                                proposal=st["proposal"], admissible=None)
    return {"episode": {k: v for k, v in ep.items() if k != "trace"},
            "trace": [{k: s[k] for k in ("t", "proposal", "prediction_shown", "prediction_type", "executed",
                                         "real_obs", "proposal_valid", "executed_raw")} for s in ep["trace"]],
            "messages_at_step": RQ3_STEP, "messages": msgs,
            "note": "rebuilt without the admissible-action list, which the trace does not store"}


def stage1a_example(items: dict) -> dict:
    states = ds.build_states(list(items.values()))
    state = next(s for s in states if (s["game_id"], s["state_idx"]) == STAGE1A_STATE)
    out = {"game_id": state["game_id"], "state_idx": state["state_idx"], "proposal": state["proposal"],
           "swap": state["swap"], "task_line": state["candidates"][0]["task_line"], "models": {}}
    out["candidates"] = [{"id": t["id"], "action": t["action"], "succeeds": ds.succeeded(t),
                          "goal_object": bool(t.get("targets_goal_object")), "determined": bool(t.get("context_determined")),
                          "true_obs": t["true_obs"]} for t in state["candidates"]]
    for tag, key in (("qwen3-8b", "decisionsim_qwen"), ("llama3.1-8b", "decisionsim_llama")):
        rows = [r for r in load_rows(key) if r["game_id"] == state["game_id"] and r["state_idx"] == state["state_idx"]]
        L: dict[str, dict[str, float]] = {}
        for r in rows:
            L.setdefault(r["condition"], {})[r["transition_id"]] = ds.quantize(r["L"])
        choices = {arm: ds.rule_choice("rule2_verification", state, L[arm])["action"] for arm in L}
        out["models"][tag] = {"scores": L, "rule2_choice": choices}
    return out


def stage1b_example() -> dict:
    labs = []
    for f in sorted(glob.glob(str(ARCHIVE / "text-wm-fix29/runs/decisionsim_agent_label/shard_*/labels.jsonl"))):
        labs += [x for x in read_jsonl(f) if x["game_id"] == STAGE1B_GAME]
    vs = da.views(labs)
    rows = []
    for f in sorted(glob.glob(str(ARCHIVE / "text-wm-fix37/runs/decisionsim_agent_score_llama3.1-8b/shard_*/scoring.jsonl"))):
        rows += [r for r in read_jsonl(f) if r["game_id"] == STAGE1B_GAME]
    L, _ = da.scores_by_arm(rows)
    found = []
    for s in sorted(vs, key=lambda s: s["state_idx"]):
        if not da.encounter(s) or len(da.scored_candidates(s)) < 2:
            continue
        ca, cb = da.choose("aware", s, L), da.choose("blind", s, L)
        cands = []
        for c in da.scored_candidates(s):
            cid = da.cand_id(s, c)
            cands.append({"rank": c["rank"], "count": c["count"], "action": c["action"], "failed": c["failed"],
                          "regime": da.regime(c), "true_obs": c["true_obs"],
                          "L": {arm: L.get(arm, {}).get(cid) for arm in da.ARMS}})
        found.append({"state_idx": s["state_idx"], "task_line": s["task_line"], "history_text": s["history_text"],
                      "aware_choice": ca["action"], "aware_fails": ca["failed"],
                      "blind_choice": cb["action"], "blind_fails": cb["failed"], "candidates": cands})
    return {"game_id": STAGE1B_GAME, "model": "llama3.1-8b", "encounter_states": found}


def optionc_step0_example() -> dict:
    rows = [r for r in read_jsonl(ARCHIVE / "text-wm-fix40/runs/optionc_step0_qwen3-8b/scoring.jsonl")
            if (r["game_id"], r["state_idx"]) == STAGE1A_STATE]
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["action"], {})[f"{r['readout']}_{r['arm']}"] = r["L"]
    return {"game_id": STAGE1A_STATE[0], "state_idx": STAGE1A_STATE[1], "model": "qwen3-8b", "scores": out}


def main() -> None:
    items = load_items()
    data = {"rq3": rq3_example(items), "stage1a": stage1a_example(items), "stage1b": stage1b_example(),
            "optionc_step0": optionc_step0_example()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
