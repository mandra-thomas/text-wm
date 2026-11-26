"""Collect the sample decisions shown in notes/walkthroughs/01_alfworld.md, with every stored score for each.

For each chosen item this writes, to `OUT` (default: notes/walkthroughs/data/alfworld_samples.json):
  - the item as collected (history, action, task lines, candidates, true reply, why the failure is determined);
  - the exact message lists the studies scored, built with the repository's own prompt builders
    (wf/scoring/logodds.build_messages through wf/rq1, wf/rq1v, wf/rq1x, wf/rq1d, wf/mech2);
  - every stored score of every model under every condition, read from the run archive and checked against
    the item hash stored with it (wf.runs._item_hash), so a score cannot belong to another item.

CPU only; nothing is scored. ARCHIVE must point at a local copy of the run archive (alfworld_data.py).

    ARCHIVE=/path/to/text-wm-runs-archive PYTHONPATH=. python scripts/walkthroughs/alfworld_samples.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alfworld_data import RUNS, index, load_items, load_rows  # noqa: E402

from wf import mech2, rq1, rq1d, rq1v, rq1x  # noqa: E402
from wf.data.transitions import determination_reason  # noqa: E402
from wf.rq2 import mention_lines  # noqa: E402
from wf.runs import _item_hash, _pilotA_item_sets  # noqa: E402

RUNS.update({f"family_score_{m}": f"text-wm-fix52/runs/family_score_{m}"
             for m in ("ministral3-8b-it", "qwen3.5-9b", "qwen3.5-9b-plain", "qwen3.5-9b-base-plain", "qwen3.8-27b")})

OUT = Path(os.environ.get("OUT", "notes/walkthroughs/data/alfworld_samples.json"))

# The sample decisions, in the order the walkthrough shows them, with the reason each was chosen.
SAMPLES = [
    ("t_002295", "worked example: the repository's golden item (closed cabinet, one step of history)"),
    ("t_1004121", "step 0, the agent has not moved: typical step-0 item"),
    ("t_2003104", "later step, the agent is at another receptacle: typical"),
    ("t_000835", "receptacle seen without the object: typical"),
    ("t_2001576", "hands full: typical"),
    ("t_1004224", "move without holding the object: typical"),
    ("t_1003683", "step 0, cool without holding the object (also in the object-mention 2x2)"),
    ("t_1001297", "extreme: the largest Qwen3-8B effect on a take action (aligned score above zero)"),
    ("t_004346", "extreme: the largest Llama-3.1-8B effect"),
    ("t_001217", "reversed in both reference models"),
    ("t_000388", "effect absent in both reference models (both prompts predict success)"),
    ("t_001712", "Llama-3.1-8B changes its yes/no answer from No to Yes"),
]
SUCCESS_SAMPLES = [
    # filled by pick_success() below when empty
]

# Model rows of the per-item table: (label, run key, model tag in the run, condition names: aligned, swapped,
# neutral, yes/no aligned, swapped, neutral). The Llama-3.1-8B row reads the runs families_analysis reads for it.
TABLE_MODELS = [
    ("Qwen3-8B", "size_score_qwen3-8b", "qwen3-8b"),
    ("Llama-3.1-8B-Instruct", ("rq1_score", "rq1v_score"), "llama3.1-8b-fewshot"),
    ("Gemma-2-9B-it", "family_score_gemma2-9b-it", "gemma2-9b-it"),
    ("Mistral-7B-Instruct-v0.3", "family_score_mistral-7b-v0.3", "mistral-7b-v0.3"),
    ("Llama-3.2-3B-Instruct", "family_score_llama3.2-3b", "llama3.2-3b"),
    ("Llama-3.2-1B-Instruct (not competent)", "family_score_llama3.2-1b", "llama3.2-1b"),
    ("Qwen3-0.6B (not competent)", "size_score_qwen3-0.6b", "qwen3-0.6b"),
    ("Qwen3-1.7B (not competent)", "size_score_qwen3-1.7b", "qwen3-1.7b"),
    ("Qwen3-4B", "size_score_qwen3-4b", "qwen3-4b"),
    ("Qwen3-14B", "size_score_qwen3-14b", "qwen3-14b"),
    ("Qwen3-32B", "size_score_qwen3-32b", "qwen3-32b"),
    ("Ministral-3-8B (wave B)", "family_score_ministral3-8b-it", "ministral3-8b-it"),
    ("Qwen3.5-9B, chat (wave B)", "family_score_qwen3.5-9b", "qwen3.5-9b"),
    ("Qwen3.5-9B, plain text (wave B)", "family_score_qwen3.5-9b-plain", "qwen3.5-9b-plain"),
    ("Qwen3.5-9B-Base, plain text (wave B)", "family_score_qwen3.5-9b-base-plain", "qwen3.5-9b-base-plain"),
    ("Qwen3.8-27B (wave B)", "family_score_qwen3.8-27b", "qwen3.8-27b"),
]
CONDS = ("aligned", "swapped", "absent", "yn_aligned", "yn_swapped", "yn_absent")


def checked(rec: dict | None, t: dict) -> dict | None:
    if rec is None:
        return None
    if "item_hash" in rec and rec["item_hash"] != _item_hash(t):
        raise ValueError(f"stale score for {t['id']}: {rec['item_hash']} != {_item_hash(t)}")
    return rec


def pick(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    keep = ("L", "L_first", "first_mass", "p_success", "logp_plus", "logp_minus", "logp_first_plus",
            "logp_first_minus", "n_tokens_plus", "n_tokens_minus", "first_token_plus", "first_token_minus")
    return {k: rec[k] for k in keep if k in rec}


def main() -> None:
    items = load_items()
    sets = _pilotA_item_sets(list(items.values()))
    primary_ids = {t["id"] for t in sets["agnostic"]}
    successes = rq1x.rq1x_sets(list(items.values()))["successes"]
    success_ids = {t["id"] for t in successes}

    idx: dict[str, dict] = {}
    for key in {k for _, r, _ in TABLE_MODELS for k in ((r,) if isinstance(r, str) else r)}:
        idx[key] = index(load_rows(key))
    rq2 = index(load_rows("rq2_score"))
    rq2_gen = {(g["model"], g["transition_id"], g["condition"]): g for g in load_rows("rq2_score", "generations.jsonl")}
    rq1_all = idx["rq1_score"]
    rq1_gen = {(g["model"], g["transition_id"], g["condition"]): g for g in load_rows("rq1_score", "generations.jsonl")}
    rq1x_s = index(load_rows("rq1x_score"))
    rq1v_s = idx["rq1v_score"]
    rq1d_s = index(load_rows("rq1d_score"))
    mech_s = index(load_rows("mech_score"))
    mech2_rows = load_rows("mech2_score")
    mech2_s = {(r["model"], r["transition_id"], r["condition"]): r for r in mech2_rows}
    think = {(r["transition_id"], r["condition"]): r for r in load_rows("thinking_probe", "thinking.jsonl")}

    def table(t: dict) -> list[dict]:
        rows = []
        for label, run, tag in TABLE_MODELS:
            row = {"model": label}
            for c in CONDS:
                if isinstance(run, tuple):
                    key = run[1] if c.startswith("yn_") else run[0]
                    rec = idx[key].get((tag, t["id"], c))
                    if rec is None and not c.startswith("yn_"):
                        rec = rq1x_s.get((tag, t["id"], c))  # the successes' plain arms are in rq1x_score
                else:
                    rec = idx[run].get((tag, t["id"], c))
                row[c] = pick(checked(rec, t))
            rows.append(row)
        return rows

    def build(t: dict, kind: str) -> dict:
        out = {
            "id": t["id"], "kind": kind, "game_id": t["game_id"], "state_idx": t["state_idx"],
            "history_len": t["history_len"], "action": t["action"], "true_obs": t["true_obs"],
            "candidates": t["candidates"], "task_line": t["task_line"], "swapped_task_line": t.get("swapped_task_line"),
            "goal_object": t.get("goal_object"), "swapped_goal_object": t.get("swapped_goal_object"),
            "reason": determination_reason(t) if t.get("context_determined") else None,
            "outcome_class": t.get("outcome_class"), "in_primary": t["id"] in primary_ids,
            "in_successes": t["id"] in success_ids, "history_text": t["history_text"],
            "messages_two_candidate_aligned": rq1.arm_messages(t, "aligned"),
            "yes_no_final_turn_aligned": rq1v.arm_messages(t, "yn_aligned")[-1]["content"],
            "item_hash": _item_hash(t),
        }
        lines = {"neutral": rq1.ABSENT_LINE}
        if t.get("swapped_task_line"):
            for arm in ("aligned_plus", "swapped_plus"):
                lines[arm] = rq1.goal_line(t, arm)
            if rq1x.desc_ok(t):
                lines["aligned_desc"] = rq1x.describe_line(t["task_line"], t["goal_object"])
                lines["swapped_desc"] = rq1x.describe_line(t["swapped_task_line"], t["swapped_goal_object"])
            ml = mention_lines(t)
            if ml:
                lines["mention_2x2"], lines["fourth_2x2"] = ml
            cf = rq1v.cf_lines(t)
            if cf:
                lines["cf_aligned"], lines["cf_swapped"] = cf
            for kind_ in rq1d.SENTENCES:
                s = rq1d.sentence(t, kind_)
                if s:
                    lines[f"rq1d_{kind_}_sentence"] = s
            for arm in mech2.ARMS:
                line = mech2.task_line(t, arm)
                if line is not None:
                    lines[f"mech2_{arm}"] = line
        out["task_lines"] = lines
        out["table"] = table(t)
        if kind == "success":  # the reference models scored the successes under all three task lines in rq1x/rq1v
            out["reference_success"] = {
                tag: {**{c: pick(checked(rq1x_s.get((tag, t["id"], c)), t)) for c in ("aligned", "swapped", "absent")},
                      **{c: pick(checked(rq1v_s.get((tag, t["id"], c)), t)) for c in ("yn_aligned", "yn_swapped", "yn_absent")}}
                for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")}
        if kind == "failure":
            conf = {}
            for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot"):
                conf[tag] = {c: pick(checked(rq2.get((tag, t["id"], c)), t)) for c in ("aligned", "swapped", "mention", "fourth")}
                conf[tag]["generations"] = {c: (rq2_gen.get((tag, t["id"], c)) or {}).get("generated") for c in ("aligned", "swapped")}
                conf[tag]["generation_classes"] = {c: (rq2_gen.get((tag, t["id"], c)) or {}).get("generated_class") for c in ("aligned", "swapped")}
            out["rq2"] = conf
            sec = {}
            for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot", "w2w-qwen2.5-7b", "w2w-llama3.1-8b"):
                sec[tag] = {c: pick(checked(rq1_all.get((tag, t["id"], c)), t)) for c in rq1.ARMS}
                sec[tag] = {k: v for k, v in sec[tag].items() if v}
                sec[tag]["generations"] = {c: (rq1_gen.get((tag, t["id"], c)) or {}).get("generated") for c in rq1.FEWSHOT_GEN_ARMS}
                sec[tag]["generation_classes"] = {c: (rq1_gen.get((tag, t["id"], c)) or {}).get("generated_class") for c in rq1.FEWSHOT_GEN_ARMS}
            out["rq1"] = sec
            out["rq1x"] = {tag: {c: pick(checked(rq1x_s.get((tag, t["id"], c)), t)) for c in ("aligned_desc", "swapped_desc")}
                           for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")}
            out["rq1v"] = {tag: {c: pick(checked(rq1v_s.get((tag, t["id"], c)), t)) for c in
                                 ("cf_aligned", "cf_swapped", "cf_aligned_plus", "cf_swapped_plus", "yn_aligned", "yn_swapped", "yn_absent")}
                           for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")}
            out["rq1d"] = {tag: {c: pick(checked(rq1d_s.get((tag, t["id"], c)), t)) for c in rq1d.ARMS}
                           for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")}
            mech_conds = sorted({k[2] for k in mech_s if k[1] == t["id"]})
            out["mech"] = {tag: {c: pick(checked(mech_s.get((tag, t["id"], c)), t)) for c in mech_conds}
                           for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")}
            out["mech_messages"] = {arm: (lambda m: m[-1]["content"] if m else None)(__import__("wf.mech", fromlist=["x"]).arm_messages(t, arm))
                                    for arm in ("pred_facts_aligned", "pred_rules_aligned", "state_aligned", "statetwin_aligned")}
            m2 = {}
            for tag in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot", "mistral-7b-v0.3"):
                m2[tag] = {c: pick(mech2_s[(tag, t["id"], c)]) for (tg, tid, c) in mech2_s if tg == tag and tid == t["id"]}
            out["mech2"] = m2
            out["thinking"] = {c: {k: think[(t["id"], c)].get(k) for k in ("L", "answer", "text", "finished", "forced", "says_success")}
                               for c in ("aligned", "swapped", "absent") if (t["id"], c) in think}
        return out

    picked_success = SUCCESS_SAMPLES or pick_success(successes, items)
    data = {"about": __doc__.strip().splitlines()[0], "runs": RUNS,
            "samples": [{**build(items[i], "failure"), "why_chosen": why} for i, why in SAMPLES],
            "successes": [{**build(items[i], "success"), "why_chosen": why} for i, why in picked_success]}
    for s in data["samples"]:
        if not s["in_primary"]:
            raise ValueError(f"{s['id']} is not in RQ2's population")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(data['samples'])} failures, {len(data['successes'])} successes")


def pick_success(successes: list[dict], items: dict) -> list[tuple[str, str]]:
    """Two working actions: a take of the task's own object (one of the 37 matched successes, if any) and a
    go to, the action an agent most often proposes; the first in a fixed order (by id)."""
    take = sorted((t for t in successes if t["action"].startswith("take") and t.get("targets_goal_object")), key=lambda t: t["id"])
    go = sorted((t for t in successes if t["action"].startswith("go to")), key=lambda t: t["id"])
    out = []
    if take:
        out.append((take[0]["id"], "working action: a take of the task's own object"))
    if go:
        out.append((go[0]["id"], "working action: a go to"))
    return out


if __name__ == "__main__":
    main()
