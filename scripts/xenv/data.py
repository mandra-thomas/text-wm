"""Loaders for the cross-environment comparison (exploratory, post hoc; notes/textcraft_vs_alfworld.md).

Builds one table per environment, model and item population: one row per item, with every scored arm as a column
(two-candidate L and its first-token term, the reply log-probabilities, the yes/no log-odds) and the item fields
used for strata. Nothing here scores anything; it only reads saved scores.

ALFWorld (archived runs, read via RUNS_DIR-independent paths):
  rq2_score   aligned / swapped two-candidate on the 4,666 RQ2 failures (the confirmatory scores)
  rq1_score   aligned / swapped / absent (neutral "explore the room") two-candidate on the same failures
  rq1x_score  aligned / swapped / absent two-candidate on the 292 determined successes
  rq1v_score  yn_aligned / yn_swapped / yn_absent on failures and successes; cf_* counterfactual-aligned arms
TextCraft (copied from ICE, text-wm-fix24/runs/textcraft_score): every arm on every population of
wf/textcraft/data/items.jsonl (branch textcraft).
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
SHORT = {"qwen3-8b-fewshot": "Qwen3-8B", "llama3.1-8b-fewshot": "Llama-3.1-8B"}
SCORE_FIELDS = ("L", "L_first", "logp_plus", "logp_minus", "logp_first_plus", "logp_first_minus",
                "n_tokens_plus", "n_tokens_minus", "first_mass")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _scores(paths: list[str]) -> dict[tuple[str, str, str], dict]:
    out = {}
    for p in paths:
        for r in _read_jsonl(p):
            out[(r["model"], r["transition_id"], r["condition"])] = r
    return out


def _table(items: list[dict], scores: dict, model: str, arms: dict[str, str], cluster: str,
           fields: dict) -> dict[str, np.ndarray]:
    """Columns: for each (prefix, condition) in arms, every score field as f'{prefix}.{field}'; plus item fields."""
    t: dict[str, list] = defaultdict(list)
    for it in items:
        t["id"].append(it["id"])
        t["cluster"].append(it[cluster])
        for name, fn in fields.items():
            t[name].append(fn(it))
        for prefix, cond in arms.items():
            s = scores.get((model, it["id"], cond))
            for f in SCORE_FIELDS:
                v = s.get(f) if s else None
                t[f"{prefix}.{f}"].append(np.nan if v is None else float(v))
    return {k: np.asarray(v) for k, v in t.items()}


# ---------------------------------------------------------------- ALFWorld

def _alf_fields() -> dict:
    from wf.data.goals import parse_task_line

    def target(t):
        try:
            return parse_task_line(t["task_line"]).receptacle
        except ValueError:
            return ""

    def task_kind(t):
        line = t["task_line"].split("Your task is to:", 1)[-1].strip().lower()
        for k, pat in (("two", r"\btwo\b"), ("look", r"^(look at|examine)"), ("heat", r"\b(heat|hot)\b"),
                       ("cool", r"\b(cool|cold)\b"), ("clean", r"\bclean\b")):
            if re.search(pat, line):
                return k
        return "place"

    def moved_to(t):
        m = re.fullmatch(r"move (.+?) \d+ to ([a-z]+) \d+", t["action"])
        return m[2] if m else ""

    def named(t, obj):
        hist = " ".join(o for _, o in t.get("history_pairs") or [])
        return bool(obj) and re.search(rf"\b{re.escape(obj)}\b", hist) is not None

    return {
        "history_len": lambda t: t["history_len"],
        "reason": lambda t: t.get("determination_reason") or "",
        "action_type": lambda t: t["action_type"],
        "task_kind": task_kind,
        "target": target,
        "move_dest": moved_to,
        # a move of the goal object to a receptacle of the task's target type: the act that completes a
        # plain placement task (or the final step of a heat/cool/clean task)
        "move_to_target": lambda t: t["action_type"] == "move" and moved_to(t) == target(t) != "",
        "goal_seen_in_history": lambda t: named(t, t.get("goal_object")),
        "swap_seen_in_history": lambda t: named(t, t.get("swapped_goal_object")),
        "o_plus": lambda t: (t.get("candidates") or {}).get("o_plus", ""),
    }


def load_alfworld(archive: str) -> dict:
    """{model: {'fail': table, 'succ': table}} plus item lists."""
    os.environ.setdefault("RUNS_DIR", os.path.join(archive, "text-wm-fix17", "runs"))
    from wf import rq2

    transitions = rq2.load_items()
    fails = rq2.primary_set(transitions)
    g = lambda run: sorted(glob.glob(os.path.join(archive, run, "shard_*", "scoring.jsonl")))
    rq2s = _scores(g("text-wm-fix11/runs/rq2_score"))
    rq1s = _scores(g("text-wm-fix12/runs/rq1_score"))
    rq1x = _scores(g("text-wm-fix13/runs/rq1x_score"))
    rq1v = _scores(g("text-wm-fix17/runs/rq1v_score"))
    succ_ids = {k[1] for k in rq1x if k[2] == "absent"}
    by_id = {t["id"]: t for t in transitions}
    succs = [by_id[i] for i in sorted(succ_ids)]
    fields = _alf_fields()
    out = {}
    for m in MODELS:
        f2 = _table(fails, rq2s, m, {"al": "aligned", "sw": "swapped"}, "game_id", fields)
        f1 = _table(fails, rq1s, m, {"al1": "aligned", "sw1": "swapped", "neu": "absent"}, "game_id", {})
        fv = _table(fails, rq1v, m, {"yal": "yn_aligned", "ysw": "yn_swapped", "yneu": "yn_absent",
                                     "cfal": "cf_aligned", "cfsw": "cf_swapped", "val": "aligned", "vsw": "swapped"},
                    "game_id", {})
        s1 = _table(succs, rq1x, m, {"al": "aligned", "sw": "swapped", "neu": "absent"}, "game_id", fields)
        sv = _table(succs, rq1v, m, {"yal": "yn_aligned", "ysw": "yn_swapped", "yneu": "yn_absent"}, "game_id", {})
        fail = {**f2, **{k: v for k, v in f1.items() if "." in k}, **{k: v for k, v in fv.items() if "." in k}}
        succ = {**s1, **{k: v for k, v in sv.items() if "." in k}}
        out[m] = {"fail": fail, "succ": succ}
    return out


# ---------------------------------------------------------------- TextCraft

TC_ARMS = {"al": "aligned", "sw": "swapped", "neu": "neutral", "yal": "yn_aligned", "ysw": "yn_swapped",
           "yneu": "yn_neutral", "pal": "prefix_aligned", "psw": "prefix_swapped"}
TC_FIELDS = {
    "history_len": lambda t: t["history_len"],
    "role": lambda t: t["role"],
    "failure_class": lambda t: t["failure_class"],
    "history_fit": lambda t: t.get("history_fit") or "",
    "action_kind": lambda t: t["action_kind"],
    "o_plus": lambda t: t["candidates"]["o_plus"],
}


def load_textcraft(items_path: str, score_paths: list[str]) -> dict:
    """{model: {population: table}} for main, success, empty_history, usefulness, secondary_get."""
    items = _read_jsonl(items_path)
    scores = _scores(score_paths)
    out = {}
    for m in MODELS:
        out[m] = {pop: _table([t for t in items if t["population"] == pop], scores, m, TC_ARMS, "pair_id", TC_FIELDS)
                  for pop in ("main", "success", "empty_history", "usefulness", "secondary_get")}
    return out
