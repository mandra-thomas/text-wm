"""RQ1, H1 and the secondary RQ2 conditions on the confirmatory items (decided 2026-11-18, after RQ2).

One scoring run over the verified `valid_unseen` items, sharded by game and saved per record:

- RQ1 (written predictions on every determined item, failures and successes): the few-shot models with
  example sets 1, 2 and 3 (history only) and set 1 with the full state; the Word2World checkpoints with
  the full state, as trained.
- RQ2 secondary (teacher-forced scores on the preregistered population): ΔL with the full state and per
  example set; the two mitigations (an instruction to ignore the goal; no goal); the extra-mention control,
  which names both objects in every prompt so only the goal role differs; and ΔL for the Word2World models.

`rq1_analysis` reads every shard once all are complete and at one commit, re-reads every written prediction
with the parser as it is now, and reports A, H1's two tests and the RQ2 secondary effects.
"""
from __future__ import annotations

import os
import statistics
from typing import Any

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import _asymmetry, asymmetry_A, asymmetry_ci, cluster_bootstrap, gen_says_success, with_partial_rule
from wf.data.transitions import determination_reason, main_test_items
from wf.rq2 import SOURCES, _mean_ci_p, complete_shards, load_items, primary_set, shard_items
from wf.runs import PILOT_A_MODELS, _effect, _finished, _false_success_by_reason, _init_run, _item_hash, _pilotA_item_sets
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.scoring.parser import parse_prediction
from wf.tools.parser_audit import parser_version
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, runs_root, write_json, write_status

FEWSHOT = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
MODELS = PILOT_A_MODELS + [(C.WM_FT_LLAMA, "w2w-llama3.1-8b", True)]
IGNORE_NOTE = ("(The agent's goal does not change what happens. Predict the observation from the history "
               "and the action alone.)")
ABSENT_LINE = "Your task is to: explore the room."

# arm: (goal line, prompt options). Goals: the item's own, the swapped one, none ("explore the room"), or
# a goal plus a sentence naming the other goal's object, so both objects are named in both prompts.
ARMS: dict[str, tuple[str, dict]] = {
    "aligned": ("aligned", {}),
    "swapped": ("swapped", {}),
    "aligned_full": ("aligned", {"full_state": True}),
    "swapped_full": ("swapped", {"full_state": True}),
    "aligned_set2": ("aligned", {"fewshot_set": 2}),
    "swapped_set2": ("swapped", {"fewshot_set": 2}),
    "aligned_set3": ("aligned", {"fewshot_set": 3}),
    "swapped_set3": ("swapped", {"fewshot_set": 3}),
    "aligned_ignore": ("aligned", {"note": IGNORE_NOTE}),
    "swapped_ignore": ("swapped", {"note": IGNORE_NOTE}),
    "absent": ("absent", {}),
    "aligned_plus": ("aligned_plus", {}),
    "swapped_plus": ("swapped_plus", {}),
}
FEWSHOT_SCORE_ARMS = tuple(ARMS)
FEWSHOT_GEN_ARMS = ("aligned", "aligned_full", "aligned_set2", "aligned_set3")
FT_SCORE_ARMS = ("aligned", "swapped")
FT_GEN_ARMS = ("aligned",)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def mention_sentence(obj: str) -> str:
    return f"There is {_article(obj)} {obj} somewhere in the room."


def goal_line(t: dict, goal: str) -> str:
    """The task line an arm shows. The plus lines name the other goal's object after the goal itself."""
    aligned, swapped = t["task_line"], t.get("swapped_task_line")
    if goal == "aligned":
        return aligned
    if goal == "absent":
        return ABSENT_LINE
    if not swapped or not t.get("swapped_goal_object") or not t.get("goal_object"):
        raise ValueError(f"{t['id']}: arm {goal} needs a swapped goal")  # would quietly score the aligned goal
    if goal == "swapped":
        return swapped
    if goal == "aligned_plus":
        return f"{aligned} {mention_sentence(t['swapped_goal_object'])}"
    if goal == "swapped_plus":
        return f"{swapped} {mention_sentence(t['goal_object'])}"
    raise ValueError(f"unknown goal {goal!r}")


def arm_messages(t: dict, arm: str, ft: bool = False) -> list[dict[str, str]]:
    goal, opts = ARMS[arm]
    if ft:
        if opts or goal not in ("aligned", "swapped"):
            raise ValueError(f"the fine-tuned prompt takes only the aligned and swapped goals, not {arm}")
        return build_messages(t, goal, ft_prompt=True)
    return build_messages(t, "aligned", task_line=goal_line(t, goal), **opts)


def rq1_sets(transitions: list[dict]) -> dict[str, list[dict]]:
    """RQ1's determined items (failures and successes, not at a shared spot) and RQ2's population."""
    sets = _pilotA_item_sets(transitions)
    return {"failures": main_test_items(sets["eligible"]), "successes": sets["successes"], "primary": primary_set(transitions)}


# ---------------- scoring ----------------


def run_rq1_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    if problems:
        write_status(run_id, f"collection not finished cleanly: {problems}", ok=False)
        return 1
    unverified = {s: why for s in sources for ok, why in [verified(s)] if not ok}
    if unverified:
        write_status(run_id, f"collection not verified: {unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(runs_root() / s / "transitions.jsonl" for s in sources))
    sets = rq1_sets(load_items(sources))
    gen_items = shard_items(sets["failures"] + sets["successes"], i, n)
    score_items = shard_items(sets["primary"], i, n)
    hashes = {t["id"]: _item_hash(t) for t in gen_items + score_items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    saved_gens = read_jsonl(d / "generations.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved + saved_gens):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    gen_done = {(g["model"], g["transition_id"], g["condition"]) for g in saved_gens}
    for model_name, tag, ft in [m for m in MODELS if m[1] in cfg.get("models", [m[1] for m in MODELS])]:
        score_arms, gen_arms = (FT_SCORE_ARMS, FT_GEN_ARMS) if ft else (FEWSHOT_SCORE_ARMS, FEWSHOT_GEN_ARMS)
        todo_s = [(t, a) for t in score_items for a in score_arms if (tag, t["id"], a) not in done]
        todo_g = [(t, a) for t in gen_items for a in gen_arms if (tag, t["id"], a) not in gen_done]
        if not todo_s and not todo_g:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, arm in todo_s:
            sc = scorer.score_pair(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm, ft),
                                   o_plus=t["candidates"]["o_plus"], o_minus=t["candidates"]["o_minus"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}])
        for t, arm in todo_g:
            text = scorer.free_generate(arm_messages(t, arm, ft))
            append_jsonl(d / "generations.jsonl", [{
                "model": tag, "transition_id": t["id"], "condition": arm, "game_id": t["game_id"], "generated": text,
                "generated_class": parse_prediction(text, t.get("goal_object"), t["action"]),
                "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(gen_items)} RQ1 items and {len(score_items)} RQ2 items, "
                         f"{len({t['game_id'] for t in gen_items + score_items})} games", ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def _as_aligned(gens: list[dict], arm: str) -> list[dict]:
    """One arm's predictions, relabelled so the RQ1 helpers (which read the aligned condition) take them."""
    return [{**g, "condition": "aligned"} for g in gens if g["condition"] == arm]


def _labelled(gens: list[dict], items: list[dict]) -> list[tuple[str, bool, bool]]:
    """(game, true failure, says success) per parseable prediction on these items."""
    meta = {t["id"]: t for t in items}
    out = []
    for g in gens:
        t = meta.get(g["transition_id"])
        said = gen_says_success(g["generated_class"]) if t else None
        if said is not None:
            out.append((t["game_id"], (t.get("true_obs") or "").strip() == "Nothing happens.", said))
    return out


def a_difference(first: list[dict], second: list[dict], items: list[dict], absolute: bool) -> dict[str, Any]:
    """A(first) - A(second) on the same items, or |A(first)| - |A(second)|, with a game-resampled CI.

    Both sets of predictions stay with their game in every resample, so the comparison is paired.
    """
    tagged = [(g, ("a", f, s)) for g, f, s in _labelled(first, items)] + [(g, ("b", f, s)) for g, f, s in _labelled(second, items)]

    def stat(rows: list[tuple[str, bool, bool]]) -> float:
        a = _asymmetry([(f, s) for tag, f, s in rows if tag == "a"])["A"]
        b = _asymmetry([(f, s) for tag, f, s in rows if tag == "b"])["A"]
        return (abs(a) - abs(b)) if absolute else (a - b)

    boot = cluster_bootstrap(stat, [x for _, x in tagged], [g for g, _ in tagged], seed=C.RNG_SEED)
    return {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]],
            "supported": boot["ci_low"] == boot["ci_low"] and boot["ci_low"] > 0}


def _rq1_block(aligned: list[dict], items: list[dict], failures: list[dict]) -> dict:
    """A and its CI, sensitivity rules and false-success rate by reason, for one arm's predictions."""
    return {
        **asymmetry_A(aligned, items),
        "A_ci": asymmetry_ci(aligned, items, seed=C.RNG_SEED),
        "A_sensitivity": {rule: asymmetry_A(with_partial_rule(aligned, rule), items)["A"]
                          for rule in ("exclude_partial", "partial_as_failure")},
        "false_success_by_reason": _false_success_by_reason(aligned, failures),
    }


def _did(scores: list[dict], items: list[dict], a: str, b: str, c: str, d: str) -> dict:
    """Mean over items of (L_a - L_b) - (L_c - L_d), with a game CI and a game-level permutation p."""
    by = {(s["transition_id"], s["condition"]): s["L"] for s in scores}
    diffs, games = [], []
    for t in items:
        v = [by.get((t["id"], x)) for x in (a, b, c, d)]
        if all(x is not None and x == x for x in v):
            diffs.append((v[0] - v[1]) - (v[2] - v[3]))
            games.append(t["game_id"])
    return _mean_ci_p(diffs, games)


def _mean_p(scores: list[dict], items: list[dict], arm: str) -> float:
    ids = {t["id"] for t in items}
    ps = [s["p_success"] for s in scores if s["condition"] == arm and s["transition_id"] in ids]
    return sum(ps) / len(ps) if ps else float("nan")


def _secondary(scores: list[dict], items: list[dict]) -> dict:
    """RQ2's secondary effects for one few-shot model, on the preregistered population."""
    take = [t for t in items if t["action"].startswith("take ")]
    rest = [t for t in items if not t["action"].startswith("take ")]
    step0 = [t for t in items if t.get("history_len", 0) == 0]
    later = [t for t in items if t.get("history_len", 0) > 0]
    per_set = {f"set{k}": _effect(scores, items, primary=a, control=b)
               for k, a, b in ((1, "aligned", "swapped"), (2, "aligned_set2", "swapped_set2"), (3, "aligned_set3", "swapped_set3"))}
    set_dl = [v["delta_L_nats"] for v in per_set.values()]

    def mention(subset: list[dict]) -> dict:
        return {
            # both prompts name both objects; only which one is the goal differs
            "goal_role_both_named": _effect(scores, subset, primary="aligned_plus", control="swapped_plus"),
            # naming the action's object in a sentence that is not the goal
            "mention_added_to_swapped": _effect(scores, subset, primary="swapped_plus", control="swapped"),
            "other_object_added_to_aligned": _effect(scores, subset, primary="aligned_plus", control="aligned"),
        }

    return {
        "per_example_set": per_set,
        "per_example_set_sd_nats": statistics.stdev(set_dl) if all(x == x for x in set_dl) else float("nan"),
        "full_state": _effect(scores, items, primary="aligned_full", control="swapped_full"),
        "ignore_goal": {
            "effect": _effect(scores, items, primary="aligned_ignore", control="swapped_ignore"),
            "reduction_in_delta_L": _did(scores, items, "aligned", "swapped", "aligned_ignore", "swapped_ignore"),
        },
        "no_goal": {
            "aligned_vs_none": _effect(scores, items, primary="aligned", control="absent"),
            "swapped_vs_none": _effect(scores, items, primary="swapped", control="absent"),
        },
        "mean_p_success": {arm: _mean_p(scores, items, arm) for arm in
                           ("aligned", "swapped", "absent", "aligned_ignore", "swapped_ignore", "aligned_full", "swapped_full")},
        "extra_mention": {"all": mention(items), "take": mention(take), "not_take": mention(rest),
                          "history_0": mention(step0), "history_1plus": mention(later),
                          "n_take": len(take), "n_not_take": len(rest)},
    }


def run_rq1_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False},
                  inputs=tuple(p / "scoring.jsonl" for p in folders if (p / "scoring.jsonl").exists()))
    if problems:
        write_status(run_id, "RQ1 waits for every scoring shard:\n" + "\n".join(problems), ok=False)
        return 1
    sets = rq1_sets(load_items(tuple(cfg.get("sources", SOURCES))))
    failures, successes, primary = sets["failures"], sets["successes"], sets["primary"]
    items = failures + successes
    meta = {t["id"]: t for t in items}
    scores = [s for p in folders for s in read_jsonl(p / "scoring.jsonl")]
    # every written prediction re-read with the parser as it is now (the version the audit grades)
    gens = [{**g, "generated_class": parse_prediction(g["generated"], meta[g["transition_id"]].get("goal_object"),
                                                      meta[g["transition_id"]]["action"])}
            for p in folders for g in read_jsonl(p / "generations.jsonl")]
    models = [m for m in MODELS if m[1] in cfg.get("models", [m[1] for m in MODELS])]
    wanted_g = {(tag, t["id"], a) for _, tag, ft in models for t in items for a in (FT_GEN_ARMS if ft else FEWSHOT_GEN_ARMS)}
    wanted_s = {(tag, t["id"], a) for _, tag, ft in models for t in primary for a in (FT_SCORE_ARMS if ft else FEWSHOT_SCORE_ARMS)}
    missing = (len(wanted_g - {(g["model"], g["transition_id"], g["condition"]) for g in gens}),
               len(wanted_s - {(s["model"], s["transition_id"], s["condition"]) for s in scores}))
    per_model: dict[str, dict] = {}
    for _, tag, ft in models:
        mg = [g for g in gens if g["model"] == tag]
        ms = [s for s in scores if s["model"] == tag]
        rep: dict[str, Any] = {"rq1_history_only" if not ft else "rq1_full_state_as_trained": _rq1_block(_as_aligned(mg, "aligned"), items, failures)}
        if ft:
            rep["delta_L_full_state"] = _effect(ms, primary)
        else:
            rep["rq1_full_state"] = _rq1_block(_as_aligned(mg, "aligned_full"), items, failures)
            per_set_A = {f"set{k}": asymmetry_A(_as_aligned(mg, arm), items)["A"]
                         for k, arm in ((1, "aligned"), (2, "aligned_set2"), (3, "aligned_set3"))}
            rep["A_per_example_set"] = per_set_A
            rep["A_sd_across_sets"] = statistics.stdev(per_set_A.values()) if all(x == x for x in per_set_A.values()) else float("nan")
            # H1, first prediction: the lean is larger from the history alone than with the full state
            rep["H1_history_vs_full_state"] = a_difference(_as_aligned(mg, "aligned"), _as_aligned(mg, "aligned_full"),
                                                           items, absolute=True)
            rep["H1_history_vs_full_state_signed"] = a_difference(_as_aligned(mg, "aligned"), _as_aligned(mg, "aligned_full"),
                                                                  items, absolute=False)
            rep["rq2_secondary"] = _secondary(ms, primary)
        per_model[tag] = rep
    qwen, llama = (_as_aligned([g for g in gens if g["model"] == m], "aligned") for m in FEWSHOT)
    report = {
        "items": {"n_failures": len(failures), "n_successes": len(successes), "n_primary": len(primary),
                  "n_games": len({t["game_id"] for t in items}), "missing_predictions": missing[0], "missing_scores": missing[1],
                  "parser_version": parser_version()},
        # H1, second prediction: Llama-3.1-8B leans more optimistic than Qwen3-8B (history only, set 1)
        "H1_llama_minus_qwen": a_difference(llama, qwen, items, absolute=False),
        "per_model": per_model,
    }
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "RQ1 and RQ2 secondary", report)
    lines = []
    for tag, rep in per_model.items():
        key = "rq1_history_only" if "rq1_history_only" in rep else "rq1_full_state_as_trained"
        lines.append(f"{tag}: A={rep[key]['A']:.3f} CI=[{rep[key]['A_ci']['ci_low']:.3f}, {rep[key]['A_ci']['ci_high']:.3f}]"
                     + (f" full-state A={rep['rq1_full_state']['A']:.3f} H1(|A| hist - full)={rep['H1_history_vs_full_state']['estimate']:.3f}"
                        f" CI={rep['H1_history_vs_full_state']['ci']}" if "rq1_full_state" in rep else ""))
    h1b = report["H1_llama_minus_qwen"]
    ok = not any(missing)
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines)
                 + f"\nH1 Llama - Qwen A={h1b['estimate']:.3f} CI={h1b['ci']} supported={h1b['supported']}",
                 ok=False, reported="RQ1 report" if ok else None)
    return 0 if ok else 1
