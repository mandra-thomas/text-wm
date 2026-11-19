"""Exploratory follow-up to RQ1/RQ2 (decided 2026-11-19, after the RQ1 results; notes/deviations.md).

Three questions, none preregistered:

1. Object-free goal. Is the goal effect there when the goal describes the action's object without naming
   it? The aligned and swapped task lines are kept, with the goal object's name replaced by a fixed
   description ("put some thick drinking vessel with a handle for hot drinks on cabinet."). If the effect
   survives, the goal acts through its meaning; if it vanishes, through the object's name.
   A manipulation check asks each model to name the object from its description.
2. Working actions. Does the goal help tell actions that work from actions that fail, or shift both alike?
   Determined successes are scored under the aligned, swapped and neutral goals and compared with the
   failures already scored in rq1_score.
3. Free splits of rq1_score: whether the action's object was already named in an earlier observation,
   and why the action fails.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from wf import constants as C
from wf import rq1
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap
from wf.rq2 import SOURCES, complete_shards, load_items, shard_items
from wf.runs import _effect, _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

# Fixed before any object-free score existed. Each names no ALFWorld object or receptacle, and no part of
# its own object's name (OWN_PARTS); tests/test_rq1x.py checks both.
DESCRIPTIONS = {
    "alarmclock": "timepiece that rings in the morning to wake you up",
    "apple": "round crunchy fruit that grows on trees",
    "baseballbat": "wooden club swung to hit a ball in a ballgame",
    "basketball": "large orange sphere bounced and thrown through a hoop",
    "book": "bound volume of printed pages to read",
    "bowl": "deep round dish for soup or cereal",
    "box": "cardboard container with flaps",
    "bread": "baked loaf made from flour",
    "butterknife": "blunt table utensil for spreading",
    "candle": "wax stick with a wick that you light",
    "cd": "shiny disc that stores music or data",
    "cellphone": "handheld device for making calls and sending texts",
    "cloth": "square piece of fabric for wiping surfaces",
    "creditcard": "plastic payment rectangle issued by a bank",
    "cup": "small drinking vessel without a handle",
    "dishsponge": "porous scrubbing pad for washing up",
    "egg": "oval item with a shell, laid by a hen",
    "fork": "eating utensil with prongs",
    "glassbottle": "clear see-through vessel with a narrow neck",
    "handtowel": "small absorbent fabric for drying fingers and palms after washing",
    "kettle": "spouted vessel for boiling water for tea",
    "keychain": "small ring carried in a pocket to hold the things that unlock doors",
    "knife": "large sharp kitchen blade for chopping",
    "ladle": "long-handled deep scoop for serving soup",
    "laptop": "portable folding computer",
    "lettuce": "leafy green vegetable used in salads",
    "mug": "thick drinking vessel with a handle for hot drinks",
    "newspaper": "daily printed publication of current events",
    "pan": "flat shallow cooking vessel for frying",
    "pen": "writing instrument that uses ink",
    "pencil": "wooden writing stick with graphite and an eraser",
    "peppershaker": "container with holes for sprinkling black spice",
    "pillow": "soft cushion you rest your head on while sleeping",
    "plate": "flat round dish you eat a meal from",
    "plunger": "rubber suction tool on a stick for clearing clogged drains",
    "pot": "deep cooking vessel for boiling",
    "potato": "starchy underground tuber",
    "remotecontrol": "handheld clicker for changing channels",
    "saltshaker": "container with holes for sprinkling white seasoning",
    "scrubbrush": "stiff-bristled cleaning implement",
    "soapbar": "small solid block you lather when washing",
    "soapbottle": "pump dispenser of liquid cleanser",
    "spatula": "flat flexible kitchen utensil for flipping food",
    "spoon": "eating utensil with a small rounded scoop end",
    "spraybottle": "trigger-operated mister of cleaning liquid",
    "statue": "small sculpted figure for decoration",
    "teddybear": "stuffed toy animal for children",
    "tennisracket": "strung frame for hitting a fuzzy ball over a net",
    "tissuebox": "carton of soft facial wipes",
    "toiletpaper": "soft absorbent sheets on a spindle used in the bathroom",
    "tomato": "soft red fruit used to make ketchup and sauces",
    "vase": "decorative vessel for holding cut flowers",
    "watch": "timepiece worn on the wrist",
    "wateringcan": "spouted container for giving plants a drink",
    "winebottle": "tall vessel that holds an alcoholic drink made from grapes",
}
# Words inside compound names, so a description cannot reuse half of its own object's name.
OWN_PARTS = {
    "alarmclock": ("alarm", "clock"), "baseballbat": ("baseball", "bat"), "basketball": ("basket", "ball"),
    "butterknife": ("butter", "knife"), "cellphone": ("cell", "phone"), "creditcard": ("credit", "card"),
    "dishsponge": ("dish", "sponge"), "glassbottle": ("glass", "bottle"), "handtowel": ("hand", "towel"),
    "keychain": ("key", "keys", "chain"), "newspaper": ("news", "paper"), "peppershaker": ("pepper", "shaker"),
    "remotecontrol": ("remote", "control"), "saltshaker": ("salt", "shaker"), "scrubbrush": ("scrub", "brush"),
    "soapbar": ("soap", "bar"), "soapbottle": ("soap", "bottle"), "spraybottle": ("spray", "bottle"),
    "teddybear": ("teddy", "bear"), "tennisracket": ("tennis", "racket"), "tissuebox": ("tissue", "box"),
    "toiletpaper": ("toilet", "paper"), "wateringcan": ("watering", "can"), "winebottle": ("wine", "bottle"),
}
SCORE_ARMS = ("aligned_desc", "swapped_desc")  # on RQ2's population
SUCCESS_ARMS = ("aligned", "swapped", "absent")  # on the determined successes
CHECK_PROMPT = 'Which household object is described here: "{desc}"? Answer with just the name of the object.'


def describe_line(line: str, obj: str) -> str:
    """The task line with its one mention of `obj` replaced by the object's description."""
    if obj not in DESCRIPTIONS:
        raise KeyError(f"no description for {obj!r}")
    new, n = re.subn(rf"\b{re.escape(obj)}\b", DESCRIPTIONS[obj], line)
    if n != 1:
        raise ValueError(f"{line!r} names {obj!r} {n} times, expected once")
    return new


def desc_ok(t: dict) -> bool:
    return t.get("goal_object") in DESCRIPTIONS and t.get("swapped_goal_object") in DESCRIPTIONS and bool(t.get("swapped_task_line"))


def arm_messages(t: dict, arm: str) -> list[dict[str, str]]:
    if arm == "aligned_desc":
        return build_messages(t, "aligned", task_line=describe_line(t["task_line"], t["goal_object"]))
    if arm == "swapped_desc":
        return build_messages(t, "aligned", task_line=describe_line(t["swapped_task_line"], t["swapped_goal_object"]))
    return rq1.arm_messages(t, arm)


def check_says(obj: str, answer: str) -> bool:
    squash = lambda s: re.sub(r"[^a-z]", "", s.lower())
    return squash(obj) in squash(answer)


def rq1x_sets(transitions: list[dict]) -> dict[str, list[dict]]:
    sets = rq1.rq1_sets(transitions)
    return {"primary": [t for t in sets["primary"] if desc_ok(t)],
            "successes": [t for t in sets["successes"] if t.get("swapped_task_line") and t.get("candidates")
                          and t.get("goal_object") and t.get("swapped_goal_object")],
            "n_primary_all": len(sets["primary"]), "n_successes_all": len(sets["successes"])}


def run_rq1x_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    unverified = [s for s in sources if not verified(s)[0]]
    if problems or unverified:
        write_status(run_id, f"collection not finished or not verified: {problems + unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=())
    sets = rq1x_sets(load_items(sources))
    jobs = [(t, a) for t in shard_items(sets["primary"], i, n) for a in SCORE_ARMS]
    jobs += [(t, a) for t in shard_items(sets["successes"], i, n) for a in SUCCESS_ARMS]
    hashes = {t["id"]: _item_hash(t) for t, _ in jobs}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    checks_done = {(c["model"], c["object"]) for c in (read_jsonl(d / "checks.jsonl") if cfg.get("resume") else [])}
    for model_name, tag, ft in [m for m in rq1.MODELS if m[1] in cfg.get("models", rq1.FEWSHOT)]:
        todo = [(t, a) for t, a in jobs if (tag, t["id"], a) not in done]
        todo_checks = [o for o in sorted(DESCRIPTIONS) if i == 0 and (tag, o) not in checks_done]
        if not todo and not todo_checks:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, arm in todo:
            sc = scorer.score_pair(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm),
                                   o_plus=t["candidates"]["o_plus"], o_minus=t["candidates"]["o_minus"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}])
        for obj in todo_checks:  # manipulation check, once per model, in shard 0
            ans = scorer.free_generate([{"role": "user", "content": CHECK_PROMPT.format(desc=DESCRIPTIONS[obj])}], max_new_tokens=16)
            append_jsonl(d / "checks.jsonl", [{"model": tag, "object": obj, "answer": ans, "correct": check_says(obj, ans)}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} scores", ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def _auc(rows: list[tuple[bool, float]]) -> float:
    """P(L of a random success > L of a random failure), ties half: how well L tells them apart."""
    pos = sorted(x for s, x in rows if s)
    neg = sorted(x for s, x in rows if not s)
    if not pos or not neg:
        return float("nan")
    import bisect
    total = sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) for x in pos)
    return total / (len(pos) * len(neg))


def _auc_ci(by: dict, items: list[dict], arm: str) -> dict:
    rows = [(t["game_id"], ((t.get("true_obs") or "").strip() != "Nothing happens.", by[(t["id"], arm)]))
            for t in items if (t["id"], arm) in by]
    boot = cluster_bootstrap(_auc, [r for _, r in rows], [g for g, _ in rows], seed=C.RNG_SEED)
    return {"auc": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]], "n": len(rows)}


def _auc_diff(by: dict, items: list[dict], a: str, b: str) -> dict:
    rows = [(t["game_id"], ((t.get("true_obs") or "").strip() != "Nothing happens.", by[(t["id"], a)], by[(t["id"], b)]))
            for t in items if (t["id"], a) in by and (t["id"], b) in by]
    stat = lambda xs: _auc([(s, x) for s, x, _ in xs]) - _auc([(s, y) for s, _, y in xs])
    boot = cluster_bootstrap(stat, [r for _, r in rows], [g for g, _ in rows], seed=C.RNG_SEED)
    return {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]], "n": len(rows)}


def _named_before(t: dict) -> bool:
    """Whether an earlier observation in the history already names the action's object."""
    obj = t.get("goal_object") or ""
    return any(re.search(rf"\b{re.escape(obj)}\b", obs or "") for _, obs in (t.get("history_pairs") or []))


def _three_way(s: list[dict], items: list[dict]) -> dict:
    """ΔL and its two main parts from the extra-mention control, on a subset."""
    return {"n": len(items),
            "delta_L": _effect(s, items),
            "mention": _effect(s, items, primary="swapped_plus", control="swapped"),
            "goal_role_both_named": _effect(s, items, primary="aligned_plus", control="swapped_plus"),
            "object_free_goal": _effect(s, items, primary="aligned_desc", control="swapped_desc")}


def run_rq1x_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    base_folders, base_problems = complete_shards(cfg["base"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems or base_problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems + base_problems), ok=False)
        return 1
    sets = rq1x_sets(load_items(tuple(cfg.get("sources", SOURCES))))
    primary, successes = sets["primary"], sets["successes"]
    scores = [s for p in folders + base_folders for s in read_jsonl(p / "scoring.jsonl")]
    checks = [c for p in folders for c in (read_jsonl(p / "checks.jsonl") if (p / "checks.jsonl").exists() else [])]
    take = [t for t in primary if t["action"].startswith("take ")]
    rest = [t for t in primary if not t["action"].startswith("take ")]
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_primary_all": sets["n_primary_all"],
                                        "n_successes": len(successes), "n_successes_all": sets["n_successes_all"]}}
    for tag in cfg.get("models", rq1.FEWSHOT):
        s = [x for x in scores if x["model"] == tag]
        ok = {c["object"] for c in checks if c["model"] == tag and c["correct"]}
        understood = [t for t in primary if t["goal_object"] in ok and t["swapped_goal_object"] in ok]
        by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
        both = primary + successes
        rep: dict[str, Any] = {
            "manipulation_check": {"correct": sorted(ok), "wrong": {c["object"]: c["answer"] for c in checks
                                                                    if c["model"] == tag and not c["correct"]}},
            "named_goal": _effect(s, primary),
            "object_free_goal": _effect(s, primary, primary="aligned_desc", control="swapped_desc"),
            "object_free_goal_check_passed": {"n_items": len(understood),
                                              **_effect(s, understood, primary="aligned_desc", control="swapped_desc")},
            "naming_the_goal_object": _effect(s, primary, primary="aligned", control="aligned_desc"),
            "naming_the_swapped_object": _effect(s, primary, primary="swapped", control="swapped_desc"),
            "object_free_aligned_vs_neutral": _effect(s, primary, primary="aligned_desc", control="absent"),
            "object_free_swapped_vs_neutral": _effect(s, primary, primary="swapped_desc", control="absent"),
            "object_free_by_action": {"take": _effect(s, take, primary="aligned_desc", control="swapped_desc"),
                                      "not_take": _effect(s, rest, primary="aligned_desc", control="swapped_desc")},
            "object_free_by_history": {
                "history_0": _effect(s, [t for t in primary if t.get("history_len", 0) == 0], primary="aligned_desc", control="swapped_desc"),
                "history_1plus": _effect(s, [t for t in primary if t.get("history_len", 0) > 0], primary="aligned_desc", control="swapped_desc")},
            "working_actions": {
                "goal_effect_on_successes": _effect(s, successes, require_agnostic=False),
                "goal_effect_on_failures": _effect(s, primary),
                "auc": {arm: _auc_ci(by, both, arm) for arm in SUCCESS_ARMS},
                "auc_aligned_minus_neutral": _auc_diff(by, both, "aligned", "absent"),
                "auc_aligned_minus_swapped": _auc_diff(by, both, "aligned", "swapped"),
                "mean_L": {arm: {kind: (lambda v: sum(v) / len(v) if v else float("nan"))(
                    [by[(t["id"], arm)] for t in its if (t["id"], arm) in by]) for kind, its in (("successes", successes), ("failures", primary))}
                    for arm in SUCCESS_ARMS},
            },
            "object_named_before": {"yes": _three_way(s, [t for t in primary if _named_before(t)]),
                                    "no": _three_way(s, [t for t in primary if not _named_before(t)])},
        }
        reasons = defaultdict(list)
        for t in primary:
            reasons[t.get("determination_reason") or "unknown"].append(t)
        rep["by_failure_reason"] = {r: _three_way(s, its) for r, its in sorted(reasons.items()) if len(its) >= 50}
        report[tag] = rep
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "RQ1/RQ2 follow-up (exploratory)", report)
    lines = [f"{tag}: named ΔL={report[tag]['named_goal']['delta_L_nats']:.2f}, object-free ΔL="
             f"{report[tag]['object_free_goal']['delta_L_nats']:.2f} {report[tag]['object_free_goal']['delta_L_ci']}"
             for tag in cfg.get("models", rq1.FEWSHOT)]
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines), ok=False, reported="follow-up report")
    return 0
