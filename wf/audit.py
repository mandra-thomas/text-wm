"""Audit analyses on existing items and scores (decided 2026-11-20, documentation audit, part C; CPU only).

Nothing here is preregistered and nothing is scored anew. `audit_analysis` reads the confirmatory items
(rq12_transitions_s0..s2) and the score files already on disk, and reports the numbers the documentation audit
asked for:

- C1 `exclusion_flow`: every collected item, followed through each selection stage to RQ2's primary set, with
  the number dropped at each stage and why.
- C2 `seen_without_object`: the items whose failure is decided by a receptacle already seen without the object
  (788 on the confirmatory data). Some differ from a visible object only by instance number ("take mug 1 from
  shelf 1" while shelf 1 visibly holds mug 2). Counted, and the goal effect reported with and without them.
- C3 `size_yes_share`: for each size, the share of items scored Yes (L > 0 on the yes/no question) under each
  task, next to the mean of s = sigmoid(L), which is a mean probability, not a share of answers.
- C4 `ratios`: game-clustered intervals for the headline ratios (object-free goal kept, thinking-mode
  reduction, the ignore-goal instruction, the full-layout reduction), computed inside each bootstrap resample
  (wf.mech._ratio_ci).
- C5 `first_token`: how much next-token mass the two first tokens cover (first_mass), the first-token
  distribution per model and condition, and how many top-candidate comparisons are exact ties.
- C6 `items_per_game`: items per game and near-duplicate items (same game, action and deciding facts).
- C7 `twin_plan_wins`: failures "take <goal> from <target>" left without a success twin because the scripted
  plan that would make the take work also completes the task (the game ends, so the take is never tried).

Inputs are configurable (`inputs:` in wf/config/audit_analysis.yaml): each is a run name under runs/ or an
absolute path to a run folder in another checkout. A part whose inputs are missing is reported as not computed,
with the reason; the others still run.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap
from wf.data.goals import parse_task_line
from wf.data.transitions import exclusion_counts, history_facts
from wf.mech import _ratio_ci
from wf.provenance import provenance, shard_inputs
from wf.rq2 import SOURCES, complete_shards, load_items, primary_set
from wf.runs import _effect, _init_run, _pilotA_item_sets
from wf.util import read_jsonl, require_filed, runs_root, write_json, write_status

FEWSHOT = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
NOOP = "Nothing happens."
_SIMPLE_PUT = re.compile(r"^(?:put|find) (?:a|an|some) (?P<obj>[a-z]+)(?: and put it)? (?:in|on) (?:the )?(?P<rec>[a-z]+)\.?$")


def _failed(t: dict) -> bool:
    return (t.get("true_obs") or "").strip() == NOOP


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_ci(rows: list[tuple[str, float]]) -> dict[str, Any]:
    boot = cluster_bootstrap(_mean, [x for _, x in rows], [g for g, _ in rows], seed=C.RNG_SEED)
    return {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]], "n": len(rows),
            "n_games": len({g for g, _ in rows})}


def _type(instance: str) -> str:
    return instance.rsplit(" ", 1)[0]


# ---------------- C1: exclusion flow ----------------


def exclusion_flow(transitions: list[dict]) -> dict[str, Any]:
    """Each stage keeps the items that pass it; `dropped` says how many left there and, where recorded, why."""
    stages: list[tuple[str, Callable[[dict], bool]]] = [
        ("true failure (reply was \"Nothing happens.\")", _failed),
        ("environment reply parseable", lambda t: t.get("outcome_class") != "unparseable"),
        ("failure determined by the history", lambda t: bool(t.get("context_determined"))),
        ("success twin attached", lambda t: t.get("candidates") is not None),
        ("swapped goal available", lambda t: bool(t.get("swapped_goal_object"))),
        ("not at a shared spot", lambda t: not t.get("shared_location")),
        ("action names the goal object", lambda t: bool(t.get("targets_goal_object"))),
        ("history does not act on either goal object", lambda t: bool(t.get("history_goal_agnostic"))),
    ]
    left = list(transitions)
    flow = [{"stage": "collected", "remaining": len(left), "games": len({t["game_id"] for t in left})}]
    for name, keep in stages:
        kept = [t for t in left if keep(t)]
        gone = [t for t in left if not keep(t)]
        reasons = Counter(t.get("exclusion_reason") or "-" for t in gone)
        flow.append({"stage": name, "dropped": len(gone), "remaining": len(kept),
                     "games": len({t["game_id"] for t in kept}), "dropped_by_recorded_reason": dict(reasons)})
        left = kept
    games_all = {t["game_id"] for t in transitions}
    no_swap_games = sorted(g for g in games_all
                           if not any(t.get("swapped_goal_object") for t in transitions if t["game_id"] == g))
    sets = _pilotA_item_sets(transitions)
    primary = primary_set(transitions)
    return {
        "flow": flow,
        "primary_set_size": len(primary),
        "flow_matches_primary_set": {t["id"] for t in left} == {t["id"] for t in primary},
        "games_with_no_swap_object_at_all": no_swap_games,
        "collector_exclusion_counts": exclusion_counts(transitions),
        "rq1_items": {"determined_failures_not_shared": len([t for t in sets["eligible"] if not t.get("shared_location")]),
                      "determined_successes_not_shared": len(sets["successes"]),
                      "determined_successes_all": len(sets["all_successes"])},
        "true_successes_collected": sum(1 for t in transitions if not _failed(t)),
    }


# ---------------- C2: receptacle seen without the object ----------------


def seen_without(t: dict) -> tuple[bool, list[str]]:
    """(the failure is a take from a receptacle last seen without the object, other instances seen there)."""
    m = re.fullmatch(r"take (.+) from (.+)", t.get("action") or "")
    if not m or t.get("determination_reason") != "contents_or_open_state":
        return False, []
    facts = history_facts(t["history_text"])
    seen = facts.contents.get(m[2])
    if seen is None or m[1] in seen:
        return False, []
    return True, sorted(i for i in seen if _type(i) == _type(m[1]))


def seen_without_split(primary: list[dict], scores_by_model: dict[str, list[dict]]) -> dict[str, Any]:
    flagged = [(t, others) for t in primary for ok, others in [seen_without(t)] if ok]
    items = [t for t, _ in flagged]
    instance = [t for t, others in flagged if others]
    rest = [t for t, others in flagged if not others]
    out: dict[str, Any] = {
        "n_seen_without_object": len(items), "n_instance_number_only": len(instance), "n_without_them": len(rest),
        "examples": [{"action": t["action"], "visible_instances": others} for t, others in flagged if others][:10],
        "definition": "a take whose receptacle the history last showed without that instance; 'instance number "
                      "only' means another instance of the same object type was visible there",
    }
    for m, s in scores_by_model.items():
        out[m] = {name: _short(_effect(s, its)) for name, its in
                  (("all", items), ("instance_number_only", instance), ("without_instance_number_items", rest))}
    return out


def _short(e: dict) -> dict:
    return {k: e[k] for k in ("n_items", "n_games", "delta_L_nats", "delta_L_ci", "delta_L_first_nats",
                              "delta_L_first_ci", "delta_p_pp", "delta_p_ci")}


# ---------------- C3: yes-share next to mean s ----------------


def yes_share(scores: list[dict], items: list[dict], arm: str) -> dict[str, Any]:
    """Share of items answered Yes (L > 0) and the mean of s = sigmoid(L), each with a game-clustered CI."""
    meta = {t["id"]: t["game_id"] for t in items}
    rows = [(meta[s["transition_id"]], s) for s in scores
            if s["condition"] == arm and s["transition_id"] in meta and s["L"] == s["L"]]
    return {"share_yes": _mean_ci([(g, 1.0 if s["L"] > 0 else 0.0) for g, s in rows]),
            "mean_s": _mean_ci([(g, s["p_success"]) for g, s in rows])}


def size_yes_shares(size_scores: dict[str, list[dict]], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    out = {}
    for tag, s in size_scores.items():
        out[tag] = {"failures": {a: yes_share(s, primary, a) for a in ("yn_aligned", "yn_swapped", "yn_absent")},
                    "successes": {a: yes_share(s, successes, a) for a in ("yn_aligned", "yn_absent")}}
    return out


# ---------------- C4: headline ratios ----------------


def _diffs(by: dict, t: dict, a: str, b: str) -> float | None:
    x, y = by.get((t["id"], a)), by.get((t["id"], b))
    return None if x is None or y is None else x - y


def ratio_of_effects(items: list[dict], num: tuple[dict, str, str], den: tuple[dict, str, str]) -> dict | None:
    """mean(num diff) / mean(den diff) over items with both, with the interval from the same game resamples."""
    rows = []
    for t in items:
        a, b = _diffs(num[0], t, num[1], num[2]), _diffs(den[0], t, den[1], den[2])
        if a is not None and b is not None:
            rows.append((t["game_id"], (a, b)))

    def stat(xs):
        d = sum(b for _, b in xs)
        return sum(a for a, _ in xs) / d if xs and d else float("nan")

    out = _ratio_ci(rows, stat)
    if out:
        out["numerator_mean"] = _mean([a for _, (a, _) in rows])
        out["denominator_mean"] = _mean([b for _, (_, b) in rows])
    return out


def _by(scores: list[dict], model: str) -> dict[tuple[str, str], float]:
    return {(s["transition_id"], s["condition"]): s["L"] for s in scores if s.get("model", model) == model and s["L"] == s["L"]}


def headline_ratios(primary_desc: list[dict], primary: list[dict], rq1: list[dict], rq1x: list[dict] | None,
                    checks: list[dict], thinking: list[dict] | None, thinking_items: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in FEWSHOT:
        named = _by(rq1, m)
        rep: dict[str, Any] = {
            "ignore_note_share_removed": _one_minus(ratio_of_effects(
                primary, (named, "aligned_ignore", "swapped_ignore"), (named, "aligned", "swapped"))),
            "full_layout_share_removed": _one_minus(ratio_of_effects(
                primary, (named, "aligned_full", "swapped_full"), (named, "aligned", "swapped"))),
        }
        if rq1x is not None:
            desc = _by(rq1x, m)
            ok = {c["object"] for c in checks if c["model"] == m and c["correct"]}
            passed = [t for t in primary_desc if t["goal_object"] in ok and t["swapped_goal_object"] in ok]
            rep["object_free_kept"] = ratio_of_effects(primary_desc, (desc, "aligned_desc", "swapped_desc"),
                                                       (named, "aligned", "swapped"))
            rep["object_free_kept_check_passed"] = ratio_of_effects(passed, (desc, "aligned_desc", "swapped_desc"),
                                                                    (named, "aligned", "swapped"))
        out[m] = rep
    if thinking is not None:
        think = {(r["transition_id"], r["condition"]): r["L"] for r in thinking if r.get("L") is not None and r["L"] == r["L"]}
        off = _by(rq1, "qwen3-8b-fewshot")
        r = ratio_of_effects(thinking_items, (think, "aligned", "swapped"), (off, "aligned", "swapped"))
        out["thinking_qwen3-8b"] = {"ratio_with_to_without_reasoning": r, "share_removed": _one_minus(r),
                                    "note": "paired on items with an L after reasoning (finished answers); the "
                                            "4.01 and 6.06 in the notes are each averaged over their own items"}
    return out


def _one_minus(r: dict | None) -> dict | None:
    if not r:
        return r
    return {**r, "estimate": 1 - r["estimate"], "ci": [1 - r["ci"][1], 1 - r["ci"][0]],
            "reads": "share of the plain effect removed, 1 - ratio"}


# ---------------- C5: first token and ties ----------------


def first_token_report(scores: list[dict]) -> dict[str, Any]:
    """Per model and condition: first_mass quantiles and the commonest first tokens of each candidate."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in scores:
        groups[(s.get("model", "?"), s["condition"])].append(s)
    out: dict[str, Any] = {}
    for (m, cond), rows in sorted(groups.items()):
        mass = sorted(s["first_mass"] for s in rows if s.get("first_mass") is not None and s["first_mass"] == s["first_mass"])
        q = lambda p: mass[min(len(mass) - 1, int(p * len(mass)))] if mass else float("nan")
        out.setdefault(m, {})[cond] = {
            "n": len(rows),
            "first_mass": {"median": statistics.median(mass) if mass else float("nan"), "mean": _mean(mass),
                           "p05": q(0.05), "p25": q(0.25), "share_below_0.5": _mean([float(x < 0.5) for x in mass]),
                           "share_below_0.9": _mean([float(x < 0.9) for x in mass])},
            "first_token_plus": dict(Counter(s.get("first_token_plus", "") for s in rows).most_common(8)),
            "first_token_minus": dict(Counter(s.get("first_token_minus", "") for s in rows).most_common(4)),
            "first_token_plus_starts_with_You": _mean([float((s.get("first_token_plus") or "").strip() == "You") for s in rows]),
        }
    return out


def top_candidate_ties(by: dict, cands: list[dict], arm: str) -> dict[str, Any]:
    """At states with a working and a failing candidate scored: how often the top score is shared.

    rq1v's top-candidate rate breaks a tie in favour of the working action (`max` over (score, works)), so a
    tie between a working and a failing candidate is the case where that choice matters.
    """
    states = defaultdict(list)
    for t in cands:
        states[(t["game_id"], t.get("state_idx"), tuple(t.get("prefix_actions") or []))].append(t)
    n = tied = mixed = 0
    for ts in states.values():
        sc = [(by[(t["id"], arm)], not _failed(t)) for t in ts if (t["id"], arm) in by]
        if not (any(w for _, w in sc) and not all(w for _, w in sc)):
            continue
        n += 1
        top = max(x for x, _ in sc)
        at_top = [w for x, w in sc if x == top]
        tied += len(at_top) > 1
        mixed += len(set(at_top)) > 1
    return {"n_states": n, "top_score_tied": tied, "tie_between_working_and_failing": mixed}


# ---------------- C6: items per game, near-duplicates ----------------


def _facts_key(t: dict) -> tuple:
    f = history_facts(t["history_text"])
    return (t["game_id"], t["action"], t.get("swapped_goal_object"), f.location, f.holding, tuple(sorted(f.opened)),
            tuple(sorted(f.closed)), tuple(sorted((k, tuple(sorted(v))) for k, v in f.contents.items())))


def items_per_game(primary: list[dict]) -> dict[str, Any]:
    per = Counter(t["game_id"] for t in primary)
    counts = sorted(per.values())
    top_decile = sorted(counts, reverse=True)[: max(1, len(counts) // 10)]
    exact = Counter((t["game_id"], t["history_text"], t["action"]) for t in primary)
    near = Counter(_facts_key(t) for t in primary)
    same_action = Counter((t["game_id"], t["action"]) for t in primary)
    return {
        "n_games": len(per), "n_items": len(primary),
        "items_per_game": {"min": counts[0] if counts else None, "median": statistics.median(counts) if counts else None,
                           "mean": _mean(counts), "max": counts[-1] if counts else None},
        "share_of_items_in_top_10pct_of_games": sum(top_decile) / len(primary) if primary else float("nan"),
        "largest_games": dict(per.most_common(5)),
        "exact_duplicates": sum(v - 1 for v in exact.values() if v > 1),
        "near_duplicates": {
            "definition": "same game, action and swapped goal, and the same facts read from the history (location, "
                          "held object, open and closed receptacles, last-seen contents); typically the agent "
                          "repeating a step that changed nothing",
            "groups": sum(1 for v in near.values() if v > 1),
            "items_in_groups": sum(v for v in near.values() if v > 1),
            "extra_items": sum(v - 1 for v in near.values() if v > 1),
        },
        "same_game_and_action": {"groups": sum(1 for v in same_action.values() if v > 1),
                                 "extra_items": sum(v - 1 for v in same_action.values() if v > 1)},
    }


# ---------------- C7: twin plan completes the task ----------------


def twin_plan_wins(t: dict) -> bool:
    """A failed "take <goal> from <target>" whose success-twin plan would finish a simple put task.

    The twin plan (wf.data.candidates._setup_plan) for a take from a receptacle that does not hold the object
    carries the object there and moves it in; when that receptacle is the task's target, the move completes the
    task, the game ends and the take is never tried, so the item gets no twin. Identified from recorded fields;
    the collector did not record that the game ended.
    """
    if not _failed(t) or t.get("exclusion_reason") != "no_success_twin":
        return False
    m = re.fullmatch(r"take (.+) from (.+)", t.get("action") or "")
    task = (t.get("task_line") or "").split("Your task is to:", 1)[-1].strip().lower()
    spec = _SIMPLE_PUT.match(task)
    if not m or not spec:
        return False
    obj, rec = m[1], m[2]
    locations = (t.get("scene_locations") or {}).get(obj) or []
    return (_type(obj) == spec["obj"] and _type(rec) == spec["rec"] and bool(locations) and rec not in locations
            and rec in (t.get("scene_receptacles") or []))


def twin_plan_wins_report(transitions: list[dict]) -> dict[str, Any]:
    hits = [t for t in transitions if twin_plan_wins(t)]
    otherwise_primary = [t for t in hits if t.get("context_determined") and t.get("swapped_goal_object")
                         and not t.get("shared_location") and t.get("targets_goal_object") and t.get("history_goal_agnostic")]
    no_twin = [t for t in transitions if _failed(t) and t.get("exclusion_reason") == "no_success_twin"]
    return {
        "n_failures_without_twin": len(no_twin),
        "n_take_goal_from_target_plan_wins": len(hits),
        "n_that_would_otherwise_be_primary": len(otherwise_primary),
        "games": len({t["game_id"] for t in hits}),
        "by_determination_reason": dict(Counter(t.get("determination_reason") or "-" for t in otherwise_primary)),
        "caveat": "counted from recorded fields (task, action, scene, exclusion reason); the collector did not record "
                  "that the replay ended the game, so this is the number of no-twin items with exactly that "
                  "signature, not a replayed confirmation",
    }


# ---------------- run ----------------


def _resolve(ref: str) -> Path:
    """A run name under runs/, or an absolute path to a run folder in another checkout."""
    p = Path(ref)
    return p if p.is_absolute() else runs_root() / ref


def _shards(ref: str | None) -> tuple[list[Path], list[str]]:
    if not ref:
        return [], ["not configured"]
    folders, problems = complete_shards(str(_resolve(ref)))
    return folders, problems


def _scores(folders: list[Path]) -> list[dict]:
    return [s for p in folders for s in read_jsonl(p / "scoring.jsonl")]


def run_audit_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    inputs = cfg.get("inputs") or {}
    sources = [str(_resolve(s)) for s in cfg.get("sources", SOURCES)]
    transitions = load_items(tuple(sources))
    if not transitions:
        write_status(run_id, f"no items found in {sources}", ok=False)
        return 1
    from wf import rq1x, thinking

    primary = primary_set(transitions)
    xsets = rq1x.rq1x_sets(transitions)
    read: dict[str, list[Path]] = {}
    missing: dict[str, Any] = {}

    def load(role: str) -> list[dict] | None:
        folders, problems = _shards(inputs.get(role))
        if problems:
            missing[role] = problems
            return None
        read[role] = folders
        return _scores(folders)

    report: dict[str, Any] = {"exclusion_flow": exclusion_flow(transitions),
                              "items_per_game": items_per_game(primary),
                              "twin_plan_wins": twin_plan_wins_report(transitions)}
    rq2 = load("rq2_score")
    if rq2 is not None:
        report["seen_without_object"] = seen_without_split(
            primary, {m: [s for s in rq2 if s["model"] == m and s["condition"] in ("aligned", "swapped")] for m in FEWSHOT})
    rq1 = load("rq1_score")
    rq1x_scores = load("rq1x_score")
    checks = [c for p in read.get("rq1x_score", []) if (p / "checks.jsonl").exists() for c in read_jsonl(p / "checks.jsonl")]
    think_ref = inputs.get("thinking_probe")
    think = None
    if think_ref and (_resolve(think_ref) / "thinking.jsonl").exists():
        think = read_jsonl(_resolve(think_ref) / "thinking.jsonl")
    elif think_ref:
        missing["thinking_probe"] = [f"{_resolve(think_ref)}/thinking.jsonl not found"]
    if rq1 is not None:
        think_items = thinking.sample_items(xsets["primary"], int(cfg.get("thinking_n_items", 1000)))
        report["ratios"] = headline_ratios(xsets["primary"], primary, rq1, rq1x_scores, checks, think, think_items)
    sizes = {}
    for tag, ref in (inputs.get("size_scores") or {}).items():
        folders, problems = _shards(ref)
        if problems:
            missing[f"size_score_{tag}"] = problems
            continue
        read[f"size_score_{tag}"] = folders
        sizes[tag] = [s for s in _scores(folders) if s["model"] == tag]
    if sizes:
        report["size_yes_share"] = size_yes_shares(sizes, xsets["primary"], xsets["successes"])
    first: dict[str, Any] = {}
    for role, scores in (("rq2_score", rq2), ("rq1_score", rq1), ("rq1x_score", rq1x_scores)):
        if scores is not None:
            first[role] = first_token_report(scores)
    rq1v = load("rq1v_score")
    if rq1v is not None:
        first["rq1v_score"] = first_token_report(rq1v)
    for tag, s in sizes.items():
        first[f"size_score_{tag}"] = first_token_report(s)
    ties: dict[str, Any] = {}
    cands = xsets["primary"] + xsets["successes"]
    for role, scores, arms in (("rq1x_score", (rq1 or []) + (rq1x_scores or []), ("aligned", "absent")),
                               ("rq1v_score", rq1v, ("yn_aligned", "yn_absent"))):
        if scores:
            ties[role] = {m: {a: top_candidate_ties(_by(scores, m), cands, a) for a in arms} for m in FEWSHOT}
    report["first_token"] = {"by_source": first, "top_candidate_ties": ties}
    report["not_computed"] = missing
    report["provenance"] = provenance(
        runs={**{s: Path(s) for s in sources}, **{role: _resolve(inputs[role]) for role in inputs if role in read},
              **{f"size_score_{t}": _resolve(r) for t, r in (inputs.get("size_scores") or {}).items() if f"size_score_{t}" in read}},
        inputs=[Path(s) / "transitions.jsonl" for s in sources] + shard_inputs([p for fs in read.values() for p in fs])
        + ([_resolve(think_ref) / "thinking.jsonl"] if think is not None else []))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Audit analyses (exploratory, 2026-11-20)", report)
    flow = report["exclusion_flow"]["flow"]
    lines = [f"{s['stage']}: {s['remaining']}" for s in flow]
    write_status(run_id, "exclusion flow:\n" + "\n".join(lines) + f"\nnot computed: {sorted(missing) or 'none'}",
                 ok=False, reported="audit report" if not missing else "partial audit report")
    return 0
