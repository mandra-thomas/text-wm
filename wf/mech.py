"""Exploratory mechanism diagnostic (decided 2026-11-19, after outside feedback; notes/deviations.md).

Two accounts of why the goal moves predictions on determined failures:
(a) state inference: the model cannot recover the facts that decide the failure, and falls back on the task;
(b) judgement: it recovers them, but judges the outcome by how it would solve the task.

One scoring run on RQ2's primary items, few-shot models, example set 1, history only:

- Prediction interventions (two-candidate L, aligned and swapped tasks): the plain prompt; the deciding
  facts as the history establishes them, stated before the prediction (`facts`); the environment's rules
  stated (`rules`); both.
- Questions under the aligned, swapped and neutral tasks, scored as "Yes" against "No" in place of
  "Next observation:":
  - `state`: the deciding fact (e.g. "Before this action, is the agent at the cabinet 3?"; correct answer No), with a
    twin whose correct answer is Yes, so a model that always says No is not counted as knowing;
  - `apply`: whether every condition the action needs holds now (correct: No);
  - `rule` (a fixed subsample): whether the rule behind the failure holds in this environment, with a twin.
- `policy`: the agent's own preference for the action (only its full L is read: both candidates begin with "Action",
  so their first-token term is zero by construction). The ReAct agent prompt, built from the item's history,
  asks for the action only; "Action: <the item's action>" is scored against "Action: look".

The facts come from wf.data.transitions.history_facts, the parser that decided the item was determined, so
they never go beyond the history.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any

from wf import constants as C
from wf import rq1
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap
from wf.data.transitions import HistoryFacts, history_facts
from wf.rq2 import SOURCES, analysis_provenance, complete_shards, load_items, shard_items
from wf.runs import _effect, _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer
from wf.scoring.prompts import REACT_FEWSHOT, REACT_SYSTEM, _replace_task_line
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

GOALS = ("aligned", "swapped", "absent")
RULES = ("Rules of this environment: the agent can take an object from, put an object in or on, open or close a "
         "receptacle, or heat, cool or clean an object with one, only when it is at that receptacle (or at another "
         "receptacle in the same spot of the room). It cannot take an object from, or put one into, a closed receptacle. "
         "It holds at most one object, so it can take an object only when its hands are empty, and only if the object "
         "is in or on that receptacle. It must be holding an object to put it somewhere or to heat, cool or clean it. "
         "When an action is not possible, the environment replies \"Nothing happens.\"")
PRED_ARMS = tuple(f"pred_{v}_{g}" for v in ("plain", "facts", "rules", "both") for g in ("aligned", "swapped"))
Q_ARMS = tuple(f"{q}_{g}" for q in ("state", "statetwin", "apply") for g in GOALS)
RULE_ARMS = tuple(f"{q}_{g}" for q in ("rule", "ruletwin") for g in GOALS)
POLICY_ARMS = tuple(f"policy_{g}" for g in GOALS)
RULE_SAMPLE = 1000  # items with rule questions; the rule does not depend on the item's history
POLICY_REF = "Action: look"
_ACT = re.compile(r"^(take|move|heat|cool|clean) (.+?) (?:from|to|with) (.+)$")


def _parse(action: str) -> tuple[str, str, str] | None:
    m = _ACT.match(action)
    return (m[1], m[2], m[3]) if m else None


def _rtype(receptacle: str) -> str:
    return re.sub(r"\s+\d+$", "", receptacle)


def facts_text(f: HistoryFacts) -> str:
    """The facts the history establishes, as sentences; unknowns are left out, never guessed."""
    parts = []
    if f.location:
        parts.append(f"The agent is at the {f.location}.")
    elif f.knows_position:
        parts.append("The agent is in the middle of the room and has not gone to any receptacle.")
    if f.knows_position:
        parts.append(f"The agent is holding the {f.holding}." if f.holding else "The agent is holding nothing.")
    if f.opened:
        parts.append("Open: " + ", ".join(sorted(f.opened)) + ".")
    if f.closed:
        parts.append("Closed: " + ", ".join(sorted(f.closed)) + ".")
    for r in sorted(f.contents):
        seen = ", ".join(sorted(f.contents[r])) or "nothing"
        parts.append(f"When the agent last saw the {r}, it contained: {seen}.")
    return "Current state, as the history establishes it: " + " ".join(parts)


def state_questions(t: dict) -> tuple[str, str] | None:
    """(question whose correct answer is No, twin whose correct answer is Yes) about the deciding fact."""
    p = _parse(t["action"])
    if not p:
        return None
    verb, obj, rec = p
    f = history_facts(t["history_text"])
    reason = t.get("determination_reason")
    if reason in ("away_from_receptacle", "away_from_receptacle_never_moved"):
        twin = (f"Before this action, is the agent at the {f.location}?" if f.location else
                "Before this action, is the agent still in the middle of the room, without having gone to any receptacle?")
        return f"Before this action, is the agent at the {rec}?", twin
    if reason == "hands_full" and f.holding:
        return "Before this action, are the agent's hands empty?", f"Before this action, is the agent holding the {f.holding}?"
    if reason == "not_holding_object":
        twin = f"Before this action, is the agent holding the {f.holding}?" if f.holding else "Before this action, are the agent's hands empty?"
        return f"Before this action, is the agent holding the {obj}?", twin
    if reason == "contents_or_open_state" and verb == "take":
        if rec in f.closed:
            return f"Before this action, is the {rec} open?", f"Before this action, is the {rec} closed?"
        if rec in f.contents and obj not in f.contents[rec]:
            seen = sorted(f.contents[rec])
            twin = (f"When the agent last saw the {rec}, was the {seen[0]} there?" if seen else
                    f"When the agent last saw the {rec}, was it empty?")
            return f"When the agent last saw the {rec}, was the {obj} there?", twin
    return None


def rule_questions(t: dict) -> tuple[str, str] | None:
    """(rule question whose correct answer is No, twin whose correct answer is Yes) for the deciding rule."""
    p = _parse(t["action"])
    if not p:
        return None
    verb, _obj, rec = p
    reason = t.get("determination_reason")
    take_ok = ("can the agent take an object from an open receptacle it is at, when the object is there and the "
               "agent's hands are empty?")
    # Yes-keys must hold under ALFWorld's PDDL: putting needs an open (or non-opening) receptacle that can hold the
    # object; heating, cooling and cleaning need an object of a kind that can be heated, cooled or cleaned.
    put_ok = ("can the agent put an object it is holding in or on a receptacle it is at, when the receptacle is open "
              "(or does not open) and can hold that object?")
    use_ok = lambda v, r: f"can the agent {v} an object it is holding, of a kind that can be {v}ed, with a {r} it is at?"
    if reason in ("away_from_receptacle", "away_from_receptacle_never_moved"):
        act = {"take": "take an object from a receptacle", "move": "put an object it is holding in or on a receptacle"}.get(
            verb, f"{verb} an object it is holding with a {_rtype(rec)}")
        twin = take_ok if verb == "take" else put_ok if verb == "move" else use_ok(verb, _rtype(rec))
        return (f"In this environment, can the agent {act} that is in another part of the room, without going there first?",
                f"In this environment, {twin}")
    if reason == "hands_full":
        return ("In this environment, can the agent pick up an object while it is already holding another object?",
                f"In this environment, {take_ok}")
    if reason == "not_holding_object":
        doing = "put an object in or on a receptacle" if verb == "move" else f"{verb} an object with a {_rtype(rec)}"
        twin = put_ok if verb == "move" else use_ok(verb, _rtype(rec))
        return (f"In this environment, can the agent {doing} when it is not holding that object?",
                f"In this environment, {twin}")
    if reason == "contents_or_open_state" and verb == "take":
        f = history_facts(t["history_text"])
        if rec in f.closed:
            return "In this environment, can the agent take an object from a closed receptacle without opening it?", \
                f"In this environment, {take_ok}"
        return "In this environment, can the agent take an object from a receptacle where that object is not?", \
            f"In this environment, {take_ok}"
    return None


def rule_sample(items: list[dict]) -> set[str]:
    """A fixed RULE_SAMPLE of the items, by hash of the id, independent of sharding."""
    ranked = sorted(items, key=lambda t: hashlib.sha256(f"20261119|mech|{t['id']}".encode()).hexdigest())
    return {t["id"] for t in ranked[:RULE_SAMPLE]}


def _question(msgs: list[dict], q: str) -> list[dict]:
    last = msgs[-1]
    if not last["content"].endswith("Next observation:"):
        raise ValueError("prompt does not end with 'Next observation:'")
    return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + f"Question: {q} Answer Yes or No.\nAnswer:"}]


def policy_messages(t: dict, goal: str) -> list[dict[str, str]]:
    """The ReAct agent's prompt at this item's state, asking for the action only (no admissible list is stored)."""
    lines = t["history_text"].split("\n> ")[0]  # opening observation, task line included
    opening = _replace_task_line(lines, rq1.goal_line(t, goal))
    obs = [opening] + [o for _, o in (t.get("history_pairs") or [])]
    acts = [a for a, _ in (t.get("history_pairs") or [])]
    body = [x for o, a in zip(obs, acts) for x in (f"Observation: {o}", f"Action: {a}")]
    body += [f"Observation: {obs[-1]}", "Respond with the Action line only."]
    return [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": REACT_FEWSHOT},
            {"role": "user", "content": "\n".join(body)}]


def arm_messages(t: dict, arm: str) -> list[dict[str, str]] | None:
    kind, goal = arm.rsplit("_", 1)
    if goal == "swapped" and not t.get("swapped_task_line"):
        return None
    if kind == "policy":
        return policy_messages(t, goal)
    base = rq1.arm_messages(t, goal)
    if kind.startswith("pred_"):
        variant = kind.split("_", 1)[1]
        note = {"plain": None, "facts": facts_text(history_facts(t["history_text"])), "rules": RULES,
                "both": facts_text(history_facts(t["history_text"])) + "\n\n" + RULES}[variant]
        if note is None:
            return base
        last = base[-1]
        return base[:-1] + [{**last, "content": last["content"].replace("Next observation:", f"{note}\n\nNext observation:")}]
    if kind in ("state", "statetwin"):
        qs = state_questions(t)
        return _question(base, qs[kind == "statetwin"]) if qs else None
    if kind in ("rule", "ruletwin"):
        qs = rule_questions(t)
        return _question(base, qs[kind == "ruletwin"]) if qs else None
    if kind == "apply":
        return _question(base, "Before this action is taken, are all the conditions it needs met?")
    raise ValueError(arm)


def mech_jobs(items: list[dict], sample: set[str]) -> list[tuple[dict, str]]:
    jobs = [(t, a) for t in items for a in PRED_ARMS + Q_ARMS + POLICY_ARMS]
    jobs += [(t, a) for t in items if t["id"] in sample for a in RULE_ARMS]
    return [(t, a) for t, a in jobs if arm_messages(t, a) is not None]


def _candidates(t: dict, arm: str) -> tuple[str, str]:
    if arm.startswith("pred_"):
        return t["candidates"]["o_plus"], t["candidates"]["o_minus"]
    if arm.startswith("policy_"):
        return f"Action: {t['action']}", POLICY_REF
    return "Yes", "No"


def run_mech_score(cfg: dict) -> int:
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
    primary = rq1.rq1_sets(load_items(sources))["primary"]
    jobs = mech_jobs(shard_items(primary, i, n), rule_sample(primary))
    hashes = {t["id"]: _item_hash(t) for t, _ in jobs}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    for model_name, tag, _ft in [m for m in rq1.MODELS if m[1] in cfg.get("models", rq1.FEWSHOT)]:
        todo = [(t, a) for t, a in jobs if (tag, t["id"], a) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, arm in todo:
            plus, minus = _candidates(t, arm)
            score = scorer.score_one_token_pair if (plus, minus) == ("Yes", "No") else scorer.score_pair
            sc = score(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm), o_plus=plus, o_minus=minus)
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} scores", ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    r = [0.0] * len(xs)
    k = 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[k]]:
            j += 1
        for m in range(k, j + 1):
            r[order[m]] = (k + j) / 2
        k = j + 1
    return r


def spearman(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    a, b = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5 if va and vb else float("nan")


def _mean_ci(rows: list[tuple[str, float]]) -> dict:
    boot = cluster_bootstrap(lambda xs: sum(xs) / len(xs) if xs else float("nan"), [x for _, x in rows],
                             [g for g, _ in rows], seed=C.RNG_SEED)
    return {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]], "n": len(rows)}


def _ratio_ci(rows: list[tuple[str, Any]], stat) -> dict:
    """A ratio statistic over item-level tuples, with a game-clustered interval from the same resamples."""
    if not rows:
        return None
    boot = cluster_bootstrap(stat, [x for _, x in rows], [g for g, _ in rows], seed=C.RNG_SEED)
    return {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]], "n": len(rows)}


def _diff_rows(by: dict, items: list[dict], a: str, b: str) -> list[tuple[str, float]]:
    return [(t["game_id"], by[(t["id"], a)] - by[(t["id"], b)]) for t in items if (t["id"], a) in by and (t["id"], b) in by]


def _correct(by: dict, t: dict, q: str, goal: str) -> bool | None:
    main, twin = by.get((t["id"], f"{q}_{goal}")), by.get((t["id"], f"{q}twin_{goal}"))
    if main is None or twin is None:
        return None
    return main < 0 < twin


def _model_block(s: list[dict], items: list[dict]) -> dict:
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    out: dict[str, Any] = {"interventions": {}, "questions": {}, "split_by_knowledge": {}, "policy": {}}
    for v in ("plain", "facts", "rules", "both"):
        out["interventions"][v] = {
            "goal_effect": _effect(s, items, primary=f"pred_{v}_aligned", control=f"pred_{v}_swapped"),
            "mean_L_aligned": _mean_ci([(t["game_id"], by[(t["id"], f"pred_{v}_aligned")]) for t in items if (t["id"], f"pred_{v}_aligned") in by]),
        }
        if v != "plain":
            rows = [(t["game_id"], (by[(t["id"], "pred_plain_aligned")] - by[(t["id"], "pred_plain_swapped")])
                     - (by[(t["id"], f"pred_{v}_aligned")] - by[(t["id"], f"pred_{v}_swapped")]))
                    for t in items if all((t["id"], a) in by for a in ("pred_plain_aligned", "pred_plain_swapped",
                                                                         f"pred_{v}_aligned", f"pred_{v}_swapped"))]
            out["interventions"][v]["reduction_vs_plain"] = _mean_ci(rows)
            # Readout 1 as logged: the share of the plain effect removed, (plain - v) / plain on the same items,
            # computed inside each resample so its interval is the ratio's own.
            quads = [(t["game_id"], (by[(t["id"], "pred_plain_aligned")] - by[(t["id"], "pred_plain_swapped")],
                                     by[(t["id"], f"pred_{v}_aligned")] - by[(t["id"], f"pred_{v}_swapped")]))
                     for t in items if all((t["id"], a) in by for a in ("pred_plain_aligned", "pred_plain_swapped",
                                                                          f"pred_{v}_aligned", f"pred_{v}_swapped"))]
            out["interventions"][v]["share_of_effect_removed"] = _ratio_ci(
                quads, lambda xs: 1 - sum(b for _, b in xs) / sum(a for a, _ in xs) if sum(a for a, _ in xs) else float("nan"))
    for q in ("state", "apply", "rule"):
        block = {}
        for g in GOALS:
            vals = [(t["game_id"], 1.0 if by[(t["id"], f"{q}_{g}")] < 0 else 0.0) for t in items if (t["id"], f"{q}_{g}") in by]
            block[f"says_no_{g}"] = _mean_ci(vals) if vals else None
            if q != "apply":
                both = [(t["game_id"], 1.0 if _correct(by, t, q, g) else 0.0) for t in items if _correct(by, t, q, g) is not None]
                block[f"both_correct_{g}"] = _mean_ci(both) if both else None
        block["goal_effect_on_main_L"] = _mean_ci(_diff_rows(by, items, f"{q}_aligned", f"{q}_swapped")) if _diff_rows(by, items, f"{q}_aligned", f"{q}_swapped") else None
        if q != "apply":
            # Readout 3's control: a goal that raises "Yes" to every question would also raise the twin's.
            block["goal_effect_on_twin_L"] = _mean_ci(_diff_rows(by, items, f"{q}twin_aligned", f"{q}twin_swapped")) if _diff_rows(by, items, f"{q}twin_aligned", f"{q}twin_swapped") else None
            mt = [(t["game_id"], (by[(t["id"], f"{q}_aligned")] - by[(t["id"], f"{q}_swapped")])
                   - (by[(t["id"], f"{q}twin_aligned")] - by[(t["id"], f"{q}twin_swapped")]))
                  for t in items if all((t["id"], f"{a}_{g}") in by for a in (q, f"{q}twin") for g in ("aligned", "swapped"))]
            block["goal_effect_main_minus_twin"] = _mean_ci(mt) if mt else None
        block["aligned_minus_neutral_main_L"] = _mean_ci(_diff_rows(by, items, f"{q}_aligned", f"{q}_absent")) if _diff_rows(by, items, f"{q}_aligned", f"{q}_absent") else None
        out["questions"][q] = block
    plain = lambda ts: _mean_ci(_diff_rows(by, ts, "pred_plain_aligned", "pred_plain_swapped")) if ts else None
    for q in ("state", "rule"):
        known = [t for t in items if _correct(by, t, q, "absent") is True]
        unknown = [t for t in items if _correct(by, t, q, "absent") is False]
        out["split_by_knowledge"][q] = {"known_under_neutral": plain(known), "not_known_under_neutral": plain(unknown)}
        # Readout 2 as logged: the known-state items' effect as a share of the effect on all items with a verdict.
        flagged = [(t["game_id"], (by[(t["id"], "pred_plain_aligned")] - by[(t["id"], "pred_plain_swapped")],
                                   _correct(by, t, q, "absent")))
                   for t in known + unknown if (t["id"], "pred_plain_aligned") in by and (t["id"], "pred_plain_swapped") in by]

        def _share(xs):
            k = [d for d, ok in xs if ok]
            return (sum(k) / len(k)) / (sum(d for d, _ in xs) / len(xs)) if k and xs and sum(d for d, _ in xs) else float("nan")

        out["split_by_knowledge"][q]["known_effect_share_of_overall"] = _ratio_ci(flagged, _share)
    by_reason = {}
    for r in sorted({t.get("determination_reason") for t in items}):
        ts = [t for t in items if t.get("determination_reason") == r]
        by_reason[r] = {"plain": plain(ts),
                        "state_known": plain([t for t in ts if _correct(by, t, "state", "absent") is True]),
                        "state_goal_effect": _mean_ci(_diff_rows(by, ts, "state_aligned", "state_swapped")) if _diff_rows(by, ts, "state_aligned", "state_swapped") else None}
    out["by_reason"] = by_reason
    pairs = [(t["game_id"], (by[(t["id"], "policy_aligned")] - by[(t["id"], "policy_swapped")],
                             by[(t["id"], "pred_plain_aligned")] - by[(t["id"], "pred_plain_swapped")]))
             for t in items if all((t["id"], a) in by for a in ("policy_aligned", "policy_swapped", "pred_plain_aligned", "pred_plain_swapped"))]
    if pairs:
        boot = cluster_bootstrap(lambda xs: spearman(xs), [p for _, p in pairs], [g for g, _ in pairs], seed=C.RNG_SEED)
        out["policy"] = {"policy_goal_effect": _mean_ci([(g, p[0]) for g, p in pairs]),
                         "spearman_policy_shift_vs_delta_L": {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]], "n": len(pairs)}}
        within = {}
        for r in sorted({t.get("determination_reason") for t in items}):
            ids = {t["id"] for t in items if t.get("determination_reason") == r}
            sub = [(t["game_id"], (by[(t["id"], "policy_aligned")] - by[(t["id"], "policy_swapped")],
                                   by[(t["id"], "pred_plain_aligned")] - by[(t["id"], "pred_plain_swapped")]))
                   for t in items if t["id"] in ids and all((t["id"], a) in by for a in ("policy_aligned", "policy_swapped", "pred_plain_aligned", "pred_plain_swapped"))]
            if len(sub) >= 3:
                b = cluster_bootstrap(lambda xs: spearman(xs), [p for _, p in sub], [g for g, _ in sub], seed=C.RNG_SEED)
                within[r] = {"spearman": b["estimate"], "ci": [b["ci_low"], b["ci_high"]], "n": len(sub)}
            else:
                within[r] = {"spearman": float("nan"), "n": len(sub)}
        out["policy"]["within_reason"] = within
    return out


def run_mech_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    primary = rq1.rq1_sets(load_items(tuple(cfg.get("sources", SOURCES))))["primary"]
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_rule_sample": len(rule_sample(primary) & {t["id"] for t in primary}),
                                        "n_state_question": sum(1 for t in primary if state_questions(t))}}
    for tag in cfg.get("models", rq1.FEWSHOT):
        report[tag] = _model_block([x for x in scores if x["model"] == tag], primary)
    report["provenance"] = analysis_provenance(cfg["source"], folders, cfg.get("sources", SOURCES))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Mechanism diagnostic: state inference against judgement (exploratory)", report)
    lines = []
    for tag in cfg.get("models", rq1.FEWSHOT):
        iv = report[tag]["interventions"]
        lines.append(f"{tag}: plain {iv['plain']['goal_effect']['delta_L_nats']:.2f}, facts {iv['facts']['goal_effect']['delta_L_nats']:.2f}, "
                     f"rules {iv['rules']['goal_effect']['delta_L_nats']:.2f}, both {iv['both']['goal_effect']['delta_L_nats']:.2f}")
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines), ok=False, reported="mechanism report")
    return 0
