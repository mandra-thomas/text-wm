"""Option A: offline decision simulation (exploratory; design in notes/decisionsim_design.md, committed before any
decision result).

Does the task-induced bias change which action a planner that consults the world model would choose, and how
often does it make that planner choose an action that fails? A decision-level diagnostic, not a measure of
end-to-end planning harm: nothing is executed after the choice.

At eligible states of the collected episodes (the agent's own next action is among the candidates and known, at
least one candidate succeeds and at least one is a failure the history determines), every candidate is scored with
the yes/no feasibility score L = log P(Yes) - log P(No) under the aligned (real) task and the neutral task
("explore the room"), optionally under one state-level swapped task. The candidate set is fixed across conditions.
Three rules choose an action: (1) the largest L; (2) verification: the agent's proposal if L > 0, else the largest
L among the other candidates; (3) the largest L among goal-relevant candidates (secondary, only where one of them
succeeds). The analysis compares the chosen-failure rate and the choices themselves between conditions, by game.

`decisionsim_score_<tag>` scores one model (yes/no only, pinned revision, every condition of a state in one job);
`decisionsim_analysis` reads them all.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Callable

from wf import constants as C
from wf import families, rq1v, sizes
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, pair_permutation
from wf.data.goals import make_swapped_task_line
from wf.rq2 import SOURCES, analysis_provenance, complete_shards, load_items, shard_items
from wf.runs import _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

NOTHING = "Nothing happens."
YN_ARMS = rq1v.YN_ARMS  # yn_aligned, yn_swapped, yn_absent
DEFAULT_ARMS = ("yn_aligned", "yn_absent")
ALIGNED, NEUTRAL, SWAPPED = "yn_aligned", "yn_absent", "yn_swapped"
PRIMARY = ("qwen3-8b", "llama3.1-8b")
# tag: Hugging Face id. Family models take their prompts and loading settings from wf/families.py.
REFERENCE_MODELS = {"qwen3-8b": C.WM_FEWSHOT_MODEL, "llama3.1-8b": C.WM_FEWSHOT_LLAMA}
FAMILY_MODELS = ("mistral-7b-v0.3", "gemma2-9b-it", "llama3.2-3b")
MODEL_TAGS = tuple(REFERENCE_MODELS) + FAMILY_MODELS
RULES = ("rule1_with_proposal", "rule1_generated_only", "rule2_verification", "rule3_goal_relevant")
OPTION_C_RULES = ("rule1_with_proposal", "rule2_verification")
OPTION_C_CHANGE = 0.05  # logged in the design note before any score


def succeeded(t: dict) -> bool:
    return (t.get("true_obs") or "").strip() != NOTHING


# ---------------- states and candidate sets ----------------


def agent_actions(transitions: list[dict]) -> dict[tuple[str, int], str]:
    """The agent's own action at (game, step), read from the longer saved prefixes. The last saved step of a game
    has none: no later state records what the agent did there."""
    out: dict[tuple[str, int], str] = {}
    for t in transitions:
        for i, a in enumerate(t.get("prefix_actions") or []):
            prev = out.setdefault((t["game_id"], i), a)
            if prev != a:
                raise ValueError(f"{t['game_id']} step {i}: two different agent actions in the saved prefixes")
    return out


def _named_types(action: str) -> set[str]:
    """Object types an action names with an instance number ("take mug 1 from cabinet 2" -> mug, cabinet)."""
    return {m for m in re.findall(r"([a-z]+) \d+", action or "")}


def state_swap(cands: list[dict], game_id: str, state_idx: int, seed: int = C.RNG_SEED) -> str | None:
    """One swap object for the whole state: among the swap objects already drawn for its candidates (each feasible
    for the task and unrelated to the history), excluding any type named in a candidate action; None if none."""
    drawn = sorted({t["swapped_goal_object"] for t in cands if t.get("swapped_goal_object")})
    named = set().union(*(_named_types(t["action"]) for t in cands)) if cands else set()
    options = [o for o in drawn if o not in named]
    if not options:
        return None
    return random.Random(f"{seed}|decisionsim|{game_id}|{state_idx}").choice(options)


def build_states(transitions: list[dict], seed: int = C.RNG_SEED) -> list[dict]:
    """Every eligible state, sorted by (game, step), with its full candidate set (sorted by action)."""
    agent = agent_actions(transitions)
    by: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for t in transitions:
        by[(t["game_id"], t["state_idx"])].append(t)
    states = []
    for (g, i), ts in sorted(by.items()):
        if len({t["action"] for t in ts}) != len(ts):
            raise ValueError(f"{g} step {i}: duplicate candidate actions")
        if len({t["task_line"] for t in ts}) != 1 or len({t["history_text"] for t in ts}) != 1:
            raise ValueError(f"{g} step {i}: candidates disagree on the task or the history")
        proposal = agent.get((g, i))
        if proposal is None or not any(t["action"] == proposal for t in ts):
            continue
        if not any(succeeded(t) for t in ts):
            continue
        if not any(not succeeded(t) and t.get("context_determined") for t in ts):
            continue
        cands = sorted(ts, key=lambda t: t["action"])
        states.append({"game_id": g, "state_idx": i, "proposal": proposal, "candidates": cands,
                       "swap": state_swap(cands, g, i, seed)})
    return states


def select_states(states: list[dict], n_states: int | str, seed: int = C.RNG_SEED) -> list[dict]:
    """A simple random sample of `n_states` eligible states (or all of them, n_states="all"), in (game, step)
    order."""
    if n_states == "all" or int(n_states) >= len(states):
        return list(states)
    keys = sorted((s["game_id"], s["state_idx"]) for s in states)
    chosen = set(random.Random(seed).sample(keys, int(n_states)))
    return [s for s in states if (s["game_id"], s["state_idx"]) in chosen]


def sample_digest(states: list[dict]) -> str:
    """What the sample is: its states and their candidate ids, so scores never attach to another sample."""
    key = [[s["game_id"], s["state_idx"], [t["id"] for t in s["candidates"]], s["swap"]] for s in states]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]


def prompt_item(t: dict, state: dict, arm: str) -> dict | None:
    """The item as the arm's prompt sees it. The swapped arm uses the state's single swap object, never the
    candidate's own draw, so every candidate at a state is scored under the same task line."""
    if arm != SWAPPED:
        return t
    if not state.get("swap"):
        return None
    return {**t, "swapped_goal_object": state["swap"],
            "swapped_task_line": make_swapped_task_line(t["task_line"], t["goal_object"], state["swap"])}


def score_hash(t: dict, arm: str, state: dict) -> str:
    item = prompt_item(t, state, arm)
    return hashlib.sha256(f"{_item_hash(item)}|{arm}".encode()).hexdigest()[:16] if item else ""


def messages_for(tag: str) -> Callable[[dict, str], list[dict[str, str]] | None]:
    if tag in REFERENCE_MODELS:
        return sizes.arm_messages
    spec = families.MODELS[tag]
    return lambda t, arm: families.family_messages(spec, t, arm)


def score_jobs(states: list[dict], arms: tuple[str, ...]) -> list[tuple[dict, dict, str]]:
    """(state, candidate, arm) for every candidate of every state, the arms of a state next to each other."""
    return [(s, t, a) for s in states for t in s["candidates"] for a in arms if prompt_item(t, s, a) is not None]


def open_scorer(tag: str) -> LogoddsScorer:
    if tag in REFERENCE_MODELS:
        return LogoddsScorer(REFERENCE_MODELS[tag], model_tag=tag)
    spec = families.MODELS[tag]
    return LogoddsScorer(spec.hf_id, model_tag=tag, attn_implementation=spec.attn_implementation,
                         template_kwargs=spec.template_kwargs)


def yes_no_gate(scorer: LogoddsScorer, prompts: list[list[dict[str, str]]]) -> dict[str, Any]:
    """Wave A's Yes/No check, through the scorer's own encoding: "Yes" and "No" are single, different tokens, each
    equal to the template's rendering of that word as the reply after every prompt."""
    problems: list[str] = []
    scored = {w: families._ids(scorer._encode_candidate(w)) for w in families.YES_NO}
    for w, ids in scored.items():
        if len(ids) != 1:
            problems.append(f"{w!r} is not one token: {ids}")
    if scored["Yes"] == scored["No"]:
        problems.append("'Yes' and 'No' are the same token")
    for k, msgs in enumerate(prompts):
        for w in families.YES_NO:
            try:
                reply = families.template_reply(scorer, msgs, w)
            except families.TokenizationMismatch as e:
                problems.append(f"prompt {k} {w!r}: {e}")
                continue
            if reply != scored[w]:
                problems.append(f"prompt {k}: {w!r} alone {scored[w]}, template {reply}")
    return {"ok": not problems and bool(prompts), "n_prompts": len(prompts), "problems": problems[:50],
            "n_problems": len(problems), "yes_no_ids": scored}


def run_decisionsim_score(cfg: dict) -> int:
    tag = cfg["model"]
    if tag not in MODEL_TAGS:
        raise SystemExit(f"unknown model {tag!r}; known: {MODEL_TAGS}")
    arms = tuple(cfg.get("arms", DEFAULT_ARMS))
    if not set(arms) <= set(YN_ARMS) or ALIGNED not in arms or NEUTRAL not in arms:
        raise SystemExit(f"arms must include {ALIGNED} and {NEUTRAL}, from {YN_ARMS}; got {arms}")
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
    seed = int(cfg.get("seed", C.RNG_SEED))
    states = select_states(build_states(load_items(sources), seed), cfg.get("n_states", 300), seed)
    digest = sample_digest(states)
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n, "arms": list(arms), "sample_digest": digest}
    d = _init_run(run_id, cfg, inputs=())
    games = {x["game_id"] for x in shard_items([{"game_id": s["game_id"]} for s in states], i, n)}  # by game
    mine = [s for s in states if s["game_id"] in games]
    write_json(d / "states.json", {"sample_digest": digest, "n_states_sampled": len(states),
                                   "states": [[s["game_id"], s["state_idx"]] for s in mine]})
    jobs = score_jobs(mine, arms)
    hashes = {(t["id"], a): score_hash(t, a, s) for s, t, a in jobs}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get((x["transition_id"], x["condition"])) != x.get("score_hash") for x in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items or another sample; rerun without --resume.")
    done = {(x["transition_id"], x["condition"]) for x in saved}
    todo = [(s, t, a) for s, t, a in jobs if (t["id"], a) not in done]
    build = messages_for(tag)
    if todo:
        scorer = open_scorer(tag)
        if tag in FAMILY_MODELS and not scorer.mock:  # wave A's Yes/No gate, before any score (a resume too)
            gate = yes_no_gate(scorer, [build(prompt_item(t, s, a), a) for s, t, a in jobs])
            write_json(d / "tokenization_checks.json", gate)
            if not gate["ok"]:
                write_status(run_id, f"stopped before scoring: Yes/No gate failed ({gate['n_problems']} problems)", ok=False)
                raise families.TokenizationMismatch(f"{tag}: Yes/No gate failed: {gate['problems'][:3]}")
        for s, t, arm in todo:
            sc = scorer.score_one_token_pair(transition_id=t["id"], condition=arm, messages=build(prompt_item(t, s, arm), arm),
                                             o_plus="Yes", o_minus="No")
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "state_idx": t["state_idx"],
                                                "score_hash": hashes[(t["id"], arm)]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} yes/no scores at {len(mine)} states ({tag})", ok=False,
                 reported="shard complete")
    return 0


# ---------------- decision rules ----------------


def choose_max(cands: list[dict], L: dict[str, float]) -> dict:
    """The candidate with the largest L; a tie goes to the action that sorts first (blind to outcomes)."""
    if not cands:
        raise ValueError("no candidates to choose from")
    return min(cands, key=lambda t: (-L[t["id"]], t["action"]))


def _proposal(state: dict) -> dict:
    return next(t for t in state["candidates"] if t["action"] == state["proposal"])


def _generated(state: dict) -> list[dict]:
    return [t for t in state["candidates"] if t["action"] != state["proposal"]]


def _goal_relevant(state: dict) -> list[dict]:
    return [t for t in state["candidates"] if t.get("targets_goal_object")]


def rule_choice(rule: str, state: dict, L: dict[str, float]) -> dict:
    if rule == "rule1_with_proposal":
        return choose_max(state["candidates"], L)
    if rule == "rule1_generated_only":
        return choose_max(_generated(state), L)
    if rule == "rule2_verification":
        p = _proposal(state)
        return p if L[p["id"]] > 0 else choose_max(_generated(state), L)
    if rule == "rule3_goal_relevant":
        return choose_max(_goal_relevant(state), L)
    raise ValueError(rule)


def rule_applies(rule: str, state: dict) -> bool:
    """Rule 3 is read only where a goal-relevant candidate succeeds; elsewhere it must choose a failure."""
    if rule == "rule3_goal_relevant":
        return any(succeeded(t) for t in _goal_relevant(state))
    return True


def rule_set(rule: str, state: dict) -> list[dict]:
    return {"rule1_generated_only": _generated, "rule3_goal_relevant": _goal_relevant}.get(
        rule, lambda s: s["candidates"])(state)


def _ties(cands: list[dict], L: dict[str, float]) -> bool:
    top = max(L[t["id"]] for t in cands)
    return sum(1 for t in cands if L[t["id"]] == top) > 1


# ---------------- measures ----------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _boot(values: list[float], games: list[str]) -> dict[str, Any]:
    b = cluster_bootstrap(_mean, values, games, seed=C.RNG_SEED)
    return {"estimate": b["estimate"], "ci": [b["ci_low"], b["ci_high"]], "n_states": len(values),
            "n_games": len(set(games))}


def _paired(a: list[float], b: list[float], games: list[str]) -> dict[str, Any]:
    """Mean of a - b by state, with its game bootstrap and the game sign-flip permutation p."""
    diffs = [x - y for x, y in zip(a, b)]
    out = _boot(diffs, games)
    out["p"] = pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=games) if diffs else float("nan")
    return out


def contrast(rule: str, states: list[dict], La: dict[str, float], Lb: dict[str, float]) -> dict[str, Any]:
    """Choices under condition a against condition b, at the states where the rule applies."""
    rows = []
    for s in states:
        if not rule_applies(rule, s):
            continue
        ca, cb = rule_choice(rule, s, La), rule_choice(rule, s, Lb)
        cands = rule_set(rule, s)
        rows.append({"game": s["game_id"], "fail_a": float(not succeeded(ca)), "fail_b": float(not succeeded(cb)),
                     "det_a": bool(not succeeded(ca) and ca.get("context_determined")),
                     "det_b": bool(not succeeded(cb) and cb.get("context_determined")),
                     "changed": float(ca["action"] != cb["action"]),
                     "random_fail": _mean([float(not succeeded(t)) for t in cands]),
                     "tie_a": _ties(cands, La), "tie_b": _ties(cands, Lb)})
    if not rows:
        return {"n_states": 0}
    games = [r["game"] for r in rows]
    fa, fb = [r["fail_a"] for r in rows], [r["fail_b"] for r in rows]
    changed = [r for r in rows if r["changed"]]
    harm = sum(1 for r in changed if r["fail_a"] and not r["fail_b"])
    help_ = sum(1 for r in changed if not r["fail_a"] and r["fail_b"])
    both_fail = sum(1 for r in changed if r["fail_a"] and r["fail_b"])
    both_ok = len(changed) - harm - help_ - both_fail
    n = len(rows)
    diff = _paired(fa, fb, games)
    share = lambda k: k / len(changed) if changed else float("nan")
    return {
        "n_states": n, "n_games": len(set(games)),
        "fail_rate_a": _boot(fa, games), "fail_rate_b": _boot(fb, games),
        "fail_rate_difference": diff,
        "determined_share_of_chosen_failures": {
            "a": sum(r["det_a"] for r in rows) / max(1, sum(fa)), "b": sum(r["det_b"] for r in rows) / max(1, sum(fb))},
        "change_rate": _boot([r["changed"] for r in rows], games),
        "changed": {"n": len(changed), "a_fails_b_succeeds": harm, "a_succeeds_b_fails": help_,
                    "both_fail": both_fail, "both_succeed": both_ok,
                    "share_a_fails_b_succeeds": share(harm), "share_a_succeeds_b_fails": share(help_),
                    "share_both_fail": share(both_fail), "share_both_succeed": share(both_ok)},
        # the identity the design note promises: (harm - help) / n is the difference in failure rates
        "identity_holds": abs((harm - help_) / n - (sum(fa) - sum(fb)) / n) < 1e-12,
        "random_choice_fail_rate": _mean([r["random_fail"] for r in rows]),
        "ties": {"a": sum(r["tie_a"] for r in rows), "b": sum(r["tie_b"] for r in rows)},
    }


def verification_errors(states: list[dict], La: dict[str, float], Lb: dict[str, float]) -> dict[str, Any]:
    """Rule 2's two error rates: failing proposals accepted (L > 0) and succeeding proposals rejected."""
    out: dict[str, Any] = {}
    for name, keep, wrong in (("false_acceptance", lambda p: not succeeded(p), lambda L, p: L[p["id"]] > 0),
                              ("false_rejection", succeeded, lambda L, p: not L[p["id"]] > 0)):
        ps = [(s["game_id"], _proposal(s)) for s in states if keep(_proposal(s))]
        games = [g for g, _ in ps]
        a = [float(wrong(La, p)) for _, p in ps]
        b = [float(wrong(Lb, p)) for _, p in ps]
        out[name] = {"n_states": len(ps), "rate_a": _boot(a, games), "rate_b": _boot(b, games),
                     "difference": _paired(a, b, games) if ps else {"n_states": 0}}
    return out


def baseline(states: list[dict], L: dict[str, float]) -> dict[str, Any]:
    """The model's judgement under one condition, per candidate: accuracy and Yes rates (Yes means L > 0)."""
    cands = [t for s in states for t in s["candidates"]]
    fails = [t for t in cands if not succeeded(t)]
    det = [t for t in fails if t.get("context_determined")]
    oks = [t for t in cands if succeeded(t)]
    yes = lambda ts: _mean([float(L[t["id"]] > 0) for t in ts])
    return {"n_candidates": len(cands), "accuracy": _mean([float((L[t["id"]] > 0) == succeeded(t)) for t in cands]),
            "yes_rate_failures": yes(fails), "yes_rate_determined_failures": yes(det), "yes_rate_successes": yes(oks),
            "mean_L_failures": _mean([L[t["id"]] for t in fails]), "mean_L_successes": _mean([L[t["id"]] for t in oks])}


def contrasts_for(arms: set[str]) -> list[tuple[str, str]]:
    out = [(ALIGNED, NEUTRAL)]
    if SWAPPED in arms:
        out += [(ALIGNED, SWAPPED), (SWAPPED, NEUTRAL)]
    return out


def model_report(states: list[dict], L: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Everything for one model. `L[arm][transition_id]` must cover every candidate of every state the arm is
    defined at."""
    arms = set(L)
    rep: dict[str, Any] = {"baseline": {a: baseline([s for s in states if a != SWAPPED or s["swap"]], L[a])
                                        for a in sorted(arms)}}
    for a, b in contrasts_for(arms):
        sts = [s for s in states if s["swap"]] if SWAPPED in (a, b) else states
        key = f"{a}_vs_{b}"
        rep[key] = {rule: contrast(rule, sts, L[a], L[b]) for rule in RULES}
        rep[key]["rule2_errors"] = verification_errors(sts, L[a], L[b])
    return rep


def option_c(reports: dict[str, dict], primary: tuple[str, ...] = PRIMARY) -> dict[str, Any]:
    """The rule logged in the design note: worth doing if, for a primary model, under rule 1 or rule 2 (sets with
    the proposal), aligned against neutral changes the choice at >= 5% of states and raises the chosen-failure rate
    with a 95% interval above zero."""
    per: dict[str, Any] = {}
    for tag in primary:
        if tag not in reports:
            per[tag] = "waiting"
            continue
        cells = {}
        for rule in OPTION_C_RULES:
            c = reports[tag][f"{ALIGNED}_vs_{NEUTRAL}"][rule]
            change = c["change_rate"]["estimate"]
            lo = c["fail_rate_difference"]["ci"][0]
            cells[rule] = {"change_rate": change, "fail_rate_difference_ci_low": lo,
                           "changes_enough": change >= OPTION_C_CHANGE, "toward_failure": lo > 0}
        per[tag] = cells
    if any(v == "waiting" for v in per.values()):
        verdict = "waiting for a primary model"
    elif any(c["changes_enough"] and c["toward_failure"] for v in per.values() for c in v.values()):
        verdict = "worth doing"
    elif any(c["changes_enough"] for v in per.values() for c in v.values()):
        verdict = "weak case"
    else:
        verdict = "not worth doing for these models"
    return {"verdict": verdict, "per_model": per, "threshold_change_rate": OPTION_C_CHANGE}


def scores_by_arm(records: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for x in records:
        if x["L"] == x["L"]:
            out[x["condition"]][x["transition_id"]] = x["L"]
    return dict(out)


def complete_states(states: list[dict], L: dict[str, dict[str, float]]) -> tuple[list[dict], int]:
    """The states with a score for every candidate under every arm defined there, and how many fell short."""
    keep = [s for s in states if all(t["id"] in L.get(a, {}) for a in L for t in s["candidates"]
                                     if a != SWAPPED or s["swap"])]
    return keep, len(states) - len(keep)


def run_decisionsim_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    sources = tuple(cfg.get("sources", SOURCES))
    seed = int(cfg.get("seed", C.RNG_SEED))
    all_states = build_states(load_items(sources), seed)
    states = select_states(all_states, cfg.get("n_states", 300), seed)
    digest = sample_digest(states)
    report: dict[str, Any] = {
        "design": "notes/decisionsim_design.md (committed before any decision result); decision-level diagnostic, "
                  "not end-to-end planning harm",
        "states": {"n_eligible": len(all_states), "n_sampled": len(states), "n_games": len({s["game_id"] for s in states}),
                   "sample_digest": digest, "n_candidates": sum(len(s["candidates"]) for s in states),
                   "n_with_swap": sum(bool(s["swap"]) for s in states),
                   "n_failing_proposals": sum(not succeeded(_proposal(s)) for s in states),
                   "n_rule3_states": sum(rule_applies("rule3_goal_relevant", s) for s in states)},
        "models": {}, "waiting": {}}
    folders_all = {}
    for tag, source in cfg["models"].items():
        folders, problems = complete_shards(source)
        for p in folders:
            saved = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
            if saved.get("sample_digest") != digest:
                problems.append(f"{p.name} scored another sample ({saved.get('sample_digest')} != {digest})")
        if problems:
            report["waiting"][tag] = problems
            continue
        folders_all[tag] = folders
        L = scores_by_arm([x for p in folders for x in read_jsonl(p / "scoring.jsonl")])
        if ALIGNED not in L or NEUTRAL not in L:
            report["waiting"][tag] = [f"needs {ALIGNED} and {NEUTRAL}; has {sorted(L)}"]
            continue
        sts, short = complete_states(states, L)
        report["models"][tag] = {"n_states_incomplete": short, **model_report(sts, L)}
    report["option_c"] = option_c(report["models"], tuple(cfg.get("primary", PRIMARY)))
    report["provenance"] = analysis_provenance(None, [p for fs in folders_all.values() for p in fs], sources)
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Decision simulation, Option A (exploratory)", report)
    lines = [f"{tag}: rule 1 change {m[f'{ALIGNED}_vs_{NEUTRAL}']['rule1_with_proposal']['change_rate']['estimate']:.3f}, "
             f"ΔF {m[f'{ALIGNED}_vs_{NEUTRAL}']['rule1_with_proposal']['fail_rate_difference']['estimate']:+.3f}"
             for tag, m in report["models"].items()]
    write_status(run_id, f"Option C: {report['option_c']['verdict']}\n" + "\n".join(lines)
                 + (f"\nwaiting: {sorted(report['waiting'])}" if report["waiting"] else ""),
                 ok=False, reported="decision simulation report")
    return 0
