"""Option A: offline decision simulation (exploratory; design in notes/decisionsim_design.md, committed before any
decision result).

Does the task-induced bias change which action a planner that consults the world model would choose, and how
often does it make that planner choose an action that fails? A decision-level diagnostic, not a measure of
end-to-end planning harm: nothing is executed after the choice.

At eligible states of the collected episodes (the agent's own next action is among the candidates and known, at
least one candidate succeeds and at least one is a failure the history determines), every candidate is scored with
the yes/no feasibility score L = log P(Yes) - log P(No) under the aligned (real) task, the neutral task ("explore the
room") and one state-level swapped task. The candidate set is fixed across conditions. Scores are rounded to 1/64
nat (the bf16 resolution of the logit difference) before any comparison. Three rules choose an action: (1) the
largest L; (2) verification: the agent's proposal if L > 0, else the largest L among the other candidates; (3) the
largest L among goal-relevant candidates (secondary, only where one of them succeeds). Ties go to a seeded per-state
hash of the action, the same under every condition. The analysis compares the chosen-failure rate and the choices
themselves between conditions, by game, on even-step states only: at odd steps the collector ran the broad generator
only when the agent's proposal failed, so those states are selected on the outcome (they enter only rule 2's false
acceptance, which is conditional on a failing proposal, reported by parity). Amendments after review 1 are listed in
the design note.

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
DEFAULT_ARMS = ("yn_aligned", "yn_absent", "yn_swapped")  # amended after review 1: swapped scored for every model
ALIGNED, NEUTRAL, SWAPPED = "yn_aligned", "yn_absent", "yn_swapped"
PRIMARY = ("qwen3-8b", "llama3.1-8b")
# tag: Hugging Face id. Family models take their prompts and loading settings from wf/families.py.
REFERENCE_MODELS = {"qwen3-8b": C.WM_FEWSHOT_MODEL, "llama3.1-8b": C.WM_FEWSHOT_LLAMA}
FAMILY_MODELS = ("mistral-7b-v0.3", "gemma2-9b-it", "llama3.2-3b")
MODEL_TAGS = tuple(REFERENCE_MODELS) + FAMILY_MODELS
RULES = ("rule1_with_proposal", "rule1_generated_only", "rule2_verification", "rule3_goal_relevant")
# The Option C rule (design note, amended after review 1 and before any score): rule 2 only, even-step states.
OPTION_C_RULE = "rule2_verification"
OPTION_C_HARM = 0.05  # task-induced harm rate (aligned choice fails, neutral choice succeeds), share of states
OPTION_C_P = 0.05  # game sign-flip permutation p of the aligned-minus-neutral difference in failure rates
OPTION_C_SWAP_SHARE = 0.5  # aligned-minus-swapped difference, at least this share of aligned-minus-neutral
# Overall verdict when every primary model is in: the first of these that some primary model reaches.
OPTION_C_CATEGORIES = ("worth doing", "toward failure, not attributable to the real task", "toward failure, too rare",
                       "changes toward success", "no reliable change in failures")
# wf/data/transitions.py build_transitions_from_rollout(k=2): the broad generator (whose candidates include the
# agent's proposal) ran at every k-th step, and at the other steps only when the proposal was invalid. States at the
# other steps are therefore selected on the proposal's outcome and are left out of every state-level rate.
BROAD_EVERY_K = 2
GRID = 64  # scores are rounded to 1/GRID nat, the bf16 resolution of the logit difference, before any comparison
TIE_SEED = C.RNG_SEED


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
    states = select_states(build_states(load_items(sources), seed), cfg.get("n_states", "all"), seed)
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


# ---------------- scores, ties and decision rules ----------------


def quantize(x: float) -> float:
    """Round a score to 1/64 nat, the bf16 resolution of the logit difference, so that ties are exact."""
    return round(x * GRID) / GRID


def even_step(state: dict) -> bool:
    """A state where the broad generator ran whatever the agent's proposal did (not selected on its outcome)."""
    return state["state_idx"] % BROAD_EVERY_K == 0


def tie_key(t: dict, seed: int = TIE_SEED) -> str:
    """A seeded per-state hash of the action: blind to outcomes and identical across conditions."""
    return hashlib.sha256(f"{seed}|{t['game_id']}|{t['state_idx']}|{t['action']}".encode()).hexdigest()


def choose_max(cands: list[dict], L: dict[str, float]) -> dict:
    """The candidate with the largest (rounded) L; a tie goes to the smallest tie key."""
    if not cands:
        raise ValueError("no candidates to choose from")
    return min(cands, key=lambda t: (-L[t["id"]], tie_key(t)))


def top_set(cands: list[dict], L: dict[str, float]) -> list[dict]:
    top = max(L[t["id"]] for t in cands)
    return [t for t in cands if L[t["id"]] == top]


def _proposal(state: dict) -> dict:
    return next(t for t in state["candidates"] if t["action"] == state["proposal"])


def _generated(state: dict) -> list[dict]:
    return [t for t in state["candidates"] if t["action"] != state["proposal"]]


def _goal_relevant(state: dict) -> list[dict]:
    return [t for t in state["candidates"] if t.get("targets_goal_object")]


def accepts(L: dict[str, float], t: dict) -> bool:
    """Yes: the model prefers Yes to No, L > 0 (a rounded score of zero is No)."""
    return L[t["id"]] > 0


def rule_set(rule: str, state: dict) -> list[dict]:
    """The candidates a rule ranks (rule 2's full set: the proposal and its fallback)."""
    if rule in ("rule1_with_proposal", "rule2_verification"):
        return state["candidates"]
    if rule == "rule1_generated_only":
        return _generated(state)
    if rule == "rule3_goal_relevant":
        return _goal_relevant(state)
    raise ValueError(rule)


def rule_menu(rule: str, state: dict, L: dict[str, float]) -> list[dict]:
    """The candidates the rule takes the largest L among under these scores: rule 2 takes the proposal alone when it
    is accepted, and otherwise the other candidates, never the rejected proposal."""
    if rule == "rule2_verification":
        p = _proposal(state)
        return [p] if accepts(L, p) else _generated(state)
    return rule_set(rule, state)


def rule_choice(rule: str, state: dict, L: dict[str, float]) -> dict:
    return choose_max(rule_menu(rule, state, L), L)


def rule_applies(rule: str, state: dict) -> bool:
    """Rule 3 is read only where a goal-relevant candidate succeeds; elsewhere it must choose a failure."""
    if rule == "rule3_goal_relevant":
        return any(succeeded(t) for t in _goal_relevant(state))
    return True


# ---------------- measures ----------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _quartiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"q25": float("nan"), "median": float("nan"), "q75": float("nan")}
    v = sorted(xs)
    q = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]
    return {"q25": q(0.25), "median": q(0.5), "q75": q(0.75)}


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


def _tie_counts(cands: list[dict], L: dict[str, float]) -> tuple[bool, bool]:
    """(the top score is shared, the tied top candidates differ in outcome)."""
    top = top_set(cands, L)
    return len(top) > 1, len({succeeded(t) for t in top}) > 1


def _acceptance_cell(acc_a: bool, acc_b: bool) -> str:
    return {(True, True): "accepted_both", (False, False): "rejected_both",
            (True, False): "accepted_only_a", (False, True): "accepted_only_b"}[(acc_a, acc_b)]


def contrast(rule: str, states: list[dict], La: dict[str, float], Lb: dict[str, float]) -> dict[str, Any]:
    """Choices under condition a against condition b, at the even-step states where the rule applies (odd-step
    states are selected on the proposal's outcome and are left out here)."""
    rows = []
    n_odd = 0
    for s in states:
        if not even_step(s):
            n_odd += 1
            continue
        if not rule_applies(rule, s):
            continue
        ma, mb = rule_menu(rule, s, La), rule_menu(rule, s, Lb)
        ca, cb = choose_max(ma, La), choose_max(mb, Lb)
        cands = rule_set(rule, s)
        fa, fb = not succeeded(ca), not succeeded(cb)
        tie_a, mixed_a = _tie_counts(ma, La)
        tie_b, mixed_b = _tie_counts(mb, Lb)
        p = _proposal(s)
        rows.append({"game": s["game_id"], "fail_a": float(fa), "fail_b": float(fb),
                     "harm": float(fa and not fb), "help": float(fb and not fa),
                     "det_a": bool(fa and ca.get("context_determined")), "det_b": bool(fb and cb.get("context_determined")),
                     "changed": ca["action"] != cb["action"],
                     "prop_a": ca["action"] == s["proposal"], "prop_b": cb["action"] == s["proposal"],
                     "random_fail": _mean([float(not succeeded(t)) for t in cands]),
                     "split_a": _mean([float(not succeeded(t)) for t in top_set(ma, La)]),
                     "split_b": _mean([float(not succeeded(t)) for t in top_set(mb, Lb)]),
                     "tie_a": tie_a, "tie_b": tie_b, "mixed_a": mixed_a, "mixed_b": mixed_b,
                     "cell": _acceptance_cell(accepts(La, p), accepts(Lb, p))})
    if not rows:
        return {"n_states": 0, "n_odd_step_states_left_out": n_odd}
    games = [r["game"] for r in rows]
    fa, fb = [r["fail_a"] for r in rows], [r["fail_b"] for r in rows]
    changed = [r for r in rows if r["changed"]]
    harm = sum(1 for r in changed if r["fail_a"] and not r["fail_b"])
    help_ = sum(1 for r in changed if not r["fail_a"] and r["fail_b"])
    both_fail = sum(1 for r in changed if r["fail_a"] and r["fail_b"])
    both_ok = len(changed) - harm - help_ - both_fail
    n = len(rows)
    share = lambda k: k / len(changed) if changed else float("nan")
    by_source = {}
    for side in ("a", "b"):
        by_source[side] = {}
        for name, is_prop in (("proposal", True), ("generated", False)):
            sel = [r[f"fail_{side}"] for r in rows if r[f"prop_{side}"] == is_prop]
            by_source[side][name] = {"n": len(sel), "fail_rate": _mean(sel)}
    out = {
        "n_states": n, "n_games": len(set(games)), "n_odd_step_states_left_out": n_odd,
        "fail_rate_a": _boot(fa, games), "fail_rate_b": _boot(fb, games),
        "fail_rate_difference": _paired(fa, fb, games),
        # task-induced harm and help as shares of all states (not of the changed ones)
        "harm_rate": _boot([r["harm"] for r in rows], games), "help_rate": _boot([r["help"] for r in rows], games),
        "determined_share_of_chosen_failures": {
            "a": sum(r["det_a"] for r in rows) / max(1, sum(fa)), "b": sum(r["det_b"] for r in rows) / max(1, sum(fb))},
        "change_rate": _boot([float(r["changed"]) for r in rows], games),
        "change_rate_outcome_changing": (harm + help_) / n,
        "change_rate_success_to_success": both_ok / n,
        "change_rate_failure_to_failure": both_fail / n,
        "changed": {"n": len(changed), "a_fails_b_succeeds": harm, "a_succeeds_b_fails": help_,
                    "both_fail": both_fail, "both_succeed": both_ok,
                    "share_a_fails_b_succeeds": share(harm), "share_a_succeeds_b_fails": share(help_),
                    "share_both_fail": share(both_fail), "share_both_succeed": share(both_ok)},
        # the identity the design note promises: (harm - help) / n is the difference in failure rates
        "identity_holds": abs((harm - help_) / n - (sum(fa) - sum(fb)) / n) < 1e-12,
        "random_choice_fail_rate": _mean([r["random_fail"] for r in rows]),
        "proposal_share": {"a": _mean([float(r["prop_a"]) for r in rows]), "b": _mean([float(r["prop_b"]) for r in rows])},
        "fail_rate_by_source": by_source,
        "ties": {"a": sum(r["tie_a"] for r in rows), "b": sum(r["tie_b"] for r in rows),
                 "mixed_outcome_a": sum(r["mixed_a"] for r in rows), "mixed_outcome_b": sum(r["mixed_b"] for r in rows)},
        # sensitivity: every tied top set split evenly, so each state counts its tied set's failure share
        "tie_split": {"fail_rate_a": _mean([r["split_a"] for r in rows]), "fail_rate_b": _mean([r["split_b"] for r in rows]),
                      "difference": _mean([r["split_a"] - r["split_b"] for r in rows])},
    }
    if rule == "rule2_verification":
        cells = {}
        for cell in ("accepted_both", "rejected_both", "accepted_only_a", "accepted_only_b"):
            rs = [r for r in rows if r["cell"] == cell]
            cells[cell] = {"n": len(rs), "harm": sum(r["harm"] for r in rs), "help": sum(r["help"] for r in rs),
                           "contribution": sum(r["fail_a"] - r["fail_b"] for r in rs) / n}
        cells["contributions_add_up"] = abs(sum(c["contribution"] for c in cells.values())
                                            - (sum(fa) - sum(fb)) / n) < 1e-12
        out["by_acceptance"] = cells
    return out


def verification_errors(states: list[dict], La: dict[str, float], Lb: dict[str, float]) -> dict[str, Any]:
    """Rule 2's two error rates. False acceptance (failing proposals with L > 0) is conditional on a failing
    proposal, so it is reported at even-step and odd-step states separately and pooled; false rejection
    (succeeding proposals with L <= 0) at even-step states only."""
    def rates(sts: list[dict], keep: Callable[[dict], bool], wrong: Callable[[dict, dict], bool]) -> dict[str, Any]:
        ps = [(s["game_id"], _proposal(s)) for s in sts if keep(_proposal(s))]
        games = [g for g, _ in ps]
        a = [float(wrong(La, p)) for _, p in ps]
        b = [float(wrong(Lb, p)) for _, p in ps]
        return {"n_states": len(ps), "rate_a": _boot(a, games), "rate_b": _boot(b, games),
                "difference": _paired(a, b, games) if ps else {"n_states": 0}}
    even = [s for s in states if even_step(s)]
    odd = [s for s in states if not even_step(s)]
    failing = lambda p: not succeeded(p)
    return {"false_acceptance": {"even_steps": rates(even, failing, accepts), "odd_steps": rates(odd, failing, accepts),
                                 "pooled": rates(states, failing, accepts)},
            "false_rejection": rates(even, succeeded, lambda L, p: not accepts(L, p))}


def _group(state: dict, t: dict) -> str:
    return (f"{'proposal' if t['action'] == state['proposal'] else 'generated'}/"
            f"{'goal_object' if t.get('targets_goal_object') else 'other'}")


def _groups(states: list[dict]) -> dict[str, list[dict]]:
    """Candidates by source (the agent's proposal or generated) x goal relevance x outcome, and over outcomes."""
    out: dict[str, list[dict]] = defaultdict(list)
    for s in states:
        for t in s["candidates"]:
            g = _group(s, t)
            out[f"{g}/{'success' if succeeded(t) else 'failure'}"].append(t)
            out[f"{g}/all"].append(t)
    return dict(sorted(out.items()))


def baseline(states: list[dict], L: dict[str, float], L_raw: dict[str, float] | None = None) -> dict[str, Any]:
    """The model's judgement under one condition, per candidate: accuracy and Yes rates (Yes means rounded L > 0),
    with mean raw L; overall and by source x goal relevance x outcome."""
    raw = L_raw if L_raw is not None else L
    cands = [t for s in states for t in s["candidates"]]
    fails = [t for t in cands if not succeeded(t)]
    det = [t for t in fails if t.get("context_determined")]
    oks = [t for t in cands if succeeded(t)]
    yes = lambda ts: _mean([float(accepts(L, t)) for t in ts])
    return {"n_candidates": len(cands), "accuracy": _mean([float(accepts(L, t) == succeeded(t)) for t in cands]),
            "yes_rate_failures": yes(fails), "yes_rate_determined_failures": yes(det), "yes_rate_successes": yes(oks),
            "mean_L_failures": _mean([raw[t["id"]] for t in fails]), "mean_L_successes": _mean([raw[t["id"]] for t in oks]),
            "by_group": {k: {"n": len(ts), "yes_rate": yes(ts), "mean_L": _mean([raw[t["id"]] for t in ts])}
                         for k, ts in _groups(states).items()}}


def baseline_shifts(states: list[dict], La: dict[str, float], Lb: dict[str, float],
                    La_raw: dict[str, float], Lb_raw: dict[str, float]) -> dict[str, Any]:
    """Condition a minus condition b, per candidate, by source x goal relevance x outcome: the Yes-rate shift and
    the mean shift in raw L."""
    return {k: {"n": len(ts), "yes_rate_shift": _mean([float(accepts(La, t)) - float(accepts(Lb, t)) for t in ts]),
                "mean_L_shift": _mean([La_raw[t["id"]] - Lb_raw[t["id"]] for t in ts])}
            for k, ts in _groups(states).items()}


def calibration(states: list[dict], La: dict[str, float], Ln: dict[str, float]) -> dict[str, Any]:
    """How far rule 1 (full set) is from choosing a failure under the neutral task, and how far the aligned task
    moves the failure that comes closest. Margin: best success minus best failure under neutral (rounded L); shift:
    aligned minus neutral on that best failure. Even-step states."""
    margins, shifts, closes = [], [], []
    for s in states:
        if not even_step(s):
            continue
        oks = [t for t in s["candidates"] if succeeded(t)]
        fails = [t for t in s["candidates"] if not succeeded(t)]
        bf = choose_max(fails, Ln)
        m = max(Ln[t["id"]] for t in oks) - Ln[bf["id"]]
        sh = La[bf["id"]] - Ln[bf["id"]]
        margins.append(m)
        shifts.append(sh)
        if m > 0:
            closes.append(float(sh >= m))
    return {"n_states": len(margins), "neutral_margin": _quartiles(margins),
            "share_margin_not_positive": _mean([float(m <= 0) for m in margins]),
            "aligned_shift_on_best_failure": _quartiles(shifts), "mean_aligned_shift_on_best_failure": _mean(shifts),
            "share_shift_closes_margin": _mean(closes), "n_positive_margin": len(closes)}


def lure_rate(states: list[dict], La: dict[str, float], Lb: dict[str, float]) -> dict[str, Any]:
    """Descriptive small-menu rule 1: at even-step states with a succeeding proposal, the proposal against one
    failing goal-object candidate at a time; a lure is the failure chosen."""
    rows = []
    for s in states:
        if not even_step(s):
            continue
        p = _proposal(s)
        if not succeeded(p):
            continue
        for g in _generated(s):
            if g.get("targets_goal_object") and not succeeded(g):
                rows.append((s["game_id"], s["state_idx"], float(choose_max([p, g], La) is g),
                             float(choose_max([p, g], Lb) is g)))
    games = [r[0] for r in rows]
    a, b = [r[2] for r in rows], [r[3] for r in rows]
    return {"n_pairs": len(rows), "n_states": len({r[:2] for r in rows}),
            "rate_a": _boot(a, games), "rate_b": _boot(b, games),
            "difference": _paired(a, b, games) if rows else {"n_states": 0}}


def decomposition(states: list[dict], L: dict[str, dict[str, float]]) -> dict[str, Any]:
    """aligned - neutral = (aligned - swapped) + (swapped - neutral) in the chosen-failure rate, per rule, on the
    even-step states where the swapped task is defined (all three on the same states, so the identity is exact)."""
    sts = [s for s in states if even_step(s) and s["swap"]]
    out: dict[str, Any] = {}
    for rule in RULES:
        rs = [s for s in sts if rule_applies(rule, s)]
        f = {a: _mean([float(not succeeded(rule_choice(rule, s, L[a]))) for s in rs]) for a in (ALIGNED, NEUTRAL, SWAPPED)}
        an, as_, sn = f[ALIGNED] - f[NEUTRAL], f[ALIGNED] - f[SWAPPED], f[SWAPPED] - f[NEUTRAL]
        out[rule] = {"n_states": len(rs), "fail_rates": f, "aligned_minus_neutral": an, "aligned_minus_swapped": as_,
                     "swapped_minus_neutral": sn, "identity_holds": (not rs) or abs(an - (as_ + sn)) < 1e-12}
    return out


def contrasts_for(arms: set[str]) -> list[tuple[str, str]]:
    out = [(ALIGNED, NEUTRAL)]
    if SWAPPED in arms:
        out += [(ALIGNED, SWAPPED), (SWAPPED, NEUTRAL)]
    return out


def model_report(states: list[dict], L: dict[str, dict[str, float]],
                 L_raw: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Everything for one model. `L[arm][transition_id]` (rounded scores) must cover every candidate of every state
    the arm is defined at; `L_raw` (unrounded) feeds only the mean-L readouts. Every state-level rate is on even-step
    states; odd-step states enter only rule 2's false acceptance."""
    raw = L_raw if L_raw is not None else L
    arms = set(L)
    even = [s for s in states if even_step(s)]
    even_swap = [s for s in even if s["swap"]]
    rep: dict[str, Any] = {
        "n_states": {"all": len(states), "even_step": len(even), "odd_step": len(states) - len(even),
                     "even_step_with_swap": len(even_swap)},
        "baseline": {a: baseline(even_swap if a == SWAPPED else even, L[a], raw[a]) for a in sorted(arms)},
        "baseline_shifts": {f"{ALIGNED}_minus_{NEUTRAL}": baseline_shifts(even, L[ALIGNED], L[NEUTRAL], raw[ALIGNED], raw[NEUTRAL])}}
    for a, b in contrasts_for(arms):
        if SWAPPED in (a, b):
            rep["baseline_shifts"][f"{a}_minus_{b}"] = baseline_shifts(even_swap, L[a], L[b], raw[a], raw[b])
    for a, b in contrasts_for(arms):
        sts = [s for s in states if s["swap"]] if SWAPPED in (a, b) else states
        key = f"{a}_vs_{b}"
        rep[key] = {rule: contrast(rule, sts, L[a], L[b]) for rule in RULES}
        rep[key]["rule2_errors"] = verification_errors(sts, L[a], L[b])
        rep[key]["lure"] = lure_rate(sts, L[a], L[b])
    rep["calibration"] = calibration(states, L[ALIGNED], L[NEUTRAL])
    if SWAPPED in arms:
        rep["decomposition"] = decomposition(states, L)
    return rep


def option_c_model(rep: dict[str, Any]) -> dict[str, Any]:
    """One primary model's category under the Option C rule (rule 2, even-step states, aligned against neutral)."""
    c = rep[f"{ALIGNED}_vs_{NEUTRAL}"][OPTION_C_RULE]
    lo, hi = c["fail_rate_difference"]["ci"]
    p = c["fail_rate_difference"]["p"]
    harm = c["harm_rate"]["estimate"]
    dec = rep["decomposition"][OPTION_C_RULE]
    toward_failure = lo > 0 and p < OPTION_C_P
    attributable = (dec["aligned_minus_swapped"] > 0
                    and dec["aligned_minus_swapped"] >= OPTION_C_SWAP_SHARE * dec["aligned_minus_neutral"])
    if toward_failure and harm >= OPTION_C_HARM and attributable:
        category = "worth doing"
    elif toward_failure and harm >= OPTION_C_HARM:
        category = "toward failure, not attributable to the real task"
    elif toward_failure:
        category = "toward failure, too rare"
    elif hi < 0:
        category = "changes toward success"
    else:
        category = "no reliable change in failures"
    r1 = rep[f"{ALIGNED}_vs_{NEUTRAL}"]["rule1_with_proposal"]
    return {"category": category, "harm_rate": harm, "fail_rate_difference": c["fail_rate_difference"]["estimate"],
            "fail_rate_difference_ci": [lo, hi], "p": p, "toward_failure": toward_failure,
            "harm_enough": harm >= OPTION_C_HARM, "attributable": attributable,
            "aligned_minus_swapped": dec["aligned_minus_swapped"],
            "aligned_minus_neutral_same_states": dec["aligned_minus_neutral"],
            "change_rate": c["change_rate"]["estimate"],
            "change_rate_outcome_changing": c["change_rate_outcome_changing"],
            "change_rate_success_to_success": c["change_rate_success_to_success"],
            # reported, never decisive: rule 1's difference follows from the item-level effect and the menu
            "rule1_reported_only": {"fail_rate_difference": r1["fail_rate_difference"]["estimate"],
                                    "fail_rate_difference_ci": r1["fail_rate_difference"]["ci"],
                                    "harm_rate": r1["harm_rate"]["estimate"], "change_rate": r1["change_rate"]["estimate"]}}


def option_c(reports: dict[str, dict], primary: tuple[str, ...] = PRIMARY) -> dict[str, Any]:
    """The rule in the design note (amended after review 1, before any score). Option C is worth doing if, for at
    least one primary model, on even-step states, under rule 2 (verification of the agent's own proposal, with
    fallback), the task-induced harm rate is at least 5% of states, the aligned-minus-neutral difference in failure
    rates has a 95% game-bootstrap interval above zero and a game-permutation p below 0.05, and the aligned-minus-
    swapped difference under the same rule is positive and at least half the aligned-minus-neutral one."""
    per: dict[str, Any] = {}
    for tag in primary:
        if tag not in reports:
            per[tag] = "waiting"
        elif "decomposition" not in reports[tag]:
            per[tag] = "waiting: swapped arm not scored"
        else:
            per[tag] = option_c_model(reports[tag])
    if any(isinstance(v, str) for v in per.values()):
        verdict = "waiting for a primary model"
    else:
        reached = {v["category"] for v in per.values()}
        verdict = next(c for c in OPTION_C_CATEGORIES if c in reached)
    return {"verdict": verdict, "worth_doing": verdict == "worth doing", "per_model": per, "rule": OPTION_C_RULE,
            "thresholds": {"harm_rate": OPTION_C_HARM, "p": OPTION_C_P, "swap_share": OPTION_C_SWAP_SHARE}}


def scores_by_arm(records: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """(rounded, raw) scores by arm and transition id. Every decision, tie and threshold reads the rounded ones."""
    q: dict[str, dict[str, float]] = defaultdict(dict)
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    for x in records:
        if x["L"] == x["L"]:
            raw[x["condition"]][x["transition_id"]] = x["L"]
            q[x["condition"]][x["transition_id"]] = quantize(x["L"])
    return dict(q), dict(raw)


def complete_states(states: list[dict], L: dict[str, dict[str, float]]) -> tuple[list[dict], int]:
    """The states with a score for every candidate under every arm defined there, and how many fell short."""
    keep = [s for s in states if all(t["id"] in L.get(a, {}) for a in L for t in s["candidates"]
                                     if a != SWAPPED or s["swap"])]
    return keep, len(states) - len(keep)


def analyse_model(states: list[dict], records: list[dict]) -> tuple[dict[str, Any] | None, list[str]]:
    """One model's report from its saved score records, or (None, problems)."""
    Lq, Lraw = scores_by_arm(records)
    if ALIGNED not in Lq or NEUTRAL not in Lq:
        return None, [f"needs {ALIGNED} and {NEUTRAL}; has {sorted(Lq)}"]
    sts, short = complete_states(states, Lq)
    return {"n_states_incomplete": short, **model_report(sts, Lq, Lraw)}, []


def state_counts(all_states: list[dict], states: list[dict]) -> dict[str, Any]:
    even = [s for s in states if even_step(s)]
    failing = lambda ss: sum(not succeeded(_proposal(s)) for s in ss)
    return {"n_eligible": len(all_states), "n_selected": len(states), "n_games": len({s["game_id"] for s in states}),
            "n_candidates": sum(len(s["candidates"]) for s in states),
            "n_even_step": len(even), "n_odd_step": len(states) - len(even),
            "n_even_step_games": len({s["game_id"] for s in even}),
            "n_with_swap": sum(bool(s["swap"]) for s in states), "n_even_step_with_swap": sum(bool(s["swap"]) for s in even),
            "n_failing_proposals": {"even_step": failing(even), "odd_step": failing(states) - failing(even)},
            "n_failing_goal_object_proposals": sum(not succeeded(_proposal(s)) and bool(_proposal(s).get("targets_goal_object"))
                                                   for s in states),
            "n_rule3_states_even_step": sum(rule_applies("rule3_goal_relevant", s) for s in even),
            "n_scores_per_model": sum(len(s["candidates"]) * (3 if s["swap"] else 2) for s in states)}


def run_decisionsim_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    sources = tuple(cfg.get("sources", SOURCES))
    seed = int(cfg.get("seed", C.RNG_SEED))
    all_states = build_states(load_items(sources), seed)
    states = select_states(all_states, cfg.get("n_states", "all"), seed)
    digest = sample_digest(states)
    report: dict[str, Any] = {
        "design": "notes/decisionsim_design.md (committed before any decision result, amended after review 1 before "
                  "any score); decision-level diagnostic, not end-to-end planning harm",
        "states": {**state_counts(all_states, states), "sample_digest": digest},
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
        rep, problems = analyse_model(states, [x for p in folders for x in read_jsonl(p / "scoring.jsonl")])
        if rep is None:
            report["waiting"][tag] = problems
            continue
        report["models"][tag] = rep
    report["option_c"] = option_c(report["models"], tuple(cfg.get("primary", PRIMARY)))
    report["provenance"] = analysis_provenance(None, [p for fs in folders_all.values() for p in fs], sources)
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Decision simulation, Option A (exploratory)", report)
    key = f"{ALIGNED}_vs_{NEUTRAL}"
    lines = [f"{tag}: rule 2 harm {m[key][OPTION_C_RULE]['harm_rate']['estimate']:.3f}, "
             f"ΔF {m[key][OPTION_C_RULE]['fail_rate_difference']['estimate']:+.3f}"
             for tag, m in report["models"].items()]
    write_status(run_id, f"Option C: {report['option_c']['verdict']}\n" + "\n".join(lines)
                 + (f"\nwaiting: {sorted(report['waiting'])}" if report["waiting"] else ""),
                 ok=False, reported="decision simulation report")
    return 0
