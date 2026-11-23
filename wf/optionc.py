"""Option C: the RQ3 agent in closed loop with a world-model verifier (design in notes/optionc_design.md, committed
before any Option C data or code).

The RQ3 Qwen3-8B ReAct agent (greedy, thinking off) plays the 134 valid_unseen games to the 30-step cap. Before each
action a verifier scores the agent's greedy proposal under its arm's prompt: no verifier (none), the task line deleted
(blind), the real task line (aware), the task line with its object swapped for another (swapped) or "explore the room"
(neutral). Primary rule: execute the proposal if accepted; otherwise sample the agent's own alternatives (Stage 1b's
sampler and ranking) and execute the first one the verifier accepts, else the proposal. Secondary rule: Stage 1a's
rule 2, the highest-scored alternative whatever its sign. Readouts: the yes/no question (primary) and the first token
of the predicted observation, log(1 - P(Nothing)) - log P(Nothing) (secondary). The verifier is invisible to the agent.

Within a game every agent reply, verifier score and counterfactual outcome is cached under exactly what produced it, so
arms play the same episode until their verifiers first choose differently (common random numbers).

Runs:
  optionc_step0_<model>      GPU: both readouts, four arms, at Stage 1a's step-0 states (the task-success check)
  optionc_step0_analysis     CPU: its classification and the observation readout's competence gate
  optionc_<model>_<readout>  GPU: vLLM agent + verifier + ALFWorld server; the episodes (optionc_smoke: two train games)
  optionc_analysis           CPU: success, invalid actions, overrides, loops; game-paired contrasts and verdicts
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf.agent.react import FORMAT_ERROR_ACTION, AgentClient
from wf.analysis.report import write_run_report
from wf.data.goals import make_swapped_task_line, names_object, parse_task_line, swap_feasible, swapped_goal
from wf.decisionsim_agent import (
    COMPETENCE,
    MAX_TOKENS,
    PRIMARY_LIST,
    SAMPLINGS,
    AgentSampler,
    _boot,
    _paired,
    auc,
    blind_messages,
    complete_shards,
    rank_candidates,
    resolve_run,
    shard_games,
    shard_spec,
    tie_key,
)
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.scene import scan_scene
from wf.runs import _client, _games, _init_run
from wf.scoring.prompts import agent_react_messages, parse_react_action
from wf.util import append_jsonl, read_jsonl, repair_jsonl, require_filed, write_json, write_status

NOTHING = "Nothing happens."
SEED = C.RNG_SEED
GRID = 64  # scores are rounded to 1/64 nat before any comparison
# Arms: only the verifier's task line differs.
NONE, BLIND, AWARE, SWAPPED, NEUTRAL = "none", "blind", "aware", "swapped", "neutral"
VERIFIER_ARMS = (BLIND, AWARE, SWAPPED, NEUTRAL)  # also the order in which a game's episodes are played
ARMS = (NONE,) + VERIFIER_ARMS
# Readouts and the earlier prompts each arm's prompt is built from (blind: the aware prompt minus its task line).
YN, OBS = "yn", "obs"
READOUTS = (YN, OBS)
BASE_ARM = {YN: {AWARE: "yn_aligned", SWAPPED: "yn_swapped", NEUTRAL: "yn_absent"},
            OBS: {AWARE: "aligned", SWAPPED: "swapped", NEUTRAL: "absent"}}
FAIL_TOKEN, SUCCESS_TOKEN = "Nothing", "You"
# Decision rules.
ACCEPTED_ONLY, ARGMAX = "accepted_only", "argmax"
RULES = (ACCEPTED_ONLY, ARGMAX)
MODEL_TAGS = ("qwen3-8b", "llama3.1-8b", "gemma2-9b-it")
PRIMARY = {"model": "qwen3-8b", "readout": YN, "rule": ACCEPTED_ONLY}
# Alternatives: Stage 1b's primary list settings.
SAMPLING = SAMPLINGS[PRIMARY_LIST]
N_SAMPLES = 20
K_MAX = 8
# The step-0 check (design note, "The step-0 check", amended after review 1): fixed before any score. Each readout's
# aware-minus-blind gap in accepting the working step-0 proposals, paired by state, with its 95% game interval:
COLLAPSE_GAP = 0.30  # collapse: the interval's upper end is below -0.30 (aware accepts at least 30 points fewer)
NO_GAP = 0.10  # no gap: the whole interval lies within -0.10 to +0.10
# The obstruction check (amended after review 1, must-fix M1): the swapped verifier obstructs the real task when it
# accepts the agent's working proposals on the real goal object more than this much less often than the aware one.
OBSTRUCTION_MARGIN = 0.10
FIRST_STEPS = 10  # key secondary: invalid actions among the first 10 steps
SHADOW_ARMS = (BLIND, SWAPPED)  # matched-state scoring at every step of the aware episodes (never acted on)
COST_RULE = 1.5  # a run stops before its first episode if the smoke projects it above 1.5 times its upper estimate
SMOKE_RUN = "optionc_smoke"
# Verdicts.
ALPHA = 0.05
SWAP_SHARE = 0.5
TOL = 1e-12
SMOKE_FORMAT_MAX = 0.05
SHARD_COMPLETE = "shard complete"
STEP0_ANALYSIS = "optionc_step0_analysis"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def quantize(x: float) -> float:
    return round(x * GRID) / GRID


def accepts(L: float) -> bool:
    return L > 0


# ---------------- verifier prompts and readouts ----------------


def _builder(tag: str) -> Callable[[dict, str], list[dict[str, str]] | None]:
    from wf.decisionsim import messages_for  # the earlier runs' prompts, per model (Gemma's system text folded)

    return messages_for(tag)


def verifier_item(game: str, t: int, task_line: str, history_text: str, action: str,
                  swap: str | None = None) -> dict:
    """What the prompt builders read. The swapped line replaces only the goal object (make_swapped_task_line)."""
    goal = parse_task_line(task_line).goal_object
    return {"id": f"{game}#{t:02d}#{_sha(action)}", "game_id": game, "state_idx": t, "task_line": task_line,
            "goal_object": goal, "history_text": history_text, "action": action, "swapped_goal_object": swap,
            "swapped_task_line": make_swapped_task_line(task_line, goal, swap) if swap else None}


def verifier_messages(tag: str, item: dict, arm: str, readout: str) -> list[dict[str, str]]:
    """The arm's prompt under the readout: yes/no (rq1v's yn_* prompts) or the next-observation prompt (rq1's). Only
    the query's task line differs between arms; blind deletes it with Stage 1b's guards."""
    if readout not in READOUTS:
        raise ValueError(readout)
    build = _builder(tag)
    if arm == BLIND:
        return blind_messages(build(item, BASE_ARM[readout][AWARE]), item["history_text"], item["task_line"])
    if arm == SWAPPED and not item.get("swapped_task_line"):
        raise ValueError("the swapped arm needs a swap object")
    if arm not in BASE_ARM[readout]:
        raise ValueError(arm)
    msgs = build(item, BASE_ARM[readout][arm])
    if msgs is None:
        raise ValueError(f"no {arm} prompt for {item['id']}")
    return msgs


def log_odds_not(lp_not: float, lp_fail: float) -> float:
    return lp_not - lp_fail


def read_score(scorer: Any, msgs: list[dict[str, str]], readout: str, tid: str = "", cond: str = "") -> dict[str, float]:
    """One forward pass. yes/no: L = log P(Yes) - log P(No). obs: L = log(1 - P(Nothing)) - log P(Nothing), with
    L_you = log P(You) - log P(Nothing) from the same pass. L is rounded to 1/64 nat; the raw value is kept."""
    if readout == YN and getattr(scorer, "mock", False):
        raw = _mock_yn(msgs)
        return {"L": quantize(raw), "L_raw": raw, "logp_yes": None, "logp_no": None}
    if readout == YN:
        # The same next-token log-probabilities score_one_token_pair reads (one pass, the last position), through the
        # method that can keep only that position's logits (review 1, S6).
        lp = scorer.next_token_logprobs(msgs, ("Yes", "No"))
        raw = lp["Yes"] - lp["No"]
        if not math.isfinite(raw):
            raise ValueError(f"non-finite yes/no readout for {tid}/{cond}: {lp}")
        return {"L": quantize(raw), "L_raw": raw, "logp_yes": lp["Yes"], "logp_no": lp["No"]}
    lp = scorer.next_token_logprobs(msgs, (FAIL_TOKEN, SUCCESS_TOKEN), complement=FAIL_TOKEN)
    raw = log_odds_not(lp[f"not:{FAIL_TOKEN}"], lp[FAIL_TOKEN])
    if not math.isfinite(raw):
        raise ValueError(f"non-finite observation readout for {tid}/{cond}: {lp}")
    return {"L": quantize(raw), "L_raw": raw, "logp_nothing": lp[FAIL_TOKEN], "logp_not_nothing": lp[f"not:{FAIL_TOKEN}"],
            "logp_you": lp[SUCCESS_TOKEN], "L_you": lp[SUCCESS_TOKEN] - lp[FAIL_TOKEN]}


def _mock_yn(msgs: list[dict[str, str]]) -> float:
    """The mock scorer's yes/no stand-in (WF_MOCK_SCORE=1): the scorer's own mock puts Yes and No at the same
    log-probability, so every L would round to zero and every action would be rejected. This one accepts about two
    thirds of prompts, by a hash of the whole prompt, so arms whose prompts differ can disagree."""
    h = int(hashlib.sha256(repr(msgs).encode()).hexdigest()[:4], 16)
    return 2.0 + (h % 64) / 8 if h % 3 else -2.0 - (h % 64) / 8


def first_token_gate(scorer: Any, prompts: list[list[dict[str, str]]]) -> dict[str, Any]:
    """The observation readout's tokenization check, through the scorer's own encoding: "Nothing" and "You" are single,
    different tokens, and after every prompt the template writes the reply "Nothing happens." starting with exactly
    that "Nothing" token."""
    from wf import families

    problems: list[str] = []
    ids = {w: families._ids(scorer._encode_candidate(w)) for w in (FAIL_TOKEN, SUCCESS_TOKEN)}
    for w, i in ids.items():
        if len(i) != 1:
            problems.append(f"{w!r} is not one token: {i}")
    if ids[FAIL_TOKEN] == ids[SUCCESS_TOKEN]:
        problems.append("'Nothing' and 'You' are the same token")
    for k, msgs in enumerate(prompts):
        try:
            reply = families.template_reply(scorer, msgs, NOTHING)
        except families.TokenizationMismatch as e:
            problems.append(f"prompt {k}: {e}")
            continue
        if not reply or reply[:1] != ids[FAIL_TOKEN][:1]:
            problems.append(f"prompt {k}: the reply starts {reply[:2]}, not {ids[FAIL_TOKEN]}")
    return {"ok": not problems and bool(prompts), "n_prompts": len(prompts), "problems": problems[:50],
            "n_problems": len(problems), "ids": ids}


# ---------------- the per-game cache (common random numbers) ----------------


class GameCache:
    """Every agent reply, sample list, verifier score and counterfactual reply of one game, keyed by exactly what
    produced it (the prompt, or the prefix and action), appended to the run's cache file as it is made."""

    KINDS = ("greedy", "samples", "score", "cf", "swap")

    def __init__(self, path: Path | None, game: str, rows: list[dict] | None = None):
        self.path, self.game = path, game
        self.data: dict[str, dict[str, Any]] = {k: {} for k in self.KINDS}
        self.hits = Counter()
        self.misses = Counter()
        for r in rows or []:
            if r["game_id"] == game:
                self.data[r["kind"]][r["key"]] = r["value"]

    def get(self, kind: str, key: str, make: Callable[[], Any]) -> Any:
        if key in self.data[kind]:
            self.hits[kind] += 1
            return self.data[kind][key]
        self.misses[kind] += 1
        value = make()
        self.data[kind][key] = value
        if self.path is not None:
            append_jsonl(self.path, [{"game_id": self.game, "kind": kind, "key": key, "value": value}])
        return value


# ---------------- decisions ----------------


def decide(rule: str, proposal: str, score: Callable[[str], float], alternatives: Callable[[], list[str]],
           tie: Callable[[str], str]) -> dict[str, Any]:
    """The rule's choice. `score(action)` is the verifier's rounded score; `alternatives()` the agent's other ranked
    candidates, drawn only when the proposal is rejected. Accepted-only: the first accepted alternative in rank order,
    else the proposal. Argmax (Stage 1a's rule 2): the highest-scored alternative whatever its sign, ties by the seeded
    hash; with no alternative, the proposal."""
    if rule not in RULES:
        raise ValueError(rule)
    L1 = score(proposal)
    out: dict[str, Any] = {"proposal_L": L1, "proposal_accepted": accepts(L1), "alternatives": None, "alt_L": {}}
    if accepts(L1):
        return {**out, "chosen": proposal, "reason": "accepted"}
    alts = alternatives()
    out["alternatives"] = alts
    if not alts:
        return {**out, "chosen": proposal, "reason": "no_alternative"}
    if rule == ACCEPTED_ONLY:
        for a in alts:
            out["alt_L"][a] = score(a)
            if accepts(out["alt_L"][a]):
                return {**out, "chosen": a, "reason": "accepted_alternative"}
        return {**out, "chosen": proposal, "reason": "abstained"}
    for a in alts:
        out["alt_L"][a] = score(a)
    best = min(alts, key=lambda a: (-out["alt_L"][a], tie(a)))
    return {**out, "chosen": best, "reason": "argmax"}


def sample_seed(seed: int, game: str, prompt_sha: str) -> int:
    """The alternatives' request seed: the sampling seed, the game and the exact agent prompt, nothing that depends on
    the verifier arm."""
    return int(hashlib.sha256(f"{SEED}|optionc|{seed}|{game}|{prompt_sha}".encode()).hexdigest()[:8], 16)


def game_swap_object(client: Any, game: str, split: str, task_line: str) -> str | None:
    """One swap object per game, by the collection's rules: a type in the room (scene scan, never shown to a model),
    not the goal, that could do the task by the game's own flags; seeded per game."""
    scene = scan_scene(client, game, split)
    goal = parse_task_line(task_line).goal_object
    return swapped_goal(scene.objects, goal, None, rng=random.Random(f"{SEED}|optionc|{game}"),
                        locations=scene.locations, exclude=set(),
                        feasible=lambda o: swap_feasible(task_line, o, scene.traits, len(scene.instances(o))))


# ---------------- episodes ----------------


class Models:
    """The agent (greedy, as in RQ3), its sampler, and the verifier scorer (None for a run with no verifier arm)."""

    def __init__(self, agent: Any, sampler: Any, scorer: Any, tag: str, readout: str):
        self.agent, self.sampler, self.scorer, self.tag, self.readout = agent, sampler, scorer, tag, readout


def play_episode(env: Any, models: Models, game: str, split: str, arm: str, rule: str | None, seed: int | None,
                 cache: GameCache, swap: str | None = None, step_cap: int = C.STEP_CAP, n_samples: int = N_SAMPLES,
                 k_max: int = K_MAX, shadow: tuple[str, ...] = ()) -> dict[str, Any]:
    """One episode. `shadow` (aware episodes only): at every step, the proposal and every alternative the arm scored are
    also scored under these arms' prompts, never acted on (matched-state attribution, amended after review 1)."""
    if arm not in ARMS:
        raise ValueError(arm)
    if arm != NONE and (rule not in RULES or seed is None):
        raise ValueError("a verifier arm needs a rule and a seed")
    if arm == SWAPPED and not swap:
        raise ValueError(f"{game}: no swap object, so no swapped arm")
    if env._env_id is None:
        env.create()
    state = env.reset(game, split)
    task_line = state.task_line
    goal = parse_task_line(task_line).goal_object
    history: list[dict[str, str]] = []
    prefix: list[str] = []
    trace: list[dict] = []
    seen: set[tuple[str, str]] = set()
    t0 = time.monotonic()
    sampling_seconds = [0.0]
    for t in range(step_cap):
        if state.done:
            break
        msgs = agent_react_messages(observation=state.observation, history=history, admissible=state.admissible)
        psha = _sha(msgs)
        raw = cache.get("greedy", psha, lambda: models.agent.complete(msgs))
        parsed = parse_react_action(raw)
        proposal = parsed or FORMAT_ERROR_ACTION
        rec: dict[str, Any] = {"t": t, "proposal": proposal, "proposal_raw": raw, "proposal_format_error": parsed is None,
                               "prompt_sha": psha}
        if arm == NONE:
            chosen, decision = proposal, {"reason": "no_verifier"}
        else:
            hist = history_text_from_state(state)

            def score_as(as_arm: str, action: str) -> float:
                item = verifier_item(game, t, task_line, hist, action, swap)
                vm = verifier_messages(models.tag, item, as_arm, models.readout)
                key = _sha([models.tag, models.readout, vm])
                return cache.get("score", key, lambda: read_score(models.scorer, vm, models.readout, item["id"], as_arm))["L"]

            def score(action: str) -> float:
                return score_as(arm, action)

            def draw(rs: int) -> list[str]:
                t1 = time.monotonic()
                texts = [r["text"] for r in models.sampler.sample(msgs, n_samples, SAMPLING, MAX_TOKENS, rs)]
                sampling_seconds[0] += time.monotonic() - t1
                return texts

            def alternatives() -> list[str]:
                rs = sample_seed(seed, game, psha)
                texts = cache.get("samples", f"{rs}|{psha}", lambda: draw(rs))
                ranked = rank_candidates(proposal, texts, game, t, k_max, SEED)
                rec["candidates"] = [{"action": c["action"], "count": c["count"]} for c in ranked["candidates"]]
                rec["n_sample_format_errors"] = ranked["n_format_errors"]
                return [c["action"] for c in ranked["candidates"][1:]]

            decision = decide(rule, proposal, score, alternatives, lambda a: tie_key(game, t, a, SEED))
            chosen = decision["chosen"]
            if arm == AWARE and shadow:
                rec["shadow"] = {sa: {"proposal_L": score_as(sa, proposal),
                                      "alt_L": {a: score_as(sa, a) for a in decision["alt_L"]}}
                                 for sa in shadow if sa != AWARE and (sa != SWAPPED or swap)}
        new = env.step(state, chosen)
        obs = new.observation
        rec.update({"chosen": chosen, "reason": decision["reason"], "real_obs": obs, "failed": obs.strip() == NOTHING,
                    "won": bool(new.won), "override": chosen != proposal,
                    "chosen_targets_goal": names_object(chosen, goal),
                    "proposal_targets_goal": names_object(proposal, goal),
                    "repeat": (chosen, obs) in seen})
        if arm != NONE:
            rec.update({"proposal_L": decision["proposal_L"], "proposal_accepted": decision["proposal_accepted"],
                        "alternatives": decision["alternatives"], "alt_L": decision["alt_L"]})
        if chosen != proposal:
            # The proposal's own outcome, on a private replay of the state, so the override can be classified.
            cf = cache.get("cf", _sha([prefix, proposal]), lambda: counterfactual(env, game, split, prefix, proposal))
            rec["proposal_obs"] = cf
            rec["proposal_failed"] = cf.strip() == NOTHING
        else:
            rec["proposal_obs"], rec["proposal_failed"] = obs, rec["failed"]
        seen.add((chosen, obs))
        trace.append(rec)
        history.append({"observation": state.observation, "action": chosen})
        prefix.append(chosen)
        state = new
    ep = {"game_id": game, "split": split, "arm": arm, "rule": rule if arm != NONE else None,
          "seed": seed if arm != NONE else None, "model": models.tag if arm != NONE else None,
          "readout": models.readout if arm != NONE else None, "task_line": task_line,
          "swap_object": swap if arm == SWAPPED else None, "game_swap_object": swap, "steps": len(trace), "success": bool(state.won),
          "trace": trace, "seconds": round(time.monotonic() - t0, 3), "seconds_sampling": round(sampling_seconds[0], 3)}
    return {**ep, **episode_summary(ep, step_cap)}


def counterfactual(env: Any, game: str, split: str, prefix: list[str], action: str) -> str:
    handle = replay(env.sandbox(), game, list(prefix), split=split)
    obs, _ = step_counterfactual(handle, action)
    return obs


def episode_summary(ep: dict, step_cap: int = C.STEP_CAP) -> dict[str, Any]:
    tr = ep["trace"]
    run = longest = 0
    for s in tr:
        run = run + 1 if s.get("proposal_accepted") is False else 0
        longest = max(longest, run)
    ov = [s for s in tr if s["override"]]
    return {
        "score_S": -ep["steps"] if ep["success"] else -(step_cap + 1),
        "n_invalid": sum(s["failed"] for s in tr),
        "n_override": len(ov),
        "n_override_harm": sum((not s["proposal_failed"]) and s["failed"] for s in ov),
        "n_override_help": sum(s["proposal_failed"] and not s["failed"] for s in ov),
        # key secondaries after review 1 (S1): the verifier's own net contribution to failures, and early failures
        "override_net_harm": (sum((not s["proposal_failed"]) and s["failed"] for s in ov)
                              - sum(s["proposal_failed"] and not s["failed"] for s in ov)),
        "n_invalid_first10": sum(s["failed"] for s in tr[:FIRST_STEPS]),
        "n_rejected": sum(s.get("proposal_accepted") is False for s in tr),
        "n_abstained": sum(s["reason"] in ("abstained", "no_alternative") for s in tr),
        "step0_override": bool(tr and tr[0]["override"]),
        "step0_rejected": bool(tr and tr[0].get("proposal_accepted") is False),
        "n_repeats": sum(s["repeat"] for s in tr),
        "longest_rejection_run": longest,
        "n_format_errors": sum(s["proposal_format_error"] for s in tr),
    }


def seeds_of(arms: list[str], seeds: list[int] | dict[str, list[int]]) -> dict[str, list[int]]:
    """Seeds per verifier arm: one list for every arm, or a list per arm (`seeds_by_arm`, review 1: the swapped arm can
    run at fewer seeds). The no-verifier arm never samples, so it has no seed: one episode per game serves every seed."""
    if isinstance(seeds, dict):
        return {a: [int(x) for x in seeds.get(a, [])] for a in arms if a != NONE}
    return {a: [int(x) for x in seeds] for a in arms if a != NONE}


def episode_jobs(arms: list[str], rules: list[str], seeds: list[int] | dict[str, list[int]]
                 ) -> list[tuple[str, str | None, int | None]]:
    """A game's episodes in play order: no verifier first (its greedy replies seed the cache), then, rule by rule and
    seed by seed, the verifier arms that run at that seed, in the fixed order."""
    by_arm = seeds_of(arms, seeds)
    jobs: list[tuple[str, str | None, int | None]] = [(NONE, None, None)] if NONE in arms else []
    for rule in rules:
        for s in sorted({x for v in by_arm.values() for x in v}):
            jobs += [(a, rule, s) for a in VERIFIER_ARMS if a in arms and s in by_arm[a]]
    return jobs


def episode_key(ep: dict) -> tuple:
    return (ep["game_id"], ep["arm"], ep["rule"], ep["seed"])


# ---------------- the episode run ----------------


def obs_gate_check(cfg: dict) -> dict[str, Any]:
    """The observation readout's episodes run only if the step-0 analysis found the readout competent for this model
    (design note); an explicit override (`obs_gate_override`, with a reason) is logged."""
    name = cfg.get("step0_analysis", STEP0_ANALYSIS)
    d = resolve_run(name)
    why = None
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else None
    status = (d / "STATUS.md").read_text(encoding="utf-8").splitlines()[0] if (d / "STATUS.md").exists() else ""
    if m is None or "REPORTED" not in status:
        why = f"{name} has not finished"
    elif not m.get("models", {}).get(cfg["model"], {}).get("obs_competence", {}).get("passed"):
        why = f"{name}: the observation readout is not competent for {cfg['model']} (or was not scored)"
    if why and cfg.get("obs_gate_override"):
        return {"ok": True, "step0_analysis": name, "override": str(cfg["obs_gate_override"]), "would_have_refused": why}
    return {"ok": why is None, "step0_analysis": name, "why": why}


def _status(d: Path) -> str:
    return (d / "STATUS.md").read_text(encoding="utf-8").splitlines()[0] if (d / "STATUS.md").exists() else ""


def _metrics(d: Path) -> dict | None:
    return json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else None


MEMSMOKE_MODELS = ("gemma2-9b-it",)  # verifiers whose memory beside vLLM must be shown on a 30-step prompt first


def project_gpu_hours(smoke_metrics: list[dict], jobs: list[tuple], n_games: int) -> float | None:
    """GPU-hours a run would take at the smoke's seconds per episode of each arm (arms the smoke did not play take the
    slowest arm's figure); no credit for the cache, so the projection leans high."""
    secs: dict[str, float] = defaultdict(float)
    counts: Counter = Counter()
    for m in smoke_metrics:
        for a, x in m.get("seconds_by_arm", {}).items():
            secs[a] += x
            counts[a] += m.get("by_arm", {}).get(a, 0)
    per = {a: secs[a] / counts[a] for a in secs if counts[a]}
    if not per:
        return None
    return n_games * sum(per.get(a, max(per.values())) for a, _, _ in jobs) / 3600


def order_gate_check(cfg: dict, jobs: list[tuple]) -> dict[str, Any]:
    """Every episode run except the smoke itself (review 1, S5): every smoke shard REPORTED with its smoke gate passed
    (format errors and the cache check), the step-0 analysis REPORTED, the cost rule (the smoke's projection at most
    COST_RULE times the run's `cost_upper_gpu_hours`), and for Gemma a passed memory smoke. An explicit
    `order_override` (with a reason) lets a run start anyway and is logged."""
    problems: list[str] = []
    smoke = resolve_run(cfg.get("smoke_run", SMOKE_RUN))
    shards = sorted(p for p in smoke.glob("shard_*_of_*") if p.is_dir())
    passed = []
    if not shards:
        problems.append(f"no smoke run at {smoke.name}")
    for p in shards:
        m = _metrics(p)
        if "REPORTED" not in _status(p) or m is None or not m.get("smoke_gate", {}).get("passed"):
            problems.append(f"{smoke.name}/{p.name}: smoke gate not passed")
        else:
            passed.append(m)
    st0 = resolve_run(cfg.get("step0_analysis", STEP0_ANALYSIS))
    if "REPORTED" not in _status(st0):
        problems.append(f"{st0.name} has not reported")
    projection = project_gpu_hours(passed, jobs, int(cfg.get("n_games", C.CONFIRMATORY_GAMES))) if passed else None
    upper = cfg.get("cost_upper_gpu_hours")
    if upper is None:
        problems.append("no cost_upper_gpu_hours in the config")
    elif projection is not None and projection > COST_RULE * float(upper):
        problems.append(f"the smoke projects {projection:.1f} GPU-hours, above {COST_RULE} x {upper}")
    if cfg.get("model") in MEMSMOKE_MODELS:
        ms = resolve_run(cfg.get("memsmoke_run", f"optionc_memsmoke_{cfg['model']}"))
        if "REPORTED" not in _status(ms) or not (_metrics(ms) or {}).get("ok"):
            problems.append(f"{ms.name}: no passed memory smoke for {cfg['model']}")
    out = {"ok": not problems, "problems": problems, "projected_gpu_hours": projection, "cost_upper_gpu_hours": upper}
    if problems and cfg.get("order_override"):
        out.update(ok=True, override=str(cfg["order_override"]))
    return out


def _validate(cfg: dict) -> tuple[str, str, list[str], list[str], dict[str, list[int]]]:
    tag, readout = cfg.get("model"), cfg.get("readout", YN)
    arms = list(cfg.get("arms", [NONE, BLIND, AWARE, SWAPPED]))
    rules = list(cfg.get("rules", [ACCEPTED_ONLY]))
    if not set(arms) <= set(ARMS) or len(set(arms)) != len(arms):
        raise SystemExit(f"arms must be distinct, from {ARMS}; got {arms}")
    by_arm = seeds_of(arms, cfg.get("seeds_by_arm") or cfg.get("seeds", [1]))
    verifying = [a for a in arms if a != NONE]
    if verifying and tag not in MODEL_TAGS:
        raise SystemExit(f"unknown verifier model {tag!r}; known: {MODEL_TAGS}")
    if readout not in READOUTS or not set(rules) <= set(RULES) or not rules:
        raise SystemExit(f"readout from {READOUTS}, rules from {RULES}; got {readout}, {rules}")
    if any(not by_arm[a] for a in verifying):
        raise SystemExit(f"every verifier arm needs at least one seed; got {by_arm}")
    return tag, readout, arms, rules, by_arm


def last_only_check(scorer: Any, prompts: list[list[dict[str, str]]], readout: str, n: int = 8) -> dict[str, Any]:
    """Review 1, S6: the scorer keeps only the last position's logits if, on the first `n` gate prompts, that gives the
    same rounded score as the full forward pass; otherwise it keeps every position. Recorded either way."""
    if getattr(scorer, "mock", False) or not prompts:
        return {"checked": 0, "adopted": False}
    diffs, same = [], True
    for msgs in prompts[:n]:
        scorer.last_only = False
        full = read_score(scorer, msgs, readout)
        scorer.last_only = True
        last = read_score(scorer, msgs, readout)
        diffs.append(abs(full["L_raw"] - last["L_raw"]))
        same = same and full["L"] == last["L"]
    scorer.last_only = same
    return {"checked": len(diffs), "adopted": same, "max_abs_diff_raw": max(diffs)}


def open_verifier(tag: str) -> Any:
    from wf.decisionsim import open_scorer

    return open_scorer(tag)


def gate_prompts(client: Any, games: list[str], split: str, tag: str, arms: list[str], readout: str,
                 swaps: dict[str, str | None]) -> list[list[dict[str, str]]]:
    """Every verifier arm's prompt for each game's first state (action `look`): what the tokenization gates read."""
    out = []
    for g in games:
        st = client.reset(g, split)
        item = verifier_item(g, 0, st.task_line, history_text_from_state(st), "look", swaps.get(g))
        out += [verifier_messages(tag, item, a, readout) for a in arms if a != NONE and (a != SWAPPED or swaps.get(g))]
    return out


def run_optionc_episodes(cfg: dict) -> int:
    from wf import families
    from wf.decisionsim import FAMILY_MODELS, yes_no_gate

    tag, readout, arms, rules, by_arm = _validate(cfg)
    base = cfg["run_id"]
    require_filed(base)
    i, n = shard_spec(cfg)
    run_id = f"{base}/shard_{i:02d}_of_{n:02d}"
    order_gate = None
    if not cfg.get("smoke"):
        order_gate = order_gate_check(cfg, episode_jobs(arms, rules, by_arm))
        if not order_gate["ok"]:
            write_status(run_id, "this run waits for its gates (design note, order of runs): "
                         + "; ".join(order_gate["problems"]), ok=False)
            return 1
    obs_gate = None
    if readout == OBS and any(a != NONE for a in arms):
        obs_gate = obs_gate_check(cfg)
        if not obs_gate["ok"]:
            write_status(run_id, "the observation readout's episodes wait for the step-0 check: " + obs_gate["why"],
                         ok=False)
            return 1
    split = cfg.get("split", "valid_unseen")
    step_cap = int(cfg.get("step_cap", C.STEP_CAP))
    shadow = tuple(a for a in cfg.get("shadow_arms", SHADOW_ARMS) if a in VERIFIER_ARMS) if AWARE in arms else ()
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n, "arms": arms, "rules": rules, "seeds_by_arm": by_arm,
           "readout": readout, "sampling": SAMPLING, "n_samples": int(cfg.get("n_samples", N_SAMPLES)),
           "k_max": int(cfg.get("k_max", K_MAX)), "shadow_arms": list(shadow)}
    d = _init_run(run_id, cfg)
    resume = bool(cfg.get("resume"))
    cache_path = d / "cache.jsonl"
    if resume:
        repair_jsonl(cache_path)
    elif cache_path.exists():
        cache_path.unlink()
    client = _client(cfg)
    client.create()
    all_games = _games(client, split, int(cfg.get("n_games", C.CONFIRMATORY_GAMES)), cfg)
    games = shard_games(all_games, i, n)
    write_json(d / "shard.json", {"shard": i, "n_shards": n, "games": games, "all_games": sorted(all_games)})
    if obs_gate is not None:
        write_json(d / "obs_gate.json", obs_gate)  # which step-0 analysis was read, or the logged override
    if order_gate is not None:
        write_json(d / "order_gate.json", order_gate)  # the smoke, step-0, cost and memory gates, or the override
    saved = {episode_key(e): e for e in read_jsonl(d / "episodes.jsonl")} if resume else {}
    done = {r["game_id"] for r in read_jsonl(d / "games_done.jsonl")} if resume else set()
    cache_rows = read_jsonl(cache_path) if resume else []
    swaps: dict[str, str | None] = {}
    for g in games:
        if SWAPPED in arms or SWAPPED in shadow:
            c = GameCache(cache_path, g, cache_rows)
            swaps[g] = c.get("swap", "swap", lambda g=g: game_swap_object(client, g, split, client.reset(g, split).task_line))
    scorer = None
    if any(a != NONE for a in arms) and any(episode_key({"game_id": g, "arm": a, "rule": r, "seed": s}) not in saved
                                            for g in games for a, r, s in episode_jobs(arms, rules, by_arm) if a != NONE):
        scorer = open_verifier(tag)
        if not scorer.mock:  # tokenization gates, on every arm's first-state prompt of every game, before any score
            prompts = gate_prompts(client, games, split, tag, arms, readout, swaps)
            gate = yes_no_gate(scorer, prompts) if readout == YN else first_token_gate(scorer, prompts)
            write_json(d / "tokenization_checks.json", gate)
            if (readout == OBS or tag in FAMILY_MODELS) and not gate["ok"]:
                write_status(run_id, f"stopped before any episode: tokenization gate failed ({gate['n_problems']})", ok=False)
                raise families.TokenizationMismatch(f"{tag}/{readout}: gate failed: {gate['problems'][:3]}")
            write_json(d / "last_only_check.json", last_only_check(scorer, prompts, readout))
    models = Models(AgentClient(base_url=cfg.get("vllm_base_url")), AgentSampler(base_url=cfg.get("vllm_base_url")),
                    scorer, tag, readout)
    max_eps = int(cfg.get("max_episodes_per_job") or 0)
    new_eps = 0
    for g in games:
        jobs = [j for j in episode_jobs(arms, rules, by_arm)
                if (g, j[0], j[1], j[2]) not in saved and (j[0] != SWAPPED or swaps.get(g))]
        cache = GameCache(cache_path, g, cache_rows)
        for arm, rule, seed in jobs:
            if max_eps and new_eps == max_eps:
                write_status(run_id, f"PARTIAL — {len(done)}/{len(games)} games and {len(saved)} episodes saved; "
                                     f"resubmit with WF_RESUME=1", ok=False)
                return 1
            ep = play_episode(client, models, g, split, arm, rule, seed, cache, swaps.get(g), step_cap,
                              cfg["n_samples"], cfg["k_max"], shadow)
            ep = {**ep, "run_id": base, "shard": i, "cache_hits": dict(cache.hits), "cache_misses": dict(cache.misses)}
            append_jsonl(d / "episodes.jsonl", [ep])  # saved as it finishes
            saved[episode_key(ep)] = ep
            new_eps += 1
        if g not in done:
            append_jsonl(d / "games_done.jsonl", [{"game_id": g, "swap_object": swaps.get(g)}])
            done.add(g)
    if scorer is not None:
        scorer.close()
    mine = [e for e in saved.values() if e["game_id"] in set(games)]
    book = bookkeeping(mine)
    book["games_without_swap"] = sorted(g for g in games if SWAPPED in arms and not swaps.get(g))
    if cfg.get("cache_check"):
        book["cache_check"] = cache_check(client, mine, games[0], split, tag, readout, cache_path, step_cap,
                                          cfg["n_samples"], cfg["k_max"], shadow)
        book["smoke_gate"] = smoke_gate(book)
    write_json(d / "metrics.json", book)
    write_status(run_id, f"shard {i} of {n}: {len(mine)} episodes in {len(games)} games\n{book}", ok=False,
                 reported=SHARD_COMPLETE)
    return 0


def long_history_state(client: Any, game: str, split: str, steps: int) -> Any:
    """A state `steps` actions into a game, by a seeded walk over admissible actions (the runs' _random_walk), for the
    memory smoke: the longest verifier prompt an episode can produce."""
    from wf.runs import _random_walk

    actions, _ = _random_walk(client, game, split, n=steps)
    return replay(client, game, actions, split=split).state


def run_optionc_memsmoke(cfg: dict) -> int:
    """Review 1, S6: before any run of a verifier listed in MEMSMOKE_MODELS, score its longest prompts (a 30-step
    history, every arm, each readout) while vLLM holds its share of the GPU. ok only if every score is finite and
    nothing ran out of memory; the peak memory is recorded."""
    run_id = cfg["run_id"]
    require_filed(run_id)
    tag = cfg["model"]
    if tag not in MODEL_TAGS:
        raise SystemExit(f"unknown model {tag!r}; known: {MODEL_TAGS}")
    readouts = list(cfg.get("readouts", READOUTS))
    d = _init_run(run_id, {**cfg, "resume": False})
    client = _client(cfg)
    client.create()
    split = cfg.get("split", "valid_unseen")
    game = cfg.get("game") or sorted(_games(client, split, int(cfg.get("n_games", C.CONFIRMATORY_GAMES)), cfg))[0]
    state = long_history_state(client, game, split, int(cfg.get("steps", C.STEP_CAP)))
    swap = game_swap_object(client, game, split, state.task_line)
    hist = history_text_from_state(state)
    item = verifier_item(game, len(state.history), state.task_line, hist, "look", swap)
    prompts = [(a, r, verifier_messages(tag, item, a, r)) for r in readouts for a in (AWARE, BLIND, SWAPPED, NEUTRAL)
               if a != SWAPPED or swap]
    scorer = open_verifier(tag)
    out: dict[str, Any] = {"game": game, "history_steps": len(state.history), "n_prompts": len(prompts)}
    try:
        out["last_only_check"] = {r: last_only_check(scorer, [m for _, rr, m in prompts if rr == r], r) for r in readouts}
        scores = [read_score(scorer, m, r) for _, r, m in prompts]
        out["ok"] = all(math.isfinite(x["L_raw"]) for x in scores)
    except Exception as e:  # an out-of-memory error is the finding, not a crash
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    if not scorer.mock:
        import torch

        out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        out["prompt_tokens_max"] = max(len(scorer._encode(m)[0]) for _, _, m in prompts)
    scorer.close()
    write_json(d / "metrics.json", out)
    write_status(run_id, f"memory smoke {tag}: ok={out['ok']}\n{out}", ok=False, reported="memory smoke")
    return 0


class _NoModel:
    """Stands in for every model during the cache check: any call means the cache missed something (it is not a mock
    scorer, so no stand-in score can hide a miss)."""

    mock = False

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"cache check: a model was called ({name})")


def _comparable(ep: dict) -> dict:
    return {k: v for k, v in ep.items()
            if k not in ("seconds", "seconds_sampling", "cache_hits", "cache_misses", "run_id", "shard")}


def cache_check(client: Any, episodes: list[dict], game: str, split: str, tag: str, readout: str, cache_path: Path,
                step_cap: int, n_samples: int, k_max: int, shadow: tuple[str, ...] = ()) -> dict[str, Any]:
    """Replay every episode of one game from the cache with no model at all: the episodes must come out identical."""
    rows = read_jsonl(cache_path)
    cache = GameCache(None, game, rows)
    models = Models(_NoModel(), _NoModel(), _NoModel(), tag, readout)
    mismatched, errors = [], []
    for ep in sorted((e for e in episodes if e["game_id"] == game), key=lambda e: ARMS.index(e["arm"])):
        try:
            again = play_episode(client, models, game, split, ep["arm"], ep["rule"], ep["seed"], cache,
                                 ep.get("swap_object") or ep.get("game_swap_object"), step_cap, n_samples, k_max,
                                 shadow)
        except RuntimeError as e:
            errors.append(f"{ep['arm']}/{ep['rule']}/{ep['seed']}: {e}")
            continue
        if _comparable(again) != {k: v for k, v in _comparable(ep).items() if k in _comparable(again)}:
            mismatched.append(f"{ep['arm']}/{ep['rule']}/{ep['seed']}")
    return {"game": game, "episodes": sum(e["game_id"] == game for e in episodes), "mismatched": mismatched,
            "errors": errors, "ok": not mismatched and not errors}


def bookkeeping(eps: list[dict]) -> dict[str, Any]:
    steps = [s for e in eps for s in e["trace"]]
    return {"n_episodes": len(eps), "n_steps": len(steps),
            "format_error_share": (sum(s["proposal_format_error"] for s in steps) / len(steps)) if steps else 0.0,
            "seconds_per_step": (sum(e["seconds"] for e in eps) / len(steps)) if steps else None,
            "seconds_per_step_by_arm": {a: (sum(e["seconds"] for e in eps if e["arm"] == a)
                                            / max(1, sum(e["steps"] for e in eps if e["arm"] == a)))
                                        for a in sorted({e["arm"] for e in eps})},
            "seconds_by_arm": {a: sum(e["seconds"] for e in eps if e["arm"] == a) for a in sorted({e["arm"] for e in eps})},
            "seconds_sampling": sum(e.get("seconds_sampling", 0.0) for e in eps),
            "by_arm": {a: len([e for e in eps if e["arm"] == a]) for a in sorted({e["arm"] for e in eps})}}


def smoke_gate(book: dict[str, Any]) -> dict[str, Any]:
    """Engineering only (design note, "Order of runs and gates"); outcomes are never read."""
    checks = {"format_errors_under_5pct": book["format_error_share"] < SMOKE_FORMAT_MAX,
              "cache_reproduces": bool(book.get("cache_check", {}).get("ok")),
              "episodes_played": book["n_episodes"] > 0}
    return {"passed": all(checks.values()), "checks": checks}


# ---------------- the step-0 check ----------------


def step0_states(sources: list[str], steps: tuple[int, ...] = (0,), seed: int = SEED) -> list[dict]:
    """Stage 1a's eligible states at the given steps, with their full candidate sets, outcomes and swap object."""
    from wf.decisionsim import build_states, load_items

    return [s for s in build_states(load_items(sources), seed) if s["state_idx"] in steps]


def step0_item(state: dict, t: dict) -> dict:
    """A collected candidate as the prompt builders read it, with the state's single swap object (Stage 1a)."""
    swap = state.get("swap")
    return {**t, "swapped_goal_object": swap,
            "swapped_task_line": make_swapped_task_line(t["task_line"], t["goal_object"], swap) if swap else None}


def step0_jobs(states: list[dict], arms: tuple[str, ...], readouts: tuple[str, ...]) -> list[tuple[dict, dict, str, str]]:
    return [(s, t, a, r) for s in states for t in s["candidates"] for r in readouts for a in arms
            if a != SWAPPED or s.get("swap")]


def run_optionc_step0(cfg: dict) -> int:
    from wf import families
    from wf.decisionsim import FAMILY_MODELS, yes_no_gate
    from wf.rq2 import SOURCES
    from wf.runs import _finished
    from wf.tools.verify_items import verified

    tag = cfg["model"]
    if tag not in MODEL_TAGS:
        raise SystemExit(f"unknown model {tag!r}; known: {MODEL_TAGS}")
    arms = tuple(cfg.get("arms", (AWARE, BLIND, SWAPPED, NEUTRAL)))
    readouts = tuple(cfg.get("readouts", READOUTS))
    if not set(arms) <= set(VERIFIER_ARMS) or not set(readouts) <= set(READOUTS):
        raise SystemExit(f"arms from {VERIFIER_ARMS}, readouts from {READOUTS}")
    run_id = cfg["run_id"]
    require_filed(run_id)
    sources = list(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    unverified = [s for s in sources if not verified(s)[0]]
    if problems or unverified:
        write_status(run_id, f"collection not finished or not verified: {problems + unverified}", ok=False)
        return 1
    states = step0_states(sources, tuple(int(x) for x in cfg.get("steps", [0])))
    d = _init_run(run_id, {**cfg, "arms": list(arms), "readouts": list(readouts)})
    write_json(d / "states.json", {"n_states": len(states), "states": [[s["game_id"], s["state_idx"]] for s in states]})
    jobs = step0_jobs(states, arms, readouts)
    prompts = {(t["id"], a, r): verifier_messages(tag, step0_item(s, t), a, r) for s, t, a, r in jobs}
    hashes = {k: _sha([tag, k[2], m]) for k, m in prompts.items()}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get((x["transition_id"], x["arm"], x["readout"])) != x.get("score_hash") for x in saved):
        raise SystemExit(f"{run_id}: saved scores belong to other prompts; rerun without --resume")
    done = {(x["transition_id"], x["arm"], x["readout"]) for x in saved}
    todo = [j for j in jobs if (j[1]["id"], j[2], j[3]) not in done]
    if todo:
        scorer = open_verifier(tag)
        if not scorer.mock:
            checks = {}
            if YN in readouts:
                checks[YN] = yes_no_gate(scorer, [m for k, m in prompts.items() if k[2] == YN])
            if OBS in readouts:
                checks[OBS] = first_token_gate(scorer, [m for k, m in prompts.items() if k[2] == OBS])
            write_json(d / "tokenization_checks.json", checks)
            bad = [r for r, c in checks.items() if not c["ok"] and (r == OBS or tag in FAMILY_MODELS)]
            if bad:
                write_status(run_id, f"stopped before scoring: tokenization gate failed for {bad}", ok=False)
                raise families.TokenizationMismatch(f"{tag}: gate failed for {bad}")
        for s, t, a, r in todo:
            key = (t["id"], a, r)
            sc = read_score(scorer, prompts[key], r, t["id"], a)
            append_jsonl(d / "scoring.jsonl", [{"transition_id": t["id"], "game_id": s["game_id"], "state_idx": s["state_idx"],
                                                "action": t["action"], "arm": a, "readout": r, **sc,
                                                "score_hash": hashes[key]}])
        scorer.close()
    write_status(run_id, f"{len(jobs)} scores at {len(states)} states ({tag})", ok=False, reported=SHARD_COMPLETE)
    return 0


def _ok(t: dict) -> bool:
    return (t.get("true_obs") or "").strip() != NOTHING


def _share(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def step0_cell(states: list[dict], L: dict[tuple[str, str, str], dict], arm: str, readout: str) -> dict[str, Any]:
    """The step-0 measures of one model under one arm and readout."""
    def get(t: dict) -> dict | None:
        return L.get((t["id"], arm, readout))

    props = [(s, next(t for t in s["candidates"] if t["action"] == s["proposal"])) for s in states]
    props = [(s, t) for s, t in props if _ok(t) and get(t)]
    gotos = [(s, t) for s in states for t in s["candidates"] if _ok(t) and t["action"].startswith("go to ") and get(t)]
    work = [get(t)["L"] for s in states for t in s["candidates"] if _ok(t) and get(t)]
    fail = [get(t)["L"] for s in states for t in s["candidates"] if not _ok(t) and get(t)]
    scored = [s for s in states if any(get(t) for t in s["candidates"])]
    return {
        "working_proposals_accepted": _boot([float(accepts(get(t)["L"])) for _, t in props], [s["game_id"] for s, _ in props]),
        "working_goto_accepted": _boot([float(accepts(get(t)["L"])) for _, t in gotos], [s["game_id"] for s, _ in gotos]),
        "median_L_working_proposal": statistics.median([get(t)["L"] for _, t in props]) if props else float("nan"),
        "median_L_working_goto": statistics.median([get(t)["L"] for _, t in gotos]) if gotos else float("nan"),
        "states_with_no_accepted_candidate": _share([not any(get(t) and accepts(get(t)["L"]) for t in s["candidates"])
                                                     for s in scored]),
        "auc": auc(work, fail),
        "yes_failures": _share([accepts(x) for x in fail]),
        "yes_successes": _share([accepts(x) for x in work]),
        "n_working": len(work), "n_failing": len(fail), "n_states": len(scored),
    }


def _mean_shift(pairs: list[tuple[dict, dict]], L: dict, a: str, b: str, readout: str) -> dict[str, Any]:
    vals, games = [], []
    for s, t in pairs:
        x, y = L.get((t["id"], a, readout)), L.get((t["id"], b, readout))
        if x and y:
            vals.append(x["L_raw"] - y["L_raw"])
            games.append(s["game_id"])
    return _boot(vals, games)


def proposal_gap(states: list[dict], L: dict, readout: str) -> dict[str, Any]:
    """Aware minus blind acceptance of each state's working proposal, paired by state, with its game interval."""
    a, b, games = [], [], []
    for s in states:
        t = next(c for c in s["candidates"] if c["action"] == s["proposal"])
        x, y = L.get((t["id"], AWARE, readout)), L.get((t["id"], BLIND, readout))
        if _ok(t) and x and y:
            a.append(float(accepts(x["L"])))
            b.append(float(accepts(y["L"])))
            games.append(s["game_id"])
    return _paired(a, b, games)


def gap_state(gap: dict[str, Any]) -> str:
    """'collapse' if the interval's upper end is below -COLLAPSE_GAP, 'no gap' if the whole interval lies within
    +-NO_GAP, otherwise 'unclear' (an empty or undefined interval is unclear)."""
    lo, hi = gap["ci"]
    if hi < -COLLAPSE_GAP:
        return "collapse"
    if -NO_GAP <= lo and hi <= NO_GAP:
        return "no gap"
    return "unclear"


def classify(gaps: dict[str, dict], competent: dict[str, bool]) -> str:
    """The pre-registered classification of the step-0 collapse (design note, amended after review 1, S3): the
    aware-minus-blind gap of each readout, read from its interval, with the blind arm of each readout competent."""
    if YN not in gaps or OBS not in gaps:
        return "not classified (a readout is missing)"
    if not competent.get(YN):
        return "not classifiable (the task-blind yes/no verifier is not competent)"
    yn = gap_state(gaps[YN])
    if yn == "no gap":
        return "no collapse to explain"
    if yn != "collapse":
        return "mixed (the yes/no interval does not settle it)"
    if not competent.get(OBS):
        return "not classifiable (the observation readout is not competent)"
    obs = gap_state(gaps[OBS])
    if obs == "no gap":
        return "question-specific (supports the task-success reading)"
    if obs == "collapse":
        return "not question-specific"
    return "mixed (the observation interval does not settle it)"


def competence(cell: dict[str, Any]) -> dict[str, Any]:
    ok = {"auc": cell["auc"] >= COMPETENCE["auc_min"],
          "yes_failures": cell["yes_failures"] <= COMPETENCE["yes_failures_max"],
          "yes_successes": cell["yes_successes"] >= COMPETENCE["yes_successes_min"]}
    return {"passed": all(ok.values()), "checks": ok, "auc": cell["auc"], "yes_failures": cell["yes_failures"],
            "yes_successes": cell["yes_successes"]}


def step0_report(states: list[dict], records: list[dict], stage1a: list[dict] | None = None) -> dict[str, Any]:
    L = {(r["transition_id"], r["arm"], r["readout"]): r for r in records}
    arms = sorted({r["arm"] for r in records}, key=VERIFIER_ARMS.index)
    readouts = sorted({r["readout"] for r in records}, key=READOUTS.index)
    by = {r: {a: step0_cell(states, L, a, r) for a in arms} for r in readouts}
    fail_goal = [(s, t) for s in states for t in s["candidates"] if not _ok(t) and t.get("targets_goal_object")]
    goto = [(s, t) for s in states for t in s["candidates"] if _ok(t) and t["action"].startswith("go to ")]
    shifts = {r: {"naming_lift_aware_minus_swapped": _mean_shift(fail_goal, L, AWARE, SWAPPED, r),
                  "anytask_drop_aware_minus_blind": _mean_shift(goto, L, AWARE, BLIND, r),
                  "anytask_drop_swapped_minus_blind": _mean_shift(goto, L, SWAPPED, BLIND, r)}
              for r in readouts if {AWARE, BLIND, SWAPPED} <= set(arms)}
    gaps = {r: proposal_gap(states, L, r) for r in readouts if {AWARE, BLIND} <= set(arms)}
    comp = {r: competence(by[r][BLIND]) for r in readouts if BLIND in by[r]}
    out: dict[str, Any] = {"cells": by, "shifts": shifts, "proposal_gaps": gaps,
                           "gap_states": {r: gap_state(g) for r, g in gaps.items()}, "blind_competence": comp,
                           "classification": classify(gaps, {r: c["passed"] for r, c in comp.items()})}
    if OBS in comp:
        out["obs_competence"] = comp[OBS]
    if stage1a is not None:
        out["stage1a_agreement"] = stage1a_agreement(records, stage1a)
    return out


def stage1a_agreement(records: list[dict], stage1a: list[dict]) -> dict[str, Any]:
    """The yes/no aware, swapped and neutral prompts are Stage 1a's: its saved scores, rounded, against these."""
    arm_of = {AWARE: "yn_aligned", SWAPPED: "yn_swapped", NEUTRAL: "yn_absent"}
    old = {(r["transition_id"], r["condition"]): quantize(r["L"]) for r in stage1a}
    diffs = [abs(r["L"] - old[(r["transition_id"], arm_of[r["arm"]])]) for r in records
             if r["readout"] == YN and r["arm"] in arm_of and (r["transition_id"], arm_of[r["arm"]]) in old]
    return {"n_compared": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
            "n_differing": sum(x > 0 for x in diffs)}


def run_optionc_step0_analysis(cfg: dict) -> int:
    from wf.rq2 import SOURCES

    run_id = cfg["run_id"]
    require_filed(run_id)
    cfg = {**cfg, "resume": False}
    runs = dict(cfg.get("models", {}))
    missing = []
    for m, r in runs.items():
        st = resolve_run(r) / "STATUS.md"
        first = st.read_text(encoding="utf-8").splitlines()[0] if st.exists() else ""
        if "REPORTED" not in first:
            missing.append(f"{m}: {r} ({first or 'no status'})")
    d = _init_run(run_id, cfg)
    if missing:
        write_status(run_id, "the step-0 analysis waits for every scoring run:\n" + "\n".join(missing), ok=False)
        return 1
    states = step0_states(list(cfg.get("sources", SOURCES)), tuple(int(x) for x in cfg.get("steps", [0])))
    out: dict[str, Any] = {"n_states": len(states), "models": {}}
    for tag, r in runs.items():
        recs = read_jsonl(resolve_run(r) / "scoring.jsonl")
        s1a = cfg.get("stage1a_scores", {}).get(tag)
        s1a_rows = None
        if s1a:
            files = sorted(resolve_run(s1a).glob("shard_*/scoring.jsonl")) or sorted(resolve_run(s1a).glob("scoring.jsonl"))
            s1a_rows = [x for f in files for x in read_jsonl(f)] if files else None
        out["models"][tag] = step0_report(states, recs, s1a_rows)
    write_json(d / "metrics.json", out)
    write_run_report(run_id, "Option C step-0 check", {t: {"classification": m["classification"],
                                                           "obs_competence": m.get("obs_competence")}
                                                       for t, m in out["models"].items()})
    write_status(run_id, "\n".join(f"{t}: {m['classification']}; observation readout competent: "
                                   f"{m.get('obs_competence', {}).get('passed')}" for t, m in out["models"].items()),
                 ok=False, reported="step-0 check")
    return 0


# ---------------- analysis ----------------


OUTCOMES = ("success", "n_invalid", "score_S", "steps", "n_override", "n_override_harm", "n_override_help",
            "override_net_harm", "n_invalid_first10", "n_abstained", "n_repeats", "step0_override", "step0_rejected",
            "longest_rejection_run")
# Outcomes given a verdict, and whether more is better. Success is primary; override harm minus help and the first 10
# steps' invalid actions are the key secondaries (review 1, S1); invalid actions per episode is secondary, since it
# largely restates success (a lost episode executes about 8 more failing actions than a won one).
VERDICT_OUTCOMES = {"success": True, "override_net_harm": False, "n_invalid_first10": False, "n_invalid": False}


def per_game(eps: list[dict], arm: str, rule: str | None, outcome: str) -> dict[str, float]:
    """Each game's mean over its seeds (the no-verifier arm has one seed-free episode per game)."""
    vals: dict[str, list[float]] = defaultdict(list)
    for e in eps:
        if e["arm"] == arm and (arm == NONE or e["rule"] == rule):
            vals[e["game_id"]].append(float(e[outcome]))
    return {g: sum(v) / len(v) for g, v in vals.items()}


def paired(eps: list[dict], a: str, b: str, rule: str | None, outcome: str,
           games: set[str] | None = None) -> dict[str, Any]:
    """Arm a minus arm b, paired by game: game bootstrap interval and whole-game sign-flip p."""
    x, y = per_game(eps, a, rule, outcome), per_game(eps, b, rule, outcome)
    common = sorted(set(x) & set(y) & (games if games is not None else set(x)))
    out = _paired([x[g] for g in common], [y[g] for g in common], common)
    out["n_games"] = len(common)
    return out


def decomposition(eps: list[dict], rule: str, outcome: str, games: set[str]) -> dict[str, Any]:
    """aware - blind = (aware - swapped) + (swapped - blind), on the games with a swap object; the identity checked."""
    ab = paired(eps, AWARE, BLIND, rule, outcome, games)
    asw = paired(eps, AWARE, SWAPPED, rule, outcome, games)
    sb = paired(eps, SWAPPED, BLIND, rule, outcome, games)
    ok = abs(ab["estimate"] - (asw["estimate"] + sb["estimate"])) < 1e-9 if ab["n_games"] else True
    return {"aware_minus_blind": ab, "aware_minus_swapped": asw, "swapped_minus_blind": sb, "identity_holds": ok,
            "n_games": ab["n_games"]}


NOT_IDENTIFIABLE = "attribution not identifiable in the loop (the swapped verifier obstructs the real task)"


def attribution(dec: dict[str, Any] | None, higher_is_better: bool, obstruction: dict[str, Any] | None) -> str:
    """Attribution of a harm to the real task (amended after review 1, M1 and S2), on the games with a swap object:
    not identifiable when the obstruction check fails or cannot be computed; unresolved when the aware verifier's cost
    is not positive there, or when the aware-minus-swapped interval contains the threshold (half of the cost);
    otherwise attributable when the swapped verifier recovers at least half of the cost."""
    if dec is None or not dec.get("n_games"):
        return "attribution not tested (no swapped arm)"
    if obstruction is None or obstruction.get("state") != "clear":
        return NOT_IDENTIFIABLE
    sign = -1.0 if higher_is_better else 1.0
    cost = sign * dec["aware_minus_blind"]["estimate"]
    if not cost > TOL:
        return "attribution unresolved"
    own = sign * dec["aware_minus_swapped"]["estimate"]
    lo, hi = sorted(sign * x for x in dec["aware_minus_swapped"]["ci"])
    threshold = SWAP_SHARE * cost
    if lo != lo or hi != hi or lo - TOL <= threshold <= hi + TOL:
        return "attribution unresolved"
    return "attributable to the real task" if own >= threshold - TOL else "not attributable to the real task"


def verdict(diff: dict[str, Any], dec: dict[str, Any] | None, higher_is_better: bool = True,
            obstruction: dict[str, Any] | None = None) -> str:
    lo, hi = diff["ci"]
    p = diff.get("p", float("nan"))
    significant = p == p and p < ALPHA
    harm = significant and ((hi < 0) if higher_is_better else (lo > 0))
    help_ = significant and ((lo > 0) if higher_is_better else (hi < 0))
    if harm:
        return "harm, " + attribution(dec, higher_is_better, obstruction)
    if help_:
        return "help"
    return "no reliable difference"


def _steps(eps: list[dict], arm: str, rule: str | None) -> list[tuple[str, dict]]:
    return [(e["game_id"], s) for e in eps if e["arm"] == arm and (arm == NONE or e["rule"] == rule) for s in e["trace"]]


def goal_acceptance(eps: list[dict], arm: str, rule: str, failing: bool) -> dict[str, Any]:
    """The share of the arm's verified proposals on the real goal object, working (failing=False) or failing, that its
    verifier accepted, with a game interval."""
    rows = [(g, s) for g, s in _steps(eps, arm, rule) if "proposal_accepted" in s and s["proposal_targets_goal"]
            and bool(s["proposal_failed"]) == failing]
    return _boot([float(s["proposal_accepted"]) for _, s in rows], [g for g, _ in rows])


def obstruction_check(eps: list[dict], rule: str) -> dict[str, Any]:
    """Must-fix M1: per verifier arm, acceptance of working and of failing proposals on the real goal object (the latter
    the in-loop naming lift). The swapped verifier obstructs the real task when it accepts working goal-object proposals
    more than OBSTRUCTION_MARGIN less often than the aware one; then success-level attribution is not identifiable."""
    arms = [a for a in VERIFIER_ARMS if any(e["arm"] == a and e["rule"] == rule for e in eps)]
    acc = {a: {"working_goal": goal_acceptance(eps, a, rule, False), "failing_goal": goal_acceptance(eps, a, rule, True)}
           for a in arms}
    out: dict[str, Any] = {"acceptance": acc, "margin": OBSTRUCTION_MARGIN}
    if not {AWARE, SWAPPED} <= set(arms):
        return {**out, "state": "not tested"}
    aw, sw = acc[AWARE]["working_goal"], acc[SWAPPED]["working_goal"]
    if not aw["n_states"] or not sw["n_states"]:
        return {**out, "state": "not computable"}
    gap = aw["estimate"] - sw["estimate"]
    return {**out, "working_goal_gap_aware_minus_swapped": gap,
            "state": "obstructs" if gap > OBSTRUCTION_MARGIN + TOL else "clear"}


def matched_states(eps: list[dict], rule: str) -> dict[str, Any]:
    """Must-fix M1 (c), descriptive: at every step of the aware episodes, the proposal (and the alternatives the aware
    verifier scored) also scored under the blind and swapped prompts, never acted on. On these identical states: how
    often each prompt would accept the failing goal-object proposals, reject the working proposals and reject the
    working goal-object proposals, and accept goal-object alternatives (outcome unknown); paired by step, game
    intervals."""
    rows = [(e["game_id"], s, parse_task_line(e["task_line"]).goal_object) for e in eps
            if e["arm"] == AWARE and e["rule"] == rule for s in e["trace"] if "shadow" in s]
    kinds = {"accept_failing_goal_proposal": (lambda s: s["proposal_failed"] and s["proposal_targets_goal"], True),
             "reject_working_proposal": (lambda s: not s["proposal_failed"], False),
             "reject_working_goal_proposal": (lambda s: not s["proposal_failed"] and s["proposal_targets_goal"], False)}
    out: dict[str, Any] = {"n_steps": len(rows)}
    for other in SHADOW_ARMS:
        block: dict[str, Any] = {}
        for name, (keep, want_accept) in kinds.items():
            sel = [(g, s) for g, s, _ in rows if other in s["shadow"] and keep(s)]
            own = [float(accepts(s["proposal_L"]) == want_accept) for _, s in sel]
            oth = [float(accepts(s["shadow"][other]["proposal_L"]) == want_accept) for _, s in sel]
            block[name] = {"aware": _boot(own, [g for g, _ in sel]), other: _boot(oth, [g for g, _ in sel]),
                           f"aware_minus_{other}": _paired(own, oth, [g for g, _ in sel])}
        alts = [(g, a, s["alt_L"][a], s["shadow"][other]["alt_L"][a]) for g, s, goal in rows if other in s["shadow"]
                for a in s["alt_L"] if a in s["shadow"][other]["alt_L"] and names_object(a, goal)]
        block["accept_goal_alternative"] = _paired([float(accepts(x)) for _, _, x, _ in alts],
                                                   [float(accepts(y)) for _, _, _, y in alts], [g for g, _, _, _ in alts])
        out[other] = block
    return out


def divergence(eps: list[dict], rule: str, a: str = AWARE, b: str = BLIND) -> dict[str, Any]:
    """The first step at which a's and b's episodes of one game and seed execute different actions."""
    by = {(e["game_id"], e["seed"], e["arm"]): e for e in eps if e["rule"] == rule and e["arm"] in (a, b)}
    firsts, never = [], 0
    for (g, s, arm), e in sorted(by.items()):
        if arm != a or (g, s, b) not in by:
            continue
        other = by[(g, s, b)]
        k = next((i for i, (x, y) in enumerate(zip(e["trace"], other["trace"])) if x["chosen"] != y["chosen"]), None)
        if k is None and len(e["trace"]) == len(other["trace"]):
            never += 1
        else:
            firsts.append(k if k is not None else min(len(e["trace"]), len(other["trace"])))
    n = len(firsts) + never
    return {"n_pairs": n, "never_diverge": never, "share_never": never / n if n else float("nan"),
            "median_first_divergence": statistics.median(firsts) if firsts else None}


def step_rates(eps: list[dict], arm: str, rule: str | None) -> dict[str, Any]:
    """Per-step acceptance of the proposal by its true outcome (known at every step: executed, or replayed at
    overrides), and override reasons."""
    steps = [s for e in eps if e["arm"] == arm and (arm == NONE or e["rule"] == rule) for s in e["trace"]]
    ver = [s for s in steps if "proposal_accepted" in s]
    return {"n_steps": len(steps),
            "accept_working_proposal": _share([s["proposal_accepted"] for s in ver if not s["proposal_failed"]]),
            "accept_failing_proposal": _share([s["proposal_accepted"] for s in ver if s["proposal_failed"]]),
            "accept_failing_goal_proposal": _share([s["proposal_accepted"] for s in ver
                                                    if s["proposal_failed"] and s["proposal_targets_goal"]]),
            "reasons": dict(Counter(s["reason"] for s in steps)),
            "chosen_fails": _share([s["failed"] for s in steps]),
            "proposal_fails": _share([s["proposal_failed"] for s in steps])}


def arm_summary(eps: list[dict], arm: str, rule: str | None, outcomes: tuple[str, ...] = OUTCOMES) -> dict[str, Any]:
    out = {}
    for o in outcomes:
        pg = per_game(eps, arm, rule, o)
        out[o] = _boot([pg[g] for g in sorted(pg)], sorted(pg))
    won = [e["steps"] for e in eps if e["arm"] == arm and (arm == NONE or e["rule"] == rule) and e["success"]]
    out["steps_when_won_mean"] = sum(won) / len(won) if won else float("nan")
    out["step_rates"] = step_rates(eps, arm, rule)
    return out


def run_report(eps: list[dict], none_eps: list[dict] | None = None,
               outcomes: tuple[str, ...] = OUTCOMES) -> dict[str, Any]:
    """Every rule of one run: arm summaries, contrasts, decomposition, verdicts (success; invalid actions reversed)."""
    allv = eps + [e for e in (none_eps or []) if e["arm"] == NONE and not any(x["arm"] == NONE for x in eps)]
    arms = [a for a in ARMS if any(e["arm"] == a for e in allv)]
    swap_games = {e["game_id"] for e in eps if e["arm"] == SWAPPED}
    out: dict[str, Any] = {"arms": arms, "rules": {}}
    for rule in sorted({e["rule"] for e in eps if e["rule"]}):
        r: dict[str, Any] = {"arms": {a: arm_summary(allv, a, None if a == NONE else rule, outcomes) for a in arms}}
        contrasts = {}
        pairs = [(AWARE, BLIND), (AWARE, SWAPPED), (SWAPPED, BLIND), (BLIND, NONE), (AWARE, NONE), (SWAPPED, NONE),
                 (NEUTRAL, BLIND), (AWARE, NEUTRAL)]
        for a, b in pairs:
            if a in arms and b in arms:
                contrasts[f"{a}_minus_{b}"] = {o: paired(allv, a, b, rule, o) for o in outcomes}
        r["contrasts"] = contrasts
        if {AWARE, BLIND} <= set(arms):
            vo = [o for o in VERDICT_OUTCOMES if o in outcomes]
            dec = {o: decomposition(allv, rule, o, swap_games) for o in vo} if SWAPPED in arms else {}
            r["decomposition"] = dec
            r["obstruction"] = obstruction_check(eps, rule)
            ab = contrasts["aware_minus_blind"]
            r["verdicts"] = {o: verdict(ab[o], dec.get(o), VERDICT_OUTCOMES[o], r["obstruction"]) for o in vo}
            r["verdict_success"] = r["verdicts"].get("success")
            r["verdict_invalid"] = r["verdicts"].get("n_invalid")
            r["divergence"] = divergence(eps, rule)
            r["matched_states"] = matched_states(eps, rule)
        out["rules"][rule] = r
    return out


def drift(none_eps: list[dict], archived: list[dict]) -> dict[str, Any]:
    """The no-verifier episodes against RQ3's archived no-foresight ones (amended after review 1, S7): agreement of the
    first action, the step of first divergence, and the paired success difference with its game interval. The share
    reproduced action for action is a reproduction rate, not a test."""
    old = {e["game_id"]: e for e in archived if e.get("condition") == "none"}
    pairs = [(e, old[e["game_id"]]) for e in none_eps if e["game_id"] in old]
    firsts = []
    for e, o in pairs:
        a, b = [s["chosen"] for s in e["trace"]], [s["executed"] for s in o["trace"]]
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        firsts.append(None if k is None and len(a) == len(b) else (k if k is not None else min(len(a), len(b))))
    div = sorted(k for k in firsts if k is not None)
    q = (lambda f: div[min(len(div) - 1, int(f * len(div)))]) if div else (lambda f: None)
    games = [e["game_id"] for e, _ in pairs]
    return {"n_games": len(pairs),
            "first_action_agreement": _share([e["trace"][0]["chosen"] == o["trace"][0]["executed"]
                                              for e, o in pairs if e["trace"] and o["trace"]]),
            "reproduction_rate": _share([k is None for k in firsts]),
            "first_divergence_quartiles": [q(0.25), q(0.5), q(0.75)], "n_diverged": len(div),
            "success_now": _share([e["success"] for e, _ in pairs]),
            "success_archived": _share([o["success"] for _, o in pairs]),
            "success_now_minus_archived": _paired([float(e["success"]) for e, _ in pairs],
                                                  [float(o["success"]) for _, o in pairs], games)}


def primary_problems(run_cfg: dict, step0: dict | None) -> list[str]:
    """The primary run must be Qwen3-8B, accepted-only, with aware and blind arms, on the yes/no readout; or on the
    observation readout only under the pre-declared reallocation (the step-0 check 'question-specific' and the
    observation readout competent for Qwen3-8B)."""
    out = []
    if run_cfg.get("model") != PRIMARY["model"] or PRIMARY["rule"] not in run_cfg.get("rules", []):
        out.append(f"the primary run is not {PRIMARY['model']} with the {PRIMARY['rule']} rule")
    if not {AWARE, BLIND} <= set(run_cfg.get("arms", [])):
        out.append("the primary run lacks the aware or the blind arm")
    if run_cfg.get("readout") == OBS:
        m = (step0 or {}).get("models", {}).get(PRIMARY["model"], {})
        if not (m.get("classification", "").startswith("question-specific") and m.get("obs_competence", {}).get("passed")):
            out.append("the observation readout is primary only after a 'question-specific' step-0 check with a "
                       "competent observation readout")
    elif run_cfg.get("readout") != YN:
        out.append(f"unknown primary readout {run_cfg.get('readout')!r}")
    return out


def load_run(name: str) -> tuple[list[dict], list[str]]:
    folders, problems = complete_shards(name)
    if problems:
        return [], problems
    eps = [e for p in folders for e in read_jsonl(p / "episodes.jsonl")]
    cfg = json.loads((folders[0] / "config.json").read_text(encoding="utf-8"))
    games = [g for p in folders for g in json.loads((p / "shard.json").read_text(encoding="utf-8"))["games"]]
    swapless = {r["game_id"] for p in folders for r in read_jsonl(p / "games_done.jsonl") if not r.get("swap_object")}
    expected = {(g, a, r, s) for g in games for a, r, s in episode_jobs(cfg["arms"], cfg["rules"], cfg["seeds_by_arm"])
                if not (a == SWAPPED and g in swapless)}
    keys = [episode_key(e) for e in eps]
    if set(keys) != expected or len(keys) != len(set(keys)):
        return [], [f"{name}: episodes do not match the design: {len(set(keys))} distinct of {len(expected)} expected, "
                    f"{len(keys) - len(set(keys))} repeated"]
    return eps, []


def run_optionc_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    cfg = {**cfg, "resume": False}
    runs = dict(cfg["runs"])
    loaded, problems = {}, []
    for name, src in runs.items():
        eps, pr = load_run(src)
        loaded[name], problems = eps, problems + pr
    d = _init_run(run_id, cfg)
    if problems:
        write_status(run_id, "Option C results wait for every shard of every run:\n" + "\n".join(problems), ok=False)
        return 1
    primary = cfg.get("primary_run")
    if primary:
        folders, _ = complete_shards(runs[primary])
        run_cfg = json.loads((folders[0] / "config.json").read_text(encoding="utf-8"))
        bad = primary_problems(run_cfg, _metrics(resolve_run(cfg.get("step0_analysis", STEP0_ANALYSIS))))
        if bad:
            write_status(run_id, "the primary run does not qualify:\n" + "\n".join(bad), ok=False)
            return 1
    none_eps = [e for e in loaded.get(primary, []) if e["arm"] == NONE] if primary else []
    out: dict[str, Any] = {"runs": {}}
    for name, eps in loaded.items():
        meta = sorted({(e["model"], e["readout"]) for e in eps if e["arm"] != NONE})
        out["runs"][name] = {"source": runs[name], "verifier": meta, **run_report(eps, none_eps)}
    if primary:
        prim = out["runs"][primary]["rules"].get(PRIMARY["rule"], {})
        out["primary"] = {"run": primary, "verdict_success": prim.get("verdict_success"),
                          "aware_minus_blind_success": prim.get("contrasts", {}).get("aware_minus_blind", {}).get("success"),
                          "key_secondaries": {o: prim.get("verdicts", {}).get(o)
                                              for o in ("override_net_harm", "n_invalid_first10")},
                          "verdict_invalid_secondary": prim.get("verdict_invalid"),
                          "obstruction": prim.get("obstruction", {}).get("state")}
    if cfg.get("rq3_source") and none_eps:
        from wf.decisionsim_agent import episode_files

        try:  # a missing archive is recorded, never fatal: every other number is still written (review 1, S7)
            archived = [e for f in episode_files(cfg["rq3_source"]) for e in read_jsonl(f)]
            out["drift"] = drift(none_eps, archived)
        except (SystemExit, OSError) as e:
            out["drift"] = {"missing": f"{cfg['rq3_source']}: {e}"}
    write_json(d / "metrics.json", out)
    write_run_report(run_id, "Option C", {k: v for k, v in out.items() if k != "runs"})
    lines = [f"{n}: " + "; ".join(f"{rule}: success {r.get('verdict_success')}, invalid {r.get('verdict_invalid')}"
                                  for rule, r in v["rules"].items()) for n, v in out["runs"].items()]
    write_status(run_id, "\n".join(lines), ok=False, reported="Option C")
    return 0
