"""Confirmatory RQ3 runner: capped false successes, yoked false failures, sharded and resumable.

Decided 2026-11-17 (notes/deviations.md, "RQ3 error matching: cap and yoke"). A false success
(FS) shows the environment's own success text for a proposal that will fail; a false failure (FF)
shows "Nothing happens." for one that would work. Each (game, schedule, retention) runs its FS
episode first; the FF episode then gets as many errors as the FS episode received.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf.agent.foresight import ForesightConfig, foresight_prediction
from wf.agent.react import FORMAT_ERROR_ACTION, AgentClient, _followed, run_episode
from wf.analysis.report import write_run_report
from wf.analysis.rq3_stats import rq3_summary
from wf.data.candidates import NOOP, attach_candidates
from wf.data.goals import names_object, parse_task_line
from wf.data.transitions import action_type
from wf.env.expert import on_expert_path
from wf.env.server import AlfworldClient, GameState
from wf.runs import _client, _games, _init_run
from wf.scoring.prompts import agent_react_messages, parse_react_action
from wf.util import append_jsonl, file_sha, read_jsonl, require_filed, runs_root, write_json, write_status

BASELINES = (("none", "discarded"), ("oracle", "discarded"), ("oracle", "retained"), ("validity_only", "discarded"))
RETENTIONS = ("discarded", "retained")
SHARD_COMPLETE = "shard complete"


# ---------------- schedules ----------------


@dataclass
class FalseSuccessPlan:
    """At each chance (a failing proposal with a success twin) inject with probability p, until `cap`.

    Draws are indexed by chance, not by step, so both retention modes of one (game, schedule)
    read the same sequence even after their paths part.
    """

    game_id: str
    seed: int
    p: float
    cap: int
    chances: int = 0
    injections: list[dict] = field(default_factory=list)

    def draw(self, k: int) -> bool:
        h = hashlib.sha256(f"rq3-fs|{self.game_id}|{self.seed}|{k}".encode()).hexdigest()
        return random.Random(int(h[:16], 16)).random() < self.p

    def false_text(self, t: int, proposal: str, valid: bool, format_error: bool,
                   twin: Callable[[], str | None]) -> str | None:
        if valid or len(self.injections) >= self.cap:
            return None
        o_plus = twin()
        if o_plus is None:  # nothing false can be shown, so this is not a chance
            return None
        k = self.chances
        self.chances += 1
        if not self.draw(k):
            return None
        self.injections.append({"i": len(self.injections), "step": t, "verb": action_type(proposal),
                                "proposal": proposal, "chance": k})
        return o_plus


@dataclass
class FalseFailurePlan:
    """The i-th error goes at the first working proposal with the i-th false success's verb, at or
    after its step. Indices in `any_verb` take the first working proposal of any verb instead; an
    index is put there only after a full pass showed no same-verb proposal (`run_false_failure`).
    """

    targets: list[dict]
    any_verb: frozenset[int] = frozenset()
    injections: list[dict] = field(default_factory=list)

    def start(self, i: int) -> int:
        """First step where error i may go: its partner's step, and after error i-1."""
        after = self.injections[i - 1]["step"] + 1 if i else 0
        return max(self.targets[i]["step"], after)

    def false_text(self, t: int, proposal: str, valid: bool, format_error: bool,
                   twin: Callable[[], str | None] | None = None) -> str | None:
        i = len(self.injections)
        # A format error's `look` was never proposed by the agent, so it is not a working proposal.
        if i >= len(self.targets) or t < self.targets[i]["step"] or not valid or format_error:
            return None
        verb, target = action_type(proposal), self.targets[i]
        if verb != target["verb"] and i not in self.any_verb:
            return None
        self.injections.append({"i": i, "step": t, "verb": verb, "proposal": proposal, "fs_step": target["step"],
                                "fs_verb": target["verb"], "fallback": verb != target["verb"]})
        return NOOP


# ---------------- episodes ----------------


def _twin(env: AlfworldClient, state: GameState, proposal: str, oracle_obs: str, cache: dict) -> str | None:
    item = {"game_id": state.game_id, "split": state.split, "action": proposal, "true_obs": oracle_obs,
            "history_pairs": state.history}
    attach_candidates(env.sandbox(), item, success_cache=cache)
    return item["candidates"]["o_plus"] if item.get("candidates") else None


def play_episode(
    env: AlfworldClient,
    agent: AgentClient,
    game_id: str,
    split: str,
    retention: str,
    plan: FalseSuccessPlan | FalseFailurePlan | None = None,
    step_cap: int = C.STEP_CAP,
    success_cache: dict | None = None,
    replayed: list[dict] | None = None,
) -> dict[str, Any]:
    """The ReAct loop of `run_episode` under perfect foresight, with `plan` choosing the false steps.

    `replayed` steps are re-executed from their saved records with no agent call, so a yoked
    pass can restart part-way through an earlier pass on exactly the same path.
    """
    if env._env_id is None:
        env.create()
    state = env.reset(game_id, split)
    goal_object = parse_task_line(state.task_line).goal_object
    fc = ForesightConfig("oracle", retention=retention)
    cache = success_cache if success_cache is not None else {}
    prefix: list[str] = []
    history: list[dict[str, str]] = []
    trace: list[dict] = []
    arm = "favorable" if isinstance(plan, FalseSuccessPlan) else "unfavorable"

    for t in range(step_cap):
        if state.done:
            break
        if replayed is not None and t < len(replayed):
            step_rec = dict(replayed[t])
            if plan is not None:  # bring the plan to the same point, and check it would choose the same
                hit = plan.false_text(t, step_rec["proposal"], step_rec["proposal_valid"],
                                      step_rec["proposal_format_error"], lambda: None) is not None
                if hit != step_rec["injected"]:
                    raise RuntimeError(f"{game_id}: replayed step {t} would be injected differently")
            new_state = env.step(state, step_rec["executed"])
            if new_state.observation != step_rec["real_obs"]:
                raise RuntimeError(f"{game_id}: replayed step {t} gave a different observation")
            shown = step_rec["prediction_shown"]
        else:
            replies: dict[str, str] = {}
            last_raw = [""]

            def ask(prediction: str | None = None, proposed: str | None = None) -> str | None:
                msgs = agent_react_messages(observation=state.observation, history=history,
                                            foresight_prediction=prediction, proposal=proposed,
                                            admissible=state.admissible)
                key = repr(msgs)
                if key not in replies:
                    replies[key] = agent.complete(msgs)
                last_raw[0] = replies[key]
                return parse_react_action(replies[key])

            parsed = ask()
            proposal_raw, proposal_format_error = last_raw[0], parsed is None
            proposal = parsed or FORMAT_ERROR_ACTION
            fs = foresight_prediction(env, state, proposal, prefix, t, fc, success_cache=cache)
            oracle_obs, valid = fs["oracle_obs"], fs["proposal_valid"]
            false = None if plan is None else plan.false_text(
                t, proposal, valid, proposal_format_error, lambda: _twin(env, state, proposal, oracle_obs, cache))
            shown = oracle_obs if false is None else false
            action_under_true = ask(oracle_obs, proposal)
            under_true_format_error = action_under_true is None
            parsed = ask(shown, proposal)
            executed_raw, executed_format_error = last_raw[0], parsed is None
            executed = parsed or FORMAT_ERROR_ACTION
            new_state = env.step(state, executed)
            step_rec = {
                "t": t,
                "proposal": proposal,
                "proposal_raw": proposal_raw,
                "proposal_format_error": proposal_format_error,
                "proposal_valid": valid,
                "proposal_verb": action_type(proposal),
                "prediction_shown": shown,
                "prediction_type": "oracle" if false is None else f"corrupted_{arm}",
                "injected": false is not None,
                "action_under_true_prediction": action_under_true,
                "executed": executed,
                "executed_raw": executed_raw,
                "executed_format_error": executed_format_error,
                "real_obs": new_state.observation,
                "decision_critical": not under_true_format_error and action_under_true != proposal,
                "under_true_format_error": under_true_format_error,
                "followed": _followed({"prediction_shown": shown}, executed, proposal),
                "on_expert_path": on_expert_path(env, state, executed),
                "proposal_on_expert_path": on_expert_path(env, state, proposal),
                "targets_goal_object": names_object(executed, goal_object),
            }
        trace.append(step_rec)
        turn = {"observation": state.observation, "action": step_rec["executed"]}
        if retention == "retained" and shown is not None:
            turn.update(proposal=step_rec["proposal"], expected_outcome=shown)
        history.append(turn)
        prefix.append(step_rec["executed"])
        state = new_state

    return {"game_id": game_id, "split": split, "condition": "corrupted" if plan is not None else "oracle",
            "arm": arm if plan is not None else None, "retention": retention, "steps": len(trace),
            "success": bool(state.won), "trace": trace}


def run_false_success(env: AlfworldClient, agent: AgentClient, game_id: str, split: str, retention: str,
                      seed: int, p: float, cap: int, step_cap: int = C.STEP_CAP,
                      success_cache: dict | None = None) -> dict[str, Any]:
    plan = FalseSuccessPlan(game_id, seed, p, cap)
    ep = play_episode(env, agent, game_id, split, retention, plan, step_cap, success_cache)
    delivered = len(plan.injections)
    return {**ep, "schedule_seed": seed, "p_arm": p, "cap": cap, "chances": plan.chances,
            "planned_errors": cap, "delivered_errors": delivered, "n_corrupted": delivered,
            "shortfall": cap - delivered, "injections": plan.injections, "yoked_to": None, "fallbacks": 0,
            "restarts": []}


def run_false_failure(env: AlfworldClient, agent: AgentClient, game_id: str, split: str, retention: str,
                      partner: dict, step_cap: int = C.STEP_CAP,
                      success_cache: dict | None = None) -> dict[str, Any]:
    """Yoked to `partner`, the FS episode of the same game, schedule and retention.

    Whether a same-verb working proposal ever comes cannot be known at the step, so a pass first
    waits for one. If error i finds none by the end, the pass is restarted at the first working
    proposal at or after error i's start, which then takes error i as a logged fallback. The path
    up to that step is the earlier pass's, replayed exactly.
    """
    targets = [{"step": j["step"], "verb": j["verb"]} for j in partner["injections"]]
    any_verb: set[int] = set()
    replayed: list[dict] | None = None
    restarts: list[int] = []
    while True:
        plan = FalseFailurePlan(targets, frozenset(any_verb))
        ep = play_episode(env, agent, game_id, split, retention, plan, step_cap, success_cache, replayed)
        i = len(plan.injections)
        if i == len(targets) or i in any_verb:
            break
        start = plan.start(i)
        working = [st["t"] for st in ep["trace"]
                   if st["t"] >= start and st["proposal_valid"] and not st["proposal_format_error"]]
        if not working:
            break
        any_verb.add(i)
        replayed = ep["trace"][:working[0]]
        restarts.append(working[0])
    delivered = len(plan.injections)
    return {**ep, "schedule_seed": partner["schedule_seed"], "p_arm": None, "cap": partner["cap"], "chances": None,
            "planned_errors": len(targets), "delivered_errors": delivered, "n_corrupted": delivered,
            "shortfall": len(targets) - delivered, "injections": plan.injections, "yoked_to": targets,
            "fallbacks": sum(j["fallback"] for j in plan.injections), "restarts": restarts}


# ---------------- shards and jobs ----------------


def shard_spec(cfg: dict) -> tuple[int, int]:
    """(index, count): WF_SHARD / WF_N_SHARDS override the config's shard / n_shards."""
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"{cfg['run_id']}: shard {i} of {n} does not exist")
    return i, n


def shard_games(games: list[str], i: int, n: int) -> list[str]:
    return games[i::n]


def shard_run_id(base: str, i: int, n: int) -> str:
    return f"{base}/shard_{i:02d}_of_{n:02d}"


def episode_key(ep: dict) -> tuple:
    return (ep["game_id"], ep["condition"], ep.get("arm"), ep["retention"], ep.get("schedule_seed"))


def game_jobs(game_id: str, seeds: list[int]) -> list[tuple]:
    """Every episode of one game, each FS episode directly before its yoked FF partner."""
    jobs = [(game_id, cond, None, ret, None) for cond, ret in BASELINES]
    for seed in seeds:
        for ret in RETENTIONS:
            jobs += [(game_id, "corrupted", "favorable", ret, seed), (game_id, "corrupted", "unfavorable", ret, seed)]
    return jobs


def _run_job(job: tuple, client: AlfworldClient, agent: AgentClient, split: str, cfg: dict,
             saved: dict[tuple, dict], cache: dict) -> dict:
    gid, cond, arm, ret, seed = job
    step_cap = int(cfg.get("step_cap", C.STEP_CAP))
    if cond != "corrupted":
        return {**run_episode(client, agent, gid, split, cfg["run_id"], ForesightConfig(cond, retention=ret),
                              step_cap=step_cap), "schedule_seed": None}
    if arm == "favorable":
        return run_false_success(client, agent, gid, split, ret, seed, float(cfg["fs_probability"]),
                                 int(cfg["fs_cap"]), step_cap, cache)
    partner = saved[(gid, cond, "favorable", ret, seed)]
    baseline = saved.get((gid, "oracle", None, ret, None))
    if partner["delivered_errors"] == 0 and baseline is not None and cfg.get("copy_unyoked_ff", True):
        # No error to yoke, so the episode is the perfect-foresight baseline of this retention mode;
        # the analysis checks that assumption on the false-success episodes that also got none.
        return {**baseline, "condition": "corrupted", "arm": "unfavorable", "schedule_seed": seed, "p_arm": None,
                "cap": partner["cap"], "chances": None, "planned_errors": 0, "delivered_errors": 0, "n_corrupted": 0,
                "shortfall": 0, "injections": [], "yoked_to": [], "fallbacks": 0, "restarts": [],
                "copied_from_baseline": True}
    return run_false_failure(client, agent, gid, split, ret, partner, step_cap, cache)


def run_rq3(cfg: dict) -> int:
    base = cfg["run_id"]
    i, n = shard_spec(cfg)
    run_id = shard_run_id(base, i, n)
    require_filed(run_id)
    cfg = {**cfg, "run_id": run_id, "base_run_id": base, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    client.create()
    split = cfg.get("split", "train")
    seeds = [int(s) for s in cfg["schedule_seeds"]]
    all_games = _games(client, split, int(cfg["n_games"]), cfg)
    games = shard_games(all_games, i, n)
    write_json(d / "shard.json", {"shard": i, "n_shards": n, "games": games, "all_games": all_games})

    episodes = read_jsonl(d / "episodes.jsonl") if cfg.get("resume") else []
    saved = {episode_key(ep): ep for ep in episodes}
    done_games = {r["game_id"] for r in read_jsonl(d / "games_done.jsonl")} if cfg.get("resume") else set()
    # The server leaks memory on every reset, so a job stops after this much new work and a
    # resume job (fresh server) carries on.
    max_eps = int(cfg.get("max_episodes_per_job") or 0)
    max_games = int(cfg.get("max_games_per_job") or 0)
    new_eps = new_games = 0
    for gid in games:
        jobs = [job for job in game_jobs(gid, seeds) if job not in saved]
        if jobs:
            if max_games and new_games == max_games:
                return _partial(run_id, games, done_games, saved)
            new_games += 1
        cache: dict = {}
        for job in jobs:
            if max_eps and new_eps == max_eps:
                return _partial(run_id, games, done_games, saved)
            ep = {**_run_job(job, client, agent, split, cfg, saved, cache), "run_id": base, "shard": i}
            append_jsonl(d / "episodes.jsonl", [ep])
            saved[episode_key(ep)] = ep
            new_eps += 1
        if gid not in done_games:
            append_jsonl(d / "games_done.jsonl", [{"game_id": gid}])
            done_games.add(gid)

    # Bookkeeping only: outcomes wait for every shard (rq3_analysis), so no shard is an interim look.
    mine = [ep for ep in saved.values() if ep["game_id"] in games]
    corrupted = [ep for ep in mine if ep["condition"] == "corrupted"]
    book = {
        "games": len(games), "episodes": len(mine),
        "delivered_by_cell": {f"{a}/{r}": sum(ep["delivered_errors"] for ep in corrupted
                                              if ep["arm"] == a and ep["retention"] == r)
                              for a in ("favorable", "unfavorable") for r in RETENTIONS},
        "ff_shortfall_episodes": sum(ep["shortfall"] > 0 for ep in corrupted if ep["arm"] == "unfavorable"),
        "ff_fallbacks": sum(ep["fallbacks"] for ep in corrupted),
        "ff_restarts": sum(len(ep["restarts"]) for ep in corrupted),
    }
    write_json(d / "metrics.json", book)
    write_run_report(run_id, "rq3 shard", book)
    write_status(run_id, f"shard {i} of {n}: {len(games)} games, {len(mine)} episodes saved\n{book}",
                 ok=False, reported=SHARD_COMPLETE)
    return 0


def _partial(run_id: str, games: list[str], done_games: set[str], saved: dict) -> int:
    done = sum(g in done_games for g in games)
    write_status(run_id, f"PARTIAL — {done}/{len(games)} games and {len(saved)} episodes saved; "
                         f"resubmit with WF_RESUME=1", ok=False)
    return 1


# ---------------- analysis over all shards ----------------


def _shard_problems(source: Path) -> tuple[list[str], list[Path]]:
    folders = sorted(p for p in source.glob("shard_*_of_*") if p.is_dir())
    if not folders:
        return [f"no shard folders under {source}"], []
    counts = {p.name.rsplit("_of_", 1)[1] for p in folders}
    if len(counts) > 1:
        return [f"shard folders from different splits: {sorted(p.name for p in folders)}"], folders
    n = int(counts.pop())
    problems = []
    indices = sorted(int(p.name.split("_")[1]) for p in folders)
    if indices != list(range(n)):
        problems.append(f"shards present {indices}, expected 0..{n - 1}")
    shas, configs, metas = set(), set(), []
    for p in folders:
        status = (p / "STATUS.md").read_text(encoding="utf-8").splitlines()[0] if (p / "STATUS.md").exists() else ""
        if f"REPORTED ({SHARD_COMPLETE})" not in status:
            problems.append(f"{p.name} is not complete: {status or 'no status'}")
        shas.add((p / "git_sha").read_text(encoding="utf-8").strip() if (p / "git_sha").exists() else "missing")
        config = json.loads((p / "config.json").read_text(encoding="utf-8")) if (p / "config.json").exists() else {}
        configs.add(json.dumps({k: v for k, v in config.items() if k not in ("run_id", "shard")}, sort_keys=True))
        metas.append(json.loads((p / "shard.json").read_text(encoding="utf-8")) if (p / "shard.json").exists() else {})
    if len(shas) > 1:
        problems.append(f"shards ran at different commits: {sorted(shas)}")
    if len(configs) > 1:
        problems.append("shards ran with different settings")
    all_games = {json.dumps(m.get("all_games")) for m in metas}
    assigned = [g for m in metas for g in m.get("games", [])]
    if len(all_games) > 1:
        problems.append("shards were given different game lists")
    elif len(assigned) != len(set(assigned)) or set(assigned) != set(metas[0].get("all_games") or []):
        problems.append("shards do not split the game list into disjoint, complete parts")
    return problems, folders


def run_rq3_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    cfg = {**cfg, "resume": False}  # nothing to carry over: it reads the shards afresh each time
    source = runs_root() / cfg["source"]
    problems, folders = _shard_problems(source)
    d = _init_run(run_id, cfg, inputs=tuple(p / "episodes.jsonl" for p in folders if (p / "episodes.jsonl").exists()))
    if problems:
        write_status(run_id, "RQ3 results wait for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    shard_cfg = json.loads((folders[0] / "config.json").read_text(encoding="utf-8"))
    seeds = [int(s) for s in shard_cfg["schedule_seeds"]]
    episodes = [ep for p in folders for ep in read_jsonl(p / "episodes.jsonl")]
    games = [g for p in folders for g in json.loads((p / "shard.json").read_text(encoding="utf-8"))["games"]]
    expected = {job for g in games for job in game_jobs(g, seeds)}
    keys = [episode_key(ep) for ep in episodes]
    if set(keys) != expected or len(keys) != len(set(keys)):
        write_status(run_id, f"episodes do not match the design: {len(set(keys))} distinct of {len(expected)} "
                             f"expected, {len(keys) - len(set(keys))} repeated", ok=False)
        return 1
    summary = rq3_summary(episodes, step_cap=int(shard_cfg.get("step_cap", C.STEP_CAP)), seed=C.RNG_SEED)
    summary["provenance"] = {"source": cfg["source"], "shards": len(folders),
                             "git_sha": (folders[0] / "git_sha").read_text(encoding="utf-8").strip(),
                             "episodes_sha256": {p.name: file_sha(p / "episodes.jsonl") for p in folders}}
    from wf.provenance import provenance  # the common block (2026-11-20), added beside the original keys

    summary["provenance"].update(provenance(runs={cfg["source"]: source},
                                            inputs=[p / "episodes.jsonl" for p in folders]))
    write_json(d / "metrics.json", summary)
    write_run_report(run_id, "RQ3 (confirmatory)", summary)
    prim = summary["primary"]
    write_status(
        run_id,
        f"primary set: {prim['n_units']} units in {prim['n_games']} games\n"
        f"I_primary={prim['I']}\nH_discarded={prim['H_discarded']}\nH_retained={prim['H_retained']}\n"
        f"I_all_units={summary['all_units']['I']}\n{summary['reading']}\n"
        f"shortfalls={summary['shortfalls']}\nfallbacks={summary['fallbacks']}\n"
        f"determinism_check={summary['determinism_check']}",
        ok=False, reported="RQ3 interaction",
    )
    return 0
