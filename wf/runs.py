"""CLI: python -m wf.runs <run_id>"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

from wf import constants as C
from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.analysis.report import write_run_report
from wf.analysis.stats import (
    H_cell,
    H_success_points,
    asymmetry_A,
    asymmetry_ci,
    asymmetry_items,
    cluster_bootstrap,
    corruption_diagnostics,
    decision_flip_share,
    delta_c,
    delta_L,
    delta_L_first,
    delta_p,
    gen_says_success,
    gen_success_shift,
    icc,
    implied_se_delta_p,
    interaction_ci,
    pair_permutation,
    rank_gain_ci,
    rank_outcome,
    scoring_agreement_on_failures,
    with_partial_rule,
    success_gain_ci,
)
from wf.corruption.schedule import calibrate_p_arm, schedule
from wf.data.candidates import twin_available
from wf.data.transitions import (
    action_type,
    build_transitions_from_rollout,
    determination_reason,
    exclusion_counts,
    filter_pilot_a_items,
    main_test_items,
    primary_items,
    success_exclusion_counts,
)
from wf.env.expert import expert_available, label_action
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.server import AlfworldClient
from wf.scoring.logodds import LogoddsScorer, build_messages, score_transition
from wf.scoring.parser import parse_observation, parse_prediction
from wf.util import (
    REPO_ROOT,
    append_jsonl,
    calibration_dir,
    file_sha,
    git_dirty,
    git_sha,
    is_mock,
    read_jsonl,
    repair_jsonl,
    require_filed,
    run_dir,
    runs_root,
    write_json,
    write_jsonl,
    write_status,
)


def load_config(run_id: str) -> dict:
    path = REPO_ROOT / "wf" / "config" / f"{run_id}.yaml"
    if not path.exists():
        return {"run_id": run_id}
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["run_id"] = run_id
    return cfg


RUN_OUTPUTS = ("transitions.jsonl", "scoring.jsonl", "episodes.jsonl", "observations.jsonl",
               "items.jsonl", "generations.jsonl", "games_done.jsonl", "jobs.jsonl")
RULE_STOP_EXIT = 3  # a pilot rule says stop; distinct from a crash or a failed gate


def _check_resume(d: Path, run_id: str, config: dict, inputs: dict[str, str]) -> None:
    """Saved work may continue only under the same code, settings and input files."""
    saved_sha = d / "git_sha"
    if not is_mock() and git_dirty():
        raise SystemExit(f"{run_id}: the checkout has uncommitted changes, so a resume could mix code versions.")
    if saved_sha.exists() and saved_sha.read_text(encoding="utf-8").strip() != git_sha():
        raise SystemExit(f"{run_id}: saved work is from commit {saved_sha.read_text(encoding='utf-8').strip()}, "
                         f"this is {git_sha()}. Results must not mix versions; rerun without --resume.")
    saved_config = d / "config.json"
    if saved_config.exists():
        saved = {k: v for k, v in json.loads(saved_config.read_text(encoding="utf-8")).items() if k != "calibrated"}
        if saved != json.loads(json.dumps(config)):
            raise SystemExit(f"{run_id}: settings changed since the saved work; rerun without --resume.")
    saved_inputs = d / "inputs.json"
    if saved_inputs.exists() and json.loads(saved_inputs.read_text(encoding="utf-8"))["inputs"] != inputs:
        raise SystemExit(f"{run_id}: its input files changed since the saved work; rerun without --resume.")


_RUN_LOCKS: dict[str, Any] = {}


def _lock_run(run_id: str, d: Path) -> None:
    """Two jobs on one run would append to the same files; the second refuses to start."""
    if run_id in _RUN_LOCKS:
        return
    handle = (d / ".lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f"{run_id}: another job is already running this run ({d}).")
    _RUN_LOCKS[run_id] = handle


def _init_run(run_id: str, cfg: dict, inputs: tuple[Path, ...] = ()) -> Path:
    d = run_dir(run_id)
    _lock_run(run_id, d)
    if not is_mock() and git_sha() == "unknown":
        raise SystemExit(f"{run_id}: no git commit to attribute results to.")
    config = {k: v for k, v in cfg.items() if k != "resume"}
    hashes = {str(path): file_sha(path) for path in inputs}
    if cfg.get("resume"):
        _check_resume(d, run_id, config, hashes)
    # a crash must not leave the previous run's results standing
    for name in ("STATUS.md", "REPORT.md", "metrics.json"):
        (d / name).unlink(missing_ok=True)
    # fresh outputs for reproducibility of this invocation, unless resuming after a timeout
    for name in RUN_OUTPUTS:
        p = d / name
        if cfg.get("resume"):
            repair_jsonl(p)
        elif p.exists():
            p.unlink()
    write_json(d / "config.json", config)
    write_json(d / "inputs.json", {"inputs": hashes, "git_dirty": git_dirty()})
    (d / "git_sha").write_text(git_sha() + "\n", encoding="utf-8")
    # Host, Slurm job and GPU of every job that works on this run (a resumed run gets one line per job), so an
    # analysis can say which GPU scored which shard (wf/provenance.py).
    from wf.provenance import JOBS_FILE, job_record

    append_jsonl(d / JOBS_FILE, [job_record()])
    return d


def _client(cfg: dict) -> AlfworldClient:
    return AlfworldClient(base_url=cfg.get("alfworld_base_url"))


def _games(client: AlfworldClient, split: str, n: int, cfg: dict) -> list[str]:
    if cfg.get("games"):
        return list(cfg["games"])[:n]
    games = list(client.list_games(split))
    random.Random(C.RNG_SEED).shuffle(games)
    return games[:n]


def _random_walk(client: AlfworldClient, game_id: str, split: str, n: int = 12) -> tuple[list[str], list[str]]:
    """Seeded walk over admissible actions, for items that need no LLM: (actions, observations)."""
    rng = random.Random(f"{C.RNG_SEED}|{game_id}")
    state = client.reset(game_id, split)
    actions: list[str] = []
    observations: list[str] = []
    for _ in range(n):
        choices = sorted(a for a in state.admissible if a not in ("help", "look", "inventory"))
        if not choices:
            break
        action = rng.choice(choices)
        following = client.step(state, action)
        if following.done:
            break
        actions.append(action)
        observations.append(following.observation)
        state = following
    return actions, observations


def _episode_key(ep: dict) -> tuple:
    return (ep["game_id"], ep["condition"], ep.get("arm"), ep.get("retention"), ep.get("schedule_seed"), ep.get("p_arm"))


def _status_line(run_id: str) -> str:
    path = run_dir(run_id) / "STATUS.md"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return lines[0] if lines else ""


def _finished(run_id: str, commits: tuple[str, ...] = ()) -> bool:
    """Did this run end cleanly, on this code or a named commit, and are its inputs unchanged?"""
    d = run_dir(run_id)
    first = _status_line(run_id)
    if not first or not (d / "git_sha").exists():
        return False
    finished = "— PASS" in first or "— REPORTED" in first
    if ("(MOCK)" in first) != is_mock():
        return False
    sha = (d / "git_sha").read_text(encoding="utf-8").strip()
    if not is_mock() and sha == "unknown":
        return False
    if any(sha.startswith(c) for c in commits):
        return finished
    saved = d / "inputs.json"
    inputs = json.loads(saved.read_text(encoding="utf-8"))["inputs"] if saved.exists() else {}
    if any(not Path(path).exists() or file_sha(Path(path)) != digest for path, digest in inputs.items()):
        return False
    return finished and (d / "git_sha").read_text(encoding="utf-8").strip() == git_sha()


def _already_passed(run_id: str) -> bool:
    """--resume skips a run that finished cleanly at this commit, unless a pilot rule said stop."""
    return _finished(run_id) and "STOP:" not in _status_line(run_id)


def require_passed(run_id: str, cfg: dict, dep: str) -> None:
    """A run reads another's output only if that run finished. Otherwise it stops here.

    `accept_commits: {dep: [sha]}` in the run's config names an earlier commit to trust, and
    it stays on record in config.json.
    """
    if is_mock() and os.environ.get("WF_MOCK_CONTINUE") == "1":  # the mock dry run walks past gates
        return
    commits = tuple((cfg.get("accept_commits") or {}).get(dep, ()))
    problem = None if _finished(dep, commits) else f"{dep} has not finished cleanly at this commit: {_status_line(dep) or 'no status'}"
    # PILOT_RULE_HEADROOM is advisory since 2026-11-17, so a headroom stop from older code does not block.
    if (not problem and "STOP:" in _status_line(dep) and "PILOT_RULE_HEADROOM" not in _status_line(dep)
            and not _already_passed("pilotB_headroom_recheck")):
        problem = f"{dep} says stop and its recheck has not cleared it: {_status_line(dep)}"
    if problem:
        write_status(run_id, problem, ok=False)
        raise SystemExit(1)


def _saved_games(d: Path, name: str, cfg: dict) -> tuple[list[dict], set[str]]:
    """Records of fully saved games. A game counts only once games_done.jsonl lists it."""
    if not cfg.get("resume"):
        return [], set()
    finished = {r["game_id"] for r in read_jsonl(d / "games_done.jsonl")}
    records = read_jsonl(d / name)
    kept = [r for r in records if r["game_id"] in finished]
    if len(kept) != len(records):
        write_jsonl(d / name, kept)
    return kept, finished


def _save_game(d: Path, name: str, game_id: str, records: list[dict]) -> None:
    append_jsonl(d / name, records)
    append_jsonl(d / "games_done.jsonl", [{"game_id": game_id}])


def _run_jobs(d: Path, cfg: dict, client: AlfworldClient, agent: AgentClient, jobs: list[tuple],
              step_cap: int = C.STEP_CAP) -> list[dict]:
    """Run (game, ForesightConfig, schedule_seed) jobs, saving each episode as it finishes."""
    episodes = read_jsonl(d / "episodes.jsonl") if cfg.get("resume") else []
    done = {_episode_key(ep) for ep in episodes}
    for gid, fc, seed in jobs:
        if (gid, fc.condition, fc.arm, fc.retention, seed, fc.p_arm) in done:
            continue
        ep = run_episode(client, agent, gid, "train", cfg["run_id"], fc, step_cap=step_cap, schedule_seed=seed)
        append_jsonl(d / "episodes.jsonl", [ep])
        episodes.append(ep)
    return episodes


# ---------------- Tier 0 ----------------


def run_t0_smoke(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    n = int(cfg.get("n_games", 5))
    games = _games(client, "train", n, cfg)
    client.create()
    jobs = [(gid, ForesightConfig(condition="none"), None) for gid in games]
    episodes = _run_jobs(d, cfg, client, agent, jobs, step_cap=int(cfg.get("step_cap", C.STEP_CAP)))
    (d / "observations.jsonl").unlink(missing_ok=True)  # rebuilt from the saved episodes
    obs_count = wins = noops = steps = 0
    for ep in episodes:
        gid = ep["game_id"]
        first = client.reset(gid, "train").observation
        obs = [first] + [tr["real_obs"] for tr in ep["trace"]]
        append_jsonl(d / "observations.jsonl", [{"game_id": gid, "obs": o} for o in obs])
        obs_count += len(obs)
        wins += int(ep["success"])
        steps += len(ep["trace"])
        noops += sum(tr["real_obs"].strip() == "Nothing happens." for tr in ep["trace"])
    ok = obs_count >= 50
    noop_rate = noops / steps if steps else float("nan")
    write_run_report(
        run_id,
        "t0_smoke",
        {"observations": obs_count, "games": len(games), "wins": wins, "noop_rate": noop_rate, "STEP_CAP": C.STEP_CAP},
    )
    write_status(
        run_id,
        f"games={len(games)} wins={wins} observations={obs_count} noop_rate={noop_rate:.3f} STEP_CAP={C.STEP_CAP}\n"
        f"Acceptance: ≥50 observations — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    return 0 if ok else 1


def _logged_states(client: AlfworldClient, cfg: dict, want: int) -> list[dict]:
    """States from t0_smoke, with what the live run saw there; seeded walks if none exist."""
    states: list[dict] = []
    for ep in read_jsonl(runs_root() / "t0_smoke" / "episodes.jsonl"):
        trace = ep["trace"]
        executed = [tr["executed"] for tr in trace]
        for t in range(1, len(trace)):
            states.append(
                {"game_id": ep["game_id"], "prefix": executed[:t], "logged_obs": trace[t - 1]["real_obs"]}
            )
    if not states:
        for gid in _games(client, "train", want, cfg):
            actions, observations = _random_walk(client, gid, "train")
            for t in range(1, len(actions) + 1):
                states.append({"game_id": gid, "prefix": actions[:t], "logged_obs": observations[t - 1]})
    rng = random.Random(C.RNG_SEED)
    rng.shuffle(states)
    return states[:want]


def run_t0_replay(cfg: dict) -> int:
    """20 logged states × 3 counterfactual actions, executed twice (§6)."""
    run_id = cfg["run_id"]
    smoke = runs_root() / "t0_smoke" / "episodes.jsonl"
    if smoke.exists():
        require_passed(run_id, cfg, "t0_smoke")
    d = _init_run(run_id, {**cfg, "resume": False},  # minutes long: restarts rather than resumes
                  inputs=(smoke,) if smoke.exists() else ())
    client = _client(cfg)
    client.create()
    rng = random.Random(C.RNG_SEED)
    states = _logged_states(client, cfg, 20)
    source = "t0_smoke" if smoke.exists() else "seeded walks"
    total = mismatches = live_mismatches = informative = 0
    for st in states:
        handle = replay(client, st["game_id"], st["prefix"], "train")
        live_ok = handle.observation == st["logged_obs"]
        live_mismatches += not live_ok
        admissible = sorted(a for a in handle.admissible if a not in ("help", "look", "inventory"))
        actions = rng.sample(admissible, min(2, len(admissible))) + ["take mug 1 from nowhere 1"]
        for act in actions:
            total += 1
            first = replay(client, st["game_id"], st["prefix"], "train")
            o1, i1 = step_counterfactual(first, act)
            second = replay(client, st["game_id"], st["prefix"], "train")
            o2, i2 = step_counterfactual(second, act)
            match = (o1, i1["admissible_commands"]) == (o2, i2["admissible_commands"])
            mismatches += not match
            informative += o1.strip() != "Nothing happens."
            append_jsonl(
                d / "observations.jsonl",
                [{"game_id": st["game_id"], "action": act, "o1": o1, "o2": o2, "match": match, "live_ok": live_ok}],
            )
    ok = total > 0 and mismatches == 0 and live_mismatches == 0
    write_run_report(
        run_id,
        "t0_replay",
        {"states": len(states), "source": source, "checks": total, "mismatches": mismatches,
         "informative": informative, "replay_vs_live_mismatches": live_mismatches},
    )
    write_status(
        run_id,
        f"states={len(states)} from {source} replay checks={total} identical={total - mismatches} "
        f"informative={informative} replay_vs_live_mismatches={live_mismatches}\n"
        f"Acceptance: 100% identical — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    return 0 if ok else 1


def run_t0_agent_determinism(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    client.create()
    games = _games(client, "train", int(cfg.get("n_games", 10)), cfg)
    saved: dict[str, list[dict]] = {}
    for ep in read_jsonl(d / "episodes.jsonl") if cfg.get("resume") else []:
        saved.setdefault(ep["game_id"], []).append(ep)
    agreements = []
    for gid in games:
        pair = saved.get(gid, [])[-2:]
        if len(pair) < 2:
            pair = [run_episode(client, agent, gid, "train", run_id, ForesightConfig("none")) for _ in range(2)]
            append_jsonl(d / "episodes.jsonl", pair)
        ep1, ep2 = pair
        a1 = [t["executed"] for t in ep1["trace"]]
        a2 = [t["executed"] for t in ep2["trace"]]
        n = max(len(a1), len(a2), 1)
        agree = sum(1 for i in range(min(len(a1), len(a2))) if a1[i] == a2[i]) / n
        agreements.append(agree)
    mean_agree = sum(agreements) / len(agreements) if agreements else 0.0
    ok = mean_agree >= 0.95
    note = ""
    if not ok:
        note = ("\nTrajectory agreement < 95%, so schedules are not pure replicates. The agent picks greedily at "
                "temperature 0, so a seed cannot help: report this and investigate.")
    write_run_report(run_id, "t0_agent_determinism", {"mean_agreement": mean_agree})
    write_status(
        run_id,
        f"mean step agreement={mean_agree:.3f} over {len(games)} games{note}",
        reported=f"agreement {mean_agree:.3f}" if ok else f"BELOW TARGET: agreement {mean_agree:.3f} < 0.95",
    )
    return 0


def _dataset_observations(n: int) -> list[str]:
    """Environment observations from LLMasWorldModels. No fallback: failure must be visible."""
    if is_mock():
        return [
            "Nothing happens.",
            "You pick up the mug 1 from the drawer 1.",
            "You open the drawer 1. The drawer 1 is open. In it, you see a mug 1.",
            "You move the mug 1 to the coffeemachine 1.",
            "You arrive at countertop 1. On the countertop 1, you see a apple 1.",
            "You are not carrying anything.",
        ] * (n // 6 + 1)
    local = Path(os.environ.get("WORD2WORLD_ROOT", "")) / "data" / C.PARSER_DATASET_FILE
    path = os.environ.get("WF_PARSER_DATASET_FILE") or (str(local) if local.is_file() else None)
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(C.TRANSITION_DATASET, C.PARSER_DATASET_FILE, repo_type="dataset")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = [
        m["content"].split(" [SUCCESS]")[0]
        for row in rows
        for m in row["messages"]
        if m["role"] == "assistant" and m["content"]
    ]
    random.Random(C.RNG_SEED).shuffle(texts)
    return texts[:n]


def run_t0_parser(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, {**cfg, "resume": False})  # minutes long: restarts rather than resumes
    try:
        texts = _dataset_observations(2000)[:2000]
    except Exception as exc:
        write_status(run_id, f"could not read {C.TRANSITION_DATASET}: {exc!r}", ok=False)
        return 1
    if not texts:
        write_status(run_id, f"no observations in {C.TRANSITION_DATASET}", ok=False)
        return 1
    unparse = 0
    for t in texts:
        cls = parse_observation(t, "mug")
        if cls == "unparseable":
            unparse += 1
    rate = unparse / len(texts)
    ok = rate <= C.MAX_UNPARSEABLE
    write_run_report(run_id, "t0_parser", {"n": len(texts), "unparseable_rate": rate})
    append_jsonl(d / "observations.jsonl", [{"n": len(texts), "unparseable_rate": rate}])
    write_status(
        run_id,
        f"n={len(texts)} unparseable={unparse} rate={rate:.4f} "
        f"threshold={C.MAX_UNPARSEABLE} — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    return 0 if ok else 1


def _walk_transitions(client: AlfworldClient, game_id: str, split: str) -> list[dict]:
    actions, _ = _random_walk(client, game_id, split)
    return build_transitions_from_rollout(client, game_id, split, actions, k=2, rng_seed=C.RNG_SEED)


def run_t0_ckpt_accuracy(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    # Read-only valid_unseen, 200 items
    n = int(cfg.get("n_items", 200))
    scorer = LogoddsScorer(C.WM_FT_QWEN, model_tag="w2w-qwen2.5-7b")
    client = _client(cfg)
    client.create()
    games = _games(client, "valid_unseen", 40, {**cfg, "games": cfg.get("valid_unseen_games")})
    per_game = max(1, n // max(len(games), 1) + 1)
    items, finished = _saved_games(d, "items.jsonl", cfg)
    for gid in games:
        if len(items) >= n:
            break
        if gid in finished:
            continue
        built = _walk_transitions(client, gid, "valid_unseen")
        random.Random(f"{C.RNG_SEED}|{gid}").shuffle(built)
        kept = [{**t, "id": f"ckpt_{len(items) + i}"} for i, t in enumerate(built[:per_game])]
        _save_game(d, "items.jsonl", gid, kept)
        items.extend(kept)
    items = items[:n]
    if not items:
        write_status(run_id, "no valid_unseen items collected — FAIL", ok=False)
        return 1
    task_line_present = sum("Your task is to:" in build_messages(it, "aligned", ft_prompt=True)[0]["content"] for it in items)
    counts = {True: [0, 0], False: [0, 0]}
    done = {g["id"]: g for g in read_jsonl(d / "generations.jsonl")} if cfg.get("resume") else {}
    for it in items:
        rec = done.get(it["id"])
        if rec is None:
            gen = scorer.free_generate(build_messages(it, "aligned", ft_prompt=True), max_new_tokens=256)
            rec = {"id": it["id"], "generated": gen, "exact": gen.split(" [SUCCESS]")[0].strip() == it["true_obs"].strip()}
            append_jsonl(d / "generations.jsonl", [rec])
        counts[bool(it["context_determined"])][0] += int(rec["exact"])
        counts[bool(it["context_determined"])][1] += 1
    det_acc = counts[True][0] / counts[True][1] if counts[True][1] else float("nan")
    und_acc = counts[False][0] / counts[False][1] if counts[False][1] else float("nan")
    write_run_report(
        run_id,
        "t0_ckpt_accuracy",
        {"determined_acc": det_acc, "uncertain_acc": und_acc, "n": len(items),
         "n_determined": counts[True][1], "task_line_present": task_line_present},
    )
    write_json(d / "metrics.json", {"determined_acc": det_acc, "uncertain_acc": und_acc})
    write_status(
        run_id,
        f"valid_unseen n={len(items)} exact-match determined_acc={det_acc} (n={counts[True][1]}) "
        f"uncertain_acc={und_acc} (n={counts[False][1]}) task_line_present={task_line_present}/{len(items)} — reported",
        reported="no target",
    )
    return 0


def run_t0_scoring_agreement(cfg: dict) -> int:
    """100 distinct train items, half true failures: sign(L) vs parsed free generation."""
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    client.create()
    scorer = LogoddsScorer(C.WM_FEWSHOT_MODEL, model_tag="qwen3-8b-fewshot")
    items, finished = _saved_games(d, "items.jsonl", cfg)
    for gid in _games(client, "train", 30, cfg):
        if len(items) >= 100:
            break
        if gid in finished:
            continue
        built = [t for t in _walk_transitions(client, gid, "train") if t.get("candidates")]
        random.Random(f"{C.RNG_SEED}|{gid}").shuffle(built)
        failures = [t for t in built if t["true_obs"].strip() == "Nothing happens."][:5]
        successes = [t for t in built if t["true_obs"].strip() != "Nothing happens."][:5]
        kept = [{**t, "id": f"agree_{len(items) + i}"} for i, t in enumerate(failures + successes)]
        _save_game(d, "items.jsonl", gid, kept)
        items.extend(kept)
    items = items[:100]
    if len(items) < 100:
        write_status(run_id, f"only {len(items)} distinct items with candidates; need 100 — FAIL", ok=False)
        return 1
    agree = 0
    done = {s["transition_id"]: s for s in read_jsonl(d / "scoring.jsonl")} if cfg.get("resume") else {}
    for it in items:
        rec = done.get(it["id"])
        if rec is None:
            gen_cls = parse_prediction(scorer.free_generate(build_messages(it, "aligned")), None, it["action"])
            sc = score_transition(scorer, it, "aligned")
            assert sc is not None
            gen_success = bool(gen_says_success(gen_cls))
            rec = {**sc.to_dict(), "generated_class": gen_cls, "agree": (sc.L > 0) == gen_success}
            append_jsonl(d / "scoring.jsonl", [rec])
        agree += rec["agree"]
    rate = agree / len(items)
    ok = rate >= C.SCORING_AGREEMENT_MIN
    write_run_report(run_id, "t0_scoring_agreement", {"agreement": rate, "n_distinct": len(items)})
    write_status(
        run_id,
        f"agreement={agree}/{len(items)} ({rate:.2f}) min={C.SCORING_AGREEMENT_MIN} — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    return 0 if ok else 1


def run_t0_expert_labels(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, {**cfg, "resume": False})  # minutes long: restarts rather than resumes
    client = _client(cfg)
    client.create()
    games = _games(client, "train", 5, cfg)
    available = expert_available(client)
    labels = []
    for gid in games:
        st = client.reset(gid, "train")
        for act in (st.admissible or ["look"])[:5]:
            labels.append(label_action(client, st, act))
    append_jsonl(d / "episodes.jsonl", labels)
    if available:
        msg = f"expert labels available: yes; n_labels={len(labels)}"
    else:
        msg = "expert labels available: no — on_expert_path=null; do not approximate with a heuristic."
    write_run_report(run_id, "t0_expert_labels", {"available": available, "n": len(labels)})
    write_status(run_id, msg, reported="no target")
    return 0


# ---------------- Pilot A / B ----------------


def run_pilotA_transitions(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    client.create()
    n_games = int(cfg.get("n_games", 40))
    split = cfg.get("split", "train")
    games = _games(client, split, n_games, cfg)
    # Shards split one collection across parallel jobs: disjoint games, disjoint item ids.
    shard, shards = int(cfg.get("shard_index", 0)), int(cfg.get("shard_count", 1))
    games = games[shard::shards]
    all_t, finished = _saved_games(d, "transitions.jsonl", cfg)
    tid = shard * 1_000_000 + len(all_t)
    # The server leaks memory on every reset, so a long collection stops after this many new games
    # and a resume job (fresh server) carries on.
    per_job = int(cfg.get("max_games_per_job", 0)) or len(games)
    new_games = 0
    for gid in games:
        if gid in finished:
            continue
        if new_games == per_job:
            done = sum(g in finished for g in games)
            write_status(run_id, f"PARTIAL — {done}/{len(games)} games saved; resubmit with WF_RESUME=1", ok=False)
            return 1
        new_games += 1
        # short rollout to collect proposals
        ep = run_episode(client, agent, gid, split, run_id, ForesightConfig("none"),
                         step_cap=int(cfg.get("step_cap", 12)))
        actions = [t["executed"] for t in ep["trace"]]
        recs = build_transitions_from_rollout(
            client, gid, split, actions, id_start=tid, k=2, rng_seed=C.RNG_SEED
        )
        tid += len(recs)
        all_t.extend(recs)
        _save_game(d, "transitions.jsonl", gid, recs)
        finished.add(gid)
    eligible = filter_pilot_a_items(all_t)
    main_items = primary_items(eligible)
    states = {t["game_id"] + f"#{t['state_idx']}" for t in main_items}
    no_hidden_state = sorted({t["game_id"] for t in all_t if "hidden_state" in t and not t["hidden_state"]})
    mismatched = _hidden_state_mismatches(all_t)
    verify = _verify_items(client, all_t, split, int(cfg.get("verify_sample", 0)))
    min_items, min_states = int(cfg.get("min_items", 500)), int(cfg.get("min_states", 80))
    ok = (len(main_items) >= min_items and len(states) >= min_states and not no_hidden_state and not mismatched
          and not verify["mismatches"])
    by_reason: dict[str, int] = {}
    for t in eligible:
        reason = t.get("determination_reason") or determination_reason(t)
        by_reason[reason] = by_reason.get(reason, 0) + 1
    counts: dict[str, int] = {}
    for t in eligible:
        kind = "goal-agnostic" if t.get("history_goal_agnostic") else "goal-revealing"
        key = f"{kind}/{t.get('error_type')}"
        counts[key] = counts.get(key, 0) + 1
    n_fail = sum((t.get("true_obs") or "").strip() == "Nothing happens." for t in all_t)
    balance = {"failures": n_fail, "successes": len(all_t) - n_fail}
    targeted = sum(bool(t.get("targets_goal_object")) for t in eligible)
    exclusions = {"failures": exclusion_counts(all_t), "successes": success_exclusion_counts(all_t)}
    n_flagged = len(eligible) - len(main_test_items(eligible))
    write_run_report(
        run_id,
        "pilotA_transitions",
        {"n_total": len(all_t), "n_eligible": len(eligible), "n_main_test": len(main_items),
         "n_shared_location": n_flagged, "n_targeted": targeted, "n_states": len(states),
         "counts": counts, "balance": balance, "exclusions": exclusions,
         "eligible_by_reason": by_reason, "games_without_hidden_state": no_hidden_state,
         "games_with_mismatched_hidden_state": mismatched, "replay_check": verify},
    )
    write_status(
        run_id,
        f"total={len(all_t)} eligible_failure_determined={len(eligible)} main_test={len(main_items)} "
        f"shared_location={n_flagged} targeted={targeted} states={len(states)}\n"
        f"counts={counts}\nbalance={balance}\nexclusions={exclusions}\n"
        f"eligible_by_reason={by_reason}\ngames_without_hidden_state={no_hidden_state}\n"
        f"games_with_mismatched_hidden_state={mismatched}\n"
        f"replay_check: {verify['checked']} items replayed, mismatches={verify['mismatches']}\n"
        f"Acceptance ≥{min_items} main-test items / ≥{min_states} states — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    return 0 if ok else 1


def _hidden_state_mismatches(records: list[dict]) -> list[str]:
    """Games whose hidden state does not list every object instance the live scan found.

    The hidden state must describe this game's own scene. Instances are matched exactly, as parsed from the
    hidden state's object list; until 2026-11-20 this was a substring test, so "mug 1" was found inside
    "mug 10" (audit B8; wf.tools.verify_items has always compared exactly).
    """
    from wf.tools.verify_items import hidden_locations

    parsed: dict[str, set[str]] = {}
    bad = set()
    for t in records:
        if is_mock() or not t.get("hidden_state") or not t.get("scene_locations"):
            continue
        if t["hidden_state"] not in parsed:
            parsed[t["hidden_state"]] = set(hidden_locations(t["hidden_state"]))
        if any(inst not in parsed[t["hidden_state"]] for inst in t["scene_locations"]):
            bad.add(t["game_id"])
    return sorted(bad)


def _verify_items(client: AlfworldClient, records: list[dict], split: str, n: int) -> dict:
    """Replay a seeded sample of saved items on the server: the history and the true outcome must
    reproduce exactly, or the saved data does not describe what the environment does."""
    if n <= 0 or not records:
        return {"checked": 0, "mismatches": []}
    sample = random.Random(f"{C.RNG_SEED}|verify").sample(records, min(n, len(records)))
    mismatches = []
    for rec in sample:
        handle = replay(client, rec["game_id"], list(rec["prefix_actions"]), split=split)
        history = history_text_from_state(handle.state)
        obs, _ = step_counterfactual(handle, rec["action"])
        problems = []
        if history != rec["history_text"]:
            problems.append("history")
        if obs.strip() != (rec["true_obs"] or "").strip():
            problems.append("true_obs")
        if problems:
            mismatches.append({"id": rec["id"], "game_id": rec["game_id"], "action": rec["action"], "fields": problems})
    return {"checked": len(sample), "mismatches": mismatches}


def _effect(scores: list[dict], items: list[dict], require_agnostic: bool = True,
            primary: str = "aligned", control: str = "swapped", outcome: str = "failure") -> dict:
    """Δp, ΔL and first-token ΔL on one population, each with a game-clustered CI and permutation p.

    `outcome` picks the items by their true outcome: "failure" (the default, the preregistered population),
    "success" or "any" (wf.analysis.stats.paired_conditions).
    """
    dp = delta_p(scores, items, primary, control, require_agnostic, outcome=outcome)
    dl = delta_L(scores, items, primary, control, require_agnostic, outcome=outcome)
    df = delta_L_first(scores, items, primary, control, require_agnostic, outcome=outcome)
    # ΔL = Δlogit(q_first) - Δc: if the penalty moves with the goal, ΔL is not a belief shift alone.
    dc = delta_c(scores, items, primary, control, require_agnostic, outcome=outcome)
    masses = [s["first_mass"] for s in scores if s.get("first_mass") is not None and s["first_mass"] == s["first_mass"]
              and s.get("condition") in (primary, control) and s["transition_id"] in {t["id"] for t in items}]
    games = dp["item_games"]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else float("nan")
    boot_p = cluster_bootstrap(mean, dp["item_diffs"], games, seed=C.RNG_SEED)
    boot_l = cluster_bootstrap(mean, dl["item_diffs"], games, seed=C.RNG_SEED)
    boot_f = cluster_bootstrap(mean, df["item_diffs"], df["item_games"], seed=C.RNG_SEED)
    boot_c = cluster_bootstrap(mean, dc["item_diffs"], dc["item_games"], seed=C.RNG_SEED)
    return {
        "n_items": dp["n_items"],
        "n_games": dp["n_games"],
        "delta_p_pp": dp["delta_p_pp"],
        "delta_p_ci": [boot_p["ci_low"], boot_p["ci_high"]],
        "delta_p_perm_p": pair_permutation(dp["item_diffs"], seed=C.RNG_SEED, cluster_ids=games),
        "delta_L_nats": dl["delta_L_nats"],
        "delta_L_ci": [boot_l["ci_low"], boot_l["ci_high"]],
        "delta_L_perm_p": pair_permutation(dl["item_diffs"], seed=C.RNG_SEED, cluster_ids=games),
        # Secondary (decision 5): the same shift read off the first token only, free of the wording penalty.
        "delta_L_first_nats": df["delta_L_first_nats"],
        "delta_L_first_ci": [boot_f["ci_low"], boot_f["ci_high"]],
        "delta_L_first_perm_p": pair_permutation(df["item_diffs"], seed=C.RNG_SEED, cluster_ids=df["item_games"]),
        "delta_c_nats": dc["delta_c_nats"],
        "delta_c_ci": [boot_c["ci_low"], boot_c["ci_high"]],
        "first_mass_median": statistics.median(masses) if masses else float("nan"),
        "icc_L": icc(dl["item_diffs"], games),
        "icc_L_first": icc(df["item_diffs"], df["item_games"]),
        "icc_p": icc(dp["item_diffs"], games),
        "sd_L": sd(dl["item_diffs"]),
        "sd_L_first": sd(df["item_diffs"]),
        "sd_p": sd(dp["item_diffs"]),
    }


def _item_hash(t: dict) -> str:
    """What a score depends on, so it can never attach to a different item with the same id."""
    fields = {k: t.get(k) for k in ("action", "history_text", "task_line", "swapped_task_line", "candidates")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def _n_for_power(icc_val: float, sd: float, effect: float) -> int | None:
    """Smallest n (items over CONFIRMATORY_GAMES games) whose implied SE detects `effect` at 80% power."""
    se_needed = effect / (C.POWER_Z_ALPHA + C.POWER_Z_BETA)
    for n in range(200, 20001, 100):
        se = implied_se_delta_p(icc_val, n, C.CONFIRMATORY_GAMES, sd)
        if se == se and se <= se_needed:
            return n
    return None


def _false_success_by_reason(generations: list[dict], failures: list[dict]) -> dict:
    """Share of determined failures the model writes as successes, per deciding fact (decision 6)."""
    out = {}
    for reason in sorted({determination_reason(t) for t in failures}):
        said = [s for _, (failed, s) in asymmetry_items(generations, [t for t in failures if determination_reason(t) == reason]) if failed]
        out[reason] = {"rate": sum(said) / len(said) if said else float("nan"), "n": len(said)}
    return out


PILOT_A_MODELS = [
    # (model id, tag, fine-tuned prompt with hidden state)
    (C.WM_FEWSHOT_MODEL, "qwen3-8b-fewshot", False),
    (C.WM_FEWSHOT_LLAMA, "llama3.1-8b-fewshot", False),
    (C.WM_FT_QWEN, "w2w-qwen2.5-7b", True),
]


def _pilotA_item_sets(transitions: list[dict]) -> dict[str, list[dict]]:
    """The item sets Pilot A scores and analyses, built the same way for scoring and for re-parsing."""
    eligible = filter_pilot_a_items(transitions)
    # True successes get the aligned condition only, for A: false-failure needs them.
    all_successes = [
        t for t in transitions
        if t.get("context_determined") and t.get("candidates")
        and (t.get("true_obs") or "").strip() != "Nothing happens."
        and t.get("outcome_class") != "unparseable"
    ]
    # The aligned goal must be one the action's success would advance (prereg).
    targeted = [t for t in eligible if t.get("targets_goal_object")]
    return {
        "eligible": eligible,
        "all_successes": all_successes,
        "successes": main_test_items(all_successes),
        "agnostic_all": [t for t in targeted if t.get("history_goal_agnostic")],
        "agnostic": primary_items(eligible),
        "revealing": main_test_items(t for t in targeted if not t.get("history_goal_agnostic")),
    }


def _generation_report(gens: list[dict], scores: list[dict], sets: dict[str, list[dict]]) -> dict:
    """Everything Pilot A reads from the model's own text (Ŷ_gen), so a parser fix can recompute it."""
    failures = main_test_items(sets["eligible"])
    everything = failures + sets["successes"]
    return {
        # Decision 5: RQ1 reads the model's own text (Ŷ_gen), never the sign of L.
        "A": asymmetry_A(gens, everything),
        "A_ci": asymmetry_ci(gens, everything, seed=C.RNG_SEED),
        "A_sensitivity": {rule: asymmetry_A(with_partial_rule(gens, rule), everything)
                          for rule in ("exclude_partial", "partial_as_failure")},
        # what the model predicted for actions that fail, by kind of predicted effect
        "failure_predictions_by_class": dict(Counter(
            g["generated_class"] for g in gens
            if g.get("condition", "aligned") == "aligned" and g["transition_id"] in {t["id"] for t in failures})),
        "false_success_by_reason": _false_success_by_reason(gens, failures),
        # Checks that ΔL reflects belief: does the goal change what the model writes?
        "gen_success_shift": gen_success_shift(gens, sets["agnostic"]),
        "scoring_agreement_failures": scoring_agreement_on_failures(scores, gens, failures),
        "with_shared_location": {
            "n_flagged": len(sets["agnostic_all"]) - len(sets["agnostic"]),
            "A": asymmetry_A(gens, sets["eligible"] + sets["all_successes"]),
        },
    }


def run_pilotA_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, "pilotA_transitions")
    src = Path(cfg.get("transitions_path", runs_root() / "pilotA_transitions" / "transitions.jsonl"))
    if not src.exists():
        write_status(run_id, f"No transitions at {src}", ok=False)
        return 1
    d = _init_run(run_id, cfg, inputs=(src,))
    transitions = read_jsonl(src)
    if not transitions:
        write_status(run_id, f"No transitions at {src}", ok=False)
        return 1
    sets = _pilotA_item_sets(transitions)
    eligible, all_successes, agnostic = sets["eligible"], sets["all_successes"], sets["agnostic"]
    agnostic_all, revealing = sets["agnostic_all"], sets["revealing"]
    conditions = ["aligned", "swapped", "absent", "committed", "candidate"]
    models = PILOT_A_MODELS
    bins = {
        "0": lambda n: n == 0,
        "1-3": lambda n: 1 <= n <= 3,
        "4-7": lambda n: 4 <= n <= 7,
        "8+": lambda n: n >= 8,
    }
    if not agnostic:
        write_status(run_id, "no goal-agnostic targeted items to analyse — FAIL", ok=False)
        return 1

    reports = {}
    skipped: dict[str, str] = {}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    saved_gens = read_jsonl(d / "generations.jsonl") if cfg.get("resume") else []
    hashes = {t["id"]: _item_hash(t) for t in eligible + all_successes}
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved + saved_gens):
        raise SystemExit(f"{run_id}: saved scores belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    gen_done = {(g["model"], g["transition_id"], g["condition"]) for g in saved_gens}
    # Ŷ_gen for RQ1 (aligned, every determined item); aligned and swapped on the main test, to check
    # that ΔL moves with what the model writes.
    gen_jobs = [(t, "aligned") for t in eligible + all_successes] + [(t, "swapped") for t in agnostic]
    agnostic_per_game = len(agnostic) / max(len({t["game_id"] for t in transitions}), 1)
    for model_name, tag, ft in models:
        try:
            scorer = LogoddsScorer(model_name, model_tag=tag)
        except OSError as e:  # gated or missing weights: record it, score the rest, finish later
            skipped[tag] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            continue
        scores = [s for s in saved if s["model"] == tag]
        gens = [g for g in saved_gens if g["model"] == tag]
        for t, cond in [(t, c) for t in eligible for c in conditions] + [(t, "aligned") for t in all_successes]:
            if (tag, t["id"], cond) in done:
                continue
            sc = score_transition(scorer, t, cond, ft_prompt=ft)
            if sc:
                rec = {**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}
                append_jsonl(d / "scoring.jsonl", [rec])
                scores.append(rec)
        for t, cond in gen_jobs:
            if (tag, t["id"], cond) in gen_done:
                continue
            text = scorer.free_generate(build_messages(t, cond, ft_prompt=ft))
            rec = {"model": tag, "transition_id": t["id"], "condition": cond, "game_id": t["game_id"],
                   "generated": text, "generated_class": parse_prediction(text, t.get("goal_object"), t["action"]),
                   "item_hash": hashes[t["id"]]}
            append_jsonl(d / "generations.jsonl", [rec])
            gens.append(rec)
        scorer.close()
        core = _effect(scores, agnostic)
        reasons = sorted({determination_reason(t) for t in agnostic})
        reports[tag] = {
            "goal_agnostic": core,
            "goal_revealing": _effect(scores, revealing, require_agnostic=False),
            "by_history_len": {
                name: _effect(scores, [t for t in agnostic if test(t.get("history_len", 0))])
                for name, test in bins.items()
            },
            # Decision 6: which fact decides each failure, so a reader can see where an effect lives.
            "by_reason": {r: _effect(scores, [t for t in agnostic if determination_reason(t) == r]) for r in reasons},
            "committed_vs_candidate": _effect(scores, agnostic, primary="committed", control="candidate"),
            **_generation_report(gens, scores, sets),
            "decision_flip": decision_flip_share(scores, agnostic),
        }
        # Sensitivity check, not a primary result: see notes/deviations.md
        reports[tag]["with_shared_location"]["goal_agnostic"] = _effect(scores, agnostic_all)
        reports[tag]["implied_se_pp"] = {
            n: implied_se_delta_p(core["icc_p"], n, C.CONFIRMATORY_GAMES, core["sd_p"]) for n in (1000, 2000, 3000)
        }
        reports[tag]["implied_se_nats"] = {
            n: implied_se_delta_p(core["icc_L"], n, C.CONFIRMATORY_GAMES, core["sd_L"]) for n in (1000, 2000, 3000)
        }
        reports[tag]["n_for_80pct_power"] = {
            str(effect): _n_for_power(core["icc_L"], core["sd_L"], effect) for effect in (0.5, 0.75, 1.0)
        }

    if not reports:
        write_status(run_id, f"no model could be scored: {skipped}", ok=False)
        return 1
    # PILOT_RULE_GOAL_CHANNEL, as filed: Δp on every tested model. Advisory since decision 4.
    fired = all(
        rep["goal_agnostic"]["delta_p_pp"] < C.FUTILITY_EST_PP
        and rep["goal_agnostic"]["delta_p_ci"][1] < C.FUTILITY_UPPER_PP
        for rep in reports.values()
    )
    # The same rule on the prereg's scale: every scored model's ΔL CI sits below 0.5 nats.
    fired_nats = all(rep["goal_agnostic"]["delta_L_ci"][1] < C.FUTILITY_UPPER_NATS for rep in reports.values())
    # Δp compresses when L is far from 0, so surface a filed rule that contradicts the prereg metric.
    conflict = [tag for tag, rep in reports.items() if fired and rep["goal_agnostic"]["delta_L_nats"] >= C.DELTA_L_SUPPORT_NATS]
    # Decision 13: the few-shot models' scores stand only if sign(L) agrees with their own text on failures.
    stop = [
        f"{tag} {rep['scoring_agreement_failures']['agree']}/{rep['scoring_agreement_failures']['n']}"
        for tag, rep in reports.items()
        if not tag.startswith("w2w") and not rep["scoring_agreement_failures"]["agreement"] >= C.SCORING_STOP_FAILURES_MIN
    ]

    # ELIGIBLE_SET_TARGET: smallest n ≥ the prereg floor with SE ≤ 1.5 on every model
    target_n = None
    for n in range(1000, 5001, 100):
        ses = [
            implied_se_delta_p(r["goal_agnostic"]["icc_p"], n, C.CONFIRMATORY_GAMES, r["goal_agnostic"]["sd_p"])
            for r in reports.values()
        ]
        if all(se == se and se <= 1.5 for se in ses):
            target_n = n
            break

    cal = {
        "ELIGIBLE_SET_TARGET": target_n,
        "PILOT_RULE_GOAL_CHANNEL": fired,
        "PILOT_RULE_GOAL_CHANNEL_nats": fired_nats,
        "goal_channel_rule_is_advisory": True,
        "rule_conflicts_with_delta_L": conflict,
        "scoring_stop": stop,
        "models_not_scored": skipped,
        "agnostic_items_per_game": agnostic_per_game,
        "agnostic_items_supply_estimate": round(agnostic_per_game * C.CONFIRMATORY_GAMES),
        "exclusions": {"failures": exclusion_counts(transitions), "successes": success_exclusion_counts(transitions)},
        "reports": reports,
    }
    with (calibration_dir() / "eligible_target.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({"ELIGIBLE_SET_TARGET": target_n}, f)
    write_json(d / "metrics.json", cal)
    summary = {
        tag: {k: rep["goal_agnostic"][k] for k in ("n_items", "n_games", "delta_p_pp", "delta_p_ci", "delta_L_nats",
                                                    "delta_L_ci", "delta_L_first_nats", "icc_L", "sd_L")}
        | {"A": rep["A"]["A"], "n_for_80pct_power": rep["n_for_80pct_power"]}
        for tag, rep in reports.items()
    }
    sensitivity = {
        tag: {"n_flagged": rep["with_shared_location"]["n_flagged"],
              "delta_L_nats": rep["with_shared_location"]["goal_agnostic"]["delta_L_nats"],
              "A": rep["with_shared_location"]["A"]["A"]}
        for tag, rep in reports.items()
    }
    write_run_report(run_id, "pilotA_score", cal)
    body = (
        f"{'STOP: ' + ', '.join(stop) + ' — scoring agreement on determined failures below ' + str(C.SCORING_STOP_FAILURES_MIN) + chr(10) if stop else ''}"
        f"PILOT_RULE_GOAL_CHANNEL (advisory) as filed in Δp: {'fired' if fired else 'not fired'}; "
        f"in nats: {'fired' if fired_nats else 'not fired'}; ELIGIBLE_SET_TARGET={target_n}\n"
        f"rule conflicts with ΔL on: {conflict or 'none'}\n{summary}\nsensitivity with shared_location items: {sensitivity}"
    )
    if skipped:
        # Not finished: a --resume after the weights become available scores only what is missing.
        write_status(run_id, f"INCOMPLETE — not scored: {skipped}\n{body}", ok=False)
        return 1
    write_status(run_id, body, reported="rule output")
    return 0


def _format_error_rate(episodes: list[dict]) -> float:
    steps = [st for e in episodes for st in e["trace"]]
    return sum(bool(st.get("proposal_format_error") or st.get("executed_format_error")) for st in steps) / max(len(steps), 1)


def run_pilotB_headroom(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    client.create()
    n = int(cfg.get("n_games", 50))
    games = _games(client, "train", n, cfg)
    conditions = [
        ForesightConfig("none"),
        ForesightConfig("oracle", retention="discarded"),
        ForesightConfig("oracle", retention="retained"),
        ForesightConfig("validity_only", retention="discarded"),
    ]
    episodes = _run_jobs(d, cfg, client, agent, [(gid, fc, None) for gid in games for fc in conditions])

    summary = _headroom_summary(episodes)
    if any(v != v for v in (summary["oracle_gain_pp"]["estimate"], summary["oracle_gain_pp"]["ci_low"],
                            summary["oracle_gain_pp"]["ci_high"])):
        write_status(run_id, f"oracle gain cannot be computed: {summary['oracle_gain_pp']}\n{summary['per_condition']}", ok=False)
        return 1

    # "Materially negative": the whole interval sits below zero, not one game's flip.
    broken = summary["retained_minus_discarded_pp"]["ci_high"] < 0
    write_run_report(run_id, "pilotB_headroom", summary)
    write_status(
        run_id,
        f"{summary['per_condition']}\noracle_gain={summary['oracle_gain_pp']}\noracle_steps_gain={summary['oracle_steps_gain']}"
        f"\nrecovery={summary['validity_recovery']}\nretained_minus_discarded={summary['retained_minus_discarded_pp']}"
        f"\n{summary['rule']}"
        + ("\nretained-oracle minus discarded-oracle is materially negative — retained format broken; "
           "stop before mini-run." if broken else ""),
        ok=False,
        reported=None if broken else "headroom advisory",
    )
    return 1 if broken else 0


def _headroom_summary(episodes: list[dict]) -> dict:
    """What perfect foresight does to success and to steps-to-completion. Reported, never a gate.

    The proposal's pilot check is whether perfect foresight changes the agent's actions
    (`pilotB_consumption`); the 5-point success line came only from the pilot code (2026-11-17).
    """
    def cell(cond: str, ret: str | None = None) -> list[dict]:
        return [e for e in episodes if e["condition"] == cond and (ret is None or e.get("retention") == ret)]

    def succ_rate(eps: list[dict]) -> float:
        return 100.0 * sum(1 for e in eps if e["success"]) / len(eps) if eps else float("nan")

    none, oracle_d, oracle_r, validity = cell("none"), cell("oracle", "discarded"), cell("oracle", "retained"), cell("validity_only")
    gain = success_gain_ci(oracle_d, none, seed=C.RNG_SEED)
    oracle_gain = succ_rate(oracle_d) - succ_rate(none)
    recovery = (succ_rate(validity) - succ_rate(none)) / oracle_gain if oracle_gain > 0 else float("nan")
    per_condition = {
        name: {"success": succ_rate(eps), "mean_S": rank_outcome(eps)["mean_S"], "format_error_rate": _format_error_rate(eps)}
        for name, eps in (("none", none), ("oracle_discarded", oracle_d), ("oracle_retained", oracle_r), ("validity_only", validity))
    }
    below = gain["estimate"] < C.HEADROOM_MIN_PP
    return {
        "per_condition": per_condition,
        "oracle_gain_pp": gain,
        "oracle_steps_gain": rank_gain_ci(oracle_d, none, seed=C.RNG_SEED),
        "validity_recovery": recovery,
        "retained_minus_discarded_pp": success_gain_ci(oracle_r, oracle_d, seed=C.RNG_SEED),
        "rule": f"PILOT_RULE_HEADROOM advisory (not a gate): oracle gain {gain['estimate']:.1f} pp is "
                f"{'below' if below else 'at or above'} the {C.HEADROOM_MIN_PP:g}-pp line",
    }


def run_pilotB_consumption(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, "pilotB_headroom")
    # Reuse headroom oracle episodes if present
    src = Path(cfg.get("episodes_path", runs_root() / "pilotB_headroom" / "episodes.jsonl"))
    d = _init_run(run_id, cfg, inputs=(src,) if src.exists() else ())
    all_episodes = read_jsonl(src)
    episodes = [e for e in all_episodes if e.get("condition") == "oracle"]
    if not episodes:
        write_status(run_id, f"No oracle episodes at {src}; run pilotB_headroom first", ok=False)
        return 1
    crit = follow = total = 0
    for ep in episodes:
        for tr in ep.get("trace", []):
            total += 1
            crit += int(bool(tr.get("decision_critical")))
            follow += int(bool(tr.get("followed")))
    rate = crit / total if total else 0.0
    follow_rate = follow / total if total else 0.0
    # The proposal's pilot check (perfect foresight changes the agent's actions) is the gate; headroom rides along.
    ok = rate >= C.CONSUMPTION_MIN
    headroom = _headroom_summary(all_episodes)
    metrics = {"decision_critical_rate": rate, "follow_rate": follow_rate,
               "headroom_advisory": {k: headroom[k] for k in ("oracle_gain_pp", "oracle_steps_gain", "rule")}}
    write_json(d / "metrics.json", metrics)
    write_run_report(run_id, "pilotB_consumption", metrics)
    write_status(
        run_id,
        f"decision_critical_rate={rate:.3f} follow_rate={follow_rate:.3f} "
        f"min={C.CONSUMPTION_MIN} — {'PASS' if ok else 'FAIL: strengthen foresight presentation'}"
        f"\noracle_gain={headroom['oracle_gain_pp']}\noracle_steps_gain={headroom['oracle_steps_gain']}\n{headroom['rule']}",
        ok=ok,
    )
    return 0 if ok else 1


def run_pilotB_eligibility(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, "pilotB_headroom")
    require_passed(run_id, cfg, "pilotB_consumption")
    src = Path(cfg.get("episodes_path", runs_root() / "pilotB_headroom" / "episodes.jsonl"))
    d = _init_run(run_id, cfg, inputs=(src,) if src.exists() else ())
    # Corrupted arms run with a prediction on screen, so only oracle episodes show their rates.
    episodes = [e for e in read_jsonl(src) if e.get("condition") == "oracle"]
    if not episodes:
        write_status(run_id, f"No oracle episodes at {src}; run pilotB_headroom first", ok=False)
        return 1
    client = _client(cfg)
    client.create()
    sandbox = client.sandbox()
    twins: dict = {}
    invalid = corruptible = valid = expert_valid = labelled_valid = steps = 0
    mix: dict[str, dict[str, int]] = {"favorable": {}, "unfavorable": {}}
    for ep in episodes:
        history: list[tuple[str, str]] = []
        for tr in ep.get("trace", []):
            steps += 1
            kind = action_type(tr["proposal"])
            if tr.get("proposal_valid"):
                valid += 1
                mix["unfavorable"][kind] = mix["unfavorable"].get(kind, 0) + 1
                if tr.get("proposal_on_expert_path") is not None:
                    labelled_valid += 1
                    expert_valid += int(tr["proposal_on_expert_path"])
            else:
                invalid += 1
                if twin_available(sandbox, ep["game_id"], ep["split"], tr["proposal"], history, twins):
                    corruptible += 1
                    mix["favorable"][kind] = mix["favorable"].get(kind, 0) + 1
            history.append((tr["executed"], tr["real_obs"]))
    exp_steps = steps / len(episodes)
    corruptible_rate = corruptible / steps if steps else 0.0
    val_rate = valid / steps if steps else 0.0
    achievable = {"favorable": corruptible_rate * exp_steps, "unfavorable": val_rate * exp_steps}
    target = C.CORRUPTIONS_PER_EPISODE_TARGET
    # One target for both arms, or a gap in error counts would read as a gap in cost.
    common_target = min(target, *achievable.values())
    p_fav = calibrate_p_arm(corruptible_rate, common_target, exp_steps)
    p_unfav = calibrate_p_arm(val_rate, common_target, exp_steps)
    expected = {"favorable": corruptible_rate * exp_steps * p_fav, "unfavorable": val_rate * exp_steps * p_unfav}
    short = {arm: round(e, 2) for arm, e in achievable.items() if e < target - 1e-9}
    cal = {
        "invalid_proposal_rate": invalid / steps if steps else 0.0,
        "corruptible_proposal_rate": corruptible_rate,
        "valid_proposal_rate": val_rate,
        "expert_among_valid": (expert_valid / labelled_valid) if labelled_valid else None,
        "expected_steps": exp_steps,
        "p_favorable": p_fav,
        "p_unfavorable": p_unfav,
        "expected_corruptions": expected,
        "common_target": common_target,
        "corruption_mix": mix,
        "arms_below_target": short,
        "CORRUPTIONS_PER_EPISODE_TARGET": target,
        "derivation": "p = common target / (corruptible rate * expected_steps) on oracle episodes, clipped to 1",
    }
    # write calibrated.yaml
    cal_path = calibration_dir() / "calibrated.yaml"
    with cal_path.open("w", encoding="utf-8") as f:
        yaml.dump(cal, f, sort_keys=True)
    write_json(d / "metrics.json", cal)
    write_run_report(run_id, "pilotB_eligibility", cal)
    note = f"\nWARNING: too few eligible steps to reach {target} corruptions per episode: {short}" if short else ""
    write_status(run_id, f"wrote {cal_path.name}: p_favorable={p_fav:.4f} p_unfavorable={p_unfav:.4f} "
                         f"common_target={common_target:.2f} mix={mix}{note}",
                 reported=f"BELOW TARGET: {short}" if short else "calibration")
    return 0


def run_pilotB_retention_minirun(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, "pilotB_eligibility")
    cal_path = calibration_dir() / "calibrated.yaml"
    if not cal_path.exists():
        write_status(run_id, f"{cal_path} missing; run pilotB_eligibility first", ok=False)
        return 1
    if C.RETENTION_MARGIN is None:
        write_status(run_id, "RETENTION_MARGIN is None — set before filing", ok=False)
        return 1
    d = _init_run(run_id, cfg, inputs=(cal_path,))
    with cal_path.open(encoding="utf-8") as f:
        cal = yaml.safe_load(f) or {}
    calibrated = {"p_favorable": cal["p_favorable"], "p_unfavorable": cal["p_unfavorable"]}
    write_json(d / "config.json", {**{k: v for k, v in cfg.items() if k != "resume"}, "calibrated": calibrated})
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    client.create()
    n = int(cfg.get("n_games", 30))
    games = _games(client, "train", n, cfg)
    seed = int(cfg.get("schedule_seed", C.RNG_SEED))
    jobs = []
    for gid in games:
        for arm, p in (("favorable", cal["p_favorable"]), ("unfavorable", cal["p_unfavorable"])):
            for retention in ("discarded", "retained"):
                fc = ForesightConfig(
                    condition="corrupted",
                    arm=arm,
                    retention=retention,
                    corrupt_fn=schedule(gid, seed, arm, float(p)),
                    p_arm=float(p),
                )
                jobs.append((gid, fc, seed))
    episodes = _run_jobs(d, cfg, client, agent, jobs)
    Hd = H_cell(episodes, "discarded")
    Hr = H_cell(episodes, "retained")
    I = interaction_ci(episodes, seed=C.RNG_SEED)
    H_points = {"discarded": H_success_points(episodes, "discarded"), "retained": H_success_points(episodes, "retained")}
    sign_claims = {ret: bool(abs(v) >= C.H_FLOOR_PP) for ret, v in H_points.items()}
    # retention margin on favorable arm
    fav_disc = [e for e in episodes if e["arm"] == "favorable" and e["retention"] == "discarded"]
    fav_ret = [e for e in episodes if e["arm"] == "favorable" and e["retention"] == "retained"]
    ret_S, disc_S = rank_outcome(fav_ret)["per_game_S"], rank_outcome(fav_disc)["per_game_S"]
    paired = sorted(set(ret_S) & set(disc_S))  # a game missing from one cell must not move the shift
    if not paired:
        write_status(run_id, "no game has both favorable cells; retention shift cannot be computed", ok=False)
        return 1
    shift = abs(sum(ret_S[g] - disc_S[g] for g in paired) / len(paired))
    if shift < C.RETENTION_MARGIN:
        fmt_rule = (
            "PILOT_RULE_RETAINED_FORMAT fired — switch to state persistence for full runs"
        )
    else:
        fmt_rule = "PILOT_RULE_RETAINED_FORMAT not fired — keep trace-line retained format"

    cells = {}
    for arm in ("favorable", "unfavorable"):
        for ret in ("discarded", "retained"):
            eps = [e for e in episodes if e["arm"] == arm and e["retention"] == ret]
            cells[f"{arm}/{ret}"] = {
                "success": sum(1 for e in eps if e["success"]) / max(len(eps), 1),
                "mean_S": rank_outcome(eps)["mean_S"],
                "mean_n_corrupted": sum(e["n_corrupted"] for e in eps) / max(len(eps), 1),
                "mean_n_unavailable": sum(e["n_corruption_unavailable"] for e in eps) / max(len(eps), 1),
                # Rule R reads this arm: replies with no Action line must not pass for agent choices.
                "format_error_rate": _format_error_rate(eps),
            }
    oracle = [
        e for e in read_jsonl(runs_root() / "pilotB_headroom" / "episodes.jsonl")
        if e.get("condition") == "oracle" and e.get("retention") == "discarded"
    ]
    diagnostics = corruption_diagnostics(episodes, oracle)
    write_run_report(
        run_id,
        "pilotB_retention_minirun",
        {"H_discarded": Hd, "H_retained": Hr, "H_success_points": H_points, "sign_claim_allowed": sign_claims,
         "I": I, "retention_shift": shift, "rule": fmt_rule, "cells": cells, "diagnostics": diagnostics,
         "calibrated": calibrated},
    )
    write_status(
        run_id,
        f"H_discarded={Hd} H_retained={Hr} I={I} retention_shift={shift}\n{fmt_rule}\ncells={cells}",
        reported="rule output",
    )
    return 0


def run_pilotA_reparse(cfg: dict) -> int:
    """Re-read Pilot A's saved generations with `parse_prediction`; no model is run (2026-11-17).

    The first parser dropped failure predictions written in the model's own words, so every
    measure built on Ŷ_gen is recomputed here, before and after, from the same saved text.
    """
    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, "pilotA_score")
    src_t = runs_root() / "pilotA_transitions" / "transitions.jsonl"
    src_s = runs_root() / "pilotA_score" / "scoring.jsonl"
    src_g = runs_root() / "pilotA_score" / "generations.jsonl"
    d = _init_run(run_id, cfg, inputs=(src_t, src_s, src_g))
    transitions = read_jsonl(src_t)
    goal_object = {t["id"]: t.get("goal_object") for t in transitions}
    action = {t["id"]: t.get("action") for t in transitions}
    sets = _pilotA_item_sets(transitions)
    scores, original = read_jsonl(src_s), read_jsonl(src_g)
    reparsed = [
        {**g, "generated_class_v1": g["generated_class"],
         "generated_class": parse_prediction(g["generated"], goal_object.get(g["transition_id"]),
                                             action.get(g["transition_id"]))}
        for g in original
    ]
    write_jsonl(d / "generations.jsonl", reparsed)

    per_model = {}
    for tag in sorted({g["model"] for g in reparsed}):
        before = [g for g in original if g["model"] == tag]
        after = [g for g in reparsed if g["model"] == tag]
        mine = [s for s in scores if s["model"] == tag]
        per_model[tag] = {
            "before": _generation_report(before, mine, sets),
            "after": _generation_report(after, mine, sets),
            "classes_before": dict(Counter(g["generated_class"] for g in before)),
            "classes_after": dict(Counter(g["generated_class"] for g in after)),
        }
    stop = [
        f"{tag} {r['after']['scoring_agreement_failures']['agree']}/{r['after']['scoring_agreement_failures']['n']}"
        for tag, r in per_model.items()
        if not tag.startswith("w2w")
        and not r["after"]["scoring_agreement_failures"]["agreement"] >= C.SCORING_STOP_FAILURES_MIN
    ]
    metrics = {"per_model": per_model, "scoring_stop_after": stop}
    write_json(d / "metrics.json", metrics)
    write_run_report(run_id, "pilotA_reparse", metrics)
    lines = [
        f"{tag}: A {r['before']['A']['A']:.3f} -> {r['after']['A']['A']:.3f}; "
        f"unparseable {r['before']['A']['n_unparseable']} -> {r['after']['A']['n_unparseable']}; "
        f"written shift {r['before']['gen_success_shift']['shift_pp']:.1f} -> {r['after']['gen_success_shift']['shift_pp']:.1f} pp; "
        f"agreement {r['before']['scoring_agreement_failures']['agreement']:.3f} -> "
        f"{r['after']['scoring_agreement_failures']['agreement']:.3f}"
        for tag, r in per_model.items()
    ]
    write_status(run_id, "\n".join(lines) + f"\nscoring stop after re-parse: {stop or 'none'}",
                 ok=False, reported="parser v2 re-read; not yet audited")
    return 0


RUNNERS: dict[str, Callable[[dict], int]] = {
    "t0_smoke": run_t0_smoke,
    "t0_replay": run_t0_replay,
    "t0_agent_determinism": run_t0_agent_determinism,
    "t0_parser": run_t0_parser,
    "t0_ckpt_accuracy": run_t0_ckpt_accuracy,
    "t0_scoring_agreement": run_t0_scoring_agreement,
    "t0_expert_labels": run_t0_expert_labels,
    "pilotA_transitions": run_pilotA_transitions,
    "rq12_transitions": run_pilotA_transitions,
    "rq12_transitions_smoke": run_pilotA_transitions,
    **{f"rq12_transitions_s{i}": run_pilotA_transitions for i in range(3)},
    "pilotA_score": run_pilotA_score,
    "pilotA_reparse": run_pilotA_reparse,
    "pilotB_headroom": run_pilotB_headroom,
    "pilotB_headroom_recheck": run_pilotB_headroom,
    "pilotB_consumption": run_pilotB_consumption,
    "pilotB_eligibility": run_pilotB_eligibility,
    "pilotB_retention_minirun": run_pilotB_retention_minirun,
    "rq3": lambda cfg: _rq3("run_rq3")(cfg),
    "rq3_smoke": lambda cfg: _rq3("run_rq3")(cfg),
    "rq3_analysis": lambda cfg: _rq3("run_rq3_analysis")(cfg),
    "rq3_smoke_analysis": lambda cfg: _rq3("run_rq3_analysis")(cfg),
    "rq1_score": lambda cfg: _rq1("run_rq1_score")(cfg),
    "rq1_score_smoke": lambda cfg: _rq1("run_rq1_score")(cfg),
    "rq1_analysis": lambda cfg: _rq1("run_rq1_analysis")(cfg),
    "rq1_smoke_analysis": lambda cfg: _rq1("run_rq1_analysis")(cfg),
    "rq1x_score": lambda cfg: _rq1x("run_rq1x_score")(cfg),
    "rq1x_analysis": lambda cfg: _rq1x("run_rq1x_analysis")(cfg),
    "rq1d_score": lambda cfg: _rq1d("run_rq1d_score")(cfg),
    "rq1v_score": lambda cfg: _rq1v("run_rq1v_score")(cfg),
    "rq1v_analysis": lambda cfg: _rq1v("run_rq1v_analysis")(cfg),
    "rq1d_analysis": lambda cfg: _rq1d("run_rq1d_analysis")(cfg),
    "mech_score": lambda cfg: _lazy("mech", "run_mech_score")(cfg),
    "mech_analysis": lambda cfg: _lazy("mech", "run_mech_analysis")(cfg),
    "mech_score_smoke": lambda cfg: _lazy("mech", "run_mech_score")(cfg),
    "size_analysis": lambda cfg: _lazy("sizes", "run_size_analysis")(cfg),
    **{f"size_score_{tag}": (lambda cfg: _lazy("sizes", "run_size_score")(cfg))
       for tag in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b")},
    "thinking_probe": lambda cfg: _thinking("run_thinking")(cfg),
    "thinking_probe_smoke": lambda cfg: _thinking("run_thinking")(cfg),
    "thinking_analysis": lambda cfg: _thinking("run_thinking_analysis")(cfg),
    "rq2_score": lambda cfg: _rq2("run_rq2_score")(cfg),
    "rq2_score_smoke": lambda cfg: _rq2("run_rq2_score")(cfg),
    "rq2_analysis": lambda cfg: _rq2("run_rq2_analysis")(cfg),
    "rq2_smoke_analysis": lambda cfg: _rq2("run_rq2_analysis")(cfg),
    # τ-bench retail replication (notes/tau_design.md)
    "tau_build": lambda cfg: _lazy("tau.pipeline", "run_tau_build")(cfg),
    "tau_verify": lambda cfg: _lazy("tau.pipeline", "run_tau_verify")(cfg),
    "tau_score": lambda cfg: _lazy("tau.pipeline", "run_tau_score")(cfg),
    "tau_score_smoke": lambda cfg: _lazy("tau.pipeline", "run_tau_score")(cfg),
    "tau_analysis": lambda cfg: _lazy("tau.pipeline", "run_tau_analysis")(cfg),
    "tau_smoke_analysis": lambda cfg: _lazy("tau.pipeline", "run_tau_analysis")(cfg),
    "tau_contamination": lambda cfg: _lazy("tau.pipeline", "run_tau_contamination")(cfg),
    # τ-bench: relevance bias or a belief in request-matching authorization? (notes/tau_auth_design.md)
    "tau_auth_score": lambda cfg: _lazy("tau.auth", "run_tau_auth_score")(cfg),
    "tau_auth_score_smoke": lambda cfg: _lazy("tau.auth", "run_tau_auth_score")(cfg),
    "tau_auth_analysis": lambda cfg: _lazy("tau.auth", "run_tau_auth_analysis")(cfg),
    "tau_auth_smoke_analysis": lambda cfg: _lazy("tau.auth", "run_tau_auth_analysis")(cfg),
    # its extension to Llama-3.2-3B (design note, "Extension 2026-11-25"): the same runners, new run ids
    "tau_auth_l32_score": lambda cfg: _lazy("tau.auth", "run_tau_auth_score")(cfg),
    "tau_auth_l32_score_smoke": lambda cfg: _lazy("tau.auth", "run_tau_auth_score")(cfg),
    "tau_auth_l32_analysis": lambda cfg: _lazy("tau.auth", "run_tau_auth_analysis")(cfg),
    "tau_auth_l32_smoke_analysis": lambda cfg: _lazy("tau.auth", "run_tau_auth_analysis")(cfg),
    # 2026-11-20, documentation audit: the shared-spot sensitivity (A2), the audit analyses (C) and re-runs of
    # analyses under new run ids, so the original runs stay as they were (A3, B1, B4).
    "rq2_shared_score": lambda cfg: _lazy("rq2_shared", "run_rq2_shared_score")(cfg),
    "rq2_shared_analysis": lambda cfg: _lazy("rq2_shared", "run_rq2_shared_analysis")(cfg),
    "audit_analysis": lambda cfg: _lazy("audit", "run_audit_analysis")(cfg),
    "rq1_analysis_v2": lambda cfg: _rq1("run_rq1_analysis")(cfg),
    "rq1x_analysis_v2": lambda cfg: _rq1x("run_rq1x_analysis")(cfg),
    "rq1v_analysis_v2": lambda cfg: _rq1v("run_rq1v_analysis")(cfg),
    "thinking_analysis_v2": lambda cfg: _thinking("run_thinking_analysis")(cfg),
    "size_analysis_v2": lambda cfg: _lazy("sizes", "run_size_analysis")(cfg),
    # 2026-11-20, model families wave A (wf/families.py, notes/families_design.md): one scoring run per model.
    **{f"family_score_{tag}": (lambda cfg: _lazy("families", "run_family_score")(cfg))
       for tag in ("gemma2-9b-it", "mistral-7b-v0.3", "llama3.2-1b", "llama3.2-3b")},
    "families_analysis": lambda cfg: _lazy("families", "run_families_analysis")(cfg),
    # 2026-11-20, logged before any wave A score was read: Mistral's system text in the first user turn.
    "family_placement_mistral-7b-v0.3": lambda cfg: _lazy("families", "run_family_placement")(cfg),
}


def _rq2(name: str) -> Callable[[dict], int]:
    from wf import rq2  # wf/rq2.py imports this module, so it loads on first use

    return getattr(rq2, name)


def _rq1(name: str) -> Callable[[dict], int]:
    from wf import rq1  # wf/rq1.py imports this module, so it loads on first use

    return getattr(rq1, name)


def _rq1x(name: str) -> Callable[[dict], int]:
    from wf import rq1x  # wf/rq1x.py imports this module, so it loads on first use

    return getattr(rq1x, name)


def _lazy(module: str, name: str) -> Callable[[dict], int]:
    """A run function from a module that imports this one, loaded on first use."""
    import importlib

    return getattr(importlib.import_module(f"wf.{module}"), name)


def _rq1v(name: str) -> Callable[[dict], int]:
    from wf import rq1v  # wf/rq1v.py imports this module, so it loads on first use

    return getattr(rq1v, name)


def _rq1d(name: str) -> Callable[[dict], int]:
    from wf import rq1d  # wf/rq1d.py imports this module, so it loads on first use

    return getattr(rq1d, name)


def _thinking(name: str) -> Callable[[dict], int]:
    from wf import thinking  # wf/thinking.py imports this module, so it loads on first use

    return getattr(thinking, name)


def _rq3(name: str) -> Callable[[dict], int]:
    from wf import rq3  # wf/rq3.py imports this module, so it loads on first use

    return getattr(rq3, name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Wishful Foresight run dispatcher")
    parser.add_argument("run_id", choices=sorted(RUNNERS.keys()))
    parser.add_argument("--mock", action="store_true", help="Force WF_MOCK_*=1")
    parser.add_argument("--resume", action="store_true", help="Keep saved episodes/transitions and skip finished work")
    args = parser.parse_args(argv)
    if args.mock:
        os.environ["WF_MOCK_ENV"] = "1"
        os.environ["WF_MOCK_AGENT"] = "1"
        os.environ["WF_MOCK_SCORE"] = "1"
    cfg = load_config(args.run_id)
    cfg["resume"] = args.resume
    if args.resume and _already_passed(args.run_id):
        print(f"{args.run_id}: already passed at this commit, skipping", flush=True)
        sys.exit(0)
    code = RUNNERS[args.run_id](cfg)
    sys.exit(code)


if __name__ == "__main__":
    main()
