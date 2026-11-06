"""Level 2 verification: this repo's code against the real AgentGym server. No LLM, no GPU.

Answers what the simulator cannot: does the real server behave as sim.py assumes, do scene
scans and success twins work on real games, is context_determined sound, and what does a
step really cost. Writes verify_real_env.json to RUNS_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.corruption.schedule import schedule
from wf.data.goals import parse_task_line
from wf.data.transitions import (
    _failure_determined,
    build_transitions_from_rollout,
    exclusion_counts,
    filter_pilot_a_items,
    history_facts,
)
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.scene import scan_scene
from wf.env.server import AlfworldClient
from wf.scoring.parser import parse_observation
from wf.util import runs_root, write_json

NOOP = "Nothing happens."
FLUFF = ("help", "look", "inventory")
HOSTILE = ["", " ", "🦄🦄", "a" * 2000, "take", "go to", 'x"; DROP', "take mug 1 from nowhere 1", "\n\n", "go to drawer 1\nlook"]


class Solver(AgentClient):
    """Scripted plan through the real scene: reaches a win without a model, for place tasks."""

    def __init__(self, plan: list[str]):
        super().__init__()
        self.mock = False
        self.plan = plan

    def complete(self, messages, max_tokens=256):
        done = messages[-1]["content"].count("Action: ")
        return f"Action: {self.plan[min(done, len(self.plan) - 1)]}"


def _plan_to_win(scene, task_line: str) -> list[str] | None:
    spec = parse_task_line(task_line)
    if any(word in task_line for word in ("clean", "heat", "hot", "cool", "two", "look at", "examine")):
        return None
    instances = scene.instances(spec.goal_object)
    targets = [r for r in scene.receptacles if r.rsplit(" ", 1)[0] == spec.receptacle]
    if not instances or not targets:
        return None
    obj, target = instances[0], targets[0]
    home = scene.locations[obj]
    plan = [f"go to {home}"] + ([f"open {home}"] if home in scene.closed else []) + [f"take {obj} from {home}"]
    if target != home:
        plan += [f"go to {target}"] + ([f"open {target}"] if target in scene.closed else [])
    return plan + [f"move {obj} to {target}"]


def _hidden_objects(context: str) -> dict[str, str]:
    """object instance -> receptacle, from the dataset's hidden-state block."""
    found: dict[str, str] = {}
    block = context.split("=== Toggle States ===")[0].split("# User Environment")[0]
    for line in block.splitlines():
        m = re.match(r"\s*- On the ([a-z]+ \d+), you see (.+)", line)
        m2 = re.match(r"\s*- ([a-z]+ \d+) is closed, if opened, (.+)", line)
        hit = (m[1], m[2]) if m else (m2[1], m2[2]) if m2 else None
        if hit:
            for obj in re.findall(r"\ban? ([a-z]+ \d+)", hit[1]):
                found[obj] = hit[0]
    return found


def _server_rss_mb() -> float | None:
    """Resident memory of the ALFWorld server itself, to tell a leak from file cache."""
    pid_file = os.environ.get("ALFWORLD_PID_FILE")
    try:
        status = Path(f"/proc/{Path(pid_file).read_text().strip()}/status").read_text()
    except (OSError, TypeError):
        return None
    return int(re.search(r"VmRSS:\s+(\d+)", status)[1]) / 1024


def check_memory(client: AlfworldClient, game: str, resets: int) -> list[list[float]]:
    """Server memory against reset count: flat is healthy, a steady climb is a leak."""
    samples = [[0, _server_rss_mb()]]
    for i in range(1, resets + 1):
        client.reset(game, "train")
        if i % max(resets // 8, 1) == 0:
            samples.append([i, _server_rss_mb()])
    return samples


def check_memory_recreate(client: AlfworldClient, game: str, cycles: int) -> list[list[float]]:
    """Does /close plus /create release what /reset leaks? Decides how long runs stay alive."""
    samples = [[0, _server_rss_mb()]]
    for i in range(1, cycles + 1):
        client.reset(game, "train")
        client.close()
        client.create()
        if i % max(cycles // 8, 1) == 0:
            samples.append([i, _server_rss_mb()])
    return samples


def check_contract(client: AlfworldClient, games: list[str]) -> dict:
    """Are the server semantics the same ones sim.py assumes?"""
    gid = games[0]
    a = client.reset(gid, "train")
    side = client.sandbox()
    b = side.reset(gid, "train")
    first = a.admissible[0]
    moved = side.step(b, first)
    # the sandbox has moved; the live env must still act from its own start state
    live_adm = client.step(a, first)
    independent = live_adm.observation == moved.observation and moved.observation.strip() != NOOP
    admissible_valid = invalid_noop = adm_checked = inv_checked = 0
    noop_keeps_state = noop_checked = 0
    repeated_noop = repeated = 0
    for g in games[:8]:
        state = client.reset(g, "train")
        for action in sorted(a for a in state.admissible if a not in FLUFF)[:6]:
            adm_checked += 1
            obs, _ = step_counterfactual(replay(client, g, [], "train"), action)
            admissible_valid += obs.strip() != NOOP
            repeat_handle = replay(client, g, [action], "train")
            if not repeat_handle.state.done:
                repeated += 1
                repeated_noop += step_counterfactual(repeat_handle, action)[0].strip() == NOOP
        inv_checked += 1
        before = replay(client, g, [], "train")
        adm_before = list(before.admissible)
        obs, info = step_counterfactual(before, "take mug 1 from nowhere 1")
        invalid_noop += obs.strip() == NOOP
        noop_checked += 1
        noop_keeps_state += info["admissible_commands"] == adm_before
    hostile: dict[str, str] = {}
    for text in HOSTILE:
        try:
            state = client.reset(games[0], "train")
            hostile[repr(text[:24])] = client.step(state, text).observation[:40]
        except Exception as exc:  # recorded, because the agent can emit any of these
            hostile[repr(text[:24])] = f"ERROR {type(exc).__name__}: {str(exc)[:90]}"
    return {
        "sandbox_independent": bool(independent),
        "admissible_actions_valid": [admissible_valid, adm_checked],
        "repeat_is_noop": [repeated_noop, repeated],
        "nonsense_is_noop": [invalid_noop, inv_checked],
        "noop_keeps_admissible": [noop_keeps_state, noop_checked],
        "hostile_actions": hostile,
    }


def check_games(client: AlfworldClient, games: list[str], contexts: list[str] | None) -> dict:
    objects_true = objects_found = location_right = 0
    missed: Counter = Counter()
    twins = Counter()
    twin_examples: list[str] = []
    determined_claims = determined_wrong = 0
    wrong_examples: list[str] = []
    reasons = Counter()
    unparseable: Counter = Counter()
    observed = Counter()
    replay_mismatch = 0
    transitions: list[dict] = []
    per_game: list[int] = []
    timings = {"scan_s": [], "reset_s": []}
    for gid in games:
        started = time.perf_counter()
        client.reset(gid, "train")
        timings["reset_s"].append(time.perf_counter() - started)
        started = time.perf_counter()
        scene = scan_scene(client, gid, "train")
        timings["scan_s"].append(time.perf_counter() - started)
        if contexts:
            truth = _hidden_objects(contexts[int(gid.split("/")[1])])
            objects_true += len(truth)
            objects_found += sum(o in scene.locations for o in truth)
            location_right += sum(scene.locations.get(o) == r for o, r in truth.items())
            missed.update(o.rsplit(" ", 1)[0] for o in truth if o not in scene.locations)
        rng = random.Random(gid)
        state = client.reset(gid, "train")
        actions: list[str] = []
        for _ in range(12):
            choices = sorted(a for a in state.admissible if a not in FLUFF)
            action = rng.choice(choices)
            hist = history_text_from_state(state)
            # soundness: any action the rule calls a determined failure must actually fail
            probes = rng.sample(choices, min(4, len(choices))) + [f"take {o} from {state_rec}"
                                                                 for o, state_rec in list(scene.locations.items())[:3]]
            for probe in probes:
                if _failure_determined(history_facts(hist), probe):
                    determined_claims += 1
                    obs, _ = step_counterfactual(replay(client, gid, actions, "train"), probe)
                    if obs.strip() != NOOP:
                        determined_wrong += 1
                        if len(wrong_examples) < 8:
                            wrong_examples.append(f"{gid}: {probe!r} -> {obs[:70]!r} after {actions[-3:]}")
            following = client.step(state, action)
            observed[parse_observation(following.observation)] += 1
            if parse_observation(following.observation) == "unparseable":
                unparseable[re.sub(r"\d+", "N", following.observation)[:70]] += 1
            if following.done:
                break
            actions.append(action)
            state = following
        replayed = replay(client, gid, actions, "train").state
        replay_mismatch += replayed.observation != state.observation or replayed.admissible != state.admissible
        built = build_transitions_from_rollout(client, gid, "train", actions, id_start=len(transitions), k=2, rng_seed=1)
        transitions += built
        per_game.append(len(filter_pilot_a_items(built)))
        for t in built:
            if t["true_obs"].strip() == NOOP:
                kind = t["action_type"]
                twins[(kind, "twin" if t["candidates"] else "no_twin")] += 1
                if t["candidates"] and len(twin_examples) < 6:
                    twin_examples.append(f"{t['action']} -> {t['candidates']['o_plus'][:60]}")
                reasons[_determination_reason(t)] += 1
    eligible = filter_pilot_a_items(transitions)
    return {
        "games": len(games),
        "scene_scan": {"objects_in_hidden_state": objects_true, "found": objects_found, "right_location": location_right,
                       "missed_types": missed.most_common(10)},
        "replay_mismatches": replay_mismatch,
        "context_determined_soundness": {"claims": determined_claims, "wrong": determined_wrong, "examples": wrong_examples},
        "twins_by_action": {f"{k[0]}/{k[1]}": v for k, v in sorted(twins.items())},
        "twin_examples": twin_examples,
        "determination_reasons": dict(reasons),
        "observed_classes": dict(observed),
        "unparseable": unparseable.most_common(10),
        "pilot_a": {
            "total": len(transitions), "eligible": len(eligible),
            "eligible_per_game": sum(per_game) / max(len(per_game), 1),
            "agnostic_targeted": sum(bool(t["history_goal_agnostic"] and t["targets_goal_object"]) for t in eligible),
            "exclusions": exclusion_counts(transitions),
            "error_types": dict(Counter(t["error_type"] for t in eligible)),
        },
        "timing_s": {k: sum(v) / max(len(v), 1) for k, v in timings.items()},
    }


def _determination_reason(t: dict) -> str:
    """Which single fact decides this failure, for the context_determined decision."""
    facts = history_facts(t["history_text"])
    if not t["context_determined"]:
        return "not_determined"
    m = re.fullmatch(r"(?:take|move|open|close|heat|cool|clean) .*?(?:from |to |with )?([a-z]+ \d+)", t["action"])
    receptacle = m[1] if m else None
    if receptacle and facts.location != receptacle:
        if facts.location is None:
            return "away_from_receptacle_never_moved"
        return "away_from_receptacle"
    if t["action"].startswith("take") and facts.holding:
        return "hands_full"
    if t["action"].startswith(("move", "heat", "cool", "clean")):
        return "not_holding_object"
    return "contents_or_open_state"


def check_winning(client: AlfworldClient, games: list[str], cap: int) -> dict:
    """The C1 crash and every foresight mode, on real wins with no model involved."""
    configs = {
        "none": lambda g: ForesightConfig("none"),
        "oracle_discarded": lambda g: ForesightConfig("oracle", retention="discarded"),
        "oracle_retained": lambda g: ForesightConfig("oracle", retention="retained"),
        "validity_only": lambda g: ForesightConfig("validity_only"),
        "corrupted_unfavorable": lambda g: ForesightConfig("corrupted", arm="unfavorable", retention="retained",
                                                           corrupt_fn=schedule(g, 1, "unfavorable", 0.25)),
    }
    wins: Counter = Counter()
    errors: list[str] = []
    timing: dict[str, list[float]] = {}
    tried = 0
    for gid in games:
        scene = scan_scene(client, gid, "train")
        task = client.reset(gid, "train").task_line
        plan = _plan_to_win(scene, task)
        if not plan or tried >= cap:
            continue
        tried += 1
        for name, make in configs.items():
            started = time.perf_counter()
            try:
                ep = run_episode(client, Solver(plan), gid, "train", "verify", make(gid), step_cap=len(plan) + 3)
                wins[name] += int(ep["success"])
                timing.setdefault(name, []).append((time.perf_counter() - started) / max(ep["steps"], 1))
            except Exception as exc:
                errors.append(f"{gid} {name}: {type(exc).__name__}: {str(exc)[:120]}")
    return {"games_with_plan": tried, "wins": dict(wins), "errors": errors,
            "seconds_per_step": {k: sum(v) / len(v) for k, v in timing.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--win-games", type=int, default=20)
    parser.add_argument("--memory-resets", type=int, default=400)
    parser.add_argument("--only", choices=["memory_recreate"], default=None)
    args = parser.parse_args()
    client = AlfworldClient()
    client.create()
    games = list(client.list_games("train"))
    random.Random(20260924).shuffle(games)
    if args.only == "memory_recreate":
        samples = check_memory_recreate(client, games[0], args.memory_resets)
        write_json(runs_root() / "verify_memory_recreate.json", samples)
        print("memory_recreate", samples, flush=True)
        return
    games = games[: max(args.games, args.win_games * 6)]
    context_file = Path(os.environ.get("WF_INIT_CONTEXTS", Path(os.environ["WORD2WORLD_ROOT"]) / "data/init_contexts/alfworld")) / "wm_instruct_train.json"
    contexts = [r["messages"][0]["content"] for r in json.loads(context_file.read_text())]
    out = {"solvable_train_games": len(client.list_games("train"))}
    path = runs_root() / "verify_real_env.json"
    checks = {
        "memory_before": lambda: check_memory(client, games[0], args.memory_resets),
        "contract": lambda: check_contract(client, games),
        "winning": lambda: check_winning(client, games, args.win_games),
        "games": lambda: check_games(client, games[: args.games], contexts),
        "memory_after": lambda: check_memory(client, games[0], args.memory_resets),
    }
    for name, check in checks.items():  # saved after each check, so a failure keeps the rest
        started = time.perf_counter()
        try:
            out[name] = check()
        except Exception as exc:
            out[name] = {"FAILED": f"{type(exc).__name__}: {exc}"}
        print(name, f"{time.perf_counter() - started:.0f}s", json.dumps(out[name], default=str)[:1500], flush=True)
        write_json(path, out)


if __name__ == "__main__":
    main()
