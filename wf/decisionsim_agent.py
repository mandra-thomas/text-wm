"""Stage 1b: verification of the agent's own ranked candidates (exploratory; design in
notes/decisionsim_agent_design.md, committed before any data of this stage).

At every decision point of the 134 RQ3 no-foresight episodes (2,973 states), the Qwen3-8B ReAct agent's own candidate
list is frozen: rank 1 is the action the agent actually took (greedy), followed by up to K_MAX - 1 distinct actions
from N_SAMPLES samples of its reply at temperature 1, ranked by how often they were sampled (ties by a seeded hash).
Each candidate's true outcome comes from a fresh replay on the real server ("Nothing happens." is a failure), with the
RQ2 labels (history-determined, acts on the task's object, shared spot). A yes/no world-model verifier scores every
candidate with the task line shown (aware), deleted (blind) or swapped to another object (attribution). The agent acts
on its highest-ranked candidate the verifier accepts (rounded L > 0), or on rank 1 if it accepts none. Two
configurations need no model: none (rank 1, the real agent) and the simulator oracle.

Runs:
  decisionsim_agent_sample           GPU: vLLM agent + ALFWorld server; samples and freezes the ranked lists
  decisionsim_agent_label            CPU: ALFWorld server; one fresh replay per candidate, labels, swap object
  decisionsim_agent_score_<model>    GPU: yes/no scores, every candidate under every arm, one job per shard
  decisionsim_agent_analysis         CPU: decisions, Delta_invalid, encounter rates, attribution
  decisionsim_agent_dryrun           CPU, no model and no server: the archived episodes, rank 1 only
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import requests

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, pair_permutation
from wf.data.goals import interacted_objects, make_swapped_task_line, names_object, parse_task_line, swap_feasible, swapped_goal
from wf.data.transitions import (
    _receptacle,
    action_type,
    at_shared_spot,
    context_determined_label,
    determination_reason,
    history_facts,
)
from wf.env.replay import history_text_from_state, replay, step_counterfactual
from wf.env.scene import Scene, scan_scene
from wf.runs import _client, _init_run
from wf.scoring.prompts import agent_react_messages, parse_react_action
from wf.util import append_jsonl, read_jsonl, repair_jsonl, require_filed, runs_root, write_json, write_status

NOTHING = "Nothing happens."
SEED = C.RNG_SEED
# Candidate lists (design note, "Candidate generation and ranking").
N_SAMPLES = 20
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 256  # as the RQ3 agent
K_MAX = 8  # rank 1 (the greedy action) and at most K_MAX - 1 sampled actions
# Verifier arms: only the query's task line differs.
AWARE, BLIND, SWAPPED = "yn_aligned", "yn_blind", "yn_swapped"
ARMS = (AWARE, BLIND, SWAPPED)
CONFIG_ARM = {"aware": AWARE, "blind": BLIND, "swapped": SWAPPED}
CONFIGS = ("none", "aware", "blind", "swapped", "oracle")
MODEL_TAGS = ("llama3.1-8b", "mistral-7b-v0.3", "qwen3-8b")
PRIORITY = ("llama3.1-8b", "mistral-7b-v0.3")  # Holm-corrected primary tests; Qwen3-8B is a contrast
GRID = 64  # scores are rounded to 1/64 nat before any comparison
ALPHA = 0.05
# Competence rule of wave A, on this stage's candidates under the task-blind prompt.
COMPETENCE = {"auc_min": 0.70, "yes_failures_max": 0.95, "yes_successes_min": 0.05}
STEP_BANDS = (("step_0", 0, 0), ("steps_1_11", 1, 11), ("steps_12_29", 12, 10**6))
EARLY_STEPS = 12  # sensitivity: the first 12 steps, as in Stage 1a's collection
SHARD_COMPLETE = "shard complete"
TASK_PREFIX = "Your task is to:"


# ---------------- sources ----------------


def resolve_run(name: str) -> Path:
    """A run folder: an absolute path as given, otherwise a run under this checkout's runs/."""
    p = Path(os.path.expandvars(name)).expanduser()
    return p if p.is_absolute() else runs_root() / name


def episode_files(source: str) -> list[Path]:
    root = resolve_run(source)
    files = sorted(root.glob("shard_*/episodes.jsonl")) or sorted(root.glob("episodes.jsonl"))
    if not files:
        raise SystemExit(f"no episodes under {root}")
    return files


def load_none_episodes(source: str) -> dict[str, dict]:
    """The condition "none" episodes of an RQ3 run, one per game."""
    out: dict[str, dict] = {}
    for f in episode_files(source):
        for ep in read_jsonl(f):
            if ep.get("condition") != "none":
                continue
            if ep["game_id"] in out:
                raise SystemExit(f"two 'none' episodes for {ep['game_id']} in {source}")
            out[ep["game_id"]] = ep
    if not out:
        raise SystemExit(f"no condition 'none' episodes in {source}")
    return out


def shard_spec(cfg: dict) -> tuple[int, int]:
    return int(os.environ.get("WF_SHARD") or cfg.get("shard", 0)), int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))


def shard_games(games: list[str] | set[str], i: int, n: int) -> list[str]:
    """Games dealt to shards by their position in the sorted list: independent of any outcome."""
    return sorted(games)[i::n]


def complete_shards(source: str) -> tuple[list[Path], list[str]]:
    """A sharded run's folders, and why a later step must wait (missing, unfinished, mixed commits or settings)."""
    root = resolve_run(source)
    folders = sorted(p for p in root.glob("shard_*_of_*") if p.is_dir())
    problems: list[str] = []
    counts = {p.name.rsplit("_of_", 1)[1] for p in folders}
    if not folders or len(counts) != 1:
        return folders, [f"{source}: shards missing or from different splits: {[p.name for p in folders]}"]
    n = int(counts.pop())
    if sorted(int(p.name.split("_")[1]) for p in folders) != list(range(n)):
        problems.append(f"{source}: expected shards 0..{n - 1}, have {[p.name for p in folders]}")
    configs, shas = set(), set()
    for p in folders:
        first = (p / "STATUS.md").read_text(encoding="utf-8").splitlines()[0] if (p / "STATUS.md").exists() else ""
        if f"REPORTED ({SHARD_COMPLETE})" not in first:
            problems.append(f"{source}/{p.name} not complete: {first or 'no status'}")
        shas.add((p / "git_sha").read_text(encoding="utf-8").strip() if (p / "git_sha").exists() else "missing")
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8")) if (p / "config.json").exists() else {}
        configs.add(json.dumps({k: v for k, v in cfg.items() if k not in ("run_id", "shard")}, sort_keys=True))
    if len(shas) != 1:
        problems.append(f"{source}: shards ran at different commits: {sorted(shas)}")
    if len(configs) != 1:
        problems.append(f"{source}: shards ran with different settings")
    return folders, problems


def _fresh(d: Path, names: tuple[str, ...], resume: bool) -> None:
    """This module's output files: repaired on a resume, removed on a fresh run (never appended to stale work)."""
    for name in names:
        p = d / name
        if resume:
            repair_jsonl(p)
        elif p.exists():
            p.unlink()


# ---------------- candidate lists ----------------


def sample_seed(game: str, t: int, seed: int = SEED) -> int:
    return int(hashlib.sha256(f"{seed}|decisionsim_agent|{game}|{t}".encode()).hexdigest()[:8], 16)


def tie_key(game: str, t: int, action: str, seed: int = SEED) -> str:
    """Seeded per-state hash of an action (Stage 1a's tie rule): blind to outcomes."""
    return hashlib.sha256(f"{seed}|{game}|{t}|{action}".encode()).hexdigest()


def normalize_action(action: str | None) -> str | None:
    """Whitespace collapsed, lower case, no trailing full stop; None if nothing is left."""
    if action is None:
        return None
    a = " ".join(action.split()).lower().rstrip(".").strip()
    return a or None


def rank_candidates(greedy: str, replies: list[str], game: str, t: int, k_max: int = K_MAX,
                    seed: int = SEED) -> dict[str, Any]:
    """The frozen list: rank 1 the greedy action (its recorded string, the one the agent executed), then the other
    distinct sampled actions by count, ties by the seeded hash, at most k_max in all. A reply without an Action line
    is a format error and contributes no candidate."""
    parsed = [normalize_action(parse_react_action(r)) for r in replies]
    n_format = sum(p is None for p in parsed)
    counts = Counter(p for p in parsed if p is not None)
    greedy_count = counts.pop(normalize_action(greedy), 0)
    rest = sorted(counts.items(), key=lambda kv: (-kv[1], tie_key(game, t, kv[0], seed)))
    kept, cut = rest[: k_max - 1], rest[k_max - 1:]
    cands = [{"rank": 1, "action": greedy, "count": greedy_count, "greedy": True}]
    cands += [{"rank": r + 2, "action": a, "count": c, "greedy": False} for r, (a, c) in enumerate(kept)]
    return {"candidates": cands, "n_samples": len(replies), "n_format_errors": n_format,
            "n_distinct_sampled": len(counts) + (greedy_count > 0), "n_cut": len(cut),
            "cut_counts": [c for _, c in cut]}


class AgentSampler:
    """n completions of the RQ3 agent's reply from the vLLM server (thinking off); a deterministic stand-in when
    WF_MOCK_AGENT=1."""

    def __init__(self, base_url: str | None = None, model: str = C.AGENT_MODEL):
        self.base_url = (os.environ.get("VLLM_BASE_URL") or base_url or C.VLLM_BASE_URL_DEFAULT).rstrip("/")
        self.model = model
        self.mock = os.environ.get("WF_MOCK_AGENT", "0") == "1"
        self._session = requests.Session()

    def sample(self, messages: list[dict[str, str]], n: int, temperature: float, top_p: float, max_tokens: int,
               seed: int) -> list[dict[str, str]]:
        if self.mock:
            return self._mock(messages, n, seed)
        payload = {"model": self.model, "messages": messages, "n": n, "temperature": temperature, "top_p": top_p,
                   "top_k": -1, "max_tokens": max_tokens, "seed": seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        r = self._session.post(f"{self.base_url}/chat/completions", json=payload, timeout=600)
        r.raise_for_status()
        choices = r.json()["choices"]
        if len(choices) != n:
            raise RuntimeError(f"asked for {n} completions, got {len(choices)}")
        return [{"text": c["message"]["content"] or "", "finish_reason": c.get("finish_reason")} for c in choices]

    @staticmethod
    def _mock(messages: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
        text = messages[-1]["content"]
        body, _, tail = text.partition("Admissible actions:")
        adm = [ln.strip()[2:] for ln in tail.splitlines() if ln.strip().startswith("- ")]
        task = re.search(r"Your task is to:.*", "\n".join(m["content"] for m in messages[2:]))
        goal = parse_task_line(task.group(0)).goal_object if task else "mug"
        out = []
        for i in range(n):
            h = int(hashlib.sha256(f"{seed}|{i}|{body}".encode()).hexdigest()[:8], 16)
            k = h % 20
            if k == 0:
                reply = "Thought: hmm, let me think."  # format error
            elif k < 5:
                reply = f"Thought: grab it\nAction: take {goal} 1 from countertop 1"  # a free-form goal-object action
            elif k < 7:
                reply = "Thought: check\nAction: Go to Drawer 1."  # normalisation
            else:
                reply = f"Thought: ok\nAction: {adm[h % min(4, len(adm))] if adm else 'look'}"
            out.append({"text": reply, "finish_reason": "stop"})
        return out


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


SAMPLE_OUTPUTS = ("states.jsonl", "mismatches.jsonl", "games_done.jsonl")


def run_decisionsim_agent_sample(cfg: dict) -> int:
    base = cfg["run_id"]
    require_filed(base)
    i, n = shard_spec(cfg)
    run_id = f"{base}/shard_{i:02d}_of_{n:02d}"
    episodes = load_none_episodes(cfg["episodes_source"])
    games = shard_games(episodes, i, n)
    if cfg.get("max_games"):
        games = games[: int(cfg["max_games"])]
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(episode_files(cfg["episodes_source"])))
    _fresh(d, SAMPLE_OUTPUTS, bool(cfg.get("resume")))
    write_json(d / "shard.json", {"shard": i, "n_shards": n, "games": games, "all_games": sorted(episodes)})
    n_samples, temp = int(cfg.get("n_samples", N_SAMPLES)), float(cfg.get("temperature", TEMPERATURE))
    top_p, max_tokens = float(cfg.get("top_p", TOP_P)), int(cfg.get("max_tokens", MAX_TOKENS))
    k_max, seed = int(cfg.get("k_max", K_MAX)), int(cfg.get("seed", SEED))
    saved = {(r["game_id"], r["state_idx"]): r for r in read_jsonl(d / "states.jsonl")}
    done = {r["game_id"] for r in read_jsonl(d / "games_done.jsonl")}
    client = _client(cfg)
    client.create()
    sampler = AgentSampler(base_url=cfg.get("vllm_base_url"))
    for g in games:
        if g in done:
            continue
        ep = episodes[g]
        split = ep.get("split") or cfg.get("split", "valid_unseen")
        state = client.reset(g, split)
        history: list[dict[str, str]] = []
        prefix: list[str] = []
        for step in ep["trace"]:
            t = int(step["t"])
            msgs = agent_react_messages(observation=state.observation, history=history, admissible=state.admissible)
            psha = _sha(msgs)
            if (g, t) in saved:
                if saved[(g, t)]["prompt_sha"] != psha:
                    raise SystemExit(f"{run_id}: {g} step {t}: the saved list was sampled on another prompt; "
                                     f"rerun without --resume")
            else:
                replies = sampler.sample(msgs, n_samples, temp, top_p, max_tokens, sample_seed(g, t, seed))
                ranked = rank_candidates(step["executed"], [r["text"] for r in replies], g, t, k_max, seed)
                rec = {"game_id": g, "split": split, "state_idx": t, "task_line": state.task_line,
                       "prefix_actions": list(prefix), "greedy": step["executed"], "recorded_obs": step["real_obs"],
                       "admissible": list(state.admissible), "prompt_sha": psha,
                       "sample_seed": sample_seed(g, t, seed), "replies": replies, **ranked}
                append_jsonl(d / "states.jsonl", [rec])  # saved as it finishes
                saved[(g, t)] = rec
            new = client.step(state, step["executed"])
            if new.observation != step["real_obs"]:
                # The states after this one are not the episode's; they are never sampled.
                append_jsonl(d / "mismatches.jsonl", [{"game_id": g, "state_idx": t, "action": step["executed"],
                                                       "recorded": step["real_obs"], "replayed": new.observation}])
                break
            history.append({"observation": state.observation, "action": step["executed"]})
            prefix.append(step["executed"])
            state = new
        append_jsonl(d / "games_done.jsonl", [{"game_id": g}])
        done.add(g)
    mine = [r for r in saved.values() if r["game_id"] in set(games)]
    book = sampling_bookkeeping(mine)
    book["replay_mismatches"] = len(read_jsonl(d / "mismatches.jsonl"))
    write_json(d / "metrics.json", book)
    write_status(run_id, f"shard {i} of {n}: {len(mine)} states in {len(games)} games\n{book}", ok=False,
                 reported=SHARD_COMPLETE)
    return 0


def sampling_bookkeeping(states: list[dict]) -> dict[str, Any]:
    n_samples = sum(s["n_samples"] for s in states)
    return {"n_states": len(states), "n_samples": n_samples,
            "format_error_share": sum(s["n_format_errors"] for s in states) / n_samples if n_samples else float("nan"),
            "mean_list_length": _mean([len(s["candidates"]) for s in states]),
            "list_length_counts": dict(sorted(Counter(len(s["candidates"]) for s in states).items())),
            "states_with_cut": sum(s["n_cut"] > 0 for s in states),
            "greedy_never_sampled": sum(s["candidates"][0]["count"] == 0 for s in states),
            "truncated_replies": sum(r.get("finish_reason") == "length" for s in states for r in s.get("replies", [])),
            # a seeding fault would return n copies of one reply; at temperature 1 that should almost never happen
            "states_all_replies_identical": sum(len({r["text"] for r in s.get("replies", [])}) == 1 < len(s.get("replies", []))
                                                for s in states)}


# ---------------- labels ----------------


def progress_event(action: str, obs: str | None, goal: str, target: str, history_text: str, won: bool) -> bool:
    """A descriptive task-progress event read off the real reply (design note, "Progress")."""
    if obs is None or obs.strip() == NOTHING:
        return False
    if won:
        return True
    kind = lambda inst: inst.rsplit(" ", 1)[0]
    m = re.fullmatch(r"take (.+) from (.+)", action)
    if m and kind(m[1]) == goal and obs.startswith("You pick up"):
        return True
    m = re.fullmatch(r"move (.+) to (.+)", action)
    if m and kind(m[1]) == goal and kind(m[2]) == target and obs.startswith("You move"):
        return True
    m = re.fullmatch(r"(heat|cool|clean) (.+) with (.+)", action)
    if m and kind(m[2]) == goal and obs.startswith(f"You {m[1]}"):
        return True
    if target == "desklamp" and action.startswith("use ") and obs.startswith("You turn on"):
        return True
    if "you see" not in obs.lower():
        return False
    seen = set(re.findall(rf"\b{re.escape(goal)} \d+\b", history_text))
    return bool(set(re.findall(rf"\b{re.escape(goal)} \d+\b", obs)) - seen)


def state_swap_object(scene: Scene, task_line: str, goal: str, actions: list[str], history_text: str, game: str,
                      t: int, seed: int = SEED) -> str | None:
    """One swap object for the state: feasible for the task by the game's flags, not named by any candidate, not
    acted on in the history, with no instance at any candidate's receptacle at reset; None if none qualifies."""
    named = {o for o in scene.objects if any(names_object(a, o) for a in actions)}
    receptacles = {r for r in (_receptacle(a) for a in actions) if r}
    there = {inst.rsplit(" ", 1)[0] for inst, places in scene.locations.items() if places & receptacles}
    return swapped_goal(scene.objects, goal, None, rng=random.Random(f"{seed}|decisionsim_agent|{game}|{t}"),
                        locations=scene.locations, exclude=named | interacted_objects(history_text) | there,
                        feasible=lambda o: swap_feasible(task_line, o, scene.traits, len(scene.instances(o))))


def label_state(client: Any, scene: Scene, s: dict, seed: int = SEED) -> tuple[dict, int]:
    """Every candidate executed on a fresh replay of the state; (labelled state, replays used)."""
    g, t, split = s["game_id"], int(s["state_idx"]), s.get("split", "valid_unseen")
    spec = parse_task_line(s["task_line"])
    goal, target = spec.goal_object, spec.receptacle
    hist: str | None = None
    adm: list[str] = []
    out = []
    for c in s["candidates"]:
        h = replay(client, g, s["prefix_actions"], split=split)
        here = history_text_from_state(h.state)
        if hist is None:
            hist, adm = here, list(h.state.admissible)
        elif here != hist:
            raise RuntimeError(f"{g} step {t}: two replays of one state gave different histories")
        a = c["action"]
        rec = {**c, "true_obs": None, "server_error": None}
        try:
            obs, info = step_counterfactual(h, a)
        except RuntimeError as e:
            rec["server_error"] = str(e)[:300]
            out.append(rec)
            continue
        failed = obs.strip() == NOTHING
        det = context_determined_label(hist, a, obs)
        rec.update({"true_obs": obs, "failed": failed, "context_determined": det,
                    "targets_goal_object": names_object(a, goal),
                    "shared_location": at_shared_spot(history_facts(hist), a, scene),
                    "admissible": a in adm, "action_type": action_type(a), "won": bool(info.get("won")),
                    "progress": progress_event(a, obs, goal, target, hist, bool(info.get("won")))})
        rec["determination_reason"] = (determination_reason({"history_text": hist, "context_determined": det,
                                                             "action": a}) if failed else None)
        out.append(rec)
    assert hist is not None
    swap = state_swap_object(scene, s["task_line"], goal, [c["action"] for c in out], hist, g, t, seed)
    rank1 = out[0]
    rank1_ok = rank1["server_error"] is None and rank1["true_obs"] == s["recorded_obs"]
    lab = {"game_id": g, "split": split, "state_idx": t, "task_line": s["task_line"], "goal_object": goal,
           "target": target, "history_text": hist, "history_goal_agnostic": goal not in interacted_objects(hist),
           "prefix_actions": s["prefix_actions"], "swap": swap,
           "swapped_task_line": make_swapped_task_line(s["task_line"], goal, swap) if swap else None,
           "rank1_replay_ok": rank1_ok, "n_format_errors": s["n_format_errors"], "n_samples": s["n_samples"],
           "n_cut": s["n_cut"], "candidates": out}
    return lab, len(out)


LABEL_OUTPUTS = ("labels.jsonl",)


def _sample_states(source: str) -> tuple[list[dict], list[Path], list[str]]:
    folders, problems = complete_shards(source)
    states = [r for p in folders for r in read_jsonl(p / "states.jsonl")] if not problems else []
    return states, folders, problems


def run_decisionsim_agent_label(cfg: dict) -> int:
    base = cfg["run_id"]
    require_filed(base)
    i, n = shard_spec(cfg)
    run_id = f"{base}/shard_{i:02d}_of_{n:02d}"
    states, folders, problems = _sample_states(cfg["sample_source"])
    if problems:
        write_status(run_id, "labelling waits for every sampling shard:\n" + "\n".join(problems), ok=False)
        return 1
    games = shard_games({s["game_id"] for s in states}, i, n)
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(p / "states.jsonl" for p in folders))
    _fresh(d, LABEL_OUTPUTS, bool(cfg.get("resume")))
    write_json(d / "shard.json", {"shard": i, "n_shards": n, "games": games})
    seed = int(cfg.get("seed", SEED))
    saved = {(r["game_id"], r["state_idx"]) for r in read_jsonl(d / "labels.jsonl")}
    # The server leaks memory on every reset: stop after this many replays, restart it, resume.
    budget = int(cfg.get("max_replays_per_job") or 0)
    used = 0
    client = _client(cfg)
    client.create()
    by_game: dict[str, list[dict]] = defaultdict(list)
    for s in states:
        by_game[s["game_id"]].append(s)
    for g in games:
        todo = [s for s in sorted(by_game[g], key=lambda s: s["state_idx"]) if (g, s["state_idx"]) not in saved]
        if not todo:
            continue
        scene = scan_scene(client, g, todo[0].get("split", "valid_unseen"))
        for s in todo:
            if budget and used and used + len(s["candidates"]) > budget:
                done = len(saved)
                write_status(run_id, f"PARTIAL — {done} states labelled; resubmit with WF_RESUME=1", ok=False)
                return 1
            lab, k = label_state(client, scene, s, seed)
            used += k
            append_jsonl(d / "labels.jsonl", [lab])  # saved as it finishes
            saved.add((g, s["state_idx"]))
    labs = [r for r in read_jsonl(d / "labels.jsonl") if r["game_id"] in set(games)]
    book = {"n_states": len(labs), "rank1_replay_mismatch": sum(not r["rank1_replay_ok"] for r in labs),
            "server_errors": sum(c["server_error"] is not None for r in labs for c in r["candidates"]),
            "n_candidates": sum(len(r["candidates"]) for r in labs), "states_without_swap": sum(not r["swap"] for r in labs)}
    write_json(d / "metrics.json", book)
    write_status(run_id, f"shard {i} of {n}: {book}", ok=False, reported=SHARD_COMPLETE)
    return 0


# ---------------- verifier prompts and scoring ----------------


def remove_task_line(text: str) -> str:
    """The text without its one task line (and one newline next to it); every other byte unchanged."""
    lines = text.split("\n")
    idx = [k for k, ln in enumerate(lines) if ln.startswith(TASK_PREFIX)]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one task line, found {len(idx)}")
    del lines[idx[0]]
    return "\n".join(lines)


def blind_messages(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """The aware prompt with the query's task line deleted; the examples (earlier turns) are untouched."""
    last = msgs[-1]
    if last["role"] != "user":
        raise ValueError("the query must be the last, user turn")
    return [dict(m) for m in msgs[:-1]] + [{**last, "content": remove_task_line(last["content"])}]


def cand_id(state: dict, cand: dict) -> str:
    return f"{state['game_id']}#{int(state['state_idx']):02d}#{int(cand['rank'])}"


def prompt_item(state: dict, cand: dict) -> dict:
    """What the yes/no prompt builders read (wf/rq1v.py through wf/sizes.py or wf/families.py)."""
    return {"id": cand_id(state, cand), "game_id": state["game_id"], "state_idx": state["state_idx"],
            "task_line": state["task_line"], "goal_object": state["goal_object"],
            "history_text": state["history_text"], "action": cand["action"],
            "swapped_goal_object": state.get("swap"), "swapped_task_line": state.get("swapped_task_line")}


def _builder(tag: str) -> Callable[[dict, str], list[dict[str, str]] | None]:
    from wf.decisionsim import messages_for  # the yes/no prompts of the earlier runs, per model

    return messages_for(tag)


def arm_messages(tag: str, state: dict, cand: dict, arm: str) -> list[dict[str, str]] | None:
    build = _builder(tag)
    item = prompt_item(state, cand)
    if arm == AWARE:
        return build(item, AWARE)
    if arm == BLIND:
        return blind_messages(build(item, AWARE))
    if arm == SWAPPED:
        return build(item, SWAPPED) if state.get("swap") else None
    raise ValueError(arm)


def usable(state: dict) -> bool:
    """Rank 1 replayed to the episode's own reply (otherwise the state is not the agent's)."""
    return bool(state.get("rank1_replay_ok"))


def scored_candidates(state: dict) -> list[dict]:
    """The list in rank order, without candidates the server could not execute."""
    return [c for c in state["candidates"] if c.get("server_error") is None]


def score_jobs(states: list[dict], arms: tuple[str, ...]) -> list[tuple[dict, dict, str]]:
    return [(s, c, a) for s in states for c in scored_candidates(s) for a in arms if a != SWAPPED or s.get("swap")]


def load_labels(source: str) -> tuple[list[dict], list[Path], list[str]]:
    folders, problems = complete_shards(source)
    labs = [r for p in folders for r in read_jsonl(p / "labels.jsonl")] if not problems else []
    return sorted(labs, key=lambda r: (r["game_id"], r["state_idx"])), folders, problems


SCORE_OUTPUTS = ("scoring.jsonl",)


def run_decisionsim_agent_score(cfg: dict) -> int:
    from wf import families
    from wf.decisionsim import FAMILY_MODELS, open_scorer, yes_no_gate

    tag = cfg["model"]
    if tag not in MODEL_TAGS:
        raise SystemExit(f"unknown model {tag!r}; known: {MODEL_TAGS}")
    arms = tuple(cfg.get("arms", ARMS))
    if AWARE not in arms or BLIND not in arms or not set(arms) <= set(ARMS):
        raise SystemExit(f"arms must include {AWARE} and {BLIND}, from {ARMS}; got {arms}")
    base = cfg["run_id"]
    require_filed(base)
    i, n = shard_spec(cfg)
    run_id = f"{base}/shard_{i:02d}_of_{n:02d}"
    labs, folders, problems = load_labels(cfg["label_source"])
    if problems:
        write_status(run_id, "scoring waits for every labelling shard:\n" + "\n".join(problems), ok=False)
        return 1
    states = [s for s in labs if usable(s)]
    games = set(shard_games({s["game_id"] for s in states}, i, n))
    mine = [s for s in states if s["game_id"] in games]
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n, "arms": list(arms)}
    d = _init_run(run_id, cfg, inputs=tuple(p / "labels.jsonl" for p in folders))
    _fresh(d, SCORE_OUTPUTS, bool(cfg.get("resume")))
    jobs = score_jobs(mine, arms)
    prompts = {(cand_id(s, c), a): arm_messages(tag, s, c, a) for s, c, a in jobs}
    hashes = {k: _sha([m, "Yes", "No"]) for k, m in prompts.items()}
    saved = read_jsonl(d / "scoring.jsonl")
    if any(hashes.get((x["transition_id"], x["condition"])) != x.get("score_hash") for x in saved):
        raise SystemExit(f"{run_id}: saved scores belong to other prompts; rerun without --resume")
    done = {(x["transition_id"], x["condition"]) for x in saved}
    todo = [(s, c, a) for s, c, a in jobs if (cand_id(s, c), a) not in done]
    if todo:
        scorer = open_scorer(tag)
        if tag in FAMILY_MODELS and not scorer.mock:  # wave A's Yes/No gate, on every prompt, before any score
            gate = yes_no_gate(scorer, list(prompts.values()))
            write_json(d / "tokenization_checks.json", gate)
            if not gate["ok"]:
                write_status(run_id, f"stopped before scoring: Yes/No gate failed ({gate['n_problems']} problems)", ok=False)
                raise families.TokenizationMismatch(f"{tag}: Yes/No gate failed: {gate['problems'][:3]}")
        for s, c, a in todo:
            key = (cand_id(s, c), a)
            sc = scorer.score_one_token_pair(transition_id=key[0], condition=a, messages=prompts[key],
                                             o_plus="Yes", o_minus="No")
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": s["game_id"], "state_idx": s["state_idx"],
                                                "rank": c["rank"], "score_hash": hashes[key]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} yes/no scores at {len(mine)} states ({tag})", ok=False,
                 reported=SHARD_COMPLETE)
    return 0


# ---------------- decisions ----------------


def quantize(x: float) -> float:
    return round(x * GRID) / GRID


def accepts(L: dict[str, float], cid: str) -> bool:
    return L[cid] > 0


def regime(c: dict) -> bool:
    """A failure that acts on the task's object, is determined by the history and is not at a shared spot."""
    return bool(c.get("failed") and c.get("targets_goal_object") and c.get("context_determined")
                and not c.get("shared_location"))


def choose(config: str, state: dict, L: dict[str, dict[str, float]] | None = None, k: int | None = None,
           fallback: str = "rank1") -> dict:
    """The action the agent acts on: none, rank 1; oracle, the highest-ranked true success; a verifier, its
    highest-ranked accepted candidate, else rank 1 (or, as a sensitivity, the candidate with the largest L)."""
    cands = scored_candidates(state)[:k] if k else scored_candidates(state)
    if config == "none":
        return cands[0]
    if config == "oracle":
        return next((c for c in cands if not c["failed"]), cands[0])
    arm = CONFIG_ARM[config]
    sc = L[arm]
    ids = [cand_id(state, c) for c in cands]
    for c, cid in zip(cands, ids):
        if accepts(sc, cid):
            return c
    if fallback == "max_L":
        return max(zip(cands, ids), key=lambda ci: (sc[ci[1]], -ci[0]["rank"]))[0]
    return cands[0]


def fell_back(config: str, state: dict, L: dict[str, dict[str, float]]) -> bool:
    arm = CONFIG_ARM[config]
    return not any(accepts(L[arm], cand_id(state, c)) for c in scored_candidates(state))


def encounter(state: dict, strict: bool = False) -> bool:
    return any(regime(c) for c in scored_candidates(state)) and (not strict or bool(state.get("history_goal_agnostic")))


def reachable(state: dict) -> bool:
    """A regime candidate ranked above the first candidate that truly succeeds."""
    for c in scored_candidates(state):
        if not c["failed"]:
            return False
        if regime(c):
            return True
    return False


# ---------------- statistics ----------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _boot(values: list[float], games: list[str]) -> dict[str, Any]:
    if not values:
        return {"estimate": float("nan"), "ci": [float("nan"), float("nan")], "n_states": 0, "n_games": 0}
    b = cluster_bootstrap(_mean, values, games, seed=SEED)
    return {"estimate": b["estimate"], "ci": [b["ci_low"], b["ci_high"]], "n_states": len(values),
            "n_games": len(set(games))}


def _paired(a: list[float], b: list[float], games: list[str]) -> dict[str, Any]:
    diffs = [x - y for x, y in zip(a, b)]
    out = _boot(diffs, games)
    out["p"] = pair_permutation(diffs, seed=SEED, cluster_ids=games) if diffs else float("nan")
    return out


def holm(ps: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values; a missing or NaN p counts as 1."""
    items = sorted(((p if p == p else 1.0), k) for k, p in ps.items())
    m = len(items)
    out, running = {}, 0.0
    for j, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - j) * p))
        out[k] = running
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a success scores above a failure), ties counting one half."""
    if not pos or not neg:
        return float("nan")
    neg_sorted = sorted(neg)
    wins = sum(bisect.bisect_left(neg_sorted, p) + 0.5 * (bisect.bisect_right(neg_sorted, p) - bisect.bisect_left(neg_sorted, p))
               for p in pos)
    return wins / (len(pos) * len(neg))


def _kind_of_failure(c: dict, state: dict) -> str:
    if regime(c):
        return "regime"
    if c.get("targets_goal_object"):
        return "goal_object_other"
    return "go_to" if c.get("action_type") == "go to" else "other"


# ---------------- model-free readouts ----------------


def encounter_report(states: list[dict]) -> dict[str, Any]:
    games = [s["game_id"] for s in states]
    cands = [(s, c) for s in states for c in scored_candidates(s)]
    reg = [(s, c) for s, c in cands if regime(c)]
    bands = {}
    for name, lo, hi in STEP_BANDS:
        sel = [s for s in states if lo <= s["state_idx"] <= hi]
        bands[name] = _boot([float(encounter(s)) for s in sel], [s["game_id"] for s in sel])
    by_rank = defaultdict(lambda: [0, 0])
    for s, c in cands:
        by_rank[c["rank"]][0] += 1
        by_rank[c["rank"]][1] += regime(c)
    n_samples = sum(s.get("n_samples", 0) for s in states)
    return {
        "n_states": len(states), "n_games": len(set(games)), "n_candidates": len(cands),
        "mean_list_length": _mean([len(scored_candidates(s)) for s in states]),
        "encounter": _boot([float(encounter(s)) for s in states], games),
        "encounter_strict": _boot([float(encounter(s, strict=True)) for s in states], games),
        "reachable": _boot([float(reachable(s)) for s in states], games),
        "encounter_by_step_band": bands,
        "regime_candidates": {"n": len(reg), "free_form": sum(c.get("admissible") is False for _, c in reg),
                              "rank1": sum(c["rank"] == 1 for _, c in reg)},
        "regime_share_by_rank": {r: {"n": v[0], "regime": v[1], "share": v[1] / v[0]} for r, v in sorted(by_rank.items())},
        "candidates": {"failed": sum(c["failed"] for _, c in cands), "free_form": sum(c.get("admissible") is False for _, c in cands),
                       "free_form_failed": sum(c["failed"] and c.get("admissible") is False for _, c in cands),
                       "admissible_failed": sum(c["failed"] and c.get("admissible") is True for _, c in cands),
                       "goal_object_failures": sum(c["failed"] and bool(c["targets_goal_object"]) for _, c in cands),
                       "determined_failures": sum(c["failed"] and bool(c["context_determined"]) for _, c in cands)},
        "fail_rate": {cfg: _boot([float(choose(cfg, s)["failed"]) for s in states], games) for cfg in ("none", "oracle")},
        "progress_rate": {cfg: _boot([float(choose(cfg, s)["progress"]) for s in states], games) for cfg in ("none", "oracle")},
        "format_error_share": (sum(s.get("n_format_errors", 0) for s in states) / n_samples) if n_samples else float("nan"),
        "states_with_cut_list": sum(s.get("n_cut", 0) > 0 for s in states),
        "states_without_swap": sum(not s.get("swap") for s in states)}


# ---------------- one model ----------------


def scores_by_arm(records: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    q: dict[str, dict[str, float]] = defaultdict(dict)
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    for x in records:
        if x["L"] == x["L"]:
            raw[x["condition"]][x["transition_id"]] = x["L"]
            q[x["condition"]][x["transition_id"]] = quantize(x["L"])
    return dict(q), dict(raw)


def complete_states(states: list[dict], L: dict[str, dict[str, float]]) -> tuple[list[dict], int]:
    """States with a score for every candidate under aware, blind and (where a swap exists) swapped."""
    def ok(s: dict) -> bool:
        need = [AWARE, BLIND] + ([SWAPPED] if s.get("swap") and SWAPPED in L else [])
        return all(cand_id(s, c) in L.get(a, {}) for a in need for c in scored_candidates(s))
    keep = [s for s in states if ok(s)]
    return keep, len(states) - len(keep)


def contrast(states: list[dict], L: dict[str, dict[str, float]], a: str, b: str) -> dict[str, Any]:
    """Configuration a against b: failure rates, the paired difference, change, harm and help."""
    games = [s["game_id"] for s in states]
    ca = [choose(a, s, L) for s in states]
    cb = [choose(b, s, L) for s in states]
    fa = [float(c["failed"]) for c in ca]
    fb = [float(c["failed"]) for c in cb]
    harm = [float(x and not y) for x, y in zip(fa, fb)]
    help_ = [float(y and not x) for x, y in zip(fa, fb)]
    changed = [x["rank"] != y["rank"] for x, y in zip(ca, cb)]
    n = len(states)
    both_fail = sum(ch and x and y for ch, x, y in zip(changed, fa, fb))
    both_ok = sum(ch and not x and not y for ch, x, y in zip(changed, fa, fb))
    kinds = Counter(_kind_of_failure(c, s) for c, s, h in zip(ca, states, harm) if h)
    return {"n_states": n, "n_games": len(set(games)),
            "fail_rate_a": _boot(fa, games), "fail_rate_b": _boot(fb, games),
            "difference": _paired(fa, fb, games),
            "harm_rate": _boot(harm, games), "help_rate": _boot(help_, games),
            "change_rate": _boot([float(x) for x in changed], games),
            "changed": {"n": sum(changed), "harm": int(sum(harm)), "help": int(sum(help_)), "both_fail": both_fail,
                        "both_succeed": both_ok},
            "harm_by_kind_of_choice": dict(sorted(kinds.items())),
            "identity_holds": (not n) or abs((sum(harm) - sum(help_)) / n - (sum(fa) - sum(fb)) / n) < 1e-12}


def acceptance(states: list[dict], L: dict[str, dict[str, float]], Lraw: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Per candidate, per arm: acceptance by outcome x goal relevance x determination, the AUC, and aware-minus-blind
    shifts."""
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for s in states:
        for c in scored_candidates(s):
            out = "failure" if c["failed"] else "success"
            goal = "goal_object" if c.get("targets_goal_object") else "other"
            det = "determined" if c.get("context_determined") else "not_determined"
            groups[f"{out}/{goal}/{det}"].append((s, c))
            groups[f"{out}/all"].append((s, c))
            groups["regime" if regime(c) else "not_regime"].append((s, c))
    rep: dict[str, Any] = {}
    for arm in (AWARE, BLIND, SWAPPED):
        if arm not in L:
            continue
        sts = [s for s in states if arm != SWAPPED or s.get("swap")]
        pos = [L[arm][cand_id(s, c)] for s in sts for c in scored_candidates(s) if not c["failed"]]
        neg = [L[arm][cand_id(s, c)] for s in sts for c in scored_candidates(s) if c["failed"]]
        rep[arm] = {"auc": auc(pos, neg), "yes_rate_successes": _mean([float(x > 0) for x in pos]),
                    "yes_rate_failures": _mean([float(x > 0) for x in neg]),
                    "by_group": {k: {"n": len(v), "yes_rate": _mean([float(accepts(L[arm], cand_id(s, c))) for s, c in v
                                                                     if cand_id(s, c) in L[arm]])}
                                 for k, v in sorted(groups.items())}}
    rep["aware_minus_blind"] = {
        k: {"n": len(v), "yes_rate_shift": _mean([float(accepts(L[AWARE], cand_id(s, c))) - float(accepts(L[BLIND], cand_id(s, c)))
                                                  for s, c in v]),
            "mean_L_shift": _mean([Lraw[AWARE][cand_id(s, c)] - Lraw[BLIND][cand_id(s, c)] for s, c in v])}
        for k, v in sorted(groups.items())}
    return rep


def competence(acc: dict[str, Any]) -> dict[str, Any]:
    b = acc[BLIND]
    ok = (b["auc"] >= COMPETENCE["auc_min"] and b["yes_rate_failures"] <= COMPETENCE["yes_failures_max"]
          and b["yes_rate_successes"] >= COMPETENCE["yes_successes_min"])
    return {"interpretable": bool(ok), "auc_blind": b["auc"], "yes_rate_failures_blind": b["yes_rate_failures"],
            "yes_rate_successes_blind": b["yes_rate_successes"], "rule": COMPETENCE}


def decomposition(states: list[dict], L: dict[str, dict[str, float]]) -> dict[str, Any]:
    """aware - blind = (aware - swapped) + (swapped - blind) in chosen-failure rate, on the states with a swap."""
    sts = [s for s in states if s.get("swap")]
    games = [s["game_id"] for s in sts]
    f = {cfg: [float(choose(cfg, s, L)["failed"]) for s in sts] for cfg in ("aware", "blind", "swapped")}
    ab, as_, sb = (_paired(f["aware"], f["blind"], games), _paired(f["aware"], f["swapped"], games),
                   _paired(f["swapped"], f["blind"], games))
    return {"n_states": len(sts), "aware_minus_blind": ab, "aware_minus_swapped": as_, "swapped_minus_blind": sb,
            "identity_holds": (not sts) or abs(ab["estimate"] - as_["estimate"] - sb["estimate"]) < 1e-9}


def _delta(states: list[dict], L: dict[str, dict[str, float]], **kw: Any) -> dict[str, Any]:
    games = [s["game_id"] for s in states]
    fa = [float(choose("aware", s, L, **kw)["failed"]) for s in states]
    fb = [float(choose("blind", s, L, **kw)["failed"]) for s in states]
    return _paired(fa, fb, games)


def game_weighted_delta(states: list[dict], L: dict[str, dict[str, float]]) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    for s in states:
        by[s["game_id"]].append(float(choose("aware", s, L)["failed"]) - float(choose("blind", s, L)["failed"]))
    games = sorted(by)
    per = [_mean(by[g]) for g in games]
    out = _boot(per, games)
    out["p"] = pair_permutation(per, seed=SEED, cluster_ids=games) if per else float("nan")
    return out


def model_report(states: list[dict], L: dict[str, dict[str, float]], Lraw: dict[str, dict[str, float]]) -> dict[str, Any]:
    games = [s["game_id"] for s in states]
    configs = [c for c in CONFIGS if c != "swapped"]
    rep: dict[str, Any] = {"n_states": len(states), "n_games": len(set(games))}
    rep["fail_rate"] = {c: _boot([float(choose(c, s, L)["failed"]) for s in states], games) for c in configs}
    rep["primary"] = contrast(states, L, "aware", "blind")
    rep["fallback_rate"] = {c: _mean([float(fell_back(c, s, L)) for s in states]) for c in ("aware", "blind")}
    rep["verifier_value"] = {
        c: {"none_minus_verifier": _paired([float(choose("none", s, L)["failed"]) for s in states],
                                           [float(choose(c, s, L)["failed"]) for s in states], games),
            "verifier_minus_oracle": _paired([float(choose(c, s, L)["failed"]) for s in states],
                                             [float(choose("oracle", s, L)["failed"]) for s in states], games)}
        for c in ("aware", "blind")}
    enc = [s for s in states if encounter(s)]
    non = [s for s in states if not encounter(s)]
    d_enc, d_non = _delta(enc, L), _delta(non, L)
    rep["by_encounter"] = {"encounter": d_enc, "non_encounter": d_non,
                           "parts_add_up": (not states) or abs(
                               (len(enc) * (d_enc["estimate"] if enc else 0.0) + len(non) * (d_non["estimate"] if non else 0.0))
                               / len(states) - rep["primary"]["difference"]["estimate"]) < 1e-9}
    rep["progress"] = {c: _boot([float(choose(c, s, L)["progress"]) for s in states], games) for c in configs}
    rep["progress_aware_minus_blind"] = _paired([float(choose("aware", s, L)["progress"]) for s in states],
                                                [float(choose("blind", s, L)["progress"]) for s in states], games)
    rep["acceptance"] = acceptance(states, L, Lraw)
    rep["competence"] = competence(rep["acceptance"])
    if SWAPPED in L:
        sw = [s for s in states if s.get("swap")]
        rep["swapped"] = {"fail_rate": _boot([float(choose("swapped", s, L)["failed"]) for s in sw],
                                             [s["game_id"] for s in sw])}
        rep["decomposition"] = decomposition(states, L)
    early = [s for s in states if s["state_idx"] < EARLY_STEPS]
    rep["sensitivity"] = {"game_weighted": game_weighted_delta(states, L), "first_12_steps": _delta(early, L),
                          "k_2": _delta(states, L, k=2), "k_4": _delta(states, L, k=4),
                          "fallback_max_L": _delta(states, L, fallback="max_L")}
    return rep


def verdict(diff: dict[str, Any], p_adj: float) -> str:
    lo, hi = diff["ci"]
    if p_adj < ALPHA and lo > 0:
        return "toward failure"
    if p_adj < ALPHA and hi < 0:
        return "toward success"
    return "no reliable difference"


def run_decisionsim_agent_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    labs, folders, problems = load_labels(cfg["label_source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(p / "labels.jsonl" for p in folders))
    if problems:
        write_status(run_id, "the analysis waits for every labelling shard:\n" + "\n".join(problems), ok=False)
        return 1
    states = [s for s in labs if usable(s)]
    report: dict[str, Any] = {
        "design": "notes/decisionsim_agent_design.md (committed before any data of this stage)",
        "states": {"n_labelled": len(labs), "n_usable": len(states), "rank1_replay_mismatch": len(labs) - len(states)},
        "encounter": encounter_report(states), "models": {}, "waiting": {}}
    score_folders: list[Path] = []
    for tag, source in cfg.get("models", {}).items():
        sfolders, sproblems = complete_shards(source)
        for p in sfolders:
            saved = json.loads((p / "config.json").read_text(encoding="utf-8")) if (p / "config.json").exists() else {}
            if saved.get("label_source") != cfg["label_source"]:
                sproblems.append(f"{p.name} scored another label run ({saved.get('label_source')})")
        if sproblems:
            report["waiting"][tag] = sproblems
            continue
        L, Lraw = scores_by_arm([x for p in sfolders for x in read_jsonl(p / "scoring.jsonl")])
        if AWARE not in L or BLIND not in L:
            report["waiting"][tag] = [f"needs {AWARE} and {BLIND}; has {sorted(L)}"]
            continue
        sts, short = complete_states(states, L)
        report["models"][tag] = {"n_states_incomplete": short, **model_report(sts, L, Lraw),
                                 "disagreements": disagreements(sts, L)}
        score_folders += sfolders
    priority = [t for t in cfg.get("priority", PRIORITY)]
    ps = {t: report["models"][t]["primary"]["difference"]["p"] for t in priority if t in report["models"]}
    adj = holm(ps) if len(ps) == len(priority) else {}
    report["primary"] = {
        "priority_models": priority, "holm_family_size": len(priority),
        "per_model": {t: {"delta_invalid": report["models"][t]["primary"]["difference"]["estimate"],
                          "ci": report["models"][t]["primary"]["difference"]["ci"],
                          "p": report["models"][t]["primary"]["difference"]["p"], "p_holm": adj.get(t),
                          "interpretable": report["models"][t]["competence"]["interpretable"],
                          "verdict": (verdict(report["models"][t]["primary"]["difference"], adj[t])
                                      if t in adj and report["models"][t]["competence"]["interpretable"]
                                      else "not interpreted" if t in adj else "waiting for every priority model")}
                      for t in priority if t in report["models"]}}
    report["stage2_feed"] = {t: stage2_feed(m["disagreements"]) for t, m in report["models"].items()}
    from wf.provenance import provenance

    report["provenance"] = provenance(runs={cfg["label_source"]: resolve_run(cfg["label_source"])},
                                      inputs=[p / "labels.jsonl" for p in folders] + [p / "scoring.jsonl" for p in score_folders])
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Stage 1b: verification of the agent's ranked candidates (exploratory)", report)
    lines = [f"{t}: Delta_invalid {v['delta_invalid']:+.4f} {v['ci']}, p_holm {v['p_holm']}, {v['verdict']}"
             for t, v in report["primary"]["per_model"].items()]
    write_status(run_id, f"encounter {report['encounter']['encounter']['estimate']:.3f}\n" + "\n".join(lines)
                 + (f"\nwaiting: {sorted(report['waiting'])}" if report["waiting"] else ""), ok=False,
                 reported="Stage 1b report")
    return 0


def disagreements(states: list[dict], L: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Stage 2's input: every state where the aware and blind choices differ, with its outcome cell."""
    out = []
    for s in states:
        a, b = choose("aware", s, L), choose("blind", s, L)
        if a["rank"] == b["rank"]:
            continue
        cell = ("harm" if a["failed"] and not b["failed"] else "help" if b["failed"] and not a["failed"]
                else "both_fail" if a["failed"] else "both_succeed")
        out.append({"game_id": s["game_id"], "state_idx": s["state_idx"], "aware": a["action"], "blind": b["action"],
                    "cell": cell})
    return out


def stage2_feed(dis: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts a Stage 2 gate would read: disagreement states by outcome cell, and the games holding harm states."""
    cells = Counter(x["cell"] for x in dis)
    return {"disagreement_states": len(dis), **{c: cells.get(c, 0) for c in ("harm", "help", "both_fail", "both_succeed")},
            "harm_games": len({x["game_id"] for x in dis if x["cell"] == "harm"})}


# ---------------- dry run on archived data (no model, no server) ----------------


def first_observations(sources: list[str]) -> dict[str, str]:
    """Each game's opening observation (with its task line), from the item collection's step-0 histories."""
    out: dict[str, str] = {}
    for src in sources:
        for t in read_jsonl(resolve_run(src) / "transitions.jsonl"):
            if t["history_len"] == 0:
                prev = out.setdefault(t["game_id"], t["history_text"])
                if prev != t["history_text"]:
                    raise SystemExit(f"{t['game_id']}: two different opening observations")
    return out


def archived_history(first: str, trace: list[dict], t: int) -> str:
    """The world model's history text at step t (`history_text_from_state`), rebuilt from the episode."""
    parts = [first]
    for s in trace[:t]:
        parts += [f"> {s['executed']}", s["real_obs"]]
    return "\n".join(parts)


def rank1_state(game: str, ep: dict, t: int, first: str) -> dict:
    """A rank-1-only labelled state built from the archived episode (outcome = the recorded reply)."""
    step = ep["trace"][t]
    hist = archived_history(first, ep["trace"], t)
    task_line = re.search(r"Your task is to:.*", first).group(0).strip()
    spec = parse_task_line(task_line)
    a, obs = step["executed"], step["real_obs"]
    failed = obs.strip() == NOTHING
    won = bool(ep.get("success")) and t == len(ep["trace"]) - 1  # an episode ends at its win
    # Not recorded in the episode: whether the action was admissible, and the shared-spot flag (scene scan).
    c = {"rank": 1, "action": a, "count": None, "greedy": True, "true_obs": obs, "server_error": None,
         "failed": failed, "context_determined": context_determined_label(hist, a, obs),
         "targets_goal_object": names_object(a, spec.goal_object), "shared_location": False, "admissible": None,
         "action_type": action_type(a), "won": won,
         "progress": progress_event(a, obs, spec.goal_object, spec.receptacle, hist, won)}
    return {"game_id": game, "state_idx": t, "task_line": task_line, "goal_object": spec.goal_object,
            "target": spec.receptacle, "history_text": hist,
            "history_goal_agnostic": spec.goal_object not in interacted_objects(hist), "swap": None,
            "swapped_task_line": None, "rank1_replay_ok": True, "candidates": [c]}


def repick_proxy(source: str, firsts: dict[str, str]) -> dict[str, Any]:
    """What the same agent chose instead when shown that its proposal would fail (other RQ3 conditions, predictions
    dropped): a stand-in for its second-ranked action."""
    rows: dict[str, Counter] = defaultdict(Counter)
    for f in episode_files(source):
        for ep in read_jsonl(f):
            if ep.get("condition") == "none" or ep.get("retention") != "discarded" or ep["game_id"] not in firsts:
                continue
            goal = parse_task_line(re.search(r"Your task is to:.*", firsts[ep["game_id"]]).group(0)).goal_object
            for s in ep["trace"]:
                if s.get("prediction_shown") not in (NOTHING, "This action is not valid.") or s.get("executed_format_error"):
                    continue
                key = f"{ep['condition']}/{'proposal_works' if s['proposal_valid'] else 'proposal_fails'}"
                r = rows[key]
                r["shown_failing"] += 1
                if s["executed"] == s["proposal"]:
                    continue
                r["repicks"] += 1
                fail = s["real_obs"].strip() == NOTHING
                r["repick_fails"] += fail
                if fail and names_object(s["executed"], goal):
                    r["repick_goal_object_failure"] += 1
                    hist = archived_history(firsts[ep["game_id"]], ep["trace"], s["t"])
                    r["repick_determined_goal_object_failure"] += context_determined_label(hist, s["executed"], s["real_obs"])
    return {k: {**dict(v), "share_goal_object_failure": v["repick_goal_object_failure"] / v["repicks"] if v["repicks"] else float("nan"),
                "share_determined": v["repick_determined_goal_object_failure"] / v["repicks"] if v["repicks"] else float("nan")}
            for k, v in sorted(rows.items())}


def run_decisionsim_agent_dryrun(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    files = episode_files(cfg["episodes_source"])
    trans = [resolve_run(s) / "transitions.jsonl" for s in cfg["first_obs_sources"]]
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(files) + tuple(trans))
    episodes = load_none_episodes(cfg["episodes_source"])
    firsts = first_observations(cfg["first_obs_sources"])
    missing = sorted(set(episodes) - set(firsts))
    if missing:
        write_status(run_id, f"no opening observation for {len(missing)} games: {missing[:5]}", ok=False)
        return 1
    states, problems = [], []
    for g in sorted(episodes):
        for t in range(len(episodes[g]["trace"])):
            s = rank1_state(g, episodes[g], t, firsts[g])
            try:
                blind = remove_task_line(s["history_text"])
                if TASK_PREFIX in blind or blind.replace("\n", "") != s["history_text"].replace(s["task_line"], "").replace("\n", ""):
                    problems.append(f"{g} step {t}: the blind history is not the history without its task line")
            except ValueError as e:
                problems.append(f"{g} step {t}: {e}")
            states.append(s)
    # Known answer: with one candidate per state every configuration chooses rank 1, so Delta_invalid is exactly 0
    # whatever the scores; seeded stand-in scores exercise the analysis without any model.
    rng = random.Random(SEED)
    L = {a: {cand_id(s, c): quantize(rng.gauss(0, 3)) for s in states for c in s["candidates"]} for a in (AWARE, BLIND)}
    plumbing = model_report(states, L, L)
    report = {"design": "notes/decisionsim_agent_design.md", "note": "rank 1 only, from the archived episodes; no model",
              "n_states": len(states), "n_games": len(episodes), "episodes_won": sum(bool(e["success"]) for e in episodes.values()),
              "episodes_at_cap": sum(len(e["trace"]) >= C.STEP_CAP for e in episodes.values()),
              "history_problems": problems[:20], "n_history_problems": len(problems),
              "rank1": encounter_report(states),
              "rank1_failing_by_verb": dict(sorted(Counter(s["candidates"][0]["action_type"] for s in states
                                                            if s["candidates"][0]["failed"]).items())),
              "repick_proxy": repick_proxy(cfg["episodes_source"], firsts),
              "plumbing_known_answer": {"delta_invalid": plumbing["primary"]["difference"]["estimate"],
                                        "is_zero": plumbing["primary"]["difference"]["estimate"] == 0.0,
                                        "identity_holds": plumbing["primary"]["identity_holds"]}}
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Stage 1b dry run (archived episodes, rank 1 only, no model)", report)
    ok = not problems and report["plumbing_known_answer"]["is_zero"]
    write_status(run_id, f"{len(states)} states in {len(episodes)} games; history problems {len(problems)}; "
                         f"rank-1 failure rate {report['rank1']['fail_rate']['none']['estimate']:.4f}", ok=ok)
    return 0 if ok else 1
