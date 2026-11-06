"""Known-answer checks on real models. Each has a result we can predict in advance.

A wrong result here points at a bug, unlike a pilot number, which could just be the science.
  scoring (HF, needs the GPU to itself): checkpoint accuracy, placebo, signal, repeatability,
    tokenizer behaviour and prompt length.
  agent (vLLM): perfect foresight and total corruption as positive controls.
Each writes runs/verify_<part>.json as it goes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path

from wf import constants as C
from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.analysis.stats import cluster_bootstrap, pair_permutation, rank_outcome
from wf.corruption.schedule import schedule
from wf.data.goals import make_swapped_task_line, swapped_goal
from wf.env.scene import scan_scene
from wf.env.server import AlfworldClient
from wf.runs import _games, _walk_transitions
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.scoring.prompts import agent_react_messages
from wf.util import runs_root, sigmoid, write_json

NOOP = "Nothing happens."


def _items(client: AlfworldClient, games: list[str]) -> list[dict]:
    items: list[dict] = []
    for gid in games:
        for t in _walk_transitions(client, gid, "train"):
            if t.get("candidates") and t.get("swapped_goal_object"):
                items.append({**t, "id": f"v_{len(items)}"})
    return items


def _with_second_swap(item: dict, client: AlfworldClient, rng: random.Random) -> dict | None:
    """Same item with a different irrelevant goal, for the placebo comparison."""
    scene = scan_scene(client, item["game_id"], "train")
    other = swapped_goal(scene.objects, item["goal_object"], rng=rng, locations=scene.locations,
                         exclude={item["swapped_goal_object"]})
    if other is None:
        return None
    line = make_swapped_task_line(item["task_line"], item["goal_object"], other)
    return {**item, "swapped_goal_object": other, "swapped_task_line": line}


def _score(scorer: LogoddsScorer, item: dict, condition: str, ft: bool, tag: str) -> float:
    cands = item["candidates"]
    return scorer.score_pair(f"{item['id']}|{tag}", condition, build_messages(item, condition, ft_prompt=ft),
                             cands["o_plus"], cands["o_minus"]).L


def _auc(positive: list[float], negative: list[float]) -> float:
    """Chance a true success scores above a true failure; 0.5 means no signal."""
    if not positive or not negative:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    pick = lambda q: ordered[min(int(q * len(ordered)), len(ordered) - 1)]
    return {"n": len(values), "p10": pick(0.1), "median": pick(0.5), "p90": pick(0.9)}


def checkpoint_accuracy(scorer: LogoddsScorer, path: Path, n: int, report=None) -> dict:
    """Upstream's own data and format; the proposal cites about 99% single-step accuracy."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        messages = row["messages"]
        for i in range(1, len(messages) - 1, 2):
            cases.append((messages[: i + 1], messages[i + 1]["content"]))
    random.Random(C.RNG_SEED).shuffle(cases)
    cases = cases[:n]
    correct = 0
    misses = []
    for done, (conversation, truth) in enumerate(cases, 1):
        guess = scorer.free_generate(conversation, max_new_tokens=256)
        hit = guess.strip() == truth.strip()
        correct += hit
        if not hit and len(misses) < 6:
            misses.append({"truth": truth[:120], "guess": guess[:120]})
        if report and done % 25 == 0:
            report({"n": done, "exact_match": correct / done, "misses": misses})
    return {"n": len(cases), "exact_match": correct / max(len(cases), 1), "misses": misses}


def tokenizer_checks(scorer: LogoddsScorer, item: dict) -> dict:
    if scorer.mock:
        return {"skipped": "mock"}
    tok = scorer._tok
    messages = build_messages(item, "aligned")
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = scorer._encode(messages)[0].tolist()
    out = {
        "prompt_tail": prompt_text[-60:],
        "thinking_disabled_in_prompt": "<think>\n\n</think>" in prompt_text[-40:],
        "prompt_tokens": len(prompt_ids),
    }
    for name in ("o_plus", "o_minus"):
        text = item["candidates"][name]
        separate = prompt_ids + tok(text, add_special_tokens=False).input_ids
        together = tok(prompt_text + text, add_special_tokens=False).input_ids
        out[f"{name}_same_tokens_joined_or_apart"] = separate == together
    return out


def agent_prompt_tokens(scorer: LogoddsScorer, client: AlfworldClient, game: str) -> dict:
    """Worst realistic agent prompt: a full step-cap history with predictions retained."""
    if scorer.mock:
        return {"skipped": "mock"}
    state = client.reset(game, "train")
    history = []
    for t in range(C.STEP_CAP):
        action = state.admissible[t % len(state.admissible)]
        following = client.step(state, action)
        if following.done:
            break
        history.append({"observation": state.observation, "proposal": action,
                        "expected_outcome": following.observation, "action": action})
        state = following
    messages = agent_react_messages(state.observation, history, state.observation, state.admissible, "look")
    ids = scorer._tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return {"steps": len(history), "prompt_tokens": len(ids), "limit": 8192 - 256}


def scoring_checks(args: argparse.Namespace, out: dict, save) -> None:
    client = AlfworldClient()
    client.create()
    games = _games(client, "train", args.games, {})
    items = _items(client, games)
    rng = random.Random(C.RNG_SEED)
    rng.shuffle(items)
    failures = [t for t in items if t["true_obs"].strip() == NOOP][: args.items]
    successes = [t for t in items if t["true_obs"].strip() != NOOP][: args.items]
    out["items"] = {"failures": len(failures), "successes": len(successes), "games": len(games)}
    data = Path(os.environ.get("WORD2WORLD_ROOT", "")) / "data" / C.PARSER_DATASET_FILE
    for model, tag, ft in ((C.WM_FEWSHOT_MODEL, "qwen3-8b-fewshot", False), (C.WM_FT_QWEN, "w2w-qwen2.5-7b", True)):
        scorer = LogoddsScorer(model, model_tag=tag)
        report: dict = {"tokenizer": tokenizer_checks(scorer, failures[0])}
        out[tag] = report

        def partial_accuracy(partial: dict) -> None:
            report["checkpoint_accuracy"] = partial
            save()

        if not ft:
            report["agent_prompt"] = agent_prompt_tokens(scorer, client, games[0])
        elif data.is_file():
            report["checkpoint_accuracy"] = checkpoint_accuracy(scorer, data, args.accuracy_items, partial_accuracy)
        save()

        # Same prompt twice must give the same number.
        repeats = []
        for t in failures[:30]:
            first = _score(scorer, t, "aligned", ft, "r1")
            repeats.append(abs(first - _score(scorer, t, "aligned", ft, "r2")))
        report["repeat_max_abs_diff"] = max(repeats)

        # True successes should outscore true failures; otherwise L carries no information.
        fail_L = [_score(scorer, t, "aligned", ft, "s") for t in failures]
        succ_L = [_score(scorer, t, "aligned", ft, "s") for t in successes]
        report["signal_auc"] = _auc(succ_L, fail_L)
        report["L_on_failures"] = _summary(fail_L)
        report["L_on_successes"] = _summary(succ_L)
        report["p_success_median_on_failures"] = sigmoid(report["L_on_failures"].get("median", 0.0))
        save()

        # Placebo: two goals the action does not serve. No true difference, so about 0.
        diffs, clusters = [], []
        for t in failures:
            other = _with_second_swap(t, client, rng)
            if other is None:
                continue
            diffs.append(_score(scorer, t, "swapped", ft, "p1") - _score(scorer, other, "swapped", ft, "p2"))
            clusters.append(t["game_id"])
        boot = cluster_bootstrap(lambda v: sum(v) / len(v) if v else float("nan"), diffs, clusters, seed=C.RNG_SEED)
        report["placebo_delta_L"] = {**boot, "n": len(diffs), "p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=clusters),
                                     "sd": statistics.stdev(diffs) if len(diffs) > 1 else None}
        save()
        del scorer


def _agent_summary(episodes: list[dict]) -> dict:
    steps = [s for e in episodes for s in e["trace"]]
    return {
        "games": len(episodes),
        "success": sum(e["success"] for e in episodes) / len(episodes),
        "mean_S": rank_outcome(episodes)["mean_S"],
        "mean_steps": sum(e["steps"] for e in episodes) / len(episodes),
        "invalid_proposal_rate": sum(not s["proposal_valid"] for s in steps) / max(len(steps), 1),
        "noop_rate": sum(s["real_obs"].strip() == NOOP for s in steps) / max(len(steps), 1),
        "not_admissible_executed": sum(s["real_obs"].strip() == NOOP and s["executed"] == s["proposal"] for s in steps) / max(len(steps), 1),
        "decision_critical_rate": sum(bool(s["decision_critical"]) for s in steps) / max(len(steps), 1),
        "follow_rate": sum(bool(s["followed"]) for s in steps) / max(len(steps), 1),
        "mean_corrupted": sum(e["n_corrupted"] for e in episodes) / len(episodes),
        "mean_corruption_unavailable": sum(e["n_corruption_unavailable"] for e in episodes) / len(episodes),
    }


def agent_checks(args: argparse.Namespace, out: dict, save) -> None:
    client = AlfworldClient()
    client.create()
    agent = AgentClient()
    games = _games(client, "train", args.games, {})
    conditions = {
        "none": lambda g: ForesightConfig("none"),
        "oracle_discarded": lambda g: ForesightConfig("oracle", retention="discarded"),
        "oracle_retained": lambda g: ForesightConfig("oracle", retention="retained"),
        "validity_only": lambda g: ForesightConfig("validity_only"),
        "all_false_failures": lambda g: ForesightConfig("corrupted", arm="unfavorable",
                                                        corrupt_fn=schedule(g, 1, "unfavorable", 1.0)),
        "all_false_successes": lambda g: ForesightConfig("corrupted", arm="favorable",
                                                         corrupt_fn=schedule(g, 1, "favorable", 1.0)),
    }
    for name, make in conditions.items():
        episodes: list[dict] = []
        for g in games:
            episodes.append(run_episode(client, agent, g, "train", "verify", make(g)))
            out[name] = _agent_summary(episodes)
            save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["scoring", "agent"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--items", type=int, default=150)
    parser.add_argument("--accuracy-items", type=int, default=300)
    args = parser.parse_args()
    out: dict = {}
    path = runs_root() / f"verify_{args.part}.json"

    def save() -> None:
        write_json(path, out)
        print(json.dumps(out, default=str)[-1800:], flush=True)

    (scoring_checks if args.part == "scoring" else agent_checks)(args, out, save)
    save()


if __name__ == "__main__":
    main()
