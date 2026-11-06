"""Write a Markdown file of real items and agent traces for a person to read."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from wf import constants as C
from wf.data.transitions import filter_pilot_a_items
from wf.env.server import AlfworldClient
from wf.runs import _games, _walk_transitions
from wf.util import append_jsonl, git_sha, is_mock, read_jsonl, runs_root

CHECKLIST = """# Human review: real Pilot A items and agent traces

For each item, ask:

1. **Goal:** is the "real goal" what the task says?
2. **Aligned:** would this action actually help that goal if it worked?
3. **Swapped goal:** is it a sensible task, and is only the object different?
4. **Determined:** can you tell the action fails from the history alone? `determined` should be yes only if you can.
5. **Agnostic:** the history should show no take/move/heat/cool/clean/examine of either object.
6. **Success text:** does it read like what the game would say for this action? It came from the game, so it should be exact.
7. **True outcome:** does "Nothing happens." make sense given the history?

Write down the item number and what looks wrong. "This question makes no sense" counts as a finding.
"""


def _item_block(number: int, t: dict) -> str:
    candidates = t.get("candidates") or {}
    history = "\n".join(f"    {line}" for line in t["history_text"].splitlines())
    yes_no = lambda value: "yes" if value else "no"
    return (
        f"## Item {number} — {t['game_id']}, step {t['state_idx']}\n\n"
        f"- **Real goal:** {t['task_line']}\n"
        f"- **Swapped goal:** {t.get('swapped_task_line')}\n"
        f"- **Action tested:** `{t['action']}`\n"
        f"- **True outcome:** {t['true_obs']}\n"
        f"- **Success text (o+):** {candidates.get('o_plus')}\n"
        f"- **Labels:** error_type={t['error_type']}, determined={yes_no(t['context_determined'])}, "
        f"goal-agnostic={yes_no(t['history_goal_agnostic'])}, targets goal={yes_no(t['targets_goal_object'])}"
        f"{', shares a spot with the agent' if t.get('shared_location') else ''}"
        f"{', excluded: ' + t['exclusion_reason'] if t.get('exclusion_reason') else ''}\n\n"
        f"History the world model sees:\n\n{history}\n"
    )


def _trace_block(ep: dict) -> str:
    lines = [f"## Episode {ep['game_id']} — success={ep['success']}, steps={ep['steps']}\n"]
    for step in ep["trace"]:
        lines.append(f"{step['t']:>2}. `{step['executed']}` → {step['real_obs'][:110]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--games", type=int, default=36)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    client = AlfworldClient()
    client.create()
    # Items are saved per game, so a killed job keeps them for reading or a rerun of this step.
    progress = runs_root() / "review_items.jsonl"
    progress.unlink(missing_ok=True)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for gid in _games(client, "train", args.games, {}):
        built = _walk_transitions(client, gid, "train")
        by_type[built[0]["task_line"].split(":")[1].split()[0] if built else ""].extend(built)
        append_jsonl(progress, built)

    rng = random.Random(C.RNG_SEED)
    picked: list[dict] = []
    pools = {name: filter_pilot_a_items(items) for name, items in by_type.items()}
    for items in pools.values():
        rng.shuffle(items)
    while len(picked) < args.items - 10 and any(pools.values()):
        for name in sorted(pools):
            if pools[name] and len(picked) < args.items - 10:
                picked.append(pools[name].pop())
    excluded = [t for items in by_type.values() for t in items if t["candidates"] is None and t["true_obs"].strip() == "Nothing happens."]
    rng.shuffle(excluded)
    successes = [t for items in by_type.values() for t in items if t["true_obs"].strip() != "Nothing happens."]
    rng.shuffle(successes)

    source = "MOCK DATA, not the real game." if is_mock() else "Generated from the real ALFWorld server."
    parts = [CHECKLIST, f"{source} Commit {git_sha()}.\n", "\n# A. Items that would be scored\n"]
    parts += [_item_block(i + 1, t) for i, t in enumerate(picked)]
    parts.append("\n# B. Failures excluded (no success text found). Should these really be excluded?\n")
    parts += [_item_block(len(picked) + i + 1, t) for i, t in enumerate(excluded[:5])]
    parts.append("\n# C. Real successes, used for the asymmetry measure\n")
    parts += [_item_block(len(picked) + 6 + i, t) for i, t in enumerate(successes[:5])]
    episodes = read_jsonl(runs_root() / "t0_smoke" / "episodes.jsonl")
    if episodes:
        parts.append("\n# D. Real agent episodes from t0_smoke. Does the agent behave sensibly?\n")
        parts += [_trace_block(ep) for ep in episodes]
    out = Path(args.out) if args.out else runs_root() / "HUMAN_REVIEW.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}: {len(picked)} scored, {min(5, len(excluded))} excluded, {min(5, len(successes))} successes, {len(episodes)} episodes")


if __name__ == "__main__":
    main()
