"""Verify collected items against the real environment and an independent source (2026-11-17).

Offline (no server):
  - ids, and (game, state, action), are unique;
  - rules: no item the history-reading rules call a sure failure succeeded on the server;
  - templates: every real reply is one the code knows (parse_observation reads it);
  - scan vs hidden state: each game's live scene scan places every object where Word2World's hidden
    state (built independently) places it at the start;
  - contents labels vs hidden state: a take that failed for "contents" is checked against the hidden
    state when neither the object nor the receptacle has been touched in the history.
With the server (--server): every item's history and true outcome replay exactly, and every attached
success text regenerates exactly (and is not "Nothing happens.").

Writes <run>/VERIFY.json with the transitions file's SHA-256; `rq2_score` refuses items without a pass.

    python -m wf.tools.verify_items rq12_transitions_s0 [--server] [--sample N]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from wf import constants as C
from wf.data.candidates import NOOP, _find_success_obs
from wf.data.transitions import _failure_determined, history_facts
from wf.scoring.parser import parse_observation
from wf.util import file_sha, read_jsonl, runs_root

_HIDDEN_LINE = re.compile(
    r"^\s*-\s*(?:On the ([a-z]+ \d+), you see (.+)|([a-z]+ \d+) is (?:closed|open), if opened, "
    r"(?:in it|on the [a-z]+ \d+), you see (.+))$"
)
_INSTANCE = re.compile(r"\ban? ([a-z]+ \d+)")


def hidden_locations(hidden: str) -> dict[str, set[str]]:
    """instance -> receptacles, from the '=== Objects on Receptacles ===' block of a hidden state."""
    out: dict[str, set[str]] = defaultdict(set)
    for line in (hidden or "").splitlines():
        m = _HIDDEN_LINE.match(line)
        if not m:
            continue
        recep, listed = (m[1], m[2]) if m[1] else (m[3], m[4])
        for inst in _INSTANCE.findall(listed):
            out[inst].add(recep)
    return dict(out)


def _touched(history: str, *names: str) -> bool:
    lines = [ln for ln in history.splitlines() if ln.startswith("> ")]
    return any(re.search(rf"\b{re.escape(n)}\b", ln) for ln in lines for n in names
               if ln.startswith(("> take", "> move", "> put")))


def offline_checks(items: list[dict]) -> dict:
    ids = Counter(t["id"] for t in items)
    keys = Counter((t["game_id"], t["state_idx"], t["action"]) for t in items)
    problems: dict[str, list] = {
        "duplicate_ids": [i for i, k in ids.items() if k > 1],
        "duplicate_state_actions": [list(k) for k, v in keys.items() if v > 1],
        "rule_says_fail_but_succeeded": [],
        "unknown_reply_template": [],
        "scan_disagrees_with_hidden_state": [],
        "contents_label_disagrees_with_hidden_state": [],
    }
    per_game_checked: set[str] = set()
    for t in items:
        obs = (t.get("true_obs") or "").strip()
        if obs != NOOP and not t.get("shared_location") and _failure_determined(history_facts(t["history_text"]), t["action"]):
            problems["rule_says_fail_but_succeeded"].append([t["id"], t["action"], obs[:80]])
        if obs != NOOP and parse_observation(obs) == "unparseable":
            problems["unknown_reply_template"].append([t["id"], t["action"], obs[:80]])
        hidden = hidden_locations(t.get("hidden_state") or "")
        if t["game_id"] not in per_game_checked and t.get("scene_locations") and hidden:
            per_game_checked.add(t["game_id"])
            scan = {k: set(v) for k, v in t["scene_locations"].items()}
            for inst in sorted(set(scan) | set(hidden)):
                if scan.get(inst, set()) != hidden.get(inst, set()):
                    problems["scan_disagrees_with_hidden_state"].append(
                        [t["game_id"], inst, sorted(scan.get(inst, ())), sorted(hidden.get(inst, ()))])
        m = re.fullmatch(r"take (.+) from (.+)", t["action"])
        if (m and obs == NOOP and t.get("determination_reason") == "contents_or_open_state" and hidden
                and m[2] not in history_facts(t["history_text"]).closed and not _touched(t["history_text"], m[1], m[2])):
            if m[2] in hidden.get(m[1], set()):
                problems["contents_label_disagrees_with_hidden_state"].append([t["id"], t["action"]])
    return {"n_items": len(items), "games_scan_checked": len(per_game_checked),
            "problems": problems, "passed": not any(problems.values())}


def server_checks(client, items: list[dict], split: str, sample: int | None = None) -> dict:
    from wf.env.replay import history_text_from_state, replay, step_counterfactual

    chosen = items if not sample else random.Random(f"{C.RNG_SEED}|verify_items").sample(items, min(sample, len(items)))
    replay_bad, twin_bad, twins = [], [], 0
    cache: dict = {}
    for t in chosen:
        handle = replay(client, t["game_id"], list(t["prefix_actions"]), split=split)
        fields = []
        if history_text_from_state(handle.state) != t["history_text"]:
            fields.append("history")
        obs, _ = step_counterfactual(handle, t["action"])
        if obs.strip() != (t.get("true_obs") or "").strip():
            fields.append("true_obs")
        if fields:
            replay_bad.append([t["id"], t["action"], fields])
        cands = t.get("candidates")
        if cands and (t.get("true_obs") or "").strip() == NOOP:
            twins += 1
            again = _find_success_obs(client, t["game_id"], split, t["action"], cache)
            if again is None or again.strip() != cands["o_plus"].strip() or again.strip() == NOOP:
                twin_bad.append([t["id"], t["action"], (again or "")[:80], cands["o_plus"][:80]])
    return {"replayed": len(chosen), "twins_checked": twins, "replay_mismatches": replay_bad,
            "twin_mismatches": twin_bad, "passed": not replay_bad and not twin_bad}


def verify_run(run: str, server: bool = False, sample: int | None = None) -> dict:
    path = runs_root() / run / "transitions.jsonl"
    items = read_jsonl(path)
    result = {"run": run, "transitions_sha256": file_sha(path), "offline": offline_checks(items)}
    if server:
        from wf.env.server import AlfworldClient

        client = AlfworldClient()
        client.create()
        split = items[0].get("split", "train") if items else "train"
        result["server"] = server_checks(client, items, split, sample)
        result["server_sample"] = sample
    result["passed"] = result["offline"]["passed"] and result.get("server", {"passed": True})["passed"]
    result["full"] = bool(server and not sample)
    (runs_root() / run / "VERIFY.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def verified(run: str) -> tuple[bool, str]:
    """Did a full server verification pass on exactly the current transitions file?"""
    report = runs_root() / run / "VERIFY.json"
    if not report.exists():
        return False, "no VERIFY.json"
    r = json.loads(report.read_text(encoding="utf-8"))
    if r.get("transitions_sha256") != file_sha(runs_root() / run / "transitions.jsonl"):
        return False, "transitions changed after verification"
    if not (r.get("passed") and r.get("full")):
        return False, f"verification passed={r.get('passed')} full={r.get('full')}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--server", action="store_true", help="also replay on the real server")
    p.add_argument("--sample", type=int, default=None, help="replay only this many items (not a full pass)")
    args = p.parse_args(argv)
    ok = True
    for run in args.runs:
        r = verify_run(run, args.server, args.sample)
        summary = {k: len(v) for k, v in r["offline"]["problems"].items()}
        if "server" in r:
            summary.update(replay_mismatches=len(r["server"]["replay_mismatches"]),
                           twin_mismatches=len(r["server"]["twin_mismatches"]), replayed=r["server"]["replayed"])
        print(run, "PASS" if r["passed"] else "FAIL", summary, flush=True)
        ok &= r["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
