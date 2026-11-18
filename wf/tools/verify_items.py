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
`--part i/k` replays only every k-th item from the i-th (a fresh server per part, since the server leaks
memory on each reset) and writes VERIFY_part_i_of_k.json; a complete set of passing parts on the same
file counts as a full verification.

    python -m wf.tools.verify_items rq12_transitions_s0 [--server] [--sample N | --part i/k]
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


def _part_name(i: int, k: int) -> str:
    return f"VERIFY_part_{i}_of_{k}.json"


def verify_run(run: str, server: bool = False, sample: int | None = None,
               part: tuple[int, int] | None = None) -> dict:
    if sample and part:
        raise ValueError("--sample and --part are exclusive")
    path = runs_root() / run / "transitions.jsonl"
    items = read_jsonl(path)
    result = {"run": run, "transitions_sha256": file_sha(path), "offline": offline_checks(items)}
    if server:
        from wf.env.server import AlfworldClient

        client = AlfworldClient()
        client.create()
        split = items[0].get("split", "train") if items else "train"
        chosen = items[part[0]::part[1]] if part else items
        result["server"] = server_checks(client, chosen, split, sample)
        result["server_sample"] = sample
        result["part"] = list(part) if part else None
        result["part_items"] = len(chosen)
    result["passed"] = result["offline"]["passed"] and result.get("server", {"passed": True})["passed"]
    result["full"] = bool(server and not sample)
    name = _part_name(*part) if part else "VERIFY.json"
    (runs_root() / run / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _check_report(r: dict, sha: str) -> str | None:
    if r.get("transitions_sha256") != sha:
        return "transitions changed after verification"
    if not (r.get("passed") and r.get("full")):
        return f"verification passed={r.get('passed')} full={r.get('full')}"
    return None


def verified(run: str) -> tuple[bool, str]:
    """Did a full server verification pass on exactly the current transitions file (whole, or every part)?"""
    d = runs_root() / run
    sha = file_sha(d / "transitions.jsonl")
    whole = None
    if (d / "VERIFY.json").exists():
        whole = _check_report(json.loads((d / "VERIFY.json").read_text(encoding="utf-8")), sha)
        if whole is None:
            return True, "ok"
    parts = sorted(d.glob("VERIFY_part_*_of_*.json"))
    if not parts:
        return False, whole or "no VERIFY.json"
    ks = {p.stem.rsplit("_of_", 1)[1] for p in parts}
    if len(ks) != 1:
        return False, f"verification parts from different splits: {[p.name for p in parts]}"
    k = int(ks.pop())
    have = {int(p.stem.split("_")[2]) for p in parts}
    if have != set(range(k)):
        return False, f"verification parts missing: {sorted(set(range(k)) - have)} of {k}"
    n_items = len(read_jsonl(d / "transitions.jsonl"))
    replayed = 0
    for p in parts:
        r = json.loads(p.read_text(encoding="utf-8"))
        problem = _check_report(r, sha)
        if problem:
            return False, f"{p.name}: {problem}"
        if r.get("part") != [int(p.stem.split("_")[2]), k]:
            return False, f"{p.name}: records part {r.get('part')}"
        replayed += r["server"]["replayed"]
    if replayed != n_items:
        return False, f"parts replayed {replayed} of {n_items} items"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--server", action="store_true", help="also replay on the real server")
    p.add_argument("--sample", type=int, default=None, help="replay only this many items (not a full pass)")
    p.add_argument("--part", default=None, help="i/k: replay every k-th item from the i-th")
    args = p.parse_args(argv)
    part = tuple(map(int, args.part.split("/"))) if args.part else None
    if part and not (0 <= part[0] < part[1]):
        p.error("--part i/k needs 0 <= i < k")
    ok = True
    for run in args.runs:
        r = verify_run(run, args.server, args.sample, part)
        summary = {k: len(v) for k, v in r["offline"]["problems"].items()}
        if "server" in r:
            summary.update(replay_mismatches=len(r["server"]["replay_mismatches"]),
                           twin_mismatches=len(r["server"]["twin_mismatches"]), replayed=r["server"]["replayed"])
        print(run, "PASS" if r["passed"] else "FAIL", summary, flush=True)
        ok &= r["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
