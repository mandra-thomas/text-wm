"""Assemble runs/PILOT_REPORT.md from per-run REPORT/STATUS files.

Refuses to write unless every run has a STATUS from this commit, or from a commit named with
--accept-commit, and real reports hold no mock runs. Each run's status and commit are printed
beside its result.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from wf.util import git_sha, is_mock, runs_root

TIER0 = ("t0_smoke", "t0_replay", "t0_agent_determinism", "t0_parser", "t0_ckpt_accuracy",
         "t0_scoring_agreement", "t0_expert_labels")
PILOT_A = ("pilotA_transitions", "pilotA_score")
PILOT_B = ("pilotB_headroom", "pilotB_consumption", "pilotB_eligibility", "pilotB_retention_minirun")
OPTIONAL = ("pilotB_headroom_recheck",)  # runs only after a borderline headroom result


def _slurp(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "_missing_"


def _status(root: Path, rid: str) -> tuple[str, str]:
    """(first line of STATUS.md, saved commit); empty when missing."""
    status = root / rid / "STATUS.md"
    sha = root / rid / "git_sha"
    lines = status.read_text(encoding="utf-8").splitlines() if status.exists() else []
    return (lines[0] if lines else ""), (sha.read_text(encoding="utf-8").strip() if sha.exists() else "")


def problems(root: Path, accepted: dict[str, str]) -> list[str]:
    found = []
    required = TIER0 + PILOT_A + PILOT_B
    optional = tuple(r for r in OPTIONAL if (root / r / "STATUS.md").exists())
    for rid in required + optional:
        first, sha = _status(root, rid)
        allowed = [git_sha()] + ([accepted[rid]] if rid in accepted else [])
        if not first:
            found.append(f"{rid}: no STATUS.md")
        elif not any(sha.startswith(c) for c in allowed if c):
            found.append(f"{rid}: from commit {sha or 'unknown'}, not {git_sha()}")
        elif ("(MOCK)" in first) != is_mock():
            found.append(f"{rid}: mock and real runs must not share a report")
    return found


def _section(root: Path, rids: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for rid in rids:
        first, sha = _status(root, rid)
        lines += [f"### {rid}", f"{first.removeprefix('# ')} — git {sha}", "", _slurp(root / rid / "REPORT.md"), ""]
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept-commit", action="append", default=[], metavar="RUN=SHA",
                        help="Accept this run from an earlier commit; it stays visible in the report")
    args = parser.parse_args(argv if argv is not None else [])
    accepted = dict(item.split("=", 1) for item in args.accept_commit)
    root = runs_root()
    found = problems(root, accepted)
    if found:
        print("PILOT_REPORT.md not written:", *found, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)
    filed = _slurp(root / "FILED.txt").strip().replace("\n", " | ")
    lines = [
        f"# Pilot report — {datetime.now(timezone.utc).date().isoformat()} — git {git_sha()} — FILED {filed}",
        "",
        "## Tier 0",
        "| check | status | commit | result |",
        "| --- | --- | --- | --- |",
    ]
    for rid in TIER0:
        first, sha = _status(root, rid)
        status = first.split(" — ")[-1]
        verdict = "below target" if "BELOW TARGET" in first else status
        body = " ".join(_slurp(root / rid / "STATUS.md").splitlines()[2:])[:200]
        lines.append(f"| {rid} | {verdict} | {sha[:7]} | {body} |")
    lines += ["", "## Pilot A (RQ2)", ""] + _section(root, PILOT_A)
    optional = tuple(r for r in OPTIONAL if (root / r / "STATUS.md").exists())
    lines += ["## Pilot B (RQ3)", ""] + _section(root, PILOT_B[:1] + optional + PILOT_B[1:])
    lines += [
        "## Deviations / failures",
        "See per-run STATUS.md files under runs/ and notes/deviations.md.",
        "",
    ]
    if is_mock():
        lines += ["## Note", "MOCK DATA from WF_MOCK_*=1. Not pilot results.", ""]
    out = root / "PILOT_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
