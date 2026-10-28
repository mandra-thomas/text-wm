"""Shared I/O helpers for runs."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]


def runs_root() -> Path:
    return Path(os.environ.get("RUNS_DIR", REPO_ROOT / "runs"))


def run_dir(run_id: str) -> Path:
    d = runs_root() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_status(run_id: str, body: str, ok: bool = True) -> Path:
    p = run_dir(run_id) / "STATUS.md"
    status = "PASS" if ok else "FAIL"
    p.write_text(f"# STATUS — {run_id} — {status}\n\n{body.rstrip()}\n", encoding="utf-8")
    return p


def write_report(run_id: str, body: str) -> Path:
    p = run_dir(run_id) / "REPORT.md"
    p.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    return p


def filed_ok() -> bool:
    return (runs_root() / "FILED.txt").exists()


def require_filed(run_id: str) -> None:
    if not filed_ok():
        write_status(
            run_id,
            "Pilot A/B refuse to run without runs/FILED.txt (protocol §6).",
            ok=False,
        )
        raise SystemExit(2)


def sigmoid(x: float) -> float:
    import math

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
