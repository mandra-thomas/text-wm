"""Shared I/O helpers for runs."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]


def is_mock() -> bool:
    return any(os.environ.get(k) == "1" for k in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE"))


def runs_root() -> Path:
    """Mock runs default to runs_mock/ so they can never overwrite real results."""
    return Path(os.environ.get("RUNS_DIR", REPO_ROOT / ("runs_mock" if is_mock() else "runs")))


def calibration_dir() -> Path:
    """Where pilots write calibrated parameters; mock values stay out of wf/config."""
    return runs_root() if is_mock() else REPO_ROOT / "wf" / "config"


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


def _write_atomic(path: Path, text: str) -> None:
    """A kill mid-write leaves the old file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    _write_atomic(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(rec, sort_keys=True) + "\n" for rec in records)
    with path.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())  # saved means on disk, whatever kills the job next


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    _write_atomic(path, "".join(json.dumps(rec, sort_keys=True) + "\n" for rec in records))


def repair_jsonl(path: Path) -> bool:
    """Cut a half-written last line left by a kill, so the next append starts clean."""
    if not path.exists():
        return False
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep == len(data):
        return False
    with path.open("r+b") as f:
        f.truncate(keep)
    return True


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


def write_status(run_id: str, body: str, ok: bool = True, reported: str | None = None) -> Path:
    """PASS means a target was met. `reported` is for runs with no target, or a result below one."""
    p = run_dir(run_id) / "STATUS.md"
    status = (f"REPORTED ({reported})" if reported else "PASS" if ok else "FAIL") + (" (MOCK)" if is_mock() else "")
    p.write_text(f"# STATUS — {run_id} — {status}\n\n{body.rstrip()}\n", encoding="utf-8")
    return p


def write_report(run_id: str, body: str) -> Path:
    p = run_dir(run_id) / "REPORT.md"
    p.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    return p


def filed_ok() -> bool:
    filed = runs_root() / "FILED.txt"
    if not filed.exists():
        return False
    return is_mock() or "mock" not in filed.read_text(encoding="utf-8").lower()


def require_filed(run_id: str) -> None:
    if not filed_ok():
        write_status(
            run_id,
            "Pilot A/B refuse to run without a real runs/FILED.txt (protocol §6).",
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
