"""Provenance block for every analysis's metrics.json (decided 2026-11-20, audit B6).

An analysis records which code produced it and exactly what it read: the analysis commit, the commit and status
of every run (or scoring shard) whose output it read, the GPU each of those jobs ran on where the job recorded
it, the sha256 of every input file, and the parser version where written predictions are read. With this, a
number in a metrics file can be traced to its inputs without the private tracker.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from wf.util import file_sha, git_dirty, git_sha, read_jsonl

JOBS_FILE = "jobs.jsonl"  # one line per job that worked on a run (a resumed run has several)


def gpu_names() -> list[str] | None:
    """The GPUs this process can see, by name, or None where there is no NVIDIA driver (CPU jobs, laptops)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and names and all(x.strip().isdigit() for x in visible.split(",")):
        names = [names[int(x)] for x in visible.split(",") if int(x) < len(names)]
    return names or None


def job_record() -> dict[str, Any]:
    """Where this job ran: time, host, Slurm job id and GPU names."""
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu": gpu_names(), "git_sha": git_sha()}


def _first_line(path: Path) -> str | None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return lines[0] if lines else None


def run_record(folder: Path) -> dict[str, Any]:
    """Commit, status line and recorded GPUs of one run folder (a scoring shard or a whole run)."""
    sha = folder / "git_sha"
    jobs = read_jsonl(folder / JOBS_FILE) if (folder / JOBS_FILE).exists() else []
    gpus = sorted({g for j in jobs for g in (j.get("gpu") or [])})
    return {
        "path": str(folder),
        "commit": sha.read_text(encoding="utf-8").strip() if sha.exists() else None,
        "status": _first_line(folder / "STATUS.md"),
        # Jobs before 2026-11-20 did not record their GPU; it is then only in the Slurm logs and the tracker.
        "gpu": gpus or "not recorded",
        "slurm_job_ids": [j.get("slurm_job_id") for j in jobs] or None,
    }


def runs_block(folder: Path) -> dict[str, Any]:
    """A run's record, with one record per shard folder if the run is sharded."""
    shards = sorted(p for p in Path(folder).glob("shard_*_of_*") if p.is_dir())
    if not shards:
        return run_record(Path(folder))
    return {"path": str(folder), "shards": {p.name: run_record(p) for p in shards},
            "commits": sorted({c for p in shards for c in [run_record(p)["commit"]] if c})}


def provenance(runs: dict[str, Path] | None = None, inputs: Iterable[Path] = (), parser: bool = False,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The block to store under "provenance" in an analysis's metrics.json."""
    files = {}
    for path in inputs:
        path = Path(path)
        files[str(path)] = file_sha(path) if path.exists() else None
    block: dict[str, Any] = {
        "analysis_commit": git_sha(),
        "analysis_checkout_dirty": git_dirty(),
        "analysis_job": job_record(),
        "runs_read": {name: runs_block(Path(folder)) for name, folder in (runs or {}).items()},
        "input_sha256": files,
    }
    if parser:
        from wf.tools.parser_audit import parser_version

        block["parser_version"] = parser_version()
    if extra:
        block.update(extra)
    return block


def shard_inputs(folders: Iterable[Path], names: tuple[str, ...] = ("scoring.jsonl", "generations.jsonl", "checks.jsonl")) -> list[Path]:
    """The output files of these shard folders that exist."""
    return [p / n for p in folders for n in names if (p / n).exists()]
