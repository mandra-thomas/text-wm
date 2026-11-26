"""Shared loader for the ALFWorld walkthrough (notes/walkthroughs/01_alfworld.md).

Reads the collected items and the stored score files of the ALFWorld studies from a local copy of the run
archive (not public; `notes/reproducing.md` section 2) and indexes every score by (model, item, condition).
Nothing is recomputed from a model: every number the walkthrough shows is a stored score or a difference of
two stored scores. Set ARCHIVE to the folder that holds the `text-wm-fix<N>/runs/` checkouts:

    ARCHIVE=/path/to/text-wm-runs-archive PYTHONPATH=. python scripts/walkthroughs/alfworld_samples.py

CPU only. The items' `hidden_state` field (Word2World text, no licence; DATA_LICENSES.md) is dropped on load
and never read.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ARCHIVE = Path(os.path.expanduser(os.environ.get("ARCHIVE", "~/text-wm-runs-archive")))

# Which checkout holds each run used here (notes/reproducing.md section 3 and each study's results note).
RUNS: dict[str, str | list[str]] = {
    "items": ["text-wm-fix11/runs/rq12_transitions_s0", "text-wm-fix11/runs/rq12_transitions_s1",
              "text-wm-fix11/runs/rq12_transitions_s2"],
    "rq2_score": "text-wm-fix11/runs/rq2_score",
    "rq1_score": "text-wm-fix12/runs/rq1_score",
    "rq1x_score": "text-wm-fix13/runs/rq1x_score",
    "rq1v_score": "text-wm-fix17/runs/rq1v_score",
    "rq1d_score": "text-wm-fix15/runs/rq1d_score",
    "mech_score": "text-wm-fix19/runs/mech_score",
    "mech2_score": "text-wm-fix36/runs/mech2_score",
    "thinking_probe": "text-wm-fix14/runs/thinking_probe",
    "rq3": "text-wm-fix11/runs/rq3",
    "decisionsim_qwen": "text-wm-fix30/runs/decisionsim_score_qwen3-8b",
    "decisionsim_llama": "text-wm-fix30/runs/decisionsim_score_llama3.1-8b",
    **{f"size_score_qwen3-{s}": f"text-wm-fix18/runs/size_score_qwen3-{s}"
       for s in ("0.6b", "1.7b", "4b", "8b", "14b", "32b")},
    **{f"family_score_{m}": f"text-wm-fix21/runs/family_score_{m}"
       for m in ("gemma2-9b-it", "mistral-7b-v0.3", "llama3.2-1b", "llama3.2-3b")},
}


def read_jsonl(path: Path | str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_items() -> dict[str, dict]:
    items: dict[str, dict] = {}
    for run in RUNS["items"]:
        for t in read_jsonl(ARCHIVE / run / "transitions.jsonl"):
            t.pop("hidden_state", None)
            items[t["id"]] = t
    return items


def load_rows(run_key: str, name: str = "scoring.jsonl") -> list[dict]:
    root = ARCHIVE / str(RUNS[run_key])
    rows: list[dict] = []
    for f in sorted(glob.glob(str(root / "**" / name), recursive=True)):
        rows += read_jsonl(f)
    return rows


def index(rows: list[dict], key: str = "condition") -> dict[tuple[str, str, str], dict]:
    """(model, transition_id, condition) -> score record. A duplicate must carry the same L, or the load fails."""
    out: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        k = (r["model"], r["transition_id"], r[key])
        if k in out and abs(out[k]["L"] - r["L"]) > 1e-9:
            raise ValueError(f"conflicting duplicate score {k}")
        out[k] = r
    return out
