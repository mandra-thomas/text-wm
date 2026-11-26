"""Load τ-bench retail items and stored scores for the walkthrough (notes/walkthroughs/03_taubench.md).

Read-only helper: it reads the item files written by the build runs and the scoring.jsonl files of the archived run
folders; it never recomputes a score. Expected run folders under ``runs_root``:

- ``tau_build/items.jsonl`` (confirmatory build, frozen commit c737cd6) and ``tau_score/shard_*`` (Qwen3-8B and
  Llama-3.1-8B-Instruct; arms main, rules_only, matched; readouts first, yn, full, target)
- ``tau_family_score_llama3.2-3b`` and ``tau_family_score_mistral-7b-v0.3`` (model families, commit a078a8d)
- ``tau_auth_score`` (authorization experiment, commit f70b285; arms main, blind, auth; readouts first, yn, K1-K4)
- ``tau_auth_l32_score`` (its Llama-3.2-3B extension, tag tau-auth-ext-llama32-2026-11-25)
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

MODEL_NAMES = {"qwen3-8b-fewshot": "Qwen3-8B", "llama3.1-8b-fewshot": "Llama-3.1-8B-Instruct",
               "llama3.2-3b": "Llama-3.2-3B-Instruct", "mistral-7b-v0.3": "Mistral-7B-Instruct-v0.3"}
RUNS = {"confirmatory": "tau_score", "families_llama3.2-3b": "tau_family_score_llama3.2-3b",
        "families_mistral-7b-v0.3": "tau_family_score_mistral-7b-v0.3", "auth": "tau_auth_score",
        "auth_llama3.2-3b": "tau_auth_l32_score"}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scores(run_dir: str | Path) -> dict[tuple, dict]:
    """{(model, arm, action, readout, condition, transition_id): record} for one run folder (all shards)."""
    files = sorted(glob.glob(os.path.join(str(run_dir), "shard_*", "scoring.jsonl")))
    if not files:
        raise FileNotFoundError(f"no scoring.jsonl under {run_dir}")
    out: dict[tuple, dict] = {}
    for f in files:
        for r in read_jsonl(f):
            key = (r["model"], r["arm"], r["action"], r["readout"], r["condition"], r["transition_id"])
            if key in out:
                raise ValueError(f"duplicate score {key} in {run_dir}")
            out[key] = r
    return out


def load_all(runs_root: str | Path) -> dict[str, dict]:
    return {name: load_scores(Path(runs_root) / run) for name, run in RUNS.items()}


def items_by_id(path: str | Path) -> dict[str, dict]:
    return {it["id"]: it for it in read_jsonl(path)}


def primary(items: dict[str, dict]) -> list[dict]:
    """The 856 status-matched main-population items (notes/tau_design.md, amendment A3)."""
    return [it for it in items.values() if it.get("main_pool") and it.get("status_matched")]
