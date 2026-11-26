"""Load TextCraft items and stored scores for the walkthrough (notes/walkthroughs/02_textcraft.md).

Read-only helper: it reads the frozen item files and the scoring.jsonl files of the archived run folders, and never
recomputes a score. Paths are passed in, so the same code works on a local copy of the run archive.

Layout expected under ``runs_root`` (one folder per run, as written by the pipeline):

- round 1: ``textcraft_score/shard_*/scoring.jsonl`` (Qwen3-8B and Llama-3.1-8B, frozen commit 5a3702a)
- round 2: ``textcraft_r2_score_<model>/shard_*/scoring.jsonl`` (five models, frozen commit 4ccf84b)
- round 3 (interim): ``textcraft_r3_score_<model>``, ``textcraft_r3_rescore_<model>``, ``textcraft_r3_think_qwen3.5-9b``
  and ``textcraft_r3x_gemma4-e4b-it`` (commit fae00b3). Only the models of the interim analysis are listed.

Round 3 runs whose analysis has not been run (Gemma 4 12B, Qwen3.8-27B, Gemma 4 31B) are deliberately not listed and
are never read.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

R1_MODELS = {"qwen3-8b-fewshot": "Qwen3-8B", "llama3.1-8b-fewshot": "Llama-3.1-8B-Instruct"}
R2_MODELS = {"qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B-Instruct", "gemma2-9b-it": "Gemma-2-9B-it",
             "mistral-7b-v0.3": "Mistral-7B-Instruct-v0.3", "llama3.2-3b": "Llama-3.2-3B-Instruct"}
# Round 3: only what the interim analysis read (notes/textcraft_r3_interim_results.md).
R3_RUNS = {
    "qwen3.5-9b": ("textcraft_r3_score_qwen3.5-9b", "Qwen3.5-9B"),
    "ministral3-8b-it": ("textcraft_r3_score_ministral3-8b-it", "Ministral-3-8B"),
    "olmo3-7b-it": ("textcraft_r3_score_olmo3-7b-it", "OLMo 3 7B"),
    "qwen3-8b@serve2": ("textcraft_r3_rescore_qwen3-8b", "Qwen3-8B (rescored in round 3's software)"),
    "mistral-7b-v0.3@serve2": ("textcraft_r3_rescore_mistral-7b-v0.3", "Mistral-7B (rescored in round 3's software)"),
    "gemma2-9b-it@serve2": ("textcraft_r3_rescore_gemma2-9b-it", "Gemma-2-9B (rescored in round 3's software)"),
    "qwen3.5-9b-think": ("textcraft_r3_think_qwen3.5-9b", "Qwen3.5-9B, reasoning on (exploratory)"),
    "gemma4-e4b-it": ("textcraft_r3x_gemma4-e4b-it", "Gemma 4 E4B (exploratory)"),
}
NOT_READ = ("gemma4-12b", "qwen3.8-27b", "gemma4-31b")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _shards(run_dir: str | Path) -> list[str]:
    files = sorted(glob.glob(os.path.join(str(run_dir), "shard_*", "scoring.jsonl")))
    if not files:
        raise FileNotFoundError(f"no scoring.jsonl under {run_dir}")
    for f in files:
        if any(tag in f for tag in NOT_READ):
            raise RuntimeError(f"refusing to read {f}: its analysis has not run")
    return files


def load_scores(run_dir: str | Path, model_key: str | None = None) -> dict[tuple[str, str], dict]:
    """{(transition_id, condition): record} for one run folder (all shards)."""
    out: dict[tuple[str, str], dict] = {}
    for f in _shards(run_dir):
        for r in read_jsonl(f):
            if model_key is not None and r.get("model") != model_key:
                continue
            key = (r["transition_id"], r["condition"])
            if key in out:
                raise ValueError(f"duplicate score {key} in {run_dir}")
            out[key] = r
    return out


def load_round1(runs_root: str | Path) -> dict[str, dict]:
    run = Path(runs_root) / "textcraft_score"
    return {m: load_scores(run, m) for m in R1_MODELS}


def load_round2(runs_root: str | Path) -> dict[str, dict]:
    return {m: load_scores(Path(runs_root) / f"textcraft_r2_score_{m}") for m in R2_MODELS}


def load_round3(runs_root: str | Path) -> dict[str, dict]:
    return {m: load_scores(Path(runs_root) / run) for m, (run, _) in R3_RUNS.items()}


def load_think_generations(runs_root: str | Path) -> dict[tuple[str, str], dict]:
    out = {}
    for f in sorted(glob.glob(str(Path(runs_root) / R3_RUNS["qwen3.5-9b-think"][0] / "shard_*" / "generations.jsonl"))):
        for r in read_jsonl(f):
            out[(r["transition_id"], r["condition"])] = r  # batch_key names the batch's first record, not this one
    return out


def items_by_id(path: str | Path) -> dict[str, dict]:
    return {it["id"]: it for it in read_jsonl(path)}
