"""Scoring runs for the TextCraft replication (notes/textcraft_design.md, sections 5 and 6).

`textcraft_score`: every verified item (main, usefulness, success, secondary get, empty history), both few-shot models, the aligned, swapped and neutral goal lines, each as
the two-candidate prediction (success twin against the failure reply) and as the yes/no question. Sharded by goal
pair (WF_SHARD / WF_N_SHARDS); each score is saved as it is made, so a resumed shard skips finished work.
`textcraft_analysis` (wf/textcraft/analysis.py) combines the shards.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from wf import constants as C
from wf.runs import _init_run
from wf.scoring.logodds import LogoddsScorer
from wf.textcraft import prompts as P
from wf.util import REPO_ROOT, append_jsonl, read_jsonl, require_filed, write_json, write_status

MODELS = ((C.WM_FEWSHOT_MODEL, "qwen3-8b-fewshot"), (C.WM_FEWSHOT_LLAMA, "llama3.1-8b-fewshot"))
CO_PRIMARY = tuple(tag for _, tag in MODELS)
DEFAULT_MANIFEST = "wf/textcraft/data/manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_files(cfg: dict) -> tuple[Path, Path, Path] | str:
    """(items, few-shot, manifest) when the manifest is verified and hashes exactly these files, else the reason."""
    manifest_path = REPO_ROOT / cfg.get("manifest", DEFAULT_MANIFEST)
    if not manifest_path.exists():
        return f"no manifest at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verified") is not True:
        return f"manifest not verified: {manifest.get('why_not')}"
    items, shots = REPO_ROOT / manifest["items_file"], REPO_ROOT / manifest["fewshot_file"]
    if _sha(items) != manifest["items_sha256"]:
        return f"{items} does not match the manifest's hash"
    if _sha(shots) != manifest["fewshot_sha256"]:
        return f"{shots} does not match the manifest's hash"
    return items, shots, manifest_path


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def item_hash(t: dict) -> str:
    """What a score depends on, so a saved score can never attach to a changed item with the same id."""
    fields = {k: t.get(k) for k in ("recipe_lines", "history", "action", "task_line", "swapped_task_line",
                                    "neutral_task_line", "candidates", "prefix_candidates")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def smoke_subset(items: list[dict], n: int) -> list[dict]:
    """A fixed, seeded sample for format and timing checks, never for results."""
    chosen = set(random.Random(f"{C.RNG_SEED}|textcraft|smoke").sample(sorted(t["id"] for t in items), n))
    return [t for t in items if t["id"] in chosen]


def shard_items(items: list[dict], i: int, n: int) -> list[dict]:
    keep = set(sorted({t["pair_id"] for t in items})[i::n])
    return [t for t in items if t["pair_id"] in keep]


def selected_items(cfg: dict, items: list[dict]) -> list[dict]:
    populations = set(cfg.get("populations", ("main", "usefulness", "success", "secondary_get", "empty_history")))
    items = [t for t in items if t["population"] in populations]
    return smoke_subset(items, int(cfg["max_items"])) if cfg.get("max_items") else items


def token_report(scorer, items: list[dict], shots, arms: tuple[str, ...], tag: str | None = None,
                 all_items: list[dict] | None = None) -> dict:
    """wf/textcraft/tokcheck.py with the scorer's own tokenizer and chat encoder, and (given the model's tag and the
    whole items file) the golden rendered prompt of wf/textcraft/golden_chat.py; skipped in mock mode."""
    if scorer.mock:
        return {"ok": True, "skipped": "mock scorer has no tokenizer"}
    from wf.textcraft import golden_chat, tokcheck
    encode = lambda m: [int(i) for i in scorer._encode(m)[0]]
    report = tokcheck.check(scorer._tok, items, shots, arms, encode=encode)
    if tag is not None and all_items is not None:
        report["golden_problems"] = golden_chat.problems(scorer._tok, tag, all_items, shots)
        report["ok"] = report["ok"] and not report["golden_problems"]
    return report


def run_textcraft_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"items not verified: {files}", ok=False)
        return 1
    items_path, shots_path, manifest_path = files
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=(items_path, shots_path, manifest_path))
    items = shard_items(selected_items(cfg, load_items(items_path)), i, n)
    shots = P.fewshot_examples(shots_path)
    hashes = {t["id"]: item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    arms = tuple(cfg.get("arms", P.ARMS))
    for model_name, tag in [m for m in MODELS if m[1] in cfg.get("models", CO_PRIMARY)]:
        todo = [(t, a) for t in items for a in arms if (tag, t["id"], a) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        if cfg.get("token_check"):  # the smoke run: tokenization on these real prompts before any score (B7)
            report = token_report(scorer, items, shots, arms, tag, load_items(items_path))
            write_json(d / f"token_check_{tag}.json", report)
            if not report["ok"]:
                write_status(run_id, f"tokenization check failed for {tag}: {report['examples']}", ok=False)
                scorer.close()
                return 1
        for t, arm in todo:
            plus, minus = P.candidates(t, arm)
            sc = scorer.score_pair(transition_id=t["id"], condition=arm, messages=P.messages(t, arm, shots),
                                   o_plus=plus, o_minus=minus)
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["pair_id"], "pair_id": t["pair_id"],
                                                "population": t["population"], "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} items from {len({t['pair_id'] for t in items})} pairs, "
                         f"{len(arms)} arms", ok=False, reported="shard complete")
    return 0
