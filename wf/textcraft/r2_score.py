"""TextCraft round 2 scoring: `textcraft_r2_score_<tag>`, one run per model (notes/textcraft_r2_design.md).

Every record and arm of wf/textcraft/data/r2_items.jsonl under the clean prompt, sharded by goal pair (WF_SHARD /
WF_N_SHARDS). The records file must be the one r2_manifest.json marks verified, and the prompt examples the verified
ones. Once the model is loaded, and before any score (on a resume with work left too), the tokenization gate
(r2_tokcheck.gate) runs on every record and arm of the shard and the golden renderings are compared; any problem stops
the job before it writes a score. Two-candidate arms are scored with score_pair, yes/no arms with
score_one_token_pair (equal by construction for one-token candidates, one forward pass). Each score is saved as it is
made; a resumed shard skips finished work and refuses saved scores whose record hash changed.
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
from wf.textcraft import r2 as R
from wf.util import REPO_ROOT, append_jsonl, read_jsonl, require_filed, write_json, write_status

DEFAULT_MANIFEST = "wf/textcraft/data/r2_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_files(cfg: dict) -> tuple[Path, Path, Path] | str:
    """(records, prompt examples, manifest) when the round-2 manifest is verified and hashes exactly these files."""
    mpath = REPO_ROOT / cfg.get("manifest", DEFAULT_MANIFEST)
    if not mpath.exists():
        return f"no manifest at {mpath}"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if m.get("verified") is not True:
        return f"manifest not verified: {m.get('why_not')}"
    records, shots = REPO_ROOT / m["records_file"], REPO_ROOT / m["fewshot_file"]
    if _sha(records) != m["records_sha256"]:
        return f"{records} does not match the manifest's hash"
    if _sha(shots) != m["fewshot_sha256"]:
        return f"{shots} does not match the manifest's hash"
    return records, shots, mpath


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shard_records(records: list[dict], i: int, n: int) -> list[dict]:
    keep = set(sorted({r["pair_id"] for r in records})[i::n])
    return [r for r in records if r["pair_id"] in keep]


def smoke_subset(records: list[dict], n: int) -> list[dict]:
    """A fixed, seeded sample of records for format and timing checks, never for results."""
    chosen = set(random.Random(f"{C.RNG_SEED}|textcraft|r2|smoke").sample(sorted(r["id"] for r in records), n))
    return [r for r in records if r["id"] in chosen]


def open_scorer(tag: str) -> LogoddsScorer:
    spec = R.MODELS[tag]
    return LogoddsScorer(spec.hf_id, model_tag=tag, attn_implementation=spec.attn_implementation,
                         template_kwargs=spec.template_kwargs)


def run_gate(d: Path, scorer: LogoddsScorer, tag: str, records: list[dict], all_records: list[dict], shots) -> bool:
    """The tokenization gate and the golden comparison, written to tokenization_checks.json; True when scoring may
    start. Skipped (and recorded as skipped) in mock mode, which has no tokenizer."""
    if scorer.mock:
        write_json(d / "tokenization_checks.json", {"ok": None, "skipped": "mock"})
        return True
    from wf.textcraft import r2_tokcheck as T

    spec = R.MODELS[tag]
    rec = T.gate(scorer, spec, records, shots)
    rec["golden_problems"] = T.golden_problems(scorer, tag, all_records, shots)
    rec["ok"] = rec["ok"] and not rec["golden_problems"]
    rec["attn_implementation_loaded"] = getattr(scorer._model.config, "_attn_implementation", None)
    rec["dtype_loaded"] = str(getattr(scorer._model, "dtype", ""))
    if spec.attn_implementation and rec["attn_implementation_loaded"] != spec.attn_implementation:
        rec["ok"] = False
        rec["problems"].append(f"attention {rec['attn_implementation_loaded']} loaded, {spec.attn_implementation} required")
    write_json(d / "tokenization_checks.json", rec)
    return bool(rec["ok"])


def score_one(scorer: LogoddsScorer, tag: str, rec: dict, arm: str, shots):
    msgs = R.model_messages(R.MODELS[tag], rec, arm, shots)
    plus, minus = R.candidates(rec, arm)
    if R.parse_arm(arm)[0]:
        return scorer.score_one_token_pair(transition_id=rec["id"], condition=arm, messages=msgs, o_plus=plus, o_minus=minus)
    return scorer.score_pair(transition_id=rec["id"], condition=arm, messages=msgs, o_plus=plus, o_minus=minus)


def run_textcraft_r2_score(cfg: dict) -> int:
    tag = cfg["model"]
    if tag not in R.MODELS:
        raise SystemExit(f"unknown model {tag!r}; known: {list(R.MODELS)}")
    run_id = cfg["run_id"]
    require_filed(run_id)
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"records not verified: {files}", ok=False)
        return 1
    rpath, spath, mpath = files
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n, "model_spec": R.spec_record(tag),
           "system_text": R.SYSTEM}
    d = _init_run(run_id, cfg, inputs=(rpath, spath, mpath))
    all_records = load_records(rpath)
    records = smoke_subset(all_records, int(cfg["max_records"])) if cfg.get("max_records") else all_records
    records = shard_records(records, i, n)
    shots = P.fewshot_examples(spath)
    hashes = {r["id"]: R.item_hash(r, shots) for r in records}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(s.get("model") != tag or hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different records, prompts or models; rerun without --resume.")
    done = {(s["transition_id"], s["condition"]) for s in saved}
    todo = [(r, a) for r, a in R.jobs(records) if (r["id"], a) not in done]
    if todo:
        scorer = open_scorer(tag)
        if not run_gate(d, scorer, tag, records, all_records, shots):
            write_status(run_id, f"stopped before scoring: tokenization gate failed for {tag} "
                                 f"(see {d / 'tokenization_checks.json'})", ok=False)
            scorer.close()
            return 1
        for rec, arm in todo:
            sc = score_one(scorer, tag, rec, arm, shots)
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": rec["pair_id"], "pair_id": rec["pair_id"],
                                                "kind": rec["kind"], "population": rec["population"],
                                                "role": rec["role"], "item_hash": hashes[rec["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(records)} records, {len(R.jobs(records))} scores ({tag})", ok=False,
                 reported="shard complete")
    return 0
