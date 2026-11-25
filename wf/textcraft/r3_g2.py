"""TextCraft round 3, deviation D-G2 (2026-11-25, after the freeze; notes/textcraft_r3_deviation_attention_float32.md).

`textcraft_r3_attnf32_<tag>`: the attention-agreement criterion of design section 4 (eager against SDPA attention on
the smoke run's first five main failures, aligned two-candidate and yes/no arms; wave B's limits unchanged: yes/no mean
at most 0.10 nats, two-candidate mean at most 0.50, no single score more than 1.0 apart) evaluated in float32. It runs
the frozen function `r3_score.attention_agreement` on a float32 copy of the model, loaded from the same verified
node-local snapshot with every setting of the model's frozen spec, and files the result as `attention_f32.json` in
its shard folder. The gates (`r3_gates.attention_f32`) read the attention agreement from that record instead of the
bf16 check each job filed. A filed record is final.

A float32 copy of Gemma 4 12B (about 48 GB) does not fit an L40S, and one of 31B (about 125 GB) needs two A100 80 GB
on one node, so this check runs on A100 80 GB cards; the scoring keeps each model's pinned GPU type.
"""
from __future__ import annotations

import json
from typing import Any

from wf import families as FAM
from wf.textcraft import prompts as P
from wf.textcraft import r3 as R3
from wf.textcraft import r3_gates as G
from wf.textcraft.r2_score import load_records, verified_files
from wf.util import require_filed, write_json, write_status


def f32_scorer(tag: str):
    """A float32 copy of `tag` with every setting of its frozen spec (strict loading included)."""
    from wf.scoring.logodds import LogoddsScorer

    s = R3.MODELS[tag]
    return LogoddsScorer(s.hf_id, model_tag=tag, dtype="float32", attn_implementation=s.attn_implementation,
                         template_kwargs=s.template_kwargs, model_class=s.model_class,
                         tokenizer_kwargs=s.tokenizer_kwargs, prompt_format=s.prompt_format,
                         strict_load=s.strict_load, unexpected_keys_allowed=s.unexpected_keys_allowed)


def run_textcraft_r3_attnf32(cfg: dict) -> int:
    from wf.runs import _init_run
    from wf.textcraft import r3_score as S
    from wf.util import runs_root

    tag = cfg["model"]
    run_id = cfg["run_id"]
    if tag not in G.f32_tags() or run_id != f"textcraft_r3_attnf32_{tag}":
        raise SystemExit(f"the float32 attention check runs as textcraft_r3_attnf32_<tag> for {G.f32_tags()}, "
                         f"not {run_id} on {tag!r}")
    require_filed(run_id)
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"records not verified: {files}", ok=False)
        return 1
    rpath, spath, mpath = files
    spec = R3.MODELS[tag]
    env: dict[str, Any] = {"environment": FAM.require_environment(spec)}
    env["freeze_commit"] = G.require_freeze_commit(run_id=run_id)  # D-G2's deviation commit
    shard = f"{run_id}/shard_00_of_01"
    if (runs_root() / shard / G.ATTN_F32_FILE).exists():
        raise SystemExit(f"{shard} holds a filed float32 attention record; it is final")
    d = _init_run(shard, {**cfg, "run_id": shard, "model_spec": spec.record(), "deviation": "D-G2"},
                  inputs=(rpath, spath, mpath))
    records = S.smoke_subset(load_records(rpath))  # the smoke run's one shard: its first five main failures
    shots = P.fewshot_examples(spath)
    scorer = f32_scorer(tag)
    if scorer.mock:
        rec: dict[str, Any] = {"tag": tag, "ok": None, "skipped": "mock", "dtype_loaded": None}
    else:
        import torch

        rec = S.attention_agreement(scorer, spec, records, shots)
        rec.update({"tag": tag, "dtype_loaded": str(scorer._model.dtype), "loading": scorer.load_info,
                    "device_map": {k: str(v) for k, v in (getattr(scorer._model, "hf_device_map", None) or {}).items()},
                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    rec.update({"deviation": "D-G2", "precision": "float32", "environment": env})
    write_json(d / G.ATTN_F32_FILE, json.loads(json.dumps(rec, default=str)))
    scorer.close()
    write_status(shard, f"float32 attention agreement of {tag}: ok {rec.get('ok')}", ok=False, reported="shard complete")
    return 0
