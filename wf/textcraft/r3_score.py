"""TextCraft round 3 scoring (notes/textcraft_r3_design.md), in serve2 (transformers 5).

Runs, all on round 2's verified records (wf/textcraft/data/r2_items.jsonl through r2_manifest.json) and prompts:

- `textcraft_r3_score_<tag>`: every record and arm, sharded by goal pair (WF_SHARD / WF_N_SHARDS);
- `textcraft_r3_smoke_<tag>`: 20 seeded records, every arm (read for loading, attention agreement, timing and the
  median yes/no mass only, never for results);
- `textcraft_r3_envcheck_<tag>` (Qwen3-8B, Mistral-7B-v0.3): 300 seeded records, every arm, the earlier model as
  serve2 loads it (families.envcheck_spec), scores saved under `<tag>-serve2`;
- `textcraft_r3_rescore_<tag>` (Qwen3-8B, Mistral-7B-v0.3, Gemma-2-9B-it): every record, the same loading; only if the
  environment check fails (design section 8.1);
- `textcraft_r3_placement_gemma4-12b-it` and `textcraft_r3_nopreserve_qwen3.8-27b`: 300 seeded main failures under
  the aligned and swapped lines in both formats, the format variant saved under `<tag>-firstturn` / `<tag>-nopreserve`.

Before any score (and on a resume with work left): the environment guard, the resource check, the GPU-type check,
the weights check against the manifest (node-local weights), the round-3 gate on every record and arm of the shard
with the golden renderings, and for Gemma 4 the eager-against-SDPA agreement check. Any failure stops the job before
it writes a score. Each score is saved as it is made; a resumed shard skips finished work and refuses saved scores
whose record hash or model differs.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable

from wf import families as FAM
from wf.runs import _init_run
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r3 as R3
from wf.textcraft import r3_gates as G
from wf.textcraft.r2_score import load_records, shard_records, verified_files
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status


def seeded(records: list[dict], n: int, name: str) -> list[dict]:
    """A fixed, seeded sample of `n` records (in file order), the same whatever order the records arrive in."""
    ids = sorted(r["id"] for r in records)
    chosen = set(random.Random(R3.seed(name)).sample(ids, min(n, len(ids))))
    return [r for r in records if r["id"] in chosen]


def smoke_subset(records: list[dict]) -> list[dict]:
    return seeded(records, R3.SMOKE_N, "smoke")


def envcheck_subset(records: list[dict]) -> list[dict]:
    return seeded(records, R3.ENVCHECK_N, "envcheck")


def format_subset(records: list[dict]) -> list[dict]:
    """300 seeded main failures (design section 8.3)."""
    main = [r for r in records if r["kind"] == "main" and r["population"] == "main"]
    return seeded(main, R3.FORMAT_N, "format")


# ---------------- checks at job start ----------------


def gpu_check(tag: str, gpus: list[str] | None = None, exact: list[str] | None = None) -> dict[str, Any]:
    """The job's GPUs against the model's pinned type (design section 10); the check variants run on their model's
    type. With `exact` (the environment check's planned GPU name, the fallback rescoring's L40S; design 14.M4) the
    job's GPU list must be exactly that list. Mock runs are not checked."""
    if os.environ.get("WF_MOCK_SCORE", "0") == "1":
        return {"checked": "mock"}
    from wf.provenance import gpu_names

    gpus = gpu_names() if gpus is None else gpus
    if exact is not None:
        if gpus != exact:
            raise SystemExit(f"{tag} runs on exactly {exact}; this job has {gpus}")
        return {"checked": True, "gpus": gpus, "exact": exact}
    base = R3.base_tag(tag)
    if base not in R3.GPU_PIN:
        return {"checked": "recorded only (earlier model; checked by the analysis)", "gpus": gpus}
    if not R3.gpu_type_ok(base, gpus):
        raise SystemExit(f"{tag} runs on {R3.GPU_PIN[base]}; this job has {gpus}")
    return {"checked": True, "gpus": gpus, "class": R3.gpu_class(gpus)}


def attention_agreement(scorer, spec, records: list[dict], shots) -> dict[str, Any]:
    """Wave B's eager-against-SDPA check (families.attention_agreement, rule families.ATTN_AGREEMENT) on the shard's
    first main failures under the aligned two-candidate and yes/no arms."""
    rule = FAM.ATTN_AGREEMENT
    model = scorer._model
    items = [r for r in records if r["kind"] == "main" and r["population"] == "main"][:rule["n_items"]]
    arms = ("aligned", "yn_aligned")
    problems: list[str] = []

    def score_all() -> dict[tuple[str, str], float]:
        scorer._cache.clear()
        out = {}
        for r in items:
            for arm in arms:
                out[(r["id"], arm)] = score_one(scorer, spec, r, arm, shots).L
        scorer._cache.clear()
        return out

    loaded = FAM._attn_loaded(model)
    if set(loaded) != {"sdpa"}:
        return {"ok": False, "problems": [f"loaded attention is {loaded}, not sdpa"], "loaded": loaded, "rule": rule}
    sdpa = score_all()
    FAM._set_text_attn(model, "eager")
    switched = FAM._attn_loaded(model)
    eager = score_all() if set(switched) == {"eager"} else {}
    FAM._set_text_attn(model, "sdpa")
    restored = FAM._attn_loaded(model)
    if set(switched) != {"eager"}:
        problems.append(f"could not switch to eager attention (config says {switched})")
    if set(restored) != {"sdpa"}:
        problems.append(f"could not switch back to sdpa (config says {restored})")
    diffs = {k: eager[k] - sdpa[k] for k in sdpa if k in eager}
    out: dict[str, Any] = {"n_items": len(items), "rule": rule, "loaded": loaded, "switched": switched,
                           "restored": restored}
    for fmt, keep in (("yes_no", lambda a: a.startswith("yn_")), ("two_candidate", lambda a: not a.startswith("yn_"))):
        v = [abs(x) for (i, a), x in diffs.items() if keep(a)]
        m = sum(v) / len(v) if v else float("nan")
        out[f"{fmt}_mean_abs_diff"] = m
        if not m <= rule[f"{fmt}_mean_abs_max"]:
            problems.append(f"eager and sdpa differ by {m:.3f} nats on average ({fmt})")
    worst = max((abs(x) for x in diffs.values()), default=float("nan"))
    out["max_abs_diff"] = worst
    if not worst <= rule["any_abs_max"]:
        problems.append(f"eager and sdpa differ by {worst:.3f} nats on one score")
    if not items or len(diffs) != len(items) * len(arms):
        problems.append(f"{len(diffs)} scores compared, {len(items) * len(arms)} expected")
    return {**out, "ok": not problems, "problems": problems}


def run_gate(d: Path, scorer, spec, tag: str, records: list[dict], all_records: list[dict], shots,
             env: dict[str, Any]) -> bool:
    """The round-3 gate, the golden comparison, the loading record, the weights check and (Gemma 4) the attention
    agreement, written to tokenization_checks.json; True when scoring may start. Mock runs skip it (no tokenizer)."""
    if scorer.mock:
        write_json(d / "tokenization_checks.json", {"ok": None, "skipped": "mock", "environment": env})
        return True
    from wf.textcraft import r3_tokcheck as T

    rec = T.gate(scorer, spec, records, shots, T.reference_encoder(spec))
    rec["golden_problems"] = T.golden_problems(scorer, tag, all_records, shots)
    rec["ok"] = rec["ok"] and not rec["golden_problems"]
    rec["environment"] = env
    rec["loading"] = scorer.load_info
    rec["device_map"] = {k: str(v) for k, v in (getattr(scorer._model, "hf_device_map", None) or {}).items()}
    rec["model_class_loaded"] = type(scorer._model).__name__
    rec["attn_implementation_loaded"] = FAM._attn_loaded(scorer._model)
    rec["dtype_loaded"] = str(getattr(scorer._model, "dtype", ""))
    weights = FAM.weights_check(spec)
    if weights is not None:
        write_json(d / "weights_check.json", weights)
        rec["weights_check"] = {k: weights.get(k) for k in ("ok", "hf_id", "revision", "n_files", "total_bytes")}
    if rec["ok"] and spec.attn_agreement_check:
        rec["attention_agreement"] = attention_agreement(scorer, spec, records, shots)
        rec["ok"] = rec["attention_agreement"]["ok"]
        if not rec["ok"]:
            rec["problems"] = rec["problems"] + rec["attention_agreement"]["problems"]
    write_json(d / "tokenization_checks.json", rec)
    return bool(rec["ok"])


def score_one(scorer, spec, rec: dict, arm: str, shots):
    msgs = R3.model_messages(spec, rec, arm, shots)
    plus, minus = R.candidates(rec, arm)
    if R.parse_arm(arm)[0]:
        return scorer.score_one_token_pair(transition_id=rec["id"], condition=arm, messages=msgs, o_plus=plus, o_minus=minus)
    return scorer.score_pair(transition_id=rec["id"], condition=arm, messages=msgs, o_plus=plus, o_minus=minus)


# ---------------- the shared scoring loop ----------------


def top_next_token(scorer, msgs) -> tuple[str, bool]:
    """The most probable next token after a prompt, decoded, and whether it is a special token (the smoke runs'
    format diagnostic, design 14.M3 and S9). The mock scorer answers "Yes"."""
    if scorer.mock:
        return "Yes", False
    import torch

    ids = scorer._encode(msgs).to(next(scorer._model.parameters()).device)
    with torch.no_grad():
        top = int(scorer._model(input_ids=ids).logits[0, -1].argmax())
    return scorer._tok.decode([top]), top in FAM._special_ids(scorer._tok)


def score_shard(cfg: dict, tag: str, spec, select: Callable[[list[dict]], list[dict]],
                arms: tuple[str, ...] | None = None, envcheck: bool = False, exact_gpu: list[str] | None = None,
                shard: tuple[int, int] | None = None) -> int:
    """The scoring loop. `shard` (index, count) names the run's shard folder when `select` already returns exactly
    this job's records (the environment check, which mirrors round 2's shards)."""
    run_id = cfg["run_id"]
    require_filed(run_id)
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"records not verified: {files}", ok=False)
        return 1
    rpath, spath, mpath = files
    env = FAM.require_environment(spec, envcheck=envcheck)
    env["resources"] = FAM.require_resources(spec)
    env["gpu"] = gpu_check(tag, exact=exact_gpu)
    env["freeze_commit"] = G.require_freeze_commit(run_id=run_id)  # D-G1: only textcraft_r3x_ runs are exempt
    if shard is None:
        i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
        n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    else:
        i, n = shard
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n, "model_spec": spec.record(), "system_text": R.SYSTEM,
           "scored_tag": tag, "arms": list(arms) if arms else "all"}
    from wf.util import runs_root

    G.refuse_after_gate_failure(runs_root() / run_id)  # review 2, S5: a filed gate failure is final
    d = _init_run(run_id, cfg, inputs=(rpath, spath, mpath))
    all_records = load_records(rpath)
    records = select(all_records) if shard is not None else shard_records(select(all_records), i, n)
    shots = P.fewshot_examples(spath)
    hashes = {r["id"]: R.item_hash(r, shots) for r in records}
    jobs = [(r, a) for r in records for a in r["arms"] if arms is None or a in arms]
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(s.get("model") != tag or hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved scores belong to different records, prompts or models; rerun without --resume.")
    done = {(s["transition_id"], s["condition"]) for s in saved}
    todo = [(r, a) for r, a in jobs if (r["id"], a) not in done]
    if todo:
        try:
            scorer = FAM.open_scorer(spec, tag)
        except RuntimeError as e:  # the strict loading check (logodds.check_loading): filed, so the model is "not scored"
            if "loading check failed" not in str(e):
                raise
            write_json(d / "tokenization_checks.json", {"ok": False, "loading_failed": str(e)[:2000], "environment": env})
            write_status(run_id, f"stopped before scoring: loading check failed for {tag}", ok=False)
            return 1
        if not run_gate(d, scorer, spec, tag, records, all_records, shots, env):
            write_status(run_id, f"stopped before scoring: gate failed for {tag} (see {d / 'tokenization_checks.json'})",
                         ok=False)
            scorer.close()
            return 1
        for rec, arm in todo:
            sc = score_one(scorer, spec, rec, arm, shots)
            extra = {}
            if cfg.get("smoke") and R.parse_arm(arm)[0]:  # the format diagnostic of the smoke runs (design 14.M3)
                tok, special = top_next_token(scorer, R3.model_messages(spec, rec, arm, shots))
                extra = {"top_token": tok, "top_token_special": special}
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": rec["pair_id"], "pair_id": rec["pair_id"],
                                                "kind": rec["kind"], "population": rec["population"],
                                                "role": rec["role"], "item_hash": hashes[rec["id"]], **extra}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(records)} records, {len(jobs)} scores ({tag})", ok=False,
                 reported="shard complete")
    return 0


# ---------------- the runs ----------------


def run_textcraft_r3_score(cfg: dict) -> int:
    """`textcraft_r3_score_<tag>` (every record) and `textcraft_r3_smoke_<tag>` (cfg smoke: true, 20 seeded records)."""
    tag = cfg["model"]
    if tag not in R3.MODELS:
        raise SystemExit(f"unknown round-3 model {tag!r}; known: {list(R3.MODELS)}")
    if tag == R3.GEMMA_FALLBACK:  # design 14.M3: the fallback runs only when the rule fired on 12B's smoke record
        from wf.util import runs_root

        slot = G.gemma_slot(runs_root())
        if slot["slot"] != R3.GEMMA_FALLBACK:
            raise SystemExit(f"{tag} runs only if the Gemma fallback fired; the rule gives {slot['slot']!r} "
                             f"(triggers {slot['triggers']})")
    select = smoke_subset if cfg.get("smoke") else (lambda rs: rs)
    return score_shard(cfg, tag, R3.MODELS[tag], select)


def envcheck_job_records(records: list[dict], job: dict) -> list[dict]:
    """The seeded environment-check records that lie in this job's round-2 shards, in round 2's own partition."""
    keep = {r["id"] for r in envcheck_subset(records)}
    ids = {r["id"] for i in job["r2_shards"] for r in shard_records(records, i, job["n_r2_shards"])}
    return [r for r in records if r["id"] in keep and r["id"] in ids]


def run_textcraft_r3_envcheck(cfg: dict) -> int:
    """`textcraft_r3_envcheck_<tag>` (design 14.M4): one job of the committed plan (WF_ENVCHECK_JOB or cfg job): the
    seeded records of the planned round-2 shards, every arm, the earlier model as serve2 loads it, on exactly the
    planned GPU name (the job refuses any other)."""
    tag = cfg["model"]
    if tag not in R3.ENVCHECK_MODELS:
        raise SystemExit(f"the environment check rescores {R3.ENVCHECK_MODELS}, not {tag!r}")
    spec = FAM.envcheck_spec(tag)
    job = G.plan_job(tag, int(os.environ.get("WF_ENVCHECK_JOB") or cfg.get("job", 0)), cfg.get("plan"))
    return score_shard({**cfg, "plan_job": job}, spec.tag, spec, lambda rs: envcheck_job_records(rs, job), envcheck=True,
                       exact_gpu=[job["gpu"]], shard=(job["job"], job["n_jobs"]))


def run_textcraft_r3_rescore(cfg: dict) -> int:
    """`textcraft_r3_rescore_<tag>` (only if the environment check fails): an earlier model rescored in serve2 on every
    record, for analysis (b) serve2 against serve2."""
    tag = cfg["model"]
    if tag not in R3.RESCORE_MODELS:
        raise SystemExit(f"the fallback rescores {R3.RESCORE_MODELS}, not {tag!r}")
    spec = FAM.envcheck_spec(tag)
    return score_shard(cfg, spec.tag, spec, lambda rs: rs, envcheck=True, exact_gpu=[R3.RESCORE_GPU])


FORMAT_CHECKS = {"placement": ("gemma4-12b-it", FAM.first_turn_spec), "nopreserve": ("qwen3.8-27b", FAM.nopreserve_spec)}


def run_textcraft_r3_format(cfg: dict) -> int:
    """`textcraft_r3_placement_gemma4-12b-it` / `textcraft_r3_nopreserve_qwen3.8-27b` (cfg check: placement or
    nopreserve): 300 seeded main failures, aligned and swapped in both formats, under the variant's tag."""
    check = cfg["check"]
    model, make = FORMAT_CHECKS[check]
    if cfg["model"] != model:
        raise SystemExit(f"the {check} check is for {model}, not {cfg['model']!r}")
    spec = make(R3.MODELS[model])
    return score_shard(cfg, spec.tag, spec, format_subset, arms=R3.FORMAT_ARMS)
