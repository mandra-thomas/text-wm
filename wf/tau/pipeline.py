"""Runs for the τ-bench retail replication (notes/tau_design.md §9).

tau_build (CPU) -> tau_verify (CPU) -> tau_score / tau_score_smoke (GPU, sharded by user) ->
tau_analysis / tau_smoke_analysis (CPU). tau_contamination (GPU) is the exploratory check of §10.
Scoring refuses items that tau_verify has not passed on the exact files, or whose sha256 differs from the one
pinned in the scoring config.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation
from wf.runs import _finished, _init_run, load_config
from wf.scoring.logodds import LogoddsScorer
from wf.tau import items as I
from wf.tau.prompts import CONDITIONS, MENTION_CONDITIONS, build_messages, render_history
from wf.tau.retail import check_vendor, pristine_sha256, vendor_manifest
from wf.tau.verify import fresh_process_sample, sample_ids, verify_sets
from wf.util import append_jsonl, file_sha, read_jsonl, require_filed, runs_root, write_json, write_status

MODELS = {"qwen3-8b-fewshot": C.WM_FEWSHOT_MODEL, "llama3.1-8b-fewshot": C.WM_FEWSHOT_LLAMA}
CO_PRIMARY = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
FIRST_PLUS, FIRST_MINUS = '{"', "Error"
READOUTS = ("first", "yn", "full")  # "target" (the manipulation check) is scheduled by jobs_for
OUTPUT_FILES = ("items.jsonl", "natural_items.jsonl", "fewshot.json")
SUPPORT = C.DELTA_L_SUPPORT_NATS  # 0.5 nats, the ALFWorld threshold
P_MAX = C.PERMUTATION_P_MAX


def _build_dir(cfg: dict) -> Path:
    return runs_root() / cfg.get("build_run", "tau_build")


def load_built(cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    d = _build_dir(cfg)
    gen = read_jsonl(d / "items.jsonl")
    nat = read_jsonl(d / "natural_items.jsonl") if (d / "natural_items.jsonl").exists() else []
    few = json.loads((d / "fewshot.json").read_text(encoding="utf-8"))
    return gen, nat, few


def _sha_of_outputs(d: Path) -> dict[str, str]:
    return {name: file_sha(d / name) for name in OUTPUT_FILES if (d / name).exists()}


# ---------------- build ----------------


def _natural_paths(cfg: dict) -> tuple[list[Path] | None, str | None]:
    """The released trajectory files, checked against the config's sha256 values; (None, why) if unusable."""
    files = cfg.get("natural_files") or {}
    root = Path(os.environ.get("TAU_NATURAL_DIR") or cfg.get("natural_dir") or "")
    if not files:
        return None, "no natural_files in the config"
    paths = [root / name for name in sorted(files)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return None, f"missing: {missing}"
    bad = [p.name for p in paths if file_sha(p) != files[p.name]]
    if bad:
        return None, f"sha256 differs: {bad}"
    return paths, None


def strata_counts(items: list[dict]) -> dict[str, Any]:
    main = [t for t in items if t["main_pool"]]
    users = lambda xs: len({t["user_id"] for t in xs})
    by = lambda xs, f: {str(k): {"items": len(v), "users": users(v)} for k, v in sorted(_group(xs, f).items(), key=lambda kv: str(kv[0]))}
    return {"all": {"items": len(items), "users": users(items)},
            "main": {"items": len(main), "users": users(main)},
            "main_status_matched": {"items": sum(t["status_matched"] for t in main), "users": users([t for t in main if t["status_matched"]])},
            "main_by_class": by(main, lambda t: t["cls"]),
            "main_by_swapped_achievable": by(main, lambda t: t["swapped_achievable"]),
            "main_by_class_and_matched": by(main, lambda t: f"{t['cls']}|matched={t['status_matched']}"),
            "main_by_action": by(main, lambda t: t["action"]["name"]),
            "unachievable_by_class_action": by([t for t in items if not t["main_pool"]], lambda t: f"{t['cls']}|{t['action']['name']}")}


def _group(xs: list[dict], f) -> dict:
    out: dict = defaultdict(list)
    for x in xs:
        out[f(x)].append(x)
    return out


def run_tau_build(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    bad = check_vendor()
    if bad:
        write_status(run_id, f"vendored files differ from MANIFEST.json: {bad}", ok=False)
        return 1
    paths, why = _natural_paths(cfg)
    if paths is None and cfg.get("natural_required", True):
        write_status(run_id, f"natural trajectories unusable: {why}", ok=False)
        return 1
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(paths or ()))
    built = I.build_all(paths, cfg.get("max_users"))  # max_users: tests only
    gen = sorted(built["generated"], key=lambda t: t["id"])
    nat = sorted(built["natural"], key=lambda t: t["id"])
    (d / "items.jsonl").write_text("".join(json.dumps(t) + "\n" for t in gen), encoding="utf-8")
    (d / "natural_items.jsonl").write_text("".join(json.dumps(t) + "\n" for t in nat), encoding="utf-8")
    (d / "fewshot.json").write_text(json.dumps(built["fewshot"], indent=1) + "\n", encoding="utf-8")
    write_json(d / "build_log.json", built["log"])
    manifest = {"vendor": vendor_manifest(), "pristine_db_state_sha256": pristine_sha256(),
                "natural_inputs": {p.name: file_sha(p) for p in paths or ()}, "natural_skipped": why if paths is None else None,
                "outputs": _sha_of_outputs(d), "strata_generated": strata_counts(gen),
                "strata_natural": strata_counts(nat) if nat else None,
                "fewshot_users": [ex["user_id"] for ex in built["fewshot"]]}
    write_json(d / "manifest.json", manifest)
    s = manifest["strata_generated"]
    write_status(run_id, f"generated {s['all']['items']} items ({s['main']['items']} main, {s['main']['users']} users; "
                         f"status-matched {s['main_status_matched']['items']}); natural {len(nat)}"
                         + (f" (skipped: {why})" if paths is None else "")
                         + f"\nitems.jsonl sha256 {manifest['outputs']['items.jsonl']}",
                 ok=False, reported="items built")
    return 0


# ---------------- verify ----------------


def run_tau_verify(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    build = cfg.get("build_run", "tau_build")
    if not _finished(build):
        write_status(run_id, f"{build} has not finished at this commit", ok=False)
        return 1
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(_build_dir(cfg) / n for n in OUTPUT_FILES
                                                                   if (_build_dir(cfg) / n).exists()))
    gen, nat, few = load_built(cfg)
    report = verify_sets(gen, nat, few)
    sample = sample_ids(gen + nat, int(cfg.get("fresh_sample", 100)))
    gen_ids = {t["id"] for t in gen}
    fresh = {"generated": fresh_process_sample(_build_dir(cfg) / "items.jsonl", [i for i in sample if i in gen_ids])}
    nat_ids = [i for i in sample if i not in gen_ids]
    if nat_ids:
        fresh["natural"] = fresh_process_sample(_build_dir(cfg) / "natural_items.jsonl", nat_ids)
    report["fresh_process"] = fresh
    report["passed"] = report["passed"] and all(v.get("passed") for v in fresh.values())
    report["files"] = _sha_of_outputs(_build_dir(cfg))
    write_json(d / "verify.json", report)
    write_status(run_id, f"{report['n_generated']} generated and {report['n_natural']} natural items; "
                         f"{report['n_failed_items']} failed; problems {report['problems']}; fresh-process sample "
                         f"{ {k: v.get('passed') for k, v in fresh.items()} }", ok=report["passed"])
    return 0 if report["passed"] else 1


def verified(cfg: dict) -> tuple[bool, str]:
    """Has tau_verify passed on exactly the files scoring would read, and does every one of them match the sha256
    pinned in the scoring config (items, natural items and few-shot examples)?"""
    vrun = cfg.get("verify_run", "tau_verify")
    if not _finished(vrun, tuple(cfg.get("accept_commits", {}).get(vrun, ()))):
        return False, f"{vrun} has not passed"
    rep = json.loads((runs_root() / vrun / "verify.json").read_text(encoding="utf-8"))
    now = _sha_of_outputs(_build_dir(cfg))
    if not rep.get("passed"):
        return False, "verify.json does not say passed"
    if rep.get("files") != now:
        return False, "verify.json does not cover the current item files"
    pins = cfg.get("expected_outputs_sha256")
    if not pins:  # a scoring config without pins is an error (amendment after review 2), never a pass
        return False, "the scoring config pins no sha256 values (expected_outputs_sha256)"
    for name in OUTPUT_FILES:
        if pins.get(name) is None:
            return False, f"no pinned sha256 for {name}"
        if now.get(name) != pins[name]:
            return False, f"{name} sha256 {now.get(name)} differs from the pinned {pins[name]}"
    return True, ""


# ---------------- scoring ----------------


def prompt_chars(t: dict) -> int:
    """Length of the item's rendered history and call, in characters (a tokenizer-free proxy for prompt length)."""
    return len(render_history(t["history"])) + len(json.dumps(t["action"]))


def scoring_sets(cfg: dict) -> list[dict]:
    gen, nat, _ = load_built(cfg)
    out = [t for t in gen if t["main_pool"]]
    if cfg.get("score_unachievable"):
        out += [t for t in gen if not t["main_pool"]]
    if cfg.get("score_natural", True):
        out += nat
    if cfg.get("limit_items"):
        out = sorted(out, key=lambda t: I.h(t["id"], "smoke"))[: int(cfg["limit_items"])]
        if cfg.get("add_longest"):  # the smoke run (review 2): also the longest generated main item and natural item
            ids = {t["id"] for t in out}
            for pool in ([t for t in gen if t["main_pool"]], nat):
                if pool:
                    longest = max(pool, key=lambda t: (prompt_chars(t), t["id"]))
                    if longest["id"] not in ids:
                        out.append(longest)
                        ids.add(longest["id"])
    return out


def is_primary(t: dict) -> bool:
    """The confirmatory population (amendment of 2026-11-20): generated main-population items whose comparison
    order has the target's status."""
    return t["source"] == "generated" and t["main_pool"] and t["status_matched"]


EXTRA_ARMS = {"rules_only": "rules_arm_items", "matched": "matched_arm_items"}  # arm -> config key of its size


def arm_ids(items: list[dict], n: int) -> set[str]:
    """An exploratory arm's items: the first n primary items of the whole scoring set in hash(id, "rules") order.
    The rules-only and the ALFWorld-matched arms both draw this way, so with the same n they hold the same items."""
    return set(sorted((t["id"] for t in items if is_primary(t)), key=lambda i: I.h(i, "rules"))[: max(0, int(n))])


def extra_arm_ids(items: list[dict], cfg: dict) -> dict[str, set[str]]:
    """Each exploratory arm's items, drawn from the whole scoring set (never per shard)."""
    return {arm: arm_ids(items, cfg.get(key, 0)) for arm, key in EXTRA_ARMS.items()}


def shard_by_user(items: list[dict], i: int, n: int) -> list[dict]:
    keep = set(sorted({t["user_id"] for t in items})[i::n])
    return [t for t in items if t["user_id"] in keep]


def item_hash(t: dict) -> str:
    fields = {k: t.get(k) for k in ("history", "action", "failure_reply", "success_twin", "task_aligned",
                                     "task_swapped", "task_neutral", "task_mention_aligned", "task_mention_swapped",
                                     "aligned_witness", "witness_output", "witness_failure")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def jobs_for(t: dict, cfg: dict, arms: dict[str, set[str]] | None = None) -> list[tuple[str, str, str, str]]:
    """Every scored (arm, action, readout, condition) of one item (design §6.0, amended 2026-11-20)."""
    readouts = tuple(cfg.get("readouts", READOUTS))
    conds = [c for c in CONDITIONS if c != "swapped" or t.get("task_swapped")]
    single = [r for r in readouts if r in ("first", "yn")]
    jobs = [("main", "fail", r, c) for r in readouts for c in conds]
    if cfg.get("mention", True):
        jobs += [("main", "fail", r, c) for r in single for c in MENTION_CONDITIONS if t.get(f"task_{c}")]
    if cfg.get("target_check", True):
        jobs += [("main", "fail", "target", c) for c in conds if c != "neutral"]
    if cfg.get("witness", True) and t.get("aligned_witness"):
        jobs += [("main", "witness", r, c) for r in readouts for c in conds]
    for arm in EXTRA_ARMS:
        if t["id"] in (arms or {}).get(arm, ()):
            jobs += [(arm, "fail", r, c) for r in single for c in conds]
    return jobs


def token_report(scorer: LogoddsScorer, sample: dict) -> dict[str, Any]:
    """Are '{"', 'Error', 'Yes' and 'No' single tokens, and do '{"' and 'Error' open the tokenised replies (twin,
    failure, witness output) with a generation prompt ending in a newline (design §6.1)? Mock: trivially yes."""
    if scorer.mock:
        return {"mock": True, "first_token_ok": True}
    tok = scorer._tok
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    rep = {v: enc(v) for v in (FIRST_PLUS, FIRST_MINUS, " " + FIRST_PLUS, " " + FIRST_MINUS, "Yes", "No")}
    twin, fail = enc(sample["success_twin"]), enc(sample["failure_reply"])
    wit = enc(sample["witness_output"]) if sample.get("witness_output") else rep[FIRST_PLUS]
    gen_prompt = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    ok = (all(len(rep[k]) == 1 for k in (FIRST_PLUS, FIRST_MINUS, "Yes", "No")) and twin[:1] == rep[FIRST_PLUS]
          and wit[:1] == rep[FIRST_PLUS] and fail[:1] == rep[FIRST_MINUS] and gen_prompt.endswith("\n"))
    return {"ids": {k: v for k, v in rep.items()}, "twin_first": twin[:3], "failure_first": fail[:3],
            "witness_first": wit[:3], "generation_prompt_tail": gen_prompt[-30:], "first_token_ok": ok}


def _score_job(scorer: LogoddsScorer, t: dict, fewshot: list[dict], job: tuple[str, str, str, str]):
    arm, action, readout, cond = job
    if readout == "first":
        return scorer.score_one_token_pair(t["id"], cond, build_messages(t, cond, fewshot, action=action, arm=arm), FIRST_PLUS, FIRST_MINUS)
    if readout in ("yn", "target"):
        cue = "yn" if readout == "yn" else "target"
        return scorer.score_one_token_pair(t["id"], cond, build_messages(t, cond, fewshot, cue=cue, action=action, arm=arm), "Yes", "No")
    if readout == "full":
        plus, minus = ((t["success_twin"], t["failure_reply"]) if action == "fail" else (t["witness_output"], t["witness_failure"]))
        return scorer.score_pair(t["id"], cond, build_messages(t, cond, fewshot, action=action, arm=arm), plus, minus)
    raise ValueError(readout)


def run_tau_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    ok, why = verified(cfg)
    if not ok:
        write_status(run_id, f"items not verified: {why}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(_build_dir(cfg) / name for name in OUTPUT_FILES if (_build_dir(cfg) / name).exists()))
    _, _, fewshot = load_built(cfg)
    every = scoring_sets(cfg)
    arms = extra_arm_ids(every, cfg)
    items = shard_by_user(every, i, n)
    hashes = {t["id"]: item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {score_key(s) for s in saved}
    for tag in cfg.get("models", CO_PRIMARY):
        todo = [(t, job) for t in items for job in jobs_for(t, cfg, arms) if (tag, t["id"], *job) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(MODELS[tag], model_tag=tag)
        tokens_path = d / f"tokens_{tag}.json"
        report = token_report(scorer, next((t for t in items if t.get("witness_output")), items[0]))
        write_json(tokens_path, report)
        if not report.get("first_token_ok"):
            scorer.close()
            write_status(run_id, f"{tag}: the token check fails ({tokens_path.name}); the first-token readout is not "
                                 "valid for this model, and the design has no fallback (amendment of 2026-11-20)", ok=False)
            return 1
        for t, job in todo:
            t0 = time.perf_counter()
            sc = _score_job(scorer, t, fewshot, job)
            arm, action, readout, _ = job
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "arm": arm, "action": action, "readout": readout,
                                                "user_id": t["user_id"], "item_hash": hashes[t["id"]],
                                                "seconds": round(time.perf_counter() - t0, 4)}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} items, {len({t['user_id'] for t in items})} users",
                 ok=False, reported="shard complete")
    return 0


def score_key(s: dict) -> tuple[str, str, str, str, str, str]:
    return (s["model"], s["transition_id"], s.get("arm", "main"), s.get("action", "fail"), s["readout"], s["condition"])


# ---------------- analysis ----------------


def value(rec: dict, readout: str, scale: str = "logit") -> float:
    if readout == "full_norm":
        return rec["logp_plus"] / rec["n_tokens_plus"] - rec["logp_minus"] / rec["n_tokens_minus"]
    if scale == "prob":
        return 1.0 / (1.0 + math.exp(-rec["L"]))
    return rec["L"]


def cluster_of(t: dict, by: str) -> str:
    """Users (the design's cluster), or the target item's product for the product-clustered sensitivity analysis;
    an item without a target item (gift-card payment changes) is its own product cluster."""
    if by == "user":
        return t["user_id"]
    if by == "product":
        return t.get("target_product_id") or f"no-product:{t['id']}"
    raise ValueError(by)


def paired(by: dict, items: list[dict], readout: str, a: str, b: str, *, action: str = "fail", action_b: str | None = None,
           arm: str = "main", cluster: str = "user", scale: str = "logit", inference: bool = True) -> dict[str, Any]:
    """Mean of value(a) - value(b) over items, with a clustered CI and a clustered sign-flip p. `a` and `b` are
    conditions; `action_b` (default `action`) lets the discrimination check compare the witness with the failing call."""
    base = "full" if readout.startswith("full") else readout
    diffs, clusters = [], []
    for t in items:
        x, y = by.get((t["id"], arm, action, base, a)), by.get((t["id"], arm, action_b or action, base, b))
        if x is None or y is None:
            continue
        v = value(x, readout, scale) - value(y, readout, scale)
        if v == v:
            diffs.append(v)
            clusters.append(cluster_of(t, cluster))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    users = {t["user_id"] for t in items if (t["id"], arm, action, base, a) in by and (t["id"], arm, action_b or action, base, b) in by}
    out = {"estimate": mean(diffs), "n_items": len(diffs), "n_users": len(users), "n_clusters": len(set(clusters)),
           "cluster": cluster, "share_positive": mean([1.0 if v > 0 else 0.0 for v in diffs])}
    if inference:
        boot = cluster_bootstrap(mean, diffs, clusters, seed=C.RNG_SEED)
        out.update({"ci": [boot["ci_low"], boot["ci_high"]], "perm_p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=clusters)})
    return out


def contrast_of_effects(by: dict, items: list[dict], readout: str, first: tuple[str, str, str], second: tuple[str, str, str],
                        arms: tuple[str, str] = ("main", "main")) -> dict[str, Any]:
    """Per item, (the a - b effect of `first`) minus (that of `second`), each given as (action, condition a,
    condition b) and scored in `arms` (default both main); user-clustered CI and sign-flip p. Used for failing call
    against witness, write request against mention, and main arm against the ALFWorld-matched arm."""
    diffs, users = [], []
    for t in items:
        keys = [(t["id"], arm, act, readout, c) for arm, (act, a, b) in zip(arms, (first, second)) for c in (a, b)]
        if not all(k in by for k in keys):
            continue
        L = [by[k]["L"] for k in keys]
        v = (L[0] - L[1]) - (L[2] - L[3])
        if v == v:
            diffs.append(v)
            users.append(t["user_id"])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    boot = cluster_bootstrap(mean, diffs, users, seed=C.RNG_SEED)
    return {"estimate": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]], "n_items": len(diffs), "n_users": len(set(users)),
            "perm_p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=users)}


def effects(by: dict, items: list[dict], readout: str, *, action: str = "fail", arm: str = "main", inference: bool = True) -> dict[str, Any]:
    kw = {"action": action, "arm": arm, "inference": inference}
    return {"aligned_minus_swapped": paired(by, items, readout, "aligned", "swapped", **kw),
            "aligned_minus_neutral": paired(by, items, readout, "aligned", "neutral", **kw),
            "swapped_minus_neutral": paired(by, items, readout, "swapped", "neutral", **kw)}


def tau_verdict(est: float, ci: list[float], holm_p: float) -> str:
    """The design's decision rule for one model (notes/tau_design.md §6.3), two-sided."""
    if any(v != v for v in (est, ci[0], ci[1], holm_p)):
        return "not computable"
    excludes_zero = ci[0] > 0 or ci[1] < 0
    if excludes_zero and holm_p < P_MAX:
        if est >= SUPPORT:
            return "replicates (ALFWorld direction)"
        if est <= -SUPPORT:
            return "reversed (naming sharpens failure prediction)"
        return "small effect, direction " + ("positive" if est > 0 else "negative")
    if -SUPPORT < ci[0] and ci[1] < SUPPORT:
        return "no effect larger than 0.5 nats"
    return "inconclusive"


CHECK_MIN_NATS = 1.0  # manipulation and discrimination checks: mean at least 1 nat and the CI above 0


def check_passes(block: dict | None) -> bool:
    if not block or block.get("n_items", 0) == 0:
        return False
    est, ci = block["estimate"], block.get("ci", [float("nan")] * 2)
    return est == est and ci[0] == ci[0] and est >= CHECK_MIN_NATS and ci[0] > 0


POSITIVE = ("replicates", "small effect, direction positive")
NEGATIVE = ("reversed", "small effect, direction negative")


def joint_verdict(per_model: dict[str, str]) -> str:
    """Both co-primary models must agree for a joint claim (amendments of 2026-11-20 after reviews 1 and 2)."""
    v = list(per_model.values())
    rep = sum(x.startswith("replicates") for x in v)
    rev = sum(x.startswith("reversed") for x in v)
    pos = sum(x.startswith("small effect, direction positive") for x in v)
    neg = sum(x.startswith("small effect, direction negative") for x in v)
    null = sum(x.startswith("no effect") for x in v)
    if (rep and (rev or neg)) or (rev and pos):
        return "mixed (the models disagree in direction)"
    if rep == len(v):
        return "replicates (both models)"
    if rev == len(v):
        return "reversed (both models)"
    if null == len(v):
        return "no effect larger than 0.5 nats (both models)"
    if rep and rep + pos == len(v):
        return "ALFWorld direction in both models, size met by one (not a replication)"
    if rev and rev + neg == len(v):
        return "reversed direction in both models, size met by one"
    if rep or rev:
        return "one model only (not a replication)"
    return "inconclusive"


def _ci_above(block: dict | None) -> bool:
    return bool(block) and block.get("n_items", 0) > 0 and block["ci"][0] > 0


def _ci_inside(block: dict | None) -> bool:
    return bool(block) and block.get("n_items", 0) > 0 and -SUPPORT < block["ci"][0] and block["ci"][1] < SUPPORT


def outcome_reading(verdict: str, disc: bool | None, fail_minus_witness: dict | None = None,
                    write_minus_mention: dict | None = None) -> dict[str, Any]:
    """The outcome table of design A9 (as amended after review 2), applied to one model's primary results:
    discrimination, specificity to failing calls (fail minus witness) and the mention pair (write minus mention)."""
    out: dict[str, Any] = {"discrimination_holds": disc}
    if fail_minus_witness is not None:
        out["specificity"] = ("specific to failing calls (fail minus witness above 0)" if _ci_above(fail_minus_witness)
                              else "not specific to failing calls (fail minus witness about 0, negative or uncertain)")
    if write_minus_mention is not None:
        out["mention"] = ("the request to act adds to naming the order (write minus mention above 0)" if _ci_above(write_minus_mention)
                          else "naming the order is enough; the request to act adds nothing (write minus mention inside ±0.5)"
                          if _ci_inside(write_minus_mention) else "uncertain")
    if verdict.startswith(POSITIVE):
        size = "" if verdict.startswith("replicates") else " (below the 0.5-nat size)"
        if not disc:
            out["reading"] = ("The task shifts predictions, but the model is not shown to use the visible record: not read "
                              "as the task overriding visible evidence" + size)
        elif _ci_above(fail_minus_witness):
            out["reading"] = ("The task overrides a visible fact about the named object, more for failing calls than "
                              "for working ones: the ALFWorld effect in its stronger form" + size)
        else:
            out["reading"] = ("A general task-consistency bias: naming the order raises the success belief of calls on it "
                              "whether they fail or not; reported as goal leakage, not as an override specific to "
                              "failures" + size)
    elif verdict.startswith(NEGATIVE):
        size = "" if verdict.startswith("reversed") else " (below the 0.5-nat size)"
        out["reading"] = ("Naming the object sharpens failure prediction (attention to its record); the ALFWorld effect "
                          "would then need the deciding fact to lie outside the named object" + size if disc else
                          "A shift toward failure, but the model is not shown to use the visible record: inconclusive" + size)
    elif verdict.startswith("no effect"):
        out["reading"] = "No leak of more than 0.5 nats, with the task line read and the visible record used"
    else:
        out["reading"] = "Inconclusive: " + verdict
    return out


def verdicts(primary: dict[str, dict], checks: dict[str, dict] | None = None) -> dict[str, Any]:
    """Per-model verdicts on the primary contrast, Holm over the two models. A null needs the manipulation check (the
    task line is read) and discrimination (the visible record is used); a reversal needs discrimination; every
    non-null effect is read through the outcome table (outcome_reading)."""
    checks = checks or {}
    adjusted = holm({m: primary[m]["perm_p"] if m in primary else 1.0 for m in CO_PRIMARY})
    per = {}
    for m in CO_PRIMARY:
        if m not in primary:
            per[m] = {"verdict": "not scored", "holm_p": adjusted[m]}
            continue
        v = tau_verdict(primary[m]["estimate"], primary[m]["ci"], adjusted[m])
        c = checks.get(m, {})
        manip, disc = c.get("manipulation_check_passed"), c.get("discrimination_holds")
        if v.startswith("no effect") and not manip:
            v = "inconclusive (null, but the task line is not shown to be read)"
        elif v.startswith(("no effect", "reversed")) and not disc:
            v = (f"inconclusive ({'null' if v.startswith('no effect') else 'reversed'}, but the model is not shown to "
                 "discriminate failing from working calls)")
        reading = outcome_reading(v, disc, c.get("fail_minus_witness"), c.get("write_minus_mention"))
        per[m] = {"verdict": v, "holm_p": adjusted[m], "manipulation_check_passed": manip, "discrimination_holds": disc,
                  "reading": reading["reading"], "outcome_table": reading}
    return {"per_model": per, "overall": joint_verdict({m: x["verdict"] for m, x in per.items()})}


def sensitivity_verdicts(sens: dict[str, dict], primary: dict[str, dict]) -> dict[str, Any]:
    """The few-shot tool sensitivity analysis (review 2): the §6.3 rule (Holm over the two models) on the primary
    population without the classes whose tool is a few-shot example's, beside the primary's ungated verdict. A
    different category is reported as the primary result being sensitive to few-shot tool matching."""
    adj_s = holm({m: sens[m]["perm_p"] if m in sens else 1.0 for m in CO_PRIMARY})
    adj_p = holm({m: primary[m]["perm_p"] if m in primary else 1.0 for m in CO_PRIMARY})
    out = {}
    for m in CO_PRIMARY:
        if m in sens and m in primary:
            v = tau_verdict(sens[m]["estimate"], sens[m]["ci"], adj_s[m])
            p = tau_verdict(primary[m]["estimate"], primary[m]["ci"], adj_p[m])
            out[m] = {"verdict": v, "primary_ungated_verdict": p, "same_category": v == p}
    return out


def matched_reading(main_verdict: str, matched_verdict: str) -> str:
    """The ALFWorld-matched arm's prestated reading (amendment after review 2), from the confirmatory main-arm verdict
    and the matched arm's own verdict under the same rule."""
    eff = lambda v: v.startswith(POSITIVE)
    null = lambda v: v.startswith("no effect")
    if eff(main_verdict) and eff(matched_verdict):
        return "effect under both prompts: stating that the backend does not see the task does not remove it"
    if null(main_verdict) and eff(matched_verdict):
        return ("null under the main prompt but an effect under the matched prompt: an explicit statement that the backend "
                "does not see the task (with the other-order few-shot examples) suppresses the effect here")
    if null(main_verdict) and null(matched_verdict):
        return "no effect larger than 0.5 nats under either prompt: the null does not rest on the debiasing statement"
    if eff(main_verdict) and null(matched_verdict):
        return "effect under the main prompt only: not predicted; reported, not interpreted"
    return "inconclusive"


def matched_verdicts(matched: dict[str, dict], main_per_model: dict[str, dict]) -> dict[str, Any]:
    adjusted = holm({m: matched[m]["perm_p"] if m in matched else 1.0 for m in CO_PRIMARY})
    out = {}
    for m in CO_PRIMARY:
        if m not in matched:
            continue
        v = tau_verdict(matched[m]["estimate"], matched[m]["ci"], adjusted[m])
        main_v = main_per_model.get(m, {}).get("verdict", "not scored")
        out[m] = {"verdict": v, "holm_p": adjusted[m], "main_verdict": main_v, "reading": matched_reading(main_v, v)}
    return out


def flip_shares(by: dict, items: list[dict]) -> dict[str, float]:
    """Shares of items whose first-token L changes sign between the swapped and the aligned task, each way."""
    pairs = [(by[(t["id"], "main", "fail", "first", "aligned")]["L"], by[(t["id"], "main", "fail", "first", "swapped")]["L"])
             for t in items if (t["id"], "main", "fail", "first", "aligned") in by and (t["id"], "main", "fail", "first", "swapped") in by]
    if not pairs:
        return {"n": 0, "toward_success": float("nan"), "toward_failure": float("nan"), "either": float("nan")}
    up = sum(1 for a, s in pairs if a > 0 > s) / len(pairs)
    down = sum(1 for a, s in pairs if s > 0 > a) / len(pairs)
    return {"n": len(pairs), "toward_success": up, "toward_failure": down, "either": up + down}


def _agreement(by: dict, items: list[dict]) -> dict[str, Any]:
    """Primary pass vs the full pass's first-token log-probabilities (design §6.1): within 0.05 nats?"""
    gaps = []
    for t in items:
        for c in CONDITIONS:
            f, g = by.get((t["id"], "main", "fail", "first", c)), by.get((t["id"], "main", "fail", "full", c))
            if f and g:
                gaps.append(max(abs(f["logp_plus"] - g["logp_first_plus"]), abs(f["logp_minus"] - g["logp_first_minus"])))
    within = sum(1 for x in gaps if x <= 0.05)
    return {"n": len(gaps), "share_within_0.05": within / len(gaps) if gaps else float("nan"),
            "max_gap": max(gaps) if gaps else float("nan"), "ok": bool(gaps) and within / len(gaps) >= 0.99}


PRESPECIFIED_CLASSES = ("status", "unavailable")
DESCRIPTIVE_CLASSES = ("wrong_return_pm", "gc_balance")
# Classes whose failing tool is the tool of a few-shot example (return: example 3; payment change: example 4), so the
# example's (task-shape, outcome) cell can be matched by tool; the sensitivity analysis of review 2 leaves them out.
FEWSHOT_TOOL_CLASSES = ("wrong_return_pm", "gc_balance")
KEEPS_STATUS = ("modify_pending_order_address", "modify_pending_order_payment")


def twin_status_incoherent(t: dict) -> bool:
    """Status items on the two tools that keep the status: the twin shows a non-pending status after a successful
    change, which the backend could never print (561 unachievable-set items; documented 2026-11-20)."""
    return t["cls"] == "status" and t["action"]["name"] in KEEPS_STATUS


def population_block(by: dict, pop: list[dict]) -> dict[str, Any]:
    """Every readout on one population: the failing call's effects, and its practical size."""
    out: dict[str, Any] = {"n_items": len(pop), "n_users": len({t["user_id"] for t in pop})}
    for r in ("first", "yn", "full", "full_norm"):
        out[r] = effects(by, pop, r)
    out["first_probability_shift"] = paired(by, pop, "first", "aligned", "swapped", scale="prob")
    out["first_product_clustered"] = paired(by, pop, "first", "aligned", "swapped", cluster="product")
    out["first_flips"] = flip_shares(by, pop)
    out["first_without_fewshot_tool_classes"] = effects(by, [t for t in pop if t["cls"] not in FEWSHOT_TOOL_CLASSES], "first")
    out["by_class"] = {k: effects(by, v, "first") for k, v in sorted(_group(pop, lambda t: t["cls"]).items())
                       if k in PRESPECIFIED_CLASSES}
    out["by_class_descriptive"] = {k: effects(by, v, "first", inference=False) for k, v in sorted(_group(pop, lambda t: t["cls"]).items())
                                   if k in DESCRIPTIVE_CLASSES}
    return out


def discrimination(by: dict, items: list[dict], readout: str) -> dict[str, Any]:
    """The witness's score minus the failing call's, both under the neutral task, paired by item."""
    return paired(by, items, readout, "neutral", "neutral", action="witness", action_b="fail")


def controls_block(by: dict, pop: list[dict]) -> dict[str, Any]:
    """Success controls, the mention-only pair and the manipulation check, on one population."""
    out: dict[str, Any] = {}
    out["witness"] = {r: effects(by, pop, r, action="witness") for r in ("first", "yn", "full", "full_norm")}
    # discrimination is read on the items whose witness success the text-only checker can also see (PASS; every class
    # but status, whose witness needs a variant's availability that the status items' histories do not show); all
    # items are reported, and per class (review 2): status has no discrimination evidence and is flagged.
    determined = [t for t in pop if t.get("witness_checker") == "PASS"]
    out["discrimination"] = {r: discrimination(by, determined, r) for r in ("first", "yn")}
    out["discrimination_all_witnesses"] = {r: discrimination(by, pop, r) for r in ("first", "yn")}
    out["discrimination_by_class"] = {}
    for k, v in sorted(_group(pop, lambda t: t["cls"]).items()):
        if k not in PRESPECIFIED_CLASSES:
            continue
        det = [t for t in v if t.get("witness_checker") == "PASS"]
        if not det:
            out["discrimination_by_class"][k] = {
                "evidence": False, "n_items": 0, "holds": None,
                "note": ("no witness of this class is determined by its history (checker UNKNOWN), so there is no "
                         "discrimination evidence: a null or a reversal in this class is not read as use of the record")}
            continue
        blk = discrimination(by, det, "first")
        out["discrimination_by_class"][k] = {"evidence": True, "n_items": blk["n_items"], "first": blk, "holds": check_passes(blk)}
    out["mention"] = {r: {"mention_aligned_minus_mention_swapped": paired(by, pop, r, "mention_aligned", "mention_swapped"),
                          "mention_aligned_minus_neutral": paired(by, pop, r, "mention_aligned", "neutral"),
                          "aligned_minus_mention_aligned": paired(by, pop, r, "aligned", "mention_aligned")}
                      for r in ("first", "yn")}
    out["manipulation"] = paired(by, pop, "target", "aligned", "swapped")
    # is the goal effect specific to failing calls, and does it need the write request or only the order's name?
    out["fail_minus_witness_effect"] = {r: contrast_of_effects(by, pop, r, ("fail", "aligned", "swapped"), ("witness", "aligned", "swapped"))
                                        for r in ("first", "yn")}
    out["write_minus_mention_effect"] = {r: contrast_of_effects(by, pop, r, ("fail", "aligned", "swapped"),
                                                                ("fail", "mention_aligned", "mention_swapped")) for r in ("first", "yn")}
    return out


def format_block(by: dict) -> dict[str, Any]:
    """Format checks (design §6.4 and the smoke run): two-way probability mass of each single-token readout, and
    the seconds per scored job by readout (median and mean; the projection uses the means)."""
    out: dict[str, Any] = {}
    for r in ("first", "yn", "target"):
        masses = [math.exp(x["logp_plus"]) + math.exp(x["logp_minus"]) for k, x in by.items() if k[3] == r]
        out[f"{r}_mass_median"] = statistics.median(masses) if masses else float("nan")
        out[f"{r}_mass_below_0.5_share"] = sum(m < 0.5 for m in masses) / len(masses) if masses else float("nan")
    secs: dict[str, list[float]] = defaultdict(list)
    for k, x in by.items():
        if x.get("seconds") is not None:
            secs[k[3]].append(x["seconds"])
    out["seconds_median_by_readout"] = {r: statistics.median(v) for r, v in sorted(secs.items())}
    out["seconds_mean_by_readout"] = {r: sum(v) / len(v) for r, v in sorted(secs.items())}
    out["seconds_total"] = sum(sum(v) for v in secs.values())
    return out


def arm_block(by: dict, items: list[dict], arm: str) -> dict[str, Any] | None:
    """An exploratory arm beside the main arm on the same items, and per item main minus that arm's effect."""
    if not items:
        return None
    return {r: {arm: effects(by, items, r, arm=arm), "main_same_items": effects(by, items, r),
                f"main_minus_{arm}": contrast_of_effects(by, items, r, ("fail", "aligned", "swapped"), ("fail", "aligned", "swapped"),
                                                         arms=("main", arm))}
            for r in ("first", "yn")}


def model_block(by: dict, primary: list[dict], main: list[dict], natural: list[dict], unach: list[dict],
                arm_items: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"primary": population_block(by, primary), "primary_controls": controls_block(by, primary),
                           "main_pool": population_block(by, main), "main_pool_controls": controls_block(by, main)}
    out["main_pool"]["by_swapped_achievable"] = {str(k): effects(by, v, "first") for k, v in
                                                 sorted(_group(main, lambda t: str(t["swapped_achievable"])).items())}
    out["main_pool"]["by_status_matched"] = {str(k): effects(by, v, "first") for k, v in
                                             sorted(_group(main, lambda t: str(t["status_matched"])).items())}
    out["natural"] = {r: effects(by, natural, r) for r in ("first", "yn")} if natural else None
    out["natural_by_class"] = ({k: effects(by, v, "first", inference=False) for k, v in sorted(_group(natural, lambda t: t["cls"]).items())}
                               if natural else None)
    out["unachievable"] = ({"all": effects(by, unach, "first"),
                            "without_incoherent_twins": effects(by, [t for t in unach if not twin_status_incoherent(t)], "first")}
                           if unach else None)
    out["rules_only_arm"] = arm_block(by, arm_items.get("rules_only", []), "rules_only")
    out["matched_arm"] = arm_block(by, arm_items.get("matched", []), "matched")
    out["format"] = format_block(by)
    out["first_vs_full_agreement"] = _agreement(by, main)
    return out


def sd_of_differences(by: dict, items: list[dict], readout: str) -> dict[str, Any]:
    """Standard deviation of the failing call's per-item aligned - swapped difference (no mean is reported)."""
    diffs = []
    for t in items:
        a, b = by.get((t["id"], "main", "fail", readout, "aligned")), by.get((t["id"], "main", "fail", readout, "swapped"))
        if a and b:
            diffs.append(value(a, readout) - value(b, readout))
    return {"n": len(diffs), "sd": statistics.stdev(diffs) if len(diffs) > 1 else float("nan")}


def run_tau_analysis(cfg: dict) -> int:
    from wf.rq2 import complete_shards

    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(p / "scoring.jsonl" for p in folders if (p / "scoring.jsonl").exists()))
    if problems:
        write_status(run_id, "waits for every scoring shard:\n" + "\n".join(problems), ok=False)
        return 1
    scfg = {**load_config(cfg["source"]), "build_run": cfg.get("build_run", "tau_build")}
    items = scoring_sets(scfg)  # exactly the scoring run's item set
    arms = extra_arm_ids(items, scfg)
    primary_items = [t for t in items if is_primary(t)]
    main = [t for t in items if t["source"] == "generated" and t["main_pool"]]
    natural = [t for t in items if t["source"] == "natural"]
    unach = [t for t in items if t["source"] == "generated" and not t["main_pool"]]
    arm_items = {arm: [t for t in items if t["id"] in ids] for arm, ids in arms.items()}
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    hashes = {t["id"]: item_hash(t) for t in items}
    stale = sum(1 for s in scores if hashes.get(s["transition_id"]) != s.get("item_hash"))
    models = tuple(cfg.get("models", CO_PRIMARY))
    wanted = {(m, t["id"], *job) for m in models for t in items for job in jobs_for(t, scfg, arms)}
    got = {score_key(s) for s in scores}
    missing = len(wanted - got)
    ok = missing == 0 and stale == 0
    report: dict[str, Any] = {"items": {"primary": len(primary_items), "primary_users": len({t["user_id"] for t in primary_items}),
                                        "main": len(main), "main_users": len({t["user_id"] for t in main}),
                                        "natural": len(natural), "unachievable": len(unach),
                                        **{f"{arm}_arm": len(v) for arm, v in arm_items.items()}},
                              "missing_scores": missing, "stale_scores": stale, "models": {}}
    tokens = {}
    for p in folders:
        for f in p.glob("tokens_*.json"):
            tokens.setdefault(f.stem[len("tokens_"):], json.loads(f.read_text(encoding="utf-8")))
    report["token_checks"] = tokens
    by_model = {m: {(s["transition_id"], s.get("arm", "main"), s.get("action", "fail"), s["readout"], s["condition"]): s
                    for s in scores if s["model"] == m} for m in models}
    if cfg.get("project_to"):
        # the smoke run (review 2): format, timing and precision only; no estimate, no verdict is computed or printed
        for m in models:
            report["models"][m] = {"format": format_block(by_model[m]),
                                   "sd_of_differences": {r: sd_of_differences(by_model[m], main, r) for r in ("first", "yn", "full")}}
        report["projection"] = projection(cfg["project_to"], report, scfg, cfg)
        write_json(d / "metrics.json", report)
        write_run_report(run_id, "τ-bench retail smoke run: format, timing and precision only (no estimates)", report)
        pr = report["projection"]
        write_status(run_id, f"items={report['items']} missing={missing} stale={stale}\nprojected hours per shard, both models: "
                             f"{pr['hours_max_shard_both_models']:.2f} (limit {pr['time_limit_hours']}; fits: {pr['fits_time_limit']})",
                     ok=False, reported="τ-bench smoke: format and timing" if ok else None)
        return 0 if ok else 1
    primary, checks, matched = {}, {}, {}
    for m in models:
        by = by_model[m]
        blk = model_block(by, primary_items, main, natural, unach, arm_items)
        report["models"][m] = blk
        primary[m] = blk["primary"]["first"]["aligned_minus_swapped"]
        ctl = blk["primary_controls"]
        checks[m] = {"manipulation_check_passed": check_passes(ctl["manipulation"]),
                     "discrimination_holds": check_passes(ctl["discrimination"]["first"]),
                     "fail_minus_witness": ctl["fail_minus_witness_effect"]["first"],
                     "write_minus_mention": ctl["write_minus_mention_effect"]["first"]}
        if blk["matched_arm"]:
            matched[m] = blk["matched_arm"]["first"]["matched"]["aligned_minus_swapped"]
    report["verdicts"] = verdicts(primary, checks)
    report["sensitivity_without_fewshot_tool_classes"] = sensitivity_verdicts(
        {m: report["models"][m]["primary"]["first_without_fewshot_tool_classes"]["aligned_minus_swapped"] for m in models}, primary)
    report["matched_arm_verdicts"] = matched_verdicts(matched, report["verdicts"]["per_model"]) if matched else None
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "τ-bench retail: does the task override a visible fact about the named object?", report)
    lines = [f"{m}: ΔL1 {primary[m]['estimate']:.3f} CI {primary[m]['ci']} -> {report['verdicts']['per_model'][m]['verdict']}"
             for m in models]
    write_status(run_id, f"items={report['items']} missing={missing} stale={stale}\n" + "\n".join(lines)
                 + f"\noverall: {report['verdicts']['overall']}", ok=False,
                 reported="τ-bench verdict" if ok else None)
    return 0 if ok else 1


def m_eff(items: list[dict]) -> float:
    """Effective cluster size, sum over users of n_u^2 divided by n: the design effect when items of a user are
    perfectly correlated."""
    n_u = defaultdict(int)
    for t in items:
        n_u[t["user_id"]] += 1
    n = sum(n_u.values())
    return sum(v * v for v in n_u.values()) / n if n else float("nan")


def precision(sd: float, items: list[dict]) -> dict[str, Any]:
    """Projected standard error and 95% CI half-width of a mean paired difference over `items`, between no
    within-user correlation (SE = sd / sqrt(n)) and perfect within-user correlation (SE = sd * sqrt(m_eff / n))."""
    n, m = len(items), m_eff(items)
    lo, hi = sd / math.sqrt(n), sd * math.sqrt(m / n)
    return {"n": n, "m_eff": m, "se": [lo, hi], "ci_half_width": [1.96 * lo, 1.96 * hi]}


def power_at(delta: float, se: float) -> dict[str, float]:
    """For a true effect `delta` and standard error `se` (normal approximation): the chance that the point estimate
    reaches 0.5 nats, that the CI lies above 0, and both (the size and significance parts of "replicates")."""
    phi = statistics.NormalDist().cdf
    if not se > 0:  # no spread measured (mock scores, or fewer than two items): nothing to say
        return {k: float("nan") for k in ("p_estimate_at_least_0.5", "p_ci_above_0", "p_both", "p_null_verdict_if_true_effect_0")}
    return {"p_estimate_at_least_0.5": phi((delta - SUPPORT) / se), "p_ci_above_0": phi(delta / se - 1.96),
            "p_both": phi((delta - max(SUPPORT, 1.96 * se)) / se),
            "p_null_verdict_if_true_effect_0": max(0.0, 2 * phi((SUPPORT - 1.96 * se) / se) - 1)}


def projection(target_run: str, report: dict, smoke_cfg: dict, analysis_cfg: dict | None = None) -> dict[str, Any]:
    """From a smoke run: (1) the GPU hours per shard of the full scoring run (one model after the other, as
    run_tau_score does), from the mean seconds per job by readout, checked against the sbatch time limit; (2) the
    projected CI half-widths of the primary contrast and the matched arm's, from the standard deviation of the smoke
    items' paired differences; (3) for each model, the chance of each part of a "replicates" verdict if its τ effect
    equalled its ALFWorld first-token ΔL (amendments after review 2). No estimate of any τ effect is used."""
    analysis_cfg = analysis_cfg or {}
    full = {**load_config(target_run), "build_run": smoke_cfg.get("build_run", "tau_build")}
    items = scoring_sets(full)
    arms = extra_arm_ids(items, full)
    n = int(full.get("n_shards", 1))
    shards = [shard_by_user(items, i, n) for i in range(n)]
    limit = float(analysis_cfg.get("time_limit_hours", float("nan")))
    out: dict[str, Any] = {"run": target_run, "n_shards": n, "time_limit_hours": limit, "models": {}}
    primary = [t for t in items if is_primary(t)]
    matched = [t for t in items if t["id"] in arms["matched"]]
    refs = analysis_cfg.get("alfworld_first_token_delta_l", {})
    for m, blk in report["models"].items():
        sec = blk["format"]["seconds_mean_by_readout"]
        hours = []
        for sh in shards:
            jobs = [j for t in sh for j in jobs_for(t, full, arms)]
            hours.append(sum(sec.get(j[2], float("nan")) for j in jobs) / 3600)
        prec = {}
        for r, s in blk.get("sd_of_differences", {}).items():
            prec[r] = {"smoke_sd": s["sd"], "smoke_n": s["n"], "primary": precision(s["sd"], primary) if primary else None,
                       "matched_arm": precision(s["sd"], matched) if matched else None}
        power = None
        if m in refs and prec.get("first", {}).get("primary"):
            se = prec["first"]["primary"]["se"]
            power = {"alfworld_first_token_delta_l": refs[m],
                     "at_se_without_clustering": power_at(refs[m], se[0]), "at_se_with_full_clustering": power_at(refs[m], se[1])}
        out["models"][m] = {"hours_max_shard": max(hours), "hours_all_shards": sum(hours), "precision": prec,
                            "power_if_alfworld_sized": power}
    out["hours_max_shard_both_models"] = sum(v["hours_max_shard"] for v in out["models"].values())
    out["fits_time_limit"] = out["hours_max_shard_both_models"] <= limit
    return out


# ---------------- contamination (exploratory) ----------------

CONTAM_SYSTEM = "You are a helpful assistant."
CONTAM_PROMPT = "Here is one record from the tau-bench retail database, as JSON."


ID_DIGITS = re.compile(r"\d{5,}")
KEEP_SUBTREES = ("address",)  # zips and street numbers are tied to the city and state: kept verbatim


def resample_ids(text: str, salt: str) -> str:
    """The record with every identifier's digits resampled (amendment of 2026-11-20 after review 1).

    The record is parsed as JSON. Every run of 5 or more digits in a key or a string value is replaced by seeded
    random digits of the same length whose first digit is 1-9, the same run getting the same new digits
    everywhere in the record. Addresses (zip, street and suite numbers) are kept verbatim, because a zip goes
    with a city and state and resampling it would test geography, not memorisation. Numbers (prices, balances)
    are not identifiers and are untouched. The result is serialised as the tools serialise records, so the two
    texts differ only in identifier digits.
    """
    obj = json.loads(text)
    if json.dumps(obj) != text:
        raise ValueError("record is not in the tools' JSON serialisation")
    mapping: dict[str, str] = {}
    rng = random.Random(f"{I.SEED}|resample|{salt}")

    def sub(m: re.Match) -> str:
        s = m.group(0)
        if s not in mapping:
            new = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(len(s) - 1))
            mapping[s] = new if new != s else str((int(new[0]) % 9) + 1) + new[1:]
        return mapping[s]

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {ID_DIGITS.sub(sub, k): (v if k in KEEP_SUBTREES else walk(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return ID_DIGITS.sub(sub, x)
        return x

    return json.dumps(walk(obj))


def contamination_records(items: list[dict], per_kind: int = 50) -> list[dict]:
    recs: dict[str, dict[str, str]] = {"user": {}, "order": {}, "product": {}}
    for t in items:
        for x in t["history"]:
            kind = {"get_user_details": "user", "get_order_details": "order", "get_product_details": "product"}.get(x["name"])
            if kind:
                recs[kind].setdefault(json.dumps(x["arguments"], sort_keys=True), x["output"])
    out = []
    for kind, by in recs.items():
        for key in sorted(by, key=lambda k: I.h(kind, k, "contam"))[:per_kind]:
            out.append({"id": f"{kind}:{key}", "kind": kind, "verbatim": by[key], "resampled": resample_ids(by[key], key)})
    return out


LLAMA_PREDATES = "llama3.1-8b-fewshot"  # training cutoff December 2023; tau-bench was released in June 2024


def contamination_report(scores: list[dict], recs: list[dict], models: tuple[str, ...]) -> dict[str, Any]:
    """Per model and record kind, the mean log P(verbatim) - log P(resampled); then each other model minus
    Llama-3.1, paired by record: Llama's cutoff predates tau-bench, so its difference is the no-memorisation
    baseline for records of this shape (amendment of 2026-11-20)."""
    report: dict[str, Any] = {"n_records": len(recs), "models": {}, "minus_llama": {}}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    kinds = ("all", "user", "order", "product")
    by_model = {tag: {x["transition_id"]: x for x in scores if x["model"] == tag} for tag in models}
    for tag in models:
        s = list(by_model[tag].values())
        block = {}
        for kind in kinds:
            xs = [x for x in s if kind == "all" or x["kind"] == kind]
            diffs = [x["L"] for x in xs]
            boot = cluster_bootstrap(mean, diffs, [x["transition_id"] for x in xs], seed=C.RNG_SEED)
            block[kind] = {"n": len(xs), "mean_logp_verbatim_minus_resampled": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]],
                           "share_verbatim_higher": mean([1.0 if v > 0 else 0.0 for v in diffs])}
        report["models"][tag] = block
    if LLAMA_PREDATES in by_model:
        base = by_model[LLAMA_PREDATES]
        for tag in models:
            if tag == LLAMA_PREDATES:
                continue
            block = {}
            for kind in kinds:
                ids = [i for i, x in by_model[tag].items() if i in base and (kind == "all" or x["kind"] == kind)]
                diffs = [by_model[tag][i]["L"] - base[i]["L"] for i in ids]
                boot = cluster_bootstrap(mean, diffs, ids, seed=C.RNG_SEED)
                block[kind] = {"n": len(ids), "mean": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]]}
            report["minus_llama"][tag] = block
    return report


def run_tau_contamination(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    ok, why = verified(cfg)
    if not ok:
        write_status(run_id, f"items not verified: {why}", ok=False)
        return 1
    d = _init_run(run_id, cfg, inputs=())
    gen, _, _ = load_built(cfg)
    recs = contamination_records([t for t in gen if t["main_pool"]], int(cfg.get("per_kind", 50)))
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    done = {(s["model"], s["transition_id"]) for s in saved}
    messages = [{"role": "system", "content": CONTAM_SYSTEM}, {"role": "user", "content": CONTAM_PROMPT}]
    for tag in cfg.get("models", CO_PRIMARY):
        todo = [r for r in recs if (tag, r["id"]) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(MODELS[tag], model_tag=tag)
        for r in todo:
            sc = scorer.score_pair(r["id"], "contamination", messages, r["verbatim"], r["resampled"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "kind": r["kind"]}])
        scorer.close()
    scores = read_jsonl(d / "scoring.jsonl")
    report = contamination_report(scores, recs, tuple(cfg.get("models", CO_PRIMARY)))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "τ-bench contamination check (exploratory)", report)
    write_status(run_id, json.dumps({m: b["all"] for m, b in report["models"].items()}), ok=False,
                 reported="contamination check (exploratory)")
    return 0
