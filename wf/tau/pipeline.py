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
from collections import defaultdict
from pathlib import Path
from typing import Any

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation
from wf.runs import _finished, _init_run, load_config
from wf.scoring.logodds import LogoddsScorer
from wf.tau import items as I
from wf.tau.prompts import CONDITIONS, build_messages
from wf.tau.retail import check_vendor, pristine_sha256, vendor_manifest
from wf.tau.verify import fresh_process_sample, sample_ids, verify_sets
from wf.util import append_jsonl, file_sha, read_jsonl, require_filed, runs_root, write_json, write_status

MODELS = {"qwen3-8b-fewshot": C.WM_FEWSHOT_MODEL, "llama3.1-8b-fewshot": C.WM_FEWSHOT_LLAMA}
CO_PRIMARY = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
FIRST_PLUS, FIRST_MINUS = '{"', "Error"
READOUTS = ("first", "yn", "full")
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
    by = lambda xs, f: {str(k): {"items": len(v), "users": users(v)} for k, v in sorted(_group(xs, f).items())}
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
    """Has tau_verify passed on exactly the files scoring would read, and do they match the pinned sha256?"""
    vrun = cfg.get("verify_run", "tau_verify")
    if not _finished(vrun, tuple(cfg.get("accept_commits", {}).get(vrun, ()))):
        return False, f"{vrun} has not passed"
    rep = json.loads((runs_root() / vrun / "verify.json").read_text(encoding="utf-8"))
    now = _sha_of_outputs(_build_dir(cfg))
    if not rep.get("passed") or rep.get("files") != now:
        return False, "verify.json does not cover the current item files"
    pinned = cfg.get("expected_items_sha256")
    if pinned and now.get("items.jsonl") != pinned:
        return False, f"items.jsonl sha256 {now.get('items.jsonl')} differs from the pinned {pinned}"
    return True, ""


# ---------------- scoring ----------------


def scoring_sets(cfg: dict) -> list[dict]:
    gen, nat, _ = load_built(cfg)
    out = [t for t in gen if t["main_pool"]]
    if cfg.get("score_unachievable"):
        out += [t for t in gen if not t["main_pool"]]
    if cfg.get("score_natural", True):
        out += nat
    if cfg.get("limit_items"):
        out = sorted(out, key=lambda t: I.h(t["id"], "smoke"))[: int(cfg["limit_items"])]
    return out


def shard_by_user(items: list[dict], i: int, n: int) -> list[dict]:
    keep = set(sorted({t["user_id"] for t in items})[i::n])
    return [t for t in items if t["user_id"] in keep]


def item_hash(t: dict) -> str:
    fields = {k: t.get(k) for k in ("history", "action", "failure_reply", "success_twin", "task_aligned",
                                     "task_swapped", "task_neutral")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def jobs_for(t: dict, readouts: tuple[str, ...]) -> list[tuple[str, str]]:
    conds = [c for c in CONDITIONS if c != "swapped" or t.get("task_swapped")]
    return [(r, c) for r in readouts for c in conds]


def token_report(scorer: LogoddsScorer, sample: dict) -> dict[str, Any]:
    """Are '{"' and 'Error' single tokens that open the tokenised replies (design §6.1)? Mock: trivially yes."""
    if scorer.mock:
        return {"mock": True, "first_token_ok": True}
    tok = scorer._tok
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    rep = {v: enc(v) for v in (FIRST_PLUS, FIRST_MINUS, " " + FIRST_PLUS, " " + FIRST_MINUS, "Yes", "No")}
    twin, fail = enc(sample["success_twin"]), enc(sample["failure_reply"])
    gen_prompt = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    ok = (len(rep[FIRST_PLUS]) == 1 and len(rep[FIRST_MINUS]) == 1 and twin[:1] == rep[FIRST_PLUS]
          and fail[:1] == rep[FIRST_MINUS] and gen_prompt.endswith("\n"))
    return {"ids": {k: v for k, v in rep.items()}, "twin_first": twin[:3], "failure_first": fail[:3],
            "generation_prompt_tail": gen_prompt[-30:], "first_token_ok": ok}


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
    items = shard_by_user(scoring_sets(cfg), i, n)
    readouts = tuple(cfg.get("readouts", READOUTS))
    hashes = {t["id"]: item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["readout"], s["condition"]) for s in saved}
    for tag in cfg.get("models", CO_PRIMARY):
        todo = [(t, r, c) for t in items for r, c in jobs_for(t, readouts) if (tag, t["id"], r, c) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(MODELS[tag], model_tag=tag)
        tokens_path = d / f"tokens_{tag}.json"
        if items and not tokens_path.exists():
            write_json(tokens_path, token_report(scorer, items[0]))
        for t, r, c in todo:
            if r == "first":
                sc = scorer.score_one_token_pair(t["id"], c, build_messages(t, c, fewshot), FIRST_PLUS, FIRST_MINUS)
            elif r == "yn":
                sc = scorer.score_one_token_pair(t["id"], c, build_messages(t, c, fewshot, yes_no=True), "Yes", "No")
            else:
                sc = scorer.score_pair(t["id"], c, build_messages(t, c, fewshot), t["success_twin"], t["failure_reply"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "readout": r, "user_id": t["user_id"],
                                                "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} items, {len({t['user_id'] for t in items})} users",
                 ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def value(rec: dict, readout: str) -> float:
    if readout == "full_norm":
        return rec["logp_plus"] / rec["n_tokens_plus"] - rec["logp_minus"] / rec["n_tokens_minus"]
    return rec["L"]


def paired(by: dict, items: list[dict], readout: str, a: str, b: str) -> dict[str, Any]:
    """Mean of value(a) - value(b) over items, with a user-clustered CI and a user-level sign-flip p."""
    base = "full" if readout.startswith("full") else readout
    diffs, users = [], []
    for t in items:
        x, y = by.get((t["id"], base, a)), by.get((t["id"], base, b))
        if x is None or y is None:
            continue
        v = value(x, readout) - value(y, readout)
        if v == v:
            diffs.append(v)
            users.append(t["user_id"])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    boot = cluster_bootstrap(mean, diffs, users, seed=C.RNG_SEED)
    return {"estimate": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]], "n_items": len(diffs),
            "n_users": len(set(users)), "perm_p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=users)}


def effects(by: dict, items: list[dict], readout: str) -> dict[str, Any]:
    return {"aligned_minus_swapped": paired(by, items, readout, "aligned", "swapped"),
            "aligned_minus_neutral": paired(by, items, readout, "aligned", "neutral"),
            "swapped_minus_neutral": paired(by, items, readout, "swapped", "neutral")}


def tau_verdict(est: float, ci: list[float], holm_p: float) -> str:
    """The design's decision rule (notes/tau_design.md §6.3), two-sided."""
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


def verdicts(primary: dict[str, dict]) -> dict[str, Any]:
    adjusted = holm({m: primary[m]["perm_p"] if m in primary else 1.0 for m in CO_PRIMARY})
    per = {m: {"verdict": tau_verdict(primary[m]["estimate"], primary[m]["ci"], adjusted[m]) if m in primary else "not scored",
               "holm_p": adjusted[m]} for m in CO_PRIMARY}
    rep = [m for m, v in per.items() if v["verdict"].startswith("replicates")]
    rev = [m for m, v in per.items() if v["verdict"].startswith("reversed")]
    overall = "replicates" if rep and not rev else "reversed" if rev and not rep else "mixed" if rep and rev else "does not replicate"
    return {"per_model": per, "overall": overall}


def _flip_share(by: dict, items: list[dict]) -> float:
    pairs = [(by[(t["id"], "first", "aligned")]["L"], by[(t["id"], "first", "swapped")]["L"]) for t in items
             if (t["id"], "first", "aligned") in by and (t["id"], "first", "swapped") in by]
    return sum(1 for a, s in pairs if a > 0 > s) / len(pairs) if pairs else float("nan")


def _agreement(by: dict, items: list[dict]) -> dict[str, Any]:
    """Primary pass vs the full pass's first-token log-probabilities (design §6.1): within 0.05 nats?"""
    gaps = []
    for t in items:
        for c in CONDITIONS:
            f, g = by.get((t["id"], "first", c)), by.get((t["id"], "full", c))
            if f and g:
                gaps.append(max(abs(f["logp_plus"] - g["logp_first_plus"]), abs(f["logp_minus"] - g["logp_first_minus"])))
    within = sum(1 for x in gaps if x <= 0.05)
    return {"n": len(gaps), "share_within_0.05": within / len(gaps) if gaps else float("nan"),
            "max_gap": max(gaps) if gaps else float("nan"), "ok": bool(gaps) and within / len(gaps) >= 0.99}


def model_block(by: dict, main: list[dict], natural: list[dict], unach: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in ("first", "yn", "full", "full_norm"):
        out[r] = {"main": effects(by, main, r),
                  "by_class": {k: effects(by, v, r) for k, v in sorted(_group(main, lambda t: t["cls"]).items())},
                  "by_swapped_achievable": {str(k): effects(by, v, r) for k, v in sorted(_group(main, lambda t: t["swapped_achievable"]).items())},
                  "by_status_matched": {str(k): effects(by, v, r) for k, v in sorted(_group(main, lambda t: t["status_matched"]).items())},
                  "natural": effects(by, natural, r) if natural else None,
                  "natural_aligned_achievable": effects(by, [t for t in natural if t["main_pool"]], r) if natural else None,
                  "natural_by_class": {k: effects(by, v, r) for k, v in sorted(_group(natural, lambda t: t["cls"]).items())} if natural else None,
                  "unachievable": effects(by, unach, r) if unach else None}
    masses = [math.exp(x["logp_plus"]) + math.exp(x["logp_minus"]) for (iid, r, c), x in by.items() if r == "first"]
    out["first_mass_median"] = statistics.median(masses) if masses else float("nan")
    out["flip_to_success_share"] = _flip_share(by, main)
    out["first_vs_full_agreement"] = _agreement(by, main)
    return out


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
    main = [t for t in items if t["source"] == "generated" and t["main_pool"]]
    natural = [t for t in items if t["source"] == "natural"]
    unach = [t for t in items if t["source"] == "generated" and not t["main_pool"]]
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    hashes = {t["id"]: item_hash(t) for t in items}
    stale = sum(1 for s in scores if hashes.get(s["transition_id"]) != s.get("item_hash"))
    models = tuple(cfg.get("models", CO_PRIMARY))
    wanted = {(m, t["id"], "first", c) for m in models for t in main for c in ("aligned", "swapped")}
    got = {(s["model"], s["transition_id"], s["readout"], s["condition"]) for s in scores}
    missing = len(wanted - got)
    report: dict[str, Any] = {"items": {"main": len(main), "main_users": len({t["user_id"] for t in main}),
                                        "natural": len(natural), "unachievable": len(unach)},
                              "missing_primary_scores": missing, "stale_scores": stale, "models": {}}
    tokens = {}
    for p in folders:
        for f in p.glob("tokens_*.json"):
            tokens.setdefault(f.stem[len("tokens_"):], json.loads(f.read_text(encoding="utf-8")))
    report["token_checks"] = tokens
    primary = {}
    for m in models:
        by = {(s["transition_id"], s["readout"], s["condition"]): s for s in scores if s["model"] == m}
        report["models"][m] = model_block(by, main, natural, unach)
        primary[m] = report["models"][m]["first"]["main"]["aligned_minus_swapped"]
    report["verdicts"] = verdicts(primary)
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "τ-bench retail: goal effect on determined tool failures", report)
    lines = [f"{m}: ΔL1 {primary[m]['estimate']:.3f} CI {primary[m]['ci']} -> {report['verdicts']['per_model'][m]['verdict']}"
             for m in models]
    ok = missing == 0 and stale == 0
    write_status(run_id, f"items={report['items']} missing={missing} stale={stale}\n" + "\n".join(lines)
                 + f"\noverall: {report['verdicts']['overall']}", ok=False,
                 reported="τ-bench verdict" if ok else None)
    return 0 if ok else 1


# ---------------- contamination (exploratory) ----------------

CONTAM_SYSTEM = "You are a helpful assistant."
CONTAM_PROMPT = "Here is one record from the tau-bench retail database, as JSON."


def resample_ids(text: str, salt: str) -> str:
    """Every run of 5+ digits replaced, consistently within the text, by seeded random digits of equal length."""
    mapping: dict[str, str] = {}
    rng = random.Random(f"{I.SEED}|resample|{salt}")

    def sub(m: re.Match) -> str:
        s = m.group(0)
        if s not in mapping:
            new = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(len(s) - 1))
            mapping[s] = new if new != s else str((int(s[0]) % 9) + 1) + new[1:]
        return mapping[s]

    return re.sub(r"\d{5,}", sub, text)


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
    report: dict[str, Any] = {"n_records": len(recs), "models": {}}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for tag in cfg.get("models", CO_PRIMARY):
        s = [x for x in scores if x["model"] == tag]
        block = {}
        for kind in ("all", "user", "order", "product"):
            xs = [x for x in s if kind == "all" or x["kind"] == kind]
            diffs = [x["L"] for x in xs]
            boot = cluster_bootstrap(mean, diffs, [x["transition_id"] for x in xs], seed=C.RNG_SEED)
            block[kind] = {"n": len(xs), "mean_logp_verbatim_minus_resampled": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]],
                           "share_verbatim_higher": mean([1.0 if v > 0 else 0.0 for v in diffs])}
        report["models"][tag] = block
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "τ-bench contamination check (exploratory)", report)
    write_status(run_id, json.dumps({m: b["all"] for m, b in report["models"].items()}), ok=False,
                 reported="contamination check (exploratory)")
    return 0
