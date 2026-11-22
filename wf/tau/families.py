"""τ-bench retail, wave-A model families (notes/tau_families_design.md, written 2026-11-22 before any score of these
models).

The τ-bench replication's items, prompts and per-model decision rule (wf/tau/pipeline.py, notes/tau_design.md),
applied to two more instruct models with wave A's chat-template handling (wf/families.py), under a cost-reduced
protocol on the 856 primary items: per item 14 single-token forward passes (PROTOCOL_JOBS). Gemma 2 9B is excluded
before data: its 8,192-token trained context is smaller than the prompt's fixed part (design §2.3).

Runs (wf/runs.py):
- tau_family_token_check (CPU): every scored prompt of every model through the scorer's own encoding: the scored
  tokens are single tokens and the ones the template writes at the start of each reply, the prompts under different
  task lines differ only in the task line, and every prompt fits the model's context (Gemma: lengths only).
- tau_family_score_smoke, tau_family_score_<tag>, tau_family_placement_mistral-7b-v0.3 (GPU): scoring, sharded by
  user, each score saved as it finishes. Refuses unverified or unpinned items, a model whose token check has not
  passed on those files, an excluded model, and any prompt at or above the model's context.
- tau_family_smoke_analysis, tau_family_analysis (CPU): format, timing and precision only (smoke); per-model verdicts
  under the τ rule with Holm over the scored models and the families competence rule as a third gate, descriptive
  comparisons with the reference models' τ scores and the Mistral placement check.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from wf import constants as C
from wf import families as F
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, pair_permutation
from wf.runs import _finished, _init_run, load_config
from wf.scoring.logodds import LogoddsScorer
from wf.tau import items as I
from wf.tau import pipeline as P
from wf.tau.prompts import CONDITIONS, MENTION_CONDITIONS, build_messages, task_for
from wf.util import REPO_ROOT, append_jsonl, is_mock, read_jsonl, require_filed, runs_root, write_json, write_status

# ---------------- models (design §2) ----------------

SCORED = ("mistral-7b-v0.3", "llama3.2-3b")  # the confirmatory family of design §7.2, in this order
EXCLUDED = {"gemma2-9b-it": ("excluded before data (design §2.3): the prompt's fixed part, the system message and the "
                             "four few-shot examples, is 9,575 Gemma tokens, more than Gemma 2's 8,192-token trained "
                             "context; reported as not run, never as a null")}
CONTEXT = {"mistral-7b-v0.3": 32768, "llama3.2-3b": 131072, "gemma2-9b-it": 8192}  # max_position_embeddings
REFERENCE = P.CO_PRIMARY  # Qwen3-8B and Llama-3.1-8B, read from the reference τ scoring run
PLACEMENT_MODEL = "mistral-7b-v0.3"
PLACEMENT_ITEMS = 300
PLACEMENT_SALT = f"{C.RNG_SEED}|tau_family_placement"
FIRST_TURN = F.FIRST_TURN  # tag suffix of the placement check's scores
BUDGET_GPU_HOURS = 6.5  # design §4.4 and amendment R1-S4: the aim was about 6; restoring the witness's yes/no puts
# Mistral at about 6.1, so the projection flags a total above 6.5 (informational; the shard limit decides)
NOT_INTERPRETABLE = "not interpretable (fails the families competence rule on the first-token readout)"

READOUT_WORDS = {"first": (P.FIRST_PLUS, P.FIRST_MINUS), "yn": ("Yes", "No"), "target": ("Yes", "No")}
CUE = {"first": None, "yn": "yn", "target": "target"}

# Design §4.1, as amended after review 1 (R1-S4: the witness's yes/no under all three task lines): per primary item,
# 16 single-token passes: (arm, action, readout, condition)
PROTOCOL_JOBS: tuple[tuple[str, str, str, str], ...] = (
    *[("main", "fail", "first", c) for c in CONDITIONS],
    *[("main", "fail", "yn", c) for c in CONDITIONS],
    *[("main", "fail", "first", c) for c in MENTION_CONDITIONS],
    *[("main", "fail", "target", c) for c in ("aligned", "swapped")],
    *[("main", "witness", "first", c) for c in CONDITIONS],
    *[("main", "witness", "yn", c) for c in CONDITIONS],
)
PLACEMENT_JOBS: tuple[tuple[str, str, str, str], ...] = (("main", "fail", "first", "aligned"),
                                                          ("main", "fail", "first", "swapped"))
REQUIRED_FIELDS = ("task_swapped", "task_mention_aligned", "task_mention_swapped", "aligned_witness", "witness_output",
                   "witness_failure")


def base_tag(tag: str) -> str:
    return tag[: -len(FIRST_TURN)] if tag.endswith(FIRST_TURN) else tag


def spec_for(tag: str, *, allow_excluded: bool = False) -> F.ModelSpec:
    """The wave-A spec of a model (its template handling, attention and template settings); `<tag>-firstturn` is the
    placement check's spec (system text folded into the first user turn). An excluded model is refused unless only
    its prompt lengths are wanted."""
    base = base_tag(tag)
    if base in EXCLUDED and not allow_excluded:
        raise ValueError(f"{base}: {EXCLUDED[base]}")
    if base not in F.MODELS:
        raise ValueError(f"unknown model {tag!r}")
    return F.first_turn_spec(F.MODELS[base]) if tag.endswith(FIRST_TURN) else F.MODELS[base]


def messages(spec: F.ModelSpec, t: dict, condition: str, fewshot: list[dict], readout: str = "first",
             action: str = "fail") -> list[dict[str, str]]:
    """The τ main-arm prompt of one job, with the model's own template handling (Gemma-style fold or native)."""
    msgs = build_messages(t, condition, fewshot, cue=CUE[readout], action=action, arm="main")
    return F.fold_system(msgs) if spec.system == "fold" else msgs


def jobs_for(t: dict) -> list[tuple[str, str, str, str]]:
    """Every job of one primary item (design §4.1). An item lacking a scored line or its witness is refused, never
    scored with fewer jobs."""
    missing = [k for k in REQUIRED_FIELDS if not t.get(k)]
    if missing:
        raise ValueError(f"{t['id']}: missing {missing}; every primary item needs all 16 jobs")
    return list(PROTOCOL_JOBS)


def score_key(s: dict) -> tuple:
    return P.score_key(s)


# ---------------- items ----------------


def primary_items(cfg: dict) -> list[dict]:
    """The 856 primary items in build order; the smoke run takes the first `limit_items` by hash plus the longest
    primary item (by rendered history and call)."""
    gen, _, _ = P.load_built(cfg)
    prim = [t for t in gen if P.is_primary(t)]
    if cfg.get("limit_items"):
        out = sorted(prim, key=lambda t: I.h(t["id"], "smoke"))[: int(cfg["limit_items"])]
        if cfg.get("add_longest") and prim:
            longest = max(prim, key=lambda t: (P.prompt_chars(t), t["id"]))
            if longest["id"] not in {t["id"] for t in out}:
                out.append(longest)
        keep = {t["id"] for t in out}
        return [t for t in prim if t["id"] in keep]
    return prim


def placement_items(primary: list[dict], k: int = PLACEMENT_ITEMS) -> list[dict]:
    """A seeded sample of `k` primary items in id order; the same whatever order the items arrive in."""
    pool = sorted(primary, key=lambda t: t["id"])
    picked = random.Random(PLACEMENT_SALT).sample(pool, min(k, len(pool)))
    return sorted(picked, key=lambda t: t["id"])


# ---------------- token check (design §6.2) ----------------


def _load_tokenizer(spec: F.ModelSpec):  # tests replace this with a stand-in tokenizer
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, local_files_only=True)


def _scorer_for(spec: F.ModelSpec, tok: Any) -> LogoddsScorer:
    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, attn_implementation=spec.attn_implementation,
                                       template_kwargs=spec.template_kwargs)


def _render_text(scorer: LogoddsScorer, msgs: list[dict]) -> str:
    return scorer._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **scorer.template_settings())


def _only_task_line(texts: dict[str, str], t: dict) -> list[str]:
    """Conditions whose rendered prompt differs from the group's first one (aligned where it is scored) outside the
    query's task line (the last occurrence of "Task: <line>\\n", which is the query's); a condition whose task line is
    not found is reported too. A group of one condition has nothing to compare."""
    def masked(c: str) -> str | None:
        key = f"Task: {task_for(t, c)}\n"
        i = texts[c].rfind(key)
        return None if i < 0 else texts[c][:i] + "Task: <X>\n" + texts[c][i + len(key):]
    base = masked(next(iter(texts)))
    return [c for c in texts if base is None or masked(c) is None or masked(c) != base]


def check_model(tag: str, tok: Any, items: list[dict], fewshot: list[dict], *, placement: list[dict] | None = None,
                length_only: bool = False) -> dict[str, Any]:
    """The token check of design §6.2 for one model on every scored prompt; "passed" says whether it may be scored.
    With `length_only` (Gemma) only the prompt lengths are recorded."""
    spec = spec_for(tag, allow_excluded=length_only)
    scorer = _scorer_for(spec, tok)
    limit = CONTEXT[base_tag(tag)]
    ids = lambda x: F._ids(x)
    alone = {w: ids(scorer._encode_candidate(w)) for w in (P.FIRST_PLUS, P.FIRST_MINUS, "Yes", "No")}
    problems: dict[str, list] = {k: [] for k in ("single_tokens", "context", "only_task_line", "first_token",
                                                  "boundary", "yes_no", "template")}
    for w, v in alone.items():
        if len(v) != 1:
            problems["single_tokens"].append(f"{w!r} -> {v}")
    if alone[P.FIRST_PLUS] == alone[P.FIRST_MINUS] or alone["Yes"] == alone["No"]:
        problems["single_tokens"].append("the two tokens of a readout are the same")
    lengths: dict[str, list[int]] = defaultdict(list)
    longest: list[Any] = [0]
    n_prompts = n_yes_no = n_replies = 0

    def prompt_len(msgs: list[dict], key: str, where: tuple) -> int:
        nonlocal n_prompts
        n = len(ids(scorer._encode(msgs)))
        n_prompts += 1
        lengths[key].append(n)
        if n > longest[0]:
            longest[:] = [n, *where]
        if n >= limit:
            problems["context"].append((*where, n))
        return n

    for t in items:
        groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for _, action, readout, cond in jobs_for(t):
            msgs = messages(spec, t, cond, fewshot, readout, action)
            prompt_len(msgs, f"{action}/{readout}", (t["id"], action, readout, cond))
            if not length_only:
                groups[(action, readout)][cond] = _render_text(scorer, msgs)
        if length_only:
            continue
        for (action, readout), texts in groups.items():
            for c in _only_task_line(texts, t):
                problems["only_task_line"].append((t["id"], action, readout, c))
        for action in ("fail", "witness"):
            plus, minus = ((t["success_twin"], t["failure_reply"]) if action == "fail"
                           else (t["witness_output"], t["witness_failure"]))
            msgs = messages(spec, t, "aligned", fewshot, "first", action)
            for text, want in ((plus, alone[P.FIRST_PLUS]), (minus, alone[P.FIRST_MINUS])):
                try:
                    reply = F.template_reply(scorer, msgs, text)
                except F.TokenizationMismatch as e:
                    problems["template"].append((t["id"], action, str(e)))
                    continue
                n_replies += 1
                if reply[:1] != want:
                    problems["first_token"].append((t["id"], action, text[:12], reply[:2]))
                if ids(scorer._encode_candidate(text)) != reply:
                    problems["boundary"].append((t["id"], action, text[:12]))
        for _, action, readout, cond in jobs_for(t):
            if readout not in ("yn", "target"):
                continue
            msgs = messages(spec, t, cond, fewshot, readout, action)
            n_yes_no += 1
            for w in ("Yes", "No"):
                try:
                    reply = F.template_reply(scorer, msgs, w)
                except F.TokenizationMismatch as e:
                    problems["template"].append((t["id"], action, readout, cond, str(e)))
                    continue
                if reply != alone[w]:
                    problems["yes_no"].append((t["id"], action, readout, cond, w, reply))
    out_place = None
    if placement is not None and not length_only:
        fspec = F.first_turn_spec(spec)
        fsc = _scorer_for(fspec, tok)
        plen: list[int] = []
        place_problems: list = []
        for t in placement:
            texts = {}
            for _, action, readout, cond in PLACEMENT_JOBS:
                msgs = messages(fspec, t, cond, fewshot, readout, action)
                n = len(ids(fsc._encode(msgs)))
                plen.append(n)
                if n >= limit:
                    place_problems.append(("context", t["id"], cond, n))
                texts[cond] = _render_text(fsc, msgs)
            for c in _only_task_line(texts, t):
                place_problems.append(("only_task_line", t["id"], c))
            msgs = messages(fspec, t, "aligned", fewshot, "first", "fail")
            for text, want in ((t["success_twin"], alone[P.FIRST_PLUS]), (t["failure_reply"], alone[P.FIRST_MINUS])):
                try:
                    if F.template_reply(fsc, msgs, text)[:1] != want:
                        place_problems.append(("first_token", t["id"], text[:12]))
                except F.TokenizationMismatch as e:
                    place_problems.append(("template", t["id"], str(e)))
        out_place = {"n_items": len(placement), "n_prompts": len(plen), "max_prompt_tokens": max(plen) if plen else None,
                     "n_problems": len(place_problems), "problems": place_problems[:20]}
    summary = {k: {"median": statistics.median(v), "p90": sorted(v)[int(0.9 * len(v))], "max": max(v), "n": len(v),
                   "share_at_or_above_context": sum(x >= limit for x in v) / len(v)} for k, v in sorted(lengths.items())}
    rec: dict[str, Any] = {
        "model": tag, "hf_id": spec.hf_id, "revision": spec.revision, "system_handling": spec.system,
        "template_settings": scorer.template_settings(), "context": limit, "n_items": len(items),
        "n_prompts": n_prompts, "prompt_tokens": summary, "longest_prompt": longest,
        "scored_tokens": {w: tok.convert_ids_to_tokens(v) if hasattr(tok, "convert_ids_to_tokens") else v
                          for w, v in alone.items()},
        "length_only": length_only}
    if length_only:
        rec.update({"excluded": EXCLUDED.get(tag), "n_at_or_above_context": len(problems["context"]), "passed": None})
        return rec
    rec.update({"n_replies_checked": n_replies, "n_yes_no_prompts_checked": n_yes_no,
                "problems": {k: len(v) for k, v in problems.items()}, "examples": {k: v[:5] for k, v in problems.items()},
                "placement": out_place})
    rec["passed"] = (not any(problems.values()) and n_prompts > 0 and n_replies == 4 * len(items)
                     and (out_place is None or out_place["n_problems"] == 0))
    return rec


def _check_one(args: tuple) -> dict:
    tag, items, fewshot, placement, length_only = args
    return check_model(tag, _load_tokenizer(spec_for(tag, allow_excluded=length_only)), items, fewshot,
                       placement=placement, length_only=length_only)


def run_tau_family_token_check(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    ok, why = P.verified(cfg)
    if not ok:
        write_status(run_id, f"items not verified: {why}", ok=False)
        return 1
    build = P._build_dir(cfg)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(build / n for n in P.OUTPUT_FILES if (build / n).exists()))
    _, _, fewshot = P.load_built(cfg)
    items = primary_items(cfg)
    place = placement_items(items, int(cfg.get("placement_items", PLACEMENT_ITEMS)))
    tasks = [(tag, items, fewshot, place if tag == PLACEMENT_MODEL else None, False) for tag in cfg["models"]]
    tasks += [(tag, items, fewshot, None, True) for tag in cfg.get("length_only_models", [])]
    if is_mock() and not cfg.get("tokenize_in_mock"):
        results = [{"model": a[0], "skipped": "mock (no tokenizer)", "passed": None, "length_only": a[4]} for a in tasks]
    elif int(cfg.get("workers", 1)) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(cfg["workers"])) as pool:
            results = list(pool.map(_check_one, tasks))
    else:
        results = [_check_one(a) for a in tasks]
    report = {"models": {r["model"]: r for r in results}, "files": P._sha_of_outputs(build), "n_items": len(items)}
    scored = [r for r in results if not r.get("length_only")]
    report["passed"] = bool(scored) and all(r["passed"] is True or (is_mock() and r.get("skipped")) for r in scored)
    write_json(d / "token_check.json", report)
    lines = [f"{r['model']}: " + ("skipped (mock)" if r.get("skipped") else
                                  f"lengths only, {r['n_at_or_above_context']} prompts at or above {r['context']}"
                                  if r.get("length_only") else f"passed={r['passed']} problems={r['problems']}")
             for r in results]
    write_status(run_id, "\n".join(lines), ok=report["passed"])
    return 0 if report["passed"] else 1


def token_check_ok(cfg: dict, tag: str) -> tuple[bool, str]:
    """Has the token check passed for this model, at this commit (or one the config names), on exactly the item
    files scoring reads?"""
    run = cfg.get("token_check_run", "tau_family_token_check")
    if not _finished(run, tuple(cfg.get("accept_commits", {}).get(run, ()))):
        return False, f"{run} has not passed at this commit"
    path = runs_root() / run / "token_check.json"
    if not path.exists():
        return False, f"{run} wrote no token_check.json"
    rep = json.loads(path.read_text(encoding="utf-8"))
    if rep.get("files") != P._sha_of_outputs(P._build_dir(cfg)):
        return False, f"{run} checked other item files"
    rec = rep.get("models", {}).get(base_tag(tag))
    if not rec or rec.get("length_only"):
        return False, f"{run} did not check {base_tag(tag)}"
    if rec.get("passed") is True:
        return True, ""
    if is_mock() and rec.get("skipped"):
        return True, ""
    return False, f"{run}: {base_tag(tag)} failed ({rec.get('problems')})"


# ---------------- scoring ----------------


def prompt_tokens(scorer: LogoddsScorer, msgs: list[dict]) -> int | None:
    """The prompt's length in tokens as the scorer encodes it (None in mock mode: no tokenizer)."""
    return None if scorer.mock else len(F._ids(scorer._encode(msgs)))


def check_prompt_length(n: int | None, tag: str) -> None:
    if n is not None and n >= CONTEXT[base_tag(tag)]:
        raise SystemExit(f"{tag}: a prompt of {n} tokens is at or above the model's context ({CONTEXT[base_tag(tag)]})")


def readout_report(scorer: LogoddsScorer, spec: F.ModelSpec, t: dict, fewshot: list[dict]) -> dict[str, Any]:
    """In the scoring job, with the loaded tokenizer: are the four scored words single tokens, and the ones the
    template writes at the start of this item's replies and after its yes/no prompt? Mock: trivially yes."""
    if scorer.mock:
        return {"mock": True, "ok": True}
    ids = F._ids
    alone = {w: ids(scorer._encode_candidate(w)) for w in (P.FIRST_PLUS, P.FIRST_MINUS, "Yes", "No")}
    starts = {}
    try:
        return _readout_report(scorer, spec, t, fewshot, alone, starts)
    except F.TokenizationMismatch as e:
        return {"alone": alone, "ok": False, "error": str(e)}


def _readout_report(scorer: LogoddsScorer, spec: F.ModelSpec, t: dict, fewshot: list[dict], alone: dict,
                    starts: dict) -> dict[str, Any]:
    m = messages(spec, t, "aligned", fewshot, "first", "fail")
    for name, text in (("twin", t["success_twin"]), ("failure", t["failure_reply"])):
        starts[name] = F.template_reply(scorer, m, text)[:1]
    if t.get("witness_output"):
        mw = messages(spec, t, "aligned", fewshot, "first", "witness")
        starts["witness"] = F.template_reply(scorer, mw, t["witness_output"])[:1]
    my = messages(spec, t, "aligned", fewshot, "yn", "fail")
    yn = {w: F.template_reply(scorer, my, w) for w in ("Yes", "No")}
    ok = (all(len(v) == 1 for v in alone.values()) and starts["twin"] == alone[P.FIRST_PLUS]
          and starts.get("witness", alone[P.FIRST_PLUS]) == alone[P.FIRST_PLUS] and starts["failure"] == alone[P.FIRST_MINUS]
          and yn["Yes"] == alone["Yes"] and yn["No"] == alone["No"])
    return {"alone": alone, "reply_starts": starts, "yes_no_replies": yn, "ok": ok,
            "attn_implementation_loaded": getattr(getattr(scorer._model, "config", None), "_attn_implementation", None)}


def run_tau_family_score(cfg: dict) -> int:
    """tau_family_score_<tag>, tau_family_score_smoke and tau_family_placement_<tag>: sharded by user
    (WF_SHARD / WF_N_SHARDS), every score saved as it finishes; a resume refuses mixed commits, settings, item files
    or item hashes."""
    import os

    run_id = cfg["run_id"]
    require_filed(run_id)
    models = list(cfg["models"])
    for tag in models:
        if base_tag(tag) in EXCLUDED or base_tag(tag) not in SCORED:
            write_status(run_id, f"{tag} is not a scored model of this extension: {EXCLUDED.get(base_tag(tag), 'unknown')}", ok=False)
            return 1
    ok, why = P.verified(cfg)
    if not ok:
        write_status(run_id, f"items not verified: {why}", ok=False)
        return 1
    for tag in models:
        ok, why = token_check_ok(cfg, tag)
        if not ok:
            write_status(run_id, f"token check: {why}", ok=False)
            return 1
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    build = P._build_dir(cfg)
    d = _init_run(run_id, cfg, inputs=tuple(build / name for name in P.OUTPUT_FILES if (build / name).exists()))
    _, _, fewshot = P.load_built(cfg)
    every = primary_items(cfg)
    placement = bool(cfg.get("placement"))
    if placement:
        every = placement_items(every, int(cfg.get("n_items", PLACEMENT_ITEMS)))
    items = P.shard_by_user(every, i, n)
    hashes = {t["id"]: P.item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {score_key(s) for s in saved}
    for tag in models:
        spec = spec_for(tag + FIRST_TURN if placement else tag)
        todo = [(t, job) for t in items for job in (PLACEMENT_JOBS if placement else jobs_for(t))
                if (spec.tag, t["id"], *job) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(spec.hf_id, model_tag=spec.tag, attn_implementation=spec.attn_implementation,
                               template_kwargs=spec.template_kwargs)
        report = readout_report(scorer, spec, todo[0][0], fewshot)
        write_json(d / f"tokens_{spec.tag}.json", report)
        if not report.get("ok"):
            scorer.close()
            write_status(run_id, f"{spec.tag}: the scored tokens are not the template's (tokens_{spec.tag}.json)", ok=False)
            return 1
        for t, job in todo:
            arm, action, readout, cond = job
            msgs = messages(spec, t, cond, fewshot, readout, action)
            t0 = time.perf_counter()
            n_tok = prompt_tokens(scorer, msgs)
            check_prompt_length(n_tok, spec.tag)
            sc = scorer.score_one_token_pair(t["id"], cond, msgs, *READOUT_WORDS[readout])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "arm": arm, "action": action, "readout": readout,
                                                "user_id": t["user_id"], "item_hash": hashes[t["id"]],
                                                "prompt_tokens": n_tok, "seconds": round(time.perf_counter() - t0, 4)}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} items, {len({t['user_id'] for t in items})} users",
                 ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def _by(scores: list[dict], model: str) -> dict[tuple, dict]:
    return {(s["transition_id"], s.get("arm", "main"), s.get("action", "fail"), s["readout"], s["condition"]): s
            for s in scores if s["model"] == model}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def item_share(by: dict, items: list[dict], readout: str) -> dict[str, Any]:
    """The share of items whose L is higher under the aligned than under the swapped task (ties: not moved); null one
    half; user-clustered CI and the user-level sign-flip p of the per-item signs (descriptive)."""
    rows, users, signs, diffs = [], [], [], []
    for t in items:
        a, s = by.get((t["id"], "main", "fail", readout, "aligned")), by.get((t["id"], "main", "fail", readout, "swapped"))
        if a is None or s is None:
            continue
        d = a["L"] - s["L"]
        rows.append(1.0 if d > 0 else 0.0)
        signs.append((d > 0) - (d < 0))
        diffs.append(d)
        users.append(t["user_id"])
    boot = cluster_bootstrap(_mean, rows, users, seed=C.RNG_SEED)
    return {"estimate": _mean(rows), "ci": [boot["ci_low"], boot["ci_high"]], "null": 0.5, "n_items": len(rows),
            "perm_p": pair_permutation(signs, seed=C.RNG_SEED, cluster_ids=users) if signs else float("nan"),
            # amendment R1-S1: the median item effect beside the mean, and the share of exact ties (not moved)
            "median_item_effect": statistics.median(diffs) if diffs else float("nan"),
            "tie_share": _mean([1.0 if d == 0 else 0.0 for d in diffs])}


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def competence(by: dict, items: list[dict], readout: str, *, population: str = "discrimination") -> dict[str, Any]:
    """The families competence rule (wf.families.COMPETENCE) on τ-bench (design §7.3): under the neutral task, the
    witnesses are the determined successes and the failing calls the determined failures, on the discrimination
    population (items whose witness the text-only checker determines, PASS) or, descriptively, on all items."""
    pop = [t for t in items if t.get("witness_checker") == "PASS"] if population == "discrimination" else list(items)
    succ, fail, mass = [], [], []
    for t in pop:
        w = by.get((t["id"], "main", "witness", readout, "neutral"))
        f = by.get((t["id"], "main", "fail", readout, "neutral"))
        for rec, out in ((w, succ), (f, fail)):
            if rec is not None and rec["L"] == rec["L"]:
                out.append(rec["L"])
                mass.append(math.exp(rec["logp_plus"]) + math.exp(rec["logp_minus"]))
    rule = F.COMPETENCE
    auc = _auc(succ, fail)
    yes_f = _mean([1.0 if x > 0 else 0.0 for x in fail])
    yes_s = _mean([1.0 if x > 0 else 0.0 for x in succ])
    med = statistics.median(mass) if mass else float("nan")
    checks = {"auc_neutral": {"value": auc, "pass": auc >= rule["auc_neutral_min"]},
              "yes_share_neutral_failures": {"value": yes_f, "pass": yes_f <= rule["yes_share_neutral_failures_max"]},
              "yes_share_neutral_successes": {"value": yes_s, "pass": yes_s >= rule["yes_share_neutral_successes_min"]},
              "median_two_way_mass_neutral": {"value": med, "pass": med >= rule["median_yes_no_mass_neutral_min"]}}
    return {"interpretable": all(c["pass"] for c in checks.values()), "checks": checks, "rule": rule,
            "readout": readout, "population": population, "n_successes": len(succ), "n_failures": len(fail)}


def population_block(by: dict, pop: list[dict]) -> dict[str, Any]:
    """The failing call's readouts on the primary population (the τ design's block without the full replies)."""
    out: dict[str, Any] = {"n_items": len(pop), "n_users": len({t["user_id"] for t in pop})}
    for r in ("first", "yn"):
        out[r] = P.effects(by, pop, r)
    out["first_probability_shift"] = P.paired(by, pop, "first", "aligned", "swapped", scale="prob")
    out["first_product_clustered"] = P.paired(by, pop, "first", "aligned", "swapped", cluster="product")
    out["first_flips"] = P.flip_shares(by, pop)
    out["first_without_fewshot_tool_classes"] = P.effects(by, [t for t in pop if t["cls"] not in P.FEWSHOT_TOOL_CLASSES], "first")
    groups = P._group(pop, lambda t: t["cls"])
    out["by_class"] = {k: P.effects(by, v, "first") for k, v in sorted(groups.items()) if k in P.PRESPECIFIED_CLASSES}
    out["by_class_descriptive"] = {k: P.effects(by, v, "first", inference=False) for k, v in sorted(groups.items())
                                   if k in P.DESCRIPTIVE_CLASSES}
    out["item_share"] = {r: item_share(by, pop, r) for r in ("first", "yn")}
    return out


def controls_block(by: dict, pop: list[dict]) -> dict[str, Any]:
    """The controls the reading table needs, on the readouts this protocol scores (design §4.1)."""
    out: dict[str, Any] = {"witness": {r: P.effects(by, pop, r, action="witness") for r in ("first", "yn")}}
    determined = [t for t in pop if t.get("witness_checker") == "PASS"]
    out["discrimination"] = {r: P.discrimination(by, determined, r) for r in ("first", "yn")}
    out["discrimination_all_witnesses"] = {"first": P.discrimination(by, pop, "first")}
    out["discrimination_by_class"] = {}
    for k, v in sorted(P._group(pop, lambda t: t["cls"]).items()):
        if k not in P.PRESPECIFIED_CLASSES:
            continue
        det = [t for t in v if t.get("witness_checker") == "PASS"]
        if not det:
            out["discrimination_by_class"][k] = {"evidence": False, "n_items": 0, "holds": None,
                                                 "note": "no witness of this class is determined by its history"}
            continue
        blk = P.discrimination(by, det, "first")
        out["discrimination_by_class"][k] = {"evidence": True, "n_items": blk["n_items"], "first": blk,
                                             "holds": P.check_passes(blk)}
    out["mention"] = {"first": {"mention_aligned_minus_mention_swapped": P.paired(by, pop, "first", "mention_aligned", "mention_swapped"),
                                "mention_aligned_minus_neutral": P.paired(by, pop, "first", "mention_aligned", "neutral"),
                                "aligned_minus_mention_aligned": P.paired(by, pop, "first", "aligned", "mention_aligned")}}
    out["manipulation"] = P.paired(by, pop, "target", "aligned", "swapped")
    out["fail_minus_witness_effect"] = {r: P.contrast_of_effects(by, pop, r, ("fail", "aligned", "swapped"),
                                                                 ("witness", "aligned", "swapped")) for r in ("first", "yn")}
    # amendment R1-S4 (descriptive): fail minus witness anchored on the neutral line, which separates a lift under the
    # named order from a penalty under the other order (an authorization belief lowers swapped below neutral for both)
    out["fail_minus_witness_vs_neutral"] = {
        r: {f"{c}_minus_neutral": P.contrast_of_effects(by, pop, r, ("fail", c, "neutral"), ("witness", c, "neutral"))
            for c in ("aligned", "swapped")} for r in ("first", "yn")}
    # amendment R1-M1: per pre-specified class, the failing call's and the witness's effect and fail minus witness
    out["by_class"] = {}
    for k, v in sorted(P._group(pop, lambda t: t["cls"]).items()):
        if k in P.PRESPECIFIED_CLASSES:
            out["by_class"][k] = {
                "failing": P.paired(by, v, "first", "aligned", "swapped"),
                "witness": P.paired(by, v, "first", "aligned", "swapped", action="witness"),
                "fail_minus_witness": P.contrast_of_effects(by, v, "first", ("fail", "aligned", "swapped"),
                                                            ("witness", "aligned", "swapped"))}
    out["write_minus_mention_effect"] = {"first": P.contrast_of_effects(by, pop, "first", ("fail", "aligned", "swapped"),
                                                                        ("fail", "mention_aligned", "mention_swapped"))}
    return out


def format_block(by: dict) -> dict[str, Any]:
    out = P.format_block(by)
    toks = [x["prompt_tokens"] for x in by.values() if x.get("prompt_tokens") is not None]
    out["prompt_tokens"] = ({"median": statistics.median(toks), "max": max(toks), "n": len(toks)} if toks else None)
    return out


def model_block(by: dict, items: list[dict]) -> dict[str, Any]:
    ctl = controls_block(by, items)
    out = {"primary": population_block(by, items), "primary_controls": ctl,
           "competence": {"first": competence(by, items, "first"), "yn": competence(by, items, "yn"),
                          "first_all_items": competence(by, items, "first", population="all")},
           "checks": P.checks_of(ctl), "task_judgement_yes_no": task_judgement(by, items), "format": format_block(by)}
    out["description"] = describe(out)
    return out


CONFIDENT_JSON = 3.0  # notes/tau_dx.md: a first-token score of at least +3 "confidently predicts a JSON success"


def _intercepts(rows: list[tuple]) -> dict[str, float]:
    """Yes/no log-odds regressed on the first-token log-odds with one slope and an intercept per task line; rows are
    (line, first-token L, yes/no L)."""
    by: dict[str, list] = defaultdict(list)
    for c, x, y in rows:
        by[c].append((x, y))
    num = den = 0.0
    for v in by.values():
        mx, my = _mean([x for x, _ in v]), _mean([y for _, y in v])
        num += sum((x - mx) * (y - my) for x, y in v)
        den += sum((x - mx) ** 2 for x, _ in v)
    b = num / den if den > 0 else float("nan")
    a = {c: _mean([y for _, y in v]) - b * _mean([x for x, _ in v]) for c, v in by.items()}
    return {"slope": b, **{f"intercept_{c}": a[c] for c in a}}


def task_judgement(by: dict, items: list[dict]) -> dict[str, Any]:
    """Amendment R1-S4 (descriptive): does the yes/no answer carry a task judgement beyond the predicted output
    (notes/tau_dx.md, section 5)? Per call (failing, witness) and task line: among calls whose first token confidently
    predicts JSON (L1 >= +3), the share answered No on yes/no; and yes/no regressed on the first token with a common
    slope and one intercept per line, with user-clustered intervals for the intercept differences."""
    out: dict[str, Any] = {"confident_json_threshold": CONFIDENT_JSON}
    for action in ("fail", "witness"):
        share, rows, users = {}, [], []
        for c in CONDITIONS:
            sel = []
            for t in items:
                f, y = by.get((t["id"], "main", action, "first", c)), by.get((t["id"], "main", action, "yn", c))
                if f is None or y is None:
                    continue
                rows.append((c, f["L"], y["L"]))
                users.append(t["user_id"])
                if f["L"] >= CONFIDENT_JSON:
                    sel.append(1.0 if y["L"] < 0 else 0.0)
            share[c] = {"n_confident_json": len(sel), "share_no": _mean(sel)}
        fit = _intercepts(rows)
        diffs = {}
        for a, b in (("aligned", "neutral"), ("swapped", "neutral"), ("aligned", "swapped")):
            stat = lambda v, a=a, b=b: (lambda r: r.get(f"intercept_{a}", float("nan")) - r.get(f"intercept_{b}", float("nan")))(_intercepts(v))
            boot = cluster_bootstrap(stat, rows, users, seed=C.RNG_SEED)
            diffs[f"{a}_minus_{b}"] = {"estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]]}
        out[action] = {"share_no_when_first_token_confident_json": share, "fit": fit, "intercept_differences": diffs}
    return out


def _ci_below(block: dict | None) -> bool:
    return bool(block) and block.get("n_items", 0) > 0 and block["ci"][1] < 0


def describe(blk: dict[str, Any]) -> dict[str, Any]:
    """Amendment R1-M1 and R1-S1: the extension's own description of a model's effect, beside (never instead of) the τ
    verdict and its outcome-table reading.

    - row: the mechanical criterion of the τ table, "fail minus witness above 0" or "fail minus witness not above 0";
    - qualifiers always reported: the witness's effect as a share of the failing call's; per class (status,
      unavailable) the failing call's effect and fail minus witness;
    - label: "failure-specific" only if fail minus witness has its interval above 0, the witness moves less than half as
      much as the failing call, and fail minus witness is not below 0 (interval below 0) in either class; otherwise
      "a shift shared by working and failing calls";
    - class_mix: the two classes' failing-call effects have intervals on opposite sides of 0: the pooled effect depends on
      the class mix and is reported per class;
    - minority: the item share's interval includes or lies below one half: the mean is carried by a minority of items.
    """
    ctl = blk["primary_controls"]
    fail = blk["primary"]["first"]["aligned_minus_swapped"]
    wit = ctl["witness"]["first"]["aligned_minus_swapped"]
    fmw = ctl["fail_minus_witness_effect"]["first"]
    share = wit["estimate"] / fail["estimate"] if fail["estimate"] == fail["estimate"] and fail["estimate"] > 0 else float("nan")
    classes = {k: {"failing": v["failing"], "fail_minus_witness": v["fail_minus_witness"]} for k, v in ctl.get("by_class", {}).items()}
    class_fmw_below = [k for k, v in classes.items() if _ci_below(v["fail_minus_witness"])]
    fs = bool(P._ci_above(fmw) and share == share and share < 0.5 and not class_fmw_below)
    cls_fail = [v["failing"] for v in classes.values()]
    class_mix = (len(cls_fail) == 2 and ((P._ci_above(cls_fail[0]) and _ci_below(cls_fail[1]))
                                         or (_ci_below(cls_fail[0]) and P._ci_above(cls_fail[1]))))
    item = blk["primary"]["item_share"]["first"]
    minority = bool(item["n_items"] > 0 and not item["ci"][0] > 0.5)
    label = "failure-specific" if fs else "a shift shared by working and failing calls"
    if not fs and _ci_below(fmw):
        label += " (working calls move more)"
    return {"row": "fail minus witness above 0" if P._ci_above(fmw) else "fail minus witness not above 0",
            "label": label, "failure_specific": fs, "witness_share_of_failing_effect": share,
            "fail_minus_witness": fmw, "witness_effect": wit, "failing_effect": fail, "by_class": classes,
            "classes_with_fail_minus_witness_below_0": class_fmw_below,
            "class_mix": class_mix,
            "class_mix_reading": "the pooled effect depends on the class mix; reported per class" if class_mix else None,
            "minority": minority, "item_share_first": item["estimate"], "item_share_first_ci": item["ci"],
            "median_item_effect": item.get("median_item_effect"),
            "minority_reading": ("the mean is carried by a minority of items (item share's interval not above one half); "
                                 "reported as such, not as a general shift") if minority else None}


def family_verdicts(blocks: dict[str, dict], models: tuple[str, ...] = SCORED) -> dict[str, Any]:
    """The τ per-model rule (pipeline.verdicts: Holm over `models`, a fixed family; a missing model enters with
    p = 1; the manipulation and discrimination gates; the outcome table), then the families competence rule on the
    first-token readout as a third gate (design §7.1-§7.3). No joint verdict."""
    primary = {m: blocks[m]["primary"]["first"]["aligned_minus_swapped"] for m in models if m in blocks}
    checks = {m: blocks[m]["checks"] for m in models if m in blocks}
    v = P.verdicts(primary, checks, models=models)
    per = v["per_model"]
    for m in models:
        if m not in blocks:
            continue
        comp = blocks[m]["competence"]["first"]
        per[m]["competence_first_token_interpretable"] = comp["interpretable"]
        per[m]["competence_yes_no_interpretable"] = blocks[m]["competence"]["yn"]["interpretable"]
        if not comp["interpretable"]:
            per[m]["verdict_before_competence"] = per[m]["verdict"]
            per[m]["verdict"] = NOT_INTERPRETABLE
            per[m]["reading"] = ("Not interpretable: on these prompts the model is not shown to tell working calls from "
                                 "failing ones, or answers one way to everything; no statement about goal sensitivity")
    for m in models:  # amendment R1-S1 and R1-M1: qualifiers beside the verdict (the verdict itself is unchanged)
        if m in blocks and "description" in blocks[m]:
            dsc = blocks[m]["description"]
            q = []
            if per[m]["verdict"].startswith(P.POSITIVE) and dsc["minority"]:
                q.append(dsc["minority_reading"])
            if dsc["class_mix"]:
                q.append(dsc["class_mix_reading"])
            if per[m]["verdict"].startswith(P.POSITIVE) and not per[m].get("discrimination_holds"):
                q.append("the model is not shown to use the visible record (discrimination fails)")
            per[m]["qualifiers"] = q
    return {"per_model": per, "family": list(models), "alpha_holm": P.P_MAX,
            "correction": "Holm over the scored models at a fixed size (a missing model enters with p = 1)"}


def interpretable_for_comparison(verdict: str) -> bool:
    return not verdict.startswith(("not interpretable", "inconclusive", "not scored", "not computable"))


def specificity(entry: dict, description: dict | None) -> str | None:
    """Amendment R1-M1: the cross-model label of a positive, interpretable verdict with discrimination holding: the
    extension's `describe` label ("failure-specific" only under the three pre-stated conditions, otherwise "a shift
    shared by working and failing calls"), never the τ outcome table's row; None otherwise."""
    if not entry["verdict"].startswith(P.POSITIVE) or not entry.get("discrimination_holds") or not description:
        return None
    return "failure-specific" if description["failure_specific"] else "shared"


def _paired_draws(rows: list[tuple], users: list[str], stat, B: int = C.BOOTSTRAP_RESAMPLES) -> list:
    """Paired cluster bootstrap: users resampled once per draw, both models read on the same resample."""
    rng = random.Random(C.RNG_SEED)
    by: dict[str, list] = defaultdict(list)
    for r, u in zip(rows, users):
        by[u].append(r)
    keys = sorted(by)
    return [stat([r for u in rng.choices(keys, k=len(keys)) for r in by[u]]) for _ in range(B)] if keys else []


def _pct(v: list[float], q: float) -> float:
    v = sorted(x for x in v if x == x)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def compare(by_a: dict, by_b: dict, items: list[dict], readout: str = "first") -> dict[str, Any]:
    """Model A against model B on the items both have (design §7.5, descriptive): the item-share difference (A minus
    B), "differs" when its 95% paired interval excludes 0 and "matches" when its 90% interval lies inside
    ±SHARE_MARGIN; and the ΔL difference in nats, "differs" when its 95% interval excludes 0 and it is at least the
    noise floor in size."""
    rows, users = [], []
    for t in items:
        k = [(t["id"], "main", "fail", readout, c) for c in ("aligned", "swapped")]
        if all(x in by_a for x in k) and all(x in by_b for x in k):
            da, db = by_a[k[0]]["L"] - by_a[k[1]]["L"], by_b[k[0]]["L"] - by_b[k[1]]["L"]
            rows.append(((da > 0) - 0.0, (db > 0) - 0.0, da, db))
            users.append(t["user_id"])
    share = lambda v: _mean([a - b for a, b, _, _ in v])
    dl = lambda v: _mean([x - y for _, _, x, y in v])
    draws = _paired_draws(rows, users, lambda v: (share(v), dl(v)))
    s_est, l_est = share(rows), dl(rows)
    s95 = [_pct([d[0] for d in draws], 0.025), _pct([d[0] for d in draws], 0.975)]
    s90 = [_pct([d[0] for d in draws], 0.05), _pct([d[0] for d in draws], 0.95)]
    l95 = [_pct([d[1] for d in draws], 0.025), _pct([d[1] for d in draws], 0.975)]
    s_differs = bool(rows) and (s95[0] > 0 or s95[1] < 0)
    s_matches = bool(rows) and -F.SHARE_MARGIN < s90[0] and s90[1] < F.SHARE_MARGIN
    l_differs = bool(rows) and (l95[0] > 0 or l95[1] < 0) and abs(l_est) >= F.NOISE_FLOOR_NATS
    return {"readout": readout, "n_items": len(rows), "n_users": len(set(users)),
            "item_share_difference": {"estimate": s_est, "ci95": s95, "ci90": s90, "margin": F.SHARE_MARGIN,
                                      "differs": s_differs, "matches": s_matches,
                                      "reading": ("differs, within the margin" if s_differs and s_matches else "differs" if s_differs
                                                  else "matches" if s_matches else "no difference detected")},
            "delta_L_difference_nats": {"estimate": l_est, "ci95": l95, "noise_floor": F.NOISE_FLOOR_NATS,
                                        "reading": "differs" if l_differs else "no difference detected"}}


def placement_block(by_native: dict, by_first: dict, items: list[dict]) -> dict[str, Any]:
    """Mistral's goal effect (first token, aligned minus swapped) under the native and the first-turn placement on the
    placement items, and the per-item difference (first turn minus native); format-dependent when that difference's
    CI excludes 0 and it exceeds a fifth of the native effect in size (design §7.6)."""
    comb = {**{k: v for k, v in by_native.items()}, **{(k[0], "firstturn", *k[2:]): v for k, v in by_first.items()}}
    nat = P.paired(by_native, items, "first", "aligned", "swapped")
    alt = P.paired(comb, items, "first", "aligned", "swapped", arm="firstturn")
    diff = P.contrast_of_effects(comb, items, "first", ("fail", "aligned", "swapped"), ("fail", "aligned", "swapped"),
                                 arms=("firstturn", "main"))
    excludes = diff["n_items"] > 0 and (diff["ci"][0] > 0 or diff["ci"][1] < 0)
    dependent = bool(excludes and abs(diff["estimate"]) > abs(nat["estimate"]) / 5)
    return {"n_items": diff["n_items"], "native": nat, "first_turn": alt, "first_turn_minus_native": diff,
            "format_dependent": dependent,
            "reading": ("Mistral's τ result depends on prompt format: both numbers are reported in every comparison"
                        if dependent else "placement is not shown to move Mistral's effect on this sample")}


def pattern_readings(per: dict[str, dict], comparisons: dict[str, dict], descriptions: dict[str, dict] | None = None,
                     reference: dict[str, dict] | None = None) -> dict[str, Any]:
    """The descriptive readings of design §7.5, as amended after review 1 (R1-M1, R1-S1): labels by the pre-stated
    rule of `describe`, never by the τ outcome table's row; the class-mix and minority qualifiers; the reference models
    described by the same rule."""
    descriptions = descriptions or {}
    scored = {m: per[m] for m in SCORED if m in per}
    readable = {m: x for m, x in scored.items() if interpretable_for_comparison(x["verdict"])}
    rows = {m: specificity(x, descriptions.get(m)) for m, x in readable.items()}
    out: dict[str, Any] = {"left_out": {m: x["verdict"] for m, x in scored.items() if m not in readable},
                           "labels": rows, "readings": [],
                           "reference_labels": {m: (d["label"] if d else None) for m, d in (reference or {}).items()}}
    if len(readable) == len(SCORED) and all(x.get("ci_above_0") for x in readable.values()):
        caveats = []
        for m in SCORED:
            d = descriptions.get(m) or {}
            if d.get("minority"):
                caveats.append(f"{m}: carried by a minority of items")
            if not readable[m].get("discrimination_holds"):
                caveats.append(f"{m}: not shown to use the visible record")
        out["readings"].append("The direction of the τ goal effect is not specific to Qwen3-8B and Llama-3.1-8B: it "
                               "appears in every interpretable model tested on τ-bench"
                               + (f" (qualified: {'; '.join(caveats)})" if caveats else ""))
    vals = [rows.get(m) for m in SCORED]
    if len(readable) == len(SCORED) and None not in vals:
        if set(vals) == {"shared"}:
            out["readings"].append("In both scored models the task shifts working and failing calls alike (or working "
                                   "calls more), as in both reference models (notes/tau_dx.md); no model tested shows a "
                                   "shift specific to failing calls by the pre-stated rule")
        elif set(vals) == {"failure-specific"}:
            out["readings"].append("Both scored models show a shift specific to failing calls by the pre-stated rule, "
                                   "which neither reference model meets")
        else:
            out["readings"].append("The pattern varies by model; with one model per developer outside Meta, no family "
                                   "or developer explanation is drawn")
    for m in SCORED:
        d = descriptions.get(m) or {}
        if d.get("class_mix"):
            out["readings"].append(f"{m}: the pooled effect depends on the class mix; reported per class")
    c = comparisons.get("llama3.2-3b vs llama3.1-8b-fewshot", {})
    ref = (reference or {}).get("llama3.1-8b-fewshot")
    if (rows.get("llama3.2-3b") and ref and rows["llama3.2-3b"] == ("failure-specific" if ref["failure_specific"] else "shared")
            and c.get("item_share_difference", {}).get("reading") == "matches"):
        out["readings"].append("Llama 3.2 3B has Llama-3.1-8B's label by the pre-stated rule and their item shares "
                               "match: descriptive similarity within the Llama line; no cause is named")
    return out


def reference_check(blocks: dict[str, dict], published_path: Path | None) -> dict[str, Any]:
    """Do the recomputed reference primary estimates and intervals equal the published ones (notes/tau_metrics.json)?"""
    if published_path is None:
        return {"checked": False, "why": "no published metrics named"}
    pub = json.loads(Path(published_path).read_text(encoding="utf-8"))
    out: dict[str, Any] = {"checked": True, "path": str(published_path), "models": {}}
    for m in REFERENCE:
        if m not in blocks:
            out["models"][m] = {"equal": False, "why": "reference model not read"}
            continue
        mine = blocks[m]["primary"]["first"]["aligned_minus_swapped"]
        theirs = pub["models"][m]["primary"]["first"]["aligned_minus_swapped"]
        eq = all(abs(a - b) <= 1e-9 for a, b in zip([mine["estimate"], *mine["ci"]], [theirs["estimate"], *theirs["ci"]]))
        out["models"][m] = {"equal": eq and mine["n_items"] == theirs["n_items"], "recomputed": [mine["estimate"], mine["ci"]],
                            "published": [theirs["estimate"], theirs["ci"]]}
    out["all_equal"] = all(x["equal"] for x in out["models"].values())
    return out


def _read_reference(cfg: dict, items: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """The reference models' records of exactly this protocol's jobs on these items, from the reference τ scoring run;
    refuses a run at another commit than the ones named, and counts records whose item hash differs."""
    from wf.rq2 import complete_shards

    run = cfg.get("reference_run", "tau_score_reference")
    folders, problems = complete_shards(run)
    info: dict[str, Any] = {"run": run, "problems": problems}
    commits = tuple(cfg.get("reference_commits", ()))
    shas = {(p / "git_sha").read_text().strip() for p in folders if (p / "git_sha").exists()}
    if commits and not all(any(s.startswith(c) for c in commits) for s in shas):
        info["problems"] = problems + [f"reference shards at {sorted(shas)}, not {list(commits)}"]
    if info["problems"]:
        return [], info
    ids = {t["id"]: P.item_hash(t) for t in items}
    wanted = {(m, i, *job) for m in REFERENCE for i in ids for job in PROTOCOL_JOBS}
    recs = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if score_key(x) in wanted]
    info["stale"] = sum(1 for x in recs if ids[x["transition_id"]] != x.get("item_hash"))
    info["missing"] = len(wanted - {score_key(x) for x in recs})
    info["git_sha"] = sorted(shas)
    info["folders"] = [str(p) for p in folders]
    return recs, info


def sd_of_differences(by: dict, items: list[dict], readout: str) -> dict[str, Any]:
    return P.sd_of_differences(by, items, readout)


def projection(cfg: dict, report: dict, items_all: list[dict]) -> dict[str, Any]:
    """From the smoke run: hours per shard of each model's full run (mean seconds per job by readout), the largest
    against the job limit, the model's total (with Mistral's placement check) against the budget, and the projected
    precision of the primary contrast. No estimate of any effect is used."""
    limit = float(cfg.get("time_limit_hours", float("nan")))
    out: dict[str, Any] = {"time_limit_hours": limit, "budget_gpu_hours": BUDGET_GPU_HOURS, "models": {}}
    for m, blk in report["models"].items():
        sec = blk["format"]["seconds_mean_by_readout"]
        target = load_config(f"tau_family_score_{m}")
        n = int(target.get("n_shards", 1))
        hours = [sum(sec.get(j[2], float("nan")) for t in P.shard_by_user(items_all, i, n) for j in jobs_for(t)) / 3600
                 for i in range(n)]
        place = (len(placement_items(items_all)) * len(PLACEMENT_JOBS) * sec.get("first", float("nan")) / 3600
                 if m == PLACEMENT_MODEL else 0.0)
        smoke = blk["format"]["seconds_total"] / 3600
        prec = {r: {"smoke_sd": s["sd"], "smoke_n": s["n"], "primary": P.precision(s["sd"], items_all)}
                for r, s in blk["sd_of_differences"].items()}
        total = sum(hours) + place + smoke
        out["models"][m] = {"n_shards": n, "hours_per_shard": hours, "hours_max_shard": max(hours),
                            "fits_time_limit": max(hours) <= limit, "hours_placement": place, "hours_smoke": smoke,
                            "gpu_hours_total": total, "within_budget": total <= BUDGET_GPU_HOURS, "precision": prec}
    out["fits_time_limit"] = bool(out["models"]) and all(v["fits_time_limit"] for v in out["models"].values())
    out["within_budget"] = bool(out["models"]) and all(v["within_budget"] for v in out["models"].values())
    return out


def comparison_gate(report: dict, model: str, readout: str) -> str | None:
    """Why a model's comparisons on a readout are not read (amendment R1-S6), or None: a scored model whose verdict is
    not interpretable or gated to inconclusive, or any model failing the competence rule on that readout."""
    v = report.get("verdicts", {}).get("per_model", {}).get(model)
    if v is not None and not interpretable_for_comparison(v["verdict"]):
        return f"{model}: {v['verdict']}"
    blk = report.get("models", {}).get(model) or report.get("reference", {}).get(model)
    comp = (blk or {}).get("competence", {}).get(readout)
    if comp is not None and not comp["interpretable"]:
        return f"{model}: fails the competence rule on the {readout} readout"
    return None


def mark_comparisons(report: dict) -> None:
    """Every comparison entry says whether it is read; an entry with a gated model has its readings replaced by
    'not read: <why>' (the computed reading kept under reading_unread), so no unmarked 'differs' or 'matches' remains."""
    for key, by_readout in report.get("comparisons", {}).items():
        a, b = key.split(" vs ")
        for readout, cmp in by_readout.items():
            why = [w for w in (comparison_gate(report, a, readout), comparison_gate(report, b, readout)) if w]
            cmp["read"] = not why
            cmp["not_read_because"] = why
            if why:
                for part in ("item_share_difference", "delta_L_difference_nats"):
                    cmp[part]["reading_unread"] = cmp[part]["reading"]
                    cmp[part]["reading"] = "not read: " + "; ".join(why)


def run_tau_family_analysis(cfg: dict) -> int:
    """tau_family_analysis, and tau_family_smoke_analysis (with `project_to`: format, timing and precision only)."""
    from wf.rq2 import complete_shards

    run_id = cfg["run_id"]
    require_filed(run_id)
    scfg = {**cfg, "build_run": cfg.get("build_run", "tau_build")}
    smoke = bool(cfg.get("project_to"))
    sources = {m: (cfg["source"] if smoke else f"tau_family_score_{m}") for m in cfg["models"]}
    shard_folders: dict[str, list[Path]] = {}
    waiting: dict[str, list[str]] = {}
    for m, src in sources.items():
        folders, problems = complete_shards(src)
        shard_folders[m] = folders
        if problems:
            waiting[m] = problems
    inputs = tuple(p / "scoring.jsonl" for fs in shard_folders.values() for p in fs if (p / "scoring.jsonl").exists())
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=inputs)
    if waiting:
        write_status(run_id, "waits for every scoring shard:\n" + "\n".join(f"{m}: {p}" for m, ps in waiting.items() for p in ps), ok=False)
        return 1
    if smoke:  # the items the smoke run scored, from its own saved settings
        saved = json.loads((shard_folders[cfg["models"][0]][0] / "config.json").read_text(encoding="utf-8"))
        every = primary_items({**saved, "build_run": scfg["build_run"]})
    else:
        every = primary_items(scfg)
    hashes = {t["id"]: P.item_hash(t) for t in every}
    scores = {m: [x for p in shard_folders[m] for x in read_jsonl(p / "scoring.jsonl") if x["model"] == m] for m in cfg["models"]}
    wanted = {m: {(m, t["id"], *job) for t in every for job in jobs_for(t)} for m in cfg["models"]}
    missing = {m: len(wanted[m] - {score_key(x) for x in scores[m]}) for m in cfg["models"]}
    stale = {m: sum(1 for x in scores[m] if hashes.get(x["transition_id"]) != x.get("item_hash")) for m in cfg["models"]}
    report: dict[str, Any] = {"items": {"primary": len(every), "primary_users": len({t["user_id"] for t in every})},
                              "missing_scores": missing, "stale_scores": stale, "models": {},
                              "excluded": EXCLUDED, "design": "notes/tau_families_design.md"}
    ok = not any(missing.values()) and not any(stale.values())
    tokens = {}
    for m, fs in shard_folders.items():
        for p in fs:
            for f in p.glob("tokens_*.json"):
                tokens.setdefault(f.stem[len("tokens_"):], json.loads(f.read_text(encoding="utf-8")))
    report["token_reports"] = tokens
    by_model = {m: _by(scores[m], m) for m in cfg["models"]}
    if smoke:  # format, timing and precision only: no estimate, no verdict is computed or printed
        for m in cfg["models"]:
            report["models"][m] = {"format": format_block(by_model[m]),
                                   "sd_of_differences": {r: sd_of_differences(by_model[m], every, r) for r in ("first", "yn")}}
        full_items = primary_items(scfg)
        report["projection"] = projection(cfg, report, full_items)
        write_json(d / "metrics.json", report)
        write_run_report(run_id, "τ-bench wave-A families smoke run: format, timing and precision only (no estimates)", report)
        pr = report["projection"]
        write_status(run_id, f"items={report['items']} missing={missing} stale={stale}\n" + "\n".join(
            f"{m}: largest shard {v['hours_max_shard']:.2f} h (limit {pr['time_limit_hours']}); total {v['gpu_hours_total']:.2f} "
            f"GPU-hours (budget {BUDGET_GPU_HOURS})" for m, v in pr["models"].items()),
                     ok=False, reported="τ families smoke: format and timing" if ok else None)
        return 0 if ok else 1
    for m in cfg["models"]:
        report["models"][m] = model_block(by_model[m], every)
    ref_recs, ref_info = _read_reference(cfg, every)
    report["reference_run"] = ref_info
    ref_by = {m: _by(ref_recs, m) for m in REFERENCE} if ref_recs else {}
    report["reference"] = {m: model_block(b, every) for m, b in ref_by.items()}
    pub = cfg.get("published_metrics")
    report["reference_check"] = reference_check(report["reference"], (REPO_ROOT / pub) if pub else None)
    # amendment R1-S2: the family is always both scored models; a model absent from this config enters with p = 1
    report["verdicts"] = family_verdicts(report["models"], SCORED)
    report["sensitivity_without_fewshot_tool_classes"] = P.sensitivity_verdicts(
        {m: report["models"][m]["primary"]["first_without_fewshot_tool_classes"]["aligned_minus_swapped"] for m in report["models"]},
        {m: report["models"][m]["primary"]["first"]["aligned_minus_swapped"] for m in report["models"]},
        models=SCORED)
    for m, x in report["verdicts"]["per_model"].items():
        blk = report["models"].get(m)
        x["ci_above_0"] = bool(blk) and P._ci_above(blk["primary"]["first"]["aligned_minus_swapped"])
    everyone = {**by_model, **ref_by}
    pairs = [(a, b) for a in cfg["models"] for b in ref_by] + [(a, b) for i, a in enumerate(cfg["models"]) for b in cfg["models"][i + 1:]]
    report["comparisons"] = {f"{a} vs {b}": {r: compare(everyone[a], everyone[b], every, r) for r in ("first", "yn")}
                             for a, b in pairs}
    report["comparisons_note"] = ("descriptive; a model not interpretable or gated to inconclusive is left out of every "
                                  "cross-model reading (design §7.5); the yes/no comparison is read only for models "
                                  "interpretable on the yes/no readout")
    mark_comparisons(report)
    report["patterns"] = pattern_readings(report["verdicts"]["per_model"],
                                          {k: v["first"] for k, v in report["comparisons"].items()},
                                          {m: b["description"] for m, b in report["models"].items()},
                                          {m: b["description"] for m, b in report["reference"].items()})
    place_run = cfg.get("placement_run")
    if place_run and PLACEMENT_MODEL in by_model:
        folders, problems = complete_shards(place_run)
        if problems:
            report["placement"] = {"waiting": problems}
            ok = False
        else:
            alt = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == PLACEMENT_MODEL + FIRST_TURN]
            pitems = placement_items(every, int(cfg.get("placement_items", PLACEMENT_ITEMS)))
            report["placement"] = {**placement_block(by_model[PLACEMENT_MODEL], _by(alt, PLACEMENT_MODEL + FIRST_TURN), pitems),
                                   "run": place_run}
    if ref_info.get("problems") or ref_info.get("missing") or ref_info.get("stale"):
        ok = False
    if report["reference_check"].get("checked") and not report["reference_check"].get("all_equal"):
        ok = False  # the recomputed reference results differ from the published ones: something is wrong
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "τ-bench retail, wave-A model families: does the task override a visible fact?", report)
    lines = [f"{m}: ΔL1 {report['models'][m]['primary']['first']['aligned_minus_swapped']['estimate']:.3f} "
             f"CI {report['models'][m]['primary']['first']['aligned_minus_swapped']['ci']} -> {v['verdict']}"
             for m, v in report["verdicts"]["per_model"].items() if m in report["models"]]
    write_status(run_id, f"items={report['items']} missing={missing} stale={stale} reference={ref_info.get('problems') or 'ok'}\n"
                 + "\n".join(lines), ok=False, reported="τ families verdicts" if ok else None)
    return 0 if ok else 1
