"""Post-training pilot: score one checkpoint's arms (notes/posttraining_design.md, sections 4-7).

    python -m wf.posttrain_score --ckpt tulu-sft --arms tulu-sft_plain,tulu-sft_native,tulu-sft_w2w [--resume]
    python -m wf.posttrain_score --ckpt tulu-sft --arms ... --smoke 5      # timing and validity only

Every number is produced by the lexsem diagnostic's own functions (scripts/lexsem/score.py: `load`, `sequence`,
`score_one`, `readout_ids`, `token_checks`, `environment`), so a prompt and its score here are those lexsem would
produce for the same checkpoint, item, condition and format: one unpadded sequence per forward pass, bf16 weights, a
float32 log-softmax of the bf16 logits. Only the bookkeeping is this module's: which cells to score, run folders with
commit, config and input hashes (wf.runs._init_run), resume by item, refusal of mock mode and of a dirty tree.

Output per arm, in runs/posttrain/<arm>/: scoring.jsonl (one line per item and item set), env.json, recognition.jsonl
(plain arms of the main kind), plus the standard config.json, inputs.json, git_sha and jobs.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from wf import lexsem
from wf import posttrain as P

FAILURE_TEXT = "Nothing happens."


def _ls():
    from scripts.lexsem import score as ls  # the lexsem scorer's functions (same code path)

    return ls


def count_occurrences(seq: list[int], sub: list[int]) -> int:
    return sum(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def messages(arm: str, t: dict, cond: str, fmt: str, specs: dict | None) -> list[dict]:
    a = P.ARMS[arm]
    if a["kind"] == "bridge":
        if fmt == "yn":
            raise ValueError("the bridge has no yes/no format")
        return P.bridge_messages(t, cond)
    return P.fewshot_messages(t, P.spec(t, cond, specs), "obs" if fmt == "fail" else fmt, a.get("fewshot"))


def build_jobs(tok, arm: str, t: dict, cells: list[tuple[str, str]], specs: dict | None, item_set: str) -> list[dict]:
    """The sequences of one item. Token checks: in plain mode the tokenizer's own BOS exactly once (lexsem's
    check_bos, inside prompt_ids); on the failures, a name condition adds exactly one copy of the name's tokens to
    the swapped prompt and no other condition adds one (lexsem design section 5); the observation text starts with
    the readout token on every failure."""
    ls = _ls()
    mode = P.ARMS[arm]["mode"]
    jobs = []
    check_names = item_set == "failure" and P.ARMS[arm]["kind"] != "bridge"
    if check_names:
        name_ids = tok(" " + t["goal_object"], add_special_tokens=False)["input_ids"]
        base = count_occurrences(ls.prompt_ids(tok, messages(arm, t, "swapped", "obs", specs), mode), name_ids)
    for cond, fmt in cells:
        msgs = messages(arm, t, cond, fmt, specs)
        cont = {"obs": t["candidates"]["o_plus"], "fail": FAILURE_TEXT}.get(fmt)
        ids, n_p = ls.sequence(tok, msgs, mode, cont)
        if check_names and fmt == "obs" and (cond in lexsem.MENTIONS or cond in (P.DESC, P.NEUTRAL)):
            want = 1 if cond in lexsem.MENTIONS and lexsem.MENTIONS[cond][0] == "name" else 0
            added = count_occurrences(ids[:n_p], name_ids) - base
            if added != want:
                raise SystemExit(f"{t['id']} {cond}: adds {added} copies of the name's tokens, expected {want}")
        jobs.append({"cond": cond, "fmt": fmt, "ids": ids, "n_prompt": n_p})
    return jobs


def readouts(jobs: list[dict], rows: list, pairs: dict, tok, item_set: str) -> dict:
    """lexsem's per-sequence record (readout-token log-probabilities and raw logits at the reply's first position,
    every continuation token's log-probability), plus the continuation's first token. A success observation may
    start with another word than "You" ("There's ...", "The ..."); its first-token decision is then read from its own
    first token (design section 5). Every failure observation must start with the readout token."""
    out: dict = {}
    for j, (lp, z) in zip(jobs, rows):
        a, b = pairs["obs" if j["fmt"] == "fail" else j["fmt"]]
        r = {"lp_a": round(lp[0, a].item(), 5), "lp_b": round(lp[0, b].item(), 5), "z_a": round(z[a].item(), 4),
             "z_b": round(z[b].item(), 4), "n_prompt": j["n_prompt"], "n_tokens": len(j["ids"]),
             "ids_sha": hashlib.sha256(json.dumps(j["ids"]).encode()).hexdigest()[:16]}
        if j["fmt"] in ("obs", "fail"):
            cont = j["ids"][j["n_prompt"]:]
            r["tok_lp"] = [round(lp[k, cont[k]].item(), 5) for k in range(len(cont))]
            r["toks"] = [tok.decode([x]) for x in cont]
            r["first_is_readout"] = cont[0] == (a if j["fmt"] == "obs" else b)
            if j["fmt"] == "fail" and not r["first_is_readout"]:
                raise SystemExit(f"failure text does not start with its readout token: {r['toks'][0]!r}")
            if item_set == "failure" and not r["first_is_readout"]:
                raise SystemExit(f"a failure item's observation does not start with the readout token: {r['toks'][0]!r}")
        out.setdefault(j["cond"], {})[j["fmt"]] = r
    return out


def strict_load_check(model, repo: str, rev: str) -> dict:
    """Loading gate (design section 5): the checkpoint's weight names (its safetensors index) against the loaded
    model's parameters and buffers; any name on one side only is reported, and the run refuses to score."""
    from huggingface_hub import hf_hub_download

    idx = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=rev)).read_text())
    ckpt = set(idx["weight_map"])
    loaded = set(model.state_dict())
    tied = {"lm_head.weight"} if getattr(model.config, "tie_word_embeddings", False) else set()
    missing = sorted(loaded - ckpt - tied)
    unexpected = sorted(ckpt - loaded)
    return {"ok": not missing and not unexpected, "missing": missing[:20], "unexpected": unexpected[:20],
            "n_checkpoint": len(ckpt), "n_loaded": len(loaded)}


def recognition(tok, model, objects: list[str]) -> list[dict]:
    """The recognition check of the described goal (design section 4), teacher-forced in plain text with the
    tokenizer's own BOS. For each goal object, every ALFWorld name with a description (55) is scored after
    RECOG_PROMPT with the object's description and after the same prompt with a neutral phrase; the name's pointwise
    association is the difference. The object passes if its own name ranks among the top RECOG_TOP by that
    difference."""
    from wf import rq1x

    ls = _ls()
    names = sorted(rq1x.DESCRIPTIONS)

    def lp_names(text: str) -> dict[str, float]:
        prompt = tok(P.RECOG_PROMPT.format(desc=text), add_special_tokens=True)["input_ids"]
        out = {}
        for name in names:
            cont = tok(" " + name, add_special_tokens=False)["input_ids"]
            lp, _ = ls.score_one(model, prompt + cont, len(prompt), True)
            out[name] = sum(lp[k, cont[k]].item() for k in range(len(cont)))
        return out

    neutral = lp_names(P.RECOG_NEUTRAL)
    res = []
    for obj in objects:
        given = lp_names(rq1x.DESCRIPTIONS[obj])
        pmi = {n: given[n] - neutral[n] for n in names}
        order = sorted(names, key=lambda n: -pmi[n])
        rank = order.index(obj) + 1
        res.append({"object": obj, "rank": rank, "top": order[:5], "pmi_object": round(pmi[obj], 5),
                    "passed": rank <= P.RECOG_TOP})
    return res


def item_sets(arm: str, its: dict, reuse: bool) -> list[tuple[str, list[dict], list[tuple[str, str]]]]:
    cells = P.cells_to_score(arm, reuse)
    kind = P.ARMS[arm]["kind"]
    fails = its["failure"]
    if kind == "floor":
        fails = [t for t in fails if t["lexsem_stratum"] == "later"]  # all 300 later items (Rule 4: at least 300)
    return [("failure", fails, cells["failure"]), ("success", its["success"] if cells["success"] else [],
                                                   cells["success"])]


def run_arm(arm: str, tok, model, rev: str, its: dict, recs_specs: dict, cfg: dict, pairs: dict,
            recog_objects: list) -> None:
    from wf.runs import _init_run
    from wf.rq2 import SOURCES
    from wf.util import append_jsonl, read_jsonl, runs_root, write_json, write_status  # noqa: F401

    ls = _ls()
    a = P.ARMS[arm]
    run_id = f"posttrain/{arm}"
    inputs = tuple(runs_root() / s / "transitions.jsonl" for s in SOURCES)
    d = _init_run(run_id, {**cfg, "run_id": run_id, "arm": arm, **{k: v for k, v in a.items()},
                           "repo": P.CHECKPOINTS[a["ckpt"]]["repo"], "revision": rev}, inputs=inputs)
    env = ls.environment(P.CHECKPOINTS[a["ckpt"]]["repo"], rev, a["mode"])
    if (d / "env.json").exists() and json.loads((d / "env.json").read_text()) != env:
        raise SystemExit(f"{arm}: resumed in another environment: {env} vs {(d / 'env.json').read_text()}")
    write_json(d / "env.json", env)
    done = {(r["item_set"], r["id"]) for r in read_jsonl(d / "scoring.jsonl")} if cfg.get("resume") else set()
    reuse_cells = P.lexsem_cells(P.LEXSEM_REUSE[arm]) if (cfg["reuse"] and arm in P.LEXSEM_REUSE) else None
    t0, n = time.time(), 0
    checked: dict = {}
    for item_set, ts, cells in item_sets(arm, its, cfg["reuse"]):
        for t in ts:
            if (item_set, t["id"]) in done or not cells:
                continue
            specs = recs_specs.get(t["id"])
            jobs = build_jobs(tok, arm, t, cells, specs, item_set)
            rec = {"id": t["id"], "game_id": t["game_id"], "item_set": item_set, "stratum": t["lexsem_stratum"],
                   "goal_object": t["goal_object"], "fewshot_object": t["goal_object"] in lexsem.FEWSHOT_OBJECTS,
                   "arm": arm, "revision": rev, "mode": a["mode"],
                   "scores": readouts(jobs, ls.run_item(model, jobs), pairs, tok, item_set)}
            if reuse_cells and item_set == "failure" and checked.get(t["lexsem_stratum"], 0) < P.REUSE_CHECK_ITEMS:
                checked[t["lexsem_stratum"]] = checked.get(t["lexsem_stratum"], 0) + 1
                extra = sorted(reuse_cells["failure"] & set(P.arm_cells(arm)["failure"]))
                ej = build_jobs(tok, arm, t, extra, specs, item_set)
                rec["reuse_check"] = readouts(ej, ls.run_item(model, ej), pairs, tok, item_set)
            append_jsonl(d / "scoring.jsonl", [rec])
            n += 1
            if n % 50 == 0:
                print(f"[{arm}] {n} items, {time.time() - t0:.0f}s", flush=True)
    if a["kind"] == "main" and a["mode"] == "plain" and not a.get("fewshot"):
        path = d / "recognition.jsonl"
        if not (cfg.get("resume") and path.exists()):
            path.write_text("".join(json.dumps(r) + "\n" for r in recognition(tok, model, recog_objects)))
    write_status(run_id, f"{arm} scored ({n} items this job, {time.time() - t0:.0f}s)", ok=True)
    print(f"[{arm}] done: {n} items in {time.time() - t0:.0f}s", flush=True)


def smoke(arms: list[str], tok, model, its: dict, recs_specs: dict, n: int, pairs: dict, out: Path, extra: dict) -> None:
    """Timing and validity only, scored exactly as the full run scores (every cell, reused ones included): seconds
    per sequence by format, the readout tokens' probability mass per arm, condition and format, the token checks
    (build_jobs), the failure text's tail after "Nothing". No effect (no difference between conditions) is
    computed or written."""
    import torch

    ls = _ls()
    rep: dict = {**extra, "n_items": n, "arms": {}}
    for arm in arms:
        times: dict = {}
        mass: dict = {}
        tails: list = []
        first_not_readout = 0
        extra_rows: list = []
        for item_set, ts, _ in item_sets(arm, its, False):
            cells = P.arm_cells(arm)[item_set]
            for t in ts[:n]:
                jobs = build_jobs(tok, arm, t, cells, recs_specs.get(t["id"]), item_set)
                rows = []
                for j in jobs:
                    torch.cuda.synchronize()
                    t1 = time.time()
                    rows.append(ls.score_one(model, j["ids"], j["n_prompt"], j["fmt"] in ("obs", "fail")))
                    if rows[-1][0].shape[-1] > 128256 and rows[-1][0].shape[-1] < 150000:  # Tulu's added rows
                        extra_rows.append(rows[-1][0][0, 128256:].exp().sum().item())
                    torch.cuda.synchronize()
                    times.setdefault(j["fmt"], []).append(time.time() - t1)
                r = readouts(jobs, rows, pairs, tok, item_set)
                for c, fm in r.items():
                    for f, x in fm.items():
                        if f == "fail":
                            tails.append(sum(x["tok_lp"][1:]))
                            continue
                        first = x["tok_lp"][0] if f == "obs" else x["lp_a"]
                        if f == "obs" and not x["first_is_readout"]:
                            first_not_readout += 1
                        mass.setdefault(f"{item_set}|{c}|{f}", []).append(math.exp(first) + math.exp(x["lp_b"]))
        rep["arms"][arm] = {
            "s_per_sequence": {f: round(sum(v) / len(v), 4) for f, v in times.items()},
            "n_by_format": {f: len(v) for f, v in times.items()},
            "mass_mean": {k: round(sum(v) / len(v), 4) for k, v in sorted(mass.items())},
            "mass_min": {k: round(min(v), 4) for k, v in sorted(mass.items())},
            "failure_text_after_first_token_nats": [round(min(tails), 3), round(max(tails), 3)] if tails else None,
            "success_obs_not_starting_with_readout": first_not_readout,
            "max_prob_on_ids_from_128256": max(extra_rows) if extra_rows else None,
            "cells_full_run": {s: len(v) for s, v in P.cells_to_score(arm, True).items()},
        }
    if extra.get("_recog"):
        t1 = time.time()
        rec = recognition(tok, model, extra["_pairs"][:2])
        rep["recognition_s_for_2_objects"] = round(time.time() - t1, 2)
    rep.pop("_pairs", None)
    rep.pop("_recog", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({a: rep["arms"][a]["s_per_sequence"] for a in arms}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=sorted(P.CHECKPOINTS))
    ap.add_argument("--arms", required=True, help="comma list of wf.posttrain.ARMS of this checkpoint")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-reuse", action="store_true", help="score every cell, also those lexsem scores")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="testing only: this many items per item set (recorded)")
    args = ap.parse_args()

    from wf.rq2 import load_items
    from wf.util import git_dirty, git_sha, is_mock, runs_root

    if is_mock():
        raise SystemExit("mock mode: refused")
    if git_dirty():
        raise SystemExit("the working tree has uncommitted changes to tracked files: refused")
    arms = args.arms.split(",")
    bad = [a for a in arms if P.ARMS.get(a, {}).get("ckpt") != args.ckpt]
    if bad:
        raise SystemExit(f"arms not of checkpoint {args.ckpt}: {bad}")
    ls = _ls()
    ck = P.CHECKPOINTS[args.ckpt]
    import os

    import torch

    if os.environ.get("WF_PT_DRYRUN") != "1":  # refuse a wrong GPU type before loading anything
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        for a in arms:
            want = "A40" if P.ARMS[a]["kind"] == "floor" else "L40S"
            if want not in gpu:
                raise SystemExit(f"{a} must run on an {want}, this is {gpu}")
    tok, model, rev = ls.load(ck["repo"])
    if rev != ck["revision"]:
        raise SystemExit(f"loaded revision {rev} != pinned {ck['revision']}")
    strict = strict_load_check(model, ck["repo"], rev)
    if not strict["ok"]:
        raise SystemExit(f"loading gate failed: {strict}")
    bad_tok = ls.token_checks(tok)
    if bad_tok:
        raise SystemExit("frozen token counts disagree with the loaded tokenizer: " + "; ".join(bad_tok))
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    its = P.items(load_items())
    if ls.sample_digest([t["id"] for t in its["failure"]]) != ls.EXPECTED_SAMPLE_DIGEST:
        raise SystemExit("the rebuilt sample differs from lexsem's frozen one")
    if args.limit:  # testing only: the first items of each stratum and of the successes
        its = {"failure": [t for t in its["failure"] if t["lexsem_stratum"] == "later"][: args.limit]
               + [t for t in its["failure"] if t["lexsem_stratum"] == "step0"][: args.limit],
               "success": its["success"][: args.limit]}
    built = lexsem.build(its["failure"], tuple(c for c in P.LEXSEM_CONDS), count_tokens=count)
    recs_specs = {r["id"]: r["specs"] for r in built}
    recog_objects = sorted({t["goal_object"] for t in its["failure"]})
    pairs_by_mode = {m: ls.readout_ids(tok, m) for m in {P.ARMS[a]["mode"] for a in arms}}
    cfg = {"ckpt": args.ckpt, "reuse": not args.no_reuse, "resume": args.resume, "strict_load": strict,
           "n_failures": len(its["failure"]), "n_successes": len(its["success"]), "limit": args.limit}
    if args.smoke:
        extra = {"ckpt": args.ckpt, "repo": ck["repo"], "revision": rev, "commit": git_sha(), "strict_load": strict,
                 "environment": ls.environment(ck["repo"], rev, "+".join(sorted(pairs_by_mode))), "_pairs": recog_objects}
        t0 = time.time()
        out = runs_root() / "posttrain_smoke" / f"{args.ckpt}.json"
        for k, a in enumerate(arms):  # one pairs table per mode; the recognition timing once
            smoke([a], tok, model, its, recs_specs, args.smoke, pairs_by_mode[P.ARMS[a]["mode"]],
                  out.with_name(f"{args.ckpt}__{a}.json"), {**extra, "_recog": k == 0})
        print(f"smoke wall {time.time() - t0:.0f}s", flush=True)
        return
    for a in arms:
        run_arm(a, tok, model, rev, its, recs_specs, dict(cfg), pairs_by_mode[P.ARMS[a]["mode"]], recog_objects)


if __name__ == "__main__":
    main()
