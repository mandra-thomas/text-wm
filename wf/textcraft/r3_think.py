"""TextCraft round 3, the thinking arm (exploratory; notes/textcraft_r3_design.md section 8.2).

Qwen3.5-9B at its pinned revision with thinking on: the model writes its own reasoning after round 2's prompt, then
the two replies (or "Yes" and "No") are teacher-forced after the reasoning, and L after reasoning is their log-odds.
Outside every test family; never gated or pooled with the instruction runs.

Procedure (fixed in the design note):
- records and arms: the parent failures under ext_aligned and ext_parent; the ingredient main failures under aligned
  and swapped; yn_neutral on the determined successes and a seeded 600 of the main failures (4,530 generations);
- generation with HF `generate`: Qwen's recommended thinking sampling (temperature 0.6, top-p 0.95, top-k 20), at most
  MAX_REASONING new tokens, stopping at the end of the reasoning block; fixed batches of BATCH prompts in record-and-arm
  order, each with a sampling seed fixed from RNG_SEED and its first record and arm; every batch saved as it finishes;
- budget forcing: reasoning not closed within MAX_REASONING tokens is closed with FORCE_CLOSE and flagged `forced`;
- the readout: after the rendered prompt, the reasoning, the closing marker and a blank line; the boundary between that
  prefix and each candidate is checked (no merged tokens) and recorded.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Any

from wf import constants as C
from wf import families as FAM
from wf.runs import _init_run
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r3 as R3
from wf.textcraft.r2_score import load_records, shard_records, verified_files
from wf.util import append_jsonl, read_jsonl, require_filed, write_status

MAX_REASONING = 2048
BATCH = 8
SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
END_THINK = "</think>"
FORCE_CLOSE = "\n</think>"
AFTER_CLOSE = "\n\n"  # the blank line Qwen writes after its reasoning; the readout is taken after it
N_FAILURES_YN = 600
GOAL_WORDS = re.compile(r"\b(goal|task|objective)\b", re.I)
P_ARMS = ("ext_aligned", "ext_parent")
LIFT_ARMS = ("aligned", "swapped")
YN_ARM = "yn_neutral"


def think_jobs(records: list[dict]) -> list[tuple[dict, str]]:
    """Every (record, arm) the thinking arm generates for, in record-and-arm order."""
    parent = [r for r in records if r["kind"] == "parent" and r["population"] == "main"]
    main_f = [r for r in records if r["kind"] == "main" and r["population"] == "main"]
    ingredient = [r for r in main_f if r["role"] == "ingredient"]
    successes = [r for r in records if r["kind"] == "main" and r["population"] == "success"]
    ids = sorted(r["id"] for r in main_f)
    yn_f = set(random.Random(R3.seed("think|failures")).sample(ids, min(N_FAILURES_YN, len(ids))))
    jobs = [(r, a) for r in parent for a in P_ARMS] + [(r, a) for r in ingredient for a in LIFT_ARMS]
    jobs += [(r, YN_ARM) for r in successes] + [(r, YN_ARM) for r in main_f if r["id"] in yn_f]
    return sorted(jobs, key=lambda x: (x[0]["id"], x[1]))


def batch_seed(key: str) -> int:
    return int(hashlib.sha256(R3.seed(f"think|batch|{key}").encode()).hexdigest()[:8], 16)


def close_reasoning(text: str) -> tuple[str, bool]:
    """(the reasoning up to and including its closing marker, forced). Text after the marker (or an end-of-turn token
    before it) is dropped; reasoning that never closed is closed with FORCE_CLOSE."""
    if END_THINK in text:
        return text[: text.index(END_THINK) + len(END_THINK)], False
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text[: text.index(stop)]
    return text + FORCE_CLOSE, True


def mentions(text: str, rec: dict) -> dict[str, bool]:
    """Whether the reasoning names the goal ("goal", "task", "objective") and the action's object."""
    item = rec.get("object")
    return {"goal_word": bool(GOAL_WORDS.search(text)),
            "crafted_item": bool(item and re.search(rf"\b{re.escape(str(item))}\b", text, re.I))}


class _Engine:
    """The real model: Qwen3.5-9B loaded as the instruction runs load it, generating with thinking on."""

    def __init__(self):
        self.scorer = FAM.open_scorer(R3.THINK_SPEC, R3.THINK_TAG)
        self.tok = self.scorer._tok
        self.model = self.scorer._model
        self.revision = self.scorer.revision

    def prompt_text(self, msgs) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def generate(self, texts: list[str], seed: int) -> list[tuple[str, int]]:
        import torch

        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_REASONING, stop_strings=[END_THINK], tokenizer=self.tok,
                                      **SAMPLING)
        new = out[:, enc["input_ids"].shape[1]:]
        res = []
        for row in new:
            ids = [t for t in row.tolist() if t != self.tok.pad_token_id]
            res.append((self.tok.decode(ids, skip_special_tokens=False), len(ids)))
        return res

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def logprobs(self, context: str, candidates: tuple[str, str]) -> dict[str, Any]:
        """Each candidate's log-probability (whole, and its first token) teacher-forced after `context`, and whether
        the context and the candidate tokenize together as the context's ids followed by the candidate's."""
        import torch

        ctx = self.ids(context)
        out: dict[str, Any] = {"boundary_ok": True}
        for name, cand in zip(("plus", "minus"), candidates):
            c = self.ids(cand)
            out["boundary_ok"] &= self.ids(context + cand) == ctx + c
            x = torch.tensor([ctx + c], device=self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids=x).logits[0].float()
            lps = [torch.log_softmax(logits[len(ctx) + i - 1], dim=-1)[t].item() for i, t in enumerate(c)]
            out[f"logp_{name}"], out[f"logp_first_{name}"], out[f"n_tokens_{name}"] = sum(lps), lps[0], len(c)
        return out


class _MockEngine:
    revision = "mock"

    def prompt_text(self, msgs) -> str:
        return repr(msgs)

    def generate(self, texts: list[str], seed: int) -> list[tuple[str, int]]:
        # every fourth reasoning in a batch never closes, so budget forcing runs in the dry run
        return [("reasoning about the goal" + (END_THINK if k % 4 else ""), 5) for k, _t in enumerate(texts)]

    def logprobs(self, context: str, candidates: tuple[str, str]) -> dict[str, Any]:
        h = int(hashlib.sha256(context.encode()).hexdigest()[:6], 16) / 0xFFFFFF
        return {"boundary_ok": True, "logp_plus": -2 - 8 * h, "logp_minus": -2 - 8 * (1 - h), "logp_first_plus": -1 - h,
                "logp_first_minus": -1 - (1 - h), "n_tokens_plus": 5, "n_tokens_minus": 7}


def run_textcraft_r3_think(cfg: dict) -> int:
    """`textcraft_r3_think_qwen3.5-9b` (and `textcraft_r3_think_smoke_qwen3.5-9b`, cfg max_jobs): generate, then score."""
    run_id = cfg["run_id"]
    require_filed(run_id)
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"records not verified: {files}", ok=False)
        return 1
    rpath, spath, mpath = files
    mock = os.environ.get("WF_MOCK_SCORE", "0") == "1"
    spec = R3.THINK_SPEC
    env = FAM.require_environment(spec)
    env["resources"] = FAM.require_resources(spec)
    from wf.textcraft.r3_score import gpu_check

    env["gpu"] = gpu_check(R3.THINK_TAG)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n, "model_spec": spec.record(), "environment": env,
           "procedure": {"max_reasoning": MAX_REASONING, "batch": BATCH, "sampling": SAMPLING, "end": END_THINK,
                         "force_close": FORCE_CLOSE, "after_close": AFTER_CLOSE}}
    d = _init_run(run_id, cfg, inputs=(rpath, spath, mpath))
    records = load_records(rpath)
    shots = P.fewshot_examples(spath)
    jobs = think_jobs(records)
    keep = {r["id"] for r in shard_records(records, i, n)}
    jobs = [(r, a) for r, a in jobs if r["id"] in keep]
    if cfg.get("max_jobs"):
        jobs = jobs[: int(cfg["max_jobs"])]
    hashes = {r["id"]: R.item_hash(r, shots) for r, _a in jobs}
    gens = {(g["transition_id"], g["condition"]): g for g in read_jsonl(d / "generations.jsonl")} if cfg.get("resume") else {}
    if any(hashes.get(k[0]) != g.get("item_hash") for k, g in gens.items()):
        raise SystemExit(f"{run_id}: saved generations belong to different records or prompts; rerun without --resume.")
    eng = _MockEngine() if mock else _Engine()
    todo = [(r, a) for r, a in jobs if (r["id"], a) not in gens]
    for b in range(0, len(todo), BATCH):  # saved per batch: a cut-off job loses at most one batch
        part = todo[b:b + BATCH]
        key = f"{part[0][0]['id']}|{part[0][1]}"
        s = batch_seed(key)
        texts = [eng.prompt_text(R3.model_messages(spec, r, a, shots)) for r, a in part]
        out = eng.generate(texts, s)
        rows = []
        for (r, a), t, (gen, n_new) in zip(part, texts, out):
            trace, forced = close_reasoning(gen)
            rows.append({"transition_id": r["id"], "condition": a, "item_hash": hashes[r["id"]], "batch_key": key,
                         "seed": s, "prompt_sha256": hashlib.sha256(t.encode()).hexdigest(), "generated": gen,
                         "reasoning": trace, "forced": forced, "n_new_tokens": n_new, "model_revision": eng.revision})
            gens[(r["id"], a)] = rows[-1]
        append_jsonl(d / "generations.jsonl", rows)
    scored = {(s["transition_id"], s["condition"]) for s in read_jsonl(d / "scoring.jsonl")} if cfg.get("resume") else set()
    for r, a in jobs:
        if (r["id"], a) in scored:
            continue
        g = gens[(r["id"], a)]
        context = eng.prompt_text(R3.model_messages(spec, r, a, shots)) + g["reasoning"] + AFTER_CLOSE
        lp = eng.logprobs(context, R.candidates(r, a))
        L = lp["logp_plus"] - lp["logp_minus"]
        append_jsonl(d / "scoring.jsonl", [{"transition_id": r["id"], "model": R3.THINK_TAG, "condition": a, "L": L,
                                            **lp, "forced": g["forced"], "n_new_tokens": g["n_new_tokens"],
                                            **{f"mentions_{k}": v for k, v in mentions(g["reasoning"], r).items()},
                                            "pair_id": r["pair_id"], "kind": r["kind"], "population": r["population"],
                                            "role": r["role"], "item_hash": hashes[r["id"]],
                                            "model_revision": eng.revision}])
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} generations and scores ({R3.THINK_TAG})", ok=False,
                 reported="shard complete")
    return 0
