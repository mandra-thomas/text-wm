"""Scoring runs (notes/bayesenv_design.md, section 6). `python -m wf.runs bayesenv_<...>`.

The model is loaded as the study's scorer loads it (pinned revision, bf16, device_map auto, trust_remote_code). Every
sequence is scored alone and unpadded (batch size 1); the readout is a float32 log-softmax of the model's bf16 logits at
the position after the prompt. Two rendering modes:

- "plain" (Qwen3-8B): no chat template, so no empty think block. The messages are rendered by the Qwen audit's
  `plain_text` (system text, then each turn, separated by blank lines; an answer follows its turn after one space);
  an answer prefix follows after one space; the two candidate words are read with a leading space.
- "chat" (Llama-3.1-8B): the model's own chat template with the generation prompt; an answer prefix is appended as
  text; candidates are read as the first token of the reply (no leading space) or, after a prefix, with one space.

Every candidate must be a single token whose joint tokenization with the prompt equals the prompt's ids plus that
token; otherwise the run stops. Records are appended per unit (one item with all its prompts); a resumed run skips
finished units and refuses a changed commit, configuration, environment or prompt. Mock and dirty checkouts are
refused for real runs.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from wf.bayesenv import design as D
from wf.bayesenv import render as R
from wf.util import append_jsonl, git_dirty, is_mock, read_jsonl, write_json, write_status

ENV_FILE = "env.json"
SCORES = "scoring.jsonl"


# ----------------------------------------------------------------------------- tokenization


def render_text(prompt: R.Prompt) -> str:
    """Plain-text rendering (no chat template)."""
    from scripts.qwen_audit.followup_score import plain_text

    text = plain_text([{"role": r, "content": c} for r, c in prompt.messages])
    return text + (" " + prompt.prefix if prompt.prefix else "")


def encode(tok: Any, prompt: R.Prompt, mode: str, template_kwargs: dict | None = None) -> tuple[list[int], tuple[int, int]]:
    """(prompt ids, candidate token ids). Refuses a candidate that is not one token, a joint tokenization that does not
    extend the prompt by exactly that token, and special tokens inside a plain-text prompt."""
    if mode == "plain":
        text = render_text(prompt)
        if any(s in text for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")):
            raise SystemExit("plain-text prompt contains chat-template markup")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        special = set(getattr(tok, "all_special_ids", []) or [])
        if special & set(ids):
            raise SystemExit("plain-text prompt contains special tokens")
        cands = []
        for w in prompt.candidates:
            joint = tok(text + " " + w, add_special_tokens=False)["input_ids"]
            if joint[: len(ids)] != ids or len(joint) != len(ids) + 1:
                raise SystemExit(f"candidate {w!r} does not extend the plain prompt by one token")
            cands.append(joint[-1])
        return ids, (cands[0], cands[1])
    if mode != "chat":
        raise ValueError(mode)
    msgs = [{"role": r, "content": c} for r, c in prompt.messages]
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **(template_kwargs or {}))
    if hasattr(out, "keys"):
        out = out["input_ids"]
    ids = list(out[0] if out and isinstance(out[0], list) else out)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))
    sp = ""
    if prompt.prefix:
        ids = ids + tok(prompt.prefix, add_special_tokens=False)["input_ids"]
        text = text + prompt.prefix
        sp = " "
    base = tok(text, add_special_tokens=False)["input_ids"]
    if base != ids:
        raise SystemExit("the chat template's ids differ from the tokenization of its rendered text")
    cands = []
    for w in prompt.candidates:
        c = tok(sp + w, add_special_tokens=False)["input_ids"]
        if len(c) != 1:
            raise SystemExit(f"candidate {sp + w!r} is not one token: {c}")
        joint = tok(text + sp + w, add_special_tokens=False)["input_ids"]
        if joint != ids + c:
            raise SystemExit(f"candidate {w!r} does not extend the rendered prompt by one token")
        cands.append(c[0])
    return ids, (cands[0], cands[1])


# ----------------------------------------------------------------------------- model


def load(model_id: str):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wf.constants import revision_for

    rev = revision_for(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    kw = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, device_map="auto", trust_remote_code=True, **kw)
    model.eval()
    return tok, model, rev


def forward_last(model: Any, ids: list[int], cands: tuple[int, int]) -> dict:
    """One unpadded sequence; the readout at the next position."""
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    top = int(torch.argmax(lp).item())
    a, b = cands
    return {"lp_a": lp[a].item(), "lp_b": lp[b].item(), "z_a": logits[a].float().item(),
            "z_b": logits[b].float().item(), "top": top, "lp_top": lp[top].item(), "n_tok": len(ids)}


def environment(model_id: str, rev: str, mode: str, model: Any = None) -> dict:
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    attn = getattr(getattr(model, "config", None), "_attn_implementation", None) if model is not None else None
    return {"gpu": gpu, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "model": model_id, "revision": rev, "dtype": "bfloat16",
            "attn": attn, "batch_size": 1, "padding": "none", "mode": mode}


# ----------------------------------------------------------------------------- the run


def score_units(units: list[D.Unit], out: Path, scorer: Callable[[D.Unit], dict], resume: bool,
                log_every: int = 25) -> int:
    """Score each unit not yet saved and append it at once. `scorer(unit)` returns {"type|framing|readout": readout}."""
    done: dict[str, str] = {}
    if resume and out.exists():
        for r in read_jsonl(out):
            done[r["unit"]] = r["unit_hash"]
    n_new = 0
    t0 = time.time()
    for i, u in enumerate(units):
        h = D.unit_hash(u)
        if u.uid in done:
            if done[u.uid] != h:
                raise SystemExit(f"{u.uid}: saved scores were made for a different prompt (hash {done[u.uid]} != {h})")
            continue
        rec = {"unit": u.uid, "unit_hash": h, "space": u.space, "gen": u.gen, "hist": u.hist, "k": u.k,
               "scores": scorer(u)}
        append_jsonl(out, [rec])
        n_new += 1
        if n_new % log_every == 0:
            print(f"[{i + 1}/{len(units)}] {time.time() - t0:.0f}s", flush=True)
    return n_new


def model_scorer(tok: Any, model: Any, mode: str, template_kwargs: dict | None = None) -> Callable[[D.Unit], dict]:
    def scorer(u: D.Unit) -> dict:
        res = {}
        for t, fr, r in u.sequences():
            ids, cands = encode(tok, D.prompt_for(u, t, fr, r), mode, template_kwargs)
            v = forward_last(model, ids, cands)
            if not all(math.isfinite(v[x]) for x in ("lp_a", "lp_b")):
                raise SystemExit(f"{u.uid} {t} {fr} {r}: non-finite readout")
            res[f"{t}|{fr}|{r}"] = {k: (round(x, 6) if isinstance(x, float) else x) for k, x in v.items()}
        return res
    return scorer


def units_for(cfg: dict) -> list[D.Unit]:
    which = cfg["units"]
    if which == "smoke":
        return D.smoke_units()
    d = D.Design(**cfg.get("design", {}))
    if which == "full":
        return D.full_units(d)
    if which == "xgpu":
        return D.full_units(d, k_max=D.XGPU_K)
    raise ValueError(which)


def run_bayesenv_score(cfg: dict, scorer_factory: Callable | None = None) -> int:
    """Smoke, full and cross-GPU scoring runs. Config: model, mode, units (smoke|full|xgpu), design, gpu_type (a
    substring of the GPU name the run must use), template_kwargs; shards from WF_SHARD and WF_N_SHARDS."""
    from wf.runs import _init_run

    run_id = cfg["run_id"]
    if is_mock() and scorer_factory is None:
        raise SystemExit(f"{run_id}: mock scores are never written for this study")
    if not is_mock() and git_dirty():
        raise SystemExit(f"{run_id}: the checkout has uncommitted changes; scores must come from a commit")
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    shard_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": shard_id, "shard": i, "n_shards": n}
    d = _init_run(shard_id, cfg)
    units = D.shard(units_for(cfg), i, n)
    if scorer_factory is None:
        tok, model, rev = load(cfg["model"])
        env = environment(cfg["model"], rev, cfg["mode"], model)
        want = cfg.get("gpu_type")
        if want and not any(x in env["gpu"] for x in want.split("|")):
            raise SystemExit(f"{shard_id}: runs on {env['gpu']}, but this run is pinned to {want}")
        for lib, version in (cfg.get("stack") or {}).items():
            if env.get(lib, "").split("+")[0] != version:
                raise SystemExit(f"{shard_id}: {lib} {env.get(lib)} is not the pinned {version}")
        scorer = model_scorer(tok, model, cfg["mode"], cfg.get("template_kwargs"))
    else:
        env, scorer = scorer_factory(cfg)
    saved_env = d / ENV_FILE
    if cfg.get("resume") and saved_env.exists():
        old = json.loads(saved_env.read_text())
        if old != env:
            raise SystemExit(f"{shard_id}: environment changed since the saved work: {old} -> {env}")
    write_json(saved_env, env)
    t0 = time.time()
    n_new = score_units(units, d / SCORES, scorer, bool(cfg.get("resume")))
    n_seq = sum(len(u.sequences()) for u in units)
    write_json(d / "timing.json", {"units": len(units), "new_units": n_new, "sequences": n_seq,
                                   "seconds": round(time.time() - t0, 1)})
    write_status(shard_id, f"shard {i} of {n}: {len(units)} units, {n_seq} sequences ({n_new} scored by this job)",
                 ok=False, reported="shard complete")
    return 0


# ----------------------------------------------------------------------------- smoke-only checks (validation)


def classify_reply(space: str, framing: str, readout: str, text: str) -> str:
    """Class of a sampled continuation: 'a' (begins with the first candidate word and is the success form),
    'a_other' (begins with the first candidate word but is not the success form), 'b' (begins with the second
    candidate word and is the failure form), 'b_other', or 'other' (begins with neither)."""
    s = text.strip()
    low = s.lower()
    if space == "alf":
        if low.startswith("a "):
            return "a" if low.startswith("a keychain") else "a_other"
        if low.startswith("nothing"):
            return "b" if low.startswith("nothing.") or low == "nothing" else "b_other"
        return "other"
    if framing == "wm":
        if s.startswith("You"):
            return "a" if s.startswith("You take the key from box 3") else "a_other"
        if s.startswith("Nothing"):
            return "b" if s.startswith("Nothing happens") else "b_other"
        return "other"
    if low.startswith("holds"):
        ok = (low.startswith("holds the key") or low.startswith("holds a key")) and \
            ("not locked" in low or "unlocked" in low)
        return "a" if ok else "a_other"
    if low.startswith("does"):
        return "b" if low.startswith("does not hold") or low.startswith("doesn't hold") else "b_other"
    return "other"


def sample_check(tok: Any, model: Any, mode: str, units: list[D.Unit], per_readout: int = 40, n_samples: int = 10,
                 max_new: int = 16, seed: int = 20261125, template_kwargs: dict | None = None) -> dict:
    """The reply-form and 'other'-rate check (section 5): for each question-free readout, `per_readout` prompts,
    half drawn from prompts whose exact success probability is at least 0.25 and half from the rest, each with
    `n_samples` continuations sampled at temperature 1, classified by `classify_reply`. Validation only: sampled in
    batches, never used as a score."""
    import random

    import torch

    rng = random.Random(seed)
    groups: dict = {}
    for u in units:
        for t, fr, r in u.sequences():
            if r in ("first", "narr"):
                p = D.exact_for(u.space, t, u.gen, u.hist)
                hi = p is not None and p >= 0.25
                groups.setdefault((u.space, fr, r), {True: [], False: []})[hi].append((u, t, fr, r))
    chosen = []
    for key, g in sorted(groups.items()):
        for hi in (True, False):
            lst = sorted(g[hi], key=lambda x: (x[0].uid, x[1]))
            rng.shuffle(lst)
            chosen += lst[: per_readout // 2]
    by: dict = {}
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    for u, t, fr, r in chosen:
        p = D.prompt_for(u, t, fr, r)
        ids, _ = encode(tok, p, mode, template_kwargs)
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0,
                               top_p=1.0, top_k=0, max_new_tokens=max_new, num_return_sequences=n_samples,
                               pad_token_id=tok.eos_token_id)
        key = f"{u.space}|{fr}|{r}"
        for row in g[:, x.shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True)
            c = classify_reply(u.space, fr, r, text)
            by.setdefault(key, {}).setdefault(c, 0)
            by[key][c] += 1
            ex = by.setdefault(key + "|examples", {})
            if len(ex.setdefault(c, [])) < 5:
                ex[c].append(text)
    return by


DESCRIPTION_QUESTIONS = (
    f"Question: does \"{R.DESCRIPTION}\" describe a key? Answer Yes or No.\nAnswer:",
    f"Question: is \"{R.DESCRIPTION}\" a way of referring to a key? Answer Yes or No.\nAnswer:",
    f"Question: a house tag reads \"{R.DESCRIPTION}\". Does the tag refer to a key? Answer Yes or No.\nAnswer:",
)


def description_check(tok: Any, model: Any, mode: str, template_kwargs: dict | None = None) -> dict:
    """Does the model take the tag's description to mean a key? Three yes/no phrasings; all must favour Yes."""
    rows = []
    for q in DESCRIPTION_QUESTIONS:
        p = R.Prompt(messages=(("user", q),), prefix="", candidates=("Yes", "No"))
        ids, c = encode(tok, p, mode, template_kwargs)
        v = forward_last(model, ids, c)
        rows.append(v["lp_a"] - v["lp_b"])
    return {"yes_minus_no": rows, "passed": all(x > 0 for x in rows)}


def run_bayesenv_smoke_checks(cfg: dict) -> int:
    """The smoke's validation checks that are not scores: sampled continuations and the description check."""
    from wf.runs import _init_run

    run_id = cfg["run_id"]
    if is_mock():
        raise SystemExit(f"{run_id}: mock runs are refused")
    if git_dirty():
        raise SystemExit(f"{run_id}: the checkout has uncommitted changes")
    d = _init_run(run_id, cfg)
    tok, model, rev = load(cfg["model"])
    env = environment(cfg["model"], rev, cfg["mode"], model)
    want = cfg.get("gpu_type")
    if want and not any(x in env["gpu"] for x in want.split("|")):
        raise SystemExit(f"{run_id}: runs on {env['gpu']}, but this run is pinned to {want}")
    for lib, version in (cfg.get("stack") or {}).items():
        if env.get(lib, "").split("+")[0] != version:
            raise SystemExit(f"{run_id}: {lib} {env.get(lib)} is not the pinned {version}")
    write_json(d / ENV_FILE, env)
    units = D.smoke_units()
    out = {"description": description_check(tok, model, cfg["mode"], cfg.get("template_kwargs")),
           "samples": sample_check(tok, model, cfg["mode"], units, template_kwargs=cfg.get("template_kwargs"))}
    write_json(d / "checks.json", out)
    write_status(run_id, "smoke validation checks written to checks.json", ok=False, reported="checks")
    return 0
