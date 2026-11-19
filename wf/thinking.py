"""Exploratory: does Qwen3's thinking mode remove the goal effect? (decided 2026-11-19; notes/deviations.md)

Qwen3-8B with thinking enabled, on a fixed random sample of RQ2's items, under the aligned, swapped,
neutral and object-free goals. For each prompt the model reasons and answers (vLLM, Qwen's recommended
thinking sampling, fixed seed per prompt). Three readings:

1. the written answer, read by the RQ1 parser: does the aligned goal raise predicted success?
2. L after reasoning: log P(o+) - log P(o-) at the answer position, given the model's own reasoning;
3. whether the reasoning mentions the goal ("goal", "task") or the action's object.

The thinking-off scores for the same items come from rq1_score and rq1x_score.
"""
from __future__ import annotations

import os
import random
import re
from typing import Any

from wf import constants as C
from wf import rq1, rq1x
from wf.analysis.report import write_run_report
from wf.analysis.stats import gen_says_success
from wf.rq2 import SOURCES, _mean_ci_p, complete_shards, load_items
from wf.runs import _finished, _init_run, _item_hash
from wf.scoring.parser import parse_prediction
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

ARMS = ("aligned", "swapped", "absent", "aligned_desc", "swapped_desc")
END_THINK = "</think>"
GOAL_WORDS = re.compile(r"\b(goal|task|objective)\b", re.I)


def sample_items(primary: list[dict], n: int, seed: str = "20261119|thinking") -> list[dict]:
    items = sorted(primary, key=lambda t: t["id"])
    return sorted(random.Random(seed).sample(items, min(n, len(items))), key=lambda t: t["id"])


def messages(t: dict, arm: str) -> list[dict[str, str]]:
    return rq1x.arm_messages(t, arm) if arm.endswith("_desc") else rq1.arm_messages(t, arm)


def split_output(text: str) -> tuple[str, str, bool]:
    """(reasoning, answer, finished): the answer is what follows the end of the reasoning block."""
    if END_THINK not in text:
        return text, "", False
    think, answer = text.split(END_THINK, 1)
    return think.replace("<think>", "").strip(), answer.strip(), True


def reasoning_prefix(text: str) -> str:
    """The generated text up to and including the end of reasoning and the blank line Qwen3 puts after it."""
    i = text.index(END_THINK) + len(END_THINK)
    return text[:i] + "\n\n"


def _seed(key: str) -> int:
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class _Engine:
    def __init__(self, model: str, max_len: int):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(model)
        # Prompt log-probabilities are computed over the whole vocabulary for every prompt token in a step, so the
        # step size is capped to keep that transient within the memory left free.
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.80,
                       max_num_batched_tokens=4096, seed=C.RNG_SEED)

    def prompt(self, msgs: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def generate(self, prompts: list[str], keys: list[str], max_tokens: int) -> list[tuple[str, int]]:
        from vllm import SamplingParams
        params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=max_tokens,
                                 seed=_seed(k), skip_special_tokens=False) for k in keys]
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]

    def logprob(self, prefixes: list[str], candidates: list[str]) -> list[float]:
        """log P(candidate | prefix), the candidate tokenized alone and appended, as the HF scorer does."""
        from vllm import SamplingParams
        reqs, spans = [], []
        for p, c in zip(prefixes, candidates):
            pid = self.tok(p, add_special_tokens=False).input_ids
            cid = self.tok(c, add_special_tokens=False).input_ids
            reqs.append({"prompt_token_ids": pid + cid})
            spans.append((len(pid), cid))
        outs = self.llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        res = []
        for o, (start, cid) in zip(outs, spans):
            lps = o.prompt_logprobs
            res.append(sum(lps[start + j][tid].logprob for j, tid in enumerate(cid)))
        return res


class _MockEngine:
    def prompt(self, msgs):
        return "\n".join(m["content"] for m in msgs)

    def generate(self, prompts, keys, max_tokens):
        out = []
        for p, k in zip(prompts, keys):
            say = "You pick up the mug 1." if _seed(k) % 2 else "Nothing happens."
            out.append((f"<think>\nThe task says something. Checking the room.\n</think>\n\n{say}", 20))
        return out

    def logprob(self, prefixes, candidates):
        return [-(_seed(p[-40:] + c) % 1000) / 100.0 for p, c in zip(prefixes, candidates)]


def run_thinking(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    sources = tuple(cfg.get("sources", SOURCES))
    bad = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ()))) or not verified(s)[0]]
    if bad:
        write_status(run_id, f"collection not finished or not verified: {bad}", ok=False)
        return 1
    d = _init_run(run_id, cfg, inputs=())
    items = sample_items(rq1x.rq1x_sets(load_items(sources))["primary"], int(cfg.get("n_items", 1000)))
    hashes = {t["id"]: _item_hash(t) for t in items}
    saved = read_jsonl(d / "thinking.jsonl") if cfg.get("resume") else []
    if any(hashes.get(r["transition_id"]) != r.get("item_hash") for r in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(r["transition_id"], r["condition"]) for r in saved}
    todo = [(t, a) for t in items for a in ARMS if (t["id"], a) not in done]
    mock = bool(os.environ.get("WF_MOCK_SCORE"))
    eng = _MockEngine() if mock else _Engine(cfg.get("model", C.WM_FEWSHOT_MODEL), int(cfg.get("max_model_len", 8192)))
    chunk, max_tokens = int(cfg.get("chunk", 250)), int(cfg.get("max_tokens", 4096))
    for s in range(0, len(todo), chunk):  # saved per chunk, so a cut-off job loses at most one chunk
        part = todo[s:s + chunk]
        prompts = [eng.prompt(messages(t, a)) for t, a in part]
        gens = eng.generate(prompts, [f"{t['id']}|{a}" for t, a in part], max_tokens)
        recs = []
        for (t, a), p, (text, n_tok) in zip(part, prompts, gens):
            think, answer, finished = split_output(text)
            cls = parse_prediction(answer, t.get("goal_object"), t["action"]) if finished else "unparseable"
            recs.append({"transition_id": t["id"], "game_id": t["game_id"], "condition": a, "item_hash": hashes[t["id"]],
                         "text": text, "answer": answer, "finished": finished, "n_tokens": n_tok,
                         "generated_class": cls, "says_success": gen_says_success(cls),
                         "mentions_goal_word": bool(GOAL_WORDS.search(think)),
                         "mentions_object": bool(re.search(rf"\b{re.escape(t.get('goal_object') or '~')}\b", think, re.I)),
                         "_prefix": p + reasoning_prefix(text) if finished else None,
                         "_o": (t["candidates"]["o_plus"], t["candidates"]["o_minus"])})
        scored = [r for r in recs if r["_prefix"]]
        lps = eng.logprob([r["_prefix"] for r in scored for _ in (0, 1)], [o for r in scored for o in r["_o"]])
        for k, r in enumerate(scored):
            r["logp_plus"], r["logp_minus"] = lps[2 * k], lps[2 * k + 1]
            r["L"] = r["logp_plus"] - r["logp_minus"]
        append_jsonl(d / "thinking.jsonl", [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
    write_status(run_id, f"{len(items)} items x {len(ARMS)} goals", ok=False, reported="run complete")
    return 0


# ---------------- analysis ----------------


def _paired(by: dict, items: list[dict], a: str, b: str, key: str) -> dict:
    diffs, games = [], []
    for t in items:
        x, y = by.get((t["id"], a), {}).get(key), by.get((t["id"], b), {}).get(key)
        if x is None or y is None or x != x or y != y:
            continue
        diffs.append(float(x) - float(y))
        games.append(t["game_id"])
    return _mean_ci_p(diffs, games)


def run_thinking_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    src = cfg["source"]
    from wf.util import runs_root
    status = (runs_root() / src / "STATUS.md")
    if not status.exists() or "REPORTED (run complete)" not in status.read_text().splitlines()[0]:
        write_status(run_id, f"{src} is not complete", ok=False)
        return 1
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    items = sample_items(rq1x.rq1x_sets(load_items(tuple(cfg.get("sources", SOURCES))))["primary"], int(cfg.get("n_items", 1000)))
    recs = read_jsonl(runs_root() / src / "thinking.jsonl")
    by = {(r["transition_id"], r["condition"]): {**r, "says": None if r["says_success"] is None else float(r["says_success"])}
          for r in recs}
    base = []
    for name in cfg.get("direct_sources", []):  # thinking-off scores for the same items
        folders, problems = complete_shards(name)
        if not problems:
            base += [s for p in folders for s in read_jsonl(p / "scoring.jsonl") if s["model"] == "qwen3-8b-fewshot"]
    direct = {(s["transition_id"], s["condition"]): {"L": s["L"]} for s in base}
    pairs = (("aligned", "swapped"), ("aligned", "absent"), ("swapped", "absent"), ("aligned_desc", "swapped_desc"))
    rep: dict[str, Any] = {
        "n_items": len(items), "n_records": len(recs),
        "finished_rate": sum(r["finished"] for r in recs) / len(recs) if recs else float("nan"),
        "median_tokens": sorted(r["n_tokens"] for r in recs)[len(recs) // 2] if recs else None,
        "says_success_rate": {a: (lambda v: sum(v) / len(v) if v else float("nan"))(
            [by[(t["id"], a)]["says"] for t in items if (t["id"], a) in by and by[(t["id"], a)]["says"] is not None]) for a in ARMS},
        "written_shift_pp": {f"{a}_vs_{b}": {k: (v * 100 if k == "estimate" else [x * 100 for x in v] if k == "ci" else v)
                                             for k, v in _paired(by, items, a, b, "says").items()} for a, b in pairs},
        "L_after_reasoning": {f"{a}_vs_{b}": _paired(by, items, a, b, "L") for a, b in pairs},
        "L_thinking_off_same_items": {f"{a}_vs_{b}": _paired(direct, items, a, b, "L") for a, b in pairs},
        "reasoning_mentions": {a: {
            "goal_word_when_says_success": _rate([by[k] for k in by if k[1] == a and by[k]["says"] == 1.0], "mentions_goal_word"),
            "goal_word_when_says_nothing": _rate([by[k] for k in by if k[1] == a and by[k]["says"] == 0.0], "mentions_goal_word"),
            "object_when_says_success": _rate([by[k] for k in by if k[1] == a and by[k]["says"] == 1.0], "mentions_object"),
        } for a in ARMS},
    }
    write_json(d / "metrics.json", rep)
    write_run_report(run_id, "Thinking mode (exploratory)", rep)
    ws, la = rep["written_shift_pp"]["aligned_vs_swapped"], rep["L_after_reasoning"]["aligned_vs_swapped"]
    write_status(run_id, f"written shift {ws['estimate']:.1f} pp {ws['ci']}; L after reasoning {la['estimate']:.2f} {la['ci']}",
                 ok=False, reported="thinking report")
    return 0


def _rate(rows: list[dict], key: str) -> dict:
    return {"rate": sum(bool(r[key]) for r in rows) / len(rows) if rows else float("nan"), "n": len(rows)}
