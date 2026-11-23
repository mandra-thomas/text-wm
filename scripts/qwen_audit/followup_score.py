"""Exploratory follow-up to the pipeline audit: the study's own scorer with and without Qwen3's empty think block.

The scorer is wf.scoring.logodds.LogoddsScorer and the prompts are the study's (build_messages, as rq2_score; the
decision simulation's messages_for for `dsim`). The only change is the template setting: "block" renders the chat
template with enable_thinking=False (the study's format: the scored answer follows `<think>\\n\\n</think>\\n\\n`),
"noblock" with enable_thinking=True and no reasoning (the answer follows a bare `<|im_start|>assistant\\n`, as the
few-shot answers do). The format is set through the scorer's template_kwargs, and every score's condition label
carries the format, so the scorer's result cache (keyed by messages, not by template settings) can never return one
format's score for the other.

A third format, "plain" (added after the first follow-up run), uses no chat template at all: the same messages as
plain text (`plain_text`), the candidate after one space, read from one forward pass of the scorer's loaded model.
It exists because the "noblock" render leaves almost no probability on either answer's first token (Qwen3 expects
`<think>` there), so "noblock" compares two continuations the model treats as near-impossible; plain text keeps
most of the first-token mass on the two answers. Plain scores record that mass (`first_mass`).

Tasks:
  rq2    items from sample_items.py; --stored-shards keeps the items of those stored rq2_score shards (so the job can
         run on the GPU type those shards ran on); --formats (default noblock); --check-block N also rescores the
         first N items with the block, to compare with the stored values.
  dsim   the decision simulation's step-0 states (rebuilt from the collection at --src), every candidate, the four
         default arms, yes/no score (score_one_token_pair, or " Yes"/" No" in plain text), --formats as given
         (the first run used block,noblock).

Output is appended per item (rq2) or per state (dsim); a rerun skips what is done.

    python scripts/qwen_audit/followup_score.py rq2 --items runs/qwen_audit/items_all.jsonl --model Qwen/Qwen3-8B \
        --stored-shards 0 --check-block 20 --out runs/qwen_audit/fu_8b_shard0.jsonl
    python scripts/qwen_audit/followup_score.py rq2 --items runs/qwen_audit/items_sizes.jsonl --model Qwen/Qwen3-4B \
        --formats block,noblock --out runs/qwen_audit/fu_sizes_4b.jsonl
    python scripts/qwen_audit/followup_score.py dsim --src /storage/ice1/2/1/gmat3/text-wm-fix11/runs \
        --out runs/qwen_audit/fu_dsim_step0.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

FORMATS = {"block": {}, "noblock": {"enable_thinking": True}}
EMPTY_THINK = "<think>\n\n</think>\n\n"
DSIM_DIGEST = "5b6fa14d3ff20383"  # the stored decision-simulation sample (fix30, decisionsim_score_qwen3-8b)


def set_format(sc, fmt: str) -> None:
    sc.template_kwargs = dict(FORMATS[fmt])


def check_render(sc, msgs) -> None:
    """Each format must render as intended: the block right before the answer, or a bare assistant header."""
    for fmt in FORMATS:
        set_format(sc, fmt)
        text = sc._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **sc.template_settings())
        ok = text.endswith("<|im_start|>assistant\n" + EMPTY_THINK) if fmt == "block" else \
            (text.endswith("<|im_start|>assistant\n") and EMPTY_THINK not in text)
        if not ok:
            raise SystemExit(f"format {fmt} renders wrongly: ...{text[-80:]!r}")
    print("render check passed for both formats", flush=True)


def slim(r) -> dict:
    d = r.to_dict()
    return {k: d[k] for k in ("L", "logp_plus", "logp_minus", "L_first", "n_tokens_plus", "n_tokens_minus",
                              "first_token_plus", "first_token_minus", "first_mass")}


def plain_text(msgs: list[dict]) -> str:
    """No chat template (section 4's plain-text format): the system text, each example as its user text, one space
    and its answer, then the query's user text, separated by blank lines. The candidate follows after one space."""
    parts, i = [], 0
    if msgs and msgs[0]["role"] == "system":
        parts.append(msgs[0]["content"])
        i = 1
    while i < len(msgs):
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            parts.append(f"{msgs[i]['content']} {msgs[i + 1]['content']}")
            i += 2
        else:
            parts.append(msgs[i]["content"])
            i += 1
    return "\n\n".join(parts)


def plain_pair(sc, msgs: list[dict], plus: str, minus: str) -> dict:
    """Teacher-forced log-probs of " plus" and " minus" after the plain-text prompt (one sequence per forward pass,
    candidate read from the joint tokenization), and the first-token mass the two answers' first tokens hold."""
    import torch

    prompt = plain_text(msgs)
    ids = lambda s: sc._tok(s, add_special_tokens=False)["input_ids"]
    p = ids(prompt)
    out, firsts = {}, {}
    dev = next(sc._model.parameters()).device
    for name, cand in (("plus", plus), ("minus", minus)):
        f = ids(prompt + " " + cand)
        if f[: len(p)] != p:
            f = p + ids(" " + cand)
        c = f[len(p):]
        with torch.no_grad():
            logits = sc._model(input_ids=torch.tensor([f], device=dev)).logits[0]
        lp = torch.log_softmax(logits[len(p) - 1: len(f) - 1].float(), dim=-1)
        tok = [lp[k, c[k]].item() for k in range(len(c))]
        out[f"logp_{name}"], out[f"n_tokens_{name}"], firsts[name] = sum(tok), len(c), (lp[0], c[0])
    out["L"] = out["logp_plus"] - out["logp_minus"]
    vec = firsts["plus"][0]
    out["L_first"] = vec[firsts["plus"][1]].item() - vec[firsts["minus"][1]].item()
    out["first_mass"] = math.exp(vec[firsts["plus"][1]].item()) + math.exp(vec[firsts["minus"][1]].item())
    return out


def run_rq2(args, sc) -> None:
    from wf.scoring.logodds import build_messages

    shards = {int(x) for x in args.stored_shards.split(",")} if args.stored_shards else None
    items = [json.loads(x) for x in Path(args.items).open()]
    items = [it for it in items if shards is None or it["shard"] in shards][: args.limit]
    formats = args.formats.split(",")
    out = Path(args.out)
    done = {json.loads(x)["id"] for x in out.open()} if out.exists() else set()
    check_render(sc, build_messages(items[0]["transition"], "aligned"))
    t0 = time.time()
    for k, it in enumerate(items):
        if it["id"] in done:
            continue
        t = it["transition"]
        fmts = formats + (["block"] if k < args.check_block and "block" not in formats else [])
        rec = {"id": it["id"], "shard": it["shard"], "model": sc.model_name, "scores": {}}
        for fmt in fmts:
            for cond in ("aligned", "swapped"):
                if fmt == "plain":
                    rec["scores"][f"{fmt}/{cond}"] = plain_pair(sc, build_messages(t, cond), t["candidates"]["o_plus"],
                                                                t["candidates"]["o_minus"])
                    continue
                set_format(sc, fmt)
                r = sc.score_pair(t["id"], f"{fmt}/{cond}", build_messages(t, cond), t["candidates"]["o_plus"],
                                  t["candidates"]["o_minus"])
                rec["scores"][f"{fmt}/{cond}"] = slim(r)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 100 == 0:
            print(f"[{k}/{len(items)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(items)} items in {time.time() - t0:.0f}s", flush=True)


def run_dsim(args, sc) -> None:
    from wf import decisionsim as D

    src = Path(args.src)
    transitions = [json.loads(x) for k in range(3) for x in (src / f"rq12_transitions_s{k}" / "transitions.jsonl").open()]
    states = D.build_states(transitions)
    if D.sample_digest(states) != DSIM_DIGEST:
        raise SystemExit("the rebuilt decision-simulation states differ from the stored run's")
    step0 = [s for s in states if s["state_idx"] == 0]
    build = D.messages_for("qwen3-8b")
    out = Path(args.out)
    done = {(r["game_id"], r["state_idx"]) for r in map(json.loads, out.open())} if out.exists() else set()
    check_render(sc, build(step0[0]["candidates"][0], D.ALIGNED))
    t0 = time.time()
    for k, s in enumerate(step0):
        if (s["game_id"], s["state_idx"]) in done:
            continue
        rec = {"game_id": s["game_id"], "state_idx": s["state_idx"], "scores": {}}
        for t in s["candidates"]:
            for arm in D.DEFAULT_ARMS:
                item = D.prompt_item(t, s, arm)
                if item is None:
                    continue
                msgs = build(item, arm)
                for fmt in args.formats.split(","):
                    if fmt == "plain":
                        r = plain_pair(sc, msgs, "Yes", "No")
                        rec["scores"].setdefault(t["id"], {})[f"{fmt}/{arm}"] = r["L"]
                        rec.setdefault("first_mass", {}).setdefault(t["id"], {})[arm] = r["first_mass"]
                        continue
                    set_format(sc, fmt)
                    r = sc.score_one_token_pair(t["id"], f"{fmt}/{arm}", msgs, "Yes", "No")
                    rec["scores"].setdefault(t["id"], {})[f"{fmt}/{arm}"] = r.L
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 10 == 0:
            print(f"[{k}/{len(step0)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(step0)} states in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("rq2", "dsim"))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--items")
    ap.add_argument("--stored-shards", default=None)
    ap.add_argument("--formats", default="noblock")
    ap.add_argument("--check-block", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--src", default="/storage/ice1/2/1/gmat3/text-wm-fix11/runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from wf.scoring.logodds import LogoddsScorer

    sc = LogoddsScorer(args.model, model_tag=args.model.split("/")[-1].lower())
    print(f"model={args.model} rev={sc.revision}", flush=True)
    (run_rq2 if args.task == "rq2" else run_dsim)(args, sc)
    sc.close()


if __name__ == "__main__":
    main()
