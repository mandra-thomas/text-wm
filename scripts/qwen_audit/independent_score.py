"""Exploratory pipeline audit: an independent teacher-forced scorer, written without wf.scoring's code path.

What it shares with the study's scorer (wf/scoring/logodds.py) on purpose: the model snapshot (wf.constants.REVISIONS)
and the frozen few-shot examples (wf.scoring.prompts.WM_FEWSHOT_SET0, imported as data). Everything else is
written here from the item text: the task-line swap, the user turns, the chat rendering, the tokenization of
prompt and candidate, the forward pass and the indexing of candidate log-probabilities.

How a score is read. The prompt is rendered to TEXT (the model's own apply_chat_template, tokenize=False, or one of
the format variants below), the candidate text is appended to that string, and the whole string is tokenized once.
The prompt's own tokenization must be an exact prefix of the joint one (checked and recorded per score); the
candidate's tokens are what follows. One unbatched, unpadded forward pass with an explicit all-ones attention mask;
log_softmax in float32 at positions p-1 .. p+n-2 (p prompt tokens, n candidate tokens), indexed by the candidate
token ids, summed in float64. L = log p(success text) - log p("Nothing happens.").

Formats (`--formats`, comma-separated):
  native     the model's chat template, generation prompt on, enable_thinking=False (the study's prompt)
  nothink    Qwen only: enable_thinking=True, so the generation prompt has no empty think block
  think_all  Qwen only: the empty think block also opens each few-shot answer, as it opens the scored answer
  sysfold    no system message; the system text opens the first user turn
  zeroshot   system message and the query turn only, no examples
  plain      no chat template: system text, examples and query as plain text; the candidate follows
             "Next observation:" after one space (Llama's text starts with its BOS token)

Cross-checks recorded per item: the independently built messages equal wf.scoring.logodds.build_messages's for
both conditions; the native text tokenized equals apply_chat_template(tokenize=True, return_dict=True)'s ids; the
candidate's in-context tokens equal its tokenization alone (what the study's scorer concatenates).

Results are appended per item (all formats and conditions of one item in one line) and a rerun skips items done.

    python scripts/qwen_audit/independent_score.py --items runs/qwen_audit/items.jsonl --model Qwen/Qwen3-8B \
        --formats native,nothink,think_all,sysfold,zeroshot,plain --out runs/qwen_audit/qwen_bf16.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

SYSTEM_TEXT = (
    "You are a text world model for household embodied tasks. "
    "Given the interaction history and the next action, predict the environment's "
    "next observation exactly in ALFWorld style. Output only the observation text."
)
EMPTY_THINK = "<think>\n\n</think>\n\n"
QWEN_ONLY = {"nothink", "think_all"}
TAGS = {"Qwen/Qwen3-8B": "qwen3-8b-fewshot", "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b-fewshot"}


# ---------------- prompts, from the item text ----------------


def swap_task_line(history: str, line: str) -> str:
    lines = history.split("\n")
    at = [i for i, x in enumerate(lines) if x.startswith("Your task is to:")]
    if len(at) != 1:
        raise ValueError(f"expected one task line, found {len(at)}")
    lines[at[0]] = line
    return "\n".join(lines)


def own_swapped_line(t: dict) -> str | None:
    """The swapped goal built here: the first whole-word occurrence of the goal object replaced."""
    task, goal, other = t["task_line"], t["goal_object"], t.get("swapped_goal_object")
    words = task.split(" ")
    for i, w in enumerate(words):
        if other and w.rstrip(".") == goal:
            words[i] = other + w[len(goal):]
            return " ".join(words)
    return None


def swapped_line(t: dict) -> str:
    """The stored swapped line, as the study used it (own_swapped_line is compared with it per item)."""
    return t.get("swapped_task_line") or own_swapped_line(t)


def turn(history: str, action: str) -> str:
    return f"History:\n{history}\n\nAction: {action}\n\nNext observation:"


def messages_for(t: dict, cond: str, shots: list[dict], *, fewshot: bool = True, fold_system: bool = False) -> list[dict]:
    line = t["task_line"] if cond == "aligned" else swapped_line(t)
    query = turn(swap_task_line(t["history_text"], line), t["action"])
    msgs: list[dict] = []
    if fewshot:
        for ex in shots:
            msgs += [{"role": "user", "content": turn(ex["history"], ex["action"])},
                     {"role": "assistant", "content": ex["observation"]}]
    msgs.append({"role": "user", "content": query})
    if fold_system:
        msgs[0] = {**msgs[0], "content": f"{SYSTEM_TEXT}\n\n{msgs[0]['content']}"}
        return msgs
    return [{"role": "system", "content": SYSTEM_TEXT}] + msgs


def plain_text(t: dict, cond: str, shots: list[dict], bos: str) -> str:
    """No chat template. The candidate is appended after one space."""
    line = t["task_line"] if cond == "aligned" else swapped_line(t)
    parts = [SYSTEM_TEXT]
    for ex in shots:
        parts.append(f"{turn(ex['history'], ex['action'])} {ex['observation']}")
    parts.append(turn(swap_task_line(t["history_text"], line), t["action"]))
    return bos + "\n\n".join(parts)


def render(tok, fmt: str, t: dict, cond: str, shots: list[dict], is_qwen: bool) -> tuple[str, str]:
    """(prompt text, separator put before the candidate)."""
    def chat(msgs, **kw):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    if fmt == "native":
        return chat(messages_for(t, cond, shots), enable_thinking=False), ""
    if fmt == "nothink":
        text = chat(messages_for(t, cond, shots), enable_thinking=True)
        if text.endswith(EMPTY_THINK) or not text.endswith("<|im_start|>assistant\n"):
            raise ValueError("thinking-on render does not end with a bare assistant header")
        return text, ""
    if fmt == "think_all":
        text = chat(messages_for(t, cond, shots), enable_thinking=False)
        head = "<|im_start|>assistant\n"
        out = re.sub(re.escape(head) + r"(?!<think>)", lambda _: head + EMPTY_THINK, text)
        if out.count(EMPTY_THINK) != len(shots) + 1:
            raise ValueError("think_all: wrong number of think blocks")
        return out, ""
    if fmt == "sysfold":
        return chat(messages_for(t, cond, shots, fold_system=True), enable_thinking=False), ""
    if fmt == "zeroshot":
        return chat(messages_for(t, cond, shots, fewshot=False), enable_thinking=False), ""
    if fmt == "plain":
        return plain_text(t, cond, shots, "" if is_qwen else (tok.bos_token or "")), " "
    raise ValueError(fmt)


# ---------------- scoring ----------------


class Scorer:
    def __init__(self, model_id: str, revision: str, dtype: str, attn: str | None, load: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.loaded_attn = None
        if not load:
            return
        kw = {"attn_implementation": attn} if attn else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                          torch_dtype=getattr(torch, dtype), **kw).to("cuda").eval()
        self.loaded_attn = getattr(self.model.config, "_attn_implementation", None)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def logprob(self, prompt: str, sep: str, cand: str) -> dict:
        torch = self.torch
        p_ids = self.ids(prompt)
        f_ids = self.ids(prompt + sep + cand)
        prefix_ok = f_ids[: len(p_ids)] == p_ids
        if not prefix_ok:  # the boundary merged across prompt and candidate: score the candidate tokenized alone
            f_ids = p_ids + self.ids(sep + cand)
        c_ids = f_ids[len(p_ids):]
        x = torch.tensor([f_ids], device="cuda")
        with torch.no_grad():
            logits = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0]
        p, n = len(p_ids), len(c_ids)
        lp = torch.log_softmax(logits[p - 1: p - 1 + n].float(), dim=-1)
        tok_lp = lp[torch.arange(n, device="cuda"), torch.tensor(c_ids, device="cuda")].double().cpu().tolist()
        return {"logp": math.fsum(tok_lp), "tok_lp": tok_lp, "n": n, "n_prompt": p,
                "first_tok": self.tok.decode(c_ids[:1]),
                "alone_ok": self.ids(sep + cand) == c_ids, "prefix_ok": prefix_ok,
                "logits_dtype": str(logits.dtype)}


def render_check(sc, items, formats, is_qwen, build_messages, shots) -> None:
    """Render every format for every item and report the checks; print one item's prompts in full."""
    from collections import Counter

    c = Counter()
    for k, it in enumerate(items):
        t = it["transition"]
        c["own_swap_equal"] += own_swapped_line(t) == t.get("swapped_task_line")
        for cond in ("aligned", "swapped"):
            mine = messages_for(t, cond, shots)
            c[f"messages_equal_repo_{cond}"] += mine == build_messages(t, cond)
            native, _ = render(sc.tok, "native", t, cond, shots, is_qwen)
            dict_ids = sc.tok.apply_chat_template(mine, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                  enable_thinking=False)["input_ids"]
            c[f"template_ids_equal_{cond}"] += list(dict_ids) == sc.ids(native)
            for fmt in formats:
                prompt, sep = render(sc.tok, fmt, t, cond, shots, is_qwen)
                for cand in t["candidates"].values():
                    p, f = sc.ids(prompt), sc.ids(prompt + sep + cand)
                    c[f"prefix_ok/{fmt}"] += f[: len(p)] == p
                    c[f"alone_ok/{fmt}"] += sc.ids(sep + cand) == f[len(p):]
                if k == 0 and cond == "swapped":
                    print(f"===== {fmt}/{cond} =====\n{prompt}{sep}<CANDIDATE>\n")
    print(f"{len(items)} items:", dict(sorted(c.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--formats", default="native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default=None, help="attn_implementation; default leaves transformers' choice")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render-only", action="store_true", help="CPU: render every format and run the checks, no model")
    args = ap.parse_args()

    from wf.constants import REVISIONS
    from wf.scoring.prompts import WM_FEWSHOT_SET0, WM_SYSTEM

    if WM_SYSTEM != SYSTEM_TEXT:
        raise SystemExit("system text differs from the study's WM_SYSTEM")
    is_qwen = args.model.startswith("Qwen/")
    formats = [f for f in args.formats.split(",") if f and (is_qwen or f not in QWEN_ONLY)]
    items = [json.loads(x) for x in Path(args.items).open()][: args.limit]
    out = Path(args.out)
    done = {json.loads(x)["id"] for x in out.open()} if out.exists() else set()

    try:  # the study's prompt builder, for the cross-check only
        from wf.scoring.logodds import build_messages
    except Exception as e:  # noqa: BLE001
        build_messages = None
        print(f"build_messages unavailable: {e!r}")

    sc = Scorer(args.model, REVISIONS[args.model], args.dtype, args.attn, load=not args.render_only)
    if args.render_only:
        render_check(sc, items, formats, is_qwen, build_messages, WM_FEWSHOT_SET0)
        return
    print(f"model={args.model} rev={REVISIONS[args.model]} dtype={args.dtype} attn={sc.loaded_attn} formats={formats}",
          flush=True)
    t0 = time.time()
    for k, it in enumerate(items):
        if it["id"] in done:
            continue
        t = it["transition"]
        cands = t["candidates"]
        rec = {"id": it["id"], "model": args.model, "dtype": args.dtype, "attn": sc.loaded_attn, "checks": {}, "scores": {}}
        rec["checks"]["own_swapped_line_equal_stored"] = own_swapped_line(t) == t.get("swapped_task_line")
        for cond in ("aligned", "swapped"):
            mine = messages_for(t, cond, WM_FEWSHOT_SET0)
            if build_messages is not None:
                try:
                    rec["checks"][f"messages_equal_repo_{cond}"] = mine == build_messages(t, cond)
                except Exception as e:  # noqa: BLE001
                    rec["checks"][f"messages_equal_repo_{cond}"] = f"error: {e!r}"
            dict_ids = sc.tok.apply_chat_template(mine, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, enable_thinking=False)["input_ids"]
            native_text, _ = render(sc.tok, "native", t, cond, WM_FEWSHOT_SET0, is_qwen)
            rec["checks"][f"template_ids_equal_{cond}"] = list(dict_ids) == sc.ids(native_text)
            rec["checks"][f"n_bos_{cond}"] = sc.ids(native_text).count(sc.tok.bos_token_id) if sc.tok.bos_token_id is not None else 0
            for fmt in formats:
                prompt, sep = render(sc.tok, fmt, t, cond, WM_FEWSHOT_SET0, is_qwen)
                plus = sc.logprob(prompt, sep, cands["o_plus"])
                minus = sc.logprob(prompt, sep, cands["o_minus"])
                rec["scores"][f"{fmt}/{cond}"] = {
                    "logp_plus": plus["logp"], "logp_minus": minus["logp"], "L": plus["logp"] - minus["logp"],
                    "L_first": plus["tok_lp"][0] - minus["tok_lp"][0], "n_plus": plus["n"], "n_minus": minus["n"],
                    "n_prompt": plus["n_prompt"], "first_plus": plus["first_tok"], "first_minus": minus["first_tok"],
                    "alone_ok": plus["alone_ok"] and minus["alone_ok"],
                    "prefix_ok": plus["prefix_ok"] and minus["prefix_ok"], "logits_dtype": plus["logits_dtype"],
                    "tok_lp_minus": minus["tok_lp"]}
                if k == 0:
                    print(f"--- {fmt}/{cond} prompt tail ---\n{prompt[-400:]!r}\n+ {sep + cands['o_plus']!r}", flush=True)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 20 == 0:
            s = rec["scores"]
            st = it["stored"][TAGS[args.model]]
            print(f"[{k}] {it['id']} native dL={s['native/aligned']['L'] - s['native/swapped']['L']:.3f} "
                  f"stored dL={st['aligned']['L'] - st['swapped']['L']:.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
