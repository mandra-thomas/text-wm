"""Option C follow-up, task 2 (2026-11-24; spec in notes/optionc_followups_2026-11-24.md): the step-0 check in plain
text, Qwen3-8B.

Scores every job of optionc_step0_qwen3-8b (Stage 1a's 134 step-0 states, every candidate, the aware, blind, swapped
and neutral arms, the yes/no and observation readouts), with Option C's own prompts (wf.optionc.verifier_messages on
step0_item), rendered without Qwen3's chat template: the plain-text format of the Qwen pipeline audit (section 7,
scripts/qwen_audit/followup_score.py on main at 484933e; `plain_text` and `plain_pair` are copied from it unchanged).

- yes/no: `plain_pair(sc, msgs, "Yes", "No")`, the teacher-forced log-probabilities of " Yes" and " No" after one
  space, exactly as the audit's plain-text step-0 rescoring (the 38.8-point figure) read them; L = log P(" Yes") -
  log P(" No"); the two answers' first-token mass is recorded.
- observation: the next-token distribution after the plain-text prompt, at the position where the reply " Nothing
  happens." would begin (one space, as the examples' answers follow their prompts in plain text). The failure token is
  the first token of " Nothing happens." in the joint tokenization of prompt and reply (it must be the token after the
  prompt's own tokens), and L = log(1 - P(fail)) - log P(fail), Option C's observation readout with that token;
  L_you = log P(" You") - log P(fail) and the two tokens' mass are recorded.

Scores are rounded to 1/64 nat (L) with the raw value kept (L_raw), in the record format of optionc_step0_<tag>'s
scoring.jsonl, so wf.optionc.step0_report reads them unchanged. Output is appended per state; a rerun skips finished
states. --check-block N first rescores every job of the first N states with the block, through Option C's own path
(read_score, every position's logits), for comparison with the logged with-block scores (same GPU type expected).

    python scripts/optionc_followup/plain_step0.py --out runs/optionc_fu/plain_step0_qwen3-8b.jsonl --check-block 3
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

MODEL = "qwen3-8b"


# ---- copied unchanged from scripts/qwen_audit/followup_score.py (main, 484933e) ----

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

# ---- end of the copy ----


def next_token_id(sc, prompt_ids: list[int], prompt: str, reply: str) -> int:
    """The first token of `reply` after `prompt` in their joint tokenization; the prompt's own tokens must be a prefix."""
    f = sc._tok(prompt + reply, add_special_tokens=False)["input_ids"]
    if f[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"the prompt's tokens are not a prefix of prompt+{reply!r}")
    return f[len(prompt_ids)]


def plain_obs(sc, msgs: list[dict]) -> dict:
    import torch

    from wf.optionc import NOTHING

    prompt = plain_text(msgs)
    p = sc._tok(prompt, add_special_tokens=False)["input_ids"]
    fail = next_token_id(sc, p, prompt, " " + NOTHING)
    you = next_token_id(sc, p, prompt, " You")
    dev = next(sc._model.parameters()).device
    with torch.no_grad():
        last = sc._model(input_ids=torch.tensor([p], device=dev)).logits[0, -1]
    lp = torch.log_softmax(last.float(), dim=-1)
    mask = torch.ones_like(lp, dtype=torch.bool)
    mask[fail] = False
    lp_not = torch.logsumexp(lp[mask], dim=0).item()
    lp_fail, lp_you = lp[fail].item(), lp[you].item()
    return {"L_raw": lp_not - lp_fail, "logp_nothing": lp_fail, "logp_not_nothing": lp_not, "logp_you": lp_you,
            "L_you": lp_you - lp_fail, "first_mass": math.exp(lp_fail) + math.exp(lp_you),
            "fail_token": sc._tok.decode([fail]), "you_token": sc._tok.decode([you])}


def gpu_name() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().split("\n")[0]
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-block", type=int, default=0)
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()

    from wf import optionc as O
    from wf.rq2 import SOURCES

    states = O.step0_states(list(SOURCES))[: args.limit_states]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["game_id"] for x in out.open()} if out.exists() else set()
    sc = O.open_verifier(MODEL)
    assert sc.last_only is False
    gpu = gpu_name()
    print(f"model={MODEL} rev={sc.revision} gpu={gpu} states={len(states)} done={len(done)}", flush=True)
    t0 = time.time()
    for k, s in enumerate(states):
        if s["game_id"] in done:
            continue
        recs, block = [], []
        for _, t, arm, readout in O.step0_jobs([s], tuple(O.VERIFIER_ARMS), O.READOUTS):
            msgs = O.verifier_messages(MODEL, O.step0_item(s, t), arm, readout)
            base = {"transition_id": t["id"], "game_id": s["game_id"], "state_idx": s["state_idx"], "action": t["action"],
                    "arm": arm, "readout": readout, "score_hash": O._sha([MODEL, readout, msgs])}
            if readout == O.YN:
                r = plain_pair(sc, msgs, "Yes", "No")
                rec = {"L_raw": r["L"], "logp_yes": r["logp_plus"], "logp_no": r["logp_minus"], "L_first": r["L_first"],
                       "first_mass": r["first_mass"], "n_tokens_yes": r["n_tokens_plus"], "n_tokens_no": r["n_tokens_minus"]}
            else:
                rec = plain_obs(sc, msgs)
            recs.append({**base, **rec, "L": O.quantize(rec["L_raw"]), "format": "plain"})
            if k < args.check_block:
                block.append({**base, **O.read_score(sc, msgs, readout, t["id"], arm), "format": "block"})
        with out.open("a") as f:
            f.write(json.dumps({"game_id": s["game_id"], "state_idx": s["state_idx"], "gpu": gpu, "revision": sc.revision,
                                "records": recs, "block_check": block}) + "\n")
        if k % 10 == 0:
            print(f"[{k}/{len(states)}] {time.time() - t0:.0f}s", flush=True)
    sc.close()
    print(f"done: {len(states)} states in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
