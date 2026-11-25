"""Lexsem diagnostic: score the frozen conditions (notes/lexsem_design.md). Not run until the design's rules are
frozen at an annotated tag.

The model is loaded by the study's scorer (`LogoddsScorer`, pinned revision, bf16) and every readout is read with
float32 log-softmax as there. Two prompt modes:

- "plain" (Qwen3-8B, primary): no chat template, the study's messages rendered by the Qwen audit's `plain_text`
  (scripts/qwen_audit/followup_score.py), the reply after one space. This is the format without the empty think
  block on which the audit relies (notes/qwen_effect_audit_pipeline.md, section 4 correction).
- "chat" (optional Llama-3.1-8B arm): the model's own chat template, as `LogoddsScorer.score_one_token_pair`.

Per item and condition, two forward passes:
- obs: the observation prompt followed by the item's success text. The logits at the reply's first position give
  the first-token readout (" You" against " Nothing"; neither names the object) and the two tokens' probability
  mass; the same pass gives the success text's per-token log-probabilities (descriptive copying check).
- yn: rq1v's anchored yes/no prompt; " Yes" against " No" at the answer position, and their mass.

Output is one JSON line per item, appended as each item finishes; a rerun skips finished items.

    python scripts/lexsem/score.py --model Qwen/Qwen3-8B --mode plain --out runs/lexsem/qwen_plain.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

EXPECTED_SAMPLE_DIGEST = "cc8da36181976edd"  # sha256 of the sorted sample ids, first 16 hex digits


def sample_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def token_checks(tok) -> list[str]:
    """Frozen token counts against the loaded tokenizer; any mismatch is returned (and refuses the run)."""
    from wf import lexsem

    n = lambda w: len(tok(" " + w, add_special_tokens=False)["input_ids"])  # noqa: E731
    bad = [f"{o}: {n(o)} != {k}" for o, k in lexsem.OBJECT_TOKENS.items() if n(o) != k]
    bad += [f"{w}: {n(w)} != {k}" for w, (k, _) in lexsem.CONTROL_POOL.items() if n(w) != k]
    return bad


def readout_ids(tok, mode: str) -> dict[str, tuple[int, int]]:
    sp = " " if mode == "plain" else ""
    out = {}
    for fmt, (a, b) in {"obs": ("You", "Nothing"), "yn": ("Yes", "No")}.items():
        ids = [tok(sp + w, add_special_tokens=False)["input_ids"] for w in (a, b)]
        if any(len(i) != 1 for i in ids):
            raise SystemExit(f"readout words are not single tokens in mode {mode}: {a!r}, {b!r} -> {ids}")
        out[fmt] = (ids[0][0], ids[1][0])
    return out


def prompt_ids(sc, msgs, mode: str) -> list[int]:
    if mode == "plain":
        from scripts.qwen_audit.followup_score import plain_text

        return sc._tok(plain_text(msgs), add_special_tokens=False)["input_ids"]
    return sc._encode(msgs)[0].tolist()


def read(sc, msgs, mode: str, pair: tuple[int, int], continuation: str | None = None) -> dict:
    """One forward pass: the pair's log-odds and mass at the reply's first position; with `continuation`, also the
    per-token log-probabilities of that text after the prompt (its first token must be the pair's first)."""
    import torch

    p = prompt_ids(sc, msgs, mode)
    full = p
    if continuation is not None:
        if mode == "plain":
            from scripts.qwen_audit.followup_score import plain_text

            full = sc._tok(plain_text(msgs) + " " + continuation, add_special_tokens=False)["input_ids"]
            if full[: len(p)] != p:
                full = p + sc._tok(" " + continuation, add_special_tokens=False)["input_ids"]
        else:
            full = p + sc._encode_candidate(continuation)[0].tolist()
        if full[len(p)] != pair[0]:
            raise ValueError(f"the continuation does not start with the readout token: {sc._tok.decode(full[len(p)])!r}")
    dev = next(sc._model.parameters()).device
    with torch.no_grad():
        logits = sc._model(input_ids=torch.tensor([full], device=dev)).logits[0]
    lp = torch.log_softmax(logits[len(p) - 1:].float(), dim=-1)
    a, b = lp[0, pair[0]].item(), lp[0, pair[1]].item()
    out = {"L": a - b, "mass": math.exp(a) + math.exp(b), "n_prompt": len(p)}
    if continuation is not None:
        c = full[len(p):]
        out["tok_lp"] = [round(lp[k, c[k]].item(), 5) for k in range(len(c))]
        out["toks"] = [sc._tok.decode([x]) for x in c]
    return out


def manipulation_check(sc, recs: list[dict]) -> list[dict]:
    """rq1x's naming check (its prompt and reader, the model's chat template) for every alias and description the
    sample uses, the control descriptions included."""
    from wf import rq1x

    asked = {}
    for r in recs:
        refs = r["references"]
        asked[(r["goal_object"], "desc", refs["desc"])] = None
        asked[(refs["cdesc_object"], "cdesc", refs["cdesc"])] = None
        if refs.get("alias"):
            asked[(r["goal_object"], "alias", refs["alias"])] = None
    out = []
    for obj, form, text in sorted(asked):
        ans = sc.free_generate([{"role": "user", "content": rq1x.CHECK_PROMPT.format(desc=text)}], max_new_tokens=16)
        out.append({"object": obj, "form": form, "text": text, "answer": ans, "passed": rq1x.check_says(obj, ans)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--mode", choices=("plain", "chat"), default="plain")
    ap.add_argument("--conditions", default="core", help="core, or a comma list of condition names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from wf import lexsem, rq1x
    from wf.rq2 import load_items
    from wf.scoring.logodds import LogoddsScorer

    conds = lexsem.CORE if args.conditions == "core" else tuple(args.conditions.split(","))
    sc = LogoddsScorer(args.model, model_tag=args.model.split("/")[-1].lower())
    bad = token_checks(sc._tok)
    if bad:
        raise SystemExit("frozen token counts disagree with the loaded tokenizer: " + "; ".join(bad))
    pairs = readout_ids(sc._tok, args.mode)
    count = lambda s: len(sc._tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    items = lexsem.sample(rq1x.rq1x_sets(load_items())["primary"])
    if sample_digest([t["id"] for t in items]) != EXPECTED_SAMPLE_DIGEST:
        raise SystemExit("the rebuilt sample differs from the frozen one")
    recs = lexsem.build(items, conds, count_tokens=count)
    by_id = {t["id"]: t for t in items}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.stem}_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    check = out.parent / f"{out.stem}_check.json"
    if not check.exists():
        check.write_text(json.dumps(manipulation_check(sc, recs), indent=1))
    done = {json.loads(x)["id"] for x in out.open()} if out.exists() else set()
    t0 = time.time()
    for k, r in enumerate(recs[: args.limit]):
        if r["id"] in done:
            continue
        t = by_id[r["id"]]
        rec = {"id": r["id"], "game_id": r["game_id"], "stratum": r["stratum"], "model": args.model,
               "mode": args.mode, "scores": {}}
        for c, line in r["lines"].items():
            rec["scores"][c] = {
                "obs": read(sc, lexsem.messages(t, line, "obs"), args.mode, pairs["obs"], continuation=r["o_plus"]),
                "yn": read(sc, lexsem.messages(t, line, "yn"), args.mode, pairs["yn"]),
            }
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 25 == 0:
            print(f"[{k}/{len(recs)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(recs)} items in {time.time() - t0:.0f}s", flush=True)
    sc.close()


if __name__ == "__main__":
    main()
