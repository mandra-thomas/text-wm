"""Lexsem diagnostic: score the frozen conditions (notes/lexsem_design.md). Not run, except the smoke mode, until the
design's rules are frozen at an annotated tag.

The model is loaded as the study's scorer loads it (`LogoddsScorer._load`: pinned revision, bf16, device_map auto,
trust_remote_code), with the dtype keyword the installed transformers expects, and every readout is a float32
log-softmax of the model's own bf16 logits. Three prompt modes:

- "plain" (Qwen3-8B, primary): no chat template; the study's messages rendered by the Qwen audit's `plain_text`
  (scripts/qwen_audit/followup_score.py); the reply after one space;
- "block" (Qwen3-8B, the study's own format): the chat template with thinking off, which ends in the empty think
  block; readout tokens without a leading space, as `LogoddsScorer.score_one_token_pair`;
- "chat" (Llama-3.1-8B): the model's own chat template, likewise.

Per item, condition and format, one sequence:
- obs: the observation prompt followed by the item's success text. At the reply's first position it gives the
  log-probabilities of "You" and "Nothing" (both unnormalized, from the full vocabulary); along the text, every
  token's log-probability (the model's probability of the whole success observation, and the token split).
- yn: rq1v's anchored yes/no prompt; the log-probabilities of "Yes" and "No".

The sequences of an item are scored in right-padded batches (`--batch-size`); only the needed positions go through
the output layer. Output is one JSON line per item, appended as each item finishes; a rerun skips finished items.

    python scripts/lexsem/score.py --mode plain --groups core --out runs/lexsem/qwen3-8b_plain.jsonl
    python scripts/lexsem/score.py --mode plain --groups core,ignore,position --smoke 5 --out runs/lexsem/smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

EXPECTED_SAMPLE_DIGEST = "cc8da36181976edd"  # sha256 of the sorted sample ids, first 16 hex digits
READOUT = {"obs": ("You", "Nothing"), "yn": ("Yes", "No")}
FAILURE_TEXT = "Nothing happens."  # the "fail" format (optional): the failure text after the observation prompt


def sample_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def token_checks(tok) -> list[str]:
    """Frozen token counts against the loaded tokenizer; any mismatch is returned (and refuses the run)."""
    from wf import lexsem

    n = lambda w: len(tok(" " + w, add_special_tokens=False)["input_ids"])  # noqa: E731
    return [f"{o}: {n(o)} != {k}" for o, k in lexsem.NAME_TOKENS.items() if n(o) != k]


def count_occurrences(seq: list[int], sub: list[int]) -> int:
    return sum(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def readout_ids(tok, mode: str) -> dict[str, tuple[int, int]]:
    sp = " " if mode == "plain" else ""
    out = {}
    for fmt, (a, b) in READOUT.items():
        ids = [tok(sp + w, add_special_tokens=False)["input_ids"] for w in (a, b)]
        if any(len(i) != 1 for i in ids):
            raise SystemExit(f"readout words are not single tokens in mode {mode}: {a!r}, {b!r} -> {ids}")
        out[fmt] = (ids[0][0], ids[1][0])
    return out


def load(model_id: str):
    """Tokenizer and model, loaded with the study scorer's settings (wf/scoring/logodds.py, LogoddsScorer._load)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wf.constants import revision_for

    rev = revision_for(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    dtype = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, device_map="auto", trust_remote_code=True,
                                                 **dtype)
    model.eval()
    return tok, model, rev


def prompt_ids(tok, msgs, mode: str) -> list[int]:
    if mode == "plain":
        from scripts.qwen_audit.followup_score import plain_text

        return tok(plain_text(msgs), add_special_tokens=False)["input_ids"]
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(out, "keys"):  # transformers 5 returns a mapping
        out = out["input_ids"]
    return list(out[0] if out and isinstance(out[0], list) else out)


def sequence(tok, msgs, mode: str, continuation: str | None) -> tuple[list[int], int]:
    """(token ids, prompt length). With `continuation` the ids run on through it, tokenized as the study's scorers
    do: jointly after one space in plain text (as `plain_pair`), alone after a chat template (as `score_pair`)."""
    p = prompt_ids(tok, msgs, mode)
    if continuation is None:
        return p, len(p)
    if mode == "plain":
        from scripts.qwen_audit.followup_score import plain_text

        full = tok(plain_text(msgs) + " " + continuation, add_special_tokens=False)["input_ids"]
        if full[: len(p)] != p:
            full = p + tok(" " + continuation, add_special_tokens=False)["input_ids"]
        return full, len(p)
    return p + tok(continuation, add_special_tokens=False)["input_ids"], len(p)


def score_batch(model, seqs: list[tuple[list[int], int, bool]], pad_id: int) -> list:
    """Right-padded batch; per sequence, float32 log-softmax rows at the reply's first position and (when the third
    element is True) at every continuation position. Returns one tensor (positions x vocab) per sequence."""
    import torch

    dev = next(model.parameters()).device
    T = max(len(s) for s, _, _ in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros((len(seqs), T), dtype=torch.long, device=dev)
    for i, (s, _, _) in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=dev)
        mask[i, : len(s)] = 1
    base, head = model.get_decoder(), model.get_output_embeddings()
    with torch.no_grad():
        h = base(input_ids=ids, attention_mask=mask).last_hidden_state
        out = []
        for i, (s, n_p, cont) in enumerate(seqs):
            pos = list(range(n_p - 1, len(s) - 1)) if cont else [n_p - 1]
            out.append(torch.log_softmax(head(h[i, pos]).float(), dim=-1).cpu())
    return out


def item_jobs(t: dict, rec: dict, tok, mode: str, formats: tuple[str, ...]) -> list[dict]:
    """Every (condition, format) sequence of one item, with the token-identity check of design section 5: a name or
    homonym condition must add exactly one copy of the name's token sequence to the swapped prompt, every other
    condition none."""
    from wf import lexsem

    name_ids = tok(" " + t["goal_object"], add_special_tokens=False)["input_ids"]
    base = count_occurrences(prompt_ids(tok, lexsem.messages(t, rec["specs"]["swapped"], "obs"), mode), name_ids)
    jobs = []
    for c, spec in rec["specs"].items():
        for fmt in formats:
            msgs = lexsem.messages(t, spec, "obs" if fmt == "fail" else fmt)
            cont = {"obs": rec["o_plus"], "fail": FAILURE_TEXT}.get(fmt)
            ids, n_p = sequence(tok, msgs, mode, cont)
            if fmt == "obs":
                form = lexsem.MENTIONS[c][0] if c in lexsem.MENTIONS else ("aligned" if c == "aligned" else None)
                added = count_occurrences(ids[:n_p], name_ids) - base
                want = 1 if form in ("name", "homonym") else (0 if c != "aligned" else added)
                if added != want:
                    raise SystemExit(f"{t['id']} {c}: adds {added} copies of the name's tokens, expected {want}")
            jobs.append({"cond": c, "fmt": fmt, "ids": ids, "n_prompt": n_p})
    return jobs


def readouts(jobs: list[dict], rows: list, pairs: dict, tok) -> dict:
    out: dict = {}
    for j, lp in zip(jobs, rows):
        a, b = pairs["obs" if j["fmt"] == "fail" else j["fmt"]]
        r = {"lp_a": round(lp[0, a].item(), 5), "lp_b": round(lp[0, b].item(), 5), "n_prompt": j["n_prompt"]}
        if j["fmt"] in ("obs", "fail"):
            cont = j["ids"][j["n_prompt"]:]
            r["tok_lp"] = [round(lp[k, cont[k]].item(), 5) for k in range(len(cont))]
            r["toks"] = [tok.decode([x]) for x in cont]
            if cont[0] != (a if j["fmt"] == "obs" else b):
                raise SystemExit(f"{j['fmt']} text does not start with its readout token: {r['toks'][0]!r}")
        out.setdefault(j["cond"], {})[j["fmt"]] = r
    return out


def run_item(model, jobs: list[dict], batch_size: int, pad_id: int) -> list:
    order = sorted(range(len(jobs)), key=lambda k: len(jobs[k]["ids"]))
    rows: list = [None] * len(jobs)
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        seqs = [(jobs[k]["ids"], jobs[k]["n_prompt"], jobs[k]["fmt"] in ("obs", "fail")) for k in chunk]
        res = score_batch(model, seqs, pad_id)
        for k, r in zip(chunk, res):
            rows[k] = r
    return rows


def manipulation_check(tok, model, recs: list[dict]) -> list[dict]:
    """Design section 5, M4. Descriptions: rq1x's naming question, greedy, in the model's chat template with
    thinking off (as rq1x), read by `lexsem.check_reads` (rq1x's reader plus the frozen lenient synonyms). Aliases:
    a forced choice, "Can the word "<alias>" refer to a <object>?", which must favour Yes, and the same question
    with the frozen foil object, which must favour No."""
    import torch

    from wf import lexsem, rq1x

    dev = next(model.parameters()).device
    yes, no = (tok(w, add_special_tokens=False)["input_ids"][0] for w in ("Yes", "No"))

    def ids(q: str) -> list[int]:
        return prompt_ids(tok, [{"role": "user", "content": q}], "block")

    out = []
    descs = sorted({(r["goal_object"], r["references"]["desc"]) for r in recs}
                   | {(r["references"]["ctrl"], r["references"]["cdesc"]) for r in recs})
    for obj, text in descs:
        x = torch.tensor([ids(rq1x.CHECK_PROMPT.format(desc=text))], device=dev)
        with torch.no_grad():
            g = model.generate(input_ids=x, max_new_tokens=16, do_sample=False, eos_token_id=tok.eos_token_id)
        ans = tok.decode(g[0, x.shape[1]:], skip_special_tokens=True).strip()
        out.append({"form": "desc", "object": obj, "text": text, "answer": ans, "passed": lexsem.check_reads(obj, ans)})
    for obj, alias in sorted(lexsem.ALIASES.items()):
        res = {}
        for target in (obj, lexsem.ALIAS_FOIL[obj]):
            q = lexsem.CHECK_ALIAS.format(alias=alias, art=lexsem._article(target), obj=target)
            lp = score_batch(model, [(ids(q), len(ids(q)), False)], tok.eos_token_id)[0]
            res[target] = lp[0, yes].item() - lp[0, no].item()
        out.append({"form": "alias", "object": obj, "text": alias, "yes_minus_no": res,
                    "passed": res[obj] > 0 > res[lexsem.ALIAS_FOIL[obj]]})
    return out


def smoke(args, tok, model, recs, by_id, pairs) -> dict:
    """Timing and validity only: seconds per sequence at each batch size, batched against one-at-a-time readouts,
    the two readout tokens' probability mass per condition, the failure text's tail after "Nothing". No effect
    (no difference between conditions) is computed or written."""
    import math

    import torch

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    recs = recs[: args.smoke]
    jobs = [item_jobs(by_id[r["id"]], r, tok, args.mode, tuple(args.formats.split(","))) for r in recs]
    rep: dict = {"mode": args.mode, "model": args.model, "n_items": len(recs),
                 "n_sequences": sum(len(j) for j in jobs),
                 "mean_tokens": sum(len(x["ids"]) for j in jobs for x in j) / sum(len(j) for j in jobs),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "timing": {}}
    rows_by_bs = {}
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        torch.cuda.synchronize()
        t0 = time.time()
        rows_by_bs[bs] = [run_item(model, j, bs, pad) for j in jobs]
        torch.cuda.synchronize()
        dt = time.time() - t0
        rep["timing"][str(bs)] = {"seconds": round(dt, 2), "s_per_sequence": round(dt / rep["n_sequences"], 4)}
    ref = rows_by_bs[min(rows_by_bs)]
    diffs = {}
    for bs, rows in rows_by_bs.items():
        d = 0.0
        for j, r0, r1 in zip(jobs, ref, rows):
            for job, x, y in zip(j, r0, r1):
                a, b = pairs["obs" if job["fmt"] == "fail" else job["fmt"]]
                d = max(d, abs((x[0, a] - x[0, b]).item() - (y[0, a] - y[0, b]).item()))
        diffs[str(bs)] = round(d, 4)
    rep["max_abs_logodds_diff_vs_smallest_batch"] = diffs
    mass: dict = {}
    for j, rows in zip(jobs, ref):
        for job, lp in zip(j, rows):
            a, b = pairs["obs" if job["fmt"] == "fail" else job["fmt"]]
            mass.setdefault(f"{job['cond']}|{job['fmt']}", []).append(math.exp(lp[0, a].item()) + math.exp(lp[0, b].item()))
    rep["mass_mean"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(mass.items())}
    rep["mass_min"] = {k: round(min(v), 4) for k, v in sorted(mass.items())}
    if "obs" in args.formats:
        from wf import lexsem

        tails = []
        for r in recs:
            t = by_id[r["id"]]
            for c in ("aligned", "swapped"):
                msgs = lexsem.messages(t, r["specs"][c], "obs")
                ids, n_p = sequence(tok, msgs, args.mode, "Nothing happens.")
                lp = score_batch(model, [(ids, n_p, True)], pad)[0]
                cont = ids[n_p:]
                tails.append(sum(lp[k, cont[k]].item() for k in range(1, len(cont))))
        rep["failure_text_after_first_token_nats"] = {"min": round(min(tails), 4), "max": round(max(tails), 4),
                                                       "n": len(tails)}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--mode", choices=("plain", "block", "chat"), default="plain")
    ap.add_argument("--groups", default="core", help="comma list of wf.lexsem.GROUPS names")
    ap.add_argument("--formats", default="obs,yn",
                    help="obs, yn and optionally fail (the failure text, for the full two-candidate score)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0, help="timing and validity only, on this many items")
    ap.add_argument("--batch-sizes", default="1,8,16,32", help="smoke only")
    args = ap.parse_args()

    from wf import lexsem, rq1x
    from wf.rq2 import load_items

    conds = tuple(dict.fromkeys(c for g in args.groups.split(",") for c in lexsem.GROUPS[g]))
    tok, model, rev = load(args.model)
    bad = token_checks(tok)
    if bad:
        raise SystemExit("frozen token counts disagree with the loaded tokenizer: " + "; ".join(bad))
    pairs = readout_ids(tok, args.mode)
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    items = lexsem.sample(rq1x.rq1x_sets(load_items())["primary"])
    if sample_digest([t["id"] for t in items]) != EXPECTED_SAMPLE_DIGEST:
        raise SystemExit("the rebuilt sample differs from the frozen one")
    recs = lexsem.build(items, conds, count_tokens=count)
    by_id = {t["id"]: t for t in items}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        t0 = time.time()
        rep = smoke(args, tok, model, recs, by_id, pairs)
        rep.update({"revision": rev, "groups": args.groups, "formats": args.formats, "wall_s": round(time.time() - t0, 1)})
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep["timing"]), flush=True)
        return
    (out.parent / f"{out.stem}_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    check = out.parent / f"{out.stem}_check.json"
    if not check.exists():
        check.write_text(json.dumps(manipulation_check(tok, model, recs), indent=1))
    pad =tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    done = {json.loads(x)["id"] for x in out.open()} if out.exists() else set()
    t0 = time.time()
    for k, r in enumerate(recs[: args.limit]):
        if r["id"] in done:
            continue
        jobs = item_jobs(by_id[r["id"]], r, tok, args.mode, tuple(args.formats.split(",")))
        rec = {"id": r["id"], "game_id": r["game_id"], "stratum": r["stratum"], "model": args.model,
               "revision": rev, "mode": args.mode,
               "scores": readouts(jobs, run_item(model, jobs, args.batch_size, pad), pairs, tok)}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 25 == 0:
            print(f"[{k}/{len(recs)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(recs)} items in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
