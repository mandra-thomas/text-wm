"""Option C follow-up, task 1 (2026-11-24; spec in notes/optionc_followups_2026-11-24.md), CPU only.

The step-0 yes/no prompts that the step-0 check's `stage1a_agreement` compares (Option C's aware, swapped and neutral
arms against Stage 1a's yn_aligned, yn_swapped and yn_absent), built by the code that made each set of scores:

  --code optionc   this checkout's wf.optionc (verifier_messages on step0_item), as optionc_step0_<tag> built them;
  --code stage1a   wf.decisionsim at Stage 1a's scoring commit eeeb7dd (messages_for on prompt_item), as
                   decisionsim_score_<tag> built them. PYTHONPATH must point at a copy of that commit's wf/.

Each record holds the key (transition id, Stage 1a condition), the sha256 of the messages (sorted-key JSON), the
sha256 of the token ids the scorer reads after (the scorer's own _encode: its chat template and template settings,
tokenizer at its pinned revision), and whether the hash that run stored with its own score matches the prompt rebuilt
here (--stored: that run's scoring.jsonl files). `compare` reads two dumps and reports how many keys agree.

    python scripts/optionc_followup/prompt_dump.py dump --code optionc --model qwen3-8b \
        --stored runs/optionc_step0_qwen3-8b/scoring.jsonl --out runs/optionc_fu/prompts_oc_qwen3-8b.jsonl
    PYTHONPATH=/path/to/eeeb7dd python scripts/optionc_followup/prompt_dump.py dump --code stage1a --model qwen3-8b \
        --stored runs/decisionsim_score_qwen3-8b/shard_00_of_01/scoring.jsonl --out runs/optionc_fu/prompts_s1a_qwen3-8b.jsonl
    python scripts/optionc_followup/prompt_dump.py compare runs/optionc_fu/prompts_oc_qwen3-8b.jsonl \
        runs/optionc_fu/prompts_s1a_qwen3-8b.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARM_OF = {"aware": "yn_aligned", "swapped": "yn_swapped", "neutral": "yn_absent"}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def stored_hashes(paths: list[str], key_field: str) -> dict[tuple[str, str], str]:
    out = {}
    for p in paths:
        for line in Path(p).open():
            r = json.loads(line)
            if key_field == "arm":
                if r.get("readout") != "yn" or r["arm"] not in ARM_OF:
                    continue
                out[(r["transition_id"], ARM_OF[r["arm"]])] = r["score_hash"]
            else:
                out[(r["transition_id"], r["condition"])] = r["score_hash"]
    return out


def dump(args) -> None:
    from transformers import AutoTokenizer

    from wf.rq2 import SOURCES

    rows = []
    if args.code == "optionc":
        from wf import optionc as O

        states = O.step0_states(list(SOURCES))
        stored = stored_hashes(args.stored, "arm")
        for s in states:
            for t in s["candidates"]:
                for arm, cond in ARM_OF.items():
                    if arm == "swapped" and not s.get("swap"):
                        continue
                    msgs = O.verifier_messages(args.model, O.step0_item(s, t), arm, "yn")
                    rows.append(((t["id"], cond), msgs, O._sha([args.model, "yn", msgs])))
        from wf.decisionsim import REFERENCE_MODELS
    else:
        from wf import decisionsim as D

        states = [s for s in D.select_states(D.build_states(D.load_items(SOURCES), D.C.RNG_SEED), "all", D.C.RNG_SEED)
                  if s["state_idx"] == 0]
        stored = stored_hashes(args.stored, "condition")
        build = D.messages_for(args.model)
        for s in states:
            for t in s["candidates"]:
                for cond in ARM_OF.values():
                    item = D.prompt_item(t, s, cond)
                    if item is None:
                        continue
                    rows.append(((t["id"], cond), build(item, cond), D.score_hash(t, cond, s)))
        REFERENCE_MODELS = D.REFERENCE_MODELS
    from wf.scoring.logodds import LogoddsScorer, revision_for

    hf = REFERENCE_MODELS[args.model]
    rev = revision_for(hf)
    sc = LogoddsScorer.for_tokenizer(hf, AutoTokenizer.from_pretrained(hf, revision=rev, trust_remote_code=True),
                                     model_tag=args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_match = n_stored = 0
    with out.open("w") as f:
        for key, msgs, h in rows:
            ids = sc._encode(msgs)[0].tolist()
            st = stored.get(key)
            n_stored += st is not None
            n_match += st == h
            f.write(json.dumps({"transition_id": key[0], "condition": key[1], "msg_sha": sha(msgs), "ids_sha": sha(ids),
                                "n_tokens": len(ids), "ends_with_empty_think": sc._tok.decode(ids[-6:]).endswith("</think>\n\n"),
                                "stored_hash_matches": st == h if st is not None else None}) + "\n")
    print(json.dumps({"code": args.code, "model": args.model, "revision": rev, "template": sc.template_settings(),
                      "n_prompts": len(rows), "n_states": len(states), "n_with_stored_record": n_stored,
                      "stored_hash_matches": n_match}))


def compare(args) -> None:
    a = {(r["transition_id"], r["condition"]): r for r in map(json.loads, Path(args.a).open())}
    b = {(r["transition_id"], r["condition"]): r for r in map(json.loads, Path(args.b).open())}
    common = sorted(set(a) & set(b))
    print(json.dumps({"n_a": len(a), "n_b": len(b), "n_common": len(common),
                      "messages_identical": sum(a[k]["msg_sha"] == b[k]["msg_sha"] for k in common),
                      "token_ids_identical": sum(a[k]["ids_sha"] == b[k]["ids_sha"] for k in common),
                      "empty_think_block_a": sum(a[k]["ends_with_empty_think"] for k in common),
                      "empty_think_block_b": sum(b[k]["ends_with_empty_think"] for k in common)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--code", choices=("optionc", "stage1a"), required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--stored", nargs="+", required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    (dump if args.cmd == "dump" else compare)(args)


if __name__ == "__main__":
    main()
