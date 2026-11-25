"""Option C follow-up, task 1 (2026-11-24; spec in notes/optionc_followups_2026-11-24.md).

GPU diagnostic: Option C's own step-0 scoring path (wf.optionc.read_score on verifier_messages, the scorer opened by
open_verifier, every position's logits as in optionc_step0_<tag>) applied to the yes/no prompts that the step-0
check's stage1a_agreement compares (aware, swapped and neutral arms), on the GPU type Stage 1a's scores were made on.
Output is appended per state; a rerun skips finished states.

    python scripts/optionc_followup/rescore_stage1a.py score --model qwen3-8b --out runs/optionc_fu/rescore_qwen3-8b.jsonl
    python scripts/optionc_followup/rescore_stage1a.py compare --model qwen3-8b --new runs/optionc_fu/rescore_qwen3-8b.jsonl \
        --stage1a runs/decisionsim_score_qwen3-8b/shard_00_of_01/scoring.jsonl --optionc runs/optionc_step0_qwen3-8b/scoring.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

ARM_OF = {"aware": "yn_aligned", "swapped": "yn_swapped", "neutral": "yn_absent"}


def gpu_name() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().split("\n")[0]
    except Exception:
        return "unknown"


def score(args) -> None:
    from wf import optionc as O
    from wf.rq2 import SOURCES

    states = O.step0_states(list(SOURCES))[: args.limit_states]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["game_id"] for x in out.open()} if out.exists() else set()
    sc = O.open_verifier(args.model)
    assert sc.last_only is False  # the step-0 run never ran last_only_check, so it kept every position's logits
    gpu = gpu_name()
    print(f"model={args.model} rev={sc.revision} gpu={gpu}", flush=True)
    t0 = time.time()
    for k, s in enumerate(states):
        if s["game_id"] in done:
            continue
        recs = []
        for t in s["candidates"]:
            for arm in ARM_OF:
                if arm == "swapped" and not s.get("swap"):
                    continue
                msgs = O.verifier_messages(args.model, O.step0_item(s, t), arm, "yn")
                r = O.read_score(sc, msgs, "yn", t["id"], arm)
                recs.append({"transition_id": t["id"], "arm": arm, "readout": "yn", **r,
                             "score_hash": O._sha([args.model, "yn", msgs])})
        with out.open("a") as f:
            f.write(json.dumps({"game_id": s["game_id"], "state_idx": s["state_idx"], "gpu": gpu, "revision": sc.revision,
                                "records": recs}) + "\n")
        if k % 10 == 0:
            print(f"[{k}/{len(states)}] {time.time() - t0:.0f}s", flush=True)
    sc.close()
    print(f"done: {len(states)} states in {time.time() - t0:.0f}s", flush=True)


def q(x: float) -> float:
    return round(x * 64) / 64


def agreement(pairs: list[tuple[float, float]]) -> dict:
    d = [abs(a - b) for a, b in pairs]
    dq = [abs(q(a) - q(b)) for a, b in pairs]
    return {"n": len(pairs), "identical_after_rounding": sum(x == 0 for x in dq),
            "share_identical": sum(x == 0 for x in dq) / len(dq) if dq else None,
            "mean_abs_diff_raw": statistics.mean(d) if d else None, "max_abs_diff_rounded": max(dq) if dq else None,
            "acceptance_flips": sum((q(a) > 0) != (q(b) > 0) for a, b in pairs)}


def compare(args) -> None:
    new, gpus, hashes_ok = {}, set(), 0
    oc_rows = [json.loads(x) for x in Path(args.optionc).open()]
    oc = {(r["transition_id"], r["arm"]): r for r in oc_rows if r["readout"] == "yn"}
    for line in Path(args.new).open():
        st = json.loads(line)
        gpus.add(st["gpu"])
        for r in st["records"]:
            new[(r["transition_id"], r["arm"])] = r
            hashes_ok += oc.get((r["transition_id"], r["arm"]), {}).get("score_hash") == r["score_hash"]
    s1a = {(r["transition_id"], r["condition"]): r["L"] for r in map(json.loads, Path(args.stage1a).open())}
    keys = [k for k in new if (k[0], ARM_OF[k[1]]) in s1a and k in oc]
    res = {"model": args.model, "gpu_new": sorted(gpus), "n_new": len(new), "prompt_hash_equals_optionc_record": hashes_ok,
           "new_vs_stage1a": agreement([(new[k]["L_raw"], s1a[(k[0], ARM_OF[k[1]])]) for k in keys]),
           "new_vs_optionc_step0": agreement([(new[k]["L_raw"], oc[k]["L_raw"]) for k in keys]),
           "optionc_step0_vs_stage1a": agreement([(oc[k]["L_raw"], s1a[(k[0], ARM_OF[k[1]])]) for k in keys])}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit-states", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("--model", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--stage1a", required=True)
    c.add_argument("--optionc", required=True)
    c.add_argument("--out", default=None)
    args = ap.parse_args()
    (score if args.cmd == "score" else compare)(args)


if __name__ == "__main__":
    main()
