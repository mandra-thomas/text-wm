"""Blind human audit of `parse_prediction` on model-written predictions (2026-11-17).

export: writes a CSV a person labels S (says the action works), F (says it does not) or ? (does not
        commit), without the parser's answer or the true outcome, and a separate key file.
score:  compares the filled CSV with the key against criteria fixed in wf/constants.py.

    python -m wf.tools.parser_audit export runs/pilotA_reparse/generations.jsonl \
        runs/pilotA_transitions/transitions.jsonl runs/parser_audit.csv
    python -m wf.tools.parser_audit score runs/parser_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from wf import constants as C
from wf.analysis.stats import gen_says_success

# How many rows each kind of reply gets. The hard cases are over-sampled on purpose.
# Exact "Nothing happens." is one string, so it needs only a token presence.
STRATA = {
    "noop_paraphrase": 35,
    "success_paraphrase": 20,
    "unparseable": 20,
    "templated_success": 23,
    "noop": 2,
}
TEMPLATED_SUCCESS = ("state_change", "reveal_with_goal", "reveal_without_goal", "info")
FEWSHOT_MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")


def parser_label(cls: str) -> str:
    said = gen_says_success(cls)
    return "?" if said is None else "S" if said else "F"


def _stratum(cls: str) -> str:
    return "templated_success" if cls in TEMPLATED_SUCCESS else cls


def export(gens_path: Path, transitions_path: Path, out_csv: Path, seed: int = C.RNG_SEED,
           exclude: Path | None = None) -> Path:
    actions = {}
    for line in transitions_path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        actions[t["id"]] = t["action"]
    gens = [json.loads(line) for line in gens_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    picked: list[dict] = []
    seen_text: set[str] = set()
    if exclude:  # rows an earlier audit (or parser development) already looked at
        seen_text |= {json.loads(x)["text"] for x in exclude.read_text(encoding="utf-8").splitlines() if x.strip()}
    for stratum, n in STRATA.items():
        pools = {}
        for model in FEWSHOT_MODELS:
            pools[model] = [g for g in gens if g["model"] == model and _stratum(g["generated_class"]) == stratum]
            rng.shuffle(pools[model])

        def take(model: str, want: int) -> int:
            got = 0
            while got < want and pools[model]:
                g = pools[model].pop()
                if g["generated"] in seen_text:  # distinct texts only
                    continue
                seen_text.add(g["generated"])
                picked.append({**g, "stratum": stratum})
                got += 1
            return got

        # Half from each few-shot model; a model that runs out leaves its share to the other.
        first, second = FEWSHOT_MODELS
        got = take(first, n - n // 2)
        got += take(second, n - got)
        take(first, n - got)
    rng.shuffle(picked)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    key = out_csv.with_name(out_csv.stem + "_key.jsonl")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "action", "prediction", "your_label (S/F/?)", "note"])
        for i, g in enumerate(picked, 1):
            w.writerow([i, actions.get(g["transition_id"], ""), g["generated"], "", ""])
    with key.open("w", encoding="utf-8") as f:
        for i, g in enumerate(picked, 1):
            f.write(json.dumps({"row": i, "model": g["model"], "condition": g["condition"],
                                "transition_id": g["transition_id"], "stratum": g["stratum"],
                                "parser_class": g["generated_class"], "text": g["generated"],
                                "parser_label": parser_label(g["generated_class"])}) + "\n")
    return key


def score(filled_csv: Path) -> dict:
    key_path = filled_csv.with_name(filled_csv.stem + "_key.jsonl")
    key = {k["row"]: k for k in (json.loads(x) for x in key_path.read_text(encoding="utf-8").splitlines() if x.strip())}
    with filled_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    human = {int(r[0]): (r[3] or "").strip().upper()[:1] for r in rows if r and r[0].strip()}
    missing = sorted(r for r, lab in human.items() if lab not in ("S", "F", "?"))
    decided = {r: lab for r, lab in human.items() if lab in ("S", "F")}
    agree = sum(1 for r, lab in decided.items() if key[r]["parser_label"] == lab)
    opposite = [r for r, lab in decided.items() if key[r]["parser_label"] in ("S", "F") and key[r]["parser_label"] != lab]
    # The bug being fixed: failure wording must never be read as a success, nor the reverse.
    paraphrase_opposite = [r for r in opposite if key[r]["parser_class"] == "noop_paraphrase"
                           or (decided[r] == "F" and key[r]["parser_label"] == "S")]
    by_stratum: dict[str, list[int]] = {}
    for r, lab in decided.items():
        by_stratum.setdefault(key[r]["stratum"], [0, 0])
        by_stratum[key[r]["stratum"]][0] += key[r]["parser_label"] == lab
        by_stratum[key[r]["stratum"]][1] += 1
    rate = agree / len(decided) if decided else float("nan")
    result = {
        "n_rows": len(human), "n_unlabelled": len(missing), "n_decided": len(decided),
        "agreement": rate, "min_agreement": C.PARSER_AUDIT_MIN_AGREEMENT,
        "opposite_rows": opposite, "failure_wording_read_as_success": paraphrase_opposite,
        "parser_unsure_rows": [r for r in decided if key[r]["parser_label"] == "?"],
        "by_stratum": {k: {"agree": v[0], "n": v[1]} for k, v in sorted(by_stratum.items())},
    }
    result["passed"] = (not missing and rate == rate and rate >= C.PARSER_AUDIT_MIN_AGREEMENT
                        and not paraphrase_opposite)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("generations", type=Path)
    e.add_argument("transitions", type=Path)
    e.add_argument("out_csv", type=Path)
    e.add_argument("--exclude", type=Path, help="a key file whose rows must not be drawn again")
    s = sub.add_parser("score")
    s.add_argument("filled_csv", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "export":
        key = export(args.generations, args.transitions, args.out_csv, exclude=args.exclude)
        print(f"wrote {args.out_csv} and {key}")
        return 0
    result = score(args.filled_csv)
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
