"""Blind human audit of `parse_prediction` on model-written predictions (2026-11-17).

export: writes a CSV a person labels N (no effect), R (requested success), P (process only, result not
        stated), D (a different effect: another action or object) or ? (commits to nothing), without the
        parser's answer or the true outcome, and a separate key file.
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
import hashlib

from wf.analysis.stats import gen_says_success
from wf.scoring import parser as _parser
from wf.scoring.parser import parse_prediction


def parser_version() -> str:
    """SHA-256 of the parser's source, recorded with every audit row.

    Every byte counts, comments included, so parser.py is not edited even for its header (2026-11-20): the last
    audited version is 3c25d40cc71529e2.
    """
    return hashlib.sha256(Path(_parser.__file__).read_bytes()).hexdigest()[:16]

# How many rows each kind of reply gets (third audit, 2026-11-17). The hard cases are over-sampled on purpose:
# clean/heat/cool actions and Llama's walk-there replies are the ones the categories hinge on.
STRATA = {
    "noop_paraphrase": 40,
    "success_clean_heat_cool": 25,
    "success_other": 20,
    "process_only": 30,
    "different_effect_llama": 25,
    "different_effect_other": 10,
    "unparseable": 25,
    "noop": 1,  # "Nothing happens." is one string, so one row
}
SUCCESS = ("state_change", "reveal_with_goal", "reveal_without_goal", "info", "success_paraphrase")
FEWSHOT_MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
LABELS = ("N", "R", "P", "D", "?")


def parser_label(cls: str) -> str:
    if cls in ("noop", "noop_paraphrase"):
        return "N"
    return {"process_only": "P", "different_effect": "D", "unparseable": "?"}.get(cls, "R")


def _stratum_of(g: dict, action: str) -> str:
    cls = g["generated_class"]
    if cls in SUCCESS:
        return "success_clean_heat_cool" if action.split(" ", 1)[0] in ("clean", "heat", "cool") else "success_other"
    if cls == "different_effect":
        return "different_effect_llama" if g["model"].startswith("llama") else "different_effect_other"
    return cls


def export(gens_path: Path, transitions_path: Path, out_csv: Path, seed: int = C.RNG_SEED,
           exclude: Path | list[Path] | None = None) -> Path:
    actions, goals = {}, {}
    for line in transitions_path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        actions[t["id"]], goals[t["id"]] = t["action"], t.get("goal_object")
    # Re-read every reply with the parser as it is now, so the audit grades the version being frozen.
    gens = [json.loads(line) for line in gens_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gens = [{**g, "generated_class": parse_prediction(g["generated"], goals.get(g["transition_id"]),
                                                      actions.get(g["transition_id"]))} for g in gens]
    version = parser_version()
    rng = random.Random(seed)
    picked: list[dict] = []
    seen_text: set[str] = set()
    for path in ([exclude] if isinstance(exclude, Path) else exclude or []):  # rows already looked at
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if any("text" not in r for r in rows):
            raise SystemExit(f"{path}: rows without their text cannot be excluded; pass a file of texts")
        seen_text |= {r["text"] for r in rows}
    for stratum, n in STRATA.items():
        pools = {}
        for model in FEWSHOT_MODELS:
            pools[model] = [g for g in gens if g["model"] == model
                            and _stratum_of(g, actions.get(g["transition_id"], "")) == stratum]
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
        w.writerow(["row", "action", "prediction", "your_label (N/R/P/D/?)", "note"])
        for i, g in enumerate(picked, 1):
            w.writerow([i, actions.get(g["transition_id"], ""), g["generated"], "", ""])
    with key.open("w", encoding="utf-8") as f:
        for i, g in enumerate(picked, 1):
            f.write(json.dumps({"row": i, "model": g["model"], "condition": g["condition"],
                                "transition_id": g["transition_id"], "stratum": g["stratum"],
                                "parser_class": g["generated_class"], "text": g["generated"],
                                "parser_version": version,
                                "parser_label": parser_label(g["generated_class"])}) + "\n")
    return key


def score(filled_csv: Path) -> dict:
    """Pre-specified (notes/deviations.md, third audit): on rows the reader does not mark "?", the parser must
    match the reader on no effect against effect for at least PARSER_AUDIT_MIN_AGREEMENT, with its own "?"
    counted as a miss, and no row may be read in the opposite direction. The four-way agreement is reported."""
    key_path = filled_csv.with_name(filled_csv.stem + "_key.jsonl")
    key = {k["row"]: k for k in (json.loads(x) for x in key_path.read_text(encoding="utf-8").splitlines() if x.strip())}
    with filled_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    human = {int(r[0]): (r[3] if len(r) > 3 else "").strip().upper()[:1] for r in rows if r and r[0].strip()}
    human.update({r: "" for r in key if r not in human})  # a row deleted from the file is a missing label
    missing = sorted(r for r, lab in human.items() if lab not in LABELS)
    versions = {k.get("parser_version") for k in key.values()}
    if versions != {parser_version()}:
        raise SystemExit(f"the parser changed after this sample was drawn ({versions} vs {parser_version()}); "
                         "draw a new sample")
    decided = {r: lab for r, lab in human.items() if lab in ("N", "R", "P", "D")}
    effect = lambda lab: lab != "N"
    agree = sum(1 for r, lab in decided.items() if key[r]["parser_label"] != "?"
                and effect(key[r]["parser_label"]) == effect(lab))
    opposite = [r for r, lab in decided.items() if key[r]["parser_label"] != "?"
                and effect(key[r]["parser_label"]) != effect(lab)]
    four_way = sum(1 for r, lab in decided.items() if key[r]["parser_label"] == lab)
    by_stratum: dict[str, list[int]] = {}
    for r, lab in decided.items():
        cell = by_stratum.setdefault(key[r]["stratum"], [0, 0])
        cell[0] += key[r]["parser_label"] != "?" and effect(key[r]["parser_label"]) == effect(lab)
        cell[1] += 1
    rate = agree / len(decided) if decided else float("nan")
    result = {
        "n_rows": len(human), "n_unlabelled": len(missing), "n_reader_unsure": sum(lab == "?" for lab in human.values()),
        "n_decided": len(decided), "agreement_effect_vs_none": rate, "min_agreement": C.PARSER_AUDIT_MIN_AGREEMENT,
        "opposite_rows": opposite, "parser_unsure_rows": [r for r in decided if key[r]["parser_label"] == "?"],
        "agreement_four_way": four_way / len(decided) if decided else float("nan"),
        "by_stratum": {k: {"agree": v[0], "n": v[1]} for k, v in sorted(by_stratum.items())},
    }
    result["passed"] = not missing and rate == rate and rate >= C.PARSER_AUDIT_MIN_AGREEMENT and not opposite
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("generations", type=Path)
    e.add_argument("transitions", type=Path)
    e.add_argument("out_csv", type=Path)
    e.add_argument("--exclude", type=Path, nargs="*", default=[], help="files of rows (with text) not to draw again")
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
