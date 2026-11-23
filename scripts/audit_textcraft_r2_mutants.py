"""Auditor's own mutants for the TextCraft round-2 sign-off logic (notes/textcraft_r2_audit.md).

Each mutant is applied to a fresh copy of the repo (outside the working tree) and tests/test_textcraft_r2.py is run.
Strict rule: a mutant is killed only if pytest exits 1 and names at least one FAILED test.
Usage: python scripts/audit_textcraft_r2_mutants.py <scratch dir> [jobs]
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = "wf/textcraft/r2_analysis.py"

MUTANTS = [
    ("X1 R3 fires without a signed unmatched gap", F,
     "    disagree = bool(p_sign and rg_sign and p_sign != rg_sign)", "    disagree = bool(p_sign and p_sign != rg_sign)"),
    ("X2 R3 matched gap only needs not to be opposite", F,
     "    fires = bool(testable and disagree and m_sign == rg_sign)", "    fires = bool(testable and disagree and m_sign != -rg_sign)"),
    ("X3 R2 counts Llama-3.1 among the others", F,
     '    others = [t for t in passing if t != "llama3.1-8b"]', "    others = list(passing)"),
    ("X4 role gap on a different row set from Dn and Dc", F,
     '        "role_gap": ("between", *gap_rows(by, roles, lambda i: L(by, i, "aligned") - L(by, i, "swapped"))),',
     '        "role_gap": ("between", *gap_rows(by, [r for r in pops["goal"] + pops["ingredient"] if r["both_task_line"]], lambda i: L(by, i, "aligned") - L(by, i, "swapped"))),'),
    ("X5 margin SD from the ingredient failures", F,
     '    d = [L(by, r["id"], "aligned") - L(by, r["id"], "swapped") for r in pops["main_failures"]]',
     '    d = [L(by, r["id"], "aligned") - L(by, r["id"], "swapped") for r in pops["ingredient"]]'),
    ("X6 TOST takes the smaller one-sided p", F, "    return max(low, high)", "    return min(low, high)"),
    ("X7 Λ and P sign flip by item", F,
     '"n_items": len(values), "n_pairs": len(set(clusters)), "p_differs": sign_flip_p(values, clusters),',
     '"n_items": len(values), "n_pairs": len(set(clusters)), "p_differs": sign_flip_p(values, [str(i) for i in range(len(values))]),'),
    ("X8 equivalence verdict on the unadjusted p", F,
     '    if fam[f"{tag}:{contrast}:equivalent"]["p_holm"] < ALPHA:', '    if fam[f"{tag}:{contrast}:equivalent"]["p_in_family"] < ALPHA:'),
    ("X9 lift difference over class-matched ingredients only", F,
     '    for r in pops["ingredient"]:\n        a = L(by_a, r["id"], "aligned")', '    for r in pops["ingredient_matched"]:\n        a = L(by_a, r["id"], "aligned")'),
    ("X10 H6-best needs strictly better tracking", F,
     '    h6_best = bool(defined) and all(tr[k]["estimate"] <= 0 for k in defined)',
     '    h6_best = bool(defined) and all(tr[k]["estimate"] < 0 for k in defined)'),
    ("X11 NaN ps dropped from the family instead of entering as 1", F,
     "                p[key] = pv if pv == pv else 1.0", "                p[key] = pv"),
    ("X12 R4 difference reported as Qwen minus Llama", F,
     '                                  lift_difference(bys["llama3.1-8b"], bys["qwen3-8b"], pops)',
     '                                  lift_difference(bys["qwen3-8b"], bys["llama3.1-8b"], pops)'),
]


def run(m, scratch: Path, python: str) -> tuple[str, str, str]:
    name, path, good, bad = m
    work = scratch / re.sub(r"\W+", "_", name)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "runs", "runs_mock", "__pycache__", ".pytest_cache"))
    f = work / path
    text = f.read_text(encoding="utf-8")
    if text.count(good) != 1:
        return name, "NOT APPLIED", f"correct text found {text.count(good)} times"
    f.write_text(text.replace(good, bad), encoding="utf-8")
    r = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no", "-rf",
                        "tests/test_textcraft_r2.py"], cwd=work, capture_output=True, text=True, timeout=1800)
    failed = [ln.split(" ")[1] for ln in r.stdout.splitlines() if ln.startswith("FAILED ")]
    shutil.rmtree(work)
    if r.returncode == 1 and failed:
        return name, "KILLED", failed[0]
    return name, "SURVIVED", f"exit {r.returncode}; {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}"


if __name__ == "__main__":
    scratch = Path(sys.argv[1])
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    scratch.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(jobs) as ex:
        for name, status, detail in ex.map(lambda m: run(m, scratch, sys.executable), MUTANTS):
            print(f"{status:9s} {name}: {detail}", flush=True)
