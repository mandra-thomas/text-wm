#!/usr/bin/env python3
"""Run chosen mutants one at a time through scripts/mutation_check.py (its strict kill rule), and append each result
to a JSONL checkpoint as it finishes, so a stopped run resumes where it was (added 2026-11-24, wave B review 2).

    python scripts/mutation_batch.py OUT.jsonl SHARD/N "name prefix or exact name" [...]

Each argument after the shard selects every mutant whose name starts with it (an exact name selects one). A mutant
already in OUT.jsonl is skipped. Each mutant runs alone (MUTANTS_MATCHING set to its exact name, which must match no
other mutant's name as a substring, else it is recorded as "ambiguous" and not run).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mutation_check  # noqa: E402

LINE = re.compile(r"^(killed|SURVIVED|survived|NOT APPLIED|broken|timeout|TIMEOUT|BROKEN)\s+(.+?)(?:\s+\[by (.+)\])?$")


def main() -> int:
    out, shard = Path(sys.argv[1]), sys.argv[2]
    k, n = map(int, shard.split("/"))
    prefixes = sys.argv[3:]
    names = [m[0] for m in mutation_check.MUTANTS]
    chosen = [x for x in names if any(x.startswith(p) for p in prefixes)][k::n]
    done = set()
    if out.exists():
        done = {json.loads(line)["name"] for line in out.read_text().splitlines() if line.strip()}
    for name in chosen:
        if name in done:
            continue
        others = [x for x in names if x != name and name in x]
        t0 = time.time()
        if others:
            rec = {"name": name, "outcome": "ambiguous", "others": others}
        else:
            env = {**os.environ, "MUTANTS_MATCHING": name}
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "mutation_check.py")], cwd=ROOT, env=env,
                               capture_output=True, text=True)
            outcome, killer = "unknown", None
            for line in r.stdout.splitlines():
                m = LINE.match(line.strip())
                if m and m.group(2).strip() == name:
                    outcome, killer = m.group(1).lower(), m.group(3)
            rec = {"name": name, "outcome": outcome, "killer": killer, "exit": r.returncode,
                   "tail": r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else r.stderr.strip().splitlines()[-3:]}
        rec["seconds"] = round(time.time() - t0)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
