"""Compare a rerun of rq2_analysis with the committed headline metrics.

    python repro/compare_rq2_metrics.py runs/rq2_analysis/metrics.json notes/rq2_confirmatory_metrics.json

Every key and value must match, except the `provenance` block, which a rerun adds (it records where and when the
analysis ran, so it always differs). Strings, integers and booleans must be equal; floats must agree to a relative
difference of 1e-12, which allows last-digit noise from a different numpy, BLAS or CPU but no real change.
Prints the headline numbers, the count of floats that are not bit-identical and the largest relative difference,
and exits 0 on a match and 1 otherwise.
"""
from __future__ import annotations

import json
import math
import sys

REL_TOL = 1e-12


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if path == "" and k == "provenance":
                continue
            if k not in a or k not in b:
                out["problems"].append(f"{path}/{k}: only in the {'expected' if k in b else 'rerun'} file")
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["problems"].append(f"{path}: list lengths {len(a)} and {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        out["floats"] += 1
        if a == b or (math.isnan(a) and math.isnan(b)):
            return
        rel = abs(a - b) / max(abs(a), abs(b))
        out["not_identical"] += 1
        out["max_rel"] = max(out["max_rel"], rel)
        out["examples"].append(f"{path}: {a!r} vs {b!r} (relative {rel:.1e})")
        if rel > REL_TOL:
            out["problems"].append(f"{path}: {a!r} vs {b!r} (relative difference {rel:.1e})")
    elif type(a) is not type(b) or a != b:
        out["problems"].append(f"{path}: {a!r} vs {b!r}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    rerun, expected = (json.load(open(p, encoding="utf-8")) for p in argv)
    out = {"problems": [], "floats": 0, "not_identical": 0, "max_rel": 0.0, "examples": []}
    walk(rerun, expected, "", out)
    for m, v in rerun.get("per_model", {}).items():
        verdict = rerun.get("verdicts", {}).get("per_model", {}).get(m, {}).get("verdict")
        print(f"{m}: delta_L = {v['delta_L_nats']:.3f} nats, 95% CI {v['delta_L_ci'][0]:.3f} to "
              f"{v['delta_L_ci'][1]:.3f}, verdict {verdict}")
    print(f"items: {rerun.get('items')}")
    print(f"floats compared: {out['floats']}; not bit-identical: {out['not_identical']}; "
          f"largest relative difference: {out['max_rel']:.1e}")
    for e in out["examples"][:5]:
        print(f"  {e}")
    if out["problems"]:
        print("MISMATCH:")
        for p in out["problems"][:20]:
            print(f"  {p}")
        return 1
    print("MATCH: the rerun reproduces notes/rq2_confirmatory_metrics.json (provenance block excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
