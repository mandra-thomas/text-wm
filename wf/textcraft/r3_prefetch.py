"""Node-local weights for round 3 (route A, decision D5; notes/textcraft_r3_design.md section 9).

    python -m wf.textcraft.r3_prefetch <tag> --cache <node-local folder> --out <record.json>

wave B's tool (wf/tools/prefetch_models.py) knows only the families' tags, and wave B's files are frozen, so this is
its route-A path for the round-3 tags, built from its own functions without changing them: the folder must not be
on the shared file system; the model's committed manifest (wf/config/weights_manifests) must exist; the free space
must exceed the manifest's total plus 10%; after the download every file is hashed against the manifest, and any
missing, extra or different file fails. The record has wave B's format, so families.weights_check reads it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from wf.constants import REVISIONS
from wf.textcraft import r3 as R3
from wf.tools import prefetch_models as PF
from wf.tools import weights_manifest as wm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag", choices=sorted(R3.MODELS))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = R3.MODELS[args.tag]
    wm.refuse_shared_storage(args.cache)
    os.makedirs(args.cache, exist_ok=True)
    rev = REVISIONS[spec.hf_id]
    try:
        manifest = wm.load_manifest(spec.hf_id, rev)
    except (FileNotFoundError, ValueError) as e:
        print(f"no usable manifest: {e}", file=sys.stderr)
        return 2
    r = PF.prefetch(spec.hf_id, rev, args.cache, manifest=manifest)
    Path(args.out).write_text(json.dumps({args.tag: {"hf_id": spec.hf_id, "revision": rev, **r}}, indent=1) + "\n",
                              encoding="utf-8")
    print(f"{args.tag} {spec.hf_id} {rev[:12]} {r['status']}", flush=True)
    return 0 if r["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
