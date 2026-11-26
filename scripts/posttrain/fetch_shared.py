"""Download one checkpoint at its pinned revision into a shared Hugging Face cache and verify every file against the
Hub's record for that revision (sha256 of LFS files, git blob sha1 of the others). Only the files transformers loads
(config and generation config, tokenizer files, safetensors weights and their index); no original/ folder, no .pth
or .bin. The token is HF_TOKEN from the environment and is never printed.

    python scripts/posttrain/fetch_shared.py --repo meta-llama/Llama-3.1-8B \
        --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --cache /storage/ice1/2/1/gmat3/hfcache --out x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ALLOW = ["*.json", "*.safetensors", "tokenizer*"]
IGNORE = ["original/*", "consolidated*", "*.pth", "*.bin", "*.gguf"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(a.repo, revision=a.revision, files_metadata=True)
    if info.sha != a.revision:
        raise SystemExit(f"Hub sha {info.sha} != pinned {a.revision}")
    path = Path(snapshot_download(a.repo, revision=a.revision, cache_dir=a.cache, allow_patterns=ALLOW,
                                  ignore_patterns=IGNORE))
    if path.name != a.revision:
        raise SystemExit(f"snapshot folder {path} is not the pinned revision")
    rows, bad = [], []
    for s in info.siblings:
        f = path / s.rfilename
        if not f.exists():
            continue
        if s.lfs:
            got, want, kind = sha256(f), s.lfs.sha256, "sha256"
        else:
            got, want, kind = git_blob_sha1(f), s.blob_id, "git-blob-sha1"
        rows.append({"file": s.rfilename, "bytes": f.stat().st_size, kind: got, "matches_hub": got == want})
        if got != want:
            bad.append(s.rfilename)
    expected = {s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors")
                and not s.rfilename.startswith("original/")}
    missing = sorted(expected - {r["file"] for r in rows})
    out = {"repo": a.repo, "revision": a.revision, "snapshot": str(path), "files": rows, "mismatched": bad,
           "missing_weights": missing, "total_bytes": sum(r["bytes"] for r in rows)}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{r['file']:40s} {r['bytes']:>12d} {r.get('sha256') or r.get('git-blob-sha1')} "
              f"{'OK' if r['matches_hub'] else 'MISMATCH'}", flush=True)
    print(f"snapshot {path}; {len(rows)} files, {out['total_bytes'] / 1e9:.2f} GB; mismatched {bad}; missing {missing}")
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
