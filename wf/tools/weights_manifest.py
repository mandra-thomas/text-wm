"""Manifests of the files a pinned model snapshot must hold, and the check of a download against them (wave B review 1,
M3; 2026-11-22, before any wave B weight was downloaded).

    python -m wf.tools.weights_manifest build TAG [TAG ...]   # Hub metadata only; writes wf/config/weights_manifests/

A manifest lists exactly the files the prefetch tool's patterns select (wf.tools.prefetch_models ALLOW and IGNORE,
matched as huggingface_hub matches them) at the pinned revision: each file's name, size and, for a file stored in
Git LFS (the weights, large tokenizer files), the LFS SHA-256 of its content; for a small file kept in Git itself
(configs, templates) the Git blob SHA-1, which is also a hash of its content. It is built from
`HfApi().model_info(id, revision=sha, files_metadata=True)` with no token (none of the models is gated), and
cross-checked against `snapshot_download(dry_run=True)` under the same patterns: same names, same sizes.

After a node-local download (pace/family_score_serve2.sbatch, WF_WEIGHTS=local), `verify_snapshot` hashes every file
and compares it with the manifest; any mismatch, missing file or extra file fails the job before a model is loaded,
and the hashes are saved in the shard's record (wf/families.py, weights_check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from wf.constants import REVISIONS

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "config" / "weights_manifests"
DISK_MARGIN = 0.10  # review 1, S2: free space must exceed the manifest's total by this share before a download
SHARED_STORAGE = "/storage"  # review 1, S1: node-local weights must never land on the shared file system
CHUNK = 16 * 1024 * 1024


def manifest_path(hf_id: str) -> Path:
    return MANIFEST_DIR / (hf_id.replace("/", "__") + ".json")


def load_manifest(hf_id: str, revision: str, folder: Path | None = None) -> dict[str, Any]:
    """The committed manifest of `hf_id`, which must be for exactly `revision`."""
    path = (folder / (hf_id.replace("/", "__") + ".json")) if folder else manifest_path(hf_id)
    if not path.is_file():
        raise FileNotFoundError(f"no weights manifest for {hf_id}: {path}")
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("hf_id") != hf_id or m.get("revision") != revision:
        raise ValueError(f"{path} is for {m.get('hf_id')}@{m.get('revision')}, not {hf_id}@{revision}")
    if not m.get("files"):
        raise ValueError(f"{path} lists no files")
    return m


def refuse_shared_storage(path: str | os.PathLike) -> str:
    """The resolved folder, or SystemExit if it is on the shared file system (/storage), where node-local weights
    would use the scratch quota and outlive the job."""
    real = os.path.realpath(os.fspath(path))
    if real == SHARED_STORAGE or real.startswith(SHARED_STORAGE + "/"):
        raise SystemExit(f"node-local weights asked for, but {path} resolves to {real}, on the shared file system")
    return real


def check_disk(folder: str | os.PathLike, manifest: dict[str, Any],
               usage: Callable[[str], Any] = shutil.disk_usage) -> dict[str, Any]:
    """Free space in `folder` against the manifest's total size plus DISK_MARGIN (review 1, S2)."""
    need = math.ceil(manifest["total_bytes"] * (1 + DISK_MARGIN))
    free = usage(os.fspath(folder)).free
    return {"ok": free >= need, "free_bytes": free, "needed_bytes": need, "manifest_total_bytes": manifest["total_bytes"]}


def file_digest(path: str | os.PathLike, kind: str) -> str:
    """The content hash a manifest records: "sha256" (Git LFS files) or "git_sha1" (the Git blob id of a small file)."""
    if kind not in ("sha256", "git_sha1"):
        raise ValueError(f"unknown hash kind {kind!r}")
    size = os.path.getsize(path)
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _kind(entry: dict[str, Any]) -> str:
    return "sha256" if entry.get("sha256") else "git_sha1"


def snapshot_files(folder: str | os.PathLike) -> dict[str, str]:
    """{path relative to the snapshot folder: the real file it resolves to} for every file under the folder (the Hub
    cache keeps symlinks to blobs there)."""
    root = Path(folder)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = os.path.realpath(p)
    return out


def verify_snapshot(folder: str | os.PathLike, manifest: dict[str, Any],
                    digest: Callable[[str, str], str] = file_digest) -> dict[str, Any]:
    """Hash every file of a downloaded snapshot and compare with the manifest. Fails on a missing file, an extra
    file, a size that differs, or a hash that differs; the record holds every file's size and hash as found."""
    want = {f["name"]: f for f in manifest["files"]}
    have = snapshot_files(folder)
    problems = [f"missing: {n}" for n in sorted(set(want) - set(have))]
    problems += [f"extra: {n}" for n in sorted(set(have) - set(want))]
    files: dict[str, Any] = {}
    for name in sorted(set(want) & set(have)):
        entry, real = want[name], have[name]
        kind = _kind(entry)
        size = os.path.getsize(real)
        rec = {"size": size, kind: None}
        if size != entry["size"]:
            problems.append(f"size: {name} is {size} bytes, the manifest says {entry['size']}")
        else:
            rec[kind] = digest(real, kind)
            if rec[kind] != entry[kind]:
                problems.append(f"hash: {name} {kind} {rec[kind]} differs from the manifest's {entry[kind]}")
        files[name] = rec
    return {"ok": not problems, "problems": problems[:50], "n_problems": len(problems), "n_files": len(files),
            "total_bytes": sum(f["size"] for f in files.values()), "files": files,
            "manifest": {"hf_id": manifest["hf_id"], "revision": manifest["revision"], "n_files": len(want),
                         "total_bytes": manifest["total_bytes"]}}


# ---------------- building a manifest (Hub metadata only) ----------------


def select(names: list[str]) -> list[str]:
    """The files the prefetch tool's patterns select, matched exactly as snapshot_download matches them."""
    from huggingface_hub.utils import filter_repo_objects

    from wf.tools.prefetch_models import ALLOW, IGNORE

    return sorted(filter_repo_objects(names, allow_patterns=ALLOW, ignore_patterns=IGNORE))


def build_manifest(hf_id: str, revision: str, api: Any = None, dry_run: Callable[..., Any] | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    from wf.tools.prefetch_models import ALLOW, IGNORE

    api = api or HfApi(token=False)
    info = api.model_info(hf_id, revision=revision, files_metadata=True, token=False)
    if info.sha != revision:
        raise ValueError(f"{hf_id}: the Hub answered for {info.sha}, not {revision}")
    if getattr(info, "gated", False):
        raise SystemExit(f"{hf_id} is gated; stopping rather than using a token")
    siblings = {s.rfilename: s for s in info.siblings}
    files = []
    for name in select(list(siblings)):
        s = siblings[name]
        lfs = getattr(s, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if s.size is None:
            raise ValueError(f"{hf_id}: no size for {name}")
        entry = {"name": name, "size": int(s.size)}
        if sha:
            entry["sha256"] = sha
        else:
            if not getattr(s, "blob_id", None):
                raise ValueError(f"{hf_id}: {name} has neither an LFS SHA-256 nor a Git blob id")
            entry["git_sha1"] = s.blob_id
        files.append(entry)
    manifest: dict[str, Any] = {"hf_id": hf_id, "revision": revision, "allow_patterns": ALLOW, "ignore_patterns": IGNORE,
                                "files": files, "n_files": len(files), "total_bytes": sum(f["size"] for f in files),
                                "source": "HfApi.model_info(files_metadata=True), no token"}
    dry_run = dry_run or snapshot_download
    try:
        planned = dry_run(repo_id=hf_id, revision=revision, allow_patterns=ALLOW, ignore_patterns=IGNORE, dry_run=True,
                          token=False)
    except TypeError as e:  # a huggingface_hub without dry_run
        manifest["dry_run_cross_check"] = {"available": False, "error": str(e)[:200]}
        return manifest
    got = {p.filename: int(p.file_size) for p in planned}
    mine = {f["name"]: f["size"] for f in files}
    manifest["dry_run_cross_check"] = {"available": True, "same_files_and_sizes": got == mine,
                                       "only_in_dry_run": sorted(set(got) - set(mine)),
                                       "only_in_metadata": sorted(set(mine) - set(got)),
                                       "size_differs": sorted(n for n in set(got) & set(mine) if got[n] != mine[n])}
    if got != mine:
        raise ValueError(f"{hf_id}: snapshot_download(dry_run=True) plans other files: {manifest['dry_run_cross_check']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    from wf import families

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("tags", nargs="+")
    b.add_argument("--out-dir", default=str(MANIFEST_DIR))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for tag in args.tags:
        spec = families.spec_for(tag)
        if spec.hf_id in done:
            continue
        done.add(spec.hf_id)
        m = build_manifest(spec.hf_id, REVISIONS[spec.hf_id])
        (out / (spec.hf_id.replace("/", "__") + ".json")).write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
        print(f"{spec.hf_id:46s} {m['n_files']:3d} files {m['total_bytes'] / 1e9:7.2f} GB  dry run agrees: "
              f"{m['dry_run_cross_check'].get('same_files_and_sizes')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
