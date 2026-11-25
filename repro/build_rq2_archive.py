"""Build the public input archive for the headline result (RQ2, `python -m wf.runs rq2_analysis`).

Run by the owner, once, from the private copy of the confirmatory checkout's `runs/` folder (fix11 on ICE):

    python repro/build_rq2_archive.py --source <checkout>/runs --out <empty folder> [--forbid <username> ...]

It writes `rq2-headline-inputs.tar.gz` and `SHA256SUMS` into `--out`. The archive unpacks into a repository's
`runs/` folder and holds exactly what `rq2_analysis` reads, plus a manifest and the upstream licences:

- `rq12_transitions_s{0,1,2}/transitions.jsonl`, the 12,987 collected and replay-verified items (the analysis
  selects its 4,666 from them), with one field removed: `hidden_state`, the starting layout copied verbatim from
  Word2World's data download, which declares no licence. The analysis never reads it; its SHA-256 per item
  (`hidden_state_sha256`) is kept. Each run's `config.json`, `git_sha` and `STATUS.md` are kept too.
- `rq2_score/shard_0k_of_04/`: `scoring.jsonl` (per-item scores of both models), `generations.jsonl` (both
  models' short written predictions, read for the provisional written shift), `config.json`, `git_sha` and
  `STATUS.md`. The shards' `inputs.json` is left out: the analysis does not read it, and it names cluster paths.
- `rq2_release/`: `MANIFEST.json` (what each file is, its SHA-256 here and in the original run folder),
  `README.md`, `DATA_LICENSES.md` and the upstream licence texts.

The archive is deterministic (sorted members, fixed times and owners, gzip without a timestamp), so rebuilding
it from the same source gives the same bytes. The build refuses to write if any member contains a local or
cluster path, an email address, something shaped like an access token, or a string given with --forbid.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = "rq2-headline-inputs.tar.gz"
SOURCES = ("rq12_transitions_s0", "rq12_transitions_s1", "rq12_transitions_s2")
SHARDS = tuple(f"rq2_score/shard_{i:02d}_of_04" for i in range(4))
SOURCE_FILES = ("transitions.jsonl", "config.json", "git_sha", "STATUS.md")
SHARD_FILES = ("scoring.jsonl", "generations.jsonl", "config.json", "git_sha", "STATUS.md")
DROPPED_FIELDS = ("hidden_state",)
MTIME = 1795564800  # 2026-11-25 00:00:00 UTC, the release date, for every member
LICENCE_FILES = ("ALFWorld-LICENSE", "ALFRED-LICENSE", "TextWorld-LICENSE.txt")

# What must never be published: generic patterns here, plus any personal ones given with --forbid at build time
# (usernames are not written into this public file).
FORBIDDEN = [
    re.compile(rb"/Users/"),
    re.compile(rb"/storage/"),
    re.compile(rb"/home/"),
    re.compile(rb"pace\.gatech\.edu"),
    re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email addresses
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}"),  # Hugging Face tokens
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),  # OpenAI and Anthropic style keys
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),  # AWS access keys
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_fields(raw: bytes) -> tuple[bytes, int]:
    """The JSONL file with DROPPED_FIELDS removed from every record, key order and values unchanged."""
    out, n = [], 0
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for f in DROPPED_FIELDS:
            rec.pop(f, None)
        out.append(json.dumps(rec, ensure_ascii=False))
        n += 1
    return ("\n".join(out) + "\n").encode("utf-8"), n


def scan(name: str, data: bytes, patterns: list[re.Pattern]) -> list[str]:
    return [f"{name}: matches {p.pattern!r} ({m.group(0)[:40]!r})" for p in patterns for m in [p.search(data)] if m]


def build(source: Path, out: Path, forbid: tuple[str, ...] = ()) -> dict:
    members: dict[str, bytes] = {}
    files: dict[str, dict] = {}
    for run in SOURCES:
        for f in SOURCE_FILES:
            raw = (source / run / f).read_bytes()
            rel = f"{run}/{f}"
            entry = {"original_sha256": sha256(raw)}
            if f == "transitions.jsonl":
                raw, n = strip_fields(raw)
                entry.update(records=n, fields_removed=list(DROPPED_FIELDS))
            members[rel] = raw
            files[rel] = entry
    for shard in SHARDS:
        for f in SHARD_FILES:
            raw = (source / shard / f).read_bytes()
            rel = f"{shard}/{f}"
            members[rel] = raw
            files[rel] = {"original_sha256": sha256(raw)}
            if f.endswith(".jsonl"):
                files[rel]["records"] = sum(1 for ln in raw.splitlines() if ln.strip())
    verification = {}
    for run in SOURCES:
        parts = sorted((source / run).glob("VERIFY_part_*_of_*.json"))
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in parts]
        verification[run] = {
            "parts": len(recs),
            "all_passed_in_full": bool(recs) and all(r.get("passed") and r.get("full") for r in recs),
            "verified_file_sha256": sorted({r.get("transitions_sha256") for r in recs}),
            "items_replayed_on_server": sum(r.get("server", {}).get("replayed", 0) for r in recs),
        }
        if verification[run]["verified_file_sha256"] != [files[f"{run}/transitions.jsonl"]["original_sha256"]]:
            raise SystemExit(f"{run}: the verification records are not for this transitions.jsonl")
    for rel, data in members.items():
        files[rel]["sha256"] = sha256(data)
        files[rel]["bytes"] = len(data)
    manifest = {
        "archive": ARCHIVE,
        "what": "Inputs of the preregistered RQ2 analysis (`python -m wf.runs rq2_analysis`): the collected items "
                "and both co-primary models' per-item scores and written predictions.",
        "unpack_into": "runs/ (the repository's runs folder)",
        "expected_result": "notes/rq2_confirmatory_metrics.json in the repository",
        "collection_commit": "f8d20c1469075e16212dd37a3973243eded0cf08",
        "scoring_and_analysis_commit": "ddeabc2072d2db26c6dd0504873294b7f5826b49",
        "item_verification": verification,
        "fields_removed": {
            "hidden_state": "Word2World's starting layout for the game, copied verbatim from its data download "
                            "(no licence declared). Not read by rq2_analysis; hidden_state_sha256 is kept.",
        },
        "files_left_out": {
            "rq2_score/shard_*/inputs.json": "names cluster paths; not read by the analysis (its hashes are the "
                                             "original_sha256 values above)",
            "rq12_transitions_s*/VERIFY_part_*_of_4.json": "summarised under item_verification",
            "rq12_transitions_s*/games_done.jsonl, REPORT.md, inputs.json": "not read by the analysis",
        },
        "files": files,
    }
    docs = REPO / "repro" / "archive_docs"
    members["rq2_release/MANIFEST.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    members["rq2_release/README.md"] = (docs / "README.md").read_bytes()
    members["rq2_release/DATA_LICENSES.md"] = (REPO / "DATA_LICENSES.md").read_bytes()
    for name in LICENCE_FILES:
        members[f"rq2_release/licenses/{name}"] = (REPO / "licenses" / name).read_bytes()

    patterns = FORBIDDEN + [re.compile(re.escape(s.encode()), re.I) for s in forbid]
    problems = [p for rel, data in sorted(members.items()) for p in scan(rel, data, patterns)]
    if problems:
        raise SystemExit("refusing to build; forbidden content:\n" + "\n".join(problems))

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        dirs = sorted({str(Path(rel).parents[i]) for rel in members for i in range(len(Path(rel).parents) - 1)})
        for d in dirs:
            info = tarfile.TarInfo(d)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, MTIME
            tar.addfile(info)
        for rel in sorted(members):
            info = tarfile.TarInfo(rel)
            info.size, info.mode, info.mtime = len(members[rel]), 0o644, MTIME
            tar.addfile(info, io.BytesIO(members[rel]))
    gz = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gz, compresslevel=9, mtime=0) as g:
        g.write(buf.getvalue())
    out.mkdir(parents=True, exist_ok=True)
    (out / ARCHIVE).write_bytes(gz.getvalue())
    digest = sha256(gz.getvalue())
    (out / "SHA256SUMS").write_text(f"{digest}  {ARCHIVE}\n", encoding="utf-8")
    return {"archive": str(out / ARCHIVE), "bytes": len(gz.getvalue()), "unpacked_bytes": len(buf.getvalue()),
            "sha256": digest, "members": len(members)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="the confirmatory checkout's runs/ folder")
    ap.add_argument("--out", type=Path, required=True, help="folder to write the archive and SHA256SUMS into")
    ap.add_argument("--forbid", action="append", default=[],
                    help="a further string that must not appear in any member, case-insensitive (repeatable); "
                         "use it for usernames and names")
    args = ap.parse_args(argv)
    print(json.dumps(build(args.source, args.out, tuple(args.forbid)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
