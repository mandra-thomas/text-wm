"""Environment check, part 1 (wave B, 2026-11-22; notes/families_waveb_design.md section 8): are the token ids the
same in the study's environment (transformers 4.51.3) and in serve2 (transformers 5.23)?

    python -m wf.tools.envcheck_tokens dump TAG --out FILE [--cache DIR] [--sources ...]
    python -m wf.tools.envcheck_tokens compare LEFT RIGHT --out FILE

`dump` encodes, through the scorer's own encoding (LogoddsScorer._encode and _encode_candidate, with the model's
template settings), every prompt a families scoring run of the model covers (all primary items and determined
successes, every arm of sizes.size_jobs), both candidates of every item, and "Yes" and "No", and writes each id
list's length and SHA-256. It loads the pinned tokenizer offline, as the scoring run of that environment does: in
transformers 5 with the environment check's tokenizer settings (mistral_format=False), in 4.x as the earlier runs
did. `compare` says whether two dumps hold the same keys with the same ids; it exits 1 if not.

`dump --rescore` (2026-11-23, wave B section 8 amendment): the same for every older comparison model of the serve2
rescoring (wf.families.RESCORE_MODELS), with the rescoring's tokenizer settings (rescore_spec) in transformers 5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from wf.constants import REVISIONS


def _digest(ids: list[int]) -> list[Any]:
    return [len(ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()]


def dump(tag: str, primary: list[dict], successes: list[dict], tok: Any, transformers_version: str) -> dict[str, Any]:
    from wf import families, sizes
    from wf.scoring.logodds import LogoddsScorer

    spec = families.spec_for(tag)
    scorer = LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=tag, template_kwargs=spec.template_kwargs)
    rec: dict[str, list[Any]] = {}
    for t, arm in sizes.size_jobs(primary, successes):
        rec[f"{t['id']}|{arm}|prompt"] = _digest(families._ids(scorer._encode(families.family_messages(spec, t, arm))))
    for t in primary + successes:
        for name in ("o_plus", "o_minus"):
            rec[f"{t['id']}|{name}"] = _digest(families._ids(scorer._encode_candidate(t["candidates"][name])))
    for word in families.YES_NO:
        rec[f"answer|{word}"] = _digest(families._ids(scorer._encode_candidate(word)))
    return {"model": tag, "hf_id": spec.hf_id, "revision": REVISIONS[spec.hf_id], "transformers": transformers_version,
            "tokenizer_class": type(tok).__name__, "n": len(rec), "records": rec}


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = left["records"], right["records"]
    only = sorted(set(a) ^ set(b))
    diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {"ok": bool(a) and not only and not diff and left["model"] == right["model"]
            and left["revision"] == right["revision"],
            "model": left["model"], "n_compared": len(set(a) & set(b)), "n_different": len(diff),
            "n_only_one_side": len(only), "examples": (diff + only)[:20],
            "left": {k: left.get(k) for k in ("transformers", "tokenizer_class", "revision", "n")},
            "right": {k: right.get(k) for k in ("transformers", "tokenizer_class", "revision", "n")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("tag")
    d.add_argument("--out", required=True)
    d.add_argument("--cache", default=None)
    d.add_argument("--sources", nargs="*", default=None)
    d.add_argument("--rescore", action="store_true",
                   help="an older comparison model of the serve2 rescoring (wave B section 8 amendment): any tag of "
                        "wf.families.RESCORE_MODELS, with the rescoring's tokenizer settings in transformers 5")
    c = sub.add_parser("compare")
    c.add_argument("left")
    c.add_argument("right")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        res = compare(json.loads(Path(args.left).read_text()), json.loads(Path(args.right).read_text()))
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"{res['model']}: {'identical' if res['ok'] else 'DIFFERENT'} ({res['n_compared']} compared, "
              f"{res['n_different']} different, {res['n_only_one_side']} on one side only)")
        return 0 if res["ok"] else 1
    import transformers

    from wf import families, rq1x
    from wf.rq2 import SOURCES, load_items
    from wf.scoring.logodds import transformers_major

    spec = families.spec_for(args.tag)
    allowed = families.RESCORE_MODELS if args.rescore else families.ENVCHECK_MODELS
    if args.tag not in allowed:
        print(f"{'the serve2 rescoring' if args.rescore else 'the environment check'} covers {allowed}", file=sys.stderr)
        return 2
    serve2 = (families.rescore_spec if args.rescore else families.envcheck_spec)(args.tag)
    extra = serve2.tokenizer_kwargs if transformers_major(transformers) >= 5 else {}
    tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=REVISIONS[spec.hf_id], trust_remote_code=True,
                                                     cache_dir=args.cache, local_files_only=True, **extra)
    sets = rq1x.rq1x_sets(load_items(tuple(args.sources or SOURCES)))
    res = dump(args.tag, sets["primary"], sets["successes"], tok, transformers.__version__)
    Path(args.out).write_text(json.dumps(res) + "\n", encoding="utf-8")
    print(f"{args.tag}: {res['n']} id lists with transformers {res['transformers']} ({res['tokenizer_class']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
