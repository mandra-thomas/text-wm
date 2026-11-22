"""Download exactly the pinned revision of each model into the cache transformers reads (2026-11-20).

    python -m wf.tools.prefetch_models [tag ...] [--check] [--out FILE]

Tags are keys of wf.families.MODELS (default: every wave A model). For each model it prints OK, or the HTTP status
of the failure (403: access not granted yet, as for a gated model still pending; 401: no or a bad token), and goes
on to the next, so one refused model does not stop the others. The exit code is 1 if any model failed.

The token is HF_TOKEN from the environment (pace.env), which huggingface_hub reads itself; it is never printed.
Only the files transformers loads are fetched: no `original/` folder (Llama's Meta-format weights), no
`consolidated.safetensors` (Mistral's second copy of the weights), no .pth or .bin.

--check then loads each downloaded tokenizer offline and runs the families tokenization gate (wf/families.py) on the
golden item (tests/golden/default_prompt.json), so a template or tokenizer problem shows up here, on a CPU, before
a GPU job starts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from wf.constants import REVISIONS

# Wave B (2026-11-22): Gemma 4, Ministral 3 and Qwen3.8 keep their chat template only in chat_template.jinja, and
# Qwen's tokenizer ships merges.txt; without them a downloaded snapshot could not render a prompt.
ALLOW = ["*.json", "*.safetensors", "*.model", "*.model.v3", "*.jinja", "merges.txt"]
IGNORE = ["original/*", "consolidated*", "*.pth", "*.bin", "*.gguf", "*.msgpack", "*.h5"]
GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "default_prompt.json"


def transformers_cache() -> str | None:
    """The folder from_pretrained reads when given no cache_dir: TRANSFORMERS_CACHE if set (as pace/_common.sh
    sets it), else the hub cache. Prefetching anywhere else would leave the scoring job to download again.

    transformers 5 (wave B's serve2, 2026-11-22) no longer reads TRANSFORMERS_CACHE: it reads huggingface_hub's
    HF_HUB_CACHE, so that is the folder for it (pace/family_score_serve2.sbatch points HF_HUB_CACHE at the folder
    transformers 4 reads, so both environments share one cache)."""
    try:
        from transformers.utils import hub

        if hasattr(hub, "TRANSFORMERS_CACHE"):
            return str(hub.TRANSFORMERS_CACHE)
        from huggingface_hub import constants

        return str(constants.HF_HUB_CACHE)
    except Exception:  # transformers missing: same rule from the environment
        return os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HUB_CACHE")


def _status(e: BaseException) -> str:
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    return f"HTTP {code}" if code else f"ERROR {type(e).__name__}"


def _snapshot_download(**kw: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kw)


def prefetch(hf_id: str, revision: str, cache_dir: str | None, download=None) -> dict[str, Any]:
    download = download or _snapshot_download
    try:
        path = download(repo_id=hf_id, revision=revision, cache_dir=cache_dir, allow_patterns=ALLOW,
                        ignore_patterns=IGNORE)
    except Exception as e:  # noqa: BLE001 - every failure is reported, then the next model is tried
        return {"status": _status(e), "error": str(e).splitlines()[0][:300] if str(e) else type(e).__name__}
    if Path(path).name != revision:
        return {"status": "ERROR wrong revision", "path": str(path)}
    files = sorted(p.name for p in Path(path).iterdir())
    return {"status": "OK", "path": str(path), "files": files}


def check_tokenizer(spec: Any, cache_dir: str | None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from wf import families
    from wf.scoring.logodds import LogoddsScorer

    item = json.loads(GOLDEN.read_text(encoding="utf-8"))["item"]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=REVISIONS[spec.hf_id], cache_dir=cache_dir,
                                        local_files_only=True, **spec.tokenizer_kwargs)
    scorer = LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs,
                                         prompt_format=spec.prompt_format)
    ref = families.mistral_common_encoder(spec, cache_dir=cache_dir) if spec.mistral_common_check else None
    rec = families.tokenization_checks(scorer, spec, [item], [], ref)
    return {k: rec[k] for k in ("ok", "n_problems", "problems", "first_token_pairs", "max_prompt_tokens")} | {
        "yes_no_tokens": {w: v["alone"] for w, v in rec["yes_no"].items()}, "tokenizer_class": type(tok).__name__}


def main(argv: list[str] | None = None) -> int:
    from wf import families

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tags", nargs="*", help=f"default: {sorted(families.MODELS)} (wave A); wave B tags: "
                                            f"{sorted(families.WAVE_B)}")
    ap.add_argument("--check", action="store_true", help="run the tokenization gate on the golden item afterwards")
    ap.add_argument("--out", default=None, help="also write the results as JSON here")
    args = ap.parse_args(argv)
    tags = args.tags or list(families.MODELS)
    known = {**families.MODELS, **families.WAVE_B}
    unknown = [t for t in tags if t not in known]
    if unknown:
        print(f"unknown tags {unknown}; known: {sorted(known)}", file=sys.stderr)
        return 2
    cache = transformers_cache()
    print(f"cache: {cache}; token from HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set'}", flush=True)
    results: dict[str, Any] = {}
    for tag in tags:
        spec = known[tag]
        rev = REVISIONS[spec.hf_id]
        r = prefetch(spec.hf_id, rev, cache)
        if r["status"] == "OK" and args.check:
            try:
                r["tokenizer_check"] = check_tokenizer(spec, cache)
                if not r["tokenizer_check"]["ok"]:
                    r["status"] = "OK (download) but TOKENIZER CHECK FAILED"
            except Exception as e:  # noqa: BLE001
                r["status"] = f"OK (download) but tokenizer check crashed: {type(e).__name__}: {str(e)[:200]}"
        results[tag] = {"hf_id": spec.hf_id, "revision": rev, **r}
        print(f"{tag:18s} {spec.hf_id:40s} {rev[:12]}  {r['status']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["status"] == "OK" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
