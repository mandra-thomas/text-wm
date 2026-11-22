"""The families tokenization gate on every item, with tokenizers only, on a CPU (wave B, 2026-11-22).

    python -m wf.tools.family_gate_cpu TAG [TAG ...] --out FILE [--golden FILE] [--cache DIR]

For each tag (wave A or wave B, wf/families.py) it loads the pinned tokenizer offline, with the spec's tokenizer
settings, and runs `families.tokenization_checks` exactly as a scoring shard does before its first score, on all
primary items and determined successes of RQ2's collection (the families population), and, for Ministral 3, the
agreement with mistral-common's encoding. No model is loaded. It writes one record per tag (ok, problems,
first-token pairs, Yes/No tokens, longest prompt, the wave B prompt checks, timing) and exits 1 if any gate fails.

--golden writes the golden item's renders (aligned, swapped and yes/no aligned prompts as text, and their token
counts) per tag, for tests/golden/families_waveb_rendered.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from wf.constants import REVISIONS

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "default_prompt.json"
GOLDEN_ARMS = ("aligned", "swapped", "yn_aligned")


def load_tokenizer(spec: Any, cache_dir: str | None) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, revision=REVISIONS[spec.hf_id], cache_dir=cache_dir,
                                         local_files_only=True, **spec.tokenizer_kwargs)


def scorer_for(spec: Any, tok: Any) -> Any:
    from wf.scoring.logodds import LogoddsScorer

    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs,
                                       prompt_format=spec.prompt_format)


def render(scorer: Any, spec: Any, messages: list[dict]) -> str:
    """The prompt as text, as scoring renders it: the template with the scorer's settings, or the plain format."""
    from wf.scoring.prompts import plain_text_prompt

    if spec.prompt_format == "plain":
        return plain_text_prompt(messages)
    return scorer._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **scorer.template_settings())


def golden_renders(spec: Any, scorer: Any) -> dict[str, Any]:
    from wf import families

    item = json.loads(GOLDEN.read_text(encoding="utf-8"))["item"]
    out: dict[str, Any] = {"hf_id": spec.hf_id, "revision": REVISIONS[spec.hf_id], "prompt_format": spec.prompt_format}
    for arm in GOLDEN_ARMS:
        msgs = families.family_messages(spec, item, arm)
        out[arm] = render(scorer, spec, msgs)
        out[f"{arm}_n_tokens"] = len(families._ids(scorer._encode(msgs)))
    return out


def gate_all_items(spec: Any, scorer: Any, primary: list[dict], successes: list[dict], cache_dir: str | None) -> dict[str, Any]:
    from wf import families

    ref = families.mistral_common_encoder(spec, cache_dir=cache_dir, local_files_only=True) if spec.mistral_common_check else None
    t0 = time.time()
    rec = families.tokenization_checks(scorer, spec, primary, successes, ref)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["n_primary"], rec["n_successes"] = len(primary), len(successes)
    rec["tokenizer_class"] = type(scorer._tok).__name__
    return rec


def main(argv: list[str] | None = None) -> int:
    from wf import families, rq1x
    from wf.rq2 import SOURCES, load_items

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--cache", default=None, help="model cache (default: the one the installed library reads)")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    args = ap.parse_args(argv)
    from wf.tools.prefetch_models import transformers_cache

    cache = args.cache or transformers_cache()
    sets = rq1x.rq1x_sets(load_items(tuple(args.sources)))
    primary, successes = sets["primary"], sets["successes"]
    print(f"cache {cache}; {len(primary)} primary items, {len(successes)} successes", flush=True)
    results: dict[str, Any] = {"environment": families.environment_record(), "cache": cache, "models": {}}
    golden: dict[str, Any] = {}
    for tag in args.tags:
        spec = families.spec_for(tag)
        tok = load_tokenizer(spec, cache)
        scorer = scorer_for(spec, tok)
        rec = gate_all_items(spec, scorer, primary, successes, cache)
        results["models"][tag] = {"hf_id": spec.hf_id, "revision": REVISIONS[spec.hf_id], **rec}
        if args.golden:
            golden[tag] = golden_renders(spec, scorer)
        print(f"{tag:24s} {'ok' if rec['ok'] else 'FAILED'}: {rec['n_candidates_checked']} candidates, "
              f"{rec['n_yes_no_prompts_checked']} yes/no prompts, {rec['n_problems']} problems, longest "
              f"{rec['max_prompt_tokens']} tokens, {rec['seconds']} s; first problems {rec['problems'][:3]}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    if args.golden:
        Path(args.golden).write_text(json.dumps({"note": "The golden item rendered by each wave B model's pinned "
                                                 "template (or the plain format) with the scorer's settings.",
                                                 "transformers": results["environment"].get("transformers"),
                                                 "models": golden}, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["ok"] for r in results["models"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
