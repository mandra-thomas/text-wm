"""TextCraft round 3: a tokenizer-only Hugging Face cache for the CPU checks (notes/textcraft_r3_design.md, section 13).

    python scripts/textcraft_r3_tokcache.py <cache> [--shared <read-only shared cache>]

For every round-3 model and every earlier model the checks render (Qwen3-8B, Mistral-7B-v0.3, Gemma-2-9B-it), at the
pinned revision, the tokenizer, template and config files only (no weights): downloaded without a token for an
ungated repository; for a gated one (no token is ever read here), copied file by file from the shared cache's
snapshot of that exact revision, which is only read. Prints what it did per model.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wf.constants import REVISIONS  # noqa: E402
from wf.textcraft import r3 as R3  # noqa: E402

PATTERNS = ["*.json", "*.jinja", "merges.txt", "vocab.json", "tokenizer.model", "tekken.json", "*.model.v3"]
EARLIER = ("Qwen/Qwen3-8B", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--shared", default=None)
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download

    ids = sorted({s.hf_id for s in R3.MODELS.values()} | set(EARLIER))
    api = HfApi(token=False)
    for hf_id in ids:
        rev = REVISIONS[hf_id]
        gated = bool(getattr(api.model_info(hf_id, token=False), "gated", False))
        if not gated:
            path = snapshot_download(hf_id, revision=rev, cache_dir=a.cache, allow_patterns=PATTERNS,
                                     ignore_patterns=["*.safetensors", "model.safetensors.index.json"], token=False)
            print(f"{hf_id} {rev[:12]} downloaded {len(list(Path(path).iterdir()))} files", flush=True)
            continue
        if not a.shared:
            print(f"{hf_id} is gated and no shared cache was given; skipped", flush=True)
            continue
        folder = "models--" + hf_id.replace("/", "--")
        src = Path(a.shared) / folder / "snapshots" / rev
        if not src.is_dir():
            print(f"{hf_id} is gated and the shared cache has no snapshot {rev[:12]}; skipped", flush=True)
            continue
        dst = Path(a.cache) / folder / "snapshots" / rev
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.iterdir()):
            if f.suffix == ".safetensors" or f.name.endswith(".index.json") or f.name.endswith((".bin", ".pth")):
                continue
            if any(Path(f.name).match(p) for p in PATTERNS):
                shutil.copyfile(f, dst / f.name)  # follows the snapshot's link to its blob; the source is only read
                n += 1
        (Path(a.cache) / folder / "refs").mkdir(parents=True, exist_ok=True)
        (Path(a.cache) / folder / "refs" / "main").write_text(rev)
        print(f"{hf_id} {rev[:12]} gated: copied {n} tokenizer and config files from the shared cache", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
