"""Record environment versions into runs/env.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from wf.constants import STEP_CAP
from wf.util import REPO_ROOT, git_sha, runs_root, write_json


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def main() -> None:
    w2w = Path(os.environ.get("WORD2WORLD_ROOT", REPO_ROOT / "third_party" / "Word2World"))
    w2w_sha = "missing"
    if (w2w / ".git").exists():
        try:
            w2w_sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=w2w).decode().strip()
            )
        except Exception:
            w2w_sha = "unknown"

    payload = {
        "repo_git_sha": git_sha(),
        "word2world_root": str(w2w),
        "word2world_sha": w2w_sha,
        "python": sys.version,
        "vllm": _pkg_version("vllm"),
        "transformers": _pkg_version("transformers"),
        "torch": _pkg_version("torch"),
        "STEP_CAP": STEP_CAP,
        "WF_MOCK_ENV": os.environ.get("WF_MOCK_ENV", "0"),
        "WF_MOCK_AGENT": os.environ.get("WF_MOCK_AGENT", "0"),
        "WF_MOCK_SCORE": os.environ.get("WF_MOCK_SCORE", "0"),
    }
    out = runs_root() / "env.json"
    write_json(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
