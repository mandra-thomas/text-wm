"""TextCraft round 3, deviation D-G1 (2026-11-23, after the freeze; notes/textcraft_r3_deviation_gemma_e4b.md).

`textcraft_r3x_gemma4-e4b-it`: an exploratory run of Gemma 4 E4B, whose confirmatory slot is "not scored" (it failed
the bf16 eager-against-SDPA attention gate at its smoke run). It scores the same items, arms, prompts and scorer as a
core model (r3_score.score_shard: SDPA, bf16, the round-3 gate, the golden renderings, node-local weights), with the
attention-agreement gate replaced by the float32 diagnostic recorded in wf/config/textcraft_r3x_e4b_attention.json,
which must meet the criterion stated before it ran. The analysis reads it in a separate exploratory block
(r3_analysis.r3x_block); it never enters A3, A3x, B3, H3-P, R2 or the replication reading.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from wf.textcraft import r3 as R3

TAG = "gemma4-e4b-it"
RUN_ID = f"textcraft_r3x_{TAG}"
DIAG_RECORD = Path(__file__).resolve().parents[1] / "config" / "textcraft_r3x_e4b_attention.json"
CRITERION = {"yes_no_mean_abs_max": 0.01, "yes_no_any_abs_max": 0.05}  # float32 eager against SDPA, stated first


def criterion_problems(path: Path | None = None) -> list[str]:
    """The run starts only if the committed float32 diagnostic meets CRITERION (recomputed from its numbers)."""
    path = path or DIAG_RECORD
    if not path.exists():
        return [f"no float32 attention diagnostic at {path}"]
    rec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    f32 = rec.get("float32", {})
    if rec.get("tag") != TAG:
        return [f"the diagnostic is for {rec.get('tag')!r}, not {TAG}"]
    out = []
    if not f32.get("yes_no_mean_abs", float("inf")) <= CRITERION["yes_no_mean_abs_max"]:
        out.append(f"float32 eager and SDPA differ by {f32.get('yes_no_mean_abs')} nats on average (yes/no)")
    if not f32.get("yes_no_max_abs", float("inf")) <= CRITERION["yes_no_any_abs_max"]:
        out.append(f"float32 eager and SDPA differ by {f32.get('yes_no_max_abs')} nats on one item (yes/no)")
    return out


def spec():
    """E4B's core spec with the bf16 attention-agreement gate off (D-G1 replaces it with the float32 diagnostic)."""
    return replace(R3.MODELS[TAG], attn_agreement_check=False)


def run_textcraft_r3x(cfg: dict) -> int:
    from wf.textcraft import r3_score as S

    if cfg["run_id"] != RUN_ID or cfg.get("model") != TAG:
        raise SystemExit(f"D-G1's exploratory run is {RUN_ID} on {TAG}, not {cfg['run_id']} on {cfg.get('model')}")
    bad = criterion_problems()
    if bad:
        raise SystemExit(f"{RUN_ID} does not run: {bad} (design deviation D-G1)")
    return S.score_shard(cfg, TAG, spec(), lambda rs: rs)
