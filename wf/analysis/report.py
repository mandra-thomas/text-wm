"""Write runs/<id>/REPORT.md from JSONL + stats."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wf.util import write_report


def write_run_report(run_id: str, title: str, sections: dict[str, Any]) -> Path:
    lines = [f"# REPORT — {run_id}", "", f"## {title}", ""]
    for name, body in sections.items():
        lines.append(f"### {name}")
        if isinstance(body, dict):
            for k, v in body.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(body))
        lines.append("")
    return write_report(run_id, "\n".join(lines))


def format_pilot_report_skeleton() -> str:
    return """# Pilot report — <date> — git <sha> — FILED <date>

## Tier 0
| check | result | pass |
| --- | --- | --- |
| t0_replay determinism | | |
| t0_agent_determinism | | |
| t0_parser unparseable | | |
| t0_ckpt_accuracy | | |
| t0_scoring_agreement | | |
| expert labels | | |

## Pilot A (RQ2)
| model | population | n items / states | Δ_p (pp) | 95% CI | perm p | ICC |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-8b | goal-agnostic | | | | | |
| qwen3-8b | goal-revealing | | | | | |
| w2w-qwen2.5-7b | goal-agnostic | | | | | |
A (asymmetry): ... ; Δ_p by history length: ... ; committed vs candidate: ...
Rule applied: PILOT_RULE_GOAL_CHANNEL -> fired / not fired
ELIGIBLE_SET_TARGET: ...

## Pilot B (RQ3)
| condition | success | rank outcome (mean S) |
| --- | --- | --- |
| none | | |
| oracle/discarded | | |
| oracle/retained | | |
| validity_only | | |
Oracle gain over none: ... (CI) ; validity-only recovery fraction: ...
Decision-critical rate under oracle: ... ; follow-rate: ...
Eligibility: invalid-proposal rate ... ; p_favorable ... ; p_unfavorable ...
Mini-run cells (S, success): fav/disc ... fav/ret ... unfav/disc ... unfav/ret ...
H_discarded ... H_retained ... I ... ; realized corruptions/episode per arm ...
Rules applied: HEADROOM -> ... ; RETAINED_FORMAT -> ...

## Deviations / failures
...
"""
