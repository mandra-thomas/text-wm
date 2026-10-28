"""Assemble runs/PILOT_REPORT.md from per-run REPORT/STATUS files."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from wf.util import REPO_ROOT, git_sha, runs_root


def _slurp(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "_missing_"


def main() -> None:
    root = runs_root()
    filed = _slurp(root / "FILED.txt").strip().replace("\n", " | ")
    lines = [
        f"# Pilot report — {datetime.now(timezone.utc).date().isoformat()} — git {git_sha()} — FILED {filed}",
        "",
        "## Tier 0",
        "| check | result | pass |",
        "| --- | --- | --- |",
    ]
    for rid in (
        "t0_smoke",
        "t0_replay",
        "t0_agent_determinism",
        "t0_parser",
        "t0_ckpt_accuracy",
        "t0_scoring_agreement",
        "t0_expert_labels",
    ):
        st = _slurp(root / rid / "STATUS.md").replace("\n", " ")[:200]
        ok = "yes" if "PASS" in st or "reported" in st.lower() or "available" in st.lower() else "see STATUS"
        if "FAIL" in st:
            ok = "no"
        lines.append(f"| {rid} | {st} | {ok} |")

    lines += ["", "## Pilot A (RQ2)", ""]
    for rid in ("pilotA_transitions", "pilotA_score"):
        lines.append(f"### {rid}")
        lines.append(_slurp(root / rid / "REPORT.md"))
        lines.append("")

    lines += ["## Pilot B (RQ3)", ""]
    for rid in (
        "pilotB_headroom",
        "pilotB_consumption",
        "pilotB_eligibility",
        "pilotB_retention_minirun",
    ):
        lines.append(f"### {rid}")
        lines.append(_slurp(root / rid / "REPORT.md"))
        lines.append("")

    lines += [
        "## Deviations / failures",
        "See per-run STATUS.md files under runs/.",
        "",
        "## Note",
        "If this report was produced with WF_MOCK_*=1, replace with ICE results before claiming pilot completion.",
        "",
    ]
    out = root / "PILOT_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
