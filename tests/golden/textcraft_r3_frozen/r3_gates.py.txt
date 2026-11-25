"""TextCraft round 3: the mechanical states read from filed run records (notes/textcraft_r3_design.md, section 14,
amendments after review 1). Nothing here reads a score of round 2; it reads round-3 run records, and of round 2 only
the GPU names its scoring jobs recorded (jobs.jsonl) and its record partition.

- `job_gpus`, `shard_gpu`: each job's own GPU list (jobs.jsonl), and a shard's list when every job recorded the same
  one (M1: wf.provenance.run_record keeps a set of names, so two L40S would read as one).
- `model_state`: "scored", "not scored" (a filed gate record with ok false: the loading check, the tokenization gate or
  the attention agreement) or "waiting" (M2).
- `gemma_slot`: the Gemma fallback rule applied to the filed smoke record of Gemma 4 12B (M3, S9).
- `envcheck_plan`: the environment check's plan from round 2's recorded GPU names (M4), and its committed copy.
- `freeze_commit`, `commit_problems`: every round-3 job runs from the freeze tag's commit (S12).

    python -m wf.textcraft.r3_gates plan --r2-runs <round-2 runs folder>     # writes the committed plan file
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from wf.textcraft import r3 as R3

FREEZE_TAG_PREFIX = "textcraft-r3-design-frozen-"
CODE_PATHS = ("wf/", "scripts/", "pace/")  # a commit that differs from the tag only outside these is code-identical
PLAN_FILE = Path(__file__).resolve().parents[1] / "config" / "textcraft_r3_envcheck_plan.json"
UNOBTAINABLE = "gpu_unobtainable.json"  # filed in an envcheck run folder when a planned GPU type was not obtainable
UNOBTAINABLE_DAYS = 7
MASS_MIN = 0.5
STRUCTURAL_SHARE_MIN = 0.5  # "most" smoke yes/no prompts


def mock() -> bool:
    return os.environ.get("WF_MOCK_SCORE", "0") == "1"


# ---------------- GPUs per job (M1) ----------------


def job_gpus(folder: Path) -> list[list[str] | None]:
    """Each job's own GPU list, as it recorded it (None where it recorded none)."""
    from wf.provenance import JOBS_FILE
    from wf.util import read_jsonl

    f = Path(folder) / JOBS_FILE
    return [j.get("gpu") or None for j in (read_jsonl(f) if f.exists() else [])]


def shard_gpu(folder: Path) -> list[str] | None:
    """The shard's GPU list when every job of it recorded the same list; None otherwise (untyped)."""
    lists = job_gpus(folder)
    if not lists or any(x is None for x in lists) or any(x != lists[0] for x in lists):
        return None
    return list(lists[0])


def shard_commits(folder: Path) -> list[str]:
    from wf.provenance import JOBS_FILE
    from wf.util import read_jsonl

    f = Path(folder) / JOBS_FILE
    out = [j.get("git_sha") for j in (read_jsonl(f) if f.exists() else [])]
    sha = Path(folder) / "git_sha"
    if sha.exists():
        out.append(sha.read_text(encoding="utf-8").strip())
    return [c for c in out if c]


# ---------------- the freeze commit (S12) ----------------


def _git(args: list[str]) -> str:
    from wf.util import REPO_ROOT

    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()


def freeze_commit(run_git: Callable[[list[str]], str] = _git) -> str | None:
    """The commit of the one tag named textcraft-r3-design-frozen-*; None if there is none (before the freeze)."""
    try:
        tags = [t for t in run_git(["tag", "-l", FREEZE_TAG_PREFIX + "*"]).splitlines() if t.strip()]
    except Exception:  # noqa: BLE001 - no git: treated as no tag
        return None
    if len(tags) != 1:
        if len(tags) > 1:
            raise SystemExit(f"more than one round-3 freeze tag: {tags}")
        return None
    return run_git(["rev-parse", f"{tags[0]}^{{commit}}"])


def code_identical(commit: str, tag_commit: str, run_git: Callable[[list[str]], str] = _git) -> bool:
    """True when `commit` is the tag's commit, or differs from it only outside wf/, scripts/ and pace/."""
    if commit == tag_commit:
        return True
    try:
        changed = [p for p in run_git(["diff", "--name-only", tag_commit, commit]).splitlines() if p.strip()]
    except Exception:  # noqa: BLE001 - an unknown commit is not code-identical
        return False
    return not any(p.startswith(CODE_PATHS) for p in changed)


def commit_problems(commits: dict[str, list[str]], tag_commit: str | None,
                    run_git: Callable[[list[str]], str] = _git) -> dict[str, list[str]]:
    """{tag: problems}: every job of every shard at the freeze tag's commit (or a code-identical one)."""
    out: dict[str, list[str]] = {}
    for tag, cs in commits.items():
        if tag_commit is None:
            out[tag] = ["no round-3 freeze tag in this checkout"]
            continue
        bad = sorted({c for c in cs if not code_identical(c, tag_commit, run_git)})
        if bad or not cs:
            out[tag] = [f"jobs at commits {bad or 'none recorded'}, not the freeze tag's {tag_commit[:12]}"]
    return out


def _dirty() -> bool:
    from wf.util import git_dirty

    return git_dirty()


def require_freeze_commit(run_git: Callable[[list[str]], str] = _git, dirty: Callable[[], bool] = _dirty) -> dict[str, Any]:
    """At job start: this checkout's HEAD is the freeze tag's commit, or code-identical to it, and its tracked files
    carry no uncommitted change (review 2, S2). Mock runs are not checked."""
    if mock():
        return {"checked": "mock"}
    tag = freeze_commit(run_git)
    if tag is None:
        raise SystemExit("no round-3 freeze tag in this checkout: round-3 jobs run only from the freeze tag")
    head = run_git(["rev-parse", "HEAD"])
    if not code_identical(head, tag, run_git):
        raise SystemExit(f"this checkout is at {head[:12]}, not the freeze tag's {tag[:12]} (or a code-identical commit)")
    if dirty():
        raise SystemExit("this checkout has uncommitted changes to tracked files: round-3 jobs run from a clean checkout")
    return {"checked": True, "head": head, "tag_commit": tag}


def shard_dirty(folder: Path) -> bool:
    """Whether the shard's run recorded uncommitted changes (inputs.json, git_dirty; review 2, S2)."""
    f = Path(folder) / "inputs.json"
    return bool(json.loads(f.read_text(encoding="utf-8")).get("git_dirty")) if f.exists() else False


def refuse_after_gate_failure(folder: Path) -> None:
    """A filed gate failure is final (review 2, S5): a run into a shard folder that holds one refuses to start."""
    f = Path(folder) / "tokenization_checks.json"
    if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("ok") is False:
        raise SystemExit(f"{folder} holds a filed gate failure; it is final (design 14.M2) and is never rerun")


# ---------------- gate records and model states (M2) ----------------


def gate_records(root: Path, tag: str) -> list[tuple[str, dict]]:
    """(run/shard, tokenization_checks.json) of every filed smoke and score shard of `tag`."""
    out = []
    for kind in ("smoke", "score"):
        for f in sorted((Path(root) / f"textcraft_r3_{kind}_{tag}").glob("shard_*/tokenization_checks.json")):
            out.append((f"{f.parent.parent.name}/{f.parent.name}", json.loads(f.read_text(encoding="utf-8"))))
    return out


def gate_failure(rec: dict) -> str | None:
    """Why a filed gate record stops the model (loading, gate or attention agreement), or None."""
    if rec.get("ok") is not False:
        return None
    if rec.get("loading_failed"):
        return "loading check failed"
    att = rec.get("attention_agreement")
    if att is not None and att.get("ok") is False:
        return "attention agreement failed"
    return "tokenization gate failed"


def model_state(root: Path, tag: str) -> dict[str, Any]:
    """"not scored" once any filed smoke or score gate record of the model failed; "scored" when the score run has
    a gate record and none failed; "waiting" otherwise."""
    recs = gate_records(root, tag)
    fails = [(where, gate_failure(r)) for where, r in recs if gate_failure(r)]
    if fails:
        return {"state": "not scored", "reasons": [f"{w}: {why}" for w, why in fails]}
    if any(w.startswith(f"textcraft_r3_score_{tag}/") for w, _r in recs):
        return {"state": "scored"}
    return {"state": "waiting"}


# ---------------- the Gemma fallback (M3, S9) ----------------


def structural(token: str, special: bool) -> bool:
    """A structural or template marker rather than an answer word: a special token, whitespace, markdown or a
    bracketed marker."""
    s = (token or "").strip()
    if s.lower() in ("yes", "no"):
        return False
    return bool(special or not s or s[0] in "*#`>-_~|<[{")


def smoke_diagnostic(rows: list[dict]) -> dict[str, Any]:
    """Over a smoke run's yes/no scores: the median P(Yes) + P(No), and the share of prompts whose most probable next
    token is a structural marker (`top_token`, `top_token_special`, recorded by the smoke run)."""
    yn = [r for r in rows if str(r.get("condition", "")).startswith("yn_")]
    mass = [math.exp(r["logp_plus"]) + math.exp(r["logp_minus"]) for r in yn]
    tops = [structural(r.get("top_token", ""), bool(r.get("top_token_special"))) for r in yn if "top_token" in r]
    return {"n_yes_no": len(yn), "median_mass": statistics.median(mass) if mass else float("nan"),
            "structural_share": sum(tops) / len(tops) if tops else float("nan"), "n_top_tokens": len(tops)}


def fallback_triggers(rec: dict | None, rows: list[dict]) -> list[str]:
    """The fallback rule's triggers on Gemma 4 12B's filed smoke record (design 14.M3, with S9): the loading check,
    the tokenization gate, the attention agreement, or a format mismatch (median yes/no mass below one half AND on
    most smoke yes/no prompts the most probable next token is a structural or template marker)."""
    out = []
    if rec is not None:
        why = gate_failure(rec)
        if why:
            out.append(why)
    diag = smoke_diagnostic(rows)
    if diag["median_mass"] < MASS_MIN and diag["structural_share"] > STRUCTURAL_SHARE_MIN:
        out.append("format mismatch (low yes/no mass with structural next tokens)")
    return out


def gemma_slot(root: Path) -> dict[str, Any]:
    """The Gemma core slot: gemma4-12b-it unless its filed smoke run triggered the fallback, then gemma4-e4b-it;
    "waiting" until the 12B smoke run has a gate record. A low mass without format-mismatch evidence does not fire:
    it goes to the competence gate. Whether the slot's model is scored is model_state's question."""
    d = Path(root) / f"textcraft_r3_smoke_{R3.GEMMA_CORE}" / "shard_00_of_01"
    gate = d / "tokenization_checks.json"
    if not gate.exists():
        return {"slot": "waiting", "triggers": [], "reason": "Gemma 4 12B's smoke run has no gate record"}
    from wf.util import read_jsonl

    rec = json.loads(gate.read_text(encoding="utf-8"))
    rows = read_jsonl(d / "scoring.jsonl") if (d / "scoring.jsonl").exists() else []
    trig = fallback_triggers(rec, rows)
    return {"slot": R3.GEMMA_FALLBACK if trig else R3.GEMMA_CORE, "triggers": trig,
            "diagnostic": smoke_diagnostic(rows)}


# ---------------- declarations filed by the operator (review 2, S4 and S6) ----------------


def _date(text: str):
    import datetime as dt

    return dt.date.fromisoformat(str(text)[:10])


def unobtainable_check(folder: Path, today=None) -> dict[str, Any]:
    """`gpu_unobtainable.json` in an environment-check run folder: it must name the first submission's Slurm job id and
    date, and at least UNOBTAINABLE_DAYS must have passed since that date (review 2, S4)."""
    import datetime as dt

    f = Path(folder) / UNOBTAINABLE
    if not f.exists():
        return {"filed": False, "valid": False}
    rec = json.loads(f.read_text(encoding="utf-8"))
    today = today or dt.date.today()
    try:
        first = _date(rec["first_submission_date"])
        job = str(rec["first_submission_job_id"]).strip()
    except (KeyError, ValueError):
        return {"filed": True, "valid": False, "problem": "needs first_submission_job_id and first_submission_date"}
    days = (today - first).days
    ok = bool(job) and days >= UNOBTAINABLE_DAYS
    return {"filed": True, "valid": ok, "first_submission_job_id": job, "first_submission_date": str(first),
            "days": days, **({} if ok else {"problem": f"{days} days since the first submission; {UNOBTAINABLE_DAYS} needed"})}


FIRST_CORE = "textcraft_r3_first_core_reading.json"  # written once, when the analysis first reads the core models


def not_run_declaration(root: Path, tag: str, first_core: str | None) -> dict[str, Any]:
    """`runs/textcraft_r3_score_<tag>/not_run.json` (review 2, S6): an extension model declared not run for GPU
    availability, with its date. Valid only for an extension model and only when dated before the analysis first
    read any core result; then the model counts as not scored (p = 1)."""
    f = Path(root) / f"textcraft_r3_score_{tag}" / "not_run.json"
    if not f.exists():
        return {"declared": False}
    rec = json.loads(f.read_text(encoding="utf-8"))
    problems = []
    if tag not in R3.EXTENSION:
        problems.append(f"{tag} is a core model: it cannot be declared not run")
    try:
        when = str(rec["date"])
        _date(when)
    except (KeyError, ValueError):
        problems.append("the declaration needs a date (ISO 8601)")
        when = None
    if when and first_core and when >= first_core:
        problems.append(f"dated {when}, not before the first core reading {first_core}")
    return {"declared": True, "date": when, "reason": rec.get("reason"), "valid": not problems, "problems": problems}


# ---------------- the environment check's per-score rule, pinned at the freeze (review 2) ----------------

RULE_PIN = Path(__file__).resolve().parents[1] / "config" / "textcraft_r3_envcheck_rule.json"


def rule_fingerprint() -> dict[str, Any]:
    """Round 3's environment-check rule (r3.ENVCHECK_RULE: wave B's original criteria, unamended) and the sha256 of
    its canonical JSON."""
    import hashlib

    rule = dict(R3.ENVCHECK_RULE)
    return {"rule": rule, "sha256": hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest(),
            "source": "wave B's original criteria, unamended (design 14.M4)"}


def rule_problems(path: Path = RULE_PIN) -> list[str]:
    """Round 3's rule constant must equal the one pinned in the committed file (values and sha256)."""
    if not path.exists():
        return [f"no pinned environment-check rule at {path}"]
    pinned = json.loads(path.read_text(encoding="utf-8"))
    now = rule_fingerprint()
    return [f"the environment-check rule's {k} differs from the one pinned at the freeze" for k in now
            if now[k] != pinned.get(k)]


# ---------------- the environment-check plan (M4) ----------------


def envcheck_plan(r2_root: Path, tag: str, records: list[dict]) -> dict[str, Any]:
    """From round 2's recorded GPU names only: each round-2 shard of `tag` typed (every job the same single-GPU list)
    or not, the seeded environment-check records it holds (round 2's own partition), and the jobs: one per typed
    shard on its exact GPU name, or one job when every typed shard shares one type."""
    from wf.textcraft.r2_score import shard_records
    from wf.textcraft.r3_score import envcheck_subset

    run = Path(r2_root) / f"textcraft_r2_score_{tag}"
    folders = sorted(p for p in run.glob("shard_*_of_*") if p.is_dir())
    if not folders:
        return {"tag": tag, "status": "waiting", "reason": f"no round-2 shards in {run}"}
    n = int(folders[0].name.split("_of_")[1])
    seeded = {r["id"] for r in envcheck_subset(records)}
    shards = []
    for p in folders:
        i = int(p.name.split("_")[1])
        g = shard_gpu(p)
        typed = g is not None and len(g) == 1
        ids = sorted(r["id"] for r in shard_records(records, i, n) if r["id"] in seeded)
        shards.append({"shard": i, "gpu": g[0] if typed else None, "typed": typed, "n_records": len(ids)})
    typed = [s for s in shards if s["typed"]]
    types = sorted({s["gpu"] for s in typed})
    if len(types) == 1:
        jobs = [{"job": 0, "gpu": types[0], "r2_shards": [s["shard"] for s in typed]}]
    else:
        jobs = [{"job": k, "gpu": s["gpu"], "r2_shards": [s["shard"]]} for k, s in enumerate(typed)]
    n_comp = sum(s["n_records"] for s in typed)
    return {"tag": tag, "status": "planned", "n_r2_shards": n, "shards": shards, "jobs": jobs,
            "single_type": len(types) == 1, "n_comparable_planned": n_comp,
            "runnable": n_comp >= R3.ENVCHECK_MIN_COMPARABLE and len(shards) == n}


def read_plan(path: Path = PLAN_FILE) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def plan_job(tag: str, job: int, plan: dict | None = None) -> dict[str, Any]:
    plan = read_plan() if plan is None else plan
    if plan is None or tag not in plan.get("models", {}):
        raise SystemExit(f"no committed environment-check plan for {tag} ({PLAN_FILE})")
    p = plan["models"][tag]
    if not p.get("runnable"):
        raise SystemExit(f"the plan for {tag} is not runnable: the environment check is 'not run (GPU types)'")
    jobs = p["jobs"]
    if not 0 <= job < len(jobs):
        raise SystemExit(f"{tag}: the plan has jobs 0 to {len(jobs) - 1}, not {job}")
    return {**jobs[job], "n_jobs": len(jobs), "n_r2_shards": p["n_r2_shards"]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("plan")
    b.add_argument("--r2-runs", required=True)
    b.add_argument("--out", default=str(PLAN_FILE))
    pin = sub.add_parser("pin-rule")
    pin.add_argument("--out", default=str(RULE_PIN))
    a = ap.parse_args(argv)
    if a.cmd == "pin-rule":
        Path(a.out).write_text(json.dumps(rule_fingerprint(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(rule_fingerprint()))
        return 0
    from wf.textcraft.r2_items import read_records

    records = read_records()
    plan = {"rule": "design 14.M4", "models": {t: envcheck_plan(Path(a.r2_runs), t, records) for t in R3.ENVCHECK_MODELS}}
    Path(a.out).write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    for t, p in plan["models"].items():
        print(t, p.get("status"), p.get("n_comparable_planned"), [(j["gpu"], j["r2_shards"]) for j in p.get("jobs", [])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
