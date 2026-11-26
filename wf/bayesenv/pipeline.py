"""CPU runs: the smoke readout and the frozen analysis (notes/bayesenv_design.md, sections 9 and 15).

Both read scoring runs' shard folders and refuse inputs that are incomplete, mock, scored from a dirty checkout, mixed
across commits, or mixed across numerical environments (rule 1). The smoke checks must exist and be a real, clean run.
The cross-GPU subset must use the main run's design and differ from it in the GPU type only. Every stored prompt hash
is recomputed from the current code. The analysis runs from a clean checkout that contains the design tag.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from wf.bayesenv import analysis as A
from wf.bayesenv import design as D
from wf.util import REPO_ROOT, git_dirty, is_mock, runs_root, write_json, write_report, write_status


def _real_clean(p: Path, wanted_status: str) -> list[str]:
    problems = []
    status = (p / "STATUS.md").read_text().splitlines()[0] if (p / "STATUS.md").exists() else ""
    if wanted_status not in status or "MOCK" in status:
        problems.append(f"{p}: not a complete real run ({status or 'no status'})")
    inputs = json.loads((p / "inputs.json").read_text()) if (p / "inputs.json").exists() else {}
    if inputs.get("git_dirty", True):
        problems.append(f"{p}: run from a dirty checkout or no record")
    return problems


def _gpu_ok(env: dict, gpu_type: str | None) -> bool:
    return not gpu_type or any(x in env.get("gpu", "") for x in gpu_type.split("|"))


def check_run(name: str, gpu_type: str | None) -> tuple[list[Path], dict, list[str]]:
    """(shard folders, the shared environment, problems) of one scoring run."""
    root = runs_root() / name
    folders = sorted(p for p in root.glob("shard_*_of_*") if p.is_dir())
    problems: list[str] = []
    if not folders:
        return [], {}, [f"{name}: no shards"]
    counts = {p.name.rsplit("_of_", 1)[1] for p in folders}
    if len(counts) != 1 or sorted(int(p.name.split("_")[1]) for p in folders) != list(range(int(counts.pop()))):
        problems.append(f"{name}: shards missing: {[p.name for p in folders]}")
    envs, shas, designs = [], set(), set()
    for p in folders:
        problems += _real_clean(p, "REPORTED (shard complete)")
        shas.add((p / "git_sha").read_text().strip() if (p / "git_sha").exists() else None)
        envs.append(json.loads((p / "env.json").read_text()) if (p / "env.json").exists() else None)
        cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
        designs.add(json.dumps(cfg.get("design", {}), sort_keys=True))
    if len(shas) != 1 or None in shas:
        problems.append(f"{name}: shards at different or unknown commits: {sorted(map(str, shas))}")
    if any(e is None for e in envs) or any(e != envs[0] for e in envs):
        problems.append(f"{name}: shards scored in different environments: {envs}")
    if len(designs) != 1:
        problems.append(f"{name}: shards with different designs")
    env = envs[0] or {}
    if not _gpu_ok(env, gpu_type):
        problems.append(f"{name}: GPU {env.get('gpu')} is not the pinned type {gpu_type}")
    return folders, env, problems


def load_checks(name: str, gpu_type: str | None) -> tuple[dict | None, list[str]]:
    p = runs_root() / name
    problems = _real_clean(p, "REPORTED (checks)")
    env = json.loads((p / "env.json").read_text()) if (p / "env.json").exists() else {}
    if not _gpu_ok(env, gpu_type):
        problems.append(f"{name}: GPU {env.get('gpu')} is not the pinned type {gpu_type}")
    f = p / "checks.json"
    if not f.exists():
        return None, problems + [f"{name}: no checks.json"]
    return json.loads(f.read_text()), problems


def design_of(folders: list[Path]) -> D.Design:
    cfg = json.loads((folders[0] / "config.json").read_text())
    return D.Design(**cfg.get("design", {}))


def run_bayesenv_smoke_readout(cfg: dict) -> int:
    """Config: smoke (run id), checks (run id), gpu_type, budget_gpu_h, fixed_gpu_h, n_models."""
    from wf.runs import _init_run

    run_id = cfg["run_id"]
    if not is_mock() and git_dirty():
        raise SystemExit(f"{run_id}: the checkout has uncommitted changes")
    folders, env, problems = check_run(cfg["smoke"], cfg.get("gpu_type"))
    checks, cp = load_checks(cfg["checks"], cfg.get("gpu_type"))
    problems += cp
    d = _init_run(run_id, {**cfg, "resume": False})
    records = A.read_folders(folders) if folders else []
    bad = A.verify_hashes(records)
    if bad:
        problems.append(f"{len(bad)} smoke units scored for prompts the current code does not render: {bad[:3]}")
    if problems:
        write_status(run_id, "smoke readout waits:\n" + "\n".join(problems), ok=False)
        return 1
    timing = {"sequences": 0, "seconds": 0.0}
    for f in folders:
        t = json.loads((f / "timing.json").read_text())
        timing["sequences"] += t["sequences"]
        timing["seconds"] += t["seconds"]
    out = A.smoke_readout(records, timing, checks, budget_gpu_h=cfg.get("budget_gpu_h", 6.3),
                          fixed_gpu_h=cfg.get("fixed_gpu_h", 0.0), n_models=cfg.get("n_models", 2))
    out["environment"] = env
    write_json(d / "metrics.json", out)
    dec = out.get("decision", {})
    write_report(run_id, "# Smoke readout\n\n```json\n" + json.dumps(dec, indent=1) + "\n```\n")
    write_status(run_id, f"band: {dec.get('band')}; design {dec.get('design')}", ok=False, reported="smoke readout")
    return 0


def tag_is_ancestor(tag: str) -> bool:
    r = subprocess.run(["git", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=REPO_ROOT, capture_output=True)
    return r.returncode == 0


def run_bayesenv_analysis(cfg: dict) -> int:
    """Config: tag; models: {tag: {main, xgpu, checks, gpu_type, xgpu_gpu_type, low_floor_family, sizing}}."""
    from wf.runs import _init_run

    run_id = cfg["run_id"]
    if not is_mock():
        if git_dirty():
            raise SystemExit(f"{run_id}: the analysis must run from a clean checkout")
        if not tag_is_ancestor(cfg["tag"]):
            raise SystemExit(f"{run_id}: this checkout does not contain the design tag {cfg['tag']}")
    d_ = _init_run(run_id, {**cfg, "resume": False})
    problems: list[str] = []
    loaded = {}
    for tag, m in cfg["models"].items():
        try:
            A.check_sizing(m.get("sizing"))
        except ValueError as e:
            problems.append(f"{tag}: {e}")
        folders, env, pr = check_run(m["main"], m.get("gpu_type"))
        problems += pr
        xf, xenv, xpr = check_run(m["xgpu"], m.get("xgpu_gpu_type"))
        problems += xpr
        checks, cp = load_checks(m["checks"], m.get("gpu_type"))
        problems += cp
        if folders and xf:
            same = {k: v for k, v in env.items() if k != "gpu"} == {k: v for k, v in xenv.items() if k != "gpu"}
            if not same or env.get("gpu") == xenv.get("gpu"):
                problems.append(f"{tag}: the cross-GPU subset must differ from the main run in the GPU type only")
            if design_of(folders) != design_of(xf):
                problems.append(f"{tag}: the cross-GPU subset was scored with a different design")
        main_records = A.read_folders(folders) if folders else []
        x_records = A.read_folders(xf) if xf else []
        bad = A.verify_hashes(main_records + x_records)
        if bad:
            problems.append(f"{tag}: {len(bad)} units scored for prompts the current code does not render")
        loaded[tag] = (folders, env, xf, xenv, checks, main_records, x_records)
    if problems:
        write_status(run_id, "analysis waits:\n" + "\n".join(problems), ok=False)
        return 1
    from wf.provenance import provenance, shard_inputs

    out = {}
    for tag, m in cfg["models"].items():
        folders, env, xf, xenv, checks, main_records, x_records = loaded[tag]
        res = A.analyse(main_records, design_of(folders), bool(m["low_floor_family"]), checks, m["sizing"],
                        x_records)
        res["environment"], res["cross_gpu_environment"] = env, xenv
        out[tag] = res
    out["provenance"] = provenance(runs={tag: runs_root() / m["main"] for tag, m in cfg["models"].items()},
                                   inputs=shard_inputs([f for v in loaded.values() for f in v[0] + v[2]]))
    write_json(d_ / "metrics.json", out)
    write_report(run_id, report(out))
    write_status(run_id, "analysis complete; see metrics.json", ok=False, reported="analysis")
    return 0


def report(out: dict) -> str:
    lines = ["# Controlled environment: analysis", ""]
    for tag, res in out.items():
        if tag == "provenance":
            continue
        lines += [f"## {tag}", "", f"Primary readout: {res.get('primary_readout')}", ""]
        for ro, v in res["readouts"].items():
            lines += [f"### {ro} ({v['role']})", "", "| estimand | estimate | 95% CI | p | MDE |", "|---|---|---|---|---|"]
            for k in A.HOLM_FAMILY + ("evidence_slope", "goal_slope", "tag_slope", "note_slope", "base_rate_slope"):
                e = v["estimands"].get(k)
                if e:
                    lines.append(f"| {k} | {e['estimate']:.3f} | [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] | "
                                 f"{e['p']:.3g} | {e['mde']:.3f} |")
            lines.append("")
    return "\n".join(lines) + "\n"
