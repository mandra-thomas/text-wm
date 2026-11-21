"""The minimal interpretability pilot (notes/interp_design.md, amendment 1, section M): Qwen3-8B, yes/no format.

Runs (wired in `wf.runs`):

- `interp_pilot_stage0` (GPU, about 0.8 A100-hour): readouts and answer-position residuals at every layer for
  RQ2's determined failures (aligned, swapped, neutral and swapped-plus-mention prompts) and the determined
  successes (aligned, swapped, neutral); the first-token readout of the failures' aligned and swapped
  two-candidate prompts (no residuals); the batch noise; the reproduction gates against the logged rq1v scores
  (yes/no L and first-token L_first, with item hashes). Every later run requires this one to have passed.
- `interp_pilot_patch` (GPU, about 4 A100-hours): residual-stream patching at every layer by the pilot's groups,
  noising and denoising, on 100 clear-effect items of the calibration fold's games (at least 30 at step 0 and 25
  whose history names the action's object, weighted back to the population) and 50 placebo pairs.
- `interp_pilot_ablate` (GPU, about 0.7 A100-hour): directions estimated on the other fold's games and applied at
  the answer position to calibration-fold items only (amendment 2, B3), at the depth-profile layers, the
  near-linear caveat layer and the last layer (sanity only).
- `interp_pilot_analysis` (CPU): the pre-registered readouts per stratum (step 0, later, all).

The GPU runs refuse to start on a GPU type other than the one stage 0 ran on (amendment 2, B8).

`load_model` and `load_sets` are module functions so the tests can substitute a tiny model and synthetic items.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from wf import constants as C
from wf.util import read_jsonl, require_filed, runs_root, write_json, write_status

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_TAG = "qwen3-8b-fewshot"
STAGE0 = "interp_pilot_stage0"
FAIL_GOALS = ("aligned", "swapped", "absent", "swapped_plus")
SUCC_GOALS = ("aligned", "swapped", "absent")


# ---------------- inputs ----------------


def load_model(cfg: dict) -> tuple[Any, Any]:
    """(Parts, tokenizer) of the pinned model."""
    from wf.interp.model import load, parts

    model, tok = load(cfg.get("model", MODEL_ID))
    return parts(model), tok


def load_sets(cfg: dict) -> dict[str, list[dict]]:
    """RQ2's determined failures ("failures") and the determined successes ("successes"), as rq1v scored them."""
    from wf import rq1x
    from wf.rq2 import SOURCES, load_items

    sets = rq1x.rq1x_sets(load_items(tuple(cfg.get("sources", SOURCES))))
    return {"failures": sets["primary"], "successes": sets["successes"]}


def all_items(cfg: dict) -> list[dict]:
    """Every collected item of the sources (for the placebo pool: swaps other items of the game received)."""
    from wf.rq2 import SOURCES, load_items

    return load_items(tuple(cfg.get("sources", SOURCES)))


LOGGED_NOTE = ("rq1v_score rows carry no model_revision (null): the revision is the one pinned in wf.constants at "
               "the scoring run's commit, not recorded per row; each row's item_hash is checked against the current "
               "item instead (amendment 2, B10)")


def logged_scores(cfg: dict) -> dict[str, dict]:
    """The logged rq1v scores of this model: "yn" {(item id, arm): L} for the yes/no arms; "obs" {(item id, arm):
    L_first} for the plain two-candidate arms "aligned" and "swapped" (the first-token readout); "item_hash"
    {item id: the item hash the scorer recorded}."""
    from wf.rq2 import complete_shards

    folders, problems = complete_shards(cfg.get("logged_run", "rq1v_score"))
    if problems:
        raise SystemExit(f"logged scores incomplete: {problems}")
    tag = cfg.get("model_tag", MODEL_TAG)
    out: dict[str, dict] = {"yn": {}, "obs": {}, "item_hash": {}}
    for f in folders:
        for s in read_jsonl(f / "scoring.jsonl"):
            if s["model"] != tag:
                continue
            out["item_hash"].setdefault(s["transition_id"], s.get("item_hash"))
            if s["condition"].startswith("yn_") and s["L"] == s["L"]:
                out["yn"][(s["transition_id"], s["condition"])] = s["L"]
            elif s["condition"] in ("aligned", "swapped"):
                lf = s.get("L_first")
                if lf is not None and lf == lf:
                    out["obs"][(s["transition_id"], s["condition"])] = lf
    return out


def folds(items: list[dict], seed: int) -> dict[str, int]:
    """Game -> fold (0 or 1), the two halves of a seeded shuffle of the games (amendment 1: two folds, one seed)."""
    from wf.interp.probes import split_games

    a, b = split_games([t["game_id"] for t in items], seed)
    return {**{g: 0 for g in a}, **{g: 1 for g in b}}


def _stage0_store(cfg: dict):
    from wf.interp.run import ActStore

    return ActStore(runs_root() / cfg.get("stage0", STAGE0) / "acts", min_free_gb=cfg.get("min_free_gb", 5.0))


def effects(records: list[dict], fmt: str = "yn") -> dict[str, float]:
    """Per item, D(aligned) - D(swapped) from stage 0."""
    by = {(r["id"], r["goal"]): r["d"] for r in records if r.get("fmt") == fmt}
    return {i: by[(i, "aligned")] - by[(i, "swapped")] for i, g in by if g == "aligned" and (i, "swapped") in by}


def _versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "numpy", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def _gpu_names() -> list[str] | None:
    from wf.provenance import gpu_names

    return gpu_names()


def check_gpu(run_id: str, cfg: dict) -> str | None:
    """None if this job may work on `run_id`, else the reason it may not (amendment 2, B8).

    bfloat16 forward passes differ across GPU types (deviations.md, "Scores across GPU types"), so every stage of
    the pilot runs on the GPU type stage 0 ran on, and a resumed stage on the type its earlier jobs ran on. The
    GPUs are read from each run's jobs.jsonl (one line per job, written by `_init_run`); call this before
    `_init_run` adds this job's line. A job that cannot read its own GPU is refused if any GPU is on record.
    """
    from wf.provenance import JOBS_FILE

    root = runs_root()
    files = [root / run_id / JOBS_FILE] if cfg.get("resume") else []
    stage0 = cfg.get("stage0", STAGE0)
    if run_id != stage0:
        files.append(root / stage0 / JOBS_FILE)
    recorded = sorted({g for f in files if f.exists() for j in read_jsonl(f) for g in (j.get("gpu") or [])})
    if not recorded:
        return None
    if len(recorded) > 1:
        return f"earlier jobs ran on more than one GPU type: {recorded}"
    here = _gpu_names()
    if not here or set(here) != set(recorded):
        return f"this job's GPU {here} is not {recorded[0]}, the GPU of stage 0 and of this stage's earlier jobs"
    return None


def _refuse_gpu(run_id: str, cfg: dict) -> bool:
    problem = check_gpu(run_id, cfg)
    if problem:
        write_status(run_id, f"refused: {problem}; resubmit with --constraint for that GPU type", ok=False)
    return bool(problem)


def _check_sources(run_id: str, cfg: dict) -> bool:
    from wf.rq2 import SOURCES
    from wf.runs import _finished
    from wf.tools.verify_items import verified

    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    unverified = [s for s in sources if not verified(s)[0]]
    if problems or unverified:
        write_status(run_id, f"collection not finished or not verified: {problems + unverified}", ok=False)
        return False
    return True


# ---------------- stage 0 ----------------


def run_interp_pilot_stage0(cfg: dict) -> int:
    import numpy as np

    from wf.interp.analysis import reproduction_gate
    from wf.interp.lens import lens_direction
    from wf.interp.prompts import encode, messages, readout_ids
    from wf.interp.run import baseline, batch_noise, write_manifest
    from wf.runs import _init_run

    from wf.runs import _item_hash

    run_id = cfg["run_id"]
    require_filed(run_id)
    if not cfg.get("skip_source_check") and not _check_sources(run_id, cfg):
        return 1
    if _refuse_gpu(run_id, cfg):
        return 1
    d = _init_run(run_id, cfg)
    logged = logged_scores(cfg)  # before any GPU work: the gate cannot be computed without them
    if not cfg.get("resume") and (d / "acts").exists():
        shutil.rmtree(d / "acts")  # a fresh run never mixes with an earlier attempt's activations
    sets = load_sets(cfg)
    fails, succs = sets["failures"], sets["successes"]
    if cfg.get("limit"):
        fails, succs = fails[: cfg["limit"]], succs[: cfg["limit"]]
    p, tok = load_model(cfg)
    plus, minus = readout_ids(tok, "yn")
    np.save(d / "readout_vec.npy", p.readout_vector(plus, minus).cpu().numpy())
    np.save(d / "lens_dir.npy", lens_direction(p, plus, minus).detach().cpu().numpy())
    first = encode(tok, messages(fails[0], "yn", "aligned"))
    noise = batch_noise(p, first.ids, plus, minus, cfg.get("batch_size", 16))
    write_manifest(d, {"model": cfg.get("model", MODEL_ID), "revision": C.revision_for(cfg.get("model", MODEL_ID)),
                       "n_layers": p.n_layers, "d_model": p.d_model, "readout_ids": [plus, minus],
                       "batch_noise_nats": noise, "versions": _versions(), "n_failures": len(fails),
                       "n_successes": len(succs), "logged_scores_note": LOGGED_NOTE,
                       "norm_eps": float(getattr(p.norm, "variance_epsilon", None) or getattr(p.norm, "eps", 1e-6)),
                       "first_token": bool(cfg.get("first_token", True))})
    min_gb = cfg.get("min_free_gb", 5.0)
    store = baseline(p, tok, fails, d / "acts", formats=("yn",), goals=FAIL_GOALS, min_free_gb=min_gb)
    store = baseline(p, tok, succs, d / "acts", formats=("yn",), goals=SUCC_GOALS, min_free_gb=min_gb)
    if cfg.get("first_token", True):  # amendment 2, B6: the first-token readout, gated; no residuals kept
        store = baseline(p, tok, fails, d / "acts", formats=("obs",), goals=("aligned", "swapped"),
                         min_free_gb=min_gb, save_acts=False)
    recs = store.records()
    tag = cfg.get("model_tag", MODEL_TAG)
    current = {t["id"]: _item_hash(t) for t in fails + succs}
    gates = {"yn": reproduction_gate(recs, logged["yn"], tag, "yn", logged["item_hash"], current)}
    if cfg.get("first_token", True):
        gates["obs"] = reproduction_gate(recs, logged["obs"], tag, "obs", logged["item_hash"], current)
    passed = all(g["passed"] for g in gates.values())
    eff = effects(recs)
    write_json(d / "metrics.json", {"gate": gates["yn"], "gate_first_token": gates.get("obs"), "passed": passed,
                                    "batch_noise_nats": noise, "n_records": len(recs),
                                    "n_items_with_effect": len(eff), "logged_scores_note": LOGGED_NOTE,
                                    "n_clear_effect": sum(v >= cfg.get("min_effect", 1.0) for v in eff.values())})
    parts_ = []
    for fmt, gate in gates.items():
        parts_.append(f"{fmt} gate {'passed' if gate['passed'] else 'FAILED'}: mean effect {gate['mean_effect']:.3f} "
                      f"vs logged {gate['mean_effect_logged_same_items']:.3f} on {gate['n_items']} items; median "
                      f"|D-L| {gate['median_abs_diff']:.3f}; r {gate['r_prompt']:.3f} (prompts), "
                      f"{gate['r_effect']:.3f} (effects); checks {gate['checks']}")
    write_status(run_id, "\n".join(parts_) + f"\nbatch noise {noise:.2e} nats", ok=passed)
    return 0 if passed else 1


# ---------------- stage 2 (minimal): patching ----------------


def scaled_minimum(minimum: int, n: int, reference: int) -> int:
    """A stratum minimum set for `reference` items, scaled down in proportion when fewer are drawn (amendment 3,
    C6), rounded half up: the 10-item timing run of patching gets 3 step-0 items and 3 with an object mention
    instead of minimums larger than the run."""
    if n >= reference:
        return int(minimum)
    return int(math.floor(minimum * n / reference + 0.5))


def patch_selection(cfg: dict, sets: dict, recs: list[dict]) -> tuple[list[dict], dict[str, str], dict[str, float]]:
    """The patching items (clear effect, calibration fold only, stratified, at least `min_step0` at step 0 and
    `min_mention` whose history names the action's object; amendment 2, B3, B4), placebo objects for the first
    `n_placebo` of them that have one (in a seeded order), and each item's weight (population share over sample
    share of its stratum)."""
    import random

    from wf.interp.prompts import placebo_object
    from wf.interp.run import select_stratified

    fails = sets["failures"]
    fold = folds(fails + sets["successes"], cfg.get("split_seed", 0))
    calib = {g for g, f in fold.items() if f == cfg.get("calibration_fold", 0)}
    n, ref = cfg.get("n_items", 100), cfg.get("quota_reference_n", 100)
    chosen, weights = select_stratified(fails, effects(recs), n, cfg.get("min_effect", 1.0),
                                        calib, f"interp|patch|{cfg.get('model_tag', MODEL_TAG)}",
                                        min_step0=scaled_minimum(cfg.get("min_step0", 30), n, ref),
                                        min_mention=scaled_minimum(cfg.get("min_mention", 25), n, ref))
    pool = all_items(cfg) if not cfg.get("placebo_pool_from_sets") else fails
    by_game: dict[str, list[dict]] = {}
    for t in pool:
        by_game.setdefault(t["game_id"], []).append(t)
    order = sorted(chosen, key=lambda t: t["id"])
    random.Random(f"interp|placebo_order|{cfg.get('model_tag', MODEL_TAG)}").shuffle(order)
    placebo: dict[str, str] = {}
    for t in order:
        if len(placebo) >= cfg.get("n_placebo", 50):
            break
        po = placebo_object(t, by_game.get(t["game_id"], []))
        if po:
            placebo[t["id"]] = po
    return chosen, placebo, weights


def run_interp_pilot_patch(cfg: dict) -> int:
    from collections import Counter

    from wf.interp.patching import Cell, grid
    from wf.interp.prompts import PILOT_GROUPS
    from wf.interp.run import patch_items, weighting_stratum, write_manifest
    from wf.runs import _init_run, require_passed

    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, cfg.get("stage0", STAGE0))
    if _refuse_gpu(run_id, cfg):
        return 1
    d = _init_run(run_id, cfg)
    if not cfg.get("resume"):
        (d / "patch.jsonl").unlink(missing_ok=True)
    recs = _stage0_store(cfg).records()
    sets = load_sets(cfg)
    chosen, placebo, weights = patch_selection(cfg, sets, recs)
    p, tok = load_model(cfg)
    layers = cfg.get("layers") or list(range(p.n_layers))
    cells: list[Cell] = [c for c in grid(p.n_layers, ("resid",), tuple(cfg.get("groups", PILOT_GROUPS)))
                         if c.layer in layers]
    mention = bool(cfg.get("min_mention", 25))
    write_manifest(d, {"items": [t["id"] for t in chosen], "placebo": placebo, "n_cells": len(cells),
                       "groups": list(cfg.get("groups", PILOT_GROUPS)), "layers": layers, "weights": weights,
                       "calibration_fold": cfg.get("calibration_fold", 0),
                       "strata": dict(Counter(weighting_stratum(t, mention) for t in chosen))})
    bs = cfg.get("batch_size", 16)
    out = d / "patch.jsonl"
    patch_items(p, tok, chosen, out, cells, fmt="yn", directions=("noise", "denoise"), batch_size=bs)
    patch_items(p, tok, [t for t in chosen if t["id"] in placebo], out, cells, fmt="yn",
                directions=("noise", "denoise"), source_goal="placebo", placebo=placebo, batch_size=bs)
    n = len(read_jsonl(out))
    write_status(run_id, f"{len(chosen)} items, {len(placebo)} placebo pairs, {len(cells)} cells, {n} records",
                 ok=False, reported="patching complete")
    return 0


# ---------------- stage 4 (minimal): direction ablation, cross-fitted ----------------


def ablate_layers(n_layers: int) -> list[int]:
    """The depth profile (amendment 2, B1): every fourth layer from 4 up to 0.8 of depth (Qwen3-8B: 4, 8, ...,
    28). The last layer is not in it (there the ablation removes the mean effect by construction; it is run only
    as the sanity edit `task_last`), nor the near-linear layer before it (`analysis.caveat_layers`, Qwen3-8B 32)."""
    from wf.interp.analysis import profile_layers

    return profile_layers(n_layers)


def answer_projection(lens_dir: Any, succ_rows: Any, fail_rows: Any, succ_verbs: list[str] | None = None,
                      fail_verbs: list[str] | None = None) -> list:
    """The directions projected out of answer-position residuals before a belief probe reads them (amendment 1,
    A1): the readout direction and the fold's (within-verb) success direction."""
    from wf.interp.analysis import success_direction

    sv = success_direction(succ_rows, fail_rows, succ_verbs, fail_verbs)
    return [lens_dir, sv]


def fold_directions(cfg: dict, store: Any, sets: dict, fold: dict[str, int], layers: list[int],
                    random_layers: tuple[int, ...] = (), seed: int = 0, lens_dir: Any = None) -> dict:
    """{fold: {layer: directions}} estimated on each fold's items (`analysis.estimate_directions`); at
    `random_layers` also two random unit directions with their midpoints on the fold's aligned and swapped
    prompts ("random", sanity only).

    The success direction is estimated within verbs, from the fold's determined successes and its failures
    after step 0 under the neutral task (amendment 2, B2): no determined success is at step 0, so step-0
    failures would add a step-0-versus-later contrast to it. With `lens_dir`, the success direction without its
    readout component is added, and every steering amount is also estimated per stratum (amendment 3, C4).
    """
    import numpy as np

    from wf.interp.analysis import estimate_directions, stratum, verb
    from wf.interp.directions import midpoint_for, random_units

    recs = store.records()
    have = {(r["id"], r["goal"]) for r in recs if r.get("fmt", "yn") == "yn"}
    # One streaming pass over the store for every prompt needed, keeping only the ablation layers (float16).
    keys = sorted(f"{i}|yn|{g}" for i, g in have if g in ("aligned", "swapped", "absent", "swapped_plus"))
    rows_, arr = store.load("last", keys, sel=list(layers))
    at = {r["key"]: k for k, r in enumerate(rows_)}
    out: dict[int, dict[int, dict]] = {0: {}, 1: {}}
    for f in (0, 1):
        fails = [t for t in sets["failures"] if fold[t["game_id"]] == f and (t["id"], "aligned") in have
                 and (t["id"], "swapped") in have]
        succ = [t for t in sets["successes"] if fold[t["game_id"]] == f and (t["id"], "absent") in have]
        neutral_f = [t for t in fails if (t["id"], "absent") in have and (t.get("history_len") or 0) > 0]
        plus_ids = [t["id"] for t in fails if (t["id"], "swapped_plus") in have]
        for j, layer in enumerate(layers):
            def rows(ids: list[str], goal: str) -> Any:
                return arr[[at[f"{i}|yn|{goal}"] for i in ids], j].astype(np.float64)
            ids = [t["id"] for t in fails]
            H = {"aligned": rows(ids, "aligned"), "swapped": rows(ids, "swapped")}
            if plus_ids:
                P = np.full_like(H["aligned"], np.nan)
                idx = {i: k for k, i in enumerate(ids)}
                P[[idx[i] for i in plus_ids]] = rows(plus_ids, "swapped_plus")
                H["swapped_plus"] = P
            out[f][layer] = estimate_directions(H, rows([t["id"] for t in succ], "absent"),
                                                rows([t["id"] for t in neutral_f], "absent"),
                                                [verb(t) for t in succ], [verb(t) for t in neutral_f],
                                                readout_dir=lens_dir, strata=[stratum(t) for t in fails])
            if layer in random_layers:
                R = random_units(H["aligned"].shape[1], 2, seed + layer)
                out[f][layer]["random"] = R
                out[f][layer]["random_mid"] = np.array([midpoint_for(u, H["aligned"], H["swapped"]) for u in R])
    return out


def ablation_edits(dirs: dict[int, dict], layers: list[int], caveat: list[int] = (), last: int | None = None,
                   checks: dict | None = None, stratum: str | None = None) -> dict[str, list[tuple[int, Any]]]:
    """Named edits at the answer position, from one fold's directions.

    At every profile layer L: task|L, task_perp_success|L, mention|L, mean ablation at L; task_from|L, the task
    direction ablated at L and at every later profile layer (never at a caveat layer or the last); steer_success|L,
    add the mean aligned-minus-swapped projection on the success direction; steer_success_perp_readout|L, the
    same along the success direction without its readout component (amendment 2, B2); steer_task|L, the same
    along the task direction, the reference the H4 steering is read against (amendment 3, C3); with `stratum`,
    every steering amount is that stratum's, estimated on the other fold (amendment 3, C4); random|L|k, a random unit
    direction, mean-ablated (sanity only, amendment 1 A7). At a caveat layer only task|L; at the last layer only
    task_last|L (amendment 2, B1: tautological there, sanity only). With `checks`, it is filled with
    {name: [(layer, unit vector, target, kind)]} for `run.edit_items` to log the realized edit (amendment 2, B9).
    """
    import numpy as np
    import torch

    from wf.interp.hooks import direction_edit

    def ab(u: Any, c: float):
        uu = torch.tensor(np.asarray(u), dtype=torch.float32)
        return lambda pos: direction_edit(uu, pos, value=float(c))

    def add(u: Any, amount: float):
        uu = torch.tensor(np.asarray(u), dtype=torch.float32)
        return lambda pos: direction_edit(uu, pos, add=float(amount))

    edits: dict[str, list[tuple[int, Any]]] = {}
    chk: dict[str, list] = {} if checks is None else checks

    def put(name: str, parts_: list[tuple[int, Any, float, str]]) -> None:
        edits[name] = [(layer, ab(u, c) if kind == "set" else add(u, c)) for layer, u, c, kind in parts_]
        chk[name] = [(layer, np.asarray(u), float(c), kind) for layer, u, c, kind in parts_]

    for layer in layers:
        x = dirs[layer]
        amt = {**{k: x[k] for k in ("steer_success", "steer_task", "steer_success_perp_readout") if k in x},
               **((x.get("steer_by_stratum") or {}).get(stratum) or {})}
        put(f"task|{layer}", [(layer, x["task"], x["task_mid"], "set")])
        put(f"task_perp_success|{layer}", [(layer, x["task_perp_success"], x["task_perp_success_mid"], "set")])
        if "mention" in x:
            put(f"mention|{layer}", [(layer, x["mention"], x["mention_mid"], "set")])
        put(f"steer_success|{layer}", [(layer, x["success"], amt["steer_success"], "add")])
        put(f"steer_task|{layer}", [(layer, x["task"], amt["steer_task"], "add")])
        if "success_perp_readout" in x:
            put(f"steer_success_perp_readout|{layer}",
                [(layer, x["success_perp_readout"], amt["steer_success_perp_readout"], "add")])
        put(f"task_from|{layer}", [(m, dirs[m]["task"], dirs[m]["task_mid"], "set") for m in layers if m >= layer])
        for k, (u, c) in enumerate(zip(x.get("random", []), x.get("random_mid", []))):
            put(f"random|{layer}|{k}", [(layer, u, c, "set")])
    for layer in caveat:
        put(f"task|{layer}", [(layer, dirs[layer]["task"], dirs[layer]["task_mid"], "set")])
    if last is not None:
        put(f"task_last|{last}", [(last, dirs[last]["task"], dirs[last]["task_mid"], "set")])
    return edits


def ablation_selection(cfg: dict, sets: dict, recs: list[dict], fold: dict[str, int],
                       folds_: tuple[int, ...] = (0, 1)) -> tuple[dict[int, list[dict]], dict[str, float]]:
    """Per fold in `folds_`: `n_failures_per_fold` failures (at least `min_step0_per_fold` at step 0, amendment 2,
    B4) and `n_successes_per_fold` successes, stratified by failure reason and step 0 (seeded); only items stage
    0 scored under every goal used. Also each item's weight (population share over sample share)."""
    from wf.interp.run import select_stratified

    have = {(r["id"], r["goal"]) for r in recs if r.get("fmt", "yn") == "yn"}
    out, weights = {}, {}
    for f in folds_:
        games = {g for g, x in fold.items() if x == f}
        fails = [t for t in sets["failures"] if all((t["id"], g) in have for g in ("aligned", "swapped", "absent"))]
        succ = [t for t in sets["successes"] if all((t["id"], g) in have for g in SUCC_GOALS)]
        anyeff = {t["id"]: 0.0 for t in fails + succ}
        nf = cfg.get("n_failures_per_fold", 75)
        fl, wf_ = select_stratified(fails, anyeff, nf, float("-inf"), games, f"interp|ablate|fail|{f}",
                                    min_step0=scaled_minimum(cfg.get("min_step0_per_fold", 20), nf,
                                                             cfg.get("quota_reference_n", 75)))
        sc, ws = select_stratified(succ, anyeff, cfg.get("n_successes_per_fold", 50), float("-inf"), games,
                                   f"interp|ablate|succ|{f}")
        out[f] = fl + sc
        weights.update(wf_)
        weights.update(ws)
    return out, weights


def fold_cosines(dirs: dict[int, dict], lens_dir: Any, layers: list[int]) -> dict:
    """Cosines per fold and layer (`analysis.direction_cosines`) and their null baselines across folds
    (amendment 2, B2): the two folds' task directions (how far an estimate agrees with itself; no cosine with
    it can be read above this), the two folds' success directions, and one fold's task direction with the other
    fold's success direction after the readout direction is projected out of both (no shared estimation
    noise). A random pair of directions in d dimensions has cosine about +-1/sqrt(d)."""
    import numpy as np

    from wf.interp.analysis import direction_cosines, disattenuated, project_unit
    from wf.interp.directions import cosine

    cos: dict[str, Any] = {f"{f}|{layer}": direction_cosines(dirs[f][layer], lens_dir) for f in dirs for layer in dirs[f]}
    cos.update({f"between_folds|{layer}": float(np.dot(dirs[0][layer]["task"], dirs[1][layer]["task"]))
                for layer in layers})
    cos.update({f"between_folds_success|{layer}": float(np.dot(dirs[0][layer]["success"], dirs[1][layer]["success"]))
                for layer in layers})
    # Amendment 3, C3: the reliabilities of the projected directions (each the cosine of its two folds'
    # estimates after the readout direction is projected out) and the H4 cosine disattenuated by them.
    for key in ("task", "success", "mention"):
        cos.update({f"between_folds_{key}_perp_readout|{layer}":
                    cosine(project_unit(dirs[0][layer][key], lens_dir), project_unit(dirs[1][layer][key], lens_dir))
                    for layer in layers if key in dirs[0][layer] and key in dirs[1][layer]})
    for f in (0, 1):
        for layer in layers:
            c = cos[f"{f}|{layer}"]
            for a in ("task", "mention"):
                rel_a = cos.get(f"between_folds_{a}_perp_readout|{layer}")
                rel_s = cos.get(f"between_folds_success_perp_readout|{layer}")
                if f"{a}_success_perp_readout" in c:
                    c[f"{a}_success_perp_readout_disattenuated"] = disattenuated(c[f"{a}_success_perp_readout"],
                                                                                 rel_a, rel_s)
    for f in (0, 1):
        cos.update({f"cross_fold_task_success_perp_readout|{f}|{layer}":
                    cosine(project_unit(dirs[f][layer]["task"], lens_dir),
                           project_unit(dirs[1 - f][layer]["success"], lens_dir)) for layer in layers})
    width = len(np.asarray(lens_dir))
    cos["random_pair_sd"] = float(1 / np.sqrt(width))
    return cos


def ablation_layer_sets(cfg: dict, n_layers: int) -> tuple[list[int], list[int], int | None]:
    """(profile layers, caveat layers, last layer) for the ablation: the configured or default profile; caveat
    layers that are neither profile layers nor the last layer; the last layer as the sanity edit."""
    from wf.interp.analysis import caveat_layers

    layers = cfg.get("layers") or ablate_layers(n_layers)
    caveat = [m for m in cfg.get("caveat_layers", caveat_layers(n_layers)) if m not in layers and m < n_layers - 1]
    last = n_layers - 1 if cfg.get("last_layer_sanity", True) else None
    return layers, caveat, last


def run_interp_pilot_ablate(cfg: dict) -> int:
    import numpy as np

    from wf.interp.analysis import stratum
    from wf.interp.run import edit_items, write_manifest
    from wf.runs import _init_run, require_passed

    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, cfg.get("stage0", STAGE0))
    if _refuse_gpu(run_id, cfg):
        return 1
    d = _init_run(run_id, cfg)
    if not cfg.get("resume"):
        (d / "ablate.jsonl").unlink(missing_ok=True)
    store = _stage0_store(cfg)
    recs = store.records()
    sets = load_sets(cfg)
    fold = folds(sets["failures"] + sets["successes"], cfg.get("split_seed", 0))
    s0 = json.loads((runs_root() / cfg.get("stage0", STAGE0) / "manifest.json").read_text())
    n_layers = s0["n_layers"]
    layers, caveat, last = ablation_layer_sets(cfg, n_layers)
    default_random = [layers[len(layers) // 2], layers[-1]] if len(layers) >= 2 else layers
    random_layers = tuple(sorted({m for m in cfg.get("random_layers", default_random) if m in layers}))
    every = sorted(set(layers) | set(caveat) | ({last} if last is not None else set()))
    lens_dir = np.load(runs_root() / cfg.get("stage0", STAGE0) / "lens_dir.npy")
    dirs = fold_directions(cfg, store, sets, fold, every, random_layers, cfg.get("seed", C.RNG_SEED), lens_dir)
    np.savez(d / "directions.npz", **{f"{f}|{layer}|{k}": np.asarray(v) for f in dirs for layer in dirs[f]
                                     for k, v in dirs[f][layer].items()})
    cos = fold_cosines(dirs, lens_dir, every)
    # Amendment 2, B3: the minimal pilot edits the calibration fold's items only (with the other fold's
    # directions); the other fold's items are not patched or edited before the full pilot.
    edit_folds = tuple(cfg.get("edit_folds", [cfg.get("calibration_fold", 0)]))
    sel, weights = ablation_selection(cfg, sets, recs, fold, edit_folds)
    write_manifest(d, {"layers": layers, "caveat_layers": caveat, "last_layer": last,
                       "random_layers": list(random_layers), "cosines": cos, "edit_folds": list(edit_folds),
                       "items": {str(f): [t["id"] for t in sel[f]] for f in sel}, "weights": weights,
                       "fold_of_game": fold})
    p, tok = load_model(cfg)
    realized: dict[str, Any] = {}
    for f in edit_folds:
        for st in ("step0", "later"):  # amendment 3, C4: each item is steered by its own stratum's amounts
            items = [t for t in sel[f] if stratum(t) == st]
            if not items:
                continue
            checks: dict = {}
            edits = ablation_edits(dirs[1 - f], layers, caveat, last, checks, stratum=st)
            out = edit_items(p, tok, items, d / "ablate.jsonl", edits, fmt="yn", goals=("aligned", "swapped", "absent"),
                             positions="last", batch_size=cfg.get("batch_size", 16), checks=checks)
            ids = {t["id"] for t in items}
            first = next((r for r in read_jsonl(out) if "edit_realized" in r and r["id"] in ids), None)
            realized[f"{f}|{st}"] = first["edit_realized"] if first else None
    worst = max((abs(v) for x in realized.values() if x for vals in x.values() for v in vals), default=float("nan"))
    write_json(d / "edit_checks.json", {"realized": realized, "max_abs": worst,
                                        "what": "u.h - target after each edit on the first edited prompt, at the "
                                                "answer position in the model's dtype (amendment 2, B9)"})
    n = len(read_jsonl(d / "ablate.jsonl"))
    write_status(run_id, f"{sum(len(v) for v in sel.values())} items (folds {list(edit_folds)}), {n} records; "
                 f"largest realized edit error {worst:.3g}", ok=False, reported="ablation complete")
    return 0


# ---------------- analysis (CPU) ----------------


def run_interp_pilot_analysis(cfg: dict) -> int:
    import numpy as np

    from wf.interp.analysis import STRATA, h4_layers, in_stratum, patch_summary
    from wf.runs import _init_run, require_passed

    run_id = cfg["run_id"]
    require_filed(run_id)
    for dep in (cfg.get("stage0", STAGE0), cfg.get("patch", "interp_pilot_patch"), cfg.get("ablate", "interp_pilot_ablate")):
        require_passed(run_id, cfg, dep)
    d = _init_run(run_id, {**cfg, "resume": False})
    root = runs_root()
    s0dir = root / cfg.get("stage0", STAGE0)
    s0 = json.loads((s0dir / "manifest.json").read_text())
    n_layers = s0["n_layers"]
    B = cfg.get("bootstrap", 2000)
    report: dict[str, Any] = {"stage0": json.loads((s0dir / "metrics.json").read_text()), "bootstrap": B,
                              "bonferroni": "normal intervals from the bootstrap SE (amendment 2, B5)",
                              "note": "calibration readouts against provisional bands; no verdicts (section M)"}

    pdir = root / cfg.get("patch", "interp_pilot_patch")
    patch = read_jsonl(pdir / "patch.jsonl")
    pman = json.loads((pdir / "manifest.json").read_text())
    pw = pman.get("weights")
    pat: dict[str, Any] = {"weights_by_stratum": pman.get("strata")}
    for direction in ("noise", "denoise"):
        goal = [r for r in patch if r["direction"] == direction and r["source_goal"] == "swapped"]
        plac = [r for r in patch if r["direction"] == direction and r["source_goal"] == "placebo"]
        pat[direction] = {s: patch_summary([r for r in goal if in_stratum(r, s)], [r for r in plac if in_stratum(r, s)],
                                           n_layers, B, C.RNG_SEED, weights=pw) for s in STRATA}
        for s in STRATA:  # amendment 3, C1: the arrival layer under the onset's rule, on the answer position
            pat[direction][s]["arrival_last_grid"] = arrival_on_answer_position(pat[direction][s], n_layers)
    report["patching"] = pat

    store = _stage0_store(cfg)
    lens_dir = np.load(s0dir / "lens_dir.npy")
    aman = json.loads((root / cfg.get("ablate", "interp_pilot_ablate") / "manifest.json").read_text())
    sets = load_sets(cfg)
    succ_ids = {t["id"] for t in sets["successes"]}
    abl_fails = [t for t in sets["failures"] if any(t["id"] in ids for ids in aman["items"].values())]
    report["direct_path"] = direct_path_report(store, abl_fails, aman["layers"], n_layers, lens_dir,
                                               s0.get("norm_eps", 1e-6), aman.get("weights"), B)
    report["ablation"] = ablation_report(cfg, B, report["direct_path"])
    report["cosines"] = aman["cosines"]
    fold = folds(sets["failures"] + sets["successes"], cfg.get("split_seed", 0))
    report["h4_cosines"] = h4_cosine_report(store, sets, fold, h4_layers(aman["layers"], n_layers), lens_dir,
                                            cfg.get("bootstrap_cosine", B))
    if cfg.get("hidden_probe", True):
        report["hidden_probe"] = hidden_probe_report(cfg)
    report["table_rows"] = table_rows(report, n_layers)
    report["sample_size"] = sample_sizes(report)
    write_json(d / "metrics.json", report)
    lines = []
    for s in STRATA:
        hs = pat["noise"][s]["history_share"]
        ab = report["ablation"][s]
        lines.append(f"{s}: history share {hs.get('max', float('nan')) if hs.get('defined') else 'undefined'}; "
                     f"arrival (answer position) {pat['noise'][s]['arrival_last_grid']}; onset layer "
                     f"{ab.get('onset_layer')} (read on {ab.get('onset_layers')}); "
                     f"{report['table_rows'][s]['onset_versus_arrival']}")
    write_status(run_id, "\n".join(lines), ok=False, reported="minimal interpretability pilot (calibration)")
    return 0


def arrival_on_answer_position(summary: dict, n_layers: int) -> int | None:
    """The patching arrival layer compared with the onset (amendment 3, C1): the `last` group's noising curve
    (the answer position, where the ablation acts) under the onset's rule, on the grid of every fourth layer."""
    from wf.interp.analysis import grid_layers, patch_arrival_layer

    return patch_arrival_layer(summary.get("curves", {}).get("last", {}), grid_layers(n_layers))


def direct_path_report(store: Any, fails: list[dict], layers: list[int], n_layers: int, lens_dir: Any, eps: float,
                       weights: dict[str, float] | None, B: int = 2000) -> dict:
    """{stratum: {layer: direct-path share}} on the ablation's failure items (amendment 3, C2), from stage 0's
    answer-position residuals: r.h at each profile layer over the prompt's last-layer RMS, aligned minus swapped,
    summed, over the summed final readout effect (`analysis.direct_path_share`)."""
    import numpy as np

    from wf.interp.analysis import STRATA, direct_path_share, in_stratum

    recs = {(r["id"], r["goal"]): r for r in store.records() if r.get("fmt", "yn") == "yn"}
    fails = [t for t in fails if (t["id"], "aligned") in recs and (t["id"], "swapped") in recs]
    if not fails:
        return {}
    sel = list(layers) + [n_layers - 1]
    keys = [f"{t['id']}|yn|{g}" for t in fails for g in ("aligned", "swapped")]
    _, arr = store.load("last", keys, sel=sel)
    H = np.asarray(arr, dtype=np.float64).reshape(len(fails), 2, len(sel), -1)
    rho = np.sqrt((H[:, :, -1] ** 2).mean(-1) + eps)  # (items, 2): the final norm's scale per prompt
    proj = (H[:, :, :-1] @ np.asarray(lens_dir, dtype=np.float64)) / rho[:, :, None]  # (items, 2, layers)
    D = np.array([[recs[(t["id"], g)]["d"] for g in ("aligned", "swapped")] for t in fails])
    out: dict[str, Any] = {}
    for s in STRATA:
        idx = [k for k, t in enumerate(fails) if in_stratum(t, s)]
        if not idx:
            continue
        ids = [fails[k]["id"] for k in idx]
        games = [fails[k]["game_id"] for k in idx]
        out[s] = {m: direct_path_share(proj[idx, 0, j], proj[idx, 1, j], D[idx, 0], D[idx, 1], games, ids, weights,
                                       B, C.RNG_SEED) for j, m in enumerate(layers)}
    return out


def h4_cosine_report(store: Any, sets: dict, fold: dict[str, int], layers: list[int], lens_dir: Any,
                     B: int = 2000) -> dict:
    """Per fold and H4 layer (amendment 3, C3): the projected H4 cosine with a game-bootstrap interval, with the
    readout direction projected out ("readout") and with the readout direction and the layer's own answer
    direction projected out ("readout_and_answer"; `analysis.answer_direction`, fit on the fold's neutral
    prompts within outcome and verb)."""
    import numpy as np

    from wf.interp.analysis import answer_direction, projected_cosine_boot, verb

    recs = {(r["id"], r["goal"]): r for r in store.records() if r.get("fmt", "yn") == "yn"}
    if not layers:
        return {}
    out: dict[str, Any] = {}
    for f in (0, 1):
        fl = [t for t in sets["failures"] if fold[t["game_id"]] == f and (t["id"], "aligned") in recs
              and (t["id"], "swapped") in recs]
        nf = [t for t in fl if (t["id"], "absent") in recs and (t.get("history_len") or 0) > 0]
        sc = [t for t in sets["successes"] if fold[t["game_id"]] == f and (t["id"], "absent") in recs]
        if not fl or not nf or not sc:
            continue
        keys = ([f"{t['id']}|yn|aligned" for t in fl] + [f"{t['id']}|yn|swapped" for t in fl]
                + [f"{t['id']}|yn|absent" for t in nf] + [f"{t['id']}|yn|absent" for t in sc])
        _, arr = store.load("last", keys, sel=list(layers))
        n1, n2, n3 = len(fl), len(nf), len(sc)
        for j, m in enumerate(layers):
            X = np.asarray(arr[:, j], dtype=np.float64)
            A_, S_, F_, Sc = X[:n1], X[n1:2 * n1], X[2 * n1:2 * n1 + n2], X[2 * n1 + n2:]
            neutral = np.vstack([F_, Sc])
            Dn = [recs[(t["id"], "absent")]["d"] for t in nf + sc]
            ans = answer_direction(neutral, Dn, [False] * n2 + [True] * n3, [verb(t) for t in nf + sc])
            out[f"{f}|{m}"] = projected_cosine_boot(
                A_ - S_, [t["game_id"] for t in fl], Sc, [verb(t) for t in sc], [t["game_id"] for t in sc],
                F_, [verb(t) for t in nf], [t["game_id"] for t in nf],
                {"readout": [lens_dir], "readout_and_answer": [lens_dir, ans]}, B, C.RNG_SEED + f)
    return out


def ablation_report(cfg: dict, B: int = 2000, direct: dict | None = None) -> dict:
    """Cross-fitted ablation readouts per stratum (`analysis.ablation_summary`) from the ablation run's files."""
    from wf.interp.analysis import ablation_summary

    root = runs_root()
    ad = root / cfg.get("ablate", "interp_pilot_ablate")
    man = json.loads((ad / "manifest.json").read_text())
    recs = read_jsonl(ad / "ablate.jsonl")
    fold_of = {i: int(f) for f, ids in man["items"].items() for i in ids}
    sets = load_sets(cfg)
    succ_ids = {t["id"] for t in sets["successes"]}
    return ablation_summary(recs, fold_of, succ_ids, man["layers"], man.get("caveat_layers", []),
                            man.get("last_layer"), man.get("weights"), B, C.RNG_SEED, direct)


def _median(xs: list[float]) -> float:
    import numpy as np

    v = [x for x in xs if x is not None and x == x]
    return float(np.median(v)) if v else float("nan")


def h4_rows(ab: dict, cos: dict, layers: list[int], boot: dict | None = None) -> dict:
    """The H4 rows (amendment 2, B2; amendment 3, C3) at `layers` (up to two thirds of depth and strictly before
    the patching arrival layer), each next to its null baselines. Medians over the layers; cosines averaged over
    the two folds' estimates where both exist. The goal-role row (task against mention cosine) is reported apart:
    a mention is itself an object match, so it is not a null for H4."""
    boot = boot or {}

    def fold_mean(key: str, layer: int) -> float:
        v = [cos.get(f"{f}|{layer}", {}).get(key) for f in (0, 1)]
        v = [x for x in v if x is not None and x == x]
        return sum(v) / len(v) if v else float("nan")

    by = ab.get("by_layer", {})

    def get(m: int, kind: str) -> dict:
        return (by.get(m) or by.get(str(m)) or {}).get(kind, {}) or {}

    def boot_mean(m: int, which: str, part: str) -> Any:
        v = [boot.get(f"{f}|{m}", {}).get(which, {}).get(part) for f in (0, 1)]
        v = [x for x in v if x is not None]
        if not v:
            return float("nan") if part != "ci" else (float("nan"), float("nan"))
        if part == "ci":
            return (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v))
        return sum(v) / len(v)

    perp = [get(m, "perp_over_task_removal") for m in layers]
    assessable = [x.get("ratio") for x in perp if x.get("assessable")]
    return {
        "layers": layers,
        "cos_task_success_perp_readout": _median([fold_mean("task_success_perp_readout", m) for m in layers]),
        "cos_task_success_perp_readout_ci_by_layer": {m: boot_mean(m, "readout", "ci") for m in layers},
        "cos_task_success_perp_readout_se": _median([boot_mean(m, "readout", "se") for m in layers]),
        "cos_task_success_perp_readout_and_answer": _median([boot_mean(m, "readout_and_answer", "cos") for m in layers]),
        "cos_task_success_perp_readout_and_answer_ci_by_layer": {m: boot_mean(m, "readout_and_answer", "ci")
                                                                 for m in layers},
        "cos_task_success_perp_readout_disattenuated": _median(
            [fold_mean("task_success_perp_readout_disattenuated", m) for m in layers]),
        "null_between_folds_task": _median([cos.get(f"between_folds|{m}") for m in layers]),
        "null_between_folds_task_perp_readout": _median([cos.get(f"between_folds_task_perp_readout|{m}") for m in layers]),
        "null_between_folds_success_perp_readout": _median(
            [cos.get(f"between_folds_success_perp_readout|{m}") for m in layers]),
        "null_between_folds_mention_perp_readout": _median(
            [cos.get(f"between_folds_mention_perp_readout|{m}") for m in layers]),
        "cross_fold_task_success_perp_readout": _median(
            [(cos.get(f"cross_fold_task_success_perp_readout|0|{m}", float("nan"))
              + cos.get(f"cross_fold_task_success_perp_readout|1|{m}", float("nan"))) / 2 for m in layers]),
        "null_random_pair_sd": cos.get("random_pair_sd"),
        "cos_task_success_raw": _median([fold_mean("task_success", m) for m in layers]),
        "steer_perp_over_task": _median([get(m, "steer_perp_over_task").get("ratio") for m in layers]),
        "steer_perp_over_task_se": _median([get(m, "steer_perp_over_task").get("se") for m in layers]),
        "steer_success_perp_readout_share": _median([get(m, "steer_success_perp_readout").get("ratio") for m in layers]),
        "steer_task_share": _median([get(m, "steer_task").get("ratio") for m in layers]),
        "steer_success_raw_share": _median([get(m, "steer_success").get("ratio") for m in layers]),
        "perp_success_over_task_removal": _median(assessable) if assessable else "not assessable",
        "perp_success_over_task_removal_layers": [m for m, x in zip(layers, perp) if x.get("assessable")],
        "goal_role": {
            "cos_mention_success_perp_readout": _median([fold_mean("mention_success_perp_readout", m) for m in layers]),
            "cos_mention_success_perp_readout_disattenuated": _median(
                [fold_mean("mention_success_perp_readout_disattenuated", m) for m in layers]),
            "task_minus_mention": _median([fold_mean("task_success_perp_readout", m)
                                           - fold_mean("mention_success_perp_readout", m) for m in layers]),
        },
    }


def table_rows(report: dict, n_layers: int | None = None) -> dict:
    """The outcome-table rows the minimal pilot can fill, per stratum, as readings against the provisional bands
    (amendment 1, A6; amendment 2, B1, B2; amendment 3, C1). No hypothesis is named and no verdict drawn: the
    minimal pilot only calibrates (section M). Rows it cannot fill are listed."""
    from wf.interp.analysis import h4_layers, onset_versus_arrival

    rows: dict[str, Any] = {}
    for s in ("step0", "later", "all"):
        noise = report["patching"]["noise"][s]
        ab = report["ablation"].get(s, {})
        hs = noise["history_share"]
        arrival = noise["arrival_layer"]
        arrival_index = round(arrival * (n_layers - 1), 6) if n_layers and arrival == arrival else float("nan")
        arrival_last = noise.get("arrival_last_grid")
        layers = report["ablation"].get("layers", [])
        onset = ab.get("onset_layer")
        task = ab.get("task", {})
        g = ab.get("gap_neutral", {})
        h4l = h4_layers(layers, n_layers or 0, arrival_last) if n_layers else []
        rows[s] = {
            "history_share": (None if not hs.get("defined") else
                              "high (>=0.3)" if hs["ci"][0] >= 0.3 else "low (<0.15)" if hs["ci"][1] < 0.15 else "between"),
            "arrival_layer_final_group_half": arrival, "arrival_layer_final_group_half_index": arrival_index,
            "arrival_layer_answer_position": arrival_last,
            "single_direction_onset_layer": onset, "onset_read_on_layers": ab.get("onset_layers"),
            "onset_versus_arrival": onset_versus_arrival(onset, arrival_last,
                                                         ab["onset_layers"] if "onset_layers" in ab else layers,
                                                         ab.get("deepest_upper")),
            "task_last_sanity_share": ab.get("task_last", {}).get("share"),
            "task_direction_removed_at_chosen_layer": task.get("share"),
            "neutral_gap_ratio": g.get("ratio"),
            "discrimination_kept": None if g.get("ratio") is None else g["ratio"] >= 0.9,
            "h4": h4_rows(ab, report.get("cosines") or {}, h4l, report.get("h4_cosines")),
            "not_filled_by_minimal_pilot": ["belief shift (history-fact probes need stage 1)", "head-level rows",
                                            "policy-direction transfer", "described-goal transfer",
                                            "first-token and full-reply patching", "Llama"],
        }
    return rows


def sample_sizes(report: dict) -> dict:
    """The calibration rule's output (amendment 2, B3; amendment 3, C6): per stratum, each summary measure's
    game-bootstrap SE at the minimal pilot's n, and the n the full pilot needs for a 95% half-width of 0.075
    (times the margin of 1.2). The bands are not moved."""
    from wf.interp.analysis import SAMPLE_SIZE_MARGIN, TARGET_HALF_WIDTH, n_for_half_width

    def nn(n: Any, se: Any) -> dict:
        se = float("nan") if se is None else se
        return {"n": n, "se": se, "n_needed": n_for_half_width(n or 0, se)}

    out: dict[str, Any] = {"target_half_width": TARGET_HALF_WIDTH, "margin": SAMPLE_SIZE_MARGIN}
    for s in ("step0", "later", "all"):
        noise = report["patching"]["noise"][s]
        ab = report["ablation"].get(s, {})
        n_p = noise["n_items"]
        row: dict[str, Any] = {}
        for name in ("history_share", "action_share", "object_mention_share"):
            x = noise.get(name, {})
            row[name] = nn(n_p, x.get("se") if x.get("defined", True) else None)
        last = noise.get("curves", {}).get("last", {})
        a = noise.get("arrival_last_grid")
        row["arrival_answer_position"] = nn(n_p, (last.get("se") or [None] * (a + 1))[a] if a is not None else None)
        row["removal_by_layer"] = {m: nn(v.get("n"), v.get("se")) for m, v in ab.get("profile", {}).items()}
        g = ab.get("gap_neutral", {})
        row["gap_ratio"] = nn(ab.get("n_failures", 0) + ab.get("n_successes", 0), g.get("ratio_se"))
        h4 = report.get("table_rows", {}).get(s, {}).get("h4", {})
        by = ab.get("by_layer", {})
        row["h4"] = {"cos_task_success_perp_readout": nn(ab.get("n_failures"), h4.get("cos_task_success_perp_readout_se")),
                     "steer_perp_over_task": nn(ab.get("n_failures"), h4.get("steer_perp_over_task_se")),
                     "perp_over_task_removal": nn(ab.get("n_failures"), _median(
                         [(by.get(m) or {}).get("perp_over_task_removal", {}).get("se") for m in h4.get("layers", [])]))}
        out[s] = row
    return out


def hidden_probe_report(cfg: dict) -> dict:
    """The hidden-state belief probe (amendment 1, A4), cross-fitted over the two folds of games.

    Label: the action's object was at the action's receptacle at reset (`probes.hidden_labels`). Trained on
    neutral-task prompts (balanced within true outcome), at the action's receptacle token and at the answer
    position after projecting out the readout direction and the fold's success direction; tested on the other
    fold: AUC on neutral prompts, then the standardised shift aligned minus swapped on the failures (positive =
    toward "the object is there"), and at the answer position also the shift left after removing what the
    readout's own change predicts. Descriptive in the minimal pilot: its stage 0 has only RQ2's items and the
    determined successes, not the full neutral probe set of stage 1.
    """
    import numpy as np

    from wf.interp.analysis import in_stratum, verb
    from wf.interp.probes import (balanced_indices, belief_shifts, fit_probe, hidden_labels, probe_auc,
                                  probe_logit, project_out, readout_slope, residual_shifts, separation)

    store = _stage0_store(cfg)
    recs = [r for r in store.records() if r.get("fmt", "yn") == "yn"]
    have = {(r["id"], r["goal"]): r for r in recs}
    sets = load_sets(cfg)
    fold = folds(sets["failures"] + sets["successes"], cfg.get("split_seed", 0))
    succ_ids = {t["id"] for t in sets["successes"]}
    items = [t for t in sets["failures"] + sets["successes"] if (t["id"], "absent") in have]
    lab = {t["id"]: hidden_labels(t)["object_at_receptacle"] for t in items}
    items = [t for t in items if lab[t["id"]] is not None]
    if not items:
        return {"n": 0}
    pl = recs[0]["probe_layers"]
    lens_dir = np.load(runs_root() / cfg.get("stage0", STAGE0) / "lens_dir.npy")
    out: dict[str, Any] = {"n_labelled": len(items), "share_true": float(np.mean([lab[t["id"]] for t in items])),
                           "layers": pl, "by_layer": {}}

    # One streaming pass per array, keeping the probe layers only (float16): the receptacle token's residual and
    # the answer position's, for the neutral, aligned and swapped prompts of the labelled items.
    keys = sorted(f"{t['id']}|yn|{g}" for t in items for g in ("absent", "aligned", "swapped") if (t["id"], g) in have)
    krows, rec_arr = store.load("probe", keys, sel=(slice(None), 1))
    _, ans_arr = store.load("last", keys, sel=list(pl))
    at = {r["key"]: n for n, r in enumerate(krows)}

    def X(ids: list[str], goal: str, pos: str, k: int, layer: int) -> Any:
        idx = [at[f"{i}|yn|{goal}"] for i in ids]
        return (rec_arr if pos == "rec" else ans_arr)[idx, k].astype(np.float64)

    for k, layer in enumerate(pl):
        row: dict[str, Any] = {}
        for pos in ("rec", "answer"):
            aucs, shifts, resid = [], [], []
            for f in (0, 1):
                tr = [t for t in items if fold[t["game_id"]] != f]
                te = [t for t in items if fold[t["game_id"]] == f]
                y = [bool(lab[t["id"]]) for t in tr]
                keep = balanced_indices(y, [t["id"] in succ_ids for t in tr], seed=f)
                if len(keep) < 20 or not te:
                    continue
                tr = [tr[i] for i in keep]
                Xtr = X([t["id"] for t in tr], "absent", pos, k, layer)
                proj = None
                if pos == "answer":
                    st = [t for t in tr if t["id"] in succ_ids]
                    ft = [t for t in tr if t["id"] not in succ_ids and (t.get("history_len") or 0) > 0]
                    proj = answer_projection(lens_dir, X([t["id"] for t in st], "absent", pos, k, layer),
                                             X([t["id"] for t in ft], "absent", pos, k, layer),
                                             [verb(t) for t in st], [verb(t) for t in ft])
                    Xtr = project_out(Xtr, proj)
                ytr = [bool(lab[t["id"]]) for t in tr]
                if len(set(ytr)) < 2:
                    continue
                m = fit_probe(Xtr, ytr, [t["game_id"] for t in tr], seed=f)
                Xte = X([t["id"] for t in te], "absent", pos, k, layer)
                Xte = project_out(Xte, proj) if proj else Xte
                yte = [bool(lab[t["id"]]) for t in te]
                if len(set(yte)) < 2:
                    continue
                aucs.append(probe_auc(m, Xte, yte))
                sep = separation(m, Xte, yte)
                fl = [t for t in te if t["id"] not in succ_ids and (t["id"], "swapped") in have and (t["id"], "aligned") in have]
                if not fl or not sep:
                    continue
                Xa, Xs = X([t["id"] for t in fl], "aligned", pos, k, layer), X([t["id"] for t in fl], "swapped", pos, k, layer)
                if proj:
                    Xa, Xs = project_out(Xa, proj), project_out(Xs, proj)
                up = [True] * len(fl)
                sh = belief_shifts(m, Xa, Xs, sep, up)
                shifts += [(x, t) for x, t in zip(sh, fl)]
                if pos == "answer":
                    slope = readout_slope(probe_logit(m, Xte), [have[(t["id"], "absent")]["d"] for t in te])
                    rs = residual_shifts(m, Xa, Xs, [have[(t["id"], "aligned")]["d"] for t in fl],
                                         [have[(t["id"], "swapped")]["d"] for t in fl], slope, sep, up)
                    resid += [(x, t) for x, t in zip(rs, fl)]
            cell: dict[str, Any] = {"auc": float(np.mean(aucs)) if aucs else float("nan")}
            for st in ("step0", "later", "all"):
                v = [x for x, t in shifts if in_stratum(t, st) and x == x]
                cell[f"shift_{st}"] = float(np.mean(v)) if v else float("nan")
                if pos == "answer":
                    v = [x for x, t in resid if in_stratum(t, st) and x == x]
                    cell[f"residual_shift_{st}"] = float(np.mean(v)) if v else float("nan")
            row[pos] = cell
        out["by_layer"][layer] = row
    return out
