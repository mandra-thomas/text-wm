"""The minimal interpretability pilot (notes/interp_design.md, amendment 1, section M): Qwen3-8B, yes/no format.

Runs (wired in `wf.runs`):

- `interp_pilot_stage0` (GPU, about 1 A100-hour): readouts and answer-position residuals at every layer for RQ2's
  determined failures (aligned, swapped, neutral and swapped-plus-mention prompts) and the determined successes
  (aligned, swapped, neutral); the batch noise; the reproduction gate against the logged rq1v scores. Every later
  run requires this one to have passed.
- `interp_pilot_patch` (GPU, about 3 A100-hours): residual-stream patching at every layer by the pilot's groups,
  noising and denoising, on 100 clear-effect items of the calibration fold's games and 50 placebo pairs.
- `interp_pilot_ablate` (GPU, about 1.5 A100-hours): directions estimated on one fold of games and applied at the
  answer position to the other fold's items (two-fold cross-fitting), at nine layers.
- `interp_pilot_analysis` (CPU): the pre-registered readouts per stratum (step 0, later, all).

`load_model` and `load_sets` are module functions so the tests can substitute a tiny model and synthetic items.
"""
from __future__ import annotations

import json
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


def logged_scores(cfg: dict) -> dict[tuple[str, str], float]:
    """{(item id, arm): L} of the yes/no arms in the logged rq1v scoring run, for this model."""
    from wf.rq2 import complete_shards

    folders, problems = complete_shards(cfg.get("logged_run", "rq1v_score"))
    if problems:
        raise SystemExit(f"logged scores incomplete: {problems}")
    tag = cfg.get("model_tag", MODEL_TAG)
    return {(s["transition_id"], s["condition"]): s["L"] for f in folders for s in read_jsonl(f / "scoring.jsonl")
            if s["model"] == tag and s["condition"].startswith("yn_") and s["L"] == s["L"]}


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

    run_id = cfg["run_id"]
    require_filed(run_id)
    if not cfg.get("skip_source_check") and not _check_sources(run_id, cfg):
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
                       "n_successes": len(succs)})
    min_gb = cfg.get("min_free_gb", 5.0)
    store = baseline(p, tok, fails, d / "acts", formats=("yn",), goals=FAIL_GOALS, min_free_gb=min_gb)
    store = baseline(p, tok, succs, d / "acts", formats=("yn",), goals=SUCC_GOALS, min_free_gb=min_gb)
    recs = store.records()
    gate = reproduction_gate(recs, logged, cfg.get("model_tag", MODEL_TAG))
    eff = effects(recs)
    write_json(d / "metrics.json", {"gate": gate, "batch_noise_nats": noise, "n_records": len(recs),
                                    "n_items_with_effect": len(eff),
                                    "n_clear_effect": sum(v >= cfg.get("min_effect", 1.0) for v in eff.values())})
    body = (f"gate {'passed' if gate['passed'] else 'FAILED'}: mean effect {gate['mean_effect']:.3f} vs logged "
            f"{gate['mean_effect_logged_same_items']:.3f} on {gate['n_items']} items; median |D-L| "
            f"{gate['median_abs_diff']:.3f}; r {gate['r_prompt']:.3f} (prompts), {gate['r_effect']:.3f} (effects); "
            f"checks {gate['checks']}; batch noise {noise:.2e} nats")
    write_status(run_id, body, ok=gate["passed"])
    return 0 if gate["passed"] else 1


# ---------------- stage 2 (minimal): patching ----------------


def patch_selection(cfg: dict, sets: dict, recs: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """The patching items (clear effect, calibration fold, stratified) and placebo objects for the first
    `n_placebo` of them that have one (in a seeded order)."""
    import random

    from wf.interp.prompts import placebo_object
    from wf.interp.run import select_patch_items

    fails = sets["failures"]
    fold = folds(fails + sets["successes"], cfg.get("split_seed", 0))
    calib = {g for g, f in fold.items() if f == cfg.get("calibration_fold", 0)}
    chosen = select_patch_items(fails, effects(recs), cfg.get("n_items", 100), cfg.get("min_effect", 1.0), calib,
                                f"interp|patch|{cfg.get('model_tag', MODEL_TAG)}")
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
    return chosen, placebo


def run_interp_pilot_patch(cfg: dict) -> int:
    from wf.interp.patching import Cell, grid
    from wf.interp.prompts import PILOT_GROUPS
    from wf.interp.run import patch_items, write_manifest
    from wf.runs import _init_run, require_passed

    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, cfg.get("stage0", STAGE0))
    d = _init_run(run_id, cfg)
    if not cfg.get("resume"):
        (d / "patch.jsonl").unlink(missing_ok=True)
    recs = _stage0_store(cfg).records()
    sets = load_sets(cfg)
    chosen, placebo = patch_selection(cfg, sets, recs)
    p, tok = load_model(cfg)
    layers = cfg.get("layers") or list(range(p.n_layers))
    cells: list[Cell] = [c for c in grid(p.n_layers, ("resid",), tuple(cfg.get("groups", PILOT_GROUPS)))
                         if c.layer in layers]
    write_manifest(d, {"items": [t["id"] for t in chosen], "placebo": placebo, "n_cells": len(cells),
                       "groups": list(cfg.get("groups", PILOT_GROUPS)), "layers": layers})
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
    """Every fourth layer from 4 and the last (for Qwen3-8B: 4, 8, ..., 32, 35)."""
    return sorted(set(range(4, n_layers, 4)) | {n_layers - 1})


def fold_directions(cfg: dict, store: Any, sets: dict, fold: dict[str, int], layers: list[int],
                    random_layers: tuple[int, ...] = (), seed: int = 0) -> dict:
    """{fold: {layer: directions}} estimated on each fold's items (`analysis.estimate_directions`); at
    `random_layers` also two random unit directions with their midpoints on the fold's aligned and swapped
    prompts ("random", sanity only)."""
    import numpy as np

    from wf.interp.analysis import estimate_directions
    from wf.interp.directions import midpoint_for, random_units

    recs = store.records()
    have = {(r["id"], r["goal"]) for r in recs}
    # One streaming pass over the store for every prompt needed, keeping only the ablation layers (float16).
    keys = sorted(f"{i}|yn|{g}" for i, g in have if g in ("aligned", "swapped", "absent", "swapped_plus"))
    rows_, arr = store.load("last", keys, sel=list(layers))
    at = {r["key"]: k for k, r in enumerate(rows_)}
    out: dict[int, dict[int, dict]] = {0: {}, 1: {}}
    for f in (0, 1):
        fails = [t for t in sets["failures"] if fold[t["game_id"]] == f and (t["id"], "aligned") in have
                 and (t["id"], "swapped") in have]
        succ = [t for t in sets["successes"] if fold[t["game_id"]] == f and (t["id"], "absent") in have]
        neutral_f = [t for t in fails if (t["id"], "absent") in have]
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
                                                rows([t["id"] for t in neutral_f], "absent"))
            if layer in random_layers:
                R = random_units(H["aligned"].shape[1], 2, seed + layer)
                out[f][layer]["random"] = R
                out[f][layer]["random_mid"] = np.array([midpoint_for(u, H["aligned"], H["swapped"]) for u in R])
    return out


def ablation_edits(dirs: dict[int, dict], layers: list[int]) -> dict[str, list[tuple[int, Any]]]:
    """Named edits at the answer position, from one fold's directions.

    task|L, task_perp_success|L, mention|L: mean ablation at layer L; task_from|L: the task direction ablated at L
    and at every later ablation layer; steer_success|L: add the mean aligned-minus-swapped projection on the
    success direction; random|L|k: a random unit direction, mean-ablated (sanity only, never a pass criterion,
    amendment 1 A7).
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
    for layer in layers:
        x = dirs[layer]
        edits[f"task|{layer}"] = [(layer, ab(x["task"], x["task_mid"]))]
        edits[f"task_perp_success|{layer}"] = [(layer, ab(x["task_perp_success"], x["task_perp_success_mid"]))]
        if "mention" in x:
            edits[f"mention|{layer}"] = [(layer, ab(x["mention"], x["mention_mid"]))]
        edits[f"steer_success|{layer}"] = [(layer, add(x["success"], x["steer_success"]))]
        edits[f"task_from|{layer}"] = [(m, ab(dirs[m]["task"], dirs[m]["task_mid"])) for m in layers if m >= layer]
        for k, (u, c) in enumerate(zip(x.get("random", []), x.get("random_mid", []))):
            edits[f"random|{layer}|{k}"] = [(layer, ab(u, c))]
    return edits


def ablation_selection(cfg: dict, sets: dict, recs: list[dict], fold: dict[str, int]) -> dict[int, list[dict]]:
    """Per fold: `n_failures_per_fold` failures and `n_successes_per_fold` successes, stratified by failure reason
    and step 0 (seeded); only items stage 0 scored under every goal used."""
    from wf.interp.run import select_patch_items

    have = {(r["id"], r["goal"]) for r in recs}
    out = {}
    for f in (0, 1):
        games = {g for g, x in fold.items() if x == f}
        fails = [t for t in sets["failures"] if all((t["id"], g) in have for g in ("aligned", "swapped", "absent"))]
        succ = [t for t in sets["successes"] if all((t["id"], g) in have for g in SUCC_GOALS)]
        anyeff = {t["id"]: 0.0 for t in fails + succ}
        out[f] = (select_patch_items(fails, anyeff, cfg.get("n_failures_per_fold", 75), float("-inf"), games,
                                     f"interp|ablate|fail|{f}")
                  + select_patch_items(succ, anyeff, cfg.get("n_successes_per_fold", 50), float("-inf"), games,
                                       f"interp|ablate|succ|{f}"))
    return out


def run_interp_pilot_ablate(cfg: dict) -> int:
    import numpy as np

    from wf.interp.analysis import direction_cosines
    from wf.interp.run import edit_items, write_manifest
    from wf.runs import _init_run, require_passed

    run_id = cfg["run_id"]
    require_filed(run_id)
    require_passed(run_id, cfg, cfg.get("stage0", STAGE0))
    d = _init_run(run_id, cfg)
    if not cfg.get("resume"):
        (d / "ablate.jsonl").unlink(missing_ok=True)
    store = _stage0_store(cfg)
    recs = store.records()
    sets = load_sets(cfg)
    fold = folds(sets["failures"] + sets["successes"], cfg.get("split_seed", 0))
    s0 = json.loads((runs_root() / cfg.get("stage0", STAGE0) / "manifest.json").read_text())
    n_layers = s0["n_layers"]
    layers = cfg.get("layers") or ablate_layers(n_layers)
    default_random = [layers[len(layers) // 2], layers[-2]] if len(layers) >= 2 else layers
    random_layers = tuple(sorted({m for m in cfg.get("random_layers", default_random) if m in layers}))
    dirs = fold_directions(cfg, store, sets, fold, layers, random_layers, cfg.get("seed", C.RNG_SEED))
    lens_dir = np.load(runs_root() / cfg.get("stage0", STAGE0) / "lens_dir.npy")
    np.savez(d / "directions.npz", **{f"{f}|{layer}|{k}": np.asarray(v) for f in dirs for layer in dirs[f]
                                     for k, v in dirs[f][layer].items()})
    cos = {f"{f}|{layer}": direction_cosines(dirs[f][layer], lens_dir) for f in dirs for layer in dirs[f]}
    cos.update({f"between_folds|{layer}": float(np.dot(dirs[0][layer]["task"], dirs[1][layer]["task"]))
                for layer in layers})
    sel = ablation_selection(cfg, sets, recs, fold)
    write_manifest(d, {"layers": layers, "random_layers": list(random_layers), "cosines": cos,
                       "items": {str(f): [t["id"] for t in sel[f]] for f in sel}, "fold_of_game": fold})
    p, tok = load_model(cfg)
    for f in (0, 1):
        edits = ablation_edits(dirs[1 - f], layers)
        edit_items(p, tok, sel[f], d / "ablate.jsonl", edits, fmt="yn", goals=("aligned", "swapped", "absent"),
                   positions="last", batch_size=cfg.get("batch_size", 16))
    n = len(read_jsonl(d / "ablate.jsonl"))
    write_status(run_id, f"{sum(len(v) for v in sel.values())} items, {n} records", ok=False,
                 reported="ablation complete")
    return 0


# ---------------- analysis (CPU) ----------------


def run_interp_pilot_analysis(cfg: dict) -> int:
    from wf.interp.analysis import STRATA, in_stratum, patch_summary
    from wf.runs import _init_run, require_passed

    run_id = cfg["run_id"]
    require_filed(run_id)
    for dep in (cfg.get("stage0", STAGE0), cfg.get("patch", "interp_pilot_patch"), cfg.get("ablate", "interp_pilot_ablate")):
        require_passed(run_id, cfg, dep)
    d = _init_run(run_id, {**cfg, "resume": False})
    root = runs_root()
    s0 = json.loads((root / cfg.get("stage0", STAGE0) / "manifest.json").read_text())
    n_layers = s0["n_layers"]
    B = cfg.get("bootstrap", 2000)
    report: dict[str, Any] = {"stage0": json.loads((root / cfg.get("stage0", STAGE0) / "metrics.json").read_text())}

    patch = read_jsonl(root / cfg.get("patch", "interp_pilot_patch") / "patch.jsonl")
    pat: dict[str, Any] = {}
    for direction in ("noise", "denoise"):
        goal = [r for r in patch if r["direction"] == direction and r["source_goal"] == "swapped"]
        plac = [r for r in patch if r["direction"] == direction and r["source_goal"] == "placebo"]
        pat[direction] = {s: patch_summary([r for r in goal if in_stratum(r, s)], [r for r in plac if in_stratum(r, s)],
                                           n_layers, B, C.RNG_SEED) for s in STRATA}
    report["patching"] = pat
    report["ablation"] = ablation_report(cfg, B)
    if cfg.get("hidden_probe", True):
        report["hidden_probe"] = hidden_probe_report(cfg)
    report["verdicts"] = verdict_rows(report)
    write_json(d / "metrics.json", report)
    lines = []
    for s in STRATA:
        hs = pat["noise"][s]["history_share"]
        lines.append(f"{s}: history share {hs.get('max', float('nan')) if hs.get('defined') else 'undefined'}; "
                     f"arrival {pat['noise'][s]['arrival_layer']:.2f}; "
                     f"task removed {report['ablation'][s].get('task', {}).get('share', float('nan')):.2f}")
    write_status(run_id, "\n".join(lines), ok=False, reported="minimal interpretability pilot (calibration)")
    return 0


def ablation_report(cfg: dict, B: int = 2000) -> dict:
    """Cross-fitted ablation readouts per stratum (amendment 1, A3, A5): the layer for each fold's items is the one
    whose task-direction ablation removed the most on the other fold's items."""
    from wf.interp.analysis import STRATA, choose_layer, gap, in_stratum, index_records, mean_shift, removed_share

    root = runs_root()
    ad = root / cfg.get("ablate", "interp_pilot_ablate")
    man = json.loads((ad / "manifest.json").read_text())
    layers = man["layers"]
    recs = read_jsonl(ad / "ablate.jsonl")
    by = index_records(recs)
    meta = {r["id"]: r for r in recs}
    games = {r["id"]: r["game_id"] for r in recs}
    fold_of = {i: int(f) for f, ids in man["items"].items() for i in ids}
    sets = load_sets(cfg)
    succ_ids = {t["id"] for t in sets["successes"]}
    seed = C.RNG_SEED

    def ids_of(f: int | None, kind: str, s: str) -> list[str]:
        return sorted(i for i, x in fold_of.items() if (f is None or x == f) and ((i in succ_ids) == (kind == "succ"))
                      and in_stratum(meta.get(i, {}), s))

    chosen = {}
    for f in (0, 1):
        other = ids_of(1 - f, "fail", "all")
        shares = {m: removed_share(by, other, lambda _i, m=m: f"task|{m}", games, B=200, seed=seed).get("share", float("nan"))
                  for m in layers}
        chosen[f] = choose_layer(shares)
    name = lambda kind: (lambda i: f"{kind}|{chosen[fold_of[i]]}")  # noqa: E731
    out: dict[str, Any] = {"chosen_layer_by_fold": chosen, "layers": layers, "cosines": man.get("cosines")}
    for s in STRATA:
        fails, succ = ids_of(None, "fail", s), ids_of(None, "succ", s)
        row: dict[str, Any] = {}
        for kind in ("task", "task_perp_success", "mention", "task_from"):
            row[kind] = removed_share(by, fails, name(kind), games, B, seed)
        row["task_on_successes"] = removed_share(by, succ, name("task"), games, B, seed)
        row["steer_success_gain_on_swapped"] = mean_shift(by["swapped"], fails, name("steer_success"), games, B, seed)
        row["collateral_neutral_shift"] = mean_shift(by["absent"], fails + succ, name("task"), games, B, seed)
        row["gap_neutral"] = gap(by["absent"], succ, fails, name("task"), games, B, seed)
        row["by_layer"] = {m: {"task": removed_share(by, fails, lambda _i, m=m: f"task|{m}", games, 200, seed).get("share"),
                               "task_perp_success": removed_share(by, fails, lambda _i, m=m: f"task_perp_success|{m}",
                                                                  games, 200, seed).get("share"),
                               "mention": removed_share(by, fails, lambda _i, m=m: f"mention|{m}", games, 200, seed).get("share")}
                           for m in layers}
        row["random_sanity"] = {k: removed_share(by, fails, lambda _i, k=k: k, games, 200, seed).get("share")
                                for k in sorted({k for r in recs for k in r["edits"] if k.startswith("random|")})}
        out[s] = row
    return out


def verdict_rows(report: dict) -> dict:
    """The outcome-table rows the minimal pilot can fill, per stratum, read against the provisional thresholds of
    amendment 1 (A6; interval bounds against the bands). Rows it cannot fill are listed; no verdict is drawn here."""
    rows: dict[str, Any] = {}
    for s in ("step0", "later", "all"):
        noise = report["patching"]["noise"][s]
        ab = report["ablation"].get(s, {})
        hs = noise["history_share"]
        task = ab.get("task", {})
        g = ab.get("gap_neutral", {})
        rows[s] = {
            "history_share": (None if not hs.get("defined") else
                              "high (>=0.3)" if hs["ci"][0] >= 0.3 else "low (<0.15)" if hs["ci"][1] < 0.15 else "between"),
            "arrival_layer": noise["arrival_layer"],
            "task_direction_removed": task.get("share"),
            "single_direction": (None if not task.get("n") else
                                 "carries (>=0.6)" if task["ci"][0] >= 0.6 else "does not (<0.3)" if task["ci"][1] < 0.3
                                 else "between"),
            "neutral_gap_ratio": g.get("ratio"),
            "discrimination_kept": None if g.get("ratio") is None else g["ratio"] >= 0.9,
            "not_filled_by_minimal_pilot": ["belief shift (history-fact probes need stage 1)", "head-level rows",
                                            "policy-direction transfer", "described-goal transfer",
                                            "first-token and full-reply readouts", "Llama"],
        }
    return rows


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

    from wf.interp.analysis import in_stratum
    from wf.interp.probes import (balanced_indices, belief_shifts, fit_probe, hidden_labels, probe_auc,
                                  probe_logit, project_out, readout_slope, residual_shifts, separation)

    store = _stage0_store(cfg)
    recs = store.records()
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
                    sv = X([t["id"] for t in tr if t["id"] in succ_ids], "absent", pos, k, layer).mean(0) - \
                        X([t["id"] for t in tr if t["id"] not in succ_ids], "absent", pos, k, layer).mean(0)
                    proj = [lens_dir, sv]
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
