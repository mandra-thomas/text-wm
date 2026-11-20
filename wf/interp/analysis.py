"""CPU analyses of the interpretability pilot's saved records (notes/interp_design.md, amendment 1).

Pure functions on plain records and arrays, so they are tested without a model:

- `reproduction_gate`: stage 0's readouts against the logged rq1v scores, per item and on average.
- `game_bootstrap`, `pooled_curve`, `max_over_layers`: ratio-of-sums fractions with game-clustered bootstraps,
  vectorised over cells, including the bootstrap of a maximum over layers (the layer is chosen inside every
  resample, so the interval accounts for the choice).
- `patch_summary`: the pre-registered patching readouts (history share, object-mention share and its matched
  control, arrival layer, action share, early-span share) for one stratum, and the placebo floor on the nats
  scale.
- `estimate_directions`: the task, success, mention and orthogonalised directions from stage 0's answer-position
  residuals of one fold.
- `ablation_summary`: removed shares with the layer chosen on the other fold, the success-versus-failure gap in
  nats, the collateral shift on neutral prompts, steering along the success direction, controls.

Strata: "step0" (no action yet, so no history token follows the task line and the history share is 0 by
construction) and "later"; every readout is reported per stratum and pooled.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable

# Reproduction gate (design section 2, amendment 1 A10). A GPU-type change moves one item's aligned-minus-swapped
# score with SD 0.55 nats for Qwen (deviations.md, "Scores across GPU types"), so one prompt's score moves with
# SD about 0.39 and a median absolute difference of about 0.26; the per-item bounds leave room for that, and a
# prompt or tokenisation difference moves many items by much more.
GATE_MEAN_TOL = 0.15  # nats, mean aligned-minus-swapped effect against the logged one on the same items
GATE_MEDIAN_ABS = 0.35  # nats, median |D - logged L| over prompts
GATE_R_PROMPT = 0.95  # Pearson r of D with the logged L over prompts
GATE_R_EFFECT = 0.90  # Pearson r of the item's effect with the logged effect
GATE_COVERAGE = 0.99  # share of the logged items that stage 0 reproduced
LOGGED_YN_EFFECT = {"qwen3-8b-fewshot": 2.80, "llama3.1-8b-fewshot": 2.21}
STRATA = ("step0", "later", "all")


def stratum(rec: dict) -> str:
    return "step0" if (rec.get("history_len") or 0) == 0 else "later"


def in_stratum(rec: dict, s: str) -> bool:
    return s == "all" or stratum(rec) == s


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return float("nan") if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def _median(x: list[float]) -> float:
    if not x:
        return float("nan")
    s = sorted(x)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


# ---------------- reproduction gate ----------------


def reproduction_gate(records: list[dict], logged: dict[tuple[str, str], float], model_tag: str,
                      fmt: str = "yn") -> dict:
    """Stage 0's readouts against the logged scores of the same prompts.

    `records`: stage 0 index rows (id, fmt, goal, d). `logged`: {(item id, arm): L} from `rq1v_score` for this
    model, arms "yn_aligned", "yn_swapped", "yn_absent". Passes only if every check holds: the mean effect on the
    same items within `GATE_MEAN_TOL`, the median absolute per-prompt difference at most `GATE_MEDIAN_ABS`, the
    per-prompt and per-item correlations at least `GATE_R_PROMPT` and `GATE_R_EFFECT`, and at least
    `GATE_COVERAGE` of the logged items reproduced. A mean that matches while items do not would hide a prompt
    difference that cancels on average; per-item checks catch it.
    """
    mine = {(r["id"], f"yn_{r['goal']}"): r["d"] for r in records if r.get("fmt") == fmt
            and r.get("goal") in ("aligned", "swapped", "absent")}
    shared = sorted(k for k in mine if k in logged)
    d = [mine[k] for k in shared]
    lg = [logged[k] for k in shared]
    diffs = [abs(a - b) for a, b in zip(d, lg)]
    ids = sorted({i for i, a in shared if (i, "yn_swapped") in mine and (i, "yn_swapped") in logged
                  and (i, "yn_aligned") in mine and (i, "yn_aligned") in logged})
    eff = [mine[(i, "yn_aligned")] - mine[(i, "yn_swapped")] for i in ids]
    eff_l = [logged[(i, "yn_aligned")] - logged[(i, "yn_swapped")] for i in ids]
    logged_ids = {i for i, a in logged if a == "yn_aligned" and (i, "yn_swapped") in logged}
    out = {
        "n_prompts": len(shared), "n_items": len(ids), "n_logged_items": len(logged_ids),
        "coverage": len(ids) / len(logged_ids) if logged_ids else float("nan"),
        "mean_effect": sum(eff) / len(eff) if eff else float("nan"),
        "mean_effect_logged_same_items": sum(eff_l) / len(eff_l) if eff_l else float("nan"),
        "logged_effect_published": LOGGED_YN_EFFECT.get(model_tag),
        "median_abs_diff": _median(diffs),
        "share_abs_diff_over_1": sum(x > 1 for x in diffs) / len(diffs) if diffs else float("nan"),
        "r_prompt": _pearson(d, lg), "r_effect": _pearson(eff, eff_l),
        "thresholds": {"mean_tol": GATE_MEAN_TOL, "median_abs": GATE_MEDIAN_ABS, "r_prompt": GATE_R_PROMPT,
                       "r_effect": GATE_R_EFFECT, "coverage": GATE_COVERAGE},
    }
    out["mean_gap"] = out["mean_effect"] - out["mean_effect_logged_same_items"]
    checks = {
        "mean": abs(out["mean_gap"]) <= GATE_MEAN_TOL,
        "median_abs": out["median_abs_diff"] <= GATE_MEDIAN_ABS,
        "r_prompt": out["r_prompt"] >= GATE_R_PROMPT,
        "r_effect": out["r_effect"] >= GATE_R_EFFECT,
        "coverage": out["coverage"] >= GATE_COVERAGE,
    }
    out["checks"] = checks
    out["passed"] = all(checks.values())  # nan compares False, so a missing quantity fails the gate
    return out


# ---------------- game bootstrap, vectorised ----------------


def game_weights(games: list[str], B: int, seed: int) -> Any:
    """(B + 1, items) resampling weights: row 0 is the point estimate (all ones); row b counts how often item i's
    game was drawn in the b-th resample of games with replacement."""
    import numpy as np

    uniq = sorted(set(games))
    idx = np.array([uniq.index(g) for g in games])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    counts = np.zeros((B, len(uniq)))
    for b in range(B):
        np.add.at(counts[b], draws[b], 1)
    return np.vstack([np.ones(len(games)), counts[:, idx]])


def _interval(boot: Any, alpha: float = 0.05) -> tuple[float, float]:
    import numpy as np

    b = boot[np.isfinite(boot)]
    if b.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def ratio_curve(num: Any, den: Any, mask: Any, W: Any) -> Any:
    """Ratio of sums per cell under every weight row: (W @ (mask * num)) / (W @ (mask * den)); (B + 1, cells)."""
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ np.where(mask, num, 0.0)) / (W @ np.where(mask, den, 0.0))


def mean_curve(x: Any, mask: Any, W: Any) -> Any:
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ np.where(mask, x, 0.0)) / (W @ mask.astype(float))


# ---------------- patching ----------------


def patch_matrix(records: list[dict], keys: list[str]) -> dict:
    """Arrays (items, cells) of removed effect (clean - patched), effect (clean - source) and presence."""
    import numpy as np

    from wf.interp.patching import cell_values

    n, m = len(records), len(keys)
    num, den, mask = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m), dtype=bool)
    for i, r in enumerate(records):
        for j, k in enumerate(keys):
            v = cell_values(r, k)
            if v is not None:
                clean, source, patched = v
                num[i, j], den[i, j], mask[i, j] = clean - patched, clean - source, True
    return {"num": num, "den": den, "mask": mask, "games": [r["game_id"] for r in records]}


def _layer_keys(group: str, n_layers: int, site: str = "resid") -> list[str]:
    return [f"{site}|{layer}|{group}" for layer in range(n_layers)]


def group_curve(records: list[dict], group: str, n_layers: int, B: int = 2000, seed: int = 0,
                site: str = "resid") -> dict:
    """Pooled fraction restored per layer for one group, its pointwise and Bonferroni-over-layers intervals,
    and the maximum over layers with a bootstrap interval in which the layer is re-chosen in every resample."""
    import numpy as np

    keys = _layer_keys(group, n_layers, site)
    if not records:
        return {"n": 0}
    pm = patch_matrix(records, keys)
    if not pm["mask"].any():
        return {"n": 0, "defined": False}
    W = game_weights(pm["games"], B, seed)
    R = ratio_curve(pm["num"], pm["den"], pm["mask"], W)
    removed_nats = mean_curve(pm["num"], pm["mask"], W)
    point = R[0]
    mx = np.nanmax(np.where(np.isfinite(R), R, -np.inf), axis=1)
    mx[~np.isfinite(mx)] = np.nan
    bonf = 0.05 / n_layers
    return {
        "n": int(pm["mask"].any(axis=1).sum()), "defined": True,
        "fraction": [float(x) for x in point],
        "ci": [_interval(R[1:, j]) for j in range(len(keys))],
        "ci_bonferroni": [_interval(R[1:, j], bonf) for j in range(len(keys))],
        "removed_nats": [float(x) for x in removed_nats[0]],
        "removed_nats_ci_bonferroni": [_interval(removed_nats[1:, j], bonf) for j in range(len(keys))],
        "max": float(mx[0]), "argmax_layer": int(np.nanargmax(np.where(np.isfinite(point), point, -np.inf))),
        "max_ci": _interval(mx[1:]),
    }


def placebo_floor(records: list[dict], group: str, n_layers: int, B: int = 2000, seed: int = 0,
                  site: str = "resid") -> dict:
    """Per layer, the placebo pairs' mean absolute change of the readout (nats) and its Bonferroni upper bound:
    the noise floor, on the same scale as the goal pairs' removed nats (amendment 1, A7)."""
    import numpy as np

    keys = _layer_keys(group, n_layers, site)
    if not records:
        return {"n": 0}
    pm = patch_matrix(records, keys)
    if not pm["mask"].any():
        return {"n": 0}
    W = game_weights(pm["games"], B, seed)
    A = mean_curve(np.abs(pm["num"]), pm["mask"], W)
    bonf = 0.05 / n_layers
    return {"n": int(pm["mask"].any(axis=1).sum()), "mean_abs_nats": [float(x) for x in A[0]],
            "upper_bonferroni": [_interval(A[1:, j], bonf)[1] for j in range(len(keys))]}


def arrival_from_curve(fractions: list[float], threshold: float = 0.5) -> float:
    from wf.interp.patching import arrival_layer

    return arrival_layer(fractions, threshold)


def patch_summary(records: list[dict], placebo: list[dict], n_layers: int, B: int = 2000, seed: int = 0) -> dict:
    """The pre-registered patching readouts for one set of records (one direction, one stratum).

    - history share: the maximum over layers of the fraction restored by the whole history after the task line,
      with a bootstrap interval for the maximum; undefined where no item has such tokens (every step-0 item);
    - object-mention share: the same for the history tokens naming the action's object, next to its position-
      matched control (as many history tokens naming neither object) and their difference per layer;
    - arrival layer of the final group (share of depth at which noising it restores half the effect for good);
    - action share; early-span share (equal-length items: span fraction at the end of the first third of the
      layers minus at the end of the second third);
    - for every group, the per-layer removed nats against the placebo floor (Bonferroni over layers).
    """
    out: dict[str, Any] = {"n_items": len(records), "n_placebo": len(placebo)}
    curves = {g: group_curve(records, g, n_layers, B, seed)
              for g in ("hist_all", "hist_obj", "hist_obj_matched", "action", "final", "last", "span", "task_rest",
                        "query")}
    out["curves"] = curves
    out["history_share"] = ({"defined": False, "why": "no history token after the task line (step 0)"}
                            if not curves["hist_all"].get("defined") else
                            {"defined": True, "max": curves["hist_all"]["max"], "ci": curves["hist_all"]["max_ci"],
                             "layer": curves["hist_all"]["argmax_layer"]})
    ho, hm = curves["hist_obj"], curves["hist_obj_matched"]
    out["object_mention_share"] = ({"defined": False} if not ho.get("defined") else
                                   {"defined": True, "max": ho["max"], "ci": ho["max_ci"],
                                    "matched_max": hm.get("max"), "matched_ci": hm.get("max_ci")})
    out["action_share"] = {"max": curves["action"].get("max"), "ci": curves["action"].get("max_ci")}
    fin = curves["final"].get("fraction")
    out["arrival_layer"] = arrival_from_curve(fin) if fin else float("nan")
    sp = curves["span"].get("fraction")
    if sp:
        a, b = n_layers // 3 - 1, 2 * n_layers // 3 - 1
        out["early_span_share"] = sp[a] - sp[b] if sp[a] == sp[a] and sp[b] == sp[b] else float("nan")
    else:
        out["early_span_share"] = float("nan")
    floors = {g: placebo_floor(placebo, g, n_layers, B, seed) for g in curves}
    out["placebo_floor"] = floors
    above = {}
    for g, c in curves.items():
        f = floors[g]
        if c.get("defined") and f.get("n"):
            above[g] = [lo > ub for (lo, _hi), ub in zip(c["removed_nats_ci_bonferroni"], f["upper_bonferroni"])]
    out["above_placebo_floor"] = above
    return out


def by_stratum(records: list[dict], fn: Callable[[list[dict]], Any]) -> dict:
    return {s: fn([r for r in records if in_stratum(r, s)]) for s in STRATA}


# ---------------- directions ----------------


def _unit(v: Any) -> Any:
    import numpy as np

    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def estimate_directions(H: dict[str, Any], success_rows: Any, failure_rows: Any) -> dict:
    """Directions at one layer from one fold's answer-position residuals.

    `H[goal]` is (items, width) for the fold's failure items, paired across goals ("aligned", "swapped", and
    "swapped_plus" where it exists, rows aligned with H["aligned"] or NaN); `success_rows`/`failure_rows` are
    (n, width) neutral-task residuals of the fold's determined successes and failures.

    - task: unit mean of aligned minus swapped, with the midpoint of the mean projections (the mean-ablation
      value);
    - success: unit mean of neutral successes minus neutral failures (the model's own "this works" direction);
    - task_perp_success: the task direction with its success component removed, renormalised, with its midpoint;
    - mention: unit mean of swapped_plus minus swapped (merely naming the action's object), with its midpoint;
    - steer_success: the mean projection of aligned minus swapped on the success direction (nats of residual),
      the amount added along it to steer a swapped prompt.
    """
    import numpy as np

    from wf.interp.directions import diff_of_means, midpoint_for

    A, S = np.asarray(H["aligned"], dtype=np.float64), np.asarray(H["swapped"], dtype=np.float64)
    task = diff_of_means(A, S)
    s = _unit(np.asarray(success_rows, dtype=np.float64).mean(0) - np.asarray(failure_rows, dtype=np.float64).mean(0))
    perp = _unit(task["u"] - (task["u"] @ s) * s)
    out = {"task": task["u"], "task_mid": task["midpoint"], "task_norm": task["norm"], "success": s,
           "task_perp_success": perp, "task_perp_success_mid": midpoint_for(perp, A, S),
           "steer_success": float(((A - S) @ s).mean())}
    P = H.get("swapped_plus")
    if P is not None:
        P = np.asarray(P, dtype=np.float64)
        ok = np.isfinite(P).all(axis=1)
        if ok.sum() >= 2:
            m = diff_of_means(P[ok], S[ok])
            out["mention"], out["mention_mid"] = m["u"], m["midpoint"]
    return out


def direction_cosines(dirs: dict, lens_dir: Any) -> dict:
    from wf.interp.directions import cosine

    out = {"task_success": cosine(dirs["task"], dirs["success"]), "task_readout": cosine(dirs["task"], lens_dir),
           "success_readout": cosine(dirs["success"], lens_dir)}
    if "mention" in dirs:
        out["task_mention"] = cosine(dirs["task"], dirs["mention"])
    return out


# ---------------- ablation ----------------


def _edit_value(rec: dict, name: str) -> tuple[float, float] | None:
    v = rec["edits"].get(name)
    if v is None:
        return None
    return rec.get("edit_clean", {}).get(name, rec["d"]), v


def removed_share(recs_by_goal: dict[str, dict[str, dict]], ids: list[str], name_for: Callable[[str], str],
                  games: dict[str, str], B: int = 2000, seed: int = 0, a: str = "aligned",
                  b: str = "swapped") -> dict:
    """1 - (sum over items of the edited a-minus-b effect) / (sum of the unedited effect), with `name_for(id)`
    giving each item's edit (so each fold can use the layer chosen on the other); game bootstrap."""
    import numpy as np

    rows = []
    for i in ids:
        ra, rb = recs_by_goal[a].get(i), recs_by_goal[b].get(i)
        if not ra or not rb:
            continue
        va, vb = _edit_value(ra, name_for(i)), _edit_value(rb, name_for(i))
        if va is None or vb is None:
            continue
        rows.append((va[0] - vb[0], va[1] - vb[1], games[i]))
    if not rows:
        return {"n": 0}
    before = np.array([[r[0] for r in rows]])
    after = np.array([[r[1] for r in rows]])
    W = game_weights([r[2] for r in rows], B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 1 - (W @ after.T)[:, 0] / (W @ before.T)[:, 0]
    return {"n": len(rows), "share": float(share[0]), "ci": _interval(share[1:]),
            "effect_before": float(before.mean()), "effect_after": float(after.mean())}


def mean_shift(recs: dict[str, dict], ids: list[str], name_for: Callable[[str], str], games: dict[str, str],
               B: int = 2000, seed: int = 0) -> dict:
    """Mean of edited minus unedited readout over the items' prompts (collateral shift, or a steering gain)."""
    import numpy as np

    rows = [(v[1] - v[0], games[i]) for i in ids if i in recs for v in [_edit_value(recs[i], name_for(i))] if v]
    if not rows:
        return {"n": 0}
    x = np.array([[r[0] for r in rows]])
    W = game_weights([r[1] for r in rows], B, seed)
    m = (W @ x.T)[:, 0] / W.sum(1)
    return {"n": len(rows), "mean": float(m[0]), "ci": _interval(m[1:])}


def gap(recs_neutral: dict[str, dict], success_ids: list[str], failure_ids: list[str],
        name_for: Callable[[str], str] | None, games: dict[str, str], B: int = 2000, seed: int = 0) -> dict:
    """Success-versus-failure gap in nats on neutral prompts: mean D over successes minus mean D over failures,
    unedited, and edited if `name_for` is given (amendment 1, A3: the AUC is at its ceiling, the gap is not)."""
    import numpy as np

    def val(i: str, edited: bool) -> float | None:
        r = recs_neutral.get(i)
        if not r:
            return None
        if not edited:
            return r["d"]
        v = _edit_value(r, name_for(i))
        return None if v is None else v[1]

    out = {}
    for label, edited in (("clean", False), ("edited", True)):
        if edited and name_for is None:
            continue
        rows = [(val(i, edited), 1.0, games[i]) for i in success_ids] + [(val(i, edited), 0.0, games[i]) for i in failure_ids]
        rows = [r for r in rows if r[0] is not None]
        if not rows or not any(r[1] for r in rows) or all(r[1] for r in rows):
            out[label] = {"n": 0}
            continue
        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        W = game_weights([r[2] for r in rows], B, seed)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = (W @ (x * y)) / (W @ y) - (W @ (x * (1 - y))) / (W @ (1 - y))
        from wf.interp.directions import auc

        out[label] = {"n": len(rows), "gap": float(g[0]), "ci": _interval(g[1:]),
                      "auc": auc([r[0] for r in rows if r[1]], [r[0] for r in rows if not r[1]])}
    if "edited" in out and out["clean"].get("n") and out["edited"].get("n"):
        out["ratio"] = out["edited"]["gap"] / out["clean"]["gap"] if out["clean"]["gap"] else float("nan")
    return out


def choose_layer(shares_by_layer: dict[int, float]) -> int | None:
    """The layer with the largest removed share (ties to the earlier layer); None if none is finite."""
    ok = {k: v for k, v in shares_by_layer.items() if v == v}
    return min(ok, key=lambda k: (-ok[k], k)) if ok else None


def index_records(records: Iterable[dict]) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by[r["goal"]][r["id"]] = r
    return by
