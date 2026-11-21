"""CPU analyses of the interpretability pilot's saved records (notes/interp_design.md, amendments 1 and 2).

Pure functions on plain records and arrays, so they are tested without a model:

- `reproduction_gate`: stage 0's readouts against the logged rq1v scores (yes/no L, first-token L_first), per
  item and on average, with the logged item hashes checked.
- `game_bootstrap`, `pooled_curve`, `max_over_layers`: ratio-of-sums fractions with game-clustered bootstraps,
  vectorised over cells, including the bootstrap of a maximum over layers (the layer is chosen inside every
  resample, so the interval accounts for the choice).
- `patch_summary`: the pre-registered patching readouts (history share, object-mention share and its matched
  control, arrival layer, action share, early-span share) for one stratum, and the placebo floor on the nats
  scale.
- `estimate_directions`: the task, within-verb success, mention and orthogonalised directions from stage 0's
  answer-position residuals of one fold, and the success direction without its readout component.
- `profile_layers`, `caveat_layers`, `h4_layers`, `onset_layer`: the ablation's depth profile and its
  pre-registered summary (amendment 2, B1, B2).
- `ablation_summary`: the removed share at every profile layer and the onset layer, readouts at a chosen layer,
  the success-versus-failure gap in nats, the collateral shift on neutral prompts, steering along the success
  direction with and without its readout component, controls.

Strata: "step0" (no action yet, so no history token follows the task line and the history share is 0 by
construction) and "later"; every readout is reported per stratum and pooled, pooled estimates weighted back to
the population when a stratum was oversampled (amendment 2, B4).
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
# First-token two-candidate effect (rq1v's plain arms, L_first; design §2), gated the same way (amendment 2, B6).
LOGGED_FIRST_EFFECT = {"qwen3-8b-fewshot": 2.52, "llama3.1-8b-fewshot": 0.68}
STRATA = ("step0", "later", "all")
# Amendment 2 (B1, B2): the ablation's depth profile stops at 0.8 of depth (layer 28 of Qwen3-8B's 36), because at
# the last layer the readout is linear in the normed residual and mean ablation along the mean difference removes
# the mean effect by construction, and three layers before it (layer 32) the network is close to that; the H4 rows
# are read at layers up to two thirds of depth (24 of 36), before the answer position is dominated by the readout.
PROFILE_MAX_DEPTH = 0.8
H4_MAX_DEPTH = 2 / 3
LAYER_STEP = 4
ONSET_THRESHOLD = 0.6  # removal whose cross-fitted lower bound stays at or above this from the onset layer on
TARGET_HALF_WIDTH = 0.075  # amendment 2, B3: the full pilot's sample size is set to reach this half-width
SAMPLE_SIZE_MARGIN = 1.2  # amendment 3, C6: games as clusters make the 1/sqrt(n) scaling optimistic
NO_DIRECTION_UPPER = 0.3  # amendment 3, C1: "no single direction" needs the deepest layer's upper bound below this
# Amendment 3, C2: a profile layer whose direct-path share (the part of the effect already in readout form at that
# layer, read at the final norm's scale) has a 95% upper bound at or above this is treated as a caveat layer.
DIRECT_PATH_MAX = 0.3
PERP_TASK_MIN_REMOVAL = 0.3  # amendment 3, C3: the perp/task removal ratio is read only where task removal's
# lower bound reaches this


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
                      fmt: str = "yn", logged_hash: dict[str, str] | None = None,
                      current_hash: dict[str, str] | None = None) -> dict:
    """Stage 0's readouts against the logged scores of the same prompts.

    `records`: stage 0 index rows (id, fmt, goal, d). `logged`: {(item id, arm): score} from `rq1v_score` for
    this model: for `fmt` "yn" the arms "yn_aligned", "yn_swapped", "yn_absent" with their L; for "obs" (the
    first-token readout, amendment 2, B6) the plain arms "aligned" and "swapped" with their first-token term
    L_first. Passes only if every check holds: the mean effect on the same items within `GATE_MEAN_TOL`, the
    median absolute per-prompt difference at most `GATE_MEDIAN_ABS`, the per-prompt and per-item correlations at
    least `GATE_R_PROMPT` and `GATE_R_EFFECT`, at least `GATE_COVERAGE` of the logged items reproduced, and, when
    `logged_hash` and `current_hash` ({item id: `runs._item_hash`}) are given, every compared item's logged hash
    equal to the current item's (a score never attaches to a changed item). A mean that matches while items do
    not would hide a prompt difference that cancels on average; per-item checks catch it.
    """
    if fmt == "yn":
        goals, arm = ("aligned", "swapped", "absent"), (lambda g: f"yn_{g}")
    elif fmt == "obs":
        goals, arm = ("aligned", "swapped"), (lambda g: g)
    else:
        raise ValueError(f"no logged scores for format {fmt!r}")
    al, sw = arm("aligned"), arm("swapped")
    mine = {(r["id"], arm(r["goal"])): r["d"] for r in records if r.get("fmt") == fmt and r.get("goal") in goals}
    shared = sorted(k for k in mine if k in logged)
    d = [mine[k] for k in shared]
    lg = [logged[k] for k in shared]
    diffs = [abs(a - b) for a, b in zip(d, lg)]
    ids = sorted({i for i, a in shared if (i, sw) in mine and (i, sw) in logged
                  and (i, al) in mine and (i, al) in logged})
    eff = [mine[(i, al)] - mine[(i, sw)] for i in ids]
    eff_l = [logged[(i, al)] - logged[(i, sw)] for i in ids]
    logged_ids = {i for i, a in logged if a == al and (i, sw) in logged}
    hash_bad = (None if logged_hash is None or current_hash is None else
                sorted({i for i, _a in shared if logged_hash.get(i) != current_hash.get(i)}))
    out = {
        "fmt": fmt, "n_prompts": len(shared), "n_items": len(ids), "n_logged_items": len(logged_ids),
        "coverage": len(ids) / len(logged_ids) if logged_ids else float("nan"),
        "mean_effect": sum(eff) / len(eff) if eff else float("nan"),
        "mean_effect_logged_same_items": sum(eff_l) / len(eff_l) if eff_l else float("nan"),
        "logged_effect_published": (LOGGED_YN_EFFECT if fmt == "yn" else LOGGED_FIRST_EFFECT).get(model_tag),
        "item_hash_mismatches": None if hash_bad is None else len(hash_bad),
        "item_hash_mismatch_examples": None if hash_bad is None else hash_bad[:5],
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
    if hash_bad is not None:
        checks["item_hash"] = not hash_bad
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


def _interval_normal(point: float, boot: Any, alpha: float) -> tuple[float, float]:
    """point +- z(1 - alpha/2) * SE, the SE being the bootstrap draws' standard deviation (amendment 2, B5).

    Used for the Bonferroni-corrected intervals: at alpha = 0.05/36 a percentile interval reads the 0.07% tails,
    which 2,000 resamples place on one or two draws, so the bound would be noise; the normal interval needs only
    the SE, which 2,000 resamples estimate well."""
    import numpy as np
    from statistics import NormalDist

    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    if b.size < 2 or point != point:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    se = float(np.std(b, ddof=1))
    return float(point - z * se), float(point + z * se)


def _se(boot: Any) -> float:
    import numpy as np

    b = np.asarray(boot, dtype=float)
    b = b[np.isfinite(b)]
    return float(np.std(b, ddof=1)) if b.size >= 2 else float("nan")


def item_weights(ids: list[str], weights: dict[str, float] | None) -> Any:
    """Per-item weights (amendment 2, B4: population share over sample share of the item's stratum), 1 if none."""
    import numpy as np

    return np.array([1.0 if weights is None else float(weights.get(i, 1.0)) for i in ids])


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
    return {"num": num, "den": den, "mask": mask, "games": [r["game_id"] for r in records],
            "ids": [r.get("id") for r in records]}


def _layer_keys(group: str, n_layers: int, site: str = "resid") -> list[str]:
    return [f"{site}|{layer}|{group}" for layer in range(n_layers)]


def group_curve(records: list[dict], group: str, n_layers: int, B: int = 2000, seed: int = 0,
                site: str = "resid", weights: dict[str, float] | None = None) -> dict:
    """Pooled fraction restored per layer for one group, its pointwise and Bonferroni-over-layers intervals,
    and the maximum over layers with a bootstrap interval in which the layer is re-chosen in every resample.

    `weights` ({item id: weight}) reweights items to their population shares (amendment 2, B4). Bonferroni
    intervals are normal intervals from the bootstrap SE (amendment 2, B5)."""
    import numpy as np

    keys = _layer_keys(group, n_layers, site)
    if not records:
        return {"n": 0}
    pm = patch_matrix(records, keys)
    if not pm["mask"].any():
        return {"n": 0, "defined": False}
    W = game_weights(pm["games"], B, seed) * item_weights(pm["ids"], weights)[None, :]
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
        "se": [_se(R[1:, j]) for j in range(len(keys))],
        "ci_bonferroni": [_interval_normal(float(point[j]), R[1:, j], bonf) for j in range(len(keys))],
        "removed_nats": [float(x) for x in removed_nats[0]],
        "removed_nats_ci_bonferroni": [_interval_normal(float(removed_nats[0, j]), removed_nats[1:, j], bonf)
                                       for j in range(len(keys))],
        "max": float(mx[0]), "argmax_layer": int(np.nanargmax(np.where(np.isfinite(point), point, -np.inf))),
        "max_ci": _interval(mx[1:]), "max_se": _se(mx[1:]),
    }


def placebo_floor(records: list[dict], group: str, n_layers: int, B: int = 2000, seed: int = 0,
                  site: str = "resid") -> dict:
    """Per layer, the placebo pairs' mean absolute change of the readout (nats) and its Bonferroni upper bound
    (normal, from the bootstrap SE): the noise floor, on the same scale as the goal pairs' removed nats
    (amendment 1, A7)."""
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
            "upper_bonferroni": [_interval_normal(float(A[0, j]), A[1:, j], bonf)[1] for j in range(len(keys))]}


def arrival_from_curve(fractions: list[float], threshold: float = 0.5) -> float:
    from wf.interp.patching import arrival_layer

    return arrival_layer(fractions, threshold)


def patch_summary(records: list[dict], placebo: list[dict], n_layers: int, B: int = 2000, seed: int = 0,
                  weights: dict[str, float] | None = None) -> dict:
    """The pre-registered patching readouts for one set of records (one direction, one stratum).

    - history share: the maximum over layers of the fraction restored by the whole history after the task line,
      with a bootstrap interval for the maximum; undefined where no item has such tokens (every step-0 item);
    - object-mention share: the same for the history tokens naming the action's object, next to its position-
      matched control (as many history tokens naming neither object) and their difference per layer;
    - arrival layer of the final group (share of depth at which noising it restores half the effect for good);
    - action share; early-span share (equal-length items: span fraction at the end of the first third of the
      layers minus at the end of the second third);
    - for every group, the per-layer removed nats against the placebo floor (Bonferroni over layers).

    `weights` reweight the goal pairs to the population's strata (amendment 2, B4); the placebo floor is a
    per-cell noise level and is not reweighted.
    """
    out: dict[str, Any] = {"n_items": len(records), "n_placebo": len(placebo)}
    curves = {g: group_curve(records, g, n_layers, B, seed, weights=weights)
              for g in ("hist_all", "hist_obj", "hist_obj_matched", "hist_swap", "action", "action_obj", "action_rest",
                        "final", "last", "span", "task_rest", "query")}
    out["curves"] = curves
    out["history_share"] = ({"defined": False, "why": "no history token after the task line (step 0)"}
                            if not curves["hist_all"].get("defined") else
                            {"defined": True, "max": curves["hist_all"]["max"], "ci": curves["hist_all"]["max_ci"],
                             "se": curves["hist_all"]["max_se"], "layer": curves["hist_all"]["argmax_layer"]})
    ho, hm = curves["hist_obj"], curves["hist_obj_matched"]
    out["object_mention_share"] = ({"defined": False} if not ho.get("defined") else
                                   {"defined": True, "max": ho["max"], "ci": ho["max_ci"], "se": ho["max_se"],
                                    "matched_max": hm.get("max"), "matched_ci": hm.get("max_ci")})
    out["action_share"] = {"max": curves["action"].get("max"), "ci": curves["action"].get("max_ci"),
                           "se": curves["action"].get("max_se")}
    out["action_object_share"] = {"max": curves["action_obj"].get("max"), "ci": curves["action_obj"].get("max_ci")}
    out["action_rest_share"] = {"max": curves["action_rest"].get("max"), "ci": curves["action_rest"].get("max_ci")}
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


def verb(t: dict) -> str:
    """The action's verb ("take", "move", ...), the stratum of the within-verb success direction."""
    a = (t.get("action") or "").split()
    return a[0] if a else "?"


def success_direction(success_rows: Any, failure_rows: Any, success_verbs: list[str] | None = None,
                      failure_verbs: list[str] | None = None) -> Any:
    """Unit "this works" direction from neutral-task residuals of determined successes and failures.

    With verbs (amendment 2, B2): the weighted mean over verbs of (mean success residual - mean failure residual)
    within the verb, weight n_s * n_f / (n_s + n_f), over verbs that have both. The determined successes include
    verbs no failure has (examine, go to, close, use, inventory: 106 of the 292), so a plain difference of means would also carry
    "which verb is this"; within verbs it cannot. Without verbs, the plain difference of means.
    """
    import numpy as np

    Sx, Fx = np.asarray(success_rows, dtype=np.float64), np.asarray(failure_rows, dtype=np.float64)
    if success_verbs is None or failure_verbs is None:
        return _unit(Sx.mean(0) - Fx.mean(0))
    sv, fv = np.asarray(success_verbs), np.asarray(failure_verbs)
    num, den = np.zeros(Sx.shape[1]), 0.0
    for v in sorted(set(sv.tolist()) & set(fv.tolist())):
        a, b = sv == v, fv == v
        w = a.sum() * b.sum() / (a.sum() + b.sum())
        num += w * (Sx[a].mean(0) - Fx[b].mean(0))
        den += w
    if den == 0:
        raise ValueError("no verb has both a success and a failure")
    return _unit(num / den)


def project_unit(v: Any, r: Any) -> Any:
    """`v` with its component along `r` removed, renormalised."""
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    rr = _unit(np.asarray(r, dtype=np.float64))
    return _unit(v - (v @ rr) * rr)


def estimate_directions(H: dict[str, Any], success_rows: Any, failure_rows: Any,
                        success_verbs: list[str] | None = None, failure_verbs: list[str] | None = None,
                        readout_dir: Any = None, strata: list[str] | None = None) -> dict:
    """Directions at one layer from one fold's answer-position residuals.

    `H[goal]` is (items, width) for the fold's failure items, paired across goals ("aligned", "swapped", and
    "swapped_plus" where it exists, rows aligned with H["aligned"] or NaN); `success_rows`/`failure_rows` are
    (n, width) neutral-task residuals of the fold's determined successes and failures, with their verbs.

    - task: unit mean of aligned minus swapped, with the midpoint of the mean projections (the mean-ablation
      value);
    - success: the within-verb success direction (`success_direction`);
    - task_perp_success: the task direction with its success component removed, renormalised, with its midpoint;
    - mention: unit mean of swapped_plus minus swapped (merely naming the action's object), with its midpoint;
    - steer_success: the mean projection of aligned minus swapped on the success direction (nats of residual),
      the amount added along it to steer a swapped prompt;
    - with `readout_dir` (amendment 2, B2): success_perp_readout, the success direction with its component along
      the readout direction removed, and steer_success_perp_readout, the mean aligned-minus-swapped projection on
      it. Steering along the raw success direction moves the readout through its readout component whatever the
      mechanism; this one cannot.
    - steer_task (amendment 3, C3): the mean aligned-minus-swapped projection on the task direction, the amount
      of the reference edit `steer_task` (steering along the task direction itself);
    - with `strata` (one label per failure row, "step0" or "later"; amendment 3, C4): `steer_by_stratum`, every
      steering amount estimated on that stratum's rows only, so each item is steered by its own stratum's amount
      (Qwen's effect is about twice as large at step 0).
    """
    import numpy as np

    from wf.interp.directions import diff_of_means, midpoint_for

    A, S = np.asarray(H["aligned"], dtype=np.float64), np.asarray(H["swapped"], dtype=np.float64)
    task = diff_of_means(A, S)
    s = success_direction(success_rows, failure_rows, success_verbs, failure_verbs)
    perp = _unit(task["u"] - (task["u"] @ s) * s)
    out = {"task": task["u"], "task_mid": task["midpoint"], "task_norm": task["norm"], "success": s,
           "task_perp_success": perp, "task_perp_success_mid": midpoint_for(perp, A, S),
           "steer_success": float(((A - S) @ s).mean()), "steer_task": float(((A - S) @ task["u"]).mean())}
    steer_dirs = {"steer_success": s, "steer_task": task["u"]}
    if readout_dir is not None:
        sp = project_unit(s, readout_dir)
        out["success_perp_readout"] = sp
        out["steer_success_perp_readout"] = float(((A - S) @ sp).mean())
        steer_dirs["steer_success_perp_readout"] = sp
    if strata is not None:
        lab = np.asarray(strata)
        out["steer_by_stratum"] = {st: {k: float(((A[lab == st] - S[lab == st]) @ v).mean()) for k, v in steer_dirs.items()}
                                   for st in sorted(set(lab.tolist()))}
    P = H.get("swapped_plus")
    if P is not None:
        P = np.asarray(P, dtype=np.float64)
        ok = np.isfinite(P).all(axis=1)
        if ok.sum() >= 2:
            m = diff_of_means(P[ok], S[ok])
            out["mention"], out["mention_mid"] = m["u"], m["midpoint"]
    return out


def direction_cosines(dirs: dict, lens_dir: Any) -> dict:
    """Cosines of one fold's directions at one layer (amendment 1, A3; amendment 2, B2).

    task_success_perp_readout is the H4 cosine: the task and success directions after the readout direction is
    projected out of both, so that two directions that each move the answer cannot look alike only through the
    answer. Its null baseline from the same fold is mention_success_perp_readout (merely naming the object, not
    as the goal); the between-fold cosines are added by the caller.
    """
    from wf.interp.directions import cosine

    out = {"task_success": cosine(dirs["task"], dirs["success"]), "task_readout": cosine(dirs["task"], lens_dir),
           "success_readout": cosine(dirs["success"], lens_dir),
           "task_success_perp_readout": cosine(project_unit(dirs["task"], lens_dir),
                                               project_unit(dirs["success"], lens_dir))}
    if "mention" in dirs:
        out["task_mention"] = cosine(dirs["task"], dirs["mention"])
        out["mention_success"] = cosine(dirs["mention"], dirs["success"])
        out["mention_success_perp_readout"] = cosine(project_unit(dirs["mention"], lens_dir),
                                                     project_unit(dirs["success"], lens_dir))
    return out


def answer_direction(X: Any, D: Any, outcome: list[bool], verbs: list[str],
                     alphas: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)) -> Any:
    """A layer-specific answer direction (amendment 3, C3): the unit coefficient vector of a ridge regression of
    the final readout D on the layer's residual X, fit on neutral prompts after both are centred within each
    (outcome, verb) cell, so that it follows how the answer varies at that layer within a cell and not the
    success or verb contrast itself. Penalty by efficient leave-one-out (sklearn RidgeCV)."""
    import numpy as np
    from sklearn.linear_model import RidgeCV

    X = np.asarray(X, dtype=np.float64).copy()
    y = np.asarray(D, dtype=np.float64).copy()
    cells = [f"{o}|{v}" for o, v in zip(outcome, verbs)]
    for c in set(cells):
        m = np.array([x == c for x in cells])
        X[m] -= X[m].mean(0)
        y[m] -= y[m].mean()
    coef = RidgeCV(alphas=alphas, fit_intercept=False).fit(X, y).coef_
    return _unit(coef)


def project_out_all(v: Any, dirs: list[Any]) -> Any:
    """`v` with its components along every direction in `dirs` removed (Gram-Schmidt), renormalised."""
    import numpy as np

    basis: list[Any] = []
    for d in dirs:
        d = np.asarray(d, dtype=np.float64)
        for b in basis:
            d = d - (d @ b) * b
        basis.append(_unit(d))
    v = np.asarray(v, dtype=np.float64)
    for b in basis:
        v = v - (v @ b) * b
    return _unit(v)


def projected_cosine_boot(diffs: Any, diff_games: list[str], succ: Any, succ_verbs: list[str], succ_games: list[str],
                          fail: Any, fail_verbs: list[str], fail_games: list[str], removes: dict[str, list[Any]],
                          B: int = 2000, seed: int = 0) -> dict:
    """The H4 cosine cos(u_perp, s_perp) with a game-bootstrap interval (amendment 3, C3): games are resampled
    with replacement (one draw for every item set of the fold), the task direction u (mean of `diffs`, aligned
    minus swapped) and the within-verb success direction s are re-estimated in every resample, and for each
    entry of `removes` ({name: directions, fixed, estimated once}) those directions are projected out of both
    before the cosine. Returns {name: {cos, ci, se}}."""
    import numpy as np

    games = sorted(set(diff_games) | set(succ_games) | set(fail_games))
    rng = np.random.default_rng(seed)
    counts = np.vstack([np.ones(len(games))] + [np.bincount(rng.integers(0, len(games), len(games)),
                                                            minlength=len(games)) for _ in range(B)])
    gi = {g: k for k, g in enumerate(games)}

    def wmean(rows: Any, gs: list[str], mask: Any = None) -> tuple[Any, Any]:
        w = counts[:, [gi[g] for g in gs]]
        if mask is not None:
            w = w * mask[None, :]
        tot = w.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (w @ np.asarray(rows, dtype=np.float64)) / tot[:, None], tot

    U, _ = wmean(diffs, diff_games)
    sv, fv = np.asarray(succ_verbs), np.asarray(fail_verbs)
    num, den = 0.0, 0.0
    for v in sorted(set(sv.tolist()) & set(fv.tolist())):
        ms, ns = wmean(succ, succ_games, (sv == v).astype(float))
        mf, nf = wmean(fail, fail_games, (fv == v).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where((ns > 0) & (nf > 0), ns * nf / (ns + nf), 0.0)
        num = num + w[:, None] * np.nan_to_num(ms - mf)
        den = den + w
    with np.errstate(invalid="ignore", divide="ignore"):
        Sd0 = num / den[:, None]
    out = {}
    for name, remove in removes.items():
        basis: list[Any] = []
        for d in remove:
            d = np.asarray(d, dtype=np.float64)
            for b in basis:
                d = d - (d @ b) * b
            basis.append(d / np.linalg.norm(d))
        Uu, Ss = U, Sd0
        for b in basis:
            Uu = Uu - np.outer(Uu @ b, b)
            Ss = Ss - np.outer(Ss @ b, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (Uu * Ss).sum(1) / (np.linalg.norm(Uu, axis=1) * np.linalg.norm(Ss, axis=1))
        out[name] = {"cos": float(c[0]), "ci": _interval(c[1:]), "se": _se(c[1:])}
    return out


def disattenuated(cos: float, rel_a: float | None, rel_b: float | None) -> float:
    """A cosine between two noisy direction estimates divided by the square root of their reliabilities (each
    the cosine between its two folds' estimates; amendment 3, C3): what the cosine would be between noise-free
    directions, to first order. NaN if a reliability is not positive; clipped to [-1, 1]."""
    if rel_a is None or rel_b is None or not (rel_a > 0 and rel_b > 0) or cos != cos:
        return float("nan")
    return max(-1.0, min(1.0, cos / math.sqrt(rel_a * rel_b)))


# ---------------- layers ----------------


def profile_layers(n_layers: int) -> list[int]:
    """The ablation depth profile (amendment 2, B1): every fourth layer from 4 up to 0.8 of depth (Qwen3-8B: 4, 8,
    ..., 28; Llama-3.1-8B: 4, ..., 24). The last layer is excluded because there the ablation is tautological,
    and the layer three before it because the network is close to linear from there on."""
    return list(range(LAYER_STEP, int(PROFILE_MAX_DEPTH * n_layers) + 1, LAYER_STEP))


def caveat_layers(n_layers: int) -> list[int]:
    """Grid layers after the profile and before the last layer (Qwen3-8B: 32): ablated and reported, flagged as
    near-linear, never in the onset, the layer choice or task_from."""
    prof = set(profile_layers(n_layers))
    return [m for m in range(LAYER_STEP, n_layers - 1, LAYER_STEP) if m not in prof]


def h4_layers(layers: list[int], n_layers: int, arrival_index: float | None = None) -> list[int]:
    """Layers at which the H4 rows are read (amendment 2, B2): profile layers at or below two thirds of depth
    (Qwen3-8B: up to 24), and, once the patching arrival layer is known, strictly before it."""
    keep = [m for m in layers if m <= int(H4_MAX_DEPTH * n_layers)]
    if arrival_index is not None and arrival_index == arrival_index:
        keep = [m for m in keep if m < arrival_index]
    return keep


def onset_layer(profile: dict[int, dict], threshold: float = ONSET_THRESHOLD) -> int | None:
    """The earliest profile layer from which the removal's lower bound stays at or above `threshold` at every
    later profile layer (amendment 2, B1); None if the deepest layer's bound is below it. Reading "every layer
    from L on" is an intersection of per-layer tests, so the per-layer 95% bounds need no correction."""
    layers = sorted(profile)
    ok = [(profile[m].get("ci") or (float("nan"), float("nan")))[0] for m in layers]
    onset = None
    for m, lo in zip(reversed(layers), reversed(ok)):
        if not (lo == lo and lo >= threshold):
            break
        onset = m
    return onset


def onset_versus_arrival(onset: int | None, arrival_index: float | None, layers: list[int],
                         deepest_upper: float | None = None, step: int = LAYER_STEP) -> str:
    """The onset layer against the patching arrival layer (amendment 2, B1; amendment 3, C1), without hypothesis
    labels (the minimal pilot reports no verdicts).

    The arrival layer is rounded up to the ablation grid (after rounding to 6 decimals, so that 32.000000001
    is 32). It is "not assessable" when there is no arrival or it lies after the deepest layer the onset was
    read on (`layers`). Without an onset, "no single direction" only when the deepest layer's removal upper
    bound is below 0.3; otherwise the interval is too wide to say."""
    if arrival_index is None or arrival_index != arrival_index:
        return "not assessable (no arrival on the grid)"
    grid_arrival = step * math.ceil(round(arrival_index, 6) / step)
    if not layers:
        return "not assessable (every profile layer excluded by the direct-path rule)"
    if grid_arrival > max(layers):
        return "not assessable (arrival after the profile)"
    if onset is None:
        if deepest_upper is not None and deepest_upper == deepest_upper and deepest_upper < NO_DIRECTION_UPPER:
            return "no single direction"
        return "inconclusive (no onset, interval too wide)"
    return "at or before arrival" if onset <= grid_arrival else "after arrival"


def grid_layers(n_layers: int, step: int = LAYER_STEP) -> list[int]:
    """Every fourth layer from 4 before the last (Qwen3-8B: 4, ..., 32), the grid both arrival and onset use."""
    return list(range(step, n_layers - 1, step))


def patch_arrival_layer(curve: dict, layers: list[int], threshold: float = ONSET_THRESHOLD) -> int | None:
    """The patching arrival layer under the onset's rule (amendment 3, C1): the earliest grid layer from which the
    noising fraction's pointwise 95% lower bound stays at or above `threshold`, on the `last` group (the answer
    position, where the ablation acts). Replaces the final group at a point value of 0.5 for the comparison."""
    ci = curve.get("ci") or []
    return onset_layer({m: {"ci": ci[m]} for m in layers if m < len(ci)}, threshold)


def direct_path_share(dA: Any, dS: Any, DA: Any, DS: Any, games: list[str], ids: list[str],
                      weights: dict[str, float] | None = None, B: int = 2000, seed: int = 0) -> dict:
    """The share of the goal effect already in readout form at one layer (amendment 3, C2):

    sum_i (r.h_A/rho_A - r.h_S/rho_S) / sum_i (D_A - D_S), with r the readout direction gamma * (W_U[Yes] -
    W_U[No]), h the layer's answer-position residual, rho the prompt's last-layer RMS (the final norm's scale)
    and D the final readout. `dA`, `dS` are the per-item r.h/rho of the aligned and swapped prompts; weighted,
    game bootstrap. A mean ablation at that layer removes this part by construction as far as the ablated
    direction lies along it."""
    import numpy as np

    num = np.asarray(dA, dtype=float) - np.asarray(dS, dtype=float)
    den = np.asarray(DA, dtype=float) - np.asarray(DS, dtype=float)
    if not len(num):
        return {"n": 0}
    W = game_weights(games, B, seed) * item_weights(ids, weights)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = (W @ num) / (W @ den)
    return {"n": len(num), "share": float(lam[0]), "ci": _interval(lam[1:]), "se": _se(lam[1:])}


def direct_path_kept(layers: list[int], direct: dict | None, bound: float = DIRECT_PATH_MAX) -> list[int]:
    """Profile layers on which the onset is read (amendment 3, C2): those whose direct-path share has a 95% upper
    bound below `bound`. A layer without a direct-path estimate is kept."""
    if not direct:
        return list(layers)
    out = []
    for m in layers:
        x = direct.get(m) or direct.get(str(m)) or {}
        hi = (x.get("ci") or (float("nan"), float("nan")))[1]
        if not (hi == hi and hi >= bound):
            out.append(m)
    return out


def n_for_half_width(n: int, se: float, half_width: float = TARGET_HALF_WIDTH, z: float = 1.96,
                     margin: float = SAMPLE_SIZE_MARGIN) -> int | None:
    """Items needed for a 95% interval of the given half-width, from a pilot SE at n items (amendment 2, B3): the
    SE scales as 1/sqrt(n), game clustering included, since the SE is a game-bootstrap one; times `margin`
    (amendment 3, C6: 1.2, because with games as clusters the SE falls more slowly than 1/sqrt(n))."""
    if not n or se != se or se <= 0:
        return None
    return int(math.ceil(margin * n * (z * se / half_width) ** 2))


# ---------------- ablation ----------------


def _edit_value(rec: dict, name: str) -> tuple[float, float] | None:
    v = rec["edits"].get(name)
    if v is None:
        return None
    return rec.get("edit_clean", {}).get(name, rec["d"]), v


def removed_share(recs_by_goal: dict[str, dict[str, dict]], ids: list[str], name_for: Callable[[str], str],
                  games: dict[str, str], B: int = 2000, seed: int = 0, a: str = "aligned",
                  b: str = "swapped", weights: dict[str, float] | None = None) -> dict:
    """1 - (sum over items of the edited a-minus-b effect) / (sum of the unedited effect), with `name_for(id)`
    giving each item's edit (so each fold can use the layer chosen on the other); game bootstrap; items
    weighted by `weights` (amendment 2, B4)."""
    import numpy as np

    rows = []
    for i in ids:
        ra, rb = recs_by_goal[a].get(i), recs_by_goal[b].get(i)
        if not ra or not rb:
            continue
        va, vb = _edit_value(ra, name_for(i)), _edit_value(rb, name_for(i))
        if va is None or vb is None:
            continue
        rows.append((va[0] - vb[0], va[1] - vb[1], games[i], i))
    if not rows:
        return {"n": 0}
    before = np.array([[r[0] for r in rows]])
    after = np.array([[r[1] for r in rows]])
    W = game_weights([r[2] for r in rows], B, seed) * item_weights([r[3] for r in rows], weights)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 1 - (W @ after.T)[:, 0] / (W @ before.T)[:, 0]
    return {"n": len(rows), "share": float(share[0]), "ci": _interval(share[1:]), "se": _se(share[1:]),
            "effect_before": float(before.mean()), "effect_after": float(after.mean())}


def mean_shift(recs: dict[str, dict], ids: list[str], name_for: Callable[[str], str], games: dict[str, str],
               B: int = 2000, seed: int = 0, weights: dict[str, float] | None = None) -> dict:
    """Mean of edited minus unedited readout over the items' prompts (collateral shift, or a steering gain)."""
    import numpy as np

    rows = [(v[1] - v[0], games[i], i) for i in ids if i in recs for v in [_edit_value(recs[i], name_for(i))] if v]
    if not rows:
        return {"n": 0}
    x = np.array([[r[0] for r in rows]])
    W = game_weights([r[1] for r in rows], B, seed) * item_weights([r[2] for r in rows], weights)[None, :]
    m = (W @ x.T)[:, 0] / W.sum(1)
    return {"n": len(rows), "mean": float(m[0]), "ci": _interval(m[1:])}


def steer_ratio(by: dict[str, dict[str, dict]], ids: list[str], name_for: Callable[[str], str],
                games: dict[str, str], B: int = 2000, seed: int = 0,
                weights: dict[str, float] | None = None) -> dict:
    """Steering gain on swapped prompts as a share of the goal effect on the same items (amendment 1, A5): the
    summed edited-minus-unedited readout of the swapped prompts over the summed unedited aligned-minus-swapped
    effect."""
    import numpy as np

    rows = []
    for i in ids:
        ra, rs = by["aligned"].get(i), by["swapped"].get(i)
        v = _edit_value(rs, name_for(i)) if rs else None
        if ra is None or v is None:
            continue
        rows.append((v[1] - v[0], ra["d"] - rs["d"], games[i], i))
    if not rows:
        return {"n": 0}
    gain = np.array([[r[0] for r in rows]])
    eff = np.array([[r[1] for r in rows]])
    W = game_weights([r[2] for r in rows], B, seed) * item_weights([r[3] for r in rows], weights)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (W @ gain.T)[:, 0] / (W @ eff.T)[:, 0]
    return {"n": len(rows), "ratio": float(ratio[0]), "ci": _interval(ratio[1:])}


def steer_relative(by: dict[str, dict[str, dict]], ids: list[str], name_num: Callable[[str], str],
                   name_den: Callable[[str], str], games: dict[str, str], B: int = 2000, seed: int = 0,
                   weights: dict[str, float] | None = None) -> dict:
    """Summed steering gain of one edit over that of another on the same swapped prompts (amendment 3, C3: the
    gain along the success direction without its readout component over the gain along the task direction at
    the same layer); weighted, game bootstrap."""
    import numpy as np

    rows = []
    for i in ids:
        rs = by["swapped"].get(i)
        a = _edit_value(rs, name_num(i)) if rs else None
        b = _edit_value(rs, name_den(i)) if rs else None
        if a is None or b is None:
            continue
        rows.append((a[1] - a[0], b[1] - b[0], games[i], i))
    if not rows:
        return {"n": 0}
    x = np.array([[r[0] for r in rows]])
    y = np.array([[r[1] for r in rows]])
    W = game_weights([r[2] for r in rows], B, seed) * item_weights([r[3] for r in rows], weights)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (W @ x.T)[:, 0] / (W @ y.T)[:, 0]
    return {"n": len(rows), "ratio": float(ratio[0]), "ci": _interval(ratio[1:]), "se": _se(ratio[1:])}


def gap(recs_neutral: dict[str, dict], success_ids: list[str], failure_ids: list[str],
        name_for: Callable[[str], str] | None, games: dict[str, str], B: int = 2000, seed: int = 0,
        weights: dict[str, float] | None = None) -> dict:
    """Success-versus-failure gap in nats on neutral prompts: mean D over successes minus mean D over failures,
    unedited, and edited if `name_for` is given (amendment 1, A3: the AUC is at its ceiling, the gap is not).
    Undefined (`defined` False) without successes, as at step 0, where no determined success exists."""
    import numpy as np

    if not success_ids:
        return {"defined": False, "why": "no determined success in this stratum (none exists at step 0)"}

    def val(i: str, edited: bool) -> float | None:
        r = recs_neutral.get(i)
        if not r:
            return None
        if not edited:
            return r["d"]
        v = _edit_value(r, name_for(i))
        return None if v is None else v[1]

    out: dict[str, Any] = {"defined": True}
    draws: dict[str, Any] = {}
    for label, edited in (("clean", False), ("edited", True)):
        if edited and name_for is None:
            continue
        rows = [(val(i, edited), 1.0, games[i], i) for i in success_ids] + \
            [(val(i, edited), 0.0, games[i], i) for i in failure_ids]
        rows = [r for r in rows if r[0] is not None]
        if not rows or not any(r[1] for r in rows) or all(r[1] for r in rows):
            out[label] = {"n": 0}
            continue
        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        W = game_weights([r[2] for r in rows], B, seed) * item_weights([r[3] for r in rows], weights)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            g = (W @ (x * y)) / (W @ y) - (W @ (x * (1 - y))) / (W @ (1 - y))
        from wf.interp.directions import auc

        out[label] = {"n": len(rows), "gap": float(g[0]), "ci": _interval(g[1:]),
                      "auc": auc([r[0] for r in rows if r[1]], [r[0] for r in rows if not r[1]])}
        draws[label] = (g, [r[3] for r in rows])
    if "edited" in out and out["clean"].get("n") and out["edited"].get("n"):
        out["ratio"] = out["edited"]["gap"] / out["clean"]["gap"] if out["clean"]["gap"] else float("nan")
        if draws["clean"][1] == draws["edited"][1]:  # the same items in the same order: a paired ratio per draw
            with np.errstate(invalid="ignore", divide="ignore"):
                r = draws["edited"][0] / draws["clean"][0]
            out["ratio_ci"], out["ratio_se"] = _interval(r[1:]), _se(r[1:])
    return out


def choose_layer(shares_by_layer: dict[int, float]) -> int | None:
    """The layer with the largest removed share (ties to the earlier layer); None if none is finite."""
    ok = {k: v for k, v in shares_by_layer.items() if v == v}
    return min(ok, key=lambda k: (-ok[k], k)) if ok else None


def ablation_summary(recs: list[dict], fold_of: dict[str, int], succ_ids: set[str], layers: list[int],
                     caveat: list[int] = (), last: int | None = None, weights: dict[str, float] | None = None,
                     B: int = 2000, seed: int = 0, direct: dict[str, dict] | None = None) -> dict:
    """The ablation readouts per stratum (amendment 1, A3, A5; amendment 2, B1, B2; amendment 3, C2, C3) from
    edit records. `direct` ({stratum: {layer: direct_path_share}}) excludes from the onset every profile layer
    whose direct-path share's upper bound reaches 0.3 (`direct_path_kept`).

    `layers` is the depth profile; `caveat` layers are ablated and reported but enter nothing else; `last` is
    the last layer's sanity edit ("task_last", expected to remove about everything by construction). Every item
    was edited with directions estimated on the other fold's games. Per stratum:

    - profile: the task direction's removed share and 95% interval at every profile and caveat layer;
    - onset_layer: the earliest profile layer from which the interval's lower bound stays at or above 0.6
      (`onset_layer`), the pre-registered summary, read on the profile layers the direct-path rule keeps
      (`onset_layers`), with the deepest of them's upper bound (`deepest_upper`);
    - by_layer: at every profile layer, the removal by the task direction without its success component, by the
      mention direction and from that layer on; the steering gain along the success direction and along it
      without its readout component and along the task direction itself, as shares of the goal effect, and the
      readout-free success steering over the task steering (`steer_perp_over_task`); the perp-over-task removal
      ratio where the task removal's lower bound reaches 0.3; the collateral shift on neutral prompts; the
      neutral success-versus-failure gap; the removal on determined successes;
    - the same readouts at one chosen layer, the profile layer with the largest removal on the other fold's
      failures (cross-fitted) or, when only one fold was edited (the minimal pilot, amendment 2, B3), on the
      fold's own failures, flagged as in-sample.
    """
    by = index_records(recs)
    meta = {r["id"]: r for r in recs}
    games = {r["id"]: r["game_id"] for r in recs}

    def ids_of(f: int | None, kind: str, s: str) -> list[str]:
        return sorted(i for i, x in fold_of.items() if (f is None or x == f) and ((i in succ_ids) == (kind == "succ"))
                      and i in meta and in_stratum(meta[i], s))

    folds_edited = sorted({fold_of[i] for i in meta if i in fold_of})
    chosen: dict[int, int | None] = {}
    choice: dict[int, str] = {}
    for f in folds_edited:
        other, own = ids_of(1 - f, "fail", "all"), ids_of(f, "fail", "all")
        src = other if other else own
        choice[f] = "cross-fitted (other fold)" if other else "in-sample (only this fold was edited)"
        shares = {m: removed_share(by, src, lambda _i, m=m: f"task|{m}", games, B=200, seed=seed,
                                   weights=weights).get("share", float("nan")) for m in layers}
        chosen[f] = choose_layer(shares)
    name = lambda kind: (lambda i: f"{kind}|{chosen[fold_of[i]]}")  # noqa: E731
    at = lambda kind, m: (lambda _i: f"{kind}|{m}")  # noqa: E731
    out: dict[str, Any] = {"chosen_layer_by_fold": chosen, "layer_choice": choice, "layers": list(layers),
                           "caveat_layers": list(caveat), "last_layer": last, "folds_edited": folds_edited}
    for s in STRATA:
        fails, succ = ids_of(None, "fail", s), ids_of(None, "succ", s)
        row: dict[str, Any] = {"n_failures": len(fails), "n_successes": len(succ)}
        row["profile"] = {m: removed_share(by, fails, at("task", m), games, B, seed, weights=weights)
                          for m in list(layers) + list(caveat)}
        kept = direct_path_kept(layers, (direct or {}).get(s))
        row["onset_layers"] = kept
        row["onset_layer"] = onset_layer({m: row["profile"][m] for m in kept})
        row["deepest_upper"] = (row["profile"][max(kept)].get("ci") or (float("nan"),) * 2)[1] if kept else float("nan")
        if direct and s in direct:
            row["direct_path"] = direct[s]
        if last is not None:
            row["task_last"] = removed_share(by, fails, at("task_last", last), games, B, seed, weights=weights)
        row["by_layer"] = {m: {
            "task_perp_success": removed_share(by, fails, at("task_perp_success", m), games, B, seed, weights=weights),
            "mention": removed_share(by, fails, at("mention", m), games, B, seed, weights=weights),
            "task_from": removed_share(by, fails, at("task_from", m), games, B, seed, weights=weights),
            "steer_success": steer_ratio(by, fails, at("steer_success", m), games, B, seed, weights=weights),
            "steer_success_perp_readout": steer_ratio(by, fails, at("steer_success_perp_readout", m), games, B,
                                                      seed, weights=weights),
            "steer_task": steer_ratio(by, fails, at("steer_task", m), games, B, seed, weights=weights),
            "steer_perp_over_task": steer_relative(by, fails, at("steer_success_perp_readout", m), at("steer_task", m),
                                                   games, B, seed, weights=weights),
            "perp_over_task_removal": perp_over_task(row["profile"][m], by, fails, m, games, B, seed, weights),
            "collateral_neutral_shift": mean_shift(by["absent"], fails + succ, at("task", m), games, B, seed,
                                                   weights=weights),
            "gap_neutral": gap(by["absent"], succ, fails, at("task", m), games, B, seed, weights=weights),
            "task_on_successes": (removed_share(by, succ, at("task", m), games, B, seed, weights=weights) if succ
                                  else {"n": 0, "defined": False, "why": "no determined success in this stratum"}),
        } for m in layers}
        for kind in ("task", "task_perp_success", "mention", "task_from"):
            row[kind] = removed_share(by, fails, name(kind), games, B, seed, weights=weights)
        row["task_on_successes"] = (removed_share(by, succ, name("task"), games, B, seed, weights=weights) if succ
                                    else {"n": 0, "defined": False, "why": "no determined success in this stratum"})
        row["steer_success_gain_on_swapped"] = mean_shift(by["swapped"], fails, name("steer_success"), games, B, seed,
                                                          weights=weights)
        row["steer_success_ratio"] = steer_ratio(by, fails, name("steer_success"), games, B, seed, weights=weights)
        row["steer_success_perp_readout_ratio"] = steer_ratio(by, fails, name("steer_success_perp_readout"), games, B,
                                                              seed, weights=weights)
        row["collateral_neutral_shift"] = mean_shift(by["absent"], fails + succ, name("task"), games, B, seed,
                                                     weights=weights)
        row["gap_neutral"] = gap(by["absent"], succ, fails, name("task"), games, B, seed, weights=weights)
        row["random_sanity"] = {k: removed_share(by, fails, lambda _i, k=k: k, games, 200, seed).get("share")
                                for k in sorted({k for r in recs for k in r["edits"] if k.startswith("random|")})}
        out[s] = row
    return out


def perp_over_task(task_row: dict, by: dict, fails: list[str], m: int, games: dict[str, str], B: int, seed: int,
                   weights: dict[str, float] | None) -> dict:
    """Removal by the task direction without its success component over removal by the task direction at layer m
    (amendment 3, C3), read only where the task direction's removal has a lower bound of at least 0.3; elsewhere
    the ratio of two small numbers says nothing and it is "not assessable"."""
    import numpy as np

    lo = (task_row.get("ci") or (float("nan"),))[0]
    if not (lo == lo and lo >= PERP_TASK_MIN_REMOVAL):
        return {"assessable": False, "why": f"task removal's lower bound {lo:.3g} is below {PERP_TASK_MIN_REMOVAL}"}
    rows = []
    for i in fails:
        ra, rb = by["aligned"].get(i), by["swapped"].get(i)
        if not ra or not rb:
            continue
        vals = [(_edit_value(ra, f"{k}|{m}"), _edit_value(rb, f"{k}|{m}")) for k in ("task", "task_perp_success")]
        if any(a is None or b is None for a, b in vals):
            continue
        (ta, tb), (pa, pb) = vals
        rows.append((ta[0] - tb[0], ta[1] - tb[1], pa[1] - pb[1], games[i], i))
    if not rows:
        return {"assessable": False, "why": "no items"}
    before = np.array([r[0] for r in rows])
    W = game_weights([r[3] for r in rows], B, seed) * item_weights([r[4] for r in rows], weights)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        rt = 1 - (W @ np.array([r[1] for r in rows])) / (W @ before)
        rp = 1 - (W @ np.array([r[2] for r in rows])) / (W @ before)
        ratio = rp / rt
    return {"assessable": True, "ratio": float(ratio[0]), "ci": _interval(ratio[1:]), "se": _se(ratio[1:])}


def index_records(records: Iterable[dict]) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by[r["goal"]][r["id"]] = r
    return by
