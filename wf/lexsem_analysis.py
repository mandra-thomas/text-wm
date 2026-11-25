"""Lexsem diagnostic: estimands, game bootstrap and the pre-committed readings (notes/lexsem_design.md, sections 6
to 8). Reads the scorer's per-item records (scripts/lexsem/score.py), its item file and its recognition checks;
writes one metrics dictionary. The rules themselves are in wf/lexsem_rules.py.

The core package (Qwen3-8B in plain text, the core condition group, first-token and yes/no readouts) is required.
Every optional arm is read only if its scores are present: "Ignore the X", the position arm, E4's reliability
pair, the full two-candidate score (the "fail" format), and separate score files for the with-block yes/no arm and
the Llama yes/no arm. An arm that is absent is reported as "not scored".

    python -m wf.lexsem_analysis --scores runs/lexsem/qwen3-8b_plain.jsonl --out runs/lexsem/analysis.json \
        [--block runs/lexsem/qwen3-8b_block.jsonl] [--llama runs/lexsem/llama3.1-8b_chat.jsonl] \
        [--expect-commit <tag commit>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from wf import lexsem
from wf import lexsem_rules as R

SEED = 20260924
B_DEFAULT = 2000
MASS_FIRST, MASS_FIRST_GAP, MASS_YN = 0.7, 0.1, 0.9  # gate 2
MIN_RECOG = {"later": 100, "step0": 50}  # gate 5
STEP0_BRIDGE_BLOCK = (0.29, 0.23, 0.37)  # stage 0 with the think block, same items (section 8)
REF_FRACTION = 0.5  # a ratio is read only if its reference's interval lies above zero and above half its estimate
# Numerical floor (nats) of a mean paired effect under a change of environment, per model and format, from
# notes/numerical_sensitivity.md (2026-11-25): 0.13 for Qwen with the empty think block, 0.05 for Llama; for Qwen in
# plain text, which that note calls less extreme than the block case, the upper end (0.08) of its measured range for
# Qwen means over a few hundred items. Rules adopted from that note: a margin must be at least MARGIN_MULT times the
# floor, and a reading whose interval ends within FRAGILE_MULT times the floor of a boundary is "numerically fragile".
FLOORS = {"plain": 0.08, "block": 0.13, "chat": 0.05}
MARGIN_MULT, FRAGILE_MULT = 5, 2
NUMERICAL_FLOOR = FLOORS["plain"]
LEVEL_BOUNDS = (-0.25, 0.25, 0.75, 1.25)
STAGE0_ALLOWANCE = 0.12  # environment alone moves "new minus archived" stage-0 means by up to about this much
READOUT_LABELS = {"first": "first-token observation", "yn": "yes/no", "full2c": "full two-candidate"}
# What every score file must share (section 10, decided by the user 2026-11-25): GPU type, stack, dtype, scoring.
ENV_REQUIRED = {"gpu_contains": "L40S", "dtype": "bfloat16", "unbatched": True}
# contrast -> (condition, its control); the rules read the ones in R.PREDICTIONS
CONTRASTS = {c: (c, k) for c, k in lexsem.CONTROL_OF.items()}
COMPETITION = ("aligned", "aligned_ctrl_assert", "swapped", "ctrl_assert")


# ---------------- readouts ----------------

def value(rec: dict, cond: str, readout: str) -> float | None:
    """One readout of one condition for one item record (None if not scored)."""
    s = rec["scores"].get(cond, {})
    obs, yn, fail = s.get("obs"), s.get("yn"), s.get("fail")
    if readout == "first":
        return obs["lp_a"] - obs["lp_b"] if obs else None
    if readout == "you":
        return obs["lp_a"] if obs else None
    if readout == "nothing":
        return obs["lp_b"] if obs else None
    if readout == "success_text":
        return sum(obs["tok_lp"]) if obs and "tok_lp" in obs else None
    if readout == "yn":
        return yn["lp_a"] - yn["lp_b"] if yn else None
    if readout == "full2c":
        return sum(obs["tok_lp"]) - sum(fail["tok_lp"]) if obs and fail else None
    raise ValueError(readout)


def mass(rec: dict, cond: str, fmt: str) -> float | None:
    s = rec["scores"].get(cond, {}).get(fmt)
    return math.exp(s["lp_a"]) + math.exp(s["lp_b"]) if s else None


# ---------------- game bootstrap ----------------

class Boot:
    """Game-clustered bootstrap by draw counts: each draw resamples the games with replacement, and any sum over
    items becomes (counts @ per-game sums). One Boot is shared by every estimand of a stratum, so ratios, differences
    and weights are computed from the same draws."""

    def __init__(self, games: list[str], B: int, seed: int):
        self.games = sorted(set(games))
        self.index = {g: i for i, g in enumerate(self.games)}
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.games), size=(B, len(self.games)))
        self.counts = np.zeros((B, len(self.games)))
        np.add.at(self.counts, (np.repeat(np.arange(B), len(self.games)), draws.ravel()), 1)

    def sums(self, games: list[str], x: np.ndarray) -> tuple[float, np.ndarray]:
        per = np.zeros(len(self.games))
        np.add.at(per, [self.index[g] for g in games], x)
        return float(per.sum()), self.counts @ per


def interval(point: float, draws: np.ndarray) -> dict:
    d = draws[np.isfinite(draws)]
    lo, hi = np.quantile(d, [0.025, 0.975]) if len(d) else (math.nan, math.nan)
    return {"estimate": point, "ci": [float(lo), float(hi)], "basic_ci": [float(2 * point - hi), float(2 * point - lo)],
            "n_draws_undefined": int(len(draws) - len(d))}


def fragile(ci: list[float], ref: float, bounds: tuple[float, ...]) -> bool:
    """An interval end within FRAGILE_MULT floors (expressed in units of the reference) of a zone boundary."""
    if not ref or not all(map(math.isfinite, ci)):
        return False
    tol = FRAGILE_MULT * NUMERICAL_FLOOR / abs(ref)
    return any(abs(e - b) < tol for e in ci for b in bounds)


def mean(boot: Boot, games: list[str], x: np.ndarray) -> tuple[dict, np.ndarray]:
    n, nd = boot.sums(games, x)
    k, kd = boot.sums(games, np.ones_like(x))
    return {**interval(n / k if k else math.nan, nd / kd), "n_items": len(games)}, nd / kd


def ratio(boot: Boot, games: list[str], num: np.ndarray, den: np.ndarray) -> tuple[dict, np.ndarray]:
    """A ratio of means with its numerator and denominator (in nats) from the same draws, their correlation across
    draws, and whether the denominator is well enough estimated for the ratio to be read (REF_FRACTION)."""
    n, nd = boot.sums(games, num)
    d, dd = boot.sums(games, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = nd / dd
    out = {**interval(n / d if d else math.nan, draws), "n_items": len(games)}
    num_iv, num_dr = mean(boot, games, num)
    den_iv, den_dr = mean(boot, games, den)
    out["numerator_nats"], out["denominator_nats"] = num_iv, den_iv
    ok = np.isfinite(num_dr) & np.isfinite(den_dr)
    sd = (np.std(num_dr[ok]), np.std(den_dr[ok])) if ok.any() else (0.0, 0.0)
    out["num_den_correlation"] = float(np.corrcoef(num_dr[ok], den_dr[ok])[0, 1]) if min(sd) > 0 else None
    lo, est = den_iv["ci"][0], den_iv["estimate"]
    out["reference_ok"] = bool(lo > 0 and lo >= REF_FRACTION * est)
    out["resolvable"] = bool(abs(est) * 0.25 >= MARGIN_MULT * NUMERICAL_FLOOR)
    out["numerically_fragile"] = fragile(out["ci"], est, LEVEL_BOUNDS)
    out["numerator_within_floor"] = bool(-NUMERICAL_FLOOR <= num_iv["ci"][0] and num_iv["ci"][1] <= NUMERICAL_FLOOR)
    return out, draws


def agreement(boot: Boot, games: list[str], g: np.ndarray, m: np.ndarray) -> dict:
    """Correlation and slope of the mention effect m on the goal effect g across items, from per-game sufficient
    statistics (so each draw is an exact recomputation on the resampled items)."""
    stats = [g, m, g * g, m * m, g * m, np.ones_like(g)]
    pts, drs = zip(*(boot.sums(games, s) for s in stats))

    def rs(sx, sy, sxx, syy, sxy, n):
        cov, vx, vy = sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
        with np.errstate(divide="ignore", invalid="ignore"):
            return cov / np.sqrt(vx * vy), cov / vx

    r_pt, s_pt = rs(*(np.array(p) for p in pts))
    r_dr, s_dr = rs(*drs)
    r = interval(float(r_pt), r_dr)
    return {"correlation": r, "slope": interval(float(s_pt), s_dr), "reading": R.agreement_reading(*r["ci"])}


# ---------------- one stratum, one readout ----------------

def _cat(iv: dict) -> list[str]:
    return sorted(R.category(*iv["ci"]))


def _paired(recs: list[dict], conds: tuple[str, ...], readout: str) -> list[dict]:
    return [r for r in recs if all(value(r, c, readout) is not None for c in conds)]


def mass_gate(recs: list[dict], readout: str) -> dict:
    """Gate 2: which conditions fail (first token: mean mass at least 0.7 and within 0.1 of swapped; yes/no: 0.9)."""
    fmt = "obs" if readout in ("first", "you", "nothing", "success_text", "full2c") else "yn"
    means = {}
    for c in {c for r in recs for c in r["scores"]}:
        v = [m for r in recs if (m := mass(r, c, fmt)) is not None]
        if v:
            means[c] = sum(v) / len(v)
    if readout == "yn":
        failing = sorted(c for c, v in means.items() if v < MASS_YN)
    elif readout == "first":
        sw = means.get("swapped", 1.0)
        failing = sorted(c for c, v in means.items() if v < MASS_FIRST or abs(v - sw) > MASS_FIRST_GAP)
    else:
        failing = []
    return {"mean_mass": {k: round(v, 4) for k, v in sorted(means.items())}, "failing": failing}


def recognised(rec: dict, form: str, checks: dict | None) -> bool:
    if checks is None:
        return True
    key = (form, rec["goal_object"])
    return bool(checks.get(key, False))


def analyse(recs: list[dict], readout: str, boot: Boot, checks: dict | None, stratum: str) -> dict:
    """Every estimand and reading of section 6 to 8 for one stratum and one readout."""
    out: dict = {"readout": readout, "n_items": len(recs)}
    base = _paired(recs, ("aligned", "swapped"), readout)
    if not base:
        return {**out, "status": "no scores"}
    games = [r["game_id"] for r in base]
    g = np.array([value(r, "aligned", readout) - value(r, "swapped", readout) for r in base])
    G, _ = mean(boot, games, g)
    out["G"] = G
    gate1 = (G["ci"][0] > 0 and G["ci"][0] >= 0.5 * G["estimate"]
             and 0.25 * G["estimate"] >= MARGIN_MULT * NUMERICAL_FLOOR)
    out["numerical_floor_nats"] = NUMERICAL_FLOOR
    out["gate_goal_effect"] = "passed" if gate1 else "failed: nats only, no reading"
    gate2 = mass_gate(recs, readout)
    out["gate_mass"] = gate2
    if readout == "yn" and gate2["failing"]:
        out["readout_invalid"] = gate2["failing"]

    def eff(conds: tuple[str, ...]) -> str | None:
        """Gate 2's fallback: a first-token contrast whose conditions fail is read on log p(" You")."""
        if readout == "first" and set(conds) & set(gate2["failing"]):
            return "you"
        if readout == "yn" and set(conds) & set(gate2["failing"]):
            return None
        return readout

    deltas, rho, rho_iv, items_by = {}, {}, {}, {}
    name_pair = ("name_assert", "ctrl_assert")
    ro_a = eff(name_pair)
    if ro_a is None:
        return {**out, "status": "name assertion invalid in this readout"}
    sub = _paired(base, name_pair, ro_a)
    games_a = [r["game_id"] for r in sub]
    m_a = np.array([value(r, "name_assert", ro_a) - value(r, "ctrl_assert", ro_a) for r in sub])
    g_a = np.array([value(r, "aligned", ro_a) - value(r, "swapped", ro_a) for r in sub])
    A, A_draws = ratio(boot, games_a, m_a, g_a)
    A["readout_used"] = ro_a
    out["A"] = A
    A["numerically_fragile"] = fragile(A["ci"], G["estimate"], LEVEL_BOUNDS)
    out["mention_reading"] = R.mention_reading(*A["ci"]) if gate1 else "not read: goal-effect gate failed"
    out["relevance"] = R.relevance_reading(*A["ci"]) if gate1 else "not read: goal-effect gate failed"
    rho_draws = {}
    for c, (cond, ctrl) in CONTRASTS.items():
        if cond == "name_assert" or not any(cond in r["scores"] for r in base):
            continue
        ro = eff((cond, ctrl) + name_pair)
        if ro is None:
            rho[c] = {"status": "readout invalid"}
            continue
        s = _paired(base, (cond, ctrl) + name_pair, ro)
        form = lexsem.MENTIONS[cond][0]
        if form in ("desc", "alias"):
            s_all, s = s, [r for r in s if recognised(r, form, checks)]
            if len(s) < MIN_RECOG[stratum]:
                rho[c] = {"status": f"not assessable: {len(s)} items passed the recognition check"}
                continue
        if not s:
            continue
        gs = [r["game_id"] for r in s]
        num = np.array([value(r, cond, ro) - value(r, ctrl, ro) for r in s])
        den = np.array([value(r, "name_assert", ro) - value(r, "ctrl_assert", ro) for r in s])
        iv, dr = ratio(boot, gs, num, den)
        iv["readout_used"] = ro
        iv["category"] = _cat(iv)
        if not iv["reference_ok"]:
            iv["ratio_not_read"] = ("the name assertion's net effect on these items has an interval that includes zero "
                                    "or a lower bound below half its estimate: absolute effects only")
        elif not iv["resolvable"]:
            iv["ratio_not_read"] = (f"below numerical resolution: a quarter of the reference is less than "
                                    f"{MARGIN_MULT} times the {NUMERICAL_FLOOR}-nat floor: absolute effects only")
        if form in ("desc", "alias"):
            snum = np.array([value(r, cond, ro) - value(r, ctrl, ro) for r in s_all])
            sden = np.array([value(r, "name_assert", ro) - value(r, "ctrl_assert", ro) for r in s_all])
            iv["all_items"] = ratio(boot, [r["game_id"] for r in s_all], snum, sden)[0]
        rho[c], rho_draws[c], items_by[c] = iv, dr, s
        deltas[c] = mean(boot, gs, num)[0]
        if c in R.PREDICTIONS and iv["reference_ok"] and iv["resolvable"]:
            rho_iv[c] = tuple(iv["ci"])
    out["delta_nats"] = deltas
    out["relative_lift"] = rho
    statuses = R.account_statuses(rho_iv)
    out["accounts"] = statuses
    out["single_account_reading"] = (R.single_account_reading(rho_iv, tuple(A["ci"])) if gate1
                                     else "not read: goal-effect gate failed")
    out["broad_evidence"] = R.broad_evidence_status(statuses)
    out["numerically_fragile"] = sorted([c for c in rho_iv if rho[c].get("numerically_fragile")]
                                        + (["A"] if A.get("numerically_fragile") else []))
    basic = {c: tuple(rho[c]["basic_ci"]) for c in rho_iv}
    basic_reading = R.single_account_reading(basic, tuple(A["basic_ci"]))
    out["interval_sensitive"] = basic_reading != out["single_account_reading"] and gate1
    # mixture weights from the same draws
    keys = [k for k in R.PURE["lexical"] if k in rho_draws]
    if A["ci"][0] >= R.A_GATE and keys:
        X = np.array([[R.PURE[a][k] for a in R.ACCOUNTS] for k in keys], dtype=float)
        pts = np.array([rho[k]["estimate"] for k in keys])
        w_pt = np.linalg.lstsq(X, pts, rcond=None)[0]
        Y = np.stack([rho_draws[k] for k in keys], axis=1)
        ok = np.all(np.isfinite(Y), axis=1)
        W = np.linalg.lstsq(X, Y[ok].T, rcond=None)[0].T if ok.any() else np.full((1, 3), math.nan)
        out["mixture_weights"] = {a: interval(float(w_pt[i]), W[:, i]) for i, a in enumerate(R.ACCOUNTS)}
        out["shares"] = {"name_share": 1 - rho["desc_assert"]["estimate"] if "desc_assert" in rho_iv else None,
                         "claim_share_listed": 1 - rho["name_use"]["estimate"] if "name_use" in rho_iv else None}
    else:
        out["mixture_weights"] = "not read (A below 0.5 or no relative lifts)"
    # item-level agreement with the goal effect
    agr = {}
    for c, s in {"name_assert": sub, **items_by}.items():
        cond, ctrl = CONTRASTS[c]
        ro = rho[c]["readout_used"] if c in rho else ro_a
        gs = [r["game_id"] for r in s]
        gg = np.array([value(r, "aligned", ro) - value(r, "swapped", ro) for r in s])
        mm = np.array([value(r, cond, ro) - value(r, ctrl, ro) for r in s])
        agr[c] = agreement(boot, gs, gg, mm)
    out["item_agreement"] = agr
    if "homonym" in rho_iv:
        out["homonym_reading"] = R.homonym_reading(*rho_iv["homonym"])
    # optional arms
    if "name_ignore" in rho and "estimate" in rho["name_ignore"]:
        out["ignore"] = {"relative_lift": rho["name_ignore"],
                         "reading": R.hint_relevance_reading(*rho["name_ignore"]["ci"])}
    else:
        out["ignore"] = "not scored"
    pos = ("name_assert_pre", "ctrl_assert_pre", "name_assert_post", "ctrl_assert_post")
    s = _paired(base, pos + name_pair, readout) if eff(pos + name_pair) == readout else []
    if s:
        gs = [r["game_id"] for r in s]
        net = {p: np.array([value(r, f"name_{p}", readout) - value(r, f"ctrl_{p}", readout) for r in s])
               for p in ("assert", "assert_pre", "assert_post")}
        gg = np.array([value(r, "aligned", readout) - value(r, "swapped", readout) for r in s])
        rec_iv = ratio(boot, gs, net["assert_pre"] - net["assert"], gg)[0]
        vis_iv = ratio(boot, gs, net["assert_post"] - net["assert_pre"], gg)[0]
        rec_m, vis_m = ("recency, with the region change, matters",
                        "visibility to the action's tokens, with the region change, matters")
        for iv_ in (rec_iv, vis_iv):
            iv_["numerically_fragile"] = fragile(iv_["ci"], G["estimate"], (-0.25, 0.25))
        out["position"] = {
            "recency": {**rec_iv, "reading": R.difference_reading(*rec_iv["ci"], rec_m,
                                                                   "no recency effect beyond 0.25", rec_m)},
            "visibility": {**vis_iv, "reading": R.difference_reading(*vis_iv["ci"], vis_m,
                                                                      "no visibility effect beyond 0.25", vis_m)},
            "confounds": "task line to history end also changes the prompt region; history end to after the action "
                         "also moves the sentence next to the reply label"}
    else:
        out["position"] = "not scored"
    rel = ("name_reliable", "ctrl_reliable", "name_unreliable", "ctrl_unreliable")
    s = _paired(base, rel, readout) if eff(rel) == readout else []
    if s:
        gs = [r["game_id"] for r in s]
        d = np.array([(value(r, "name_reliable", readout) - value(r, "ctrl_reliable", readout))
                      - (value(r, "name_unreliable", readout) - value(r, "ctrl_unreliable", readout)) for r in s])
        gg = np.array([value(r, "aligned", readout) - value(r, "swapped", readout) for r in s])
        iv = ratio(boot, gs, d, gg)[0]
        iv["numerically_fragile"] = fragile(iv["ci"], G["estimate"], (-0.25, 0.25))
        out["reliability"] = {**iv, "reading": R.difference_reading(*iv["ci"], "the unreliable source lifts more",
                                                                     "no reliability effect beyond 0.25",
                                                                     "the reliable source lifts more"),
                              "scope": "ordering by stated reliability at most (one inequality), net of each "
                                       "sentence's control object; on items whose history already settles the "
                                       "outcome, with settledness and redundancy not crossed (no unsettled items; no "
                                       "sampled history shows the object); no proportional or quantitative claim"}
    else:
        out["reliability"] = "not scored"
    # competition: the control object's assertion after the real task, net of the same sentence after the swapped
    # task (stored rq1 values: Qwen -1.09 nats, Llama -0.06, two-candidate, with the block for Qwen)
    s = _paired(base, COMPETITION, readout) if eff(COMPETITION) == readout else []
    if s:
        gs = [r["game_id"] for r in s]
        d = np.array([(value(r, "aligned_ctrl_assert", readout) - value(r, "aligned", readout))
                      - (value(r, "ctrl_assert", readout) - value(r, "swapped", readout)) for r in s])
        gg = np.array([value(r, "aligned", readout) - value(r, "swapped", readout) for r in s])
        iv = ratio(boot, gs, d, gg)[0]
        iv["numerically_fragile"] = fragile(iv["ci"], G["estimate"], (-0.25, 0.25))
        out["competition"] = {**iv, "reading": R.difference_reading(
            *iv["ci"], "competition: naming another object after the real task lowers its reading",
            "no competition beyond 0.25 of the goal effect",
            "naming another object after the real task raises its reading")}
    else:
        out["competition"] = "not scored"
    # the stage-0 bridge, as stage 0 defined it (minus the plain swapped task), and net of the control
    s = _paired(base, ("name_assert",), readout)
    if s:
        gs = [r["game_id"] for r in s]
        gg = np.array([value(r, "aligned", readout) - value(r, "swapped", readout) for r in s])
        b = ratio(boot, gs, np.array([value(r, "name_assert", readout) - value(r, "swapped", readout) for r in s]),
                  gg)[0]
        out["bridge"] = {**b, "net_of_control": A}
        if stratum == "step0":
            out["bridge"]["reading"] = R.bridge_reading(*b["ci"])
    return out


# ---------------- whole run ----------------

def load_scores(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def load_checks(path: str | Path | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    out = {}
    for c in json.loads(Path(path).read_text()):
        out[(c["form"], c["object"])] = out.get((c["form"], c["object"]), True) and c["passed"]
    return out


def attach(scores: list[dict], items: list[dict]) -> list[dict]:
    meta = {r["id"]: r for r in items}
    out = []
    for s in scores:
        m = meta[s["id"]]
        refs = m.get("references", {})
        out.append({**s, "goal_object": m["goal_object"], "fewshot_object": m["fewshot_object"],
                    "stratum": m["stratum"], "game_id": m["game_id"],
                    "prevalence_gap": (refs["target_prevalence"] - refs["ctrl_prevalence"]
                                       if "ctrl_prevalence" in refs else None)})
    return out


def prevalence_covariate(recs: list[dict], ro: str, boot: Boot) -> dict | str:
    """Control audit covariate (descriptive): does the name assertion's net effect vary with how much more often the
    action's object than the control object occurs in ALFWorld rooms? Correlation and slope across items."""
    rs = [r for r in _paired(recs, ("name_assert", "ctrl_assert"), ro) if r.get("prevalence_gap") is not None]
    if len(rs) < 10:
        return "not available"
    gs = [r["game_id"] for r in rs]
    gap = np.array([r["prevalence_gap"] for r in rs])
    net = np.array([value(r, "name_assert", ro) - value(r, "ctrl_assert", ro) for r in rs])
    res = agreement(boot, gs, gap, net)
    return {"correlation": res["correlation"], "slope_nats_per_unit_gap": res["slope"],
            "mean_gap": float(gap.mean())}


def run_model(recs: list[dict], readouts: tuple[str, ...], checks: dict | None, B: int, seed: int,
              floor: float | None = None) -> dict:
    global NUMERICAL_FLOOR
    if floor is not None:
        NUMERICAL_FLOOR = floor
    out: dict = {"numerical_floor_nats": NUMERICAL_FLOOR}
    strata = ("later", "step0")
    boots = {s: Boot([r["game_id"] for r in recs if r["stratum"] == s], B, seed) for s in strata}
    joint = Boot([r["game_id"] for r in recs], B, seed)
    for ro in readouts:
        out[ro] = {}
        for s in strata:
            rs = [r for r in recs if r["stratum"] == s]
            out[ro][s] = analyse(rs, ro, boots[s], checks, s)
            out[ro][s]["prevalence_covariate"] = prevalence_covariate(rs, ro, boots[s])
            out[ro][s]["without_fewshot_objects"] = analyse([r for r in rs if not r["fewshot_object"]], ro,
                                                            boots[s], checks, s)
        out[ro]["step"] = step_comparison(recs, ro, joint, checks, out[ro])
    if "first" in out and "you" in readouts:
        out["normalization_flags"] = flags(out["first"], out["you"])
    # each readout keeps its own reading; disagreements are reported, none decides for the others
    out["readings_by_readout"] = {READOUT_LABELS[ro]: {s: out[ro][s].get("single_account_reading") for s in strata}
                                  for ro in readouts if ro in READOUT_LABELS}
    pairs = [(a, b) for a in ("first", "yn", "full2c") for b in ("first", "yn", "full2c") if a < b]
    out["readout_disagreements"] = {f"{READOUT_LABELS[a]} vs {READOUT_LABELS[b]}": flags(out[a], out[b])
                                    for a, b in pairs if a in out and b in out}
    return out


def step_comparison(recs: list[dict], ro: str, joint: Boot, checks: dict | None, per: dict) -> dict:
    """Step 0 minus later for A and every relative lift, from one bootstrap over the games of both strata."""
    res = {}
    by = {s: per[s] for s in ("later", "step0")}
    if "A" not in by["later"] or "A" not in by["step0"]:
        return {"status": "not assessable"}

    def recompute(s: str, c: str) -> np.ndarray | None:
        a = by[s]
        rs = [r for r in recs if r["stratum"] == s]
        name_pair = ("name_assert", "ctrl_assert")
        if c == "A":
            rr = _paired(rs, ("aligned", "swapped") + name_pair, a["A"]["readout_used"])
            gs = [r["game_id"] for r in rr]
            ro_ = a["A"]["readout_used"]
            return ratio(joint, gs, np.array([value(r, "name_assert", ro_) - value(r, "ctrl_assert", ro_) for r in rr]),
                         np.array([value(r, "aligned", ro_) - value(r, "swapped", ro_) for r in rr]))[1]
        iv = a["relative_lift"].get(c, {})
        if "readout_used" not in iv:
            return None
        cond, ctrl = CONTRASTS[c]
        ro_ = iv["readout_used"]
        rr = _paired(rs, ("aligned", "swapped", cond, ctrl) + name_pair, ro_)
        form = lexsem.MENTIONS[cond][0]
        if form in ("desc", "alias"):
            rr = [r for r in rr if recognised(r, form, checks)]
        gs = [r["game_id"] for r in rr]
        return ratio(joint, gs, np.array([value(r, cond, ro_) - value(r, ctrl, ro_) for r in rr]),
                     np.array([value(r, "name_assert", ro_) - value(r, "ctrl_assert", ro_) for r in rr]))[1]

    diffs = {}
    for c in ["A"] + sorted(R.PREDICTIONS):
        d0, d1 = recompute("step0", c), recompute("later", c)
        if d0 is None or d1 is None:
            continue
        pt = ((by["step0"]["A"]["estimate"] - by["later"]["A"]["estimate"]) if c == "A"
              else by["step0"]["relative_lift"][c]["estimate"] - by["later"]["relative_lift"][c]["estimate"])
        diffs[c] = interval(pt, d0 - d1)
        ref = by["later"]["G"]["estimate"] if c == "A" else by["later"]["relative_lift"][c]["denominator_nats"]["estimate"]
        diffs[c]["numerically_fragile"] = fragile(diffs[c]["ci"], ref, (-0.25, 0.25))
    if "A" in diffs:
        res["A"] = {**diffs["A"], "reading": R.difference_reading(*diffs["A"]["ci"], "step 0 lower",
                                                                    "no step difference beyond 0.25",
                                                                    "step 0 higher")}
    res["relative_lifts"] = {c: v for c, v in diffs.items() if c != "A"}
    if "accounts" in per["later"] and "accounts" in per["step0"]:
        res["stratum_specific_accounts"] = R.stratum_specific(
            per["step0"]["accounts"], per["later"]["accounts"],
            {c: tuple(v["ci"]) for c, v in diffs.items() if c != "A"})
    return res


def flags(a: dict, b: dict) -> dict:
    """Contrasts whose categories in two readouts share no zone, per stratum."""
    out = {}
    for s in ("later", "step0"):
        fa, fb = a.get(s, {}), b.get(s, {})
        names = []
        if "A" in fa and "A" in fb and not R.category(*fa["A"]["ci"]) & R.category(*fb["A"]["ci"]):
            names.append("A")
        for c, iv in fa.get("relative_lift", {}).items():
            jv = fb.get("relative_lift", {}).get(c, {})
            if "ci" in iv and "ci" in jv and not R.category(*iv["ci"]) & R.category(*jv["ci"]):
                names.append(c)
        out[s] = names
    return out


def check_env(paths: list[Path]) -> list[str]:
    """Every score file must come with an environment record, and all must share GPU type, torch, transformers,
    dtype and unbatched scoring, with each model at its pinned revision."""
    from wf.constants import REVISIONS

    problems, seen = [], set()
    for p in paths:
        e = p.parent / f"{p.stem}_env.json"
        if not e.exists():
            problems.append(f"{p.name}: no environment record")
            continue
        env = json.loads(e.read_text())
        if ENV_REQUIRED["gpu_contains"] not in env.get("gpu", ""):
            problems.append(f"{p.name}: GPU {env.get('gpu')}")
        if env.get("dtype") != ENV_REQUIRED["dtype"] or env.get("unbatched") is not True:
            problems.append(f"{p.name}: dtype {env.get('dtype')}, unbatched {env.get('unbatched')}")
        if REVISIONS.get(env.get("model")) != env.get("revision"):
            problems.append(f"{p.name}: revision {env.get('revision')} is not the pinned one")
        seen.add((env.get("torch"), env.get("transformers")))
    if len(seen) > 1:
        problems.append(f"score files come from different stacks: {sorted(seen)}")
    return problems


def bridge_compare(plain: list[dict], block: list[dict], B: int, seed: int) -> dict:
    """The stage-0 bridge in plain text against the same prompts rescored with the think block in this run (same
    items, stack, GPU type and draws): (name assertion minus swapped) over the goal effect, yes/no."""
    by = {r["id"]: r for r in block}
    out = {}
    for s in ("later", "step0"):
        rs = [r for r in plain if r["stratum"] == s and r["id"] in by]
        rs = [r for r in rs if all(value(x, c, "yn") is not None for x in (r, by[r["id"]])
                                   for c in ("aligned", "swapped", "name_assert"))]
        if not rs:
            continue
        boot = Boot([r["game_id"] for r in rs], B, seed)
        gs = [r["game_id"] for r in rs]
        res, dr = {}, {}
        for name, src in (("plain", lambda r: r), ("block", lambda r: by[r["id"]])):
            num = np.array([value(src(r), "name_assert", "yn") - value(src(r), "swapped", "yn") for r in rs])
            den = np.array([value(src(r), "aligned", "yn") - value(src(r), "swapped", "yn") for r in rs])
            res[name], dr[name] = ratio(boot, gs, num, den)
        diff = interval(res["plain"]["estimate"] - res["block"]["estimate"], dr["plain"] - dr["block"])
        reading = R.difference_reading(*diff["ci"], "plain text lower than the block format",
                                       "the same in both formats (within 0.25)", "plain text higher than the block format")
        diff["numerically_fragile"] = fragile(diff["ci"], res["block"]["denominator_nats"]["estimate"],
                                              (-0.25, 0.25))
        out[s] = {**res, "plain_minus_block": diff, "reading": reading}
        if s == "step0" and reading == "plain text higher than the block format":
            out[s]["stage0_reading"] = "stage 0's low step-0 share belongs to the block format"
    return out


def ignore_compare(arms: dict[str, list[dict]], B: int, seed: int) -> dict:
    """"Ignore the X" across arms on the same items (design section 5): its relative lift (net of its control, over
    the name assertion's net effect) in Qwen plain text, Qwen with the block and Llama, and the pre-specified
    reading of whether Qwen's share without the block falls toward Llama's. Yes/no for all three; first token for
    the two Qwen arms (Llama's first token is reported, not read)."""
    conds = ("name_ignore", "ctrl_ignore", "name_assert", "ctrl_assert")
    out = {}
    for ro in ("yn", "first"):
        for st in ("later", "step0"):
            have = {a: {r["id"]: r for r in recs if r["stratum"] == st
                        and all(value(r, c, ro) is not None for c in conds)} for a, recs in arms.items()}
            have = {a: v for a, v in have.items() if v}
            if "qwen_plain" not in have or len(have) < 2:
                continue
            ids = sorted(set.intersection(*(set(v) for v in have.values())))
            first = have["qwen_plain"]
            boot = Boot([first[i]["game_id"] for i in ids], B, seed)
            gs = [first[i]["game_id"] for i in ids]
            res, dr = {}, {}
            for a, v in have.items():
                num = np.array([value(v[i], "name_ignore", ro) - value(v[i], "ctrl_ignore", ro) for i in ids])
                den = np.array([value(v[i], "name_assert", ro) - value(v[i], "ctrl_assert", ro) for i in ids])
                res[a], dr[a] = ratio(boot, gs, num, den)
            entry: dict = {"relative_lift_ignore": res, "n_items": len(ids)}
            diffs = {}
            for other in ("qwen_block", "llama_chat"):
                if other in res:
                    diffs[other] = interval(res["qwen_plain"]["estimate"] - res[other]["estimate"],
                                            dr["qwen_plain"] - dr[other])
            entry["plain_minus"] = diffs
            if {"qwen_block", "llama_chat"} <= set(diffs):
                b, l_ = diffs["qwen_block"]["ci"], diffs["llama_chat"]["ci"]
                falls = b[1] < -0.25
                stays_block = -0.25 <= b[0] and b[1] <= 0.25
                like_llama = -0.25 <= l_[0] and l_[1] <= 0.25
                above_llama = l_[0] > 0.25
                if falls and like_llama:
                    reading = "block-dependent: without the block, Qwen's 'Ignore' share falls to Llama's"
                elif falls and above_llama:
                    reading = "partly block-dependent: it falls without the block but stays above Llama's"
                elif stays_block and above_llama:
                    reading = "model-level: without the block Qwen's 'Ignore' share stays at its with-block size, above Llama's"
                else:
                    reading = "inconclusive"
                entry["reading"] = reading + "; neither outcome identifies a mechanism"
            elif "qwen_block" in diffs:
                entry["reading"] = R.difference_reading(*diffs["qwen_block"]["ci"], "lower without the block",
                                                        "the same with and without the block (within 0.25)",
                                                        "higher without the block") + " (no Llama comparison in this readout)"
            out[f"{ro}|{st}"] = entry
    return out


def stage0_paired(index_path: str, block: list[dict], B: int, seed: int) -> dict:
    """Archived stage-0 scores (original stack) against the same prompts rescored here (serve2): the difference in
    paired effects (goal effect, mention effect), not only per score."""
    st = {}
    for line in Path(index_path).read_text().splitlines():
        r = json.loads(line)
        if r["fmt"] == "yn" and r["goal"] in ("aligned", "swapped", "swapped_plus"):
            st.setdefault(r["id"], {})[r["goal"]] = r["d"]
    out = {}
    for s in ("later", "step0"):
        rs = [r for r in block if r["stratum"] == s and len(st.get(r["id"], {})) == 3
              and all(value(r, c, "yn") is not None for c in ("aligned", "swapped", "name_assert"))]
        if not rs:
            continue
        boot = Boot([r["game_id"] for r in rs], B, seed)
        gs = [r["game_id"] for r in rs]
        a = {r["id"]: st[r["id"]] for r in rs}
        new_g = np.array([value(r, "aligned", "yn") - value(r, "swapped", "yn") for r in rs])
        old_g = np.array([a[r["id"]]["aligned"] - a[r["id"]]["swapped"] for r in rs])
        new_m = np.array([value(r, "name_assert", "yn") - value(r, "swapped", "yn") for r in rs])
        old_m = np.array([a[r["id"]]["swapped_plus"] - a[r["id"]]["swapped"] for r in rs])
        per = np.array([abs(value(r, c, "yn") - a[r["id"]][o]) for r in rs
                        for c, o in (("aligned", "aligned"), ("swapped", "swapped"), ("name_assert", "swapped_plus"))])
        gdiff, mdiff = mean(boot, gs, new_g - old_g)[0], mean(boot, gs, new_m - old_m)[0]
        within = lambda iv: (-STAGE0_ALLOWANCE <= iv["ci"][0] and iv["ci"][1] <= STAGE0_ALLOWANCE)  # noqa: E731
        out[s] = {"goal_effect_new_minus_archived": gdiff, "mention_effect_new_minus_archived": mdiff,
                  "numerical_allowance_nats": STAGE0_ALLOWANCE,
                  "reading": ("within the numerical allowance of a change of environment" if within(gdiff)
                              and within(mdiff) else "beyond the numerical allowance in at least one paired effect; "
                              "read as a difference between environments, never as a change in the model"),
                  "mean_abs_per_score_difference": float(per.mean()), "n_items": len(rs)}
    return out


def run(scores_path: str, out_path: str | None = None, items_path: str | None = None, checks_path: str | None = None,
        block_path: str | None = None, llama_path: str | None = None, B: int = B_DEFAULT, seed: int = SEED,
        expect_commit: str | None = None, stage0_index: str | None = None, require_env: bool = True,
        floor: float | None = None) -> dict:
    sp = Path(scores_path)

    def floor_for(path: Path, default_mode: str) -> float:
        if floor is not None:
            return floor
        e = path.parent / f"{path.stem}_env.json"
        mode = json.loads(e.read_text()).get("mode", default_mode) if e.exists() else default_mode
        return FLOORS[mode]
    if require_env:
        problems = check_env([sp] + [Path(x) for x in (block_path, llama_path) if x and Path(x).exists()])
        if problems:
            raise SystemExit("environment check failed: " + "; ".join(problems))
    items_path = items_path or sp.parent / f"{sp.stem}_items.jsonl"
    checks_path = checks_path or sp.parent / f"{sp.stem}_check.json"
    if expect_commit:
        for p in [sp] + [Path(x) for x in (block_path, llama_path) if x]:
            c = p.parent / f"{p.stem}_commit.txt"
            got = c.read_text().strip() if c.exists() else None
            if got != expect_commit:
                raise SystemExit(f"{p.name}: scored at {got}, not at the frozen commit {expect_commit}")
    items = load_scores(items_path)
    checks = load_checks(checks_path)
    recs = attach(load_scores(sp), items)
    has_fail = any("fail" in v for r in recs for v in r["scores"].values())
    readouts = ("first", "yn", "you", "nothing", "success_text") + (("full2c",) if has_fail else ())
    report = {"numerical_rules": {"floors_nats": FLOORS, "margin_multiple": MARGIN_MULT,
                                  "fragile_multiple": FRAGILE_MULT, "stage0_allowance_nats": STAGE0_ALLOWANCE},
              "recognition_checks": "read" if checks is not None else "absent: every item counted as recognised",
              "qwen_plain": run_model(recs, readouts, checks, B, seed, floor_for(sp, "plain")),
              "full_two_candidate": "scored" if has_fail else "not scored"}
    for name, path in (("block_yes_no", block_path), ("llama_yes_no", llama_path)):
        if not path or not Path(path).exists():
            report[name] = "not scored"
            continue
        own = load_checks(Path(path).parent / f"{Path(path).stem}_check.json")  # each model's own recognition
        arm = attach(load_scores(path), items)
        ros = ("yn",) + (("first",) if any("obs" in v for r in arm for v in r["scores"].values()) else ())
        report[name] = run_model(arm, ros, own if own is not None else checks, B,
                                 seed, floor_for(Path(path), "block" if name == "block_yes_no" else "chat"))
    if isinstance(report["llama_yes_no"], dict):
        report["model_comparison_yes_no"] = flags(report["qwen_plain"]["yn"], report["llama_yes_no"]["yn"])
    arms = {"qwen_plain": recs}
    if block_path and Path(block_path).exists():
        arms["qwen_block"] = attach(load_scores(block_path), items)
    if llama_path and Path(llama_path).exists():
        arms["llama_chat"] = attach(load_scores(llama_path), items)
    report["ignore_across_arms"] = ignore_compare(arms, B, seed) if len(arms) > 1 else "not scored"
    if block_path and Path(block_path).exists():
        block = attach(load_scores(block_path), items)
        report["bridge_plain_vs_block"] = bridge_compare(recs, block, B, seed)
        report["archived_stage0_vs_rescored"] = (stage0_paired(stage0_index, block, B, seed) if stage0_index
                                                  else "not computed")
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=1, default=float))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items")
    ap.add_argument("--checks")
    ap.add_argument("--block")
    ap.add_argument("--llama")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--expect-commit")
    ap.add_argument("--stage0-index")
    a = ap.parse_args()
    rep = run(a.scores, a.out, a.items, a.checks, a.block, a.llama, a.B, SEED, a.expect_commit, a.stage0_index)
    print(json.dumps(rep["qwen_plain"]["readings_by_readout"], indent=1))


if __name__ == "__main__":
    main()
