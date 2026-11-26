"""The pre-specified analysis (notes/bayesenv_design.md, sections 7 to 12) and the smoke readout (section 9).

Scores: for a prompt, L = log p(first candidate) - log p(second candidate) at the readout position, the model's
log-odds of success normalized over the two candidates. Exact references come from `world`. Items are the statistical
unit; cells (generator x history) are strata with equal weight. Almost every estimand is linear in cell means,
sum_c w_c mean_i y_ci, with variance sum_c w_c^2 var_i(y_ci) / n_c. Estimands that average over goal or tag wordings
add the wording variance estimated from the wording panel (eight wordings of each on eight cells) by a two-way
random-effects decomposition, with Satterthwaite degrees of freedom. Slopes use a stratified item bootstrap.

Every requested cell must have complete items: a missing cell or prompt type stops the analysis rather than changing
an estimand's definition.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from wf.bayesenv import design as D
from wf.bayesenv import world as W

PRIMARY = ("wm", "first")
READOUTS = (("wm", "first"), ("wm", "yn"), ("si", "narr"), ("si", "yn"))
ALF_READOUT_KEYS = (("alf", "first"), ("alf", "yn"))
QUESTION_FREE = (("wm", "first"), ("si", "narr"), ("alf", "first"))
SLOPE_HISTS = ("H0", "H2", "H3", "H6", "H9")    # graded histories in which a cue's Bayes effect is non-zero
BOOT = 2000
SEED = 20261125
ALPHA = 0.05
STD_MARGIN = 0.20             # equivalence margin on the standardized scale (>= 5 x 0.034)
REF_SCALE = 0.7               # the evidence slope at which the pattern target of 0.5 nats was set
PATTERN_TARGET_AT_REF = 0.5   # nats of the model's log-odds at evidence slope 0.7
FLOOR_CLASSES = {             # numerical floor constants by readout logit size (notes/numerical_sensitivity.md 4.8)
    "small": (0.013, 0.20),   # top-candidate logits below 32 (Llama-sized)
    "large": (0.04, 0.58),    # 32 to 64 (Qwen3-8B-sized)
    "very_large": (0.08, 1.16),  # 64 or more: twice the large class
}
DESIGN_EFFECT = 1.3
GATE_SLOPE_LOWER = 0.1        # competence: the evidence slope's 95% lower bound must exceed this
GATE_MASS = 0.5               # median two-candidate mass
GATE_FORM = 0.8               # sampled continuations starting with a candidate word that take that candidate's form
GATE_OTHER = 0.5              # at most this share of sampled continuations start with neither candidate word
GATE_MIN_SAMPLES = 30         # sampled continuations needed per candidate word for the form gate
SIZING_KEYS = ("goal_minus_tag_negative_allowed", "patterns_not_for_negative")


# ----------------------------------------------------------------------------- loading


@dataclass
class Scores:
    """L[(framing, readout)][(gen, hist, ptype)] = {k: L}; plus mass and raw logits for gates."""
    L: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))
    mass: dict = field(default_factory=lambda: defaultdict(list))
    zmax: dict = field(default_factory=lambda: defaultdict(list))
    lpa: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))

    def get(self, ro: tuple, g: str, h: str, t: str) -> dict:
        return self.L[ro].get((g, h, t), {})


def load_records(records) -> Scores:
    s = Scores()
    for rec in records:
        space = rec["space"]
        for key, v in rec["scores"].items():
            t, fr, r = key.split("|")
            if (space == "alf") != (fr == "alf"):
                raise ValueError("ALFWorld records, and only they, carry framing 'alf'")
            ro = (fr, r)
            s.L[ro][(rec["gen"], rec["hist"], t)][rec["k"]] = v["lp_a"] - v["lp_b"]
            s.lpa[ro][(rec["gen"], rec["hist"], t)][rec["k"]] = v["lp_a"]
            s.mass[ro].append(math.exp(v["lp_a"]) + math.exp(v["lp_b"]))
            s.zmax[ro].append(max(v["z_a"], v["z_b"]))
    return s


def read_folders(folders) -> list[dict]:
    out = []
    for f in folders:
        p = Path(f) / "scoring.jsonl"
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


def verify_hashes(records: list[dict]) -> list[str]:
    """Units whose stored prompt hash differs from the prompts the current code renders."""
    bad = []
    for r in records:
        u = D.Unit(r["space"], r["gen"], r["hist"], r["k"], tuple(dict.fromkeys(k.split("|")[0] for k in r["scores"])))
        if D.unit_hash(u) != r["unit_hash"]:
            bad.append(r["unit"])
    return bad


# ----------------------------------------------------------------------------- linear estimands


def bayes_logit(space: str, t: str, g: str, h: str) -> float:
    return W.logit(D.exact_for(space, t, g, h))


@dataclass
class Est:
    est: float
    se: float
    df: float = math.inf
    n_items: int = 0
    sd_item: float = float("nan")
    n_eff: float = 0.0            # 1 / sum_c (w_c^2 / n_c): the item count with the estimand's precision
    parts: dict = field(default_factory=dict)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        q = stats.t.ppf(0.5 + level / 2, self.df) if math.isfinite(self.df) else stats.norm.ppf(0.5 + level / 2)
        return self.est - q * self.se, self.est + q * self.se

    def p(self) -> float:
        if not self.se > 0:
            return 0.0 if self.est != 0 else 1.0
        z = abs(self.est) / self.se
        return float(2 * (stats.t.sf(z, self.df) if math.isfinite(self.df) else stats.norm.sf(z)))

    def mde(self) -> float:
        df = self.df if math.isfinite(self.df) else 1e9
        return float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * self.se)

    def standardized_ci90(self) -> tuple[float, float] | None:
        if not (self.sd_item and self.sd_item > 0):
            return None
        lo, hi = self.ci(0.90)
        return lo / self.sd_item, hi / self.sd_item

    def to_dict(self) -> dict:
        lo, hi = self.ci()
        lo90, hi90 = self.ci(0.90)
        sc = self.standardized_ci90()
        return {"estimate": self.est, "se": self.se, "df": self.df, "ci95": [lo, hi], "ci90": [lo90, hi90],
                "p": self.p(), "mde": self.mde(), "n_items": self.n_items, "n_eff": self.n_eff,
                "sd_item": self.sd_item,
                "standardized": self.est / self.sd_item if self.sd_item and self.sd_item > 0 else None,
                "standardized_ci90": list(sc) if sc else None, **({"parts": self.parts} if self.parts else {})}


def item_contrast(S: Scores, ro: tuple, g: str, h: str, plus: dict, minus: dict, space: str = "toy",
                  k_max: int | None = None, adjust: bool = True, with_items: bool = False):
    """Per-item value of sum_t w_t L_t (plus) - sum_t w_t L_t (minus). With `adjust`, the same contrast of the exact
    logits is subtracted where every side's exact probability lies strictly inside (0, 1); a contrast in a settled
    cell is allowed only if its sides share the exact probability (its Bayes value is then exactly 0). Items need
    every type; a cell without at least two complete items stops the analysis."""
    types = list(plus) + list(minus)
    ks = sorted(set.intersection(*[set(S.get(ro, g, h, t)) for t in types]))
    if k_max is not None:
        ks = [k for k in ks if k < k_max]
    if len(ks) < 2:
        raise ValueError(f"cell {(g, h)} on {ro} has {len(ks)} complete items for {types}")
    val = np.zeros(len(ks))
    for t, w in plus.items():
        val += w * np.array([S.get(ro, g, h, t)[k] for k in ks])
    for t, w in minus.items():
        val -= w * np.array([S.get(ro, g, h, t)[k] for k in ks])
    if adjust:
        pb = {t: D.exact_for(space, t, g, h) for t in types}
        if all(0 < p < 1 for p in pb.values()):
            val -= (sum(w * W.logit(pb[t]) for t, w in plus.items()) -
                    sum(w * W.logit(pb[t]) for t, w in minus.items()))
        elif len({round(p, 12) for p in pb.values()}) != 1:
            raise ValueError(f"contrast in a settled cell with unequal exact probabilities: {g} {h} {types}")
    return (ks, val) if with_items else val


def linear(parts: dict, weights: dict) -> Est:
    """parts: cell -> per-item contributions (already weighted by `weights[cell]`). Estimate sum_c mean,
    variance sum_c var / n, effective item count 1 / sum_c (w_c^2 / n_c)."""
    est, var, n, inv = 0.0, 0.0, 0, 0.0
    for c, arr in parts.items():
        est += float(arr.mean())
        var += float(arr.var(ddof=1)) / len(arr)
        n += len(arr)
        inv += weights[c] ** 2 / len(arr)
    return Est(est=est, se=math.sqrt(var), n_items=n, n_eff=1 / inv if inv > 0 else 0.0)


def mean_contrast(S: Scores, ro: tuple, cells, plus: dict, minus: dict, space: str = "toy",
                  k_max: int | None = None, adjust: bool = True) -> Est:
    """Equal-weight mean over cells of the per-item contrast; also its pooled within-cell item SD."""
    cells = list(cells)
    parts, weights = {}, {}
    ss, dfs = 0.0, 0
    for g, h in cells:
        arr = item_contrast(S, ro, g, h, plus, minus, space, k_max, adjust=adjust)
        parts[(g, h)], weights[(g, h)] = arr / len(cells), 1 / len(cells)
        ss += float(((arr - arr.mean()) ** 2).sum())
        dfs += len(arr) - 1
    e = linear(parts, weights)
    e.sd_item = math.sqrt(ss / dfs) if dfs else float("nan")
    return e


def combine(a: Est, b: Est, sign: float = -1.0) -> Est:
    """a + sign * b for estimands on disjoint item sets (variances add)."""
    return Est(est=a.est + sign * b.est, se=math.sqrt(a.se ** 2 + b.se ** 2), df=min(a.df, b.df),
               n_items=a.n_items + b.n_items, sd_item=float(np.nanmean([a.sd_item, b.sd_item])),
               n_eff=1 / (1 / a.n_eff + 1 / b.n_eff) if a.n_eff > 0 and b.n_eff > 0 else 0.0)


def linear_from(S: Scores, ro: tuple, terms: list[tuple[float, list, dict, dict]], space: str = "toy",
                k_max: int | None = None) -> Est:
    """sum over terms of weight x (equal-weight mean over that term's cells of the per-item contrast). Terms that
    meet in a cell must use the same items there; their per-item contributions are added, so the covariance of
    terms sharing items is kept."""
    parts: dict = {}
    weights: dict = defaultdict(float)
    for wt, cells, plus, minus in terms:
        for c in cells:
            ks, arr = item_contrast(S, ro, c[0], c[1], plus, minus, space, k_max, with_items=True)
            arr = arr * wt / len(cells)
            weights[c] += abs(wt) / len(cells)
            if c in parts:
                if parts[c][0] != ks:
                    raise ValueError(f"terms meeting in cell {c} use different items")
                parts[c] = (ks, parts[c][1] + arr)
            else:
                parts[c] = (ks, arr)
    return linear({c: v for c, (_, v) in parts.items()}, weights)


# ----------------------------------------------------------------------------- wording random effects


@dataclass
class VarComp:
    """Mean squares of a two-way (wording x cell, items within cells) layout of one cue family on the panel."""
    ms_w: float          # wordings
    ms_wc: float         # wording x cell
    ms_e: float          # wording x item residual
    n_wordings: int
    n_cells: int
    n_items: int

    @property
    def s2_alpha(self) -> float:
        return max(0.0, (self.ms_w - self.ms_wc) / (self.n_items * self.n_cells))

    @property
    def s2_gamma(self) -> float:
        return max(0.0, (self.ms_wc - self.ms_e) / self.n_items)

    @property
    def s2_e(self) -> float:
        return self.ms_e

    def wording_variance(self, w: int, cells: int) -> tuple[float, float]:
        """(variance, Satterthwaite df) of the mean of w wordings' shared effects over `cells` cells:
        (s2_alpha + s2_gamma / cells) / w, estimated without clipping the components separately (the sum is
        clipped at 0)."""
        n, cp = self.n_items, self.n_cells
        coef = {"w": 1 / (n * cp), "wc": -1 / (n * cp) + 1 / (n * cells), "e": -1 / (n * cells)}
        ms = {"w": self.ms_w, "wc": self.ms_wc, "e": self.ms_e}
        dfs = {"w": self.n_wordings - 1, "wc": (self.n_wordings - 1) * (cp - 1),
               "e": (self.n_wordings - 1) * cp * (n - 1)}
        theta = sum(coef[k] * ms[k] for k in ms)
        den = sum((coef[k] * ms[k]) ** 2 / dfs[k] for k in ms)
        df = theta ** 2 / den if den > 0 and theta > 0 else float(self.n_wordings - 1)
        return max(0.0, theta) / w, max(1.0, df)


def wording_components(S: Scores | None, ro: tuple | None, family: str, cells=D.PANEL_CELLS,
                       transform=None) -> VarComp:
    """Two-way random-effects decomposition of one cue family's wordings on the panel cells.
    family "goal": key1..key8; "tag": tag1..tag8. With `transform`, values are transform(g, h, t) dicts."""
    types = [f"{'key' if family == 'goal' else 'tag'}{i + 1}" for i in range(D.PANEL_WORDINGS)]
    blocks = []
    for g, h in cells:
        ds = [(transform(g, h, t) if transform else S.get(ro, g, h, t)) for t in types]
        ks = sorted(set.intersection(*[set(d) for d in ds]))
        if len(ks) < 2:
            # never drop the wording term silently: an incomplete panel stops the analysis
            raise ValueError(f"wording panel incomplete: cell {(g, h)} has {len(ks)} items with all {family} wordings")
        blocks.append(np.array([[d[k] for k in ks] for d in ds]))   # (W, n)
    n = min(b.shape[1] for b in blocks)
    X = np.stack([b[:, :n] for b in blocks], axis=1)                 # (W, C, n)
    Wn, C, n = X.shape
    m_w = X.mean(axis=(1, 2))
    m_c = X.mean(axis=(0, 2))
    m_wc = X.mean(axis=2)
    m_ci = X.mean(axis=0)
    g_ = X.mean()
    ms_w = n * C * ((m_w - g_) ** 2).sum() / (Wn - 1)
    ms_wc = n * ((m_wc - m_w[:, None] - m_c[None, :] + g_) ** 2).sum() / ((Wn - 1) * (C - 1))
    resid = X - m_wc[:, :, None] - m_ci[None, :, :] + m_c[None, :, None]
    ms_e = (resid ** 2).sum() / ((Wn - 1) * C * (n - 1))
    return VarComp(float(ms_w), float(ms_wc), float(ms_e), Wn, C, n)


def add_wording(e: Est, sides: list[tuple[VarComp, int]], n_cells: int) -> Est:
    """Add the sampling variance of the wordings used: for each side (components, number of wordings averaged),
    (s2_alpha + s2_gamma / cells) / W from the panel. Satterthwaite df (the item term has n_items - 1 df)."""
    v_items = e.se ** 2
    terms = [vc.wording_variance(w, n_cells) for vc, w in sides]
    v = v_items + sum(t for t, _ in terms)
    den = (v_items ** 2 / max(e.n_items - 1, 1)) + sum(t ** 2 / df for t, df in terms if t > 0)
    df = v ** 2 / den if den > 0 else math.inf
    return Est(est=e.est, se=math.sqrt(v), df=df, n_items=e.n_items, sd_item=e.sd_item, n_eff=e.n_eff,
               parts={"var_items": v_items, "var_wordings": [t for t, _ in terms]})


# ----------------------------------------------------------------------------- slopes (bootstrap)


def _cell_items(S: Scores, ro: tuple, cells, ytypes: dict, xminus: dict | None, space: str) -> dict:
    """cell -> (y values per item, exact x) for slope regressions; y = sum w L(types) - [xminus types]."""
    out = {}
    for g, h in cells:
        types = list(ytypes) + list(xminus or {})
        ks = sorted(set.intersection(*[set(S.get(ro, g, h, t)) for t in types]))
        if len(ks) < 2:
            raise ValueError(f"cell {(g, h)} on {ro} has {len(ks)} complete items for {types}")
        y = sum(w * np.array([S.get(ro, g, h, t)[k] for k in ks]) for t, w in ytypes.items())
        x = sum(w * bayes_logit(space, t, g, h) for t, w in ytypes.items())
        if xminus:
            y = y - sum(w * np.array([S.get(ro, g, h, t)[k] for k in ks]) for t, w in xminus.items())
            x = x - sum(w * bayes_logit(space, t, g, h) for t, w in xminus.items())
        out[(g, h)] = (y, x)
    return out


def _ols_slope(cells: dict, idx: dict | None = None) -> float:
    xs, ys = [], []
    for c, (y, x) in cells.items():
        yy = y if idx is None else y[idx[c]]
        xs.append(np.full(len(yy), x))
        ys.append(yy)
    X, Y = np.concatenate(xs), np.concatenate(ys)
    vx = X.var()
    return float(((X - X.mean()) * (Y - Y.mean())).mean() / vx) if vx > 0 else float("nan")


def slope(S: Scores, ro: tuple, cells, ytypes: dict, xminus: dict | None = None, space: str = "toy",
          boot: int = BOOT) -> Est:
    """OLS slope of the model's (contrast of) log-odds on the exact one, items pooled; stratified item bootstrap."""
    data = _cell_items(S, ro, cells, ytypes, xminus, space)
    est = _ols_slope(data)
    rng = np.random.default_rng(SEED)
    bs = [_ols_slope(data, {c: rng.integers(0, len(y), len(y)) for c, (y, _) in data.items()}) for _ in range(boot)]
    return Est(est=est, se=float(np.nanstd(bs, ddof=1)), n_items=sum(len(y) for y, _ in data.values()))


def ratio_of_means(S: Scores, ro: tuple, cells, plus: dict, minus: dict, space: str = "toy",
                   boot: int = BOOT) -> Est:
    """mean model contrast / mean exact contrast over cells (exact contrasts finite)."""
    data = _cell_items(S, ro, cells, plus, minus, space)
    num = np.mean([y.mean() for y, _ in data.values()])
    den = np.mean([x for _, x in data.values()])
    rng = np.random.default_rng(SEED)
    bs = [np.mean([y[rng.integers(0, len(y), len(y))].mean() for y, _ in data.values()]) / den for _ in range(boot)]
    return Est(est=float(num / den), se=float(np.std(bs, ddof=1)), n_items=sum(len(y) for y, _ in data.values()))


def ratio(a: Est, b: Est) -> Est:
    """a / b by the delta method, treating a and b as independent (they are estimated on different contrasts)."""
    if not (b.est and math.isfinite(b.est)):
        return Est(float("nan"), float("nan"))
    r = a.est / b.est
    return Est(est=r, se=abs(r) * math.sqrt((a.se / a.est) ** 2 + (b.se / b.est) ** 2) if a.est else a.se / abs(b.est))


# ----------------------------------------------------------------------------- the estimands


GRADED_CELLS = [(g, h) for g in W.GENERATOR_ORDER for h in W.GRADED]
SLOPE_CELLS = [(g, h) for g in W.GENERATOR_ORDER for h in SLOPE_HISTS]
H4_CELLS = [(g, "H4") for g in W.GENERATOR_ORDER]
REDUNDANCY_NOTES = {"note50": 1 / 3, "note80": 1 / 3, "note95": 1 / 3}


def goal_side(w: int) -> dict:
    return {f"key{i + 1}": 1.0 / w for i in range(w)}


def tag_side(w: int) -> dict:
    return {f"tag{i + 1}": 1.0 / w for i in range(w)}


def evidence_slopes(S: Scores, ro: tuple, boot: int = BOOT) -> dict:
    """Overall evidence slope (explore prompts, graded cells), history slope (generators with base rate 0.5) and
    base-rate slope (generator D against the three generators with base rate 0.5, by graded history other than H4,
    where existence is settled and the base rate has no effect)."""
    a = slope(S, ro, GRADED_CELLS, {"explore": 1.0}, boot=boot)
    hist = slope(S, ro, [(g, h) for g in ("A", "R3", "C") for h in W.GRADED], {"explore": 1.0}, boot=boot)
    rng = np.random.default_rng(SEED)
    pairs = []
    for h in SLOPE_HISTS:
        d_vals = np.array(list(S.get(ro, "D", h, "explore").values()))
        o_vals = [np.array(list(S.get(ro, g, h, "explore").values())) for g in ("A", "R3", "C")]
        if len(d_vals) < 2 or any(len(o) < 2 for o in o_vals):
            raise ValueError(f"base-rate slope: incomplete explore items in history {h} on {ro}")
        dx = bayes_logit("toy", "explore", "D", h) - bayes_logit("toy", "explore", "A", h)
        pairs.append((d_vals, o_vals, dx))
    mean_o = lambda os_, draw: np.mean([o[draw(o)].mean() for o in os_])  # noqa: E731
    ident = lambda x: slice(None)  # noqa: E731
    num = np.mean([d.mean() - mean_o(o, ident) for d, o, _ in pairs])
    den = np.mean([dx for _, _, dx in pairs])
    rs = lambda x: rng.integers(0, len(x), len(x))  # noqa: E731
    bs = [np.mean([d[rs(d)].mean() - mean_o(o, rs) for d, o, _ in pairs]) / den for _ in range(boot)]
    base = Est(est=float(num / den), se=float(np.std(bs, ddof=1)), n_items=sum(len(d) for d, _, _ in pairs))
    return {"evidence_slope": a, "history_slope": hist, "base_rate_slope": base}


def by_variant(S: Scores, ro: tuple, cells, plus: dict, minus: dict) -> dict:
    """The estimand separately for the two wordings of the announcement, provenance or existence statement
    (wording = item index mod 2)."""
    out = {}
    for v in (0, 1):
        sub = Scores()
        for (g, h, t), ks in S.L[ro].items():
            sub.L[ro][(g, h, t)] = {k: x for k, x in ks.items() if k % 2 == v}
        out[f"wording_{v + 1}"] = mean_contrast(sub, ro, cells, plus, minus).to_dict()
    return out


def estimands(S: Scores, ro: tuple, d: D.Design, k_max: int | None = None, boot: int = BOOT) -> dict[str, Est]:
    """Every pre-specified estimand for one framing x readout (toy). Names follow section 7 of the design note."""
    cells_all = list(D.CELLS)
    out: dict[str, Est] = {}
    vc_g = wording_components(S, ro, "goal")
    vc_t = wording_components(S, ro, "tag")
    G, T = goal_side(d.w_goal), tag_side(d.w_tag)
    mc = lambda cells, plus, minus: mean_contrast(S, ro, cells, plus, minus, k_max=k_max)  # noqa: E731
    out["goal_minus_tag"] = add_wording(mc(cells_all, G, T), [(vc_g, d.w_goal), (vc_t, d.w_tag)], len(cells_all))
    out["goal_minus_tag_four_prompt"] = add_wording(mc(cells_all, {**G, "tagO": 1.0}, {**T, "explore": 1.0}),
                                                    [(vc_g, d.w_goal), (vc_t, d.w_tag)], len(cells_all))
    a_cells = [(g, h) for g, h in cells_all if g == "A"]
    out["goal_minus_tag_zero_information"] = add_wording(mc(a_cells, G, T), [(vc_g, d.w_goal), (vc_t, d.w_tag)],
                                                         len(a_cells))
    out["tag_naming_zero_information"] = mc(a_cells, {"tag1": 1.0}, {"tagO": 1.0})
    out["competing_goal"] = mc(cells_all, {"coin_tag": 1.0}, {"tag1": 1.0})
    out["competing_task_no_tag"] = mc(cells_all, {"coin": 1.0}, {"explore": 1.0})
    out["name_minus_description"] = add_wording(mc(cells_all, T, {"tagN": 1.0}), [(vc_t, d.w_tag), (vc_t, 1)],
                                                len(cells_all))
    out["goal_zero_information"] = add_wording(mc(a_cells, G, {"explore": 1.0}), [(vc_g, d.w_goal)], len(a_cells))
    out["swap_zero_information"] = mc(a_cells, {"key1": 1.0}, {"coin": 1.0})
    out["named_rate_leakage"] = combine(mc([("C", h) for h in W.GRADED], {"explore": 1.0}, {}),
                                        mc([("A", h) for h in W.GRADED], {"explore": 1.0}, {}))
    out["zero_reliability_note"] = mc(cells_all, {"note0": 1.0}, {"explore": 1.0})
    out["redundancy_residual"] = mc(H4_CELLS, REDUNDANCY_NOTES, {"explore": 1.0})
    out["redundancy_pattern"] = linear_from(S, ro, [(1.0, H4_CELLS, REDUNDANCY_NOTES, {"explore": 1.0}),
                                                    (-1.0, cells_all, {"note0": 1.0}, {"explore": 1.0})],
                                            k_max=k_max)
    out["redundancy_pattern"].sd_item = out["redundancy_residual"].sd_item
    wm = ro[0] == "wm"      # the announced and provenance arms exist only in the world-model framing
    ann, prov = list(D.ANN_CELLS), list(D.PROV_CELLS)
    if wm:
        out["announced_goal_effect"] = mc(ann, {"ann_key": 1.0}, {"ann_coin": 1.0})
        ann_inf = [(g, h) for g, h in ann if g in ("C", "D")]
        ann_a = [(g, h) for g, h in ann if g == "A"]
        out["announced_slope"] = combine(mc(ann_inf, {"ann_key": 1.0}, {"ann_coin": 1.0}),
                                         mc(ann_a, {"ann_key": 1.0}, {"ann_coin": 1.0}))
        out["history_only_provenance"] = mc(prov, {"P1_key": 1.0}, {"key1": 1.0})
        out["history_only_provenance_slope"] = combine(
            mc([c for c in prov if c[0] == "C"], {"P1_key": 1.0}, {"key1": 1.0}),
            mc([c for c in prov if c[0] == "A"], {"P1_key": 1.0}, {"key1": 1.0}))
    # slopes and ratios
    out.update(evidence_slopes(S, ro, boot=boot))
    out["goal_slope"] = slope(S, ro, SLOPE_CELLS, G, {"explore": 1.0}, boot=boot)
    out["tag_slope"] = slope(S, ro, SLOPE_CELLS, T, {"tagO": 1.0}, boot=boot)
    notes = {rho: _cell_items(S, ro, SLOPE_CELLS, {rho: 1.0}, {"explore": 1.0}, "toy") for rho in REDUNDANCY_NOTES}
    pooled = {(c, r): v for r, cs in notes.items() for c, v in cs.items()}
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(boot):   # items resampled within cells, the same draw for every reliability of a cell
        idx_c = {c: rng.integers(0, len(v[0]), len(v[0])) for c, v in notes["note95"].items()}
        bs.append(_ols_slope(pooled, {(c, r): idx_c[c] for (c, r) in pooled}))
    out["note_slope"] = Est(est=_ols_slope(pooled), se=float(np.nanstd(bs, ddof=1)))
    out["goal_calibration_ratio"] = ratio(out["goal_slope"], out["base_rate_slope"])
    out["tag_calibration_ratio"] = ratio(out["tag_slope"], out["base_rate_slope"])
    out["note_calibration_ratio"] = ratio(out["note_slope"], out["history_slope"])
    out["note_ordering"] = mean_contrast(S, ro, SLOPE_CELLS, {"note95": 1.0}, {"note50": 1.0}, k_max=k_max,
                                         adjust=False)
    out["note_ordering"].parts = {"bayes_value": float(np.mean([bayes_logit("toy", "note95", g, h) -
                                                                 bayes_logit("toy", "note50", g, h)
                                                                 for g, h in SLOPE_CELLS]))}
    if wm:
        out["seeing_guide_ratio"] = ratio_of_means(S, ro, prov, {"P2_key": 1.0}, {"key1": 1.0}, boot=boot)
        out["seeing_guide_ratio_over_evidence_slope"] = ratio(out["seeing_guide_ratio"], out["evidence_slope"])
    # matched-information goal against note (descriptive): C against the 0.8 note, D against the 0.95 note
    gm = [mc([(g, h) for h in SLOPE_HISTS], G, {n: 1.0}) for g, n in (("C", "note80"), ("D", "note95"))]
    out["goal_minus_matched_note"] = add_wording(
        Est(est=(gm[0].est + gm[1].est) / 2, se=math.sqrt(gm[0].se ** 2 + gm[1].se ** 2) / 2,
            n_items=gm[0].n_items + gm[1].n_items, sd_item=float(np.mean([gm[0].sd_item, gm[1].sd_item])),
            n_eff=gm[0].n_eff + gm[1].n_eff), [(vc_g, d.w_goal)], 2 * len(SLOPE_HISTS))
    # goal minus tag by generator (with its wording term) and its dependence on the likelihood ratio
    by_gen = {}
    for g in W.GENERATOR_ORDER:
        cg = [(gg, h) for gg, h in cells_all if gg == g]
        by_gen[g] = add_wording(mc(cg, G, T), [(vc_g, d.w_goal), (vc_t, d.w_tag)], len(cg))
        out[f"goal_minus_tag_gen_{g}"] = by_gen[g]
    xs = np.array([math.log(W.likelihood_ratio(g)) for g in W.GENERATOR_ORDER])
    ys = np.array([by_gen[g].est for g in W.GENERATOR_ORDER])
    ws = np.array([1 / max(by_gen[g].se, 1e-9) ** 2 for g in W.GENERATOR_ORDER])
    xb = (ws * xs).sum() / ws.sum()
    b = (ws * (xs - xb) * (ys - (ws * ys).sum() / ws.sum())).sum() / (ws * (xs - xb) ** 2).sum()
    out["goal_minus_tag_lr_slope"] = Est(est=float(b), se=float(math.sqrt(1 / (ws * (xs - xb) ** 2).sum())),
                                         parts={"note": "weights treat the four generators as independent"})
    by_hist = [add_wording(mc([(g, h) for g in W.GENERATOR_ORDER], G, T), [(vc_g, d.w_goal), (vc_t, d.w_tag)], 4)
               for h in W.HISTORY_ORDER]
    ests = np.array([e.est for e in by_hist])
    wts = np.array([1 / max(e.se, 1e-9) ** 2 for e in by_hist])
    q = float((wts * (ests - (wts * ests).sum() / wts.sum()) ** 2).sum())
    out["goal_minus_tag_history_heterogeneity"] = Est(
        est=q, se=float("nan"), parts={"Q": q, "df": len(by_hist) - 1, "p": float(stats.chi2.sf(q, len(by_hist) - 1)),
                                       "by_history": dict(zip(W.HISTORY_ORDER, ests.tolist())),
                                       "note": "wording effects shared across histories make Q approximate"})
    if k_max is None and wm:
        out["announced_goal_effect"].parts = {"by_wording": by_variant(S, ro, ann, {"ann_key": 1.0},
                                                                       {"ann_coin": 1.0})}
        out["history_only_provenance"].parts = {"by_wording": by_variant(S, ro, prov, {"P1_key": 1.0},
                                                                         {"key1": 1.0})}
        out["redundancy_residual"].parts = {"by_wording": by_variant(S, ro, H4_CELLS, REDUNDANCY_NOTES,
                                                                     {"explore": 1.0})}
    return out


HOLM_FAMILY = ("goal_minus_tag", "tag_naming_zero_information", "competing_goal", "name_minus_description",
               "zero_reliability_note", "redundancy_residual", "redundancy_pattern", "announced_goal_effect",
               "announced_slope", "history_only_provenance", "history_only_provenance_slope")
# A pattern contrast is one that a constant tied to one sentence or one mention cannot produce.
PATTERN_CONTRASTS = ("redundancy_pattern", "announced_slope", "history_only_provenance_slope")
# Single residuals: a content-blind constant attached to one cue's text produces any of them on its own.
CONSTANTS = ("tag_naming_zero_information", "zero_reliability_note", "redundancy_residual", "announced_goal_effect",
             "history_only_provenance")
EQUIVALENCE_PATTERNS = ("redundancy_pattern", "announced_slope")


def holm(ps: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    order = sorted(ps, key=lambda k: ps[k])
    m = len(order)
    out, stop = {}, False
    for i, k in enumerate(order):
        if stop or ps[k] > alpha / (m - i):
            stop = True
            out[k] = False
        else:
            out[k] = True
    return out


def floor_class(zmax: list[float]) -> str:
    med = float(np.median(zmax)) if zmax else 0.0
    return "very_large" if med >= 64 else "large" if med >= 32 else "small"


def numerical_floor(cls: str, n_eff: float) -> float:
    tau, sig = FLOOR_CLASSES[cls]
    return math.sqrt(tau ** 2 + DESIGN_EFFECT * sig ** 2 / max(n_eff, 1.0))


def pattern_target(a: float) -> float:
    """0.5 nats at the reference evidence slope 0.7, scaled to the model's own evidence slope."""
    return PATTERN_TARGET_AT_REF * a / REF_SCALE


def check_sizing(sizing: dict) -> None:
    if not isinstance(sizing, dict) or any(k not in sizing for k in SIZING_KEYS):
        raise ValueError(f"the analysis needs the committed sizing outcome with keys {SIZING_KEYS}, got {sizing!r}")


def equivalence(e: Est, margin: float) -> dict:
    """Equivalence needs both the 90% interval inside +-margin and the standardized 90% interval inside +-0.20."""
    lo, hi = e.ci(0.90)
    sc = e.standardized_ci90()
    nats = -margin < lo and hi < margin
    std = sc is not None and -STD_MARGIN < sc[0] and sc[1] < STD_MARGIN
    return {"margin": margin, "nats": bool(nats), "standardized": bool(std), "both": bool(nats and std)}


def readings(E: dict[str, Est], cls: str, low_floor_family: bool, target: float, p_target: float, sizing: dict,
             exclude: tuple = ()) -> dict:
    """Decisions for one readout: Holm over the family (two-sided), the pattern rule, and equivalence where allowed.
    Equivalence (a negative decision) needs: the low-floor family; a measured logit-size class 'small'; the sizing
    rule's permission; and both the nats and the standardized conditions."""
    check_sizing(sizing)
    fam = [k for k in HOLM_FAMILY if k in E and k not in exclude]
    ps = {k: E[k].p() for k in fam}
    rej = holm(ps)
    out: dict = {"holm": {k: {"p": ps[k], "rejected": rej[k], "estimate": E[k].est, "ci95": list(E[k].ci()),
                              "mde": E[k].mde()} for k in fam},
                 "excluded_from_family": list(exclude)}
    pattern = [k for k in PATTERN_CONTRASTS if rej.get(k)]
    constants = [k for k in CONSTANTS if rej.get(k)]
    out["residual_pattern_detected"] = bool(pattern)
    out["pattern_contrasts_detected"] = {k: float(np.sign(E[k].est)) for k in pattern}
    out["content_blind_constants_detected"] = {k: float(np.sign(E[k].est)) for k in constants}
    negative_ok = bool(low_floor_family and cls == "small")
    out["negative_decisions_allowed"] = negative_ok
    gt = E["goal_minus_tag"]
    margin = max(target, 5 * numerical_floor(cls, gt.n_eff))
    eq = equivalence(gt, margin)
    allowed = negative_ok and bool(sizing["goal_minus_tag_negative_allowed"])
    detected = bool(rej.get("goal_minus_tag"))
    if detected:
        label = "detected"
    elif not low_floor_family:
        label = f"not detected at an MDE of {gt.mde():.2f} nats"
    elif not sizing["goal_minus_tag_negative_allowed"]:
        label = "not detected; not assessable for a negative conclusion at this noise (sizing rule)"
    elif allowed and eq["both"]:
        label = f"smaller than the target: equivalent to zero within {margin:.2f} nats and 0.20 standardized"
    else:
        label = "not detected; equivalence not shown"
    out["goal_minus_tag"] = {"detected": detected, "sign": float(np.sign(gt.est)), "target": target,
                             "at_or_above_target": detected and abs(gt.est) >= target, "mde": gt.mde(),
                             "numerical_floor": numerical_floor(cls, gt.n_eff), "equivalence": eq,
                             "equivalent": bool(allowed and eq["both"]), "label": label}
    pat = {}
    for k in [x for x in EQUIVALENCE_PATTERNS if x in E]:
        e = E[k]
        m = max(p_target, 5 * numerical_floor(cls, e.n_eff))
        q = equivalence(e, m)
        ok = negative_ok and k not in sizing["patterns_not_for_negative"]
        pat[k] = {"mde": e.mde(), "equivalence": q, "equivalent": bool(ok and q["both"]),
                  "negative_allowed": ok}
    out["patterns_equivalence"] = pat
    out["no_pattern_shown"] = bool(not pattern and all(v["equivalent"] for v in pat.values()))
    return out


# ----------------------------------------------------------------------------- the specificity rule

SPEC_ESTIMANDS = ("goal_minus_tag", "zero_reliability_note", "redundancy_residual")


def _spec_cells(name: str, d: D.Design):
    G, T = goal_side(d.w_goal), tag_side(d.w_tag)
    return {"goal_minus_tag": (list(D.CELLS), G, T),
            "zero_reliability_note": (list(D.CELLS), {"note0": 1.0}, {"explore": 1.0}),
            "redundancy_residual": (H4_CELLS, REDUNDANCY_NOTES, {"explore": 1.0})}[name]


def normalized_difference(S: Scores, d: D.Design, name: str, ro_a: tuple, ro_b: tuple, slopes: dict,
                          k_max: int | None = None) -> tuple[Est, float]:
    """((estimand in ro_a / slope_a) - (estimand in ro_b / slope_b), the world-model normalized estimate) on the same
    items. Wording variance from the panel on the transformed per-wording values; slope uncertainty by the delta
    method (covariances ignored)."""
    a_a, a_b = slopes[ro_a], slopes[ro_b]
    cells, plus, minus = _spec_cells(name, d)
    ests = [mean_contrast(S, ro, cells, plus, minus, k_max=k_max).est for ro in (ro_a, ro_b)]
    parts, weights = {}, {}
    for g, h in cells:
        ka, xa = item_contrast(S, ro_a, g, h, plus, minus, k_max=k_max, with_items=True)
        kb, xb = item_contrast(S, ro_b, g, h, plus, minus, k_max=k_max, with_items=True)
        if ka != kb:
            raise ValueError(f"specificity: framings scored on different items in cell {(g, h)}")
        parts[(g, h)], weights[(g, h)] = (xa / a_a.est - xb / a_b.est) / len(cells), 1 / len(cells)
    e = linear(parts, weights)
    if name == "goal_minus_tag":
        def tf(g, h, t):
            da, db = S.get(ro_a, g, h, t), S.get(ro_b, g, h, t)
            return {k: da[k] / a_a.est - db[k] / a_b.est for k in set(da) & set(db)}
        e = add_wording(e, [(wording_components(None, None, "goal", transform=tf), d.w_goal),
                            (wording_components(None, None, "tag", transform=tf), d.w_tag)], len(cells))
    v_slope = (ests[0] / a_a.est) ** 2 * (a_a.se / a_a.est) ** 2 + (ests[1] / a_b.est) ** 2 * (a_b.se / a_b.est) ** 2
    return Est(est=e.est, se=math.sqrt(e.se ** 2 + v_slope), df=e.df, n_items=e.n_items), ests[0] / a_a.est


def specificity(S: Scores, d: D.Design, slopes: dict, gates: dict, k_max: int | None = None) -> dict:
    """Section 10. The excess is read in the direction of the world-model effect: with s the sign of the world
    model's slope-normalized estimate on the arm's world-model readout, the arm shows an excess if the 95% interval
    of s x (WM - SI) lies above 0. Specific: both arms show an excess and the two world-model signs agree. Not
    assessable: an arm cannot be computed (a readout failed its gate, or a slope is not positive)."""
    out = {}
    ok = lambda ro: gates.get(ro, {}).get("passed") and slopes[ro].est > 0  # noqa: E731
    for name in SPEC_ESTIMANDS:
        r: dict = {}
        arms = {}
        for arm, (ra, rb) in (("yes_no", (("wm", "yn"), ("si", "yn"))),
                              ("question_free", (("wm", "first"), ("si", "narr")))):
            if ok(ra) and ok(rb):
                diff, wm_norm = normalized_difference(S, d, name, ra, rb, slopes, k_max)
                s = 1.0 if wm_norm >= 0 else -1.0
                lo, _ = Est(s * diff.est, diff.se, diff.df).ci()
                arms[arm] = {"difference": diff.to_dict(), "world_model_sign": s, "excess": bool(lo > 0)}
            else:
                arms[arm] = None
            r[arm] = arms[arm] or "not assessable (a readout failed its gate or its slope is not positive)"
        yn, qf = arms["yes_no"], arms["question_free"]
        if yn is not None and qf is not None:
            if yn["world_model_sign"] != qf["world_model_sign"]:
                r["specific_to_transition_prediction"] = None
                r["label"] = "not assessable: the world-model effect has opposite signs on the two readouts"
            else:
                spec = yn["excess"] and qf["excess"]
                r["specific_to_transition_prediction"] = bool(spec)
                r["label"] = "specific by the rule" if spec else "no excess shown"
        elif yn is not None and ok(("wm", "first")) and not ok(("si", "narr")):
            # fallback: the narrator readout failed; the yes/no excess plus the reply readout at least as large in
            # standardized units, in the direction of the world-model effect, as the yes/no readout
            cells, plus, minus = _spec_cells(name, d)
            st = {ro: mean_contrast(S, ro, cells, plus, minus, k_max=k_max) for ro in (("wm", "first"), ("wm", "yn"))}
            sd = {ro: v.est / v.sd_item for ro, v in st.items()}
            s = yn["world_model_sign"]
            spec = yn["excess"] and s * sd[("wm", "first")] >= s * sd[("wm", "yn")]
            r["specific_to_transition_prediction"] = bool(spec)
            r["label"] = "excess on yes/no; question reading not excluded" if spec else "no excess shown"
        else:
            r["specific_to_transition_prediction"] = None
            r["label"] = "not assessable"
        out[name] = r
    return out


# ----------------------------------------------------------------------------- gates and regrets


def form_shares(checks: dict, space: str, fr: str, r: str) -> dict:
    c = (checks.get("samples") or {}).get(f"{space}|{fr}|{r}", {})
    na, nb = c.get("a", 0) + c.get("a_other", 0), c.get("b", 0) + c.get("b_other", 0)
    tot = na + nb + c.get("other", 0)
    return {"n_first": na, "n_second": nb, "n_total": tot,
            "success_form_share": c.get("a", 0) / na if na else None,
            "failure_form_share": c.get("b", 0) / nb if nb else None,
            "other_share": c.get("other", 0) / tot if tot else None}


def gates(S: Scores, slopes: dict, checks: dict) -> dict:
    """Competence and readout-validity gates per readout (section 5). The sampled-continuation checks are required
    for the question-free readouts; too few samples on either candidate word fail the gate."""
    if checks is None:
        raise ValueError("the readout gates need the smoke checks (sampled continuations)")
    out = {}
    for ro in READOUTS + ALF_READOUT_KEYS:
        if ro not in S.mass:
            continue
        a = slopes.get(ro)
        lo = a.ci()[0] if a is not None and math.isfinite(a.se) else float("nan")
        mass = float(np.median(S.mass[ro]))
        g = {"slope_lower": lo, "median_mass": mass}
        ok = lo > GATE_SLOPE_LOWER and mass >= GATE_MASS
        if ro in QUESTION_FREE:
            f = form_shares(checks, "alf" if ro[0] == "alf" else "toy", ro[0], ro[1])
            g.update(f)
            ok = ok and f["n_first"] >= GATE_MIN_SAMPLES and f["n_second"] >= GATE_MIN_SAMPLES \
                and f["success_form_share"] >= GATE_FORM and f["failure_form_share"] >= GATE_FORM \
                and f["other_share"] <= GATE_OTHER
        g["passed"] = bool(ok)
        out[ro] = g
    return out


def regrets(S: Scores, ro: tuple) -> dict:
    """Expected Brier and log-score (KL) regret against the exact posterior, normalized over the two candidates and
    unnormalized (all other probability counted as failure), by prompt family (descriptive)."""
    fam = lambda t: "cue_goal" if t.startswith("key") else "cue_tag" if t.startswith("tag") else \
        "note" if t.startswith("note") else t  # noqa: E731
    acc: dict = defaultdict(lambda: defaultdict(list))
    for (g, h, t), ks in S.L[ro].items():
        pb = D.exact_for("toy", t, g, h)
        hb = 0.0 if pb in (0.0, 1.0) else -(pb * math.log(pb) + (1 - pb) * math.log(1 - pb))
        for k, L in ks.items():
            p = 1 / (1 + math.exp(-L))
            acc[fam(t)]["brier"].append((p - pb) ** 2)
            acc[fam(t)]["kl"].append(pb * float(np.logaddexp(0, -L)) + (1 - pb) * float(np.logaddexp(0, L)) - hb)
            pa = math.exp(S.lpa[ro][(g, h, t)][k])
            lfail = math.log1p(-pa) if pa < 1 else -50.0
            acc[fam(t)]["brier_unnormalized"].append((pa - pb) ** 2)
            acc[fam(t)]["kl_unnormalized"].append(pb * (-math.log(max(pa, 1e-300))) + (1 - pb) * (-lfail) - hb)
    return {f: {m: float(np.mean(v)) for m, v in d.items()} for f, d in acc.items()}


def settled_probability_shifts(S: Scores, ro: tuple) -> dict:
    """In the settled history H7 the exact probability is 0 under every cue: the model's probability shift of each
    cue relative to the explore prompt (descriptive)."""
    out = {}
    for t in ("key1", "tag1", "note95", "coin", "coin_tag"):
        vals = []
        for g in W.GENERATOR_ORDER:
            a, b = S.get(ro, g, "H7", t), S.get(ro, g, "H7", "explore")
            vals += [1 / (1 + math.exp(-a[k])) - 1 / (1 + math.exp(-b[k])) for k in set(a) & set(b)]
        if vals:
            out[t] = float(np.mean(vals))
    return out


# ----------------------------------------------------------------------------- ALFWorld surface


def alf_estimands(S: Scores, ro: tuple, boot: int = BOOT, k_max: int | None = None) -> dict:
    """The ALFWorld-surface rendering (section 13): secondary, wordings treated as fixed."""
    cells = list(D.ALF_CELLS)
    a_cells = [(g, h) for g, h in cells if g == "A"]
    G = {"key1": 0.5, "key2": 0.5}
    T = {"tag1": 1 / 3, "tag2": 1 / 3, "tag3": 1 / 3}
    mc = lambda cs, p, m, adj=True: mean_contrast(S, ro, cs, p, m, space="alf", k_max=k_max, adjust=adj)  # noqa: E731
    out = {"evidence_slope": slope(S, ro, cells, {"explore": 1.0}, space="alf", boot=boot),
           "goal_minus_tag": mc(cells, G, T),
           "tag_naming_zero_information": mc(a_cells, {"tag1": 1.0}, {"tagO": 1.0}),
           "goal_zero_information": mc(a_cells, G, {"explore": 1.0}),
           "swap_zero_information": mc(a_cells, {"key1": 1.0}, {"coin": 1.0}),
           "goal_slope": slope(S, ro, cells, G, {"explore": 1.0}, space="alf", boot=boot)}
    imp = list(D.ALF_IMPLICIT_CELLS)
    out["implicit_swap"] = mc(imp, {"imp_key": 1.0}, {"imp_coin": 1.0}, False)
    out["implicit_goal_vs_explore"] = mc(imp, {"imp_key": 1.0}, {"imp_explore": 1.0}, False)
    return out


# ----------------------------------------------------------------------------- full analysis


def analyse(main_records: list[dict], d: D.Design, low_floor_family: bool, checks: dict, sizing: dict,
            xgpu_records: list[dict] | None = None, boot: int = BOOT) -> dict:
    """Everything the results note reports for one model."""
    check_sizing(sizing)
    S = load_records(main_records)
    out: dict = {"design": d.to_dict(), "sizing": sizing}
    slopes = {ro: slope(S, ro, GRADED_CELLS, {"explore": 1.0}, boot=boot) for ro in READOUTS if ro in S.mass}
    for ro in ALF_READOUT_KEYS:
        if ro in S.mass:
            slopes[ro] = slope(S, ro, list(D.ALF_CELLS), {"explore": 1.0}, space="alf", boot=boot)
    gt = gates(S, slopes, checks)
    out["gates"] = {"|".join(ro): v for ro, v in gt.items()}
    desc = checks.get("description") or {}
    exclude = () if desc.get("passed") else ("name_minus_description",)
    out["description_check"] = desc or "missing: name minus description not assessable"
    primary = PRIMARY if gt.get(PRIMARY, {}).get("passed") else ("wm", "yn") if gt.get(("wm", "yn"), {}).get("passed") \
        else None
    out["primary_readout"] = "|".join(primary) if primary else None
    per = {}
    for ro in READOUTS:
        if ro not in S.mass:
            continue
        E = estimands(S, ro, d, boot=boot)
        cls = floor_class(S.zmax[ro])
        a = slopes[ro].est
        entry = {"role": "primary" if ro == primary else "secondary", "gate": gt[ro], "floor_class": cls,
                 "target": 0.5 * a, "pattern_target": pattern_target(a),
                 "estimands": {k: v.to_dict() for k, v in E.items()},
                 "regrets": regrets(S, ro), "settled_probability_shifts": settled_probability_shifts(S, ro)}
        if primary is None:
            entry["readings"] = "no world-model readout passed its gates: nothing is read for this model"
        elif not gt[ro]["passed"]:
            entry["readings"] = "readout failed its gate"
        else:
            entry["readings"] = readings(E, cls, low_floor_family, 0.5 * a, pattern_target(a), sizing, exclude)
        per["|".join(ro)] = entry
    out["readouts"] = per
    out["specificity"] = specificity(S, d, slopes, gt) if primary else "not assessable"
    out["readout_disagreements"] = disagreements(per)
    alf = {}
    for ro in ALF_READOUT_KEYS:
        if ro in S.mass:
            alf["|".join(ro)] = {"gate": gt.get(ro), "floor_class": floor_class(S.zmax[ro]),
                                 "estimands": {k: v.to_dict() for k, v in alf_estimands(S, ro, boot).items()}}
    out["alfworld_surface"] = alf
    toy_std = {k: per.get("|".join(PRIMARY), {}).get("estimands", {}).get(k, {}).get("standardized")
               for k in ("goal_minus_tag", "tag_naming_zero_information", "goal_zero_information",
                         "swap_zero_information")}
    alf_std = {k: (alf.get("alf|first", {}).get("estimands", {}).get(k) or {}).get("standardized") for k in toy_std}
    out["signatures_toy_vs_alfworld_standardized"] = {"toy_reply": toy_std, "alfworld_reply": alf_std}
    if xgpu_records is not None:
        out["cross_gpu"] = cross_gpu(main_records, xgpu_records, d, low_floor_family, per, sizing, exclude, boot=200)
    return out


def disagreements(per: dict) -> list[dict]:
    """Family estimands whose detection or sign differs between readouts that passed their gates."""
    out = []
    ok = {k: v for k, v in per.items() if isinstance(v["readings"], dict)}
    for name in HOLM_FAMILY:
        st = {ro: (v["readings"]["holm"][name]["rejected"], math.copysign(1, v["readings"]["holm"][name]["estimate"]))
              for ro, v in ok.items() if name in v["readings"]["holm"]}
        if len({(r, s if r else 0) for r, s in st.values()}) > 1:
            out.append({"estimand": name, "by_readout": {k: {"detected": r, "sign": s} for k, (r, s) in st.items()}})
    return out


def cross_gpu(main_records: list[dict], xgpu_records: list[dict], d: D.Design, low_floor_family: bool, per: dict,
              sizing: dict, exclude: tuple = (), boot: int = 200) -> dict:
    """Rule 4: every decision statistic on the subset (items k < 9) in both environments, their shifts, and whether
    a decision changes or an estimate moves by more than a fifth of its main-run margin."""
    sub_main = [r for r in main_records if r["k"] < D.XGPU_K]
    keys_m = {(r["space"], r["gen"], r["hist"], r["k"]): sorted(r["scores"]) for r in sub_main}
    keys_x = {(r["space"], r["gen"], r["hist"], r["k"]): sorted(r["scores"]) for r in xgpu_records}
    if keys_m != keys_x:
        raise ValueError("the cross-GPU subset does not score the same items and prompts as the main run's subset")
    Sm, Sx = load_records(sub_main), load_records(xgpu_records)
    out = {}
    for ro in READOUTS:
        key = "|".join(ro)
        if key not in per or not isinstance(per[key]["readings"], dict):
            continue
        Em, Ex = estimands(Sm, ro, d, boot=boot), estimands(Sx, ro, d, boot=boot)
        cls, tgt, ptg = per[key]["floor_class"], per[key]["target"], per[key]["pattern_target"]
        rm = readings(Em, cls, low_floor_family, tgt, ptg, sizing, exclude)
        rx = readings(Ex, cls, low_floor_family, tgt, ptg, sizing, exclude)
        main_E = per[key]["estimands"]
        rows = {}
        for name in rm["holm"]:
            if name == "goal_minus_tag":
                margin = per[key]["readings"]["goal_minus_tag"]["equivalence"]["margin"]
            else:
                base = ptg if name in PATTERN_CONTRASTS else tgt
                margin = max(base, 5 * numerical_floor(cls, main_E[name]["n_eff"]))
            shift = Ex[name].est - Em[name].est
            changed = rm["holm"][name]["rejected"] != rx["holm"][name]["rejected"]
            rows[name] = {"main_subset": Em[name].est, "other_gpu": Ex[name].est, "shift": shift, "margin": margin,
                          "decision_changed": changed, "environment_dependent": bool(changed or abs(shift) > margin / 5)}
        rows["goal_minus_tag_equivalence_changed"] = (rm["goal_minus_tag"]["equivalence"]["both"] !=
                                                      rx["goal_minus_tag"]["equivalence"]["both"])
        rows["pattern_equivalence_changed"] = {k: rm["patterns_equivalence"][k]["equivalence"]["both"] !=
                                               rx["patterns_equivalence"][k]["equivalence"]["both"]
                                               for k in rm["patterns_equivalence"]}
        out[key] = rows
    return out


# ----------------------------------------------------------------------------- smoke readout and sizing rule

MENU_WORDINGS = ((2, 3), (3, 4), (4, 6), (6, 8), (8, 8))
MENU_N = (30, 45, 60)
MENU_PATTERN_N = (30, 45, 60, 90)
JOB_OVERHEAD_H = 0.3          # model loading and start-up of the full run's shards, the cross-GPU job and the checks
XGPU_SLOWDOWN = 1.3           # the cross-GPU subset runs on A40s, slower than L40S


def _panel_family_values(S: Scores, ro: tuple, family: str) -> np.ndarray:
    types = [f"{'key' if family == 'goal' else 'tag'}{i + 1}" for i in range(D.PANEL_WORDINGS)]
    blocks = []
    for g, h in D.PANEL_CELLS:
        ds = [S.get(ro, g, h, t) for t in types]
        ks = sorted(set.intersection(*[set(x) for x in ds]))
        blocks.append(np.array([[x[k] for k in ks] for x in ds]))
    n = min(b.shape[1] for b in blocks)
    return np.stack([b[:, :n] for b in blocks], axis=1)


def predicted_e1(vc_g: VarComp, vc_t: VarComp, s2_u: float, wg: int, wt: int, n: int, cells: int = 28) -> Est:
    v_items = (s2_u + vc_g.s2_e / wg + vc_t.s2_e / wt) / (n * cells)
    return add_wording(Est(est=0.0, se=math.sqrt(v_items), n_items=n * cells), [(vc_g, wg), (vc_t, wt)], cells)


def smoke_readout(records: list[dict], timing: dict, checks: dict, budget_gpu_h: float = 6.3,
                  fixed_gpu_h: float = 0.0, n_models: int = 2) -> dict:
    """Section 9: noise and variance measurements, the detection targets, the band and the design for the full run.
    No contrast between conditions is reported (only slopes, variances, masses and logit sizes)."""
    S = load_records(records)
    out: dict = {}
    slopes = {ro: slope(S, ro, GRADED_CELLS, {"explore": 1.0}) for ro in READOUTS if ro in S.mass}
    for ro in ALF_READOUT_KEYS:
        if ro in S.mass:
            slopes[ro] = slope(S, ro, list(D.ALF_CELLS), {"explore": 1.0}, space="alf")
    gt = gates(S, slopes, checks)
    out["evidence_slopes"] = {"|".join(ro): {"estimate": a.est, "se": a.se} for ro, a in slopes.items()}
    out["gates"] = {"|".join(ro): v for ro, v in gt.items()}
    out["logit_size"] = {"|".join(ro): {"median_top_logit": float(np.median(S.zmax[ro])),
                                        "floor_class": floor_class(S.zmax[ro])} for ro in S.zmax}
    out["description_check"] = checks.get("description")
    primary = PRIMARY if gt.get(PRIMARY, {}).get("passed") else ("wm", "yn") if gt.get(("wm", "yn"), {}).get("passed") \
        else None
    out["primary_readout"] = "|".join(primary) if primary else None
    sec = timing["seconds"] / max(timing["sequences"], 1)
    out["seconds_per_sequence"] = sec
    if primary is None:
        out["decision"] = {"band": "stop: neither world-model readout passed its gates; no full run for this model"}
        return out
    a = slopes[primary].est
    T, Tp = 0.5 * a, pattern_target(a)
    out["target"], out["pattern_target"] = T, Tp
    comps = {}
    for ro in READOUTS:
        if ro not in S.mass:
            continue
        vg, vt = wording_components(S, ro, "goal"), wording_components(S, ro, "tag")
        Xg, Xt = _panel_family_values(S, ro, "goal"), _panel_family_values(S, ro, "tag")
        n = min(Xg.shape[2], Xt.shape[2])
        dbar = Xg[:, :, :n].mean(axis=0) - Xt[:, :, :n].mean(axis=0)          # (C, n): all-wording contrast per item
        s2_d = float(np.mean([np.var(dbar[c], ddof=1) for c in range(dbar.shape[0])]))
        s2_u = max(0.0, s2_d - vg.s2_e / D.PANEL_WORDINGS - vt.s2_e / D.PANEL_WORDINGS)

        def sd_within(t1, t2=None):
            v = []
            for g, h in D.PANEL_CELLS:
                x1, x2 = S.get(ro, g, h, t1), (S.get(ro, g, h, t2) if t2 else None)
                ks = sorted(set(x1) & set(x2)) if x2 is not None else sorted(x1)
                x = np.array([x1[k] - (x2[k] if x2 is not None else 0) for k in ks])
                if len(x) > 1:
                    v.append(np.var(x, ddof=1))
            return float(math.sqrt(np.mean(v))) if v else float("nan")
        comps["|".join(ro)] = {
            "goal_wording": {"s2_alpha": vg.s2_alpha, "s2_gamma": vg.s2_gamma, "s2_e": vg.s2_e},
            "tag_wording": {"s2_alpha": vt.s2_alpha, "s2_gamma": vt.s2_gamma, "s2_e": vt.s2_e},
            "s2_item_contrast": s2_u,
            "sd_prompt_difference": {"goal_vs_explore": sd_within("key1", "explore"),
                                     "tag_vs_tagO": sd_within("tag1", "tagO"),
                                     "note95_vs_explore": sd_within("note95", "explore")},
            "sd_across_items": {"explore": sd_within("explore"), "key1": sd_within("key1"), "tag1": sd_within("tag1"),
                                "note95": sd_within("note95")},
        }
    out["variance"] = comps
    vg, vt = wording_components(S, primary, "goal"), wording_components(S, primary, "tag")
    s2_u = comps["|".join(primary)]["s2_item_contrast"]

    def s2_of(t1, t2, cells):
        v = []
        for g, h in cells:
            x1, x2 = S.get(primary, g, h, t1), S.get(primary, g, h, t2)
            x = np.array([x1[k] - x2[k] for k in sorted(set(x1) & set(x2))])
            if len(x) > 1:
                v.append(np.var(x, ddof=1))
        return float(np.mean(v))
    s2_note, s2_note0 = s2_of("note95", "explore", D.PANEL_CELLS), s2_of("note0", "explore", D.PANEL_CELLS)
    s2_ann, s2_p1 = s2_of("ann_key", "ann_coin", D.PROV_CELLS), s2_of("P1_key", "key1", D.PROV_CELLS)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    n_ann_c, n_ann_a = len([c for c in D.ANN_CELLS if c[0] != "A"]), len([c for c in D.ANN_CELLS if c[0] == "A"])
    mde_red = lambda n_h4, n: z * math.sqrt(s2_note / (len(H4_CELLS) * n_h4) + s2_note0 / (28 * n))  # noqa: E731
    mde_ann = lambda m: z * math.sqrt(s2_ann / (n_ann_c * m) + s2_ann / (n_ann_a * m))  # noqa: E731

    def cost_h(dd: D.Design) -> float:
        c = sum(D.count_sequences(D.full_units(dd)).values())
        cx = sum(D.count_sequences(D.full_units(dd, k_max=D.XGPU_K)).values())
        return (c + XGPU_SLOWDOWN * cx) * sec / 3600

    def projected(dd: D.Design) -> float:
        return n_models * (cost_h(dd) + timing["seconds"] / 3600 + JOB_OVERHEAD_H) + fixed_gpu_h

    def with_patterns(wg, wt, n) -> D.Design:
        n_h4 = next((m for m in MENU_PATTERN_N if m >= n and mde_red(m, n) <= Tp), max(MENU_PATTERN_N[-1], n))
        n_ann = next((m for m in MENU_PATTERN_N if m >= n and mde_ann(m) <= Tp), max(MENU_PATTERN_N[-1], n))
        dd = D.Design(w_goal=wg, w_tag=wt, n=n, n_h4=n_h4, n_ann=n_ann)
        while projected(dd) > budget_gpu_h and (dd.n_h4 > n or dd.n_ann > n):   # pattern items yield to the budget
            dd = D.Design(w_goal=wg, w_tag=wt, n=n, n_h4=max(n, _prev(dd.n_h4)), n_ann=max(n, _prev(dd.n_ann)))
        return dd

    base = D.Design()
    mde_base = predicted_e1(vg, vt, s2_u, base.w_goal, base.w_tag, base.n).mde()
    out["mde_goal_minus_tag_base"] = mde_base
    menu = []
    for wg, wt in MENU_WORDINGS:
        for n in MENU_N:
            dd = with_patterns(wg, wt, n)
            menu.append({"w_goal": wg, "w_tag": wt, "n": n, "mde": predicted_e1(vg, vt, s2_u, wg, wt, n).mde(),
                         "gpu_h_full_and_xgpu": cost_h(D.Design(w_goal=wg, w_tag=wt, n=n)),
                         "projected_total_gpu_h": projected(dd)})
    out["menu"] = menu
    base_cost = cost_h(base)
    negative_allowed = False
    if mde_base <= T:
        band, chosen, negative_allowed = "at or below target", (2, 3, 30), True
    elif mde_base <= 1.5 * T:
        ok = [m for m in menu if m["mde"] <= T and m["gpu_h_full_and_xgpu"] <= 2 * base_cost
              and m["projected_total_gpu_h"] <= budget_gpu_h]
        if ok:
            m = min(ok, key=lambda m: m["gpu_h_full_and_xgpu"])
            band, chosen, negative_allowed = "redesigned to reach target", (m["w_goal"], m["w_tag"], m["n"]), True
        else:
            band, chosen = ("above target after one redesign: goal minus tag not used for negative conclusions",
                            (2, 3, 30))
    else:
        band, chosen = "above 1.5 x target: goal minus tag not used for negative conclusions", (2, 3, 30)
    final = with_patterns(*chosen)
    pm = {"redundancy_pattern": mde_red(final.n_h4, final.n), "announced_slope": mde_ann(final.n_ann),
          "history_only_provenance": z * math.sqrt(s2_p1 / (len(D.PROV_CELLS) * final.n))}
    underpowered = [k for k in EQUIVALENCE_PATTERNS if pm[k] > Tp]
    out["pattern_mde"] = {**pm, "underpowered": underpowered}
    tot = projected(final)
    out["decision"] = {"band": band, "design": final.to_dict(),
                       "mde_goal_minus_tag": predicted_e1(vg, vt, s2_u, final.w_goal, final.w_tag, final.n).mde(),
                       "gpu_h_full_and_xgpu_per_model": cost_h(final), "projected_total_gpu_h": tot,
                       "budget_gpu_h": budget_gpu_h, "within_budget": tot <= budget_gpu_h,
                       "sizing": {"goal_minus_tag_negative_allowed": negative_allowed,
                                  "patterns_not_for_negative": underpowered}}
    if tot > budget_gpu_h:
        out["decision"]["band"] += "; STOP: projected total above budget, report before submitting"
    return out


def _prev(m: int) -> int:
    i = MENU_PATTERN_N.index(m) if m in MENU_PATTERN_N else len(MENU_PATTERN_N)
    return MENU_PATTERN_N[max(i - 1, 0)]
