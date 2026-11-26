"""The pre-specified analysis (notes/bayesenv_design.md, sections 7 to 12) and the smoke readout (section 9).

Scores: for a prompt, L = log p(first candidate) - log p(second candidate) at the readout position, the model's
log-odds of success normalized over the two candidates. Exact references come from `world`. Items are the statistical
unit; cells (generator x history) are strata with equal weight. Almost every estimand is linear in cell means,
sum_c mean_i y_ci, with variance sum_c var_i(y_ci) / n_c. Estimands that average over goal or tag wordings add the
wording variance estimated from the wording panel (eight wordings of each on four cells) by a two-way random-effects
decomposition, with Satterthwaite degrees of freedom. Slopes and ratios use a stratified item bootstrap.
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
SLOPE_HISTS = ("H0", "H2", "H3", "H6", "H9")    # graded histories in which a cue's Bayes effect is non-zero
BOOT = 2000
SEED = 20261125
ALPHA = 0.05
STD_MARGIN = 0.20             # equivalence margin on the standardized scale (>= 5 x 0.034)
PATTERN_TARGET = 0.5          # nats: the detection target for residual patterns
FLOOR_CLASSES = {             # numerical floor constants by readout logit size (notes/numerical_sensitivity.md 4.8)
    "small": (0.013, 0.20),   # top-candidate logits below 32 (Llama-sized)
    "large": (0.04, 0.58),    # 32 to 64 (Qwen3-8B-sized)
    "very_large": (0.08, 1.16),  # 64 or more: twice the large class
}
DESIGN_EFFECT = 1.3
GATE_SLOPE_LOWER = 0.1        # competence: the evidence slope's 95% lower bound must exceed this
GATE_MASS = 0.5               # median two-candidate mass
GATE_FORM = 0.8               # share of sampled replies starting with the first candidate that are the success form


# ----------------------------------------------------------------------------- loading


@dataclass
class Scores:
    """L[(space, framing, readout)][(gen, hist, ptype)] = {k: L}; plus mass and raw logits for gates."""
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
            ro = (fr, r)
            L = v["lp_a"] - v["lp_b"]
            s.L[ro][(rec["gen"], rec["hist"], t)][rec["k"]] = L
            s.lpa[ro][(rec["gen"], rec["hist"], t)][rec["k"]] = v["lp_a"]
            s.mass[ro].append(math.exp(v["lp_a"]) + math.exp(v["lp_b"]))
            s.zmax[ro].append(max(v["z_a"], v["z_b"]))
            if space == "alf" and fr != "alf":
                raise ValueError("ALFWorld records carry framing 'alf'")
    return s


def read_folders(folders) -> list[dict]:
    out = []
    for f in folders:
        p = Path(f) / "scoring.jsonl"
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


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
    parts: dict = field(default_factory=dict)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        q = stats.t.ppf(0.5 + level / 2, self.df) if math.isfinite(self.df) else stats.norm.ppf(0.5 + level / 2)
        return self.est - q * self.se, self.est + q * self.se

    def p(self) -> float:
        if self.se <= 0:
            return 0.0 if self.est != 0 else 1.0
        z = abs(self.est) / self.se
        return float(2 * (stats.t.sf(z, self.df) if math.isfinite(self.df) else stats.norm.sf(z)))

    def mde(self) -> float:
        df = self.df if math.isfinite(self.df) else 1e9
        return float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * self.se)

    def to_dict(self) -> dict:
        lo, hi = self.ci()
        lo90, hi90 = self.ci(0.90)
        std = self.est / self.sd_item if self.sd_item and self.sd_item > 0 else float("nan")
        return {"estimate": self.est, "se": self.se, "df": self.df, "ci95": [lo, hi], "ci90": [lo90, hi90],
                "p": self.p(), "mde": self.mde(), "n_items": self.n_items, "sd_item": self.sd_item,
                "standardized": std,
                "standardized_ci90": [lo90 / self.sd_item, hi90 / self.sd_item] if self.sd_item > 0 else None,
                **({"parts": self.parts} if self.parts else {})}


def item_contrast(S: Scores, ro: tuple, g: str, h: str, plus: dict, minus: dict, space: str = "toy",
                  k_max: int | None = None, adjust: bool = True, with_items: bool = False):
    """Per-item value of sum_t w_t L_t (plus) - sum_t w_t L_t (minus). With `adjust`, the same contrast of the exact
    logits is subtracted where every side's exact probability lies strictly inside (0, 1); a contrast in a settled
    cell is allowed only if its sides share the exact probability (its Bayes value is then exactly 0). Items need
    every type. With `with_items`, returns (item indices, values)."""
    types = list(plus) + list(minus)
    dicts = [S.get(ro, g, h, t) for t in types]
    ks = sorted(set.intersection(*[set(d) for d in dicts])) if dicts else []
    if k_max is not None:
        ks = [k for k in ks if k < k_max]
    if not ks:
        return (ks, np.array([])) if with_items else np.array([])
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


def linear(parts: dict) -> Est:
    """parts: cell -> per-item contributions (already weighted). Estimate sum_c mean, variance sum_c var/n."""
    est, var, n, pooled, dfp = 0.0, 0.0, 0, 0.0, 0
    for arr in parts.values():
        if len(arr) == 0:
            continue
        est += float(arr.mean())
        if len(arr) > 1:
            var += float(arr.var(ddof=1)) / len(arr)
        n += len(arr)
    return Est(est=est, se=math.sqrt(var), n_items=n)


def mean_contrast(S: Scores, ro: tuple, cells, plus: dict, minus: dict, space: str = "toy",
                  k_max: int | None = None, adjust: bool = True) -> Est:
    """Equal-weight mean over cells of the per-item contrast; also its pooled within-cell item SD."""
    cells = list(cells)
    parts = {}
    ss, dfs = 0.0, 0
    for g, h in cells:
        arr = item_contrast(S, ro, g, h, plus, minus, space, k_max, adjust=adjust)
        if len(arr):
            parts[(g, h)] = arr / len(cells)
            if len(arr) > 1:
                ss += float(((arr - arr.mean()) ** 2).sum())
                dfs += len(arr) - 1
    e = linear(parts)
    e.est = e.est * len(cells) / max(len(parts), 1)   # equal weights over the cells present
    e.se = e.se * len(cells) / max(len(parts), 1)
    e.sd_item = math.sqrt(ss / dfs) if dfs else float("nan")
    return e


def combine(a: Est, b: Est, sign: float = -1.0) -> Est:
    """a + sign * b for estimands on disjoint or independent item sets (variances add)."""
    return Est(est=a.est + sign * b.est, se=math.sqrt(a.se ** 2 + b.se ** 2), df=min(a.df, b.df),
               n_items=a.n_items + b.n_items, sd_item=float(np.nanmean([a.sd_item, b.sd_item])))


def linear_from(S: Scores, ro: tuple, terms: list[tuple[float, list, dict, dict]], space: str = "toy",
                k_max: int | None = None) -> Est:
    """sum over terms of weight x (equal-weight mean over that term's cells of the per-item contrast). Terms that
    meet in a cell must use the same items there; their per-item contributions are added, so the covariance of
    terms sharing items is kept."""
    parts: dict = {}
    for wt, cells, plus, minus in terms:
        for c in cells:
            ks, arr = item_contrast(S, ro, c[0], c[1], plus, minus, space, k_max, with_items=True)
            if not len(arr):
                continue
            arr = arr * wt / len(cells)
            if c in parts:
                if parts[c][0] != ks:
                    raise ValueError(f"terms meeting in cell {c} use different items")
                parts[c] = (ks, parts[c][1] + arr)
            else:
                parts[c] = (ks, arr)
    return linear({c: v for c, (_, v) in parts.items()})


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


def wording_components(S: Scores, ro: tuple, family: str, cells=D.PANEL_CELLS, transform=None) -> VarComp | None:
    """Two-way random-effects decomposition of one cue family's wordings on the panel cells.
    family "goal": key1..key8; "tag": tag1..tag8. With `transform`, values are transform(ro, g, h, t) dicts."""
    W_ = D.PANEL_WORDINGS
    types = [f"{'key' if family == 'goal' else 'tag'}{i + 1}" for i in range(W_)]
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


def add_wording(e: Est, sides: list[tuple[VarComp | None, int]], n_cells: int) -> Est:
    """Add the sampling variance of the wordings used: for each side (components, number of wordings averaged),
    (s2_alpha + s2_gamma / cells) / W from the panel. Satterthwaite df (the item term has n_items - 1 df)."""
    v_items = e.se ** 2
    terms = [vc.wording_variance(w, n_cells) for vc, w in sides if vc is not None and w > 0]
    v = v_items + sum(t for t, _ in terms)
    den = (v_items ** 2 / max(e.n_items - 1, 1)) + sum(t ** 2 / df for t, df in terms if t > 0)
    df = v ** 2 / den if den > 0 else math.inf
    return Est(est=e.est, se=math.sqrt(v), df=df, n_items=e.n_items, sd_item=e.sd_item,
               parts={"var_items": v_items, "var_wordings": [t for t, _ in terms]})


# ----------------------------------------------------------------------------- slopes (bootstrap)


def _cell_items(S: Scores, ro: tuple, cells, ytypes: dict, xminus: dict | None, space: str) -> dict:
    """cell -> (items, y values, x value) for slope regressions; y = sum w L(types) - [xminus types], x = Bayes."""
    out = {}
    for g, h in cells:
        types = list(ytypes) + list(xminus or {})
        ks = sorted(set.intersection(*[set(S.get(ro, g, h, t)) for t in types]))
        if not ks:
            continue
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
    if not data:
        return Est(float("nan"), float("nan"))
    est = _ols_slope(data)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(boot):
        idx = {c: rng.integers(0, len(y), len(y)) for c, (y, _) in data.items()}
        bs.append(_ols_slope(data, idx))
    return Est(est=est, se=float(np.nanstd(bs, ddof=1)), n_items=sum(len(y) for y, _ in data.values()),
               parts={"boot": None})


def ratio_of_means(S: Scores, ro: tuple, cells, plus: dict, minus: dict, space: str = "toy",
                   boot: int = BOOT) -> Est:
    """mean model contrast / mean exact contrast over cells (exact contrasts finite)."""
    data = _cell_items(S, ro, cells, plus, minus, space)
    num = np.mean([y.mean() for y, _ in data.values()])
    den = np.mean([x for _, x in data.values()])
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(boot):
        bs.append(np.mean([y[rng.integers(0, len(y), len(y))].mean() for y, _ in data.values()]) / den)
    return Est(est=float(num / den), se=float(np.std(bs, ddof=1)), n_items=sum(len(y) for y, _ in data.values()))


# ----------------------------------------------------------------------------- the estimands


GRADED_CELLS = [(g, h) for g in W.GENERATOR_ORDER for h in W.GRADED]
SLOPE_CELLS = [(g, h) for g in W.GENERATOR_ORDER for h in SLOPE_HISTS]


def goal_side(w: int) -> dict:
    return {f"key{i + 1}": 1.0 / w for i in range(w)}


def tag_side(w: int) -> dict:
    return {f"tag{i + 1}": 1.0 / w for i in range(w)}


def evidence_slopes(S: Scores, ro: tuple, boot: int = BOOT) -> dict:
    """Overall evidence slope (explore prompts, graded cells), history slope (generators with base rate 0.5) and
    base-rate slope (generator D against the three generators with base rate 0.5, by history)."""
    a = slope(S, ro, GRADED_CELLS, {"explore": 1.0}, boot=boot)
    hist = slope(S, ro, [(g, h) for g in ("A", "R3", "C") for h in W.GRADED], {"explore": 1.0}, boot=boot)
    # base-rate slope: ratio of mean differences (D minus the base-rate-0.5 generators), stratified bootstrap
    rng = np.random.default_rng(SEED)
    pairs = []
    for h in W.GRADED:
        d_vals = np.array(list(S.get(ro, "D", h, "explore").values()))
        o_vals = np.concatenate([np.array(list(S.get(ro, g, h, "explore").values())) for g in ("A", "R3", "C")])
        dx = bayes_logit("toy", "explore", "D", h) - bayes_logit("toy", "explore", "A", h)
        if len(d_vals) and len(o_vals):
            pairs.append((d_vals, o_vals, dx))
    if pairs:
        num = np.mean([d.mean() - o.mean() for d, o, _ in pairs])
        den = np.mean([dx for _, _, dx in pairs])
        bs = [np.mean([d[rng.integers(0, len(d), len(d))].mean() - o[rng.integers(0, len(o), len(o))].mean()
                       for d, o, _ in pairs]) / den for _ in range(boot)]
        base = Est(est=float(num / den), se=float(np.std(bs, ddof=1)), n_items=sum(len(d) + len(o) for d, o, _ in pairs))
    else:
        base = Est(float("nan"), float("nan"))
    return {"evidence_slope": a, "history_slope": hist, "base_rate_slope": base}


def estimands(S: Scores, ro: tuple, d: D.Design, k_max: int | None = None, boot: int = BOOT,
              with_wording: bool = True) -> dict[str, Est]:
    """Every pre-specified estimand for one framing x readout (toy). Names follow section 7 of the design note."""
    cells_all = list(D.CELLS)
    out: dict[str, Est] = {}
    vc_g = wording_components(S, ro, "goal") if with_wording else None
    vc_t = wording_components(S, ro, "tag") if with_wording else None
    G, T = goal_side(d.w_goal), tag_side(d.w_tag)
    mc = lambda cells, plus, minus: mean_contrast(S, ro, cells, plus, minus, k_max=k_max)  # noqa: E731
    e1 = mc(cells_all, G, T)
    out["goal_minus_tag"] = add_wording(e1, [(vc_g, d.w_goal), (vc_t, d.w_tag)], len(cells_all))
    e2 = mc(cells_all, {**G, "tagO": 1.0}, {**T, "explore": 1.0})
    out["goal_minus_tag_four_prompt"] = add_wording(e2, [(vc_g, d.w_goal), (vc_t, d.w_tag)], len(cells_all))
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
    out["zero_reliability_note"] = mc(cells_all, {"note0": 1.0}, {"explore": 1.0})
    h4 = [(g, h) for g, h in cells_all if h == "H4"]
    red = {"note50": 1 / 3, "note80": 1 / 3, "note95": 1 / 3}
    out["redundancy_residual"] = mc(h4, red, {"explore": 1.0})
    out["redundancy_pattern"] = linear_from(S, ro, [(1.0, h4, red, {"explore": 1.0}),
                                                    (-1.0, cells_all, {"note0": 1.0}, {"explore": 1.0})],
                                            k_max=k_max)
    out["redundancy_pattern"].sd_item = out["redundancy_residual"].sd_item
    ann = list(D.ANN_CELLS)
    out["announced_goal_effect"] = mc(ann, {"ann_key": 1.0}, {"ann_coin": 1.0})
    ann_inf = [(g, h) for g, h in ann if g in ("C", "D")]
    ann_a = [(g, h) for g, h in ann if g == "A"]
    out["announced_slope"] = combine(mc(ann_inf, {"ann_key": 1.0}, {"ann_coin": 1.0}),
                                     mc(ann_a, {"ann_key": 1.0}, {"ann_coin": 1.0}))
    out["history_only_provenance"] = mc(list(D.PROV_CELLS), {"P1_key": 1.0}, {"key1": 1.0})
    # slopes and ratios
    out.update(evidence_slopes(S, ro, boot=boot))
    out["goal_slope"] = slope(S, ro, SLOPE_CELLS, G, {"explore": 1.0}, boot=boot)
    out["tag_slope"] = slope(S, ro, SLOPE_CELLS, T, {"tagO": 1.0}, boot=boot)
    notes = {}
    for rho in ("note50", "note80", "note95"):
        notes[rho] = _cell_items(S, ro, SLOPE_CELLS, {rho: 1.0}, {"explore": 1.0}, "toy")
    pooled = {(c, r): v for r, cs in notes.items() for c, v in cs.items()}
    if pooled:
        est = _ols_slope(pooled)
        rng = np.random.default_rng(SEED)
        # bootstrap items within cells, the same draw for every reliability of a cell
        bs = []
        for _ in range(boot):
            idx_c = {c: rng.integers(0, len(v[0]), len(v[0])) for c, v in notes["note95"].items()}
            bs.append(_ols_slope(pooled, {(c, r): idx_c[c] for (c, r) in pooled}))
        out["note_slope"] = Est(est=est, se=float(np.nanstd(bs, ddof=1)))
    out["note_ordering"] = mean_contrast(S, ro, SLOPE_CELLS, {"note95": 1.0}, {"note50": 1.0}, k_max=k_max,
                                         adjust=False)
    out["note_ordering"].parts = {"bayes_value": float(np.mean([bayes_logit("toy", "note95", g, h) -
                                                                 bayes_logit("toy", "note50", g, h)
                                                                 for g, h in SLOPE_CELLS]))}
    out["seeing_guide_ratio"] = ratio_of_means(S, ro, list(D.PROV_CELLS), {"P2_key": 1.0}, {"key1": 1.0}, boot=boot)
    # matched-information goal against note (descriptive): C against the 0.8 note, D against the 0.95 note
    gm = [mc([(g, h) for h in SLOPE_HISTS], G, {n: 1.0}) for g, n in (("C", "note80"), ("D", "note95"))]
    out["goal_minus_matched_note"] = add_wording(
        Est(est=(gm[0].est + gm[1].est) / 2, se=math.sqrt(gm[0].se ** 2 + gm[1].se ** 2) / 2,
            n_items=gm[0].n_items + gm[1].n_items, sd_item=float(np.nanmean([gm[0].sd_item, gm[1].sd_item]))),
        [(vc_g, d.w_goal)], 2 * len(SLOPE_HISTS))
    # goal minus tag by generator and its dependence on the likelihood ratio (diagnostics)
    by_gen = {}
    for g in W.GENERATOR_ORDER:
        cg = [(gg, h) for gg, h in cells_all if gg == g]
        by_gen[g] = mc(cg, G, T)
        out[f"goal_minus_tag_gen_{g}"] = by_gen[g]
    xs = np.array([math.log(W.likelihood_ratio(g)) for g in W.GENERATOR_ORDER])
    ys = np.array([by_gen[g].est for g in W.GENERATOR_ORDER])
    ws = np.array([1 / max(by_gen[g].se, 1e-9) ** 2 for g in W.GENERATOR_ORDER])
    xb = (ws * xs).sum() / ws.sum()
    b = (ws * (xs - xb) * (ys - (ws * ys).sum() / ws.sum())).sum() / (ws * (xs - xb) ** 2).sum()
    out["goal_minus_tag_lr_slope"] = Est(est=float(b), se=float(math.sqrt(1 / (ws * (xs - xb) ** 2).sum())))
    by_hist = [mc([(g, h) for g in W.GENERATOR_ORDER], G, T) for h in W.HISTORY_ORDER]
    ests = np.array([e.est for e in by_hist])
    wts = np.array([1 / max(e.se, 1e-9) ** 2 for e in by_hist])
    pooled_m = (wts * ests).sum() / wts.sum()
    q = float((wts * (ests - pooled_m) ** 2).sum())
    out["goal_minus_tag_history_heterogeneity"] = Est(est=q, se=float("nan"),
                                                      parts={"Q": q, "df": len(by_hist) - 1,
                                                             "p": float(stats.chi2.sf(q, len(by_hist) - 1)),
                                                             "by_history": dict(zip(W.HISTORY_ORDER,
                                                                                    ests.tolist()))})
    return out


HOLM_FAMILY = ("goal_minus_tag", "tag_naming_zero_information", "competing_goal", "name_minus_description",
               "zero_reliability_note", "redundancy_residual", "redundancy_pattern", "announced_goal_effect",
               "announced_slope", "history_only_provenance")
PATTERN_CONTRASTS = ("redundancy_pattern", "announced_slope", "history_only_provenance")
CONSTANTS = ("tag_naming_zero_information", "zero_reliability_note", "announced_goal_effect")


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


def numerical_floor(cls: str, n_items: int) -> float:
    tau, sig = FLOOR_CLASSES[cls]
    return math.sqrt(tau ** 2 + DESIGN_EFFECT * sig ** 2 / max(n_items, 1))


def readings(E: dict[str, Est], cls: str, low_floor_family: bool, target: float) -> dict:
    """Decisions for one readout: Holm over the family, the pattern rule, and equivalence where allowed."""
    ps = {k: E[k].p() for k in HOLM_FAMILY if k in E}
    rej = holm(ps)
    out = {"holm": {k: {"p": ps[k], "rejected": rej[k], "estimate": E[k].est, "ci95": list(E[k].ci())}
                    for k in ps}}
    pattern = [k for k in PATTERN_CONTRASTS if rej.get(k) and (k == "history_only_provenance" or E[k].est > 0)]
    constants = [k for k in CONSTANTS if rej.get(k) and E[k].est > 0]
    out["residual_pattern_detected"] = bool(pattern)
    out["pattern_contrasts_detected"] = pattern
    out["content_blind_constants_detected"] = constants
    gt = E["goal_minus_tag"]
    floor = numerical_floor(cls, gt.n_items)
    margin = max(target, 5 * floor)
    lo90, hi90 = gt.ci(0.90)
    out["goal_minus_tag"] = {
        "detected": bool(rej.get("goal_minus_tag")), "sign": float(np.sign(gt.est)), "target": target,
        "at_or_above_target": bool(rej.get("goal_minus_tag")) and abs(gt.est) >= target,
        "mde": gt.mde(), "numerical_floor": floor, "margin": margin,
        "equivalent_within_margin": bool(low_floor_family and -margin < lo90 and hi90 < margin),
        "standardized_ci90": gt.to_dict()["standardized_ci90"],
        "equivalent_standardized": bool(low_floor_family and gt.sd_item > 0 and
                                        -STD_MARGIN < lo90 / gt.sd_item and hi90 / gt.sd_item < STD_MARGIN),
    }
    pat = {}
    for k in ("redundancy_pattern", "announced_slope"):
        e = E[k]
        fl = numerical_floor(cls, e.n_items)
        m = max(PATTERN_TARGET, 5 * fl)
        lo, hi = e.ci(0.90)
        pat[k] = {"mde": e.mde(), "margin": m, "equivalent_within_margin": bool(low_floor_family and -m < lo and hi < m)}
    out["patterns_equivalence"] = pat
    out["no_pattern_shown"] = bool(low_floor_family and not pattern and
                                   all(v["equivalent_within_margin"] for v in pat.values()))
    return out


def normalized_difference(S: Scores, d: D.Design, name: str, ro_a: tuple, ro_b: tuple, slopes: dict,
                          k_max: int | None = None) -> Est | None:
    """(estimand in ro_a / slope_a) - (estimand in ro_b / slope_b) on the same items, for the specificity rule.
    Wording variance from the panel applied to the transformed per-wording values; slope uncertainty by the delta
    method."""
    a_a, a_b = slopes[ro_a], slopes[ro_b]
    if not (a_a.est > 0 and a_b.est > 0):
        return None
    G, T = goal_side(d.w_goal), tag_side(d.w_tag)
    spec = {"goal_minus_tag": (list(D.CELLS), G, T),
            "zero_reliability_note": (list(D.CELLS), {"note0": 1.0}, {"explore": 1.0}),
            "redundancy_residual": ([(g, h) for g, h in D.CELLS if h == "H4"],
                                    {"note50": 1 / 3, "note80": 1 / 3, "note95": 1 / 3}, {"explore": 1.0})}[name]
    cells, plus, minus = spec
    parts = {}
    ests = []
    for ro, a in ((ro_a, a_a), (ro_b, a_b)):
        ests.append(mean_contrast(S, ro, cells, plus, minus, k_max=k_max).est)
    for g, h in cells:
        xa = item_contrast(S, ro_a, g, h, plus, minus, k_max=k_max)
        xb = item_contrast(S, ro_b, g, h, plus, minus, k_max=k_max)
        if len(xa) and len(xa) == len(xb):
            parts[(g, h)] = (xa / a_a.est - xb / a_b.est) / len(cells)
    e = linear(parts)
    if name == "goal_minus_tag":
        def tf(g, h, t):
            da, db = S.get(ro_a, g, h, t), S.get(ro_b, g, h, t)
            return {k: da[k] / a_a.est - db[k] / a_b.est for k in set(da) & set(db)}
        vc_g = wording_components(S, None, "goal", transform=tf)
        vc_t = wording_components(S, None, "tag", transform=tf)
        e = add_wording(e, [(vc_g, d.w_goal), (vc_t, d.w_tag)], len(cells))
    v_slope = (ests[0] / a_a.est) ** 2 * (a_a.se / a_a.est) ** 2 + (ests[1] / a_b.est) ** 2 * (a_b.se / a_b.est) ** 2
    return Est(est=e.est, se=math.sqrt(e.se ** 2 + v_slope), df=e.df, n_items=e.n_items)


def specificity(S: Scores, d: D.Design, slopes: dict, gates: dict, k_max: int | None = None) -> dict:
    """The specificity rule (section 10): an estimand counts as specific to predicting the action's outcome only if
    its slope-normalized value is larger in the world-model framing than in state inference on the shared yes/no
    readout AND on each framing's own question-free readout (reply first token against the narrator)."""
    out = {}
    for name in ("goal_minus_tag", "zero_reliability_note", "redundancy_residual"):
        r = {}
        yn = normalized_difference(S, d, name, ("wm", "yn"), ("si", "yn"), slopes, k_max) \
            if gates[("wm", "yn")]["passed"] and gates[("si", "yn")]["passed"] else None
        qf = normalized_difference(S, d, name, ("wm", "first"), ("si", "narr"), slopes, k_max) \
            if gates[("wm", "first")]["passed"] and gates[("si", "narr")]["passed"] else None
        r["yes_no"] = yn.to_dict() if yn else "not assessable (a readout failed its gate)"
        r["question_free"] = qf.to_dict() if qf else "not assessable (a readout failed its gate)"
        yn_pos = bool(yn and yn.ci()[0] > 0)
        if qf is not None:
            r["specific_to_transition_prediction"] = bool(yn_pos and qf.ci()[0] > 0)
            r["label"] = "specific by the rule" if r["specific_to_transition_prediction"] else "no excess shown"
        elif gates[("wm", "first")]["passed"] and yn is not None:
            # fallback: the narrator readout failed; the yes/no excess plus the reply readout at least as large in
            # standardized units as the yes/no readout
            st = {ro: mean_contrast(S, ro, *_spec_cells(name, d), k_max=k_max) for ro in (("wm", "first"), ("wm", "yn"))}
            sd = {ro: v.est / v.sd_item for ro, v in st.items()}
            ok = yn_pos and sd[("wm", "first")] >= sd[("wm", "yn")]
            r["specific_to_transition_prediction"] = bool(ok)
            r["label"] = "excess on yes/no; question reading not excluded" if ok else "no excess shown"
        else:
            r["specific_to_transition_prediction"] = None
            r["label"] = "not assessable"
        out[name] = r
    return out


def _spec_cells(name: str, d: D.Design):
    G, T = goal_side(d.w_goal), tag_side(d.w_tag)
    return {"goal_minus_tag": (list(D.CELLS), G, T),
            "zero_reliability_note": (list(D.CELLS), {"note0": 1.0}, {"explore": 1.0}),
            "redundancy_residual": ([(g, h) for g, h in D.CELLS if h == "H4"],
                                    {"note50": 1 / 3, "note80": 1 / 3, "note95": 1 / 3}, {"explore": 1.0})}[name]


# ----------------------------------------------------------------------------- gates and regrets


def gates(S: Scores, slopes: dict, checks: dict | None) -> dict:
    """Competence and readout-validity gates per readout (section 5)."""
    out = {}
    for ro in READOUTS + ALF_READOUT_KEYS:
        if ro not in S.mass:
            continue
        a = slopes.get(ro)
        lo = a.ci()[0] if a is not None and math.isfinite(a.se) else float("nan")
        mass = float(np.median(S.mass[ro]))
        form = None
        if checks and ro[1] in ("first", "narr"):
            space = "alf" if ro[0] == "alf" else "toy"
            c = (checks.get("samples") or {}).get(f"{space}|{ro[0]}|{ro[1]}", {})
            n_first = c.get("a", 0) + c.get("a_other", 0)
            form = c.get("a", 0) / n_first if n_first else None
        ok = (lo > GATE_SLOPE_LOWER) and mass >= GATE_MASS and (form is None or form >= GATE_FORM)
        out[ro] = {"slope_lower": lo, "median_mass": mass, "success_form_share": form, "passed": bool(ok)}
    return out


def regrets(S: Scores, ro: tuple) -> dict:
    """Expected Brier and log-score (KL) regret against the exact posterior, normalized over the two candidates
    and unnormalized (all other probability counted as failure), by prompt family (descriptive)."""
    fam = lambda t: "cue_goal" if t.startswith("key") else "cue_tag" if t.startswith("tag") else \
        "note" if t.startswith("note") else t  # noqa: E731
    acc: dict = defaultdict(lambda: defaultdict(list))
    for (g, h, t), ks in S.L[ro].items():
        pb = D.exact_for("toy", t, g, h)
        hb = 0.0 if pb in (0.0, 1.0) else -(pb * math.log(pb) + (1 - pb) * math.log(1 - pb))
        for k, L in ks.items():
            p = 1 / (1 + math.exp(-L))
            acc[fam(t)]["brier"].append((p - pb) ** 2)
            sp_neg = math.log1p(math.exp(-L)) if L > -30 else -L
            sp_pos = math.log1p(math.exp(L)) if L < 30 else L
            acc[fam(t)]["kl"].append(pb * sp_neg + (1 - pb) * sp_pos - hb)
            lpa = S.lpa[ro][(g, h, t)][k]
            pa = math.exp(lpa)
            lfail = math.log1p(-pa) if pa < 1 else -50.0
            acc[fam(t)]["kl_unnormalized"].append(pb * (-lpa) + (1 - pb) * (-lfail) - hb)
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


def alf_estimands(S: Scores, ro: tuple, boot: int = BOOT) -> dict:
    """The ALFWorld-surface rendering (section 13): secondary, wordings treated as fixed."""
    cells = list(D.ALF_CELLS)
    a_cells = [(g, h) for g, h in cells if g == "A"]
    G = {"key1": 0.5, "key2": 0.5}
    T = {"tag1": 1 / 3, "tag2": 1 / 3, "tag3": 1 / 3}
    mc = lambda cs, p, m: mean_contrast(S, ro, cs, p, m, space="alf")  # noqa: E731
    out = {"evidence_slope": slope(S, ro, cells, {"explore": 1.0}, space="alf", boot=boot),
           "goal_minus_tag": mc(cells, G, T),
           "tag_naming_zero_information": mc(a_cells, {"tag1": 1.0}, {"tagO": 1.0}),
           "goal_zero_information": mc(a_cells, G, {"explore": 1.0}),
           "swap_zero_information": mc(a_cells, {"key1": 1.0}, {"coin": 1.0}),
           "goal_slope": slope(S, ro, cells, G, {"explore": 1.0}, space="alf", boot=boot)}
    imp = list(D.ALF_IMPLICIT_CELLS)
    for name, plus, minus in (("implicit_swap", {"imp_key": 1.0}, {"imp_coin": 1.0}),
                              ("implicit_goal_vs_explore", {"imp_key": 1.0}, {"imp_explore": 1.0})):
        parts = {}
        for g, h in imp:
            dk = [S.get(ro, g, h, t) for t in list(plus) + list(minus)]
            ks = sorted(set.intersection(*[set(x) for x in dk]))
            arr = np.array([sum(w * S.get(ro, g, h, t)[k] for t, w in plus.items()) -
                            sum(w * S.get(ro, g, h, t)[k] for t, w in minus.items()) for k in ks])
            if len(arr):
                parts[(g, h)] = arr / len(imp)
        e = linear(parts)
        allv = np.concatenate([v * len(imp) for v in parts.values()]) if parts else np.array([])
        e.sd_item = float(allv.std(ddof=1)) if len(allv) > 1 else float("nan")
        out[name] = e
    return out


# ----------------------------------------------------------------------------- full analysis


def analyse(main_records: list[dict], d: D.Design, low_floor_family: bool, checks: dict | None = None,
            xgpu_records: list[dict] | None = None, boot: int = BOOT) -> dict:
    """Everything the results note reports for one model."""
    S = load_records(main_records)
    out: dict = {"design": d.to_dict()}
    slopes = {ro: evidence_slopes(S, ro, boot=boot)["evidence_slope"] for ro in READOUTS if ro in S.mass}
    for ro in ALF_READOUT_KEYS:
        if ro in S.mass:
            slopes[ro] = slope(S, ro, list(D.ALF_CELLS), {"explore": 1.0}, space="alf", boot=boot)
    gt = gates(S, slopes, checks)
    out["gates"] = {f"{a}|{b}": v for (a, b), v in gt.items()}
    primary = PRIMARY if gt.get(PRIMARY, {}).get("passed") else ("wm", "yn") if gt.get(("wm", "yn"), {}).get("passed") \
        else None
    out["primary_readout"] = "|".join(primary) if primary else None
    per = {}
    for ro in READOUTS:
        if ro not in S.mass:
            continue
        E = estimands(S, ro, d, boot=boot)
        cls = floor_class(S.zmax[ro])
        a = slopes[ro]
        target = 0.5 * a.est if a.est == a.est else float("nan")
        per["|".join(ro)] = {
            "role": "primary" if ro == primary else "secondary",
            "gate": gt[ro], "floor_class": cls, "target": target,
            "estimands": {k: v.to_dict() for k, v in E.items()},
            "readings": readings(E, cls, low_floor_family, target) if gt[ro]["passed"] else "readout failed its gate",
            "regrets": regrets(S, ro), "settled_probability_shifts": settled_probability_shifts(S, ro),
        }
    out["readouts"] = per
    out["specificity"] = specificity(S, d, slopes, gt)
    out["readout_disagreements"] = disagreements(per)
    alf = {}
    for ro in ALF_READOUT_KEYS:
        if ro in S.mass:
            alf["|".join(ro)] = {"gate": gt.get(ro), "estimands": {k: v.to_dict() for k, v in alf_estimands(S, ro, boot).items()}}
    out["alfworld_surface"] = alf
    if xgpu_records is not None:
        out["cross_gpu"] = cross_gpu(main_records, xgpu_records, d, low_floor_family, per, boot=200)
    return out


def disagreements(per: dict) -> list[dict]:
    """Holm-family estimands whose detection or sign differs between readouts that passed their gates."""
    out = []
    ok = {k: v for k, v in per.items() if isinstance(v["readings"], dict)}
    for name in HOLM_FAMILY:
        st = {}
        for ro, v in ok.items():
            h = v["readings"]["holm"].get(name)
            if h:
                st[ro] = (h["rejected"], math.copysign(1, h["estimate"]))
        if len({(r, s if r else 0) for r, s in st.values()}) > 1:
            out.append({"estimand": name, "by_readout": {k: {"detected": r, "sign": s} for k, (r, s) in st.items()}})
    return out


def cross_gpu(main_records: list[dict], xgpu_records: list[dict], d: D.Design, low_floor_family: bool, per: dict,
              boot: int = 200) -> dict:
    """Rule 4: the decision statistics on the subset (items k < 9) in both environments, their shifts, and whether
    a decision changes or an estimate moves by more than a fifth of its margin."""
    Sm = load_records([r for r in main_records if r["k"] < D.XGPU_K])
    Sx = load_records(xgpu_records)
    out = {}
    for ro in READOUTS:
        key = "|".join(ro)
        if key not in per or not isinstance(per[key]["readings"], dict):
            continue
        Em = estimands(Sm, ro, d, boot=boot)
        Ex = estimands(Sx, ro, d, boot=boot)
        target = per[key]["target"]
        cls = per[key]["floor_class"]
        rm, rx = readings(Em, cls, low_floor_family, target), readings(Ex, cls, low_floor_family, target)
        rows = {}
        for name in HOLM_FAMILY:
            margin = per[key]["readings"]["goal_minus_tag"]["margin"] if name == "goal_minus_tag" else \
                max(PATTERN_TARGET if name in PATTERN_CONTRASTS else target, 5 * numerical_floor(cls, Em[name].n_items))
            shift = Ex[name].est - Em[name].est
            changed = rm["holm"][name]["rejected"] != rx["holm"][name]["rejected"]
            rows[name] = {"main_subset": Em[name].est, "other_gpu": Ex[name].est, "shift": shift, "margin": margin,
                          "decision_changed": changed,
                          "environment_dependent": bool(changed or abs(shift) > margin / 5)}
        rows["goal_minus_tag_equivalence_changed"] = (rm["goal_minus_tag"]["equivalent_within_margin"] !=
                                                      rx["goal_minus_tag"]["equivalent_within_margin"])
        out[key] = rows
    return out


# ----------------------------------------------------------------------------- smoke readout and sizing rule

MENU_WORDINGS = ((2, 3), (3, 4), (4, 6), (6, 8), (8, 8))
MENU_N = (30, 45, 60)
MENU_PATTERN_N = (30, 45, 60, 90)
JOB_OVERHEAD_H = 0.3          # model loading and start-up of the full run's shards, the cross-GPU job and the checks


def _panel_family_values(S: Scores, ro: tuple, family: str) -> np.ndarray | None:
    types = [f"{'key' if family == 'goal' else 'tag'}{i + 1}" for i in range(D.PANEL_WORDINGS)]
    blocks = []
    for g, h in D.PANEL_CELLS:
        ds = [S.get(ro, g, h, t) for t in types]
        ks = sorted(set.intersection(*[set(x) for x in ds]))
        blocks.append(np.array([[x[k] for k in ks] for x in ds]))
    n = min(b.shape[1] for b in blocks)
    return np.stack([b[:, :n] for b in blocks], axis=1) if n >= 2 else None


def predicted_e1(vc_g: VarComp, vc_t: VarComp, s2_u: float, wg: int, wt: int, n: int, cells: int = 28) -> Est:
    v_items = (s2_u + vc_g.s2_e / wg + vc_t.s2_e / wt) / (n * cells)
    e = Est(est=0.0, se=math.sqrt(v_items), n_items=n * cells)
    return add_wording(e, [(vc_g, wg), (vc_t, wt)], cells)


def smoke_readout(records: list[dict], timing: dict, checks: dict | None, budget_gpu_h: float = 6.3,
                  fixed_gpu_h: float = 0.0, n_models: int = 2) -> dict:
    """Section 9: noise and variance measurements, the detection target, the band and the design for the full run.
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
    out["description_check"] = (checks or {}).get("description")
    primary = PRIMARY if gt.get(PRIMARY, {}).get("passed") else ("wm", "yn") if gt.get(("wm", "yn"), {}).get("passed") \
        else None
    out["primary_readout"] = "|".join(primary) if primary else None
    seq = timing["sequences"]
    sec = timing["seconds"] / max(seq, 1)
    out["seconds_per_sequence"] = sec
    if primary is None:
        out["decision"] = {"band": "stop", "reason": "neither world-model readout passed its gates"}
        return out
    T = 0.5 * slopes[primary].est
    out["target"] = T
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
        # prompt-difference noise and arm-specific variance (within cell, across items)
        def sd_within(t1, t2=None):
            v = []
            for g, h in D.PANEL_CELLS:
                a, b = S.get(ro, g, h, t1), (S.get(ro, g, h, t2) if t2 else None)
                ks = sorted(set(a) & set(b)) if b is not None else sorted(a)
                x = np.array([a[k] - (b[k] if b is not None else 0) for k in ks])
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
    # sequence counts and cost of each candidate design
    def cost_h(d: D.Design) -> float:
        c = D.count_sequences(D.full_units(d))
        cx = D.count_sequences(D.full_units(d, k_max=D.XGPU_K))
        return (sum(c.values()) + sum(cx.values())) * sec / 3600
    base = D.Design()
    mde_base = predicted_e1(vg, vt, s2_u, base.w_goal, base.w_tag, base.n).mde()
    out["mde_goal_minus_tag_base"] = mde_base
    menu = []
    for wg, wt in MENU_WORDINGS:
        for n in MENU_N:
            dd = D.Design(w_goal=wg, w_tag=wt, n=n)
            menu.append({"w_goal": wg, "w_tag": wt, "n": n, "mde": predicted_e1(vg, vt, s2_u, wg, wt, n).mde(),
                         "gpu_h": cost_h(dd)})
    out["menu"] = menu
    base_cost = cost_h(base)
    if mde_base <= T:
        band, chosen = "at or below target", (base.w_goal, base.w_tag, base.n)
    elif mde_base <= 1.5 * T:
        ok = [m for m in menu if m["mde"] <= T and m["gpu_h"] <= 2 * base_cost]
        if ok:
            m = min(ok, key=lambda m: m["gpu_h"])
            band, chosen = "redesigned to reach target", (m["w_goal"], m["w_tag"], m["n"])
        else:
            band, chosen = "above target after one redesign: goal minus tag not used for negative conclusions", \
                (base.w_goal, base.w_tag, base.n)
    else:
        band, chosen = "above 1.5 x target: goal minus tag not used for negative conclusions", \
            (base.w_goal, base.w_tag, base.n)
    # pattern contrasts: items in the existence-settled and announced cells
    def s2_of(t1, t2, cells):
        v = []
        for g, h in cells:
            a, b = S.get(primary, g, h, t1), S.get(primary, g, h, t2)
            x = np.array([a[k] - b[k] for k in sorted(set(a) & set(b))])
            if len(x) > 1:
                v.append(np.var(x, ddof=1))
        return float(np.mean(v)) if v else float("nan")
    s2_note = s2_of("note95", "explore", D.PANEL_CELLS)
    s2_note0 = s2_of("note0", "explore", D.PANEL_CELLS)
    s2_ann = s2_of("ann_key", "ann_coin", D.PROV_CELLS)
    s2_p1 = s2_of("P1_key", "key1", D.PROV_CELLS)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    mde_red = lambda n_h4, n: z * math.sqrt(s2_note / (4 * n_h4) + s2_note0 / (28 * n))  # noqa: E731
    mde_ann = lambda n_ann: z * math.sqrt(s2_ann / (12 * n_ann) + s2_ann / (6 * n_ann))  # noqa: E731
    n_h4 = next((m for m in MENU_PATTERN_N if mde_red(m, chosen[2]) <= PATTERN_TARGET), MENU_PATTERN_N[-1])
    n_ann = next((m for m in MENU_PATTERN_N if mde_ann(m) <= PATTERN_TARGET), MENU_PATTERN_N[-1])
    final = D.Design(w_goal=chosen[0], w_tag=chosen[1], n=chosen[2], n_h4=max(n_h4, chosen[2]),
                     n_ann=max(n_ann, chosen[2]))
    per_model = cost_h(final) + timing["seconds"] / 3600 + JOB_OVERHEAD_H
    out["pattern_mde"] = {"redundancy_pattern": mde_red(final.n_h4, final.n), "announced_slope": mde_ann(final.n_ann),
                          "history_only_provenance": z * math.sqrt(s2_p1 / (6 * final.n)),
                          "underpowered": [k for k, v in (("redundancy_pattern", mde_red(final.n_h4, final.n)),
                                                          ("announced_slope", mde_ann(final.n_ann)))
                                           if v > PATTERN_TARGET]}
    projected = n_models * per_model + fixed_gpu_h
    out["decision"] = {"band": band, "design": final.to_dict(), "mde_goal_minus_tag":
                       predicted_e1(vg, vt, s2_u, final.w_goal, final.w_tag, final.n).mde(),
                       "gpu_h_full_and_xgpu_per_model": cost_h(final), "gpu_h_per_model_with_smoke": per_model,
                       "projected_total_gpu_h": projected, "budget_gpu_h": budget_gpu_h,
                       "within_budget": projected <= budget_gpu_h}
    if projected > budget_gpu_h:
        out["decision"]["band"] += "; STOP: projected total above budget, report before submitting"
    return out
