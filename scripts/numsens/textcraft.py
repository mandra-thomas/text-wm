"""TextCraft round 2 records scored in two environments, and the round-2 decisions re-run on both.

Old: textcraft_r2_score_<model> (torch 2.6 / transformers 4.51.3, L40S, 4 jobs per model).
New: textcraft_r3_rescore_<model> (serve2: torch 2.14.1 / transformers 5.23.0, L40S, 2 jobs per model).
Rerun: textcraft_r3_envcheck_<model> (serve2, L40S, one job, 300 seeded records) against the serve2 rescoring.

For Qwen3-8B, Mistral-7B and Gemma-2-9B (the three round-2 models that were rescored) this script
  * describes per-score and per-contrast differences,
  * recomputes the five confirmatory contrasts (lift Λ, parent penalty P, naming Dn, completion Dc, timing T) exactly as
    wf/textcraft/r2_analysis.py at tag textcraft-r2-design-frozen-2026-11-23 defines them (populations from
    wf/textcraft/data/r2_items.jsonl at that tag), with the margin min(0.3 SD, 0.4 nats), 95% pair-bootstrap intervals,
    sign-flip / bootstrap differs p and TOST p, using the SAME resamples for both environments (common random numbers),
    so differences between the two columns are due to the scores only;
  * applies Holm over the 50-test family, taking the two models that were not rescored (Llama-3.1-8B, Llama-3.2-3B) from
    notes/textcraft_r2_metrics.json, and reads the verdicts in both environments;
  * computes round 3's B3 standardized contrasts z for each old model in both environments (how far a stack change
    alone moves z, against B3's 0.20 margin).
Output: out/textcraft.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import DATA, describe, dump, independence

REPO = Path(__file__).resolve().parents[4]
MODELS = ("qwen3-8b", "mistral-7b-v0.3", "gemma2-9b-it")
B = 20000
SEED = 20261125
MATCHED = ("too_few", "never_obtained")
FAMILY = ("lift", "parent_penalty", "naming", "completion", "timing")


def load_items():
    return {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "r2_items.jsonl")}


def load(glob_):
    by = {}
    for f in sorted(DATA.glob(glob_)):
        for line in open(f):
            r = json.loads(line)
            by[(r["transition_id"], r["condition"])] = r["L"]
    return by


def populations(items):
    recs = list(items.values())
    mf = [r for r in recs if r["kind"] == "main" and r["population"] == "main"]
    goal = [r for r in mf if r["role"] == "goal" and r["failure_class"] != "used_up"]
    ing = [r for r in mf if r["role"] == "ingredient"]
    ing_m = [r for r in ing if r["failure_class"] in MATCHED]
    padded = [r for r in mf if "pad8_aligned" in r["arms"]]
    return {"main_failures": mf, "goal": goal, "ingredient": ing,
            "roles": [r for r in goal + ing_m if r["both_task_line"]],
            "padded": [r for r in padded if (r["role"] == "goal" and r["failure_class"] != "used_up")
                       or (r["role"] == "ingredient" and r["failure_class"] in MATCHED)],
            "parent": [r for r in recs if r["kind"] == "parent" and r["population"] == "main"]}


def rows(by, P):
    L = lambda i, a: by.get((i, a), np.nan)
    d8 = lambda i, k: L(i, f"pad{k}_aligned") - L(i, f"pad{k}_swapped")
    spec = {
        "lift": ("within", P["ingredient"], lambda i: L(i, "aligned") - L(i, "swapped")),
        "parent_penalty": ("within", P["parent"], lambda i: L(i, "ext_aligned") - L(i, "ext_parent")),
        "naming": ("between", P["roles"], lambda i: L(i, "both") - L(i, "swapped")),
        "completion": ("between", P["roles"], lambda i: L(i, "aligned") - L(i, "both")),
        "timing": ("between", P["padded"], lambda i: d8(i, 8) - d8(i, 1)),
        "role_gap": ("between", P["roles"], lambda i: L(i, "aligned") - L(i, "swapped")),
        # the frozen applicability gate of T: the goal-minus-ingredient effect at one look-up must have a 95% interval
        # entirely below zero (r2_analysis "timing_role_gap_at_one")
        "timing_gate_role_gap_at_one": ("between", P["padded"], lambda i: d8(i, 1)),
    }
    out = {}
    for name, (kind, recs, fn) in spec.items():
        v = [(r["id"], r["role"], r["pair_id"], fn(r["id"])) for r in recs]
        out[name] = (kind, [x for x in v if x[3] == x[3]])
    return out


class Resampler:
    """Fixed pair resamples and sign flips per contrast, shared by both environments."""

    def __init__(self, pairs):
        self.keys = sorted(set(pairs))
        rng = np.random.default_rng(SEED)
        self.idx = rng.integers(0, len(self.keys), size=(B, len(self.keys)))
        self.flip = rng.choice([-1.0, 1.0], size=(B, len(self.keys)))

    def agg(self, data, role=None):
        s = defaultdict(float)
        n = defaultdict(int)
        for _id, r, p, x in data:
            if role is None or r == role:
                s[p] += x
                n[p] += 1
        return np.array([s[k] for k in self.keys]), np.array([n[k] for k in self.keys], float)


def stats(kind, data, rs: Resampler, margin):
    vals = np.array([x for *_r, x in data])
    if kind == "within":
        s, n = rs.agg(data)
        draws = s[rs.idx].sum(1) / n[rs.idx].sum(1)
        est = vals.mean()
        obs = abs(s.sum()) / n.sum()
        flips = np.abs((rs.flip * s).sum(1)) / n.sum()
        p_diff = (np.sum(flips >= obs - 1e-12) + 1) / (B + 1)
        sd_z = vals.std(ddof=1)
        z = est / sd_z
    else:
        sg, ng = rs.agg(data, "goal")
        si, ni = rs.agg(data, "ingredient")
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = sg[rs.idx].sum(1) / ng[rs.idx].sum(1) - si[rs.idx].sum(1) / ni[rs.idx].sum(1)
        draws = draws[~np.isnan(draws)]
        g = np.array([x for _i, r, _p, x in data if r == "goal"])
        i = np.array([x for _i, r, _p, x in data if r == "ingredient"])
        est = g.mean() - i.mean()
        p_diff = min(1.0, (2 * min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (len(draws) + 1))
        pooled = np.sqrt(((len(g) - 1) * g.var(ddof=1) + (len(i) - 1) * i.var(ddof=1)) / (len(g) + len(i) - 2))
        z = est / pooled
    srt = np.sort(draws)
    q = lambda a: float(srt[min(len(srt) - 1, int(a * len(srt)))])
    low = (np.sum(draws <= -margin) + 1) / (len(draws) + 1)
    high = (np.sum(draws >= margin) + 1) / (len(draws) + 1)
    return {"estimate": float(est), "ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)],
            "p_differs": float(p_diff), "p_equivalent": float(max(low, high)), "margin": float(margin),
            "bootstrap_se": float(draws.std(ddof=1)), "standardized_z": float(z)}


def _frozen_block(frozen, m, name):
    c = frozen["per_model"][m]["contrasts"]
    if name in c:
        return c[name]
    if name == "timing_gate_role_gap_at_one":
        return c.get("timing", {}).get("role_gap_at_one_lookup") or {}
    return {}


def holm(ps: dict):
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for k, (key, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[key] = run
    return out


def verdict(c, hd, he):
    if hd < 0.05 and c["estimate"] > 0 and c["ci95"][0] > 0:
        return "confirmed above zero"
    if hd < 0.05 and c["estimate"] < 0 and c["ci95"][1] < 0:
        return "confirmed below zero"
    if he < 0.05:
        return "equivalent to zero"
    return "inconclusive"


def main():
    items = load_items()
    P = populations(items)
    frozen = json.loads((REPO / "notes/textcraft_r2_metrics.json").read_text())
    res = {"per_model": {}}
    pvals = {"old": {}, "serve2": {}}
    for m in MODELS:
        env = {"old": load(f"text-wm-fix42/runs/textcraft_r2_score_{m}/*/scoring.jsonl"),
               "serve2": load(f"text-wm-fix47/runs/textcraft_r3_rescore_{m}/*/scoring.jsonl"),
               "envcheck": load(f"text-wm-fix47/runs/textcraft_r3_envcheck_{m}/*/scoring.jsonl")}
        M = {}
        # ---- noise description
        keys = sorted(env["old"].keys() & env["serve2"].keys())
        d = np.array([env["serve2"][k] - env["old"][k] for k in keys])
        yn = np.array([k[1].startswith("yn_") for k in keys])
        M["stack_per_score_two_candidate"] = describe(d[~yn])
        M["stack_per_score_yes_no"] = describe(d[yn])
        rk = sorted(env["envcheck"].keys() & env["serve2"].keys())
        M["rerun_serve2_envcheck_vs_rescore"] = describe(np.array([env["serve2"][k] - env["envcheck"][k] for k in rk]))
        # per-item aligned - swapped on main failures, both readouts
        for a1, a2 in (("aligned", "swapped"), ("yn_aligned", "yn_swapped"), ("aligned", "neutral")):
            ids = [r["id"] for r in P["main_failures"] if (r["id"], a1) in env["old"] and (r["id"], a2) in env["old"]]
            eo = np.array([env["old"][(i, a1)] - env["old"][(i, a2)] for i in ids])
            es = np.array([env["serve2"][(i, a1)] - env["serve2"][(i, a2)] for i in ids])
            d1 = np.array([env["serve2"][(i, a1)] - env["old"][(i, a1)] for i in ids])
            d2 = np.array([env["serve2"][(i, a2)] - env["old"][(i, a2)] for i in ids])
            M[f"stack_contrast_{a1}-{a2}"] = {"n": len(ids), "effect_old": float(eo.mean()), "effect_serve2": float(es.mean()),
                                              "per_item": describe(es - eo), "item_r": float(np.corrcoef(eo, es)[0, 1]),
                                              "independence_of_arms": independence(d1, d2),
                                              "sign_flips": int(((np.sign(eo) != np.sign(es)) & (eo != 0) & (es != 0)).sum())}
        # ---- the confirmatory contrasts in both environments, common resamples
        margins = {}
        for e in ("old", "serve2"):
            ids = [r["id"] for r in P["main_failures"]]
            dd = np.array([env[e].get((i, "aligned"), np.nan) - env[e].get((i, "swapped"), np.nan) for i in ids])
            dd = dd[~np.isnan(dd)]
            margins[e] = {"sd": float(dd.std(ddof=1)), "margin": float(min(0.3 * dd.std(ddof=1), 0.4))}
        M["margin"] = margins
        R_old, R_new = rows(env["old"], P), rows(env["serve2"], P)
        C = {}
        for name in R_old:
            kind, dold = R_old[name]
            _, dnew = R_new[name]
            rs = Resampler([p for _i, _r, p, _x in dold])
            new_by = {i: x for i, _r, _p, x in dnew}
            ddiff = [(i, r, p, new_by[i] - x) for i, r, p, x in dold if i in new_by]
            C[name] = {"old": stats(kind, dold, rs, margins["old"]["margin"]),
                       "numerical_shift": stats(kind, ddiff, rs, margins["old"]["margin"]),
                       "serve2_same_margin": stats(kind, dnew, rs, margins["old"]["margin"]),
                       "serve2_own_margin": stats(kind, dnew, rs, margins["serve2"]["margin"]),
                       "frozen": {k: _frozen_block(frozen, m, name).get(k) for k in ("estimate", "ci95", "p_differs", "p_equivalent")}}
            if name in FAMILY:
                pvals["old"][(m, name, "differs")] = C[name]["old"]["p_differs"]
                pvals["old"][(m, name, "equivalent")] = C[name]["old"]["p_equivalent"]
                pvals["serve2"][(m, name, "differs")] = C[name]["serve2_same_margin"]["p_differs"]
                pvals["serve2"][(m, name, "equivalent")] = C[name]["serve2_same_margin"]["p_equivalent"]
        M["contrasts"] = C
        res["per_model"][m] = M
    # the two models that were not rescored: frozen p-values in both columns
    for m in ("llama3.1-8b", "llama3.2-3b"):
        for name in FAMILY:
            c = frozen["per_model"][m]["contrasts"][name]
            for e in ("old", "serve2"):
                pvals[e][(m, name, "differs")] = c["p_differs"]
                pvals[e][(m, name, "equivalent")] = c["p_equivalent"]
    H = {e: holm(pvals[e]) for e in pvals}
    V = {}
    for m in MODELS:
        for name in FAMILY:
            c = res["per_model"][m]["contrasts"][name]
            vo = verdict(c["old"], H["old"][(m, name, "differs")], H["old"][(m, name, "equivalent")])
            vn = verdict(c["serve2_same_margin"], H["serve2"][(m, name, "differs")], H["serve2"][(m, name, "equivalent")])
            fz = frozen["verdicts"][m]["contrasts"][name]
            V[f"{m}/{name}"] = {"frozen": fz, "recomputed_old": vo, "serve2": vn,
                                "holm_equiv_old": H["old"][(m, name, "equivalent")], "holm_equiv_serve2": H["serve2"][(m, name, "equivalent")],
                                "holm_differs_old": H["old"][(m, name, "differs")], "holm_differs_serve2": H["serve2"][(m, name, "differs")]}
    res["verdicts"] = V
    dump("textcraft", res)
    # ---- print
    for m, M in res["per_model"].items():
        print("==", m, "margin", M["margin"])
        for k in ("stack_per_score_two_candidate", "stack_per_score_yes_no", "rerun_serve2_envcheck_vs_rescore"):
            p = M[k]
            print(f"  {k}: n {p['n']} ident {p['share_identical']:.3f} mean|d| {p['mean_abs']:.3f} sd {p['sd']:.3f} "
                  f"p99 {p['p99_abs']:.3f} max {p['max_abs']:.3f} kurt {p['excess_kurtosis']}")
        for k, v in M.items():
            if k.startswith("stack_contrast"):
                print(f"  {k}: n {v['n']} eff {v['effect_old']:.3f}->{v['effect_serve2']:.3f} sd {v['per_item']['sd']:.3f} "
                      f"r {v['item_r']:.4f} flips {v['sign_flips']} ratio {v['independence_of_arms']['ratio_var_contrast_to_sum']:.2f}")
        for name, c in M["contrasts"].items():
            o, n_ = c["old"], c["serve2_same_margin"]
            print(f"  {name}: frozen {c['frozen']['estimate'] if c['frozen']['estimate'] is not None else float('nan'):.4f} old {o['estimate']:.4f} [{o['ci95'][0]:.3f},{o['ci95'][1]:.3f}] "
                  f"pe {o['p_equivalent']:.4f} pd {o['p_differs']:.4f} z {o['standardized_z']:.3f} | serve2 {n_['estimate']:.4f} "
                  f"[{n_['ci95'][0]:.3f},{n_['ci95'][1]:.3f}] pe {n_['p_equivalent']:.4f} pd {n_['p_differs']:.4f} z {n_['standardized_z']:.3f} "
                  f"| dz {n_['standardized_z'] - o['standardized_z']:+.4f} dest {n_['estimate'] - o['estimate']:+.4f} "
                  f"(noise se {c['numerical_shift']['bootstrap_se']:.4f})")
    for k, v in V.items():
        print(k, v)


if __name__ == "__main__":
    main()
