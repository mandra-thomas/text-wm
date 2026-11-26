"""Lexsem diagnostic: analyses across arms (notes/lexsem_design.md, section 10): numerical floors, direct
between-model and between-format contrasts on the same items, the model x think-block x sentence-meaning
interaction, and the action-type breakdown. Every contrast is paired on the items the arms share and uses one
game-clustered bootstrap; every share is reported with the absolute effects it divides.
"""
from __future__ import annotations

import math

import numpy as np

from wf import lexsem
from wf import lexsem_rules as R
from wf.lexsem_analysis import Boot, interval, mean, value

MEANINGS = {"assertion": "name_assert", "reference without assertion": "name_ignore", "denial": "name_deny",
            "listed word": "name_use"}
SENTENCES = ("name_use", "name_ignore", "name_deny", "name_elsewhere", "desc_assert")
# Pre-specified direct comparisons: (arm A, arm B, what differs)
PAIRS = (
    ("qwen_plain", "llama_chat", "model (Qwen3-8B plain against Llama-3.1-8B)"),
    ("qwen_block", "llama_chat", "model (Qwen3-8B with the block against Llama-3.1-8B)"),
    ("qwen_plain", "qwen_block", "format (Qwen3-8B)"),
    ("qwen4b_plain", "qwen4b_block", "format (Qwen3-4B)"),
    ("qwen14b_plain", "qwen14b_block", "format (Qwen3-14B)"),
    ("qwen4b_plain", "qwen_plain", "size (Qwen3-4B against 8B, plain)"),
    ("qwen14b_plain", "qwen_plain", "size (Qwen3-14B against 8B, plain)"),
    ("qwen4b_block", "qwen_block", "size (Qwen3-4B against 8B, block)"),
    ("qwen14b_block", "qwen_block", "size (Qwen3-14B against 8B, block)"),
    ("llama3b_chat", "llama_chat", "size (Llama-3.2-3B against 3.1-8B)"),
    ("qwen4b_plain", "llama3b_chat", "model at small size (Qwen3-4B plain against Llama-3.2-3B)"),
)
QWEN_SIZES = (("qwen4b", "Qwen3-4B"), ("qwen", "Qwen3-8B"), ("qwen14b", "Qwen3-14B"))
TAKE = "take"
BAND_HIGH = 32.0  # readout logits at or above this sit on a 0.25-nat bf16 grid (numerical_sensitivity.md section 4.4)


def _by_id(recs: list[dict]) -> dict:
    return {r["id"]: r for r in recs}


def _common(arms: dict[str, dict], names: tuple[str, ...], st: str, ro: str, conds: tuple[str, ...]) -> list[str]:
    sets = []
    for n in names:
        sets.append({i for i, r in arms[n].items() if r["stratum"] == st
                     and all(value(r, c, ro) is not None for c in conds)})
    return sorted(set.intersection(*sets)) if sets else []


def _net(r: dict, c: str, ro: str) -> float:
    return value(r, c, ro) - value(r, lexsem.CONTROL_OF[c], ro)


def _goal(r: dict, ro: str) -> float:
    return value(r, "aligned", ro) - value(r, "swapped", ro)


def _share_draws(boot: Boot, gs: list[str], num: np.ndarray, den: np.ndarray) -> tuple[float, np.ndarray]:
    n, nd = boot.sums(gs, num)
    d, dd = boot.sums(gs, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (n / d if d else math.nan), nd / dd


def standardized(boot: Boot, gs: list[str], x: np.ndarray) -> dict:
    """Mean over item SD (a scale-free effect), from per-game sums so each draw recomputes it."""
    s1, d1 = boot.sums(gs, x)
    s2, d2 = boot.sums(gs, x * x)
    k, dk = boot.sums(gs, np.ones_like(x))

    def z(a, b, n):
        with np.errstate(divide="ignore", invalid="ignore"):
            m = a / n
            return m / np.sqrt(np.maximum(b / n - m * m, 0) * n / np.maximum(n - 1, 1))
    return interval(float(z(np.array(s1), np.array(s2), np.array(k))), z(d1, d2, dk))


def three_way(diff: dict, what: str) -> str:
    return R.difference_reading(*diff["ci"], f"{what}: lower", f"{what}: the same within 0.25", f"{what}: higher")


# ---------------- numerical floors ----------------

def measured_floor(main: list[dict], alt: list[dict], ro: str, B: int, seed: int) -> dict | None:
    """The shift of the mean paired effect (goal effect, name-assertion net effect) between the main environment and
    the floor rescoring (A40), on the floor subset: floor = sqrt(mean shift^2 + its bootstrap SD^2), the larger of
    the two contrasts. A subset of 100 items makes this conservative for means over 150 to 300 items."""
    m, a = _by_id(main), _by_id(alt)
    ids = [i for i in a if i in m and all(value(x, c, ro) is not None for x in (m[i], a[i])
                                          for c in lexsem.GROUPS["floor"])]
    if len(ids) < 20:
        return None
    gs = [m[i]["game_id"] for i in ids]
    boot = Boot(gs, B, seed)
    out = {}
    for name, f in (("goal effect", _goal), ("name assertion net", lambda r, ro_: _net(r, "name_assert", ro_))):
        d = np.array([f(a[i], ro) - f(m[i], ro) for i in ids])
        iv, dr = mean(boot, gs, d)
        out[name] = {"mean_shift": iv, "floor": float(math.sqrt(iv["estimate"] ** 2 + np.nanstd(dr) ** 2))}
    out["floor"] = max(v["floor"] for v in out.values())
    out["n_items"] = len(ids)
    return out


def band_floor(recs: list[dict], floors: dict) -> float | None:
    """Fallback when no floor rescoring exists: the measured floor of the logit band the readout logits sit in."""
    z = [max(abs(s[f]["z_a"]), abs(s[f]["z_b"])) for r in recs for s in r["scores"].values() for f in ("obs", "yn")
         if f in s and "z_a" in s[f]]
    if not z:
        return None
    return floors["block"] if float(np.median(z)) >= BAND_HIGH else floors["chat"]


# ---------------- direct comparisons ----------------

def between(arms: dict[str, dict], B: int, seed: int) -> dict:
    """For each pre-specified pair and each sentence, on the same items: the difference in its share of the goal
    effect, in its relative lift (over the name assertion) and in its standardized net effect; plus the goal
    effects themselves (standardized), so no ratio stands alone."""
    out = {}
    for a, b, what in PAIRS:
        if a not in arms or b not in arms:
            continue
        for ro in ("yn", "first"):
            for st in ("later", "step0"):
                base = ("aligned", "swapped", "name_assert", "ctrl_assert")
                entry = {}
                ids0 = _common(arms, (a, b), st, ro, base)
                if not ids0:
                    continue
                for c in SENTENCES:
                    ids = _common(arms, (a, b), st, ro, base + (c, lexsem.CONTROL_OF[c]))
                    if len(ids) < 20:
                        continue
                    gs = [arms[a][i]["game_id"] for i in ids]
                    boot = Boot(gs, B, seed)
                    res, sh, rl = {}, {}, {}
                    for n in (a, b):
                        rr = [arms[n][i] for i in ids]
                        num = np.array([_net(r, c, ro) for r in rr])
                        g = np.array([_goal(r, ro) for r in rr])
                        ref = np.array([_net(r, "name_assert", ro) for r in rr])
                        sp, sd = _share_draws(boot, gs, num, g)
                        rp, rd = _share_draws(boot, gs, num, ref)
                        sh[n], rl[n] = (sp, sd), (rp, rd)
                        res[n] = {"net_nats": mean(boot, gs, num)[0], "net_standardized": standardized(boot, gs, num),
                                  "goal_effect_nats": mean(boot, gs, g)[0],
                                  "goal_standardized": standardized(boot, gs, g),
                                  "share_of_goal_effect": interval(sp, sd), "relative_lift": interval(rp, rd)}
                    d_share = interval(sh[a][0] - sh[b][0], sh[a][1] - sh[b][1])
                    d_rel = interval(rl[a][0] - rl[b][0], rl[a][1] - rl[b][1])
                    d_std = interval(res[a]["net_standardized"]["estimate"] - res[b]["net_standardized"]["estimate"],
                                     np.array([np.nan]))
                    entry[c] = {"arms": res, "share_difference": {**d_share, "reading": three_way(d_share, what)},
                                "relative_lift_difference": {**d_rel, "reading": three_way(d_rel, what)},
                                "standardized_difference_point": d_std["estimate"], "n_items": len(ids)}
                if entry:
                    out[f"{a} vs {b}|{ro}|{st}"] = {"what": what, "sentences": entry}
    return out


def interaction(arms: dict[str, dict], B: int, seed: int) -> dict:
    """Model x think-block x sentence meaning, as differences of differences of shares of the goal effect, on the
    same items: for each Qwen size, the format effect on each meaning relative to the assertion,

        [s_block(m) - s_plain(m)] - [s_block(assertion) - s_plain(assertion)],

    then the same difference between sizes; and against Llama-3.1-8B in each Qwen format,

        [s_qwen(m) - s_qwen(assertion)] - [s_llama(m) - s_llama(assertion)]."""
    out = {}
    meanings = [m for m in MEANINGS if m != "assertion"]
    for ro in ("yn", "first"):
        for st in ("later", "step0"):
            conds = ("aligned", "swapped") + tuple(x for c in MEANINGS.values() for x in (c, lexsem.CONTROL_OF[c]))
            present = [n for n in arms if any(True for _ in arms[n])]
            fmt_pairs = [(f"{k}_plain", f"{k}_block", lab) for k, lab in QWEN_SIZES
                         if f"{k}_plain" in arms and f"{k}_block" in arms]
            names = sorted({x for p in fmt_pairs for x in p[:2]} | ({"llama_chat"} & set(present)))
            if len(names) < 2:
                continue
            ids = _common(arms, tuple(names), st, ro, conds)
            if len(ids) < 20:
                continue
            gs = [arms[names[0]][i]["game_id"] for i in ids]
            boot = Boot(gs, B, seed)
            share = {}
            for n in names:
                rr = [arms[n][i] for i in ids]
                g = np.array([_goal(r, ro) for r in rr])
                for m, c in MEANINGS.items():
                    share[(n, m)] = _share_draws(boot, gs, np.array([_net(r, c, ro) for r in rr]), g)
            res: dict = {"n_items": len(ids), "shares": {f"{n}|{m}": interval(*share[(n, m)]) for n, m in share}}
            fmt = {}
            for p, b, lab in fmt_pairs:
                for m in meanings:
                    pt = (share[(b, m)][0] - share[(p, m)][0]) - (share[(b, "assertion")][0] - share[(p, "assertion")][0])
                    dr = (share[(b, m)][1] - share[(p, m)][1]) - (share[(b, "assertion")][1] - share[(p, "assertion")][1])
                    fmt[(lab, m)] = (pt, dr)
                    res.setdefault("format_x_meaning", {})[f"{lab}|{m}"] = {
                        **interval(pt, dr), "reading": three_way(interval(pt, dr), f"{lab}, block minus plain, {m} "
                                                                                    f"relative to assertion")}
            labs = [lab for _, _, lab in fmt_pairs]
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    for m in meanings:
                        a_, b_ = fmt[(labs[j], m)], fmt[(labs[i], m)]
                        iv = interval(a_[0] - b_[0], a_[1] - b_[1])
                        res.setdefault("size_x_format_x_meaning", {})[f"{labs[j]} minus {labs[i]}|{m}"] = {
                            **iv, "reading": three_way(iv, f"{labs[j]} minus {labs[i]}")}
            if "llama_chat" in names:
                for n in names:
                    if n == "llama_chat":
                        continue
                    for m in meanings:
                        pt = (share[(n, m)][0] - share[(n, "assertion")][0]) - (
                            share[("llama_chat", m)][0] - share[("llama_chat", "assertion")][0])
                        dr = (share[(n, m)][1] - share[(n, "assertion")][1]) - (
                            share[("llama_chat", m)][1] - share[("llama_chat", "assertion")][1])
                        res.setdefault("model_x_meaning", {})[f"{n} vs llama_chat|{m}"] = {
                            **interval(pt, dr), "reading": three_way(interval(pt, dr), f"{n} minus Llama-3.1-8B")}
            out[f"{ro}|{st}"] = res
    return out


def by_action(arms: dict[str, dict], B: int, seed: int) -> dict:
    """Take actions against move, heat, cool and clean: each meaning's share of the goal effect per arm, and the
    Qwen3-8B (plain and block) minus Llama-3.1-8B difference, on the same items."""
    out = {}
    for ro in ("yn", "first"):
        for st in ("later", "step0"):
            for kind in ("take", "other"):
                names = [n for n in arms]
                conds = ("aligned", "swapped") + tuple(x for c in MEANINGS.values() for x in (c, lexsem.CONTROL_OF[c]))
                per = {}
                for n in names:
                    ids = [i for i, r in arms[n].items() if r["stratum"] == st
                           and (r.get("action", "").split(" ")[0] == TAKE) == (kind == "take")
                           and all(value(r, c, ro) is not None for c in conds)]
                    if len(ids) < 20:
                        continue
                    gs = [arms[n][i]["game_id"] for i in ids]
                    boot = Boot(gs, B, seed)
                    rr = [arms[n][i] for i in ids]
                    g = np.array([_goal(r, ro) for r in rr])
                    per[n] = {"n_items": len(ids), "goal_effect_nats": mean(boot, gs, g)[0],
                              **{m: interval(*_share_draws(boot, gs, np.array([_net(r, c, ro) for r in rr]), g))
                                 for m, c in MEANINGS.items()}}
                if per:
                    out[f"{ro}|{st}|{kind}"] = per
            sub = {k: v for k, v in between({n: {i: r for i, r in arms[n].items()
                                                if (r.get("action", "").split(" ")[0] == TAKE)}
                                             for n in arms}, B, seed).items()
                   if k.startswith(("qwen_plain vs llama_chat", "qwen_block vs llama_chat")) and f"|{ro}|{st}" in k}
            if sub:
                out[f"{ro}|{st}|take, Qwen3-8B minus Llama-3.1-8B"] = sub
    return out


def run_family(arms: dict[str, list[dict]], B: int, seed: int) -> dict:
    a = {n: _by_id(r) for n, r in arms.items()}
    return {"between": between(a, B, seed), "interaction": interaction(a, B, seed),
            "by_action": by_action(a, B, seed),
            "scope": ("three sizes of one family do not establish a scaling relationship; behavioural contrasts "
                      "describe how predictions respond, not internal computation; rejecting the pure accounts "
                      "identifies neither a mechanism nor a mixture")}
