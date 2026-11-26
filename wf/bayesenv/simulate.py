"""Planted models: synthetic score records with known effects, for the tests of the analysis (section 12).

A planted model's log-odds on a prompt is scale x (the exact log-odds, computed as if every certain report could be
wrong one time in a thousand) + an item effect shared by every prompt of the item + wording effects shared by every
item of a wording (and of a wording x cell) + independent noise, plus the planted deviation. The records have the
same form as the scorer's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from wf.bayesenv import design as D
from wf.bayesenv import render as R
from wf.bayesenv import world as W

DOUBT = 1e-3


def soft_exact(space: str, t: str, g: str, h: str) -> float:
    """The exact probability with a doubt of 1e-3 on every certain report (a doubting model's target)."""
    if space == "alf":
        task, tag, stated = R.ALF_TYPES[t]
        p = R.alf_exact(t, g, h) if stated else 0.5 / 6   # implicit types: no stated generator; a stand-in target
        return min(max(p, DOUBT), 1 - DOUBT)
    p = R.TYPES[t]
    ev = W.HISTORIES[h]
    if ev.close_look_target:
        ev = replace(ev, close_look_target=False, looks=ev.looks)
        pr = W.prior(g, p.names_key, p.arm == "announced")
        # a close look that misses with probability DOUBT
        base = W.posterior(pr, replace(ev, note=p.note, provenance=p.provenance))
        return max(base * DOUBT / (base * DOUBT + (1 - base)), DOUBT * 1e-3)
    return R.exact(t, g, h)


@dataclass
class Planted:
    scale: float = 0.7
    sd_item: float = 1.0
    sd_noise: float = 1.0          # per prompt; a within-item difference then has SD sqrt(2) x this
    sd_wording: float = 0.0        # wording main effect (goal and tag wordings)
    sd_wording_cell: float = 0.0   # wording x cell
    goal_role: float = 0.0         # added on prompts whose task line names the key (observational arm)
    lexical_mention: float = 0.0   # added wherever the word "key" appears in a cue line (task, tag or note)
    competing_goal: float = 0.0    # added on prompts whose task names the coin
    no_redundancy: bool = False    # in H4 a note acts as if existence were not settled
    announcement_ignored: bool = False
    provenance_overread: bool = False
    wm_only: bool = False          # the planted deviations act only in the world-model framing
    mass: float = 0.9
    logit_size: float = 20.0
    extra: dict = field(default_factory=dict)


def _target(space: str, t: str, g: str, h: str, pm: Planted) -> float:
    if space == "toy":
        p = R.TYPES[t]
        if pm.announcement_ignored and p.arm == "announced":
            return W.logit(soft_exact_obs(t, g, h))
        if pm.provenance_overread and p.provenance == "P1":
            return W.logit(R.exact("P2_key", g, h))
        if pm.no_redundancy and h == "H4" and p.note is not None:
            ev = replace(W.HISTORIES["H4"], exists=False)
            pr = W.prior(g, p.names_key)
            base = W.logit(W.posterior(pr, replace(ev, note=None)))
            with_note = W.logit(W.posterior(pr, replace(ev, note=p.note)))
            return W.logit(R.exact("explore", g, h)) + (with_note - base)
    return W.logit(soft_exact(space, t, g, h))


def soft_exact_obs(t: str, g: str, h: str) -> float:
    p = R.TYPES[t]
    return W.exact(g, h, p.names_key, note=p.note, provenance=p.provenance, announced=False)


def _deviation(space: str, t: str, fr: str, pm: Planted) -> float:
    if pm.wm_only and fr not in ("wm", "alf"):
        return 0.0
    dev = 0.0
    if space == "toy":
        p = R.TYPES[t]
        if pm.goal_role and p.task not in (R.EXPLORE, R.COIN) and p.arm == "obs":
            dev += pm.goal_role
        if pm.lexical_mention and ("key" in p.task.lower() or "key" in p.tag.lower() or p.note is not None):
            dev += pm.lexical_mention
        if pm.competing_goal and p.task == R.COIN:
            dev += pm.competing_goal
    else:
        task, tag, _ = R.ALF_TYPES[t]
        if pm.goal_role and task in R.ALF_GOALS:
            dev += pm.goal_role
        if pm.lexical_mention and ("keychain" in task or "keychain" in tag.lower()):
            dev += pm.lexical_mention
    return dev


def _wording(t: str) -> str | None:
    if t.startswith("key") and t[3:].isdigit():
        return t
    if t.startswith("tag") and t[3:].isdigit():
        return t
    return None


def planted_records(units: list[D.Unit], pm: Planted, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    word_eff: dict = {}
    word_cell: dict = {}
    item_eff: dict = {}
    recs = []
    for u in units:
        ie = item_eff.setdefault((u.space, u.gen, u.hist, u.k), rng.normal(0, pm.sd_item))
        scores = {}
        for t, fr, r in u.sequences():
            T = _target(u.space, t, u.gen, u.hist, pm)
            T = max(min(T, 12.0), -12.0)
            L = pm.scale * T + ie + _deviation(u.space, t, fr, pm) + rng.normal(0, pm.sd_noise)
            w = _wording(t)
            if w and u.space == "toy":
                key = (w, fr, r)
                L += word_eff.setdefault(key, rng.normal(0, pm.sd_wording))
                L += word_cell.setdefault((key, u.gen, u.hist), rng.normal(0, pm.sd_wording_cell))
            pa = 1 / (1 + math.exp(-L))
            lp_a = math.log(pm.mass * pa) if pa > 0 else -50.0
            lp_b = math.log(pm.mass * (1 - pa)) if pa < 1 else -50.0
            scores[f"{t}|{fr}|{r}"] = {"lp_a": lp_a, "lp_b": lp_b, "z_a": pm.logit_size + L / 2,
                                       "z_b": pm.logit_size - L / 2, "top": 0, "lp_top": max(lp_a, lp_b), "n_tok": 400}
        recs.append({"unit": u.uid, "unit_hash": "planted", "space": u.space, "gen": u.gen, "hist": u.hist, "k": u.k,
                     "scores": scores})
    return recs
