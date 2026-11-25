"""The lexsem diagnostic's interpretation rules (notes/lexsem_design.md, section 8), as code, so that they can be
tested on planted scenarios before any score exists. Inputs are intervals; nothing here reads scores.

Quantities. The mention lift A is the name assertion's net effect (minus its control) over the goal effect G. Every
other mention condition is read as its net effect over the name assertion's net effect: its *relative lift* rho,
where 1 means "lifts as much as asserting the name" and 0 "lifts nothing". Accounts predict levels of rho, so a
two-component mixture shows up as a level that one pure account requires to be zero or full.
"""
from __future__ import annotations

import math

INF = math.inf
# Level zones; the two zones that accounts predict ("about zero", "full") are bounded and equally wide.
LEVEL_ZONES = (("below zero", -INF, -0.25), ("about zero", -0.25, 0.25), ("partial", 0.25, 0.75),
               ("full", 0.75, 1.25), ("above", 1.25, INF))
ACCOUNTS = ("lexical", "semantic", "evidence")
FULL, ZERO, PART, NEG = "full", "about zero", "partial", "below zero"
# relative lift -> account -> zones it allows
PREDICTIONS = {
    "name_use": {"lexical": {FULL}, "semantic": {PART, FULL}, "evidence": {ZERO}},
    "name_elsewhere": {"lexical": {FULL}, "semantic": {FULL}, "evidence": {ZERO}},
    "name_deny": {"lexical": {FULL}, "semantic": {FULL}, "evidence": {ZERO, NEG}},
    "desc_assert": {"lexical": {ZERO}, "semantic": {FULL}, "evidence": {FULL}},
    "desc_use": {"lexical": {ZERO}, "semantic": {PART, FULL}, "evidence": {ZERO}},
    "alias_assert": {"lexical": {ZERO}, "semantic": {FULL}, "evidence": {FULL}},
    # A homonym in an unrelated sense may still activate the object's concept, so semantic activation allows a
    # partial lift; the homonym never decides a reading alone (design section 5).
    "homonym": {"lexical": {PART, FULL}, "semantic": {ZERO, PART}, "evidence": {ZERO}},
}
# Contrasts that may not decide an account's reading on their own: the alias (rarer words than the names) and the
# homonym (not a pure lexical isolate; section 5).
NOT_ALONE = {"alias_assert", "homonym"}
NO_SINGLE = "no single proposed account fits; mixtures and unlisted mechanisms are not distinguished"
# Appended to every reading that names or lists proposed accounts: none identifies a mechanism (design section 2).
OMITTED = "; a mechanism not on the list with the same predictions is not excluded"
PRECISION_LIMIT = 0.25  # half-width above which a contrast is "imprecise" and counts as undecided
A_GATE = 0.25  # lower bound of A needed before relative lifts are read (a lift that is not about zero)


def category(lo: float, hi: float, zones=LEVEL_ZONES) -> set[str]:
    """The zones (each the half-open range from its lower to its upper bound) that a closed interval touches."""
    return {name for name, a, b in zones if lo < b and hi >= a}


def status(lo: float, hi: float, allowed: set[str]) -> str:
    """consistent / contradicted / undecided / imprecise for one interval against an account's allowed zones."""
    if (hi - lo) / 2 > PRECISION_LIMIT:
        return "imprecise"
    cat = category(lo, hi)
    if cat <= allowed:
        return "consistent"
    if not cat & allowed:
        return "contradicted"
    return "undecided"


def account_statuses(rho: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """Per account: each contrast's status and the overall status (contradicted if any contrast contradicts it,
    except contrasts in NOT_ALONE, which contradict only together with another; consistent if every contrast is
    consistent; otherwise undecided)."""
    out = {}
    for acc in ACCOUNTS:
        per = {c: status(*rho[c], PREDICTIONS[c][acc]) for c in PREDICTIONS if c in rho}
        contra = [c for c, s in per.items() if s == "contradicted"]
        decisive = [c for c in contra if c not in NOT_ALONE]
        if decisive:
            overall = "contradicted"
        elif contra:
            overall = "undecided (alias or homonym only)"
        elif per and all(s == "consistent" for s in per.values()):
            overall = "consistent"
        else:
            overall = "undecided"
        out[acc] = {"per_contrast": per, "overall": overall, "contradicted_by": contra}
    return out


def single_account_reading(rho: dict[str, tuple[float, float]], a_interval: tuple[float, float]) -> str:
    """The frozen wording of the single-account reading for one stratum and readout."""
    if a_interval[0] < A_GATE:
        return "not assessable: no mention lift above a quarter of the goal effect to decompose"
    st = account_statuses(rho)
    contra = [a for a in ACCOUNTS if st[a]["overall"] == "contradicted"]
    standing = [a for a in ACCOUNTS if a not in contra]
    if not standing:
        return NO_SINGLE
    if len(standing) == 1:
        a = standing[0]
        if st[a]["overall"] == "consistent":
            return (f"only {a} fits: no other component larger than about a quarter of the mention lift was "
                    f"detected" + OMITTED)
        return f"{a} not contradicted, not shown consistent (the other accounts are contradicted)" + OMITTED
    return "not distinguished between " + " and ".join(standing) + OMITTED


def mention_reading(lo: float, hi: float) -> str:
    """Explanation 1, from A (name assertion net of its control, over the goal effect)."""
    cat = category(lo, hi)
    if cat <= {FULL, "above"}:
        return "a non-goal assertion reproduces the goal effect"
    if cat == {PART}:
        return "a non-goal assertion reproduces part of the goal effect"
    if cat <= {ZERO, NEG}:
        return "a non-goal assertion does not reproduce the goal effect"
    return "inconclusive"


def relevance_reading(lo: float, hi: float) -> dict[str, str]:
    """Task relevance as a sole account (needs A about zero or below) and as a component (A's shortfall from 1)."""
    cat = category(lo, hi)
    sole = "consistent" if cat <= {ZERO, NEG} else "contradicted" if not cat & {ZERO, NEG} else "undecided"
    if hi < 0.75:
        comp = "the goal adds beyond a non-goal assertion"
    elif lo >= 0.75:
        comp = "no goal-role component beyond a quarter of the goal effect"
    else:
        comp = "inconclusive"
    return {"sole": sole, "component": comp}


def difference_reading(lo: float, hi: float, lower: str, same: str, higher: str) -> str:
    """Three-way reading of a difference (step, position, bridge): wholly below -0.25, within +-0.25, above."""
    if hi < -0.25:
        return lower
    if lo >= -0.25 and hi <= 0.25:
        return same
    if lo > 0.25:
        return higher
    return "inconclusive"


def stratum_specific(st_step0: dict, st_later: dict, step_diffs: dict[str, tuple[float, float]]) -> list[str]:
    """Accounts read differently by stratum: contradicted in one stratum and consistent in the other, with the
    responsible contrast's step difference outside +-0.25 (design S6)."""
    out = []
    for acc in ACCOUNTS:
        a, b = st_step0[acc], st_later[acc]
        pairs = ((a, b), (b, a))
        for x, y in pairs:
            if x["overall"] == "contradicted" and y["overall"] == "consistent":
                if any(c in step_diffs and (step_diffs[c][1] < -0.25 or step_diffs[c][0] > 0.25)
                       for c in x["contradicted_by"]):
                    out.append(acc)
    return sorted(set(out))


# Mixture weights: each account's predicted relative lift on each contrast (pure accounts), for a least-squares
# description of the observed relative lifts as weights on the three accounts (descriptive, never a verdict).
PURE = {
    "lexical": {"name_use": 1, "name_elsewhere": 1, "name_deny": 1, "desc_assert": 0, "desc_use": 0,
                "alias_assert": 0, "homonym": 1},
    "semantic": {"name_use": 1, "name_elsewhere": 1, "name_deny": 1, "desc_assert": 1, "desc_use": 1,
                 "alias_assert": 1, "homonym": 0},
    "evidence": {"name_use": 0, "name_elsewhere": 0, "name_deny": 0, "desc_assert": 1, "desc_use": 0,
                 "alias_assert": 1, "homonym": 0},
}


def mixture_weights(points: dict[str, float]) -> dict[str, float]:
    """Least-squares weights on the three pure patterns for the observed relative lifts (no constraint, so a
    misfit shows as weights outside 0..1 or not summing to 1)."""
    import numpy as np

    keys = [k for k in PURE["lexical"] if k in points]
    X = np.array([[PURE[a][k] for a in ACCOUNTS] for k in keys], dtype=float)
    y = np.array([points[k] for k in keys], dtype=float)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(zip(ACCOUNTS, (float(v) for v in w)))


def planted(weights: dict[str, float], evidence_deny: float = 0.0) -> dict[str, float]:
    """True relative lifts of a mixture of the pure patterns (the evidence account's denial lift can be set below
    zero). Used by the tests and the simulation."""
    out = {}
    for k in PURE["lexical"]:
        v = sum(w * PURE[a][k] for a, w in weights.items())
        if k == "name_deny":
            v += weights.get("evidence", 0.0) * evidence_deny
        out[k] = v
    total = sum(weights.values())
    return {k: v / total for k, v in out.items()}


def broad_evidence_status(statuses: dict[str, dict]) -> str:
    """Evidence including pragmatic cues (section 8): contradicted only if the denial contradicts it too, since a
    listed word or an elsewhere sentence may be read as a hint."""
    ev = statuses["evidence"]
    if ev["overall"] != "contradicted":
        return ev["overall"]
    return "contradicted" if ev["per_contrast"].get("name_deny") == "contradicted" else "not contradicted by the denial"


def hint_relevance_reading(lo: float, hi: float) -> str:
    """Relevance read as a hint, from "Ignore the X" relative to the name assertion: it allows only about zero or
    below zero; the three content accounts allow full."""
    cat = category(lo, hi)
    if (hi - lo) / 2 > PRECISION_LIMIT:
        return "imprecise"
    if cat <= {ZERO, NEG}:
        return "consistent with relevance as a hint"
    if not cat & {ZERO, NEG}:
        return "contradicts relevance as a hint"
    return "undecided"


def bridge_reading(lo: float, hi: float) -> str:
    """The stage-0 bridge at step 0 (stage 0, with the think block: 0.29 [0.23, 0.37] on these items)."""
    if lo > 0.5:
        return "specific to the block format"
    if hi < 0.5:
        return "reproduced in plain text"
    return "inconclusive"


def agreement_reading(lo: float, hi: float) -> str:
    """Item-level agreement of a mention effect with the goal effect (descriptive)."""
    if lo >= 0.7:
        return "tracks the goal effect item by item"
    if hi < 0.5:
        return "weak item tracking"
    return "intermediate"


def homonym_reading(lo: float, hi: float) -> str:
    """Descriptive reading of the homonym's relative lift. A lift means the lexical form lifts without the intended
    sense in context; it is not proof of lexical-only processing, since the original concept may also be activated."""
    cat = category(lo, hi)
    if (hi - lo) / 2 > PRECISION_LIMIT:
        return "imprecise"
    if cat <= {PART, FULL, "above"}:
        return "the lexical form lifts without the intended sense in context"
    if cat <= {ZERO, NEG}:
        return "the lexical form without the intended sense does not lift"
    return "undecided"
