"""The frozen design: cells, prompt types, items and what each scoring run scores (notes/bayesenv_design.md, §§4, 9).

A unit is one item of one cell with every prompt type scored on it, in every framing and readout that type has. Units
are the scoring and resume grain; items are the statistical unit (cells are strata).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from wf.bayesenv import render as R
from wf.bayesenv import world as W

CELLS = tuple((g, h) for g in W.GENERATOR_ORDER for h in W.HISTORY_ORDER)                 # 28 cells
PANEL_CELLS = (("A", "H0"), ("A", "H6"), ("R3", "H2"), ("R3", "H9"), ("C", "H3"), ("C", "H6"), ("D", "H0"),
               ("D", "H9"))                                                                # the wording panel
ANN_CELLS = tuple((g, h) for g in ("A", "C", "D") for h in W.GRADED)                       # announced arm, 18
PROV_CELLS = tuple((g, h) for g in ("A", "C") for h in ("H0", "H2", "H6"))                 # provenance, 6
ALF_CELLS = tuple((g, h) for g in ("A", "C", "D") for h in R.ALF_HISTORIES)               # ALFWorld surface, 6
ALF_IMPLICIT_CELLS = tuple(("none", h) for h in R.ALF_HISTORIES)                           # no generator stated
PANEL_WORDINGS = len(R.GOAL_WORDINGS)                                                      # 8 goal and 8 tag
FRAMING_READOUTS = {"wm": ("first", "yn"), "si": ("narr", "yn")}
ALF_READOUTS = ("first", "yn")
SMOKE_K0 = 1000          # smoke items are disjoint from the full run's (k < 1000)


@dataclass(frozen=True)
class Design:
    """Sizes the smoke-run rule may set (section 9). Defaults are the base design."""
    w_goal: int = 2          # goal wordings on every cell
    w_tag: int = 3           # tag wordings on every cell
    n: int = 30              # items per cell for the core prompt types
    n_h4: int = 30           # items in the existence-settled cells for the explore and note prompts
    n_ann: int = 30          # items per announced-arm cell
    alf: bool = True         # include the ALFWorld-surface rendering

    def to_dict(self) -> dict:
        return asdict(self)


def core_types(d: Design) -> tuple[str, ...]:
    return ("explore", "coin", *[f"key{i + 1}" for i in range(d.w_goal)], "note0", "note50", "note80", "note95",
            *[f"tag{i + 1}" for i in range(d.w_tag)], "tagO", "tagN", "coin_tag")


def panel_extra(d: Design) -> tuple[str, ...]:
    return (*[f"key{i + 1}" for i in range(d.w_goal, PANEL_WORDINGS)],
            *[f"tag{i + 1}" for i in range(d.w_tag, PANEL_WORDINGS)])


@dataclass(frozen=True)
class Unit:
    space: str               # "toy" or "alf"
    gen: str
    hist: str
    k: int
    types: tuple[str, ...]

    @property
    def uid(self) -> str:
        return f"{self.space}|{self.gen}|{self.hist}|{self.k}"

    def sequences(self) -> list[tuple[str, str, str]]:
        """(prompt type, framing, readout) of every sequence this unit scores."""
        out = []
        for t in self.types:
            if self.space == "alf":
                out += [(t, "alf", r) for r in ALF_READOUTS]
                continue
            for fr in (("wm",) if t in R.WM_ONLY else ("wm", "si")):
                out += [(t, fr, r) for r in FRAMING_READOUTS[fr]]
        return out


def _merge(units: dict, space: str, gen: str, hist: str, k: int, types) -> None:
    key = (space, gen, hist, k)
    old = units.get(key, ())
    units[key] = old + tuple(t for t in types if t not in old)


def full_units(d: Design, k_max: int | None = None) -> list[Unit]:
    """Every unit of the full run (or, with k_max, only items k < k_max: the cross-GPU subset)."""
    u: dict = {}
    core = core_types(d)
    for g, h in CELLS:
        n_cell = max(d.n, d.n_h4 if h == "H4" else 0, d.n_ann if (g, h) in ANN_CELLS else 0)
        for k in range(n_cell):
            if k_max is not None and k >= k_max:
                break
            if k < d.n:
                _merge(u, "toy", g, h, k, core)
                if (g, h) in PANEL_CELLS:
                    _merge(u, "toy", g, h, k, panel_extra(d))
                if (g, h) in PROV_CELLS:
                    _merge(u, "toy", g, h, k, ("P1_key", "P2_key"))
            if h == "H4" and k < d.n_h4:
                _merge(u, "toy", g, h, k, ("explore", "note0", "note50", "note80", "note95"))
            if (g, h) in ANN_CELLS and k < d.n_ann:
                _merge(u, "toy", g, h, k, ("ann_key", "ann_coin"))
    if d.alf:
        stated = tuple(t for t, v in R.ALF_TYPES.items() if v[2])
        implicit = tuple(t for t, v in R.ALF_TYPES.items() if not v[2])
        for g, h in ALF_CELLS:
            for k in range(d.n if k_max is None else min(d.n, k_max)):
                _merge(u, "alf", g, h, k, stated)
        for g, h in ALF_IMPLICIT_CELLS:
            for k in range(d.n if k_max is None else min(d.n, k_max)):
                _merge(u, "alf", g, h, k, implicit)
    return [Unit(*key, types=types) for key, types in sorted(u.items(), key=lambda kv: _order(kv[0]))]


def _order(key) -> tuple:
    space, g, h, k = key
    gi = W.GENERATOR_ORDER.index(g) if g in W.GENERATOR_ORDER else 9
    return (space != "toy", k, gi, h)


XGPU_K = 9   # the cross-GPU subset: items k < 9 of every cell, every arm on them (28 x 9 + 8 x 9 = 324 items)


def smoke_units() -> list[Unit]:
    """The smoke run (section 9.1): evidence slope on every cell; the wording panel with all eight goal and tag
    wordings; the announced and provenance arms on six cells; a few ALFWorld-surface items. Items k >= 1000."""
    u: dict = {}
    all_core = core_types(Design(w_goal=PANEL_WORDINGS, w_tag=PANEL_WORDINGS))
    for g, h in CELLS:
        for k in range(SMOKE_K0, SMOKE_K0 + 10):
            _merge(u, "toy", g, h, k, ("explore",))
    for g, h in PANEL_CELLS:
        for k in range(SMOKE_K0, SMOKE_K0 + 10):
            _merge(u, "toy", g, h, k, tuple(t for t in all_core if t not in ("note50", "note80")))
    for g, h in PROV_CELLS:
        for k in range(SMOKE_K0, SMOKE_K0 + 10):
            _merge(u, "toy", g, h, k, ("key1", "ann_key", "ann_coin", "P1_key", "P2_key"))
    stated = tuple(t for t, v in R.ALF_TYPES.items() if v[2])
    for g, h in ALF_CELLS:
        for k in range(SMOKE_K0, SMOKE_K0 + 5):
            _merge(u, "alf", g, h, k, stated)
    for g, h in ALF_IMPLICIT_CELLS:
        for k in range(SMOKE_K0, SMOKE_K0 + 5):
            _merge(u, "alf", g, h, k, tuple(t for t, v in R.ALF_TYPES.items() if not v[2]))
    return [Unit(*key, types=types) for key, types in sorted(u.items(), key=lambda kv: _order(kv[0]))]


def count_sequences(units: list[Unit]) -> dict[str, int]:
    out: dict[str, int] = {"toy": 0, "alf": 0}
    for un in units:
        out[un.space] += len(un.sequences())
    return out


def shard(units: list[Unit], i: int, n: int) -> list[Unit]:
    return units[i::n]


def prompt_for(unit: Unit, ptype: str, framing: str, readout: str) -> R.Prompt:
    if unit.space == "alf":
        return R.alf_render(ptype, unit.gen, unit.hist, unit.k, readout)
    return R.render(ptype, unit.gen, unit.hist, unit.k, framing, readout)


def exact_for(space: str, ptype: str, gen: str, hist: str) -> float | None:
    if space == "alf":
        return R.alf_exact(ptype, gen, hist)
    return R.exact(ptype, gen, hist)


def unit_hash(unit: Unit) -> str:
    """What a unit's scores depend on: every rendered prompt (so a score never attaches to a changed prompt)."""
    h = hashlib.sha256()
    for t, fr, r in unit.sequences():
        p = prompt_for(unit, t, fr, r)
        h.update(json.dumps([t, fr, r, p.messages, p.prefix, p.candidates]).encode())
    return h.hexdigest()[:16]
