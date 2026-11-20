"""Activation patching for one prompt pair, and the fraction of the effect restored (design §5).

Noising: target = aligned prompt, source = swapped (`build_pair(..., "aligned", "swapped")`).
Denoising: the same pair flipped (`pair.flipped()`), so the swapped prompt receives aligned activations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wf.interp.hooks import Patch, capture, run_patched
from wf.interp.model import Parts
from wf.interp.prompts import COARSE_GROUPS, GROUPS, Pair


@dataclass(frozen=True)
class Cell:
    site: str
    layer: int
    group: str
    head: int | None = None

    def key(self) -> str:
        return f"{self.site}|{self.layer}|{self.group}" + ("" if self.head is None else f"|h{self.head}")


def grid(n_layers: int, sites: tuple[str, ...] = ("resid",), groups: tuple[str, ...] = GROUPS) -> list[Cell]:
    return [Cell(s, layer, g) for s in sites for layer in range(n_layers) for g in groups]


def design_grid(n_layers: int) -> dict[str, list[Cell]]:
    """The grids of the design (§5): residual by every group; attention and MLP by the coarse groups."""
    return {"resid": grid(n_layers, ("resid",), GROUPS),
            "blocks": grid(n_layers, ("attn", "mlp"), COARSE_GROUPS)}


def head_cells(layers: list[int], n_heads: int, group: str = "final") -> list[Cell]:
    return [Cell("head_in", layer, group, h) for layer in layers for h in range(n_heads)]


def _ids(p: Parts, ids: list[int], rows: int) -> Any:
    import torch

    return torch.tensor([ids] * rows, dtype=torch.long, device=p.device)


def readout(logits: Any, plus: int, minus: int) -> list[float]:
    return (logits[:, plus] - logits[:, minus]).tolist()


def sweep(p: Parts, pr: Pair, cells: list[Cell], plus: int, minus: int, batch_size: int = 16) -> dict:
    """Readout of `pr.a` with each cell's activations taken from `pr.b`; one unpatched row in every batch.

    Every batch has exactly `batch_size` rows (the last is padded with unpatched rows), and the source run is a
    batch of the same size, because in bfloat16 on a GPU a row's numbers can depend on the batch's size. Each
    cell's patched readout is paired with the unpatched readout of its own batch (`cell_clean`), so a fraction
    never mixes two batches. Readouts are float32 from the final normed hidden state. Returns the clean target
    readout of the first batch (`d_target`), the source readout, every batch's clean readout, and one patched
    readout per cell (None for an empty group).
    """
    import torch

    from wf.interp.hooks import readout_rows

    if batch_size < 2:
        raise ValueError("a batch needs an unpatched row and at least one patched row")
    needed = sorted({(c.site, c.layer) for c in cells})
    _, src, src_h = capture(p, _ids(p, pr.b.ids, batch_size), needed, rows=1, with_hidden=True)
    d_source = float(p.readout(src_h[:1], plus, minus)[0])
    todo = [c for c in cells if pr.groups.get(c.group)]
    out: dict[str, float | None] = {c.key(): None for c in cells}
    cell_clean: dict[str, float] = {}
    clean: list[float] = []
    for k in range(0, max(1, len(todo)), batch_size - 1):
        chunk = todo[k:k + batch_size - 1]
        patches = []
        for r, c in enumerate(chunk, start=1):
            tgt = pr.groups[c.group]
            src_pos = torch.tensor(pr.to_b(tgt), device=p.device)
            value = src[(c.site, c.layer)][0, src_pos]
            if c.head is not None:
                value = value[:, c.head * p.head_dim:(c.head + 1) * p.head_dim]
            patches.append(Patch(c.site, c.layer, r, torch.tensor(tgt, device=p.device), value, c.head))
        d = readout_rows(p, _ids(p, pr.a.ids, batch_size), plus, minus, patches)
        clean.append(d[0])
        for c, v in zip(chunk, d[1:]):
            out[c.key()] = v
            cell_clean[c.key()] = d[0]
    return {"d_target": clean[0], "d_source": d_source, "clean": clean, "clean_spread": max(clean) - min(clean),
            "cells": out, "cell_clean": cell_clean, "batch_size": batch_size,
            "n_target": len(pr.a), "n_source": len(pr.b), "equal_span": pr.equal_span,
            "group_sizes": {g: len(v) for g, v in pr.groups.items()}}


def cell_values(record: dict, cell_key: str) -> tuple[float, float, float] | None:
    """(clean target of the cell's own batch, source, patched) for one sweep record, or None if not patched."""
    v = record["cells"].get(cell_key)
    if v is None:
        return None
    return record.get("cell_clean", {}).get(cell_key, record["d_target"]), record["d_source"], v


def fraction_restored(d_target: float, d_source: float, d_patched: float) -> float:
    """Share of the target-minus-source effect that patching the cell removes (1 = all, 0 = none)."""
    effect = d_target - d_source
    return float("nan") if effect == 0 else (d_target - d_patched) / effect


def ratio_of_sums(rows: list[tuple[float, float, float]]) -> float:
    """Pooled fraction over items: sum of removed effect over sum of effect. Rows are (target, source, patched)."""
    num = sum(t - pt for t, _s, pt in rows)
    den = sum(t - s for t, s, _pt in rows)
    return float("nan") if den == 0 else num / den


def pooled_fraction(records: list[dict], cell_key: str, seed: int = 0, B: int = 2000) -> dict:
    """Ratio-of-sums fraction for one cell over item records from `sweep`, with a game-clustered bootstrap."""
    from wf.analysis.stats import cluster_bootstrap

    rows = [(*v, r["game_id"]) for r in records for v in [cell_values(r, cell_key)] if v is not None]
    if not rows:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    res = cluster_bootstrap(lambda xs: ratio_of_sums([x[:3] for x in xs]), rows, [x[3] for x in rows], B=B, seed=seed)
    return {**res, "n": len(rows)}


def arrival_layer(fractions_by_layer: list[float], threshold: float = 0.5) -> float:
    """First layer (as a share of depth, 0 to 1) from which the fraction stays at or above `threshold`; nan if never."""
    n = len(fractions_by_layer)
    for i in range(n):
        if all(f == f and f >= threshold for f in fractions_by_layer[i:]):
            return i / max(1, n - 1)
    return float("nan")
