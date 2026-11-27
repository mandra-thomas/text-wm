"""Figures for notes/walkthroughs/01_alfworld.md (static PNG, matplotlib, CPU only).

  alfworld_sample_dumbbells.png   the walkthrough's sample items: swapped -> aligned two-candidate score, per model
  alfworld_item_effects.png       distribution of the item-level goal effect over RQ2's 4,666 items, per model
  alfworld_step0_by_history.png   yes/no goal effect by history length, per model (notes/step0_by_history_metrics.json)
  alfworld_family_shares.png      share of failures pushed toward Yes, every family model (families metrics files)
  alfworld_stage1a_rates.png      Stage 1a: how often the chosen action fails, by verifier (decisionsim_metrics.json)

The first reads notes/walkthroughs/data/alfworld_samples.json (alfworld_samples.py); the second reads the
confirmatory scores from the run archive (ARCHIVE, as in alfworld_data.py; the same scores are in the public
release `repro-rq2-2026-11-25`); the rest read committed metrics files only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NOTES = REPO / "notes"
FIG = NOTES / "walkthroughs" / "figures"
DATA = NOTES / "walkthroughs" / "data"

# Reference palette (dataviz skill, palette.md), validated with its validator: slots 1-3 pass all-pairs in light mode.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, ORANGE_LIGHT = "#9ec5f4", "#f5b79c"
GRAY = "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "legend.frameon": False, "savefig.dpi": 160,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def sample_dumbbells() -> None:
    d = json.loads((DATA / "alfworld_samples.json").read_text())
    samples = d["samples"]
    labels = [f"S{i + 1}  {s['action']}" for i, s in enumerate(samples)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, tag, name, dark, light in ((axes[0], "qwen3-8b-fewshot", "Qwen3-8B", BLUE, BLUE_LIGHT),
                                       (axes[1], "llama3.1-8b-fewshot", "Llama-3.1-8B-Instruct", ORANGE, ORANGE_LIGHT)):
        for y, s in enumerate(samples):
            a, b = s["rq2"][tag]["aligned"]["L"], s["rq2"][tag]["swapped"]["L"]
            ax.plot([b, a], [y, y], color=GRAY, lw=2, solid_capstyle="round", zorder=1)
            ax.scatter([b], [y], s=55, color=light, edgecolor=SURFACE, linewidth=2, zorder=2,
                       label=f"{name}: swapped task" if y == 0 else None)
            ax.scatter([a], [y], s=55, color=dark, edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=f"{name}: aligned (real) task" if y == 0 else None)
        ax.axvline(0, color=AXIS, lw=1, zorder=0)
        ax.set_title(name, loc="left")
        ax.set_xlabel("two-candidate log-odds L (nats)")
        ax.grid(axis="y", visible=False)
    handles, labs = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labs += lab
    fig.legend(handles, labs, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.55, 0.0))
    axes[0].set_yticks(range(len(samples)), labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Sample decisions: each failing action's score under the swapped and the aligned task\n"
                 "(right of the vertical line at 0 = the success sentence is preferred to \"Nothing happens.\")", x=0.01,
                 ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "alfworld_sample_dumbbells.png")


def item_effects() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(REPO))
    from alfworld_data import index, load_items, load_rows
    from wf.runs import _pilotA_item_sets

    items = load_items()
    prim = _pilotA_item_sets(list(items.values()))["agnostic"]
    rq2 = index(load_rows("rq2_score"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, tag, name, color in ((axes[0], "qwen3-8b-fewshot", "Qwen3-8B", BLUE),
                                 (axes[1], "llama3.1-8b-fewshot", "Llama-3.1-8B-Instruct", ORANGE)):
        diffs = np.array([rq2[(tag, t["id"], "aligned")]["L"] - rq2[(tag, t["id"], "swapped")]["L"] for t in prim])
        lo, hi = np.percentile(diffs, [0.5, 99.5])
        bins = np.linspace(min(lo, -2), hi, 60)
        ax.hist(np.clip(diffs, bins[0], bins[-1]), bins=bins, color=color, edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(diffs.mean(), color=INK, lw=2)
        ymax = ax.get_ylim()[1]
        ax.text(diffs.mean(), ymax * 0.95, f"  mean {diffs.mean():.2f}", color=INK, va="top", fontsize=9)
        ax.text(0.98, 0.95, f"{(diffs < 0).mean() * 100:.0f}% of items below 0", color=INK2, va="top", ha="right",
                fontsize=9, transform=ax.transAxes)
        ax.set_title(f"{name} (n = {len(diffs):,})", loc="left")
        ax.set_xlabel("goal effect on one item, L(aligned) - L(swapped), nats")
        ax.set_ylabel("items")
        ax.grid(axis="x", visible=False)
    fig.suptitle("The item-level goal effect on RQ2's 4,666 determined failures (confirmatory scores; tails clipped "
                 "at the 0.5th and 99.5th percentiles)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "alfworld_item_effects.png")


def step0_by_history() -> None:
    d = json.loads((NOTES / "step0_by_history_metrics.json").read_text())["models"]
    order = [("qwen3-8b (with the empty think block)", "Qwen3-8B"), ("llama3.1-8b", "Llama-3.1-8B"),
             ("mistral-7b-v0.3", "Mistral-7B"), ("gemma2-9b-it", "Gemma-2-9B"), ("llama3.2-3b", "Llama-3.2-3B")]
    bands = ["0", "1-3", "4-6", "7+"]
    fig, axes = plt.subplots(1, len(order), figsize=(12, 3.4), sharey=True)
    for ax, (key, name) in zip(axes, order):
        rows = d[key]["yes/no"]
        m = [rows[b]["mean_nats"] for b in bands]
        lo = [rows[b]["mean_nats"] - rows[b]["ci95"][0] for b in bands]
        hi = [rows[b]["ci95"][1] - rows[b]["mean_nats"] for b in bands]
        x = np.arange(len(bands))
        ax.errorbar(x, m, yerr=[lo, hi], color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, capsize=0,
                    elinewidth=1.5)
        ax.set_xticks(x, bands)
        ax.set_title(name, loc="left")
        ax.set_xlabel("actions in the history")
        ax.axhline(0, color=AXIS, lw=1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("yes/no goal effect (nats)")
    fig.suptitle("Yes/no goal effect by history length, with 95% game-clustered intervals "
                 "(notes/step0_by_history_metrics.json)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "alfworld_step0_by_history.png")


def family_shares() -> None:
    a = json.loads((NOTES / "families_metrics.json").read_text())
    b = json.loads((NOTES / "families_waveb_metrics_2026-11-26.json").read_text())  # wave B, final
    rows = []
    names = {"qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B", "gemma2-9b-it": "Gemma-2-9B",
             "mistral-7b-v0.3": "Mistral-7B-v0.3", "llama3.2-3b": "Llama-3.2-3B", "llama3.2-1b": "Llama-3.2-1B",
             "ministral3-8b-it": "Ministral-3-8B", "qwen3.5-9b": "Qwen3.5-9B (chat)",
             "qwen3.5-9b-plain": "Qwen3.5-9B (plain)", "qwen3.5-9b-base-plain": "Qwen3.5-9B-Base (plain)",
             "qwen3.8-27b": "Qwen3.8-27B", "gemma4-e4b-it": "Gemma-4-E4B", "gemma4-31b-it": "Gemma-4-31B"}
    for src, wave in ((a, "A"), (b, "B")):
        pool = dict(src["models"])
        if wave == "A":
            pool.update(src["reference"])
        for k, x in pool.items():
            s = x["primary_metrics"]["item_share_toward_yes"]
            comp = x.get("competence", {})
            ok = comp.get("interpretable", comp.get("passes", True)) if isinstance(comp, dict) else True
            rows.append((names.get(k, k) + f"  [wave {wave}]" if k not in ("qwen3-8b", "llama3.1-8b") else names[k] + "  [reference]",
                         s["estimate"], s["ci"][0], s["ci"][1], k != "llama3.2-1b"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for y, (n, e, lo, hi, ok) in enumerate(rows):
        c = BLUE if ok else GRAY
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.scatter([e], [y], s=60, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.008, y, f"{e:.3f}", va="center", fontsize=9, color=INK2)
    ax.axvline(0.5, color=INK2, lw=1)
    ax.text(0.5, -0.9, " 0.5 = no effect", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("share of the 4,666 failures whose yes/no score is higher under the aligned than the swapped task")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.3, len(rows) - 0.5)
    fig.suptitle("Item share toward Yes, with 95% game-clustered intervals\n(gray: fails the competence rule)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "alfworld_family_shares.png")


def stage1a_rates() -> None:
    d = json.loads((NOTES / "decisionsim_metrics.json").read_text())["models"]
    arms = [("none", "no verifier"), ("yn_blind", "task-blind"), ("yn_absent", "neutral task"),
            ("yn_swapped", "swapped task"), ("yn_aligned", "real (aligned) task")]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for j, (tag, name, color) in enumerate((("qwen3-8b", "Qwen3-8B", BLUE), ("llama3.1-8b", "Llama-3.1-8B", ORANGE))):
        ref = d[tag]["reference_points"]
        for i, (arm, _) in enumerate(arms):
            r = ref["by_rule"]["rule2_verification"]["no_verifier_fail_rate"] if arm == "none" else \
                ref["by_arm"][arm]["rule2_verification"]["fail_rate"]
            y = i + (0.17 if j == 0 else -0.17)
            ax.plot([100 * r["ci"][0], 100 * r["ci"][1]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.scatter([100 * r["estimate"]], [y], s=55, color=color, edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=name if i == 0 else None)
            if arm == "yn_aligned" or arm == "none":
                ax.text(100 * r["ci"][1] + 0.6, y, f"{100 * r['estimate']:.1f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(arms)), [a[1] for a in arms])
    ax.set_xlim(0, 30)
    ax.set_xlabel("share of the 728 states at which the chosen action fails (%), rule 2")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Stage 1a: which verifier makes the agent pick failing actions (95% game-clustered intervals)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "alfworld_stage1a_rates.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["sample_dumbbells", "item_effects", "step0_by_history", "family_shares", "stage1a_rates"]
    for w in which:
        globals()[w]()
