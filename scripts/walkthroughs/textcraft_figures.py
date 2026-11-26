"""Figures for notes/walkthroughs/02_textcraft.md (static PNG, matplotlib).

Headline numbers come from the committed metrics files (notes/textcraft_metrics.json, notes/textcraft_r2_metrics.json,
notes/textcraft_r3_interim_metrics.json). The item-level scatter reads the stored round-1 scores (see
textcraft_data.py). Palette: the validated reference categorical slots blue #2a78d6, orange #eb6834, aqua #1baf7a,
with grey for de-emphasis; checked with the dataviz skill's palette validator (light mode, all checks pass; aqua sits
below 3:1 contrast on the surface, so every series also has a legend entry and a distinct marker).

    python scripts/walkthroughs/textcraft_figures.py --runs RUNS_ROOT --out notes/walkthroughs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textcraft_data as T  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NOTES = REPO / "notes"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, MUTED, GRID, AXIS, SURFACE = "#b9b8b2", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Sample numbers used in the walkthrough (section 4), item id -> label.
SAMPLES = {"tc-p001-1e56f799a78e": "1a", "tc-p001-e0e4f5606dd4": "1b", "tc-p071-7f77863f789c": "2",
           "tc-p154-c2758e20ccc1": "3", "tc-p012-7de7a9ee1543": "4", "tc-p009-90026de11831": "5",
           "tc-p068-b4203105258d": "6"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def ci_dot(ax, y, est, lo, hi, color, filled=True, marker="o", label=None):
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([est], [y], marker=marker, ms=8, color=color, mfc=color if filled else SURFACE, mew=2, zorder=3,
            ls="none", label=label, markeredgecolor=color)


def fig_items(runs: Path, out: Path):
    items = T.items_by_id(REPO / "wf/textcraft/data/items.jsonl")
    r1 = T.load_round1(runs)
    main = [i for i in items.values() if i["population"] == "main"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, m in zip(axes, T.R1_MODELS):
        sc = r1[m]
        xs = [sc[(i["id"], "swapped")]["L"] for i in main]
        ys = [sc[(i["id"], "aligned")]["L"] for i in main]
        ax.scatter(xs, ys, s=9, color=GREY, alpha=0.55, lw=0, label="main failure items (1,500)", zorder=1)
        lo, hi = min(xs + ys) - 1, max(xs + ys) + 1
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for tid, lab in SAMPLES.items():
            x, y = sc[(tid, "swapped")]["L"], sc[(tid, "aligned")]["L"]
            ax.plot([x], [y], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
        ax.plot([], [], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, label="samples of section 4 (labelled)")
        ax.set_title(T.R1_MODELS[m], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("L under the swapped goal line (nats)")
        ax.set_ylabel("L under the aligned goal line (nats)")
        ax.text(0.03, 0.97, "above the diagonal: the aligned line\nraises predicted success", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK2)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("TextCraft round 1: every main failure item, aligned against swapped", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "textcraft_r1_items.png", dpi=150)
    plt.close(fig)


def fig_role_class(out: Path):
    d = json.loads((NOTES / "textcraft_metrics.json").read_text())["per_model"]
    cells = [("goal|never_obtained", "goal craft, input never obtained"), ("goal|too_few", "goal craft, too few"),
             ("ingredient|never_obtained", "ingredient craft, never obtained"),
             ("ingredient|too_few", "ingredient craft, too few"), ("ingredient|used_up", "ingredient craft, used up")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, (m, color, marker, off) in enumerate([("qwen3-8b-fewshot", BLUE, "o", 0.14),
                                                  ("llama3.1-8b-fewshot", ORANGE, "s", -0.14)]):
        for j, (key, _lab) in enumerate(cells):
            e = d[m]["descriptive"]["main"]["strata"]["role_x_class"][key]["delta_L"]
            ci_dot(ax, len(cells) - 1 - j + off, e["estimate"], e["ci"][0], e["ci"][1], color, marker=marker,
                   label=T.R1_MODELS[m] if j == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([lab for _k, lab in cells][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔL = L(aligned) − L(swapped), mean over items (nats), with 95% pair-bootstrap interval")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("TextCraft round 1: where the goal effect sits (role × failure class)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "textcraft_r1_role_class.png", dpi=150)
    plt.close(fig)


R2_ORDER = ["qwen3-8b", "llama3.1-8b", "gemma2-9b-it", "mistral-7b-v0.3", "llama3.2-3b"]
R2_LABEL = {"qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B", "gemma2-9b-it": "Gemma-2-9B",
            "mistral-7b-v0.3": "Mistral-7B", "llama3.2-3b": "Llama-3.2-3B"}
CONTRASTS = [("lift", "Intermediate lift Λ"), ("parent_penalty", "Within-item penalty P"),
             ("naming", "Naming Dn"), ("completion", "Completion Dc"), ("timing", "Added empty steps T")]


def fig_r2(out: Path):
    d = json.loads((NOTES / "textcraft_r2_metrics.json").read_text())["per_model"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.9), sharey=True)
    for ax, (c, title) in zip(axes, CONTRASTS):
        for j, m in enumerate(R2_ORDER):
            e = d[m]["contrasts"][c]
            comp = d[m]["competence"]["local"]["per_contrast"][c]["competent"]
            y = len(R2_ORDER) - 1 - j
            ci_dot(ax, y, e["estimate"], e["ci95"][0], e["ci95"][1], BLUE if comp else GREY, filled=comp)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("nats")
    axes[0].set_yticks(range(len(R2_ORDER)))
    axes[0].set_yticklabels([R2_LABEL[m] for m in R2_ORDER][::-1])
    axes[0].plot([], [], "o", ms=8, color=BLUE, label="model discriminates in this contrast's populations")
    axes[0].plot([], [], "o", ms=8, color=GREY, mfc=SURFACE, mew=2, label="does not discriminate (effect is "
                 "'measurable only')")
    fig.legend(loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("TextCraft round 2: the five confirmatory contrasts, estimate and 95% pair-bootstrap interval",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "textcraft_r2_contrasts.png", dpi=150)
    plt.close(fig)


def fig_r3(out: Path):
    r2 = json.loads((NOTES / "textcraft_r2_metrics.json").read_text())["per_model"]
    r3 = json.loads((NOTES / "textcraft_r3_interim_metrics.json").read_text())
    rows = []  # (label, series, lift (est, lo, hi), penalty (est, lo, hi), competent)
    for m in R2_ORDER:
        e = r2[m]["contrasts"]
        comp = r2[m]["competence"]["local"]["per_contrast"]
        rows.append((R2_LABEL[m], "round 2 (frozen, 2026-11-23)",
                     (e["lift"]["estimate"], *e["lift"]["ci95"]),
                     (e["parent_penalty"]["estimate"], *e["parent_penalty"]["ci95"]),
                     (comp["lift"]["competent"], comp["parent_penalty"]["competent"])))
    for m, lab in [("qwen3.5-9b", "Qwen3.5-9B"), ("ministral3-8b-it", "Ministral-3-8B"), ("olmo3-7b-it", "OLMo 3 7B")]:
        pm = r3["per_model"][m]
        e = pm["contrasts"]
        comp = pm["competence"]["local"]["per_contrast"]
        rows.append((lab, "round 3 INTERIM", (e["lift"]["estimate"], *e["lift"]["ci95"]),
                     (e["parent_penalty"]["estimate"], *e["parent_penalty"]["ci95"]),
                     (comp["lift"]["competent"], comp["parent_penalty"]["competent"])))
    ex = r3["exploratory"]
    g = ex["r3x_gemma4-e4b-it"]["contrasts"]
    rows.append(("Gemma 4 E4B", "round 3 exploratory (no test)", (g["lift"]["estimate"], *g["lift"]["ci95"]),
                 (g["parent_penalty"]["estimate"], *g["parent_penalty"]["ci95"]), (True, True)))
    t = ex["thinking_arm"]["all"]
    rows.append(("Qwen3.5-9B, reasoning on", "round 3 exploratory (no test)",
                 (t["lift"]["thinking_on"]["estimate"], *t["lift"]["thinking_on"]["ci95"]),
                 (t["parent_penalty"]["thinking_on"]["estimate"], *t["parent_penalty"]["thinking_on"]["ci95"]),
                 (True, True)))
    colors = {"round 2 (frozen, 2026-11-23)": (BLUE, "o"), "round 3 INTERIM": (ORANGE, "s"),
              "round 3 exploratory (no test)": (AQUA, "D")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, k, title in [(axes[0], 2, "Intermediate lift Λ\n(ingredient crafts, aligned − swapped)"),
                         (axes[1], 3, "Within-item penalty P\n(same craft: goal names it − goal is its parent)")]:
        seen = set()
        for j, row in enumerate(rows):
            color, marker = colors[row[1]]
            est, lo, hi = row[k]
            ci_dot(ax, len(rows) - 1 - j, est, lo, hi, color, filled=row[4][k - 2], marker=marker,
                   label=row[1] if row[1] not in seen else None)
            seen.add(row[1])
        ax.axvline(0, color=INK2, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("nats, with 95% pair-bootstrap interval")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows][::-1])
    axes[1].plot([], [], "o", ms=8, color=MUTED, mfc=SURFACE, mew=2, ls="none",
                 label="hollow: model does not discriminate here")
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    labels = [lab for ax in axes for lab in ax.get_legend_handles_labels()[1]]
    keep = dict(zip(labels, handles))
    fig.legend(keep.values(), keep.keys(), loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("TextCraft rounds 2 and 3: lift and penalty by model (round 3 is INTERIM)", x=0.01, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "textcraft_r3_lift_penalty.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "notes/walkthroughs/figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    fig_items(args.runs, args.out)
    fig_role_class(args.out)
    fig_r2(args.out)
    fig_r3(args.out)


if __name__ == "__main__":
    main()
