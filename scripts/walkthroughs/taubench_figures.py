"""Figures for notes/walkthroughs/03_taubench.md (static PNG, matplotlib).

Headline numbers come from the committed metrics files (notes/tau_families_metrics.json, which carries the two
reference models' confirmatory numbers too, notes/tau_auth_metrics.json and notes/tau_auth_l32_metrics.json). The
item-level scatter reads the stored confirmatory scores (see taubench_data.py). Palette as in textcraft_figures.py
(validated reference slots blue, orange; grey for de-emphasis).

    python scripts/walkthroughs/taubench_figures.py --runs RUNS_ROOT --out notes/walkthroughs/figures
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
import taubench_data as T  # noqa: E402
from textcraft_figures import AXIS, BLUE, GREY, INK, INK2, MUTED, ORANGE, SURFACE, ci_dot, style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NOTES = REPO / "notes"
SAMPLES = {"tau-g-0f4d9dfa8045": "1", "tau-g-6a254cc8d278": "2", "tau-g-4c09d40843dc": "3",
           "tau-g-ca918df46a5f": "4", "tau-g-5b42db1f2d81": "5", "tau-g-77169873036d": "6",
           "tau-g-f3ca5064c885": "7", "tau-g-183ce6de13f7": "8"}


def fig_items(runs: Path, out: Path):
    items = T.items_by_id(runs / "tau_build" / "items.jsonl")
    sc = T.load_scores(runs / "tau_score")
    prim = T.primary(items)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, m in zip(axes, ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")):
        g = lambda c, t: sc[(m, "main", "fail", "first", c, t)]["L"]
        xs = [g("swapped", i["id"]) for i in prim]
        ys = [g("aligned", i["id"]) for i in prim]
        ax.scatter(xs, ys, s=9, color=GREY, alpha=0.55, lw=0, label="primary failing calls (856)", zorder=1)
        lo, hi = min(xs + ys) - 1, max(xs + ys) + 1
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for tid, lab in SAMPLES.items():
            x, y = g("swapped", tid), g("aligned", tid)
            ax.plot([x], [y], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
        ax.plot([], [], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, label="samples of section 4 (labelled)")
        ax.set_title(T.MODEL_NAMES[m], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("first-token L under the swapped task line (nats)")
        ax.set_ylabel("first-token L under the aligned task line (nats)")
        ax.text(0.03, 0.97, "above the diagonal: the aligned task\nraises predicted success", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK2)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("τ-bench retail: every primary failing call, aligned against swapped", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "taubench_items.png", dpi=150)
    plt.close(fig)


def fig_fail_witness(out: Path):
    fm = json.loads((NOTES / "tau_families_metrics.json").read_text())
    rows = [("Qwen3-8B", fm["reference"]["qwen3-8b-fewshot"], True),
            ("Llama-3.1-8B", fm["reference"]["llama3.1-8b-fewshot"], True),
            ("Llama-3.2-3B", fm["models"]["llama3.2-3b"], True),
            ("Mistral-7B\n(fails competence)", fm["models"]["mistral-7b-v0.3"], False)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, (lab, m, ok) in enumerate(rows):
        y = len(rows) - 1 - j
        f = m["primary"]["first"]["aligned_minus_swapped"]
        w = m["checks"]["witness_effect"]
        ci_dot(ax, y + 0.13, f["estimate"], *f["ci"], BLUE, filled=ok, label="failing call" if j == 0 else None)
        ci_dot(ax, y - 0.13, w["estimate"], *w["ci"], ORANGE, filled=ok, marker="s",
               label="working witness call, same order" if j == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("aligned − swapped, first-token log-odds (nats), with 95% user-bootstrap interval")
    ax.plot([], [], "o", ms=8, color=MUTED, mfc=SURFACE, mew=2, ls="none", label="hollow: not interpretable")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("τ-bench retail: failing call and working witness on the same order, aligned − swapped", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "taubench_fail_witness.png", dpi=150)
    plt.close(fig)


def fig_auth(out: Path):
    a = json.loads((NOTES / "tau_auth_metrics.json").read_text())["models"]
    b = json.loads((NOTES / "tau_auth_l32_metrics.json").read_text())["models"]
    rows = [("Qwen3-8B\n(gates pass)", a["qwen3-8b-fewshot"], True),
            ("Llama-3.1-8B\n(gates fail)", a["llama3.1-8b-fewshot"], False),
            ("Llama-3.2-3B\n(gates fail)", b["llama3.2-3b"], False)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, comp, title in [(axes[0], "W", "Request component W\n(what the write request adds over a bare question)"),
                            (axes[1], "M", "Naming component M\n(bare question about the right order − wrong order)")]:
        for j, (lab, m, ok) in enumerate(rows):
            y = len(rows) - 1 - j
            p = m["primary"][comp]
            ci_dot(ax, y + 0.13, p["main"]["estimate"], *p["main"]["ci"], BLUE, filled=ok,
                   label="main prompt (backend said not to see the task)" if j == 0 else None)
            ci_dot(ax, y - 0.13, p["R"]["estimate"], *p["R"]["ci"], ORANGE, filled=ok, marker="s",
                   label="task-blind prompt (stated and shown)" if j == 0 else None)
        ax.axvline(0, color=INK2, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("first-token log-odds (nats), 95% user-bootstrap interval")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows][::-1])
    axes[1].plot([], [], "o", ms=8, color=MUTED, mfc=SURFACE, mew=2, ls="none",
                 label="hollow: a gate failed, category 'inconclusive'")
    h, lab = axes[0].get_legend_handles_labels()
    h2, lab2 = axes[1].get_legend_handles_labels()
    fig.legend(h + h2[-1:], lab + lab2[-1:], loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("τ-bench authorization experiment: does describing a task-blind backend remove the effect?",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "taubench_auth.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "notes/walkthroughs/figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    fig_items(args.runs, args.out)
    fig_fail_witness(args.out)
    fig_auth(args.out)


if __name__ == "__main__":
    main()
