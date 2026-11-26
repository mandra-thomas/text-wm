"""The overview's worked example: one ALFWorld item, rendered with the repository's own prompt builders and
checked against the stored scores of every model that scored it (exploratory, descriptive; CPU only).

The item is `t_002295` (game `valid_unseen/191`), the item kept as the golden test fixture
`tests/golden/default_prompt.json`. This script

1. renders the aligned, swapped and yes/no prompts with `wf.scoring.logodds.build_messages` and
   `wf.rq1v.arm_messages`, and checks the two-candidate renders against the golden fixture byte for byte;
2. recomputes the item hash (`wf.runs._item_hash`) and requires every stored score line to carry it;
3. reads the stored score lines for the item, recomputes each log-odds from its two stored log-probabilities,
   and prints the per-model table quoted in notes/walkthroughs/00_overview.md, section 1;
4. draws notes/walkthroughs/figures/overview_worked_example.png.

The score lines come from the run folders on ICE (copied to the off-GitHub run archive). Pass the
`scoring.jsonl` files, or extracts holding this item's lines, as arguments:

    python scripts/walkthroughs/overview_worked_example.py \
        <archive>/text-wm-fix11/runs/rq2_score/shard_*/scoring.jsonl \
        <archive>/text-wm-fix17/runs/rq1v_score/shard_*/scoring.jsonl \
        <archive>/text-wm-fix21/runs/family_score_*/shard_*/scoring.jsonl

Runs used: `rq2_score` (commit ddeabc2; Qwen3-8B and Llama-3.1-8B two-candidate scores, the confirmatory
values), `rq1v_score` (2155b8d; the same two models' yes/no scores, scored in one job with a repeat of the
two-candidate scores), `family_score_<model>` (65eb591; families wave A, every arm in one job per model).
No field of Word2World's starting-layout text is read or printed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wf import rq1v  # noqa: E402
from wf.runs import _item_hash  # noqa: E402
from wf.scoring.logodds import build_messages  # noqa: E402

from overview_palette import INK_2, MUTED, apply_style, model_color, model_label  # noqa: E402

ITEM_ID = "t_002295"
GOLDEN = ROOT / "tests" / "golden" / "default_prompt.json"
FIG = ROOT / "notes" / "walkthroughs" / "figures" / "overview_worked_example.png"

# model tag in the score files -> which run holds its confirmatory two-candidate scores
TWO_CAND_SOURCE = {"qwen3-8b-fewshot": "rq2_score", "llama3.1-8b-fewshot": "rq2_score"}
ORDER = ["qwen3-8b-fewshot", "llama3.1-8b-fewshot", "mistral-7b-v0.3", "gemma2-9b-it", "llama3.2-3b", "llama3.2-1b"]
COMPETENT = {"qwen3-8b-fewshot", "llama3.1-8b-fewshot", "mistral-7b-v0.3", "gemma2-9b-it", "llama3.2-3b"}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def render() -> tuple[dict, dict[str, list[dict]]]:
    golden = json.loads(GOLDEN.read_text())
    item = golden["item"]
    msgs = {
        "aligned": build_messages(item, "aligned"),
        "swapped": build_messages(item, "swapped"),
        "yn_aligned": rq1v.arm_messages(item, "yn_aligned"),
        "yn_swapped": rq1v.arm_messages(item, "yn_swapped"),
        "yn_absent": rq1v.arm_messages(item, "yn_absent"),
    }
    assert msgs["aligned"] == golden["aligned"], "aligned render differs from the golden fixture"
    assert msgs["swapped"] == golden["swapped"], "swapped render differs from the golden fixture"
    # the paired prompts differ only in the task line
    a, s = msgs["aligned"][-1]["content"].splitlines(), msgs["swapped"][-1]["content"].splitlines()
    diff = [(x, y) for x, y in zip(a, s) if x != y]
    assert len(a) == len(s) and len(diff) == 1 and diff[0][0].startswith("Your task is to:"), diff
    assert msgs["aligned"][:-1] == msgs["swapped"][:-1]
    return item, msgs


def read_scores(paths: list[str], item_hash: str) -> dict[tuple[str, str, str], dict]:
    """(run, model, condition) -> score line, for this item only; every line must carry the item hash."""
    out: dict[tuple[str, str, str], dict] = {}
    for p in paths:
        run = next((part for part in Path(p).parts if part.endswith("_score") or part.startswith("family_score_")),
                   Path(p).stem)
        if run.startswith("family_score_"):
            run = "family_score"
        for line in Path(p).read_text().splitlines():
            if f'"{ITEM_ID}"' not in line:
                continue
            d = json.loads(line)
            if d.get("transition_id") != ITEM_ID:
                continue
            if d["item_hash"] != item_hash:
                raise SystemExit(f"{p}: stored item hash {d['item_hash']} != {item_hash}")
            # the stored L is the difference of the two stored log-probabilities
            assert abs(d["L"] - (d["logp_plus"] - d["logp_minus"])) < 1e-6, d
            out[(run, d["model"], d["condition"])] = d
    return out


def table(scores: dict) -> list[dict]:
    rows = []
    for model in ORDER:
        src = TWO_CAND_SOURCE.get(model, "family_score")
        yn_src = "rq1v_score" if model in TWO_CAND_SOURCE else "family_score"
        try:
            al, sw = scores[(src, model, "aligned")], scores[(src, model, "swapped")]
            ya, ys = scores[(yn_src, model, "yn_aligned")], scores[(yn_src, model, "yn_swapped")]
            yn0 = scores[(yn_src, model, "yn_absent")]
        except KeyError:
            continue
        rows.append({
            "model": model,
            "L_aligned": al["L"], "L_swapped": sw["L"], "dL": al["L"] - sw["L"],
            "s_aligned": sigmoid(al["L"]), "s_swapped": sigmoid(sw["L"]),
            "first_aligned": al["L_first"], "first_swapped": sw["L_first"],
            "first_mass_aligned": al["first_mass"], "first_mass_swapped": sw["first_mass"],
            "yn_aligned": ya["L"], "yn_swapped": ys["L"], "yn_neutral": yn0["L"], "dYN": ya["L"] - ys["L"],
            "p_yes_aligned": sigmoid(ya["L"]), "p_yes_swapped": sigmoid(ys["L"]),
            "n_tokens_plus": al["n_tokens_plus"], "n_tokens_minus": al["n_tokens_minus"],
            "logp_plus_aligned": al["logp_plus"], "logp_minus_aligned": al["logp_minus"],
            "logp_plus_swapped": sw["logp_plus"], "logp_minus_swapped": sw["logp_minus"],
        })
    # the yes/no run repeats the two-candidate scores of the reference models in the same job
    for model in TWO_CAND_SOURCE:
        for cond in ("aligned", "swapped"):
            a, b = scores.get(("rq2_score", model, cond)), scores.get(("rq1v_score", model, cond))
            if a and b:
                print(f"  repeat check {model:22s} {cond:8s} rq2_score {a['L']:.6f}  rq1v_score {b['L']:.6f}")
    return rows


def draw(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    ys = list(range(len(rows)))[::-1]
    for ax, (kind, a_key, s_key, title) in zip(axes, [
        ("two", "L_aligned", "L_swapped", "Two-candidate readout: success sentence vs \"Nothing happens.\""),
        ("yn", "yn_aligned", "yn_swapped", "Yes/no readout: \"Yes\" vs \"No\" to the anchored question"),
    ]):
        ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
        for y, r in zip(ys, rows):
            c = model_color(r["model"])
            hollow = r["model"] not in COMPETENT
            ax.plot([r[s_key], r[a_key]], [y, y], color=c, lw=2, alpha=0.55 if not hollow else 0.4, zorder=2)
            ax.plot(r[s_key], y, "o", ms=7, mfc="white", mec=c, mew=1.8, zorder=3)
            ax.plot(r[a_key], y, "o", ms=8, mfc=c if not hollow else "white", mec="white" if not hollow else c,
                    mew=1.6 if not hollow else 1.8, zorder=4)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("log-odds of success, nats")
        ax.grid(axis="y", visible=False)
        ax.text(0.3, -0.85, "success favoured", color=MUTED, fontsize=8, va="center")
        ax.text(-0.3, -0.85, "failure favoured", color=MUTED, fontsize=8, va="center", ha="right")
        ax.set_ylim(-1.2, len(rows) - 0.4)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([model_label(r["model"]) + ("" if r["model"] in COMPETENT else "\n(not competent)")
                             for r in rows])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=INK_2, mew=1.8, label="swapped task (another object)"),
               Line2D([], [], marker="o", ls="", ms=8, mfc=INK_2, mec="white", label="aligned task (the real task)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.02))
    fig.suptitle("One item, every model: take soapbar 1 from cabinet 1 while cabinet 1 is closed (it fails)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), w_pad=2.5)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"wrote {FIG.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", nargs="+", help="scoring.jsonl files (or extracts) holding this item's lines")
    ap.add_argument("--print-prompts", action="store_true", help="print every rendered message")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    item, msgs = render()
    h = _item_hash(item)
    print(f"item {ITEM_ID}: renders match tests/golden/default_prompt.json; item hash {h}")
    if args.print_prompts:
        for k, m in msgs.items():
            print(f"\n===== {k} ({len(m)} messages)")
            for x in m:
                print(f"--- {x['role']}\n{x['content']}")
    scores = read_scores(args.scores, h)
    rows = table(scores)
    print()
    hdr = ("model", "L al", "L sw", "dL", "s al", "s sw", "1st al", "1st sw", "yn al", "yn sw", "yn neu", "dYN",
           "P(Yes) al", "P(Yes) sw")
    print("  ".join(f"{x:>10s}" for x in hdr))
    for r in rows:
        vals = [r["L_aligned"], r["L_swapped"], r["dL"], r["s_aligned"], r["s_swapped"], r["first_aligned"],
                r["first_swapped"], r["yn_aligned"], r["yn_swapped"], r["yn_neutral"], r["dYN"],
                r["p_yes_aligned"], r["p_yes_swapped"]]
        print(f"{r['model'][:10]:>10s}  " + "  ".join(f"{v:10.4g}" for v in vals))
    print()
    for r in rows:
        print(f"{r['model']:22s} tokens o+ {r['n_tokens_plus']}, o- {r['n_tokens_minus']}; "
              f"aligned logp+ {r['logp_plus_aligned']:.4f} logp- {r['logp_minus_aligned']:.4f}; "
              f"swapped logp+ {r['logp_plus_swapped']:.4f} logp- {r['logp_minus_swapped']:.4f}; "
              f"first-token mass {r['first_mass_aligned']:.4f} / {r['first_mass_swapped']:.4f}")
    if not args.no_figure:
        draw(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
