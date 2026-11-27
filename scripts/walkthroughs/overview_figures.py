"""Cross-benchmark summary figures for notes/walkthroughs/00_overview.md (exploratory, descriptive; CPU only).

Every number drawn here is read from a metrics file committed in notes/ (the path and key are next to each
read below). Nothing is recomputed from scores and no study whose analysis has not run is read. A few
values are simple transformations of committed numbers, marked where they occur: a "share kept" is
1 minus a committed "share removed", with the interval transformed the same way.

    python scripts/walkthroughs/overview_figures.py          # writes notes/walkthroughs/figures/overview_*.png

Figures:
  overview_share_moved.png     share of failing items whose predicted success rises under the real task
  overview_standardized.png    standardized goal effect (mean item shift / SD of item shifts), three readouts
  overview_what_changes.png    share of the goal effect kept or reproduced under prompt changes
  overview_step0.png           goal effect by how many actions the history holds
  overview_decisions.png       consequences for decisions: Stage 1a, Stage 1b, Option C
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
OUT = NOTES / "walkthroughs" / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from overview_palette import AXIS, INK, INK_2, MUTED, apply_style, model_color, model_label  # noqa: E402


def load(name: str) -> dict:
    return json.loads((NOTES / name).read_text())


def get(d: dict, path: str):
    for p in path.split("/"):
        d = d[p]
    return d


def row(model, est, lo, hi, label=None, hollow=False, source=""):
    return {"model": model, "est": est, "lo": lo, "hi": hi, "label": label or model_label(model),
            "hollow": hollow, "source": source}


def dot(ax, y, r, ms=7.5):
    c = model_color(r["model"])
    ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    if r["hollow"]:
        ax.plot(r["est"], y, "o", ms=ms, mfc="white", mec=c, mew=2, zorder=3)
    else:
        ax.plot(r["est"], y, "o", ms=ms, mfc=c, mec="white", mew=1.5, zorder=3)


def grouped_rows(ax, groups, xlabel, ref=None, ref_label=None):
    """Draw groups [(header, [rows])] top to bottom; returns the y positions used."""
    y, ticks, labels, header_ys = 0.0, [], [], []
    for header, rows in groups:
        header_ys.append((y, header))
        y -= 1.0
        for r in rows:
            dot(ax, y, r)
            ticks.append(y)
            labels.append(r["label"])
            y -= 1.0
        y -= 0.35
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for hy, h in header_ys:
        ax.annotate(h, xy=(0, hy), xycoords=("axes fraction", "data"), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, fontweight="bold", color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK_2, lw=1.0, zorder=1)
        if ref_label:
            ax.annotate(ref_label, xy=(ref, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10),
                        textcoords="offset points", fontsize=8, color=MUTED, va="top")
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlabel(xlabel)
    return ticks


# ---------------------------------------------------------------------------------------------------
# Figure 1: share of failing items moved toward success
# ---------------------------------------------------------------------------------------------------
def fig_share():
    fa = load("families_metrics.json")            # families wave A (final)
    fb = load("families_waveb_metrics_2026-11-26.json")   # families wave B (final, 2026-11-26)
    tf = load("tau_families_metrics.json")        # τ-bench families extension (final)

    def wa(path, model, **kw):
        d = get(fa, path + "/primary_metrics/item_share_toward_yes")
        return row(model, d["estimate"], *d["ci"], source="families_metrics.json:" + path, **kw)

    def wb(path, model, **kw):
        d = get(fb, path + "/primary_metrics/item_share_toward_yes")
        return row(model, d["estimate"], *d["ci"], source="families_waveb_metrics_2026-11-26.json:" + path, **kw)

    def tau(path, model, **kw):
        d = get(tf, path + "/primary/item_share/first")
        return row(model, d["estimate"], *d["ci"], source="tau_families_metrics.json:" + path, **kw)

    alf = [
        wa("reference/qwen3-8b", "qwen3-8b"),
        wa("reference/llama3.1-8b", "llama3.1-8b"),
        wa("models/mistral-7b-v0.3", "mistral-7b-v0.3"),
        wa("models/llama3.2-3b", "llama3.2-3b"),
        wa("models/gemma2-9b-it", "gemma2-9b-it"),
        wb("reference/qwen3-32b", "qwen3-32b", label="Qwen3-32B (rescored for wave B)"),
        wb("models/ministral3-8b-it", "ministral3-8b-it"),
        wb("models/qwen3.5-9b", "qwen3.5-9b", label="Qwen3.5-9B (chat template)"),
        wb("models/qwen3.5-9b-plain", "qwen3.5-9b", label="Qwen3.5-9B (plain text)"),
        wb("models/qwen3.5-9b-base-plain", "qwen3.5-9b", label="Qwen3.5-9B-Base (plain text)"),
        wb("models/gemma4-e4b-it", "gemma4-e4b-it"),
        wb("models/qwen3.8-27b", "qwen3.8-27b"),
        wb("models/gemma4-31b-it", "gemma4-31b-it"),
    ]
    taub = [
        tau("reference/qwen3-8b-fewshot", "qwen3-8b"),
        tau("reference/llama3.1-8b-fewshot", "llama3.1-8b"),
        tau("models/llama3.2-3b", "llama3.2-3b"),
    ]
    for r in alf + taub:
        print(f"  share  {r['label']:34s} {r['est']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  {r['source']}")
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    grouped_rows(ax, [("ALFWorld, yes/no question", alf), ("τ-bench retail, first token", taub)],
                 "share of determined failures whose score is higher under the real task than under the swapped task",
                 ref=0.5, ref_label="0.5 = no effect")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("The real task moves most failing actions toward success, in every competent model")
    fig.tight_layout()
    fig.savefig(OUT / "overview_share_moved.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Figure 2: standardized effect, three readouts
# ---------------------------------------------------------------------------------------------------
def fig_standardized():
    fo = load("qwen_effect_audit_forensic_metrics.json")["models"]
    xa = load("textcraft_vs_alfworld_metrics.json")
    fu = load("qwen_effect_audit_followup_metrics.json")["A_qwen3_8b_all_items"]["all"]
    alf_models = [("qwen3-8b", "qwen3-8b"), ("llama3.1-8b", "llama3.1-8b"), ("mistral-7b", "mistral-7b-v0.3"),
                  ("llama3.2-3b", "llama3.2-3b"), ("gemma2-9b", "gemma2-9b-it"),
                  ("qwen3-4b (size run)", "qwen3-4b"), ("qwen3-14b (size run)", "qwen3-14b"),
                  ("qwen3-32b (size run)", "qwen3-32b")]
    panels = [("Two-candidate (full reply)", "scale/L/dz_ci", "standardized_sd"),
              ("First token of the reply", "scale/F/dz_ci", "standardized_sd_first"),
              ("Yes/no question", "yes_no/scale/dz_ci", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.4), sharey=True)
    for k, (ax, (title, key, xkey)) in enumerate(zip(axes, panels)):
        alf = []
        for tag, m in alf_models:
            d = get(fo[tag], key)
            alf.append(row(m, d["value"], d["lo"], d["hi"]))
        if k == 0:
            alf.insert(1, row("qwen3-8b", fu["standardized_without"], *fu["standardized_without_ci"],
                              label="Qwen3-8B, no empty think block", hollow=True))
        else:
            alf.insert(1, None)
        tc = []
        for tag, m in (("qwen3-8b-fewshot", "qwen3-8b"), ("llama3.1-8b-fewshot", "llama3.1-8b")):
            if xkey:
                d = xa[tag]["scales"]["textcraft_main"][xkey]
                tc.append(row(m, d["est"], d["lo"], d["hi"]))
            else:
                tc.append(None)
        # draw with placeholders so rows line up across panels
        y, ticks, labels, headers = 0.0, [], [], []
        for header, rows, names in (("ALFWorld (4,666 failures)", alf,
                                     [model_label(m) for _, m in alf_models[:1]] + ["Qwen3-8B, no empty think block"]
                                     + [model_label(m) for _, m in alf_models[1:]]),
                                    ("TextCraft round 1 (1,500 failures)", tc, ["Qwen3-8B", "Llama-3.1-8B-Instruct"])):
            headers.append((y, header))
            y -= 1.0
            for r, name in zip(rows, names):
                if r is not None:
                    dot(ax, y, r)
                ticks.append(y)
                labels.append(name)
                y -= 1.0
            y -= 0.35
        if k == 2:
            ax.text(0.5, y + 2.5, "not compared: TextCraft round 1's\nquestion was not anchored to the reply",
                    transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, ha="center", va="center")
        ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
        ax.set_ylim(y + 0.6, 0.6)
        ax.set_xlim(-0.35, 1.6)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("standardized effect d")
        for k2, rr in enumerate(alf + tc):
            if rr is not None:
                print(f"  d  {title[:12]:12s} {rr['label']:32s} {rr['est']:.2f} [{rr['lo']:.2f}, {rr['hi']:.2f}]")
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels)
    for hy, h in headers:
        axes[0].annotate(h, xy=(0, hy), xycoords=("axes fraction", "data"), xytext=(-6, 0),
                         textcoords="offset points", ha="right", va="center", fontsize=9, fontweight="bold", color=INK)
    fig.suptitle("Standardized goal effect (mean aligned-minus-swapped shift / SD of the item shifts), 95% intervals",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "overview_standardized.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Figure 3: what changes the effect
# ---------------------------------------------------------------------------------------------------
def fig_what_changes():
    au = load("audit_metrics.json")["ratios"]
    me = load("mech_metrics.json")
    m2 = load("mechanism2_metrics.json")
    ta = load("tau_auth_metrics.json")["decision"]["per_model"]["qwen3-8b-fewshot"]["removed_shares"]
    Q, L, M = "qwen3-8b-fewshot", "llama3.1-8b-fewshot", "mistral-7b-v0.3"
    tag = {Q: "qwen3-8b", L: "llama3.1-8b", M: "mistral-7b-v0.3"}

    def kept(d):            # committed ratio "kept"
        return d["estimate"], d["ci"][0], d["ci"][1]

    def removed(d):         # 1 - committed "share removed"; interval transformed the same way
        return 1 - d["estimate"], 1 - d["ci"][1], 1 - d["ci"][0]

    def battery(model, arm):
        d = m2[model]["arms_on_sample"][arm]["vs_swapped_L"]["ratio"]
        return d["estimate"], d["ci"][0], d["ci"][1]

    groups = [
        ("Changes to both task lines or the whole prompt (share of the effect that remains)", [
            ("Object only described, never named", {Q: kept(au[Q]["object_free_kept"]), L: kept(au[L]["object_free_kept"]),
                                                     M: battery(M, "described")}),
            ("\"Ignore the goal\" note appended", {Q: removed(au[Q]["ignore_note_share_removed"]),
                                                   L: removed(au[L]["ignore_note_share_removed"])}),
            ("Facts that decide the failure stated", {Q: removed(me[Q]["interventions"]["facts"]["share_of_effect_removed"]),
                                                      L: removed(me[L]["interventions"]["facts"]["share_of_effect_removed"])}),
            ("Environment's rules stated", {Q: removed(me[Q]["interventions"]["rules"]["share_of_effect_removed"]),
                                            L: removed(me[L]["interventions"]["rules"]["share_of_effect_removed"])}),
            ("Hidden starting layout shown", {Q: removed(au[Q]["full_layout_share_removed"]),
                                              L: removed(au[L]["full_layout_share_removed"])}),
            ("τ-bench: backend said to ignore the task\n(request part, first token)", {Q: removed(ta["W"])}),
            ("τ-bench: backend said to ignore the task\n(naming part, first token)", {Q: removed(ta["M"])}),
        ]),
        ("A sentence added to the swapped task only (share of the effect reproduced)", [
            ("\"Ignore the <object>.\"", {Q: battery(Q, "mention_ignore"), L: battery(L, "mention_ignore"),
                                          M: battery(M, "mention_ignore")}),
            ("\"There is a <object> somewhere in the room.\"", {Q: battery(Q, "mention_exists"),
                                                                L: battery(L, "mention_exists")}),
        ]),
    ]
    sizes = [len(items) for _, items in groups]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.6), sharex=True, gridspec_kw={"height_ratios": sizes})
    offs = {Q: 0.22, L: 0.0, M: -0.22}
    for ax, (header, items) in zip(axes, groups):
        ticks, labels = [], []
        for i, (name, vals) in enumerate(items):
            y = -float(i)
            for mod, (e, lo, hi) in vals.items():
                dot(ax, y + offs[mod], row(tag[mod], e, lo, hi), ms=6.5)
                print(f"  kept  {name.splitlines()[0][:45]:45s} {model_label(tag[mod]):26s} {e:.2f} [{lo:.2f}, {hi:.2f}]")
            ticks.append(y)
            labels.append(name)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-len(items) + 0.5, 0.5)
        ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
        ax.axvline(1, color=AXIS, lw=1.0, zorder=1)
        ax.set_title(header, fontsize=9.5)
    axes[0].annotate("0 = effect gone", xy=(0, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
                     textcoords="offset points", fontsize=8, color=MUTED, va="top")
    axes[0].annotate("1 = effect unchanged", xy=(1, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
                     textcoords="offset points", fontsize=8, color=MUTED, va="top")
    axes[1].set_xlim(-0.15, 1.85)
    axes[1].set_xlabel("effect under the change / plain aligned-minus-swapped effect, same items and readout")
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=model_color(tag[m]), mec="white", label=model_label(tag[m]))
               for m in (Q, L, M)]
    axes[0].legend(handles=handles, loc="lower right", ncol=1)
    fig.suptitle("What changes the goal effect, and what does not (two-candidate readout unless noted; 95% intervals)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "overview_what_changes.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Figure 4: step 0 against later steps
# ---------------------------------------------------------------------------------------------------
def fig_step0():
    s0 = load("step0_by_history_metrics.json")["models"]
    order = [("qwen3-8b (with the empty think block)", "qwen3-8b", "Qwen3-8B (with the empty think block)"),
             ("llama3.1-8b", "llama3.1-8b", "Llama-3.1-8B-Instruct"),
             ("mistral-7b-v0.3", "mistral-7b-v0.3", "Mistral-7B-Instruct-v0.3"),
             ("gemma2-9b-it", "gemma2-9b-it", "Gemma-2-9B-it"),
             ("llama3.2-3b", "llama3.2-3b", "Llama-3.2-3B-Instruct")]
    bins = ["0", "1-3", "4-6", "7+"]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.4))
    for ax, (key, m, title) in zip(axes, order):
        c = model_color(m)
        for readout, ls, mk in (("two-candidate", "-", "o"), ("yes/no", (0, (4, 2)), "s")):
            b = s0[key][readout]
            est = [b[h]["mean_nats"] for h in bins]
            lo = [b[h]["ci95"][0] for h in bins]
            hi = [b[h]["ci95"][1] for h in bins]
            xs = range(len(bins))
            ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(xs, est, ls=ls, color=c, marker=mk, ms=5, mfc=c, mec="white")
            print(f"  step0  {title[:24]:24s} {readout:13s} " + "  ".join(f"{e:.2f}" for e in est))
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(["0", "1–3", "4–6", "7+"])
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("actions in the history")
    axes[0].set_ylabel("goal effect, nats (own scale per model)")
    handles = [Line2D([], [], color=INK_2, ls="-", marker="o", label="two-candidate readout"),
               Line2D([], [], color=INK_2, ls=(0, (4, 2)), marker="s", label="yes/no readout")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Goal effect by history length on the RQ2 failures, 95% game-clustered intervals",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "overview_step0.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Figure 5: decisions
# ---------------------------------------------------------------------------------------------------
def fig_decisions():
    ds = load("decisionsim_metrics.json")["option_c"]["per_model"]
    dx = {d["model"]: d for d in load("decisionsim_dx_metrics.json")["frozen_functions_secondary_models"]
          if d["comparison"] == "yn_blind"}
    ag = load("decisionsim_agent_metrics.json")
    oc = load("optionc_metrics.json")["runs"]

    def pts(d, est="fail_rate_difference", ci="fail_rate_difference_ci"):
        return 100 * d[est], 100 * d[ci][0], 100 * d[ci][1]

    a = [row("qwen3-8b", *pts(ds["qwen3-8b"])), row("llama3.2-3b", *pts(dx["llama3.2-3b"])),
         row("gemma2-9b-it", *pts(dx["gemma2-9b-it"])), row("mistral-7b-v0.3", *pts(dx["mistral-7b-v0.3"])),
         row("llama3.1-8b", *pts(ds["llama3.1-8b"]))]
    for r, sec in zip(a, (False, True, True, True, False)):
        if sec:
            r["label"] += " (secondary)"
    po = ag["primary"]["overall"]["per_model"]
    qd = ag["models"]["qwen3-8b"]["primary"]["difference"]
    b = [row("llama3.1-8b", 100 * po["llama3.1-8b"]["delta_invalid"], *[100 * x for x in po["llama3.1-8b"]["ci"]]),
         row("mistral-7b-v0.3", 100 * po["mistral-7b-v0.3"]["delta_invalid"], *[100 * x for x in po["mistral-7b-v0.3"]["ci"]]),
         row("qwen3-8b", 100 * qd["estimate"], *[100 * x for x in qd["ci"]], label="Qwen3-8B (contrast)")]

    def oc_row(run, rule, label):
        d = oc[run]["rules"][rule]["contrasts"]["aware_minus_blind"]["success"]
        return row("qwen3-8b", 100 * d["estimate"], 100 * d["ci"][0], 100 * d["ci"][1], label=label)

    c = [oc_row("optionc_qwen3-8b_obs", "accepted_only", "observation readout,\naccepted-only rule (primary)"),
         oc_row("optionc_qwen3-8b_yn", "accepted_only", "yes/no readout,\naccepted-only rule"),
         oc_row("optionc_qwen3-8b_yn_argmax", "argmax", "yes/no readout,\nargmax rule")]
    panels = [
        ("Stage 1a: fixed candidate menus", a,
         "extra states where the chosen action fails,\nreal-task minus task-blind verifier (points)"),
        ("Stage 1b: the agent's own candidates", b,
         "change in share of decisions acting on a\nfailing action, real-task minus task-blind (points)"),
        ("Option C: closed loop, Qwen3-8B verifier", c,
         "change in episode success, real-task minus task-blind verifier (points; below 0 = success lost)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 8.4), gridspec_kw={"hspace": 0.95, "height_ratios": [5, 3, 3]})
    for ax, (title, rows, xl) in zip(axes, panels):
        ys = [-(i) for i in range(len(rows))]
        for yy, r in zip(ys, rows):
            dot(ax, yy, r)
            print(f"  dec  {title[:10]:10s} {r['label'].splitlines()[0]:36s} {r['est']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
        ax.axvline(0, color=INK_2, lw=1.0, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(min(ys) - 0.6, 0.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xl.replace("\n", " "), fontsize=8.5)
    fig.suptitle("Does the goal effect change decisions? 95% game-clustered intervals",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.36, right=0.98, top=0.91, bottom=0.07)
    fig.savefig(OUT / "overview_decisions.png", dpi=200)
    plt.close(fig)


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("figure 1"); fig_share()
    print("figure 2"); fig_standardized()
    print("figure 3"); fig_what_changes()
    print("figure 4"); fig_step0()
    print("figure 5"); fig_decisions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
