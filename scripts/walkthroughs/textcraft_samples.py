"""Render and verify the sample decisions of notes/walkthroughs/02_textcraft.md.

For each sample this script

1. renders the query turn with the prompt builder of the commit the scores were produced at (round 1:
   ``wf/textcraft/prompts.py`` at 5a3702a; round-1 diagnostics: ``wf/textcraft/dx.py`` at f74c262; round 2:
   ``wf/textcraft/r2.py`` at 4ccf84b; round 3: the same module at fae00b3);
2. recomputes the item hash with that commit's own ``item_hash`` and checks it against the ``item_hash`` stored on
   every score record of the item (a mismatch means the prompt shown is not the prompt scored, and stops the script);
3. prints the stored scores of every model under every condition, and the per-item effects.

Nothing is rescored. Usage (CPU only, no model weights needed):

    python scripts/walkthroughs/textcraft_samples.py --runs RUNS_ROOT --trees TREES_ROOT > samples.md

``TREES_ROOT`` holds one checkout per commit, named tc1 (5a3702a), tcdx (f74c262), tc2 (4ccf84b) and tc3 (fae00b3),
for example made with ``git archive <commit> wf | tar -x -C TREES_ROOT/<name>``. ``RUNS_ROOT`` is laid out as
described in textcraft_data.py; the round-1 diagnostics read ``textcraft_dx_objectgoal`` and
``textcraft_dx_prompts`` as well.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textcraft_data as T  # noqa: E402

ROUND1 = ["tc-p001-1e56f799a78e", "tc-p001-e0e4f5606dd4", "tc-p071-7f77863f789c", "tc-p154-c2758e20ccc1",
          "tc-p012-7de7a9ee1543", "tc-p009-90026de11831", "tc-p068-b4203105258d", "tc-p006-e4c60e48426a",
          "tc-p017-6287c5725565", "tc-p000-e4d689132db0", "tc-p002-b035902c35d2"]
ROUND2_PARENT = "tc-p010-38a14f0049db"
ROUND2_MAIN = "tc-p001-1e56f799a78e"
ROUND3_PARENT = "tc-p057-a1b1aa9a2a86"
# Round-1 diagnostics: two of the shortest main-failure prompts in each run (chosen once, pinned here).
OBJECT_GOAL_ID = "tc-p206-b641eac5a2c5"
PROMPT_VARIANT_ID = "tc-p065-0228d31b0208"


def use_tree(root: Path):
    """Import wf.* from one checkout (purging any previously imported tree)."""
    for name in [m for m in sys.modules if m == "wf" or m.startswith("wf.")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(root.parent))]
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()


def fmt(x):
    return "—" if x is None else f"{x:+.2f}"


def block(text: str) -> str:
    return "```text\n" + text + "\n```"


def check_hash(expected: str, records: list[dict], what: str) -> int:
    bad = [r for r in records if r["item_hash"] != expected]
    if bad:
        raise SystemExit(f"STOP: {what}: stored item_hash {bad[0]['item_hash']} != recomputed {expected}")
    return len(records)


def round1(args, items):
    use_tree(args.trees / "tc1")
    P = importlib.import_module("wf.textcraft.prompts")
    S = importlib.import_module("wf.textcraft.score")
    r1 = T.load_round1(args.runs)
    print("## Round 1 samples (prompt of commit 5a3702a)\n")
    for tid in ROUND1:
        it = items[tid]
        h = S.item_hash(it)
        recs = [r for sc in r1.values() for (t, _c), r in sc.items() if t == tid]
        n = check_hash(h, recs, tid)
        msgs = P.messages(it, "aligned")
        print(f"### {tid}  ({it['population']}, role {it['role']}, class {it['failure_class']}, "
              f"history {it['history_len']} steps)\n")
        print(block(msgs[-1]["content"]))
        print(f"\nGoal lines: aligned `{it['task_line']}` · swapped `{it['swapped_task_line']}` · "
              f"neutral `{it['neutral_task_line']}`")
        print(f"Candidates: o+ `{it['candidates']['o_plus']}` · o- `{it['candidates']['o_minus']}` · true reply "
              f"`{it['reply']}`")
        if it.get("short_inputs"):
            print("Short inputs: " + "; ".join(f"{s['name']} need {s['need']} held {s['held']} peak {s['peak']} "
                                               f"({s['kind']})" for s in it["short_inputs"]))
        print(f"Item hash recomputed {h} = stored on all {n} score records.\n")
        print("| Model | L aligned | L swapped | L neutral | ΔL | first-token ΔL | yes/no aligned | yes/no swapped "
              "| yes/no neutral | yes/no ΔL |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m, sc in r1.items():
            g = lambda c, k="L": sc[(tid, c)][k]
            print(f"| {T.R1_MODELS[m]} | {fmt(g('aligned'))} | {fmt(g('swapped'))} | {fmt(g('neutral'))} | "
                  f"{fmt(g('aligned') - g('swapped'))} | {fmt(g('aligned', 'L_first') - g('swapped', 'L_first'))} | "
                  f"{fmt(g('yn_aligned'))} | {fmt(g('yn_swapped'))} | {fmt(g('yn_neutral'))} | "
                  f"{fmt(g('yn_aligned') - g('yn_swapped'))} |")
        print()
    # The numerical example of section 3: the stored log-probabilities of the first couple member.
    sc = r1["qwen3-8b-fewshot"]
    for c in ("aligned", "swapped"):
        r = sc[(ROUND1[0], c)]
        print(f"worked example, Qwen3-8B, {c}: logp_plus {r['logp_plus']:.4f} logp_minus {r['logp_minus']:.4f} "
              f"L {r['L']:.4f} logp_first_plus {r['logp_first_plus']:.4f} logp_first_minus "
              f"{r['logp_first_minus']:.4f} L_first {r['L_first']:.4f} first_mass {r['first_mass']:.4f} "
              f"n_tokens {r['n_tokens_plus']}/{r['n_tokens_minus']}")
    print()


def diagnostics(args, items):
    """One object-goal sample and one prompt-variant sample (pinned ids, both among the shortest prompts)."""
    use_tree(args.trees / "tcdx")
    DX = importlib.import_module("wf.textcraft.dx")
    P = importlib.import_module("wf.textcraft.prompts")
    r1 = T.load_round1(args.runs)
    og = T.read_jsonl(Path(args.runs) / "textcraft_dx_objectgoal" / "scoring.jsonl")
    pv = T.read_jsonl(Path(args.runs) / "textcraft_dx_prompts" / "scoring.jsonl")
    print("## Round-1 diagnostics (commit f74c262)\n")
    tid = OBJECT_GOAL_ID
    assert any(r["transition_id"] == tid and r["population"] == "main" for r in og), tid
    it = DX.object_goal_item(items[tid])
    print(f"### object goal: {tid} ({items[tid]['role']}, {items[tid]['failure_class']})\n")
    print(block(P.messages(it, "aligned")[-1]["content"]))
    print(f"\nObject-goal line `{[r for r in og if r['transition_id'] == tid][0]['task_line']}`; "
          f"verified aligned `{items[tid]['task_line']}`, swapped `{items[tid]['swapped_task_line']}`\n")
    print("| Model | L aligned (craft G) | L swapped (craft P) | L object goal (craft Y) | yes/no object goal |")
    print("|---|---|---|---|---|")
    for m in T.R1_MODELS:
        o = {r["condition"]: r["L"] for r in og if r["transition_id"] == tid and r["model"] == m}
        sc = r1[m]
        print(f"| {T.R1_MODELS[m]} | {fmt(sc[(tid, 'aligned')]['L'])} | {fmt(sc[(tid, 'swapped')]['L'])} | "
              f"{fmt(o['object_goal'])} | {fmt(o['yn_object_goal'])} |")
    tid = PROMPT_VARIANT_ID
    assert any(r["transition_id"] == tid and r["population"] == "main" for r in pv), tid
    print(f"\n### prompt variants: {tid} ({items[tid]['role']}, {items[tid]['failure_class']})\n")
    it = DX.variant_item(items[tid], "inventory")
    print(block(P.messages(it, "aligned")[-1]["content"]))
    print("\n| Model | prompt | L aligned | L swapped | L neutral | ΔL |")
    print("|---|---|---|---|---|---|")
    for m in T.R1_MODELS:
        sc = r1[m]
        print(f"| {T.R1_MODELS[m]} | verified | {fmt(sc[(tid, 'aligned')]['L'])} | {fmt(sc[(tid, 'swapped')]['L'])} | "
              f"{fmt(sc[(tid, 'neutral')]['L'])} | {fmt(sc[(tid, 'aligned')]['L'] - sc[(tid, 'swapped')]['L'])} |")
        for v in DX.PROMPT_VARIANTS:
            o = {r["condition"].split("|")[1]: r["L"] for r in pv
                 if r["transition_id"] == tid and r["model"] == m and r["condition"].startswith(v + "|")}
            print(f"| {T.R1_MODELS[m]} | {v} | {fmt(o['aligned'])} | {fmt(o['swapped'])} | {fmt(o['neutral'])} | "
                  f"{fmt(o['aligned'] - o['swapped'])} |")
    print()


def round23(args, tree: str, scores: dict[str, dict], names: dict[str, str], title: str, ids: tuple):
    use_tree(args.trees / tree)
    R = importlib.import_module("wf.textcraft.r2")
    P = importlib.import_module("wf.textcraft.prompts")
    recs = T.items_by_id(args.trees / tree / "wf/textcraft/data/r2_items.jsonl")
    shots = P.fewshot_examples()
    print(f"## {title}\n")
    for tid in ids:
        rec = recs[tid]
        h = R.item_hash(rec, shots)
        n = check_hash(h, [r for sc in scores.values() for (t, _c), r in sc.items() if t == tid], tid)
        print(f"### {tid} ({rec['kind']}, role {rec['role']}, class {rec['failure_class']}); item hash {h} = stored "
              f"on {n} records\n")
        arms = ["ext_aligned", "ext_parent"] if rec["kind"] == "parent" else ["aligned", "both", "pad8_aligned"]
        for arm in arms:
            print(f"Arm {arm}:\n")
            print(block(R.messages(rec, arm, shots)[-1]["content"]))
            print()
        for m, sc in scores.items():
            g = lambda c: sc[(tid, c)]["L"] if (tid, c) in sc else None
            if rec["kind"] == "parent" and g("ext_aligned") is not None:
                print(f"- {names[m]}: penalty on this record (ext_aligned minus ext_parent) "
                      f"{fmt(g('ext_aligned') - g('ext_parent'))}")
            elif rec["kind"] == "main" and g("aligned") is not None and g("both") is not None:
                print(f"- {names[m]}: aligned-swapped {fmt(g('aligned') - g('swapped'))}, both-swapped "
                      f"{fmt(g('both') - g('swapped'))}, aligned-both {fmt(g('aligned') - g('both'))}, with one look-up "
                      f"{fmt(g('pad1_aligned') - g('pad1_swapped'))}, with eight {fmt(g('pad8_aligned') - g('pad8_swapped'))}")
        print()
        conds = sorted({c for sc in scores.values() for (t, c) in sc if t == tid})
        print("| Model | " + " | ".join(conds) + " |")
        print("|---|" + "---|" * len(conds))
        for m, sc in scores.items():
            print(f"| {names[m]} | " + " | ".join(fmt(sc[(tid, c)]["L"]) if (tid, c) in sc else "—" for c in conds)
                  + " |")
        print()


def think_excerpt(args):
    gens = T.load_think_generations(args.runs)
    for tid, c in [(t, c) for t in (ROUND3_PARENT, ROUND2_PARENT) for c in ("ext_aligned", "ext_parent")]:
        g = gens.get((tid, c))
        if g:
            print(f"### reasoning trace, {tid} {c}: {g['n_new_tokens']} new tokens, forced={g['forced']}\n")
            print(block(g["reasoning"][:1500] + ("\n[...]" if len(g["reasoning"]) > 1500 else "")))
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--trees", type=Path, required=True)
    args = ap.parse_args()
    items = T.items_by_id(args.trees / "tc1" / "wf/textcraft/data/items.jsonl")
    round1(args, items)
    diagnostics(args, items)
    round23(args, "tc2", T.load_round2(args.runs), T.R2_MODELS, "Round 2 samples (prompt of commit 4ccf84b)",
             (ROUND2_PARENT, ROUND2_MAIN))
    r3 = T.load_round3(args.runs)
    round23(args, "tc3", r3, {k: v[1] for k, v in T.R3_RUNS.items()},
            "Round 3 interim samples (prompt of commit fae00b3)", (ROUND3_PARENT, ROUND2_PARENT, ROUND2_MAIN))
    think_excerpt(args)


if __name__ == "__main__":
    main()
