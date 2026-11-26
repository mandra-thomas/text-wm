"""Render and verify the sample decisions of notes/walkthroughs/03_taubench.md.

For each sample this script

1. renders the query turn with the prompt builder of the frozen commit (``wf/tau/prompts.py`` at c737cd6 for the
   confirmatory and families prompts, ``wf/tau/auth.py`` at f70b285 for the authorization arms);
2. recomputes the item hash with ``wf.tau.pipeline.item_hash`` of c737cd6 and checks it against the ``item_hash``
   stored on every score record of the item, in every run (a mismatch stops the script);
3. prints the stored first-token, yes/no and comprehension scores of every model, and the per-item effects.

Long tool outputs are shortened for reading; every shortening is marked in the printed text. Nothing is rescored.

    python scripts/walkthroughs/taubench_samples.py --runs RUNS_ROOT --trees TREES_ROOT > samples.md

``TREES_ROOT`` holds checkouts named ``tau`` (c737cd6) and ``tauauth`` (f70b285); ``RUNS_ROOT`` is laid out as in
taubench_data.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taubench_data as T  # noqa: E402

SAMPLES = ["tau-g-0f4d9dfa8045", "tau-g-6a254cc8d278", "tau-g-4c09d40843dc", "tau-g-ca918df46a5f",
           "tau-g-5b42db1f2d81", "tau-g-77169873036d", "tau-g-f3ca5064c885", "tau-g-183ce6de13f7"]
AUTH_SAMPLES = ["tau-g-0f4d9dfa8045", "tau-g-6a254cc8d278", "tau-g-77169873036d"]
MODELS = [("confirmatory", "qwen3-8b-fewshot"), ("confirmatory", "llama3.1-8b-fewshot"),
          ("families_llama3.2-3b", "llama3.2-3b"), ("families_mistral-7b-v0.3", "mistral-7b-v0.3")]
AUTH_MODELS = [("auth", "qwen3-8b-fewshot"), ("auth", "llama3.1-8b-fewshot"), ("auth_llama3.2-3b", "llama3.2-3b")]
VERBATIM_MAX = 1400


def use_tree(root: Path):
    for name in [m for m in sys.modules if m == "wf" or m.startswith("wf.")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(root.parent))]
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()


def fmt(x):
    return "—" if x is None else f"{x:+.2f}"


def excerpt(call: dict, keep_ids: set[str]) -> str:
    """The output verbatim when short; otherwise a marked excerpt that keeps every field the item's outcome reads."""
    out = call["output"]
    if len(out) <= VERBATIM_MAX:
        return out
    d = json.loads(out)
    if call["name"] == "get_product_details":
        variants = d.get("variants", {})
        kept = {k: v for k, v in variants.items() if k in keep_ids}
        short = {"name": d["name"], "product_id": d["product_id"], "variants": kept}
        return (json.dumps(short) + f"\n[excerpt by the walkthrough: {len(variants) - len(kept)} of the product's "
                f"{len(variants)} variants omitted; the variants named by the scored calls are kept verbatim]")
    dropped = [k for k in ("address", "email", "name", "fulfillments") if k in d]
    dropped += [k for k, v in d.items() if v is None]
    short = {k: v for k, v in d.items() if k not in dropped}
    return json.dumps(short) + f"\n[excerpt by the walkthrough: fields omitted: {', '.join(dropped)}]"


def query_text(PR, item: dict, condition: str, action: str = "fail", cue: str = "output") -> str:
    acts = [item["action"]] + ([item["aligned_witness"]] if item.get("aligned_witness") else [])
    keep = {i for a in acts for k in ("item_ids", "new_item_ids") for i in a["arguments"].get(k, [])}
    hist = [{**h, "output": excerpt(h, keep)} for h in item["history"]]
    return PR.user_turn(PR.task_for(item, condition), hist, PR.action_for(item, action), cue=cue)


def get(sc, run, m, arm, act, ro, cond, tid):
    r = sc[run].get((m, arm, act, ro, cond, tid))
    return None if r is None else r["L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--trees", type=Path, required=True)
    args = ap.parse_args()
    items = T.items_by_id(args.runs / "tau_build" / "items.jsonl")
    sc = T.load_all(args.runs)
    use_tree(args.trees / "tau")
    PR = importlib.import_module("wf.tau.prompts")
    PL = importlib.import_module("wf.tau.pipeline")
    fewshot = json.loads((args.runs / "tau_build" / "fewshot.json").read_text())
    print("## Prompt pieces (commit c737cd6)\n")
    sysmsg = PR.system_message("main")
    print(f"System message: {len(sysmsg)} characters. Instruction:\n\n```text\n{PR.INSTRUCTION}\n```\n")
    print(f"Backend section:\n\n```text\n# How the backend checks write calls\n\n{PR._BACKEND_LEAD['main']}\n\n"
          f"{PR.BACKEND_RULES}\n```\n")
    print("Few-shot examples (design order): " + "; ".join(
        f"{i + 1}. {ex['names']} order, {ex['outcome']}: task `{ex['task']}`, call `{ex['action']['name']}`, "
        f"output starts `{ex['output'][:60]}`" for i, ex in enumerate(fewshot)) + "\n")
    print(f"Yes/no question:\n\n```text\n{PR.YN_QUESTION}\n```\n\nManipulation question:\n\n```text\n"
          f"{PR.TARGET_QUESTION}\n```\n")
    for tid in SAMPLES:
        it = items[tid]
        h = PL.item_hash(it)
        recs = [r for run in sc.values() for k, r in run.items() if k[5] == tid]
        bad = [r for r in recs if r["item_hash"] != h]
        if bad:
            raise SystemExit(f"STOP: {tid}: stored hash {bad[0]['item_hash']} != recomputed {h}")
        msgs = PR.build_messages(it, "aligned", fewshot)
        assert msgs[-1]["content"].startswith("Task: " + it["task_aligned"])
        print(f"### {tid} (class {it['cls']}, call {it['action']['name']}, order status {it['target_status']}, "
              f"user {it['user_id']})\n")
        print(f"Item hash recomputed {h} = stored on all {len(recs)} score records (all runs). Few-shot order "
              f"{'reversed' if PR.fewshot_reversed(tid) else 'as designed'}. Full query turn "
              f"{len(msgs[-1]['content'])} characters.\n")
        print("```text\n" + query_text(PR, it, "aligned") + "\n```\n")
        for c in ("swapped", "neutral", "mention_aligned", "mention_swapped"):
            print(f"- {c}: `{PR.task_for(it, c)}`")
        print(f"\nTrue reply `{it['failure_reply']}`; success twin starts `{it['success_twin'][:70]}…`")
        if it.get("aligned_witness"):
            print(f"Witness call `{json.dumps(it['aligned_witness'])}` → output starts "
                  f"`{it['witness_output'][:60]}…`; text-only checker on the witness: {it['witness_checker']}\n")
        print("| Model | fail al | fail sw | fail ne | fail ment-al | fail ment-sw | fail ΔL | witness al | witness sw "
              "| witness ne | witness ΔL | fail minus witness | yes/no fail al | yes/no fail sw | yes/no fail ne "
              "| yes/no witness al | yes/no witness sw | yes/no witness ne | target al | target sw |")
        print("|" + "---|" * 20)
        for run, m in MODELS:
            g = lambda act, ro, c: get(sc, run, m, "main", act, ro, c, tid)
            fd = g("fail", "first", "aligned") - g("fail", "first", "swapped")
            wd = g("witness", "first", "aligned") - g("witness", "first", "swapped")
            cells = [g("fail", "first", c) for c in ("aligned", "swapped", "neutral", "mention_aligned",
                                                    "mention_swapped")] + [fd]
            cells += [g("witness", "first", c) for c in ("aligned", "swapped", "neutral")] + [wd, fd - wd]
            cells += [g("fail", "yn", c) for c in ("aligned", "swapped", "neutral")]
            cells += [g("witness", "yn", c) for c in ("aligned", "swapped", "neutral")]
            cells += [g("fail", "target", c) for c in ("aligned", "swapped")]
            print(f"| {T.MODEL_NAMES[m]} | " + " | ".join(fmt(x) for x in cells) + " |")
        extra = [(arm, c, get(sc, "confirmatory", m, arm, "fail", "first", c, tid))
                 for m in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot") for arm in ("rules_only", "matched")
                 for c in ("aligned", "swapped")]
        if any(x[2] is not None for x in extra):
            print("\nExploratory arms (first token, failing call): " + "; ".join(
                f"{arm} {c} {fmt(v)}" for arm, c, v in extra if v is not None))
        print()
    # The worked example: the stored first-token log-probabilities of the first sample, Qwen3-8B.
    for c in ("aligned", "swapped"):
        r = sc["confirmatory"][("qwen3-8b-fewshot", "main", "fail", "first", c, SAMPLES[0])]
        print(f"worked example {c}: logp_first_plus {r['logp_first_plus']:.4f} logp_first_minus "
              f"{r['logp_first_minus']:.4f} L {r['L']:.4f} first_mass {r['first_mass']:.4f} "
              f"tokens {r['first_token_plus']!r}/{r['first_token_minus']!r}")
    print()
    # Authorization arms.
    use_tree(args.trees / "tauauth")
    AU = importlib.import_module("wf.tau.auth")
    print("## Authorization arms (commit f70b285)\n")
    for arm in ("blind", "auth"):
        AU.system_message(arm)  # builds (and so checks) the arm's full system message
        print(f"{arm} coda:\n\n```text\n{AU.CODA[arm]}\n```\n")
        if arm == "auth":
            print(f"auth lead:\n\n```text\n{AU.AUTH_LEAD}\n```\n")
    for k, q in AU.COMPREHENSION.items():
        print(f"- {k}: `{q}`")
    comp_ids = sorted({key[5] for key in sc["auth"] if key[3] == "K1"})
    tid = comp_ids[0]
    print(f"\nComprehension item {tid}:\n")
    print("| Model | arm | K1 | K2 | K3 | K4 |")
    print("|---|---|---|---|---|---|")
    for run, m in AUTH_MODELS:
        for arm in ("main", "blind", "auth"):
            print(f"| {T.MODEL_NAMES[m]} | {arm} | " + " | ".join(
                fmt(get(sc, run, m, arm, "fail", k, "neutral", tid)) for k in ("K1", "K2", "K3", "K4")) + " |")
    print()
    for tid in AUTH_SAMPLES:
        it = items[tid]
        print(f"### {tid} under the three arms\n")
        print("| Model | arm | fail al | fail sw | fail ment-al | fail ment-sw | G | M | W | witness al | witness sw "
              "| witness G |")
        print("|" + "---|" * 12)
        for run, m in AUTH_MODELS:
            for arm in ("main", "blind", "auth"):
                g = lambda act, c: get(sc, run, m, arm, act, "first", c, tid)
                G = g("fail", "aligned") - g("fail", "swapped")
                M = g("fail", "mention_aligned") - g("fail", "mention_swapped")
                print(f"| {T.MODEL_NAMES[m]} | {arm} | " + " | ".join(fmt(x) for x in (
                    g("fail", "aligned"), g("fail", "swapped"), g("fail", "mention_aligned"),
                    g("fail", "mention_swapped"), G, M, G - M, g("witness", "aligned"), g("witness", "swapped"),
                    g("witness", "aligned") - g("witness", "swapped"))) + " |")
        print()


if __name__ == "__main__":
    main()
