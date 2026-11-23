"""Exploratory pipeline audit: compare the re-scored sample with the stored confirmatory scores, and the formats.

Reads runs/qwen_audit/ (items.jsonl and every *_bf16, *_repo, *_fp32, *_vllm result present) and writes
summary.json there, printing the tables the note quotes. CPU only.

    python scripts/qwen_audit/compare.py --dir runs/qwen_audit
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

TAGS = {"qwen": "qwen3-8b-fewshot", "llama": "llama3.1-8b-fewshot"}
SOURCES = {  # file stem -> (model key, label)
    "qwen_bf16": ("qwen", "independent, bf16"),
    "qwen_repo": ("qwen", "study scorer, rerun"),
    "qwen_fp32": ("qwen", "independent, fp32 weights, eager attention"),
    "qwen_vllm": ("qwen", "vLLM prompt_logprobs"),
    "llama_bf16": ("llama", "independent, bf16"),
    "llama_vllm": ("llama", "vLLM prompt_logprobs"),
}


def corr(x, y):
    mx, my = st.fmean(x), st.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def boot_ci(vals, games, n=2000, seed=20260924):
    by: dict[str, list[float]] = {}
    for v, g in zip(vals, games):
        by.setdefault(g, []).append(v)
    keys = sorted(by)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [by[k] for k in (rng.choice(keys) for _ in keys)]
        flat = [v for p in pick for v in p]
        means.append(st.fmean(flat))
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/qwen_audit")
    args = ap.parse_args()
    d = Path(args.dir)
    items = [json.loads(x) for x in (d / "items.jsonl").open()]
    order = [it["id"] for it in items]
    meta = {it["id"]: it for it in items}
    summary: dict = {"n_items": len(items), "reproduction": {}, "formats": {}, "checks": {}}

    for stem, (mk, label) in SOURCES.items():
        res = load(d / f"{stem}.jsonl")
        if not res:
            continue
        tag = TAGS[mk]
        ids = [i for i in order if i in res]
        # Reproduction of the stored native scores.
        L_st, L_me, dL_st, dL_me, lp_diff, by_shard = [], [], [], [], [], {}
        for i in ids:
            s, m = meta[i]["stored"][tag], res[i]["scores"]
            for c in ("aligned", "swapped"):
                L_st.append(s[c]["L"]); L_me.append(m[f"native/{c}"]["L"])
                for w in ("plus", "minus"):
                    lp_diff.append(abs(s[c][f"logp_{w}"] - m[f"native/{c}"][f"logp_{w}"]))
                by_shard.setdefault(meta[i]["shard"], []).append(abs(s[c]["L"] - m[f"native/{c}"]["L"]))
            dL_st.append(s["aligned"]["L"] - s["swapped"]["L"])
            dL_me.append(m["native/aligned"]["L"] - m["native/swapped"]["L"])
        absL = [abs(a - b) for a, b in zip(L_st, L_me)]
        absdL = [abs(a - b) for a, b in zip(dL_st, dL_me)]
        rep = {"label": label, "model": tag, "n_items": len(ids),
               "corr_L": corr(L_st, L_me), "corr_dL": corr(dL_st, dL_me),
               "mean_abs_diff_L": st.fmean(absL), "median_abs_diff_L": st.median(absL), "max_abs_diff_L": max(absL),
               "share_L_within_0.01": sum(a < 0.01 for a in absL) / len(absL),
               "mean_abs_diff_logp": st.fmean(lp_diff), "mean_abs_diff_dL": st.fmean(absdL),
               "mean_signed_diff_dL": st.fmean(b - a for a, b in zip(dL_st, dL_me)),
               "effect_stored": st.fmean(dL_st), "effect_here": st.fmean(dL_me),
               "mean_abs_diff_L_by_stored_shard": {str(k): st.fmean(v) for k, v in sorted(by_shard.items())}}
        if mk == "llama":
            k = [i for i in order[:50] if i in res]
            rep["first50"] = {
                "effect_stored": st.fmean(meta[i]["stored"][tag]["aligned"]["L"] - meta[i]["stored"][tag]["swapped"]["L"] for i in k),
                "effect_here": st.fmean(res[i]["scores"]["native/aligned"]["L"] - res[i]["scores"]["native/swapped"]["L"] for i in k),
                "mean_abs_diff_L": st.fmean(abs(meta[i]["stored"][tag][c]["L"] - res[i]["scores"][f"native/{c}"]["L"])
                                            for i in k for c in ("aligned", "swapped"))}
        summary["reproduction"][stem] = rep

        # Format sensitivity.
        fmts = sorted({k.split("/")[0] for r in res.values() for k in r["scores"]})
        native_dL = {i: res[i]["scores"]["native/aligned"]["L"] - res[i]["scores"]["native/swapped"]["L"] for i in ids}
        for f in fmts:
            fi = [i for i in ids if f"{f}/aligned" in res[i]["scores"] and f"{f}/swapped" in res[i]["scores"]]
            a = [res[i]["scores"][f"{f}/aligned"]["L"] for i in fi]
            b = [res[i]["scores"][f"{f}/swapped"]["L"] for i in fi]
            dl = [x - y for x, y in zip(a, b)]
            dlf = [res[i]["scores"][f"{f}/aligned"]["L_first"] - res[i]["scores"][f"{f}/swapped"]["L_first"] for i in fi]
            games = [meta[i]["transition"]["game_id"] for i in fi]
            h0 = [x for x, i in zip(dl, fi) if meta[i]["transition"]["history_len"] == 0]
            h1 = [x for x, i in zip(dl, fi) if meta[i]["transition"]["history_len"] > 0]
            summary["formats"][f"{stem}/{f}"] = {
                "n": len(fi), "mean_L_aligned": st.fmean(a), "mean_L_swapped": st.fmean(b), "effect": st.fmean(dl),
                "effect_ci": boot_ci(dl, games), "median_effect": st.median(dl), "effect_first_token": st.fmean(dlf),
                "share_positive": sum(x > 0 for x in dl) / len(dl),
                "effect_history0": st.fmean(h0) if h0 else None, "n_history0": len(h0),
                "effect_history_gt0": st.fmean(h1) if h1 else None,
                "corr_dL_with_native": corr(dl, [native_dL[i] for i in fi]) if f != "native" else 1.0}

        # Recorded checks.
        chk: dict[str, int] = {}
        for i in ids:
            for k, v in res[i].get("checks", {}).items():
                chk[f"{k}={v}"] = chk.get(f"{k}={v}", 0) + 1
            for key, sc in res[i]["scores"].items():
                for flag in ("alone_ok", "prefix_ok"):
                    if flag in sc:
                        kk = f"{key.split('/')[0]}/{flag}={sc[flag]}"
                        chk[kk] = chk.get(kk, 0) + 1
                if "first_plus" in sc and isinstance(sc["first_plus"], str):
                    kk = f"{key.split('/')[0]}/first_tokens={sc['first_plus']!r}|{sc['first_minus']!r}"
                    chk[kk] = chk.get(kk, 0) + 1
        summary["checks"][stem] = dict(sorted(chk.items()))

    # Implementation-only difference: both ran on the same GPU in one job.
    a, b = load(d / "qwen_bf16.jsonl"), load(d / "qwen_repo.jsonl")
    common = [i for i in order if i in a and i in b]
    if common:
        diffs = [abs(a[i]["scores"][f"native/{c}"]["L"] - b[i]["scores"][f"native/{c}"]["L"]) for i in common
                 for c in ("aligned", "swapped")]
        summary["independent_vs_study_scorer_same_gpu"] = {"n_items": len(common), "mean_abs_diff_L": st.fmean(diffs),
                                                            "max_abs_diff_L": max(diffs)}
    for g in d.glob("*_gpu.txt"):
        summary.setdefault("gpus", {})[g.stem] = g.read_text().strip()
    (d / "summary.json").write_text(json.dumps(summary, indent=1))

    print("## Reproduction of stored native scores")
    for stem, r in summary["reproduction"].items():
        print(f"{stem:11s} n={r['n_items']} corrL={r['corr_L']:.5f} corrdL={r['corr_dL']:.5f} MAD_L={r['mean_abs_diff_L']:.4f} "
              f"medAD={r['median_abs_diff_L']:.4f} max={r['max_abs_diff_L']:.3f} within.01={r['share_L_within_0.01']:.2f} "
              f"MAD_dL={r['mean_abs_diff_dL']:.4f} effect stored={r['effect_stored']:.3f} here={r['effect_here']:.3f} "
              f"byshard={ {k: round(v, 4) for k, v in r['mean_abs_diff_L_by_stored_shard'].items()} }")
        if "first50" in r:
            print("   first 50:", {k: round(v, 3) for k, v in r["first50"].items()})
    print("## Formats")
    for k, r in summary["formats"].items():
        print(f"{k:22s} n={r['n']} La={r['mean_L_aligned']:7.3f} Ls={r['mean_L_swapped']:7.3f} dL={r['effect']:6.3f} "
              f"CI=[{r['effect_ci'][0]:.2f},{r['effect_ci'][1]:.2f}] med={r['median_effect']:.3f} "
              f"dLfirst={r['effect_first_token']:.3f} pos={r['share_positive']:.2f} h0={r['effect_history0']} "
              f"(n={r['n_history0']}) h>0={r['effect_history_gt0']:.3f} corr_native={r['corr_dL_with_native']:.3f}")
    print("## Checks")
    for k, v in summary["checks"].items():
        print(k, v)
    print(summary.get("independent_vs_study_scorer_same_gpu"), summary.get("gpus"))


if __name__ == "__main__":
    main()
