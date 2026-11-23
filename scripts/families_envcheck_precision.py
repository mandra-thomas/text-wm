"""Wave B environment check, precision diagnostic (notes/families_waveb_design.md, section 8 amendment).

Why: Qwen3-8B's serve2 rescoring failed only the yes/no per-score criterion (mean |diff| 0.248 nats against 0.07),
with every difference a whole multiple of 0.25 nats. The hypothesis is bf16 rounding of large logits: at magnitude
32-64 the bf16 grid is 0.25 apart, so a tiny numerical change from the software stack moves a logit by a whole step.

What it does, on a GPU, in one environment per call (`score`): for a fixed subset of the environment check's 300 items
(every sixth, 50 items), each yes/no prompt (three conditions) and each two-candidate prompt with each candidate (three
conditions), one forward pass exactly as the scorer makes it, with a hook on the final norm. From that pass it saves

- the bf16 logits the scorer reads (Yes and No; each candidate token) and the scorer's own L (log-softmax in float32);
- `head64`: the same final hidden state (bf16, after the final norm) times the output layer in float64, so the
  logits are not rounded to bf16 (the "upcast final hidden state and lm_head" reading; float64 rather than float32
  so no TF32 path can enter);
- `norm64`: the final norm itself recomputed in float64 from the bf16 residual stream, then the float64 head;
- the token ids' hash, the environment, the attention implementation and every load-time setting it can read.

`compare` (CPU) reads the old-environment and serve2 files and the two archived runs and reports: whether each
environment's bf16 L reproduces its archived score exactly; logit magnitudes and their bf16 spacing; old-vs-new
differences in bf16 and in the float64 readings, by condition; and whether every bf16 difference is what rounding the
float64 logits to bf16 gives (the rounding account).

    python scripts/families_envcheck_precision.py score --label old --out runs_diag/old.jsonl   # study env
    python scripts/families_envcheck_precision.py score --label new --out runs_diag/new.jsonl   # serve2
    python scripts/families_envcheck_precision.py compare --old .../old.jsonl --new .../new.jsonl \
        --archived-old runs/size_score_qwen3-8b --archived-new runs/family_envcheck_qwen3-8b --out report.json

Each record is appended as it finishes and a rerun skips what is saved (checkpoint per unit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

YN = ("yn_aligned", "yn_swapped", "yn_absent")
PRED = ("aligned", "swapped", "absent")
STEP = 6  # every sixth of the 300 items: 50 items


def bf16_spacing(x: float) -> float:
    """The gap between adjacent bf16 values at |x| (8 significant bits: 2^(e-7) for |x| in [2^e, 2^(e+1)))."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def round_bf16(x: float) -> float:
    """x rounded to the nearest bf16 value (ties to even), as a GPU writes an accumulator to bf16 (normal range)."""
    if x == 0 or not math.isfinite(x):
        return x
    s = bf16_spacing(x)
    return round(x / s) * s  # Python's round is ties-to-even; x / s and the product are exact in float64


def _items(n_items: int = 300) -> list[dict]:
    from wf import rq1x
    from wf.families import placement_items
    from wf.rq2 import SOURCES, load_items

    sets = rq1x.rq1x_sets(load_items(SOURCES))
    return placement_items(sets["primary"], n_items)[::STEP]


def _environment(scorer: Any) -> dict[str, Any]:
    import platform

    import tokenizers
    import torch
    import transformers

    m = scorer._model
    cfg = m.config
    b = torch.backends
    rec: dict[str, Any] = {
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__, "cuda": torch.version.cuda, "cudnn": b.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_class": type(m).__name__, "dtype": str(m.dtype),
        "attn_implementation": getattr(cfg, "_attn_implementation", None),
        "sdp_flash": b.cuda.flash_sdp_enabled(), "sdp_mem_efficient": b.cuda.mem_efficient_sdp_enabled(),
        "sdp_math": b.cuda.math_sdp_enabled(),
        "sdp_cudnn": b.cuda.cudnn_sdp_enabled() if hasattr(b.cuda, "cudnn_sdp_enabled") else None,
        "matmul_allow_tf32": b.cuda.matmul.allow_tf32,
        "matmul_bf16_reduced_precision_reduction": b.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cudnn_allow_tf32": b.cudnn.allow_tf32, "deterministic": torch.are_deterministic_algorithms_enabled(),
        "revision": scorer.revision, "tokenizer_class": type(scorer._tok).__name__,
        "config": json.loads(cfg.to_json_string(use_diff=False)),
    }
    return rec


def _hooked_forward(scorer: Any, ids: Any) -> tuple[Any, Any, Any]:
    """(bf16 logits [n, V], bf16 residual entering the final norm [n, d], bf16 final hidden state [n, d]) from one
    forward pass made exactly as the scorer makes it (self._model(input_ids=...).logits)."""
    import torch

    model = scorer._model
    norm = model.model.norm
    seen: dict[str, Any] = {}

    def hook(mod: Any, inp: Any, out: Any) -> None:
        seen["pre"] = inp[0].detach()
        seen["post"] = out.detach()

    h = norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=ids.to(next(model.parameters()).device)).logits
    finally:
        h.remove()
    return logits[0], seen["pre"][0], seen["post"][0]


class HighPrecisionHead:
    """The output layer and final norm in float64, for the unrounded logits."""

    def __init__(self, model: Any):
        import torch

        self.W = model.get_output_embeddings().weight.detach().to(torch.float64)
        norm = model.model.norm
        self.norm_w = norm.weight.detach().to(torch.float64)
        self.eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))

    def head(self, h: Any) -> Any:
        return h.to(self.W.dtype) @ self.W.T

    def norm(self, x: Any) -> Any:
        import torch

        x = x.to(torch.float64)
        return self.norm_w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def _row(logits_bf16: Any, pre: Any, post: Any, hp: HighPrecisionHead, pos: int, tids: list[int]) -> dict[str, Any]:
    import torch

    z = logits_bf16[pos].float()
    lp = torch.log_softmax(z, dim=-1)  # the scorer's reading: bf16 logits, log-softmax in float32
    z64a = hp.head(post[pos])
    z64b = hp.head(hp.norm(pre[pos]))
    lp64a, lp64b = torch.log_softmax(z64a, -1), torch.log_softmax(z64b, -1)
    rounded = z64a.to(torch.bfloat16).float()
    return {
        "tids": tids,
        "z_bf16": [z[t].item() for t in tids], "z_head64": [z64a[t].item() for t in tids],
        "z_norm64": [z64b[t].item() for t in tids],
        "lp_bf16": [lp[t].item() for t in tids], "lp_head64": [lp64a[t].item() for t in tids],
        "lp_norm64": [lp64b[t].item() for t in tids],
        "zmax_bf16": z.max().item(), "lse_bf16": torch.logsumexp(z, -1).item(),
        "lse_head64": torch.logsumexp(z64a, -1).item(),
        # does the bf16 logit row equal the float64 head rounded to bf16, over the whole vocabulary?
        "vocab_share_bf16_equals_rounded_head64": (rounded == z).float().mean().item(),
        "max_abs_hidden_post_vs_norm64": (post[pos].double() - hp.norm(pre[pos])).abs().max().item(),
    }


def score(args: argparse.Namespace) -> int:
    import torch

    from wf import families as F
    from wf import sizes

    tag = args.model
    if args.label == "old":  # as the earlier run loaded it (size_score / family_score: no strict load, default kwargs)
        spec = F.spec_for(tag)
        scorer = F.LogoddsScorer(spec.hf_id, model_tag=tag, attn_implementation=spec.attn_implementation,
                                 template_kwargs=spec.template_kwargs)
        messages = lambda t, a: sizes.arm_messages(t, a)  # noqa: E731
    else:  # as family_envcheck_<tag> loads it in serve2
        spec = F.envcheck_spec(tag)
        scorer = F.open_scorer(spec)
        messages = lambda t, a: F.family_messages(spec, t, a)  # noqa: E731
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r.get("transition_id"), r.get("condition"), r.get("candidate")))
    with out.open("a") as f:
        if ("meta", None, None) not in done:
            f.write(json.dumps({"transition_id": "meta", "condition": None, "candidate": None, "label": args.label,
                                "model": tag, "environment": _environment(scorer)}) + "\n")
            f.flush()
        hp = HighPrecisionHead(scorer._model)
        items = _items()
        yes, no = (scorer._encode_candidate(c)[0].tolist() for c in ("Yes", "No"))
        assert len(yes) == 1 and len(no) == 1
        for t in items:
            for arm in YN + PRED:
                msgs = messages(t, arm)
                if msgs is None:
                    continue
                prompt = scorer._encode(msgs)
                cands = [("yes_no", None)] if arm in YN else [("o_plus", t["candidates"]["o_plus"]),
                                                                ("o_minus", t["candidates"]["o_minus"])]
                for which, text in cands:
                    key = (t["id"], arm, which)
                    if key in done:
                        continue
                    if text is None:
                        ids, tids, positions = prompt, [yes[0], no[0]], [prompt.shape[1] - 1]
                    else:
                        cand = scorer._encode_candidate(text)
                        ids = torch.cat([prompt, cand], dim=1)
                        tids = cand[0].tolist()
                        positions = [prompt.shape[1] - 1 + i for i in range(len(tids))]
                    logits, pre, post = _hooked_forward(scorer, ids)
                    if text is None:
                        rows = [_row(logits, pre, post, hp, positions[0], tids)]
                    else:
                        rows = [_row(logits, pre, post, hp, p, [tid]) for p, tid in zip(positions, tids)]
                    rec = {"transition_id": t["id"], "condition": arm, "candidate": which, "label": args.label,
                           "n_ids": int(ids.shape[1]),
                           "ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
                           "rows": rows}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
    scorer.close()
    print(f"{args.label}: wrote {out}")
    return 0


# ---------------- compare (CPU) ----------------


def _read(path: Path) -> tuple[dict, dict[tuple[str, str, str], dict]]:
    meta, recs = {}, {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["transition_id"] == "meta":
            meta = r
        else:
            recs[(r["transition_id"], r["condition"], r["candidate"])] = r
    return meta, recs


def _archived(folder: Path, model: str) -> dict[tuple[str, str], float]:
    out = {}
    for p in sorted(folder.glob("shard_*/scoring.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["model"] == model:
                out[(r["transition_id"], r["condition"])] = r["L"]
    return out


def _L(recs: dict, tid: str, arm: str, field: str) -> float | None:
    """A score's L from the saved rows: yes/no from one row, two-candidate as the sum over each candidate's tokens."""
    if arm in YN:
        r = recs.get((tid, arm, "yes_no"))
        if r is None:
            return None
        row = r["rows"][0]
        if field == "lp_bf16":
            return row["lp_bf16"][0] - row["lp_bf16"][1]
        return row[field][0] - row[field][1]
    p, m = recs.get((tid, arm, "o_plus")), recs.get((tid, arm, "o_minus"))
    if p is None or m is None:
        return None
    return sum(x[field][0] for x in p["rows"]) - sum(x[field][0] for x in m["rows"])


def _summ(d: list[float]) -> dict[str, Any]:
    if not d:
        return {"n": 0}
    a = [abs(x) for x in d]
    return {"n": len(d), "mean_abs": statistics.fmean(a), "median_abs": statistics.median(a), "max_abs": max(a),
            "mean": statistics.fmean(d), "share_zero": sum(x == 0 for x in d) / len(d)}


def compare(args: argparse.Namespace) -> int:
    mo, old = _read(Path(args.old))
    mn, new = _read(Path(args.new))
    model = mo.get("model") or args.model
    arch_old = _archived(Path(args.archived_old), model)
    arch_new = _archived(Path(args.archived_new), model + "-serve2")
    eo, en = mo.get("environment", {}), mn.get("environment", {})
    settings = {k: (eo.get(k), en.get(k)) for k in sorted(set(eo) | set(en)) if k != "config"}
    co, cn = eo.get("config", {}), en.get("config", {})
    config_diff = {k: (co.get(k), cn.get(k)) for k in sorted(set(co) | set(cn)) if co.get(k) != cn.get(k)}
    keys = sorted({(k[0], k[1]) for k in old} & {(k[0], k[1]) for k in new})
    report: dict[str, Any] = {"model": model, "settings_old_vs_new": settings, "config_keys_differing": config_diff}

    # tokens identical between environments
    same_ids = [old[k]["ids_sha256"] == new[k]["ids_sha256"] for k in old if k in new]
    report["ids_identical"] = {"n": len(same_ids), "identical": sum(same_ids)}

    # each environment's bf16 L reproduces its archived score exactly
    repro = {}
    for label, recs, arch in (("old", old, arch_old), ("new", new, arch_new)):
        eq, n, dmax = 0, 0, 0.0
        for tid, arm in keys:
            L = _L(recs, tid, arm, "lp_bf16")
            if L is None or (tid, arm) not in arch:
                continue
            n += 1
            eq += abs(L - arch[(tid, arm)]) <= 1e-5
            dmax = max(dmax, abs(L - arch[(tid, arm)]))
        repro[label] = {"n": n, "equal_to_archived": eq, "max_abs_diff": dmax}
    report["reproduces_archived"] = repro

    for fmt, arms in (("yes_no", YN), ("two_candidate", PRED)):
        block: dict[str, Any] = {}
        for field in ("lp_bf16", "lp_head64", "lp_norm64"):
            d = {a: [] for a in arms}
            for tid, arm in keys:
                if arm not in arms:
                    continue
                a, b = _L(old, tid, arm, field), _L(new, tid, arm, field)
                if a is not None and b is not None:
                    d[arm].append(b - a)
            alld = [x for v in d.values() for x in v]
            block[field] = {**_summ(alld), "by_condition": {a: _summ(v) for a, v in d.items()}}
        # magnitudes of the logits read and their bf16 spacing (yes/no: the Yes and No logits; two-candidate: each
        # candidate token's logit), and the rounding account
        mags, spac, zmax = [], Counter(), []
        explained, n_changed, n_logits, n_logit_changed, n_logit_rounding = 0, 0, 0, 0, 0
        within_one_step, head64_logit_diff = 0, []
        for k in old:
            if k not in new or k[1] not in arms:
                continue
            for ro, rn in zip(old[k]["rows"], new[k]["rows"]):
                zmax.append(ro["zmax_bf16"])
                for zo, zn, ho, hn in zip(ro["z_bf16"], rn["z_bf16"], ro["z_head64"], rn["z_head64"]):
                    mags.append(abs(zo))
                    spac[bf16_spacing(zo)] += 1
                    n_logits += 1
                    head64_logit_diff.append(hn - ho)
                    within_one_step += abs(hn - ho) < bf16_spacing(max(abs(zo), abs(zn)))
                    if zo != zn:
                        n_logit_changed += 1
                        # the change is rounding: each side's bf16 logit is its own float64 logit rounded to bf16
                        n_logit_rounding += (round_bf16(ho) == zo and round_bf16(hn) == zn)
        for tid, arm in keys:
            if arm not in arms:
                continue
            a, b = _L(old, tid, arm, "lp_bf16"), _L(new, tid, arm, "lp_bf16")
            if a is None or b is None or a == b:
                continue
            n_changed += 1
            # predicted bf16 L difference from rounding the float64 logits alone (yes/no only: L = z_yes - z_no)
            if arm in YN:
                ro, rn = old[(tid, arm, "yes_no")]["rows"][0], new[(tid, arm, "yes_no")]["rows"][0]
                pred = ((round_bf16(rn["z_head64"][0]) - round_bf16(rn["z_head64"][1]))
                        - (round_bf16(ro["z_head64"][0]) - round_bf16(ro["z_head64"][1])))
                explained += abs(pred - (b - a)) < 1e-3
        block["logits"] = {
            "n": n_logits, "abs_quantiles": _quantiles(mags), "spacing_counts": {str(s): c for s, c in sorted(spac.items())},
            "max_logit_quantiles": _quantiles(zmax),
            "n_bf16_logit_changed": n_logit_changed, "n_changed_equal_to_rounded_head64_both_sides": n_logit_rounding,
            "share_head64_change_under_one_bf16_step": within_one_step / n_logits if n_logits else None,
            "head64_logit_change": _summ(head64_logit_diff),
        }
        block["n_scores_changed_bf16"] = n_changed
        if fmt == "yes_no":
            block["n_changed_exactly_predicted_by_rounding_head64"] = explained
        report[fmt] = block
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def _quantiles(x: list[float]) -> dict[str, float]:
    if not x:
        return {}
    s = sorted(x)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    return {"min": s[0], "p05": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": s[-1]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--label", choices=("old", "new"), required=True)
    s.add_argument("--model", default="qwen3-8b")
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--old", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--model", default="qwen3-8b")
    c.add_argument("--archived-old", required=True)
    c.add_argument("--archived-new", required=True)
    c.add_argument("--out")
    args = ap.parse_args(argv)
    return score(args) if args.cmd == "score" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
