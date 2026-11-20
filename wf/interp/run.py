"""Stages of the interpretability pilot (design §8). Each saves its work per item as it goes and resumes.

The stage functions take a loaded model and tokenizer, so they run unchanged on a tiny random model in the
tests. Run folders, provenance and job scripts are added when the stages are wired into `wf.runs`.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from wf.interp.hooks import capture, forward_last_logits, run_patched
from wf.interp.lens import logit_lens
from wf.interp.model import Parts
from wf.interp.patching import Cell, sweep
from wf.interp.prompts import Encoded, action_positions, build_pair, encode, messages, readout_ids
from wf.util import append_jsonl, read_jsonl, repair_jsonl

MIN_FREE_GB = 5.0


def check_disk(path: Path, min_free_gb: float = MIN_FREE_GB) -> None:
    free = shutil.disk_usage(path).free / 1e9
    if free < min_free_gb:
        raise SystemExit(f"only {free:.1f} GB free at {path}; stopping before writing (need {min_free_gb} GB)")


def probe_layers(n_layers: int) -> list[int]:
    """Every fourth layer and the last (design §6)."""
    return sorted(set(range(0, n_layers, 4)) | {n_layers - 1})


class ActStore:
    """fp16 activations in numbered .npz chunks, with the records of each chunk appended to a jsonl index only
    after the chunk is on disk, so the index never names an activation that was not saved."""

    def __init__(self, d: Path, chunk: int = 64, min_free_gb: float = MIN_FREE_GB):
        self.d = Path(d)
        self.d.mkdir(parents=True, exist_ok=True)
        self.index = self.d / "index.jsonl"
        repair_jsonl(self.index)
        self.chunk = chunk
        self.min_free_gb = min_free_gb
        self._buf: list[tuple[dict, dict]] = []
        rows = read_jsonl(self.index) if self.index.exists() else []
        self.n_chunks = 1 + max((r["chunk"] for r in rows), default=-1)

    def done(self) -> set[str]:
        return {r["key"] for r in read_jsonl(self.index)} if self.index.exists() else set()

    def add(self, record: dict, arrays: dict[str, Any]) -> None:
        self._buf.append((record, arrays))
        if len(self._buf) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        import numpy as np

        if not self._buf:
            return
        check_disk(self.d, self.min_free_gb)
        k = self.n_chunks
        names = sorted(self._buf[0][1])
        data = {n: np.stack([np.asarray(a[n], dtype=np.float16) for _, a in self._buf]) for n in names}
        for n, arr in data.items():  # residuals with very large entries would overflow fp16 silently
            if not np.isfinite(arr).all():
                raise ValueError(f"array {n!r} does not fit in fp16 (largest entries overflow); store it as float32")
        tmp = self.d / f"chunk_{k:05d}.tmp.npz"
        np.savez(tmp, **data)
        os.replace(tmp, self.d / f"chunk_{k:05d}.npz")
        append_jsonl(self.index, [{**r, "chunk": k, "row": i} for i, (r, _) in enumerate(self._buf)])
        self.n_chunks += 1
        self._buf = []

    def load(self, name: str, keys: list[str] | None = None) -> tuple[list[dict], Any]:
        """(index rows, stacked array) for array `name`, in index order or in the order of `keys`."""
        import numpy as np

        rows = read_jsonl(self.index)
        if keys is not None:
            by = {r["key"]: r for r in rows}
            rows = [by[k] for k in keys]
        cache: dict[int, Any] = {}
        out = []
        for r in rows:
            if r["chunk"] not in cache:
                cache[r["chunk"]] = np.load(self.d / f"chunk_{r['chunk']:05d}.npz")[name]
            out.append(cache[r["chunk"]][r["row"]])
        return rows, (np.stack(out) if out else np.zeros((0,)))


def _ids(p: Parts, e: Encoded) -> Any:
    import torch

    return torch.tensor([e.ids], dtype=torch.long, device=p.device)


def final_positions(e: Encoded) -> list[int]:
    """Positions from the end of the final user message to the end (the template's tail, answer position last)."""
    start = next((i for i in range(len(e)) if e.offsets[i][1] > e.offsets[i][0] and e.offsets[i][0] >= e.user_end),
                 len(e) - 1)
    return list(range(start, len(e)))


# ---------------- stage 0 and 1: readouts and activations ----------------


def baseline(p: Parts, tok: Any, items: list[dict], out: Path, formats: Iterable[str] = ("yn", "obs"),
             goals: Iterable[str] = ("aligned", "swapped", "absent"), placebo: dict[str, str] | None = None,
             chunk: int = 64, min_free_gb: float = MIN_FREE_GB) -> ActStore:
    """Readouts, the logit lens, and fp16 activations for every (item, format, goal).

    Saved per prompt: "last" (layers, width), the residual at the answer position after every layer; "probe"
    (probe layers, 2, width), the residual at the action line's last token and its receptacle token. The jsonl
    index holds the readout, the two readout tokens' probability mass, the lens values and the token count.
    Resumes from whatever the index already holds.
    """
    import torch

    store = ActStore(Path(out), chunk=chunk, min_free_gb=min_free_gb)
    done = store.done()
    layers = list(range(p.n_layers))
    pl = probe_layers(p.n_layers)
    for t in items:
        for fmt in formats:
            plus, minus = readout_ids(tok, fmt)
            for goal in goals:
                po = (placebo or {}).get(t["id"]) if goal == "placebo" else None
                if goal == "placebo" and not po:
                    continue
                if goal == "swapped" and not t.get("swapped_task_line"):
                    continue
                key = f"{t['id']}|{fmt}|{goal}"
                if key in done:
                    continue
                e = encode(tok, messages(t, fmt, goal, po))
                pos = action_positions(e, t)
                logits, acts = capture(p, _ids(p, e), [("resid", layer) for layer in layers],
                                       positions=[pos["answer"], pos["action_end"], pos["action_rec"]])
                last = torch.stack([acts[("resid", layer)][0, 0] for layer in layers])
                probe = torch.stack([acts[("resid", layer)][0, 1:3] for layer in pl])
                lp = torch.log_softmax(logits[0], dim=-1)
                rec = {"key": key, "id": t["id"], "game_id": t["game_id"], "fmt": fmt, "goal": goal,
                       "d": float(logits[0, plus] - logits[0, minus]),
                       "mass": float(lp[plus].exp() + lp[minus].exp()), "n_tokens": len(e),
                       "lens": logit_lens(p, last, plus, minus), "probe_layers": pl}
                store.add(rec, {"last": last.float().cpu().numpy(), "probe": probe.float().cpu().numpy()})
    store.flush()
    return store


# ---------------- item selection ----------------


def _stratum(t: dict) -> tuple[str, bool]:
    return (t.get("determination_reason") or "?", (t.get("history_len") or 0) == 0)


def select_patch_items(items: list[dict], effect: dict[str, float], n: int, min_effect: float, games: set[str],
                       seed_text: str) -> list[dict]:
    """Up to `n` items of `games` with effect >= `min_effect`, stratified by failure reason and step 0 (design §3).

    Allocation is proportional with largest remainders; within a stratum the draw is a seeded shuffle.
    """
    eligible = [t for t in items if t["game_id"] in games and effect.get(t["id"], float("-inf")) >= min_effect]
    by: dict = defaultdict(list)
    for t in sorted(eligible, key=lambda x: x["id"]):
        by[_stratum(t)].append(t)
    if len(eligible) <= n:
        return sorted(eligible, key=lambda x: x["id"])
    quotas = {k: n * len(v) / len(eligible) for k, v in by.items()}
    take = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - take[k]), str(k)))[: n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed_text)
    chosen = []
    for k in sorted(by, key=str):
        pool = list(by[k])
        rng.shuffle(pool)
        chosen += pool[: take[k]]
    return sorted(chosen, key=lambda x: x["id"])


# ---------------- stage 2 and 3: patching ----------------


def check_prefix_noop(p: Parts, pr: Any, plus: int, minus: int, tol: float = 1e-3) -> float:
    """Patching prefix positions (identical in both prompts) must not change the readout. Returns the change."""
    import torch

    from wf.interp.hooks import Patch

    if pr.prefix == 0:
        return 0.0
    mid = p.n_layers // 2
    pos = list(range(pr.prefix))
    _, src = capture(p, torch.tensor([pr.b.ids], device=p.device), [("resid", mid)])
    ids = torch.tensor([pr.a.ids] * 2, device=p.device)
    logits = run_patched(p, ids, [Patch("resid", mid, 1, torch.tensor(pos, device=p.device), src[("resid", mid)][0, pos])])
    change = float((logits[1, plus] - logits[1, minus]) - (logits[0, plus] - logits[0, minus]))
    if abs(change) > tol:
        raise RuntimeError(f"patching identical prefix positions changed the readout by {change}")
    return change


def patch_items(p: Parts, tok: Any, items: list[dict], out: Path, cells: list[Cell], fmt: str = "yn",
                directions: tuple[str, ...] = ("noise", "denoise"), source_goal: str = "swapped",
                placebo: dict[str, str] | None = None, batch_size: int = 16) -> Path:
    """One record per (item, direction) in `out` (jsonl), from `sweep`. Resumes by (id, fmt, direction, source)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(out)
    done = {(r["id"], r["fmt"], r["direction"], r["source_goal"]) for r in (read_jsonl(out) if out.exists() else [])}
    plus, minus = readout_ids(tok, fmt)
    checked = False
    for t in items:
        po = (placebo or {}).get(t["id"])
        if source_goal == "placebo" and not po:
            continue
        # Placebo pairs compare two non-goal objects: the swapped prompt is the target, the placebo the source.
        goal_a = "swapped" if source_goal == "placebo" else "aligned"
        pr = None
        for direction in directions:
            if (t["id"], fmt, direction, source_goal) in done:
                continue
            check_disk(out.parent)
            pr = pr or build_pair(tok, t, fmt, goal_a, source_goal, po)
            if not checked:
                check_prefix_noop(p, pr, plus, minus)
                checked = True
            res = sweep(p, pr if direction == "noise" else pr.flipped(), cells, plus, minus, batch_size)
            append_jsonl(out, [{"id": t["id"], "game_id": t["game_id"], "fmt": fmt, "direction": direction,
                                "source_goal": source_goal, "history_len": t.get("history_len"),
                                "determination_reason": t.get("determination_reason"), **res}])
    return out


# ---------------- stage 4: direction edits ----------------


def _row_edit(rows: dict[int, Callable]) -> Callable:
    def fn(h):
        h = h.clone()
        for r, f in rows.items():
            h[r:r + 1] = f(h[r:r + 1])
        return h
    return fn


def edit_items(p: Parts, tok: Any, items: list[dict], out: Path, edits: dict[str, list[tuple[int, Callable]]],
               fmt: str = "yn", goals: Iterable[str] = ("aligned", "swapped", "absent"),
               positions: str = "final", batch_size: int = 16) -> Path:
    """Readouts with each named edit applied, one batch row per edit plus an unedited row.

    `edits[name]` is a list of (layer, fn_factory) where fn_factory(positions) returns an edit for the residual
    after that layer (for example `lambda pos: direction_edit(u, pos, value=c)`); `positions` is "final" (the
    template tail, design §7) or "all". One record per (item, goal), resumable.
    """
    import torch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(out)
    done = {(r["id"], r["fmt"], r["goal"]) for r in (read_jsonl(out) if out.exists() else [])}
    plus, minus = readout_ids(tok, fmt)
    names = sorted(edits)
    for t in items:
        for goal in goals:
            if (t["id"], fmt, goal) in done or (goal == "swapped" and not t.get("swapped_task_line")):
                continue
            check_disk(out.parent)
            e = encode(tok, messages(t, fmt, goal))
            pos = final_positions(e) if positions == "final" else None
            result: dict[str, float] = {}
            clean = []
            for k in range(0, len(names), max(1, batch_size - 1)):
                chunk = names[k:k + batch_size - 1]
                by_layer: dict[int, dict[int, Callable]] = defaultdict(dict)
                for r, name in enumerate(chunk, start=1):
                    for layer, factory in edits[name]:
                        by_layer[layer][r] = factory(pos)
                hook_edits = [("resid", layer, _row_edit(rows)) for layer, rows in by_layer.items()]
                ids = torch.tensor([e.ids] * (len(chunk) + 1), device=p.device)
                logits = run_patched(p, ids, [], edits=hook_edits)
                d = (logits[:, plus] - logits[:, minus]).tolist()
                clean.append(d[0])
                result.update(dict(zip(chunk, d[1:])))
            if not clean:
                clean.append(float((lambda lg: lg[0, plus] - lg[0, minus])(forward_last_logits(p, _ids(p, e)))))
            append_jsonl(out, [{"id": t["id"], "game_id": t["game_id"], "fmt": fmt, "goal": goal, "d": clean[0],
                                "clean_spread": max(clean) - min(clean), "edits": result}])
    return out


def write_manifest(out: Path, info: dict) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "manifest.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
