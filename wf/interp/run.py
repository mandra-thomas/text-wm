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

from wf.interp.hooks import capture
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
        if data:  # index rows without activations (save_acts=False) need no chunk file
            tmp = self.d / f"chunk_{k:05d}.tmp.npz"
            np.savez(tmp, **data)
            os.replace(tmp, self.d / f"chunk_{k:05d}.npz")
        append_jsonl(self.index, [{**r, "chunk": k, "row": i} for i, (r, _) in enumerate(self._buf)])
        self.n_chunks += 1
        self._buf = []

    def load(self, name: str, keys: list[str] | None = None, sel: Any = None) -> tuple[list[dict], Any]:
        """(index rows, stacked array) for array `name`, in index order or in the order of `keys`.

        Streams: the output is allocated once (float16, as stored) and filled one chunk at a time, so memory
        holds the result and a single chunk, never every chunk. `sel` indexes each record's array before it is
        kept (for example `sel=5` keeps layer 5 of "last", `sel=(slice(None), 0)` the first probe position at
        every probe layer), which is how an analysis reads one layer of a large store.
        """
        import numpy as np

        rows = read_jsonl(self.index) if self.index.exists() else []
        if keys is not None:
            by = {r["key"]: r for r in rows}
            missing = [k for k in keys if k not in by]
            if missing:
                raise KeyError(f"{len(missing)} key(s) not in the store, e.g. {missing[0]!r}")
            rows = [by[k] for k in keys]
        if not rows:
            return rows, np.zeros((0,), dtype=np.float16)
        by_chunk: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_chunk[r["chunk"]].append(i)
        out = None
        for chunk in sorted(by_chunk):
            with np.load(self.d / f"chunk_{chunk:05d}.npz") as z:
                arr = z[name]
            for i in by_chunk[chunk]:
                x = arr[rows[i]["row"]]
                x = x if sel is None else x[sel]
                if out is None:
                    out = np.empty((len(rows),) + np.shape(x), dtype=arr.dtype)
                out[i] = x
            del arr
        return rows, out

    def records(self) -> list[dict]:
        return read_jsonl(self.index) if self.index.exists() else []


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
             chunk: int = 64, min_free_gb: float = MIN_FREE_GB, save_acts: bool = True) -> ActStore:
    """Readouts, the logit lens, and fp16 activations for every (item, format, goal).

    Saved per prompt: "last" (layers, width), the residual at the answer position after every layer; "probe"
    (probe layers, 2, width), the residual at the action line's last token and its receptacle token. The jsonl
    index holds the readout, the two readout tokens' probability mass, the lens values and the token count.
    With `save_acts` False only the index rows are kept (the first-token prompts of the minimal pilot, which
    need their readout for the reproduction gate and no residual; amendment 2, B6). Every call flushes its own
    chunks, so a chunk never mixes prompts with and without activations. Resumes from whatever the index holds.
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
                try:
                    e = encode(tok, messages(t, fmt, goal, po))
                except ValueError:
                    if goal in ("cf_aligned", "cf_swapped", "aligned_plus", "swapped_plus"):
                        continue  # not every item has a counterfactual task or a mention line
                    raise
                pos = action_positions(e, t)
                logits, acts, h = capture(p, _ids(p, e), [("resid", layer) for layer in layers],
                                          positions=[pos["answer"], pos["action_end"], pos["action_rec"]],
                                          with_hidden=True)
                last = torch.stack([acts[("resid", layer)][0, 0] for layer in layers])
                probe = torch.stack([acts[("resid", layer)][0, 1:3] for layer in pl])
                lp = torch.log_softmax(logits[0], dim=-1)
                rec = {"key": key, "id": t["id"], "game_id": t["game_id"], "fmt": fmt, "goal": goal,
                       "d": float(p.readout(h, plus, minus)[0]),  # float32, from the final normed hidden state
                       "d_logits": float(logits[0, plus] - logits[0, minus]),  # the model's own (bf16) logits
                       "mass": float(lp[plus].exp() + lp[minus].exp()), "n_tokens": len(e),
                       "lens": logit_lens(p, last, plus, minus), "probe_layers": pl,
                       "history_len": t.get("history_len"), "determination_reason": t.get("determination_reason")}
                store.add(rec, {"last": last.float().cpu().numpy(), "probe": probe.float().cpu().numpy()}
                          if save_acts else {})
    store.flush()
    return store


# ---------------- item selection ----------------


def _stratum(t: dict) -> tuple[str, bool]:
    return (t.get("determination_reason") or "?", (t.get("history_len") or 0) == 0)


def weighting_stratum(t: dict, mention: bool = False) -> str:
    """The stratum an item's weight is computed in (amendment 2, B4): "step0" (no action yet), and after step 0
    "later_mention" (the history after the task line names the action's object; only when `mention` is on) or
    "later"."""
    from wf.interp.prompts import mentions_action_object

    if (t.get("history_len") or 0) == 0:
        return "step0"
    return "later_mention" if mention and mentions_action_object(t) else "later"


def _proportional(sizes: dict, n: int) -> dict:
    """Largest-remainder allocation of `n` proportional to `sizes`, never above a stratum's size."""
    total = sum(sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in sizes}
    quotas = {k: n * v / total for k, v in sizes.items()}
    take = {k: min(sizes[k], int(q)) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - take[k]), str(k))):
        if sum(take.values()) >= n:
            break
        if take[k] < sizes[k]:
            take[k] += 1
    return take


def _allocate_with_minimums(sizes: dict[str, int], n: int, mins: dict[str, int]) -> dict[str, int]:
    """Proportional allocation in which a stratum with a minimum gets at least min(minimum, its size); the other
    strata share what is left in proportion to their sizes. Repeats until no further minimum binds."""
    fixed: dict[str, int] = {}
    while True:
        rest = {k: v for k, v in sizes.items() if k not in fixed}
        alloc = {**fixed, **_proportional(rest, n - sum(fixed.values()))}
        binding = {k: min(mins[k], sizes[k]) for k in rest if mins.get(k, 0) and alloc[k] < min(mins[k], sizes[k])}
        if not binding:
            return alloc
        fixed.update(binding)


def select_stratified(items: list[dict], effect: dict[str, float], n: int, min_effect: float, games: set[str],
                      seed_text: str, min_step0: int = 0, min_mention: int = 0) -> tuple[list[dict], dict[str, float]]:
    """Up to `n` items of `games` with effect >= `min_effect`, and each chosen item's weight (design §3,
    amendment 2, B4).

    Items are first split into weighting strata (`weighting_stratum`: step 0; later; and, with `min_mention`,
    later items whose history names the action's object). The step-0 stratum gets at least `min_step0` items and
    the mention stratum at least `min_mention` (or all there are); the rest of `n` is shared in proportion to
    size. Within a weighting stratum, allocation over failure reasons is proportional with largest remainders,
    and the draw is a seeded shuffle. An item's weight is its stratum's share of the eligible items over its
    share of the chosen ones, so weighted pooled estimates describe the eligible population, not the
    oversampled mix.
    """
    eligible = [t for t in items if t["game_id"] in games and effect.get(t["id"], float("-inf")) >= min_effect]
    if len(eligible) <= n:
        return sorted(eligible, key=lambda x: x["id"]), {t["id"]: 1.0 for t in eligible}
    use_mention = min_mention > 0
    strata: dict[str, list[dict]] = defaultdict(list)
    for t in sorted(eligible, key=lambda x: x["id"]):
        strata[weighting_stratum(t, use_mention)].append(t)
    alloc = _allocate_with_minimums({k: len(v) for k, v in strata.items()}, n,
                                    {"step0": min_step0, "later_mention": min_mention})
    rng = random.Random(seed_text)
    chosen: list[dict] = []
    got_by: dict[str, list[dict]] = {}
    for w in sorted(strata):
        by: dict = defaultdict(list)
        for t in strata[w]:
            by[_stratum(t)].append(t)
        take = _proportional({k: len(v) for k, v in by.items()}, alloc[w])
        got = []
        for k in sorted(by, key=str):
            pool = list(by[k])
            rng.shuffle(pool)
            got += pool[: take[k]]
        chosen += got
        got_by[w] = got
    weights = {t["id"]: (len(strata[w]) / len(eligible)) / (len(got) / len(chosen))
               for w, got in got_by.items() for t in got}
    return sorted(chosen, key=lambda x: x["id"]), weights


def select_patch_items(items: list[dict], effect: dict[str, float], n: int, min_effect: float, games: set[str],
                       seed_text: str, min_step0: int = 0, min_mention: int = 0) -> list[dict]:
    """`select_stratified` without the weights."""
    return select_stratified(items, effect, n, min_effect, games, seed_text, min_step0, min_mention)[0]


# ---------------- stage 2 and 3: patching ----------------


def batch_noise(p: Parts, ids: list[int], plus: int, minus: int, batch_size: int = 16) -> float:
    """The largest change in one prompt's readout across batch rows and batch sizes (1, 2 and `batch_size`).

    On a CPU in float32 or bfloat16 this is usually exactly 0; on a GPU bfloat16 kernels can depend on the
    batch's shape. It sets the tolerance of the identical-prefix check (`prefix_tolerance`).
    """
    import torch

    from wf.interp.hooks import readout_rows

    vals = []
    for n in sorted({1, 2, batch_size}):
        vals += readout_rows(p, torch.tensor([ids] * n, device=p.device), plus, minus)
    return max(vals) - min(vals)


PREFIX_TOL_FLOOR = 0.01  # nats; a patch at the wrong positions changes the readout by far more
PREFIX_TOL_NOISE = 5.0  # times the measured batch noise


def prefix_tolerance(noise: float) -> float:
    return max(PREFIX_TOL_FLOOR, PREFIX_TOL_NOISE * noise)


def check_prefix_noop(p: Parts, pr: Any, plus: int, minus: int, tol: float | None = None,
                      batch_size: int = 16) -> float:
    """Patching prefix positions (identical in both prompts) must not change the readout. Returns the change.

    With `tol` None the tolerance is set from the batch noise measured on the target prompt here
    (`prefix_tolerance(batch_noise(...))`), so a GPU's bfloat16 noise cannot fail a correct job and a real
    mistake (positions off, the wrong row) is still far above it. The source activations come from a run of the
    same batch size as the patched run, as in `sweep`.
    """
    import torch

    from wf.interp.hooks import Patch, readout_rows

    if pr.prefix == 0:
        return 0.0
    if tol is None:
        tol = prefix_tolerance(batch_noise(p, pr.a.ids, plus, minus, batch_size))
    mid = p.n_layers // 2
    pos = list(range(pr.prefix))
    _, src = capture(p, torch.tensor([pr.b.ids] * 2, device=p.device), [("resid", mid)], rows=1)
    ids = torch.tensor([pr.a.ids] * 2, device=p.device)
    d = readout_rows(p, ids, plus, minus,
                     [Patch("resid", mid, 1, torch.tensor(pos, device=p.device), src[("resid", mid)][0, pos])])
    change = d[1] - d[0]
    if abs(change) > tol:
        raise RuntimeError(f"patching identical prefix positions changed the readout by {change} (tolerance {tol})")
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
                check_prefix_noop(p, pr, plus, minus, batch_size=batch_size)
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
               positions: str = "last", batch_size: int = 16,
               checks: dict[str, list[tuple[int, Any, float, str]]] | None = None) -> Path:
    """Readouts with each named edit applied, one batch row per edit plus an unedited row in every batch.

    `edits[name]` is a list of (layer, fn_factory) where fn_factory(positions) returns an edit for the residual
    after that layer (for example `lambda pos: direction_edit(u, pos, value=c)`). `positions` is where the edit
    applies and must be where its direction was estimated (amendment 1, A11): "last" (the answer position, the
    default; the task direction is estimated there), "final" (the template tail; only for directions estimated
    per position) or "all". Batches are padded to `batch_size` rows and each edit's readout is paired with the
    unedited row of its own batch (`edit_clean`). Readouts are float32. One record per (item, goal), resumable.

    `checks[name]` lists (layer, unit vector, target, kind) for edits along a direction (amendment 2, B9). On the
    first prompt this call edits, the residual at the answer position after each such layer is read after the
    edit, in the model's own dtype, and the record gets `edit_realized[name]`: for kind "set" the realized
    u.h - target (0 up to the rounding of the cast back to bfloat16), for kind "add" the realized u.h minus the
    unedited row's u.h minus the amount added.
    """
    import torch

    from wf.interp.hooks import readout_rows

    if positions not in ("last", "final", "all"):
        raise ValueError(f"unknown positions {positions!r}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(out)
    done = {(r["id"], r["fmt"], r["goal"]) for r in (read_jsonl(out) if out.exists() else [])}
    plus, minus = readout_ids(tok, fmt)
    names = sorted(edits)
    logged = not checks
    for t in items:
        for goal in goals:
            if (t["id"], fmt, goal) in done or (goal == "swapped" and not t.get("swapped_task_line")):
                continue
            check_disk(out.parent)
            e = encode(tok, messages(t, fmt, goal))
            pos = {"last": [len(e) - 1], "final": final_positions(e), "all": None}[positions]
            result: dict[str, float] = {}
            edit_clean: dict[str, float] = {}
            realized: dict[str, list[float]] = {}
            clean = []
            for k in range(0, max(1, len(names)), max(1, batch_size - 1)):
                chunk = names[k:k + batch_size - 1]
                by_layer: dict[int, dict[int, Callable]] = defaultdict(dict)
                for r, name in enumerate(chunk, start=1):
                    for layer, factory in edits[name]:
                        by_layer[layer][r] = factory(pos)
                hook_edits = [("resid", layer, _row_edit(rows)) for layer, rows in by_layer.items()]
                seen: dict[int, Any] = {}
                if not logged:  # read after the edits: hooks on one module run in the order they were added
                    for layer in sorted({c[0] for name in chunk for c in (checks or {}).get(name, [])}):
                        def keep(h, layer=layer):
                            seen[layer] = h[:, -1].detach().clone()
                            return h
                        hook_edits.append(("resid", layer, keep))
                ids = torch.tensor([e.ids] * batch_size, device=p.device)
                d = readout_rows(p, ids, plus, minus, edits=hook_edits)
                clean.append(d[0])
                result.update(dict(zip(chunk, d[1:])))
                edit_clean.update({name: d[0] for name in chunk})
                for r, name in enumerate(chunk, start=1):
                    for layer, u, target, kind in ([] if logged else (checks or {}).get(name, [])):
                        uu = torch.as_tensor(u, dtype=torch.float32, device=seen[layer].device)
                        got = float(seen[layer][r].float() @ uu)
                        base = float(seen[layer][0].float() @ uu) if kind == "add" else 0.0
                        realized.setdefault(name, []).append(got - base - float(target))
            rec = {"id": t["id"], "game_id": t["game_id"], "fmt": fmt, "goal": goal, "d": clean[0],
                   "clean_spread": max(clean) - min(clean), "edits": result, "edit_clean": edit_clean,
                   "positions": positions, "history_len": t.get("history_len"),
                   "determination_reason": t.get("determination_reason")}
            if not logged:
                rec["edit_realized"] = realized
                logged = True
            append_jsonl(out, [rec])
    return out


def write_manifest(out: Path, info: dict) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "manifest.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
