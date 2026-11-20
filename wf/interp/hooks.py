"""Forward hooks: capture, patch (per batch row), patch one attention head, edit along a direction.

Sites, at decoder layer `layer`:

- "resid": the residual stream after the layer (the decoder layer's output); layer -1 is the embedding output.
- "attn": the attention block's output, before it is added to the residual stream.
- "mlp": the MLP block's output, before it is added to the residual stream.
- "head_in": the input of the attention output projection, i.e. every head's output side by side
  (n_heads * head_dim); a head's slice is its contribution before the projection mixes heads.

Every hook is registered inside `HookSet`, which removes all of them on exit, also after an exception, so a
failed patch can never leak into the next forward pass.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from wf.interp.model import Parts

SITES = ("resid", "attn", "mlp", "head_in")


def module_for(p: Parts, site: str, layer: int) -> Any:
    if site == "resid":
        return p.embed if layer == -1 else p.layers[layer]
    if layer < 0:
        raise ValueError(f"site {site} has no layer {layer}")
    if site == "attn":
        return p.attn(layer)
    if site == "mlp":
        return p.mlp(layer)
    if site == "head_in":
        return p.o_proj(layer)
    raise ValueError(f"unknown site {site!r}")


def _first(out: Any) -> Any:
    return out[0] if isinstance(out, tuple) else out


def _replace(out: Any, new: Any) -> Any:
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class HookSet:
    """Registers hooks and removes every one of them on exit."""

    def __init__(self) -> None:
        self.handles: list = []

    def __enter__(self) -> "HookSet":
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def edit(self, p: Parts, site: str, layer: int, fn: Callable[[Any], Any]) -> None:
        """Replace the site's activation `h` (batch, positions, width) with `fn(h)`."""
        mod = module_for(p, site, layer)
        if site == "head_in":
            def pre(_m, args, kwargs):
                return (fn(args[0]),) + tuple(args[1:]), kwargs
            self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        else:
            def post(_m, _args, out):
                return _replace(out, fn(_first(out)))
            self.handles.append(mod.register_forward_hook(post))

    def read(self, p: Parts, site: str, layer: int, sink: Callable[[Any], None]) -> None:
        """Pass the site's activation to `sink` without changing it."""
        def keep(h):
            sink(h)
            return h
        self.edit(p, site, layer, keep)


def n_hooks(model: Any) -> int:
    """Hooks currently registered anywhere in the model (tests use it to prove removal)."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def forward_last_logits(p: Parts, input_ids: Any) -> Any:
    """Logits at the last position only, float32, shape (batch, vocab)."""
    import torch

    with torch.no_grad():
        out = p.model(input_ids=input_ids.to(p.device), logits_to_keep=1, use_cache=False)
    return out.logits[:, -1].float()


def capture(p: Parts, input_ids: Any, sites: list[tuple[str, int]], positions: Any = None) -> tuple[Any, dict]:
    """(last-position logits, {(site, layer): activation}) for one forward pass.

    Activations are detached copies, shape (batch, positions, width), on the model's device. `positions`
    (a list or 1-d tensor) keeps only those positions, to bound memory.
    """
    acts: dict = {}
    with HookSet() as hs:
        for site, layer in sites:
            def sink(h, key=(site, layer)):
                acts[key] = (h if positions is None else h[:, positions]).detach().clone()
            hs.read(p, site, layer, sink)
        logits = forward_last_logits(p, input_ids)
    return logits, acts


@dataclass
class Patch:
    """Overwrite `site` at `layer` in batch row `row`, at target positions `tgt`, with `value` (len(tgt), width).

    With `head` set (site "head_in" only), only that head's slice is overwritten and `value` is (len(tgt), head_dim).
    """

    site: str
    layer: int
    row: int
    tgt: Any
    value: Any
    head: int | None = None


def _patch_fn(patches: list[Patch], head_dim: int) -> Callable[[Any], Any]:
    def fn(h):
        h = h.clone()
        for pt in patches:
            v = pt.value.to(device=h.device, dtype=h.dtype)
            if pt.head is None:
                h[pt.row, pt.tgt] = v
            else:
                lo = pt.head * head_dim
                h[pt.row, pt.tgt, lo:lo + head_dim] = v
        return h
    return fn


def run_patched(p: Parts, input_ids: Any, patches: list[Patch], edits: list[tuple[str, int, Callable]] = ()) -> Any:
    """Last-position logits with every patch applied (one hook per site and layer covers all rows).

    `edits` are extra (site, layer, fn) edits, applied after the patches at the same site.
    """
    by_site: dict = defaultdict(list)
    for pt in patches:
        if pt.head is not None and pt.site != "head_in":
            raise ValueError("a head patch needs site 'head_in'")
        by_site[(pt.site, pt.layer)].append(pt)
    with HookSet() as hs:
        for (site, layer), group in by_site.items():
            hs.edit(p, site, layer, _patch_fn(group, p.head_dim))
        for site, layer, fn in edits:
            hs.edit(p, site, layer, fn)
        return forward_last_logits(p, input_ids)


def direction_edit(u: Any, positions: Any = None, value: Any = None, add: float | None = None) -> Callable[[Any], Any]:
    """An edit along unit vector `u`: set the component to `value` (a mean ablation; 0 if None), or add `add * u`.

    Applied at `positions` (all positions if None), in float32, then cast back.
    """
    def fn(h):
        import torch

        uu = u.to(device=h.device, dtype=torch.float32)
        sel = h if positions is None else h[:, positions]
        x = sel.float()
        if add is not None:
            x = x + add * uu
        else:
            target = 0.0 if value is None else float(value)
            x = x - ((x @ uu) - target).unsqueeze(-1) * uu
        h = h.clone()
        if positions is None:
            return x.to(h.dtype)
        h[:, positions] = x.to(h.dtype)
        return h
    return fn


@contextmanager
def no_hooks_left(model: Any) -> Iterator[None]:
    """Fails if a block leaves a hook behind."""
    before = n_hooks(model)
    yield
    after = n_hooks(model)
    if after != before:
        raise RuntimeError(f"{after - before} hook(s) left registered")
