"""Model loading and the parts the hooks attach to, for HF decoder-only models (Qwen3, Llama)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wf.constants import revision_for


@dataclass
class Parts:
    """Handles on one model's modules. `layers[i]` returns (hidden, ...); its output is the residual after layer i."""

    model: Any
    embed: Any
    layers: list
    norm: Any
    lm_head: Any
    n_layers: int
    n_heads: int
    head_dim: int
    d_model: int

    def attn(self, layer: int) -> Any:
        return self.layers[layer].self_attn

    def mlp(self, layer: int) -> Any:
        return self.layers[layer].mlp

    def o_proj(self, layer: int) -> Any:
        return self.layers[layer].self_attn.o_proj

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device


def parts(model: Any) -> Parts:
    """Find the modules by the layout both Qwen3ForCausalLM and LlamaForCausalLM share in transformers 4.51."""
    inner = model.model
    cfg = model.config
    layers = list(inner.layers)
    for name in ("self_attn", "mlp"):
        if not all(hasattr(layer, name) for layer in layers):
            raise TypeError(f"{type(model).__name__}: decoder layers lack `{name}`; this adapter supports Qwen3 and Llama")
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return Parts(model=model, embed=inner.embed_tokens, layers=layers, norm=inner.norm, lm_head=model.lm_head,
                 n_layers=len(layers), n_heads=cfg.num_attention_heads, head_dim=head_dim, d_model=cfg.hidden_size)


def load(model_id: str, dtype: str = "bfloat16", device_map: str | None = "auto", attn: str = "sdpa") -> tuple[Any, Any]:
    """(model, tokenizer) at the pinned snapshot, in eval mode, as the scoring runs load them.

    `attn="eager"` is needed only to read attention weights; patching works with either.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = revision_for(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, torch_dtype=getattr(torch, dtype),
                                                 device_map=device_map, trust_remote_code=True, attn_implementation=attn)
    model.eval()
    return model, tok
