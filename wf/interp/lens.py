"""Logit lens and readout lens at the answer position (design §4)."""
from __future__ import annotations

from typing import Any

from wf.interp.model import Parts


def readout_vector(p: Parts, plus: int, minus: int) -> Any:
    """W_U[plus] - W_U[minus], float32."""
    w = p.lm_head.weight
    return (w[plus] - w[minus]).float()


def logit_lens(p: Parts, resid_last: Any, plus: int, minus: int) -> list[float]:
    """Readout of each layer's residual at the answer position, through the final norm and unembedding.

    `resid_last` is (layers, width) for one prompt: the decoder layers' outputs at the last position. At the last
    layer this equals the model's own readout (the model applies the same norm and unembedding).
    """
    import torch

    with torch.no_grad():
        h = resid_last.to(device=p.device, dtype=next(p.norm.parameters()).dtype)
        normed = p.norm(h).float()
        return (normed @ readout_vector(p, plus, minus).to(normed.device)).tolist()


def fit_readout_lens(X_by_layer: Any, y: Any, groups: Any, alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 1e4)) -> list:
    """Per layer, a ridge regression from the residual to the final readout, penalty chosen by grouped 5-fold CV.

    `X_by_layer` is (layers, items, width) for neutral-task prompts of train games; `y` the items' final readout;
    `groups` their game ids. Returns one fitted sklearn pipeline per layer.
    """
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=np.float64)
    n_groups = len(set(groups))
    cv = list(GroupKFold(n_splits=min(5, n_groups)).split(X_by_layer[0], y, groups)) if n_groups >= 2 else None
    lenses = []
    for X in X_by_layer:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=cv))
        lenses.append(model.fit(np.asarray(X, dtype=np.float64), y))
    return lenses


def apply_readout_lens(lenses: list, X_by_layer: Any) -> Any:
    """(layers, items) predictions of the readout lens."""
    import numpy as np

    return np.stack([lens.predict(np.asarray(X, dtype=np.float64)) for lens, X in zip(lenses, X_by_layer)])


def divergence_layer(effect_by_layer: list[float], final_effect: float) -> float:
    """First layer (share of depth) from which the lens effect stays at or above half the final effect."""
    from wf.interp.patching import arrival_layer

    if final_effect == 0:
        return float("nan")
    return arrival_layer([e / final_effect for e in effect_by_layer], 0.5)
