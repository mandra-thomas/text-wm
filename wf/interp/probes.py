"""State probes (design §6): labels from the history, logistic probes on held-out games, the belief shift."""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from wf.data.transitions import history_facts

FACTS = ("at_receptacle", "object_listed", "holding_any", "holding_object", "receptacle_closed")
# Which fact decides each determined failure (wf.data.transitions.determination_reason).
DECIDING = {"away_from_receptacle": "at_receptacle", "away_from_receptacle_never_moved": "at_receptacle",
            "hands_full": "holding_any", "not_holding_object": "holding_object"}
_ACT = re.compile(r"^(take|move|heat|cool|clean) ([a-z]+ \d+) (?:from|to|with) ([a-z]+ \d+)$")


def fact_labels(t: dict) -> dict[str, bool | None]:
    """Each fact as the history establishes it before the action; None where the history does not settle it."""
    f = history_facts(t["history_text"])
    m = _ACT.match(t["action"])
    obj, rec = (m[2], m[3]) if m else (None, None)
    out: dict[str, bool | None] = {k: None for k in FACTS}
    if f.knows_position:
        out["holding_any"] = f.holding is not None
        if obj:
            out["holding_object"] = f.holding == obj
        if rec:
            out["at_receptacle"] = f.location == rec
    if rec and rec in f.contents and obj:
        out["object_listed"] = obj in f.contents[rec]
    if rec and (rec in f.closed or rec in f.opened):
        out["receptacle_closed"] = rec in f.closed
    return out


def success_value(t: dict, fact: str) -> bool | None:
    """The value of `fact` that the action needs in order to work (the sign of a 'toward success' shift)."""
    verb = t["action"].split(" ", 1)[0]
    return {"at_receptacle": True, "object_listed": True, "receptacle_closed": False,
            "holding_any": False if verb == "take" else None,
            "holding_object": True if verb in ("move", "heat", "cool", "clean") else None}[fact]


def deciding_fact(t: dict) -> str | None:
    reason = t.get("determination_reason")
    if reason in DECIDING:
        return DECIDING[reason]
    if reason == "contents_or_open_state":
        labels = fact_labels(t)
        if labels["receptacle_closed"]:
            return "receptacle_closed"
        if labels["object_listed"] is False:
            return "object_listed"
    return None


def split_games(games: list[str], seed: int) -> tuple[set[str], set[str]]:
    """(train, test) halves of the games by a seeded shuffle (design §3)."""
    g = sorted(set(games))
    random.Random(f"interp|split|{seed}").shuffle(g)
    half = len(g) // 2
    return set(g[:half]), set(g[half:])


def balanced_indices(labels: list[bool], outcomes: list[bool], seed: int) -> list[int]:
    """Subsample so that, among successes and among failures separately, the two fact values are equally common."""
    rng = random.Random(f"interp|balance|{seed}")
    keep: list[int] = []
    for outcome in (True, False):
        pos = [i for i, (y, o) in enumerate(zip(labels, outcomes)) if o == outcome and y]
        neg = [i for i, (y, o) in enumerate(zip(labels, outcomes)) if o == outcome and not y]
        n = min(len(pos), len(neg))
        keep += rng.sample(pos, n) + rng.sample(neg, n)
    return sorted(keep)


def fit_probe(X: Any, y: Any, groups: list[str], seed: int = 0) -> Any:
    """Standardised logistic regression, L2 penalty chosen by grouped 5-fold cross-validation over games."""
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    folds = min(5, len(set(groups)))
    cv = list(GroupKFold(n_splits=folds).split(X, y, groups))
    clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1.0], cv=cv, max_iter=2000, scoring="roc_auc", random_state=seed)
    return make_pipeline(StandardScaler(), clf).fit(X, y)


def probe_logit(model: Any, X: Any) -> Any:
    import numpy as np

    return model.decision_function(np.asarray(X, dtype=np.float64))


def probe_auc(model: Any, X: Any, y: Any) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, probe_logit(model, X)))


def separation(model: Any, X: Any, y: Any) -> float:
    """Mean probe logit of the positive class minus that of the negative class (the shift's unit)."""
    import numpy as np

    z, y = probe_logit(model, X), np.asarray(y, dtype=bool)
    return float(z[y].mean() - z[~y].mean())


def belief_shifts(model: Any, X_aligned: Any, X_swapped: Any, sep: float, toward_success: list[bool]) -> list[float]:
    """Per item, the standardised shift of the probe logit, aligned minus swapped, signed toward success."""
    za, zs = probe_logit(model, X_aligned), probe_logit(model, X_swapped)
    return [float((a - s) / sep * (1 if up else -1)) for a, s, up in zip(za, zs, toward_success)]


def probe_direction(model: Any) -> Any:
    """The probe's weight vector in the residual's own coordinates (undoing the standardisation), unit length."""
    import numpy as np

    scaler, clf = model[0], model[-1]
    w = clf.coef_[0] / scaler.scale_
    return w / np.linalg.norm(w)


def item_seed(t: dict) -> int:
    return int(hashlib.sha256(f"interp|{t['id']}".encode()).hexdigest()[:8], 16)
