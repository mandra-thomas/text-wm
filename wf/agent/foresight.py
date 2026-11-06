"""Foresight hooks: none | oracle | validity_only | corrupted (§4.7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from wf.data.candidates import attach_candidates
from wf.env.replay import replay, step_counterfactual
from wf.env.server import AlfworldClient, GameState


@dataclass
class ForesightConfig:
    condition: str = "none"  # none | oracle | validity_only | corrupted
    arm: str | None = None  # favorable | unfavorable
    retention: str = "discarded"  # discarded | retained
    corrupt_fn: Callable[[int, bool], bool] | None = None
    p_arm: float | None = None  # the rate behind corrupt_fn, kept with the episode


def foresight_prediction(
    client: AlfworldClient,
    state: GameState,
    proposal: str,
    prefix_actions: list[str],
    t: int,
    cfg: ForesightConfig,
    success_cache: dict | None = None,
) -> dict[str, Any]:
    """Return prediction_shown, prediction_type, proposal_valid, oracle_obs."""
    # Always get oracle via replay (environment-generated only), on a private env id so the
    # live episode is untouched.
    sandbox = client.sandbox()
    handle = replay(sandbox, state.game_id, prefix_actions, split=state.split)
    oracle_obs, _info = step_counterfactual(handle, proposal)
    proposal_valid = oracle_obs.strip() != "Nothing happens."

    if cfg.condition == "none":
        return {
            "prediction_shown": None,
            "prediction_type": "none",
            "proposal_valid": proposal_valid,
            "oracle_obs": oracle_obs,
            "corrupted": False,
        }

    if cfg.condition == "validity_only":
        shown = "This action is valid." if proposal_valid else "This action is not valid."
        return {
            "prediction_shown": shown,
            "prediction_type": "validity_only",
            "proposal_valid": proposal_valid,
            "oracle_obs": oracle_obs,
            "corrupted": False,
        }

    if cfg.condition == "oracle":
        return {
            "prediction_shown": oracle_obs,
            "prediction_type": "oracle",
            "proposal_valid": proposal_valid,
            "oracle_obs": oracle_obs,
            "corrupted": False,
        }

    # corrupted
    do_corrupt = False
    if cfg.corrupt_fn is not None:
        do_corrupt = cfg.corrupt_fn(t, proposal_valid)

    if not do_corrupt:
        return {
            "prediction_shown": oracle_obs,
            "prediction_type": "oracle",
            "proposal_valid": proposal_valid,
            "oracle_obs": oracle_obs,
            "corrupted": False,
        }

    # Inject FS / FF using env-generated success twin when needed
    if cfg.arm == "favorable" and not proposal_valid:
        # show o_plus for failing action
        tmp = {
            "game_id": state.game_id,
            "split": state.split,
            "action": proposal,
            "true_obs": oracle_obs,
            "history_pairs": state.history,
        }
        attach_candidates(sandbox, tmp, success_cache=success_cache)
        cands = tmp.get("candidates")
        if cands is None:
            # No success twin exists, so nothing false can be shown: not a corruption.
            return {
                "prediction_shown": oracle_obs,
                "prediction_type": "oracle",
                "proposal_valid": proposal_valid,
                "oracle_obs": oracle_obs,
                "corrupted": False,
                "corruption_unavailable": True,
            }
        shown = cands["o_plus"]
        ptype = "corrupted_favorable"
    elif cfg.arm == "unfavorable" and proposal_valid:
        shown = "Nothing happens."
        ptype = "corrupted_unfavorable"
    else:
        shown = oracle_obs
        ptype = "oracle"
        do_corrupt = False

    return {
        "prediction_shown": shown,
        "prediction_type": ptype,
        "proposal_valid": proposal_valid,
        "oracle_obs": oracle_obs,
        "corrupted": do_corrupt,
    }
