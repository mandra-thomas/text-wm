"""Preregistered constants — single source of truth (PILOT_PROTOCOL §7).

Do not redefine elsewhere. After filing, only change via human edit + disclosure.
"""

# Preregistered — do not change after filing
DELTA_SUPPORT_PP = 5.0  # H2 support threshold (percentage points)
DELTA_STRONG_PP = 10.0
PERMUTATION_P_MAX = 0.01
H_FLOOR_PP = 3.0  # cell-level sign claims need |H| >= this in success-rate points
FUTILITY_EST_PP = 2.0  # pilot goal-channel rule: estimate below this AND ...
FUTILITY_UPPER_PP = 5.0  # ... 95% CI upper bound below this, on every tested model
HEADROOM_MIN_PP = 5.0  # oracle must beat none by this in the pilot, else RQ3 -> ScienceWorld
HEADROOM_RECHECK_GAMES = 100  # borderline headroom triggers a recheck before moving
CORRUPTIONS_PER_EPISODE_TARGET = 3
N_SCHEDULES = 3
# Minimum |Δ mean rank-outcome S| for retained vs discarded on the favorable arm
# in pilotB_retention_minirun; below it, switch to state-persistence format.
RETENTION_MARGIN = 1.0
STEP_CAP = 30  # confirm against the stack default in t0_smoke; record in runs/env.json

# Engineering thresholds — may be tuned, changes logged in STATUS.md
MAX_UNPARSEABLE = 0.05
SCORING_AGREEMENT_MIN = 0.90
CONSUMPTION_MIN = 0.10  # decision-critical rate below this => stop and report
BOOTSTRAP_RESAMPLES = 2000
PERMUTATIONS = 1000
RNG_SEED = 20260924

# Model IDs
AGENT_MODEL = "Qwen/Qwen3-8B"
WM_FEWSHOT_MODEL = "Qwen/Qwen3-8B"
WM_FT_QWEN = "X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B"
WM_FT_LLAMA = "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B"
TRANSITION_DATASET = "X1AOX1A/LLMasWorldModels"

# Serving defaults
VLLM_BASE_URL_DEFAULT = "http://127.0.0.1:8000/v1"
ALFWORLD_BASE_URL_DEFAULT = "http://127.0.0.1:36001"
