"""Preregistered constants — single source of truth (PILOT_PROTOCOL §7).

Do not redefine elsewhere. After filing, only change via human edit + disclosure.
"""

# Preregistered — do not change after filing
DELTA_SUPPORT_PP = 5.0  # H2 support threshold (percentage points)
DELTA_STRONG_PP = 10.0
PERMUTATION_P_MAX = 0.01
DELTA_L_SUPPORT_NATS = 0.5  # prereg PDF: H2 needs mean log-odds shift >= this
DELTA_L_STRONG_NATS = 1.0  # prereg PDF: reported as a strong effect
H_FLOOR_PP = 3.0  # cell-level sign claims need |H| >= this in success-rate points
FUTILITY_EST_PP = 2.0  # pilot goal-channel rule: estimate below this AND ...
FUTILITY_UPPER_PP = 5.0  # ... 95% CI upper bound below this, on every tested model
# Decision 4 (2026-11-17): the goal-channel rule is advisory and also read in nats, the prereg's
# headline scale: it "fires" when every scored model's ΔL 95% CI upper bound is below this.
FUTILITY_UPPER_NATS = 0.5
HEADROOM_MIN_PP = 5.0  # advisory line for the oracle-over-none success gain; not a gate since 2026-11-17
HEADROOM_RECHECK_GAMES = 100  # unused since 2026-11-17 (the recheck is skipped)
CORRUPTIONS_PER_EPISODE_TARGET = 3
N_SCHEDULES = 3
# Minimum |Δ mean rank-outcome S| for retained vs discarded on the favorable arm
# in pilotB_retention_minirun; below it, switch to state-persistence format.
RETENTION_MARGIN = 1.0
STEP_CAP = 30  # confirm against the stack default in t0_smoke; record in runs/env.json

ENV_STEP_LIMIT = 50  # ALFWorld ends every episode here (max_nb_steps_per_episode); not ours to change

# Engineering thresholds — may be tuned, changes logged in STATUS.md
MAX_UNPARSEABLE = 0.05
SCORING_AGREEMENT_MIN = 0.90
# Decision 13 (2026-11-17): Pilot A stops if sign(L) and the model's own text agree on fewer
# than this share of determined true failures, the population RQ2 scores.
SCORING_STOP_FAILURES_MIN = 0.90
# Sample-size planning for ΔL: two-sided alpha 0.005 (Holm's first step at 0.01, two models), power 0.80.
POWER_Z_ALPHA = 2.807
POWER_Z_BETA = 0.842
CONSUMPTION_MIN = 0.10  # decision-critical rate below this => stop and report
BOOTSTRAP_RESAMPLES = 2000
PERMUTATIONS = 1000
RNG_SEED = 20260924

# Model IDs
AGENT_MODEL = "Qwen/Qwen3-8B"
WM_FEWSHOT_MODEL = "Qwen/Qwen3-8B"
WM_FEWSHOT_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"  # co-primary (prereg); gated on Hugging Face
WM_FT_QWEN = "X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B"
WM_FT_LLAMA = "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B"

# Hugging Face snapshot (commit) of every model the study loads (decided 2026-11-20, audit A5). Each model had
# exactly one snapshot in the ICE caches (hfcache and hfcache/hub), so these are the revisions every reported
# run used. Weights, tokenizer and chat template are all read at this revision, so a later push to the Hub
# cannot change a prompt or a score. A model missing here is refused unless WF_ALLOW_UNPINNED=1.
REVISIONS = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B": "8d4f069c4bf5a739763664897b26ca17c161f0da",
    "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B": "8f089d2126f0406a296747041ebf33b902837cd3",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    # Model families, wave A (2026-11-20, wf/families.py): the Hub's current commit ("sha" from
    # https://huggingface.co/api/models/<id>, read 2026-11-20, before any download); the prefetch job downloads
    # exactly these.
    "google/gemma-2-9b-it": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "meta-llama/Llama-3.2-1B-Instruct": "9213176726f574b556790deb65791e0c5aa438b6",
    "meta-llama/Llama-3.2-3B-Instruct": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    # Interpretability pilot, base-versus-instruct stage 0 (notes/interp_design.md, amendment 1, A9): the `main`
    # revision reported by the Hugging Face API on 2026-11-20, before any download.
    "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    # Post-training pilot (notes/posttraining_design.md, wf/posttrain.py): the Hub's `main` sha read 2026-11-26 from
    # https://huggingface.co/api/models/<id>, before any download.
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "allenai/Llama-3.1-Tulu-3-8B-SFT": "f2a0b46b0cfda21003c6141b1ff837b7e165524d",
    "allenai/Llama-3.1-Tulu-3-8B-DPO": "a7beb67e33ffd01cc87ac3b46cadc1000985b8db",
    "allenai/Llama-3.1-Tulu-3-8B": "666943798adbde0b1aff34626007e26986a3c107",
    "allenai/Llama-3.1-Tulu-3.1-8B": "46239c2d07db76b412e1f1b0b4542f65b81fe01f",
}


def revision_for(model_id: str) -> str | None:
    """The pinned snapshot for a model id. An unknown id fails loudly, unless WF_ALLOW_UNPINNED=1 is set,
    in which case it loads the Hub's current revision (None) and the run record says so."""
    import os

    if model_id in REVISIONS:
        return REVISIONS[model_id]
    if os.environ.get("WF_ALLOW_UNPINNED") == "1":
        return None
    raise KeyError(f"model {model_id!r} has no pinned revision in wf.constants.REVISIONS; add its snapshot hash "
                   f"or set WF_ALLOW_UNPINNED=1 to load the Hub's current revision")


TRANSITION_DATASET = "X1AOX1A/LLMasWorldModels"
PARSER_DATASET_FILE = "llama_factory/alfworld_test_with_env_195.json"
CONFIRMATORY_GAMES = 134  # solvable valid_unseen; the game count sample sizes are planned for

# Serving defaults
VLLM_BASE_URL_DEFAULT = "http://127.0.0.1:8000/v1"
ALFWORLD_BASE_URL_DEFAULT = "http://127.0.0.1:36001"

# Parser audit for model-written predictions (2026-11-17), fixed before the audit is read
PARSER_AUDIT_N = 100
PARSER_AUDIT_MIN_AGREEMENT = 0.95  # on rows the reader labels S or F; the parser's "?" counts as a miss
