---
description: Wishful Foresight pilot protocol — implementation spec for the coding agent. Governs all work in this repo until the pilot report is written.
alwaysApply: true
---

# Wishful Foresight — Pilot Protocol (implementation spec)

**Install:** copy this file to `.cursor/rules/pilot-protocol.mdc` (keep the frontmatter) so it applies to every chat, or keep it at the repo root as `PILOT_PROTOCOL.md` and `@`-mention it. Treat everything below as fixed unless a human edits this file.

**What this repo does.** It runs the preliminary experiments for a study of goal-conditioned bias in text world models and its effect on LLM agents in ALFWorld. There are three research questions; the pilot exists to (a) prove the infrastructure, (b) calibrate parameters the preregistration leaves to calibration, and (c) apply three preregistered pilot rules. The pilot does **not** decide what the study tests. Do not add experiments, change thresholds, or "improve" the design; log ideas in `notes/extensions.md`.

**Coding agent rules**

- No LLM judges anywhere. Outcomes are parsed from ALFWorld's templated observations with deterministic regexes.
- Never hand-write a corrupted or counterfactual observation. Every observation shown to a model comes from the environment.
- Pilot games and transitions come from ALFWorld's **train split**. Never touch `valid_seen` or `valid_unseen` in the pilot except for the single-step accuracy check in Tier 0 (read-only, 200 items).
- Every run writes JSONL to `runs/<run_id>/` with a `config.json` copy and a `git_sha`. Re-running a run id must reproduce byte-identical outputs where the pipeline is deterministic.
- Constants in §7 are preregistered. They live in one file (`wf/constants.py`) and are read, never redefined.
- When a check in §6 fails its acceptance criterion, stop and write the failure into `runs/<run_id>/STATUS.md`; do not silently adjust.

---

## 1. Fixed facts

| Item | Value |
| --- | --- |
| Environment | ALFWorld via the AgentGym stack released with Word2World (`put` is renamed `move`; a `help` action exists; use their data download, not the original ALFWorld download) |
| Splits | `train` (pilot), `valid_seen` (140 games), `valid_unseen` (134 games). These counts are the **solvable** universe (see §5.1). |
| Agent | `Qwen/Qwen3-8B`, ReAct prompt, temperature 0, thinking disabled, step cap `STEP_CAP` (default 30; confirm the stack's default in Tier 0 and record it) |
| Primary world model | `Qwen/Qwen3-8B`, few-shot next-observation prompt, thinking disabled, scored by teacher forcing |
| Fine-tuned world models | `X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B`, `X1AOX1A/WorldModel-Alfworld-Llama3.1-8B` (use the prompt protocol in Word2World's `scripts/interact_with_world_model/run.py`) |
| Transition data | HF dataset `X1AOX1A/LLMasWorldModels` (ALFWorld split) plus our own counterfactual transitions |
| Serving | vLLM for agent generation; HF transformers (fp16/bf16) for teacher-forced scoring |
| Stats unit | games (ALFWorld game = one task instance) |

Upstream repo: `https://github.com/X1AOX1A/Word2World` — env setup `scripts/env_setup/uv_agentgym_rl.sh` and `uv_alfworld.sh`; data `scripts/download_data/download_data.py`; env server `scripts/env_server/start_alfworld.sh`.

Qwen3 thinking off: with the vLLM OpenAI-compatible server pass `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`; with HF `tokenizer.apply_chat_template(..., enable_thinking=False)`.

---

## 2. Repo layout to create

```
wf/
  constants.py          # §7 constants, single source of truth
  config/               # YAML per run (see §6)
  env/
    server.py           # start/stop AgentGym ALFWorld server; split selection
    replay.py           # deterministic prefix replay + counterfactual action execution
    expert.py           # expert-path labels (see §4.6)
  data/
    transitions.py      # counterfactual transition builder + labels
    goals.py            # scene-matched swapped goals; goal-agnostic history labeling
    candidates.py       # matched candidate observations (o_plus, o_minus)
  scoring/
    parser.py           # templated-observation parser -> outcome class
    prompts.py          # world-model few-shot prompt; agent ReAct prompt
    logodds.py          # teacher-forced scoring
  agent/
    react.py            # agent loop with foresight hooks
    foresight.py        # none | oracle | validity_only | corrupted; discarded | retained
  corruption/
    schedule.py         # seeded per-game schedules; eligibility; per-arm probabilities
  analysis/
    stats.py            # game-clustered bootstrap, within-pair permutation, Δ_p, A, rank outcome, I
    report.py           # writes runs/<id>/REPORT.md from JSONL
runs/                   # outputs (gitignored except REPORT.md and STATUS.md)
notes/extensions.md     # ideas that are NOT in scope
tests/                  # unit tests for parser, replay determinism, schedule matching, stats
```

---

## 3. Data schemas (JSONL, one record per line)

**`transition`** (from `data/transitions.py`)

```json
{"id": "t_000123", "game_id": "train/pick_and_place_simple-...-Mug", "split": "train",
 "state_idx": 4, "prefix_actions": ["go to countertop 1", "open drawer 1"],
 "history_text": "...", "task_line": "Your task is to: put a mug in coffeemachine.",
 "goal_object": "mug", "scene_objects": ["mug", "apple", "spraybottle", "..."],
 "swapped_goal_object": "spraybottle",
 "action": "take mug 1 from drawer 1", "action_type": "take",
 "true_obs": "Nothing happens.", "outcome_class": "noop",
 "error_type": "validity", "context_determined": true,
 "history_goal_agnostic": true, "history_len": 4,
 "targets_goal_object": true, "source": "counterfactual",
 "candidates": {"o_plus": "You pick up the mug 1 from the drawer 1.", "o_minus": "Nothing happens."}}
```

`outcome_class ∈ {noop, state_change, reveal_with_goal, reveal_without_goal}`; `error_type ∈ {validity, precondition, presence, completion}` (§4.2). `context_determined` is true when validity/precondition/contents are decidable from `history_text` alone.

**`scoring`** (from `scoring/logodds.py`)

```json
{"transition_id": "t_000123", "model": "qwen3-8b-fewshot",
 "condition": "aligned", "logp_plus": -12.31, "logp_minus": -3.02,
 "L": -9.29, "p_success": 0.0000923, "n_tokens_plus": 14, "n_tokens_minus": 4}
```

`condition ∈ {aligned, swapped, absent, committed, candidate}`. `L = logp_plus − logp_minus`; `p_success = sigmoid(L)`. Candidates are identical across conditions for a given `transition_id`.

**`episode`** (from `agent/react.py`)

```json
{"run_id": "pilotB_headroom", "game_id": "train/...", "split": "train",
 "condition": "corrupted", "arm": "favorable", "retention": "retained",
 "schedule_seed": 17, "steps": 23, "success": false, "n_corrupted": 3,
 "trace": [{"t": 5, "proposal": "take mug 1 from drawer 1", "proposal_valid": false,
            "prediction_shown": "You pick up the mug 1 from the drawer 1.",
            "prediction_type": "corrupted_favorable",
            "action_under_true_prediction": "go to drawer 2",
            "executed": "take mug 1 from drawer 1", "real_obs": "Nothing happens.",
            "decision_critical": true, "followed": true,
            "on_expert_path": false, "targets_goal_object": true}]}
```

`decision_critical` = `action_under_true_prediction != no-foresight proposal` (§4.7). `followed` = executed action consistent with the shown prediction (for a validity prediction: executed the proposal iff prediction said it works).

---

## 4. Modules and interfaces

### 4.1 `env/replay.py`

```python
def replay(game_id: str, prefix_actions: list[str]) -> EnvHandle
def step_counterfactual(handle: EnvHandle, action: str) -> tuple[str, dict]  # (observation, info)
```

`replay` resets the game and re-executes `prefix_actions`, returning a handle positioned at the state. `step_counterfactual` executes one action and returns the observation; the handle is then discarded. Determinism test: replaying the same prefix twice must yield identical observations for the same counterfactual actions (Tier 0.2).

### 4.2 `data/transitions.py`

At each sampled state (every k-th step of a logged ReAct rollout, k=2, plus every step where the agent's proposal was invalid), generate alternatives and execute each via `replay`:

- the agent's own proposal;
- an invalid action (object not present / not visible) — `error_type=validity`;
- an unmet-precondition action (`move X to Y` while not holding X; `heat X with microwave` while not holding X; `open` an already-open receptacle) — `error_type=precondition`;
- wrong-object and wrong-receptacle variants of the proposal;
- **targeted goal-object failures**: `take <goal_object> from <receptacle where it is not>`, `move <goal_object> to <target>` while not holding it — `error_type=validity` or `precondition` as applicable, `targets_goal_object=true`.

Label `context_determined` by rule: validity and precondition failures are determined when the relevant fact (object location, holding state, receptacle open/closed) appears in `history_text`; reveal outcomes of unopened receptacles are `context_determined=false`. Balance success/failure within state where possible; record the imbalance.

### 4.3 `data/goals.py`

- `history_goal_agnostic(history_text, goal_object, swapped_goal_object) -> bool`: true iff the prefix contains no `take/move/heat/cool/clean/examine` action naming either object. Navigation and `open/close` are allowed.
- `swapped_goal(scene_objects, goal_object, receptacle_in_question) -> str`: choose an object present in the scene (from the environment's object list, not from observations) but not in the receptacle the action concerns, of the same task type where possible. Record the choice; use a fixed RNG seed.
- `task_line(goal_object, task_type) -> str`: reconstruct the first-observation task line in the exact template the stack uses.

Conditions are built by replacing **only** the task line in the first observation. Nothing else in `history_text` changes.

### 4.4 `data/candidates.py`

For each transition produce `o_plus` (favorable) and `o_minus` (unfavorable), both environment-generated:

- If the true action fails (`true_obs == "Nothing happens."`): `o_minus = true_obs`; `o_plus` = the observation the environment produces for the same action type and objects in a state where it succeeds. Obtain it by replaying a state in the same game where the action is valid (e.g., after moving to the receptacle that holds the object) and executing it; cache by `(game_id, action)`.
- If the true action succeeds: `o_plus = true_obs`; `o_minus = "Nothing happens."`.

Never template `o_plus` by hand. If no succeeding state exists for that action in that game, mark the transition `candidates=null` and exclude it from scoring (count exclusions).

### 4.5 `scoring/parser.py` and `scoring/logodds.py`

Parser: regex table over the templates observed in Tier 0.1 logs (starting points: `^Nothing happens\.$` → `noop`; `^You (pick up|put|move|open|close|heat|cool|clean|turn on|turn off)` → `state_change`; `In it, you see` / `On the .* you see` → `reveal_*`, with goal-object presence checked by substring on `<goal_object> \d+`). Return `unparseable` otherwise; the unparseable rate is reported.

Scoring: for prompt `P` (few-shot world-model prompt + history under the condition's task line + action) and candidate `c`, compute `logp(c | P) = Σ log p(token_i | P, c_<i)` over candidate tokens only, thinking disabled, bf16, no sampling. Cache `(model, condition, transition_id)`.

Validation (Tier 0.6): on 100 items, generate freely (greedy, max 64 tokens) under the aligned condition, parse to a class, and compare with `sign(L)`; report agreement.

### 4.6 `env/expert.py`

Label each executed or proposed action `on_expert_path` using ALFWorld's expert (handcoded expert plan exposed through the underlying environment; if the AgentGym wrapper does not expose it, instantiate the underlying `alfworld` env with `expert_type: handcoded` for labeling only). If neither is available, set `on_expert_path=null` and record it in STATUS.md — do not approximate with a heuristic.

### 4.7 `agent/react.py` and `agent/foresight.py`

Loop per step:

1. Agent proposes `a_t` (no foresight).
2. Foresight hook returns `prediction_shown` per condition:
   - `none`: nothing shown;
   - `oracle`: true observation from `replay` + `step_counterfactual`;
   - `validity_only`: "This action is valid." / "This action is not valid." from the same replay;
   - `corrupted`: oracle, except at scheduled eligible steps where the arm's corruption replaces it (favorable arm: proposal invalid → show `o_plus`; unfavorable arm: proposal valid → show `"Nothing happens."`).
3. **Also** compute `action_under_true_prediction` by querying the agent with the oracle prediction (one extra call at corrupted steps only), for `decision_critical`.
4. Agent decides `executed` given `prediction_shown`; environment returns `real_obs`.
5. Retention: `discarded` removes the prediction from the history after execution; `retained` keeps it in the trace as the agent's "Expected outcome:" line immediately before the action, followed by the real observation. **The prediction is presented identically at decision time in both modes.**

The ReAct prompt, the "Expected outcome:" wording, and the few-shot examples are frozen in `scoring/prompts.py` before Pilot B.

### 4.8 `corruption/schedule.py`

```python
def schedule(game_id: str, seed: int, arm: str, p_arm: float) -> Callable[[int, bool], bool]
```

Returns a function `(t, proposal_valid) -> corrupt_now`, deterministic in `(game_id, seed)`. Eligibility: favorable arm corrupts only steps with `proposal_valid=False`; unfavorable arm only `proposal_valid=True`. `p_arm` is set in Pilot B so that the expected realized corruptions per episode ≈ `CORRUPTIONS_PER_EPISODE_TARGET` in each arm. Three seeds per game (`N_SCHEDULES`), shared across arms and retention modes. Log realized counts.

### 4.9 `analysis/stats.py`

- `delta_p(scores, primary_condition="aligned", control="swapped")`: mean over items of `sigmoid(L_aligned) − sigmoid(L_swapped)` in percentage points, restricted to `context_determined & true failure & history_goal_agnostic`.
- `asymmetry_A(scores)`: on context-determined items, `P(sign(L)>0 | true failure) − P(sign(L)<0 | true success)`.
- `cluster_bootstrap(stat_fn, items, cluster="game_id", B=BOOTSTRAP_RESAMPLES)`: percentile CI.
- `pair_permutation(diffs, n=PERMUTATIONS)`: sign-flip within pairs; p-value.
- `icc(items, cluster="game_id")`: intraclass correlation of `L_aligned − L_swapped`.
- `rank_outcome(episodes)`: per episode `S = −min(steps, STEP_CAP)` with failures set to `−(STEP_CAP+1)`; average over schedules within game.
- `H_cell(episodes, retention)`: mean over games of `S_favorable − S_unfavorable` (higher S is better, so H > 0 means unfavorable errors are worse).
- `interaction_I(episodes)`: `H_retained − H_discarded`, game-bootstrapped.

Sign convention (preregistered): predicted `H_discarded > 0`, `H_retained < 0`, `I < 0`.

---

## 5. Environment setup (once)

1. Clone Word2World; run `scripts/env_setup/uv_agentgym_rl.sh` and `uv_alfworld.sh`.
2. `python scripts/download_data/download_data.py` (their renamed-action ALFWorld data). Remove any pre-existing ALFWorld data first.
3. Start the server with `scripts/env_server/start_alfworld.sh`; confirm `train`, `valid_seen`, `valid_unseen` are selectable and record **solvable** game counts (§5.1).
4. Pull `X1AOX1A/LLMasWorldModels` (ALFWorld) and the two fine-tuned checkpoints.
5. Serve `Qwen/Qwen3-8B` with vLLM (`--max-model-len 8192`, `--enable-prefix-caching`); confirm `enable_thinking=false` takes effect by checking that no `<think>` block appears in 20 sampled responses.

Record versions (commit SHA of Word2World, vLLM, transformers, model revisions) in `runs/env.json`.

### 5.1 Solvable games only (AgentGym vs protocol counts)

Word2World AgentGym builds one flat game list from its mapping JSONs. Those mappings include trials whose `game.tw-pddl` has `"solvable": false`. Loading them via `/reset` fails inside TextWorld/PDDL (commonly `KeyError: 'val1'`). Upstream `AlfredTWEnv.collect_game_files` already skips unsolvable files; AgentGym does not.

**Study population = solvable games only.** That is what §1’s split sizes mean:

| Split | AgentGym mapping rows | Solvable (`solvable: true`) | Protocol / this repo |
| --- | ---: | ---: | ---: |
| `train` | 2420 | 2179 | pilot sampling pool |
| `valid_train` | 200 | 197 | unused in pilot |
| `valid_seen` | 187 | 140 | confirmatory (§1) |
| `valid_unseen` | 195 | 134 | confirmatory + Tier 0 ckpt check |

Implementation: `wf/env/server.py` `AlfworldClient.list_games` returns only solvable local indices (reads `ALFWORLD_DATA` / `~/.cache/alfworld` + Word2World mapping JSONs). Do **not** treat this as an ad-hoc subsample of the protocol universe — it *is* that universe. Do not invent substitute games for unsolvable IDs.

---

## 6. Runs, in order

Run ids are fixed. Each run has a YAML in `wf/config/`, an acceptance criterion, and the decision it feeds. Tier 0 produces no hypothesis-relevant numbers and may run before the proposal is filed. **Pilot A and Pilot B results may only be produced after the preregistration is filed**; put a `FILED.txt` with the filing date in `runs/` and have the Pilot A/B entry points refuse to run without it.

### Tier 0 — infrastructure

| Run id | What | Acceptance | Feeds |
| --- | --- | --- | --- |
| `t0_smoke` | 5 train games, Qwen3-8B ReAct, no foresight; dump every observation | Loop completes; step cap recorded; ≥ 50 observations captured for the parser table | parser templates, `STEP_CAP` |
| `t0_replay` | 20 logged states × 3 counterfactual actions, executed twice | 100% identical observations across the two replays | go/no-go on `replay.py`; if it fails, build a wrapper before anything else |
| `t0_agent_determinism` | 10 games at temperature 0, run twice | Trajectory agreement reported; if < 95% step-level agreement, add agent seeding and note that schedules are not pure replicates | Section 5 inference wording |
| `t0_parser` | Parser over ~2,000 logged observations from `LLMasWorldModels` | Unparseable ≤ `MAX_UNPARSEABLE` (0.05) | parser readiness |
| `t0_ckpt_accuracy` | Word2World Qwen2.5-7B checkpoint, 200 `valid_unseen` items, single-step exact match, split by `context_determined` | Accuracy reported per class; confirm task line is present in first observation | whether the uncertain subset is where the checkpoint must guess |
| `t0_scoring_agreement` | 100 train items: `sign(L)` vs parsed free generation, Qwen3-8B few-shot | Agreement ≥ `SCORING_AGREEMENT_MIN` (0.90); else revise the few-shot prompt and rerun (log both) | scoring method defensibility |
| `t0_expert_labels` | Expert-path labels on 5 games | Labels produced, or `on_expert_path=null` recorded with the reason | RQ3 covariates |

### Pilot A — RQ2 signal (after filing; ≈ 1 GPU-hour)

`pilotA_transitions`: build ≥ 500 counterfactual items from ≥ 80 train states, `context_determined & true failure`, with `history_goal_agnostic` labels and `targets_goal_object` set; report the count per (goal-agnostic, goal-revealing) and per `error_type`.

`pilotA_score`: score all items under `aligned`, `swapped`, `absent`, `committed` on (a) Qwen3-8B few-shot and (b) `WorldModel-Alfworld-Qwen2.5-7B`.

Report (in `REPORT.md`):

- Δ_p on goal-agnostic items, game-clustered 95% CI, within-pair permutation p, per model;
- Δ_p on goal-revealing items separately; Δ_p by `history_len` bin (1–3, 4–7, 8+);
- ICC of `L_aligned − L_swapped`; implied standard error of Δ_p at 1,000 / 2,000 / 3,000 items;
- Δ_p for `committed` vs `candidate` framing (exploratory);
- A (asymmetry) with CI;
- exclusion counts (no candidate, unparseable).

Decisions: apply `PILOT_RULE_GOAL_CHANNEL` (§7); set `ELIGIBLE_SET_TARGET` from the ICC to reach an SE of ≤ 1.5 points.

### Pilot B — RQ3 headroom, consumption, calibration (after filing; ≈ 3–4 GPU-hours)

`pilotB_headroom`: 50 train games × {`none`, `oracle/discarded`, `oracle/retained`, `validity_only`}; 1 run each (schedules irrelevant without corruption).

Report: success rate and rank outcome per condition; oracle gain over none with a game-bootstrap CI; validity-only recovery fraction = (validity_only − none) / (oracle − none); retained-oracle minus discarded-oracle (must not be materially negative — if it is, the retained format is broken; stop and fix before the mini-run).

`pilotB_consumption`: from the oracle episodes, the **decision-critical rate** = fraction of steps where `action_under_true_prediction` differs from the no-foresight proposal, and the follow-rate on oracle predictions. If the decision-critical rate is below `CONSUMPTION_MIN` (0.10), stop: the agent is not consuming foresight and the presentation format must be strengthened before any corruption run is meaningful. Report this before proceeding.

`pilotB_eligibility`: per step, the fraction of proposals that are invalid vs valid; expert-path membership among valid proposals; compute `p_favorable`, `p_unfavorable` such that expected realized corruptions per episode ≈ `CORRUPTIONS_PER_EPISODE_TARGET` (3) in each arm. Write both to `wf/config/calibrated.yaml` with the derivation.

`pilotB_retention_minirun`: 30 train games × {favorable, unfavorable} × {discarded, retained} × 1 schedule (120 episodes). Report all four cells' rank outcome and success rate, `H_discarded`, `H_retained`, preliminary `I`, realized corruptions per episode per arm, follow-rate on corrupted vs correct predictions, steps lost after each error type, retry rate after unfavorable errors on valid actions.

Decisions: apply `PILOT_RULE_HEADROOM` and `PILOT_RULE_RETAINED_FORMAT` (§7).

### Tier 3 — only if Pilots A and B finish cleanly

`t3_base_vs_instruct` (Qwen3-8B-Base few-shot on the Pilot A items, if the base checkpoint scores cleanly); `t3_api_check` (one API model, 200 items, 10 samples each); `t3_sciworld_parser` (200 ScienceWorld observations; start the 200-item human-label sample as a CSV in `data/human_labels/`).

---

## 7. Preregistered constants (`wf/constants.py`)

```python
# Preregistered — do not change after filing
DELTA_SUPPORT_PP = 5.0          # H2 support threshold (percentage points)
DELTA_STRONG_PP = 10.0
PERMUTATION_P_MAX = 0.01
H_FLOOR_PP = 3.0                # cell-level sign claims need |H| >= this in success-rate points
FUTILITY_EST_PP = 2.0           # pilot goal-channel rule: estimate below this AND ...
FUTILITY_UPPER_PP = 5.0         # ... 95% CI upper bound below this, on every tested model
HEADROOM_MIN_PP = 5.0           # oracle must beat none by this in the pilot, else RQ3 -> ScienceWorld
HEADROOM_RECHECK_GAMES = 100    # borderline headroom triggers a recheck before moving
CORRUPTIONS_PER_EPISODE_TARGET = 3
N_SCHEDULES = 3
RETENTION_MARGIN = None         # REQUIRED before filing: minimum rank-outcome shift for the trace-line
                                # retained format to be kept; below it, switch to state persistence.
                                # Set it, commit it, and record the commit SHA in FILED.txt.
STEP_CAP = 30                   # confirm against the stack default in t0_smoke; record in runs/env.json

# Engineering thresholds — may be tuned, changes logged in STATUS.md
MAX_UNPARSEABLE = 0.05
SCORING_AGREEMENT_MIN = 0.90
CONSUMPTION_MIN = 0.10          # decision-critical rate below this => stop and report
BOOTSTRAP_RESAMPLES = 2000
PERMUTATIONS = 1000
RNG_SEED = 20260924
```

Pilot rules, verbatim from the preregistration:

- **PILOT_RULE_GOAL_CHANNEL**: if on every tested model the Pilot A Δ_p point estimate < `FUTILITY_EST_PP` and its 95% CI upper bound < `FUTILITY_UPPER_PP`, the goal channel is dropped from the headline; H3 and the validity-oracle decomposition become the headline; RQ2 is reported as a null. A wide interval alone does not fire the rule.
- **PILOT_RULE_HEADROOM**: if oracle foresight improves ALFWorld success by less than `HEADROOM_MIN_PP` over none in `pilotB_headroom`, RQ3 moves to ScienceWorld before full runs; a borderline result (CI includes the threshold) triggers `pilotB_headroom_recheck` on `HEADROOM_RECHECK_GAMES` games first.
- **PILOT_RULE_RETAINED_FORMAT**: if in `pilotB_retention_minirun` retention moves the favorable arm's rank outcome by less than `RETENTION_MARGIN`, the retained format switches to *state persistence* (the shown prediction updates the agent's maintained belief lines — inventory, receptacle contents, object states — and stays until a later observation naming the same fact overwrites it) for the full runs.

---

## 8. Pilot report template (`runs/PILOT_REPORT.md`)

```
# Pilot report — <date> — git <sha> — FILED <date>

## Tier 0
| check | result | pass |
| t0_replay determinism | 60/60 identical | yes |
| t0_agent_determinism | 97% step agreement over 10 games | yes |
| t0_parser unparseable | 2.1% | yes |
| t0_ckpt_accuracy | determined 99.4% / uncertain 61.0% (n=200) | reported |
| t0_scoring_agreement | 93/100 | yes |
| expert labels | available: yes/no | |

## Pilot A (RQ2)
| model | population | n items / states | Δ_p (pp) | 95% CI | perm p | ICC |
| qwen3-8b | goal-agnostic | | | | | |
| qwen3-8b | goal-revealing | | | | | |
| w2w-qwen2.5-7b | goal-agnostic | | | | | |
A (asymmetry): ... ; Δ_p by history length: ... ; committed vs candidate: ...
Rule applied: PILOT_RULE_GOAL_CHANNEL -> fired / not fired
ELIGIBLE_SET_TARGET: ...

## Pilot B (RQ3)
| condition | success | rank outcome (mean S) |
| none | | |
| oracle/discarded | | |
| oracle/retained | | |
| validity_only | | |
Oracle gain over none: ... (CI) ; validity-only recovery fraction: ...
Decision-critical rate under oracle: ... ; follow-rate: ...
Eligibility: invalid-proposal rate ... ; p_favorable ... ; p_unfavorable ...
Mini-run cells (S, success): fav/disc ... fav/ret ... unfav/disc ... unfav/ret ...
H_discarded ... H_retained ... I ... ; realized corruptions/episode per arm ...
Rules applied: HEADROOM -> ... ; RETAINED_FORMAT -> ...

## Deviations / failures
...
```

---

## 9. Out of scope for this repo until the pilot report exists

Full RQ3 runs, ScienceWorld replication, the rationale ladder, unified self-simulation regimes, base-vs-instruct beyond `t3`, presence-type corruption, lookahead arms, context separation. Log them in `notes/extensions.md` and stop.
