# Pilot report — 2026-10-28 — git unknown — FILED filed_date: 2026-10-28 | constants_commit: unknown | RETENTION_MARGIN: 1.0 | note: mock filing for pipeline dry-run; replace with real filing date before ICE Pilot A/B

## Tier 0
| check | result | pass |
| --- | --- | --- |
| t0_smoke | # STATUS — t0_smoke — PASS  games=5 observations≈155 STEP_CAP=30 Acceptance: ≥50 observations — PASS  | yes |
| t0_replay | # STATUS — t0_replay — PASS  replay checks=60 identical=60 Acceptance: 100% identical — PASS  | yes |
| t0_agent_determinism | # STATUS — t0_agent_determinism — PASS  mean step agreement=1.000 over 5 games  | yes |
| t0_parser | # STATUS — t0_parser — PASS  n=2000 unparseable=0 rate=0.0000 threshold=0.05 — PASS  | yes |
| t0_ckpt_accuracy | # STATUS — t0_ckpt_accuracy — PASS  valid_unseen n=25 determined_acc=nan (n=0) uncertain_acc=0.4 (n=25) — reported  | yes |
| t0_scoring_agreement | # STATUS — t0_scoring_agreement — PASS  agreement=100/100 (1.00) min=0.9 — PASS  | yes |
| t0_expert_labels | # STATUS — t0_expert_labels — PASS  expert labels available: yes; n_labels=25  | yes |

## Pilot A (RQ2)

### pilotA_transitions
# REPORT — pilotA_transitions

## pilotA_transitions

### n_total
360

### n_eligible
55

### n_states
11

### counts
- **(True, 'validity')**: 55


### pilotA_score
# REPORT — pilotA_score

## pilotA_score

### ELIGIBLE_SET_TARGET
1000

### PILOT_RULE_GOAL_CHANNEL
False

### reports
- **qwen3-8b-fewshot**: {'delta_p_agnostic': 18.180167349974955, 'ci': (18.180167349974955, 18.180167349974955), 'perm_p': 0.15384615384615385, 'icc': nan, 'A': {'A': nan, 'false_success_rate': 0.5454545454545454, 'false_failure_rate': nan, 'n_fail': 55, 'n_succ': 0}, 'n_agnostic': 55, 'n_revealing': 0, 'se_at_1000': nan, 'se_at_2000': nan, 'se_at_3000': nan}
- **w2w-qwen2.5-7b**: {'delta_p_agnostic': 9.090083674987477, 'ci': (9.090083674987477, 9.090083674987477), 'perm_p': 0.4915084915084915, 'icc': nan, 'A': {'A': nan, 'false_success_rate': 0.45454545454545453, 'false_failure_rate': nan, 'n_fail': 55, 'n_succ': 0}, 'n_agnostic': 55, 'n_revealing': 0, 'se_at_1000': nan, 'se_at_2000': nan, 'se_at_3000': nan}


## Pilot B (RQ3)

### pilotB_headroom
# REPORT — pilotB_headroom

## pilotB_headroom

### none
0.0

### oracle_discarded
0.0

### oracle_retained
0.0

### validity_only
0.0

### oracle_gain_pp
0.0

### validity_recovery
nan

### rule
PILOT_RULE_HEADROOM fired — RQ3 moves to ScienceWorld (or recheck if borderline)


### pilotB_consumption
# REPORT — pilotB_consumption

## pilotB_consumption

### decision_critical_rate
0.48333333333333334

### follow_rate
0.5166666666666667


### pilotB_eligibility
# REPORT — pilotB_eligibility

## pilotB_eligibility

### invalid_proposal_rate
0.9666666666666667

### valid_proposal_rate
0.03333333333333333

### expert_among_valid
1.0

### expected_steps
30.0

### p_favorable
0.10344827586206896

### p_unfavorable
1.0

### CORRUPTIONS_PER_EPISODE_TARGET
3

### derivation
p = target / (eligibility_rate * expected_steps)


### pilotB_retention_minirun
# REPORT — pilotB_retention_minirun

## pilotB_retention_minirun

### H_discarded
0.0

### H_retained
0.0

### I
0.0

### retention_shift
0.0

### rule
PILOT_RULE_RETAINED_FORMAT fired — switch to state persistence for full runs

### cells
- **favorable/discarded**: {'success': 0.0, 'mean_S': -31.0, 'mean_n_corrupted': 0.0}
- **favorable/retained**: {'success': 0.0, 'mean_S': -31.0, 'mean_n_corrupted': 0.0}
- **unfavorable/discarded**: {'success': 0.0, 'mean_S': -31.0, 'mean_n_corrupted': 30.0}
- **unfavorable/retained**: {'success': 0.0, 'mean_S': -31.0, 'mean_n_corrupted': 30.0}


## Deviations / failures
See per-run STATUS.md files under runs/.

## Note
If this report was produced with WF_MOCK_*=1, replace with ICE results before claiming pilot completion.
