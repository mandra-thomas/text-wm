# TextCraft round 3, deviation D-G2: the attention-agreement check is evaluated in float32

**Date and status.** 2026-11-25, approved by the user. It comes after the freeze tag
`textcraft-r3-design-frozen-2026-11-23` (0d6f460) and after the deviation tags of D-T1/D-G1 (98dfbbf) and D-T2
(313dc8d). It was written and committed before the round-3 analysis ran and before any round-3 score, contrast or
effect was read. It applies to round 3 only; wave B records its own deviation.

**Evidence.** `notes/numerical_sensitivity.md` on main (§4.7, Rule 2), and D-G1's diagnostic (job 7252801).

## The rule, before and after

**The frozen rule (design §4, 14.M3).** Each Gemma 4 job scores five items twice at its start, once with eager
attention and once with SDPA attention, in bf16. The five items are the shard's first main failures, under the aligned
two-candidate arm and the aligned yes/no arm. The job stops if the two attention implementations disagree by more
than wave B's limits: a mean of 0.10 nats on yes/no, a mean of 0.50 nats on two-candidate, or 1.0 nats on any single
score. Gemma 4 12B's smoke failed this check (yes/no 0.216). So did E4B's (yes/no 0.125). The fallback rule then gave
the Gemma slot to E4B, which was itself "not scored". The slot, and B3's Gemma pair, therefore read "not scored".

**Why the rule is changed.** The 0.10 limit is below one bf16 step at these models' readout logits. A bf16 step there
is 0.125 nats: the yes/no logits lie between 16 and 32. Every individual bf16 difference observed is a whole number of
steps, and in float32 the two implementations agree to 0.00001 nats (E4B: mean 0.000011, max 0.000027). The check
was meant to catch a wrong attention implementation. In bf16, at these logits, it measures rounding instead.

**The new rule (D-G2).** The criterion is the same function and the same limits, on the same five items and arms.
Only the precision changes: it runs on a float32 copy of the model.

- **Where it runs.** The registered run `textcraft_r3_attnf32_<tag>` calls the frozen `r3_score.attention_agreement`
  on a float32 copy, loaded from the same verified node-local snapshot with every setting of the model's frozen spec
  (strict loading included).
- **Its record.** The run files `attention_f32.json`, which records the dtype the model was actually loaded in. A
  filed record is final.
- **Where it is read.** The gates (`r3_gates.attention_f32`) read the attention agreement from that record for every
  Gemma 4 model: 12B, E4B and 31B. This applies whether the job's own bf16 check passed or failed, so the rule is the
  same for all three.
- **Inside jobs.** A Gemma 4 job at D-G2's commit reads the float32 record in place of running the bf16 check.
- **Hardware.** A float32 copy of 12B is about 48 GB and does not fit an L40S, and one of 31B is about 125 GB. So the
  check runs on A100 80 GB cards: one for 12B and E4B, two on one node for 31B. Scoring keeps each model's pinned GPU
  type.

**What does not change.** Every other gate, including:

- the strict loading check, the tokenization gate, the golden renderings, the weights manifests and the GPU types;
- the S9 format diagnostic of the fallback rule;
- the competence rule;
- every family, test, threshold, Holm size, margin and reading rule;
- the S5 rule that a filed gate record is final. The filed bf16 records stay as filed. D-G2 changes how their
  attention component is read, not the records.

## What it changes in round 3

**1. The Gemma slot.** The frozen procedure runs again, in its frozen order, with the attention agreement read in
float32.

- **Gemma 4 12B comes first.** Its filed smoke passed loading, the tokenization gate, the golden renderings and the
  weights check. It stopped only at its bf16 attention check, so it filed no score. The fallback rule also needs the
  smoke's yes/no rows for its format diagnostic (S9). So:
  - (a) `textcraft_r3_attnf32_gemma4-12b-it` files 12B's float32 check.
  - (b) If that check passes, `textcraft_r3_smoke2_gemma4-12b-it` reruns 12B's smoke. It uses the same 20 seeded
    records, every arm and the same gates, with the float32 record as its attention gate, and it records the top
    next token on each yes/no prompt. The filed smoke folder is left as filed (S5).
- **The slot.** `r3_gates.gemma_slot` reads the original smoke record, the float32 record and the rerun's record and
  rows. It gives:
  - Gemma 4 12B if none of the frozen triggers fires (loading, gate, float32 attention, or S9's format mismatch);
  - Gemma 4 E4B if any fires;
  - "waiting" until the records it needs are filed.
- **Expected outcome.** On the numerical evidence, 12B's float32 check is expected to pass, since every observed bf16
  difference is whole rounding steps. Then **Gemma 4 12B holds the slot as the frozen design intended**, unless its
  smoke rerun shows S9's format mismatch.

**2. If Gemma 4 12B holds the slot.**

- **Scoring.** It is scored by its core run `textcraft_r3_score_gemma4-12b-it`: two shards, each on one L40S (its
  pinned type), at D-G2's commit. That commit is needed because the frozen code would rerun the bf16 check inside the
  job and stop.
- **What it enters.** It enters family A3, and B3's Gemma pair compares Gemma-2-9B's serve2 rescoring with Gemma 4 12B.
- **The placement check.** The frozen run order also lists 12B's placement check as an exploratory format check
  (`textcraft_r3_placement_gemma4-12b-it`). It becomes runnable as frozen.
- **E4B.** E4B does not enter any family. D-G1's exploratory E4B block stays as it is, exploratory.

**3. If 12B's float32 check fails, or its rerun shows S9's format mismatch.**

- **The slot goes to E4B.** Its own state then follows its float32 record. The record is refiled by
  `textcraft_r3_attnf32_gemma4-e4b-it`, so that all three Gemma 4 models are judged by the same run. D-G1's diagnostic
  found 0.000011.
- **E4B is then scored by its core run** `textcraft_r3_score_gemma4-e4b-it`, at D-G2's commit.
- **Why not reuse D-G1's scores.** The existing E4B scores (jobs 7299374–75, run `textcraft_r3x_gemma4-e4b-it`) were
  produced the way a core run produces them:
  - the same scoring loop (`r3_score.score_shard`), prompts, arms, readouts and scorer;
  - bf16 with SDPA attention, at batch size 1, on one L40S, in serve2;
  - node-local weights checked against the manifest;
  - the same tokenization gate and golden renderings, with the attention check replaced by the float32 record, which
    is what D-G2 now does for every Gemma 4 job.

  They differ from a core run in three ways: their run id and folder, their commit (D-T1's 98dfbbf, whose code
  differs from the freeze commit's only outside the scoring path), and the job-start commit rule. The frozen analysis
  reads core scores only from `textcraft_r3_score_*` folders, at the freeze commit. **This deviation chooses to
  rescore under the core run id (about 1.4 GPU-hours) instead of adding a reading rule for the exploratory folder.**
  That keeps one provenance rule for every core score. The exploratory block stays exploratory either way.

**4. Gemma 4 31B (extension).** Its smoke passed the bf16 check (0.083), and its scoring is queued at the freeze tag
(jobs 7314883–86). Under D-G2 its attention component is read from `textcraft_r3_attnf32_gemma4-31b-it`, and its
state is "waiting" until that is filed. Its queued scoring is not touched. The scoring code at the freeze tag is the
same: the only difference is the bf16 check, which those jobs pass.

**5. B3's older Gemma scores.** These do not change: B3 reads the serve2 rescoring of Gemma-2-9B (jobs
7252999–7253000), because the environment check failed.

## Where the runs start, and the analysis

**Job start (S12).** The runs D-G2 needs may start at the commit of the annotated tag
`textcraft-r3-deviation-g2-2026-11-25`, or at a code-identical commit:

- the three `textcraft_r3_attnf32_*` runs;
- `textcraft_r3_smoke2_gemma4-12b-it`;
- `textcraft_r3_score_gemma4-12b-it`;
- `textcraft_r3_score_gemma4-e4b-it`;
- `textcraft_r3_placement_gemma4-12b-it`.

This holds only if that commit changes code only in the deviations' files (`r3_gates.G2_FILES`), and only from a clean
checkout.

- Every other run still refuses anything but the freeze tag's commit.
- D-T1's thinking and exploratory runs still need D-T1's commit. A thinking run cannot start at D-G2's.
- The scoring loop passes its run id to the check (D-G1), so a core run of any other model, or an environment-check or
  rescoring run, is refused at D-G2's commit. Tests show this.

**The analysis.** It runs once, as `textcraft_r3_analysis_dg2`, after every arm is filed, from a clean checkout at
D-G2's tag. It supersedes `textcraft_r3_analysis_dt2`: same keys, both values taken from the gates, and the D-T2 note's
rules for the run folder, the tags and the first-core marker apply unchanged. The bundle must carry four tags: the
freeze tag, D-T1's, D-T2's and D-G2's.

- **Commits it accepts:**
  - Gemma 4 12B's and E4B's score shards, and the placement check's, at the freeze commit or D-G2's;
  - the float32 checks and the smoke rerun, at D-G2's commit;
  - the thinking arm and the exploratory run, at D-T1's;
  - every other shard at the freeze commit only.
- **What it records.** It writes `metrics["deviation_g2"]`: the D-G2 commit, the three float32 records and each D-G2
  shard's commits.

## Jobs and cost (none submitted; the coordinator confirms first)

| Order | Run | GPU | GPU-hours (estimate) |
|---|---|---|---|
| 1 | `textcraft_r3_attnf32_gemma4-12b-it` | 1 A100 80 GB | about 0.3 |
| 1 | `textcraft_r3_attnf32_gemma4-e4b-it` | 1 A100 80 GB | about 0.2 |
| 1 | `textcraft_r3_attnf32_gemma4-31b-it` | 2 A100 80 GB, one node | about 1.0 |
| 2 | `textcraft_r3_smoke2_gemma4-12b-it` (if 12B's float32 check passes) | 1 L40S | about 0.2 |
| 3 | `textcraft_r3_score_gemma4-12b-it`, two shards (if the slot is 12B) | 1 L40S each | about 2.5 to 3.5 |
| 3 | `textcraft_r3_placement_gemma4-12b-it` (exploratory, frozen run order) | 1 L40S | about 0.3 |
| 3' | `textcraft_r3_score_gemma4-e4b-it`, two shards (only if the slot is E4B) | 1 L40S each | about 1.4 |

If 12B holds the slot, the total is about 4.5 to 5.5 GPU-hours, including the 31B check. If the slot goes to E4B,
the total is about 3 GPU-hours.

Steps 2 and 3 wait for the filed records of step 1. Each is read for gates and format only, before the next step is
submitted.

## D-G2a (2026-11-26): Gemma 4 31B's scoring is added

Approved by the user on 2026-11-26. This amendment was written and committed before any Gemma 4 31B score existed.
At that point no 31B score had been computed by any job, and no round-3 score had been read after the interim reading
of 2026-11-25.

### What the attention-agreement check is, and why it exists

A transformer can compute its attention step with different implementations. "Eager" attention is the plain,
step-by-step one. "SDPA" (scaled dot-product attention) is PyTorch's fused version, which is faster. Mathematically
the two give the same answer. Every round-3 model is scored with SDPA.

Gemma 4 is new in the software used here (transformers 5), so the design added a safety check at the start of each
Gemma 4 job. The job scores five items twice, once with each implementation, and stops if the two disagree by more
than set limits:

- on the yes/no question, more than 0.10 nats on average;
- on the two-candidate replies, more than 0.50 nats on average;
- on any single score, more than 1.0 nats.

The purpose is to catch an implementation that computes something different, which would make every score wrong.

### Why 31B passed at its smoke run but failed in every scoring shard

The check runs in bf16, the 16-bit number format all scoring uses. At these models' output sizes, bf16 can represent
a log-probability only to the nearest 1/8 nat. So the two implementations, which add up the same numbers in a
different order, land on slightly different rounded values. How large that difference is depends on which five items
are checked.

- **The smoke run** checked the first five main failures of its 20 smoke records. For those, the two disagreed by
  0.083 nats on yes/no, 0.348 on two-candidate and at most 0.68 on a single score. That is inside all three limits.
- **Each of the four scoring shards** checked the first five main failures of its own quarter of the items, so each
  checked different items. All four failed:

  | Shard | Yes/no mean | Two-candidate mean | Largest single score |
  |---|---|---|---|
  | 0 | 0.084 | 0.690 | 1.33 |
  | 1 | 0.125 | 0.809 | 1.73 |
  | 2 | 0.328 | 0.479 | 1.23 |
  | 3 | 0.147 | 0.420 | 0.62 |

  Loading, the tokenization gate, the golden renderings and the weights check all passed in every shard. The jobs
  stopped before writing any score.

D-G2 moves this check to float32, where rounding is negligible. In float32, E4B's two implementations agree to about
0.00001 nats. Its own float32 check (job 7347267) shows the same for 31B. The 31B scoring is submitted only if that
check passes.

### Why D-G2 missed it

D-G2's list of runs allowed to start at its commit (`G2_RUNS`) covered only the runs D-G2 itself required:

- the float32 checks;
- Gemma 4 12B's smoke rerun, scoring and placement check;
- E4B's core scoring.

D-G2 assumed that 31B's scoring, already queued at the freeze tag, would pass its bf16 check, as its smoke had. That
assumption was wrong, because every shard checks different items. At the freeze tag the frozen code reruns the bf16
check in every job, so 31B cannot be scored there.

### What D-G2a changes

- **`r3_gates.G2_RUNS`** now includes `textcraft_r3_score_gemma4-31b-it`. Those runs may start at the commit of the
  annotated tag `textcraft-r3-deviation-g2a-2026-11-26`. The commit must change code only in D-G2's files, and the
  checkout must be clean.
- **D-G2's runs** may start at either D-G2's commit or D-G2a's.
- **The analysis** (`r3_analysis.G2_FAMILY_TAGS`, `r3_gates.g2_commits`) accepts Gemma 4 31B's score shards at D-G2's
  or D-G2a's commit, as it already does for 12B and E4B. It records both commits.
- **The failed bf16 records of 31B** (checkout `text-wm-fix47`, jobs 7314883–86) stay filed as they are. They are not
  read by the analysis that is reported. The final analysis checkout keeps them under
  `runs/_superseded_bf16_textcraft_r3_score_gemma4-31b-it/`. Under D-G2 their attention component is replaced by the
  float32 record, and every other component passed.

### What D-G2a does not change

- **Scoring.** 31B is scored like every other model: bf16 with SDPA attention, batch size 1, its pinned GPU type (one
  A100 80 GB per shard, as chosen at its first submission), node-local weights checked against the manifest, and
  the same gates.
- **Rules.** Every family, test, threshold, margin, reading rule and the competence gate are unchanged, and so is
  every other gate.

### Impact on the study

- **Which tests change.** 31B is one of the three models of the extension family A3x. Its 10 tests (five contrasts,
  each tested for differing from zero and for equivalence to zero) are now padded at p = 1. Once scored they get
  real p-values, and A3x's family readings become computable once Qwen3.8-27B (filed) and OLMo 3 (filed) are joined
  by 31B: H3-P, R2 and replication for the extension family. 31B is in no B3 pair. It also enters the exploratory
  size comparison of Gemma 4 12B against 31B.
- **No reported verdict can be lost.** Each family keeps its fixed size of 30 tests. When a padded p of 1 is replaced
  by a real p-value, the Holm-adjusted p of every other test can only stay the same or fall.
- **Cost.** 4 shards, about 6.5 GPU-hours on A100 80 GB, projected from 31B's smoke at 0.55 s per forward pass. ICE
  has four A100 80 GB cards, and wave B is rerunning its own Gemma 4 31B on the same cards, so the jobs will queue.

### 31B's numerical floor, and which readings are labelled fragile

The bf16 disagreement between two equivalent implementations is itself a measurement of how much rounding moves a
single score for this model. That is larger for 31B than for the other models.

**The rule, stated before any 31B score exists.**

- **Per-score numerical SD (σ).** It is taken from the largest shard mean of the two-candidate disagreement, 0.81
  nats. That disagreement is the difference of two independently rounded values, so for a roughly normal difference
  the per-score SD is

  $$\sigma = \frac{0.81}{\sqrt{2}}\sqrt{\frac{\pi}{2}} \approx 0.72 \text{ nats}.$$

  All five confirmatory contrasts are on the two-candidate scale.
- **A contrast's numerical SD.** It follows the formula of `notes/numerical_sensitivity.md` (§4.8), with the arms
  treated as independent, which is conservative because the measured correlation between arms is about +0.1:
  - for Λ (990 items) and P (675 items), $\sqrt{2}\,\sigma/\sqrt{n}$;
  - for Dn and Dc, which compare 500 goal-role items with 496 ingredient-role items, $\sqrt{2}\,\sigma\sqrt{1/500 + 1/496}$;
  - for T, which uses four scores per item over 336 and 332 items, $2\sigma\sqrt{1/336 + 1/332}$.
- **The allowance** (Rule 3) is twice that SD. A 31B reading whose deciding interval ends within its allowance of its
  boundary is labelled **numerically fragile**. For a "differs" verdict the boundary is zero; for an equivalence
  verdict it is ± the margin.

| Contrast | 31B numerical SD (nats) | Allowance | Gemma 4 12B, same rule (σ ≈ 0.39, from its smoke's 0.444) |
|---|---|---|---|
| Λ | 0.032 | 0.065 | 0.035 |
| P | 0.039 | 0.078 | 0.043 |
| Dn, Dc | 0.064 | 0.129 | 0.071 |
| T | 0.111 | 0.22 | 0.12 |

**Other Gemma 4 readings.** The same rule labels Gemma 4 12B's readings, using its smoke's two-candidate
disagreement of 0.444. If 12B holds the Gemma slot, B3's Gemma pair also crosses environments for the older model.
Gemma-2-9B's serve2 rescoring is in the same stack as 12B, so section 5.2's bound of about 0.05 standardized units
applies there.

**What D-G2 does not settle.** D-G2's float32 check establishes that the implementations agree, so the attention
code is not wrong. It does not reduce the bf16 rounding in the scores themselves. That rounding is what these
allowances account for.
