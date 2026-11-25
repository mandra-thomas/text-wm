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
