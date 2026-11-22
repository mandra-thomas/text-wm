# τ-bench authorization experiment: independent review, round 1 (2026-11-22)

Reviewed: branch `tau-auth` at `d4cc451` (design `6dd0026`, build `6bc8a41` and `069ff00`). Read: `notes/tau_auth_design.md`
with its implementation record, `wf/tau/auth.py`, `wf/tau/auth_token_check.py`, the four `wf/config/tau_auth_*.yaml`,
`pace/tau_auth_score.sbatch`, `pace/tau_auth_tokcheck.sbatch`, `tests/test_tau_auth.py`, `notes/tau_auth_tokcheck.json`,
the `tau-auth` mutants in `scripts/mutation_check.py`, and for context `notes/tau_results.md`, `notes/tau_metrics.json`,
`notes/tau_dx.md` (branch `tau-dx`), `wf/tau/items.py` (how the few-shot examples are built), `wf/tau/pipeline.py`,
`wf/analysis/stats.py` and `wf/scoring/logodds.py`. Nothing was scored and no ICE job was run.

**Verdict: not yet ready to freeze.** Two must-fix items, both in how a removal is read, not in what is scored. Neither
needs a new forward pass: the jobs already scored contain what the fixes need. The code computes what the note says,
and the confirmatory code is unchanged. The 66 tests pass, but 6 of my 12 mutants survived with real gaps
(clustering of the inference, the positive control's CI, two prompt-check branches, the scorer's stale-record
refusal; should-fix S6), and one Holm edge case departs from the note (S1).

Classification: **must-fix** blocks the freeze; **should-fix** should be done before the freeze but would not by itself
invalidate a result; **note** is for the record.

## What the experiment asks, in one paragraph

On the failing call, the confirmatory goal effect (aligned minus swapped task line) splits into a naming component
$M$ (the same contrast between two bare-question lines, which name an order but request nothing) and a request
component $W$ (what the write request adds). Each is split again across prompt arms into the part a "task-blind"
backend description removes, $Z = X^{\text{main}} - X^{\text{blind}}$, and the part that persists,
$R = X^{\text{blind}}$. A removed $W$ with a persisting $M$ is read as a belief that the backend refuses unrequested
writes (reading (b)); a persisting $W$ as a task-relevance bias (reading (a)). The naming component is the control:
a general suppression of task effects by the task-blind arm should remove $M$ too.

## Must-fix

### M1. The general-suppression caveat fires only when the naming removal is itself shown material, so the most likely outcomes are read as an authorization belief without it

**Where.** `wf/tau/auth.py:501` (`model_reading`): the caveat is appended only if `m in (AUTHORIZATION, BOTH)`. Design
§6.5 has the same gap: it lists the caveat for "authorization-like / authorization-like" and "authorization-like /
absent", but the rows "both / any" and "part shown / any" say nothing about suppression.

**Why it matters.** The control logic of §1.2 is that a removal of $W$ counts as authorization-like only if the arm
did not also remove $M$. The code asks the opposite question: it adds the caveat only if a removal of $M$ is
*demonstrated* (at least $\delta$, CI above 0, Holm $p < 0.01$). When $M$'s removal is merely not excluded, which is
the usual state of a smaller contrast, the reading states the authorization interpretation without qualification.

**Concrete scenario 1 (Qwen, proportional suppression).** Suppose the task-blind statement and its extra 2,000 tokens
damp every task effect by about 35%, with no belief involved. With Qwen's confirmatory sizes ($W \approx 2.17$,
$M \approx 1.81$), that gives $Z_W \approx 0.76$ and $Z_M \approx 0.63$. With realistic intervals, $Z_W$ is material
(at least 0.72, CI above 0), $R_W$ is material, $Z_M$ is below 0.72 but its CI is not inside $(-0.72, 0.72)$, and
$R_M$ is material. I ran this through the code (`component_category` and `model_reading`): $W$ is "both", $M$ is
"relevance-like part shown", and the reading is "Part of the request component (removed part +0.76 [+0.50, +1.00]
nats) behaves like an authorization belief and part ... like a relevance bias", with no caveat. The data are exactly
what a uniform suppression predicts.

**Concrete scenario 2 (Llama, no naming component).** Llama's $M$ is absent (−0.03), so §6.5 and §11 say a removal of
its $W$ cannot be separated from a general suppression. The code keeps that caveat only for $W$ "authorization-like".
For $W$ "both" or "authorization-like part shown" with $M$ absent, which the power table makes likely for any partial
removal, the reading again says "behaves like an authorization belief" with no caveat (checked in the code).

**Fix (no new data).** Append the caveat whenever $W$'s removed part is shown (authorization-like, both, or
authorization-like part shown) unless $M$ is "relevance-like" (that is, $Z_M$ is shown negligible). For $M$ absent,
always append the no-control sentence. Write the rule into §6.5, and add a test for each of the two scenarios above.
I would also report, as a prespecified descriptive line beside the reading, the share of each component that the arm
removes, $Z_W / W^{\text{main}}$ and $Z_M / M^{\text{main}}$, with bootstrap intervals from the same resamples: a
uniform suppression predicts the two shares are equal, a request-specific belief predicts the first larger. It
decides nothing, but it is the comparison a reader will want.

### M2. The demonstrations cover write-request lines only, so copying the demonstrated pairs produces the headline "(a) for naming, (b) for the request" pattern, and the naming control cannot catch it

**Where.** Every few-shot example in every arm has a write-request task line (`wf/tau/items.py:150-181`; `task` and
`task_own` are both `task_line(...)`). The two minimal pairs that arm (iii) adds, E2/E2' and E3/E3'
(`wf/tau/auth.py:121-124`), each contrast "request on this order" with "request on another order", which is the
aligned-versus-swapped contrast $G = W + M$ evaluated on request lines. No example in any arm has a question or
neutral task line. The closing paragraph mentions those lines in words; the demonstrations do not.

**Why it matters.** The design's limit (§1.2) is that a non-belief bias which instruction and demonstration can
suppress would be classed as authorization-like, and it relies on the naming component to catch that. But a
suppression produced by the demonstrations acts where the demonstrations resemble the scored prompt. If the model
learns from the pairs "on request lines, own-order and other-order give the same output", the swapped line rises
toward the aligned line while the two question lines do not move. Then $Z_M \approx 0$ (naming "relevance-like") and
the whole change lands in $Z_W$: the AB reading, "the write request's extra effect disappears ... read as a belief
that the backend refuses writes the user did not request". The naming control is blind to this because the
manipulation never touches question lines.

Worse, the complete form of that copying is labelled as authorization. If the swapped line rises all the way to the
aligned line in arm (iii), then $W^{\text{blind}} = (al - ma) - (al - ms) = -M^{\text{blind}}$, so $R_W \approx -1.8$
for Qwen. `component_category` has no case for a persisting part significantly opposite in sign (an overshoot): $Z_W$
material plus $R_W$ "not material, not negligible" returns "authorization-like part shown" (`auth.py:460`; checked).

**The data already discriminate.** An authorization belief, as §1.1 states it, refuses every write the task does not
request: under arm (i) it penalises the swapped, both question and the neutral lines alike relative to aligned. Its
removal therefore lives in the aligned half of $W$, the term $al - ma$, together with an equal drop of
aligned minus neutral, and leaves $sw - ms$ and $sw - ne$ unchanged. Copying of the request-line pairs does the
opposite: it lives in the swapped half, $sw - ms$, raises swapped minus neutral, and leaves $al - ma$ and
$al - ne$ in place. Writing

$$Z_W = Z_{al-ma} - Z_{sw-ms},$$

the two halves are per-item contrasts of jobs already scored (aligned − neutral and swapped − neutral are already in
the `per_arm` block). In the confirmatory Qwen numbers the aligned half is $+1.42 - (-0.17) = +1.59$ nats and the
swapped half $-2.56 - (-1.98) = -0.58$, so $W = 1.59 + 0.58 = 2.17$. The design's own AB prediction (§5: $G$ falls to
about $M$, and swapped − neutral to about −1.98) has both halves vanish, the aligned one carrying most of the change;
the strict belief of §1.1 has the aligned half carry all of it. Copying of the request-line pairs instead leaves the
aligned half in place and drives the swapped half positive.

**Fix (no new data).** (1) Prespecify the two halves of $Z_W$ (and $Z_{AN}$, $Z_S$) with the prediction of each
account, as above, and make the reading conditional on them: the AB reading ("read as a belief that the backend
refuses writes the user did not request") is stated only if the aligned half shows a removal (estimate at least half
of $Z_W$, say, with CI above 0; the exact rule is the authors' choice but must be fixed now). If the removal is
carried by the swapped half, the reading becomes "removed on the demonstrated request lines; not separated from
copying of the demonstrated minimal pairs". (2) Give the overshoot its own label ("the task-blind arm removes more
than the component; reported, not interpreted") instead of "authorization-like part shown", and test it. (3) Say in
§11 that the demonstrations cover request lines only. A design alternative is to add a question-line minimal pair
to arms (ii) and (iii); it costs examples and tokens and is not needed if the reading is fixed.

## Should-fix

### S1. A non-computable primary test shrinks the Holm family instead of entering as p = 1

`holm_family` (`auth.py:543`) maps a *missing model* to p = 1 but passes a NaN permutation p of a present model to
`wf.analysis.stats.holm`, which drops NaN from the family (`stats.py:508`). Scenario: one model's blind-arm witness or
fail records are lost for a whole action, so its $Z$ test is NaN; the remaining seven tests are then adjusted over 7
instead of 8 (checked: 0.004 becomes 0.028 instead of 0.032). Design §6.3 fixes the family at eight. Fix: map NaN to
1.0 in `holm_family`, and add a test. (The existing mutant "Holm family drops a missing model" covers only the
missing-model path.)

### S2. Llama's pooled request component averages opposite class effects

The confirmatory analysis found Llama's goal effect +1.23 in the status class and −0.33 in the unavailable class
(`tau_dx.md`, by class). The 96/96 subset keeps that mix, so Llama's pooled $W$, $Z_W$ and $R_W$ can be averages of
opposite-signed class parts. Scenario: the task-blind arm removes the status-class lift and leaves the unavailable
reversal; pooled $R_W$ lands near 0 and the pooled category reads "authorization-like" while a reversed effect
persists in half the items. The per-class split is already computed (§7.6). Fix: prespecify that each model's reading
quotes the per-class $W^{\text{main}}$, $Z_W$ and $R_W$, and that when the per-class $W^{\text{main}}$ have opposite
signs (each CI excluding 0) the pooled reading carries "averages opposite class effects; see per class". No new data,
no per-class decision.

### S3. The comprehension context pulls toward No, and the gate decides the main model

The four questions are asked after the scored *failing* call (`jobs_for`, `auth.py:195`). In arm (iii), three of the
four correct answers (K1, K3, K4) are Yes, and the visible call next to the question is one that fails. Qwen's yes/no
leans to No (it said No to every status-class call in the confirmatory run, `tau_dx.md` §4). So Qwen could fail the
gate because it answers about the visible call, not because it misreads the backend, and then every Qwen category is
"inconclusive". The 24 items also ask the identical question; they test robustness across contexts, not 24
independent samples. Options before freeze: pose the questions with the history but without the "Next call" line, or
after the witness (which pulls toward Yes, with K2 still catching a Yes lean); or keep the design and state this risk
in §11. Either way, also decide now whether the smoke run's comprehension scores may be looked at (they are not
estimates of the contrasts, but changing the gate after seeing them would be a forking path).

### S4. The §6.5 reading table is not exhaustive

Combinations not in the table fall through to "Reported per component; no combined reading" in the code:
$W$ relevance-like with $M$ authorization-like, both or part shown; $W$ authorization-like with $M$ both or part shown;
$W$ absent. That default is reasonable, but the note should say so, and "$W$ relevance-like, $M$ authorization-like"
deserves a sentence because it is the one pattern where the control moves and the target does not.

### S5. The smoke run does not include the longest prompt

The longest scored prompt is 15,535 Qwen tokens (`tau-g-77cc59152d73`, arm (iii), yes/no), about 2,100 tokens longer than
the longest arm (i) prompt (13,471, the same prompts the confirmatory run scored). The confirmatory smoke deliberately added its longest items; this
smoke set is the first three items per class in hash order. `score_one_token_pair` computes logits at every position
(`wf/scoring/logodds.py:285`): about 4.7 GB in bf16 for Qwen at 15.5k (9.4 GB more if the installed transformers
upcasts to float32), plus 16.4 GB of weights and about 2.3 GB of KV cache. That fits 48 GB (L40S, A40) and an 80 GB
A100 comfortably; on a 40 GB A100, if ICE assigns one, it is near the limit. An out-of-memory on a fixed item would
recur on every resume. Fix: add the longest item to the smoke set (or record peak memory per job in the smoke
analysis). The time budget is fine: the largest shard is 55 items, about 2,700 passes per model, about 2.4 hours for
both models against a 4-hour limit, and every score is appended as it finishes.

## Notes

- **What the arms hold fixed, checked.** The system messages differ exactly as designed (byte-level guard, verified by
  reading `check_system_messages` and the golden hashes). The final user turn, and so the task line's position and its
  distance to the readout, is identical in every arm. The example next to the final turn is the same in every arm (E4
  in design order, E1 reversed). What changes: arm (iii) has 3 of 6 successes against 2 of 4 in arm (i) (both 50%), arm
  (ii) 2 of 6; own-order task lines rise from 2 of 4 to 4 of 6; cancel and return examples double; the distance from
  the system text to the task line grows by 1,700–2,100 tokens. Within-arm contrasts remove level shifts, but not a
  uniform damping of task effects, which is why M1 matters. I do not think an example- or length-matched arm (i) is
  needed once M1 and M2 are fixed: the suppression caveat and the half split carry the alternatives it would test.
- **Arm (iii)'s closing paragraph** quotes the wording of the neutral and question lines ("only a question about an
  order, or general help with the account"), so the statement covers question lines even though the examples do not.
- **Positive control.** Correctly specified and implemented: the witness's aligned − swapped effect, authorization
  arm minus task-blind arm, at least 1 nat with CI above 0 (`auth.py:522-527`). Its pass probability depends on the
  outcome it gates: under AB the witness's goal effect falls in arm (iii), so the control can pass through that fall
  alone; under A it must come entirely from arm (ii). That suits its purpose (it guards "nothing removed"), but a
  secondary witness goal effect of the authorization arm minus arm (i) would show whether arm (ii) itself moves
  predictions. For Llama, whose witness goal effect is +0.38 nats and whose matched-arm sensitivity was 0.35 nats, a
  1-nat control may well fail and leave Llama inconclusive; that is a risk, not an error.
- **Margins.** One third of each model's confirmatory request component is a defensible, prespecified scale. The
  confirmatory component includes the 36 returns, where Qwen's goal effect was +10.6 nats, so the subset's own request
  component may be smaller and the margin larger relative to it; for Qwen the 0.72-nat margin is 40% of the naming
  component. Fixed in advance, so acceptable.
- **Categories.** `component_category` is a cascade, so its categories are mutually exclusive and exhaustive by
  construction, and its order matches §6.4. "Material" and "negligible" cannot both hold unless a point estimate lies
  outside its own percentile interval.
- **Item population.** Restricting to status and unavailable items whose tools no example uses is sound (it avoids
  tool matching, confirmatory B6). The class dependence is handled by S2.
- **Engineering, verified.** The prompt checks run before any scoring and refuse on any problem (`auth.py:323-327`);
  the permutation flips all of a user's items together and the bootstrap resamples users (`lin`, `auth.py:405-407`);
  10,000 permutations for the eight primary tests, so the smallest p times 8 is below 0.01; the gates and the 80%
  rule (ties incorrect) match §3.5 and §6.1; resume keys on model, item, arm, action, readout and line, checks item
  hashes, and refuses a different commit or config (`wf/runs.py` `_check_resume`); the reproducibility check uses the
  confirmatory record names (`first`, `yn`). `git diff c737cd6 HEAD -- wf/` touches only `wf/tau/auth.py`,
  `wf/tau/auth_token_check.py`, four configs and five lines of `wf/runs.py`: the confirmatory scoring, analysis and
  statistics code is unchanged.

## Tests and mutants

`tests/test_tau_auth.py`: 66 passed (141 s, single process, at `d4cc451`).

Twelve mutants of my own, each in a fresh copy of the tree, run with `pytest -x tests/test_tau_auth.py`; killed only if
pytest exits 1 with a named failing test:

| Mutant (all in `wf/tau/auth.py`) | Outcome | Killing test |
|---|---|---|
| Permutation test flips items, not users (`cluster_ids=None` in `lin`) | survived | none |
| Bootstrap resamples items, not users (`lin`) | survived | none |
| "Material" accepts a CI touching 0 (`lo >= 0`) | survived | none (boundary; near-equivalent) |
| Positive control ignores its CI (only the estimate of at least 1 nat) | survived | none |
| Comprehension passes if any one question passes | killed | `test_a_failed_gate_makes_the_model_inconclusive[comprehension]` |
| "Relevance-like part shown" when R is uncertain and Z negligible | killed | `test_component_category_table[...-inconclusive]` |
| Comprehension asked after the witness | killed | `test_jobs_per_item` |
| Resume ignores the item hash of saved records | survived | none |
| Arms (ii) and (iii) never reversed | killed | `test_fewshot_order_is_one_per_item_in_every_arm` |
| Prompt check skips "final turn identical across arms" | survived | none |
| Prompt check skips "arm (i) equals the confirmatory prompt" | survived | none |
| NaN per-item differences kept | survived | none (near-equivalent: the scorer refuses a non-finite score) |

Four killed, eight survived. Two survivors are near-equivalent (boundary of a continuous CI; a NaN the scorer cannot
produce). The other six are real gaps, listed as S6 below. The code they mutate is correct today; the point is that
a regression in any of them would pass the suite.

### S6. Six untested properties (from the surviving mutants)

The authors' 58 mutants are all killed, but none of them touches these, and the suite does not catch them:

1. **User clustering of the inference.** Neither the permutation test nor the bootstrap in `lin` (`auth.py:405-407`)
   is tested for being clustered by user; `wf.analysis.stats` is tested, but not this call site. Add a synthetic set
   with a strong user-level effect and few users, where item-level and user-level intervals and p-values differ
   clearly, and assert the user-level ones.
2. **The positive control's CI condition.** Add a case with estimate at least 1 nat and a CI crossing 0 that must not
   pass (`auth.py:526`).
3. **Resume's stale-record refusal** at scoring time (`auth.py:335-336`): write a saved record with a wrong
   `item_hash` and assert the refusal. (The analysis' stale count is tested; the scorer's refusal is not.)
4. **Two prompt-check branches never fire in a test**: "final turn differs from main's" (`auth.py:253-254`) and
   "differs from the confirmatory prompt" (`auth.py:258-259`). Today the rendering makes both hold by construction,
   which is exactly why the guard needs a test that monkeypatches a deviation (for example an arm-dependent final
   turn, or a main-arm example change) and asserts the named problem, as the existing test does for the examples.

Also worth a test: S1's NaN-in-family case.

## Review commit

Recorded in the hand-off.
