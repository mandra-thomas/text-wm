# Post-training pilot: independent review 1 (2026-11-26) and what was done

One adversarial review of the design note (`notes/posttraining_design.md`, commit fbc4d64) and the code (`wf/posttrain.py`, `wf/posttrain_score.py`, `wf/posttrain_analysis.py`, the job scripts and tests), by a reviewer given the design and the code but not the author's conclusions, before the freeze and before any score. The reviewer read files and ran the tests only. Each point is summarized below with the change made; line references are to the reviewed commit.

## Must-fix

| # | Point | What was done |
|---|---|---|
| 1 | The recognition filter rebuilt each item's (object, control) pair with a word-count tie-break while the scorer used token counts, so some items' pairs differed; one pair map was shared by both lineages. | The check was redesigned (point 14): it is now per object, written by each arm with the object as key, and every scored record carries its goal object; the analysis filters by the record's object. No pair is rebuilt. |
| 2 | A `--no-reuse` rescore could never pass the analysis (the merge ignored the arm's own `reuse` setting and then refused it). | The merge is skipped for an arm scored with `reuse: false`; tested. |
| 3 | The decision-token share $\pi$ was read without the competence gate. | $\pi$ now needs both arms' first-token and full readouts to pass their gates; ungated it reads "not interpretable". |
| 4 | Holm was computed but not applied to $C$, $B$ and $\pi$; a missing p counted as a pass; the continuation and step 0 got unlabelled "higher/lower"; missing arms shrank the family. | The family is fixed at 35 readings, a missing or ungated reading enters with p = 1; "higher/lower" for $C$, $B$, $\pi$ and the standardized goal effect needs Holm p < 0.05; every reading outside the family is labelled "descriptive". |
| 5 | A missing floor counted as zero; the seven-example arms had no floor arms; "the same" and "suppressed" never checked five floors. | A missing floor makes a reading "not readable (floor not measured)"; floor arms exist for the seven-example arms; "the same" (±0.15), "suppressed" (±0.10) and share "the same" (±0.25) need the margin to be at least five floors, else "inconclusive (margin under five floors)". |
| 6 | Lexsem's success scoring stops at the first success whose observation does not start with "You" (56 of 292), so reused success cells would be partial and the Qwen base's gate biased; reused cells were not checked for completeness. | Success cells are never reused (the pilot scores all of them, about 900 sequences per Qwen arm); every reused failure cell must exist for all 450 failures, else the reuse is refused. The lexsem crash was reported to the coordinator. |
| 7 | The reference rule required a positive goal effect, so a reversed effect could never have its shares read, against the no-direction scope. | Made two-sided: the interval excludes zero and its end nearest zero is at least half the absolute estimate; tested. |

## Should-fix

| # | Point | What was done |
|---|---|---|
| 8 | The seven-example fallback was applied only to the primary pairs. | Applied to the secondary pairs; format effects and stage-by-format interactions of a fallback lineage are marked "not read". |
| 9 | The run-time token-identity gate across a lineage did not exist. | Every scored sequence now stores a hash of its token ids; the analysis requires identical hashes per item, condition and format across the plain arms of a lineage with the same examples, and in their floor rescorings. |
| 10 | Lexsem provenance unverified; the bit-identity check covered only five later items. | The merge requires lexsem's recorded commit and that every scorer file is identical between that commit and the analysis commit (`git diff --quiet`); the check covers five items in each stratum. |
| 11 | The rank gain among all twelve conditions moves when only mention conditions move. | The robustness rule now uses the rank among the three task lines (real, swapped, neutral); the twelve-line rank is descriptive. |
| 12 | The standardized effect mixes the mean shift with item spread. | Each arm's item standard deviation and the ratio between compared arms are reported. The sign-agreement rule is kept as approved; with nats, net share and rank gain all required to agree, a change carried by the spread alone is read as scale-dependent. |
| 13 | The competence gate pooled step-0 and later failures against successes that all have a history. | The gate compares the 300 later failures with the successes; step-0 discrimination is reported descriptively. |
| 14 | The two-way recognition check against a different-class control was nearly uninformative. | Replaced: all 55 names scored after the description and after a neutral phrase; the object must rank in the top three by the difference. |
| 15 | The Tülu chat arms run without BOS, possibly unlike training. | Checked: AI2's Tülu 3 8B SFT script does not pass open-instruct's `--add_bos`, and the template renders none, so training most likely had none; stated in the note as checked on the script, not on tokenized data. The shipped template is kept. |
| 16 | 2,000 resamples are too coarse for the first Holm step. | 10,000 resamples. |
| 17 | Parts of the note were not implemented (R, few-shot items apart, described-goal standardized effect, quartile ranges, rule 4's environment-dependent label, floors for the pairs $C$ and $B$ use). | Implemented: $R$, the analysis without the few-shot-object items, the described goal's standardized effect, quartile ranges of the neutral score, the "environment-dependent" label (A40 shift of an arm's standardized goal effect above a fifth of the margin), and floors for the denial and listed-word pairs in the plain arms. |
| 18 | The GPU type was checked only at analysis. | The scorer refuses to start on the wrong GPU type. |
| 19 | The single-commit rule forbade any legitimate rescore. | Replaced by code identity: every arm's commit must have scorer files identical to the analysis commit's. |

## Minor

| # | Point | What was done |
|---|---|---|
| 20 | Step-0 readings use floors measured on later items. | Stated with every step-0 reading. |
| 21 | The described goal was not in the name-copy check. | The described goal and the neutral task must add zero copies of the name's tokens. |
| 22 | The plain bridge prompt has no examples and no "Next observation:" cue. | Stated; the bridge is descriptive, with its own discrimination reported. |
| 23 | The continuation's standardized effect was outside Holm and the note; its gate was unused. | It is labelled descriptive; the unused gate was removed. |
| 24 | Tülu's 8 extra embedding rows enter the normalizer. | The smoke records the largest probability on ids from 128,256 up. |
| 25 | The smoke repeated the recognition check for every arm. | Once per job. |
