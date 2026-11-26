# Lexsem diagnostic: deviations after the freeze

Frozen design: `notes/lexsem_design.md` at tag `lexsem-design-frozen-2026-11-26` (commit 46e255b). Each deviation is dated and committed before the scores it affects are read (design section 13, step 6).

## D1 (2026-11-26): no recognition check for the Llama-3.1-8B base arm

**What happened.** The full-run job for the Llama-3.1-8B base arm (`llama8b_base_plain`, job 7353903) stopped before scoring any item. The frozen scorer runs the recognition check of design section 5 (rq1x's naming question for the descriptions, a forced choice for the aliases) in the model's chat template, and the Llama-3.1-8B base tokenizer has no chat template. The Qwen3-8B-Base tokenizer ships the Qwen3 chat template, so that base arm ran its check. The smoke runs skip the recognition check, which is why they did not show this.

**What is done.** No scorer file is changed: the post-training pilot compares its cells with these scores, so the scorer stays as tagged. The frozen scorer skips the check when the arm's check file exists. An empty check file (`runs/lexsem/llama8b_base_plain_check.json` containing `[]`: no check was run) is placed before the arm is resubmitted at the tag. The frozen analysis reads an empty list as "no item recognised", so for this arm every description and alias contrast is reported as "not assessable: 0 items passed the recognition check" (gate 6). Every other contrast of the arm is unaffected.

**Alternatives not taken.** One alternative was to run the check in plain text for this arm only. That would be a new procedure added after the freeze, for a model whose instruction following the design does not rely on. The other was to let the analysis fall back to Qwen3-8B's recognition checks, which would gate this arm on another model's recognition.

**Consequence for reading.** For the Llama-3.1-8B base arm, the semantic-identity contrasts that use a description or an alias are not read. Account readings for that arm rest on the name, homonym, position and relevance contrasts only, and the rules report the missing contrasts as undecided.

**Outcome (added 2026-11-26, after the analysis).** The rescored arm (job 7360053) completed at the tag. It then failed its competence gate (AUC 0.50 on the first token, 0.52 on the full two-candidate score), so none of its readouts is read, and D1 changes no reading (`notes/lexsem_results.md`, section 6).
