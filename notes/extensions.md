# Extensions (archived 2026-11-20)

**Archived.** This list was written before the pilots, and is kept unchanged below as a record. It is no longer the place for new ideas or a statement of what is left to do. As of 2026-11-20:

- **Done:** confirmatory RQ2 on `valid_unseen` with the 0.5-nat threshold and Holm (`rq2_results.md`); Llama-3.1-8B-Instruct as co-primary; the three few-shot sets with their SD (`rq1_results.md`); both mitigations, the ignore-the-goal instruction and the neutral task (`rq1_results.md`); RQ3 on the 134 solvable `valid_unseen` games rather than 274, under the proposal's contingency (`rq3_results.md`, `deviations.md`).
- **Not triggered:** the ScienceWorld fallback for RQ3, because the headroom rule became advisory (`deviations.md`). A ScienceWorld replication of RQ2 is being developed separately and is not in this repository.
- **Not run:** no `t3_*` run exists, so `t3_base_vs_instruct` and the other Tier 3 ideas were not done; the rationale ladder, presence-type corruption, lookahead arms and the 50-game natural-error check were not done either.
- **Since added, exploratory:** object-free goals, thinking mode, the dissociation test, the counterfactual task and yes/no question, the mechanism diagnostic and the Qwen3 size series (`exploratory_results.md`).

---

*Original text (pre-pilot):* Log ideas here. Do **not** implement until `runs/PILOT_REPORT.md` is written.

## Implementation notes (already decided)

- **Solvable-only AgentGym indices** — not an extension. Protocol §1/§5.1; client filter in `wf/env/server.py`. Do not re-open unless upstream AgentGym ships a solvable-only list.

- Full RQ3: 274 games × 3 error schedules × discarded/retained
- ScienceWorld replication if `PILOT_RULE_HEADROOM` fires
- Rationale ladder / unified self-simulation regimes
- Base-vs-instruct beyond `t3_base_vs_instruct`
- Presence-type corruption; lookahead arms; context separation
- Mitigations: instruction to ignore goal; remove goal from prompt
- Optional 50-game real-WM (natural errors) check
- Confirmatory RQ2 on `valid_unseen` with ΔL ≥ 0.5 nats + Holm (prereg PDF)
- Llama-3.1-8B-Instruct co-primary scoring (prereg)
- Three few-shot sets SD reporting for RQ1–2
