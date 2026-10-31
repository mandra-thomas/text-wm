# Extensions (out of scope until pilot report exists)

Log ideas here. Do **not** implement until `runs/PILOT_REPORT.md` is written.

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
