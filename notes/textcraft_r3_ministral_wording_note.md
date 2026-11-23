# TextCraft round 3: Ministral 3's two-candidate reply format (note for the results, 2026-11-23)

This is a format observation from the smoke run, for the results write-up only. It changes no code, gate or rule, and
no arm difference or effect estimate was computed for it.

Ministral-3-8B-Instruct's smoke run (`textcraft_r3_smoke_ministral3-8b-it`, job 7252011, 115 two-candidate scores)
puts little probability on the exact wording of the two replies:

- **First-token mass.** The mass on the two replies' first tokens ("C" from "Crafted", and "Could") has a median of
  0.49, with a 10th to 90th percentile range of 0.13 to 0.91. The median probability of "C" is 0.078 and of "Could"
  0.226.
- **The rest of the reply.** Given its first token, the rest of each reply has a median probability of 0.13 (success
  reply) and 0.28 (failure reply).
- **Both whole replies together** have a median probability of 0.147.

Comparisons, reading the same recorded fields only:

- **Qwen3.5-9B (round 3):** first-token mass 0.99, rest of reply 0.93 and 0.98.
- **Mistral-7B-v0.3 (round 2, the other member of B3's pair):** first-token mass 0.97 (10th percentile 0.82).
- **Gemma-2-9B-it (round 2):** first-token mass 0.97. Gemma 2 has the same "C" + "rafted" split, so the split
  alone does not explain Ministral's low mass.

The yes/no format is unaffected: Ministral's median P(Yes) + P(No) is 0.95. The most probable next token on its yes/no
prompts is "No" or "Yes" every time.

The design has no rule on two-candidate mass. Competence reads the yes/no mass only (design 7.1), and round 2's
analysis reports the median first-token mass under the neutral line descriptively. The two-candidate L is the log-odds
between the two whole replies, so probability placed on other wordings does not shift L directly. It does mean that a
large part of Ministral's L is a wording component, and a contrast stays free of wording only if that component does
not move with the goal line.

For the results, two places should show this:

- **Ministral's own model block.** Report the finding beside its first-token / tail split, which shows how much of
  each contrast the later tokens carry.
- **The B3 pair Mistral-7B against Ministral 3.** Even after standardization, this pair compares a model that follows
  the reply format (first-token mass 0.97) with one that mostly does not (0.49).
