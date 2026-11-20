"""Interpretability tooling for the goal-leak finding (design: notes/interp_design.md).

Plain PyTorch forward hooks on HF transformers decoder models (Qwen3, Llama); no nnsight. Modules:

- `model`: load a pinned model and find its layers, attention, MLP, final norm and unembedding.
- `hooks`: capture activations, patch them (per batch row), patch single heads, ablate a direction.
- `prompts`: the yes/no and first-token prompts of an item, the aligned/swapped token pairing, token groups.
- `lens`: logit lens and the readout lens (ridge from a layer's residual to the final readout).
- `patching`: the patching grid for one item pair and the fraction of the effect restored.
- `probes`: fact labels from the history, logistic probes with held-out games, the standardised belief shift.
- `directions`: difference-of-means task direction, random directions, AUC.
- `run`: stages that save one record per item and resume.

torch, transformers and scikit-learn are imported lazily, so the rest of `wf` never needs them.
"""
