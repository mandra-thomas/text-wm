"""Controlled Bayesian environment (notes/bayesenv_design.md).

A six-box house in which a key may be hidden. Every prompt states how the house was set up, so every prompt has one
exact posterior probability that the queried action succeeds. The package holds the world and its exact posterior
(`world`), the prompt texts and item surfaces (`render`), the frozen design (`design`), the GPU scorer (`score`) and
the pre-specified analysis (`analysis`).
"""
