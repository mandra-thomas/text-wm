"""Reviewer (TextCraft round 3, review 2), no scores read: review 1's failure scenarios against the amended code, and
the new defects suspected in review 2.

    PYTHONPATH=. python scripts/review2_textcraft_r3_checks.py
"""
import json
import tempfile
from pathlib import Path

from wf import constants as C
from wf.textcraft import r2_analysis as A
from wf.textcraft import r3 as R3
from wf.textcraft import r3_analysis as Z
from wf.textcraft import r3_gates as G

V = lambda **kw: {c: kw.get(c, "inconclusive") for c in A.CONTRASTS}  # noqa: E731

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # M1: a shard of the 27B model on two L40S (each job's own list, read from jobs.jsonl)
    d = tmp / "shard_00_of_04"
    d.mkdir()
    (d / "git_sha").write_text("c1\n")
    (d / "jobs.jsonl").write_text(json.dumps({"gpu": ["NVIDIA L40S", "NVIDIA L40S"], "git_sha": "c1"}) + "\n")
    pin = C.REVISIONS["Qwen/Qwen3.8-27B"]
    git = lambda args: "" if args[0] == "diff" else "c1"  # noqa: E731
    print("M1 two-L40S problems:", Z.provenance_problems({"qwen3.8-27b": [d]}, {"qwen3.8-27b": [{"model_revision": pin}]},
                                                       "c1", run_git=git))

    # M2: a core model whose filed gate record failed
    root = tmp / "runs"
    s = root / "textcraft_r3_score_ministral3-8b-it" / "shard_00_of_02"
    s.mkdir(parents=True)
    (s / "tokenization_checks.json").write_text(json.dumps({"ok": False, "loading_failed": "x"}))
    print("M2 model_state:", G.model_state(root, "ministral3-8b-it"))

    # M3 / S9: the Gemma slot from a filed smoke record
    sm = root / "textcraft_r3_smoke_gemma4-12b-it" / "shard_00_of_01"
    sm.mkdir(parents=True)
    (sm / "tokenization_checks.json").write_text(json.dumps({"ok": True}))
    low = [{"condition": "yn_aligned", "logp_plus": -3.0, "logp_minus": -3.0, "top_token": t, "top_token_special": sp}
           for t, sp in [("<|channel>", True)] * 6 + [("Crafted", False)] * 4]
    (sm / "scoring.jsonl").write_text("".join(json.dumps(r) + "\n" for r in low))
    print("M3/S9 low mass, structural tops:", G.gemma_slot(root)["slot"])
    words = [{**r, "top_token": "Could", "top_token_special": False} for r in low]
    (sm / "scoring.jsonl").write_text("".join(json.dumps(r) + "\n" for r in words))
    print("M3/S9 low mass, answer-like words (goes to competence):", G.gemma_slot(root)["slot"])
    for tok in ["\n", " ", "**", "<", "[", "-", "Yes", " no", "The", "Craft", "C"]:
        print(f"   structural({tok!r}) = {G.structural(tok, False)}")

# S1: one competent model, supported
v = {"a": V(parent_penalty="confirmed below zero"), "b": V(), "c": V()}
comp = {"a": True, "b": False, "c": False}
print("S1 H3-P:", Z.h3p_reading(v, comp, {t: False for t in v}, Z.r2_contemporary(v, comp))["overall"])

# S12: what the code-identical rule accepts
git2 = lambda changed: (lambda args: "\n".join(changed) if args[0] == "diff" else "")  # noqa: E731
for changed in (["notes/x.md"], ["wf/config/textcraft_r3_envcheck_plan.json"], ["tests/test_x.py"], ["pyproject.toml"],
                ["runs/FILED.txt"], ["wf/config/textcraft_r3_analysis.yaml"]):
    print("S12 code_identical with", changed, "->", G.code_identical("head", "tag", git2(changed)))

# Envcheck: plan runnable on an all-L40S round 2
print("M4 shards' seeded records (round-2 partition):", R3.ENVCHECK_MIN_COMPARABLE)
