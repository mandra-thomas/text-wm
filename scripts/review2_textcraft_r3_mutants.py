"""Reviewer's mutants for TextCraft round 3, review 2 (the code added after review 1), run with the project's strict
named-kill rule (scripts/mutation_check.py: killed only when pytest exits 1 and names a failing test).

    cd <worktree> && <venv python> scripts/review2_textcraft_r3_mutants.py [i/n]
"""
import importlib.util
import sys
from pathlib import Path

WORKTREE = Path.cwd()
spec = importlib.util.spec_from_file_location("mutation_check", WORKTREE / "scripts" / "mutation_check.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

G, Z, K = "wf/textcraft/r3_gates.py", "wf/textcraft/r3_analysis.py", "wf/textcraft/r3_think.py"
mc.MUTANTS = [
    ("rv2-1 structural ignores the special-token flag", G,
     "return bool(special or not s or s[0] in", "return bool(not s or s[0] in"),
    ("rv2-2 format trigger on low mass alone", G,
     'if diag["median_mass"] < MASS_MIN and diag["structural_share"] > STRUCTURAL_SHARE_MIN:',
     'if diag["median_mass"] < MASS_MIN or diag["structural_share"] > STRUCTURAL_SHARE_MIN:'),
    ("rv2-3 model state ignores smoke gate failures", G,
     'for kind in ("smoke", "score"):', 'for kind in ("score",):'),
    ("rv2-4 code paths omit pace", G,
     'CODE_PATHS = ("wf/", "scripts/", "pace/")', 'CODE_PATHS = ("wf/", "scripts/")'),
    ("rv2-5 shard typed despite mixed job GPUs", G,
     "if not lists or any(x is None for x in lists) or any(x != lists[0] for x in lists):",
     "if not lists or any(x is None for x in lists):"),
    ("rv2-6 golden-token disagreement does not fail the environment check", Z,
     'if any(s == "fail" for s in states) or not all(tok_ok):', 'if any(s == "fail" for s in states):'),
    ("rv2-7 a failed environment check keeps round 2's scores", Z,
     '"fail": "rescore"', '"fail": "round2"'),
    ("rv2-8 envcheck records compared across GPU types", Z,
     'and old_gpu[r["id"]] == new_gpu[r["id"]]]', "]"),
    ("rv2-9 a not-scored model counts as competent", Z,
     '"competent": {c: False for c in CONTRASTS},', '"competent": {c: True for c in CONTRASTS},'),
    ("rv2-10 cap needs more than the budget", K,
     "n_new >= MAX_REASONING", "n_new > MAX_REASONING"),
    ("rv2-11 freeze commit not enforced at job start", G,
     "if not code_identical(head, tag, run_git):", "if False:"),
    ("rv2-12 b_draws accepted outside mock", Z,
     'if cfg.get("b_draws") not in (None, B_DRAWS) and not G.mock():', "if False:"),
]
sys.exit(mc.main())
