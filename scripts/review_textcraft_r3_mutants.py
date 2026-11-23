"""Reviewer's own mutants for TextCraft round 3, run with the project's strict named-kill rule
(scripts/mutation_check.py: killed only when pytest exits 1 and names a failing test).

    cd <worktree> && <venv python> <this file> [i/n]
"""
import importlib.util
import sys
from pathlib import Path

WORKTREE = Path.cwd()
spec = importlib.util.spec_from_file_location("mutation_check", WORKTREE / "scripts" / "mutation_check.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

A, S, K = "wf/textcraft/r3_analysis.py", "wf/textcraft/r3_score.py", "wf/textcraft/r3_think.py"
mc.MUTANTS = [
    ("rev1 standardized SD with n instead of n-1", A,
     "pooled = math.sqrt(max((q - n * m * m) / (n - 1), 0.0))", "pooled = math.sqrt(max((q - n * m * m) / n, 0.0))"),
    ("rev2 envcheck goal effect over every chosen record", A,
     'mf = [r for r in chosen if r["kind"] == "main" and r["population"] == "main"]', "mf = list(chosen)"),
    ("rev3 envcheck yes/no share criterion off", A,
     'checks["yes_no_share"] = abs(rec["share_diff"]) <= ENV_RULE["yes_no_share_diff_max"]',
     'checks["yes_no_share"] = True'),
    ("rev4 thinking unforced subset keeps forced traces", A,
     'if subset == "all" or not s.get("forced")}', "if True}"),
    ("rev5 thinking batch seed constant", K,
     "key = f\"{part[0][0]['id']}|{part[0][1]}\"", 'key = "same"'),
    ("rev6 provenance accepts mixed revisions", A,
     "if revs and revs != {want}:", "if revs and want not in revs:"),
    ("rev7 B3 family on 2,000 resamples", A,
     'B=int(cfg.get("b_draws", B_DRAWS))', "B=DESC_DRAWS"),
    ("rev8 attention agreement on one item", S,
     '[:rule["n_items"]]', "[:1]"),
    ("rev9 envcheck accepts rescoring shards on mixed GPUs", A,
     "want = gpu_of(new[0][0]) if new and len(new_gpus) == 1 else None",
     "want = gpu_of(new[0][0]) if new else None"),
    ("rev10 thinking keeps text after the closing marker", K,
     "return text[: text.index(END_THINK) + len(END_THINK)], False", "return text, False"),
    ("rev11 thinking jobs not in record-and-arm order", K,
     'return sorted(jobs, key=lambda x: (x[0]["id"], x[1]))', "return jobs"),
]
sys.argv = [sys.argv[0]] + sys.argv[1:]
sys.exit(mc.main())
