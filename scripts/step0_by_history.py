"""Goal effect (aligned minus swapped task line) by history length, per model and readout,
from stored scores (exploratory, post hoc; 2026-11-25). No model is scored.

Inputs: a copy of the run folders `rq12_transitions_s0..s2`, `rq1v_score` and
`family_score_<model>` (families wave A), under RUNS (default ./runs).
Items: the 4,666 RQ2 primary items, i.e. the items that carry the two-candidate
aligned and swapped scores; the yes/no rows are restricted to the same items.
Intervals: 95% game-clustered bootstrap, 1,000 resamples, seed 1.
Output: notes/step0_by_history_metrics.json (or the path given as the first argument).
"""
import collections, glob, json, os, random, sys

R = os.environ.get("RUNS", "runs")
OUT = sys.argv[1] if len(sys.argv) > 1 else "notes/step0_by_history_metrics.json"

hist = {}
for s in range(3):
    for line in open(f"{R}/rq12_transitions_s{s}/transitions.jsonl"):
        r = json.loads(line)
        hist[(r["game_id"], r["id"])] = r["history_len"]


def load(pattern, model=None):
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line)
            if model and r["model"] != model:
                continue
            d[(r["game_id"], r["transition_id"])][r["condition"]] = r["L"]
    return d


def bin_of(h):
    return "0" if h == 0 else "1-3" if h <= 3 else "4-6" if h <= 6 else "7+"


def boot(by_game, B=1000, seed=1):
    games = sorted(by_game)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        xs = []
        for g in rng.choices(games, k=len(games)):
            xs += by_game[g]
        means.append(sum(xs) / len(xs))
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]


SOURCES = {
    "gemma2-9b-it": (f"{R}/family_score_gemma2-9b-it/shard_*/scoring.jsonl", None),
    "mistral-7b-v0.3": (f"{R}/family_score_mistral-7b-v0.3/shard_*/scoring.jsonl", None),
    "llama3.2-3b": (f"{R}/family_score_llama3.2-3b/shard_*/scoring.jsonl", None),
    "llama3.1-8b": (f"{R}/rq1v_score/shard_*/scoring.jsonl", "llama3.1-8b-fewshot"),
    "qwen3-8b (with the empty think block)": (f"{R}/rq1v_score/shard_*/scoring.jsonl", "qwen3-8b-fewshot"),
}
READOUTS = {"yes/no": ("yn_aligned", "yn_swapped"), "two-candidate": ("aligned", "swapped")}

result = {"_about": __doc__.strip(), "models": {}}
for name, (pattern, model) in SOURCES.items():
    d = load(pattern, model)
    primary = {k for k, c in d.items() if "aligned" in c and "swapped" in c and k in hist}
    result["models"][name] = {}
    for readout, (a, b) in READOUTS.items():
        by_bin = collections.defaultdict(lambda: collections.defaultdict(list))
        for k in primary:
            c = d[k]
            if a in c and b in c:
                by_bin[bin_of(hist[k])][k[0]].append(c[a] - c[b])
        rows = {}
        for bn in ("0", "1-3", "4-6", "7+"):
            g = by_bin[bn]
            v = [x for xs in g.values() for x in xs]
            if not v:
                continue
            lo, hi = boot(g)
            rows[bn] = {"mean_nats": round(sum(v) / len(v), 3), "ci95": [round(lo, 3), round(hi, 3)], "n_items": len(v)}
            print(f"{name:40s} {readout:14s} step {bn:4s} {rows[bn]['mean_nats']:6.2f} [{lo:.2f}, {hi:.2f}] n={len(v)}")
        result["models"][name][readout] = rows
json.dump(result, open(OUT, "w"), indent=1)
