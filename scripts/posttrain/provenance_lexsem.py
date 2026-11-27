"""Provenance comparison fixed in notes/posttraining_design.md section 17 (D-PT2): every cell scored by both the
post-training pilot and lexsem, matched by item, condition and format; expected bit-identical (values as both write
them, rounded to five decimals). Successes: lexsem's own-first-token lp_a is compared with the pilot's tok_lp[0]."""
import json
import sys
from collections import Counter

PT = "/storage/ice1/2/1/gmat3/text-wm-posttrain-run/runs/posttrain"
LX = "/storage/ice1/2/1/gmat3/text-wm-lexsem/runs/lexsem"
PAIRS = {"llama8b_base_plain": "llama-base_plain", "qwen8b_base_plain": "qwen-base_plain", "qwen_plain": "qwen_plain",
         "qwen_block": "qwen_native", "llama_chat": "llama-instruct_native"}
metrics = json.load(open(sys.argv[1]))


def rd(p):
    try:
        return [json.loads(x) for x in open(p)]
    except FileNotFoundError:
        return []


out = {}
for lx, pt in PAIRS.items():
    envl = json.load(open(f"{LX}/{lx}_env.json"))
    envp = json.load(open(f"{PT}/{pt}/env.json"))
    keys = ("gpu", "torch", "transformers", "python", "revision", "dtype", "unbatched", "mode")
    env_diff = {k: (envp.get(k), envl.get(k)) for k in keys if envp.get(k) != envl.get(k)}
    mine = {(r["item_set"], r["id"]): r for r in rd(f"{PT}/{pt}/scoring.jsonl")}
    theirs = {("failure", r["id"]): r for r in rd(f"{LX}/{lx}.jsonl")}
    theirs.update({("success", r["id"]): r for r in rd(f"{LX}/{lx}_successes.jsonl")})
    cnt, maxdiff, examples = Counter(), {}, []
    for key, t in theirs.items():
        m = mine.get(key)
        if m is None:
            cnt["item missing in pilot"] += 1
            continue
        for c, fm in t["scores"].items():
            for f, cell in fm.items():
                mc = m["scores"].get(c, {}).get(f)
                if mc is None:
                    cnt["cell not scored by pilot"] += 1
                    continue
                pairs = []
                if key[0] == "success" and f == "obs":
                    pairs.append(("lp_own_first", cell["lp_a"], mc["tok_lp"][0]))
                    if "lp_you" in cell:
                        pairs.append(("lp_you", cell["lp_you"], mc["lp_a"]))
                else:
                    pairs.append(("lp_a", cell["lp_a"], mc["lp_a"]))
                pairs.append(("lp_b", cell["lp_b"], mc["lp_b"]))
                if "tok_lp" in cell:
                    if len(cell["tok_lp"]) != len(mc["tok_lp"]):
                        cnt[f"{key[0]}|{f}|length differs"] += 1
                        continue
                    pairs += [("tok_lp", a, b) for a, b in zip(cell["tok_lp"], mc["tok_lp"])]
                diff = max(abs(a - b) for _, a, b in pairs)
                k2 = f"{key[0]}|{f}"
                cnt[f"{k2}|cells"] += 1
                if diff == 0:
                    cnt[f"{k2}|identical"] += 1
                else:
                    cnt[f"{k2}|differ"] += 1
                    maxdiff[k2] = max(maxdiff.get(k2, 0), diff)
                    if len(examples) < 5:
                        examples.append({"item": key, "cond": c, "fmt": f, "max_abs_diff": diff})
    fl = (metrics.get("floors", {}).get(pt) or {})
    out[f"{lx} vs {pt}"] = {"environment_differences": env_diff, "counts": dict(sorted(cnt.items())),
                            "max_abs_diff": maxdiff, "examples": examples,
                            "pilot_floor_nats": {r: (v or {}).get("floor_nats") for r, v in fl.items()}}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out.items():
    print(k, "env diff:", v["environment_differences"], v["counts"], "maxdiff:", v["max_abs_diff"], flush=True)
