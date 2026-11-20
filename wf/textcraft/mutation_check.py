"""Put deliberate bugs into the TextCraft pipeline, one at a time, and confirm the TextCraft tests notice.

    python -m wf.textcraft.mutation_check

Works on a copy of the repository, so the working tree is never touched. Each mutant runs against
tests/test_textcraft_*.py; a timeout is reported as inconclusive, never as caught (as scripts/mutation_check.py).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = "wf/textcraft/"

# (bug, file, correct text, buggy text)
MUTANTS = [
    # text-only checker
    ("checker ignores what a craft uses up", T + "checker.py", "                ledger.remove(n, c)\n", "                pass\n"),
    ("checker counts every get as one", T + "checker.py", "ledger.add(got.group(2), int(got.group(1)))", "ledger.add(got.group(2), 1)"),
    ("checker calls an exact holding short", T + "checker.py", "        if held < c:\n", "        if held <= c:\n"),
    ("checker writes the item with spaces", T + "checker.py", 'f"Could not find enough items to craft {to_id(target)}"',
     'f"Could not find enough items to craft minecraft:{target}"'),
    ("checker class precedence reversed", T + "checker.py", 'CLASS_ORDER = ("used_up", "too_few", "never_obtained")',
     'CLASS_ORDER = ("never_obtained", "too_few", "used_up")'),
    ("checker reads used up from the current holding", T + "checker.py", 'kind = "used_up" if peak >= c else', 'kind = "used_up" if held >= c else'),
    ("checker trusts inventory replies", T + "checker.py", "            if shown != {k: v for k, v in ledger.held.items() if v}:\n",
     "            if False:\n"),
    ("checker get rule ignores the list", T + "checker.py", "        if name in displayed_outputs(recipe_lines):\n", "        if True:\n"),
    # item builder
    ("walk may craft a goal", T + "build.py", "        if ln.out in pair.goals:\n            continue\n", ""),
    ("walk depends on which goal comes first", T + "build.py", 'f"{SEED}|textcraft|walk|{pair.pair_id}|{seed}"',
     'f"{SEED}|textcraft|walk|{pair.goals[0]}|{seed}"'),
    ("list order depends on the direction", T + "build.py", "    a, b = sorted((a, b))\n    lines =", "    lines ="),
    ("partners may share craftable items", T + "build.py", "    return not (W.tree_craftables(a_iid) & W.tree_craftables(b_iid))",
     "    return True"),
    ("goals named in the instruction kept", T + "build.py", "        if W.name_of(iid) not in EXCLUDED_GOALS:\n", "        if True:\n"),
    ("stored top-up is the full count", T + "build.py", '"topup": {s["name"]: s["need"] - s["held"] for s in short},',
     '"topup": {s["name"]: s["need"] for s in short},'),
    ("empty-history crafts in the main test", T + "build.py", '"population": "main" if t >= 1 else "empty_history"', '"population": "main"'),
    ("no deduplication", T + "build.py", "                if key in seen:\n", "                if False:\n"),
    ("builder class from the current holding", T + "build.py", 'kind = "used_up" if peak >= c else', 'kind = "used_up" if held >= c else'),
    ("generic ingredient ignores holdings", T + "build.py", "    held = [m for m in members if state.held(m) > 0]\n", "    held = []\n"),
    ("prompt-example pairs not held out", T + "build.py", "if p.pair_id not in held_out for it in pool_by_pair", "for it in pool_by_pair"),
    ("walk fetches craftable inputs", T + "build.py", "if not generic and W.gettable(n)})", "if not generic})"),
    # prompts
    ("swapped prompt names the real goal", T + "prompts.py", '"swapped": item["swapped_task_line"]', '"swapped": item["task_line"]'),
    ("examples show the item's goal", T + "prompts.py", "f\"Goal: craft {ex['goal']}.\")})", 'item["task_line"])})'),
    ("question added after Next observation", T + "prompts.py", "query = query[: -len(NEXT)] + YN_QUESTION", 'query = query + "\\n" + YN_QUESTION'),
    ("history loses the action marker", T + "prompts.py", 'f"\\n> {a}\\n{o}"', 'f"\\n{a}\\n{o}"'),
    ("agent output format kept in the instruction", T + "prompts.py", '.replace(AGENT_FORMAT_SENTENCE, "")', ""),
    # scoring and analysis
    ("scoring ignores the items hash", T + "score.py", '    if _sha(items) != manifest["items_sha256"]:\n', "    if False:\n"),
    ("scoring accepts an unverified manifest", T + "score.py", '    if manifest.get("verified") is not True:\n', "    if False:\n"),
    ("resume rescores saved work", T + "score.py", ' for a in arms if (tag, t["id"], a) not in done]', " for a in arms]"),
    ("analysis clusters by item", T + "analysis.py", '        pairs.append(t["pair_id"])\n', '        pairs.append(t["id"])\n'),
    ("verdict skips Holm", T + "analysis.py", 'ok = r["estimate"] > 0 and r["ci"][0] > 0 and adjusted[m] < P_MAX',
     'ok = r["estimate"] > 0 and r["ci"][0] > 0 and r.get("perm_p", 1.0) < P_MAX'),
    # vendored environment
    ("environment lets an exact holding fail", T + "upstream/environment.py",
     "or self.inventory[itemtag_count.item_tag.item_id] < itemtag_count.count", "or self.inventory[itemtag_count.item_tag.item_id] <= itemtag_count.count"),
    # review 1's mutants (scratch review_textcraft/mutants.py), R11 and R15 rewritten for the amended code
    ("R1 selection seed drops the bin", T + "build.py", 'f"{SEED}|textcraft|select|{population}|{c}|{b}"', 'f"{SEED}|textcraft|select|{population}|{c}"'),
    ("R2 round robin never shuffles pairs", T + "build.py", "    rng.shuffle(order)\n", ""),
    ("R3 generic fallback ignores displayed kinds", T + "build.py", "    return (shown or members)[0]\n", "    return members[-1]\n"),
    ("R4 checker used-up boundary peak>c", T + "checker.py", 'kind = "used_up" if peak >= c else', 'kind = "used_up" if peak > c else'),
    ("R5 verify drops partner-goal replay", T + "verify.py", '    if other != item["history"]:\n', "    if False:\n"),
    ("R6 verify prompt check returns nothing", T + "verify.py", '    problems = []\n    base = P.messages(item, "aligned", shots)\n',
     '    return []\n    base = P.messages(item, "aligned", shots)\n'),
    ("R7 primary test pools every population", T + "analysis.py", 'primary[m] = effect(mine, pops["main"], "aligned", "swapped")',
     'primary[m] = effect(mine, items, "aligned", "swapped")'),
    ("R8 scoring ignores the few-shot hash", T + "score.py", '    if _sha(shots) != manifest["fewshot_sha256"]:\n', "    if False:\n"),
    ("R9 resume ignores item hashes", T + "score.py", '    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):\n', "    if False:\n"),
    ("R10 first-token readout reads L", T + "analysis.py", '"delta_L_first": effect(scores, items, "aligned", "swapped", "L_first")',
     '"delta_L_first": effect(scores, items, "aligned", "swapped", "L")'),
    ("R11 item hash omits candidates", T + "score.py", '"neutral_task_line", "candidates", "prefix_candidates")}', '"neutral_task_line", "prefix_candidates")}'),
    ("R12 missing scores never flagged", T + "analysis.py", '    missing = wanted - {(s["model"], s["transition_id"], s["condition"]) for s in scores}\n',
     "    missing = set()\n"),
    ("R13 neutral arm scores the aligned line", T + "prompts.py", '"neutral": item["neutral_task_line"]}', '"neutral": item["task_line"]}'),
    ("R14 yes/no effect reads swapped minus aligned", T + "analysis.py", '"yes_no_aligned_minus_swapped": effect(scores, items, "yn_aligned", "yn_swapped")',
     '"yes_no_aligned_minus_swapped": effect(scores, items, "yn_swapped", "yn_aligned")'),
    ("R15 strata use aligned minus neutral as delta", T + "analysis.py", '{"delta_L": effect(scores, v, "aligned", "swapped"),',
     '{"delta_L": effect(scores, v, "aligned", "neutral"),'),
    ("R16 permutation flips items not pairs", T + "analysis.py", '"perm_p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=pairs)',
     '"perm_p": pair_permutation(diffs, seed=C.RNG_SEED)'),
    ("R17 walk fetch counts 1..3", T + "build.py", "GET_COUNTS = (1, 2, 3, 4)", "GET_COUNTS = (1, 2, 3)"),
    ("R18 smoke subset seed changes", T + "score.py", 'f"{C.RNG_SEED}|textcraft|smoke"', 'f"{C.RNG_SEED}|textcraft|smoke2"'),
    # seeds (fix round 1): every seeded draw must be pinned by a test, not only by verify's rebuild
    ("seed: main selection drops the bin", T + "build.py", 'f"{SEED}|textcraft|select|main|{c}|{b}|{r}|{f}"', 'f"{SEED}|textcraft|select|main|{c}|{r}|{f}"'),
    ("seed: list shuffle", T + "build.py", 'f"{SEED}|textcraft|list|{a}|{b}"', 'f"{SEED}|textcraft|list2|{a}|{b}"'),
    ("seed: pairing shuffle", T + "build.py", 'f"{SEED}|textcraft|pairs|{depth}"', 'f"{SEED}|textcraft|pairs2|{depth}"'),
    ("seed: walk", T + "build.py", 'f"{SEED}|textcraft|walk|{pair.pair_id}|{seed}"', 'f"{SEED}|textcraft|walk2|{pair.pair_id}|{seed}"'),
    ("seed: prompt-example pairs", T + "build.py", 'f"{SEED}|textcraft|fewshot")', 'f"{SEED}|textcraft|fewshot2")'),
    ("seed: prompt-example failure item", T + "build.py", 'f"{SEED}|textcraft|fewshot|item"', 'f"{SEED}|textcraft|fewshot|item2"'),
    # amendments after review 1 (design section 11)
    ("A1 tag-reading items kept", T + "build.py", "                if tag_reading_ambiguous(state, short):\n", "                if False:\n"),
    ("A1 warped stems not read as logs", T + "build.py", '**{f"{s} stems": (f"{s} stem", f"{s} hyphae", f"stripped {s} stem", f"stripped {s} hyphae")\n                for s in ("crimson", "warped")}}',
     '**{f"{s} stems": (f"{s} stem", f"{s} hyphae", f"stripped {s} stem", f"stripped {s} hyphae")\n                for s in ("crimson",)}}'),
    ("A1 checker counts white wool as wool", T + "checker.py", '"wool": ("black wool",', '"wool": ("white wool", "black wool",'),
    ("A2 arithmetic flag set by any obtained input", T + "build.py", '"needs_arithmetic": all(s["peak"] > 0 for s in short)}',
     '"needs_arithmetic": any(s["peak"] > 0 for s in short)}'),
    ("A2 checker purity check off", T + "checker.py", '        if item.get("pure_class") != (len({s["kind"] for s in v.short_inputs}) == 1):\n',
     "        if False:\n"),
    ("A2 secondary readout on all main items", T + "analysis.py", '[t for t in pops["main"] if t["needs_arithmetic"]]', 'pops["main"]'),
    ("A3 builder fit counts ties as aligned", T + "build.py", 'fit = "aligned" if n["aligned"] > n["swapped"] else', 'fit = "aligned" if n["aligned"] >= n["swapped"] else'),
    ("A3 builder fit counts shared names", T + "build.py", 'n = {"aligned": sum(o in a - p for o in objs), "swapped": sum(o in p - a for o in objs)}\n    fit',
     'n = {"aligned": sum(o in a for o in objs), "swapped": sum(o in p for o in objs)}\n    fit'),
    ("A3 checker tree ignores generic members", T + "checker.py", "        wanted = {name} | set(GENERIC_MEMBERS.get(name, ()))\n", "        wanted = {name}\n"),
    ("A3 main selection draws ties", T + "build.py", "                for f in COUPLE_FITS:\n", "                for f in FITS:\n"),
    ("A3 main selection one-sided", T + "build.py", "    cap = {(c, b, r): min(len(cells[(c, b, r, f)]) for f in COUPLE_FITS)",
     "    cap = {(c, b, r): max(len(cells[(c, b, r, f)]) for f in COUPLE_FITS)"),
    ("A3 verify accepts unbalanced cells", T + "verify.py", '            and not r["main_fit_unbalanced_cells"])', "            )"),
    ("A4 prefix arm scores the full replies", T + "prompts.py", '        return item["prefix_candidates"]["o_plus"], item["prefix_candidates"]["o_minus"]',
     '        return item["candidates"]["o_plus"], item["candidates"]["o_minus"]'),
    ("A4 prefix keeps the object name", T + "build.py", '"o_minus": "Could not find enough items to craft"}', '"o_minus": "Could not find enough items to craft minecraft"}'),
    ("A4 name remainder sign flipped", T + "analysis.py", "else (v[0] - v[1]) - (v[2] - v[3])", "else (v[0] - v[1]) + (v[2] - v[3])"),
    ("A4 prefix arm uses the swapped prompt", T + "prompts.py", 'arm.removeprefix("prefix_") if arm in PREFIX_ARMS', '"swapped" if arm in PREFIX_ARMS'),
    ("A5 analysis joins stale scores", T + "analysis.py", "    if stale:\n", "    if False:\n"),
    ("A5 verify goal-line check off", T + "verify.py", '        if [line for line in query.split("\\n") if line.startswith("Goal: ")] != [item[field]]:\n',
     "        if False:\n"),
    ("A6 rebuild skips the hash check", T + "freeze.py", "    if got != want:\n", "    if False:\n"),
    ("A6 rebuild keeps a loop file", T + "freeze.py", '    ("honey_block.json", "craft 1 honey block using 4 honey bottle", "honey_bottle.json"),\n', ""),
]


def _run_tests(work: Path, timeout: int) -> str:
    tests = sorted(str(p.relative_to(work)) for p in (work / "tests").glob("test_textcraft_*.py"))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no", *tests],
                           cwd=work, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    return "killed" if r.returncode != 0 else "survived"


def main() -> int:
    timeout = 600
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"

        def skip(folder: str, names: list[str]) -> list[str]:
            top = {".git", ".venv", "runs", "runs_mock", "docs"} if Path(folder) == ROOT else set()
            return [n for n in names if n in top or n in ("__pycache__", ".pytest_cache") or n.endswith(".pdf")]

        shutil.copytree(ROOT, work, ignore=skip)
        baseline = _run_tests(work, timeout)
        if baseline != "survived":
            print(f"the unmutated tests do not pass ({baseline}); nothing to measure")
            return 1
        for name, rel, good, bad in MUTANTS:
            path = work / rel
            source = path.read_text(encoding="utf-8")
            if source.count(good) != 1:
                results.append((name, "NOT APPLIED"))
                print(f"NOT APPLIED  {name}", flush=True)
                continue
            path.write_text(source.replace(good, bad), encoding="utf-8")
            outcome = _run_tests(work, timeout)
            path.write_text(source, encoding="utf-8")
            label = {"killed": "killed", "survived": "SURVIVED", "timeout": "TIMEOUT"}[outcome]
            results.append((name, label))
            print(f"{label:<12} {name}", flush=True)
    killed = sum(r == "killed" for _, r in results)
    print(f"\n{killed}/{len(MUTANTS)} killed; others: {[(n, r) for n, r in results if r != 'killed']}")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
