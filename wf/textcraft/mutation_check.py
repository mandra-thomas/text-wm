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
