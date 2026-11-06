#!/usr/bin/env python3
"""Put each known bug back, one at a time, and confirm the test suite notices.

A surviving mutant means the tests would not catch that bug coming back. Runs on a copy of
the repo, so the working tree is never touched.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (bug, file, correct text, buggy text)
MUTANTS = [
    ("C1 foresight on live env", "wf/agent/foresight.py", "sandbox = client.sandbox()", "sandbox = client"),
    ("C1 sandbox shares env id", "wf/env/server.py", "            side.create()\n", "            side._env_id = self._env_id\n"),
    ("C2 no history unless retained", "wf/agent/react.py", "history=history,", 'history=history if foresight.retention == "retained" else None,'),
    ("C2 prediction kept when discarded", "wf/agent/react.py", 'if foresight.retention == "retained" and shown is not None:', "if shown is not None:"),
    ("C3 rewrite invalid action", "wf/agent/react.py", "        new_state = env.step(state, executed)\n",
     "        if state.admissible and executed not in state.admissible:\n            executed = state.admissible[0]\n        new_state = env.step(state, executed)\n"),
    ("C3 no-twin counted", "wf/agent/foresight.py", '                "corrupted": False,\n                "corruption_unavailable": True,', '                "corrupted": True,\n                "corruption_unavailable": True,'),
    ("corrupt the wrong arm", "wf/corruption/schedule.py", 'if arm == "favorable" and proposal_valid:', 'if arm == "favorable" and not proposal_valid:'),
    ("schedule hashes arm", "wf/corruption/schedule.py", 'f"{game_id}|{seed}|{t}"', 'f"{game_id}|{seed}|{arm}|{t}"'),
    ("C4 goal needs an article", "wf/data/goals.py", r"(?:a|an|some|two) (?:(?:clean|hot|cool) )?", r"(?:a|an) "),
    ("C5 swap changes template", "wf/data/goals.py", 'return re.sub(rf"\\b{re.escape(goal_object)}\\b", swapped_object, task_line, count=1)', 'return f"Your task is to: put a {swapped_object} in coffeemachine."'),
    ("C5 swap may be in receptacle", "wf/data/goals.py", "        if receptacle == receptacle_in_question:\n", "        if False:\n"),
    ("C5 swap may name the action object", "wf/data/transitions.py", "exclude=named | seen,", "exclude=seen,"),
    ("C6 every failure determined", "wf/data/transitions.py", "return _failure_determined(history_facts(history_text), action)", "return True"),
    ("goal adjectives count as interactions", "wf/data/goals.py", 'r"^> (?:take|move|heat|cool|clean|examine)', 'r"(?:take|move|heat|cool|clean|examine)'),
    ("task swap drops bytes", "wf/scoring/prompts.py", "    replaced, n = re.subn(r\"^Your task is to:.*$\", lambda _: task_line, history_text, flags=re.MULTILINE)\n    return replaced if n else f\"{task_line}\\n{history_text}\"\n",
     "    return \"\\n\".join(task_line if ln.startswith(\"Your task is to:\") else ln for ln in history_text.splitlines())\n"),
    ("hand-written o_plus", "wf/data/candidates.py", '            transition["candidates"] = None\n            transition["exclusion_reason"] = "no_success_twin"\n            return transition\n', '            o_plus = f"You did {action}."\n'),
    ("twin error treated as no twin", "wf/data/candidates.py", "            trial, _ = step_counterfactual(handle, action)\n", "            try:\n                trial, _ = step_counterfactual(handle, action)\n            except RuntimeError:\n                trial = ''\n"),
    ("H1 padded agreement items", "wf/runs.py", '    items = items[:100]\n    if len(items) < 100:\n', '    while len(items) < 100:\n        items.append(items[0])\n    if len(items) < 100:\n'),
    ("H1 too few items pass", "wf/runs.py", '        write_status(run_id, f"only {len(items)} distinct items with candidates; need 100 — FAIL", ok=False)\n        return 1\n', '        pass\n'),
    ("H2 parser misses info", "wf/scoring/parser.py", "    if _INFO.match(first):\n        return \"info\"\n", ""),
    ("H3 unlisted items counted", "wf/analysis/stats.py", 'if meta is None or not meta.get("context_determined"):\n                continue', 'if meta is None:\n                pass\n            elif not meta.get("context_determined"):\n                continue'),
    ("H4 flips single items", "wf/analysis/stats.py", "totals[cluster_ids[i] if cluster_ids else i] += d", "totals[i] += d"),
    ("delta_L is really delta_p", "wf/analysis/stats.py", 'return _paired_effect(pairs, lambda a, c: a - c, "delta_L_nats")', 'return _paired_effect(pairs, lambda a, c: (sigmoid(a) - sigmoid(c)) * 100.0, "delta_L_nats")'),
    ("interaction sign flipped", "wf/analysis/stats.py", 'return H_cell(episodes, "retained") - H_cell(episodes, "discarded")', 'return H_cell(episodes, "discarded") - H_cell(episodes, "retained")'),
    ("failures rank like successes", "wf/analysis/stats.py", "S = -(step_cap + 1)", "S = -step_cap"),
    ("ICC crash on identical diffs", "wf/analysis/stats.py", 'if msb != msb or msw != msw or denominator == 0:', "if msb != msb or msw != msw:"),
    ("H7 failures only for A", "wf/runs.py", "everything = eligible + successes", "everything = eligible"),
    ("H8 episodes saved at the end", "wf/runs.py", '        append_jsonl(d / "episodes.jsonl", [ep])\n        episodes.append(ep)\n    return episodes', '        episodes.append(ep)\n    append_jsonl(d / "episodes.jsonl", episodes)\n    return episodes'),
    ("resume reruns everything", "wf/runs.py", "        if (gid, fc.condition, fc.arm, fc.retention, seed) in done:\n            continue\n", ""),
    ("mock writes to real runs", "wf/util.py", '("runs_mock" if is_mock() else "runs")', '"runs"'),
    ("mock filing accepted", "wf/util.py", 'return is_mock() or "mock" not in filed.read_text(encoding="utf-8").lower()', "return True"),
    ("minirun guesses rates", "wf/runs.py", '        write_status(run_id, f"{cal_path} missing; run pilotB_eligibility first", ok=False)\n        return 1\n', '        cal_path = None\n'),
    ("FT prompt loses hidden state", "wf/scoring/logodds.py", "system = _replace_task_line(world_model_context(transition), task)", 'system = f"{task}\\n"'),
    ("duplicate LLM calls", "wf/agent/react.py", "            if key not in replies:\n                replies[key] = agent.complete(msgs)\n", "            replies[key] = agent.complete(msgs)\n"),
    ("prediction hides its action", "wf/scoring/prompts.py", '        body_parts.append(f"Proposed action: {proposal}")\n        body_parts.append(f"Expected outcome: {foresight_prediction}")', '        body_parts.append(f"Expected outcome: {foresight_prediction}")'),
]


def main() -> int:
    python = sys.executable
    survivors: list[str] = []
    broken: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        def skip(folder: str, names: list[str]) -> list[str]:
            top = {".git", ".venv", "runs", "runs_mock", "docs"} if Path(folder) == ROOT else set()
            return [n for n in names if n in top or n in ("__pycache__", ".pytest_cache") or n.endswith(".pdf")]

        shutil.copytree(ROOT, work, ignore=skip)
        for name, rel, good, bad in MUTANTS:
            path = work / rel
            source = path.read_text(encoding="utf-8")
            if good not in source:
                broken.append(name)
                print(f"NOT APPLIED  {name}")
                continue
            path.write_text(source.replace(good, bad), encoding="utf-8")
            result = subprocess.run(
                [python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no"],
                cwd=work, capture_output=True, text=True, timeout=300,
            )
            path.write_text(source, encoding="utf-8")
            killed = result.returncode != 0
            print(f"{'killed  ' if killed else 'SURVIVED'}  {name}", flush=True)
            if not killed:
                survivors.append(name)
    print(f"\n{len(MUTANTS) - len(survivors) - len(broken)}/{len(MUTANTS)} killed; survivors={survivors}; not applied={broken}")
    return 1 if survivors or broken else 0


if __name__ == "__main__":
    sys.exit(main())
