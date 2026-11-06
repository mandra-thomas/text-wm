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
    ("C5 swap may be in receptacle", "wf/data/goals.py", "        if receptacle_in_question in receptacles:\n", "        if False:\n"),
    ("C5 swap may name the action object", "wf/data/transitions.py", "exclude=named | interacted_objects(history_text) | (also_exclude or set()),", "exclude=interacted_objects(history_text) | (also_exclude or set()),"),
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
    ("H7 failures only for A", "wf/runs.py", "everything = main_test_items(eligible) + successes", "everything = main_test_items(eligible)"),
    ("H8 episodes saved at the end", "wf/runs.py", '        append_jsonl(d / "episodes.jsonl", [ep])\n        episodes.append(ep)\n    return episodes', '        episodes.append(ep)\n    append_jsonl(d / "episodes.jsonl", episodes)\n    return episodes'),
    ("resume reruns everything", "wf/runs.py", "        if (gid, fc.condition, fc.arm, fc.retention, seed, fc.p_arm) in done:\n            continue\n", ""),
    ("mock writes to real runs", "wf/util.py", '("runs_mock" if is_mock() else "runs")', '"runs"'),
    ("mock filing accepted", "wf/util.py", 'return is_mock() or "mock" not in filed.read_text(encoding="utf-8").lower()', "return True"),
    ("minirun guesses rates", "wf/runs.py", '        write_status(run_id, f"{cal_path} missing; run pilotB_eligibility first", ok=False)\n        return 1\n', '        cal_path = None\n'),
    ("FT prompt loses hidden state", "wf/scoring/logodds.py", "system = _replace_task_line(world_model_context(transition), task)", 'system = f"{task}\\n"'),
    ("scores saved at the end", "wf/runs.py", '                append_jsonl(d / "scoring.jsonl", [rec])\n                scores.append(rec)\n', '                scores.append(rec)\n'),
    ("resume rescores saved items", "wf/runs.py", '            if (tag, t["id"], cond) in done:\n                continue\n', ''),
    ("resume trusts a half-saved game", "wf/runs.py", 'kept = [r for r in records if r["game_id"] in finished]', 'kept = records'),
    ("stale PASS survives a crash", "wf/runs.py", 'for name in ("STATUS.md", "REPORT.md", "metrics.json"):', 'for name in ("REPORT.md", "metrics.json"):'),
    ("resume keeps a torn line", "wf/runs.py", '            repair_jsonl(p)\n', '            pass\n'),
    ("resume mixes commits", "wf/runs.py", 'if saved_sha.exists() and saved_sha.read_text(encoding="utf-8").strip() != git_sha():', 'if False:'),
    ("skips a pass from another commit", "wf/runs.py", 'return finished and (d / "git_sha").read_text(encoding="utf-8").strip() == git_sha()', 'return finished'),
    ("skips a failed run", "wf/runs.py", 'finished = "— PASS" in first or "— REPORTED" in first', 'finished = True'),
    ("reported run is rerun on resume", "wf/runs.py", 'finished = "— PASS" in first or "— REPORTED" in first', 'finished = "— PASS" in first'),
    ("low agreement looks fine", "wf/runs.py", 'reported=f"agreement {mean_agree:.3f}" if ok else f"BELOW TARGET: agreement {mean_agree:.3f} < 0.95",', 'reported=f"agreement {mean_agree:.3f}",'),
    ("reported shows as PASS", "wf/util.py", 'status = (f"REPORTED ({reported})" if reported else "PASS" if ok else "FAIL")', 'status = ("PASS" if ok else "FAIL")'),
    ("main ignores finished runs", "wf/runs.py", '    if args.resume and _already_passed(args.run_id):\n', '    if False:\n'),
    ("generations saved at the end", "wf/runs.py", '            append_jsonl(d / "generations.jsonl", [rec])\n', ''),
    ("built games not saved", "wf/runs.py", '        _save_game(d, "items.jsonl", gid, kept)\n        items.extend(kept)\n    items = items[:n]\n', '        items.extend(kept)\n    items = items[:n]\n'),
    ("non-atomic json write", "wf/util.py", '    tmp = path.with_name(path.name + ".tmp")\n    tmp.write_text(text, encoding="utf-8")\n    os.replace(tmp, path)\n', '    path.write_text("", encoding="utf-8")\n    os.replace(path, path)\n    path.write_text(text, encoding="utf-8")\n'),
    ("scan skips a shared receptacle", "wf/env/scene.py", 'elif f"go to {receptacle}" not in state.admissible:', 'elif False:'),
    ("shared spots not recorded", "wf/env/scene.py", 'if r != receptacle and f"go to {r}" not in state.admissible}', 'if False}'),
    ("flag never set", "wf/data/transitions.py", 'return bool(target and facts.location and scene.shares_spot(facts.location, target))', 'return False'),
    ("flagged items in the main test", "wf/data/transitions.py", 'return [t for t in items if not t.get("shared_location")]', 'return list(items)'),
    ("sensitivity omits flagged items", "wf/runs.py", '"goal_agnostic": _effect(scores, agnostic_all),', '"goal_agnostic": _effect(scores, agnostic),'),
    ("A sensitivity omits flagged items", "wf/runs.py", '"A": asymmetry_A(scores, eligible + all_successes),', '"A": asymmetry_A(scores, main_test_items(eligible) + successes),'),
    ("sample gate counts flagged items", "wf/runs.py", 'ok = len(main_items) >= 500 and len(states) >= 80', 'ok = len(eligible) >= 500 and len(states) >= 80'),
    ("to inside potato matches", "wf/data/transitions.py", 'r"\\b(?:from|to|with) (.+)$"', 'r"(?:from|to|with) (.+)$"'),
    ("sim has no shared spots", "wf/env/sim.py", 'shared = ("drawer 1", "countertop 1") if game % 2 else ()', 'shared = ()'),
    ("prompts looked up by position", "wf/scoring/logodds.py", 'return {int(row["id"]): row["messages"][0]["content"] for row in rows}', 'return dict(enumerate(row["messages"][0]["content"] for row in rows))'),
    ("no check that a prompt is this game's", "wf/scoring/logodds.py", 'if transition["task_line"].strip() not in context:', 'if False:'),
    ("missing prompt is not an error", "wf/scoring/logodds.py", '    if context is None:\n        raise KeyError', '    if False:\n        raise KeyError'),
    ("swap ignores heat, cool and clean rules", "wf/data/goals.py", "if any(re.search(words, text) and obj not in traits[trait] for trait, words in needed):", "if False:"),
    ("swap ignores pick-two count", "wf/data/goals.py", 'if re.search(r"\\btwo\\b", text) and n_instances < 2:', "if False:"),
    ("swap can be the target", "wf/data/goals.py", 'if obj == spec.receptacle or obj not in traits["pickupable"]:', 'if obj not in traits["pickupable"]:'),
    ("swap may be unpickupable", "wf/data/goals.py", 'if obj == spec.receptacle or obj not in traits["pickupable"]:', "if obj == spec.receptacle:"),
    ("swap ignores what the target holds", "wf/data/goals.py", 'return is_look_task or obj in traits["can_contain"].get(spec.receptacle, set())', "return True"),
    ("items skip the feasibility check", "wf/data/transitions.py", "feasible=lambda o: swap_feasible(task_line, o, scene.traits, len(scene.instances(o))),", "feasible=None,"),
    ("traits lose canContain", "wf/env/server.py", 'traits["can_contain"] = contain', 'traits["can_contain"] = {}'),
    ("population can change quietly", "wf/env/server.py", "if len(good) != SPLIT_SOLVABLE_SIZES.get(split):", "if False:"),
    ("closed arrival counts as determined", "wf/data/transitions.py", "if arrival and arrival[1] not in history_facts(history_text).closed:", "if False:"),
    ("contents filed under where the agent stands", "wf/data/transitions.py", "place = m[1] or (is_open[1] if is_open else facts.location)", "place = m[1] or facts.location"),
    ("move invents contents", "wf/data/transitions.py", "            if m[2] in facts.contents:  # a receptacle never seen stays unknown\n                facts.contents[m[2]].add(m[1])\n", "            facts.contents.setdefault(m[2], set()).add(m[1])\n"),
    ("use is not an action type", "wf/data/transitions.py", "examine|use|slice|inventory", "examine|inventory"),
    ("scan keeps one location", "wf/env/scene.py", "scene.locations.setdefault(instance, set()).add(receptacle)", "scene.locations[instance] = {receptacle}"),
    ("true location called away", "wf/data/transitions.py", 'away = [r for r in scene.receptacles if r not in scene.locations.get(goal, ())]', "away = [r for r in scene.receptacles if r != home]"),
    ("plan knows one location", "wf/data/candidates.py", "if receptacle in scene.locations.get(obj, ()):", "if receptacle == scene.home(obj):"),
    ("swap bar sees one location", "wf/data/goals.py", "if receptacle_in_question in receptacles:", "if receptacle_in_question == min(receptacles):"),
    ("twin may contradict state", "wf/data/candidates.py", 'if _contents_changed(action, transition.get("history_pairs") or []):', "if False:"),
    ("failed actions count as changes", "wf/data/candidates.py", 'obs.strip() != NOOP and re.fullmatch(', "re.fullmatch("),
    ("corruption ignores the agent's state", "wf/agent/foresight.py", '            "history_pairs": state.history,\n', ""),
    ("sim hides double listings", "wf/env/sim.py", '"sinkbasin 1": {"closed": False, "items": ["pen 1"]},', '"sinkbasin 1": {"closed": False, "items": []},'),
    ("sim opens with the wrong text", "wf/env/sim.py", 'return f"The {name} is open. In it, you see {seen}." if name in OPENABLE else f"On the {name}, you see {seen}."', 'return f"On the {name}, you see {seen}."'),
    ("generation ignores end of turn", "wf/scoring/logodds.py", "                eos_token_id=self._stop_tokens(),\n", ""),
    ("checkpoint output cut short", "wf/runs.py", 'build_messages(it, "aligned", ft_prompt=True), max_new_tokens=256)', 'build_messages(it, "aligned", ft_prompt=True))'),
    ("flagged successes stay in A", "wf/runs.py", "successes = main_test_items(all_successes)", "successes = all_successes"),
    ("unparseable successes stay in A", "wf/runs.py", '        and t.get("outcome_class") != "unparseable"\n', ""),
    ("non-finite score accepted", "wf/scoring/logodds.py", "if not math.isfinite(L):", "if False:"),
    ("NaN looks significant", "wf/analysis/stats.py", "if not diffs or any(d != d for d in diffs):", "if not diffs:"),
    ("score cache ignores content", "wf/scoring/logodds.py", "key = (self.model_tag, condition, transition_id, content)", 'key = (self.model_tag, condition, transition_id, "")'),
    ("missing swap scores aligned", "wf/scoring/prompts.py", '        if not swapped_task:\n            raise ValueError("swapped condition needs a swapped goal")  # would quietly score the aligned goal\n', ""),
    ("placebo may repeat the swap", "wf/tools/verify_models.py", 'also_exclude={item["swapped_goal_object"]})', "also_exclude=set())"),
    ("tokenizer check skips fine-tuned prompt", "wf/tools/verify_models.py", 'messages = build_messages(item, "aligned", ft_prompt=ft)', 'messages = build_messages(item, "aligned")'),
    ("calibration pools none with oracle", "wf/runs.py", '    # Corrupted arms run with a prediction on screen, so only oracle episodes show their rates.\n    episodes = [e for e in read_jsonl(src) if e.get("condition") == "oracle"]', '    episodes = [e for e in read_jsonl(src) if e.get("condition") in ("none", "oracle")]'),
    ("calibration counts steps with no twin", "wf/runs.py", 'if twin_available(sandbox, ep["game_id"], ep["split"], tr["proposal"], history, twins):', "if True:"),
    ("arms keep separate targets", "wf/runs.py", "common_target = min(target, *achievable.values())", "common_target = target"),
    ("expert share uses executed action", "wf/runs.py", 'if tr.get("proposal_on_expert_path") is not None:\n                    labelled_valid += 1\n                    expert_valid += int(tr["proposal_on_expert_path"])', 'if tr.get("on_expert_path") is not None:\n                    labelled_valid += 1\n                    expert_valid += int(tr["on_expert_path"])'),
    ("recovery at negative gain", "wf/runs.py", "if oracle_gain > 0 else float", "if oracle_gain else float"),
    ("retention rule unpaired", "wf/runs.py", "shift = abs(sum(ret_S[g] - disc_S[g] for g in paired) / len(paired))", 'shift = abs(rank_outcome(fav_ret)["mean_S"] - rank_outcome(fav_disc)["mean_S"])'),
    ("episodes lose their rate", "wf/runs.py", "                    p_arm=float(p),\n", ""),
    ("determinism advises a seed", "wf/runs.py", "so a seed cannot help: report this and investigate.", "add agent seeding."),
    ("old report survives a rerun", "wf/runs.py", 'for name in ("STATUS.md", "REPORT.md", "metrics.json"):', 'for name in ("STATUS.md",):'),
    ("status not atomic", "wf/util.py", '_write_atomic(p, f"# STATUS — {run_id} — {status}\\n\\n{body.rstrip()}\\n")', 'p.write_text(f"# STATUS — {run_id} — {status}\\n\\n{body.rstrip()}\\n", encoding="utf-8")'),
    ("report accepts a missing status", "wf/tools/assemble_pilot_report.py", '            found.append(f"{rid}: no STATUS.md")', "            pass"),
    ("report accepts any commit", "wf/tools/assemble_pilot_report.py", "elif not any(sha.startswith(c) for c in allowed if c):", "elif False:"),
    ("report accepts mock runs", "wf/tools/assemble_pilot_report.py", 'elif ("(MOCK)" in first) != is_mock():', "elif False:"),
    ("report writes despite problems", "wf/tools/assemble_pilot_report.py", "    if found:\n", "    if False:\n"),
    ("scores join by id alone", "wf/runs.py", 'if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):', "if False:"),
    ("resume ignores changed inputs", "wf/runs.py", 'if saved_inputs.exists() and json.loads(saved_inputs.read_text(encoding="utf-8"))["inputs"] != inputs:', "if False:"),
    ("skip ignores changed inputs", "wf/runs.py", "if any(not Path(path).exists() or file_sha(Path(path)) != digest for path, digest in inputs.items()):", "if False:"),
    ("resume ignores changed settings", "wf/runs.py", "if saved != json.loads(json.dumps(config)):", "if False:"),
    ("resume allows dirty code", "wf/runs.py", "if not is_mock() and git_dirty():", "if False:"),
    ("real run with no commit", "wf/runs.py", 'if not is_mock() and git_sha() == "unknown":', "if False:"),
    ("unknown commit counts as passed", "wf/runs.py", 'if not is_mock() and sha == "unknown":', "if False:"),
    ("dirty flag counts run output", "wf/util.py", '"--", ".", ":(exclude)runs",\n             ":(exclude)wf/config/calibrated.yaml", ":(exclude)wf/config/eligible_target.yaml"]', '"--", "."]'),
    ("mock pass counts as real", "wf/runs.py", 'if ("(MOCK)" in first) != is_mock():', "if False:"),
    ("mock may overwrite real runs", "wf/util.py", "if is_mock() and not _mock_may_write(root):", "if False:"),
    ("mock records unmarked", "wf/util.py", 'return {**rec, "mock": True} if is_mock() else rec', "return rec"),
    ("real run reads mock records", "wf/util.py", 'if rec.get("mock") and not is_mock():', "if False:"),
    ("resume skips a rule stop", "wf/runs.py", 'return _finished(run_id) and "STOP:" not in _status_line(run_id)', "return _finished(run_id)"),
    ("stages ignore a rule stop", "wf/runs.py", 'if not problem and "STOP:" in _status_line(dep) and not _already_passed("pilotB_headroom_recheck"):', "if False:"),
    ("dependencies not enforced", "wf/runs.py", "    if problem:\n        write_status(run_id, problem, ok=False)\n        raise SystemExit(1)\n", ""),
    ("earlier commits accepted silently", "wf/runs.py", "if any(sha.startswith(c) for c in commits):", "if True:"),
    ("headroom passes on no data", "wf/runs.py", 'if any(v != v for v in (gain["estimate"], gain["ci_low"], gain["ci_high"])):', "if False:"),
    ("rule stop exits zero", "wf/runs.py", "return 1 if broken else RULE_STOP_EXIT if stop else 0", "return 1 if broken else 0"),
    ("headroom says PASS", "wf/runs.py", 'reported=None if broken else f"STOP: {rule}" if stop else "rule not fired",\n', "reported=None,\n"),
    ("gate counts the broad population", "wf/runs.py", "main_items = primary_items(eligible)", "main_items = main_test_items(eligible)"),
    ("verify job hides a failed check", "wf/tools/verify_real_env.py", "    if failed:\n", "    if False:\n"),
    ("review saved only at the end", "wf/tools/export_review.py", "        append_jsonl(progress, built)\n", ""),
    ("review claims real on mock", "wf/tools/export_review.py", 'source = "MOCK DATA, not the real game." if is_mock() else "Generated from the real ALFWorld server."', 'source = "Generated from the real ALFWorld server."'),
    ("job runs in mock mode", "pace/_common.sh", 'if [[ -n "${!mock_var:-}" ]]; then', "if false; then"),
    ("WF_RESUME=0 resumes", "pace/_common.sh", 'if [[ "${WF_RESUME:-}" == "1" ]]; then', 'if [[ -n "${WF_RESUME:-}" ]]; then'),
    ("shell picks the checkout", "pace/_common.sh", 'export PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"\n', 'export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"\n'),
    ("pace.env mismatch passes", "pace/_common.sh", 'if [[ "$(cd "${PROJECT_ROOT}" 2>/dev/null && pwd -P)" != "${PROJECT_ROOT_DEFAULT}" ]]; then', "if false; then"),
    ("job can requeue", "pace/pilotA.sbatch", "#SBATCH --no-requeue\n", ""),
    ("scan ignores the step limit", "wf/env/scene.py", "if state.step + 2 >= ENV_STEP_LIMIT:", "if False:"),
    ("sim has no step limit", "wf/env/sim.py", "self.done[idx] = world.won or world.steps >= ENV_STEP_LIMIT", "self.done[idx] = world.won"),
    ("duplicate LLM calls", "wf/agent/react.py", "            if key not in replies:\n                replies[key] = agent.complete(msgs)\n", "            replies[key] = agent.complete(msgs)\n"),
    ("prediction hides its action", "wf/scoring/prompts.py", '        body_parts.append(f"Proposed action: {proposal}")\n        body_parts.append(f"Expected outcome: {foresight_prediction}")', '        body_parts.append(f"Expected outcome: {foresight_prediction}")'),
]


def main() -> int:
    python = sys.executable
    # `mutation_check.py 2/4` runs the third of four shards, so the check can use several cores.
    shard, shards = map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1)
    chosen = MUTANTS[shard::shards]
    survivors: list[str] = []
    broken: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        def skip(folder: str, names: list[str]) -> list[str]:
            top = {".git", ".venv", "runs", "runs_mock", "docs"} if Path(folder) == ROOT else set()
            return [n for n in names if n in top or n in ("__pycache__", ".pytest_cache") or n.endswith(".pdf")]

        shutil.copytree(ROOT, work, ignore=skip)
        for name, rel, good, bad in chosen:
            path = work / rel
            source = path.read_text(encoding="utf-8")
            if good not in source:
                broken.append(name)
                print(f"NOT APPLIED  {name}")
                continue
            path.write_text(source.replace(good, bad), encoding="utf-8")
            try:
                result = subprocess.run(
                    [python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no", "tests/test_fix_round.py", "tests"],
                    cwd=work, capture_output=True, text=True, timeout=300,
                )
                killed = result.returncode != 0
            except subprocess.TimeoutExpired:  # a mutant that hangs the tests has been noticed
                killed = True
            path.write_text(source, encoding="utf-8")
            print(f"{'killed  ' if killed else 'SURVIVED'}  {name}", flush=True)
            if not killed:
                survivors.append(name)
    print(f"\n{len(chosen) - len(survivors) - len(broken)}/{len(chosen)} killed; survivors={survivors}; not applied={broken}")
    return 1 if survivors or broken else 0


if __name__ == "__main__":
    sys.exit(main())
