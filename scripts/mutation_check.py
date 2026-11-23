#!/usr/bin/env python3
"""Put each known bug back, one at a time, and confirm the test suite notices.

A surviving mutant means the tests would not catch that bug coming back. Runs on a copy of
the repo, so the working tree is never touched.

Kill rule (strict, the default from 2026-11-23): a mutant is killed only when pytest exits 1 and names a failing
test. A collection or import error, any other non-zero exit, or a run with only setup or teardown errors is
"error-only", not a kill; a timeout is inconclusive. `--legacy-any-exit` restores the earlier rule (any non-zero
exit is a kill) to reproduce logs written before 2026-11-23. See notes/testing.md, "Mutation runner: strict kill rule".

    python scripts/mutation_check.py [i/n] [--results out.jsonl] [--root CHECKOUT] [--mutants-from FILE]

`--root` and `--mutants-from` run another checkout's mutants against that checkout's code and tests (for instance a
worktree of a frozen study's tag), mutating a temporary copy so the checkout stays byte-identical. `--results`
appends one JSON record per mutant as it finishes and skips mutants already recorded there, so a cut-off run resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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
    ("task swap drops bytes", "wf/scoring/prompts.py", "    replaced, n = re.subn(r\"^Your task is to:.*$\", lambda _: task_line, history_text, flags=re.MULTILINE)\n",
     "    replaced, n = \"\\n\".join(task_line if ln.startswith(\"Your task is to:\") else ln for ln in history_text.splitlines()), 1\n"),
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
    ("generations saved at the end", "wf/runs.py", '            append_jsonl(d / "generations.jsonl", [rec])\n        counts[', '        counts['),
    ("built games not saved", "wf/runs.py", '        _save_game(d, "items.jsonl", gid, kept)\n        items.extend(kept)\n    items = items[:n]\n', '        items.extend(kept)\n    items = items[:n]\n'),
    ("non-atomic json write", "wf/util.py", '    tmp = path.with_name(path.name + ".tmp")\n    tmp.write_text(text, encoding="utf-8")\n    os.replace(tmp, path)\n', '    path.write_text("", encoding="utf-8")\n    os.replace(path, path)\n    path.write_text(text, encoding="utf-8")\n'),
    ("scan skips a shared receptacle", "wf/env/scene.py", 'elif f"go to {receptacle}" not in state.admissible:', 'elif False:'),
    ("shared spots not recorded", "wf/env/scene.py", 'if r != receptacle and f"go to {r}" not in state.admissible}', 'if False}'),
    ("flag never set", "wf/data/transitions.py", 'return bool(target and facts.location and scene.shares_spot(facts.location, target))', 'return False'),
    ("flagged items in the main test", "wf/data/transitions.py", 'return [t for t in items if not t.get("shared_location")]', 'return list(items)'),
    ("sensitivity omits flagged items", "wf/runs.py", '"goal_agnostic": _effect(scores, agnostic_all),', '"goal_agnostic": _effect(scores, agnostic),'),
    ("A sensitivity omits flagged items", "wf/runs.py", '"A": asymmetry_A(gens, eligible + all_successes),', '"A": asymmetry_A(gens, main_test_items(eligible) + successes),'),
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
    ("scores join by id alone", "wf/runs.py", 'if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved + saved_gens):', "if False:"),
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
    ("stages ignore a rule stop", "wf/runs.py", 'if (not problem and "STOP:" in _status_line(dep) and "PILOT_RULE_HEADROOM" not in _status_line(dep)', "if (False"),
    ("dependencies not enforced", "wf/runs.py", "    if problem:\n        write_status(run_id, problem, ok=False)\n        raise SystemExit(1)\n", ""),
    ("earlier commits accepted silently", "wf/runs.py", "if any(sha.startswith(c) for c in commits):", "if True:"),
    ("headroom passes on no data", "wf/runs.py", 'if any(v != v for v in (summary["oracle_gain_pp"]["estimate"], summary["oracle_gain_pp"]["ci_low"],', "if False and ((0,"),
    ("rule stop exits zero", "wf/runs.py", "return 1 if broken else RULE_STOP_EXIT if stop else 0", "return 1 if broken else 0"),
    ("headroom says PASS", "wf/runs.py", 'reported=None if broken else "headroom advisory",\n', "reported=None,\n"),
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
    # Decisions of 2026-11-17
    ("2026-11-17 A reads the sign of L", "wf/analysis/stats.py", '        said = gen_says_success(g["generated_class"])\n        if said is None:\n            continue\n        out.append', '        said = g.get("L", 0) > 0\n        out.append'),
    ("2026-11-17 Pilot A generations not saved", "wf/runs.py", '            append_jsonl(d / "generations.jsonl", [rec])\n            gens.append(rec)', '            gens.append(rec)'),
    ("2026-11-17 stop rule never fires", "wf/runs.py", 'not rep["scoring_agreement_failures"]["agreement"] >= C.SCORING_STOP_FAILURES_MIN', 'False'),
    ("2026-11-17 nats rule reads the lower bound", "wf/runs.py", 'rep["goal_agnostic"]["delta_L_ci"][1] < C.FUTILITY_UPPER_NATS', 'rep["goal_agnostic"]["delta_L_ci"][0] < C.FUTILITY_UPPER_NATS'),
    ("2026-11-17 gated model crashes the run", "wf/runs.py", 'except OSError as e:  # gated', 'except ValueError as e:  # gated'),
    ("2026-11-17 parser guesses the last line", "wf/scoring/prompts.py", '            return action or None\n    return None\n', '            return action or None\n    return lines[-1] if lines else None\n'),
    ("2026-11-17 malformed reply counts as critical", "wf/agent/react.py", '"decision_critical": not under_true_format_error and action_under_true != proposal,', '"decision_critical": action_under_true != proposal,'),
    ("2026-11-17 raw agent text dropped", "wf/agent/react.py", '            "proposal_raw": proposal_raw,\n', '            "proposal_raw": "",\n'),
    ("2026-11-17 c_hat sign flipped", "wf/scoring/logodds.py", '        return self.L_first - self.L\n', '        return self.L - self.L_first\n'),
    ("2026-11-17 prefix cache back on", "pace/_common.sh", '--no-enable-prefix-caching', '--enable-prefix-caching'),
    ("2026-11-17 headroom gates again", "wf/runs.py", "    return 1 if broken else 0\n", "    return 1 if broken else RULE_STOP_EXIT if summary['oracle_gain_pp']['estimate'] < C.HEADROOM_MIN_PP else 0\n"),
    ("2026-11-17 old headroom stop blocks", "wf/runs.py", ' and "PILOT_RULE_HEADROOM" not in _status_line(dep)', ""),
    ("2026-11-17 eligibility skips consumption", "wf/runs.py", '    require_passed(run_id, cfg, "pilotB_consumption")\n', ""),
    ("2026-11-17 parser ignores failure wording", "wf/scoring/parser.py", "        if _no_effect_words(body):\n            return \"noop_paraphrase\"\n        if verb in (", "        if verb in ("),
    ("2026-11-17 partial effects not counted as effects", "wf/analysis/stats.py", '\n                       "process_only", "different_effect")', ")"),
    ("2026-11-17 Pilot A keeps the old parser", "wf/runs.py", '"generated_class": parse_prediction(text, t.get("goal_object"), t["action"]),', '"generated_class": parse_observation(text, t.get("goal_object")),'),
    ("2026-11-17 audit shows the parser label", "wf/tools/parser_audit.py", 'w.writerow([i, actions.get(g["transition_id"], ""), g["generated"], "", ""])', 'w.writerow([i, actions.get(g["transition_id"], ""), g["generated"], parser_label(g["generated_class"]), ""])'),
    ("2026-11-17 audit ignores opposite readings", "wf/tools/parser_audit.py", " and not opposite\n", "\n"),
    ("2026-11-17 walk-there read as no effect", "wf/scoring/parser.py", '            return "different_effect"  # walks there instead of doing the action', '            return "noop_paraphrase"'),
    ("2026-11-17 collector samples contents failures", "wf/data/transitions.py", "            contents_holding_only=not ((i % k == 0) or invalid),\n", ""),
    ("2026-11-17 hidden state not saved", "wf/data/transitions.py", '        rec["hidden_state"] = hidden\n', '        rec["hidden_state"] = None\n'),
    ("2026-11-17 open-state failures not generated", "wf/data/transitions.py", '    if loc in facts.opened:\n        alts.append((f"open {loc}", "precondition", False))\n', ""),
    ("2026-11-17 replay check ignored", "wf/runs.py", "          and not verify[\"mismatches\"])", ")"),
    ("2026-11-17 run lock disabled", "wf/runs.py", "    _lock_run(run_id, d)\n", ""),
    ("2026-11-17 winning state dropped", "wf/runs.py", '        actions = [t["executed"] for t in ep["trace"]]\n', '        actions = [t["executed"] for t in ep["trace"]]\n        if ep["success"]:\n            actions = actions[:-1]\n'),
    ("2026-11-17 effect action read by whole reply", "wf/scoring/parser.py", "        if _no_effect_words(first) or _ALREADY_STATE.search(first):", "        if _no_effect_words(body) or _ALREADY_STATE.search(body):"),
    ("2026-11-17 steps gain sign flipped", "wf/analysis/stats.py", "[a[g] - b[g] for g in games],\n        games,\n        seed=seed,\n    )\n\n\ndef H_success_points", "[b[g] - a[g] for g in games],\n        games,\n        seed=seed,\n    )\n\n\ndef H_success_points"),
    ("2026-11-18 verification parts may be missing", "wf/tools/verify_items.py", "    if have != set(range(k)):\n", "    if False:\n"),
    ("2026-11-18 verification parts ignore the file hash", "wf/tools/verify_items.py", "    if r.get(\"transitions_sha256\") != sha:\n", "    if False:\n"),
    ("2026-11-18 scorer rounds to bfloat16", "wf/scoring/logodds.py", "logits[0, pos].float()", "logits[0, pos]"),
    ("2026-11-18 config run_id differs from its file", "wf/config/rq2_score.yaml", "run_id: rq2_score\n", "run_id: rq2\n"),
    # Confirmatory RQ3 runner (cap and yoke, primary set, shards)
    ("rq3 draws follow the step", "wf/rq3.py", "        k = self.chances\n", "        k = t\n"),
    ("rq3 no cap on false successes", "wf/rq3.py", "if valid or len(self.injections) >= self.cap:", "if valid:"),
    ("rq3 chance without a twin", "wf/rq3.py", "        if o_plus is None:  # nothing false can be shown, so this is not a chance\n            return None\n", ""),
    ("rq3 yoke ignores the verb", "wf/rq3.py", 'if verb != target["verb"] and i not in self.any_verb:', "if False:"),
    ("rq3 yoke goes before its partner", "wf/rq3.py", 'if i >= len(self.targets) or t < self.targets[i]["step"] or not valid or format_error:', "if i >= len(self.targets) or not valid or format_error:"),
    ("rq3 yoke takes a format error", "wf/rq3.py", 'if i >= len(self.targets) or t < self.targets[i]["step"] or not valid or format_error:', 'if i >= len(self.targets) or t < self.targets[i]["step"] or not valid:'),
    ("rq3 no fallback", "wf/rq3.py", "        if not working:\n            break\n", "        if True:\n            break\n"),
    ("rq3 replay skips the plan", "wf/rq3.py", "            if plan is not None:  # bring the plan", "            if False:  # bring the plan"),
    ("rq3 yoked to the other retention", "wf/rq3.py", 'saved[(gid, cond, "favorable", ret, seed)]', 'saved[(gid, cond, "favorable", "discarded", seed)]'),
    ("rq3 shards overlap", "wf/rq3.py", "    return games[i::n]\n", "    return games[i * len(games) // n:]\n"),
    ("rq3 resume reruns saved episodes", "wf/rq3.py", "jobs = [job for job in game_jobs(gid, seeds) if job not in saved]", "jobs = game_jobs(gid, seeds)"),
    ("rq3 episodes saved per game", "wf/rq3.py", '            append_jsonl(d / "episodes.jsonl", [ep])\n            saved[episode_key(ep)] = ep\n            new_eps += 1\n',
     '            saved[episode_key(ep)] = ep\n            new_eps += 1\n        append_jsonl(d / "episodes.jsonl", [saved[j] for j in jobs])\n'),
    ("rq3 job limit ignored", "wf/rq3.py", "            if max_eps and new_eps == max_eps:\n", "            if False:\n"),
    ("rq3 analysis accepts a missing shard", "wf/rq3.py", "    if indices != list(range(n)):\n", "    if False:\n"),
    ("rq3 analysis accepts an unfinished shard", "wf/rq3.py", 'if f"REPORTED ({SHARD_COMPLETE})" not in status:', "if False:"),
    ("rq3 analysis mixes commits", "wf/rq3.py", "    if len(shas) > 1:\n", "    if False:\n"),
    ("rq3 primary set needs one mode", "wf/analysis/rq3_stats.py", 'if all(by_unit[u][("favorable", r)]["delivered_errors"] >= 1 for r in RETENTIONS)]',
     'if any(by_unit[u][("favorable", r)]["delivered_errors"] >= 1 for r in RETENTIONS)]'),
    ("rq3 units pooled across games", "wf/analysis/rq3_stats.py", "        vals[u[0]].append(value(u))\n", "        vals[str(u)].append(value(u))\n"),
    ("rq3 interaction sign flipped", "wf/analysis/rq3_stats.py", 'lambda u: H(u, "retained") - H(u, "discarded")', 'lambda u: H(u, "discarded") - H(u, "retained")'),
    ("rq3 cost against the other mode's oracle", "wf/analysis/rq3_stats.py", 'e["condition"] == "oracle" and e["retention"] == ret]', 'e["condition"] == "oracle" and e["retention"] == "discarded"]'),
    ("rq3 job can requeue", "pace/rq3.sbatch", "#SBATCH --no-requeue\n", ""),
    # Newer runners (added 2026-11-20, documentation audit B7): rq1x, rq1v, mech, sizes, thinking, rq2 shards,
    # the shared-spot run, the audit analyses, provenance, pinned revisions and submit.sh.
    ("rq1x success effect reads failures", "wf/rq1x.py", '_effect(s, successes, require_agnostic=False, outcome="success")',
     "_effect(s, successes, require_agnostic=False)"),
    ("rq1v yes/no success effect reads failures", "wf/rq1v.py", 'control="yn_swapped", outcome="success")', 'control="yn_swapped")'),
    ("paired effect ignores the outcome asked for", "wf/analysis/stats.py", "if not _outcome_ok(meta, outcome):",
     'if not _outcome_ok(meta, "failure"):'),
    ("task line prepended when missing", "wf/scoring/prompts.py", "    if n != 1:\n        raise ValueError",
     "    if n == 0:\n        return f\"{task_line}\\n{history_text}\"\n    if n != 1:\n        raise ValueError"),
    ("thinking mock on any flag value", "wf/thinking.py", 'mock = os.environ.get("WF_MOCK_SCORE") == "1"',
     'mock = bool(os.environ.get("WF_MOCK_SCORE"))'),
    ("thinking analysis ignores reparse", "wf/thinking.py", 'reparse = reparse_answers(recs, items) if cfg.get("reparse") else None',
     "reparse = None"),
    ("size flips count NaN pairs", "wf/sizes.py", 'audit B4).\n    """\n    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}',
     'audit B4).\n    """\n    by = {(x["transition_id"], x["condition"]): x["L"] for x in s}'),
    ("size slope sorts NaN resamples in", "wf/sizes.py", "boots = sorted(b for b in draws if b == b)", "boots = sorted(draws)"),
    ("mech share removed is the ratio itself", "wf/mech.py", "lambda xs: 1 - sum(b for _, b in xs) / sum(a for a, _ in xs)",
     "lambda xs: sum(b for _, b in xs) / sum(a for a, _ in xs)"),
    ("shard analysis mixes commits", "wf/rq2.py", "        if len(shas) != 1:\n", "        if False:\n"),
    ("collector hidden-state check by substring", "wf/runs.py",
     'parsed[t["hidden_state"]] = set(hidden_locations(t["hidden_state"]))', 'parsed[t["hidden_state"]] = t["hidden_state"]'),
    ("unpinned model loads silently", "wf/constants.py", '    raise KeyError(f"model {model_id!r} has no pinned revision',
     '    return None\n    raise KeyError(f"model {model_id!r} has no pinned revision'),
    ("scorer ignores the pinned revision", "wf/scoring/logodds.py",
     "self.revision = REVISIONS.get(model_name) if self.mock else revision_for(model_name)", "self.revision = None"),
    ("provenance omits input hashes", "wf/provenance.py", "files[str(path)] = file_sha(path) if path.exists() else None",
     "files[str(path)] = None"),
    ("shared-spot run scores the primary set", "wf/rq2_shared.py", '["agnostic_all"] if t.get("shared_location")]', '["agnostic_all"] if not t.get("shared_location")]'),
    ("shared-spot analysis leaves its scores out of (b)", "wf/rq2_shared.py", "_effect(mine_base + mine_extra, primary + shared)",
     "_effect(mine_base, primary + shared)"),
    ("audit flow skips the twin stage", "wf/audit.py", '("success twin attached", lambda t: t.get("candidates") is not None),',
     '("success twin attached", lambda t: True),'),
    ("audit instance items matched by name", "wf/audit.py", "if _type(i) == _type(m[1])", "if i == m[1]"),
    ("submit.sh overrides a script's GPU request", "pace/submit.sh",
     'if [[ -n "${PACE_GRES:-}" ]] && ! script_sets gres && ! script_sets partition; then',
     'if [[ -n "${PACE_GRES:-}" ]]; then'),
    # Mechanism battery (wf/mech2.py, 2026-11-20): templates, positions, items, contrasts, categories, gate, resume.
    ('mech2 desire_keep not reversed', 'wf/mech2.py', 'DESIRE_KEEP = "Your task is to: make sure that the {obj} is not {pp}', 'DESIRE_KEEP = "Your task is to: make sure that the {obj} is {pp}'),
    ('mech2 desire_check presupposes the object', 'wf/mech2.py', 'DESIRE_CHECK = "Your task is to: find out whether {a} {obj} can be', 'DESIRE_CHECK = "Your task is to: find out whether the {obj} can be'),
    ('mech2 ignore sentence on the real task', 'wf/mech2.py', 'MENTION_IGNORE.format(swapped=t["swapped_task_line"], obj=obj)', 'MENTION_IGNORE.format(swapped=t["task_line"], obj=obj)'),
    ('mech2 done quotes the swapped task', 'wf/mech2.py', 'return DONE.format(task=_task_text(t["task_line"]))', 'return DONE.format(task=_task_text(t["swapped_task_line"]))'),
    ('mech2 described arm keeps the name', 'wf/mech2.py', 'return rq1x.describe_line(t["task_line"], obj) if rq1x.desc_ok(t) else None', 'return t["task_line"] if rq1x.desc_ok(t) else None'),
    ('mech2 start puts the task line last', 'wf/mech2.py', 'return "\\n".join([line] + lines)', 'return "\\n".join(lines + [line])'),
    ('mech2 position arms after step 0', 'wf/mech2.py', '    if arm in POSITION_ARMS and t.get("history_len", 0) > 0:\n        return None', '    if False:\n        return None'),
    ('mech2 position arms dropped at step 0 too', 'wf/mech2.py', '    if arm in POSITION_ARMS and t.get("history_len", 0) > 0:\n        return None', '    if arm in POSITION_ARMS and t.get("history_len", 0) >= 0:\n        return None'),
    ('mech2 missing score read from another arm', 'wf/mech2.py', '    rec = by.get((t["id"], condition(arm, fmt)))\n    if rec is None:\n        return None', '    rec = by.get((t["id"], condition(arm, fmt))) or by.get((t["id"], condition("aligned", fmt)))\n    if rec is None:\n        return None'),
    ('mech2 yes/no keeps the observation cue', 'wf/mech2.py', 'last["content"][: -len("Next observation:")] + rq1v.YN_QUESTION', 'last["content"] + "\\n" + rq1v.YN_QUESTION'),
    ('mech2 sample allocated equally', 'wf/mech2.py', 'exact = {k: len(v) * n / total for k, v in strata.items()}', 'exact = {k: n / len(strata) for k, v in strata.items()}'),
    ('mech2 sample follows input order', 'wf/mech2.py', 'chosen |= {t["id"] for t in sorted(ts, key=lambda t: _rank(t["id"]))[:quota[k]]}', 'chosen |= {t["id"] for t in ts[:quota[k]]}'),
    ('mech2 step-0 items not added', 'wf/mech2.py', '"union": [t for t in primary if t["id"] in sample | step0]', '"union": [t for t in primary if t["id"] in sample]'),
    ('mech2 contrast sign flipped', 'wf/mech2.py', 'return sum(c * v for (c, _), v in zip(terms, vals))', 'return -sum(c * v for (c, _), v in zip(terms, vals))'),
    ('mech2 ratio over its own size', 'wf/mech2.py', '    den = _mean([r for _, r in xs])\n', '    den = _mean([abs(d) for d, _ in xs])\n'),
    ('mech2 Holm skipped', 'wf/mech2.py', 'r["holm_p"] = adj[name]', 'r["holm_p"] = r["p"]'),
    ('mech2 null band widened', 'wf/mech2.py', 'TRIVIAL_BAND = 0.25', 'TRIVIAL_BAND = 0.5'),
    ('mech2 large threshold lowered', 'wf/mech2.py', 'LARGE = 0.5\n', 'LARGE = 0.3\n'),
    ('mech2 P5 against swapped', 'wf/mech2.py', '"terms": ((1, "done"), (-1, "cancelled")), "population": "p5"', '"terms": ((1, "done"), (-1, "swapped")), "population": "p5"'),
    ('mech2 P5 includes examine tasks', 'wf/mech2.py', '    return (task_type(t["task_line"]) in PLACEMENT_TYPES and not takes_from_target_type(t)', '    return (not takes_from_target_type(t)'),
    ('mech2 item share counts ties toward Yes', 'wf/mech2.py', 'lambda xs: sum(x > 0 for x in xs) / len(xs)', 'lambda xs: sum(x >= 0 for x in xs) / len(xs)'),
    ('mech2 gate failure ignored', 'wf/mech2.py', '    if not rec["ok"]:\n        raise TokenizationMismatch', '    if False:\n        raise TokenizationMismatch'),
    ('mech2 gate accepts any revision', 'wf/mech2.py', 'if not scorer.revision or scorer.revision != C.REVISIONS.get(model_name):', 'if not scorer.revision:'),
    ('mech2 gate accepts a multi-token answer', 'wf/mech2.py', '        if len(ids) != 1:\n', '        if len(ids) < 1:\n'),
    ('mech2 resume rescores saved work', 'wf/mech2.py', 'todo = [(t, a, f) for t, a, f in work[tag] if (tag, t["id"], condition(a, f)) not in done]', 'todo = list(work[tag])'),
    ('mech2 done contradicted with a target unseen', 'wf/mech2.py', 'if not targets or any(r not in facts.contents for r in targets):', 'if not targets or all(r not in facts.contents for r in targets):'),
    ('mech2 overlap read against set 1 only', 'wf/mech2.py', 'kind = ("aligned" if t["task_line"].strip() in lines[k] else', 'kind = ("aligned" if t["task_line"].strip() in lines[1] else'),
    ('mech2 knowledge split reversed', 'wf/mech2.py', 'return None if a is None or b is None else a < 0 < b', 'return None if a is None or b is None else a > 0'),
    # Mechanism battery, review 1 (2026-11-20): the review's twenty mutants R1-R20, adapted to the amended code where
    # their text moved (R1, R11, R14, R15, R17, R20), and mutants of the amendment (zones, joint rule, P5/P7 items,
    # the negated arm, Mistral).
    ('mech2 R1 non-trivial zones read without significance', 'wf/mech2.py', '    if "Z" not in zs and not significant:\n        return "INCONCLUSIVE", None', '    if False:\n        return "INCONCLUSIVE", None'),
    ('mech2 R2 P6 on the sample', 'wf/mech2.py', '"population": "step0", "reference"', '"population": "sample", "reference"'),
    ('mech2 R3 desire participle always taken', 'wf/mech2.py', 'pp=PARTICIPLE[verb], prep=', 'pp="taken", prep='),
    ('mech2 R4 move preposition from', 'wf/mech2.py', '"move": "to", "heat": "with"', '"move": "from", "heat": "with"'),
    ('mech2 R5 yes/no family read on cand', 'wf/mech2.py', '"primary_yes_no": primary_family(by, pops, "yn", **kw)', '"primary_yes_no": primary_family(by, pops, "cand", **kw)'),
    ('mech2 R6 r denominator always cand', 'wf/mech2.py', 'ref = _combo(by, t, reference, fmt, field) if reference else 0.0', 'ref = _combo(by, t, reference, "cand", field) if reference else 0.0'),
    ('mech2 R7 sigma read as L', 'wf/mech2.py', 'v = sigmoid(rec["L"]) if field == "s" else rec.get(field)', 'v = rec["L"] if field == "s" else rec.get(field)'),
    ('mech2 R8 knowledge: either answer right', 'wf/mech2.py', 'return None if a is None or b is None else a < 0 < b', 'return None if a is None or b is None else (a < 0 or 0 < b)'),
    ('mech2 R9 done contradicted with one target seen', 'wf/mech2.py', 'if not targets or any(r not in facts.contents for r in targets):', 'if not targets or all(r not in facts.contents for r in targets):'),
    ('mech2 R10 smallest remainder gets the extra seat', 'wf/mech2.py', 'key=lambda k: (-(exact[k] - quota[k]), _rank', 'key=lambda k: ((exact[k] - quota[k]), _rank'),
    ('mech2 R11 any overlap read as consistent', 'wf/mech2.py', '    if set(zs) <= predicted:\n        return "consistent"', '    if set(zs) & predicted:\n        return "consistent"'),
    ('mech2 R12 permutation flips items, not games', 'wf/mech2.py', '"p": pair_permutation([d for _, d, _ in rows], n=perms, seed=C.RNG_SEED, cluster_ids=games)', '"p": pair_permutation([d for _, d, _ in rows], n=perms, seed=C.RNG_SEED)'),
    ('mech2 R13 later includes step 0', 'wf/mech2.py', '"later": [t for t in s if t.get("history_len", 0) > 0]', '"later": [t for t in s if t.get("history_len", 0) >= 0]'),
    ('mech2 R14 gate checks only the first item', 'wf/mech2.py', '    for t in items.values():\n        msgs = prompt(arm_messages(t, "aligned"))', '    for t in list(items.values())[:1]:\n        msgs = prompt(arm_messages(t, "aligned"))'),
    ('mech2 R15 gate skips yes/no prompts', 'wf/mech2.py', '        if fmt != "yn":\n            continue\n        msgs = prompt(arm_messages(t, arm, "yn"))', '        if fmt != "cand":\n            continue\n        msgs = prompt(arm_messages(t, arm, "yn"))'),
    ('mech2 R16 resume ignores item hashes', 'wf/mech2.py', 'if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):', 'if False:'),
    ('mech2 R17 ratio accepts negative denominators', 'wf/mech2.py', 'return _mean([d for d, _ in xs]) / den if den > 0 else float("nan")', 'return _mean([d for d, _ in xs]) / den if den != 0 else float("nan")'),
    ('mech2 R18 mention_ignore on neutral line', 'wf/mech2.py', 'return MENTION_IGNORE.format(swapped=t["swapped_task_line"], obj=obj)', 'return MENTION_IGNORE.format(swapped=rq1.ABSENT_LINE, obj=obj)'),
    ('mech2 R19 cancelled same as done', 'wf/mech2.py', 'CANCELLED = "Your task was to: {task} The user has cancelled it; nothing is left to do."', 'CANCELLED = "Your task was to: {task} It is already done; nothing is left to do."'),
    ('mech2 R20 Holm over six (P6 dropped)', 'wf/mech2.py', '    adj = holm({name: (r["p"] if r["p"] == r["p"] else 1.0) for name, r in res.items()})',
     '    adj = holm({name: r["p"] for name, r in res.items() if not name.startswith("P6")})\n    adj.setdefault("P6_position_step0", res["P6_position_step0"]["p"])'),
    ('mech2 trivial band read as a contradiction of 0', 'wf/mech2.py', '    if x <= TRIVIAL_BAND:\n        return "Z"', '    if x <= 0:\n        return "Z"'),
    ('mech2 dropped draws ignored', 'wf/mech2.py', 'dropped <= DROPPED_DRAWS_MAX * B)', 'True)'),
    ('mech2 P6 counts for evidence', 'wf/mech2.py', '"P8_negated_vs_affirmed": _N},\n    "H3_salience"', '"P8_negated_vs_affirmed": _N, "P6_position_step0": _Z},\n    "H3_salience"'),
    ('mech2 untested read as consistent', 'wf/mech2.py', '    return "consistent" if _risky(h, rows) else "untested"', '    return "consistent"'),
    ('mech2 one format decides', 'wf/mech2.py', '    if a == b and a in ("consistent", "contradicted"):\n        return a', '    if a in ("consistent", "contradicted"):\n        return a'),
    ('mech2 mixture never named', 'wf/mech2.py', 'text = ("no single account; mixture or an account not on the list (each content account is contradicted, so "', 'text = ("accounts not contradicted: none (each content account is contradicted, so "'),
    ('mech2 P5 keeps takes from a target-type receptacle', 'wf/mech2.py', '    return re.sub(r"\\s+\\d+$", "", p[2]) == parse_task_line(t["task_line"]).receptacle\n\n\ndef p5_eligible', '    return False\n\n\ndef p5_eligible'),
    ('mech2 P5 keeps contradicted items', 'wf/mech2.py', '            and not done_contradicted(t))', '            and True)'),
    ('mech2 P7 keeps contradicted claims', 'wf/mech2.py', 'return is_take(t) and negated_status(t)[1] != "shown, holds it"', 'return is_take(t)'),
    ('mech2 negated built for every verb', 'wf/mech2.py', '        if not p or p[0] != "take" or p[1] != obj:\n            return None\n        sw = t["swapped_task_line"]', '        if not p or p[1] != obj:\n            return None\n        sw = t["swapped_task_line"]'),
    ('mech2 negated drops the user', 'wf/mech2.py', 'NEGATED = "{swapped} The user says there is no {obj} in the {rec}."', 'NEGATED = "{swapped} There is no {obj} in the {rec}."'),
    ('mech2 Mistral without its template settings', 'wf/mech2.py', 'kw = {"attn_implementation": spec.attn_implementation, "template_kwargs": dict(spec.template_kwargs)}', 'kw = {}'),
    ('mech2 folding model not folded', 'wf/mech2.py', 'return spec.hf_id, kw, (families.fold_system if spec.system == "fold" else (lambda msgs: msgs))', 'return spec.hf_id, kw, (lambda msgs: msgs)'),
    ('mech2 scoring skips the template handling', 'wf/mech2.py', '            msgs = prompt(arm_messages(t, arm, fmt))', '            msgs = arm_messages(t, arm, fmt)'),
    # Mechanism battery, review 2 (2026-11-20, before any score): the review's 22 mutants Q1-Q22, adapted where the
    # amended code moved their text (Q4 and Q10 now remove an allowed zone, since the amended cells allow what the
    # original mutants added; Q5-Q8 follow the one-line-per-account table; Q13 the two-format rule's helper; Q19 the
    # shared desire_check template), and mutants of the amendment (minimal pairs, P8, verdicts, summary, integrity,
    # Mistral's arms, secondary readouts).
    ('mech2 Q1 NaN p not replaced by 1 (family shrinks)', 'wf/mech2.py', 'adj = holm({name: (r["p"] if r["p"] == r["p"] else 1.0) for name, r in res.items()})', 'adj = holm({name: r["p"] for name, r in res.items()})'),
    ('mech2 Q2 P6 on sample step-0 only', 'wf/mech2.py', '"step0": [t for t in union if t["id"] in step0],', '"step0": [t for t in s if t["id"] in step0],'),
    ('mech2 Q3 strata by game only', 'wf/mech2.py', 'strata[(t["game_id"], t.get("determination_reason") or "")].append(t)', 'strata[(t["game_id"], "")].append(t)'),
    ('mech2 Q4 H2 P1 without L', 'wf/mech2.py', '"H2_evidence": {"P1_keep_vs_ignore": _ZSL,', '"H2_evidence": {"P1_keep_vs_ignore": _ZS,'),
    ('mech2 Q5 H3 P5 allows N', 'wf/mech2.py', '"P4_need_without_name": _P, "P5_done_vs_cancelled": _Z, "P7_negated_existence": _P, "P8_negated_vs_affirmed": _Z},\n    "H4_wishful"', '"P4_need_without_name": _P, "P5_done_vs_cancelled": _NZ, "P7_negated_existence": _P, "P8_negated_vs_affirmed": _Z},\n    "H4_wishful"'),
    ('mech2 Q6 H4 P7 allows N', 'wf/mech2.py', '"P7_negated_existence": _Z, "P8_negated_vs_affirmed": _Z},\n    "H5_policy_blend"', '"P7_negated_existence": _NZ, "P8_negated_vs_affirmed": _Z},\n    "H5_policy_blend"'),
    ('mech2 Q7 H5 P3 only Z', 'wf/mech2.py', '"H5_policy_blend": {"P1_keep_vs_ignore": _NZ, "P2_hope_polarity": _Z, "P3_mention_without_need": _NZ,', '"H5_policy_blend": {"P1_keep_vs_ignore": _NZ, "P2_hope_polarity": _Z, "P3_mention_without_need": _Z,'),
    ('mech2 Q8 H6_lexical P4 allows S', 'wf/mech2.py', '"P4_need_without_name": _Z, "P5_done_vs_cancelled": _Z, "P7_negated_existence": _P, "P8_negated_vs_affirmed": _Z},\n    "H6_position"', '"P4_need_without_name": _ZS, "P5_done_vs_cancelled": _Z, "P7_negated_existence": _P, "P8_negated_vs_affirmed": _Z},\n    "H6_position"'),
    ('mech2 Q9 H6_position allows only N', 'wf/mech2.py', '"H6_position": {"P6_position_step0": frozenset({"N", "S", "L"})}', '"H6_position": {"P6_position_step0": frozenset({"N"})}'),
    ('mech2 Q10 H2 P2 allows S', 'wf/mech2.py', '"H2_evidence": {"P1_keep_vs_ignore": _ZSL, "P2_hope_polarity": _Z,', '"H2_evidence": {"P1_keep_vs_ignore": _ZSL, "P2_hope_polarity": _ZS,'),
    ('mech2 Q11 dropped-draw rule uses 2% of n_items', 'wf/mech2.py', 'dropped <= DROPPED_DRAWS_MAX * B)', 'dropped <= DROPPED_DRAWS_MAX * len(rows))'),
    ('mech2 Q12 r interval from mean bootstrap', 'wf/mech2.py', '"ratio"] = {"estimate": rb["estimate"], "ci": [rb["ci_low"], rb["ci_high"]]', '"ratio"] = {"estimate": rb["estimate"], "ci": [boot["ci_low"] / out["reference_effect"], boot["ci_high"] / out["reference_effect"]]'),
    ('mech2 Q13 contradicted only needs one format', 'wf/mech2.py', '    if {a, b} == {"consistent", "contradicted"}:\n        return "format_dependent"', '    if "contradicted" in (a, b) and "consistent" not in (a, b):\n        return "contradicted"\n    if {a, b} == {"consistent", "contradicted"}:\n        return "format_dependent"'),
    ('mech2 Q14 negated_status reads exact instances', 'wf/mech2.py', 'seen = "shown, holds it" if any(re.sub(r"\\s+\\d+$", "", o) == obj for o in facts.contents[rec]) else "shown, without it"', 'seen = "shown, holds it" if any(o == obj for o in facts.contents[rec]) else "shown, without it"'),
    ('mech2 Q15 all-placement done against swapped', 'wf/mech2.py', '"done_vs_cancelled_all_placement": contrast(by, pops["sample_placement"], ((1, "done"), (-1, "cancelled"))', '"done_vs_cancelled_all_placement": contrast(by, pops["sample_placement"], ((1, "done"), (-1, "swapped"))'),
    ('mech2 Q16 analysis default models omit mistral', 'wf/mech2.py', 'tags = list(cfg.get("models", PRIMARY_MODELS + SECONDARY_MODELS))', 'tags = list(cfg.get("models", PRIMARY_MODELS))'),
    ('mech2 Q17 zones_of returns only endpoints', 'wf/mech2.py', '    return ZONES[min(a, b):max(a, b) + 1]', '    return tuple(sorted({ZONES[a], ZONES[b]}, key=ZONES.index))'),
    ('mech2 Q18 categorize ignores estimable', 'wf/mech2.py', 'if est != est or lo != lo or hi != hi or not r.get("estimable", False):', 'if est != est or lo != lo or hi != hi:'),
    ('mech2 Q19 desire_check article always a', 'wf/mech2.py', 'return hope.format(a=rq1._article(obj), obj=obj', 'return hope.format(a="a", obj=obj'),
    ('mech2 Q20 knowledge split on step-0 terms too', 'wf/mech2.py', '**{n: s["terms"] for n, s in PRIMARY.items() if s["population"] != "step0"},', '**{n: s["terms"] for n, s in PRIMARY.items()},'),
    ('mech2 Q21 P7 by history on takes of any object', 'wf/mech2.py', 'if is_take(t) and negated_status(t)[1] == k]', 'if t["action"].startswith("take ") and negated_status(t)[1] == k]'),
    ('mech2 Q22 TRIVIAL label replaced by POS_SMALL when significant', 'wf/mech2.py', '        return ("TRIVIAL" if significant else "NULL"), zs', '        return ("POS_SMALL" if significant else "NULL"), ("S",) if significant else zs'),
    ('mech2 N1 P1 against swapped', 'wf/mech2.py', '"P1_keep_vs_ignore": {"terms": ((1, "desire_keep_sw"), (-1, "mention_ignore"))', '"P1_keep_vs_ignore": {"terms": ((1, "desire_keep_sw"), (-1, "swapped"))'),
    ('mech2 N2 P2 against swapped', 'wf/mech2.py', '"P2_hope_polarity": {"terms": ((1, "desire_check"), (-1, "desire_check_can"))', '"P2_hope_polarity": {"terms": ((1, "desire_check"), (-1, "swapped"))'),
    ('mech2 N3 P8 against swapped', 'wf/mech2.py', '"P8_negated_vs_affirmed": {"terms": ((1, "negated"), (-1, "affirmed"))', '"P8_negated_vs_affirmed": {"terms": ((1, "negated"), (-1, "swapped"))'),
    ('mech2 N4 desire_check_can hopes it cannot', 'wf/mech2.py', 'the user hopes it can."', 'the user hopes it cannot."'),
    ('mech2 N5 desire_keep_sw without the swapped task', 'wf/mech2.py', 'DESIRE_KEEP_SW = "{swapped} Make sure', 'DESIRE_KEEP_SW = "Your task is to: Make sure'),
    ('mech2 N6 affirmed denies too', 'wf/mech2.py', 'AFFIRMED = "{swapped} The user says there is {a} {obj} in the {rec}."', 'AFFIRMED = "{swapped} The user says there is no {obj} in the {rec}."'),
    ('mech2 N7 Mistral scored on every arm', 'wf/mech2.py', 'return ARMS if tag in PRIMARY_MODELS else PRIMARY_ARMS', 'return ARMS'),
    ('mech2 N8 rival readings left out', 'wf/mech2.py', '            for h, preds in {**PREDICTIONS, **RIVAL_READINGS}.items()}\n    return _account_rows(rows)', '            for h, preds in PREDICTIONS.items()}\n    return _account_rows({**rows, **{r: {} for r in RIVAL_READINGS}})'),
    ('mech2 N9 consistent without a rival contradicted', 'wf/mech2.py', 'and any(rows[o].get(c) == "contradicted" for o in rows if o != h)]', 'and any(rows[o].get(c) != "undecided" for o in rows)]'),
    ('mech2 N10 untested listed as nothing', 'wf/mech2.py', '"content_accounts_untested": by["untested"],', '"content_accounts_untested": [],'),
    ('mech2 N11 contradicted read as rejected', 'wf/mech2.py', "' (cannot be the sole account)' if v == 'contradicted'", "' (rejected)' if v == 'contradicted'"),
    ('mech2 N12 integrity ignores item hashes', 'wf/mech2.py', 'x["transition_id"] in items and x.get("item_hash") != hashes[x["transition_id"]] for x in mine', 'x["transition_id"] in items and False for x in mine'),
    ('mech2 N13 integrity mismatch never refuses', 'wf/mech2.py', '    out["refuse"] = any(out[t]["status"] == "mismatch" or', '    out["refuse"] = any(False or'),
    ('mech2 N14 analysis ignores the integrity check', 'wf/mech2.py', '    if integrity["refuse"]:', '    if False:'),
    ('mech2 N15 item files not recorded', 'wf/mech2.py', 'd = _init_run(run_id, cfg, inputs=tuple(runs_root() / s / "transitions.jsonl" for s in sources))', 'd = _init_run(run_id, cfg, inputs=())'),
    ('mech2 N16 duplicates not counted', 'wf/mech2.py', '"n_duplicates": sum(n - 1 for n in keys.values() if n > 1),', '"n_duplicates": 0,'),
    ('mech2 N17 missing scores not counted', 'wf/mech2.py', '"n_missing": len(expected - set(keys)),', '"n_missing": 0,'),
    ('mech2 N18 P5 against the aligned effect, not cancelled', 'wf/mech2.py', 'reference=((1, "cancelled"), (-1, "swapped")), **kw) for f in both},', 'reference=ref, **kw) for f in both},'),
    ('mech2 N19 contents-not-shown reads every take item', 'wf/mech2.py', 'not_shown = [t for t in pops["sample_take"] if negated_status(t)[1] == "contents not shown"]', 'not_shown = pops["sample_take"]'),
    ('mech2 N20 first-token split flagged on agreement', 'wf/mech2.py', 'ft["estimate"] * full < 0)', 'ft["estimate"] * full > 0)'),
    ('mech2 N21 mixture readout reads cand for yes/no', 'wf/mech2.py', '(("cand", "primary"), ("yn", "primary_yes_no"))', '(("cand", "primary"), ("yn", "primary"))'),
    ('mech2 N22 status line without yes/no categories', 'wf/mech2.py', "({v['category']}; yes/no {yn[k]['category']})", "({v['category']})"),
    ('mech2 N23 old P1 reads the new keep line', 'wf/mech2.py', '"old_P1_desire_keep_vs_ignore": {f: contrast(by, pops["sample"], ((1, "desire_keep"), (-1, "mention_ignore"))', '"old_P1_desire_keep_vs_ignore": {f: contrast(by, pops["sample"], ((1, "desire_keep_sw"), (-1, "mention_ignore"))'),
    ('mech2 N24 separations read overlap as separation', 'wf/mech2.py', '"separate": [c for c in both if not PREDICTIONS[a][c] & PREDICTIONS[b][c]],', '"separate": [c for c in both if PREDICTIONS[a][c] != PREDICTIONS[b][c]],'),
    ('mech2 N25 affirmed with "a" for every object', 'wf/mech2.py', 'return AFFIRMED.format(swapped=sw, a=rq1._article(obj)', 'return AFFIRMED.format(swapped=sw, a="a"'),
    ('mech2 N26 H4 P2 allows Z', 'wf/mech2.py', '"H4_wishful": {"P1_keep_vs_ignore": _N, "P2_hope_polarity": _N,', '"H4_wishful": {"P1_keep_vs_ignore": _N, "P2_hope_polarity": _NZ,'),
    # Mechanism battery, review 3 (2026-11-21, before any score): the review's mutants A1-A22, adapted where the amended
    # code moved their text (A4: the refusal list now ends with the primary-arm NaN count; A13: the mixture readout is
    # followed by M; A16: the affirmed line reads the swapped line from `sw`; A18: the NaN count goes through
    # `_undefined`). A15 (analysis reads a mismatched secondary model) and A19 (H6_position consistent without a rival)
    # are equivalent and left out: a mismatch of any model refuses the analysis before `scored` is read, and an
    # H6_position consistent on P6 (zones within N, S, L) always contradicts the rival "no position effect" (Z).
    ('mech2 A1 rival contradicted on any contrast counts', 'wf/mech2.py', 'and any(rows[o].get(c) == "contradicted" for o in rows if o != h)]', 'and any("contradicted" in rows[o].values() for o in rows if o != h)]'),
    ('mech2 A2 category from raw p, not Holm', 'wf/mech2.py', 'r["category"], zs = categorize(r, adj[name])', 'r["category"], zs = categorize(r, r["p"])'),
    ('mech2 A3 P8 on the whole sample', 'wf/mech2.py', '"P8_negated_vs_affirmed": {"terms": ((1, "negated"), (-1, "affirmed")), "population": "sample_take"', '"P8_negated_vs_affirmed": {"terms": ((1, "negated"), (-1, "affirmed")), "population": "sample"'),
    ('mech2 A4 integrity ignores unexpected scores', 'wf/mech2.py', '"n_missing", "n_unexpected",\n', '"n_missing",\n'),
    ('mech2 A5 integrity ignores unknown items', 'wf/mech2.py', 'for k in ("n_unknown_items", "n_hash_mismatch",', 'for k in ("n_hash_mismatch",'),
    ('mech2 A6 untested accounts not counted as fitting', 'wf/mech2.py', 'fitting = by["consistent"] + by["untested"]', 'fitting = by["consistent"]'),
    ('mech2 A7 rival reading taken from H6_position row', 'wf/mech2.py', 'return ro["rivals"][h] if h in RIVAL_READINGS else ro[h]["per_contrast"]', 'return ro["H6_position"]["per_contrast"] if h in RIVAL_READINGS else ro[h]["per_contrast"]'),
    ('mech2 A8 first-token split against yes/no full L', 'wf/mech2.py', 'full = block["primary"]["contrasts"][name]["estimate"]', 'full = block["primary_yes_no"]["contrasts"][name]["estimate"]'),
    ('mech2 A9 P1 reference is aligned minus mention_ignore', 'wf/mech2.py', '"P1_keep_vs_ignore": {"terms": ((1, "desire_keep_sw"), (-1, "mention_ignore")), "population": "sample", "reference": _REF}', '"P1_keep_vs_ignore": {"terms": ((1, "desire_keep_sw"), (-1, "mention_ignore")), "population": "sample", "reference": ((1, "aligned"), (-1, "mention_ignore"))}'),
    ('mech2 A10 rival P6 band includes N', 'wf/mech2.py', 'RIVAL_READINGS: dict[str, dict[str, frozenset[str]]] = {"no_position_effect": {"P6_position_step0": _Z}}', 'RIVAL_READINGS: dict[str, dict[str, frozenset[str]]] = {"no_position_effect": {"P6_position_step0": _NZ}}'),
    ('mech2 A11 joint position verdict from cand only', 'wf/mech2.py', '"position": out["H6_position"]["verdict"],', '"position": cand["H6_position"]["verdict"],'),
    ('mech2 A12 contents-not-shown keeps shown-without', 'wf/mech2.py', 'not_shown = [t for t in pops["sample_take"] if negated_status(t)[1] == "contents not shown"]', 'not_shown = [t for t in pops["sample_take"] if negated_status(t)[1] != "shown, holds it"]'),
    ('mech2 A13 mixture readout drops P5', 'wf/mech2.py', 'for name in ("P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed")}', 'for name in ("P7_negated_existence", "P8_negated_vs_affirmed")}'),
    ('mech2 A14 secondary not-scored refuses', 'wf/mech2.py', 'or (t in PRIMARY_MODELS and out[t]["status"] == "not scored")', 'or (out[t]["status"] == "not scored")'),
    ('mech2 A16 negated/affirmed receptacle without its number', 'wf/mech2.py', 'return AFFIRMED.format(swapped=sw, a=rq1._article(obj), obj=obj, rec=p[2])', 'return AFFIRMED.format(swapped=sw, a=rq1._article(obj), obj=obj, rec=p[2].split()[0])'),
    ('mech2 A17 desire_keep_sw lower-case sentence', 'wf/mech2.py', 'DESIRE_KEEP_SW = "{swapped} Make sure that', 'DESIRE_KEEP_SW = "{swapped} make sure that'),
    ('mech2 A18 undefined L never counted', 'wf/mech2.py', '"n_undefined_L": sum(_undefined(x) for x in mine),', '"n_undefined_L": 0,'),
    ('mech2 A20 old P2 secondary against can', 'wf/mech2.py', '"old_P2_desire_check_vs_swapped": {f: contrast(by, pops["sample"], ((1, "desire_check"), (-1, "swapped"))', '"old_P2_desire_check_vs_swapped": {f: contrast(by, pops["sample"], ((1, "desire_check"), (-1, "desire_check_can"))'),
    ('mech2 A21 P5 lift ratio on all placement items', 'wf/mech2.py', 'f: contrast(by, pops["p5"], PRIMARY["P5_done_vs_cancelled"]["terms"], f,', 'f: contrast(by, pops["sample_placement"], PRIMARY["P5_done_vs_cancelled"]["terms"], f,'),
    ('mech2 A22 mixture text when any account is contradicted', 'wf/mech2.py', '    if not fitting:\n', '    if by["contradicted"]:\n'),
    # Mutants of the amendment after review 3: M and the two summary sentences, the risky rule and the labels, H2 on
    # P3, the placebo pair and the named secondaries, Mistral's jobs, the integrity changes, H7_check's readouts.
    ('mech2 T1 M is P7', 'wf/mech2.py', 'MENTION_TERMS = ((0.5, "negated"), (0.5, "affirmed"), (-1, "swapped"))', 'MENTION_TERMS = ((1, "negated"), (-1, "swapped"))'),
    ('mech2 T2 M above the band in one format suffices', 'wf/mech2.py', '    for f in FORMATS:\n        r = (res_by_format.get(f) or {}).get("ratio") or {}', '    for f in FORMATS[:1]:\n        r = (res_by_format.get(f) or {}).get("ratio") or {}'),
    ('mech2 T3 M read above zero, not above the band', 'wf/mech2.py', 'lambda lo, hi: lo > TRIVIAL_BAND)', 'lambda lo, hi: lo > 0)'),
    ('mech2 T4 salience sentence without the claims-only condition', 'wf/mech2.py', 'mention_remains = claims_only and _mention_above_band(mention)', 'mention_remains = bool(h3_against) and _mention_above_band(mention)'),
    ('mech2 T5 claims-only allows other contrasts', 'wf/mech2.py', 'set(h3_against) <= set(CLAIM_CONTRASTS)', 'bool(set(h3_against) & set(CLAIM_CONTRASTS))'),
    ('mech2 T6 claims sentence without H2 consistent', 'wf/mech2.py', 'claims_read = claims_only and out["H2_evidence"]["verdict"] == "consistent"', 'claims_read = claims_only'),
    ('mech2 T7 risky rule dropped', 'wf/mech2.py', '    return "consistent" if _risky(h, rows) else "untested"', '    return "consistent" if _discriminating(h, rows) else "untested"'),
    ('mech2 T8 risky read as excluding N', 'wf/mech2.py', 'if "Z" not in _allowed(h, c)]', 'if "N" not in _allowed(h, c)]'),
    ('mech2 T9 label hides the contrasts that allow zero', 'wf/mech2.py', 'rest = [c for c in disc if c not in risky]', 'rest = []'),
    ('mech2 T10 untested reason from risky contrasts', 'wf/mech2.py', '"consistent only where it allows zero" if disc else', '"consistent only where it allows zero" if risky else'),
    ('mech2 T11 H2 P3 back to positive only', 'wf/mech2.py', '"P2_hope_polarity": _Z, "P3_mention_without_need": _ZSL,', '"P2_hope_polarity": _Z, "P3_mention_without_need": _P,'),
    ('mech2 T12 placebo denies the action object', 'wf/mech2.py', 'return PLACEBO_NEGATED.format(swapped=sw, swobj=swobj, rec=p[2])', 'return PLACEBO_NEGATED.format(swapped=sw, swobj=obj, rec=p[2])'),
    ('mech2 T13 placebo article of the action object', 'wf/mech2.py', 'a=rq1._article(swobj)', 'a=rq1._article(obj)'),
    ('mech2 T14 placebo readout reversed', 'wf/mech2.py', 'PLACEBO_TERMS = ((1, "placebo_negated"), (-1, "placebo_affirmed"))', 'PLACEBO_TERMS = ((-1, "placebo_negated"), (1, "placebo_affirmed"))'),
    ('mech2 T15 P8 net adds the placebo', 'wf/mech2.py', 'P8_NET_TERMS = ((1, "negated"), (-1, "affirmed"), (-1, "placebo_negated"), (1, "placebo_affirmed"))', 'P8_NET_TERMS = ((1, "negated"), (-1, "affirmed"), (1, "placebo_negated"), (-1, "placebo_affirmed"))'),
    ('mech2 T16 S1 read in the two-candidate format only', 'wf/mech2.py', '((1, "desire_check"), (-1, "desire_keep")), f, reference=ref, **kw) for f in FORMATS}', '((1, "desire_check"), (-1, "desire_keep")), "cand", reference=ref, **kw) for f in FORMATS}'),
    ('mech2 T17 Mistral on every item', 'wf/mech2.py', 'return [(t, a, f) for t, a, f in out if t["id"] in sample or a in P6_ARMS]', 'return out'),
    ('mech2 T18 Mistral without P6 arms outside the sample', 'wf/mech2.py', 'if t["id"] in sample or a in P6_ARMS]', 'if t["id"] in sample]'),
    ('mech2 T19 P6 arms are the position arms only', 'wf/mech2.py', 'P6_ARMS = tuple(arm for _, arm in PRIMARY["P6_position_step0"]["terms"])', 'P6_ARMS = POSITION_ARMS'),
    ('mech2 T20 integrity expects every job of the arms', 'wf/mech2.py', 'for t, a, f in model_jobs(tag, union, sample)}', 'for t, a, f in jobs(union, model_arms(tag))}'),
    ('mech2 T21 analysis checks integrity without the sample', 'wf/mech2.py', 'integrity = score_integrity(scores, sets["union"], tags, sets["sample"])', 'integrity = score_integrity(scores, sets["union"], tags)'),
    ('mech2 T22 runner gives Mistral every job', 'wf/mech2.py', 'work = {tag: model_jobs(tag, shard, sets["sample"]) for tag in tags}', 'work = {tag: jobs(shard, model_arms(tag)) for tag in tags}'),
    ('mech2 T23 undefined L on a primary arm does not refuse', 'wf/mech2.py', '"n_unexpected",\n                                   "n_undefined_L_primary_arms"))', '"n_unexpected"))'),
    ('mech2 T24 undefined L on any arm counted as primary', 'wf/mech2.py', '_undefined(x) and split_condition(x.get("condition", ""))[0] in PRIMARY_ARMS', '_undefined(x)'),
    ('mech2 T25 unknown items counted as unexpected too', 'wf/mech2.py', '"n_unexpected": len({k for k in keys if k[0] in items} - expected),', '"n_unexpected": len(set(keys) - expected),'),
    ('mech2 T26 status line hides undefined scores', 'wf/mech2.py', '+ (f"; undefined L on {undefined} scores of secondary arms" if undefined else ""))', '+ "")'),
    ('mech2 T27 every move completes the task', 'wf/mech2.py', '    return re.sub(r"\\s+\\d+$", "", p[2]) == parse_task_line(t["task_line"]).receptacle\n\n\ndef swapped_target_at_rec', '    return True\n\n\ndef swapped_target_at_rec'),
    ('mech2 T28 group difference reversed', 'wf/mech2.py', 'return _mean(a) - _mean(b) if a and b else float("nan")', 'return _mean(b) - _mean(a) if a and b else float("nan")'),
    ('mech2 T29 group r over group a reference only', 'wf/mech2.py', '        den = _mean([r for _, _, r in xs])\n        return diff(xs) / den', '        den = _mean([r for g, _, r in xs if g])\n        return diff(xs) / den'),
    ('mech2 T30 H7a groups swapped', 'wf/mech2.py', '"H7a_desire_check_shown_vs_not": {f: group_difference(by, shown, not_shown,', '"H7a_desire_check_shown_vs_not": {f: group_difference(by, not_shown, shown,'),
    ('mech2 T31 H7c against every item', 'wf/mech2.py', 'other = [t for t in pops["sample"] if not completes_task(t)]', 'other = pops["sample"]'),
    ('mech2 T32 known minus unknown reversed', 'wf/mech2.py', 'return _mean(kn) - _mean(un) if kn and un else float("nan")', 'return _mean(un) - _mean(kn) if kn and un else float("nan")'),
    ('mech2 T33 knowledge split without desire_check', 'wf/mech2.py', ',\n             "desire_check_vs_swapped": ((1, "desire_check"), (-1, "swapped"))}', '}'),
    ('mech2 T34 secondary joint reading from one format', 'wf/mech2.py', 'per[h]["joint"] = _joint(per[h]["cand"], per[h]["yn"])', 'per[h]["joint"] = per[h]["cand"]'),
    ('mech2 T35 secondary zones ignore estimable', 'wf/mech2.py', '    if not r.get("estimable") or lo != lo or hi != hi:\n        return None\n    return zones_of(lo, hi)', '    if lo != lo or hi != hi:\n        return None\n    return zones_of(lo, hi)'),
    ('mech2 T36 placebo predicted below zero under H2', 'wf/mech2.py', '"S2_negation_placebo": {"H2_evidence": _Z,', '"S2_negation_placebo": {"H2_evidence": _N,'),
    ('mech2 T37 summary not given M', 'wf/mech2.py', 'block["primary_yes_no"]["readout"], mention,', 'block["primary_yes_no"]["readout"], None,'),
    ('mech2 T38 M on every sample item', 'wf/mech2.py', 'mention = {f: contrast(by, take, MENTION_TERMS', 'mention = {f: contrast(by, pops["sample"], MENTION_TERMS'),
    # Mechanism battery, review 4 (2026-11-21, before any score): the review's mutants R1-R20, adapted where the
    # amended code moved their text (R6, R20: the interval tests now go through `_r_interval_in`; R7, R15: S2 and S3
    # are computed before the joint readout; R8: M reads `take`; R17: the flag is `mention_remains`). R10 (the risky
    # rule reads PREDICTIONS only) and R19 (a comment added) are equivalent and left out: `_risky` is only called for
    # accounts, whose cells are PREDICTIONS' own.
    ('mech2 R1 salience contradictions read from the two-candidate format', 'wf/mech2.py', 'h3_against = [c for c, v in out["H3_salience"]["per_contrast"].items() if v == "contradicted"]', 'h3_against = [c for c, v in cand["H3_salience"]["per_contrast"].items() if v == "contradicted"]'),
    ('mech2 R2 claims sentence from the two-candidate H2 verdict', 'wf/mech2.py', 'claims_read = claims_only and out["H2_evidence"]["verdict"] == "consistent"', 'claims_read = claims_only and cand["H2_evidence"]["verdict"] == "consistent"'),
    ('mech2 R3 claims-only true when salience is not contradicted', 'wf/mech2.py', 'claims_only = bool(h3_against) and set(h3_against) <= set(CLAIM_CONTRASTS)', 'claims_only = set(h3_against) <= set(CLAIM_CONTRASTS)'),
    ('mech2 R4 claim contrasts include P3', 'wf/mech2.py', 'CLAIM_CONTRASTS = ("P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed")', 'CLAIM_CONTRASTS = ("P3_mention_without_need", "P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed")'),
    ('mech2 R5 claim contrasts drop P7', 'wf/mech2.py', 'CLAIM_CONTRASTS = ("P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed")', 'CLAIM_CONTRASTS = ("P5_done_vs_cancelled", "P8_negated_vs_affirmed")'),
    ('mech2 R6 M above the band from its edge', 'wf/mech2.py', 'lambda lo, hi: lo > TRIVIAL_BAND)', 'lambda lo, hi: lo >= TRIVIAL_BAND)'),
    ('mech2 R7 placebo on every sample item', 'wf/mech2.py', 'placebo = {f: contrast(by, take, PLACEBO_TERMS', 'placebo = {f: contrast(by, pops["sample"], PLACEBO_TERMS'),
    ('mech2 R8 M over the affirmed lift', 'wf/mech2.py', 'mention = {f: contrast(by, take, MENTION_TERMS, f, reference=ref, **kw) for f in FORMATS}', 'mention = {f: contrast(by, take, MENTION_TERMS, f, reference=((1, "affirmed"), (-1, "swapped")), **kw) for f in FORMATS}'),
    ('mech2 R9 M zones from the two-candidate format only', 'wf/mech2.py', '"zones": secondary_zones(mention[f])}', '"zones": secondary_zones(mention["cand"])}'),
    ('mech2 R11 discriminating only against content rivals', 'wf/mech2.py', 'and any(rows[o].get(c) == "contradicted" for o in rows if o != h)]', 'and any(rows[o].get(c) == "contradicted" for o in rows if o != h and o in PREDICTIONS)]'),
    ('mech2 R12 Mistral outside the sample on the start arms only', 'wf/mech2.py', 'return [(t, a, f) for t, a, f in out if t["id"] in sample or a in P6_ARMS]', 'return [(t, a, f) for t, a, f in out if t["id"] in sample or (a in P6_ARMS and t.get("history_len", 0) == 0 and a.endswith("_start"))]'),
    ('mech2 R13 placebo on takes of another object', 'wf/mech2.py', '        if not p or p[0] != "take" or p[1] != obj:\n            return None\n        sw = t["swapped_task_line"]', '        if not p or p[0] != "take":\n            return None\n        sw = t["swapped_task_line"]'),
    ('mech2 R14 undefined L on yes/no primary arms not counted', 'wf/mech2.py', '_undefined(x) and split_condition(x.get("condition", ""))[0] in PRIMARY_ARMS', '_undefined(x) and x.get("condition", "") in PRIMARY_ARMS'),
    ('mech2 R15 P8 net read in the two-candidate format only', 'wf/mech2.py', 'p8_net = {f: contrast(by, take, P8_NET_TERMS, f, reference=ref', 'p8_net = {f: contrast(by, take, P8_NET_TERMS, "cand", reference=ref'),
    ('mech2 R16 H7a within groups reversed', 'wf/mech2.py', 'group_difference(by, [t for t in within if t["id"] in shown_ids], [t for t in within if t["id"] not in shown_ids],', 'group_difference(by, [t for t in within if t["id"] not in shown_ids], [t for t in within if t["id"] in shown_ids],'),
    ('mech2 R17 mention sentence needs H2 consistent too', 'wf/mech2.py', 'mention_remains = claims_only and _mention_above_band(mention)', 'mention_remains = claims_only and _mention_above_band(mention) and out["H2_evidence"]["verdict"] == "consistent"'),
    ('mech2 R18 hash check skipped for the secondary model', 'wf/mech2.py', '"n_hash_mismatch": sum(x["transition_id"] in items and x.get("item_hash") != hashes[x["transition_id"]] for x in mine),', '"n_hash_mismatch": sum(tag in PRIMARY_MODELS and x["transition_id"] in items and x.get("item_hash") != hashes[x["transition_id"]] for x in mine),'),
    ('mech2 R20 estimable ignored in the interval tests', 'wf/mech2.py', 'if not r.get("estimable") or lo != lo or hi != hi or not test(lo, hi):', 'if lo != lo or hi != hi or not test(lo, hi):'),
    # Mutants of the amendment after review 4: M's halves, the absent-M sentence, the placebo sentence, S2 without
    # the swapped-target items, the S1 line.
    ('mech2 U1 affirmed half reads the denial', 'wf/mech2.py', '"M_half_affirmed_minus_swapped": ((1, "affirmed"), (-1, "swapped")),', '"M_half_affirmed_minus_swapped": ((1, "negated"), (-1, "swapped")),'),
    ('mech2 U2 denied half reversed', 'wf/mech2.py', '"M_half_swapped_minus_negated": ((1, "swapped"), (-1, "negated"))}', '"M_half_swapped_minus_negated": ((1, "negated"), (-1, "swapped"))}'),
    ('mech2 U3 absent M checks the upper edge only', 'wf/mech2.py', 'lambda lo, hi: -TRIVIAL_BAND <= lo and hi <= TRIVIAL_BAND)', 'lambda lo, hi: hi <= TRIVIAL_BAND)'),
    ('mech2 U4 absent M without the claims-only condition', 'wf/mech2.py', 'mention_absent = claims_only and _mention_inside_band(mention)', 'mention_absent = _mention_inside_band(mention)'),
    ('mech2 U5 placebo read below zero, not below the band', 'wf/mech2.py', '    return _r_interval_in(placebo, lambda lo, hi: hi < -TRIVIAL_BAND)', '    return _r_interval_in(placebo, lambda lo, hi: hi < 0)'),
    ('mech2 U6 verdicts on P8 omit consistent accounts', 'wf/mech2.py', 'or (out[h]["verdict"] == "consistent" and contrast_name in out[h]["consistent_where_a_rival_fails"])]', 'or False]'),
    ('mech2 U7 placebo sentence names the verdicts on P5', 'wf/mech2.py', 'on_p8 = _resting_on(out, "P8_negated_vs_affirmed")', 'on_p8 = _resting_on(out, "P5_done_vs_cancelled")'),
    ('mech2 U8 summary not given the placebo', 'wf/mech2.py', 'mention,\n                                                  placebo, p8_net)', 'mention)'),
    ('mech2 U9 placebo subset keeps only the left-out items', 'wf/mech2.py', 'kept = [t for t in take if not swapped_target_at_rec(t)]', 'kept = [t for t in take if swapped_target_at_rec(t)]'),
    ('mech2 U10 swapped target read from the real task', 'wf/mech2.py', 'return re.sub(r"\\s+\\d+$", "", p[2]) == parse_task_line(t["swapped_task_line"]).receptacle', 'return re.sub(r"\\s+\\d+$", "", p[2]) == parse_task_line(t["task_line"]).receptacle'),
    ('mech2 U11 S1 line whatever the verdicts', 'wf/mech2.py', '    if not all(ro[h]["verdict"] in ("consistent", "untested") for h in ("H4_wishful", "H5_policy_blend")):\n        return ""', '    if False:\n        return ""'),
    ('mech2 U12 status line drops S1', 'wf/mech2.py', '+ (f"; {s1}" if (s1 := s1_line(report[tag])) else "")', '+ ""'),
    # The adversarial check of M (2026-11-21): the summary must never attribute the residual, or its absence, to salience.
    ('mech2 W1 residual sentence attributes the effect to salience', 'wf/mech2.py', '"that salience contributes. Word overlap with the action would give the same.")', '"that salience is likely part of the effect.")'),
    ('mech2 W2 absent residual read as no salience', 'wf/mech2.py', '"the same, so it does not show that salience is absent.")', '"the same, so salience is absent.")'),
    # Mechanism battery, review 5 (2026-11-22, before any score): the review's surviving mutants X1, X2, X8, X9, and
    # mutants of the M-below-band sentence (Y1, Y2).
    ('mech2 X1 attribution appended after the residual sentence', 'wf/mech2.py', '"that salience contributes. Word overlap with the action would give the same.")', '"that salience contributes. Word overlap with the action would give the same. Salience therefore explains the residual.")'),
    ('mech2 X2 claims sentence says M indicates salience', 'wf/mech2.py', '"than salience; see M.")', '"than salience; see M. M indicates salience.")'),
    ('mech2 X8 S1 line only when both accounts are consistent', 'wf/mech2.py', 'in ("consistent", "untested") for h in ("H4_wishful", "H5_policy_blend")', 'in ("consistent",) for h in ("H4_wishful", "H5_policy_blend")'),
    ('mech2 X9 P8 net quoted without an interval in both formats', 'wf/mech2.py', 'if p8_net and all(\n            (p8_net.get(f) or {}).get("ratio", {}).get("ci") for f in FORMATS) else ""', 'if p8_net else ""'),
    ('mech2 Y1 below-band sentence without the claims-only condition', 'wf/mech2.py', 'mention_below = claims_only and _mention_below_band(mention)', 'mention_below = _mention_below_band(mention)'),
    ('mech2 Y2 below-band read below zero, not below the band', 'wf/mech2.py', '    return _r_interval_in(mention, lambda lo, hi: hi < -TRIVIAL_BAND)', '    return _r_interval_in(mention, lambda lo, hi: hi < 0)'),
    # Model families, wave A (2026-11-20, wf/families.py, the scorer settings and the prefetch tool).
    ("families fold drops the system text", "wf/families.py",
     'first = {"role": "user", "content": messages[0]["content"] + FOLD_SEPARATOR + messages[1]["content"]}',
     'first = {"role": "user", "content": messages[1]["content"]}'),
    ("families fold only in the aligned arm", "wf/families.py", 'return fold_system(msgs) if spec.system == "fold" else msgs',
     'return fold_system(msgs) if spec.system == "fold" and arm == "aligned" else msgs'),
    ("families Gemma on default attention", "wf/families.py", '"Gemma 2", "2024-06-27", 9.24, "fold", "eager", {},',
     '"Gemma 2", "2024-06-27", 9.24, "fold", None, {},'),
    ("families Llama 3.2 dated today", "wf/families.py", '{"date_string": "26 Jul 2024"},\n        "Llama 3 header format',
     '{},\n        "Llama 3 header format'),
    ("families gate never stops a job", "wf/families.py", '        if not rec["ok"]:\n            raise TokenizationMismatch',
     '        if False:\n            raise TokenizationMismatch'),
    ("families candidates not checked against the template", "wf/families.py", 'if not r["alone_equals_template_reply"]:', "if False:"),
    ("families multi-token Yes accepted", "wf/families.py", "        if len(scored[word]) != 1:", "        if False:"),
    ("families shared first token accepted", "wf/families.py",
     'if not ids["o_plus"] or not ids["o_minus"] or ids["o_plus"][0] == ids["o_minus"][0]:',
     'if not ids["o_plus"] or not ids["o_minus"]:'),
    ("families standardised by the aligned spread", "wf/families.py", "sd = statistics.stdev(lb for _, lb in rows)\n        return",
     "sd = statistics.stdev(la for la, _ in rows)\n        return"),
    ("families reference keeps repeated records", "wf/families.py", "if key in out:\n                    repeats += 1",
     "if False:\n                    repeats += 1"),
    ("families scorer ignores the attention setting", "wf/scoring/logodds.py",
     'extra = {"attn_implementation": self.attn_implementation} if self.attn_implementation else {}', "extra = {}"),
    ("families scorer drops the template settings", "wf/scoring/logodds.py",
     'return {"enable_thinking": False, **self.template_kwargs}', 'return {"enable_thinking": False}'),
    ("families prefetch stops at the first refusal", "wf/tools/prefetch_models.py",
     "    except Exception as e:  # noqa: BLE001 - every failure is reported, then the next model is tried",
     "    except ImportError as e:"),
    ("families prefetch fetches Meta-format weights", "wf/tools/prefetch_models.py", 'IGNORE = ["original/*", ', "IGNORE = ["),
    ("families gate skipped before scoring", "wf/sizes.py", "        if gate is not None:\n            try:", "        if False:\n            try:"),
    # Families fix round 1 (2026-11-20): the review's ten surviving mutants, adapted where the gate now goes through
    # the scorer's encoding, and the new code (one population for every model, the placement check).
    ("families gate skips the successes", "wf/families.py",
     '    for t in primary + successes:\n        msgs = family_messages(spec, t, "aligned")',
     '    for t in primary:\n        msgs = family_messages(spec, t, "aligned")'),
    ("families gate checks the first item's candidates only", "wf/families.py",
     '    for t in primary + successes:\n        msgs = family_messages(spec, t, "aligned")',
     '    for t in (primary + successes)[:1]:\n        msgs = family_messages(spec, t, "aligned")'),
    ("families gate renders the reply without the template settings", "wf/families.py",
     'full = _ids(scorer._encode(convo + [{"role": "assistant", "content": reply}], add_generation_prompt=False))',
     'full = _ids(scorer._tok.apply_chat_template(convo + [{"role": "assistant", "content": reply}], tokenize=True, '
     'add_generation_prompt=False))'),
    ("families gate renders the prompt without the template settings", "wf/families.py",
     "prompt = _ids(scorer._encode(convo))",
     "prompt = _ids(scorer._tok.apply_chat_template(convo, tokenize=True, add_generation_prompt=True))"),
    ("families runner gives the gate the first primary item only", "wf/sizes.py", "gate(d, scorer, primary, successes)",
     "gate(d, scorer, primary[:1], [])"),
    ("families runner skips the gate on resume", "wf/sizes.py", "        if gate is not None:\n            try:",
     '        if gate is not None and not cfg.get("resume"):\n            try:'),
    ("families tokenizer loaded without the pinned revision", "wf/scoring/logodds.py",
     "AutoTokenizer.from_pretrained(self.model_name, revision=self.revision, trust_remote_code=True)",
     "AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)"),
    ("families flip names swapped", "wf/families.py",
     '"flips": {"flip_full_L_toward_success": block["flips_to_success"],\n                     "flip_first_token_sign_change": block["decision_flip"]',
     '"flips": {"flip_full_L_toward_success": block["decision_flip"],\n                     "flip_first_token_sign_change": block["flips_to_success"]'),
    ("families standardised yes/no reads the two-candidate arms", "wf/families.py",
     '"yes_no": standardised_effect(s, primary, "yn_aligned", "yn_swapped")', '"yes_no": standardised_effect(s, primary, "aligned", "swapped")'),
    ("families runs not filtered by model tag", "wf/families.py", 'read_jsonl(p / "scoring.jsonl") if x["model"] == tag],',
     'read_jsonl(p / "scoring.jsonl")],'),
    ("families Yes/No compared with the template against itself", "wf/families.py", "            if reply != scored[word]:",
     "            if reply != reply:"),
    ("families Yes/No checked on the first item only", "wf/families.py", "    for t, arm in sizes.size_jobs(primary, successes):",
     "    for t, arm in sizes.size_jobs(primary, successes)[:6]:"),
    ("families Yes/No checked after the aligned prompt only", "wf/families.py", "        if arm not in sizes.YN_ARMS:\n            continue",
     '        if arm != "yn_aligned":\n            continue'),
    ("families reference read beyond the family population", "wf/families.py",
     's, info["records_outside_population"] = same_population(s, primary, successes)', 'info["records_outside_population"] = 0'),
    ("families placement keeps the native placement", "wf/families.py", 'return replace(spec, tag=spec.tag + FIRST_TURN, system="fold",',
     'return replace(spec, tag=spec.tag + FIRST_TURN, system="native",'),
    ("families placement compares different items", "wf/families.py",
     'ids = {x["transition_id"] for x in first_turn} & {x["transition_id"] for x in native}', 'ids = {x["transition_id"] for x in native}'),
    ("families placement difference reversed", "wf/families.py", "rows = [alt[k] - nat[k] for k in both]", "rows = [nat[k] - alt[k] for k in both]"),
    ("families shard runner ignores the arm selection", "wf/sizes.py",
     "jobs = [(t, a) for t, a in size_jobs(primary, successes) if arms is None or a in arms]", "jobs = size_jobs(primary, successes)"),
    # Families amendment (2026-11-20, pre-data): primary metrics, competence gate, equivalence, primary family.
    ("families R evidence effect reversed", "wf/families.py",
     "evidence = sum(sc) / len(sc) - sum(r[2] for r in f) / len(f)", "evidence = sum(r[2] for r in f) / len(f) - sum(sc) / len(sc)"),
    ("families R kept with no evidence effect", "wf/families.py", 'if evidence > 0 else float("nan")', 'if evidence else float("nan")'),
    ("families item share counts ties as toward Yes", "wf/families.py", "return sum(x > 0 for x in d) / len(d)",
     "return sum(x >= 0 for x in d) / len(d)"),
    ("families Yes share as mean P(Yes)", "wf/families.py", 'v = [by[(t["id"], arm)] > 0 for t in items if (t["id"], arm) in by]',
     'v = [1 / (1 + math.exp(-by[(t["id"], arm)])) for t in items if (t["id"], arm) in by]'),
    ("families flips away from Yes counted as toward", "wf/families.py", "for sw, al in v if sw and not al)", "for sw, al in v if al and not sw)"),
    ("families competence AUC at the bound fails", "wf/families.py", '"pass": auc >= COMPETENCE["auc_neutral_min"]',
     '"pass": auc > COMPETENCE["auc_neutral_min"]'),
    ("families competence ignores Yes to everything", "wf/families.py",
     '"pass": yes_f <= COMPETENCE["yes_share_neutral_failures_max"]}', '"pass": True}'),
    ("families competence ignores No to everything", "wf/families.py",
     '"pass": yes_s >= COMPETENCE["yes_share_neutral_successes_min"]}', '"pass": True}'),
    ("families competence keeps the old lower bound on failures", "wf/families.py",
     '"pass": yes_f <= COMPETENCE["yes_share_neutral_failures_max"]}', '"pass": 0.05 <= yes_f <= COMPETENCE["yes_share_neutral_failures_max"]}'),
    ("families competence Yes share on successes read under the aligned task", "wf/families.py",
     'yes_s = block["argmax_answers"]["yes_share_successes"]["yn_absent"]', 'yes_s = block["argmax_answers"]["yes_share_successes"]["yn_aligned"]'),
    ("families competence coverage ignored", "wf/families.py",
     '"pass": mass >= COMPETENCE["median_yes_no_mass_neutral_min"]}', '"pass": True}'),
    ("families equivalence margin a factor of 2", "wf/families.py", "bound = math.log(EQUIVALENCE_FACTOR)", "bound = math.log(2.0)"),
    ("families equivalence from the point estimate", "wf/families.py", "matches = bool(-bound < lo90 and hi90 < bound)",
     "matches = bool(abs(est) < bound)"),
    ("families noise floor lowered", "wf/families.py", "< NOISE_FLOOR_NATS", "< 0.1"),
    ("families primary family without Holm", "wf/families.py", "    adj = holm(p)\n", "    adj = dict(p)\n"),
    ("families primary family ignores the competence gate", "wf/families.py",
     '            elif not ok:\n                verdict = "not interpretable"', '            elif False:\n                verdict = "not interpretable"'),
    ("families primary family ignores the direction", "wf/families.py", 'side = rec["estimate"] > rec["null"]', "side = True"),
    # Families fix round 2 (2026-11-20): review 2's mutants M1-M14, then the third amendment and the R fallback.
    ("families M1 R successes read under the aligned task", "wf/families.py",
     'if (t["id"], "yn_absent") in by:\n            rows.append(("s", float("nan"), by[(t["id"], "yn_absent")]))',
     'if (t["id"], "yn_aligned") in by:\n            rows.append(("s", float("nan"), by[(t["id"], "yn_aligned")]))'),
    ("families M2 R failures' evidence baseline read under the swapped task", "wf/families.py",
     'rows.append(("f", by[k[0]] - by[k[1]], by[k[2]]))', 'rows.append(("f", by[k[0]] - by[k[1]], by[k[1]]))'),
    ("families M3 share p not game-clustered", "wf/families.py",
     "p_share = pair_permutation([(r[1] > 0) - (r[1] < 0) for r, _ in f], seed=C.RNG_SEED, cluster_ids=fg)",
     "p_share = pair_permutation([(r[1] > 0) - (r[1] < 0) for r, _ in f], seed=C.RNG_SEED)"),
    ("families M4 R p computed on signs, not effects", "wf/families.py", "p_ratio = pair_permutation([r[1] for r, _ in f]",
     "p_ratio = pair_permutation([(r[1] > 0) - (r[1] < 0) for r, _ in f]"),
    ("families M5 R comparison: model B not resampled", "wf/families.py",
     "pair = lambda v: (_ratio([x for x, _ in v]), _ratio([y for _, y in v]))",
     "pair = lambda v: (_ratio([x for x, _ in v]), _ratio([y for _, y in rows]))"),
    ("families M6 R 'differs' from the 90% interval", "wf/families.py", "differs = bool(lo95 > 0 or hi95 < 0)",
     "differs = bool(lo90 > 0 or hi90 < 0)"),
    ("families M6 share 'differs' from the 90% interval", "wf/families.py",
     '"differs": bool(est == est and (lo95 > 0 or hi95 < 0))', '"differs": bool(est == est and (lo90 > 0 or hi90 < 0))'),
    ("families M7 R TOST from the 95% interval", "wf/families.py", "matches = bool(-bound < lo90 and hi90 < bound)",
     "matches = bool(-bound < lo95 and hi95 < bound)"),
    ("families M7 share TOST from the 95% interval", "wf/families.py", "-SHARE_MARGIN < lo90 and hi90 < SHARE_MARGIN",
     "-SHARE_MARGIN < lo95 and hi95 < SHARE_MARGIN"),
    ("families M8 primary family excludes non-interpretable models", "wf/families.py", "    for tag in expected:\n        key =",
     '    for tag in [t for t in expected if t not in blocks or blocks[t]["competence"]["interpretable"]]:\n        key ='),
    ("families M9 argmax Yes at L >= 0", "wf/families.py", 'v = [by[(t["id"], arm)] > 0 for t in items',
     'v = [by[(t["id"], arm)] >= 0 for t in items'),
    ("families M10 goal side at estimate >= null", "wf/families.py", 'side = rec["estimate"] > rec["null"]',
     'side = rec["estimate"] >= rec["null"]'),
    ("families M11 NaN scores not dropped in yes/no rows", "wf/families.py",
     'by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}\n    rows, games, ids',
     'by = {(x["transition_id"], x["condition"]): x["L"] for x in s}\n    rows, games, ids'),
    ("families M12 noise floor without abs", "wf/families.py",
     'out["delta_L_difference_nats"]["within_noise_floor"] = abs(out["delta_L_difference_nats"]["estimate"]) < NOISE_FLOOR_NATS',
     'out["delta_L_difference_nats"]["within_noise_floor"] = out["delta_L_difference_nats"]["estimate"] < NOISE_FLOOR_NATS'),
    ("families M13 R log ratio inverted (B over A)", "wf/families.py", "est = math.log(ra / rb)", "est = math.log(rb / ra)"),
    ("families M14 flips: no-to-yes counts aligned Yes only", "wf/families.py",
     "to_yes = lambda v: sum(1 for sw, al in v if al and not sw)", "to_yes = lambda v: sum(1 for sw, al in v if al)"),
    ("families R ratio kept whatever the dropped draws", "wf/families.py",
     "estimable = bool(ra > 0 and rb > 0 and dropped <= DROPPED_DRAWS_MAX * B)", "estimable = bool(ra > 0 and rb > 0)"),
    ("families R drop threshold 20%", "wf/families.py", "DROPPED_DRAWS_MAX = 0.02", "DROPPED_DRAWS_MAX = 0.2"),
    ("families R fallback never differs", "wf/families.py", "differs = bool(computable and (lo95 > 0 or hi95 < 0))",
     "differs = False"),
    ("families Holm family shrinks with missing p", "wf/families.py", "p[key] = pv if pv == pv else 1.0", "p[key] = pv"),
    ("families share margin 0.1", "wf/families.py", "SHARE_MARGIN = 0.05", "SHARE_MARGIN = 0.1"),
    ("families share comparison p not game-clustered", "wf/families.py",
     '"p_differs": pair_permutation(d, seed=C.RNG_SEED, cluster_ids=fg)', '"p_differs": pair_permutation(d, seed=C.RNG_SEED)'),
    ("families share TOST p from the smaller side", "wf/families.py", '"p_equivalent": max(p_low, p_high)',
     '"p_equivalent": min(p_low, p_high)'),
    ("families comparisons among wave A models skipped", "wf/families.py",
     "pairs += [(a, b) for a, b in planned if (a, b) not in pairs and a in both and b in both]", "pairs += []"),
    ("families comparison readings ignore the equivalence test", "wf/families.py",
     'readings[f"{a} vs {b}"] = "matches" if ev == "supported" else', 'readings[f"{a} vs {b}"] = "inconclusive" if ev == "supported" else'),
    # Option A decision simulation (2026-11-20, wf/decisionsim.py, notes/decisionsim_design.md); adapted to fix round 1.
    ("decisionsim tie goes to the last action", "wf/decisionsim.py",
     'return min(cands, key=lambda t: (-L[t["id"]], tie_key(t)))', 'return max(cands, key=lambda t: (L[t["id"]], tie_key(t)))'),
    ("decisionsim verification accepts a score of zero", "wf/decisionsim.py",
     'return L[t["id"]] > 0', 'return L[t["id"]] >= 0'),
    ("decisionsim verification falls back to the rejected proposal", "wf/decisionsim.py",
     'return [p] if accepts(L, p) else _generated(state)', 'return [p] if accepts(L, p) else state["candidates"]'),
    ("decisionsim generated-only set keeps the proposal", "wf/decisionsim.py",
     'if rule == "rule1_generated_only":\n        return _generated(state)',
     'if rule == "rule1_generated_only":\n        return state["candidates"]'),
    ("decisionsim rule 3 read where it must fail", "wf/decisionsim.py",
     'return any(succeeded(t) for t in _goal_relevant(state))', 'return True'),
    ("decisionsim states need no determined failure", "wf/decisionsim.py",
     'if not any(not succeeded(t) and t.get("context_determined") for t in ts):', 'if not any(not succeeded(t) for t in ts):'),
    ("decisionsim swapped line from each candidate's own draw", "wf/decisionsim.py",
     '"swapped_task_line": make_swapped_task_line(t["task_line"], t["goal_object"], state["swap"])}',
     '"swapped_task_line": t["swapped_task_line"]}'),
    ("decisionsim swap object may be a candidate's object", "wf/decisionsim.py",
     'options = [o for o in drawn if o not in named]', 'options = drawn'),
    ("decisionsim bootstrap resamples states", "wf/decisionsim.py",
     'b = cluster_bootstrap(_mean, values, games, seed=C.RNG_SEED)', 'b = cluster_bootstrap(_mean, values, list(range(len(values))), seed=C.RNG_SEED)'),
    ("decisionsim permutation flips states", "wf/decisionsim.py",
     'pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=games)', 'pair_permutation(diffs, seed=C.RNG_SEED)'),
    ("decisionsim sample depends on input order", "wf/decisionsim.py",
     'keys = sorted((s["game_id"], s["state_idx"]) for s in states)', 'keys = [(s["game_id"], s["state_idx"]) for s in states]'),
    ("decisionsim resume trusts saved scores", "wf/decisionsim.py",
     'if any(hashes.get((x["transition_id"], x["condition"])) != x.get("score_hash") for x in saved):', 'if False:'),
    ("decisionsim analysis reads another sample's scores", "wf/decisionsim.py",
     'if saved.get("sample_digest") != digest:', 'if False:'),
    ("decisionsim Option C ignores the direction", "wf/decisionsim.py",
     'toward_failure = lo > 0 and p < OPTION_C_P', 'toward_failure = p < OPTION_C_P'),
    ("decisionsim incomplete states kept", "wf/decisionsim.py",
     'keep = [s for s in states if all(t["id"] in L.get(a, {})', 'keep = [s for s in states if True or all(t["id"] in L.get(a, {})'),
    ("decisionsim baseline Yes read from raw scores", "wf/decisionsim.py",
     'yes = lambda ts: _mean([float(accepts(L, t)) for t in ts])', 'yes = lambda ts: _mean([float(accepts(raw, t)) for t in ts])'),
    ("decisionsim family gate skipped", "wf/decisionsim.py",
     'if tag in FAMILY_MODELS and not scorer.mock:', 'if False:'),
    ("decisionsim scores saved at the end", "wf/decisionsim.py",
     '            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(),', '            if (s, t, arm) == todo[-1]: append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(),'),
    # Review 1 of the decision simulation (2026-11-20): its 17 mutants, adapted; eleven had survived the first tests.
    ("decisionsim review1 M1 a succeeding proposal at zero is not a false rejection", "wf/decisionsim.py",
     '"false_rejection": rates(even, succeeded, lambda L, p: not accepts(L, p))',
     '"false_rejection": rates(even, succeeded, lambda L, p: L[p["id"]] < 0)'),
    ("decisionsim review1 M2 swap draw ignores the state index", "wf/decisionsim.py",
     'random.Random(f"{seed}|decisionsim|{game_id}|{state_idx}")', 'random.Random(f"{seed}|decisionsim|{game_id}")'),
    ("decisionsim review1 M3 complete_states needs swapped scores where no swap exists", "wf/decisionsim.py",
     'keep = [s for s in states if all(t["id"] in L.get(a, {}) for a in L for t in s["candidates"]\n                                     if a != SWAPPED or s["swap"])]',
     'keep = [s for s in states if all(t["id"] in L.get(a, {}) for a in L for t in s["candidates"])]'),
    ("decisionsim review1 M4 every arm scored with the aligned prompt", "wf/decisionsim.py",
     'messages=build(prompt_item(t, s, arm), arm),', 'messages=build(prompt_item(t, s, ALIGNED), ALIGNED),'),
    ("decisionsim review1 M5 neutral scored with the aligned prompt", "wf/decisionsim.py",
     'messages=build(prompt_item(t, s, arm), arm),', 'messages=build(prompt_item(t, s, arm), ALIGNED if arm == NEUTRAL else arm),'),
    ("decisionsim review1 M6 random-choice rate over all candidates for every rule", "wf/decisionsim.py",
     '"random_fail": _mean([float(not succeeded(t)) for t in cands]),', '"random_fail": _mean([float(not succeeded(t)) for t in s["candidates"]]),'),
    ("decisionsim review1 M7 shard ignored", "wf/decisionsim.py",
     'mine = [s for s in states if s["game_id"] in games]', 'mine = list(states)'),
    ("decisionsim review1 M8 duplicate-candidate check removed", "wf/decisionsim.py",
     'if len({t["action"] for t in ts}) != len(ts):', 'if False:'),
    ("decisionsim review1 M9 Option C reads the interval's upper bound", "wf/decisionsim.py",
     'toward_failure = lo > 0 and p < OPTION_C_P', 'toward_failure = hi > 0 and p < OPTION_C_P'),
    ("decisionsim review1 M10 tie count reports nothing", "wf/decisionsim.py",
     'return len(top) > 1, len({succeeded(t) for t in top}) > 1', 'return False, False'),
    ("decisionsim review1 M11 determined share over all chosen actions", "wf/decisionsim.py",
     '"b": sum(r["det_b"] for r in rows) / max(1, sum(fb))},', '"b": sum(r["det_b"] for r in rows) / len(rows)},'),
    ("decisionsim review1 M12 rejected proposal chosen when it scores highest", "wf/decisionsim.py",
     'return [p] if accepts(L, p) else _generated(state)',
     'return [p] if accepts(L, p) or L[p["id"]] >= max(L[t["id"]] for t in state["candidates"]) else _generated(state)'),
    ("decisionsim review1 M13 analysis reads a model without both arms", "wf/decisionsim.py",
     'if ALIGNED not in Lq or NEUTRAL not in Lq:', 'if False:'),
    ("decisionsim review1 M14 every baseline only on states with a swap", "wf/decisionsim.py",
     '"baseline": {a: baseline(even_swap if a == SWAPPED else even, L[a], raw[a])',
     '"baseline": {a: baseline(even_swap, L[a], raw[a])'),
    ("decisionsim review1 M15 gate checks only the prompts still to score", "wf/decisionsim.py",
     'gate = yes_no_gate(scorer, [build(prompt_item(t, s, a), a) for s, t, a in jobs])',
     'gate = yes_no_gate(scorer, [build(prompt_item(t, s, a), a) for s, t, a in todo])'),
    ("decisionsim review1 M16 eligibility drops the success requirement", "wf/decisionsim.py",
     '        if not any(succeeded(t) for t in ts):\n            continue\n', ''),
    ("decisionsim review1 M17 changes toward success never reported", "wf/decisionsim.py",
     'elif hi < 0:\n        category = "changes toward success"', 'elif False:\n        category = "changes toward success"'),
    # Fix round 1 of the decision simulation (2026-11-20): new code.
    ("decisionsim fix1 odd-step states kept in contrasts", "wf/decisionsim.py",
     '        if not even_step(s):\n            n_odd += 1\n            continue\n', '        if False:\n            n_odd += 1\n            continue\n'),
    ("decisionsim fix1 broad generator every step", "wf/decisionsim.py", 'BROAD_EVERY_K = 2', 'BROAD_EVERY_K = 1'),
    ("decisionsim fix1 pooled false acceptance drops odd steps", "wf/decisionsim.py",
     '"pooled": rates(states, failing, accepts)', '"pooled": rates(even, failing, accepts)'),
    ("decisionsim fix1 false rejection at odd steps", "wf/decisionsim.py",
     '"false_rejection": rates(even,', '"false_rejection": rates(states,'),
    ("decisionsim fix1 baseline on every state", "wf/decisionsim.py",
     '"baseline": {a: baseline(even_swap if a == SWAPPED else even,', '"baseline": {a: baseline([s for s in states if s["swap"]] if a == SWAPPED else states,'),
    ("decisionsim fix1 calibration at odd steps", "wf/decisionsim.py",
     '        if not even_step(s):\n            continue\n        oks =', '        oks ='),
    ("decisionsim fix1 decomposition at odd steps", "wf/decisionsim.py",
     'sts = [s for s in states if even_step(s) and s["swap"]]', 'sts = [s for s in states if s["swap"]]'),
    ("decisionsim fix1 decomposition where no swap exists", "wf/decisionsim.py",
     'sts = [s for s in states if even_step(s) and s["swap"]]', 'sts = [s for s in states if even_step(s)]'),
    ("decisionsim fix1 scores not rounded", "wf/decisionsim.py",
     'q[x["condition"]][x["transition_id"]] = quantize(x["L"])', 'q[x["condition"]][x["transition_id"]] = x["L"]'),
    ("decisionsim fix1 rounded to a 16th", "wf/decisionsim.py", 'GRID = 64', 'GRID = 16'),
    ("decisionsim fix1 decisions on raw scores", "wf/decisionsim.py",
     '**model_report(sts, Lq, Lraw)}', '**model_report(sts, Lraw, Lraw)}'),
    ("decisionsim fix1 tie hash ignores the state", "wf/decisionsim.py",
     'f"{seed}|{t[\'game_id\']}|{t[\'state_idx\']}|{t[\'action\']}"', 'f"{seed}|{t[\'action\']}"'),
    ("decisionsim fix1 tie hash ignores the seed", "wf/decisionsim.py",
     'f"{seed}|{t[\'game_id\']}|{t[\'state_idx\']}|{t[\'action\']}"', 'f"{t[\'game_id\']}|{t[\'state_idx\']}|{t[\'action\']}"'),
    ("decisionsim fix1 harm counts help too", "wf/decisionsim.py",
     '"harm": float(fa and not fb)', '"harm": float(fa != fb)'),
    ("decisionsim fix1 success-to-success changes counted as failure-to-failure", "wf/decisionsim.py",
     '"change_rate_success_to_success": both_ok / n,', '"change_rate_success_to_success": both_fail / n,'),
    ("decisionsim fix1 proposal share inverted", "wf/decisionsim.py",
     '"prop_a": ca["action"] == s["proposal"]', '"prop_a": ca["action"] != s["proposal"]'),
    ("decisionsim fix1 tie split reads the chosen action", "wf/decisionsim.py",
     '"split_a": _mean([float(not succeeded(t)) for t in top_set(ma, La)]),', '"split_a": float(fa),'),
    ("decisionsim fix1 mixed-outcome ties are any ties", "wf/decisionsim.py",
     'len({succeeded(t) for t in top}) > 1', 'len(top) > 1'),
    ("decisionsim fix1 acceptance cells swapped", "wf/decisionsim.py",
     '(True, False): "accepted_only_a", (False, True): "accepted_only_b"', '(True, False): "accepted_only_b", (False, True): "accepted_only_a"'),
    ("decisionsim fix1 acceptance contributions per cell", "wf/decisionsim.py",
     '"contribution": sum(r["fail_a"] - r["fail_b"] for r in rs) / n}', '"contribution": sum(r["fail_a"] - r["fail_b"] for r in rs) / max(1, len(rs))}'),
    ("decisionsim fix1 calibration margin under the aligned task", "wf/decisionsim.py",
     'm = max(Ln[t["id"]] for t in oks) - Ln[bf["id"]]', 'm = max(La[t["id"]] for t in oks) - La[bf["id"]]'),
    ("decisionsim fix1 lure from a failing proposal", "wf/decisionsim.py",
     '        if not succeeded(p):\n            continue\n', ''),
    ("decisionsim fix1 lure from any failure", "wf/decisionsim.py",
     'if g.get("targets_goal_object") and not succeeded(g):', 'if not succeeded(g):'),
    ("decisionsim fix1 shifts in rounded scores", "wf/decisionsim.py",
     '"mean_L_shift": _mean([La_raw[t["id"]] - Lb_raw[t["id"]] for t in ts])', '"mean_L_shift": _mean([La[t["id"]] - Lb[t["id"]] for t in ts])'),
    ("decisionsim fix1 groups by goal instead of source", "wf/decisionsim.py",
     "{'proposal' if t['action'] == state['proposal'] else 'generated'}", "{'proposal' if t.get('targets_goal_object') else 'generated'}"),
    # review 2 replaced the 5% harm threshold by the net condition: these two now mutate that condition
    ("decisionsim fix1 Option C harm threshold 10%", "wf/decisionsim.py", 'OPTION_C_NET_SHARE = 0.5', 'OPTION_C_NET_SHARE = 0.75'),
    ("decisionsim fix1 Option C ignores p", "wf/decisionsim.py",
     'toward_failure = lo > 0 and p < OPTION_C_P', 'toward_failure = lo > 0'),
    ("decisionsim fix1 Option C p at the bound passes", "wf/decisionsim.py",
     'toward_failure = lo > 0 and p < OPTION_C_P', 'toward_failure = lo > 0 and p <= OPTION_C_P'),
    ("decisionsim fix1 Option C harm at the bound fails", "wf/decisionsim.py",
     'net_enough = net - net_required >= -TOL', 'net_enough = net - net_required > TOL'),
    ("decisionsim fix1 attribution ignores the half", "wf/decisionsim.py",
     '\n                    and dec["aligned_minus_swapped"] - OPTION_C_SWAP_SHARE * aligned_minus_base >= -TOL)', ')'),
    ("decisionsim fix1 Option C reads rule 1", "wf/decisionsim.py",
     'OPTION_C_RULE = "rule2_verification"', 'OPTION_C_RULE = "rule1_with_proposal"'),
    ("decisionsim fix1 a secondary model decides Option C", "wf/decisionsim.py",
     '    for tag in primary:\n        if tag not in reports:', '    for tag in sorted(set(primary) | set(reports)):\n        if tag not in reports:'),
    ("decisionsim fix1 verdict takes the last category", "wf/decisionsim.py",
     'verdict = next(c for c in OPTION_C_CATEGORIES if c in reached)', 'verdict = next(c for c in reversed(OPTION_C_CATEGORIES) if c in reached)'),
    ("decisionsim fix1 Option C without the swapped arm", "wf/decisionsim.py",
     'elif DECOMPOSITION_KEY[base] not in reports[tag]:', 'elif False:'),
    ("decisionsim fix1 toward success read from the lower bound", "wf/decisionsim.py",
     'elif hi < 0:', 'elif lo < 0:'),
    ("decisionsim fix1 swapped arm not scored by default", "wf/decisionsim.py",
     'DEFAULT_ARMS = ("yn_aligned", "yn_absent", "yn_swapped", "yn_blind")', 'DEFAULT_ARMS = ("yn_aligned", "yn_absent", "yn_blind")'),
    # Stage 1a (2026-11-20, before any score): the task-blind arm, the Option C rule against it, and the no-verifier
    # and simulator-oracle reference points (notes/decisionsim_design.md, "Stage 1a additions, before any score").
    ("decisionsim stage1a blind arm keeps the task sentence", "wf/decisionsim.py",
     'return blind_messages(base(t, ALIGNED), t) if arm == BLIND else base(t, arm)',
     'return base(t, ALIGNED) if arm == BLIND else base(t, arm)'),
    ("decisionsim stage1a blind arm is the neutral prompt", "wf/decisionsim.py",
     'return blind_messages(base(t, ALIGNED), t) if arm == BLIND else base(t, arm)',
     'return base(t, NEUTRAL) if arm == BLIND else base(t, arm)'),
    ("decisionsim stage1a blind history leaves an empty line", "wf/decisionsim.py",
     'return "\\n".join(lines[:i] + lines[i + 1:])', 'return "\\n".join(lines[:i] + [""] + lines[i + 1:])'),
    ("decisionsim stage1a blind history drops the room description", "wf/decisionsim.py",
     'return "\\n".join(lines[:i] + lines[i + 1:])', 'return "\\n".join(lines[i + 1:])'),
    ("decisionsim stage1a blind arm strips the examples' task lines", "wf/decisionsim.py",
     'return aligned[:-1] + [{**last, "content": last["content"].replace(old, new)}]',
     'return [{**m, "content": "\\n".join(x for x in m["content"].split("\\n") if not x.startswith(TASK_PREFIX))} '
     'for m in aligned[:-1]] + [{**last, "content": last["content"].replace(old, new)}]'),
    ("decisionsim stage1a blind history accepts a task sentence after an action", "wf/decisionsim.py",
     'if i == 0 or i > first_action or lines[i] != task_line:', 'if i == 0 or lines[i] != task_line:'),
    ("decisionsim stage1a blind history accepts another game's task", "wf/decisionsim.py",
     'if i == 0 or i > first_action or lines[i] != task_line:', 'if i == 0 or i > first_action:'),
    ("decisionsim stage1a blind history accepts two task sentences", "wf/decisionsim.py",
     '    if len(at) != 1:\n        raise ValueError(f"expected exactly one task sentence', '    if not at:\n        raise ValueError(f"expected exactly one task sentence'),
    ("decisionsim stage1a blind prompt not checked for the query's history", "wf/decisionsim.py",
     'if last["content"].count(old) != 1:', 'if False:'),
    ("decisionsim stage1a blind arm refused by the scoring run", "wf/decisionsim.py",
     'ARMS = YN_ARMS + BLIND_BASES', 'ARMS = YN_ARMS'),
    ("decisionsim stage1a blind arm not scored by default", "wf/decisionsim.py",
     'DEFAULT_ARMS = ("yn_aligned", "yn_absent", "yn_swapped", "yn_blind")', 'DEFAULT_ARMS = ("yn_aligned", "yn_absent", "yn_swapped")'),
    ("decisionsim stage1a aligned-vs-blind contrast missing", "wf/decisionsim.py",
     'out = [(ALIGNED, b) for b in blinds] + [(ALIGNED, NEUTRAL)]', 'out = [(ALIGNED, NEUTRAL)]'),
    ("decisionsim stage1a aligned-vs-blind contrast reversed", "wf/decisionsim.py",
     'out = [(ALIGNED, b) for b in blinds] + [(ALIGNED, NEUTRAL)]', 'out = [(b, ALIGNED) for b in blinds] + [(ALIGNED, NEUTRAL)]'),
    ("decisionsim stage1a swapped-vs-blind contrast missing", "wf/decisionsim.py",
     '[(ALIGNED, SWAPPED)] + [(SWAPPED, b) for b in blinds]', '[(ALIGNED, SWAPPED)]'),
    ("decisionsim stage1a neutral-vs-blind contrast missing", "wf/decisionsim.py",
     '    out += [(NEUTRAL, b) for b in blinds]\n', '    out += []\n'),
    ("decisionsim stage1a decomposition against blind uses neutral", "wf/decisionsim.py",
     'rep[DECOMPOSITION_KEY[b]] = decomposition(states, L, b)', 'rep[DECOMPOSITION_KEY[b]] = decomposition(states, L, NEUTRAL)'),
    ("decisionsim stage1a decomposition labels every base neutral", "wf/decisionsim.py",
     '    name = BASE_NAME[base]\n', '    name = "neutral"\n'),
    ("decisionsim stage1a calibration against blind uses neutral", "wf/decisionsim.py",
     'rep[CALIBRATION_KEY[b]] = calibration(states, L[ALIGNED], L[b])', 'rep[CALIBRATION_KEY[b]] = calibration(states, L[ALIGNED], L[NEUTRAL])'),
    ("decisionsim stage1a blind shifts only where a swap exists", "wf/decisionsim.py",
     'baseline_shifts(even_swap if SWAPPED in (a, b) else even,', 'baseline_shifts(even_swap if SWAPPED in (a, b) or BLIND in (a, b) else even,'),
    ("decisionsim stage1a Option C decides against neutral", "wf/decisionsim.py",
     'OPTION_C_BASE = BLIND', 'OPTION_C_BASE = NEUTRAL'),
    ("decisionsim stage1a Option C harm and difference read against neutral", "wf/decisionsim.py",
     'c = rep[f"{ALIGNED}_vs_{base}"][OPTION_C_RULE]', 'c = rep[f"{ALIGNED}_vs_{NEUTRAL}"][OPTION_C_RULE]'),
    ("decisionsim stage1a Option C attribution read against neutral", "wf/decisionsim.py",
     '    dec = rep[DECOMPOSITION_KEY[base]][OPTION_C_RULE]\n    aligned_minus_base = dec[f"aligned_minus_{BASE_NAME[base]}"]',
     '    dec = rep["decomposition"][OPTION_C_RULE]\n    aligned_minus_base = dec["aligned_minus_neutral"]'),
    ("decisionsim stage1a Option C does not wait for the blind arm", "wf/decisionsim.py",
     'elif f"{ALIGNED}_vs_{base}" not in reports[tag]:', 'elif False:'),
    ("decisionsim stage1a secondary verdict decides", "wf/decisionsim.py",
     'report["option_c"] = option_c(report["models"], primary, OPTION_C_BASE)', 'report["option_c"] = option_c(report["models"], primary, OPTION_C_SECONDARY_BASE)'),
    ("decisionsim stage1a every verdict marked deciding", "wf/decisionsim.py",
     '"decides": base == OPTION_C_BASE,', '"decides": True,'),
    ("decisionsim stage1a no verifier for rule 2 is a random choice", "wf/decisionsim.py",
     '    if rule == "rule2_verification":\n        return float(not succeeded(_proposal(state)))\n', ''),
    ("decisionsim stage1a random choice over all candidates", "wf/decisionsim.py",
     'return _mean([float(not succeeded(t)) for t in rule_set(rule, state)])', 'return _mean([float(not succeeded(t)) for t in state["candidates"]])'),
    ("decisionsim stage1a oracle knows nothing", "wf/decisionsim.py",
     'return {t["id"]: (1.0 if succeeded(t) else -1.0) for s in states for t in s["candidates"]}',
     'return {t["id"]: 1.0 for s in states for t in s["candidates"]}'),
    ("decisionsim stage1a reference points at odd steps", "wf/decisionsim.py",
     '    even = [s for s in states if even_step(s)]\n    O = oracle_scores(states)', '    even = list(states)\n    O = oracle_scores(states)'),
    ("decisionsim stage1a reference points ignore where rule 3 applies", "wf/decisionsim.py",
     '            if rule_applies(rule, s):\n                fa = ', '            if True:\n                fa = '),
    ("decisionsim stage1a swapped reference points where no swap exists", "wf/decisionsim.py",
     'sts = [s for s in even if s["swap"]] if arm == SWAPPED else even', 'sts = even'),
    ("decisionsim stage1a headroom share inverted", "wf/decisionsim.py",
     '(_mean(none) - _mean(fa)) / head if head > 0', '(_mean(fa) - _mean(none)) / head if head > 0'),
    ("decisionsim stage1a verifier minus no verifier reversed", "wf/decisionsim.py",
     '"minus_no_verifier": _paired(fa, none, games)', '"minus_no_verifier": _paired(none, fa, games)'),
    ("decisionsim stage1a half-oracle accept/reject reads the model", "wf/decisionsim.py",
     'oracle_accept = choose_max([p] if succeeded(p) else _generated(state), L)', 'oracle_accept = choose_max([p] if accepts(L, p) else _generated(state), L)'),
    ("decisionsim stage1a half-oracle fallback reads the model", "wf/decisionsim.py",
     'oracle_fallback = choose_max([p] if accepts(L, p) else _generated(state), O)', 'oracle_fallback = choose_max([p] if accepts(L, p) else _generated(state), L)'),
    ("decisionsim stage1a reference points left out of the report", "wf/decisionsim.py",
     '    rep["reference_points"] = reference_points(states, L)\n', ''),
    ("decisionsim stage1a pass count without the blind arm", "wf/decisionsim.py",
     '"secondary": len(score_jobs(states, DEFAULT_ARMS))', '"secondary": len(score_jobs(states, YN_ARMS))'),
    # Review 2 (2026-11-21, before any score). The reviewer's mutants (scratchpad review_decisionsim_2/mutants.py),
    # adapted to the amended code. Two are left out as equivalent: "decomposition against blind takes aligned-minus-
    # swapped from neutral" (aligned minus swapped is the same number in every decomposition: same states, same two
    # arms) and "half-oracle fallback may return the rejected proposal" (the oracle picks a true success either way, and
    # every generated-only set has one). Retired at the same time, equivalent after review 2: fix1 "attribution ignores
    # the sign" (aligned minus base above zero and aligned minus swapped at least half of it already make the latter
    # positive).
    ("decisionsim review2 R1 blind task sentence may be the first line", "wf/decisionsim.py",
     "if i == 0 or i > first_action", "if i > first_action"),
    ("decisionsim review2 R2 complete_states ignores the blind arm", "wf/decisionsim.py",
     'for a in L for t in s["candidates"]', 'for a in L if a != BLIND for t in s["candidates"]'),
    ("decisionsim review2 R3 aligned-vs-blind contrast only on swap states", "wf/decisionsim.py",
     "if SWAPPED in (a, b) else states\n        key", "if (SWAPPED in (a, b) or BLIND in (a, b)) else states\n        key"),
    ("decisionsim review2 R4 blind scores not rounded", "wf/decisionsim.py",
     'q[x["condition"]][x["transition_id"]] = quantize(x["L"])',
     'q[x["condition"]][x["transition_id"]] = quantize(x["L"]) if x["condition"] != BLIND else x["L"]'),
    ("decisionsim review2 R5 secondary verdict computed against blind", "wf/decisionsim.py",
     'option_c(report["models"], primary, OPTION_C_SECONDARY_BASE)', 'option_c(report["models"], primary, OPTION_C_BASE)'),
    ("decisionsim review2 R6 harm rate over changed states only", "wf/decisionsim.py",
     '"harm_rate": _boot([r["harm"] for r in rows], games)',
     '"harm_rate": _boot([r["harm"] for r in rows if r["changed"]] or [0.0], [r["game"] for r in rows if r["changed"]] or ["x"])'),
    ("decisionsim review2 R7 paired difference reversed", "wf/decisionsim.py",
     "diffs = [x - y for x, y in zip(a, b)]", "diffs = [y - x for x, y in zip(a, b)]"),
    ("decisionsim review2 R8 toward success includes an interval touching zero", "wf/decisionsim.py",
     "elif hi < 0:", "elif hi <= 0:"),
    ("decisionsim review2 R9 calibration vs blind arguments reversed", "wf/decisionsim.py",
     "rep[CALIBRATION_KEY[b]] = calibration(states, L[ALIGNED], L[b])", "rep[CALIBRATION_KEY[b]] = calibration(states, L[b], L[ALIGNED])"),
    ("decisionsim review2 R10 decomposition aligned-minus-swapped sign", "wf/decisionsim.py",
     "f[ALIGNED] - f[SWAPPED], f[SWAPPED] - f[base]", "f[SWAPPED] - f[ALIGNED], f[SWAPPED] - f[base]"),
    ("decisionsim review2 R11 rounding truncates", "wf/decisionsim.py",
     "return round(x * GRID) / GRID", "return int(x * GRID) / GRID"),
    ("decisionsim review2 R12 Option C harm read against neutral only", "wf/decisionsim.py",
     '    harm = c["harm_rate"]["estimate"]', '    harm = rep[f"{ALIGNED}_vs_{NEUTRAL}"][OPTION_C_RULE]["harm_rate"]["estimate"]'),
    ("decisionsim review2 R13 Option C p from rule 1", "wf/decisionsim.py",
     '    p = c["fail_rate_difference"]["p"]', '    p = rep[f"{ALIGNED}_vs_{base}"]["rule1_with_proposal"]["fail_rate_difference"]["p"]'),
    ("decisionsim review2 R14 gate skips blind prompts", "wf/decisionsim.py",
     "[build(prompt_item(t, s, a), a) for s, t, a in jobs]", "[build(prompt_item(t, s, a), a) for s, t, a in jobs if a != BLIND]"),
    ("decisionsim review2 R15 blind prompt drops the examples", "wf/decisionsim.py",
     'return aligned[:-1] + [{**last, "content": last["content"].replace(old, new)}]',
     'return aligned[:1] + [{**last, "content": last["content"].replace(old, new)}]'),
    ("decisionsim review2 R16 lure counts pairs at odd steps", "wf/decisionsim.py",
     "        if not even_step(s):\n            continue\n        p = _proposal(s)\n        if not succeeded(p):",
     "        p = _proposal(s)\n        if not succeeded(p):"),
    ("decisionsim review2 R17 swapped headroom read on all even states", "wf/decisionsim.py",
     "head = _mean(none) - _mean(orc)\n            entry", 'head = by_rule[rule]["headroom"]\n            entry'),
    ("decisionsim review2 R18 attribution read on the contrast's states", "wf/decisionsim.py",
     'aligned_minus_base = dec[f"aligned_minus_{BASE_NAME[base]}"]', 'aligned_minus_base = c["fail_rate_difference"]["estimate"]'),
    ("decisionsim review2 R20 no-verifier rule 2 counts only determined failures", "wf/decisionsim.py",
     "return float(not succeeded(_proposal(state)))",
     'return float(not succeeded(_proposal(state)) and bool(_proposal(state).get("context_determined")))'),
    ("decisionsim review2 R21 tie split for b reads La", "wf/decisionsim.py",
     "for t in top_set(mb, Lb)]", "for t in top_set(mb, La)]"),
    ("decisionsim review2 R22 blind history accepts two task sentences (count >= 1)", "wf/decisionsim.py",
     "if len(at) != 1:", "if len(at) < 1:"),
    ("decisionsim review2 R23 rule 3 states counted over all states", "wf/decisionsim.py",
     '"n_rule3_states_even_step": sum(rule_applies("rule3_goal_relevant", s) for s in even)',
     '"n_rule3_states_even_step": sum(rule_applies("rule3_goal_relevant", s) for s in states)'),
    # Review 2's new code: the net condition, the fallback-menu condition, attribution sign, yn_blind_all, the step
    # split and the pays-for-itself note.
    ("decisionsim review2 N1 net condition ignored", "wf/decisionsim.py",
     "net_enough = net - net_required >= -TOL", "net_enough = True"),
    ("decisionsim review2 N2 headroom counts odd-step states", "wf/decisionsim.py",
     "        evens = [s for s in states if even_step(s)]\n", "        evens = list(states)\n"),
    ("decisionsim review2 N3 net counts harm only", "wf/decisionsim.py",
     '"net_states": harm - help_,', '"net_states": harm,'),
    ("decisionsim review2 N4 cell net counts harm only", "wf/decisionsim.py",
     '"harm": h, "help": g, "net": h - g,', '"harm": h, "help": g, "net": h,'),
    ("decisionsim review2 N5 fallback-menu condition ignored", "wf/decisionsim.py",
     "through_fallback = fallback_net - OPTION_C_FALLBACK_SHARE * net > TOL", "through_fallback = False"),
    ("decisionsim review2 N6 fallback at exactly half counts as through it", "wf/decisionsim.py",
     "through_fallback = fallback_net - OPTION_C_FALLBACK_SHARE * net > TOL",
     "through_fallback = fallback_net - OPTION_C_FALLBACK_SHARE * net >= -TOL"),
    ("decisionsim review2 N7 fallback read from another cell", "wf/decisionsim.py",
     'fallback_net = c["by_acceptance"]["rejected_both"]["net"]', 'fallback_net = c["by_acceptance"]["accepted_only_a"]["net"]'),
    ("decisionsim review2 N8 fallback checked before attribution", "wf/decisionsim.py",
     '    elif toward_failure and net_enough and attributable:\n        category = "toward failure, through the fallback menu"',
     '    elif toward_failure and net_enough and through_fallback:\n        category = "toward failure, through the fallback menu"'),
    ("decisionsim review2 N9 fallback category after not attributable", "wf/decisionsim.py",
     'OPTION_C_CATEGORIES = ("worth doing", "toward failure, through the fallback menu",\n                       "toward failure, not attributable to the real task",',
     'OPTION_C_CATEGORIES = ("worth doing", "toward failure, not attributable to the real task",\n                       "toward failure, through the fallback menu",'),
    ("decisionsim review2 N10 attribution ignores the sign of aligned minus base", "wf/decisionsim.py",
     "attributable = (aligned_minus_base > TOL and ", "attributable = ("),
    ("decisionsim review2 N11 pays-for-itself strict", "wf/decisionsim.py",
     'ref["fail_rate"]["estimate"] - ref["no_verifier_fail_rate"] <= TOL', 'ref["fail_rate"]["estimate"] - ref["no_verifier_fail_rate"] < -TOL'),
    ("decisionsim review2 N12 yn_blind_all allowed for a secondary model", "wf/decisionsim.py",
     "if BLIND_ALL in arms and tag not in PRIMARY:", "if False:"),
    ("decisionsim review2 N13 example task lines not checked", "wf/decisionsim.py",
     "    if seen != want:\n", "    if False:\n"),
    ("decisionsim review2 N14 yn_blind_all keeps the query's task", "wf/decisionsim.py",
     "return strip_example_tasks(blind_messages(base(t, ALIGNED), t))", "return strip_example_tasks(base(t, ALIGNED))"),
    ("decisionsim review2 N15 yn_blind_all not scored for the primary models", "wf/decisionsim.py",
     "PRIMARY_ARMS = DEFAULT_ARMS + (BLIND_ALL,)", "PRIMARY_ARMS = DEFAULT_ARMS"),
    ("decisionsim review2 N16 format verdict against yn_blind", "wf/decisionsim.py",
     'option_c(report["models"], primary, OPTION_C_FORMAT_BASE)', 'option_c(report["models"], primary, OPTION_C_BASE)'),
    ("decisionsim review2 N17 format sensitivity inverted", "wf/decisionsim.py",
     'else fmt != report["option_c"]["verdict"])', 'else fmt == report["option_c"]["verdict"])'),
    ("decisionsim review2 N18 calibration keys collide", "wf/decisionsim.py",
     'CALIBRATION_KEY = {BLIND: "calibration_vs_blind", BLIND_ALL: "calibration_vs_blind_all"}',
     'CALIBRATION_KEY = {BLIND: "calibration_vs_blind", BLIND_ALL: "calibration_vs_blind"}'),
    ("decisionsim review2 N19 no decomposition against yn_blind_all", "wf/decisionsim.py",
     "for b in (NEUTRAL, *blinds):", "for b in (NEUTRAL, BLIND):"),
    ("decisionsim review2 N20 the two task-blind prompts not compared", "wf/decisionsim.py",
     "        out += [(BLIND_ALL, BLIND)]\n", "        out += []\n"),
    ("decisionsim review2 N21 step 0 split takes step 2 too", "wf/decisionsim.py",
     'contrast(OPTION_C_RULE, [s for s in states if s["state_idx"] == 0], L[ALIGNED], L[b])',
     'contrast(OPTION_C_RULE, [s for s in states if s["state_idx"] <= 2], L[ALIGNED], L[b])'),
    ("decisionsim review2 N22 primary pass count without yn_blind_all", "wf/decisionsim.py",
     '"primary": len(score_jobs(states, PRIMARY_ARMS))', '"primary": len(score_jobs(states, DEFAULT_ARMS))'),
    ("decisionsim review2 N23 net share a quarter", "wf/decisionsim.py",
     "OPTION_C_NET_SHARE = 0.5", "OPTION_C_NET_SHARE = 0.25"),
    ("decisionsim review2 N24 fallback share three quarters", "wf/decisionsim.py",
     "OPTION_C_FALLBACK_SHARE = 0.5", "OPTION_C_FALLBACK_SHARE = 0.75"),
    # Review 3 (2026-11-21, before any score): the reviewer's non-equivalent survivors (scratchpad
    # review_decisionsim_3/mutants3.py) and mutants of the review 3 fixes. Left out as equivalent: S9 (the same as the
    # retired fix1 "attribution ignores the sign") and S11 (the task-blind query has no task line left to strip).
    ("decisionsim review3 S1 pays flag read from the deciding base for every verdict", "wf/decisionsim.py",
     '.get("by_arm", {}).get(base, {})', '.get("by_arm", {}).get(OPTION_C_BASE, {})'),
    ("decisionsim review3 S2 worth-doing-inside flag set when pays is unknown", "wf/decisionsim.py",
     'category == "worth doing" and pays is False', 'category == "worth doing" and not pays'),
    ("decisionsim review3 S3 fallback share read as an absolute value", "wf/decisionsim.py",
     "through_fallback = fallback_net - OPTION_C_FALLBACK_SHARE * net > TOL",
     "through_fallback = abs(fallback_net) - OPTION_C_FALLBACK_SHARE * net > TOL"),
    ("decisionsim review3 S4 headroom counts only determined failing proposals", "wf/decisionsim.py",
     'out["headroom_states"] = sum(int(no_verifier_fail(rule, s))',
     'out["headroom_states"] = sum(int(no_verifier_fail(rule, s) and bool(_proposal(s).get("context_determined")))'),
    ("decisionsim review3 S6 step split against neutral", "wf/decisionsim.py",
     '"step0": contrast(OPTION_C_RULE, [s for s in states if s["state_idx"] == 0], L[ALIGNED], L[b])',
     '"step0": contrast(OPTION_C_RULE, [s for s in states if s["state_idx"] == 0], L[ALIGNED], L[NEUTRAL])'),
    ("decisionsim review3 S10 format sensitivity ignores a waiting deciding verdict", "wf/decisionsim.py",
     '(None if "waiting" in fmt or "waiting" in report["option_c"]["verdict"]', '(None if "waiting" in fmt'),
    ("decisionsim review3 T1 half compared without tolerance", "wf/decisionsim.py",
     'dec["aligned_minus_swapped"] - OPTION_C_SWAP_SHARE * aligned_minus_base >= -TOL)',
     'dec["aligned_minus_swapped"] >= OPTION_C_SWAP_SHARE * aligned_minus_base)'),
    ("decisionsim review3 T3 verdict does not report its states", "wf/decisionsim.py",
     '"comparison": base, "n_states": c.get("n_states"),', '"comparison": base, "n_states": None,'),
    ("decisionsim review3 T4 verdict does not report incomplete states", "wf/decisionsim.py",
     '"n_states_incomplete": rep.get("n_states_incomplete"),', '"n_states_incomplete": None,'),
]



# Stage 1b mutants (2026-11-20) are in their own file, so the Stage 1a and Stage 1b branches merge cleanly.
from mutants_decisionsim_agent import MUTANTS as _DSA_MUTANTS  # noqa: E402

MUTANTS += _DSA_MUTANTS


def _covering_tests(work: Path, rel: str) -> list[str]:
    """Test files that reference the mutated module (dotted name, `from pkg import mod`, or file name)."""
    path = Path(rel)
    names = [path.name]
    if path.suffix == ".py":
        dotted = ".".join(path.with_suffix("").parts)
        names += [dotted, f"from {'.'.join(path.parent.parts)} import {path.stem}"]
    found = [str(p.relative_to(work)) for p in sorted((work / "tests").glob("test_*.py"))
             if any(n in p.read_text(encoding="utf-8") for n in names)]
    return found or ["tests"]


# Kill rules (notes/testing.md, "Mutation runner: strict kill rule", 2026-11-23).
#
# "strict", the default: a mutant is killed only when pytest exits 1 (tests ran and failed) and at least one named test
# FAILED. Anything else that is not a pass is "error-only": a collection or import error, a usage or internal error,
# no tests collected, or a run whose only named problems are ERRORs in setup or teardown. Such a mutant made the
# suite unrunnable, which says nothing about whether any test checks the behaviour it changed. A timeout is
# inconclusive and never a kill. This is families wave B's rule (scripts/mutation_check.py on families-waveb-r1,
# 2026-11-22) with one tightening: wave B also accepted a named ERROR line as a kill.
#
# "any-exit" (--legacy-any-exit): any non-zero exit is a kill, the rule of every log of this runner before 2026-11-23.
# Kept only to reproduce those logs; it skips the baseline check, as they did.
FAILED_RE = re.compile(r"^FAILED (\S+::\S+)", re.MULTILINE)
ERROR_RE = re.compile(r"^ERROR (\S+::\S+)", re.MULTILINE)
STRICT, ANY_EXIT = "strict", "any-exit"


def classify(returncode: int | None, stdout: str, rule: str = STRICT) -> tuple[str, str]:
    """(outcome, detail) of one pytest run: killed, error, survived or timeout (returncode None)."""
    if returncode is None:
        return "timeout", "no result within the time limit"
    if returncode == 0:
        return "survived", ""
    failed = FAILED_RE.findall(stdout)
    if rule == ANY_EXIT:
        return "killed", failed[0] if failed else f"pytest exit {returncode} (no failing test named)"
    if returncode == 1 and failed:
        return "killed", failed[0]
    errors = ERROR_RE.findall(stdout)
    if errors:
        return "error", f"pytest exit {returncode}, no named test failed (first error {errors[0]})"
    return "error", f"pytest exit {returncode} (no failing test named)"


def _pytest(python: str, work: Path, targets: list[str], timeout: int, stop_first: bool = True) -> tuple[int | None, str, str]:
    cmd = [python, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", "--tb=line", *(["-x"] if stop_first else []), *targets]
    # No bytecode: a .pyc records the source's mtime in whole seconds and its size, so a same-length mutant written
    # within a second of an earlier run of the same file would be read from the stale cache and look like a survivor.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "", ""
    return r.returncode, r.stdout, r.stderr


def _tail(stdout: str, stderr: str, n: int = 8) -> str:
    return "\n".join((stdout.strip().splitlines() + stderr.strip().splitlines())[-n:])


def _run_tests(python: str, work: Path, targets: list[str], timeout: int, rule: str) -> dict:
    rc, out, err = _pytest(python, work, targets, timeout)
    outcome, detail = classify(rc, out, rule)
    if rule == STRICT and outcome == "error" and rc == 1 and not FAILED_RE.search(out):
        # -x stopped at a setup or teardown error; a later test may still fail by name, so run on without -x.
        rc2, out2, err2 = _pytest(python, work, targets, timeout, stop_first=False)
        outcome2, detail2 = classify(rc2, out2, rule)
        if outcome2 == "killed":
            return {"outcome": outcome2, "by": detail2 + " (run without -x after a setup error)", "returncode": rc2,
                    "targets": targets, "tail": _tail(out2, err2)}
    return {"outcome": outcome, "by": detail, "returncode": rc, "targets": targets, "tail": _tail(out, err)}


def mutant_key(mutant: tuple[str, str, str, str]) -> str:
    return hashlib.sha256("\x00".join(mutant).encode("utf-8")).hexdigest()[:16]


def load_mutants(path: Path) -> list[tuple[str, str, str, str]]:
    """MUTANTS of another runner or mutant file (for instance a frozen study's, from a checkout of its tag)."""
    path = path.resolve()
    siblings = {p.stem for p in path.parent.glob("*.py")}
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k in siblings}  # the file's own imports, not ours
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"_mutants_from_{mutant_key((str(path), '', '', ''))}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
        for k in siblings:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return [tuple(m) for m in mod.MUTANTS]


def _read_results(path: Path | None, rule: str) -> dict[str, dict]:
    """Records already written under this rule, by mutant key (resume); a cut-off last line is ignored."""
    done: dict[str, dict] = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("rule") == rule:
                done[rec["key"]] = rec
    return done


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


LABELS = {"killed": "killed     ", "error": "ERROR-ONLY ", "survived": "SURVIVED   ", "timeout": "TIMEOUT    ",
          "not applied": "NOT APPLIED"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shard", nargs="?", default="0/1", help="i/n runs MUTANTS[i::n] (several cores, one process each)")
    ap.add_argument("--legacy-any-exit", action="store_true",
                    help="count any non-zero pytest exit as a kill (the rule before 2026-11-23; to reproduce old logs)")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repository to copy and mutate (default: this one); e.g. a worktree of a frozen study's tag")
    ap.add_argument("--mutants-from", type=Path, default=None,
                    help="take MUTANTS from this file instead of this runner's list")
    ap.add_argument("--results", type=Path, default=None,
                    help="JSONL file: one record appended per mutant as it finishes; mutants already in it are skipped")
    ap.add_argument("--require-unique", action="store_true",
                    help="a mutant whose correct text is not found exactly once is not applied (the τ-bench runner's rule)")
    args = ap.parse_args(argv)
    rule = ANY_EXIT if args.legacy_any_exit else STRICT
    root = args.root.resolve()
    python = sys.executable
    shard, shards = map(int, args.shard.split("/"))
    mutants = load_mutants(args.mutants_from) if args.mutants_from else MUTANTS
    chosen = mutants[shard::shards]
    # MUTANTS_MATCHING="2026-11-17" runs only the mutants whose name contains that text ("a|b": either).
    only = os.environ.get("MUTANTS_MATCHING")
    if only:
        chosen = [m for m in chosen if any(o in m[0] for o in only.split("|"))]
    # MUTATION_TIMEOUT: seconds per test run (raise it when shards share a machine and the suite is slow).
    timeout = int(os.environ.get("MUTATION_TIMEOUT", "900"))
    done = _read_results(args.results, rule)
    text = args.results.read_text(encoding="utf-8") if args.results and args.results.exists() else ""
    if text and not text.endswith("\n"):
        with args.results.open("a", encoding="utf-8") as f:  # a run cut off mid-record: start the next on its own line
            f.write("\n")
    commit, dirty = _git(root, "rev-parse", "HEAD"), bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    counts: dict[str, list[str]] = {k: [] for k in LABELS}
    passed_baselines: set[tuple[str, ...]] = set()

    def baseline(work: Path, targets: list[str]) -> None:
        """The unmutated copy must pass the tests a mutant is judged by, or every mutant would look killed."""
        if rule == ANY_EXIT or tuple(targets) in passed_baselines:
            return
        rc, out, err = _pytest(python, work, targets, timeout)
        if rc != 0:
            raise SystemExit(f"BASELINE FAILS on {targets} (pytest exit {rc}): no mutant can be judged\n{_tail(out, err, 20)}")
        passed_baselines.add(tuple(targets))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"

        def skip(folder: str, names: list[str]) -> list[str]:
            top = {".git", ".venv", ".envs", "runs", "runs_mock", "docs"} if Path(folder) == root else set()
            return [n for n in names if n in top or n in ("__pycache__", ".pytest_cache", "pace.env") or n.endswith(".pdf")]

        shutil.copytree(root, work, ignore=skip)
        for mutant in chosen:
            name, rel, good, bad = mutant
            key = mutant_key(mutant)
            if key in done:
                rec = done[key]
            else:
                path = work / rel
                source = path.read_text(encoding="utf-8") if path.exists() else ""
                n = source.count(good)
                rec = {"name": name, "file": rel, "key": key, "rule": rule, "root": str(root), "root_commit": commit,
                       "root_dirty": dirty, "occurrences": n}
                start = time.time()
                if n == 0 or (args.require_unique and n != 1):
                    rec.update(outcome="not applied", by=f"correct text found {n} times")
                else:
                    covering = _covering_tests(work, rel)
                    baseline(work, covering)
                    path.write_text(source.replace(good, bad), encoding="utf-8")
                    try:
                        # Fast path: the test files that name the mutated module. A mutant that survives them is run
                        # against the whole suite before it counts as a survivor.
                        res = _run_tests(python, work, covering, timeout, rule)
                        if res["outcome"] == "survived" and covering != ["tests"]:
                            path.write_text(source, encoding="utf-8")
                            baseline(work, ["tests"])
                            path.write_text(source.replace(good, bad), encoding="utf-8")
                            res = _run_tests(python, work, ["tests"], timeout, rule)
                    finally:
                        path.write_text(source, encoding="utf-8")
                    rec.update(res)
                rec["seconds"] = round(time.time() - start, 1)
                if args.results:
                    with args.results.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
            counts[rec["outcome"]].append(name)
            by = f"  [{rec['by']}]" if rec.get("by") and rec["outcome"] != "survived" else ""
            print(f"{LABELS[rec['outcome']]}  {name}{by}", flush=True)
    print(f"\nrule {rule}: {len(counts['killed'])}/{len(chosen)} killed by a named failing test; "
          f"error-only (not counted as killed)={counts['error']}; survivors={counts['survived']}; "
          f"inconclusive (timeout)={counts['timeout']}; not applied={counts['not applied']}")
    return 1 if any(counts[k] for k in ("error", "survived", "timeout", "not applied")) else 0


if __name__ == "__main__":
    sys.exit(main())
