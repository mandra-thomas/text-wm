#!/usr/bin/env python3
"""Put each known bug back, one at a time, and confirm the test suite notices.

A surviving mutant means the tests would not catch that bug coming back. Runs on a copy of
the repo, so the working tree is never touched.
"""
from __future__ import annotations

import os
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
]


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


def _run_tests(python: str, work: Path, targets: list[str], timeout: int = 900) -> str:
    try:
        result = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=no", *targets],
                                cwd=work, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    return "killed" if result.returncode != 0 else "survived"


def main() -> int:
    python = sys.executable
    # `mutation_check.py 2/4` runs the third of four shards, so the check can use several cores.
    shard, shards = map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1)
    chosen = MUTANTS[shard::shards]
    # MUTANTS_MATCHING="2026-11-17" runs only the mutants whose name contains that text ("a|b": either).
    only = os.environ.get("MUTANTS_MATCHING")
    if only:
        chosen = [m for m in chosen if any(o in m[0] for o in only.split("|"))]
    # MUTATION_TIMEOUT: seconds per test run (raise it when shards share a machine and the suite is slow).
    timeout = int(os.environ.get("MUTATION_TIMEOUT", "900"))
    survivors: list[str] = []
    broken: list[str] = []
    timeouts: list[str] = []
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
            # Fast path: the test files that name the mutated module. A mutant that survives them is run
            # against the whole suite before it counts as a survivor; a timeout is never counted as a kill.
            outcome = _run_tests(python, work, _covering_tests(work, rel), timeout)
            if outcome == "survived":
                outcome = _run_tests(python, work, ["tests"], timeout)
            path.write_text(source, encoding="utf-8")
            label = {"killed": "killed  ", "survived": "SURVIVED", "timeout": "TIMEOUT "}[outcome]
            print(f"{label}  {name}", flush=True)
            if outcome == "survived":
                survivors.append(name)
            elif outcome == "timeout":
                timeouts.append(name)
    killed = len(chosen) - len(survivors) - len(broken) - len(timeouts)
    print(f"\n{killed}/{len(chosen)} killed; survivors={survivors}; inconclusive (timeout)={timeouts}; "
          f"not applied={broken}")
    return 1 if survivors or broken or timeouts else 0


if __name__ == "__main__":
    sys.exit(main())
