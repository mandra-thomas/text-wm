"""Put deliberate bugs into the TextCraft pipeline, one at a time, and confirm the TextCraft tests notice.

    python -m wf.textcraft.mutation_check [jobs] [log.json] [--results FILE.jsonl] [--legacy-any-exit]

jobs: parallel workers, each with its own copy. Each worker takes a fixed share (every jobs-th mutant in list order),
so a run is reproducible; the killing test of each mutant (the first failing test, pytest stops there) is printed
and, given a path, logged as JSON at the end. --results appends one JSON line per mutant as it finishes, and a rerun
skips mutants already recorded there under the same rule, so an interrupted run resumes.

Works on a copy of the repository made without bytecode caches, so the working tree is never touched, and pytest
runs with PYTHONDONTWRITEBYTECODE=1. Each mutant runs against tests/test_textcraft_*.py. The kill rule is the strict
one of scripts/mutation_check.py (below); a timeout is reported as inconclusive, never as caught.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = "wf/textcraft/"

# (bug, file, correct text, buggy text)
MUTANTS = [
    # text-only checker
    ("checker ignores what a craft uses up", T + "checker.py", "                ledger.remove(n, c)\n", "                pass\n"),
    ("checker counts every get as one", T + "checker.py", "ledger.add(got.group(2), int(got.group(1)))", "ledger.add(got.group(2), 1)"),
    ("checker calls an exact holding short", T + "checker.py", "        if held < c:\n", "        if held <= c:\n"),
    ("checker writes the item with spaces", T + "checker.py", 'return Verdict(True, reply=f"Could not find enough items to craft {to_id(target)}"',
     'return Verdict(True, reply=f"Could not find enough items to craft minecraft:{target}"'),
    ("checker class precedence reversed", T + "checker.py", 'CLASS_ORDER = ("used_up", "too_few", "never_obtained")',
     'CLASS_ORDER = ("never_obtained", "too_few", "used_up")'),
    ("checker reads used up from the current holding", T + "checker.py", 'kind = "used_up" if peak >= c else', 'kind = "used_up" if held >= c else'),
    ("checker trusts inventory replies", T + "checker.py", "            if shown != {k: v for k, v in ledger.held.items() if v}:\n",
     "            if False:\n"),
    ("checker get rule ignores the list", T + "checker.py", "        if line is not None:  # the object", "        if True:  # the object"),
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
    ("analysis clusters by item", T + "analysis.py", '        diffs.append(d)\n        pairs.append(t["pair_id"])\n', '        diffs.append(d)\n        pairs.append(t["id"])\n'),
    ("verdict skips Holm", T + "analysis.py", 'ok = r["estimate"] > 0 and r["ci"][0] > 0 and adjusted[m] < P_MAX',
     'ok = r["estimate"] > 0 and r["ci"][0] > 0 and r.get("perm_p", 1.0) < P_MAX'),
    # vendored environment
    ("environment lets an exact holding fail", T + "upstream/environment.py",
     "or self.inventory[itemtag_count.item_tag.item_id] < itemtag_count.count", "or self.inventory[itemtag_count.item_tag.item_id] <= itemtag_count.count"),
    # review 1's mutants (scratch review_textcraft/mutants.py), R11 and R15 rewritten for the amended code
    ("R1 selection seed drops the bin", T + "build.py", 'f"{SEED}|textcraft|select|{population}|{c}|{b}"', 'f"{SEED}|textcraft|select|{population}|{c}"'),
    ("R2 round robin never shuffles pairs", T + "build.py", "    rng.shuffle(order)\n    chosen = []\n", "    chosen = []\n"),
    ("B1 couple draw never shuffles pairs", T + "build.py", "    rng.shuffle(order)\n    chosen: list[dict] = []\n", "    chosen: list[dict] = []\n"),
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
    ("R12 missing scores never flagged", T + "analysis.py", '    missing = {(m, t["id"], a) for m in models for t in items for a in ARMS} - have',
     "    missing = set()"),
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
    ("seed: main selection drops the bin", T + "build.py", 'f"{SEED}|textcraft|select|{population}|{c}|{b}|{r}|couples"', 'f"{SEED}|textcraft|select|{population}|{c}|{r}|couples"'),
    ("seed: list shuffle", T + "build.py", 'f"{SEED}|textcraft|list|{a}|{b}"', 'f"{SEED}|textcraft|list2|{a}|{b}"'),
    ("seed: pairing shuffle", T + "build.py", 'f"{SEED}|textcraft|pairs|{depth}"', 'f"{SEED}|textcraft|pairs2|{depth}"'),
    ("seed: walk", T + "build.py", 'f"{SEED}|textcraft|walk|{pair.pair_id}|{seed}"', 'f"{SEED}|textcraft|walk2|{pair.pair_id}|{seed}"'),
    ("seed: prompt-example pairs", T + "build.py", 'f"{SEED}|textcraft|fewshot")', 'f"{SEED}|textcraft|fewshot2")'),
    ("seed: prompt-example failure item", T + "build.py", 'f"{SEED}|textcraft|fewshot|item"', 'f"{SEED}|textcraft|fewshot|item2"'),
    # amendments after review 1 (design section 11)
    ("A1 tag-reading items kept", T + "build.py", "    if tag_reading_ambiguous(state, short):\n", "    if False:\n"),
    ("A1 warped stems not read as logs", T + "build.py", '**{f"{s} stems": (f"{s} stem", f"{s} hyphae", f"stripped {s} stem", f"stripped {s} hyphae")\n                for s in ("crimson", "warped")}}',
     '**{f"{s} stems": (f"{s} stem", f"{s} hyphae", f"stripped {s} stem", f"stripped {s} hyphae")\n                for s in ("crimson",)}}'),
    ("A1 checker counts white wool as wool", T + "checker.py", '"wool": ("black wool",', '"wool": ("white wool", "black wool",'),
    ("A2 arithmetic flag set by any obtained input", T + "build.py", '"needs_arithmetic": all(s["peak"] > 0 for s in short),',
     '"needs_arithmetic": any(s["peak"] > 0 for s in short),'),
    ("A2 checker purity check off", T + "checker.py", '        if item.get("pure_class") != (len({s["kind"] for s in v.short_inputs}) == 1):\n',
     "        if False:\n"),
    ("A2 secondary readout on all main items", T + "analysis.py", '[t for t in pops["main"] if t["needs_arithmetic"]]', 'pops["main"]'),
    ("A3 builder fit counts ties as aligned", T + "build.py", 'fit = "aligned" if n["aligned"] > n["swapped"] else', 'fit = "aligned" if n["aligned"] >= n["swapped"] else'),
    ("A3 builder fit counts shared names", T + "build.py", 'n = {"aligned": sum(o in a - p for o in objs), "swapped": sum(o in p - a for o in objs)}\n    fit',
     'n = {"aligned": sum(o in a for o in objs), "swapped": sum(o in p for o in objs)}\n    fit'),
    ("A3 checker tree ignores generic members", T + "checker.py", "        wanted = {name} | set(GENERIC_MEMBERS.get(name, ()))\n", "        wanted = {name}\n"),
    ("A3 verify accepts unbalanced cells", T + "verify.py", '            and not r["main_fit_unbalanced_cells"] and', "            and"),
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
    # review 2's mutants (scratch review_textcraft2/mutants2.py); N7 rewritten for the couple draw
    ("N1 tag exclusion reads peak not current holding", T + "build.py", '    return any(state.held(m) > 0 for s in short for m in TAG_READING.get(s["name"], ()))',
     '    return any(state.peaked(m) > 0 for s in short for m in TAG_READING.get(s["name"], ()))'),
    ("N2 coals has no members", T + "build.py", '    "coals": ("coal",),', '    "coals": (),'),
    ("N3 stone materials drop blackstone", T + "build.py", '    "stone crafting materials": ("blackstone", "cobblestone"),', '    "stone crafting materials": ("cobblestone",),'),
    ("N4 verify tag-reading check off", T + "verify.py", '    if any(held & set(B.TAG_READING.get(s["name"], ())) for s in item["short_inputs"]):\n', "    if False:\n"),
    ("N5 builder step_object keeps minecraft id", T + "build.py", '        return W.name_of(reply.split(" ", 2)[2])', '        return reply.split(" ", 2)[2]'),
    ("N6 checker tree does not recurse into inputs", T + "checker.py", '                todo += [out] + [n for _c, n in inputs]', '                todo += [out]'),
    ("N7 couple draw shares one rng over roles", T + "build.py", 'f"{SEED}|textcraft|select|{population}|{c}|{b}|{r}|couples"', 'f"{SEED}|textcraft|select|{population}|{c}|{b}|couples"'),
    ("N8 item hash omits prefix candidates", T + "score.py", '"neutral_task_line", "candidates", "prefix_candidates")}', '"neutral_task_line", "candidates")}'),
    ("N9 resume refuses only if every saved score is stale", T + "score.py", '    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):',
     '    if saved and all(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):'),
    ("N10 stale check ignores unknown items", T + "analysis.py", 'for s in scores if hashes.get(s["transition_id"]) != s.get("item_hash")})',
     'for s in scores if s["transition_id"] in hashes and hashes[s["transition_id"]] != s.get("item_hash")})'),
    ("N11 overlap by substring", T + "build.py", '    return bool(set(a.split(" ")) & set(b.split(" ")))', '    return a in b or b in a'),
    ("N12 get prefix drops the count", T + "build.py", '    return {"o_plus": " ".join(twin.split(" ")[:2]), "o_minus": "Could not find"}', '    return {"o_plus": twin.split(" ")[0], "o_minus": "Could not find"}'),
    ("N13 secondary readout on items NOT needing arithmetic", T + "analysis.py", '    return t["needs_arithmetic"] and not t["inventory_shown_after_last_change"]',
     '    return not t["needs_arithmetic"] and not t["inventory_shown_after_last_change"]'),
    ("N14 verify overlap check off", T + "verify.py", '    if item["object_goal_overlap"] != B.shares_word(item["object"], item["goal"]) or \\\n', '    if False and \\\n'),
    ("N15 missing model counts as p=0", T + "analysis.py", 'primary[m]["perm_p"] == primary[m]["perm_p"] else 1.0', 'primary[m]["perm_p"] == primary[m]["perm_p"] else 0.0'),
    ("N16 name remainder uses prefix_swapped twice", T + "analysis.py", 'for arm in ("aligned", "prefix_aligned", "swapped", "prefix_swapped")]',
     'for arm in ("aligned", "prefix_swapped", "swapped", "prefix_swapped")]'),
    ("N17 builder arithmetic flag from current holding", T + "build.py", '"needs_arithmetic": all(s["peak"] > 0 for s in short),', '"needs_arithmetic": all(s["held"] > 0 for s in short),'),
    ("N18 checker arithmetic check from current holding", T + "checker.py", '        if item.get("needs_arithmetic") != all(s["peak"] > 0 for s in v.short_inputs):',
     '        if item.get("needs_arithmetic") != all(s["held"] > 0 for s in v.short_inputs):'),
    ("N19 prefix strata read the full replies", T + "analysis.py", '"delta_L_prefix": effect(scores, complete(scores, v, PREFIX_SET), "prefix_aligned", "prefix_swapped")}',
     '"delta_L_prefix": effect(scores, complete(scores, v, PREFIX_SET), "aligned", "swapped")}'),
    ("N20 selection role split ignores role capacity", T + "build.py", "            per_role = _split(per_bin[b], {r: cap[(c, b, r)] for r in ROLES})",
     "            per_role = {r: per_bin[b] // 2 for r in ROLES}"),
    # fix round 2 (design section 12)
    ("B1 couple takes both items from one goal", T + "build.py", "for it in (rng.choice(first), rng.choice(second)):", "for it in (rng.choice(first), rng.choice(first)):"),
    ("B1 couple slots ignore the history", T + "build.py", '    return it["pair_id"], json.dumps(it["history"]), it["failure_class"], it["role"]',
     '    return it["pair_id"], "", it["failure_class"], it["role"]'),
    ("B1 couple slots ignore the role", T + "build.py", '    return it["pair_id"], json.dumps(it["history"]), it["failure_class"], it["role"]',
     '    return it["pair_id"], json.dumps(it["history"]), it["failure_class"], ""'),
    ("B1 couple slots accept one goal", T + "build.py", "        if len(goals) == 2:\n            first, second", "        if len(goals) >= 1:\n            first, second"),
    ("B1 verify couple check off", T + "verify.py", '        if len(its) != 2 or len({json.dumps(', '        if False and len({json.dumps('),
    ("B1 verify mirror check off", T + "verify.py", '        elif any(F.measures(t["recipe_lines"]', '        elif False and any(F.measures(t["recipe_lines"]'),
    ("B1 gate ignores couple problems", T + "verify.py", ' and r.get("main_couple_problems", ["missing"]) == [])', ")"),
    ("B1 structure couple id check off", T + "verify.py", '    if item.get("couple_id", "missing") != want_couple:\n', "    if False:\n"),
    ("B1 fit first step reads the last", T + "fit.py", '"first_step": side(tagged[0][0])', '"first_step": side(tagged[-1][0])'),
    ("B1 fit completion ignores generic members", T + "fit.py", "        have = lambda n: held.get(n, 0) + sum(held.get(m, 0) for m in GENERIC_MEMBERS.get(n, ()))",
     "        have = lambda n: held.get(n, 0)"),
    ("B1 fit goal words keep shared words", T + "fit.py", "wa, wp = set(goal.split()) - set(partner.split()),", "wa, wp = set(goal.split()),"),
    ("B1 fit held_now counts shared names", T + "fit.py", '"held_now": [sum(v for n, v in held.items() if n in ea)', '"held_now": [sum(v for n, v in held.items() if n in ea or n in ep)'),
    ("B1 no-fit subset drops neutral items", T + "fit.py", "    return all(s <= 0 for s in signs(fit_measures).values())", "    return all(s < 0 for s in signs(fit_measures).values())"),
    ("B1 verify fit-measure check off", T + "verify.py", '    if item["fit_measures"] != F.measures(', '    if False and item["fit_measures"] != F.measures('),
    ("B1 regression readout reports a slope", T + "analysis.py", 'return {"intercept": float(coef[0]),', 'return {"intercept": float(coef[1]),'),
    ("B1 bootstrap of the regression reads a slope", T + "analysis.py", "    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])", "    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])"),
    ("B1 no-fit readout on every item", T + "analysis.py", '"delta_L_no_fit_for_aligned": effect(scores, sub, "aligned", "swapped"),',
     '"delta_L_no_fit_for_aligned": effect(scores, items, "aligned", "swapped"),'),
    ("B2 class contrast pools roles", T + "analysis.py", '    cell = lambda c, r: [t for t in main if t["failure_class"] == c and t["role"] == r]',
     '    cell = lambda c, r: [t for t in main if t["failure_class"] == c]'),
    ("B3 prefix readouts on different item sets", T + "analysis.py", "    same = complete(scores, items, PREFIX_SET)", "    same = items"),
    ("B3 missing check covers the primary arms only", T + "analysis.py", 'for t in items for a in ARMS} - have', 'for t in items for a in ("aligned", "swapped")} - have'),
    ("B4 checker certificate keeps the first change", T + "checker.py", "        self.changed_at[name] = self.step\n        if self.held[name]",
     "        self.changed_at.setdefault(name, self.step)\n        if self.held[name]"),
    ("B4 checker peak step is the last time at the peak", T + "checker.py", "        if self.held[name] > self.peak.get(name, 0):", "        if self.held[name] >= self.peak.get(name, 0):"),
    ("B4 builder certificate takes the first change", T + "build.py", '"held_step": changed[-1] if changed else "never"', '"held_step": changed[0] if changed else "never"'),
    ("B4 checker certificate check off", T + "checker.py", '    if item.get("certificate") != v.certificate:\n', "    if False:\n"),
    ("B4 verify certificate check off", T + "verify.py", '        if item["certificate"] != B.certificate(state, inputs):\n', "        if False:\n"),
    ("B5 plural items kept", T + "build.py", "    if plural_ambiguous(state, short):\n", "    if False:\n"),
    ("B5 nether brick twins dropped", T + "build.py", '    ("nether brick", "nether bricks"), ', "    "),
    ("B5 verify plural check off", T + "verify.py", '    if any(B._TWIN_OF.get(s["name"]) in held for s in item["short_inputs"]):\n', "    if False:\n"),
    ("B6 inventory flag counts the last change itself", T + "build.py", '    return any(o.startswith("Inventory: ") for _a, o in history[last_change + 1:])',
     '    return any(o.startswith("Inventory: ") for _a, o in history)'),
    ("B6 checker inventory-flag check off", T + "checker.py", '    if item.get("inventory_shown_after_last_change") is not inventory_shown_after_last_change(item["history"]):\n', "    if False:\n"),
    ("B6 arithmetic readout keeps shown inventories", T + "analysis.py", '    return t["needs_arithmetic"] and not t["inventory_shown_after_last_change"]', '    return t["needs_arithmetic"]'),
    ("B7 tokcheck joined-vs-alone off", T + "tokcheck.py", "            if ids(tok, text + cand) != prompt + alone:\n", "            if False:\n"),
    ("B7 tokcheck prefix check off", T + "tokcheck.py", "                if b[: len(a)] != a:\n", "                if False:\n"),
    ("B7 smoke run skips the token check", T + "score.py", '        if cfg.get("token_check"):', "        if False:"),
    ("B7 token check failure does not stop the run", T + "score.py", "            if not report[\"ok\"]:\n", "            if False:\n"),
    ("C2 mirrored couple takes the same profile", T + "build.py", "                    want = tuple(-v for v in fit_profile(x))", "                    want = fit_profile(x)"),
    ("C2 usefulness admits partner overlap", T + "build.py", '    return it["role"] == "ingredient" and not it["object_goal_overlap"] and not it["object_partner_overlap"]',
     '    return it["role"] == "ingredient" and not it["object_goal_overlap"]'),
    ("C2 verify mirrored-couple check off", T + "verify.py", "                    or B.fit_profile(its[0]) != tuple(-v for v in B.fit_profile(its[1])):", "                    or False:"),
    ("C2 usefulness readout on single items (rewritten in section 13)", T + "analysis.py", 'effect(scores, couples_where(core, B_usefulness_only), "aligned", "swapped")',
     'effect(scores, [t for t in core if B_usefulness_only(t)], "aligned", "swapped")'),
    ("C3 checker success ignores the removal", T + "checker.py", 'for k in ("reply", "twin", "certificate", "removal")', 'for k in ("reply", "twin", "certificate")'),
    ("C3 checker line match ignores counts", T + "checker.py", "if x[0] == c and (x[1] == n or", "if (x[1] == n or"),
    ("C3 checker success needs no displayed line", T + "checker.py", "        if not any(matches_line(line, count, target, inputs) for line in recipe_lines):\n", "        if False:\n"),
    ("C3 success candidates reversed", T + "build.py", '{"o_plus": item["reply"], "o_minus": item["twin"]} if success else', '{"o_plus": item["twin"], "o_minus": item["reply"]} if success else'),
    ("C3 verify success-reply check off", T + "verify.py", '    if item["reply"].startswith(("Crafted ", "Got ")) != success:\n', "    if False:\n"),
    ("C3 competence threshold 0.5", T + "analysis.py", '"discriminates": bool(boots) and a >= 0.6 and', '"discriminates": bool(boots) and a >= 0.5 and'),
    ("C4 equivalence on the 95% interval", T + "analysis.py", "    ci90 = [_pct(boots, 0.05), _pct(boots, 0.95)]", "    ci90 = [_pct(boots, 0.025), _pct(boots, 0.975)]"),
    ("C4 margins exchanged", T + "analysis.py", 'EQUIVALENCE_MARGIN = {"qwen3-8b-fewshot": 1.0, "llama3.1-8b-fewshot": 0.3}', 'EQUIVALENCE_MARGIN = {"qwen3-8b-fewshot": 0.3, "llama3.1-8b-fewshot": 1.0}'),
    ("C7 named readouts keep used up on the goal", T + "analysis.py", "    core = [t for t in main if not_used_up_goal(t)]\n", "    core = main\n"),
    ("C7 agreement contrast not halved", T + "analysis.py", '"estimate": r["estimate"] / 2,', '"estimate": r["estimate"],'),
    ("C5 inventory contrast reads the last-change flag", T + "analysis.py", '    return any(o.startswith("Inventory: ") for _a, o in t["history"])', '    return t["inventory_shown_after_last_change"]'),
    ("C8 prompt example goal by coin only", T + "build.py", "                    goal = outside[0] if len(outside) == 1 else", "                    goal = outside[0] if False else"),
    ("B1 prompt examples use every walk", T + "build.py", '\n                       and it["walk_seed"] < FEWSHOT_WALKS)', ")"),
    # review 3's mutants (scratch review_textcraft3/mut3.py); R3-12 and R3-20 rewritten, R3-21 reversed (its "bug"
    # was the fix made in section 13: the counting readout on whole couples)
    ("R3-01 couple slots accept a single goal", T + "build.py", "        if len(goals) == 2:\n            first, second",
     "        if len(goals) >= 1 and len(goals) == len(goals):\n            first, second = (sorted(goals[g], key=lambda it: it['id']) for g in (sorted(goals) * 2)[:2])\n            if False: first, second"),
    ("R3-02 couple draws both items from the first goal", T + "build.py", "for it in (rng.choice(first), rng.choice(second)):", "for it in (rng.choice(first), rng.choice(first)):"),
    ("R3-03 mirrored couple matches the same profile", T + "build.py", "want = tuple(-v for v in fit_profile(x))", "want = fit_profile(x)"),
    ("R3-04 mirrored draw ignores main slots", T + "build.py", "and usefulness_only(it) and slot_key(it) not in exclude)", "and usefulness_only(it))"),
    ("R3-05 fit first_step reads the last tagged step", T + "fit.py", '"first_step": side(tagged[0][0])', '"first_step": side(tagged[-1][0])'),
    ("R3-06 fit completion ignores generic members", T + "fit.py", "held.get(n, 0) + sum(held.get(m, 0) for m in GENERIC_MEMBERS.get(n, ()))", "held.get(n, 0)"),
    ("R3-07 fit goal_words keeps shared words", T + "fit.py", "wa, wp = set(goal.split()) - set(partner.split()), set(partner.split()) - set(goal.split())", "wa, wp = set(goal.split()), set(partner.split())"),
    ("R3-08 checker peak_step is the last time the peak is reached", T + "checker.py", "        if self.held[name] > self.peak.get(name, 0):", "        if self.held[name] >= self.peak.get(name, 0):"),
    ("R3-09 builder certificate held_step is the first change", T + "build.py", '"held_step": changed[-1] if changed else "never"', '"held_step": changed[0] if changed else "never"'),
    ("R3-10 checker success ignores the line's output count", T + "checker.py", "    if (k, out) != (count, target) or len(wanted) != len(inputs):", "    if out != target or len(wanted) != len(inputs):"),
    ("R3-11 competence ignores the interval's lower end", T + "analysis.py", '"discriminates": bool(boots) and a >= 0.6 and _pct(boots, 0.025) > 0.5}', '"discriminates": bool(boots) and a >= 0.6}'),
    ("R3-12 competence per class uses every failure", T + "analysis.py", '(cls, [t for t in failures if cls == "all" or t["failure_class"] == cls])', "(cls, list(failures))"),
    ("R3-13 named readouts keep used up on the goal", T + "analysis.py", "    core = [t for t in main if not_used_up_goal(t)]\n    ingredient", "    core = list(main)\n    ingredient"),
    ("R3-14 named readouts not Holm-adjusted", T + "analysis.py", "    for k, p in holm(with_p).items():", "    for k, p in with_p.items():"),
    ("R3-15 overlap readout on single items, not couples", T + "analysis.py", 'couples_where(core, lambda t: ingredient(t) and t["object_goal_overlap"])', '[t for t in core if ingredient(t) and t["object_goal_overlap"]]'),
    ("R3-16 agreement contrast not halved", T + "analysis.py", 'return {**r, "estimate": r["estimate"] / 2, "ci": [x / 2 for x in r["ci"]]}', 'return {**r, "estimate": r["estimate"], "ci": [x / 2 for x in r["ci"]]}'),
    ("R3-17 inventory contrast unweighted across cells", T + "analysis.py", "                w = len(yes) + len(no)\n", "                w = 1\n"),
    ("R3-18 equivalence margins swapped between models", T + "analysis.py", 'EQUIVALENCE_MARGIN = {"qwen3-8b-fewshot": 1.0, "llama3.1-8b-fewshot": 0.3}', 'EQUIVALENCE_MARGIN = {"qwen3-8b-fewshot": 0.3, "llama3.1-8b-fewshot": 1.0}'),
    ("R3-19 checker inventory-shown flag not reset by a later get", T + "checker.py", "        if GOT.fullmatch(reply) or CRAFTED.fullmatch(reply):\n            shown = False", "        if GOT.fullmatch(reply) or CRAFTED.fullmatch(reply):\n            shown = shown"),
    ("R3-20 usefulness readout adds non-couple items (rewritten: the extras are descriptive now)", T + "analysis.py", '"usefulness_only": effect(scores, couples_where(core, B_usefulness_only),',
     '"usefulness_only": effect(scores, couples_where(core, B_usefulness_only) + [t for t in core if B_usefulness_only(t)],'),
    ("R3-21 counting readout on single items (reversed)", T + "analysis.py", "    return couples_where(main, counting_item)", "    return [t for t in main if counting_item(t)]"),
    ("R3-22 item hash ignores the candidates", T + "score.py", '"neutral_task_line", "candidates", "prefix_candidates")}', '"neutral_task_line")}'),
    # section 13
    ("13 counting competence on every failure", T + "analysis.py", '{"counting": arithmetic_items(core)})', '{"counting": failures})'),
    ("13 counting gate ignored", T + "analysis.py", '    if not comp["counting"]["discriminates"]:', "    if False:"),
    ("13 coherence-robust keeps history-favoured members", T + "analysis.py", 'effect(scores, [t for t in main if t["history_fit"] != "aligned"], "aligned", "swapped")', 'effect(scores, main, "aligned", "swapped")'),
    ("13 inventory contrast keeps readable inventories", T + "analysis.py", '        if t["inventory_shown_after_last_change"]:\n            continue\n', ""),
    ("13 inventory contrast strata by bin only", T + "analysis.py", 'rows.append(((t["failure_class"], t["role"], t["length_bin"]), ', 'rows.append((t["length_bin"], '),
    ("13 yes/no margins exchanged", T + "analysis.py", 'YES_NO_MARGIN = {"qwen3-8b-fewshot": 0.47, "llama3.1-8b-fewshot": 0.37}', 'YES_NO_MARGIN = {"qwen3-8b-fewshot": 0.37, "llama3.1-8b-fewshot": 0.47}'),
    ("13 standardized equivalence on the raw mean", T + "analysis.py", "STANDARDIZED_MARGIN[model], standardized=True)", "STANDARDIZED_MARGIN[model], standardized=False)"),
    ("13 yes/no equivalence reads the two-candidate arms", T + "analysis.py", 'YES_NO_MARGIN[model], "yn_aligned", "yn_swapped")', 'YES_NO_MARGIN[model], "aligned", "swapped")'),
    ("13 withheld run writes per-model results", T + "analysis.py", '"incomplete_estimates_not_results": per_model}', '"per_model": per_model}'),
    ("13 reading skips the confirmed-and-equivalent row", T + "analysis.py", '        if eq["two_candidate"]["equivalent"]:', "        if False:"),
    ("13 reading ignores competence for a confirmed effect", T + "analysis.py", "        if not competent:\n            out +=", "        if False:\n            out +="),
    ("13 reading ignores the coherence-robust interval", T + "analysis.py", '        if not robust["ci"][0] > 0:', "        if False:"),
    ("13 negative rule ignores the interval", T + "analysis.py", 'r["estimate"] < 0 and r["ci"][1] < 0 and p < P_MAX', 'r["estimate"] < 0 and p < P_MAX'),
    ("13 null reading needs only one scale", T + "analysis.py", "    if len(scales) == len(eq):", "    if scales:"),
    ("13 item hash ignores the prompt examples", T + "score.py", '"fewshot": list(P.fewshot_examples() if shots is None else shots)}', '"fewshot": []}'),
    ("13 item hash ignores the system text", T + "score.py", '"system": P.SYSTEM if system is None else system,', '"system": P.SYSTEM,'),
    ("13 scorer hashes a variant with the verified context", T + "score.py", "    hashes = {t[\"id\"]: item_hash(t, shots, system) for t in items}", "    hashes = {t[\"id\"]: item_hash(t) for t in items}"),
    ("13 variant prompt drops the variant system text", T + "score.py", "messages=P.messages(t, arm, shots, system),", "messages=P.messages(t, arm, shots),"),
    ("13 variant sample seed ignores the variant", T + "score.py", 'f"{C.RNG_SEED}|textcraft|variant|{variant}"', 'f"{C.RNG_SEED}|textcraft|variant"'),
    ("13 variant comparison sign reversed", T + "analysis.py", "lambda _by, tid: dl(by_v, tid) - dl(by_m, tid)", "lambda _by, tid: dl(by_m, tid) - dl(by_v, tid)"),
    ("13 variant report reads scores made under another prompt", T + "analysis.py", "        if bad:\n            out[name] = {\"not_read\"", "        if False:\n            out[name] = {\"not_read\""),
    ("13 golden check uses the variant's examples", T + "score.py", "shots if golden_shots is None else golden_shots)", "shots)"),
    ("13 analysis stale check ignores the manifest's examples", T + "analysis.py", "    stale = stale_scores(scores, items, shots)", "    stale = stale_scores(scores, items)"),
    ("13 prompt drops the system override", T + "prompts.py", '"content": SYSTEM if system is None else system}', '"content": SYSTEM}'),
    ("13 tokcheck ignores the variant system text", T + "tokcheck.py", "        messages = P.messages(item, arm, shots, system)", "        messages = P.messages(item, arm, shots)"),
    ("13 tie couples keep single tie members", T + "analysis.py", 'couples_where(main, lambda t: t["history_fit"] == "tie")', '[t for t in main if t["history_fit"] == "tie"]'),
    ("13 tie couples read every couple", T + "analysis.py", 'couples_where(main, lambda t: t["history_fit"] == "tie")', 'couples_where(main, lambda t: True)'),
]


# Kill rules: the same as scripts/mutation_check.py (notes/testing.md, "Mutation runner: strict kill rule", 2026-11-23;
# this runner since 2026-11-24). "strict", the default: a mutant is killed only when pytest exits 1 (tests ran and
# failed) and at least one named test FAILED. Any other non-zero exit is "error-only": a collection or import error,
# a usage or internal error, or a run whose only named problems are ERRORs in setup or teardown (then the run is
# repeated without -x, and a named failure further on still counts). A timeout is inconclusive, never a kill.
# "any-exit" (--legacy-any-exit): any non-zero exit is a kill, the rule of every log of this runner before 2026-11-24.
FAILED_RE = re.compile(r"^FAILED (\S+::\S+)", re.MULTILINE)
ERROR_RE = re.compile(r"^ERROR (\S+)", re.MULTILINE)
STRICT, ANY_EXIT = "strict", "any-exit"
TARGETS = "tests/test_textcraft_*.py"


def classify(returncode: int | None, stdout: str, rule: str = STRICT) -> tuple[str, str]:
    """(outcome, detail) of one pytest run: killed, error, survived or timeout (returncode None)."""
    if returncode is None:
        return "timeout", "no result within the time limit"
    if returncode == 0:
        return "survived", ""
    failed = FAILED_RE.findall(stdout)
    if rule == ANY_EXIT:  # the old report: the first FAILED or ERROR line's id
        first = next((line.split(" ", 2)[1] for line in stdout.splitlines() if line.startswith(("FAILED ", "ERROR "))), "")
        return "killed", first
    if returncode == 1 and failed:
        return "killed", failed[0]
    errors = ERROR_RE.findall(stdout)
    if errors:
        return "error", f"pytest exit {returncode}, no named test failed (first error {errors[0]})"
    return "error", f"pytest exit {returncode} (no failing test named)"


def _pytest(work: Path, timeout: int, stop_first: bool = True) -> tuple[int | None, str, str]:
    tests = sorted(str(p.relative_to(work)) for p in (work / "tests").glob("test_textcraft_*.py"))
    cmd = [sys.executable, "-m", "pytest", "-q", *(["-x"] if stop_first else []), "-rfE", "-p", "no:cacheprovider",
           "--tb=no", *tests]
    # No bytecode (2026-11-24, as scripts/mutation_check.py since cc1db87). A .pyc is trusted when its source's mtime,
    # in whole seconds, and size match. A file restored within the second its same-length mutant was compiled would
    # be read back from the mutant's .pyc, so every later mutant would "die" against code that is still broken.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, "", ""
    return r.returncode, r.stdout, r.stderr


def _run_tests(work: Path, timeout: int, rule: str = STRICT) -> dict:
    """Outcome, the killing test (the first failing test, as pytest's short summary names it) and the exit code."""
    rc, out, _ = _pytest(work, timeout)
    outcome, by = classify(rc, out, rule)
    if rule == STRICT and outcome == "error" and rc == 1 and not FAILED_RE.search(out):
        # -x stopped at a setup or teardown error; a later test may still fail by name, so run on without -x.
        rc2, out2, _ = _pytest(work, timeout, stop_first=False)
        outcome2, by2 = classify(rc2, out2, rule)
        if outcome2 == "killed":
            return {"outcome": outcome2, "by": by2 + " (run without -x after a setup error)", "returncode": rc2}
    return {"outcome": outcome, "by": by, "returncode": rc}


def _fresh_copy(root: Path, work: Path) -> None:
    """Copy the repository without bytecode caches, git data, environments and run outputs."""
    def skip(folder: str, names: list[str]) -> list[str]:
        top = {".git", ".venv", ".envs", "runs", "runs_mock", "docs"} if Path(folder) == root else set()
        return [n for n in names if n in top or n in ("__pycache__", ".pytest_cache", "pace.env")
                or n.endswith((".pdf", ".pyc"))]

    shutil.copytree(root, work, ignore=skip)
    for cache in sorted(work.rglob("__pycache__"), reverse=True):  # none should be left; make sure
        shutil.rmtree(cache)


def mutant_key(mutant: tuple[str, str, str, str]) -> str:
    """The key scripts/mutation_check.py gives the same mutant, so records of the two runners can be matched."""
    return hashlib.sha256("\x00".join(mutant).encode("utf-8")).hexdigest()[:16]


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


LABELS = {"killed": "killed", "error": "ERROR-ONLY", "survived": "SURVIVED", "timeout": "TIMEOUT",
          "not applied": "NOT APPLIED"}


def _one(work: Path, mutant: tuple[str, str, str, str], timeout: int, rule: str, meta: dict) -> dict:
    """Apply one mutant in this worker's own copy, run the tests, restore the file."""
    name, rel, good, bad = mutant
    path = work / rel
    source = path.read_text(encoding="utf-8")
    n = source.count(good)
    rec = {"name": name, "file": rel, "key": mutant_key(mutant), "rule": rule, **meta, "occurrences": n}
    start = time.time()
    if n != 1:
        rec.update(outcome="not applied", by=f"correct text found {n} times")
    else:
        path.write_text(source.replace(good, bad), encoding="utf-8")
        try:
            rec.update(_run_tests(work, timeout, rule), targets=[TARGETS])
        finally:
            path.write_text(source, encoding="utf-8")
    rec["seconds"] = round(time.time() - start, 1)
    return rec


def _worker(args: tuple[Path, list, int, str, dict, dict, str | None]) -> list[dict]:
    work, mutants, timeout, rule, meta, done, results = args
    out = []
    for m in mutants:
        rec = done.get(mutant_key(m))
        if rec is None:
            rec = _one(work, m, timeout, rule, meta)
            if results:
                with open(results, "a", encoding="utf-8") as f:  # one short append per record
                    f.write(json.dumps(rec) + "\n")
        out.append(rec)
        print(f"{LABELS[rec['outcome']]:<12} {rec['name']}  <- {rec.get('by') or '-'}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    """Each of `jobs` workers runs a fixed share of the mutants (every jobs-th, in list order) in its own copy of the
    repo; with a log path, the outcome and killing test of every mutant are written to it as JSON at the end."""
    import argparse
    import multiprocessing

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="?", type=int, default=1, help="parallel workers, each with its own copy")
    ap.add_argument("log", nargs="?", type=Path, default=None, help="JSON summary of every mutant, written at the end")
    ap.add_argument("--legacy-any-exit", action="store_true",
                    help="count any non-zero pytest exit as a kill (the rule before 2026-11-24; to reproduce old logs)")
    ap.add_argument("--results", type=Path, default=None,
                    help="JSONL file: one record appended per mutant as it finishes; mutants already in it are skipped")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    rule = ANY_EXIT if args.legacy_any_exit else STRICT
    jobs = args.jobs
    timeout = int(os.environ.get("MUTATION_TIMEOUT", "900"))
    done = _read_results(args.results, rule)
    if args.results and args.results.exists():
        text = args.results.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):  # a run cut off mid-record: start the next on its own line
            with args.results.open("a", encoding="utf-8") as f:
                f.write("\n")
    meta = {"root": str(ROOT), "root_commit": _git(ROOT, "rev-parse", "HEAD"),
            "root_dirty": bool(_git(ROOT, "status", "--porcelain", "--untracked-files=no"))}
    results = str(args.results) if args.results else None
    with tempfile.TemporaryDirectory() as tmp:
        works = [Path(tmp) / f"repo{k}" for k in range(jobs)]
        for w in works:
            _fresh_copy(ROOT, w)
        rc, out, err = _pytest(works[0], timeout)
        if rc != 0:
            print(f"the unmutated tests do not pass (pytest exit {rc}); nothing to measure\n"
                  + "\n".join((out + err).strip().splitlines()[-20:]))
            return 1
        shares = [(works[k], MUTANTS[k::jobs], timeout, rule, meta, done, results) for k in range(jobs)]
        if jobs == 1:
            recs = _worker(shares[0])
        else:
            with multiprocessing.get_context("spawn").Pool(jobs) as pool:
                recs = [r for share in pool.map(_worker, shares) for r in share]
    by_name = {r["name"]: r for r in recs}
    if args.log is not None:
        args.log.write_text(json.dumps([{"mutant": name, "outcome": LABELS[by_name[name]["outcome"]],
                                         "killed_by": by_name[name].get("by", ""),
                                         "returncode": by_name[name].get("returncode")}
                                        for name, *_ in MUTANTS], indent=1) + "\n", encoding="utf-8")
    others = [(name, LABELS[by_name[name]["outcome"]]) for name, *_ in MUTANTS if by_name[name]["outcome"] != "killed"]
    killed = len(MUTANTS) - len(others)
    named = " by a named failing test" if rule == STRICT else ""
    print(f"\nrule {rule}: {killed}/{len(MUTANTS)} killed{named}; others: {others}")
    return 0 if not others else 1


if __name__ == "__main__":
    sys.exit(main())
