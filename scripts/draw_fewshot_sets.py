"""Draw prompt-example sets 2 and 3 from Pilot A train transitions by rule and seed only (no scores)."""
import json, random, re, collections, sys
from wf.data.transitions import context_determined_label, determination_reason, history_facts
src, out_path = sys.argv[1], sys.argv[2]
T = [json.loads(l) for l in open(src)]
Tid = {t["id"]: t for t in T}
SET1_GAMES = ("train/700", "train/270", "train/2332")

def trim(h, pairs=2):
    lines = h.splitlines(); task = [l for l in lines if l.startswith("Your task is to:")][0]
    idx = [i for i, l in enumerate(lines) if l.startswith("> ")]
    return None if len(idx) < pairs else "\n".join([task] + lines[idx[-pairs]:])

def success_shown(f, action):
    """The history shows every precondition this action needs, so its success follows from the context."""
    if action.startswith("go to "):
        return f.knows_position and f.location != action[6:]
    m = re.fullmatch(r"take (.+) from (.+)", action)
    if m:
        return f.location == m[2] and not f.holding and m[1] in f.contents.get(m[2], set())
    m = re.fullmatch(r"move (.+) to (.+)", action)
    if m:
        return f.holding == m[1] and f.location == m[2] and m[2] not in f.closed
    m = re.fullmatch(r"(?:heat|cool|clean) (.+) with (.+)", action)
    if m:
        return f.holding == m[1] and f.location == m[2]
    m = re.fullmatch(r"open (.+)", action)
    if m:
        return f.location == m[1] and m[1] in f.closed
    return False


cells = collections.defaultdict(list)
for t in T:
    if t["game_id"] in SET1_GAMES or t.get("outcome_class") == "unparseable":
        continue
    h = trim(t["history_text"])
    if not h:
        continue
    obs = t["true_obs"].strip(); fail = obs == "Nothing happens."; rel = bool(t.get("targets_goal_object"))
    f = history_facts(h)
    # every example's outcome must follow from the history shown
    if fail and not context_determined_label(h, t["action"], obs):
        continue
    if not fail and not success_shown(f, t["action"]):
        continue
    if fail:
        r = determination_reason({**t, "history_text": h, "context_determined": True})
        m = re.fullmatch(r"take (.+) from (.+)", t["action"])
        if (r == "contents_or_open_state" and m and m[2] == f.location and m[2] in f.contents
                and m[1] not in f.contents[m[2]] and not f.holding):
            cells[(rel, "F", "contents")].append((t["id"], h))
        elif r == "away_from_receptacle" and not f.holding and t["action"].split()[0] in ("take", "open", "close"):
            cells[(rel, "F", "location")].append((t["id"], h))
    elif obs and len(obs) <= 160:
        cells[(rel, "S", None)].append((t["id"], h))
print({str(k): len(v) for k, v in sorted(cells.items(), key=str)})

# Redraw of set 2 (decided 2026-11-18, set 3 approved as drafted): `python scripts/draw_fewshot_sets.py
# <transitions> <out> <drafts.json>` keeps set 3 from the drafts and draws set 2 again with its own seed.
# Both of set 2's failures are takes, so one success must be a take too, or the verb alone predicts the
# outcome. The history may hold no malformed command (an object or receptacle without its number) and no
# walk that failed ("go to" answered "Nothing happens."), and a success may not undo the step before it
# (moving an object back where it was just taken from).
ACTION_OK = re.compile(r"(?:go to|open|close|examine|use) [a-z]+ \d+|take [a-z]+ \d+ from [a-z]+ \d+|"
                       r"move [a-z]+ \d+ to [a-z]+ \d+|(?:heat|cool|clean) [a-z]+ \d+ with [a-z]+ \d+|inventory|look")

def well_formed(h):
    lines = h.splitlines()
    failed_walk = any(l.startswith("> go to ") and i + 1 < len(lines) and lines[i + 1].strip() == "Nothing happens."
                      for i, l in enumerate(lines))
    return not failed_walk and all(ACTION_OK.fullmatch(l[2:].strip()) for l in lines if l.startswith("> "))

def undoes_last(h, action):
    m = re.fullmatch(r"move (.+) to (.+)", action)
    last = [l[2:] for l in h.splitlines() if l.startswith("> ")][-1]
    return bool(m) and last == f"take {m[1]} from {m[2]}"

if len(sys.argv) > 3:
    drafts = json.load(open(sys.argv[3]))
    used = set(SET1_GAMES) | {e["game"] for e in drafts["set3"]}
    rng = random.Random("20260924|set2-redraw")
    spec = [(True, "S", None), (False, "S", None), (True, "F", "location"), (False, "F", "contents")]
    take_success = rng.choice([0, 1])  # which success is the take, drawn, not chosen
    ex = []
    for k, cell in enumerate(spec):
        pool = [c for c in cells[cell] if Tid[c[0]]["game_id"] not in used and well_formed(c[1])
                and not undoes_last(c[1], Tid[c[0]]["action"])]
        if k == take_success:
            pool = [c for c in pool if Tid[c[0]]["action"].startswith("take ")]
        tid, h = rng.choice(sorted(pool)); t = Tid[tid]
        used.add(t["game_id"])
        ex.append({"cell": list(cell), "id": tid, "game": t["game_id"], "history": h, "action": t["action"],
                   "observation": t["true_obs"].strip()})
    rng.shuffle(ex)
    out = {"set2": ex, "set3": drafts["set3"], "set2_draft_replaced": drafts["set2"]}
    json.dump(out, open(out_path, "w"), indent=1)
    for e in ex:
        rel = "goal-relevant" if e["cell"][0] else "goal-irrelevant"
        kind = "success" if e["cell"][1] == "S" else f"failure ({e['cell'][2]})"
        print(f"--- {rel} {kind} | {e['game']} {e['id']}"); print(e["history"]); print("ACTION:", e["action"]); print("OBSERVATION:", e["observation"])
    sys.exit(0)

rng = random.Random(20260924)
design = {"set2": [(True, "S", None), (False, "S", None), (True, "F", "location"), (False, "F", "contents")],
          "set3": [(True, "S", None), (False, "S", None), (False, "F", "location"), (True, "F", "contents")]}
used, out = set(), {}
for name, spec in design.items():
    ex = []
    for cell in spec:
        tid, h = rng.choice(sorted(c for c in cells[cell] if Tid[c[0]]["game_id"] not in used)); t = Tid[tid]
        used.add(t["game_id"])
        ex.append({"cell": list(cell), "id": tid, "game": t["game_id"], "history": h, "action": t["action"],
                   "observation": t["true_obs"].strip()})
    rng.shuffle(ex); out[name] = ex
json.dump(out, open(out_path, "w"), indent=1)
for name, ex in out.items():
    print("=================", name)
    for e in ex:
        rel = "goal-relevant" if e["cell"][0] else "goal-irrelevant"
        kind = "success" if e["cell"][1] == "S" else f"failure ({e['cell'][2]})"
        print(f"--- {rel} {kind} | {e['game']} {e['id']}"); print(e["history"]); print("ACTION:", e["action"]); print("OBSERVATION:", e["observation"])
