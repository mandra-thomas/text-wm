"""Build every mechanism-battery prompt on the real items (no GPU) and check it, before any scoring job starts.

    RUNS_DIR=<runs with rq12_transitions_s*> python scripts/check_mech2_prompts.py [--examples]

Checks, for every (item, arm, format) the run would score:
- a content arm's prompt equals the aligned prompt with only the task line replaced (every other message and every
  other byte of the query turn identical);
- a position arm's prompt equals the plain prompt of its goal with only the task line's position changed, and the
  task line is the first (start) or last (end) line of the history; aliases (step 0, end) are not scored;
- the yes/no prompt equals the two-candidate prompt up to the final question;
- the item's real task line appears in the query turn only for the aligned, done and cancelled arms;
- task-line form: one sentence-final period, no doubled spaces or periods, articles agree, no object or receptacle
  instance except the action's receptacle in desire_check and negated; negated only on take items.
Prints counts by task type and action verb, the share of items whose "done" claim the history contradicts, and,
with --examples, one task line per arm and task type.
"""
import re
import sys
from collections import Counter, defaultdict

from wf import mech2, rq1
from wf.rq2 import SOURCES, load_items

primary = rq1.rq1_sets(load_items(SOURCES))["primary"]
sets = mech2.battery_items(primary)
union = sets["union"]
problems: Counter = Counter()
coverage: Counter = Counter()
examples: dict = {}
n_jobs = Counter()


def query(msgs):
    return msgs[-1]["content"]


def own_text(line: str, arm: str, t: dict) -> str:
    """The part of a task line this battery writes; the rest is inherited unchanged from earlier runs (RQ2's swapped
    line, rq1x's descriptions, the real task), whose wording, ALFWorld's "a apple" included, is kept for comparability."""
    if arm in ("mention_ignore", "mention_exists", "negated"):
        return line[len(t["swapped_task_line"]):].strip()
    if arm in ("done", "cancelled"):
        return line.split(mech2._task_text(t["task_line"]), 1)[1].strip()
    if arm in ("desire_keep", "desire_check"):
        return line
    return ""


def form(line: str, arm: str, t: dict) -> list[str]:
    out = []
    own = own_text(line, arm, t)
    if not line.endswith("."):
        out.append("no final period")
    if "  " in line or ".." in line or " ;" in line or " ." in line:
        out.append("doubled space or period")
    for m in re.finditer(r"\b(a|an) ([a-z]+)", own):
        if (m[1] == "an") != (m[2][0] in "aeiou"):
            out.append(f"article: {m[0]}")
    inst = set(re.findall(r"\b[a-z]+ \d+\b", line))
    allowed = {mech2.parse_action(t["action"])[2]} if arm in ("desire_check", "negated") else set()
    if inst - allowed:
        out.append(f"instance named: {sorted(inst - allowed)}")
    return out


for t, arm, fmt in mech2.jobs(union):
    n_jobs[fmt] += 1
    msgs = mech2.arm_messages(t, arm, fmt)
    base_arm = arm.rsplit("_", 1)[0] if arm in mech2.POSITION_ARMS else "aligned"
    plain = mech2.arm_messages(t, base_arm, fmt)
    line = mech2.task_line(t, arm)
    if msgs[:-1] != plain[:-1]:
        problems[f"{arm}/{fmt}: an earlier message differs"] += 1
    if arm in mech2.POSITION_ARMS:
        where = arm.rsplit("_", 1)[1]
        q_lines, p_lines = query(msgs).split("\n"), query(plain).split("\n")
        if sorted(q_lines) != sorted(p_lines) or [x for x in q_lines if x != line] != [x for x in p_lines if x != line]:
            problems[f"{arm}/{fmt}: more than the task line's position changed"] += 1
        hist = query(msgs).split("History:\n", 1)[1].split("\n\nAction:", 1)[0].split("\n")
        if (hist[0] if where == "start" else hist[-1]) != line:
            problems[f"{arm}/{fmt}: task line not at the {where}"] += 1
        if mech2.alias_of(t, arm):
            problems[f"{arm}: alias scored"] += 1
    else:
        aligned_q = query(plain)
        if aligned_q.count(t["task_line"]) != 1 or query(msgs) != aligned_q.replace(t["task_line"], line):
            problems[f"{arm}/{fmt}: query differs beyond the task line"] += 1
        for p in form(line, arm, t):
            problems[f"{arm}: {p}"] += 1
    shows_real = t["task_line"] in query(msgs) or mech2._task_text(t["task_line"]) in query(msgs)
    if shows_real != (arm in ("aligned", "done", "cancelled", "aligned_start", "aligned_end")):
        problems[f"{arm}/{fmt}: real task shown or missing"] += 1
    if fmt == "yn":
        cand = mech2.arm_messages(t, arm, "cand")
        if query(msgs)[: -len(mech2.rq1v.YN_QUESTION)] != query(cand)[: -len("Next observation:")]:
            problems[f"{arm}: yes/no prompt differs before the question"] += 1
    else:
        ttype = mech2.task_type(t["task_line"])
        coverage[(arm, ttype, t["action"].split()[0])] += 1
        examples.setdefault((arm, ttype), line)
        if arm.startswith("desire"):
            examples.setdefault((arm, "verb " + t["action"].split()[0]), line)

print(f"items: primary {len(primary)}, sample {len(sets['sample'])}, step-0 {len(sets['step0'])}, union {len(union)}, "
      f"games {len({t['game_id'] for t in union})}")
print(f"jobs per model: {dict(n_jobs)}")
print("problems:", dict(problems) or "none")
by_arm = defaultdict(Counter)
for (arm, ttype, verb), c in coverage.items():
    by_arm[arm][ttype] += c
for arm in mech2.ARMS:
    print(f"{arm:18s}", dict(sorted(by_arm[arm].items())))
verbs = Counter((ttype, verb) for (arm, ttype, verb), c in coverage.items() if arm == "desire_check")
print("task type x verb combinations carrying desire_check:", len(verbs))
sample = [t for t in union if t["id"] in sets["sample"]]
contra = [t for t in sample if mech2.done_contradicted(t)]
print(f"done contradicted by the history: {len(contra)} of {len(sample)} sample items; "
      f"{sum(mech2.done_contradicted(t) for t in union)} of {len(union)} in the union")
print("sample by reason:", dict(Counter(t["determination_reason"] for t in sample)))
print("sample by task type:", dict(Counter(mech2.task_type(t["task_line"]) for t in sample)))
placement = [t for t in sample if mech2.task_type(t["task_line"]) in mech2.PLACEMENT_TYPES]
ftt = [t for t in placement if mech2.takes_from_target_type(t)]
print(f"P5: placement items {len(placement)}; take from a target-type receptacle {len(ftt)}; done contradicted "
      f"{sum(mech2.done_contradicted(t) for t in placement)}; both {sum(mech2.done_contradicted(t) for t in ftt)}; "
      f"P5 items {sum(mech2.p5_eligible(t) for t in sample)}")
take = [t for t in sample if mech2.is_take(t)]
neg_built = sum(mech2.task_line(t, "negated") is not None for t in union)
if neg_built != sum(mech2.is_take(t) for t in union):
    problems["negated: built on an item that is not a take of the goal object, or missing on one that is"] += 1


print(f"P7 (negated, take items): {len(take)} sample items; claim status (scene at start, history):",
      dict(Counter(mech2.negated_status(t) for t in take)))
print("P7 take items by reason:", dict(Counter(t["determination_reason"] for t in take)), "; P7 items (history does not show the object there):", sum(mech2.p7_eligible(t) for t in sample))
print("sample step-0 items:", sum(t.get("history_len", 0) == 0 for t in sample), "games in sample:", len({t["game_id"] for t in sample}))
if "--examples" in sys.argv:
    for (arm, ttype), line in sorted(examples.items()):
        print(f"{arm:18s} {ttype:9s} {line}")
sys.exit(1 if problems else 0)
