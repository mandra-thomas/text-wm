"""Build every RQ1-run prompt on the real items (no GPU) and check it, before any scoring job starts.

    python scripts/check_rq1_prompts.py [tokenizer]

Checks: every arm builds for every item it applies to; no prompt but the aligned ones shows the item's true
task line (the full-state layout included); every full-state prompt shows a layout in each turn; the
fine-tuned prompts carry the arm's goal. Prints prompt lengths in tokens, for the time estimate.
"""
import statistics
import sys
from collections import Counter

from wf import rq1
from wf.rq2 import SOURCES, load_items

sets = rq1.rq1_sets(load_items(SOURCES))
problems: Counter = Counter()
examples_share_task: Counter = Counter()
lengths: dict[str, list[int]] = {}
tok = None
if len(sys.argv) > 1:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sys.argv[1])


def check(t: dict, arm: str, ft: bool) -> None:
    msgs = rq1.arm_messages(t, arm, ft)
    text = "\n".join(m["content"] for m in msgs)
    goal = rq1.ARMS[arm][0]
    query = msgs[0]["content"] if ft else msgs[-1]["content"]
    if goal != "aligned" and goal != "aligned_plus" and t["task_line"] in query:
        problems[f"{arm}: true task line shown"] += 1
    if goal != "aligned" and goal != "aligned_plus" and t["task_line"] in text:
        examples_share_task[arm] += 1  # an example from another game has the same task; same in both goals
    if goal in ("aligned", "aligned_plus") and t["task_line"] not in query:
        problems[f"{arm}: aligned task line missing"] += 1
    users = [m for m in msgs if m["role"] == "user"]
    if arm.endswith("_full") and not all(m["content"].startswith("Hidden state") for m in users):
        problems[f"{arm}: a turn without its layout"] += 1
    if ft and rq1.goal_line(t, goal) not in msgs[0]["content"]:
        problems[f"{arm} (fine-tuned): goal line missing"] += 1
    if tok is not None and len(lengths.setdefault(f"{arm}{' ft' if ft else ''}", [])) < 300:
        lengths[f"{arm}{' ft' if ft else ''}"].append(len(tok.apply_chat_template(msgs, tokenize=True)))


for t in sets["primary"]:
    for arm in rq1.FEWSHOT_SCORE_ARMS:
        check(t, arm, False)
    for arm in rq1.FT_SCORE_ARMS:
        check(t, arm, True)
for t in sets["failures"] + sets["successes"]:
    for arm in rq1.FEWSHOT_GEN_ARMS:
        check(t, arm, False)
    check(t, "aligned", True)
print({k: len(v) for k, v in sets.items()})
print("problems:", dict(problems) or "none")
print("items whose task is also an example's task (not a leak):", dict(examples_share_task))
for k, v in lengths.items():
    print(f"{k}: median {statistics.median(v)} tokens, max {max(v)}")
sys.exit(1 if problems else 0)
