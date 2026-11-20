"""Prompts, readout tokens, the aligned/swapped token pairing and token groups (notes/interp_design.md, §2, §5).

The prompts are the scored ones, byte for byte: the yes/no prompt of `rq1v` and the observation prompt of
`rq1`. Only the task line differs between goals. Tokens are read from the chat-templated text with character
offsets, and the result is checked against `apply_chat_template(tokenize=True)`, the scorer's own encoding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from wf.data.goals import make_swapped_task_line

FORMATS = ("yn", "obs")
GOALS = ("aligned", "swapped", "absent", "placebo")
READOUT = {"yn": ("Yes", "No"), "obs": ("You", "Nothing")}
# Finest groups first; history subgroups are disjoint and "hist_all" is their union.
GROUPS = ("span", "task_rest", "hist_obj", "hist_swap", "hist_rec", "hist_other", "hist_all", "action", "query",
          "final", "last")
COARSE_GROUPS = ("span", "task_rest", "hist_all", "action", "query", "final", "last")
HIST_SUB = ("hist_obj", "hist_swap", "hist_rec", "hist_other")


def messages(t: dict, fmt: str, goal: str, placebo_object: str | None = None) -> list[dict[str, str]]:
    """The item's prompt in format `fmt` ("yn" or "obs") under `goal`."""
    from wf import rq1, rq1v
    from wf.scoring.logodds import build_messages

    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    if goal == "placebo":
        if not placebo_object:
            raise ValueError(f"{t['id']}: the placebo goal needs a placebo object")
        line = make_swapped_task_line(t["task_line"], t["goal_object"], placebo_object)
        msgs = build_messages(t, "aligned", task_line=line)
        if fmt == "obs":
            return msgs
        last = msgs[-1]
        return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + rq1v.YN_QUESTION}]
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}")
    return rq1v.arm_messages(t, f"yn_{goal}") if fmt == "yn" else rq1.arm_messages(t, goal)


def task_object(t: dict, goal: str, placebo_object: str | None = None) -> str | None:
    return {"aligned": t.get("goal_object"), "swapped": t.get("swapped_goal_object"), "absent": None,
            "placebo": placebo_object}[goal]


def readout_ids(tok: Any, fmt: str) -> tuple[int, int]:
    """Token ids of the two readout words, each tokenised alone as the scorer does; both must be one token."""
    ids = [tok(w, add_special_tokens=False).input_ids for w in READOUT[fmt]]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"readout words are not single tokens: {READOUT[fmt]} -> {ids}")
    return ids[0][0], ids[1][0]


@dataclass
class Encoded:
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    user_start: int  # character offset of the final user message's content in `text`
    user_end: int

    def __len__(self) -> int:
        return len(self.ids)


def encode(tok: Any, msgs: list[dict[str, str]]) -> Encoded:
    """Render with the chat template as the scorer does (thinking off), tokenise with offsets, check the ids."""
    kw = dict(add_generation_prompt=True, enable_thinking=False)
    text = tok.apply_chat_template(msgs, tokenize=False, **kw)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    ref = tok.apply_chat_template(msgs, tokenize=True, **kw)
    ref = list(ref["input_ids"] if isinstance(ref, dict) else ref)
    if ids != ref:
        raise ValueError("tokenising the rendered text does not reproduce apply_chat_template's ids")
    content = msgs[-1]["content"]
    start = text.rfind(content)
    if start < 0:
        raise ValueError("the final user message is not in the rendered prompt")
    return Encoded(text=text, ids=ids, offsets=[tuple(o) for o in enc["offset_mapping"]], user_start=start,
                   user_end=start + len(content))


@dataclass
class Pair:
    """Two prompts that differ only inside one span: a shared prefix, a differing span and a shared suffix.

    Position i of `a` in the suffix corresponds to position i - len(a) + len(b) of `b`.
    """

    a: Encoded
    b: Encoded
    prefix: int
    suffix: int
    groups: dict[str, list[int]] = field(default_factory=dict)  # positions in `a`

    @property
    def span_a(self) -> tuple[int, int]:
        return self.prefix, len(self.a) - self.suffix

    @property
    def span_b(self) -> tuple[int, int]:
        return self.prefix, len(self.b) - self.suffix

    @property
    def equal_span(self) -> bool:
        return len(self.a) == len(self.b)

    def to_b(self, positions: list[int]) -> list[int]:
        """Map positions of `a` (suffix, or span when the spans have equal length) to `b`."""
        lo, hi = self.span_a
        out = []
        for i in positions:
            if i < self.prefix:
                out.append(i)
            elif i < hi:
                if not self.equal_span:
                    raise ValueError("span positions map across prompts only when the spans have equal length")
                out.append(i)
            else:
                out.append(i - len(self.a) + len(self.b))
        return out

    def flipped(self) -> "Pair":
        """The same pair with `b` as the target; groups re-expressed in `b`'s positions."""
        q = Pair(a=self.b, b=self.a, prefix=self.prefix, suffix=self.suffix)
        q.groups = {g: (self.to_b(pos) if g != "span" or self.equal_span else []) for g, pos in self.groups.items()}
        return q


def pair(a: Encoded, b: Encoded) -> Pair:
    n = min(len(a), len(b))
    p = 0
    while p < n and a.ids[p] == b.ids[p]:
        p += 1
    s = 0
    while s < n - p and a.ids[-1 - s] == b.ids[-1 - s]:
        s += 1
    if p == len(a) == len(b):
        raise ValueError("the two prompts are identical")
    return Pair(a=a, b=b, prefix=p, suffix=s)


def _receptacle(action: str) -> str | None:
    m = re.search(r"\b(?:from|to|with) ([a-z]+ \d+)$", action) or re.fullmatch(r"(?:open|close|go to) ([a-z]+ \d+)", action)
    return m[1] if m else None


def _action_object(action: str) -> str | None:
    m = re.match(r"(?:take|move|heat|cool|clean|examine) ([a-z]+)(?: \d+)?\b", action)
    return m[1] if m else None


def char_regions(e: Encoded, t: dict, other_object: str | None) -> dict[str, list[tuple[int, int]]]:
    """Character ranges of each region in the rendered text of `e` (one item's prompt)."""
    text, u0, u1 = e.text, e.user_start, e.user_end
    content = text[u0:u1]
    h0 = content.index("History:\n") + len("History:\n")
    act_at = content.index("\n\nAction: ", h0)
    act_end = content.index("\n\n", act_at + 2)
    hist = (u0 + h0, u0 + act_at)
    task_m = re.search(r"^Your task is to:.*$", text[hist[0]:hist[1]], flags=re.MULTILINE)
    if not task_m:
        raise ValueError(f"{t['id']}: no task line in the history")
    task = (hist[0] + task_m.start(), hist[0] + task_m.end())
    regions: dict[str, list[tuple[int, int]]] = {
        "task": [task],
        "action": [(u0 + act_at + 2, u0 + act_end)],
        "query": [(u0 + act_end + 2, u1)],
        "final": [(u1, len(text))],
    }
    words = {"hist_obj": _action_object(t["action"]) or t.get("goal_object"), "hist_swap": other_object,
             "hist_rec": _receptacle(t["action"])}
    hist_parts = [(hist[0], task[0]), (task[1], hist[1])]
    for g, w in words.items():
        regions[g] = []
        if not w:
            continue
        pat = re.compile(rf"\b{re.escape(w)}\b" + (r"(?: \d+)?" if g != "hist_rec" else ""))
        for lo, hi in hist_parts:
            regions[g] += [(lo + m.start(), lo + m.end()) for m in pat.finditer(text[lo:hi])]
    regions["hist"] = hist_parts
    return regions


def _inside(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    lo, hi = span
    return any(lo < r1 and hi > r0 for r0, r1 in ranges) if hi > lo else any(r0 <= lo < r1 for r0, r1 in ranges)


def token_groups(pr: Pair, t: dict, other_object: str | None) -> dict[str, list[int]]:
    """Positions of `pr.a` in each group of `GROUPS`. Prefix positions belong to no group (they never differ)."""
    e = pr.a
    reg = char_regions(e, t, other_object)
    lo, hi = pr.span_a
    groups: dict[str, list[int]] = {g: [] for g in GROUPS}
    groups["span"] = list(range(lo, hi)) if pr.equal_span else []
    final_from = next((i for i in range(len(e)) if e.offsets[i][1] > e.offsets[i][0] and e.offsets[i][0] >= e.user_end),
                      len(e))
    prev = "task_rest"
    for i in range(hi, len(e)):
        span = e.offsets[i]
        if i >= final_from:
            g = "final"
        elif span[1] <= span[0]:  # a token with no characters (some special tokens) joins the one before it
            g = prev
        elif _inside(span, reg["query"]):
            g = "query"
        elif _inside(span, reg["action"]):
            g = "action"
        elif _inside(span, reg["task"]):
            g = "task_rest"
        elif _inside(span, reg["hist"]):
            g = next((h for h in ("hist_obj", "hist_swap", "hist_rec") if _inside(span, reg[h])), "hist_other")
        else:  # separators between regions ("\n\n") go with the region they precede
            g = "query" if span[0] >= reg["action"][0][1] else "action"
        groups[g].append(i)
        prev = g
    groups["hist_all"] = sorted(groups["hist_obj"] + groups["hist_swap"] + groups["hist_rec"] + groups["hist_other"])
    groups["last"] = [len(e) - 1]
    return groups


def action_positions(e: Encoded, t: dict) -> dict[str, int]:
    """Probe positions in one prompt: the answer position, the action line's last token, its receptacle's first token."""
    reg = char_regions(e, t, None)
    a0, a1 = reg["action"][0]
    in_action = [i for i, s in enumerate(e.offsets) if s[1] > s[0] and a0 <= s[0] < a1]
    rec = _receptacle(t["action"])
    rec_pos = in_action[-1]
    if rec:
        r0 = e.text.rfind(rec, a0, a1)
        if r0 >= 0:
            hits = [i for i in in_action if e.offsets[i][0] < r0 + len(rec) and e.offsets[i][1] > r0]
            rec_pos = hits[0] if hits else rec_pos
    return {"answer": len(e) - 1, "action_end": in_action[-1], "action_rec": rec_pos}


def build_pair(tok: Any, t: dict, fmt: str, goal_a: str = "aligned", goal_b: str = "swapped",
               placebo_object: str | None = None) -> Pair:
    """The tokenised pair (target `a`, source `b`) with token groups on `a`."""
    a = encode(tok, messages(t, fmt, goal_a, placebo_object))
    b = encode(tok, messages(t, fmt, goal_b, placebo_object))
    pr = pair(a, b)
    if not pr.span_a[0] >= 0 or pr.span_a[1] > len(a):
        raise ValueError("bad span")
    # The span must sit inside the task line: anything else means the prompts differ elsewhere.
    task = char_regions(a, t, None)["task"][0]
    lo, hi = pr.span_a
    if hi > lo and not (a.offsets[lo][0] >= task[0] and a.offsets[hi - 1][1] <= task[1] + 1):
        raise ValueError(f"{t['id']}: the prompts differ outside the task line")
    other = task_object(t, goal_b, placebo_object)
    pr.groups = token_groups(pr, t, other)
    return pr


def placebo_object(t: dict, game_items: list[dict]) -> str | None:
    """A second non-goal object for the placebo pair (design §3), or None.

    Drawn from the swapped objects other items of the same game received (so it passed the same feasibility
    checks for the same task), excluding the item's own swap, anything named in its action or acted on in its
    history, and anything at the action's receptacle at reset. The choice is the first in sorted order after a
    hash of the item id, so it is fixed.
    """
    import hashlib

    from wf.data.goals import interacted_objects, names_object

    pool = {s["swapped_goal_object"] for s in game_items
            if s.get("swapped_goal_object") and s.get("task_line") == t.get("task_line")}
    rec = _receptacle(t["action"])
    at_rec = {re.sub(r" \d+$", "", inst) for inst, places in (t.get("scene_locations") or {}).items()
              if rec and rec in places}
    bad = {t.get("goal_object"), t.get("swapped_goal_object")} | interacted_objects(t["history_text"]) | at_rec
    options = sorted(o for o in pool if o not in bad and not names_object(t["action"], o))
    if not options:
        return None
    h = int(hashlib.sha256(f"interp|placebo|{t['id']}".encode()).hexdigest()[:8], 16)
    return options[h % len(options)]
