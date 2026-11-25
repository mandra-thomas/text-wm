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
# aligned/swapped/absent: the scored arms. placebo: a second non-goal object. aligned_plus/swapped_plus: the goal
# line followed by a sentence naming the other goal's object (rq1's mention arms). cf_aligned/cf_swapped: rq1v's
# counterfactual task (the real task's destination replaced), so neither task of the pair is the real one.
GOALS = ("aligned", "swapped", "absent", "placebo", "aligned_plus", "swapped_plus", "cf_aligned", "cf_swapped")
READOUT = {"yn": ("Yes", "No"), "obs": ("You", "Nothing")}
# Finest groups first; history subgroups are disjoint and "hist_all" is their union. "hist_obj_matched" is a
# control, not part of the partition: as many history tokens as "hist_obj" has, drawn (seeded) from the history's
# tokens that name neither object, so it sits in the same region (amendment 1, A7).
# "action_obj" (the action's object and its instance number in the action line) and "action_rest" (the rest of
# the action line) split "action" (amendment 2, B7): an action share carried by the object's name and one carried
# by the verb and receptacle mean different things.
GROUPS = ("span", "task_rest", "hist_obj", "hist_swap", "hist_rec", "hist_other", "hist_all", "hist_obj_matched",
          "action", "action_obj", "action_rest", "query", "final", "last")
COARSE_GROUPS = ("span", "task_rest", "hist_all", "action", "query", "final", "last")
# The minimal pilot's residual grid (amendment 1, section M; amendment 2, B7): the coarse groups, the object-mention
# history tokens reported separately from the rest of the history with their position-matched control, the history
# tokens naming the swapped object, and the action line split into its object and the rest.
PILOT_GROUPS = ("span", "task_rest", "hist_obj", "hist_obj_matched", "hist_swap", "hist_all", "action", "action_obj",
                "action_rest", "query", "final", "last")
HIST_SUB = ("hist_obj", "hist_swap", "hist_rec", "hist_other")


def _to_yn(t: dict, msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """The observation prompt turned into rq1v's yes/no prompt (the same rule as `rq1v.arm_messages`)."""
    from wf import rq1v

    last = msgs[-1]
    if not last["content"].endswith("Next observation:"):
        raise ValueError(f"{t['id']}: prompt does not end with 'Next observation:'")
    return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + rq1v.YN_QUESTION}]


def messages(t: dict, fmt: str, goal: str, placebo_object: str | None = None) -> list[dict[str, str]]:
    """The item's prompt in format `fmt` ("yn" or "obs") under `goal`."""
    from wf import rq1, rq1v
    from wf.scoring.logodds import build_messages

    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}")
    if goal == "placebo":
        if not placebo_object:
            raise ValueError(f"{t['id']}: the placebo goal needs a placebo object")
        line = make_swapped_task_line(t["task_line"], t["goal_object"], placebo_object)
        msgs = build_messages(t, "aligned", task_line=line)
    elif goal in ("cf_aligned", "cf_swapped"):
        msgs = rq1v.arm_messages(t, goal)
        if msgs is None:
            raise ValueError(f"{t['id']}: no counterfactual task can be built for this item")
    elif goal in ("aligned", "swapped", "absent") and fmt == "yn":
        return rq1v.arm_messages(t, f"yn_{goal}")  # the scored yes/no prompt itself
    else:
        msgs = rq1.arm_messages(t, goal)
    return msgs if fmt == "obs" else _to_yn(t, msgs)


def task_object(t: dict, goal: str, placebo_object: str | None = None) -> str | None:
    """The object the goal line is about (for a plus line, the goal's own object, not the one mentioned)."""
    own = {"aligned": t.get("goal_object"), "swapped": t.get("swapped_goal_object"), "absent": None,
           "placebo": placebo_object}
    return own[goal.split("_")[-1] if goal.startswith("cf_") else goal.replace("_plus", "")]


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


def char_diff(x: str, y: str) -> tuple[int, int, int]:
    """(start, end in x, end in y) of the characters where two texts differ, widened to whole words.

    Two object names can share an ending ("glassbottle" and "winebottle") or a start; matching only token ids
    would put the shared sub-word tokens ("b", "ottle") outside the span and into the task line's rest. The
    region is therefore widened to word boundaries, so the whole name belongs to the span.
    """
    n = min(len(x), len(y))
    lo = 0
    while lo < n and x[lo] == y[lo]:
        lo += 1
    s = 0
    while s < n - lo and x[-1 - s] == y[-1 - s]:
        s += 1
    hx, hy = len(x) - s, len(y) - s
    while lo > 0 and x[lo - 1].isalnum():  # x[:lo] == y[:lo], so the same holds in y
        lo -= 1
    while hx < len(x) and x[hx].isalnum():  # x[hx:] == y[hy:]
        hx += 1
        hy += 1
    return lo, hx, hy


def pair(a: Encoded, b: Encoded) -> Pair:
    """The shared token prefix and suffix of two prompts, cut back so that no token of either prompt that
    overlaps the differing characters (widened to whole words, `char_diff`) is counted as shared."""
    n = min(len(a), len(b))
    p = 0
    while p < n and a.ids[p] == b.ids[p]:
        p += 1
    s = 0
    while s < n - p and a.ids[-1 - s] == b.ids[-1 - s]:
        s += 1
    if p == len(a) == len(b):
        raise ValueError("the two prompts are identical")
    lo, ha, hb = char_diff(a.text, b.text)
    wide = lambda o: o[1] > o[0]  # noqa: E731  (tokens with no characters never overlap anything)
    while p > 0 and any(wide(e.offsets[p - 1]) and e.offsets[p - 1][1] > lo for e in (a, b)):
        p -= 1
    while s > 0 and ((wide(a.offsets[len(a) - s]) and a.offsets[len(a) - s][0] < ha)
                     or (wide(b.offsets[len(b) - s]) and b.offsets[len(b) - s][0] < hb)):
        s -= 1
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
    regions["action_obj"] = []
    obj = _action_object(t["action"])
    if obj:
        a0, a1 = regions["action"][0]
        m = re.compile(rf"\b{re.escape(obj)}\b(?: \d+)?").search(text, a0 + len("Action: "), a1)
        if m:
            regions["action_obj"] = [(m.start(), m.end())]
    return regions


def mentions_action_object(t: dict) -> bool:
    """Whether the history after the task line names the action's object (the same word and rule as the
    `hist_obj` group, read from the item's history text, so items can be stratified before tokenising)."""
    w = _action_object(t["action"]) or t.get("goal_object")
    h = t.get("history_text") or ""
    m = re.search(r"^Your task is to:.*$", h, flags=re.MULTILINE)
    if not w or not m:
        return False
    return re.search(rf"\b{re.escape(w)}\b", h[m.end():]) is not None


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
    groups["action_obj"] = [i for i in groups["action"]
                            if e.offsets[i][1] > e.offsets[i][0] and _inside(e.offsets[i], reg["action_obj"])]
    in_obj = set(groups["action_obj"])
    groups["action_rest"] = [i for i in groups["action"] if i not in in_obj]
    groups["hist_all"] = sorted(groups["hist_obj"] + groups["hist_swap"] + groups["hist_rec"] + groups["hist_other"])
    groups["hist_obj_matched"] = matched_positions(groups["hist_obj"], groups["hist_other"] + groups["hist_rec"],
                                                   f"interp|matched|{t['id']}")
    groups["last"] = [len(e) - 1]
    return groups


def matched_positions(target: list[int], pool: list[int], seed_text: str) -> list[int]:
    """As many positions as `target` has, drawn without replacement from `pool` by a seeded draw (all of the pool
    if it is smaller); sorted."""
    import random

    pool = sorted(set(pool) - set(target))
    k = min(len(target), len(pool))
    return sorted(random.Random(seed_text).sample(pool, k)) if k else []


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
