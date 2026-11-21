"""The text heuristics of notes/textcraft_llama_competence.md (definitions fixed at commit 9a83738, before any fit),
for round 2's descriptive readouts (notes/textcraft_r2_design.md, section 9): H1 (every input's name mentioned in
the history), H2 (gross units obtained reach each input's count, consumption ignored), H6 (the true inventory rule),
H9 (the action's output is not a top item of the displayed list) and H2 and H9. Ported line for line from
`features` in scripts/textcraft_llama_competence.py, without numpy; a test checks that the two agree on every item.
"""
from __future__ import annotations

import re
from collections import defaultdict

from wf.textcraft.checker import GENERIC_MEMBERS, account, from_id, parse_craft

HEURISTICS = ("H1", "H2", "H6", "H2andH9")


def _mentions(action: str, reply: str) -> set[str]:
    out: set[str] = set()
    if action.startswith("get "):
        out.add(action.split(" ", 2)[2])
    elif action.startswith("craft "):
        _k, y, inputs = parse_craft(action)
        out.add(y)
        out |= {n for _c, n in inputs}
    if reply.startswith("Got "):
        out.add(reply.split(" ", 2)[2])
    elif reply.startswith("Crafted "):
        out.add(from_id(reply.split(" ", 2)[2]))
    elif reply.startswith("Inventory: ") and "not carrying" not in reply:
        out |= set(re.findall(r"\[([^\]]+)\] \(", reply))
    return out


def features(item: dict) -> dict[str, int]:
    """Each heuristic's prediction (1 = success) for an item, from the prompt text only."""
    hist = [tuple(h) for h in item["history"]]
    _k, y, inputs = parse_craft(item["action"])
    ledger = account(hist)
    gained: dict[str, int] = defaultdict(int)
    mentioned: set[str] = set()
    for a, r in hist:
        mentioned |= _mentions(a, r)
        if r.startswith("Got "):
            nn, x = r.split(" ", 2)[1:]
            gained[x] += int(nn)
        elif r.startswith("Crafted "):
            kk, x = r.split(" ", 2)[1:]
            gained[from_id(x)] += int(kk)
    h = {n: ledger.held.get(n, 0) for _c, n in inputs}
    consumed_names = set()
    for _kk, _out, ins in (parse_craft(line) for line in item["recipe_lines"]):
        for _c, n in ins:
            consumed_names.add(n)
            consumed_names |= set(GENERIC_MEMBERS.get(n, ()))
    f = {"H1": int(all(n in mentioned for _c, n in inputs)),
         "H2": int(all(gained[n] >= c for c, n in inputs)),
         "H6": int(all(h[n] >= c for c, n in inputs)),
         "H9": int(y in consumed_names)}
    f["H2andH9"] = int(f["H2"] and f["H9"])
    return f
