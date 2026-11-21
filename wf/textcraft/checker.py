"""Text-only determination checker for TextCraft items (notes/textcraft_design.md, sections 2 and 3).

It reads only what a world model is shown: the recipe lines, the history's (action, reply) pairs and the action.
It must never import the environment (wf/textcraft/upstream, wf/textcraft/world); a test enforces this, so that
agreement between the checker and the environment is evidence, not an echo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CRAFT = re.compile(r"craft (.*) using (.*)")
GET = re.compile(r"get ([0-9]+) (.*)")
COUNTED = re.compile(r"([0-9]+) (.*)")
GOT = re.compile(r"Got ([0-9]+) (.*)")
CRAFTED = re.compile(r"Crafted ([0-9]+) minecraft:(\S+)")
INVENTORY_ENTRY = re.compile(r"\[([^\]]+)\] \(([0-9]+)\) ")


class CheckerError(ValueError):
    """The text breaks a rule the checker relies on (an unknown reply, an inconsistent inventory)."""


def from_id(name: str) -> str:
    """"minecraft:oak_planks" or "oak_planks" -> "oak planks"."""
    return name.removeprefix("minecraft:").replace("_", " ")


def to_id(name: str) -> str:
    return "minecraft:" + name.replace(" ", "_")


def parse_craft(action: str) -> tuple[int, str, list[tuple[int, str]]]:
    """(output count, output name, [(count, input name)]) of a craft action or recipe line."""
    m = CRAFT.match(action)
    if not m:
        raise CheckerError(f"not a craft: {action!r}")
    out = COUNTED.match(m.group(1))
    count, target = (int(out.group(1)), out.group(2)) if out else (1, m.group(1))
    inputs = []
    for part in m.group(2).split(","):
        p = COUNTED.match(part.strip())
        if not p:
            raise CheckerError(f"input without a count: {part!r}")
        inputs.append((int(p.group(1)), p.group(2)))
    return count, target, inputs


def displayed_outputs(recipe_lines: list[str]) -> set[str]:
    return {parse_craft(line)[1] for line in recipe_lines}


@dataclass
class Ledger:
    """The inventory the history implies, and each item's peak holding."""

    held: dict[str, int] = field(default_factory=dict)
    peak: dict[str, int] = field(default_factory=dict)
    changed_at: dict[str, int] = field(default_factory=dict)  # the last history step (from 1) that changed the count
    peak_at: dict[str, int] = field(default_factory=dict)  # the first history step at which the peak was reached
    step: int = 0

    def add(self, name: str, n: int) -> None:
        self.held[name] = self.held.get(name, 0) + n
        self.changed_at[name] = self.step
        if self.held[name] > self.peak.get(name, 0):
            self.peak[name], self.peak_at[name] = self.held[name], self.step

    def remove(self, name: str, n: int) -> None:
        left = self.held.get(name, 0) - n
        if left < 0:
            raise CheckerError(f"history crafts with {n} {name} but holds {self.held.get(name, 0)}")
        if n:
            self.changed_at[name] = self.step
        if left:
            self.held[name] = left
        else:
            self.held.pop(name, None)


def account(history: list[list[str]] | list[tuple[str, str]]) -> Ledger:
    """Replay the history's replies on an empty inventory (rules in design section 2)."""
    ledger = Ledger()
    for action, reply in history:
        ledger.step += 1
        got, crafted = GOT.fullmatch(reply), CRAFTED.fullmatch(reply)
        if got:
            ledger.add(got.group(2), int(got.group(1)))
        elif crafted:
            _count, _target, inputs = parse_craft(action)
            for c, n in inputs:
                ledger.remove(n, c)
            ledger.add(from_id(crafted.group(2)), int(crafted.group(1)))
        elif reply.startswith("Inventory: "):
            shown = _inventory(reply)
            if shown != {k: v for k, v in ledger.held.items() if v}:
                raise CheckerError(f"inventory reply {reply!r} disagrees with the accounting {ledger.held}")
        elif reply.startswith("Could not"):
            continue  # a failed action changes nothing
        else:
            raise CheckerError(f"unknown reply {reply!r}")
    return ledger


def _inventory(reply: str) -> dict[str, int]:
    body = reply[len("Inventory: "):]
    if body == "You are not carrying anything.":
        return {}
    entries = INVENTORY_ENTRY.findall(body)
    if "".join(f"[{n}] ({c}) " for n, c in entries) != body:
        raise CheckerError(f"unreadable inventory reply {reply!r}")
    return {n: int(c) for n, c in entries}


@dataclass
class Verdict:
    determined_failure: bool
    reply: str | None = None
    failure_class: str | None = None
    short_inputs: list[dict] = field(default_factory=list)
    twin: str | None = None
    topup: dict[str, int] = field(default_factory=dict)
    certificate: list[dict] = field(default_factory=list)
    determined_success: bool = False
    removal: dict[str, int] = field(default_factory=dict)


CLASS_ORDER = ("used_up", "too_few", "never_obtained")


def decide(recipe_lines: list[str], history: list, action: str) -> Verdict:
    """Is the action a sure failure given only the visible text, and with which reply, class and twin?"""
    ledger = account(history)
    get = GET.fullmatch(action)
    if get:
        n, name = int(get.group(1)), get.group(2)
        line = next((k for k, text in enumerate(recipe_lines, 1) if parse_craft(text)[1] == name), None)
        if line is not None:  # the object is the output of a displayed line, so it is craftable and cannot be got
            return Verdict(True, reply=f"Could not find {name}", failure_class="get_craftable",
                           twin=f"Got {n} {name}", certificate=[{"name": name, "line": line}])
        return Verdict(False)
    count, target, inputs = parse_craft(action)
    short = []
    for c, name in inputs:
        held, peak = ledger.held.get(name, 0), ledger.peak.get(name, 0)
        if held < c:
            kind = "used_up" if peak >= c else "too_few" if peak > 0 else "never_obtained"
            short.append({"name": name, "need": c, "held": held, "peak": peak, "kind": kind})
    if not short:
        # Every input is held. The craft is a sure success when it is a displayed line with its generic inputs
        # (planks, wool) filled by members (section 2, rule 8 of section 11, A1); its failure twin empties the inputs.
        if not any(matches_line(line, count, target, inputs) for line in recipe_lines):
            return Verdict(False)
        cert = [{"name": n, "held_step": ledger.changed_at.get(n, "never"), "peak_step": ledger.peak_at.get(n, "never")}
                for _c, n in inputs]
        return Verdict(False, reply=f"Crafted {count} {to_id(target)}", failure_class="success",
                       twin=f"Could not find enough items to craft {to_id(target)}", certificate=cert,
                       determined_success=True, removal={n: ledger.held[n] for _c, n in inputs})
    kinds = {s["kind"] for s in short}
    cls = next(k for k in CLASS_ORDER if k in kinds)
    # The certificate: for each short input, the step whose reply set the count held now and the step at which it
    # first reached its peak ("never" when no step did): with the need, these settle the failure and its class.
    cert = [{"name": s["name"], "held_step": ledger.changed_at.get(s["name"], "never"),
             "peak_step": ledger.peak_at.get(s["name"], "never")} for s in short]
    return Verdict(True, reply=f"Could not find enough items to craft {to_id(target)}", failure_class=cls,
                   short_inputs=short, twin=f"Crafted {count} {to_id(target)}",
                   topup={s["name"]: s["need"] - s["held"] for s in short}, certificate=cert)


def matches_line(line: str, count: int, target: str, inputs: list[tuple[int, str]]) -> bool:
    """The action crafts this displayed line: same output and count, and its inputs are the line's, in any order,
    with each generic input of the line filled by one of its members and the counts equal."""
    k, out, wanted = parse_craft(line)
    if (k, out) != (count, target) or len(wanted) != len(inputs):
        return False
    left = list(inputs)
    for c, n in sorted(wanted, key=lambda x: x[1] in GENERIC_MEMBERS):  # exact names first, then generic groups
        hit = next((x for x in left if x[0] == c and (x[1] == n or x[1] in GENERIC_MEMBERS.get(n, ()))), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def inventory_shown_after_last_change(history: list) -> bool:
    """An "Inventory:" reply comes after the last reply that got or crafted something (or there is no such reply):
    the inventory before the action is shown verbatim (design section 12, B6)."""
    shown = False
    for _action, reply in history:
        if GOT.fullmatch(reply) or CRAFTED.fullmatch(reply):
            shown = False
        elif reply.startswith("Inventory: "):
            shown = True
    return shown


# The only generic groups that appear as inputs on recipe lines, with their members (design section 2 and 11, A1).
# White wool is not a member: its recipe carries no "wool" group.
GENERIC_MEMBERS = {
    "planks": ("acacia planks", "birch planks", "crimson planks", "dark oak planks", "jungle planks", "oak planks",
               "spruce planks", "warped planks"),
    "wool": ("black wool", "blue wool", "brown wool", "cyan wool", "gray wool", "green wool", "light blue wool",
             "light gray wool", "lime wool", "magenta wool", "orange wool", "pink wool", "purple wool", "red wool",
             "yellow wool"),
}


def tree_names(recipe_lines: list[str], goal: str) -> set[str]:
    """Names on the displayed lines of the goal's tree: the lines for the goal, then for each of their inputs
    (a generic group's lines are those of its members), and so on; their outputs and inputs (design 11, A3)."""
    parsed = [parse_craft(line) for line in recipe_lines]
    seen, todo, names = set(), [goal], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        wanted = {name} | set(GENERIC_MEMBERS.get(name, ()))
        for _k, out, inputs in parsed:
            if out in wanted:
                names.add(out)
                names |= {n for _c, n in inputs}
                todo += [out] + [n for _c, n in inputs]
    return names


def step_object(reply: str) -> str | None:
    got, crafted = GOT.fullmatch(reply), CRAFTED.fullmatch(reply)
    return got.group(2) if got else from_id(crafted.group(2)) if crafted else None


def history_fit(recipe_lines: list[str], history: list, goal: str, partner: str) -> tuple[str, dict[str, int]]:
    """"aligned" when more history steps produce a name found only in the goal's tree than only in the partner's."""
    a, p = tree_names(recipe_lines, goal), tree_names(recipe_lines, partner)
    objs = [step_object(o) for _a, o in history]
    n = {"aligned": sum(o in a - p for o in objs), "swapped": sum(o in p - a for o in objs)}
    return ("aligned" if n["aligned"] > n["swapped"] else "swapped" if n["swapped"] > n["aligned"] else "tie"), n


def check_item(item: dict) -> list[str]:
    """Every way the stored item disagrees with the checker; empty when it agrees."""
    try:
        v = decide(item["recipe_lines"], item["history"], item["action"])
    except CheckerError as e:
        return [f"checker error: {e}"]
    problems = []
    if item["failure_class"] == "success":
        if not v.determined_success:
            return ["not a determined success"]
        problems += [f"{k} {getattr(v, k)!r} != stored {item[k]!r}" for k in ("reply", "twin", "certificate", "removal")
                     if getattr(v, k) != item[k]]
        if item["short_inputs"] or item["topup"]:
            problems.append("a success has short inputs or a top-up")
    elif not v.determined_failure:
        return ["not a determined failure"]
    else:
        problems += _failure_problems(item, v)
    if item.get("inventory_shown_after_last_change") is not inventory_shown_after_last_change(item["history"]):
        problems.append("inventory_shown_after_last_change disagrees")
    fit = history_fit(item["recipe_lines"], item["history"], item["goal"], item["partner"])
    if [item.get("history_fit"), item.get("history_fit_counts")] != [fit[0], fit[1]]:
        problems.append(f"history fit {fit} != stored {item.get('history_fit')} {item.get('history_fit_counts')}")
    return problems


def _failure_problems(item: dict, v: Verdict) -> list[str]:
    problems = []
    if v.reply != item["reply"]:
        problems.append(f"reply {v.reply!r} != stored {item['reply']!r}")
    if v.failure_class != item["failure_class"]:
        problems.append(f"class {v.failure_class} != stored {item['failure_class']}")
    if v.twin != item["twin"]:
        problems.append(f"twin {v.twin!r} != stored {item['twin']!r}")
    if item["action_kind"] == "craft":
        if v.short_inputs != item["short_inputs"]:
            problems.append(f"short inputs {v.short_inputs} != stored {item['short_inputs']}")
        if v.topup != item["topup"]:
            problems.append(f"top-up {v.topup} != stored {item['topup']}")
        if item.get("pure_class") != (len({s["kind"] for s in v.short_inputs}) == 1):
            problems.append("pure_class disagrees")
        if item.get("needs_arithmetic") != all(s["peak"] > 0 for s in v.short_inputs):
            problems.append("needs_arithmetic disagrees")
    if item.get("certificate") != v.certificate:
        problems.append(f"certificate {v.certificate} != stored {item.get('certificate')}")
    if item.get("removal"):
        problems.append("a failure has a removal")
    return problems
