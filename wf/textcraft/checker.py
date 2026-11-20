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

    def add(self, name: str, n: int) -> None:
        self.held[name] = self.held.get(name, 0) + n
        self.peak[name] = max(self.peak.get(name, 0), self.held[name])

    def remove(self, name: str, n: int) -> None:
        left = self.held.get(name, 0) - n
        if left < 0:
            raise CheckerError(f"history crafts with {n} {name} but holds {self.held.get(name, 0)}")
        if left:
            self.held[name] = left
        else:
            self.held.pop(name, None)


def account(history: list[list[str]] | list[tuple[str, str]]) -> Ledger:
    """Replay the history's replies on an empty inventory (rules in design section 2)."""
    ledger = Ledger()
    for action, reply in history:
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


CLASS_ORDER = ("used_up", "too_few", "never_obtained")


def decide(recipe_lines: list[str], history: list, action: str) -> Verdict:
    """Is the action a sure failure given only the visible text, and with which reply, class and twin?"""
    ledger = account(history)
    get = GET.fullmatch(action)
    if get:
        n, name = int(get.group(1)), get.group(2)
        if name in displayed_outputs(recipe_lines):
            return Verdict(True, reply=f"Could not find {name}", failure_class="get_craftable",
                           twin=f"Got {n} {name}")
        return Verdict(False)
    count, target, inputs = parse_craft(action)
    short = []
    for c, name in inputs:
        held, peak = ledger.held.get(name, 0), ledger.peak.get(name, 0)
        if held < c:
            kind = "used_up" if peak >= c else "too_few" if peak > 0 else "never_obtained"
            short.append({"name": name, "need": c, "held": held, "peak": peak, "kind": kind})
    if not short:
        return Verdict(False)
    kinds = {s["kind"] for s in short}
    cls = next(k for k in CLASS_ORDER if k in kinds)
    return Verdict(True, reply=f"Could not find enough items to craft {to_id(target)}", failure_class=cls,
                   short_inputs=short, twin=f"Crafted {count} {to_id(target)}",
                   topup={s["name"]: s["need"] - s["held"] for s in short})


def check_item(item: dict) -> list[str]:
    """Every way the stored item disagrees with the checker; empty when it agrees."""
    try:
        v = decide(item["recipe_lines"], item["history"], item["action"])
    except CheckerError as e:
        return [f"checker error: {e}"]
    problems = []
    if not v.determined_failure:
        return ["not a determined failure"]
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
    return problems
