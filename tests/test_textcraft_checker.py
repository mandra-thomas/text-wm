"""Text-only checker (wf/textcraft/checker.py): known answers worked by hand, independence, and the committed items."""
import ast
import json
from pathlib import Path

import pytest

from wf.textcraft import checker as K

REPO = Path(__file__).resolve().parents[1]
LINES = ["craft 4 stick using 2 planks", "craft 4 oak planks using 1 oak logs", "craft 1 minecart using 5 iron ingot",
         "craft 3 ladder using 7 stick"]
USED_UP = [["get 1 oak logs", "Got 1 oak logs"],
           ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"],
           ["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"],
           ["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"]]


def test_checker_imports_no_environment_code():
    tree = ast.parse((REPO / "wf" / "textcraft" / "checker.py").read_text())
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods <= {"__future__", "re", "dataclasses"}


def test_accounting():
    led = K.account(USED_UP + [["inventory", "Inventory: [stick] (8) "]])
    assert led.held == {"stick": 8} and led.peak == {"oak logs": 1, "oak planks": 4, "stick": 8}


def test_used_up():
    v = K.decide(LINES, USED_UP, "craft 4 stick using 2 oak planks")
    assert v.determined_failure and v.failure_class == "used_up"
    assert v.reply == "Could not find enough items to craft minecraft:stick"
    assert v.short_inputs == [{"name": "oak planks", "need": 2, "held": 0, "peak": 4, "kind": "used_up"}]
    assert v.twin == "Crafted 4 minecraft:stick" and v.topup == {"oak planks": 2}


def test_too_few_tops_up_only_the_missing_amount():
    v = K.decide(LINES, [["get 2 iron ingot", "Got 2 iron ingot"]], "craft 1 minecart using 5 iron ingot")
    assert v.failure_class == "too_few" and v.topup == {"iron ingot": 3}
    assert v.reply == "Could not find enough items to craft minecraft:minecart"


def test_too_few_when_obtained_used_and_never_enough():
    hist = [["get 1 oak logs", "Got 1 oak logs"], ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"],
            ["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"]]
    v = K.decide(LINES, hist, "craft 3 ladder using 7 stick")
    assert v.failure_class == "too_few" and v.short_inputs[0] == {"name": "stick", "need": 7, "held": 4, "peak": 4, "kind": "too_few"}


def test_never_obtained_and_class_precedence():
    hist = [["get 2 iron ingot", "Got 2 iron ingot"]]
    assert K.decide(LINES, hist, "craft 4 oak planks using 1 oak logs").failure_class == "never_obtained"
    # one input used up, another never obtained: used up wins
    lines = LINES + ["craft 1 fence using 2 oak planks, 1 iron ingot"]
    v = K.decide(lines, USED_UP, "craft 1 fence using 2 oak planks, 1 iron ingot")
    assert v.failure_class == "used_up" and [s["kind"] for s in v.short_inputs] == ["used_up", "never_obtained"]
    assert v.topup == {"oak planks": 2, "iron ingot": 1}


def test_held_inputs_are_not_a_determined_failure():
    assert not K.decide(LINES, USED_UP[:2], "craft 4 stick using 2 oak planks").determined_failure


def test_get_rule_uses_the_displayed_outputs():
    v = K.decide(LINES, [], "get 1 stick")
    assert v.determined_failure and v.reply == "Could not find stick" and v.twin == "Got 1 stick"
    assert v.failure_class == "get_craftable"
    assert not K.decide(LINES, [], "get 1 oak logs").determined_failure


def test_inconsistent_inventory_reply_is_an_error():
    with pytest.raises(K.CheckerError):
        K.account([["get 1 oak logs", "Got 1 oak logs"], ["inventory", "Inventory: [oak logs] (2) "]])
    with pytest.raises(K.CheckerError):
        K.account([["inventory", "Inventory: [oak logs] (1)"]])


def test_unknown_reply_and_impossible_craft_are_errors():
    with pytest.raises(K.CheckerError):
        K.account([["look", "You see a tree."]])
    with pytest.raises(K.CheckerError):
        K.account([["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"]])


def test_failed_history_actions_change_nothing():
    led = K.account([["get 1 stick", "Could not find stick"], ["get 1 oak logs", "Got 1 oak logs"]])
    assert led.held == {"oak logs": 1}


def test_check_item_reports_each_disagreement():
    item = {"recipe_lines": LINES, "history": USED_UP, "action": "craft 4 stick using 2 oak planks", "action_kind": "craft",
            "reply": "Could not find enough items to craft minecraft:stick", "failure_class": "used_up",
            "twin": "Crafted 4 minecraft:stick", "topup": {"oak planks": 2},
            "short_inputs": [{"name": "oak planks", "need": 2, "held": 0, "peak": 4, "kind": "used_up"}],
            "pure_class": True, "needs_arithmetic": True, "goal": "ladder", "partner": "minecart",
            "history_fit": "aligned", "history_fit_counts": {"aligned": 4, "swapped": 0}}
    assert K.check_item(item) == []
    assert K.check_item({**item, "needs_arithmetic": False})
    assert K.check_item({**item, "pure_class": False})
    assert K.check_item({**item, "history_fit": "swapped"})
    assert K.check_item({**item, "failure_class": "too_few"})
    assert K.check_item({**item, "topup": {"oak planks": 4}})
    assert K.check_item({**item, "reply": "Could not find enough items to craft stick"})


def test_checker_agrees_with_every_committed_item():
    items = [json.loads(line) for line in (REPO / "wf/textcraft/data/items.jsonl").read_text().splitlines()]
    assert len(items) == 2000
    assert [it["id"] for it in items if K.check_item(it)] == []
