"""Model-written predictions (2026-11-17): failure wording counts as a failure, and the audit tool."""
import csv
import json

import pytest

from wf.analysis.stats import asymmetry_A, gen_says_success
from wf.scoring.parser import parse_observation, parse_prediction
from wf.tools import parser_audit


@pytest.mark.parametrize("text, cls", [
    ("Nothing happens.", "noop"),
    ("You can't find a tomato 1 in the fridge 1.", "noop_paraphrase"),
    ("You cannot take the spraybottle 1 from the drawer 1 because it is closed.", "noop_paraphrase"),
    ("You try to clean the pot 1 with the sinkbasin 1, but the pot 1 is not dirty.", "noop_paraphrase"),
    ("There is no tomato 1 in the environment.", "noop_paraphrase"),
    ("You don't see a tomato 1 on the diningtable 1.", "noop_paraphrase"),
    ("You have no cup in your inventory.", "noop_paraphrase"),
    ("You are already holding the handtowel 1.", "noop_paraphrase"),
    ("You already have the pencil 1 in your inventory.", "noop_paraphrase"),
    ("You pick up the mug 1 from the cabinet 1.", "state_change"),
    ("You open the drawer 1. The drawer 1 is open. In it, you see nothing.", "reveal_without_goal"),
    ("You arrive at countertop 1. On the countertop 1, you see a mug 1.", "reveal_with_goal"),
    ("You are not carrying anything.", "info"),
    ("You place the candle 1 on the cabinet 1.", "success_paraphrase"),
    ("You wash the potato 1 in the sinkbasin 1.", "success_paraphrase"),
    ("You try to clean the apple 1 with the sinkbasin 1.", "unparseable"),
    ("The mug is probably somewhere else.", "unparseable"),
    ("", "unparseable"),
])
def test_parse_prediction(text, cls):
    assert parse_prediction(text, "mug") == cls


def test_environment_parser_is_unchanged():
    assert parse_observation("You can't find a tomato 1 in the fridge 1.") == "unparseable"
    assert parse_observation("You place the candle 1 on the cabinet 1.") == "unparseable"


def test_failure_wording_counts_as_failure_and_enters_A():
    assert gen_says_success("noop_paraphrase") is False
    assert gen_says_success("success_paraphrase") is True
    items = [{"id": "f", "game_id": "g", "context_determined": True, "true_obs": "Nothing happens."},
             {"id": "s", "game_id": "g", "context_determined": True, "true_obs": "You pick up the mug 1."}]
    gens = [{"transition_id": "f", "condition": "aligned", "generated_class": "noop_paraphrase"},
            {"transition_id": "s", "condition": "aligned", "generated_class": "state_change"}]
    out = asymmetry_A(gens, items)
    assert out["n_unparseable"] == 0 and out["n_fail"] == 1 and out["false_success_rate"] == 0.0


def _gens(tmp_path):
    # (action, reply) templates the current parser reads as each kind; the export re-parses them
    kinds = {
        "no_effect": ("take mug {i} from shelf 1", "You can't find the mug {i}."),
        "success_take": ("take mug {i} from shelf 1", "You pick up the mug {i} from the shelf 1."),
        "success_clean": ("clean mug {i} with sinkbasin 1", "You clean the mug {i} using the sinkbasin 1."),
        "process": ("clean mug {i} with sinkbasin 1", "You submerge the mug {i} in the sinkbasin 1."),
        "walk": ("take mug {i} from shelf 1", "You arrive at shelf {i}."),
        "unsure": ("take mug {i} from shelf 1", "Maybe the mug {i}."),
        "noop": ("take mug {i} from shelf 1", "Nothing happens."),
    }
    gens, trans = [], []
    for model in parser_audit.FEWSHOT_MODELS:
        for kind, (action, text) in kinds.items():
            for i in range(1, 61):
                tid = f"{model}-{kind}-{i}"
                trans.append({"id": tid, "action": action.format(i=i), "goal_object": "mug"})
                gens.append({"model": model, "transition_id": tid, "condition": "aligned", "generated_class": "stale",
                             "generated": text.format(i=i)})
    g, t = tmp_path / "g.jsonl", tmp_path / "t.jsonl"
    g.write_text("\n".join(json.dumps(x) for x in gens))
    t.write_text("\n".join(json.dumps(x) for x in trans))
    return g, t


def _fill(path, label_for):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        r[3] = label_for(r)
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _export(tmp_path, **kw):
    g, t = _gens(tmp_path)
    out = tmp_path / "audit.csv"
    key = parser_audit.export(g, t, out, **kw)
    return out, {json.loads(x)["row"]: json.loads(x) for x in key.read_text().splitlines()}


def test_audit_export_is_blind_and_scores(tmp_path):
    out, keyed = _export(tmp_path)
    header = out.read_text().splitlines()[0]
    assert "parser" not in header and "true" not in header  # the reader sees neither answer
    assert "N/R/P/D/?" in header
    assert len(keyed) == sum(parser_audit.STRATA.values())
    _fill(out, lambda r: keyed[int(r[0])]["parser_label"])
    result = parser_audit.score(out)
    assert result["passed"] and result["agreement_four_way"] == 1.0


def test_audit_counts_effect_kinds_as_one_side(tmp_path):
    out, keyed = _export(tmp_path)
    # the reader calls every process-only row a requested success: same side, so the primary rule still passes
    _fill(out, lambda r: "R" if keyed[int(r[0])]["parser_label"] == "P" else keyed[int(r[0])]["parser_label"])
    result = parser_audit.score(out)
    assert result["passed"] and result["agreement_four_way"] < 1.0


def test_audit_fails_on_any_opposite_reading(tmp_path):
    out, keyed = _export(tmp_path)
    flip = next(r for r, k in keyed.items() if k["parser_label"] == "N")
    _fill(out, lambda r: "R" if int(r[0]) == flip else keyed[int(r[0])]["parser_label"])
    result = parser_audit.score(out)
    assert not result["passed"] and result["opposite_rows"] == [flip]


def test_audit_parser_unsure_counts_as_a_miss_and_reader_unsure_is_excluded(tmp_path):
    out, keyed = _export(tmp_path)
    # reader decides every row the parser left "?": all misses, so agreement falls below the bar
    _fill(out, lambda r: "N" if keyed[int(r[0])]["parser_label"] == "?" else keyed[int(r[0])]["parser_label"])
    assert not parser_audit.score(out)["passed"]
    # reader "?" rows are left out of the criterion
    _fill(out, lambda r: keyed[int(r[0])]["parser_label"])
    assert parser_audit.score(out)["n_reader_unsure"] == parser_audit.STRATA["unparseable"]


def test_audit_fails_with_unlabelled_rows(tmp_path):
    out, _ = _export(tmp_path)
    assert not parser_audit.score(out)["passed"]


def test_audit_covers_clean_heat_cool_and_llama_walk_there(tmp_path):
    _, keyed = _export(tmp_path)
    strata = [k["stratum"] for k in keyed.values()]
    assert strata.count("success_clean_heat_cool") == parser_audit.STRATA["success_clean_heat_cool"]
    assert strata.count("different_effect_llama") == parser_audit.STRATA["different_effect_llama"]


def test_audit_export_can_exclude_rows_already_seen(tmp_path):
    g, t = _gens(tmp_path)
    first = parser_audit.export(g, t, tmp_path / "a.csv")
    second = parser_audit.export(g, t, tmp_path / "b.csv", exclude=first)
    seen = {json.loads(x)["text"] for x in first.read_text().splitlines()}
    again = {json.loads(x)["text"] for x in second.read_text().splitlines()}
    assert not (seen & again)


def test_reparse_recomputes_pilot_a_from_saved_text(monkeypatch, tmp_path):
    import yaml
    from pathlib import Path
    import wf.runs as runs_mod
    from wf.util import write_jsonl

    for key in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE", "WF_MOCK_CONTINUE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    monkeypatch.setattr(runs_mod, "git_sha", lambda: "fffffff")
    monkeypatch.setattr(runs_mod, "git_dirty", lambda: False)
    score_dir = tmp_path / "pilotA_score"
    score_dir.mkdir()
    (score_dir / "STATUS.md").write_text("# STATUS — pilotA_score — REPORTED (rule output)\n\nbody\n")
    (score_dir / "git_sha").write_text("28a396d0000\n")
    (score_dir / "inputs.json").write_text(json.dumps({"inputs": {}, "git_dirty": False}))

    trans, gens, scores = [], [], []
    for i in range(6):
        tid = f"t{i}"
        trans.append({"id": tid, "game_id": f"train/{i % 3}", "context_determined": True, "true_obs": "Nothing happens.",
                      "candidates": {"plus": "You pick up the mug 1.", "minus": "Nothing happens."},
                      "swapped_goal_object": "cup", "targets_goal_object": True, "history_goal_agnostic": True,
                      "goal_object": "mug", "action": "take mug 1 from shelf 1", "history_text": "", "history_len": 0})
        text = "You can't find the mug 1." if i % 2 else "You pick up the mug 1 from the shelf 1."
        for cond in ("aligned", "swapped"):
            gens.append({"model": "llama3.1-8b-fewshot", "transition_id": tid, "condition": cond, "game_id": f"train/{i % 3}",
                         "generated": text, "generated_class": parse_observation(text, "mug")})
            scores.append({"model": "llama3.1-8b-fewshot", "transition_id": tid, "condition": cond, "L": -1.0 if i % 2 else 1.0,
                           "game_id": f"train/{i % 3}"})
    (tmp_path / "pilotA_transitions").mkdir()
    write_jsonl(tmp_path / "pilotA_transitions" / "transitions.jsonl", trans)
    write_jsonl(score_dir / "generations.jsonl", gens)
    write_jsonl(score_dir / "scoring.jsonl", scores)

    cfg = yaml.safe_load((Path(runs_mod.__file__).parent / "config" / "pilotA_reparse.yaml").read_text())
    assert runs_mod.run_pilotA_reparse(cfg) == 0
    m = json.loads((tmp_path / "pilotA_reparse" / "metrics.json").read_text())["per_model"]["llama3.1-8b-fewshot"]
    assert m["before"]["A"]["n_unparseable"] == 3 and m["before"]["A"]["false_success_rate"] == 1.0
    assert m["after"]["A"]["n_unparseable"] == 0 and m["after"]["A"]["false_success_rate"] == 0.5
    assert m["after"]["scoring_agreement_failures"]["agreement"] == 1.0
    saved = [json.loads(x) for x in (tmp_path / "pilotA_reparse" / "generations.jsonl").read_text().splitlines()]
    assert {g["generated_class_v1"] for g in saved} == {"unparseable", "state_change"}
    assert "REPORTED (parser v2 re-read" in (tmp_path / "pilotA_reparse" / "STATUS.md").read_text().splitlines()[0]


def test_audit_tops_up_from_the_other_model(tmp_path):
    g, t = _gens(tmp_path)
    gens = [json.loads(x) for x in g.read_text().splitlines()]
    gens = [x for x in gens if not (x["model"] == "qwen3-8b-fewshot" and "-process-" in x["transition_id"])]
    g.write_text("\n".join(json.dumps(x) for x in gens))
    key = parser_audit.export(g, t, tmp_path / "audit.csv")
    rows = [json.loads(x) for x in key.read_text().splitlines()]
    assert sum(r["stratum"] == "process_only" for r in rows) == parser_audit.STRATA["process_only"]


@pytest.mark.parametrize("action, text, cls", [
    # examine: any description of the object is the action working
    ("examine soapbar 1", "You examine the soapbar 1.", "info"),
    ("examine pillow 1", "The pillow 1 is soft and fluffy.", "info"),
    # failures worded in new ways
    ("heat potato 1 with microwave 1", "The action is not possible. You do not have a potato 1.", "noop_paraphrase"),
    ("take bread 1 from garbagecan 1", "You look around but do not see a bread 1.", "noop_paraphrase"),
    ("cool mug 1 with fridge 1", "You try to cool the mug 1 with the fridge 1. The mug 1 is not in your inventory.",
     "noop_paraphrase"),
    ("heat potato 1 with microwave 1", "The microwave 1 is not hot. You need to heat it up first.", "noop_paraphrase"),
    # successes stated as a result
    ("cool egg 1 with fridge 1", "The egg 1 is now cool.", "success_paraphrase"),
    # a different action is not a success of this one
    ("heat cup 1 with microwave 1", "You arrive at the microwave 1. On the microwave 1, you see a cup 1.", "different_effect"),
    ("clean potato 1 with sinkbasin 1", "You go to the sinkbasin 1. The sinkbasin 1 is empty.", "different_effect"),
    ("clean potato 1 with sinkbasin 1", "You wash your hands with the sinkbasin 1.", "different_effect"),
    ("heat apple 1 with microwave 1", "The apple 1 is in the microwave 1. The microwave 1 is closed.", "noop_paraphrase"),
    # walking there and then doing it counts
    ("clean potato 1 with sinkbasin 1",
     "You go to the sinkbasin 1. The sinkbasin 1 is empty. You clean the potato 1 with the sinkbasin 1.", "state_change"),
    # go to keeps its own templates
    ("go to drawer 1", "You arrive at drawer 1. The drawer 1 is closed.", "state_change"),
    # unchanged: the environment's own success text
    ("take mug 1 from cabinet 1", "You pick up the mug 1 from the cabinet 1.", "state_change"),
])
def test_parse_prediction_knows_the_action(action, text, cls):
    assert parse_prediction(text, "mug", action) == cls


@pytest.mark.parametrize("action, text, cls", [
    # no effect, in the decided equivalents
    ("heat potato 1 with microwave 1", "You attempt to heat the potato 1 with the microwave 1. The microwave 1 is closed.",
     "noop_paraphrase"),
    ("heat apple 1 with microwave 1", "The microwave 1 is closed. You must open the microwave 1 before you can heat anything.",
     "noop_paraphrase"),
    ("clean apple 1 with sinkbasin 1", "You try to clean the apple 1 with the sinkbasin 1. The apple 1 is already clean.",
     "noop_paraphrase"),
    # process only: an effect, result not stated
    ("clean plate 1 with sinkbasin 1", "You submerge the plate 1 in the sinkbasin 1.", "process_only"),
    ("cool potato 1 with fridge 1", "You put the potato 1 in the fridge 1.", "process_only"),
    ("clean pot 1 with sinkbasin 1", "You submerge the pot 1 in the sinkbasin 1. The pot 1 is now wet.", "process_only"),
    # requested success with the result stated
    ("cool potato 1 with fridge 1", "You cool the potato 1 using the fridge 1.", "state_change"),
    ("clean kettle 1 with sinkbasin 1", "You rinse the kettle 1 with the water from the sinkbasin 1.", "success_paraphrase"),
    # different effect: walking there instead of the action
    ("take mug 1 from cabinet 3", "You arrive at cabinet 3. The cabinet 3 is closed.", "different_effect"),
    ("heat cup 1 with microwave 1", "You arrive at the microwave 1.", "different_effect"),
])
def test_effect_categories(action, text, cls):
    assert parse_prediction(text, None, action) == cls


def test_partial_effects_count_as_effects_with_two_sensitivity_rules():
    from wf.analysis.stats import with_partial_rule
    gens = [{"generated_class": c} for c in ("process_only", "different_effect", "noop", "state_change")]
    assert [gen_says_success(g["generated_class"]) for g in gens] == [True, True, False, True]
    excl = [gen_says_success(g["generated_class"]) for g in with_partial_rule(gens, "exclude_partial")]
    fail = [gen_says_success(g["generated_class"]) for g in with_partial_rule(gens, "partial_as_failure")]
    assert excl == [None, None, False, True] and fail == [False, False, False, True]


# Findings of the adversarial review and of reading real replies (2026-11-17): each was wrong before.
@pytest.mark.parametrize("action, text, side", [
    ("take mug 1 from cabinet 1", "Nothing happened.", False),
    ("take mug 1 from cabinet 1", "The cabinet 1 is empty.", False),
    ("cool potato 1 with fridge 1", "The potato 1 is not in the world.", False),
    ("clean kettle 1 with sinkbasin 1", "You see nothing.", False),
    ("take mug 1 from cabinet 1", "You open the cabinet 1. You see no mug 1 inside.", False),
    ("take mug 1 from cabinet 1", "You open the cabinet 1.\nYou do not see a mug 1 inside.", False),
    ("move mug 1 to cabinet 1", "You're not holding the mug 1.", False),
    ("move mug 1 to cabinet 1", "You are not carrying anything.", False),
    ("take mug 1 from cabinet 1", "You can’t take the mug 1.", False),
    ("open drawer 1", "The drawer 1 is already open.", False),
    ("move handtowel 1 to toilet 1", "You have already moved the handtowel 1 to the toilet 1.", False),
    ("go to sinkbasin 1", "You are already at sinkbasin 1.", False),
    ("heat apple 1 with microwave 1", "The apple 1 is in the microwave 1. The microwave 1 is closed.", False),
    # effects: an arrival or opening that mentions something absent is still the action working
    ("go to countertop 1", "You arrive at countertop 1. There is no mug 1 on the countertop 1.", True),
    ("open fridge 1", "You open the fridge 1. There is no tomato in it.", True),
    ("inventory", "You are not carrying anything.", True),
    ("take mug 1 from countertop 1", "You pick up the mug 1 from the countertop 1. It has already been cleaned.", True),
    ("take mug 1 from drawer 1", "The drawer 1 is closed. You open it and pick up the mug 1.", True),
    ("take tissuebox 1 from drawer 4",
     "You open the drawer 4. In it, you see a tissuebox 1. You pick up the tissuebox 1 from the drawer 4.", True),
    ("take mug 1 from cabinet 1", "Observation: You pick up the mug 1 from the cabinet 1.", True),
    ("take mug 1 from cabinet 1", "You now have the mug 1.", True),
    ("go to cabinet 1", "You walk to the cabinet 1.", True),
    ("clean tomato 1 with sinkbasin 1", "You have cleaned the tomato 1 with the sinkbasin 1.", True),
])
def test_reviewed_cases_land_on_the_right_side(action, text, side):
    assert gen_says_success(parse_prediction(text, None, action)) is side


@pytest.mark.parametrize("action, text, cls", [
    ("cool mug 1 with fridge 1", "You put the mug 1 in the fridge 1 to cool.", "process_only"),
    ("cool mug 1 with fridge 1", "You put the cool mug 1 in the fridge 1.", "process_only"),
    ("heat cup 1 with microwave 1", "You put the cup 1 in the microwave 1. The microwave 1 starts to heat the cup 1.",
     "process_only"),
    ("clean pot 1 with sinkbasin 1", "You start to clean the pot 1 with the sinkbasin 1.", "process_only"),
    ("clean ladle 1 with sinkbasin 1", "You see a ladle 1 on the countertop 1.", "different_effect"),
    ("heat mug 1 with microwave 1", "You heat the mug 1 using the stoveburner 1.", "different_effect"),
    ("move mug 1 to cabinet 1", "You move the mug 1 to the countertop 1.", "different_effect"),
    ("move pot 1 to stoveburner 1", "You pick up the pot 1 from the cabinet 1.", "different_effect"),
    ("take mug 2 from cabinet 1", "You pick up the mug 1 from the cabinet 1.", "different_effect"),
    ("examine cabinet 4", "You arrive at cabinet 4. The cabinet 4 is closed.", "different_effect"),
    # third audit (2026-11-18): holding or carrying the object after a take says the take worked
    ("take knife 1 from drawer 3", "You are holding the knife 1.", "success_paraphrase"),
    ("take cd 2 from desk 1", "You are carrying the cd 2.", "success_paraphrase"),
    ("take potato 1 from countertop 2", "You are carrying: a lettuce 1, a potato 1.", "success_paraphrase"),
    ("take cd 2 from desk 1", "You are carrying the mug 1.", "different_effect"),
    ("take cd 2 from desk 1", "You are already holding the cd 2.", "noop_paraphrase"),
    ("cool tomato 1 with fridge 1",
     "You try to cool the tomato 1 with the fridge 1. The tomato 1 is already at room temperature.", "noop_paraphrase"),
    ("clean egg 1 with sinkbasin 1", "You try to clean the egg 1 with the sinkbasin 1. The egg 1 is not dirty.",
     "noop_paraphrase"),
    ("heat bread 1 with microwave 1", "The action 'heat bread 1 with microwave 1' is not valid.", "noop_paraphrase"),
])
def test_reviewed_cases_land_in_the_right_category(action, text, cls):
    assert parse_prediction(text, None, action) == cls
