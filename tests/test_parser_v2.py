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
    texts = {"noop_paraphrase": "You can't find the mug {i}.", "success_paraphrase": "You place the mug {i} on the shelf 1.",
             "unparseable": "Maybe the mug {i}.", "state_change": "You pick up the mug {i} from the shelf 1.",
             "noop": "Nothing happens."}
    gens, trans = [], []
    for model in parser_audit.FEWSHOT_MODELS:
        for cls, tmpl in texts.items():
            for i in range(40):
                tid = f"{model}-{cls}-{i}"
                trans.append({"id": tid, "action": f"take mug {i} from shelf 1"})
                gens.append({"model": model, "transition_id": tid, "condition": "aligned", "generated_class": cls,
                             "generated": tmpl.format(i=i) + (f" ({model} {i})" if cls == "noop" else "")})
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


def test_audit_export_is_blind_and_scores(tmp_path):
    g, t = _gens(tmp_path)
    out = tmp_path / "audit.csv"
    key = parser_audit.export(g, t, out)
    header = out.read_text().splitlines()[0]
    assert "parser" not in header and "true" not in header  # the reader sees neither answer
    keyed = {json.loads(x)["row"]: json.loads(x) for x in key.read_text().splitlines()}
    assert len(keyed) == sum(parser_audit.STRATA.values())
    _fill(out, lambda r: keyed[int(r[0])]["parser_label"] if keyed[int(r[0])]["parser_label"] != "?" else "?")
    assert parser_audit.score(out)["passed"]


def test_audit_fails_when_failure_wording_is_read_as_success(tmp_path):
    g, t = _gens(tmp_path)
    out = tmp_path / "audit.csv"
    key = parser_audit.export(g, t, out)
    keyed = {json.loads(x)["row"]: json.loads(x) for x in key.read_text().splitlines()}
    # the reader says S on one failure-worded row the parser calls F
    flip = next(r for r, k in keyed.items() if k["parser_class"] == "noop_paraphrase")
    _fill(out, lambda r: "S" if int(r[0]) == flip else keyed[int(r[0])]["parser_label"])
    result = parser_audit.score(out)
    assert not result["passed"] and result["failure_wording_read_as_success"] == [flip]


def test_audit_fails_with_unlabelled_rows(tmp_path):
    g, t = _gens(tmp_path)
    out = tmp_path / "audit.csv"
    parser_audit.export(g, t, out)
    assert not parser_audit.score(out)["passed"]


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
