"""TextCraft round 2 (notes/textcraft_r2_design.md): the clean prompt, the arms, the records and their checks, the
tokenization gate on a stand-in tokenizer, the scoring run in mock mode, and the analysis on planted scores."""
from __future__ import annotations

import copy
import math
import json
import re
import types
from collections import Counter

import pytest

from wf.runs import RUNNERS
from wf.textcraft import dx
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r2_analysis as A
from wf.textcraft import r2_items as I
from wf.textcraft import r2_score as S
from wf.textcraft import r2_tokcheck as T
from wf.textcraft import world as W
from wf.textcraft.verify import sha256

SHOTS = P.fewshot_examples()


@pytest.fixture(scope="module")
def records():
    return I.read_records()


@pytest.fixture(scope="module")
def small(records):
    """Every record of eight pairs with parent records and eight with padded goal and ingredient records, and of the
    pair whose goal name contains "and" (no two-item line)."""
    parent_pairs = sorted({r["pair_id"] for r in records if r["kind"] == "parent"})[:8]
    roles: dict[str, set] = {}
    for r in records:
        if r["kind"] == "main" and "pad8_aligned" in r["arms"]:
            roles.setdefault(r["pair_id"], set()).add(r["role"])
    both_roles = [p for p in sorted(roles) if roles[p] == {"goal", "ingredient"}][:8]
    odd = {r["pair_id"] for r in records if r["both_task_line"] is None and r["kind"] == "main"}
    keep = set(parent_pairs) | set(both_roles) | odd
    return [r for r in records if r["pair_id"] in keep]


def _first(records, **kv):
    return next(r for r in records if all(r.get(k) == v for k, v in kv.items()))


# ---------------- the clean prompt ----------------


def test_clean_system_text_is_the_diagnostics_text_and_has_no_agent_sentence():
    assert P.CLEAN_SYSTEM == dx.CLEAN_SYSTEM == R.SYSTEM
    assert P.CLEAN_SYSTEM.startswith("You are a text world model for TextCraft")
    for agent in ("Every round", "you have to", "from the inventory or the environment", "Thought:", "Reminder",
                  "The agent was given"):
        assert agent not in P.CLEAN_SYSTEM
    assert "empty" not in P.CLEAN_SYSTEM  # the empty start is not stated
    assert P.SYSTEM != P.CLEAN_SYSTEM  # the confirmatory prompt is unchanged


def test_clean_system_text_hash_is_pinned():
    """The text the diagnostics scored (dx run 4) and the design quotes; any edit fails here."""
    import hashlib
    assert hashlib.sha256(P.CLEAN_SYSTEM.encode()).hexdigest() == "540d6ab7e17088f08e8a1d04bc4d1abb6a58f7cb5b58560636058c6c625a5a51"


# ---------------- arms and prompts ----------------


def test_parse_arm():
    assert R.parse_arm("aligned") == (False, 0, False, "aligned")
    assert R.parse_arm("yn_both") == (True, 0, False, "both")
    assert R.parse_arm("pad8_swapped") == (False, 8, False, "swapped")
    assert R.parse_arm("ext_parent") == (False, 0, True, "parent")
    for bad in ("pad3_aligned", "ext_banana", "yn_", "partner"):
        with pytest.raises(ValueError):
            R.parse_arm(bad)


def test_two_item_line_is_sorted_and_shared_by_a_couple(records):
    main = [r for r in records if r["kind"] == "main"]
    by_couple = {}
    for r in main:
        a, b = sorted((r["goal"], r["partner"]))
        if " and " in f" {a} " or " and " in f" {b} ":
            assert r["both_task_line"] is None and not set(R.BOTH_ARMS) & set(r["arms"])
        else:
            assert r["both_task_line"] == f"Goal: craft {a} and {b}." and set(R.BOTH_ARMS) <= set(r["arms"])
        by_couple.setdefault(r["couple_id"], set()).add(r["both_task_line"])
    assert all(len(v) == 1 for v in by_couple.values())
    assert R.two_item_line("melon", "apple") == R.two_item_line("apple", "melon") == "Goal: craft apple and melon."
    assert R.two_item_line("flint and steel", "pink dye") is None
    assert sum(r["both_task_line"] is None for r in main) == 10


def test_every_arm_changes_only_what_its_name_says(small):
    assert all(I.prompt_problems(r, SHOTS) == [] for r in small)
    rec = next(r for r in small if r["kind"] == "main" and r["population"] == "main" and r["length_bin"] == "1-3"
               and r["both_task_line"])
    m = {a: R.messages(rec, a, SHOTS) for a in rec["arms"]}
    assert m["aligned"][-1]["content"].replace(rec["task_line"], rec["both_task_line"]) == m["both"][-1]["content"]
    assert m["pad8_aligned"][-1]["content"].count("> inventory\nInventory: You are not carrying anything.") >= 8
    assert m["yn_aligned"][-1]["content"].endswith(R.YN_QUESTION) and P.YN_QUESTION not in m["yn_aligned"][-1]["content"]
    assert m["aligned"][0]["content"] == P.CLEAN_SYSTEM


def test_padding_goes_before_the_first_step(small):
    rec = _first(small, kind="main", population="main", length_bin="1-3")
    lines, hist, goal = R.shown(rec, "pad8_aligned")
    assert hist[:8] == [list(R.PAD_STEP)] * 8 and hist[8:] == rec["history"] and goal == rec["task_line"]
    assert R.shown(rec, "pad1_swapped")[1][:1] == [list(R.PAD_STEP)] and R.shown(rec, "pad1_swapped")[2] == rec["swapped_task_line"]


def test_prompt_check_catches_a_changed_arm(small, monkeypatch):
    rec = _first(small, kind="parent")
    orig = R.shown

    def leaky(r, arm):
        lines, hist, goal = orig(r, arm)
        return (lines[:-1] if arm == "ext_parent" else lines), hist, goal
    monkeypatch.setattr(R, "shown", leaky)
    assert any("ext_parent" in p for p in I.prompt_problems(rec, SHOTS))


def test_gemma_folds_the_system_text_and_the_others_do_not(small):
    rec = small[0]
    g = R.model_messages(R.MODELS["gemma2-9b-it"], rec, "aligned", SHOTS)
    assert g[0]["role"] == "user" and g[0]["content"].startswith(P.CLEAN_SYSTEM + "\n\n")
    assert all(m["role"] != "system" for m in g)
    for tag in ("qwen3-8b", "llama3.1-8b", "mistral-7b-v0.3", "llama3.2-3b"):
        assert R.model_messages(R.MODELS[tag], rec, "aligned", SHOTS)[0] == {"role": "system", "content": P.CLEAN_SYSTEM}


def test_models_are_pinned_with_wave_a_handling():
    from wf.constants import REVISIONS
    assert list(R.MODELS) == ["qwen3-8b", "llama3.1-8b", "gemma2-9b-it", "mistral-7b-v0.3", "llama3.2-3b"]
    assert all(R.MODELS[t].hf_id in REVISIONS for t in R.MODELS)
    assert R.MODELS["gemma2-9b-it"].system == "fold" and R.MODELS["gemma2-9b-it"].attn_implementation == "eager"
    assert R.MODELS["llama3.2-3b"].template_kwargs == {"date_string": "26 Jul 2024"}
    assert R.MODELS["mistral-7b-v0.3"].system == "native"


def test_item_hash_covers_fields_and_context(small, monkeypatch):
    rec = _first(small, kind="parent")
    h = R.item_hash(rec, SHOTS)
    for field, value in (("ext_recipe_lines", rec["ext_recipe_lines"][:-1]), ("parent_task_line", "Goal: craft x."),
                         ("both_task_line", "Goal: craft x and y."), ("arms", rec["arms"][:-1]), ("history", []),
                         ("candidates", {"o_plus": "x", "o_minus": "y"})):
        assert R.item_hash({**rec, field: value}, SHOTS) != h, field
    assert R.item_hash(rec, SHOTS[:2]) != h
    monkeypatch.setattr(R, "SYSTEM", R.SYSTEM + " ")
    assert R.item_hash(rec, SHOTS) != h
    monkeypatch.setattr(R, "SYSTEM", P.CLEAN_SYSTEM)
    monkeypatch.setattr(R, "PAD_STEP", ("inventory", "Inventory: "))
    assert R.item_hash(rec, SHOTS) != h


# ---------------- records ----------------


def test_record_counts_and_arms(records):
    kinds = Counter(r["kind"] for r in records)
    assert kinds == {"main": 2100, "parent": 882}
    assert sum(len(r["arms"]) for r in records) == 25743  # amended before freeze: + yn_neutral and ext_both on parents
    padded = [r for r in records if "pad8_aligned" in r["arms"]]
    assert len(padded) == 920 and all(r["population"] == "main" and r["length_bin"] in ("1-3", "4-8") for r in padded)
    parents = [r for r in records if r["kind"] == "parent"]
    assert all(r["arms"] == [a for a in R.PARENT_ARMS if r["both_task_line"] or a != "ext_both"] for r in parents)
    assert sum("ext_both" in r["arms"] for r in parents) == 873 and all("yn_neutral" in r["arms"] for r in parents)


def test_main_records_are_the_frozen_items_unchanged(records):
    frozen = {t["id"]: t for t in (json.loads(x) for x in I.B.ITEMS.read_text().splitlines() if x.strip())}
    for r in records:
        if r["kind"] == "main":
            assert {k: v for k, v in r.items() if k not in ("kind", "both_task_line", "arms")} == frozen[r["id"]]
    assert sum(t["population"] in ("main", "success") for t in frozen.values()) == 2100


def test_structure_check_catches_changes(small):
    frozen = {t["id"]: t for t in (json.loads(x) for x in I.B.ITEMS.read_text().splitlines() if x.strip())}
    main = _first(small, kind="main")
    parent = _first(small, kind="parent")
    assert I.structure_problems(main, frozen) == [] and I.structure_problems(parent, frozen) == []
    bad = [({**main, "reply": main["reply"] + "x"}, "differs from its frozen"),
           ({**main, "arms": main["arms"][:-1]}, "arms"),
           ({**main, "both_task_line": "Goal: craft a and b."}, "two-item"),
           ({**parent, "parent_task_line": "Goal: craft stick."}, "parent line"),
           ({**parent, "ext_recipe_lines": list(reversed(parent["ext_recipe_lines"]))}, "seeded insertion"),
           ({**parent, "added_lines": parent["added_lines"] + ["craft 1 x using 1 y"]}, "added lines"),
           ({**parent, "parent_rule": "strict" if parent["parent_rule"] != "strict" else "word_overlap"}, "rule's choice"),
           ({**parent, "parent_shares_word_with_goal": not parent["parent_shares_word_with_goal"]}, "word flags"),
           ({**parent, "id": main["id"]}, "reuses a frozen")]
    for rec, words in bad:
        assert any(words in p for p in I.structure_problems(rec, frozen)), words


def test_parent_rules(records):
    parents = [r for r in records if r["kind"] == "parent"]
    assert len({(r["pair_id"], r["goal"]) for r in parents}) == 118
    for r in parents[:40]:
        assert I.qualifies(r["goal"], r["partner"], r["parent"])
        assert r["goal"] in I.consumers()[r["parent"]] if False else r["parent"] in I.consumers()[r["goal"]]
        assert not (W.tree_craftables(W.item_id(r["parent"])) & W.tree_craftables(W.item_id(r["partner"])))
        assert [x for x in r["ext_recipe_lines"] if x not in set(r["added_lines"])] == r["recipe_lines"]
        assert I.text_parent_problems(r) == []
    strict = [r for r in parents if r["parent_rule"] == "strict"]
    assert len(strict) == 250 and not any(set(r["parent"].split()) & set(r["goal"].split()) for r in strict)
    assert not I.qualifies("stick", "flower pot", "stick")  # the goal itself
    some = parents[0]
    assert not I.qualifies(some["goal"], some["partner"], some["partner"])


def test_extended_list_is_seeded_and_keeps_order():
    lines = [f"craft 1 a{i} using 1 b" for i in range(6)]
    added = ["craft 1 z using 1 a0", "craft 2 y using 1 c"]
    one = I.extended_list(lines, added, "p001", "a0", "z")
    assert one == I.extended_list(lines, added, "p001", "a0", "z") and len(one) == 8
    assert [x for x in one if x not in added] == lines
    assert any(I.extended_list(lines, added, "p002", "a0", "z") != one for _ in [0])


def test_text_parent_check_catches_bad_parents(small):
    parent = _first(small, kind="parent")
    no_line = {**parent, "ext_recipe_lines": [x for x in parent["ext_recipe_lines"]
                                               if not x.startswith(f"craft ") or f" {parent['parent']} using" not in x]}
    assert any("lists the goal" in p for p in I.text_parent_problems(no_line))
    partner_line = next(x for x in parent["recipe_lines"] if I.K.parse_craft(x)[1] in I.K.tree_names(parent["recipe_lines"], parent["partner"]))
    out = I.K.parse_craft(partner_line)[1]
    leak = {**parent, "ext_recipe_lines": parent["ext_recipe_lines"] + [f"craft 1 {parent['parent']} using 1 {out}"]}
    assert any("shares craftables" in p for p in I.text_parent_problems(leak))
    if parent["parent_rule"] == "strict":
        named = {**parent, "parent": parent["goal"] + " block"}
        assert any("shares a word" in p or "lists the goal" in p for p in I.text_parent_problems(named))


def test_replay_of_parents_and_padding(small):
    for r in [x for x in small if x["kind"] == "parent"][:6] + [x for x in small if "pad8_aligned" in x["arms"]][:6]:
        assert I.replay_problems(r) == [], r["id"]
        assert I.checker_problems(r) == [], r["id"]


def test_replay_catches_a_wrong_padding_reply(small, monkeypatch):
    rec = next(x for x in small if "pad8_aligned" in x["arms"])
    monkeypatch.setattr(R, "PAD_STEP", ("inventory", "Inventory: [stick] (1) "))
    assert any("history does not replay" in p for p in I.replay_problems(rec))
    assert any("padding" in p for p in I.checker_problems(rec))


def test_replies_do_not_depend_on_the_goal(small):
    """The two-item line's faithfulness: replies under goal G, P, the parent and no goal are the same."""
    rec = _first(small, kind="parent", population="main")
    acts = [a for a, _ in rec["history"]]
    outs = []
    for goal in (rec["goal"], rec["partner"], rec["parent"], None):
        env, hist = W.replay(rec["ext_recipe_lines"], W.item_id(goal) if goal else None, acts)
        outs.append((hist, W.step(W.clone(env), rec["action"])))
    assert all(o == outs[0] for o in outs) and outs[0][1] == rec["reply"]


def test_check_records_reports_and_passes(small):
    report = I.check_records(small, SHOTS)
    assert I.passes(report, len(small))
    assert not I.passes({**report, "prompt_failures": 1}, len(small))
    assert not I.passes({**report, "duplicate_ids": 1}, len(small))
    assert not I.passes(report, len(small) + 1)
    assert not I.passes({**report, "fewshot_goals_in_records": ["x"]}, len(small))


def test_committed_manifest_is_verified_and_hashes_the_records():
    m = json.loads(I.R2_MANIFEST.read_text())
    assert m["verified"] is True and m["records_sha256"] == sha256(I.R2_ITEMS)
    assert not isinstance(S.verified_files({}), str)


# ---------------- tokenization gate on a stand-in tokenizer ----------------


class FakeTok:
    """Word-level chat tokenizer stand-in: '<s>', turns '<u>' / '<a>' ... '<e>', words as ids."""

    def __init__(self, reply_prefix="", yes_pieces=1, same_first=False, no_system=False):
        self.no_system = no_system
        self.vocab = {"<s>": 0, "<e>": 1, "<u>": 2, "<a>": 3}
        self.all_special_ids = [0, 1, 2, 3]
        self.added_tokens_decoder = {}
        self.reply_prefix, self.yes_pieces, self.same_first = reply_prefix, yes_pieces, same_first

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def _words(self, text):
        out = []
        for w in re.findall(r"<[seua]>|[^\s<]+", text):
            if w == "Yes" and self.yes_pieces == 2:
                out += ["Y", "es"]
            elif w in ("Crafted", "Could") and self.same_first:
                out += ["C", w]
            else:
                out.append(w)
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return types.SimpleNamespace(input_ids=[self._id(w) for w in self._words(text)])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        if self.no_system and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")  # as Gemma 2's template
        text = "<s> " + " ".join(f"<{'a' if m['role'] == 'assistant' else 'u'}> "
                                 + (self.reply_prefix if m["role"] == "assistant" else "") + m["content"] + " <e>"
                                 for m in messages if m["role"] != "system") + (" <a>" if add_generation_prompt else "")
        return self(text).input_ids if tokenize else text

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def _fake_scorer(tok, tag="qwen3-8b"):
    from wf.scoring.logodds import LogoddsScorer
    spec = R.MODELS[tag]
    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=tag, template_kwargs=spec.template_kwargs)


def test_gate_passes_on_a_clean_tokenizer_and_counts_every_arm(small):
    recs = small[:30]
    rep = T.gate(_fake_scorer(FakeTok()), R.MODELS["qwen3-8b"], recs, SHOTS)
    assert rep["ok"], rep["problems"]
    n_two = sum(1 for r in recs for a in r["arms"] if not R.parse_arm(a)[0])
    n_yn = sum(1 for r in recs for a in r["arms"] if R.parse_arm(a)[0])
    assert rep["n_candidates_checked"] == 2 * n_two and rep["n_yes_no_prompts_checked"] == n_yn
    assert rep["yes_no"]["Yes"]["n_checked"] == n_yn


@pytest.mark.parametrize("tok,words", [(FakeTok(reply_prefix="▁ "), "vs template"), (FakeTok(yes_pieces=2), "not one token"),
                                       (FakeTok(same_first=True), "share their first token")])
def test_gate_fails_on_each_tokenization_problem(small, tok, words):
    rep = T.gate(_fake_scorer(tok), R.MODELS["qwen3-8b"], small[:10], SHOTS)
    assert not rep["ok"] and any(words in p for p in rep["problems"])


def test_gate_folds_for_gemma(small):
    rep = T.gate(_fake_scorer(FakeTok(no_system=True), "gemma2-9b-it"), R.MODELS["gemma2-9b-it"], small[:5], SHOTS)
    assert rep["ok"] and rep["system_handling"] == "fold"


def test_golden_cases_and_comparison(records, small):
    cases = T.golden_cases(records)
    assert [a for _i, a in cases] == ["aligned", "both", "yn_both", "pad8_aligned", "ext_parent"]
    scorer = _fake_scorer(FakeTok())
    report = {"tag": "qwen3-8b", "revision": "x",
              "golden": {f"{i}|{a}": T.render_one(scorer, R.MODELS["qwen3-8b"], next(r for r in records if r["id"] == i), a, SHOTS)
                         for i, a in cases}}
    golden = T.make_golden([report], records)
    assert T.golden_problems(scorer, "qwen3-8b", records, SHOTS, golden) == []
    golden2 = copy.deepcopy(golden)
    k = next(iter(golden2["models"]["qwen3-8b"]))
    if k != "revision":
        golden2["models"]["qwen3-8b"][k]["n_ids"] += 1
    else:
        k = list(golden2["models"]["qwen3-8b"])[1]
        golden2["models"]["qwen3-8b"][k]["n_ids"] += 1
    assert T.golden_problems(scorer, "qwen3-8b", records, SHOTS, golden2)
    assert T.golden_problems(scorer, "mistral-7b-v0.3", records, SHOTS, golden) == ["no golden rendering stored for mistral-7b-v0.3"]


def test_stored_golden_holds_every_message_in_order(records):
    if not T.GOLDEN.exists():
        pytest.skip("golden renderings not made yet (ICE CPU check)")
    golden = json.loads(T.GOLDEN.read_text())
    by_id = {r["id"]: r for r in records}
    assert set(golden["models"]) == set(R.TAGS)
    for tag, rend in golden["models"].items():
        for case in golden["cases"]:
            text = rend[f"{case['id']}|{case['arm']}"]["text"]
            pos = 0
            msgs = R.model_messages(R.MODELS[tag], by_id[case["id"]], case["arm"], SHOTS)
            for m in msgs:  # the turns in order; Mistral's template moves the system text into the last user turn
                if m["role"] == "system":
                    assert m["content"].split("\n")[-1] in text, (tag, case)
                    continue
                piece = m["content"].strip().split("\n")[-1]
                found = text.find(piece, pos)
                assert found >= 0, (tag, case)
                pos = found
            assert text.rstrip().endswith({"qwen3-8b": "<|im_start|>assistant\n<think>\n\n</think>",
                                           "gemma2-9b-it": "<start_of_turn>model",
                                           "mistral-7b-v0.3": "[/INST]"}.get(tag, "<|start_header_id|>assistant<|end_header_id|>")), tag
            if tag == "mistral-7b-v0.3":  # the system text sits in the last user turn, after the examples
                assert text.rfind(P.CLEAN_SYSTEM.split("\n")[-1]) > text.find(SHOTS[-1]["observation"])


# ---------------- scoring run, mock ----------------


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "FILED.txt").write_text("filed_date: x\n")
    return tmp_path


def _manifest(tmp_path, recs, verified=True, tamper=False):
    path = tmp_path / "r2_items.jsonl"
    path.write_text(I.dumps(recs))
    m = {"records_file": str(path), "records_sha256": sha256(path), "fewshot_file": str(P.FEWSHOT),
         "fewshot_sha256": sha256(P.FEWSHOT), "verified": verified, "why_not": [] if verified else ["x"]}
    if tamper:
        path.write_text(path.read_text().replace("Could not", "Could  not", 1))
    (tmp_path / "r2_manifest.json").write_text(json.dumps(m))
    return str(tmp_path / "r2_manifest.json")


def test_runners_and_configs_registered():
    for tag in R.TAGS:
        assert f"textcraft_r2_score_{tag}" in RUNNERS and f"textcraft_r2_smoke_{tag}" in RUNNERS
    assert "textcraft_r2_analysis" in RUNNERS


def test_score_refuses_unverified_or_changed_records(mock_runs, small):
    for kw in ({"verified": False}, {"tamper": True}):
        cfg = {"run_id": "textcraft_r2_score_qwen3-8b", "model": "qwen3-8b", "manifest": _manifest(mock_runs, small, **kw)}
        assert S.run_textcraft_r2_score(cfg) == 1
    with pytest.raises(SystemExit):
        S.run_textcraft_r2_score({"run_id": "x", "model": "gpt", "manifest": _manifest(mock_runs, small)})


def _score_all(mock_runs, recs, tags, n_shards=2):
    man = _manifest(mock_runs, recs)
    for tag in tags:
        for i in range(n_shards):
            cfg = {"run_id": f"textcraft_r2_score_{tag}", "model": tag, "manifest": man, "n_shards": n_shards, "shard": i}
            assert S.run_textcraft_r2_score(cfg) == 0
    return man


def test_scores_every_record_and_arm_then_resumes(mock_runs, small):
    _score_all(mock_runs, small, ["mistral-7b-v0.3"])
    scores = A.load_scores("mistral-7b-v0.3", mock_runs / "runs")
    assert len(scores) == sum(len(r["arms"]) for r in small)
    by, missing = A.index_scores(scores, small, SHOTS, "mistral-7b-v0.3")
    assert missing == []
    d = mock_runs / "runs" / "textcraft_r2_score_mistral-7b-v0.3" / "shard_00_of_02"
    assert json.loads((d / "tokenization_checks.json").read_text())["skipped"] == "mock"
    n0 = len((d / "scoring.jsonl").read_text().splitlines())
    cfg = {"run_id": "textcraft_r2_score_mistral-7b-v0.3", "model": "mistral-7b-v0.3",
           "manifest": str(mock_runs / "r2_manifest.json"), "n_shards": 2, "shard": 0, "resume": True}
    assert S.run_textcraft_r2_score(cfg) == 0
    assert len((d / "scoring.jsonl").read_text().splitlines()) == n0


def test_resume_refuses_scores_of_changed_records(mock_runs, small):
    _score_all(mock_runs, small, ["qwen3-8b"], n_shards=1)
    d = mock_runs / "runs" / "textcraft_r2_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl"
    lines = d.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["item_hash"] = "0" * 16
    d.write_text("\n".join([json.dumps(rec)] + lines[1:-3]) + "\n")
    cfg = {"run_id": "textcraft_r2_score_qwen3-8b", "model": "qwen3-8b", "manifest": str(mock_runs / "r2_manifest.json"),
           "n_shards": 1, "resume": True}
    with pytest.raises(SystemExit):
        S.run_textcraft_r2_score(cfg)


def test_gate_failure_stops_before_any_score(mock_runs, small, monkeypatch):
    man = _manifest(mock_runs, small)
    monkeypatch.setattr(S, "run_gate", lambda *a, **k: False)
    cfg = {"run_id": "textcraft_r2_score_qwen3-8b", "model": "qwen3-8b", "manifest": man, "n_shards": 1}
    assert S.run_textcraft_r2_score(cfg) == 1
    assert not (mock_runs / "runs" / "textcraft_r2_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl").exists()


def test_yes_no_arms_use_one_forward_pass(small, monkeypatch):
    calls = []

    class Sc:
        def score_one_token_pair(self, **kw):
            calls.append(("one", kw["condition"]))

        def score_pair(self, **kw):
            calls.append(("pair", kw["condition"]))
    rec = small[0]
    for arm in rec["arms"]:
        S.score_one(Sc(), "qwen3-8b", rec, arm, SHOTS)
    assert all((k == "one") == a.startswith("yn_") for k, a in calls)


def test_smoke_subset_is_seeded(records):
    a = S.smoke_subset(records, 20)
    assert len(a) == 20 and [r["id"] for r in a] == [r["id"] for r in S.smoke_subset(records, 20)]


# ---------------- analysis on planted scores ----------------

PLANT = {"lam": 0.7, "nu": 0.4, "kappa": -1.1, "tau": 0.5, "list": 0.2}


def _planted(recs, tag, lam=0.7, nu=0.4, kappa=-1.1, tau=0.5, lst=0.2, noise=0.0, seed=0):
    """L = base + lam [object in the line's goal tree] + nu [line names the object] + kappa(k) [the craft completes
    the line's task], with kappa(8) = kappa + tau under eight look-ups; yes/no the same plus a success offset."""
    import random
    rng = random.Random(seed)
    out = []
    for r in recs:
        noise0 = rng.gauss(0, 0.3)
        goal = r["role"] == "goal"
        for arm in r["arms"]:
            q, pad, ext, line = R.parse_arm(arm)
            # the yes/no question separates successes from failures; the two-candidate score does not
            base = ((3.0 if r["population"] == "success" else -3.0) if q else -3.0) + noise0
            rel = line in ("aligned", "both", "parent")
            named = goal and line in ("aligned", "both")
            done = goal and line == "aligned"
            k = kappa + (tau if pad == 8 else 0.0)
            val = base + lam * rel + nu * named + k * done + (lst if (r["kind"] == "parent" and arm == "aligned") else 0.0)
            val += -0.3 if (line == "neutral" and goal) else 0.0  # the neutral line sits apart from the swapped one, by role
            val += rng.gauss(0, noise) if noise else 0.0
            out.append({"model": tag, "transition_id": r["id"], "condition": arm, "L": val, "item_hash": R.item_hash(r, SHOTS),
                        "logp_plus": -0.05 if val > 0 else -3.0, "logp_minus": -3.0 if val > 0 else -0.05,
                        "logp_first_plus": -0.1, "logp_first_minus": -2.5})
    return out


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(A, "B_DRAWS", 300)
    monkeypatch.setattr(A, "N_FLIPS", 300)
    monkeypatch.setattr(A, "DESC_DRAWS", 50)
    monkeypatch.setattr(A, "AUC_DRAWS", 50)


def test_planted_effects_are_recovered_exactly(small, fast):
    recs = copy.deepcopy(small)
    scores = {t: _planted(recs, t) for t in R.TAGS}
    out = A.analyse(recs, SHOTS, scores, list(R.TAGS))
    c = out["per_model"]["qwen3-8b"]["contrasts"]
    assert c["lift"]["estimate"] == pytest.approx(0.7)
    assert c["parent_penalty"]["estimate"] == pytest.approx(0.4 - 1.1)
    assert c["naming"]["estimate"] == pytest.approx(0.4)
    assert c["completion"]["estimate"] == pytest.approx(-1.1)
    assert c["timing"]["estimate"] == pytest.approx(0.5)
    d = out["per_model"]["qwen3-8b"]["descriptive"]
    assert d["parent"]["list_control"]["estimate"] == pytest.approx(-0.2)
    assert d["parent"]["lift_as_intermediate"]["estimate"] == pytest.approx(0.7)
    assert d["padding"]["goal_eight_minus_one"]["estimate"] == pytest.approx(0.5)
    assert d["padding"]["ingredient_eight_minus_one"]["estimate"] == pytest.approx(0.0)


def test_naming_reading_when_naming_carries_the_penalty(small, fast):
    recs = copy.deepcopy(small)
    scores = {t: _planted(recs, t, lam=0.0, nu=-1.2, kappa=0.0, tau=0.0) for t in R.TAGS}
    out = A.analyse(recs, SHOTS, scores, list(R.TAGS))
    c = out["per_model"]["llama3.1-8b"]["contrasts"]
    assert c["naming"]["estimate"] == pytest.approx(-1.2) and c["completion"]["estimate"] == pytest.approx(0.0)
    assert c["parent_penalty"]["estimate"] == pytest.approx(-1.2)


def test_family_is_fixed_at_fifty_and_pads_missing(small, fast):
    recs = copy.deepcopy(small)
    out = A.analyse(recs, SHOTS, {"qwen3-8b": _planted(recs, "qwen3-8b")}, list(R.TAGS))
    assert out["family_size"] == 50
    assert all(v["p_in_family"] == 1.0 and v["padded"] for k, v in out["family"].items() if not k.startswith("qwen3-8b"))
    assert isinstance(out["verdicts"], str) and out["verdicts"].startswith("withheld")


def test_missing_scores_withhold_verdicts(small, fast):
    recs = copy.deepcopy(small)
    scores = {t: _planted(recs, t) for t in R.TAGS}
    scores["gemma2-9b-it"] = scores["gemma2-9b-it"][:-1]
    out = A.analyse(recs, SHOTS, scores, list(R.TAGS))
    assert out["missing_scores"]["gemma2-9b-it"] == 1 and out["verdicts"].startswith("withheld")


def test_stale_unknown_duplicate_or_foreign_scores_stop(small):
    recs = copy.deepcopy(small)
    good = _planted(recs, "qwen3-8b")
    for bad in ([{**good[0], "item_hash": "x"}] + good[1:], [{**good[0], "transition_id": "nope"}] + good[1:],
                good + good[:1], [{**good[0], "model": "llama3.1-8b"}] + good[1:]):
        with pytest.raises(A.StaleScore):
            A.index_scores(bad, recs, SHOTS, "qwen3-8b")


def test_competence_gate_uses_wave_a_rule(small, fast):
    recs = copy.deepcopy(small)
    pops = A.populations(recs)
    scores = _planted(recs, "qwen3-8b")
    by, _ = A.index_scores(scores, recs, SHOTS, "qwen3-8b")
    g = A.competence(by, pops)
    assert g["rule"] == {"auc_neutral_min": 0.70, "yes_share_neutral_failures_max": 0.95,
                         "yes_share_neutral_successes_min": 0.05, "median_yes_no_mass_neutral_min": 0.5}
    assert g["interpretable"] and g["checks"]["auc_neutral"]["value"] == pytest.approx(1.0)
    from wf import families
    assert A.COMPETENCE == families.COMPETENCE
    # a model that answers Yes to everything fails
    yes = [{**s, "L": abs(s["L"]) + 1} if s["condition"] == "yn_neutral" else s for s in scores]
    by2, _ = A.index_scores(yes, recs, SHOTS, "qwen3-8b")
    g2 = A.competence(by2, pops)
    assert not g2["interpretable"] and not g2["checks"]["yes_share_neutral_failures"]["pass"]
    # an AUC of 0.69 fails, 0.70 passes (the boundary)
    assert not (0.69 >= A.COMPETENCE["auc_neutral_min"]) and 0.70 >= A.COMPETENCE["auc_neutral_min"]
    assert "role_goal" in g["descriptive"]["yes_no"] and "role_ingredient" in g["descriptive"]["two_candidate"]


def test_statistics_helpers():
    assert A.bootstrap_p([1.0] * 99) == pytest.approx(1 / 100)
    assert A.bootstrap_p([-1.0, 1.0] * 50) == 1.0
    assert A.tost_p([0.0] * 99, 0.5) == pytest.approx(1 / 100)
    assert A.tost_p([0.6] * 99, 0.5) == 1.0
    assert A.sign_flip_p([1.0] * 30, [str(i) for i in range(30)], n=500) < 0.01
    assert A.sign_flip_p([1.0, -1.0] * 15, [str(i) for i in range(30)], n=500) > 0.5
    assert A.auc([1, 2, 3], [0, 0.5]) == 1.0 and A.auc([0], [1]) == 0.0 and A.auc([1], [1]) == 0.5
    # whole pairs are resampled together
    d = A.draws(A.mean, [1.0, 1.0, -1.0, -1.0], ["a", "a", "b", "b"], B=200)
    assert set(d) <= {1.0, 0.0, -1.0}


def test_between_uses_goal_minus_ingredient():
    rows = [("goal", 2.0), ("goal", 2.0), ("ingredient", 0.5), ("ingredient", 0.5)]
    r = A.between(rows, ["a", "b", "a", "b"], 1.0)
    assert r["estimate"] == pytest.approx(1.5) and r["goal_mean"] == 2.0 and r["ingredient_mean"] == 0.5


def test_mock_end_to_end(mock_runs, small, monkeypatch, fast):
    man = _score_all(mock_runs, small, list(R.TAGS))
    assert A.run_textcraft_r2_analysis({"run_id": "textcraft_r2_analysis", "manifest": man, "models": list(R.TAGS)}) in (0, 1)
    m = json.loads((mock_runs / "runs" / "textcraft_r2_analysis" / "metrics.json").read_text())
    assert m["family_size"] == 50 and set(m["per_model"]) == set(R.TAGS)
    assert all(v == 0 for v in m["missing_scores"].values())
    assert isinstance(m["verdicts"], dict) and set(m["verdicts"]) == set(R.TAGS)


# ---------------- added for the mutation check ----------------


def test_candidates(small):
    rec = small[0]
    assert R.candidates(rec, "yn_aligned") == ("Yes", "No") == R.candidates(rec, "yn_swapped")
    assert R.candidates(rec, "aligned") == (rec["candidates"]["o_plus"], rec["candidates"]["o_minus"])
    assert R.candidates(rec, "aligned")[0].startswith("Crafted ")


def test_a_parent_sharing_a_craftable_with_the_partner_does_not_qualify():
    found = 0
    for pair in I.B.all_pairs():
        for goal in pair.goals:
            partner = next(g for g in pair.goals if g != goal)
            for z in sorted(I.consumers().get(goal, ())):
                if z in pair.goals or z in I.B.EXCLUDED_GOALS or z in I.fewshot_goals():
                    continue
                if W.tree_craftables(W.item_id(z)) & W.tree_craftables(W.item_id(partner)):
                    assert not I.qualifies(goal, partner, z)
                    found += 1
    assert found > 0


def test_every_parent_is_the_rules_choice(records):
    pairs = {p.pair_id: p for p in I.B.all_pairs()}
    seen = {}
    for r in records:
        if r["kind"] == "parent" and (r["pair_id"], r["goal"]) not in seen:
            c = I.choose_parent(pairs[r["pair_id"]], r["goal"])
            seen[(r["pair_id"], r["goal"])] = (c["parent"], c["rule"]) == (r["parent"], r["parent_rule"])
    assert len(seen) == 118 and all(seen.values())


def test_rebuilding_two_pairs_gives_the_committed_records(records):
    frozen_ids = {json.loads(x)["id"] for x in I.B.ITEMS.read_text().splitlines() if x.strip()}
    pids = sorted({r["pair_id"] for r in records if r["kind"] == "parent"})[:2]
    rebuilt, _stats = I.parent_records(frozen_ids, 1, set(pids))
    assert I.dumps(rebuilt) == I.dumps([r for r in records if r["kind"] == "parent" and r["pair_id"] in pids])


def test_padded_twin_and_class_checks(small):
    rec = next(x for x in small if "pad8_aligned" in x["arms"])
    bad_twin = {**rec, "twin": rec["twin"] + "x"}
    assert any("twin does not reproduce (goal" in p for p in I.replay_problems(bad_twin))
    bad_class = {**rec, "failure_class": "used_up" if rec["failure_class"] != "used_up" else "too_few"}
    assert any("padded history changes the class" in p for p in I.checker_problems(bad_class))


def _gate_scorer(tok, tag, attn=None):
    sc = _fake_scorer(tok, tag)
    sc._model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation=attn), dtype="bf16")
    return sc


def test_run_gate_requires_the_golden_and_the_attention(small, tmp_path, monkeypatch):
    assert not S.run_gate(tmp_path, _gate_scorer(FakeTok(), "qwen3-8b"), "qwen3-8b", small[:5], small, SHOTS)
    rec = json.loads((tmp_path / "tokenization_checks.json").read_text())
    assert rec["golden_problems"] and rec["ok"] is False
    monkeypatch.setattr(T, "golden_problems", lambda *a, **k: [])
    assert S.run_gate(tmp_path, _gate_scorer(FakeTok(), "qwen3-8b"), "qwen3-8b", small[:5], small, SHOTS)
    assert not S.run_gate(tmp_path, _gate_scorer(FakeTok(no_system=True), "gemma2-9b-it", "sdpa"), "gemma2-9b-it",
                          small[:5], small, SHOTS)
    assert S.run_gate(tmp_path, _gate_scorer(FakeTok(no_system=True), "gemma2-9b-it", "eager"), "gemma2-9b-it",
                      small[:5], small, SHOTS)


def test_timing_not_applicable_without_a_role_gap(small, fast):
    recs = copy.deepcopy(small)
    scores = {t: _planted(recs, t, lam=0.5, nu=0.0, kappa=0.0, tau=0.0, noise=0.2) for t in R.TAGS}
    out = A.analyse(recs, SHOTS, scores, list(R.TAGS))
    assert out["per_model"]["qwen3-8b"]["contrasts"]["timing"]["applicable"] is False
    assert out["verdicts"]["qwen3-8b"]["contrasts"]["timing"] == "not applicable"


def test_family_is_holm_adjusted(small, fast):
    recs = copy.deepcopy(small)
    out = A.analyse(recs, SHOTS, {t: _planted(recs, t, noise=0.5) for t in R.TAGS}, list(R.TAGS))
    fam = out["family"]
    smallest = min(v["p_in_family"] for v in fam.values())
    k = next(k for k, v in fam.items() if v["p_in_family"] == smallest)
    assert fam[k]["p_holm"] == pytest.approx(min(1.0, 50 * smallest))
    assert all(v["p_holm"] >= v["p_in_family"] for v in fam.values())


def test_populations_on_the_full_records(records):
    pops = A.populations(records)
    assert len(pops["goal"]) == 504 and len(pops["ingredient"]) == 990 and len(pops["padded"]) == 920
    assert len(pops["parent_failures"]) == 675 and len(pops["parent_successes"]) == 207
    assert len(pops["main_failures"]) == 1500 and len(pops["successes"]) == 600


def test_sign_flip_flips_whole_pairs():
    values = [3.0, -1.0] * 15
    clusters = [str(i // 2) for i in range(30)]
    assert A.sign_flip_p(values, clusters, n=500) < 0.005


def test_heuristics_agree_with_the_competence_script_on_every_item():
    """r2_heuristics is a port of features() in scripts/textcraft_llama_competence.py (definitions fixed at 9a83738)."""
    import importlib.util
    pytest.importorskip("numpy")
    from wf.textcraft import r2_heuristics as H
    spec = importlib.util.spec_from_file_location("tlc", I.REPO / "scripts" / "textcraft_llama_competence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    items = [json.loads(x) for x in I.B.ITEMS.read_text().splitlines() if x.strip()]
    items = [t for t in items if t["population"] in ("main", "success")]
    for t in items:
        ours, theirs = H.features(t), mod.features(t)
        assert all(ours[h] == theirs[h] for h in H.HEURISTICS + ("H9",)), t["id"]
        assert ours["H6"] == int(t["population"] == "success")


def test_heuristic_fits_and_role_matched_used_up(small, fast):
    recs = copy.deepcopy(small)
    scores = _planted(recs, "qwen3-8b")
    by, _ = A.index_scores(scores, recs, SHOTS, "qwen3-8b")
    g = A.competence(by, A.populations(recs))["descriptive"]
    assert g["heuristics"]["yes_no"]["H6"]["tracking_auc"]["auc"] == pytest.approx(1.0)
    assert g["heuristics"]["yes_no"]["H6"]["kappa"] == pytest.approx(1.0)
    assert set(g["heuristics"]["two_candidate"]) == {"H1", "H2", "H6", "H2andH9"}
    assert "used_up_vs_ingredient_successes" in g["yes_no"] and "bin_1-3" in g["two_candidate"]
    assert A.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0 and A.kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_used_up_auc_is_role_matched(records, fast):
    pops = A.populations(records)
    uu = [r for r in pops["main_failures"] if r["failure_class"] == "used_up" and r["role"] == "ingredient"][:40]
    succ = [r for r in pops["successes"] if r["role"] == "ingredient"][:40] + [r for r in pops["successes"] if r["role"] == "goal"][:40]
    sub = {k: [] for k in pops}
    sub.update({"main_failures": uu, "successes": succ, "ingredient": uu, "ingredient_successes": succ[:40]})
    by = {}
    for r in uu + succ:
        L = 0.0 if r["population"] == "main" else (5.0 if r["role"] == "ingredient" else -10.0)
        for arm in ("neutral", "yn_neutral"):
            by[(r["id"], arm)] = {"L": L, "logp_plus": -1.0, "logp_minus": -1.0, "logp_first_plus": -1.0, "logp_first_minus": -1.0}
    d = A.competence(by, sub)["descriptive"]
    assert d["yes_no"]["used_up_vs_ingredient_successes"]["auc"] == 1.0
    assert d["yes_no"]["class_used_up"]["auc"] == 0.5


def test_gate_catches_a_candidate_written_differently_by_the_template(small):
    """Two-candidate arms only, so the yes/no check cannot be what fails."""
    recs = [{**r, "arms": [a for a in r["arms"] if not R.parse_arm(a)[0]]} for r in small[:5]]
    rep = T.gate(_fake_scorer(FakeTok(reply_prefix="▁ ")), R.MODELS["qwen3-8b"], recs, SHOTS)
    assert not rep["ok"] and any(" o_plus: alone " in p for p in rep["problems"])
    assert rep["n_yes_no_prompts_checked"] == 0


# ---------------- amendments before freeze (user decisions and review 1, 2026-11-21) ----------------


def test_verdicts_are_statistical_and_never_withheld():
    fam = {f"m:{c}:{t}": {"p_holm": 0.5} for c in A.CONTRASTS for t in A.TESTS}
    rec = {"estimate": -0.5, "ci95": [-0.8, -0.2], "applicable": True}
    assert A.verdict(rec, fam, "m", "parent_penalty") == "inconclusive"
    fam["m:parent_penalty:differs"]["p_holm"] = 0.01
    assert A.verdict(rec, fam, "m", "parent_penalty") == "confirmed below zero"
    assert A.verdict({**rec, "ci95": [-0.8, 0.1]}, fam, "m", "parent_penalty") == "inconclusive"
    assert A.verdict({"estimate": 0.5, "ci95": [0.2, 0.8]}, fam, "m", "parent_penalty") == "confirmed above zero"
    fam["m:lift:equivalent"]["p_holm"] = 0.049
    assert A.verdict({"estimate": 0.01, "ci95": [-0.1, 0.1]}, fam, "m", "lift") == "equivalent to zero"
    fam["m:lift:equivalent"]["p_holm"] = 0.05
    assert A.verdict({"estimate": 0.01, "ci95": [-0.1, 0.1]}, fam, "m", "lift") == "inconclusive"
    assert A.verdict({**rec, "applicable": False}, fam, "m", "timing") == "not applicable"
    assert A.verdict(None, fam, "m", "lift") == "waiting"


def test_interpretation_separates_measurable_effects_from_distortions():
    assert A.interpretation("confirmed below zero", True, {}) == "distortion in a competent predictor"
    assert A.interpretation("confirmed above zero", False, {}).startswith("statistically measurable goal effect")
    assert A.interpretation("equivalent to zero", True, {}).startswith("no effect larger than the margin, in a competent")
    assert "limited interpretation" in A.interpretation("equivalent to zero", False, {})
    assert A.interpretation("inconclusive", True, {"equivalence_underpowered": True}) == "inconclusive (equivalence underpowered)"
    assert A.interpretation("inconclusive", False, {"equivalence_underpowered": False}) == "inconclusive"
    assert "no effect" not in A.interpretation("inconclusive", True, {"equivalence_underpowered": True})


@pytest.mark.parametrize("p,dn,dc,want", [
    ("confirmed below zero", "inconclusive", "confirmed below zero", "completion contributes; naming not constrained"),
    ("confirmed below zero", "confirmed above zero", "confirmed below zero", "completion penalty, naming constrained"),
    ("confirmed below zero", "equivalent to zero", "confirmed below zero", "completion penalty, naming constrained"),
    ("confirmed below zero", "confirmed below zero", "confirmed below zero", "naming / partial-completion penalty"),
    ("confirmed below zero", "confirmed below zero", "inconclusive", "naming / partial-completion penalty"),
    ("confirmed below zero", "inconclusive", "inconclusive", "penalty present, mechanism unresolved"),
    ("confirmed below zero", "equivalent to zero", "inconclusive", "penalty present, mechanism unresolved"),
    ("equivalent to zero", "confirmed below zero", "inconclusive", "no penalty larger than the margin"),
    ("confirmed above zero", "inconclusive", "inconclusive", "completing a named goal raises predicted success"),
    ("inconclusive", "confirmed below zero", "confirmed below zero", "inconclusive on the penalty"),
])
def test_reading_table(p, dn, dc, want):
    r = A.reading({"lift": "inconclusive", "parent_penalty": p, "naming": dn, "completion": dc, "timing": "inconclusive"})
    assert r["penalty"].startswith(want)


def test_timing_qualifier_and_no_partial_naming_rule():
    v = {"lift": "inconclusive", "parent_penalty": "confirmed below zero", "naming": "inconclusive",
         "completion": "inconclusive", "timing": "confirmed above zero"}
    r = A.reading(v, timing_comparable=True)
    assert "partial naming" not in r["penalty"]  # user point 4: the subset comparison is reported, not read as a rule
    assert r["timing"].startswith("sensitivity to added empty-inventory steps") and r["timing"].endswith("not read as timing")
    assert "not read as timing" not in A.reading(v)["timing"]
    base = {**v, "parent_penalty": "inconclusive"}
    assert A.reading({**base, "timing": "equivalent to zero"})["timing"].startswith("no sensitivity")
    assert A.reading({**base, "timing": "not applicable"})["timing"] == "not applicable"
    assert A.reading({**base, "timing": "confirmed below zero"})["timing"].endswith("(F4)")
    assert A.reading({**base, "lift": "equivalent to zero"})["lift"] == "no lift larger than the margin"


def test_model_verdicts_keep_estimates_and_limit_interpretation():
    block = {"contrasts": {c: {"estimate": -1.0, "ci95": [-2, -0.5], "applicable": True} for c in A.CONTRASTS},
             "competence": {"interpretable": False, "local": {"per_contrast": {c: {"competent": True} for c in A.CONTRASTS}}},
             "descriptive": {"parent": {"penalty_by_subset": {"strict": {"estimate": 0.2}}}}}
    block["contrasts"]["timing"]["comparable_to_one_lookup_change"] = False
    fam = {f"m:{c}:{t}": {"p_holm": 0.001 if t == "differs" else 1.0} for c in A.CONTRASTS for t in A.TESTS}
    block["competence"]["local"]["per_contrast"]["lift"]["competent"] = False
    out = A.model_verdicts(block, fam, "m")
    assert out["contrasts"]["lift"] == "confirmed below zero" and out["interpretation"]["lift"].startswith("statistically measurable")
    assert out["interpretation"]["parent_penalty"] == "distortion in a competent predictor"
    assert out["overall_gate_interpretable"] is False and "partial naming" not in out["reading"]["penalty"]


def test_across_models_patterns_and_f3():
    v = lambda p: {"parent_penalty": p}
    comp = {t: True for t in R.TAGS}
    allb = {t: v("confirmed below zero") for t in R.TAGS}
    a = A.across(allb, comp)
    assert a["penalty_pattern"] == "penalty general across the competent models" and a["F3_penalty_not_shared"] is False
    only = {t: v("confirmed below zero" if t == "llama3.1-8b" else "inconclusive") for t in R.TAGS}
    a = A.across(only, comp)
    assert a["penalty_pattern"] == "penalty below zero for Llama-3.1 only" and a["F3_penalty_not_shared"] == "undecided"
    one_eq = {**only, "gemma2-9b-it": v("equivalent to zero")}
    assert A.across(one_eq, comp)["F3_penalty_not_shared"] is True
    mixed = {**one_eq, "qwen3-8b": v("confirmed below zero")}
    a = A.across(mixed, comp)
    assert a["F3_penalty_not_shared"] is False and a["penalty_pattern"] == "penalty model-specific"
    above = {**only, "mistral-7b-v0.3": v("confirmed above zero")}
    assert A.across(above, comp)["F3_penalty_not_shared"] is True
    # a model not competent for the penalty does not count either way
    assert A.across(one_eq, {**comp, "gemma2-9b-it": False})["F3_penalty_not_shared"] == "undecided"
    assert A.across(allb, {t: t == "qwen3-8b" for t in R.TAGS})["penalty_pattern"].startswith("fewer than two")
    assert A.across(allb, {**comp, "llama3.1-8b": False})["F1_llama_no_penalty"].startswith("not testable")
    assert A.across({**allb, "llama3.1-8b": v("equivalent to zero")}, comp)["F1_llama_no_penalty"] is True
    assert "top-item status" in A.across(allb, comp)["F1_note"]


def test_margin_is_capped_at_four_tenths(small, fast):
    import statistics
    recs = copy.deepcopy(small)
    pops = A.populations(recs)
    for noise, seed in ((0.3, 3), (5.0, 4)):
        scores = _planted(recs, "qwen3-8b", noise=noise, seed=seed)
        by, _ = A.index_scores(scores, recs, SHOTS, "qwen3-8b")
        d = [A.L(by, r["id"], "aligned") - A.L(by, r["id"], "swapped") for r in pops["main_failures"]]
        m = A.margin_of(by, pops)
        assert m["margin"] == pytest.approx(min(0.3 * statistics.stdev(d), 0.4)) and m["cap_nats"] == 0.4
    assert m["margin"] == 0.4  # a wide spread hits the cap
    c = A.contrasts(by, pops, m["margin"])
    assert all(c[k]["margin"] == m["margin"] for k in A.CONTRASTS)  # every contrast uses margin_of's value


def test_equivalence_power():
    assert A.equivalence_power(0.08, 0.4, 3.090232) == pytest.approx(0.94, abs=0.01)
    assert A.equivalence_power(0.24, 0.4, 3.090232) == 0.0
    assert A.equivalence_power(0.04, 0.4, 1.644854) == pytest.approx(1.0, abs=1e-6)
    blk = A._power_block([0.0, 0.5, -0.5, 0.2, -0.2] * 10, 0.4)
    assert blk["equivalence_underpowered"] is True
    assert A.Z_HOLM_FIRST == pytest.approx(3.090232) and A.POWER_MIN == 0.8


def test_family_tests_use_at_least_20000_draws():
    assert A.B_DRAWS >= 20000 and A.N_FLIPS >= 20000


def test_fast_draws_equal_the_general_path():
    import random
    rng = random.Random(1)
    rows = [("goal" if i % 3 else "ingredient", rng.gauss(0, 1)) for i in range(60)]
    cl = [f"p{i % 11}" for i in range(60)]
    fast = A.draws(A._gap, rows, cl, B=200)
    slow = A.draws(lambda r: A._gap(r), rows, cl, B=200)
    assert fast == pytest.approx(slow)
    vals = [x for _r, x in rows]
    assert A.draws(A.mean, vals, cl, B=200) == pytest.approx(A.draws(lambda v: A.mean(v), vals, cl, B=200))


def _yn_by(recs, value):
    by = {}
    for r in recs:
        L = value(r)
        m = 0.9
        by[(r["id"], "yn_neutral")] = {"L": L, "logp_plus": math.log(m / 2), "logp_minus": math.log(m / 2),
                                       "logp_first_plus": math.log(m / 2), "logp_first_minus": math.log(m / 2)}
    return by


def test_local_competence_rules_and_the_boundary(records, fast):
    pops = A.populations(records)
    every = pops["main_failures"] + pops["successes"] + pops["parent_failures"] + pops["parent_successes"]
    base = lambda r: 3.0 if r["population"] == "success" else -3.0
    by = _yn_by(every, base)
    overall = A.gate_checks(by, pops["main_failures"], pops["successes"])
    lc = A.local_competence(by, pops, overall)
    assert all(lc["per_contrast"][c]["competent"] for c in A.CONTRASTS)
    assert lc["per_contrast"]["parent_penalty"]["populations"].keys() == {"parent", "goal_main"}
    assert lc["per_contrast"]["naming"]["populations"].keys() == {"goal_main", "ingredient_matched"}
    assert lc["per_contrast"]["timing"]["populations"].keys() == {"goal_padded", "ingredient_padded_matched"}
    assert lc["per_contrast"]["lift"]["populations"].keys() == {"ingredient_all"}
    # the parent population alone failing makes the penalty not competent, and nothing else
    bad_parent = _yn_by(every, lambda r: -base(r) if r["kind"] == "parent" else base(r))
    lc2 = A.local_competence(bad_parent, pops, overall)
    assert not lc2["per_contrast"]["parent_penalty"]["competent"]
    assert all(lc2["per_contrast"][c]["competent"] for c in A.CONTRASTS if c != "parent_penalty")
    # used-up ingredient failures do not enter the class-matched populations
    uu = _yn_by(every, lambda r: 5.0 if (r["population"] == "main" and r.get("failure_class") == "used_up") else base(r))
    lc3 = A.local_competence(uu, pops, A.gate_checks(uu, pops["main_failures"], pops["successes"]))
    assert lc3["populations"]["ingredient_matched"]["passes"] and not lc3["populations"]["ingredient_all"]["passes"]
    assert lc3["per_contrast"]["naming"]["competent"] and not lc3["per_contrast"]["lift"]["competent"]
    # the other three criteria on the full population
    yes_all = _yn_by(every, lambda r: abs(base(r)) + (1.0 if r["population"] == "success" else 0.5))
    lc4 = A.local_competence(yes_all, pops, A.gate_checks(yes_all, pops["main_failures"], pops["successes"]))
    assert not lc4["other_criteria_on_full_population_pass"] and not any(v["competent"] for v in lc4["per_contrast"].values())
    assert lc["populations"]["parent"]["classification"]["yes_no"]["success_classification_rate"] == 1.0
    assert "H6" in lc["populations"]["goal_main"]["heuristics"]["yes_no"]


@pytest.mark.parametrize("n_low,want", [(3, True), (4, False)])
def test_local_auc_boundary_is_inclusive(records, n_low, want, fast):
    """10 successes at 1 against 10 failures, 7 at 0 and 3 at 2: AUC 0.70 passes; one more failure at 2 fails."""
    pops = A.populations(records)
    succ, fail = pops["ingredient_successes"][:10], pops["ingredient"][:10]
    sub = {k: list(v) for k, v in pops.items()}
    sub["ingredient"], sub["ingredient_successes"] = fail, succ
    every = pops["main_failures"] + pops["successes"] + pops["parent_failures"] + pops["parent_successes"]
    vals = {r["id"]: (3.0 if r["population"] == "success" else -3.0) for r in every}
    vals.update({r["id"]: 1.0 for r in succ})
    vals.update({r["id"]: (2.0 if i < n_low else 0.0) for i, r in enumerate(fail)})
    by = _yn_by(every, lambda r: vals[r["id"]])
    lc = A.local_competence(by, sub, A.gate_checks(by, pops["main_failures"], pops["successes"]))
    assert lc["populations"]["ingredient_all"]["yes_no_auc"] == pytest.approx(1 - n_low / 10)
    assert lc["per_contrast"]["lift"]["competent"] is want


def test_contrasts_use_class_matched_ingredients(small, fast):
    recs = copy.deepcopy(small)
    base = _planted(recs, "qwen3-8b")
    uu = {r["id"] for r in recs if r.get("failure_class") == "used_up" and r["role"] == "ingredient"}
    assert uu
    shifted = [{**s, "L": s["L"] + 5.0} if (s["transition_id"] in uu and s["condition"] in ("both", "pad8_aligned")) else s
               for s in base]
    pops = A.populations(recs)
    a = A.contrasts(A.index_scores(base, recs, SHOTS, "qwen3-8b")[0], pops, 0.4)
    b = A.contrasts(A.index_scores(shifted, recs, SHOTS, "qwen3-8b")[0], pops, 0.4)
    for c in ("naming", "completion", "timing"):
        assert a[c]["estimate"] == pytest.approx(b[c]["estimate"])
    assert a["lift"]["estimate"] == pytest.approx(b["lift"]["estimate"])


def test_timing_applicability_reads_the_interval(small, fast):
    recs = copy.deepcopy(small)
    pops = A.populations(recs)
    for seed in range(20):
        sc = _planted(recs, "qwen3-8b", lam=0.0, nu=-0.03, kappa=0.0, tau=0.0, noise=1.0, seed=seed)
        c = A.contrasts(A.index_scores(sc, recs, SHOTS, "qwen3-8b")[0], pops, 0.4)
        g = c["timing"]["role_gap_at_one_lookup"]
        if g["estimate"] < 0 < g["ci95"][1]:
            assert c["timing"]["applicable"] is False
            break
    else:
        pytest.fail("no seed gave a negative estimate with an interval across zero")


def test_timing_reports_the_one_lookup_change(small, fast):
    recs = copy.deepcopy(small)
    c = A.contrasts(A.index_scores(_planted(recs, "qwen3-8b"), recs, SHOTS, "qwen3-8b")[0], A.populations(recs), 0.4)
    one = c["timing"]["one_lookup_vs_none"]
    assert one["estimate"] == pytest.approx(0.0, abs=1e-9)  # planted: one look-up changes nothing
    assert c["timing"]["comparable_to_one_lookup_change"] is False  # T = +0.5 with no noise


def test_parent_subsets_overlap_kinds_and_splits(records, small, fast):
    pf = [r for r in records if r["kind"] == "parent" and r["population"] == "main"]
    cnt = Counter(A.parent_subset(r) for r in pf)
    assert cnt == {"strict": 176, "word_only": 298, "full_name": 201}
    assert len({r["pair_id"] for r in pf if A.parent_subset(r) == "full_name"}) == 33
    assert A.word_contained("oak planks", "oak planks sign") and not A.word_contained("stone stairs", "smooth sandstone stairs")
    assert A.overlap_kind("stone stairs", "smooth sandstone stairs") == "word_only"
    assert A.overlap_kind("melon", "glistering melon slice") == "full_name" and A.overlap_kind("stick", "lantern") == "none"
    ing = [r for r in records if r["kind"] == "main" and r["population"] == "main" and r["role"] == "ingredient"]
    assert Counter(A.overlap_kind(r["object"], r["goal"]) for r in ing) == {"word_only": 503, "none": 307, "full_name": 180}
    recs = copy.deepcopy(small)
    d = A.descriptive(A.index_scores(_planted(recs, "qwen3-8b"), recs, SHOTS, "qwen3-8b")[0], A.populations(recs))["parent"]
    assert d["split_completion_ext_aligned_minus_ext_both"]["estimate"] + d["split_naming_ext_both_minus_ext_parent"]["estimate"] \
        == pytest.approx(0.4 - 1.1)
    assert d["penalty_on_own_displays"]["estimate"] == pytest.approx(0.4 - 1.1 + 0.2)
    assert set(d["penalty_subset_differences"]) == {"strict_minus_word_only", "strict_minus_full_name", "word_only_minus_full_name"}


def test_hash_covers_every_field_and_context_key(small, monkeypatch):
    rec = _first(small, kind="parent")
    h = R.item_hash(rec, SHOTS)
    for field in R.HASHED:
        assert R.item_hash({**rec, field: "changed"}, SHOTS) != h, field
    for mod, name, value in ((P, "NEXT", "Next:"), (R, "YN_QUESTION", "Q?")):
        monkeypatch.setattr(mod, name, value)
        assert R.item_hash(rec, SHOTS) != h, name
        monkeypatch.undo()


def test_prompt_check_catches_a_changed_system_text(small, monkeypatch):
    rec = next(r for r in small if r["kind"] == "main" and r["both_task_line"])
    orig = R.messages

    def changed(r, arm, shots, system=R.SYSTEM):
        m = orig(r, arm, shots, system)
        return ([{**m[0], "content": m[0]["content"] + " "}] + m[1:]) if arm == "both" else m
    monkeypatch.setattr(R, "messages", changed)
    assert any("both: system text or examples differ" in p for p in I.prompt_problems(rec, SHOTS))


def test_excluded_goals_never_qualify_as_parents():
    z = "blue dye"
    g = next(n for n, outs in I.consumers().items() if z in outs)
    p = next(pp.goals[0] for pp in I.B.all_pairs()
             if not (W.tree_craftables(W.item_id(z)) & W.tree_craftables(W.item_id(pp.goals[0]))) and pp.goals[0] != g)
    assert not I.qualifies(g, p, z)


def test_text_checks_catch_a_named_strict_parent_and_a_changed_list(small):
    frozen = {t["id"]: t for t in (json.loads(x) for x in I.B.ITEMS.read_text().splitlines() if x.strip())}
    rec = next(r for r in small if r["kind"] == "parent" and r["parent_rule"] == "word_overlap")
    assert any("strict parent shares a word" in p for p in I.text_parent_problems({**rec, "parent_rule": "strict"}))
    moved = {**rec, "recipe_lines": rec["recipe_lines"][1:] + rec["recipe_lines"][:1]}
    assert any("list is not the pair's list" in p for p in I.structure_problems(moved, frozen))


def test_padded_reply_check(small):
    rec = next(x for x in small if "pad8_aligned" in x["arms"])
    bad = {**rec, "reply": rec["reply"].replace("Could not find enough items", "Could not find any items")}
    assert any("padded history changes the verdict" in p for p in I.checker_problems(bad))


def test_padded_replay_covers_no_goal(small, monkeypatch):
    rec = next(x for x in small if "pad8_aligned" in x["arms"])
    orig = W.replay

    def odd(lines, goal_iid, actions):
        env, hist = orig(lines, goal_iid, actions)
        if goal_iid is None and actions[:1] == ["inventory"]:
            hist = [[hist[0][0], hist[0][1] + " "]] + hist[1:]
        return env, hist
    monkeypatch.setattr(W, "replay", odd)
    assert any("goal None" in p for p in I.replay_problems(rec))


def test_nan_scores_count_as_missing(small):
    recs = copy.deepcopy(small)
    sc = _planted(recs, "qwen3-8b")
    sc[0] = {**sc[0], "L": float("nan")}
    _by, missing = A.index_scores(sc, recs, SHOTS, "qwen3-8b")
    assert (sc[0]["transition_id"], sc[0]["condition"]) in missing


def test_mass_criterion_uses_failures_and_successes(records):
    pops = A.populations(records)
    f, s = pops["main_failures"][:5], pops["successes"][:10]
    by = {}
    for r, m in [(r, 0.4) for r in f] + [(r, 0.9) for r in s]:
        by[(r["id"], "yn_neutral")] = {"L": 1.0 if r in s else -1.0, "logp_plus": math.log(m / 2), "logp_minus": math.log(m / 2)}
    assert A.gate_checks(by, f, s)["median_yes_no_mass_neutral"]["pass"] is True


def test_resume_refuses_scores_of_another_model(mock_runs, small):
    _score_all(mock_runs, small, ["qwen3-8b"], n_shards=1)
    d = mock_runs / "runs" / "textcraft_r2_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl"
    lines = d.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["model"] = "llama3.1-8b"
    d.write_text("\n".join([json.dumps(rec)] + lines[1:-3]) + "\n")
    cfg = {"run_id": "textcraft_r2_score_qwen3-8b", "model": "qwen3-8b", "manifest": str(mock_runs / "r2_manifest.json"),
           "n_shards": 1, "resume": True}
    with pytest.raises(SystemExit):
        S.run_textcraft_r2_score(cfg)


def test_shards_are_sorted_pair_slices():
    recs = [{"pair_id": f"p{i:03d}", "id": str(i)} for i in (7, 3, 9, 1, 5, 2, 8, 4, 6, 10)]
    assert {r["pair_id"] for r in S.shard_records(recs, 0, 2)} == {"p001", "p003", "p005", "p007", "p009"}
    assert {r["pair_id"] for r in S.shard_records(recs, 1, 3)} == {"p002", "p005", "p008"}


def test_sign_flip_p_is_smoothed():
    p = A.sign_flip_p([1.0] * 30, [str(i) for i in range(30)], n=50)
    assert p == pytest.approx(1 / 51)


def test_provenance_requires_one_commit_and_pinned_revisions(mock_runs, small):
    _score_all(mock_runs, small, ["qwen3-8b", "gemma2-9b-it"])
    root = mock_runs / "runs"
    assert A.provenance(["qwen3-8b", "gemma2-9b-it"], root) == []
    (root / "textcraft_r2_score_gemma2-9b-it" / "shard_01_of_02" / "git_sha").write_text("deadbeef\n")
    assert any("commits" in p for p in A.provenance(["qwen3-8b", "gemma2-9b-it"], root))
    (root / "textcraft_r2_score_gemma2-9b-it" / "shard_01_of_02" / "git_sha").write_text(
        (root / "textcraft_r2_score_qwen3-8b" / "shard_00_of_02" / "git_sha").read_text())
    f = root / "textcraft_r2_score_qwen3-8b" / "shard_00_of_02" / "scoring.jsonl"
    lines = f.read_text().splitlines()
    f.write_text("\n".join([json.dumps({**json.loads(lines[0]), "model_revision": "other"})] + lines[1:]) + "\n")
    assert any("model revisions" in p for p in A.provenance(["qwen3-8b"], root))


def test_role_gap_descriptive_equals_naming_plus_completion(small, fast):
    recs = copy.deepcopy(small)
    by = A.index_scores(_planted(recs, "qwen3-8b"), recs, SHOTS, "qwen3-8b")[0]
    pops = A.populations(recs)
    c = A.contrasts(by, pops, 0.4)
    d = A.descriptive(by, pops)
    assert d["role_gap_main"]["estimate"] == pytest.approx(c["naming"]["estimate"] + c["completion"]["estimate"])


def test_round2_question_is_anchored_to_the_replies():
    """14.Q: the round-2 yes/no question names TextCraft's success and failure replies, as ALFWorld's names "Nothing
    happens."; round 1's unanchored question is kept for round 1 only."""
    q = R.YN_QUESTION
    assert q.startswith("Question: will the environment reply") and q.endswith("\nAnswer:")
    for reply_start in ('"Crafted ', '"Got ', '"Could not find '):
        assert reply_start in q
    assert "Answer Yes if it succeeds and No if it fails." in q
    assert P.YN_QUESTION == "Question: Will this action succeed? Answer Yes or No.\nAnswer:"
    import hashlib
    assert hashlib.sha256(q.encode()).hexdigest() == "5e31ac7a4a0136367d51dfec938347caa71c0b2f9d740fac95d34b0034180f05"
    # every reply a round-2 action can get starts with one of the named forms
    recs = I.read_records()
    for r in recs[:400]:
        for text in (r["candidates"]["o_plus"], r["candidates"]["o_minus"]):
            assert text.startswith(("Crafted ", "Got ", "Could not find ")), text


# ---------------- user points 1-4 (2026-11-21) ----------------


def test_no_two_item_record_has_a_named_goal_already_made(records):
    both = [r for r in records if r.get("both_task_line")]
    assert len(both) == 2963 and not any(R.two_item_partner_done(r) for r in both)
    rep = I.check_records([r for r in records if r["kind"] == "main"][:50], SHOTS)
    assert rep["two_item_records_with_a_named_goal_already_made_or_held"] == {}


def test_a_made_partner_is_detected_and_leaves_naming_and_completion(small, fast):
    recs = copy.deepcopy(small)
    goal_rec = next(r for r in recs if r["kind"] == "main" and r["role"] == "goal" and r["population"] == "main"
                    and r["both_task_line"])
    made = f"Crafted 1 minecraft:{goal_rec['partner'].replace(' ', '_')}"
    fake = {**goal_rec, "history": [["get 1 y", "Got 1 y"], ["craft 1 x using 1 y", made]]}
    assert R.two_item_partner_done(fake) is True  # made by a craft
    assert R.two_item_partner_done({**goal_rec, "history": [["get 1 x", f"Got 1 {goal_rec['partner']}"]]}) is True
    scores = _planted(recs, "qwen3-8b")
    by = A.index_scores(scores, recs, SHOTS, "qwen3-8b")[0]
    pops = A.populations(recs)
    before = A.contrasts(by, pops, 0.4)["naming"]["n_goal"]
    for r in pops["goal"]:
        if r["id"] == goal_rec["id"]:
            r["history"] = [["get 1 x", f"Got 1 {goal_rec['partner']}"]] + r["history"]
    assert A.contrasts(by, pops, 0.4)["naming"]["n_goal"] == before - 1


def test_matched_sensitivity_removes_a_profile_confound():
    """Goal-role items sit in one cell, ingredients in another, plus matched cells with no role difference: the
    unmatched gap is large, the matched one is zero."""
    recs, vals = [], {}
    k = 0
    for role, prof, x, n in (("goal", ("1-3", "raw", "one"), 0.0, 10), ("ingredient", ("1-3", "raw", "one"), 0.0, 10),
                             ("goal", ("4-8", "craftable", "several"), 5.0, 10), ("ingredient", ("9-12", "raw", "one"), -5.0, 10)):
        for _ in range(n):
            k += 1
            recs.append({"id": f"i{k}", "pair_id": f"p{k % 7}", "role": role, "prof": prof})
            vals[f"i{k}"] = x
    out = A.matched_gap({}, recs, lambda i: vals[i], profile=lambda r: r["prof"], B=50)
    assert out["estimate"] == pytest.approx(0.0) and out["n_cells_used"] == 1 and out["n_items_in_used_cells"] == 20
    # a small cell (3 of each role) is left out; a second used cell of 5 per role with gap 1.5 is weighted by 5
    for role, prof, x, n in (("goal", ("1-3", "mixed", "one"), 9.0, 3), ("ingredient", ("1-3", "mixed", "one"), 0.0, 3),
                             ("goal", ("9-12", "craftable", "one"), 1.5, 5), ("ingredient", ("9-12", "craftable", "one"), 0.0, 5)):
        for _ in range(n):
            k += 1
            recs.append({"id": f"i{k}", "pair_id": f"p{k % 7}", "role": role, "prof": prof})
            vals[f"i{k}"] = x
    out = A.matched_gap({}, recs, lambda i: vals[i], profile=lambda r: r["prof"], B=50)
    assert out["n_cells_used"] == 2 and out["estimate"] == pytest.approx((10 * 0.0 + 5 * 1.5) / 15)
    assert A.short_profile({"length_bin": "4-8", "recipe_lines": ["craft 1 a using 1 b"],
                            "short_inputs": [{"name": "a"}]}) == ("4-8", "craftable", "one")
    assert A.short_profile({"length_bin": "4-8", "recipe_lines": ["craft 1 a using 1 b"],
                            "short_inputs": [{"name": "a"}, {"name": "b"}]}) == ("4-8", "mixed", "several")


def test_sensitivity_and_strict_comparison_are_reported(small, fast, monkeypatch):
    monkeypatch.setattr(A, "MATCH_MIN", 1)
    recs = copy.deepcopy(small)
    strict = {r["id"] for r in recs if r["kind"] == "parent" and r["parent_rule"] == "strict"}
    other = {r["id"] for r in recs if r["kind"] == "parent" and r["parent_rule"] != "strict" and r["population"] == "main"}
    assert strict and other
    sc = [{**x, "L": x["L"] + 1.0} if (x["transition_id"] in strict and x["condition"] == "ext_aligned") else x
          for x in _planted(recs, "qwen3-8b")]
    by = A.index_scores(sc, recs, SHOTS, "qwen3-8b")[0]
    d = A.descriptive(by, A.populations(recs))
    assert set(d["sensitivity_matched"]) == {"naming", "completion", "timing", "role_gap"}
    assert d["sensitivity_matched"]["naming"]["estimate"] == pytest.approx(0.4)
    assert d["sensitivity_matched"]["completion"]["estimate"] == pytest.approx(-1.1)
    assert d["parent"]["penalty_strict_minus_overlapping"]["estimate"] == pytest.approx(1.0)
