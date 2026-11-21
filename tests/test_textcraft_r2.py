"""TextCraft round 2 (notes/textcraft_r2_design.md): the clean prompt, the arms, the records and their checks, the
tokenization gate on a stand-in tokenizer, the scoring run in mock mode, and the analysis on planted scores."""
from __future__ import annotations

import copy
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
    assert m["yn_aligned"][-1]["content"].endswith(P.YN_QUESTION)
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
    assert sum(len(r["arms"]) for r in records) == 24008 - 20
    padded = [r for r in records if "pad8_aligned" in r["arms"]]
    assert len(padded) == 920 and all(r["population"] == "main" and r["length_bin"] in ("1-3", "4-8") for r in padded)
    assert all(r["arms"] == list(R.PARENT_ARMS) for r in records if r["kind"] == "parent")


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
            for m in R.model_messages(R.MODELS[tag], by_id[case["id"]], case["arm"], SHOTS):
                piece = m["content"].strip().split("\n")[-1]
                found = text.find(piece, pos)
                assert found >= 0, (tag, case)
                pos = found


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


def test_failed_gate_makes_every_contrast_not_interpretable(small, fast):
    recs = copy.deepcopy(small)
    scores = {t: _planted(recs, t) for t in R.TAGS}
    scores["mistral-7b-v0.3"] = [{**s, "L": abs(s["L"]) + 1} if s["condition"] == "yn_neutral" else s
                                 for s in scores["mistral-7b-v0.3"]]
    out = A.analyse(recs, SHOTS, scores, list(R.TAGS))
    v = out["verdicts"]["mistral-7b-v0.3"]
    assert v["competence"] is False and set(v["contrasts"].values()) == {"not interpretable"}
    assert v["reading"]["penalty"] == "not interpretable"
    assert "mistral-7b-v0.3" not in out["across_models"]["models_passing_gate"]


def test_verdict_rules():
    fam = {f"m:{c}:{t}": {"p_holm": 0.5} for c in A.CONTRASTS for t in A.TESTS}
    rec = {"estimate": -0.5, "ci95": [-0.8, -0.2], "applicable": True}
    assert A.verdict(rec, fam, "m", "parent_penalty", True) == "inconclusive"
    fam["m:parent_penalty:differs"]["p_holm"] = 0.01
    assert A.verdict(rec, fam, "m", "parent_penalty", True) == "confirmed below zero"
    assert A.verdict({**rec, "ci95": [-0.8, 0.1]}, fam, "m", "parent_penalty", True) == "inconclusive"
    assert A.verdict({"estimate": 0.5, "ci95": [0.2, 0.8]}, fam, "m", "parent_penalty", True) == "confirmed above zero"
    fam["m:lift:equivalent"]["p_holm"] = 0.049
    assert A.verdict({"estimate": 0.01, "ci95": [-0.1, 0.1]}, fam, "m", "lift", True) == "equivalent to zero"
    fam["m:lift:equivalent"]["p_holm"] = 0.05
    assert A.verdict({"estimate": 0.01, "ci95": [-0.1, 0.1]}, fam, "m", "lift", True) == "inconclusive"
    assert A.verdict(rec, fam, "m", "parent_penalty", False) == "not interpretable"
    assert A.verdict({**rec, "applicable": False}, fam, "m", "timing", True) == "not applicable"
    assert A.verdict(None, fam, "m", "lift", True) == "waiting"


@pytest.mark.parametrize("p,dn,dc,want", [
    ("confirmed below zero", "inconclusive", "confirmed below zero", "completion penalty"),
    ("confirmed below zero", "confirmed above zero", "confirmed below zero", "completion penalty"),
    ("confirmed below zero", "equivalent to zero", "confirmed below zero", "completion penalty"),
    ("confirmed below zero", "confirmed below zero", "confirmed below zero", "naming penalty"),
    ("confirmed below zero", "confirmed below zero", "inconclusive", "naming penalty"),
    ("confirmed below zero", "inconclusive", "inconclusive", "penalty present, mechanism unresolved"),
    ("equivalent to zero", "confirmed below zero", "inconclusive", "no penalty under the clean prompt"),
    ("confirmed above zero", "inconclusive", "inconclusive", "completing a named goal raises predicted success"),
    ("inconclusive", "confirmed below zero", "confirmed below zero", "inconclusive on the penalty"),
])
def test_reading_table(p, dn, dc, want):
    r = A.reading({"lift": "inconclusive", "parent_penalty": p, "naming": dn, "completion": dc, "timing": "inconclusive"})
    assert r["penalty"].startswith(want)


def test_timing_and_lift_readings():
    base = {"lift": "equivalent to zero", "parent_penalty": "inconclusive", "naming": "inconclusive", "completion": "inconclusive"}
    assert A.reading({**base, "timing": "confirmed above zero"})["timing"] == "timing"
    assert A.reading({**base, "timing": "equivalent to zero"})["timing"] == "content"
    assert A.reading({**base, "timing": "not applicable"})["timing"] == "not applicable"
    assert A.reading({**base, "timing": "confirmed below zero"})["timing"].endswith("(F4)")
    assert A.reading({**base, "timing": "inconclusive"})["lift"] == "no lift"


def test_across_models_patterns():
    v = lambda p: {"parent_penalty": p}
    gates = {t: True for t in R.TAGS}
    allb = {t: v("confirmed below zero") for t in R.TAGS}
    assert A.across(allb, gates)["penalty_pattern"] == "penalty general across the models that pass"
    only = {t: v("confirmed below zero" if t == "llama3.1-8b" else "inconclusive") for t in R.TAGS}
    a = A.across(only, gates)
    assert a["penalty_pattern"].endswith("(F3)") and a["F3_penalty_not_shared"] is True
    mixed = {t: v("confirmed below zero" if t in ("qwen3-8b", "llama3.1-8b") else "equivalent to zero") for t in R.TAGS}
    assert A.across(mixed, gates)["penalty_pattern"] == "penalty model-specific"
    one = {t: (t == "qwen3-8b") for t in R.TAGS}
    assert A.across(allb, one)["penalty_pattern"].startswith("fewer than two")
    assert A.across(allb, {**gates, "llama3.1-8b": False})["F1_llama_no_penalty"].startswith("not testable")
    eq = {**allb, "llama3.1-8b": v("equivalent to zero")}
    assert A.across(eq, gates)["F1_llama_no_penalty"] is True


def test_margin_is_three_tenths_of_the_item_sd(small, fast):
    recs = copy.deepcopy(small)
    scores = _planted(recs, "qwen3-8b", noise=1.0, seed=3)
    by, _ = A.index_scores(scores, recs, SHOTS, "qwen3-8b")
    pops = A.populations(recs)
    m = A.margin_of(by, pops)
    import statistics
    d = [A.L(by, r["id"], "aligned") - A.L(by, r["id"], "swapped") for r in pops["main_failures"]]
    assert m["margin"] == pytest.approx(0.3 * statistics.stdev(d))


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
    sub = {"main_failures": uu, "successes": succ, "parent_failures": [], "parent_successes": []}
    by = {}
    for r in uu + succ:
        L = 0.0 if r["population"] == "main" else (5.0 if r["role"] == "ingredient" else -10.0)
        for arm in ("neutral", "yn_neutral"):
            by[(r["id"], arm)] = {"L": L, "logp_plus": -1.0, "logp_minus": -1.0, "logp_first_plus": -1.0, "logp_first_minus": -1.0}
    d = A.competence(by, sub)["descriptive"]
    assert d["yes_no"]["used_up_vs_ingredient_successes"]["auc"] == 1.0
    assert d["yes_no"]["class_used_up"]["auc"] == 0.5
