"""Fixes after review 2 (notes/textcraft_design.md, section 12): fit measures, certificates, the plural exclusion,
the inventory flag, the verify checks review 2's mutants slipped past, analysis readouts and the tokenization check."""
import ast
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from wf.textcraft import analysis as A
from wf.textcraft import build as B
from wf.textcraft import checker as K
from wf.textcraft import fit as F
from wf.textcraft import score as S
from wf.textcraft import tokcheck as T
from wf.textcraft import verify as V
from wf.textcraft import world as W

REPO = Path(__file__).resolve().parents[1]
LINES = ["craft 1 ladder using 7 stick", "craft 4 stick using 2 planks", "craft 4 oak planks using 1 oak logs",
         "craft 1 minecart using 5 iron ingot", "craft 1 rail using 6 iron ingot, 1 stick"]


@pytest.fixture(scope="module")
def items():
    return V.read_items()


# ---------------- B1: fit measures ----------------

def test_fit_imports_no_environment_code():
    tree = ast.parse((REPO / "wf" / "textcraft" / "fit.py").read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"__future__", "wf.textcraft.checker"}


def test_fit_measures_known_answers():
    """Ladder's exclusive names: ladder, stick, planks, oak planks, oak logs; minecart's: minecart, iron ingot.
    (Rail is in neither goal's tree.)"""
    hist = [["get 2 oak logs", "Got 2 oak logs"],                                   # ladder side, 2
            ["get 5 iron ingot", "Got 5 iron ingot"],                               # minecart side, 5
            ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"],  # ladder side, craft, 4
            ["inventory", "Inventory: [oak logs] (1) [iron ingot] (5) [oak planks] (4) "],
            ["get 1 iron ingot", "Got 1 iron ingot"]]                               # minecart side, 1
    m = F.measures(LINES, hist, "ladder", "minecart")
    assert m == {"steps": [2, 2], "quantity": [6, 6], "crafts": [1, 0], "distinct_names": [2, 1],
                 "first_step": [1, 0], "last_step": [0, 1], "held_now": [5, 6], "completion": [0.0, 1.0],
                 "goal_words": [0, 0], "last_inventory": [2, 1]}
    assert F.signs(m) == {"steps": 0, "quantity": 0, "crafts": 1, "distinct_names": 1, "first_step": 1,
                          "last_step": -1, "held_now": -1, "completion": -1, "goal_words": 0, "last_inventory": 1}
    # every measure is antisymmetric in the two goals
    assert F.measures(LINES, hist, "minecart", "ladder") == {k: v[::-1] for k, v in m.items()}
    assert not F.none_favours_aligned(m)
    assert F.none_favours_aligned(F.measures(LINES, hist[1:2], "ladder", "minecart"))
    assert F.measures(LINES, [], "ladder", "minecart") == {k: [0, 0] if k != "completion" else [0.0, 0.0] for k in F.MEASURES}
    # goal words: "oak" belongs to "oak boat" only, "iron" to "iron door" only; the history says oak 7 times (2 inside
    # the inventory reply's brackets) and iron 5 times
    lines = LINES + ["craft 1 oak boat using 5 oak planks", "craft 3 iron door using 6 iron ingot"]
    assert F.measures(lines, hist, "oak boat", "iron door")["goal_words"] == [7, 5]


def test_completion_counts_generic_members():
    lines = ["craft 1 crafting table using 4 planks", "craft 4 oak planks using 1 oak logs", "craft 1 minecart using 5 iron ingot"]
    hist = [["get 1 oak logs", "Got 1 oak logs"], ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"]]
    assert F.measures(lines, hist, "crafting table", "minecart")["completion"] == [1.0, 0.0]


def test_stored_fit_measures_recompute(items):
    for it in items[::5]:
        assert it["fit_measures"] == F.measures(it["recipe_lines"], it["history"], it["goal"], it["partner"])
    it = next(t for t in items if t["population"] == "main")
    bad = {**it, "fit_measures": {**it["fit_measures"], "crafts": [9, 0]}}
    assert "fit measures do not recompute from the text" in V.structure_problems(bad)


def test_fit_regression_intercept_and_subset():
    """d = 2 + 1.5 * sign(crafts): the intercept is 2 whatever the mix of signs; the no-fit subset reads only the
    items whose history never speaks for the aligned goal."""
    rng = random.Random(3)
    items, scores = [], []
    for i in range(80):
        s = rng.choice([-1, 0, 1])
        fm = {m: [0, 0] for m in F.MEASURES}
        fm["crafts"] = [1, 0] if s > 0 else [0, 1] if s < 0 else [0, 0]
        t = {"id": f"t{i}", "pair_id": f"p{i % 9}", "fit_measures": fm}
        items.append(t)
        scores += [{"transition_id": t["id"], "condition": "aligned", "L": 2.0 + 1.5 * s + (i % 2) * 0.0},
                   {"transition_id": t["id"], "condition": "swapped", "L": 0.0},
                   {"transition_id": t["id"], "condition": "neutral", "L": 0.0}]
    r = A.fit_regression(scores, items)
    assert r["intercept"] == pytest.approx(2.0) and r["slopes"]["crafts"] == pytest.approx(1.5)
    sub = A.no_fit_for_aligned(items)
    assert sub and all(F.sign(t["fit_measures"]["crafts"]) <= 0 for t in sub)
    expect = sum(2.0 + 1.5 * F.sign(t["fit_measures"]["crafts"]) for t in sub) / len(sub)
    assert A.sensitivity(scores, items)["delta_L_no_fit_for_aligned"]["estimate"] == pytest.approx(expect)


# ---------------- B2: class contrasts within a role ----------------

def test_class_contrasts_stay_within_a_role():
    items, scores = [], []
    vals = {("used_up", "ingredient"): 5.0, ("too_few", "ingredient"): 3.0, ("never_obtained", "ingredient"): 1.0,
            ("too_few", "goal"): 10.0, ("never_obtained", "goal"): 4.0, ("used_up", "goal"): 100.0}
    for k, ((c, r), v) in enumerate(vals.items()):
        for i in range(6):
            t = {"id": f"{c}-{r}-{i}", "pair_id": f"p{i}", "failure_class": c, "role": r}
            items.append(t)
            scores += [{"transition_id": t["id"], "condition": "aligned", "L": v}, {"transition_id": t["id"], "condition": "swapped", "L": 0.0}]
    out = A.class_contrasts(scores, items)
    assert out["ingredient|used_up_minus_too_few"]["estimate"] == pytest.approx(2.0)
    assert out["ingredient|used_up_minus_never_obtained"]["estimate"] == pytest.approx(4.0)
    assert out["ingredient|too_few_minus_never_obtained"]["estimate"] == pytest.approx(2.0)
    assert out["goal|too_few_minus_never_obtained"]["estimate"] == pytest.approx(6.0)  # never touches used up x goal


# ---------------- B3: one item set for ΔL, its prefix part and the remainder ----------------

def test_prefix_readouts_share_one_item_set(items):
    main = [t for t in items if t["population"] == "main"][:20]
    scores = [{"transition_id": t["id"], "condition": arm, "L": L, "L_first": 0.0} for t in main
              for arm, L in (("aligned", 5.0), ("swapped", 2.0), ("prefix_aligned", 4.0), ("prefix_swapped", 3.5))]
    # one item loses its prefix_swapped score and carries a huge ΔL: it must leave all three readouts together
    drop = main[0]["id"]
    scores = [s for s in scores if not (s["transition_id"] == drop and s["condition"] == "prefix_swapped")]
    scores = [{**s, "L": 1000.0} if s["transition_id"] == drop and s["condition"] == "aligned" else s for s in scores]
    r = A.readouts(scores, main)
    assert r["delta_L_on_prefix_set"]["n_items"] == r["delta_L_prefix"]["n_items"] == r["delta_L_name_remainder"]["n_items"] == 19
    assert r["delta_L_on_prefix_set"]["estimate"] == pytest.approx(3.0)
    assert r["delta_L_on_prefix_set"]["estimate"] == pytest.approx(
        r["delta_L_prefix"]["estimate"] + r["delta_L_name_remainder"]["estimate"])
    assert r["delta_L"]["n_items"] == 20  # the primary-style readout keeps every item with both arms


# ---------------- B4: certificates ----------------

def test_certificate_known_answers():
    hist = [["get 1 oak logs", "Got 1 oak logs"], ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"],
            ["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"], ["get 3 iron ingot", "Got 3 iron ingot"],
            ["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"], ["get 1 iron ingot", "Got 1 iron ingot"]]
    v = K.decide(LINES, hist, "craft 4 stick using 2 oak planks")  # used up: peak 4 at step 2, emptied at step 5
    assert v.certificate == [{"name": "oak planks", "held_step": 5, "peak_step": 2}]
    v = K.decide(LINES, hist, "craft 1 minecart using 5 iron ingot")  # too few: 4 held since step 6, peak then
    assert v.certificate == [{"name": "iron ingot", "held_step": 6, "peak_step": 6}]
    v = K.decide(LINES, hist[:1], "craft 1 rail using 6 iron ingot, 1 stick")  # never obtained either
    assert v.certificate == [{"name": "iron ingot", "held_step": "never", "peak_step": "never"},
                             {"name": "stick", "held_step": "never", "peak_step": "never"}]
    assert K.decide(LINES, hist, "get 1 stick").certificate == [{"name": "stick", "line": 2}]
    # the environment's version agrees
    state = B.State(W.new_env(LINES, None)[0], [], {})
    for a, _o in hist:
        B.apply(state, a)
    _c, short = B.classify(state, [(2, "oak planks")])
    assert B.certificate(state, short) == [{"name": "oak planks", "held_step": 5, "peak_step": 2}]


def test_stored_certificates_are_checked(items):
    craft = next(t for t in items if t["action_kind"] == "craft" and t["certificate"][0]["held_step"] != "never")
    assert K.check_item(craft) == [] and V.replay_problems(craft) == []
    bad = {**craft, "certificate": [{**craft["certificate"][0], "held_step": "never"}] + craft["certificate"][1:]}
    assert any(p.startswith("certificate") for p in K.check_item(bad))
    assert "certificate does not follow from the environment's inventory after each step" in V.replay_problems(bad)
    get = next(t for t in items if t["action_kind"] == "get")
    assert K.check_item({**get, "certificate": [{"name": get["object"], "line": 0}]})
    for t in items[::11]:
        assert t["certificate"] == K.decide(t["recipe_lines"], t["history"], t["action"]).certificate


# ---------------- B5: singular and plural twins ----------------

def test_plural_twins_are_every_such_pair_in_the_database():
    t = W.tree()
    names = {W.name_of(r.output_item.item_tag.item_id) for rs in t.itemid_recipes.values() for r in rs}
    names |= {W.name_of(i.item_tag.name) for rs in t.itemid_recipes.values() for r in rs for i in r.input_items}
    valid = {n for n in names if t.is_valid_item(W.item_id(n))}
    found = {(n, m) for n in valid for m in (n + "s", n + "es") if m in valid}
    assert found == set(B.PLURAL_TWINS)


def test_plural_twin_held_is_excluded():
    lines = ("craft 1 nether brick fence using 4 nether bricks, 2 nether brick", "craft 1 nether bricks using 4 nether brick")
    state = B.State(W.new_env(lines, None)[0], [], {})
    B.apply(state, "get 4 nether brick")
    B.apply(state, "craft 1 nether bricks using 4 nether brick")
    _c, short = B.classify(state, [(4, "nether bricks"), (2, "nether brick")])
    assert [s["name"] for s in short] == ["nether bricks", "nether brick"]
    assert B.plural_ambiguous(state, short)  # 1 nether bricks held while 2 nether brick are short
    _c, short = B.classify(state, [(4, "nether bricks")])
    assert not B.plural_ambiguous(state, short)  # no nether brick held


def test_verify_reports_a_plural_twin_and_a_tag_member(items, monkeypatch):
    """Kills review 2's N4 (verify's tag-reading check off), and the same for the plural check: an item whose short
    input is declared to have a held member (or twin) must be reported."""
    it = next(t for t in items if t["short_inputs"] and K.account(t["history"]).held)
    name, held = it["short_inputs"][0]["name"], sorted(K.account(it["history"]).held)[0]
    assert V.replay_problems(it) == []
    with monkeypatch.context() as m:
        m.setitem(B.TAG_READING, name, (held,))
        assert V.replay_problems(it) == ["a short input is a tag-derived name while a Minecraft member of it is held"]
    with monkeypatch.context() as m:
        m.setitem(B._TWIN_OF, name, held)
        assert V.replay_problems(it) == ["a short input's singular or plural twin is held"]


def test_no_committed_item_has_a_plural_twin_held(items):
    for it in items:
        if it["action_kind"] == "craft":
            held = K.account(it["history"]).held
            assert not any(held.get(B._TWIN_OF.get(s["name"], ""), 0) > 0 for s in it["short_inputs"]), it["id"]


# ---------------- B6: the inventory shown after its last change ----------------

def test_inventory_shown_after_last_change():
    got, inv = ["get 1 stick", "Got 1 stick"], ["inventory", "Inventory: [stick] (1) "]
    for f in (K.inventory_shown_after_last_change, B.inventory_shown_after_last_change):
        assert f([got, inv]) and f([inv]) and f([got, inv, inv])
        assert not f([]) and not f([got]) and not f([inv, got]) and not f([got, inv, got])


def test_arithmetic_readout_leaves_out_shown_inventories(items):
    main = [t for t in items if t["population"] == "main"]
    arith = A.arithmetic_items(main)
    assert arith and all(t["needs_arithmetic"] and not t["inventory_shown_after_last_change"] for t in arith)
    assert len(arith) == sum(t["needs_arithmetic"] and not t["inventory_shown_after_last_change"] for t in main)
    assert any(t["needs_arithmetic"] and t["inventory_shown_after_last_change"] for t in main)
    for t in items[::9]:
        assert t["inventory_shown_after_last_change"] is K.inventory_shown_after_last_change(t["history"])


# ---------------- review 2's surviving mutants ----------------

def test_stale_scores_include_unknown_items(items):
    """Kills review 2's N10: a score for an item id that is not in the items file is stale."""
    t = items[0]
    good = {"transition_id": t["id"], "item_hash": S.item_hash(t)}
    assert A.stale_scores([good], [t]) == []
    assert A.stale_scores([good, {"transition_id": "tc-nowhere", "item_hash": S.item_hash(t)}], [t]) == ["tc-nowhere (unknown item)"]


def test_tag_reading_member_lists():
    """Kills review 2's N3: the member lists are fixed by the design note (section 11, A1)."""
    assert B.TAG_READING["stone crafting materials"] == B.TAG_READING["stone tool materials"] == ("blackstone", "cobblestone")
    assert B.TAG_READING["coals"] == ("coal",) and B.TAG_READING["soul fire base blocks"] == ()
    assert B.TAG_READING["warped stems"] == ("warped stem", "warped hyphae", "stripped warped stem", "stripped warped hyphae")
    assert len(B.TAG_READING["logs"]) == 8 + 32 and len(B.TAG_READING["wooden slabs"]) == 8
    assert all(W.tree().is_valid_item(W.item_id(m)) for ms in B.TAG_READING.values() for m in ms)


# ---------------- B7: tokenization ----------------

class _CharTok:
    """A tokenizer whose tokens are characters, with a chat template that joins the contents."""

    def __init__(self, merge: str | None = None):
        self.merge = merge

    def _ids(self, text):
        out, i = [], 0
        while i < len(text):
            if self.merge and text.startswith(self.merge, i):
                out.append(-1)
                i += len(self.merge)
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": self._ids(text)})()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self._ids(text) if tokenize else text


def test_tokcheck_passes_a_clean_tokenizer_and_catches_merges(items):
    it = next(t for t in items if t["action_kind"] == "craft")
    shots = V.P.fewshot_examples()
    plain = tuple(a for a in T.P.ARMS if not a.startswith("yn_"))
    assert T.item_problems(_CharTok(), it, shots, plain) == []
    # a token that spans the end of the prompt and the start of a candidate breaks joined == alone
    assert any("after the prompt than alone" in p for p in T.item_problems(_CharTok(merge=">C"), it, shots))
    # a token that spans the end of the prefix and the object name breaks the prefix split
    merged = T.item_problems(_CharTok(merge="craft minecraft"), it, shots)
    assert any("not a token prefix" in p and not p.endswith("after the prompt") for p in merged)  # alone
    assert any("not a token prefix" in p and p.endswith("after the prompt") for p in merged)  # after the prompt
    assert any("single token" in p for p in T.item_problems(_CharTok(), it, shots, arms=("yn_aligned",)))  # 'Yes' is 3 chars


def test_smoke_config_checks_tokens():
    import yaml
    cfg = yaml.safe_load((REPO / "wf" / "config" / "textcraft_score_smoke.yaml").read_text())
    assert cfg["token_check"] is True
    assert "token_check" not in yaml.safe_load((REPO / "wf" / "config" / "textcraft_score.yaml").read_text())


class _FakeScorer:
    def __init__(self, tok):
        self.mock, self._tok = False, tok

    def _encode(self, messages):
        return [self._tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)]


def test_token_report_uses_the_scorers_tokenizer(items):
    it = next(t for t in items if t["action_kind"] == "craft")
    shots = V.P.fewshot_examples()
    assert S.token_report(type("M", (), {"mock": True})(), [it], shots, ("aligned",))["skipped"]
    assert S.token_report(_FakeScorer(_CharTok()), [it], shots, ("aligned", "prefix_aligned"))["ok"]
    bad = S.token_report(_FakeScorer(_CharTok(merge="craft minecraft")), [it], shots, ("aligned", "prefix_aligned"))
    assert not bad["ok"] and bad["n_items_with_problems"] == 1


def test_smoke_run_stops_on_a_failed_token_check(monkeypatch, tmp_path):
    """With token_check set, a failed check stops the run before any score is written."""
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "FILED.txt").write_text("filed_date: x\n")
    calls = []
    monkeypatch.setattr(S, "token_report", lambda *a: calls.append(a) or {"ok": False, "examples": {"x": ["bad"]}})
    cfg = {"run_id": "textcraft_score_smoke", "max_items": 2, "token_check": True, "models": [S.CO_PRIMARY[0]]}
    assert S.run_textcraft_score(cfg) == 1 and calls
    out = tmp_path / "runs" / "textcraft_score_smoke" / "shard_00_of_01"
    assert not (out / "scoring.jsonl").exists()
    assert json.loads((out / f"token_check_{S.CO_PRIMARY[0]}.json").read_text())["ok"] is False


def test_prompt_examples_come_from_the_first_six_walks():
    """The failure example is drawn among walks 0-5 only, so adding walks (section 12, B1) left the prompt alone."""
    pairs = {p.pair_id: p for p in B.all_pairs()}
    p1, p2, p3 = B.fewshot_pair_ids()
    shots = json.loads(V.P.FEWSHOT.read_text())
    real = shots[1]
    fake = {"pair_id": p2, "goal": real["goal"], "recipe_lines": list(pairs[p2].lines), "walk_seed": 6, "history": [["a", "b"]] * 3,
            "history_len": 3, "action": "x", "reply": "y", "population": "main", "id": "tc-zzz"}
    with pytest.raises(RuntimeError):
        B.fewshot_examples({p2: [fake]})
    assert B.fewshot_examples({p2: [{**fake, "walk_seed": 5}]})[1]["action"] == "x"


# ---------------- C3: determined successes ----------------

def test_matches_line_known_answers():
    assert K.matches_line("craft 4 stick using 2 planks", 4, "stick", [(2, "oak planks")])
    assert not K.matches_line("craft 4 stick using 2 planks", 4, "stick", [(2, "oak logs")])
    assert not K.matches_line("craft 4 stick using 2 planks", 4, "stick", [(3, "oak planks")])
    assert not K.matches_line("craft 4 stick using 2 planks", 2, "stick", [(2, "oak planks")])
    assert K.matches_line("craft 1 rail using 6 iron ingot, 1 stick", 1, "rail", [(1, "stick"), (6, "iron ingot")])
    assert not K.matches_line("craft 1 white bed using 3 white wool, 3 planks", 1, "white bed", [(3, "red wool"), (3, "oak planks")])


def test_checker_decides_a_success():
    hist = [["get 6 iron ingot", "Got 6 iron ingot"], ["get 1 oak logs", "Got 1 oak logs"],
            ["craft 4 oak planks using 1 oak logs", "Crafted 4 minecraft:oak_planks"],
            ["craft 4 stick using 2 oak planks", "Crafted 4 minecraft:stick"]]
    v = K.decide(LINES, hist, "craft 1 rail using 6 iron ingot, 1 stick")
    assert v.determined_success and not v.determined_failure and v.failure_class == "success"
    assert v.reply == "Crafted 1 minecraft:rail" and v.twin == "Could not find enough items to craft minecraft:rail"
    assert v.removal == {"iron ingot": 6, "stick": 4}
    assert v.certificate == [{"name": "iron ingot", "held_step": 1, "peak_step": 1}, {"name": "stick", "held_step": 4, "peak_step": 4}]
    assert not K.decide(LINES, hist, "craft 2 rail using 6 iron ingot, 1 stick").determined_success  # not a displayed line


def test_success_items_are_checked(items):
    succ = [t for t in items if t["population"] == "success"]
    assert len(succ) == 600 and all(t["candidates"] == {"o_plus": t["reply"], "o_minus": t["twin"]} for t in succ)
    t = succ[0]
    assert K.check_item(t) == [] and V.replay_problems(t) == [] and V.structure_problems(t) == []
    assert K.check_item({**t, "removal": {}})
    assert "twin does not reproduce" in V.replay_problems({**t, "removal": {}})
    assert V.structure_problems({**t, "candidates": {"o_plus": t["twin"], "o_minus": t["reply"]}})
    assert V.structure_problems({**t, "population": "main"})


def test_usefulness_items(items):
    use = [t for t in items if t["population"] == "usefulness"]
    assert len(use) == 600 and all(B.usefulness_only(t) and t["failure_class"] == "never_obtained" for t in use)
    main_slots = {B.slot_key(t) for t in items if t["population"] == "main"}
    assert not main_slots & {B.slot_key(t) for t in use}
    assert V.couple_problems(items) == []
    assert V.structure_problems({**use[0], "population": "main"})  # population is checked


def test_named_readouts_and_competence():
    """Wiring: each named readout reads its own items; used up on the goal is left out; AUC and equivalence."""
    items, scores = [], []
    def add(i, cls, role, ov, pov, fit, inv, arith, d, pop="main", pair=None):
        t = {"id": f"t{i}", "pair_id": pair or f"p{i % 7}", "couple_id": f"c{i // 2}", "failure_class": cls, "role": role,
             "object_goal_overlap": ov, "object_partner_overlap": pov, "history_fit": fit, "length_bin": "4-8",
             "history": [["inventory", "Inventory: [x] (1) "]] if inv else [], "needs_arithmetic": arith,
             "inventory_shown_after_last_change": False, "population": pop}
        items.append(t)
        scores.extend([{"transition_id": t["id"], "condition": "aligned", "L": d}, {"transition_id": t["id"], "condition": "swapped", "L": 0.0},
                       {"transition_id": t["id"], "condition": "yn_aligned", "L": d / 10}, {"transition_id": t["id"], "condition": "yn_swapped", "L": 0.0}])
    k = 0
    for _ in range(10):
        add(k, "too_few", "goal", True, False, "aligned", True, True, 8.0); add(k + 1, "too_few", "goal", True, False, "swapped", False, True, 4.0); k += 2
        add(k, "never_obtained", "ingredient", True, False, "aligned", False, False, 3.0); add(k + 1, "never_obtained", "ingredient", True, False, "swapped", False, False, 3.0); k += 2
        add(k, "never_obtained", "ingredient", False, False, "tie", False, False, 1.0); add(k + 1, "never_obtained", "ingredient", False, False, "tie", False, False, 1.0); k += 2
        add(k, "used_up", "goal", True, False, "tie", False, True, 100.0); add(k + 1, "used_up", "goal", True, False, "tie", False, True, 100.0); k += 2
    extra = [{"id": f"u{i}", "pair_id": f"q{i}", "couple_id": f"m{i // 2}", "failure_class": "never_obtained",
              "role": "ingredient", "object_goal_overlap": False, "object_partner_overlap": False, "population": "usefulness"}
             for i in range(20)]
    scores += [{"transition_id": t["id"], "condition": c, "L": v} for t in extra for c, v in (("aligned", 3.0), ("swapped", 0.0))]
    r = A.named_readouts(scores, items, extra)
    assert r["goal_role"]["estimate"] == pytest.approx(6.0)  # used up on the goal left out
    assert r["ingredient_with_overlap"]["estimate"] == pytest.approx(3.0)
    assert r["usefulness_only"]["estimate"] == pytest.approx((20 * 1.0 + 20 * 3.0) / 40)  # main couples and the extras
    assert r["history_agreement"]["estimate"] == pytest.approx(((8.0 + 3.0) / 2 - (4.0 + 3.0) / 2) / 2)
    assert r["yes_no"]["estimate"] == pytest.approx(sum([0.8, 0.4, 0.3, 0.3, 0.1, 0.1]) / 6)
    assert r["counting"]["estimate"] == pytest.approx(6.0)
    assert r["inventory_shown"]["estimate"] == pytest.approx(8.0 - (4 + 3 + 3 + 1 + 1) / 5)
    assert "holm_p_within_model" in r["goal_role"]
    assert A.auc([2.0, 3.0], [1.0, 2.0]) == pytest.approx((1.5 + 2.0) / 4)
    succ = [{"id": f"s{i}", "pair_id": f"p{i % 7}"} for i in range(20)]
    sc = [{"transition_id": t["id"], "condition": "neutral", "L": 5.0} for t in succ]
    sc += [{"transition_id": t["id"], "condition": "neutral", "L": -1.0} for t in items]
    comp = A.competence(sc, succ, items)
    assert comp["all"]["auc"] == pytest.approx(1.0) and comp["all"]["discriminates"] and comp["gap"] == pytest.approx(6.0)
    eq = A.equivalence([s for s in scores if s["condition"] in ("aligned", "swapped")], items[:6], 100.0)
    assert eq["equivalent"] and not A.equivalence(scores, items[:6], 0.5)["equivalent"]


def test_equivalence_margins_and_competence_threshold():
    assert A.EQUIVALENCE_MARGIN == {"qwen3-8b-fewshot": 1.0, "llama3.1-8b-fewshot": 0.3}
    rng = random.Random(5)
    succ = [{"id": f"s{i}", "pair_id": f"p{i % 40}"} for i in range(400)]
    fail = [{"id": f"f{i}", "pair_id": f"p{i % 40}", "failure_class": "too_few"} for i in range(400)]
    sc = [{"transition_id": t["id"], "condition": "neutral", "L": rng.gauss(0.15, 1.0)} for t in succ]
    sc += [{"transition_id": t["id"], "condition": "neutral", "L": rng.gauss(0.0, 1.0)} for t in fail]
    comp = A.competence(sc, succ, fail)
    assert 0.5 < comp["all"]["auc"] < 0.6 and not comp["all"]["discriminates"]  # below 0.6 never counts
    assert set(comp) >= {"all", "too_few", "gap", "mean_L_success", "mean_L_failure"}


# ---------------- B7: golden rendered prompts ----------------

def test_golden_chat_prompts_match_todays_prompt(items):
    """The golden item is the first main item and is unchanged; each model's stored rendering holds every message
    of today's prompt, in order, and ends in that model's generation prompt (thinking off for Qwen)."""
    from wf.textcraft import golden_chat as G
    golden = json.loads(G.GOLDEN.read_text())
    item = G.golden_item(items)
    assert golden["item_id"] == item["id"] and set(golden["models"]) == set(G.TAGS)
    ends = {"qwen3-8b-fewshot": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "llama3.1-8b-fewshot": "<|start_header_id|>assistant<|end_header_id|>\n\n"}
    shots = V.P.fewshot_examples()
    for tag, g in golden["models"].items():
        for arm in G.ARMS:
            text, at = g[arm]["text"], 0
            for m in V.P.messages(item, arm, shots):
                at = text.index(m["content"], at) + len(m["content"])
            assert text.endswith(ends[tag]) and g[arm]["n_ids"] > 1000


def test_golden_check_catches_a_different_rendering(items):
    from wf.textcraft import golden_chat as G
    shots = V.P.fewshot_examples()
    tok = _CharTok()
    item = G.golden_item(items)
    golden = {"item_id": item["id"], "models": {"x": G.render_one(tok, item, shots)}}
    assert G.problems(tok, "x", items, shots, golden) == []
    assert G.problems(_CharTok(merge="Goal"), "x", items, shots, golden)
    assert G.problems(tok, "x", items[1:], shots, {**golden, "item_id": "tc-gone"})


def test_populations_by_history_length_and_outcome():
    """Crafts at the empty history are the empty-history stratum, never main; successes are their own population."""
    p = next(p for p in B.all_pairs() if p.pair_id not in B.fewshot_pair_ids())
    states = list(B.states_of(p, B.walk(p, 0)))
    made0 = B.items_at(p, 0, 0, states[0][1], Counter())
    assert made0 and {t["population"] for t in made0 if t["action_kind"] == "craft"} == {"empty_history"}
    later = [t for k, st in states[1:] for t in B.items_at(p, 0, k, st, Counter())]
    assert {t["population"] for t in later if t["action_kind"] == "craft"} <= {"main", "success"}
    assert all((t["population"] == "success") == t["reply"].startswith("Crafted ") for t in later)


def _uncoupled_pool():
    """More secondary-get and empty-history items than their quotas, so the seeded round-robin decides."""
    pool = []
    for pop, (_t, classes, bins) in B.QUOTAS.items():
        if pop in ("secondary_get", "empty_history"):
            for c in classes:
                for b in bins:
                    pool += [{"id": f"{pop}-{b}-{k:03d}", "pair_id": f"p{k % 12:03d}", "population": pop, "failure_class": c,
                              "length_bin": b} for k in range(300)]
    return pool


def test_uncoupled_selection_golden():
    """Kills a seed that drops the bin and a round-robin that never shuffles: the draw binds here."""
    import hashlib
    chosen = B.select(_uncoupled_pool())
    assert Counter(t["population"] for t in chosen) == Counter({"secondary_get": 400, "empty_history": 100})
    digest = hashlib.sha256(json.dumps([t["id"] for t in chosen]).encode()).hexdigest()
    assert digest == GOLDEN_UNCOUPLED, digest


GOLDEN_UNCOUPLED = "e7def9625db116269ad3b6d54bc168c8d3ad51b3086feb610670675414d35c02"


def test_couple_check_needs_mirrored_measures(items, monkeypatch):
    """A fit measure that does not change sign with the goals would leave a couple unbalanced: verify reports it."""
    main = [t for t in items if t["population"] == "main"]
    a = main[0]
    b = next(t for t in main if t["couple_id"] == a["couple_id"] and t is not a)
    assert V.couple_problems([a, b]) == []
    monkeypatch.setattr(F, "measures", lambda *args: {m: [1, 0] for m in F.MEASURES})
    assert V.couple_problems([a, b]) == [f"couple {a['couple_id']}: fit measures not mirrored"]


def test_regression_interval_is_the_intercepts():
    assert A.ols_intercept([(2.0 + 1.5 * s, (s,)) for s in (-1, 0, 1, 1, 0)]) == pytest.approx(2.0)
    items, scores = [], []
    for i in range(60):
        s = (-1, 0, 1)[i % 3]
        fm = {m: [0, 0] for m in F.MEASURES}
        fm["crafts"] = [1, 0] if s > 0 else [0, 1] if s < 0 else [0, 0]
        items.append({"id": f"t{i}", "pair_id": f"p{i % 10}", "fit_measures": fm})
        scores += [{"transition_id": f"t{i}", "condition": "aligned", "L": 2.0 + 5.0 * s},
                   {"transition_id": f"t{i}", "condition": "swapped", "L": 0.0}]
    r = A.fit_regression(scores, items)
    assert r["ci"] == pytest.approx([2.0, 2.0]) and r["slopes"]["crafts"] == pytest.approx(5.0)


def test_builder_drops_a_plural_reading_item():
    """p071 after "get 4 nether brick": the fence needs 4 nether bricks (none held) while 4 nether brick are held."""
    p = {x.pair_id: x for x in B.all_pairs()}["p071"]
    st = B.State(W.new_env(p.lines, None)[0], [], {})
    B.apply(st, "get 4 nether brick")
    excluded = Counter()
    made = {t["action"] for t in B.items_at(p, 0, 1, st, excluded)}
    assert excluded["plural_reading"] == 1
    assert "craft 6 nether brick fence using 4 nether bricks, 2 nether brick" not in made
    assert "craft 1 red nether bricks using 2 nether wart, 2 nether brick" in made  # nether brick is not short here


def test_verify_reports_a_broken_mirrored_couple(items):
    use = [t for t in items if t["population"] == "usefulness"]
    by = {}
    for t in use:
        by.setdefault(t["couple_id"], []).append(t)
    a, b = next(v for v in by.values() if any(B.fit_profile(v[0])))
    assert V.couple_problems([a, b]) == []
    assert V.couple_problems([a, {**b, "fit_measures": a["fit_measures"]}]) == [f"couple {a['couple_id']}"]
    assert V.couple_problems([a, {**b, "length_bin": "0"}]) and V.couple_problems([a, {**b, "couple_id": a["couple_id"], "id": "x"}])


def test_equivalence_reads_the_90_percent_interval():
    rng = random.Random(11)
    items = [{"id": f"t{i}", "pair_id": f"p{i % 30}"} for i in range(300)]
    scores = []
    for t in items:
        scores += [{"transition_id": t["id"], "condition": "aligned", "L": rng.gauss(0.3, 2.0)},
                   {"transition_id": t["id"], "condition": "swapped", "L": 0.0}]
    by = {s["transition_id"]: s["L"] for s in scores if s["condition"] == "aligned"}
    boots = A._cluster_boots(lambda xs: sum(xs) / len(xs), [by[t["id"]] for t in items], [t["pair_id"] for t in items])
    hi90, hi95 = A._pct(boots, 0.95), A._pct(boots, 0.975)
    assert hi90 < hi95 and -A._pct(boots, 0.05) < hi90
    margin = (hi90 + hi95) / 2  # inside the 95% upper end, outside the 90% one
    eq = A.equivalence(scores, items, margin)
    assert eq["equivalent"] and eq["ci90"] == pytest.approx([A._pct(boots, 0.05), hi90])


def test_verify_reports_a_success_label_on_a_failure(items):
    fail = next(t for t in items if t["population"] == "main")
    assert "reply and class disagree on success" in V.replay_problems({**fail, "failure_class": "success"})
    succ = next(t for t in items if t["population"] == "success")
    assert "reply and class disagree on success" in V.replay_problems({**succ, "failure_class": "too_few"})
