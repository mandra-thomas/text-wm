"""Regressions from the four 2026-11-06 reviews. Real strings are quoted from real data."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

import wf.runs as runs_mod
import wf.scoring.logodds as logodds_mod
import wf.tools.assemble_pilot_report as assemble
import wf.tools.export_review as export_review
import wf.tools.verify_models as verify_models
import wf.tools.verify_real_env as verify_env
import wf.util as util
from wf.agent.foresight import ForesightConfig, foresight_prediction
from wf.agent.react import AgentClient, run_episode
from wf.analysis.stats import pair_permutation
from wf.corruption.schedule import schedule
from wf.data.candidates import _setup_plan, attach_candidates, twin_available
from wf.data.goals import swap_feasible, swapped_goal
from wf.data.transitions import (
    action_type,
    build_transitions_from_rollout,
    context_determined_label,
    filter_pilot_a_items,
    generate_alternatives,
    history_facts,
    pick_swap,
    success_exclusion_counts,
)
from wf.env.scene import Scene, scan_scene
from wf.env.server import AlfworldClient, GameState
from wf.env.sim import World
from wf.scoring.logodds import LogoddsScorer, build_messages, score_transition
from wf.scoring.prompts import WM_FEWSHOT_SET0, condition_task_line
from wf.util import append_jsonl, read_jsonl, write_jsonl

MOCK_KEYS = ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE")
NOOP = "Nothing happens."
REPO = Path(__file__).resolve().parents[1]
TRAITS = {  # flags as they appear in real initial_state.pddl files
    "pickupable": {"pot", "houseplant_not_pickupable", "candle", "fork", "egg", "mug", "cellphone", "bowl", "keychain"},
    "heatable": {"egg", "mug"},
    "coolable": {"egg", "mug", "bowl", "pot"},
    "cleanable": {"egg", "mug", "bowl", "pot", "fork"},
    "can_contain": {"cabinet": {"pot", "mug", "egg", "candle", "cellphone", "keychain", "houseplant"},
                    "countertop": {"fork", "mug"}, "drawer": {"cellphone", "fork", "keychain"}},
}


@pytest.fixture
def mock_runs(monkeypatch, tmp_path):
    for key in MOCK_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    return tmp_path


@pytest.fixture
def real_runs(monkeypatch, tmp_path):
    """Real mode at a clean commit, with the dependency gate on."""
    for key in MOCK_KEYS + ("WF_MOCK_CONTINUE",):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    (tmp_path / "FILED.txt").write_text("filed_date: x\n")
    monkeypatch.setattr(runs_mod, "git_sha", lambda: "abc1234")
    monkeypatch.setattr(runs_mod, "git_dirty", lambda: False)
    return tmp_path


def finished(root: Path, run_id: str, status: str = "PASS", sha: str = "abc1234", inputs: dict | None = None) -> Path:
    d = root / run_id
    d.mkdir(exist_ok=True)
    (d / "STATUS.md").write_text(f"# STATUS — {run_id} — {status}\n\nbody\n")
    (d / "git_sha").write_text(sha + "\n")
    (d / "inputs.json").write_text(json.dumps({"inputs": inputs or {}, "git_dirty": False}))
    (d / "REPORT.md").write_text(f"# REPORT — {run_id}\n")
    return d


# --- swapped goals ---------------------------------------------------------------------------------


@pytest.mark.parametrize("task, candidate, n, ok", [
    ("put a hot cup in cabinet.", "pot", 1, False),  # real swap: "put a hot pot in cabinet."
    ("clean some spatula and put it in cabinet.", "candle", 1, False),  # real: "clean some candle ..."
    ("cool some tomato and put it in countertop.", "fork", 1, False),  # real: "cool some fork ..."
    ("put a hot cup in cabinet.", "houseplant", 1, False),  # real: "put a hot houseplant in cabinet."
    ("examine the bowl with the desklamp.", "desklamp", 1, False),  # real: "examine the desklamp with the desklamp."
    ("put a hot cup in cabinet.", "egg", 1, True),
    ("clean some spatula and put it in cabinet.", "mug", 1, True),
    ("cool some tomato and put it in countertop.", "mug", 1, True),
    ("put some saltshaker on cabinet.", "candle", 1, True),
    ("look at bowl under the desklamp.", "cellphone", 1, True),
    ("find two soapbar and put them in cabinet.", "keychain", 1, False),  # only one to pick up
    ("put two pillow in drawer.", "keychain", 2, True),
    ("put some saltshaker on countertop.", "candle", 1, False),  # the game says countertop cannot hold it here
    ("put some saltshaker on cabinet.", "houseplant", 1, False),  # the cabinet could hold it, but it cannot be picked up
])
def test_a_swap_must_be_achievable_by_the_games_own_rules(task, candidate, n, ok):
    assert swap_feasible(f"Your task is to: {task}", candidate, TRAITS, n) is ok


def test_the_target_itself_is_never_the_swap():
    traits = {**TRAITS, "pickupable": TRAITS["pickupable"] | {"desklamp"}}
    assert swap_feasible("Your task is to: examine the bowl with the desklamp.", "desklamp", traits, 1) is False


def test_swapped_goal_takes_only_feasible_objects_and_none_when_there_is_none():
    objects = ["candle", "egg", "fork"]
    fits = lambda o: o == "egg"
    assert swapped_goal(objects, "mug", feasible=fits) == "egg"
    assert swapped_goal(objects, "mug", feasible=lambda o: False) is None


def test_built_items_only_carry_feasible_swaps(mock_runs):
    client = AlfworldClient()
    client.create()
    seen = 0
    for gid in ("train/4", "train/2", "train/8"):  # heat, clean and pick-two tasks
        actions = ["go to countertop 1", "go to drawer 1"]
        for t in build_transitions_from_rollout(client, gid, "train", actions, rng_seed=1):
            scene = scan_scene(client, gid, "train")
            if t["swapped_goal_object"]:
                seen += 1
                assert t["swap_feasible"] is True
                assert swap_feasible(t["task_line"], t["swapped_goal_object"], scene.traits,
                                     len(scene.instances(t["swapped_goal_object"])))
            else:
                assert t["swap_feasible"] is False and t["exclusion_reason"]
    assert seen


def _real_game_tree(root: Path, solvable: int = 2179, pddl: str = "") -> None:
    configs = root / "w2w" / "AgentGym" / "agentenv-alfworld" / "configs"
    configs.mkdir(parents=True)
    rows = []
    for i in range(solvable):
        rows.append({"item_id": "x", "task_type": f"t{i}", "task_id": "trial"})
        game = root / "alf" / "json_2.1.1" / "train" / f"t{i}" / "trial"
        game.mkdir(parents=True)
        (game / "game.tw-pddl").write_text(json.dumps({"solvable": True}))
        (game / "initial_state.pddl").write_text(pddl)
    (configs / "mappings_train.json").write_text(json.dumps(rows))


REAL_PDDL = """
        (objectType Mug_bar__minus_02_dot_99_bar__plus_01_dot_00 MugType)
        (canContain CabinetType MugType)
        (canContain CounterTopType CreditCardType)
        (pickupable Mug_bar__minus_02_dot_99_bar__plus_01_dot_00)
        (pickupable CreditCard_bar__minus_03_dot_11_bar__plus_01_dot_00)
        (heatable Mug_bar__minus_02_dot_99_bar__plus_01_dot_00)
        (cleanable Mug_bar__minus_02_dot_99_bar__plus_01_dot_00)
"""


def test_traits_come_from_the_games_own_file(monkeypatch, tmp_path):
    monkeypatch.delenv("WF_MOCK_ENV", raising=False)
    _real_game_tree(tmp_path, 3, REAL_PDDL)
    monkeypatch.setenv("WORD2WORLD_ROOT", str(tmp_path / "w2w"))
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path / "alf"))
    traits = AlfworldClient().object_traits("train/1", "train")
    assert traits["pickupable"] == {"mug", "creditcard"}
    assert traits["heatable"] == {"mug"} and traits["coolable"] == set() and traits["cleanable"] == {"mug"}
    assert traits["can_contain"] == {"cabinet": {"mug"}, "countertop": {"creditcard"}}


def test_a_changed_game_population_stops_the_run(monkeypatch, tmp_path):
    monkeypatch.delenv("WF_MOCK_ENV", raising=False)
    _real_game_tree(tmp_path, 3)
    monkeypatch.setenv("WORD2WORLD_ROOT", str(tmp_path / "w2w"))
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path / "alf"))
    with pytest.raises(RuntimeError, match="3 solvable games, expected 2179"):
        AlfworldClient().list_games("train")


def test_missing_game_paths_stop_the_run(monkeypatch, tmp_path):
    for key in ("WF_MOCK_ENV", "WF_WORD2WORLD_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORD2WORLD_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path / "nowhere"))
    with pytest.raises(RuntimeError, match="cannot find the train game list"):
        AlfworldClient().list_games("train")


# --- labels and history facts ---------------------------------------------------------------------


def test_arriving_at_a_closed_receptacle_is_news():
    obs = "You arrive at drawer 2. The drawer 2 is closed."  # real item, labelled determined before
    assert context_determined_label("You are in the middle of a room.", "go to drawer 2", obs) is False
    seen = "You arrive at drawer 2. The drawer 2 is closed.\n> go to desk 1\nYou arrive at desk 1. On the desk 1, you see nothing."
    assert context_determined_label(seen, "go to drawer 2", obs) is True


def test_other_success_labels_are_unchanged():
    assert context_determined_label("", "take mug 1 from cabinet 2", "You pick up the mug 1 from the cabinet 2.") is True


def test_contents_are_filed_under_the_receptacle_the_line_names():
    # Real trajectory 2452, at cabinet 1, which shares a spot with cabinet 2.
    text = ("You arrive at cabinet 1. The cabinet 1 is closed.\n"
            "You open the cabinet 2. The cabinet 2 is open. In it, you see a plate 2.")
    facts = history_facts(text)
    assert facts.contents == {"cabinet 2": {"plate 2"}}


def test_a_move_does_not_invent_contents_for_a_receptacle_never_seen():
    facts = history_facts("You move the mug 1 to the cabinet 4.")
    assert "cabinet 4" not in facts.contents
    seen = history_facts("You arrive at cabinet 4. On the cabinet 4, you see nothing.\nYou move the mug 1 to the cabinet 4.")
    assert seen.contents["cabinet 4"] == {"mug 1"}


@pytest.mark.parametrize("action, kind", [("use desklamp 1", "use"), ("slice apple 1 with knife 1", "slice"), ("go to shelf 1", "go to")])
def test_real_verbs_have_their_own_action_type(action, kind):
    assert action_type(action) == kind


# --- objects in two receptacles -------------------------------------------------------------------


def test_scan_keeps_every_receptacle_listing_an_object(mock_runs):
    client = AlfworldClient()
    client.create()
    scene = scan_scene(client, "train/0")
    assert scene.locations["pen 1"] == {"drawer 1", "sinkbasin 1"}  # real: pan 1 on countertop 2 and stoveburner 1
    assert scene.home("pen 1") == "drawer 1"


def _state(task: str = "Your task is to: put a pan in cabinet.") -> GameState:
    return GameState("train/1", "train", task, "obs", ["look"])


def test_alternatives_never_call_a_true_location_away():
    scene = Scene(receptacles=["countertop 2", "stoveburner 1", "shelf 1", "drawer 1"],
                  locations={"pan 1": {"countertop 2", "stoveburner 1"}})
    alts = generate_alternatives(_state(), "look", "pan", "cabinet", scene, random.Random(0))
    claimed = [a for a, kind, _ in alts if a.startswith("take pan 1 from") and kind == "validity"]
    assert "take pan 1 from countertop 2" in claimed  # the chosen home, tried once
    assert "take pan 1 from stoveburner 1" not in claimed


def test_success_plan_uses_any_receptacle_that_lists_the_object():
    scene = Scene(receptacles=["countertop 2", "stoveburner 1"], locations={"pan 1": {"countertop 2", "stoveburner 1"}})
    assert _setup_plan("take pan 1 from stoveburner 1", scene) == ["go to stoveburner 1"]
    assert _setup_plan("take pan 1 from countertop 2", scene) == ["go to countertop 2"]


def test_swap_bar_covers_every_receptacle_listing_an_object():
    locations = {"pan 1": {"countertop 2", "stoveburner 1"}, "cup 1": {"shelf 1"}}
    assert swapped_goal(["pan", "cup"], "mug", receptacle_in_question="stoveburner 1", locations=locations) == "cup"


# --- success twins --------------------------------------------------------------------------------


def _open_failure(history):
    return {"game_id": "train/0", "split": "train", "action": "open drawer 1", "true_obs": NOOP, "history_pairs": history}


def test_a_twin_cannot_contradict_what_the_agent_has_done(mock_runs):
    client = AlfworldClient()
    client.create()
    # Real ckpt_108: o_plus offered "In it, you see a toiletpaper 1." with that object in hand.
    history = [("open drawer 1", "You open the drawer 1. The drawer 1 is open. In it, you see a mug 1."),
               ("take mug 1 from drawer 1", "You pick up the mug 1 from the drawer 1.")]
    item = attach_candidates(client.sandbox(), _open_failure(history))
    assert item["candidates"] is None and item["exclusion_reason"] == "twin_contradicts_state"
    item = {"game_id": "train/0", "split": "train", "action": "go to drawer 1", "true_obs": NOOP, "history_pairs": history}
    assert attach_candidates(client.sandbox(), item)["candidates"] is None


def test_a_corruption_cannot_contradict_the_agents_state(mock_runs):
    client = AlfworldClient()
    client.create()
    state = client.reset("train/0", "train")
    for action in ("go to drawer 1", "open drawer 1", "take mug 1 from drawer 1"):
        state = client.step(state, action)
    cfg = ForesightConfig("corrupted", arm="favorable", corrupt_fn=lambda t, valid: True)
    out = foresight_prediction(client, state, "open drawer 1", [a for a, _ in state.history], 3, cfg)
    assert out["corrupted"] is False and out["corruption_unavailable"] is True


def test_a_twin_survives_actions_that_changed_nothing(mock_runs):
    client = AlfworldClient()
    client.create()
    failed = [("take mug 1 from drawer 1", NOOP), ("go to shelf 1", "You arrive at shelf 1.")]
    assert attach_candidates(client.sandbox(), _open_failure(failed))["candidates"] is not None


def test_done_can_never_be_corrupted(mock_runs):
    client = AlfworldClient()
    client.create()
    assert twin_available(client.sandbox(), "train/0", "train", "done", [], {}) is False


# --- simulator (rule 1) ---------------------------------------------------------------------------


def test_sim_prints_real_text():
    world = World()
    assert world.step("go to drawer 1") == "You arrive at drawer 1. The drawer 1 is closed."
    assert world.step("open drawer 1") == "You open the drawer 1. The drawer 1 is open. In it, you see a mug 1, and a pen 1."
    assert world.step("go to countertop 1") == "You arrive at countertop 1. On the countertop 1, you see a apple 1."
    assert world.step("go to drawer 1") == "You arrive at drawer 1. The drawer 1 is open. In it, you see a mug 1, and a pen 1."
    assert World().step("look") == "You are in the middle of a room. Looking quickly around you, you see nothing."
    world = World(shared=("drawer 1", "countertop 1"))
    world.step("go to countertop 1")
    assert world.step("look") == "You are facing the drawer 1, and countertop 1. Next to it, you see nothing."


def test_sim_heats_cools_cleans_and_examines_with_real_text():
    world = World()
    for action in ("go to drawer 1", "open drawer 1", "take mug 1 from drawer 1", "go to microwave 1"):
        world.step(action)
    assert world.step("heat mug 1 with microwave 1") == "You heat the mug 1 using the microwave 1."
    world.step("go to fridge 1")
    assert world.step("cool mug 1 with fridge 1") == "You cool the mug 1 using the fridge 1."
    world.step("go to sinkbasin 1")
    assert world.step("clean mug 1 with sinkbasin 1") == "You clean the mug 1 using the sinkbasin 1."
    assert world.step("examine mug 1") == "This is a normal mug 1. In it, you see nothing."


def test_sim_refuses_tools_on_objects_the_game_forbids():
    world = World()
    world.hold, world.loc = "pen 1", "microwave 1"
    assert world.step("heat pen 1 with microwave 1") == NOOP


def test_sim_wins_on_use_for_look_tasks_and_needs_the_prep_for_hot_tasks():
    world = World("Your task is to: look at mug under the desklamp.", "mug", "desklamp 1")
    world.hold, world.loc = "mug 1", "sidetable 1"
    assert world.step("use desklamp 1") == "You turn on the desklamp 1."
    assert world.won
    world = World("Your task is to: put a hot mug in shelf.", "mug", "shelf 1")
    world.hold, world.loc = "mug 1", "shelf 1"
    world.step("move mug 1 to shelf 1")
    assert not world.won
    world = World("Your task is to: put a hot mug in shelf.", "mug", "shelf 1")
    world.done_prep, world.hold, world.loc = {"mug 1"}, "mug 1", "shelf 1"
    world.step("move mug 1 to shelf 1")
    assert world.won


def test_picking_up_clears_only_one_listing():
    world = World()
    world.loc = "sinkbasin 1"
    world.step("take pen 1 from sinkbasin 1")
    assert "pen 1" in world.rec["drawer 1"]["items"]  # real trajectory 2741


# --- few-shot prompt -------------------------------------------------------------------------------


def test_few_shot_examples_use_real_templates():
    histories = [shot["history"] for shot in WM_FEWSHOT_SET0]
    observations = [shot["observation"] for shot in WM_FEWSHOT_SET0]
    assert observations.count(NOOP) == 2 and len(observations) == 3
    for line in "\n".join(histories).splitlines():
        if line.startswith("You arrive at"):
            assert "is closed." in line or "you see" in line  # 0 of 295 real arrivals stop at the name
        if line.startswith("You open the"):
            assert " is open. " in line  # 64 of 64 real opens say so
    assert not any(" an " in text for text in histories)  # 0 uses in every real context


# --- scoring ---------------------------------------------------------------------------------------


class _FakeTok:
    eos_token_id, pad_token_id = 151645, 151643

    def decode(self, ids, skip_special_tokens=True):
        return "Nothing happens."


class _FakeTensor:
    shape = (1, 2)

    def to(self, device):
        return self

    def __getitem__(self, index):
        return [0]


def _fake_scorer(monkeypatch, configured):
    """The real free_generate, without torch or a model."""
    grad = type("NoGrad", (), {"__enter__": lambda self: None, "__exit__": lambda self, *a: None})
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=grad))
    scorer = LogoddsScorer.__new__(LogoddsScorer)
    calls: dict = {}

    class Model:
        generation_config = type("G", (), {"eos_token_id": configured})()

        def parameters(self):
            return iter([type("P", (), {"device": "cpu"})()])

        def generate(self, **kwargs):
            calls.update(kwargs)
            return _FakeTensor()

    scorer.mock, scorer._model, scorer._tok = False, Model(), _FakeTok()
    monkeypatch.setattr(scorer, "_encode", lambda messages: _FakeTensor())
    return scorer, calls


def test_generation_stops_at_every_end_of_turn_token(monkeypatch):
    scorer, calls = _fake_scorer(monkeypatch, 151643)  # the Qwen2.5 checkpoint's config lacks <|im_end|>
    scorer.free_generate([{"role": "user", "content": "x"}])
    assert calls["eos_token_id"] == [151643, 151645]


def test_stop_tokens_do_not_need_torch():
    scorer = LogoddsScorer.__new__(LogoddsScorer)
    scorer._tok = type("T", (), {"eos_token_id": 128009, "pad_token_id": None})()
    scorer._model = type("M", (), {"generation_config": type("G", (), {"eos_token_id": [128001, 128008]})()})()
    assert scorer._stop_tokens() == [128001, 128008, 128009]


def test_checkpoint_accuracy_generates_full_observations(mock_runs, monkeypatch):
    seen = []
    real = LogoddsScorer.free_generate

    def spy(self, messages, max_new_tokens=64):
        seen.append(max_new_tokens)
        return real(self, messages, max_new_tokens)

    monkeypatch.setattr(LogoddsScorer, "free_generate", spy)
    runs_mod.run_t0_ckpt_accuracy({"run_id": "t0_ckpt_accuracy", "n_items": 5})
    assert seen and set(seen) == {256}  # 3 of 204 real true_obs reach 64 tokens


def _pair(monkeypatch):
    monkeypatch.setenv("WF_MOCK_SCORE", "1")
    return LogoddsScorer("x", model_tag="x")


def test_a_non_finite_score_stops_the_run(monkeypatch):
    scorer = _pair(monkeypatch)
    monkeypatch.setattr(scorer, "_teacher_force_logprob", lambda m, c: (float("nan"), 1))
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score_pair("t", "aligned", [], "a", "b")


def test_nan_does_not_look_significant():
    p = pair_permutation([1.0, float("nan"), 2.0])
    assert p != p


def test_score_cache_sees_the_prompt_and_candidates(monkeypatch):
    scorer = _pair(monkeypatch)
    first = scorer.score_pair("t_1", "aligned", [{"role": "user", "content": "a"}], "You pick up.", NOOP)
    assert scorer.score_pair("t_1", "aligned", [{"role": "user", "content": "a"}], "You pick up.", NOOP) is first
    assert scorer.score_pair("t_1", "aligned", [{"role": "user", "content": "b"}], "You pick up.", NOOP) is not first
    assert scorer.score_pair("t_1", "aligned", [{"role": "user", "content": "a"}], "You move.", NOOP) is not first


def test_a_missing_swap_is_an_error_not_the_aligned_goal():
    with pytest.raises(ValueError, match="swapped goal"):
        condition_task_line("swapped", "Your task is to: put a mug in shelf.", None)


def test_placebo_swap_uses_the_real_swap_rules(mock_runs):
    client = AlfworldClient()
    client.create()
    items = [t for t in build_transitions_from_rollout(client, "train/1", "train", ["go to cabinet 1", "go to fridge 1"], rng_seed=1)
             if t["swapped_goal_object"]]
    scene = scan_scene(client, "train/1")
    others = [verify_models._with_second_swap(t, client, random.Random(3)) for t in items]
    pairs = [(t, o["swapped_goal_object"]) for t, o in zip(items, others) if o]
    assert pairs
    for item, name in pairs:
        assert name != item["swapped_goal_object"] and name not in item["action"]
        assert swap_feasible(item["task_line"], name, scene.traits, len(scene.instances(name)))


def test_tokenizer_check_uses_the_fine_tuned_prompt(monkeypatch):
    seen = []

    def spy(item, condition, ft_prompt=False):
        seen.append(ft_prompt)
        return [{"role": "user", "content": "x"}]

    class Tok:
        def apply_chat_template(self, *a, **k):
            return "prompt"

        def __call__(self, text, add_special_tokens=False):
            return type("R", (), {"input_ids": [1]})()

    ids = type("Ids", (), {"tolist": lambda self: [1]})()
    scorer = type("S", (), {"mock": False, "_tok": Tok(), "_encode": lambda self, m: [ids]})()
    monkeypatch.setattr(verify_models, "build_messages", spy)
    verify_models.tokenizer_checks(scorer, {"candidates": {"o_plus": "a", "o_minus": "b"}}, True)
    assert seen == [True]


# --- A ---------------------------------------------------------------------------------------------


def _write_transitions(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t) + "\n" for t in items))


def _item(i: int, **extra) -> dict:
    base = {"id": f"t_{i}", "game_id": f"train/{i % 3}", "split": "train", "state_idx": i, "action": f"take mug {i} from shelf 1",
            "history_text": "h", "history_pairs": [], "task_line": "Your task is to: put a mug in shelf.",
            "swapped_task_line": "Your task is to: put a cup in shelf.", "goal_object": "mug", "swapped_goal_object": "cup",
            "true_obs": NOOP, "outcome_class": "noop", "context_determined": True, "shared_location": False,
            "history_goal_agnostic": True, "history_len": 1, "targets_goal_object": True,
            "candidates": {"o_plus": "You pick up.", "o_minus": NOOP}}
    return {**base, **extra}


def _pick_up(i: int, **extra) -> dict:
    return _item(i, **{"true_obs": "You pick up the mug 1.", "outcome_class": "state_change", **extra})


def test_excluded_successes_stay_out_of_A_and_come_back_only_in_the_sensitivity_check(mock_runs):
    items = [_item(i) for i in range(6)] + [_item(6, shared_location=True)]
    items += [_pick_up(10), _pick_up(11), _pick_up(12, shared_location=True), _pick_up(13, outcome_class="unparseable")]
    _write_transitions(mock_runs / "pilotA_transitions" / "transitions.jsonl", items)
    runs_mod.run_pilotA_score({"run_id": "pilotA_score"})
    metrics = json.loads((mock_runs / "pilotA_score" / "metrics.json").read_text().replace("NaN", "null"))
    report = metrics["reports"]["qwen3-8b-fewshot"]
    assert report["A"]["n_succ"] == 2 and report["A"]["n_fail"] == 6
    assert report["with_shared_location"]["A"]["n_succ"] == 3 and report["with_shared_location"]["A"]["n_fail"] == 7
    assert metrics["exclusions"]["successes"] == {"shared_location": 1, "unparseable": 1}


def test_success_exclusions_are_counted():
    items = [_pick_up(1, context_determined=False), _pick_up(2, shared_location=True), _pick_up(3), _item(4)]
    assert success_exclusion_counts(items) == {"not_context_determined": 1, "shared_location": 1}


# --- RQ3 calibration -------------------------------------------------------------------------------


def _episode(cond: str, steps: list[tuple[str, bool]], game: str = "train/0") -> dict:
    trace = [{"proposal": a, "proposal_valid": v, "executed": a, "real_obs": "ok" if v else NOOP,
              "proposal_on_expert_path": None, "on_expert_path": True, "decision_critical": False, "followed": True}
             for a, v in steps]
    return {"game_id": game, "split": "train", "condition": cond, "arm": None, "retention": "discarded", "steps": len(trace),
            "success": False, "trace": trace, "n_corrupted": 0, "n_corruption_unavailable": 0}


def _eligibility(root: Path, episodes: list[dict]) -> dict:
    write_jsonl(root / "pilotB_headroom" / "episodes.jsonl", episodes)
    assert runs_mod.run_pilotB_eligibility({"run_id": "pilotB_eligibility"}) == 0
    return json.loads((root / "pilotB_eligibility" / "metrics.json").read_text())


def test_calibration_reads_oracle_episodes_only(mock_runs):
    oracle = _episode("oracle", [("go to drawer 1", True)] * 8 + [("take mug 1 from drawer 1", False)] * 2)
    none = _episode("none", [("take mug 1 from drawer 1", False)] * 30)
    cal = _eligibility(mock_runs, [oracle, none])
    assert cal["expected_steps"] == 10
    assert cal["invalid_proposal_rate"] == pytest.approx(0.2)


def test_steps_with_no_twin_do_not_count_as_corruptible(mock_runs):
    oracle = _episode("oracle", [("go to drawer 1", True)] * 8 + [("take mug 1 from drawer 1", False), ("done", False)])
    cal = _eligibility(mock_runs, [oracle])
    assert cal["invalid_proposal_rate"] == pytest.approx(0.2)
    assert cal["corruptible_proposal_rate"] == pytest.approx(0.1)
    assert cal["corruption_mix"]["favorable"] == {"take": 1}  # `done` is never corruptible
    assert cal["corruption_mix"]["unfavorable"] == {"go to": 8}


def test_both_arms_share_the_lowest_achievable_target(mock_runs):
    # 1 corruptible and 8 valid per episode: the unfavorable arm must drop to 1 error, or arms differ.
    oracle = _episode("oracle", [("go to drawer 1", True)] * 8 + [("take mug 1 from drawer 1", False)] * 2)
    oracle["trace"][-1]["proposal"] = "done"
    cal = _eligibility(mock_runs, [oracle])
    assert cal["common_target"] == pytest.approx(1.0)
    assert cal["p_favorable"] == pytest.approx(1.0)
    assert cal["p_unfavorable"] == pytest.approx(1 / 8)
    assert cal["expected_corruptions"]["favorable"] == pytest.approx(cal["expected_corruptions"]["unfavorable"])
    assert cal["arms_below_target"] == {"favorable": 1.0}
    assert "too few eligible steps" in (mock_runs / "pilotB_eligibility" / "STATUS.md").read_text()


def test_expert_share_is_measured_on_proposals(mock_runs):
    oracle = _episode("oracle", [("go to drawer 1", True)] * 4)
    for step, on_path in zip(oracle["trace"], (True, False, False, False)):
        step["proposal_on_expert_path"], step["on_expert_path"] = on_path, True
    assert _eligibility(mock_runs, [oracle])["expert_among_valid"] == pytest.approx(0.25)


def test_episode_records_the_proposal_expert_label_and_the_rate(mock_runs):
    client = AlfworldClient()
    client.create()
    fc = ForesightConfig("corrupted", arm="favorable", corrupt_fn=schedule("train/0", 1, "favorable", 0.4), p_arm=0.4)
    ep = run_episode(client, AgentClient(), "train/0", "train", "t", fc, step_cap=2)
    assert ep["p_arm"] == 0.4
    assert "proposal_on_expert_path" in ep["trace"][0]


def _headroom_episodes(none: float, oracle: float, games: int = 8) -> list[dict]:
    eps = []
    for g in range(games):
        for cond, ret, rate in (("none", "discarded", none), ("oracle", "discarded", oracle),
                                ("oracle", "retained", oracle), ("validity_only", "discarded", none)):
            ep = _episode(cond, [("go to drawer 1", True)], f"train/{g}")
            ep["retention"], ep["success"] = ret, g < rate * games
            eps.append(ep)
    return eps


def _headroom(monkeypatch, root, episodes, run_id="pilotB_headroom") -> int:
    monkeypatch.setattr(runs_mod, "_run_jobs", lambda *a, **k: episodes)
    return runs_mod.run_pilotB_headroom({"run_id": run_id, "n_games": 8})


def test_recovery_needs_a_positive_oracle_gain(mock_runs, monkeypatch):
    eps = _headroom_episodes(0.5, 0.25)
    _headroom(monkeypatch, mock_runs, eps)
    assert "recovery=nan" in (mock_runs / "pilotB_headroom" / "STATUS.md").read_text()


def test_retention_rule_compares_the_same_games(mock_runs, monkeypatch):
    (mock_runs / "calibrated.yaml").write_text("p_favorable: 0.5\np_unfavorable: 0.5\n")

    def ep(game, arm, ret, success):
        return {"game_id": game, "arm": arm, "retention": ret, "success": success, "steps": 10, "condition": "corrupted",
                "n_corrupted": 0, "n_corruption_unavailable": 0, "trace": []}

    eps = [ep("g1", "favorable", "discarded", True), ep("g1", "favorable", "retained", True),
           ep("g2", "favorable", "retained", False),  # lost from the other cell
           ep("g1", "unfavorable", "discarded", True), ep("g1", "unfavorable", "retained", True)]
    monkeypatch.setattr(runs_mod, "_run_jobs", lambda *a, **k: eps)
    runs_mod.run_pilotB_retention_minirun({"run_id": "pilotB_retention_minirun", "n_games": 2})
    assert "retention_shift=0.0" in (mock_runs / "pilotB_retention_minirun" / "STATUS.md").read_text()


def test_minirun_records_its_rates(mock_runs, monkeypatch):
    (mock_runs / "calibrated.yaml").write_text("p_favorable: 0.5\np_unfavorable: 0.25\n")
    runs_mod.run_pilotB_retention_minirun({"run_id": "pilotB_retention_minirun", "n_games": 1})
    d = mock_runs / "pilotB_retention_minirun"
    assert {e["p_arm"] for e in read_jsonl(d / "episodes.jsonl")} == {0.5, 0.25}
    assert json.loads((d / "config.json").read_text())["calibrated"] == {"p_favorable": 0.5, "p_unfavorable": 0.25}


def test_determinism_note_does_not_recommend_a_seed(mock_runs, monkeypatch):
    real = runs_mod.run_episode
    calls = [0]

    def drifts(*args, **kwargs):
        calls[0] += 1
        ep = real(*args, **kwargs)
        if calls[0] % 2 == 0:
            for step in ep["trace"]:
                step["executed"] = "look"
        return ep

    monkeypatch.setattr(runs_mod, "run_episode", drifts)
    runs_mod.run_t0_agent_determinism({"run_id": "t0_agent_determinism", "n_games": 2})
    assert "a seed cannot help" in (mock_runs / "t0_agent_determinism" / "STATUS.md").read_text()


# --- headroom rule ---------------------------------------------------------------------------------


def test_headroom_never_says_pass_and_a_clear_result_continues(mock_runs, monkeypatch):
    assert _headroom(monkeypatch, mock_runs, _headroom_episodes(0.0, 1.0)) == 0
    first = (mock_runs / "pilotB_headroom" / "STATUS.md").read_text().splitlines()[0]
    assert "REPORTED (rule not fired)" in first and "PASS" not in first


def test_fired_headroom_stops_the_job_with_its_own_exit_code(mock_runs, monkeypatch):
    assert _headroom(monkeypatch, mock_runs, _headroom_episodes(0.5, 0.5)) == runs_mod.RULE_STOP_EXIT
    assert "STOP: PILOT_RULE_HEADROOM" in (mock_runs / "pilotB_headroom" / "STATUS.md").read_text()


def test_borderline_headroom_stops_the_job(mock_runs, monkeypatch):
    assert _headroom(monkeypatch, mock_runs, _headroom_episodes(0.25, 0.5)) == runs_mod.RULE_STOP_EXIT
    assert "borderline" in (mock_runs / "pilotB_headroom" / "STATUS.md").read_text()


def test_headroom_with_no_data_fails_instead_of_passing(mock_runs, monkeypatch):
    assert _headroom(monkeypatch, mock_runs, []) == 1
    first = (mock_runs / "pilotB_headroom" / "STATUS.md").read_text().splitlines()[0]
    assert "FAIL" in first


def test_a_rule_stop_is_never_skipped_by_resume(real_runs):
    finished(real_runs, "pilotB_headroom", "REPORTED (STOP: PILOT_RULE_HEADROOM borderline)")
    assert runs_mod._already_passed("pilotB_headroom") is False


def test_later_stages_wait_for_the_recheck(real_runs):
    finished(real_runs, "pilotB_headroom", "REPORTED (STOP: PILOT_RULE_HEADROOM borderline)")
    with pytest.raises(SystemExit):
        runs_mod.require_passed("pilotB_consumption", {}, "pilotB_headroom")
    finished(real_runs, "pilotB_headroom_recheck", "REPORTED (STOP: PILOT_RULE_HEADROOM fired)")
    with pytest.raises(SystemExit):
        runs_mod.require_passed("pilotB_consumption", {}, "pilotB_headroom")
    finished(real_runs, "pilotB_headroom_recheck", "REPORTED (rule not fired)")
    runs_mod.require_passed("pilotB_consumption", {}, "pilotB_headroom")


# --- dependencies, resume, provenance --------------------------------------------------------------


@pytest.mark.parametrize("run, dep", [
    (runs_mod.run_pilotA_score, "pilotA_transitions"),
    (runs_mod.run_pilotB_consumption, "pilotB_headroom"),
    (runs_mod.run_pilotB_eligibility, "pilotB_headroom"),
    (runs_mod.run_pilotB_retention_minirun, "pilotB_eligibility"),
])
def test_a_run_refuses_a_dependency_that_did_not_finish(real_runs, run, dep):
    name = run.__name__.removeprefix("run_")
    with pytest.raises(SystemExit):
        run({"run_id": name})
    assert "FAIL" in (real_runs / name / "STATUS.md").read_text()
    finished(real_runs, dep, "FAIL")
    with pytest.raises(SystemExit):
        run({"run_id": name})
    assert f"{dep} has not finished cleanly" in (real_runs / name / "STATUS.md").read_text()


def test_a_dependency_from_another_commit_needs_naming(real_runs):
    finished(real_runs, "pilotB_headroom", "REPORTED (rule not fired)", sha="9b0e4d618e5e")
    with pytest.raises(SystemExit):
        runs_mod.require_passed("pilotB_eligibility", {}, "pilotB_headroom")
    runs_mod.require_passed("pilotB_eligibility", {"accept_commits": {"pilotB_headroom": ["9b0e4d6"]}}, "pilotB_headroom")


def test_failed_dependency_scoring_never_starts_with_real_gate(real_runs):
    finished(real_runs, "pilotA_transitions", "PASS", sha="old")
    with pytest.raises(SystemExit):
        runs_mod.require_passed("pilotA_score", {}, "pilotA_transitions")


def test_a_mock_pass_is_not_a_real_pass(real_runs):
    finished(real_runs, "pilotB_headroom", "PASS (MOCK)")
    assert runs_mod._already_passed("pilotB_headroom") is False


def test_old_output_does_not_survive_a_new_run(mock_runs, monkeypatch):
    d = mock_runs / "t0_smoke"
    d.mkdir()
    (d / "REPORT.md").write_text("old")
    (d / "metrics.json").write_text("{}")
    runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke"})
    assert not (d / "REPORT.md").exists() and not (d / "metrics.json").exists()


def test_status_is_written_after_the_report(mock_runs, monkeypatch):
    def boom(*args):
        raise RuntimeError("disk full")

    monkeypatch.setattr(runs_mod, "write_run_report", boom)
    for run, cfg in ((runs_mod.run_t0_smoke, {"run_id": "t0_smoke", "n_games": 12}),
                     (runs_mod.run_t0_parser, {"run_id": "t0_parser"}),
                     (runs_mod.run_t0_expert_labels, {"run_id": "t0_expert_labels"})):
        with pytest.raises(RuntimeError):
            run(cfg)
        assert not (mock_runs / cfg["run_id"] / "STATUS.md").exists()


def test_status_write_is_atomic(mock_runs, monkeypatch):
    p = util.write_status("t0_smoke", "good")
    good = p.read_text()

    def fails(*args):
        raise OSError("killed")

    monkeypatch.setattr(util.os, "replace", fails)
    with pytest.raises(OSError):
        util.write_status("t0_smoke", "bad")
    assert p.read_text() == good


def test_an_empty_status_does_not_crash_resume(real_runs):
    d = finished(real_runs, "t0_smoke")
    (d / "STATUS.md").write_text("")
    assert runs_mod._already_passed("t0_smoke") is False


def test_a_changed_input_blocks_the_skip(real_runs):
    source = real_runs / "x.jsonl"
    source.write_text("a\n")
    finished(real_runs, "t0_smoke", inputs={str(source): util.file_sha(source)})
    assert runs_mod._already_passed("t0_smoke") is True
    source.write_text("b\n")
    assert runs_mod._already_passed("t0_smoke") is False


def test_a_deleted_input_blocks_the_skip(real_runs):
    source = real_runs / "x.jsonl"
    source.write_text("a\n")
    finished(real_runs, "t0_smoke", inputs={str(source): util.file_sha(source)})
    source.unlink()
    assert runs_mod._already_passed("t0_smoke") is False


def test_resume_refuses_changed_inputs(mock_runs):
    source = mock_runs / "x.jsonl"
    source.write_text("a\n")
    runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke"}, inputs=(source,))
    source.write_text("b\n")
    with pytest.raises(SystemExit, match="input files changed"):
        runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke", "resume": True}, inputs=(source,))


def test_resume_refuses_changed_settings(mock_runs):
    runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke", "n_games": 4})
    with pytest.raises(SystemExit, match="settings changed"):
        runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke", "n_games": 2, "resume": True})


def test_resume_refuses_uncommitted_code(real_runs, monkeypatch):
    runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke"})
    assert json.loads((real_runs / "t0_smoke" / "inputs.json").read_text())["git_dirty"] is False
    monkeypatch.setattr(runs_mod, "git_dirty", lambda: True)
    with pytest.raises(SystemExit, match="uncommitted"):
        runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke", "resume": True})


def test_a_real_run_needs_a_commit(real_runs, monkeypatch):
    monkeypatch.setattr(runs_mod, "git_sha", lambda: "unknown")
    with pytest.raises(SystemExit, match="no git commit"):
        runs_mod._init_run("t0_smoke", {"run_id": "t0_smoke"})
    d = finished(real_runs, "t0_parser", sha="unknown")
    assert d.exists() and runs_mod._already_passed("t0_parser") is False


def test_dirty_means_code_not_run_output_or_calibration(monkeypatch, tmp_path):
    git = lambda *args: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "runs").mkdir()
    (tmp_path / "wf" / "config").mkdir(parents=True)
    files = [tmp_path / "code.py", tmp_path / "runs" / "env.json", tmp_path / "wf" / "config" / "calibrated.yaml"]
    for path in files:
        path.write_text("a")
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "x")
    monkeypatch.setattr(util, "REPO_ROOT", tmp_path)
    assert util.git_dirty() is False
    files[1].write_text("b")
    files[2].write_text("b")
    assert util.git_dirty() is False
    files[0].write_text("b")
    assert util.git_dirty() is True


def test_scores_carry_an_item_hash(mock_runs):
    items = [_item(i) for i in range(6)]
    _write_transitions(mock_runs / "pilotA_transitions" / "transitions.jsonl", items)
    runs_mod.run_pilotA_score({"run_id": "pilotA_score"})
    scores = read_jsonl(mock_runs / "pilotA_score" / "scoring.jsonl")
    assert scores and all(s["item_hash"] == runs_mod._item_hash(items[int(s["transition_id"][2:])]) for s in scores)


def test_resume_refuses_scores_for_rebuilt_items(mock_runs):
    # Same ids, different actions: the real failure gave a wrong ΔL with no warning.
    path = mock_runs / "pilotA_transitions" / "transitions.jsonl"
    _write_transitions(path, [_item(i) for i in range(6)])
    runs_mod.run_pilotA_score({"run_id": "pilotA_score"})
    _write_transitions(path, [_item(i, action=f"take cup {i} from shelf 2") for i in range(6)])
    with pytest.raises(SystemExit, match="input files changed"):
        runs_mod.run_pilotA_score({"run_id": "pilotA_score", "resume": True})
    (mock_runs / "pilotA_score" / "inputs.json").unlink()  # even without the file-level record
    with pytest.raises(SystemExit, match="different items"):
        runs_mod.run_pilotA_score({"run_id": "pilotA_score", "resume": True})


# --- mock and real stay apart ----------------------------------------------------------------------


def test_mock_refuses_a_real_runs_dir(monkeypatch):
    monkeypatch.setenv("WF_MOCK_ENV", "1")
    monkeypatch.setenv("RUNS_DIR", "/storage/ice1/2/1/gmat3/text-wm-fix2/runs")
    with pytest.raises(SystemExit, match="mock run refused"):
        util.runs_root()


def test_mock_allows_runs_mock_and_temp_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("WF_MOCK_ENV", "1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    assert util.runs_root() == tmp_path
    monkeypatch.setenv("RUNS_DIR", "/home/me/project/runs_mock")
    assert util.runs_root().name == "runs_mock"


def test_mock_records_say_so_and_real_mode_refuses_them(monkeypatch, tmp_path):
    path = tmp_path / "e.jsonl"
    monkeypatch.setenv("WF_MOCK_ENV", "1")
    append_jsonl(path, [{"a": 1}])
    assert read_jsonl(path) == [{"a": 1, "mock": True}]
    for key in MOCK_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="mock record"):
        read_jsonl(path)
    with pytest.raises(ValueError, match="mock record"):
        list(util.iter_jsonl(path))


# --- report ----------------------------------------------------------------------------------------


ALL_RUNS = assemble.TIER0 + assemble.PILOT_A + assemble.PILOT_B


def _report_tree(root: Path, monkeypatch, status="PASS", sha="abc1234"):
    monkeypatch.setattr(assemble, "git_sha", lambda: "abc1234")
    for rid in ALL_RUNS:
        finished(root, rid, status, sha)


def test_report_shows_each_runs_status_and_commit(real_runs, monkeypatch):
    _report_tree(real_runs, monkeypatch)
    finished(real_runs, "t0_agent_determinism", "REPORTED (BELOW TARGET: agreement 0.757 < 0.95)")
    finished(real_runs, "pilotA_transitions", "FAIL")
    assemble.main([])
    report = (real_runs / "PILOT_REPORT.md").read_text()
    assert "git abc1234" in report and "— FAIL — git abc1234" in report
    assert "| t0_agent_determinism | below target | abc1234 |" in report


def test_report_refuses_a_missing_run(real_runs, monkeypatch, capsys):
    _report_tree(real_runs, monkeypatch)
    (real_runs / "t0_smoke" / "STATUS.md").unlink()
    with pytest.raises(SystemExit):
        assemble.main([])
    assert "t0_smoke: no STATUS.md" in capsys.readouterr().err
    assert not (real_runs / "PILOT_REPORT.md").exists()


def test_report_refuses_tracked_style_output_with_no_commit(real_runs, monkeypatch):
    _report_tree(real_runs, monkeypatch)
    (real_runs / "pilotB_headroom" / "git_sha").unlink()  # a fresh clone's old files have none
    with pytest.raises(SystemExit):
        assemble.main([])


def test_report_refuses_another_commit_unless_named(real_runs, monkeypatch):
    _report_tree(real_runs, monkeypatch)
    finished(real_runs, "pilotB_headroom", "REPORTED (rule not fired)", sha="9b0e4d618e5e")
    with pytest.raises(SystemExit):
        assemble.main([])
    assemble.main(["--accept-commit", "pilotB_headroom=9b0e4d6"])
    assert "git 9b0e4d618e5e" in (real_runs / "PILOT_REPORT.md").read_text()


def test_real_report_refuses_mock_runs(real_runs, monkeypatch, capsys):
    _report_tree(real_runs, monkeypatch)
    finished(real_runs, "pilotA_score", "REPORTED (rule output) (MOCK)")
    with pytest.raises(SystemExit):
        assemble.main([])
    assert "mock" in capsys.readouterr().err


def test_report_includes_a_recheck_when_one_ran(real_runs, monkeypatch):
    _report_tree(real_runs, monkeypatch)
    finished(real_runs, "pilotB_headroom_recheck", "REPORTED (rule not fired)")
    assemble.main([])
    assert "### pilotB_headroom_recheck" in (real_runs / "PILOT_REPORT.md").read_text()


# --- verification tools ----------------------------------------------------------------------------


def test_verify_real_env_exits_nonzero_on_a_failed_check(mock_runs, monkeypatch, tmp_path):
    monkeypatch.setenv("WF_INIT_CONTEXTS", str(tmp_path))
    monkeypatch.setenv("WORD2WORLD_ROOT", str(tmp_path))
    contexts = tmp_path / "data" / "init_contexts" / "alfworld"
    contexts.mkdir(parents=True)
    (tmp_path / "wm_instruct_train.json").write_text(json.dumps([{"messages": [{"content": ""}]}] * 200))
    monkeypatch.setattr(verify_env, "check_games", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setattr("sys.argv", ["verify", "--only", "games", "--games", "1", "--memory-resets", "1"])
    with pytest.raises(SystemExit, match="checks failed"):
        verify_env.main()
    assert json.loads((mock_runs / "verify_games.json").read_text())["git_sha"]


def test_verify_job_reports_failure_after_running_everything():
    script = (REPO / "pace" / "verify_models.sbatch").read_text()
    assert script.count("|| failed=1") == 4 and 'exit "${failed}"' in script


def test_review_keeps_items_as_it_goes(mock_runs, monkeypatch):
    calls = [0]
    real = export_review._walk_transitions

    def dies(*args):
        calls[0] += 1
        if calls[0] == 3:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(export_review, "_walk_transitions", dies)
    monkeypatch.setattr("sys.argv", ["export", "--games", "4"])
    with pytest.raises(KeyboardInterrupt):
        export_review.main()
    assert {t["game_id"] for t in read_jsonl(mock_runs / "review_items.jsonl")}.__len__() == 2


def test_review_says_when_it_is_mock(mock_runs, monkeypatch):
    monkeypatch.setattr("sys.argv", ["export", "--games", "2", "--items", "12"])
    export_review.main()
    text = (mock_runs / "HUMAN_REVIEW.md").read_text()
    assert "MOCK DATA" in text and "Generated from the real ALFWorld server" not in text


# --- job scripts -----------------------------------------------------------------------------------


def _bash(script: str, env: dict | None = None, cwd: Path = REPO) -> subprocess.CompletedProcess:
    base = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    return subprocess.run(["bash", "-c", script], env={**base, **(env or {})}, cwd=cwd, capture_output=True, text=True)


def _copy_pace(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "pace").mkdir(parents=True)
    (root / "pace" / "_common.sh").write_text((REPO / "pace" / "_common.sh").read_text())
    return root


@pytest.mark.parametrize("var", MOCK_KEYS)
def test_cluster_jobs_refuse_mock_mode(tmp_path, var):
    root = _copy_pace(tmp_path)
    done = _bash("source pace/_common.sh; echo reached", {var: "1"}, cwd=root)
    assert done.returncode != 0 and "reached" not in done.stdout and var in done.stderr


def test_checkout_comes_from_the_script_not_the_shell(tmp_path):
    root = _copy_pace(tmp_path)
    done = _bash('source pace/_common.sh; echo "root=$PROJECT_ROOT"', {"PROJECT_ROOT": "/somewhere/else"}, cwd=root)
    assert f"root={root.resolve()}" in done.stdout


def test_a_copied_pace_env_for_another_checkout_stops_the_job(tmp_path):
    root = _copy_pace(tmp_path)
    (root / "pace" / "pace.env").write_text('export PROJECT_ROOT="/somewhere/else"\n')
    done = _bash("source pace/_common.sh; echo reached", cwd=root)
    assert done.returncode != 0 and "reached" not in done.stdout and "pace.env says" in done.stderr


def test_only_wf_resume_1_resumes(tmp_path):
    root = _copy_pace(tmp_path)
    stub = 'activate_serve() { :; }; python() { echo "ARGS $*"; }; run_wf t0_smoke'
    for value, resumes in (("1", True), ("0", False), ("", False)):
        done = _bash(f"source pace/_common.sh; {stub}", {"WF_RESUME": value}, cwd=root)
        assert ("--resume" in done.stdout) is resumes, done.stdout + done.stderr


def test_every_job_refuses_requeue():
    for script in (REPO / "pace").glob("*.sbatch"):
        assert "#SBATCH --no-requeue" in script.read_text(), script.name


def test_servers_are_checked_for_life_before_health():
    text = (REPO / "pace" / "_common.sh").read_text()
    for life, health in (('kill -0 "$(cat "${ALFWORLD_PID_FILE}")"', 'curl -sf "${ALFWORLD_BASE_URL}/"'),
                         ('kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then\n      echo "ERROR', '/health')):
        assert text.index(life) < text.rindex(health)


def test_pilot_b_has_the_time_and_memory_it_needs():
    script = (REPO / "pace" / "pilotB.sbatch").read_text()
    assert "--time=14:00:00" in script and "--mem=128G" in script


def test_recheck_job_runs_the_recheck_then_the_later_stages():
    script = (REPO / "pace" / "pilotB_recheck.sbatch").read_text()
    order = [script.index(f"run_wf {r}\n") for r in ("pilotB_headroom_recheck", "pilotB_consumption", "pilotB_eligibility", "pilotB_retention_minirun")]
    assert order == sorted(order) and "run_wf pilotB_headroom\n" not in script
