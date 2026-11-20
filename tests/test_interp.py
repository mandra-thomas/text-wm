"""Interpretability tooling (wf/interp) on tiny random models and a word-level tokenizer built here.

Nothing is downloaded. The identities checked: patching nothing reproduces the target run; patching every
position (or the answer position at the last layer) from the source reproduces the source's readout; patching
positions the two prompts share changes nothing; every head patched separately equals patching the attention
output; the logit lens at the last layer is the model's readout. Also hook removal, determinism, token groups,
resumable stages, item selection, probes and directions.
"""
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

from wf.interp import directions as D  # noqa: E402
from wf.interp import hooks as H  # noqa: E402
from wf.interp import probes as P  # noqa: E402
from wf.interp import prompts as PR  # noqa: E402
from wf.interp import run as R  # noqa: E402
from wf.interp.lens import logit_lens  # noqa: E402
from wf.interp.model import parts  # noqa: E402
from wf.interp.patching import Cell, arrival_layer, fraction_restored, head_cells, ratio_of_sums, sweep  # noqa: E402

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
SPECIALS = ["<|im_start|>", "<|im_end|>", "<unk>"]
SPLIT = r"\s+|\w+|[^\w\s]"


def _item(**kw):
    hist = ("You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 3, "
            "a shelf 1, and a countertop 1.\nYour task is to: put some mug on countertop.\n"
            "> go to shelf 1\nYou arrive at shelf 1. On the shelf 1, you see a mug 2, and a plate 1.\n"
            "> go to cabinet 1\nYou arrive at cabinet 1. The cabinet 1 is closed.")
    t = {"id": "t_000001", "game_id": "valid_unseen/1", "action": "take mug 1 from cabinet 3",
         "task_line": "Your task is to: put some mug on countertop.",
         "swapped_task_line": "Your task is to: put some plate on countertop.",
         "goal_object": "mug", "swapped_goal_object": "plate", "history_text": hist,
         "history_pairs": [], "history_len": 2, "determination_reason": "away_from_receptacle",
         "candidates": {"o_plus": "You pick up the mug 1 from the cabinet 3.", "o_minus": "Nothing happens."},
         "scene_locations": {"mug 1": ["cabinet 3"], "bowl 1": ["cabinet 3"], "plate 1": ["shelf 1"]}}
    return {**t, **kw}


def _unequal_item():
    """Swapped object of two words, so the two prompts' spans differ in length."""
    return _item(id="t_000002", swapped_goal_object="spray bottle",
                 swapped_task_line="Your task is to: put some spray bottle on countertop.")


def _corpus(items):
    texts = []
    for t in items:
        for fmt in PR.FORMATS:
            for goal in ("aligned", "swapped", "absent"):
                texts += [m["content"] for m in PR.messages(t, fmt, goal)] + ["system user assistant"]
            texts += [m["content"] for m in PR.messages(t, fmt, "placebo", "bowl")]
    return "\n".join(texts) + " Yes No You Nothing"


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted(set(re.findall(SPLIT, _corpus([_item(), _unequal_item()]))))
    vocab = {w: i for i, w in enumerate(SPECIALS + words)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(SPLIT), behavior="isolated")
    tk = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>", additional_special_tokens=SPECIALS[:2])
    tk.chat_template = TEMPLATE
    return tk


def _qwen(vocab, layers=4, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      use_sliding_window=False, tie_word_embeddings=False, attn_implementation="sdpa",
                      initializer_range=0.3)  # large weights, so prompts that differ in one word read out differently
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def model(tok):
    return _qwen(len(tok))


@pytest.fixture(scope="module")
def p(model):
    return parts(model)


def _ids(e):
    return torch.tensor([e.ids])


def _d(logits, plus, minus, row=0):
    return float(logits[row, plus] - logits[row, minus])


# ---------------- prompts and token groups ----------------


def test_encode_matches_the_chat_template_and_prompts_are_the_scored_ones(tok):
    from wf import rq1, rq1v

    t = _item()
    assert PR.messages(t, "yn", "aligned") == rq1v.arm_messages(t, "yn_aligned")
    assert PR.messages(t, "obs", "swapped") == rq1.arm_messages(t, "swapped")
    e = PR.encode(tok, PR.messages(t, "yn", "aligned"))
    assert e.ids == list(tok.apply_chat_template(PR.messages(t, "yn", "aligned"), add_generation_prompt=True))
    assert e.text[e.user_start:e.user_end].endswith("Answer:")
    assert PR.readout_ids(tok, "yn") != PR.readout_ids(tok, "obs")


def test_pair_and_groups_partition_the_suffix(tok):
    t = _item()
    pr = PR.build_pair(tok, t, "yn")
    assert pr.equal_span and pr.span_a[1] - pr.span_a[0] == 1
    assert tok.decode([pr.a.ids[pr.prefix]]) == "mug" and tok.decode([pr.b.ids[pr.prefix]]) == "plate"
    g = pr.groups
    fine = ["span", "task_rest", "hist_obj", "hist_swap", "hist_rec", "hist_other", "action", "query", "final"]
    covered = sorted(i for k in fine for i in g[k])
    assert covered == list(range(pr.prefix, len(pr.a)))  # every differing or later position, exactly once
    assert g["last"] == [len(pr.a) - 1] and g["last"][0] in g["final"]
    assert sorted(g["hist_all"]) == sorted(g["hist_obj"] + g["hist_swap"] + g["hist_rec"] + g["hist_other"])
    text = lambda ps: "".join(tok.decode([pr.a.ids[i]]) for i in ps)  # noqa: E731
    assert text(g["action"]).strip().startswith("Action: take mug 1 from cabinet 3")
    assert "mug" in text(g["hist_obj"]) and "plate" in text(g["hist_swap"]) and "cabinet" not in text(g["hist_obj"])
    assert text(g["query"]).strip().startswith("Question:")
    assert text(g["final"]).startswith("<|im_end|>")
    assert text(g["task_rest"]).startswith(" on countertop")


def test_unequal_spans_map_the_suffix_from_the_end(tok):
    pr = PR.build_pair(tok, _unequal_item(), "yn")
    assert not pr.equal_span and pr.groups["span"] == []
    assert len(pr.b) == len(pr.a) + 2  # "spray bottle" is three tokens, "mug" one
    for g in ("action", "final", "hist_other"):
        a_pos = pr.groups[g]
        assert [pr.a.ids[i] for i in a_pos] == [pr.b.ids[j] for j in pr.to_b(a_pos)]
    fl = pr.flipped()
    assert [fl.a.ids[i] for i in fl.groups["action"]] == [pr.a.ids[i] for i in pr.groups["action"]]
    with pytest.raises(ValueError):
        pr.to_b(list(range(*pr.span_a)))


def test_action_positions_and_final_positions(tok):
    t = _item()
    e = PR.encode(tok, PR.messages(t, "yn", "aligned"))
    pos = PR.action_positions(e, t)
    assert tok.decode([e.ids[pos["action_rec"]]]) == "cabinet" and tok.decode([e.ids[pos["action_end"]]]) == "3"
    assert pos["answer"] == len(e) - 1
    fp = R.final_positions(e)
    assert fp[-1] == len(e) - 1 and tok.decode([e.ids[fp[0]]]) == "<|im_end|>"


def test_placebo_object_follows_the_rules():
    t = _item()
    same_game = [_item(id="a", swapped_goal_object="plate"), _item(id="b", swapped_goal_object="bowl"),
                 _item(id="c", swapped_goal_object="apple"), _item(id="d", swapped_goal_object="kettle",
                                                                   task_line="Your task is to: other.")]
    # plate is the item's own swap and bowl is at cabinet 3 at reset; kettle belongs to another task
    assert PR.placebo_object(t, same_game) == "apple"
    assert PR.placebo_object(t, same_game[:2]) is None


# ---------------- hooks: removal, determinism, identities ----------------


def test_hooks_are_removed_even_after_an_exception(p, model, tok):
    e = PR.encode(tok, PR.messages(_item(), "yn", "aligned"))
    before = H.n_hooks(model)
    with pytest.raises(RuntimeError):
        with H.HookSet() as hs:
            hs.edit(p, "resid", 1, lambda h: (_ for _ in ()).throw(RuntimeError("boom")))
            hs.edit(p, "head_in", 0, lambda h: h)
            assert H.n_hooks(model) == before + 2
            H.forward_last_logits(p, _ids(e))
    assert H.n_hooks(model) == before
    with H.no_hooks_left(model):
        H.capture(p, _ids(e), [("resid", 0), ("attn", 1), ("mlp", 2), ("head_in", 3)])
        H.run_patched(p, _ids(e), [H.Patch("resid", 1, 0, torch.tensor([3]), torch.zeros(1, 64))])


def test_forward_is_deterministic_and_patching_nothing_is_the_clean_run(p, tok):
    e = PR.encode(tok, PR.messages(_item(), "yn", "aligned"))
    a1, a2 = H.forward_last_logits(p, _ids(e)), H.forward_last_logits(p, _ids(e))
    assert torch.equal(a1, a2)
    assert torch.equal(H.run_patched(p, _ids(e), []), a1)
    logits, _ = H.capture(p, _ids(e), [("resid", 2)])
    assert torch.equal(logits, a1)  # reading activations does not change the run


@pytest.mark.parametrize("site,layer", [("resid", -1), ("resid", 0), ("resid", 2), ("attn", 0), ("mlp", 1)])
def test_patching_every_position_reproduces_the_source(p, tok, site, layer):
    pr = PR.build_pair(tok, _item(), "yn")
    plus, minus = PR.readout_ids(tok, "yn")
    b_logits, b_acts = H.capture(p, _ids(pr.b), [(site, layer)])
    a_logits = H.forward_last_logits(p, _ids(pr.a))
    all_pos = torch.arange(len(pr.a))
    out = H.run_patched(p, _ids(pr.a), [H.Patch(site, layer, 0, all_pos, b_acts[(site, layer)][0])])
    assert _d(a_logits, plus, minus) != pytest.approx(_d(b_logits, plus, minus), abs=1e-4)
    if site == "resid":
        assert torch.allclose(out, b_logits, atol=1e-5)
    else:  # a block's output alone is not the whole state: the residual before it still comes from the target
        assert not torch.allclose(out, a_logits, atol=1e-6)


def test_answer_position_at_the_last_layer_reproduces_the_source_even_with_unequal_spans(p, tok):
    pr = PR.build_pair(tok, _unequal_item(), "yn")
    plus, minus = PR.readout_ids(tok, "yn")
    last = p.n_layers - 1
    b_logits, b_acts = H.capture(p, _ids(pr.b), [("resid", last)])
    out = H.run_patched(p, _ids(pr.a), [H.Patch("resid", last, 0, torch.tensor([len(pr.a) - 1]),
                                                b_acts[("resid", last)][0, [len(pr.b) - 1]])])
    assert _d(out, plus, minus) == pytest.approx(_d(b_logits, plus, minus), abs=1e-5)


def test_patching_shared_prefix_positions_changes_nothing(p, tok):
    for t in (_item(), _unequal_item()):
        pr = PR.build_pair(tok, t, "yn")
        plus, minus = PR.readout_ids(tok, "yn")
        assert R.check_prefix_noop(p, pr, plus, minus, tol=1e-6) == pytest.approx(0.0, abs=1e-6)


def test_every_head_patched_separately_equals_patching_the_attention_output(p, tok):
    pr = PR.build_pair(tok, _item(), "yn")
    layer, pos = 1, torch.tensor(pr.groups["final"])
    _, src = H.capture(p, _ids(pr.b), [("head_in", layer), ("attn", layer)])
    heads = [H.Patch("head_in", layer, 0, pos, src[("head_in", layer)][0, pos][:, h * p.head_dim:(h + 1) * p.head_dim], h)
             for h in range(p.n_heads)]
    by_heads = H.run_patched(p, _ids(pr.a), heads)
    by_attn = H.run_patched(p, _ids(pr.a), [H.Patch("attn", layer, 0, pos, src[("attn", layer)][0, pos])])
    assert torch.allclose(by_heads, by_attn, atol=1e-5)
    with pytest.raises(ValueError):
        H.run_patched(p, _ids(pr.a), [H.Patch("attn", layer, 0, pos, src[("attn", layer)][0, pos], head=0)])


def test_batched_rows_get_only_their_own_patch(p, tok):
    pr = PR.build_pair(tok, _item(), "yn")
    plus, minus = PR.readout_ids(tok, "yn")
    last = p.n_layers - 1
    b_logits, b_acts = H.capture(p, _ids(pr.b), [("resid", last)])
    ids = torch.tensor([pr.a.ids] * 3)
    out = H.run_patched(p, ids, [H.Patch("resid", last, 2, torch.tensor([len(pr.a) - 1]), b_acts[("resid", last)][0, [-1]])])
    clean = H.forward_last_logits(p, _ids(pr.a))
    assert torch.allclose(out[0], clean[0], atol=1e-5) and torch.allclose(out[1], clean[0], atol=1e-5)
    assert _d(out, plus, minus, 2) == pytest.approx(_d(b_logits, plus, minus), abs=1e-5)


def test_direction_edit_sets_or_adds_the_component():
    torch.manual_seed(1)
    h = torch.randn(2, 5, 8)
    u = torch.nn.functional.normalize(torch.randn(8), dim=0)
    out = H.direction_edit(u, positions=[1, 3], value=0.7)(h)
    assert torch.allclose(out[:, [1, 3]] @ u, torch.full((2, 2), 0.7), atol=1e-5)
    assert torch.equal(out[:, [0, 2, 4]], h[:, [0, 2, 4]])
    other = out[:, 1] - (out[:, 1] @ u).unsqueeze(-1) * u  # the orthogonal part is untouched
    assert torch.allclose(other, h[:, 1] - (h[:, 1] @ u).unsqueeze(-1) * u, atol=1e-5)
    assert torch.allclose(H.direction_edit(u, add=2.0)(h) - h, 2.0 * u.expand_as(h), atol=1e-5)


def test_logit_lens_at_the_last_layer_is_the_readout(p, tok):
    e = PR.encode(tok, PR.messages(_item(), "obs", "aligned"))
    plus, minus = PR.readout_ids(tok, "obs")
    logits, acts = H.capture(p, _ids(e), [("resid", i) for i in range(p.n_layers)], positions=[len(e) - 1])
    last = torch.stack([acts[("resid", i)][0, 0] for i in range(p.n_layers)])
    lens = logit_lens(p, last, plus, minus)
    assert len(lens) == p.n_layers and lens[-1] == pytest.approx(_d(logits, plus, minus), abs=1e-4)


def test_llama_layout_is_supported(tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                     initializer_range=0.3)).eval()
    q = parts(m)
    assert q.n_layers == 3 and q.head_dim == 16
    pr = PR.build_pair(tok, _item(), "yn")
    b_logits, b_acts = H.capture(q, _ids(pr.b), [("resid", 1)])
    out = H.run_patched(q, _ids(pr.a), [H.Patch("resid", 1, 0, torch.arange(len(pr.a)), b_acts[("resid", 1)][0])])
    assert torch.allclose(out, b_logits, atol=1e-5)


# ---------------- sweep and stages ----------------


def test_sweep_fractions_hit_the_identities(p, tok):
    pr = PR.build_pair(tok, _item(), "yn")
    plus, minus = PR.readout_ids(tok, "yn")
    last = p.n_layers - 1
    cells = [Cell("resid", last, "last"), Cell("resid", last, "final"), Cell("resid", 0, "hist_rec"),
             Cell("attn", 1, "action"), Cell("resid", -1, "span")] + head_cells([1], p.n_heads)
    res = sweep(p, pr, cells, plus, minus, batch_size=4)
    assert res["clean_spread"] == pytest.approx(0.0, abs=1e-5)
    a = H.forward_last_logits(p, _ids(pr.a))
    assert res["d_target"] == pytest.approx(_d(a, plus, minus), abs=1e-5)
    # Readouts are compared in absolute terms: batch size changes fp32 rounding by about 1e-6, and the random
    # model's effect is small, so a fraction would magnify that rounding.
    assert abs(res["d_target"] - res["d_source"]) > 1e-3
    for c in cells[:2]:  # the answer position at the last layer carries the whole effect
        assert res["cells"][c.key()] == pytest.approx(res["d_source"], abs=1e-4)
    # at an equal-length pair, the span's embedding is everything the two prompts differ in
    assert res["cells"]["resid|-1|span"] == pytest.approx(res["d_source"], abs=1e-4)
    assert fraction_restored(res["d_target"], res["d_source"], res["d_source"]) == 1.0
    den = sweep(p, pr.flipped(), [Cell("resid", last, "last")], plus, minus)
    assert den["d_target"] == pytest.approx(res["d_source"], abs=1e-5)
    assert den["cells"]["resid|%d|last" % last] == pytest.approx(res["d_target"], abs=1e-4)


def test_sweep_skips_empty_groups(p, tok):
    pr = PR.build_pair(tok, _unequal_item(), "yn")
    plus, minus = PR.readout_ids(tok, "yn")
    res = sweep(p, pr, [Cell("resid", 0, "span"), Cell("resid", 1, "action")], plus, minus)
    assert res["cells"]["resid|0|span"] is None and res["cells"]["resid|1|action"] is not None
    assert res["n_source"] == res["n_target"] + 2 and not res["equal_span"]


def test_baseline_stage_saves_and_resumes(p, tok, tmp_path):
    items = [_item(), _unequal_item()]
    store = R.baseline(p, tok, items, tmp_path / "acts", placebo={"t_000001": "bowl"},
                       goals=("aligned", "swapped", "absent", "placebo"), chunk=3, min_free_gb=0)
    rows = R.read_jsonl(store.index)
    assert len(rows) == 2 * 2 * 3 + 2  # two items, two formats, three goals, plus the one item with a placebo
    again = R.baseline(p, tok, items, tmp_path / "acts", placebo={"t_000001": "bowl"},
                       goals=("aligned", "swapped", "absent", "placebo"), chunk=3, min_free_gb=0)
    assert len(R.read_jsonl(again.index)) == len(rows)
    rows, last = store.load("last")
    _, probe = store.load("probe")
    assert last.shape == (len(rows), p.n_layers, 64) and probe.shape == (len(rows), len(R.probe_layers(p.n_layers)), 2, 64)
    for r in rows:  # the lens at the last layer equals the saved readout; mass is a probability
        assert r["lens"][-1] == pytest.approx(r["d"], abs=1e-3) and 0 < r["mass"] <= 1
    key = "t_000001|yn|aligned"
    e = PR.encode(tok, PR.messages(_item(), "yn", "aligned"))
    _, acts = H.capture(p, _ids(e), [("resid", 2)], positions=[len(e) - 1])
    _, one = store.load("last", keys=[key])
    assert np.allclose(one[0, 2], acts[("resid", 2)][0, 0].numpy(), rtol=2e-3, atol=1e-3)  # fp16 storage


def test_patch_and_edit_stages_resume(p, tok, tmp_path):
    items = [_item(), _unequal_item()]
    cells = [Cell("resid", 0, "action"), Cell("resid", p.n_layers - 1, "last")]
    out = R.patch_items(p, tok, items, tmp_path / "patch.jsonl", cells, batch_size=2)
    first = R.read_jsonl(out)
    assert len(first) == 4 and {r["direction"] for r in first} == {"noise", "denoise"}
    R.patch_items(p, tok, items, out, cells)
    assert len(R.read_jsonl(out)) == 4
    R.patch_items(p, tok, items, out, cells, directions=("noise",), source_goal="placebo", placebo={"t_000001": "bowl"})
    assert len(R.read_jsonl(out)) == 5
    u = torch.nn.functional.normalize(torch.randn(64), dim=0)
    edits = {"noop": [(1, lambda pos: H.direction_edit(u, pos, add=0.0))],
             "push": [(1, lambda pos: H.direction_edit(u, pos, add=50.0))]}
    eo = R.edit_items(p, tok, items, tmp_path / "edit.jsonl", edits, batch_size=2)
    recs = R.read_jsonl(eo)
    assert len(recs) == 6
    for r in recs:
        assert r["edits"]["noop"] == pytest.approx(r["d"], abs=1e-4) and r["edits"]["push"] != pytest.approx(r["d"], abs=1e-3)
    R.edit_items(p, tok, items, eo, edits)
    assert len(R.read_jsonl(eo)) == 6


def test_disk_check_stops_before_writing(tmp_path):
    with pytest.raises(SystemExit):
        R.check_disk(tmp_path, min_free_gb=1e12)


def test_item_selection_is_seeded_stratified_and_filtered():
    items = []
    for k in range(60):
        items.append(_item(id=f"t{k:03d}", game_id=f"g{k % 6}", history_len=0 if k % 3 == 0 else 4,
                           determination_reason="hands_full" if k % 2 else "away_from_receptacle"))
    effect = {t["id"]: (2.0 if k % 5 else 0.5) for k, t in enumerate(items)}
    games = {"g0", "g1", "g2", "g3"}
    pick = R.select_patch_items(items, effect, 10, 1.0, games, "seed")
    assert pick == R.select_patch_items(list(reversed(items)), effect, 10, 1.0, games, "seed")
    assert len(pick) == 10 and all(t["game_id"] in games and effect[t["id"]] >= 1.0 for t in pick)
    step0 = sum(t["history_len"] == 0 for t in pick)
    eligible = [t for t in items if t["game_id"] in games and effect[t["id"]] >= 1.0]
    assert abs(step0 - 10 * sum(t["history_len"] == 0 for t in eligible) / len(eligible)) <= 1.5
    assert pick != R.select_patch_items(items, effect, 10, 1.0, games, "other seed")


# ---------------- analysis helpers ----------------


def test_fractions_and_arrival():
    assert fraction_restored(5.0, 1.0, 3.0) == pytest.approx(0.5)
    assert ratio_of_sums([(5, 1, 3), (2, 1, 2)]) == pytest.approx(2 / 5)
    assert arrival_layer([0.0, 0.2, 0.6, 0.4, 0.7, 1.0]) == pytest.approx(4 / 5)
    assert arrival_layer([0.0, 0.1]) != arrival_layer([0.0, 0.1])  # nan when never reached


def test_fact_labels_and_deciding_fact():
    t = _item()
    lab = P.fact_labels(t)
    assert lab["at_receptacle"] is False and lab["holding_any"] is False and lab["holding_object"] is False
    assert lab["object_listed"] is None and lab["receptacle_closed"] is None  # cabinet 3 never seen
    assert P.deciding_fact(t) == "at_receptacle" and P.success_value(t, "at_receptacle") is True
    t2 = _item(action="take mug 1 from shelf 1", determination_reason="contents_or_open_state")
    assert P.fact_labels(t2)["object_listed"] is False and P.deciding_fact(t2) == "object_listed"
    t3 = _item(action="take mug 1 from cabinet 1", determination_reason="contents_or_open_state")
    assert P.fact_labels(t3)["receptacle_closed"] is True and P.deciding_fact(t3) == "receptacle_closed"
    assert P.success_value(t3, "receptacle_closed") is False


def test_probe_fit_shift_and_balance():
    rng = np.random.default_rng(0)
    n, d = 400, 16
    y = rng.integers(0, 2, n).astype(bool)
    w = np.zeros(d)
    w[3] = 1.0
    X = rng.standard_normal((n, d)) + 3.0 * np.outer(y, w)
    groups = [f"g{i % 20}" for i in range(n)]
    m = P.fit_probe(X, y, groups)
    assert P.probe_auc(m, X, y) > 0.9
    assert abs(P.probe_direction(m)[3]) > 0.9
    sep = P.separation(m, X, y)
    shifted = X + 1.5 * w  # half the class gap, toward the positive class
    s = P.belief_shifts(m, shifted, X, sep, [True] * n)
    assert 0.3 < float(np.mean(s)) < 0.7
    assert float(np.mean(P.belief_shifts(m, shifted, X, sep, [False] * n))) < 0
    outcomes = list(rng.integers(0, 2, n).astype(bool))
    keep = P.balanced_indices(list(y), outcomes, 0)
    for o in (True, False):
        ys = [y[i] for i in keep if outcomes[i] == o]
        assert sum(ys) == len(ys) - sum(ys)
    train, test = P.split_games([f"g{i}" for i in range(10)], 0)
    assert not train & test and len(train | test) == 10 and P.split_games([f"g{i}" for i in range(10)], 0) == (train, test)


def test_direction_helpers():
    rng = np.random.default_rng(1)
    u0 = np.zeros(32)
    u0[5] = 1.0
    S = rng.standard_normal((200, 32))
    A = S + 2.0 * u0 + 0.1 * rng.standard_normal((200, 32))
    est = D.diff_of_means(A, S)
    assert D.cosine(est["u"], u0) > 0.99 and est["norm"] == pytest.approx(2.0, abs=0.1)
    assert est["midpoint"] == pytest.approx(D.midpoint_for(est["u"], A, S))
    r = D.random_units(32, 5, 0)
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0) and np.array_equal(r, D.random_units(32, 5, 0))
    assert D.auc([3, 4], [1, 2]) == 1.0 and D.auc([1], [1]) == 0.5
    assert D.removed_share(4.0, 1.0) == pytest.approx(0.75)
