"""Model families, wave A (wf/families.py, the shared shard runner in wf/sizes.py, wf/tools/prefetch_models.py; notes/families_design.md)."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import types
from pathlib import Path

import pytest

from wf import constants as C, families, sizes
from wf.scoring.logodds import LogoddsScorer
from wf.scoring.prompts import WM_SYSTEM
from tests.test_rq1 import _item, mock_root, mock_score  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "golden" / "default_prompt.json").read_text())
RENDERED = json.loads((ROOT / "tests" / "golden" / "families_rendered.json").read_text())
NOOP = "Nothing happens."


# ---------------- registry ----------------


def test_every_family_model_is_pinned_registered_and_configured():
    from wf.runs import RUNNERS, load_config

    for tag, spec in families.MODELS.items():
        assert spec.tag == tag
        assert re.fullmatch(r"[0-9a-f]{40}", C.REVISIONS[spec.hf_id]) and spec.revision == C.REVISIONS[spec.hf_id]
        assert f"family_score_{tag}" in RUNNERS
        cfg = load_config(f"family_score_{tag}")
        assert cfg["model"] == tag and cfg["n_shards"] == 1
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec.release) and spec.params_b > 0 and spec.system in ("native", "fold")
    assert "families_analysis" in RUNNERS
    assert load_config("families_analysis")["models"] == list(families.MODELS)
    for tag in load_config("families_analysis")["reference"]:
        assert tag in families.REFERENCE and families.REFERENCE[tag].hf_id in C.REVISIONS


def test_settings_each_model_needs():
    m = families.MODELS
    assert m["gemma2-9b-it"].system == "fold" and m["gemma2-9b-it"].attn_implementation == "eager"
    for tag in ("llama3.2-1b", "llama3.2-3b"):
        assert m[tag].template_kwargs == {"date_string": "26 Jul 2024"}  # Llama 3.1's fixed date, not today's
    assert m["mistral-7b-v0.3"].system == "native" and m["mistral-7b-v0.3"].attn_implementation is None


# ---------------- prompts ----------------


def test_fold_puts_the_system_text_at_the_start_of_the_first_user_turn(mock_score):
    msgs = sizes.arm_messages(_item(), "aligned")
    folded = families.fold_system(msgs)
    assert folded[0] == {"role": "user", "content": WM_SYSTEM + "\n\n" + msgs[1]["content"]}
    assert folded[1:] == msgs[2:]
    assert [m["role"] for m in folded] == ["user", "assistant"] * 3 + ["user"]
    with pytest.raises(ValueError):
        families.fold_system(msgs[1:])


@pytest.mark.parametrize("tag", list(families.MODELS))
def test_aligned_and_swapped_messages_differ_only_in_the_task_line(mock_score, tag):
    spec, item = families.MODELS[tag], GOLDEN["item"]
    for a, b in (("aligned", "swapped"), ("yn_aligned", "yn_swapped")):
        ma, mb = families.family_messages(spec, item, a), families.family_messages(spec, item, b)
        diff = [(x["content"], y["content"]) for x, y in zip(ma, mb) if x != y]
        assert len(ma) == len(mb) and len(diff) == 1
        assert diff[0][0].replace(item["task_line"], item["swapped_task_line"]) == diff[0][1]
    if spec.system == "fold":
        assert all(m["role"] != "system" for m in families.family_messages(spec, item, "absent"))


def _expected_render(messages: list[dict], family: str, date: str = "26 Jul 2024") -> str:
    """Each family's chat format as documented, built by hand from the message list."""
    if family == "gemma":
        body = "".join(f"<start_of_turn>{'model' if m['role'] == 'assistant' else 'user'}\n{m['content'].strip()}<end_of_turn>\n"
                       for m in messages)
        return "<bos>" + body + "<start_of_turn>model\n"
    if family == "mistral":
        system, rest = messages[0]["content"], messages[1:]
        out = "<s>"
        for i, m in enumerate(rest):
            if m["role"] == "user":
                text = f"{system}\n\n{m['content']}" if i == len(rest) - 1 else m["content"]
                out += f"[INST] {text}[/INST]"
            else:
                out += f" {m['content'].strip()}</s>"
        return out
    head = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
            f"Today Date: {date}\n\n{messages[0]['content'].strip()}<|eot_id|>")
    body = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>" for m in messages[1:])
    return head + body + "<|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.mark.parametrize("tag", list(families.MODELS))
def test_golden_rendered_prompt_is_the_documented_format(mock_score, tag):
    """The stored renders (made with the real pinned templates) equal the hand-built format for today's messages,
    so a change to a prompt, the fold or a template setting shows up here without the tokenizer."""
    spec, item = families.MODELS[tag], GOLDEN["item"]
    rec = RENDERED["models"][tag]
    assert rec["revision"] == C.REVISIONS[spec.hf_id]
    for arm in ("aligned", "swapped", "yn_aligned"):
        assert rec[arm] == _expected_render(families.family_messages(spec, item, arm), spec.family), arm
    assert rec["aligned"].replace(item["task_line"], item["swapped_task_line"]) == rec["swapped"]
    assert rec["aligned"].count(item["task_line"]) == 1


def _cached_tokenizer(spec):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=C.REVISIONS[spec.hf_id], local_files_only=True,
                                                          cache_dir=os.environ.get("WF_TEST_HF_CACHE"))
    except Exception as e:  # not cached here
        pytest.skip(f"{spec.hf_id} tokenizer not available offline at the pinned revision: {str(e)[:120]}")


@pytest.mark.parametrize("tag", list(families.MODELS))
def test_pinned_template_renders_the_golden_prompt_and_passes_the_gate(mock_score, tag):
    """Runs where transformers and the pinned tokenizer are cached (ICE after the prefetch, or WF_TEST_HF_CACHE)."""
    spec, item = families.MODELS[tag], GOLDEN["item"]
    tok = _cached_tokenizer(spec)
    kw = {"enable_thinking": False, **spec.template_kwargs}
    for arm in ("aligned", "swapped", "yn_aligned"):
        text = tok.apply_chat_template(families.family_messages(spec, item, arm), tokenize=False, add_generation_prompt=True, **kw)
        assert text == RENDERED["models"][tag][arm]
    ids = {arm: families._ids(tok.apply_chat_template(families.family_messages(spec, item, arm), tokenize=True,
                                                      add_generation_prompt=True, **kw)) for arm in ("aligned", "swapped")}
    a, s = ids["aligned"], ids["swapped"]
    head = next(i for i, (x, y) in enumerate(zip(a, s)) if x != y)
    tail = next(i for i, (x, y) in enumerate(zip(a[::-1], s[::-1])) if x != y)
    assert tok.decode(a[head:len(a) - tail]).strip() in item["goal_object"] and tok.decode(s[head:len(s) - tail]).strip() in item["swapped_goal_object"]
    pytest.importorskip("torch")  # the scorer's encoding returns tensors
    scorer = LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=tag, template_kwargs=spec.template_kwargs)
    rec = families.tokenization_checks(scorer, spec, [item], [])
    assert rec["ok"], rec["problems"]
    assert all(len(v["alone"]) == 1 and v["n_checked"] == 3 for v in rec["yes_no"].values())
    assert rec["first_token_pairs"] == {"'You' vs 'Nothing'": 1}


# ---------------- tokenization gate ----------------


class FakeTok:
    """A word-level stand-in for a chat tokenizer: '<s>', turns as '<u>'/'<a>' ... '<e>', words as ids."""

    def __init__(self, reply_prefix: str = "", yes_pieces: int = 1, first_same: bool = False, system_moves: bool = False,
                 odd_reply_after: tuple[tuple[str, str], ...] = ()):
        """`odd_reply_after`: (marker, word) pairs. In a conversation whose other turns contain `marker`, a final
        assistant reply starting with `word` is written with a '▁' glued to that word (one other token), as a
        template that renders a reply differently in one prompt would."""
        self.system_moves = system_moves
        self.odd_reply_after = odd_reply_after
        self.calls: list[dict] = []
        self.vocab: dict[str, int] = {"<s>": 0, "<e>": 1, "<u>": 2, "<a>": 3}
        self.all_special_ids = [0, 1, 2, 3]
        self.added_tokens_decoder = {}
        self.reply_prefix, self.yes_pieces, self.first_same = reply_prefix, yes_pieces, first_same

    def _id(self, w: str) -> int:
        return self.vocab.setdefault(w, len(self.vocab))

    def _words(self, text: str) -> list[str]:
        out = []
        for w in re.findall(r"<[seua]>|[^\s<]+", text):
            if w == "Yes" and self.yes_pieces == 2:
                out += ["Y", "es"]
            elif w in ("You", "Nothing") and self.first_same:  # both candidates open with the same piece
                out += ["W", w]
            else:
                out.append(w)
        return out

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors=None):
        return types.SimpleNamespace(input_ids=[self._id(w) for w in self._words(text)])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        self.calls.append(kw)
        if messages[-1]["role"] == "assistant":
            context = " ".join(m["content"] for m in messages[:-1])
            for marker, word in self.odd_reply_after:
                if marker in context and messages[-1]["content"].startswith(word):
                    messages = list(messages[:-1]) + [{"role": "assistant", "content": "▁" + messages[-1]["content"]}]
        if self.system_moves and messages[-1]["role"] == "assistant":
            messages = [{"role": "user", "content": "moved"}] + list(messages)
        text = "<s> " + " ".join(f"<{'a' if m['role'] == 'assistant' else 'u'}> "
                                 + (self.reply_prefix if m["role"] == "assistant" else "") + m["content"] + " <e>"
                                 for m in messages if m["role"] != "system") + (" <a>" if add_generation_prompt else "")
        return self(text).input_ids if tokenize else text

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def _scorer(tok, spec):
    """The real scorer's encoding (`_encode`, `_encode_candidate`, its template settings) over a stand-in tokenizer."""
    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs)


def _items(n_primary: int, n_successes: int, **changes) -> tuple[list[dict], list[dict]]:
    """Copies of the golden item with their own ids and actions; `changes[i]` updates item i (successes follow the
    primary items in the numbering)."""
    base = GOLDEN["item"]
    out = []
    for i in range(n_primary + n_successes):
        t = {**base, "id": f"t{i}", "action": f"take soapbar {i + 1} from cabinet 1", **changes.get(f"t{i}", {})}
        if i >= n_primary:
            t["true_obs"] = t["candidates"]["o_plus"]
        out.append(t)
    return out[:n_primary], out[n_primary:]


def test_gate_passes_a_consistent_tokenizer(mock_score):
    spec = families.MODELS["mistral-7b-v0.3"]
    rec = families.tokenization_checks(_scorer(FakeTok(), spec), spec, [GOLDEN["item"]], [])
    assert rec["ok"] and rec["n_candidates_checked"] == 2 and rec["first_token_pairs"] == {"'You' vs 'Nothing'": 1}
    assert rec["n_yes_no_prompts_checked"] == 3 and {w: v["n_checked"] for w, v in rec["yes_no"].items()} == {"Yes": 3, "No": 3}


def test_gate_covers_every_candidate_and_yes_no_prompt_of_the_shard(mock_score):
    """2 x (items) candidates, successes included, and Yes/No after every yes/no prompt that will be scored."""
    spec = families.MODELS["gemma2-9b-it"]
    primary, successes = _items(4, 3)
    rec = families.tokenization_checks(_scorer(FakeTok(), spec), spec, primary, successes)
    assert rec["ok"], rec["problems"]
    assert rec["n_candidates_checked"] == 2 * (len(primary) + len(successes)) == 14
    yn_jobs = [(t, a) for t, a in sizes.size_jobs(primary, successes) if a in sizes.YN_ARMS]
    assert rec["n_yes_no_prompts_checked"] == len(yn_jobs) == 3 * 4 + 2 * 3


def test_gate_finds_a_bad_candidate_in_a_late_success(mock_score):
    spec = families.MODELS["llama3.2-1b"]
    primary, successes = _items(2, 3, t4={"action": "take zzmark 1 from cabinet 1"})
    rec = families.tokenization_checks(_scorer(FakeTok(odd_reply_after=(("zzmark", "You"),)), spec), spec, primary, successes)
    assert not rec["ok"] and [p.split()[:2] for p in rec["problems"]] == [["t4", "o_plus:"]]


def test_gate_compares_the_scored_yes_with_the_templates_reply_in_every_yes_no_prompt(mock_score):
    """The template writes 'Yes' as another token only after one item's swapped yes/no prompt: the gate must check
    that prompt, and compare the token the scorer reads (tokenized alone) with that reply."""
    spec = families.MODELS["mistral-7b-v0.3"]
    primary, successes = _items(3, 1, t1={"swapped_task_line": "Your task is to: put two zzswap in garbagecan.",
                                          "swapped_goal_object": "zzswap"})
    rec = families.tokenization_checks(_scorer(FakeTok(odd_reply_after=(("zzswap", "Yes"),)), spec), spec, primary, successes)
    assert not rec["ok"] and len(rec["problems"]) == 1 and rec["problems"][0].startswith("t1 yn_swapped: 'Yes'")
    assert rec["yes_no"]["Yes"]["n_mismatched"] == 1 and rec["yes_no"]["No"]["n_mismatched"] == 0


def test_gate_renders_with_the_scorers_template_settings(mock_score):
    """Llama 3.2's fixed date (and thinking off) reaches every template call the gate makes."""
    spec = families.MODELS["llama3.2-3b"]
    tok = FakeTok()
    primary, successes = _items(2, 1)
    assert families.tokenization_checks(_scorer(tok, spec), spec, primary, successes)["ok"]
    assert len(tok.calls) > 20
    assert all(c.get("date_string") == "26 Jul 2024" and c.get("enable_thinking") is False for c in tok.calls), tok.calls


@pytest.mark.parametrize("bad, words", [
    ({"reply_prefix": "SPACE "}, "o_plus: alone"),  # the template writes a reply differently from the candidate alone
    ({"system_moves": True}, "does not begin with the prompt"),  # the prompt changes once a reply follows
    ({"yes_pieces": 2}, "'Yes' is not one token"),
    ({"first_same": True}, "share their first token"),
])
def test_gate_fails_loudly_on_a_mismatch(mock_score, tmp_path, bad, words):
    spec = families.MODELS["llama3.2-1b"]
    rec = families.tokenization_checks(_scorer(FakeTok(**bad), spec), spec, [GOLDEN["item"]], [])
    assert not rec["ok"] and any(words in p for p in rec["problems"]), rec["problems"]
    scorer = _scorer(FakeTok(**bad), spec)
    scorer.mock, scorer._model = False, types.SimpleNamespace(
        config=types.SimpleNamespace(_attn_implementation="sdpa"), dtype="torch.bfloat16")
    with pytest.raises(families.TokenizationMismatch):
        families._gate(spec)(tmp_path, scorer, [GOLDEN["item"]], [])
    assert json.loads((tmp_path / "tokenization_checks.json").read_text())["ok"] is False


def test_a_failed_gate_scores_nothing(mock_root):
    def gate(d, scorer, primary, successes):
        raise families.TokenizationMismatch("no")

    cfg = {"run_id": "family_score_llama3.2-1b", "model": "llama3.2-1b", "sources": ["rq12_transitions_smoke"], "n_shards": 1}
    with pytest.raises(families.TokenizationMismatch):
        sizes.score_shard(cfg, "llama3.2-1b", lambda: LogoddsScorer("meta-llama/Llama-3.2-1B-Instruct", model_tag="llama3.2-1b"),
                          lambda t, a: families.family_messages(families.MODELS["llama3.2-1b"], t, a), gate=gate)
    d = mock_root / "family_score_llama3.2-1b" / "shard_00_of_01"
    assert not (d / "scoring.jsonl").exists() and "FAIL" in (d / "STATUS.md").read_text().splitlines()[0]


def _gate_spy():
    calls = []

    def gate(d, scorer, primary, successes):
        calls.append(([t["id"] for t in primary], [t["id"] for t in successes]))
    return gate, calls


def test_runner_gives_the_gate_every_item_of_the_shard_and_runs_it_on_resume(mock_root):
    from wf import rq1x
    from wf.rq2 import load_items, shard_items

    spec = families.MODELS["llama3.2-1b"]
    cfg = {"run_id": "family_score_llama3.2-1b", "model": "llama3.2-1b", "sources": ["rq12_transitions_smoke"], "n_shards": 2,
           "shard": 1}
    sets = rq1x.rq1x_sets(load_items(("rq12_transitions_smoke",)))
    want = ([t["id"] for t in shard_items(sets["primary"], 1, 2)], [t["id"] for t in shard_items(sets["successes"], 1, 2)])
    assert len(want[0]) > 1 and want[1]
    gate, calls = _gate_spy()
    run = lambda c: sizes.score_shard(c, "llama3.2-1b", lambda: LogoddsScorer(spec.hf_id, model_tag="llama3.2-1b"),
                                      lambda t, a: families.family_messages(spec, t, a), gate=gate)
    assert run(cfg) == 0 and calls == [want]
    f = mock_root / "family_score_llama3.2-1b" / "shard_01_of_02" / "scoring.jsonl"
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:3]) + "\n")  # cut off after three scores
    assert run({**cfg, "resume": True}) == 0
    assert calls == [want, want]  # the gate ran again, on the whole shard, before the remaining scores
    assert len(f.read_text().splitlines()) == len(lines)


# ---------------- scorer settings ----------------


def _fake_hf(monkeypatch, loaded_attn):
    calls = {}

    class Model:
        def __init__(self, attn):
            self.config = types.SimpleNamespace(_attn_implementation=attn)

        def eval(self):
            return self

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["model"] = kw
            return Model(loaded_attn or kw.get("attn_implementation", "sdpa"))

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["tokenizer"] = kw
            tok = types.SimpleNamespace()

            def apply_chat_template(messages, **kw2):
                calls["template"] = kw2
                return [[1, 2]]
            tok.apply_chat_template = apply_chat_template
            return tok

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModelForCausalLM=AutoModelForCausalLM,
                                                                          AutoTokenizer=AutoTokenizer))
    monkeypatch.delenv("WF_MOCK_SCORE", raising=False)
    return calls


def test_scorer_loads_eager_attention_and_passes_template_settings(monkeypatch):
    calls = _fake_hf(monkeypatch, None)
    g = families.MODELS["gemma2-9b-it"]
    LogoddsScorer(g.hf_id, attn_implementation=g.attn_implementation, template_kwargs=g.template_kwargs)
    assert calls["model"]["attn_implementation"] == "eager" and calls["model"]["revision"] == C.REVISIONS[g.hf_id]
    assert calls["tokenizer"]["revision"] == C.REVISIONS[g.hf_id]  # tokenizer and template at the pin too
    l = families.MODELS["llama3.2-3b"]
    sc = LogoddsScorer(l.hf_id, attn_implementation=l.attn_implementation, template_kwargs=l.template_kwargs)
    assert "attn_implementation" not in calls["model"]
    assert calls["tokenizer"]["revision"] == C.REVISIONS[l.hf_id]
    sc._encode([{"role": "user", "content": "x"}])
    assert calls["template"]["date_string"] == "26 Jul 2024" and calls["template"]["enable_thinking"] is False
    assert calls["template"]["add_generation_prompt"] is True


def test_scorer_refuses_a_model_loaded_with_other_attention(monkeypatch):
    _fake_hf(monkeypatch, "sdpa")
    with pytest.raises(RuntimeError, match="eager"):
        LogoddsScorer("google/gemma-2-9b-it", attn_implementation="eager")


def test_family_scorer_is_opened_with_the_specs_settings(monkeypatch, mock_root):
    seen = {}

    class Spy(LogoddsScorer):
        def __init__(self, name, **kw):
            seen.update(kw, name=name)
            super().__init__(name, **kw)

    monkeypatch.setattr(families, "LogoddsScorer", Spy)
    cfg = {"run_id": "family_score_gemma2-9b-it", "model": "gemma2-9b-it", "sources": ["rq12_transitions_smoke"], "n_shards": 1}
    assert families.run_family_score(cfg) == 0
    assert seen["name"] == "google/gemma-2-9b-it" and seen["attn_implementation"] == "eager"
    seen.clear()
    assert families.run_family_score({**cfg, "run_id": "family_score_llama3.2-3b", "model": "llama3.2-3b"}) == 0
    assert seen["template_kwargs"] == {"date_string": "26 Jul 2024"}
    d = mock_root / "family_score_gemma2-9b-it" / "shard_00_of_01"
    assert json.loads((d / "config.json").read_text())["model_spec"]["revision"] == C.REVISIONS["google/gemma-2-9b-it"]
    assert json.loads((d / "tokenization_checks.json").read_text())["skipped"] == "mock"


# ---------------- analysis ----------------


def _s(tid, cond, L, game="g0"):
    return {"transition_id": tid, "condition": cond, "L": L, "L_first": L, "game_id": game, "first_mass": 0.9,
            "p_success": 1 / (1 + math.exp(-L)), "model": "m"}


def _t(i):
    return {"id": f"t{i}", "game_id": f"g{i % 5}", "context_determined": True, "history_goal_agnostic": True, "true_obs": NOOP}


def test_standardised_effect_divides_by_the_spread_under_the_swapped_task():
    items = [_t(i) for i in range(20)]
    swapped = [(-1.0 if i % 2 else 1.0) * 2.0 for i in range(20)]  # SD about 2.05
    s = [_s(t["id"], "swapped", v, t["game_id"]) for t, v in zip(items, swapped)]
    s += [_s(t["id"], "aligned", 2 * v + 3.0, t["game_id"]) for t, v in zip(items, swapped)]  # twice the spread
    out = families.standardised_effect(s, items, "aligned", "swapped")
    import statistics
    assert out["estimate"] == pytest.approx(3.0 / statistics.stdev(swapped)) and out["n_items"] == 20
    assert out["ci"][0] <= out["estimate"] <= out["ci"][1]


def _block_data():
    """Twelve failures and two successes. Full L and first-token L disagree on purpose: three items flip in full L
    toward success without a first-token sign change, and six change first-token sign without a full-L flip. The
    yes/no arms carry an effect of their own (4 nats), unlike the two-candidate arms (1 nat)."""
    items = [_t(i) for i in range(12)]
    successes = [{**_t(i), "true_obs": "You pick up the mug 1."} for i in range(12, 14)]
    s = []
    for i, t in enumerate(items):
        game = t["game_id"]
        if i < 3:  # full L: +1 vs -1 (a flip toward success); first token: +1 vs +2 (no sign change)
            rows = {"aligned": (1.0, 1.0), "swapped": (-1.0, 2.0)}
        elif i < 9:  # full L: -1 vs -2 (no flip); first token: -1 vs +1 (a sign change)
            rows = {"aligned": (-1.0, -1.0), "swapped": (-2.0, 1.0)}
        else:
            rows = {"aligned": (-1.0, -1.0), "swapped": (-1.5, -1.0)}
        rows["absent"] = rows["swapped"]
        for cond, (L, Lf) in rows.items():
            s.append({**_s(t["id"], cond, L, game), "L_first": Lf})
        sw = (i % 3) - 1.0
        s += [_s(t["id"], "yn_swapped", sw, game), _s(t["id"], "yn_aligned", sw + 4.0 + 0.5 * (i % 2), game),
              _s(t["id"], "yn_absent", sw, game)]
    for t in successes:
        s += [_s(t["id"], c, 2.0, t["game_id"]) for c in sizes.SUCCESS_ARMS]
    return s, items, successes


def test_flip_definitions_are_pinned_to_their_names():
    s, items, successes = _block_data()
    b = families.family_block(s, items, successes, families.MODELS["mistral-7b-v0.3"])
    assert b["flips"]["flip_full_L_toward_success"] == pytest.approx(3 / 12)
    assert b["flips"]["flip_first_token_sign_change"]["share"] == pytest.approx(6 / 12)
    assert set(b["flips"]["definitions"]) == {"flip_full_L_toward_success", "flip_first_token_sign_change"}


def test_standardised_yes_no_effect_reads_the_yes_no_arms():
    s, items, successes = _block_data()
    b = families.family_block(s, items, successes, families.MODELS["mistral-7b-v0.3"])
    yn = families.standardised_effect(s, items, "yn_aligned", "yn_swapped")
    assert b["standardised_effect"]["yes_no"]["estimate"] == pytest.approx(yn["estimate"])
    assert b["standardised_effect"]["yes_no"]["estimate"] > 4  # 4.25 nats over an SD of about 0.85
    assert b["standardised_effect"]["two_candidate"]["estimate"] == pytest.approx(
        families.standardised_effect(s, items, "aligned", "swapped")["estimate"])


def test_every_model_is_read_on_the_family_runs_population(mock_score):
    """A reference run's extra records (a success under the swapped task, an item outside both sets) are dropped."""
    primary, successes = _items(1, 1)  # t0 a failure, t1 a success
    s = [_s("t0", "swapped", 1.0), _s("t0", "yn_swapped", 1.0), _s("t1", "aligned", 1.0), _s("t1", "swapped", 1.0),
         _s("t1", "yn_swapped", 1.0), _s("t9", "aligned", 1.0)]
    kept, dropped = families.same_population(s, primary, successes)
    assert sorted((x["transition_id"], x["condition"]) for x in kept) == [("t0", "swapped"), ("t0", "yn_swapped"), ("t1", "aligned")]
    assert dropped == 3


def test_placement_block_compares_the_two_placements_on_the_same_items():
    items = [_t(i) for i in range(10)]
    native, first = [], []
    for t in items:
        for rows, d2, dyn in ((native, 2.0, 1.0), (first, 3.0, 0.5)):
            rows += [_s(t["id"], "aligned", d2, t["game_id"]), _s(t["id"], "swapped", 0.0, t["game_id"]),
                     _s(t["id"], "yn_aligned", dyn, t["game_id"]), _s(t["id"], "yn_swapped", 0.0, t["game_id"])]
    native += [_s("t99", "aligned", 50.0), _s("t99", "swapped", 0.0)]  # scored natively only: left out
    b = families.placement_block(native, first, items + [_t(99)])
    assert b["n_items"] == 10
    assert b["two_candidate"]["native"]["delta_L_nats"] == pytest.approx(2.0)
    assert b["two_candidate"]["first_turn"]["delta_L_nats"] == pytest.approx(3.0)
    assert b["two_candidate"]["first_turn_minus_native"]["delta_L_nats"] == pytest.approx(1.0)
    assert b["yes_no"]["first_turn_minus_native"]["delta_L_nats"] == pytest.approx(-0.5)


def test_placement_items_are_seeded_and_order_free():
    items = [{"id": f"t{i:03d}"} for i in range(400)]
    a = families.placement_items(items)
    assert len(a) == 300 and a == families.placement_items(items[::-1]) and a == sorted(a, key=lambda t: t["id"])
    assert families.placement_items(items[:10]) == sorted(items[:10], key=lambda t: t["id"])


def test_placement_run_folds_mistrals_system_text_into_the_first_turn(mock_root, monkeypatch):
    seen = []
    real = LogoddsScorer.score_pair

    def spy(self, transition_id, condition, messages, o_plus, o_minus):
        seen.append((self.model_tag, condition, messages))
        return real(self, transition_id, condition, messages, o_plus, o_minus)

    monkeypatch.setattr(LogoddsScorer, "score_pair", spy)
    cfg = {"run_id": "family_placement_mistral-7b-v0.3", "model": "mistral-7b-v0.3", "n_items": 4,
           "sources": ["rq12_transitions_smoke"], "n_shards": 1}
    assert families.run_family_placement(cfg) == 0
    d = mock_root / "family_placement_mistral-7b-v0.3" / "shard_00_of_01"
    rows = [json.loads(x) for x in (d / "scoring.jsonl").read_text().splitlines()]
    assert {x["model"] for x in rows} == {"mistral-7b-v0.3-firstturn"}
    assert {x["condition"] for x in rows} == set(families.PLACEMENT_ARMS) and len({x["transition_id"] for x in rows}) == 4
    assert len(rows) == 16
    assert all(m[0]["role"] == "user" and m[0]["content"].startswith(WM_SYSTEM) for _, _, m in seen)
    assert json.loads((d / "config.json").read_text())["model_spec"]["system"] == "fold"
    with pytest.raises(ValueError):
        families.first_turn_spec(families.MODELS["gemma2-9b-it"])


# ---------------- primary metrics, competence gate, comparisons (amendment, pre-data) ----------------


def _yn_data(goal=(2.0,) * 10, fail_neutral=-3.0, succ_neutral=1.0, scale=1.0, n_succ=4):
    """Yes/no scores: failure i has L(aligned) - L(swapped) = goal[i] and neutral L `fail_neutral`; successes have
    neutral L `succ_neutral`. Every L is multiplied by `scale` (a temperature)."""
    items = [_t(i) for i in range(len(goal))]
    successes = [{**_t(100 + i), "true_obs": "You pick up the mug 1."} for i in range(n_succ)]
    s = []
    for t, g in zip(items, goal):
        s += [_s(t["id"], "yn_swapped", -1.0 * scale, t["game_id"]), _s(t["id"], "yn_aligned", (g - 1.0) * scale, t["game_id"]),
              _s(t["id"], "yn_absent", fail_neutral * scale, t["game_id"])]
    s += [_s(t["id"], "yn_absent", succ_neutral * scale, t["game_id"]) for t in successes]
    return s, items, successes


def test_goal_to_evidence_ratio_and_item_share():
    goal = (2.0,) * 8 + (-1.0, 0.0)  # one item moved away from Yes, one not moved (a tie is not "toward Yes")
    s, items, succ = _yn_data(goal)
    pm = families.primary_metrics(s, items, succ)
    r = pm["goal_to_evidence_ratio"]
    assert r["goal_effect_nats"] == pytest.approx(1.5) and r["evidence_effect_nats"] == pytest.approx(4.0)
    assert r["estimate"] == pytest.approx(1.5 / 4.0) and r["n_failures"] == 10 and r["n_successes"] == 4
    assert r["ci"][0] <= r["estimate"] <= r["ci"][1]
    sh = pm["item_share_toward_yes"]
    assert sh["estimate"] == pytest.approx(0.8) and sh["n_items"] == 10 and sh["null"] == 0.5
    # rescaling every logit leaves R and the share unchanged; ΔL scales
    pm3 = families.primary_metrics(*_yn_data(goal, scale=3.0))
    assert pm3["goal_to_evidence_ratio"]["estimate"] == pytest.approx(r["estimate"])
    assert pm3["item_share_toward_yes"]["estimate"] == pytest.approx(0.8)
    # no evidence effect (successes no higher than failures): R undefined, not huge
    assert math.isnan(families.primary_metrics(*_yn_data(goal, succ_neutral=-3.0))["goal_to_evidence_ratio"]["estimate"])


def test_primary_p_values_are_game_clustered_sign_flips():
    s, items, succ = _yn_data(tuple(0.5 if i % 2 else -0.5 for i in range(10)))  # no net effect, share 0.5
    pm = families.primary_metrics(s, items, succ)
    assert pm["item_share_toward_yes"]["estimate"] == pytest.approx(0.5) and pm["item_share_toward_yes"]["p_two_sided"] > 0.5
    s, items, succ = _yn_data(tuple(1.0 + 0.1 * i for i in range(40)))
    assert families.primary_metrics(s, items, succ)["goal_to_evidence_ratio"]["p_two_sided"] < 0.1


def test_flips_are_shares_of_argmax_answers():
    items = [_t(i) for i in range(4)]
    s = []
    for t, (sw, al) in zip(items, ((-1.0, 2.0), (-0.1, 0.1), (1.0, 3.0), (0.5, -0.5))):  # No->Yes twice, Yes->Yes, Yes->No
        s += [_s(t["id"], "yn_swapped", sw, t["game_id"]), _s(t["id"], "yn_aligned", al, t["game_id"]),
              _s(t["id"], "yn_absent", -1.0, t["game_id"])]
    a = families.argmax_answers(s, items, [])
    assert a["flip_no_to_yes"]["share"] == pytest.approx(0.5) and a["flip_yes_to_no"]["share"] == pytest.approx(0.25)
    assert a["yes_share_failures"] == {"yn_aligned": 0.75, "yn_swapped": 0.5, "yn_absent": 0.0}


def _gate_block(auc=0.8, yes_f=0.4, yes_s=0.9, mass=0.9):
    return {"yes_no": {"auc": {"yn_absent": {"auc": auc}}},
            "argmax_answers": {"yes_share_failures": {"yn_absent": yes_f, "yn_aligned": 0.99},
                               "yes_share_successes": {"yn_absent": yes_s, "yn_aligned": 0.0}},
            "first_token_coverage": {c: {"median": mass if c == "yn_absent" else 0.01} for c in sizes.YN_ARMS + ("aligned",)}}


@pytest.mark.parametrize("kw, ok", [
    ({}, True), ({"auc": 0.70}, True), ({"auc": 0.69}, False), ({"auc": float("nan")}, False),
    ({"yes_f": 0.0015}, True),  # No to nearly every failure is accuracy, not a defect (Qwen3-8B: 0.15%)
    ({"yes_f": 0.95}, True), ({"yes_f": 0.951}, False),  # Yes to (nearly) everything
    ({"yes_s": 0.05}, True), ({"yes_s": 0.049}, False),  # No to (nearly) everything
    ({"mass": 0.5}, True), ({"mass": 0.49}, False), ({"yes_s": float("nan")}, False)])
def test_competence_gate(kw, ok):
    """Read under the neutral task only: the other tasks' shares and coverage (set to extremes here) do not count."""
    g = families.competence_gate(_gate_block(**kw))
    assert g["interpretable"] is ok
    assert set(g["checks"]) == {"auc_neutral", "yes_share_neutral_failures", "yes_share_neutral_successes",
                                "median_yes_no_mass_neutral"}


@pytest.mark.parametrize("factor, matches, differs", [(1.0, True, False), (1.4, True, True), (1 / 1.4, True, True),
                                                      (1.6, False, True), (1 / 1.6, False, True)])
def test_equivalence_of_R_is_a_tost_within_a_factor_of_1_5(factor, matches, differs):
    """Model B is model A with every goal effect times `factor` and every logit times 3 (R_B = factor x R_A)."""
    goal = tuple(1.0 + 0.2 * (i % 3) for i in range(15))
    sa, items, succ = _yn_data(goal)
    sb, _, _ = _yn_data(tuple(g * factor for g in goal), scale=3.0)
    c = families.compare_models(sb, sa, items, succ)
    assert c["R_ratio"]["estimate"] == pytest.approx(factor)
    assert c["R_ratio"]["matches"] is matches and c["R_ratio"]["differs"] is differs
    assert c["n_failures"] == 15 and c["n_successes"] == 4


def test_tost_needs_the_90_percent_interval_inside_the_margin():
    """Four games that disagree: a ratio of 1 with a wide paired interval is neither a match nor a difference."""
    def model(goals):
        s, items = [], []
        for g, goal in enumerate(goals):
            for k in range(3):
                t = {**_t(10 * g + k), "game_id": f"g{g}"}
                items.append(t)
                s += [_s(t["id"], "yn_swapped", 0.0, t["game_id"]), _s(t["id"], "yn_aligned", goal, t["game_id"]),
                      _s(t["id"], "yn_absent", -2.0, t["game_id"])]
        succ = [{**_t(100 + g), "game_id": f"g{g}", "true_obs": "You pick up the mug 1."} for g in range(len(goals))]
        return s + [_s(t["id"], "yn_absent", 2.0, t["game_id"]) for t in succ], items, succ

    sa, items, succ = model([4.0, 0.2, 4.0, 0.2])
    sb, _, _ = model([0.2, 4.0, 0.2, 4.0])
    c = families.compare_models(sa, sb, items, succ)
    assert c["R_ratio"]["estimate"] == pytest.approx(1.0)
    assert c["R_ratio"]["ci90"][0] < 1 / 1.5 and c["R_ratio"]["ci90"][1] > 1.5
    assert not c["R_ratio"]["matches"] and not c["R_ratio"]["differs"]


def test_comparison_differences_and_the_noise_floor():
    goal = tuple(1.0 + 0.1 * (i % 4) for i in range(12))
    sa, items, succ = _yn_data(goal)
    sb, _, _ = _yn_data(tuple(g + 0.2 for g in goal))
    c = families.compare_models(sb, sa, items, succ)
    assert c["delta_L_difference_nats"]["estimate"] == pytest.approx(0.2) and c["delta_L_difference_nats"]["within_noise_floor"]
    sc, _, _ = _yn_data(tuple(g + 0.5 for g in goal))
    assert not families.compare_models(sc, sa, items, succ)["delta_L_difference_nats"]["within_noise_floor"]
    assert c["item_share_difference"]["estimate"] == pytest.approx(0.0)


def test_comparisons_pair_the_same_items():
    sa, items, succ = _yn_data((2.0,) * 10)
    sb = [x for x in sa if x["transition_id"] != "t3"]  # model B lacks one item
    sa_extra = sa + [_s("t3", "yn_aligned", 99.0)]
    c = families.compare_models(sa_extra, sb, items, succ)
    assert c["n_failures"] == 9 and c["R_ratio"]["estimate"] == pytest.approx(1.0)


def _pm_block(p_share, p_r, share=0.7, r=0.5, ok=True):
    return {"primary_metrics": {"item_share_toward_yes": {"estimate": share, "ci": [0, 1], "null": 0.5, "p_two_sided": p_share},
                                "goal_to_evidence_ratio": {"estimate": r, "ci": [0, 1], "null": 0.0, "p_two_sided": p_r}},
            "competence": {"interpretable": ok}}


def test_primary_family_is_holm_corrected_and_respects_the_gate():
    from wf.analysis.stats import holm

    blocks = {"a": _pm_block(0.001, 0.04), "b": _pm_block(0.01, 0.03, share=0.4), "c": _pm_block(0.0001, 0.0001, ok=False)}
    fam = families.primary_family(blocks, ["a", "b", "c", "d"])
    t = fam["tests"]
    want = holm({"a:item_share_toward_yes": 0.001, "a:goal_to_evidence_ratio": 0.04, "b:item_share_toward_yes": 0.01,
                 "b:goal_to_evidence_ratio": 0.03, "c:item_share_toward_yes": 0.0001, "c:goal_to_evidence_ratio": 0.0001})
    assert {k: v["p_holm"] for k, v in t.items()} == pytest.approx(want)
    assert t["a:item_share_toward_yes"]["verdict"] == "supported"
    assert t["a:goal_to_evidence_ratio"]["verdict"] == "not supported"  # 0.04 becomes 0.06 after Holm
    assert t["b:item_share_toward_yes"]["verdict"] == "not supported"  # significant, but on the wrong side
    assert t["c:goal_to_evidence_ratio"]["verdict"] == "not interpretable"
    assert fam["n_tests_planned"] == 8 and not fam["complete"]


def test_prediction_check_needs_both_formats_above_zero():
    block = lambda a, b: {"goal_effect": {"delta_L_ci": a}, "yes_no": {"goal_effect": {"delta_L_ci": b}}}
    assert families.prediction_check(block([0.1, 1], [0.2, 2]))["holds"]
    out = families.prediction_check(block([0.1, 1], [-0.2, 2]))
    assert not out["holds"] and out["two_candidate_ci_above_0"] and not out["yes_no_ci_above_0"]


def test_reference_scores_keep_one_record_per_item_and_condition(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for run, rows in (("ra", [_s("t1", "aligned", 1.0), _s("t1", "yn_aligned", 5.0)]),
                      ("rb", [_s("t1", "aligned", 1.25), _s("t2", "aligned", 2.0), {**_s("t3", "aligned", 9.0), "model": "x"}])):
        p = tmp_path / run / "shard_00_of_01"
        p.mkdir(parents=True)
        (p / "STATUS.md").write_text(f"# STATUS — {run}/shard_00_of_01 — REPORTED (shard complete)\n")
        (p / "git_sha").write_text("abc\n")
        (p / "scoring.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s, read, info = families.reference_scores({"model": "m", "runs": {"ra": ["aligned"], "rb": ["aligned"]}})
    assert sorted((x["transition_id"], x["condition"], x["L"]) for x in s) == [("t1", "aligned", 1.0), ("t2", "aligned", 2.0)]
    assert info["repeats"] == 1 and info["max_abs_L_difference"] == pytest.approx(0.25) and set(read) == {"ra", "rb"}
    _, _, missing = families.reference_scores({"model": "m", "runs": {"nope": ["aligned"]}})
    assert "nope" in missing["problems"]


def test_score_then_analysis_end_to_end(mock_root):
    from wf import rq1x
    from wf.rq2 import load_items

    for tag in ("gemma2-9b-it", "llama3.2-1b"):
        cfg = {"run_id": f"family_score_{tag}", "model": tag, "sources": ["rq12_transitions_smoke"], "n_shards": 1}
        assert families.run_family_score(cfg) == 0
    assert sizes.run_size_score({"run_id": "size_score_qwen3-8b", "model": "qwen3-8b", "sources": ["rq12_transitions_smoke"],
                                 "n_shards": 1}) == 0
    gemma = mock_root / "family_score_gemma2-9b-it" / "shard_00_of_01" / "scoring.jsonl"
    s = [json.loads(x) for x in gemma.read_text().splitlines()]
    assert {x["model"] for x in s} == {"gemma2-9b-it"} and {x["model_revision"] for x in s} == {C.REVISIONS["google/gemma-2-9b-it"]}
    assert {x["condition"] for x in s} == set(families.ARMS)
    # another model's records in the run's file are not the model's
    with gemma.open("a") as f:
        f.write("".join(json.dumps({**x, "model": "other", "L": x["L"] + 50.0}) + "\n" for x in s if x["condition"] == "aligned"))
    # a reference run's success scored under the swapped task is outside the family population
    sets = rq1x.rq1x_sets(load_items(("rq12_transitions_smoke",)))
    qwen = mock_root / "size_score_qwen3-8b" / "shard_00_of_01" / "scoring.jsonl"
    extra = [{**_s(t["id"], "swapped", -3.0, t["game_id"]), "model": "qwen3-8b", "first_mass": 0.01} for t in sets["successes"]]
    with qwen.open("a") as f:
        f.write("".join(json.dumps(x) + "\n" for x in extra))
    acfg = {"run_id": "families_smoke_analysis", "sources": ["rq12_transitions_smoke"],
            "models": ["gemma2-9b-it", "llama3.2-1b", "mistral-7b-v0.3"],
            "reference": {"qwen3-8b": {"model": "qwen3-8b", "runs": {"size_score_qwen3-8b": list(families.ARMS)}}},
            "placement": {"mistral-7b-v0.3": "family_placement_mistral-7b-v0.3"}}
    assert families.run_families_analysis(acfg) == 0
    m = json.loads((mock_root / "families_smoke_analysis" / "metrics.json").read_text())
    assert set(m["models"]) == {"gemma2-9b-it", "llama3.2-1b"} and "mistral-7b-v0.3" in m["waiting"]
    assert "placement mistral-7b-v0.3" in m["waiting"] and "yes/no format is the primary" in m["formats"]
    g = m["models"]["gemma2-9b-it"]
    assert set(g["primary_metrics"]) == set(families.PRIMARY_METRICS) and "interpretable" in g["competence"]
    assert set(m["reference"]["qwen3-8b"]["primary_metrics"]) == set(families.PRIMARY_METRICS)
    assert set(m["primary_family"]["tests"]) == {f"{t}:{x}" for t in ("gemma2-9b-it", "llama3.2-1b") for x in families.PRIMARY_METRICS}
    assert set(m["comparisons"]) == {"gemma2-9b-it vs qwen3-8b", "llama3.2-1b vs qwen3-8b"}
    clean = [x for x in s if x["condition"] in ("aligned", "swapped")]
    from wf.runs import _effect
    assert m["models"]["gemma2-9b-it"]["goal_effect"]["delta_L_nats"] == pytest.approx(
        _effect(clean, sets["primary"])["delta_L_nats"])
    n_swapped_primary = sum(1 for t in sets["primary"] if t.get("swapped_task_line"))
    assert m["reference"]["qwen3-8b"]["first_token_coverage"]["swapped"]["n"] == n_swapped_primary
    assert m["reference"]["qwen3-8b"]["read"]["records_outside_population"] == len(extra) > 0
    g = m["models"]["gemma2-9b-it"]
    assert g["spec"]["attn_implementation"] == "eager" and g["spec"]["release"] == "2024-06-27"
    assert set(g["flips"]) == {"flip_full_L_toward_success", "flip_first_token_sign_change", "definitions"}
    assert set(g["standardised_effect"]) == {"two_candidate", "yes_no"}
    assert {"yn_aligned", "aligned"} <= set(g["first_token_coverage"])
    assert g["tokenization_gate"]["shard_00_of_01"]["skipped"] == "mock"
    assert set(m["prediction_check"]) == {"gemma2-9b-it", "llama3.2-1b"} and "holds" in m["prediction_check"]["llama3.2-1b"]
    # the reference block is the size series' own block for the same scores
    q = m["reference"]["qwen3-8b"]
    assert q["read"]["repeats"] == 0 and q["spec"]["hf_id"] == "Qwen/Qwen3-8B"
    assert m["provenance"]["runs_read"]
    # the placement check, once Mistral's run and its first-turn run are complete
    for run, fn in (("family_score_mistral-7b-v0.3", families.run_family_score),
                    ("family_placement_mistral-7b-v0.3", families.run_family_placement)):
        assert fn({"run_id": run, "model": "mistral-7b-v0.3", "n_items": 5, "sources": ["rq12_transitions_smoke"],
                   "n_shards": 1}) == 0
    assert families.run_families_analysis(acfg) == 0
    m = json.loads((mock_root / "families_smoke_analysis" / "metrics.json").read_text())
    pl = m["placement"]["mistral-7b-v0.3"]
    assert not m["waiting"] and pl["n_items"] == 5 and pl["run"] == "family_placement_mistral-7b-v0.3"
    for fmt in ("two_candidate", "yes_no"):
        assert pl[fmt]["native"]["n_items"] == pl[fmt]["first_turn"]["n_items"] == pl[fmt]["first_turn_minus_native"]["n_items"] == 5


# ---------------- prefetch ----------------


def test_prefetch_reports_each_model_and_goes_on_after_a_refusal(monkeypatch, capsys, tmp_path):
    from wf.tools import prefetch_models as pm

    class HTTPError(Exception):
        def __init__(self, code):
            super().__init__(f"{code} Client Error")
            self.response = types.SimpleNamespace(status_code=code)

    calls = []

    def fake_download(repo_id, revision, cache_dir, allow_patterns, ignore_patterns):
        calls.append(repo_id)
        assert {"original/*", "consolidated*"} <= set(ignore_patterns) and "*.safetensors" in allow_patterns
        if repo_id.startswith("meta-llama/"):
            raise HTTPError(403)  # access still pending
        p = tmp_path / repo_id.replace("/", "--") / "snapshots" / revision
        p.mkdir(parents=True)
        (p / "config.json").write_text("{}")
        return str(p)

    monkeypatch.setattr(pm, "_snapshot_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "hf_secretvalue")
    code = pm.main(["--out", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert code == 1 and calls == [s.hf_id for s in families.MODELS.values()]  # every model tried
    for tag, spec in families.MODELS.items():
        line = next(ln for ln in out.splitlines() if ln.startswith(tag))
        assert line.endswith("HTTP 403") if spec.hf_id.startswith("meta-llama/") else line.endswith("OK")
    assert "secretvalue" not in out and "HF_TOKEN: set" in out
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["gemma2-9b-it"]["status"] == "OK" and saved["llama3.2-1b"]["status"] == "HTTP 403"


def test_prefetch_refuses_a_snapshot_at_another_revision(tmp_path):
    from wf.tools import prefetch_models as pm

    p = tmp_path / "snapshots" / ("b" * 40)
    p.mkdir(parents=True)
    assert pm.prefetch("x/y", "a" * 40, None, download=lambda **kw: str(p))["status"] == "ERROR wrong revision"
