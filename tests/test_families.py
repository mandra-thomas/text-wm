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
    rec = families.tokenization_checks(tok, spec, [item], [])
    assert rec["ok"], rec["problems"]
    assert all(len(v["alone"]) == 1 for v in rec["yes_no"].values())
    assert rec["first_token_pairs"] == {"'You' vs 'Nothing'": 1}


# ---------------- tokenization gate ----------------


class FakeTok:
    """A word-level stand-in for a chat tokenizer: '<s>', turns as '<u>'/'<a>' ... '<e>', words as ids."""

    def __init__(self, reply_prefix: str = "", yes_pieces: int = 1, first_same: bool = False, system_moves: bool = False):
        self.system_moves = system_moves
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

    def __call__(self, text: str, add_special_tokens: bool = False):
        return types.SimpleNamespace(input_ids=[self._id(w) for w in self._words(text)])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        if self.system_moves and messages[-1]["role"] == "assistant":
            messages = [{"role": "user", "content": "moved"}] + list(messages)
        text = "<s> " + " ".join(f"<{'a' if m['role'] == 'assistant' else 'u'}> "
                                 + (self.reply_prefix if m["role"] == "assistant" else "") + m["content"] + " <e>"
                                 for m in messages if m["role"] != "system") + (" <a>" if add_generation_prompt else "")
        return self(text).input_ids if tokenize else text

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def test_gate_passes_a_consistent_tokenizer(mock_score):
    rec = families.tokenization_checks(FakeTok(), families.MODELS["mistral-7b-v0.3"], [GOLDEN["item"]], [])
    assert rec["ok"] and rec["n_candidates_checked"] == 2 and rec["first_token_pairs"] == {"'You' vs 'Nothing'": 1}


@pytest.mark.parametrize("bad, words", [
    ({"reply_prefix": "SPACE "}, "o_plus: alone"),  # the template writes a reply differently from the candidate alone
    ({"system_moves": True}, "does not begin with the prompt"),  # the prompt changes once a reply follows
    ({"yes_pieces": 2}, "'Yes' is not the template's single token"),
    ({"first_same": True}, "share their first token"),
])
def test_gate_fails_loudly_on_a_mismatch(mock_score, tmp_path, bad, words):
    spec = families.MODELS["llama3.2-1b"]
    rec = families.tokenization_checks(FakeTok(**bad), spec, [GOLDEN["item"]], [])
    assert not rec["ok"] and any(words in p for p in rec["problems"]), rec["problems"]
    scorer = types.SimpleNamespace(mock=False, _tok=FakeTok(**bad), _model=types.SimpleNamespace(
        config=types.SimpleNamespace(_attn_implementation="sdpa"), dtype="torch.bfloat16"))
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
    l = families.MODELS["llama3.2-3b"]
    sc = LogoddsScorer(l.hf_id, attn_implementation=l.attn_implementation, template_kwargs=l.template_kwargs)
    assert "attn_implementation" not in calls["model"]
    sc._encode([{"role": "user", "content": "x"}])
    assert calls["template"]["date_string"] == "26 Jul 2024" and calls["template"]["enable_thinking"] is False


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
    for tag in ("gemma2-9b-it", "llama3.2-1b"):
        cfg = {"run_id": f"family_score_{tag}", "model": tag, "sources": ["rq12_transitions_smoke"], "n_shards": 1}
        assert families.run_family_score(cfg) == 0
    assert sizes.run_size_score({"run_id": "size_score_qwen3-8b", "model": "qwen3-8b", "sources": ["rq12_transitions_smoke"],
                                 "n_shards": 1}) == 0
    s = [json.loads(x) for x in (mock_root / "family_score_gemma2-9b-it" / "shard_00_of_01" / "scoring.jsonl").read_text().splitlines()]
    assert {x["model"] for x in s} == {"gemma2-9b-it"} and {x["model_revision"] for x in s} == {C.REVISIONS["google/gemma-2-9b-it"]}
    assert {x["condition"] for x in s} == set(families.ARMS)
    acfg = {"run_id": "families_smoke_analysis", "sources": ["rq12_transitions_smoke"],
            "models": ["gemma2-9b-it", "llama3.2-1b", "mistral-7b-v0.3"],
            "reference": {"qwen3-8b": {"model": "qwen3-8b", "runs": {"size_score_qwen3-8b": list(families.ARMS)}}}}
    assert families.run_families_analysis(acfg) == 0
    m = json.loads((mock_root / "families_smoke_analysis" / "metrics.json").read_text())
    assert set(m["models"]) == {"gemma2-9b-it", "llama3.2-1b"} and "mistral-7b-v0.3" in m["waiting"]
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
