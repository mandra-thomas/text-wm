"""TextCraft round 3, pre-design compatibility check (notes/textcraft_r3_proposal.md, section 5).

Tokenizers and chat templates only: no model weights are loaded and nothing is scored. It runs round 2's frozen records
(wf/textcraft/data/r2_items.jsonl, sha256 e6da991f...) and round 2's prompts (wf.textcraft.r2.messages, the clean
system text, the anchored yes/no question, the verified examples) through each candidate model's pinned tokenizer and
template, and checks what round 2's tokenization gate (design section 3) and wave B's additions
(notes/families_waveb_design.md, section 7) require, plus three things round 3 needs to know before it is designed:

- the gate itself, on every record and arm: each candidate tokenized alone equals the template's reply tokens after
  that arm's prompt (with a declared generation suffix stripped, as wave B does for Gemma 4 31B); "Yes" and "No" are
  single tokens equal to the template's reply after every yes/no prompt; the two candidates' first tokens differ;
- wave B's prompt checks: every prompt ends with the model's declared thinking-off ending, the thinking switch is read
  (thinking on renders differently and carries its marker), no forbidden default text (Ministral 3's default system
  prompt), and for Ministral 3 the template's ids equal mistral-common's encoding of the same messages;
- arm isolation in tokens: for every pair of arms that differ only in the goal line, the token ids differ only inside
  that line (no merge across its boundary changes a neighbouring token);
- input length against the context window and against each sliding-attention window;
- candidate segmentation: token counts of the success and failure replies, and the first-token pairs;
- for reasoning-oriented models (which cannot be scored directly after the generation prompt), the template's
  generation-prompt ending and whether every candidate and "Yes"/"No" tokenize without merging across the end of a
  closed reasoning block, which a score after a generated trace needs.

Also recorded: what apply_chat_template(tokenize=True) returns (transformers 5 returns a BatchEncoding, wave B review 1
M1), and, for Qwen3.8, how many prompts carry empty think blocks in their example replies.

Round-2 code is imported, never changed. Tokenizers are read from local folders named <org>__<name>@<sha>, downloaded
beforehand at the pinned revisions (config and tokenizer files only). Run offline:

    python scripts/textcraft_r3_compat.py <tokenizer root> <tag> <out.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wf.families import _ids, _special_ids  # noqa: E402
from wf.textcraft import prompts as P  # noqa: E402
from wf.textcraft import r2 as R  # noqa: E402
from wf.textcraft.r2_items import R2_ITEMS, read_records  # noqa: E402
from wf.textcraft.r2_tokcheck import golden_cases  # noqa: E402

R2_ITEMS_SHA256 = "e6da991f9eb6c927213d31a6c22814c8080db1945543060e67779eb7e47c22f0"
QWEN_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_ON = "<|im_start|>assistant\n<think>\n"
GEMMA_SUFFIX = "<|channel>thought\n<channel|>"

# Candidate models. "instruct": scored directly after the generation prompt with thinking off (round 2's protocol).
# "reasoning": always reasons first, so only the template and the boundary after a closed reasoning block are checked.
SPECS: dict[str, dict[str, Any]] = {
    "qwen3.5-9b": dict(hf="Qwen/Qwen3.5-9B", rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a", kind="instruct",
                       off_ending=QWEN_OFF, on_ending=QWEN_ON, markers=(), window=None),
    "qwen3.6-27b": dict(hf="Qwen/Qwen3.6-27B", rev="1cda14dd71a3f47115850eb790f416f165da1cb5", kind="instruct",
                        off_ending=QWEN_OFF, on_ending=QWEN_ON, markers=(), window=None),
    "qwen3.8-27b": dict(hf="Qwen/Qwen3.8-27B", rev="37561816e66b5b9153b1b64c10968c18c1e6f4a0", kind="instruct",
                        off_ending=QWEN_OFF, on_ending=QWEN_ON, markers=("Reasoning effort",), window=None),
    "gemma4-12b-it": dict(hf="google/gemma-4-12B-it", rev="43c9d014b231474fa7212dbb48a762566c2b255d",
                          kind="instruct", suffix=GEMMA_SUFFIX, off_ending="<|turn>model\n" + GEMMA_SUFFIX,
                          markers=("<|think|>",), window=1024),
    "gemma4-e4b-it": dict(hf="google/gemma-4-E4B-it", rev="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
                          kind="instruct", off_ending="<|turn>model\n", markers=("<|think|>",), window=512),
    "gemma4-31b-it": dict(hf="google/gemma-4-31B-it", rev="842da3794eaa0b77d5f08bae87a17459d91ff475",
                          kind="instruct", suffix=GEMMA_SUFFIX, off_ending="<|turn>model\n" + GEMMA_SUFFIX,
                          markers=("<|think|>",), window=1024),
    "ministral3-8b-it": dict(hf="mistralai/Ministral-3-8B-Instruct-2512-BF16",
                             rev="f6fae9795746f63c9be8344932f01275f3c63734", kind="instruct", switch=False,
                             tok_kwargs={"mistral_format": False}, off_ending="[/INST]",
                             forbidden=("Le Chat", "The current date"), mistral_common=True, window=None),
    "ministral3-14b-it": dict(hf="mistralai/Ministral-3-14B-Instruct-2512-BF16",
                              rev="3cea74c1ebaf5ce5f5a2553de470e2ceab825142", kind="instruct", switch=False,
                              tok_kwargs={"mistral_format": False}, off_ending="[/INST]",
                              forbidden=("Le Chat", "The current date"), mistral_common=True, window=None),
    "olmo3-7b-it": dict(hf="allenai/Olmo-3-7B-Instruct", rev="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
                        kind="instruct", switch=False, off_ending="<|im_start|>assistant\n",
                        forbidden=("function-calling",), window=4096),
    "granite4.2-8b": dict(hf="ibm-granite/granite-4.2-8b", rev="a7042144125ce5ae64f89610e8289647fc000b9a",
                          kind="instruct", off_ending=None, markers=(), window=None),
    # Reasoning-oriented: no thinking-off mode (or, for Qwen3.5, the mode switched on).
    "qwen3.5-9b-think": dict(hf="Qwen/Qwen3.5-9B", rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a", kind="reasoning",
                             template={"enable_thinking": True}, close=("</think>\n\n",)),
    "ministral3-8b-reasoning": dict(hf="mistralai/Ministral-3-8B-Reasoning-2512",
                                    rev="81eaece1948f3875421d9a45bc55487d10e2d894", kind="reasoning",
                                    tok_kwargs={"mistral_format": False}, close=("[/THINK]",)),
    "olmo3-7b-think": dict(hf="allenai/Olmo-3-7B-Think", rev="d97e442d7cc678210054dbcc9b440894d62c89a4",
                           kind="reasoning", close=("</think>", "</think>\n\n")),
    "gpt-oss-20b": dict(hf="openai/gpt-oss-20b", rev="6cee5e81ee83917806bbde320786a8fb61efebee", kind="reasoning",
                        close=("<|end|><|start|>assistant<|channel|>final<|message|>",)),
}

# Pairs of arms that differ only in the goal line (both must be arms of the record).
ARM_PAIRS = (("aligned", "swapped"), ("aligned", "neutral"), ("aligned", "both"), ("yn_aligned", "yn_swapped"),
             ("yn_aligned", "yn_neutral"), ("yn_aligned", "yn_both"), ("pad1_aligned", "pad1_swapped"),
             ("pad8_aligned", "pad8_swapped"), ("ext_aligned", "ext_parent"), ("ext_aligned", "ext_swapped"),
             ("ext_aligned", "ext_both"))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_tok(root: Path, spec: dict, mistral_format: bool | None = None):
    from transformers import AutoTokenizer

    d = root / (spec["hf"].replace("/", "__") + "@" + spec["rev"])
    kw = dict(spec.get("tok_kwargs", {}))
    if mistral_format is not None:
        kw["mistral_format"] = mistral_format
    return AutoTokenizer.from_pretrained(str(d), **kw), d


class Enc:
    """The scorer's encoding (wf.scoring.logodds.LogoddsScorer._encode and _encode_candidate), with the template
    settings round 2 uses (thinking off) unless the spec overrides them."""

    def __init__(self, tok, spec: dict):
        self.tok = tok
        self.kw = {"enable_thinking": False, **spec.get("template", {})}
        self.suffix = spec.get("suffix", "")
        self.special = _special_ids(tok)

    def ids(self, msgs, gen: bool = True, **extra) -> list[int]:
        return _ids(self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen, return_tensors="pt",
                                                 **{**self.kw, **extra}))

    def text(self, msgs, gen: bool = True, **extra) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **{**self.kw, **extra})

    def alone(self, s: str) -> list[int]:
        return _ids(self.tok(s, add_special_tokens=False).input_ids)

    def reply(self, msgs, reply: str) -> list[int]:
        """Wave B's template_reply: the template's ids for `reply` as the assistant's turn after `msgs` (system
        message left out, a declared generation suffix stripped), up to the first special token."""
        convo = [m for m in msgs if m["role"] != "system"]
        prompt = self.ids(convo)
        if self.suffix:
            tail = self.alone(self.suffix)
            if prompt[-len(tail):] != tail:
                raise ValueError(f"generation prompt does not end with the declared suffix {self.suffix!r}")
            prompt = prompt[:-len(tail)]
        full = self.ids(convo + [{"role": "assistant", "content": reply}], gen=False)
        if full[:len(prompt)] != prompt:
            raise ValueError("the conversation with the reply does not begin with the prompt before it")
        rest = full[len(prompt):]
        end = next((i for i, x in enumerate(rest) if x in self.special), len(rest))
        return rest[:end]


def goal_line(rec: dict, arm: str) -> str:
    return R.shown(rec, arm)[2]


def isolated(enc: Enc, a: list[int], b: list[int], line_a: str, line_b: str) -> bool:
    """True when the ids of two prompts that differ only in their goal line differ only inside that line."""
    n = min(len(a), len(b))
    p = 0
    while p < n and a[p] == b[p]:
        p += 1
    s = 0
    while s < n - p and a[-1 - s] == b[-1 - s]:
        s += 1
    mid_a = enc.tok.decode(a[p:len(a) - s]).strip()
    mid_b = enc.tok.decode(b[p:len(b) - s]).strip()
    return mid_a in line_a and mid_b in line_b


def check_instruct(root: Path, tag: str, spec: dict, records: list[dict], shots, limit: int | None) -> dict:
    tok, tdir = load_tok(root, spec)
    enc = Enc(tok, spec)
    problems: list[str] = []
    info: Counter = Counter()
    first_pairs: Counter = Counter()
    n_tokens = {"o_plus": [], "o_minus": []}
    max_input = 0
    over_window = 0
    window = spec.get("window")
    returned = type(tok.apply_chat_template(R.messages(records[0], "aligned", shots), tokenize=True,
                                            add_generation_prompt=True, return_tensors="pt", **enc.kw)).__name__
    scored = {w: enc.alone(w) for w in ("Yes", "No")}
    for w, ids in scored.items():
        if len(ids) != 1:
            problems.append(f"{w!r} is not one token: {ids}")
    if scored["Yes"] == scored["No"]:
        problems.append("'Yes' and 'No' are the same token")
    ending = enc.alone(spec["off_ending"]) if spec.get("off_ending") else []
    qwen_off = spec.get("off_ending") == QWEN_OFF  # the generation prompt itself holds one empty think block
    ref = None
    if spec.get("mistral_common"):
        try:
            mc, _ = load_tok(root, spec, mistral_format=True)
            ref = lambda msgs: _ids(mc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))  # noqa: E731
            info["mistral_common_backend"] = type(mc).__name__ == "MistralCommonBackend"
        except Exception as e:  # noqa: BLE001
            problems.append(f"mistral-common reference could not load: {e!r}")
    # Thinking switch, markers and endings on the first record's arms.
    first = records[0]
    switch = {"checked": 0}
    if spec.get("switch", True):
        for arm in first["arms"]:
            msgs = R.messages(first, arm, shots)
            off, on = enc.text(msgs), enc.text(msgs, enable_thinking=True)
            switch["checked"] += 1
            if on == off:
                problems.append(f"{first['id']} {arm}: thinking on renders like thinking off")
            if spec.get("on_ending") and not on.endswith(spec["on_ending"]):
                problems.append(f"{first['id']} {arm}: thinking on does not end with {spec['on_ending']!r}")
            for marker in spec.get("markers", ()):
                if marker not in on:
                    problems.append(f"{first['id']} {arm}: thinking on lacks {marker!r}")
                if marker in off:
                    problems.append(f"{first['id']} {arm}: thinking off contains {marker!r}")
        switch["off_ending_rendered"] = enc.text(R.messages(first, "aligned", shots))[-60:]
        switch["on_ending_rendered"] = enc.text(R.messages(first, "aligned", shots), enable_thinking=True)[-60:]
    else:
        msgs = R.messages(first, "aligned", shots)
        switch["on_equals_off"] = enc.text(msgs) == enc.text(msgs, enable_thinking=True)
        switch["off_ending_rendered"] = enc.text(msgs)[-60:]
    n_arms = n_cand = n_yn = n_pairs = 0
    t0 = time.time()
    for k, rec in enumerate(records[:limit] if limit else records):
        ids_by_arm: dict[str, list[int]] = {}
        for arm in rec["arms"]:
            question = R.parse_arm(arm)[0]
            msgs = R.messages(rec, arm, shots)
            ids = enc.ids(msgs)
            text = enc.text(msgs)
            ids_by_arm[arm] = ids
            n_arms += 1
            info["template_ids_not_plain_tokenization"] += ids != enc.alone(text)
            if ending and ids[-len(ending):] != ending:
                problems.append(f"{rec['id']} {arm}: prompt does not end with {spec['off_ending']!r}")
            for bad in spec.get("forbidden", ()):
                if bad in text:
                    problems.append(f"{rec['id']} {arm}: prompt contains {bad!r}")
            info["prompts_with_empty_think_in_examples"] += text.count("<think>\n\n</think>") > int(qwen_off)
            if ref is not None:
                info["mistral_common_checked"] += 1
                if ref(msgs) != ids:
                    problems.append(f"{rec['id']} {arm}: template ids differ from mistral-common's")
            if question:
                n_yn += 1
                max_input = max(max_input, len(ids) + 1)
                over_window += bool(window) and len(ids) + 1 > window
                for w in ("Yes", "No"):
                    try:
                        r = enc.reply(msgs, w)
                    except ValueError as e:
                        problems.append(f"{rec['id']} {arm} {w}: {e}")
                        continue
                    if r != scored[w]:
                        problems.append(f"{rec['id']} {arm}: {w!r} alone {scored[w]} vs template {r}")
                continue
            alone = {}
            for name, cand in zip(("o_plus", "o_minus"), R.candidates(rec, arm)):
                a = enc.alone(cand)
                alone[name] = a
                n_cand += 1
                n_tokens[name].append(len(a))
                max_input = max(max_input, len(ids) + len(a))
                over_window += bool(window) and len(ids) + len(a) > window
                try:
                    r = enc.reply(msgs, cand)
                except ValueError as e:
                    problems.append(f"{rec['id']} {arm} {name}: {e}")
                    continue
                if r != a:
                    problems.append(f"{rec['id']} {arm} {name}: alone {a[:6]} vs template {r[:6]}")
                info["joined_plain_text_differs"] += ids + a != enc.alone(text + cand)
            if not alone["o_plus"] or not alone["o_minus"] or alone["o_plus"][0] == alone["o_minus"][0]:
                problems.append(f"{rec['id']} {arm}: candidates share their first token")
            else:
                first_pairs[(tok.decode(alone["o_plus"][:1]), tok.decode(alone["o_minus"][:1]))] += 1
        for a, b in ARM_PAIRS:
            if a in ids_by_arm and b in ids_by_arm:
                n_pairs += 1
                if not isolated(enc, ids_by_arm[a], ids_by_arm[b], goal_line(rec, a), goal_line(rec, b)):
                    info["arm_pairs_not_isolated"] += 1
                    if info["arm_pairs_not_isolated"] <= 5:
                        info[f"example_not_isolated:{rec['id']}:{a}:{b}"] += 1
        if k % 500 == 0:
            print(f"{tag}: {k} records, {n_arms} arms, {len(problems)} problems, {time.time() - t0:.0f}s", flush=True)
    by_id = {r["id"]: r for r in records}
    golden = {}
    for rid, arm in golden_cases(records):
        msgs = R.messages(by_id[rid], arm, shots)
        ids = enc.ids(msgs)
        golden[f"{rid}|{arm}"] = {"n_ids": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                  **({"text": enc.text(msgs)} if arm in ("aligned", "yn_both") else {})}
    return {"tag": tag, "kind": "instruct", "hf_id": spec["hf"], "revision": spec["rev"], "tokenizer_dir": tdir.name,
            "tokenizer_class": type(tok).__name__, "apply_chat_template_tokenize_returns": returned,
            "template_kwargs": enc.kw, "generation_suffix": enc.suffix, "thinking_off_ending": spec.get("off_ending"),
            "ok": not problems and n_cand > 0, "n_problems": len(problems), "problems": problems[:50],
            "n_records": len(records[:limit] if limit else records), "n_arms": n_arms,
            "n_candidates_checked": n_cand, "n_yes_no_prompts_checked": n_yn, "n_arm_pairs_checked": n_pairs,
            "yes_no_ids": {w: v for w, v in scored.items()}, "yes_no_decoded": {w: tok.decode(v) for w, v in scored.items()},
            "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
            "candidate_tokens": {k: {"mean": round(mean(v), 3), "median": median(v), "min": min(v), "max": max(v)}
                                 for k, v in n_tokens.items() if v},
            "max_input_tokens": max_input, "sliding_window": window, "inputs_over_sliding_window": over_window,
            "switch": switch, "info": dict(info), "golden": golden, "seconds": round(time.time() - t0, 1)}


def check_reasoning(root: Path, tag: str, spec: dict, records: list[dict], shots) -> dict:
    tok, tdir = load_tok(root, spec)
    enc = Enc(tok, spec)
    problems: list[str] = []
    by_id = {r["id"]: r for r in records}
    renders = {}
    for rid, arm in golden_cases(records):
        msgs = R.messages(by_id[rid], arm, shots)
        try:
            text = enc.text(msgs)
            off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            renders[f"{rid}|{arm}"] = {"ending": text[-80:], "n_ids": len(enc.ids(msgs)),
                                       "thinking_off_renders_differently": off != text,
                                       "system_text_verbatim": R.SYSTEM in text}
        except Exception as e:  # noqa: BLE001
            problems.append(f"{rid} {arm}: render failed: {e!r}")
    cands = sorted({c for r in records for arm in r["arms"] if not R.parse_arm(arm)[0] for c in R.candidates(r, arm)})
    boundary = {}
    for close in spec["close"]:
        merged = [c for c in cands + ["Yes", "No"] if enc.alone(close + c) != enc.alone(close) + enc.alone(c)]
        boundary[close] = {"n_strings": len(cands) + 2, "n_merge_across_close": len(merged), "examples": merged[:5]}
    scored = {w: enc.alone(w) for w in ("Yes", "No")}
    return {"tag": tag, "kind": "reasoning", "hf_id": spec["hf"], "revision": spec["rev"], "tokenizer_dir": tdir.name,
            "tokenizer_class": type(tok).__name__, "template_kwargs": enc.kw, "renders": renders,
            "boundary_after_close": boundary, "yes_no_ids": scored,
            "yes_no_single": all(len(v) == 1 for v in scored.values()), "n_problems": len(problems),
            "problems": problems[:20]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("tag", choices=sorted(SPECS))
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="first N records only (smoke run)")
    a = ap.parse_args(argv)
    if sha256_file(R2_ITEMS) != R2_ITEMS_SHA256:
        raise SystemExit("r2_items.jsonl is not the frozen round-2 record file")
    records, shots = read_records(), P.fewshot_examples()
    spec = SPECS[a.tag]
    t0 = time.time()
    if spec["kind"] == "instruct":
        rep = check_instruct(a.root, a.tag, spec, records, shots, a.limit)
    else:
        rep = check_reasoning(a.root, a.tag, spec, records, shots)
    import transformers
    import tokenizers

    rep.update({"r2_items_sha256": R2_ITEMS_SHA256, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "python": sys.version.split()[0],
                "wall_seconds": round(time.time() - t0, 1), "limit": a.limit})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep.get(k) for k in ("tag", "ok", "n_problems", "n_arms", "max_input_tokens",
                                              "wall_seconds")}))
    return 0 if rep.get("n_problems", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
