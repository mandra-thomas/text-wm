"""TextCraft round 3: the tokenization and template gate, and the golden renderings (notes/textcraft_r3_design.md,
section 4). Needs only a tokenizer. The scoring runs call `gate` with their own scorer before any score and on every
resume; on ICE it runs on a CPU node for every model and every record before the freeze:

    python -m wf.textcraft.r3_tokcheck check <tag> <out.json>     # pinned tokenizer from the HF cache, offline
    python -m wf.textcraft.r3_tokcheck golden <report.json>...    # writes data/r3_chat_golden.json from the reports

Round 2's gate (wf/textcraft/r2_tokcheck.py), with wave B's additions (wf/families.py), on every record and arm:
each candidate tokenized alone equals the template's reply after that arm's prompt (the model's declared generation
suffix stripped); "Yes" and "No" are single tokens equal to the template's reply after every yes/no prompt; the two
candidates' first tokens differ; every prompt ends with the declared thinking-off ending; no forbidden text (and,
for a no-preserve variant, nothing forbidden before the ending); the thinking switch is read where the template has
one; Ministral 3's prompt ids equal mistral-common's; and arm isolation: two arms that differ only in the goal line
differ in token ids only inside that line.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from wf import families as FAM
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r3 as R3
from wf.textcraft.r2_tokcheck import golden_cases

GOLDEN = Path(__file__).resolve().parent / "data" / "r3_chat_golden.json"
# Pairs of arms that differ only in the goal line (both must be arms of the record).
ARM_PAIRS = (("aligned", "swapped"), ("aligned", "neutral"), ("aligned", "both"), ("yn_aligned", "yn_swapped"),
             ("yn_aligned", "yn_neutral"), ("yn_aligned", "yn_both"), ("pad1_aligned", "pad1_swapped"),
             ("pad8_aligned", "pad8_swapped"), ("ext_aligned", "ext_parent"), ("ext_aligned", "ext_swapped"),
             ("ext_aligned", "ext_both"))


def render(scorer, messages: list[dict], **overrides: Any) -> str:
    return scorer._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **{**scorer.template_settings(), **overrides})


def render_one(scorer, spec, rec: dict, arm: str, shots) -> dict[str, Any]:
    m = R3.model_messages(spec, rec, arm, shots)
    ids = FAM._ids(scorer._encode(m))
    return {"text": render(scorer, m), "n_ids": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def isolated(tok, a: list[int], b: list[int], line_a: str, line_b: str) -> bool:
    """True when the ids of two prompts that differ only in their goal line differ only inside that line."""
    n = min(len(a), len(b))
    p = 0
    while p < n and a[p] == b[p]:
        p += 1
    s = 0
    while s < n - p and a[-1 - s] == b[-1 - s]:
        s += 1
    mid_a = tok.decode(a[p:len(a) - s]).strip()
    mid_b = tok.decode(b[p:len(b) - s]).strip()
    return mid_a in line_a and mid_b in line_b


def golden_problems(scorer, tag: str, records: list[dict], shots, golden: dict | None = None) -> list[str]:
    """Where this scorer's rendering of the golden cases differs from the stored one."""
    if golden is None:
        if not GOLDEN.exists():
            return [f"no golden file at {GOLDEN}"]
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if tag not in golden.get("models", {}):
        return [f"no golden rendering stored for {tag}"]
    by_id = {r["id"]: r for r in records}
    out = []
    for case in golden["cases"]:
        rec = by_id.get(case["id"])
        if rec is None:
            out.append(f"golden record {case['id']} is not in the records file")
            continue
        now = render_one(scorer, R3.spec_for(tag), rec, case["arm"], shots)
        if now != golden["models"][tag][f"{case['id']}|{case['arm']}"]:
            out.append(f"{tag} {case['id']} {case['arm']}: rendering differs from the golden")
    return out


def gate(scorer, spec, records: list[dict], shots, reference_encoder: Callable[[list[dict]], list[int]] | None = None,
         max_problems: int = 50) -> dict[str, Any]:
    """The round-3 gate on every record and arm; "ok" says whether scoring may start."""
    tok = scorer._tok
    suffix = spec.generation_suffix
    problems: list[str] = []
    first_pairs: Counter = Counter()
    n_cand = n_yn = n_ending = n_reference = n_pairs = not_isolated = 0
    max_input = 0
    scored = {w: FAM._ids(scorer._encode_candidate(w)) for w in FAM.YES_NO}
    for w in FAM.YES_NO:
        if len(scored[w]) != 1:
            problems.append(f"{w!r} is not one token: {scored[w]}")
    if scored["Yes"] == scored["No"]:
        problems.append("'Yes' and 'No' are the same token")
    ending = FAM._ids(tok(spec.thinking_off_ending, add_special_tokens=False)) if spec.thinking_off_ending else []
    if spec.mistral_common_check and reference_encoder is None:
        problems.append("mistral-common agreement required but no reference encoder given")
    # the thinking switch, markers and endings, on the first record's arms (as text)
    first = records[0] if records else None
    n_switch = 0
    if first is not None and spec.thinking_switch:
        for arm in first["arms"]:
            msgs = R3.model_messages(spec, first, arm, shots)
            off, on = render(scorer, msgs), render(scorer, msgs, enable_thinking=True)
            n_switch += 1
            if on == off:
                problems.append(f"{first['id']} {arm}: thinking on renders like thinking off (switch not read)")
            if spec.thinking_on_ending and not on.endswith(spec.thinking_on_ending):
                problems.append(f"{first['id']} {arm}: thinking on does not end with {spec.thinking_on_ending!r}")
            for marker in spec.thinking_on_markers:
                if marker not in on:
                    problems.append(f"{first['id']} {arm}: thinking on lacks {marker!r}")
                if marker in off:
                    problems.append(f"{first['id']} {arm}: thinking off contains {marker!r}")
    for rec in records:
        ids_by_arm: dict[str, list[int]] = {}
        for arm in rec["arms"]:
            msgs = R3.model_messages(spec, rec, arm, shots)
            prompt = FAM._ids(scorer._encode(msgs))
            ids_by_arm[arm] = prompt
            text = render(scorer, msgs)
            for bad in spec.forbidden_in_prompt:
                if bad in text:
                    problems.append(f"{rec['id']} {arm}: the prompt contains {bad!r}")
            if spec.forbidden_before_ending:
                body = text[:len(text) - len(spec.thinking_off_ending)] if text.endswith(spec.thinking_off_ending) else text
                for bad in spec.forbidden_before_ending:
                    if bad in body:
                        problems.append(f"{rec['id']} {arm}: {bad!r} before the thinking-off ending")
            if ending:
                n_ending += 1
                if prompt[-len(ending):] != ending:
                    problems.append(f"{rec['id']} {arm}: the prompt does not end with {spec.thinking_off_ending!r}")
            if spec.mistral_common_check and reference_encoder is not None:
                n_reference += 1
                if list(reference_encoder(msgs)) != prompt:
                    problems.append(f"{rec['id']} {arm}: the template's ids differ from mistral-common's")
            if R.parse_arm(arm)[0]:
                n_yn += 1
                max_input = max(max_input, len(prompt) + 1)
                for w in FAM.YES_NO:
                    try:
                        reply = FAM.template_reply(scorer, msgs, w, suffix)
                    except FAM.TokenizationMismatch as e:
                        problems.append(f"{rec['id']} {arm} {w!r}: {e}")
                        continue
                    if reply != scored[w]:
                        problems.append(f"{rec['id']} {arm}: {w!r} alone {scored[w]} vs template {reply}")
                continue
            alone = {}
            for name, cand in zip(("o_plus", "o_minus"), R.candidates(rec, arm)):
                try:
                    r = FAM.check_candidate(scorer, msgs, cand, suffix)
                except FAM.TokenizationMismatch as e:
                    problems.append(f"{rec['id']} {arm} {name}: {e}")
                    alone[name] = []
                    continue
                n_cand += 1
                max_input = max(max_input, r["prompt_tokens"] + len(r["alone"]))
                if not r["alone_equals_template_reply"]:
                    problems.append(f"{rec['id']} {arm} {name}: alone {r['alone'][:6]} vs template {r['template_reply'][:6]}")
                alone[name] = r["alone"]
            if not alone["o_plus"] or not alone["o_minus"] or alone["o_plus"][0] == alone["o_minus"][0]:
                problems.append(f"{rec['id']} {arm}: the candidates share their first token")
            else:
                first_pairs[(tok.decode(alone["o_plus"][:1]), tok.decode(alone["o_minus"][:1]))] += 1
        for a, b in ARM_PAIRS:
            if a in ids_by_arm and b in ids_by_arm:
                n_pairs += 1
                if not isolated(tok, ids_by_arm[a], ids_by_arm[b], R.shown(rec, a)[2], R.shown(rec, b)[2]):
                    not_isolated += 1
                    problems.append(f"{rec['id']} {a} vs {b}: token ids differ outside the goal line")
    if spec.thinking_off_ending and not n_ending:
        problems.append("no prompt checked for the thinking-off ending")
    if not n_yn:
        problems.append("no yes/no prompt checked")
    return {"ok": not problems and n_cand > 0, "n_records": len(records), "n_candidates_checked": n_cand,
            "n_yes_no_prompts_checked": n_yn, "n_prompts_checked_ending": n_ending,
            "n_prompts_checked_mistral_common": n_reference, "n_arm_pairs_checked": n_pairs,
            "n_arm_pairs_not_isolated": not_isolated, "n_switch_renders": n_switch, "n_problems": len(problems),
            "problems": problems[:max_problems], "yes_no_ids": scored,
            "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
            "max_input_tokens": max_input, "template_kwargs": scorer.template_settings(),
            "generation_suffix": suffix, "thinking_off_ending": spec.thinking_off_ending, "system_handling": spec.system}


def scorer_for_tokenizer(spec, tok):
    from wf.scoring.logodds import LogoddsScorer

    return LogoddsScorer.for_tokenizer(spec.hf_id, tok, model_tag=spec.tag, template_kwargs=spec.template_kwargs,
                                       tokenizer_kwargs=spec.tokenizer_kwargs, prompt_format=spec.prompt_format)


def load_tokenizer(spec):
    from transformers import AutoTokenizer

    from wf.constants import revision_for

    return AutoTokenizer.from_pretrained(spec.hf_id, revision=revision_for(spec.hf_id), trust_remote_code=True,
                                         **spec.tokenizer_kwargs)


def reference_encoder(spec):
    return FAM.mistral_common_encoder(spec) if spec.mistral_common_check else None


def check(tag: str, out: Path, records: list[dict] | None = None, shots=None) -> dict:
    from wf.textcraft.r2_items import read_records

    spec = R3.spec_for(tag)
    records = read_records() if records is None else records
    shots = P.fewshot_examples() if shots is None else shots
    scorer = scorer_for_tokenizer(spec, load_tokenizer(spec))
    t0 = time.time()
    report = gate(scorer, spec, records, shots, reference_encoder(spec))
    by_id = {r["id"]: r for r in records}
    report["golden"] = {f"{i}|{a}": render_one(scorer, spec, by_id[i], a, shots) for i, a in golden_cases(records)}
    report.update({"tag": tag, "hf_id": spec.hf_id, "revision": spec.revision, "seconds": round(time.time() - t0, 1),
                   "transformers": __import__("transformers").__version__, "spec": spec.record()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def make_golden(reports: list[dict], records: list[dict]) -> dict:
    cases = [{"id": i, "arm": a} for i, a in golden_cases(records)]
    return {"cases": cases, "models": {r["tag"]: {"revision": r["revision"], **r["golden"]} for r in reports}}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["check"]:
        report = check(argv[1], Path(argv[2]))
        print(json.dumps({k: report[k] for k in ("tag", "ok", "n_problems", "n_candidates_checked",
                                                  "n_yes_no_prompts_checked", "n_arm_pairs_checked",
                                                  "max_input_tokens", "seconds")}))
        return 0 if report["ok"] else 1
    if argv[:1] == ["golden"]:
        from wf.textcraft.r2_items import read_records

        reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[1:]]
        GOLDEN.write_text(json.dumps(make_golden(reports, read_records()), indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
