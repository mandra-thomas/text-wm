"""TextCraft round 2: the tokenization gate and the golden rendered prompts (notes/textcraft_r2_design.md, sections 2
and 3). Needs only a tokenizer: the scoring run calls `gate` with its own scorer before any score, and on ICE it runs
on a CPU node for every model and every record:

    python -m wf.textcraft.r2_tokcheck check <tag> <out.json>      # pinned tokenizer from the HF cache (offline)
    python -m wf.textcraft.r2_tokcheck golden <report.json>...     # writes data/r2_chat_golden.json from the reports

Everything goes through the scorer's own encoding (`LogoddsScorer._encode`, `_encode_candidate` and its template
settings), with wave A's checks (wf/families.py): each candidate, tokenized alone, equals the chat template's tokens
for it as the assistant's reply after that arm's prompt; "Yes" and "No" are single tokens equal to the template's
reply after every yes/no prompt; the two candidates' first tokens differ. Recorded for information: whether the
template's ids are the plain tokenization of its rendered text, and whether the rendered prompt and the candidate
joined as plain text tokenize as the prompt's ids followed by the candidate's (Mistral fails this by design).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from wf import families as FAM
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R

GOLDEN = Path(__file__).resolve().parent / "data" / "r2_chat_golden.json"
GEMMA_WINDOW = 4096


def golden_cases(records: list[dict]) -> list[tuple[str, str]]:
    """(record id, arm) of the golden renderings: the first main record with padding arms under four arms, and the
    first parent record under the parent line."""
    main = next(r for r in records if r["kind"] == "main" and "pad8_aligned" in r["arms"])
    parent = next(r for r in records if r["kind"] == "parent")
    return [(main["id"], a) for a in ("aligned", "both", "yn_both", "pad8_aligned")] + [(parent["id"], "ext_parent")]


def render(scorer, messages: list[dict]) -> str:
    return scorer._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **scorer.template_settings())


def render_one(scorer, spec, rec: dict, arm: str, shots) -> dict[str, Any]:
    m = R.model_messages(spec, rec, arm, shots)
    ids = FAM._ids(scorer._encode(m))
    return {"text": render(scorer, m), "n_ids": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


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
        now = render_one(scorer, R.MODELS[tag], rec, case["arm"], shots)
        if now != golden["models"][tag][f"{case['id']}|{case['arm']}"]:
            out.append(f"{tag} {case['id']} {case['arm']}: rendering differs from the golden")
    return out


def gate(scorer, spec, records: list[dict], shots, max_problems: int = 50) -> dict[str, Any]:
    """Wave A's tokenization gate on every record and arm; "ok" says whether scoring may start."""
    tok = scorer._tok
    problems: list[str] = []
    first_pairs: Counter = Counter()
    n_cand = n_yn = plain_mismatch = joined_mismatch = 0
    max_prompt, over_window = 0, 0
    scored = {w: FAM._ids(scorer._encode_candidate(w)) for w in FAM.YES_NO}
    yes_no = {w: {"alone": scored[w], "decoded": tok.decode(scored[w]), "n_checked": 0, "n_mismatched": 0}
              for w in FAM.YES_NO}
    for w in FAM.YES_NO:
        if len(scored[w]) != 1:
            problems.append(f"{w!r} is not one token: {scored[w]}")
    if scored["Yes"] == scored["No"]:
        problems.append("'Yes' and 'No' are the same token")
    for rec in records:
        for arm in rec["arms"]:
            msgs = R.model_messages(spec, rec, arm, shots)
            prompt = FAM._ids(scorer._encode(msgs))
            text = render(scorer, msgs)
            plain_mismatch += FAM._ids(tok(text, add_special_tokens=False)) != prompt
            if R.parse_arm(arm)[0]:
                n_yn += 1
                max_prompt = max(max_prompt, len(prompt) + 1)
                over_window += len(prompt) + 1 > GEMMA_WINDOW
                for w in FAM.YES_NO:
                    try:
                        reply = FAM.template_reply(scorer, msgs, w)
                    except FAM.TokenizationMismatch as e:
                        problems.append(f"{rec['id']} {arm} {w!r}: {e}")
                        continue
                    yes_no[w]["n_checked"] += 1
                    if reply != scored[w]:
                        yes_no[w]["n_mismatched"] += 1
                        problems.append(f"{rec['id']} {arm}: {w!r} alone {scored[w]} vs template {reply}")
                continue
            ids = {}
            for name, cand in zip(("o_plus", "o_minus"), R.candidates(rec, arm)):
                try:
                    r = FAM.check_candidate(scorer, msgs, cand)
                except FAM.TokenizationMismatch as e:
                    problems.append(f"{rec['id']} {arm} {name}: {e}")
                    ids[name] = []
                    continue
                n_cand += 1
                joined_mismatch += not r["joined_plain_text_equal"]
                max_prompt = max(max_prompt, r["prompt_tokens"] + len(r["alone"]))
                over_window += r["prompt_tokens"] + len(r["alone"]) > GEMMA_WINDOW
                if not r["alone_equals_template_reply"]:
                    problems.append(f"{rec['id']} {arm} {name}: alone {r['alone'][:6]} vs template {r['template_reply'][:6]}")
                ids[name] = r["alone"]
            if not ids["o_plus"] or not ids["o_minus"] or ids["o_plus"][0] == ids["o_minus"][0]:
                problems.append(f"{rec['id']} {arm}: the candidates share their first token")
            else:
                first_pairs[(tok.decode(ids["o_plus"][:1]), tok.decode(ids["o_minus"][:1]))] += 1
    if n_yn and any(yes_no[w]["n_checked"] == 0 for w in FAM.YES_NO):
        problems.append("no yes/no prompt checked")
    return {"ok": not problems and n_cand > 0, "n_records": len(records), "n_candidates_checked": n_cand,
            "n_yes_no_prompts_checked": n_yn, "n_problems": len(problems), "problems": problems[:max_problems],
            "yes_no": yes_no, "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
            "info_template_ids_not_plain_tokenization": plain_mismatch,
            "info_joined_plain_text_mismatches": joined_mismatch, "max_input_tokens": max_prompt,
            "inputs_over_4096_tokens": over_window, "template_kwargs": scorer.template_settings(),
            "system_handling": spec.system}


def load_tokenizer(hf_id: str):
    from transformers import AutoTokenizer

    from wf.constants import revision_for

    return AutoTokenizer.from_pretrained(hf_id, revision=revision_for(hf_id), trust_remote_code=True)


def check(tag: str, out: Path, records: list[dict] | None = None, shots=None) -> dict:
    from wf.scoring.logodds import LogoddsScorer
    from wf.textcraft.r2_items import read_records

    spec = R.MODELS[tag]
    records = read_records() if records is None else records
    shots = P.fewshot_examples() if shots is None else shots
    scorer = LogoddsScorer.for_tokenizer(spec.hf_id, load_tokenizer(spec.hf_id), model_tag=tag,
                                         template_kwargs=spec.template_kwargs)
    t0 = time.time()
    report = gate(scorer, spec, records, shots)
    by_id = {r["id"]: r for r in records}
    report["golden"] = {f"{i}|{a}": render_one(scorer, spec, by_id[i], a, shots) for i, a in golden_cases(records)}
    report.update({"tag": tag, "hf_id": spec.hf_id, "revision": spec.revision, "seconds": round(time.time() - t0, 1),
                   "transformers": __import__("transformers").__version__})
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
                                                  "n_yes_no_prompts_checked", "max_input_tokens", "seconds")}))
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
