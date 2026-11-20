"""Model families, wave A (decided 2026-11-20, before any score; notes/families_design.md).

The size series' scoring (wf/sizes.py: RQ2's prompt, example set 1, history only; the aligned, swapped and neutral
tasks, each as the two-candidate score and as the yes/no question; the determined successes for AUC) applied to
models from other families and generations. One scoring run per model (`family_score_<tag>`), with three additions
the Qwen3 series did not need:

1. Chat templates. Gemma 2's template refuses a system message, so the system text is folded into the first user
   turn (`fold_system`), the same way in every condition. Llama 3.2's template writes today's date into the system
   header unless given one, so it gets the fixed date its own template falls back to (and Llama 3.1's template
   uses): the prompt cannot change with the day a shard runs.
2. A tokenization gate (audit C9). Before any score, every candidate text in the shard must tokenize alone exactly
   as the chat template tokenizes it as the assistant's reply; "Yes" and "No" must each be one token, the one the
   template gives; and the two candidates' first tokens must differ. The results go to tokenization_checks.json in
   the shard's folder, and a failure stops the job before it scores anything.
3. Attention. Gemma 2 is loaded with eager attention: transformers 4.51.3 defaults it to SDPA, which silently drops
   Gemma 2's attention-logit soft-capping, so the logits would not be the trained model's. The scorer refuses to
   continue if the loaded model reports another implementation.

`families_analysis` reads every family run and, for comparison, Qwen3-8B (size_score_qwen3-8b) and Llama-3.1-8B
(rq1_score, rq1x_score and rq1v_score, which hold the same prompts), and gives each the size series' block plus a
spread-standardised effect, both flip definitions by name, and first-token coverage.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf import rq1x, sizes
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, paired_conditions
from wf.rq2 import SOURCES, complete_shards, load_items
from wf.runs import _init_run
from wf.scoring.logodds import LogoddsScorer
from wf.util import read_jsonl, require_filed, runs_root, write_json, write_status


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    hf_id: str
    family: str
    generation: str
    release: str  # public release (announcement) date
    params_b: float  # parameters in billions, from the checkpoint's safetensors metadata on the Hub
    system: str  # "native": the template takes the system message; "fold": folded into the first user turn
    attn_implementation: str | None = None  # None: transformers' default
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    template_notes: str = ""

    @property
    def revision(self) -> str | None:
        return C.REVISIONS.get(self.hf_id)

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "revision": self.revision}


# Wave A: transformers 4.51.3 loads all four, so they run in the study's environment.
MODELS: dict[str, ModelSpec] = {
    "gemma2-9b-it": ModelSpec(
        "gemma2-9b-it", "google/gemma-2-9b-it", "gemma", "Gemma 2", "2024-06-27", 9.24, "fold", "eager", {},
        "No system role: the template raises on one, so WM_SYSTEM opens the first user turn, followed by a blank "
        "line. Roles must alternate user/model; contents are trimmed; '<bos>' comes from the template. Eager "
        "attention for the attention-logit soft-capping (50); the final logits are tanh-capped at 30."),
    "mistral-7b-v0.3": ModelSpec(
        "mistral-7b-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "mistral", "Mistral 7B v0.3", "2024-05-22", 7.25,
        "native", None, {},
        "The template takes a system message but places it in the last user turn ('[INST] system\\n\\nquery"
        "[/INST]'), not the first. Roles must alternate. A reply follows '[/INST]' after a space, so a candidate is "
        "scored as the template writes a reply (' You ...'). No generation prompt is added."),
    "llama3.2-1b": ModelSpec(
        "llama3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "llama", "Llama 3.2", "2024-09-25", 1.24, "native", None,
        {"date_string": "26 Jul 2024"},
        "Llama 3 header format, as Llama 3.1. Without date_string the template writes today's date (strftime_now) "
        "into the system header, so the date is fixed at the template's own fallback, which Llama 3.1's template "
        "always uses. Pruned and distilled from Llama 3.1 8B and 70B."),
    "llama3.2-3b": ModelSpec(
        "llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "llama", "Llama 3.2", "2024-09-25", 3.21, "native", None,
        {"date_string": "26 Jul 2024"},
        "As Llama 3.2 1B (the same tokenizer files)."),
}

# Already scored, read for comparison; their prompts are the ones above (the size series and RQ1/RQ2 runs).
REFERENCE: dict[str, ModelSpec] = {
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen", "Qwen3", "2025-04-29", 8.19, "native", None,
                          {"enable_thinking": False}, "Thinking off; the template writes an empty think block."),
    "llama3.1-8b": ModelSpec("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama", "Llama 3.1", "2024-07-23",
                             8.03, "native", None, {}, "Fixed date 26 Jul 2024 in the system header."),
}

FOLD_SEPARATOR = "\n\n"
PREDICTION = ("Logged before any family score (2026-11-20): the goal effect (aligned minus swapped) is positive under "
              "both formats, two-candidate and yes/no, for every wave A model; read as the 95% game-clustered CI "
              "lying above zero.")
ARMS = sizes.PRED_ARMS + sizes.YN_ARMS


# ---------------- prompts ----------------


def fold_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The system message's text, a blank line, then the first user turn's text, as one user turn.

    For a template with no system role (Gemma 2). Deterministic and the same in every condition: only the first
    turn changes, and it holds the first prompt example, never the task line being varied.
    """
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError("expected a system message followed by a user turn")
    first = {"role": "user", "content": messages[0]["content"] + FOLD_SEPARATOR + messages[1]["content"]}
    return [first] + [dict(m) for m in messages[2:]]


def family_messages(spec: ModelSpec, t: dict, arm: str) -> list[dict[str, str]] | None:
    msgs = sizes.arm_messages(t, arm)
    if msgs is None:
        return None
    return fold_system(msgs) if spec.system == "fold" else msgs


# ---------------- tokenization gate (audit C9) ----------------


class TokenizationMismatch(RuntimeError):
    pass


def _ids(x: Any) -> list[int]:
    """Token ids from apply_chat_template or a tokenizer call, whatever container they come in."""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)


def _special_ids(tok: Any) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", []) or [])
    for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items():
        if getattr(t, "special", True):
            ids.add(int(i))
    return ids


def template_reply(tok: Any, messages: list[dict], reply: str, kwargs: dict) -> list[int]:
    """The ids the chat template gives `reply` as the assistant's turn after `messages`, up to its end-of-turn token.

    The conversation with the reply must begin with exactly the prompt the template gives before a reply
    (generation prompt included); otherwise a candidate scored after that prompt is not the reply as the template
    writes it. A system message is left out of this check: Mistral's template writes it into the last user turn,
    which is no longer last once a reply follows, so the two renders would differ for that reason alone. Where the
    prompt ends and the reply begins does not depend on the system message.
    """
    convo = [m for m in messages if m["role"] != "system"]
    prompt = _ids(tok.apply_chat_template(convo, tokenize=True, add_generation_prompt=True, **kwargs))
    full = _ids(tok.apply_chat_template(convo + [{"role": "assistant", "content": reply}], tokenize=True,
                                        add_generation_prompt=False, **kwargs))
    if full[:len(prompt)] != prompt:
        raise TokenizationMismatch("the conversation with the reply does not begin with the prompt before it")
    tail = full[len(prompt):]
    special = _special_ids(tok)
    end = next((i for i, x in enumerate(tail) if x in special), len(tail))
    return tail[:end]


def check_candidate(tok: Any, messages: list[dict], text: str, kwargs: dict) -> dict[str, Any]:
    """Does `text`, tokenized alone (as the scorer does), equal the template's tokens for it as the reply?

    Also the verify_models check: the prompt's text and the candidate joined as plain text, tokenized together.
    That one is informative only: a template that puts a space before a reply (Mistral) fails it by design.
    """
    reply = template_reply(tok, messages, text, kwargs)
    alone = _ids(tok(text, add_special_tokens=False))
    prompt = _ids(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs))
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    joined = _ids(tok(prompt_text + text, add_special_tokens=False))
    return {"alone_equals_template_reply": alone == reply, "joined_plain_text_equal": prompt + alone == joined,
            "alone": alone, "template_reply": reply, "prompt_tokens": len(prompt)}


def tokenization_checks(tok: Any, spec: ModelSpec, primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """The gate, on every item of a shard. Returns the record; its "ok" says whether scoring may start."""
    kw = {"enable_thinking": False, **spec.template_kwargs}
    problems: list[str] = []
    first_pairs: Counter = Counter()
    joined_mismatch = 0
    max_prompt = 0
    n = 0
    for t in primary + successes:
        msgs = family_messages(spec, t, "aligned")
        ids = {}
        for name in ("o_plus", "o_minus"):
            try:
                r = check_candidate(tok, msgs, t["candidates"][name], kw)
            except TokenizationMismatch as e:  # recorded with the rest, so the record says what failed
                problems.append(f"{t['id']} {name}: {e}")
                ids[name] = []
                continue
            n += 1
            max_prompt = max(max_prompt, r["prompt_tokens"])
            joined_mismatch += not r["joined_plain_text_equal"]
            if not r["alone_equals_template_reply"]:
                problems.append(f"{t['id']} {name}: alone {r['alone'][:6]} vs template {r['template_reply'][:6]}")
            ids[name] = r["alone"]
        if not ids["o_plus"] or not ids["o_minus"] or ids["o_plus"][0] == ids["o_minus"][0]:
            problems.append(f"{t['id']}: the candidates share their first token")
        else:
            first_pairs[(tok.decode(ids["o_plus"][:1]), tok.decode(ids["o_minus"][:1]))] += 1
    yes_no: dict[str, Any] = {}
    sample = (primary + successes)[:1]
    for t in sample:
        msgs = family_messages(spec, t, "yn_aligned")
        for word in ("Yes", "No"):
            try:
                reply = template_reply(tok, msgs, word, kw)
            except TokenizationMismatch as e:
                problems.append(f"{word!r}: {e}")
                reply = []
            alone = _ids(tok(word, add_special_tokens=False))
            yes_no[word] = {"alone": alone, "template_reply": reply, "decoded": tok.decode(alone)}
            if len(alone) != 1 or alone != reply:
                problems.append(f"{word!r} is not the template's single token: alone {alone}, template {reply}")
    if yes_no and yes_no["Yes"]["alone"] == yes_no["No"]["alone"]:
        problems.append("'Yes' and 'No' are the same token")
    return {"ok": not problems and n > 0, "n_candidates_checked": n, "problems": problems[:50],
            "n_problems": len(problems), "yes_no": yes_no,
            "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
            "joined_plain_text_mismatches": joined_mismatch, "max_prompt_tokens": max_prompt,
            "template_kwargs": kw, "system_handling": spec.system}


def _gate(spec: ModelSpec) -> Callable[[Path, LogoddsScorer, list[dict], list[dict]], None]:
    def gate(d: Path, scorer: LogoddsScorer, primary: list[dict], successes: list[dict]) -> None:
        if scorer.mock:
            write_json(d / "tokenization_checks.json", {"ok": None, "skipped": "mock"})
            return
        rec = tokenization_checks(scorer._tok, spec, primary, successes)
        rec["attn_implementation_loaded"] = getattr(scorer._model.config, "_attn_implementation", None)
        rec["dtype_loaded"] = str(getattr(scorer._model, "dtype", ""))
        write_json(d / "tokenization_checks.json", rec)
        if not rec["ok"]:
            raise TokenizationMismatch(f"{spec.tag}: tokenization gate failed ({rec['n_problems']} problems, first: "
                                       f"{rec['problems'][:3]}); see {d / 'tokenization_checks.json'}. Nothing scored.")
    return gate


# ---------------- scoring ----------------


def run_family_score(cfg: dict) -> int:
    tag = cfg["model"]
    if tag not in MODELS:
        raise SystemExit(f"unknown model {tag!r}; known: {sorted(MODELS)}")
    spec = MODELS[tag]
    cfg = {**cfg, "model_spec": spec.record()}

    def open_scorer() -> LogoddsScorer:
        return LogoddsScorer(spec.hf_id, model_tag=tag, attn_implementation=spec.attn_implementation,
                             template_kwargs=spec.template_kwargs)

    return sizes.score_shard(cfg, tag, open_scorer, lambda t, arm: family_messages(spec, t, arm), gate=_gate(spec))


# ---------------- analysis ----------------


def standardised_effect(s: list[dict], items: list[dict], a: str, b: str) -> dict[str, Any]:
    """Mean (L under `a` minus L under `b`) over the SD of L under `b`, on the same items; game-clustered CI.

    Divides the goal effect by how widely the model's scores spread across items under the swapped task, so a
    model that is more extreme in log-odds everywhere does not look more goal-sensitive for that alone.
    """
    pairs = paired_conditions(s, items, a, b)

    def stat(rows: list[tuple[float, float]]) -> float:
        if len(rows) < 2:
            return float("nan")
        sd = statistics.stdev(lb for _, lb in rows)
        return sum(la - lb for la, lb in rows) / len(rows) / sd if sd > 0 else float("nan")

    rows = [(la, lb) for _, la, lb in pairs]
    boot = cluster_bootstrap(stat, rows, [g for g, _, _ in pairs], seed=C.RNG_SEED)
    return {"estimate": stat(rows), "ci": [boot["ci_low"], boot["ci_high"]], "n_items": len(rows),
            "sd_L_control": statistics.stdev(lb for _, lb in rows) if len(rows) > 1 else float("nan")}


def first_token_coverage(s: list[dict]) -> dict[str, Any]:
    """Per condition: how much next-token mass the two first tokens hold (P(t+) + P(t-); for yes/no, P(Yes) +
    P(No)), and the commonest first-token pairs."""
    out: dict[str, Any] = {}
    for cond in sorted({x["condition"] for x in s}):
        rows = [x for x in s if x["condition"] == cond]
        mass = sorted(x["first_mass"] for x in rows if x.get("first_mass") is not None and x["first_mass"] == x["first_mass"])
        out[cond] = {"n": len(rows), "median": statistics.median(mass) if mass else float("nan"),
                     "p05": mass[int(0.05 * len(mass))] if mass else float("nan"),
                     "share_below_0.5": sum(m < 0.5 for m in mass) / len(mass) if mass else float("nan"),
                     "first_tokens": dict(Counter(f"{x.get('first_token_plus', '')!r} vs {x.get('first_token_minus', '')!r}"
                                                  for x in rows).most_common(5))}
    return out


FLIP_DEFINITIONS = {
    "flip_full_L_toward_success": "share of items whose full two-candidate L is above 0 under the aligned task and "
                                  "below 0 under the swapped task (one direction; NaN pairs left out)",
    "flip_first_token_sign_change": "share of items whose first-token L changes sign between the aligned and swapped "
                                    "tasks, either direction (the confirmatory decision_flip, as in RQ2's tables)",
}


def family_block(s: list[dict], primary: list[dict], successes: list[dict], spec: ModelSpec) -> dict[str, Any]:
    block = sizes.model_block(s, primary, successes)
    return {"spec": spec.record(), **block,
            "flips": {"flip_full_L_toward_success": block["flips_to_success"],
                      "flip_first_token_sign_change": block["decision_flip"], "definitions": FLIP_DEFINITIONS},
            "standardised_effect": {"two_candidate": standardised_effect(s, primary, "aligned", "swapped"),
                                    "yes_no": standardised_effect(s, primary, "yn_aligned", "yn_swapped")},
            "first_token_coverage": first_token_coverage(s)}


def prediction_check(block: dict[str, Any]) -> dict[str, Any]:
    two, yn = block["goal_effect"]["delta_L_ci"], block["yes_no"]["goal_effect"]["delta_L_ci"]
    out = {"two_candidate_ci_above_0": two[0] > 0, "yes_no_ci_above_0": yn[0] > 0}
    return {**out, "holds": all(out.values())}


def _resolve(ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else runs_root() / ref


def reference_scores(ref: dict) -> tuple[list[dict], dict[str, list[Path]], dict[str, Any]]:
    """A comparison model's scores from finished runs: {run: [conditions]} under `runs`, records of `model` only.

    A (item, condition) seen twice (two runs holding the same prompt) is kept once, the first in the order given;
    how many repeats there were, and the largest difference in L among them, are reported.
    """
    read: dict[str, list[Path]] = {}
    problems: dict[str, Any] = {}
    out: dict[tuple[str, str], dict] = {}
    repeats, worst = 0, 0.0
    for run, conds in ref["runs"].items():
        folders, why = complete_shards(str(_resolve(run)))
        if why:
            problems[run] = why
            continue
        read[run] = folders
        for p in folders:
            for x in read_jsonl(p / "scoring.jsonl"):
                if x["model"] != ref["model"] or x["condition"] not in conds:
                    continue
                key = (x["transition_id"], x["condition"])
                if key in out:
                    repeats += 1
                    worst = max(worst, abs(out[key]["L"] - x["L"]))
                    continue
                out[key] = x
    return list(out.values()), read, {"problems": problems, "repeats": repeats, "max_abs_L_difference": worst}


def _gate_records(folders: list[Path]) -> dict[str, Any]:
    out = {}
    for p in folders:
        f = p / "tokenization_checks.json"
        if f.exists():
            import json

            r = json.loads(f.read_text(encoding="utf-8"))
            out[p.name] = {k: r.get(k) for k in ("ok", "skipped", "n_candidates_checked", "first_token_pairs",
                                                  "joined_plain_text_mismatches", "max_prompt_tokens",
                                                  "attn_implementation_loaded", "dtype_loaded")}
    return out


def run_families_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    sources = tuple(str(_resolve(s)) for s in cfg.get("sources", SOURCES))
    sets = rq1x.rq1x_sets(load_items(sources))
    primary, successes = sets["primary"], sets["successes"]
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_successes": len(successes)},
                              "prediction": PREDICTION, "models": {}, "reference": {}, "prediction_check": {},
                              "waiting": {}}
    read: dict[str, list[Path]] = {}
    for tag in cfg["models"]:
        folders, problems = complete_shards(f"family_score_{tag}")
        if problems:
            report["waiting"][tag] = problems
            continue
        read[f"family_score_{tag}"] = folders
        s = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag]
        block = family_block(s, primary, successes, MODELS[tag])
        block["tokenization_gate"] = _gate_records(folders)
        report["models"][tag] = block
        report["prediction_check"][tag] = prediction_check(block)
    for tag, ref in (cfg.get("reference") or {}).items():
        s, got, info = reference_scores(ref)
        if info["problems"]:
            report["waiting"][f"reference {tag}"] = info["problems"]
            continue
        read.update({f"{tag}:{run}": f for run, f in got.items()})
        report["reference"][tag] = {**family_block(s, primary, successes, REFERENCE[tag]), "read": info}
    from wf.provenance import provenance, shard_inputs

    report["provenance"] = provenance(
        runs={**{s: Path(s) for s in sources}, **{k: (v[0].parent if v else Path(k)) for k, v in read.items()}},
        inputs=[Path(s) / "transitions.jsonl" for s in sources] + shard_inputs([p for fs in read.values() for p in fs]))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Model families, wave A (exploratory)", report)
    lines = [f"{tag}: dL {m['goal_effect']['delta_L_nats']:.2f} {m['goal_effect']['delta_L_ci']}, "
             f"yn dL {m['yes_no']['goal_effect']['delta_L_nats']:.2f}, prediction holds: {report['prediction_check'][tag]['holds']}"
             for tag, m in report["models"].items()]
    lines += [f"{tag} (reference): dL {m['goal_effect']['delta_L_nats']:.2f}, yn dL {m['yes_no']['goal_effect']['delta_L_nats']:.2f}"
              for tag, m in report["reference"].items()]
    ok = not report["waiting"]
    write_status(run_id, "\n".join(lines) + (f"\nwaiting: {sorted(report['waiting'])}" if not ok else ""), ok=False,
                 reported="families report" if ok else "partial families report")
    return 0
