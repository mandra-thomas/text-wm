"""Model families, wave A (decided 2026-11-20, before any score; notes/families_design.md).

The size series' scoring (wf/sizes.py: RQ2's prompt, example set 1, history only; the aligned, swapped and neutral
tasks, each as the two-candidate score and as the yes/no question; the determined successes for AUC) applied to
models from other families and generations. One scoring run per model (`family_score_<tag>`), with three additions
the Qwen3 series did not need:

1. Chat templates. Gemma 2's template refuses a system message, so the system text is folded into the first user
   turn (`fold_system`), the same way in every condition. Llama 3.2's template writes today's date into the system
   header unless given one, so it gets the fixed date its own template falls back to (and Llama 3.1's template
   uses): the prompt cannot change with the day a shard runs.
2. A tokenization gate (audit C9). Before any score (on a resume too), through the scorer's own encoding: every
   candidate text in the shard (primary items and successes) must tokenize alone exactly as the chat template
   tokenizes it as the assistant's reply; "Yes" and "No" must each be one token, equal to the template's rendering
   of that word as the reply after every yes/no prompt of every item; and the two candidates' first tokens must
   differ. The results go to tokenization_checks.json in the shard's folder, and a failure stops the job before it
   scores anything.
3. Attention. Gemma 2 is loaded with eager attention: transformers 4.51.3 defaults it to SDPA, which silently drops
   Gemma 2's attention-logit soft-capping, so the logits would not be the trained model's. The scorer refuses to
   continue if the loaded model reports another implementation.

`families_analysis` reads every family run and, for comparison, Qwen3-8B (size_score_qwen3-8b) and Llama-3.1-8B
(rq1_score, rq1x_score and rq1v_score, which hold the same prompts), and gives each the size series' block plus a
spread-standardised effect, both flip definitions by name, and first-token coverage, every model on the same
(item, condition) population. `family_placement_<tag>` (added 2026-11-20, before any wave A score was read) rescores
300 seeded primary items with Mistral's system text in the first user turn instead of its template's last-turn
placement; the analysis sets the goal effect under both placements side by side on those items.

Amendment (2026-11-20, before any wave A score was read): the primary cross-model metrics are the item share moved
toward Yes and the goal-to-evidence ratio R, both on the yes/no format (`primary_metrics`), in one Holm-corrected
family over the wave A models (`primary_family`); a model must pass a competence gate fixed before data
(`competence_gate`); models are compared on the same items with a paired game bootstrap, "matching" when R is within
a factor of 1.5 by TOST (`compare_models`); yes/no answers and flips are read as argmax choices (`argmax_answers`).

Third amendment (2026-11-20, still before any wave A score was read; found on the archived reference runs only): the
item share is the single primary cross-model metric. R is secondary and descriptive: across families it mostly
reflects how far each model pushes its log-odds on failures (its denominator), not goal bias. The primary family is
the item share per wave A model plus the planned paired comparisons of item share (difference and TOST equivalence
within ±0.05), Holm-corrected at a fixed size (`primary_family`). An R comparison falls back to the paired
difference R_A - R_B when the ratio is not estimable (`compare_models`).
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf import rq1x, sizes
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation, paired_conditions
from wf.rq2 import SOURCES, complete_shards, load_items
from wf.runs import _effect, _init_run
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
FORMATS = ("Amended 2026-11-20, before any wave A score was read: the yes/no format is the primary cross-family "
           "comparison, read through the item share moved toward Yes (third amendment: the single primary metric, "
           "per model and in the planned paired comparisons, one Holm-corrected family; a model must pass the "
           "competence gate); the goal-to-evidence ratio R, ΔL and the standardised d are secondary, and the "
           "two-candidate format is read within model only. Across families R mostly reflects how far each model "
           "pushes its log-odds on failures, not goal bias. The two-candidate score's validity was checked "
           "for Qwen3 only; Llama-3.1-8B's sign of L agreed with its written predictions on 65% of pilot items "
           "(notes/deviations.md). The logged prediction is unchanged.")
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


def template_reply(scorer: LogoddsScorer, messages: list[dict], reply: str) -> list[int]:
    """The ids the chat template gives `reply` as the assistant's turn after `messages`, up to its end-of-turn token.

    Rendered by the scorer's own encoding (`_encode`, with its template settings), so the gate sees the prompt
    exactly as scoring does. The conversation with the reply must begin with exactly the prompt the template gives
    before a reply (generation prompt included); otherwise a candidate scored after that prompt is not the reply as
    the template writes it. A system message is left out of this check: Mistral's template writes it into the last
    user turn, which is no longer last once a reply follows, so the two renders would differ for that reason alone.
    Where the prompt ends and the reply begins does not depend on the system message.
    """
    convo = [m for m in messages if m["role"] != "system"]
    prompt = _ids(scorer._encode(convo))
    full = _ids(scorer._encode(convo + [{"role": "assistant", "content": reply}], add_generation_prompt=False))
    if full[:len(prompt)] != prompt:
        raise TokenizationMismatch("the conversation with the reply does not begin with the prompt before it")
    tail = full[len(prompt):]
    special = _special_ids(scorer._tok)
    end = next((i for i, x in enumerate(tail) if x in special), len(tail))
    return tail[:end]


def check_candidate(scorer: LogoddsScorer, messages: list[dict], text: str) -> dict[str, Any]:
    """Does `text`, tokenized alone as the scorer tokenizes a candidate, equal the template's tokens for it as the
    reply?

    Also the verify_models check: the prompt's text and the candidate joined as plain text, tokenized together.
    That one is informative only: a template that puts a space before a reply (Mistral) fails it by design.
    """
    tok = scorer._tok
    reply = template_reply(scorer, messages, text)
    alone = _ids(scorer._encode_candidate(text))
    prompt = _ids(scorer._encode(messages))
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **scorer.template_settings())
    joined = _ids(tok(prompt_text + text, add_special_tokens=False))
    return {"alone_equals_template_reply": alone == reply, "joined_plain_text_equal": prompt + alone == joined,
            "alone": alone, "template_reply": reply, "prompt_tokens": len(prompt)}


YES_NO = ("Yes", "No")


def tokenization_checks(scorer: LogoddsScorer, spec: ModelSpec, primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """The gate, on every item of a shard, through the scorer's own encoding. Returns the record; its "ok" says
    whether scoring may start.

    Candidates: both texts of every item (primary items and successes), as the reply after the aligned prompt.
    Yes/No: the scored token of each word (tokenized alone, as score_one_token_pair does) must be one token and
    equal the template's rendering of that word as a real reply, after every yes/no prompt the shard scores
    (sizes.size_jobs: three per primary item, two per success).
    """
    tok = scorer._tok
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
                r = check_candidate(scorer, msgs, t["candidates"][name])
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
    scored = {word: _ids(scorer._encode_candidate(word)) for word in YES_NO}
    yes_no: dict[str, Any] = {word: {"alone": scored[word], "decoded": tok.decode(scored[word]), "template_reply": None,
                                     "n_checked": 0, "n_mismatched": 0} for word in YES_NO}
    for word in YES_NO:
        if len(scored[word]) != 1:
            problems.append(f"{word!r} is not one token: alone {scored[word]}")
    n_yes_no = 0
    for t, arm in sizes.size_jobs(primary, successes):  # every yes/no prompt a shard of these items scores
        if arm not in sizes.YN_ARMS:
            continue
        msgs = family_messages(spec, t, arm)
        n_yes_no += 1
        for word in YES_NO:
            rec = yes_no[word]
            try:
                reply = template_reply(scorer, msgs, word)
            except TokenizationMismatch as e:
                problems.append(f"{t['id']} {arm} {word!r}: {e}")
                continue
            rec["n_checked"] += 1
            if rec["template_reply"] is None:
                rec["template_reply"] = reply
            if reply != scored[word]:
                rec["n_mismatched"] += 1
                problems.append(f"{t['id']} {arm}: {word!r} is not the template's single token: alone "
                                f"{scored[word]}, template {reply}")
    if n_yes_no == 0:
        problems.append("no yes/no prompt checked")
    if scored["Yes"] == scored["No"]:
        problems.append("'Yes' and 'No' are the same token")
    return {"ok": not problems and n > 0, "n_candidates_checked": n, "problems": problems[:50],
            "n_problems": len(problems), "yes_no": yes_no, "n_yes_no_prompts_checked": n_yes_no,
            "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(10)},
            "joined_plain_text_mismatches": joined_mismatch, "max_prompt_tokens": max_prompt,
            "template_kwargs": scorer.template_settings(), "system_handling": spec.system}


def _gate(spec: ModelSpec) -> Callable[[Path, LogoddsScorer, list[dict], list[dict]], None]:
    def gate(d: Path, scorer: LogoddsScorer, primary: list[dict], successes: list[dict]) -> None:
        if scorer.mock:
            write_json(d / "tokenization_checks.json", {"ok": None, "skipped": "mock"})
            return
        rec = tokenization_checks(scorer, spec, primary, successes)
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


# ---------------- Mistral's system-text placement (exploratory check, logged 2026-11-20 before any score was read) ----

PLACEMENT_ARMS = ("aligned", "swapped", "yn_aligned", "yn_swapped")
PLACEMENT_ITEMS = 300
FIRST_TURN = "-firstturn"  # suffix of the model tag the first-turn scores are saved under


def placement_items(primary: list[dict], k: int = PLACEMENT_ITEMS) -> list[dict]:
    """A seeded sample of `k` primary items, in id order; the same for any order the items arrive in."""
    pool = sorted(primary, key=lambda t: t["id"])
    picked = random.Random(f"{C.RNG_SEED}|family_placement").sample(pool, min(k, len(pool)))
    return sorted(picked, key=lambda t: t["id"])


def first_turn_spec(spec: ModelSpec) -> ModelSpec:
    """`spec` with the system text folded into the first user turn (Gemma's handling), under its own tag."""
    if spec.system != "native":
        raise ValueError(f"{spec.tag} already folds its system text")
    return replace(spec, tag=spec.tag + FIRST_TURN, system="fold",
                   template_notes="Placement check: the system text opens the first user turn, not the last.")


def run_family_placement(cfg: dict) -> int:
    """`family_placement_<tag>`: the native model, its system text folded into the first user turn, on
    PLACEMENT_ITEMS seeded primary items under the aligned and swapped tasks in both formats. Scores are saved
    under the tag `<tag>-firstturn`, so they never mix with the model's own run."""
    tag = cfg["model"]
    if tag not in MODELS:
        raise SystemExit(f"unknown model {tag!r}; known: {sorted(MODELS)}")
    spec = first_turn_spec(MODELS[tag])
    k = int(cfg.get("n_items", PLACEMENT_ITEMS))
    cfg = {**cfg, "model_spec": spec.record(), "n_items": k, "arms": list(PLACEMENT_ARMS)}

    def open_scorer() -> LogoddsScorer:
        return LogoddsScorer(spec.hf_id, model_tag=spec.tag, attn_implementation=spec.attn_implementation,
                             template_kwargs=spec.template_kwargs)

    return sizes.score_shard(cfg, spec.tag, open_scorer, lambda t, arm: family_messages(spec, t, arm), gate=_gate(spec),
                             select=lambda p, s: (placement_items(p, k), []), arms=PLACEMENT_ARMS)


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


# ---------------- primary cross-model metrics, competence gate, comparisons (amendment, 2026-11-20, pre-data) ----

# Third amendment (2026-11-20, before any wave A score was read): the item share is the one primary cross-model
# metric; R is secondary (on the reference runs R differs 3.8-fold between Qwen3-8B and Llama-3.1-8B while their
# shares are equal, because R's denominator is each model's confidence on failures; notes/families_design.md).
PRIMARY_METRICS = ("item_share_toward_yes",)
SECONDARY_METRICS = ("goal_to_evidence_ratio",)
# Amended 2026-11-20 (second amendment, still before any wave A score was read): the first version required 5% to
# 95% Yes on failures, and its lower bound excluded Qwen3-8B for answering No to failures (notes/families_design.md).
COMPETENCE = {"auc_neutral_min": 0.70, "yes_share_neutral_failures_max": 0.95, "yes_share_neutral_successes_min": 0.05,
              "median_yes_no_mass_neutral_min": 0.5}
EQUIVALENCE_FACTOR = 1.5  # "matches": R within a factor of 1.5, TOST on log R (secondary)
SHARE_MARGIN = 0.05  # "matches": item shares within ±0.05 (5 points), TOST on the paired difference (primary)
DROPPED_DRAWS_MAX = 0.02  # the R ratio is not estimable when more than 2% of paired draws have R <= 0 or undefined
NOISE_FLOOR_NATS = 0.3  # yes/no ΔL differences between models below this are treated as noise
ALPHA = 0.05
REFERENCE_TAGS = ("qwen3-8b", "llama3.1-8b")
# The comparisons the outcome table reads (notes/families_design.md), fixed before data: each is two tests in the
# primary family, a difference in item share and its equivalence within ±SHARE_MARGIN.
PLANNED_COMPARISONS = (("gemma2-9b-it", "llama3.1-8b"), ("gemma2-9b-it", "qwen3-8b"),
                       ("mistral-7b-v0.3", "llama3.1-8b"), ("mistral-7b-v0.3", "qwen3-8b"),
                       ("gemma2-9b-it", "mistral-7b-v0.3"),
                       ("llama3.2-1b", "llama3.1-8b"), ("llama3.2-3b", "llama3.1-8b"))


def _yes_no_rows(s: list[dict], primary: list[dict], successes: list[dict]) -> tuple[list[tuple], list[str], list[str]]:
    """Per item, on the yes/no format: ("f", L aligned - L swapped, L neutral) for a failure with all three tasks
    scored, ("s", nan, L neutral) for a success scored under the neutral task; each item's game; each item's id."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    rows, games, ids = [], [], []
    for t in primary:
        k = [(t["id"], a) for a in ("yn_aligned", "yn_swapped", "yn_absent")]
        if all(x in by for x in k):
            rows.append(("f", by[k[0]] - by[k[1]], by[k[2]]))
            games.append(t["game_id"])
            ids.append(t["id"])
    for t in successes:
        if (t["id"], "yn_absent") in by:
            rows.append(("s", float("nan"), by[(t["id"], "yn_absent")]))
            games.append(t["game_id"])
            ids.append(t["id"])
    return rows, games, ids


def _share_toward_yes(rows: list[tuple]) -> float:
    d = [r[1] for r in rows if r[0] == "f"]
    return sum(x > 0 for x in d) / len(d) if d else float("nan")


def _ratio(rows: list[tuple]) -> float:
    """R = mean goal effect on failures / (mean neutral L on successes - mean neutral L on failures)."""
    f = [r for r in rows if r[0] == "f"]
    sc = [r[2] for r in rows if r[0] == "s"]
    if not f or not sc:
        return float("nan")
    evidence = sum(sc) / len(sc) - sum(r[2] for r in f) / len(f)
    return (sum(r[1] for r in f) / len(f)) / evidence if evidence > 0 else float("nan")


def primary_metrics(s: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """The item share (primary) and R (secondary since the third amendment), both on the yes/no format, each with a
    game-clustered 95% CI and a two-sided game sign-flip permutation p (for R, the p of its numerator: R is zero
    exactly when the goal effect is, and its denominator is fixed under the flips).

    item_share_toward_yes: the share of failures with L(aligned) > L(swapped).
    goal_to_evidence_ratio: R, the mean goal effect over the evidence effect (determined successes against
    failures, both under the neutral task). Rescaling every logit by one factor leaves R unchanged, but across
    families R mostly reflects how far each model pushes its log-odds on failures (the denominator), so it is
    read within model only.
    """
    rows, games, _ = _yes_no_rows(s, primary, successes)
    f = [(r, g) for r, g in zip(rows, games) if r[0] == "f"]
    sc = [r for r in rows if r[0] == "s"]
    share = cluster_bootstrap(_share_toward_yes, rows, games, seed=C.RNG_SEED)
    ratio = cluster_bootstrap(_ratio, rows, games, seed=C.RNG_SEED)
    fg = [g for _, g in f]
    p_share = pair_permutation([(r[1] > 0) - (r[1] < 0) for r, _ in f], seed=C.RNG_SEED, cluster_ids=fg) if f else float("nan")
    p_ratio = pair_permutation([r[1] for r, _ in f], seed=C.RNG_SEED, cluster_ids=fg) if f else float("nan")
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return {"item_share_toward_yes": {"estimate": share["estimate"], "ci": [share["ci_low"], share["ci_high"]],
                                      "null": 0.5, "n_items": len(f), "p_two_sided": p_share, "role": "primary"},
            "goal_to_evidence_ratio": {"estimate": ratio["estimate"], "ci": [ratio["ci_low"], ratio["ci_high"]], "null": 0.0,
                                       "role": "secondary (descriptive; across families mostly the evidence effect's size)",
                                       "goal_effect_nats": mean([r[1] for r, _ in f]),
                                       "evidence_effect_nats": mean([r[2] for r in sc]) - mean([r[2] for r, _ in f]),
                                       "n_failures": len(f), "n_successes": len(sc), "p_two_sided": p_ratio}}


def argmax_answers(s: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """Yes/no answers read as the model's choice (Yes when L > 0), not as mean P(Yes): the share answering Yes per
    task, and the flips between the swapped and the aligned task on failures, with game-clustered CIs."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}

    def yes_share(items: list[dict], arm: str) -> float:
        v = [by[(t["id"], arm)] > 0 for t in items if (t["id"], arm) in by]
        return sum(v) / len(v) if v else float("nan")

    pairs = [((by[(t["id"], "yn_swapped")] > 0, by[(t["id"], "yn_aligned")] > 0), t["game_id"]) for t in primary
             if (t["id"], "yn_swapped") in by and (t["id"], "yn_aligned") in by]
    to_yes = lambda v: sum(1 for sw, al in v if al and not sw) / len(v) if v else float("nan")
    to_no = lambda v: sum(1 for sw, al in v if sw and not al) / len(v) if v else float("nan")
    out: dict[str, Any] = {"yes_share_failures": {a: yes_share(primary, a) for a in sizes.YN_ARMS},
                           "yes_share_successes": {a: yes_share(successes, a) for a in ("yn_aligned", "yn_absent")},
                           "n_failure_pairs": len(pairs)}
    for name, fn in (("flip_no_to_yes", to_yes), ("flip_yes_to_no", to_no)):
        b = cluster_bootstrap(fn, [p for p, _ in pairs], [g for _, g in pairs], seed=C.RNG_SEED)
        out[name] = {"share": b["estimate"], "ci": [b["ci_low"], b["ci_high"]]}
    return out


def competence_gate(block: dict[str, Any]) -> dict[str, Any]:
    """Fixed before any wave A score (second amendment, 2026-11-20). On the yes/no format under the neutral task, a
    model is interpretable only if all four hold: AUC (successes against failures) at least 0.70; it answers Yes
    (argmax) to at most 95% of failures, so not Yes to everything; it answers Yes to at least 5% of determined
    successes, so not No to everything; and the median P(Yes) + P(No) is at least 0.5. A model that fails is
    reported as not interpretable, never as a null result."""
    auc = block["yes_no"]["auc"]["yn_absent"]["auc"]
    yes_f = block["argmax_answers"]["yes_share_failures"]["yn_absent"]
    yes_s = block["argmax_answers"]["yes_share_successes"]["yn_absent"]
    mass = block["first_token_coverage"].get("yn_absent", {}).get("median", float("nan"))
    checks = {"auc_neutral": {"value": auc, "pass": auc >= COMPETENCE["auc_neutral_min"]},
              "yes_share_neutral_failures": {"value": yes_f, "pass": yes_f <= COMPETENCE["yes_share_neutral_failures_max"]},
              "yes_share_neutral_successes": {"value": yes_s, "pass": yes_s >= COMPETENCE["yes_share_neutral_successes_min"]},
              "median_yes_no_mass_neutral": {"value": mass, "pass": mass >= COMPETENCE["median_yes_no_mass_neutral_min"]}}
    return {"interpretable": all(c["pass"] for c in checks.values()), "checks": checks, "rule": COMPETENCE}


def _draws(stat: Callable[[list], Any], rows: list, games: list[str], B: int = C.BOOTSTRAP_RESAMPLES) -> list:
    rng = random.Random(C.RNG_SEED)
    by: dict[str, list] = {}
    for r, g in zip(rows, games):
        by.setdefault(g, []).append(r)
    keys = sorted(by)
    return [stat([r for g in rng.choices(keys, k=len(keys)) for r in by[g]]) for _ in range(B)]


def _pct(v: list[float], q: float) -> float:
    v = sorted(x for x in v if x == x)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def _exp(x: float) -> float:
    return math.exp(x) if x == x else float("nan")


def _compare_R(rows: list[tuple], games: list[str]) -> dict[str, Any]:
    """R_A against R_B on paired rows ((row A, row B) per item), secondary.

    The ratio R_A / R_B is used when it is estimable: both point Rs above zero and at most DROPPED_DRAWS_MAX of the
    paired draws with either R at or below zero (or undefined), since a log ratio cannot be taken there and dropping
    those draws silently would bias the interval. Then "matches" is the TOST within a factor of EQUIVALENCE_FACTOR
    (90% interval of log(R_A / R_B)) and "differs" is its 95% interval excluding zero. Otherwise the ratio is not
    estimable (reported as such) and "differs" comes from the 95% interval of the paired difference R_A - R_B;
    there is no equivalence statement then. A model without the bias (R near zero) is the case this is for.
    """
    pair = lambda v: (_ratio([x for x, _ in v]), _ratio([y for _, y in v]))
    ra, rb = pair(rows)
    draws = _draws(pair, rows, games)
    B = len(draws)
    ok_ratio = [math.log(a / b) for a, b in draws if a > 0 and b > 0]
    dropped = B - len(ok_ratio)
    diffs = [a - b for a, b in draws if a == a and b == b]
    diff_est = ra - rb if ra == ra and rb == rb else float("nan")
    diff = {"estimate": diff_est, "ci95": [_pct(diffs, 0.025), _pct(diffs, 0.975)],
            "n_resamples_undefined": B - len(diffs)}
    estimable = bool(ra > 0 and rb > 0 and dropped <= DROPPED_DRAWS_MAX * B)
    out: dict[str, Any] = {"R_a": ra, "R_b": rb, "n_resamples": B, "n_resamples_dropped": dropped,
                           "max_share_dropped": DROPPED_DRAWS_MAX, "ratio_estimable": estimable, "difference": diff}
    if estimable:
        bound = math.log(EQUIVALENCE_FACTOR)
        est = math.log(ra / rb)
        lo90, hi90, lo95, hi95 = (_pct(ok_ratio, q) for q in (0.05, 0.95, 0.025, 0.975))
        matches = bool(-bound < lo90 and hi90 < bound)
        differs = bool(lo95 > 0 or hi95 < 0)
        out.update({"basis": "ratio", "estimate": math.exp(est), "ci90": [_exp(lo90), _exp(hi90)],
                    "ci95": [_exp(lo95), _exp(hi95)], "matches": matches, "differs": differs})
    else:
        lo95, hi95 = diff["ci95"]
        computable = diff_est == diff_est and lo95 == lo95
        differs = bool(computable and (lo95 > 0 or hi95 < 0))
        out.update({"basis": "difference" if computable else "not computable", "estimate": float("nan"),
                    "ci90": [float("nan")] * 2, "ci95": [float("nan")] * 2, "matches": False, "differs": differs,
                    "why_not_estimable": ("R at or below zero (or undefined) in the point estimate" if not (ra > 0 and rb > 0)
                                          else f"{dropped} of {B} paired draws have R at or below zero")})
    out["verdict"] = ("differs, within the margin" if out["differs"] and out["matches"] else "differs" if out["differs"]
                      else "matches" if out["matches"] else "not computable" if out["basis"] == "not computable"
                      else "inconclusive")
    return out


def _compare_share(rows: list[tuple], games: list[str]) -> dict[str, Any]:
    """Item share A minus B on the failures both models have (primary): the paired difference with 90% and 95%
    paired-bootstrap CIs; p_differs, a two-sided game sign-flip permutation p on the per-item differences
    (1 if only A moved toward Yes, -1 if only B did; swapping the models within a game flips its signs); and
    p_equivalent, the TOST p for a difference within ±SHARE_MARGIN, the larger of the two one-sided bootstrap ps."""
    f = [(x, y, g) for (x, y), g in zip(rows, games) if x[0] == "f"]
    d = [(x[1] > 0) - (y[1] > 0) for x, y, _ in f]
    fg = [g for _, _, g in f]
    stat = lambda v: sum(v) / len(v) if v else float("nan")
    est = stat(d)
    draws = [x for x in _draws(stat, d, fg) if x == x]
    lo90, hi90, lo95, hi95 = (_pct(draws, q) for q in (0.05, 0.95, 0.025, 0.975))
    B = len(draws)
    p_low = (sum(x <= -SHARE_MARGIN for x in draws) + 1) / (B + 1) if B else float("nan")
    p_high = (sum(x >= SHARE_MARGIN for x in draws) + 1) / (B + 1) if B else float("nan")
    return {"estimate": est, "ci90": [lo90, hi90], "ci95": [lo95, hi95], "n_items": len(d), "margin": SHARE_MARGIN,
            "p_differs": pair_permutation(d, seed=C.RNG_SEED, cluster_ids=fg) if d else float("nan"),
            "p_equivalent": max(p_low, p_high),
            "differs": bool(est == est and (lo95 > 0 or hi95 < 0)),
            "matches": bool(est == est and -SHARE_MARGIN < lo90 and hi90 < SHARE_MARGIN)}


def compare_models(sa: list[dict], sb: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    """Model A against model B on the same items (yes/no format), with a paired game-clustered bootstrap: games are
    resampled once for both models.

    item_share_difference (primary): A minus B, see `_compare_share`; "differs" when its 95% CI excludes zero and
    "matches" when its 90% CI lies within ±SHARE_MARGIN (descriptive; the primary family reads the Holm-adjusted
    p_differs and p_equivalent). R (secondary): see `_compare_R`. ΔL as a difference (A minus B) with a 95% CI; a ΔL
    difference under 0.3 nats is flagged as within the noise floor.
    """
    ra, ga, ia = _yes_no_rows(sa, primary, successes)
    rb, _, ib = _yes_no_rows(sb, primary, successes)
    keyed_b = dict(zip(ib, rb))
    common = [(r, g, i) for r, g, i in zip(ra, ga, ia) if i in keyed_b]  # the items both models have
    rows = [(r, keyed_b[i]) for r, _, i in common]
    games = [g for _, g, _ in common]
    mean_goal = lambda v: (lambda d: sum(d) / len(d) if d else float("nan"))([r[1] for r in v if r[0] == "f"])
    dfn = lambda v: mean_goal([x for x, _ in v]) - mean_goal([y for _, y in v])
    d = _draws(dfn, rows, games)
    out: dict[str, Any] = {"n_failures": sum(r[0] == "f" for r, _ in rows), "n_successes": sum(r[0] == "s" for r, _ in rows),
                           "item_share_difference": _compare_share(rows, games), "R": _compare_R(rows, games),
                           "delta_L_difference_nats": {"estimate": dfn(rows), "ci95": [_pct(d, 0.025), _pct(d, 0.975)]}}
    out["delta_L_difference_nats"]["within_noise_floor"] = abs(out["delta_L_difference_nats"]["estimate"]) < NOISE_FLOOR_NATS
    return out


def planned_comparisons(models: list[str], references: list[str]) -> list[tuple[str, str]]:
    """The planned comparisons whose two models are both in the analysis's configuration (fixed before data)."""
    have = set(models) | set(references)
    return [(a, b) for a, b in PLANNED_COMPARISONS if a in have and b in have]


def primary_family(blocks: dict[str, dict[str, Any]], expected: list[str], comparisons: dict[str, dict] | None = None,
                   planned: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """The one primary family (third amendment), Holm-corrected within it at a fixed size:

    - per wave A model in `expected`, the item share toward Yes against one half;
    - per planned comparison (A, B), two tests on the paired item-share difference: that it differs from zero
      (p_differs) and that it lies within ±SHARE_MARGIN (TOST, p_equivalent).

    `blocks` holds the wave A and reference blocks by tag; `comparisons` holds compare_models results keyed
    "A vs B". The family's size is fixed by `expected` and `planned`: a test whose model is still waiting, or whose p
    is missing or NaN, enters with p = 1 so the others are corrected for the full family, never a smaller one.
    A per-model test is supported when its Holm p < 0.05, the share is above one half and the model passed the
    competence gate. A comparison involving a model that failed the gate is "not interpretable".
    """
    comparisons = comparisons or {}
    planned = list(planned or [])
    p: dict[str, float] = {}
    info: dict[str, dict] = {}

    def interpretable(tag: str) -> bool | None:
        return blocks[tag]["competence"]["interpretable"] if tag in blocks else None

    for tag in expected:
        key = f"{tag}:item_share_toward_yes"
        rec = blocks[tag]["primary_metrics"]["item_share_toward_yes"] if tag in blocks else None
        pv = rec["p_two_sided"] if rec else float("nan")
        p[key] = pv if pv == pv else 1.0
        info[key] = {"kind": "share", "tag": tag, "rec": rec, "padded": pv != pv}
    for a, b in planned:
        c = comparisons.get(f"{a} vs {b}", {}).get("item_share_difference")
        for test, field in (("differs", "p_differs"), ("equivalent", "p_equivalent")):
            key = f"{a} vs {b}:share_{test}"
            pv = c[field] if c else float("nan")
            p[key] = pv if pv == pv else 1.0
            info[key] = {"kind": test, "pair": (a, b), "rec": c, "padded": pv != pv}
    adj = holm(p)
    tests: dict[str, dict] = {}
    for key, padj in adj.items():
        i = info[key]
        rec = i["rec"]
        if i["kind"] == "share":
            ok = interpretable(i["tag"])
            if rec is None:
                verdict = "waiting"
            elif not ok:
                verdict = "not interpretable"
            elif i["padded"]:
                verdict = "not computable"
            else:
                side = rec["estimate"] > rec["null"]
                verdict = "supported" if padj < ALPHA and side else "not supported"
            tests[key] = {"estimate": rec["estimate"] if rec else float("nan"), "ci": rec["ci"] if rec else [float("nan")] * 2,
                          "p": rec["p_two_sided"] if rec else float("nan"), "p_holm": padj, "verdict": verdict}
        else:
            oks = [interpretable(t) for t in i["pair"]]
            if rec is None or None in oks:
                verdict = "waiting"
            elif not all(oks):
                verdict = "not interpretable"
            elif i["padded"]:
                verdict = "not computable"
            else:
                verdict = "supported" if padj < ALPHA else "not supported"
            field = "p_differs" if i["kind"] == "differs" else "p_equivalent"
            tests[key] = {"estimate": rec["estimate"] if rec else float("nan"),
                          "ci": (rec["ci95"] if i["kind"] == "differs" else rec["ci90"]) if rec else [float("nan")] * 2,
                          "p": rec[field] if rec else float("nan"), "p_holm": padj, "verdict": verdict}
    readings = {}
    for a, b in planned:
        dv, ev = tests[f"{a} vs {b}:share_differs"]["verdict"], tests[f"{a} vs {b}:share_equivalent"]["verdict"]
        if dv not in ("supported", "not supported"):
            readings[f"{a} vs {b}"] = dv  # waiting, not interpretable or not computable
        elif dv == "supported":
            readings[f"{a} vs {b}"] = "differs, within the margin" if ev == "supported" else "differs"
        else:
            readings[f"{a} vs {b}"] = "matches" if ev == "supported" else "inconclusive (no difference detected)"
    return {"tests": tests, "comparison_readings": readings, "n_tests_planned": len(expected) + 2 * len(planned),
            "n_tests_padded": sum(i["padded"] for i in info.values()),
            "complete": all(t in blocks for t in expected) and all(f"{a} vs {b}" in comparisons for a, b in planned),
            "share_margin": SHARE_MARGIN, "alpha": ALPHA,
            "correction": "Holm over the full planned family (a missing or NaN p enters as 1); everything else is "
                          "descriptive"}


def family_block(s: list[dict], primary: list[dict], successes: list[dict], spec: ModelSpec) -> dict[str, Any]:
    block = sizes.model_block(s, primary, successes)
    out = {"spec": spec.record(), "primary_metrics": primary_metrics(s, primary, successes),
           "argmax_answers": argmax_answers(s, primary, successes), **block,
           "flips": {"flip_full_L_toward_success": block["flips_to_success"],
                     "flip_first_token_sign_change": block["decision_flip"], "definitions": FLIP_DEFINITIONS},
           "standardised_effect": {"two_candidate": standardised_effect(s, primary, "aligned", "swapped"),
                                   "yes_no": standardised_effect(s, primary, "yn_aligned", "yn_swapped")},
           "first_token_coverage": first_token_coverage(s)}
    out["competence"] = competence_gate(out)
    return out


def prediction_check(block: dict[str, Any]) -> dict[str, Any]:
    two, yn = block["goal_effect"]["delta_L_ci"], block["yes_no"]["goal_effect"]["delta_L_ci"]
    out = {"two_candidate_ci_above_0": two[0] > 0, "yes_no_ci_above_0": yn[0] > 0}
    return {**out, "holds": all(out.values())}


def same_population(s: list[dict], primary: list[dict], successes: list[dict]) -> tuple[list[dict], int]:
    """The records whose (item, condition) a family run scores (sizes.size_jobs), and how many others were dropped.

    Every model's block is read on this one population. The reference runs hold more: rq1_score scored the
    successes under the swapped task too, which the family runs do not, so Llama-3.1-8B's first-token coverage
    under the swapped task mixed successes into what is failures only for every other model (review, 2026-11-20).
    """
    wanted = {(t["id"], a) for t, a in sizes.size_jobs(primary, successes)}
    kept = [x for x in s if (x["transition_id"], x["condition"]) in wanted]
    return kept, len(s) - len(kept)


def placement_block(native: list[dict], first_turn: list[dict], primary: list[dict]) -> dict[str, Any]:
    """The goal effect (aligned minus swapped) under the native and the first-turn placement, on the items both
    scored, in both formats, and the per-item difference between the two (first turn minus native), with
    game-clustered CIs."""
    ids = {x["transition_id"] for x in first_turn} & {x["transition_id"] for x in native}
    items = [t for t in primary if t["id"] in ids]
    out: dict[str, Any] = {"n_items": len(items)}
    for fmt, (a, b) in (("two_candidate", ("aligned", "swapped")), ("yes_no", ("yn_aligned", "yn_swapped"))):
        nat = _item_effects(native, items, a, b)
        alt = _item_effects(first_turn, items, a, b)
        both = sorted(set(nat) & set(alt))
        rows = [alt[k] - nat[k] for k in both]
        boot = cluster_bootstrap(lambda v: sum(v) / len(v) if v else float("nan"), rows, [g for g, _ in both],
                                 seed=C.RNG_SEED)
        out[fmt] = {"native": _effect(native, items, primary=a, control=b),
                    "first_turn": _effect(first_turn, items, primary=a, control=b),
                    "first_turn_minus_native": {"delta_L_nats": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]],
                                                "n_items": len(rows)}}
    return out


def _item_effects(s: list[dict], items: list[dict], a: str, b: str) -> dict[tuple[str, str], float]:
    """{(game, item): L under `a` minus L under `b`} on the items given (NaN scores left out)."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    return {(t["game_id"], t["id"]): by[(t["id"], a)] - by[(t["id"], b)] for t in items
            if (t["id"], a) in by and (t["id"], b) in by}


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
            r = json.loads(f.read_text(encoding="utf-8"))
            out[p.name] = {k: r.get(k) for k in ("ok", "skipped", "n_candidates_checked", "n_yes_no_prompts_checked", "first_token_pairs",
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
                              "prediction": PREDICTION, "formats": FORMATS, "models": {}, "reference": {},
                              "prediction_check": {}, "placement": {}, "waiting": {},
                              "rules": {"competence": COMPETENCE, "equivalence_factor_R": EQUIVALENCE_FACTOR,
                                        "share_margin": SHARE_MARGIN, "R_ratio_max_share_dropped": DROPPED_DRAWS_MAX,
                                        "noise_floor_delta_L_nats": NOISE_FLOOR_NATS, "alpha": ALPHA,
                                        "primary_metrics": list(PRIMARY_METRICS),
                                        "secondary_metrics": list(SECONDARY_METRICS)}}
    read: dict[str, list[Path]] = {}
    scores: dict[str, list[dict]] = {}
    ref_scores: dict[str, list[dict]] = {}
    for tag in cfg["models"]:
        folders, problems = complete_shards(f"family_score_{tag}")
        if problems:
            report["waiting"][tag] = problems
            continue
        read[f"family_score_{tag}"] = folders
        s, dropped = same_population([x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag],
                                     primary, successes)
        scores[tag] = s
        block = family_block(s, primary, successes, MODELS[tag])
        block["tokenization_gate"] = _gate_records(folders)
        block["records_outside_population"] = dropped
        report["models"][tag] = block
        report["prediction_check"][tag] = prediction_check(block)
    for tag, ref in (cfg.get("reference") or {}).items():
        s, got, info = reference_scores(ref)
        if info["problems"]:
            report["waiting"][f"reference {tag}"] = info["problems"]
            continue
        s, info["records_outside_population"] = same_population(s, primary, successes)
        read.update({f"{tag}:{run}": f for run, f in got.items()})
        ref_scores[tag] = s
        report["reference"][tag] = {**family_block(s, primary, successes, REFERENCE[tag]), "read": info}
    # every wave A model against each reference model, and the planned comparisons among wave A models, paired on
    # the same items; the planned ones' item-share tests are in the primary family, the rest is descriptive
    planned = planned_comparisons(list(cfg["models"]), list(cfg.get("reference") or {}))
    both = {**scores, **ref_scores}
    pairs = [(tag, ref) for tag in scores for ref in ref_scores]
    pairs += [(a, b) for a, b in planned if (a, b) not in pairs and a in both and b in both]
    report["comparisons"] = {f"{a} vs {b}": compare_models(both[a], both[b], primary, successes) for a, b in pairs}
    report["primary_family"] = primary_family({**report["models"], **report["reference"]}, list(cfg["models"]),
                                              report["comparisons"], planned)
    for tag, run in (cfg.get("placement") or {}).items():
        folders, problems = complete_shards(run)
        if problems or tag not in scores:
            report["waiting"][f"placement {tag}"] = problems or [f"family_score_{tag} not read"]
            continue
        read[run] = folders
        alt = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag + FIRST_TURN]
        report["placement"][tag] = {**placement_block(scores[tag], alt, primary), "run": run,
                                    "tokenization_gate": _gate_records(folders)}
    from wf.provenance import provenance, shard_inputs

    report["provenance"] = provenance(
        runs={**{s: Path(s) for s in sources}, **{k: (v[0].parent if v else Path(k)) for k, v in read.items()}},
        inputs=[Path(s) / "transitions.jsonl" for s in sources] + shard_inputs([p for fs in read.values() for p in fs]))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Model families, wave A (exploratory)", report)
    def head(m: dict) -> str:
        pm = m["primary_metrics"]
        return (f"{'interpretable' if m['competence']['interpretable'] else 'NOT INTERPRETABLE'}; share toward Yes "
                f"{pm['item_share_toward_yes']['estimate']:.3f} (primary), R {pm['goal_to_evidence_ratio']['estimate']:.3f} "
                f"(secondary); "
                f"yn dL {m['yes_no']['goal_effect']['delta_L_nats']:.2f}, two-candidate dL {m['goal_effect']['delta_L_nats']:.2f}")

    lines = [f"{tag}: {head(m)}; prediction holds: {report['prediction_check'][tag]['holds']}" for tag, m in report["models"].items()]
    lines += [f"{tag} (reference): {head(m)}" for tag, m in report["reference"].items()]
    lines += [f"primary family {k}: {v['verdict']} (Holm p {v['p_holm']:.4f})" for k, v in report["primary_family"]["tests"].items()]
    lines += [f"{k}: item share {v}" for k, v in report["primary_family"]["comparison_readings"].items()]
    lines += [f"{tag} placement, first turn minus native: yn {p['yes_no']['first_turn_minus_native']['delta_L_nats']:.2f}, "
              f"two-candidate {p['two_candidate']['first_turn_minus_native']['delta_L_nats']:.2f} ({p['n_items']} items)"
              for tag, p in report["placement"].items()]
    ok = not report["waiting"]
    write_status(run_id, "\n".join(lines) + (f"\nwaiting: {sorted(report['waiting'])}" if not ok else ""), ok=False,
                 reported="families report" if ok else "partial families report")
    return 0
