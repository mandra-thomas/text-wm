"""Exploratory mechanism battery (decided 2026-11-20, before any of its scores; notes/mechanism2_design.md, amended
after review 1, review 2 and review 3, still before any score).

Why does naming the action's object in the task push the world model toward success on determined failures? Candidate
accounts, not exclusive (H1 no state tracking, H2 task as evidence, H3 referent salience, H4 wishful prediction, H5
policy blending, H6 prompt artefacts, split into H6_lexical, i.e. name salience or word overlap, and H6_position),
predict different size zones for a set of task lines. Every arm differs from the others only in the task line, or, for
the position arms (step 0 only), only in where the task line sits in the history. After review 2 the primary contrasts
are minimal pairs where the design allows: P1 and P2 compare lines that differ only in the desire (the swapped task
kept in both, or only the hope's polarity changed), and P8 compares the user's report that the object is or is not in
the receptacle (the same words but "no"). Each arm is scored in two formats: the
two-candidate log-odds of the success text against "Nothing happens." (`cand`) and the yes/no question of rq1v (`yn`,
condition name prefixed "yn_").

Items: a seeded sample of RQ2's primary items stratified by game and failure reason (SAMPLE_N), plus every step-0
item, so the position test covers all of them. Every content arm is scored on every item of that union (the negated
and affirmed arms on take items only, the position arms on step-0 items only); the primary contrasts read the sample, a
restriction of it (P5, P7, P8) or the step-0 items (P6).

Models: Qwen3-8B and Llama-3.1-8B (primary, every arm) and Mistral-7B-Instruct-v0.3 (secondary, descriptive, only the
arms the primary contrasts need, on the sample, and P6's arms on every step-0 item), at pinned revisions.

After review 3: the mention component M on P7's items (half of negated plus half of affirmed, minus swapped) in the
mixture readout and the summary; "consistent" needs a risky prediction (a discriminating contrast whose allowed zones
exclude Z); a negation placebo pair (the user's report about the swapped object) as a named secondary; desire_check -
desire_keep as a second wishful-against-blending separator; exploratory readouts for H7_check (the task directs which
part of the state the model checks), outside the verdicts; NaN scores on a primary arm refuse the analysis.

`mech2_analysis` checks every score against the current items (item hashes, expected against present scores per model,
arm and format) and refuses a mismatch; it reports the eight primary contrasts per model and format (Holm within each,
family of eight), each contrast's size
zones and category, the readout per format and under the two-format rule, every arm against the swapped, neutral and
aligned tasks, plus two analysis-only checks on earlier runs: few-shot task overlap (rq1_score) and the knowledge
split (mech_score).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf import families, rq1, rq1v, rq1x
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation
from wf.data.goals import parse_task_line
from wf.data.transitions import history_facts
from wf.rq2 import SOURCES, analysis_provenance, complete_shards, load_items, shard_items
from wf.runs import _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.scoring.prompts import WM_FEWSHOT_SETS
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, runs_root, sigmoid, write_json, write_status

# ---------------- design constants (fixed before any score; notes/mechanism2_design.md) ----------------

SAMPLE_N = 1000
SAMPLE_SEED = "20261120|mech2"
FORMATS = ("cand", "yn")
# Amendment after review 2 (2026-11-20, before any score): desire_keep_sw, desire_check_can and affirmed added (the
# minimal-pair partners of mention_ignore, desire_check and negated); swapped_described dropped (rq1x measured it); the
# position arms are scored at step 0 only, and the `_end` arms are gone (at step 0 `_end` repeated the plain prompt,
# after step 0 no primary contrast read it).
# Amendment after review 3: the negation placebo pair (placebo_negated, placebo_affirmed), the user's report about the
# swapped object, on the negated arm's items.
CONTENT_ARMS = ("aligned", "swapped", "neutral", "desire_keep", "desire_keep_sw", "desire_check", "desire_check_can",
                "mention_ignore", "mention_exists", "described", "done", "cancelled", "negated", "affirmed",
                "placebo_negated", "placebo_affirmed")
POSITION_ARMS = ("aligned_start", "swapped_start")
ARMS = CONTENT_ARMS + POSITION_ARMS
TASK_PREFIX = "Your task is to: "

# Templates. {obj} is the goal object's type as the task names it (the action names an instance of it), {a} its
# article, {pp} and {prep} come from the action's verb, {rec} is the receptacle instance the action names.
PARTICIPLE = {"take": "taken", "move": "moved", "heat": "heated", "cool": "cooled", "clean": "cleaned"}
PREPOSITION = {"take": "from", "move": "to", "heat": "with", "cool": "with", "clean": "with"}
DESIRE_KEEP = "Your task is to: make sure that the {obj} is not {pp}; leave it where it is."
DESIRE_CHECK = "Your task is to: find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it cannot."
MENTION_IGNORE = "{swapped} Ignore the {obj}."
# Amendment after review 2: minimal-pair partners. desire_keep_sw is desire_keep's sentence after the swapped task line,
# so it differs from mention_ignore only in what the sentence asks (P1); desire_check_can differs from desire_check only
# in the hope's polarity ("can" for "cannot", P2).
DESIRE_KEEP_SW = "{swapped} Make sure that the {obj} is not {pp}; leave it where it is."
DESIRE_CHECK_CAN = "Your task is to: find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it can."
DONE = "Your task was to: {task} It is already done; nothing is left to do."
CANCELLED = "Your task was to: {task} The user has cancelled it; nothing is left to do."
# Amendment after review 1 (2026-11-20, before any score): the negated-existence arm, built for take actions only (the
# one verb whose success needs the object in the receptacle the action names). The claim is the user's report, not the
# narrator's: on many items an instance of {obj} is in {rec}, so a bare "There is no ..." would state a falsehood.
# "in" for every receptacle type, as ALFWorld's own task lines write it ("put a mug in countertop.").
NEGATED = "{swapped} The user says there is no {obj} in the {rec}."
# Amendment after review 2: the user's report that the object IS in the receptacle, on the negated arm's items; the
# same words as NEGATED but the article for "no" (P8 = negated - affirmed).
AFFIRMED = "{swapped} The user says there is {a} {obj} in the {rec}."
# Amendment after review 3: the negation placebo. The same report about the SWAPPED task's object (not the action's), in
# the action's receptacle, denied or affirmed: the two lines differ only in "no" against the article, as P8's do, but
# neither says anything about the action's object. Task as evidence predicts no difference (about zero); a model whose
# success reading drops at the word "no" whatever it denies predicts a difference below zero.
PLACEBO_NEGATED = "{swapped} The user says there is no {swobj} in the {rec}."
PLACEBO_AFFIRMED = "{swapped} The user says there is {a} {swobj} in the {rec}."
_ACTION = re.compile(r"^(take|move|heat|cool|clean) ([a-z]+) (\d+) (?:from|to|with) ([a-z]+ \d+)$")

# Task types, for template coverage and the placement-task population of P5.
TASK_TYPES = ("pick", "pick_two", "clean", "heat", "cool", "examine")
PLACEMENT_TYPES = ("pick", "pick_two", "clean", "heat", "cool")


def task_type(task_line: str) -> str:
    """The ALFWorld task type of a task line (wf.data.goals' templates)."""
    s = task_line.split("Your task is to:", 1)[-1].strip().lower()
    if re.match(r"^(?:find|put) two ", s):
        return "pick_two"
    if s.startswith(("look at ", "examine the ")):
        return "examine"
    for kind, adj in (("clean", "clean"), ("heat", "hot"), ("cool", "cool")):
        if s.startswith(f"{kind} some ") or s.startswith(f"put a {adj} ") or s.startswith(f"put some {adj} "):
            return kind
    if re.match(r"^put (?:a|an|some) ", s):
        return "pick"
    raise ValueError(f"unrecognized task line: {task_line!r}")


def parse_action(action: str) -> tuple[str, str, str] | None:
    """(verb, object type, receptacle instance) of take/move/heat/cool/clean actions, else None."""
    m = _ACTION.match(action or "")
    return (m[1], m[2], m[4]) if m else None


def _task_text(task_line: str) -> str:
    if not task_line.startswith(TASK_PREFIX):
        raise ValueError(f"task line does not start with {TASK_PREFIX!r}: {task_line!r}")
    text = task_line[len(TASK_PREFIX):].strip()
    if not text.endswith("."):
        raise ValueError(f"task line does not end with a period: {task_line!r}")
    return text


def task_line(t: dict, arm: str) -> str | None:
    """The task line an arm shows for item `t`, or None when the item cannot carry the arm."""
    if arm in POSITION_ARMS and t.get("history_len", 0) > 0:
        return None  # amendment after review 2: position arms at step 0 only (P6 reads no other item)
    base = arm.rsplit("_", 1)[0] if arm in POSITION_ARMS else arm
    obj = t.get("goal_object") or ""
    if base == "aligned":
        return t["task_line"]
    if base == "neutral":
        return rq1.ABSENT_LINE
    if not t.get("swapped_task_line") or not t.get("swapped_goal_object"):
        return None
    if base == "swapped":
        return t["swapped_task_line"]
    if base in ("desire_keep", "desire_keep_sw", "desire_check", "desire_check_can"):
        p = parse_action(t["action"])
        if not p or p[1] != obj:
            return None
        verb, _, rec = p
        if base == "desire_keep":
            return DESIRE_KEEP.format(obj=obj, pp=PARTICIPLE[verb])
        if base == "desire_keep_sw":
            return DESIRE_KEEP_SW.format(swapped=t["swapped_task_line"], obj=obj, pp=PARTICIPLE[verb])
        hope = DESIRE_CHECK if base == "desire_check" else DESIRE_CHECK_CAN
        return hope.format(a=rq1._article(obj), obj=obj, pp=PARTICIPLE[verb], prep=PREPOSITION[verb], rec=rec)
    if base == "mention_ignore":
        return MENTION_IGNORE.format(swapped=t["swapped_task_line"], obj=obj)
    if base in ("negated", "affirmed", "placebo_negated", "placebo_affirmed"):
        p = parse_action(t["action"])
        if not p or p[0] != "take" or p[1] != obj:
            return None
        sw = t["swapped_task_line"]
        if base == "negated":
            return NEGATED.format(swapped=sw, obj=obj, rec=p[2])
        if base == "affirmed":
            return AFFIRMED.format(swapped=sw, a=rq1._article(obj), obj=obj, rec=p[2])
        swobj = t["swapped_goal_object"]
        if base == "placebo_negated":
            return PLACEBO_NEGATED.format(swapped=sw, swobj=swobj, rec=p[2])
        return PLACEBO_AFFIRMED.format(swapped=sw, a=rq1._article(swobj), swobj=swobj, rec=p[2])
    if base == "mention_exists":
        return rq1.goal_line(t, "swapped_plus")
    if base == "described":
        return rq1x.describe_line(t["task_line"], obj) if rq1x.desc_ok(t) else None
    if base == "done":
        return DONE.format(task=_task_text(t["task_line"]))
    if base == "cancelled":
        return CANCELLED.format(task=_task_text(t["task_line"]))
    raise ValueError(f"unknown arm {arm!r}")


def move_task_line(history_text: str, where: str) -> str:
    """The history with its one task line moved to the first line ("start") or the last line ("end")."""
    lines = history_text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("Your task is to:")]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one task line, found {len(idx)}")
    line = lines.pop(idx[0])
    if where == "start":
        return "\n".join([line] + lines)
    if where == "end":
        return "\n".join(lines + [line])
    raise ValueError(where)


def usual_position(t: dict) -> str:
    """Where the task line sits in the item's own history: the last line at step 0, the second line after."""
    lines = t["history_text"].split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("Your task is to:")]
    if len(idx) != 1:
        raise ValueError(f"{t['id']}: expected exactly one task line")
    return "end" if idx[0] == len(lines) - 1 else "start" if idx[0] == 0 else f"line {idx[0] + 1}"


def history_for(t: dict, arm: str) -> str:
    if arm in POSITION_ARMS:
        return move_task_line(t["history_text"], arm.rsplit("_", 1)[1])
    return t["history_text"]


def _yes_no(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    last = msgs[-1]
    if not last["content"].endswith("Next observation:"):
        raise ValueError("prompt does not end with 'Next observation:'")
    return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + rq1v.YN_QUESTION}]


def condition(arm: str, fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(fmt)
    return arm if fmt == "cand" else f"yn_{arm}"


def split_condition(cond: str) -> tuple[str, str]:
    return (cond[3:], "yn") if cond.startswith("yn_") else (cond, "cand")


def arm_messages(t: dict, arm: str, fmt: str = "cand") -> list[dict[str, str]] | None:
    """The prompt for one item, arm and format: RQ2's prompt (example set 1, history only) with the arm's task line
    (and, for position arms, the task line moved); the yes/no format ends with rq1v's question."""
    line = task_line(t, arm)
    if line is None:
        return None
    msgs = build_messages({**t, "history_text": history_for(t, arm)}, "aligned", task_line=line)
    return _yes_no(msgs) if fmt == "yn" else msgs


# ---------------- items ----------------


def _rank(key: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}|{key}".encode()).hexdigest()


def stratified_sample(primary: list[dict], n: int = SAMPLE_N) -> set[str]:
    """`n` item ids, allocated to (game, failure reason) strata in proportion to their size (largest remainder,
    ties by a hash of the stratum), and within a stratum by a hash of the item id; independent of input order."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in primary:
        strata[(t["game_id"], t.get("determination_reason") or "")].append(t)
    total = len(primary)
    if n >= total:
        return {t["id"] for t in primary}
    exact = {k: len(v) * n / total for k, v in strata.items()}
    quota = {k: math.floor(x) for k, x in exact.items()}
    left = n - sum(quota.values())
    for k in sorted(strata, key=lambda k: (-(exact[k] - quota[k]), _rank("|".join(k))))[:left]:
        quota[k] += 1
    chosen: set[str] = set()
    for k, ts in strata.items():
        chosen |= {t["id"] for t in sorted(ts, key=lambda t: _rank(t["id"]))[:quota[k]]}
    return chosen


def battery_items(primary: list[dict], n: int = SAMPLE_N) -> dict[str, Any]:
    """The stratified sample, the step-0 items and their union (in the primary set's order)."""
    sample = stratified_sample(primary, n)
    step0 = {t["id"] for t in primary if t.get("history_len", 0) == 0}
    return {"sample": sample, "step0": step0, "union": [t for t in primary if t["id"] in sample | step0]}


def jobs(items: list[dict], arms: tuple[str, ...] = ARMS) -> list[tuple[dict, str, str]]:
    """(item, arm, format) for every arm of `arms` that builds for an item (position arms: step 0 only)."""
    out = []
    for t in items:
        for arm in arms:
            if task_line(t, arm) is None:
                continue
            out += [(t, arm, fmt) for fmt in FORMATS]
    return out


def done_contradicted(t: dict) -> bool:
    """Does the history contradict "the task is already done"? True when every receptacle of the task's target type
    in the room was seen with its contents listed and none held the goal object's type (a done placement task leaves
    the object at a target receptacle). Examine tasks are not placement tasks and are never flagged."""
    if task_type(t["task_line"]) not in PLACEMENT_TYPES:
        return False
    target = parse_task_line(t["task_line"]).receptacle
    facts = history_facts(t["history_text"])
    targets = sorted(r for r in (t.get("scene_receptacles") or facts.room) if re.sub(r"\s+\d+$", "", r) == target)
    if not targets or any(r not in facts.contents for r in targets):
        return False
    obj = t["goal_object"]
    return not any(re.sub(r"\s+\d+$", "", o) == obj for r in targets for o in facts.contents[r])


def takes_from_target_type(t: dict) -> bool:
    """Does a placement item's action take the object from a receptacle of the task's target type? There "done" (the
    object sits at a target receptacle) makes the take MORE likely to work under H2, not less, so these items are left
    out of P5 and read on their own (amendment after review 1)."""
    p = parse_action(t["action"])
    if not p or p[0] != "take" or task_type(t["task_line"]) not in PLACEMENT_TYPES:
        return False
    return re.sub(r"\s+\d+$", "", p[2]) == parse_task_line(t["task_line"]).receptacle


def p5_eligible(t: dict) -> bool:
    """P5's items: placement tasks whose action does not take from a target-type receptacle and whose history does not
    contradict "done", the items on which every reading of H2 predicts done below cancelled."""
    return (task_type(t["task_line"]) in PLACEMENT_TYPES and not takes_from_target_type(t)
            and not done_contradicted(t))


def is_take(t: dict) -> bool:
    """Does the item's action take the goal object (the items the negated arm is built for)?"""
    p = parse_action(t["action"])
    return bool(p and p[0] == "take" and p[1] == t.get("goal_object"))


def negated_status(t: dict) -> tuple[str, str]:
    """For a take item, the user's claim "there is no {obj} in the {rec}": is it true of the scene at the start of the
    game, and what does the history show of the receptacle ("contents not shown", "shown, without it" or "shown, holds
    it", the last contradicting the claim)?"""
    _, obj, rec = parse_action(t["action"])
    at_start = any(re.sub(r"\s+\d+$", "", o) == obj and rec in locs for o, locs in (t.get("scene_locations") or {}).items())
    facts = history_facts(t["history_text"])
    if rec not in facts.contents:
        seen = "contents not shown"
    else:
        seen = "shown, holds it" if any(re.sub(r"\s+\d+$", "", o) == obj for o in facts.contents[rec]) else "shown, without it"
    return ("false at start" if at_start else "true at start"), seen


def p7_eligible(t: dict) -> bool:
    """P7's items: take items whose history does not show the object in the receptacle (where it does, the user's claim
    is contradicted and H2 predicts no shift)."""
    return is_take(t) and negated_status(t)[1] != "shown, holds it"


def completes_task(t: dict) -> bool:
    """Would the action complete the task if it succeeded (H7_check (iii), amendment after review 3)? True for a move
    of the goal object into a receptacle of a placement task's target type, the last step of every placement task."""
    p = parse_action(t["action"])
    if not p or p[0] != "move" or p[1] != t.get("goal_object") or task_type(t["task_line"]) not in PLACEMENT_TYPES:
        return False
    return re.sub(r"\s+\d+$", "", p[2]) == parse_task_line(t["task_line"]).receptacle


# ---------------- models (amendment after review 1: Mistral-7B-Instruct-v0.3 as a secondary model) ----------------

# Primary: the two few-shot models, Holm within model and format. Secondary: Mistral-7B-Instruct-v0.3 (the largest
# practical effect in the families' wave A), reported descriptively, at the families registry's pin and with its
# chat-template handling (wf/families.py: system text where its own template puts it, as in its family run).
PRIMARY_MODELS = tuple(rq1.FEWSHOT)
SECONDARY_MODELS = ("mistral-7b-v0.3",)


def model_entry(tag: str) -> tuple[str, dict[str, Any], Callable[[list[dict]], list[dict]]]:
    """(Hugging Face id, LogoddsScorer keyword arguments, prompt transform) of a model the battery scores."""
    for name, t, _ft in rq1.MODELS:
        if t == tag and tag in PRIMARY_MODELS:
            return name, {}, lambda msgs: msgs
    if tag in SECONDARY_MODELS:
        spec = families.MODELS[tag]
        kw = {"attn_implementation": spec.attn_implementation, "template_kwargs": dict(spec.template_kwargs)}
        return spec.hf_id, kw, (families.fold_system if spec.system == "fold" else (lambda msgs: msgs))
    raise SystemExit(f"unknown model {tag!r}; the battery scores {PRIMARY_MODELS + SECONDARY_MODELS}")


def model_arms(tag: str) -> tuple[str, ...]:
    """The arms a model is scored on: every arm for the primary models; for the secondary model only the arms the
    primary contrasts (and their reference effect) read (amendment after review 2, GPU budget)."""
    return ARMS if tag in PRIMARY_MODELS else PRIMARY_ARMS


def model_jobs(tag: str, items: list[dict], sample: set[str]) -> list[tuple[dict, str, str]]:
    """(item, arm, format) a model is scored on. The primary models: every arm on every item of `items`. The secondary
    model (amendment after review 3, GPU budget): its arms on the sample's items only, plus P6's arms (the two position
    arms and the aligned and swapped arms they are compared with) on every step-0 item, so its P6 reads the same 514
    items as the primary models'."""
    out = jobs(items, model_arms(tag))
    if tag in PRIMARY_MODELS:
        return out
    return [(t, a, f) for t, a, f in out if t["id"] in sample or a in P6_ARMS]


def model_messages(tag: str, t: dict, arm: str, fmt: str = "cand") -> list[dict[str, str]] | None:
    """The prompt a model is scored on: `arm_messages`, through the model's template handling."""
    msgs = arm_messages(t, arm, fmt)
    return None if msgs is None else model_entry(tag)[2](msgs)


# ---------------- tokenization gate (as wf/families.py, audit C9) ----------------

TokenizationMismatch = families.TokenizationMismatch
_ids = families._ids
# The ids the chat template gives a reply after the prompt, rendered by the scorer's own encoding (`_encode` with its
# template settings), so the gate sees every prompt exactly as scoring does.
template_reply = families.template_reply


def _alone(scorer: LogoddsScorer, text: str) -> list[int]:
    """A candidate's ids as the scorer reads them (`_encode_candidate`: tokenized alone, no special tokens)."""
    return _ids(scorer._encode_candidate(text))


def tokenization_checks(scorer: LogoddsScorer, todo: list[tuple[dict, str, str]],
                        prompt: Callable[[list[dict]], list[dict]] = lambda msgs: msgs) -> dict[str, Any]:
    """The gate, through the scorer's own tokenizer, template and template settings, before any score (on a resume too):

    - each item's two candidates, tokenized alone, equal the template's tokens for them as the reply after the item's
      aligned prompt, and their first tokens differ;
    - "Yes" and "No" are one token each, different, and equal the template's reply after every yes/no prompt to score.

    `prompt` is the model's template handling (`model_entry`), applied as scoring applies it.
    """
    problems: list[str] = []
    items = {t["id"]: t for t, _, _ in todo}
    first_pairs: Counter = Counter()
    n_cand = 0
    for t in items.values():
        msgs = prompt(arm_messages(t, "aligned"))
        ids = {}
        for name in ("o_plus", "o_minus"):
            try:
                reply = template_reply(scorer, msgs, t["candidates"][name])
            except TokenizationMismatch as e:
                problems.append(f"{t['id']} {name}: {e}")
                continue
            alone = _alone(scorer, t["candidates"][name])
            n_cand += 1
            if alone != reply:
                problems.append(f"{t['id']} {name}: alone {alone[:6]} vs template {reply[:6]}")
            ids[name] = alone
        if len(ids) == 2 and (not ids["o_plus"] or not ids["o_minus"] or ids["o_plus"][0] == ids["o_minus"][0]):
            problems.append(f"{t['id']}: the candidates share their first token")
        elif len(ids) == 2:
            first_pairs[(scorer._tok.decode(ids["o_plus"][:1]), scorer._tok.decode(ids["o_minus"][:1]))] += 1
    words = {w: _alone(scorer, w) for w in ("Yes", "No")}
    for w, ids in words.items():
        if len(ids) != 1:
            problems.append(f"{w!r} is not one token: {ids}")
    if words["Yes"] == words["No"]:
        problems.append("'Yes' and 'No' are the same token")
    n_yn = 0
    for t, arm, fmt in todo:
        if fmt != "yn":
            continue
        msgs = prompt(arm_messages(t, arm, "yn"))
        n_yn += 1
        for w in ("Yes", "No"):
            try:
                reply = template_reply(scorer, msgs, w)
            except TokenizationMismatch as e:
                problems.append(f"{t['id']} {arm} {w!r}: {e}")
                continue
            if reply != words[w]:
                problems.append(f"{t['id']} {arm}: {w!r} alone {words[w]} vs template {reply}")
    return {"ok": not problems and (n_cand > 0 or not items), "n_candidates_checked": n_cand,
            "n_yes_no_prompts_checked": n_yn, "n_yes_no_jobs": sum(f == "yn" for _, _, f in todo),
            "yes_no_ids": words, "n_problems": len(problems), "problems": problems[:50],
            "template_settings": scorer.template_settings(),
            "first_token_pairs": {f"{a!r} vs {b!r}": c for (a, b), c in first_pairs.most_common(5)}}


def gate(d: Path, scorer: LogoddsScorer, todo: list[tuple[dict, str, str]], model_name: str,
         prompt: Callable[[list[dict]], list[dict]] = lambda msgs: msgs) -> None:
    """Stop before any score when the model is not at its pinned revision or the tokenization gate fails."""
    name = f"tokenization_checks_{scorer.model_tag}.json"
    if scorer.mock:
        write_json(d / name, {"ok": None, "skipped": "mock"})
        return
    if not scorer.revision or scorer.revision != C.REVISIONS.get(model_name):
        raise SystemExit(f"{model_name}: loaded revision {scorer.revision!r} is not the pinned {C.REVISIONS.get(model_name)!r}")
    rec = tokenization_checks(scorer, todo, prompt)
    rec["revision"] = scorer.revision
    write_json(d / name, rec)
    if not rec["ok"]:
        raise TokenizationMismatch(f"{scorer.model_tag}: tokenization gate failed ({rec['n_problems']} problems, first: "
                                   f"{rec['problems'][:3]}); see {d / name}. Nothing scored.")


# ---------------- scoring ----------------


def run_mech2_score(cfg: dict, open_scorer: Callable[[str, str], LogoddsScorer] | None = None) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    unverified = [s for s in sources if not verified(s)[0]]
    if problems or unverified:
        write_status(run_id, f"collection not finished or not verified: {problems + unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    tags = list(cfg.get("models", PRIMARY_MODELS))
    entries = {tag: model_entry(tag) for tag in tags}
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n,
           "revisions": {e[0]: C.REVISIONS.get(e[0]) for e in entries.values()},
           "scorer_settings": {tag: e[1] for tag, e in entries.items()}}
    # the item files' hashes are recorded (and checked on a resume) with the run's inputs (amendment after review 2)
    d = _init_run(run_id, cfg, inputs=tuple(runs_root() / s / "transitions.jsonl" for s in sources))
    primary = rq1.rq1_sets(load_items(sources))["primary"]
    sets = battery_items(primary, int(cfg.get("sample_n", SAMPLE_N)))
    shard = shard_items(sets["union"], i, n)
    work = {tag: model_jobs(tag, shard, sets["sample"]) for tag in tags}
    hashes = {t["id"]: _item_hash(t) for t in shard}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    for tag in tags:
        model_name, kw, prompt = entries[tag]
        todo = [(t, a, f) for t, a, f in work[tag] if (tag, t["id"], condition(a, f)) not in done]
        if not todo:
            continue
        scorer = open_scorer(model_name, tag) if open_scorer else LogoddsScorer(model_name, model_tag=tag, **kw)
        try:
            gate(d, scorer, todo, model_name, prompt)
        except BaseException as e:
            write_status(run_id, f"stopped before scoring {tag}: {e}", ok=False)
            raise
        for t, arm, fmt in todo:
            msgs = prompt(arm_messages(t, arm, fmt))
            cond = condition(arm, fmt)
            if fmt == "yn":
                sc = scorer.score_one_token_pair(transition_id=t["id"], condition=cond, messages=msgs, o_plus="Yes", o_minus="No")
            else:
                sc = scorer.score_pair(transition_id=t["id"], condition=cond, messages=msgs,
                                       o_plus=t["candidates"]["o_plus"], o_minus=t["candidates"]["o_minus"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]],
                                                "arm": arm, "format": fmt, "in_sample": t["id"] in sets["sample"],
                                                "step0": t["id"] in sets["step0"]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: " + ", ".join(f"{tag} {len(work[tag])} scores" for tag in tags), ok=False,
                 reported="shard complete")
    return 0


# ---------------- analysis ----------------

# Primary contrasts (amended after review 1 and after review 2, 2026-11-20, before any score): a weighted sum of arms,
# its population and its reference effect (the aligned-minus-swapped effect on the same items and in the same format,
# the denominator of the size ratio r). Eight per model and format, Holm within each at a fixed family size of eight.
_REF = ((1, "aligned"), (-1, "swapped"))
PRIMARY: dict[str, dict[str, Any]] = {
    # review 2: a minimal pair. Both lines are the swapped task plus one sentence about "the {obj}"; only what the
    # sentence asks differs (keep it untouched, or ignore it).
    "P1_keep_vs_ignore": {"terms": ((1, "desire_keep_sw"), (-1, "mention_ignore")), "population": "sample", "reference": _REF},
    # review 2: a minimal pair. The same line but the hope's polarity ("cannot" against "can").
    "P2_hope_polarity": {"terms": ((1, "desire_check"), (-1, "desire_check_can")), "population": "sample", "reference": _REF},
    "P3_mention_without_need": {"terms": ((1, "mention_ignore"), (-1, "swapped")), "population": "sample", "reference": _REF},
    "P4_need_without_name": {"terms": ((1, "described"), (-1, "swapped")), "population": "sample", "reference": _REF},
    # placement items, without those taking from a target-type receptacle and those whose history contradicts "done"
    "P5_done_vs_cancelled": {"terms": ((1, "done"), (-1, "cancelled")), "population": "p5", "reference": _REF},
    "P6_position_step0": {"terms": ((1, "aligned_start"), (-1, "swapped_start"), (-1, "aligned"), (1, "swapped")),
                          "population": "step0", "reference": _REF},
    # the object named and its presence in the action's receptacle denied, take items only: mention against evidence
    "P7_negated_existence": {"terms": ((1, "negated"), (-1, "swapped")), "population": "sample_take", "reference": _REF},
    # review 2: a minimal pair on P7's items. The user's report, the same words but "no" for the article.
    "P8_negated_vs_affirmed": {"terms": ((1, "negated"), (-1, "affirmed")), "population": "sample_take", "reference": _REF},
}
PRIMARY_ARMS = tuple(a for a in ARMS if any(a == arm for spec in PRIMARY.values()
                                            for _, arm in spec["terms"] + spec["reference"]))
P6_ARMS = tuple(arm for _, arm in PRIMARY["P6_position_step0"]["terms"])  # read on every step-0 item, by every model
ALPHA = 0.05
DROPPED_DRAWS_MAX = 0.02  # r is not estimable when more than 2% of its bootstrap draws are undefined (as in families)

# Size zones of r (amendment after review 1). Every prediction is a set of zones; an observed r is the run of zones its
# 95% interval touches. "≈0" is the band Z, whatever the p-value: an effect whose interval lies inside ±0.25 counts as
# ≈0 even when it is significant (TRIVIAL), so it never contradicts a ≈0 prediction.
TRIVIAL_BAND = 0.25
LARGE = 0.5
ZONES = ("N", "Z", "S", "L")  # N: r < -0.25; Z: |r| <= 0.25; S: 0.25 < r < 0.5; L: r >= 0.5
POSITIVE = frozenset({"S", "L"})


def zone(x: float) -> str:
    if x < -TRIVIAL_BAND:
        return "N"
    if x <= TRIVIAL_BAND:
        return "Z"
    return "S" if x < LARGE else "L"


def zones_of(lo: float, hi: float) -> tuple[str, ...]:
    """The zones an interval [lo, hi] touches, in order."""
    a, b = ZONES.index(zone(lo)), ZONES.index(zone(hi))
    return ZONES[min(a, b):max(a, b) + 1]


# Predicted zones of each primary contrast under each account (the design note's prediction table, as amended after
# review 2, section 12, and after review 3, section 13: H2 on P3 widened to Z, S or L). H1 makes no prediction here (it is read from the knowledge split only). H3 is referent
# salience (the object made salient, by name or description); H6_lexical is word overlap between the task line and the
# action string (both formats); H6_position reads P6 alone, and only beyond the band. One line per account.
_P = POSITIVE
_N, _Z, _NZ, _ZS = frozenset({"N"}), frozenset({"Z"}), frozenset({"N", "Z"}), frozenset({"Z", "S"})
_ZSL = frozenset({"Z", "S", "L"})
PREDICTIONS: dict[str, dict[str, frozenset[str]]] = {
    "H2_evidence": {"P1_keep_vs_ignore": _ZSL, "P2_hope_polarity": _Z, "P3_mention_without_need": _ZSL, "P4_need_without_name": _P, "P5_done_vs_cancelled": _N, "P7_negated_existence": _NZ, "P8_negated_vs_affirmed": _N},
    "H3_salience": {"P1_keep_vs_ignore": _ZS, "P2_hope_polarity": _Z, "P3_mention_without_need": _P, "P4_need_without_name": _P, "P5_done_vs_cancelled": _Z, "P7_negated_existence": _P, "P8_negated_vs_affirmed": _Z},
    "H4_wishful": {"P1_keep_vs_ignore": _N, "P2_hope_polarity": _N, "P3_mention_without_need": _NZ, "P4_need_without_name": _P, "P5_done_vs_cancelled": _NZ, "P7_negated_existence": _Z, "P8_negated_vs_affirmed": _Z},
    "H5_policy_blend": {"P1_keep_vs_ignore": _NZ, "P2_hope_polarity": _Z, "P3_mention_without_need": _NZ, "P4_need_without_name": _P, "P5_done_vs_cancelled": _Z, "P7_negated_existence": _Z, "P8_negated_vs_affirmed": _Z},
    "H6_lexical": {"P1_keep_vs_ignore": _ZS, "P2_hope_polarity": _Z, "P3_mention_without_need": _P, "P4_need_without_name": _Z, "P5_done_vs_cancelled": _Z, "P7_negated_existence": _P, "P8_negated_vs_affirmed": _Z},
    "H6_position": {"P6_position_step0": frozenset({"N", "S", "L"})},
}
# A contrast only one account predicts gets the plain rival reading "no effect" (P6: position does not matter), so an
# account can be called consistent only where a rival reading fails (amendment after review 2).
RIVAL_READINGS: dict[str, dict[str, frozenset[str]]] = {"no_position_effect": {"P6_position_step0": _Z}}
CONTENT_ACCOUNTS = ("H2_evidence", "H3_salience", "H4_wishful", "H5_policy_blend", "H6_lexical")

# Amendment after review 3 (2026-11-21, before any score): named secondary readouts with predictions written now, read
# descriptively (zones of r's unadjusted 95% interval against each account's allowed zones, per format), never part of
# the Holm family or the verdicts. H7_check (the task directs which part of the state the model checks) is an
# exploratory account read only here; `negation_artefact` (the success reading drops at "no", whatever it denies) is
# the artefact the placebo pair tests. An account absent from a row makes no prediction there.
SECONDARY_ACCOUNTS = CONTENT_ACCOUNTS + ("H7_check", "negation_artefact")
SECONDARY_PREDICTIONS: dict[str, dict[str, frozenset[str]]] = {
    # both bare desire lines (no household task in either, so the generic lift cancels): a second wishful-against-
    # blending separator next to P2 (blending positive, wishful about zero)
    "S1_desire_check_vs_keep": {"H2_evidence": _NZ, "H3_salience": _Z, "H4_wishful": _Z, "H5_policy_blend": _P,
                                "H6_lexical": _ZS, "H7_check": _NZ, "negation_artefact": _Z},
    # the user's report about the swapped object, denied against affirmed, on P7's items
    "S2_negation_placebo": {"H2_evidence": _Z, "H3_salience": _Z, "H4_wishful": _Z, "H5_policy_blend": _Z,
                            "H6_lexical": _Z, "H7_check": _Z, "negation_artefact": _N},
    # P8 less the placebo difference: the claim about the action's object, net of any reaction to "no" itself
    "S3_P8_net_of_placebo": {"H2_evidence": _N, "H3_salience": _Z, "H4_wishful": _Z, "H5_policy_blend": _Z,
                             "H6_lexical": _Z, "negation_artefact": _Z},
    # H7_check (i): on P7's items, the effect where the history shows the receptacle without the object minus where it
    # does not show the receptacle (confounded with the failure reason; also read within contents_or_open_state)
    "H7a_desire_check_shown_vs_not": {"H2_evidence": _Z, "H3_salience": _Z, "H4_wishful": _Z, "H5_policy_blend": _Z,
                                      "H6_lexical": _Z, "H7_check": _N},
    "H7a_aligned_shown_vs_not": {"H2_evidence": _N, "H3_salience": _Z, "H4_wishful": _Z, "H5_policy_blend": _NZ,
                                 "H6_lexical": _Z, "H7_check": _NZ},
    # H7_check (ii): desire_check - swapped on items whose deciding fact the model knows, minus the others (mech_score)
    "H7b_desire_check_known_vs_unknown": {"H2_evidence": _Z, "H3_salience": _Z, "H4_wishful": _Z,
                                          "H5_policy_blend": _Z, "H6_lexical": _Z, "H7_check": _N},
    # H7_check (iii): the aligned effect where the action would complete the task (the goal object moved into a
    # receptacle of the target type) minus where it would not (TextCraft's goal role)
    "H7c_aligned_completing_vs_other": {"H2_evidence": _Z, "H3_salience": _Z, "H4_wishful": _ZS, "H5_policy_blend": _ZSL,
                                        "H6_lexical": _ZSL, "H7_check": _N},
}


def _lookup(scores: list[dict]) -> dict[tuple[str, str], dict]:
    return {(x["transition_id"], x["condition"]): x for x in scores if x.get("L") == x.get("L")}


def _value(by: dict, t: dict, arm: str, fmt: str, field: str = "L") -> float | None:
    """An item's score for an arm and format (None when it has none)."""
    rec = by.get((t["id"], condition(arm, fmt)))
    if rec is None:
        return None
    v = sigmoid(rec["L"]) if field == "s" else rec.get(field)
    return v if v is not None and v == v else None


def _combo(by: dict, t: dict, terms, fmt: str, field: str = "L") -> float | None:
    vals = [_value(by, t, arm, fmt, field) for _, arm in terms]
    if any(v is None for v in vals):
        return None
    return sum(c * v for (c, _), v in zip(terms, vals))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _ratio(xs: list[tuple[float, float]]) -> float:
    """Mean contrast over mean reference effect; undefined unless the reference effect is above zero."""
    den = _mean([r for _, r in xs])
    return _mean([d for d, _ in xs]) / den if den > 0 else float("nan")


def contrast(by: dict, items: list[dict], terms, fmt: str = "cand", field: str = "L", reference=None,
             B: int = C.BOOTSTRAP_RESAMPLES, perms: int = C.PERMUTATIONS) -> dict[str, Any]:
    """Mean of a within-item contrast with a game-clustered 95% CI and sign-flip permutation p; with `reference`,
    the size ratio r = mean contrast / mean reference effect on the same items (the reference read in the same format),
    its interval from the same resamples, and how many resamples left r undefined (reference mean at or below 0)."""
    rows = []
    for t in items:
        d = _combo(by, t, terms, fmt, field)
        ref = _combo(by, t, reference, fmt, field) if reference else 0.0
        if d is not None and ref is not None:
            rows.append((t["game_id"], d, ref))
    if not rows:
        return {"n_items": 0, "n_games": 0, "estimate": float("nan"), "ci": [float("nan")] * 2, "p": float("nan")}
    games = [g for g, _, _ in rows]
    boot = cluster_bootstrap(lambda xs: _mean([d for d, _ in xs]), [(d, r) for _, d, r in rows], games, B=B, seed=C.RNG_SEED)
    out = {"n_items": len(rows), "n_games": len(set(games)), "estimate": boot["estimate"],
           "ci": [boot["ci_low"], boot["ci_high"]],
           "p": pair_permutation([d for _, d, _ in rows], n=perms, seed=C.RNG_SEED, cluster_ids=games)}
    if reference:
        rb = cluster_bootstrap(_ratio, [(d, r) for _, d, r in rows], games, B=B, seed=C.RNG_SEED)
        dropped = rb.get("n_dropped", 0)
        out["reference_effect"] = _mean([r for _, _, r in rows])
        out["ratio"] = {"estimate": rb["estimate"], "ci": [rb["ci_low"], rb["ci_high"]], "n_resamples": B,
                        "n_dropped": dropped, "estimable": bool(out["reference_effect"] > 0 and dropped <= DROPPED_DRAWS_MAX * B)}
    return out


def item_share(by: dict, items: list[dict], a: str, b: str, fmt: str = "yn", B: int = C.BOOTSTRAP_RESAMPLES,
               perms: int = C.PERMUTATIONS) -> dict[str, Any]:
    """The share of items whose log-odds are higher under arm `a` than under `b` (ties count against), with a
    game-clustered CI and the sign-flip p of the item signs against one half (the families' cross-model metric)."""
    rows = [(t["game_id"], d) for t in items for d in [_combo(by, t, ((1, a), (-1, b)), fmt)] if d is not None]
    if not rows:
        return {"n_items": 0, "estimate": float("nan"), "ci": [float("nan")] * 2, "p": float("nan")}
    games = [g for g, _ in rows]
    boot = cluster_bootstrap(lambda xs: sum(x > 0 for x in xs) / len(xs), [d for _, d in rows], games, B=B, seed=C.RNG_SEED)
    signs = [(d > 0) - (d < 0) for _, d in rows]
    return {"n_items": len(rows), "estimate": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]],
            "p": pair_permutation(signs, n=perms, seed=C.RNG_SEED, cluster_ids=games)}


def group_difference(by: dict, a_items: list[dict], b_items: list[dict], terms, fmt: str = "cand", reference=_REF,
                     B: int = C.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """The mean of a within-item contrast on items `a_items` minus its mean on `b_items` (a difference in differences),
    with a game-clustered 95% CI, and as r over the reference effect on both groups together (amendment after review 3,
    H7_check). A resample that lacks either group, or whose reference mean is at or below zero, is undefined."""
    rows = []
    for grp, ts in ((True, a_items), (False, b_items)):
        for t in ts:
            d, ref = _combo(by, t, terms, fmt), _combo(by, t, reference, fmt)
            if d is not None and ref is not None:
                rows.append((t["game_id"], (grp, d, ref)))
    na, nb = sum(g for _, (g, _, _) in rows), sum(not g for _, (g, _, _) in rows)
    if not na or not nb:
        return {"n_a": na, "n_b": nb, "estimate": float("nan"), "ci": [float("nan")] * 2}

    def diff(xs):
        a, b = [d for g, d, _ in xs if g], [d for g, d, _ in xs if not g]
        return _mean(a) - _mean(b) if a and b else float("nan")

    def ratio(xs):
        den = _mean([r for _, _, r in xs])
        return diff(xs) / den if den > 0 else float("nan")

    xs, games = [x for _, x in rows], [g for g, _ in rows]
    boot = cluster_bootstrap(diff, xs, games, B=B, seed=C.RNG_SEED)
    rb = cluster_bootstrap(ratio, xs, games, B=B, seed=C.RNG_SEED)
    ref_mean = _mean([r for _, _, r in xs])
    return {"n_a": na, "n_b": nb, "n_games": len(set(games)), "estimate": boot["estimate"],
            "ci": [boot["ci_low"], boot["ci_high"]], "reference_effect": ref_mean,
            "ratio": {"estimate": rb["estimate"], "ci": [rb["ci_low"], rb["ci_high"]], "n_resamples": B,
                      "n_dropped": rb.get("n_dropped", 0),
                      "estimable": bool(ref_mean > 0 and rb.get("n_dropped", 0) <= DROPPED_DRAWS_MAX * B)}}


def secondary_zones(res: dict[str, Any] | None) -> tuple[str, ...] | None:
    """The zones a secondary readout's r interval touches (descriptive: no p-value rule), None when r is not
    estimable or its interval is undefined."""
    r = (res or {}).get("ratio") or {}
    lo, hi = (r.get("ci") or [float("nan")] * 2)
    if not r.get("estimable") or lo != lo or hi != hi:
        return None
    return zones_of(lo, hi)


def read_secondary(name: str, by_format: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A named secondary readout against SECONDARY_PREDICTIONS: its zones per format, and per account the reading in
    each format and jointly (decided only when the formats agree, as for the primary contrasts)."""
    zs = {f: secondary_zones(res) for f, res in by_format.items()}
    per = {h: {f: reading(allowed, z) for f, z in zs.items()} for h, allowed in SECONDARY_PREDICTIONS[name].items()}
    if set(zs) == set(FORMATS):
        for h in per:
            per[h]["joint"] = _joint(per[h]["cand"], per[h]["yn"])
    return {"zones": {f: (list(z) if z else None) for f, z in zs.items()}, "readings": per}


_LABELS = {("N",): "NEG", ("N", "Z"): "NEG_OR_TRIVIAL", ("Z", "S"): "TRIVIAL_OR_SMALL", ("S",): "POS_SMALL",
           ("S", "L"): "POS", ("L",): "POS_LARGE"}


def categorize(res: dict[str, Any], holm_p: float) -> tuple[str, tuple[str, ...] | None]:
    """(label, zones) of a contrast, by the rules in the design note (amendment after review 1).

    The zones are the run of size zones r's 95% interval touches. They are not read (INCONCLUSIVE, zones None) when
    the estimate or r's interval is undefined, when r is not estimable (more than 2% of draws undefined), or when the
    interval lies wholly outside the band (a claim of a non-trivial effect) but Holm's p is at least 0.05. An interval
    inside the band is NULL when Holm's p is at least 0.05 and TRIVIAL when it is below: both count as ≈0."""
    est, r = res.get("estimate", float("nan")), res.get("ratio") or {}
    lo, hi = (r.get("ci") or [float("nan")] * 2)
    if est != est or lo != lo or hi != hi or not r.get("estimable", False):
        return "INCONCLUSIVE", None
    zs = zones_of(lo, hi)
    significant = holm_p == holm_p and holm_p < ALPHA
    if "Z" not in zs and not significant:
        return "INCONCLUSIVE", None
    if zs == ("Z",):
        return ("TRIVIAL" if significant else "NULL"), zs
    return _LABELS.get(zs, "WIDE"), zs


def reading(predicted: frozenset[str], zs: tuple[str, ...] | None) -> str:
    """consistent (the interval's zones all predicted), contradicted (none predicted) or undecided."""
    if zs is None:
        return "undecided"
    if set(zs) <= predicted:
        return "consistent"
    if not set(zs) & predicted:
        return "contradicted"
    return "undecided"


def _verdict(h: str, rows: dict[str, dict[str, str]]) -> str:
    """An account's verdict from the readings of every account and rival reading (`rows`, one format or joint).

    contradicted: at least one contrast contradicted. consistent (amendment after review 2, tightened after review 3):
    none contradicted, and consistent on at least one contrast where some rival account (or, for a contrast only this
    account predicts, the rival reading "no effect") is contradicted AND whose allowed zones exclude Z (a risky
    prediction: an account that allows about zero cannot fail when nothing happens). untested: anything else,
    including an account consistent only where every rival is consistent or undecided too, or only where it allows
    zero (`untested_reason`)."""
    mine = rows[h]
    if "contradicted" in mine.values():
        return "contradicted"
    return "consistent" if _risky(h, rows) else "untested"


def _allowed(h: str, c: str) -> frozenset[str]:
    return {**PREDICTIONS, **RIVAL_READINGS}[h][c]


def _discriminating(h: str, rows: dict[str, dict[str, str]]) -> list[str]:
    """The contrasts where account `h` is consistent and some rival reading is contradicted."""
    return [c for c, v in rows[h].items() if v == "consistent"
            and any(rows[o].get(c) == "contradicted" for o in rows if o != h)]


def _risky(h: str, rows: dict[str, dict[str, str]]) -> list[str]:
    """The discriminating contrasts on which `h` made a risky prediction (its allowed zones exclude Z; review 3)."""
    return [c for c in _discriminating(h, rows) if "Z" not in _allowed(h, c)]


def _account_rows(per_account: dict[str, dict[str, str]]) -> dict[str, Any]:
    out = {}
    for h in PREDICTIONS:
        vals = list(per_account[h].values())
        disc, risky = _discriminating(h, per_account), _risky(h, per_account)
        verdict = _verdict(h, per_account)
        out[h] = {"per_contrast": per_account[h], "n_consistent": vals.count("consistent"),
                  "n_contradicted": vals.count("contradicted"), "n_undecided": vals.count("undecided"),
                  "consistent_where_a_rival_fails": disc, "risky_where_a_rival_fails": risky,
                  "untested_reason": (None if verdict != "untested" else
                                      "consistent only where it allows zero" if disc else
                                      "no contrast decided where a rival fails"),
                  "verdict": verdict}
    out["rivals"] = {r: per_account[r] for r in RIVAL_READINGS}
    return out


def _short(cs) -> str:
    return ", ".join(c.split("_")[0] for c in cs)


def _label(h: str, row: dict[str, Any]) -> str:
    """An account with the contrasts its verdict rests on (amendment after review 3): for a consistent account the
    risky contrasts where a rival fails, and those where it allows zero; for an untested one the contrasts where it was
    consistent only while allowing zero; for a contradicted one the contrasts that contradict it."""
    disc, risky = row["consistent_where_a_rival_fails"], row["risky_where_a_rival_fails"]
    if row["verdict"] == "consistent":
        rest = [c for c in disc if c not in risky]
        return f"{h} [risky: {_short(risky)}" + (f"; allows zero: {_short(rest)}]" if rest else "]")
    if row["verdict"] == "untested":
        return f"{h} [consistent only where it allows zero: {_short(disc)}]" if disc else h
    return f"{h} [{_short([c for c, v in row['per_contrast'].items() if v == 'contradicted'])}]"


# The contrasts that add an explicit claim to a line (done; the user's report): a model that reads claims contradicts
# a content-blind salience account on them, whether or not the goal effect itself comes from evidence (review 3).
CLAIM_CONTRASTS = ("P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed")
# The mention component on P7's items: the mean of the two reports (denied, affirmed) minus the swapped task. With the
# claim balanced, what remains is the effect of mentioning the object and the receptacle (review 3). It assumes the
# denial lowers L about as much as the affirmation raises it; task as evidence's own fingerprint gives about +0.1.
MENTION_TERMS = ((0.5, "negated"), (0.5, "affirmed"), (-1, "swapped"))


def _mention_text(mention: dict[str, Any]) -> str:
    return "; ".join(f"{m['ratio']['estimate']:.2f} [{m['ratio']['ci'][0]:.2f}, {m['ratio']['ci'][1]:.2f}] {name}"
                     for f, name in (("cand", "two-candidate"), ("yn", "yes/no")) for m in [mention[f]])


def _mention_above_band(mention: dict[str, Any] | None) -> bool:
    """M's r interval lies wholly above the band in both formats (and r is estimable in both)."""
    if not mention:
        return False
    for f in FORMATS:
        r = (mention.get(f) or {}).get("ratio") or {}
        lo = (r.get("ci") or [float("nan")])[0]
        if not r.get("estimable") or lo != lo or lo <= TRIVIAL_BAND:
            return False
    return True


def readout(observed: dict[str, tuple[str, ...] | None]) -> dict[str, Any]:
    """For each account, each contrast it predicts as consistent, contradicted or undecided against the prediction
    table (one format), and the account's verdict; `rivals` holds the rival readings ("no position effect")."""
    rows = {h: {c: reading(allowed, observed.get(c)) for c, allowed in preds.items()}
            for h, preds in {**PREDICTIONS, **RIVAL_READINGS}.items()}
    return _account_rows(rows)


def _joint(a: str, b: str) -> str:
    """One contrast's joint reading from the two formats' readings: decided only when both agree."""
    if a == b and a in ("consistent", "contradicted"):
        return a
    if {a, b} == {"consistent", "contradicted"}:
        return "format_dependent"
    return "undecided"


def _per_contrast(ro: dict[str, Any], h: str) -> dict[str, str]:
    return ro["rivals"][h] if h in RIVAL_READINGS else ro[h]["per_contrast"]


def joint_readout(cand: dict[str, Any], yn: dict[str, Any], mention: dict[str, Any] | None = None) -> dict[str, Any]:
    """The decision rule across formats (amendment after review 1): a contrast is decided for an account only when the
    two formats agree (both consistent, or both contradicted). One consistent and one contradicted is
    `format_dependent`; anything else (one decided, one undecided) is undecided. Both count as undecided in the verdict,
    and the format-dependent contrasts are listed by name.

    The summary (amendment after review 2) lists the content accounts consistent, untested and contradicted
    separately. A contradicted account cannot be the sole account; it is not rejected, since the accounts are not
    exclusive. When every content account (H2-H5, H6_lexical) is contradicted, the summary says no single account fits
    (a mixture, or an account not on the list) and that the readout is then the estimates of P5, P7 and P8; it never
    picks the least contradicted account.

    After review 3 each account is listed with the contrasts its verdict rests on (`_label`), and two fixed sentences
    are added when salience is contradicted only on contrasts that add an explicit claim (`CLAIM_CONTRASTS`): (a) when
    the mention component M (`mention`, the contrast `MENTION_TERMS` per format) has an r interval wholly above the
    band in both formats, salience is likely part of the effect; (b) when task as evidence is consistent, its win
    there shows that the model reads claims, not that the goal effect comes from evidence."""
    rows = {h: {c: _joint(_per_contrast(cand, h)[c], _per_contrast(yn, h)[c]) for c in preds}
            for h, preds in {**PREDICTIONS, **RIVAL_READINGS}.items()}
    out = _account_rows(rows)
    for h in PREDICTIONS:
        out[h]["format_dependent"] = [c for c, v in rows[h].items() if v == "format_dependent"]
        del out[h]["n_undecided"]
    by = {v: [h for h in CONTENT_ACCOUNTS if out[h]["verdict"] == v] for v in ("consistent", "untested", "contradicted")}
    fitting = by["consistent"] + by["untested"]
    lists = "; ".join(f"{v}{' (cannot be the sole account)' if v == 'contradicted' else ''}: "
                      f"{', '.join(_label(h, out[h]) for h in by[v]) or 'none'}"
                      for v in ("consistent", "untested", "contradicted"))
    if not fitting:
        text = ("no single account; mixture or an account not on the list (each content account is contradicted, so "
                "none can be the sole account); the readout is the estimates and intervals of P5, P7 and P8; " + lists)
    else:
        text = lists
    h3_against = [c for c, v in out["H3_salience"]["per_contrast"].items() if v == "contradicted"]
    claims_only = bool(h3_against) and set(h3_against) <= set(CLAIM_CONTRASTS)
    salience_part = claims_only and _mention_above_band(mention)
    claims_read = claims_only and out["H2_evidence"]["verdict"] == "consistent"
    if salience_part:
        text += (". Salience cannot be the sole account, but a mention component remains with the claim balanced "
                 f"(r = {_mention_text(mention)}): salience is likely part of the effect.")
    if claims_read:
        text += (("" if salience_part else ".") + " H3 is contradicted only on contrasts that add an explicit claim (done; "
                 "the user's report). This shows the model reads such claims, not that the goal effect comes from "
                 "evidence rather than salience; see M.")
    out["summary"] = {
        "content_accounts_consistent": by["consistent"],
        "content_accounts_untested": by["untested"],
        "content_accounts_contradicted": by["contradicted"],
        "content_accounts_not_contradicted": fitting,
        "reading": text,
        "position": out["H6_position"]["verdict"],
        "salience_contradicted_only_on_claims": claims_only,
        "mention_component_above_band_both_formats": _mention_above_band(mention),
        "salience_likely_part": salience_part,
        "claims_read_not_evidence_shown": claims_read,
        "mention_component": ({f: {"estimate": m.get("estimate"), "ci": m.get("ci"), "ratio": m.get("ratio")}
                               for f, m in mention.items()} if mention else None),
    }
    return out


def separations() -> dict[str, dict[str, list[str]]]:
    """For each pair of accounts, the contrasts both predict whose allowed zones are disjoint (`separate`: any decided
    reading contradicts at least one of the two) and those whose allowed zones differ but overlap (`partly`)."""
    out = {}
    names = list(PREDICTIONS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [c for c in PRIMARY if c in PREDICTIONS[a] and c in PREDICTIONS[b]]
            out[f"{a} vs {b}"] = {
                "separate": [c for c in both if not PREDICTIONS[a][c] & PREDICTIONS[b][c]],
                "partly": [c for c in both if PREDICTIONS[a][c] != PREDICTIONS[b][c] and PREDICTIONS[a][c] & PREDICTIONS[b][c]],
                "identical_rows": PREDICTIONS[a] == PREDICTIONS[b]}
    return out


def populations(union: list[dict], sample: set[str], step0: set[str]) -> dict[str, list[dict]]:
    s = [t for t in union if t["id"] in sample]
    return {"sample": s, "sample_placement": [t for t in s if task_type(t["task_line"]) in PLACEMENT_TYPES],
            "p5": [t for t in s if p5_eligible(t)], "sample_take": [t for t in s if p7_eligible(t)],
            "step0": [t for t in union if t["id"] in step0],
            "later": [t for t in s if t.get("history_len", 0) > 0]}


def primary_family(by: dict, pops: dict[str, list[dict]], fmt: str = "cand", B: int = C.BOOTSTRAP_RESAMPLES,
                   perms: int = C.PERMUTATIONS) -> dict[str, Any]:
    """The eight primary contrasts in one format, Holm within the family, each categorised, and the one-format
    readout."""
    res = {name: contrast(by, pops[spec["population"]], spec["terms"], fmt, reference=spec["reference"], B=B, perms=perms)
           for name, spec in PRIMARY.items()}
    # a fixed family of eight (len(PRIMARY)): a missing or undefined p enters as 1, so the family never shrinks
    adj = holm({name: (r["p"] if r["p"] == r["p"] else 1.0) for name, r in res.items()})
    for name, r in res.items():
        r["holm_p"] = adj[name]
        r["category"], zs = categorize(r, adj[name])
        r["zones"] = list(zs) if zs else None
    return {"contrasts": res, "readout": readout({name: (tuple(r["zones"]) if r["zones"] else None)
                                                  for name, r in res.items()})}


def model_block(scores: list[dict], union: list[dict], sample: set[str], step0: set[str],
                B: int = C.BOOTSTRAP_RESAMPLES, perms: int = C.PERMUTATIONS) -> dict[str, Any]:
    by = _lookup(scores)
    pops = populations(union, sample, step0)
    kw = {"B": B, "perms": perms}
    ref = ((1, "aligned"), (-1, "swapped"))
    block: dict[str, Any] = {"primary": primary_family(by, pops, "cand", **kw),
                             "primary_yes_no": primary_family(by, pops, "yn", **kw)}
    # review 3: the mention component M on P7's items, both formats, read in the summary and the mixture readout
    mention = {f: contrast(by, pops["sample_take"], MENTION_TERMS, f, reference=ref, **kw) for f in FORMATS}
    block["readout_both_formats"] = joint_readout(block["primary"]["readout"], block["primary_yes_no"]["readout"], mention)
    arms = {}
    for arm in CONTENT_ARMS:
        if arm == "swapped":
            continue
        arms[arm] = {
            "vs_swapped_L": contrast(by, pops["sample"], ((1, arm), (-1, "swapped")), "cand", reference=ref, **kw),
            "vs_swapped_L_first": contrast(by, pops["sample"], ((1, arm), (-1, "swapped")), "cand", field="L_first", **kw),
            "vs_swapped_s_pp": _pp(contrast(by, pops["sample"], ((1, arm), (-1, "swapped")), "cand", field="s", **kw)),
            "vs_neutral_L": contrast(by, pops["sample"], ((1, arm), (-1, "neutral")), "cand", **kw),
            "vs_aligned_L": contrast(by, pops["sample"], ((1, arm), (-1, "aligned")), "cand", **kw),
            "yn_vs_swapped_L": contrast(by, pops["sample"], ((1, arm), (-1, "swapped")), "yn", reference=ref, **kw),
            "yn_vs_swapped_mean_sigma_pp": _pp(contrast(by, pops["sample"], ((1, arm), (-1, "swapped")), "yn", field="s", **kw)),
            "yn_item_share_toward_yes_vs_swapped": item_share(by, pops["sample"], arm, "swapped", **kw),
            "by_step": {k: contrast(by, pops[k], ((1, arm), (-1, "swapped")), "cand", reference=ref, **kw)
                        for k in ("step0", "later")},
        }
    block["arms_on_sample"] = arms
    placement_rest = [t for t in pops["sample_placement"] if not p5_eligible(t)]
    both = ("cand", "yn")
    not_shown = [t for t in pops["sample_take"] if negated_status(t)[1] == "contents not shown"]
    block["secondary"] = {
        # review 2: the first P1 and P2, kept as secondary readouts (not minimal pairs: the bare desire lines lack the
        # swapped task that their comparators keep), and the size of that gap for the keep line
        "old_P1_desire_keep_vs_ignore": {f: contrast(by, pops["sample"], ((1, "desire_keep"), (-1, "mention_ignore")), f, reference=ref, **kw) for f in both},
        "old_P2_desire_check_vs_swapped": {f: contrast(by, pops["sample"], ((1, "desire_check"), (-1, "swapped")), f, reference=ref, **kw) for f in both},
        "desire_keep_alone_vs_after_swapped": {f: contrast(by, pops["sample"], ((1, "desire_keep"), (-1, "desire_keep_sw")), f, reference=ref, **kw) for f in both},
        "desire_keep_vs_swapped": arms["desire_keep"]["vs_swapped_L"],
        "desire_keep_vs_swapped_yes_no": arms["desire_keep"]["yn_vs_swapped_L"],
        "negated_vs_exists_sentence_take": contrast(by, pops["sample_take"], ((1, "negated"), (-1, "mention_exists")), "cand", reference=ref, **kw),
        # P7 by what the history shows of the receptacle (the claim established, not shown, or contradicted)
        "negated_by_history": {k: contrast(by, [t for t in pops["sample"] if is_take(t) and negated_status(t)[1] == k],
                                           ((1, "negated"), (-1, "swapped")), "cand", reference=ref, **kw)
                               for k in ("shown, without it", "contents not shown", "shown, holds it")},
        # review 2: P7 and P8 on the take items whose history does not show the receptacle's contents (where the
        # user's report is the only information about them), both formats, next to the full population
        "P7_P8_contents_not_shown": {name: {f: contrast(by, not_shown, PRIMARY[name]["terms"], f, reference=ref, **kw) for f in both}
                                     for name in ("P7_negated_existence", "P8_negated_vs_affirmed")},
        "cancelled_vs_swapped": arms["cancelled"]["vs_swapped_L"],
        "done_vs_swapped": arms["done"]["vs_swapped_L"],
        # review 2: P5 read against cancelled's own lift over swapped on P5's items (the ratio is undefined when that
        # lift is at or below zero), and the lift itself
        "P5_against_cancelled_lift": {f: contrast(by, pops["p5"], PRIMARY["P5_done_vs_cancelled"]["terms"], f,
                                                  reference=((1, "cancelled"), (-1, "swapped")), **kw) for f in both},
        "cancelled_lift_on_p5_items": {f: contrast(by, pops["p5"], ((1, "cancelled"), (-1, "swapped")), f, reference=ref, **kw) for f in both},
        "ignore_vs_exists_sentence": contrast(by, pops["sample"], ((1, "mention_ignore"), (-1, "mention_exists")), "cand", **kw),
        # P5 on the placement items it leaves out: taking from a target-type receptacle (H2 predicts done ABOVE
        # cancelled there) and items whose history contradicts "done"; and on every placement item, as first designed
        "done_vs_cancelled_take_from_target_type": contrast(
            by, [t for t in pops["sample_placement"] if takes_from_target_type(t)], ((1, "done"), (-1, "cancelled")), "cand", reference=ref, **kw),
        "done_vs_cancelled_contradicted": contrast(
            by, [t for t in pops["sample_placement"] if done_contradicted(t)], ((1, "done"), (-1, "cancelled")), "cand", reference=ref, **kw),
        "done_vs_cancelled_all_placement": contrast(by, pops["sample_placement"], ((1, "done"), (-1, "cancelled")), "cand", reference=ref, **kw),
        "n_p5_left_out": {"take_from_target_type": sum(takes_from_target_type(t) for t in placement_rest),
                          "done_contradicted": sum(done_contradicted(t) for t in placement_rest),
                          "both": sum(takes_from_target_type(t) and done_contradicted(t) for t in placement_rest)},
        "n_done_contradicted_in_sample": sum(done_contradicted(t) for t in pops["sample"]),
        # review 2: position arms at step 0 only. P6 in the yes/no format and its two parts: moving the aligned line and
        # moving the swapped line to the top
        "position": {
            "step0_start_vs_usual_yes_no": contrast(by, pops["step0"], PRIMARY["P6_position_step0"]["terms"], "yn", reference=ref, **kw),
            "step0_aligned_start_vs_aligned": contrast(by, pops["step0"], ((1, "aligned_start"), (-1, "aligned")), "cand", **kw),
            "step0_swapped_start_vs_swapped": contrast(by, pops["step0"], ((1, "swapped_start"), (-1, "swapped")), "cand", **kw),
        },
    }
    # review 2: the first-token term of P7 and P8 (two-candidate). It does not enter any category; where its sign
    # differs from the full L's (rq1d's split: a denial lowered Qwen's full L but raised its first token), the report
    # names the split and P7's two-candidate reading rests on the continuation, not the first token.
    first = {}
    for name in ("P7_negated_existence", "P8_negated_vs_affirmed"):
        ft = contrast(by, pops["sample_take"], PRIMARY[name]["terms"], "cand", field="L_first", **kw)
        full = block["primary"]["contrasts"][name]["estimate"]
        ft["sign_split_with_full_L"] = bool(ft["estimate"] == ft["estimate"] and full == full and ft["estimate"] * full < 0)
        first[name] = ft
    block["secondary"]["first_token_P7_P8"] = first
    # review 2: under a mixture the readout is continuous: the point estimates and intervals of P5, P7 and P8 (and r),
    # in both formats, reported whatever the verdicts
    block["mixture_readout"] = {
        name: {fam: {k: block[key]["contrasts"][name].get(k) for k in ("estimate", "ci", "ratio", "category")}
               for fam, key in (("cand", "primary"), ("yn", "primary_yes_no"))}
        for name in ("P5_done_vs_cancelled", "P7_negated_existence", "P8_negated_vs_affirmed")}
    # review 3: M = (negated + affirmed)/2 - swapped on P7's items, the mention with the claim balanced; its zones are
    # read from r's unadjusted interval (it is not in the Holm family)
    block["mixture_readout"]["M_mention_component"] = {
        f: {**{k: mention[f].get(k) for k in ("n_items", "estimate", "ci", "ratio", "p")}, "zones": secondary_zones(mention[f])}
        for f in FORMATS}
    # review 3: named secondary readouts with predictions written before any score (SECONDARY_PREDICTIONS), outside
    # the Holm family and the verdicts
    take, shown = pops["sample_take"], [t for t in pops["sample_take"] if negated_status(t)[1] == "shown, without it"]
    shown_ids = {t["id"] for t in shown}
    within = [t for t in take if t.get("determination_reason") == "contents_or_open_state"]
    completing = [t for t in pops["sample"] if completes_task(t)]
    other = [t for t in pops["sample"] if not completes_task(t)]
    named = {
        "S1_desire_check_vs_keep": {f: contrast(by, pops["sample"], ((1, "desire_check"), (-1, "desire_keep")), f, reference=ref, **kw) for f in FORMATS},
        "S2_negation_placebo": {f: contrast(by, take, ((1, "placebo_negated"), (-1, "placebo_affirmed")), f, reference=ref, **kw) for f in FORMATS},
        "S3_P8_net_of_placebo": {f: contrast(by, take, ((1, "negated"), (-1, "affirmed"), (-1, "placebo_negated"), (1, "placebo_affirmed")),
                                             f, reference=ref, **kw) for f in FORMATS},
        "H7a_desire_check_shown_vs_not": {f: group_difference(by, shown, not_shown, ((1, "desire_check"), (-1, "swapped")), f, B=B) for f in FORMATS},
        "H7a_aligned_shown_vs_not": {f: group_difference(by, shown, not_shown, ref, f, B=B) for f in FORMATS},
        "H7c_aligned_completing_vs_other": {f: group_difference(by, completing, other, ref, f, B=B) for f in FORMATS},
    }
    block["secondary_named"] = {name: {"by_format": res, **read_secondary(name, res)} for name, res in named.items()}
    # H7_check (i) within the one failure reason where both groups are well populated (the split is confounded with
    # the reason: most shown-without items fail on contents or open state, most not-shown items on distance)
    block["secondary_named"]["H7a_within_contents_or_open_state"] = {
        name: {f: group_difference(by, [t for t in within if t["id"] in shown_ids], [t for t in within if t["id"] not in shown_ids],
                                   terms, f, B=B)
               for f in FORMATS}
        for name, terms in (("desire_check", ((1, "desire_check"), (-1, "swapped"))), ("aligned", ref))}
    block["secondary_named"]["H7a_groups_by_reason"] = {
        "shown, without it": dict(Counter(t.get("determination_reason") for t in shown)),
        "contents not shown": dict(Counter(t.get("determination_reason") for t in not_shown))}
    content = {name: spec for name, spec in PRIMARY.items() if spec["population"] != "step0"}

    def split(subsets: dict[str, list[dict]]) -> dict[str, Any]:
        """The content contrasts (all but P6) on subsets of items, each restricted as its population is."""
        return {k: {name: contrast(by, _restrict(spec, ts), spec["terms"], "cand", reference=spec["reference"], **kw)
                    for name, spec in content.items()} for k, ts in subsets.items()}

    by_reason = defaultdict(list)
    for t in pops["sample"]:
        by_reason[t.get("determination_reason") or "unknown"].append(t)
    block["secondary"]["primary_by_step"] = split({"step0": pops["step0"], "later": pops["later"]})
    block["secondary"]["primary_by_action"] = split({
        "take": [t for t in pops["sample"] if t["action"].startswith("take ")],
        "not_take": [t for t in pops["sample"] if not t["action"].startswith("take ")]})
    block["secondary"]["primary_by_reason"] = split(dict(sorted(by_reason.items())))
    return block


def _restrict(spec: dict[str, Any], items: list[dict]) -> list[dict]:
    """Items of a subset that belong to a contrast's population type (P5: its eligible placement items; P7: take items)."""
    if spec["population"] == "p5":
        return [t for t in items if p5_eligible(t)]
    if spec["population"] == "sample_take":
        return [t for t in items if p7_eligible(t)]
    return items


def _pp(res: dict) -> dict:
    """A contrast of two-candidate scores (or yes/no σ), in percentage points."""
    out = dict(res)
    out["estimate"] = res["estimate"] * 100
    out["ci"] = [x * 100 for x in res["ci"]]
    return out


# ---------------- analysis-only checks on earlier runs ----------------


def fewshot_overlap(scores: list[dict], primary: list[dict], B: int = C.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """H6's few-shot overlap check on rq1_score, which scored every primary item with example sets 1, 2 and 3.

    For set k and item i, d_ik = ΔL under set k minus the mean ΔL under the other two sets (the same item and run).
    Items whose real task line equals a task line of set k's examples ("aligned overlap"), or whose swapped line does
    ("swapped overlap"), are compared with all other items' d_ik: H6 predicts a positive difference for aligned overlap
    and a negative one for swapped overlap. Few games carry each overlap, so this is descriptive."""
    by = {(x["transition_id"], x["condition"]): x["L"] for x in scores if x.get("L") == x.get("L")}
    arm = {1: ("aligned", "swapped"), 2: ("aligned_set2", "swapped_set2"), 3: ("aligned_set3", "swapped_set3")}
    lines = {k: {ex["history"].split("\n", 1)[0].strip() for ex in shots} for k, shots in WM_FEWSHOT_SETS.items()}
    out: dict[str, Any] = {}
    for k in (1, 2, 3):
        rows = {"aligned": [], "swapped": [], "none": []}
        for t in primary:
            dl = {j: by[(t["id"], a)] - by[(t["id"], s)] for j, (a, s) in arm.items()
                  if (t["id"], a) in by and (t["id"], s) in by}
            if len(dl) != 3:
                continue
            d = dl[k] - _mean([dl[j] for j in dl if j != k])
            kind = ("aligned" if t["task_line"].strip() in lines[k] else
                    "swapped" if (t.get("swapped_task_line") or "").strip() in lines[k] else "none")
            rows[kind].append((t["game_id"], d, dl[k]))
        base = _mean([d for _, d, _ in rows["none"]])
        res = {"n_none": len(rows["none"]), "mean_d_none": base,
               "mean_delta_L_none": _mean([x for _, _, x in rows["none"]])}
        for kind in ("aligned", "swapped"):
            r = rows[kind]
            boot = cluster_bootstrap(lambda xs: _mean(xs) - base, [d for _, d, _ in r], [g for g, _, _ in r], B=B, seed=C.RNG_SEED)
            res[f"{kind}_overlap"] = {"n_items": len(r), "n_games": len({g for g, _, _ in r}),
                                      # the plain comparison: this set's ΔL on these items, against the other items
                                      "mean_delta_L": _mean([x for _, _, x in r]),
                                      "mean_d": _mean([d for _, d, _ in r]),
                                      "difference_vs_none": boot["estimate"], "ci_games": [boot["ci_low"], boot["ci_high"]]}
        out[f"set{k}"] = res
    return out


def knowledge_split(scores: list[dict], mech_scores: list[dict], items: list[dict], B: int = C.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """H1's readout: each primary content contrast (and the aligned effect) on items whose state question and its
    twin the same model answers correctly under the neutral task in mech_score, as a share of its effect on all items
    with a verdict. H1 predicts shares below 0.75 for positive contrasts.

    Amendment after review 3 (H7_check (ii)): every entry also gives its mean on known items minus its mean on the
    other items (`known_minus_unknown`, in nats and as r over the aligned effect on all these items), and desire_check
    - swapped is added; its difference is read against SECONDARY_PREDICTIONS ("H7b_desire_check_known_vs_unknown")."""
    by = _lookup(scores)
    mb = {(x["transition_id"], x["condition"]): x["L"] for x in mech_scores if x.get("L") == x.get("L")}

    def known(t):
        a, b = mb.get((t["id"], "state_absent")), mb.get((t["id"], "statetwin_absent"))
        return None if a is None or b is None else a < 0 < b

    out = {}
    specs = {"aligned_effect": _REF,
             **{n: s["terms"] for n, s in PRIMARY.items() if s["population"] != "step0"},
             "desire_check_vs_swapped": ((1, "desire_check"), (-1, "swapped"))}
    for name, terms in specs.items():
        rows = [(t["game_id"], (d, k, ref)) for t in items for d in [_combo(by, t, terms, "cand")] for k in [known(t)]
                for ref in [_combo(by, t, _REF, "cand")] if d is not None and k is not None and ref is not None]
        if not rows:
            out[name] = None
            continue

        def share(xs):
            kn = [d for d, k, _ in xs if k]
            allm = _mean([d for d, _, _ in xs])
            return _mean(kn) / allm if kn and allm == allm and allm != 0 else float("nan")

        def kmu(xs):
            kn, un = [d for d, k, _ in xs if k], [d for d, k, _ in xs if not k]
            return _mean(kn) - _mean(un) if kn and un else float("nan")

        def kmu_r(xs):
            den = _mean([r for _, _, r in xs])
            return kmu(xs) / den if den > 0 else float("nan")

        xs, games = [x for _, x in rows], [g for g, _ in rows]
        boot = cluster_bootstrap(share, xs, games, B=B, seed=C.RNG_SEED)
        diff = cluster_bootstrap(kmu, xs, games, B=B, seed=C.RNG_SEED)
        rb = cluster_bootstrap(kmu_r, xs, games, B=B, seed=C.RNG_SEED)
        ref_mean = _mean([r for _, _, r in xs])
        out[name] = {"n_items": len(rows), "n_known": sum(k for _, (_, k, _) in rows),
                     "known_share_of_overall": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]],
                     "known_minus_unknown": {"estimate": diff["estimate"], "ci": [diff["ci_low"], diff["ci_high"]],
                                             "ratio": {"estimate": rb["estimate"], "ci": [rb["ci_low"], rb["ci_high"]],
                                                       "n_resamples": B, "n_dropped": rb.get("n_dropped", 0),
                                                       "estimable": bool(ref_mean > 0 and rb.get("n_dropped", 0) <= DROPPED_DRAWS_MAX * B)}}}
    if out.get("desire_check_vs_swapped"):
        out["H7b_desire_check_known_vs_unknown"] = read_secondary(
            "H7b_desire_check_known_vs_unknown", {"cand": out["desire_check_vs_swapped"]["known_minus_unknown"]})
    return out


def _scores_of(run: str | None, tag: str) -> tuple[list[dict], list[Path], list[str]]:
    if not run:
        return [], [], []
    folders, problems = complete_shards(run)
    if problems:
        return [], folders, problems
    return [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag], folders, []


def _undefined(x: dict) -> bool:
    L = x.get("L")
    return L is None or L != L


def score_integrity(scores: list[dict], union: list[dict], tags: list[str], sample: set[str] | None = None) -> dict[str, Any]:
    """Every score checked against the current items before any analysis (amendment after review 2).

    Per model: scores of items not in the battery, scores whose item hash differs from the current item's, duplicate
    (item, condition) pairs, and, per arm and format, the scores expected (`model_jobs`: the model's arms and items;
    `sample` None reads every item as a sample item) against those present. Unknown items and unexpected conditions
    are counted apart (amendment after review 3: `n_unexpected` counts scores of battery items under a condition the
    model is not scored on). A score with an undefined L (NaN or missing) on an arm the primary contrasts read makes
    the model a mismatch (amendment after review 3); undefined L on other arms is counted and reported. A model is
    `complete` when nothing is off, `not scored` when it has no score at all, else `mismatch`. `refuse` is set when
    any model is a mismatch or a primary model is not scored."""
    items = {t["id"]: t for t in union}
    hashes = {i: _item_hash(t) for i, t in items.items()}
    sample = set(items) if sample is None else sample
    out: dict[str, Any] = {}
    for tag in tags:
        mine = [x for x in scores if x.get("model") == tag]
        expected = {(t["id"], condition(a, f)) for t, a, f in model_jobs(tag, union, sample)}
        keys = Counter((x["transition_id"], x["condition"]) for x in mine)
        cells: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "present": 0})
        for i, cond in expected:
            arm, fmt = split_condition(cond)
            cells[f"{arm}/{fmt}"]["expected"] += 1
        for (i, cond), n in keys.items():
            arm, fmt = split_condition(cond)
            cells[f"{arm}/{fmt}"]["present"] += n
        rec = {"n_scores": len(mine), "n_expected": len(expected),
               "n_unknown_items": sum(x["transition_id"] not in items for x in mine),
               "n_hash_mismatch": sum(x["transition_id"] in items and x.get("item_hash") != hashes[x["transition_id"]] for x in mine),
               "n_duplicates": sum(n - 1 for n in keys.values() if n > 1),
               "n_missing": len(expected - set(keys)),
               "n_unexpected": len({k for k in keys if k[0] in items} - expected),
               "n_undefined_L": sum(_undefined(x) for x in mine),
               "n_undefined_L_primary_arms": sum(_undefined(x) and split_condition(x.get("condition", ""))[0] in PRIMARY_ARMS
                                                 for x in mine),
               "cells_off": {k: v for k, v in sorted(cells.items()) if v["expected"] != v["present"]}}
        off = any(rec[k] for k in ("n_unknown_items", "n_hash_mismatch", "n_duplicates", "n_missing", "n_unexpected",
                                   "n_undefined_L_primary_arms"))
        rec["status"] = "not scored" if not mine else "mismatch" if off else "complete"
        out[tag] = rec
    out["refuse"] = any(out[t]["status"] == "mismatch" or (t in PRIMARY_MODELS and out[t]["status"] == "not scored")
                        for t in tags)
    return out


def run_mech2_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False},
                  inputs=tuple(p / "scoring.jsonl" for p in folders if (p / "scoring.jsonl").exists()))
    if problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    primary = rq1.rq1_sets(load_items(tuple(cfg.get("sources", SOURCES))))["primary"]
    sets = battery_items(primary, int(cfg.get("sample_n", SAMPLE_N)))
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    tags = list(cfg.get("models", PRIMARY_MODELS + SECONDARY_MODELS))
    integrity = score_integrity(scores, sets["union"], tags, sets["sample"])
    write_json(d / "integrity.json", integrity)
    if integrity["refuse"]:
        write_status(run_id, "scores do not match the current items or the expected jobs; nothing analysed:\n" +
                     "\n".join(f"{t}: {integrity[t]['status']} ({ {k: v for k, v in integrity[t].items() if k.startswith('n_')} })"
                               for t in tags), ok=False)
        return 1
    B, perms = int(cfg.get("bootstrap", C.BOOTSTRAP_RESAMPLES)), int(cfg.get("permutations", C.PERMUTATIONS))
    pops = populations(sets["union"], sets["sample"], sets["step0"])
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_sample": len(sets["sample"]), "n_step0": len(sets["step0"]),
                                        "n_union": len(sets["union"]), "n_sample_placement": len(pops["sample_placement"]),
                                        "n_p5": len(pops["p5"]), "n_sample_take": len(pops["sample_take"]),
                                        "n_games_sample": len({t["game_id"] for t in pops["sample"]}),
                                        "task_types_sample": dict(Counter(task_type(t["task_line"]) for t in pops["sample"]))},
                              "integrity": integrity,
                              "design": {"separations": separations(),
                                         "secondary_predictions": {k: {h: sorted(z, key=ZONES.index) for h, z in v.items()}
                                                                   for k, v in SECONDARY_PREDICTIONS.items()}}}
    others: dict[str, list[Path]] = {}
    waiting: dict[str, list[str]] = {}
    scored = [tag for tag in tags if integrity[tag]["status"] == "complete"]
    report["models_not_scored"] = [tag for tag in tags if tag not in scored]
    for tag in scored:
        s = [x for x in scores if x["model"] == tag]
        block = {"role": ("primary" if tag in PRIMARY_MODELS else
                          "secondary: reported descriptively, not part of any primary family or conclusion"),
                 **model_block(s, sets["union"], sets["sample"], sets["step0"], B=B, perms=perms)}
        ov, ov_folders, ov_problems = _scores_of(cfg.get("overlap_source"), tag)
        if ov_problems:
            waiting[cfg["overlap_source"]] = ov_problems
        elif ov:
            others[cfg["overlap_source"]] = ov_folders
            block["fewshot_overlap"] = fewshot_overlap(ov, primary, B=B)
        ms, m_folders, m_problems = _scores_of(cfg.get("mech_source"), tag)
        if m_problems:
            waiting[cfg["mech_source"]] = m_problems
        elif ms:
            others[cfg["mech_source"]] = m_folders
            block["knowledge_split"] = knowledge_split(s, ms, pops["sample"], B=B)
        report[tag] = block
    report["cross_model_item_share"] = {arm: {tag: report[tag]["arms_on_sample"][arm]["yn_item_share_toward_yes_vs_swapped"]["estimate"]
                                              for tag in scored} for arm in CONTENT_ARMS if arm != "swapped"}
    report["waiting_for_analysis_only_sources"] = waiting
    report["provenance"] = analysis_provenance(cfg["source"], folders, cfg.get("sources", SOURCES), others=others)
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Mechanism battery (exploratory)", report)
    write_status(run_id, "\n".join(status_lines(report, scored)), ok=False,
                 reported="mechanism battery report" if not waiting else "partial report")
    return 0


def status_lines(report: dict[str, Any], tags: list[str]) -> list[str]:
    """Per model: each primary contrast's two-candidate estimate and category, its yes/no category, and the joint
    summary: the content accounts consistent, untested and contradicted, each with the contrasts its verdict rests on
    (amendment after review 3), the position verdict, and scores with an undefined L on arms outside the primary
    contrasts (those on primary arms refuse the analysis)."""
    lines = []
    for tag in tags:
        prim, yn = report[tag]["primary"]["contrasts"], report[tag]["primary_yes_no"]["contrasts"]
        summ = report[tag]["readout_both_formats"]["summary"]
        undefined = (report.get("integrity") or {}).get(tag, {}).get("n_undefined_L", 0)
        lines.append(f"{tag}: " + ", ".join(f"{k.split('_')[0]} {v['estimate']:.2f} ({v['category']}; yes/no {yn[k]['category']})"
                                            for k, v in prim.items())
                     + f"; both formats: {summ['reading']}; position {summ['position']}"
                     + (f"; undefined L on {undefined} scores of secondary arms" if undefined else ""))
    if report.get("models_not_scored"):
        lines.append("not scored: " + ", ".join(report["models_not_scored"]))
    return lines
