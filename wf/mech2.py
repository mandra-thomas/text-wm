"""Exploratory mechanism battery (decided 2026-11-20, before any of its scores; notes/mechanism2_design.md, amended
after review 1, still before any score).

Why does naming the action's object in the task push the world model toward success on determined failures? Candidate
accounts, not exclusive (H1 no state tracking, H2 task as evidence, H3 referent salience, H4 wishful prediction, H5
policy blending, H6 prompt artefacts, split into H6_lexical, i.e. name salience or word overlap, and H6_position),
predict different size zones for a set of task lines. Every arm differs from the others only in the task line, or, for
the position arms, only in where the task line sits in the history. Each arm is scored in two formats: the
two-candidate log-odds of the success text against "Nothing happens." (`cand`) and the yes/no question of rq1v (`yn`,
condition name prefixed "yn_").

Items: a seeded sample of RQ2's primary items stratified by game and failure reason (SAMPLE_N), plus every step-0
item, so the position test covers all of them. Every arm is scored on every item of that union (the negated arm on take
items only); the primary contrasts read the sample, a restriction of it (P5, P7) or the step-0 items (P6).

Models: Qwen3-8B and Llama-3.1-8B (primary) and Mistral-7B-Instruct-v0.3 (secondary, descriptive), at pinned revisions.

`mech2_analysis` reports the seven primary contrasts per model and format (Holm within each), each contrast's size
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
CONTENT_ARMS = ("aligned", "swapped", "neutral", "desire_keep", "desire_check", "mention_ignore", "mention_exists",
                "described", "swapped_described", "done", "cancelled", "negated")
POSITION_ARMS = ("aligned_start", "swapped_start", "aligned_end", "swapped_end")
ARMS = CONTENT_ARMS + POSITION_ARMS
TASK_PREFIX = "Your task is to: "

# Templates. {obj} is the goal object's type as the task names it (the action names an instance of it), {a} its
# article, {pp} and {prep} come from the action's verb, {rec} is the receptacle instance the action names.
PARTICIPLE = {"take": "taken", "move": "moved", "heat": "heated", "cool": "cooled", "clean": "cleaned"}
PREPOSITION = {"take": "from", "move": "to", "heat": "with", "cool": "with", "clean": "with"}
DESIRE_KEEP = "Your task is to: make sure that the {obj} is not {pp}; leave it where it is."
DESIRE_CHECK = "Your task is to: find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it cannot."
MENTION_IGNORE = "{swapped} Ignore the {obj}."
DONE = "Your task was to: {task} It is already done; nothing is left to do."
CANCELLED = "Your task was to: {task} The user has cancelled it; nothing is left to do."
# Amendment after review 1 (2026-11-20, before any score): the negated-existence arm, built for take actions only (the
# one verb whose success needs the object in the receptacle the action names). The claim is the user's report, not the
# narrator's: on many items an instance of {obj} is in {rec}, so a bare "There is no ..." would state a falsehood.
# "in" for every receptacle type, as ALFWorld's own task lines write it ("put a mug in countertop.").
NEGATED = "{swapped} The user says there is no {obj} in the {rec}."
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
    if base in ("desire_keep", "desire_check"):
        p = parse_action(t["action"])
        if not p or p[1] != obj:
            return None
        verb, _, rec = p
        if base == "desire_keep":
            return DESIRE_KEEP.format(obj=obj, pp=PARTICIPLE[verb])
        return DESIRE_CHECK.format(a=rq1._article(obj), obj=obj, pp=PARTICIPLE[verb], prep=PREPOSITION[verb], rec=rec)
    if base == "mention_ignore":
        return MENTION_IGNORE.format(swapped=t["swapped_task_line"], obj=obj)
    if base == "negated":
        p = parse_action(t["action"])
        if not p or p[0] != "take" or p[1] != obj:
            return None
        return NEGATED.format(swapped=t["swapped_task_line"], obj=obj, rec=p[2])
    if base == "mention_exists":
        return rq1.goal_line(t, "swapped_plus")
    if base in ("described", "swapped_described"):
        if not rq1x.desc_ok(t):
            return None
        return (rq1x.describe_line(t["task_line"], obj) if base == "described"
                else rq1x.describe_line(t["swapped_task_line"], t["swapped_goal_object"]))
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


def alias_of(t: dict, arm: str) -> str | None:
    """The plain arm whose prompt a position arm repeats exactly for this item (step 0: `_end` is the usual place),
    so it is not scored again and the analysis reads the plain score; None when the prompts differ."""
    if arm in POSITION_ARMS and history_for(t, arm) == t["history_text"]:
        return arm.rsplit("_", 1)[0]
    return None


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


def jobs(items: list[dict]) -> list[tuple[dict, str, str]]:
    """(item, arm, format) for every arm that builds for an item and is not an alias of a plain arm."""
    out = []
    for t in items:
        for arm in ARMS:
            if alias_of(t, arm) or task_line(t, arm) is None:
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
    d = _init_run(run_id, cfg, inputs=())
    primary = rq1.rq1_sets(load_items(sources))["primary"]
    sets = battery_items(primary, int(cfg.get("sample_n", SAMPLE_N)))
    work = jobs(shard_items(sets["union"], i, n))
    hashes = {t["id"]: _item_hash(t) for t, _, _ in work}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    for tag in tags:
        model_name, kw, prompt = entries[tag]
        todo = [(t, a, f) for t, a, f in work if (tag, t["id"], condition(a, f)) not in done]
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
    write_status(run_id, f"shard {i} of {n}: {len(work)} scores per model", ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------

# Primary contrasts (amended after review 1, 2026-11-20, before any score): a weighted sum of arms, its population
# and its reference effect (the aligned-minus-swapped effect on the same items and in the same format, the denominator
# of the size ratio r). Seven per model and format, Holm within each.
_REF = ((1, "aligned"), (-1, "swapped"))
PRIMARY: dict[str, dict[str, Any]] = {
    # same presupposed object ("the {obj}") in both lines; only the desire differs: salience against wishful
    "P1_keep_vs_ignore": {"terms": ((1, "desire_keep"), (-1, "mention_ignore")), "population": "sample", "reference": _REF},
    "P2_desire_check": {"terms": ((1, "desire_check"), (-1, "swapped")), "population": "sample", "reference": _REF},
    "P3_mention_without_need": {"terms": ((1, "mention_ignore"), (-1, "swapped")), "population": "sample", "reference": _REF},
    "P4_need_without_name": {"terms": ((1, "described"), (-1, "swapped")), "population": "sample", "reference": _REF},
    # placement items, without those taking from a target-type receptacle and those whose history contradicts "done"
    "P5_done_vs_cancelled": {"terms": ((1, "done"), (-1, "cancelled")), "population": "p5", "reference": _REF},
    "P6_position_step0": {"terms": ((1, "aligned_start"), (-1, "swapped_start"), (-1, "aligned"), (1, "swapped")),
                          "population": "step0", "reference": _REF},
    # the object named and its presence in the action's receptacle denied, take items only: mention against evidence
    "P7_negated_existence": {"terms": ((1, "negated"), (-1, "swapped")), "population": "sample_take", "reference": _REF},
}
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
# review 1). H1 makes no prediction here (it is read from the knowledge split only). H3 is referent salience (the
# object made salient, by name or description); name salience is the lexical part of H6 (H6_lexical), which this
# battery cannot tell from word overlap. P6 counts only for H6_position, whose reading needs |r| beyond the band.
_P = POSITIVE
PREDICTIONS: dict[str, dict[str, frozenset[str]]] = {
    "H2_evidence": {"P1_keep_vs_ignore": frozenset({"Z", "S"}), "P2_desire_check": _P, "P3_mention_without_need": _P,
                    "P4_need_without_name": frozenset({"L"}), "P5_done_vs_cancelled": frozenset({"N"}),
                    "P7_negated_existence": frozenset({"N"})},
    "H3_salience": {"P1_keep_vs_ignore": frozenset({"Z", "S"}), "P2_desire_check": _P, "P3_mention_without_need": _P,
                    "P4_need_without_name": _P, "P5_done_vs_cancelled": frozenset({"Z"}), "P7_negated_existence": _P},
    "H4_wishful": {"P1_keep_vs_ignore": frozenset({"N"}), "P2_desire_check": frozenset({"N"}),
                   "P3_mention_without_need": frozenset({"Z"}), "P4_need_without_name": frozenset({"L"}),
                   "P5_done_vs_cancelled": frozenset({"Z"}), "P7_negated_existence": frozenset({"Z"})},
    "H5_policy_blend": {"P1_keep_vs_ignore": frozenset({"N", "Z"}), "P2_desire_check": _P,
                        "P3_mention_without_need": frozenset({"N", "Z"}), "P4_need_without_name": frozenset({"L"}),
                        "P5_done_vs_cancelled": frozenset({"Z"}), "P7_negated_existence": frozenset({"Z"})},
    "H6_lexical": {"P1_keep_vs_ignore": frozenset({"Z", "S"}), "P2_desire_check": _P, "P3_mention_without_need": _P,
                   "P4_need_without_name": frozenset({"Z"}), "P5_done_vs_cancelled": frozenset({"Z"}),
                   "P7_negated_existence": _P},
    "H6_position": {"P6_position_step0": frozenset({"N", "S", "L"})},
}
CONTENT_ACCOUNTS = ("H2_evidence", "H3_salience", "H4_wishful", "H5_policy_blend", "H6_lexical")


def _lookup(scores: list[dict]) -> dict[tuple[str, str], dict]:
    return {(x["transition_id"], x["condition"]): x for x in scores if x.get("L") == x.get("L")}


def _value(by: dict, t: dict, arm: str, fmt: str, field: str = "L") -> float | None:
    """An item's score for an arm, reading the plain arm's score where the position arm repeats its prompt."""
    rec = by.get((t["id"], condition(arm, fmt)))
    if rec is None:
        base = alias_of(t, arm)
        rec = by.get((t["id"], condition(base, fmt))) if base else None
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


def _verdict(rows: dict[str, str]) -> str:
    """contradicted (any contrast contradicted), consistent (none contradicted, at least one consistent), or
    untested (every decided contrast missing: no contradicted and no consistent contrast)."""
    vals = list(rows.values())
    if "contradicted" in vals:
        return "contradicted"
    return "consistent" if "consistent" in vals else "untested"


def readout(observed: dict[str, tuple[str, ...] | None]) -> dict[str, Any]:
    """For each account, each contrast it predicts as consistent, contradicted or undecided against the prediction
    table (one format), and the account's verdict."""
    out = {}
    for h, preds in PREDICTIONS.items():
        rows = {c: reading(allowed, observed.get(c)) for c, allowed in preds.items()}
        vals = list(rows.values())
        out[h] = {"per_contrast": rows, "n_consistent": vals.count("consistent"),
                  "n_contradicted": vals.count("contradicted"), "n_undecided": vals.count("undecided"),
                  "verdict": _verdict(rows)}
    return out


def joint_readout(cand: dict[str, Any], yn: dict[str, Any]) -> dict[str, Any]:
    """The decision rule across formats (amendment after review 1): a contrast is decided for an account only when the
    two formats agree (both consistent, or both contradicted). One consistent and one contradicted is
    `format_dependent`; anything else is undecided. Both count as undecided in the verdict, and the format-dependent
    contrasts are listed by name. When every content account (H2-H5, H6_lexical) is contradicted, the summary says
    no single account fits (a mixture, or an account not on the list); it never picks the least contradicted one."""
    out: dict[str, Any] = {}
    for h in PREDICTIONS:
        rows = {}
        for c in PREDICTIONS[h]:
            a, b = cand[h]["per_contrast"][c], yn[h]["per_contrast"][c]
            if a == b and a in ("consistent", "contradicted"):
                rows[c] = a
            elif {a, b} == {"consistent", "contradicted"}:
                rows[c] = "format_dependent"
            else:
                rows[c] = "undecided"
        vals = list(rows.values())
        out[h] = {"per_contrast": rows, "n_consistent": vals.count("consistent"),
                  "n_contradicted": vals.count("contradicted"),
                  "format_dependent": [c for c, v in rows.items() if v == "format_dependent"],
                  "verdict": _verdict(rows)}
    fitting = [h for h in CONTENT_ACCOUNTS if out[h]["verdict"] != "contradicted"]
    out["summary"] = {
        "content_accounts_not_contradicted": fitting,
        "content_accounts_consistent": [h for h in CONTENT_ACCOUNTS if out[h]["verdict"] == "consistent"],
        "reading": ("no single account; mixture or an account not on the list" if not fitting else
                    "accounts not contradicted: " + ", ".join(fitting)),
        "position": out["H6_position"]["verdict"],
    }
    return out


def populations(union: list[dict], sample: set[str], step0: set[str]) -> dict[str, list[dict]]:
    s = [t for t in union if t["id"] in sample]
    return {"sample": s, "sample_placement": [t for t in s if task_type(t["task_line"]) in PLACEMENT_TYPES],
            "p5": [t for t in s if p5_eligible(t)], "sample_take": [t for t in s if p7_eligible(t)],
            "step0": [t for t in union if t["id"] in step0],
            "later": [t for t in s if t.get("history_len", 0) > 0]}


def primary_family(by: dict, pops: dict[str, list[dict]], fmt: str = "cand", B: int = C.BOOTSTRAP_RESAMPLES,
                   perms: int = C.PERMUTATIONS) -> dict[str, Any]:
    """The seven primary contrasts in one format, Holm within the family, each categorised, and the one-format
    readout."""
    res = {name: contrast(by, pops[spec["population"]], spec["terms"], fmt, reference=spec["reference"], B=B, perms=perms)
           for name, spec in PRIMARY.items()}
    # a fixed family of seven: a missing or undefined p enters as 1, so the family never shrinks
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
    block["readout_both_formats"] = joint_readout(block["primary"]["readout"], block["primary_yes_no"]["readout"])
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
    block["secondary"] = {
        "desire_keep_vs_swapped": arms["desire_keep"]["vs_swapped_L"],
        "desire_keep_vs_swapped_yes_no": arms["desire_keep"]["yn_vs_swapped_L"],
        "negated_vs_exists_sentence_take": contrast(by, pops["sample_take"], ((1, "negated"), (-1, "mention_exists")), "cand", reference=ref, **kw),
        # P7 by what the history shows of the receptacle (the claim established, not shown, or contradicted)
        "negated_by_history": {k: contrast(by, [t for t in pops["sample"] if is_take(t) and negated_status(t)[1] == k],
                                           ((1, "negated"), (-1, "swapped")), "cand", reference=ref, **kw)
                               for k in ("shown, without it", "contents not shown", "shown, holds it")},
        "cancelled_vs_swapped": arms["cancelled"]["vs_swapped_L"],
        "done_vs_swapped": arms["done"]["vs_swapped_L"],
        "ignore_vs_exists_sentence": contrast(by, pops["sample"], ((1, "mention_ignore"), (-1, "mention_exists")), "cand", **kw),
        "described_vs_swapped_described": contrast(by, pops["sample"], ((1, "described"), (-1, "swapped_described")), "cand", reference=ref, **kw),
        "desire_check_vs_keep": contrast(by, pops["sample"], ((1, "desire_check"), (-1, "desire_keep")), "cand", **kw),
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
        "position": {
            "step0_start_vs_usual_yes_no": contrast(by, pops["step0"], PRIMARY["P6_position_step0"]["terms"], "yn", reference=ref, **kw),
            "sample_start_vs_usual": contrast(by, pops["sample"], PRIMARY["P6_position_step0"]["terms"], "cand", reference=ref, **kw),
            "later_end_vs_usual": contrast(by, pops["later"], ((1, "aligned_end"), (-1, "swapped_end"), (-1, "aligned"), (1, "swapped")), "cand", reference=ref, **kw),
            "end_vs_start_step0": contrast(by, pops["step0"], ((1, "aligned_end"), (-1, "swapped_end"), (-1, "aligned_start"), (1, "swapped_start")), "cand", **kw),
            "end_vs_start_later": contrast(by, pops["later"], ((1, "aligned_end"), (-1, "swapped_end"), (-1, "aligned_start"), (1, "swapped_start")), "cand", **kw),
        },
    }
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
    with a verdict. H1 predicts shares below 0.75 for positive contrasts."""
    by = _lookup(scores)
    mb = {(x["transition_id"], x["condition"]): x["L"] for x in mech_scores if x.get("L") == x.get("L")}

    def known(t):
        a, b = mb.get((t["id"], "state_absent")), mb.get((t["id"], "statetwin_absent"))
        return None if a is None or b is None else a < 0 < b

    out = {}
    specs = {"aligned_effect": ((1, "aligned"), (-1, "swapped")),
             **{n: s["terms"] for n, s in PRIMARY.items() if s["population"] != "step0"}}
    for name, terms in specs.items():
        rows = [(t["game_id"], (d, k)) for t in items for d in [_combo(by, t, terms, "cand")] for k in [known(t)]
                if d is not None and k is not None]
        if not rows:
            out[name] = None
            continue

        def share(xs):
            kn = [d for d, k in xs if k]
            allm = _mean([d for d, _ in xs])
            return _mean(kn) / allm if kn and allm == allm and allm != 0 else float("nan")

        boot = cluster_bootstrap(share, [x for _, x in rows], [g for g, _ in rows], B=B, seed=C.RNG_SEED)
        out[name] = {"n_items": len(rows), "n_known": sum(k for _, (_, k) in rows),
                     "known_share_of_overall": boot["estimate"], "ci": [boot["ci_low"], boot["ci_high"]]}
    return out


def _scores_of(run: str | None, tag: str) -> tuple[list[dict], list[Path], list[str]]:
    if not run:
        return [], [], []
    folders, problems = complete_shards(run)
    if problems:
        return [], folders, problems
    return [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag], folders, []


def run_mech2_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    primary = rq1.rq1_sets(load_items(tuple(cfg.get("sources", SOURCES))))["primary"]
    sets = battery_items(primary, int(cfg.get("sample_n", SAMPLE_N)))
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    B, perms = int(cfg.get("bootstrap", C.BOOTSTRAP_RESAMPLES)), int(cfg.get("permutations", C.PERMUTATIONS))
    pops = populations(sets["union"], sets["sample"], sets["step0"])
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_sample": len(sets["sample"]), "n_step0": len(sets["step0"]),
                                        "n_union": len(sets["union"]), "n_sample_placement": len(pops["sample_placement"]),
                                        "n_p5": len(pops["p5"]), "n_sample_take": len(pops["sample_take"]),
                                        "n_games_sample": len({t["game_id"] for t in pops["sample"]}),
                                        "task_types_sample": dict(Counter(task_type(t["task_line"]) for t in pops["sample"]))}}
    others: dict[str, list[Path]] = {}
    waiting: dict[str, list[str]] = {}
    tags = list(cfg.get("models", PRIMARY_MODELS + SECONDARY_MODELS))
    for tag in tags:
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
                                              for tag in tags} for arm in CONTENT_ARMS if arm != "swapped"}
    report["waiting_for_analysis_only_sources"] = waiting
    report["provenance"] = analysis_provenance(cfg["source"], folders, cfg.get("sources", SOURCES), others=others)
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Mechanism battery (exploratory)", report)
    lines = []
    for tag in tags:
        prim = report[tag]["primary"]["contrasts"]
        lines.append(f"{tag}: " + ", ".join(f"{k.split('_')[0]} {v['estimate']:.2f} ({v['category']})" for k, v in prim.items())
                     + f"; both formats: {report[tag]['readout_both_formats']['summary']['reading']}")
    write_status(run_id, "\n".join(lines), ok=False, reported="mechanism battery report" if not waiting else "partial report")
    return 0
