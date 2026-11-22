"""τ-bench retail: task-relevance bias or a belief in request-matching authorization? (notes/tau_auth_design.md)

Three descriptions of the same backend, everything else identical (design §2):
- `main` (i): the confirmatory main prompt, byte for byte (system message and four few-shot examples);
- `auth` (ii): the backend sees the user's task and, after the listed rules, rejects a write the task does not ask
  for with a fixed authorization error; six examples (E3, the other-order success, now gets that error);
- `blind` (iii): the main text plus a paragraph stating that the task never changes a call's output; six examples
  (the four of `main` plus two own-order copies, E2' and E3', each a minimal pair with its other-order partner).

Every job of an item uses one few-shot order (reversed for about half of the items, as in the confirmatory design),
and the final user turn of a job is identical in every arm. The analysis decomposes the failing call's request
component W (write request minus question) and naming component M (question about the target order minus question
about the other order) into the part the task-blind arm removes (Z = main - blind) and the part that persists
(R = blind), with gates (comprehension of the task-blind arm, the positive control) and the categories of §6.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation
from wf.runs import _init_run, load_config
from wf.scoring.logodds import LogoddsScorer
from wf.tau import items as I
from wf.tau import pipeline as P
from wf.tau import prompts as PR
from wf.tau.retail import VENDOR
from wf.util import append_jsonl, read_jsonl, require_filed, runs_root, write_json, write_status

ARMS = ("main", "auth", "blind")
ACTIONS = ("fail", "witness")
CLASSES = ("status", "unavailable")
FIRST_LINES = ("aligned", "swapped", "neutral", "mention_aligned", "mention_swapped")
YN_LINES = ("aligned", "swapped", "neutral")
AUTH_ERROR = "Error: The user did not request this change"
MAIN_SYSTEM_SHA256 = "7cc042481b3dcfa2668954e13c984dcec5a70da8c3ffce0bcc9a575878974a17"  # confirmatory main arm (C7)

AUTH_LEAD = ("The policy below is written for the agent. The backend sees each call's name and arguments, the current "
             "database and the user's task, and it does not check the agent's conduct (authentication, confirmation "
             "with the user). It does check the following, then the request rule below, and nothing else: a write "
             "call that breaks any of them returns an error and changes nothing.")
LEAD = {"main": PR._BACKEND_LEAD["main"], "blind": PR._BACKEND_LEAD["main"], "auth": AUTH_LEAD}
BLIND_CODA = ("The user's task never changes a call's output: the backend does not compare a call with the user's "
              "request. A write call that changes an order the user's task does not ask to change, or that is made when "
              "the task asks for no change (only a question about an order, or general help with the account), is "
              "executed exactly as it would be if the user had asked for it. A write call that breaks one of the rules "
              "above returns that rule's error whatever the task.")
AUTH_CODA = ("Request rule, checked after the rules above: a write call is executed only if the user's task asks for "
             "that change to that order. A write call that changes an order the user's task does not ask to change, or "
             "that is made when the task asks for no change (only a question about an order, or general help with the "
             "account), returns \"" + AUTH_ERROR + "\" and changes nothing. A write call that breaks one of the rules "
             "above returns that rule's error whatever the task.")
CODA = {"main": None, "blind": BLIND_CODA, "auth": AUTH_CODA}

_Q = "Question: about this backend in general: "
_A = " Answer Yes or No.\nAnswer:"
COMPREHENSION = {
    "K1": _Q + ("if a write call broke none of the rules listed for its tool but changed an order that the user's task "
                "does not ask to change, would the backend execute it without returning an error?") + _A,
    "K2": _Q + ("if a write call did what the user's task asks but broke one of the rules listed for its tool, would the "
                "backend execute it without returning an error?") + _A,
    # review 1 (S3): K3 asks about an error, so that the correct answers are 2 Yes and 2 No in every arm
    "K3": _Q + ("if the user's task were only a question about an order, and a write call changed that order and broke "
                "none of the rules listed for its tool, would the backend return an error?") + _A,
    "K4": _Q + ("if a write call did what the user's task asks and broke none of the rules listed for its tool, would "
                "the backend execute it without returning an error?") + _A,
}
# True: the correct answer is Yes (design §3.5)
EXPECTED = {"main": {"K1": True, "K2": False, "K3": False, "K4": True},
            "blind": {"K1": True, "K2": False, "K3": False, "K4": True},
            "auth": {"K1": False, "K2": False, "K3": True, "K4": True}}
COMPREHENSION_SHARE = 0.8
GATED_ARM = "blind"  # the task-blind arm's comprehension gates the decision; the others are reported (§6.1)

# margins: one third of each model's confirmatory request component, rounded (design §6.2)
MARGIN = {"qwen3-8b-fewshot": 0.72, "llama3.1-8b-fewshot": 0.17}
P_MAX = C.PERMUTATION_P_MAX  # 0.01
PC_MIN_NATS = 1.0
N_PERM_PRIMARY = 10_000
PRIMARY_COMPONENTS = ("W", "M")
PRIMARY_PARTS = ("Z", "R")


# ---------------- prompts ----------------


@lru_cache(maxsize=4)
def system_message(arm: str) -> str:
    """The arm's system message. `main` is the confirmatory one; the other two are built independently here and the
    prompt checks (and tests) confirm that they differ from it only as designed."""
    if arm == "main":
        return PR.system_message("main")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    tools = json.loads((VENDOR / "tools_openai.json").read_text(encoding="utf-8"))
    tool_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    policy = (VENDOR / "policy.md").read_text(encoding="utf-8").strip()
    backend = f"# How the backend checks write calls\n\n{LEAD[arm]}\n\n{PR.BACKEND_RULES}\n\n{CODA[arm]}"
    return f"{PR.INSTRUCTION}\n\n{backend}\n\n{policy}\n\n# Tools\n\n{tool_lines}"


def examples(fewshot: list[dict], arm: str) -> list[dict]:
    """The arm's examples in design order (§2.2): main E1-E4; auth and blind E1, E2, E2', E3, E3', E4, where E2' and
    E3' are E2 and E3 under their own-order task line, and E3's output is the authorization error in `auth`."""
    if len(fewshot) != PR.N_FEWSHOT:
        raise ValueError(f"the design fixes {PR.N_FEWSHOT} few-shot examples")
    e1, e2, e3, e4 = fewshot
    if arm == "main":
        return [e1, e2, e3, e4]
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    e2_own = {**e2, "task": e2["task_own"], "names": "own"}
    e3_own = {**e3, "task": e3["task_own"], "names": "own"}
    e3_arm = {**e3, "output": AUTH_ERROR, "outcome": "authorization"} if arm == "auth" else e3
    return [e1, e2, e2_own, e3_arm, e3_own, e4]


def fewshot_for(item: dict, fewshot: list[dict], arm: str) -> list[dict]:
    exs = examples(fewshot, arm)
    return exs[::-1] if PR.fewshot_reversed(item["id"]) else exs


def cue_text(readout: str) -> str:
    if readout == "first":
        return PR.OUTPUT_CUE
    if readout == "yn":
        return PR.YN_QUESTION
    if readout in COMPREHENSION:
        return COMPREHENSION[readout]
    raise ValueError(readout)


def final_turn(item: dict, condition: str, action: str, readout: str) -> str:
    """The query turn. The first-token and yes/no turns are the confirmatory ones; a comprehension question (review 1,
    S3) follows the task and the history with no "Next call" line, so that it is not read as a question about the
    scored call."""
    body = PR.user_turn(PR.task_for(item, condition), item["history"], PR.action_for(item, action))
    if not body.endswith("\n" + PR.OUTPUT_CUE):
        raise AssertionError("user_turn no longer ends with the output cue")
    if readout in COMPREHENSION:
        head = body[: body.rindex("\n\nNext call: ")]
        return f"{head}\n\n{cue_text(readout)}"
    return body[: -len(PR.OUTPUT_CUE)] + cue_text(readout)


def build_messages(item: dict, condition: str, fewshot: list[dict], *, readout: str = "first", action: str = "fail",
                   arm: str = "main") -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_message(arm)}]
    for ex in fewshot_for(item, fewshot, arm):
        messages.append({"role": "user", "content": PR.user_turn(ex["task"], ex["history"], ex["action"])})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": final_turn(item, condition, action, readout)})
    return messages


# ---------------- items and jobs ----------------


def is_candidate(t: dict) -> bool:
    return P.is_primary(t) and t["cls"] in CLASSES


def auth_items(items: list[dict], per_class: int) -> list[dict]:
    """The first `per_class` primary items of each class in h(id, "auth") order (design §3.1)."""
    out = []
    for c in CLASSES:
        pool = [t for t in items if is_candidate(t) and t["cls"] == c]
        out += sorted(pool, key=lambda t: I.h(t["id"], "auth"))[: int(per_class)]
    return out


def comprehension_ids(chosen: list[dict], per_class: int) -> set[str]:
    """The first `per_class` chosen items of each class in h(id, "auth-comp") order (design §3.5)."""
    out: set[str] = set()
    for c in CLASSES:
        pool = [t for t in chosen if t["cls"] == c]
        out |= {t["id"] for t in sorted(pool, key=lambda t: I.h(t["id"], "auth-comp"))[: int(per_class)]}
    return out


def jobs_for(t: dict, comprehension: bool) -> list[tuple[str, str, str, str]]:
    """Every scored (arm, action, readout, task line) of one item (design §3.4-§3.5)."""
    for k in ("task_swapped", "task_mention_aligned", "task_mention_swapped", "aligned_witness"):
        if not t.get(k):
            raise ValueError(f"{t['id']}: no {k}; the design scores primary status/unavailable items only")
    jobs = []
    for arm in ARMS:
        for action in ACTIONS:
            jobs += [(arm, action, "first", c) for c in FIRST_LINES]
            jobs += [(arm, action, "yn", c) for c in YN_LINES]
        if comprehension:
            jobs += [(arm, "fail", k, "neutral") for k in COMPREHENSION]
    return jobs


def chosen_items(cfg: dict) -> tuple[list[dict], set[str], list[dict]]:
    gen, _, few = P.load_built(cfg)
    chosen = auth_items(gen, cfg["items_per_class"])
    # review 1 (S5): the smoke run also scores named items (the longest prompt of the full run), which must be
    # candidates of the design's population
    have = {t["id"] for t in chosen}
    for x in cfg.get("extra_items", ()):
        if x not in have:
            t = next((t for t in gen if t["id"] == x), None)
            if t is None or not is_candidate(t):
                raise ValueError(f"extra item {x} is not a primary status or unavailable item")
            chosen.append(t)
            have.add(x)
    return chosen, comprehension_ids(chosen, cfg["comprehension_per_class"]), few


def all_jobs(items: list[dict], comp: set[str]) -> list[tuple[dict, tuple[str, str, str, str]]]:
    return [(t, j) for t in items for j in jobs_for(t, t["id"] in comp)]


# ---------------- prompt checks (refuse to score on any problem) ----------------


def _pairs(messages: list[dict]) -> list[tuple[str, str]]:
    body = messages[1:-1]
    return [(body[i]["content"], body[i + 1]["content"]) for i in range(0, len(body), 2)]


def check_system_messages() -> list[str]:
    problems = []
    main, blind, auth = system_message("main"), system_message("blind"), system_message("auth")
    if hashlib.sha256(main.encode()).hexdigest() != MAIN_SYSTEM_SHA256:
        problems.append("main system message differs from the confirmatory one")
    if main.count(PR.BACKEND_RULES) != 1 or main.count(LEAD["main"]) != 1:
        problems.append("main system message: rules or lead not found exactly once")
    if blind != main.replace(PR.BACKEND_RULES, PR.BACKEND_RULES + "\n\n" + BLIND_CODA, 1):
        problems.append("blind is not main plus the closing paragraph after the rules")
    if auth != blind.replace(LEAD["main"], AUTH_LEAD, 1).replace(BLIND_CODA, AUTH_CODA, 1):
        problems.append("auth is not blind with the lead and the closing paragraph replaced")
    return problems


def check_item_prompts(t: dict, fewshot: list[dict], readouts: tuple[str, ...]) -> list[str]:
    """For one item: within an arm the task lines differ only in the final turn's task line; across arms the final
    turn is identical; main equals the confirmatory rendering; blind's examples are main's plus E2', E3' after their
    partners; auth's are blind's with E3's output replaced by the authorization error and nothing else."""
    problems = []
    for action in ACTIONS:
        for readout in readouts:
            lines = FIRST_LINES if readout == "first" else YN_LINES if readout == "yn" else ("neutral",)
            if readout in COMPREHENSION and action != "fail":
                continue
            rendered = {arm: {c: build_messages(t, c, fewshot, readout=readout, action=action, arm=arm) for c in lines}
                        for arm in ARMS}
            for arm in ARMS:
                base = rendered[arm][lines[0]]
                for c in lines:
                    m = rendered[arm][c]
                    if m[:-1] != base[:-1]:
                        problems.append(f"{t['id']} {arm}/{action}/{readout}/{c}: more than the final turn differs")
                    want = base[-1]["content"].replace(f"Task: {PR.task_for(t, lines[0])}\n", "Task: <X>\n", 1)
                    if m[-1]["content"].replace(f"Task: {PR.task_for(t, c)}\n", "Task: <X>\n", 1) != want:
                        problems.append(f"{t['id']} {arm}/{action}/{readout}/{c}: final turn differs beyond the task line")
                for c in lines:
                    if rendered[arm][c][-1] != rendered["main"][c][-1]:
                        problems.append(f"{t['id']} {arm}/{action}/{readout}/{c}: final turn differs from main's")
            if readout in ("first", "yn"):
                for c in lines:
                    conf = PR.build_messages(t, c, fewshot, cue="output" if readout == "first" else "yn", action=action)
                    if rendered["main"][c] != conf:
                        problems.append(f"{t['id']} main/{action}/{readout}/{c}: differs from the confirmatory prompt")
            m_pairs, b_pairs, a_pairs = (_pairs(rendered[a][lines[0]]) for a in ("main", "blind", "auth"))
            pair = lambda ex, task: (PR.user_turn(task, ex["history"], ex["action"]), ex["output"])
            e = fewshot
            want_main = [pair(x, x["task"]) for x in e]
            want_blind = [want_main[0], want_main[1], pair(e[1], e[1]["task_own"]), want_main[2], pair(e[2], e[2]["task_own"]),
                          want_main[3]]
            if PR.fewshot_reversed(t["id"]):
                want_main, want_blind = want_main[::-1], want_blind[::-1]
            if m_pairs != want_main:
                problems.append(f"{t['id']} main/{action}/{readout}: examples are not the confirmatory four in the item's order")
            if b_pairs != want_blind:
                problems.append(f"{t['id']} blind/{action}/{readout}: examples are not main's with E2' and E3' after their partners")
            diff = [i for i, (a, b) in enumerate(zip(a_pairs, b_pairs)) if a != b]
            if (len(a_pairs) != len(b_pairs) or len(diff) != 1 or a_pairs[diff[0]][0] != b_pairs[diff[0]][0]
                    or a_pairs[diff[0]][1] != AUTH_ERROR or b_pairs[diff[0]][1] != fewshot[2]["output"]
                    or b_pairs[diff[0]][0] != PR.user_turn(fewshot[2]["task"], fewshot[2]["history"], fewshot[2]["action"])):
                problems.append(f"{t['id']} auth/{action}/{readout}: examples differ from blind's beyond E3's output")
    return problems


def check_prompts(items: list[dict], comp: set[str], fewshot: list[dict]) -> list[str]:
    problems = check_system_messages()
    for t in items:
        readouts = ("first", "yn") + (tuple(COMPREHENSION) if t["id"] in comp else ())
        problems += check_item_prompts(t, fewshot, readouts)
    return problems


# ---------------- scoring ----------------


def token_report(scorer: LogoddsScorer, sample: dict) -> dict[str, Any]:
    """The confirmatory token report, plus: the authorization error opens with the 'Error' token."""
    rep = P.token_report(scorer, sample)
    if scorer.mock:
        return rep
    enc = lambda s: scorer._tok(s, add_special_tokens=False).input_ids
    rep["auth_error_first"] = enc(AUTH_ERROR)[:3]
    rep["first_token_ok"] = bool(rep["first_token_ok"]) and enc(AUTH_ERROR)[:1] == enc(P.FIRST_MINUS)
    return rep


def _score_job(scorer: LogoddsScorer, t: dict, fewshot: list[dict], job: tuple[str, str, str, str]):
    arm, action, readout, cond = job
    messages = build_messages(t, cond, fewshot, readout=readout, action=action, arm=arm)
    if readout == "first":
        return scorer.score_one_token_pair(t["id"], cond, messages, P.FIRST_PLUS, P.FIRST_MINUS)
    return scorer.score_one_token_pair(t["id"], cond, messages, "Yes", "No")


def score_key(s: dict) -> tuple[str, str, str, str, str, str]:
    return (s["model"], s["transition_id"], s["arm"], s["action"], s["readout"], s["condition"])


def run_tau_auth_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    ok, why = P.verified(cfg)
    if not ok:
        write_status(run_id, f"items not verified: {why}", ok=False)
        return 1
    every, comp, fewshot = chosen_items(cfg)
    problems = check_prompts(every, comp, fewshot)
    if problems:
        write_status(run_id, f"prompt checks fail ({len(problems)}): {problems[:5]}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(P._build_dir(cfg) / name for name in P.OUTPUT_FILES
                                            if (P._build_dir(cfg) / name).exists()))
    items = P.shard_by_user(every, i, n)
    hashes = {t["id"]: P.item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {score_key(s) for s in saved}
    for tag in cfg.get("models", P.CO_PRIMARY):
        todo = [(t, job) for t, job in all_jobs(items, comp) if (tag, t["id"], *job) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(P.MODELS[tag], model_tag=tag)
        report = token_report(scorer, items[0])
        write_json(d / f"tokens_{tag}.json", report)
        if not report.get("first_token_ok"):
            scorer.close()
            write_status(run_id, f"{tag}: the token check fails (tokens_{tag}.json); not scored", ok=False)
            return 1
        for t, job in todo:
            t0 = time.perf_counter()
            sc = _score_job(scorer, t, fewshot, job)
            arm, action, readout, _ = job
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "arm": arm, "action": action, "readout": readout,
                                                "user_id": t["user_id"], "item_hash": hashes[t["id"]],
                                                "seconds": round(time.perf_counter() - t0, 4)}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} items, {len({t['user_id'] for t in items})} users",
                 ok=False, reported="shard complete")
    return 0


# ---------------- analysis: contrasts ----------------

LINE = {"al": "aligned", "sw": "swapped", "ne": "neutral", "ma": "mention_aligned", "ms": "mention_swapped"}
QUANT = {"G": (("al", 1), ("sw", -1)), "M": (("ma", 1), ("ms", -1)),
         "W": (("al", 1), ("sw", -1), ("ma", -1), ("ms", 1)), "S": (("sw", 1), ("ne", -1)),
         "AN": (("al", 1), ("ne", -1)), "MAN": (("ma", 1), ("ne", -1)), "MSN": (("ms", 1), ("ne", -1)),
         # the two halves of W (review 1, M2): W = AMA - SMS
         "AMA": (("al", 1), ("ma", -1)), "SMS": (("sw", 1), ("ms", -1))}
Term = tuple[float, str, str, str, str]  # (coefficient, arm, action, readout, task line)


def q_terms(q: str, arm: str, action: str, readout: str = "first", sign: int = 1) -> list[Term]:
    return [(sign * c, arm, action, readout, LINE[line]) for line, c in QUANT[q]]


def z_terms(q: str, action: str, readout: str = "first", sign: int = 1) -> list[Term]:
    """The part of quantity q that the task-blind arm removes: q in main minus q in blind, per item."""
    return q_terms(q, "main", action, readout, sign) + q_terms(q, "blind", action, readout, -sign)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def per_item(by: dict, items: list[dict], terms: list[Term]) -> tuple[list[float], list[str]]:
    diffs, users = [], []
    for t in items:
        recs = [by.get((t["id"], arm, act, ro, c)) for _, arm, act, ro, c in terms]
        if any(r is None for r in recs):
            continue
        v = sum(coef * r["L"] for (coef, *_), r in zip(terms, recs))
        if v == v:
            diffs.append(v)
            users.append(t["user_id"])
    return diffs, users


def lin(by: dict, items: list[dict], terms: list[Term], *, n_perm: int = C.PERMUTATIONS, inference: bool = True) -> dict[str, Any]:
    """Mean over items of a per-item linear contrast of scores, with a user-clustered bootstrap CI and a user-level
    sign-flip p."""
    diffs, users = per_item(by, items, terms)
    out: dict[str, Any] = {"estimate": mean(diffs), "n_items": len(diffs), "n_users": len(set(users)),
                           "share_positive": mean([1.0 if v > 0 else 0.0 for v in diffs]),
                           "sd": statistics.stdev(diffs) if len(diffs) > 1 else float("nan")}
    if inference:
        boot = cluster_bootstrap(mean, diffs, users, seed=C.RNG_SEED)
        out.update({"ci": [boot["ci_low"], boot["ci_high"]],
                    "perm_p": pair_permutation(diffs, n=n_perm, seed=C.RNG_SEED, cluster_ids=users), "n_perm": n_perm})
    return out


# ---------------- analysis: categories (design §6.4-§6.5) ----------------

ABSENT = "absent"
GROWS = "grows under the task-blind arm (predicted by neither; reported, not interpreted)"
RELEVANCE = "relevance-like (reading (a))"
AUTHORIZATION = "authorization-like (reading (b))"
BOTH = "both: part authorization-like, part relevance-like"
AUTH_PART = "authorization-like part shown; whether any of the component persists is not determined"
REL_PART = "relevance-like part shown; whether any of the component is removed is not determined"
OVERSHOOT = ("overshoot: the task-blind arm removes more than the component, so the persisting part has the opposite "
             "sign (consistent with copying of the demonstrated minimal pairs, not with an authorization belief); "
             "reported, not interpreted")
REMOVAL_SHOWN = ("authorization-like (reading (b))", "both: part authorization-like, part relevance-like",
                 "authorization-like part shown; whether any of the component persists is not determined")
INCONCLUSIVE = "inconclusive"
NOT_COMPUTABLE = "not computable"


def _nan(b: dict) -> bool:
    return any(v != v for v in (b["estimate"], *b["ci"]))


def negligible(b: dict, delta: float) -> bool:
    return -delta < b["ci"][0] and b["ci"][1] < delta


def material(b: dict, p_holm: float, delta: float, sign: int) -> bool:
    lo = min(sign * b["ci"][0], sign * b["ci"][1])
    return sign * b["estimate"] >= delta and lo > 0 and p_holm < P_MAX


def grows(b: dict, p_holm: float, delta: float, sign: int) -> bool:
    hi = max(sign * b["ci"][0], sign * b["ci"][1])
    return sign * b["estimate"] <= -delta and hi < 0 and p_holm < P_MAX


def component_category(x_main: dict, z: dict, r: dict, pz: float, pr: float, delta: float) -> str:
    """One component's category from its arm (i) value, the removed part Z and the persisting part R, with the
    Holm-adjusted p of Z and R (design §6.4)."""
    if any(_nan(b) for b in (x_main, z, r)) or pz != pz or pr != pr:
        return NOT_COMPUTABLE
    if negligible(x_main, delta):
        return ABSENT
    sign = 1 if x_main["estimate"] > 0 else -1
    if grows(z, pz, delta, sign):
        return GROWS
    if grows(r, pr, delta, sign):  # review 1 (M2): the persisting part materially of the opposite sign
        return OVERSHOOT
    zm, zn = material(z, pz, delta, sign), negligible(z, delta)
    rm, rn = material(r, pr, delta, sign), negligible(r, delta)
    if zn and rm:
        return RELEVANCE
    if zm and rn:
        return AUTHORIZATION
    if zm and rm:
        return BOTH
    if zm:
        return AUTH_PART
    if rm and not zn:
        return REL_PART
    return INCONCLUSIVE


SUPPRESSION_CAVEAT = (" The naming component's removal is not shown to be negligible, so the removal may be a general "
                      "suppression of task effects by the task-blind statement and demonstration rather than an "
                      "authorization belief; the two are not separated.")
NO_CONTROL = (" With no naming component in this model, a general suppression of task effects by the arm is not "
              "excluded.")
COPYING = ("The request component's removed part is not carried by its aligned half (aligned minus mention-aligned), "
           "so it lies on the demonstrated request lines (swapped minus mention-swapped): it is not separated from "
           "copying of the demonstrated minimal pairs and is not read as an authorization belief.")
NAMING_ONLY = ("The naming component is removed while the request component persists: the control moves and the target "
               "does not. Neither explanation of the request component predicts this; the request component is read "
               "as relevance-like and the naming change is reported, not interpreted.")


def _fmt(b: dict) -> str:
    return f"{b['estimate']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]"


def aligned_half_shown(w: dict) -> bool:
    """Review 1 (M2): an authorization belief refuses every unrequested write, so its removal lies in the aligned half
    of W (aligned minus mention-aligned). Shown when that half's removed part is at least half of W's removed part
    and its CI excludes 0, in the direction of W in arm (i)."""
    zh, zw = w["Z_aligned_half"], w["Z"]
    if any(_nan(b) for b in (zh, zw, w["main"])):
        return False
    s = 1 if w["main"]["estimate"] > 0 else -1
    return s * zh["estimate"] >= s * zw["estimate"] / 2 and min(s * zh["ci"][0], s * zh["ci"][1]) > 0


def model_reading(w: str, m: str, parts: dict[str, dict] | None = None, aligned_half: bool | None = True) -> str:
    """The per-model reading from the request (W) and naming (M) categories (design §6.5 as amended after review 1):
    an authorization reading needs the removal in the aligned half of W; a shown removal of W carries the
    general-suppression caveat unless the naming component is relevance-like, and the no-control sentence when the
    model has no naming component."""
    parts = parts or {}
    z = _fmt(parts["Z"]) if "Z" in parts else "(see Z)"
    r = _fmt(parts["R"]) if "R" in parts else "(see R)"
    removed = w in REMOVAL_SHOWN
    if removed and aligned_half is False:
        text = f"Request component: {w} (removed part {z} nats). " + COPYING
    elif w == RELEVANCE and m in (RELEVANCE, ABSENT):
        text = ("(a) throughout: neither the request nor the naming effect behaves like a belief about request-matching "
                "authorization; both persist when the backend is stated and shown to ignore the task.")
    elif w == RELEVANCE and m in REMOVAL_SHOWN:
        text = NAMING_ONLY
    elif w == AUTHORIZATION and m == RELEVANCE:
        text = ("(a) for naming, (b) for the request: the write request's extra effect disappears under the task-blind "
                "backend while the naming effect persists, so the extra effect is read as a belief that the backend "
                "refuses writes the user did not request.")
    elif w == AUTHORIZATION and m == AUTHORIZATION:
        text = ("Both disappear: consistent with an authorization belief throughout, or with the task-blind statement and "
                "demonstration suppressing task effects in general; the two are not separated.")
    elif w == AUTHORIZATION and m == ABSENT:
        text = "The request component disappears."
    elif w == BOTH:
        text = (f"Part of the request component (removed part {z} nats) behaves like an authorization belief and part "
                f"(persisting part {r} nats) like a relevance bias; the naming component is {m}.")
    elif w in (AUTH_PART, REL_PART):
        text = f"Request component: {w} (removed part {z}, persisting part {r} nats); the naming component is {m}."
    else:  # the fallback of §6.5: every other combination (review 1, S4)
        text = f"Reported per component; no combined reading (request: {w}; naming: {m})."
    if removed and m != RELEVANCE:
        if m == ABSENT:
            text += NO_CONTROL
        elif "not separated" not in text:
            text += SUPPRESSION_CAVEAT
    return text


def class_note(by_class: dict[str, dict] | None) -> tuple[dict[str, Any], str]:
    """Review 1 (S2): each component's arm (i) value, removed and persisting parts per class, and a fixed sentence
    when the two classes' arm (i) values have opposite signs with both CIs excluding 0."""
    if not by_class:
        return {}, ""
    table = {c: {x: {k: _fmt(blk[x][k]) for k in ("main", "Z", "R")} for x in PRIMARY_COMPONENTS} for c, blk in by_class.items()}
    notes = []
    for x, name in (("W", "request"), ("M", "naming")):
        mains = [by_class[c][x]["main"] for c in CLASSES if c in by_class]
        if len(mains) == 2 and not any(_nan(b) for b in mains):
            pos = [b["ci"][0] > 0 for b in mains]
            neg = [b["ci"][1] < 0 for b in mains]
            if (pos[0] and neg[1]) or (neg[0] and pos[1]):
                notes.append(f" The pooled {name} component averages opposite class effects (status {_fmt(mains[0])}, "
                             f"unavailable {_fmt(mains[1])}); see per class.")
    return table, "".join(notes)


def removed_shares(by: dict, items: list[dict], action: str = "fail", B: int = C.BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Review 1 (M1), descriptive: the share of each component that the task-blind arm removes, Z_W / W_main and
    Z_M / M_main (ratios of means), and their difference, with intervals from the same user resamples. A uniform
    suppression of task effects predicts equal shares; a request-specific belief a larger share for W."""
    rows: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for t in items:
        vals = []
        for terms in (q_terms("W", "main", action), z_terms("W", action), q_terms("M", "main", action), z_terms("M", action)):
            d, _ = per_item(by, [t], terms)
            vals.append(d[0] if d else None)
        if None not in vals:
            rows[t["user_id"]].append(tuple(vals))

    def shares(groups):
        sums = [sum(r[k] for g in groups for r in rows[g]) for k in range(4)]
        sw = sums[1] / sums[0] if sums[0] else float("nan")
        sm = sums[3] / sums[2] if sums[2] else float("nan")
        return sw, sm, sw - sm

    users = sorted(rows)
    out: dict[str, Any] = {"n_items": sum(len(v) for v in rows.values()), "n_users": len(users)}
    if not users:
        return {**out, **{k: {"estimate": float("nan"), "ci": [float("nan")] * 2} for k in ("W", "M", "W_minus_M")}}
    point = shares(users)
    rng = random.Random(C.RNG_SEED)
    boots = [shares(rng.choices(users, k=len(users))) for _ in range(B)]
    for i, k in enumerate(("W", "M", "W_minus_M")):
        xs = sorted(b[i] for b in boots if b[i] == b[i])
        ci = [xs[int(0.025 * len(xs))], xs[min(len(xs) - 1, int(0.975 * len(xs)))]] if xs else [float("nan")] * 2
        out[k] = {"estimate": point[i], "ci": ci}
    return out


def comprehension(by: dict, comp_items: list[dict], arm: str) -> dict[str, Any]:
    """Per question: the share of items whose more probable answer is the correct one for this arm, and the mean
    log-odds of the correct answer; the arm passes when every question reaches 80% (design §3.5)."""
    out: dict[str, Any] = {"questions": {}}
    for k in COMPREHENSION:
        yes = EXPECTED[arm][k]
        vals = [by[key]["L"] for t in comp_items if (key := (t["id"], arm, "fail", k, "neutral")) in by]
        correct = [(v > 0) if yes else (v < 0) for v in vals]
        share = mean([1.0 if c else 0.0 for c in correct])
        out["questions"][k] = {"correct_answer": "Yes" if yes else "No", "n": len(vals), "share_correct": share,
                               "mean_logodds_of_correct": mean([v if yes else -v for v in vals]),
                               "passes": bool(vals) and share >= COMPREHENSION_SHARE}
    out["passes"] = all(q["passes"] for q in out["questions"].values())
    return out


def positive_control(by: dict, items: list[dict]) -> dict[str, Any]:
    """The witness's aligned - swapped in the authorization arm minus that in the task-blind arm (design §4)."""
    blk = lin(by, items, q_terms("G", "auth", "witness") + q_terms("G", "blind", "witness", sign=-1))
    est, lo = blk["estimate"], blk["ci"][0]
    blk["passes"] = blk["n_items"] > 0 and est == est and lo == lo and est >= PC_MIN_NATS and lo > 0
    return blk


def primary_block(by: dict, items: list[dict], n_perm: int = N_PERM_PRIMARY, action: str = "fail",
                  inference_perm: bool = True) -> dict[str, dict]:
    """For W and M: the arm (i) value, the removed part Z and the persisting part R (first token)."""
    out = {}
    for x in PRIMARY_COMPONENTS:
        out[x] = {"main": lin(by, items, q_terms(x, "main", action)),
                  "Z": lin(by, items, z_terms(x, action), n_perm=n_perm),
                  "R": lin(by, items, q_terms(x, "blind", action), n_perm=n_perm)}
    # review 1 (M2), prespecified: W's removed part split into its aligned half (aligned - mention-aligned) and its
    # swapped half (minus the removed part of swapped - mention-swapped), which add up to it; and the removed parts of
    # aligned - neutral and swapped - neutral
    out["W"]["Z_aligned_half"] = lin(by, items, z_terms("AMA", action))
    out["W"]["Z_swapped_half"] = lin(by, items, z_terms("SMS", action, sign=-1))
    out["W"]["Z_aligned_minus_neutral"] = lin(by, items, z_terms("AN", action))
    out["W"]["Z_swapped_minus_neutral"] = lin(by, items, z_terms("S", action))
    return out


def holm_family(primary: dict[str, dict]) -> dict[str, float]:
    """Holm over the eight primary tests (Z and R of W and M, both models); a missing model, or a p that is not a
    number, counts p = 1, so the family always has eight members (review 1, S1)."""
    def p_of(m: str, x: str, part: str) -> float:
        p = primary[m][x][part].get("perm_p", float("nan")) if m in primary else 1.0
        return p if p == p else 1.0
    raw = {f"{m}|{x}|{part}": p_of(m, x, part) for m in P.CO_PRIMARY for x in PRIMARY_COMPONENTS for part in PRIMARY_PARTS}
    return holm(raw)


def decide(primary: dict[str, dict], gates: dict[str, dict], margins: dict[str, float] | None = None,
           by_class: dict[str, dict] | None = None, shares: dict[str, dict] | None = None) -> dict[str, Any]:
    """Per-model categories and reading (design §6 as amended after review 1), with Holm over the eight primary tests,
    the gates, the aligned-half condition, the per-class values and the removed shares."""
    margins = margins or MARGIN
    by_class, shares = by_class or {}, shares or {}
    adj = holm_family(primary)
    out: dict[str, Any] = {"holm_adjusted_p": adj, "per_model": {}}
    for m in P.CO_PRIMARY:
        if m not in primary:
            out["per_model"][m] = {"reading": "not scored"}
            continue
        d = margins[m]
        cats = {x: component_category(primary[m][x]["main"], primary[m][x]["Z"], primary[m][x]["R"],
                                      adj[f"{m}|{x}|Z"], adj[f"{m}|{x}|R"], d) for x in PRIMARY_COMPONENTS}
        g = gates.get(m, {})
        failed = []
        if not g.get("comprehension_blind_passes"):
            failed.append("the model is not shown to understand that the task-blind backend ignores the task "
                          "(comprehension check of the task-blind arm)")
        if not g.get("positive_control_passes"):
            failed.append("the model's predictions are not shown to follow the stated backend (positive control)")
        half = aligned_half_shown(primary[m]["W"]) if "Z_aligned_half" in primary[m]["W"] else None
        table, note = class_note(by_class.get(m))
        entry: dict[str, Any] = {"margin": d, "ungated_categories": cats, "gates": g, "aligned_half_shown": half,
                                 "per_class": table}
        if m in shares:
            sh = shares[m]
            entry["removed_shares"] = sh
            share_line = (f" Share removed by the task-blind arm (descriptive): request {_fmt(sh['W'])}, naming "
                          f"{_fmt(sh['M'])}; a uniform suppression predicts equal shares.") if "W" in sh else ""
        else:
            share_line = ""
        if failed:
            entry["categories"] = {x: "inconclusive (gate)" for x in cats}
            entry["reading"] = "Inconclusive: " + "; ".join(failed) + "."
        else:
            entry["categories"] = cats
            entry["reading"] = model_reading(cats["W"], cats["M"], primary[m]["W"], half) + note + share_line
        out["per_model"][m] = entry
    return out


# ---------------- analysis: blocks ----------------


def format_block(by: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"mass_median": {}, "mass_below_0.5_share": {}, "seconds_mean": {}}
    groups: dict[str, list[float]] = defaultdict(list)
    secs: dict[str, list[float]] = defaultdict(list)
    for k, x in by.items():
        kind = "comp" if k[3] in COMPREHENSION else k[3]
        groups[f"{k[1]}/{kind}"].append(math.exp(x["logp_plus"]) + math.exp(x["logp_minus"]))
        if x.get("seconds") is not None:
            secs[f"{k[1]}/{kind}"].append(x["seconds"])
    for g, ms in sorted(groups.items()):
        out["mass_median"][g] = statistics.median(ms)
        out["mass_below_0.5_share"][g] = sum(v < 0.5 for v in ms) / len(ms)
    out["seconds_mean"] = {g: mean(v) for g, v in sorted(secs.items())}
    out["red_flag"] = any(v < 0.5 for v in out["mass_median"].values())
    return out


def model_block(by: dict, items: list[dict], comp_items: list[dict], n_perm_primary: int = N_PERM_PRIMARY) -> dict[str, Any]:
    out: dict[str, Any] = {"primary": primary_block(by, items, n_perm_primary)}
    out["positive_control"] = positive_control(by, items)
    out["comprehension"] = {arm: comprehension(by, comp_items, arm) for arm in ARMS}
    out["gates"] = {"comprehension_blind_passes": out["comprehension"][GATED_ARM]["passes"],
                    "positive_control_passes": out["positive_control"]["passes"]}
    # secondary (design §7): no Holm family, no verdict
    out["witness_decomposition"] = primary_block(by, items, C.PERMUTATIONS, action="witness")
    out["per_arm"] = {arm: {a: {q: lin(by, items, q_terms(q, arm, a)) for q in QUANT} for a in ACTIONS} for arm in ARMS}
    out["per_arm_yn"] = {arm: {a: {q: lin(by, items, q_terms(q, arm, a, "yn")) for q in ("G", "S", "AN")} for a in ACTIONS}
                         for arm in ARMS}
    out["yn_goal_effect_split"] = {a: {"Z": lin(by, items, z_terms("G", a, "yn")), "R": lin(by, items, q_terms("G", "blind", a, "yn"))}
                                   for a in ACTIONS}
    out["level_changes"] = {
        f"{x}_minus_{y}": {a: {c: lin(by, items, [(1, x, a, "first", c), (-1, y, a, "first", c)]) for c in FIRST_LINES}
                           for a in ACTIONS}
        for x, y in (("blind", "main"), ("auth", "blind"))}
    out["auth_signature"] = {"witness_M_auth": lin(by, items, q_terms("M", "auth", "witness")),
                             "witness_G_auth": lin(by, items, q_terms("G", "auth", "witness"))}
    out["by_class"] = {c: primary_block(by, [t for t in items if t["cls"] == c], C.PERMUTATIONS) for c in CLASSES}
    out["removed_shares"] = removed_shares(by, items)
    out["format"] = format_block(by)
    return out


def reproducibility(by: dict, reference: dict) -> dict[str, Any]:
    """Arm (i) here against the confirmatory scores of the same jobs (byte-identical prompts): share within 0.13 nats
    (one bf16 step on this scale) and the largest difference."""
    gaps = [abs(x["L"] - reference[(k[0], "main", k[2], k[3], k[4])]["L"])
            for k, x in by.items() if k[1] == "main" and (k[0], "main", k[2], k[3], k[4]) in reference]
    return {"n": len(gaps), "share_within_0.13": mean([1.0 if g <= 0.13 else 0.0 for g in gaps]),
            "max_abs_diff": max(gaps) if gaps else float("nan")}


def _reference_scores(cfg: dict) -> dict[str, dict] | None:
    ref = cfg.get("reference_run")
    if not ref:
        return None
    root = Path(ref) if os.path.isabs(ref) else runs_root() / ref
    out: dict[str, dict] = defaultdict(dict)
    for p in sorted(root.glob("shard_*_of_*/scoring.jsonl")):
        for s in read_jsonl(p):
            out[s["model"]][(s["transition_id"], s.get("arm", "main"), s.get("action", "fail"), s["readout"], s["condition"])] = s
    return out


def projection(target_run: str, report: dict, smoke_cfg: dict, time_limit: float) -> dict[str, Any]:
    """From the smoke run: hours per shard of the full run (both models in one job), from the mean seconds per job by
    arm and readout kind; no estimate."""
    full = {**load_config(target_run), "build_run": smoke_cfg.get("build_run", "tau_build")}
    items, comp, _ = chosen_items(full)
    n = int(full.get("n_shards", 1))
    out: dict[str, Any] = {"run": target_run, "n_shards": n, "time_limit_hours": time_limit, "models": {}}
    for m, blk in report["models"].items():
        sec = blk["format"]["seconds_mean"]
        hours = []
        for i in range(n):
            jobs = all_jobs(P.shard_by_user(items, i, n), comp)
            hours.append(sum(sec.get(f"{j[0]}/{'comp' if j[2] in COMPREHENSION else j[2]}", float("nan"))
                             for _, j in jobs) / 3600)
        out["models"][m] = {"hours_max_shard": max(hours), "hours_all_shards": sum(hours)}
    out["hours_max_shard_both_models"] = sum(v["hours_max_shard"] for v in out["models"].values())
    out["gpu_hours_full_run"] = sum(v["hours_all_shards"] for v in out["models"].values())
    out["fits_time_limit"] = out["hours_max_shard_both_models"] <= time_limit
    return out


def run_tau_auth_analysis(cfg: dict) -> int:
    from wf.rq2 import complete_shards

    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(p / "scoring.jsonl" for p in folders if (p / "scoring.jsonl").exists()))
    if problems:
        write_status(run_id, "waits for every scoring shard:\n" + "\n".join(problems), ok=False)
        return 1
    scfg = {**load_config(cfg["source"]), "build_run": cfg.get("build_run", "tau_build")}
    items, comp, _ = chosen_items(scfg)
    comp_items = [t for t in items if t["id"] in comp]
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    hashes = {t["id"]: P.item_hash(t) for t in items}
    stale = sum(1 for s in scores if hashes.get(s["transition_id"]) != s.get("item_hash"))
    models = tuple(cfg.get("models", P.CO_PRIMARY))
    wanted = {(m, t["id"], *j) for m in models for t, j in all_jobs(items, comp)}
    got = {score_key(s) for s in scores}
    missing = len(wanted - got)
    ok = missing == 0 and stale == 0
    report: dict[str, Any] = {"items": len(items), "users": len({t["user_id"] for t in items}),
                              "comprehension_items": len(comp_items), "by_class": {c: sum(t["cls"] == c for t in items) for c in CLASSES},
                              "missing_scores": missing, "stale_scores": stale, "models": {}}
    report["token_checks"] = {f.stem[len("tokens_"):]: json.loads(f.read_text(encoding="utf-8"))
                              for p in folders for f in sorted(p.glob("tokens_*.json"))}
    by_model = {m: {(s["transition_id"], s["arm"], s["action"], s["readout"], s["condition"]): s for s in scores if s["model"] == m}
                for m in models}
    if cfg.get("project_to"):
        # the smoke run: format, timing, the spread of the decisive contrasts and the two gates; no estimate, CI, p or
        # category of any contrast
        for m in models:
            by = by_model[m]
            report["models"][m] = {"format": format_block(by),
                                   "sd_of_per_item": {name: {"n": len(v[0]), "sd": statistics.stdev(v[0]) if len(v[0]) > 1 else float("nan")}
                                                      for name, v in (("W_main", per_item(by, items, q_terms("W", "main", "fail"))),
                                                                      ("Z_W", per_item(by, items, z_terms("W", "fail"))),
                                                                      ("R_W", per_item(by, items, q_terms("W", "blind", "fail"))))}}
            # review 1 (S3): the gates may be read on the smoke run (they are not outcomes); a change made after
            # reading them is a logged deviation
            pc = positive_control(by, items)
            report["models"][m]["gates_readable"] = {
                "comprehension": {arm: comprehension(by, comp_items, arm) for arm in ARMS},
                "positive_control": {"pc_nats": pc["estimate"], "pc_ci": pc["ci"], "passes": pc["passes"], "n_items": pc["n_items"]}}
        report["projection"] = projection(cfg["project_to"], report, scfg, float(cfg["time_limit_hours"]))
        write_json(d / "metrics.json", report)
        write_run_report(run_id, "τ-bench authorization experiment, smoke run: format and timing only (no estimates)", report)
        pr = report["projection"]
        write_status(run_id, f"items={report['items']} missing={missing} stale={stale}\nprojected hours per shard, both "
                             f"models: {pr['hours_max_shard_both_models']:.2f} (limit {pr['time_limit_hours']}; fits: "
                             f"{pr['fits_time_limit']}); full run {pr['gpu_hours_full_run']:.2f} GPU-hours",
                     ok=False, reported="τ-bench authorization smoke: format and timing" if ok else None)
        return 0 if ok else 1
    reference = _reference_scores(cfg)
    primary, gates = {}, {}
    for m in models:
        blk = model_block(by_model[m], items, comp_items, int(cfg.get("n_perm_primary", N_PERM_PRIMARY)))
        if reference is not None:
            blk["reproducibility_main_arm"] = reproducibility(by_model[m], reference.get(m, {}))
        report["models"][m] = blk
        primary[m] = blk["primary"]
        gates[m] = blk["gates"]
    report["decision"] = decide(primary, gates, by_class={m: report["models"][m]["by_class"] for m in models},
                                shares={m: report["models"][m]["removed_shares"] for m in models})
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "τ-bench retail: relevance bias or a belief in request-matching authorization?", report)
    lines = [f"{m}: W {_fmt(primary[m]['W']['main'])} -> Z {_fmt(primary[m]['W']['Z'])}, R {_fmt(primary[m]['W']['R'])}; "
             f"categories {report['decision']['per_model'][m]['categories']}" for m in models]
    write_status(run_id, f"items={report['items']} missing={missing} stale={stale}\n" + "\n".join(lines), ok=False,
                 reported="τ-bench authorization decision" if ok else None)
    return 0 if ok else 1
