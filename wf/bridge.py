"""Natural ALFWorld bridge (exploratory, frozen procedure; notes/bridge_design.md).

On a stratified 1,500-item subsample of RQ2's 4,666 determined failures, each item is scored under the aligned and
the swapped task line with:

- provenance statements after the action line: none; a random tester (two wordings); an agent that has seen only
  the history (two wordings); a positive control (an agent that only proposes actions that will work); and a
  content-free sentence of similar length in the same place (the placebo);
- existence settled without naming the object ("Every object any task in this house could need is present in the
  room.") appended to the task line in both prompts, and the named existence sentence of the earlier dissociation
  test ("There is a {object} somewhere in the room.") as its reference;

each read three ways (two-candidate, first token, yes/no), plus yes/no comprehension questions (two per provenance
statement, on a 300-item subset) and an existence check naming the object. Qwen3-8B is scored in plain text (no chat
template) and, on the 300-item subset, also in RQ2's thinking-off chat format; Llama-3.1-8B in its chat template.

Scoring uses wf.scoring.prefix.PrefixScorer: per item and task line the shared prompt prefix is run once and every
statement, readout and question runs from its cache (batch size 1). Runs are sharded by game, save per item, resume,
refuse a dirty checkout and refuse a GPU of another type than the configuration names.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wf import constants as C
from wf import rq1
from wf.rq1v import YN_QUESTION
from wf.rq2 import SOURCES, complete_shards, load_items, primary_set, shard_items
from wf.runs import _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.util import append_jsonl, git_dirty, is_mock, read_jsonl, require_filed, runs_root, write_json, write_status

# ---------------- texts (frozen with the design) ----------------

STATEMENTS = {
    "tester_a": "A tester picked this action at random from the possible actions.",
    "tester_b": "A tester chose this action by drawing it at random from a list of commands, without looking at the room.",
    "agent_a": "An agent who has seen only what is shown above chose this action.",
    "agent_b": "This action was chosen by an agent whose only information about the room is the history shown above.",
    "positive": "This action was chosen by an agent who only proposes actions that will work.",
    "placebo": "This command is written in lower-case letters, like every command in this game.",
}
WORDINGS = {"tester": ("tester_a", "tester_b"), "agent": ("agent_a", "agent_b")}
EXIST_UNNAMED = "Every object any task in this house could need is present in the room."
PROV_ARMS = ("none",) + tuple(STATEMENTS)
EXIST_ARMS = ("exist", "named")
ARMS = PROV_ARMS + EXIST_ARMS
TASKS = ("aligned", "swapped")

C1 = ("Question: did the choice of this action depend on anything about the room that is not shown above? "
      "Answer Yes or No.\nAnswer:")
C2 = "Question: could this action have been chosen even if it will not work? Answer Yes or No.\nAnswer:"
COMP_ARMS = ("none", "tester_a", "tester_b", "agent_a", "agent_b", "positive")
STATED = {"tester_a": {"c1": "No", "c2": "Yes"}, "tester_b": {"c1": "No", "c2": "Yes"},
          "agent_a": {"c1": "No", "c2": "Yes"}, "agent_b": {"c1": "No", "c2": "Yes"},
          "positive": {"c1": "Yes", "c2": "No"}}
GATED = ("tester_a", "tester_b", "agent_a", "agent_b")  # the positive control's answers are reported, not gated


def exist_question(obj: str) -> str:
    return f"Question: is there {rq1._article(obj)} {obj} in the room? Answer Yes or No.\nAnswer:"


# ---------------- items ----------------

N_SAMPLE = 1500
N_SUBSET = 300
N_SMOKE = 20


def h(item_id: str, salt: str) -> str:
    return hashlib.sha256(f"{C.RNG_SEED}|{salt}|{item_id}".encode()).hexdigest()


def history_bin(n: int) -> str:
    return "0" if n == 0 else "1-3" if n <= 3 else "4-7" if n <= 7 else "8+"


def stratum(t: dict) -> tuple[str, str]:
    return history_bin(int(t.get("history_len") or 0)), t.get("determination_reason") or "unknown"


def sample(primary: list[dict], n: int = N_SAMPLE) -> list[dict]:
    """Proportional stratified sample by history-length bin and failure reason: each stratum gets its share of n
    rounded down, the remaining places go to the largest remainders (ties by stratum name), and within a stratum
    items are taken in h(id, "bridge-sample") order. Returned in that hash order overall."""
    by = defaultdict(list)
    for t in primary:
        by[stratum(t)].append(t)
    total = len(primary)
    exact = {k: n * len(v) / total for k, v in by.items()}
    quota = {k: int(math.floor(x)) for k, x in exact.items()}
    left = n - sum(quota.values())
    for k in sorted(by, key=lambda k: (-(exact[k] - quota[k]), k))[:left]:
        quota[k] += 1
    chosen = []
    for k, items in by.items():
        chosen += sorted(items, key=lambda t: h(t["id"], "bridge-sample"))[: quota[k]]
    return sorted(chosen, key=lambda t: h(t["id"], "bridge-sample"))


def subset(items: list[dict], n: int = N_SUBSET) -> list[dict]:
    """The 300-item subset: comprehension questions, Qwen's chat-format link and the cross-GPU rescore."""
    return sorted(items, key=lambda t: h(t["id"], "bridge-subset"))[:n]


def item_sets(primary: list[dict]) -> dict[str, list[dict]]:
    s = sample(primary)
    sub = subset(s)
    return {"sample": s, "subset": sub, "smoke": sub[:N_SMOKE]}


# ---------------- prompts ----------------

def task_line(t: dict, task: str, arm: str) -> str:
    line = t["task_line"] if task == "aligned" else t["swapped_task_line"]
    if not line:
        raise ValueError(f"{t['id']}: no {task} task line")
    if arm == "exist":
        return f"{line} {EXIST_UNNAMED}"
    if arm == "named":
        return f"{line} {rq1.mention_sentence(t['goal_object'])}"
    return line


def messages(t: dict, task: str, arm: str, ending: str = "obs") -> list[dict[str, str]]:
    """RQ2's prompt (example set 1, history only) with the arm's task line and statement. `ending` is "obs" (the
    prompt ends "Next observation:"), "yn" (RQ2's yes/no question), "c1", "c2" or "exist_check"."""
    if arm not in ARMS:
        raise ValueError(arm)
    msgs = build_messages(t, "aligned", task_line=task_line(t, task, arm), note=STATEMENTS.get(arm))
    if ending == "obs":
        return msgs
    q = {"yn": YN_QUESTION, "c1": C1, "c2": C2, "exist_check": exist_question(t["goal_object"])}[ending]
    last = msgs[-1]
    if not last["content"].endswith("Next observation:"):
        raise ValueError("prompt does not end with 'Next observation:'")
    return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + q}]


def plain_text(msgs: list[dict]) -> str:
    """No chat template (the pipeline audit's plain-text format, scripts/qwen_audit/followup_score.py): the system
    text, each example as its user text, one space and its answer, then the query's user text, separated by blank
    lines. A continuation follows after one space."""
    parts, i = [], 0
    if msgs and msgs[0]["role"] == "system":
        parts.append(msgs[0]["content"])
        i = 1
    while i < len(msgs):
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            parts.append(f"{msgs[i]['content']} {msgs[i + 1]['content']}")
            i += 2
        else:
            parts.append(msgs[i]["content"])
            i += 1
    return "\n\n".join(parts)


def jobs_for(t: dict, in_subset: bool) -> list[tuple[str, str, str]]:
    """(task, arm, readout) for one item. Readouts: cand (two-candidate and first token), yn, c1, c2, exist_check."""
    out = []
    for task in TASKS:
        for arm in PROV_ARMS:
            out += [(task, arm, "cand"), (task, arm, "yn")]
        out.append((task, "none", "exist_check"))
        out += [(task, "exist", "cand"), (task, "exist", "yn"), (task, "exist", "exist_check")]
        out += [(task, "named", "cand"), (task, "named", "yn")]
        if in_subset and task == "aligned":
            out += [(task, arm, q) for arm in COMP_ARMS for q in ("c1", "c2")]
    return out


def group_of(task: str, arm: str) -> tuple[str, str]:
    """Jobs sharing a task line share a stem: the provenance arms share one, each existence arm has its own."""
    return task, (arm if arm in EXIST_ARMS else "prov")


# ---------------- tokenization ----------------

class Renderer:
    """Token ids of prompts and continuations in one format: "plain" (no template; continuation after one space),
    "chat" (the model's template, as RQ2's scorer renders it) or "block" (Qwen3's template with thinking off, which
    puts the empty think block before the answer; identical to "chat" for Qwen3, named apart for the records)."""

    def __init__(self, scorer: LogoddsScorer, fmt: str):
        if fmt not in ("plain", "chat", "block"):
            raise ValueError(fmt)
        self.sc, self.fmt, self.tok = scorer, fmt, scorer._tok

    def _ids(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def prompt(self, msgs: list[dict]) -> list[int]:
        if self.fmt == "plain":
            return self._ids(plain_text(msgs))
        enc = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           **self.sc.template_settings())
        ids = enc["input_ids"]
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)

    def cont(self, msgs: list[dict], prompt_ids: list[int], text: str) -> list[int]:
        """A continuation's ids. Plain: from the joint tokenization of prompt + " " + text when the prompt's ids are
        its prefix, else the ids of " " + text (as the audit's plain_pair). Chat: the text tokenized alone, as
        RQ2's scorer does."""
        if self.fmt == "plain":
            joint = self._ids(plain_text(msgs) + " " + text)
            if joint[: len(prompt_ids)] == prompt_ids:
                return joint[len(prompt_ids):]
            return self._ids(" " + text)
        return self._ids(text)

    def yes_no(self) -> tuple[int, int]:
        words = (" Yes", " No") if self.fmt == "plain" else ("Yes", "No")
        ids = [self._ids(w) for w in words]
        if any(len(i) != 1 for i in ids) or ids[0] == ids[1]:
            raise ValueError(f"{self.fmt}: Yes/No are not two single tokens: {ids}")
        return ids[0][0], ids[1][0]


def build_sequences(t: dict, jobs: list[tuple[str, str, str]], r: Renderer) -> dict[tuple, list[tuple]]:
    """Per group: [(job, [(prompt ids, continuation ids, probe ids), ...])]. A cand job has two sequences: the
    success text with both first tokens as probes, and "Nothing happens."; a yes/no-type job one sequence with the
    Yes and No tokens as probes."""
    yes, no = r.yes_no()
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for task, arm, readout in jobs:
        if readout == "cand":
            msgs = messages(t, task, arm, "obs")
            p = r.prompt(msgs)
            plus = r.cont(msgs, p, t["candidates"]["o_plus"])
            minus = r.cont(msgs, p, t["candidates"]["o_minus"])
            if not plus or not minus or plus[0] == minus[0]:
                raise ValueError(f"{t['id']}: candidates empty or with the same first token")
            seqs = [(p, plus, [plus[0], minus[0]]), (p, minus, [])]
        else:
            msgs = messages(t, task, arm, readout)
            seqs = [(r.prompt(msgs), [], [yes, no])]
        groups[group_of(task, arm)].append(((task, arm, readout), seqs))
    return groups


# ---------------- scoring run ----------------

def _gpu_ok(want: str | None) -> tuple[bool, list[str] | None]:
    from wf.provenance import gpu_names

    names = gpu_names()
    if is_mock() or not want:
        return True, names
    return bool(names) and all(want.lower() in n.lower() for n in names), names


def _mock_scores(job: tuple, t: dict, fmt: str) -> dict:
    """Deterministic stand-in numbers for mock runs (pipeline tests only)."""
    x = int(hashlib.sha256(json.dumps([t["id"], job, fmt]).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    task, arm, readout = job
    if readout == "cand":
        lp, lm = -5.0 + 4 * x, -2.0
        return {"logp_plus": lp, "logp_minus": lm, "L": lp - lm, "logp_first_plus": lp / 3, "logp_first_minus": lm / 3,
                "L_first": (lp - lm) / 3, "first_mass": 0.9, "n_tokens_plus": 8, "n_tokens_minus": 3}
    ly, ln = math.log(0.2 + 0.6 * x), math.log(0.8 - 0.6 * x)
    return {"logp_yes": ly, "logp_no": ln, "L": ly - ln, "mass": math.exp(ly) + math.exp(ln)}


def score_item(t: dict, jobs: list, fmt: str, r: Renderer | None, engine: Any) -> list[dict]:
    recs = []
    if r is None:  # mock
        return [{"job": list(j), **_mock_scores(j, t, fmt)} for j in jobs]
    for (task, kind), entries in build_sequences(t, jobs, r).items():
        flat = [s for _, seqs in entries for s in seqs]
        scores = engine.score_group(flat)
        k = 0
        for job, seqs in entries:
            got = scores[k: k + len(seqs)]
            k += len(seqs)
            if job[2] == "cand":
                a, b = got
                lp, lm = sum(a.cont_logprobs), sum(b.cont_logprobs)
                f_plus, f_minus = a.probe_logprobs
                rec = {"logp_plus": lp, "logp_minus": lm, "L": lp - lm, "logp_first_plus": f_plus,
                       "logp_first_minus": f_minus, "L_first": f_plus - f_minus,
                       "first_mass": math.exp(f_plus) + math.exp(f_minus),
                       "n_tokens_plus": len(a.cont_logprobs), "n_tokens_minus": len(b.cont_logprobs)}
            else:
                ly, ln = got[0].probe_logprobs
                rec = {"logp_yes": ly, "logp_no": ln, "L": ly - ln, "mass": math.exp(ly) + math.exp(ln)}
            if not all(math.isfinite(v) for v in rec.values()):
                raise ValueError(f"non-finite score for {t['id']} {job}")
            recs.append({"job": list(job), "stem_len": got[0].stem_len, **rec})
    return recs


def token_checks(items: list[dict], fmt: str, r: Renderer, subset_ids: set[str]) -> dict:
    """Before scoring: every prompt of the run renders, every candidate is non-empty with distinct first tokens,
    Yes/No are single distinct tokens, aligned and swapped prompts of a job differ only where the task line differs
    (checked on the text), and plain-text continuations come from the joint tokenization (counted)."""
    problems, n_seq, n_fallback = [], 0, 0
    for t in items:
        jobs = jobs_for(t, t["id"] in subset_ids)
        try:
            groups = build_sequences(t, jobs, r)
        except ValueError as e:
            problems.append(str(e))
            continue
        n_seq += sum(len(s) for entries in groups.values() for _, s in entries)
        for task, arm, readout in jobs:
            if readout != "cand" or task != "aligned":
                continue
            ma, mb = messages(t, "aligned", arm), messages(t, "swapped", arm)
            a = (plain_text(ma) if fmt == "plain" else "\n".join(m["content"] for m in ma)).split("\n")
            b = (plain_text(mb) if fmt == "plain" else "\n".join(m["content"] for m in mb)).split("\n")
            diff = [(x, y) for x, y in zip(a, b) if x != y]
            if len(a) != len(b) or diff != [(task_line(t, "aligned", arm), task_line(t, "swapped", arm))]:
                problems.append(f"{t['id']}/{arm}: aligned and swapped prompts differ outside the task line")
        if fmt == "plain":
            msgs = messages(t, "aligned", "none")
            p = r.prompt(msgs)
            joint = r._ids(plain_text(msgs) + " " + t["candidates"]["o_plus"])
            n_fallback += joint[: len(p)] != p
    return {"n_items": len(items), "n_sequences": n_seq, "n_problems": len(problems), "problems": problems[:20],
            "plain_joint_tokenization_fallbacks": n_fallback}


def reference_agreement(t: dict, r: Renderer, model: Any) -> dict:
    """Smoke only: one item's cand and yn jobs for three arms, cached routine against a plain full forward pass."""
    from wf.scoring.prefix import PrefixScorer, full_pass_score

    jobs = [(task, arm, ro) for task in TASKS for arm in ("none", "tester_a", "exist") for ro in ("cand", "yn")]
    diffs = []
    for _, entries in build_sequences(t, jobs, r).items():
        flat = [s for _, seqs in entries for s in seqs]
        cached = PrefixScorer(model).score_group(flat)
        for (p, c, pr), sc in zip(flat, cached):
            ref = full_pass_score(model, p, c, pr)
            diffs += [abs(x - y) for x, y in zip(sc.cont_logprobs + sc.probe_logprobs,
                                                 ref.cont_logprobs + ref.probe_logprobs)]
    return {"n_logprobs": len(diffs), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs)}


def run_bridge_score(cfg: dict) -> int:
    """Config: model (Hugging Face id), tag, formats ({format: "sample" | "subset"}), items ("sample", "subset" or
    "smoke"), gpu_type, n_shards. A format marked "subset" is scored on the subset's items only."""
    run_id = cfg["run_id"]
    require_filed(run_id)
    if not is_mock() and git_dirty():
        write_status(run_id, "refused: the checkout has uncommitted changes", ok=False)
        return 1
    ok, names = _gpu_ok(cfg.get("gpu_type"))
    if not ok:
        write_status(run_id, f"refused: this job's GPU is {names}, the run is pinned to {cfg.get('gpu_type')}", ok=False)
        return 1
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    from wf.tools.verify_items import verified

    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    unverified = [s for s in sources if not verified(s)[0]]
    if problems or unverified:
        write_status(run_id, f"collection not finished or not verified: {problems + unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(runs_root() / s / "transitions.jsonl" for s in sources))
    sets = item_sets(primary_set(load_items(sources)))
    base = shard_items(sets[cfg.get("items", "sample")], i, n)
    sub_ids = {t["id"] for t in sets["subset"]}
    hashes = {t["id"]: _item_hash(t) for t in base}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["format"], s["transition_id"]) for s in saved}
    model_name, tag = cfg["model"], cfg["tag"]
    scorer = LogoddsScorer(model_name, model_tag=tag)
    model = None if scorer.mock else scorer._model
    from wf.scoring.prefix import PrefixScorer

    engine = None if model is None else PrefixScorer(model)
    timing, checks, agreement = {}, {}, {}
    for fmt, scope in cfg["formats"].items():
        items = [t for t in base if scope == "sample" or t["id"] in sub_ids]
        r = None
        if not scorer.mock:
            r = Renderer(scorer, fmt)
            checks[fmt] = token_checks(items, fmt, r, sub_ids)
            if checks[fmt]["n_problems"]:
                write_json(d / "token_checks.json", checks)
                write_status(run_id, f"token checks failed ({fmt}): {checks[fmt]['problems'][:3]}", ok=False)
                return 1
            if cfg.get("reference_agreement") and items:
                agreement[fmt] = reference_agreement(items[0], r, model)
        t0, n_done = time.time(), 0
        for t in items:
            if (fmt, t["id"]) in done:
                continue
            jobs = jobs_for(t, t["id"] in sub_ids)
            recs = score_item(t, jobs, fmt, r, engine)
            append_jsonl(d / "scoring.jsonl", [{**x, "transition_id": t["id"], "game_id": t["game_id"],
                                                "item_hash": hashes[t["id"]], "model": tag, "format": fmt,
                                                "model_revision": scorer.revision} for x in recs])
            n_done += 1
        timing[fmt] = {"items": n_done, "seconds": time.time() - t0}
    write_json(d / "token_checks.json", checks)
    write_json(d / "timing.json", {"timing": timing, "reference_agreement": agreement})
    scorer.close()
    write_status(run_id, f"shard {i} of {n}: {json.dumps(timing)}", ok=False, reported="shard complete")
    return 0
