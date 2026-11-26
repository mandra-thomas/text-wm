"""Post-training pilot (notes/posttraining_design.md): checkpoints, lineages, arms, conditions and prompts.

The items, sentences, controls, prompt builder and plain-text rendering are the lexsem diagnostic's (wf/lexsem.py,
scripts/lexsem/score.py), imported, not copied, so that a prompt here is byte-identical to lexsem's for the same item,
condition, format and tokenizer. This module adds the pilot's own conditions (the neutral task on every arm, the
described goal, the anchors and the neutral task on RQ2's determined successes), the Word2World bridge prompts, the
seven-example fallback prompts and the recognition check for the described goal.

Revisions are the Hub's `main` sha read on 2026-11-26 from https://huggingface.co/api/models/<id> before any download
(checked again on the cluster by scripts/posttrain/access_check.py, job 7347490), except those already pinned in
wf.constants.REVISIONS, which are the same values.
"""
from __future__ import annotations

from wf import lexsem, rq1, rq1x
from wf.constants import REVISIONS

# ---------------- checkpoints and lineages ----------------

CHECKPOINTS: dict[str, dict] = {
    "llama-base": {"repo": "meta-llama/Llama-3.1-8B", "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                   "stage": "base"},
    "tulu-sft": {"repo": "allenai/Llama-3.1-Tulu-3-8B-SFT", "revision": "f2a0b46b0cfda21003c6141b1ff837b7e165524d",
                 "stage": "Tulu 3 SFT"},
    "tulu-dpo": {"repo": "allenai/Llama-3.1-Tulu-3-8B-DPO", "revision": "a7beb67e33ffd01cc87ac3b46cadc1000985b8db",
                 "stage": "Tulu 3 DPO"},
    "tulu-rl": {"repo": "allenai/Llama-3.1-Tulu-3-8B", "revision": "666943798adbde0b1aff34626007e26986a3c107",
                "stage": "Tulu 3 RLVR (PPO)"},
    "tulu31-grpo": {"repo": "allenai/Llama-3.1-Tulu-3.1-8B", "revision": "46239c2d07db76b412e1f1b0b4542f65b81fe01f",
                    "stage": "Tulu 3.1 RLVR (GRPO)"},
    "llama-instruct": {"repo": "meta-llama/Llama-3.1-8B-Instruct",
                       "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "stage": "Meta Instruct"},
    "w2w-llama": {"repo": "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B",
                  "revision": "8f089d2126f0406a296747041ebf33b902837cd3", "stage": "Word2World domain SFT"},
    "qwen-base": {"repo": "Qwen/Qwen3-8B-Base", "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                  "stage": "base"},
    "qwen": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "stage": "post-trained"},
}
for _ck in CHECKPOINTS.values():  # one source of truth with the project's pin table
    if REVISIONS.get(_ck["repo"], _ck["revision"]) != _ck["revision"]:
        raise RuntimeError(f"{_ck['repo']}: pin differs from wf.constants.REVISIONS")

# lineage -> members; the first is the reference tokenizer for the token-identity gate
LINEAGES: dict[str, tuple[str, ...]] = {
    "llama": ("llama-base", "tulu-sft", "tulu-dpo", "tulu-rl", "tulu31-grpo", "llama-instruct", "w2w-llama"),
    "qwen": ("qwen-base", "qwen"),
}

# ---------------- conditions ----------------

DESC = "aligned_desc"  # the real task with the action's object described instead of named (rq1x's descriptions)
NEUTRAL = lexsem.NEUTRAL  # "Your task is to: explore the room."
PAIRS = {"name_assert": "ctrl_assert", "name_deny": "ctrl_deny", "name_use": "ctrl_use",
         "name_ignore": "ctrl_ignore"}
LEXSEM_CONDS = ("aligned", "swapped", NEUTRAL) + tuple(c for p in PAIRS.items() for c in p)
FAIL_CONDS = LEXSEM_CONDS + (DESC,)  # 12 conditions on the determined failures
SUCC_CONDS = ("aligned", "swapped", NEUTRAL)  # 3 conditions on the determined successes
BRIDGE_CONDS = ("aligned", "swapped")
FLOOR_CONDS = ("aligned", "swapped", "name_assert", "ctrl_assert")

# ---------------- arms ----------------
# An arm is one checkpoint in one prompt format. Modes are lexsem's: "plain" (no template, the tokenizer's own BOS
# where it adds one), "chat" (the checkpoint's own chat template as shipped at its pinned revision), "block" (Qwen3's
# chat template with thinking off, ending in the empty think block). "w2w" arms use the Word2World message format.

PLAIN = ("llama-base", "tulu-sft", "tulu-dpo", "tulu-rl", "tulu31-grpo", "llama-instruct", "qwen-base", "qwen")
NATIVE = {"tulu-sft": "chat", "tulu-dpo": "chat", "tulu-rl": "chat", "tulu31-grpo": "chat",
          "llama-instruct": "chat", "qwen": "block"}
BRIDGE = {"llama-base": "plain", "tulu-sft": "plain", "tulu-dpo": "plain", "tulu-rl": "plain",
          "tulu31-grpo": "plain", "llama-instruct": "plain", "w2w-llama": "chat"}


def _arms() -> dict[str, dict]:
    out = {}
    for ck in PLAIN:
        out[f"{ck}_plain"] = {"ckpt": ck, "mode": "plain", "kind": "main", "formats": ("obs", "fail")}
    for ck, mode in NATIVE.items():
        out[f"{ck}_native"] = {"ckpt": ck, "mode": mode, "kind": "main", "formats": ("obs", "fail", "yn")}
    for ck, mode in BRIDGE.items():
        out[f"{ck}_w2w"] = {"ckpt": ck, "mode": mode, "kind": "bridge", "formats": ("obs", "fail")}
    for name, a in list(out.items()):
        if a["kind"] == "main":
            out[f"{name}_floor"] = {**a, "kind": "floor", "formats": ("obs", "fail")}
    for ck in PLAIN:  # the fixed fallback (design section 5): seven examples, plain format, whole lineage
        out[f"{ck}_plain7"] = {"ckpt": ck, "mode": "plain", "kind": "main", "formats": ("obs", "fail"),
                               "fewshot": "1+2"}
    return out


ARMS = _arms()
LLAMA_STAGES = ("llama-base", "tulu-sft", "tulu-dpo", "tulu-rl", "tulu31-grpo", "llama-instruct")

# Arms whose cells lexsem scores in the identical environment (design section 7): pilot arm -> lexsem arm.
LEXSEM_REUSE = {"qwen-base_plain": "qwen8b_base_plain", "qwen_plain": "qwen_plain", "qwen_native": "qwen_block",
                "llama-instruct_native": "llama_chat"}
REUSE_CHECK_ITEMS = 5  # the first five failures also score the reused cells, for the bit-identity check


def lexsem_cells(lexsem_arm: str) -> dict[str, set[tuple[str, str]]]:
    """The (condition, format) cells lexsem scores for an arm, on the failures and on the successes."""
    a = lexsem.ARMS[lexsem_arm]
    conds = {c for g in a["groups"] for c in lexsem.GROUPS[g]}
    fails = {c for g in a["fail_groups"] for c in lexsem.GROUPS[g]}
    succ: set[tuple[str, str]] = set()
    if a.get("competence"):
        conds.add(NEUTRAL)
        fails.add(NEUTRAL)
        succ = {(NEUTRAL, f) for f in a["formats"] + ("fail",)}
    fail_cells = {(c, f) for c in conds for f in a["formats"]} | {(c, "fail") for c in fails}
    return {"failure": fail_cells, "success": succ}


def arm_cells(arm: str) -> dict[str, list[tuple[str, str]]]:
    """Every (condition, format) cell the arm needs, on the failures and on the successes."""
    a = ARMS[arm]
    if a["kind"] == "bridge":
        conds_f, conds_s = BRIDGE_CONDS, BRIDGE_CONDS
    elif a["kind"] == "floor":
        conds_f, conds_s = FLOOR_CONDS, ()
    else:
        conds_f, conds_s = FAIL_CONDS, SUCC_CONDS
    return {"failure": [(c, f) for c in conds_f for f in a["formats"]],
            "success": [(c, f) for c in conds_s for f in a["formats"]]}


def cells_to_score(arm: str, reuse: bool = True) -> dict[str, list[tuple[str, str]]]:
    """The arm's cells minus those lexsem scores in the identical environment (when `reuse`)."""
    need = arm_cells(arm)
    if not reuse or arm not in LEXSEM_REUSE:
        return need
    have = lexsem_cells(LEXSEM_REUSE[arm])
    return {s: [x for x in need[s] if x not in have[s]] for s in need}


# ---------------- items ----------------

def items(collected: list[dict]) -> dict[str, list[dict]]:
    """Lexsem's 450 failures (its frozen sample) and all of RQ2's determined successes that have a swapped task."""
    sets = rq1x.rq1x_sets(collected)
    fails = lexsem.sample(sets["primary"])
    succ = sorted(sets["successes"], key=lambda t: t["id"])
    for t in succ:
        t["lexsem_stratum"] = "success"
    return {"failure": fails, "success": succ}


def spec(t: dict, cond: str, refs_specs: dict | None = None) -> dict:
    """A condition's prompt change for item `t` (lexsem's specs, plus the described goal)."""
    if cond == DESC:
        return {"task_line": rq1x.describe_line(t["task_line"], t["goal_object"])}
    if refs_specs is not None and cond in refs_specs:
        return refs_specs[cond]
    if cond in ("aligned", "swapped", NEUTRAL):
        return {"task_line": {"aligned": t["task_line"], "swapped": t.get("swapped_task_line"),
                              NEUTRAL: rq1.ABSENT_LINE}[cond]}
    raise KeyError(cond)


def fewshot_messages(t: dict, sp: dict, fmt: str, fewshot: str | None) -> list[dict[str, str]]:
    """lexsem.messages, or with the seven examples of sets 1 and 2 (the fallback, design section 6)."""
    if not fewshot:
        return lexsem.messages(t, sp, fmt)
    if fewshot != "1+2":
        raise ValueError(fewshot)
    from wf.scoring import prompts as P

    base = P.world_model_prompt(t["history_text"], t["action"],
                                fewshot=list(P.WM_FEWSHOT_SETS[1]) + list(P.WM_FEWSHOT_SETS[2]),
                                task_line_override=sp["task_line"])
    orig = rq1.build_messages
    try:  # lexsem.messages builds on rq1.build_messages; give it the seven-example base, then restore
        rq1.build_messages = lambda *_a, **_k: [dict(m) for m in base]
        return lexsem.messages(t, sp, fmt)
    finally:
        rq1.build_messages = orig


def bridge_messages(t: dict, cond: str) -> list[dict[str, str]]:
    """The Word2World message format (system message with the room's full state and the task line, earlier actions
    as user turns and observations as assistant turns, the queried action last), as rq1_score scored it."""
    return rq1.arm_messages(t, cond, ft=True)


RECOG_PROMPT = 'The phrase "{desc}" refers to the household object called'


def recognition_pairs(fails: list[dict], count_tokens=None) -> list[tuple[str, str]]:
    """(object, control object) pairs for the teacher-forced recognition check of the described goal."""
    out = set()
    for t in fails:
        out.add((t["goal_object"], lexsem.control_object(t, count_tokens)))
    return sorted(out)
