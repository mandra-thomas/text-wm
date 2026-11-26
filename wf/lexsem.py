"""Lexical repetition, semantic identity, existence content and task relevance: item and condition builder for
the lexsem diagnostic (notes/lexsem_design.md). Builds prompts only; nothing here scores.

A mention condition adds one sentence about an object to the swapped task's prompt. By default the sentence follows
the swapped task line on the same line, exactly where the interpretability pilot's stage 0 put its mention sentence
(`rq1.goal_line(t, "swapped_plus")`); the position arm also puts it at the end of the history (just before the
"Action:" line) or just after the action line. The sentence crosses:

- the reference form: the action object's exact ALFWorld name, a spaced spelling of a compound name (variant), a
  different word for the same object (alias), rq1x's object-free description (desc), the same word in an unrelated
  sense (homonym); and, as controls, the name and the description of one other ALFWorld object type that is absent
  from the room and from another class (ctrl, cdesc);
- the existence content: asserts the object is in the room (assert), denies it (deny), says it is somewhere else
  (elsewhere), only lists the word without any claim (use), or tells the reader to ignore it (ignore).

Every name carrier keeps the name's leading space and ends it at a word boundary, so the name's tokens are the
same as in the action line (checked against both tokenizers when the tables were frozen, and by the scoring script
at run time).
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Callable

from wf import rq1, rq1v, rq1x
from wf.data.goals import interacted_objects, names_object

# ---------------- frozen tables (fixed before any lexsem score exists) ----------------

# Spaced spelling of the compound names among RQ2's goal objects. Under Qwen3's tokenizer "salt shaker" shares two
# of its three tokens with "saltshaker", "soap bar" one of two with "soapbar", "toilet paper" none with
# "toiletpaper"; the overlap is reported, not matched.
VARIANTS = {"soapbar": "soap bar", "peppershaker": "pepper shaker", "saltshaker": "salt shaker",
            "keychain": "key chain", "soapbottle": "soap bottle", "alarmclock": "alarm clock",
            "toiletpaper": "toilet paper"}
# A different single word for the same object, sharing no word part with the name and naming no other ALFWorld
# object. Only these eight of RQ2's 27 goal-object types have one. Four of them occur inside the object's own rq1x
# description (disc, blade, loaf, cushion), so the alias and description forms are not independent there.
ALIASES = {"cd": "disc", "potato": "spud", "pan": "skillet", "cloth": "rag", "bread": "loaf", "knife": "blade",
           "pillow": "cushion", "cup": "tumbler"}
# The alias check's foil: an unrelated object the alias must not be said to name (forced-choice check, section 5).
ALIAS_FOIL = {"cd": "apple", "potato": "book", "pan": "pillow", "cloth": "egg", "bread": "vase", "knife": "pillow",
              "pillow": "kettle", "cup": "pencil"}
# The same token in an unrelated sense, in a natural sentence with no claim about any room, and the control word
# that replaces it (one token in both tokenizers, same total token count). Nine of RQ2's 27 goal-object types have
# a natural homonym (a verb or another noun sense with the identical space-prefixed token).
HOMONYMS = {
    "mug": ("Thieves sometimes mug tourists near the station.", "rob"),
    "watch": ("Many people watch football on Sundays.", "play"),
    "pan": ("Critics often pan sequels to popular films.", "praise"),
    "plate": ("The tectonic plate shifted during the earthquake.", "fault"),
    "book": ("Remember to book a hotel for the trip.", "find"),
    "pencil": ("We can pencil in a meeting for Friday.", "fit"),
    "cup": ("She would cup her hands to catch the water.", "use"),
    "bowl": ("He will bowl the final over of the match.", "play"),
    "egg": ("His friends egg him on to sing.", "urge"),
}
# Token count of " <name>" for every ALFWorld object with an rq1x description (Qwen3-8B; Llama-3.1-8B gives the
# same counts). Computed 2026-11-25 from the pinned tokenizers; the scoring job checks them against the loaded one.
NAME_TOKENS = {"alarmclock": 2, "apple": 1, "baseballbat": 2, "basketball": 1, "book": 1, "bowl": 1, "box": 1,
               "bread": 1, "butterknife": 1, "candle": 1, "cd": 1, "cellphone": 1, "cloth": 1, "creditcard": 2,
               "cup": 1, "dishsponge": 3, "egg": 1, "fork": 1, "glassbottle": 3, "handtowel": 3, "kettle": 1,
               "keychain": 2, "knife": 1, "ladle": 2, "laptop": 1, "lettuce": 1, "mug": 1, "newspaper": 1, "pan": 1,
               "pen": 1, "pencil": 1, "peppershaker": 3, "pillow": 1, "plate": 1, "plunger": 2, "pot": 1,
               "potato": 1, "remotecontrol": 2, "saltshaker": 3, "scrubbrush": 2, "soapbar": 2, "soapbottle": 3,
               "spatula": 2, "spoon": 1, "spraybottle": 3, "statue": 1, "teddybear": 3, "tennisracket": 3,
               "tissuebox": 2, "toiletpaper": 3, "tomato": 1, "vase": 1, "watch": 1, "wateringcan": 2,
               "winebottle": 3}
# Coarse classes: the control object always comes from another class than the action's object, so that a
# semantic-activation account does not see the control as a related entity.
CLASSES = {
    "food": ("apple", "bread", "egg", "lettuce", "potato", "tomato"),
    "kitchenware": ("bowl", "butterknife", "cup", "fork", "glassbottle", "kettle", "knife", "ladle", "mug", "pan",
                    "peppershaker", "plate", "pot", "saltshaker", "spatula", "spoon", "winebottle"),
    "cleaning": ("cloth", "dishsponge", "handtowel", "plunger", "scrubbrush", "soapbar", "soapbottle", "spraybottle",
                 "tissuebox", "toiletpaper"),
    "devices": ("alarmclock", "cd", "laptop", "remotecontrol"),
    "personal": ("cellphone", "creditcard", "keychain", "watch"),
    "paper": ("book", "box", "newspaper", "pen", "pencil"),
    "decor": ("candle", "pillow", "statue", "teddybear", "vase", "wateringcan"),
    "sports": ("baseballbat", "basketball", "tennisracket"),
}
CLASS_OF = {o: c for c, objs in CLASSES.items() for o in objs}
# ALFWorld object names that occur in the few-shot examples of set 1 (the prompt's examples): never a control, and
# items whose action object is one of them are reported apart (design section 8).
FEWSHOT_OBJECTS = frozenset({"candle", "handtowel", "soapbar", "soapbottle", "spraybottle"})
# Lenient reader for the description check: answers that identify the object without its ALFWorld spelling.
# Frozen 2026-11-25 after reading Qwen's rq1x answers (stated in the design note).
LENIENT = {"soapbar": ("bar of soap", "soap bar"), "keychain": ("key ring", "keyring"),
           "saltshaker": ("shaker",), "peppershaker": ("shaker",), "cup": ("tumbler",), "cd": ("compact disc",)}
CHECK_ALIAS = 'Can the word "{alias}" refer to {art} {obj}? Answer Yes or No.'

SEED = "lexsem|2026-11-24"
N_LATER, N_STEP0 = 300, 150
CAP_LATER, CAP_STEP0 = 3, 2  # most items drawn from one game per stratum

# ---------------- sentences and conditions ----------------

TEMPLATES = {
    "assert": "There is {art} {ref} somewhere in the room.",
    "deny": "There is no {ref} in this room.",
    "use": "The following {unit} appears in this note: {ref}.",
    "elsewhere": "There is {art} {ref} somewhere in the house next door.",
    "ignore": "Ignore the {ref}.",
    "reliable": "The room's inventory list says there is {art} {ref} in the room.",
    # zero stated reliability: the content was chosen without looking (screening_off revision B1.6, 2026-11-25)
    "unreliable": "A tester who never looked in this room wrote this sentence by flipping a coin: there is {art} {ref} "
                  "in the room.",
}
ANCHORS = ("aligned", "swapped")
# condition -> (reference form, content, position)
MENTIONS = {
    "name_assert": ("name", "assert", "task"), "ctrl_assert": ("ctrl", "assert", "task"),
    "name_deny": ("name", "deny", "task"), "ctrl_deny": ("ctrl", "deny", "task"),
    "name_use": ("name", "use", "task"), "ctrl_use": ("ctrl", "use", "task"),
    "name_elsewhere": ("name", "elsewhere", "task"), "ctrl_elsewhere": ("ctrl", "elsewhere", "task"),
    "desc_assert": ("desc", "assert", "task"), "cdesc_assert": ("cdesc", "assert", "task"),
    "desc_use": ("desc", "use", "task"), "cdesc_use": ("cdesc", "use", "task"),
    "variant_assert": ("variant", "assert", "task"),  # compound names only
    "alias_assert": ("alias", "assert", "task"),  # objects in ALIASES only
    "homonym": ("homonym", "sentence", "task"), "ctrl_homonym": ("ctrl_homonym", "sentence", "task"),
    "name_ignore": ("name", "ignore", "task"), "ctrl_ignore": ("ctrl", "ignore", "task"),
    "name_assert_pre": ("name", "assert", "pre"), "ctrl_assert_pre": ("ctrl", "assert", "pre"),
    "name_assert_post": ("name", "assert", "post"), "ctrl_assert_post": ("ctrl", "assert", "post"),
    # the real task followed by the control object's assertion: the competition signature (design section 5)
    "aligned_ctrl_assert": ("ctrl", "assert", "aligned"),
    "name_reliable": ("name", "reliable", "task"), "ctrl_reliable": ("ctrl", "reliable", "task"),
    "name_unreliable": ("name", "unreliable", "task"), "ctrl_unreliable": ("ctrl", "unreliable", "task"),
}
# Condition groups; the run's list is fixed in the design (section 10) from the smoke run's measured cost.
GROUPS = {
    "core": ANCHORS + ("name_assert", "ctrl_assert", "name_deny", "ctrl_deny", "name_use", "ctrl_use",
                       "name_elsewhere", "ctrl_elsewhere", "desc_assert", "cdesc_assert", "desc_use", "cdesc_use",
                       "variant_assert", "alias_assert", "homonym", "ctrl_homonym"),
    "ignore": ("name_ignore", "ctrl_ignore"),
    "position": ("name_assert_pre", "ctrl_assert_pre", "name_assert_post", "ctrl_assert_post"),
    "reliability": ("name_reliable", "ctrl_reliable", "name_unreliable", "ctrl_unreliable"),
    # the with-block yes/no arm (decided by the user, 2026-11-25): the anchors and the name's assertion, denial and
    # listed word with their controls, which carry the content contrasts and the stage-0 bridge
    "block8": ANCHORS + ("name_assert", "ctrl_assert", "name_deny", "ctrl_deny", "name_use", "ctrl_use"),
    # the Llama yes/no arm: the content contrasts with the name, the elsewhere sentence and the description
    "names_only": ANCHORS + ("name_assert", "ctrl_assert", "name_deny", "ctrl_deny", "name_use", "ctrl_use",
                             "name_elsewhere", "ctrl_elsewhere", "desc_assert", "cdesc_assert"),
    "competition": ("aligned_ctrl_assert",),
    # the numerical-floor rescoring (design section 7): the anchors and the name-assertion pair
    "floor": ANCHORS + ("name_assert", "ctrl_assert"),
}
# The arms (decided 2026-11-25): the three core arms, then the family-generality arms (Qwen3-4B and Qwen3-14B each
# in plain text and with the think block, Llama-3.2-3B), which mirror the core arm of their kind. Each arm also has a
# floor arm (`<arm>_floor`): its first FLOOR_ITEMS items, the floor group, rescored on an A40 (design section 7).
_QWEN_PLAIN = {"mode": "plain", "groups": ("core", "ignore", "position", "reliability", "competition"),
               "formats": ("obs", "yn"), "fail_groups": ("core",)}
_QWEN_BLOCK = {"mode": "block", "groups": ("block8", "ignore", "competition"), "formats": ("obs", "yn"),
               "fail_groups": ()}
_LLAMA = {"mode": "chat", "groups": ("names_only", "ignore", "competition"), "formats": ("yn",), "fail_groups": ()}
ARMS = {
    "qwen_plain": {"model": "Qwen/Qwen3-8B", **_QWEN_PLAIN},
    "qwen_block": {"model": "Qwen/Qwen3-8B", **_QWEN_BLOCK},
    "llama_chat": {"model": "meta-llama/Llama-3.1-8B-Instruct", **_LLAMA},
    "qwen4b_plain": {"model": "Qwen/Qwen3-4B", **_QWEN_PLAIN},
    "qwen4b_block": {"model": "Qwen/Qwen3-4B", **_QWEN_BLOCK},
    "qwen14b_plain": {"model": "Qwen/Qwen3-14B", **_QWEN_PLAIN},
    "qwen14b_block": {"model": "Qwen/Qwen3-14B", **_QWEN_BLOCK},
    "llama3b_chat": {"model": "meta-llama/Llama-3.2-3B-Instruct", **_LLAMA},
}
FLOOR_ITEMS = 100
CORE = GROUPS["core"]
# Each mention condition's control: same carrier and position, the control object's reference in its place. The
# spelling and alias conditions use the name's control (their token counts differ from it by one or two tokens).
CONTROL_OF = {
    "name_assert": "ctrl_assert", "name_deny": "ctrl_deny", "name_use": "ctrl_use",
    "name_elsewhere": "ctrl_elsewhere", "desc_assert": "cdesc_assert", "desc_use": "cdesc_use",
    "variant_assert": "ctrl_assert", "alias_assert": "ctrl_assert", "homonym": "ctrl_homonym",
    "name_ignore": "ctrl_ignore", "name_assert_pre": "ctrl_assert_pre", "name_assert_post": "ctrl_assert_post",
    "name_reliable": "ctrl_reliable", "name_unreliable": "ctrl_unreliable",
}
FORMATS = ("obs", "yn")


def _article(word: str) -> str:
    return rq1._article(word)


def sentence(ref: str, content: str, phrase: bool = False) -> str:
    """One carrier sentence about `ref`. `phrase` says "phrase" instead of "word" in the use carrier."""
    return TEMPLATES[content].format(ref=ref, art=_article(ref), unit="phrase" if phrase else "word")


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def scene_types(t: dict) -> set[str]:
    """Object types present in the room at reset (instance names without their numbers)."""
    return {re.sub(r"\s+\d+$", "", inst) for inst in (t.get("scene_locations") or {})}


def control_object(t: dict, count_tokens: Callable[[str], int] | None = None) -> str:
    """The item's control object: an ALFWorld object type with an rq1x description, from another class than the
    action's object, not the goal or swapped object, not in the room at reset, not named in the history, action or
    few-shot examples, and with the same token count as the action object's name; among those, the closest in
    description length (`count_tokens`, default words), ties broken by a hash of the item id."""
    count = count_tokens or (lambda s: len(s.split()))
    obj = t["goal_object"]
    hist = t.get("history_text", "")
    bad = ({obj, t.get("swapped_goal_object")} | scene_types(t) | interacted_objects(hist) | FEWSHOT_OBJECTS)
    options = [o for o in sorted(rq1x.DESCRIPTIONS) if o not in bad and CLASS_OF[o] != CLASS_OF[obj]
               and NAME_TOKENS[o] == NAME_TOKENS[obj] and not names_object(hist, o)
               and not names_object(t["action"], o)]
    if not options:
        raise ValueError(f"{t['id']}: no control object with {NAME_TOKENS[obj]} tokens")
    target = count(" " + rq1x.DESCRIPTIONS[obj])
    return min(options, key=lambda o: (abs(count(" " + rq1x.DESCRIPTIONS[o]) - target),
                                       _h(f"{SEED}|ctrl|{t['id']}|{o}")))


def scene_prevalence(collected: list[dict]) -> dict[str, float]:
    """Share of rooms (games) whose scene at reset holds each object type: the prior probability of existence used
    in the control audit."""
    scenes: dict[str, set[str]] = {}
    for t in collected:
        scenes.setdefault(t["game_id"], scene_types(t))
    n = len(scenes) or 1
    count: dict[str, int] = defaultdict(int)
    for objs in scenes.values():
        for o in objs:
            count[o] += 1
    return {o: c / n for o, c in count.items()}


def references(t: dict, count_tokens: Callable[[str], int] | None = None) -> dict[str, str | None]:
    """The text each reference form puts in the sentence for this item (None where the form does not exist)."""
    obj = t["goal_object"]
    ctrl = control_object(t, count_tokens)
    hom = HOMONYMS.get(obj)
    return {"name": obj, "variant": VARIANTS.get(obj), "alias": ALIASES.get(obj),
            "desc": rq1x.DESCRIPTIONS[obj], "ctrl": ctrl, "cdesc": rq1x.DESCRIPTIONS[ctrl],
            "homonym": hom[0] if hom else None,
            "ctrl_homonym": hom[0].replace(f" {obj} ", f" {hom[1]} ", 1) if hom else None}


def condition_sentence(refs: dict, cond: str) -> str | None:
    form, content, _ = MENTIONS[cond]
    ref = refs[form]
    if ref is None:
        return None
    if content == "sentence":
        return ref
    return sentence(ref, content, phrase=form in ("desc", "cdesc"))


def specs(t: dict, conditions: tuple[str, ...] = CORE,
          count_tokens: Callable[[str], int] | None = None) -> dict[str, dict]:
    """Each condition's prompt change for item `t`: its task line and, for the position arm, a sentence placed
    before or after the action line. Conditions whose reference form the item lacks are left out."""
    if not t.get("swapped_task_line") or not t.get("swapped_goal_object"):
        raise ValueError(f"{t['id']}: needs a swapped task")
    refs = references(t, count_tokens)
    out = {}
    for c in conditions:
        if c in ANCHORS:
            out[c] = {"task_line": t["task_line"] if c == "aligned" else t["swapped_task_line"]}
            continue
        s = condition_sentence(refs, c)
        if s is None:
            continue
        pos = MENTIONS[c][2]
        if pos == "task":
            out[c] = {"task_line": f"{t['swapped_task_line']} {s}"}
        elif pos == "aligned":
            out[c] = {"task_line": f"{t['task_line']} {s}"}
        else:
            out[c] = {"task_line": t["swapped_task_line"], pos: s}
    return out


def messages(t: dict, spec: dict, fmt: str) -> list[dict[str, str]]:
    """The study's observation prompt (fmt "obs", ends "Next observation:") or rq1v's yes/no prompt (fmt "yn") with
    the condition's task line, and its sentence before or after the action line where the spec places one."""
    msgs = rq1.build_messages(t, "aligned", task_line=spec["task_line"])
    last = msgs[-1]
    content = last["content"]
    if not content.endswith("Next observation:"):
        raise ValueError(f"{t['id']}: prompt does not end with 'Next observation:'")
    if spec.get("pre") or spec.get("post"):
        a = content.index("\n\nAction: ")
        a_end = content.index("\n\n", a + 2)
        if spec.get("pre"):
            content = content[:a] + "\n" + spec["pre"] + content[a:]
        else:
            content = content[:a_end] + "\n\n" + spec["post"] + content[a_end:]
    if fmt == "yn":
        content = content[: -len("Next observation:")] + rq1v.YN_QUESTION
    elif fmt != "obs":
        raise ValueError(fmt)
    return msgs[:-1] + [{**last, "content": content}]


def check_reads(obj: str, answer: str) -> bool:
    """rq1x's reader, plus the frozen lenient synonyms."""
    squash = lambda s: re.sub(r"[^a-z]", "", s.lower())  # noqa: E731
    return rq1x.check_says(obj, answer) or any(squash(s) in squash(answer) for s in LENIENT.get(obj, ()))


# ---------------- items ----------------

def stratum(t: dict) -> str | None:
    """"step0" (no action yet), "later" (history after the task line does not name the action's object), or None
    (later items whose history already names it: excluded, since the word is then already repeated)."""
    from wf.interp.prompts import mentions_action_object

    if t.get("history_len", 0) == 0:
        return "step0"
    return None if mentions_action_object(t) else "later"


def sample(primary: list[dict], n_later: int = N_LATER, n_step0: int = N_STEP0, seed: str = SEED) -> list[dict]:
    """A seeded draw per stratum from RQ2's primary items that have descriptions, at most CAP_* items per game."""
    by = defaultdict(list)
    for t in primary:
        s = stratum(t)
        if s and rq1x.desc_ok(t) and t["goal_object"] in NAME_TOKENS:
            by[s].append(t)
    out = []
    for s, n, cap in (("later", n_later, CAP_LATER), ("step0", n_step0, CAP_STEP0)):
        pool = sorted(by[s], key=lambda t: t["id"])
        random.Random(f"{seed}|{s}").shuffle(pool)
        per_game: dict[str, int] = defaultdict(int)
        chosen = []
        for t in pool:
            if len(chosen) == n:
                break
            if per_game[t["game_id"]] < cap:
                per_game[t["game_id"]] += 1
                chosen.append(t)
        if len(chosen) < n:
            raise ValueError(f"stratum {s}: only {len(chosen)} items under the per-game cap, {n} wanted")
        out += [{**t, "lexsem_stratum": s} for t in sorted(chosen, key=lambda t: t["id"])]
    return out


def build(items: list[dict], conditions: tuple[str, ...] = CORE,
          count_tokens: Callable[[str], int] | None = None) -> list[dict]:
    """One record per item: identifiers, stratum, the references used and every condition's prompt change."""
    recs = []
    for t in items:
        recs.append({"id": t["id"], "game_id": t["game_id"], "stratum": t.get("lexsem_stratum") or stratum(t),
                     "action": t["action"], "goal_object": t["goal_object"],
                     "swapped_goal_object": t["swapped_goal_object"], "history_len": t.get("history_len", 0),
                     "fewshot_object": t["goal_object"] in FEWSHOT_OBJECTS,
                     "o_plus": t["candidates"]["o_plus"], "references": references(t, count_tokens),
                     "specs": specs(t, conditions, count_tokens)})
    return recs
