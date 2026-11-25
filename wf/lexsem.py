"""Lexical repetition, semantic identity, existence content and task relevance: item and condition builder for
the lexsem diagnostic (notes/lexsem_design.md). Builds prompts only; nothing here scores.

Every condition except the two anchors is the swapped task line followed, on the same line, by one sentence about
an object, exactly where the interpretability pilot's stage 0 put its mention sentence (`rq1.goal_line`,
"swapped_plus"). The sentence crosses two factors:

- the reference form: the action object's exact ALFWorld name, a spaced spelling of a compound name (variant), a
  different word for the same object (alias), the object-free description that rq1x froze (desc); and, as
  controls, an unrelated word matched on token count and word frequency (ctrl) and the description of another
  object type absent from the room (cdesc);
- the existence content: asserts the object is in the room (assert), denies it (deny), names the word without any
  claim (use), or makes a generic statement with no claim about this room (generic).

Each mention condition has a control with the same carrier sentence and the control word or description in the
reference's place, so position, recency, syntax and (for names) token count are matched within each pair.
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

# Spaced spelling of the compound names among RQ2's goal objects. Token overlap with the name is reported, not
# matched: under Qwen3's tokenizer "salt shaker" shares two of its three tokens with "saltshaker", "toilet paper"
# none with "toiletpaper".
VARIANTS = {"soapbar": "soap bar", "peppershaker": "pepper shaker", "saltshaker": "salt shaker",
            "keychain": "key chain", "soapbottle": "soap bottle", "alarmclock": "alarm clock",
            "toiletpaper": "toilet paper"}
# A different single word for the same object, sharing no word part with the name and naming no other ALFWorld
# object. Only these eight of RQ2's 27 goal-object types have one.
ALIASES = {"cd": "disc", "potato": "spud", "pan": "skillet", "cloth": "rag", "bread": "loaf", "knife": "blade",
           "pillow": "cushion", "cup": "tumbler"}

# Token count of " <name>" (Qwen3-8B; Llama-3.1-8B gives the same counts for every entry) and Zipf frequency
# (wordfreq, English; 0 means below its list). Computed 2026-11-24 from the pinned tokenizers; the scoring job
# checks the token counts against the loaded tokenizer and refuses to run on a mismatch.
OBJECT_TOKENS = {"soapbar": 2, "cd": 1, "apple": 1, "mug": 1, "potato": 1, "tomato": 1, "pan": 1, "bowl": 1, "egg": 1,
                 "cloth": 1, "peppershaker": 3, "saltshaker": 3, "lettuce": 1, "book": 1, "pencil": 1, "plate": 1,
                 "bread": 1, "vase": 1, "keychain": 2, "watch": 1, "knife": 1, "cup": 1, "soapbottle": 3,
                 "spatula": 2, "alarmclock": 2, "pillow": 1, "toiletpaper": 3}
OBJECT_ZIPF = {"soapbar": 0.0, "cd": 4.36, "apple": 4.76, "mug": 3.71, "potato": 4.11, "tomato": 3.85, "pan": 4.39,
               "bowl": 4.63, "egg": 4.46, "cloth": 4.03, "peppershaker": 0.0, "saltshaker": 1.20, "lettuce": 3.56,
               "book": 5.43, "pencil": 3.95, "plate": 4.55, "bread": 4.50, "vase": 3.54, "keychain": 2.76,
               "watch": 5.34, "knife": 4.41, "cup": 5.11, "soapbottle": 0.0, "spatula": 2.86, "alarmclock": 0.0,
               "pillow": 3.96, "toiletpaper": 1.21}
# Unrelated control words: household-plausible nouns from tools, music, toys, outdoor gear and clothing, never an
# ALFWorld object or receptacle, and excluding kitchen, bathroom, cleaning, writing and timekeeping items (which
# relate to RQ2's goal objects). word -> (Qwen3 token count of " word", Zipf frequency).
CONTROL_POOL = {
    "umbrella": (1, 3.89), "whistle": (1, 3.88), "compass": (1, 3.74), "flashlight": (1, 3.46), "magnet": (1, 3.73),
    "hammer": (1, 4.16), "wrench": (1, 3.37), "marble": (1, 3.96), "telescope": (1, 3.75), "accordion": (1, 3.09),
    "trumpet": (1, 3.58), "violin": (1, 3.74), "globe": (1, 4.28), "lantern": (1, 3.61), "scarf": (1, 3.68),
    "wallet": (1, 4.15), "backpack": (1, 3.78), "suitcase": (1, 3.55), "puzzle": (1, 3.96), "kite": (1, 3.55),
    "trophy": (1, 4.17), "medal": (1, 4.39), "joystick": (1, 2.87), "headphones": (1, 3.88), "microphone": (1, 3.72),
    "tripod": (1, 3.18), "rake": (1, 3.38), "shovel": (1, 3.50), "flute": (1, 3.52), "anchor": (1, 4.02),
    "saddle": (1, 3.73), "helmet": (1, 4.02), "goggles": (1, 3.35), "bracelet": (1, 3.70), "ladder": (1, 4.03),
    "dice": (1, 3.76), "map": (1, 4.76), "radio": (1, 5.01), "camera": (1, 4.87), "guitar": (1, 4.48),
    "piano": (1, 4.31), "shirt": (1, 4.63), "hat": (1, 4.53), "flag": (1, 4.58), "coin": (1, 4.19), "tent": (1, 4.05),
    "rope": (1, 4.06), "bike": (1, 4.53), "jacket": (1, 4.32), "drum": (1, 4.11),
    "padlock": (2, 2.68), "screwdriver": (2, 3.13), "harmonica": (2, 2.97), "kazoo": (2, 2.35), "typewriter": (2, 3.26),
    "abacus": (2, 2.65), "hanger": (2, 3.24), "earmuffs": (2, 2.17), "mitten": (2, 2.48),
    "briefcase": (2, 3.09), "domino": (2, 3.23), "pushpin": (2, 1.19), "dartboard": (2, 1.97), "hairpin": (2, 2.78),
    "birdhouse": (2, 2.32), "easel": (2, 2.68), "chisel": (2, 2.96), "bugle": (2, 2.92), "banjo": (2, 3.27),
    "pinwheel": (2, 2.29), "lasso": (2, 2.74), "necktie": (2, 2.55), "bowtie": (2, 2.40), "locket": (2, 2.88),
    "anklet": (2, 2.28), "hammock": (2, 3.14), "beanbag": (2, 2.36), "sandal": (2, 2.91), "slipper": (2, 3.08),
    "mallet": (2, 3.00), "pliers": (2, 2.90), "gavel": (2, 2.66), "monocle": (2, 2.47),
    "frisbee": (3, 2.96), "binoculars": (3, 3.10), "ukulele": (3, 3.01), "tambourine": (3, 2.64),
    "xylophone": (3, 2.37), "birdcage": (3, 2.45), "mousetrap": (3, 2.32), "shoehorn": (3, 2.15),
    "doorknob": (3, 2.65), "flyswatter": (3, 1.52), "shoelace": (3, 2.35), "snowglobe": (3, 1.71),
    "metronome": (3, 2.62), "trowel": (3, 2.45), "cymbal": (3, 2.59), "slingshot": (3, 2.95),
}
N_NEAREST = 3  # the control word is drawn by item hash from the three closest in frequency at the same token count

SEED = "lexsem|2026-11-24"
N_LATER, N_STEP0 = 300, 150
CAP_LATER, CAP_STEP0 = 3, 2  # most items drawn from one game per stratum

# ---------------- sentences and conditions ----------------

TEMPLATES = {
    "assert": "There is {art} {ref} somewhere in the room.",
    "deny": "There is no {ref} in this room.",
    "use": 'The {unit} "{ref}" appears in this note.',
    "generic": "{Art} {ref} is a common household item.",
    # optional arms (design section 9); not in the core run
    "ignore": "Ignore the {ref}.",
    "reliable": "The room's inventory list says there is {art} {ref} in the room.",
    "unreliable": "A tester inserted this sentence at random: there is {art} {ref} in the room.",
}
ANCHORS = ("aligned", "swapped")
# condition -> (reference form, content)
MENTIONS = {
    "name_assert": ("name", "assert"), "name_deny": ("name", "deny"), "name_use": ("name", "use"),
    "name_generic": ("name", "generic"),
    "ctrl_assert": ("ctrl", "assert"), "ctrl_deny": ("ctrl", "deny"), "ctrl_use": ("ctrl", "use"),
    "ctrl_generic": ("ctrl", "generic"),
    "desc_assert": ("desc", "assert"), "desc_use": ("desc", "use"),
    "cdesc_assert": ("cdesc", "assert"), "cdesc_use": ("cdesc", "use"),
    "variant_assert": ("variant", "assert"),  # compound names only
    "alias_assert": ("alias", "assert"),  # objects in ALIASES only
}
OPTIONAL = {
    "name_ignore": ("name", "ignore"), "ctrl_ignore": ("ctrl", "ignore"),
    "name_reliable": ("name", "reliable"), "name_unreliable": ("name", "unreliable"),
}
CORE = ANCHORS + tuple(MENTIONS)
# Each mention condition's control: same carrier, unrelated reference.
CONTROL_OF = {
    "name_assert": "ctrl_assert", "name_deny": "ctrl_deny", "name_use": "ctrl_use", "name_generic": "ctrl_generic",
    "desc_assert": "cdesc_assert", "desc_use": "cdesc_use", "variant_assert": "ctrl_assert",
    "alias_assert": "ctrl_assert", "name_ignore": "ctrl_ignore", "name_reliable": "ctrl_assert",
    "name_unreliable": "ctrl_assert",
}
FORMATS = ("obs", "yn")


def _article(word: str) -> str:
    return rq1._article(word)


def sentence(ref: str, content: str, phrase: bool = False) -> str:
    """One carrier sentence about `ref`. `phrase` says "phrase" instead of "word" in the use carrier."""
    art = _article(ref)
    return TEMPLATES[content].format(ref=ref, art=art, Art=art.capitalize(), unit="phrase" if phrase else "word")


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def _types_in(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def control_word(t: dict) -> str:
    """The item's unrelated control word: same token count as the action object's name, among the N_NEAREST
    closest in Zipf frequency, picked by a hash of the item id; never a word in the item's history or action."""
    obj = t["goal_object"]
    if obj not in OBJECT_TOKENS:
        raise KeyError(f"no frozen token count for {obj!r}")
    seen = _types_in(t.get("history_text", "")) | _types_in(t.get("action", ""))
    pool = sorted((abs(z - OBJECT_ZIPF[obj]), w) for w, (n, z) in CONTROL_POOL.items()
                  if n == OBJECT_TOKENS[obj] and w not in seen)
    near = [w for _, w in pool[:N_NEAREST]]
    if not near:
        raise ValueError(f"{t['id']}: no control word with {OBJECT_TOKENS[obj]} tokens")
    return near[_h(f"{SEED}|ctrl|{t['id']}") % len(near)]


def scene_types(t: dict) -> set[str]:
    """Object types present in the room at reset (instance names without their numbers)."""
    return {re.sub(r"\s+\d+$", "", inst) for inst in (t.get("scene_locations") or {})}


def control_description(t: dict, count_tokens: Callable[[str], int] | None = None) -> str:
    """Another ALFWorld object type whose description stands in for the action object's: not the goal or swapped
    object, not in the room at reset, not named in the history or action; the closest in token count to the
    item's own description (`count_tokens`, default words), ties broken by a hash of the item id."""
    count = count_tokens or (lambda s: len(s.split()))
    own = rq1x.DESCRIPTIONS[t["goal_object"]]
    hist = t.get("history_text", "")
    bad = {t["goal_object"], t.get("swapped_goal_object")} | scene_types(t) | interacted_objects(hist)
    options = [o for o in sorted(rq1x.DESCRIPTIONS) if o not in bad and not names_object(hist, o)
               and not names_object(t["action"], o)]
    if not options:
        raise ValueError(f"{t['id']}: no control description available")
    target = count(" " + own)
    return min(options, key=lambda o: (abs(count(" " + rq1x.DESCRIPTIONS[o]) - target),
                                       _h(f"{SEED}|cdesc|{t['id']}|{o}")))


def references(t: dict, count_tokens: Callable[[str], int] | None = None) -> dict[str, str | None]:
    """The text each reference form puts in the sentence for this item (None where the form does not exist)."""
    obj = t["goal_object"]
    cdesc_obj = control_description(t, count_tokens)
    return {"name": obj, "variant": VARIANTS.get(obj), "alias": ALIASES.get(obj),
            "desc": rq1x.DESCRIPTIONS[obj], "ctrl": control_word(t), "cdesc": rq1x.DESCRIPTIONS[cdesc_obj],
            "cdesc_object": cdesc_obj}


def task_lines(t: dict, conditions: tuple[str, ...] = CORE,
               count_tokens: Callable[[str], int] | None = None) -> dict[str, str]:
    """Each condition's task line for item `t`; conditions whose reference form the item lacks are left out."""
    if not t.get("swapped_task_line") or not t.get("swapped_goal_object"):
        raise ValueError(f"{t['id']}: needs a swapped task")
    refs = references(t, count_tokens)
    out = {}
    for c in conditions:
        if c == "aligned":
            out[c] = t["task_line"]
        elif c == "swapped":
            out[c] = t["swapped_task_line"]
        else:
            form, content = {**MENTIONS, **OPTIONAL}[c]
            ref = refs[form]
            if ref is None:
                continue
            out[c] = f"{t['swapped_task_line']} {sentence(ref, content, phrase=form in ('desc', 'cdesc'))}"
    return out


def messages(t: dict, line: str, fmt: str) -> list[dict[str, str]]:
    """The study's observation prompt (fmt "obs", ends "Next observation:") or rq1v's yes/no prompt (fmt "yn")
    with `line` as the task line; nothing else differs between conditions."""
    msgs = rq1.build_messages(t, "aligned", task_line=line)
    if fmt == "obs":
        return msgs
    if fmt != "yn":
        raise ValueError(fmt)
    last = msgs[-1]
    if not last["content"].endswith("Next observation:"):
        raise ValueError(f"{t['id']}: prompt does not end with 'Next observation:'")
    return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + rq1v.YN_QUESTION}]


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
        if s and rq1x.desc_ok(t) and t["goal_object"] in OBJECT_TOKENS:
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
    """One record per item: identifiers, stratum, the references used and every condition's task line."""
    recs = []
    for t in items:
        refs = references(t, count_tokens)
        recs.append({"id": t["id"], "game_id": t["game_id"], "stratum": t.get("lexsem_stratum") or stratum(t),
                     "action": t["action"], "goal_object": t["goal_object"],
                     "swapped_goal_object": t["swapped_goal_object"], "history_len": t.get("history_len", 0),
                     "o_plus": t["candidates"]["o_plus"], "references": refs,
                     "lines": task_lines(t, conditions, count_tokens)})
    return recs
