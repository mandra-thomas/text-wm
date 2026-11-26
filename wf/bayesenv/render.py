"""Prompt texts and item surfaces (notes/bayesenv_design.md, section 4).

Every prompt is a list of chat messages (system, user) plus an optional answer prefix, and is scored at the next
token after it. Two framings share the story word for word: the world-model framing ("wm") asks for the environment's
reply to "take the key from box 3"; the state-inference framing ("si") asks about box 3's current state. Each framing
has two readouts: wm "first" (the reply's first token, " You" against " Nothing") and "yn" (an anchored yes/no
question); si "narr" (a narrator sentence "Right now, box 3 ...", " holds" against " does") and "yn".

A prompt type fixes the task line, the tag line, the note and the arm. Within an item (a generator x history cell
and an item index), every prompt type shares the same surface: the rooms the boxes stand in and which boxes were
looked into, in which order. Only the line that defines the prompt type differs (tests check this).

The generator sentence never names an object. The key is named in the action, in the history where a history names
it, in a note, and in the cue lines that refer to it.
"""
from __future__ import annotations

import functools
import hashlib
import random
from dataclasses import dataclass

from wf.bayesenv import world as W

ROOMS = ("kitchen", "hall", "bedroom", "cellar", "attic", "study", "bathroom", "pantry", "porch", "garage",
         "library", "nursery")

# Cue wordings. The first entries are the ones a design uses; the full pools (8 each) are scored in the wording
# panel. Every goal wording is a task that refers to the key; every tag wording is a house tag that refers to it.
GOAL_WORDINGS = ("find the key", "get the key", "pick up the key", "look for the key and take it", "fetch the key",
                 "bring back the key", "track down the key", "collect the key")
TAG_WORDINGS = ("House tag: KEY.", "House tag: key.", "The tag on this house reads: key.",
                "This house carries the tag KEY.", "Tag attached to this house: the key.",
                "This house is tagged with the word key.", "House tag (set before the game): KEY.",
                "The house's tag names the key.")
NO_TAG = "House tag: none."
TAG_OTHER = "House tag: COIN."
TAG_DESCRIPTION = "House tag: the small metal object used to open the front door."
DESCRIPTION = "the small metal object used to open the front door"
EXPLORE = "explore the house"
COIN = "find the coin"
ACTION = "take the key from box 3"

SYSTEM = {
    "wm": ("You are a text world model for a small house game. Given the rules, what has happened so far and the next "
           "action, predict the environment's next reply exactly. Output only the reply."),
    "si": ("You are reasoning about a small house game. Given the rules and what has happened so far, describe the "
           "current state of the house exactly as asked."),
}
YN_WM = ("Question: will this action succeed, or will the environment reply \"Nothing happens.\"? "
         "Answer Yes if it succeeds and No if nothing happens.\nAnswer:")
YN_SI = "Question: right now, does box 3 hold the key, and is box 3 not locked? Answer Yes or No.\nAnswer:"
NARR_INSTRUCTION = "Complete the narrator's sentence about the key and box 3."
NARR_PREFIX = "Narrator: Right now, box 3"

# (framing, readout) -> the two candidate words read at the next position (with a leading space where the model
# reads them after text; the scorer decides the exact tokens per rendering mode).
READOUTS = {("wm", "first"): ("You", "Nothing"), ("wm", "yn"): ("Yes", "No"),
            ("si", "narr"): ("holds", "does"), ("si", "yn"): ("Yes", "No")}

LOOK_NOISY = "A look into a box reports what is in it, but misses an object that is there 30% of the time."
LOOK_SURE = "A look into a box reports what is in it and never misses an object that is there."
LOOK_CLOSE = ("A look into a box reports what is in it, but misses an object that is there 30% of the time; "
              "a close look never misses.")
LOCK_CHECK = "A lock check is wrong 2% of the time."
GLANCE = ("A glance at a box that holds a key suggests a key 90% of the time; a glance at a box without a key "
          "wrongly suggests one 10% of the time.")
EXISTENCE = (  # two wordings, alternated by item: (rule sentence, history lines)
    ("The house list names every hidden object and is always correct.",
     "> read the house list\nThe house list says: a key is hidden somewhere in this house."),
    ("The game master always tells the truth.",
     "> ask the game master\nThe game master says: there is a key somewhere in this house."),
)
TAKE_RULE = ("Taking an object from a box succeeds if and only if the object is in that box and the box is not locked; the "
             "environment then replies \"You take the <object> from box <n>.\" Otherwise it replies "
             "\"Nothing happens.\"")
P0 = {"wm": "Every action in this game, the next one included, was picked at random by a tester who cannot see "
            "inside the boxes.",
      "si": "Every action in this game was picked at random by a tester who cannot see inside the boxes."}
P1 = (  # the history-only helper (key task only), two wordings alternated by item
    "The earlier actions were picked at random by a tester who cannot see inside the boxes. The next action was chosen "
    "by a helper who knows only what is shown here: it picks, at random, one of the boxes not yet looked into, and "
    "tries to take from that box the object named by the task in the task line.",
    "A tester who cannot see inside the boxes picked the earlier actions at random. A helper who has seen nothing but "
    "the history below chose the next action: it chose at random among the boxes not yet looked into, and tries to "
    "take from that box the object named by the task in the task line.",
)
P2 = (  # the seeing guide (key task only), two wordings alternated by item
    "The earlier actions were picked at random by a tester who cannot see inside the boxes. The next action was chosen "
    "by a guide who can see inside every box. When the object named by the task in the task line is in a box that is "
    "not locked, the guide takes it from that box 90% of the time; in every other case it picks one of the twelve "
    "take actions (the key or the coin, from any box) at random.",
    "A tester who cannot see inside the boxes picked the earlier actions at random. A guide who sees inside every box "
    "chose the next action: if the object named by the task in the task line lies in a box that is not locked, 90% "
    "of the time the "
    "guide takes it from there; in every other case it picks at random among the twelve take actions (the key or the "
    "coin, from any of the six boxes).",
)
ANNOUNCE = (  # the announced replacement of the task line, two wordings alternated by item
    "The task line below is not the task the house was set up for. The house was set up for a task drawn at random "
    "from three equally likely tasks (find the key, find the coin, explore the house), with no house tag. Afterwards "
    "the task line was replaced by a new task drawn at random from the same three, independently of the house.",
    "After the house was set up for a task drawn at random from find the key, find the coin and explore the house "
    "(each with probability one third, and no house tag), the task card was swapped for a fresh random draw from the "
    "same three tasks, made independently of the house. The task line below is that fresh draw.",
)


def note_text(rho: float) -> str:
    return ("The caretaker left a note: \"There is a key in this house.\" On any day, the caretaker looks in every box "
            f"with probability {_pct(rho)} and then writes what is true; otherwise she writes \"There is a key in this "
            "house.\" or \"There is no key in this house.\" by flipping a coin, without looking.")


def _pct(p: float) -> str:
    v = round(100 * p, 1)
    return f"{v:g}%"


def generator_text(gen: str) -> str:
    lo, hi = W.GENERATORS[gen]
    return ("How the house was set up: first the player's task and the house tag (which may be none) were drawn. "
            "Then every object that the task or the house tag refers to was hidden with probability "
            f"{_pct(hi)}, and every other object with probability "
            f"{_pct(lo)}. A hidden object was put in one of the six boxes, chosen at random; an object that was not hidden "
            "is not in the house at all.")


# ----------------------------------------------------------------------------- item surfaces


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


@dataclass(frozen=True)
class Surface:
    rooms: tuple[str, ...]      # the room of box 1..6
    looked: tuple[int, ...]     # other boxes looked into, in order (box numbers, never 3)
    variant: int                # which wording of the existence statement, announcement and provenance


@functools.lru_cache(maxsize=None)
def surface(gen: str, hist: str, k: int, space: str = "toy") -> Surface:
    """The surface of item k in a cell: deterministic in (space, generator, history, k)."""
    r = _rng("bayesenv", space, gen, hist, k)
    rooms = tuple(r.sample(ROOMS, W.K))
    n = W.HISTORIES[hist].looks if hist in W.HISTORIES else 0
    others = [b for b in range(1, W.K + 1) if b != W.TARGET]
    looked = tuple(r.sample(others, n))
    return Surface(rooms=rooms, looked=looked, variant=k % 2)


# ----------------------------------------------------------------------------- prompt types


@dataclass(frozen=True)
class PromptType:
    """What distinguishes one prompt of an item from another."""
    task: str = EXPLORE
    tag: str = NO_TAG
    note: float | None = None
    arm: str = "obs"              # "obs" or "announced"
    provenance: str = "P0"        # "P0", "P1", "P2"

    @property
    def names_key(self) -> bool:
        """Does the task line or the tag refer to the key (so the generator's named rate applies to it)?"""
        return self.task not in (EXPLORE, COIN) or self.tag not in (NO_TAG, TAG_OTHER)


def prompt_types(w_goal: int = len(GOAL_WORDINGS), w_tag: int = len(TAG_WORDINGS)) -> dict[str, PromptType]:
    """Every prompt type of the toy, by name. Goal and tag wordings are numbered from 1."""
    t: dict[str, PromptType] = {"explore": PromptType(), "coin": PromptType(task=COIN)}
    for i in range(w_goal):
        t[f"key{i + 1}"] = PromptType(task=GOAL_WORDINGS[i])
    for rho in W.RHOS:
        t[f"note{int(round(100 * rho))}"] = PromptType(note=rho)
    for i in range(w_tag):
        t[f"tag{i + 1}"] = PromptType(tag=TAG_WORDINGS[i])
    t["tagO"] = PromptType(tag=TAG_OTHER)
    t["tagN"] = PromptType(tag=TAG_DESCRIPTION)
    t["coin_tag"] = PromptType(task=COIN, tag=TAG_WORDINGS[0])
    t["ann_key"] = PromptType(task=GOAL_WORDINGS[0], arm="announced")
    t["ann_coin"] = PromptType(task=COIN, arm="announced")
    t["P1_key"] = PromptType(task=GOAL_WORDINGS[0], provenance="P1")
    t["P2_key"] = PromptType(task=GOAL_WORDINGS[0], provenance="P2")
    return t


TYPES = prompt_types()
WM_ONLY = frozenset({"ann_key", "ann_coin", "P1_key", "P2_key"})


@functools.lru_cache(maxsize=None)
def exact(ptype: str, gen: str, hist: str) -> float:
    """The exact success probability of a toy prompt (world model and state inference share it)."""
    p = TYPES[ptype]
    return W.exact(gen, hist, p.names_key, note=p.note, provenance=p.provenance, announced=p.arm == "announced")


# ----------------------------------------------------------------------------- rendering


def _history_lines(hist: str, s: Surface) -> tuple[list[str], list[str]]:
    """(rule sentences about the history's instruments, history lines) for one item."""
    ev = W.HISTORIES[hist]
    rules: list[str] = []
    lines: list[str] = []
    if ev.looks:
        if ev.close_look_target:
            rules.append(LOOK_CLOSE)
        else:
            rules.append(LOOK_NOISY if ev.look_miss > 0 else LOOK_SURE)
        for b in s.looked:
            lines.append(f"> look in box {b}\nYou look in box {b}. It looks empty.")
    if ev.exists:
        rule, text = EXISTENCE[s.variant]
        rules.append(rule)
        lines.append(text)
    if ev.lock_report is not None:
        rules.append(LOCK_CHECK)
        lines.append("> check the lock of box 3\nThe lock check says: box 3 is locked.")
    if ev.close_look_target:
        lines.append("> look closely in box 3\nYou look closely in box 3. It is empty.")
    if ev.glimpse:
        rules.append(GLANCE)
        lines.append("> glance at box 3\nYour glance suggests a key in box 3.")
    if not lines:
        lines.append("(no actions yet)")
    return rules, lines


def user_text(ptype: str, gen: str, hist: str, k: int, framing: str) -> str:
    """The user turn shared by both readouts of a framing (without the readout's ending)."""
    p = TYPES[ptype]
    if framing == "si" and ptype in WM_ONLY:
        raise ValueError(f"{ptype} has no state-inference version")
    s = surface(gen, hist, k)
    rooms = ", ".join(f"box {i + 1} in the {r}" for i, r in enumerate(s.rooms[:-1]))
    rules, lines = _history_lines(hist, s)
    para = ["Rules of this game.",
            f"The house has six boxes, numbered 1 to 6: {rooms} and box 6 in the {s.rooms[-1]}. Each box is locked "
            "with probability 20%, independently of everything else.",
            generator_text(gen), *rules, TAKE_RULE]
    if p.provenance == "P0":
        para.append(P0[framing])
    else:
        para.append((P1 if p.provenance == "P1" else P2)[s.variant])
    if p.arm == "announced":
        para.append(ANNOUNCE[s.variant])
    if p.note is not None:
        para.append(note_text(p.note))
    head = "\n".join(para)
    history = "\n".join([f"Your task is to: {p.task}. {p.tag}", *lines])
    return f"{head}\n\nHistory:\n{history}"


@dataclass(frozen=True)
class Prompt:
    messages: tuple[tuple[str, str], ...]   # (role, content)
    prefix: str                             # text the answer continues from ("" for none)
    candidates: tuple[str, str]             # the two words read at the next position


def render(ptype: str, gen: str, hist: str, k: int, framing: str, readout: str) -> Prompt:
    body = user_text(ptype, gen, hist, k, framing)
    if framing == "wm":
        body += f"\n\nAction: {ACTION}"
        body += f"\n\n{YN_WM}" if readout == "yn" else "\n\nNext observation:"
        prefix = ""
    elif readout == "yn":
        body += f"\n\n{YN_SI}"
        prefix = ""
    else:
        body += f"\n\n{NARR_INSTRUCTION}"
        prefix = NARR_PREFIX
    if (framing, readout) not in READOUTS:
        raise ValueError(f"no readout {readout!r} in framing {framing!r}")
    return Prompt(messages=(("system", SYSTEM[framing]), ("user", body)), prefix=prefix,
                  candidates=READOUTS[(framing, readout)])


# ----------------------------------------------------------------------------- ALFWorld-surface rendering

ALF_OBJECT, ALF_OTHER = "keychain", "creditcard"
ALF_EXPLORE = "explore the room"            # the project's neutral task line
ALF_GOALS = ("put a keychain in safe", "put some keychain in safe")
ALF_COIN = "put a creditcard in safe"
ALF_TAGS = ("Room tag: keychain.", "The tag on this room reads: keychain.", "This room carries the tag KEYCHAIN.")
ALF_NO_TAG, ALF_TAG_OTHER = "Room tag: none.", "Room tag: creditcard."
ALF_RECEPTACLES = ("armchair 1", "bed 1", "cabinet 1", "cabinet 2", "coffeetable 1", "countertop 1", "desk 1",
                   "dresser 1", "garbagecan 1", "shelf 1", "shelf 2", "sidetable 1", "sofa 1", "laundryhamper 1")
ALF_HISTORIES = {"AH0": 0, "AH3": 4}        # drawers opened before going to drawer 3 (opening never misses)
ALF_REPLY_PREFIX = "You open the drawer 3. The drawer 3 is open. In it, you see"
ALF_YN = ("Question: when the agent opens drawer 3, will it see a keychain in it? Answer Yes if it will and No if it "
          "will not.\nAnswer:")
ALF_READOUTS = {"first": ("a", "nothing"), "yn": ("Yes", "No")}


def alf_types() -> dict[str, tuple[str, str, bool]]:
    """ALFWorld-surface prompt types: name -> (task, tag line, stated generator)."""
    t = {"explore": (ALF_EXPLORE, ALF_NO_TAG, True), "coin": (ALF_COIN, ALF_NO_TAG, True)}
    for i, g in enumerate(ALF_GOALS):
        t[f"key{i + 1}"] = (g, ALF_NO_TAG, True)
    for i, g in enumerate(ALF_TAGS):
        t[f"tag{i + 1}"] = (ALF_EXPLORE, g, True)
    t["tagO"] = (ALF_EXPLORE, ALF_TAG_OTHER, True)
    for name, task in (("imp_explore", ALF_EXPLORE), ("imp_coin", ALF_COIN), ("imp_key", ALF_GOALS[0])):
        t[name] = (task, "", False)           # implicit: no generator stated, no tag line
    return t


ALF_TYPES = alf_types()


@functools.lru_cache(maxsize=None)
def alf_exact(ptype: str, gen: str, hist: str) -> float | None:
    task, tag, stated = ALF_TYPES[ptype]
    if not stated:
        return None
    names = task in ALF_GOALS or tag in ALF_TAGS
    ev = W.Evidence(looks=ALF_HISTORIES[hist], look_miss=0.0, lock_p=0.0, outcome="in_target")
    return W.posterior(W.prior(gen, names), ev)


def alf_render(ptype: str, gen: str, hist: str, k: int, readout: str) -> Prompt:
    """The toy on ALFWorld's surface: the study's world-model system text and example set 1, ALFWorld's own action
    syntax and observation strings, six drawers, and the stated generator (omitted in the implicit types)."""
    from wf.scoring.prompts import WM_FEWSHOT_SET0, WM_SYSTEM

    task, tag, stated = ALF_TYPES[ptype]
    r = _rng("bayesenv", "alf", gen, hist, k)
    others = r.sample(ALF_RECEPTACLES, r.randint(4, 7)) + ["safe 1"] + [f"drawer {i}" for i in range(1, 7)]
    r.shuffle(others)
    seen = ", ".join(f"a {x}" for x in others[:-1]) + f", and a {others[-1]}"
    opened = r.sample([b for b in range(1, 7) if b != 3], ALF_HISTORIES[hist])
    lines = [f"Your task is to: {task}." + (f" {tag}" if tag else ""),
             f"You are in the middle of a room. Looking quickly around you, you see {seen}."]
    for b in opened:
        lines += [f"> go to drawer {b}", f"You arrive at drawer {b}. The drawer {b} is closed.", f"> open drawer {b}",
                  f"You open the drawer {b}. The drawer {b} is open. In it, you see nothing."]
    lines += ["> go to drawer 3", "You arrive at drawer 3. The drawer 3 is closed."]
    head = ("Every action in this episode, the next one included, was picked at random by a tester who cannot see "
            "inside the drawers.\n\n")
    if stated:
        lo, hi = W.GENERATORS[gen]
        head = ("How this episode was set up: the room has six drawers, numbered 1 to 6. Before the episode, a "
                f"keychain was put in one of the six drawers, chosen at random, with probability {_pct(hi)} if the "
                f"task line or the room tag names the keychain, and with probability {_pct(lo)} otherwise; otherwise "
                "the room has no keychain. Nothing else is ever put in a drawer. Opening a drawer shows everything in "
                "it. Every action in this episode, the next one included, was picked at random by a tester who cannot "
                "see inside the drawers.\n\n")
    query = f"{head}History:\n" + "\n".join(lines) + "\n\nAction: open drawer 3"
    query += f"\n\n{ALF_YN}" if readout == "yn" else "\n\nNext observation:"
    msgs = [("system", WM_SYSTEM)]
    for ex in WM_FEWSHOT_SET0:
        msgs.append(("user", f"History:\n{ex['history']}\n\nAction: {ex['action']}\n\nNext observation:"))
        msgs.append(("assistant", ex["observation"]))
    msgs.append(("user", query))
    return Prompt(messages=tuple(msgs), prefix="" if readout == "yn" else ALF_REPLY_PREFIX,
                  candidates=ALF_READOUTS[readout])
