"""CPU check of assumption A1' on ALFWorld's initial states (notes/a1prime_check.md).

A1' says: once an object type is known to exist in the room, where its instances are (which receptacle type, inside
a closed receptacle or not, how many instances) does not depend on which task was drawn. The check compares, in
ALFWorld's generated games, the placement of the task's object type with the placement of the same type in games
whose task names another object, within the same floor plan where possible.

Two steps:

  extract  reads every game.tw-pddl (and its traj_data.json and receps.json) under an ALFWorld data folder and
           writes one compact JSON record per game. Standard library only (runs with ICE's system python3.9).
  analyze  reads that file (and, optionally, the RQ2 collection's transitions) and writes the metrics JSON. Needs
           numpy. Deterministic: every random draw is seeded.

    python3 scripts/a1prime_check.py extract --data ~/.cache/alfworld --out a1prime_games.jsonl
    python scripts/a1prime_check.py analyze --games a1prime_games.jsonl \
        --transitions <dir holding rq12_transitions_s0..s2/transitions.jsonl> --out notes/a1prime_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

SEED = 20260924
N_BOOT = 2000
N_PERM = 2000
SMOOTH = 0.005  # added to every q, as in the earlier generator index (undetermined_and_decomposition.md §4.2)
MIN_TASK_TYPE_GAMES = 30
APPLIANCE = {"pick_heat_then_place_in_recep": "microwave", "pick_cool_then_place_in_recep": "fridge",
             "pick_clean_then_place_in_recep": "sinkbasin"}


# ---------------- extract ----------------

_PRED = re.compile(r"\(\s*(objectType|receptacleType|openable|pickupable|inReceptacle|inReceptacleObject)\s+([^\s()]+)"
                   r"(?:\s+([^\s()]+))?\s*\)")


def room_of(scene: int) -> str:
    """AI2-THOR floor-plan numbering: 1-30 kitchens, 201-230 living rooms, 301-330 bedrooms, 401-430 bathrooms."""
    return {0: "kitchen", 2: "livingroom", 3: "bedroom", 4: "bathroom"}.get(scene // 100, "other")


def parse_game(path: str) -> dict:
    game = json.load(open(path))
    d = os.path.dirname(path)
    traj = json.load(open(os.path.join(d, "traj_data.json")))
    receps_path = os.path.join(d, "receps.json")
    receps = json.load(open(receps_path)) if os.path.exists(receps_path) else {}
    p = game["pddl_problem"]
    init = p[p.find("(:init"): p.find("(:goal")]
    otype, rtype, openable, pickup, inrec, inrecobj = {}, {}, set(), set(), defaultdict(list), defaultdict(list)
    for pred, a, b in _PRED.findall(init):
        if pred == "objectType":
            otype[a] = b[: -len("Type")] if b.endswith("Type") else b
        elif pred == "receptacleType":
            rtype[a] = b[: -len("Type")] if b.endswith("Type") else b
        elif pred == "openable":
            openable.add(a)
        elif pred == "pickupable":
            pickup.add(a)
        elif pred == "inReceptacle":
            inrec[a].append(b)
        elif pred == "inReceptacleObject":
            inrecobj[a].append(b)
    # receps.json keys are AI2-THOR ids ("Cabinet|-03.87|+00.56|+04.20"); the PDDL ids encode the same coordinates
    def pddl_id(thor_id: str) -> str:
        return (thor_id.replace("|", "_bar_").replace("-", "_minus_").replace("+", "_plus_").replace(".", "_dot_"))
    num_ids = {pddl_id(k): v.get("num_id") for k, v in receps.items()}
    parts = os.path.normpath(path).split(os.sep)
    params = traj["pddl_params"]
    scene = int(traj["scene"]["scene_num"])
    return {
        "split": parts[-4], "task_dir": parts[-3], "trial": parts[-2],
        "solvable": game.get("solvable"),
        "task_type": traj["task_type"], "scene": scene, "room": room_of(scene),
        "object_target": params.get("object_target") or "", "parent_target": params.get("parent_target") or "",
        "mrecep_target": params.get("mrecep_target") or "", "toggle_target": params.get("toggle_target") or "",
        "sliced": bool(params.get("object_sliced")),
        "receptacles": {r: {"type": t, "openable": r in openable, "num_id": num_ids.get(r)} for r, t in rtype.items()},
        "objects": [{"id": o, "type": t, "pickupable": o in pickup, "in": inrec.get(o, []), "in_obj": inrecobj.get(o, [])}
                    for o, t in otype.items() if o not in rtype],
    }


def extract(args) -> None:
    root = os.path.expanduser(args.data)
    paths = sorted(glob.glob(os.path.join(root, "json_2.1.1", "*", "*", "*", "game.tw-pddl")))
    n = 0
    with open(args.out, "w") as f:
        for path in paths:
            f.write(json.dumps(parse_game(path)) + "\n")
            n += 1
    print(f"{n} games written to {args.out}", file=sys.stderr)


# ---------------- analyze ----------------

def rows_from_games(games: list[dict]) -> list[dict]:
    """One row per (game, pickupable object type present). Instances of the type: their receptacle types, whether
    that receptacle is openable (every openable receptacle starts closed: no game's initial state has `opened`)."""
    rows = []
    for gi, g in enumerate(games):
        by_type = defaultdict(list)
        for o in g["objects"]:
            if o["pickupable"]:
                by_type[o["type"]].append(o)
        for t, insts in by_type.items():
            places = []
            for o in insts:
                recs = [r for r in o["in"] if r in g["receptacles"]]
                if len(recs) == 1:
                    r = g["receptacles"][recs[0]]
                    places.append((r["type"].lower(), bool(r["openable"])))
                elif not recs and o["in_obj"]:
                    places.append(("(inside an object)", False))
                else:
                    places.append(("(other)", False))
            rows.append({
                "g": gi, "scene": g["scene"], "room": g["room"], "split": g["split"], "task_type": g["task_type"],
                "type": t.lower(), "is_task": t == g["object_target"],
                "is_mrecep": bool(g["mrecep_target"]) and t == g["mrecep_target"],
                "n_inst": len(insts),
                "closed_share": sum(c for _, c in places) / len(places),
                "recep_types": [p for p, _ in places],
                "appliance_share": (sum(p == APPLIANCE[g["task_type"]] for p, _ in places) / len(places)
                                    if g["task_type"] in APPLIANCE else None),
                "target_share": sum(p == g["parent_target"].lower() for p, _ in places) / len(places),
            })
    return rows


def _strata(rows: list[dict], comparison: str) -> dict:
    """(scene, type) strata holding at least one comparison row and one row of the same type in a game whose task
    names another object. `comparison` is "task" (the task's object type) or "mrecep" (the movable receptacle)."""
    key = "is_task" if comparison == "task" else "is_mrecep"
    st = defaultdict(lambda: {"cmp": [], "other": []})
    for r in rows:
        if r["is_task"] and comparison == "mrecep":
            continue
        if r["is_mrecep"] and comparison == "task":
            continue
        st[(r["scene"], r["type"])]["cmp" if r[key] else "other"].append(r)
    return {k: v for k, v in st.items() if v["cmp"] and v["other"]}


def _stratum_arrays(strata: dict, field: str) -> list[tuple]:
    """Per stratum: (scene, values of every row with the comparison rows first, number of comparison rows)."""
    import numpy as np

    out = []
    for k, s in strata.items():
        c = [r[field] for r in s["cmp"] if r[field] is not None]
        o = [r[field] for r in s["other"] if r[field] is not None]
        if c and o:
            out.append((k[0], np.array(c + o, dtype=float), len(c)))
    return out


def _diff_terms(arrs: list[tuple]) -> tuple[dict, dict]:
    """Per floor plan: sum over its strata of n_cmp * (mean_cmp - mean_other), and of n_cmp."""
    num, den = defaultdict(float), defaultdict(float)
    for scene, v, nc in arrs:
        num[scene] += nc * (v[:nc].mean() - v[nc:].mean())
        den[scene] += nc
    return num, den


def _tv_one(recs: list[list[str]], nc: int) -> float:
    c = Counter(p for r in recs[:nc] for p in r)
    o = Counter(p for r in recs[nc:] for p in r)
    a, b = sum(c.values()), sum(o.values())
    return 0.5 * sum(abs(c[p] / a - o[p] / b) for p in set(c) | set(o))


def generator_level(rows: list[dict], comparison: str) -> dict:
    """Within-floor-plan comparison of the comparison type's placement with the same type's placement in games whose
    task names another object. Each statistic is the mean over comparison rows of (row value - mean of the same
    floor plan and type's other rows); 95% interval from a bootstrap over floor plans; p from shuffling, within each
    floor plan and type, which games carry the comparison label."""
    import numpy as np

    strata = _strata(rows, comparison)
    scenes = sorted({k[0] for k in strata})
    out = {"n_strata": len(strata), "n_scenes": len(scenes),
           "n_comparison_rows": sum(len(s["cmp"]) for s in strata.values()),
           "n_other_rows": sum(len(s["other"]) for s in strata.values())}
    if not strata:
        return out
    rng = np.random.default_rng(SEED)
    for field in ("closed_share", "n_inst", "appliance_share", "target_share"):
        arrs = _stratum_arrays(strata, field)
        if not arrs:
            out[field] = None
            continue
        num, den = _diff_terms(arrs)
        sc = sorted(num)
        nv, dv = np.array([num[x] for x in sc]), np.array([den[x] for x in sc])
        est = nv.sum() / dv.sum()
        w = rng.multinomial(len(sc), [1 / len(sc)] * len(sc), size=N_BOOT)
        boots = (w @ nv) / np.where(w @ dv > 0, w @ dv, np.nan)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            tot = 0.0
            for _, v, nc in arrs:
                pv = rng.permutation(v)
                tot += nc * (pv[:nc].mean() - pv[nc:].mean())
            null[b] = tot / dv.sum()
        cmp_vals = np.concatenate([v[:nc] for _, v, nc in arrs])
        oth_vals = np.concatenate([v[nc:] for _, v, nc in arrs])
        out[field] = {"difference_within_floor_plan": float(est), "ci95": [float(lo), float(hi)],
                      "perm_p": float((1 + (np.abs(null) >= abs(est) - 1e-12).sum()) / (1 + N_PERM)),
                      "comparison_mean": float(cmp_vals.mean()), "other_mean": float(oth_vals.mean()),
                      "n_strata": len(arrs)}
    # receptacle-type distribution: total variation against its permutation null
    st = [(k[0], [r["recep_types"] for r in s["cmp"] + s["other"]], len(s["cmp"])) for k, s in strata.items()]
    wsum = sum(nc for _, _, nc in st)
    tv = sum(nc * _tv_one(recs, nc) for _, recs, nc in st) / wsum
    n_tv_perm = 500
    null = np.empty(n_tv_perm)
    for b in range(n_tv_perm):
        tot = 0.0
        for _, recs, nc in st:
            order = rng.permutation(len(recs))
            tot += nc * _tv_one([recs[i] for i in order], nc)
        null[b] = tot / wsum
    out["receptacle_type_tv"] = {"observed": tv, "null_mean": float(null.mean()), "null_95th": float(np.percentile(null, 95)),
                                 "excess_over_null": tv - float(null.mean()), "n_perm": n_tv_perm,
                                 "perm_p": float((1 + (null >= tv).sum()) / (1 + n_tv_perm))}
    cmp_c = Counter(p for s in strata.values() for r in s["cmp"] for p in r["recep_types"])
    oth_c = Counter(p for s in strata.values() for r in s["other"] for p in r["recep_types"])
    nc_, no_ = sum(cmp_c.values()), sum(oth_c.values())
    out["receptacle_type_shares"] = {p: {"comparison": cmp_c[p] / nc_, "other": oth_c[p] / no_}
                                     for p in sorted(set(cmp_c) | set(oth_c), key=lambda x: -(cmp_c[x] + oth_c[x]))}
    return out


# ---- the decision-relevant index for RQ2 take items ----

def _frac_table(games: list[dict]):
    """Per game: room, task type, task object type, and for every pickupable type present the fraction of each
    receptacle type's receptacles holding it (the q of undetermined_and_decomposition.md §4.2)."""
    out = []
    for g in games:
        n_r = Counter(r["type"].lower() for r in g["receptacles"].values())
        holding = defaultdict(lambda: defaultdict(set))  # type -> recep type -> receptacle ids
        for o in g["objects"]:
            if not o["pickupable"]:
                continue
            for r in o["in"]:
                if r in g["receptacles"]:
                    holding[o["type"].lower()][g["receptacles"][r]["type"].lower()].add(r)
        present = {o["type"].lower() for o in g["objects"] if o["pickupable"]}
        out.append({"room": g["room"], "task_type": g["task_type"], "scene": g["scene"],
                    "task_obj": g["object_target"].lower(), "present": present, "n_r": n_r,
                    "count": Counter(o["type"].lower() for o in g["objects"] if o["pickupable"]),
                    "frac": {t: {rt: len(ids) / n_r[rt] for rt, ids in d.items()} for t, d in holding.items()}})
    return out


def q_values(table: list[dict], room: str, obj: str, rt: str, task_type: str) -> tuple[float, float, float, int]:
    """(q_task, q_exists, q_any, n_task_games): mean fraction of `rt` receptacles holding `obj` in games of this room
    whose task object is `obj` (task-type specific when at least 30 such games, else pooled); in games where `obj`
    exists but is not the task's object; in all games where it is not the task's object (absent counts as 0)."""
    task = [g for g in table if g["room"] == room and g["task_obj"] == obj and g["task_type"] == task_type and rt in g["n_r"]]
    if len(task) < MIN_TASK_TYPE_GAMES:
        task = [g for g in table if g["room"] == room and g["task_obj"] == obj and rt in g["n_r"]]
    other = [g for g in table if g["room"] == room and g["task_obj"] != obj and rt in g["n_r"]]
    exists = [g for g in other if obj in g["present"]]
    mean = lambda gs: sum(g["frac"].get(obj, {}).get(rt, 0.0) for g in gs) / len(gs) if gs else float("nan")
    return mean(task), mean(exists), mean(other), len(task)


def instance_values(table: list[dict], room: str, obj: str, k: int) -> tuple[float, float]:
    """P(at least k instances of `obj`) in this room's games whose task object is `obj`, and in games where `obj`
    exists but is not the task's object: whether the instance number the action names exists, given the type."""
    task = [g for g in table if g["room"] == room and g["task_obj"] == obj]
    exists = [g for g in table if g["room"] == room and g["task_obj"] != obj and g["count"][obj] > 0]
    share = lambda gs: sum(g["count"][obj] >= k for g in gs) / len(gs) if gs else float("nan")
    return share(task), share(exists)


_TEMPLATES = [
    (r"^put a hot (\w+) in (\w+)\.$", "pick_heat_then_place_in_recep"),
    (r"^heat some (\w+) and put it in (\w+)\.$", "pick_heat_then_place_in_recep"),
    (r"^put a cool (\w+) in (\w+)\.$", "pick_cool_then_place_in_recep"),
    (r"^cool some (\w+) and put it in (\w+)\.$", "pick_cool_then_place_in_recep"),
    (r"^put a clean (\w+) in (\w+)\.$", "pick_clean_then_place_in_recep"),
    (r"^clean some (\w+) and put it in (\w+)\.$", "pick_clean_then_place_in_recep"),
    (r"^(?:put|find) two (\w+) (?:in|on) (\w+)\.$", "pick_two_obj_and_place"),
    (r"^find two (\w+) and put them in (\w+)\.$", "pick_two_obj_and_place"),
    (r"^(?:look at|examine) (?:the )?(\w+) (?:under|with) the (\w+)\.$", "look_at_obj_in_light"),
    (r"^put (?:a|an|some) (\w+) (?:in|on) (\w+)\.$", None),  # simple or movable-receptacle: resolved by the game
]


def task_type_of(line: str) -> str | None:
    s = line.replace("Your task is to:", "").strip()
    for pat, tt in _TEMPLATES:
        if re.match(pat, s):
            return tt or "pick_and_place_simple"
    return None


def room_from_receptacle_types(types: set[str]) -> str:
    """AI2-THOR's room types by their signature receptacles: a fridge only in kitchens, a toilet only in bathrooms,
    a bed only in bedrooms; the rest are living rooms. Checked against the floor-plan number of every extracted game
    (`room_rule_agreement` in the metrics)."""
    if "fridge" in types:
        return "kitchen"
    if "toilet" in types:
        return "bathroom"
    if "bed" in types:
        return "bedroom"
    return "livingroom"


def room_of_item(t: dict) -> str:
    """The room type of an RQ2 item's game, from the receptacle types the item's scene lists."""
    return room_from_receptacle_types({re.sub(r"\s+\d+$", "", r) for r in t.get("scene_receptacles") or []})


def item_level(games: list[dict], transitions_dir: str | None, items_out: str | None = None) -> dict:
    import numpy as np

    if not transitions_dir:
        return {"skipped": "no transitions given"}
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("RUNS_DIR", transitions_dir)
    from wf.rq2 import primary_set  # the preregistered population (4,666 items)
    trans = []
    for s in ("rq12_transitions_s0", "rq12_transitions_s1", "rq12_transitions_s2"):
        trans += [json.loads(x) for x in open(os.path.join(transitions_dir, s, "transitions.jsonl"))]
    items = primary_set(trans)
    train = [g for g in games if g["split"] == "train"]
    unseen = [g for g in games if g["split"] == "valid_unseen"]
    table = _frac_table(train)
    rows, skipped, qcache = [], Counter(), {}
    for t in items:
        m = re.fullmatch(r"take (\w+) (\d+) from (\w+) \d+", t["action"])
        if m:
            k_inst = int(m[2])
            m = re.fullmatch(r"take (\w+) \d+ from (\w+) \d+", t["action"])
        if not m or m[1] != t["goal_object"]:
            skipped["not a take of the goal object"] += 1
            continue
        room = room_of_item(t)
        tt = task_type_of(t["task_line"])
        if room is None or tt is None:
            skipped["room or task type not identified"] += 1
            continue
        key = (room, m[1], m[2], tt)
        if key not in qcache:
            qcache[key] = q_values(table, *key)
        qt, qe, qn, n_task = qcache[key]
        if not all(math.isfinite(x) for x in (qt, qe, qn)):
            skipped["no train games for this room, object and receptacle"] += 1
            continue
        ikey = (room, m[1], k_inst)
        if ikey not in qcache:
            qcache[ikey] = instance_values(table, *ikey)
        pt, pe = qcache[ikey]
        hist = t["history_text"]
        seen = re.search(rf"\b{m[2]} \d+\b", "\n".join(hist.splitlines()[2:])) is not None
        rows.append({"id": t["id"], "game": t["game_id"], "room": room, "obj": m[1], "rt": m[2], "task_type": tt,
                     "history_len": t.get("history_len", 0), "recep_type_named_in_history": seen,
                     "instance": k_inst,
                     "instance_given_existence": math.log((pt + SMOOTH) / (pe + SMOOTH)),
                     "location_given_existence": math.log((qt + SMOOTH) / (qe + SMOOTH)),
                     "existence": math.log((qe + SMOOTH) / (qn + SMOOTH)),
                     "total": math.log((qt + SMOOTH) / (qn + SMOOTH))})

    def summary(rs: list[dict], key: str) -> dict:
        v = np.array([r[key] for r in rs])
        if not len(v):
            return {"n": 0}
        games_ = sorted({r["game"] for r in rs})
        rng = np.random.default_rng(SEED)
        idx = defaultdict(list)
        for i, r in enumerate(rs):
            idx[r["game"]].append(i)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(len(games_), size=len(games_), replace=True)
            sel = [i for j in pick for i in idx[games_[j]]]
            boots.append(float(v[sel].mean()))
        return {"n": int(len(v)), "n_games": len(games_), "mean": float(v.mean()),
                "mean_ci95_over_games": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                "median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                "share_negative": float((v < 0).mean()), "share_above_0.5": float((v > 0.5).mean())}

    if items_out:  # the per-item indices, read by the natural bridge's analysis (wf/bridge_analysis.py)
        with open(items_out, "w") as f:
            for r in sorted(rows, key=lambda r: r["id"]):
                f.write(json.dumps({k: r[k] for k in ("id", "location_given_existence", "instance_given_existence",
                                                      "existence", "total")}, sort_keys=True) + "\n")
    out = {"n_items": len(rows), "skipped": dict(skipped)}
    for name, sel in (("all_take_items", rows),
                      ("receptacle_type_not_named_in_history", [r for r in rows if not r["recep_type_named_in_history"]]),
                      ("step0", [r for r in rows if r["history_len"] == 0])):
        out[name] = {k: summary(sel, k) for k in ("location_given_existence", "instance_given_existence", "existence", "total")}
    return out


def analyze(args) -> None:
    games = [json.loads(x) for x in open(args.games)]
    rows = rows_from_games(games)
    report = {"what": "A1' check: placement of the task's object type against the same type in games whose task names "
                      "another object, within floor plans (generator level), and the location-given-existence index "
                      "for RQ2 take items", "seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM,
              "games": dict(Counter(g["split"] for g in games)),
              "games_solvable_flag": dict(Counter(f"{g['split']}|{g['solvable']}" for g in games))}
    for split in (("train", "valid_unseen") if not args.items_only else ()):
        rs = [r for r in rows if r["split"] == split]
        report[f"generator_{split}"] = {"task_object": generator_level(rs, "task"),
                                        "movable_receptacle": generator_level(rs, "mrecep")}
    agree = Counter(room_from_receptacle_types({r["type"].lower() for r in g["receptacles"].values()}) == g["room"]
                    for g in games)
    report["room_rule_agreement"] = {"agree": agree[True], "disagree": agree[False]}
    report["rq2_take_items"] = item_level(games, args.transitions, args.items_out)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "rq2_take_items"}, indent=1)[:4000])
    print(json.dumps(report["rq2_take_items"], indent=1)[:4000])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", default="~/.cache/alfworld")
    e.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--games", required=True)
    a.add_argument("--transitions", default=None)
    a.add_argument("--out", required=True)
    a.add_argument("--items-only", action="store_true", help="skip the generator-level part (a quick check)")
    a.add_argument("--items-out", default=None, help="write the per-item indices here (JSON lines)")
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else analyze(args)


if __name__ == "__main__":
    main()
