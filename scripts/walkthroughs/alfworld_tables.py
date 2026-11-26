"""Print the markdown tables of notes/walkthroughs/01_alfworld.md from the collected sample data.

Every number in the walkthrough's sample tables is printed here from notes/walkthroughs/data/alfworld_samples.json
and alfworld_decisions.json (written by alfworld_samples.py and alfworld_decisions.py from the stored score files),
so no number is typed by hand. CPU only.

    python scripts/walkthroughs/alfworld_tables.py > /tmp/tables.md      # everything, for reading
    python scripts/walkthroughs/alfworld_tables.py --fill notes/walkthroughs/01_alfworld.md

With --fill, the text between each `<!-- TABLE:key -->` and the next `<!-- /TABLE -->` in the walkthrough is
replaced by the freshly generated table of that key (S1_rq2 ... S12_models, W1, W2, stage1a, optionc, stage1b).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "notes" / "walkthroughs" / "data"
Q, L = "qwen3-8b-fewshot", "llama3.1-8b-fewshot"


def f(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "–"
    s = f"{x:+.{nd}f}"
    return "0.00" if s in ("+0.00", "-0.00") else s.replace("-", "−")


def g(rec: dict | None, key: str = "L") -> float | None:
    return None if not rec else rec.get(key)


def model_table(s: dict) -> str:
    out = ["| Model | Two-candidate log-odds, aligned | swapped | neutral | Goal effect (aligned minus swapped) | Yes/no log-odds, aligned | swapped | neutral | Goal effect |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in s["table"]:
        a, b, n = g(r["aligned"]), g(r["swapped"]), g(r["absent"])
        ya, yb, yn = g(r["yn_aligned"]), g(r["yn_swapped"]), g(r["yn_absent"])
        out.append(f"| {r['model']} | {f(a)} | {f(b)} | {f(n)} | {f(a - b) if a is not None and b is not None else '–'} | "
                   f"{f(ya)} | {f(yb)} | {f(yn)} | {f(ya - yb) if ya is not None and yb is not None else '–'} |")
    return "\n".join(out)


def rq2_table(s: dict) -> str:
    out = ["| Model | Task line | Log-probability of the success twin | Log-probability of \"Nothing happens.\" | Two-candidate log-odds | Score (logistic of the log-odds) | First-token log-odds | Written prediction (parser class) |",
           "|---|---|---|---|---|---|---|---|"]
    for tag, name in ((Q, "Qwen3-8B"), (L, "Llama-3.1-8B")):
        for c in ("aligned", "swapped"):
            r = s["rq2"][tag][c]
            gen = s["rq2"][tag]["generations"][c]
            cls = s["rq2"][tag]["generation_classes"][c]
            out.append(f"| {name} | {c} | {f(r['logp_plus'])} | {f(r['logp_minus'])} | {f(r['L'])} | {r['p_success']:.3f} | "
                       f"{f(r['L_first'])} | \"{gen}\" ({cls}) |")
    return "\n".join(out)


def success_table(s: dict) -> str:
    out = ["| Model | Two-candidate log-odds, aligned | swapped | neutral | Yes/no log-odds, aligned | swapped | neutral |", "|---|---|---|---|---|---|---|"]
    for tag, name in ((Q, "Qwen3-8B"), (L, "Llama-3.1-8B")):
        r = s["reference_success"][tag]
        out.append(f"| {name} | " + " | ".join(f(g(r[c])) for c in ("aligned", "swapped", "absent", "yn_aligned", "yn_swapped", "yn_absent")) + " |")
    for r in s["table"][2:]:
        out.append(f"| {r['model']} | {f(g(r['aligned']))} | – | {f(g(r['absent']))} | {f(g(r['yn_aligned']))} | – | {f(g(r['yn_absent']))} |")
    return "\n".join(out)


def stage1a_table(a: dict) -> str:
    out = ["| Candidate | Works? | Acts on the task's object? | Qwen3-8B: real task | swapped | neutral | task-blind | "
           "Llama-3.1-8B: real task | swapped | neutral | task-blind |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in a["candidates"]:
        row = [f"`{c['action']}`" + (" (proposal)" if c["action"] == a["proposal"] else ""), "yes" if c["succeeds"] else "no",
               "yes" if c["goal_object"] else "no"]
        for m in ("qwen3-8b", "llama3.1-8b"):
            sc = a["models"][m]["scores"]
            row += [f(sc[arm].get(c["id"])) for arm in ("yn_aligned", "yn_swapped", "yn_absent", "yn_blind")]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def optionc_table(o: dict) -> str:
    out = ["| Candidate | Yes/no: real task | task-blind | swapped | neutral | Observation readout: real task | task-blind | "
           "swapped | neutral |", "|---|---|---|---|---|---|---|---|---|"]
    keys = ("yn_aware", "yn_blind", "yn_swapped", "yn_neutral", "obs_aware", "obs_blind", "obs_swapped", "obs_neutral")
    for act, sc in o["scores"].items():
        out.append(f"| `{act}` | " + " | ".join(f(sc.get(k)) for k in keys) + " |")
    return "\n".join(out)


def stage1b_table(st: dict) -> str:
    out = ["| Rank | Times sampled (of 20) | Candidate | True reply | Regime candidate? | Llama-3.1-8B yes/no log-odds, "
           "real task | task-blind | swapped | neutral |", "|---|---|---|---|---|---|---|---|---|"]
    for c in st["candidates"]:
        out.append(f"| {c['rank']} | {c['count']} | `{c['action']}` | {c['true_obs']} | {'yes' if c['regime'] else 'no'} | "
                   + " | ".join(f(c["L"][k], 3) for k in ("yn_aligned", "yn_blind", "yn_swapped", "yn_absent")) + " |")
    return "\n".join(out)


def study_tables(S: dict[str, dict]) -> dict[str, str]:
    """Small per-study tables on named samples (S1 = t_002295, S2 = t_1004121, S3 = t_2003104, S7 = t_1003683,
    S8 = t_1001297, S9 = t_004346, S12 = t_001712)."""
    out = {}
    s1, s2, s3, s7, s8, s9, s12 = (S[i] for i in ("t_002295", "t_1004121", "t_2003104", "t_1003683", "t_1001297",
                                                    "t_004346", "t_001712"))
    # RQ1's prompt variants (rq1_score), two-candidate log-odds
    rows = [("Example set 1 (the RQ2 prompt)", "aligned", "swapped"), ("Example set 2", "aligned_set2", "swapped_set2"),
            ("Example set 3", "aligned_set3", "swapped_set3"), ("\"Ignore the goal\" note added", "aligned_ignore", "swapped_ignore"),
            ("Full state (starting layout shown)", "aligned_full", "swapped_full"),
            ("Extra-mention lines (aligned+ and swapped+)", "aligned_plus", "swapped_plus")]
    t = ["| Prompt variant | S1 Qwen3-8B: aligned | swapped | S1 Llama-3.1-8B: aligned | swapped | S2 Qwen3-8B: aligned | swapped | S2 Llama-3.1-8B: aligned | swapped |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, a, b in rows:
        cells = []
        for s in (s1, s2):
            for tag in (Q, L):
                cells += [f(g(s["rq1"][tag].get(a))), f(g(s["rq1"][tag].get(b)))]
        t.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = []
    for s in (s1, s2):
        for tag in (Q, L):
            cells += [f(g(s["rq1"][tag].get("absent"))), "–"]
    t.append("| Neutral task (one prompt) | " + " | ".join(cells) + " |")
    out["rq1_arms"] = "\n".join(t)
    # object-mention 2x2 (rq2_score)
    t = ["| Sample | Model | Aligned (real task) | Mention (object named, task not advanced) | Swapped | Fourth (swapped object, same change) | Goal-directed part | Mention part |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s in (("S7", s7), ("S12", s12)):
        for tag, mn in ((Q, "Qwen3-8B"), (L, "Llama-3.1-8B")):
            r = {c: s["rq2"][tag][c]["L"] for c in ("aligned", "mention", "swapped", "fourth")}
            gd = (r["aligned"] - r["mention"]) - (r["swapped"] - r["fourth"])
            t.append(f"| {name} | {mn} | {f(r['aligned'])} | {f(r['mention'])} | {f(r['swapped'])} | {f(r['fourth'])} | {f(gd)} | {f(r['mention'] - r['fourth'])} |")
    out["twobytwo"] = "\n".join(t)
    # object-free goals (rq1x_score) next to the named goals (rq1_score, same item)
    t = ["| Sample | Model | Named: aligned | Named: swapped | Named: goal effect | Described: aligned | Described: swapped | Described: goal effect |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s in (("S1", s1), ("S3", s3), ("S9", s9)):
        for tag, mn in ((Q, "Qwen3-8B"), (L, "Llama-3.1-8B")):
            a, b = g(s["rq1"][tag]["aligned"]), g(s["rq1"][tag]["swapped"])
            da, db = g(s["rq1x"][tag]["aligned_desc"]), g(s["rq1x"][tag]["swapped_desc"])
            t.append(f"| {name} | {mn} | {f(a)} | {f(b)} | {f(a - b)} | {f(da)} | {f(db)} | {f(da - db)} |")
    out["rq1x"] = "\n".join(t)
    # dissociation (rq1d_score)
    t = ["| Sample | Model | Sentence added to both prompts | Aligned | Swapped | Goal effect |", "|---|---|---|---|---|---|"]
    for name, s in (("S1", s1), ("S3", s3)):
        for tag, mn in ((Q, "Qwen3-8B"), (L, "Llama-3.1-8B")):
            for kind, label in (("", "none"), ("_exist", f"\"{s['task_lines']['rq1d_exist_sentence']}\""),
                                ("_located", f"\"{s['task_lines']['rq1d_located_sentence']}\""),
                                ("_absent", f"\"{s['task_lines']['rq1d_absent_sentence']}\"")):
                a, b = g(s["rq1d"][tag]["aligned" + kind]), g(s["rq1d"][tag]["swapped" + kind])
                t.append(f"| {name} | {mn} | {label} | {f(a)} | {f(b)} | {f(a - b) if a is not None else '–'} |")
    out["rq1d"] = "\n".join(t)
    # counterfactual destination (rq1v_score)
    t = ["| Sample | Model | Real task: aligned | swapped | goal effect | Counterfactual task: aligned | swapped | goal effect |",
         "|---|---|---|---|---|---|---|---|"]
    for name, s in (("S1", s1), ("S3", s3), ("S8", s8)):
        for tag, mn in ((Q, "Qwen3-8B"), (L, "Llama-3.1-8B")):
            a, b = s["rq2"][tag]["aligned"]["L"], s["rq2"][tag]["swapped"]["L"]
            ca, cb = g(s["rq1v"][tag]["cf_aligned"]), g(s["rq1v"][tag]["cf_swapped"])
            t.append(f"| {name} | {mn} | {f(a)} | {f(b)} | {f(a - b)} | {f(ca)} | {f(cb)} | {f(ca - cb)} |")
    out["rq1v_cf"] = "\n".join(t)
    # mechanism diagnostic (mech_score), S1
    t = ["| Arm | Format | Qwen3-8B: aligned | swapped | neutral | Llama-3.1-8B: aligned | swapped | neutral |", "|---|---|---|---|---|---|---|---|"]
    for arm, fmt, label in (("pred_plain", "two-candidate", "plain prediction"), ("pred_facts", "two-candidate", "deciding facts stated"),
                            ("pred_rules", "two-candidate", "rules stated"), ("pred_both", "two-candidate", "facts and rules stated"),
                            ("state", "yes/no", "\"Before this action, is the cabinet 1 open?\" (correct: No)"),
                            ("statetwin", "yes/no", "\"Before this action, is the cabinet 1 closed?\" (correct: Yes)"),
                            ("policy", "agent prompt", "agent's preference for the action over look")):
        cells = []
        for tag in (Q, L):
            cells += [f(g(s1["mech"][tag].get(f"{arm}_{c}"))) for c in ("aligned", "swapped", "absent")]
        t.append(f"| {label} | {fmt} | " + " | ".join(cells) + " |")
    out["mech"] = "\n".join(t)
    # mechanism battery (mech2_score), S9
    arms = ("aligned", "swapped", "mention_ignore", "desire_keep_sw", "desire_check", "desire_check_can", "described", "done",
            "cancelled", "negated", "affirmed")
    t = ["| Arm | Task line | Qwen3-8B: two-candidate | yes/no | Llama-3.1-8B: two-candidate | yes/no | Mistral-7B: two-candidate | yes/no |",
         "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        cells = []
        for tag in (Q, L, "mistral-7b-v0.3"):
            cells += [f(g(s9["mech2"][tag].get(arm))), f(g(s9["mech2"][tag].get("yn_" + arm)))]
        t.append(f"| {arm.replace('_', ' ')} | {s9['task_lines']['mech2_' + arm]} | " + " | ".join(cells) + " |")
    out["mech2"] = "\n".join(t)
    # numerical sensitivity: the same aligned prompt in several runs
    t = ["| Sample | Model | RQ2 run | RQ1 run | Size-series run | Dissociation run | Mechanism run |", "|---|---|---|---|---|---|---|"]
    for name, s in (("S1", s1), ("S2", s2), ("S3", s3)):
        t.append(f"| {name} | Qwen3-8B | {f(s['rq2'][Q]['aligned']['L'], 4)} | {f(s['rq1'][Q]['aligned']['L'], 4)} | "
                 f"{f(s['table'][0]['aligned']['L'], 4)} | {f(s['rq1d'][Q]['aligned']['L'], 4)} | {f(s['mech'][Q]['pred_plain_aligned']['L'], 4)} |")
        t.append(f"| {name} | Llama-3.1-8B | {f(s['rq2'][L]['aligned']['L'], 4)} | {f(s['rq1'][L]['aligned']['L'], 4)} | – | "
                 f"{f(s['rq1d'][L]['aligned']['L'], 4)} | {f(s['mech'][L]['pred_plain_aligned']['L'], 4)} |")
    out["numsens"] = "\n".join(t)
    return out


def fragments() -> dict[str, str]:
    d = json.loads((DATA / "alfworld_samples.json").read_text())
    dd = json.loads((DATA / "alfworld_decisions.json").read_text())
    out = study_tables({s["id"]: s for s in d["samples"]})
    for i, s in enumerate(d["samples"], 1):
        out[f"S{i}_rq2"], out[f"S{i}_models"] = rq2_table(s), model_table(s)
    for j, s in enumerate(d["successes"], 1):
        out[f"W{j}"] = success_table(s)
    out["stage1a"] = stage1a_table(dd["stage1a"])
    out["optionc"] = optionc_table(dd["optionc_step0"])
    out["stage1b"] = stage1b_table(next(x for x in dd["stage1b"]["encounter_states"] if x["state_idx"] == 13))
    return out


def fill(path: Path) -> None:
    """Replace the text between `<!-- TABLE:key -->` and `<!-- /TABLE -->` with the generated table of that key."""
    import re
    frags = fragments()
    used: set[str] = set()

    def sub(m: re.Match) -> str:
        used.add(m[1])
        return f"<!-- TABLE:{m[1]} -->\n{frags[m[1]]}\n<!-- /TABLE -->"
    text = re.sub(r"<!-- TABLE:(\w+) -->.*?<!-- /TABLE -->", sub, path.read_text(), flags=re.S)
    path.write_text(text)
    print(f"filled {len(used)} tables in {path}; not used: {sorted(set(frags) - used)}")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--fill":
        fill(Path(sys.argv[2]))
        return
    d = json.loads((DATA / "alfworld_samples.json").read_text())
    for i, s in enumerate(d["samples"], 1):
        print(f"\n===== S{i} {s['id']} {s['game_id']} step {s['state_idx']} {s['reason']}\n")
        print(rq2_table(s))
        print()
        print(model_table(s))
    for s in d["successes"]:
        print(f"\n===== success {s['id']} {s['game_id']} step {s['state_idx']}\n")
        print(success_table(s))
    # study-specific rows
    S = {s["id"]: s for s in d["samples"]}
    print("\n===== rq1 arms (S1, S2)")
    for sid in ("t_002295", "t_1004121"):
        s = S[sid]
        for tag in (Q, L):
            r = s["rq1"][tag]
            print(sid, tag, {k: round(v["L"], 2) for k, v in r.items() if isinstance(v, dict) and "L" in v})
            print("   ", r["generations"], r["generation_classes"])
        for tag in ("w2w-qwen2.5-7b", "w2w-llama3.1-8b"):
            r = s["rq1"][tag]
            print(sid, tag, {k: round(v["L"], 2) for k, v in r.items() if isinstance(v, dict) and "L" in v}, r["generations"]["aligned"])
    print("\n===== other study arms, all samples")
    for i, s in enumerate(d["samples"], 1):
        print(f"S{i}", s["id"], "rq1x", {t: {k: (round(v['L'], 2) if v else None) for k, v in c.items()} for t, c in s["rq1x"].items()})
        print("    rq1v", {t: {k: (round(v['L'], 2) if v else None) for k, v in c.items()} for t, c in s["rq1v"].items()})
        print("    rq1d", {t: {k: (round(v['L'], 2) if v else None) for k, v in c.items()} for t, c in s["rq1d"].items()})
        print("    rq2 mention/fourth", {t: {k: (round(s['rq2'][t][k]['L'], 2) if s['rq2'][t][k] else None) for k in ('mention', 'fourth')} for t in (Q, L)})
        print("    mech2 in battery:", bool(s["mech2"][Q]))
    dd = json.loads((DATA / "alfworld_decisions.json").read_text())
    a = dd["stage1a"]
    print("\n===== stage 1a", a["game_id"], a["state_idx"], a["proposal"], a["swap"])
    print("| Candidate | Works? | Acts on the task's object? | Qwen3-8B: real task | swapped | neutral | task-blind | Llama-3.1-8B: real task | swapped | neutral | task-blind |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in a["candidates"]:
        row = [f"`{c['action']}`" + (" (proposal)" if c["action"] == a["proposal"] else ""), "yes" if c["succeeds"] else "no",
               "yes" if c["goal_object"] else "no"]
        for m in ("qwen3-8b", "llama3.1-8b"):
            sc = a["models"][m]["scores"]
            row += [f(sc[arm].get(c["id"])) for arm in ("yn_aligned", "yn_swapped", "yn_absent", "yn_blind")]
        print("| " + " | ".join(row) + " |")
    for m in ("qwen3-8b", "llama3.1-8b"):
        print(m, a["models"][m]["rule2_choice"])
    o = dd["optionc_step0"]
    print("\n===== option C step 0")
    print("| Candidate | Yes/no: real task | task-blind | swapped | neutral | Observation readout: real task | task-blind | swapped | neutral |")
    print("|---|---|---|---|---|---|---|---|---|")
    for act, sc in o["scores"].items():
        print(f"| `{act}` | " + " | ".join(f(sc.get(k)) for k in ("yn_aware", "yn_blind", "yn_swapped", "yn_neutral", "obs_aware", "obs_blind", "obs_swapped", "obs_neutral")) + " |")
    b = dd["stage1b"]
    st = next(x for x in b["encounter_states"] if x["state_idx"] == 13)
    print("\n===== stage 1b", b["game_id"], st["state_idx"], st["task_line"], "aware:", st["aware_choice"], "blind:", st["blind_choice"])
    print("| Rank | Times sampled (of 20) | Candidate | True reply | Regime candidate? | Llama-3.1-8B yes/no log-odds, real task | task-blind | swapped | neutral |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in st["candidates"]:
        print(f"| {c['rank']} | {c['count']} | `{c['action']}` | {c['true_obs']} | {'yes' if c['regime'] else 'no'} | " +
              " | ".join(f(c["L"][k], 3) for k in ("yn_aligned", "yn_blind", "yn_swapped", "yn_absent")) + " |")
    print("\n".join(st["history_text"].splitlines()[-8:]))


if __name__ == "__main__":
    main()
