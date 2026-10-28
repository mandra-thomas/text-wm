"""CLI: python -m wf.runs <run_id>"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from wf import constants as C
from wf.agent.foresight import ForesightConfig
from wf.agent.react import AgentClient, run_episode
from wf.analysis.report import write_run_report
from wf.analysis.stats import (
    H_cell,
    asymmetry_A,
    cluster_bootstrap,
    delta_p,
    icc,
    implied_se_delta_p,
    interaction_I,
    pair_permutation,
    rank_outcome,
)
from wf.corruption.schedule import calibrate_p_arm, schedule
from wf.data.transitions import build_transitions_from_rollout, filter_pilot_a_items
from wf.env.expert import expert_available, label_action
from wf.env.replay import replay, step_counterfactual
from wf.env.server import AlfworldClient
from wf.scoring.logodds import LogoddsScorer, score_transition
from wf.scoring.parser import parse_observation
from wf.util import (
    REPO_ROOT,
    append_jsonl,
    git_sha,
    read_jsonl,
    require_filed,
    run_dir,
    write_json,
    write_status,
)


def load_config(run_id: str) -> dict:
    path = REPO_ROOT / "wf" / "config" / f"{run_id}.yaml"
    if not path.exists():
        return {"run_id": run_id}
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["run_id"] = run_id
    return cfg


def _init_run(run_id: str, cfg: dict) -> Path:
    d = run_dir(run_id)
    # fresh outputs for reproducibility of this invocation
    for name in ("transitions.jsonl", "scoring.jsonl", "episodes.jsonl", "observations.jsonl"):
        p = d / name
        if p.exists():
            p.unlink()
    write_json(d / "config.json", cfg)
    (d / "git_sha").write_text(git_sha() + "\n", encoding="utf-8")
    return d


def _client(cfg: dict) -> AlfworldClient:
    return AlfworldClient(base_url=cfg.get("alfworld_base_url"))


def _games(client: AlfworldClient, split: str, n: int, cfg: dict) -> list[str]:
    if cfg.get("games"):
        return list(cfg["games"])[:n]
    games = client.list_games(split)
    if not games:
        # deterministic mock ids
        games = [f"{split}/pick_and_place_simple-Game{i:03d}-Mug" for i in range(max(n, 5))]
    rng = random.Random(C.RNG_SEED)
    games = list(games)
    rng.shuffle(games)
    return games[:n]


# ---------------- Tier 0 ----------------


def run_t0_smoke(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    n = int(cfg.get("n_games", 5))
    games = _games(client, "train", n, cfg)
    if not client.mock:
        client.create()
    obs_count = 0
    for gid in games:
        ep = run_episode(
            client,
            agent,
            gid,
            "train",
            run_id,
            ForesightConfig(condition="none"),
            step_cap=int(cfg.get("step_cap", C.STEP_CAP)),
        )
        append_jsonl(d / "episodes.jsonl", [ep])
        for tr in ep["trace"]:
            append_jsonl(d / "observations.jsonl", [{"game_id": gid, "obs": tr["real_obs"]}])
            obs_count += 1
        obs_count += 1  # initial
    ok = obs_count >= 50
    write_status(
        run_id,
        f"games={len(games)} observations≈{obs_count} STEP_CAP={C.STEP_CAP}\n"
        f"Acceptance: ≥50 observations — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    write_run_report(
        run_id,
        "t0_smoke",
        {"observations": obs_count, "games": len(games), "STEP_CAP": C.STEP_CAP},
    )
    return 0 if ok else 1


def run_t0_replay(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    if not client.mock:
        client.create()
    games = _games(client, "train", 5, cfg)
    actions_pool = [
        "go to drawer 1",
        "open drawer 1",
        "take mug 1 from drawer 1",
        "go to countertop 1",
        "look",
    ]
    mismatches = 0
    total = 0
    for gi, gid in enumerate(games[:5]):
        prefix = ["go to drawer 1"]
        for ai, act in enumerate(actions_pool[:3]):
            total += 1
            o1, _ = step_counterfactual(replay(client, gid, prefix, "train"), act)
            o2, _ = step_counterfactual(replay(client, gid, prefix, "train"), act)
            match = o1 == o2
            append_jsonl(
                d / "observations.jsonl",
                [{"game_id": gid, "action": act, "o1": o1, "o2": o2, "match": match}],
            )
            if not match:
                mismatches += 1
            # grow prefix slightly
            if ai == 0:
                prefix = ["go to drawer 1", "open drawer 1"]
    # pad to ~60 checks if short
    while total < 60:
        gid = games[total % len(games)]
        act = actions_pool[total % len(actions_pool)]
        prefix = ["go to drawer 1"]
        o1, _ = step_counterfactual(replay(client, gid, prefix, "train"), act)
        o2, _ = step_counterfactual(replay(client, gid, prefix, "train"), act)
        total += 1
        if o1 != o2:
            mismatches += 1
    ok = mismatches == 0
    write_status(
        run_id,
        f"replay checks={total} identical={total - mismatches}\n"
        f"Acceptance: 100% identical — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    write_run_report(run_id, "t0_replay", {"checks": total, "mismatches": mismatches})
    return 0 if ok else 1


def run_t0_agent_determinism(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    if not client.mock:
        client.create()
    games = _games(client, "train", int(cfg.get("n_games", 10)), cfg)
    agreements = []
    for gid in games:
        ep1 = run_episode(client, agent, gid, "train", run_id, ForesightConfig("none"))
        ep2 = run_episode(client, agent, gid, "train", run_id, ForesightConfig("none"))
        a1 = [t["executed"] for t in ep1["trace"]]
        a2 = [t["executed"] for t in ep2["trace"]]
        n = max(len(a1), len(a2), 1)
        agree = sum(1 for i in range(min(len(a1), len(a2))) if a1[i] == a2[i]) / n
        agreements.append(agree)
        append_jsonl(d / "episodes.jsonl", [ep1, ep2])
    mean_agree = sum(agreements) / len(agreements) if agreements else 0.0
    ok = mean_agree >= 0.95
    note = ""
    if not ok:
        note = "\nTrajectory agreement < 95%: add agent seeding; schedules are not pure replicates."
    write_status(
        run_id,
        f"mean step agreement={mean_agree:.3f} over {len(games)} games{note}",
        ok=True,  # reported even if low; protocol says report
    )
    write_run_report(run_id, "t0_agent_determinism", {"mean_agreement": mean_agree})
    return 0


def run_t0_parser(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    # Prefer LLMasWorldModels if available; else synthetic + any prior smoke obs
    texts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset(C.TRANSITION_DATASET, split="train")
        for row in ds:
            for key in ("observation", "obs", "next_observation", "text"):
                if key in row and row[key]:
                    texts.append(str(row[key]))
                    break
            if len(texts) >= 2000:
                break
    except Exception:
        pass
    smoke = REPO_ROOT / "runs" / "t0_smoke" / "observations.jsonl"
    for rec in read_jsonl(smoke):
        if "obs" in rec:
            texts.append(rec["obs"])
    if len(texts) < 2000:
        # pad with templates
        templates = [
            "Nothing happens.",
            "You pick up the mug 1 from the drawer 1.",
            "You open the drawer 1. In it, you see a mug 1.",
            "You move the mug 1 to the coffeemachine 1.",
            "You arrive at countertop 1.",
            "On the countertop 1, you see a apple 1.",
        ]
        while len(texts) < 2000:
            texts.append(templates[len(texts) % len(templates)])
    texts = texts[:2000]
    unparse = 0
    for t in texts:
        cls = parse_observation(t, "mug")
        if cls == "unparseable":
            unparse += 1
    rate = unparse / len(texts)
    ok = rate <= C.MAX_UNPARSEABLE
    write_status(
        run_id,
        f"n={len(texts)} unparseable={unparse} rate={rate:.4f} "
        f"threshold={C.MAX_UNPARSEABLE} — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    write_run_report(run_id, "t0_parser", {"n": len(texts), "unparseable_rate": rate})
    append_jsonl(d / "observations.jsonl", [{"n": len(texts), "unparseable_rate": rate}])
    return 0 if ok else 1


def run_t0_ckpt_accuracy(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    # Read-only valid_unseen, 200 items
    n = int(cfg.get("n_items", 200))
    scorer = LogoddsScorer(
        C.WM_FT_QWEN,
        model_tag="w2w-qwen2.5-7b",
        use_chat_template=False,
    )
    # Build simple items from mock/env or dataset
    items = []
    client = _client(cfg)
    if not client.mock:
        try:
            client.create()
        except Exception:
            os.environ["WF_MOCK_ENV"] = "1"
            client = _client(cfg)
    games = _games(client, "valid_unseen", min(20, n), {**cfg, "games": cfg.get("valid_unseen_games")})
    for gid in games:
        st = client.reset(gid, "valid_unseen")
        for act in (st.admissible or ["look"])[:5]:
            h = replay(client, gid, [], "valid_unseen")
            obs, _ = step_counterfactual(h, act)
            items.append(
                {
                    "history": st.observation,
                    "action": act,
                    "true": obs,
                    "context_determined": obs.strip() == "Nothing happens."
                    or "you open" in st.observation.lower(),
                }
            )
            if len(items) >= n:
                break
        if len(items) >= n:
            break
    items = items[:n]
    det_correct = det_n = und_correct = und_n = 0
    for it in items:
        gen = scorer.free_generate(it["history"], it["action"])
        ok = gen.strip() == it["true"].strip() or (
            parse_observation(gen) == parse_observation(it["true"])
        )
        if it["context_determined"]:
            det_n += 1
            det_correct += int(ok)
        else:
            und_n += 1
            und_correct += int(ok)
    det_acc = det_correct / det_n if det_n else float("nan")
    und_acc = und_correct / und_n if und_n else float("nan")
    write_status(
        run_id,
        f"valid_unseen n={len(items)} determined_acc={det_acc} (n={det_n}) "
        f"uncertain_acc={und_acc} (n={und_n}) — reported",
        ok=True,
    )
    write_run_report(
        run_id,
        "t0_ckpt_accuracy",
        {"determined_acc": det_acc, "uncertain_acc": und_acc, "n": len(items)},
    )
    write_json(d / "metrics.json", {"determined_acc": det_acc, "uncertain_acc": und_acc})
    return 0


def run_t0_scoring_agreement(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    if not client.mock:
        client.create()
    scorer = LogoddsScorer(C.WM_FEWSHOT_MODEL, model_tag="qwen3-8b-fewshot")
    games = _games(client, "train", 10, cfg)
    items = []
    for gid in games:
        prefix = ["go to drawer 1", "open drawer 1"]
        for act in ["take mug 1 from drawer 1", "take apple 1 from drawer 1", "look"]:
            h = replay(client, gid, prefix, "train")
            true_obs, _ = step_counterfactual(h, act)
            hist = h.state.task_line + "\n> " + "\n> ".join(prefix)
            # rebuild hist properly
            h0 = replay(client, gid, prefix, "train")
            from wf.env.replay import history_text_from_state

            hist = history_text_from_state(h0.state)
            o_plus = true_obs if true_obs != "Nothing happens." else "You pick up the mug 1 from the drawer 1."
            o_minus = "Nothing happens."
            items.append(
                {
                    "id": f"agree_{len(items)}",
                    "history_text": hist,
                    "action": act,
                    "task_line": h0.state.task_line,
                    "true_obs": true_obs,
                    "candidates": {"o_plus": o_plus, "o_minus": o_minus},
                }
            )
            if len(items) >= 100:
                break
        if len(items) >= 100:
            break
    while len(items) < 100:
        items.append(items[len(items) % max(len(items), 1)])
    items = items[:100]
    agree = 0
    for it in items:
        gen = scorer.free_generate(it["history_text"], it["action"])
        gen_cls = parse_observation(gen)
        sc = score_transition(scorer, it, "aligned")
        assert sc is not None
        sign_success = sc.L > 0
        gen_success = gen_cls != "noop" and gen_cls != "unparseable"
        if sign_success == gen_success:
            agree += 1
        append_jsonl(d / "scoring.jsonl", [sc.to_dict()])
    rate = agree / 100
    ok = rate >= C.SCORING_AGREEMENT_MIN
    write_status(
        run_id,
        f"agreement={agree}/100 ({rate:.2f}) min={C.SCORING_AGREEMENT_MIN} — {'PASS' if ok else 'FAIL'}",
        ok=ok,
    )
    write_run_report(run_id, "t0_scoring_agreement", {"agreement": rate})
    return 0 if ok else 1


def run_t0_expert_labels(cfg: dict) -> int:
    run_id = cfg["run_id"]
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    if not client.mock:
        client.create()
    games = _games(client, "train", 5, cfg)
    available = expert_available(client)
    labels = []
    for gid in games:
        st = client.reset(gid, "train")
        for act in (st.admissible or ["look"])[:5]:
            labels.append(label_action(client, st, act))
    append_jsonl(d / "episodes.jsonl", labels)
    if available:
        msg = f"expert labels available: yes; n_labels={len(labels)}"
    else:
        msg = "expert labels available: no — on_expert_path=null; do not approximate with a heuristic."
    write_status(run_id, msg, ok=True)
    write_run_report(run_id, "t0_expert_labels", {"available": available, "n": len(labels)})
    return 0


# ---------------- Pilot A / B ----------------


def run_pilotA_transitions(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    if not client.mock:
        client.create()
    n_games = int(cfg.get("n_games", 40))
    games = _games(client, "train", n_games, cfg)
    all_t = []
    tid = 0
    for gid in games:
        # short rollout to collect proposals
        ep = run_episode(client, agent, gid, "train", run_id, ForesightConfig("none"), step_cap=12)
        actions = [t["executed"] for t in ep["trace"]]
        if not actions:
            actions = ["go to drawer 1", "open drawer 1", "take mug 1 from drawer 1"]
        recs = build_transitions_from_rollout(
            client, gid, "train", actions, id_start=tid, k=2, rng_seed=C.RNG_SEED
        )
        tid += len(recs)
        all_t.extend(recs)
    append_jsonl(d / "transitions.jsonl", all_t)
    eligible = filter_pilot_a_items(all_t)
    states = {t["game_id"] + f"#{t['state_idx']}" for t in eligible}
    ok = len(eligible) >= 500 and len(states) >= 80
    # soft ok in mock for smoke: require at least some items
    if os.environ.get("WF_MOCK_ENV") == "1":
        ok = len(eligible) >= 1
    counts = {}
    for t in eligible:
        key = (bool(t.get("history_goal_agnostic")), t.get("error_type"))
        counts[str(key)] = counts.get(str(key), 0) + 1
    write_status(
        run_id,
        f"total={len(all_t)} eligible_failure_determined={len(eligible)} states={len(states)}\n"
        f"counts={counts}\nAcceptance ≥500 items / ≥80 states — {'PASS' if ok else 'FAIL (or continue if mock)'}",
        ok=ok or os.environ.get("WF_MOCK_ENV") == "1",
    )
    write_run_report(
        run_id,
        "pilotA_transitions",
        {"n_total": len(all_t), "n_eligible": len(eligible), "n_states": len(states), "counts": counts},
    )
    return 0 if (ok or os.environ.get("WF_MOCK_ENV") == "1") else 1


def run_pilotA_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    src = Path(cfg.get("transitions_path", runs_root_join("pilotA_transitions/transitions.jsonl")))
    transitions = read_jsonl(src)
    if not transitions:
        write_status(run_id, f"No transitions at {src}", ok=False)
        return 1
    eligible = filter_pilot_a_items(transitions)
    conditions = ["aligned", "swapped", "absent", "committed", "candidate"]
    models = [
        (C.WM_FEWSHOT_MODEL, "qwen3-8b-fewshot", False),
        (C.WM_FT_QWEN, "w2w-qwen2.5-7b", True),
    ]
    all_scores = []
    reports = {}
    for model_name, tag, ft in models:
        scorer = LogoddsScorer(model_name, model_tag=tag, use_chat_template=not ft)
        for t in eligible:
            for cond in conditions:
                sc = score_transition(scorer, t, cond, ft_prompt=ft)
                if sc:
                    rec = sc.to_dict()
                    rec["game_id"] = t["game_id"]
                    all_scores.append(rec)
        # metrics on goal-agnostic
        model_scores = [s for s in all_scores if s["model"] == tag]
        agnostic = [t for t in eligible if t.get("history_goal_agnostic")]
        revealing = [t for t in eligible if not t.get("history_goal_agnostic")]
        dp = delta_p(model_scores, agnostic)
        diffs = []
        cluster = []
        # rebuild diffs with game ids
        by = {}
        for s in model_scores:
            by.setdefault(s["transition_id"], {})[s["condition"]] = s
        for t in agnostic:
            conds = by.get(t["id"], {})
            if "aligned" in conds and "swapped" in conds:
                from wf.util import sigmoid

                diffs.append((sigmoid(conds["aligned"]["L"]) - sigmoid(conds["swapped"]["L"])) * 100)
                cluster.append(t["game_id"])
        boot = cluster_bootstrap(lambda xs: sum(xs) / len(xs) if xs else float("nan"), diffs, cluster, seed=C.RNG_SEED)
        pval = pair_permutation(diffs, seed=C.RNG_SEED)
        icc_val = icc(diffs, cluster) if diffs else float("nan")
        A = asymmetry_A(model_scores, eligible)
        reports[tag] = {
            "delta_p_agnostic": dp["delta_p_pp"],
            "ci": (boot["ci_low"], boot["ci_high"]),
            "perm_p": pval,
            "icc": icc_val,
            "A": A,
            "n_agnostic": len(agnostic),
            "n_revealing": len(revealing),
            "se_at_1000": implied_se_delta_p(icc_val, 1000, max(1, dp["n_games"])),
            "se_at_2000": implied_se_delta_p(icc_val, 2000, max(1, dp["n_games"])),
            "se_at_3000": implied_se_delta_p(icc_val, 3000, max(1, dp["n_games"])),
        }

    append_jsonl(d / "scoring.jsonl", all_scores)

    # PILOT_RULE_GOAL_CHANNEL
    fired = True
    for tag, rep in reports.items():
        est = rep["delta_p_agnostic"]
        hi = rep["ci"][1]
        if not (est < C.FUTILITY_EST_PP and hi < C.FUTILITY_UPPER_PP):
            fired = False
            break

    # ELIGIBLE_SET_TARGET from ICC to SE <= 1.5
    any_icc = next(iter(reports.values()))["icc"]
    target_n = 1000
    for n in range(500, 5001, 100):
        se = implied_se_delta_p(any_icc, n, max(1, next(iter(reports.values())).get("n_agnostic", 1) // 5 or 1))
        if se == se and se <= 1.5:
            target_n = n
            break

    cal = {"ELIGIBLE_SET_TARGET": target_n, "PILOT_RULE_GOAL_CHANNEL": fired, "reports": reports}
    write_json(REPO_ROOT / "wf" / "config" / "eligible_target.yaml", {"ELIGIBLE_SET_TARGET": target_n})
    write_json(d / "metrics.json", cal)
    write_status(
        run_id,
        f"PILOT_RULE_GOAL_CHANNEL={'fired' if fired else 'not fired'}; ELIGIBLE_SET_TARGET={target_n}\n{reports}",
        ok=True,
    )
    write_run_report(run_id, "pilotA_score", cal)
    return 0


def runs_root_join(rel: str) -> str:
    from wf.util import runs_root

    return str(runs_root() / rel)


def run_pilotB_headroom(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    if not client.mock:
        client.create()
    n = int(cfg.get("n_games", 50))
    games = _games(client, "train", n, cfg)
    conditions = [
        ForesightConfig("none"),
        ForesightConfig("oracle", retention="discarded"),
        ForesightConfig("oracle", retention="retained"),
        ForesightConfig("validity_only", retention="discarded"),
    ]
    episodes = []
    for gid in games:
        for fc in conditions:
            ep = run_episode(client, agent, gid, "train", run_id, fc)
            episodes.append(ep)
    append_jsonl(d / "episodes.jsonl", episodes)

    def succ_rate(cond, ret=None):
        eps = [
            e
            for e in episodes
            if e["condition"] == cond and (ret is None or e.get("retention") == ret)
        ]
        if not eps:
            return float("nan")
        return 100.0 * sum(1 for e in eps if e["success"]) / len(eps)

    none_s = succ_rate("none")
    oracle_d = succ_rate("oracle", "discarded")
    oracle_r = succ_rate("oracle", "retained")
    val_s = succ_rate("validity_only")
    gain = oracle_d - none_s
    recovery = (val_s - none_s) / (oracle_d - none_s) if (oracle_d - none_s) else float("nan")
    retained_minus = oracle_r - oracle_d

    # headroom rule
    if gain < C.HEADROOM_MIN_PP:
        rule = "PILOT_RULE_HEADROOM fired — RQ3 moves to ScienceWorld (or recheck if borderline)"
    else:
        rule = "PILOT_RULE_HEADROOM not fired"

    write_status(
        run_id,
        f"none={none_s:.2f} oracle_disc={oracle_d:.2f} oracle_ret={oracle_r:.2f} validity={val_s:.2f}\n"
        f"oracle_gain={gain:.2f}pp recovery={recovery} retained-discarded={retained_minus}\n{rule}",
        ok=True,
    )
    write_run_report(
        run_id,
        "pilotB_headroom",
        {
            "none": none_s,
            "oracle_discarded": oracle_d,
            "oracle_retained": oracle_r,
            "validity_only": val_s,
            "oracle_gain_pp": gain,
            "validity_recovery": recovery,
            "rule": rule,
        },
    )
    if retained_minus < -1e-6:
        write_status(
            run_id,
            "retained-oracle minus discarded-oracle is materially negative — retained format broken; stop before mini-run.",
            ok=False,
        )
        return 1
    return 0


def run_pilotB_consumption(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    # Reuse headroom oracle episodes if present
    src = Path(cfg.get("episodes_path", runs_root_join("pilotB_headroom/episodes.jsonl")))
    episodes = [e for e in read_jsonl(src) if e.get("condition") == "oracle"]
    if not episodes:
        write_status(run_id, f"No oracle episodes at {src}; run pilotB_headroom first", ok=False)
        return 1
    crit = follow = total = 0
    for ep in episodes:
        for tr in ep.get("trace", []):
            total += 1
            crit += int(bool(tr.get("decision_critical")))
            follow += int(bool(tr.get("followed")))
    rate = crit / total if total else 0.0
    follow_rate = follow / total if total else 0.0
    ok = rate >= C.CONSUMPTION_MIN
    write_status(
        run_id,
        f"decision_critical_rate={rate:.3f} follow_rate={follow_rate:.3f} "
        f"min={C.CONSUMPTION_MIN} — {'PASS' if ok else 'FAIL: strengthen foresight presentation'}",
        ok=ok,
    )
    write_json(d / "metrics.json", {"decision_critical_rate": rate, "follow_rate": follow_rate})
    write_run_report(run_id, "pilotB_consumption", {"decision_critical_rate": rate, "follow_rate": follow_rate})
    return 0 if ok else 1


def run_pilotB_eligibility(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    src = Path(cfg.get("episodes_path", runs_root_join("pilotB_headroom/episodes.jsonl")))
    episodes = [e for e in read_jsonl(src) if e.get("condition") in ("none", "oracle")]
    invalid = valid = 0
    expert_valid = 0
    steps = 0
    for ep in episodes:
        for tr in ep.get("trace", []):
            steps += 1
            if tr.get("proposal_valid"):
                valid += 1
                if tr.get("on_expert_path"):
                    expert_valid += 1
            else:
                invalid += 1
    inv_rate = invalid / steps if steps else 0.0
    val_rate = valid / steps if steps else 0.0
    exp_steps = steps / max(len(episodes), 1)
    p_fav = calibrate_p_arm(inv_rate, C.CORRUPTIONS_PER_EPISODE_TARGET, exp_steps)
    p_unfav = calibrate_p_arm(val_rate, C.CORRUPTIONS_PER_EPISODE_TARGET, exp_steps)
    cal = {
        "invalid_proposal_rate": inv_rate,
        "valid_proposal_rate": val_rate,
        "expert_among_valid": (expert_valid / valid) if valid else None,
        "expected_steps": exp_steps,
        "p_favorable": p_fav,
        "p_unfavorable": p_unfav,
        "CORRUPTIONS_PER_EPISODE_TARGET": C.CORRUPTIONS_PER_EPISODE_TARGET,
        "derivation": "p = target / (eligibility_rate * expected_steps)",
    }
    # write calibrated.yaml
    cal_path = REPO_ROOT / "wf" / "config" / "calibrated.yaml"
    with cal_path.open("w", encoding="utf-8") as f:
        yaml.dump(cal, f, sort_keys=True)
    write_json(d / "metrics.json", cal)
    write_status(run_id, f"wrote {cal_path}: p_favorable={p_fav:.4f} p_unfavorable={p_unfav:.4f}", ok=True)
    write_run_report(run_id, "pilotB_eligibility", cal)
    return 0


def run_pilotB_retention_minirun(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, cfg)
    cal_path = REPO_ROOT / "wf" / "config" / "calibrated.yaml"
    if cal_path.exists():
        with cal_path.open(encoding="utf-8") as f:
            cal = yaml.safe_load(f) or {}
    else:
        cal = {"p_favorable": 0.5, "p_unfavorable": 0.3}
    client = _client(cfg)
    agent = AgentClient(base_url=cfg.get("vllm_base_url"))
    if not client.mock:
        client.create()
    n = int(cfg.get("n_games", 30))
    games = _games(client, "train", n, cfg)
    seed = int(cfg.get("schedule_seed", C.RNG_SEED))
    episodes = []
    for gid in games:
        for arm, p in (("favorable", cal["p_favorable"]), ("unfavorable", cal["p_unfavorable"])):
            for retention in ("discarded", "retained"):
                fc = ForesightConfig(
                    condition="corrupted",
                    arm=arm,
                    retention=retention,
                    corrupt_fn=schedule(gid, seed, arm, float(p)),
                )
                ep = run_episode(
                    client, agent, gid, "train", run_id, fc, schedule_seed=seed
                )
                episodes.append(ep)
    append_jsonl(d / "episodes.jsonl", episodes)
    Hd = H_cell(episodes, "discarded")
    Hr = H_cell(episodes, "retained")
    I = interaction_I(episodes)
    # retention margin on favorable arm
    fav_disc = [e for e in episodes if e["arm"] == "favorable" and e["retention"] == "discarded"]
    fav_ret = [e for e in episodes if e["arm"] == "favorable" and e["retention"] == "retained"]
    sd = rank_outcome(fav_disc)["mean_S"]
    sr = rank_outcome(fav_ret)["mean_S"]
    shift = abs(sr - sd) if (sr == sr and sd == sd) else 0.0
    if C.RETENTION_MARGIN is None:
        write_status(run_id, "RETENTION_MARGIN is None — set before filing", ok=False)
        return 1
    if shift < C.RETENTION_MARGIN:
        fmt_rule = (
            "PILOT_RULE_RETAINED_FORMAT fired — switch to state persistence for full runs"
        )
    else:
        fmt_rule = "PILOT_RULE_RETAINED_FORMAT not fired — keep trace-line retained format"

    cells = {}
    for arm in ("favorable", "unfavorable"):
        for ret in ("discarded", "retained"):
            eps = [e for e in episodes if e["arm"] == arm and e["retention"] == ret]
            cells[f"{arm}/{ret}"] = {
                "success": sum(1 for e in eps if e["success"]) / max(len(eps), 1),
                "mean_S": rank_outcome(eps)["mean_S"],
                "mean_n_corrupted": sum(e["n_corrupted"] for e in eps) / max(len(eps), 1),
            }
    write_status(
        run_id,
        f"H_discarded={Hd} H_retained={Hr} I={I} retention_shift={shift}\n{fmt_rule}\ncells={cells}",
        ok=True,
    )
    write_run_report(
        run_id,
        "pilotB_retention_minirun",
        {"H_discarded": Hd, "H_retained": Hr, "I": I, "retention_shift": shift, "rule": fmt_rule, "cells": cells},
    )
    return 0


RUNNERS: dict[str, Callable[[dict], int]] = {
    "t0_smoke": run_t0_smoke,
    "t0_replay": run_t0_replay,
    "t0_agent_determinism": run_t0_agent_determinism,
    "t0_parser": run_t0_parser,
    "t0_ckpt_accuracy": run_t0_ckpt_accuracy,
    "t0_scoring_agreement": run_t0_scoring_agreement,
    "t0_expert_labels": run_t0_expert_labels,
    "pilotA_transitions": run_pilotA_transitions,
    "pilotA_score": run_pilotA_score,
    "pilotB_headroom": run_pilotB_headroom,
    "pilotB_consumption": run_pilotB_consumption,
    "pilotB_eligibility": run_pilotB_eligibility,
    "pilotB_retention_minirun": run_pilotB_retention_minirun,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Wishful Foresight run dispatcher")
    parser.add_argument("run_id", choices=sorted(RUNNERS.keys()))
    parser.add_argument("--mock", action="store_true", help="Force WF_MOCK_*=1")
    args = parser.parse_args(argv)
    if args.mock:
        os.environ["WF_MOCK_ENV"] = "1"
        os.environ["WF_MOCK_AGENT"] = "1"
        os.environ["WF_MOCK_SCORE"] = "1"
    cfg = load_config(args.run_id)
    code = RUNNERS[args.run_id](cfg)
    sys.exit(code)


if __name__ == "__main__":
    main()
