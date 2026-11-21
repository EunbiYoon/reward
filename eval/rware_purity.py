"""RWARE purity check, in the same form as the Overcooked one (paper Table tab:solo):
every judge-pool checkpoint plays the evaluation games (seeds 10_000 + k, as eval/coalitions.py) twice,
    with    next to the two scripted greedy robots
    alone   with both greedy robots frozen ('stay': parked on a highway corner, never moving)
and we count the learner's work steps per game (shelf lifts, requested or not, plus deliveries) and the share of
steps it stands next to a partner. Writes one row per checkpoint and condition to --csv; eval/paper_check.py
turns it into docs/paper/figures/tab_rware_solo.tex.

    python -m eval.rware_purity --workers 4

CPU only; 24 checkpoints x 2 conditions x --episodes games of 500 steps.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import replace
from multiprocessing import get_context

ALGOS = ("ppo", "grpo")
POOLS = {"ppo": "results/judge/rware3_cramped_ppo/ckpts", "grpo": "results/judge/rware3_cramped_grpo/ckpts"}
CONFIGS = {"ppo": "results/train/rware/ppo/config.json", "grpo": "results/train/rware/grpo/config.json"}


def play_checkpoint(args) -> list[dict]:
    path, cfg_path, episodes = args
    import torch
    torch.set_num_threads(1)
    from train.config import load_config
    from train.deep_policy import load_checkpoint
    from train.resume import base_env
    from train.run import make_env
    from .coalitions import EVAL_SEED0, policy_actor, torch_reseed

    cfg = load_config(cfg_path)
    policy, obs_rms, m = load_checkpoint(path)
    c = replace(cfg, pickup_bonus=m.get("pickup_bonus", cfg.pickup_bonus),
                partner_noise=m.get("partner_noise", cfg.partner_noise))
    env = make_env(c, m["reward"], m["partner"], m.get("agents", 3))
    core = base_env(env)
    actor = policy_actor(policy, obs_rms, cfg.eval_deterministic)
    rows = []
    for cond, partners in (("with", "greedy"), ("alone", "stay")):
        tot = {"lifts_requested": 0.0, "lifts_other": 0.0, "deliveries": 0.0, "near": 0.0, "team": 0.0}
        for k in range(episodes):
            torch_reseed(EVAL_SEED0 + k)
            ep = core.play(actor, partners, EVAL_SEED0 + k)
            tot["lifts_requested"] += ep["requested_pickups0"]
            tot["lifts_other"] += ep["other_pickups0"]
            tot["deliveries"] += ep["own_soups"]
            tot["near"] += ep["near_steps0"] / core.horizon
            tot["team"] += ep["soups"]
        row = {k: round(v / episodes, 4) for k, v in tot.items()}
        row["steps"] = round(row["lifts_requested"] + row["lifts_other"] + row["deliveries"], 4)
        rows.append({"algo": m["algo"], "persona": m["reward"], "seed": m["seed"], "condition": cond, **row,
                     "checkpoint": os.path.basename(path)})
    env.close()
    print(f"{os.path.basename(path)}: " + "  ".join(f"{r['condition']}: steps {r['steps']:.1f} near {r['near']:.2f}"
                                                   for r in rows), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--csv", default="results/train/rware/purity.csv")
    a = p.parse_args()
    jobs = [(path, CONFIGS[algo], a.episodes) for algo in ALGOS
            for path in sorted(glob.glob(os.path.join(POOLS[algo], "*.pt")))]
    if not jobs:
        raise SystemExit("no checkpoints found under " + ", ".join(POOLS.values()))
    if a.workers > 1:
        with get_context("spawn").Pool(a.workers) as pool:
            parts = pool.map(play_checkpoint, jobs)
    else:
        parts = [play_checkpoint(j) for j in jobs]
    rows = [r for part in parts for r in part]
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
