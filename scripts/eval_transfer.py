"""Evaluate checkpoints trained with N agents in a team of a different size (team-size transfer).

    venv/bin/python scripts/eval_transfer.py results/rware_full --agents 3
    venv/bin/python scripts/eval_transfer.py results/rware_full --agents 3 4 --algos ppo --partners greedy

The learner keeps its seat 0 and every other seat gets the same partner as in training (greedy or
stay). Shapley is exact for the new team size (eval/coalitions.py). Works for envs whose observation
size does not depend on the team size (warehouse, foraging); kitchen is 2 players only.
Writes <run>/transfer_n<k>.csv and a short table next to the training-size numbers from results.csv.
"""
import argparse
import json
import os
import sys
from dataclasses import fields
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from train.config import Config  # noqa: E402
from eval.paths import results_path  # noqa: E402

COLS = ["soups", "own_soups", "partner_soups", "pickups", "v_partner_alone", "v_learner_alone", "phi", "s",
        "own_share", "credit_gap"]


def load_cfg(run: str) -> Config:
    raw = json.load(open(os.path.join(run, "config.json")))
    names = {f.name for f in fields(Config)}
    return Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})


def job(args):
    run, row, n_agents = args
    import torch
    torch.set_num_threads(1)
    from eval.coalitions import evaluate_coalitions, policy_actor, torch_reseed
    from train.run import ckpt_name, make_env
    from train.deep_policy import load_checkpoint
    cfg = load_cfg(run)
    path = os.path.join(run, "models", ckpt_name(row["algo"], row["reward"], row["partner"], int(row["seed"]),
                                                  int(row["agents"])))
    policy, obs_rms, meta = load_checkpoint(path)
    env = make_env(cfg, row["reward"], row["partner"], n_agents)
    if env.obs_dim != meta["obs_dim"]:
        raise SystemExit(f"{cfg.env}: observation size changes with team size ({meta['obs_dim']} -> {env.obs_dim})")
    actor = policy_actor(policy, obs_rms, cfg.eval_deterministic)
    met = evaluate_coalitions(env, actor, row["partner"], cfg.eval_episodes, before_episode=torch_reseed)
    env.close() if hasattr(env, "close") else None
    out = {"algo": row["algo"], "reward": row["reward"], "partner": row["partner"], "seed": int(row["seed"]),
           "trained_agents": int(row["agents"]), "eval_agents": n_agents, **{c: met[c] for c in COLS}}
    print(f"  {row['algo']} {row['reward']:<10} {row['partner']:<6} s{int(row['seed'])}: "
          f"n={n_agents} team={met['soups']:.2f} own={met['own_soups']:.2f} s={met['s']:+.2f}", flush=True)
    return out


def reference(run: str, partner: str, n_agents: int) -> dict:
    from eval.coalitions import evaluate_coalitions
    from train.run import make_env
    cfg = load_cfg(run)
    env = make_env(cfg, "team", partner, n_agents)
    met = evaluate_coalitions(env, "greedy", partner, cfg.eval_episodes)
    return {"algo": "reference", "reward": "-", "partner": partner, "seed": -1, "trained_agents": 0,
            "eval_agents": n_agents, **{c: met[c] for c in COLS}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--agents", type=int, nargs="+", default=[3], help="team sizes to evaluate in")
    p.add_argument("--algos", nargs="+", default=["ppo"])
    p.add_argument("--partners", nargs="+", default=["greedy"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    res = pd.read_csv(results_path(a.run))
    res = res[res.algo.isin(a.algos) & res.partner.isin(a.partners)]
    for n in a.agents:
        rows = [r for _, r in res.iterrows()]
        print(f"{len(rows)} checkpoints from {a.run} (trained with {sorted(res.agents.unique())} agents) -> {n} agents")
        with get_context("spawn").Pool(min(a.workers, len(rows))) as pool:
            out = pool.map(job, [(a.run, r, n) for r in rows])
        out += [reference(a.run, pt, n) for pt in a.partners]
        df = pd.DataFrame(out)
        path = os.path.join(a.run, f"transfer_n{n}.csv")
        df.to_csv(path, index=False)
        g = df.groupby(["algo", "reward", "partner"])[["soups", "own_soups", "s", "credit_gap", "pickups"]].mean()
        base = res.groupby(["algo", "reward", "partner"])[["soups", "s", "credit_gap"]].mean()
        base.columns = [f"{c}@train" for c in base.columns]
        print(f"\n== evaluated with {n} agents (means over seeds) -> {path}")
        print(g.join(base, how="left").round(2).to_string())


if __name__ == "__main__":
    main()
