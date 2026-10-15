"""Ablations for the proposed method.

1. **Query-budget-matched comparison** -- every query strategy gets the same number of
   preference labels on the deployment map (rounds x queries_per_round) and the same
   variance penalty, so differences come only from *which* pairs are asked about:
   ``q_ours``, ``q_disagreement``, ``q_hot_states``, ``q_random``. ``q_ours_lam0`` asks
   our queries but deploys the posterior mean (isolates the penalty).
2. **lambda sweep** -- ``varpen_lam{λ}`` (contrast-parameterized posterior, the default)
   and ``rawpen_lam{λ}`` (std of the raw samples), no queries.
3. **Automatic lambda** -- ``varpen_auto`` picks λ by minimizing the α-VaR of expected
   value difference over the posterior (no access to the true reward).

    python -m rlhack.experiments.ablation --config configs/ablation.json --out results/ablation.csv
    python -m rlhack.experiments.ablation --config configs/ablation.json --seeds 0 1 --out results/abl_a.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from ..agents import train_actor_critic
from ..demos import generate_demos
from ..envs import TRUE_WEIGHTS, make_test_env, make_train_env
from ..irl import bayesian_irl
from ..method import QUERY_STRATEGIES, active_bayesian_irl, auto_lambda, penalized_reward
from ..metrics import evaluate_policy
from ..preferences import SyntheticHuman
from ..utils import scale_reward
from .run import load_config

ABL_DEFAULTS = {
    "lambdas": [0.0, 0.5, 1.0, 2.0, 4.0],
    "auto_lambdas": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "query_strategies": list(QUERY_STRATEGIES),
    "query_lambda": 1.0,
}


def run(cfg: dict, out_path: str, verbose: bool = True):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows, t0 = [], time.time()
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        train_envs = [make_train_env(rng, cfg["grid_size"], cfg["lava_prob"], slip=cfg["slip"], gamma=cfg["gamma"])
                      for _ in range(cfg["n_train_maps"])]
        demo_stream = generate_demos(train_envs, max(cfg["budgets"]), rng, TRUE_WEIGHTS, cfg["expert_beta"],
                                     cfg["demo_horizon"])
        human = SyntheticHuman(TRUE_WEIGHTS, rng=rng, **cfg["human"])
        test_envs = {k: make_test_env(k, cfg["grid_size"], cfg["slip"], cfg["gamma"]) for k in cfg["test_envs"]}
        for budget in cfg["budgets"]:
            demos = demo_stream[:budget]
            post0 = bayesian_irl(train_envs, demos, rng, **cfg["birl"])
            for kind, env in test_envs.items():
                jobs = []  # (method, reward, posterior samples, extras)
                for lam in cfg["lambdas"]:
                    jobs.append((f"varpen_lam{lam:g}", penalized_reward(env, post0, lam), post0.samples,
                                 {"lambda": lam}))
                    if lam > 0:
                        jobs.append((f"rawpen_lam{lam:g}", penalized_reward(env, post0, lam, param="raw"),
                                     post0.samples, {"lambda": lam}))
                lam_star, _ = auto_lambda(env, post0, cfg["auto_lambdas"], cfg["var_alpha"], rng=rng)
                jobs.append(("varpen_auto", penalized_reward(env, post0, lam_star), post0.samples,
                             {"lambda": lam_star}))
                for strat in cfg["query_strategies"]:
                    post, prefs = active_bayesian_irl(train_envs, demos, env, human, rng, posterior=post0,
                                                      birl_kw=cfg["birl"], query_kw={"strategy": strat},
                                                      **cfg["active"])
                    jobs.append((f"q_{strat}", penalized_reward(env, post, cfg["query_lambda"]), post.samples,
                                 {"lambda": cfg["query_lambda"], "n_queries": len(prefs)}))
                    if strat == "ours":
                        jobs.append(("q_ours_lam0", post.reward_mean(env), post.samples,
                                     {"lambda": 0.0, "n_queries": len(prefs)}))
                for method, r, samples, extra in jobs:
                    t = time.time()
                    proxy = scale_reward(env, r, cfg["reward_clip"])
                    pi, _ = train_actor_critic(env, proxy, rng, **cfg["ac"])
                    sub = samples[rng.choice(len(samples), min(100, len(samples)), replace=False)]
                    m = evaluate_policy(env, pi, TRUE_WEIGHTS, proxy_reward=proxy, posterior_samples=sub,
                                        alpha=cfg["var_alpha"])
                    rows.append({"seed": seed, "budget": budget, "env": kind, "method": method, **m,
                                 "lambda": extra.get("lambda", np.nan), "n_queries": extra.get("n_queries", 0),
                                 "deploy_time": time.time() - t})
                    if verbose:
                        print(f"[{time.time() - t0:7.1f}s] seed={seed} n={budget:<3} {kind:<14} {method:<16} "
                              f"gap={m['norm_gap']:.3f} hacked={m['hacked']} lam={extra.get('lambda')}", flush=True)
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/ablation.csv")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override seeds (for parallel shards)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(None)
    cfg.update(ABL_DEFAULTS)
    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    if args.seeds:
        cfg["seeds"] = args.seeds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    run(cfg, args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
