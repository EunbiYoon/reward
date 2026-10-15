"""Main sweep: methods x demo budgets x proxy test environments x seeds.

    python -m rlhack.experiments.run --config configs/quick.json
    python -m rlhack.experiments.run --config configs/full.json --out results/full.csv

Every reward-learning method (and the oracle) hands a reward vector to a
One-Step Actor-Critic agent trained from scratch on the deployment map; BC and
DAgger are deployed directly. All policies are scored with the TRUE reward.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from ..agents import train_actor_critic
from ..demos import generate_demos, generate_ranked_trajectories
from ..envs import TRUE_WEIGHTS, make_test_env, make_train_env
from ..imitation import behavior_cloning, dagger
from ..irl import apprenticeship_irl, bayesian_irl, brex, maxent_irl, state_gail, trex
from ..method import active_bayesian_irl, penalized_reward
from ..metrics import evaluate_policy
from ..preferences import SyntheticHuman, pairwise_preferences
from ..utils import normalize_reward

ALL_METHODS = ["oracle", "bc", "dagger", "apprenticeship", "maxent", "gail", "trex", "brex",
               "birl_mean", "birl_varpen", "birl_active"]

DEFAULTS = {
    "seeds": [0, 1, 2],
    "budgets": [1, 2, 5, 10, 20],
    "test_envs": ["lava_shortcut", "coin_trap", "combined"],
    "methods": ALL_METHODS,
    "grid_size": 8, "n_train_maps": 10, "lava_prob": 0.3, "slip": 0.1, "gamma": 0.95,
    "demo_horizon": 30, "expert_beta": 100.0,
    "irl_iters": 150,
    "birl": {"beta": 20.0, "beta_pref": 5.0, "prior_std": 1.0, "n_samples": 400, "burn_in": 400, "thin": 2},
    "brex": {"beta": 5.0, "n_samples": 800, "burn_in": 400},
    "ranked_per_demo": 2, "max_pairs_per_demo": 6,
    "lambda": 1.0,
    "active": {"rounds": 2, "queries_per_round": 5},
    "human": {"model": "partial_return", "beta": 5.0},
    "ac": {"episodes": 4000, "horizon": 40, "exploring_starts": 0.8, "alpha_theta": 1.0, "alpha_w": 0.5},
    "var_alpha": 0.95,
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def fit_reward_models(cfg, train_envs, demos, ranked, human, rng):
    """Everything that depends only on the training maps (shared across test envs)."""
    g, H, it = cfg["gamma"], cfg["demo_horizon"], cfg["irl_iters"]
    methods = set(cfg["methods"])
    out, timing = {}, {}

    def timed(name, fn):
        t = time.time()
        out[name] = fn()
        timing[name] = time.time() - t

    if "maxent" in methods:
        timed("maxent", lambda: maxent_irl(train_envs, demos, g, H, iters=it))
    if "apprenticeship" in methods:
        timed("apprenticeship", lambda: apprenticeship_irl(train_envs, demos, g, H, rng))
    if "gail" in methods:
        timed("gail", lambda: state_gail(train_envs, demos, g, H, iters=it))
    if methods & {"trex", "brex"}:
        prefs = pairwise_preferences(train_envs, ranked, human, rng,
                                     max_pairs=cfg["max_pairs_per_demo"] * len(demos))
        if "trex" in methods:
            timed("trex", lambda: trex(prefs))
        if "brex" in methods:
            timed("brex", lambda: brex(prefs, rng, **cfg["brex"]))
    if methods & {"birl_mean", "birl_varpen", "birl_active"}:
        timed("birl", lambda: bayesian_irl(train_envs, demos, rng, **cfg["birl"]))
    if "bc" in methods:
        timed("bc", lambda: behavior_cloning(train_envs, demos))
    if "dagger" in methods:
        timed("dagger", lambda: dagger(train_envs, demos, TRUE_WEIGHTS, rng, horizon=H))
    return out, timing


def run(cfg: dict, out_path: str, verbose: bool = True):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    fieldnames = None
    t0 = time.time()
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        train_envs = [make_train_env(rng, cfg["grid_size"], cfg["lava_prob"], slip=cfg["slip"], gamma=cfg["gamma"])
                      for _ in range(cfg["n_train_maps"])]
        max_b = max(cfg["budgets"])
        demo_stream = generate_demos(train_envs, max_b, rng, TRUE_WEIGHTS, cfg["expert_beta"], cfg["demo_horizon"])
        ranked_stream = generate_ranked_trajectories(train_envs, cfg["ranked_per_demo"] * max_b, rng, TRUE_WEIGHTS,
                                                     cfg["expert_beta"], cfg["demo_horizon"])
        human = SyntheticHuman(TRUE_WEIGHTS, rng=rng, **cfg["human"])
        test_envs = {k: make_test_env(k, cfg["grid_size"], cfg["slip"], cfg["gamma"]) for k in cfg["test_envs"]}

        for budget in cfg["budgets"]:
            demos = demo_stream[:budget]
            ranked = ranked_stream[:cfg["ranked_per_demo"] * budget]
            models, timing = fit_reward_models(cfg, train_envs, demos, ranked, human, rng)

            for kind, env in test_envs.items():
                for method in cfg["methods"]:
                    t = time.time()
                    proxy, policy, post_samples, extra = None, None, None, {}
                    if method in ("bc", "dagger"):
                        policy = models[method].probs(env)
                    else:
                        if method == "oracle":
                            r = env.reward(TRUE_WEIGHTS)
                        elif method in ("maxent", "apprenticeship", "gail", "trex"):
                            r = env.reward(models[method])
                        elif method == "brex":
                            r = models["brex"].reward_mean(env)
                            post_samples = models["brex"].samples
                        elif method == "birl_mean":
                            r = models["birl"].reward_mean(env)
                            post_samples = models["birl"].samples
                        elif method == "birl_varpen":
                            r = penalized_reward(env, models["birl"], cfg["lambda"])
                            post_samples = models["birl"].samples
                        elif method == "birl_active":
                            post, prefs = active_bayesian_irl(train_envs, demos, env, human, rng,
                                                              posterior=models["birl"], birl_kw=cfg["birl"],
                                                              **cfg["active"])
                            r = penalized_reward(env, post, cfg["lambda"])
                            post_samples = post.samples
                            extra["n_queries"] = len(prefs)
                        else:
                            raise ValueError(method)
                        proxy = normalize_reward(r)
                        policy, _ = train_actor_critic(env, proxy, rng, **cfg["ac"])
                    if post_samples is not None and len(post_samples) > 100:
                        post_samples = post_samples[rng.choice(len(post_samples), 100, replace=False)]
                    m = evaluate_policy(env, policy, TRUE_WEIGHTS,
                                        proxy_reward=None if method == "oracle" else proxy,
                                        posterior_samples=post_samples, alpha=cfg["var_alpha"])
                    key = "birl" if method.startswith("birl") else method
                    row = {"seed": seed, "budget": budget, "env": kind, "method": method, **m,
                           "n_queries": extra.get("n_queries", 0),
                           "fit_time": timing.get(key, 0.0), "deploy_time": time.time() - t}
                    rows.append(row)
                    if fieldnames is None:
                        fieldnames = list(row.keys())
                    if verbose:
                        print(f"[{time.time() - t0:7.1f}s] seed={seed} n={budget:<3} {kind:<14} {method:<15} "
                              f"gap={m['norm_gap']:.3f} tv={m['tv_to_opt']:.3f} lava={m['lava_occ']:.3f} "
                              f"coin={m['coin_occ']:.3f} hacked={m['hacked']}", flush=True)
            with open(out_path, "w", newline="") as f:  # checkpoint after every budget
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    run(cfg, args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
