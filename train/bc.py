"""Behaviour cloning of the scripted greedy robot (RWARE): a policy that already delivers, the start point
for persona training (python -m train.run ... --init-ckpt <this checkpoint>).

PPO from scratch never learns to deliver in RWARE (a delivery is a long chain: find a requested shelf,
lift it, drive it to a goal; random exploration almost never finishes it). The greedy robot does it ~15
times per episode, so we copy it:

  round 0      greedy drives seat 0; every (observation, greedy action) pair is kept
  rounds 1..k  DAgger: the CLONE drives seat 0 (so we see the states it really reaches, mistakes included)
               and greedy labels each state with the action it would take there
  each round   the policy is refit on all pairs so far (cross entropy); the observation normalizer is the
               mean / std of all observations collected

Labels are noise-free greedy actions (the 5% random actions of the scripted robot are off while labelling).
The checkpoint has the same format and network as train.run's (CategoricalPolicy, `hidden` units), so
eval/, judge/ and --init-ckpt read it unchanged.

    python -m train.bc --agents 3 --partners greedy --seed 0 --out results/rware3_bc
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from environment import make
from train.config import ENV_DEFAULTS, Config
from train.deep_policy import make_obs_rms, make_policy, save_checkpoint


def greedy_label(env) -> int:
    """The greedy robot's action for seat 0 right now, without its random-action noise."""
    noise, env.noise = env.noise, 0.0
    try:
        return env.scripted_index("greedy", 0)
    finally:
        env.noise = noise


def collect(env, policy, obs_rms, episodes: int, seed0: int, use_policy: bool):
    """(observations, greedy labels) of `episodes` episodes. Seat 0 is driven by greedy (use_policy=False)
    or by the clone, sampling its actions (use_policy=True)."""
    O, A = [], []
    for k in range(episodes):
        obs = env.reset(seed0 + k)
        done = False
        while not done:
            label = greedy_label(env)
            O.append(obs)
            A.append(label)
            if use_policy:
                with torch.no_grad():
                    act = int(policy.sample(torch.as_tensor(obs_rms.normalize(obs)[None]))[0].item())
            else:
                act = label
            obs, ev, done = env.step(act)
    return np.asarray(O, np.float32), np.asarray(A, np.int64)


def fit(policy, opt, obs_rms, O, A, epochs: int, batch: int, rng) -> float:
    X = torch.as_tensor(obs_rms.normalize(O))
    Y = torch.as_tensor(A)
    n = len(X)
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(policy.logits(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float((policy.logits(X).argmax(-1) == Y).float().mean())


def evaluate(env, policy, obs_rms, partner: str, episodes: int, seed0: int) -> dict:
    """Mean episode totals with the clone in seat 0 (sampled actions, as eval/ plays checkpoints)."""
    def seat0(obs):
        with torch.no_grad():
            return int(policy.sample(torch.as_tensor(obs_rms.normalize(obs)[None]))[0].item())
    tots = [env.play(seat0, partner, seed0 + k) for k in range(episodes)]
    return {k: float(np.mean([t[k] for t in tots])) for k in ("own_soups", "soups", "partner_soups", "pickups0")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--partners", default="greedy", choices=["greedy", "stay"],
                   help="the other robots while collecting (greedy: the clone learns to work among them)")
    p.add_argument("--layout", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=40, help="episodes collected per round")
    p.add_argument("--rounds", type=int, default=4, help="DAgger rounds after the greedy-driven round 0")
    p.add_argument("--epochs", type=int, default=10, help="passes over all data per round")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out", default="results/rware3_bc")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    rng = np.random.default_rng(a.seed)
    d = ENV_DEFAULTS["rware"]
    layout, horizon, hidden = a.layout or d["layout"], d["horizon"], Config().hidden
    env = make("rware", layout, "team", horizon, a.partners, n_agents=a.agents)
    policy = make_policy(env, hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
    obs_rms = make_obs_rms(env)

    os.makedirs(a.out, exist_ok=True)
    name = f"bc_greedy_{a.partners}_n{a.agents}_s{a.seed}"
    log = []
    O_all, A_all = [], []
    t0 = time.time()
    for r in range(a.rounds + 1):
        seed0 = 1_000_000 * (a.seed + 1) + 10_000 * r          # never the eval seeds (10_000 + k)
        O, A = collect(env, policy, obs_rms, a.episodes, seed0, use_policy=r > 0)
        O_all.append(O)
        A_all.append(A)
        if r == 0:
            obs_rms.update(O)                                  # normalizer from greedy-driven states
        acc = fit(policy, opt, obs_rms, np.concatenate(O_all), np.concatenate(A_all), a.epochs, a.batch, rng)
        m = evaluate(env, policy, obs_rms, a.partners, a.eval_episodes, 10_000)
        row = {"round": r, "pairs": int(sum(len(x) for x in A_all)), "train_acc": round(acc, 3),
               **{k: round(v, 2) for k, v in m.items()}, "seconds": round(time.time() - t0)}
        log.append(row)
        print(f"  round {r}: {row}", flush=True)

    ckpt = os.path.join(a.out, name + ".pt")
    save_checkpoint(ckpt, policy, obs_rms, dict(obs_dim=env.obs_dim, n_actions=env.n_actions, hidden=hidden,
                                                algo="bc", reward="bc", partner=a.partners, seed=a.seed,
                                                agents=a.agents, env="rware", layout=layout))
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump({"args": vars(a), "rounds": log}, f, indent=1)
    print(f"-> {ckpt}  (own deliveries per episode {log[-1]['own_soups']:.2f})")
    env.close()


if __name__ == "__main__":
    main()
