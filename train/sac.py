"""Soft actor-critic for discrete actions (SAC-Discrete, Christodoulou 2019) for the Overcooked learner.

Off-policy actor-critic with maximum-entropy RL (Haarnoja et al. 2018):
  - actor: the same CategoricalPolicy as PPO; critics: twin Q-networks mlp(obs) -> Q(s, .)
    with Polyak-averaged targets (`tau`)
  - with discrete actions every expectation over pi is an exact sum over the actions, no sampling:
      target  y = r + gamma * sum_a' pi(a'|s') [min Q_targ(s', a') - alpha log pi(a'|s')]
      actor   sum_a pi(a|s) [alpha log pi(a|s) - min Q(s, a)]
  - temperature alpha is learned so the policy entropy tracks
    `target_entropy_scale` * log(n_actions)
  - the first `learning_starts` steps act uniformly at random to fill the replay buffer
Like DQN: no transition is terminal (time-limit only), rewards are stored as paid.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .deep_policy import CategoricalPolicy, ReplayBuffer, RunningMeanStd, mlp, to_env_action


@dataclass
class SACConfig:
    env_steps: int = 1_000_000
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch: int = 128
    lr: float = 3e-4
    learning_starts: int = 10_000
    train_every: int = 4
    tau: float = 0.005
    target_entropy_scale: float = 0.89
    max_grad_norm: float = 10.0
    hidden: int = 64


def train_sac(env, cfg: SACConfig, seed: int, log_every: int = 0):
    """env: environment.Kitchen (discrete actions). Returns (policy, obs_rms, log)."""
    assert getattr(env, "discrete", False), "this SAC is the discrete-action variant"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = CategoricalPolicy(env.obs_dim, env.n_actions, cfg.hidden)
    q1, q2 = (mlp(env.obs_dim, env.n_actions, cfg.hidden) for _ in range(2))
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
    q_params = list(q1.parameters()) + list(q2.parameters())
    q_opt = torch.optim.Adam(q_params, lr=cfg.lr)
    pi_opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr)
    target_entropy = cfg.target_entropy_scale * float(np.log(env.n_actions))
    obs_rms = RunningMeanStd(env.obs_dim)
    buf = ReplayBuffer(cfg.buffer_size, env.obs_dim)

    ep_seed = seed * 100_000
    obs = env.reset(ep_seed)
    updates, ep_proxy, ep_returns = 0, 0.0, []
    for step in range(cfg.env_steps):
        obs_rms.update(obs[None])
        if step < cfg.learning_starts:
            a = int(rng.integers(env.n_actions))
        else:
            with torch.no_grad():
                a = to_env_action(policy.sample(torch.as_tensor(obs_rms.normalize(obs)[None]))[0])
        next_obs, info, done = env.step(a)
        buf.add(obs, a, info["proxy"], next_obs)
        ep_proxy += info["proxy"]
        if done:
            ep_returns.append(ep_proxy)
            ep_proxy = 0.0
            ep_seed += 1
            obs = env.reset(ep_seed)
        else:
            obs = next_obs

        if step < cfg.learning_starts or step % cfg.train_every:
            continue
        o, a_b, r, o2 = buf.sample(cfg.batch, obs_rms, rng)
        alpha = log_alpha.exp().detach()

        with torch.no_grad():
            lp2 = F.log_softmax(policy.logits(o2), -1)
            v2 = (lp2.exp() * (torch.min(q1_t(o2), q2_t(o2)) - alpha * lp2)).sum(-1)
            y = r + cfg.gamma * v2
        idx = a_b[:, None]
        q_loss = F.mse_loss(q1(o).gather(1, idx).squeeze(1), y) + F.mse_loss(q2(o).gather(1, idx).squeeze(1), y)
        q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(q_params, cfg.max_grad_norm)
        q_opt.step()

        lp = F.log_softmax(policy.logits(o), -1)
        p = lp.exp()
        with torch.no_grad():
            q_min = torch.min(q1(o), q2(o))
        pi_loss = (p * (alpha * lp - q_min)).sum(-1).mean()
        pi_opt.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        pi_opt.step()

        entropy = -(p * lp).sum(-1).mean().detach()
        alpha_loss = log_alpha * (entropy - target_entropy)   # entropy below target -> alpha grows
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

        with torch.no_grad():
            for net, targ in ((q1, q1_t), (q2, q2_t)):
                for w, wt in zip(net.parameters(), targ.parameters()):
                    wt.mul_(1 - cfg.tau).add_(cfg.tau * w)
        updates += 1
        if log_every and updates % log_every == 0 and ep_returns:
            print(f"    sac seed={seed} steps={step + 1} alpha={alpha.item():.3f} entropy={entropy.item():.2f} "
                  f"proxy_ep_return={np.mean(ep_returns[-5:]):.2f}", flush=True)

    log = {"env_steps_used": cfg.env_steps, "updates": updates, "episodes": len(ep_returns),
           "sac_final_alpha": float(log_alpha.exp().item()),
           "train_proxy_return_last5": float(np.mean(ep_returns[-5:])) if ep_returns else float("nan")}
    return policy, obs_rms, log
