"""Advantage actor-critic (A2C, Mnih et al. 2016; synchronous, single env) for the Overcooked learner.

Same networks and rollout as PPO (deep_policy actor + ValueNet critic, GAE, time-limit bootstrap),
but every rollout gives ONE plain policy-gradient step: no probability ratio, no clipping, no epochs,
so each sample is used once. Comparing it with PPO shows what clipping + data reuse buy.
Advantages are normalized per rollout, as in PPO, so the two differ only in the update.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .deep_policy import RunningMeanStd, ValueNet, action_buffer, make_policy, to_env_action


@dataclass
class A2CConfig:
    env_steps: int = 1_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 128
    lr: float = 7e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 64


def train_a2c(env, cfg: A2CConfig, seed: int, log_every: int = 0):
    """env: environment.Kitchen (or any env with the same interface). Returns (policy, obs_rms, log)."""
    torch.manual_seed(seed)
    policy = make_policy(env, cfg.hidden)
    value = ValueNet(env.obs_dim, cfg.hidden)
    params = list(policy.parameters()) + list(value.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    obs_rms = RunningMeanStd(env.obs_dim)

    ep_seed = seed * 100_000
    obs = env.reset(ep_seed)
    steps, updates, ep_proxy, ep_returns = 0, 0, 0.0, []
    while steps < cfg.env_steps:
        n = min(cfg.rollout_steps, cfg.env_steps - steps)
        O = np.zeros((n, env.obs_dim), np.float32)
        A = action_buffer(n, env)
        V, R, END = (np.zeros(n, np.float32) for _ in range(3))
        for t in range(n):
            obs_rms.update(obs[None])
            o = obs_rms.normalize(obs)
            with torch.no_grad():
                ot = torch.as_tensor(o[None])
                a, _ = policy.sample(ot)
                V[t] = value(ot).item()
            O[t] = o
            A[t] = to_env_action(a)
            obs, info, done = env.step(A[t])
            R[t] = info["proxy"]
            ep_proxy += info["proxy"]
            steps += 1
            if done:   # time-limit truncation: bootstrap through the cut
                with torch.no_grad():
                    R[t] += cfg.gamma * value(torch.as_tensor(obs_rms.normalize(obs)[None])).item()
                END[t] = 1.0
                ep_returns.append(ep_proxy)
                ep_proxy = 0.0
                ep_seed += 1
                obs = env.reset(ep_seed)
        with torch.no_grad():
            last_v = value(torch.as_tensor(obs_rms.normalize(obs)[None])).item()

        adv = np.zeros(n, np.float32)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            nonterm = 1.0 - END[t]
            next_v = last_v if t == n - 1 else V[t + 1]
            delta = R[t] + cfg.gamma * next_v * nonterm - V[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + V

        Ot, At = torch.as_tensor(O), torch.as_tensor(A)
        ADVt, RETt = torch.as_tensor(adv), torch.as_tensor(ret)
        if n > 1:
            ADVt = (ADVt - ADVt.mean()) / (ADVt.std() + 1e-8)
        pg = -(policy.log_prob(Ot, At) * ADVt).mean()
        vl = ((value(Ot) - RETt) ** 2).mean()
        ent = policy.entropy(Ot).mean()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        updates += 1
        if log_every and updates % log_every == 0 and ep_returns:
            print(f"    a2c seed={seed} steps={steps} proxy_ep_return={np.mean(ep_returns[-5:]):.2f}",
                  flush=True)

    log = {"env_steps_used": steps, "updates": updates, "episodes": len(ep_returns),
           "train_proxy_return_last5": float(np.mean(ep_returns[-5:])) if ep_returns else float("nan")}
    return policy, obs_rms, log
