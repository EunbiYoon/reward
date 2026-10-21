"""PPO (clip) with GAE for the Overcooked learner (Schulman et al. 2017).

Single environment, MLP policy (categorical; a Gaussian policy is also supported) + separate MLP critic, observation
normalization. Overcooked and RWARE episodes only end by the time limit, so
there an episode end is a truncation: the last reward is bootstrapped with
gamma * V(s_T) and the GAE chain is cut there. LBF episodes also end for real
when all food is eaten (info["terminal"]); those are not bootstrapped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .deep_policy import RunningMeanStd, ValueNet, action_buffer, make_policy, to_env_action


@dataclass
class PPOConfig:
    env_steps: int = 1_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 2000
    epochs: int = 10
    minibatch: int = 64
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64


def train_ppo(env, cfg: PPOConfig, seed: int, log_every: int = 0):
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
        LP, V, R, END = (np.zeros(n, np.float32) for _ in range(4))
        for t in range(n):
            obs_rms.update(obs[None])
            o = obs_rms.normalize(obs)
            with torch.no_grad():
                ot = torch.as_tensor(o[None])
                a, lp = policy.sample(ot)
                LP[t] = lp.item()
                V[t] = value(ot).item()
            O[t] = o
            A[t] = to_env_action(a)
            obs, info, done = env.step(A[t])
            R[t] = info["proxy"]
            ep_proxy += info["proxy"]
            steps += 1
            if done:
                if not info.get("terminal"):   # time-limit truncation: bootstrap through the cut
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
        LPt, ADVt, RETt = torch.as_tensor(LP), torch.as_tensor(adv), torch.as_tensor(ret)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                if len(idx) < 2:
                    continue
                a_mb = ADVt[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                lp = policy.log_prob(Ot[idx], At[idx])
                ratio = (lp - LPt[idx]).exp()
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_mb).mean()
                vl = ((value(Ot[idx]) - RETt[idx]) ** 2).mean()
                ent = policy.entropy(Ot[idx]).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
        updates += 1
        if log_every and updates % log_every == 0 and ep_returns:
            print(f"    ppo seed={seed} steps={steps} proxy_ep_return={np.mean(ep_returns[-5:]):.2f}",
                  flush=True)

    log = {"env_steps_used": steps, "updates": updates, "episodes": len(ep_returns),
           "train_proxy_return_last5": float(np.mean(ep_returns[-5:])) if ep_returns else float("nan")}
    return policy, obs_rms, log
