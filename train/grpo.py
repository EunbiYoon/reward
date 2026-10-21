"""Critic-free GRPO for the Overcooked learner (Shao et al. 2024), same policy network as PPO.

One update:
  1. G episodes from the SAME reset seed (same layout, same start), sampled with pi_old.
  2. advantages, no critic:
       process (default): normalize every step reward over the whole group,
                          r~ = (r - mean)/std, A_t = discounted sum of r~ from t on
                          (Sec. 4.1.3, "process supervision")
       outcome:           A_i = (R_i - mean_G R) / std_G R, the same for every step
     then scaled to unit std (not re-centred).
  3. `epochs` passes of the clipped surrogate over minibatches. No KL term.
The reset seed changes every update, so layouts vary across updates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .deep_policy import RunningMeanStd, make_policy, to_env_action


@dataclass
class GRPODeepConfig:
    env_steps: int = 1_000_000
    gamma: float = 0.99
    group_size: int = 8
    epochs: int = 10
    minibatch: int = 256
    lr: float = 3e-4
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    advantage: str = "process"
    hidden: int = 64


def _advantages(rewards: list[np.ndarray], gamma: float, mode: str) -> list[np.ndarray] | None:
    if mode == "outcome":
        R = np.array([np.sum(r * gamma ** np.arange(len(r))) for r in rewards])
        if R.std() < 1e-12:
            return None
        z = (R - R.mean()) / (R.std() + 1e-8)
        return [np.full(len(r), z[i], np.float32) for i, r in enumerate(rewards)]
    flat = np.concatenate(rewards)
    if flat.std() < 1e-12:
        return None
    mu, sd = flat.mean(), flat.std()
    out = []
    for r in rewards:
        z = (r - mu) / (sd + 1e-8)
        a = np.zeros_like(z)
        acc = 0.0
        for t in range(len(z) - 1, -1, -1):
            acc = z[t] + gamma * acc
            a[t] = acc
        out.append(a.astype(np.float32))
    return out


def train_grpo_deep(env, cfg: GRPODeepConfig, seed: int, log_every: int = 0):
    """env: environment.Kitchen (or any env with the same interface). Returns (policy, obs_rms, log)."""
    torch.manual_seed(seed)
    policy = make_policy(env, cfg.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    obs_rms = RunningMeanStd(env.obs_dim)

    steps, updates, skipped, group_means = 0, 0, 0, []
    while steps < cfg.env_steps:
        reset_seed = seed * 100_000 + updates
        O, A, LP, rewards = [], [], [], []
        for _ in range(cfg.group_size):
            obs = env.reset(reset_seed)
            o_ep, a_ep, lp_ep, r_ep = [], [], [], []
            done = False
            while not done:
                obs_rms.update(obs[None])
                o = obs_rms.normalize(obs)
                with torch.no_grad():
                    a, lp = policy.sample(torch.as_tensor(o[None]))
                a = to_env_action(a)
                obs, info, done = env.step(a)
                o_ep.append(o); a_ep.append(a); lp_ep.append(lp.item()); r_ep.append(info["proxy"])
            steps += len(r_ep)
            O += o_ep; A += a_ep; LP += lp_ep
            rewards.append(np.asarray(r_ep, np.float32))
        updates += 1
        group_means.append(float(np.mean([r.sum() for r in rewards])))

        advs = _advantages(rewards, cfg.gamma, cfg.advantage)
        if advs is None:
            skipped += 1
            continue
        adv = np.concatenate(advs)
        adv = adv / (adv.std() + 1e-8)

        Ot = torch.as_tensor(np.asarray(O, np.float32))
        At = torch.as_tensor(np.asarray(A, np.int64 if policy.discrete else np.float32))
        LPt = torch.as_tensor(np.asarray(LP, np.float32))
        ADVt = torch.as_tensor(adv)
        n = len(ADVt)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                lp = policy.log_prob(Ot[idx], At[idx])
                ratio = (lp - LPt[idx]).exp()
                a_mb = ADVt[idx]
                loss = -torch.min(ratio * a_mb,
                                  ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_mb).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
        if log_every and updates % log_every == 0:
            print(f"    grpo seed={seed} steps={steps} proxy_ep_return={group_means[-1]:.2f}", flush=True)

    log = {"env_steps_used": steps, "updates": updates, "episodes": updates * cfg.group_size,
           "grpo_skipped_groups": skipped,
           "train_proxy_return_last5": float(np.mean(group_means[-5:])) if group_means else float("nan")}
    return policy, obs_rms, log
