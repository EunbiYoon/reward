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

from .resume import Resumer, base_env
from .deep_policy import RunningMeanStd, make_partner_head, make_policy, partner_label, to_env_action


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
    aux_coef: float = 0.0     # > 0: auxiliary partner-action prediction on the policy trunk (H12)
    resume_path: str = ""     # mid-training checkpoint (train/resume.py); "" = none
    save_every_s: float = 300.0


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
    head = make_partner_head(env, policy, cfg.hidden, cfg.aux_coef)
    params = list(policy.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)
    obs_rms = RunningMeanStd(env.obs_dim)

    steps, updates, skipped, group_means = 0, 0, 0, []
    res = Resumer(cfg.resume_path, cfg.save_every_s)
    st = res.load()
    if st:
        policy.load_state_dict(st["policy"]); opt.load_state_dict(st["opt"])
        if head is not None:
            head.load_state_dict(st["head"])
        obs_rms = RunningMeanStd.from_state(st["obs_rms"])
        steps, updates, skipped, group_means = st["steps"], st["updates"], st["skipped"], st["group_means"]
        base_env(env).total_steps = st["env_total_steps"]
        torch.set_rng_state(st["torch_rng"])

    def state():
        return {"policy": policy.state_dict(), "opt": opt.state_dict(),
                "head": head.state_dict() if head is not None else None, "obs_rms": obs_rms.state(),
                "steps": steps, "updates": updates, "skipped": skipped, "group_means": group_means,
                "env_total_steps": base_env(env).total_steps, "torch_rng": torch.get_rng_state()}

    while steps < cfg.env_steps:
        reset_seed = seed * 100_000 + updates
        O, A, LP, P1, rewards = [], [], [], [], []
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
                if head is not None:
                    P1.append(partner_label(info))
            steps += len(r_ep)
            O += o_ep; A += a_ep; LP += lp_ep
            rewards.append(np.asarray(r_ep, np.float32))
        updates += 1
        group_means.append(float(np.mean([r.sum() for r in rewards])))

        advs = _advantages(rewards, cfg.gamma, cfg.advantage)
        if advs is None:
            skipped += 1
            res.maybe_save(state)
            continue
        adv = np.concatenate(advs)
        adv = adv / (adv.std() + 1e-8)

        Ot = torch.as_tensor(np.asarray(O, np.float32))
        At = torch.as_tensor(np.asarray(A, np.int64 if policy.discrete else np.float32))
        LPt = torch.as_tensor(np.asarray(LP, np.float32))
        ADVt = torch.as_tensor(adv)
        P1t = torch.as_tensor(np.asarray(P1, np.int64)) if head is not None else None
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
                if head is not None:
                    loss = loss + cfg.aux_coef * head.loss(policy, Ot[idx], P1t[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
        res.maybe_save(state)
        if log_every and updates % log_every == 0:
            print(f"    grpo seed={seed} steps={steps} proxy_ep_return={group_means[-1]:.2f}", flush=True)

    log = {"env_steps_used": steps, "updates": updates, "episodes": updates * cfg.group_size,
           "grpo_skipped_groups": skipped,
           "train_proxy_return_last5": float(np.mean(group_means[-5:])) if group_means else float("nan")}
    if head is not None:
        log["_partner_head"] = head       # popped by eval/run.py (held-out prediction accuracy)
    return policy, obs_rms, log
