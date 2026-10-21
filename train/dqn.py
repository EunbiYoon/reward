"""DQN for the Overcooked learner (Mnih et al. 2015) with the Double-DQN target (van Hasselt et al. 2016).

Value-based and off-policy: no policy network, the learner acts on Q.
  - replay buffer (deep_policy.ReplayBuffer), raw observations normalized when sampled
  - epsilon-greedy, epsilon falls linearly from 1 to `eps_end` over `explore_frac` of the budget
  - target network copied every `target_every` env steps, Huber loss
Overcooked episodes only end by the time limit, so no transition is terminal: every target
bootstraps with gamma * Q_target(s', argmax_a Q(s', a)).
Rewards are stored as paid, so the fading shaping in old transitions is slightly out of date.
The Q-network is a QPolicy, so checkpoints and evaluation work as for PPO (eval uses eps = `eval_eps`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .deep_policy import QPolicy, ReplayBuffer, RunningMeanStd, to_env_action


@dataclass
class DQNConfig:
    env_steps: int = 1_000_000
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch: int = 128
    lr: float = 1e-4
    learning_starts: int = 10_000
    train_every: int = 4
    target_every: int = 2_000
    explore_frac: float = 0.2
    eps_end: float = 0.05
    eval_eps: float = 0.05
    max_grad_norm: float = 10.0
    hidden: int = 64


def train_dqn(env, cfg: DQNConfig, seed: int, log_every: int = 0):
    """env: environment.Kitchen (discrete actions). Returns (policy, obs_rms, log)."""
    assert getattr(env, "discrete", False), "DQN needs a discrete action space"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = QPolicy(env.obs_dim, env.n_actions, cfg.hidden)
    q_targ = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    obs_rms = RunningMeanStd(env.obs_dim)
    buf = ReplayBuffer(cfg.buffer_size, env.obs_dim)
    explore_steps = max(1, int(cfg.explore_frac * cfg.env_steps))

    ep_seed = seed * 100_000
    obs = env.reset(ep_seed)
    updates, ep_proxy, ep_returns = 0, 0.0, []
    for step in range(cfg.env_steps):
        obs_rms.update(obs[None])
        q.eps = max(cfg.eps_end, 1.0 - (1.0 - cfg.eps_end) * step / explore_steps)
        with torch.no_grad():
            a, _ = q.sample(torch.as_tensor(obs_rms.normalize(obs)[None]))
        a = to_env_action(a)
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

        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            o, a_b, r, o2 = buf.sample(cfg.batch, obs_rms, rng)
            with torch.no_grad():
                a2 = q.logits(o2).argmax(-1, keepdim=True)
                y = r + cfg.gamma * q_targ.logits(o2).gather(1, a2).squeeze(1)
            qa = q.logits(o).gather(1, a_b[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(qa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), cfg.max_grad_norm)
            opt.step()
            updates += 1
            if log_every and updates % log_every == 0 and ep_returns:
                print(f"    dqn seed={seed} steps={step + 1} eps={q.eps:.2f} "
                      f"proxy_ep_return={np.mean(ep_returns[-5:]):.2f}", flush=True)
        if step % cfg.target_every == 0:
            q_targ.load_state_dict(q.state_dict())

    q.eps = cfg.eval_eps
    log = {"env_steps_used": cfg.env_steps, "updates": updates, "episodes": len(ep_returns),
           "train_proxy_return_last5": float(np.mean(ep_returns[-5:])) if ep_returns else float("nan")}
    return q, obs_rms, log
