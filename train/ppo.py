"""PPO (clip) with GAE for the Overcooked learner (Schulman et al. 2017).

Single environment, MLP policy (categorical; a Gaussian policy is also supported) + separate MLP critic, observation
normalization. Overcooked and RWARE episodes only end by the time limit, so
there an episode end is a truncation: the last reward is bootstrapped with
gamma * V(s_T) and the GAE chain is cut there. LBF episodes also end for real
when all food is eaten (info["terminal"]); those are not bootstrapped.

Who learns: only seat 0 (the learner). The partner in seat 1 is moved by the environment inside
env.step() with a fixed scripted rule, so to PPO it is just part of the world. The reward PPO
maximizes is info["proxy"] (team / individual / pickup / cf, see environment/kitchen.py); the run
is scored afterwards on team soups and Shapley credit (eval/coalitions.py), never on the proxy.

One update = collect `rollout_steps` steps -> GAE advantages -> `epochs` passes of clipped-PPO
minibatch updates. The paper uses 2M steps, 4000-step rollouts (10 episodes of 400), 8 epochs,
minibatch 500, lr 3e-4, gamma 0.99, lambda 0.98, entropy 0.01, hidden 128 (eval/config.py).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .resume import Resumer, base_env
from .deep_policy import (RunningMeanStd, ValueNet, action_buffer, make_partner_head, make_policy,
                          partner_label, to_env_action)


@dataclass
class PPOConfig:
    env_steps: int = 1_000_000   # total environment steps of training
    gamma: float = 0.99          # discount
    gae_lambda: float = 0.95     # GAE lambda (bias/variance of the advantage)
    rollout_steps: int = 2000    # steps collected before each update
    epochs: int = 10             # passes over one rollout per update
    minibatch: int = 64          # steps per gradient step
    lr: float = 3e-4
    clip_eps: float = 0.2        # PPO clip: new/old probability ratio kept in [1-eps, 1+eps]
    vf_coef: float = 0.5         # weight of the value loss
    ent_coef: float = 0.0        # weight of the entropy bonus (keeps exploring)
    max_grad_norm: float = 0.5   # gradient clipping
    hidden: int = 64             # MLP width
    aux_coef: float = 0.0     # > 0: auxiliary partner-action prediction on the policy trunk (H12)
    resume_path: str = ""     # mid-training checkpoint (train/resume.py); "" = none
    save_every_s: float = 300.0
    curve_path: str = ""      # CSV learning curve, one row per update ("" = none); see scripts/plot_curves.py


# per-episode totals written to the learning curve (keys of the info dict the env returns each step)
CURVE_KEYS = ("soups", "own_soups", "partner_soups", "pickups0", "hold_dish_steps0", "pot_onions0", "pot_onions1")


def write_curve(path: str, row: dict):
    """Append one row to the learning-curve CSV (header on first write)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def train_ppo(env, cfg: PPOConfig, seed: int, log_every: int = 0):
    """env: environment.Kitchen (or any env with the same interface). Returns (policy, obs_rms, log)."""
    # ---------------------------------------------------------------- networks and optimizer
    torch.manual_seed(seed)
    policy = make_policy(env, cfg.hidden)          # obs -> probabilities of the 6 actions
    value = ValueNet(env.obs_dim, cfg.hidden)      # obs -> expected future proxy reward
    head = make_partner_head(env, policy, cfg.hidden, cfg.aux_coef)   # optional: predicts the partner's action
    params = list(policy.parameters()) + list(value.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)      # one optimizer for policy, value (and head)
    obs_rms = RunningMeanStd(env.obs_dim)          # running mean/std used to normalize observations

    ep_seed = seed * 100_000                       # each episode starts from its own seed
    steps, updates, ep_proxy, ep_returns = 0, 0, 0.0, []
    ep_tot = {k: 0.0 for k in CURVE_KEYS}          # this episode's totals, for the learning curve
    ep_len = 0
    done_eps: list = []                            # finished episodes of the current rollout

    # ---------------------------------------------------------------- resume after an interruption
    res = Resumer(cfg.resume_path, cfg.save_every_s)
    st = res.load()
    if st:
        policy.load_state_dict(st["policy"]); value.load_state_dict(st["value"]); opt.load_state_dict(st["opt"])
        if head is not None:
            head.load_state_dict(st["head"])
        obs_rms = RunningMeanStd.from_state(st["obs_rms"])
        steps, updates, ep_seed, ep_returns = st["steps"], st["updates"], st["ep_seed"] + 1, st["ep_returns"]
        base_env(env).total_steps = st["env_total_steps"]     # keeps the shaping anneal where it was
        torch.set_rng_state(st["torch_rng"])
    obs = env.reset(ep_seed)

    def state():
        return {"policy": policy.state_dict(), "value": value.state_dict(), "opt": opt.state_dict(),
                "head": head.state_dict() if head is not None else None, "obs_rms": obs_rms.state(),
                "steps": steps, "updates": updates, "ep_seed": ep_seed, "ep_returns": ep_returns,
                "env_total_steps": base_env(env).total_steps, "torch_rng": torch.get_rng_state()}

    while steps < cfg.env_steps:
        # ============================================================ 1. collect a rollout
        n = min(cfg.rollout_steps, cfg.env_steps - steps)
        O = np.zeros((n, env.obs_dim), np.float32)   # normalized observations
        A = action_buffer(n, env)                    # learner actions
        LP, V, R, END = (np.zeros(n, np.float32) for _ in range(4))   # log-prob, value, reward, episode end
        P1 = np.zeros(n, np.int64)                   # partner actions (labels for the optional head)
        for t in range(n):
            # (1) the learner looks at the state; the observation includes the partner's position and item
            obs_rms.update(obs[None])
            o = obs_rms.normalize(obs)
            # (2) the learner picks one of the 6 actions (4 moves, stay, interact) by sampling its policy
            with torch.no_grad():
                ot = torch.as_tensor(o[None])
                a, lp = policy.sample(ot)
                LP[t] = lp.item()                    # old log-prob, for the PPO ratio later
                V[t] = value(ot).item()              # value estimate, for GAE later
            O[t] = o
            A[t] = to_env_action(a)
            # (3) one game step. Inside env.step the partner's scripted rule picks its action, both
            # actions are applied together, events are logged (pickups, deliveries, interceptions)
            # and the learner's reward is computed: info["proxy"].
            obs, info, done = env.step(A[t])
            R[t] = info["proxy"]                     # only the learner is rewarded; the partner never learns
            if head is not None:
                P1[t] = partner_label(info)
            ep_proxy += info["proxy"]
            for k in CURVE_KEYS:
                ep_tot[k] += float(info.get(k, 0.0))
            ep_len += 1
            steps += 1
            # (4) end of an episode (400 steps): bootstrap the cut, start the next episode
            if done:
                if not info.get("terminal"):   # time-limit truncation: bootstrap through the cut
                    with torch.no_grad():
                        R[t] += cfg.gamma * value(torch.as_tensor(obs_rms.normalize(obs)[None])).item()
                END[t] = 1.0
                ep_returns.append(ep_proxy)
                done_eps.append({"proxy_return": ep_proxy, **ep_tot, "length": ep_len})
                ep_proxy, ep_len = 0.0, 0
                ep_tot = {k: 0.0 for k in CURVE_KEYS}
                ep_seed += 1
                obs = env.reset(ep_seed)
        with torch.no_grad():
            last_v = value(torch.as_tensor(obs_rms.normalize(obs)[None])).item()   # value after the last step

        # ============================================================ 2. advantages (GAE), backwards in time
        # delta_t = r_t + gamma V(s_t+1) - V(s_t): how much better this step went than expected.
        # The advantage sums these deltas with weight (gamma lambda)^k, cut at episode ends.
        adv = np.zeros(n, np.float32)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            nonterm = 1.0 - END[t]
            next_v = last_v if t == n - 1 else V[t + 1]
            delta = R[t] + cfg.gamma * next_v * nonterm - V[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + V                                # value target

        # ============================================================ 3. PPO update
        Ot, At = torch.as_tensor(O), torch.as_tensor(A)
        LPt, ADVt, RETt = torch.as_tensor(LP), torch.as_tensor(adv), torch.as_tensor(ret)
        P1t = torch.as_tensor(P1)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)                 # shuffle the rollout every epoch
            for i in range(0, n, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                if len(idx) < 2:
                    continue
                a_mb = ADVt[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)        # normalized advantages
                lp = policy.log_prob(Ot[idx], At[idx])
                ratio = (lp - LPt[idx]).exp()                              # new prob / old prob
                # clipped objective: push up actions with positive advantage, but not past the clip
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_mb).mean()
                vl = ((value(Ot[idx]) - RETt[idx]) ** 2).mean()            # value regression
                ent = policy.entropy(Ot[idx]).mean()                       # exploration bonus
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                if head is not None:
                    loss = loss + cfg.aux_coef * head.loss(policy, Ot[idx], P1t[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
        updates += 1
        if cfg.curve_path and done_eps:
            # mean over the episodes that ended in this rollout (10 with 4000-step rollouts of 400)
            m = {k: float(np.mean([e[k] for e in done_eps])) for k in done_eps[0]}
            write_curve(cfg.curve_path, {
                "update": updates, "env_steps": steps, "episodes": len(done_eps),
                "proxy_return": round(m["proxy_return"], 3),        # what PPO maximizes
                "team_soups": round(m["soups"], 3),                 # the true objective
                "own_soups": round(m["own_soups"], 3), "partner_soups": round(m["partner_soups"], 3),
                "pickups": round(m["pickups0"], 2),                 # learner's onion + dish pickups
                "hold_dish_frac": round(m["hold_dish_steps0"] / m["length"], 3),   # time holding a dish/soup
                "own_pot_onions": round(m["pot_onions0"], 2),        # onions the learner put into pots
                "partner_pot_onions": round(m["pot_onions1"], 2),    # onions the partner put into pots
                "shaping_coef": round(float(getattr(base_env(env), "anneal_coef", lambda: float("nan"))()), 3)})
        done_eps = []
        res.maybe_save(state)                        # checkpoint every save_every_s seconds
        if log_every and updates % log_every == 0 and ep_returns:
            print(f"    ppo seed={seed} steps={steps} proxy_ep_return={np.mean(ep_returns[-5:]):.2f}",
                  flush=True)

    log = {"env_steps_used": steps, "updates": updates, "episodes": len(ep_returns),
           "train_proxy_return_last5": float(np.mean(ep_returns[-5:])) if ep_returns else float("nan")}
    if head is not None:
        log["_partner_head"] = head       # popped by eval/run.py (held-out prediction accuracy)
    return policy, obs_rms, log
