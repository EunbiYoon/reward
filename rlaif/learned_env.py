"""Training reward "learned": the environment pays the learner what the reward model says.

LearnedReward(env, rm_path) wraps any environment of environment/ and replaces the training
reward info["proxy"] with r_theta(o_t, a_t) from rlaif/reward_model.py; everything else
(observations, partners, `play` for the Shapley evaluation, the true score info["true"]) is the
wrapped environment's. The learner is still scored on team soups, never on the learned reward.
"""
from __future__ import annotations

import torch

from .reward_model import load_rm


class LearnedReward:
    def __init__(self, env, rm_path: str):
        self.env = env
        self.rm = load_rm(rm_path)
        self.rm_path = rm_path
        if self.rm.n_actions != env.n_actions or self.rm.mean.shape[0] != env.obs_dim:
            raise ValueError(f"{rm_path} was fitted on obs_dim={self.rm.mean.shape[0]}, "
                             f"n_actions={self.rm.n_actions}; this env has {env.obs_dim}, {env.n_actions}")
        self._obs = None

    def __getattr__(self, name):            # everything not overridden is the wrapped env's
        return getattr(self.env, name)

    def reset(self, seed: int):
        self._obs = self.env.reset(seed)
        return self._obs

    @torch.no_grad()
    def step(self, action):
        o = torch.as_tensor(self._obs, dtype=torch.float32)[None]
        r = float(self.rm(o, torch.as_tensor([int(action)]))[0])
        obs, info, done = self.env.step(action)
        info["proxy"] = r
        self._obs = obs
        return obs, info, done
