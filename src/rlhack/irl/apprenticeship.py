"""Apprenticeship learning via IRL, projection method (Abbeel & Ng, ICML 2004)."""
from __future__ import annotations

import numpy as np

from ..mdp import greedy_policy, value_iteration
from .common import demo_feature_expectations, policy_feature_expectations


def apprenticeship_irl(envs, demos, gamma: float, horizon: int, rng, iters: int = 30, eps: float = 1e-3):
    mu_E = demo_feature_expectations(envs, demos, gamma)

    def mu_of(w):
        return policy_feature_expectations(
            envs, demos, lambda e: greedy_policy(value_iteration(e.P, e.reward(w), gamma, tol=1e-4)[0]),
            gamma, horizon)

    w = rng.normal(size=mu_E.shape)
    mu_bar = mu_of(w)
    for _ in range(iters):
        w = mu_E - mu_bar
        if np.linalg.norm(w) < eps:
            break
        mu = mu_of(w)
        diff = mu - mu_bar
        denom = diff @ diff
        if denom < 1e-12:
            break
        mu_bar = mu_bar + (diff @ (mu_E - mu_bar)) / denom * diff
    return w / (np.linalg.norm(w) + 1e-12)
