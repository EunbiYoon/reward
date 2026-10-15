"""Maximum Entropy IRL (Ziebart et al., AAAI 2008) with linear state features.

Gradient of the demo log-likelihood: mu_demo - E_{pi_w}[mu], where pi_w is the
soft-optimal policy for r = phi @ w (discounted soft value iteration).
"""
from __future__ import annotations

import numpy as np

from ..mdp import soft_value_iteration
from ..utils import Adam
from .common import demo_feature_expectations, policy_feature_expectations


def maxent_irl(envs, demos, gamma: float, horizon: int, iters: int = 200, lr: float = 0.05,
               l2: float = 1e-2, return_history: bool = False):
    K = envs[0].phi.shape[1]
    w = np.zeros(K)
    mu_D = demo_feature_expectations(envs, demos, gamma)
    opt = Adam(lr)
    hist = []
    for _ in range(iters):
        mu_pi = policy_feature_expectations(
            envs, demos, lambda e: soft_value_iteration(e.P, e.reward(w), gamma, tol=1e-4)[2], gamma, horizon)
        grad = mu_D - mu_pi - l2 * w  # ascent direction
        w = opt.step(w, -grad)
        hist.append(np.linalg.norm(mu_D - mu_pi))
    return (w, hist) if return_history else w
