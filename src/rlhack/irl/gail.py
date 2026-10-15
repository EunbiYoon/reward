"""Tabular, state-only adversarial imitation (GAIL, Ho & Ermon 2016; observation-only
variant in the spirit of BCO/GAIfO).

The discriminator is logistic in the state features, D(s) = sigmoid(theta . phi(s)),
so its logit is a linear reward that transfers to new layouts. The generator is the
soft-optimal policy for that reward on each training map (exact, no sampling).
"""
from __future__ import annotations

import numpy as np

from ..mdp import soft_value_iteration
from ..utils import Adam, sigmoid
from .common import demo_feature_expectations, policy_feature_expectations


def state_gail(envs, demos, gamma: float, horizon: int, iters: int = 150, lr: float = 0.05,
               l2: float = 1e-2, d_steps: int = 5):
    K = envs[0].phi.shape[1]
    theta = np.zeros(K)
    m_E = demo_feature_expectations(envs, demos, gamma)
    m_E = m_E / m_E.sum()
    opt = Adam(lr)
    for _ in range(iters):
        m_G = policy_feature_expectations(
            envs, demos, lambda e: soft_value_iteration(e.P, e.reward(theta), gamma, tol=1e-4)[2], gamma, horizon)
        m_G = m_G / m_G.sum()
        for _ in range(d_steps):
            D = sigmoid(theta)  # one-hot features -> per-feature discriminator output
            grad = m_E * (1 - D) - m_G * D - l2 * theta
            theta = opt.step(theta, -grad)
    return theta  # reward = discriminator logit
