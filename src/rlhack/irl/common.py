"""Helpers shared by the IRL baselines."""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..demos import feature_counts
from ..mdp import transition_under_policy


def expected_feature_counts(env, pi: np.ndarray, gamma: float, horizon: int) -> np.ndarray:
    """E_pi[sum_{t<H} gamma^t phi(s_t)] from env.start (exact forward pass)."""
    P_pi = transition_under_policy(env.P, pi)
    d = env.p0.copy()
    mu = np.zeros(env.phi.shape[1])
    for t in range(horizon):
        mu += gamma**t * (d @ env.phi)
        d = d @ P_pi
    return mu


def demo_feature_expectations(envs, demos, gamma: float) -> np.ndarray:
    return np.mean([feature_counts(envs[d.env_id], d, gamma) for d in demos], axis=0)


def env_weights(demos) -> dict:
    """Fraction of demonstrations collected on each training map."""
    c = Counter(d.env_id for d in demos)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def policy_feature_expectations(envs, demos, policy_fn, gamma: float, horizon: int) -> np.ndarray:
    """Demo-weighted mixture of expected feature counts of policy_fn(env) on each map."""
    return sum(wt * expected_feature_counts(envs[k], policy_fn(envs[k]), gamma, horizon)
               for k, wt in env_weights(demos).items())
