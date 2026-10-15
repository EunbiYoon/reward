"""Behavior cloning with a layout-transferable, feature-based softmax policy.

pi(a|s) ∝ exp(theta . psi(s, a)) with psi = [phi(intended next cell), progress toward
goal, is_stay]. Because psi is defined relative to the map, the cloned policy can be
run on unseen test layouts (unlike a tabular policy).
"""
from __future__ import annotations

import numpy as np

from ..utils import Adam, softmax

N_ACTION_FEATURES = 6


def action_features(env) -> np.ndarray:
    S, A = env.n_states, env.n_actions
    psi = np.zeros((S, A, N_ACTION_FEATURES))
    for s in range(S):
        if s == env.sink or s in env.goal_states:
            continue
        d0 = env.manhattan_to_goal(s)
        for a in range(A):
            n = env.next_state[s, a]
            psi[s, a, :4] = env.phi[n]
            psi[s, a, 4] = d0 - env.manhattan_to_goal(n)
            psi[s, a, 5] = float(a == 0)
    return psi


class FeaturePolicy:
    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def probs(self, env) -> np.ndarray:
        return softmax(action_features(env) @ self.theta, axis=-1)


def fit_feature_policy(envs, data, l2: float = 1e-3, iters: int = 300, lr: float = 0.05) -> FeaturePolicy:
    """data: iterable of (env_id, state, action). Maximum-likelihood fit."""
    data = [(e, s, a) for e, s, a in data if s != envs[e].sink and s not in envs[e].goal_states]
    psi_cache = {}
    for e, _, _ in data:
        if e not in psi_cache:
            psi_cache[e] = action_features(envs[e])
    Psi = np.stack([psi_cache[e][s] for e, s, _ in data])  # (N, A, F)
    acts = np.array([a for _, _, a in data])
    theta = np.zeros(N_ACTION_FEATURES)
    opt = Adam(lr)
    for _ in range(iters):
        pi = softmax(Psi @ theta, axis=-1)  # (N, A)
        grad = -(Psi[np.arange(len(acts)), acts] - np.einsum("na,naf->nf", pi, Psi)).mean(0) + l2 * theta
        theta = opt.step(theta, grad)
    return FeaturePolicy(theta)


def behavior_cloning(envs, demos, **kw) -> FeaturePolicy:
    data = [(d.env_id, int(s), int(a)) for d in demos for s, a in zip(d.states, d.actions)]
    return fit_feature_policy(envs, data, **kw)
