"""Exact tabular MDP solvers. All functions accept batched inputs (leading dims)."""
from __future__ import annotations

import numpy as np

from .utils import logsumexp, softmax


def _expect_next(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    # P: (..., S, A, S), V: (..., S) -> (..., S, A)
    return np.einsum("...sat,...t->...sa", P, V)


def value_iteration(P, R, gamma, V0=None, tol=1e-6, max_iter=2000):
    """Bellman optimality. R is a state reward (..., S). Returns (Q, V)."""
    V = np.zeros(R.shape) if V0 is None else V0.copy()
    for _ in range(max_iter):
        Q = R[..., None] + gamma * _expect_next(P, V)
        V_new = Q.max(-1)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    Q = R[..., None] + gamma * _expect_next(P, V)
    return Q, V


def soft_value_iteration(P, R, gamma, beta=1.0, tol=1e-6, max_iter=2000):
    """Soft (MaxEnt) Bellman backup. Returns (Q, V, pi)."""
    V = np.zeros(R.shape)
    for _ in range(max_iter):
        Q = R[..., None] + gamma * _expect_next(P, V)
        V_new = logsumexp(beta * Q, axis=-1) / beta
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    Q = R[..., None] + gamma * _expect_next(P, V)
    return Q, V, softmax(beta * Q, axis=-1)


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    pi = np.zeros_like(Q)
    np.put_along_axis(pi, Q.argmax(-1)[..., None], 1.0, axis=-1)
    return pi


def boltzmann_policy(Q: np.ndarray, beta: float) -> np.ndarray:
    return softmax(beta * Q, axis=-1)


def transition_under_policy(P, pi):
    return np.einsum("sa,sat->st", pi, P)


def policy_evaluation(P, R, pi, gamma):
    P_pi = transition_under_policy(P, pi)
    S = P_pi.shape[0]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R)


def expected_return(P, R, pi, p0, gamma):
    return float(p0 @ policy_evaluation(P, R, pi, gamma))


def state_occupancy(P, pi, p0, gamma):
    """Normalized discounted state occupancy d(s) = (1-g) p0^T (I - g P_pi)^-1."""
    P_pi = transition_under_policy(P, pi)
    S = P_pi.shape[0]
    return (1 - gamma) * np.linalg.solve((np.eye(S) - gamma * P_pi).T, p0)


def state_action_occupancy(P, pi, p0, gamma):
    return state_occupancy(P, pi, p0, gamma)[:, None] * pi
