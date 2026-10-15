"""Small numerical helpers shared across the package."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(log_sigmoid(x))


def sample_categorical(p: np.ndarray, rng: np.random.Generator) -> int:
    idx = int(np.searchsorted(np.cumsum(p), rng.random() * p.sum()))
    return min(idx, len(p) - 1)


class Adam:
    """Minimal Adam optimizer for numpy parameter vectors (minimization)."""

    def __init__(self, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def normalize_reward(r: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale a reward vector to max |r| = 1 (preserves the optimal policy)."""
    return r / (np.abs(r).max() + eps)
