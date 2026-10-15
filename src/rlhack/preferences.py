"""Synthetic human preferences over trajectory pairs.

Two human models from Knox et al. (2022), "Models of Human Preference for
Learning Reward Functions": Bradley-Terry over *partial return* and over
*regret* (sum of optimal advantages). The learner always assumes the
partial-return model, so ``model="regret"`` is a model-misspecification ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demos import Trajectory, discounted_return, feature_counts
from .mdp import value_iteration
from .utils import log_sigmoid, sigmoid


@dataclass
class Preference:
    mu_a: np.ndarray
    mu_b: np.ndarray
    label: float  # 1.0 if a is preferred


class SyntheticHuman:
    def __init__(self, w_true, model: str = "partial_return", beta: float = 5.0, rng=None):
        assert model in ("partial_return", "regret")
        self.w_true = np.asarray(w_true)
        self.model, self.beta = model, beta
        self.rng = rng or np.random.default_rng()
        self._cache = {}

    def _regret(self, env, traj: Trajectory) -> float:
        key = id(env)
        if key not in self._cache:
            self._cache[key] = value_iteration(env.P, env.reward(self.w_true), env.gamma)
        Q, V = self._cache[key]
        disc = env.gamma ** np.arange(len(traj.states))
        return float(disc @ (V[traj.states] - Q[traj.states, traj.actions]))

    def prob_a(self, env_a, ta: Trajectory, tb: Trajectory, env_b=None) -> float:
        env_b = env_a if env_b is None else env_b
        if self.model == "partial_return":
            d = (discounted_return(env_a, ta, self.w_true, env_a.gamma)
                 - discounted_return(env_b, tb, self.w_true, env_b.gamma))
        else:
            d = self._regret(env_b, tb) - self._regret(env_a, ta)
        return float(sigmoid(np.array(self.beta * d)))

    def query(self, env_a, ta: Trajectory, tb: Trajectory, env_b=None) -> Preference:
        env_b = env_a if env_b is None else env_b
        label = float(self.rng.random() < self.prob_a(env_a, ta, tb, env_b))
        return Preference(feature_counts(env_a, ta, env_a.gamma), feature_counts(env_b, tb, env_b.gamma), label)


def preference_loglik(W: np.ndarray, prefs, beta: float) -> np.ndarray:
    """Bradley-Terry log-likelihood for one (K,) or many (n, K) weight vectors."""
    if not prefs:
        return np.zeros(W.shape[:-1]) if W.ndim > 1 else 0.0
    X = np.stack([p.mu_a - p.mu_b for p in prefs])  # (m, K)
    y = np.array([p.label for p in prefs])
    logits = beta * (W @ X.T)  # (..., m)
    return (y * log_sigmoid(logits) + (1 - y) * log_sigmoid(-logits)).sum(-1)


def pairwise_preferences(envs, trajs, human: SyntheticHuman, rng, max_pairs: int):
    """All (or a random subset of) pairs among trajectories.

    Pairs may span different training maps: both human models score each
    trajectory on its own map, and the learner only sees feature counts.
    """
    pairs = [(i, j) for i in range(len(trajs)) for j in range(i + 1, len(trajs))]
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    return [human.query(envs[trajs[i].env_id], trajs[i], trajs[j], envs[trajs[j].env_id]) for i, j in pairs]
