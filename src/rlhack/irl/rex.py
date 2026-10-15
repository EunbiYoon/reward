"""Reward learning from ranked trajectories.

* T-REX (Brown et al., ICML 2019): maximum-likelihood Bradley-Terry over
  pairwise rankings of (possibly suboptimal) trajectories.
* B-REX (Brown et al., ICML 2020, "Safe Imitation Learning via Fast Bayesian
  Reward Inference from Preferences"): MCMC over the same likelihood with
  unit-norm reward weights; cheap because no MDP solve is needed.
"""
from __future__ import annotations

import numpy as np

from ..preferences import preference_loglik
from ..utils import Adam, sigmoid
from .bayesian import Posterior, metropolis_hastings


def trex(prefs, iters: int = 500, lr: float = 0.05, l2: float = 1e-3, K: int | None = None) -> np.ndarray:
    X = np.stack([p.mu_a - p.mu_b for p in prefs])
    y = np.array([p.label for p in prefs])
    w = np.zeros(X.shape[1] if K is None else K)
    opt = Adam(lr)
    for _ in range(iters):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y) + l2 * w  # minimize NLL
        w = opt.step(w, grad)
    return w


def brex(prefs, rng, beta: float = 5.0, n_samples: int = 1000, burn_in: int = 500, thin: int = 2,
         step: float = 0.2) -> Posterior:
    K = prefs[0].mu_a.shape[0]

    def log_target(w, _):
        return float(preference_loglik(w, prefs, beta)), None

    return metropolis_hastings(log_target, rng.normal(size=K), n_samples, rng, step=step, burn_in=burn_in,
                               thin=thin, sphere=True)
