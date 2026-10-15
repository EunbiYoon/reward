"""Bayesian IRL (Ramachandran & Amir, IJCAI 2007) via Metropolis-Hastings, plus the
alpha-VaR performance bound of Brown & Niekum (AAAI 2018) computed in metrics.py.

Likelihood: Boltzmann-rational demonstrator, pi(a|s) ∝ exp(beta Q*_w(s, a)),
evaluated jointly over every training map that contributed demonstrations
(value iteration is batched across maps and warm-started from the current chain state).
Optional pairwise preferences add a Bradley-Terry term (used by the active method).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mdp import value_iteration
from ..preferences import preference_loglik
from ..utils import logsumexp


@dataclass
class Posterior:
    samples: np.ndarray  # (n, K)
    accept_rate: float = float("nan")
    info: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.samples.mean(0)

    @property
    def std(self):
        return self.samples.std(0)

    def reward_samples(self, env) -> np.ndarray:
        return self.samples @ env.phi.T  # (n, S)

    def reward_mean(self, env):
        return self.reward_samples(env).mean(0)

    def reward_std(self, env):
        return self.reward_samples(env).std(0)

    def contrast(self, zero_feature: int = 0, unit_feature: int = 1, min_gap: float = 1e-3) -> "Posterior":
        """Map every sample to an identifiable parameterization.

        Linear Bayesian IRL is only identified up to scale (Boltzmann temperature vs.
        reward magnitude) and, partly, shift. Raw per-feature posterior std is therefore
        dominated by that global ambiguity and is nearly uniform across features, so it
        does not single out the feature the demonstrations never pinned down. Here each
        sample is rescaled so that w[zero_feature] = 0 and w[unit_feature] = 1
        (default: empty cell -> 0, goal -> 1); samples with w[unit] <= w[zero] are
        dropped (fraction kept is stored in ``info['contrast_kept']``). The std of
        the remaining coordinates then measures genuinely unidentified trade-offs.
        """
        W = self.samples
        d = W[:, unit_feature] - W[:, zero_feature]
        keep = d > min_gap
        if keep.sum() < 2:  # degenerate posterior: fall back to the raw samples
            return Posterior(W, self.accept_rate, {**self.info, "contrast_kept": float(keep.mean())})
        Wc = (W[keep] - W[keep, zero_feature][:, None]) / d[keep, None]
        return Posterior(Wc, self.accept_rate, {**self.info, "contrast_kept": float(keep.mean())})


class DemoLikelihood:
    def __init__(self, envs, demos, beta: float, gamma: float):
        ids = sorted({d.env_id for d in demos})
        remap = {k: i for i, k in enumerate(ids)}
        self.P = np.stack([envs[k].P for k in ids])
        self.phi = np.stack([envs[k].phi for k in ids])
        self.e = np.concatenate([np.full(len(d.states), remap[d.env_id]) for d in demos])
        self.s = np.concatenate([d.states for d in demos])
        self.a = np.concatenate([d.actions for d in demos])
        self.beta, self.gamma = beta, gamma

    def __call__(self, w, V0=None):
        R = self.phi @ w
        Q, V = value_iteration(self.P, R, self.gamma, V0=V0, tol=1e-4, max_iter=500)
        bq = self.beta * Q
        logpi = bq - logsumexp(bq, axis=-1)[..., None]
        return float(logpi[self.e, self.s, self.a].sum()), V


def metropolis_hastings(log_target, w0, n_samples: int, rng, step: float = 0.1, burn_in: int = 500,
                        thin: int = 2, sphere: bool = False, adapt: bool = True) -> Posterior:
    w = np.array(w0, dtype=float)
    if sphere:
        w = w / np.linalg.norm(w)
    lp, aux = log_target(w, None)
    samples, acc, acc_window = [], 0, 0
    total = burn_in + n_samples * thin
    for i in range(total):
        prop = w + step * rng.normal(size=w.shape)
        if sphere:
            prop = prop / np.linalg.norm(prop)
        lp_p, aux_p = log_target(prop, aux)
        if np.log(rng.random()) < lp_p - lp:
            w, lp, aux = prop, lp_p, aux_p
            acc_window += 1
            if i >= burn_in:
                acc += 1
        if adapt and i < burn_in and (i + 1) % 50 == 0:  # tune toward ~0.3 acceptance
            rate = acc_window / 50
            step *= 1.3 if rate > 0.4 else (0.7 if rate < 0.2 else 1.0)
            acc_window = 0
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(w.copy())
    return Posterior(np.array(samples), acc / max(1, total - burn_in), {"step": step})


def bayesian_irl(envs, demos, rng, prefs=None, beta: float = 10.0, beta_pref: float = 5.0,
                 prior_std: float = 1.0, n_samples: int = 500, burn_in: int = 500, thin: int = 2,
                 step: float = 0.1, w0=None, gamma: float | None = None) -> Posterior:
    gamma = envs[0].gamma if gamma is None else gamma
    lik = DemoLikelihood(envs, demos, beta, gamma) if demos else None
    prefs = prefs or []

    def log_target(w, V0):
        lp = -0.5 * np.sum((w / prior_std) ** 2)
        V = None
        if lik is not None:
            ll, V = lik(w, V0)
            lp += ll
        lp += float(preference_loglik(w, prefs, beta_pref))
        return lp, V

    K = envs[0].phi.shape[1]
    w0 = np.zeros(K) if w0 is None else w0
    return metropolis_hastings(log_target, w0, n_samples, rng, step=step, burn_in=burn_in, thin=thin)
