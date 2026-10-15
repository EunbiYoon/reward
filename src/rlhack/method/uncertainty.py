"""Proposed method: uncertainty-aware reward models from Bayesian IRL.

1. **Variance penalty** -- deploy the pessimistic reward
       r_lam(s) = E_post[r(s)] - lam * Std_post[r(s)]
   so states whose reward the demonstrations do not pin down (unseen hazards,
   spurious correlates) are not exploited by the RL agent.
2. **Active preference queries in high-uncertainty states** -- on the deployment
   map, roll out trajectories under posterior samples, keep pairs that pass
   through the top-uncertainty states, and ask the (synthetic) human about the
   pair the posterior disagrees on most (max p(1-p), an information-gain proxy).
   Answers are added as a Bradley-Terry likelihood term and the posterior is
   re-sampled.
"""
from __future__ import annotations

import numpy as np

from ..demos import feature_counts, rollout
from ..irl.bayesian import Posterior, bayesian_irl
from ..mdp import boltzmann_policy, value_iteration
from ..utils import sigmoid


def penalized_reward(env, posterior: Posterior, lam: float, param: str = "contrast") -> np.ndarray:
    """E[r(s)] - lam * Std[r(s)] under the posterior.

    ``param="contrast"`` (default) first maps samples to the identifiable
    empty->0 / goal->1 parameterization (see ``Posterior.contrast``); ``"raw"`` uses
    the samples as drawn. With ``raw`` the penalty is almost a uniform extra step
    cost and does not target unidentified features (see docs/findings.md).
    """
    if param == "contrast":
        posterior = posterior.contrast()
    elif param != "raw":
        raise ValueError(param)
    return posterior.reward_mean(env) - lam * posterior.reward_std(env)


def high_uncertainty_states(env, posterior: Posterior, top_frac: float = 0.1) -> np.ndarray:
    std = posterior.contrast().reward_std(env).copy()
    std[env.sink] = -np.inf
    k = max(1, int(np.ceil(top_frac * (env.n_states - 1))))
    return np.argsort(-std)[:k]


QUERY_STRATEGIES = ("ours", "disagreement", "hot_states", "random")


def select_query_pairs(env, posterior: Posterior, rng, n_queries: int, n_candidates: int = 24,
                       beta_pref: float = 5.0, beta_rollout: float = 5.0, horizon: int = 30,
                       top_frac: float = 0.1, strategy: str = "ours"):
    """Pick ``n_queries`` disjoint trajectory pairs from posterior-sample rollouts.

    strategy (all draw from the same candidate pool, so only the *selection* differs):
      * ``ours``         -- disagreement p(1-p), doubled if a pair touches a top-uncertainty state
      * ``disagreement`` -- disagreement p(1-p) only (query-by-committee / info-gain proxy)
      * ``hot_states``   -- uniformly random among pairs that touch a top-uncertainty state
      * ``random``       -- uniformly random pairs
    """
    assert strategy in QUERY_STRATEGIES, strategy
    W = posterior.samples
    idx = rng.choice(len(W), size=min(n_candidates, len(W)), replace=False)
    trajs = []
    for i in idx:
        Q, _ = value_iteration(env.P, env.reward(W[i]), env.gamma, tol=1e-4)
        trajs.append(rollout(env, boltzmann_policy(Q, beta_rollout), rng, horizon))
    mus = np.stack([feature_counts(env, t, env.gamma) for t in trajs])
    hot = set(high_uncertainty_states(env, posterior, top_frac).tolist())
    touches_hot = np.array([bool(hot.intersection(t.states.tolist())) for t in trajs])

    scored = []
    for a in range(len(trajs)):
        for b in range(a + 1, len(trajs)):
            diff = mus[a] - mus[b]
            if np.allclose(diff, 0):
                continue
            hot_pair = float(touches_hot[a] or touches_hot[b])
            if strategy == "random":
                score = rng.random()
            elif strategy == "hot_states":
                score = hot_pair + rng.random() * 0.5
            else:
                p = sigmoid(beta_pref * (W @ diff)).mean()
                score = p * (1 - p) * ((1.0 + hot_pair) if strategy == "ours" else 1.0)
            scored.append((score, a, b))
    scored.sort(reverse=True)
    used, pairs = set(), []
    for _, a, b in scored:
        if a in used or b in used:
            continue
        pairs.append((trajs[a], trajs[b]))
        used.update((a, b))
        if len(pairs) == n_queries:
            break
    return pairs


def active_bayesian_irl(train_envs, demos, deploy_env, human, rng, posterior: Posterior | None = None,
                        rounds: int = 2, queries_per_round: int = 5, birl_kw: dict | None = None,
                        query_kw: dict | None = None):
    """Returns (final posterior, list of collected preferences)."""
    birl_kw = dict(birl_kw or {})
    query_kw = dict(query_kw or {})
    prefs = []
    post = posterior or bayesian_irl(train_envs, demos, rng, **birl_kw)
    for _ in range(rounds):
        for ta, tb in select_query_pairs(deploy_env, post, rng, queries_per_round,
                                         beta_pref=birl_kw.get("beta_pref", 5.0), **query_kw):
            prefs.append(human.query(deploy_env, ta, tb))
        post = bayesian_irl(train_envs, demos, rng, prefs=prefs, w0=post.mean, **birl_kw)
    return post, prefs


def auto_lambda(env, posterior: Posterior, lambdas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), alpha: float = 0.95,
                n_eval: int = 100, rng=None, param: str = "contrast"):
    """Choose lambda without the true reward: for each candidate, solve the MDP under the
    penalized reward exactly and score that policy by the alpha-VaR of its expected value
    difference over the posterior (Brown & Niekum, 2018). Returns (best_lambda, {lam: VaR})."""
    from ..metrics import var_evd  # local import avoids a cycle
    from ..mdp import greedy_policy

    rng = rng or np.random.default_rng(0)
    W = (posterior.contrast() if param == "contrast" else posterior).samples
    if len(W) > n_eval:
        W = W[rng.choice(len(W), n_eval, replace=False)]
    scores = {}
    for lam in lambdas:
        Q, _ = value_iteration(env.P, penalized_reward(env, posterior, lam, param), env.gamma, tol=1e-5)
        scores[lam] = var_evd(env, greedy_policy(Q), W, alpha)
    best = min(scores, key=lambda k: (scores[k], k))
    return best, scores
