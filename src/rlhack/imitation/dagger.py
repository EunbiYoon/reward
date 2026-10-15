"""DAgger (Ross, Gordon & Bagnell, AISTATS 2011) on the training maps.

Note: DAgger queries the expert interactively on learner-visited states, so it
receives strictly more supervision than BC for the same initial demo budget.
"""
from __future__ import annotations

import numpy as np

from ..demos import rollout
from ..mdp import value_iteration
from .bc import FeaturePolicy, fit_feature_policy


def dagger(envs, demos, w_true, rng, rounds: int = 4, rollouts_per_round: int | None = None,
           horizon: int = 30, **fit_kw) -> FeaturePolicy:
    expert_act = {}
    data = [(d.env_id, int(s), int(a)) for d in demos for s, a in zip(d.states, d.actions)]
    policy = fit_feature_policy(envs, data, **fit_kw)
    n_roll = rollouts_per_round or max(1, len(demos))
    for _ in range(rounds):
        for k in range(n_roll):
            e = k % len(envs)
            env = envs[e]
            if e not in expert_act:
                expert_act[e] = value_iteration(env.P, env.reward(w_true), env.gamma)[0].argmax(-1)
            traj = rollout(env, policy.probs(env), rng, horizon, env_id=e)
            data += [(e, int(s), int(expert_act[e][s])) for s in traj.states]
        policy = fit_feature_policy(envs, data, **fit_kw)
    return policy
