"""Trajectories, demonstrations, and ranked (noise-injected) rollouts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import boltzmann_policy, value_iteration
from .utils import sample_categorical


@dataclass
class Trajectory:
    env_id: int
    states: np.ndarray
    actions: np.ndarray


def rollout(env, pi: np.ndarray, rng: np.random.Generator, horizon: int, env_id: int = -1) -> Trajectory:
    s = env.start
    states, actions = [], []
    for _ in range(horizon):
        a = sample_categorical(pi[s], rng)
        states.append(s)
        actions.append(a)
        s = env.sample_next(s, a, rng)
    return Trajectory(env_id, np.array(states), np.array(actions))


def expert_policy(env, w, beta: float) -> np.ndarray:
    Q, _ = value_iteration(env.P, env.reward(w), env.gamma)
    return boltzmann_policy(Q, beta)


def feature_counts(env, traj: Trajectory, gamma: float) -> np.ndarray:
    disc = gamma ** np.arange(len(traj.states))
    return disc @ env.phi[traj.states]


def discounted_return(env, traj: Trajectory, w, gamma: float) -> float:
    return float(feature_counts(env, traj, gamma) @ np.asarray(w))


def generate_demos(envs, n: int, rng, w_true, beta: float = 100.0, horizon: int = 30):
    """n Boltzmann-rational expert demonstrations, round-robin over training maps."""
    pis = [expert_policy(e, w_true, beta) for e in envs]
    return [rollout(envs[i % len(envs)], pis[i % len(envs)], rng, horizon, env_id=i % len(envs)) for i in range(n)]


def generate_ranked_trajectories(envs, n: int, rng, w_true, beta: float = 100.0, horizon: int = 30):
    """Noise-injected rollouts (epsilon-mixtures of expert and uniform) for T-REX / B-REX."""
    pis = [expert_policy(e, w_true, beta) for e in envs]
    out = []
    for i in range(n):
        k = i % len(envs)
        eps = rng.random()
        pi = (1 - eps) * pis[k] + eps / envs[k].n_actions
        out.append(rollout(envs[k], pi, rng, horizon, env_id=k))
    return out
