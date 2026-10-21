"""Shapley contribution of the learner from frozen-cook evaluations.

n agents: L (learner, seat 0) and partners P = {1 .. n-1}. v(S) = soups delivered when only
the agents in S move (the others are frozen and never move), v(empty) = 0:

    phi_L = sum over T subset of P of  |T|! (n-1-|T|)! / n!  [ v(T + L) - v(T) ]
    s     = phi_L / v(all)          the learner's Shapley share of the team's soups

For n = 2 this is phi_L = 1/2 [ (v(L,P) - v(P)) + v(L) ]. Every subset is evaluated, so n = 4
plays 2^4 - 1 = 15 coalitions (8 with the learner) per evaluation episode.

All coalitions are evaluated on the SAME episode seeds (common random numbers).
A frozen cook still stands on its cell, so in a cramped kitchen it can block the other
cook; this is a known limitation of the frozen-cook baseline.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

EVAL_SEED0 = 10_000


def _mean_play(env, seat0, seat1, n: int, before_episode=None) -> dict:
    acc: dict[str, list] = {}
    for k in range(n):
        if before_episode is not None:
            before_episode(EVAL_SEED0 + k)
        ep = env.play(seat0, seat1, EVAL_SEED0 + k)
        for key, val in ep.items():
            acc.setdefault(key, []).append(val)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def shapley(v: dict, n_agents: int) -> tuple[float, float]:
    """v: frozenset of moving partner seats T -> (v(T + L), v(T)). Returns (phi_L, s)."""
    phi = 0.0
    for T, (with_l, without_l) in v.items():
        k = len(T)
        phi += factorial(k) * factorial(n_agents - 1 - k) / factorial(n_agents) * (with_l - without_l)
    v_all = v[frozenset(range(1, n_agents))][0]
    s = phi / v_all if v_all > 0 else 0.0       # no team soups: the learner's share of nothing is 0
    return float(phi), float(s)


def evaluate_coalitions(env, seat0, partner: str, n: int, before_episode=None) -> dict:
    """seat0: a policy callable or 'greedy' (the scripted reference). partner: 'stay' | 'greedy',
    the kind of every partner seat 1 .. env.n_agents-1."""
    n_agents = getattr(env, "n_agents", 2)
    seats = tuple(range(1, n_agents))

    def kinds(T):
        """Per-partner kinds with only the seats in T moving (a str for 2 agents, as Kitchen expects)."""
        k = tuple(partner if i in T else "stay" for i in seats)
        return k[0] if n_agents == 2 else k

    # A 'stay' partner never moves anyway, so every coalition equals the one with no partner moving.
    subsets = [frozenset(c) for r in range(n_agents) for c in combinations(seats, r)]
    plays: dict = {}

    def play(learner_moves: bool, T: frozenset) -> dict:
        T = T if partner == "greedy" else frozenset()
        key = (learner_moves, T)
        if key not in plays:
            if not learner_moves and not T:
                plays[key] = {"soups": 0.0}                             # v(empty) = 0
            elif learner_moves:
                plays[key] = _mean_play(env, seat0, kinds(T), n, before_episode)
            else:
                plays[key] = _mean_play(env, "stay", kinds(T), n)
        return plays[key]

    v = {T: (play(True, T)["soups"], play(False, T)["soups"]) for T in subsets}
    phi, s = shapley(v, n_agents)
    both = play(True, frozenset(seats))                                 # v(all), the true objective
    p_alone = play(False, frozenset(seats))                             # v(P): learner frozen
    l_alone = play(True, frozenset())                                   # v(L): every partner frozen
    own_share = both["own_soups"] / both["soups"] if both["soups"] > 0 else 0.0
    return {
        "agents": n_agents,
        "soups": both["soups"],                       # v(all), the true objective
        "own_soups": both["own_soups"],
        "partner_soups": both["partner_soups"],       # all partners together
        "pickups": both["pickups0"],                  # learner's pickups (onion + dish, or shelves)
        **{f"{k}_pickups": both[f"{k}_pickups0"] for k in env.pickup_kinds},
        "v_partner_alone": p_alone["soups"],          # v(P): every partner moves, learner frozen
        "v_learner_alone": l_alone["soups"],          # v(L): every partner frozen
        "phi": phi,
        "s": s,
        "own_share": own_share,                       # share of soups the learner delivered itself
        "credit_gap": own_share - s,
    }


def policy_actor(policy, obs_rms, deterministic: bool):
    """Wrap a trained policy as a seat-0 controller."""
    from train.deep_policy import act
    return lambda obs: act(policy, obs_rms, obs, deterministic=deterministic)


def torch_reseed(seed: int):
    """Seed the policy's action sampling per episode, so every coalition sees the same draws."""
    try:
        import torch
        torch.manual_seed(int(seed))
    except ImportError:
        pass
