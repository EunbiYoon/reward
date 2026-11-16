"""Seat-0 behaviours that fill the trajectory pool the LLM judge compares (rlaif/collect.py).

The pool needs agents that help, agents that do nothing and, most importantly, agents that LOOK
busy without helping, so the judge's preferences can be audited against Shapley credit:

    greedy     the scripted cook / robot / forager (a competent teammate)
    eps25      greedy, but a uniformly random action 25% of the time
    eps50      greedy, but a uniformly random action 50% of the time
    random     a uniformly random action every step
    fidget     never idle: random moves and, half the time, INTERACT / LOAD / TOGGLE. Picks things
               up and puts them down all episode. Busy-looking, contributes almost nothing.
    stay       frozen in a parking cell (contributes exactly nothing)
    ckpt:<path>  a trained policy (e.g. a pickup-trained run from train/run.py: learned fake help)

Every behaviour is a callable obs -> action index with `reseed(seed)`, so all Shapley coalitions
of one episode see the same random draws (common random numbers, as in eval/coalitions.py).
"""
from __future__ import annotations

import numpy as np

BUILTIN = ("greedy", "eps25", "eps50", "random", "fidget", "stay")


class Behavior:
    name = "?"

    def __init__(self, env):
        self.env = env
        self.rng = np.random.RandomState(0)

    def reseed(self, seed: int):
        self.rng = np.random.RandomState((int(seed) * 7919 + 17) % (2 ** 32))

    def __call__(self, obs) -> int:
        raise NotImplementedError


class EpsGreedy(Behavior):
    def __init__(self, env, eps: float):
        super().__init__(env)
        self.eps = eps
        self.name = "greedy" if eps == 0 else "random" if eps >= 1 else f"eps{int(round(eps * 100))}"

    def __call__(self, obs) -> int:
        if self.eps > 0 and self.rng.rand() < self.eps:
            return int(self.rng.randint(self.env.n_actions))
        return self.env.scripted_index("greedy", 0)


class Fidget(Behavior):
    """Always doing something: never the idle action, the env's 'use' action (last index) half the time."""
    name = "fidget"

    def __call__(self, obs) -> int:
        use = self.env.n_actions - 1                    # INTERACT / LOAD / TOGGLE_LOAD / DO in every env
        if self.rng.rand() < 0.5:
            return use
        moves = [a for a in range(self.env.n_actions) if a not in (use, self.env.stay_action)]
        return int(moves[self.rng.randint(len(moves))])


class Checkpoint(Behavior):
    def __init__(self, env, path: str):
        super().__init__(env)
        from train.deep_policy import load_checkpoint
        self.policy, self.obs_rms, meta = load_checkpoint(path)
        if meta["obs_dim"] != env.obs_dim:
            raise ValueError(f"{path}: obs_dim {meta['obs_dim']} != env obs_dim {env.obs_dim}")
        self.name = "ckpt:" + "_".join(str(meta[k]) for k in ("algo", "reward", "partner")) + f"_s{meta['seed']}"

    def reseed(self, seed: int):
        import torch
        torch.manual_seed(int(seed))

    def __call__(self, obs) -> int:
        from train.deep_policy import act
        return act(self.policy, self.obs_rms, obs, deterministic=False)


def make_behavior(env, spec: str):
    """spec: one of BUILTIN or 'ckpt:<path>'. Returns 'stay' (the frozen seat) or a Behavior."""
    if spec == "stay":
        return "stay"
    if spec == "greedy":
        return EpsGreedy(env, 0.0)
    if spec in ("eps25", "eps50"):
        return EpsGreedy(env, int(spec[3:]) / 100)
    if spec == "random":
        return EpsGreedy(env, 1.0)
    if spec == "fidget":
        return Fidget(env)
    if spec.startswith("ckpt:"):
        return Checkpoint(env, spec[5:])
    raise ValueError(f"unknown behaviour {spec!r}; choose from {BUILTIN} or ckpt:<path>")
