"""Mid-training checkpoints, so a run on a preemptible / time-limited cluster job resumes where it stopped.

The trainer saves its full state (networks, optimizer, observation normalizer, counters, the env's
training-step counter that drives the fading shaping, torch RNG) at most every `every_s` seconds,
at an update boundary, atomically (write + rename). On start it loads the file if present.
A resumed run is not bit-identical to an uninterrupted one (PPO restarts the episode in progress
and the env's partner RNG is reseeded), but it trains the same amount on the same objective.
"""
from __future__ import annotations

import os
import time

import torch


def base_env(env):
    """The environment under rlaif's LearnedReward wrapper (it owns total_steps)."""
    return env.env if hasattr(env, "rm") and hasattr(env, "rm_path") else env


class Resumer:
    def __init__(self, path: str, every_s: float = 300.0):
        self.every_s = float(os.environ.get("RESUME_EVERY_S", every_s))     # env override for tests
        self.path, self.last = path, time.time()

    def load(self) -> dict | None:
        if not self.path or not os.path.exists(self.path):
            return None
        st = torch.load(self.path, map_location="cpu", weights_only=False)
        print(f"    resumed from {self.path} at step {st['steps']:,}", flush=True)
        return st

    def maybe_save(self, make_state, force: bool = False):
        if not self.path or not (force or time.time() - self.last >= self.every_s):
            return
        tmp = self.path + ".tmp"
        torch.save(make_state(), tmp)
        os.replace(tmp, self.path)
        self.last = time.time()

    def done(self):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
