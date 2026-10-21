"""Neural-network pieces shared by the trainers (PPO, GRPO, A2C, DQN, SAC) for the Overcooked learner.

Two policy types with the same interface:
    sample(obs)          -> (actions, log_probs)      used while collecting data
    log_prob(obs, act)   -> log pi(a|s) per sample   used in the update
    entropy(obs)         -> per-sample entropy
    mode(obs)            -> deterministic action
  GaussianPolicy     continuous actions, actions are float vectors (kept for continuous envs)
  CategoricalPolicy  discrete actions (Overcooked), actions are int64 indices
  QPolicy            DQN's Q-network: argmax Q, epsilon-greedy sampling
make_policy(env) picks one from env.discrete. ReplayBuffer is used by the off-policy trainers (DQN, SAC).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


class RunningMeanStd:
    """Observation normalizer (frozen and saved with the policy for evaluation)."""

    def __init__(self, dim: int, clip: float = 10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x)
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
    def from_state(cls, st: dict) -> "RunningMeanStd":
        r = cls(len(st["mean"]))
        r.mean, r.var, r.count = np.asarray(st["mean"]), np.asarray(st["var"]), float(st["count"])
        return r


def mlp(i: int, o: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, o))


class GaussianPolicy(nn.Module):
    discrete = False

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mu(obs), self.log_std.exp())

    def sample(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1)

    def log_prob(self, obs, act):
        return self.dist(obs).log_prob(act).sum(-1)

    def entropy(self, obs):
        return self.dist(obs).entropy().sum(-1)

    def mode(self, obs):
        return self.mu(obs)


class CategoricalPolicy(nn.Module):
    discrete = True

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.logits = mlp(obs_dim, n_actions, hidden)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits(obs))

    def sample(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a)

    def log_prob(self, obs, act):
        return self.dist(obs).log_prob(act.long())

    def entropy(self, obs):
        return self.dist(obs).entropy()

    def mode(self, obs):
        return self.logits(obs).argmax(-1)


class QPolicy(CategoricalPolicy):
    """DQN's Q-network behind the policy interface: `logits` holds Q(s, .), mode() is argmax Q and
    sample() is epsilon-greedy (its log-prob is a dummy 0; log_prob/entropy of softmax(Q) are unused)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, eps: float = 0.05):
        super().__init__(obs_dim, n_actions, hidden)
        self.eps = eps

    def sample(self, obs):
        q = self.logits(obs)
        a = q.argmax(-1)
        explore = torch.rand(a.shape) < self.eps
        a = torch.where(explore, torch.randint(q.shape[-1], a.shape), a)
        return a, torch.zeros(a.shape)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.v = mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class ReplayBuffer:
    """Ring buffer of (obs, action, reward, next_obs) for discrete actions. Observations are stored raw
    and normalized with the CURRENT obs_rms when sampled, so old data follows the normalizer."""

    def __init__(self, size: int, obs_dim: int):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.size, self.n, self.i = size, 0, 0

    def add(self, o, a, r, o2):
        self.o[self.i], self.a[self.i], self.r[self.i], self.o2[self.i] = o, a, r, o2
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch: int, obs_rms: RunningMeanStd, rng: np.random.Generator):
        idx = rng.integers(0, self.n, batch)
        return (torch.as_tensor(obs_rms.normalize(self.o[idx])), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(obs_rms.normalize(self.o2[idx])))


def make_policy(env, hidden: int):
    if getattr(env, "discrete", False):
        return CategoricalPolicy(env.obs_dim, env.n_actions, hidden)
    return GaussianPolicy(env.obs_dim, env.act_dim, hidden)


def action_buffer(n: int, env) -> np.ndarray:
    if getattr(env, "discrete", False):
        return np.zeros(n, np.int64)
    return np.zeros((n, env.act_dim), np.float32)


def to_env_action(a: torch.Tensor):
    """Batch-of-one tensor -> what env.step expects (int index or float vector)."""
    x = a[0]
    return int(x.item()) if x.dim() == 0 else x.numpy()


@torch.no_grad()
def act(policy, obs_rms: RunningMeanStd, obs: np.ndarray, deterministic: bool = True):
    o = torch.as_tensor(obs_rms.normalize(obs)[None])
    a = policy.mode(o) if deterministic else policy.sample(o)[0]
    return to_env_action(a)


def save_checkpoint(path: str, policy, obs_rms: RunningMeanStd, meta: dict):
    meta = {**meta, "discrete": bool(getattr(policy, "discrete", False)), "q": isinstance(policy, QPolicy)}
    torch.save({"policy": policy.state_dict(), "obs_rms": obs_rms.state(), "meta": meta}, path)


def load_checkpoint(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    if m.get("q"):
        pol = QPolicy(m["obs_dim"], m["n_actions"], m.get("hidden", 64))
    elif m.get("discrete"):
        pol = CategoricalPolicy(m["obs_dim"], m["n_actions"], m.get("hidden", 64))
    else:
        pol = GaussianPolicy(m["obs_dim"], m["act_dim"], m.get("hidden", 64))
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, RunningMeanStd.from_state(ck["obs_rms"]), m
