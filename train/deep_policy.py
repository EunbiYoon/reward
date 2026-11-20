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

    def __init__(self, dim: int, clip: float = 10.0, identity: bool = False):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip
        self.identity = identity        # True: pass observations through unchanged (grid / CNN policies)

    def update(self, x: np.ndarray):
        if self.identity:
            return
        x = np.atleast_2d(x)
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.identity:
            return np.asarray(x, np.float32)
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count, "identity": self.identity}

    @classmethod
    def from_state(cls, st: dict) -> "RunningMeanStd":
        r = cls(len(st["mean"]), identity=bool(st.get("identity", False)))
        r.mean, r.var, r.count = np.asarray(st["mean"]), np.asarray(st["var"]), float(st["count"])
        return r


def mlp(i: int, o: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, o))


class GridConv(nn.Module):
    """Flat observation = a (C, H, W) grid raveled + `extra` scalars at the end (RWARE: time left).
    Two 3x3 convolutions over the grid (same pattern recognised in every cell), flattened, scalars appended."""

    def __init__(self, grid: tuple, extra: int, channels: int = 32):
        super().__init__()
        self.grid, self.extra = tuple(int(g) for g in grid), int(extra)
        c = self.grid[0]
        self.conv = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.out_dim = channels * self.grid[1] * self.grid[2] + self.extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        n = self.grid[0] * self.grid[1] * self.grid[2]
        g = self.conv(x[:, :n].reshape(-1, *self.grid)).flatten(1)
        return torch.cat([g, x[:, n:]], 1).reshape(*lead, self.out_dim)


def net(i: int, o: int, hidden: int, grid: tuple | None = None) -> nn.Sequential:
    """mlp(), or with `grid` a CNN trunk (GridConv) followed by the same two hidden layers. Either way an
    nn.Sequential whose [:-1] is the feature trunk (used by PartnerHead)."""
    if grid is None:
        return mlp(i, o, hidden)
    conv = GridConv(grid, i - int(np.prod(grid)))
    return nn.Sequential(conv, nn.Linear(conv.out_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
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

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, grid: tuple | None = None):
        super().__init__()
        self.grid = tuple(grid) if grid else None
        self.logits = net(obs_dim, n_actions, hidden, self.grid)

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
    def __init__(self, obs_dim: int, hidden: int = 64, grid: tuple | None = None):
        super().__init__()
        self.v = net(obs_dim, 1, hidden, grid)

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


class PartnerHead(nn.Module):
    """Auxiliary partner model (H12): predicts the partner's next action from the policy's TRUNK
    (every layer of the categorical policy but the last), so the cross-entropy shapes the features the
    policy acts on. Kept out of the policy module so checkpoints load unchanged."""

    def __init__(self, hidden: int, n_partner_actions: int):
        super().__init__()
        self.head = nn.Linear(hidden, n_partner_actions)

    def forward(self, policy: CategoricalPolicy, obs: torch.Tensor) -> torch.Tensor:
        return self.head(policy.logits[:-1](obs))

    def loss(self, policy: CategoricalPolicy, obs: torch.Tensor, partner_actions: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(self(policy, obs), partner_actions.long())


def make_partner_head(env, policy, hidden: int, aux_coef: float):
    """PartnerHead for aux_coef > 0, else None. Needs a discrete policy and an env whose step events carry
    the partner's action index as 'a1' (overcooked)."""
    if aux_coef <= 0:
        return None
    if not getattr(policy, "discrete", False):
        raise ValueError("the partner-prediction head needs a discrete policy")
    return PartnerHead(hidden, env.n_actions)


def partner_label(info: dict) -> int:
    if "a1" not in info:
        raise ValueError("--aux-coef > 0 needs an env that reports the partner's action ('a1'); use --env overcooked")
    return int(info["a1"])


def make_policy(env, hidden: int):
    """An env with `grid_shape` (RWARE) gets a CNN trunk; the others an MLP."""
    if getattr(env, "discrete", False):
        return CategoricalPolicy(env.obs_dim, env.n_actions, hidden, getattr(env, "grid_shape", None))
    return GaussianPolicy(env.obs_dim, env.act_dim, hidden)


def make_value(env, hidden: int) -> ValueNet:
    return ValueNet(env.obs_dim, hidden, getattr(env, "grid_shape", None))


def make_obs_rms(env) -> RunningMeanStd:
    """Observation normalizer; identity for grid observations (0/1 layers go to the CNN as they are)."""
    return RunningMeanStd(env.obs_dim, identity=getattr(env, "grid_shape", None) is not None)


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
    meta = {**meta, "discrete": bool(getattr(policy, "discrete", False)), "q": isinstance(policy, QPolicy),
            "grid": list(policy.grid) if getattr(policy, "grid", None) else None}
    torch.save({"policy": policy.state_dict(), "obs_rms": obs_rms.state(), "meta": meta}, path)


def load_checkpoint(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    if m.get("q"):
        pol = QPolicy(m["obs_dim"], m["n_actions"], m.get("hidden", 64))
    elif m.get("discrete"):
        pol = CategoricalPolicy(m["obs_dim"], m["n_actions"], m.get("hidden", 64), m.get("grid"))
    else:
        pol = GaussianPolicy(m["obs_dim"], m["act_dim"], m.get("hidden", 64))
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, RunningMeanStd.from_state(ck["obs_rms"]), m
