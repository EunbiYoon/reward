"""Experiment configuration (matches the proposal). Override from the CLI in eval/run.py."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass
class Config:
    env: str = "kitchen"              # kitchen (Overcooked-AI) | warehouse (RWARE) | foraging (LBF)
    layout: str = "cramped_room"
    rewards: tuple = ("pickup", "team", "individual")
    partners: tuple = ("stay", "greedy")
    algos: tuple = ("ppo", "grpo")
    agents: tuple = (2,)              # number of agents (learner + identical partners); 3, 4 need warehouse / foraging
    seeds: tuple = (0, 1, 2)
    env_steps: int = 2_000_000        # per run, same budget for PPO and GRPO
    horizon: int = 400
    anneal_frac: float = 0.5          # official shaping reaches 0 at this fraction of env_steps
    pickup_bonus: float = 3.0
    eval_episodes: int = 10           # fixed eval seeds 10_000 + k, shared by every coalition
    eval_deterministic: bool = False  # argmax policies often get stuck in Overcooked
    hidden: int = 128

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.98
    ppo_rollout_steps: int = 4000
    ppo_epochs: int = 8
    ppo_minibatch: int = 500
    ppo_lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # GRPO (critic free; one group = G episodes from the same reset)
    grpo_group_size: int = 8
    grpo_epochs: int = 8
    grpo_minibatch: int = 500
    grpo_lr: float = 3e-4
    grpo_advantage: str = "process"

    # A2C (PPO's rollout and critic, one unclipped gradient step per rollout; not in the default grid)
    a2c_rollout_steps: int = 128
    a2c_lr: float = 7e-4

    # DQN (Double-DQN target, replay, epsilon-greedy; not in the default grid)
    dqn_buffer: int = 200_000
    dqn_batch: int = 128
    dqn_lr: float = 1e-4
    dqn_learning_starts: int = 10_000
    dqn_train_every: int = 4
    dqn_target_every: int = 2_000
    dqn_explore_frac: float = 0.2
    dqn_eps_end: float = 0.05

    # SAC (discrete-action SAC, twin Q, learned temperature; not in the default grid)
    sac_buffer: int = 200_000
    sac_batch: int = 128
    sac_lr: float = 3e-4
    sac_learning_starts: int = 10_000
    sac_train_every: int = 4
    sac_tau: float = 0.005
    sac_target_entropy_scale: float = 0.89

    # evaluation criteria (proposal Sec. 5)
    h1_s_max: float = 0.1             # H1: s <= 0.1 ...
    h1_frac: float = 0.5              # ... and more pickups than the reference in >= half the runs
    h3_gap: float = 0.2               # H3: own share of soups - s >= 0.2

    def to_dict(self) -> dict:
        return asdict(self)


QUICK = dict(seeds=(0,), env_steps=200_000, eval_episodes=3)

# Per-environment defaults, applied before --quick and the CLI. The pickup bonus is about the
# same fraction of one unit of team reward as in Overcooked (3 / 20 per soup).
ENV_DEFAULTS = {
    "kitchen": {},
    "warehouse": dict(layout="tiny", horizon=500, pickup_bonus=0.15),     # 1 per delivery
    "foraging": dict(layout="8x8-3f", horizon=50, pickup_bonus=0.05,      # a whole field = 1
                     ppo_rollout_steps=2000, ppo_minibatch=250),
}


def make_config(quick: bool = False, **overrides) -> Config:
    env = overrides.get("env") or "kitchen"
    cfg = replace(Config(), env=env, **ENV_DEFAULTS[env])
    if quick:
        cfg = replace(cfg, **QUICK)
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def load_config(path: str) -> Config:
    import json
    d = json.load(open(path))
    known = Config.__dataclass_fields__
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in known})
