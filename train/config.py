"""Experiment configuration (matches the proposal). Override from the CLI in train/run.py."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from environment import canon


@dataclass
class Config:
    env: str = "overcooked"           # overcooked (Overcooked-AI) | rware (multi-robot warehouse)
    layout: str = "cramped_room"
    rewards: tuple = ("learned", "team", "theft")   # + "pickup" (fake-help bonus), "cf" (counterfactual, H10)
    reward_model: str = ""            # rm_<judge>.pt from judge/reward_model.py, needed by "learned"
    partners: tuple = ("stay", "greedy")
    algos: tuple = ("ppo", "grpo")
    agents: tuple = (2,)              # number of agents (learner + identical partners); 3, 4 need rware
    seeds: tuple = (0, 1, 2)
    env_steps: int = 2_000_000        # per run, same budget for PPO and GRPO
    horizon: int = 400
    anneal_frac: float = 0.5          # official shaping reaches 0 at this fraction of env_steps
    pickup_bonus: float = 3.0
    partner_noise: float | None = None  # overcooked: P(partner takes a random action) (H8); None = env default
    aux_coef: float = 0.0             # PPO/GRPO partner-action prediction head weight (H12); 0 = no head
    # Grid dimensions: every value is its own run (empty = just the single value above).
    pickup_bonuses: tuple = ()        # H7 dose-response
    partner_noises: tuple = ()        # H8 partner competence
    aux_coefs: tuple = ()             # H12 partner-modeling strength
    resume: bool = False              # --job-index mode: mid-training checkpoints in <out>/resume/
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

    # evaluation criteria (proposal Sec. 5)
    h1_s_max: float = 0.1             # pickup persona: s <= 0.1 and more pickups than the reference
    h1_frac: float = 0.5              # ... in >= half the runs
    h3_gap: float = 0.2               # credit theft: own share of soups - s >= reference gap + 0.2

    def to_dict(self) -> dict:
        return asdict(self)


QUICK = dict(seeds=(0,), env_steps=200_000, eval_episodes=3)

# Per-environment defaults, applied before --quick and the CLI. The pickup bonus is about the
# same fraction of one unit of team reward as in Overcooked (3 / 20 per soup).
ENV_DEFAULTS = {
    "overcooked": {},
    "rware": dict(layout="tiny", horizon=500, pickup_bonus=0.15),     # 1 per delivery
}


def make_config(quick: bool = False, **overrides) -> Config:
    env = canon(overrides.get("env") or "overcooked")
    cfg = replace(Config(), env=env, **ENV_DEFAULTS[env])
    if quick:
        cfg = replace(cfg, **QUICK)
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def load_config(path: str) -> Config:
    import json
    d = json.load(open(path))
    known = Config.__dataclass_fields__
    if "env" in d:
        d["env"] = canon(d["env"])          # kitchen / warehouse in older folders
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in known})
