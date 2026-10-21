"""Train every (seed, algo, reward, partner) and measure the learner's Shapley contribution.

Usage (from the project root):
    python -m eval.run --quick --workers 4 --out results/quick      # smoke test
    python -m eval.run --workers 4 --out results/full               # the proposal's grid
    python -m eval.run --algos ppo --rewards team individual --partners greedy --out results/x
    python -m eval.run --algos all --quick --out results/algos_quick    # all 5 optimizers
    python -m eval.run --env foraging --agents 2 3 4 --quick --out results/lbf_n   # 2, 3, 4 agents
    python -m eval.run --env warehouse --quick --out results/rware_quick       # RWARE, layout tiny
    python -m eval.run --env foraging --quick --out results/lbf_quick          # LBF, layout 8x8-3f

Writes to <out>/:
    config.json
    reference.csv   the scripted greedy cook in the learner's seat, per partner (s*, pickups*)
    results.csv     one row per training run (rewritten after every finished run)
    models/<algo>_<reward>_<partner>[_n<agents>]_s<seed>.pt   (_n only for 3 or 4 agents)
Then:  python -m eval.summarize <out>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

from .coalitions import evaluate_coalitions, policy_actor, torch_reseed
from .config import Config, make_config


def make_env(cfg: Config, reward: str, partner: str, n_agents: int = 2):
    from environment import make
    return make(cfg.env, cfg.layout, reward, cfg.horizon, partner,
                anneal_steps=int(cfg.anneal_frac * cfg.env_steps), pickup_bonus=cfg.pickup_bonus,
                n_agents=n_agents)


ALGOS = ("ppo", "grpo", "a2c", "dqn", "sac")


def ckpt_name(algo: str, reward: str, partner: str, seed: int, n_agents: int = 2) -> str:
    n = f"_n{n_agents}" if n_agents != 2 else ""     # 2-agent names unchanged, old runs still load
    return f"{algo}_{reward}_{partner}{n}_s{seed}.pt"


def train(env, algo: str, cfg: Config, seed: int):
    if algo == "ppo":
        from train.ppo import PPOConfig, train_ppo
        pc = PPOConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                       rollout_steps=cfg.ppo_rollout_steps, epochs=cfg.ppo_epochs,
                       minibatch=cfg.ppo_minibatch, lr=cfg.ppo_lr, clip_eps=cfg.clip_eps,
                       vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                       max_grad_norm=cfg.max_grad_norm, hidden=cfg.hidden)
        return train_ppo(env, pc, seed)
    if algo == "grpo":
        from train.grpo import GRPODeepConfig, train_grpo_deep
        gc = GRPODeepConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, group_size=cfg.grpo_group_size,
                            epochs=cfg.grpo_epochs, minibatch=cfg.grpo_minibatch, lr=cfg.grpo_lr,
                            clip_eps=cfg.clip_eps, max_grad_norm=cfg.max_grad_norm,
                            advantage=cfg.grpo_advantage, hidden=cfg.hidden)
        return train_grpo_deep(env, gc, seed)
    if algo == "a2c":
        from train.a2c import A2CConfig, train_a2c
        ac = A2CConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                       rollout_steps=cfg.a2c_rollout_steps, lr=cfg.a2c_lr, vf_coef=cfg.vf_coef,
                       ent_coef=cfg.ent_coef, max_grad_norm=cfg.max_grad_norm, hidden=cfg.hidden)
        return train_a2c(env, ac, seed)
    if algo == "dqn":
        from train.dqn import DQNConfig, train_dqn
        dc = DQNConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, buffer_size=cfg.dqn_buffer,
                       batch=cfg.dqn_batch, lr=cfg.dqn_lr, learning_starts=cfg.dqn_learning_starts,
                       train_every=cfg.dqn_train_every, target_every=cfg.dqn_target_every,
                       explore_frac=cfg.dqn_explore_frac, eps_end=cfg.dqn_eps_end, hidden=cfg.hidden)
        return train_dqn(env, dc, seed)
    if algo == "sac":
        from train.sac import SACConfig, train_sac
        sc = SACConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, buffer_size=cfg.sac_buffer,
                       batch=cfg.sac_batch, lr=cfg.sac_lr, learning_starts=cfg.sac_learning_starts,
                       train_every=cfg.sac_train_every, tau=cfg.sac_tau,
                       target_entropy_scale=cfg.sac_target_entropy_scale, hidden=cfg.hidden)
        return train_sac(env, sc, seed)
    raise ValueError(f"algos are {', '.join(ALGOS)}, got {algo!r}")


def run_job(args):
    seed, algo, reward, partner, n_agents, cfg, out = args
    import torch
    torch.set_num_threads(1)
    from train.deep_policy import save_checkpoint

    t0 = time.time()
    env = make_env(cfg, reward, partner, n_agents)
    policy, obs_rms, log = train(env, algo, cfg, seed)
    train_s = time.time() - t0

    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    ckpt = os.path.join(out, "models", ckpt_name(algo, reward, partner, seed, n_agents))
    save_checkpoint(ckpt, policy, obs_rms, dict(obs_dim=env.obs_dim, n_actions=env.n_actions,
                                                hidden=cfg.hidden, algo=algo, reward=reward,
                                                partner=partner, seed=seed, agents=n_agents))
    actor = policy_actor(policy, obs_rms, cfg.eval_deterministic)
    met = evaluate_coalitions(env, actor, partner, cfg.eval_episodes, before_episode=torch_reseed)
    env.close()
    return {"seed": seed, "algo": algo, "reward": reward, "partner": partner, **met, **log,
            "train_seconds": train_s, "checkpoint": os.path.relpath(ckpt, out)}


def reference(cfg: Config) -> list[dict]:
    """The scripted greedy cook in the learner's seat: what a good teammate looks like."""
    rows = []
    for n_agents in cfg.agents:
        for partner in cfg.partners:
            env = make_env(cfg, "team", partner, n_agents)
            met = evaluate_coalitions(env, "greedy", partner, cfg.eval_episodes)
            rows.append({"partner": partner, **met})
            print(f"  reference (greedy cook in the learner seat), agents={n_agents} partner={partner}: "
                  f"team soups={met['soups']:.2f}  s*={met['s']:.2f}  pickups*={met['pickups']:.1f}")
    return rows


def preflight(cfg: Config):
    """Fail fast if PyTorch, the package, or the parts of it the wrapper uses are missing."""
    try:
        import torch  # noqa: F401
        from torch import nn  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"needs PyTorch: {e}")
    for n_agents in cfg.agents:
        for partner in cfg.partners:
            try:
                env = make_env(cfg, cfg.rewards[0], partner, n_agents)
                env.reset(0)
                for a in range(env.n_actions):
                    env.step(a)
            except ImportError as e:
                raise SystemExit(str(e))
            except ValueError as e:
                raise SystemExit(f"{cfg.env}: {e}")
            except Exception as e:
                raise SystemExit(f"the {cfg.env} package is installed but the wrapper could not use it "
                                 f"(agents={n_agents} partner={partner}): {type(e).__name__}: {e}")
    print(f"preflight ok: obs_dim={env.obs_dim} n_actions={env.n_actions}")


def write_csv(path: str, rows: list[dict]):
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="1 seed, 200k steps, 3 eval episodes")
    p.add_argument("--out", default="results/run")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--algos", nargs="+", choices=[*ALGOS, "all"], help="'all' = " + " ".join(ALGOS))
    p.add_argument("--agents", type=int, nargs="+", choices=[2, 3, 4],
                   help="number of agents (learner + identical partners); 3 and 4 need warehouse or foraging")
    p.add_argument("--rewards", nargs="+", choices=["pickup", "team", "individual"])
    p.add_argument("--partners", nargs="+", choices=["stay", "greedy"])
    p.add_argument("--env", choices=["kitchen", "warehouse", "foraging"],
                   help="kitchen = Overcooked-AI (default), warehouse = RWARE, foraging = LBF")
    p.add_argument("--layout", help="default per env: cramped_room / tiny / 8x8-3f")
    p.add_argument("--env-steps", type=int)
    a = p.parse_args()
    t = lambda x: tuple(x) if x else None
    if a.algos and "all" in a.algos:
        a.algos = list(ALGOS)
    cfg = make_config(a.quick, seeds=t(a.seeds), algos=t(a.algos), agents=t(a.agents), rewards=t(a.rewards),
                      partners=t(a.partners), env=a.env, layout=a.layout, env_steps=a.env_steps)
    return cfg, a.out, a.workers


def main():
    cfg, out, workers = parse_args()
    preflight(cfg)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    write_csv(os.path.join(out, "reference.csv"), reference(cfg))

    jobs = [(s, a, r, pt, n, cfg, out) for s in cfg.seeds for a in cfg.algos for r in cfg.rewards
            for pt in cfg.partners for n in cfg.agents]
    print(f"{cfg.env} {cfg.layout}: rewards={list(cfg.rewards)} partners={list(cfg.partners)} "
          f"agents={list(cfg.agents)} algos={list(cfg.algos)} seeds={list(cfg.seeds)} "
          f"steps/run={cfg.env_steps:,}")
    print(f"{len(jobs)} runs on {workers} workers -> {out}")
    rows, t0 = [], time.time()

    def report(i, row):
        rows.append(row)
        write_csv(os.path.join(out, "results.csv"), rows)       # partial results after every run
        print(f"  [{i}/{len(jobs)}] seed={row['seed']} {row['algo']:>4} {row['reward']:<10} "
              f"partner={row['partner']:<6} n={row['agents']} team={row['soups']:.2f} own={row['own_soups']:.2f} "
              f"s={row['s']:+.2f} gap={row['credit_gap']:+.2f} pickups={row['pickups']:.1f} "
              f"{row['train_seconds']:.0f}s", flush=True)

    if workers > 1:
        with Pool(min(workers, len(jobs))) as pool:
            for i, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
                report(i, row)
    else:
        for i, job in enumerate(jobs, 1):
            report(i, run_job(job))
    print(f"done in {time.time() - t0:.0f}s. Next:  python -m eval.summarize {out}")


if __name__ == "__main__":
    main()
