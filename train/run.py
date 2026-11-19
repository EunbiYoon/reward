"""Train the learner for every (seed, algo, reward, partner) of a grid and save the checkpoints.

Training only. To measure what a checkpoint learned (Shapley share, credit gap), run eval/run.py on the
folder afterwards, or pass --eval here to do it right after training.

Usage (from the project root):
    python -m train.run --algos ppo --rewards team --partners greedy --seeds 0 1 2 --out results/ppo/A
    python -m train.run --algos ppo grpo --rewards team --partners greedy --out A      # results/<algo>/A
    python -m train.run --algos ppo --rewards team theft pickup shadow \\
        --partners ckpt:results/ppo/A/models/ppo_team_greedy_s0.pt --seeds 0 1 2 --out results/ppo/A_personas
    python -m train.run --algos ppo --rewards team --partners self --out sp_A          # self-play partner (PPO)
    python -m train.run ... --eval                                                     # train, then eval.run
    python -m train.run --env foraging --agents 2 3 4 --quick --out results/lbf_n

--out: a bare name (no '/') is saved per algorithm as results/<algo>/<name>; a path is used as is.
Adding runs to an existing folder keeps the earlier ones (same run = replaced), so seeds can be added later.

Cluster (one run per Slurm array task, preemption-safe, see scripts/cluster/submit.sh):
    python -m train.run <grid args> --out results/x --list-jobs          # prints the number of runs N
    python -m train.run <grid args> --out results/x --job-index $SLURM_ARRAY_TASK_ID   # run one of 0..N-1
  A task skips a run whose checkpoint is already in <out>/parts/, resumes a half-trained run from
  <out>/resume/ (checkpoint every 5 min), and rewrites <out>/train_log.csv from all finished parts.

Writes to <out>/:
    config.json
    models/<algo>_<reward>_<partner>[_n<agents>][<variant>]_s<seed>.pt   (the policy + its settings)
    curves/*.csv    learning curve, one row per update (PPO) or group (GRPO)
    curves/curves.png  all curves of the folder in one figure, drawn after training (--no-plot to skip)
    train_log.csv   one row per run: steps, updates, last training return, seconds, checkpoint
Then:  python -m eval.run <out>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import get_context

from train.config import Config, make_config

ALGOS = ("ppo", "grpo")
REWARD_NAMES = ("learned", "team", "theft", "pickup", "cf", "prep", "shadow")
VARIANTS = (("pickup_bonus", "pickup_bonuses", "b"), ("partner_noise", "partner_noises", "e"),
            ("aux_coef", "aux_coefs", "w"))     # (Config field, its grid field, checkpoint tag)


# ------------------------------------------------------------------ shared with eval/run.py and scripts
def variant_cfgs(cfg: Config) -> list:
    """One Config per combination of the grid values (--pickup-bonus / --partner-noise / --aux-coef)."""
    from dataclasses import replace
    from itertools import product
    axes = [[(field, v) for v in getattr(cfg, grid)] for field, grid, _ in VARIANTS if getattr(cfg, grid)]
    return [replace(cfg, **dict(combo)) for combo in product(*axes)] if axes else [cfg]


def variant_tag(cfg: Config) -> str:
    """Checkpoint-name suffix for the grid dimensions in use, e.g. '_b3_e0.25'."""
    return "".join(f"_{t}{getattr(cfg, f):g}" for f, grid, t in VARIANTS if getattr(cfg, grid))


def make_env(cfg: Config, reward: str, partner: str, n_agents: int = 2):
    """reward 'learned' is the team environment with its training reward replaced by the reward model."""
    from environment import make
    extra = {} if cfg.partner_noise is None else {"noise": cfg.partner_noise}
    env = make(cfg.env, cfg.layout, "team" if reward == "learned" else reward, cfg.horizon, partner,
               anneal_steps=int(cfg.anneal_frac * cfg.env_steps), pickup_bonus=cfg.pickup_bonus,
               n_agents=n_agents, **extra)
    if reward == "learned":
        from judge.learned_env import LearnedReward
        env = LearnedReward(env, cfg.reward_model)
    return env


def partner_tag(partner: str) -> str:
    """File-safe partner name: 'ckpt:results/full/models/ppo_theft_greedy_s0.pt' -> 'ckpt-ppo_theft_greedy_s0'."""
    return "ckpt-" + os.path.basename(partner[5:]).removesuffix(".pt") if partner.startswith("ckpt:") else partner


def ckpt_name(algo: str, reward: str, partner: str, seed: int, n_agents: int = 2, tag: str = "") -> str:
    partner = partner_tag(partner)
    n = f"_n{n_agents}" if n_agents != 2 else ""     # 2-agent names unchanged, old runs still load
    return f"{algo}_{reward}_{partner}{n}{tag}_s{seed}.pt"


def write_csv(path: str, rows: list[dict]):
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = f"{path}.{os.getpid()}.tmp"            # atomic: array tasks may rewrite it concurrently
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def out_dirs(cfg: Config, out: str) -> list:
    """(cfg, folder) pairs: a bare --out name becomes results/<algo>/<name>, one folder per algorithm."""
    if "/" in out or os.sep in out:
        return [(cfg, out)]
    from dataclasses import replace
    return [(replace(cfg, algos=(a,)), os.path.join("results", a, out)) for a in cfg.algos]


# ------------------------------------------------------------------ training
def fit(env, algo: str, cfg: Config, seed: int, resume_path: str = "", curve_path: str = ""):
    if algo == "ppo":
        from train.ppo import PPOConfig, train_ppo
        pc = PPOConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                       rollout_steps=cfg.ppo_rollout_steps, epochs=cfg.ppo_epochs,
                       minibatch=cfg.ppo_minibatch, lr=cfg.ppo_lr, clip_eps=cfg.clip_eps,
                       vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
                       max_grad_norm=cfg.max_grad_norm, hidden=cfg.hidden, aux_coef=cfg.aux_coef,
                       resume_path=resume_path, curve_path=curve_path)
        return train_ppo(env, pc, seed)
    if algo == "grpo":
        from train.grpo import GRPODeepConfig, train_grpo_deep
        gc = GRPODeepConfig(env_steps=cfg.env_steps, gamma=cfg.gamma, group_size=cfg.grpo_group_size,
                            epochs=cfg.grpo_epochs, minibatch=cfg.grpo_minibatch, lr=cfg.grpo_lr,
                            clip_eps=cfg.clip_eps, max_grad_norm=cfg.max_grad_norm,
                            advantage=cfg.grpo_advantage, hidden=cfg.hidden, aux_coef=cfg.aux_coef,
                            resume_path=resume_path, curve_path=curve_path)
        return train_grpo_deep(env, gc, seed)
    raise ValueError(f"algos are {', '.join(ALGOS)}, got {algo!r}")


def run_job(args) -> dict:
    """Train one run and save its checkpoint. Returns its train_log.csv row."""
    seed, algo, reward, partner, n_agents, cfg, out = args
    import torch
    torch.set_num_threads(1)
    from train.deep_policy import save_checkpoint

    t0 = time.time()
    name = ckpt_name(algo, reward, partner, seed, n_agents, variant_tag(cfg))
    resume_path = ""
    if cfg.resume:
        os.makedirs(os.path.join(out, "resume"), exist_ok=True)
        resume_path = os.path.join(out, "resume", name)
    os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    curve_path = os.path.join(out, "curves", name.removesuffix(".pt") + ".csv")   # learning curve (PPO, GRPO)
    if not resume_path and os.path.exists(curve_path):
        os.remove(curve_path)                       # a fresh run starts a fresh curve
    env = make_env(cfg, reward, partner, n_agents)
    policy, obs_rms, log = fit(env, algo, cfg, seed, resume_path, curve_path)
    head = log.pop("_partner_head", None)

    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    ckpt = os.path.join(out, "models", name)
    save_checkpoint(ckpt, policy, obs_rms, dict(obs_dim=env.obs_dim, n_actions=env.n_actions,
                                                hidden=cfg.hidden, algo=algo, reward=reward,
                                                partner=partner, seed=seed, agents=n_agents,
                                                pickup_bonus=cfg.pickup_bonus, partner_noise=cfg.partner_noise,
                                                aux_coef=cfg.aux_coef))
    if head is not None:
        torch.save(head.state_dict(), ckpt[:-3] + "_head.pt")
    env.close()
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    return {"seed": seed, "algo": algo, "reward": reward, "partner": partner, "agents": n_agents,
            "pickup_bonus": cfg.pickup_bonus, "partner_noise": cfg.partner_noise, "aux_coef": cfg.aux_coef,
            **log, "train_seconds": round(time.time() - t0, 1), "checkpoint": os.path.relpath(ckpt, out)}


def preflight(cfg: Config):
    """Fail fast if PyTorch, the package, or the parts of it the wrapper uses are missing."""
    try:
        import torch  # noqa: F401
        from torch import nn  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"needs PyTorch: {e}")
    if "learned" in cfg.rewards and not cfg.reward_model:
        raise SystemExit("reward 'learned' needs --reward-model <run>/rm_<judge>.pt (see judge/). "
                         "Or drop it:  --rewards team theft")
    if cfg.reward_model and not os.path.exists(cfg.reward_model):
        raise SystemExit(f"--reward-model {cfg.reward_model} not found")
    if "self" in cfg.partners and any(a != "ppo" for a in cfg.algos):
        raise SystemExit("partner 'self' (self-play) is only wired into PPO: use --algos ppo")
    for n_agents in cfg.agents:
        for partner in cfg.partners:
            if partner.startswith("ckpt:") and not os.path.exists(partner[5:]):
                raise SystemExit(f"partner checkpoint not found: {partner[5:]}")
            try:
                for reward in cfg.rewards:
                    env = make_env(variant_cfgs(cfg)[-1], reward, partner, n_agents)
                    if partner == "self":         # self-play: an untrained policy stands in for the learner
                        from train.deep_policy import RunningMeanStd, make_policy
                        env.self_policy = (make_policy(env, cfg.hidden), RunningMeanStd(env.obs_dim))
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


def make_jobs(cfg: Config, out: str) -> list:
    return [(s, a, r, pt, n, c, out) for c in variant_cfgs(cfg) for s in cfg.seeds for a in cfg.algos
            for r in cfg.rewards for pt in cfg.partners for n in cfg.agents]


def earlier_rows(out: str, jobs: list) -> list[dict]:
    """Rows already in <out>/train_log.csv that this grid does not rerun (so seeds can be added later)."""
    path = os.path.join(out, "train_log.csv")
    new = {ckpt_name(a, r, pt, s, n, variant_tag(c)) for s, a, r, pt, n, c, _ in jobs}
    keep = [r for r in read_csv(path) if os.path.basename(r.get("checkpoint", "")) not in new]
    if keep:
        print(f"keeping {len(keep)} earlier runs from {path}")
    return keep


def write_config(out: str, cfg: Config):
    """config.json of the folder; when runs are added to an existing folder, seeds and partners become the union."""
    d = cfg.to_dict()
    path = os.path.join(out, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        for k in ("seeds", "partners", "rewards"):
            merged = list(old.get(k) or [])
            merged += [v for v in d.get(k) or [] if v not in merged]
            d[k] = sorted(merged) if k == "seeds" else merged
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def draw_curves(out: str):
    """Draw <out>/curves/curves.png from the learning-curve CSVs (scripts/plot_curves.py).
    A plotting problem is reported but never stops a run: the checkpoints are already saved."""
    try:
        from scripts.plot_curves import draw
        print(f"learning curves -> {draw([out])}")
    except (Exception, SystemExit) as e:
        print(f"could not draw the learning curves ({type(e).__name__}: {e}); "
              f"run later:  python scripts/plot_curves.py {out}")


def run_one(cfg: Config, out: str, index: int, evaluate: bool, plot: bool = True):
    """Slurm-array mode: train job `index` of the grid (skipped if already finished), then merge parts."""
    from dataclasses import replace
    jobs = make_jobs(replace(cfg, resume=True), out)
    if not 0 <= index < len(jobs):
        raise SystemExit(f"--job-index {index} out of range: this grid has {len(jobs)} runs (0..{len(jobs) - 1})")
    s, a, r, pt, n, c, _ = jobs[index]
    name = ckpt_name(a, r, pt, s, n, variant_tag(c))
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    part = os.path.join(parts, name[:-3] + ".json")
    if not os.path.exists(os.path.join(out, "config.json")):
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    if os.path.exists(part):
        print(f"job {index} ({name}) already trained: {part}")
    else:
        print(f"job {index}/{len(jobs)}: {name}", flush=True)
        row = run_job(jobs[index])
        with open(part + ".tmp", "w") as f:
            json.dump(row, f)
        os.replace(part + ".tmp", part)
        print(f"  trained in {row['train_seconds']:.0f}s -> {row['checkpoint']}", flush=True)
    rows = []
    for fn in sorted(os.listdir(parts)):
        if fn.endswith(".json") and not fn.endswith(".eval.json"):
            with open(os.path.join(parts, fn)) as f:
                rows.append(json.load(f))
    write_csv(os.path.join(out, "train_log.csv"), rows)
    print(f"{len(rows)}/{len(jobs)} runs trained in {out}")
    if plot and len(rows) >= len(jobs):      # the last array task to finish draws the graph once
        draw_curves(out)
    if evaluate:
        from eval.run import evaluate_part
        evaluate_part(out, name)


def run_grid(cfg: Config, out: str, workers: int, evaluate: bool, plot: bool = True):
    preflight(cfg)
    os.makedirs(out, exist_ok=True)
    write_config(out, cfg)
    jobs = make_jobs(cfg, out)
    print(f"{cfg.env} {cfg.layout}: rewards={list(cfg.rewards)} partners={list(cfg.partners)} "
          f"agents={list(cfg.agents)} algos={list(cfg.algos)} seeds={list(cfg.seeds)} "
          f"steps/run={cfg.env_steps:,}")
    print(f"{len(jobs)} runs on {workers} workers -> {out}")
    earlier = earlier_rows(out, jobs)
    rows, t0 = [], time.time()

    def report(i, row):
        rows.append(row)
        write_csv(os.path.join(out, "train_log.csv"), earlier + rows)   # after every finished run
        tag = "".join(f" {f}={row[f]:g}" for f, grid, _ in VARIANTS if getattr(cfg, grid))
        print(f"  [{i}/{len(jobs)}] seed={row['seed']} {row['algo']:>4} {row['reward']:<10}{tag} "
              f"partner={partner_tag(row['partner']):<6} n={row['agents']} "
              f"train_return={row['train_proxy_return_last5']:.1f} {row['train_seconds']:.0f}s "
              f"-> {row['checkpoint']}", flush=True)

    if workers > 1:
        ctx = get_context("spawn" if cfg.env == "alem" else None)     # JAX is not fork-safe
        with ctx.Pool(min(workers, len(jobs))) as pool:
            for i, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
                report(i, row)
    else:
        for i, job in enumerate(jobs, 1):
            report(i, run_job(job))
    print(f"trained in {time.time() - t0:.0f}s.")
    if plot:
        draw_curves(out)
    if evaluate:
        from eval.run import evaluate_run
        evaluate_run(out, workers, names=[os.path.basename(r["checkpoint"]) for r in rows])
    else:
        print(f"Next:  python -m eval.run {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="1 seed, 200k steps, 3 eval episodes")
    p.add_argument("--out", default="run",
                   help="a bare name (no '/') is saved per algorithm as results/<algo>/<name>; a path is used as is")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--algos", nargs="+", choices=[*ALGOS, "all"], help="'all' = " + " ".join(ALGOS))
    p.add_argument("--agents", type=int, nargs="+", choices=[2, 3, 4],
                   help="number of agents (learner + identical partners); 3 and 4 need warehouse, foraging or alem")
    p.add_argument("--rewards", nargs="+",
                   help=f"{' '.join(REWARD_NAMES)} or hsp_<spec> (HSP event reward, kitchen only, see hsp/); "
                        "cf = counterfactual contribution reward (H10, kitchen only)")
    p.add_argument("--rewards-file", help="read the rewards from this file, one per line (e.g. hsp/weights.py output)")
    p.add_argument("--reward-model", help="rm_<judge>.pt from python -m judge.reward_model (reward 'learned')")
    p.add_argument("--partners", nargs="+",
                   help="stay, greedy, self (self-play, PPO only) or ckpt:<path> (a frozen trained partner)")
    p.add_argument("--env", choices=["kitchen", "warehouse", "foraging", "alem"],
                   help="kitchen = Overcooked-AI (default), warehouse = RWARE, foraging = LBF, alem = Alem")
    p.add_argument("--layout", help="default per env: cramped_room / tiny / 8x8-3f / overworld-easy")
    p.add_argument("--env-steps", type=int)
    p.add_argument("--pickup-bonus", type=float, nargs="+",
                   help="pickup bonus per pickup; several values = a dose-response grid (H7)")
    p.add_argument("--partner-noise", type=float, nargs="+",
                   help="kitchen: P(the partner takes a random action); several values = a grid (H8)")
    p.add_argument("--aux-coef", type=float, nargs="+",
                   help="PPO/GRPO partner-prediction head weight (0 = no head); several values = a grid (H12)")
    p.add_argument("--eval", action="store_true", help="measure the new checkpoints right after training (eval.run)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip drawing <out>/curves/curves.png after training (drawn by default)")
    p.add_argument("--list-jobs", action="store_true", help="print the number of runs in this grid and exit")
    p.add_argument("--job-index", type=int, help="run only this run of the grid (Slurm array task id)")
    a = p.parse_args()
    if a.rewards_file:
        with open(a.rewards_file) as f:
            a.rewards = (a.rewards or []) + [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for r in a.rewards or []:
        if r not in REWARD_NAMES and not r.startswith("hsp_"):
            p.error(f"unknown reward {r!r}: choose from {', '.join(REWARD_NAMES)} or hsp_<spec>")
    t = lambda x: tuple(x) if x else None
    if a.algos and "all" in a.algos:
        a.algos = list(ALGOS)
    cfg = make_config(a.quick, seeds=t(a.seeds), algos=t(a.algos), agents=t(a.agents), rewards=t(a.rewards),
                      partners=t(a.partners), env=a.env, layout=a.layout, env_steps=a.env_steps,
                      reward_model=a.reward_model)
    from dataclasses import replace
    for vals, field, grid in ((a.pickup_bonus, "pickup_bonus", "pickup_bonuses"),
                              (a.partner_noise, "partner_noise", "partner_noises"),
                              (a.aux_coef, "aux_coef", "aux_coefs")):
        if vals:
            cfg = replace(cfg, **{field: vals[0]}, **({grid: tuple(vals)} if len(vals) > 1 else {}))
    return cfg, a


def main():
    cfg, a = parse_args()
    for c, out in out_dirs(cfg, a.out):
        if a.list_jobs:
            print(len(make_jobs(c, out)))
        elif a.job_index is not None:
            os.makedirs(out, exist_ok=True)
            run_one(c, out, a.job_index, a.eval, plot=not a.no_plot)
        else:
            run_grid(c, out, a.workers, a.eval, plot=not a.no_plot)


if __name__ == "__main__":
    main()
