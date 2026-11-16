"""Train every (seed, algo, reward, partner) and measure the learner's Shapley contribution.

Rewards: learned (the RLAIF reward model, --reward-model rm_<judge>.pt from rlaif/), team,
individual, and pickup (a hand-made fake-help bonus, kept as a control).

Usage (from the project root):
    python -m eval.run --reward-model results/rlaif/kitchen/rm_claude.pt --workers 4 --out results/rlaif_kitchen
    python -m eval.run --rewards team individual pickup --quick --workers 4 --out results/quick   # no RM needed
    python -m eval.run --workers 4 --out results/full               # the proposal's grid
    python -m eval.run --algos ppo --rewards team individual --partners greedy --out results/x
    python -m eval.run --algos all --quick --out results/algos_quick    # all 5 optimizers
    python -m eval.run --env foraging --agents 2 3 4 --quick --out results/lbf_n   # 2, 3, 4 agents
    python -m eval.run --env warehouse --quick --out results/rware_quick       # RWARE, layout tiny
    python -m eval.run --env foraging --quick --out results/lbf_quick          # LBF, layout 8x8-3f
    python -m eval.run --env alem --rewards team individual pickup --quick --workers 1 --out results/alem_quick

Cluster (one run per Slurm array task, preemption-safe, see scripts/cluster/submit.sh):
    python -m eval.run <grid args> --out results/x --list-jobs          # prints the number of runs N
    python -m eval.run <grid args> --out results/x --job-index $SLURM_ARRAY_TASK_ID   # run one of 0..N-1
  A task skips a run whose result is already in <out>/parts/, resumes a half-trained run from
  <out>/resume/ (checkpoint every 5 min), and rewrites <out>/results.csv from all finished parts.

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
from multiprocessing import get_context

from .coalitions import evaluate_coalitions, policy_actor, torch_reseed
from .config import Config, make_config


VARIANTS = (("pickup_bonus", "pickup_bonuses", "b"), ("partner_noise", "partner_noises", "e"),
            ("aux_coef", "aux_coefs", "w"))     # (Config field, its grid field, checkpoint tag)


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
        from rlaif.learned_env import LearnedReward
        env = LearnedReward(env, cfg.reward_model)
    return env


def rm_score(cfg: Config, traces: list) -> float:
    """Mean learned-reward score of the evaluation episodes (nan without a reward model)."""
    if not cfg.reward_model or not traces:
        return float("nan")
    import numpy as np
    from rlaif.reward_model import load_rm, score
    rm = load_rm(cfg.reward_model)
    return float(np.mean([score(rm, np.stack([s["obs"] for s in tr]), np.array([s["a"][0] for s in tr]))
                          for tr in traces]))


ALGOS = ("ppo", "grpo")


def partner_tag(partner: str) -> str:
    """File-safe partner name: 'ckpt:results/full/models/ppo_individual_greedy_s0.pt' -> 'ckpt-ppo_individual_greedy_s0'."""
    return "ckpt-" + os.path.basename(partner[5:]).removesuffix(".pt") if partner.startswith("ckpt:") else partner


def ckpt_name(algo: str, reward: str, partner: str, seed: int, n_agents: int = 2, tag: str = "") -> str:
    partner = partner_tag(partner)
    n = f"_n{n_agents}" if n_agents != 2 else ""     # 2-agent names unchanged, old runs still load
    return f"{algo}_{reward}_{partner}{n}{tag}_s{seed}.pt"


def train(env, algo: str, cfg: Config, seed: int, resume_path: str = "", curve_path: str = ""):
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
                            resume_path=resume_path)
        return train_grpo_deep(env, gc, seed)
    raise ValueError(f"algos are {', '.join(ALGOS)}, got {algo!r}")


def run_job(args):
    seed, algo, reward, partner, n_agents, cfg, out = args
    import torch
    torch.set_num_threads(1)
    from train.deep_policy import save_checkpoint

    t0 = time.time()
    name = ckpt_name(algo, reward, partner, seed, n_agents, variant_tag(cfg))
    resume_path = ""
    if cfg.resume:
        if algo not in ("ppo", "grpo"):
            print(f"  note: {algo} has no mid-training checkpoint; a preempted run restarts", flush=True)
        else:
            os.makedirs(os.path.join(out, "resume"), exist_ok=True)
            resume_path = os.path.join(out, "resume", name)
    os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    curve_path = os.path.join(out, "curves", name.removesuffix(".pt") + ".csv")   # PPO learning curve
    if not resume_path and os.path.exists(curve_path):
        os.remove(curve_path)                       # a fresh run starts a fresh curve
    env = make_env(cfg, reward, partner, n_agents)
    policy, obs_rms, log = train(env, algo, cfg, seed, resume_path, curve_path)
    head = log.pop("_partner_head", None)
    train_s = time.time() - t0

    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    ckpt = os.path.join(out, "models", name)
    save_checkpoint(ckpt, policy, obs_rms, dict(obs_dim=env.obs_dim, n_actions=env.n_actions,
                                                hidden=cfg.hidden, algo=algo, reward=reward,
                                                partner=partner, seed=seed, agents=n_agents,
                                                pickup_bonus=cfg.pickup_bonus, partner_noise=cfg.partner_noise,
                                                aux_coef=cfg.aux_coef))
    if head is not None:
        import torch
        torch.save(head.state_dict(), ckpt[:-3] + "_head.pt")
    actor = policy_actor(policy, obs_rms, cfg.eval_deterministic)
    traces: list = []
    met = evaluate_coalitions(env, actor, partner, cfg.eval_episodes, before_episode=torch_reseed,
                              traces=traces if (cfg.reward_model or head is not None) else None)
    met["rm_score"] = rm_score(cfg, traces)        # what the reward model thinks of the trained policy
    met["partner_pred_acc"] = partner_pred_acc(policy, obs_rms, head, traces)
    env.close()
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    return {"seed": seed, "algo": algo, "reward": reward, "partner": partner,
            "pickup_bonus": cfg.pickup_bonus, "partner_noise": cfg.partner_noise, "aux_coef": cfg.aux_coef,
            **met, **log, "train_seconds": train_s, "checkpoint": os.path.relpath(ckpt, out)}


def partner_pred_acc(policy, obs_rms, head, traces: list) -> float:
    """Held-out accuracy of the partner-prediction head on the evaluation episodes (nan without a head)."""
    if head is None or not traces:
        return float("nan")
    import numpy as np
    import torch
    steps = [s for tr in traces for s in tr]
    obs = torch.as_tensor(obs_rms.normalize(np.stack([s["obs"] for s in steps])))
    with torch.no_grad():
        pred = head(policy, obs).argmax(-1).numpy()
    return float(np.mean(pred == np.array([s["a"][1] for s in steps])))


def reference(cfg: Config) -> list[dict]:
    """The scripted greedy cook in the learner's seat: what a good teammate looks like.
    One row per partner noise level, since a clumsier partner changes the reference."""
    from dataclasses import replace
    rows = []
    for noise in cfg.partner_noises or (cfg.partner_noise,):
        c = replace(cfg, partner_noise=noise)
        for n_agents in cfg.agents:
            for partner in cfg.partners:
                if partner == "self":         # no fixed partner to put the scripted cook next to
                    continue
                env = make_env(c, "team", partner, n_agents)
                traces: list = []
                met = evaluate_coalitions(env, "greedy", partner, cfg.eval_episodes,
                                          traces=traces if cfg.reward_model else None)
                rows.append({"partner": partner, "partner_noise": noise, **met, "rm_score": rm_score(cfg, traces)})
                print(f"  reference (greedy cook in the learner seat), agents={n_agents} partner={partner} "
                      f"noise={noise}: team soups={met['soups']:.2f}  s*={met['s']:.2f}  pickups*={met['pickups']:.1f}")
    return rows


def preflight(cfg: Config):
    """Fail fast if PyTorch, the package, or the parts of it the wrapper uses are missing."""
    try:
        import torch  # noqa: F401
        from torch import nn  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"needs PyTorch: {e}")
    if "learned" in cfg.rewards and not cfg.reward_model:
        raise SystemExit("reward 'learned' needs --reward-model <run>/rm_<judge>.pt (see rlaif/). "
                         "Or drop it:  --rewards team individual")
    if cfg.reward_model and not os.path.exists(cfg.reward_model):
        raise SystemExit(f"--reward-model {cfg.reward_model} not found")
    for n_agents in cfg.agents:
        for partner in cfg.partners:
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


def make_jobs(cfg: Config, out: str) -> list:
    return [(s, a, r, pt, n, c, out) for c in variant_cfgs(cfg) for s in cfg.seeds for a in cfg.algos
            for r in cfg.rewards for pt in cfg.partners for n in cfg.agents]


def run_one(cfg: Config, out: str, index: int):
    """Slurm-array mode: run job `index` of the grid (skipped if already finished), then merge parts."""
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
    if not os.path.exists(os.path.join(out, "reference.csv")):
        write_csv(os.path.join(out, "reference.csv"), reference(cfg))
    if os.path.exists(part):
        print(f"job {index} ({name}) already finished: {part}")
    else:
        print(f"job {index}/{len(jobs)}: {name}", flush=True)
        row = run_job(jobs[index])
        with open(part + ".tmp", "w") as f:
            json.dump(row, f)
        os.replace(part + ".tmp", part)
        print(f"  team={row['soups']:.2f} s={row['s']:+.2f} gap={row['credit_gap']:+.2f} "
              f"{row['train_seconds']:.0f}s -> {part}", flush=True)
    rows = []
    for fn in sorted(os.listdir(parts)):
        if fn.endswith(".json"):
            with open(os.path.join(parts, fn)) as f:
                rows.append(json.load(f))
    write_csv(os.path.join(out, "results.csv"), rows)
    print(f"{len(rows)}/{len(jobs)} runs finished in {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="1 seed, 200k steps, 3 eval episodes")
    p.add_argument("--out", default="run",
                   help="a bare name (no '/') is saved per algorithm as results/<algo>/<name>; a path is used as is. "
                        "Runs added to an existing folder are merged into its results.csv (same run = replaced)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--algos", nargs="+", choices=[*ALGOS, "all"], help="'all' = " + " ".join(ALGOS))
    p.add_argument("--agents", type=int, nargs="+", choices=[2, 3, 4],
                   help="number of agents (learner + identical partners); 3 and 4 need warehouse, foraging or alem")
    p.add_argument("--rewards", nargs="+", choices=["learned", "team", "individual", "pickup", "cf", "prep", "shadow"],
                   help="cf = counterfactual contribution reward (H10, kitchen only)")
    p.add_argument("--reward-model", help="rm_<judge>.pt from python -m rlaif.reward_model (reward 'learned')")
    p.add_argument("--partners", nargs="+", help="stay, greedy, or ckpt:<path> (a frozen trained partner)")
    p.add_argument("--env", choices=["kitchen", "warehouse", "foraging", "alem"],
                   help="kitchen = Overcooked-AI (default), warehouse = RWARE, foraging = LBF, alem = Alem")
    p.add_argument("--layout", help="default per env: cramped_room / tiny / 8x8-3f / overworld-easy")
    p.add_argument("--env-steps", type=int)
    p.add_argument("--pickup-bonus", type=float, nargs="+",
                   help="pickup bonus per pickup; several values = a dose-response grid (H7)")
    p.add_argument("--partner-noise", type=float, nargs="+",
                   help="kitchen: P(the partner takes a random action); several values = a grid (H8)")
    p.add_argument("--list-jobs", action="store_true", help="print the number of runs in this grid and exit")
    p.add_argument("--job-index", type=int, help="run only this run of the grid (Slurm array task id)")
    p.add_argument("--aux-coef", type=float, nargs="+",
                   help="PPO/GRPO partner-prediction head weight (0 = no head); several values = a grid (H12)")
    a = p.parse_args()
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
    return cfg, a.out, a.workers, a.list_jobs, a.job_index


def previous_rows(out: str, jobs: list) -> list[dict]:
    """Rows already in <out>/results.csv that this grid does not rerun (so extra seeds can be added later)."""
    path = os.path.join(out, "results.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        old = list(csv.DictReader(f))
    # a checkpoint partner is matched by its file name, so a moved results folder still counts as the same partner
    tag = {partner_tag(pt): pt for *_, pt, n, c, _o in jobs}
    for r in old:
        r["partner"] = tag.get(partner_tag(r.get("partner", "")), r.get("partner"))
    new = {(str(s), a, r, pt, str(n)) for s, a, r, pt, n, c, _ in jobs}
    keep = [r for r in old if (r.get("seed"), r.get("algo"), r.get("reward"), r.get("partner"), r.get("agents")) not in new]
    if keep:
        print(f"keeping {len(keep)} earlier runs from {path}")
    return keep


def write_config(out: str, cfg: Config):
    """config.json of the folder; when runs are added to an existing folder, its seeds become the union."""
    d = cfg.to_dict()
    path = os.path.join(out, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        d["seeds"] = sorted(set(old.get("seeds", [])) | set(d.get("seeds", [])))
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def main():
    cfg, out, workers, list_jobs, job_index = parse_args()
    if "/" in out or os.sep in out:
        outs = [(cfg, out)]
    else:                                   # bare name: one folder per algorithm, results/<algo>/<name>
        from dataclasses import replace
        outs = [(replace(cfg, algos=(a,)), os.path.join("results", a, out)) for a in cfg.algos]
    for c, o in outs:
        run_grid(c, o, workers, list_jobs, job_index)


def run_grid(cfg: Config, out: str, workers: int, list_jobs: bool, job_index):
    if list_jobs:
        print(len(make_jobs(cfg, out)))
        return
    if job_index is not None:
        os.makedirs(out, exist_ok=True)
        run_one(cfg, out, job_index)
        return
    preflight(cfg)
    os.makedirs(out, exist_ok=True)
    write_config(out, cfg)
    write_csv(os.path.join(out, "reference.csv"), reference(cfg))

    jobs = make_jobs(cfg, out)
    print(f"{cfg.env} {cfg.layout}: rewards={list(cfg.rewards)} partners={list(cfg.partners)} "
          f"agents={list(cfg.agents)} algos={list(cfg.algos)} seeds={list(cfg.seeds)} "
          f"steps/run={cfg.env_steps:,}")
    print(f"{len(jobs)} runs on {workers} workers -> {out}")
    earlier = previous_rows(out, jobs)
    rows, t0 = [], time.time()

    def report(i, row):
        rows.append(row)
        write_csv(os.path.join(out, "results.csv"), earlier + rows)   # partial results after every run
        tag = "".join(f" {f}={row[f]:g}" for f, grid, _ in VARIANTS if getattr(cfg, grid))
        print(f"  [{i}/{len(jobs)}] seed={row['seed']} {row['algo']:>4} {row['reward']:<10}{tag} "
              f"partner={row['partner']:<6} n={row['agents']} team={row['soups']:.2f} own={row['own_soups']:.2f} "
              f"s={row['s']:+.2f} gap={row['credit_gap']:+.2f} pickups={row['pickups']:.1f} "
              f"rm={row['rm_score']:+.2f} "
              f"{row['train_seconds']:.0f}s", flush=True)

    if workers > 1:
        ctx = get_context("spawn" if cfg.env == "alem" else None)     # JAX is not fork-safe
        with ctx.Pool(min(workers, len(jobs))) as pool:
            for i, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
                report(i, row)
    else:
        for i, job in enumerate(jobs, 1):
            report(i, run_job(job))
    print(f"done in {time.time() - t0:.0f}s. Next:  python -m eval.summarize {out}")


if __name__ == "__main__":
    main()
