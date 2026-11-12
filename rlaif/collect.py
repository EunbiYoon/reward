"""Step 1 of the RLAIF pipeline: a pool of trajectories with Shapley ground truth.

For every partner and pool seed, each seat-0 behaviour (rlaif/behaviors.py) plays the same
episode. The real game (everyone moves) is recorded step by step; the counterfactual coalitions
of eval/coalitions.py are played on the same seed, so every trajectory carries its agent 0's
Shapley contribution phi and share s. The judge never sees those numbers; the audit does.

    python -m rlaif.collect --env kitchen --episodes 30 --workers 4 --out results/rlaif/kitchen
    python -m rlaif.collect --env kitchen --policies "results/full/models/*_greedy_s0.pt" --out ...

Writes to <out>/pool/:
    meta.jsonl         one record per trajectory: id, behaviour, partner, seed, Shapley metrics,
                       activity totals and the step events the judge reads
    trajs/<id>.npz     obs (T, obs_dim) float16 = agent 0's observations, acts (T, n_agents) int8
    config.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from eval.coalitions import evaluate_coalitions
from eval.config import ENV_DEFAULTS

from .behaviors import BUILTIN, make_behavior

POOL_SEED0 = 50_000          # disjoint from training (seed * 100_000 + k) and evaluation (10_000 + k)


def make_pool_env(env_name: str, layout: str | None, partner: str, horizon: int | None, n_agents: int):
    from environment import make
    d = ENV_DEFAULTS[env_name]
    layout = layout or d.get("layout", "cramped_room")
    horizon = horizon or d.get("horizon", 400)
    return make(env_name, layout, "team", horizon, partner, n_agents=n_agents)


def safe_id(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s)


def collect_job(args):
    env_name, layout, partner, seed, behaviors, horizon, n_agents, out = args
    import torch
    torch.set_num_threads(1)
    env = make_pool_env(env_name, layout, partner, horizon, n_agents)
    records = []
    for spec in behaviors:
        beh = make_behavior(env, spec)
        name = spec if beh == "stay" else beh.name
        traces: list = []
        met = evaluate_coalitions(env, beh, partner, 1, seed0=POOL_SEED0 + seed, traces=traces,
                                  before_episode=None if beh == "stay" else beh.reseed)
        tr = traces[0]
        tid = safe_id(f"{partner}_n{n_agents}_s{seed}_{name}")
        obs = np.stack([s["obs"] for s in tr]).astype(np.float16)
        acts = np.asarray([s["a"] for s in tr], np.int8)
        np.savez_compressed(os.path.join(out, "trajs", tid + ".npz"), obs=obs, acts=acts)
        events = [[t, s["ev"]] for t, s in enumerate(tr) if s["ev"]]
        tot: dict = {}
        for _, ev in events:
            for k, v in ev.items():
                tot[k] = tot.get(k, 0) + v
        records.append({"id": tid, "env": env_name, "layout": env.layout, "partner": partner,
                        "agents": n_agents, "seed": seed, "behavior": name, "unit": env.unit,
                        "pickup_kinds": list(env.pickup_kinds), "T": len(tr),
                        "active0": int((acts[:, 0] != env.stay_action).sum()),
                        "use0": int((acts[:, 0] == env.n_actions - 1).sum()),
                        **met, "events_total": tot, "events": events})
    env.close()
    return records


def load_pool(pool_dir: str) -> list[dict]:
    with open(os.path.join(pool_dir, "meta.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_traj(pool_dir: str, tid: str):
    z = np.load(os.path.join(pool_dir, "trajs", tid + ".npz"))
    return z["obs"].astype(np.float32), z["acts"].astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="kitchen", choices=["kitchen", "warehouse", "foraging", "alem"])
    p.add_argument("--layout")
    p.add_argument("--horizon", type=int)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--partners", nargs="+", default=["stay", "greedy"], choices=["stay", "greedy"])
    p.add_argument("--behaviors", nargs="+", default=list(BUILTIN))
    p.add_argument("--policies", nargs="*", default=[],
                   help="glob(s) of trained checkpoints added as behaviours, e.g. results/full/models/*.pt")
    p.add_argument("--episodes", type=int, default=30, help="pool seeds per partner")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    behaviors = list(a.behaviors) + ["ckpt:" + f for g in a.policies for f in sorted(glob.glob(g))]
    pool = os.path.join(a.out, "pool")
    os.makedirs(os.path.join(pool, "trajs"), exist_ok=True)
    with open(os.path.join(pool, "config.json"), "w") as f:
        json.dump({**vars(a), "behaviors": behaviors, "pool_seed0": POOL_SEED0}, f, indent=2)
    jobs = [(a.env, a.layout, pt, s, behaviors, a.horizon, a.agents, pool)
            for pt in a.partners for s in range(a.episodes)]
    print(f"{len(jobs)} episodes x {len(behaviors)} behaviours on {a.workers} workers -> {pool}")
    t0, n = time.time(), 0
    with open(os.path.join(pool, "meta.jsonl"), "w") as f:
        ctx = get_context("spawn" if a.env == "alem" else None)        # JAX is not fork-safe
        it = ctx.Pool(min(a.workers, len(jobs))).imap_unordered(collect_job, jobs) if a.workers > 1 \
            else map(collect_job, jobs)
        for i, recs in enumerate(it, 1):
            for r in recs:
                f.write(json.dumps(r) + "\n")
                n += 1
            f.flush()
            r0 = recs[0]
            print(f"  [{i}/{len(jobs)}] partner={r0['partner']} seed={r0['seed']}: "
                  + "  ".join(f"{r['behavior']} phi={r['phi']:.2f}" for r in recs), flush=True)
    print(f"{n} trajectories in {time.time() - t0:.0f}s. Next:  python -m rlaif.judge --run {a.out}")


if __name__ == "__main__":
    main()
