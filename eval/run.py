"""Measure trained checkpoints: the learner's Shapley contribution in every checkpoint of a run folder.

Checks whether training produced what we wanted. The checkpoints come from train/run.py.

Usage (from the project root):
    python -m eval.run results/ppo/A                          # every checkpoint in results/ppo/A/models
    python -m eval.run results/ppo/A results/ppo/A_personas --workers 12
    python -m eval.run results/ppo/A --only-new               # only checkpoints not in the latest evaluation
    python -m eval.run results/ppo/A --episodes 20            # more evaluation games than config.json says

Each checkpoint carries its own settings (algo, reward, partner, seed, agents, pickup bonus, ...), and
<run>/config.json gives the environment, layout, horizon and evaluation settings, so a folder is all
that is needed. Every checkpoint is replayed three ways on the same seeds (eval/coalitions.py): the
real team, the learner alone (partners frozen) and the partners alone (learner frozen).

Writes a new folder <run>/eval/<YYYYmmdd_HHMMSS>/ each time (earlier evaluations are kept; readers use
the newest, see eval/paths.py):
    eval_info.json  when, how many games, which checkpoints were measured this time
    persona_check.md  did training produce the intended persona: soup steps 1-7 per seat and a verdict
    reference.csv   the scripted greedy cook in the learner's seat, per partner (s*, pickups*)
    results.csv     one row per checkpoint: team soups, own soups, Shapley phi and share s, credit gap,
                    pickups, soup-step counts, intended persona and persona_match (eval/persona.py),
                    ..., plus its training columns from train_log.csv
Then:  python -m eval.summarize <run>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import replace
from multiprocessing import get_context

from train.run import make_env, partner_tag, read_csv, write_csv

from .paths import latest, new_eval_dir
from .coalitions import EVAL_SEED0, evaluate_coalitions, policy_actor, torch_reseed
from . import persona
from train.config import Config, load_config


def rm_score(cfg: Config, traces: list) -> float:
    """Mean learned-reward score of the evaluation episodes (nan without a reward model)."""
    if not cfg.reward_model or not traces:
        return float("nan")
    import numpy as np
    from rlaif.reward_model import load_rm, score
    rm = load_rm(cfg.reward_model)
    return float(np.mean([score(rm, np.stack([s["obs"] for s in tr]), np.array([s["a"][0] for s in tr]))
                          for tr in traces]))


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


def reference(cfg: Config, partners: list, agents: list) -> list[dict]:
    """The scripted greedy cook in the learner's seat: what a good teammate looks like, per partner."""
    rows = []
    for noise in cfg.partner_noises or (cfg.partner_noise,):
        c = replace(cfg, partner_noise=noise)
        for n_agents in agents:
            for partner in partners:
                if partner == "self":         # no fixed partner to put the scripted cook next to
                    continue
                env = make_env(c, "team", partner, n_agents)
                traces: list = []
                met = evaluate_coalitions(env, "greedy", partner, cfg.eval_episodes,
                                          traces=traces if cfg.reward_model else None)
                rows.append({"partner": partner, "partner_noise": noise, **met, "rm_score": rm_score(cfg, traces)})
                print(f"  reference (greedy cook in the learner seat), agents={n_agents} partner={partner_tag(partner)} "
                      f"noise={noise}: team soups={met['soups']:.2f}  s*={met['s']:.2f}  pickups*={met['pickups']:.1f}")
    return rows


def checkpoints(run: str) -> list[str]:
    """The policy checkpoints of a run folder (partner-prediction heads, *_head.pt, are not policies)."""
    return sorted(p for p in glob.glob(os.path.join(run, "models", "*.pt")) if not p.endswith("_head.pt"))


def checkpoint_meta(path: str) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["meta"]


def evaluate_checkpoint(args) -> dict:
    """Measure one checkpoint. args = (run, cfg, path). Returns its results.csv row."""
    run, cfg, path = args
    import torch
    torch.set_num_threads(1)
    from train.deep_policy import load_checkpoint, make_partner_head

    policy, obs_rms, m = load_checkpoint(path)
    algo, reward, partner, seed, n_agents = m["algo"], m["reward"], m["partner"], m["seed"], m.get("agents", 2)
    c = replace(cfg, pickup_bonus=m.get("pickup_bonus", cfg.pickup_bonus),
                partner_noise=m.get("partner_noise", cfg.partner_noise), aux_coef=m.get("aux_coef", 0.0) or 0.0)
    if partner.startswith("ckpt:") and not os.path.exists(partner[5:]):
        raise SystemExit(f"{path}: its partner checkpoint {partner[5:]} is missing (moved or deleted). "
                         f"Put it back at that path to evaluate this run.")
    env = make_env(c, reward, partner, n_agents)
    if partner == "self":                      # self-play: the partner is the learner itself
        env.self_policy = (policy, obs_rms)
    head = None
    head_path = path[:-3] + "_head.pt"
    if os.path.exists(head_path):
        head = make_partner_head(env, policy, m.get("hidden", cfg.hidden), c.aux_coef or 1.0)
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
    actor = policy_actor(policy, obs_rms, cfg.eval_deterministic)
    traces: list = []
    met = evaluate_coalitions(env, actor, partner, cfg.eval_episodes, before_episode=torch_reseed,
                              traces=traces if (cfg.reward_model or head is not None) else None)
    met["rm_score"] = rm_score(cfg, traces)        # what the reward model thinks of the trained policy
    met["partner_pred_acc"] = partner_pred_acc(policy, obs_rms, head, traces)
    met.update(persona_columns(env, actor, reward, partner, n_agents, cfg.eval_episodes))
    env.close()
    return {"seed": seed, "algo": algo, "reward": reward, "partner": partner,
            "pickup_bonus": c.pickup_bonus, "partner_noise": c.partner_noise, "aux_coef": c.aux_coef,
            **met, "checkpoint": os.path.relpath(path, run)}


def persona_columns(env, actor, reward: str, partner: str, n_agents: int, episodes: int) -> dict:
    """Soup-step counts and the intended-persona verdict (eval/persona.py); Kitchen with 2 agents only."""
    from train.resume import base_env
    from environment.kitchen import Kitchen
    core = base_env(env)
    if not isinstance(core, Kitchen) or n_agents != 2:
        return {}
    m = persona.measure(core, actor, partner, episodes, EVAL_SEED0, torch_reseed)
    return persona.columns(persona.intended(reward, partner), m)


TRAIN_KEYS = ("env_steps_used", "updates", "episodes", "grpo_skipped_groups", "train_proxy_return_last5",
              "train_seconds")


def training_info(run: str) -> dict:
    """checkpoint file name -> its training columns, from train_log.csv or, for folders made before the
    train/eval split, from the old <run>/results.csv."""
    info = {os.path.basename(r.get("checkpoint", "")): r for r in read_csv(os.path.join(run, "results.csv"))}
    info.update({os.path.basename(r.get("checkpoint", "")): r for r in read_csv(os.path.join(run, "train_log.csv"))})
    return {name: {k: r[k] for k in TRAIN_KEYS if r.get(k) not in (None, "")} for name, r in info.items()}


def with_training(info: dict, rows: list[dict]) -> list[dict]:
    """Each row with its training columns appended (checkpoint stays the last column)."""
    out = []
    for row in rows:
        rest = {k: v for k, v in row.items() if k != "checkpoint"}
        out.append({**rest, **info.get(os.path.basename(row["checkpoint"]), {}), "checkpoint": row["checkpoint"]})
    return out


def run_settings(run: str, episodes: int | None) -> Config:
    path = os.path.join(run, "config.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: is {run} a folder made by python -m train.run?")
    cfg = load_config(path)
    return replace(cfg, eval_episodes=episodes) if episodes else cfg


def evaluate_run(run: str, workers: int = 1, names: list | None = None, only_new: bool = False,
                 episodes: int | None = None) -> str | None:
    """Measure the checkpoints of one run folder (all, or only `names`) into a new <run>/eval/<time>/.
    Checkpoints not measured this time are carried over from the latest evaluation, so the newest
    folder always holds every checkpoint. Returns that folder (None if there was nothing to do)."""
    cfg = run_settings(run, episodes)
    paths = checkpoints(run)
    if not paths:
        raise SystemExit(f"no checkpoints in {run}/models")
    prev = latest(run)                                   # the latest evaluation (or the old layout)
    old = read_csv(os.path.join(prev, "results.csv"))
    done = {os.path.basename(r.get("checkpoint", "")) for r in old}
    if names is not None:
        paths = [p for p in paths if os.path.basename(p) in set(names)]
    if only_new:
        paths = [p for p in paths if os.path.basename(p) not in done]
    if not paths:
        print(f"{run}: nothing to evaluate (latest evaluation: {prev})")
        return None

    info = training_info(run)                            # before anything is written
    out = new_eval_dir(run)
    metas = [checkpoint_meta(p) for p in paths]
    partners = list(dict.fromkeys(m["partner"] for m in metas))
    agents = sorted({m.get("agents", 2) for m in metas})
    print(f"{run}: {len(paths)} checkpoints, {cfg.eval_episodes} evaluation games each, {workers} workers -> {out}")
    write_info(out, cfg, paths, prev if old else None)
    ref_new = reference(cfg, partners, agents)
    ref_keep = [r for r in read_csv(os.path.join(prev, "reference.csv")) if r.get("partner") not in set(partners)]
    write_csv(os.path.join(out, "reference.csv"), ref_keep + ref_new)

    evaluated = {os.path.basename(p) for p in paths}
    keep = [r for r in old if os.path.basename(r.get("checkpoint", "")) not in evaluated]
    rows, t0 = [], time.time()

    def report(i, row):
        rows.append(row)
        write_csv(os.path.join(out, "results.csv"), keep + with_training(info, rows))
        print(f"  [{i}/{len(paths)}] seed={row['seed']} {row['algo']:>4} {row['reward']:<10} "
              f"partner={partner_tag(row['partner']):<6} team={row['soups']:.2f} own={row['own_soups']:.2f} "
              f"s={row['s']:+.2f} gap={row['credit_gap']:+.2f}"
              + (f"  persona {row['intended'] or '?'}: {row['persona_match']}" if "intended" in row else ""),
              flush=True)

    jobs = [(run, cfg, p) for p in paths]
    if workers > 1 and len(jobs) > 1:
        ctx = get_context("spawn" if cfg.env == "alem" else None)     # JAX is not fork-safe
        with ctx.Pool(min(workers, len(jobs))) as pool:
            for i, row in enumerate(pool.imap_unordered(evaluate_checkpoint, jobs), 1):
                report(i, row)
    else:
        for i, job in enumerate(jobs, 1):
            report(i, evaluate_checkpoint(job))
    write_persona_log(out, keep + rows)
    print(f"evaluated in {time.time() - t0:.0f}s -> {os.path.join(out, 'results.csv')}, "
          f"{os.path.join(out, 'persona_check.md')}. "
          f"Next:  python -m eval.summarize {run}")
    return out


def write_persona_log(out: str, rows: list[dict]):
    if any(r.get("persona_match") not in (None, "") for r in rows):
        with open(os.path.join(out, "persona_check.md"), "w") as f:
            f.write(persona.log(rows))


def write_info(out: str, cfg: Config, paths: list, carried_from: str | None):
    """eval_info.json: when and how this evaluation was made."""
    with open(os.path.join(out, "eval_info.json"), "w") as f:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "eval_episodes": cfg.eval_episodes,
                   "eval_deterministic": cfg.eval_deterministic,
                   "evaluated": [os.path.basename(p) for p in paths],
                   "carried_over_from": carried_from}, f, indent=2)


def evaluate_part(run: str, name: str):
    """Slurm-array mode (train.run --job-index ... --eval): measure one checkpoint into parts/, then rebuild
    <run>/eval/slurm_<array job id>/results.csv from every measured part, so concurrent array tasks never
    overwrite each other."""
    cfg = run_settings(run, None)
    out = new_eval_dir(run, "slurm_" + os.environ.get("SLURM_ARRAY_JOB_ID", "local"))
    parts = os.path.join(run, "parts")
    part = os.path.join(parts, name[:-3] + ".eval.json")
    if not os.path.exists(os.path.join(out, "reference.csv")):
        m = checkpoint_meta(os.path.join(run, "models", name))
        write_csv(os.path.join(out, "reference.csv"), reference(cfg, list(cfg.partners), [m.get("agents", 2)]))
    if not os.path.exists(part):
        row = evaluate_checkpoint((run, cfg, os.path.join(run, "models", name)))
        with open(part + ".tmp", "w") as f:
            json.dump(row, f)
        os.replace(part + ".tmp", part)
        print(f"  team={row['soups']:.2f} s={row['s']:+.2f} gap={row['credit_gap']:+.2f} -> {part}", flush=True)
    rows = []
    for fn in sorted(os.listdir(parts)):
        if fn.endswith(".eval.json"):
            with open(os.path.join(parts, fn)) as f:
                rows.append(json.load(f))
    write_csv(os.path.join(out, "results.csv"), with_training(training_info(run), rows))
    write_persona_log(out, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folders made by python -m train.run (each has config.json and models/)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--only-new", action="store_true", help="only checkpoints missing from the latest evaluation (the rest are carried over)")
    p.add_argument("--episodes", type=int, help="evaluation games per checkpoint (default: config.json, 10)")
    a = p.parse_args()
    for run in a.runs:
        evaluate_run(run, a.workers, only_new=a.only_new, episodes=a.episodes)


if __name__ == "__main__":
    main()
