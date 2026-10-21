"""Watch the cooks play (GIF, or MP4 if the name ends in .mp4 and ffmpeg is installed).

Blue cook 0 = learner, green cook 1 = partner. The header shows running team soups,
soups delivered by each cook, and the learner's onion and dish pickups.

    # no training needed: the scripted greedy cook in the learner's seat (checks the install)
    python -m eval.visualize --preview --partner greedy

    # a trained learner
    python -m eval.visualize --run results/full --algo ppo --reward individual --partner greedy --seed 0

    # the three rewards side by side (same algo, partner, seed, episode)
    python -m eval.visualize --run results/full --algo ppo --partner greedy --seed 0 --compare

    # one cook frozen, as in the Shapley evaluation
    python -m eval.visualize --run results/full --algo ppo --reward team --partner greedy --freeze partner

    # 3 or 4 agents (warehouse / foraging only)
    python -m eval.visualize --run results/lbf_n --algo ppo --reward team --agents 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .coalitions import EVAL_SEED0, policy_actor, torch_reseed


def header(img: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw
    h = 14 * len(lines) + 6
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + h), "white")
    canvas.paste(Image.fromarray(img), (0, h))
    d = ImageDraw.Draw(canvas)
    for i, ln in enumerate(lines):
        d.text((4, 3 + 14 * i), ln, fill=(20, 20, 20))
    return np.asarray(canvas)


def record(env, seat0, seat1: str, seed: int, title: str) -> list[np.ndarray]:
    raw: list = []
    torch_reseed(seed)
    env.play(seat0, seat1, seed, frames=raw)
    out = []
    for img, tot in raw:
        out.append(header(img, [title,
                                f"team {env.unit} {tot['soups']:.4g}  learner {tot['own_soups']:.4g}  "
                                f"partner {tot['partner_soups']:.4g}",
                                f"learner pickups {tot['pickups0']:.0f}"]))
    return out


def side_by_side(clips: list[list[np.ndarray]]) -> list[np.ndarray]:
    n = min(len(c) for c in clips)
    H = max(c[0].shape[0] for c in clips)
    frames = []
    for t in range(n):
        row = []
        for c in clips:
            im = c[t]
            pad = np.full((H - im.shape[0], im.shape[1], 3), 255, np.uint8)
            row += [np.vstack([im, pad]), np.full((H, 6, 3), 255, np.uint8)]
        frames.append(np.hstack(row[:-1]))
    return frames


def save(frames: list[np.ndarray], path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".mp4"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        im = ax.imshow(frames[0])
        a = anim.FuncAnimation(fig, lambda i: im.set_data(frames[i]) or [im], frames=len(frames))
        a.save(path, writer=anim.FFMpegWriter(fps=fps))
        plt.close(fig)
    else:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    print(f"saved {path} ({len(frames)} frames)")


def load_actor(run: str, algo: str, reward: str, partner: str, seed: int, deterministic: bool,
               n_agents: int = 2):
    from train.deep_policy import load_checkpoint
    from .run import ckpt_name
    path = os.path.join(run, "models", ckpt_name(algo, reward, partner, seed, n_agents))
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Train it first:  python -m eval.run --out {run}")
    policy, obs_rms, _ = load_checkpoint(path)
    return policy_actor(policy, obs_rms, deterministic)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="results/full")
    p.add_argument("--preview", action="store_true", help="scripted greedy cook in the learner seat")
    p.add_argument("--algo", default="ppo", choices=["ppo", "grpo", "a2c", "dqn", "sac"])
    p.add_argument("--reward", default="team", choices=["pickup", "team", "individual"])
    p.add_argument("--partner", default="greedy", choices=["stay", "greedy"])
    p.add_argument("--seed", type=int, default=0, help="training seed of the checkpoint")
    p.add_argument("--episode", type=int, default=0, help="evaluation episode (seed 10000 + k)")
    p.add_argument("--compare", action="store_true", help="the three rewards side by side")
    p.add_argument("--freeze", choices=["learner", "partner"],
                   help="freeze the learner or every partner (Shapley coalitions)")
    p.add_argument("--agents", type=int, default=2, choices=[2, 3, 4],
                   help="number of agents (3, 4: warehouse / foraging only)")
    p.add_argument("--env", default=None, choices=["kitchen", "warehouse", "foraging"],
                   help="default: the run's env, else kitchen")
    p.add_argument("--layout", default=None, help="default: the run's layout, else the env's default")
    p.add_argument("--steps", type=int, default=None, help="episode length to record (default: horizon)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--renderer", default="auto", choices=["auto", "official", "simple"],
                   help="official = the package's pygame drawing; simple = matplotlib, no pygame")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from environment import make
    from .config import load_config, make_config
    cfg_path = os.path.join(a.run, "config.json")
    cfg = load_config(cfg_path) if os.path.exists(cfg_path) and not a.preview else None
    if cfg is None or (a.env and a.env != cfg.env):
        cfg = make_config(env=a.env)
    layout = a.layout or cfg.layout
    det = cfg.eval_deterministic
    env = make(a.env or cfg.env, layout, "team", a.steps or cfg.horizon, a.partner, n_agents=a.agents)
    env.renderer = a.renderer
    seed = EVAL_SEED0 + a.episode
    seat1 = "stay" if a.freeze == "partner" else a.partner

    if a.preview:
        frames = record(env, "greedy", seat1, seed, f"preview: greedy cook + {seat1} partner")
        out = a.out or f"videos/preview_{seat1}.gif"
    elif a.compare:
        clips = [record(env, load_actor(a.run, a.algo, rw, a.partner, a.seed, det, a.agents), seat1, seed,
                        f"{a.algo} {rw} reward, {seat1} partner")
                 for rw in ("pickup", "team", "individual")]
        frames = side_by_side(clips)
        out = a.out or f"videos/compare_{a.algo}_{a.partner}_s{a.seed}_ep{a.episode}.gif"
    else:
        seat0 = "stay" if a.freeze == "learner" else load_actor(a.run, a.algo, a.reward, a.partner,
                                                                 a.seed, det, a.agents)
        tag = f"_freeze-{a.freeze}" if a.freeze else ""
        note = f" ({a.freeze} frozen)" if a.freeze else ""
        frames = record(env, seat0, seat1, seed, f"{a.algo} {a.reward} reward, {a.partner} partner{note}")
        n = f"_n{a.agents}" if a.agents != 2 else ""
        out = a.out or f"videos/{a.algo}_{a.reward}_{a.partner}{n}_s{a.seed}_ep{a.episode}{tag}.gif"
    save(frames, out, a.fps)


if __name__ == "__main__":
    main()
