"""How the four personas act across the soup steps 1-7: learner vs partner actions per episode.

    venv/bin/python scripts/persona_actions.py                 # PPO -> results/ppo/persona_actions.{png,csv}
    venv/bin/python scripts/persona_actions.py --algo grpo     # GRPO folders
    venv/bin/python scripts/persona_actions.py --replot        # redraw from the csv, no replay
    venv/bin/python scripts/persona_actions.py --root results/ppo/A_personas --partner A   # next to A
    venv/bin/python scripts/persona_actions.py --algo ppo --root results/train/ppo/A_persona   # the paper's (fig:steps)
        (theft, pickup, shadow found anywhere under the folder) -> <root>/persona_actions.{png,csv}

One panel per persona (theft, quit, pickup, shadow). x = the soup steps in cooking order; step 4 (waiting
for the soup) is not an action and is shaded. y = actions per episode. Bars are the mean over the seed
checkpoints, dots the single seeds. Every checkpoint is replayed next to the partner it was trained
with on the evaluation seeds (eval/persona.py, the same counts as persona_check.md).
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# persona -> (checkpoint glob, panel subtitle); {a} = algo
PERSONAS = {
    "theft": ("results/{a}/{a}_theft/models/{a}_theft_greedy_s*.pt", "own-delivery reward, next to greedy"),
    "quit": ("results/{a}/A_personas/models/{a}_theft_ckpt-*_s*.pt", "own-delivery reward, next to A"),
    "pickup": ("results/{a}/greedy_personas/models/{a}_pickup_greedy_s*.pt", "pickup reward, next to greedy"),
    "shadow": ("results/{a}/greedy_personas/models/{a}_shadow_greedy_s*.pt", "shadow reward, next to greedy"),
}
# --root: the personas trained next to greedy, searched anywhere under the folder; {r} = root, {a} = algo
GREEDY_PERSONAS = {
    "theft": ("{r}/**/models/{a}_theft_greedy_s*.pt", "own-delivery reward, next to greedy"),
    "pickup": ("{r}/**/models/{a}_pickup_greedy_s*.pt", "pickup reward, next to greedy"),
    "shadow": ("{r}/**/models/{a}_shadow_greedy_s*.pt", "shadow reward, next to greedy"),
}
# --root with --partner A: the same rewards trained next to a frozen A checkpoint
A_PERSONAS = {
    "quit": ("{r}/**/models/{a}_theft_ckpt-*_s*.pt", "own-delivery reward, next to A"),
    "pickup": ("{r}/**/models/{a}_pickup_ckpt-*_s*.pt", "pickup reward, next to A"),
    "shadow": ("{r}/**/models/{a}_shadow_ckpt-*_s*.pt", "shadow reward, next to A"),
    "preparer": ("{r}/**/models/{a}_team_ckpt-*_s*.pt", "team reward, next to A"),
}
STEPS = ["1", "2", "3", "4", "5", "6", "7"]
NAMES = {"1": "onion", "2": "pot", "3": "cook", "4": "wait", "5": "dish", "6": "scoop", "7": "deliver"}

# light-mode chart palette (dataviz reference palette; learner/partner validated for CVD separation)
LEARNER, PARTNER = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def measure(algo: str, episodes: int, personas: dict, root: str = "") -> pd.DataFrame:
    """One row per (persona, checkpoint, step, seat) with actions per episode, plus near_frac per checkpoint."""
    from environment.overcooked import Overcooked
    from eval.coalitions import EVAL_SEED0, policy_actor, torch_reseed
    from eval.persona import KEYS, measure as count
    from train.deep_policy import load_checkpoint

    env = Overcooked(partner="greedy")
    rows = []
    for persona, (pattern, _) in personas.items():
        pattern = pattern.format(a=algo, r=root)
        paths = sorted(glob.glob(pattern, recursive=True))
        paths = [p for p in paths if not p.endswith("_head.pt")]
        if not paths:
            print(f"  skip {persona}: no checkpoint matches {pattern}")
            continue
        for path in paths:
            policy, obs_rms, meta = load_checkpoint(path)
            partner = meta["partner"]
            if partner.startswith("ckpt:") and not os.path.exists(partner[5:]):
                print(f"  skip {path}: partner {partner[5:]} missing")
                continue
            m = count(env, policy_actor(policy, obs_rms, deterministic=False), partner, episodes, EVAL_SEED0,
                      torch_reseed)
            for k in KEYS:
                rows.append({"persona": persona, "checkpoint": os.path.basename(path), "seed": meta["seed"],
                             "step": k, "learner": m[f"{k}_L"], "partner": m[f"{k}_P"], "near_frac": m["near"]})
            print(f"  {persona} seed {meta['seed']}: "
                  + "  ".join(f"{k}:{m[f'{k}_L']:.1f}/{m[f'{k}_P']:.1f}" for k in KEYS) + f"  near {m['near']:.2f}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str, algo: str, personas_def: dict):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "savefig.bbox": "tight"})
    personas = [p for p in personas_def if p in set(df.persona)]
    fig, axes = plt.subplots(1, len(personas), figsize=(3.6 * len(personas), 3.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    x = np.arange(len(STEPS))
    w = 0.38
    for ax, persona in zip(axes, personas):
        d = df[df.persona == persona]
        ax.set_facecolor(SURFACE)
        ax.axvspan(3 - 0.5, 3 + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)       # step 4: waiting, no action
        top = 0.0
        for seat, color, off in (("learner", LEARNER, -w / 2), ("partner", PARTNER, w / 2)):
            means = [d[d.step == s][seat].mean() if s != "4" else 0.0 for s in STEPS]
            means = [0.0 if v != v else v for v in means]
            ax.bar(x + off, means, w * 0.92, color=color, zorder=2, label=seat)
            for i, s in enumerate(STEPS):
                vals = d[d.step == s][seat].values if s != "4" else []
                ax.scatter(np.full(len(vals), x[i] + off), vals, s=9, color=INK, alpha=0.55, zorder=3, lw=0)
                top = max(top, max(vals, default=0))
        top = max(top, 1.0)
        ax.set_ylim(0, top * 1.18)
        ax.text(3, top * 1.1, "wait", ha="center", va="center", color=MUTED, fontsize=7.5)
        near = d.groupby("checkpoint").near_frac.first().mean()
        if persona == "shadow" or near >= 0.8:
            ax.text(0.5, 0.5, f"no soup steps\nnext to the partner\n{near:.0%} of the episode",
                    transform=ax.transAxes, ha="center", va="center", color=INK2, fontsize=8.5)
        n = d.checkpoint.nunique()
        ax.text(0, 1.13, persona, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
        ax.text(0, 1.03, f"{personas_def[persona][1]} · {n} seeds", transform=ax.transAxes, fontsize=7.5,
                color=MUTED, va="bottom")
        if top <= 1.0 and not len(d[(d.learner > 0) | (d.partner > 0)]):
            ax.set_yticks([0])
        ax.set_xticks(x, [f"{s}\n{NAMES[s]}" for s in STEPS], fontsize=7)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("actions per episode")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEARNER), plt.Rectangle((0, 0), 1, 1, color=PARTNER),
               plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.55, markersize=3.5)]
    fig.legend(handles, ["learner (persona)", "partner", "one seed"], loc="upper right", ncol=3, frameon=False,
               bbox_to_anchor=(1.0, 1.13), fontsize=8.5)
    fig.suptitle(f"Soup steps by persona ({algo.upper()}), each panel on its own y scale", x=0.0, y=1.13,
                 ha="left", fontsize=10, color=INK2)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(path + ".png", dpi=200, facecolor=SURFACE)
    print(f"-> {path}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="ppo", choices=["ppo", "grpo"])
    p.add_argument("--episodes", type=int, default=10, help="evaluation games per checkpoint")
    p.add_argument("--replot", action="store_true", help="redraw from the saved csv")
    p.add_argument("--root", help="only the personas next to one kind of partner, found anywhere under this folder")
    p.add_argument("--partner", default="greedy", choices=["greedy", "A"],
                   help="with --root: personas trained next to greedy (default) or next to a frozen A")
    a = p.parse_args()
    personas = (A_PERSONAS if a.partner == "A" else GREEDY_PERSONAS) if a.root else PERSONAS
    name = "persona_actions" + ("_A" if a.root and a.partner == "A" else "")
    out = os.path.join(a.root, name) if a.root else f"results/{a.algo}/persona_actions"
    if a.replot:
        df = pd.read_csv(out + ".csv", dtype={"step": str})
    else:
        df = measure(a.algo, a.episodes, personas, a.root or "")
        if df.empty:
            raise SystemExit("no persona checkpoints found")
        df.to_csv(out + ".csv", index=False)
    plot(df, out, a.algo, personas)


if __name__ == "__main__":
    main()
