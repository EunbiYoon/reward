"""Persona purity: does each greedy-trained persona keep its behaviour when the partner is frozen?

    venv/bin/python scripts/persona_solo.py     -> results/train/persona-fig/solo vs withgreedy/persona_solo.{png,csv}
    venv/bin/python scripts/persona_solo.py --replot     # redraw from the csv

The csv is the "alone" input of scripts/paper_fig1.py (paper Figure fig:steps) and the numbers of tab:solo.
For theft, pickup and shadow (trained next to greedy, results/train/<algo>/A_persona), every seed checkpoint is
replayed twice on the evaluation seeds: next to the moving greedy cook (as trained) and alone, with the
partner frozen and parked on a cell next to no pot, dispenser or counter (the learner-alone coalition of the
Shapley evaluation). x = the soup steps 1-7 (step 4, waiting, is not an action), y = the LEARNER's actions
per episode; bars are the mean over seeds, dots single seeds. Each panel also prints the share of the
episode the learner stands next to the partner.
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

PERSONAS = (("theft", "theft"), ("pickup", "pickup"), ("shadow", "shadow"))
ALGOS = ("ppo", "grpo")
MODES = (("greedy", "next to greedy (as trained)"), ("alone", "partner frozen (alone)"))
STEPS = ["1", "2", "3", "4", "5", "6", "7"]
NAMES = {"1": "onion", "2": "pot", "3": "cook", "4": "wait", "5": "dish", "6": "scoop", "7": "deliver"}
COLORS = {"greedy": "#2a78d6", "alone": "#eb6834"}           # validated categorical pair (CVD dE 24.7)
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def measure(episodes: int) -> pd.DataFrame:
    from environment.overcooked import Overcooked
    from eval import persona as P
    from eval.coalitions import EVAL_SEED0, policy_actor, torch_reseed
    from train.deep_policy import load_checkpoint

    env = Overcooked(partner="greedy")
    reset = env._reset_core
    rows = []
    for algo in ALGOS:
        for persona, reward in PERSONAS:
            paths = sorted(glob.glob(f"results/train/{algo}/A_persona/**/models/{algo}_{reward}_greedy_s*.pt",
                                     recursive=True))
            if not paths:
                print(f"  skip {algo} {persona}: no checkpoint under results/train/{algo}/A_persona")
            for path in paths:
                policy, obs_rms, meta = load_checkpoint(path)
                for mode, _ in MODES:
                    # alone: seat 1 frozen and parked, exactly as in the Shapley coalition v(learner)
                    env._reset_core = (lambda seed, frozen=(): reset(seed, (1,))) if mode == "alone" else reset
                    m = P.measure(env, policy_actor(policy, obs_rms, False), "stay" if mode == "alone" else "greedy",
                                  episodes, EVAL_SEED0, torch_reseed)
                    env._reset_core = reset
                    for k in P.KEYS:
                        rows.append({"algo": algo, "persona": persona, "seed": meta["seed"], "mode": mode,
                                     "step": k, "learner": m[f"{k}_L"], "team_soups": m["7_L"] + m["7_P"],
                                     "near_frac": m["near"]})
                    print(f"  {algo} {persona} s{meta['seed']} {mode:6}: "
                          + " ".join(f"{k}:{m[f'{k}_L']:.1f}" for k in P.KEYS) + f"  near {m['near']:.2f}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK2, "text.color": INK, "savefig.bbox": "tight"})
    algos = [a for a in ALGOS if a in set(df.algo)]
    fig, axes = plt.subplots(len(algos), len(PERSONAS), figsize=(3.6 * len(PERSONAS), 3.0 * len(algos)),
                             facecolor=SURFACE, squeeze=False)
    x, w = np.arange(len(STEPS)), 0.38
    for r, algo in enumerate(algos):
        for c, (persona, reward) in enumerate(PERSONAS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            d = df[(df.algo == algo) & (df.persona == persona)]
            ax.axvspan(2.5, 3.5, color=GRID, alpha=0.6, lw=0, zorder=0)
            top = 1.0
            for i, (mode, _) in enumerate(MODES):
                dm = d[d["mode"] == mode]
                off = (i - 0.5) * w
                means = [dm[dm.step == s].learner.mean() if s != "4" and len(dm) else 0.0 for s in STEPS]
                means = [0.0 if v != v else v for v in means]
                ax.bar(x + off, means, w * 0.92, color=COLORS[mode], zorder=2)
                for j, s in enumerate(STEPS):
                    vals = dm[dm.step == s].learner.values if s != "4" else []
                    ax.scatter(np.full(len(vals), x[j] + off), vals, s=8, color=INK, alpha=0.5, lw=0, zorder=3)
                    top = max(top, max(vals, default=0))
            ax.set_ylim(0, top * 1.25)
            ax.text(3, top * 1.15, "wait", ha="center", va="center", color=MUTED, fontsize=7.5)
            near = {m: d[d["mode"] == m].groupby("seed").near_frac.first().mean() for m, _ in MODES}
            soups = {m: d[d["mode"] == m].groupby("seed").team_soups.first().mean() for m, _ in MODES}
            ax.text(0, 1.22, f"{persona} · {algo.upper()}", transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom")
            ax.text(0, 1.13, f"{reward} reward, trained next to greedy", transform=ax.transAxes, fontsize=7.5,
                    color=MUTED, va="bottom")
            ax.text(0, 1.03, f"near partner {near['greedy']:.0%} \u2192 {near['alone']:.0%} · "
                    f"team soups {soups['greedy']:.1f} \u2192 {soups['alone']:.1f}",
                    transform=ax.transAxes, fontsize=7.2, color=INK2, va="bottom")
            ax.set_xticks(x, [f"{s}\n{NAMES[s]}" for s in STEPS], fontsize=7)
            ax.tick_params(length=0)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_ylabel("learner actions per episode")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m, _ in MODES]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.5, markersize=3.5))
    fig.tight_layout(h_pad=3.5, w_pad=1.6)
    fig.legend(handles, [lab for _, lab in MODES] + ["one seed"], loc="lower left", ncol=3, frameon=False,
               bbox_to_anchor=(0.0, 1.0), fontsize=8.5)
    fig.suptitle("Does the persona survive without its partner? Learner's soup steps (arrows: with greedy \u2192 alone), "
                 "each panel on its own y scale",
                 x=0.0, y=1.07, ha="left", fontsize=10, color=INK2)
    fig.savefig(path + ".png", dpi=200, facecolor=SURFACE)
    print(f"-> {path}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--replot", action="store_true")
    a = p.parse_args()
    out = "results/train/persona-fig/solo vs withgreedy/persona_solo"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.replot:
        df = pd.read_csv(out + ".csv", dtype={"step": str})
    else:
        df = measure(a.episodes)
        if df.empty:
            raise SystemExit("no greedy-trained persona checkpoints found")
        df.to_csv(out + ".csv", index=False)
    plot(df, out)


if __name__ == "__main__":
    main()
