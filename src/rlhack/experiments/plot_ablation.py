"""Figures for results/ablation.csv: (a) lambda sweep, raw vs. contrast penalty;
(b) query strategies at an equal label budget.

    python -m rlhack.experiments.plot_ablation results/ablation.csv --out figures/ablation
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ENVS = ["lava_shortcut", "coin_trap", "combined"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="figures/ablation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv)))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["method"], r["env"])].append((float(r["norm_gap"]), float(r["hacked"])))

    # (a) lambda sweep
    fig, axes = plt.subplots(2, len(ENVS), figsize=(4.2 * len(ENVS), 6), sharex=True)
    for j, env in enumerate(ENVS):
        for prefix, label, color in (("varpen_lam", "contrast std (ours)", "#08519c"),
                                     ("rawpen_lam", "raw std", "#e6550d")):
            pts = []
            for (m, e), v in agg.items():
                mm = re.fullmatch(prefix + r"([0-9.]+)", m)
                if mm and e == env and float(mm.group(1)) > 0:
                    v = np.array(v)
                    pts.append((float(mm.group(1)), v[:, 0].mean(), v[:, 0].std() / np.sqrt(len(v)),
                                v[:, 1].mean(), v[:, 1].std() / np.sqrt(len(v))))
            pts.sort()
            if not pts:
                continue
            lam, g, gse, h, hse = map(np.array, zip(*pts))
            axes[0, j].errorbar(lam, g, gse, marker="o", label=label, color=color, capsize=3)
            axes[1, j].errorbar(lam, h, hse, marker="o", label=label, color=color, capsize=3)
        axes[0, j].set_title(env)
        axes[1, j].set_xlabel("λ")
        axes[1, j].set_xscale("log", base=2)
        for i in range(2):
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("normalized true-return gap ↓")
    axes[1, 0].set_ylabel("reward-hacking rate ↓")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lambda_sweep.png"), dpi=150)
    plt.close(fig)

    # (b) query strategies
    strategies = ["varpen_lam1", "q_random", "q_hot_states", "q_disagreement", "q_ours", "q_ours_lam0"]
    labels = ["no queries", "random", "hot states", "disagreement", "ours", "ours, λ=0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(strategies))
    width = 0.8 / len(ENVS)
    for k, env in enumerate(ENVS):
        for i, col in enumerate((0, 1)):
            vals = [np.array(agg[(s, env)])[:, col] for s in strategies]
            axes[i].bar(x + k * width, [v.mean() for v in vals], width,
                        yerr=[v.std() / np.sqrt(len(v)) for v in vals], label=env, capsize=2)
    for i, t in enumerate(("normalized true-return gap ↓", "reward-hacking rate ↓")):
        axes[i].set_xticks(x + width, labels, rotation=25, ha="right", fontsize=8)
        axes[i].set_ylabel(t)
        axes[i].grid(alpha=0.3, axis="y")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Equal budget: 10 preference labels on the deployment map (λ=1 unless noted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "query_strategies.png"), dpi=150)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
