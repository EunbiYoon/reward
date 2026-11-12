"""Figures for docs/paper/main.tex, from whatever results exist (missing ones are skipped).

    venv/bin/python scripts/paper_figures.py        -> docs/paper/figures/*.pdf

fig_rl.pdf      credit gap and Shapley share per reward, PPO vs GRPO, greedy partner (Overcooked)
fig_judge.pdf   LLM-judge over-credit per agent-0 behaviour, for the three prompts
fig_bonus.pdf   pickups and Shapley share vs pickup-bonus size
fig_noise.pdf   credit gap vs partner action noise
Colors: validated categorical palette (dataviz skill), fixed order PPO=blue, GRPO=orange.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "docs/paper/figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ALGO = {"ppo": ("PPO", C[0], "o"), "grpo": ("GRPO", C[1], "s")}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def first(*paths):
    return next((p for p in paths if os.path.exists(os.path.join(p, "results.csv"))), None)


def load(run):
    df = pd.read_csv(os.path.join(run, "results.csv"))
    ref = pd.read_csv(os.path.join(run, "reference.csv"))
    return df, ref


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    from scipy.stats import t
    return float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def dots_and_means(ax, df, col, order, ref=None, ref_label=None):
    width = 0.32
    for k, (algo, (name, color, marker)) in enumerate(ALGO.items()):
        for i, rew in enumerate(order):
            v = df[(df.algo == algo) & (df.reward == rew)][col].to_numpy()
            if not len(v):
                continue
            x = i + (k - 0.5) * width
            ax.errorbar(x, v.mean(), yerr=ci(v), fmt=marker, color=color, ms=5, capsize=2, lw=1.2,
                        label=name if i == 0 else None, zorder=3)
            jitter = np.linspace(-0.06, 0.06, len(v))
            ax.scatter(x + jitter, v, s=8, color=color, alpha=0.35, lw=0, zorder=2)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1, ls="--", zorder=1)
        ax.text(len(order) - 0.5, ref, f" {ref_label}", va="bottom", ha="right", color=INK2, fontsize=7)
    ax.set_xticks(range(len(order)), [r.replace("individual", "indiv.") for r in order])
    ax.set_xlim(-0.6, len(order) - 0.4)


def fig_rl():
    main = first("results/paper/full5", "results/full")
    if not main:
        return
    df, ref = load(main)
    cf = first("results/paper/cf5", "results/h10_cf")
    if cf:
        df = pd.concat([df, pd.read_csv(os.path.join(cf, "results.csv"))], ignore_index=True)
    df = df[df.partner == "greedy"]
    r = ref[ref.partner == "greedy"].iloc[0]
    order = [x for x in ("pickup", "team", "individual", "cf") if x in set(df.reward)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    dots_and_means(axes[0], df, "credit_gap", order, r.credit_gap, f"scripted cook {r.credit_gap:.2f}")
    axes[0].set_ylabel("credit gap (own share $-$ s)")
    axes[0].set_title("(a) credit the reward gives minus real share", fontsize=8, color=INK)
    dots_and_means(axes[1], df, "s", order, r.s, f"scripted cook {r.s:.2f}")
    axes[1].set_ylabel("Shapley share s")
    axes[1].set_title("(b) real contribution", fontsize=8, color=INK)
    axes[0].legend(loc="upper left")
    seeds = sorted(df.seed.unique())
    fig.text(0.5, -0.04, f"Overcooked cramped_room, greedy partner; {len(seeds)} seeds (dots), mean and 95% t-interval.",
             ha="center", color=INK2, fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_rl.pdf"))
    print("fig_rl.pdf from", main, cf)


def fig_judge():
    path = "results/rlaif/theft/audit.json"
    if not os.path.exists(path):
        return
    rep = json.load(open(path))["judges"]
    conds = [(k, lab) for k, lab in (("ollama", "basic"), ("ollama_definition", "+ definition"),
                                    ("ollama_solo", "+ definition + solo results")) if k in rep]
    if not conds:
        return
    names = {"greedy": "scripted\ncook", "fidget": "fidget\n(blocks)", "ckpt:ppo_individual_greedy_s0": "thief\n(PPO indiv.)",
             "ckpt:grpo_team_greedy_s0": "spectator\n(GRPO team)", "ckpt:ppo_pickup_greedy_s0": "fake help\n(PPO pickup)"}
    behs = [b for b in names if any(r["behavior"] == b for k, _ in conds for r in rep[k]["per_behavior"])]
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    w = 0.8 / len(conds)
    for j, (k, lab) in enumerate(conds):
        rows = {r["behavior"]: r for r in rep[k]["per_behavior"]}
        xs = [i + (j - (len(conds) - 1) / 2) * w for i in range(len(behs))]
        ys = [rows[b]["over_credit"] if b in rows else np.nan for b in behs]
        ax.bar(xs, ys, width=w - 0.04, color=C[j], label=lab, zorder=2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(behs)), [names[b] for b in behs])
    ax.set_ylabel("judge credit $-$ Shapley share")
    ax.legend(ncol=len(conds), loc="upper left", fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_judge.pdf"))
    print("fig_judge.pdf:", [k for k, _ in conds])


def fig_bonus():
    run = first("results/h7_bonus")
    if not run:
        return
    df, ref = load(run)
    df = df[df.partner == "greedy"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0))
    for algo, (name, color, marker) in ALGO.items():
        g = df[df.algo == algo].groupby("pickup_bonus")
        for ax, col in zip(axes, ("pickups", "s")):
            m = g[col].mean()
            e = g[col].apply(ci)
            ax.errorbar(m.index, m.values, yerr=e.values, color=color, marker=marker, ms=4, lw=1.5, capsize=2, label=name)
    r = ref[ref.partner == "greedy"].iloc[0]
    axes[0].axhline(r.pickups, color=INK2, ls="--", lw=1)
    axes[1].axhline(r.s, color=INK2, ls="--", lw=1)
    axes[0].set_ylabel("pickups per episode")
    axes[1].set_ylabel("Shapley share s")
    for ax in axes:
        ax.set_xlabel("pickup bonus")
    axes[0].legend()
    fig.savefig(os.path.join(OUT, "fig_bonus.pdf"))
    print("fig_bonus.pdf")


def fig_noise():
    run = first("results/h8_noise")
    if not run:
        return
    df, ref = load(run)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for algo, (name, color, marker) in ALGO.items():
        for rew, ls in (("individual", "-"), ("team", ":")):
            g = df[(df.algo == algo) & (df.reward == rew)].groupby("partner_noise")["credit_gap"]
            if not len(g):
                continue
            m = g.mean()
            ax.errorbar(m.index, m.values, yerr=g.apply(ci).values, color=color, marker=marker, ms=4, lw=1.5,
                        ls=ls, capsize=2, label=f"{name} {rew}")
    ax.set_xlabel("partner action noise")
    ax.set_ylabel("credit gap")
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(OUT, "fig_noise.pdf"))
    print("fig_noise.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_rl, fig_judge, fig_bonus, fig_noise):
        f()
