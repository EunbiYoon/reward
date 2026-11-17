"""Paper figures for the personas trained next to greedy (results/<algo>/need_curve).

    venv/bin/python scripts/paper_persona_figs.py        -> docs/paper/figures/fig_persona_steps.pdf
                                                            docs/paper/figures/fig_curves_ppo.pdf
                                                            docs/paper/figures/fig_curves_grpo.pdf

fig_persona_steps: PPO (top) and GRPO (bottom), one panel per persona (theft, pickup, shadow): learner vs
partner actions per episode for the soup steps 1-7. Reads results/<algo>/need_curve/persona_actions.csv, so run
    venv/bin/python scripts/persona_actions.py --algo <algo> --root results/<algo>/need_curve
first (10 evaluation games per seed checkpoint).

fig_curves_<algo>: learning curves of cook (A: team reward), theft, pickup and shadow, all next to greedy.
Mean over seeds 0-2 (line) and min-max (band), smoothed over 10 updates. Dashed line: the official shaping
reaches 0 (1M steps).
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "docs/paper/figures"
STEPS = ["1", "2", "3", "4", "5", "6", "7"]
NAMES = {"1": "onion", "2": "pot", "3": "cook", "4": "wait", "5": "dish", "6": "scoop", "7": "serve"}
PERSONAS = (("theft", "individual"), ("pickup", "pickup"), ("shadow", "shadow"))
LEARNER, PARTNER = "#2a78d6", "#eb6834"                  # validated categorical pair
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# curves: categorical slots in fixed order, one per persona
CURVE_COLORS = {"cook (A)": "#2a78d6", "theft": "#eb6834", "pickup": "#1baf7a", "shadow": "#eda100"}
CURVES = {  # label -> glob of the curve csvs, {a} = algo
    "cook (A)": ["results/{a}/need_curve/A/curves/{a}_team_greedy_s*.csv", "results/{a}/A/curves/{a}_team_greedy_s*.csv"],
    "theft": ["results/{a}/need_curve/**/curves/{a}_individual_greedy_s*.csv"],
    "pickup": ["results/{a}/need_curve/**/curves/{a}_pickup_greedy_s*.csv"],
    "shadow": ["results/{a}/need_curve/**/curves/{a}_shadow_greedy_s*.csv"],
}
PANELS = (("proxy_return", "training reward"), ("team_soups", "team soups"),
          ("own_soups", "learner deliveries"), ("pickups", "learner pickups"))

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False})


def steps_figure():
    algos = [a for a in ("ppo", "grpo") if os.path.exists(f"results/{a}/need_curve/persona_actions.csv")]
    fig, axes = plt.subplots(len(algos), 3, figsize=(7.0, 2.15 * len(algos)), squeeze=False)
    x, w = np.arange(len(STEPS)), 0.38
    for r, algo in enumerate(algos):
        df = pd.read_csv(f"results/{algo}/need_curve/persona_actions.csv", dtype={"step": str})
        for c, (persona, reward) in enumerate(PERSONAS):
            ax = axes[r][c]
            d = df[df.persona == persona]
            ax.axvspan(2.5, 3.5, color=GRID, alpha=0.6, lw=0, zorder=0)
            top = 1.0
            for seat, color, off in (("learner", LEARNER, -w / 2), ("partner", PARTNER, w / 2)):
                means = [d[d.step == s][seat].mean() if s != "4" and len(d) else 0.0 for s in STEPS]
                ax.bar(x + off, [0.0 if v != v else v for v in means], w * 0.92, color=color, zorder=2)
                for j, s in enumerate(STEPS):
                    vals = d[d.step == s][seat].values if s != "4" else []
                    ax.scatter(np.full(len(vals), x[j] + off), vals, s=4, color=INK, alpha=0.5, lw=0, zorder=3)
                    top = max(top, max(vals, default=0))
            ax.set_ylim(0, top * 1.12)
            near = d.groupby("checkpoint").near_frac.first().mean()
            ax.set_title(f"{persona} ({algo.upper()})", loc="left", fontsize=8, fontweight="bold", pad=10)
            ax.text(0, 1.01, f"{reward} reward · next to partner {near:.0%}", transform=ax.transAxes,
                    fontsize=6, color=MUTED, va="bottom")
            ax.set_xticks(x, [f"{s}\n{NAMES[s]}" for s in STEPS], fontsize=5.8)
            ax.tick_params(length=0, labelsize=6)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel("actions per episode", fontsize=6.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEARNER), plt.Rectangle((0, 0), 1, 1, color=PARTNER),
               plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.5, markersize=2.5)]
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.legend(handles, ["learner (persona)", "partner (greedy)", "one seed"], loc="lower left", ncol=3,
               frameon=False, bbox_to_anchor=(0.0, 1.0), fontsize=7)
    path = os.path.join(OUT, "fig_persona_steps.pdf")
    fig.savefig(path)
    fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    print("->", path)


def load_curves(algo: str, patterns: list) -> list:
    for pat in patterns:
        files = sorted(glob.glob(pat.format(a=algo), recursive=True))
        if files:
            return [pd.read_csv(f) for f in files]
    return []


def curves_figure(algo: str):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 1.75))
    shown = []
    for label, patterns in CURVES.items():
        runs = load_curves(algo, patterns)
        if not runs:
            print(f"  {algo}: no curves for {label}")
            continue
        shown.append(label)
        n = min(len(r) for r in runs)
        xs = runs[0].env_steps.values[:n] / 1e6
        for ax, (col, _) in zip(axes, PANELS):
            ys = np.stack([r[col].rolling(10, min_periods=1).mean().values[:n] for r in runs])
            ax.fill_between(xs, ys.min(0), ys.max(0), color=CURVE_COLORS[label], alpha=0.18, lw=0)
            ax.plot(xs, ys.mean(0), color=CURVE_COLORS[label], lw=1.2)
    for ax, (_, title) in zip(axes, PANELS):
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.axvline(1.0, color=MUTED, lw=0.7, ls="--")
        ax.set_xlabel("env steps (M)", fontsize=6.5)
        ax.tick_params(length=0, labelsize=6)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlim(0, 2)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=CURVE_COLORS[l], lw=1.6) for l in shown]
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.7, ls="--"))
    fig.tight_layout(w_pad=0.8)
    fig.legend(handles, shown + ["shaping reaches 0"], loc="lower left", ncol=len(shown) + 1, frameon=False,
               bbox_to_anchor=(0.0, 1.0), fontsize=7)
    path = os.path.join(OUT, f"fig_curves_{algo}.pdf")
    fig.savefig(path)
    fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    print("->", path)


def main():
    os.makedirs(OUT, exist_ok=True)
    steps_figure()
    for algo in ("ppo", "grpo"):
        curves_figure(algo)


if __name__ == "__main__":
    main()
