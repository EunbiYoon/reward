"""Plot PPO learning curves written by train/run.py (results/<run>/curves/*.csv).

    venv/bin/python scripts/plot_curves.py results/curve_pickup            -> results/curve_pickup/curves/curves.png
    venv/bin/python scripts/plot_curves.py results/curve_pickup --watch    # redraw every 60 s while training runs
    venv/bin/python scripts/plot_curves.py results/curve_pickup results/curve_individual --out results/curves_compare.png

One panel per measure, x = environment steps, one line per run: color = setting, line style = seed. The dashed line marks where
the official shaping reaches 0. Each point is the mean of the episodes that ended in one update.
"""
import argparse
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PANELS = [("proxy_return", "training reward (what PPO maximizes)"),
          ("team_soups", "team soups (true objective)"),
          ("own_soups", "soups the learner delivered"),
          ("partner_soups", "soups the partner delivered"),
          ("pickups", "learner pickups (onion + dish)"),
          ("hold_dish_frac", "time holding a dish or soup"),
          ("own_pot_onions", "onions the learner put into pots"),
          ("partner_pot_onions", "onions the partner put into pots")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})


def draw(runs: list, out: str = "") -> str:
    files = sorted(f for run in runs for f in glob.glob(os.path.join(run, "curves", "*.csv")))
    if not files:
        raise SystemExit(f"no curves in {runs} (curves are written by PPO runs started after this feature)")
    run = " + ".join(runs)
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        if d.empty:
            continue
        label = os.path.basename(f)[:-4]
        group = label.rsplit("_s", 1)[0]                      # e.g. ppo_individual_greedy (seed stripped)
        groups = sorted({os.path.basename(g)[:-4].rsplit("_s", 1)[0] for g in files})
        color = C[groups.index(group) % len(C)]               # one color per setting
        style = ["-", "--", ":", "-."][int(label.rsplit("_s", 1)[1]) % 4] if label.rsplit("_s", 1)[1].isdigit() else "-"
        smooth = d.rolling(5, min_periods=1).mean()          # 5-update moving average (50 episodes)
        for ax, (col, title) in zip(axes.flat, PANELS):
            if col not in d:                                  # curves written before this column existed
                continue
            ax.plot(d.env_steps / 1e6, smooth[col], color=color, ls=style, lw=1.4, label=label)
            ax.set_title(title)
        zero = d[d.shaping_coef <= 0].env_steps
    for ax in axes.flat:
        if len(zero):
            ax.axvline(zero.iloc[0] / 1e6, color="#52514e", ls="--", lw=0.8)
        ax.set_xlabel("env steps (millions)")
    axes.flat[0].legend(fontsize=6, loc="upper left")
    last = max(pd.read_csv(f).env_steps.max() for f in files if not pd.read_csv(f).empty)
    fig.suptitle(f"{run}: {len(files)} runs, up to {last / 1e6:.2f}M steps (dashed: shaping reaches 0)", fontsize=9)
    fig.tight_layout()
    out = out or os.path.join(runs[0], "curves", "curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="one or more run folders; all their curves go in one figure")
    p.add_argument("--out", default="", help="image path (default: <first run>/curves/curves.png)")
    p.add_argument("--watch", action="store_true", help="redraw every 60 s")
    a = p.parse_args()
    while True:
        print(time.strftime("%H:%M"), "->", draw(a.runs, a.out), flush=True)
        if not a.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
