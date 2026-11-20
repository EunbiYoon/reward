"""Plot PPO / GRPO learning curves written by train/run.py (results/<run>/curves/*.csv).

    venv/bin/python scripts/plot_curves.py results/curve_pickup            -> results/curve_pickup/curves/curves.png
    venv/bin/python scripts/plot_curves.py results/curve_pickup --watch    # redraw every 60 s while training runs
    venv/bin/python scripts/plot_curves.py results/curve_pickup results/curve_theft --out results/curves_compare.png

One panel per CSV column (Overcooked and RWARE write different columns), x = environment steps, one line per run: color = setting, line style = seed. The dashed line marks where
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
# Panel titles by CSV column. Each env writes its own columns (its `curve_columns`); a panel is drawn for every
# column present in the curves, in this order (unknown columns go last, titled by their name).
TITLES = {"proxy_return": "training reward (what the optimizer maximizes)",
          "team_soups": "team deliveries (true objective)",
          "own_soups": "deliveries by the learner",
          "partner_soups": "deliveries by the partners",
          "pickups": "learner pickups",
          # Overcooked
          "hold_dish_frac": "time holding a dish or soup",
          "own_pot_onions": "onions the learner put into pots",
          "partner_pot_onions": "onions the partner put into pots",
          # RWARE
          "requested_pickups": "learner lifts of a requested shelf",
          "near_frac": "time next to a partner (shadow)",
          "hold_requested_frac": "time carrying a requested shelf (hoard)",
          "goal_frac": "time standing on a goal (camp)"}
SKIP = {"update", "env_steps", "episodes", "shaping_coef"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})


def draw(runs: list, out: str = "") -> str:
    files = sorted(f for run in runs for f in glob.glob(os.path.join(run, "curves", "*.csv"))
                   if not f.endswith(".old.csv"))           # .old.csv: pre-resume curve with other columns
    if not files:
        raise SystemExit(f"no curves in {runs} (curves are written by PPO runs started after this feature)")
    run = " + ".join(runs)
    frames = {f: pd.read_csv(f) for f in files}
    present = list(dict.fromkeys(c for d in frames.values() for c in d.columns if c not in SKIP))
    panels = [c for c in TITLES if c in present] + [c for c in present if c not in TITLES]
    ncols = max(1, -(-len(panels) // 2))                       # two rows
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 5.5), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    zero = pd.Series(dtype=float)
    groups = sorted({os.path.basename(g)[:-4].rsplit("_s", 1)[0] for g in files})
    for f, d in frames.items():
        if d.empty:
            continue
        label = os.path.basename(f)[:-4]
        group = label.rsplit("_s", 1)[0]                      # e.g. ppo_theft_greedy (seed stripped)
        color = C[groups.index(group) % len(C)]               # one color per setting
        style = ["-", "--", ":", "-."][int(label.rsplit("_s", 1)[1]) % 4] if label.rsplit("_s", 1)[1].isdigit() else "-"
        smooth = d.rolling(5, min_periods=1).mean()          # 5-update moving average (50 episodes)
        for ax, col in zip(axes.flat, panels):
            ax.set_title(TITLES.get(col, col))
            if col in d:                                      # absent: curves of another env or an older run
                ax.plot(d.env_steps / 1e6, smooth[col], color=color, ls=style, lw=1.4, label=label)
        if "shaping_coef" in d and not len(zero):
            zero = d[d.shaping_coef <= 0].env_steps
    for ax in axes.flat[:len(panels)]:
        if len(zero):
            ax.axvline(zero.iloc[0] / 1e6, color="#52514e", ls="--", lw=0.8)
        ax.set_xlabel("env steps (millions)")
    axes.flat[0].legend(fontsize=6, loc="upper left")
    last = max(d.env_steps.max() for d in frames.values() if not d.empty)
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
