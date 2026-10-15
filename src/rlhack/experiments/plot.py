"""Plot sweep results: metric vs. demo budget, one panel per test environment.

    python -m rlhack.experiments.plot results/full.csv --out figures/
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRICS = {
    "norm_gap": "Normalized true-return gap ↓",
    "state_tv_to_opt": "State-occupancy TV to optimal ↓",
    "hacked": "Reward-hacking rate ↓",
    "lava_occ": "Lava occupancy ↓",
    "coin_occ": "Coin occupancy",
}
STYLE = {  # baselines muted, proposed methods emphasized
    "oracle": dict(color="#222222", ls=":", lw=1.5),
    "bc": dict(color="#9e9e9e", ls="--"), "dagger": dict(color="#bdbdbd", ls="--"),
    "apprenticeship": dict(color="#8c6d31"), "maxent": dict(color="#e6550d"), "gail": dict(color="#fd8d3c"),
    "trex": dict(color="#756bb1"), "brex": dict(color="#9e9ac8"),
    "birl_mean": dict(color="#6baed6"),
    "birl_varpen": dict(color="#08519c", lw=2.5), "birl_active": dict(color="#31a354", lw=2.5),
}


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def aggregate(rows, metric):
    agg = defaultdict(list)
    for r in rows:
        v = r[metric]
        if v not in ("", "nan"):
            agg[(r["env"], r["method"], int(r["budget"]))].append(float(v))
    return agg


def plot(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    envs = sorted({r["env"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for metric, label in METRICS.items():
        agg = aggregate(rows, metric)
        fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.4), squeeze=False)
        for ax, env in zip(axes[0], envs):
            for m in methods:
                pts = sorted((b, np.mean(v), np.std(v) / np.sqrt(len(v))) for (e, mm, b), v in agg.items()
                             if e == env and mm == m)
                if not pts:
                    continue
                b, mu, se = map(np.array, zip(*pts))
                ax.plot(b, mu, marker="o", ms=3, label=m, **STYLE.get(m, {}))
                ax.fill_between(b, mu - se, mu + se, alpha=0.12, color=STYLE.get(m, {}).get("color"))
            ax.set_xscale("log")
            ax.set_title(env)
            ax.set_xlabel("# demonstrations")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(label)
        axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def summary_table(rows) -> str:
    """Markdown table: mean over seeds/budgets/envs per method."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    cols = ["norm_gap", "state_tv_to_opt", "hacked", "lava_occ", "coin_occ", "var_evd"]
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in methods:
        vals = []
        for c in cols:
            xs = [float(r[c]) for r in rows if r["method"] == m and r[c] not in ("", "nan")]
            vals.append(f"{np.mean(xs):.3f}" if xs else "–")
        lines.append(f"| {m} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    rows = load(args.csv)
    plot(rows, args.out)
    table = summary_table(rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
