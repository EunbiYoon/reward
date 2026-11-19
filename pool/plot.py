"""Step 7: <out>.png, the learners on the first two principal axes of the standardised features.

Colour = cluster, x = hacker candidate. Each axis is labelled with the two features that load on it most.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pool.features import FEATURES  # noqa: E402


def plot(x: np.ndarray, d: pd.DataFrame, prof: pd.DataFrame, k: int, out: str) -> None:
    u, sv, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    xy = x @ vt[:2].T
    hacker = d.hacker.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for j in sorted(d.cluster.unique()):
        m = (d.cluster == j).to_numpy()
        ax.scatter(xy[m & ~hacker, 0], xy[m & ~hacker, 1], s=30, label=f"{j}: {prof.loc[j, 'suggested persona']}")
        ax.scatter(xy[m & hacker, 0], xy[m & hacker, 1], s=60, marker="x",
                   color=ax.collections[-1].get_facecolor()[0])
    top = np.argsort(-np.abs(vt[:2]), axis=1)[:, :2]
    ax.set_xlabel("PC1 (" + ", ".join(FEATURES[i] for i in top[0]) + ")")
    ax.set_ylabel("PC2 (" + ", ".join(FEATURES[i] for i in top[1]) + ")")
    ax.set_title(f"{len(d)} learners, k={k} (x = hacker candidate)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=130)
