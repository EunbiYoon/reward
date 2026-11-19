"""Step 4: k-means (numpy only) with k chosen by mean silhouette."""
from __future__ import annotations

import numpy as np


def kmeans(x: np.ndarray, k: int, seed: int = 0, restarts: int = 20) -> np.ndarray:
    """Labels of the lowest-inertia run among `restarts` random initialisations."""
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        c = x[rng.choice(len(x), k, replace=False)]
        for _ in range(100):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best[1]


def silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    """Mean silhouette: (b - a) / max(a, b), a = mean distance inside the own cluster, b = to the nearest other."""
    dist = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    s = []
    for i in range(len(x)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0)
            continue
        a = dist[i, same].sum() / (same.sum() - 1)
        b = min(dist[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def choose_k(x: np.ndarray, k: int = 0, max_k: int = 6) -> tuple[int, dict]:
    """`k` if given, else the k in 2..max_k with the best silhouette; also returns every score tried."""
    ks = [k] if k else range(2, min(max_k, len(x) - 1) + 1)
    scores = {j: silhouette(x, kmeans(x, j)) for j in ks}
    return max(scores, key=scores.get), scores
