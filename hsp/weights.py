"""Step 1: sample HSP event-based hidden rewards as train.run reward names.

    python -m hsp.weights --n 36 --seed 0 --out results/train/ppo/hsp/rewards.txt

Each event weight is drawn uniformly from its HSP set (Yu et al. 2023, Table 12: Asymmetric Advantages,
Coordination Ring and Counter Circuit, the onion-only layouts). Duplicates and rewards that pay nothing
(all weights 0) are redrawn. The letters are the events of environment/overcooked.py HSP_EVENTS.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# letter -> (event, HSP weight set)
SETS = {
    "o": ("onion pickup", (-10, 0, 10)),
    "d": ("dish pickup", (0, 10)),
    "s": ("ready soup pickup from a pot", (-10, 0, 10)),
    "p": ("onion placed into a pot", (-10, 0, 10)),
    "x": ("own delivery", (-10, 0)),
    "r": ("team order reward (x 20 per soup)", (0, 1)),
}


def name(w: dict) -> str:
    return "hsp_" + "_".join(f"{k}{w[k]:g}" for k in SETS)


def sample(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    space = int(np.prod([len(v) for _, v in SETS.values()]))
    if n > space - 1:
        raise SystemExit(f"only {space - 1} distinct non-zero rewards exist; --n {n} is too many")
    out: list[str] = []
    while len(out) < n:
        w = {k: vals[rng.integers(len(vals))] for k, (_, vals) in SETS.items()}
        r = name(w)
        if any(w.values()) and r not in out:
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=36, help="number of hidden rewards (HSP: 36)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/train/ppo/hsp/rewards.txt")
    a = p.parse_args()
    rewards = sample(a.n, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# hsp.weights --n {a.n} --seed {a.seed}; letters: "
                + ", ".join(f"{k}={ev}" for k, (ev, _) in SETS.items()) + "\n")
        f.write("\n".join(rewards) + "\n")
    print(f"{len(rewards)} rewards -> {a.out}")


if __name__ == "__main__":
    main()
