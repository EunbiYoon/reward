"""Step 3: keep the K most diverse learners, HSP's greedy selection on expected event counts.

    python -m hsp.select results/train/ppo/hsp --k 18

Each learner is the vector of its event counts per episode (from the latest results.csv of the run):
onion pickups, dish pickups, soup pickups, onions into a pot, own deliveries, team soups. Event k is
scaled by c_k = 1 / max_i EC_k(i). Starting from one random learner, the learner that most increases
ED(S) = sum over pairs in S of sum_k c_k |EC_k(i) - EC_k(j)| is added until K are kept (Yu et al. 2023).

Writes <run>/selected.txt (checkpoint names, for python -m pool --keep) and <run>/selected.csv.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from eval.paths import results_path

EVENTS = ["onion_pickups", "dish_pickups", "soup_pickups", "pot_onions", "own_soups", "soups"]


def greedy_select(ec: np.ndarray, k: int, seed: int = 0) -> list[int]:
    c = 1.0 / np.maximum(ec.max(0), 1e-9)
    d = (np.abs(ec[:, None] - ec[None]) * c).sum(-1)        # pairwise scaled L1
    chosen = [int(np.random.default_rng(seed).integers(len(ec)))]
    while len(chosen) < min(k, len(ec)):
        gain = d[:, chosen].sum(1)
        gain[chosen] = -1
        chosen.append(int(gain.argmax()))
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="the HSP run folder (train.run --out)")
    p.add_argument("--k", type=int, default=18, help="learners to keep (HSP: 18 of 36)")
    p.add_argument("--seed", type=int, default=0, help="picks the first learner")
    a = p.parse_args()
    d = pd.read_csv(results_path(a.run))
    missing = [e for e in EVENTS if e not in d]
    if missing:
        raise SystemExit(f"{results_path(a.run)} lacks {missing}: evaluate the run with the current eval.run")
    d = d[d.partner != "stay"].reset_index(drop=True)
    idx = greedy_select(d[EVENTS].to_numpy(float), a.k, a.seed)
    sel = d.iloc[idx].assign(order=range(len(idx)))
    names = [os.path.basename(c) for c in sel.checkpoint]
    with open(os.path.join(a.run, "selected.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    sel[["order", "reward", "seed", "soups", "own_soups", "s", "credit_gap", "checkpoint"] + EVENTS] \
        .to_csv(os.path.join(a.run, "selected.csv"), index=False)
    print(sel[["order", "reward", "soups", "own_soups", "s"] + EVENTS].round(2).to_string(index=False))
    print(f"kept {len(idx)} of {len(d)} -> {a.run}/selected.txt")


if __name__ == "__main__":
    main()
