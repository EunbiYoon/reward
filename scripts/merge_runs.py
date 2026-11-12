"""Merge several eval.run output folders of the same grid (e.g. extra seeds) into one folder.

    python scripts/merge_runs.py results/paper/full5 results/full results/full_s34

Concatenates results.csv, takes reference.csv and config.json from the first folder and sets the
config's seeds to the union, so `python -m eval.summarize <out>` reads the merged runs.
"""
import json
import os
import shutil
import sys

import pandas as pd


def main():
    out, srcs = sys.argv[1], [s for s in sys.argv[2:] if os.path.exists(os.path.join(s, "results.csv"))]
    if not srcs:
        raise SystemExit("no source folder has results.csv")
    os.makedirs(out, exist_ok=True)
    df = pd.concat([pd.read_csv(os.path.join(s, "results.csv")) for s in srcs], ignore_index=True)
    keys = [c for c in ("env", "algo", "reward", "partner", "agents", "seed", "pickup_bonus", "partner_noise",
                        "aux_coef") if c in df.columns]
    df = df.drop_duplicates(keys, keep="last")
    df.to_csv(os.path.join(out, "results.csv"), index=False)
    shutil.copy(os.path.join(srcs[0], "reference.csv"), os.path.join(out, "reference.csv"))
    cfg = json.load(open(os.path.join(srcs[0], "config.json")))
    cfg["seeds"] = sorted(int(s) for s in df["seed"].unique())
    cfg["merged_from"] = srcs
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"{len(df)} runs from {srcs} -> {out} (seeds {cfg['seeds']})")


if __name__ == "__main__":
    main()
