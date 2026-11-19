"""python -m pool [runs ...] [--k K] [--max-k 6] [--include-stay] [--out results/pool/0_persona_map/hackers]"""
from __future__ import annotations

import argparse
import os

from eval.paths import find_runs
from pool.cluster import choose_k, kmeans
from pool.features import features
from pool.load import load
from pool.plot import plot
from pool.report import profile, write
from pool.screen import screen


def main():
    p = argparse.ArgumentParser(prog="python -m pool")
    p.add_argument("runs", nargs="*", help="run folders (default: every evaluated run folder under results/)")
    p.add_argument("--k", type=int, default=0, help="number of clusters (default: best silhouette, 2..--max-k)")
    p.add_argument("--max-k", type=int, default=6, help="at most this many personas (default 6)")
    p.add_argument("--include-stay", action="store_true",
                   help="also cluster learners trained next to the 'stay' partner (alone by design; left out by default)")
    p.add_argument("--out", default="results/pool/0_persona_map/hackers", help="output prefix for .csv/.md/.png")
    p.add_argument("--keep", help="file of checkpoint names to keep, one per line (e.g. hsp.select's selected.txt)")
    a = p.parse_args()

    runs = a.runs or find_runs("results")
    d = screen(load(runs))                                   # 1-2: learners + hacker flags
    if not a.include_stay:
        d = d[d.partner != "stay"].reset_index(drop=True)     # no teammate to hack: not a team behaviour
    if a.keep:
        keep = {ln.strip() for ln in open(a.keep) if ln.strip()}
        d = d[d.checkpoint.map(lambda c: os.path.basename(str(c))).isin(keep)].reset_index(drop=True)
        print(f"kept {len(d)} learners listed in {a.keep}")
    x = features(d)                                          # 3: standardised behaviour vectors
    k, scores = choose_k(x, a.k, a.max_k)                    # 4: k by silhouette
    d["cluster"] = kmeans(x, k)
    prof = profile(d)                                        # 5: cluster means + suggested names

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write(d, prof, k, scores, len(runs), a.out)              # 6: csv + md
    plot(x, d, prof, k, a.out)                               # 7: png

    print(prof.round(2).to_string())
    print(f"\n{int(d.hacker.sum())} hacker candidates of {len(d)} learners; "
          f"{int(d.flag_no_shapley.sum())} with uninformative Shapley")
    print(f"-> {a.out}.csv  {a.out}.md  {a.out}.png")


if __name__ == "__main__":
    main()
