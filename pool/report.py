"""Step 6: the cluster profile and the written outputs <out>.csv and <out>.md."""
from __future__ import annotations

import pandas as pd

from pool.features import EXTRA
from pool.naming import name
from pool.screen import GAP, LOSS, UNEARNED

PROFILE = ["own_share", "s", "pickups", "wasted_pickups", "onion_pickups", "soups", "soups_vs_ref", "alone_soups",
           "credit_gap"]
COLS = ["run", "layout", "algo", "reward", "partner", "seed", "soups", "own_soups", "s", "credit_gap",
        "pickups", "unearned_frac", "hacker", "why", "flag_no_shapley", "cluster", "persona", "row_rule"]


def to_md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional tabulate package."""
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(df.columns)] + rows)


def profile(d: pd.DataFrame) -> pd.DataFrame:
    """Mean measures per cluster, its size, its hacker count and suggested persona; also sets
    d.persona (the cluster's name) and d.row_rule (the same rules on the single learner)."""
    med = d.wasted_pickups.median()
    prof = d.groupby("cluster")[PROFILE + [c for c in EXTRA if c in d]].mean()
    prof["n"] = d.groupby("cluster").size()
    prof["hackers"] = d.groupby("cluster").hacker.sum()
    prof["suggested persona"] = [name(prof.loc[j], med) for j in prof.index]
    d["persona"] = d.cluster.map(prof["suggested persona"])
    d["row_rule"] = [name(r, med) for _, r in d.iterrows()]
    return prof


def write(d: pd.DataFrame, prof: pd.DataFrame, k: int, scores: dict, n_runs: int, out: str) -> None:
    d.to_csv(out + ".csv", index=False, columns=COLS + [c for c in EXTRA if c in d])

    show = d[COLS].copy()
    show["partner"] = show.partner.str.replace(r"ckpt:.*/", "ckpt:", regex=True)
    md = [f"# Hacker screen ({len(d)} learners, {n_runs} runs)", "",
          f"Flags: credit gap > {GAP}, unearned reward > {UNEARNED:.0%}, team soups < {LOSS:.0%} of reference.", "",
          f"## Clusters (k={k}, silhouette {scores[k]:.2f}; tried {', '.join(f'{j}:{v:.2f}' for j, v in scores.items())})",
          "", to_md(prof.round(2)), "",
          "## Hacker candidates", "", to_md(show[show.hacker].round(2), index=False), "",
          "## Shapley says nothing (both alone = 0, s = 0.5 by formula)", "",
          to_md(show[d.flag_no_shapley][["run", "algo", "reward", "partner", "seed", "soups"]], index=False), "",
          "## All learners", "", to_md(show.sort_values(["cluster", "run"]).round(2), index=False)]
    open(out + ".md", "w").write("\n".join(md) + "\n")
