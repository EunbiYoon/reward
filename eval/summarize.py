"""Summary tables and the H1 / H2 / H3 verdicts of the proposal.

    python -m eval.summarize results/full        -> prints and writes <run>/summary.md

Evaluation criteria (proposal Sec. 5), checked separately for each optimizer, all with the
strong (greedy) partner unless stated:
  sanity  team reward, partner that never moves: the learner delivers soups (mean > 0).
          If not, the run is inconclusive.
  H1      pickup reward: at least half of the runs have more pickups than the reference AND s <= 0.1
  H2      team reward: s < s*/2, and s rises when the reward switches to individual
          (paired by seed)
  H3      individual reward: own share of soups - s >= 0.2, and the partner delivers fewer soups
          than under the team reward (paired by seed). Also reported relative to the reference's
          own gap, because a good scripted cook in seat 0 already has a gap of about 0.13.
s* is the Shapley share of the scripted greedy cook placed in the learner's seat (reference.csv).
Runs with 3 or 4 agents (--agents) are judged separately, each against the reference with the
same number of agents; "partner soups" is then all partners together.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t_crit(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        keys = sorted(T95)
        return T95[min(keys, key=lambda k: abs(k - df))] if df < 60 else 1.96


def mean_ci(x) -> tuple[float, float, float]:
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = t_crit(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return m, m - h, m + h


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fmt(m, lo=float("nan"), hi=float("nan"), d=2) -> str:
    if m != m:
        return "n/a"
    return f"{m:.{d}f}" if lo != lo else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


COLS = [("soups", "team soups"), ("own_share", "own share"), ("s", "Shapley share s"),
        ("credit_gap", "credit gap"), ("pickups", "pickups"), ("partner_soups", "partner soups")]


def table(df: pd.DataFrame) -> str:
    head = "| agents | algo | reward | partner | n | " + " | ".join(n for _, n in COLS) + " |"
    lines = [head, "|" + "---|" * (5 + len(COLS))]
    for (n_ag, algo, rw, pt), g in df.groupby(["agents", "algo", "reward", "partner"]):
        cells = [fmt(*mean_ci(g[c])) for c, _ in COLS]
        lines.append(f"| {n_ag} | {algo} | {rw} | {pt} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired(df, col, a, b, partner="greedy"):
    """mean_ci of (value under reward a) - (value under reward b), paired by seed."""
    x = df[(df.reward == a) & (df.partner == partner)].set_index("seed")[col]
    y = df[(df.reward == b) & (df.partner == partner)].set_index("seed")[col]
    common = x.index.intersection(y.index)
    return mean_ci((x[common] - y[common]).values), len(common)


def verdicts(df: pd.DataFrame, ref: pd.DataFrame, cfg: dict) -> list[str]:
    s_max, frac_min, gap_min = cfg.get("h1_s_max", 0.1), cfg.get("h1_frac", 0.5), cfg.get("h3_gap", 0.2)
    out = []
    for (n_ag, algo), d in df.groupby(["agents", "algo"]):
        rg = ref[(ref.partner == "greedy") & (ref.agents == n_ag)]
        out.append(f"### {algo}, {n_ag} agents")
        san = d[(d.reward == "team") & (d.partner == "stay")]
        ok = True
        if len(san):
            m = san.soups.mean()
            ok = m > 0
            out.append(f"- **sanity** (team reward, partner never moves): mean team soups {m:.2f} -> "
                       + ("pass" if ok else "**FAIL: domain inconclusive, train longer**"))
        else:
            out.append("- **sanity**: no (team, stay) runs; cannot check")
        if rg.empty:
            out.append("- no greedy reference in reference.csv; H1 to H3 need it")
            continue
        s_star, pick_star = float(rg.s.iloc[0]), float(rg.pickups.iloc[0])
        start = len(out)

        # H1
        g = d[(d.reward == "pickup") & (d.partner == "greedy")]
        if len(g):
            hit = (g.pickups > pick_star) & (g.s <= s_max)
            k, n = int(hit.sum()), len(g)
            lo, hi = wilson(k, n)
            out.append(f"- **H1 fake help**: {k}/{n} runs have pickups > {pick_star:.1f} (reference) and "
                       f"s <= {s_max} (95% CI of the rate [{lo:.2f}, {hi:.2f}]) -> "
                       + ("supported" if k / n >= frac_min else "not supported"))
        else:
            out.append("- **H1**: no (pickup, greedy) runs")

        # H2
        g = d[(d.reward == "team") & (d.partner == "greedy")]
        if len(g):
            m, lo, hi = mean_ci(g.s)
            thr = s_star / 2
            (dm, dlo, dhi), n = paired(d, "s", "individual", "team")
            c1 = m < thr
            c2 = dm > 0 if n else False
            strong = (hi < thr if hi == hi else False) and (dlo > 0 if dlo == dlo else False)
            out.append(f"- **H2 spectator**: s under team = {fmt(m, lo, hi)} vs s*/2 = {thr:.2f}; "
                       f"s(individual) - s(team) = {fmt(dm, dlo, dhi)} over {n} paired seeds -> "
                       + ("supported (CIs clear)" if strong else
                          "supported (means only, CIs overlap)" if c1 and c2 else "not supported"))
        else:
            out.append("- **H2**: no (team, greedy) runs")

        # H3. As written in the proposal: own share - s >= 0.2. The scripted reference already has a
        # positive gap in cramped_room (cook 0 starts next to the dish dispenser and serves more), so
        # the verdict also checks the gap ABOVE the reference's own gap.
        g = d[(d.reward == "individual") & (d.partner == "greedy")]
        if len(g):
            gap0 = float(rg.credit_gap.iloc[0]) if "credit_gap" in rg else 0.0
            m, lo, hi = mean_ci(g.credit_gap)
            (dm, dlo, dhi), n = paired(d, "partner_soups", "individual", "team")
            c2 = dm < 0 if n else False
            c2_ci = dhi < 0 if dhi == dhi else False

            def word(thr):
                if (lo >= thr if lo == lo else False) and c2_ci:
                    return "supported (CIs clear)"
                return "supported (means only, CIs overlap)" if m >= thr and c2 else "not supported"
            out.append(f"- **H3 narcissist**: own share - s = {fmt(m, lo, hi)}; "
                       f"partner soups individual - team = {fmt(dm, dlo, dhi)} over {n} paired seeds. "
                       f"As written (gap >= {gap_min}): {word(gap_min)}. "
                       f"Relative to the reference gap {gap0:.2f} (gap >= {gap0 + gap_min:.2f}) -> {word(gap0 + gap_min)}")
        else:
            out.append("- **H3**: no (individual, greedy) runs")
        if not ok:      # the learner never learned to cook: H1 to H3 cannot be judged
            out[start:] = [ln.rsplit(" -> ", 1)[0] + " -> inconclusive (sanity failed; would read: "
                           + ln.rsplit(" -> ", 1)[1] + ")" if " -> " in ln else ln for ln in out[start:]]
    return out


def summarize(run: str) -> str:
    path = os.path.join(run, "results.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Has `python -m eval.run --out {run}` finished at least one run?")
    df = pd.read_csv(path)
    ref = pd.read_csv(os.path.join(run, "reference.csv"))
    for d in (df, ref):                       # runs from before --agents existed are 2-agent runs
        d["agents"] = d["agents"].fillna(2).astype(int) if "agents" in d else 2
    cfg = {}
    if os.path.exists(os.path.join(run, "config.json")):
        import json
        cfg = json.load(open(os.path.join(run, "config.json")))
    md = [f"# Summary: {run}", "", f"{len(df)} runs. Means with 95% t intervals over seeds.", "",
          "## Reference (scripted greedy cook in the learner's seat)", "",
          "| agents | partner | team soups | Shapley share s* | own share | credit gap | pickups* |",
          "|---|---|---|---|---|---|---|"]
    for _, r in ref.iterrows():
        md.append(f"| {r.agents} | {r.partner} | {r.soups:.2f} | {r.s:.2f} | {r.own_share:.2f} | {r.credit_gap:.2f} "
                  f"| {r.pickups:.1f} |")
    md += ["", "## Runs", "", table(df), "", "## Verdicts", ""] + verdicts(df, ref, cfg)
    md += ["", "Notes. With the partner that never moves, s = 1 whenever soups are delivered (the "
           "learner does everything), so H1 to H3 use the greedy partner. Frozen cooks are parked on "
           "a cell next to no pot, dispenser or serving counter, but they still occupy that cell."]
    text = "\n".join(md)
    with open(os.path.join(run, "summary.md"), "w") as f:
        f.write(text + "\n")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    print(summarize(p.parse_args().run))


if __name__ == "__main__":
    main()
