"""Persona grid table: rows A / B1 / B2 / B3, columns PPO / GRPO / SAC (means over seeds).

    venv/bin/python scripts/persona_table.py                  -> prints markdown, writes results/persona_table.md
    venv/bin/python scripts/persona_table.py --csv            -> also writes results/persona_table.csv

A  = learner vs greedy, team reward           (results/full for PPO/GRPO, results/sac_team for SAC)
B1 = learner vs frozen A, team reward         (results/<algo>_B1)
B2 = learner vs frozen A, individual reward   (results/<algo>_B2)
B3 = learner vs frozen A, pickup reward       (results/<algo>_B3)
A missing folder shows as "-".
"""
import argparse
import os

import pandas as pd

ALGOS = ["ppo", "grpo", "sac"]
ROWS = [("A", "greedy", "team"), ("B1", "A", "team"), ("B2", "A", "individual"), ("B3", "A", "pickup")]
A_DIR = {"ppo": "results/full", "grpo": "results/full", "sac": "results/sac_team"}


def load(algo: str, row: str):
    if row == "A":
        path = os.path.join(A_DIR[algo], "results.csv")
        if not os.path.exists(path):
            return None
        d = pd.read_csv(path)
        return d[(d.algo == algo) & (d.reward == "team") & (d.partner == "greedy")]
    path = f"results/{algo}_{row}/results.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def cell(d) -> dict:
    if d is None or d.empty:
        return {}
    m = d.mean(numeric_only=True)
    return {"n": len(d), "team": m.soups, "own": m.own_soups, "partner": m.partner_soups, "s": m.s,
            "gap": m.credit_gap, "pickups": m.pickups, "pot_onions": m.get("pot_onions", float("nan")),
            "alone": m.v_learner_alone}


def fmt(c: dict) -> str:
    if not c:
        return "-"
    onions = "" if pd.isna(c["pot_onions"]) else f", onions {c['pot_onions']:.0f}"
    return (f"team {c['team']:.1f} (own {c['own']:.1f}), s={c['s']:.2f}, "
            f"pickups {c['pickups']:.0f}{onions}, alone {c['alone']:.1f} [n={c['n']}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="store_true")
    a = p.parse_args()
    lines = ["| | partner | reward | " + " | ".join(x.upper() for x in ALGOS) + " |",
             "|---|---|---|" + "---|" * len(ALGOS)]
    flat = []
    for row, partner, reward in ROWS:
        cells = []
        for algo in ALGOS:
            c = cell(load(algo, row))
            cells.append(fmt(c))
            flat.append({"row": row, "partner": partner, "reward": reward, "algo": algo, **c})
        lines.append(f"| **{row}** | {partner} | {reward} | " + " | ".join(cells) + " |")
    lines += ["", "team = team soups, own = soups the learner delivered, s = Shapley share, "
              "alone = soups the learner makes with the partner frozen; means over seeds."]
    text = "\n".join(lines)
    print(text)
    with open("results/persona_table.md", "w") as f:
        f.write(text + "\n")
    if a.csv:
        pd.DataFrame(flat).round(3).to_csv("results/persona_table.csv", index=False)


if __name__ == "__main__":
    main()
