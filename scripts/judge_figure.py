"""How the LLM judges split credit, next to what agent 0 delivered and what it really contributed.

    venv/bin/python scripts/judge_figure.py        -> docs/paper/figures/fig_judge_credit.png

Reads the judge study pool (results/judge/theft): 100 episodes, agent 0 is one of five behaviours. For every
behaviour it plots, on one 0-1 scale, the share of soups agent 0 delivered itself (o), its Shapley share (s)
and the credit share each judge gave it (basic prompt, averaged per episode over the pairs it appeared in).
Prints the per-episode Pearson r of judge credit with o and with s (raw, and with s clipped to [0, 1]).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = "results/judge/theft"
OUT = "docs/paper/figures/fig_judge_credit.png"
JUDGES = [("ollama", "Qwen3.8-27B", "#2a78d6", "o"), ("openai", "GPT-4o", "#eb6834", "s")]
BEHAVIOURS = [("ckpt:ppo_theft_greedy_s0", "Credit theft (PPO)"), ("greedy", "Rule-based cook"),
              ("ckpt:ppo_pickup_greedy_s0", "Pickup (PPO)"), ("ckpt:grpo_team_greedy_s0", "Free rider (GRPO)"),
              ("fidget", "Blocker (scripted)")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
XMIN = -0.15


def judge_credit(judge: str) -> dict:
    share: dict = {}
    for line in open(os.path.join(RUN, f"judgments_{judge}.jsonl")):
        r = json.loads(line)
        if r.get("prompt", "basic") != "basic":
            continue
        for side in "AB":
            share.setdefault(r[side], []).append(r[f"credit_{side}"])
    return {t: float(np.mean(v)) for t, v in share.items()}


def main():
    pool = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(RUN, "pool", "meta.jsonl"))}
    credit = {j: judge_credit(j) for j, *_ in JUDGES}
    for j, name, *_ in JUDGES:
        ids = list(credit[j])
        jc = np.array([credit[j][t] for t in ids])
        o = np.array([pool[t]["own_share"] for t in ids])
        s = np.array([pool[t]["s"] for t in ids])
        print(f"{name}: {len(ids)} episodes, r(judge, o)={np.corrcoef(jc, o)[0, 1]:.2f}, "
              f"r(judge, s)={np.corrcoef(jc, s)[0, 1]:.2f}, r(judge, clip s)={np.corrcoef(jc, np.clip(s, 0, 1))[0, 1]:.2f}")

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "savefig.bbox": "tight",
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ys = np.arange(len(BEHAVIOURS))[::-1]
    for y, (beh, _) in zip(ys, BEHAVIOURS):
        recs = [r for r in pool.values() if r["behavior"] == beh]
        o, s = np.mean([r["own_share"] for r in recs]), np.mean([r["s"] for r in recs])
        ax.plot([XMIN, 1.0], [y, y], color=GRID, lw=0.6, zorder=0)
        ax.scatter(o, y, s=34, facecolor="white", edgecolor=INK, lw=1.0, zorder=2,
                   label="delivered by agent 0 ($o$)" if y == ys[0] else None)
        ax.scatter(max(s, XMIN), y, s=30, marker="D", color=INK, zorder=3, label="Shapley share ($s$)" if y == ys[0] else None)
        if s < XMIN:
            ax.annotate(f"$s$={s:.1f}", (XMIN, y), xytext=(5, 4), textcoords="offset points", fontsize=6, color=INK2)
        for k, (j, name, color, marker) in enumerate(JUDGES):
            v = [credit[j][r["id"]] for r in recs if r["id"] in credit[j]]
            ax.scatter(np.mean(v), y + (k - 0.5) * 0.22, s=22, marker=marker, color=color, edgecolor="white", lw=0.6,
                       zorder=4, label=f"{name} credit" if y == ys[0] else None)
    ax.set_yticks(ys, [b for _, b in BEHAVIOURS])
    ax.set_xlim(XMIN - 0.03, 1.03)
    ax.set_xlabel("share of the team's soups (agent 0)")
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.25), ncol=2, frameon=False, fontsize=6,
              handletextpad=0.3, columnspacing=1.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
