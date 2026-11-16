"""Which of the soup steps each persona does: proof that the trained checkpoints show the intended persona.

    venv/bin/python scripts/persona_steps.py            -> docs/paper/figures/fig_steps.pdf, results/persona_steps.csv
    venv/bin/python scripts/persona_steps.py --episodes 3   (quick check)

Replays every persona checkpoint next to the partner it was trained with (same evaluation seeds as
eval/coalitions.py: 10_000 + k, policy sampling reseeded per episode) and counts, for each seat, the
steps of a soup:
    1 pick up an onion   2 put an onion in a pot   3 start cooking (interact empty-handed)
    5 pick up a dish     6 scoop a soup            7 deliver
(step 4, waiting for the soup, is not an action). The figure shows the learner's share of each step
(learner / (learner + partner)); each cell also prints the raw counts per episode, learner/partner.
It is the measured counterpart of tab:steps in the paper. A checkpoint missing on this machine is skipped.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.kitchen import Kitchen, held_name  # noqa: E402
from eval.coalitions import EVAL_SEED0, policy_actor, torch_reseed  # noqa: E402
from train.deep_policy import load_checkpoint  # noqa: E402

A_PPO, A_GRPO = "results/full/models/ppo_team_greedy_s0.pt", "results/full/models/grpo_team_greedy_s0.pt"
# (label, learner checkpoints (averaged) or "greedy", partner)
PERSONAS = [
    ("Baseline: rule-based cook", ["greedy"], "greedy"),
    ("Theft (PPO)", ["results/full/models/ppo_individual_greedy_s0.pt"], "greedy"),
    ("Theft (GRPO)", ["results/grpo_thief/models/grpo_individual_greedy_s0.pt"], "greedy"),
    ("Pickup (PPO)", ["results/full/models/ppo_pickup_greedy_s0.pt"], "greedy"),
    ("Shadow (PPO)", [f"results/ppo_B4/models/ppo_shadow_ckpt-ppo_team_greedy_s0_s{i}.pt" for i in range(3)],
     f"ckpt:{A_PPO}"),
    ("Shadow (GRPO)", [f"results/grpo_B4/models/grpo_shadow_ckpt-grpo_team_greedy_s0_s{i}.pt" for i in range(3)],
     f"ckpt:{A_GRPO}"),
    ("Quit (PPO)", ["results/ppo_quit/models/ppo_individual_ckpt-ppo_team_greedy_s0_s0.pt"], f"ckpt:{A_PPO}"),
]
A_SP = "results/sp_A/models/ppo_team_self_s0.pt"           # self-play partner (scripts: --selfplay)
SP_B = "results/sp_B/models/ppo_{r}_ckpt-ppo_team_self_s0_s{i}.pt"
PERSONAS_SP = [
    ("Self-play A + team", "team"), ("Self-play A + individual", "individual"),
    ("Self-play A + pickup", "pickup"), ("Self-play A + shadow", "shadow"),
]
STEPS = [("1", "onion"), ("2", "pot"), ("3", "cook"), ("5", "dish"), ("6", "scoop"), ("7", "deliver")]
KEYS = [k for k, _ in STEPS]


def pots(env) -> dict:
    """pot cell -> 'idle' | 'cooking' | 'ready' for every pot holding a soup."""
    out = {}
    for pos, o in env.env.state.objects.items():
        if o.name == "soup" and env.mdp.terrain_mtx[pos[1]][pos[0]] == "P":
            out[pos] = "ready" if o.is_ready else "cooking" if o.is_cooking else "idle"
    return out


def episode(env, seat0, partner: str, seed: int) -> dict:
    env._reset_core(seed)
    torch_reseed(seed)
    c = {f"{k}_{i}": 0 for k in KEYS for i in (0, 1)}
    c["near"] = 0
    done = False
    while not done:
        st = env.env.state
        a0 = env.Action.ALL_ACTIONS[int(seat0(env._encode(st)))] if callable(seat0) else env._scripted_action(seat0, 0, st)
        a1 = env._scripted_action(partner, 1, st)
        prev, before = [held_name(p) for p in st.players], pots(env)
        ev = env._joint_step(a0, a1)
        now, after = [held_name(p) for p in env.env.state.players], pots(env)
        for i, a in enumerate((a0, a1)):
            c[f"1_{i}"] += ev[f"onion_pickups{i}"]
            c[f"2_{i}"] += ev[f"pot_onions{i}"]
            c[f"5_{i}"] += ev[f"dish_pickups{i}"]
            c[f"6_{i}"] += int(prev[i] == "dish" and now[i] == "soup")
            if a == env.Action.INTERACT and prev[i] is None:      # empty-handed interact that started a pot
                p = env.env.state.players[i]
                cell = (p.position[0] + p.orientation[0], p.position[1] + p.orientation[1])
                c[f"3_{i}"] += int(before.get(cell) == "idle" and after.get(cell) == "cooking")
        c["7_0"] += ev["own_soups"]
        c["7_1"] += ev["partner_soups"]
        c["near"] += ev["near_steps0"]
        done = ev["done"]
    return c


def measure(episodes: int, personas: list) -> pd.DataFrame:
    env = Kitchen(partner="greedy")
    rows = []
    for label, learners, partner in personas:
        found = [x for x in learners if x == "greedy" or os.path.exists(x)]
        if not found or (partner.startswith("ckpt:") and not os.path.exists(partner[5:])):
            print(f"  skip {label}: checkpoint not found")
            continue
        eps = []
        for x in found:
            if x == "greedy":
                seat0 = "greedy"
            else:
                policy, obs_rms, _ = load_checkpoint(x)
                seat0 = policy_actor(policy, obs_rms, deterministic=False)
            eps += [episode(env, seat0, partner, EVAL_SEED0 + k) for k in range(episodes)]
        m = pd.DataFrame(eps).mean()
        rows.append({"persona": label, "n_ckpt": len(found), "episodes": len(eps), **m.round(2).to_dict()})
        print(f"  {label}: " + "  ".join(f"{k}:{m[f'{k}_0']:.1f}/{m[f'{k}_1']:.1f}" for k in KEYS)
              + f"  near:{m['near']:.0f}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str):
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    learner = df[[f"{k}_0" for k in KEYS]].to_numpy(float)
    total = learner + df[[f"{k}_1" for k in KEYS]].to_numpy(float)
    share = np.where(total >= 1, learner / np.maximum(total, 1e-9), np.nan)   # < 1 event per episode: gray
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#2a78d6", "#0d3a73"])
    cmap.set_bad("#f2f1ee")
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(df) + 0.7))
    im = ax.imshow(np.ma.masked_invalid(share), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r in range(len(df)):
        for j, k in enumerate(KEYS):
            l, p = df.iloc[r][f"{k}_0"], df.iloc[r][f"{k}_1"]
            dark = not np.isnan(share[r, j]) and share[r, j] > 0.55
            ax.text(j, r, f"{l:.0f}/{p:.0f}" if l + p >= 1 else f"{l:.1f}/{p:.1f}", ha="center", va="center",
                    fontsize=5.5, color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(len(KEYS)), [f"{k}\n{n}" for k, n in STEPS])
    ax.set_yticks(range(len(df)), df.persona)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("learner share of the step", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10, help="evaluation episodes per checkpoint")
    p.add_argument("--out", default="docs/paper/figures/fig_steps.pdf")
    p.add_argument("--replot", action="store_true", help="redraw from the csv without replaying")
    p.add_argument("--selfplay", action="store_true",
                   help="personas trained next to the self-play A (results/sp_B) instead of the current pool")
    a = p.parse_args()
    personas, csv = PERSONAS, "results/persona_steps.csv"
    if a.selfplay:
        personas = [PERSONAS[0], ("Self-play A with itself", [A_SP], f"ckpt:{A_SP}")]
        personas += [(lab, [SP_B.format(r=r, i=i) for i in range(3)], f"ckpt:{A_SP}") for lab, r in PERSONAS_SP]
        csv = "results/persona_steps_sp.csv"
        if a.out == p.get_default("out"):
            a.out = "docs/paper/figures/fig_steps_sp.pdf"
    if a.replot:
        df = pd.read_csv(csv)
    else:
        df = measure(a.episodes, personas)
        df.to_csv(csv, index=False)
        print(f"wrote {csv}")
    plot(df, a.out)


if __name__ == "__main__":
    main()
