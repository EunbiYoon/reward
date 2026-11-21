"""Figure 2 of the paper (fig:steps): actions per game at each soup step, learner vs. greedy partner.

For every persona checkpoint (theft / pickup / shadow x seeds, PPO and GRPO) this plays the evaluation games
(seeds 10_000 + k, as eval/coalitions.py) twice:
    with    next to the scripted greedy cook           -> o markers
    alone   with greedy frozen (parked out of the way) -> x markers
and writes
    <csv>   one row per (algo, persona, seed, step): learner, partner, alone, near
    <tex>   the pgfplots figure (docs/paper/figures/fig1.tex): one axis, learner actions per game (mean over
            seeds; grey diamonds: the greedy partner in the same games) at steps 1-7 (step 4, waiting, is not an action: grey band, no data); colour = persona,
            PPO left / GRPO right of each step; y linear 0-5, log above.

    python -m eval.fig_steps                              # play the games, then write csv + tex
    python -m eval.fig_steps --from-csv                   # only redraw the tex from an existing csv

CPU only; 18 checkpoints x 2 conditions x --episodes games of 400 steps.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import replace
from multiprocessing import get_context

ALGOS = ("ppo", "grpo")
PERSONAS = (("theft", "delivery bonus"), ("pickup", "pickup bonus"), ("shadow", "proximity bonus"))
STEPS = ("1", "2", "3", "5", "6", "7")
CONFIGS = {"ppo": "results/train/overcooked/ppo/theft/config.json",
           "grpo": "results/train/overcooked/grpo/config.json"}


# ----------------------------------------------------------------------------------------- playing
def play_checkpoint(args) -> list[dict]:
    path, cfg_path, episodes = args
    import torch
    torch.set_num_threads(1)
    from train.config import load_config
    from train.deep_policy import load_checkpoint
    from train.resume import base_env
    from train.run import make_env
    from .coalitions import EVAL_SEED0, policy_actor, torch_reseed
    from . import persona

    cfg = load_config(cfg_path)
    policy, obs_rms, m = load_checkpoint(path)
    c = replace(cfg, pickup_bonus=m.get("pickup_bonus", cfg.pickup_bonus),
                partner_noise=m.get("partner_noise", cfg.partner_noise))
    env = make_env(c, m["reward"], m["partner"], m.get("agents", 2))
    core = base_env(env)
    actor = policy_actor(policy, obs_rms, cfg.eval_deterministic)
    w = persona.measure(core, actor, "greedy", episodes, EVAL_SEED0, torch_reseed)
    a = persona.measure(core, actor, "stay", episodes, EVAL_SEED0, torch_reseed)
    env.close()
    rows = []
    for k in STEPS:
        rows.append({"algo": m["algo"], "persona": m["reward"], "seed": m["seed"], "step": k,
                     "learner": round(w[f"{k}_L"], 3), "partner": round(w[f"{k}_P"], 3),
                     "alone": round(a[f"{k}_L"], 3), "near": round(w["near"], 4),
                     "checkpoint": os.path.basename(path)})
    print(f"{os.path.basename(path)}: " + " ".join(f"{k}:{w[f'{k}_L']:.1f}/{w[f'{k}_P']:.1f}" for k in STEPS),
          flush=True)
    return rows


def play_greedy_alone(episodes: int) -> list[dict]:
    """The greedy partner alone (seat 1), the learner seat frozen and parked: one row per (algo config, step)."""
    from train.config import load_config
    from train.resume import base_env
    from train.run import make_env
    from .coalitions import EVAL_SEED0
    from . import persona

    rows = []
    for algo in ALGOS:
        cfg = load_config(CONFIGS[algo])
        env = make_env(cfg, "team", "greedy", 2)
        a = persona.measure(base_env(env), "stay", "greedy", episodes, EVAL_SEED0, frozen=(0,))
        env.close()
        rows += [{"algo": algo, "step": k, "greedy_alone": round(a[f"{k}_P"], 3)} for k in STEPS]
        print(f"greedy alone ({algo} config): " + " ".join(f"{k}:{a[f'{k}_P']:.1f}" for k in STEPS), flush=True)
    return rows


def play_all(ckpts: str, episodes: int, workers: int) -> list[dict]:
    jobs = []
    for algo in ALGOS:
        for p, _ in PERSONAS:
            for path in sorted(glob.glob(os.path.join(ckpts, f"overcooked-{algo}", f"{algo}_{p}_greedy_s*.pt"))):
                jobs.append((path, CONFIGS[algo], episodes))
    if not jobs:
        raise SystemExit(f"no checkpoints under {ckpts}/overcooked-{{ppo,grpo}}/")
    if workers > 1:
        with get_context("spawn").Pool(workers) as pool:
            parts = pool.map(play_checkpoint, jobs)
    else:
        parts = [play_checkpoint(j) for j in jobs]
    return [r for part in parts for r in part]


# ----------------------------------------------------------------------------------------- drawing
COLORS = {"theft": "red!75!black", "pickup": "blue!70!black", "shadow": "green!50!black"}
PARTNER = "orange!85!black"                # greedy's own actions, drawn like a persona: next to each learner (o), alone (x)
BREAK, LOWFRAC, TOP = 5.0, 0.45, 215.0      # 0..BREAK linear in the lower LOWFRAC of the height, BREAK..TOP log
YTICKS = (0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 200)
WAVE_N = 26                                # wave periods across the 7.2cm plot (+2.4pt over the y axis)
WAVE_A = (7.2 * 28.4528 + 2.4) / (4 * WAVE_N)   # quarter-period length (pt)
DX_ALGO, DX_PERSONA = 0.22, 0.09            # PPO left / GRPO right of each step; personas spread inside


def tex(rows: list[dict], sources: str, greedy_rows: list[dict] | None = None) -> str:
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    trafo = (f"\\pgfmathparse{{(#1<={BREAK}) ? ({LOWFRAC}*#1/{BREAK}) : "
             f"({LOWFRAC}+{1 - LOWFRAC}*ln(max(#1,{BREAK})/{BREAK})/ln({TOP / BREAK}))}}")
    ticks = ",".join(map(str, YTICKS))
    out = [
        "% Figure 2: learner actions per soup step, all personas and both algorithms in one plot.",
        "% colour = persona; P (left) = PPO, G (right) = GRPO; o = next to greedy, x = alone (greedy frozen);",
        "% orange o at the centre of each step = the greedy partner's own actions next to the learners (mean over",
        "%   personas, algorithms and seeds);",
        "% orange x at the centre of each step = greedy alone (learner frozen), mean over the PPO and GRPO configs.",
        f"% y axis: linear 0-{BREAK:g}, log above. Step 4 (wait) is not an action: grey band, no data.",
        "% GENERATED by eval/fig_steps.py from " + sources + ". Do not edit by hand; rerun the script.",
        "% Use inside a one-column figure (width = \\columnwidth).",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  scale only axis, width=7.2cm, height=4.6cm, font=\\scriptsize,",
        "  xmin=0.45, xmax=7.55, xtick={1,...,7}, xlabel={soup step},",
        "  y coord trafo/.code={" + trafo + "},",
        "  y coord inv trafo/.code={\\pgfmathparse{#1}},",
        f"  ymin=0, ymax={TOP:g}, ytick={{{ticks}}}, yticklabels={{{ticks}}}, ylabel={{actions / game}},",
        "  tick align=outside, xtick pos=bottom, ytick pos=left, major tick length=1.5pt,",
        "  axis x line*=bottom, axis y line*=left, axis line style={gray!70}, every tick/.style={gray!70},",
        "  ymajorgrids, grid style={gray!18}, clip=false,",
        "  legend style={font=\\scriptsize, draw=none, fill=none, at={(0.5,1.03)}, anchor=south, legend columns=4,",
        "                cells={anchor=west}, /tikz/every even column/.append style={column sep=5pt}},",
        "]",
        f"\\fill[gray!12] (axis cs:3.55,0) rectangle (axis cs:4.45,{TOP:g});",
        "\\node[gray!60!black] at (axis cs:4,2.5) {wait};",
        # scale change at BREAK: a wavy line across the plot (and through the y axis) instead of a straight one
        f"\\draw[gray!60, line width=0.5pt] ([xshift=-2.4pt]axis cs:0.45,{BREAK:g}) "
        + " ".join([f"sin ++({WAVE_A:.3f}pt,1.1pt) cos ++({WAVE_A:.3f}pt,-1.1pt) "
                    f"sin ++({WAVE_A:.3f}pt,-1.1pt) cos ++({WAVE_A:.3f}pt,1.1pt)"] * WAVE_N) + ";",
    ]
    for k in STEPS:
        x = int(k)
        out.append(f"\\node[gray!60!black, font=\\tiny, anchor=north] at (axis cs:{x - DX_ALGO:.2f},{TOP:g}) {{P}};")
        out.append(f"\\node[gray!60!black, font=\\tiny, anchor=north] at (axis cs:{x + DX_ALGO:.2f},{TOP:g}) {{G}};")
    for pi, (p, _) in enumerate(PERSONAS):
        for ai, algo in enumerate(ALGOS):
            rs = [r for r in rows if r["algo"] == algo and r["persona"] == p]
            w, al = [], []
            for k in STEPS:
                x = int(k) + (-DX_ALGO if ai == 0 else DX_ALGO) + (pi - 1) * DX_PERSONA
                lw = mean([float(r["learner"]) for r in rs if r["step"] == k])
                la = mean([float(r["alone"]) for r in rs if r["step"] == k])
                w.append(f"({x:.2f},{lw:.2f})")
                if lw > 0.05 or la > 0.05:                      # skip 'alone 0' where it does 0 next to greedy too
                    al.append(f"({x:.2f},{la:.2f})")
            out.append(f"% {algo} {p}")
            out.append(f"\\addplot[{COLORS[p]}, only marks, mark=o, mark size=1.5pt, line width=0.6pt, forget plot] "
                       f"coordinates {{{' '.join(w)}}};")
            if al:
                out.append(f"\\addplot[{COLORS[p]}, only marks, mark=x, mark size=2pt, line width=0.7pt, forget plot] "
                           f"coordinates {{{' '.join(al)}}};")
    gw = [f"({int(k)},{mean([float(r['partner']) for r in rows if r['step'] == k]):.2f})" for k in STEPS]
    out.append("% greedy next to the learners (mean over personas, algorithms and seeds)")
    out.append(f"\\addplot[{PARTNER}, only marks, mark=o, mark size=1.5pt, line width=0.6pt, forget plot] "
               f"coordinates {{{' '.join(gw)}}};")
    if greedy_rows:                                             # greedy alone: one x per step, centred
        ga = [f"({int(k)},{mean([float(r['greedy_alone']) for r in greedy_rows if r['step'] == k]):.2f})"
              for k in STEPS]
        out.append("% greedy alone (learner frozen)")
        out.append(f"\\addplot[{PARTNER}, only marks, mark=x, mark size=2pt, line width=0.7pt, forget plot] "
                   f"coordinates {{{' '.join(ga)}}};")
    for p, _ in PERSONAS:
        out.append(f"\\addlegendimage{{only marks, mark=square*, mark size=2pt, {COLORS[p]}}}"
                   f"\\addlegendentry{{\\emph{{{p}}}}}")
    out.append(f"\\addlegendimage{{only marks, mark=square*, mark size=2pt, {PARTNER}}}"
               "\\addlegendentry{\\emph{greedy}}")
    out.append("\\addlegendimage{only marks, mark=o, mark size=1.5pt}\\addlegendentry{with \\emph{greedy} partner}")
    out.append("\\addlegendimage{only marks, mark=x, mark size=2pt}\\addlegendentry{alone}")
    out.append("\\addlegendimage{empty legend}\\addlegendentry{P: PPO, G: GRPO}")
    out += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", default="results/ckpts", help="holds overcooked-ppo/ and overcooked-grpo/")
    p.add_argument("--episodes", type=int, default=10, help="evaluation games per checkpoint and condition")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--csv", default="results/train/overcooked/fig_steps.csv")
    p.add_argument("--tex", default="docs/paper/figures/fig1.tex")
    p.add_argument("--greedy-csv", default="results/train/overcooked/fig_steps_greedy.csv",
                   help="greedy alone (learner frozen); played if missing")
    p.add_argument("--from-csv", action="store_true", help="skip playing; redraw the tex from --csv")
    a = p.parse_args()
    if os.path.exists(a.greedy_csv):
        with open(a.greedy_csv) as f:
            greedy_rows = list(csv.DictReader(f))
    else:
        greedy_rows = play_greedy_alone(a.episodes)
        os.makedirs(os.path.dirname(a.greedy_csv), exist_ok=True)
        with open(a.greedy_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(greedy_rows[0]))
            w.writeheader()
            w.writerows(greedy_rows)
        print(f"wrote {a.greedy_csv}")
    if a.from_csv:
        with open(a.csv) as f:
            rows = list(csv.DictReader(f))
    else:
        rows = play_all(a.ckpts, a.episodes, a.workers)
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv} ({len(rows)} rows)")
    with open(a.tex, "w") as f:
        f.write(tex(rows, a.csv + " and " + a.greedy_csv, greedy_rows))
    print(f"wrote {a.tex}")


if __name__ == "__main__":
    main()
