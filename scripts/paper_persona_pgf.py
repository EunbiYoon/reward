"""Learning-curve figure of the paper (appendix, fig:curves) as native LaTeX (pgfplots).

    python3 -I scripts/paper_persona_pgf.py      (needs numpy, pandas)
        -> docs/paper/figures/fig_curves.tex      (pgfplots source; PPO row on top, GRPO row below)
           docs/paper/figures/data/curves_<algo>_<persona>.dat   (the numbers each plot reads)

Reads results/train/<algo>/**/curves/<algo>_<persona>_greedy_s*.csv (written by train.run) for theft, pickup and
shadow trained next to greedy. Panels: training reward, team soups, learner deliveries, learner pickups. Line =
mean over seeds 0-2, band = min-max, smoothed over 10 updates and thinned to every 5th update; the dashed line marks
1M steps, where the official shaping reaches 0.
"""
import glob
import os

import numpy as np
import pandas as pd

OUT = "docs/paper/figures"
DATA = os.path.join(OUT, "data")
CURVES = {  # name -> (legend label, color, csv globs); {a} = algo
    "theft": ("theft", "cOrange", ["results/train/{a}/**/curves/{a}_theft_greedy_s*.csv"]),
    "pickup": ("pickup", "cAqua", ["results/train/{a}/**/curves/{a}_pickup_greedy_s*.csv"]),
    "shadow": ("shadow", "cYellow", ["results/train/{a}/**/curves/{a}_shadow_greedy_s*.csv"]),
}
PANELS = (("proxy_return", "학습 보상"), ("team_soups", "팀 수프"),
          ("own_soups", "학습자 배달"), ("pickups", "학습자 줍기"))

# validated categorical palette (slots 1-4) and chart ink, shared by every figure
PREAMBLE = r"""\definecolor{cBlue}{HTML}{2A78D6}\definecolor{cOrange}{HTML}{EB6834}%
\definecolor{cAqua}{HTML}{1BAF7A}\definecolor{cYellow}{HTML}{EDA100}%
\definecolor{cInk}{HTML}{0B0B0B}\definecolor{cMuted}{HTML}{898781}%
\definecolor{cGrid}{HTML}{E1E0D9}\definecolor{cBase}{HTML}{C3C2B7}%
\pgfplotsset{personastyle/.style={scale only axis, font=\scriptsize, axis line style={cBase},
  axis x line*=bottom, axis y line=left, y axis line style={draw=none}, tick style={draw=none},
  ymajorgrids, grid style={cGrid, line width=0.3pt}, tick label style={text=cMuted},
  label style={text=cInk}, title style={align=left, anchor=base west, at={(0,1)}, yshift=6pt, font=\scriptsize}}}%
"""


def load(algo: str, globs: list) -> list:
    for g in globs:
        files = sorted(glob.glob(g.format(a=algo), recursive=True))
        if files:
            return [pd.read_csv(f) for f in files]
    return []


def write_dat(algo: str) -> list:
    """figures/data/curves_<algo>_<persona>.dat (seed mean / min / max, smoothed over 10 updates, every 5th update)."""
    os.makedirs(DATA, exist_ok=True)
    shown = []
    for name, (_, _, globs) in CURVES.items():
        runs = load(algo, globs)
        if not runs:
            print(f"  {algo}: no curves for {name}")
            continue
        n = min(len(r) for r in runs)
        out = {"x": runs[0].env_steps.values[:n] / 1e6}
        for col, _ in PANELS:
            ys = np.stack([r[col].rolling(10, min_periods=1).mean().values[:n] for r in runs])
            out[f"{col}_m"], out[f"{col}_lo"], out[f"{col}_hi"] = ys.mean(0), ys.min(0), ys.max(0)
        t = pd.DataFrame(out).iloc[::5].round(3)
        t.to_csv(os.path.join(DATA, f"curves_{algo}_{name}.dat"), sep=" ", index=False)
        shown.append(name)
    return shown


def curves_tex(algos=("ppo", "grpo")) -> str:
    """One figure: a row per algorithm (PPO top, GRPO bottom), a column per panel, one legend on top."""
    shown = {a: write_dat(a) for a in algos}
    lines = [PREAMBLE, r"\begin{tikzpicture}",
             rf"\begin{{groupplot}}[group style={{group size=4 by {len(algos)}, horizontal sep=1.15cm, vertical sep=0.55cm}},",
             r"  personastyle, width=3.15cm, height=2.3cm, xmin=0, xmax=2, ymin=0, xtick={0,0.5,1,1.5,2},",
             r"  xlabel style={font=\tiny}, ylabel style={font=\tiny, align=center}]"]
    for r, algo in enumerate(algos):
        last = r == len(algos) - 1
        for i, (col, title) in enumerate(PANELS):
            ylab = rf"\textbf{{{algo.upper()}}}\\{title} / 에피소드" if i == 0 else f"{title} / 에피소드"
            opts = f"ylabel={{{ylab}}}"
            opts += r", xlabel={환경 step (백만)}" if last else r", xticklabels={}"
            first = r == 0 and i == 0
            if first:
                opts += (r", legend columns=5, legend style={draw=none, font=\scriptsize, at={(0,1)}, "
                         r"anchor=south west, yshift=0.35cm, /tikz/every even column/.append style={column sep=8pt}}")
            lines.append(rf"\nextgroupplot[{opts}]")
            for name in shown[algo]:
                label, color, _ = CURVES[name]
                f = f"figures/data/curves_{algo}_{name}.dat"
                tag = f"{algo}{name}{col}"
                lines.append(rf"\addplot[name path={tag}hi, draw=none, forget plot] table[x=x, y={col}_hi] {{{f}}};")
                lines.append(rf"\addplot[name path={tag}lo, draw=none, forget plot] table[x=x, y={col}_lo] {{{f}}};")
                lines.append(rf"\addplot[{color}, fill opacity=0.18, draw=none, forget plot] "
                             rf"fill between[of={tag}hi and {tag}lo];")
                legend = rf"\addlegendentry{{{label}}}" if first else ""
                lines.append(rf"\addplot[{color}, line width=0.9pt{'' if legend else ', forget plot'}] "
                             rf"table[x=x, y={col}_m] {{{f}}};{legend}")
            lines.append(r"\draw[cMuted, densely dashed, line width=0.5pt] ({axis cs:1,0}|-{rel axis cs:0,0}) -- "
                         r"({axis cs:1,0}|-{rel axis cs:0,1});")
            if first:
                lines.append(r"\addlegendimage{cMuted, densely dashed, line width=0.5pt}\addlegendentry{공식 shaping이 0이 되는 지점}")
    lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fig_curves.tex"), "w") as f:
        f.write(curves_tex())
    print(f"-> {OUT}/fig_curves.tex, data/")


if __name__ == "__main__":
    main()
