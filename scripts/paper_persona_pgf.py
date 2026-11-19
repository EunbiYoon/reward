"""The persona figures as native LaTeX (pgfplots), drawn by the paper's own fonts.

    venv/bin/python scripts/paper_persona_pgf.py
        -> docs/paper/figures/fig_curves.tex  (pgfplots source; PPO and GRPO rows in one figure)
           docs/paper/figures/data/*.dat              (the numbers each plot reads)
           docs/paper/figures/fig_*.png               (each source compiled once with pdflatex in a temp dir and
                                                       rasterized with ghostscript at 300 dpi; only the PNG is kept)

Same content as scripts/paper_persona_figs.py (the matplotlib version):
- persona steps: PPO (top) and GRPO (bottom) x theft, pickup, shadow, all trained next to greedy. Learner vs
  partner actions per episode for the soup steps 1-7 (mean over seeds = bar, one seed = dot). Reads
  results/<algo>/need_curve/persona_actions.csv (run scripts/persona_actions.py --root ... first).
- learning curves: theft, pickup, shadow (next to greedy); mean over seeds 0-2 (line) and min-max (band),
  smoothed over 10 updates and thinned to every 5th update.
The PDFs use the paper's fonts (times/helvet), so they look the same as drawing them inside the paper.
"""
import glob
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

OUT = "docs/paper/figures"
DATA = os.path.join(OUT, "data")
PERSONAS = (("theft", "theft"), ("pickup", "pickup"), ("shadow", "shadow"))
STEP_LABELS = ["1\\\\onion", "2\\\\pot", "3\\\\cook", "5\\\\dish", "6\\\\scoop", "7\\\\serve"]  # step 4 (wait) is no action
POS = {"1": 1, "2": 2, "3": 3, "5": 4, "6": 5, "7": 6}  # x position of each step, no gap for step 4
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


def num(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def steps_tex() -> str:
    lines = [PREAMBLE, r"\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=3 by 2, horizontal sep=0.95cm, vertical sep=0.95cm},",
             r"  personastyle, width=5.0cm, height=1.55cm, xmin=0.45, xmax=6.55, ymin=0, xtick={1,...,6},",
             r"  xticklabels={" + ",".join("{" + s + "}" for s in STEP_LABELS) + "},",
             r"  xticklabel style={align=center, font=\tiny, inner sep=1pt}, yticklabel style={font=\tiny},",
             r"  title style={yshift=-2pt}, enlarge x limits=false]"]
    first = True
    for algo in ("ppo", "grpo"):
        path = f"results/{algo}/need_curve/persona_actions.csv"
        if not os.path.exists(path):
            print(f"  skip {algo}: {path} missing")
            continue
        df = pd.read_csv(path, dtype={"step": str})
        df = df[df.step != "4"]
        for c, (persona, reward) in enumerate(PERSONAS):
            d = df[df.persona == persona]
            top = max(1.0, d.learner.max(), d.partner.max()) * 1.1
            near = d.groupby("checkpoint").near_frac.first().mean()
            ylabel = r", ylabel={actions / ep.}, ylabel style={font=\tiny, yshift=-3pt}" if c == 0 else ""
            # legend drawn in the first panel, above its title (no `legend to name` + \ref: it hangs with kotex)
            leg = (r"legend columns=3, legend style={draw=none, font=\scriptsize, at={(0,1)}, anchor=south west, "
                   r"yshift=0.55cm, /tikz/every even column/.append style={column sep=8pt}}, ") if first else ""
            lines.append(rf"\nextgroupplot[{leg}ymax={num(top)}{ylabel}, title={{\textbf{{{persona}}} ({algo.upper()})"
                         rf"\ {{\color{{cMuted}}\tiny {reward} $\cdot$ near partner {near * 100:.0f}\%}}}}]")
            for seat, color, shift in (("learner", "cBlue", -0.19), ("partner", "cOrange", 0.19)):
                means = d.groupby("step")[seat].mean()
                coords = " ".join(f"({POS[s]},{num(v)})" for s, v in means.items())
                legend = rf"\addlegendentry{{{seat} ({'persona' if seat == 'learner' else 'greedy'})}}" if first else ""
                lines.append(rf"\addplot[ybar, bar width=0.34, bar shift={shift}, fill={color}, draw=none, area legend] "
                             rf"coordinates {{{coords}}};{legend}")
            for seat, shift in (("learner", -0.19), ("partner", 0.19)):
                dots = " ".join(f"({POS[r.step] + shift:.2f},{num(getattr(r, seat))})" for r in d.itertuples())
                legend = r"\addlegendentry{one seed}" if first and seat == "learner" else ""
                opts = "only marks, mark=*, mark size=0.5pt, cInk, fill opacity=0.5, draw opacity=0.5"
                lines.append(rf"\addplot[{opts}{'' if legend else ', forget plot'}] coordinates {{{dots}}};{legend}")
            first = False
    lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


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


STANDALONE = r"""\documentclass[10pt,border=1pt]{standalone}
\usepackage{times}\usepackage{helvet}\usepackage{courier}
\usepackage{pgfplots}\usepgfplotslibrary{groupplots,fillbetween}\pgfplotsset{compat=1.18}
\begin{document}
\input{figures/%s.tex}
\end{document}
"""


def compile_png(name: str):
    """figures/<name>.tex -> figures/<name>.png (300 dpi), compiled from docs/paper so figures/data/... resolves.
    The intermediate PDF stays in the temp dir."""
    paper = os.path.dirname(OUT)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, name + ".tex")
        with open(src, "w") as f:
            f.write(STANDALONE % name)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", tmp, src],
                           cwd=paper, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"pdflatex failed on {name}:\n" + r.stdout[-2000:])
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300",
                        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        "-sOutputFile=" + os.path.abspath(os.path.join(OUT, name + ".png")),
                        os.path.join(tmp, name + ".pdf")], check=True)


def main():
    # the paper \input-s the .tex (no PNG); the persona steps figure is now scripts/paper_fig1.py
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fig_curves.tex"), "w") as f:
        f.write(curves_tex())
    print(f"-> {OUT}/fig_curves.tex, data/")


if __name__ == "__main__":
    main()
