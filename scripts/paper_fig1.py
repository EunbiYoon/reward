"""Figure 1 for the paper: persona reproduction check and solo purity test in one figure (pgfplots).

    python3 scripts/paper_fig1.py      -> docs/paper/figures/fig1.tex  (pgfplots source, \\input-able)
                                          docs/paper/figures/fig1.png  (rendered at 300 dpi, \\includegraphics-able)

Layout: columns theft / pickup / shadow, rows PPO / GRPO; sized for a one-column figure (\\columnwidth).
For every soup step (x: 1 onion ... 7 deliver; 4 wait is shaded, it is not an action), y = actions per game:
    blue bar      learner next to greedy      mean over the seed checkpoints
    orange bar    greedy (the partner)        mean over the seed checkpoints
    small dots    one seed each, on top of both bars (only where the mean is not 0)
    white diamond learner ALONE (greedy frozen): mean over seeds, whisker = min-max over seeds;
                  drawn where the learner does that step with greedy or alone
Sources: persona_actions.csv (learner, partner per seed) and persona_solo.csv (mode alone).
Only the standard library plus pdflatex and ghostscript.
"""
import argparse
import csv
import os
import subprocess
import tempfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIONS = {"ppo": "results/train/ppo/A_persona/persona_actions.csv",
           "grpo": "results/train/grpo/A_persona/persona_actions.csv"}
SOLO = "results/train/persona-fig/solo vs withgreedy/persona_solo.csv"
OUT = "docs/paper/figures/fig1.tex"
PERSONAS = (("theft", "개인 성과급"), ("pickup", "줍기 보너스"), ("shadow", "근접 보너스"))
ALGOS = ("ppo", "grpo")
STEPS = (1, 2, 3, 5, 6, 7)
YAX = {"theft": (40, "0,10,20,30,40"), "pickup": (210, "0,50,100,150,200"), "shadow": (40, "0,10,20,30,40")}
DX = 0.25   # learner bar at step - DX, alone diamond at step, partner bar at step + DX


def load(actions: dict, solo: str) -> dict:
    """(algo, persona) -> step -> {"L": [per seed], "P": [...], "A": [...]}"""
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for algo, path in actions.items():
        for r in csv.DictReader(open(os.path.join(ROOT, path))):
            cell = d[(algo, r["persona"])][int(r["step"])]
            cell["L"].append(float(r["learner"]))
            cell["P"].append(float(r["partner"]))
    for r in csv.DictReader(open(os.path.join(ROOT, solo))):
        if r["mode"] == "alone":
            d[(r["algo"], r["persona"])][int(r["step"])]["A"].append(float(r["learner"]))
    return d


def mean(xs):
    return sum(xs) / len(xs)


def coords(pts) -> str:
    return " ".join(f"({x:.2f},{y:.2f})" for x, y in pts)


def panel(cells: dict) -> list:
    bars_l = [(s - DX, mean(cells[s]["L"])) for s in STEPS if mean(cells[s]["L"]) > 0]
    bars_p = [(s + DX, mean(cells[s]["P"])) for s in STEPS if mean(cells[s]["P"]) > 0]
    dots = [(s - DX, v) for s in STEPS if mean(cells[s]["L"]) > 0 for v in cells[s]["L"]] + \
           [(s + DX, v) for s in STEPS if mean(cells[s]["P"]) > 0 for v in cells[s]["P"]]
    alone = []
    for s in STEPS:
        a = cells[s]["A"]
        if mean(cells[s]["L"]) > 0 or mean(a) > 0:
            m = mean(a)
            alone.append(f"({s},{m:.2f}) += (0,{max(a) - m:.2f}) -= (0,{m - min(a):.2f})")
    out = []
    if bars_l:
        out.append(f"\\addplot[fig1learner] coordinates {{{coords(bars_l)}}};")
    if bars_p:
        out.append(f"\\addplot[fig1partner] coordinates {{{coords(bars_p)}}};")
    if dots:
        out.append(f"\\addplot[fig1seed] coordinates {{{coords(dots)}}};")
    if alone:
        out.append(f"\\addplot[fig1alone] coordinates {{{' '.join(alone)}}};")
    return out


HEAD = r"""\pgfplotsset{
  fig1learner/.style={ybar, area legend, bar width=2.6pt, bar shift=0pt, fill=blue!65!black, draw=none},
  fig1partner/.style={ybar, area legend, bar width=2.6pt, bar shift=0pt, fill=orange!85!red, draw=none},
  fig1seed/.style={only marks, mark=*, mark size=0.4pt, black!70},
  fig1alone/.style={only marks, mark=diamond*, mark size=1.5pt, draw=blue!65!black, fill=white,
                    line width=0.4pt,
                    error bars/.cd, y dir=both, y explicit,
                    error bar style={blue!65!black, line width=0.4pt},
                    error mark options={rotate=90, mark size=0.9pt, blue!65!black}},
}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=3 by 2, horizontal sep=13pt, vertical sep=10pt,
               xticklabels at=edge bottom},
  scale only axis, width=0.255\columnwidth, height=1.75cm,
  xmin=0.45, xmax=7.55, ymin=0,
  xtick={1,2,3,4,5,6,7},
  xticklabels={{1\\양파},{2\\냄비},{3\\요리},{4\\대기},{5\\접시},{6\\담기},{7\\배달}},
  xlabel style={font=\tiny, yshift=2pt},
  xticklabel style={font=\tiny, align=center, inner sep=0.5pt},
  yticklabel style={font=\tiny, inner sep=0.5pt},
  tick align=outside, xtick pos=bottom, ytick pos=left, major tick length=1pt,
  axis x line*=bottom, axis y line*=left,
  axis line style={gray!70}, every tick/.style={gray!70},
  ymajorgrids, grid style={gray!18},
  title style={font=\tiny, align=center, yshift=-5pt},
  ylabel style={font=\tiny, align=center, yshift=-3pt},
  clip=false,
  legend style={font=\tiny, draw=none, legend columns=2,
                /tikz/every even column/.append style={column sep=6pt}},
]
"""

WAIT = (r"\fill[gray!12] ({axis cs:3.55,0}|-{rel axis cs:0,0}) rectangle ({axis cs:4.45,0}|-{rel axis cs:0,1});"
        "\n" r"\node[font=\tiny, gray!80!black, anchor=north] at ({axis cs:4,0}|-{rel axis cs:0,1}) {대기};")

LEGEND = r"\legend{학습자 (seed 평균), greedy 동료, seed 하나, 학습자 혼자 (greedy 멈춤)}"   # details are in the caption


def render(d: dict, sources: str) -> str:
    out = ["% Figure 1: persona reproduction check + solo purity test.",
           "% GENERATED by scripts/paper_fig1.py from:",
           f"%   {sources}",
           "% Do not edit by hand; rerun the script. Use inside a one-column figure (width = \\columnwidth).",
           "% preamble: \\usepackage{pgfplots} \\usepgfplotslibrary{groupplots} \\usetikzlibrary{calc}"
           " \\pgfplotsset{compat=1.17}",
           HEAD]
    first = True
    for algo in ALGOS:
        out.append(f"% ---- {algo.upper()} ----")
        for persona, reward in PERSONAS:
            ymax, ticks = YAX[persona]
            opts = [f"ymax={ymax}", f"ytick={{{ticks}}}"]
            if algo == ALGOS[0]:
                opts.insert(0, f"title={{\\textbf{{{persona}}}\\\\{reward}}}")
            if persona == PERSONAS[0][0]:
                opts.insert(0, f"ylabel={{\\textbf{{{algo.upper()}}}\\\\게임당 행동 횟수}}")
            if algo == ALGOS[-1]:
                opts.append("xlabel={수프 단계}")
            if first:
                opts.append("legend to name=fig1legend")
            out.append(f"\\nextgroupplot[{', '.join(opts)}]")
            out.append(WAIT)
            out += panel(d[(algo, persona)])
            if first:
                out.append(LEGEND)
                first = False
            out.append("")
    out.append(r"""\end{groupplot}
\node[anchor=north] at ($(group c2r2.south)+(0,-18pt)$) {\pgfplotslegendfromname{fig1legend}};
\end{tikzpicture}
""")
    return "\n".join(out)


STANDALONE = r"""\documentclass[border=4pt]{standalone}
\usepackage{times,pgfplots,kotex}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.17}
\setlength{\columnwidth}{3.3in}  % AAAI column width
\begin{document}
\input{FIG}
\end{document}
"""


def to_png(tex: str, png: str):
    """Compile tex standalone in a temp dir and rasterize to png (300 dpi); only the PNG is kept."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "fig.tex"), "w") as f:
            f.write(STANDALONE.replace("FIG", tex))
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                           cwd=tmp, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("pdflatex failed:\n" + r.stdout[-2000:])
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300",
                        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-sOutputFile=" + png,
                        os.path.join(tmp, "fig.pdf")], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppo", default=ACTIONS["ppo"])
    p.add_argument("--grpo", default=ACTIONS["grpo"])
    p.add_argument("--solo", default=SOLO)
    p.add_argument("--out", default=OUT)
    p.add_argument("--png", action="store_true", help="also rasterize a PNG preview (needs kotex)")
    a = p.parse_args()
    d = load({"ppo": a.ppo, "grpo": a.grpo}, a.solo)
    missing = [(al, pe) for al in ALGOS for pe, _ in PERSONAS if (al, pe) not in d]
    if missing:
        raise SystemExit(f"missing in the csvs: {missing}")
    out = os.path.join(ROOT, a.out)
    with open(out, "w") as f:
        f.write(render(d, f"{a.ppo}; {a.grpo}; {a.solo}"))
    print("->", out)
    if a.png:     # needs kotex for the Korean labels (not installed on the cluster)
        png = out[:-4] + ".png"
        to_png(out, png)
        print("->", png)


if __name__ == "__main__":
    main()
