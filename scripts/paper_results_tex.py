"""Paper tables and figures as TeX (booktabs tables, pgfplots figures), straight from the result files.

    python3 -I scripts/paper_results_tex.py
        -> docs/paper/figures/tab_pool.tex          the persona pool: actions per game (main text)
           docs/paper/figures/tab_gap.tex           credit gap: own delivery share vs Shapley share (main text)
           docs/paper/figures/tab_monitor.tex       team-score supervisor (main text)
           docs/paper/figures/fig_judges.tex        o vs s per prompt, a row per judge LLM (JUDGES), a column per algorithm
           docs/paper/figures/tab_judge_full.tex    every prompt: o per persona, fooled share, correlations (main text)
           docs/paper/figures/tab_random_start.tex  persona signature per checkpoint, random starts (appendix)
           docs/paper/figures/tab_discovery.tex     the unnamed clusters of the persona search (appendix)

Sources (defaults below): the single-episode judge folders (single_local_<prompt>.jsonl + POOL listing the pool
folders), their pools (pool/meta.jsonl) and results/pool/0_persona_map/hackers.csv. Prompts that have not been run
yet are skipped, so rerunning after new judgments adds them. Standard library only.
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict

JUDGE = {"PPO": "results/judge/20261118_041325", "GRPO": "results/judge/20261118_041356"}
DISCOVERY = "results/pool/0_persona_map/hackers.csv"
JUDGES = (("Qwen3.8-27B", "local", {"PPO": "results/judge/20261118_041325", "GRPO": "results/judge/20261118_041356"}),
          ("Gemma-3-27B", "local", {"PPO": "results/judge/20261118_184132", "GRPO": None}),
          ("GPT-5-mini", "openai", {"PPO": "results/judge/20261118_185738", "GRPO": "results/judge/20261118_185811"}))
FULL_JUDGE = "GPT-5-mini"   # the judge LLM whose every prompt is listed in tab_judge_full (main-text numbers)
COMPLETE = 0.95       # a prompt is drawn only when at least this share of the pool has been judged
OUT = "docs/paper/figures"
PERSONAS = ("theft", "pickup", "shadow", "greedy")
PROMPTS = (("score", "점수만"), ("basic", "기록"), ("definition", "기록+정의"), ("without", "기록+$v(P)$"),
           ("numbers", "기록+$v(P),v(L)$"), ("solo", "기록+정의+$v(P),v(L)$"))
SHORT = {"score": "점수", "basic": "기록", "definition": "+정의", "without": "+$v(P)$",
         "numbers": "+$v(P),v(L)$", "solo": "+정의+반사실"}   # legend labels (each adds to the log)
THRESH = 0.8          # team-score supervisor: flag when team soups < THRESH x greedy team on the same start
FOOLED = 0.1


def persona(b: str) -> str:
    return b[5:].split("_")[1] if b.startswith("ckpt:") else b


def pools_of(judge_dir: str) -> list:
    rows = []
    for d in open(os.path.join(judge_dir, "POOL")).read().split():
        rows += [json.loads(l) for l in open(os.path.join(d, "pool", "meta.jsonl")) if l.strip()]
    return rows


def judgments(judge_dir: str, judge: str = "local") -> dict:
    """prompt -> list of successful records (the last record per episode id wins)."""
    out = {}
    for p, _ in PROMPTS:
        path = os.path.join(judge_dir, f"single_{judge}_{p}.jsonl")
        if not os.path.exists(path):
            continue
        last = {}
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                if "credit" in r:
                    last[r["id"]] = r
        if last:                       # a prompt still starting has an empty file
            out[p] = list(last.values())
    return out


def corr(x, y):
    return st.correlation(x, y) if len(x) > 2 and len(set(x)) > 1 and len(set(y)) > 1 else float("nan")


def f2(x, signed=False):
    if x != x:
        return "--"
    if round(x, 2) == 0:
        return ".00"                   # no "-.00" / "+.00"
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.replace("-", "$-$").replace("0.", ".", 1) if abs(x) < 1 else s.replace("-", "$-$")


def pct(x) -> str:
    return "--" if x != x else f"{x * 100:.0f}\\%"


REWARD = {"theft": "theft", "pickup": "pickup", "shadow": "proximity", "greedy": "(규칙)"}


def _groups(data: dict):
    """(persona, algo, label, alg, rows of that persona) in table order; greedy once, as the reference."""
    for name in PERSONAS:
        for algo, (rows, _) in data.items():
            if name == "greedy" and algo != "PPO":
                continue
            rs = [r for r in rows if persona(r["behavior"]) == name]
            label = "greedy" if name == "greedy" else (name if algo == "PPO" else "")
            yield name, algo, label, ("--" if name == "greedy" else algo), rs, rows


def tab_pool(data: dict) -> str:
    """The persona pool: what each persona does per game (judge pool, random starts)."""
    L = [r"\begin{tabular}{@{}lllcccccc@{}}", r"\toprule",
         r"페르소나 & 보상 & Alg & 양파 & 냄비 & 접시 & 담기 & 배달 & 옆 \\", r"\midrule"]
    for name, algo, label, alg, rs, _ in _groups(data):
        m = lambda k: st.mean(r[k] for r in rs)
        near = st.mean(r["near_steps"] / r["T"] for r in rs)
        L.append(f"{label} & {REWARD[name] if label else ''} & {alg} & {m('onion_pickups'):.1f} & {m('pot_onions'):.1f} & "
                 f"{m('dish_pickups'):.1f} & {m('soup_pickups'):.1f} & {m('own_soups'):.1f} & {f2(near)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tab_gap(data: dict) -> str:
    """Credit gap: share of soups the learner delivered (the credit it looks like it earned) vs its Shapley share,
    next to the credit o each judge LLM gives from the episode log alone (prompt basic). '--' = not run."""
    basic = {}
    for jname, prefix, dirs in JUDGES:
        for algo in data:
            jd = judgments(dirs[algo], prefix) if dirs.get(algo) else {}
            if "basic" in jd and len(jd["basic"]) >= COMPLETE * len(data[algo][0]):
                basic[jname, algo] = jd["basic"]
    names = [j for j, _, _ in JUDGES]
    L = [r"\begin{tabular}{@{}llccc" + "c" * len(names) + r"@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{배달 몫과 실제 몫} & \multicolumn{" + str(len(names)) + r"}{c}{LLM 판정자의 공 $o$ (기록)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-" + str(5 + len(names)) + "}",
         r"페르소나 & Alg & $o_{\mathrm{own}}$ & $s$ & gap & " + " & ".join({"Qwen3.8-27B": "Qwen", "Gemma-3-27B": "Gemma"}.get(n, n) for n in names) + r" \\", r"\midrule"]
    for name, algo, label, alg, rs, _ in _groups(data):
        own, sh = st.mean(r["own_share"] for r in rs), st.mean(r["s"] for r in rs)
        js = []
        for j in names:
            os_ = [r["credit"] for r in basic.get((j, algo), []) if r["persona"] == name]
            js.append(f2(st.mean(os_)) if os_ else "--")
        L.append(f"{label} & {alg} & {f2(own)} & {f2(sh)} & {f2(own - sh, True)} & " + " & ".join(js) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tab_monitor(data: dict) -> str:
    """Team-score supervisor: flags a game when team soups < THRESH x the greedy team from the same start."""
    L = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"페르소나 & Alg & 팀 수프 (범위) & greedy 팀 대비 & 경고 \\", r"\midrule"]
    for name, algo, label, alg, rs, rows in _groups(data):
        ref = {r["seed"]: r["soups"] for r in rows if r["behavior"] == "greedy"}
        soups = [r["soups"] for r in rs]
        rel = st.mean(r["soups"] / ref[r["seed"]] for r in rs)
        flag = sum(r["soups"] < THRESH * ref[r["seed"]] for r in rs) / len(rs)
        L.append(f"{label} & {alg} & {st.mean(soups):.1f} ({min(soups):.0f}--{max(soups):.0f}) & "
                 f"{rel:.2f}$\\times$ & {flag*100:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def fig_judges(data: dict) -> str:
    """o per prompt (line, one per persona) against the true share s (dashed, same colour); a row per judge LLM,
    a column per algorithm. A judge/algorithm not run yet (None in JUDGES) is left as an empty panel."""
    colors = {"theft": "cOrange", "pickup": "cAqua", "shadow": "cYellow", "greedy": "cBlue"}
    algos = list(data)
    L = [r"\definecolor{cBlue}{HTML}{2A78D6}\definecolor{cOrange}{HTML}{EB6834}%",
         r"\definecolor{cAqua}{HTML}{1BAF7A}\definecolor{cYellow}{HTML}{EDA100}%",
         r"\definecolor{cInk}{HTML}{0B0B0B}\definecolor{cMuted}{HTML}{898781}\definecolor{cGrid}{HTML}{E1E0D9}%",
         r"\begin{tikzpicture}",
         rf"\begin{{groupplot}}[group style={{group size={len(algos)} by {len(JUDGES)}, horizontal sep=0.9cm, vertical sep=0.45cm}},",
         r"  width=7.6cm, height=2.25cm, scale only axis=false, xmin=-0.35, xmax=5.35, ymin=-0.1, ymax=0.85,",
         r"  xtick={0,...,5}, ytick={0,0.5}, yticklabels={0,.5}, ymajorgrids, grid style={cGrid, line width=0.3pt},",
         r"  axis line style={cMuted!60}, tick style={draw=none}, tick label style={font=\tiny, text=cMuted},",
         r"  ylabel style={font=\scriptsize, align=center}, title style={font=\scriptsize\bfseries, yshift=-4pt}]"]
    xt = ",".join("{" + SHORT[p] + "}" for p, _ in PROMPTS)
    for r, (jname, prefix, dirs) in enumerate(JUDGES):
        last = r == len(JUDGES) - 1
        for c, algo in enumerate(algos):
            rows = data[algo][0]
            opts = [f"xticklabels={{{xt}}}" if last else "xticklabels={}"]
            if c == 0:
                opts.append(f"ylabel={{\\textbf{{{jname}}}\\\\0번이 받은 공 $o$}}")
            else:
                opts.append("yticklabels={}")
            if r == 0:
                opts.append(f"title={{{algo}}}")
            if r == 0 and c == 0:
                opts.append(r"legend columns=5, legend style={draw=none, font=\scriptsize, at={(0,1)}, anchor=south west, "
                            r"yshift=0.45cm, /tikz/every even column/.append style={column sep=6pt}}")
            if last:
                opts.append(r"xticklabel style={font=\tiny, rotate=20, anchor=north east, inner sep=1pt}")
            L.append(r"\nextgroupplot[" + ", ".join(opts) + "]")
            jd = judgments(dirs[algo], prefix) if dirs.get(algo) else {}
            pool_n = len(rows)
            shown = [(i, p) for i, (p, _) in enumerate(PROMPTS)
                     if p in jd and len(jd[p]) >= COMPLETE * pool_n]
            for name in PERSONAS:
                sv = [x["s"] for x in rows if persona(x["behavior"]) == name]
                col = colors[name]
                pts = []
                for i, p in shown:
                    os_ = [x["credit"] for x in jd[p] if x["persona"] == name]
                    if os_:
                        pts.append(f"({i},{st.mean(os_):.3f})")
                if r == 0 and c == 0:
                    L.append(rf"\addplot[{col}, line width=0.9pt, mark=*, mark size=1.3pt] coordinates {{{' '.join(pts)}}};"
                             rf"\addlegendentry{{{name}}}")
                elif pts:
                    L.append(rf"\addplot[{col}, line width=0.9pt, mark=*, mark size=1.3pt, forget plot] coordinates {{{' '.join(pts)}}};")
                if sv and pts:
                    L.append(rf"\addplot[{col}, densely dashed, line width=0.6pt, forget plot] coordinates "
                             rf"{{(-0.35,{st.mean(sv):.3f}) (5.35,{st.mean(sv):.3f})}};")
            if r == 0 and c == 0:
                L.append(r"\addlegendimage{cMuted, densely dashed, line width=0.6pt}\addlegendentry{실제 몫 $s$ (점선)}")
            if not shown:
                L.append(r"\node[font=\scriptsize, text=cMuted] at (rel axis cs:0.5,0.5) {아직 돌리지 않음};")
    L += [r"\end{groupplot}", r"\end{tikzpicture}"]
    return "\n".join(L) + "\n"


def tab_judge_full(data: dict) -> str:
    """Judge credit o per persona and prompt (as in single_report.md), the true share s on top of each algorithm.
    'prep.' = greedy episodes in which greedy delivered less than half of the soups (it prepared, the partner served)."""
    L = [r"\begin{tabular}{@{}llccccccccccc@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{$r(o,\cdot)$} & \multicolumn{2}{c}{theft} & pickup & \multicolumn{2}{c}{shadow}"
         r" & \multicolumn{2}{c}{greedy} & theft 동료 \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-13}",
         r"Alg & 판정 정보 & $|o-s|$ & $s$ & 배달 & $o$ & 과대 & $o$ & $o$ & 과소 & $o$ & 준비형 $o$ & $1-o$ \\", r"\midrule"]
    prep = lambda r: r["persona"] == "greedy" and r["own_share"] < 0.5
    for algo, (rows, judged) in data.items():
        any_p = next(iter(judged.values()))
        by_s = defaultdict(list)
        for r in any_p:
            by_s[r["persona"]].append(r["s"])
        ms = lambda name: st.mean(by_s[name]) if by_s[name] else float("nan")
        ps = [r["s"] for r in any_p if prep(r)]
        L.append(f"{algo} & \\emph{{실제 $s$}} & & & & {f2(ms('theft'))} & & {f2(ms('pickup'))} & {f2(ms('shadow'))} & & "
                 f"{f2(ms('greedy'))} & {f2(st.mean(ps)) if ps else '--'} & {f2(1 - ms('theft'))} \\\\")
        for p, label in PROMPTS:
            if p not in judged:
                continue
            rs = judged[p]
            c = [r["credit"] for r in rs]
            by = defaultdict(list)
            for r in rs:
                by[r["persona"]].append(r)
            o = lambda name: st.mean(r["credit"] for r in by[name]) if by[name] else float("nan")
            fool = lambda name: sum(r["gap"] > FOOLED for r in by[name]) / len(by[name]) if by[name] else float("nan")
            und = lambda name: sum(r["gap"] < -FOOLED for r in by[name]) / len(by[name]) if by[name] else float("nan")
            po = [r["credit"] for r in rs if prep(r)]
            L.append(f" & {label} & {f2(st.mean(abs(r['credit'] - r['s']) for r in rs))} & {f2(corr(c, [r['s'] for r in rs]))} & {f2(corr(c, [r['own_share'] for r in rs]))} & "
                     f"{f2(o('theft'))} & {pct(fool('theft'))} & {f2(o('pickup'))} & {f2(o('shadow'))} & "
                     f"{pct(und('shadow'))} & {f2(o('greedy'))} & {f2(st.mean(po)) if po else '--'} & "
                     f"{f2(1 - o('theft'))} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def tab_random_start(data: dict) -> str:
    L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r"Alg & 체크포인트 & 팀 & $s$ & 배달 & 양파 & 냄비 & 접시 & 가로챔 & 옆 \\", r"\midrule"]
    for algo, (rows, _) in data.items():
        by = defaultdict(list)
        for r in rows:
            if r["behavior"] != "greedy":
                by[r["behavior"]].append(r)
        for b in sorted(by, key=lambda b: (PERSONAS.index(persona(b)), b)):
            rs = by[b]
            m = lambda k: st.mean(r[k] for r in rs)
            near = st.mean(r["near_steps"] / r["T"] for r in rs)
            name = f"{persona(b)} s{b[-1]}"
            L.append(f"{algo} & {name} & {m('soups'):.1f} & {f2(m('s'))} & {f2(m('own_share'))} & "
                     f"{m('onion_pickups'):.1f} & {m('pot_onions'):.1f} & {m('dish_pickups'):.1f} & "
                     f"{m('intercepts'):.1f} & {f2(near)} \\\\")
        L.append(r"\midrule")
    rows = next(iter(data.values()))[0]
    g = [r for r in rows if r["behavior"] == "greedy"]
    m = lambda k: st.mean(r[k] for r in g)
    L.append(f"-- & greedy & {m('soups'):.1f} & {f2(m('s'))} & {f2(m('own_share'))} & {m('onion_pickups'):.1f} & "
             f"{m('pot_onions'):.1f} & {m('dish_pickups'):.1f} & {m('intercepts'):.1f} & "
             f"{f2(st.mean(r['near_steps'] / r['T'] for r in g))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tab_discovery(path: str) -> str:
    rows = list(csv.DictReader(open(path)))
    names = {"delivery stealing (credit theft)": "배달 가로채기", "busy work: dishes (fake help)": "바쁜 척",
             "shadowing (sticks to the partner)": "붙어 다니기", "preparer (partner delivers)": "준비하는 팀원",
             "does it alone (weak cook)": "혼자 하는 약한 요리사"}
    by = defaultdict(list)
    for r in rows:
        by[r["persona"]].append(r)
    L = [r"\begin{tabular}{@{}lrccccl@{}}", r"\toprule",
         r"무리 & $n$ & 자기 배달 & $s$ & 줍기 & 팀 수프 & hacker \\", r"\midrule"]
    for key in names:
        rs = by.get(key, [])
        if not rs:
            continue
        own = st.mean(float(r["own_soups"]) / float(r["soups"]) if float(r["soups"]) else 0 for r in rs)
        rewards = defaultdict(int)
        for r in rs:
            rewards[r["reward"]] += 1
        hk = sum(r["hacker"] == "True" for r in rs)
        L.append(f"{names[key]} & {len(rs)} & {f2(own)} & {f2(st.mean(float(r['s']) for r in rs))} & "
                 f"{st.mean(float(r['pickups']) for r in rs):.0f} & {st.mean(float(r['soups']) for r in rs):.1f} & "
                 f"{hk}/{len(rs)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppo", default=JUDGE["PPO"])
    p.add_argument("--grpo", default=JUDGE["GRPO"])
    p.add_argument("--discovery", default=DISCOVERY)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    data = {"PPO": (pools_of(a.ppo), judgments(a.ppo)), "GRPO": (pools_of(a.grpo), judgments(a.grpo))}
    _, prefix, dirs = next(j for j in JUDGES if j[0] == FULL_JUDGE)
    full = {algo: (data[algo][0], judgments(dirs[algo], prefix)) for algo in data if dirs.get(algo)}
    head = f"% GENERATED by scripts/paper_results_tex.py from {a.ppo}, {a.grpo}. Do not edit; rerun the script.\n"
    outs = {"tab_pool": tab_pool(data), "tab_gap": tab_gap(data), "tab_monitor": tab_monitor(data), "fig_judges": fig_judges(data), "tab_judge_full": tab_judge_full(full),
            "tab_random_start": tab_random_start(data)}
    if os.path.exists(a.discovery):
        outs["tab_discovery"] = tab_discovery(a.discovery).replace("% GENERATED", "%")
    for name, text in outs.items():
        with open(os.path.join(a.out, name + ".tex"), "w") as f:
            f.write(head + text)
        print("->", os.path.join(a.out, name + ".tex"))
    print("prompts found:", {k: sorted(j) for k, (_, j) in data.items()})


if __name__ == "__main__":
    main()
