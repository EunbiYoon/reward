"""RWARE paper tables (English): tab_rware, tab_rware_pool, tab_rware_judge. Standard library only.

    python3 -I docs/paper/make_rware_tables.py --greedy results/judge/rware3_greedy60 --out docs/paper/figures
    python3 -I docs/paper/make_rware_tables.py --greedy pool --out /tmp/x        # greedy from the persona pools (old N=20)

s is the clipped three-player Shapley share s_0 = phi_0^+ / sum_i phi_i^+ from the pool's coalition values.
"""
import argparse
import json
import os
import statistics as st
from itertools import combinations

ROOT = "results/judge"
POOLS = {"PPO": "rware3_cramped_ppo", "GRPO": "rware3_cramped_grpo"}
JUDGES = (("Qwen3.8-27B", {"PPO": "cramped_ppo_qwen", "GRPO": "cramped_grpo_qwen"}),
          ("Gemma-3-27B", {"PPO": "cramped_ppo_gemma", "GRPO": "cramped_grpo_gemma"}))
PERSONAS = ("theft", "pickup", "shadow", "team", "greedy")
PROMPTS = (("score", "(1) score only"), ("basic", "(2) log"), ("definition", "(3) log+def."),
           ("without", "(4) log+$v(P)$"), ("numbers", "(5) log+$v(P),v(L)$"), ("solo", "(6) log+def.+$v(P),v(L)$"))
THRESH, FOOLED = 0.8, 0.1


def persona(b):
    return b[5:].split("_")[1] if b.startswith("ckpt:") else b


def v_of(r, S):
    """coalition value; seats 0 (learner), 1, 2."""
    S = frozenset(S)
    if not S:
        return 0.0
    if len(S) == 3:
        return r["v_L12"] if "v_L12" in r else r["soups"]
    key = "v_" + ("L" if 0 in S else "") + "".join(str(i) for i in sorted(S - {0}))
    return r[key]


def clip_s(r):
    n, w = 3, {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
    phi = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        phi.append(sum(w[k] * (v_of(r, set(S) | {i}) - v_of(r, S)) for k in range(n) for S in combinations(others, k)))
    pos = [max(p, 0.0) for p in phi]
    return pos[0] / sum(pos) if sum(pos) > 0 else 1 / 3


def load(d, keep=lambda r: True):
    rows = []
    for l in open(os.path.join(d, "pool", "meta.jsonl")):
        if l.strip():
            r = json.loads(l)
            if keep(r):
                r["s_c"] = clip_s(r)
                rows.append(r)
    return rows


def f2(x, signed=False):
    if x != x:
        return "--"
    if round(x, 2) == 0:
        return ".00"
    s = f"{x:+.2f}" if signed else f"{x:.2f}"
    return s.replace("-", "$-$").replace("0.", ".", 1) if abs(x) < 1 else s.replace("-", "$-$")


def judged(folder, prompt, by_id):
    path = os.path.join(ROOT, folder, f"single_local_{prompt}.jsonl")
    if not os.path.exists(path):
        return []
    last = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if "credit" in r and "error" not in r and r["id"] in by_id:
                last[r["id"]] = r
    return [{"persona": persona(by_id[i]["behavior"]), "credit": float(r["credit"]), "s": by_id[i]["s_c"]}
            for i, r in last.items()]


def mean_or(xs):
    return st.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", default=f"{ROOT}/rware3_greedy60", help="greedy pool folder, or 'pool' for the old N=20")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pools = {}
    for algo, d in POOLS.items():
        if a.greedy == "pool":
            pools[algo] = load(os.path.join(ROOT, d))
        else:
            pools[algo] = load(os.path.join(ROOT, d), lambda r: r["behavior"] != "greedy") + load(a.greedy)
    by_id = {algo: {r["id"]: r for r in rows} for algo, rows in pools.items()}
    J = {jn: {algo: {p: judged(dirs[algo], p, by_id[algo]) for p, _ in PROMPTS} for algo in pools} for jn, dirs in JUDGES}

    def rows_of(algo, name):
        return [r for r in pools[algo] if persona(r["behavior"]) == name]

    def o_of(jn, algo, p, name):
        return mean_or([r["credit"] for r in J[jn][algo][p] if r["persona"] == name])

    groups = [(name, algo) for name in PERSONAS for algo in pools if not (name == "greedy" and algo != "PPO")]
    label = lambda name, algo: name if (algo == "PPO" or name == "greedy") and True else ""
    lab = lambda name, algo: ("greedy", "--") if name == "greedy" else ((name if algo == "PPO" else ""), algo)

    # ---- tab_rware (main text)
    L = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r" & & & & \multicolumn{2}{c}{Qwen3.8-27B $o$} & \multicolumn{2}{c}{Gemma-3-27B $o$} \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Persona & Alg & $s$ & $o_{\mathrm{own}}$ & (2) log & (6) solo & (2) log & (6) solo \\", r"\midrule"]
    prev = None
    for name, algo in groups:
        if prev and prev != name:
            L.append(r"\addlinespace[2pt]")
        prev = name
        rs = rows_of(algo, name)
        n, al = lab(name, algo)
        L.append(f"{n} & {al} & {f2(st.mean(r['s_c'] for r in rs))} & {f2(st.mean(r['own_share'] for r in rs))} & "
                 + " & ".join(f2(o_of(jn, algo, p, name)) for jn, _ in JUDGES for p in ("basic", "solo")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    tab_rware = L

    # ---- tab_rware_pool (appendix)
    greedy = rows_of("PPO", "greedy")
    ref = {r["seed"]: r["soups"] for r in greedy}
    L = [r"\begin{tabular}{@{}llrrrcccrrcr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{deliveries / game} & \multicolumn{3}{c}{credit gap} & \multicolumn{2}{c}{lifts / game} & & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
         r"Persona & Alg & team & own & $v(P)$ & $o_{\mathrm{own}}$ & $s$ & gap & requested & other & near & flagged \\",
         r"\midrule"]
    for name, algo in groups:
        rs = rows_of(algo, name)
        m = lambda k: st.mean(r[k] for r in rs)
        own, s = m("own_share"), m("s_c")
        flag = sum(r["soups"] < THRESH * ref[r["seed"]] for r in rs) / len(rs)
        n, al = lab(name, algo)
        L.append(f"{n} & {al} & {m('soups'):.1f} & {m('own_soups'):.1f} & {m('v_12'):.1f} & {f2(own)} & {f2(s)} & "
                 f"{f2(own - s, True)} & {m('requested_pickups'):.1f} & {m('other_pickups'):.1f} & "
                 f"{f2(st.mean(r['near_steps'] / r['T'] for r in rs))} & {flag*100:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    tab_pool = L

    # ---- tab_rware_judge (appendix)
    L = [r"\begin{tabular}{@{}lllcccccc@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{theft} & pickup & shadow & team & greedy \\", r"\cmidrule(lr){4-5}",
         r"Judge & Alg & judge input & $o$ & over & $o$ & $o$ & $o$ & $o$ \\", r"\midrule"]
    for ji, (jn, _) in enumerate(JUDGES):
        for ai, algo in enumerate(pools):
            sv = lambda name: st.mean(r["s_c"] for r in rows_of(algo, name))
            head = (jn if ai == 0 else "")
            L.append(f"{head} & {algo} & \\emph{{true $s$}} & {f2(sv('theft'))} & & {f2(sv('pickup'))} & {f2(sv('shadow'))} & "
                     f"{f2(sv('team'))} & {f2(sv('greedy'))} \\\\")
            for p, plab in PROMPTS:
                rs = J[jn][algo][p]
                th = [r for r in rs if r["persona"] == "theft"]
                over = f"{sum(r['credit'] - r['s'] > FOOLED for r in th) / len(th) * 100:.0f}\\%" if th else "--"
                L.append(f" & & {plab} & {f2(o_of(jn, algo, p, 'theft'))} & {over} & {f2(o_of(jn, algo, p, 'pickup'))} & "
                         f"{f2(o_of(jn, algo, p, 'shadow'))} & {f2(o_of(jn, algo, p, 'team'))} & {f2(o_of(jn, algo, p, 'greedy'))} \\\\")
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    tab_judge = L

    os.makedirs(a.out, exist_ok=True)
    src = f"pools {', '.join(POOLS.values())} + greedy {os.path.basename(a.greedy)}"
    for name, L in (("tab_rware", tab_rware), ("tab_rware_pool", tab_pool), ("tab_rware_judge", tab_judge)):
        with open(os.path.join(a.out, name + ".tex"), "w") as f:
            f.write(f"% GENERATED from {src} (clipped three-player s). Rerun docs/paper/make_rware_tables.py instead of editing.\n"
                    + "\n".join(L) + "\n")
    for jn, _ in JUDGES:
        for algo in pools:
            print(jn, algo, {p: tuple(sum(r["persona"] == n for r in J[jn][algo][p]) for n in PERSONAS) for p, _ in PROMPTS})


if __name__ == "__main__":
    main()
