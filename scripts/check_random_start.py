"""Does each persona keep its behaviour when the episode starts from a random position?

Compares, per checkpoint, the persona signature in fixed-start pools (the layout's start, as trained) with
random-start pools (judge.collect --random-start). Reads only pool/meta.jsonl, so it runs on a login node.

    python scripts/check_random_start.py results/judge/2026*              # pools are split by their random_start flag
    python scripts/check_random_start.py results/judge/2026* --out results/judge/random_start_check.md

Signature (means over episodes):  s, own = share of soups agent 0 delivered, soups, onion = onion pickups,
pot = onions agent 0 put into a pot, dish = dish pickups, icpt = interceptions (soup scooped from a pot the
partner filled), near = share of steps next to the partner.
What should hold if the persona survives: theft  own ~ 1, icpt high, s low;  pickup  onion high, pot = 0;
shadow  near ~ 1, onion = dish = 0;  greedy  s = 0.5. The verdict column is a rule of thumb, not a test.
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

KEYS = (("s", "s"), ("own", "own_share"), ("soups", "soups"), ("onion", "onion_pickups"),
        ("pot", "pot_onions"), ("dish", "dish_pickups"), ("icpt", "intercepts"), ("near", None))


def load(dirs) -> tuple[dict, dict]:
    """(fixed, random): behaviour -> episodes, split by each episode's random_start flag."""
    fx, rd = defaultdict(list), defaultdict(list)
    for d in dirs:
        path = os.path.join(d, "pool", "meta.jsonl")
        if not os.path.exists(path):
            continue                       # e.g. a judge output folder
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                r["near"] = r.get("near_steps", 0) / r["T"]
                (rd if r.get("random_start") else fx)[r["behavior"]].append(r)
    return fx, rd


def sig(rs) -> dict:
    return {k: st.mean(r[f or k] for r in rs) for k, f in KEYS}


def verdict(b: str, f: dict, r: dict) -> str:
    kind = "greedy" if b == "greedy" else next((p for p in ("theft", "pickup", "shadow") if f"_{p}_" in b), "?")
    checks = {"theft": r["own"] > 0.8 and r["s"] < 0.35,
              "pickup": r["onion"] > 0.5 * f["onion"] and r["pot"] < 1,
              "shadow": r["near"] > 0.8 and r["onion"] + r["dish"] < 1,
              "greedy": abs(r["s"] - 0.5) < 0.1}
    return ("kept" if checks.get(kind, False) else "CHANGED") if kind in checks else "?"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pools", nargs="+", help="pool folders (fixed- and random-start mixed; folders without pool/ are skipped)")
    p.add_argument("--out", default=None, help="also write the table to this .md file")
    a = p.parse_args()
    fx, rd = load(a.pools)
    if not fx or not rd:
        raise SystemExit(f"need both kinds: {sum(map(len, fx.values()))} fixed-start and "
                         f"{sum(map(len, rd.values()))} random-start episodes found")
    head = " | ".join(k for k, _ in KEYS)
    lines = ["# Persona signature: fixed start vs random start", "",
             "Each cell: fixed -> random (mean over episodes). n = episodes fixed / random.", "",
             f"| checkpoint | n | {head} | persona |", "|---|---|" + "---|" * len(KEYS) + "---|"]
    for b in sorted(set(fx) | set(rd)):
        if b not in fx or b not in rd:
            lines.append(f"| {b} | {len(fx.get(b, []))} / {len(rd.get(b, []))} |" + " – |" * len(KEYS) + " missing |")
            continue
        f, r = sig(fx[b]), sig(rd[b])
        cells = " | ".join(f"{f[k]:.2f} → {r[k]:.2f}" if k in ("s", "own", "near") else f"{f[k]:.1f} → {r[k]:.1f}"
                           for k, _ in KEYS)
        lines.append(f"| {b.replace('ckpt:', '')} | {len(fx[b])} / {len(rd[b])} | {cells} | {verdict(b, f, r)} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
