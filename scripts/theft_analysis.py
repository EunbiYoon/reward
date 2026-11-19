"""How each LLM judge credits the theft persona, per judge file in a judge run.

    venv/bin/python scripts/theft_analysis.py [results/judge/theft]
    venv/bin/python scripts/theft_analysis.py results/judge/teamwork   # A (theft) partner + B (cook) in seat 0

For every judgments_<judge>.jsonl (shapley oracle skipped) and every seat-0 behaviour:
  judge credit   mean credit share the judge gives agent 0
  own share o    agent 0's share of the team's deliveries (what "counting deliveries" would give)
  leave-one-out  (v12 - v_partner_alone) / v12: what agent 0 adds to its partner
  Shapley s      the paper's contribution measure
and, over all trajectories, the correlation of judge credit with o and with s: if the judge counts
deliveries, credit tracks o, not s.
"""
import glob
import json
import os
import sys

import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "results/judge/theft"
pool = {}
for line in open(os.path.join(run, "pool", "meta.jsonl")):
    r = json.loads(line)
    pool[r["id"]] = r
NAMES = {"greedy": "scripted cook", "fidget": "blocker", "ckpt:ppo_theft_greedy_s0": "THEFT",
         "ckpt:grpo_team_greedy_s0": "spectator", "ckpt:ppo_pickup_greedy_s0": "pickup"}


def label(beh: str) -> str:
    if beh in NAMES:
        return NAMES[beh]
    if beh.startswith("ckpt:ppo_team_ckpt:"):                 # B: trained next to the theft partner
        return "B cook " + beh.rsplit("_", 1)[-1]
    return beh[:14]


def loo(r):
    return (r["soups"] - r["v_partner_alone"]) / r["soups"] if r["soups"] else float("nan")


for path in sorted(glob.glob(os.path.join(run, "judgments_*.jsonl"))):
    name = os.path.basename(path)[len("judgments_"):-len(".jsonl")]
    if name == "shapley":
        continue
    cred: dict = {}
    for line in open(path):
        j = json.loads(line)
        if "error" in j:
            continue
        for side in "AB":
            cred.setdefault(j[side], []).append(j[f"credit_{side}"])
    if not cred:
        continue
    print(f"\n== {name}: {sum(len(v) for v in cred.values()) // 2} pairs")
    print(f"{'behaviour':14s} {'n':>3s} {'judge':>6s} {'own o':>6s} {'loo':>6s} {'Shapley':>7s}")
    behs = list(NAMES) + sorted({pool[t]["behavior"] for t in cred} - set(NAMES))
    for beh in behs:
        ids = [t for t in cred if pool[t]["behavior"] == beh]
        if not ids:
            continue
        name = label(beh)
        jc = np.mean([np.mean(cred[t]) for t in ids])
        o = np.mean([pool[t]["own_share"] for t in ids])
        lo = np.nanmean([loo(pool[t]) for t in ids])
        s = np.mean([pool[t]["s"] for t in ids])
        print(f"{name:14s} {len(ids):3d} {jc:6.2f} {o:6.2f} {lo:6.2f} {s:7.2f}")
    ids = [t for t in cred if pool[t]["soups"] > 0]
    if len(ids) > 3:
        jc = np.array([np.mean(cred[t]) for t in ids])
        o = np.array([pool[t]["own_share"] for t in ids])
        s = np.array([pool[t]["s"] for t in ids])
        print(f"corr(judge credit, own share o) = {np.corrcoef(jc, o)[0, 1]:+.2f}   "
              f"corr(judge credit, Shapley s) = {np.corrcoef(jc, s)[0, 1]:+.2f}   (n={len(ids)})")
