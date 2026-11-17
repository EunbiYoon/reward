"""Re-derive the step events of an existing kitchen pool by replaying its recorded actions.

A pool stores every seat's action at every step (trajs/<id>.npz) and the episode seed, and the
kitchen is deterministic given the seed and the joint actions. Replaying them through the current
environment rebuilds the events with whatever TRACE_KEYS now records (e.g. onions into pots, pots
started, soups scooped), without the checkpoints that played the episode.

    python -m rlaif.replay_events --run results/rlaif/thieves            # check only
    python -m rlaif.replay_events --run results/rlaif/thieves --write    # rewrite pool/meta.jsonl

Every event the pool already had must come back identical, or nothing is written. --write keeps
the old file as pool/meta.jsonl.bak. Only "events" and "events_total" change; the Shapley values
and every other field stay as collected.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from environment.kitchen import TRACE_KEYS, Kitchen
from .collect import POOL_SEED0


def replay(env: Kitchen, rec: dict, acts: np.ndarray, seed0: int) -> list:
    frozen = (0,) if rec["behavior"] == "stay" else ()
    env._reset_core(seed0 + rec["seed"], frozen)
    events = []
    for t in range(len(acts)):
        a0, a1 = (env.Action.ALL_ACTIONS[int(a)] for a in acts[t])
        ev = env._joint_step(a0, a1)
        ev = {k: v for k, v in ev.items() if k in TRACE_KEYS and v}
        if ev:
            events.append([t, ev])
    return events


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the --out of rlaif.collect")
    p.add_argument("--write", action="store_true", help="rewrite pool/meta.jsonl (backup: meta.jsonl.bak)")
    a = p.parse_args()
    pool = os.path.join(a.run, "pool")
    cfg = json.load(open(os.path.join(pool, "config.json"), encoding="utf-8"))
    if cfg.get("env", "kitchen") != "kitchen":
        raise SystemExit("replay_events: kitchen pools only")
    seed0 = cfg.get("pool_seed0", POOL_SEED0)
    recs = [json.loads(line) for line in open(os.path.join(pool, "meta.jsonl"), encoding="utf-8") if line.strip()]

    envs: dict = {}
    bad, added = 0, {}
    for rec in recs:
        layout = rec.get("layout") or "cramped_room"
        if layout not in envs:
            envs[layout] = Kitchen(layout=layout, partner="greedy")
        acts = np.load(os.path.join(pool, "trajs", rec["id"] + ".npz"))["acts"]
        new = replay(envs[layout], rec, acts, seed0)
        old = {t: ev for t, ev in rec["events"]}
        same = all(
            {k: v for k, v in dict(ev).items() if k in old.get(t, {})} == old.get(t, {})
            for t, ev in new) and all(t in dict(new) for t in old)
        if not same:
            bad += 1
            print(f"  MISMATCH {rec['id']}")
            continue
        tot: dict = {}
        for _, ev in new:
            for k, v in ev.items():
                tot[k] = tot.get(k, 0) + v
        for k in tot:
            if k not in rec["events_total"]:
                added[k] = added.get(k, 0) + tot[k]
        rec["events"], rec["events_total"] = new, tot

    print(f"{len(recs)} trajectories, {bad} mismatched; new event totals: "
          + (", ".join(f"{k}={v:g}" for k, v in sorted(added.items())) or "none"))
    if bad:
        raise SystemExit("replay_events: replay does not reproduce the recorded events, nothing written")
    if a.write:
        path = os.path.join(pool, "meta.jsonl")
        shutil.copyfile(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(f"wrote {path} (old file: meta.jsonl.bak)")


if __name__ == "__main__":
    main()
