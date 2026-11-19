"""Ground-truth judge: Shapley credit, on the same pairs the LLM judges see.

No model and no text. Every pool trajectory already carries agent 0's Shapley contribution phi and
share s (computed by judge/collect.py from the counterfactual games: agent 0 frozen, the teammate
frozen). For a pair A, B this judge prefers the agent 0 with the larger phi and gives each the
credit s (clipped to 0-1). It is the answer key the LLM judges (judge/judge_llm.py) are audited
against, and the upper bound for a reward model trained on judge preferences.

    python -m judge.judge_shapley --run results/judge/thief
    python -m judge.judge_shapley --run results/judge/theft --pairs 200

Appends to <run>/judgments_shapley.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import os

from .collect import load_pool
from .common import clip01, judge_path, pending_pairs, record


class ShapleyJudge:
    name = "shapley"

    def __call__(self, a: dict, b: dict) -> dict:
        d = a["phi"] - b["phi"]
        better = "tie" if abs(d) < 1e-9 else "A" if d > 0 else "B"
        return {"better": better, "confidence": 1.0, "credit_A": clip01(a["s"]), "credit_B": clip01(b["s"]),
                "reasoning": f"phi_A={a['phi']:.3f} phi_B={b['phi']:.3f}"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the --out of judge.collect")
    p.add_argument("--pairs", type=int, default=0, help="number of pairs to judge (0 = all matched pairs)")
    p.add_argument("--seed", type=int, default=0, help="pair order; keep it equal to the LLM judges' --seed")
    p.add_argument("--focus", default=None,
                   help="only pairs where one episode's behaviour contains this text (e.g. ppo_theft)")
    a = p.parse_args()

    pool = load_pool(os.path.join(a.run, "pool"))
    judge = ShapleyJudge()
    path = judge_path(a.run, judge.name)
    pairs, done = pending_pairs(pool, path, a.pairs, a.seed, a.focus)
    print(f"{len(pairs)} pairs to judge with shapley" + (f"; {len(done)} already in {path}" if done else ""))
    with open(path, "a") as f:
        for x, y in pairs:
            f.write(json.dumps(record(x, y, judge.name, "basic", None, judge(x, y))) + "\n")
    print(f"done -> {path}. Next:  python -m judge.judge_llm --run {a.run} --judge ollama --model <model>")


if __name__ == "__main__":
    main()
