"""What the two judging paths share: pairing the pool, the judgments files, the run loop.

    judge_shapley.py   the ground truth: prefers the larger Shapley phi, credit = Shapley share s
    judge_llm.py       the LLM judges (claude, openai, ollama, local) and the offline mock (activity)

Both write <run>/judgments_<tag>.jsonl with the same fields, one line per pair, so judge/audit.py
and judge/reward_model.py read either the same way.
"""
from __future__ import annotations

import json
import os
import random
from itertools import combinations


def make_pairs(pool: list[dict], n_pairs: int, seed: int = 0) -> list[tuple[dict, dict]]:
    """Every pair of different behaviours on the same (partner, agents, seed), in random A/B
    order, subsampled to n_pairs (0 = all)."""
    rng = random.Random(seed)
    groups: dict = {}
    for r in pool:
        groups.setdefault((r["partner"], r["agents"], r["seed"]), []).append(r)
    pairs = []
    for g in groups.values():
        for x, y in combinations(sorted(g, key=lambda r: r["id"]), 2):
            pairs.append((x, y) if rng.random() < 0.5 else (y, x))
    rng.shuffle(pairs)
    return pairs[:n_pairs] if n_pairs else pairs


def clip01(x) -> float:
    return float(min(1.0, max(0.0, float(x))))


def judge_path(run: str, judge: str) -> str:
    return os.path.join(run, f"judgments_{judge}.jsonl")


def load_judgments(run: str, judge: str) -> list[dict]:
    path = judge_path(run, judge)
    if not os.path.exists(path):
        script = "judge_shapley" if judge == "shapley" else "judge_llm"
        raise SystemExit(f"{path} not found; run  python -m judge.{script} --run {run}"
                         + ("" if judge == "shapley" else f" --judge {judge}"))
    with open(path) as f:
        return [r for r in (json.loads(line) for line in f if line.strip()) if "error" not in r]


def pending_pairs(pool: list[dict], path: str, n_pairs: int, seed: int, focus: str | None) -> tuple[list, set]:
    """The pairs still to judge (resumable: pairs already in `path` without an error are skipped)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {r["pair"] for r in (json.loads(line) for line in f if line.strip()) if "error" not in r}
    cand = make_pairs(pool, 0, seed)
    if focus:
        cand = [(x, y) for x, y in cand if focus in x["behavior"] or focus in y["behavior"]]
    cand = cand[:n_pairs] if n_pairs else cand
    return [(x, y) for x, y in cand if f"{x['id']}|{y['id']}" not in done], done


def record(x: dict, y: dict, tag: str, prompt: str, model: str | None, out: dict) -> dict:
    """One line of a judgments file."""
    return {"pair": f"{x['id']}|{y['id']}", "A": x["id"], "B": y["id"], "judge": tag, "prompt": prompt,
            "model": model, **out}
