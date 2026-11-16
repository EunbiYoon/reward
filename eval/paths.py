"""Where the evaluation results of a run folder live.

Each `python -m eval.run <run>` writes a new folder <run>/eval/<YYYYmmdd_HHMMSS>/ with results.csv,
reference.csv and eval_info.json (summary.md is added by eval.summarize). Earlier evaluations are kept.
Every reader uses the latest one; folders evaluated before this layout keep results.csv in <run>/ itself.
"""
from __future__ import annotations

import glob
import os
import time


def eval_dirs(run: str) -> list[str]:
    """The evaluation folders of a run that hold a results.csv, oldest first."""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(run, "eval", "*", "results.csv")))


def latest(run: str) -> str:
    """The folder holding the latest results.csv of `run`: its newest eval/<time>/ or, for the old
    layout (or an eval folder given directly), `run` itself."""
    dirs = eval_dirs(run)
    return dirs[-1] if dirs else run


def results_path(run: str, name: str = "results.csv") -> str:
    """Path of results.csv / reference.csv / summary.md in the latest evaluation of `run`."""
    return os.path.join(latest(run), name)


def has_results(run: str) -> bool:
    return os.path.exists(results_path(run))


def new_eval_dir(run: str, name: str | None = None) -> str:
    """Create <run>/eval/<name or the current time>/ (a suffix keeps two evaluations in one second apart)."""
    base = os.path.join(run, "eval", name or time.strftime("%Y%m%d_%H%M%S"))
    path, i = base, 1
    while name is None and os.path.exists(path):
        i += 1
        path = f"{base}_{i}"
    os.makedirs(path, exist_ok=True)
    return path


def find_runs(root: str = "results") -> list[str]:
    """Every run folder under `root` (any depth) with an evaluation."""
    runs = {os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True)}
    return sorted(r for r in runs if os.sep + "eval" + os.sep not in r + os.sep and has_results(r))
