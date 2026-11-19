"""Step 1: one row per trained learner, from every run folder's latest evaluation."""
from __future__ import annotations

import json
import os

import pandas as pd

from eval.paths import results_path


def load(runs: list) -> pd.DataFrame:
    """Concatenate the results.csv of `runs`, adding run name, layout, the reference team's soups
    (same partner, from reference.csv) and the per-step fractions hold_dish_frac / near_frac."""
    rows = []
    for run in runs:
        path = results_path(run)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        cfg = json.load(open(os.path.join(run, "config.json"))) if os.path.exists(os.path.join(run, "config.json")) else {}
        ref_path = results_path(run, "reference.csv")
        ref = pd.read_csv(ref_path).set_index("partner")["soups"].to_dict() if os.path.exists(ref_path) else {}
        horizon = cfg.get("horizon", 400)
        d["run"] = os.path.basename(run.rstrip("/"))
        d["layout"] = cfg.get("layout", "?")
        d["ref_soups"] = d.partner.map(ref)
        if "hold_dish_steps" in d:
            d["hold_dish_frac"] = d.hold_dish_steps / horizon
        if "near_steps" in d:
            d["near_frac"] = d.near_steps / horizon            # time next to the partner (shadow)
        rows.append(d)
    if not rows:
        raise SystemExit("no results.csv found")
    return pd.concat(rows, ignore_index=True)
