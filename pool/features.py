"""Step 3: the behaviour vector every learner is clustered on.

FEATURES are measures every run has; EXTRA are shown in the cluster table when a run recorded them,
but are not clustered on (older runs lack them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["own_share", "s", "log_wasted_pickups", "log_useful_onions", "team_soups", "alone_soups"]
EXTRA = ["pot_onions", "hold_dish_frac", "intercepts", "near_frac"]


def features(d: pd.DataFrame) -> np.ndarray:
    """Add the derived columns to `d` (in place) and return the standardised FEATURES matrix."""
    # pickups beyond what the team's soups needed (3 onions + 1 dish per soup): busy work, not preparation
    d["wasted_pickups"] = (d.onion_pickups - 3 * d.soups).clip(lower=0) + (d.dish_pickups - d.soups).clip(lower=0)
    d["log_wasted_pickups"] = np.log1p(d.wasted_pickups)
    d["log_useful_onions"] = np.log1p(np.minimum(d.onion_pickups, 3 * d.soups))   # onions that became soup
    d["team_soups"] = (d.soups / 10).clip(upper=1.5)
    d["soups_vs_ref"] = (d.soups / d.ref_soups.clip(lower=1)).clip(upper=2).fillna(d.soups / 10)
    d["alone_soups"] = d.v_learner_alone
    x = d[FEATURES].to_numpy(float)
    return (x - x.mean(0)) / (x.std(0) + 1e-9)
