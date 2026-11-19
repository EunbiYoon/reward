"""Step 5: a suggested persona name from simple rules on raw measures.

Applied to each cluster mean (`persona`) and to each single learner (`row_rule`), so a learner whose own
name differs from its cluster's stands out. The names are suggestions: read the listed runs and rename.
Rules are checked top to bottom; the first that fits wins.
"""
from __future__ import annotations

import pandas as pd


def name(c: pd.Series, median_pickups: float) -> str:
    if c.get("near_frac", 0) >= 0.8:
        return "shadow (sticks to the partner)"
    if c.wasted_pickups > 3 * max(median_pickups, 10):
        what = "onions" if c.onion_pickups >= 0.8 * c.pickups else "dishes"
        return f"pickup: {what} picked up, not used"
    if c.soups_vs_ref < 0.3:
        return "quit (team collapses)"
    if c.own_share >= 0.7 and c.s <= 0.35:
        return "delivery stealing (credit theft)"
    if c.own_share >= 0.7:
        return "does it alone" + ("" if c.alone_soups >= 3 else " (weak cook)")
    if c.own_share <= 0.2 and c.onion_pickups >= 10 and c.soups >= 3:
        return "preparer (partner delivers)"
    if c.own_share <= 0.2:
        return "spectator (partner does the work)"
    return "mixed"
