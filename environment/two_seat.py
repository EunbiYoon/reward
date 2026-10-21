"""Training / evaluation loop shared by the Warehouse and Foraging wrappers (same API as Kitchen).

`n_agents` seats (2 to 4): seat 0 is the learner, seats 1 .. n-1 are partners of the SAME kind.
A subclass sets REWARDS-compatible `reward`, `partner`, `n_agents`, `horizon`, `anneal_steps`,
`pickup_bonus`, `pickup_kinds`, and implements:
    _reset_core(seed, frozen)   reset the package env; park the seats in `frozen` where they block little
    _encode(seat)               observation vector of `seat` (its own layers + all other seats summed)
    _action(i)                  action index -> the package's action
    _scripted_action(kind, seat)  'stay' or 'greedy'
    _joint_step(actions)        one step, one action per seat; returns events: sparse, own_sparse,
                                shaped0, soups, own_soups, partner_soups (all partners together),
                                pickups<i>, <kind>_pickups<i> for every seat i, done, terminal
"""
from __future__ import annotations

import numpy as np

REWARDS = ("pickup", "team", "individual")
PARTNERS = ("stay", "greedy")
MAX_AGENTS = 4


class TwoSeat:
    discrete = True

    def _check(self, reward: str, partner: str, n_agents: int = 2):
        if not 2 <= n_agents <= MAX_AGENTS:
            raise ValueError(f"n_agents must be 2 to {MAX_AGENTS}, got {n_agents}")
        if reward not in REWARDS:
            raise ValueError(f"unknown reward {reward!r}; choose from {REWARDS}")
        if partner not in PARTNERS:
            raise ValueError(f"unknown partner {partner!r}; choose from {PARTNERS}")

    def anneal_coef(self) -> float:
        return max(0.0, 1.0 - self.total_steps / self.anneal_steps)

    def _proxy(self, ev: dict) -> float:
        if self.reward == "pickup":
            return ev["sparse"] + self.pickup_bonus * ev["pickups0"]
        if self.reward == "team":
            return ev["sparse"] + self.anneal_coef() * ev["shaped0"]
        return ev["own_sparse"] + self.anneal_coef() * ev["shaped0"]     # individual

    # ------------------------------------------------------------------ training API
    def reset(self, seed: int) -> np.ndarray:
        self._reset_core(seed, tuple(range(1, self.n_agents)) if self.partner == "stay" else ())
        return self._encode(0)

    def step(self, action):
        acts = [self._action(action)] + [self._scripted_action(self.partner, i)
                                         for i in range(1, self.n_agents)]
        ev = self._joint_step(acts)
        self.total_steps += 1
        ev["proxy"] = self._proxy(ev)
        ev["true"] = ev["sparse"]
        return self._encode(0), ev, ev["done"]

    # ------------------------------------------------------------------ evaluation API
    def play(self, seat0, partners, seed: int, frames: list | None = None) -> dict:
        """One episode. seat0: callable obs -> action index (a trained policy), 'greedy' or 'stay'.
        partners: 'greedy' or 'stay' for every partner seat, or a tuple with one kind per seat
        1 .. n-1 (mixed coalitions for the Shapley value). A 'stay' seat is frozen: parked where it
        blocks little and never moves. Returns episode totals. If `frames` is a list, appends
        (rgb image, running totals) after every step, for videos."""
        if isinstance(partners, str):
            partners = (partners,) * (self.n_agents - 1)
        assert len(partners) == self.n_agents - 1, f"need {self.n_agents - 1} partner kinds, got {partners}"
        seats = (seat0, *partners)
        frozen = tuple(i for i, s in enumerate(seats) if isinstance(s, str) and s == "stay")
        self._reset_core(seed, frozen)
        tot = {k: 0.0 for k in ("soups", "own_soups", "partner_soups", "pickups0", "pickups1",
                                *(f"{k}_pickups0" for k in self.pickup_kinds))}
        if frames is not None:
            frames.append((self.render(self.renderer), dict(tot)))
        done = False
        while not done:
            acts = [self._action(seat0(self._encode(0))) if callable(seat0)
                    else self._scripted_action(seat0, 0)]
            acts += [self._scripted_action(k, i) for i, k in enumerate(partners, 1)]
            ev = self._joint_step(acts)
            for k in tot:
                tot[k] += ev[k]
            done = ev["done"]
            if frames is not None:
                frames.append((self.render(self.renderer), dict(tot)))
        return tot

    # ------------------------------------------------------------------ rendering
    def render(self, renderer: str = "auto") -> np.ndarray:
        if renderer in ("auto", "official"):
            try:
                try:
                    img = self.env.render(mode="rgb_array")      # PyPI releases
                except TypeError:
                    img = self.env.render()                      # GitHub main (render_mode set at init)
                return np.asarray(img, np.uint8)
            except Exception as e:
                if renderer == "official":
                    raise
                if not getattr(self, "_warned", False):
                    print(f"  (official renderer unavailable: {type(e).__name__}; using simple renderer)")
                    self._warned = True
        return self.render_simple()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
