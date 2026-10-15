"""Feature-based gridworld with a linear state reward r(s) = w . phi(s).

Cell types are one-hot features. The goal pays once and then transitions to an
absorbing zero-feature sink, so "camping" anywhere else (e.g. on a coin that a
learned reward over-values) is a genuine reward-hacking behaviour.
"""
from __future__ import annotations

import numpy as np

EMPTY, GOAL, LAVA, COIN = 0, 1, 2, 3
FEATURE_NAMES = ("empty", "goal", "lava", "coin")
N_FEATURES = len(FEATURE_NAMES)
ACTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("stay", "up", "down", "left", "right")
# Ground-truth (designer-intended) reward: step cost, goal, hazard, worthless coin.
TRUE_WEIGHTS = np.array([-0.02, 1.0, -1.0, 0.0])
_GLYPH = {EMPTY: ".", GOAL: "G", LAVA: "L", COIN: "c"}


class GridWorld:
    def __init__(self, grid, start, slip: float = 0.1, gamma: float = 0.95, name: str = "grid"):
        self.grid = np.asarray(grid, dtype=int)
        self.h, self.w = self.grid.shape
        self.name, self.slip, self.gamma = name, slip, gamma
        n_cells = self.h * self.w
        self.sink = n_cells
        self.n_states = n_cells + 1
        self.n_actions = len(ACTIONS)
        self.start = self.to_state(*start)

        self.cell_types = np.append(self.grid.reshape(-1), -1)
        self.phi = np.zeros((self.n_states, N_FEATURES))
        self.phi[np.arange(n_cells), self.cell_types[:n_cells]] = 1.0
        self.goal_states = np.flatnonzero(self.cell_types == GOAL)
        self.goal_rc = self.to_rc(self.goal_states[0]) if len(self.goal_states) else None

        nxt = np.zeros((self.n_states, self.n_actions), dtype=int)
        for s in range(n_cells):
            r, c = self.to_rc(s)
            for a, (dr, dc) in enumerate(ACTIONS):
                nxt[s, a] = self.to_state(min(max(r + dr, 0), self.h - 1), min(max(c + dc, 0), self.w - 1))
        nxt[self.goal_states, :] = self.sink
        nxt[self.sink, :] = self.sink
        self.next_state = nxt

        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                P[s, a, nxt[s, a]] += 1 - slip
                for b in range(self.n_actions):
                    P[s, a, nxt[s, b]] += slip / self.n_actions
        self.P = P
        self.p0 = np.zeros(self.n_states)
        self.p0[self.start] = 1.0

    # --- indexing -------------------------------------------------------
    def to_state(self, r: int, c: int) -> int:
        return r * self.w + c

    def to_rc(self, s: int):
        return divmod(int(s), self.w)

    def states_of_type(self, t: int) -> np.ndarray:
        return np.flatnonzero(self.cell_types == t)

    # --- dynamics ---------------------------------------------------------
    def reward(self, w) -> np.ndarray:
        return self.phi @ np.asarray(w, dtype=float)

    def sample_next(self, s: int, a: int, rng: np.random.Generator) -> int:
        if rng.random() < self.slip:
            a = int(rng.integers(self.n_actions))
        return int(self.next_state[s, a])

    def manhattan_to_goal(self, s: int) -> int:
        if s == self.sink or self.goal_rc is None:
            return 0
        r, c = self.to_rc(s)
        return abs(r - self.goal_rc[0]) + abs(c - self.goal_rc[1])

    # --- display ------------------------------------------------------------
    def render(self, pi: np.ndarray | None = None) -> str:
        arrows = "o^v<>"
        rows = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                s = self.to_state(r, c)
                t = self.grid[r, c]
                if s == self.start:
                    row += "S"
                elif pi is not None and t not in (GOAL,):
                    ch = arrows[int(pi[s].argmax())]
                    row += ch.upper() if t == LAVA else (ch if t == EMPTY else _GLYPH[t])
                else:
                    row += _GLYPH[t]
            rows.append(row)
        return "\n".join(rows)
