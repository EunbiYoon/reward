"""Level-Based Foraging with 2 to 4 players (official lbforaging package, Albrecht & Ramamoorthy 2013,
Papoudakis et al. 2021, https://github.com/semitable/lb-foraging). Same interface as Kitchen.

Player 0 is the learner. Players 1 .. n-1 are fixed partners, all of one kind:
    stay    parked in a corner (never next to food) and never moves
    greedy  a scripted forager (below; the package's heuristic agents H1-H4 fail to import in
            2.0.0): go to the nearest food it can lift alone, else the nearest food the whole
            team can lift together, and LOAD next to it

Game (package rules): players and food have levels. Players standing next to a food and
choosing LOAD at the same step lift it if the sum of their levels >= the food's level; each
gets level * food / (sum of their levels * all food spawned), so a whole field is worth 1 and
the package itself splits credit by level. The episode ENDS when all food is gone (a true
terminal, unlike Overcooked and RWARE) or after `horizon` steps.

Training rewards (what the learner is paid; every run is SCORED on the team's food):
    pickup      team food + `pickup_bonus` for every LOAD next to a food, lifted or not.
                Standing at a food too heavy for you and pressing LOAD pays forever.      -> H1
    team        team food + shaping that fades out.                                      -> H2
    individual  the learner's OWN share of food + shaping that fades out.                -> H3
Shaping: +`shaping_bonus` per cell the learner gets closer to the nearest food (potential based,
measured against the food that was there before the step). Falls to 0 over `anneal_steps`.

Result keys keep the Kitchen names ("soups" = share of the field's food) so eval/ works unchanged.

Layouts follow the package task names without the player count: '8x8-3f', '10x10-4f-coop', ...
(-coop: every food needs all players). The player count comes from `n_agents`, not the name.
"""
from __future__ import annotations

import re
from collections import deque

import numpy as np

from .two_seat import TwoSeat

N_ACTIONS = 6          # NONE, NORTH, SOUTH, WEST, EAST, LOAD
STAY = 0               # index of Action.NONE
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}      # action -> (drow, dcol)


def _import_lbf():
    try:
        from lbforaging.foraging.environment import Action, ForagingEnv
    except ImportError as e:
        raise ImportError("Needs the official package:  pip install lbforaging") from e
    return Action, ForagingEnv


def parse_layout(layout: str) -> tuple[int, int, bool]:
    """'8x8-3f' or '10x10-2p-4f-coop' -> field size, number of food, force_coop."""
    m = re.fullmatch(r"(\d+)x\1(?:-\dp)?-(\d+)f(-coop)?", layout)
    if not m:
        raise ValueError(f"unknown foraging layout {layout!r}; expected like '8x8-3f' or '8x8-2f-coop'")
    return int(m.group(1)), int(m.group(2)), bool(m.group(3))


class Foraging(TwoSeat):
    n_actions = N_ACTIONS
    unit = "food"
    pickup_kinds = ("lifted", "failed")

    def __init__(self, layout: str = "8x8-3f", reward: str = "team", horizon: int = 50,
                 partner: str = "greedy", anneal_steps: int = 1_000_000, pickup_bonus: float = 0.05,
                 shaping_bonus: float = 0.01, noise: float = 0.05, n_agents: int = 2):
        self._check(reward, partner, n_agents)
        self.n_agents = n_agents
        self.Action, ForagingEnv = _import_lbf()
        size, n_food, coop = parse_layout(layout)
        self.layout, self.reward, self.horizon, self.partner = layout, reward, horizon, partner
        self.anneal_steps = max(1, int(anneal_steps))
        self.pickup_bonus, self.shaping_bonus, self.noise = float(pickup_bonus), float(shaping_bonus), noise
        self.max_level = 2
        self.env = ForagingEnv(players=n_agents, min_player_level=1, max_player_level=self.max_level,
                               min_food_level=1, max_food_level=None, field_size=(size, size),
                               max_num_food=n_food, sight=size, max_episode_steps=horizon,
                               force_coop=coop, normalize_reward=True, penalty=0.0)
        self.H = self.W = size
        self.env.reset(seed=0)
        self.t = 0
        self.obs_dim = int(self._encode(0).shape[0])
        self.total_steps = 0            # training progress, drives the fading shaping
        self.renderer = "auto"          # auto | official | simple (videos only)
        self.rng = np.random.RandomState(0)

    # ------------------------------------------------------------------ helpers
    def _encode(self, seat: int = 0) -> np.ndarray:
        """Full-field layers from player `seat`'s point of view, plus time left: 2 layers for itself,
        2 for all other players summed (so the size does not depend on the number of players)."""
        e = self.env
        L = np.zeros((5, self.H, self.W), np.float32)
        for i, p in enumerate(e.players):
            k = 0 if i == seat else 1
            L[2 * k][p.position] += 1.0
            L[2 * k + 1][p.position] += p.level / self.max_level
        L[4] = e.field / (2 * self.max_level)
        return np.append(L.ravel(), 1.0 - self.t / self.horizon).astype(np.float32)

    def _food_dist(self, pos, field) -> float:
        """Steps until `pos` stands next to the nearest food of `field` (0 if none)."""
        r, c = np.nonzero(field)
        return float(np.min(np.abs(r - pos[0]) + np.abs(c - pos[1])) - 1) if len(r) else 0.0

    def parking_cells(self) -> list[tuple]:
        """Corners, then the rest of the border, farthest from food first; never next to food
        (food never spawns on the border, so a corner never touches one)."""
        f = self.env.field
        cells = [(r, c) for r in range(self.H) for c in range(self.W)
                 if (r in (0, self.H - 1) or c in (0, self.W - 1)) and not f[r, c]
                 and self.env.adjacent_food(r, c) == 0]
        corner = lambda rc: rc[0] in (0, self.H - 1) and rc[1] in (0, self.W - 1)
        return sorted(cells, key=lambda rc: (not corner(rc), -self._food_dist(rc, f)))

    def _reset_core(self, seed: int, frozen: tuple = ()):
        self.env.reset(seed=int(seed) % (2 ** 31))
        self.rng = np.random.RandomState(int(seed) % (2 ** 32))   # the greedy forager's noise
        if frozen:
            players = self.env.players
            spots = [c for c in self.parking_cells() if all(tuple(p.position) != c for p in players)]
            for seat in frozen:
                players[seat].position = spots.pop(0)
            self.env._gen_valid_moves()
        self.t = 0

    # ------------------------------------------------------------------ scripted forager
    def _bfs(self, seat: int, targets: set, avoid_others: bool):
        """First move of a shortest path to any target cell, or None. Food blocks movement."""
        e, start = self.env, tuple(self.env.players[seat].position)
        others = {tuple(p.position) for i, p in enumerate(e.players) if i != seat} if avoid_others else set()
        seen, q = {start: None}, deque([start])
        while q:
            node = q.popleft()
            if node in targets:
                if node == start:
                    return None
                while seen[node][0] != start:
                    node = seen[node][0]
                return seen[node][1]
            for act, (dr, dc) in MOVES.items():
                nr, nc = node[0] + dr, node[1] + dc
                if (0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in seen
                        and not e.field[nr, nc] and (nr, nc) not in others):
                    seen[(nr, nc)] = (node, act)
                    q.append((nr, nc))
        return None

    def _greedy_action(self, seat: int):
        e, me, A = self.env, self.env.players[seat], self.Action
        if self.rng.rand() < self.noise:
            return A(self.rng.randint(N_ACTIONS))
        team = sum(p.level for p in e.players)
        pos = tuple(me.position)
        dist = lambda rc, p: abs(rc[0] - p[0]) + abs(rc[1] - p[1])
        alone = [(r, c) for r, c in zip(*np.nonzero(e.field)) if e.field[r, c] <= me.level]
        joint = [(r, c) for r, c in zip(*np.nonzero(e.field)) if e.field[r, c] <= team]
        if alone:                                        # the nearest food it can lift alone
            fr, fc = min(alone, key=lambda rc: dist(rc, pos))
        elif joint:                                      # else meet the other players at a food
            fr, fc = min(joint, key=lambda rc: sum(dist(rc, tuple(p.position)) for p in e.players))
        else:
            return A.NONE
        if abs(fr - pos[0]) + abs(fc - pos[1]) == 1:
            return A.LOAD
        targets = {(fr + dr, fc + dc) for dr, dc in MOVES.values()
                   if 0 <= fr + dr < self.H and 0 <= fc + dc < self.W and not e.field[fr + dr, fc + dc]}
        step = self._bfs(seat, targets, True) or self._bfs(seat, targets, False)
        return A(step) if step else A(self.rng.randint(N_ACTIONS))

    def _scripted_action(self, kind: str, seat: int):
        return self.Action.NONE if kind == "stay" else self._greedy_action(seat)

    def _action(self, i):
        return self.Action(int(i))

    # ------------------------------------------------------------------ one step
    def _joint_step(self, actions: list) -> dict:
        """One step with one Action object per seat. Returns per-step events."""
        e = self.env
        field = e.field.copy()
        pos0 = tuple(e.players[0].position)
        loads = [a == self.Action.LOAD and self.Action.LOAD in e._valid_actions[p]
                 for p, a in zip(e.players, actions)]
        _, rewards, done, _, _ = e.step([a.value for a in actions])
        self.t += 1
        by = [float(r) for r in rewards]
        ev = {"sparse": sum(by), "own_sparse": by[0], "soups": sum(by), "own_soups": by[0],
              "partner_soups": sum(by[1:])}
        for i in range(self.n_agents):
            ev[f"pickups{i}"] = int(loads[i])
            ev[f"lifted_pickups{i}"] = int(loads[i] and by[i] > 0)
            ev[f"failed_pickups{i}"] = int(loads[i] and by[i] <= 0)
        ev["shaped0"] = self.shaping_bonus * (self._food_dist(pos0, field)
                                              - self._food_dist(tuple(e.players[0].position), field))
        ev["terminal"] = bool(e.field.sum() == 0)
        ev["done"] = bool(done or self.t >= self.horizon)
        return ev

    # ------------------------------------------------------------------ rendering
    def render_simple(self, tile: float = 0.5) -> np.ndarray:
        """Matplotlib drawing that needs no pyglet. Blue = learner (0), green = partners (1 ..),
        numbers are levels."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        e = self.env
        fig, ax = plt.subplots(figsize=(self.W * tile, self.H * tile), dpi=100)
        ax.set_xlim(-0.5, self.W - 0.5); ax.set_ylim(self.H - 0.5, -0.5); ax.axis("off")
        for r in range(self.H):
            for c in range(self.W):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc="#f4f1ea", ec="#ccc", lw=0.4))
                if e.field[r, c]:
                    ax.add_patch(plt.Circle((c, r), 0.3, fc="#d62728", ec="black", lw=0.6))
                    ax.text(c, r, str(e.field[r, c]), ha="center", va="center", fontsize=7, color="white")
        for i, p in enumerate(e.players):
            r, c = p.position
            ax.add_patch(plt.Rectangle((c - 0.35, r - 0.35), 0.7, 0.7, fc="#1f5fbf" if i == 0 else "#2e9e4f",
                                       ec="white", lw=1.2))
            ax.text(c, r, f"{i}:{p.level}", ha="center", va="center", fontsize=6, color="white", weight="bold")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
