"""Level-Based Foraging with 2 to 4 players (official lbforaging package, Albrecht & Ramamoorthy 2013,
Papoudakis et al. 2021, https://github.com/semitable/lb-foraging). Same interface as Kitchen.

Player 0 is the learner. Players 1 .. n-1 are fixed partners, all of one kind:
    stay    parked in a corner (never next to food) and never moves
    greedy  the package's own heuristic agent H3 (lbforaging.agents.heuristic_agent.H3, the
            counterpart of Overcooked-AI's GreedyHumanModel): go to the closest food it can lift
            ALONE (food level <= its level) and LOAD next to it; random valid action when there is
            none or the straight path is blocked. It never joins a food it cannot lift alone.
            Loaded straight from the package files (see _import_h3), not reimplemented.

Game (package rules): players and food have levels. Players standing next to a food and
choosing LOAD at the same step lift it if the sum of their levels >= the food's level; each
gets level * food / (sum of their levels * all food spawned), so a whole field is worth 1 and
the package itself splits credit by level. The episode ENDS when all food is gone (a true
terminal, unlike Overcooked and RWARE) or after `horizon` steps.

Training rewards (what the learner is paid; every run is SCORED on the team's food):
    pickup      team food + `pickup_bonus` for every LOAD next to a food, lifted or not.
                Standing at a food too heavy for you and pressing LOAD pays forever.      -> H1
    team        team food + shaping that fades out.                                      -> H2
    theft  the learner's OWN share of food + shaping that fades out.                -> H3
    shadow      team food + `shadow_bonus` for every step the learner stands right next to a partner
                (Manhattan distance 1). Never fades. The proximity reward of Kitchen.
Shaping: +`shaping_bonus` per cell the learner gets closer to the nearest food (potential based,
measured against the food that was there before the step). Falls to 0 over `anneal_steps`.

Result keys keep the Kitchen names ("soups" = share of the field's food) so eval/ works unchanged.

Layouts follow the package task names without the player count: '8x8-3f', '10x10-4f-coop', ...
(-coop: every food needs all players). The player count comes from `n_agents`, not the name.
"""
from __future__ import annotations

import importlib.util
import os
import random
import re
import sys
import types

import numpy as np

from .two_seat import TwoSeat

N_ACTIONS = 6          # NONE, NORTH, SOUTH, WEST, EAST, LOAD
STAY = 0               # index of Action.NONE


def _import_lbf():
    try:
        from lbforaging.foraging.environment import Action, ForagingEnv
    except ImportError as e:
        raise ImportError("Needs the official package:  pip install lbforaging") from e
    return Action, ForagingEnv


def _import_h3():
    """The package's heuristic_agent module, unmodified. lbforaging 2.0.0 cannot import it as is:
    heuristic_agent.py says `from foraging.environment import Action` (missing the `lbforaging.`
    prefix), and lbforaging/agents/__init__.py also imports agents that need plotly, networkx,
    pydot and pandas. So: alias `foraging.environment` to the real module, and load agent.py and
    heuristic_agent.py from the package folder without running agents/__init__.py."""
    if "lbforaging.agents.heuristic_agent" in sys.modules:
        return sys.modules["lbforaging.agents.heuristic_agent"]
    import lbforaging
    import lbforaging.foraging as F
    import lbforaging.foraging.environment as FE
    sys.modules.setdefault("foraging", F)
    sys.modules.setdefault("foraging.environment", FE)
    folder = os.path.join(os.path.dirname(lbforaging.__file__), "agents")
    if "lbforaging.agents" not in sys.modules:                 # empty package: skip its __init__
        pkg = types.ModuleType("lbforaging.agents")
        pkg.__path__ = [folder]
        sys.modules["lbforaging.agents"] = pkg
    for name in ("agent", "heuristic_agent"):
        full = f"lbforaging.agents.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, os.path.join(folder, name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
    return sys.modules["lbforaging.agents.heuristic_agent"]


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
    stay_action = STAY
    action_names = ("NONE", "N", "S", "W", "E", "LOAD")

    def __init__(self, layout: str = "8x8-3f", reward: str = "team", horizon: int = 50,
                 partner: str = "greedy", anneal_steps: int = 1_000_000, pickup_bonus: float = 0.05,
                 shaping_bonus: float = 0.01, noise: float = 0.0, n_agents: int = 2,
                 shadow_bonus: float | None = None):
        self._check(reward, partner, n_agents)
        if reward == "hoard":
            raise ValueError("reward 'hoard' is warehouse only")
        self.n_agents = n_agents
        self.Action, ForagingEnv = _import_lbf()
        size, n_food, coop = parse_layout(layout)
        self.layout, self.reward, self.horizon, self.partner = layout, reward, horizon, partner
        self.anneal_steps = max(1, int(anneal_steps))
        self.pickup_bonus, self.shaping_bonus, self.noise = float(pickup_bonus), float(shaping_bonus), noise
        # Kitchen pays 3 per pickup and 1 per step next to the partner: keep that ratio by default
        self.shadow_bonus = self.pickup_bonus / 3 if shadow_bonus is None else float(shadow_bonus)
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
        self.heuristics = _import_h3() if partner == "greedy" else None
        self.py_rng = random.Random(0)  # H3's own random fallback, seeded per game (see _reset_core)

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
        self.rng = np.random.RandomState(int(seed) % (2 ** 32))   # optional extra noise on H3
        self.py_rng = random.Random(int(seed))                      # H3's random.choice fallback
        if frozen:
            players = self.env.players
            spots = [c for c in self.parking_cells() if all(tuple(p.position) != c for p in players)]
            for seat in frozen:
                players[seat].position = spots.pop(0)
            self.env._gen_valid_moves()
        self.t = 0

    # ------------------------------------------------------------------ scripted partner
    def _greedy_action(self, seat: int):
        """The package's H3 for `seat`, on the package's own observation of that player.
        `noise` (default 0) adds a random action with that probability; 0 keeps H3 unchanged."""
        A = self.Action
        if self.noise and self.rng.rand() < self.noise:
            return A(self.rng.randint(N_ACTIONS))
        H = self.heuristics
        H.random = self.py_rng          # H3 calls random.choice(...): use this game's seeded stream
        player = self.env.players[seat]
        return A(H.H3(player)._step(self.env._make_obs(player)).value)

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
        r0, c0 = e.players[0].position
        ev["near_steps0"] = int(any(abs(p.position[0] - r0) + abs(p.position[1] - c0) == 1 for p in e.players[1:]))
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
