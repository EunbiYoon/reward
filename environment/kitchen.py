"""Two-cook Overcooked kitchen (official Overcooked-AI package, Carroll et al. 2019).

Player 0 is the learner. Player 1 is a fixed partner:
    stay    parked on a cell that blocks nothing and never moves (the learner must cook alone)
    greedy  the package's scripted GreedyHumanModel (a competent teammate)

Game (package defaults): put 3 onions in a pot, INTERACT on the pot to start cooking,
take a dish, scoop the soup and deliver it for +20. Episodes last `horizon` steps.

Training rewards (what the learner is paid; every run is SCORED on team soups):
    pickup      team delivery reward + `pickup_bonus` (3) every time the learner picks up an
                onion or a dish, from anywhere. The bonus never fades.            -> H1
    team        team delivery reward + official shaping that fades out.            -> H2
    individual  the learner's OWN deliveries + official shaping that fades out.    -> H3
The official shaping (+3 ingredient into a pot, +3 useful dish pickup, +5 soup pickup) only
helps the learner get started; it falls linearly to 0 over `anneal_steps` training steps.

Evaluation can put any controller in either seat (see `play`): a trained policy, the
scripted greedy cook, or a frozen cook that never moves. That is what the Shapley value
needs (eval/coalitions.py).
"""
from __future__ import annotations

import numpy as np

REWARDS = ("pickup", "team", "individual")
PARTNERS = ("stay", "greedy")
N_ACTIONS = 6          # N, S, E, W, STAY, INTERACT
STAY = 4               # index of Action.STAY
SOUP = 20.0


def _import_overcooked():
    # overcooked-ai 1.1.0 still uses names NumPy 2 removed (np.Inf in the planners, np.int in the
    # random layout generator). Restore them instead of pinning numpy<2, which breaks scipy >= 1.15.
    for name, val in (("Inf", np.inf), ("int", int)):
        if not hasattr(np, name):
            setattr(np, name, val)
    try:
        from overcooked_ai_py.mdp.actions import Action
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    except ImportError as e:
        raise ImportError("Needs the official package:  pip install -r requirements.txt") from e
    return Action, OvercookedEnv, OvercookedGridworld


def held_name(player) -> str | None:
    obj = getattr(player, "held_object", None)
    return None if obj is None else getattr(obj, "name", str(obj))


class Kitchen:
    discrete = True
    n_actions = N_ACTIONS
    unit = "soups"
    pickup_kinds = ("onion", "dish")

    def __init__(self, layout: str = "cramped_room", reward: str = "team", horizon: int = 400,
                 partner: str = "greedy", anneal_steps: int = 1_000_000, pickup_bonus: float = 3.0,
                 n_agents: int = 2):
        if n_agents != 2:
            raise ValueError("kitchen (Overcooked-AI) supports exactly 2 cooks: the package's lossless "
                             "encoding and GreedyHumanModel assume 2 players. Use --env warehouse or "
                             "--env foraging for 3 or 4 agents.")
        self.n_agents = 2
        if reward not in REWARDS:
            raise ValueError(f"unknown reward {reward!r}; choose from {REWARDS}")
        if partner not in PARTNERS:
            raise ValueError(f"unknown partner {partner!r}; choose from {PARTNERS}")
        Action, OvercookedEnv, OvercookedGridworld = _import_overcooked()
        self.Action = Action
        self.layout, self.reward, self.horizon, self.partner = layout, reward, horizon, partner
        self.anneal_steps = max(1, int(anneal_steps))
        self.pickup_bonus = float(pickup_bonus)
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.env.reset()
        self.obs_dim = int(np.prod(self._encode(self.env.state).shape))
        self.total_steps = 0            # training progress, drives the fading shaping
        self._greedy = {}               # seat -> GreedyHumanModel
        self.renderer = "auto"          # auto | official | simple (videos only)
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def _encode(self, state, seat: int = 0) -> np.ndarray:
        try:
            enc = self.mdp.lossless_state_encoding(state, self.horizon)
        except TypeError:
            enc = self.mdp.lossless_state_encoding(state)
        return np.asarray(enc[seat], dtype=np.float32).ravel()

    def greedy(self, seat: int):
        if seat not in self._greedy:
            from overcooked_ai_py.agents.agent import GreedyHumanModel
            agent = GreedyHumanModel(self.env.mlam)
            agent.set_agent_index(seat)
            agent.set_mdp(self.mdp)
            self._greedy[seat] = agent
        return self._greedy[seat]

    def _scripted_action(self, kind: str, seat: int, state):
        """Action object for a scripted seat: 'stay' or 'greedy'."""
        if kind == "stay":
            return self.Action.STAY
        a = self.greedy(seat).action(state)
        return a[0] if isinstance(a, tuple) else a

    def anneal_coef(self) -> float:
        return max(0.0, 1.0 - self.total_steps / self.anneal_steps)

    def parking_cells(self) -> list[tuple]:
        """Floor cells next to no pot, dispenser or serving counter. A frozen cook is parked
        on one of these so it does not block an interaction tile. In cramped_room the start
        cell of cook 0 is the ONLY cell next to the dish dispenser, so a cook frozen there
        would make every dish impossible and break the Shapley evaluation."""
        grid = self.mdp.terrain_mtx
        H, W = len(grid), len(grid[0])
        cells = []
        for y in range(H):
            for x in range(W):
                if grid[y][x] != " ":
                    continue
                near = [grid[y + dy][x + dx] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= x + dx < W and 0 <= y + dy < H]
                if not any(c in "PODST" for c in near):
                    cells.append((x, y))
        return cells

    def _reset_core(self, seed: int, frozen: tuple = ()):
        np.random.seed(int(seed) % (2 ** 32))         # GreedyHumanModel uses the global RNG
        self.env.reset()
        for seat, agent in self._greedy.items():
            agent.reset()                   # the package's reset() also clears agent_index and mdp
            agent.set_agent_index(seat)
            agent.set_mdp(self.mdp)
        if frozen:
            players = self.env.state.players
            spots = [c for c in self.parking_cells()
                     if all(tuple(p.position) != c for p in players)]
            for seat in frozen:
                if spots:
                    players[seat].position = spots.pop(0)
        self.t = 0

    def _joint_step(self, a0, a1) -> dict:
        """One step with Action objects for both seats. Returns per-step events."""
        prev = [held_name(p) for p in self.env.state.players]
        state, sparse, done, info = self.env.step((a0, a1))
        self.t += 1
        now = [held_name(p) for p in state.players]
        sparse = float(sparse)
        by = info.get("sparse_r_by_agent")
        by = [sparse, 0.0] if by is None else [float(x) for x in by]
        shaped = info.get("shaped_r_by_agent")
        shaped = [float(info.get("shaped_r", 0.0)), 0.0] if shaped is None else [float(x) for x in shaped]
        ev = {"sparse": sparse, "own_sparse": float(by[0]), "shaped0": float(shaped[0]),
              "soups": sparse / SOUP, "own_soups": float(by[0]) / SOUP,
              "partner_soups": float(by[1]) / SOUP}
        for i in (0, 1):
            ev[f"onion_pickups{i}"] = int(prev[i] is None and now[i] == "onion")
            ev[f"dish_pickups{i}"] = int(prev[i] is None and now[i] == "dish")
            ev[f"pickups{i}"] = ev[f"onion_pickups{i}"] + ev[f"dish_pickups{i}"]
        ev["done"] = bool(done or self.t >= self.horizon)
        return ev

    def _proxy(self, ev: dict) -> float:
        if self.reward == "pickup":
            return ev["sparse"] + self.pickup_bonus * ev["pickups0"]
        if self.reward == "team":
            return ev["sparse"] + self.anneal_coef() * ev["shaped0"]
        return ev["own_sparse"] + self.anneal_coef() * ev["shaped0"]     # individual

    # ------------------------------------------------------------------ training API
    def reset(self, seed: int) -> np.ndarray:
        self._reset_core(seed, (1,) if self.partner == "stay" else ())
        return self._encode(self.env.state)

    def step(self, action):
        a0 = self.Action.ALL_ACTIONS[int(action)]
        a1 = self._scripted_action(self.partner, 1, self.env.state)
        ev = self._joint_step(a0, a1)
        self.total_steps += 1
        ev["proxy"] = self._proxy(ev)
        ev["true"] = ev["sparse"]
        return self._encode(self.env.state), ev, ev["done"]

    # ------------------------------------------------------------------ evaluation API
    def play(self, seat0, seat1: str, seed: int, frames: list | None = None) -> dict:
        """One episode. seat0: callable obs -> action index (a trained policy), 'greedy' or 'stay'.
        seat1: 'greedy' or 'stay'. A 'stay' cook is frozen: parked on a cell that blocks no
        pot, dispenser or counter (see parking_cells), and never moves. Returns episode totals. If `frames` is a list, appends
        (rgb image, running totals) after every step, for videos."""
        frozen = tuple(i for i, s in enumerate((seat0, seat1)) if isinstance(s, str) and s == "stay")
        self._reset_core(seed, frozen)
        tot = {k: 0.0 for k in ("soups", "own_soups", "partner_soups", "pickups0", "pickups1",
                                "onion_pickups0", "dish_pickups0")}
        if frames is not None:
            frames.append((self.render(self.renderer), dict(tot)))
        done = False
        while not done:
            st = self.env.state
            if callable(seat0):
                a0 = self.Action.ALL_ACTIONS[int(seat0(self._encode(st)))]
            else:
                a0 = self._scripted_action(seat0, 0, st)
            a1 = self._scripted_action(seat1, 1, st)
            ev = self._joint_step(a0, a1)
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
                return render_official(self.env.state, self.mdp)
            except Exception as e:
                if renderer == "official":
                    raise
                if not getattr(self, "_warned", False):
                    print(f"  (official renderer unavailable: {type(e).__name__}; using simple renderer)")
                    self._warned = True
        return render_simple(self.env.state, self.mdp)

    def close(self):
        pass


# ---------------------------------------------------------------------- renderers
def render_official(state, mdp) -> np.ndarray:
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    surf = StateVisualizer(is_rendering_hud=False).render_state(state, grid=mdp.terrain_mtx)
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).astype(np.uint8)


TERRAIN_COLORS = {"X": "#b5946a", "P": "#555555", "O": "#f2d06b", "T": "#e0533a",
                  "D": "#dddddd", "S": "#7fbf7f", " ": "#f4f1ea"}
TERRAIN_LABELS = {"P": "POT", "O": "ONION", "T": "TOMATO", "D": "DISH", "S": "SERVE"}
OBJ_COLORS = {"onion": "#e6b800", "tomato": "#d62728", "dish": "#ffffff", "soup": "#c46a2b"}
DIRS = {(0, -1): (0, -0.32), (0, 1): (0, 0.32), (1, 0): (0.32, 0), (-1, 0): (-0.32, 0)}


def render_simple(state, mdp, tile: float = 0.6) -> np.ndarray:
    """Matplotlib drawing that needs no pygame. Blue = learner (0), green = partner (1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = mdp.terrain_mtx
    H, W = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(W * tile, H * tile), dpi=100)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5); ax.axis("off")
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fc=TERRAIN_COLORS.get(ch, "#cccccc"),
                                       ec="#444", lw=0.6))
            if ch in TERRAIN_LABELS:
                ax.text(x, y + 0.36, TERRAIN_LABELS[ch], ha="center", va="center", fontsize=5,
                        color="white" if ch == "P" else "#333")
    for pos, obj in getattr(state, "objects", {}).items():
        x, y = pos
        name = getattr(obj, "name", "?")
        if name == "soup":
            ings = list(getattr(obj, "ingredients", []))
            ready = bool(getattr(obj, "is_ready", False))
            cooking = bool(getattr(obj, "is_cooking", False))
            ax.add_patch(plt.Circle((x, y), 0.3, fc=OBJ_COLORS["soup"] if (ready or cooking) else "#888",
                                    ec="black", lw=0.8))
            label = "READY" if ready else (f"{getattr(obj, 'cook_time_remaining', '')}" if cooking
                                           else f"{len(ings)}/3")
            ax.text(x, y, label, ha="center", va="center", fontsize=6, color="white")
        else:
            ax.add_patch(plt.Circle((x, y), 0.2, fc=OBJ_COLORS.get(name, "#999"), ec="black", lw=0.6))
    for i, p in enumerate(state.players):
        x, y = p.position
        col = ("#1f5fbf", "#2e9e4f")[i % 2]
        ax.add_patch(plt.Circle((x, y), 0.33, fc=col, ec="white", lw=1.5))
        dx, dy = DIRS.get(tuple(p.orientation), (0, 0))
        ax.plot([x, x + dx], [y, y + dy], color="white", lw=2)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white", weight="bold")
        h = held_name(p)
        if h:
            ax.add_patch(plt.Circle((x + 0.25, y - 0.25), 0.14, fc=OBJ_COLORS.get(h, "#999"),
                                    ec="black", lw=0.6))
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
