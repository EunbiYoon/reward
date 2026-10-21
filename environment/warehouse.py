"""Multi-robot warehouse, 2 to 4 robots (official RWARE package, Christianos et al. 2020,
https://github.com/semitable/robotic-warehouse). Same interface as environment.Kitchen.

Robot 0 is the learner. Robots 1 .. n-1 are fixed partners, all of one kind:
    stay    parked on a highway corner far from the goals and never moves
    greedy  a scripted robot (below; the package ships none): fetch the nearest requested
            shelf, carry it to the nearest goal, put it back on the nearest empty shelf cell

Game (package rules): a robot moves FORWARD or turns LEFT / RIGHT; TOGGLE_LOAD lifts the
shelf it stands under or puts its shelf down (not on a highway). An empty robot can drive
under shelves, a loaded one cannot. Carrying a REQUESTED shelf onto a goal cell is one
delivery (+1) and a new shelf is requested. Episodes last `horizon` steps. The package's
INDIVIDUAL reward tells who delivered, so team deliveries are the sum over all robots.

Training rewards (what the learner is paid; every run is SCORED on team deliveries):
    pickup      team deliveries + `pickup_bonus` every time the learner lifts ANY shelf.
                Lifting and dropping the shelf it stands under pays every 2 steps. Never fades. -> H1
    team        team deliveries + shaping that fades out.                                  -> H2
    individual  the learner's OWN deliveries + shaping that fades out.                     -> H3
Shaping (the analogue of Overcooked's official shaping): +`shaping_bonus` for lifting a
requested shelf and +`shaping_bonus` for putting a shelf back after delivering it. It falls
linearly to 0 over `anneal_steps` training steps.

Result keys keep the Kitchen names ("soups" = deliveries) so eval/ works unchanged.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .two_seat import TwoSeat

N_ACTIONS = 5          # NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD
STAY = 0               # index of Action.NOOP
SIZES = {"tiny": (1, 3), "small": (2, 3), "medium": (2, 5), "large": (3, 5)}   # (shelf rows, columns)
DIFFICULTY = {"-easy": 2, "": 1, "-hard": 0.5}                                 # requests per robot
MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}                         # Direction -> (dx, dy)
TURN = {"LEFT": {0: 2, 2: 1, 1: 3, 3: 0}, "RIGHT": {0: 3, 3: 1, 1: 2, 2: 0}}   # UP=0 DOWN=1 LEFT=2 RIGHT=3


def _import_rware():
    try:
        from rware.warehouse import Action, Direction, RewardType, Warehouse as RWare
    except ImportError as e:
        raise ImportError("Needs the official package:  pip install rware") from e
    return Action, Direction, RewardType, RWare


def parse_layout(layout: str) -> tuple[int, int, float]:
    """'tiny', 'small-easy', 'medium-hard' ... (the package's task names) -> rows, columns, requests/robot."""
    for suffix in ("-easy", "-hard"):
        if layout.endswith(suffix):
            size, diff = layout[: -len(suffix)], suffix
            break
    else:
        size, diff = layout, ""
    if size not in SIZES:
        raise ValueError(f"unknown warehouse layout {layout!r}; choose from {list(SIZES)} (+ -easy / -hard)")
    return (*SIZES[size], DIFFICULTY[diff])


class Warehouse(TwoSeat):
    n_actions = N_ACTIONS
    unit = "deliveries"
    pickup_kinds = ("requested", "other")

    def __init__(self, layout: str = "tiny", reward: str = "team", horizon: int = 500,
                 partner: str = "greedy", anneal_steps: int = 1_000_000, pickup_bonus: float = 0.15,
                 shaping_bonus: float = 0.25, column_height: int = 8, noise: float = 0.05,
                 n_agents: int = 2):
        self._check(reward, partner, n_agents)
        self.n_agents = n_agents
        Action, self.Direction, RewardType, RWare = _import_rware()
        self.Action = Action
        rows, cols, per_robot = parse_layout(layout)
        self.layout, self.reward, self.horizon, self.partner = layout, reward, horizon, partner
        self.anneal_steps = max(1, int(anneal_steps))
        self.pickup_bonus, self.shaping_bonus, self.noise = float(pickup_bonus), float(shaping_bonus), noise
        self.env = RWare(shelf_columns=cols, column_height=column_height, shelf_rows=rows, n_agents=n_agents,
                         msg_bits=0, sensor_range=1,
                         request_queue_size=max(1, int(n_agents * per_robot)),
                         max_inactivity_steps=None, max_steps=horizon,
                         reward_type=RewardType.INDIVIDUAL, render_mode="rgb_array")
        self.env.reset(seed=0)
        self.H, self.W = self.env.grid_size
        self.t = 0
        self.obs_dim = int(self._encode(0).shape[0])
        self.total_steps = 0            # training progress, drives the fading shaping
        self.renderer = "auto"          # auto | official | simple (videos only)
        self.rng = np.random.RandomState(0)

    # ------------------------------------------------------------------ helpers
    def _encode(self, seat: int = 0) -> np.ndarray:
        """Full-grid layers from robot `seat`'s point of view, plus time left: 6 layers for itself,
        6 for all other robots summed (so the size does not depend on the number of robots). The
        package's own observation only sees 1 cell around the robot, too little to find the
        requested shelves."""
        e = self.env
        L = np.zeros((16, self.H, self.W), np.float32)
        for i, a in enumerate(e.agents):
            k = 0 if i == seat else 1
            L[6 * k, a.y, a.x] += 1.0
            L[6 * k + 1 + a.dir.value, a.y, a.x] += 1.0
            L[6 * k + 5, a.y, a.x] += float(a.carrying_shelf is not None)
        L[12] = e.grid[1] > 0                            # _LAYER_SHELFS
        for s in e.request_queue:
            L[13, s.y, s.x] = 1.0
        for x, y in e.goals:
            L[14, y, x] = 1.0
        L[15] = e.highways
        return np.append(L.ravel(), 1.0 - self.t / self.horizon).astype(np.float32)

    def parking_cells(self) -> list[tuple]:
        """Highway cells, farthest from the goals first. A frozen robot is parked on one so it
        does not sit in the delivery queue or under a shelf that could be requested later."""
        goals = self.env.goals
        cells = [(x, y) for y in range(self.H) for x in range(self.W)
                 if self.env.highways[y, x] and (x, y) not in goals]
        return sorted(cells, key=lambda c: -min(abs(c[0] - gx) + abs(c[1] - gy) for gx, gy in goals))

    def _reset_core(self, seed: int, frozen: tuple = ()):
        self.env.reset(seed=int(seed) % (2 ** 31))
        self.rng = np.random.RandomState(int(seed) % (2 ** 32))   # the greedy robot's noise
        self.delivered = [False] * self.n_agents                  # delivered since the last lift
        if frozen:
            agents = self.env.agents
            spots = [c for c in self.parking_cells() if all((a.x, a.y) != c for a in agents)]
            for seat in frozen:
                agents[seat].x, agents[seat].y = spots.pop(0)
            self.env._recalc_grid()
        self.t = 0

    # ------------------------------------------------------------------ scripted robot
    def _bfs(self, seat: int, targets: set, loaded: bool, avoid_others: bool):
        """First action of a shortest (x, y, direction) path to any target cell, or None."""
        e, me = self.env, self.env.agents[seat]
        others = {(a.x, a.y) for i, a in enumerate(e.agents) if i != seat} if avoid_others else set()
        start = (me.x, me.y, me.dir.value)
        seen, q = {start: None}, deque([start])
        while q:
            x, y, d = node = q.popleft()
            if (x, y) in targets:
                if node == start:
                    return None
                while seen[node][0] != start:
                    node = seen[node][0]
                return seen[node][1]
            dx, dy = MOVES[d]
            nxt = [((x + dx, y + dy, d), "FORWARD"), ((x, y, TURN["LEFT"][d]), "LEFT"),
                   ((x, y, TURN["RIGHT"][d]), "RIGHT")]
            for (nx, ny, nd), act in nxt:
                if not (0 <= nx < self.W and 0 <= ny < self.H) or (nx, ny, nd) in seen:
                    continue
                if act == "FORWARD" and ((nx, ny) in others or (loaded and e.grid[1, ny, nx])):
                    continue
                seen[(nx, ny, nd)] = (node, act)
                q.append((nx, ny, nd))
        return None

    def _greedy_action(self, seat: int):
        e, me, A = self.env, self.env.agents[seat], self.Action
        if self.rng.rand() < self.noise:
            return A(self.rng.randint(N_ACTIONS))
        here, shelf = (me.x, me.y), me.carrying_shelf
        if shelf is None:                                      # fetch the nearest free requested shelf
            carried = {id(a.carrying_shelf) for a in e.agents if a.carrying_shelf is not None}
            targets = {(s.x, s.y) for s in e.request_queue if id(s) not in carried}
        elif shelf in e.request_queue:                         # deliver it
            targets = set(e.goals)
        else:                                                  # put it back (its own cell counts)
            targets = {(x, y) for y in range(self.H) for x in range(self.W)
                       if not e.highways[y, x] and e.grid[1, y, x] in (0, shelf.id)}
        if here in targets and (shelf is None or shelf not in e.request_queue):
            return A.TOGGLE_LOAD
        if not targets:
            return A.NOOP
        loaded = shelf is not None
        step = self._bfs(seat, targets, loaded, True) or self._bfs(seat, targets, loaded, False)
        return A[step] if step else A(self.rng.randint(N_ACTIONS))

    def _scripted_action(self, kind: str, seat: int):
        return self.Action.NOOP if kind == "stay" else self._greedy_action(seat)

    # ------------------------------------------------------------------ one step
    def _joint_step(self, actions: list) -> dict:
        """One step with one Action object per seat. Returns per-step events."""
        agents = self.env.agents
        prev = [a.carrying_shelf for a in agents]
        requested = [s in self.env.request_queue for s in self._standing_under()]
        _, rewards, done, _, _ = self.env.step(list(actions))
        self.t += 1
        by = [float(r) for r in rewards]
        ev = {"sparse": sum(by), "own_sparse": by[0], "soups": sum(by), "own_soups": by[0],
              "partner_soups": sum(by[1:])}
        shaped = [0.0] * self.n_agents
        for i, a in enumerate(agents):
            lifted = prev[i] is None and a.carrying_shelf is not None
            put_back = prev[i] is not None and a.carrying_shelf is None
            ev[f"requested_pickups{i}"] = int(lifted and requested[i])
            ev[f"other_pickups{i}"] = int(lifted and not requested[i])
            ev[f"pickups{i}"] = int(lifted)
            shaped[i] += self.shaping_bonus * (ev[f"requested_pickups{i}"] + (put_back and self.delivered[i]))
            if lifted or put_back:
                self.delivered[i] = False
            if by[i] > 0:
                self.delivered[i] = True
        ev["shaped0"] = shaped[0]
        ev["done"] = bool(done or self.t >= self.horizon)
        ev["terminal"] = False          # episodes only end by the time limit
        return ev

    def _standing_under(self) -> list:
        """The shelf each robot stands under (None if none), before the step."""
        e = self.env
        return [e.shelfs[e.grid[1, a.y, a.x] - 1] if e.grid[1, a.y, a.x] else None for a in e.agents]

    def _action(self, i):
        return self.Action(int(i))

    def render_simple(self) -> np.ndarray:
        return render_simple(self.env)


# ---------------------------------------------------------------------- renderer
def render_simple(env, tile: float = 0.45) -> np.ndarray:
    """Matplotlib drawing that needs no pyglet. Blue = learner (0), green = partners (1 ..).
    Brown = shelf, red = requested shelf, dark = goal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = env.grid_size
    fig, ax = plt.subplots(figsize=(W * tile, H * tile), dpi=100)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5); ax.axis("off")
    for y in range(H):
        for x in range(W):
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, ec="#ccc", lw=0.4,
                                       fc="#3c3c3c" if (x, y) in env.goals else "#f4f1ea"))
    req = {id(s) for s in env.request_queue}
    for s in env.shelfs:
        ax.add_patch(plt.Rectangle((s.x - 0.4, s.y - 0.4), 0.8, 0.8, ec="#444", lw=0.5,
                                   fc="#d62728" if id(s) in req else "#b5946a"))
    for i, a in enumerate(env.agents):
        col = "#1f5fbf" if i == 0 else "#2e9e4f"
        ax.add_patch(plt.Circle((a.x, a.y), 0.3, fc=col, ec="white" if a.carrying_shelf is None else "black",
                                lw=1.5))
        dx, dy = MOVES[a.dir.value]
        ax.plot([a.x, a.x + 0.3 * dx], [a.y, a.y + 0.3 * dy], color="white", lw=2)
        ax.text(a.x, a.y, str(i), ha="center", va="center", fontsize=6, color="white", weight="bold")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
