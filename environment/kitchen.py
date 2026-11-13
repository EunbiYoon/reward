"""Two-cook Overcooked kitchen (official Overcooked-AI package, Carroll et al. 2019).

Player 0 is the learner. Player 1 is a fixed partner:
    stay         parked on a cell that blocks nothing and never moves (the learner must cook alone)
    greedy       the package's scripted GreedyHumanModel (a competent teammate)
    ckpt:<path>  a trained checkpoint (e.g. a credit thief), frozen, acting from seat 1's view

Game (package defaults): put 3 onions in a pot, INTERACT on the pot to start cooking,
take a dish, scoop the soup and deliver it for +20. Episodes last `horizon` steps.

Training rewards (what the learner is paid; every run is SCORED on team soups):
    pickup      team delivery reward + `pickup_bonus` (3) every time the learner picks up an
                onion or a dish, from anywhere. The bonus never fades.            -> H1
    team        team delivery reward + official shaping that fades out.            -> H2
    individual  the learner's OWN deliveries + official shaping that fades out.    -> H3
    cf          counterfactual contribution (difference reward): team deliveries minus what the
                partner delivers ALONE from the same start (learner parked), + fading shaping. Over
                an episode it sums to v(L,P) - v(P), the learner's marginal contribution.  -> H10
    prep        prep crowding-out: only the onions the LEARNER puts into a pot, 20/3 each (a soup's worth
                per 3 onions), never annealed. The mirror of `individual`: it pays for the first step
                of the recipe instead of the last, so the learner takes over the partner's preparation.
The official shaping (+3 ingredient into a pot, +3 useful dish pickup, +5 soup pickup) only
helps the learner get started; it falls linearly to 0 over `anneal_steps` training steps.

Evaluation can put any controller in either seat (see `play`): a trained policy, the
scripted greedy cook, or a frozen cook that never moves. That is what the Shapley value
needs (eval/coalitions.py).
"""
from __future__ import annotations

import numpy as np

REWARDS = ("pickup", "team", "individual", "cf", "prep")
PARTNERS = ("stay", "greedy")
N_ACTIONS = 6          # N, S, E, W, STAY, INTERACT
STAY = 4               # index of Action.STAY
SOUP = 20.0
TRACE_KEYS = ("soups", "own_soups", "partner_soups", "onion_pickups0", "dish_pickups0",
              "onion_pickups1", "dish_pickups1")
# Extra per-episode metrics of the learner (seat 0), reported by eval/coalitions.py:
#   hold_steps0       steps holding anything
#   hold_dish_steps0  steps holding a dish or a soup (resource hoarding, H11)
#   soup_pickups0     soups scooped from a pot
#   intercepts0       soups scooped from a pot the PARTNER filled with most of the onions (credit theft, H12)
#   pot_onions0       onions the learner put into a pot (prep crowding-out)
EXTRA_METRICS = ("hold_steps0", "hold_dish_steps0", "soup_pickups0", "intercepts0", "pot_onions0")


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
    stay_action = STAY
    action_names = ("N", "S", "E", "W", "STAY", "INTERACT")
    extra_metrics = EXTRA_METRICS

    def __init__(self, layout: str = "cramped_room", reward: str = "team", horizon: int = 400,
                 partner: str = "greedy", anneal_steps: int = 1_000_000, pickup_bonus: float = 3.0,
                 n_agents: int = 2, noise: float = 0.0):
        """noise: probability that a greedy PARTNER (seat 1) takes a uniformly random action instead
        of its scripted one (a less competent teammate, H8). The learner seat is never noised, so the
        scripted reference in the learner's seat stays the same cook."""
        if n_agents != 2:
            raise ValueError("kitchen (Overcooked-AI) supports exactly 2 cooks: the package's lossless "
                             "encoding and GreedyHumanModel assume 2 players. Use --env warehouse or "
                             "--env foraging for 3 or 4 agents.")
        self.n_agents = 2
        if reward not in REWARDS:
            raise ValueError(f"unknown reward {reward!r}; choose from {REWARDS}")
        if partner not in PARTNERS and not partner.startswith("ckpt:"):
            raise ValueError(f"unknown partner {partner!r}; choose from {PARTNERS} or ckpt:<path>")
        Action, OvercookedEnv, OvercookedGridworld = _import_overcooked()
        self.Action = Action
        self.layout, self.reward, self.horizon, self.partner = layout, reward, horizon, partner
        self.anneal_steps = max(1, int(anneal_steps))
        self.pickup_bonus = float(pickup_bonus)
        self.noise = float(noise)
        self._noise_rng = np.random.RandomState(0)
        self._pot_fill: dict = {}       # pot cell -> [onions by seat 0, onions by seat 1]
        self._cf_seed, self._cf_base = None, None
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.env.reset()
        self.obs_dim = int(np.prod(self._encode(self.env.state).shape))
        self.total_steps = 0            # training progress, drives the fading shaping
        self._greedy = {}               # seat -> GreedyHumanModel
        self._ckpt = {}                 # path -> (policy, obs_rms) of a checkpoint partner
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
        """Action object for a fixed seat: 'stay', 'greedy' or 'ckpt:<path>' (a frozen trained policy)."""
        if kind == "stay":
            return self.Action.STAY
        if kind.startswith("ckpt:"):
            from train.deep_policy import act, load_checkpoint
            path = kind[5:]
            if path not in self._ckpt:
                policy, obs_rms, meta = load_checkpoint(path)
                if meta["obs_dim"] != self.obs_dim:
                    raise ValueError(f"{path}: obs_dim {meta['obs_dim']} != {self.obs_dim}")
                self._ckpt[path] = (policy, obs_rms)
            policy, obs_rms = self._ckpt[path]
            return self.Action.ALL_ACTIONS[act(policy, obs_rms, self._encode(state, seat), deterministic=False)]
        a = self.greedy(seat).action(state)
        a = a[0] if isinstance(a, tuple) else a
        if seat >= 1 and self.noise > 0 and self._noise_rng.rand() < self.noise:
            a = self.Action.ALL_ACTIONS[self._noise_rng.randint(N_ACTIONS)]    # a partner's mistake
        return a

    def action_index(self, a) -> int:
        return self.Action.ALL_ACTIONS.index(a)

    def scripted_index(self, kind: str, seat: int) -> int:
        """Action index the scripted controller `kind` would take in `seat` right now (used by the
        mixed behaviours of rlaif/behaviors.py)."""
        return self.action_index(self._scripted_action(kind, seat, self.env.state))

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
        self._noise_rng = np.random.RandomState((int(seed) + 7919) % (2 ** 32))   # partner mistakes
        self._pot_fill = {}
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
        ev["a1"] = self.action_index(a1)                   # partner's action (label of the H12 head)
        ev["hold_steps0"] = int(now[0] is not None)
        ev["hold_dish_steps0"] = int(now[0] in ("dish", "soup"))
        ev["soup_pickups0"] = ev["intercepts0"] = 0
        ev["pot_onions0"] = ev["pot_onions1"] = 0
        for i, p in enumerate(state.players):
            if prev[i] == now[i]:
                continue
            x, y = p.position
            dx, dy = p.orientation
            cell = (x + dx, y + dy)
            grid = self.mdp.terrain_mtx
            if not (0 <= cell[1] < len(grid) and 0 <= cell[0] < len(grid[0])) or grid[cell[1]][cell[0]] != "P":
                continue
            if prev[i] == "onion" and now[i] is None:        # an onion into this pot
                ev[f"pot_onions{i}"] = 1
                self._pot_fill.setdefault(cell, [0, 0])[i] += 1
            elif prev[i] == "dish" and now[i] == "soup":     # a soup scooped from this pot
                fill = self._pot_fill.pop(cell, [0, 0])
                if i == 0:
                    ev["soup_pickups0"] = 1
                    ev["intercepts0"] = int(fill[1] > fill[0])
        ev["done"] = bool(done or self.t >= self.horizon)
        return ev

    def _proxy(self, ev: dict) -> float:
        if self.reward == "prep":
            return SOUP / 3 * ev["pot_onions0"]
        if self.reward == "pickup":
            return ev["sparse"] + self.pickup_bonus * ev["pickups0"]
        if self.reward == "team":
            return ev["sparse"] + self.anneal_coef() * ev["shaped0"]
        if self.reward == "cf":
            base = self._cf_base[self.t - 1] if self.t - 1 < len(self._cf_base) else 0.0
            return ev["sparse"] - base + self.anneal_coef() * ev["shaped0"]
        return ev["own_sparse"] + self.anneal_coef() * ev["shaped0"]     # individual

    def _partner_alone(self, seed: int) -> np.ndarray:
        """Per-step team delivery reward when the learner is parked and only the partner moves,
        from the same start (the v(P) episode of eval/coalitions.py). Cached for the last seed,
        so a GRPO group (G episodes from one seed) pays for it once."""
        if self._cf_seed == seed:
            return self._cf_base
        out = []
        if self.partner != "stay":
            self._reset_core(seed, (0,))
            done = False
            while not done:
                ev = self._joint_step(self.Action.STAY, self._scripted_action(self.partner, 1, self.env.state))
                out.append(ev["sparse"])
                done = ev["done"]
        self._cf_seed, self._cf_base = seed, np.asarray(out, np.float32)
        return self._cf_base

    # ------------------------------------------------------------------ training API
    def reset(self, seed: int) -> np.ndarray:
        if self.reward == "cf":
            self._partner_alone(seed)
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
    def play(self, seat0, seat1: str, seed: int, frames: list | None = None,
             trace: list | None = None) -> dict:
        """One episode. seat0: callable obs -> action index (a trained policy), 'greedy' or 'stay'.
        seat1: 'greedy' or 'stay'. A 'stay' cook is frozen: parked on a cell that blocks no
        pot, dispenser or counter (see parking_cells), and never moves. Returns episode totals. If `frames` is a list, appends
        (rgb image, running totals) after every step, for videos. If `trace` is a list, appends one dict per
        step: the learner's observation `obs`, every seat's action index `a` and the step's non-zero events
        `ev` (what the LLM judge reads and the reward model is trained on, see rlaif/)."""
        frozen = tuple(i for i, s in enumerate((seat0, seat1)) if isinstance(s, str) and s == "stay")
        self._reset_core(seed, frozen)
        tot = {k: 0.0 for k in ("soups", "own_soups", "partner_soups", "pickups0", "pickups1",
                                "onion_pickups0", "dish_pickups0", *EXTRA_METRICS)}
        if frames is not None:
            frames.append((self.render(self.renderer), dict(tot)))
        done = False
        while not done:
            st = self.env.state
            obs0 = self._encode(st)
            if callable(seat0):
                a0 = self.Action.ALL_ACTIONS[int(seat0(obs0))]
            else:
                a0 = self._scripted_action(seat0, 0, st)
            a1 = self._scripted_action(seat1, 1, st)
            ev = self._joint_step(a0, a1)
            if trace is not None:
                trace.append({"obs": obs0, "a": [self.action_index(a0), self.action_index(a1)],
                              "ev": {k: v for k, v in ev.items() if k in TRACE_KEYS and v}})
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
