"""Alem with 2 to 4 players (official alem-env package, a JAX open-ended survival world built on
Craftax with procedurally generated coordination tasks, https://github.com/alem-world/alem-env).
Same interface as Kitchen.

Player 0 is the learner. Players 1 .. n-1 are fixed partners, all of one kind:
    stay    parked on open grass away from the spawn and never moves (NOOP)
    greedy  a scripted survivor (below; the package ships trained RL and LLM agents, not a
            scripted one): drink when thirsty, eat a cow when hungry, sleep when tired, else
            work down its role's tech tree (wood -> table -> pickaxe / sword -> stone -> ...)

Game (package rules): players have roles by seat (0 warrior, 1 forager, 2 miner, 3 warrior).
Only miners craft pickaxes and place stone, only warriors craft stone / iron swords, only
foragers drink water and get food from cows. With a coordination difficulty other than 'none'
the gates are soft (a non-specialist succeeds with probability 1 - alpha), some trees, ores and
mobs need 2 or 3 players to act on the same step, some need a handover. Every achievement
(collect wood, place table, make a pickaxe, ...) pays its value once per player. The episode
ends when every player is dead (a true terminal) or after `horizon` steps.

Training rewards (what the learner is paid; every run is SCORED on the team's achievements):
    pickup      team achievements + `pickup_bonus` for every DO facing a resource (tree, stone,
                ore, water, ripe plant) or a cow, gathered or not. A tree that needs two
                players, hit alone, pays forever.                                    -> H1
    team        team achievements + shaping that fades out.                         -> H2
    theft  the learner's OWN achievements + shaping that fades out.            -> H3
Shaping: +`shaping_bonus` per unit of material (wood, stone, coal, ...) the learner gathers.
Falls to 0 over `anneal_steps`. The package's health-change reward is left out everywhere.

Result keys keep the Kitchen names ("soups" = achievement points) so eval/ works unchanged.

Layouts: '<world>-<difficulty>', world 'overworld' (the package's single-level debug world) or
'full' (all 9 floors; the scripted partner stays on the surface), difficulty 'none', 'easy',
'medium', 'hard' or an alpha in [0, 1] ('overworld-0.5'). Actions are the 18 that matter on the
surface (moves, sleep, rest, place, craft, DO last); requests, gifts and the dungeon actions are
left out.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .two_seat import TwoSeat

# Reduced action set: index -> package Action name. DO is last (the 'use' action, as in every env).
ACTIONS = ("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "SLEEP", "REST", "PLACE_STONE", "PLACE_TABLE",
           "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE",
           "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD", "DO")
A = {name: i for i, name in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)
STAY = A["NOOP"]
MOVES = {A["LEFT"]: (0, -1), A["RIGHT"]: (0, 1), A["UP"]: (-1, 0), A["DOWN"]: (1, 0)}
# package direction (= its move action value) -> (drow, dcol)
FACING = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MATERIALS = ("wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling")
WARRIOR, FORAGER, MINER = 2, 1, 3          # package Specialization values

_CACHE: dict = {}                          # (world, difficulty, n_agents) -> compiled package env


def _dilate(mask: np.ndarray, diagonal: bool) -> np.ndarray:
    """Cells next to a True cell of `mask` (4- or 8-neighbourhood), without wrapping at the edges."""
    out = np.zeros_like(mask)
    H, W = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0) or (not diagonal and dr and dc):
                continue
            out[max(dr, 0):H + min(dr, 0), max(dc, 0):W + min(dc, 0)] |= \
                mask[max(-dr, 0):H + min(-dr, 0), max(-dc, 0):W + min(-dc, 0)]
    return out


def _import_alem():
    try:
        import jax
        import jax.numpy as jnp
        from alem.alem_coop import constants as C
        from alem.alem_coop.alem_state import EnvParams, StaticEnvParams, get_coordination_params
        from alem.alem_env import make_alem_env_from_name
    except ImportError as e:
        raise ImportError("Needs the official package (Python 3.11 or 3.12):  pip install alem-env") from e
    return jax, jnp, C, EnvParams, StaticEnvParams, get_coordination_params, make_alem_env_from_name


def parse_layout(layout: str) -> tuple[str, str]:
    """'overworld-easy' -> ('overworld', 'easy'); 'full-0.3' -> ('full', '0.3')."""
    world, _, diff = layout.partition("-")
    ok = diff in ("none", "easy", "medium", "hard")
    try:
        ok = ok or 0.0 <= float(diff) <= 1.0
    except ValueError:
        pass
    if world not in ("overworld", "full") or not ok:
        raise ValueError(f"unknown alem layout {layout!r}; expected '<overworld|full>-<none|easy|medium|hard"
                         f"|alpha>', e.g. 'overworld-easy'")
    return world, diff


def _compiled(world: str, diff: str, n_agents: int):
    """Package env plus jitted reset / step / summary, shared by every wrapper in the process
    (compiling the package's step takes a while)."""
    key = (world, diff, n_agents)
    if key in _CACHE:
        return _CACHE[key]
    jax, jnp, C, EnvParams, StaticEnvParams, coord_params, make_env = _import_alem()
    params = EnvParams().replace(**coord_params(diff), shared_reward=False, max_timesteps=10 ** 9)
    if world == "overworld":
        static = StaticEnvParams(player_count=n_agents, num_levels=1)
        env = make_env("Alem-Coop-Symbolic-Debug", params, static, compute_full_info=False)
    else:
        static = StaticEnvParams(player_count=n_agents)
        env = make_env("Alem-Coop-Symbolic", params, static, compute_full_info=False)
    ids = jnp.array([C.Action[name].value for name in ACTIONS], jnp.int32)
    reward_map = C.ACHIEVEMENT_REWARD_MAP
    if world == "overworld":
        # The one-floor world grants every player DEFEAT_NECROMANCER (8 points) on the first step:
        # its boss check passes when there is no boss floor. Count only what the surface allows,
        # the same mask the package's own score uses for this world.
        from alem.alem_coop.envs.common import OVERWORLD_ACHIEVEMENT_MASK
        reward_map = reward_map * OVERWORLD_ACHIEVEMENT_MASK

    def summary(state):
        lvl = state.player_level
        inv = state.inventory
        cows = state.passive_mobs
        return dict(
            t=state.timestep, level=lvl, map=state.map[lvl], mob_map=state.mob_map[lvl],
            pos=state.player_position, dir=state.player_direction, alive=state.player_alive,
            sleeping=state.is_sleeping, role=state.player_specialization,
            food=state.player_food, drink=state.player_drink, energy=state.player_energy,
            health=state.player_health, pickaxe=inv.pickaxe, sword=inv.sword,
            mats=jnp.stack([getattr(inv, m) for m in MATERIALS], axis=1),
            points=(state.achievements * reward_map).sum(axis=1),
            cows=cows.position[lvl], cow_mask=cows.mask[lvl])

    @jax.jit
    def reset(seed):
        obs, state = env.reset(jax.random.PRNGKey(seed))
        return state, jnp.stack([obs[a] for a in env.agents]), summary(state)

    @jax.jit
    def step(seed, t, state, actions):
        k = jax.random.fold_in(jax.random.PRNGKey(seed), t)
        obs, state, _, _, _ = env.step_env(k, state, {a: ids[actions[i]] for i, a in enumerate(env.agents)})
        return state, jnp.stack([obs[a] for a in env.agents]), summary(state)

    _CACHE[key] = (env, params, static, reset, step, C)
    return _CACHE[key]


class Alem(TwoSeat):
    n_actions = N_ACTIONS
    unit = "achievement points"
    pickup_kinds = ("gathered", "missed")
    stay_action = STAY
    action_names = ("NOOP", "W", "E", "N", "S", "z_sleep", "e_rest", "r_place_stone", "t_place_table",
                    "f_place_furnace", "p_place_plant", "1_wood_pickaxe", "2_stone_pickaxe",
                    "3_iron_pickaxe", "5_wood_sword", "6_stone_sword", "7_iron_sword", "DO")

    def __init__(self, layout: str = "overworld-easy", reward: str = "team", horizon: int = 400,
                 partner: str = "greedy", anneal_steps: int = 1_000_000, pickup_bonus: float = 0.15,
                 shaping_bonus: float = 0.1, noise: float = 0.05, n_agents: int = 2):
        self._check(reward, partner, n_agents)
        self.n_agents = n_agents
        world, diff = parse_layout(layout)
        self.env, self.params, self.static, self._reset_fn, self._step_fn, C = _compiled(world, diff, n_agents)
        self.layout, self.reward, self.horizon, self.partner = layout, reward, horizon, partner
        self.anneal_steps = max(1, int(anneal_steps))
        self.pickup_bonus, self.shaping_bonus, self.noise = float(pickup_bonus), float(shaping_bonus), noise
        self.soft_roles = bool(self.params.soft_specialization)
        B = C.BlockType
        self.B = B
        blocked = set(C.SOLID_BLOCKS) | {B.WATER.value, B.LAVA.value, B.OUT_OF_BOUNDS.value, B.INVALID.value}
        self._blocked = np.zeros(len(B), bool)
        self._blocked[list(blocked)] = True
        self._resource = np.zeros(len(B), bool)
        self._resource[[B.TREE.value, B.STONE.value, B.COAL.value, B.IRON.value, B.DIAMOND.value,
                        B.SAPPHIRE.value, B.RUBY.value, B.WATER.value, B.RIPE_PLANT.value]] = True
        self._placeable = np.zeros(len(B), bool)
        self._placeable[[B.GRASS.value, B.SAND.value, B.PATH.value]] = True
        self.seed = 0
        self._reset_core(0)
        self.obs_dim = int(self._encode(0).shape[0])
        self.total_steps = 0            # training progress, drives the fading shaping
        self.renderer = "auto"          # auto | official | simple (videos only)

    # ------------------------------------------------------------------ helpers
    def _pull(self, out):
        import jax
        self.state, obs, s = out
        self._obs, self.s = jax.device_get((obs, s))
        self.H, self.W = self.s["map"].shape

    def _encode(self, seat: int = 0) -> np.ndarray:
        """The package's symbolic observation of `seat` (its egocentric 9x11 view, teammate
        dashboard, inventory and intrinsics), plus time left."""
        return np.append(np.asarray(self._obs[seat], np.float32), 1.0 - self.t / self.horizon).astype(np.float32)

    def parking_cells(self) -> list[tuple]:
        """Open grass with only grass around it (nothing to gather, no table to block), nearest to
        the spawn first but at least 4 cells from every player."""
        m, g = self.s["map"], self.B.GRASS.value
        pos = [tuple(p) for p in self.s["pos"]]
        cells = []
        for r in range(1, self.H - 1):
            for c in range(1, self.W - 1):
                if (m[r - 1:r + 2, c - 1:c + 2] == g).all():
                    d = min(abs(r - p[0]) + abs(c - p[1]) for p in pos)
                    if d >= 4:
                        cells.append((d, r, c))
        return [(r, c) for _, r, c in sorted(cells)]

    def _reset_core(self, seed: int, frozen: tuple = ()):
        import jax.numpy as jnp
        self.seed = int(seed) % (2 ** 31)
        self._pull(self._reset_fn(self.seed))
        self.rng = np.random.RandomState(int(seed) % (2 ** 32))   # the scripted players' noise
        if frozen:
            pos = np.array(self.s["pos"])
            spots = iter(self.parking_cells())
            for seat in frozen:
                pos[seat] = next(spots)
            self.state = self.state.replace(player_position=jnp.asarray(pos, jnp.int32))
            self.s["pos"] = pos
            obs = self.env.get_obs(self.state)
            self._obs = np.stack([np.asarray(obs[a]) for a in self.env.agents])
        self.t = 0

    # ------------------------------------------------------------------ scripted survivor
    def _walkable(self, seat: int, avoid_others: bool) -> np.ndarray:
        w = ~self._blocked[self.s["map"]] & ~self.s["mob_map"]
        if avoid_others:
            for i, p in enumerate(self.s["pos"]):
                if i != seat and self.s["alive"][i]:
                    w[p[0], p[1]] = False
        return w

    def _bfs(self, seat: int, goal: np.ndarray, avoid_others: bool):
        """First move of a shortest path to any cell where `goal` is True, or None."""
        free = self._walkable(seat, avoid_others)
        start = tuple(self.s["pos"][seat])
        seen, q = {start: None}, deque([start])
        while q:
            node = q.popleft()
            if goal[node]:
                if node == start:
                    return None
                while seen[node][0] != start:
                    node = seen[node][0]
                return seen[node][1]
            for act, (dr, dc) in MOVES.items():
                nr, nc = node[0] + dr, node[1] + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in seen and free[nr, nc]:
                    seen[(nr, nc)] = (node, act)
                    q.append((nr, nc))
        return None

    def _goto(self, seat: int, goal: np.ndarray):
        step = self._bfs(seat, goal, True) or self._bfs(seat, goal, False)
        return step if step is not None else int(self.rng.choice(list(MOVES)))

    def _act_on(self, seat: int, targets: np.ndarray):
        """Walk next to the nearest True cell of `targets`, face it and DO. None if there is none."""
        if not targets.any():
            return None
        r, c = self.s["pos"][seat]
        for act, (dr, dc) in MOVES.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and targets[nr, nc]:
                return A["DO"] if FACING.get(int(self.s["dir"][seat])) == (dr, dc) else act   # turn first
        return self._goto(seat, _dilate(targets, diagonal=False))

    def _near(self, seat: int, block: int) -> bool:
        r, c = self.s["pos"][seat]
        return bool((self.s["map"][max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] == block).any())

    def _craft_at(self, seat: int, blocks: tuple, action: str):
        """Craft `action` once next to every block in `blocks` (table, furnace), else walk there."""
        missing = [b for b in blocks if not self._near(seat, b)]
        if not missing:
            return A[action]
        targets = self.s["map"] == missing[0]
        if not targets.any():
            return None
        return self._goto(seat, _dilate(targets, diagonal=True) & ~self._blocked[self.s["map"]])

    def _place(self, seat: int, action: str):
        """Place a table / furnace on the faced cell, turning towards an open cell first."""
        r, c = self.s["pos"][seat]
        d = FACING.get(int(self.s["dir"][seat]), (0, 0))
        fr, fc = r + d[0], c + d[1]
        if 0 <= fr < self.H and 0 <= fc < self.W and self._placeable[self.s["map"][fr, fc]] \
                and not self.s["mob_map"][fr, fc] and not any((p == (fr, fc)).all() for p in self.s["pos"]):
            return A[action]
        return int(self.rng.choice(list(MOVES)))

    def _greedy_action(self, seat: int) -> int:
        s, m, B = self.s, self.s["map"], self.B
        if self.rng.rand() < self.noise:
            return int(self.rng.randint(N_ACTIONS))
        if not s["alive"][seat] or s["sleeping"][seat]:
            return STAY
        role = int(s["role"][seat])
        can = lambda r: role == r or self.soft_roles       # hard role gates only without coordination
        inv = dict(zip(MATERIALS, s["mats"][seat]))
        pick, sword = int(s["pickaxe"][seat]), int(s["sword"][seat])
        if s["energy"][seat] <= 2:
            return A["SLEEP"]
        if s["drink"][seat] <= 4 and can(FORAGER) and (a := self._act_on(seat, m == B.WATER.value)) is not None:
            return a
        if s["food"][seat] <= 4 and can(FORAGER):
            cows = np.zeros_like(m, bool)
            for p, ok in zip(s["cows"], s["cow_mask"]):
                if ok:
                    cows[p[0], p[1]] = True
            if (a := self._act_on(seat, cows)) is not None:
                return a
        table = B.CRAFTING_TABLE.value
        has_table = (m == table).any()
        need_wood = (0 if has_table else 2) + (pick < 2 and can(MINER)) + (sword < 1) + (sword < 2 and can(WARRIOR))
        if inv["wood"] < need_wood:
            return self._act_on(seat, m == B.TREE.value) or A["DO"]
        if not has_table:
            return self._place(seat, "PLACE_TABLE")
        plan = []
        if pick < 1 and can(MINER):
            plan.append(((table,), "MAKE_WOOD_PICKAXE"))
        if sword < 1:
            plan.append(((table,), "MAKE_WOOD_SWORD"))
        if pick >= 1 and inv["stone"] < 2:
            if (a := self._act_on(seat, m == B.STONE.value)) is not None:
                return a
        if pick < 2 and can(MINER) and inv["stone"] >= 1:
            plan.append(((table,), "MAKE_STONE_PICKAXE"))
        if sword < 2 and can(WARRIOR) and inv["stone"] >= 1:
            plan.append(((table,), "MAKE_STONE_SWORD"))
        for blocks, action in plan:
            if (a := self._craft_at(seat, blocks, action)) is not None:
                return a
        if pick >= 1 and inv["coal"] < 1 and (a := self._act_on(seat, m == B.COAL.value)) is not None:
            return a
        if pick >= 2 and inv["iron"] < 1 and (a := self._act_on(seat, m == B.IRON.value)) is not None:
            return a
        if inv["stone"] >= 1 and not (m == B.FURNACE.value).any() and self._near(seat, table):
            return self._place(seat, "PLACE_FURNACE")
        if inv["iron"] >= 1 and inv["coal"] >= 1 and inv["stone"] >= 1 and inv["wood"] >= 1:
            action = "MAKE_IRON_PICKAXE" if pick < 3 and can(MINER) else \
                "MAKE_IRON_SWORD" if sword < 3 and can(WARRIOR) else None
            if action and (a := self._craft_at(seat, (table, B.FURNACE.value), action)) is not None:
                return a
        if inv["wood"] < 5:                                                # stock up, then wait
            return self._act_on(seat, m == B.TREE.value) or STAY
        return STAY

    def _scripted_action(self, kind: str, seat: int) -> int:
        return STAY if kind == "stay" else self._greedy_action(seat)

    def _action(self, i) -> int:
        return int(i)

    # ------------------------------------------------------------------ one step
    def _facing_resource(self, seat: int) -> bool:
        r, c = self.s["pos"][seat]
        d = FACING.get(int(self.s["dir"][seat]))
        if d is None:
            return False
        fr, fc = r + d[0], c + d[1]
        if not (0 <= fr < self.H and 0 <= fc < self.W):
            return False
        if self._resource[self.s["map"][fr, fc]]:
            return True
        return any(ok and p[0] == fr and p[1] == fc for p, ok in zip(self.s["cows"], self.s["cow_mask"]))

    def _joint_step(self, actions: list) -> dict:
        """One step with one action index per seat. Returns per-step events."""
        import jax.numpy as jnp
        before = self.s
        dos = [a == A["DO"] and before["alive"][i] and self._facing_resource(i) for i, a in enumerate(actions)]
        self._pull(self._step_fn(self.seed, self.t, self.state, jnp.asarray(actions, jnp.int32)))
        self.t += 1
        s = self.s
        by = [float(x) for x in s["points"] - before["points"]]
        gained = (s["mats"] - before["mats"]).clip(min=0).sum(axis=1) \
            + (s["food"] > before["food"]) + (s["drink"] > before["drink"])
        ev = {"sparse": sum(by), "own_sparse": by[0], "soups": sum(by), "own_soups": by[0],
              "partner_soups": sum(by[1:])}
        for i in range(self.n_agents):
            ev[f"pickups{i}"] = int(dos[i])
            ev[f"gathered_pickups{i}"] = int(dos[i] and gained[i] > 0)
            ev[f"missed_pickups{i}"] = int(dos[i] and gained[i] <= 0)
        ev["shaped0"] = self.shaping_bonus * float((s["mats"][0] - before["mats"][0]).clip(min=0).sum())
        ev["terminal"] = bool(not s["alive"].any())
        ev["done"] = bool(ev["terminal"] or self.t >= self.horizon)
        return ev

    # ------------------------------------------------------------------ rendering
    def render(self, renderer: str = "auto") -> np.ndarray:
        """'official': every player's own pixel view (the package's textures), side by side."""
        if renderer in ("auto", "official"):
            try:
                return self._render_official()
            except Exception as e:
                if renderer == "official":
                    raise
                if not getattr(self, "_warned", False):
                    print(f"  (official renderer unavailable: {type(e).__name__}; using simple renderer)")
                    self._warned = True
        return self.render_simple()

    def _render_official(self, px: int = 16) -> np.ndarray:
        if not hasattr(self, "_textures"):
            from alem.alem_coop.constants import TEXTURES, load_player_specific_textures
            self._textures = load_player_specific_textures(TEXTURES[px], self.n_agents)
        from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels
        views = np.asarray(render_alem_pixels(self.state, px, self.static, self._textures, do_night_noise=False))
        gap = np.full((views.shape[1], 4, 3), 255, np.uint8)
        return np.concatenate([x for v in views for x in (v.astype(np.uint8), gap)][:-1], axis=1)

    def render_simple(self, radius: int = 8, tile: float = 0.3) -> np.ndarray:
        """Matplotlib top-down drawing around the players. Blue = learner (0), green = partners,
        brown = cows, letters are blocks (T tree, S stone, W water, C table, F furnace, ...)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s, B = self.s, self.B
        pos = np.asarray(s["pos"])
        r0, c0 = np.clip(pos.min(0) - radius, 0, None)
        r1, c1 = np.minimum(pos.max(0) + radius + 1, (self.H, self.W))
        colors = {B.GRASS.value: "#cfe8b0", B.WATER.value: "#7fb3e6", B.STONE.value: "#9a9a9a",
                  B.TREE.value: "#2f7d32", B.SAND.value: "#eedc9a", B.PATH.value: "#c9b79c",
                  B.LAVA.value: "#e2572b", B.COAL.value: "#333333", B.IRON.value: "#c58b6a",
                  B.DIAMOND.value: "#7fe3e0", B.CRAFTING_TABLE.value: "#8d5a2b", B.FURNACE.value: "#5a3b2e"}
        letters = {B.TREE.value: "T", B.STONE.value: "S", B.COAL.value: "c", B.IRON.value: "i",
                   B.DIAMOND.value: "d", B.CRAFTING_TABLE.value: "C", B.FURNACE.value: "F"}
        fig, ax = plt.subplots(figsize=((c1 - c0) * tile, (r1 - r0) * tile), dpi=100)
        ax.set_xlim(c0 - 0.5, c1 - 0.5); ax.set_ylim(r1 - 0.5, r0 - 0.5); ax.axis("off")
        for r in range(r0, r1):
            for c in range(c0, c1):
                b = int(s["map"][r, c])
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=colors.get(b, "#dddddd"), ec="none"))
                if b in letters:
                    ax.text(c, r, letters[b], ha="center", va="center", fontsize=5, color="white")
        for p, ok in zip(s["cows"], s["cow_mask"]):
            if ok and r0 <= p[0] < r1 and c0 <= p[1] < c1:
                ax.add_patch(plt.Circle((p[1], p[0]), 0.35, fc="#8b5a2b", ec="white", lw=0.5))
        for i, p in enumerate(pos):
            fc = "#1f5fbf" if i == 0 else "#2e9e4f"
            ax.add_patch(plt.Rectangle((p[1] - 0.4, p[0] - 0.4), 0.8, 0.8, fc=fc if s["alive"][i] else "#777",
                                       ec="white", lw=1.0))
            ax.text(p[1], p[0], str(i), ha="center", va="center", fontsize=6, color="white", weight="bold")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    def close(self):
        pass
