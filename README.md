# Who Gets the Credit? Reward Design and Bad Teammates in Cooperative RL

Project page: https://eunbiyoon.github.io/reward/

This code implements the CS 690S final project proposal. In Overcooked-AI's `cramped_room`, cook 0 (the learner)
is trained with PPO and GRPO while cook 1 is a fixed partner. Each of the three training rewards is expected to
produce a different kind of bad teammate, and **Shapley credit** decides which one it actually produces.

| Training reward (`--rewards`) | Definition | Hypothesis |
|---|---|---|
| `pickup` | Team delivery reward + 3 every time the learner picks up an onion or a dish (never annealed) | H1 fake help |
| `team` | Team delivery reward + the official shaping (annealed to 0 over the first half of training) | H2 spectator |
| `individual` | **The learner's own** delivery reward + the official shaping (annealed) | H3 narcissist |

| Partner (`--partners`) | Meaning |
|---|---|
| `stay` | Never moves. The learner has to cook alone (sanity check) |
| `greedy` | The package's scripted cook GreedyHumanModel (a competent teammate) |

Every run is scored by the **number of soups the team delivered**, whatever the training reward was.

## Install and run

```bash
pip install -r requirements.txt

# 1) Check the install (no training): a clip of two scripted cooks
python -m eval.visualize --preview --partner greedy

# 2) Smoke test: 1 seed, 200k steps
python -m eval.run --quick --workers 4 --out results/quick
python -m eval.summarize results/quick

# 3) The full experiment from the proposal: 3 seeds x 3 rewards x 2 partners x 2 optimizers = 36 runs, 2M steps each
python -m eval.run --workers 4 --out results/full
python -m eval.summarize results/full

# A subset
python -m eval.run --algos ppo --rewards team individual --partners greedy --seeds 0 1 2 --out results/h2h3
```

`results.csv` is updated after every finished run, so `summarize` can be run while an experiment is still going.

Each run uses one CPU core (`torch.set_num_threads(1)`), and the time goes into the Python environment simulation,
not the small MLP. A GPU does not help; more `--workers` (up to the number of cores) does.

## Five algorithms x 2, 3, 4 agents

`--algos all` runs all five learning algorithms (`ppo grpo a2c dqn sac`), and `--agents` sets the number of agents.
Three or more agents work only in `warehouse` and `foraging` (`kitchen` is two players only).

```bash
# 1) Smoke test: 1 seed, 200k steps (3 rewards x 2 partners x 5 algos x 3 agent counts = 90 runs)
python -m eval.run --env foraging --agents 2 3 4 --algos all --quick --workers 4 --out results/lbf_n_quick
python -m eval.summarize results/lbf_n_quick

# 2) Full experiment: 3 seeds, 2M steps each (270 runs)
python -m eval.run --env foraging --agents 2 3 4 --algos all --workers 4 --out results/lbf_n
python -m eval.summarize results/lbf_n

# The same experiment in warehouse
python -m eval.run --env warehouse --agents 2 3 4 --algos all --workers 4 --out results/rware_n
python -m eval.summarize results/rware_n

# 8x8 is crowded with 4 agents, a larger map is recommended
python -m eval.run --env foraging --layout 10x10-4f --agents 4 --algos all --workers 4 --out results/lbf_n4

# A subset: pick algorithms, agent counts and rewards
python -m eval.run --env foraging --agents 3 --algos ppo sac --rewards team individual --out results/lbf_n3

# All five algorithms in kitchen (Overcooked, 2 agents)
python -m eval.run --algos all --workers 4 --out results/all_algos

# Clips: use the same --agents as in training
python -m eval.visualize --run results/lbf_n --algo sac --reward team --partner greedy --agents 3
```

- Agent 0 is the learner; agents 1..N−1 are all the same kind of partner, chosen with `--partners`.
- Shapley is computed with the N-player formula (every subset of partners), so N=4 runs 15 coalitions per evaluation episode.
- The observation is "own layer + one layer summing everyone else", so its size does not depend on N (identical to before for N=2).
- `summarize` reports each agent count separately and compares it with the reference for the same N. For 3 or more agents, `partner soups` is the sum over all partners.
- The greedy-with-greedy reference s\* is about 1/N (warehouse `tiny`: 0.50 / 0.34 / 0.26).
- Checkpoint names are unchanged for 2 agents; for 3 and 4 agents they get `_n3` / `_n4`, e.g. `ppo_team_greedy_n3_s0.pt`.
- Number of runs = seeds × rewards × partners × algos × agents. These take a long time, so always start with `--quick`.

## Clips

```bash
python -m eval.visualize --run results/full --algo ppo --reward individual --partner greedy --seed 0
python -m eval.visualize --run results/full --algo ppo --partner greedy --seed 0 --compare   # the three rewards side by side
python -m eval.visualize --run results/full --algo ppo --reward team --freeze partner        # a Shapley evaluation coalition
```

Blue agent 0 is the learner and green agent 1 is the partner. The header shows team soups, soups delivered by each
agent, and the learner's pickup count. Without pygame, use `--renderer simple` (matplotlib drawing). If the file name
ends in `.mp4`, an MP4 is written (needs ffmpeg).

The project page (`page/`) uses these clips:

```bash
python -m eval.visualize --run results/full       --algo ppo --partner greedy --compare --out page/videos/overcooked_compare_ppo_greedy.gif
python -m eval.visualize --run results/lbf_full   --algo ppo --partner greedy --compare --out page/videos/lbf_compare_ppo_greedy.gif
python -m eval.visualize --run results/rware_full --algo ppo --partner greedy --compare --out page/videos/rware_compare_ppo_greedy.gif
```

## Measurement

**Shapley credit.** The same evaluation episode (seed 10000+k) is played three times.

| Symbol | Meaning |
|---|---|
| v(L,P) | Both move (= team soups, the real objective) |
| v(P) | The learner is frozen |
| v(L) | The partner is frozen |

φ = ½[(v(L,P) − v(P)) + v(L)],  **s = φ / v(L,P)** (the learner's Shapley share),
**credit gap = share the learner delivered itself − s**.

**Reference s\*.** The scripted cook is placed in the learner's seat and measured the same way (`reference.csv`).

**Decision rules (proposal, section 5).** `summarize` judges each optimizer automatically.

| | Condition (greedy partner) |
|---|---|
| sanity | Under the team reward with the stay partner, the learner delivers soups. If this fails, H1 to H3 are all inconclusive |
| H1 | Under the pickup reward, at least half of the runs have pickups > reference and s ≤ 0.1 |
| H2 | Under the team reward, s < s\*/2, and s rises when switching to individual (paired by seed) |
| H3 | Under the individual reward, credit gap ≥ 0.2, and partner soups drop compared with the team reward (paired) |

Each verdict is one of "supported (CIs clear)", "supported (means only)" or "not supported".
With 3 seeds the confidence intervals are wide, so most verdicts are expected to be "means only".

## Other environments: RWARE, LBF (`--env`)

The same three rewards, the same two partners and the same Shapley evaluation run in two other cooperative environments.
The result column is still called `soups`, but its meaning depends on the environment.

| `--env` | Package | Default `--layout` | Meaning of `soups` | Horizon |
|---|---|---|---|---|
| `kitchen` (default) | overcooked-ai | `cramped_room` | Soups delivered | 400 |
| `warehouse` | [rware](https://github.com/semitable/robotic-warehouse) | `tiny` (`small`, `medium`, `large`, `-easy`/`-hard`) | Requested shelves delivered | 500 |
| `foraging` | [lbforaging](https://github.com/semitable/lb-foraging) | `8x8-3f` (`10x10-4f`, `8x8-2f-coop`, ...) | Share of food eaten (whole field = 1) | 50 |

```bash
pip install rware lbforaging
python -m eval.visualize --preview --env warehouse --partner greedy     # check the install
python -m eval.run --env warehouse --quick --workers 4 --out results/rware_quick
python -m eval.run --env foraging  --quick --workers 4 --out results/lbf_quick
python -m eval.summarize results/lbf_quick
```

| Reward | warehouse | foraging |
|---|---|---|
| `pickup` (H1) | Team deliveries + 0.15 every time it picks up **any** shelf (picking up and putting down in place pays every 2 steps) | Team food + 0.05 every LOAD next to food (success or not) |
| `team` (H2) | Team deliveries + shaping (annealed) | Team food + shaping (annealed) |
| `individual` (H3) | The learner's own deliveries + shaping (annealed) | The learner's share of food (the package splits it by level) + shaping (annealed) |
| shaping | Pick up a requested shelf +0.25, return a delivered shelf to an empty slot +0.25 | +0.01 for every step closer to the nearest food |

- **The greedy partner is a hand-written script.** RWARE has no scripted robot, and the H1 to H4 heuristics in LBF 2.0.0
  fail to import because of `from foraging.environment import Action`. Both scripts find paths with BFS.
  warehouse: nearest requested shelf → nearest goal → nearest empty shelf slot to put it down.
  foraging: the nearest food it can lift alone, otherwise the food with the smallest summed distance to both agents.
- **The observation is not the package default.** Both flatten the whole grid into layers (the same idea as Overcooked's lossless encoding).
  RWARE's default observation shows only one cell around the robot, so it cannot find the requested shelves.
- **Where frozen robots/players are parked:** warehouse uses the highway cell farthest from the goals (a corner); foraging uses a corner cell not adjacent to food.
- **LBF episodes really end when all food is eaten.** PPO does not bootstrap in that case (`info["terminal"]`).
- **Reference (greedy + greedy, 10 to 50 evaluation episodes):** both environments have symmetric seats, so s\* ≈ 0.50 and credit gap ≈ 0.0.
  Unlike cramped_room (gap 0.13), the proposal's H3 threshold (gap ≥ 0.2) can be used as is.

  | | Team score | Alone (v(L)) | s\* | gap |
  |---|---|---|---|---|
  | warehouse `tiny` | 28.3 deliveries | 15.5 | 0.50 | −0.01 |
  | foraging `8x8-3f` | 0.98 | 0.41 | 0.49 | 0.02 |
  | foraging `8x8-2f-coop` | 0.97 | 0.00 | 0.50 | 0.01 |

## Facts checked against the official package (read this)

The official Overcooked-AI source was run with the scripted cooks (10 evaluation episodes).

1. **Leaving a frozen cook where it stands breaks Shapley.** In cramped_room, cook 0's start cell is the **only** cell next
   to the dish dispenser. If it is frozen there, the partner can never pick up a dish and v(P) is always 0.
   So a frozen cook is moved to a cell that touches no pot, dispenser or serving counter (the middle cell (2,2)).
   The `stay` partner sits on the same cell during training.
2. **One scripted cook alone makes 7.0 soups, two make 9.1.** There is only one pot, so the second cook can add only
   about 2 soups per episode. The difference between a good teammate and a spectator lives inside those 2 soups, so effects may be small.
3. **Even the good scripted cook has a credit gap of 0.13.** Seat 0 is closer to the dishes and the serving counter, so it delivers more.
   The proposal's H3 threshold (gap ≥ 0.2) is almost passed by a good teammate, so `summarize` shows the proposal's rule together with
   a rule **relative to the reference** (gap ≥ reference gap + 0.2). The latter is recommended for the final report.
4. The package's `GreedyHumanModel.reset()` clears the agent index and the mdp. They are set again every episode.

## Folder layout

```
environment/warehouse.py RWARE wrapper + scripted robot (--env warehouse)
environment/foraging.py  LBF wrapper + scripted forager (--env foraging)
environment/two_seat.py  Training/evaluation loop shared by the two wrappers above
environment/kitchen.py   Official package wrapper. Three rewards, partners, frozen-cook parking, play() for any pairing, renderer
train/deep_policy.py     MLP policy (Categorical), critic, observation normalization, checkpoints
train/ppo.py             PPO (clip, GAE)
train/grpo.py            Critic-free GRPO (groups of 8 episodes from the same reset, process advantage)
train/a2c.py             A2C (same actor/critic/GAE as PPO, one unclipped update per rollout)
train/dqn.py             DQN (Double DQN target, replay buffer, epsilon-greedy, target network)
train/sac.py             SAC for discrete actions (twin Q, automatic entropy temperature alpha)
eval/config.py           Experiment settings (proposal values), --quick settings, decision thresholds
eval/coalitions.py       Frozen-agent evaluation and Shapley computation
eval/run.py              Train → checkpoint → Shapley evaluation → results.csv, reference.csv
eval/summarize.py        Tables and H1 to H3 verdicts → summary.md
eval/visualize.py        GIF/MP4
page/                    Project page (GitHub Pages, deployed by .github/workflows/pages.yml)
```

## Status

- The full experiment (36 runs, 2M steps each, PPO and GRPO) has been run in all three environments:
  `results/full` (kitchen), `results/lbf_full` (foraging), `results/rware_full` (warehouse). Each folder has `summary.md`.
- Checkpoints (`results/*/models/`) are not committed.
- In warehouse the learner barely learns within 2M steps (PPO sanity passes with 0.03 deliveries, GRPO fails),
  so its H1 and H2 verdicts likely reflect an untrained learner. Increase `--env-steps` before reading them.
- In kitchen, PPO learns to do all the deliveries itself under both the team and the individual reward, so the partner
  rarely delivers during training, the two rewards become nearly the same, and the two runs end up almost identical.
