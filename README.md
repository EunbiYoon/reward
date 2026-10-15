# Reward Learning and Reward Hacking in Actor–Critic Agents

Final project for **CS 690S: AI Alignment** (UMass Amherst, Fall 2026, Prof. Scott Niekum).

RLHF pipelines learn a reward model from human data and then let an RL optimizer
push hard against it. When the reward model is misspecified, the optimizer finds
and exploits the gap (*reward hacking*). This repo studies that pipeline in a
fully controlled, exactly-solvable setting:

1. **Model RLHF-style misspecification.** Learn reward models from expert
   demonstrations on *training* maps with MaxEnt IRL, Bayesian IRL and the other
   course baselines, then train a **One-Step Actor–Critic agent from scratch** on
   a *proxy-reward test* map using only the learned reward. Compare against
   behavior cloning across demonstration budgets.
2. **Mitigate it with reward uncertainty.** Penalize the Bayesian-IRL reward by
   its posterior standard deviation, and **actively query preferences** about
   trajectories that pass through the highest-uncertainty states of the
   deployment map. Measure hacking through the **true return gap** and the
   **occupancy-measure divergence** from the true-optimal policy.

## Setting

Linear rewards over one-hot cell features `[empty, goal, lava, coin]`; true weights
`[-0.02, +1, -1, 0]`. The goal pays once and leads to an absorbing sink, so any
other absorbing behaviour is a real loss. 10% action slip, γ = 0.95.

| Training maps (demos collected here) | Proxy test maps (agent deployed here) |
|---|---|
| Coins **always** sit next to the goal, between goal and start → "coin" is a spurious correlate of success | `coin_trap`: a coin right next to the start. Over-valuing coins → the agent camps on it forever (cf. the boat-race specification-gaming example) |
| Lava appears in only 30% of maps → its weight is weakly identified at small budgets | `lava_shortcut`: a lava wall; the true optimum detours, an under-penalized hazard is crossed |
| | `combined`: both |

```
lava_shortcut     coin_trap
........          .......G
....L...          ........
....L...          ........
....L...          ........
S...L..G          ........
....L...          ........
....L...          ........
........          Sc......
```

## Methods

| Name in results | Reward / policy source | Then |
|---|---|---|
| `oracle` | true reward | One-Step AC |
| `bc`, `dagger` | cloned policy | deployed directly |
| `apprenticeship`, `maxent`, `gail`, `trex` | point-estimate reward | One-Step AC |
| `brex`, `birl_mean` | posterior-mean reward | One-Step AC |
| **`birl_varpen`** (ours) | `E[r(s)] − λ·Std[r(s)]` under the Bayesian IRL posterior | One-Step AC |
| **`birl_active`** (ours) | variance penalty after preference queries on high-uncertainty states of the deployment map | One-Step AC |

See [`docs/baselines.md`](docs/baselines.md) for the mapping from every course
reading to its implementation (BC, DAgger, Abbeel & Ng, MaxEnt IRL, Bayesian IRL +
Brown & Niekum α-VaR bounds, GAIL, T-REX, B-REX, Knox et al. human models).

### Metrics (all computed exactly, against the **true** reward)

* `norm_gap` — `(J*_true − J^π_true) / (J*_true − J^unif_true)`
* `tv_to_opt`, `state_tv_to_opt`, `js_to_opt` — divergence between the discounted
  occupancy measure of π and of the true-optimal policy (ties between equally
  optimal routes are not penalized)
* `hacked` — π is at least as good as the true optimum under the *proxy* reward,
  yet loses > 20% of the normalized true return (after Skalse et al., 2022)
* `lava_occ`, `coin_occ` — occupancy mass on the hazard / spurious feature
* `var_evd` — 95% value-at-risk of expected value difference over the reward
  posterior (Brown & Niekum, 2018), for posterior-based methods
* `n_queries` — preference labels used (active method only)

## Quick start

```bash
pip install -e .            # numpy + matplotlib only
python tests/test_core.py   # or: pytest

# ~1.5 min smoke test (1 seed, budgets 2 and 10, two test maps)
python -m rlhack.experiments.run --config configs/quick.json --out results/quick.csv
python -m rlhack.experiments.plot results/quick.csv --out figures/quick

# full sweep: 5 seeds × budgets {1,2,5,10,20} × 3 test maps × 11 methods (CPU, ~1 h)
python -m rlhack.experiments.run --config configs/full.json --out results/full.csv
python -m rlhack.experiments.plot results/full.csv --out figures/full

# ablation: synthetic human follows the regret model, learner assumes partial return
python -m rlhack.experiments.run --config configs/ablation_regret_human.json --out results/regret.csv
```

Any key of `DEFAULTS` in `src/rlhack/experiments/run.py` can be overridden in a
config JSON (e.g. `"lambda": 2.0`, `"active": {"rounds": 3}`, `"human": {"model": "regret"}`).

## Layout

```
src/rlhack/
  envs/          gridworld.py (features, dynamics), proxy.py (train maps + proxy test maps)
  mdp.py         (soft) value iteration, policy evaluation, occupancy measures
  demos.py       expert demos, noise-injected ranked rollouts
  preferences.py synthetic human (partial-return / regret), Bradley–Terry likelihood
  irl/           maxent, apprenticeship, gail (state-only), bayesian (MCMC), rex (T-REX, B-REX)
  imitation/     bc (feature policy), dagger
  agents/        One-Step Actor–Critic (Sutton & Barto §13.5)
  method/        variance-penalized reward + active preference queries (ours)
  metrics.py     true-return gap, occupancy divergence, hacking flag, α-VaR
  experiments/   run.py (sweep → CSV), plot.py (figures + summary table)
configs/  tests/  docs/
```

## Implementation notes

* Everything is tabular and numpy-only, so every number is exact except the actor–critic's
  sampled training — results are cheap to reproduce on a laptop.
* The actor–critic uses exploring starts during training (evaluation always from the
  fixed start) and omits the `γ^t` factor `I` by default (`use_I=True` restores the
  textbook update); without these the *oracle* gets stuck in local optima.
* Expert demonstrations are Boltzmann-rational (β = 100); Bayesian IRL assumes β = 20
  — a deliberate, mild rationality misspecification.
* Bayesian IRL value iteration is batched across training maps and warm-started from the
  current chain state; ~1–3 s per posterior on CPU.

## Possible extensions

* λ sweep / adaptive λ chosen by the α-VaR bound; DPO/CPL-style direct baselines;
  occupancy-measure regularization (Laidlaw et al.) as an alternative mitigation.
* Swap the gridworld for a continuous-control proxy environment and the tabular
  posterior for an ensemble / Laplace reward model.
