# Credit audit: results/rlaif/need_curve_ppo

100 pool trajectories. A pair is decisive when |phi_A - phi_B| > 0.240. Busy credit: over decisive pairs where the busier-looking agent 0 contributed less, how often it was preferred (0 = never fooled, 0.5 = chance, above 0.5 = credits busy-looking agents). Theft credit: the same for the agent 0 that delivered more soups itself but contributed less (credit theft).

## 1. Judges

| judge | pairs | agreement with Shapley | busy credit | theft credit | busy bias at equal phi (H1) | says A | credit MAE | credit r | over-credit |
|---|---|---|---|---|---|---|---|---|---|
| local | 100 | 0.79 (n=100) | 0.24 [0.14, 0.36] (n=55) | 0.92 [0.67, 0.99] (n=13) | n/a n/a (n=0, p=nan) | 0.56 | 0.44 | 0.30 | +0.35 |
| local_definition | 80 | 0.75 (n=80) | 0.00 [0.00, 0.09] (n=39) | 0.95 [0.76, 0.99] (n=20) | n/a n/a (n=0, p=nan) | 0.41 | 0.38 | 0.28 | +0.30 |
| local_solo | 80 | 0.82 (n=80) | 0.00 [0.00, 0.09] (n=39) | 0.70 [0.48, 0.85] (n=20) | n/a n/a (n=0, p=nan) | 0.49 | 0.31 | 0.44 | +0.25 |
| ollama | 105 | 0.79 (n=105) | 0.24 [0.15, 0.37] (n=58) | 0.92 [0.67, 0.99] (n=13) | n/a n/a (n=0, p=nan) | 0.52 | 0.45 | 0.26 | +0.36 |
| shapley | 200 | 1.00 (n=199) | 0.00 [0.00, 0.03] (n=113) | 0.00 [0.00, 0.12] (n=29) | n/a n/a (n=0, p=nan) | 0.54 | 0.22 | 0.48 | +0.22 |

### Per behaviour, judge `local`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 19 | 1.92 | 12.0 | 0.17 | 0.57 | +0.39 | 1.00 | 0.76 |
| ckpt:ppo_pickup_greedy_s0 | 19 | -0.03 | 197.9 | -0.00 | 0.26 | +0.26 | 0.45 | 0.25 |
| ckpt:ppo_shadow_greedy_s0 | 20 | 1.43 | 0.0 | 0.14 | 0.01 | -0.13 | 0.22 | 0.51 |
| fidget | 19 | -2.53 | 22.5 | -1.12 | 0.09 | +1.21 | 0.12 | 0.00 |
| greedy | 18 | 4.64 | 20.3 | 0.50 | 0.54 | +0.04 | 0.81 | 1.00 |

### Per behaviour, judge `local_definition`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 20 | 1.93 | 12.0 | 0.17 | 0.57 | +0.40 | 0.99 | 0.75 |
| ckpt:ppo_pickup_greedy_s0 | 20 | -0.03 | 197.9 | -0.00 | 0.10 | +0.11 | 0.00 | 0.00 |
| ckpt:ppo_shadow_greedy_s0 | 20 | 1.43 | 0.0 | 0.14 | 0.00 | -0.14 | 0.00 | 0.00 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.08 | +1.16 | 0.00 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.48 | -0.02 | 0.00 | 1.00 |

### Per behaviour, judge `local_solo`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 20 | 1.93 | 12.0 | 0.17 | 0.40 | +0.23 | 0.93 | 0.75 |
| ckpt:ppo_pickup_greedy_s0 | 20 | -0.03 | 197.9 | -0.00 | 0.01 | +0.02 | 0.00 | 0.00 |
| ckpt:ppo_shadow_greedy_s0 | 20 | 1.43 | 0.0 | 0.14 | 0.20 | +0.06 | 0.00 | 0.00 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.00 | +1.08 | 0.00 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.36 | -0.14 | 0.30 | 1.00 |

### Per behaviour, judge `ollama`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 19 | 1.92 | 12.0 | 0.17 | 0.61 | +0.43 | 1.00 | 0.77 |
| ckpt:ppo_pickup_greedy_s0 | 19 | -0.03 | 197.9 | -0.00 | 0.28 | +0.29 | 0.44 | 0.24 |
| ckpt:ppo_shadow_greedy_s0 | 20 | 1.43 | 0.0 | 0.14 | 0.00 | -0.14 | 0.22 | 0.51 |
| fidget | 19 | -2.53 | 22.5 | -1.12 | 0.11 | +1.22 | 0.14 | 0.00 |
| greedy | 19 | 4.63 | 20.1 | 0.50 | 0.54 | +0.04 | 0.81 | 1.00 |

### Per behaviour, judge `shapley`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 20 | 1.93 | 12.0 | 0.17 | 0.17 | +0.00 | 0.75 | 0.75 |
| ckpt:ppo_pickup_greedy_s0 | 20 | -0.03 | 197.9 | -0.00 | 0.01 | +0.01 | 0.24 | 0.24 |
| ckpt:ppo_shadow_greedy_s0 | 20 | 1.43 | 0.0 | 0.14 | 0.14 | +0.00 | 0.50 | 0.50 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.00 | +1.08 | 0.00 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.50 | +0.00 | 1.00 | 1.00 |
