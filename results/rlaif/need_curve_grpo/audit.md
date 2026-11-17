# Credit audit: results/rlaif/need_curve_grpo

120 pool trajectories. A pair is decisive when |phi_A - phi_B| > 0.218. Busy credit: over decisive pairs where the busier-looking agent 0 contributed less, how often it was preferred (0 = never fooled, 0.5 = chance, above 0.5 = credits busy-looking agents). Theft credit: the same for the agent 0 that delivered more soups itself but contributed less (credit theft).

## 1. Judges

| judge | pairs | agreement with Shapley | busy credit | theft credit | busy bias at equal phi (H1) | says A | credit MAE | credit r | over-credit |
|---|---|---|---|---|---|---|---|---|---|
| local | 100 | 0.83 (n=93) | 0.13 [0.06, 0.24] (n=54) | 0.64 [0.39, 0.84] (n=14) | 0.20 [0.04, 0.62] (n=5, p=0.375) | 0.46 | 0.38 | 0.29 | +0.34 |
| local_definition | 99 | 0.82 (n=99) | 0.00 [0.00, 0.07] (n=55) | 1.00 [0.82, 1.00] (n=18) | n/a n/a (n=0, p=nan) | 0.41 | 0.32 | 0.26 | +0.26 |
| local_solo | 100 | 0.92 (n=100) | 0.00 [0.00, 0.07] (n=55) | 0.37 [0.19, 0.59] (n=19) | n/a n/a (n=0, p=nan) | 0.50 | 0.28 | 0.44 | +0.24 |
| ollama | 103 | 0.80 (n=96) | 0.18 [0.10, 0.30] (n=56) | 0.64 [0.39, 0.84] (n=14) | 0.50 [0.19, 0.81] (n=6, p=1) | 0.48 | 0.39 | 0.23 | +0.34 |
| shapley | 300 | 1.00 (n=276) | 0.00 [0.00, 0.02] (n=164) | 0.00 [0.00, 0.10] (n=34) | n/a n/a (n=0, p=nan) | 0.48 | 0.19 | 0.45 | +0.19 |

### Per behaviour, judge `local`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_individual_greedy_s0 | 17 | 1.65 | 10.9 | 0.16 | 0.60 | +0.45 | 0.97 | 0.81 |
| ckpt:grpo_pickup_greedy_s0 | 16 | 0.25 | 138.5 | -0.00 | 0.25 | +0.25 | 0.31 | 0.24 |
| ckpt:grpo_shadow_greedy_s0 | 18 | 0.83 | 2.8 | 0.09 | 0.04 | -0.05 | 0.40 | 0.54 |
| ckpt:grpo_team_greedy_s0 | 17 | 0.44 | 17.4 | 0.05 | 0.08 | +0.03 | 0.36 | 0.28 |
| fidget | 18 | -2.42 | 22.4 | -1.15 | 0.10 | +1.25 | 0.14 | 0.00 |
| greedy | 17 | 4.68 | 20.2 | 0.50 | 0.57 | +0.07 | 0.74 | 1.00 |

### Per behaviour, judge `local_definition`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_individual_greedy_s0 | 20 | 1.65 | 10.8 | 0.16 | 0.59 | +0.43 | 0.99 | 0.81 |
| ckpt:grpo_pickup_greedy_s0 | 20 | 0.28 | 140.9 | 0.01 | 0.06 | +0.05 | 0.00 | 0.00 |
| ckpt:grpo_shadow_greedy_s0 | 20 | 0.85 | 2.9 | 0.10 | 0.05 | -0.05 | 0.00 | 0.00 |
| ckpt:grpo_team_greedy_s0 | 20 | 0.45 | 17.4 | 0.06 | 0.03 | -0.03 | 0.00 | 0.00 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.07 | +1.15 | 0.00 | 0.00 |
| greedy | 19 | 4.63 | 20.1 | 0.50 | 0.47 | -0.03 | 0.05 | 1.00 |

### Per behaviour, judge `local_solo`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_individual_greedy_s0 | 20 | 1.65 | 10.8 | 0.16 | 0.37 | +0.21 | 0.87 | 0.80 |
| ckpt:grpo_pickup_greedy_s0 | 20 | 0.28 | 140.9 | 0.01 | 0.09 | +0.08 | 0.00 | 0.00 |
| ckpt:grpo_shadow_greedy_s0 | 20 | 0.85 | 2.9 | 0.10 | 0.18 | +0.08 | 0.00 | 0.00 |
| ckpt:grpo_team_greedy_s0 | 20 | 0.45 | 17.4 | 0.06 | 0.10 | +0.05 | 0.00 | 0.00 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.01 | +1.09 | 0.00 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.45 | -0.05 | 0.60 | 1.00 |

### Per behaviour, judge `ollama`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_individual_greedy_s0 | 18 | 1.64 | 10.8 | 0.16 | 0.62 | +0.46 | 0.97 | 0.82 |
| ckpt:grpo_pickup_greedy_s0 | 16 | 0.25 | 138.5 | -0.00 | 0.24 | +0.24 | 0.39 | 0.23 |
| ckpt:grpo_shadow_greedy_s0 | 18 | 0.83 | 2.8 | 0.09 | 0.05 | -0.05 | 0.29 | 0.54 |
| ckpt:grpo_team_greedy_s0 | 18 | 0.44 | 17.3 | 0.05 | 0.10 | +0.04 | 0.35 | 0.27 |
| fidget | 18 | -2.42 | 22.4 | -1.15 | 0.12 | +1.28 | 0.17 | 0.00 |
| greedy | 17 | 4.68 | 20.2 | 0.50 | 0.57 | +0.07 | 0.79 | 1.00 |

### Per behaviour, judge `shapley`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_individual_greedy_s0 | 20 | 1.65 | 10.8 | 0.16 | 0.16 | +0.00 | 0.80 | 0.80 |
| ckpt:grpo_pickup_greedy_s0 | 20 | 0.28 | 140.9 | 0.01 | 0.05 | +0.04 | 0.20 | 0.20 |
| ckpt:grpo_shadow_greedy_s0 | 20 | 0.85 | 2.9 | 0.10 | 0.10 | +0.00 | 0.53 | 0.53 |
| ckpt:grpo_team_greedy_s0 | 20 | 0.45 | 17.4 | 0.06 | 0.06 | +0.00 | 0.23 | 0.23 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.00 | +1.08 | 0.00 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.50 | +0.00 | 1.00 | 1.00 |
