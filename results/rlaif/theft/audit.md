# Credit audit: results/rlaif/theft

100 pool trajectories. A pair is decisive when |phi_A - phi_B| > 0.240. Busy credit: over decisive pairs where the busier-looking agent 0 contributed less, how often it was preferred (0 = never fooled, 0.5 = chance, above 0.5 = credits busy-looking agents). Theft credit: the same for the agent 0 that delivered more soups itself but contributed less (credit theft).

## 1. Judges

| judge | pairs | agreement with Shapley | busy credit | theft credit | busy bias at equal phi (H1) | says A | credit MAE | credit r | over-credit |
|---|---|---|---|---|---|---|---|---|---|
| ollama | 116 | 0.87 (n=114) | 0.04 [0.01, 0.11] (n=76) | 0.88 [0.66, 0.97] (n=17) | 0.50 [0.09, 0.91] (n=2, p=1) | 0.59 | 0.46 | 0.33 | +0.43 |
| ollama_definition | 6 | 0.67 (n=6) | 0.00 [0.00, 0.49] (n=4) | 1.00 [0.34, 1.00] (n=2) | n/a n/a (n=0, p=nan) | 1.00 | 0.53 | 0.39 | +0.52 |
| ollama_solo | 7 | 0.71 (n=7) | 0.00 [-0.00, 0.43] (n=5) | 1.00 [0.34, 1.00] (n=2) | n/a n/a (n=0, p=nan) | 1.00 | 0.47 | 0.31 | +0.40 |
| openai | 63 | 0.79 (n=63) | 0.09 [0.03, 0.20] (n=46) | 1.00 [0.65, 1.00] (n=7) | n/a n/a (n=0, p=nan) | 0.56 | 0.48 | 0.27 | +0.46 |
| shapley | 200 | 1.00 (n=195) | 0.00 [-0.00, 0.03] (n=120) | 0.00 [0.00, 0.12] (n=29) | n/a n/a (n=0, p=nan) | 0.58 | 0.22 | 0.46 | +0.22 |

### Per behaviour, judge `ollama`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_team_greedy_s0 | 20 | 0.47 | 11.8 | 0.06 | 0.13 | +0.07 | 0.44 | 0.51 |
| ckpt:ppo_individual_greedy_s0 | 19 | 1.92 | 12.1 | 0.17 | 0.90 | +0.72 | 1.00 | 0.77 |
| ckpt:ppo_pickup_greedy_s0 | 20 | 0.00 | 178.0 | -0.00 | 0.14 | +0.15 | 0.23 | 0.25 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.11 | +1.19 | 0.12 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.55 | +0.05 | 0.79 | 1.00 |

### Per behaviour, judge `ollama_definition`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 6 | 2.00 | 12.2 | 0.18 | 0.70 | +0.52 | 1.00 | 0.67 |
| ckpt:ppo_pickup_greedy_s0 | 3 | 0.17 | 178.0 | 0.02 | 0.03 | +0.01 | 0.00 | 0.00 |
| fidget | 1 | -3.00 | 22.0 | -3.00 | 0.10 | +3.10 | 0.00 | 0.00 |
| greedy | 2 | 4.75 | 20.5 | 0.50 | 0.48 | -0.02 | 0.00 | 1.00 |

### Per behaviour, judge `ollama_solo`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_individual_greedy_s0 | 6 | 2.00 | 12.2 | 0.18 | 0.68 | +0.50 | 1.00 | 0.67 |
| ckpt:ppo_pickup_greedy_s0 | 4 | 0.12 | 178.0 | 0.02 | 0.03 | +0.01 | 0.00 | 0.00 |
| fidget | 1 | -3.00 | 22.0 | -3.00 | 0.10 | +3.10 | 0.00 | 0.00 |
| greedy | 3 | 4.67 | 19.7 | 0.50 | 0.34 | -0.16 | 0.33 | 1.00 |

### Per behaviour, judge `openai`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_team_greedy_s0 | 12 | 0.46 | 12.3 | 0.06 | 0.12 | +0.06 | 0.33 | 0.61 |
| ckpt:ppo_individual_greedy_s0 | 17 | 1.97 | 12.1 | 0.18 | 0.92 | +0.74 | 1.00 | 0.80 |
| ckpt:ppo_pickup_greedy_s0 | 16 | 0.03 | 178.0 | 0.00 | 0.11 | +0.11 | 0.15 | 0.22 |
| fidget | 17 | -2.44 | 23.0 | -0.95 | 0.15 | +1.10 | 0.24 | 0.00 |
| greedy | 14 | 4.64 | 20.4 | 0.50 | 0.58 | +0.08 | 0.73 | 1.00 |

### Per behaviour, judge `shapley`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:grpo_team_greedy_s0 | 20 | 0.47 | 11.8 | 0.06 | 0.06 | +0.00 | 0.45 | 0.45 |
| ckpt:ppo_individual_greedy_s0 | 20 | 1.93 | 12.1 | 0.17 | 0.17 | +0.00 | 0.75 | 0.75 |
| ckpt:ppo_pickup_greedy_s0 | 20 | 0.00 | 178.0 | -0.00 | 0.01 | +0.01 | 0.24 | 0.24 |
| fidget | 20 | -2.48 | 22.1 | -1.08 | 0.00 | +1.08 | 0.00 | 0.00 |
| greedy | 20 | 4.65 | 20.2 | 0.50 | 0.50 | +0.00 | 1.00 | 1.00 |
