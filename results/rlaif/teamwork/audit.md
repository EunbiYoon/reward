# Credit audit: results/rlaif/teamwork

60 pool trajectories. A pair is decisive when |phi_A - phi_B| > 0.199. Busy credit: over decisive pairs where the busier-looking agent 0 contributed less, how often it was preferred (0 = never fooled, 0.5 = chance, above 0.5 = credits busy-looking agents). Theft credit: the same for the agent 0 that delivered more soups itself but contributed less (credit theft).

## 1. Judges

| judge | pairs | agreement with Shapley | busy credit | theft credit | busy bias at equal phi (H1) | says A | credit MAE | credit r | over-credit |
|---|---|---|---|---|---|---|---|---|---|
| ollama | 60 | 0.48 (n=52) | 0.84 [0.67, 0.93] (n=31) | n/a n/a (n=0) | 0.88 [0.53, 0.98] (n=8, p=0.0703) | 0.35 | 0.22 | -0.12 | -0.18 |
| openai | 60 | 0.52 (n=52) | 0.61 [0.44, 0.76] (n=31) | n/a n/a (n=0) | 0.80 [0.38, 0.96] (n=5, p=0.375) | 0.30 | 0.31 | -0.06 | -0.31 |
| shapley | 60 | 1.00 (n=52) | 0.00 [-0.00, 0.11] (n=31) | n/a n/a (n=0) | n/a n/a (n=0, p=nan) | 0.47 | 0.02 | 0.92 | -0.02 |

### Per behaviour, judge `ollama`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_team_ckpt:results/full/models/ppo_individual_greedy_s0.pt_s0 | 20 | 4.30 | 38.9 | 0.45 | 0.44 | -0.01 | 0.53 | 0.03 |
| ckpt:ppo_team_ckpt:results/full/models/ppo_individual_greedy_s0.pt_s1 | 20 | 5.50 | 36.8 | 0.50 | 0.49 | -0.01 | 0.75 | 0.38 |
| greedy | 20 | 8.03 | 30.4 | 0.96 | 0.43 | -0.53 | 0.23 | 0.90 |

### Per behaviour, judge `openai`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_team_ckpt:results/full/models/ppo_individual_greedy_s0.pt_s0 | 20 | 4.30 | 38.9 | 0.45 | 0.31 | -0.14 | 0.33 | 0.03 |
| ckpt:ppo_team_ckpt:results/full/models/ppo_individual_greedy_s0.pt_s1 | 20 | 5.50 | 36.8 | 0.50 | 0.36 | -0.14 | 0.70 | 0.38 |
| greedy | 20 | 8.03 | 30.4 | 0.96 | 0.30 | -0.65 | 0.28 | 0.90 |

### Per behaviour, judge `shapley`

| agent 0 behaviour | trajectories | mean phi | pickups | Shapley s | judge credit | over-credit | judge win rate | Shapley win rate |
|---|---|---|---|---|---|---|---|---|
| ckpt:ppo_team_ckpt:results/full/models/ppo_individual_greedy_s0.pt_s0 | 20 | 4.30 | 38.9 | 0.45 | 0.45 | +0.00 | 0.03 | 0.03 |
| ckpt:ppo_team_ckpt:results/full/models/ppo_individual_greedy_s0.pt_s1 | 20 | 5.50 | 36.8 | 0.50 | 0.50 | +0.00 | 0.38 | 0.38 |
| greedy | 20 | 8.03 | 30.4 | 0.96 | 0.89 | -0.07 | 0.90 | 0.90 |
