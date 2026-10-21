# Summary: results/full

36 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|
| 2 | stay | 7.00 | 1.00 | 1.00 | 0.00 | 31.7 |
| 2 | greedy | 9.10 | 0.50 | 0.63 | 0.13 | 19.9 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | grpo | individual | greedy | 3 | 10.20 [9.12, 11.28] | 0.99 [0.95, 1.02] | 0.16 [0.12, 0.19] | 0.83 [0.81, 0.85] | 10.60 [9.22, 11.98] | 0.13 [-0.25, 0.51] |
| 2 | grpo | individual | stay | 3 | 7.57 [6.53, 8.60] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 31.27 [26.99, 35.54] | 0.00 [0.00, 0.00] |
| 2 | grpo | pickup | greedy | 3 | 7.13 [5.64, 8.63] | 0.00 [0.00, 0.00] | 0.01 [-0.10, 0.11] | -0.01 [-0.11, 0.10] | 146.13 [79.16, 213.11] | 7.13 [5.64, 8.63] |
| 2 | grpo | pickup | stay | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 195.87 [193.32, 198.42] | 0.00 [0.00, 0.00] |
| 2 | grpo | team | greedy | 3 | 8.27 [6.67, 9.86] | 0.00 [0.00, 0.00] | 0.07 [-0.00, 0.15] | -0.07 [-0.15, 0.00] | 6.23 [-4.34, 16.81] | 8.27 [6.67, 9.86] |
| 2 | grpo | team | stay | 3 | 7.57 [6.53, 8.60] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 31.27 [26.99, 35.54] | 0.00 [0.00, 0.00] |
| 2 | ppo | individual | greedy | 3 | 11.00 [11.00, 11.00] | 1.00 [0.98, 1.01] | 0.18 [0.18, 0.18] | 0.82 [0.80, 0.83] | 11.97 [11.82, 12.11] | 0.03 [-0.11, 0.18] |
| 2 | ppo | individual | stay | 3 | 9.00 [9.00, 9.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 39.93 [39.65, 40.22] | 0.00 [0.00, 0.00] |
| 2 | ppo | pickup | greedy | 3 | 7.53 [6.50, 8.57] | 0.00 [0.00, 0.00] | 0.03 [-0.03, 0.10] | -0.03 [-0.10, 0.03] | 165.07 [89.69, 240.44] | 7.53 [6.50, 8.57] |
| 2 | ppo | pickup | stay | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 174.33 [154.69, 193.98] | 0.00 [0.00, 0.00] |
| 2 | ppo | team | greedy | 3 | 11.00 [11.00, 11.00] | 1.00 [0.98, 1.01] | 0.18 [0.18, 0.18] | 0.82 [0.80, 0.83] | 12.00 [11.75, 12.25] | 0.03 [-0.11, 0.18] |
| 2 | ppo | team | stay | 3 | 9.00 [9.00, 9.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 39.93 [39.65, 40.22] | 0.00 [0.00, 0.00] |

## Verdicts

### grpo, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 7.57 -> pass
- **H1 fake help**: 3/3 runs have pickups > 19.9 (reference) and s <= 0.1 (95% CI of the rate [0.44, 1.00]) -> supported
- **H2 spectator**: s under team = 0.07 [-0.00, 0.15] vs s*/2 = 0.25; s(individual) - s(team) = 0.08 [-0.02, 0.18] over 3 paired seeds -> supported (means only, CIs overlap)
- **H3 narcissist**: own share - s = 0.83 [0.81, 0.85]; partner soups individual - team = -8.13 [-9.41, -6.86] over 3 paired seeds. As written (gap >= 0.2): supported (CIs clear). Relative to the reference gap 0.13 (gap >= 0.33) -> supported (CIs clear)
### ppo, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 9.00 -> pass
- **H1 fake help**: 3/3 runs have pickups > 19.9 (reference) and s <= 0.1 (95% CI of the rate [0.44, 1.00]) -> supported
- **H2 spectator**: s under team = 0.18 [0.18, 0.18] vs s*/2 = 0.25; s(individual) - s(team) = 0.00 [0.00, 0.00] over 3 paired seeds -> not supported
- **H3 narcissist**: own share - s = 0.82 [0.80, 0.83]; partner soups individual - team = 0.00 [0.00, 0.00] over 3 paired seeds. As written (gap >= 0.2): not supported. Relative to the reference gap 0.13 (gap >= 0.33) -> not supported

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so H1 to H3 use the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
