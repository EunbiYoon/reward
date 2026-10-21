# Summary: results/lbf_full

36 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|
| 2 | stay | 0.49 | 1.00 | 1.00 | 0.00 | 28.9 |
| 2 | greedy | 1.00 | 0.58 | 0.56 | -0.02 | 3.5 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | grpo | individual | greedy | 3 | 0.69 [0.45, 0.93] | 0.42 [0.27, 0.57] | 0.22 [0.11, 0.34] | 0.20 [-0.05, 0.44] | 1.67 [0.01, 3.32] | 0.40 [0.23, 0.56] |
| 2 | grpo | individual | stay | 3 | 0.30 [0.21, 0.40] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 8.60 [3.71, 13.49] | 0.00 [0.00, 0.00] |
| 2 | grpo | pickup | greedy | 3 | 0.74 [0.24, 1.23] | 0.41 [0.22, 0.60] | 0.25 [0.13, 0.38] | 0.16 [0.09, 0.22] | 3.47 [-2.09, 9.02] | 0.43 [0.28, 0.57] |
| 2 | grpo | pickup | stay | 3 | 0.18 [-0.04, 0.39] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 12.87 [-1.54, 27.28] | 0.00 [0.00, 0.00] |
| 2 | grpo | team | greedy | 3 | 0.78 [0.47, 1.09] | 0.37 [0.29, 0.45] | 0.26 [0.06, 0.45] | 0.11 [-0.13, 0.36] | 2.37 [0.47, 4.26] | 0.49 [0.25, 0.74] |
| 2 | grpo | team | stay | 3 | 0.30 [0.21, 0.40] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 8.60 [3.71, 13.49] | 0.00 [0.00, 0.00] |
| 2 | ppo | individual | greedy | 3 | 0.64 [0.40, 0.88] | 0.34 [0.21, 0.47] | 0.26 [0.20, 0.31] | 0.08 [-0.05, 0.22] | 2.57 [-0.09, 5.22] | 0.42 [0.29, 0.55] |
| 2 | ppo | individual | stay | 3 | 0.25 [0.11, 0.38] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 3.27 [-3.05, 9.58] | 0.00 [0.00, 0.00] |
| 2 | ppo | pickup | greedy | 3 | 0.61 [0.45, 0.77] | 0.39 [0.29, 0.49] | 0.24 [0.07, 0.41] | 0.15 [-0.01, 0.31] | 6.27 [0.52, 12.01] | 0.37 [0.30, 0.44] |
| 2 | ppo | pickup | stay | 3 | 0.23 [0.11, 0.34] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 31.73 [30.37, 33.10] | 0.00 [0.00, 0.00] |
| 2 | ppo | team | greedy | 3 | 0.49 [0.40, 0.57] | 0.27 [0.14, 0.41] | 0.14 [0.09, 0.18] | 0.14 [0.03, 0.24] | 1.23 [-0.46, 2.92] | 0.35 [0.30, 0.40] |
| 2 | ppo | team | stay | 3 | 0.25 [0.11, 0.38] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 3.27 [-3.05, 9.58] | 0.00 [0.00, 0.00] |

## Verdicts

### grpo, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 0.30 -> pass
- **H1 fake help**: 0/3 runs have pickups > 3.5 (reference) and s <= 0.1 (95% CI of the rate [0.00, 0.56]) -> not supported
- **H2 spectator**: s under team = 0.26 [0.06, 0.45] vs s*/2 = 0.29; s(individual) - s(team) = -0.03 [-0.14, 0.07] over 3 paired seeds -> not supported
- **H3 narcissist**: own share - s = 0.20 [-0.05, 0.44]; partner soups individual - team = -0.09 [-0.21, 0.02] over 3 paired seeds. As written (gap >= 0.2): not supported. Relative to the reference gap -0.02 (gap >= 0.18) -> supported (means only, CIs overlap)
### ppo, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 0.25 -> pass
- **H1 fake help**: 0/3 runs have pickups > 3.5 (reference) and s <= 0.1 (95% CI of the rate [0.00, 0.56]) -> not supported
- **H2 spectator**: s under team = 0.14 [0.09, 0.18] vs s*/2 = 0.29; s(individual) - s(team) = 0.12 [0.07, 0.16] over 3 paired seeds -> supported (CIs clear)
- **H3 narcissist**: own share - s = 0.08 [-0.05, 0.22]; partner soups individual - team = 0.07 [-0.09, 0.22] over 3 paired seeds. As written (gap >= 0.2): not supported. Relative to the reference gap -0.02 (gap >= 0.18) -> not supported

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so H1 to H3 use the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
