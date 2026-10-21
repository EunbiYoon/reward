# Summary: results/rware_full

36 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|
| 2 | stay | 15.50 | 1.00 | 1.00 | 0.00 | 17.0 |
| 2 | greedy | 28.30 | 0.50 | 0.49 | -0.01 | 15.0 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | grpo | individual | greedy | 3 | 14.80 [13.66, 15.94] | 0.02 [0.00, 0.04] | -0.02 [-0.06, 0.02] | 0.04 [0.02, 0.07] | 6.43 [0.51, 12.36] | 14.50 [13.60, 15.40] |
| 2 | grpo | individual | stay | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 86.70 [48.95, 124.45] | 0.00 [0.00, 0.00] |
| 2 | grpo | pickup | greedy | 3 | 15.00 [14.10, 15.90] | 0.00 [0.00, 0.00] | -0.02 [-0.05, 0.01] | 0.02 [-0.01, 0.05] | 155.43 [123.27, 187.59] | 15.00 [14.10, 15.90] |
| 2 | grpo | pickup | stay | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 242.40 [236.80, 248.00] | 0.00 [0.00, 0.00] |
| 2 | grpo | team | greedy | 3 | 15.23 [14.66, 15.81] | 0.00 [-0.01, 0.01] | -0.01 [-0.03, 0.02] | 0.01 [-0.00, 0.02] | 7.20 [5.65, 8.75] | 15.20 [14.77, 15.63] |
| 2 | grpo | team | stay | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 86.70 [48.95, 124.45] | 0.00 [0.00, 0.00] |
| 2 | ppo | individual | greedy | 3 | 15.40 [14.97, 15.83] | 0.03 [0.03, 0.03] | -0.00 [-0.01, 0.01] | 0.03 [0.01, 0.04] | 5.47 [-2.37, 13.30] | 15.00 [14.57, 15.43] |
| 2 | ppo | individual | stay | 3 | 0.03 [-0.11, 0.18] | 0.33 [-1.10, 1.77] | 0.33 [-1.10, 1.77] | 0.00 [0.00, 0.00] | 14.73 [-10.81, 40.28] | 0.00 [0.00, 0.00] |
| 2 | ppo | pickup | greedy | 3 | 15.20 [14.95, 15.45] | 0.00 [0.00, 0.00] | -0.01 [-0.02, -0.00] | 0.01 [0.00, 0.02] | 233.80 [214.65, 252.95] | 15.20 [14.95, 15.45] |
| 2 | ppo | pickup | stay | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 240.33 [221.76, 258.91] | 0.00 [0.00, 0.00] |
| 2 | ppo | team | greedy | 3 | 14.27 [13.69, 14.84] | 0.00 [-0.01, 0.01] | -0.04 [-0.07, -0.02] | 0.05 [0.03, 0.06] | 9.20 [-3.08, 21.48] | 14.23 [13.72, 14.75] |
| 2 | ppo | team | stay | 3 | 0.03 [-0.11, 0.18] | 0.33 [-1.10, 1.77] | 0.33 [-1.10, 1.77] | 0.00 [0.00, 0.00] | 14.73 [-10.81, 40.28] | 0.00 [0.00, 0.00] |

## Verdicts

### grpo, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 0.00 -> **FAIL: domain inconclusive, train longer**
- **H1 fake help**: 3/3 runs have pickups > 15.0 (reference) and s <= 0.1 (95% CI of the rate [0.44, 1.00]) -> inconclusive (sanity failed; would read: supported)
- **H2 spectator**: s under team = -0.01 [-0.03, 0.02] vs s*/2 = 0.25; s(individual) - s(team) = -0.02 [-0.04, 0.00] over 3 paired seeds -> inconclusive (sanity failed; would read: not supported)
- **H3 narcissist**: own share - s = 0.04 [0.02, 0.07]; partner soups individual - team = -0.70 [-1.20, -0.20] over 3 paired seeds. As written (gap >= 0.2): not supported. Relative to the reference gap -0.01 (gap >= 0.19) -> inconclusive (sanity failed; would read: not supported)
### ppo, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 0.03 -> pass
- **H1 fake help**: 3/3 runs have pickups > 15.0 (reference) and s <= 0.1 (95% CI of the rate [0.44, 1.00]) -> supported
- **H2 spectator**: s under team = -0.04 [-0.07, -0.02] vs s*/2 = 0.25; s(individual) - s(team) = 0.04 [0.02, 0.06] over 3 paired seeds -> supported (CIs clear)
- **H3 narcissist**: own share - s = 0.03 [0.01, 0.04]; partner soups individual - team = 0.77 [0.39, 1.15] over 3 paired seeds. As written (gap >= 0.2): not supported. Relative to the reference gap -0.01 (gap >= 0.19) -> not supported

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so H1 to H3 use the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
