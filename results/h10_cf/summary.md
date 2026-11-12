# Summary: results/h10_cf

12 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | stay | default | 7.00 | 1.00 | 1.00 | 0.00 | 31.7 |
| 2 | greedy | default | 9.10 | 0.50 | 0.63 | 0.13 | 19.9 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups | dish/soup hold steps | interceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | grpo | cf | greedy | 3 | 8.07 [7.55, 8.58] | 0.00 [0.00, 0.00] | 0.07 [0.04, 0.09] | -0.07 [-0.09, -0.04] | 5.83 [0.38, 11.28] | 8.07 [7.55, 8.58] | 8.17 [-4.56, 20.90] | 0.00 [0.00, 0.00] |
| 2 | grpo | cf | stay | 3 | 7.57 [6.53, 8.60] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 31.27 [26.99, 35.54] | 0.00 [0.00, 0.00] | 157.53 [140.59, 174.47] | 0.00 [0.00, 0.00] |
| 2 | ppo | cf | greedy | 3 | 10.57 [10.05, 11.08] | 0.65 [-0.75, 2.05] | 0.17 [0.15, 0.18] | 0.48 [-0.91, 1.87] | 11.47 [9.21, 13.72] | 3.63 [-10.93, 18.20] | 200.47 [-230.81, 631.74] | 6.93 [-8.00, 21.87] |
| 2 | ppo | cf | stay | 3 | 9.00 [9.00, 9.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 39.93 [39.65, 40.22] | 0.00 [0.00, 0.00] | 123.00 [34.95, 211.05] | 0.00 [0.00, 0.00] |

## Verdicts

### grpo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- reference (scripted teammate in the learner seat): s* = 0.50, pickups* = 19.9, credit gap* = 0.13, partner soups* = 3.40
- **cf**: team soups 8.07 [7.55, 8.58], s = 0.07 [0.04, 0.09] (below s*/2 = 0.25, CI clear); fake help (pickups > 19.9 and s <= 0.1) 0/3 runs (95% CI [0.00, 0.56]); credit theft (gap >= 0.33 and partner soups < 3.40) 0/3 runs (95% CI [0.00, 0.56]) -> spectator
### ppo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- reference (scripted teammate in the learner seat): s* = 0.50, pickups* = 19.9, credit gap* = 0.13, partner soups* = 3.40
- **cf**: team soups 10.57 [10.05, 11.08], s = 0.17 [0.15, 0.18] (below s*/2 = 0.25, CI clear); fake help (pickups > 19.9 and s <= 0.1) 0/3 runs (95% CI [0.00, 0.56]); credit theft (gap >= 0.33 and partner soups < 3.40) 2/3 runs (95% CI [0.21, 0.94]) -> credit theft, spectator

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
