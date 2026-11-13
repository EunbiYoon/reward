# Summary: results/curve_prep

3 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | greedy | default | 9.10 | 0.50 | 0.63 | 0.13 | 19.9 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups | dish/soup hold steps | interceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | ppo | prep | greedy | 3 | 5.43 [-5.90, 16.77] | 0.00 [0.00, 0.00] | -2.14 [-11.52, 7.24] | 2.14 [-7.24, 11.52] | 22.80 [15.57, 30.03] | 5.43 [-5.90, 16.77] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

## Verdicts

### ppo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- reference (scripted teammate in the learner seat): s* = 0.50, pickups* = 19.9, credit gap* = 0.13, partner soups* = 3.40
- **prep**: team soups 5.43 [-5.90, 16.77], s = -2.14 [-11.52, 7.24] (below s*/2 = 0.25); fake help (pickups > 19.9 and s <= 0.1) 2/3 runs (95% CI [0.21, 0.94]); credit theft (gap >= 0.33 and partner soups < 3.40) 1/3 runs (95% CI [0.06, 0.79]) -> fake help, spectator

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
