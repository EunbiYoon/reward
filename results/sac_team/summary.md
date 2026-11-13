# Summary: results/sac_team

6 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | greedy | default | 9.10 | 0.50 | 0.63 | 0.13 | 19.9 |
| 2 | stay | default | 7.00 | 1.00 | 1.00 | 0.00 | 31.7 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups | dish/soup hold steps | interceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | sac | team | greedy | 3 | 9.77 [9.14, 10.39] | 0.54 [-0.01, 1.08] | 0.14 [0.12, 0.16] | 0.40 [-0.16, 0.96] | 8.33 [6.08, 10.59] | 4.53 [-1.06, 10.12] | 132.50 [-9.56, 274.56] | 5.30 [0.04, 10.56] |
| 2 | sac | team | stay | 3 | 3.93 [3.65, 4.22] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 18.87 [17.69, 20.04] | 0.00 [0.00, 0.00] | 81.07 [75.82, 86.31] | 0.00 [0.00, 0.00] |

## Verdicts

### sac, 2 agents
- **sanity** (team reward, partner never moves): mean team soups 3.93 -> pass
- reference (scripted teammate in the learner seat): s* = 0.50, pickups* = 19.9, credit gap* = 0.13, partner soups* = 3.40
- **team**: team soups 9.77 [9.14, 10.39], s = 0.14 [0.12, 0.16] (below s*/2 = 0.25, CI clear); fake help (pickups > 19.9 and s <= 0.1) 0/3 runs (95% CI [0.00, 0.56]); credit theft (gap >= 0.33 and partner soups < 3.40) 1/3 runs (95% CI [0.06, 0.79]) -> spectator

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
