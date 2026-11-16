# Summary: results/ppo/ppo_pickup

Evaluation: results/ppo/ppo_pickup/eval/20261115_230627. 6 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | ckpt:results/full/models/ppo_team_greedy_s0.pt | default | 10.10 | 0.85 | 0.01 | -0.84 | 34.4 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups | dish/soup hold steps | interceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | ppo | pickup | ckpt:results/full/models/ppo_team_greedy_s0.pt | 6 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 119.12 [28.30, 209.94] | 0.00 [0.00, 0.00] | 68.72 [-43.00, 180.43] | 0.00 [0.00, 0.00] |

## Verdicts

### ppo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- no greedy reference in reference.csv; the diagnosis needs it

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
