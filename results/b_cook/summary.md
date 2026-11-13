# Summary: results/b_cook

3 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | ckpt:results/full/models/ppo_individual_greedy_s0.pt | default | 8.50 | 0.91 | 0.06 | -0.85 | 29.1 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups | dish/soup hold steps | interceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | ppo | team | ckpt:results/full/models/ppo_individual_greedy_s0.pt | 3 | 9.97 [6.54, 13.39] | 0.00 [0.00, 0.00] | 0.50 [0.50, 0.50] | -0.50 [-0.50, -0.50] | 33.73 [23.21, 44.26] | 9.97 [6.54, 13.39] | 7.77 [-25.65, 41.18] | 0.00 [0.00, 0.00] |

## Verdicts

### ppo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- no greedy reference in reference.csv; the diagnosis needs it

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
