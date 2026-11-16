# Summary: results/ppo/ppo_shadow

Evaluation: results/ppo/ppo_shadow/eval/20261115_230636. 3 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | ckpt:results/full/models/ppo_team_greedy_s0.pt | default | 11.00 | 0.82 | 0.00 | -0.82 | 37.0 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups | dish/soup hold steps | interceptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | ppo | shadow | ckpt:results/full/models/ppo_team_greedy_s0.pt | 3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

## Verdicts

### ppo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- no greedy reference in reference.csv; the diagnosis needs it

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
