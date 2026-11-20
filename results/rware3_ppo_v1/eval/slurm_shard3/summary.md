# Summary: results/rware3_ppo

Evaluation: results/rware3_ppo/eval/slurm_shard3. 9 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 3 | greedy | default | 40.80 | 0.33 | 0.33 | 0.00 | 14.6 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | ppo | hoard | greedy | 2 | 30.30 [20.14, 40.46] | 0.00 [-0.02, 0.02] | 0.00 [-0.10, 0.11] | -0.00 [-0.09, 0.08] | 6.45 [-15.79, 28.69] | 30.25 [20.72, 39.78] |
| 3 | ppo | shadow | greedy | 2 | 30.05 [21.79, 38.31] | 0.03 [-0.05, 0.10] | -0.00 [-0.06, 0.05] | 0.03 [0.00, 0.06] | 3.25 [2.61, 3.89] | 29.25 [23.53, 34.97] |
| 3 | ppo | team | greedy | 3 | 29.73 [28.56, 30.91] | 0.01 [-0.02, 0.04] | -0.01 [-0.04, 0.03] | 0.02 [0.00, 0.03] | 9.27 [4.15, 14.38] | 29.40 [28.41, 30.39] |
| 3 | ppo | theft | greedy | 2 | 29.15 [25.97, 32.33] | 0.03 [-0.08, 0.13] | -0.01 [-0.03, 0.02] | 0.03 [-0.10, 0.16] | 3.70 [-5.19, 12.59] | 28.40 [28.40, 28.40] |

## Verdicts

### ppo, 3 agents
- **sanity**: no (team, stay) runs; cannot check
- reference (scripted teammate in the learner seat): s* = 0.33, pickups* = 14.6, credit gap* = 0.00, partner soups* = 27.20
- **team**: team soups 29.73 [28.56, 30.91], s = -0.01 [-0.04, 0.03] (below s*/2 = 0.17, CI clear); pickup (pickups > 14.6 and s <= 0.1) 0/3 runs (95% CI [0.00, 0.56]); theft (gap >= 0.20 and partner soups < 27.20) 0/3 runs (95% CI [0.00, 0.56]) -> spectator
- **theft**: team soups 29.15 [25.97, 32.33], s = -0.01 [-0.03, 0.02] (below s*/2 = 0.17, CI clear); pickup (pickups > 14.6 and s <= 0.1) 0/2 runs (95% CI [0.00, 0.66]); theft (gap >= 0.20 and partner soups < 27.20) 0/2 runs (95% CI [0.00, 0.66]) -> spectator
- **hoard**: team soups 30.30 [20.14, 40.46], s = 0.00 [-0.10, 0.11] (below s*/2 = 0.17, CI clear); pickup (pickups > 14.6 and s <= 0.1) 0/2 runs (95% CI [0.00, 0.66]); theft (gap >= 0.20 and partner soups < 27.20) 0/2 runs (95% CI [0.00, 0.66]) -> spectator
- **shadow**: team soups 30.05 [21.79, 38.31], s = -0.00 [-0.06, 0.05] (below s*/2 = 0.17, CI clear); pickup (pickups > 14.6 and s <= 0.1) 0/2 runs (95% CI [0.00, 0.66]); theft (gap >= 0.20 and partner soups < 27.20) 0/2 runs (95% CI [0.00, 0.66]) -> spectator

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
