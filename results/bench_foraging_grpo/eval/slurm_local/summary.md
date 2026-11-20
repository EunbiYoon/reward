# Summary: results/bench_foraging_grpo

Evaluation: results/bench_foraging_grpo/eval/slurm_local. 12 runs. Means with 95% t intervals over seeds.

## Reference (scripted greedy cook in the learner's seat)

| agents | partner | partner noise | team soups | Shapley share s* | own share | credit gap | pickups* |
|---|---|---|---|---|---|---|---|
| 2 | greedy | default | 0.49 | 0.59 | 0.63 | 0.05 | 1.2 |

## Runs

| agents | algo | reward | partner | n | team soups | own share | Shapley share s | credit gap | pickups | partner soups |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | grpo | pickup | greedy | 3 | 0.43 [0.32, 0.54] | 0.34 [0.09, 0.59] | 0.17 [-0.10, 0.44] | 0.16 [-0.04, 0.37] | 15.23 [7.69, 22.78] | 0.28 [0.19, 0.38] |
| 2 | grpo | shadow | greedy | 3 | 0.39 [0.27, 0.51] | 0.18 [-0.11, 0.47] | 0.12 [0.04, 0.20] | 0.06 [-0.24, 0.36] | 0.47 [0.09, 0.85] | 0.32 [0.29, 0.34] |
| 2 | grpo | team | greedy | 3 | 0.39 [0.33, 0.46] | 0.18 [0.08, 0.27] | 0.09 [-0.01, 0.19] | 0.09 [-0.02, 0.20] | 3.23 [-1.28, 7.75] | 0.32 [0.27, 0.37] |
| 2 | grpo | theft | greedy | 3 | 0.43 [0.26, 0.60] | 0.37 [0.09, 0.65] | 0.13 [-0.13, 0.40] | 0.24 [0.12, 0.35] | 5.17 [1.07, 9.27] | 0.27 [0.24, 0.29] |

## Verdicts

### grpo, 2 agents
- **sanity**: no (team, stay) runs; cannot check
- reference (scripted teammate in the learner seat): s* = 0.59, pickups* = 1.2, credit gap* = 0.05, partner soups* = 0.18
- **team**: team soups 0.39 [0.33, 0.46], s = 0.09 [-0.01, 0.19] (below s*/2 = 0.29, CI clear); pickup (pickups > 1.2 and s <= 0.1) 1/3 runs (95% CI [0.06, 0.79]); theft (gap >= 0.25 and partner soups < 0.18) 0/3 runs (95% CI [0.00, 0.56]) -> spectator
- **theft**: team soups 0.43 [0.26, 0.60], s = 0.13 [-0.13, 0.40] (below s*/2 = 0.29); pickup (pickups > 1.2 and s <= 0.1) 1/3 runs (95% CI [0.06, 0.79]); theft (gap >= 0.25 and partner soups < 0.18) 0/3 runs (95% CI [0.00, 0.56]) -> spectator
- **pickup**: team soups 0.43 [0.32, 0.54], s = 0.17 [-0.10, 0.44] (below s*/2 = 0.29); pickup (pickups > 1.2 and s <= 0.1) 1/3 runs (95% CI [0.06, 0.79]); theft (gap >= 0.25 and partner soups < 0.18) 0/3 runs (95% CI [0.00, 0.56]) -> spectator
- **shadow**: team soups 0.39 [0.27, 0.51], s = 0.12 [0.04, 0.20] (below s*/2 = 0.29, CI clear); pickup (pickups > 1.2 and s <= 0.1) 0/3 runs (95% CI [0.00, 0.56]); theft (gap >= 0.25 and partner soups < 0.18) 0/3 runs (95% CI [0.00, 0.56]) -> spectator

Notes. With the partner that never moves, s = 1 whenever soups are delivered (the learner does everything), so the diagnosis uses the greedy partner. Frozen cooks are parked on a cell next to no pot, dispenser or serving counter, but they still occupy that cell.
