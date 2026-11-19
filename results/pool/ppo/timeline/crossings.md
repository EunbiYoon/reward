# When the judge over-credits agent 0: results/rlaif/need_curve_ppo

Judge Qwen/Qwen3.8-27B-FP8, prompt basic, episodes cut every 50 steps; s(t) = Shapley share of agent 0 up to t (replayed counterfactuals), o(t) = judge credit, gap = o - s. A crossing is the first t with gap > 0.1; the window is the 50 steps before it.

## By behaviour

| agent 0 behaviour | trajectories | ever over-credited | median first t | mean gap at end |
|---|---|---|---|---|
| ckpt:ppo_theft_greedy_s0 | 19 | 19 | 50 | +0.40 |
| ckpt:ppo_pickup_greedy_s0 | 19 | 17 | 100 | +0.27 |
| ckpt:ppo_shadow_greedy_s0 | 20 | 0 | - | -0.14 |
| fidget | 19 | 14 | 100 | +1.20 |
| greedy | 18 | 13 | 50 | +0.04 |

## Every crossing

| trajectory | behaviour | first t | s | o | team soups so far | agent 0 in the window | agent 1 in the window |
|---|---|---|---|---|---|---|---|
| greedy_n2_s0_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 4, INTERACT 6, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s10_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.50 | 0.75 | 1 | idle 9, INTERACT 4, dish 1, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s11_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.25 | 0.50 | 2 | idle 12, INTERACT 6, dish 2, scoop soup 1, delivered 1 | INTERACT 25, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s12_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 6, INTERACT 5, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s14_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 6, INTERACT 7, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s15_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.25 | 0.50 | 2 | idle 7, INTERACT 6, dish 1, scoop soup 1, delivered 1 | INTERACT 25, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s16_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 7, INTERACT 6, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s17_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.50 | 1.00 | 1 | idle 8, INTERACT 5, dish 1, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s18_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.25 | 0.50 | 2 | idle 8, INTERACT 4, dish 1, scoop soup 1, delivered 1 | INTERACT 24, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s19_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 5, INTERACT 8, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s1_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.25 | 0.60 | 2 | idle 8, INTERACT 5, dish 1, scoop soup 1, delivered 1 | INTERACT 25, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s2_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 7, INTERACT 7, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s3_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.25 | 0.50 | 2 | idle 9, INTERACT 7, dish 1, scoop soup 1, delivered 1 | INTERACT 25, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s4_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.00 | 0.50 | 2 | idle 10, INTERACT 6, dish 1, scoop soup 1, delivered 1 | INTERACT 25, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s5_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 100 | 0.25 | 0.50 | 2 | idle 9, INTERACT 8, dish 1, scoop soup 1, delivered 1 | INTERACT 25, onion 4, onion in pot 3, start cook 1 |
| greedy_n2_s6_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.43 | 1 | idle 7, INTERACT 5, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s7_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.45 | 1 | idle 6, INTERACT 8, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s8_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.00 | 0.50 | 1 | idle 5, INTERACT 8, dish 2, scoop soup 1, delivered 1 | INTERACT 23, onion 6, onion in pot 6, start cook 2 |
| greedy_n2_s9_ckpt-ppo_theft_greedy_s0 | ckpt:ppo_theft_greedy_s0 | 50 | 0.50 | 0.83 | 1 | idle 8, INTERACT 6, dish 2, scoop soup 1, delivered 1 | INTERACT 21, onion 6, onion in pot 5, start cook 1 |
| greedy_n2_s0_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 50 | 0.00 | 0.60 | 1 | INTERACT 45, onion 23 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s10_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 200 | 0.12 | 0.33 | 4 | INTERACT 50, onion 25 | INTERACT 15, onion 3, dish 1, onion in pot 4, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s11_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 100 | 0.00 | 0.23 | 1 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s12_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 50 | 0.00 | 0.25 | 1 | INTERACT 45, onion 23 | INTERACT 15, onion 4, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s13_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 150 | 0.17 | 0.33 | 3 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s14_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 50 | 0.00 | 0.27 | 1 | INTERACT 45, onion 23 | INTERACT 15, onion 4, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s16_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 200 | 0.12 | 0.33 | 4 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s17_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 150 | 0.00 | 0.30 | 2 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1 |
| greedy_n2_s18_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 150 | 0.00 | 0.40 | 2 | INTERACT 50, onion 25 | INTERACT 13, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1 |
| greedy_n2_s19_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 50 | 0.00 | 0.25 | 1 | INTERACT 45, onion 23 | INTERACT 16, onion 4, dish 1, onion in pot 4, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s1_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 100 | 0.00 | 0.35 | 1 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s3_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 100 | 0.00 | 0.15 | 1 | INTERACT 50, onion 25 | INTERACT 13, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1 |
| greedy_n2_s4_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 200 | -0.17 | 0.40 | 3 | INTERACT 50, onion 25 | INTERACT 13, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1 |
| greedy_n2_s5_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 100 | 0.00 | 0.45 | 1 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s6_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 100 | -0.50 | 0.05 | 1 | INTERACT 50, onion 25 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s7_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 50 | 0.00 | 0.30 | 1 | INTERACT 45, onion 23 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s8_ckpt-ppo_pickup_greedy_s0 | ckpt:ppo_pickup_greedy_s0 | 50 | 0.00 | 0.35 | 1 | INTERACT 45, onion 23 | INTERACT 15, onion 4, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s11_fidget | fidget | 150 | 0.00 | 0.15 | 2 | INTERACT 20, onion 4 | INTERACT 8, onion 2, dish 1, onion in pot 3, start cook 1, scoop soup 1 |
| greedy_n2_s12_fidget | fidget | 50 | 0.00 | 0.17 | 1 | INTERACT 19, onion 3, dish 1 | INTERACT 10, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s13_fidget | fidget | 400 | -3.00 | 0.23 | 1 | INTERACT 28, scoop soup 1, delivered 1 | INTERACT 20, onion 2, dish 1, onion in pot 3, start cook 1 |
| greedy_n2_s14_fidget | fidget | 50 | 0.00 | 0.47 | 1 | INTERACT 29, onion 4 | INTERACT 12, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s16_fidget | fidget | 50 | 0.00 | 0.25 | 1 | INTERACT 29, onion 2 | INTERACT 18, onion 4, dish 1, onion in pot 3, start cook 1, scoop soup 1, delivered 1 |
| greedy_n2_s19_fidget | fidget | 100 | -0.50 | 0.20 | 1 | INTERACT 26, onion 5 | INTERACT 13, onion 3, dish 1, onion in pot 3, start cook 1, delivered 1 |
| greedy_n2_s1_fidget | fidget | 100 | 0.00 | 0.15 | 1 | INTERACT 20, onion 2, dish 2 | INTERACT 16, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 2, delivered 1 |
| greedy_n2_s2_fidget | fidget | 50 | 0.00 | 0.25 | 0 | INTERACT 22, onion 3, dish 3 | INTERACT 14, onion 3, dish 1, onion in pot 3, start cook 1, scoop soup 1 |
| greedy_n2_s3_fidget | fidget | 150 | -0.50 | 0.10 | 1 | INTERACT 35, dish 4 | INTERACT 19 |
| greedy_n2_s4_fidget | fidget | 100 | -0.50 | 0.00 | 1 | INTERACT 23, onion 3, start cook 1 | INTERACT 22, onion 2, onion in pot 1, delivered 1 |
| greedy_n2_s6_fidget | fidget | 100 | -0.50 | 0.00 | 1 | INTERACT 24, dish 3 | INTERACT 19, onion 4, onion in pot 3, start cook 1, delivered 1 |
| greedy_n2_s7_fidget | fidget | 150 | -1.00 | 0.20 | 1 | INTERACT 25, onion 2, onion in pot 1, start cook 1 | INTERACT 14, onion 1, delivered 1 |
| greedy_n2_s8_fidget | fidget | 350 | 0.00 | 0.25 | 0 | INTERACT 28, dish 1, scoop soup 1 | INTERACT 15 |
| greedy_n2_s9_fidget | fidget | 200 | -0.25 | 0.00 | 2 | INTERACT 24, dish 1 | INTERACT 17, onion 1 |
| greedy_n2_s10_greedy | greedy | 50 | 0.50 | 0.80 | 1 | INTERACT 11, onion 3, dish 1, onion in pot 2, start cook 1, scoop soup 1, delivered 1 | INTERACT 17, onion 3, onion in pot 3 |
| greedy_n2_s11_greedy | greedy | 50 | 0.50 | 0.67 | 1 | INTERACT 13, onion 2, dish 1, onion in pot 2, start cook 1, scoop soup 1, delivered 1 | INTERACT 10, onion 2, onion in pot 2 |
| greedy_n2_s12_greedy | greedy | 50 | 0.50 | 0.80 | 1 | INTERACT 11, onion 1, dish 1, onion in pot 1, start cook 1, scoop soup 1, delivered 1 | INTERACT 16, onion 3, onion in pot 3 |
| greedy_n2_s13_greedy | greedy | 50 | 0.00 | 0.25 | 0 | INTERACT 16, onion 2, onion in pot 1 | INTERACT 9, onion 2, dish 1, onion in pot 2, start cook 1, scoop soup 1 |
| greedy_n2_s14_greedy | greedy | 50 | 0.50 | 0.67 | 1 | INTERACT 11, onion 2, dish 1, onion in pot 1, start cook 1, scoop soup 1, delivered 1 | INTERACT 18, onion 4, onion in pot 3 |
| greedy_n2_s16_greedy | greedy | 100 | 0.50 | 0.63 | 2 | INTERACT 15, onion 3, dish 1, onion in pot 2, start cook 1, scoop soup 1, delivered 1 | INTERACT 20, onion 2, dish 1, onion in pot 3, start cook 1 |
| greedy_n2_s17_greedy | greedy | 50 | 0.00 | 0.20 | 0 | INTERACT 7, onion 1, dish 1, onion in pot 1, start cook 1 | INTERACT 14, onion 3, onion in pot 2 |
| greedy_n2_s18_greedy | greedy | 350 | 0.50 | 0.62 | 8 | INTERACT 12, onion 1, dish 1, onion in pot 2, start cook 2, scoop soup 1, delivered 1 | INTERACT 19, onion 3, onion in pot 2 |
| greedy_n2_s1_greedy | greedy | 150 | 0.50 | 0.65 | 3 | INTERACT 11, onion 1, dish 2, onion in pot 1, start cook 1, scoop soup 1, delivered 1 | INTERACT 25, onion 2, onion in pot 2 |
| greedy_n2_s2_greedy | greedy | 150 | 0.50 | 0.67 | 3 | INTERACT 20, onion 2, onion in pot 2 | INTERACT 12, onion 1, dish 1, onion in pot 1, start cook 1, scoop soup 2, delivered 1 |
| greedy_n2_s3_greedy | greedy | 50 | 0.50 | 0.80 | 1 | INTERACT 13, onion 3, dish 1, onion in pot 2, start cook 1, scoop soup 1, delivered 1 | INTERACT 16, onion 3, onion in pot 2 |
| greedy_n2_s4_greedy | greedy | 50 | 0.50 | 0.80 | 1 | INTERACT 9, onion 1, dish 1, onion in pot 1, start cook 1, scoop soup 1, delivered 1 | INTERACT 21, onion 3, onion in pot 2 |
| greedy_n2_s8_greedy | greedy | 150 | 0.50 | 0.67 | 3 | INTERACT 11, onion 2, dish 1, onion in pot 1, scoop soup 1, delivered 1 | INTERACT 9, onion 1, onion in pot 2, start cook 1 |
