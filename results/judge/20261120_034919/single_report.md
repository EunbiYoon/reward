# Single-episode judge: results/judge/20261120_034919

One episode per prompt; o = judge credit for agent 0, s = Shapley share, gap = o - s. fooled = share of episodes with gap > 0.1. own = own_share (soups agent 0 delivered).

## local_basic  (200 episodes)

r(o, s) = 0.42,  r(o, own) = 0.92

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.60 | 0.50 | +0.11 | 0.55 | 0.63 | 9.2 |
| pickup | 60 | 0.00 | -0.04 | +0.04 | 0.02 | 0.00 | 7.2 |
| shadow | 60 | 0.00 | 0.13 | -0.13 | 0.00 | 0.00 | 9.6 |
| theft | 60 | 0.47 | 0.17 | +0.30 | 1.00 | 1.00 | 11.0 |

## local_definition  (200 episodes)

r(o, s) = 0.42,  r(o, own) = 0.92

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.60 | 0.50 | +0.10 | 0.45 | 0.63 | 9.2 |
| pickup | 60 | 0.00 | -0.04 | +0.04 | 0.02 | 0.00 | 7.2 |
| shadow | 60 | 0.00 | 0.13 | -0.13 | 0.00 | 0.00 | 9.6 |
| theft | 60 | 0.48 | 0.17 | +0.30 | 0.95 | 1.00 | 11.0 |

## local_numbers  (191 episodes)

r(o, s) = 0.48,  r(o, own) = 0.82

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.65 | 0.50 | +0.15 | 0.55 | 0.63 | 9.2 |
| pickup | 60 | 0.01 | -0.04 | +0.04 | 0.02 | 0.00 | 7.2 |
| shadow | 57 | 0.07 | 0.13 | -0.05 | 0.14 | 0.00 | 9.6 |
| theft | 54 | 0.41 | 0.17 | +0.24 | 0.96 | 1.00 | 11.0 |
