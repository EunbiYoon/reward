# Single-episode judge: results/judge/20261119_022655

One episode per prompt; o = judge credit for agent 0, s = Shapley share, gap = o - s. fooled = share of episodes with gap > 0.1. own = own_share (soups agent 0 delivered).

## local_basic  (200 episodes)

r(o, s) = 0.50,  r(o, own) = 0.96

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.62 | 0.50 | +0.12 | 0.50 | 0.63 | 9.2 |
| pickup | 60 | 0.06 | -0.00 | +0.06 | 0.25 | 0.00 | 7.2 |
| shadow | 60 | 0.05 | 0.09 | -0.04 | 0.02 | 0.00 | 8.7 |
| theft | 60 | 0.83 | 0.15 | +0.68 | 0.95 | 0.98 | 10.2 |

## local_definition  (200 episodes)

r(o, s) = 0.51,  r(o, own) = 0.93

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.62 | 0.50 | +0.12 | 0.50 | 0.63 | 9.2 |
| pickup | 60 | 0.07 | -0.00 | +0.08 | 0.27 | 0.00 | 7.2 |
| shadow | 60 | 0.05 | 0.09 | -0.04 | 0.02 | 0.00 | 8.7 |
| theft | 60 | 0.77 | 0.15 | +0.62 | 0.97 | 0.98 | 10.2 |

## local_numbers  (200 episodes)

r(o, s) = 0.61,  r(o, own) = 0.73

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.61 | 0.50 | +0.11 | 0.45 | 0.63 | 9.2 |
| pickup | 60 | 0.05 | -0.00 | +0.05 | 0.25 | 0.00 | 7.2 |
| shadow | 60 | 0.04 | 0.09 | -0.05 | 0.02 | 0.00 | 8.7 |
| theft | 60 | 0.41 | 0.15 | +0.27 | 0.63 | 0.98 | 10.2 |

## local_score  (200 episodes)

r(o, s) = -0.01,  r(o, own) = -0.06

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.50 | 0.50 | +0.00 | 0.00 | 0.63 | 9.2 |
| pickup | 60 | 0.50 | -0.00 | +0.50 | 1.00 | 0.00 | 7.2 |
| shadow | 60 | 0.50 | 0.09 | +0.41 | 1.00 | 0.00 | 8.7 |
| theft | 60 | 0.50 | 0.15 | +0.35 | 1.00 | 0.98 | 10.2 |

## local_solo  (200 episodes)

r(o, s) = 0.74,  r(o, own) = 0.63

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.58 | 0.50 | +0.08 | 0.25 | 0.63 | 9.2 |
| pickup | 60 | 0.03 | -0.00 | +0.03 | 0.15 | 0.00 | 7.2 |
| shadow | 60 | 0.04 | 0.09 | -0.04 | 0.02 | 0.00 | 8.7 |
| theft | 60 | 0.27 | 0.15 | +0.12 | 0.50 | 0.98 | 10.2 |

## local_without  (200 episodes)

r(o, s) = 0.55,  r(o, own) = 0.90

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.61 | 0.50 | +0.11 | 0.55 | 0.63 | 9.2 |
| pickup | 60 | 0.05 | -0.00 | +0.05 | 0.27 | 0.00 | 7.2 |
| shadow | 60 | 0.05 | 0.09 | -0.03 | 0.02 | 0.00 | 8.7 |
| theft | 60 | 0.65 | 0.15 | +0.51 | 0.98 | 0.98 | 10.2 |
