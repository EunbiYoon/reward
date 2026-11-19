# Single-episode judge: results/judge/20261118_185738

One episode per prompt; o = judge credit for agent 0, s = Shapley share, gap = o - s. fooled = share of episodes with gap > 0.1. own = own_share (soups agent 0 delivered).

## openai_basic  (200 episodes)

r(o, s) = 0.26,  r(o, own) = 0.64

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.62 | 0.50 | +0.12 | 0.55 | 0.63 | 9.2 |
| pickup | 60 | 0.27 | -0.04 | +0.31 | 0.55 | 0.00 | 7.2 |
| shadow | 60 | 0.05 | 0.13 | -0.08 | 0.00 | 0.00 | 9.6 |
| theft | 60 | 0.46 | 0.17 | +0.29 | 1.00 | 1.00 | 11.0 |

## openai_definition  (200 episodes)

r(o, s) = 0.38,  r(o, own) = 0.84

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.62 | 0.50 | +0.12 | 0.55 | 0.63 | 9.2 |
| pickup | 60 | 0.09 | -0.04 | +0.13 | 0.27 | 0.00 | 7.2 |
| shadow | 60 | 0.00 | 0.13 | -0.12 | 0.00 | 0.00 | 9.6 |
| theft | 60 | 0.48 | 0.17 | +0.31 | 0.97 | 1.00 | 11.0 |

## openai_solo  (200 episodes)

r(o, s) = 0.45,  r(o, own) = 0.27

| persona | n | o | s | gap | fooled | own | soups |
|---|---|---|---|---|---|---|---|
| greedy | 20 | 0.50 | 0.50 | +0.00 | 0.00 | 0.63 | 9.2 |
| pickup | 60 | 0.10 | -0.04 | +0.14 | 0.20 | 0.00 | 7.2 |
| shadow | 60 | 0.15 | 0.13 | +0.02 | 0.05 | 0.00 | 9.6 |
| theft | 60 | 0.17 | 0.17 | -0.00 | 0.00 | 1.00 | 11.0 |
