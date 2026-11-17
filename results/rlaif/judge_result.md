# LLM judge by prompt (Qwen/Qwen3.8-27B-FP8, vLLM on Unity)

From `results/rlaif/need_curve_{ppo,grpo}/audit.md`. basic judges every pair; definition and solo judge only pairs
with an `individual` (thief) agent, so the agreement column is not comparable between basic and the other two.

- s: agreement with Shapley (which of the pair is better)
- o: over-credit, judge credit minus Shapley credit, averaged over trajectories (0 = calibrated)
- persona columns: judge credit for agent 0 of that persona (Shapley credit in brackets)

| pool | prompt | pairs | s | o | individual | pickup | shadow |
|---|---|---|---|---|---|---|---|
| PPO | basic | 100 | 0.79 | +0.35 | 0.57 (0.17) | 0.26 (-0.00) | 0.01 (0.14) |
| PPO | definition | 80 | 0.75 | +0.30 | 0.57 (0.17) | 0.10 (-0.00) | 0.00 (0.14) |
| PPO | solo | 80 | 0.82 | +0.25 | 0.40 (0.17) | 0.01 (-0.00) | 0.20 (0.14) |
| GRPO | basic | 100 | 0.83 | +0.34 | 0.60 (0.16) | 0.25 (-0.00) | 0.04 (0.09) |
| GRPO | definition | 99 | 0.82 | +0.26 | 0.59 (0.16) | 0.06 (0.01) | 0.05 (0.10) |
| GRPO | solo | 100 | 0.92 | +0.24 | 0.37 (0.16) | 0.09 (0.01) | 0.18 (0.10) |

## Busy and theft credit

How often the judge preferred the agent 0 that contributed less (Shapley) but looked busier (busy) or delivered more
soups itself (theft). 0 = never fooled, 0.5 = chance; 95% interval and number of decisive pairs in brackets.

| pool | prompt | busy credit | theft credit |
|---|---|---|---|
| PPO | basic | 0.24 [0.14, 0.36] (n=55) | 0.92 [0.67, 0.99] (n=13) |
| PPO | definition | 0.00 [0.00, 0.09] (n=39) | 0.95 [0.76, 0.99] (n=20) |
| PPO | solo | 0.00 [0.00, 0.09] (n=39) | 0.70 [0.48, 0.85] (n=20) |
| GRPO | basic | 0.13 [0.06, 0.24] (n=54) | 0.64 [0.39, 0.84] (n=14) |
| GRPO | definition | 0.00 [0.00, 0.07] (n=55) | 1.00 [0.82, 1.00] (n=18) |
| GRPO | solo | 0.00 [0.00, 0.07] (n=55) | 0.37 [0.19, 0.59] (n=19) |

## Findings

- basic: the judge credits the thief (`individual`) about 3.5x its Shapley value (0.57-0.60 vs 0.16-0.17) and
  prefers it over a partner who contributed more in 64-92% of decisive pairs. It misses the invisible contribution
  of `shadow` (0.01-0.04 vs 0.09-0.14). The same numbers as Ollama qwen3.8:27b on a Mac (PPO theft 0.92 both), so
  this is the model, not the quantization.
- definition: the Shapley definition removes the busy bias (0.00) and the pickup over-credit, but not the theft
  (0.95 / 1.00) and not the shadow under-credit.
- solo: the counterfactual numbers v(P), v(L) are what helps with theft: individual credit drops to 0.37-0.40,
  theft credit to 0.70 (PPO) / 0.37 (GRPO), and shadow is credited for the first time (0.18-0.20). Theft is still
  over-credited about 2.3x, so even with the counterfactuals the judge does not fully recover Shapley.
- One GRPO definition pair failed (the model's answer was not JSON), so that row has 99 pairs.
