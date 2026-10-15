# Findings

All numbers: normalized true-return gap / reward-hacking rate, mean over seeds.
Main sweep: `results/full.csv` (5 seeds × budgets {1,2,5,10,20} × 3 maps).
Ablations: `results/ablation.csv` (8 seeds × budgets {2,5,20} × 3 maps).

## 1. Point-estimate reward models get hacked, and more demonstrations do not fix it

Averaged over all maps and budgets (main sweep):

| method | gap ↓ | hacking ↓ | lava occ. ↓ | coin occ. |
|---|---|---|---|---|
| oracle (true reward, same actor–critic) | 0.07 | – | 0.000 | 0.18 |
| bc / dagger | 0.35 / 0.32 | – | 0.033 / 0.031 | 0.02 |
| apprenticeship | 1.71 | 0.80 | 0.102 | 0.18 |
| maxent | 1.96 | 0.83 | 0.137 | 0.20 |
| gail (state-only) | 0.52 | 0.99 | 0.024 | 0.57 |
| trex / brex | 1.45 / 2.35 | 0.73 / 0.67 | 0.09 / 0.16 | 0.34 / 0.29 |
| birl_mean | 1.17 | 0.60 | 0.087 | 0.03 |
| **birl_varpen** (contrast std, λ=1) | **0.05** | **0.00** | **0.001** | 0.00 |
| birl_active (+10 queries, λ=1) | 0.08 | 0.08 | 0.001 | 0.07 |

A gap above 1 means worse than a uniform-random policy: the agent walks through lava repeatedly.
`maxent` on `lava_shortcut` still has a gap of 2.4 with 20 demonstrations, and `birl_mean`
still hacks in 60% of runs on the two lava maps at n = 20.

## 2. Naive posterior-variance penalties do not target the misspecified feature

Linear Bayesian IRL is identified only up to scale (Boltzmann temperature vs. reward
magnitude) and, given a zero-reward terminal sink, partly up to shift (a large step cost
explains "hurrying to the goal" as well as a large goal reward does). Two consequences:

* The raw posterior is *confidently wrong* about the trade-off that matters. On
  `lava_shortcut`, 96–100% of raw posterior samples have a lava-crossing optimal policy,
  because the inferred step cost is large relative to the goal (e.g. posterior mean
  empty/goal = −0.95/0.32 at n = 2, vs. −0.02/1 true).
* Raw per-feature std is nearly uniform across features (0.45–0.83 at n = 2), so
  `E[r] − λ·Std[r]` acts like an extra *uniform step cost*, which makes shortcuts through
  the hazard *more* attractive, not less.

**Fix: penalize uncertainty in an identifiable parameterization.** Map each sample to
`w' = (w − w_empty) / (w_goal − w_empty)`, so an empty cell is worth 0 and the goal 1
(`Posterior.contrast`). The remaining coordinates (lava, coin) are exactly the trade-offs
the demonstrations did not pin down, and their std becomes large where it should be.

λ sweep (ablation, all maps pooled):

| λ | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| raw std — gap / hacking | 0.31 / 0.64 | 0.28 / 0.54 | 0.27 / 0.56 | 0.25 / 0.57 |
| contrast std — gap / hacking | 0.06 / 0.03 | 0.07 / 0.03 | 0.07 / 0.01 | 0.05 / 0.01 |

The contrast penalty is robust across a 16× range of λ. The contrast posterior *mean*
alone (λ = 0) is poor (gap 1.49, hacking 0.49): the ratio distribution is heavy-tailed,
so pessimism is what makes it usable.

**Assumption to state in a paper:** the designer specifies one "neutral" feature and one
"task" feature as anchors. This is weaker than knowing the reward, but it is prior knowledge.

## 3. Preference queries add little once the penalty is in place

Equal budget of 10 labels on the deployment map, λ = 1 (pooled over maps):

| strategy | gap | hacking |
|---|---|---|
| no queries | 0.07 | 0.03 |
| random pairs | 0.06 | 0.06 |
| pairs through high-uncertainty states | 0.07 | 0.03 |
| max posterior disagreement | 0.09 | 0.11 |
| ours (disagreement × uncertainty-state bonus) | 0.10 | 0.11 |
| ours, deploy posterior mean (λ = 0) | 0.19 | 0.38 |

* Queries do help an *unpenalized* posterior (λ = 0: hacking 0.49 → 0.38).
* With the penalty, no strategy beats "no queries"; disagreement-based selection is slightly
  worse on `coin_trap`. A plausible mechanism: answers shrink the coin posterior, which
  removes the penalty while the mean is still biased upward, and the agent returns to
  camping. This is worth checking directly before claiming anything about queries.

## 4. Automatic λ via α-VaR does not work yet

Choosing λ by the 95% VaR of expected value difference over the contrast posterior picks
λ = 0 on most runs (gap 1.42, hacking 0.42). Greedy policies under different λ often
coincide, and the heavy-tailed contrast samples dominate the VaR. Candidate fixes are
CVaR of return instead of EVD, evaluating the VaR under the raw posterior, or a
BROIL-style policy optimization.

## Threats to validity

* One small tabular domain. The scale/shift confound is generic to linear IRL, but the
  size of the effect here depends on the terminal-sink design.
* The demonstrator is Boltzmann-rational with β = 100 while Bayesian IRL assumes β = 20
  (intentional mild misspecification). Matching β did not remove the confound in a spot check.
* The actor–critic adds noise: the oracle itself has a gap of 0.07 on average, higher on `combined`.
* The synthetic human uses the partial-return model the learner assumes. The regret-model
  ablation config is provided but has not been run.
