# Baselines from the CS 690S (Fall 2026) reading list

Every reading on the [course schedule](https://people.cs.umass.edu/~sniekum/classes/AIA-F26/desc.php)
that this project builds on, what role it plays, and where it lives in the code.

## Implemented baselines

| Method (paper) | Course week | Role here | Code |
|---|---|---|---|
| **Behavior cloning** — Torabi, Warnell & Stone, *Behavioral Cloning from Observation* (IJCAI 2018) | 09/10 BC | Primary imitation baseline (feature-based softmax policy so it transfers to new layouts) | `imitation/bc.py` |
| **DAgger** — Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (AISTATS 2011) | 09/15 Limits of BC | BC + interactive expert relabeling (upper bound for imitation; uses extra supervision) | `imitation/dagger.py` |
| **One-Step Actor–Critic** — Sutton & Barto, *RL: An Introduction* §13.5 | 09/17 RL | The RL learner every reward model is handed to, trained from scratch | `agents/actor_critic.py` |
| **Apprenticeship learning (projection)** — Abbeel & Ng, *Apprenticeship Learning via IRL* (ICML 2004) | 09/24 IRL | Feature-expectation-matching reward | `irl/apprenticeship.py` |
| **MaxEnt IRL** — Ziebart et al., *Maximum Entropy IRL* (AAAI 2008) | 09/29 MaxEnt IRL | Point-estimate reward model (no uncertainty) | `irl/maxent.py` |
| **Bayesian IRL** — Ramachandran & Amir (IJCAI 2007) | 10/01 Bayesian IRL | Posterior over rewards; `birl_mean` uses the posterior mean | `irl/bayesian.py` |
| **α-VaR bounds** — Brown & Niekum, *Efficient Probabilistic Performance Bounds for IRL* (AAAI 2018) | 10/01 Bayesian IRL | High-confidence EVD bound reported for every posterior-based method | `metrics.py::var_evd` |
| **GAIL** — Ho & Ermon, *Generative Adversarial Imitation Learning* (NeurIPS 2016) | 10/06 Occupancy matching | Tabular, state-only adversarial occupancy matching; discriminator logit = reward | `irl/gail.py` |
| **T-REX** — Brown et al., *Extrapolating Beyond Suboptimal Demonstrations via IRL from Observations* (ICML 2019) | 10/15 RLHF (indirect) | Reward from ranked, noise-injected trajectories | `irl/rex.py::trex` |
| **Human preference models** — Knox et al., *Models of Human Preference for Learning Reward Functions* (2022) | 10/20 Human preference | Synthetic human: partial-return vs. regret Bradley–Terry (ablation `configs/ablation_regret_human.json`) | `preferences.py` |
| **B-REX** — Brown et al., *Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences* (ICML 2020) | 10/27 Guarantees for RLHF | Bayesian reward from preferences (no MDP solve) | `irl/rex.py::brex` |

## Readings that motivate the problem / evaluation (not run as baselines)

| Reading | Week | How it is used |
|---|---|---|
| Krakovna et al., *Specification gaming: the flip side of AI ingenuity* | 09/22 | Motivates the `coin_trap` env (camping on a proxy-rewarded tile, à la the boat-race example) |
| Shah et al., *Goal Misgeneralization: Why Correct Specifications Aren't Enough for Correct Goals* | 09/22 | Coin is perfectly correlated with the goal at train time and decorrelated at test time |
| Booth et al., *The Perils of Trial-and-Error Reward Design* (AAAI 2023) | 09/22 (opt.) | Related failure mode: rewards overfit to a training distribution |
| Zhuang & Hadfield-Menell, *Consequences of Misaligned AI* | 09/22 (opt.) | Unmodeled features (lava) get exploited when the proxy omits them |
| Ouyang et al. (InstructGPT); Casper et al., *Open Problems and Fundamental Limitations of RLHF* | 10/15 (opt.) | Framing: learned reward model → RL optimizer → over-optimization |
| Rafailov et al., *DPO*; Hejna et al., *CPL* | 10/22 | Possible extension: direct (RL-free) preference methods as a further baseline |
| Brown et al., *Value Alignment Verification* | 10/27 (opt.) | Possible extension: verifying the deployed policy with a few queries |
| Hadfield-Menell et al., *Cooperative IRL* | 11/05 | Active querying is a lightweight form of assistance-game information gathering |

## Not in the course list but useful related work

* Skalse et al., *Defining and Characterizing Reward Hacking* (NeurIPS 2022) — the
  `hacked` flag in `metrics.py` operationalizes their definition (proxy says "at least as
  good as the true optimum" while true return drops).
* Pan, Bhatia & Steinhardt, *The Effects of Reward Misspecification* (ICLR 2022) — proxy
  reward environments and phase transitions in hacking.
* Laidlaw, Singhal & Dragan (2024) — occupancy-measure regularization against reward
  hacking; motivates reporting occupancy divergence, and is a natural extra baseline.
* Sadigh et al. (2017), Bıyık et al. — active preference queries by maximizing
  disagreement / information gain.
