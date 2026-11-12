# Code structure

One learner (seat 0) is trained next to a fixed partner (seat 1). Its real contribution is measured
with the two-player Shapley value from frozen-member replays, and compared with the credit that a
reward, a team-score supervisor or an LLM judge gives it.

```
environment/  the games            train/  the learners           eval/  train + measure + report
     │                                  │                               │
     └──────────────► eval/run.py ◄─────┘  results/<run>/results.csv ──► eval/summarize.py, eval/oversight.py
                           │
                           └─ checkpoints ─► rlaif/collect.py ─► rlaif/judge.py ─► rlaif/audit.py
                                                (pool)          (LLM verdicts)    (judge vs Shapley)
                                                                      │
                                                     rlaif/reward_model.py ─► eval/run.py --rewards learned
```

## `environment/`: the games (one interface for all)
| file | what it is |
|---|---|
| `__init__.py` | `make(env, ...)`: `kitchen`, `warehouse`, `foraging`, `alem` |
| `kitchen.py` | Overcooked-AI two-cook kitchen. Rewards `team`, `individual`, `pickup`, `cf` (difference reward); partners `stay` / `greedy` (+ action `noise`); logs events (pickups, deliveries, interceptions) |
| `warehouse.py`, `foraging.py`, `alem.py` | RWARE, Level-Based Foraging and Alem behind the same interface |
| `two_seat.py` | loop shared by the warehouse and foraging wrappers |

## `train/`: the learner
| file | what it is |
|---|---|
| `ppo.py`, `grpo.py` | the two optimizers used in the paper (GRPO is critic-free, group of 8) |
| `a2c.py`, `dqn.py`, `sac.py` | extra optimizers (`--algos all`), not used in the paper |
| `deep_policy.py` | MLP policy, value net, obs normalization, checkpoints, optional partner-prediction head (`--aux-coef`) |
| `resume.py` | mid-training checkpoints so cluster jobs survive preemption |

## `eval/`: experiments and measurement
| file | what it is |
|---|---|
| `config.py` | every hyper-parameter (defaults = the proposal: 2M steps, 3 seeds, 10 eval episodes) |
| `run.py` | trains every (seed, algo, reward, partner[, bonus, noise, aux]) and measures it; writes `results.csv`, `reference.csv` (scripted cook in seat 0), `models/` |
| `coalitions.py` | **the measurement**: plays full team / learner alone / partner alone on fixed seeds, returns φ, share s, own share o, credit gap o − s |
| `summarize.py` | `summary.md`: tables with 95% t-intervals and the H1–H3 verdicts (fake help, spectator, credit theft) |
| `oversight.py` | team-score supervisor: how many hacked runs score ≥ 80% of the reference |
| `visualize.py` | GIF/MP4 of the cooks playing |

## `rlaif/`: LLM judges
| file | what it is |
|---|---|
| `behaviors.py` | seat-0 behaviours for the pool: scripted cook, ε-greedy, random, fidget (blocker), stay, or a trained checkpoint |
| `collect.py` | records a pool of episodes with Shapley ground truth (`pool/meta.jsonl`, `pool/trajs/`) |
| `describe.py` | turns an episode into the event-log text the judge reads |
| `judge.py` | judges matched pairs. `--judge` ollama / openai / claude / local (vLLM) / shapley (oracle) / activity (mock); `--prompt` basic / definition / solo; `--focus` filters pairs |
| `audit.py` | `audit.md`: agreement with Shapley, busy credit, theft credit, credit error per behaviour, reward-model and trained-agent audits |
| `reward_model.py`, `learned_env.py` | Bradley–Terry reward model on the verdicts, and the `learned` training reward that uses it |

## `scripts/`: paper pipeline
| file | what it is |
|---|---|
| `paper_experiments.sh` | every remaining paper experiment on the Mac (extra seeds, cf, judge prompts, sweeps), then reports and figures |
| `paper_figures.py` | `docs/paper/figures/*.pdf` from whatever results exist |
| `theft_analysis.py` | judge credit vs delivered share o vs leave-one-out vs Shapley, per judge (the paper's Table 2) |
| `merge_runs.py` | merges extra-seed runs into one folder (`results/paper/full5`) |

## `scripts/cluster/`: UMass Unity (Slurm)
`setup_conda.sh` (envs + judge model), `submit.sh` (preemptible < 1 h array jobs), `run_task.sh` (one run per task,
self-requeue), `judge_gpu.sh` (vLLM judge on 2×L40S), `reward_models.sh`, `finish.sh`, `activate.sh`.

## `docs/`: everything to read
| path | what it is |
|---|---|
| `docs/paper/` | WMAC 2027 short paper (`main.tex`, `who_takes_the_credit.pdf`, `figures/`, `README.md` checklist) |
| `docs/report/` | CS 690S check-in (`checkin.tex`, `checkin.pdf`) |
| `docs/page/` | project web page (GitHub Pages, `.github/workflows/pages.yml`) |
| `docs/videos/` | GIFs from `eval/visualize.py` |
| `docs/archive/` | superseded files (Windows runner, old local runner, Windows handoff) |

## Data (not code)
| path | what it is |
|---|---|
| `results/full`, `lbf_full`, `rware_full` | main 36-run grids (Overcooked, LBF, RWARE), 3 seeds |
| `results/h10_cf`, `full_s34` | cf reward; extra seeds 3-4 (partial) |
| `results/rlaif/theft` | judge pool with trained thief / free-rider / fake helper, and all judgments (`judgments_<judge>[_<prompt>].jsonl`) |
| `logs/` | run logs (not tracked) |

## Common commands
```bash
venv/bin/python -m eval.run --rewards team individual pickup --partners stay greedy --out results/x   # train + measure
venv/bin/python -m eval.summarize results/x                                                         # tables + verdicts
venv/bin/python -m rlaif.judge --run results/rlaif/theft --judge ollama --model qwen3.8:27b-mlx --pairs 30
venv/bin/python -m rlaif.audit --run results/rlaif/theft                                            # judge audit
venv/bin/python scripts/theft_analysis.py                                                           # Table 2 numbers
venv/bin/python scripts/paper_figures.py                                                            # figures
```
