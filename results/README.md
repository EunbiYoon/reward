# results/

Every run folder made by `eval/run.py` has the same files:

| file | what it is |
|---|---|
| `config.json` | the settings of the grid |
| `results.csv` | one row per run: team soups, own soups, Shapley φ and share s, credit gap, pickups, … |
| `reference.csv` | the scripted cook in seat 0 (the "good teammate" baseline: s\*, gap\*) |
| `summary.md` | tables and verdicts (`python -m eval.summarize <folder>`) |
| `models/` | checkpoints, one per run: `<algo>_<reward>_<partner>_s<seed>.pt` (not tracked by git) |

## Used (paper or 690S report)

| folder | what it is | used for |
|---|---|---|
| `full/` | Overcooked, 36 runs: PPO/GRPO × pickup/team/individual × stay/greedy × seeds 0–2, 2M steps | paper Table 1 and Figure 1; report 2(c) |
| `h10_cf/` | Overcooked, cf (difference) reward, 12 runs, seeds 0–2 | paper Figure 1 (cf) and Appendix C; report Extensions |
| `lbf_full/` | Level-Based Foraging, same 36-run grid | paper Appendix B; oversight |
| `rware_full/` | Multi-Robot Warehouse, same 36-run grid | paper Appendix B; oversight |
| `rlaif/theft/` | **LLM-judge study**: a pool of 100 episodes (20 seeds × 5 seat-0 agents) and every judgment | paper Table 2 and §5 |

`rlaif/theft/` in detail:

| file | what it is |
|---|---|
| `pool/` | the episodes; seat 0 is the scripted cook, a blocker (fidget), or one of three checkpoints from `full/models/`: the **thief** `ppo_individual_greedy_s0`, the **free-rider** `grpo_team_greedy_s0`, the **fake helper** `ppo_pickup_greedy_s0` |
| `judgments_shapley.jsonl` | the oracle (true answer), 200 pairs |
| `judgments_ollama.jsonl` | Qwen3.8-27B, basic prompt, 116 pairs |
| `judgments_openai.jsonl` | GPT-4o, basic prompt, 63 pairs |
| `judgments_ollama_definition.jsonl`, `judgments_ollama_solo.jsonl` | Qwen with the Shapley definition / with definition + solo results, 6–7 pairs so far |
| `audit.md`, `audit.json` | judge vs Shapley (`python -m rlaif.audit --run results/rlaif/theft`) |

### Checkpoints that matter
- `full/models/ppo_individual_greedy_s0.pt`, `grpo_team_greedy_s0.pt`, `ppo_pickup_greedy_s0.pt`: **in use**, they
  are the thief, free-rider and fake helper the judges evaluate. Do not delete; the pool was recorded from them.
- Every other checkpoint (`full/`, `h10_cf/`, `lbf_full/`, `rware_full/`) is only needed to make GIFs
  (`eval/visualize.py`) or to re-evaluate a run; the numbers in `results.csv` do not need them.

## In progress

| folder | what it is | status |
|---|---|---|
| `full_s34/` | extra seeds 3–4 for the `full/` grid | 15 of 24 runs; `scripts/paper_experiments.sh` finishes it and merges into `paper/full5` |
| `paper/` | merged 5-seed results, oversight report | empty until `scripts/paper_experiments.sh` runs |

