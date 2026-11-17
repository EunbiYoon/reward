# Who Takes the Credit?

One agent (seat 0) learns next to a fixed partner (seat 1) in a two-cook Overcooked kitchen. We measure its
real contribution with the Shapley value and compare it with the credit that its own reward, a team-score
overseer or an LLM judge gives it.

Every command runs from the project root. On Windows replace `venv/bin/python` with `.\venv\Scripts\python`.

## 0. Setup

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

LLM judging also needs [Ollama](https://ollama.com) with a Qwen model pulled
(`qwen3.8:27b` on Windows, `qwen3.8:27b-mlx` on Mac). Check with `ollama list`.

## 1. Train

Personas (seat 0 next to the scripted `greedy` cook, map `cramped_room`):

| persona | reward | what it is paid for |
|---|---|---|
| A (honest cook) | `team` | team deliveries |
| credit theft | `individual` | only the soups it delivers itself |
| fake help | `pickup` | team deliveries + 3 per onion or dish picked up |
| shadowing | `shadow` | team deliveries + 1 per step next to the partner |

One grid (PPO or GRPO, seeds 0-2, evaluated right after training, learning curves drawn):

```bash
venv/bin/python -m train.run --algos ppo --rewards individual pickup shadow --partners greedy --seeds 0 1 2 --out results/ppo/x --eval
```

The overnight pipeline (`results/<algo>/need_curve/`, one run at a time, keeps the PC awake):

```bash
venv/bin/python scripts/need_curve.py --list
venv/bin/python scripts/need_curve.py
venv/bin/python scripts/need_curve.py --steps ppo_theft ppo_pickup ppo_shadow
```

Default order: GRPO greedy personas -> GRPO A on the four other maps -> PPO theft, pickup, shadow.
Logs: `results/need_curve_logs/pipeline.log`. `--workers 2` runs two seeds at once, `--smoke` checks the flow
in minutes.

Each run folder gets `models/*.pt`, `curves/*.csv`, `curves/curves.png` and `train_log.csv`.

## 2. Evaluate

```bash
venv/bin/python -m eval.run results/ppo/need_curve/theft
venv/bin/python -m eval.summarize results/ppo/need_curve/theft
venv/bin/python scripts/plot_curves.py results/ppo/need_curve/theft
```

`eval.run` writes `<run>/eval/<YYYYmmdd_HHMMSS>/results.csv` (team soups, own deliveries, Shapley share `s`,
credit gap, `persona_match`) and `persona_check.md`. Skip it if the run was trained with `--eval`.

## 3. LLM judge vs Shapley ground truth

Two pools of episodes, built from the need_curve checkpoints (seed 0). Seat 1 is always `greedy`; `greedy`
and `fidget` (a blocker) in seat 0 are the baselines.

| pool | seat 0 | pairs |
|---|---|---|
| `results/rlaif/need_curve_ppo` | greedy, fidget, PPO theft, pickup, shadow | 20 seeds x 10 = 200 |
| `results/rlaif/need_curve_grpo` | greedy, fidget, GRPO theft, pickup, shadow, team (A) | 20 seeds x 15 = 300 |

A pair is two episodes from the same seed and the same partner that differ only in seat 0.

### Stage 1: ground truth (pool + Shapley)

```bash
venv/bin/python -m rlaif.collect --partners greedy --behaviors greedy fidget --policies "results/ppo/need_curve/*/models/*_s0.pt" --episodes 20 --workers 4 --out results/rlaif/need_curve_ppo
venv/bin/python -m rlaif.judge_shapley --run results/rlaif/need_curve_ppo
```
```bash
venv/bin/python -m rlaif.collect --partners greedy --behaviors greedy fidget --policies "results/grpo/need_curve/greedy_personas/models/*_s0.pt" "results/grpo/need_curve/A/models/*_s0.pt" --episodes 20 --workers 4 --out results/rlaif/need_curve_grpo
venv/bin/python -m rlaif.judge_shapley --run results/rlaif/need_curve_grpo
```

Output: `pool/meta.jsonl` (one line per episode: behaviour, seed, team soups, own deliveries, `v_partner_alone`,
`v_learner_alone`, `phi`, `s`, `own_share`, the event log) and `judgments_shapley.jsonl`.

### Stage 2: LLM judge (Qwen 27B)

```bash
venv/bin/python -m rlaif.judge_llm --run results/rlaif/need_curve_ppo --judge ollama --model qwen3.8:27b-mlx --prompt basic --workers 1
venv/bin/python -m rlaif.judge_llm --run results/rlaif/need_curve_grpo --judge ollama --model qwen3.8:27b-mlx --prompt basic --workers 1
```

On Windows the model is `qwen3.8:27b`. About 1 minute per pair on an RTX 3060, about 3 hours for the PPO pool
and 5 for GRPO. The run is resumable: rerun the same command and it judges only the missing pairs. `--pairs 50`
judges a subset first.

| `--prompt` | the judge also sees | output |
|---|---|---|
| `basic` | the game rules and the event log only | `judgments_ollama.jsonl` |
| `definition` | + the definition of the Shapley value | `judgments_ollama_definition.jsonl` |
| `solo` | + the definition and the two counterfactual results v(P), v(L) | `judgments_ollama_solo.jsonl` |

For `definition` and `solo` the theft pairs are enough: add `--focus individual`.
A second judge on the same pools: `--judge openai` (GPT-4o, `OPENAI_API_KEY`) or `--judge claude`
(`ANTHROPIC_API_KEY`).

### Stage 3: compare

```bash
venv/bin/python -m rlaif.audit --run results/rlaif/need_curve_ppo
venv/bin/python -m rlaif.audit --run results/rlaif/need_curve_grpo
```

Output: `audit.md` and `audit.json` per pool. For every judge, the agreement of its preferred agent with the
larger Shapley value, how often it prefers the thief or the busier-looking agent that contributed less, and
per persona its mean credit next to `s` (real share) and `own_share` (share of soups delivered).

### Two computers

A pool is written by one computer only, so there are no git conflicts: for example Windows judges
`need_curve_ppo` and the Mac judges `need_curve_grpo`. When a pool is done:

```bash
git add results/rlaif/need_curve_ppo
git commit -m "judge need_curve_ppo"
git push
```

## Code

- `environment/kitchen.py`: the kitchen, its rewards (`team`, `individual`, `pickup`, `shadow`, `cf`, `prep`)
  and partners (`greedy`, `stay`, `self`, `ckpt:<path>`). Each step it logs pickups, onions into pots, pots
  started, soups scooped and deliveries for both cooks. `warehouse.py`, `foraging.py`, `alem.py`: other games
  behind the same interface.
- `train/`: `run.py` (the training grid), `ppo.py`, `grpo.py`, `config.py` (every hyperparameter),
  `deep_policy.py`, `resume.py`.
- `eval/`: `run.py` (measures every checkpoint of a run), `coalitions.py` (Shapley share from the full game and
  the two frozen-cook games), `persona.py`, `summarize.py`, `visualize.py`, `paths.py`.
- `rlaif/`:

  | file | side | does |
  |---|---|---|
  | `collect.py` | both | plays the pool, computes `phi`, `s`, `own_share` |
  | `common.py` | both | pairs, judgments files, resuming |
  | `judge_shapley.py` | ground truth | prefers the larger `phi`, credit = `s` |
  | `describe_llm.py` | LLM | turns an episode into the event log the judge reads |
  | `prompt.py` | LLM | game rules, instructions, the `definition` and `solo` additions |
  | `judge_llm.py` | LLM | calls Ollama, OpenAI, Claude or a local server |
  | `audit.py` | both | compares the judges with the ground truth |
  | `replay_events.py` | both | rebuilds the event log of an old pool from its recorded actions |
  | `reward_model.py`, `learned_env.py` | optional | train a reward from the judge's verdicts |

- `scripts/`: `need_curve.py` (overnight training), `plot_curves.py`, paper figures, `cluster/` (Slurm).
- `docs/paper/`: the paper draft.
- `results/`: outputs, `results/ppo/`, `results/grpo/`, `results/rlaif/`.
