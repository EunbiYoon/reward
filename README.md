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

On Unity use the conda env instead: `CONDA_ENV=credit source scripts/cluster/activate.sh` (the run scripts do
this themselves; a different `CONDA_ENV` in your shell, e.g. `marshal`, makes them fail with "Needs the
official package").

## 1. Train

Seat 0 learns next to the scripted `greedy` cook (map `cramped_room`). One run per reward and seed; `--eval`
evaluates right after training.

| persona | reward | paid for |
|---|---|---|
| honest cook (A) | `team` | team deliveries |
| credit theft | `theft` | only the soups it delivers itself |
| fake help | `pickup` | team deliveries + 3 per onion or dish picked up |
| shadowing | `shadow` | team deliveries + 1 per step next to the partner |

```bash
venv/bin/python -m train.run --algos ppo --rewards theft pickup shadow --partners greedy --seeds 0 1 2 --out results/train/ppo/personas --eval
venv/bin/python -m train.run --algos grpo --rewards theft pickup shadow --partners greedy --seeds 0 1 2 --out results/train/grpo/personas --eval
```

Each run folder gets `models/*.pt`, `curves/*.csv` and `train_log.csv`. The judge uses the checkpoints copied
to `results/ckpts/<algo>/<algo>_<persona>_greedy_s<seed>.pt`.

## 2. Evaluate

```bash
venv/bin/python -m eval.run results/train/ppo/personas          # skip if trained with --eval
venv/bin/python -m eval.summarize results/train/ppo/personas
venv/bin/python scripts/plot_curves.py results/train/ppo/personas
```

`eval.run` writes `<run>/eval/<stamp>/results.csv` (team soups, own deliveries, Shapley share `s`, credit gap,
`persona_match`) and `persona_check.md`.

## 3. Judge

Does a supervisor give seat 0 the credit it really earned (`s`, its Shapley share)? Every episode is judged on
its own: seat 0 = a persona or greedy, seat 1 = greedy, random start positions.

**3a. Pools (CPU).** One folder per persona and algo, named by time; 3 training seeds x 20 start states each,
plus a greedy pool (the reference team). Repeat with `grpo`.

```bash
for P in theft pickup shadow; do venv/bin/python -m judge.collect --partners greedy --behaviors --policies "results/ckpts/ppo/ppo_${P}_greedy_s*.pt" --episodes 20 --random-start --workers 8 --out results/judge/$(date +%Y%m%d_%H%M%S); sleep 1; done
venv/bin/python -m judge.collect --partners greedy --behaviors greedy --episodes 20 --random-start --workers 8 --out results/judge/$(date +%Y%m%d_%H%M%S)
```

`pool/meta.jsonl`: one line per episode (behaviour, seed, soups, `v_partner_alone`, `v_learner_alone`, `phi`,
`s`, `own_share`, the event log).

**3b. Team-score supervisor (no LLM).** Flags a team that scores below 0.8 x the greedy team:

```bash
venv/bin/python scripts/team_score_monitor.py results/judge/<theft> results/judge/<pickup> results/judge/<shadow> results/judge/<greedy>
```

**3c. LLM judge.** Pass the four pools of one algo in one quoted argument; each call writes a new
`results/judge/<stamp>/` with `single_<judge>_<prompt>.jsonl` and `single_report.md`.

```bash
# Qwen3.8-27B on vLLM, inside a GPU allocation
CONDA_ENV=credit bash scripts/run_judge_single.sh "results/judge/<theft> results/judge/<pickup> results/judge/<shadow> results/judge/<greedy>" "score basic definition without numbers solo"
# Gemma 3 27B (another model family), same GPU setup
CONDA_ENV=credit MODEL=RedHatAI/gemma-3-27b-it-FP8-dynamic bash scripts/run_judge_single.sh "<4 pools>" "score basic definition without numbers solo"
# GPT / Gemini over the API (no GPU; keys in .env: OPENAI_API_KEY=..., GEMINI_API_KEY=...)
CONDA_ENV=credit REASONING=medium JUDGE=openai MODEL=gpt-5-mini bash scripts/run_judge_api.sh "<4 pools>" "score basic definition without numbers solo"
CONDA_ENV=credit JUDGE=gemini LIMIT=40 bash scripts/run_judge_api.sh "<4 pools>" "score basic solo"
```

Smoke test first with `LIMIT=4`; rerun with `STAMP=<stamp>` to resume a stopped run.

| `--prompt` | the judge sees |
|---|---|
| `score` | the team score only (no log) |
| `basic` | the game rules and every step of the episode |
| `definition` | + the definition of the Shapley value |
| `without` | + v(P): what the teammate delivers alone |
| `numbers` | + v(P) and v(L), no definition |
| `solo` | + the definition, v(P) and v(L) |

**3d. Report and paper tables.**

```bash
venv/bin/python -m judge.judge_single --out results/judge/<judge stamp> --report
venv/bin/python scripts/paper_results_tex.py          # docs/paper/figures/tab_*.tex, fig_judge.tex
```

The earlier pairwise judge (two episodes side by side, appendix of the paper) is `judge.judge_llm` +
`judge.judge_shapley` + `judge.audit`; `judge.reward_model` trains a reward from its verdicts.

## Code

- `environment/kitchen.py`: the kitchen, its rewards (`team`, `theft`, `pickup`, `shadow`, `cf`, `prep`),
  partners (`greedy`, `stay`, `self`, `ckpt:<path>`) and random starts. `warehouse.py`, `foraging.py`,
  `alem.py`: other games behind the same interface.
- `train/`: `run.py` (the training grid), `ppo.py`, `grpo.py`, `config.py` (every hyperparameter),
  `deep_policy.py`, `resume.py`.
- `eval/`: `run.py`, `coalitions.py` (Shapley share from the full game and the two frozen-cook games),
  `persona.py`, `summarize.py`, `visualize.py`, `paths.py`.
- `judge/`:

  | file | does |
  |---|---|
  | `collect.py` | plays the pool, computes `phi`, `s`, `own_share` |
  | `behaviors.py` | scripted seat-0 behaviours (greedy, ...) |
  | `describe_llm.py` | turns an episode into the log the judge reads |
  | `prompt.py` | game rules, instructions, the prompt variants |
  | `judge_single.py` | single-episode judge: vLLM, Ollama, Gemini, OpenAI; `--report` |
  | `judge_llm.py`, `judge_shapley.py`, `common.py`, `audit.py` | pairwise judge (appendix) |
  | `reward_model.py`, `learned_env.py` | reward model from judge verdicts (`train.run --rewards learned`) |

- `scripts/`: run scripts for the judge, `team_score_monitor.py`, `check_random_start.py`, paper figures and
  tables, `cluster/` (Slurm).
- `docs/paper/`: the paper.
- `results/`: `train/` (runs), `ckpts/` (checkpoints for the pools), `judge/<stamp>/` (pools and judgments).
