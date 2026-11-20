# Who Takes the Credit?

One agent (seat 0) learns next to a fixed partner (seat 1) in a two-cook Overcooked kitchen. We measure its
real contribution with the Shapley value and compare it with the credit that its own reward, a team-score
overseer or an LLM judge gives it.

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
python train.run --algos ppo --rewards theft pickup shadow --partners greedy --seeds 0 1 2 --out results/train/ppo/A_persona/greedy_personas --eval
python train.run --algos grpo --rewards theft pickup shadow --partners greedy --seeds 0 1 2 --out results/train/grpo/A_persona/greedy_personas --eval
```

Each run folder gets `models/*.pt`, `curves/*.csv` and `train_log.csv`. The judge uses the checkpoints copied
to `results/ckpts/<algo>/<algo>_<persona>_greedy_s<seed>.pt`.

## 2. Evaluate

```bash
python eval.run results/train/ppo/A_persona/greedy_personas          # skip if trained with --eval
python eval.summarize results/train/ppo/A_persona/greedy_personas
venv/bin/python scripts/plot_curves.py results/train/ppo/A_persona/greedy_personas
```

`eval.run` writes `<run>/eval/<stamp>/results.csv` (team soups, own deliveries, Shapley share `s`, credit gap,
`persona_match`) and `persona_check.md`.

## 3. Judge

Does a supervisor give seat 0 the credit it really earned (`s`, its Shapley share)? Every episode is judged on
its own: seat 0 = a persona or greedy, seat 1 = greedy, random start positions.

**3a. Pools (CPU).** One folder per persona and algo, named by time; 3 training seeds x 20 start states each,
plus a greedy pool (the reference team). Repeat with `grpo`.

```bash
for P in theft pickup shadow; do python judge.collect --partners greedy --behaviors --policies "results/ckpts/ppo/ppo_${P}_greedy_s*.pt" --episodes 20 --random-start --workers 8 --out results/judge/$(date +%Y%m%d_%H%M%S); sleep 1; done
python judge.collect --partners greedy --behaviors greedy --episodes 20 --random-start --workers 8 --out results/judge/$(date +%Y%m%d_%H%M%S)
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

**3d. Report.**

```bash
python judge.judge_single --out results/judge/<judge stamp> --report
```

The earlier pairwise judge (two episodes side by side, appendix of the paper) is `judge.judge_llm` +
`judge.judge_shapley` + `judge.audit`; `judge.reward_model` trains a reward from its verdicts.

## 4. Paper tables and figures

Every table and figure of `docs/paper/main.tex` is TeX. The generated ones are written into
`docs/paper/figures/*.tex` (Korean labels, the single source) and copied to `docs/paper/figures/en/*.tex`
(English) by `scripts/paper_tex_en.py`. Never edit a generated file by hand; rerun its script. All scripts below
read result files only (no GPU, a few seconds; `persona_actions.py` and `persona_solo.py` replay 10 games per
checkpoint on CPU, a few minutes).

### Which script makes what

| paper label | file | made by | reads |
|---|---|---|---|
| `fig:teaser` (Fig. 1) | `figures/fig_teaser.tex` | hand-written TikZ | numbers listed at the top of the file (`results/judge/20261118_185811`, GPT-5-mini, GRPO theft s1, start 4) |
| `fig:steps` | `figures/fig1.tex` | `scripts/paper_fig1.py` | `results/train/<algo>/A_persona/persona_actions.csv`, `results/train/persona-fig/solo vs withgreedy/persona_solo.csv` |
| `tab:pool_personas` | `figures/tab_pool.tex` | `scripts/paper_results_tex.py` | judge folders (`JUDGE`) and their pools (`POOL`, `pool/meta.jsonl`) |
| `tab:gap` | `figures/tab_gap.tex` | `scripts/paper_results_tex.py` | same |
| `tab:monitor` | `figures/tab_monitor.tex` | `scripts/paper_results_tex.py` | same (rule of `scripts/team_score_monitor.py`, `THRESH = 0.8`) |
| `fig:judge` | `figures/fig_judges.tex` | `scripts/paper_results_tex.py` | `JUDGES`: Qwen3.8-27B, Gemma-3-27B, GPT-5-mini judge folders |
| `tab:judgefull` | `figures/tab_judge_full.tex` | `scripts/paper_results_tex.py` | `FULL_JUDGE` (GPT-5-mini) judge folders |
| `tab:discovery` | `figures/tab_discovery.tex` | `python -m pool`, then `scripts/paper_results_tex.py` | `results/pool/0_persona_map/hackers.csv` |
| `tab:design` | inline in `main.tex` | hand-written | the persona rewards (`environment/kitchen.py`) |
| `tab:random` | `figures/tab_random_start.tex` | `scripts/paper_results_tex.py` | the judge pools; caption check vs fixed starts: `scripts/check_random_start.py` |
| `tab:solo` | inline in `main.tex` | hand-written | `persona_solo.csv` (modes `with greedy` / `alone`) |
| `tab:pool` | inline in `main.tex` | hand-written | `python -m eval.summarize` on the persona runs (`results/train/<algo>/A_persona/*`) and the team runs (`results/train/<algo>/A`) |
| `tab:prompts` and the prompt boxes | inline in `main.tex` | hand-written | verbatim from `judge/prompt.py` |
| `tab:pairwise` | inline in `main.tex` | hand-written | pairwise judge: `judge.judge_llm` (`scripts/run_judge_m1.sh` on vLLM), `judge.judge_shapley`, `judge.audit` -> `audit.md` |
| `tab:timeline` | inline in `main.tex` | hand-written | `results/pool/ppo/timeline/findings.md` (its script has been removed) |
| `tab:hparams` | inline in `main.tex` | hand-written | defaults of `train/config.py` (each run's `config.json`) |
| `fig:curves` | `figures/fig_curves.tex` + `figures/data/curves_*.dat` | `scripts/paper_persona_pgf.py` | `results/train/<algo>/**/curves/<algo>_<persona>_greedy_s*.csv` |

### Rebuild everything

Run from the project root, in this order (each step only needs the ones above it). `python3 -I` is enough for the
standard-library scripts; the others need the project env.

```bash
# 1. persona behaviour (fig:steps, tab:solo): replays the greedy-trained checkpoints
venv/bin/python scripts/persona_actions.py --algo ppo  --root results/train/ppo/A_persona
venv/bin/python scripts/persona_actions.py --algo grpo --root results/train/grpo/A_persona
venv/bin/python scripts/persona_solo.py
python3 -I scripts/paper_fig1.py                       # -> figures/fig1.tex

# 2. persona search (tab:discovery): overwrites results/pool/0_persona_map/hackers.{csv,md,png}
python pool

# 3. pool and judge tables (tab:pool_personas, tab:gap, tab:monitor, fig:judge, tab:judgefull, tab:random, tab:discovery)
python3 -I scripts/paper_results_tex.py                # judge folders: JUDGE, JUDGES, FULL_JUDGE at the top of the script
python3 -I scripts/check_random_start.py results/judge/2026*     # optional: fixed vs random start check (tab:random caption)

# 4. learning curves (fig:curves)
venv/bin/python scripts/paper_persona_pgf.py           # -> figures/fig_curves.tex, figures/data/*.dat

# 5. English copies, after any of the above or after editing fig_teaser.tex
python3 -I scripts/paper_tex_en.py                     # -> figures/en/*.tex; stops if Korean is left untranslated
```

After a new judge run, put its `results/judge/<stamp>` folders into `JUDGE` / `JUDGES` in
`scripts/paper_results_tex.py` and rerun steps 3 and 5. A prompt is drawn only when at least 95% of the pool has
been judged (`COMPLETE`), so a partial run is skipped, not drawn half done.

## Code

- `environment/overcooked.py` (`--env overcooked`): Overcooked, its rewards (`team`, `theft`, `pickup`, `shadow`,
  `cf`, `prep`), partners (`greedy`, `stay`, `self`, `ckpt:<path>`) and random starts.
  `environment/rware.py` (`--env rware`): the multi-robot warehouse, 2 to 4 robots, rewards `team`, `theft`,
  `pickup`, `shadow`, `hoard`, partners `greedy`, `stay`. The old names `kitchen` / `warehouse` still work.
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

- `scripts/`: judge run scripts (`run_judge_single.sh`, `run_judge_api.sh`, `run_judge_m1.sh` for the pairwise
  judge), `team_score_monitor.py`, `check_random_start.py`, `plot_curves.py`, the paper scripts of section 4
  (`paper_results_tex.py`, `paper_fig1.py`, `paper_persona_pgf.py`, `paper_tex_en.py`, `persona_actions.py`,
  `persona_solo.py`), the 3-robot RWARE persona run (`rware3_local.sh`, `rware3_windows.ps1`), `cluster/` (Slurm).
- `docs/paper/`: the paper (`main.tex`, `figures/`, `figures/en/`, `figures/data/`).
- `results/`: `train/` (runs), `ckpts/` (checkpoints for the pools), `judge/<stamp>/` (pools and judgments).
