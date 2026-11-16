# Who Takes the Credit?

One agent (seat 0) learns next to a fixed partner (seat 1). We measure its real contribution with the
Shapley value and compare it with the credit that its reward, a team-score supervisor or an LLM judge
gives it.

## 1. `environment/`: the games
- `kitchen.py`: two-cook Overcooked. Goal: deliver soups.
  - Rewards: `team` (team deliveries), `individual` (own deliveries), `pickup` (+3 per pickup, the
    fake-help control), `cf` (team reward minus what the partner earns alone).
  - Partners: `stay` (never moves) or `greedy` (scripted cook that plays well).
- `warehouse.py`, `foraging.py`, `alem.py`: three more games behind the same interface.

## 2. `train/`: make the checkpoints
- `run.py`: trains every (seed, optimizer, reward, partner) of a grid and saves the checkpoints
  (`models/`), learning curves (`curves/`) and `train_log.csv`. Training only.
- `ppo.py`, `grpo.py`: the two optimizers used in the paper.
- `config.py`: every experiment setting (steps, learning rates, rewards, layout, evaluation games).
- `deep_policy.py`: MLP policy, observation normalization, checkpoints.
- `resume.py`: resumes a run after a cluster job is interrupted.

## 3. `rlaif/`: LLM judges
- `collect.py`: records a pool of episodes from different seat-0 agents (scripted cook, trained theft,
  quit, pickup and shadow personas, blocker), each with its true Shapley contribution.
- `describe.py`: turns an episode into the event log the judge reads.
- `judge.py`: an LLM compares two episodes and gives each seat-0 agent a credit share.
  Judges: Qwen via Ollama, GPT-4o, Claude, vLLM. Prompts: basic, + Shapley definition, + solo results.
- `audit.py`: scores the judge against Shapley (how often it is fooled by the pickup or theft persona).
- `reward_model.py`, `learned_env.py`: turn the judge's verdicts into a training reward.

## 4. `eval/`: check what training produced
- `run.py`: loads every checkpoint of a run folder and measures it (`results.csv`, `reference.csv`)
  Each evaluation goes to a new `<run>/eval/<YYYYmmdd_HHMMSS>/`; readers use the newest (`paths.py`).
- `persona.py`: did training produce the intended persona? Counts soup steps 1-7 for learner and partner
  and checks them against the persona's rules (`persona_match` in `results.csv`, log in `persona_check.md`).
- `coalitions.py`: the measurement. It replays each episode three ways (full team, learner alone,
  partner alone) to get the Shapley share s and the credit gap (own share of deliveries − s).
- `summarize.py`: tables and verdicts (pickup, free-riding, theft).
- `visualize.py`: GIFs of the cooks.

```bash
pip install -r requirements.txt
python -m train.run --algos ppo --rewards team individual pickup --partners greedy --seeds 0 1 2 --out results/ppo/x
python -m eval.run results/ppo/x
python -m eval.summarize results/ppo/x
```

Also: `scripts/` (paper pipeline, `cluster/` for UMass Unity), `docs/page/` (project page), `results/`
(outputs, one folder per algorithm: `results/ppo/`, `results/grpo/`).
