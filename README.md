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

## 2. `train/`: the learner
- `ppo.py`, `grpo.py`: the two optimizers used in the paper.
- `deep_policy.py`: MLP policy, observation normalization, checkpoints.
- `resume.py`: resumes a run after a cluster job is interrupted.

## 3. `rlaif/`: LLM judges
- `collect.py`: records a pool of episodes from different seat-0 agents (scripted cook, trained thief,
  free-rider, fake helper, blocker), each with its true Shapley contribution.
- `describe.py`: turns an episode into the event log the judge reads.
- `judge.py`: an LLM compares two episodes and gives each seat-0 agent a credit share.
  Judges: Qwen via Ollama, GPT-4o, Claude, vLLM. Prompts: basic, + Shapley definition, + solo results.
- `audit.py`: scores the judge against Shapley (how often it is fooled by busy work or by credit theft).
- `reward_model.py`, `learned_env.py`: turn the judge's verdicts into a training reward.

## 4. `eval/`: train and measure
- `run.py`: trains every (seed, optimizer, reward, partner) and measures the result.
- `coalitions.py`: the measurement. It replays each episode three ways (full team, learner alone,
  partner alone) to get the Shapley share s and the credit gap (own share of deliveries − s).
- `summarize.py`: tables and verdicts (fake help, free-riding, credit theft).
- `oversight.py`: how many hacked runs a team-score supervisor would accept.
- `visualize.py`: GIFs of the cooks.

```bash
pip install -r requirements.txt
python -m eval.run --rewards team individual pickup --partners stay greedy --out results/x
python -m eval.summarize results/x
```

Also: `scripts/` (paper pipeline, `cluster/` for UMass Unity), `docs/` (paper, report, project page,
[file-by-file guide](docs/STRUCTURE.md)), `results/` (outputs; what each folder is and which are used: [results/README.md](results/README.md)).
