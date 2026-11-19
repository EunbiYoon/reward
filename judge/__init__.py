"""judge pipeline: trajectory pool -> LLM-as-a-judge preferences -> Bradley-Terry reward model
-> PPO / GRPO on the learned reward (train/run.py) -> credit audit against Shapley."""
