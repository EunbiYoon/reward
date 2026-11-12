"""RLAIF pipeline: trajectory pool -> LLM-as-a-judge preferences -> Bradley-Terry reward model
-> PPO / GRPO on the learned reward (eval/run.py) -> credit audit against Shapley."""
