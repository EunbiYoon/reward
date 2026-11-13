칸	명령
PPO B1	.\venv\Scripts\python -m eval.run --algos ppo --rewards team --partners ckpt:results/full/models/ppo_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/ppo_B1
PPO B2	.\venv\Scripts\python -m eval.run --algos ppo --rewards individual --partners ckpt:results/full/models/ppo_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/ppo_B2
PPO B3	.\venv\Scripts\python -m eval.run --algos ppo --rewards pickup --partners ckpt:results/full/models/ppo_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/ppo_B3
GRPO B1	.\venv\Scripts\python -m eval.run --algos grpo --rewards team --partners ckpt:results/full/models/grpo_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/grpo_B1
GRPO B2	.\venv\Scripts\python -m eval.run --algos grpo --rewards individual --partners ckpt:results/full/models/grpo_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/grpo_B2
GRPO B3	.\venv\Scripts\python -m eval.run --algos grpo --rewards pickup --partners ckpt:results/full/models/grpo_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/grpo_B3
SAC B1	.\venv\Scripts\python -m eval.run --algos sac --rewards team --partners ckpt:results/sac_team/models/sac_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/sac_B1
SAC B2	.\venv\Scripts\python -m eval.run --algos sac --rewards individual --partners ckpt:results/sac_team/models/sac_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/sac_B2
SAC B3	.\venv\Scripts\python -m eval.run --algos sac --rewards pickup --partners ckpt:results/sac_team/models/sac_team_greedy_s0.pt --seeds 0 1 2 --workers 3 --out results/sac_B3