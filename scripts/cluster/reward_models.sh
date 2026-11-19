#!/bin/bash
# After the judge: the oracle (Shapley) judge and both Bradley-Terry reward models (a few minutes, CPU).
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x.log
set -eu
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster/activate.sh
RUN=${RUN:-results/judge/theft}
python -m judge.judge_shapley --run "$RUN" --pairs 0
python -m judge.reward_model --run "$RUN" --judge local
python -m judge.reward_model --run "$RUN" --judge shapley
