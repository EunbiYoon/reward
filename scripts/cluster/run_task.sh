#!/bin/bash
# One training run per Slurm array task (submitted by scripts/cluster/submit.sh; do not sbatch by hand).
#   usage: sbatch --array=0-(N-1) scripts/cluster/run_task.sh <name> <train.run grid args...>
# Preemption: Slurm requeues the task (--requeue) and training resumes from results/<name>/resume/
# (saved every 5 min). Time limit: the run stops at 55 min and the task requeues itself to continue.
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%a.log
set -u
NAME=$1; shift
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster/activate.sh
echo "== $(date) task $SLURM_ARRAY_TASK_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
timeout 55m python -m train.run "$@" --eval --out "results/$NAME" --job-index "$SLURM_ARRAY_TASK_ID"
rc=$?
if [ $rc -eq 124 ]; then
    echo "== time limit reached; requeue to continue from the checkpoint"
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
exit $rc
