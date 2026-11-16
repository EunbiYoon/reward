#!/bin/bash
# After every array: summaries and the RLAIF credit audit.
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x.log
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster/activate.sh
RUN=${RUN:-results/rlaif/theft}
for d in "$@"; do
    ls results/$d/results.csv results/$d/eval/*/results.csv > /dev/null 2>&1 \
        && python -m eval.summarize "results/$d" | head -3
done
ls results/rlaif_kitchen/results.csv results/rlaif_kitchen/eval/*/results.csv > /dev/null 2>&1 && python -m rlaif.audit --run "$RUN" --train-run results/rlaif_kitchen
