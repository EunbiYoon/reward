#!/bin/bash
# After every array: summaries, the H3 oversight table and the RLAIF credit audit.
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
    [ -f "results/$d/results.csv" ] && python -m eval.summarize "results/$d" > /dev/null && echo "summary: results/$d/summary.md"
done
python -m eval.oversight results/full results/lbf_full $(for d in "$@"; do [ -f "results/$d/results.csv" ] && echo "results/$d"; done)
[ -f results/rlaif_kitchen/results.csv ] && python -m rlaif.audit --run "$RUN" --train-run results/rlaif_kitchen
