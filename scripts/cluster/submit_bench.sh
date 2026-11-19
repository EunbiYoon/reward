#!/bin/bash
# One benchmark end to end, as was done for Overcooked: train -> eval -> pool (Shapley ground truth).
# The judge (Qwen3.8-27B-FP8 on a GPU) comes after, with scripts/bench_judge.sh.
#
#   bash scripts/cluster/submit_bench.sh warehouse          # RWARE  (layout tiny, 500 steps)
#   bash scripts/cluster/submit_bench.sh foraging           # LBF    (layout 8x8-3f, 50 steps)
#   SMOKE=1 bash scripts/cluster/submit_bench.sh warehouse  # smoke test: 1 seed, 20k steps, 2 pool episodes
#
# Submits three Slurm jobs (run from the project root, on a login node; nothing heavy runs here):
#   bench_<env>          array, one task per (algo, reward, seed): PPO and GRPO x theft pickup shadow team x
#                        seeds 0-2 = 24 tasks next to greedy, 2M steps each, then eval (--eval)
#                        -> results/bench_<env>/models/*.pt, results.csv
#   bench_<env>_summary  after the array: python -m eval.summarize -> results/bench_<env>/summary.md
#   bench_<env>_pool     after the array succeeds: scripts/cluster/collect_pool.sh with BENCH=<env>
#                        -> results/judge/<stamp>/ (one folder per algo, files ALGO and BENCH)
# team is the control (shared team reward), as in the Overcooked appendix; the pool holds theft, pickup,
# shadow and the greedy reference.
#
#   EPISODES   pool start states per behaviour (default 20, as for Overcooked)
#   STEPS      env steps per run (default 2000000, as for Overcooked)
#   MAIL       e-mail for END/FAIL of the pool job (optional)
# Watch: squeue --me    tail logs/bench_<env>_<task>.log    logs/collect_pool-<jobid>.out
set -euo pipefail
ENV=${1:?usage: submit_bench.sh <warehouse|foraging>}
case $ENV in warehouse) MOD=rware ;; foraging) MOD=lbforaging ;; *) echo "env must be warehouse or foraging"; exit 1 ;; esac
cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/cluster/activate.sh
python -c "import torch, $MOD" 2>/dev/null || { echo "the project env lacks torch or $MOD (see requirements.txt)"; exit 1; }

NAME=bench_$ENV
GRID=(--env "$ENV" --algos ppo grpo --rewards theft pickup shadow team --partners greedy)
if [ -n "${SMOKE:-}" ]; then
    NAME=${NAME}_smoke
    GRID+=(--seeds 0 --env-steps 20000)
    EPISODES=${EPISODES:-2}
else
    GRID+=(--seeds 0 1 2 --env-steps "${STEPS:-2000000}")
fi
N=$(python -m train.run "${GRID[@]}" --out "results/$NAME" --list-jobs | tail -n 1)
echo "== $NAME: $N training runs (${GRID[*]})"

TRAIN=$(sbatch --parsable --job-name="$NAME" --array=0-$((N - 1))%24 scripts/cluster/run_task.sh "$NAME" "${GRID[@]}")
echo "train+eval: job $TRAIN (array of $N)"

SUM=$(sbatch --parsable --job-name="${NAME}_summary" --partition=cpu-preempt --mem=4G --time=00:20:00 \
      --output="logs/${NAME}_summary.log" --dependency=afterany:"$TRAIN" \
      --wrap="bash -c 'source scripts/cluster/activate.sh && python -m eval.summarize results/$NAME > /dev/null && echo results/$NAME/summary.md'")
echo "summary:    job $SUM (after the array)"

MAILOPT=()
[ -n "${MAIL:-}" ] && MAILOPT=(--mail-type=END,FAIL --mail-user="$MAIL")
POOL=$(sbatch --parsable --job-name="${NAME}_pool" ${MAILOPT[@]+"${MAILOPT[@]}"} --dependency=afterok:"$TRAIN" \
       --export=ALL,BENCH="$ENV",SRC="results/$NAME/models",EPISODES="${EPISODES:-20}" \
       ${SMOKE:+--time=00:30:00} scripts/cluster/collect_pool.sh)
echo "pool:       job $POOL (after every training run succeeds)"
echo
echo "When the pool job is done, judge it with Qwen inside a GPU allocation:"
echo "  bash scripts/bench_judge.sh $ENV"
