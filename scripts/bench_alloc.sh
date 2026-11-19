#!/bin/bash
# One (benchmark, algo) end to end INSIDE one GPU allocation: train -> eval -> pool -> judge (Qwen + Gemini).
# Meant for 4 allocations at once, one per (warehouse|foraging) x (ppo|grpo):
#
#   bash scripts/bench_alloc.sh warehouse ppo        # in allocation 1
#   bash scripts/bench_alloc.sh warehouse grpo       # in allocation 2
#   bash scripts/bench_alloc.sh foraging  ppo        # in allocation 3
#   bash scripts/bench_alloc.sh foraging  grpo       # in allocation 4
#
# Steps (each one is skipped or resumed when rerun, so a killed allocation can just run it again):
#   1 train + eval  12 runs (theft pickup shadow team x seeds 0-2) next to greedy, one after another
#                   -> results/bench_<env>_<algo>/  (models/, parts/, results.csv)
#   2 summary       results/bench_<env>_<algo>/summary.md
#   3 pool          scripts/cluster/collect_pool.sh -> results/judge/<stamp>/ (ALGO, BENCH, pool/)
#   4 judge         Gemini (API, in the background) and Qwen3.8-27B-FP8 (vLLM on this GPU) at the same time,
#                   each into its own results/judge/<stamp>/; Gemini needs GEMINI_API_KEY in .env
#
#   STEPS     env steps per run (default 2000000, as for Overcooked)
#   EPISODES  pool start states per behaviour (default 20)
#   PROMPTS   default "score basic definition without numbers solo"
#   SKIP      steps to skip, e.g. SKIP="train" or SKIP="train pool" (judge an existing pool)
# Logs: logs/bench_<env>_<algo>.log (this script), logs/bench_<env>_<algo>_gemini.log
set -euo pipefail
ENV=${1:?usage: bench_alloc.sh <warehouse|foraging> <ppo|grpo>}
ALGO=${2:?usage: bench_alloc.sh <warehouse|foraging> <ppo|grpo>}
case $ENV in warehouse|foraging) ;; *) echo "env must be warehouse or foraging"; exit 1 ;; esac
case $ALGO in ppo|grpo) ;; *) echo "algo must be ppo or grpo"; exit 1 ;; esac
cd "$(dirname "$0")/.."
mkdir -p logs
exec > >(tee -a "logs/bench_${ENV}_${ALGO}.log") 2>&1
source scripts/cluster/activate.sh
echo "== $(date) on $(hostname) (job ${SLURM_JOB_ID:-none}): $ENV $ALGO"
which python && python --version
nvidia-smi -L 2>/dev/null || echo "(no GPU visible: the Qwen judge will fail)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
skip() { [[ " ${SKIP:-} " == *" $1 "* ]]; }

OUT=results/bench_${ENV}_${ALGO}
PROMPTS=${PROMPTS:-score basic definition without numbers solo}
GRID=(--env "$ENV" --algos "$ALGO" --rewards theft pickup shadow team --partners greedy --seeds 0 1 2
      --env-steps "${STEPS:-2000000}")

if ! skip train; then
    N=$(python -m train.run "${GRID[@]}" --out "$OUT" --list-jobs | tail -n 1)
    for i in $(seq 0 $((N - 1))); do
        echo "== $(date) train+eval $((i + 1))/$N"
        python -m train.run "${GRID[@]}" --eval --no-plot --out "$OUT" --job-index "$i"
    done
    python -m eval.summarize "$OUT" > /dev/null && echo "== summary: $OUT/summary.md"
fi

if ! skip pool; then
    echo "== $(date) pool"
    ALGOS=$ALGO BENCH=$ENV SRC=$OUT/models EPISODES=${EPISODES:-20} WORKERS=${SLURM_CPUS_PER_TASK:-1} \
        bash scripts/cluster/collect_pool.sh
fi

POOL=""
for d in $(ls -d results/judge/*/ 2>/dev/null | sort -r); do              # newest pool of this env + algo
    d=${d%/}
    if [ -f "$d/pool/meta.jsonl" ] && [ "$(cat "$d/BENCH" 2>/dev/null)" = "$ENV" ] \
       && [ "$(cat "$d/ALGO" 2>/dev/null)" = "$ALGO" ]; then POOL=$d; break; fi
done
[ -n "$POOL" ] || { echo "no $ENV $ALGO pool in results/judge/"; exit 1; }

if ! skip judge; then
    echo "== $(date) judge $POOL: Gemini (background, logs/bench_${ENV}_${ALGO}_gemini.log) + Qwen"
    JUDGE=gemini bash scripts/run_judge_api.sh "$POOL" "$PROMPTS" > "logs/bench_${ENV}_${ALGO}_gemini.log" 2>&1 &
    GEM=$!
    sleep 2                                    # the two judges name their folders by the second they start
    MODEL=Qwen/Qwen3.8-27B-FP8 bash scripts/run_judge_single.sh "$POOL" "$PROMPTS"
    wait "$GEM" && echo "== Gemini done" || echo "== Gemini FAILED: see logs/bench_${ENV}_${ALGO}_gemini.log"
fi
echo "== $(date) done: $ENV $ALGO"
