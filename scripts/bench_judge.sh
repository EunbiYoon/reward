#!/bin/bash
# Judge the newest RWARE / LBF pools (made by scripts/cluster/submit_bench.sh) with all six prompts,
# one judge run per algo, as was done for Overcooked. Judge: Qwen3.8-27B-FP8 only (the paper's main
# judge), on local vLLM inside a GPU allocation.
#
#   bash scripts/bench_judge.sh warehouse
#   LIMIT=4 bash scripts/bench_judge.sh warehouse                           # smoke test
#
#   $1        warehouse | foraging
#   PROMPTS   default "score basic definition without numbers solo"
#   ALGOS     default "ppo grpo"
# Every other variable (PORT, LIMIT, STAMP, MAXLEN, ...) passes through to scripts/run_judge_single.sh.
# Each run writes results/judge/<new stamp>/ with single_*_<prompt>.jsonl and single_report.md.
set -euo pipefail
ENV=${1:?usage: bench_judge.sh <warehouse|foraging>}
PROMPTS=${PROMPTS:-score basic definition without numbers solo}
cd "$(dirname "$0")/.."

for ALGO in ${ALGOS:-ppo grpo}; do
    POOL=""
    for d in $(ls -d results/judge/*/ 2>/dev/null | sort -r); do       # newest first
        d=${d%/}
        if [ -f "$d/pool/meta.jsonl" ] && [ "$(cat "$d/BENCH" 2>/dev/null)" = "$ENV" ] \
           && [ "$(cat "$d/ALGO" 2>/dev/null)" = "$ALGO" ]; then POOL=$d; break; fi
    done
    [ -n "$POOL" ] || { echo "no $ENV $ALGO pool in results/judge/ (run scripts/cluster/submit_bench.sh $ENV first)"; exit 1; }
    echo "== $ENV $ALGO: judging $POOL ($PROMPTS)"
    MODEL=Qwen/Qwen3.8-27B-FP8 bash scripts/run_judge_single.sh "$POOL" "$PROMPTS"
done
