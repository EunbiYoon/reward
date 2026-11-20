#!/bin/bash
# 3-robot RWARE theft persona, end to end on one machine with 1 GPU (the GPU is only for the judge):
#   1 bc      clone the greedy robot (train/bc.py, CNN policy) per seed  -> results/rware3_bc/bc_greedy_greedy_n3_s<seed>.pt
#   2 train   PPO theft (10 x own deliveries) next to 2 greedy robots, STARTING from the BC cook of the same seed
#                                                                        -> results/rware3_theft_bc/ (models/, curves/, eval/)
#   3 pool    theft checkpoints + the BC cooks + greedy, EPISODES start states, every coalition value
#                                                                        -> results/judge/rware3_theft_bc_pool/
#   4 judge   Qwen on vLLM (scripts/run_judge_single.sh)              -> results/judge/<stamp>/single_report.md
# Rerun to resume: finished BC checkpoints, trained runs and the pool are kept.
#
#   bash scripts/rware3_theft_gpu.sh
#   SKIP="judge" bash scripts/rware3_theft_gpu.sh                      # no GPU: stop after the pool
#   SEEDS="0" STEPS=50000 EPISODES=2 BC_EPISODES=4 BC_ROUNDS=1 LIMIT=4 bash scripts/rware3_theft_gpu.sh   # smoke test
#
#   SEEDS       default "0 1 2"
#   STEPS       PPO steps per run after BC (default 1000000)
#   BC_EPISODES, BC_ROUNDS, BC_EPOCHS   train/bc.py (40, 6, 15)
#   EPISODES    pool start states (default 20)
#   WORKERS     parallel runs / pool workers (default: one per seed)
#   PROMPTS     judge prompts (default "basic definition solo"); MODEL, LIMIT as in scripts/run_judge_single.sh
#   SKIP        any of: bc train pool judge
#   PY          python (default: the active one)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-0 1 2}
STEPS=${STEPS:-1000000}
BC_EPISODES=${BC_EPISODES:-40}; BC_ROUNDS=${BC_ROUNDS:-6}; BC_EPOCHS=${BC_EPOCHS:-15}
EPISODES=${EPISODES:-20}
read -r -a SEED_ARR <<< "$SEEDS"
WORKERS=${WORKERS:-${#SEED_ARR[@]}}
PROMPTS=${PROMPTS:-basic definition solo}
BC=results/rware3_bc
OUT=results/rware3_theft_bc
POOL=results/judge/rware3_theft_bc_pool
mkdir -p logs
LOG=logs/rware3_theft_gpu.log
exec > >(tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
say() { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
skip() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
"$PY" -c "import torch, rware" || { echo "this python lacks torch / rware: $PY -m pip install -r requirements.txt"; exit 1; }
say "python $("$PY" --version 2>&1), seeds $SEEDS, $STEPS PPO steps per run"

# ------------------------------------------------------------ 1 bc
if ! skip bc; then
    for s in "${SEED_ARR[@]}"; do
        if [ -f "$BC/bc_greedy_greedy_n3_s$s.pt" ]; then say "bc seed $s exists"; continue; fi
        say "bc seed $s"
        "$PY" -m train.bc --agents 3 --partners greedy --seed "$s" --episodes "$BC_EPISODES" \
            --rounds "$BC_ROUNDS" --epochs "$BC_EPOCHS" --out "$BC"
    done
fi

# ------------------------------------------------------------ 2 train (+ eval)
if ! skip train; then
    say "train theft from the BC cooks -> $OUT"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m train.run --env rware --agents 3 --algos ppo --rewards theft \
        --partners greedy --seeds "${SEED_ARR[@]}" --env-steps "$STEPS" \
        --init-ckpt "$BC/bc_greedy_greedy_n3_s{seed}.pt" --eval --no-plot --workers "$WORKERS" --out "$OUT"
fi

# ------------------------------------------------------------ 3 pool
if ! skip pool; then
    if [ -f "$POOL/pool/meta.jsonl" ]; then say "pool exists: $POOL (delete the folder to rebuild)"
    else
        mkdir -p "$POOL/ckpts"
        echo ppo > "$POOL/ALGO"; echo rware > "$POOL/BENCH"
        cp "$OUT"/models/*.pt "$POOL/ckpts/"
        for s in "${SEED_ARR[@]}"; do cp "$BC/bc_greedy_greedy_n3_s$s.pt" "$POOL/ckpts/"; done
        say "pool: $(ls "$POOL"/ckpts/*.pt | wc -l | tr -d ' ') checkpoints + greedy, $EPISODES episodes -> $POOL"
        OMP_NUM_THREADS=1 "$PY" -m judge.collect --env rware --agents 3 --partners greedy --behaviors greedy \
            --policies "$POOL/ckpts/*.pt" --episodes "$EPISODES" --workers "$WORKERS" --out "$POOL"
    fi
fi

# ------------------------------------------------------------ 4 judge (vLLM on the GPU)
if ! skip judge; then
    say "judge $POOL ($PROMPTS)"
    LIMIT=${LIMIT:-0} bash scripts/run_judge_single.sh "$POOL" "$PROMPTS"
fi
say "all done"
