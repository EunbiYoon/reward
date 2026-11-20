#!/bin/bash
# Mac / Linux workstation (no Slurm), one command: 3-robot RWARE, three personas, train -> eval -> pool -> Qwen judge
# (Ollama). Same steps, folders and resume behaviour as scripts/rware3_windows.ps1.
#
#   bash scripts/rware3_local.sh                                  # ALGOS default: ppo
#   WORKERS=4 bash scripts/rware3_local.sh                        # 4 trainings at a time
#   SKIP="train" bash scripts/rware3_local.sh                     # pool + judge only
#   WORKERS=4 STEPS=20000 EPISODES=2 LIMIT=4 bash scripts/rware3_local.sh   # smoke test
#
# Personas (learner in seat 0, two scripted greedy robots in seats 1-2), seeds 0-2:
#   sniper = theft, hoarder = hoard, escort = shadow
# Steps (rerun to resume; Ctrl+C is safe):
#   1 train+eval  9 runs per algo -> results/rware3_<algo>/ (models/, parts/, eval/slurm_local/{results.csv,summary.md})
#   2 pool        9 checkpoints + greedy, EPISODES start states, every coalition value -> results/judge/rware3_<algo>_pool/
#   3 judge       Ollama $MODEL, six prompts -> results/judge/rware3_<algo>_qwen/single_report.md
#
#   PY        python of the project env (default venv/bin/python, else the active python)
#   MODEL     Ollama model (default qwen3.8:27b; `ollama serve` must be running)
#   ALGOS     default "ppo"     STEPS 2000000     EPISODES 20     WORKERS 1     LIMIT 0 (= every episode)
#   SKIP      any of: train pool judge
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$( [ -x venv/bin/python ] && echo venv/bin/python || command -v python )}
MODEL=${MODEL:-qwen3.8:27b}
STEPS=${STEPS:-2000000}; EPISODES=${EPISODES:-20}; WORKERS=${WORKERS:-1}; LIMIT=${LIMIT:-0}
mkdir -p logs
LOG=logs/rware3_local.log
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
say() { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
skip() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
"$PY" -c "import torch, rware" || { echo "the env lacks torch / rware: $PY -m pip install -r requirements.txt"; exit 1; }
say "python $("$PY" --version 2>&1) at $PY"

for ALGO in ${ALGOS:-ppo}; do
    OUT=results/rware3_$ALGO
    GRID=(--env rware --agents 3 --algos "$ALGO" --rewards theft hoard shadow
          --partners greedy --seeds 0 1 2 --env-steps "$STEPS")

    # ------------------------------------------------------------ 1 train + eval
    if ! skip train; then
        N=$("$PY" -m train.run "${GRID[@]}" --out "$OUT" --list-jobs | tail -n 1)
        say "train+eval $ALGO: $N runs -> $OUT ($WORKERS at a time)"
        for ((k = 0; k < WORKERS; k++)); do          # shard k: job indices k, k+W, ...; own eval folder each
            (
                export SLURM_ARRAY_JOB_ID=shard$k
                for ((i = k; i < N; i += WORKERS)); do
                    echo "== $(date '+%m-%d %H:%M:%S') shard $k: run $((i + 1))/$N"
                    "$PY" -m train.run "${GRID[@]}" --eval --no-plot --out "$OUT" --job-index "$i"
                done
            ) > "logs/rware3_${ALGO}_shard$k.log" 2>&1 &
        done
        say "  $WORKERS shard(s) started (logs/rware3_${ALGO}_shard*.log); waiting"
        wait
        for ((i = 0; i < N; i++)); do                # merge every evaluated run into eval/slurm_local/ (no retraining)
            "$PY" -m train.run "${GRID[@]}" --eval --no-plot --out "$OUT" --job-index "$i" > /dev/null
        done
        "$PY" -m eval.summarize "$OUT" > /dev/null
        say "train+eval $ALGO done: $OUT/eval/slurm_local/summary.md"
    fi

    # ------------------------------------------------------------ 2 pool
    POOL=results/judge/rware3_${ALGO}_pool
    if ! skip pool; then
        if [ -f "$POOL/pool/meta.jsonl" ]; then say "pool $ALGO exists: $POOL (delete the folder to rebuild)"
        else
            mkdir -p "$POOL/ckpts"
            echo "$ALGO" > "$POOL/ALGO"; echo rware > "$POOL/BENCH"
            cp "$OUT"/models/*.pt "$POOL/ckpts/"
            say "pool $ALGO: $(ls "$POOL"/ckpts/*.pt | wc -l | tr -d ' ') checkpoints + greedy, $EPISODES episodes -> $POOL"
            "$PY" -m judge.collect --env rware --agents 3 --partners greedy --behaviors greedy \
                --policies "$POOL/ckpts/*.pt" --episodes "$EPISODES" --workers "$WORKERS" --out "$POOL"
        fi
    fi

    # ------------------------------------------------------------ 3 judge (Ollama)
    if ! skip judge; then
        J=results/judge/rware3_${ALGO}_qwen
        mkdir -p "$J"; echo "ollama:$MODEL" > "$J/MODEL"; echo "$POOL" > "$J/POOL"
        for P in score basic definition without numbers solo; do
            say "judge $ALGO $P (ollama $MODEL)"
            "$PY" -m judge.judge_single --run "$POOL" --out "$J" --judge ollama --model "$MODEL" \
                --prompt "$P" --workers 2 --limit "$LIMIT" || say "  judge $ALGO $P FAILED"
        done
        "$PY" -m judge.judge_single --out "$J" --report > /dev/null
        say "judge $ALGO done: $J/single_report.md"
    fi
done
say "all done"
