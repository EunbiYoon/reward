#!/bin/bash
# Run on this Mac everything the check-in report (report/checkin.tex) still needs, then summarize.
#
#   bash scripts/local_report.sh            # all; ~6 h on a 14-core M-series Mac
#   bash scripts/local_report.sh cpu        # only the training grids (H7, H8, H10, H12)
#   bash scripts/local_report.sh rlaif      # only judge -> reward models -> RLAIF training
#   GRIDS="h10_cf h7_bonus" bash scripts/local_report.sh   # run only some grids here (others on Unity)
#
# Two streams run at the same time: the training grids on the CPU, and the Ollama judge on the GPU.
# Every step is resumable: rerunning skips finished runs and already-judged pairs.
# Not run here (too slow locally): rware_5m and alem_full; they stay on the Unity cluster.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}
WHAT=${1:-all}
W=${WORKERS:-12}                       # training processes; leave a couple of cores for Ollama
JUDGE_MODEL=${JUDGE_MODEL:-qwen3.8:27b-mlx}
PAIRS=${PAIRS:-300}                    # ~35 s per pair -> ~3 h; the cluster plan was all 1,680
RUN=results/rlaif/kitchen
mkdir -p logs

grid() {   # grid <name> <eval.run args...>
    local name=$1; shift
    echo "[$(date +%H:%M)] $name start"
    "$PY" -m eval.run "$@" --workers "$W" --out "results/$name" > "logs/local_$name.log" 2>&1
    "$PY" -m eval.summarize "results/$name" > /dev/null
    echo "[$(date +%H:%M)] $name done -> results/$name/summary.md"
}

args() {   # macOS bash 3.2 has no associative arrays
    case $1 in
        h10_cf)   echo "--rewards cf --partners stay greedy" ;;                                          # 12 runs
        h7_bonus) echo "--rewards pickup --partners greedy --pickup-bonus 0 1 3 5 10" ;;                 # 30 runs
        h8_noise) echo "--rewards team individual --partners greedy --partner-noise 0 0.1 0.25 0.5" ;;  # 48 runs
        h12_aux)  echo "--rewards individual team --partners greedy --aux-coef 0 0.1 0.5 1.0 --seeds 0 1 2" ;;  # 48 (cluster: 10 seeds)
    esac
}
GRIDS=${GRIDS:-h10_cf h7_bonus h8_noise h12_aux}   # e.g. GRIDS="h10_cf h7_bonus" to run only those here

cpu() {
    for g in $GRIDS; do grid "$g" $(args "$g"); done
}

judge() {
    echo "[$(date +%H:%M)] judge: $PAIRS pairs with $JUDGE_MODEL"
    "$PY" -m rlaif.judge --run "$RUN" --judge ollama --model "$JUDGE_MODEL" --pairs "$PAIRS" --workers 1 \
        > logs/local_judge.log 2>&1
    "$PY" -m rlaif.judge --run "$RUN" --judge shapley --pairs 0 >> logs/local_judge.log 2>&1
    "$PY" -m rlaif.reward_model --run "$RUN" --judge ollama  > logs/local_rm.log 2>&1
    "$PY" -m rlaif.reward_model --run "$RUN" --judge shapley >> logs/local_rm.log 2>&1
    echo "[$(date +%H:%M)] judge + reward models done"
}

rlaif_train() {
    grid rlaif_kitchen        --rewards learned team individual --reward-model "$RUN/rm_ollama.pt"   # 36 runs
    grid rlaif_kitchen_oracle --rewards learned --reward-model "$RUN/rm_shapley.pt"                 # 12 runs
}

case "$WHAT" in
    cpu)   cpu ;;
    rlaif) judge; rlaif_train ;;
    all)   judge & JPID=$!
           cpu
           wait "$JPID"
           rlaif_train ;;
esac

echo "[$(date +%H:%M)] final reports"
DIRS=$(for d in full lbf_full rware_full h7_bonus h8_noise h10_cf h12_aux rlaif_kitchen rlaif_kitchen_oracle; do
       [ -f "results/$d/results.csv" ] && echo "results/$d"; done)
"$PY" -m eval.oversight $DIRS --out results/oversight.md
[ -f results/rlaif_kitchen/results.csv ] && "$PY" -m rlaif.audit --run "$RUN" --train-run results/rlaif_kitchen
echo "[$(date +%H:%M)] all done"
