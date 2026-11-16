#!/bin/bash
# Every experiment the AAAI-27 workshop paper (docs/paper/main.tex) uses, on this Mac. ~6-7 h.
#
#   caffeinate -i nohup bash scripts/paper_experiments.sh > logs/paper.log 2>&1 &
#   tail -f logs/paper.log
#
# Two streams at once, both resumable (rerun skips finished runs / judged pairs):
#   CPU  more seeds (3,4) for the main grid and the cf reward, then the bonus and noise sweeps
#   GPU  the LLM judge (Ollama) on the theft pool, three prompts: basic, + Shapley definition,
#        + definition and each episode's solo results
# At the end: merged summaries, oversight, judge audit and the paper figures (docs/paper/figures/).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}
W=${WORKERS:-11}
MODEL=${JUDGE_MODEL:-qwen3.8:27b-mlx}
PAIRS=${PAIRS:-200}                    # theft pool: 20 seeds x 10 behaviour pairs = 200 (all of them)
THEFT=results/rlaif/theft
mkdir -p logs results/paper

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

grid() {   # grid <name> <train.run args...>
    local name=$1; shift
    say "$name start"
    if "$PY" -m train.run "$@" --eval --workers "$W" --out "results/$name" > "logs/paper_$name.log" 2>&1; then
        "$PY" -m eval.summarize "results/$name" > /dev/null 2>&1
        say "$name done"
    else
        say "$name FAILED (see logs/paper_$name.log)"
    fi
}

cpu() {
    grid full_s34   --rewards pickup team individual --partners stay greedy --seeds 3 4      # 24 runs
    grid h10_cf_s34 --rewards cf --partners stay greedy --seeds 3 4                         # 8 runs
    "$PY" scripts/merge_runs.py results/paper/full5 results/full results/full_s34
    "$PY" scripts/merge_runs.py results/paper/cf5 results/h10_cf results/h10_cf_s34
    "$PY" -m eval.summarize results/paper/full5 > /dev/null 2>&1
    "$PY" -m eval.summarize results/paper/cf5 > /dev/null 2>&1
    say "merged 5-seed results -> results/paper/full5, results/paper/cf5"
    grid h7_bonus --rewards pickup --partners greedy --pickup-bonus 0 1 3 5 10                  # 30 runs
    grid h8_noise --rewards team individual --partners greedy --partner-noise 0 0.1 0.25 0.5   # 48 runs
}

judge() {
    "$PY" -m rlaif.judge --run "$THEFT" --judge shapley --pairs 0 > logs/paper_judge.log 2>&1
    for prompt in basic solo definition; do
        say "judge $prompt: $PAIRS pairs with $MODEL"
        "$PY" -m rlaif.judge --run "$THEFT" --judge ollama --model "$MODEL" --prompt "$prompt" \
            --pairs "$PAIRS" --workers 1 >> logs/paper_judge.log 2>&1 || say "judge $prompt FAILED"
        "$PY" -m rlaif.audit --run "$THEFT" > /dev/null 2>&1
        say "judge $prompt done -> $THEFT/audit.md"
    done
}

say "start (CPU workers=$W, judge=$MODEL)"
judge & JPID=$!
cpu
wait "$JPID"

say "final reports"
DIRS=$(for d in paper/full5 lbf_full rware_full paper/cf5 h7_bonus h8_noise; do
       [ -f "results/$d/results.csv" ] && echo "results/$d"; done)
"$PY" -m eval.oversight $DIRS --out results/paper/oversight.md > /dev/null 2>&1
"$PY" -m rlaif.audit --run "$THEFT" > /dev/null 2>&1
"$PY" scripts/paper_figures.py > logs/paper_figures.log 2>&1 && say "figures -> docs/paper/figures/" \
    || say "figures FAILED (see logs/paper_figures.log)"
say "all done"
