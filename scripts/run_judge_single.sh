#!/bin/bash
# Single-episode LLM judge (judge/judge_single.py) inside an interactive GPU allocation (no sbatch).
# Every call makes a NEW folder results/judge/<YYYYMMDD_HHMMSS>/ (the time it starts) and writes everything
# there directly: single_local_<prompt>.jsonl per prompt, judge_<prompt>.log, single_report.md, POOL (which
# pool was judged). Starts vLLM if none is up on $PORT and LEAVES IT RUNNING for the next call; stop it with
# kill $(cat logs/vllm_single_<jobid>.pid).
#
#   bash scripts/run_judge_single.sh "results/judge/A results/judge/B ..."          # basic + definition + solo
#   bash scripts/run_judge_single.sh "results/judge/A results/judge/B" basic         # one prompt only
#   LIMIT=4 bash scripts/run_judge_single.sh "results/judge/A" basic                 # smoke test (4 episodes)
#   STAMP=20261118_033512 bash scripts/run_judge_single.sh "..."                     # resume into that folder
#
#   $1 POOLS    pool folders from judge.collect (each holds pool/), space separated in one quoted argument
#   $2 PROMPTS  default "basic definition solo"
#   MODEL       default Qwen/Qwen3.8-27B-FP8 on 1 GPU, Qwen/Qwen3.8-27B on 2+.
#               Another judge family, e.g. MODEL=RedHatAI/gemma-3-27b-it-FP8-dynamic (Gemma 3 27B, 29 GB, not gated)
#   MAXLEN, SEQS, GPU_UTIL, KVDTYPE   vLLM --max-model-len (16384), --max-num-seqs (32),
#               --gpu-memory-utilization (0.90), --kv-cache-dtype (auto); lower MAXLEN/SEQS if a big model OOMs
#   THINK=1     let a reasoning model think (sends --think: 4096 output tokens), e.g. MODEL=openai/gpt-oss-20b
#   VLLM_ARGS   extra vllm serve flags (default --language-model-only: skip the vision encoder)
#   LIMIT       at most this many episodes (default 0 = the whole pool)
#   VLLM, PORT  as in scripts/run_judge_m1.sh
set -euo pipefail
POOL=${1:?usage: run_judge_single.sh "<pool folder> [<pool folder> ...]" [prompts]}
PROMPTS=${2:-basic definition solo}
cd "$(dirname "$0")/.."
source scripts/cluster/activate.sh
mkdir -p logs

# the pools to judge: one or more folders, each holding pool/ (quote several: "dir1 dir2 ...")
RUNS=()
for d in $POOL; do
    [ -f "${d%/}/pool/meta.jsonl" ] || { echo "no pool: ${d%/}/pool/meta.jsonl"; exit 1; }
    RUNS+=("${d%/}")
done

# the output folder: a fresh timestamp (or STAMP to resume)
OUT=results/judge/${STAMP:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
printf "%s\n" "${RUNS[@]}" > "$OUT/POOL"
echo "== $(date) on $(hostname): pools ${RUNS[*]}, prompts: $PROMPTS"
echo "== results -> $OUT"
which python && python --version

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -gt 0 ] || { echo "no GPU here: run this inside the GPU allocation (srun --jobid=<id> --pty zsh)"; exit 1; }
if [ "$NGPU" -ge 2 ]; then MODEL=${MODEL:-Qwen/Qwen3.8-27B}; else MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}; fi
PORT=${PORT:-$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))}
PIDF=logs/vllm_single_${SLURM_JOB_ID:-$(hostname)}.pid   # one per allocation: several can run at once
VLOG=logs/vllm_single_${SLURM_JOB_ID:-$(hostname)}.log

if curl -sf "localhost:$PORT/v1/models" > /dev/null; then
    if ! curl -sf "localhost:$PORT/v1/models" | grep -qF "\"$MODEL\""; then
        echo "port $PORT already serves another model ($(curl -sf "localhost:$PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1))."
        echo "stop it first: kill \$(cat $PIDF)   (or use another PORT=...)"; exit 1
    fi
    echo "== reusing vLLM ($MODEL) on port $PORT"
else
    ROOT=$(cd .. && pwd)
    if [ -z "${VLLM:-}" ]; then
        VLLM=$ROOT/envs/credit-vllm/bin/vllm
        [ -x "$VLLM" ] || VLLM=$ROOT/eunbiyoon_umass_edu/.conda/envs/safety-eval/bin/vllm
    fi
    export HF_HOME=${HF_HOME:-$ROOT/hf} VLLM_USE_FLASHINFER_SAMPLER=0
    "$(dirname "$VLLM")/hf" download "$MODEL"
    nohup "$VLLM" serve "$MODEL" --tensor-parallel-size "$NGPU" --max-model-len "${MAXLEN:-16384}" --port "$PORT" \
        --gpu-memory-utilization "${GPU_UTIL:-0.90}" --max-num-seqs "${SEQS:-32}" --kv-cache-dtype "${KVDTYPE:-auto}" \
        ${VLLM_ARGS---language-model-only} >> $VLOG 2>&1 &
    echo $! > $PIDF
    until curl -sf "localhost:$PORT/v1/models" > /dev/null; do
        kill -0 "$(cat $PIDF)" 2>/dev/null || { echo "vLLM died, see $VLOG"; exit 1; }
        sleep 10
    done
    echo "== vLLM up on port $PORT (pid $(cat $PIDF), stop: kill \$(cat $PIDF))"
fi

echo "$MODEL" > "$OUT/MODEL"     # which judge model made single_local_*.jsonl here
for PROMPT in $PROMPTS; do
    echo "== $(date) $PROMPT"
    python -m judge.judge_single --run "${RUNS[@]}" --out "$OUT" --judge local --model "$MODEL" \
        --base-url "http://localhost:$PORT/v1" --prompt "$PROMPT" --limit "${LIMIT:-0}" --workers 16 ${THINK:+--think} \
        2>&1 | tee -a "$OUT/judge_${PROMPT}.log"
done
python -m judge.judge_single --out "$OUT" --report > /dev/null
echo "== $(date) done; results in $OUT (table: $OUT/single_report.md)"
