#!/bin/bash
# The six judgments of GPU.md (2 pools x 3 prompts, at most 100 pairs each) inside an interactive GPU
# allocation (no sbatch): starts vLLM on the allocated GPUs, judges one after another, stops vLLM.
# Resumes: judgments_local[_<prompt>].jsonl is appended and already-judged pairs are skipped.
#
#   bash scripts/run_judge_m1.sh                 # inside the allocation, from the project root
#   PAIRS=2 bash scripts/run_judge_m1.sh         # smoke test
#
# Settings (environment variables, all optional):
#   MODEL   Hugging Face model; default Qwen/Qwen3.8-27B-FP8 on 1 GPU (the HF twin of Ollama qwen3.8:27b;
#           bf16 is ~54 GB
#           and does not fit one 48 GB L40S), Qwen/Qwen3.8-27B on 2+ GPUs. Downloaded into $HF_HOME if missing.
#   PAIRS   at most this many pairs per judgment (default 100)
#   PROMPTS which prompts to run (default "basic definition solo")
#   POOLS   which pools to run (default "need_curve_ppo need_curve_grpo")
#   FOCUS   definition / solo judge only pairs whose behaviour contains this (default theft; shadow_quit: shadow)
#   VLLM    the vllm executable (default: ../envs/credit-vllm, else the safety-eval conda env)
#   PORT    vLLM port (default 8000 + job id % 1000)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster/activate.sh
mkdir -p logs

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -gt 0 ] || { echo "no GPU here: run this inside the GPU allocation (srun --jobid=<id> --pty zsh)"; exit 1; }
if [ "$NGPU" -ge 2 ]; then MODEL=${MODEL:-Qwen/Qwen3.8-27B}; else MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}; fi
PAIRS=${PAIRS:-100}
PORT=${PORT:-$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))}
ROOT=$(cd .. && pwd)
if [ -z "${VLLM:-}" ]; then
    VLLM=$ROOT/envs/credit-vllm/bin/vllm
    [ -x "$VLLM" ] || VLLM=$ROOT/eunbiyoon_umass_edu/.conda/envs/safety-eval/bin/vllm
fi
export HF_HOME=${HF_HOME:-$ROOT/hf}
echo "== $(date) on $(hostname): $NGPU GPU(s), $MODEL, $PAIRS pairs, vllm $VLLM"

# download once (needs internet; ~31 GB for the FP8 model)
"$(dirname "$VLLM")/hf" download "$MODEL"

# the safety-eval env is CUDA 13, but ~/.cache/flashinfer holds a sampling kernel built for CUDA 12
# (libcudart.so.12 not found): use the PyTorch sampler instead
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=logs/vllm_m1.log
"$VLLM" serve "$MODEL" --tensor-parallel-size "$NGPU" --max-model-len 16384 --port "$PORT" \
    --gpu-memory-utilization 0.90 --max-num-seqs 32 --language-model-only >> "$LOG" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
until curl -sf "localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $VPID 2>/dev/null || { echo "vLLM died, see $LOG"; exit 1; }
    sleep 10
done
echo "== vLLM up on port $PORT"

for POOL in ${POOLS:-need_curve_ppo need_curve_grpo}; do
  for PROMPT in ${PROMPTS:-basic definition solo}; do
    FOCUS_ARG=()
    [ "$PROMPT" != basic ] && FOCUS_ARG=(--focus "${FOCUS:-theft}")
    echo "== $(date) $POOL $PROMPT"
    python -m judge.judge_llm --run "results/judge/$POOL" --judge local --model "$MODEL" \
        --base-url "http://localhost:$PORT/v1" --prompt "$PROMPT" --pairs "$PAIRS" ${FOCUS_ARG[@]+"${FOCUS_ARG[@]}"} \
        --workers 16 2>&1 | tee -a "logs/judge_${POOL}_${PROMPT}.log"
  done
done

python -m judge.audit --run results/judge/need_curve_ppo
python -m judge.audit --run results/judge/need_curve_grpo
