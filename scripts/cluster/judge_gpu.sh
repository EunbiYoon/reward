#!/bin/bash
# The LLM judge (vLLM, --judge local) over the matched pairs of one pool, on a preemptible GPU job of < 1 h.
# Starts vLLM, judges until 52 min, and requeues itself until every selected pair is judged
# (judgments_local[_<prompt>].jsonl is appended and resumed).
#
#   sbatch --job-name=judge_ppo_basic --export=ALL,RUN=results/rlaif/need_curve_ppo,PROMPT=basic,PAIRS=100 \
#       scripts/cluster/judge_gpu.sh
#   sbatch --job-name=judge_ppo_solo  --export=ALL,RUN=results/rlaif/need_curve_ppo,PROMPT=solo,FOCUS=individual,PAIRS=100 \
#       scripts/cluster/judge_gpu.sh
#
# Settings (environment variables, all optional):
#   RUN     the pool folder (default results/rlaif/theft)
#   PROMPT  basic | definition | solo (default basic)
#   FOCUS   only pairs where one behaviour contains this text, e.g. individual (default: every pair)
#   PAIRS   at most this many pairs, the same ones on every computer for the same --seed (default 0 = all)
#   MODEL   the Hugging Face model vLLM serves (default Qwen/Qwen3-32B, downloaded by setup_conda.sh)
#   PORT    vLLM port (default 8000 + job id % 1000, so two jobs on one node do not collide)
#SBATCH --partition=gpu-preempt-highvram
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x.log
set -u
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster/activate.sh
MODEL=${MODEL:-Qwen/Qwen3-32B}
RUN=${RUN:-results/rlaif/theft}
PROMPT=${PROMPT:-basic}
FOCUS=${FOCUS:-}
PAIRS=${PAIRS:-0}
PORT=${PORT:-$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))}
VLLM=${VLLM:-$(cd .. && pwd)/envs/credit-vllm/bin/vllm}   # the env made by scripts/cluster/setup_conda.sh
export HF_HOME=${HF_HOME:-$(cd .. && pwd)/hf}             # where setup_conda.sh put the model
START=$(date +%s)
echo "== $(date) on $(hostname), restart ${SLURM_RESTART_COUNT:-0}: $RUN prompt=$PROMPT focus=${FOCUS:-all} pairs=$PAIRS"

remaining() {   # selected pairs not yet judged
    python - "$RUN" "$PROMPT" "$FOCUS" "$PAIRS" <<'PY'
import os, sys
from rlaif.collect import load_pool
from rlaif.common import judge_path, pending_pairs
run, prompt, focus, pairs = sys.argv[1], sys.argv[2], sys.argv[3] or None, int(sys.argv[4])
tag = "local" if prompt == "basic" else f"local_{prompt}"
todo, _ = pending_pairs(load_pool(os.path.join(run, "pool")), judge_path(run, tag), pairs, 0, focus)
print(len(todo))
PY
}
N=$(remaining) || { echo "cannot read $RUN/pool"; exit 1; }
[ "$N" -eq 0 ] && { echo "all pairs judged"; exit 0; }
echo "== $N pairs to judge"

"$VLLM" serve "$MODEL" --tensor-parallel-size 2 --max-model-len 16384 --port "$PORT" \
    --gpu-memory-utilization 0.90 >> "logs/vllm_${SLURM_JOB_NAME:-judge}.log" 2>&1 &
VPID=$!
until curl -sf "localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $VPID 2>/dev/null || { echo "vLLM died, see logs/vllm_${SLURM_JOB_NAME:-judge}.log"; exit 1; }
    sleep 10
done
echo "== vLLM up after $(( $(date +%s) - START )) s"
LEFT=$(( 52 * 60 - ($(date +%s) - START) ))
FOCUS_ARG=()
[ -n "$FOCUS" ] && FOCUS_ARG=(--focus "$FOCUS")
timeout "$LEFT" python -m rlaif.judge_llm --run "$RUN" --judge local --model "$MODEL" \
    --base-url "http://localhost:$PORT/v1" --prompt "$PROMPT" --pairs "$PAIRS" ${FOCUS_ARG[@]+"${FOCUS_ARG[@]}"} \
    --workers 64
kill $VPID; sleep 5
N=$(remaining)
echo "== $N pairs left"
if [ "$N" -gt 0 ]; then
    scontrol requeue "$SLURM_JOB_ID"
fi
