#!/bin/bash
# H1: the local LLM judge over every matched pool pair, on a preemptible GPU job of < 1 h.
# Starts vLLM, judges until 52 min, and requeues itself until every pair is judged
# (judgments_local.jsonl is appended and resumed). Submitted by scripts/cluster/submit.sh.
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
VLLM=${VLLM:-$(cd .. && pwd)/envs/credit-vllm/bin/vllm}   # the env made by scripts/cluster/setup_conda.sh
export HF_HOME=${HF_HOME:-$(cd .. && pwd)/hf}             # where setup_conda.sh put the model
START=$(date +%s)
echo "== $(date) on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"

remaining() {   # pairs not yet judged
    python - "$RUN" <<'PY'
import json, os, sys
from rlaif.collect import load_pool
from rlaif.judge import judge_path, make_pairs
run = sys.argv[1]
done = set()
p = judge_path(run, "local")
if os.path.exists(p):
    done = {r["pair"] for r in map(json.loads, filter(str.strip, open(p))) if "error" not in r}
print(sum(f"{x['id']}|{y['id']}" not in done for x, y in make_pairs(load_pool(os.path.join(run, "pool")), 0)))
PY
}
[ "$(remaining)" -eq 0 ] && { echo "all pairs judged"; exit 0; }

"$VLLM" serve "$MODEL" --tensor-parallel-size 2 --max-model-len 16384 --port 8000 \
    --gpu-memory-utilization 0.90 >> logs/vllm.log 2>&1 &
VPID=$!
until curl -sf localhost:8000/v1/models > /dev/null; do
    kill -0 $VPID 2>/dev/null || { echo "vLLM died, see logs/vllm.log"; exit 1; }
    sleep 10
done
echo "== vLLM up after $(( $(date +%s) - START )) s"
LEFT=$(( 52 * 60 - ($(date +%s) - START) ))
timeout "$LEFT" python -m rlaif.judge --run "$RUN" --judge local --model "$MODEL" \
    --base-url http://localhost:8000/v1 --pairs 0 --workers 64
kill $VPID; sleep 5
N=$(remaining)
echo "== $N pairs left"
if [ "$N" -gt 0 ]; then
    scontrol requeue "$SLURM_JOB_ID"
fi
