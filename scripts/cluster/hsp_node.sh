#!/bin/bash
# HSP pool (hsp/): train this node's share of the PPO learners, inside an interactive allocation (no sbatch).
# Node i of n takes the runs with index % n == i, so n allocations split the grid without talking to each other.
# Resumes: a finished run (parts/<name>.json) is skipped, a half-trained one continues from resume/ (saved
# every 5 min), so after a preemption just run the same command again.
#
#   python -m hsp.weights --n 36 --seed 0 --out results/train/ppo/hsp/rewards.txt     # once, before any node
#   srun --jobid=<id> --pty bash scripts/cluster/hsp_node.sh 0 3                       # node 0 of 3
#   srun --jobid=<id> --pty bash scripts/cluster/hsp_node.sh 1 3                       # node 1 of 3, ...
#
# Settings (environment variables, all optional):
#   OUT     run folder (default results/train/ppo/hsp; rewards.txt must be in it)
#   PAR     runs at the same time on this node (default: the allocation's CPUs; one CPU per run)
#   SEEDS   training seeds (default "0", one learner per hidden reward as in HSP)
# Training runs on the CPU (the GPU of the allocation is not used).
set -euo pipefail
cd "$(dirname "$0")/../.."
NODE=${1:?usage: hsp_node.sh <node index> <number of nodes>}
NNODES=${2:?usage: hsp_node.sh <node index> <number of nodes>}
OUT=${OUT:-results/train/ppo/hsp}
SEEDS=${SEEDS:-0}
PAR=${PAR:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source scripts/cluster/activate.sh
mkdir -p logs
[ -f "$OUT/rewards.txt" ] || { echo "no $OUT/rewards.txt: run  python -m hsp.weights --out $OUT/rewards.txt  first"; exit 1; }

echo "== $(date) on $(hostname): node $NODE of $NNODES, $PAR at a time, out $OUT"
which python && python --version
python -c "import overcooked_ai_py" 2>/dev/null || { echo "this python lacks overcooked_ai_py (wrong env; try CONDA_ENV=credit)"; exit 1; }
ARGS=(--algos ppo --rewards-file "$OUT/rewards.txt" --partners greedy --seeds $SEEDS --out "$OUT")
N=$(python -m train.run "${ARGS[@]}" --list-jobs 2>/dev/null | tail -1)
MINE=$(seq 0 $((N - 1)) | awk -v n="$NNODES" -v i="$NODE" '$1 % n == i')
echo "$N runs in the grid; this node: $(echo $MINE | wc -w) ($(echo $MINE | tr '\n' ' '))"

export OUT
printf '%s\n' $MINE | xargs -P "$PAR" -I{} bash -c \
    'python -m train.run "$@" --job-index {} --eval --no-plot >> logs/hsp_{}.log 2>&1 \
     && echo "  run {} done" || echo "  run {} FAILED (logs/hsp_{}.log)"' _ "${ARGS[@]}"
echo "== $(date) node $NODE finished; $(ls "$OUT"/parts/*.json 2>/dev/null | grep -vc eval) of $N runs trained in total"
