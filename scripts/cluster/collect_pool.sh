#!/bin/bash
# A persona pool with the SAME layout for PPO and GRPO, for the single-episode judge (judge/judge_single.py):
#   seat 0 = theft / pickup / shadow, each with training seeds s0, s1, s2, plus greedy (reference);
#   seat 1 = greedy; every behaviour plays the same EPISODES start states (pool seeds 0 .. EPISODES-1).
#   -> per algo: 3 personas x 3 seeds x EPISODES + EPISODES greedy trajectories (200 at 20), each with its Shapley s.
#   No fidget: the busy-but-useless case is the trained pickup persona.
# CPU only. One folder per algo, named by the time its collection starts (YYYYMMDD_HHMMSS):
#   results/judge/<stamp>/ALGO (ppo|grpo), BENCH, ckpts/*.pt (copied from $SRC),
#   pool/{meta.jsonl,trajs/,config.json}; the judge later writes its files into the same folder.
#
#   sbatch scripts/cluster/collect_pool.sh
#   sbatch --export=ALL,EPISODES=2,ALGOS=ppo --time=00:20:00 scripts/cluster/collect_pool.sh   # smoke test
#
#   BENCH     kitchen (default, Overcooked) | warehouse (RWARE) | foraging (LBF)
#   SRC       folder with <algo>_<reward>_greedy_s<seed>.pt (default: results/ckpts/<algo> for kitchen,
#             results/bench_<BENCH>/models for the others, as written by scripts/cluster/submit_bench.sh)
#   EPISODES  start states per behaviour (default 20)
#   ALGOS     default "ppo grpo"
#   WORKERS   parallel episodes (default $SLURM_CPUS_PER_TASK, else 1)
# Also runs with plain bash inside an allocation:  WORKERS=8 bash scripts/cluster/collect_pool.sh
#   EXTRA     builtin behaviours besides the personas (default "greedy")
# Judge it with  bash scripts/run_judge_single.sh <ppo|grpo> <basic|definition|solo>  (the newest folder of that algo).
#SBATCH --job-name=collect_pool
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$(dirname "$0")/../.." 2>/dev/null || true
[ -f judge/collect.py ] || cd "${SLURM_SUBMIT_DIR:?run from the project root}"   # sbatch runs a spooled copy
export CONDA_ENV=${CONDA_ENV:-credit}
source scripts/cluster/activate.sh
hostname; date; module list 2>&1 || true
which python && python --version
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EPISODES=${EPISODES:-20}
BENCH=${BENCH:-kitchen}

for ALGO in ${ALGOS:-ppo grpo}; do
  ROOT=results/judge/$(date +%Y%m%d_%H%M%S)
  while [ -e "$ROOT" ]; do sleep 1; ROOT=results/judge/$(date +%Y%m%d_%H%M%S); done
  CK=$ROOT/ckpts
  mkdir -p "$CK"
  echo "$ALGO" > "$ROOT/ALGO"
  echo "$BENCH" > "$ROOT/BENCH"
  if [ "$BENCH" = kitchen ]; then FROM=${SRC:-results/ckpts/$ALGO}; else FROM=${SRC:-results/bench_$BENCH/models}; fi
  for R in theft pickup shadow; do
    for S in 0 1 2; do
      F=${ALGO}_${R}_greedy_s$S.pt
      cp "$FROM/$F" "$CK/$F"
    done
  done
  ls -l "$CK"
  echo "== $(date) collect $BENCH $ALGO: $EPISODES episodes -> $ROOT"
  python -m judge.collect --env "$BENCH" --partners greedy --behaviors ${EXTRA:-greedy} \
      --policies "$CK/*.pt" --episodes "$EPISODES" --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}" --out "$ROOT"
  python - "$ROOT/pool/meta.jsonl" <<'EOF'
import collections, json, sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for b, n in sorted(collections.Counter(r["behavior"] for r in rs).items()):
    print(f"  {b:34s} {n}")
print(f"  total {len(rs)}")
EOF
done
echo "== $(date) done"
