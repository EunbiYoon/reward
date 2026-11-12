#!/bin/bash
# Submit every remaining experiment as preemptible jobs of < 1 h each, from the project root:
#
#   cd <project> && source scripts/cluster/activate.sh && bash scripts/cluster/submit.sh                 # everything
#   bash scripts/cluster/submit.sh cpu       # only the CPU grids (no LLM judge)
#   bash scripts/cluster/submit.sh h1h2      # only the judge -> reward models -> RLAIF training chain
#
# Each training run is one array task (1 core). A preempted task is requeued by Slurm and resumes
# from its checkpoint; a run longer than 55 min requeues itself. Finished runs are never redone.
# Watch:   squeue --me      tail logs/<name>_<task>.log      wc -l results/*/results.csv
# First time: bash scripts/cluster/setup_conda.sh (conda envs + judge model).
set -eu
WHAT=${1:-all}
MAIL=${MAIL:-}   # e.g. MAIL=you@umass.edu bash scripts/cluster/submit.sh : one e-mail when everything is done
mkdir -p logs
DEPS=()          # job ids the finishing job waits for
NAMES=()

array() {        # array <name> <mem> [--dependency=...] -- <eval.run grid args...>
    local name=$1 mem=$2; shift 2
    local dep=()
    if [ "$1" != "--" ]; then dep=("$1"); shift; fi
    shift        # the --
    local n
    n=$(python -m eval.run "$@" --out "results/$name" --list-jobs | tail -n 1)
    local id
    id=$(sbatch --parsable --job-name="$name" --mem="$mem" --array=0-$((n - 1)) ${dep[@]+"${dep[@]}"} \
         scripts/cluster/run_task.sh "$name" "$@")
    echo "$name: $n runs, job $id"
    DEPS+=("$id"); NAMES+=("$name")
}

if [ "$WHAT" = all ] || [ "$WHAT" = cpu ]; then
    array rware_5m  4G -- --env warehouse --rewards team individual pickup --env-steps 5000000  # 36, ~3 h each: requeues
    array alem_full 8G -- --env alem --rewards team individual pickup                           # 36, ~1 h each
    array h7_bonus  4G -- --rewards pickup --partners greedy --pickup-bonus 0 1 3 5 10          # 30
    array h8_noise  4G -- --rewards team individual --partners greedy --partner-noise 0 0.1 0.25 0.5   # 48
    array h10_cf    4G -- --rewards cf --partners stay greedy                                   # 12
    array h12_aux   4G -- --rewards individual team --partners greedy --aux-coef 0 0.1 0.5 1.0 \
                           --seeds 0 1 2 3 4 5 6 7 8 9                                          # 160
fi

if [ "$WHAT" = all ] || [ "$WHAT" = h1h2 ]; then
    J=$(sbatch --parsable --job-name=h1_judge scripts/cluster/judge_gpu.sh)
    R=$(sbatch --parsable --job-name=reward_models --dependency=afterok:"$J" scripts/cluster/reward_models.sh)
    echo "h1_judge: job $J (GPU), reward_models: job $R (after the judge)"
    RM=results/rlaif/theft
    array rlaif_kitchen        4G --dependency=afterok:"$R" -- --rewards learned team individual --reward-model $RM/rm_local.pt   # 36
    array rlaif_kitchen_oracle 4G --dependency=afterok:"$R" -- --rewards learned --reward-model $RM/rm_shapley.pt               # 12
fi

MAILOPT=()
[ -n "$MAIL" ] && MAILOPT=(--mail-type=END,FAIL --mail-user="$MAIL")
F=$(sbatch --parsable --job-name=finish ${MAILOPT[@]+"${MAILOPT[@]}"} --dependency=afterany:$(IFS=:; echo "${DEPS[*]}") scripts/cluster/finish.sh "${NAMES[@]}")
echo "finish (summaries, H3 oversight, audit): job $F, after everything"
