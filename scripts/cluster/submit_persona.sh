#!/bin/bash
# Persona grid on Unity: A (vs greedy, team reward) and B1-B3 (vs a frozen A), for PPO, GRPO and SAC.
# PPO-A and GRPO-A already exist (results/full/models/{ppo,grpo}_team_greedy_s0.pt) and are not retrained.
#
#   cd /scratch/workspace/eunbiyoon_umass_edu-paper/credit && bash scripts/cluster/submit_persona.sh
#
#   sac_A      SAC  vs greedy / stay, team reward, seeds 0-2                         6 tasks
#   ppo_B      PPO  vs frozen PPO-A,  rewards team (B1) theft (B2) pickup (B3)  9 tasks
#   grpo_B     GRPO vs frozen GRPO-A, same three rewards                             9 tasks
#   sac_B      SAC  vs frozen SAC-A (seed 0), same three rewards; starts after sac_A  9 tasks
# One run per array task on cpu-preempt (< 1 h, requeued on preemption, PPO/GRPO resume from checkpoints).
# Watch: squeue --me    tail logs/<name>_<task>.log    Results: results/<name>/results.csv, summary.md
set -eu
mkdir -p logs
python -c "import overcooked_ai_py, torch" 2>/dev/null || { echo "activate the env first: source scripts/cluster/activate.sh (and finish setup_conda.sh)"; exit 1; }
MAIL=${MAIL:-}

array() {        # array <name> [--dependency=...] -- <train.run grid args...>
    local name=$1; shift
    local dep=()
    if [ "$1" != "--" ]; then dep=("$1"); shift; fi
    shift
    local n
    n=$(python -m train.run "$@" --out "results/$name" --list-jobs | tail -n 1)
    local id
    id=$(sbatch --parsable --job-name="$name" --mem=4G --array=0-$((n - 1)) ${dep[@]+"${dep[@]}"} \
         scripts/cluster/run_task.sh "$name" "$@")
    echo "$name: $n runs, job $id" >&2
    echo "$id"
}

for f in results/full/models/ppo_team_greedy_s0.pt results/full/models/grpo_team_greedy_s0.pt; do
    [ -f "$f" ] || { echo "missing $f (copy results/full/models from the Mac first)"; exit 1; }
done

B_REWARDS="team theft pickup"
SAC_A=$(array sac_A -- --algos sac --rewards team --partners greedy stay --seeds 0 1 2)
PPO_B=$(array ppo_B -- --algos ppo --rewards $B_REWARDS --partners ckpt:results/full/models/ppo_team_greedy_s0.pt --seeds 0 1 2)
GRPO_B=$(array grpo_B -- --algos grpo --rewards $B_REWARDS --partners ckpt:results/full/models/grpo_team_greedy_s0.pt --seeds 0 1 2)
SAC_B=$(array sac_B --dependency=afterok:"$SAC_A" -- --algos sac --rewards $B_REWARDS --partners ckpt:results/sac_A/models/sac_team_greedy_s0.pt --seeds 0 1 2)

MAILOPT=()
[ -n "$MAIL" ] && MAILOPT=(--mail-type=END,FAIL --mail-user="$MAIL")
F=$(sbatch --parsable --job-name=persona_summary --partition=cpu-preempt --mem=4G --time=00:20:00 \
    ${MAILOPT[@]+"${MAILOPT[@]}"} --output=logs/persona_summary.log \
    --dependency=afterany:$SAC_A:$PPO_B:$GRPO_B:$SAC_B \
    --wrap="bash -c 'source scripts/cluster/activate.sh && for d in sac_A ppo_B grpo_B sac_B; do python -m eval.summarize results/\$d > /dev/null && echo results/\$d/summary.md; done'")
echo "persona_summary: job $F (after everything)" >&2
