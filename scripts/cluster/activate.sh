# Sourced by every job script: activates the project's Python environment.
#   conda (default): the env named $CONDA_ENV (default "credit", see scripts/cluster/setup_conda.sh)
#   venv:            ./venv if it exists and CONDA_ENV is not set
_opts=$-; set +u                       # conda's shell hooks reference unset variables
if [ -z "${CONDA_ENV:-}" ] && [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    module load conda/latest 2>/dev/null
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-credit}"
fi
case $_opts in *u*) set -u ;; esac
export PYTHONUNBUFFERED=1
