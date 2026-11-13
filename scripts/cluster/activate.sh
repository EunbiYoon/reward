# Sourced by every job script: activates the project's Python environment.
#   conda (default): the env named $CONDA_ENV (default "credit", see scripts/cluster/setup_conda.sh)
#   local conda env: ./env (made by `bash scripts/cluster/setup_conda.sh local`) if it exists
#   venv:            ./venv if it exists and CONDA_ENV is not set
_opts=$-; set +u                       # conda's shell hooks reference unset variables
if [ -x env/bin/python ] && [ -z "${USE_NAMED_CONDA_ENV:-}" ]; then
    # the project's own env: put it first on PATH directly (no `conda activate`, so another conda
    # install or an already active env in the calling shell cannot override it)
    export PATH="$PWD/env/bin:$PATH" CONDA_PREFIX="$PWD/env"
    hash -r 2>/dev/null
elif [ -z "${CONDA_ENV:-}" ] && [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    module load conda/latest 2>/dev/null
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-credit}"
fi
case $_opts in *u*) set -u ;; esac
export PYTHONUNBUFFERED=1
