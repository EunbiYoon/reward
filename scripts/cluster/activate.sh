# Sourced by every job script: activates the project's Python environment.
#   conda (default): the env named $CONDA_ENV (default "credit", see scripts/cluster/setup_conda.sh)
#   local conda env: ./env (made by `bash scripts/cluster/setup_conda.sh local`) if it exists
#   venv:            ./venv if it exists and CONDA_ENV is not set
_opts=$-; set +eu                      # conda's shell hooks reference unset variables, and a broken hook
                                       # left by another env (e.g. a deleted project's activate.d) must not stop us
# CONDA_ENV=... in the project's .env (git-ignored) wins over whatever the calling shell has set,
# so another project's shell setup cannot pick the wrong env. Only that one line is read from .env.
if [ -f .env ]; then
    _env=$(sed -n 's/^[[:space:]]*CONDA_ENV=["'\'']\{0,1\}\([^"'\'' #]*\).*/\1/p' .env | tail -1)
    [ -n "$_env" ] && export CONDA_ENV="$_env"
    unset _env
fi
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
case $_opts in *e*) set -e ;; esac
export PYTHONUNBUFFERED=1
