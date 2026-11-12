#!/bin/bash
# One-time setup on a Unity login node (needs internet), from the project root:
#   bash scripts/cluster/setup_conda.sh            # both envs
#   bash scripts/cluster/setup_conda.sh credit     # only the project env (enough for the CPU experiments)
#   credit        named env (conda's default envs dir), Python 3.12 + requirements.txt, CPU PyTorch
#                 (training and evaluation run on CPU; ~2 GB)
#   credit-vllm   ../envs/credit-vllm next to the project (~10 GB, kept out of $HOME): vLLM with CUDA
#                 PyTorch, serves the local judge model on a GPU node (H1)
# The judge model is downloaded into $HF_HOME (default: ../hf, ~64 GB) by the last step.
set -e                                   # not -u: conda's shell hooks use unset variables
WHAT=${1:-all}
module load conda/latest
eval "$(conda shell.bash hook)"

if [ "$WHAT" = all ] || [ "$WHAT" = credit ]; then
    conda env list | grep -qE '^credit\s' || conda create -y -n credit python=3.12 pip
    conda activate credit
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python -c "import torch, numpy, pandas, overcooked_ai_py, rware, lbforaging; print('credit env ok, torch', torch.__version__)"
    python -c "import alem" 2>/dev/null && echo "alem ok" || echo "WARNING: alem-env did not import (only --env alem needs it)"
    conda deactivate
fi

if [ "$WHAT" = all ] || [ "$WHAT" = vllm ]; then
    VENV=$(cd .. && pwd)/envs/credit-vllm
    [ -x "$VENV/bin/python" ] || conda create -y -p "$VENV" python=3.12 pip
    conda activate "$VENV"
    pip install --upgrade pip
    pip install vllm "huggingface_hub[cli]"
    export HF_HOME=${HF_HOME:-$(cd .. && pwd)/hf}
    hf download "${MODEL:-Qwen/Qwen3-32B}" || huggingface-cli download "${MODEL:-Qwen/Qwen3-32B}"
    echo "credit-vllm env ok; model in $HF_HOME (export HF_HOME=$HF_HOME before scripts/cluster/submit.sh)"
fi
