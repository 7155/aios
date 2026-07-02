#!/usr/bin/env bash
# Source this file before running AIOS lessons in an interactive shell:
#   source scripts/activate_aios.sh

export VIRTUAL_ENV="${VIRTUAL_ENV:-/home/codex/ai/venvs/minimind}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HOME="${HF_HOME:-/home/codex/ai/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"

echo "AIOS environment ready"
echo "  python:    $VIRTUAL_ENV/bin/python"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  HF_HOME:   $HF_HOME"
