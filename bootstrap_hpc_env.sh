#!/usr/bin/env bash
# bootstrap_hpc_env.sh — minimal environment bootstrap for the canonical HPC path.
#
# Usage:
#   conda activate judge-bias
#   pip install torch vllm
#   bash bootstrap_hpc_env.sh
#
# Run from the repo root.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/pip_cache}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/torch_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/vllm_cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${VLLM_CACHE_ROOT}"

pip install -r requirements-hpc.txt

echo "Bootstrap complete."
echo "Repo root: ${ROOT_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "NOTE: bootstrap_hpc_env.sh runs in its own shell process."
echo "NOTE: export cache variables in your current shell (or source a shell preamble)"
echo "NOTE: before installs, downloads, and sbatch commands."
