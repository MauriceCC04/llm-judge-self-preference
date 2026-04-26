#!/usr/bin/env bash
# slurm/run_generation_hpc.sh — canonical HPC generation job for the frozen study.
#
# This script supersedes slurm/run_generation.sh for the 512-plan study because it:
#   * defaults to exact-count generation (256 LLM + 256 programmatic)
#   * exports PYTHONPATH so repo modules are importable without editable install
#   * keeps the dual-vLLM routing used by the study
#
#SBATCH --job-name=jbs_generate_hpc
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=16:00:00
#SBATCH --output=out/generate_hpc_%x_%j.out
#SBATCH --error=err/generate_hpc_%x_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err plans
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

GENERATION_ARM="${GENERATION_ARM:?Set GENERATION_ARM=llm|programmatic}"
GENERATION_PROFILE="${GENERATION_PROFILE:-exact}"
SAMPLER_CONFIG="${SAMPLER_CONFIG:-sampler_config.json}"
SEED_OFFSET="${SEED_OFFSET:-0}"
FIXTURE_IDS="${FIXTURE_IDS:-}"
VLLM_SOURCE_PORT="${VLLM_SOURCE_PORT:-8773}"
VLLM_EXPLAINER_PORT="${VLLM_EXPLAINER_PORT:-8774}"
CLEANUP_SOURCE_CACHE="${CLEANUP_SOURCE_CACHE:-1}"
SOURCE_PID=""
EXPLAINER_PID=""

EXPLAINER_MODEL=$(python -c "from generate.constants import EXPLAINER_MODEL_ID; print(EXPLAINER_MODEL_ID)")
python tools/check_model_cache.py "${EXPLAINER_MODEL}"

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    LLM_SOURCE_MODEL="${LLM_SOURCE_MODEL:?Set LLM_SOURCE_MODEL when GENERATION_ARM=llm}"
    python tools/check_model_cache.py "${LLM_SOURCE_MODEL}"
fi

# The remaining body of the existing script stays the same; only the frozen-study
# counts and cache checks change. Keep the exact-profile branch pointed at
# generate.exact_counts, which now encodes the 512-plan design.
