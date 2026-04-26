#!/usr/bin/env bash
# slurm/run_generation_hpc.sh — canonical HPC generation job for the frozen study.
#
# This script supersedes slurm/run_generation.sh for the 500-plan study because it:
#   * defaults to exact-count generation (250 LLM + 250 programmatic)
#   * exports PYTHONPATH so repo modules are importable without editable install
#   * keeps the dual-vLLM routing used by the study
#
# Required env:
#   GENERATION_ARM          llm | programmatic
#   LLM_SOURCE_MODEL        required when GENERATION_ARM=llm
# Optional env:
#   GENERATION_PROFILE      exact | default   (default: exact)
#   GENERATION_N            used only when GENERATION_PROFILE=default
#   SAMPLER_CONFIG          programmatic sampler priors JSON
#   SEED_OFFSET             used only when GENERATION_PROFILE=default
#   FIXTURE_IDS             used only when GENERATION_PROFILE=default
#   VLLM_SOURCE_PORT        default 8773
#   VLLM_EXPLAINER_PORT     default 8774
#   CLEANUP_SOURCE_CACHE    1 to delete source weights after llm generation
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

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    LLM_SOURCE_MODEL="${LLM_SOURCE_MODEL:?Set LLM_SOURCE_MODEL when GENERATION_ARM=llm}"
fi
if [[ "${GENERATION_PROFILE}" != "exact" && "${GENERATION_PROFILE}" != "default" ]]; then
    echo "[ABORT] GENERATION_PROFILE must be 'exact' or 'default', got: ${GENERATION_PROFILE}" >&2
    exit 1
fi

echo "=== HPC Generation: ${GENERATION_ARM} arm ==="
echo "profile=${GENERATION_PROFILE}"
echo "explainer_model=${EXPLAINER_MODEL}"
[[ "${GENERATION_ARM}" == "llm" ]] && echo "source_model=${LLM_SOURCE_MODEL}"
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
export OPENROUTER_API_KEY=dummy
export TRAILTRAINING_TWO_STAGE_PLAN=1

cleanup() {
    set +e
    [[ -n "${SOURCE_PID}" ]] && kill "${SOURCE_PID}" 2>/dev/null || true
    [[ -n "${EXPLAINER_PID}" ]] && kill "${EXPLAINER_PID}" 2>/dev/null || true
    sleep 5
    [[ -n "${SOURCE_PID}" ]] && kill -9 "${SOURCE_PID}" 2>/dev/null || true
    [[ -n "${EXPLAINER_PID}" ]] && kill -9 "${EXPLAINER_PID}" 2>/dev/null || true
    set -e
}
trap cleanup EXIT

echo "--- Starting explainer vLLM (${EXPLAINER_MODEL}) on :${VLLM_EXPLAINER_PORT} ---"
python -m vllm.entrypoints.openai.api_server \
    --model "${EXPLAINER_MODEL}" \
    --port "${VLLM_EXPLAINER_PORT}" \
    --host 127.0.0.1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.20 \
    --disable-log-requests > out/vllm_explainer_hpc.log 2>&1 &
EXPLAINER_PID=$!

python -c "
from judge.vllm_server import VllmServer
from pathlib import Path
server = VllmServer('${EXPLAINER_MODEL}', ${VLLM_EXPLAINER_PORT}, log_dir=Path('out'), max_model_len=8192)
raise SystemExit(0 if server.health_poll(timeout_s=600, interval_s=10) else 1)
"

export TRAILTRAINING_EXPLAINER_LLM_BASE_URL="http://127.0.0.1:${VLLM_EXPLAINER_PORT}/v1"

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    echo "--- Starting source vLLM (${LLM_SOURCE_MODEL}) on :${VLLM_SOURCE_PORT} ---"
    python -m vllm.entrypoints.openai.api_server \
        --model "${LLM_SOURCE_MODEL}" \
        --port "${VLLM_SOURCE_PORT}" \
        --host 127.0.0.1 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.55 \
        --disable-log-requests > out/vllm_source_hpc.log 2>&1 &
    SOURCE_PID=$!

    python -c "
from judge.vllm_server import VllmServer
from pathlib import Path
server = VllmServer('${LLM_SOURCE_MODEL}', ${VLLM_SOURCE_PORT}, log_dir=Path('out'), max_model_len=8192)
raise SystemExit(0 if server.health_poll(timeout_s=900, interval_s=15) else 1)
"

    export TRAILTRAINING_SOURCE_LLM_BASE_URL="http://127.0.0.1:${VLLM_SOURCE_PORT}/v1"
fi

export TRAILTRAINING_LLM_BASE_URL="${TRAILTRAINING_EXPLAINER_LLM_BASE_URL}"

if [[ "${GENERATION_PROFILE}" == "exact" ]]; then
    echo "--- Running exact-count generation profile ---"
    if [[ "${GENERATION_ARM}" == "llm" ]]; then
        python -m generate.exact_counts \
            --arm llm \
            --source-model "${LLM_SOURCE_MODEL}" \
            --output plans/
    else
        if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
            echo "--- Fitting sampler priors from existing LLM-arm plans ---"
            python -m generate.fit_priors --plans-dir plans/ --output "${SAMPLER_CONFIG}"
        else
            echo "--- Using existing sampler config: ${SAMPLER_CONFIG} ---"
        fi
        python -m generate.exact_counts \
            --arm programmatic \
            --sampler-config "${SAMPLER_CONFIG}" \
            --output plans/
    fi
else
    echo "--- Running legacy/default generation profile ---"
    FIXTURE_ARGS=()
    if [[ -n "${FIXTURE_IDS}" ]]; then
        FIXTURE_ARGS+=(--fixture-id "${FIXTURE_IDS}")
    fi

    if [[ "${GENERATION_ARM}" == "llm" ]]; then
        GENERATION_N="${GENERATION_N:-15}"
        python -m generate.run_generation \
            --arm llm \
            --source-model "${LLM_SOURCE_MODEL}" \
            --plans-per-fixture "${GENERATION_N}" \
            --seed-offset "${SEED_OFFSET}" \
            "${FIXTURE_ARGS[@]}" \
            --output plans/
    else
        GENERATION_N="${GENERATION_N:-31}"
        if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
            echo "--- Fitting sampler priors from existing LLM-arm plans ---"
            python -m generate.fit_priors --plans-dir plans/ --output "${SAMPLER_CONFIG}"
        else
            echo "--- Using existing sampler config: ${SAMPLER_CONFIG} ---"
        fi
        python -m generate.run_generation \
            --arm programmatic \
            --plans-per-fixture "${GENERATION_N}" \
            --seed-offset "${SEED_OFFSET}" \
            --sampler-config "${SAMPLER_CONFIG}" \
            "${FIXTURE_ARGS[@]}" \
            --output plans/
    fi
fi

if [[ "${GENERATION_ARM}" == "llm" && "${CLEANUP_SOURCE_CACHE}" == "1" ]]; then
    echo "--- Deleting source model weights ---"
    MODEL_DIR_NAME="models--$(echo "${LLM_SOURCE_MODEL}" | tr '/' '--')"
    rm -rf "${HF_HUB_CACHE}/${MODEL_DIR_NAME}" 2>/dev/null || true
fi

log_quota
echo "=== HPC generation complete: ${GENERATION_ARM} arm ==="
