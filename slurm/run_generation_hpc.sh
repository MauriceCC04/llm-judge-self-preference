#!/usr/bin/env bash
# slurm/run_generation_hpc.sh — canonical HPC generation job for the frozen 512-plan study.
#
# Required env:
#   GENERATION_ARM        llm | programmatic
#   LLM_SOURCE_MODEL      required when GENERATION_ARM=llm
# Optional env:
#   GENERATION_PROFILE    exact (default) — uses generate.exact_counts shard sizes
#   GENERATION_N          override plans-per-fixture (only when GENERATION_PROFILE != exact)
#   SAMPLER_CONFIG        path to fitted priors JSON (default sampler_config.json)
#   SEED_OFFSET           integer seed offset (default 0)
#   FIXTURE_IDS           comma-separated fixture subset (default ALL 8)
#   VLLM_SOURCE_PORT      default 8773
#   VLLM_EXPLAINER_PORT   default 8774
#   CLEANUP_SOURCE_CACHE  1 (default) deletes source weights after llm generation
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
elif [[ "${GENERATION_ARM}" != "programmatic" ]]; then
    echo "[ABORT] GENERATION_ARM must be 'llm' or 'programmatic'" >&2
    exit 1
fi

# Resolve plans-per-fixture from the exact-count profile or env override.
if [[ "${GENERATION_PROFILE}" == "exact" ]]; then
    PLANS_PER_FIXTURE=$(python -c "
from generate.exact_counts import LLM_BASE_PER_FIXTURE_PER_MODEL, PROGRAMMATIC_BASE_PER_FIXTURE
print(LLM_BASE_PER_FIXTURE_PER_MODEL if '${GENERATION_ARM}' == 'llm' else PROGRAMMATIC_BASE_PER_FIXTURE)
")
else
    PLANS_PER_FIXTURE="${GENERATION_N:?Set GENERATION_N when GENERATION_PROFILE != exact}"
fi

echo "=== Generation: ${GENERATION_ARM} arm (profile=${GENERATION_PROFILE}) ==="
echo "  plans_per_fixture: ${PLANS_PER_FIXTURE}"
echo "  seed_offset:       ${SEED_OFFSET}"
echo "  fixture_ids:       ${FIXTURE_IDS:-ALL}"
echo "  explainer_model:   ${EXPLAINER_MODEL}"
[[ "${GENERATION_ARM}" == "llm" ]] && echo "  source_model:      ${LLM_SOURCE_MODEL}"
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
export OPENROUTER_API_KEY=dummy
export TRAILTRAINING_TWO_STAGE_PLAN=1

cleanup() {
    set +e
    [[ -n "${SOURCE_PID}" ]] && kill "${SOURCE_PID}" 2>/dev/null
    [[ -n "${EXPLAINER_PID}" ]] && kill "${EXPLAINER_PID}" 2>/dev/null
    sleep 5
    [[ -n "${SOURCE_PID}" ]] && kill -9 "${SOURCE_PID}" 2>/dev/null
    [[ -n "${EXPLAINER_PID}" ]] && kill -9 "${EXPLAINER_PID}" 2>/dev/null
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
    --disable-log-requests > out/vllm_explainer.log 2>&1 &
EXPLAINER_PID=$!

python -c "
from judge.vllm_server import VllmServer
from pathlib import Path
import sys
server = VllmServer('${EXPLAINER_MODEL}', ${VLLM_EXPLAINER_PORT}, log_dir=Path('out'), max_model_len=8192)
sys.exit(0 if server.health_poll(timeout_s=600, interval_s=10) else 1)
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
        --disable-log-requests > out/vllm_source.log 2>&1 &
    SOURCE_PID=$!

    python -c "
from judge.vllm_server import VllmServer
from pathlib import Path
import sys
server = VllmServer('${LLM_SOURCE_MODEL}', ${VLLM_SOURCE_PORT}, log_dir=Path('out'), max_model_len=8192)
sys.exit(0 if server.health_poll(timeout_s=900, interval_s=15) else 1)
"

    export TRAILTRAINING_SOURCE_LLM_BASE_URL="http://127.0.0.1:${VLLM_SOURCE_PORT}/v1"
fi

# compat/trailtraining_client.py uses TRAILTRAINING_LLM_BASE_URL as the generic
# fallback when stage-specific URLs aren't set. Point it at the explainer so
# that any unstaged call still has a working endpoint.
export TRAILTRAINING_LLM_BASE_URL="${TRAILTRAINING_EXPLAINER_LLM_BASE_URL}"

FIXTURE_ARGS=()
if [[ -n "${FIXTURE_IDS}" ]]; then
    FIXTURE_ARGS+=(--fixture-id "${FIXTURE_IDS}")
fi

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    python -m generate.run_generation \
        --arm llm \
        --source-model "${LLM_SOURCE_MODEL}" \
        --plans-per-fixture "${PLANS_PER_FIXTURE}" \
        --seed-offset "${SEED_OFFSET}" \
        "${FIXTURE_ARGS[@]}" \
        --output plans/
else
    if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
        echo "--- Fitting sampler priors from existing LLM-arm plans ---"
        python -m generate.fit_priors --plans-dir plans/ --output "${SAMPLER_CONFIG}"
    else
        echo "--- Using existing sampler config: ${SAMPLER_CONFIG} ---"
    fi
    python -m generate.run_generation \
        --arm programmatic \
        --plans-per-fixture "${PLANS_PER_FIXTURE}" \
        --seed-offset "${SEED_OFFSET}" \
        --sampler-config "${SAMPLER_CONFIG}" \
        "${FIXTURE_ARGS[@]}" \
        --output plans/
fi

if [[ "${GENERATION_ARM}" == "llm" && "${CLEANUP_SOURCE_CACHE}" == "1" ]]; then
    echo "--- Deleting source model weights ---"
    MODEL_DIR_NAME="models--$(echo "${LLM_SOURCE_MODEL}" | tr '/' '--')"
    rm -rf "${HF_HUB_CACHE}/${MODEL_DIR_NAME}" 2>/dev/null || true
fi

log_quota
echo "=== Generation COMPLETE: ${GENERATION_ARM} arm ==="