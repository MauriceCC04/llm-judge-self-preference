#!/usr/bin/env bash
# slurm/run_generation_hpc.sh — canonical HPC generation job for the frozen 512-plan study.
#
# Required env:
#   GENERATION_ARM        llm | programmatic
#   LLM_SOURCE_MODEL      required when GENERATION_ARM=llm
# Optional env:
#   GENERATION_PROFILE    exact (default) — uses generate.exact_counts shard sizes
#   GENERATION_N          override plans-per-fixture (only when GENERATION_PROFILE != exact)
#   PLANS_DIR            condition-local plans directory (default plans/)
#   OUTPUT_DIR           deprecated alias for PLANS_DIR
#   SAMPLER_CONFIG       path to fitted priors JSON
#                        (default: <plans-parent>/sampler_config.json)
#   SEED_OFFSET           integer seed offset (default 0)
#   FIXTURE_IDS           comma-separated fixture subset (default ALL 8)
#   SOURCE_TEMPERATURE    default 0.7 for llm arm
#   EXPLAINER_TEMPERATURE default 0.0 for both arms
#   VLLM_SOURCE_PORT      default 8773
#   VLLM_EXPLAINER_PORT   default 8774
#   GUIDED_DECODING_BACKEND xgrammar (default) or outlines
#   CLEANUP_SOURCE_CACHE  1 (default) deletes source weights after llm generation
#   CLEANUP_EXPLAINER_CACHE 1 (default) deletes explainer weights after generation
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
mkdir -p out err
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

GENERATION_ARM="${GENERATION_ARM:?Set GENERATION_ARM=llm|programmatic}"
GENERATION_PROFILE="${GENERATION_PROFILE:-exact}"
PLANS_DIR="${PLANS_DIR:-${OUTPUT_DIR:-plans/}}"
PLANS_PARENT="$(dirname "${PLANS_DIR%/}")"
SAMPLER_CONFIG="${SAMPLER_CONFIG:-${PLANS_PARENT}/sampler_config.json}"
SEED_OFFSET="${SEED_OFFSET:-0}"
FIXTURE_IDS="${FIXTURE_IDS:-}"
SOURCE_TEMPERATURE="${SOURCE_TEMPERATURE:-0.7}"
EXPLAINER_TEMPERATURE="${EXPLAINER_TEMPERATURE:-0.0}"
VLLM_SOURCE_PORT="${VLLM_SOURCE_PORT:-8773}"
VLLM_EXPLAINER_PORT="${VLLM_EXPLAINER_PORT:-8774}"
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-xgrammar}"
CLEANUP_SOURCE_CACHE="${CLEANUP_SOURCE_CACHE:-1}"
CLEANUP_EXPLAINER_CACHE="${CLEANUP_EXPLAINER_CACHE:-1}"
SOURCE_PID=""
EXPLAINER_PID=""

mkdir -p "${PLANS_DIR}"

EXPLAINER_MODEL=$(python -c "from generate.constants import EXPLAINER_MODEL_ID; print(EXPLAINER_MODEL_ID)")
python tools/check_model_cache.py "${EXPLAINER_MODEL}"

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    LLM_SOURCE_MODEL="${LLM_SOURCE_MODEL:?Set LLM_SOURCE_MODEL when GENERATION_ARM=llm}"
    python tools/check_model_cache.py "${LLM_SOURCE_MODEL}"
elif [[ "${GENERATION_ARM}" != "programmatic" ]]; then
    echo "[ABORT] GENERATION_ARM must be 'llm' or 'programmatic'" >&2
    exit 1
fi

purge_model_cache() {
    local model_id="$1"
    python -c "
from hpc.quota import purge_hf_model_cache
removed = purge_hf_model_cache('${model_id}')
print('Removed cached model: ${model_id}' if removed else 'No cached model to remove: ${model_id}')
"
}

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
echo "  plans_per_fixture:       ${PLANS_PER_FIXTURE}"
echo "  plans_dir:               ${PLANS_DIR}"
echo "  sampler_config:          ${SAMPLER_CONFIG}"
echo "  seed_offset:             ${SEED_OFFSET}"
echo "  fixture_ids:             ${FIXTURE_IDS:-ALL}"
echo "  explainer_model:         ${EXPLAINER_MODEL}"
echo "  guided_decoding_backend: ${GUIDED_DECODING_BACKEND}"
echo "  explainer_temperature:   ${EXPLAINER_TEMPERATURE}"
if [[ "${GENERATION_ARM}" == "llm" ]]; then
    echo "  source_model:            ${LLM_SOURCE_MODEL}"
    echo "  source_temperature:      ${SOURCE_TEMPERATURE}"
fi
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
export OPENROUTER_API_KEY=dummy
export TRAILTRAINING_TWO_STAGE_PLAN=1
export TRAILTRAINING_FORCE_API="${TRAILTRAINING_FORCE_API:-chat}"
export TRAILTRAINING_GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND}"

echo "--- Preflighting structured-output schemas (${TRAILTRAINING_GUIDED_DECODING_BACKEND}) ---"
python tools/preflight_schemas.py --backend "${TRAILTRAINING_GUIDED_DECODING_BACKEND}"

echo "--- Structured API mode: ${TRAILTRAINING_FORCE_API} ---"

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
    --guided-decoding-backend "${GUIDED_DECODING_BACKEND}" \
    --gpu-memory-utilization 0.20 \
    --no-enable-log-requests > out/vllm_explainer.log 2>&1 &
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
        --guided-decoding-backend "${GUIDED_DECODING_BACKEND}" \
        --gpu-memory-utilization 0.55 \
        --no-enable-log-requests > out/vllm_source.log 2>&1 &
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
        --source-temperature "${SOURCE_TEMPERATURE}" \
        --explainer-temperature "${EXPLAINER_TEMPERATURE}" \
        "${FIXTURE_ARGS[@]}" \
        --output "${PLANS_DIR}"
else
    if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
        echo "--- Fitting sampler priors from existing LLM-arm plans ---"
        python -m generate.fit_priors --plans-dir "${PLANS_DIR}" --output "${SAMPLER_CONFIG}"
    else
        echo "--- Using existing sampler config: ${SAMPLER_CONFIG} ---"
    fi
    python -m generate.run_generation \
        --arm programmatic \
        --plans-per-fixture "${PLANS_PER_FIXTURE}" \
        --seed-offset "${SEED_OFFSET}" \
        --sampler-config "${SAMPLER_CONFIG}" \
        --explainer-temperature "${EXPLAINER_TEMPERATURE}" \
        "${FIXTURE_ARGS[@]}" \
        --output "${PLANS_DIR}"
fi

if [[ "${GENERATION_ARM}" == "llm" && "${CLEANUP_SOURCE_CACHE}" == "1" ]]; then
    echo "--- Deleting source model weights ---"
    purge_model_cache "${LLM_SOURCE_MODEL}"
fi

if [[ "${CLEANUP_EXPLAINER_CACHE}" == "1" ]]; then
    echo "--- Deleting explainer model weights ---"
    purge_model_cache "${EXPLAINER_MODEL}"
fi

log_quota
echo "=== Generation COMPLETE: ${GENERATION_ARM} arm ==="
