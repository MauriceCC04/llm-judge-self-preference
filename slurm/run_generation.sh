#!/usr/bin/env bash
# slurm/run_generation.sh — two-vLLM batch generation job for one arm.
#
# Required env:
#   GENERATION_ARM        llm | programmatic
#   LLM_SOURCE_MODEL      required for llm arm
# Optional env:
#   GENERATION_N          plans per fixture
#   SAMPLER_CONFIG        path to fitted priors JSON
#   SEED_OFFSET           integer offset for plan ids / seeds
#   FIXTURE_IDS           comma-separated fixture ids for shard runs
#   VLLM_SOURCE_PORT      default 8773
#   VLLM_EXPLAINER_PORT   default 8774
#   CLEANUP_SOURCE_CACHE  1 to delete source weights after llm generation

#SBATCH --job-name=jbs_generate
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=12:00:00
#SBATCH --output=out/generate_%x_%j.out
#SBATCH --error=err/generate_%x_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err plans

GENERATION_ARM="${GENERATION_ARM:?Set GENERATION_ARM=llm|programmatic}"
SAMPLER_CONFIG="${SAMPLER_CONFIG:-sampler_config.json}"
SEED_OFFSET="${SEED_OFFSET:-0}"
FIXTURE_IDS="${FIXTURE_IDS:-}"
VLLM_SOURCE_PORT="${VLLM_SOURCE_PORT:-8773}"
VLLM_EXPLAINER_PORT="${VLLM_EXPLAINER_PORT:-8774}"
CLEANUP_SOURCE_CACHE="${CLEANUP_SOURCE_CACHE:-1}"
SOURCE_PID=""
EXPLAINER_PID=""

EXPLAINER_MODEL=$(python -c "
import sys; sys.path.insert(0, '.')
from generate.constants import EXPLAINER_MODEL_ID
print(EXPLAINER_MODEL_ID)
")

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    GENERATION_N="${GENERATION_N:-15}"
    LLM_SOURCE_MODEL="${LLM_SOURCE_MODEL:?Set LLM_SOURCE_MODEL when GENERATION_ARM=llm}"
elif [[ "${GENERATION_ARM}" == "programmatic" ]]; then
    GENERATION_N="${GENERATION_N:-31}"
else
    echo "[ABORT] GENERATION_ARM must be 'llm' or 'programmatic', got: ${GENERATION_ARM}" >&2
    exit 1
fi

echo "=== Generation: ${GENERATION_ARM} arm ==="
echo "plans_per_fixture=${GENERATION_N}"
echo "seed_offset=${SEED_OFFSET}"
echo "fixture_ids=${FIXTURE_IDS:-ALL}"
echo "explainer_model=${EXPLAINER_MODEL}"
[[ "${GENERATION_ARM}" == "llm" ]] && echo "source_model=${LLM_SOURCE_MODEL}"
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
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
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
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
server = VllmServer('${LLM_SOURCE_MODEL}', ${VLLM_SOURCE_PORT}, log_dir=Path('out'), max_model_len=8192)
sys.exit(0 if server.health_poll(timeout_s=900, interval_s=15) else 1)
"

    export TRAILTRAINING_SOURCE_LLM_BASE_URL="http://127.0.0.1:${VLLM_SOURCE_PORT}/v1"
fi

export TRAILTRAINING_LLM_BASE_URL="${TRAILTRAINING_EXPLAINER_LLM_BASE_URL}"

FIXTURE_ARGS=()
if [[ -n "${FIXTURE_IDS}" ]]; then
    FIXTURE_ARGS+=(--fixture-id "${FIXTURE_IDS}")
fi

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    python -m generate.run_generation \
        --arm llm \
        --source-model "${LLM_SOURCE_MODEL}" \
        --plans-per-fixture "${GENERATION_N}" \
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
        --plans-per-fixture "${GENERATION_N}" \
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
