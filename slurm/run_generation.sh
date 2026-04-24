#!/usr/bin/env bash
# slurm/run_generation.sh — batch generation job for one arm.
#
# Environment variables (set before sbatch --wrap):
#   GENERATION_ARM   llm | programmatic   (required)
#   GENERATION_N     plans per fixture    (default: 20)
#   SAMPLER_CONFIG   path to fitted priors JSON (programmatic arm only)
#   VLLM_PORT        port for vLLM server  (default: 8773)
#
# Example (LLM arm, 20 plans per fixture):
#   GENERATION_ARM=llm GENERATION_N=20 sbatch --exclude=gnode04 \
#       --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'
#
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

GENERATION_ARM="${GENERATION_ARM:?'Set GENERATION_ARM=llm|programmatic'}"
GENERATION_N="${GENERATION_N:-20}"
SAMPLER_CONFIG="${SAMPLER_CONFIG:-sampler_config.json}"
export VLLM_PORT="${VLLM_PORT:-8773}"

echo "=== Generation: ${GENERATION_ARM} arm, ${GENERATION_N} plans/fixture ==="
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    # Determine which source model to use based on available cache
    LLM_SOURCE_MODEL=$(python -c "
import sys; sys.path.insert(0, '.')
import os
from generate.constants import LLM_SOURCE_MODELS
from pathlib import Path
hub = Path(os.environ.get('HF_HUB_CACHE', os.path.expanduser('~/hf_cache/hub')))
for m in LLM_SOURCE_MODELS:
    tag = 'models--' + m.replace('/', '--')
    if (hub / tag).exists():
        print(m)
        sys.exit(0)
print(LLM_SOURCE_MODELS[0])
")
    echo "Using source model: ${LLM_SOURCE_MODEL}"

    echo "--- Starting vLLM (${LLM_SOURCE_MODEL}) ---"
    python -m vllm.entrypoints.openai.api_server \
        --model "${LLM_SOURCE_MODEL}" \
        --port "${VLLM_PORT}" \
        --host 127.0.0.1 \
        --max-model-len 8192 \
        --disable-log-requests &
    VLLM_PID=$!

    python -c "
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
s = VllmServer('${LLM_SOURCE_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
if not s.health_poll(timeout_s=600, interval_s=15):
    print('[ABORT] vLLM health poll timed out.')
    sys.exit(1)
print('vLLM healthy')
"

    export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENROUTER_API_KEY=dummy
    export TRAILTRAINING_TWO_STAGE_PLAN=1

    echo "--- Generating LLM plans ---"
    python -m generate.run_generation \
        --arm llm \
        --plans-per-fixture "${GENERATION_N}" \
        --output plans/

    echo "--- Shutting down vLLM ---"
    kill "${VLLM_PID}" 2>/dev/null || true; sleep 10
    kill -9 "${VLLM_PID}" 2>/dev/null || true

    MODEL_DIR="${HF_HUB_CACHE}/models--$(echo '${LLM_SOURCE_MODEL}' | tr '/' '--')"
    rm -rf "${MODEL_DIR}" 2>/dev/null && echo "Deleted ${MODEL_DIR}" || true

elif [[ "${GENERATION_ARM}" == "programmatic" ]]; then
    # Programmatic arm: start vLLM for the shared explainer model
    EXPLAINER_MODEL=$(python -c "
import sys; sys.path.insert(0, '.')
from generate.constants import EXPLAINER_MODEL_ID
print(EXPLAINER_MODEL_ID)
")
    echo "Using explainer model: ${EXPLAINER_MODEL}"

    python -m vllm.entrypoints.openai.api_server \
        --model "${EXPLAINER_MODEL}" \
        --port "${VLLM_PORT}" \
        --host 127.0.0.1 \
        --max-model-len 8192 \
        --disable-log-requests &
    VLLM_PID=$!

    python -c "
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
s = VllmServer('${EXPLAINER_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
if not s.health_poll(timeout_s=600, interval_s=15):
    print('[ABORT] vLLM health poll timed out.')
    sys.exit(1)
print('vLLM healthy')
"

    export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENROUTER_API_KEY=dummy
    export TRAILTRAINING_TWO_STAGE_PLAN=1

    # Fit priors if not already done
    if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
        echo "--- Fitting sampler priors ---"
        python -m generate.fit_priors \
            --plans-dir plans/ \
            --output "${SAMPLER_CONFIG}"
    else
        echo "--- Using existing sampler config: ${SAMPLER_CONFIG} ---"
    fi

    echo "--- Generating programmatic plans ---"
    python -m generate.run_generation \
        --arm programmatic \
        --plans-per-fixture "${GENERATION_N}" \
        --sampler-config "${SAMPLER_CONFIG}" \
        --output plans/

    echo "--- Shutting down vLLM ---"
    kill "${VLLM_PID}" 2>/dev/null || true; sleep 10
    kill -9 "${VLLM_PID}" 2>/dev/null || true

    MODEL_DIR="${HF_HUB_CACHE}/models--$(echo '${EXPLAINER_MODEL}' | tr '/' '--')"
    rm -rf "${MODEL_DIR}" 2>/dev/null && echo "Deleted ${MODEL_DIR}" || true
fi

log_quota
echo "=== Generation COMPLETE: ${GENERATION_ARM} arm ==="
