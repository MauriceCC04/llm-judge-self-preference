#!/usr/bin/env bash
# slurm/run_generation.sh — two-vLLM batch generation job for one arm.
#
# Why two vLLMs?  trailtraining's two-stage pipeline calls a *source* model and
# a separate *explainer* model.  We must therefore expose two endpoints:
#   TRAILTRAINING_SOURCE_LLM_BASE_URL   — vLLM serving the source LLM
#   TRAILTRAINING_EXPLAINER_LLM_BASE_URL — vLLM serving Qwen2.5-3B-Instruct
#
# Layout for the LLM arm on a 40 GB MIG slice:
#   vLLM #1  source     (8B FP16)   --gpu-memory-utilization 0.55  ≈ 22 GB
#   vLLM #2  explainer  (3B FP16)   --gpu-memory-utilization 0.20  ≈  8 GB
#                                    nominal total                 ≈ 30 GB
# leaves ~10 GB headroom on the slice.
#
# Programmatic arm only needs the explainer vLLM.
#
# Required env (set before sbatch --wrap):
#   GENERATION_ARM        llm | programmatic           (required)
#   LLM_SOURCE_MODEL      e.g. meta-llama/Llama-3.1-8B-Instruct (required for llm arm)
# Optional:
#   GENERATION_N          plans per fixture (default: 16 llm, 32 programmatic)
#   SAMPLER_CONFIG        path to fitted priors JSON (programmatic arm)
#   VLLM_SOURCE_PORT      default 8773
#   VLLM_EXPLAINER_PORT   default 8774
#
# Pre-cache (login node) BEFORE submitting:
#   bash slurm/pre_cache_models.sh qwen_3b_explainer
#   bash slurm/pre_cache_models.sh llama_8b_source        # llm arm only
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

GENERATION_ARM="${GENERATION_ARM:?Set GENERATION_ARM=llm|programmatic}"
SAMPLER_CONFIG="${SAMPLER_CONFIG:-sampler_config.json}"
VLLM_SOURCE_PORT="${VLLM_SOURCE_PORT:-8773}"
VLLM_EXPLAINER_PORT="${VLLM_EXPLAINER_PORT:-8774}"
SOURCE_PID=""
EXPLAINER_PID=""

EXPLAINER_MODEL=$(python -c "
import sys; sys.path.insert(0, '.')
from generate.constants import EXPLAINER_MODEL_ID
print(EXPLAINER_MODEL_ID)
")

if [[ "${GENERATION_ARM}" == "llm" ]]; then
    GENERATION_N="${GENERATION_N:-16}"
    LLM_SOURCE_MODEL="${LLM_SOURCE_MODEL:?Set LLM_SOURCE_MODEL when GENERATION_ARM=llm}"
elif [[ "${GENERATION_ARM}" == "programmatic" ]]; then
    GENERATION_N="${GENERATION_N:-32}"
else
    echo "[ABORT] GENERATION_ARM must be 'llm' or 'programmatic', got: ${GENERATION_ARM}" >&2
    exit 1
fi

echo "=== Generation: ${GENERATION_ARM} arm, ${GENERATION_N} plans/fixture ==="
echo "Explainer model: ${EXPLAINER_MODEL}"
[[ "${GENERATION_ARM}" == "llm" ]] && echo "Source model:    ${LLM_SOURCE_MODEL}"
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
# Upstream make_openrouter_client raises if OPENROUTER_API_KEY is empty even
# when we patch the module attribute. Set a dummy to satisfy the constructor.
export OPENROUTER_API_KEY=dummy
export TRAILTRAINING_TWO_STAGE_PLAN=1

# ── Cleanup trap: always kill both vLLMs ─────────────────────────────────────
cleanup() {
    set +e
    [[ -n "${SOURCE_PID}" ]]    && kill "${SOURCE_PID}"    2>/dev/null
    [[ -n "${EXPLAINER_PID}" ]] && kill "${EXPLAINER_PID}" 2>/dev/null
    sleep 5
    [[ -n "${SOURCE_PID}" ]]    && kill -9 "${SOURCE_PID}"    2>/dev/null
    [[ -n "${EXPLAINER_PID}" ]] && kill -9 "${EXPLAINER_PID}" 2>/dev/null
    set -e
}
trap cleanup EXIT

# ── Always start the explainer vLLM ──────────────────────────────────────────
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
s = VllmServer('${EXPLAINER_MODEL}', ${VLLM_EXPLAINER_PORT}, log_dir=Path('out'), max_model_len=8192)
ok = s.health_poll(timeout_s=600, interval_s=10)
sys.exit(0 if ok else 1)
"

export TRAILTRAINING_EXPLAINER_LLM_BASE_URL="http://127.0.0.1:${VLLM_EXPLAINER_PORT}/v1"

# ── If LLM arm, also start the source vLLM ───────────────────────────────────
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
s = VllmServer('${LLM_SOURCE_MODEL}', ${VLLM_SOURCE_PORT}, log_dir=Path('out'), max_model_len=8192)
ok = s.health_poll(timeout_s=900, interval_s=15)
sys.exit(0 if ok else 1)
"

    export TRAILTRAINING_SOURCE_LLM_BASE_URL="http://127.0.0.1:${VLLM_SOURCE_PORT}/v1"
fi

# Generic fallback for any code path that only checks the generic var.
# Must point at the explainer (3B), since the explainer is the only vLLM that
# is *guaranteed* to be running under both arms.
export TRAILTRAINING_LLM_BASE_URL="${TRAILTRAINING_EXPLAINER_LLM_BASE_URL}"

# ── Generate ─────────────────────────────────────────────────────────────────
if [[ "${GENERATION_ARM}" == "llm" ]]; then
    python -m generate.run_generation \
        --arm llm \
        --plans-per-fixture "${GENERATION_N}" \
        --output plans/
else
    if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
        echo "--- Fitting sampler priors from existing LLM-arm plans ---"
        python -m generate.fit_priors \
            --plans-dir plans/ \
            --output "${SAMPLER_CONFIG}"
    else
        echo "--- Using existing sampler config: ${SAMPLER_CONFIG} ---"
    fi
    python -m generate.run_generation \
        --arm programmatic \
        --plans-per-fixture "${GENERATION_N}" \
        --sampler-config "${SAMPLER_CONFIG}" \
        --output plans/
fi

log_quota
echo "=== Generation COMPLETE: ${GENERATION_ARM} arm ==="