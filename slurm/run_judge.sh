#!/usr/bin/env bash
# slurm/run_judge_hpc.sh — canonical HPC judge job for the frozen study.
#
# Required env:
#   JUDGE_NAME            (default qwen_7b_judge)
# Optional env:
#   JUDGE_MODE            pilot | full (default full)
#   RUN_PAIRWISE          1 | 0 (default 1)
#   RUN_SOFT_EVAL         1 | 0 (default 1)
#   PAIR_LIMIT, PLAN_LIMIT integer caps for sharded runs
#   PAIRWISE_VIEW         raw_normalized | canonical_masked (default raw_normalized)
#   REQUIRE_STYLE_GATE    1 | 0 (default 1)
#   STYLE_GATE_SUMMARY    path (default results/style_audit_summary.json)
#   VLLM_PORT             default 8772
#
# NOTE: walltime is set per-judge by submit_judge_hpc.sh based on
# walltime_hours_with_buffer(judge); the SBATCH default below is a safe upper
# bound when this script is run directly without the wrapper.
#
#SBATCH --job-name=jbs_judge_hpc
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=20:00:00
#SBATCH --output=out/judge_hpc_%x_%j.out
#SBATCH --error=err/judge_hpc_%x_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err judgments results
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

JUDGE_NAME="${JUDGE_NAME:-qwen_7b_judge}"
JUDGE_MODE="${JUDGE_MODE:-full}"
RUN_PAIRWISE="${RUN_PAIRWISE:-1}"
RUN_SOFT_EVAL="${RUN_SOFT_EVAL:-1}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
PLAN_LIMIT="${PLAN_LIMIT:-}"
PAIRWISE_VIEW="${PAIRWISE_VIEW:-raw_normalized}"
REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE:-1}"
STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY:-results/style_audit_summary.json}"
export VLLM_PORT="${VLLM_PORT:-8772}"

echo "=== Judge run: ${JUDGE_NAME} (${JUDGE_MODE}, view=${PAIRWISE_VIEW}) ==="
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
export OPENROUTER_API_KEY=dummy

echo "--- Resolving judge spec ---"
python -c "
from judge.panel import get_judge, assert_judge_fits_quota
judge = get_judge('${JUDGE_NAME}')
assert_judge_fits_quota(judge)
print(f'model_id={judge.model_id} quant={judge.quant} disk_gb={judge.disk_gb} time_hours={judge.time_hours}')
"

JUDGE_MODEL=$(python -c "from judge.panel import get_judge; print(get_judge('${JUDGE_NAME}').model_id)")
JUDGE_QUANT=$(python -c "
from judge.panel import get_judge
j = get_judge('${JUDGE_NAME}')
print(j.quant if j.quant != 'fp16' else '')
")

# Verify model is cached (login-node pre-cache step is mandatory).
python tools/check_model_cache.py "${JUDGE_MODEL}"

echo "--- Starting vLLM (${JUDGE_MODEL}) on :${VLLM_PORT} ---"
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${JUDGE_MODEL} \
    --port ${VLLM_PORT} \
    --host 127.0.0.1 \
    --max-model-len 8192 \
    --disable-log-requests"
if [[ -n "${JUDGE_QUANT}" ]]; then
    VLLM_CMD="${VLLM_CMD} --quantization ${JUDGE_QUANT}"
fi
eval "${VLLM_CMD}" > "out/vllm_judge_${JUDGE_NAME}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    set +e
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 15
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    set -e
}
trap cleanup EXIT

echo "--- Health poll (up to 15 min) ---"
python -c "
from judge.vllm_server import VllmServer
from pathlib import Path
import sys
server = VllmServer('${JUDGE_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
sys.exit(0 if server.health_poll(timeout_s=900, interval_s=15) else 1)
"

export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

CLI_ARGS=(judge --judge "${JUDGE_NAME}" --plans plans/ --pairs matched_pairs.json --output judgments/ --pairwise-view "${PAIRWISE_VIEW}")
if [[ "${JUDGE_MODE}" == "pilot" ]]; then
    CLI_ARGS+=(--pilot)
fi
if [[ "${RUN_PAIRWISE}" == "0" ]]; then
    CLI_ARGS+=(--skip-pairwise)
fi
if [[ "${RUN_SOFT_EVAL}" == "0" ]]; then
    CLI_ARGS+=(--skip-soft-eval)
fi
if [[ -n "${PAIR_LIMIT}" ]]; then
    CLI_ARGS+=(--pair-limit "${PAIR_LIMIT}")
fi
if [[ -n "${PLAN_LIMIT}" ]]; then
    CLI_ARGS+=(--plan-limit "${PLAN_LIMIT}")
fi
if [[ "${REQUIRE_STYLE_GATE}" == "1" ]]; then
    CLI_ARGS+=(--require-style-gate --style-gate-summary "${STYLE_GATE_SUMMARY}")
fi

echo "--- Running CLI judge path ---"
python cli.py "${CLI_ARGS[@]}"

echo "--- Deleting model weights ---"
MODEL_DIR_NAME="models--$(echo "${JUDGE_MODEL}" | tr '/' '--')"
rm -rf "${HF_HUB_CACHE}/${MODEL_DIR_NAME}" 2>/dev/null || true

log_quota
echo "=== Judge run COMPLETE: ${JUDGE_NAME} ==="