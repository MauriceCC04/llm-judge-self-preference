#!/usr/bin/env bash
# slurm/run_judge.sh — one judge, routed through the real Python CLI.
#
# Optional env:
#   JUDGE_NAME            default qwen_7b_judge
#   JUDGE_MODE            pilot | full
#   RUN_PAIRWISE          1 | 0
#   RUN_SOFT_EVAL         1 | 0
#   PAIR_LIMIT            integer limit for sharded pairwise runs
#   PLAN_LIMIT            integer limit for sharded soft-eval runs
#   REQUIRE_STYLE_GATE    1 | 0
#   STYLE_GATE_SUMMARY    path to style gate summary JSON
#   VLLM_PORT             default 8772

#SBATCH --job-name=jbs_judge
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=06:00:00
#SBATCH --output=out/judge_%x_%j.out
#SBATCH --error=err/judge_%x_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err judgments

JUDGE_NAME="${JUDGE_NAME:-qwen_7b_judge}"
JUDGE_MODE="${JUDGE_MODE:-full}"
RUN_PAIRWISE="${RUN_PAIRWISE:-1}"
RUN_SOFT_EVAL="${RUN_SOFT_EVAL:-1}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
PLAN_LIMIT="${PLAN_LIMIT:-}"
REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE:-1}"
STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY:-results/style_audit_summary.json}"
export VLLM_PORT="${VLLM_PORT:-8772}"

echo "=== Judge run: ${JUDGE_NAME} (${JUDGE_MODE}) ==="
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
export OPENROUTER_API_KEY=dummy

echo "--- Resolving judge spec ---"
python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge, assert_judge_fits_quota
judge = get_judge('${JUDGE_NAME}')
assert_judge_fits_quota(judge)
print(f'model_id={judge.model_id} quant={judge.quant} disk_gb={judge.disk_gb} time_hours={judge.time_hours}')
"

JUDGE_MODEL=$(python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge
print(get_judge('${JUDGE_NAME}').model_id)
")

JUDGE_QUANT=$(python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge
judge = get_judge('${JUDGE_NAME}')
print(judge.quant if judge.quant != 'fp16' else '')
")

echo "--- Starting vLLM (${JUDGE_MODEL}) ---"
VLLM_CMD="python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL} --port ${VLLM_PORT} --host 127.0.0.1 --max-model-len 8192 --disable-log-requests"
if [[ -n "${JUDGE_QUANT}" ]]; then
    VLLM_CMD="${VLLM_CMD} --quantization ${JUDGE_QUANT}"
fi
eval "${VLLM_CMD}" > out/vllm_judge_${JUDGE_NAME}.log 2>&1 &
VLLM_PID=$!

echo "--- Health poll (up to 15 min) ---"
python -c "
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
server = VllmServer('${JUDGE_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
sys.exit(0 if server.health_poll(timeout_s=900, interval_s=15) else 1)
"

cleanup() {
    set +e
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 15
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    set -e
}
trap cleanup EXIT

export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

CLI_ARGS=(judge --judge "${JUDGE_NAME}" --plans plans/ --pairs matched_pairs.json --output judgments/)
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
