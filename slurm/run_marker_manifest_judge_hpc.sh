#!/usr/bin/env bash
#SBATCH --job-name=jbs_manifest_judge
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=out/manifest_judge_%x_%j.out
#SBATCH --error=err/manifest_judge_%x_%j.err
#SBATCH --exclude=gnode02

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/beegfsstudents/home/3202029/llm-judge-self-preference}"
export TRAILTRAINING_REPO="${TRAILTRAINING_REPO:-/mnt/beegfsstudents/home/3202029/trailtraining}"
export PY="${PY:-/home/3202029/.conda/envs/judge-bias/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

mkdir -p out err

JUDGE_NAME="${JUDGE_NAME:?set JUDGE_NAME}"
MANIFEST="${MANIFEST:?set MANIFEST}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
VLLM_PORT="${VLLM_PORT:-8772}"
LIMIT="${LIMIT:-}"

export HF_HOME="${HF_HOME:-/mnt/beegfsstudents/home/3202029/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export OPENAI_API_KEY=dummy
export OPENROUTER_API_KEY=dummy
export TRAILTRAINING_STRUCTURED_MAX_TOKENS="${TRAILTRAINING_STRUCTURED_MAX_TOKENS:-768}"
export MARKER_MAX_TOKENS="${MARKER_MAX_TOKENS:-768}"

JUDGE_MODEL=$("$PY" - <<PY
from judge.panel import get_judge
print(get_judge("${JUDGE_NAME}").model_id)
PY
)

JUDGE_QUANT=$("$PY" - <<PY
from judge.panel import get_judge
j = get_judge("${JUDGE_NAME}")
q = "" if j.quant == "fp16" else j.quant
if q == "awq_int4":
    q = "awq"
print(q)
PY
)

echo "judge=${JUDGE_NAME}"
echo "model=${JUDGE_MODEL}"
echo "manifest=${MANIFEST}"
echo "output_dir=${OUTPUT_DIR}"
echo "temperature=${JUDGE_TEMPERATURE}"

"$PY" tools/check_model_cache.py "${JUDGE_MODEL}"

VLLM_CMD="$PY -m vllm.entrypoints.openai.api_server \
  --model ${JUDGE_MODEL} \
  --port ${VLLM_PORT} \
  --host 127.0.0.1 \
  --max-model-len 4096 \
  --no-enable-log-requests"

if [[ -n "${JUDGE_QUANT}" ]]; then
  VLLM_CMD="${VLLM_CMD} --quantization ${JUDGE_QUANT}"
fi

eval "${VLLM_CMD}" > "out/vllm_manifest_${JUDGE_NAME}.log" 2>&1 &
VLLM_PID=$!

echo "VLLM_PID=${VLLM_PID}"
echo "VLLM_LOG=out/vllm_manifest_${JUDGE_NAME}.log"
sleep 30
echo "--- initial vLLM log ---"
tail -n 120 "out/vllm_manifest_${JUDGE_NAME}.log" || true
echo "--- process check ---"
ps -p "${VLLM_PID}" -f || true

cleanup() {
  set +e
  kill "${VLLM_PID}" 2>/dev/null || true
  sleep 10
  kill -9 "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

"$PY" - <<PY
from judge.vllm_server import VllmServer
from pathlib import Path
import sys
server = VllmServer("${JUDGE_MODEL}", ${VLLM_PORT}, log_dir=Path("out"), max_model_len=4096)
sys.exit(0 if server.health_poll(timeout_s=1800, interval_s=15) else 1)
PY

export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export TRAILTRAINING_JUDGE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "$PY" tools/run_marker_manifest_pairwise.py
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --judge-name "${JUDGE_NAME}"
  --temperature "${JUDGE_TEMPERATURE}"
  --max-tokens "${MARKER_MAX_TOKENS:-768}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"

echo "done ${JUDGE_NAME}"
