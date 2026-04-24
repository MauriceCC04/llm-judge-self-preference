#!/usr/bin/env bash
# slurm/run_judge.sh — Gate 4 workhorse: one judge, full pairwise + soft-eval.
#
# Usage (submit from login node):
#   JUDGE_NAME=qwen_7b_judge sbatch --exclude=gnode04 \
#       --wrap='cd $PROJECT_ROOT && bash slurm/run_judge.sh'
#
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
export VLLM_PORT="${VLLM_PORT:-8772}"

echo "=== Judge run: ${JUDGE_NAME} ==="
log_quota
preflight_quota_gate

# Require model to be pre-cached (login-node step)
export HF_HUB_OFFLINE=1

echo "--- Resolving judge spec ---"
python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge
j = get_judge('${JUDGE_NAME}')
print(f'model_id={j.model_id}  quant={j.quant}  disk_gb={j.disk_gb}')
"

JUDGE_MODEL=$(python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge
print(get_judge('${JUDGE_NAME}').model_id)
")

JUDGE_QUANT=$(python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge
j = get_judge('${JUDGE_NAME}')
print(j.quant if j.quant != 'fp16' else '')
")

echo "--- Starting vLLM (${JUDGE_MODEL}) ---"
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model ${JUDGE_MODEL} \
    --port ${VLLM_PORT} \
    --host 127.0.0.1 \
    --max-model-len 8192 \
    --disable-log-requests"
if [[ -n "${JUDGE_QUANT}" ]]; then
    VLLM_CMD="${VLLM_CMD} --quantization ${JUDGE_QUANT}"
fi
eval "${VLLM_CMD}" &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "--- Health poll (up to 15 min) ---"
python -c "
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
s = VllmServer('${JUDGE_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
healthy = s.health_poll(timeout_s=900, interval_s=15)
if not healthy:
    print('[ABORT] vLLM failed to become healthy.', flush=True)
    import os, signal
    os.killpg(os.getpgid(${VLLM_PID}), signal.SIGKILL)
    sys.exit(1)
print('vLLM healthy')
"

export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENROUTER_API_KEY=dummy

echo "--- Running pairwise + soft-eval harness ---"
python -c "
import sys, json; sys.path.insert(0, '.')
from pathlib import Path
from judge.panel import get_judge
from judge.harness import run_pairwise_harness, run_soft_eval_harness

judge = get_judge('${JUDGE_NAME}')
pairs_file = Path('matched_pairs.json')
plans_dir = Path('plans')
fixture_dir = Path('fixtures/data')
judgments_dir = Path('judgments')
judgments_dir.mkdir(exist_ok=True)

if not pairs_file.exists():
    print('[WARN] matched_pairs.json not found — skipping pairwise')
else:
    pairs = json.loads(pairs_file.read_text())
    # Use rollups from the first fixture as a representative context
    rollups_path = None
    for pair in pairs[:1]:
        candidate = fixture_dir / pair['fixture_id'] / 'combined_rollups.json'
        if candidate.exists():
            rollups_path = candidate
            break
    run_pairwise_harness(
        pairs=pairs,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=rollups_path,
        output_path=judgments_dir / f'pairwise_{judge.name}.jsonl',
        n_runs=3,
        n_positions=2,
    )
    print(f'Pairwise done: {judge.name}')

# Soft eval on all plans
plan_ids = [p.stem for p in plans_dir.glob('*.json')
            if not p.name.endswith('.provenance.json')] if plans_dir.exists() else []
if plan_ids:
    run_soft_eval_harness(
        plan_ids=plan_ids,
        plans_dir=plans_dir,
        judge=judge,
        rollups_path=rollups_path,
        output_path=judgments_dir / f'softeval_{judge.name}.jsonl',
    )
    print(f'Soft eval done: {judge.name}')
"

echo "--- Shutting down vLLM ---"
kill "${VLLM_PID}" 2>/dev/null || true
sleep 15
kill -9 "${VLLM_PID}" 2>/dev/null || true

echo "--- Deleting model weights ---"
MODEL_DIR_NAME="models--$(echo '${JUDGE_MODEL}' | tr '/' '--')"
rm -rf "${HF_HUB_CACHE}/${MODEL_DIR_NAME}" 2>/dev/null && echo "Deleted" || echo "(not found)"

log_quota
echo "=== Judge run COMPLETE: ${JUDGE_NAME} ==="
