#!/usr/bin/env bash
# slurm/run_vllm_smoke.sh — Gate 2: spin up Qwen2.5-7B, run 5 compare_plans, delete.
#
#SBATCH --job-name=jbs_vllm_smoke
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=00:45:00
#SBATCH --output=out/vllm_smoke_%j.out
#SBATCH --error=err/vllm_smoke_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err

export SMOKE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VLLM_PORT=8771

echo "=== Gate 2: vLLM smoke test ==="
log_quota
preflight_quota_gate

# Pre-cache check (model should already be cached from login node)
export HF_HUB_OFFLINE=1

echo "--- Starting vLLM (${SMOKE_MODEL}) ---"
python -m vllm.entrypoints.openai.api_server \
    --model "${SMOKE_MODEL}" \
    --port "${VLLM_PORT}" \
    --host 127.0.0.1 \
    --max-model-len 8192 \
    --disable-log-requests &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "--- Health poll (up to 8 min) ---"
python -c "
import sys; sys.path.insert(0, '.')
from judge.vllm_server import VllmServer
from pathlib import Path
s = VllmServer('${SMOKE_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
healthy = s.health_poll(timeout_s=480, interval_s=10)
if not healthy:
    import sys; sys.exit(1)
print('vLLM healthy')
"

echo "--- Set env and run 5 compare_plans calls ---"
export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENROUTER_API_KEY=dummy

python -c "
import sys, json; sys.path.insert(0, '.')
from tests.create_test_fixtures import create_test_fixture
from pathlib import Path
import tempfile, shutil

tmp = Path(tempfile.mkdtemp())
fixture = create_test_fixture(tmp)

from trailtraining.llm.soft_eval import SoftEvalConfig, compare_plans
from trailtraining.contracts import TrainingPlanArtifact

plan_raw = json.loads((fixture / 'readiness_and_risk_forecast.json').read_text())
plan_stub = json.loads((fixture / 'combined_rollups.json').read_text())

# Use a minimal valid plan for smoke test
from tests.mock_llm_client import _training_plan_payload
plan = _training_plan_payload()
rollups = json.loads((fixture / 'combined_rollups.json').read_text())

cfg = SoftEvalConfig(
    enabled=True,
    model='${SMOKE_MODEL}',
    reasoning_effort='none',
    skip_synthesis=True,
)

ok = 0
for i in range(5):
    try:
        r = compare_plans(plan, plan, rollups=rollups, cfg=cfg)
        assert r['preferred'] in ('plan_a', 'plan_b', 'tie')
        ok += 1
        print(f'  Call {i+1}/5: preferred={r[\"preferred\"]} OK')
    except Exception as e:
        print(f'  Call {i+1}/5: FAILED — {e}')

shutil.rmtree(tmp, ignore_errors=True)
print(f'{ok}/5 calls succeeded')
if ok < 3:
    sys.exit(1)
"

echo "--- Shutting down vLLM ---"
kill "${VLLM_PID}" 2>/dev/null || true
sleep 10
kill -9 "${VLLM_PID}" 2>/dev/null || true

echo "--- Deleting model weights ---"
MODEL_DIR="${HF_HUB_CACHE}/models--$(echo '${SMOKE_MODEL}' | tr '/' '--')"
rm -rf "${MODEL_DIR}" 2>/dev/null && echo "Deleted ${MODEL_DIR}" || echo "(not found)"

log_quota
echo "=== Gate 2 PASSED ==="
