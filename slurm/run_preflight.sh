#!/usr/bin/env bash
# slurm/run_preflight.sh — Gate 1: HPC preflight checks (~5 min).
#
#SBATCH --job-name=jbs_preflight
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=00:10:00
#SBATCH --output=out/preflight_%j.out
#SBATCH --error=err/preflight_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err

echo "=== Gate 1: HPC Preflight ==="
log_quota
preflight_quota_gate

echo "--- Frozen dependency surface ---"
python tools/verify_dependency_surface.py

echo "--- Python imports ---"
python -c "
import trailtraining, openai, pydantic, torch, vllm
print('imports OK')
print('trailtraining:', trailtraining.__file__)
print('vllm:', getattr(vllm, '__version__', 'unknown'))
"

echo "--- vLLM entrypoint ---"
python -m vllm.entrypoints.openai.api_server --help >/dev/null

echo "--- GPU visibility ---"
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"

echo "--- Frozen study summary ---"
python -c "
import sys; sys.path.insert(0, '.')
from generate.study_manifest import frozen_study_summary
print(frozen_study_summary())
"

echo "--- Required model caches ---"
python tools/check_model_cache.py Qwen/Qwen2.5-3B-Instruct

echo "--- Mock LLM call ---"
python -c "
import sys; sys.path.insert(0, '.')
from tests.mock_llm_client import MockLLMClient, _training_plan_payload
import trailtraining.llm.shared as sh
sh.make_openrouter_client = lambda: MockLLMClient()
from trailtraining.contracts import TrainingPlanArtifact
plan = _training_plan_payload()
TrainingPlanArtifact.model_validate(plan)
print('Mock LLM -> valid TrainingPlanArtifact: OK')
"

log_quota
echo "=== Gate 1 PASSED ==="
