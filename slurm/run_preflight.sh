#!/usr/bin/env bash
# slurm/run_preflight.sh — Gate 1: HPC preflight checks (~5 min).
#
# Submit from login node:
#   sbatch --exclude=gnode04 --wrap='cd $PROJECT_ROOT && bash slurm/run_preflight.sh'
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

echo "--- Python imports ---"
python -c "
import trailtraining, openai, pydantic
print('imports OK')
print('trailtraining:', trailtraining.__file__)
"

echo "--- GPU visibility ---"
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
"

echo "--- Study imports ---"
python -c "
import sys; sys.path.insert(0, '.')
from fixtures.spec import ALL_FIXTURE_SPECS
from generate.constants import EXPLAINER_MODEL_ID
from judge.panel import JUDGES
print(f'Fixtures: {len(ALL_FIXTURE_SPECS)}')
print(f'Explainer: {EXPLAINER_MODEL_ID}')
print(f'Judges: {[j.name for j in JUDGES]}')
"

echo "--- Mock LLM call ---"
python -c "
import sys; sys.path.insert(0, '.')
from tests.mock_llm_client import MockLLMClient, _training_plan_payload
import trailtraining.llm.shared as sh
sh.make_openrouter_client = lambda: MockLLMClient()
from trailtraining.contracts import TrainingPlanArtifact
import json
plan = _training_plan_payload()
art = TrainingPlanArtifact.model_validate(plan)
print('Mock LLM → valid TrainingPlanArtifact: OK')
"

log_quota
echo "=== Gate 1 PASSED ==="
