#!/usr/bin/env bash
# slurm/submit_judge_hpc.sh — login-node wrapper for judge jobs.
#
# Usage:
#   bash slurm/submit_judge_hpc.sh qwen_7b_judge
#   JUDGE_MODE=pilot bash slurm/submit_judge_hpc.sh qwen_7b_judge
#
# This wrapper computes the buffered walltime from judge.panel and submits the
# canonical run_judge_hpc.sh script with the correct --time override.

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ABORT] submit_judge_hpc.sh must run on the login node, not inside sbatch." >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

JUDGE_NAME="${1:?Usage: bash slurm/submit_judge_hpc.sh <judge_name>}"
JUDGE_MODE="${JUDGE_MODE:-full}"
RUN_PAIRWISE="${RUN_PAIRWISE:-1}"
RUN_SOFT_EVAL="${RUN_SOFT_EVAL:-1}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
PLAN_LIMIT="${PLAN_LIMIT:-}"
REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE:-1}"
STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY:-results/style_audit_summary.json}"

WALLTIME=$(python -c "
from judge.panel import get_judge, walltime_hours_with_buffer
judge = get_judge('${JUDGE_NAME}')
print(f'{walltime_hours_with_buffer(judge):02d}:00:00')
")

echo "Submitting ${JUDGE_NAME} with walltime ${WALLTIME}"
sbatch \
    --exclude=gnode04 \
    --time="${WALLTIME}" \
    --export=ALL,JUDGE_NAME="${JUDGE_NAME}",JUDGE_MODE="${JUDGE_MODE}",RUN_PAIRWISE="${RUN_PAIRWISE}",RUN_SOFT_EVAL="${RUN_SOFT_EVAL}",PAIR_LIMIT="${PAIR_LIMIT}",PLAN_LIMIT="${PLAN_LIMIT}",REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE}",STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY}" \
    --wrap='cd "$PROJECT_ROOT" && bash slurm/run_judge_hpc.sh'
