#!/usr/bin/env bash
# slurm/submit_judge_hpc.sh — login-node wrapper for one judge job.
#
# Usage:
#   bash slurm/submit_judge_hpc.sh qwen_7b_judge
#   JUDGE_MODE=pilot bash slurm/submit_judge_hpc.sh qwen_7b_judge
#   PAIRWISE_VIEW=canonical_masked bash slurm/submit_judge_hpc.sh qwen_7b_judge
#   WAIT_FOR_JOB=1 bash slurm/submit_judge_hpc.sh qwen_7b_judge
#
# This wrapper:
# - must run on the login node
# - computes buffered walltime from judge.panel
# - forwards pairwise view and other judge controls
# - uses an absolute repo-path wrap rather than relying on ambient $PROJECT_ROOT

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
PAIRWISE_VIEW="${PAIRWISE_VIEW:-raw_normalized}"
REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE:-1}"
STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY:-results/style_audit_summary.json}"
CLEANUP_MODEL_CACHE="${CLEANUP_MODEL_CACHE:-1}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-0}"
ACCOUNT="${ACCOUNT:-3202029}"

WALLTIME=$(python -c "
from judge.panel import get_judge, walltime_hours_with_buffer
judge = get_judge('${JUDGE_NAME}')
print(f'{walltime_hours_with_buffer(judge):02d}:00:00')
")

ROOT_ESCAPED=$(printf '%q' "${PROJECT_ROOT}")
WRAP_CMD="cd ${ROOT_ESCAPED} && bash slurm/run_judge_hpc.sh"

SBATCH_ARGS=(
    "--job-name=judge-${JUDGE_NAME}"
    "--account=${ACCOUNT}"
    "--partition=stud"
    "--qos=stud"
    "--exclude=gnode04"
    "--time=${WALLTIME}"
    "--output=out/%x_%j.out"
    "--error=err/%x_%j.err"
    "--export=ALL,JUDGE_NAME=${JUDGE_NAME},JUDGE_MODE=${JUDGE_MODE},RUN_PAIRWISE=${RUN_PAIRWISE},RUN_SOFT_EVAL=${RUN_SOFT_EVAL},PAIR_LIMIT=${PAIR_LIMIT},PLAN_LIMIT=${PLAN_LIMIT},PAIRWISE_VIEW=${PAIRWISE_VIEW},REQUIRE_STYLE_GATE=${REQUIRE_STYLE_GATE},STYLE_GATE_SUMMARY=${STYLE_GATE_SUMMARY},CLEANUP_MODEL_CACHE=${CLEANUP_MODEL_CACHE}"
    "--wrap=${WRAP_CMD}"
)

if [[ "${WAIT_FOR_JOB}" == "1" ]]; then
    SBATCH_ARGS+=("--wait")
fi

echo "Submitting ${JUDGE_NAME} with walltime ${WALLTIME} (view=${PAIRWISE_VIEW}, mode=${JUDGE_MODE})"
sbatch "${SBATCH_ARGS[@]}"