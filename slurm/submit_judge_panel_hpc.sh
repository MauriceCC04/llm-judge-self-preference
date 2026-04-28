#!/usr/bin/env bash
# slurm/submit_judge_panel_hpc.sh — sequential login-node launcher for multiple judges.
#
# Why this exists:
# - Bocconi's `stud` QoS is unfriendly to bursty multi-job submission.
# - pre_cache_models.sh must run on the login node, not inside sbatch.
# - compute nodes may not reach the Hugging Face CDN.
#
# This wrapper therefore:
# 1. runs on the login node
# 2. pre-caches one judge model at a time
# 3. submits exactly one judge job
# 4. waits for it to finish
# 5. moves on to the next judge
#
# Usage:
#   bash slurm/submit_judge_panel_hpc.sh
#   bash slurm/submit_judge_panel_hpc.sh llama_8b_judge qwen_7b_judge
#   PAIRWISE_VIEW=canonical_masked bash slurm/submit_judge_panel_hpc.sh
#   JUDGE_MODE=pilot bash slurm/submit_judge_panel_hpc.sh qwen_7b_judge

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ABORT] submit_judge_panel_hpc.sh must run on the login node, not inside sbatch." >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

JUDGE_MODE="${JUDGE_MODE:-full}"
RUN_PAIRWISE="${RUN_PAIRWISE:-1}"
RUN_SOFT_EVAL="${RUN_SOFT_EVAL:-1}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
PLAN_LIMIT="${PLAN_LIMIT:-}"
PLANS_DIR="${PLANS_DIR:-plans/}"
PLANS_PARENT="$(dirname "${PLANS_DIR%/}")"
PAIRS_FILE="${PAIRS_FILE:-${PLANS_PARENT}/matched_pairs.json}"
JUDGMENTS_DIR="${JUDGMENTS_DIR:-${PLANS_PARENT}/judgments/}"
PAIRWISE_VIEW="${PAIRWISE_VIEW:-raw_normalized}"
REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE:-1}"
STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY:-results/style_audit_summary.json}"
CLEANUP_MODEL_CACHE="${CLEANUP_MODEL_CACHE:-1}"
ACCOUNT="${ACCOUNT:-3202029}"

if [[ "$#" -gt 0 ]]; then
    JUDGES=("$@")
else
    mapfile -t JUDGES < <(python -c "
from generate.constants import ACTIVE_JUDGE_NAMES
for name in ACTIVE_JUDGE_NAMES:
    print(name)
")
fi

echo "=== Sequential judge panel launcher ==="
echo "Judges: ${JUDGES[*]}"
echo "Mode: ${JUDGE_MODE}"
echo "View: ${PAIRWISE_VIEW}"
echo "Plans dir: ${PLANS_DIR}"
echo "Pairs file: ${PAIRS_FILE}"
echo "Judgments dir: ${JUDGMENTS_DIR}"
echo "Account: ${ACCOUNT}"
echo

for JUDGE_NAME in "${JUDGES[@]}"; do
    echo "=== Pre-caching ${JUDGE_NAME} on the login node ==="
    bash slurm/pre_cache_models.sh "${JUDGE_NAME}"

    echo "=== Submitting ${JUDGE_NAME} and waiting for completion ==="
    WAIT_FOR_JOB=1 \
    ACCOUNT="${ACCOUNT}" \
    JUDGE_MODE="${JUDGE_MODE}" \
    RUN_PAIRWISE="${RUN_PAIRWISE}" \
    RUN_SOFT_EVAL="${RUN_SOFT_EVAL}" \
    PAIR_LIMIT="${PAIR_LIMIT}" \
    PLAN_LIMIT="${PLAN_LIMIT}" \
    PLANS_DIR="${PLANS_DIR}" \
    PAIRS_FILE="${PAIRS_FILE}" \
    JUDGMENTS_DIR="${JUDGMENTS_DIR}" \
    PAIRWISE_VIEW="${PAIRWISE_VIEW}" \
    REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE}" \
    STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY}" \
    CLEANUP_MODEL_CACHE="${CLEANUP_MODEL_CACHE}" \
    bash slurm/submit_judge_hpc.sh "${JUDGE_NAME}"

    echo "=== Completed ${JUDGE_NAME} ==="
    echo
done

echo "=== Sequential judge panel complete ==="