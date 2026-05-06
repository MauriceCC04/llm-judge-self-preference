#!/usr/bin/env bash
#SBATCH --job-name=jbs_match_structural
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=out/match_structural_%j.out
#SBATCH --error=err/match_structural_%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the llm-judge-self-preference checkout}"
: "${TRAILTRAINING_REPO:?Set TRAILTRAINING_REPO to the trailtraining checkout}"
PY="${PY:-/home/${USER}/.conda/envs/judge-bias/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
mkdir -p out err

PLANS_DIR="${PLANS_DIR:-artifacts/gen_src_t070_exp_t000/matching_pool/plans}"
PAIRS_FILE="${PAIRS_FILE:-artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json}"
DIAG_DIR="${DIAG_DIR:-artifacts/gen_src_t070_exp_t000/matching_pool/structural_diagnostics}"

"$PY" cli.py match \
  --plans "$PLANS_DIR" \
  --output "$PAIRS_FILE" \
  --allow-mixed-generation-conditions \
  --tolerance "${STRUCTURAL_TOLERANCE:-2.0}" \
  --target-pairs "${TARGET_PAIRS:-250}" \
  --fail-below-target-ratio "${FAIL_BELOW_TARGET_RATIO:-0.9765625}"

"$PY" cli.py match-diagnostics \
  --plans "$PLANS_DIR" \
  --output "$DIAG_DIR"
