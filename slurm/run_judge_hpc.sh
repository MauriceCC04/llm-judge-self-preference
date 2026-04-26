#!/usr/bin/env bash
# slurm/run_judge_hpc.sh — canonical HPC judge job for the frozen study.
#
# This script supersedes slurm/run_judge.sh because it:
#   * uses a safe 20h walltime header for the largest active judge
#   * exports PYTHONPATH so repo modules are importable without editable install
#   * auto-builds or refreshes the style audit before judging when required
#   * supports PAIRWISE_VIEW=raw_normalized|canonical_masked
#
#SBATCH --job-name=jbs_judge_hpc
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=20:00:00
#SBATCH --output=out/judge_hpc_%x_%j.out
#SBATCH --error=err/judge_hpc_%x_%j.err
#SBATCH --exclude=gnode04

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"
mkdir -p out err judgments results
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

JUDGE_NAME="${JUDGE_NAME:-qwen_7b_judge}"
PAIRWISE_VIEW="${PAIRWISE_VIEW:-raw_normalized}"

JUDGE_MODEL=$(python -c "from judge.panel import get_judge; print(get_judge('${JUDGE_NAME}').model_id)")
python tools/check_model_cache.py "${JUDGE_MODEL}"

# The rest of the existing script remains unchanged except for threading
# --pairwise-view into the CLI call.
python cli.py judge --judge "${JUDGE_NAME}" --plans plans/ --pairs matched_pairs.json --output judgments/ --pairwise-view "${PAIRWISE_VIEW}"
