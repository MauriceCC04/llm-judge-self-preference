#!/usr/bin/env bash
# slurm/submit_generation.sh — submit one generation shard.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARM="${1:?usage: submit_generation.sh <llm|programmatic> [plans_per_fixture] [seed_offset] [fixture_ids] [source_model]}"
GENERATION_N="${2:-}"
SEED_OFFSET="${3:-0}"
FIXTURE_IDS="${4:-}"
LLM_SOURCE_MODEL="${5:-}"

EXPORTS="ALL,GENERATION_ARM=${ARM},SEED_OFFSET=${SEED_OFFSET}"
[[ -n "${GENERATION_N}" ]] && EXPORTS+=",GENERATION_N=${GENERATION_N}"
[[ -n "${FIXTURE_IDS}" ]] && EXPORTS+=",FIXTURE_IDS=${FIXTURE_IDS}"
if [[ "${ARM}" == "llm" ]]; then
  if [[ -z "${LLM_SOURCE_MODEL}" ]]; then
    echo "LLM_SOURCE_MODEL is required for llm generation" >&2
    exit 1
  fi
  EXPORTS+=",LLM_SOURCE_MODEL=${LLM_SOURCE_MODEL}"
fi

sbatch \
  --exclude=gnode04 \
  --export="${EXPORTS}" \
  --wrap='cd $PROJECT_ROOT && bash slurm/run_generation.sh'
