#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd "${PROJECT_ROOT}"

JUDGE_NAME="${1:?usage: submit_judge.sh <judge_name> [pilot|full] [pairwise|soft_eval|both]}"
JUDGE_MODE="${2:-full}"
JUDGE_TASK="${3:-both}"

HOURS=$(python - <<PY
import math
import sys
sys.path.insert(0, ".")
from judge.panel import get_judge
judge = get_judge("${JUDGE_NAME}")
print(int(math.ceil(judge.time_hours * 1.2)))
PY
)

sbatch \
  --time="${HOURS}:00:00" \
  --exclude=gnode04 \
  --export=ALL,JUDGE_NAME="${JUDGE_NAME}",JUDGE_MODE="${JUDGE_MODE}",JUDGE_TASK="${JUDGE_TASK}" \
  --wrap='cd '"${PROJECT_ROOT}"' && bash slurm/run_judge.sh'