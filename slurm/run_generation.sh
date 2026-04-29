#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec bash "${PROJECT_ROOT}/slurm/run_generation_hpc.sh" "$@"