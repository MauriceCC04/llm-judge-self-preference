#!/usr/bin/env bash
# slurm/pre_cache_models.sh — login-node only; pre-download model weights.
#
# Run BEFORE sbatch — compute nodes may not reach the HuggingFace CDN.
# Downloads one model at a time to stay within the 30 GB soft ceiling.
#
# Usage:
#   bash slurm/pre_cache_models.sh qwen_7b_judge
#   bash slurm/pre_cache_models.sh all    # all judges one by one

set -euo pipefail

# Refuse to run inside a SLURM job
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ABORT] pre_cache_models.sh must run on the LOGIN NODE, not inside sbatch." >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"

JUDGE_NAME="${1:-all}"

_cache_model() {
    local model_id="$1"
    echo "=== Caching ${model_id} ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${model_id}', ignore_patterns=['*.msgpack','*.h5'])
print('Done: ${model_id}')
"
    echo "--- lquota after download ---"
    lquota 2>/dev/null || du -sh "${HOME}" 2>/dev/null || true
}

if [[ "${JUDGE_NAME}" == "all" ]]; then
    python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import JUDGES
for j in JUDGES:
    print(j.model_id)
" | while read -r model_id; do
        _cache_model "${model_id}"
    done
else
    MODEL_ID=$(python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import get_judge
print(get_judge('${JUDGE_NAME}').model_id)
")
    _cache_model "${MODEL_ID}"
fi

echo "=== Pre-cache complete ==="
