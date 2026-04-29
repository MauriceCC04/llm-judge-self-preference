#!/usr/bin/env bash
# slurm/pre_cache_models.sh — login-node only; pre-download model weights.
#
# Run BEFORE sbatch — compute nodes may not reach the Hugging Face CDN.
# To stay within quota, this script purges previously cached model weights before
# downloading the next model.
#
# Usage:
#   bash slurm/pre_cache_models.sh qwen_7b_judge
#   bash slurm/pre_cache_models.sh all

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ABORT] pre_cache_models.sh must run on the LOGIN NODE, not inside sbatch." >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_env

cd "${PROJECT_ROOT}"

JUDGE_NAME="${1:-all}"

_purge_all_model_caches() {
    echo "--- Purging previously cached model weights ---"
    python -c "
from hpc.quota import purge_all_hf_model_caches
count = purge_all_hf_model_caches()
print(f'Removed {count} cached model directories')
"
    echo "--- lquota after purge ---"
    lquota 2>/dev/null || du -sh "${HOME}" 2>/dev/null || true
}

_cache_model() {
    local model_id="$1"
    _purge_all_model_caches
    echo "=== Caching ${model_id} ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${model_id}', ignore_patterns=['*.msgpack', '*.h5'])
print('Done: ${model_id}')
"
    echo "--- lquota after download ---"
    lquota 2>/dev/null || du -sh "${HOME}" 2>/dev/null || true
}

if [[ "${JUDGE_NAME}" == "all" ]]; then
    echo "[WARN] 'all' now validates/downloads models sequentially under quota."
    echo "[WARN] Only the last cached model will remain on disk after this command completes."
    python -c "
import sys; sys.path.insert(0, '.')
from judge.panel import PANEL
seen = set()
for spec in PANEL:
    if spec.model_id not in seen:
        seen.add(spec.model_id)
        print(spec.model_id)
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