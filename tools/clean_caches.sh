#!/usr/bin/env bash
# tools/clean_caches.sh — recover from orphaned HuggingFace downloads.
# Run from the login node after a crashed job leaves partial model weights.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../slurm/common.sh" 2>/dev/null || true

echo "=== HF cache before ==="
du -sh "${HF_HUB_CACHE:-$HOME/hf_cache/hub}" 2>/dev/null || echo "(not found)"

MODEL=${1:-}
if [[ -n "$MODEL" ]]; then
    # Delete a specific model (replace / with --  for the directory name)
    DIR_NAME="models--$(echo "$MODEL" | tr '/' '--')"
    TARGET="${HF_HUB_CACHE:-$HOME/hf_cache/hub}/$DIR_NAME"
    if [[ -d "$TARGET" ]]; then
        echo "Removing $TARGET"
        rm -rf "$TARGET"
    else
        echo "Not found: $TARGET"
    fi
else
    echo "Usage: $0 <org/model-name>"
    echo "  e.g.: $0 Qwen/Qwen2.5-32B-Instruct-AWQ"
fi

echo "=== HF cache after ==="
du -sh "${HF_HUB_CACHE:-$HOME/hf_cache/hub}" 2>/dev/null || echo "(empty)"
