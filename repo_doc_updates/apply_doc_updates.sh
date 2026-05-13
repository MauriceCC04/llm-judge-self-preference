#!/usr/bin/env bash
set -euo pipefail

# Apply documentation replacements for the current Qwen/Gemma study state.
# Run from the repository root:
#   bash apply_doc_updates.sh

ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="docs_backup_${STAMP}"
mkdir -p "$BACKUP_DIR"

FILES=(
  README.md
  HPC_RUNBOOK.md
  PREREGISTRATION.md
  INCIDENTS.md
  TEMPERATURE_SWEEPS.md
  MARKER_EVAL_RUNBOOK.md
  RESULTS_SUMMARY.md
  ARTIFACT_MANIFEST.md
)

for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp "$f" "$BACKUP_DIR/$f"
  fi
done

# This script expects the replacement markdown files next to it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "${FILES[@]}"; do
  if [[ -f "$SCRIPT_DIR/$f" ]]; then
    cp "$SCRIPT_DIR/$f" "$ROOT/$f"
    echo "updated $f"
  else
    echo "missing replacement $SCRIPT_DIR/$f" >&2
    exit 1
  fi
done

echo "Backups, if any, are in $BACKUP_DIR"
echo "Review with: git diff -- README.md HPC_RUNBOOK.md PREREGISTRATION.md INCIDENTS.md TEMPERATURE_SWEEPS.md MARKER_EVAL_RUNBOOK.md RESULTS_SUMMARY.md ARTIFACT_MANIFEST.md"
