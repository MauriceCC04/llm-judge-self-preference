#!/usr/bin/env bash
# slurm/common.sh — shared environment setup for all SLURM scripts.
#
# USAGE: source slurm/common.sh
#
# Sets:
#   PROJECT_ROOT  — absolute path to the repo root (resolved from this file)
#   HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE, PIP_CACHE_DIR,
#   TORCH_HOME, VLLM_CACHE_ROOT — all pinned under $HOME so du can find them
#
# Also defines:
#   activate_env()   — activates the conda/venv environment
#   log_quota()      — prints lquota output (login-node safe)

set -euo pipefail

# ── Resolve PROJECT_ROOT from this script's location (works with sbatch --wrap) ──
_COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_COMMON_SH_DIR}/.." && pwd)"
export PROJECT_ROOT

# ── Pin all caches under $HOME ────────────────────────────────────────────────
export HF_HOME="${HOME}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${HOME}/pip_cache"
export TORCH_HOME="${HOME}/torch_cache"
export VLLM_CACHE_ROOT="${HOME}/vllm_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# ── vLLM port (override in individual scripts) ────────────────────────────────
export VLLM_PORT="${VLLM_PORT:-8765}"

# ── Activate conda environment (set CONDA_ENV_NAME before sourcing) ──────────
activate_env() {
    local env_name="${CONDA_ENV_NAME:-judge-bias}"
    if command -v conda &>/dev/null; then
        # shellcheck disable=SC1090
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate "${env_name}" 2>/dev/null || true
    elif [[ -d "${HOME}/.venv/bin" ]]; then
        source "${HOME}/.venv/bin/activate"
    fi
}

# ── Quota reporter ────────────────────────────────────────────────────────────
log_quota() {
    echo "=== lquota ==="
    lquota 2>/dev/null || du -sh "${HOME}" 2>/dev/null || echo "(quota unavailable)"
}

# ── Preflight: refuse to run if used > 30 GB ─────────────────────────────────
preflight_quota_gate() {
    local used_kb
    used_kb=$(du -sk "${HOME}" 2>/dev/null | awk '{print $1}' || echo 0)
    local used_gb
    used_gb=$(echo "scale=1; ${used_kb}/1048576" | bc 2>/dev/null || echo 99)
    if (( $(echo "${used_gb} > 30" | bc -l 2>/dev/null || echo 0) )); then
        echo "[ABORT] Disk usage ${used_gb} GB > 30 GB soft ceiling. Clean caches first." >&2
        exit 1
    fi
    echo "[OK] Disk usage: ${used_gb} GB"
}
