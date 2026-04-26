#!/usr/bin/env bash
# slurm/run_judge_hpc.sh — canonical HPC judge job for the frozen study.
#
# This script supersedes slurm/run_judge.sh because it:
#   * uses a safe 20h walltime header for the largest active judge
#   * exports PYTHONPATH so repo modules are importable without editable install
#   * auto-builds or refreshes the style audit before judging when required
#   * checks that the allocated SLURM walltime is not below the judge-specific
#     buffered requirement from judge.panel.walltime_hours_with_buffer()
#
# Optional env:
#   JUDGE_NAME            default qwen_7b_judge
#   JUDGE_MODE            pilot | full
#   RUN_PAIRWISE          1 | 0
#   RUN_SOFT_EVAL         1 | 0
#   PAIR_LIMIT            integer limit for sharded pairwise runs
#   PLAN_LIMIT            integer limit for sharded soft-eval runs
#   REQUIRE_STYLE_GATE    1 | 0
#   STYLE_GATE_SUMMARY    path to style gate summary JSON
#   VLLM_PORT             default 8772
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
JUDGE_MODE="${JUDGE_MODE:-full}"
RUN_PAIRWISE="${RUN_PAIRWISE:-1}"
RUN_SOFT_EVAL="${RUN_SOFT_EVAL:-1}"
PAIR_LIMIT="${PAIR_LIMIT:-}"
PLAN_LIMIT="${PLAN_LIMIT:-}"
REQUIRE_STYLE_GATE="${REQUIRE_STYLE_GATE:-1}"
STYLE_GATE_SUMMARY="${STYLE_GATE_SUMMARY:-results/style_audit_summary.json}"
export VLLM_PORT="${VLLM_PORT:-8772}"

echo "=== HPC Judge run: ${JUDGE_NAME} (${JUDGE_MODE}) ==="
log_quota
preflight_quota_gate

export HF_HUB_OFFLINE=1
export OPENROUTER_API_KEY=dummy

echo "--- Resolving judge spec ---"
python -c "
from judge.panel import get_judge, assert_judge_fits_quota, walltime_hours_with_buffer
judge = get_judge('${JUDGE_NAME}')
assert_judge_fits_quota(judge)
print(f'model_id={judge.model_id} quant={judge.quant} disk_gb={judge.disk_gb} time_hours={judge.time_hours} buffered_hours={walltime_hours_with_buffer(judge)}')
"

python -c "
import os
from judge.panel import get_judge, walltime_hours_with_buffer

def parse_hours(raw: str) -> float:
    if not raw:
        return 0.0
    day = 0
    if '-' in raw:
        d, raw = raw.split('-', 1)
        day = int(d)
    parts = [int(p) for p in raw.split(':')]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m = parts
        s = 0
    else:
        h = parts[0]
        m = s = 0
    return day * 24 + h + (m / 60.0) + (s / 3600.0)

limit = parse_hours(os.getenv('SLURM_TIMELIMIT', ''))
required = float(walltime_hours_with_buffer(get_judge('${JUDGE_NAME}')))
if limit and limit + 1e-9 < required:
    raise SystemExit(f'[ABORT] Allocated walltime {limit:.2f}h is below required buffered judge walltime {required:.2f}h.')
print(f'Walltime check OK: allocated={limit or float(20):.2f}h required={required:.2f}h')
"

if [[ "${REQUIRE_STYLE_GATE}" == "1" ]]; then
    export STYLE_GATE_SUMMARY
    echo "--- Checking style-gate freshness ---"
    NEEDS_STYLE_AUDIT=$(python -c "
from pathlib import Path
import os
summary = Path(os.environ['STYLE_GATE_SUMMARY'])
pairs = Path('matched_pairs.json')
plans = Path('plans')
if not summary.exists():
    print('1')
    raise SystemExit(0)
summary_mtime = summary.stat().st_mtime
needs_refresh = False
if pairs.exists() and pairs.stat().st_mtime > summary_mtime:
    needs_refresh = True
else:
    for pattern in ('*.json.provenance.json',):
        for path in plans.glob(pattern):
            if path.stat().st_mtime > summary_mtime:
                needs_refresh = True
                break
        if needs_refresh:
            break
print('1' if needs_refresh else '0')
")
    if [[ "${NEEDS_STYLE_AUDIT}" == "1" ]]; then
        STYLE_GATE_OUTPUT_DIR=$(python -c "from pathlib import Path; import os; print(Path(os.environ['STYLE_GATE_SUMMARY']).parent)")
        echo "--- Building or refreshing style audit in ${STYLE_GATE_OUTPUT_DIR} ---"
        python cli.py audit-style --plans plans/ --pairs matched_pairs.json --output "${STYLE_GATE_OUTPUT_DIR}"
    else
        echo "Style-gate summary is current: ${STYLE_GATE_SUMMARY}"
    fi
fi

JUDGE_MODEL=$(python -c "from judge.panel import get_judge; print(get_judge('${JUDGE_NAME}').model_id)")
JUDGE_QUANT=$(python -c "from judge.panel import get_judge; judge = get_judge('${JUDGE_NAME}'); print(judge.quant if judge.quant != 'fp16' else '')")

echo "--- Starting vLLM (${JUDGE_MODEL}) ---"
VLLM_CMD="python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL} --port ${VLLM_PORT} --host 127.0.0.1 --max-model-len 8192 --disable-log-requests"
if [[ -n "${JUDGE_QUANT}" ]]; then
    VLLM_CMD="${VLLM_CMD} --quantization ${JUDGE_QUANT}"
fi
eval "${VLLM_CMD}" > out/vllm_judge_hpc_${JUDGE_NAME}.log 2>&1 &
VLLM_PID=$!

echo "--- Health poll (up to 15 min) ---"
python -c "
from judge.vllm_server import VllmServer
from pathlib import Path
server = VllmServer('${JUDGE_MODEL}', ${VLLM_PORT}, log_dir=Path('out'), max_model_len=8192)
raise SystemExit(0 if server.health_poll(timeout_s=900, interval_s=15) else 1)
"

cleanup() {
    set +e
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 15
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    set -e
}
trap cleanup EXIT

export TRAILTRAINING_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

CLI_ARGS=(judge --judge "${JUDGE_NAME}" --plans plans/ --pairs matched_pairs.json --output judgments/)
if [[ "${JUDGE_MODE}" == "pilot" ]]; then
    CLI_ARGS+=(--pilot)
fi
if [[ "${RUN_PAIRWISE}" == "0" ]]; then
    CLI_ARGS+=(--skip-pairwise)
fi
if [[ "${RUN_SOFT_EVAL}" == "0" ]]; then
    CLI_ARGS+=(--skip-soft-eval)
fi
if [[ -n "${PAIR_LIMIT}" ]]; then
    CLI_ARGS+=(--pair-limit "${PAIR_LIMIT}")
fi
if [[ -n "${PLAN_LIMIT}" ]]; then
    CLI_ARGS+=(--plan-limit "${PLAN_LIMIT}")
fi
if [[ "${REQUIRE_STYLE_GATE}" == "1" ]]; then
    CLI_ARGS+=(--require-style-gate --style-gate-summary "${STYLE_GATE_SUMMARY}")
fi

echo "--- Running CLI judge path ---"
python cli.py "${CLI_ARGS[@]}"

echo "--- Deleting model weights ---"
MODEL_DIR_NAME="models--$(echo "${JUDGE_MODEL}" | tr '/' '--')"
rm -rf "${HF_HUB_CACHE}/${MODEL_DIR_NAME}" 2>/dev/null || true

log_quota
echo "=== HPC judge run complete: ${JUDGE_NAME} ==="
