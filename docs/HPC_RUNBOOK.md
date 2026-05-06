# HPC Runbook for the Current Qwen/Gemma Study

This runbook is the operational guide for the 1024-candidate Qwen/Gemma matching pool and the 10,000 pairwise-judgment full study.

## Environment setup

Do not rely on `conda activate` in batch scripts. Use the environment Python directly and explicitly set both repository paths.

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

$PY - <<'PY'
import sys
import pydantic
import trailtraining
print('python', sys.executable)
print('pydantic', pydantic.__version__)
print('trailtraining', trailtraining.__file__)
PY
```

This prevents the known failures:

- `ModuleNotFoundError: No module named 'pydantic'`
- `ModuleNotFoundError: No module named 'trailtraining'`
- broken `conda activate`
- using the wrong interpreter

## Hugging Face cache

```bash
export HF_HOME=/mnt/beegfsstudents/home/<USER_ID>/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
lquota
```

Check exactly the models needed for the current job:

```bash
$PY tools/check_model_cache.py "Qwen/Qwen2.5-3B-Instruct"
$PY tools/check_model_cache.py "Qwen/Qwen2.5-7B-Instruct"
$PY tools/check_model_cache.py "google/gemma-3-4b-it"
```

For judge jobs, also check the active judge model. Do not delete cached models unless that deletion is intentional and safe for later jobs.

## Matching pool audit

```bash
$PY cli.py match-diagnostics \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/structural_diagnostics
```

Inspect:

- structural score distributions by source/arm
- per-fixture overlap
- possible match counts by tolerance
- final pair coverage by fixture and source family
- feature-balance gaps
- style/leakage audit outputs

## Structural matching SLURM command

Do not run full matching or judging on a login node except for tiny diagnostics.

```bash
mkdir -p out err
sbatch \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --time=00:45:00 \
  --chdir=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference \
  --output=out/match_structural_%j.out \
  --error=err/match_structural_%j.err \
  --wrap='
set -euo pipefail
export REPO_ROOT=/mnt/beegfsstudents/home/<USER_ID>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER_ID>/trailtraining
export PY=/home/<USER_ID>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
$PY cli.py match \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --allow-mixed-generation-conditions \
  --tolerance 2.0 \
  --target-pairs 250 \
  --fail-below-target-ratio 1.0
$PY cli.py match-diagnostics \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/structural_diagnostics
'
```

## Full-study launch gate

Full judging requires:

```text
matched_pairs >= 250
expected_pairwise_documents >= 10,000
exactly 4 valid Qwen/Gemma judge models
5 repeats
AB and BA positions
both Qwen and Gemma source plans represented
structural matching audit passes
source masking audit passes
style audit passes
```

Run:

```bash
$PY cli.py audit-style \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/results

$PY cli.py build-eval-manifest \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/eval_manifest \
  --seed 20260506

$PY cli.py launch-gate \
  --plans artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --style-audit artifacts/gen_src_t070_exp_t000/matching_pool/results/style_audit_summary.json
```

Do not proceed if the gate fails.

## Judge jobs

Run one judge per job. Keep temperature-specific outputs separate.

```bash
for J in qwen_7b_judge qwen_14b_judge gemma_4b_judge gemma_12b_judge; do
  JUDGE_NAME="$J" \
  JUDGE_MODE=full \
  RUN_PAIRWISE=1 \
  RUN_SOFT_EVAL=0 \
  PLANS_DIR=artifacts/gen_src_t070_exp_t000/matching_pool/plans \
  PAIRS_FILE=artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  JUDGMENTS_DIR=artifacts/gen_src_t070_exp_t000/matching_pool/judgments_eval_t000 \
  PAIRWISE_VIEW=canonical_masked \
  JUDGE_TEMPERATURE=0.0 \
  CLEANUP_MODEL_CACHE=0 \
  sbatch slurm/run_judge_hpc.sh
done
```

## Analysis

```bash
$PY tools/analyze_pairwise_results.py \
  --judgments artifacts/gen_src_t070_exp_t000/matching_pool/judgments_eval_t000 \
  --pairs artifacts/gen_src_t070_exp_t000/matching_pool/matched_pairs.json \
  --output artifacts/gen_src_t070_exp_t000/matching_pool/results_eval_t000
```

Do not report pilot outputs as the full study.
