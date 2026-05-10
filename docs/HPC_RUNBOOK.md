# HPC Runbook

This runbook records the working HPC setup, common failures, and current next steps for the Qwen/Gemma judge-bias study.

## Environment setup

Use explicit paths. Do not rely on `conda activate`.

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PY=/home/3202029/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

Verify:

```bash
"$PY" - <<'PY'
import sys, pydantic, trailtraining
print(sys.executable)
print("pydantic", pydantic.__version__)
print("trailtraining", trailtraining.__file__)
PY
```

## Hugging Face cache setup

```bash
export HF_HOME=/mnt/beegfsstudents/home/3202029/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_XET=1
```

For downloads:

```bash
unset HF_HUB_OFFLINE
```

For offline model execution:

```bash
export HF_HUB_OFFLINE=1
```

Check quota:

```bash
lquota
du -sh "$HF_HOME" 2>/dev/null || true
find "$HF_HUB_CACHE" -maxdepth 4 -type d -name 'models--*' -print 2>/dev/null
```

Do not delete model cache unless the relevant judgment output has been completed, hashed, and synced/backed up.

## Model cache checks

```bash
"$PY" tools/check_model_cache.py Qwen/Qwen2.5-7B-Instruct
"$PY" tools/check_model_cache.py Qwen/Qwen2.5-14B-Instruct-AWQ
"$PY" tools/check_model_cache.py google/gemma-3-4b-it
"$PY" tools/check_model_cache.py google/gemma-3-12b-it
```

Because the quota is tight, the study was run one model at a time.

## Primary pairwise judging status

The primary 10,000 pairwise judgments are complete.

Primary output directory:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/
```

Combined primary file:

```text
pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Verify from repo root:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"

wc -l "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
cat "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.record_id' | sort | uniq -d | wc -l
cat "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.judge' | sort | uniq -c
cat "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.order' | sort | uniq -c
cat "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.run' | sort | uniq -c
```

Expected:

```text
10,000 rows
0 duplicate record_ids
2,500 per judge
5,000 AB / 5,000 BA
2,000 per run index
```

## Primary pairwise incidents and fixes

### vLLM log request flag mismatch

Observed error:

```text
api_server.py: error: unrecognized arguments: --disable-log-requests
```

Fix:

```text
Use --no-enable-log-requests
```

### Qwen 14B AWQ quantization mismatch

Observed error:

```text
Quantization method specified in the model config (awq) does not match the quantization argument (awq_int4)
```

Fix in SLURM script:

```python
q = "" if j.quant == "fp16" else j.quant
if "${JUDGE_NAME}" == "qwen_14b_judge":
    q = "awq"
elif q == "awq_int4":
    q = "awq"
print(q)
```

### Qwen 14B max token/context issue

Observed error:

```text
model maximum context length is 4096; requested 4096 output tokens plus prompt
```

Fix:

```bash
export TRAILTRAINING_STRUCTURED_MAX_TOKENS=768
```

For marker reruns also use:

```bash
export MARKER_MAX_TOKENS=768
```

### Wrong directory while monitoring

A primary Qwen 14B job was mistakenly thought to be stalled because the shell was in the wrong directory. Always run monitoring commands from repo root or use absolute paths.

## Sync artifacts back to laptop

From laptop:

```bash
LOCAL="/Users/cameroncaputa/PycharmProjects/llm-judge-self-preference"
REMOTE_HOST="bocconi-hpc"
REMOTE_ROOT="/mnt/beegfsstudents/home/3202029/llm-judge-self-preference"

rsync -avz --itemize-changes \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.DS_Store' \
  --exclude='.env' \
  --exclude='*.token' \
  --exclude='hf_cache/' \
  --exclude='hub/' \
  --exclude='transformers/' \
  --exclude='models/' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL}/"
```

## Explicit marker-level rerun current status

A new marker-level evaluation pass is in progress. It should not replace the completed primary pairwise run.

Output directory:

```bash
export MARKER_OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
export PRIMARY_MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"
```

Current known issue:

The first Qwen 14B full marker run wrote 701 valid rows into a nested wrong path:

```text
$MARKER_OUT/pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

Correct target:

```text
$MARKER_OUT/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

See `MARKER_LEVEL_RERUN.md` for patch/move/resume commands.

