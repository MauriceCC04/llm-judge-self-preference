# HPC Runbook

This runbook records the reproducible HPC procedure for the Qwen/Gemma judge-bias study.

## 1. Required environment

Use the environment Python directly. Do not rely on `conda activate` in interactive or SLURM shells unless the shell has been initialized correctly.

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER>/trailtraining
export PY=/home/<USER>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

Observed failure prevented by this setup:

```text
CondaError: Run 'conda init' before 'conda activate'
ModuleNotFoundError: No module named 'trailtraining'
ModuleNotFoundError: No module named 'pydantic'
```

## 2. Login node policy

Do not run full matching, vLLM, pairwise judging, marker judging, or model downloads on the login node unless the operation is a tiny diagnostic. Use SLURM for computation.

Allowed on login node:

- `wc -l`, `jq`, `find`, `sha256sum`
- tiny manifest construction
- cache existence checks
- copying/syncing files

Not allowed on login node:

- vLLM server
- full judge jobs
- full generation jobs
- full analysis if it is CPU-heavy

## 3. Model cache policy

Cache one model at a time if quota is tight. Do not delete a model until its outputs are synced and verified locally.

Common cache environment:

```bash
unset HF_HUB_OFFLINE
export HF_HUB_DISABLE_XET=1
export HF_HOME=/mnt/beegfsstudents/home/<USER>/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
```

Check quota:

```bash
lquota
du -sh "$HF_HOME" 2>/dev/null || true
```

Check model cache:

```bash
"$PY" tools/check_model_cache.py Qwen/Qwen2.5-7B-Instruct
"$PY" tools/check_model_cache.py Qwen/Qwen2.5-14B-Instruct-AWQ
"$PY" tools/check_model_cache.py google/gemma-3-4b-it
"$PY" tools/check_model_cache.py google/gemma-3-12b-it
```

Example cache command:

```bash
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
repo_id = "google/gemma-3-12b-it"
path = snapshot_download(
    repo_id=repo_id,
    local_files_only=False,
    ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot", "flax_model*", "tf_model*"],
)
print("cached", repo_id)
print(path)
PY
```

Only after outputs are verified locally, remove a model if quota requires:

```bash
export HF_HOME=/mnt/beegfsstudents/home/<USER>/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
rm -rf "${HF_HUB_CACHE}/models--Qwen--Qwen2.5-7B-Instruct"
rm -rf "${HF_HUB_CACHE}/.locks/models--Qwen--Qwen2.5-7B-Instruct"
```

## 4. Primary pairwise study

Primary frozen manifest:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
```

Primary output directory:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged
```

Full primary study is complete:

```text
250 matched pairs × 4 judges × 5 runs × 2 AB/BA orders = 10,000 pairwise judgments
```

Current combined output:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Verify primary output:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"
P="$D/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl"

wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
jq -r '.winner_role' "$P" | sort | uniq -c
```

Expected:

```text
10000 rows
0 duplicate record_ids
2500 per judge
5000 AB / 5000 BA
2000 per run index
3653 llm / 6347 programmatic
```

## 5. Explicit marker-level study

Marker output directory:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged
```

Current status in the uploaded frozen archive:

```text
qwen_14b_judge: complete, 2500 rows
qwen_7b_judge: complete, 2500 rows
gemma_12b_judge: pending/missing from marker archive
gemma_4b_judge: pending/missing from marker archive
```

Run a marker judge job:

```bash
MARKER_OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
PRIMARY_MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"
mkdir -p "$MARKER_OUT"

VLLM_PORT=8815 \
JUDGE_NAME=gemma_12b_judge \
MANIFEST="$PRIMARY_MANIFEST" \
OUTPUT_DIR="$MARKER_OUT" \
JUDGE_TEMPERATURE=0.0 \
TRAILTRAINING_STRUCTURED_MAX_TOKENS=768 \
MARKER_MAX_TOKENS=768 \
CLEANUP_MODEL_CACHE=0 \
sbatch slurm/run_marker_manifest_judge_hpc.sh
```

Monitor from the repo root:

```bash
watch -n 60 '
date
squeue -u "$USER"
echo
wc -l artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null || echo missing
echo
wc -l artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/failures_marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null || echo no-failure-file
echo
tail -n 1 artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null | jq "{record_id, judge, run, order, marker_count:(.markers|length), timestamp}" 2>/dev/null || true
'
```

Verify a completed marker judge file:

```bash
P="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl"

wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
jq -r '.markers | length' "$P" | sort | uniq -c
jq '[.markers[] | select(.normalization_warnings != null)] | length' "$P" | awk '{s+=$1} END {print "marker fields repaired:", s+0}'
sha256sum "$P" > "${P%.jsonl}.sha256"
```

Expected for a complete marker judge:

```text
2500 rows
0 duplicate record_ids
1250 AB / 1250 BA
500 each run 0-4
2500 rows with 9 markers
```

## 6. Resumability

The marker runner skips existing `record_id`s. If a job times out, resubmit the same judge. If only a few rows are missing, build a missing-record manifest:

```bash
python - <<'PY'
import json
from pathlib import Path
judge = "qwen_7b_judge"
full = Path("artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl")
out = Path(f"artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_{judge}_canonical_masked_scrubbed_v1_t000.jsonl")
missing = Path(f"artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/missing_{judge}_marker_manifest.jsonl")

done = set()
if out.exists():
    with out.open() as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["record_id"])
rows = []
with full.open() as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("judge") == judge and r["record_id"] not in done:
            rows.append(r)
with missing.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
print("done:", len(done))
print("missing:", len(rows))
print("wrote:", missing)
PY
```

## 7. Sync artifacts to laptop

Sync the marker output directory only:

```bash
LOCAL="/Users/<LOCAL_USER>/PycharmProjects/llm-judge-self-preference"
REMOTE_HOST="bocconi-hpc"
REMOTE_ROOT="/mnt/beegfsstudents/home/<USER>/llm-judge-self-preference"

rsync -avz --itemize-changes \
  "${REMOTE_HOST}:${REMOTE_ROOT}/artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/" \
  "${LOCAL}/artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/"
```

Verify locally before deleting model cache on HPC.

## 8. Known job/script issues and fixes

- vLLM on the cluster rejected `--disable-log-requests`; do not use that flag on this installed vLLM version.
- `qwen_14b_judge` uses `Qwen/Qwen2.5-14B-Instruct-AWQ`; ensure AWQ is configured correctly in the SLURM script.
- Marker runner must be called with `--output-dir`, not `--output` and `--failures` pairwise-style arguments.
- If monitoring from `~`, relative artifact paths will appear missing. Always `cd "$REPO_ROOT"` first.
- Some Qwen 7B marker responses omitted `preferred`; the runner repairs blank `preferred` deterministically from marker scores and records `normalization_warnings`.
