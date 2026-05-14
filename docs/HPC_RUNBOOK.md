# HPC Runbook

This runbook records the final operational state of the Qwen/Gemma study. The main study is complete. The HPC should now be used only for archival verification, reruns, or optional future robustness checks.

## Final status

Completed on HPC:

- Primary overall pairwise evaluation: 10,000 records
- Explicit marker-level evaluation: 10,000 records / 90,000 marker decisions
- Qwen/Gemma judges only
- Temperature condition: primary evaluation temperature `t000`
- Judge-facing view: `canonical_masked_scrubbed_v1`

Final output roots:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/
```

Final results root:

```text
results/primary_t000_scrubbed_v1/
```

## Environment setup

Always use explicit paths. Do not rely on `conda activate` on the HPC.

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/3202029/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/3202029/trailtraining
export PY=/home/3202029/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

Hugging Face cache variables:

```bash
export HF_HOME=/mnt/beegfsstudents/home/3202029/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_XET=1
```

For gated Gemma models, set a Hugging Face token in the shell before caching:

```bash
export HF_TOKEN="hf_..."
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Never paste tokens into logs, scripts committed to the repo, or chat.

## Final judge model set

| Judge name | Model |
|---|---|
| `qwen_7b_judge` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen_14b_judge` | `Qwen/Qwen2.5-14B-Instruct-AWQ` |
| `gemma_4b_judge` | `google/gemma-3-4b-it` |
| `gemma_12b_judge` | `google/gemma-3-12b-it` |

No Llama judge is included in the completed Qwen/Gemma study.

## Primary pairwise run status

The primary pairwise dataset is complete:

```text
250 pairs x 4 judges x 5 runs x 2 AB/BA positions = 10,000 records
```

Final combined file:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Verification:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"
wc -l "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
jq -r '.record_id' "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -d | wc -l
jq -r '.judge' "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
jq -r '.order' "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
jq -r '.run' "$D"/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
```

Expected:

```text
10000 rows
0 duplicate record IDs
2500 per judge
5000 AB / 5000 BA
2000 each run index 0-4
```

## Marker-level run status

The explicit marker-level dataset is complete:

```text
10,000 marker records x 9 markers = 90,000 marker decisions
```

Final combined marker file:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Verification:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
wc -l "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
jq -r '.record_id' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -d | wc -l
jq -r '.judge' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
jq -r '.order' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
jq -r '.run' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
jq -r '.markers | length' "$D"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | sort | uniq -c
```

Expected:

```text
10000 marker records
0 duplicate record IDs
2500 per judge
5000 AB / 5000 BA
2000 each run index 0-4
10000 rows with 9 markers
```

## Model-specific operational notes

### Qwen 14B AWQ

Qwen 14B required the vLLM quantization argument to be `awq`, not `awq_int4`.

It also required capping structured output tokens below the 4096 context length:

```bash
TRAILTRAINING_STRUCTURED_MAX_TOKENS=768
MARKER_MAX_TOKENS=768
```

Qwen 14B marker evaluation took multiple resumable chunks. Final output is complete and deduplicated.

### Qwen 7B

Qwen 7B returned a small number of blank `preferred` fields during marker evaluation. The marker runner was patched to repair blank `preferred` values deterministically from `plan_a_score` and `plan_b_score`, or to `tie` if scores were equal or unavailable. Repaired fields are flagged with `normalization_warnings`.

Final repaired count across the full marker dataset: 31 marker decisions out of 90,000.

### Gemma 12B and Gemma 4B

Gemma models produced a small number of malformed JSON responses during first-pass marker runs. The marker runner was patched to request compact JSON using `response_format={"type":"json_object"}` and empty rationale strings for retry runs. All failed records were recovered. Final failure files are empty.

## Sync from HPC to laptop

Run from the laptop:

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
  --exclude='*.md' \
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

The `--exclude='*.md'` rule is intentional if local documentation is being edited separately.

## Cache cleanup

Only delete a model cache after the corresponding artifacts have been synced and locally verified.

Example:

```bash
export HF_HOME=/mnt/beegfsstudents/home/3202029/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
rm -rf "${HF_HUB_CACHE}/models--Qwen--Qwen2.5-14B-Instruct-AWQ"
rm -rf "${HF_HUB_CACHE}/.locks/models--Qwen--Qwen2.5-14B-Instruct-AWQ"
lquota
```

Never delete cached models during an active job or before local verification.

## Known errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pydantic'` | Wrong Python interpreter | Use `/home/3202029/.conda/envs/judge-bias/bin/python` |
| `ModuleNotFoundError: No module named 'trailtraining'` | Missing `PYTHONPATH` | Export `PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"` |
| Broken `conda activate` | HPC shell setup | Use direct env Python path |
| `Model not found in cache` | Cache absent or deleted | Cache model or set HF cache vars correctly |
| Gemma gated repo error | HF token/license missing | Export `HF_TOKEN` and accept license on Hugging Face account |
| Qwen 14B `awq_int4` mismatch | Wrong vLLM quantization arg | Use `--quantization awq` |
| Qwen 14B context error | Output cap too high | Set `TRAILTRAINING_STRUCTURED_MAX_TOKENS=768` |
| Marker blank `preferred` | Model omitted categorical field | Repair from marker scores and flag normalization warning |
| Marker JSON decode failures | Malformed model JSON | Use compact JSON prompt and `response_format` |
| Proceeding from 30 matched pairs | Invalid old matching run | Use frozen 250-pair structural matched set only |

## Final recommendation

Do not run additional full HPC jobs for the current report unless performing a new scrubbed-v2 robustness study. The current primary, marker, and leakage-filtered sensitivity datasets are complete and auditable.
