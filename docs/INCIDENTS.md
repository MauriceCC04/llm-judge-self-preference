# Incidents and Fixes

This document records major execution and methodology issues encountered during the Qwen/Gemma self-preference study.

## 1. Original matcher produced only about 30 pairs

### Symptom

The original matcher produced only about 30 matched pairs from the 1,024-plan matching pool, far below the required 250 pairs.

### Root cause

The old deterministic score did not provide adequate overlap between LLM-source and programmatic plans. Programmatic plans mostly scored around 30 while LLM-source plans mostly scored 85--100.

### Fix

Use source-neutral structural matching rather than the original presentation-contaminated quality score.

### Current status

Resolved.

Final matching:

```text
n_pairs: 250
coverage_ok: true
coverage_ratio: 1.0
mean_structural_score_gap: 0.3076
p95_structural_score_gap: 2.0
max_structural_score_gap: 2.0
```

## 2. `$PY` / heredoc shell paste failures

### Symptom

Commands failed with malformed fragments like:

```text
PYint("PASS...")
-bash: : command not found
```

### Root cause

Heredoc text was pasted incorrectly, gluing the closing marker to Python text.

### Fix

Use one-line Python commands or shell/JQ checks when the terminal is in a bad heredoc state.

### Current status

Resolved operationally.

## 3. `pytest` unavailable locally

### Symptom

```text
No module named pytest
```

### Root cause

Local virtual environment did not have `pytest` installed.

### Fix

Install/test environment as needed or use shell integrity checks for artifact validation.

## 4. Missing model cache under quota constraints

### Symptom

```text
Model not found in cache: Qwen/Qwen2.5-7B-Instruct
Model not found in cache: Qwen/Qwen2.5-14B-Instruct-AWQ
Model not found in cache: google/gemma-3-12b-it
```

### Root cause

50 GB quota made it impossible to keep all four judge models cached at once.

### Fix

Run staged judging one model at a time. After each model completes and outputs are hashed/synced, remove only that model cache if needed.

### Current status

Resolved for primary pairwise study. Still relevant for explicit marker-level rerun.

## 5. vLLM flag mismatch

### Symptom

```text
api_server.py: error: unrecognized arguments: --disable-log-requests
```

### Root cause

Installed vLLM version expected `--no-enable-log-requests`, not `--disable-log-requests`.

### Fix

Patch SLURM runner to use:

```text
--no-enable-log-requests
```

### Current status

Resolved.

## 6. Qwen 14B AWQ quantization mismatch

### Symptom

```text
Quantization method specified in the model config (awq) does not match the quantization argument (awq_int4)
```

### Root cause

Repository judge panel or wrapper used `awq_int4`, but vLLM expected `awq` for `Qwen/Qwen2.5-14B-Instruct-AWQ`.

### Fix

Normalize quantization argument:

```python
q = "" if j.quant == "fp16" else j.quant
if "${JUDGE_NAME}" == "qwen_14b_judge":
    q = "awq"
elif q == "awq_int4":
    q = "awq"
print(q)
```

### Current status

Resolved for primary pairwise run. Ensure the same fix is present in marker SLURM runner.

## 7. Qwen 14B structured output token issue

### Symptom

```text
model maximum context length is 4096 tokens; requested 4096 output tokens plus prompt
```

### Root cause

The TrailTraining client default requested 4,096 output tokens, leaving no room for the input prompt under `max_model_len=4096`.

### Fix

Set:

```bash
export TRAILTRAINING_STRUCTURED_MAX_TOKENS=768
```

For marker-level runner also set:

```bash
export MARKER_MAX_TOKENS=768
```

### Current status

Resolved for primary pairwise run. Required for Qwen 14B marker rerun.

## 8. Qwen 14B primary run was cancelled before completion

### Symptom

The Qwen 14B job was cancelled after 2,197 rows, initially thought to have no output because the shell was in the wrong directory.

### Fix

Resume the same output file. The runner skips existing `record_id`s. The final Qwen 14B pairwise file reached 2,500 rows with 0 duplicate record IDs.

### Current status

Resolved.

## 9. Empty Gemma 4B per-file checksum

### Symptom

The per-file `.sha256` for the Gemma 4B pairwise JSONL was empty in one uploaded artifact.

### Impact

The Gemma 4B JSONL was covered by `SHA256SUMS.txt`, so this was not a data integrity issue.

### Fix

Regenerate:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"
sha256sum "$D/pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl" \
  > "$D/pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.sha256"
```

## 10. macOS metadata in uploaded archives

### Symptom

Uploaded archives contained:

```text
__MACOSX/
.DS_Store
```

### Impact

Harmless, but noisy for reproducibility bundles.

### Fix

Exclude in final zip:

```bash
zip -r final_primary_artifacts.zip artifacts results \
  -x '*/.DS_Store' \
  -x '__MACOSX/*'
```

## 11. Marker rerun output path bug

### Symptom

Qwen 14B explicit marker run produced 701 valid rows but wrote them into:

```text
marker_eval_t000_scrubbed_v1_staged/pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

instead of:

```text
marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

### Root cause

The marker SLURM script was copied from the pairwise runner and still constructed an output-file path, then passed that path as `--output-dir` to the marker runner.

### Fix

Patch `slurm/run_marker_manifest_judge_hpc.sh` so the marker runner receives:

```bash
--output-dir "$OUTPUT_DIR"
```

Move the 701 rows to the correct path and resume.

### Current status

In progress. See `MARKER_LEVEL_RERUN.md`.
