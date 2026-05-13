# Incidents and Reproducibility Notes

This document records issues encountered during the Qwen/Gemma study and the fixes that should remain visible for reproducibility.

## 1. Initial matching failure from inappropriate score

### Symptom

The original matcher produced only about 30 matched pairs from a 1,024-plan matching pool.

Observed state:

```text
Pairs yielded: 30
Target: 256
Coverage ratio: 0.117
Coverage OK: False
```

Relaxing score-bin and tolerance settings did not solve the problem.

### Root cause

The deterministic score used by the matcher was not appropriate as the primary matching score. LLM plans clustered near 97-100, while most programmatic plans scored around 30. The score appeared to include presentation/explanation quality rather than source-neutral structural quality.

### Fix

A source-neutral structural score was introduced for matching. It excludes prose quality, titles, rationale richness, citations, claim attributions, data notes, model names, and source labels.

Frozen matching after the fix:

```text
matched pairs: 250
target pairs: 250
coverage_ok: true
mean structural score gap: 0.3076
p95 structural score gap: 2.0
max structural score gap: 2.0
```

### Lesson

Do not use TrailTraining/global quality scores as structural matching scores unless it has been proven that presentation fields do not affect the score.

## 2. Broken shell snippets with placeholders

### Symptom

Commands using placeholder values were pasted directly:

```bash
export REPO_ROOT=/path/to/llm-judge-self-preference
export PY=/home/<USER>/.conda/envs/judge-bias/bin/python
```

This produced errors such as:

```text
-bash: USER: No such file or directory
cd: /path/to/llm-judge-self-preference: No such file or directory
```

### Fix

Use actual HPC paths or documented placeholders only in docs. For this cluster:

```bash
export REPO_ROOT=/mnt/beegfsstudents/home/<USER>/llm-judge-self-preference
export TRAILTRAINING_REPO=/mnt/beegfsstudents/home/<USER>/trailtraining
export PY=/home/<USER>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
```

### Lesson

Docs should make clear which commands contain placeholders and which are literal.

## 3. Wrong working directory during monitoring

### Symptom

Artifact files appeared missing while jobs were running:

```text
no output file yet
missing
find: artifacts/gen_src_t070_exp_t000/frozen_primary_v1: No such file or directory
```

### Root cause

Commands were run from `~` rather than the repository root.

### Fix

Always run:

```bash
cd /mnt/beegfsstudents/home/<USER>/llm-judge-self-preference
```

before using relative artifact paths.

### Lesson

Monitoring commands in runbooks must begin with `cd "$REPO_ROOT"` or use absolute paths.

## 4. Conda activation and Python environment errors

### Symptom

Observed errors included:

```text
CondaError: Run 'conda init' before 'conda activate'
ModuleNotFoundError: No module named 'pydantic'
ModuleNotFoundError: No module named 'trailtraining'
```

### Fix

Use the environment Python directly:

```bash
export PY=/home/<USER>/.conda/envs/judge-bias/bin/python
export PYTHONPATH="${REPO_ROOT}:${TRAILTRAINING_REPO}/src:${PYTHONPATH:-}"
```

### Lesson

Avoid relying on interactive shell conda activation in SLURM jobs.

## 5. vLLM argument incompatibility

### Symptom

A vLLM job failed with:

```text
api_server.py: error: unrecognized arguments: --disable-log-requests
```

### Fix

Remove `--disable-log-requests` for the installed vLLM version.

### Lesson

Do not assume vLLM CLI flags are stable across installed versions. Keep cluster-specific SLURM scripts in the repo.

## 6. Model cache and quota constraints

### Symptom

Jobs failed or could not launch because required models were not cached:

```text
Model not found in cache: Qwen/Qwen2.5-7B-Instruct
Model not found in cache: Qwen/Qwen2.5-14B-Instruct-AWQ
Model not found in cache: google/gemma-3-12b-it
```

The quota was limited, so all four judge models could not always be kept simultaneously.

### Fix

Cache one model at a time, run its jobs, sync outputs locally, verify hashes, then remove cache only if necessary.

### Lesson

Never delete a model cache until its outputs are complete, synced, and verified.

## 7. Qwen 14B primary pairwise resume

### Symptom

A Qwen 14B pairwise job appeared to run too long and was manually cancelled. It had actually produced a partial output file.

### Fix

The job was resumed and eventually produced a complete 2,500-row Qwen 14B pairwise output.

Final Qwen 14B pairwise integrity:

```text
rows: 2,500
duplicate record IDs: 0
orders: 1,250 AB / 1,250 BA
runs: 500 each for 0-4
```

### Lesson

Before cancelling, inspect the correct repository path and output file. Resumability by `record_id` is important.

## 8. Primary pairwise completion status typo

### Symptom

A shell heredoc intended to write `PRIMARY_COMPLETION_STATUS.md` was malformed and inserted garbled text.

### Fix

The primary output data itself is valid. Documentation should be rewritten cleanly rather than relying on the malformed status file.

### Lesson

Do not use ad hoc heredocs for archival status without checking the resulting file.

## 9. Empty Gemma 4B checksum file

### Symptom

The Gemma 4B primary pairwise `.sha256` file exists but is empty in the uploaded artifact bundle.

### Fix

Regenerate:

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"
sha256sum "$D/pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl" \
  > "$D/pairwise_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.sha256"
```

### Lesson

Verify each checksum file is non-empty and matches before archiving.

## 10. Top-level SHA256SUMS self-hash mismatch

### Symptom

The top-level `SHA256SUMS.txt` includes a hash for itself, causing a mismatch after the file is written.

### Fix

Regenerate `SHA256SUMS.txt` while excluding itself, or write the checksum file after hashing all other files.

### Lesson

Do not self-hash mutable manifest files.

## 11. Marker-level nested output path bug

### Symptom

Early marker jobs wrote to nested paths like:

```text
marker_eval_t000_scrubbed_v1_staged/
  pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl/
    marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

### Root cause

The marker SLURM script was copied from the pairwise runner and still passed pairwise-style output path variables. The marker runner expects `--output-dir`, not `--output` or `--failures`.

### Fix

Patch the marker SLURM script command block to use:

```bash
CMD=(
  "$PY" tools/run_marker_manifest_pairwise.py
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --judge-name "${JUDGE_NAME}"
  --temperature "${JUDGE_TEMPERATURE}"
  --max-tokens "${MARKER_MAX_TOKENS:-768}"
)
```

### Lesson

Runner interface changes must be reflected in SLURM wrappers. Monitor both expected and known-bad paths when debugging.

## 12. Qwen 14B marker-level resumed chunks

### Symptom

Qwen 14B marker evaluation required multiple chunks due to timeout/runtime.

Observed progress included:

```text
701 rows
1404 rows
2106 rows
2500 rows
```

### Fix

Resubmit the same marker job. The runner skips existing `record_id`s.

### Lesson

Marker jobs are resumable. Do not discard partial marker files.

## 13. Qwen 7B blank `preferred` marker fields

### Symptom

Qwen 7B returned valid marker scores but left `preferred` blank for a small number of marker fields:

```text
Bad preferred value for recovery_safety: ''
Bad preferred value for readiness_alignment: ''
```

### Fix

The marker runner was patched to repair blank/invalid `preferred` values deterministically:

- If `plan_a_score > plan_b_score`, set `preferred = plan_a`.
- If `plan_b_score > plan_a_score`, set `preferred = plan_b`.
- If scores are tied or unavailable, set `preferred = tie`.
- Add `normalization_warnings` to repaired marker fields.

Final Qwen 7B marker file contains 31 repaired marker fields and 0 bad preferred values.

### Lesson

Structured LLM outputs may be almost valid but incomplete. Repairs must be deterministic, conservative, and auditable.

## 14. Marker-level status is partial

### Symptom

The uploaded marker artifact contains complete Qwen marker files but no Gemma marker files.

Current marker state:

```text
qwen_14b_judge: complete
qwen_7b_judge: complete
gemma_12b_judge: missing/pending
gemma_4b_judge: missing/pending
```

### Fix

Complete Gemma 12B and Gemma 4B marker passes before reporting full four-judge marker-level results.

### Lesson

Label marker-level analyses as Qwen-only until all four judge files are present.

## 15. macOS metadata in archives

### Symptom

Archives include `.DS_Store` and `__MACOSX` files.

### Fix

Exclude from final public archives:

```bash
zip -r final_primary_artifacts.zip artifacts results \
  -x "*/.DS_Store" \
  -x "__MACOSX/*"
```

### Lesson

Clean archives before release.
