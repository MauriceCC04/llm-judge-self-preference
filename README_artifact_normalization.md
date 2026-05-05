# Artifact presentation normalization fix

This bundle adds a source-symmetric normalization gate for saved training-plan artifacts. It is intended to run after generation/top-up and before deterministic scoring, matching, and judging.

## Files

- `generate/artifact_normalization.py` — canonical title/prose repair plus post-normalization presentation audit helpers.
- `tools/normalize_plan_artifacts.py` — in-place corpus normalizer, cleanup, sidecar metadata writer, and manifest generator.
- `tools/audit_plan_corpus.py` — strict downstream corpus audit.
- `tests/test_artifact_normalization.py` and `tests/test_corpus_audit.py` — regression tests.

## Apply

```bash
unzip artifact_normalization_fix_files.zip -d artifact_normalization_fix_files
python artifact_normalization_fix_files/apply_artifact_normalization_fix.py --repo-root .
pytest tests/test_artifact_normalization.py tests/test_corpus_audit.py -v
```

The apply script makes `.bak` backups for existing target files unless `--force` is used.

## Normalize completed source-family corpora

```bash
python tools/normalize_plan_artifacts.py \
  --plans artifacts/gen_src_t070_exp_t000/full_qwen/plans \
  --audit-output artifacts/gen_src_t070_exp_t000/full_qwen/normalization_audit.json \
  --fail-on-errors

python tools/normalize_plan_artifacts.py \
  --plans artifacts/gen_src_t070_exp_t000/full_gemma3/plans \
  --audit-output artifacts/gen_src_t070_exp_t000/full_gemma3/normalization_audit.json \
  --fail-on-errors
```

For a dry run, add `--dry-run`. By default the tool moves `raw_failures/` and `failed_plans.jsonl` out of `plans/`, removes macOS/cache junk, and writes `<corpus>/manifest_sha256.txt`.

## Audit normalized corpora

```bash
python tools/audit_plan_corpus.py \
  --plans artifacts/gen_src_t070_exp_t000/full_qwen/plans \
  --expected-plan-count 192 \
  --expected-sidecar-count 192 \
  --expected-cell-count 32 \
  --expected-per-cell 6 \
  --require-manifest \
  --strict \
  --output artifacts/gen_src_t070_exp_t000/full_qwen/corpus_audit_normalized.json

python tools/audit_plan_corpus.py \
  --plans artifacts/gen_src_t070_exp_t000/full_gemma3/plans \
  --expected-plan-count 192 \
  --expected-sidecar-count 192 \
  --expected-cell-count 32 \
  --expected-per-cell 6 \
  --require-manifest \
  --strict \
  --output artifacts/gen_src_t070_exp_t000/full_gemma3/corpus_audit_normalized.json
```

Run the same commands for the programmatic corpus with its expected plan and per-cell counts. The normalizer is source-symmetric and should be applied to LLM and programmatic artifacts.

## Provenance behavior

Normalization metadata is stored only in `runtime_metadata.artifact_normalization` inside the provenance sidecar. The judge-visible plan JSON is not annotated with source-family labels or normalizer metadata.

The normalizer preserves the score-relevant projection of each plan: day dates, `session_type`, `is_rest_day`, `is_hard_day`, `duration_minutes`, intensity, weekly totals, fixture metadata, and other non-day-prose fields are not changed. Irreparable structural contradictions are flagged/quarantined instead of repaired by modifying structure.
