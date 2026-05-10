# Explicit Marker-Level Rerun

The primary 10,000 overall pairwise judgments are complete. The current task is to run a secondary explicit marker-level evaluation over the same frozen 10,000 AB/BA records.

## Why this rerun is needed

The completed primary judgment files contain:

```text
preferred
winner_role
llm_won
reasoning
plan_a_advantages
plan_b_advantages
metadata
```

They do not contain explicit marker/rubric preferences or scores.

A previous marker-like analysis was exploratory and keyword-coded the free-text rationales. It is useful but not sufficient for strong marker-level conclusions.

The explicit marker rerun asks each judge to evaluate each pair on named markers directly.

## Marker schema

Markers:

```text
plan_coherence
training_specificity
load_progression
recovery_safety
quality_session_design
endurance_development
readiness_alignment
clarity_actionability
explanation_quality
```

Each marker record should include:

```json
{
  "preferred": "plan_a | plan_b | tie",
  "plan_a_score": 1,
  "plan_b_score": 1,
  "confidence": 0.0,
  "rationale": "short explanation"
}
```

Winner-role mapping is done outside the judge prompt using existing metadata:

```text
Plan A / Plan B -> llm / programmatic
```

## Expected full marker dataset

```text
10,000 marker judgment records
x 9 markers per record
= 90,000 explicit marker decisions
```

Expected output files:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

## Current status

As of the latest session:

- `tools/run_marker_manifest_pairwise.py` was created.
- `tools/analyze_marker_pairwise.py` was created.
- `slurm/run_marker_manifest_judge_hpc.sh` was created by copying the working primary runner.
- Qwen 14B marker smoke succeeded.
- Full Qwen 14B marker run produced 701 valid rows with 0 failures.
- Those 701 rows were written to a nested incorrect directory because `run_marker_manifest_judge_hpc.sh` passed an old pairwise output-file path as `--output-dir`.

Wrong current path:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

Correct target path:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

## Immediate next action

### 1. Patch `slurm/run_marker_manifest_judge_hpc.sh`

Inspect the runner call:

```bash
sed -n '95,125p' slurm/run_marker_manifest_judge_hpc.sh
```

Patch output-dir handling:

```bash
python - <<'PY'
from pathlib import Path

p = Path("slurm/run_marker_manifest_judge_hpc.sh")
s = p.read_text()

repls = {
    '--output-dir "$OUT_FILE"': '--output-dir "$OUTPUT_DIR"',
    '--output-dir "${OUT_FILE}"': '--output-dir "${OUTPUT_DIR}"',
    '--output-dir "$OUTPUT_PATH"': '--output-dir "$OUTPUT_DIR"',
    '--output-dir "${OUTPUT_PATH}"': '--output-dir "${OUTPUT_DIR}"',
    '--output-dir "$JUDGE_OUT"': '--output-dir "$OUTPUT_DIR"',
    '--output-dir "${JUDGE_OUT}"': '--output-dir "${OUTPUT_DIR}"',
}

changed = False
for old, new in repls.items():
    if old in s:
        s = s.replace(old, new)
        changed = True

p.write_text(s)
print("changed:", changed)
PY

sed -n '95,125p' slurm/run_marker_manifest_judge_hpc.sh
```

The runner call should pass:

```bash
--output-dir "$OUTPUT_DIR"
```

not a `pairwise_*.jsonl` file path.

### 2. Move the 701 good rows to the correct path

```bash
MARKER_OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"

BAD_DIR="$MARKER_OUT/pairwise_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl"
BAD_P="$BAD_DIR/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl"
GOOD_P="$MARKER_OUT/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl"

wc -l "$BAD_P"
mv "$BAD_P" "$GOOD_P"

if [ -f "$BAD_DIR/failures_marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl" ]; then
  mv "$BAD_DIR/failures_marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl" \
     "$MARKER_OUT/failures_marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl"
fi

rmdir "$BAD_DIR" 2>/dev/null || true

wc -l "$GOOD_P"
```

Expected:

```text
701
```

### 3. Resume Qwen 14B marker run

```bash
MARKER_OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
PRIMARY_MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"

VLLM_PORT=8801 \
JUDGE_NAME=qwen_14b_judge \
MANIFEST="$PRIMARY_MANIFEST" \
OUTPUT_DIR="$MARKER_OUT" \
JUDGE_TEMPERATURE=0.0 \
TRAILTRAINING_STRUCTURED_MAX_TOKENS=768 \
MARKER_MAX_TOKENS=768 \
CLEANUP_MODEL_CACHE=0 \
sbatch slurm/run_marker_manifest_judge_hpc.sh
```

Monitor:

```bash
watch -n 60 'date; wc -l artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl 2>/dev/null; squeue -u 3202029'
```

The marker runner skips existing `record_id`s, so it should resume from 701 and finish the remaining 1,799 rows.

### 4. Verify Qwen 14B marker file

```bash
P="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl"

wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
jq -r '.markers | length' "$P" | sort | uniq -c
```

Expected:

```text
2,500 rows
0 duplicate record_ids
2,500 qwen_14b_judge
1,250 AB / 1,250 BA
500 per run 0--4
2,500 rows with 9 markers
```

Hash:

```bash
sha256sum "$P" > "${P%.jsonl}.sha256"
```

## Continue with remaining judges

After Qwen 14B reaches 2,500 rows, run the same marker pass for:

```text
qwen_7b_judge
gemma_12b_judge
gemma_4b_judge
```

Use the same `MARKER_OUT` and `PRIMARY_MANIFEST` variables. Change only:

```text
JUDGE_NAME
VLLM_PORT
cached model
```

Because of quota, stage models one at a time.

## Combine marker outputs

After all four marker files have 2,500 rows:

```bash
MARKER_OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"

cat "$MARKER_OUT"/marker_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
    "$MARKER_OUT"/marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
    "$MARKER_OUT"/marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
    "$MARKER_OUT"/marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl \
  > "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl

wc -l "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
cat "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.record_id' | sort | uniq -d | wc -l
cat "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.judge' | sort | uniq -c
cat "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl | jq -r '.markers | length' | sort | uniq -c

sha256sum "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl \
  > "$MARKER_OUT"/marker_all_judges_canonical_masked_scrubbed_v1_t000.sha256
```

Expected:

```text
10,000 marker records
0 duplicate record_ids
2,500 per judge
10,000 rows with 9 markers
```

## Run marker analysis

```bash
mkdir -p results/primary_t000_scrubbed_v1/marker_analysis

"$PY" tools/analyze_marker_pairwise.py \
  --marker-judgments artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl \
  --output-dir results/primary_t000_scrubbed_v1/marker_analysis
```

Important output files:

```text
marker_integrity_summary.json
marker_overall_summary.csv
marker_by_judge_summary.csv
marker_by_judge_family_summary.csv
marker_by_source_family_summary.csv
marker_by_judge_family_source_family_summary.csv
marker_self_family_summary.csv
marker_order_bias_summary.csv
marker_pair_run_consistency_summary.csv
marker_pair_run_consistency_by_judge.csv
```

## Reporting rules

The explicit marker-level analysis should be reported as a secondary analysis, not as a replacement for the completed primary overall pairwise analysis.

Always report both:

1. row-level marker win rates, and
2. AB/BA pair-run marker consistency.

Because the primary study found large position bias, marker-level row rates alone are not sufficient.
