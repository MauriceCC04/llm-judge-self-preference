# Temperature Sweeps and Sensitivity Runs

This document defines how generation and evaluation temperature conditions should be handled.

## Current primary temperature state

The primary frozen pairwise study uses:

```text
source generation temperature: src_t070
generation explanation temperature: exp_t000
primary judge evaluation temperature: t000
```

Primary artifact root:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

Primary pairwise output:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Primary marker output directory:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/
```

The uploaded explicit marker archive currently contains Qwen-only marker outputs at `t000`.

## Rule: do not mix temperature conditions

Each evaluation temperature must use a separate output directory and separate combined file.

Recommended naming:

```text
judgments_eval_t000_scrubbed_v1_staged/
judgments_eval_t020_scrubbed_v1_sensitivity/
judgments_eval_t070_scrubbed_v1_sensitivity/

marker_eval_t000_scrubbed_v1_staged/
marker_eval_t020_scrubbed_v1_sensitivity/
marker_eval_t070_scrubbed_v1_sensitivity/
```

Do not append sensitivity runs into the primary `t000` directories.

## Primary vs sensitivity

Primary result:

```text
eval_t000 on canonical_masked_scrubbed_v1 judge inputs
```

Sensitivity results may explore:

- `eval_t020`
- `eval_t070`
- other temperatures if justified

Sensitivity results should not replace the primary result. They should be reported separately as robustness checks.

## Required metadata

Every temperature-specific record must include or be recoverable from file path/metadata:

- judge model
- judge family
- source family
- pair ID
- run index
- AB/BA order
- evaluation temperature
- judge view / masking version
- marker schema version for marker runs

## Required checks before a temperature run

For pairwise:

```bash
MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"
wc -l "$MANIFEST"
jq -r '.judge' "$MANIFEST" | sort | uniq -c
jq -r '.order' "$MANIFEST" | sort | uniq -c
jq -r '.run' "$MANIFEST" | sort | uniq -c
```

Expected:

```text
10000 rows
2500 per judge
5000 AB / 5000 BA
2000 per run index
```

For marker runs, confirm the marker output directory is temperature-specific and empty or intentionally resumable.

## Pairwise sensitivity run example

```bash
PRIMARY_MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"
OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t020_scrubbed_v1_sensitivity"
mkdir -p "$OUT"

VLLM_PORT=8820 \
JUDGE_NAME=qwen_7b_judge \
MANIFEST="$PRIMARY_MANIFEST" \
OUTPUT_DIR="$OUT" \
JUDGE_TEMPERATURE=0.2 \
CLEANUP_MODEL_CACHE=0 \
sbatch slurm/run_manifest_judge_hpc.sh
```

## Marker sensitivity run example

```bash
PRIMARY_MANIFEST="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl"
OUT="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t020_scrubbed_v1_sensitivity"
mkdir -p "$OUT"

VLLM_PORT=8821 \
JUDGE_NAME=qwen_7b_judge \
MANIFEST="$PRIMARY_MANIFEST" \
OUTPUT_DIR="$OUT" \
JUDGE_TEMPERATURE=0.2 \
TRAILTRAINING_STRUCTURED_MAX_TOKENS=768 \
MARKER_MAX_TOKENS=768 \
CLEANUP_MODEL_CACHE=0 \
sbatch slurm/run_marker_manifest_judge_hpc.sh
```

## Reporting temperature checks

For every temperature condition, report:

- number of records
- duplicate record IDs
- per-judge row counts
- AB/BA balance
- run balance
- primary LLM win rates
- pair-run consistency
- marker-level rates if marker run
- comparison to primary `t000`

Do not pool temperatures in primary tables unless explicitly modeling temperature as a factor.

## Current status

As of the uploaded frozen artifact bundle:

- Primary pairwise `eval_t000` is complete.
- Explicit marker `eval_t000` is complete for Qwen judges only.
- No complete Gemma marker files are present in the uploaded marker archive.
- No complete nonzero-temperature sensitivity results are present in the uploaded artifact bundle.

Future temperature runs should be clearly labeled as sensitivity analyses.
