# Reproducibility Checklist

Use this checklist before archiving or sharing the completed study.

## Required artifact checks

### Primary pairwise file

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged"
P="$D/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl"
wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
```

Expected:

```text
10000 rows
0 duplicate record IDs
2500 per judge
5000 AB / 5000 BA
2000 per run index
```

### Marker file

```bash
D="artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged"
P="$D/marker_all_judges_canonical_masked_scrubbed_v1_t000.jsonl"
wc -l "$P"
jq -r '.record_id' "$P" | sort | uniq -d | wc -l
jq -r '.judge' "$P" | sort | uniq -c
jq -r '.order' "$P" | sort | uniq -c
jq -r '.run' "$P" | sort | uniq -c
jq -r '.markers | length' "$P" | sort | uniq -c
```

Expected:

```text
10000 rows
0 duplicate record IDs
2500 per judge
5000 AB / 5000 BA
2000 per run index
10000 rows with 9 markers
```

### Leakage-filtered results

Check:

```text
results/primary_t000_scrubbed_v1/leakage_filtered/leakage_filtered_integrity.json
```

Expected:

- 250 total pairs
- 103 clean pairs
- 147 flagged pairs
- 4,120 clean primary records
- 37,080 clean marker decisions

## Local analysis outputs

Primary results:

```text
results/primary_t000_scrubbed_v1/analysis/
```

Marker results:

```text
results/primary_t000_scrubbed_v1/marker_analysis/
```

Leakage-filtered sensitivity results:

```text
results/primary_t000_scrubbed_v1/leakage_filtered/
```

## Files to include in final archive

Include:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matched_pairs.json
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matching_audit.json
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/scrub_audit.json
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/*.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/marker_eval_t000_scrubbed_v1_staged/*.jsonl
results/primary_t000_scrubbed_v1/
README.md
HPC_RUNBOOK.md
PREREGISTRATION.md
INCIDENTS.md
TEMPERATURE_SWEEPS.md
docs/
tools/run_marker_manifest_pairwise.py
tools/analyze_marker_pairwise.py
tools/build_leakage_filtered_results.py
```

Exclude:

```text
.git/
.venv/
venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.DS_Store
__MACOSX/
.env
*.token
hf_cache/
hub/
transformers/
models/
*.safetensors
*.bin
*.pt
*.pth
```

## Suggested final zip command

```bash
zip -r final_qwen_gemma_study_results.zip \
  artifacts/gen_src_t070_exp_t000/frozen_primary_v1 \
  results/primary_t000_scrubbed_v1 \
  README.md HPC_RUNBOOK.md PREREGISTRATION.md INCIDENTS.md TEMPERATURE_SWEEPS.md docs tools \
  -x "*/.DS_Store" \
  -x "__MACOSX/*" \
  -x "*.safetensors" \
  -x "*.bin" \
  -x "*.pt" \
  -x "*.pth" \
  -x ".venv/*" \
  -x "venv/*" \
  -x ".git/*" \
  -x "hf_cache/*" \
  -x "hub/*" \
  -x "transformers/*" \
  -x "models/*"
```
