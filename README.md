# LLM Judge Self-Preference Study

This repository supports an empirical study of whether LLM judges systematically prefer LLM-generated trail-training plans over structurally matched programmatic plans.

## Current study state

The primary Qwen/Gemma study has completed its main pairwise-judgment run.

Primary design:

```text
250 structurally matched LLM-vs-programmatic pairs
x 4 local Qwen/Gemma judge models
x 5 repeated runs
x 2 AB/BA orders
= 10,000 pairwise judgment records
```

Primary source families:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-4b-it`

Primary judge models:

- `qwen_7b_judge`
- `qwen_14b_judge`
- `gemma_4b_judge`
- `gemma_12b_judge`

No Llama source or judge is part of the current frozen study.

## Primary artifact locations

Primary frozen input bundle:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/
```

Important files:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matched_pairs.json
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matching_audit.json
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matching_prefilter_audit.json
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/pairwise_eval_manifest.jsonl
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/eval_manifest/judge_inputs/
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/
```

Primary combined judgment file:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Primary analysis output:

```text
results/primary_t000_scrubbed_v1/analysis/
results/primary_t000_scrubbed_v1/final_audit/final_integrity_summary.json
```

## Completed data processing

Final corpora used for the primary study:

| Corpus | Path | Count |
|---|---|---:|
| Qwen LLM plans | `artifacts/gen_src_t070_exp_t000/full_qwen/plans` | 192 |
| Gemma LLM plans | `artifacts/gen_src_t070_exp_t000/full_gemma3/plans` | 192 |
| Programmatic plans | `artifacts/gen_src_t070_exp_t000/full_programmatic/plans` | 640 |
| Matching pool | `artifacts/gen_src_t070_exp_t000/matching_pool/plans` | 1,024 |

The original matcher failed because the original deterministic score was not structurally source-neutral. It produced only about 30 usable matches under the old score. The study was repaired by using a source-neutral structural matching score and a canonical masked judge view.

Final matching audit:

```text
matched pairs: 250
coverage_ok: true
coverage_ratio: 1.0
pairs_by_source_family:
  qwen: 134
  gemma: 116
pairs_by_athlete_band:
  A1: 60
  A2: 55
  A3: 74
  A4: 61
mean_structural_score_gap: 0.3076
p95_structural_score_gap: 2.0
max_structural_score_gap: 2.0
```

Final prefilter audit:

```text
n_input: 1024
n_kept: 1023
n_dropped: 1
drop_reasons:
  impossible_duration: 1
```

## Completed primary pairwise judgments

Final primary judgment integrity:

```text
total records: 10,000
duplicate record_ids: 0
judges:
  gemma_4b_judge: 2,500
  gemma_12b_judge: 2,500
  qwen_7b_judge: 2,500
  qwen_14b_judge: 2,500
orders:
  AB: 5,000
  BA: 5,000
runs:
  0: 2,000
  1: 2,000
  2: 2,000
  3: 2,000
  4: 2,000
```

Primary row-level result:

```text
LLM wins: 3,653 / 10,000 = 36.53%
Programmatic wins: 6,347 / 10,000 = 63.47%
```

By judge:

| Judge | LLM wins | Programmatic wins | LLM win rate |
|---|---:|---:|---:|
| `gemma_4b_judge` | 836 | 1,664 | 33.44% |
| `gemma_12b_judge` | 740 | 1,760 | 29.60% |
| `qwen_7b_judge` | 1,042 | 1,458 | 41.68% |
| `qwen_14b_judge` | 1,035 | 1,465 | 41.40% |

By judge family:

| Judge family | LLM win rate |
|---|---:|
| Gemma | 31.52% |
| Qwen | 41.54% |

By source family:

| Source family | LLM win rate |
|---|---:|
| Gemma-source | 38.64% |
| Qwen-source | 34.70% |

Self-family row-level result:

| Self-family match | LLM win rate |
|---|---:|
| False | 36.34% |
| True | 36.72% |

The row-level self-family difference is tiny and does not support a strong self-family preference claim.

## Position-bias finding

Order effects are large and must be treated as a central result/threat to validity.

| Order | LLM win rate |
|---|---:|
| AB, LLM as Plan A | 57.54% |
| BA, LLM as Plan B | 15.52% |

AB/BA pair-run consistency summary:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_llm` | 626 | 12.52% |
| `source_consistent_programmatic` | 1,973 | 39.46% |
| `position_consistent_plan_a` | 2,251 | 45.02% |
| `position_consistent_plan_b` | 150 | 3.00% |

This means the primary interpretation should not rely only on raw row-level win rates. Pair-run consistency is essential.

## Marker-level status

Two marker-related analyses must be distinguished.

### Completed: exploratory rationale-marker analysis

An exploratory keyword coding was run over the primary judges' free-text rationales and advantage lists. This analysis suggested that programmatic plans were more often credited with training-relevant winning advantages across most marker-like categories.

This is useful but not a formal marker-level result because the original primary outputs do not include explicit marker ratings.

### In progress: explicit marker-level rerun

A secondary explicit marker-level evaluation pass was started. It should use the frozen 10,000 manifest rows and produce 9 marker judgments per record:

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

Expected marker output:

```text
10,000 marker judgment records x 9 markers = 90,000 explicit marker decisions
```

Current marker rerun status from the latest working session:

- The Qwen 14B marker smoke test succeeded.
- The full Qwen 14B marker run produced 701 valid rows with 0 failures.
- Those rows were written into an unintended nested directory because the copied SLURM script passed an old pairwise output-file path as `--output-dir`.
- The next step is to patch `slurm/run_marker_manifest_judge_hpc.sh`, move the 701 valid rows to the correct marker output directory, and resume Qwen 14B to 2,500 rows.

See `MARKER_LEVEL_RERUN.md` for exact commands.

## What has not been completed yet

Not yet complete:

- Full explicit marker-level rerun for all 4 judges.
- Bootstrap confidence intervals.
- Mixed/logistic modeling with order, judge family, source family, and self-family effects.
- Final report populated with all final marker-level tables.
- Temperature sensitivity runs for evaluation temperature beyond the primary `t000` condition.
- Clean final archive excluding macOS metadata and historical/pilot artifacts.

## Minimal local integrity check

From repo root:

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
