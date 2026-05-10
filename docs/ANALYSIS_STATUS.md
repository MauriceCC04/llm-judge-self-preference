# Analysis Status

## Completed analyses

### Primary row-level analysis

Primary file:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Analysis directory:

```text
results/primary_t000_scrubbed_v1/analysis/
```

Completed tables include:

```text
row_level_summary.csv
per_judge_summary.csv
per_judge_family_summary.csv
per_source_family_summary.csv
self_family_summary.csv
order_bias_summary.csv
run_summary.csv
fixture_band_summary.csv
pair_run_consistency_rows.csv
pair_run_consistency_summary.csv
pair_run_consistency_by_judge.csv
```

Final integrity summary:

```text
results/primary_t000_scrubbed_v1/final_audit/final_integrity_summary.json
```

## Primary findings

Overall:

```text
LLM wins: 3,653 / 10,000 = 36.53%
Programmatic wins: 6,347 / 10,000 = 63.47%
```

By judge:

| Judge | LLM win rate |
|---|---:|
| `gemma_4b_judge` | 33.44% |
| `gemma_12b_judge` | 29.60% |
| `qwen_7b_judge` | 41.68% |
| `qwen_14b_judge` | 41.40% |

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

Self-family:

| Self-family match | LLM win rate |
|---|---:|
| False | 36.34% |
| True | 36.72% |

Interpretation: there is no strong row-level evidence of self-family preference.

## Position bias

Order effects are large:

| Order | LLM win rate |
|---|---:|
| AB | 57.54% |
| BA | 15.52% |

Pair-run consistency:

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_llm` | 626 | 12.52% |
| `source_consistent_programmatic` | 1,973 | 39.46% |
| `position_consistent_plan_a` | 2,251 | 45.02% |
| `position_consistent_plan_b` | 150 | 3.00% |

Interpretation: nearly half of pair-run units are position-consistent rather than source-consistent. This must be central in the paper.

## Exploratory marker-rationale analysis

A post-hoc keyword analysis of free-text rationales and advantage fields was performed. This was not an explicit marker-level rating pass.

Exploratory marker-like results suggested programmatic plans were more often credited with winning advantages across most training-relevant categories:

- endurance/aerobic base
- quality/speed/threshold
- recovery/rest/safety
- variety/balance/cross-training
- trail specificity
- load/structure-related concepts

Interpretation: exploratory rationale analysis does not support a strong marker-level self-family preference. It suggests judges often justified programmatic wins on substantive training-plan criteria.

## In-progress analyses

### Explicit marker-level rerun

A secondary marker-level pass is underway. It will create 9 explicit marker judgments per primary record.

Expected final marker dataset:

```text
10,000 marker records
90,000 marker decisions
```

Current status:

```text
Qwen 14B marker smoke: passed
Qwen 14B full marker run: 701 valid rows produced
Issue: 701 rows were written into nested wrong output directory
Next: patch SLURM output-dir, move 701 rows, resume Qwen 14B to 2,500 rows
Remaining: qwen_7b_judge, gemma_12b_judge, gemma_4b_judge
```

## Analyses still needed

- Complete explicit marker-level rerun.
- Analyze marker-level row rates and marker-level AB/BA pair-run consistency.
- Bootstrap confidence intervals, preferably clustered by pair ID.
- Model-based analysis with order, judge family, source family, self-family, and fixture/band controls.
- Score-gap sensitivity using structural score and feature gaps.
- Fixture/cell subgroup analysis.
- Report-ready tables and plots.

## Recommended interpretation language now

Safe current claim:

```text
In the completed primary pairwise evaluation, source-masked Qwen/Gemma judges preferred structurally matched programmatic plans more often than LLM-source plans. The result varies by judge model and is strongly affected by Plan-A position bias. Row-level self-family differences are tiny. Explicit marker-level conclusions require the in-progress secondary marker-level evaluation.
```

Avoid claiming:

```text
LLM judges show self-family preference at specific markers.
```

That has not yet been established.
