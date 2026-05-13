# Results Summary

This document summarizes the current public results for the Qwen/Gemma `llm-judge-self-preference` study.

## Current result status

| Component | Status |
|---|---|
| Primary structural matching | Complete |
| Primary 10,000 pairwise judgments | Complete |
| Primary result analysis | Complete enough for reporting, further CIs/models recommended |
| Explicit marker-level Qwen 14B | Complete |
| Explicit marker-level Qwen 7B | Complete |
| Explicit marker-level Gemma 12B | Pending/missing from uploaded marker archive |
| Explicit marker-level Gemma 4B | Pending/missing from uploaded marker archive |

## Matching results

Frozen matched set:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/matched_pairs.json
```

Summary:

```text
matched pairs: 250
coverage_ok: true
coverage_ratio: 1.0
mean structural score gap: 0.3076
p95 structural score gap: 2.0
max structural score gap: 2.0
pairs by source family: gemma=116, qwen=134
pairs by athlete band: A1=60, A2=55, A3=74, A4=61
```

## Primary pairwise result

Output:

```text
artifacts/gen_src_t070_exp_t000/frozen_primary_v1/judgments_eval_t000_scrubbed_v1_staged/pairwise_all_judges_canonical_masked_scrubbed_v1_t000.jsonl
```

Integrity:

```text
rows: 10,000
unique record IDs: 10,000
duplicate IDs: 0
pairs: 250
judges: 2,500 each
orders: 5,000 AB / 5,000 BA
runs: 2,000 each for run 0-4
```

Overall:

| Winner role | Count | Rate |
|---|---:|---:|
| LLM | 3,653 | 36.53% |
| Programmatic | 6,347 | 63.47% |

By judge:

| Judge | LLM wins | Programmatic wins | LLM win rate |
|---|---:|---:|---:|
| `gemma_12b_judge` | 740 | 1,760 | 29.60% |
| `gemma_4b_judge` | 836 | 1,664 | 33.44% |
| `qwen_14b_judge` | 1,035 | 1,465 | 41.40% |
| `qwen_7b_judge` | 1,042 | 1,458 | 41.68% |

By judge family:

| Judge family | LLM wins | N | LLM win rate |
|---|---:|---:|---:|
| Gemma | 1,576 | 5,000 | 31.52% |
| Qwen | 2,077 | 5,000 | 41.54% |

By source family:

| Source family | LLM wins | N | LLM win rate |
|---|---:|---:|---:|
| Gemma-source | 1,793 | 4,640 | 38.64% |
| Qwen-source | 1,860 | 5,360 | 34.70% |

Self-family:

| Self-family match | LLM wins | N | LLM win rate |
|---|---:|---:|---:|
| False | 1,817 | 5,000 | 36.34% |
| True | 1,836 | 5,000 | 36.72% |

Current interpretation: row-level self-family difference is very small and does not support a strong self-family preference claim.

## Position/order effects

| Order | LLM position | LLM wins | N | LLM win rate |
|---|---|---:|---:|---:|
| AB | Plan A | 2,877 | 5,000 | 57.54% |
| BA | Plan B | 776 | 5,000 | 15.52% |

This is a major position effect. Primary conclusions should not rely only on row-level win rates.

## AB/BA pair-run consistency

Pair-run unit:

```text
pair_id × judge × run
```

Total units:

```text
250 pairs × 4 judges × 5 runs = 5,000
```

| Category | Count | Rate |
|---|---:|---:|
| `source_consistent_programmatic` | 1,973 | 39.46% |
| `source_consistent_llm` | 626 | 12.52% |
| `position_consistent_plan_a` | 2,251 | 45.02% |
| `position_consistent_plan_b` | 150 | 3.00% |

Interpretation:

- Programmatic plans are source-consistently preferred more often than LLM-source plans.
- A very large share of pair-run units are position-consistent, especially Plan A.
- Position bias is an important substantive finding and a threat to naive row-level interpretation.

## Explicit marker-level results: current Qwen-only state

Current marker files present:

```text
marker_qwen_14b_judge_canonical_masked_scrubbed_v1_t000.jsonl
marker_qwen_7b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

Missing from the uploaded marker archive:

```text
marker_gemma_12b_judge_canonical_masked_scrubbed_v1_t000.jsonl
marker_gemma_4b_judge_canonical_masked_scrubbed_v1_t000.jsonl
```

Qwen-only marker integrity:

```text
records: 5,000
marker decisions: 45,000
unique record IDs: 5,000
duplicate record IDs: 0
orders: 2,500 AB / 2,500 BA
runs: 1,000 each for run 0-4
marker fields per row: 9
```

Qwen 14B marker file:

```text
rows: 2,500
normalization repairs: 0 marker fields
tie marker fields: 5,734
```

Qwen 7B marker file:

```text
rows: 2,500
normalization repairs: 31 marker fields
tie marker fields: 5,500
```

The 31 Qwen 7B repairs occurred when the model returned scores but left `preferred` blank. The runner repaired `preferred` deterministically from the scores and recorded `normalization_warnings`.

## Qwen-only marker rates

| Marker | N | LLM wins | Programmatic wins | Ties | LLM win rate excluding ties |
|---|---:|---:|---:|---:|---:|
| `clarity_actionability` | 5,000 | 769 | 1,090 | 3,141 | 41.37% |
| `endurance_development` | 5,000 | 1,456 | 3,189 | 355 | 31.35% |
| `explanation_quality` | 5,000 | 1,325 | 2,487 | 1,188 | 34.76% |
| `load_progression` | 5,000 | 1,599 | 2,756 | 645 | 36.72% |
| `plan_coherence` | 5,000 | 1,183 | 2,258 | 1,559 | 34.38% |
| `quality_session_design` | 5,000 | 1,527 | 2,530 | 943 | 37.64% |
| `readiness_alignment` | 5,000 | 1,437 | 2,337 | 1,226 | 38.08% |
| `recovery_safety` | 5,000 | 1,282 | 2,169 | 1,549 | 37.15% |
| `training_specificity` | 5,000 | 1,930 | 2,442 | 628 | 44.14% |

Current Qwen-only marker conclusion:

> Qwen judges favor programmatic plans over LLM-source plans on every explicit marker, including explanation quality. Qwen 7B is more LLM-favorable than Qwen 14B, but neither Qwen judge gives LLM-source plans a majority on any marker after excluding ties.

## Qwen-only marker self-family result

Self-family here means Qwen judge × Qwen-source plan. Current Qwen-only marker results do not show robust self-family preference. Some markers are slightly higher under self-family, but most are equal or lower.

Careful statement:

> Among Qwen judges only, explicit marker ratings do not show robust self-family preference. Do not generalize to Gemma self-family effects until Gemma marker files are complete.

## Recommended next analyses

Before writing final claims, run:

1. Clustered bootstrap confidence intervals by matched pair.
2. Logistic or mixed-effects models with fixed effects for order, source family, judge family, self-family match, and run.
3. Pair-run consistency analyses for primary and marker-level outcomes.
4. Marker-level models after Gemma marker files are complete.
5. Sensitivity excluding pairs with high structural feature gaps.
6. Sensitivity excluding stylistically imbalanced pairs.

## Main current conclusion

The primary experiment does not support broad LLM-over-programmatic preference. Programmatic plans are preferred more often overall and source-consistently. Large position effects require cautious interpretation. Current Qwen-only explicit marker ratings also favor programmatic plans on every marker and do not show robust Qwen self-family preference. Full marker-level conclusions require completing the Gemma marker passes.
